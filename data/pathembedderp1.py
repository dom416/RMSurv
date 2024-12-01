import gc
import os
import json
import datasets
from dotenv import load_dotenv
import minds
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, HfFolder, login
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import warnings

from honeybee.loaders import (
    PDFreport,
    Scan,
    Slide,
    generate_summary_from_json,
    get_chunk_text,
)
from honeybee.models import REMEDIS, UNI, HuggingFaceEmbedder, TissueDetector

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
load_dotenv()

def manifest_to_df(manifest_path, modality):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Initialize an empty DataFrame for the modality
    modality_df = pd.DataFrame()

    # Process each patient in the manifest
    for patient in manifest:
        patient_id = patient["PatientID"]
        gdc_case_id = patient["gdc_case_id"]

        # Check if the current patient has the requested modality
        if modality in patient:
            # Convert the list of dictionaries into a DataFrame
            df = pd.DataFrame(patient[modality])
            # Add 'PatientID' and 'gdc_case_id' columns
            df["PatientID"] = patient_id
            df["gdc_case_id"] = gdc_case_id

            # Append the new data to the existing DataFrame for this modality
            modality_df = pd.concat([modality_df, df], ignore_index=True)

    # Check if the modality DataFrame is not empty before returning
    if not modality_df.empty:
        return modality_df
    else:
        return None

def process_group(group):
    common_fields = {}
    nested_objects = []
    for col in group.columns:
        unique_values = group[col].dropna().unique()
        if len(unique_values) == 1:
            # If only one unique value exists, it's a common field
            common_fields[col] = unique_values[0]

    # Create nested objects for fields that are not common
    for idx, row in group.iterrows():
        nested_object = {
            col: row[col]
            for col in group.columns
            if col not in common_fields and pd.notna(row[col])
        }
        if nested_object:  # Only add if the nested object is not empty
            nested_objects.append(nested_object)

    return common_fields, nested_objects

PROJECT = "TCGA-LUSC"

# --- THIS CAN BE IGNORED ---
DATA_DIR = f"/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/LUSCmindsdata"
MANIFEST_PATH = DATA_DIR + "/manifest.json"
MODALITY = "Pathology Report"
df = manifest_to_df(MANIFEST_PATH, MODALITY)

# --- CONFIGURATION ---
embedding_model = HuggingFaceEmbedder(model_name="UFNLP/gatortron-large")
pdf_report = PDFreport(chunk_size=512, chunk_overlap=10)

report_texts = []
df["report_text"] = None
df["embedding"] = None
df["embedding_shape"] = None

# Ensure the directory exists before writing the Parquet file
output_dir = f"./embeddings/{PROJECT}/"
os.makedirs(output_dir, exist_ok=True)

writer = None
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
    try:
        file_path = f"{DATA_DIR}/raw/{row['PatientID']}/{MODALITY}/{row['id']}/{row['file_name']}"
        report_text = pdf_report.load(file_path)
        report_texts.append(report_text)

        if len(report_text) > 0:
            embeddings = embedding_model.generate_embeddings(report_text)
            df.at[index, "embedding_shape"] = embeddings.shape
            embeddings = embeddings.reshape(-1)
            embeddings = embeddings.tobytes()
            df.at[index, "embedding"] = embeddings
        else:
            df.at[index, "embedding"] = None
        df.at[index, "report_text"] = report_text

    except Exception as e:
        print(f"Error: {e}")
        report_texts.append(None)
        df.at[index, "embedding"] = None

    # Ensure the writer is initialized with the correct schema
    if writer is None:
        schema = pa.Table.from_pandas(df.iloc[[index]]).schema
        writer = pq.ParquetWriter(
            f"./embeddings/{PROJECT}/pathology_report.parquet", schema
        )

    table = pa.Table.from_pandas(df.iloc[[index]], schema=schema)
    try:
        writer.write_table(table)
    except ValueError as e:
        print(f"Schema mismatch error: {e}")
        # Re-initialize writer with new schema if needed
        schema = table.schema
        writer = pq.ParquetWriter(
            f"./embeddings/{PROJECT}/pathology_report.parquet", schema
        )
        writer.write_table(table)

if writer is not None:
    writer.close()

gc.collect()
torch.cuda.empty_cache()

# dataset = datasets.load_dataset(
#     "parquet",
#     data_files=f"/mnt/d/TCGA-LUAD/parquet/{MODALITY}.parquet",
#     split="train",
# )
# dataset.save_to_disk(f"/mnt/d/TCGA-LUAD/hf_dataset/{MODALITY}")
