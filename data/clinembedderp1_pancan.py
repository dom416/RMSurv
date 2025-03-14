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

# List of projects to process
PROJECTS = [
    "TCGA-ACC", "TCGA-COAD", "TCGA-KICH", "TCGA-LIHC", "TCGA-PAAD", "TCGA-SKCM",
    "TCGA-UCEC", "TCGA-BLCA", "TCGA-DLBC", "TCGA-KIRC", "TCGA-LUAD", "TCGA-PCPG",
    "TCGA-STAD", "TCGA-UCS", "TCGA-BRCA", "TCGA-ESCA", "TCGA-KIRP", "TCGA-LUSC",
    "TCGA-PRAD", "TCGA-TGCT", "TCGA-UVM", "TCGA-CESC", "TCGA-GBM", "TCGA-LAML",
    "TCGA-MESO", "TCGA-READ", "TCGA-THCA", "TCGA-CHOL", "TCGA-HNSC", "TCGA-LGG",
    "TCGA-OV", "TCGA-SARC", "TCGA-THYM",
]

# Initialize the embedding model
embedding_model = HuggingFaceEmbedder(model_name="UFNLP/gatortron-medium")

# Initialize an empty list to collect data from all projects
df_all = []

for PROJECT in PROJECTS:
    print(f"Processing {PROJECT}")
    MODALITY = "Clinical Data"

    # Get the list of tables from the MINDS database
    tables = minds.get_tables()
    json_objects = {}

    # Fetch data for each table
    for table in tqdm(tables, desc=f"Getting data from tables for {PROJECT}"):
        query = f"SELECT * FROM minds.{table} WHERE project_id='{PROJECT}'"
        df = minds.query(query)
        for case_id, group in tqdm(df.groupby("case_submitter_id"), leave=False):
            if case_id not in json_objects:
                json_objects[case_id] = {}
            common_fields, nested_objects = process_group(group)
            json_objects[case_id].update(common_fields)
            json_objects[case_id][table] = nested_objects

    # Process each patient and generate embeddings
    df_project = []
    for case_id, patient_data in tqdm(json_objects.items(), desc=f"Processing patients for {PROJECT}"):
        summary = generate_summary_from_json(patient_data)
        if len(summary) > 0:
            summary_chunks = get_chunk_text(summary)
            chunk_embeddings = []
            for chunk in summary_chunks:
                chunk_embedding = embedding_model.generate_embeddings([chunk])
                chunk_embeddings.append(chunk_embedding)
            clinical_embedding = np.array(chunk_embeddings)
        else:
            clinical_embedding = torch.zeros(1, 1024)
        patient_data["text"] = summary
        patient_data["embedding_shape"] = clinical_embedding.shape
        clinical_embedding = clinical_embedding.reshape(-1)
        clinical_embedding = np.array(clinical_embedding, dtype=np.float32)
        clinical_embedding = clinical_embedding.tobytes()
        patient_data["embedding"] = clinical_embedding
        # Create a new dictionary for DataFrame conversion, excluding lists
        patient_data_for_df = {
            key: value
            for key, value in patient_data.items()
            if not isinstance(value, list)
        }
        patient_data_for_df["project_id"] = PROJECT  # Add project ID to data
        df_project.append(patient_data_for_df)

    # Append data from this project to the overall list
    df_all.extend(df_project)

# After processing all projects, convert the list to a DataFrame
clinical_df = pd.DataFrame(df_all)

# Save the combined DataFrame to a single parquet file
PARQUET = "./embeddings/PANCAN/clinical_data.parquet"
os.makedirs(os.path.dirname(PARQUET), exist_ok=True)
clinical_df.to_parquet(PARQUET, index=False)
