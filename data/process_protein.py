import pandas as pd
import numpy as np
import os
from feature_engine.selection import DropDuplicateFeatures, DropConstantFeatures

def load_and_transpose(file_path):
    df = pd.read_table(file_path)
    df = df.transpose()
    df = df.reset_index()
    df.columns = df.iloc[0]
    df = df[1:]
    df = df.rename(columns={df.columns[0]: 'sample'})
    return df

# Load and transpose LUAD and LUSC datasets
luad = load_and_transpose('/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/TCGA.LUAD.sampleMap_RPPA.tsv.gz')
lusc = load_and_transpose('/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/TCGA.LUSC.sampleMap_RPPA.gz')

# Add a column to keep track of the dataset type
luad['dataset'] = 'LUAD'
lusc['dataset'] = 'LUSC'

# Concatenate the datasets
combined = pd.concat([luad, lusc], axis=0, ignore_index=True)

# Drop columns with NaNs
combined = combined.dropna(axis=1)

# Drop constant features
sel1 = DropConstantFeatures(tol=0.9, variables=None, missing_values='raise')
sel1.fit(combined)
combined = sel1.transform(combined)

# Drop duplicate features
sel2 = DropDuplicateFeatures(variables=None, missing_values='raise')
sel2.fit(combined)
combined = sel2.transform(combined)

# Separate the sample and dataset columns for later use
samples = combined['sample']
dataset = combined['dataset']
combined = combined.drop(columns=['sample', 'dataset'])

# Insert sample and dataset columns back into the dataset
combined.insert(0, 'sample', samples)
combined.insert(1, 'dataset', dataset)

# Separate the combined dataset back into LUAD and LUSC
luad_data = combined[combined['dataset'] == 'LUAD'].drop(columns=['dataset']).copy()
lusc_data = combined[combined['dataset'] == 'LUSC'].drop(columns=['dataset']).copy()

def modify_sample_ids(df):
    df['sample'] = df['sample'].apply(lambda x: x[:-3])
    return df

# Apply the function to modify the sample IDs
luad_data = modify_sample_ids(luad_data)
lusc_data = modify_sample_ids(lusc_data)

def save_data_by_case_id(df, config_name, dataset_type):
    base_directory = f"/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/LUAD_LUSC_data/{dataset_type}/"
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)
    grouped = df.groupby('sample')
    for case_id, group in grouped:
        case_directory = os.path.join(base_directory, str(case_id))
        if not os.path.exists(case_directory):
            os.makedirs(case_directory)
        embedding = group.drop(columns=['sample']).values.astype(np.float32)
        file_path = os.path.join(case_directory, f"{config_name}.npy")
        np.save(file_path, embedding)
        print(f"Saved {config_name} embedding to {file_path}")

# Save the protein expression data as embeddings for both datasets
save_data_by_case_id(luad_data, 'protein_expression', 'LUAD')
save_data_by_case_id(lusc_data, 'protein_expression', 'LUSC')
