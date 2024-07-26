import pandas as pd
import numpy as np
import os
from feature_engine.selection import DropDuplicateFeatures, DropConstantFeatures, SmartCorrelatedSelection

def load_and_transpose(file_path):
    df = pd.read_table(file_path)
    df = df.transpose()
    df = df.reset_index()
    df.columns = df.iloc[0]
    df = df[1:]
    df = df.rename(columns={df.columns[0]: 'sample'})
    return df

# Load and transpose LUAD and LUSC datasets
luad = load_and_transpose('/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/TCGA-LUAD.mirna.tsv.gz')
lusc = load_and_transpose('/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/TCGA-LUSC.mirna.tsv.gz')

# Add a column to keep track of the dataset type
luad['dataset'] = 'LUAD'
lusc['dataset'] = 'LUSC'

# Concatenate the datasets
combined = pd.concat([luad, lusc], axis=0, ignore_index=True)

# Drop constant features
sel1 = DropConstantFeatures(tol=0.9, variables=None, missing_values='raise')
sel1.fit(combined)
combined = sel1.transform(combined)

# Separate the sample and dataset columns for later use
samples = combined['sample']
dataset = combined['dataset']
combined = combined.drop(columns=['sample', 'dataset'])

# Convert the variables to numerical variables
combined = combined.astype(float)

# Feature selection based on correlation
sel2 = SmartCorrelatedSelection(
    variables=None,
    method="pearson",
    threshold=0.9,
    missing_values="raise",
    selection_method="variance",
    estimator=None,
)

sel2.fit(combined)
combined = sel2.transform(combined)

# Insert sample and dataset columns back into the dataset
combined.insert(0, 'sample', samples)
combined.insert(1, 'dataset', dataset)

# Separate the combined dataset back into LUAD and LUSC
luad_mirna = combined[combined['dataset'] == 'LUAD'].drop(columns=['dataset']).copy()
lusc_mirna = combined[combined['dataset'] == 'LUSC'].drop(columns=['dataset']).copy()

def modify_sample_ids(df):
    df['sample'] = df['sample'].apply(lambda x: x[:-4])
    return df

# Apply the function to modify the sample IDs
luad_mirna = modify_sample_ids(luad_mirna)
lusc_mirna = modify_sample_ids(lusc_mirna)

def save_data_by_case_id(df, config_name, dataset_type):
    base_directory = f"/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/LUAD_LUSC_data/{dataset_type}/"
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)
    for idx, row in df.iterrows():
        case_id = row['sample']
        case_directory = os.path.join(base_directory, case_id)
        if not os.path.exists(case_directory):
            os.makedirs(case_directory)
        cnv_vector = row.drop('sample').values.astype(np.float32)
        file_path = os.path.join(case_directory, f"{config_name}.npy")
        np.save(file_path, cnv_vector)
        print(f"Saved {config_name} embedding to {file_path}")

# Save the miRNA data as embeddings for both datasets
save_data_by_case_id(luad_mirna, 'mirna', 'LUAD')
save_data_by_case_id(lusc_mirna, 'mirna', 'LUSC')
