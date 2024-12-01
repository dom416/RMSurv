import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import VarianceThreshold
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
luad = load_and_transpose('/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/TCGA-LUAD.methylation450.tsv.gz')
lusc = load_and_transpose('/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/TCGA.LUSC.sampleMap_HumanMethylation450.gz')

# Add a column to keep track of the dataset type
luad['dataset'] = 'LUAD'
lusc['dataset'] = 'LUSC'

# Concatenate the datasets
combined = pd.concat([luad, lusc], axis=0, ignore_index=True)

# Drop columns with NaNs before using the DropConstantFeatures transformer
combined = combined.dropna(axis=1)

# Drop constant features
sel1 = DropConstantFeatures(tol=0.95, variables=None, missing_values='raise')
sel1.fit(combined)
combined = sel1.transform(combined)

# Drop duplicate features
sel2 = DropDuplicateFeatures(variables=None, missing_values='raise')
sel2.fit(combined)
combined = sel2.transform(combined)

# Drop columns with NaNs again to ensure no NaN values
combined = combined.dropna(axis=1)

# Separate the sample and dataset columns for later use
samples = combined['sample']
dataset = combined['dataset']
combined = combined.drop(columns=['sample', 'dataset'])

# Feature selection based on variance threshold
def variance_threshold_selector(data, threshold=0.9):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]

combined_hvdnameth = variance_threshold_selector(combined, 0.045)

# Insert sample and dataset columns back into the dataset
combined_hvdnameth.insert(0, 'sample', samples)
combined_hvdnameth.insert(1, 'dataset', dataset)

# Separate the combined dataset back into LUAD and LUSC
luad_hvdnameth = combined_hvdnameth[combined_hvdnameth['dataset'] == 'LUAD'].drop(columns=['dataset']).copy()
lusc_hvdnameth = combined_hvdnameth[combined_hvdnameth['dataset'] == 'LUSC'].drop(columns=['dataset']).copy()

def modify_sample_ids(df):
    df['sample'] = df['sample'].apply(lambda x: x[:-3])
    return df

# Apply the function to modify the sample IDs
luad_hvdnameth = modify_sample_ids(luad_hvdnameth)
lusc_hvdnameth = modify_sample_ids(lusc_hvdnameth)

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

# Save the DNA methylation data as embeddings for both datasets
save_data_by_case_id(luad_hvdnameth, 'dna_methylation', 'LUAD')
save_data_by_case_id(lusc_hvdnameth, 'dna_methylation', 'LUSC')
