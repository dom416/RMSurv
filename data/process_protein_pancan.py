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

# Load and transpose the Pancancer protein expression dataset
pancan = load_and_transpose('/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/pancan_protein.gz')

# Drop columns with NaNs
pancan = pancan.dropna(axis=1)

# Drop constant features
sel1 = DropConstantFeatures(tol=0.9, variables=None, missing_values='raise')
sel1.fit(pancan)
pancan = sel1.transform(pancan)

# Drop duplicate features
sel2 = DropDuplicateFeatures(variables=None, missing_values='raise')
sel2.fit(pancan)
pancan = sel2.transform(pancan)

# Separate the sample column for later use
samples = pancan['sample']
pancan = pancan.drop(columns=['sample'])

# Insert sample column back into the dataset
pancan.insert(0, 'sample', samples)

def modify_sample_ids(df):
    df['sample'] = df['sample'].apply(lambda x: x[:-3])
    return df

# Apply the function to modify the sample IDs
pancan_data = modify_sample_ids(pancan)

def save_data_by_case_id(df, config_name):
    base_directory = "/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/LUAD_LUSC_data/PAN/"
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

# Save the protein expression data as embeddings
save_data_by_case_id(pancan_data, 'protein_expression')
