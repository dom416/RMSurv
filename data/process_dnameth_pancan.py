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

# Load and transpose the Pancancer DNA methylation dataset
pancan = load_and_transpose('/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/pancan_dnameth.gz')

# Drop columns with NaNs before using the DropConstantFeatures transformer
pancan = pancan.dropna(axis=1)

# Drop constant features
sel1 = DropConstantFeatures(tol=0.95, variables=None, missing_values='raise')
sel1.fit(pancan)
pancan = sel1.transform(pancan)

# Drop duplicate features
sel2 = DropDuplicateFeatures(variables=None, missing_values='raise')
sel2.fit(pancan)
pancan = sel2.transform(pancan)

# Drop columns with NaNs again to ensure no NaN values
pancan = pancan.dropna(axis=1)

# Separate the sample column for later use
samples = pancan['sample']
pancan = pancan.drop(columns=['sample'])

# Feature selection based on variance threshold
def variance_threshold_selector(data, threshold=0.045):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]

pancan_hvdnameth = variance_threshold_selector(pancan, 0.045)

# Insert sample column back into the dataset
pancan_hvdnameth.insert(0, 'sample', samples)

def modify_sample_ids(df):
    df['sample'] = df['sample'].apply(lambda x: x[:-3])
    return df

# Apply the function to modify the sample IDs
pancan_hvdnameth = modify_sample_ids(pancan_hvdnameth)

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

# Save the DNA methylation data as embeddings
save_data_by_case_id(pancan_hvdnameth, 'dna_methylation')
