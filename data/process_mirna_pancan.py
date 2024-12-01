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

# Load and transpose the Pancancer miRNA dataset
pancan = load_and_transpose('/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/pancan_mirna.gz')

# Drop constant features
sel1 = DropConstantFeatures(tol=0.9, variables=None, missing_values='raise')
sel1.fit(pancan)
pancan = sel1.transform(pancan)

# Separate the sample column for later use
samples = pancan['sample']
pancan = pancan.drop(columns=['sample'])

# Convert the variables to numerical variables
pancan = pancan.astype(float)

# Feature selection based on correlation
sel2 = SmartCorrelatedSelection(
    variables=None,
    method="pearson",
    threshold=0.9,
    missing_values="raise",
    selection_method="variance",
    estimator=None,
)
sel2.fit(pancan)
pancan = sel2.transform(pancan)

# Insert sample column back into the dataset
pancan.insert(0, 'sample', samples)

def modify_sample_ids(df):
    df['sample'] = df['sample'].apply(lambda x: x[:-3])
    return df

# Apply the function to modify the sample IDs
pancan_mirna = modify_sample_ids(pancan)

def save_data_by_case_id(df, config_name):
    base_directory = "/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/LUAD_LUSC_data/PAN/"
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)
    for idx, row in df.iterrows():
        case_id = row['sample']
        case_directory = os.path.join(base_directory, case_id)
        if not os.path.exists(case_directory):
            os.makedirs(case_directory)
        mirna_vector = row.drop('sample').values.astype(np.float32)
        file_path = os.path.join(case_directory, f"{config_name}.npy")
        np.save(file_path, mirna_vector)
        print(f"Saved {config_name} embedding to {file_path}")

# Save the miRNA data as embeddings
save_data_by_case_id(pancan_mirna, 'mirna')
