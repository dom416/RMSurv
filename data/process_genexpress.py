import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import VarianceThreshold
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures

def load_and_transpose(file_path):
    df = pd.read_table(file_path)
    df = df.transpose()
    df = df.reset_index()
    df.columns = df.iloc[0]
    df = df[1:]
    df = df.rename(columns={df.columns[0]: 'sample'})
    return df

# Load and transpose LUAD and LUSC datasets
luad = load_and_transpose('/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/TCGA-LUAD.htseq_fpkm.tsv.gz')
lusc = load_and_transpose('/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/TCGA-LUSC.htseq_fpkm.tsv.gz')

# Add a column to keep track of the dataset type
luad['dataset'] = 'LUAD'
lusc['dataset'] = 'LUSC'

# Concatenate the datasets
combined = pd.concat([luad, lusc], axis=0, ignore_index=True)

# Drop constant and duplicate features
sel1 = DropConstantFeatures(tol=0.95, variables=None, missing_values='raise')
sel1.fit(combined)
combined = sel1.transform(combined)

sel2 = DropDuplicateFeatures(variables=None, missing_values='raise')
sel2.fit(combined)
combined = sel2.transform(combined)

# Drop columns with NaNs
combined = combined.dropna(axis=1)

# Apply variance threshold selector to the combined dataset
def variance_threshold_selector(data, threshold=0.88):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]

combined_genes = combined.drop(columns=['sample', 'dataset'])
combined_hvgenes = variance_threshold_selector(combined_genes, 0.2)

# Insert sample names back into the dataset
combined_hvgenes.insert(0, 'sample', combined['sample'])

# Separate the combined dataset back into LUAD and LUSC
luad_hvgenes = combined_hvgenes[combined['dataset'] == 'LUAD'].copy()
lusc_hvgenes = combined_hvgenes[combined['dataset'] == 'LUSC'].copy()

def modify_sample_ids(df):
    df['sample'] = df['sample'].apply(lambda x: x[:-4])
    return df

# Apply the function to modify the sample IDs
luad_hvgenes = modify_sample_ids(luad_hvgenes)
lusc_hvgenes = modify_sample_ids(lusc_hvgenes)

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
        print(embedding.shape)
        print(f"Saved {config_name} embedding to {file_path}")

# Save the gene expression data as embeddings for both datasets
save_data_by_case_id(luad_hvgenes, 'gene_expression', 'LUAD')
save_data_by_case_id(lusc_hvgenes, 'gene_expression', 'LUSC')
