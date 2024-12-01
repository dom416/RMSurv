import pandas as pd
import numpy as np
import os

def find_valid_shape(dataset):
    for _, data in dataset.iterrows():
        if data['embedding'] is not None:
            return np.frombuffer(data['embedding'], dtype=np.float32).reshape(data['embedding_shape']).shape
    return (1, 14, 14, 2048)  # Default shape if no valid embedding is found

def save_data_by_case_id(parquet_file_directory, config_name):
    # Construct the full path to the Parquet file
    parquet_file_path = os.path.join(parquet_file_directory, f"{config_name}.parquet")
    
    # Load the dataset from the Parquet file
    dataset = pd.read_parquet(parquet_file_path)

    valid_shape = find_valid_shape(dataset)

    # Determine the case_id field based on config
    if config_name == 'clinical_data':
        case_id_field = 'case_submitter_id'
    else:
        case_id_field = 'PatientID'
    
    # Sort the DataFrame by case_id
    dataset.sort_values(case_id_field, inplace=True)

    # Define the base directory for saving embeddings
    base_directory = f"/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/LUAD_LUSC_data/PAN/"

    # Ensure the base directory exists
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    # Group by case_id and save each embedding in the corresponding folder
    grouped = dataset.groupby(case_id_field)
    for case_id, group in grouped:
        case_directory = os.path.join(base_directory, str(case_id))
        if not os.path.exists(case_directory):
            os.makedirs(case_directory)
        
        # Iterate through each entry in the group
        for idx, row in group.iterrows():
            if row['embedding'] is None:
                embedding = np.zeros(valid_shape, dtype=np.float32)
            else:
                embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                embedding = embedding.reshape(row['embedding_shape'])

            # Save each embedding with an index to differentiate
            file_path = os.path.join(case_directory, f"{config_name}.npy")
            np.save(file_path, embedding)
            print(f"Saved {config_name} embedding to {file_path}")

# Example usage
parquet_file_directory = '/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/HoneyBeemain/HoneyBeemain/embeddings/PANCAN/'
save_data_by_case_id(parquet_file_directory, 'pathology_report')

