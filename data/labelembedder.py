import pandas as pd
import numpy as np
import os

def save_data_by_case_id(parquet_file_directory, config_name):
    # Construct the full path to the Parquet file
    parquet_file_path = os.path.join(parquet_file_directory, f"{config_name}.parquet")
    
    # Load the dataset from the Parquet file
    dataset = pd.read_parquet(parquet_file_path)

    # Determine the case_id field based on config
    if config_name == 'clinical_data':
        case_id_field = 'case_submitter_id'
    else:
        case_id_field = 'PatientID'

    # Define the base directory for saving data
    base_directory = "/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/LUAD_LUSC_data/PAN/"

    # Ensure the base directory exists
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    # Process each case based on case_id
    grouped = dataset.groupby(case_id_field)
    for case_id, group in grouped:
        case_directory = os.path.join(base_directory, str(case_id))
        if not os.path.exists(case_directory):
            os.makedirs(case_directory)
        
        # Iterate through each entry in the group
        for idx, row in group.iterrows():
            # Check if days_to_death is null
            if pd.isnull(row['days_to_death']):
                censor = 1
                survival_time = row['days_to_last_follow_up']
            else:
                censor = 0
                survival_time = row['days_to_death']

            # Save censor and survival_time to the folder as .npy files
            np.save(os.path.join(case_directory, 'censor.npy'), censor)
            np.save(os.path.join(case_directory, 'survival_time.npy'), survival_time)
            print(survival_time)

            print(f"Saved censor and survival time for case_id {case_id} at index {idx}")

# Example usage
save_data_by_case_id('/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/HoneyBeemain/HoneyBeemain/embeddings/PANCAN/', 'clinical_data')
