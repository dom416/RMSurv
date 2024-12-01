import pandas as pd
import numpy as np
import os

def clean_column_names(df):
    df.columns = df.columns.str.replace('.demographic', '', regex=False)
    df.columns = df.columns.str.replace('.diagnoses', '', regex=False)
    df.columns = df.columns.str.replace('.samples', '', regex=False)
    return df

def process_clinical_data(clin):
    clin = clean_column_names(clin)
    
    print(clin.shape)
    
    # Drop rows with NaNs in 'age_at_index'
    clin = clin.dropna(subset=['age_at_index'])
    
    # Convert the 'age_at_index' and 'age_at_diagnosis' columns to float
    clin['age_at_index'] = clin['age_at_index'].astype(float)
    clin['age_at_diagnosis'] = clin['age_at_diagnosis'].astype(float)
    
    # Check the variable format with pandas dtypes
    print(clin.dtypes)
    
    # Find the number of samples having age > 50
    if 'age_at_index' in clin.columns:
        age50 = clin['age_at_index'] > 50
        print(age50.value_counts())
    
    # Process gender
    if 'gender' in clin.columns:
        num_unique_genders = clin['gender'].nunique()
        unique_genders = clin['gender'].unique()
        print(f"The 'gender' column has {num_unique_genders} unique values.")
        print(f"The unique values are: {unique_genders}")
        
        # Count the number of each unique value
        value_counts = clin['gender'].value_counts()
        print(value_counts)
        
        # Convert 'gender' to numerical values: male=1, female=2
        gender_mapping = {'male': 1, 'female': 2}
        clin['gender_numeric'] = clin['gender'].map(gender_mapping)
    
    # Process race
    if 'race' in clin.columns:
        num_unique_race = clin['race'].nunique()
        unique_race = clin['race'].unique()
        print(f"The 'race' column has {num_unique_race} unique values.")
        print(f"The unique values in the 'race' column are: {unique_race}")
        
        # Count the number of each unique value
        value_counts = clin['race'].value_counts()
        print(value_counts)
        
        # Convert 'race' to numerical values
        race_mapping = {'white': 1, 'asian': 2, 'black or african american': 3, 'not reported': 4, 'american indian or alaska native': 5}
        clin['race_numeric'] = clin['race'].map(race_mapping)
    
    # Process ajcc_pathologic_stage
    if 'ajcc_pathologic_stage' in clin.columns:
        num_unique_stages = clin['ajcc_pathologic_stage'].nunique()
        unique_stages = clin['ajcc_pathologic_stage'].unique()
        print(f"The 'ajcc_pathologic_stage' column has {num_unique_stages} unique values.")
        print(f"The unique values are: {unique_stages}")
        
        # Count the number of each unique value
        value_counts = clin['ajcc_pathologic_stage'].value_counts()
        print(value_counts)
        
        # Convert 'ajcc_pathologic_stage' to numerical values
        stage_mapping = {
            'Stage 0': 1, 'Stage I': 10, 'Stage IA': 11, 'Stage IB': 12, 'Stage IC': 13, 
            'Stage II': 20, 'Stage IIA': 21, 'Stage IIB': 22, 'Stage IIC': 23, 
            'Stage III': 30, 'Stage IIIA': 31, 'Stage IIIB': 32, 'Stage IIIC': 33, 
            'Stage IV': 40, 'Stage IVA': 41, 'Stage IVB': 42, 'Stage IVC': 43, 
            'Not Reported': 50, 'Stage X': 50, 'None': 50, 'Stage is': 10
        }
        clin['ajcc_pathologic_stage_numeric'] = clin['ajcc_pathologic_stage'].map(stage_mapping)
    
        # Check for NaNs in the mapped column and fill them with a default value if necessary
        if clin['ajcc_pathologic_stage_numeric'].isna().any():
            print("Some stages could not be mapped. Filling NaNs with 50 (default value for 'not reported').")
            clin['ajcc_pathologic_stage_numeric'].fillna(50, inplace=True)
    
    # Drop unnecessary columns
    clin.drop(['age_at_diagnosis', 'gender', 'race', 'ajcc_pathologic_stage'], axis=1, inplace=True)
    
    # Rename columns
    clin.rename(columns={'age_at_index': 'age', 'submitter_id': 'sample'}, inplace=True)
    
    print(clin.head())
    
    return clin

def save_data_by_case_id(parquet_file_directory, config_name):
    # Construct the full path to the Parquet file
    parquet_file_path = os.path.join(parquet_file_directory, f"{config_name}.parquet")
    
    # Load the dataset from the Parquet file
    dataset = pd.read_parquet(parquet_file_path)

    # Clean column names
    dataset = clean_column_names(dataset)

    # Process the clinical data
    dataset = process_clinical_data(dataset)

    # Determine the case_id field based on config
    case_id_field = 'case_submitter_id'
    
    # Sort the DataFrame by case_id
    dataset.sort_values(case_id_field, inplace=True)

    # Define the base directory for saving embeddings
    base_directory = f"/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/LUAD_LUSC_data/PAN/"

    # Ensure the base directory exists
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    # Group by case_id and save each clin vector in the corresponding folder
    grouped = dataset.groupby(case_id_field)
    for case_id, group in grouped:
        case_directory = os.path.join(base_directory, str(case_id))
        if not os.path.exists(case_directory):
            os.makedirs(case_directory)
        
        # Iterate through each entry in the group
        for idx, row in group.iterrows():
            clin_vector = row[['age', 'gender_numeric', 'race_numeric', 'ajcc_pathologic_stage_numeric']].values.astype(np.float32)
            print(clin_vector)
            
            # Save each clin vector
            file_path = os.path.join(case_directory, f"{config_name}.npy")
            np.save(file_path, clin_vector)
            print(f"Saved {config_name} vector to {file_path}")

# Example usage
parquet_file_directory = '/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/HoneyBeemain/HoneyBeemain/embeddings/PANCAN/'
save_data_by_case_id(parquet_file_directory, 'clinical_data')
