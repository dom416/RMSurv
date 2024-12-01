import torch
from torch.utils.data import Dataset
import numpy as np
import os

class SurvivalDataset(Dataset):
    def __init__(self, data_dir, fold=1, seed=41, n_folds=5, split='train', nested_cv=False, cv_fold=None):
        self.data_dir = data_dir
        self.data_types = ['clinical_data', 'pathology_report', 
                           'gene_expression', 'dna_methylation', 'copy_number', 'mirna', 
                           'protein_expression', 
                           'modality_one', 'modality_two', 'modality_three', 'modality_four', 
                           'modality_five', 'modality_six', 'modality_seven']
        self.data, self.median_embeddings = self.load_and_process_data()
        self.split_data(split=split, fold=fold, n_folds=n_folds, seed=seed, nested_cv=nested_cv, cv_fold=cv_fold)

    def load_and_process_data(self):
        data_entries = []
        period_length = 182.5  # One period is 365/2 days
        num_periods = 20  # Total 20 periods
        embeddings_dict = {data_type: [] for data_type in self.data_types}

        # List all case directories
        case_ids = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]

        for case_id in case_ids:
            case_path = os.path.join(self.data_dir, case_id)
            survival_time_path = os.path.join(case_path, 'survival_time.npy')
            censor_path = os.path.join(case_path, 'censor.npy')

            try:
                survival_time = np.load(survival_time_path).astype(float)
                censor = np.load(censor_path).astype(float)
                if survival_time <= 0:
                    continue
            except (FileNotFoundError, ValueError):
                continue

            true_time_bin = min(int(survival_time // period_length), num_periods - 1)

            data_entry = {
                'case_id': case_id,
                'survival_time': survival_time,
                'true_time_bin': true_time_bin,
                'censor': torch.tensor(censor, dtype=torch.float32),
            }

            # Load embeddings for each data type
            for data_type in self.data_types:
                file_path = os.path.join(case_path, f'{data_type}.npy')
                if os.path.exists(file_path):
                    embedding = self.load_embedding_from_file(file_path)
                    data_entry[data_type] = embedding
                    embeddings_dict[data_type].append(embedding)
                else:
                    data_entry[data_type] = None

            data_entries.append(data_entry)

        # Compute median embeddings
        median_embeddings = {}
        for data_type, embeddings in embeddings_dict.items():
            median_embedding = self.compute_median_embedding(embeddings)
            if median_embedding is not None:
                median_embeddings[data_type] = median_embedding
            else:
                median_embeddings[data_type] = torch.zeros((1, 1024))  # Adjust size as needed

        return data_entries, median_embeddings

    def load_embedding_from_file(self, file_path):
        try:
            embedding = np.load(file_path, allow_pickle=True)
            embedding = np.nan_to_num(embedding, nan=0.0)  # Replace NaNs with zeros
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            elif embedding.ndim > 2:
                embedding = embedding.reshape(1, -1)
            else:
                embedding = embedding[:1, :]  # Take the first row if multiple
            return torch.tensor(embedding, dtype=torch.float32)
        except (FileNotFoundError, ValueError):
            return None

    def compute_median_embedding(self, embeddings):
        if not embeddings:
            return None
        embeddings_tensor = torch.stack(embeddings, dim=0)
        median_embedding = torch.median(embeddings_tensor, dim=0)[0]
        return median_embedding

    # The rest of the class remains unchanged
    def split_data(self, split, fold, n_folds, seed, nested_cv=False, cv_fold=None):
        # Your existing split_data method
        torch.manual_seed(seed)
        total_cases = len(self.data)
        indices = torch.randperm(total_cases).tolist()

        fold_size = total_cases // n_folds
        remainders = total_cases % n_folds
        sizes = [fold_size + (1 if i < remainders else 0) for i in range(n_folds)]

        start_index = sum(sizes[:fold - 1])
        end_index = start_index + sizes[fold - 1]

        test_indices = indices[start_index:end_index]
        train_indices = indices[:start_index] + indices[end_index:]

        self.test_data = [self.data[i] for i in test_indices]
        self.train_data = [self.data[i] for i in train_indices]

        if not nested_cv:
            self.data = self.train_data if split == 'train' else self.test_data
        else:
            # Nested CV logic
            pass

    def __getitem__(self, idx):
        entry = self.data[idx]
        data = {
            'case_id': entry['case_id'],
            'survival_time': entry['survival_time'],
            'censor': entry['censor'],
            'true_time_bin': entry['true_time_bin'],
        }
        for data_type in self.data_types:
            embedding = entry.get(data_type)
            if embedding is None:
                embedding = self.median_embeddings[data_type]
            else:
                nan_mask = torch.isnan(embedding)
                if nan_mask.any():
                    median_embedding = self.median_embeddings[data_type]
                    embedding = embedding.clone()
                    embedding[nan_mask] = median_embedding[nan_mask]
            data[data_type] = embedding
        return data

    def __len__(self):
        return len(self.data)
