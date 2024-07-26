import torch
from torch.utils.data import Dataset
import numpy as np
import os
from torchvision import transforms

def standardize(tensor):
    mean = tensor.mean(dim=1, keepdim=True)
    std = tensor.std(dim=1, keepdim=True)
    std[std == 0] = 1  # Prevent division by zero
    return (tensor - mean) / std

class SurvivalDataset(Dataset):
    def __init__(self, data_dir, fold=1, seed=41, n_folds=5, split='train'):
        self.data_dir = data_dir
        self.data = self.load_data(data_dir)
        self.embedding_shapes = self.compute_embedding_shapes()
        self.split_data(split, fold, n_folds, seed)
    
    def compute_embedding_shapes(self):
        embedding_shapes = {}
        for entry in self.data:
            for data_type, embedding in entry.items():
                if data_type not in ['case_id', 'survival_time', 'true_time_bin', 'censor'] and embedding is not None:
                    embedding_shapes[data_type] = embedding.shape
        return embedding_shapes

    def load_embedding(self, case_path, data_type, allow_pickle=False):
        try:
            file_path = next(os.path.join(case_path, f) for f in os.listdir(case_path) if data_type in f)
            embedding = np.load(file_path, allow_pickle=allow_pickle)
            if isinstance(embedding, np.ndarray) and embedding.dtype == object:
                embedding = np.array([float(e) if isinstance(e, (int, float)) else 0 for e in embedding.flatten()], dtype=np.float32).reshape(embedding.shape)
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            elif embedding.ndim == 3 and embedding.shape[0] == 1 and embedding.shape[1] == 1:
                embedding = embedding.reshape(1, -1)
            elif embedding.ndim == 2 and embedding.shape[0] > 1:
                embedding = embedding[0:1, :]
            embedding = torch.tensor(embedding, dtype=torch.float32).reshape(1, -1)  # Adjust size as needed
            return embedding
        except (FileNotFoundError, StopIteration, ValueError):
            return None

    def load_data(self, data_dir):
        data_entries = []
        period_length = 182.5  # One period is 365/2 days
        num_periods = 20  # Total 20 periods
        
        for case_id in os.listdir(data_dir):
            case_path = os.path.join(data_dir, case_id)
            if not os.path.isdir(case_path):  # Skip non-directory files
                continue
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
                'clinical_data': self.load_embedding(case_path, 'clinical_data', allow_pickle=True),
                'pathology_report': self.load_embedding(case_path, 'pathology_report', allow_pickle=True),
                'x_omic': self.load_embedding(case_path, 'x_omic', allow_pickle=True),
                'senmo_48': self.load_embedding(case_path, 'senmo_48', allow_pickle=True),
                'senmo_1024': self.load_embedding(case_path, 'senmo_1024', allow_pickle=True),
                'gene_expression': self.load_embedding(case_path, 'gene_expression', allow_pickle=True),
                'dna_methylation': self.load_embedding(case_path, 'dna_methylation', allow_pickle=True),
                'copy_number': self.load_embedding(case_path, 'copy_number', allow_pickle=True),
                'mirna': self.load_embedding(case_path, 'mirna', allow_pickle=True),
                'protein_expression': self.load_embedding(case_path, 'protein_expression', allow_pickle=True),
                'somatic_mutation': self.load_embedding(case_path, 'somatic_mutation', allow_pickle=True),
                'slide_image_features': self.load_embedding(case_path, 'slide_image_features', allow_pickle=True),
                'modality_one': self.load_embedding(case_path, 'modality_one', allow_pickle=True),
                'modality_two': self.load_embedding(case_path, 'modality_two', allow_pickle=True),
                'modality_three': self.load_embedding(case_path, 'modality_three', allow_pickle=True),
                'modality_four': self.load_embedding(case_path, 'modality_four', allow_pickle=True),
                'modality_five': self.load_embedding(case_path, 'modality_five', allow_pickle=True),
                'modality_six': self.load_embedding(case_path, 'modality_six', allow_pickle=True),
                'modality_seven': self.load_embedding(case_path, 'modality_seven', allow_pickle=True)
            }
            data_entries.append(data_entry)

        return data_entries

    def split_data(self, split, fold, n_folds, seed):
        torch.manual_seed(seed)  # Set the random seed for reproducibility
        total_cases = len(self.data)
        indices = torch.randperm(total_cases).tolist()  # Shuffle indices

        fold_size = total_cases // n_folds
        remainders = total_cases % n_folds

        sizes = [fold_size + (1 if i < remainders else 0) for i in range(n_folds)]

        start_index = sum(sizes[:fold - 1])
        end_index = start_index + sizes[fold - 1]

        test_indices = indices[start_index:end_index]
        train_indices = indices[:start_index] + indices[end_index:]

        self.test_data = [self.data[i] for i in test_indices]
        self.train_data = [self.data[i] for i in train_indices]

        self.data = self.train_data if split == 'train' else self.test_data

    def __getitem__(self, idx):
        entry = self.data[idx]
        data = {
            'case_id': entry['case_id'],
            'survival_time': entry['survival_time'],
            'censor': entry['censor'],
            'true_time_bin': entry['true_time_bin'],
        }
        for data_type, shape in self.embedding_shapes.items():
            embedding = entry.get(data_type)
            if embedding is None:
                embedding = torch.zeros(shape)  # Adjust size to match the typical size for the data type
            data[data_type] = embedding

        return data

    def __len__(self):
        return len(self.data)
