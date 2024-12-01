import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from sksurv.metrics import concordance_index_censored, integrated_brier_score
import itertools
import pandas as pd

import psutil
import threading


def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}!")
    if torch.isinf(tensor).any():
        print(f"Infinity detected in {name}!")

def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 ** 2):.2f} MB")

def log_thread_info(msg=""):
    print(f"Thread {threading.current_thread().name}: {msg}")

def log_thread_results(weight_comb, cindex_test):
    print(f"Thread {threading.current_thread().name}: Weight combination: {weight_comb}, C-Index: {cindex_test:.4f}")

def normalize_predictions(test_pred, mean_predictions, std_predictions):
    # Avoid division by zero
    std_predictions = torch.where(std_predictions == 0, torch.ones_like(std_predictions), std_predictions)
    # Calculate the mean and standard deviation of the test predictions
    test_pred_mean = torch.mean(test_pred, dim=0)
    test_pred_std = torch.std(test_pred, dim=0)
    test_pred_std = torch.where(test_pred_std == 0, torch.ones_like(test_pred_std), test_pred_std)
    # Standardize the test predictions to mean 0 and std 1
    standardized_test_pred = (test_pred - test_pred_mean) / test_pred_std
    # Scale and shift to match the training set statistics
    normalized_test_pred = standardized_test_pred * std_predictions + mean_predictions
    return normalized_test_pred

def test_ensemble(train_data, test_data, device, avg_best_weights, num_modalities_to_include,modalities_order):

    test_loader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False)
    train_loader = DataLoader(dataset=train_data, batch_size=len(train_data), shuffle=False)

    # Collect all training predictions to compute mean and std
    for batch_idx, batch in enumerate(train_loader):
        all_predictions = []
        for modality in modalities_order:
            x = batch[modality].to(device)
            all_predictions.append(x)
        all_predictions = torch.cat(all_predictions, dim=0)
        # Compute the mean and standard deviation along the batch dimension (dimension 0)
        mean_predictions = torch.mean(all_predictions, dim=0)
        std_predictions = torch.std(all_predictions, dim=0)
        mean_predictions = mean_predictions.to(device)
        std_predictions = std_predictions.to(device)
        # Handle cases where std is zero
        std_predictions = torch.where(std_predictions == 0, torch.ones_like(std_predictions), std_predictions)

    best_mult = 1
    best_ibs = float('inf')

    # Find the best std multiplier
    for batch_idx, batch in enumerate(train_loader):
        censor = batch['censor'].to(device).detach().cpu().numpy()
        survtime = batch['survival_time'].to(device).detach().cpu().numpy()
        true_time_bin = batch['true_time_bin'].to(device).detach().cpu().numpy()

        for std_mult in np.arange(0.1, 10.1, 0.1):
            modality_predictions = []
            for modality in modalities_order:
                x = batch[modality].squeeze(1).to(device)
                x = normalize_predictions(x, mean_predictions, std_predictions * std_mult).to(device)
                modality_predictions.append(x)

            weights = avg_best_weights[:len(modalities_order)]
            x = sum(w * x_mod for w, x_mod in zip(weights, modality_predictions)) / sum(weights)
            x = normalize_predictions(x, mean_predictions, std_predictions * std_mult).to(device)

            hazards = torch.sigmoid(x)
            survival = torch.cumprod(1 - hazards, dim=1).to(device)

            dtype = np.dtype([('event', bool), ('time', float)])
            survival_train = np.array(list(zip((1 - censor).astype(bool), survtime)), dtype=dtype)
            estimate = survival.detach().cpu().numpy()
            times = np.arange(1, 21) * (365 / 2)

            ibs = integrated_brier_score(survival_train, survival_train, estimate, times)

            if ibs < best_ibs:
                best_ibs = ibs 
                best_mult = std_mult

    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_case_ids = []
    all_survival_scores = []

    # Evaluate on test data
    for batch_idx, batch in enumerate(test_loader):
        censor = batch['censor'].to(device)
        survtime = batch['survival_time'].to(device)
        true_time_bin = batch['true_time_bin'].to(device)
        case_id = batch['case_id']

        modality_predictions = []
        for modality in modalities_order:
            x = batch[modality].squeeze(1).to(device)
            x = normalize_predictions(x, mean_predictions, std_predictions * best_mult).to(device)
            modality_predictions.append(x)

        weights = avg_best_weights[:len(modalities_order)]
        x = sum(w * x_mod for w, x_mod in zip(weights, modality_predictions)) / sum(weights)
        x = normalize_predictions(x, mean_predictions, std_predictions * best_mult).to(device)
        

        hazards = torch.sigmoid(x)
        survival = torch.cumprod(1 - hazards, dim=1).to(device)

        risk = -torch.sum(survival, dim=1).detach().cpu().numpy()

        all_risk_scores.append(risk)
        all_survival_scores.append(survival.detach().cpu().numpy())
        all_censorships.append(censor.detach().cpu().numpy())
        all_event_times.append(survtime.detach().cpu().numpy())
        all_case_ids.append(case_id)

    all_risk_scores = np.concatenate(all_risk_scores)
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)

    cindex_test = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08
    )[0]
    return cindex_test

def test_ensemble_weight_finding(train_data, test_data, device, num_modalities_to_include=6, num_runs=10, num_weights=100):
    """
    Finds the ideal weights for combining multiple risk score modalities based on the validation set
    using an iterative forward selection approach.
    """
    # Define the list of modalities to include, in the desired order
    modalities_order = modalities_order_full[:num_modalities_to_include]

    test_loader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False)
    train_loader = DataLoader(dataset=train_data, batch_size=len(train_data), shuffle=False)

    # Collect training predictions to compute mean and std
    for batch_idx, batch in enumerate(train_loader):
        all_predictions = []
        for modality in modalities_order:
            x = batch[modality].to(device)
            all_predictions.append(x)
        all_predictions = torch.cat(all_predictions, dim=0)
        mean_predictions = torch.mean(all_predictions, dim=0)
        std_predictions = torch.std(all_predictions, dim=0)
        mean_predictions = mean_predictions.to(device)
        std_predictions = std_predictions.to(device)
        std_predictions = torch.where(std_predictions == 0, torch.ones_like(std_predictions), std_predictions)

    # Prepare test data
    test_data_list = []
    for batch_idx, batch in enumerate(test_loader):
        censor = batch['censor'].to(device)
        survtime = batch['survival_time'].to(device)
        modality_preds = []
        for modality in modalities_order:
            x = batch[modality].squeeze(1).to(device)
            x = normalize_predictions(x, mean_predictions, std_predictions).detach().cpu().numpy()
            modality_preds.append(x)
        test_data_list.append((censor.detach().cpu().numpy(), survtime.detach().cpu().numpy(), modality_preds))

    censorships = []
    event_times = []
    modality_preds_combined = [[] for _ in range(len(modalities_order))]

    for censor, survtime, modality_preds in test_data_list:
        for i, pred in enumerate(modality_preds):
            modality_preds_combined[i].append(pred)
        censorships.append(censor)
        event_times.append(survtime)

    modality_preds_combined = [np.concatenate(preds, axis=0) for preds in modality_preds_combined]
    censorships = np.concatenate(censorships)
    event_times = np.concatenate(event_times)

    num_modalities = len(modalities_order)
    weights_range = np.linspace(0, 1, num_weights)

    def find_best_weight(existing_x, new_x, censorships, event_times, weights_range):
        best_weight = None
        best_cindex = -np.inf

        for w in weights_range:
            combined_x = w * existing_x + (1 - w) * new_x
            hazards = torch.sigmoid(torch.from_numpy(combined_x).to(device))
            survival = torch.cumprod(1 - hazards, dim=1).cpu().numpy()
            risk_scores = 1 - np.sum(survival, axis=1)
            cindex = concordance_index_censored(
                (1 - censorships).astype(bool), event_times, risk_scores, tied_tol=1e-08
            )[0]
            if cindex > best_cindex:
                best_cindex = cindex
                best_weight = w

        return best_weight, best_cindex

    overall_weights = np.zeros((num_runs, num_modalities))
    combined_vector = None
    overall_weights_run = np.zeros(num_modalities)

    print('\nStarting Iterative Forward Selection for Weight Finding...')

    for run in range(num_runs):
        modality_index = run % num_modalities
        print(f'\n--- Run {run + 1} ---')
        if run == 0:
            modality1 = 0
            modality2 = 1

            best_weight, best_cindex = find_best_weight(
                existing_x=modality_preds_combined[modality1],
                new_x=modality_preds_combined[modality2],
                censorships=censorships,
                event_times=event_times,
                weights_range=weights_range
            )

            combined_vector = best_weight * modality_preds_combined[modality1] + \
                              (1 - best_weight) * modality_preds_combined[modality2]
            overall_weights_run[modality1] = best_weight
            overall_weights_run[modality2] = 1 - best_weight

            print(f'Combined Risk Scores: {modalities_order[modality1]} and {modalities_order[modality2]} '
                  f'with weight {best_weight:.4f}, c-index = {best_cindex:.4f}')
        else:
            next_modality = modality_index

            best_weight, best_cindex = find_best_weight(
                existing_x=combined_vector,
                new_x=modality_preds_combined[next_modality],
                censorships=censorships,
                event_times=event_times,
                weights_range=weights_range
            )

            combined_vector = best_weight * combined_vector + \
                              (1 - best_weight) * modality_preds_combined[next_modality]
            overall_weights_run = overall_weights_run * best_weight
            overall_weights_run[next_modality] += (1 - best_weight)

            print(f'Added Risk Score: {modalities_order[next_modality]} with weight {best_weight:.4f}, '
                  f'new c-index = {best_cindex:.4f}')

        overall_weights[run, :] = overall_weights_run.copy()

    hazards = torch.sigmoid(torch.from_numpy(combined_vector).to(device))
    survival = torch.cumprod(1 - hazards, dim=1).cpu().numpy()
    final_risk_scores = - np.sum(survival, axis=1)
    final_cindex = concordance_index_censored(
        (1 - censorships).astype(bool), event_times, final_risk_scores, tied_tol=1e-08
    )[0]
    final_weights = tuple(overall_weights[-1, :])

    print(f'\nFinal Combined c-index (Run {num_runs}): {final_cindex:.4f}')
    print('Final Weights Vector:')
    for i in range(num_modalities):
        print(f'  {modalities_order[i]}: Weight = {final_weights[i]:.6f}')

    return final_cindex, final_weights
    
    
def compute_rho_between(train_data, test_data, device, modalities_order):
    """
    Computes the Pearson correlation matrix (rho_between) between the unimodal test set outputs.

    Parameters:
        train_data (Dataset): Training dataset containing predictions from each modality.
        test_data (Dataset): Validation dataset containing predictions from each modality and survival information.
        device (torch.device): Computational device ('cuda' for GPU usage).
        modalities_order (list): List of modality keys to include (e.g., ['modality_one', 'modality_five', ...]).

    Returns:
        rho_between (np.ndarray): Pearson correlation matrix.
    """

    # Modality names corresponding to the modality keys
    modality_names_dict = {
        'modality_one': 'Modality 1',
        'modality_two': 'Modality 2',
        'modality_three': 'Modality 3',
        'modality_four': 'Modality 4',
        'modality_five': 'Modality 5',
        'modality_seven': 'Modality 6'
    }

    # Get modality names in the same order
    modality_names = [modality_names_dict[key] for key in modalities_order]
    num_modalities = len(modalities_order)

    # Step 1: Load and process the training data to compute mean and std
    train_loader = DataLoader(dataset=train_data, batch_size=len(train_data), shuffle=False)
    for batch_idx, batch in enumerate(train_loader):
        # Extract modality predictions and move to device
        modality_predictions_train = []
        for key in modalities_order:
            pred = batch[key].to(device)  # Shape: (N, 20) assuming each modality has 20 time periods
            modality_predictions_train.append(pred)

        # Concatenate all modality predictions along the batch dimension
        # Shape: (num_modalities*N, 20)
        all_predictions_train = torch.cat(modality_predictions_train, dim=0)

        # Compute mean and std across the batch and time dimensions
        # Shape: (20,)
        mean_predictions = torch.mean(all_predictions_train, dim=0)
        std_predictions = torch.std(all_predictions_train, dim=0)

        # Move mean and std to device
        mean_predictions = mean_predictions.to(device)
        std_predictions = std_predictions.to(device)

        break  # Assuming only one batch in train_loader

    # Step 2: Load and normalize the test data
    test_loader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False)
    test_data_list = []

    for batch_idx, batch in enumerate(test_loader):
        censor = batch['censor'].to(device)
        survtime = batch['survival_time'].to(device)
        true_time_bin = batch['true_time_bin'].to(device)

        # Collect modality predictions
        modality_predictions = []
        for key in modalities_order:
            pred = batch[key].squeeze(1).to(device)  # Shape: (N, 20)
            modality_predictions.append(pred)

        test_data_list.append((censor, survtime, *modality_predictions))

    # Step 3: Extract and normalize test data
    modality_preds = [[] for _ in range(num_modalities)]  # List to store normalized predictions per modality

    for censor, survtime, *modalities in test_data_list:
        for i in range(num_modalities):
            norm_pred = normalize_predictions(modalities[i], mean_predictions, std_predictions).detach().cpu().numpy()  # Shape: (N, 20)
            modality_preds[i].append(norm_pred)

    # Concatenate all normalized predictions for each modality
    modality_preds = [np.concatenate(preds, axis=0) for preds in modality_preds]  # List of arrays, each shape: (N, 20)

    # Step 4: Compute risk scores for each modality
    def compute_risk_scores(modality_pred):
        """
        Computes the risk scores for a single modality.

        Parameters:
            modality_pred (np.ndarray): Normalized predictions from a modality. Shape: (N, 20)

        Returns:
            risk_scores (np.ndarray): Risk scores for each sample. Shape: (N,)
        """
        # Convert to tensor and move to device
        modality_tensor = torch.from_numpy(modality_pred).to(device)  # Shape: (N, 20)

        # Compute hazards using sigmoid activation
        hazards = torch.sigmoid(modality_tensor)  # Shape: (N, 20)

        # Compute survival probabilities
        survival = torch.cumprod(1 - hazards, dim=1).cpu().numpy()  # Shape: (N, 20)

        # Compute risk scores
        risk_scores = - np.sum(survival, axis=1)  # Shape: (N,)
        
        risk_scores = torch.cumsum(modality_tensor, dim=1)[:, -1]  # Shape: (N,)
    
        # Convert the result back to a NumPy array
        risk_scores = risk_scores.cpu().numpy()


        return risk_scores

    # Compute risk scores for all modalities
    risk_scores = [compute_risk_scores(modality_preds[i]) for i in range(num_modalities)]  # List of arrays, each shape: (N,)

    # Step 5: Stack risk scores into a (num_modalities, N) matrix
    risk_scores_matrix = np.stack(risk_scores, axis=0)  # Shape: (num_modalities, N)

    # Step 6: Compute Pearson correlation matrix
    rho_between = np.corrcoef(risk_scores_matrix)  # Shape: (num_modalities, num_modalities)

    # Step 7: Create a pandas DataFrame for better readability
    rho_between_df = pd.DataFrame(rho_between, index=modality_names, columns=modality_names)

    # Print the Pearson Correlation Matrix
    print("\nPearson Correlation Matrix (rho_between):")
    print(rho_between_df)

    return rho_between
