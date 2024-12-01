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
    """
    Normalize the test predictions using the provided mean and standard deviation.

    Parameters:
    - test_pred (torch.Tensor): Predictions to normalize.
    - mean_predictions (torch.Tensor): Mean from training data.
    - std_predictions (torch.Tensor): Standard deviation from training data.

    Returns:
    - normalized_test_pred (torch.Tensor): Normalized predictions.
    """
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


def test_ensemble_TD(train_data, test_data, device, avg_best_weights, num_modalities_to_include, modalities_order):
    """
    Evaluates the ensemble model using modality-specific weights per time bin.

    Parameters:
    - train_data (Dataset): Training dataset.
    - test_data (Dataset): Test dataset.
    - device (torch.device): Device to perform computations on (CPU or CUDA).
    - avg_best_weights (numpy.ndarray): Array of optimal weights with shape (N, 20), where N is the number of modalities.
    - num_modalities_to_include (int): Number of modalities to include in the ensemble.
    - modalities_order (list): List of modality keys to include (e.g., ['modality_one', 'modality_two']).

    Returns:
    - cindex_test (float): Concordance index on the test set.
    """
    test_loader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False)
    train_loader = DataLoader(dataset=train_data, batch_size=len(train_data), shuffle=False)

    # Collect all training predictions to compute mean and std
    for batch_idx, batch in enumerate(train_loader):
        all_predictions = []
        for modality in modalities_order:
            x = batch[modality].to(device)  # Shape: [num_cases, 20]
            all_predictions.append(x)
        all_predictions = torch.stack(all_predictions, dim=0)  # Shape: [N, num_cases, 20]
        # Compute the mean and standard deviation across modalities and cases for each time bin
        mean_predictions = torch.mean(all_predictions, dim=(0, 1))  # Shape: [20,]
        std_predictions = torch.std(all_predictions, dim=(0, 1))    # Shape: [20,]
        # Handle cases where std is zero
        std_predictions = torch.where(std_predictions == 0, torch.ones_like(std_predictions), std_predictions)
        mean_predictions = mean_predictions.to(device)
        std_predictions = std_predictions.to(device)
        break  # All data loaded in one batch

    best_mult = 1
    best_ibs = float('inf')

    # Find the best std multiplier
    for batch_idx, batch in enumerate(train_loader):
        censor = batch['censor'].to(device).detach().cpu().numpy()
        survtime = batch['survival_time'].to(device).detach().cpu().numpy()
        # true_time_bin is defined but not used; remove if unnecessary
        # true_time_bin = batch['true_time_bin'].to(device).detach().cpu().numpy()

        for std_mult in np.arange(0.1, 10.1, 0.1):
            modality_predictions = []
            for modality in modalities_order:
                x = batch[modality].squeeze(1).to(device)  # Shape: [num_cases, 20]
                # Normalize predictions with the current std multiplier
                x_normalized = normalize_predictions(x, mean_predictions, std_predictions * std_mult).to(device)  # [num_cases, 20]
                modality_predictions.append(x_normalized)  # List of [num_cases, 20]

            # Convert avg_best_weights to torch tensor and select relevant modalities
            weights = torch.tensor(avg_best_weights[:num_modalities_to_include], dtype=torch.float32, device=device)  # Shape: [N, 20]
            # Ensure weights are of shape [N, 20] and normalized per time bin
            weights_normalized = weights / weights.sum(dim=0, keepdim=True)  # Shape: [N, 20]

            # Stack modality_predictions into [N, num_cases, 20]
            modality_preds_tensor = torch.stack(modality_predictions, dim=0)  # Shape: [N, num_cases, 20]

            # Apply weights: element-wise multiplication and sum over modalities
            combined_pred = (weights_normalized[:, None, :] * modality_preds_tensor).sum(dim=0)  # Shape: [num_cases, 20]

            # Normalize the combined predictions
            combined_pred = normalize_predictions(combined_pred, mean_predictions, std_predictions * std_mult).to(device)

            # Calculate hazards and survival
            hazards = torch.sigmoid(combined_pred)  # Shape: [num_cases, 20]
            survival = torch.cumprod(1 - hazards, dim=1).to(device)  # Shape: [num_cases, 20]

            dtype = np.dtype([('event', bool), ('time', float)])
            survival_train = np.array(list(zip((1 - censor).astype(bool), survtime)), dtype=dtype)
            estimate = survival.detach().cpu().numpy()  # Shape: [num_cases, 20]
            times = np.arange(1, 21) * (365 / 2)

            # Calculate Integrated Brier Score (IBS)
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
        censor = batch['censor'].to(device).detach().cpu().numpy()
        survtime = batch['survival_time'].to(device).detach().cpu().numpy()
        # true_time_bin = batch['true_time_bin'].to(device).detach().cpu().numpy()
        case_id = batch['case_id']

        modality_predictions = []
        for modality in modalities_order:
            x = batch[modality].squeeze(1).to(device)  # Shape: [num_cases, 20]
            # Normalize predictions with the best std multiplier
            x_normalized = normalize_predictions(x, mean_predictions, std_predictions * best_mult).to(device)  # [num_cases, 20]
            modality_predictions.append(x_normalized)  # List of [num_cases, 20]

        # Convert avg_best_weights to torch tensor and select relevant modalities
        weights = torch.tensor(avg_best_weights[:num_modalities_to_include], dtype=torch.float32, device=device)  # Shape: [N, 20]
        # Ensure weights are of shape [N, 20] and normalized per time bin
        weights_normalized = weights / weights.sum(dim=0, keepdim=True)  # Shape: [N, 20]

        # Stack modality_predictions into [N, num_cases, 20]
        modality_preds_tensor = torch.stack(modality_predictions, dim=0)  # Shape: [N, num_cases, 20]

        # Apply weights: element-wise multiplication and sum over modalities
        combined_pred = (weights_normalized[:, None, :] * modality_preds_tensor).sum(dim=0)  # Shape: [num_cases, 20]

        # Normalize the combined predictions
        combined_pred = normalize_predictions(combined_pred, mean_predictions, std_predictions * best_mult).to(device)

        # Calculate hazards and survival
        hazards = torch.sigmoid(combined_pred)  # Shape: [num_cases, 20]
        survival = torch.cumprod(1 - hazards, dim=1).to(device)  # Shape: [num_cases, 20]

        # Compute risk scores
        risk = -torch.sum(survival, dim=1).detach().cpu().numpy()  # Shape: [num_cases,]

        # Collect all relevant information
        all_risk_scores.append(risk)
        all_survival_scores.append(survival.detach().cpu().numpy())
        all_censorships.append(censor)
        all_event_times.append(survtime)
        all_case_ids.append(case_id)

    # Concatenate all batches
    all_risk_scores = np.concatenate(all_risk_scores)
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)

    # Compute Concordance Index (C-Index)
    cindex_test = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08
    )[0]
    return cindex_test
