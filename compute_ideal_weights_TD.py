import numpy as np
import torch
from torch.utils.data import DataLoader
from sksurv.metrics import concordance_index_censored, integrated_brier_score
import random
from tqdm import tqdm
from scipy.optimize import differential_evolution
from scipy.special import expit  # For sigmoid in numpy

def compute_ideal_weights_TD(
    train_data,
    test_data,
    modalities_order,
    desired_cindices_N,
    device,
    num_iterations=10000,
    early_stopping_threshold=1e-6,
    verbose=True
):
    """
    Simulates survival times to achieve desired c-indices for each modality and computes optimal weights for combining modalities
    with a distinct weight for each modality per time bin.

    Parameters:
    - train_data (Dataset): Training dataset containing survival times.
    - test_data (Dataset): Test dataset containing risk scores and survival information.
    - modalities_order (list): List of modality keys to include (e.g., ['modality_one', 'modality_two']).
    - desired_cindices_N (list): Desired c-indices for each modality, ordered according to modalities_order.
    - device (torch.device): Device to perform computations on (CPU or CUDA).
    - num_iterations (int): Number of optimization iterations for survival time assignment. Default is 10,000.
    - early_stopping_threshold (float): Threshold for early stopping based on loss improvement. Default is 1e-6.
    - verbose (bool): If True, prints progress and debug information. Default is True.

    Returns:
    - optimal_weights (numpy.ndarray): Array of calculated optimal weights for the modalities per time bin (shape: N*20).
    """
    # Helper function for normalization
    def normalize_predictions(test_pred, mean_predictions, std_predictions):
        # Prevent division by zero
        std_predictions = torch.where(std_predictions == 0, torch.ones_like(std_predictions), std_predictions)
        test_pred_mean = torch.mean(test_pred, dim=0)
        test_pred_std = torch.std(test_pred, dim=0)
        test_pred_std = torch.where(test_pred_std == 0, torch.ones_like(test_pred_std), test_pred_std)
        standardized_test_pred = (test_pred - test_pred_mean) / test_pred_std
        normalized_test_pred = standardized_test_pred * std_predictions + mean_predictions
        return normalized_test_pred

    if verbose:
        print("Starting to compute mean and standard deviation from training data...")

    ### Step 1: Compute Mean and Std from Training Data ###
    train_loader = DataLoader(dataset=train_data, batch_size=len(train_data), shuffle=False)
    for batch in train_loader:
        all_predictions = []
        for modality in modalities_order:
            x = batch[modality].to(device)  # Shape: [num_cases, 20]
            all_predictions.append(x)
        # Stack all modality predictions along the modality dimension
        all_predictions = torch.stack(all_predictions, dim=0)  # Shape: [num_modalities, num_cases, 20]
        # Compute the mean and standard deviation across modalities and cases for each time bin
        mean_predictions = torch.mean(all_predictions, dim=(0, 1))  # Shape: [20,]
        std_predictions = torch.std(all_predictions, dim=(0, 1))    # Shape: [20,]
        # Handle cases where std is zero to avoid division errors
        std_predictions = torch.where(std_predictions == 0, torch.ones_like(std_predictions), std_predictions)
        mean_predictions = mean_predictions.to(device)
        std_predictions = std_predictions.to(device)
        break  # All data loaded

    best_mult = 1
    best_ibs = float('inf')

    ### Step 2: Find the Best Std Multiplier ###
    # Reinitialize the train_loader to iterate again
    train_loader = DataLoader(dataset=train_data, batch_size=len(train_data), shuffle=False)

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

            # Combine modality predictions by averaging across modalities
            combined_pred = torch.mean(torch.stack(modality_predictions, dim=0), dim=0)  # Shape: [num_cases, 20]

            # Calculate hazards and survival
            hazards = torch.sigmoid(combined_pred)  # Shape: [num_cases, 20]
            survival = torch.cumprod(1 - hazards, dim=1)  # Shape: [num_cases, 20]

            dtype = np.dtype([('event', bool), ('time', float)])
            survival_train = np.array(list(zip((1 - censor).astype(bool), survtime)), dtype=dtype)
            estimate = survival.detach().cpu().numpy()  # Shape: [num_cases, 20]
            times = np.arange(1, 21) * (365 / 2)

            # Calculate Integrated Brier Score (IBS)
            ibs = integrated_brier_score(survival_train, survival_train, estimate, times)

            if ibs < best_ibs:
                best_ibs = ibs 
                best_mult = std_mult
                if verbose:
                    print(f"New best_mult found: {best_mult} with IBS: {best_ibs:.6f}")

    if verbose:
        print("Starting survival time simulation...")

    ### Step 3: Sample Survival Times from Training Data ###
    # Reinitialize the test_loader
    test_loader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False)

    # Sample survival times
    for batch in train_loader:
        train_survival_times = batch['survival_time'].numpy()
        break  # All data loaded in one batch

    test_set_size = len(test_data)
    sampled_survival_times = np.random.choice(train_survival_times, size=test_set_size, replace=True)

    ### Step 4: Collect Risk Scores and Predictions for Each Modality from Test Set ###
    for batch in test_loader:
        modality_risk_scores = []
        modality_preds = []
        for modality in modalities_order:
            x = batch[modality].squeeze(1).to(device)  # Shape: [num_cases, 20]
            x_normalized = normalize_predictions(x, mean_predictions, std_predictions * best_mult).to(device)  # [num_cases, 20]
            # Calculate hazards and survival
            hazards = torch.sigmoid(x_normalized)  # Shape: [num_cases, 20]
            survival = torch.cumprod(1 - hazards, dim=1)  # Shape: [num_cases, 20]
            # Compute risk scores as per-sample values
            risk_scores = (1 - torch.sum(survival, dim=1)).detach().cpu().numpy()  # Shape: [num_cases,]
            modality_risk_scores.append(risk_scores)  # List of [num_cases,] arrays
            modality_preds.append(x_normalized)  # List of [num_cases, 20] tensors

        # Collect censoring information and event times
        censor = batch['censor'].detach().cpu().numpy()
        break  # All data loaded in one batch

    # Stack modality_preds into a 3D NumPy array: [num_cases, num_modalities, 20]
    modality_preds_tensor = torch.stack(modality_preds, dim=1)  # Shape: [num_cases, num_modalities, 20]
    modality_preds_array = modality_preds_tensor.detach().cpu().numpy()  # Shape: [num_cases, num_modalities, 20]

    # Stack modality_risk_scores into a 2D NumPy array: [num_cases, num_modalities]
    modality_risk_scores_array = np.stack(modality_risk_scores, axis=1)  # Shape: [num_cases, num_modalities]

    ### Step 5: Initialize Survival Time Assignment Randomly ###
    initial_permutation = np.random.permutation(test_set_size)
    assigned_survival_times = sampled_survival_times[initial_permutation]

    ### Step 6: Define the Objective Function ###
    def compute_total_loss(assigned_survival_times, modality_risk_scores, desired_cindices, censor):
        total_loss = 0
        for i, risk_scores in enumerate(modality_risk_scores.T):  # modality_risk_scores_array is [num_cases, num_modalities]
            achieved_cindex = concordance_index_censored(
                (1 - censor).astype(bool),
                assigned_survival_times,
                risk_scores,
                tied_tol=1e-08
            )[0]
            total_loss += (achieved_cindex - desired_cindices[i]) ** 2
        return total_loss

    ### Step 7: Optimize the Assignment of Survival Times ###
    best_permutation = initial_permutation.copy()
    best_loss = compute_total_loss(assigned_survival_times, modality_risk_scores_array, desired_cindices_N, censor)
    assigned_survival_times_best = assigned_survival_times.copy()

    if verbose:
        print(f"Initial total loss: {best_loss:.6f}")

    for iteration in tqdm(range(num_iterations), desc="Optimizing survival time assignments", disable=not verbose):
        # Swap two randomly selected survival times
        new_permutation = best_permutation.copy()
        idx1, idx2 = random.sample(range(test_set_size), 2)
        new_permutation[idx1], new_permutation[idx2] = new_permutation[idx2], new_permutation[idx1]
        new_assigned_survival_times = sampled_survival_times[new_permutation]

        # Compute the total loss for the new assignment
        new_loss = compute_total_loss(new_assigned_survival_times, modality_risk_scores_array, desired_cindices_N, censor)

        # Accept the new assignment if it improves the total loss
        if new_loss < best_loss:
            best_loss = new_loss
            best_permutation = new_permutation.copy()
            assigned_survival_times_best = new_assigned_survival_times.copy()
            if verbose:
                print(f"Iteration {iteration}: Improved loss to {best_loss:.6f}")

        # Early stopping criteria
        if best_loss < early_stopping_threshold:
            if verbose:
                print(f"Early stopping at iteration {iteration} with loss {best_loss:.6f}")
            break

    if verbose:
        print("Survival time simulation optimization complete.")

    ### Step 8: Update Test Data with Optimized Survival Times ###
    updated_survival_times = assigned_survival_times_best

    # Update the survival times in your test data
    for idx in range(len(test_data)):
        test_data[idx]['survival_time'] = updated_survival_times[idx]

    ### Recalculate c-indices Within the Script ###
    if verbose:
        print("Recalculating achieved c-indices for each modality...")

    achieved_cindices = []
    for i in range(len(desired_cindices_N)):
        risk_scores = modality_risk_scores_array[:, i]  # [num_cases,]
        achieved_cindex = concordance_index_censored(
            (1 - censor).astype(bool),
            assigned_survival_times_best,
            risk_scores,
            tied_tol=1e-08
        )[0]
        achieved_cindices.append(achieved_cindex)
        print(f"Modality {modalities_order[i]}: Desired c-index = {desired_cindices_N[i]:.4f}, Achieved c-index = {achieved_cindex:.4f}")

    # Calculate total error
    total_error = sum((achieved_cindex - desired_cindex) ** 2 for achieved_cindex, desired_cindex in zip(achieved_cindices, desired_cindices_N))
    print(f"Total Squared Error: {total_error:.6f}")

    ### Step 9: Compute Optimal Weights ###
    if verbose:
        print("Starting weight optimization using Differential Evolution...")

    # Define the objective function for weight optimization to maximize combined c-index
    def weight_objective_function(weights_flat, modality_preds, censor, updated_survival_times, num_modalities, num_time_bins):
        try:
            weights = np.array(weights_flat)
            # Reshape weights to (num_modalities, num_time_bins)
            weights = weights.reshape((num_modalities, num_time_bins))

            # Prevent negative weights
            if np.any(weights < 0):
                if verbose:
                    print("Negative weights detected.")
                return np.inf  # Penalize negative weights

            # Normalize weights per time bin to sum to 1 across modalities
            sum_weights = np.sum(weights, axis=0)  # Shape: [num_time_bins,]
            if np.any(sum_weights == 0):
                if verbose:
                    print("At least one time bin has a sum of weights equal to zero.")
                return np.inf  # Prevent division by zero

            normalized_weights = weights / sum_weights  # Broadcasting division

            # Combine modality predictions by weighted sum across modalities for each time bin
            # modality_preds shape: [num_cases, num_modalities, 20]
            combined_pred = np.sum(modality_preds * normalized_weights[np.newaxis, :, :], axis=1)  # Shape: [num_cases, 20]

            # Compute hazards using sigmoid
            hazards = expit(combined_pred)  # Shape: [num_cases, 20]

            # Compute survival probabilities
            survival = np.cumprod(1 - hazards, axis=1)  # Shape: [num_cases, 20]

            # Compute risk scores
            risk_scores = -np.sum(survival, axis=1).flatten()  # Ensure it's 1D

            # Check for NaN or Inf in risk_scores
            if np.isnan(risk_scores).any() or np.isinf(risk_scores).any():
                if verbose:
                    print("Combined risk scores contain NaN or Inf.")
                return np.inf

            # Compute c-index
            achieved_cindex = concordance_index_censored(
                (1 - censor).astype(bool),
                updated_survival_times,
                risk_scores,
                tied_tol=1e-08
            )[0]

            # Check if c-index is finite
            if not np.isfinite(achieved_cindex):
                if verbose:
                    print(f"Computed c-index is not finite: {achieved_cindex}")
                return np.inf

            # Since we want to maximize c-index, return negative c-index for minimization
            return -achieved_cindex

        except Exception as e:
            if verbose:
                print(f"Exception in objective function: {e}")
            return np.inf

    # modality_preds_array is [num_cases, num_modalities, 20]
    num_modalities = len(modalities_order)
    num_time_bins = 20

    # Define bounds for weights: between 0 and 1 for each modality per time bin
    bounds = [(0, 1) for _ in range(num_modalities * num_time_bins)]

    # Run Differential Evolution optimization
    result = differential_evolution(
        func=weight_objective_function,
        bounds=bounds,
        args=(modality_preds_array, censor, updated_survival_times, num_modalities, num_time_bins),
        strategy='best1bin',
        maxiter=200,
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=verbose,
        polish=True,
        init='latinhypercube'
    )

    optimal_weights_flat = result.x
    final_cindex = -result.fun if result.fun != np.inf else None

    # Reshape weights to (num_modalities, num_time_bins)
    optimal_weights = optimal_weights_flat.reshape((num_modalities, num_time_bins))

    if verbose:
        print("\nOptimal weights found (per modality per time bin):")
        for m in range(num_modalities):
            print(f"  {modalities_order[m]} weights: {optimal_weights[m]}")
        if final_cindex is not None:
            print(f"Final Combined c-index: {final_cindex:.4f}")
        else:
            print("Final c-index could not be computed due to infinite loss.")

    return optimal_weights
