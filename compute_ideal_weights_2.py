import numpy as np
from scipy.stats import norm, kendalltau
from scipy.linalg import cholesky
from numpy.random import default_rng
import warnings
from scipy.optimize import differential_evolution

def compute_ideal_weights(
    num_modalities: int,
    desired_cindices: list,
    rho_between: np.ndarray,
    N: int,
    num_runs: int = 1,
    num_weights: int = 100
):
    """
    Generates synthetic data with specified Pearson correlations and c-indices,
    then computes ideal weights for combining risk scores using differential evolution.

    Parameters:
        num_modalities (int): Number of risk score modalities.
        desired_cindices (list or np.ndarray): Desired c-indices for each modality.
            Length must be equal to num_modalities.
        rho_between (np.ndarray): Desired Pearson correlation matrix between modalities.
            Shape must be (num_modalities, num_modalities). Must be symmetric with ones on the diagonal.
        N (int): Number of samples to generate.
        num_runs (int, optional): Number of runs (default is 1 for differential evolution).
        num_weights (int, optional): Not used in differential evolution but kept for compatibility.

    Returns:
        overall_weights (np.ndarray): Array of shape (num_runs, num_modalities) containing
            the optimal weights for each run.
        observed_rho (np.ndarray): Observed Pearson correlation matrix from the synthetic data.
        observed_cindices (np.ndarray): Observed c-indices for each modality.
    """

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Validate inputs
    desired_cindices = np.array(desired_cindices)
    if desired_cindices.shape[0] != num_modalities:
        raise ValueError("Length of desired_cindices must be equal to num_modalities.")

    if rho_between.shape != (num_modalities, num_modalities):
        raise ValueError("rho_between must be a square matrix with shape (num_modalities, num_modalities).")

    if not np.allclose(rho_between, rho_between.T):
        raise ValueError("rho_between must be a symmetric matrix.")

    if not np.allclose(np.diag(rho_between), np.ones(num_modalities)):
        raise ValueError("Diagonal elements of rho_between must be 1.")

    # Initialize variables
    weights = np.linspace(0, 1, num_weights)

    def spearman_to_pearson_corr(spearman_corr):
        """
        Converts a Spearman correlation matrix to a Pearson correlation matrix.

        Parameters:
            spearman_corr (np.ndarray): Spearman correlation matrix.

        Returns:
            rho_Pearson (np.ndarray): Pearson correlation matrix.
        """
        rho_Pearson = spearman_corr.copy()
        for i in range(spearman_corr.shape[0]):
            for j in range(spearman_corr.shape[1]):
                if i != j:
                    rho_S = spearman_corr[i, j]
                    rho_Pearson[i, j] = 2 * np.sin((np.pi / 6) * rho_S)
                else:
                    rho_Pearson[i, j] = 1.0
        return rho_Pearson

    def nearestSPD(A):
        """
        Finds the nearest positive-definite matrix to input A using Higham's algorithm.

        Parameters:
            A (np.ndarray): Input matrix.

        Returns:
            Ahat (np.ndarray): Nearest positive-definite matrix.
        """
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = np.dot(V.T * s, V)
        Ahat = (B + H) / 2
        Ahat = (Ahat + Ahat.T) / 2

        try:
            cholesky(Ahat)
        except np.linalg.LinAlgError:
            spacing = np.spacing(np.linalg.norm(A))
            I = np.eye(A.shape[0])
            k = 1
            while True:
                try:
                    Ahat += I * spacing
                    cholesky(Ahat)
                    break
                except np.linalg.LinAlgError:
                    spacing *= 2
                    k += 1
                    if k > 1000:
                        raise ValueError('Unable to make the matrix positive definite.')
        return Ahat

    def generate_synthetic_data_copula_N(N, num_modalities, rho_between, rho_surv):
        """
        Generates synthetic data using Gaussian copula with specified Pearson correlations.

        Parameters:
            N (int): Number of samples.
            num_modalities (int): Number of risk score modalities.
            rho_between (np.ndarray): Pearson correlation matrix between modalities.
            rho_surv (np.ndarray): Spearman correlations between modalities and survival times.

        Returns:
            risk_scores (np.ndarray): Generated risk scores (N x num_modalities).
            survival_times (np.ndarray): Generated survival times (N,).
        """
        total_vars = num_modalities + 1
        spearman_corr = np.eye(total_vars)
        spearman_corr[:num_modalities, :num_modalities] = rho_between
        spearman_corr[:num_modalities, -1] = rho_surv
        spearman_corr[-1, :num_modalities] = rho_surv

        # Convert Spearman to Pearson correlation
        gauss_copula_corr = spearman_to_pearson_corr(spearman_corr)
        np.fill_diagonal(gauss_copula_corr, 1)

        # Ensure positive definiteness
        try:
            cholesky(gauss_copula_corr)
        except np.linalg.LinAlgError:
            gauss_copula_corr = nearestSPD(gauss_copula_corr)

        # Generate samples
        rng = default_rng()
        Z = rng.multivariate_normal(np.zeros(total_vars), gauss_copula_corr, size=N)

        # Convert to uniform with clipping to avoid exact 0 or 1
        U = norm.cdf(Z)
        U = np.clip(U, 1e-10, 1 - 1e-10)

        # Transform to desired marginals
        risk_scores = norm.ppf(U[:, :num_modalities])
        lambda_param = 1  # Rate parameter for exponential
        survival_times = -np.log(1 - U[:, -1]) / lambda_param

        return risk_scores, survival_times

    def compute_c_index(risk_score, survival_time):
        """
        Computes the concordance index (c-index) between risk_score and survival_time.

        Parameters:
            risk_score (np.ndarray): Vector of risk scores.
            survival_time (np.ndarray): Vector of survival times.

        Returns:
            c_index (float): Concordance index.
        """
        tau, _ = kendalltau(risk_score, survival_time)
        if np.isnan(tau):
            return 0.5  # Default value if tau is undefined
        c_index = (tau + 1) / 2
        return c_index

    def find_spearman_for_cindex(desired_cindex, N, rho_between, risk_score_index, num_modalities):
        """
        Finds the Spearman correlation that results in the desired c-index
        between a specific risk score and survival times.

        Parameters:
            desired_cindex (float): Desired c-index.
            N (int): Number of samples.
            rho_between (np.ndarray): Pearson correlation matrix between modalities.
            risk_score_index (int): Index of the risk score to adjust.
            num_modalities (int): Total number of modalities.

        Returns:
            rho_S (float): Spearman correlation achieving the desired c-index.
        """
        tol = 1e-4
        max_iter = 200
        rho_S_low = 0.0
        rho_S_high = 0.9999

        iter_count = 0
        while iter_count < max_iter:
            iter_count += 1
            rho_S = (rho_S_low + rho_S_high) / 2

            # Set rho_surv vector with zero correlations except at risk_score_index
            rho_surv_temp = np.zeros(num_modalities)
            rho_surv_temp[risk_score_index] = rho_S

            # Generate data
            risk_scores_iter, survival_times_iter = generate_synthetic_data_copula_N(
                N, num_modalities, rho_between, rho_surv_temp
            )

            # Select the appropriate risk score
            risk_score = risk_scores_iter[:, risk_score_index]

            # Compute c-index
            c_idx = compute_c_index(risk_score, survival_times_iter)

            # Check convergence
            cidx_diff = c_idx - desired_cindex
            if abs(cidx_diff) < tol:
                break
            elif cidx_diff > 0:
                # Observed c-index is higher than desired; decrease rho_S
                rho_S_high = rho_S
            else:
                # Observed c-index is lower than desired; increase rho_S
                rho_S_low = rho_S

        if iter_count == max_iter:
            print(f'Warning: Maximum iterations reached while finding Spearman correlation for modality {risk_score_index + 1}.')

        return rho_S

    # Step 1: Find Spearman correlations for desired c-indices
    print('Finding Spearman correlations for desired c-indices...')
    rho_surv = np.zeros(num_modalities)
    for i in range(num_modalities):
        rho_surv[i] = find_spearman_for_cindex(
            desired_cindices[i], N, rho_between, i, num_modalities
        )

    # Step 2: Generate synthetic data
    risk_scores, survival_times = generate_synthetic_data_copula_N(
        N, num_modalities, rho_between, rho_surv
    )

    # Step 3: Verify Pearson correlations
    print('\nVerifying Pearson correlations...')
    observed_rho = np.corrcoef(risk_scores, rowvar=False)
    for i in range(num_modalities):
        for j in range(i + 1, num_modalities):
            desired = rho_between[i, j]
            observed = observed_rho[i, j]
            error = observed - desired
            print(f'Correlation between Risk Score {i + 1} and {j + 1}: '
                  f'Desired = {desired:.4f}, Observed = {observed:.4f}, '
                  f'Error = {error:.4f}')

    # Step 4: Verify c-indices
    print('\nVerifying c-indices...')
    observed_cindices = np.zeros(num_modalities)
    for i in range(num_modalities):
        c_idx = compute_c_index(risk_scores[:, i], survival_times)
        observed_cindices[i] = c_idx
        error_cindex = c_idx - desired_cindices[i]
        print(f'Risk Score {i + 1}: Desired c-index = {desired_cindices[i]:.4f}, '
              f'Observed c-index = {c_idx:.4f}, Error = {error_cindex:.4f}')

    # Step 5: Differential Evolution Grid Search to Find Optimal Weights
    print('\nStarting Differential Evolution Grid Search...')

    # Step 5a: Check synthetic data for NaN or Inf
    print("Checking synthetic data for NaN or Inf values...")
    print(f"Risk scores contain NaN: {np.isnan(risk_scores).any()}")
    print(f"Risk scores contain Inf: {np.isinf(risk_scores).any()}")
    print(f"Survival times contain NaN: {np.isnan(survival_times).any()}")
    print(f"Survival times contain Inf: {np.isinf(survival_times).any()}")

    # Define the objective function with enhanced debugging and normalization
    def objective_function(weights):
        try:
            weights = np.array(weights)

            # Prevent negative weights
            if np.any(weights < 0):
                print("Negative weights detected.")
                return np.inf  # Penalize negative weights

            # Normalize weights to sum to 1
            sum_weights = np.sum(weights)
            if sum_weights == 0:
                print("Sum of weights is zero.")
                return np.inf  # Prevent division by zero

            normalized_weights = weights / sum_weights

            # Combine risk scores
            combined_score = np.dot(risk_scores, normalized_weights)

            # Check for NaN or Inf in combined_score
            if np.isnan(combined_score).any() or np.isinf(combined_score).any():
                print("Combined score contains NaN or Inf.")
                return np.inf

            # Compute c-index
            c_idx = compute_c_index(combined_score, survival_times)

            # Check if c_idx is finite
            if not np.isfinite(c_idx):
                print(f"Computed c-index is not finite: {c_idx}")
                return np.inf

            # Return negative c-index to minimize
            return -c_idx

        except Exception as e:
            print(f"Exception in objective function: {e}")
            return np.inf

    # Define bounds for weights: between 0 and 1
    bounds = [(0, 1) for _ in range(num_modalities)]

    # Run differential evolution without constraints
    result = differential_evolution(
        objective_function,
        bounds,
        strategy='best1bin',
        maxiter=1000,
        popsize=50,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=None,
        callback=None,
        disp=True,
        polish=True,
        init='latinhypercube'
    )

    # Get the optimal weights
    optimal_weights = result.x
    # Compute the final c-index
    final_cindex = -result.fun

    # Print results
    print(f'\nOptimal weights:')
    for i in range(num_modalities):
        print(f'  Risk Score {i + 1}: Weight = {optimal_weights[i]:.6f}')
    print(f'Final Combined c-index: {final_cindex:.4f}')

    # Prepare the outputs
    overall_weights = optimal_weights.reshape(1, num_modalities)
    observed_cindices = np.array([compute_c_index(risk_scores[:, i], survival_times) for i in range(num_modalities)])

    return overall_weights, observed_rho, observed_cindices
