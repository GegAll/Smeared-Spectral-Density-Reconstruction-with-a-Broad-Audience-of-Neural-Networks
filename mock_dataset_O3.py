import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
# import matplotlib.pyplot as plt # Keep commented unless debugging interactively

# --- Import project files ---
# IMPORTANT: Use models trained for SMEARED DENSITY task (output size 47)
from pytorch_models_t32 import ArcS_t32, ArcL_t32
# --- Import the CORRECT utility functions for the sigma-model task ---
# These define the kernels used during the ORIGINAL training
from sigma_model_utils import (
    compute_correlator_vectorized_sigma,
    compute_smeared_spectral_density
)
# -----------------------------------------------------------------------

# --- CONFIGURATION ---
training_jobs = [
    {"label": "Nmax", "architecture": ArcL_t32, "nb": 128, "nrho": 400000},
    {"label": "Nref_b", "architecture": ArcL_t32, "nb": 16, "nrho": 400000},
    {"label": "Nref_n", "architecture": ArcS_t32, "nb": 128, "nrho": 400000},
    {"label": "Nref_rho", "architecture": ArcL_t32, "nb": 128, "nrho": 25000},
]
NR_REPLICAS = 5

REAL_DATA_FILE = "data/octet-psq0_nb1000_bin2.h5"
# Point to the dataset dir used for the ORIGINAL sigma model training
TRAINING_DATA_DIR = "training_datasets_T32"
# Base dir where sigma-specific model folders reside
TRAINED_MODELS_DIR_BASE = "trained_models_sigma"
# Where to save the output predictions for this O3 test
OUTPUT_DIR = "O3_mock_predictions_sigma"

N_BOOTSTRAP_SAMPLES = 800

# O(3) Model Parameters
O3_ETH = 1.0 # GeV (Choose one Eth for this test)

# Sigmas to predict
SIGMAS_TO_PREDICT = [0.44, 0.63]

# --- O(3) Model Spectral Density Function ---
def construct_O3_spectral_density(energy_grid, Eth):
    """Calculates the O(3) spectral density from Eq. 63."""
    rho = np.zeros_like(energy_grid)
    mask = energy_grid > Eth
    E = energy_grid[mask]
    # Add small epsilon to avoid potential log(1) issues if E==Eth exactly
    theta = 2 * np.arccosh(np.maximum(E / Eth, 1.0 + 1e-15))
    safe_theta_sq = np.maximum(theta**2, 1e-15)
    factor1 = (3 * np.pi**3) / (4 * safe_theta_sq)
    factor2 = (theta**2 + np.pi**2) / (theta**2 + 4 * np.pi**2)
    factor3 = np.tanh(theta / 2)**3
    rho[mask] = factor1 * factor2 * factor3
    # Ensure rho is finite (can sometimes get inf near threshold due to numerics)
    rho[~np.isfinite(rho)] = 0.0
    return rho

# --- Utility Function (Prediction Loop - 47 outputs) ---
def predict_ensemble(models, bootstrap_samples, mu, gamma, device):
    """Passes all bootstrap samples through all models in the ensemble."""
    n_bootstrap, n_replicas = bootstrap_samples.shape[0], len(models)
    output_features = 47 # Smeared density has 47 points
    all_predictions = np.zeros((n_bootstrap, n_replicas, output_features))
    for model in models: model.eval()
    with torch.no_grad():
        for i in tqdm(range(n_bootstrap), desc="  Predicting on samples"):
            standardized_sample = (bootstrap_samples[i] - mu) / gamma
            input_tensor = torch.from_numpy(standardized_sample).float().unsqueeze(0).unsqueeze(0).to(device)
            for r, model in enumerate(models):
                prediction = model(input_tensor)
                if prediction.shape[-1] != output_features:
                     raise ValueError(f"Model output size {prediction.shape[-1]} != expected {output_features}")
                all_predictions[i, r, :] = prediction.cpu().numpy()
    return all_predictions

# --- Main Execution ---
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Step 1: Load real noise characteristics ---
    print(f"--- Step 1: Loading Real Correlator Data from '{REAL_DATA_FILE}' ---")
    try:
        with h5py.File(REAL_DATA_FILE, 'r') as f:
            if 'octet(0)' in f: data = f['octet(0)']
            else: data = f
            mean_correlator_real = data['mean'][:]
            real_cov_matrix = data['cov'][:]
    except Exception as e:
        raise IOError(f"Could not read 'mean' or 'cov' from {REAL_DATA_FILE}. Error: {e}")

    TIME_EXTENT = len(mean_correlator_real)
    print(f"  Noise data loaded. TIME_EXTENT = {TIME_EXTENT}")

    # --- Step 2: Generate clean "ground truth" O(3) data ---
    print(f"\n--- Step 2: Generating clean O(3) mock data (Eth={O3_ETH} GeV) ---")
    integration_grid = np.linspace(0.0, 10.0, 4000) # Fine grid for integration
    rho_o3 = construct_O3_spectral_density(integration_grid, O3_ETH)

    # --- THIS IS THE CRUCIAL PART ---
    # Calculate the clean correlator using the SIGMA-MODEL KERNEL
    clean_mock_correlator = compute_correlator_vectorized_sigma(integration_grid, rho_o3, TIME_EXTENT)
    # -------------------------------

    # Calculate the true smeared spectral densities (ground truth targets)
    true_smeared_rhos = {}
    energy_grid_smeared = None
    for sigma_val in SIGMAS_TO_PREDICT:
        grid, rho_s = compute_smeared_spectral_density(integration_grid, rho_o3, sigma_val)
        true_smeared_rhos[sigma_val] = rho_s
        if energy_grid_smeared is None:
            energy_grid_smeared = grid # Store the grid
    print("  Done.")

    # --- Step 3: Generate Bootstrap Samples ---
    print(f"\n--- Step 3: Generating {N_BOOTSTRAP_SAMPLES} bootstrap samples ---")
    real_C_latt_a_for_scaling = mean_correlator_real[1] # C(t=1) for scaling
    bootstrap_samples = np.random.multivariate_normal(clean_mock_correlator, real_cov_matrix, size=N_BOOTSTRAP_SAMPLES)
    print(f"  Bootstrap samples shape: {bootstrap_samples.shape}")

    # --- Loop over sigma = 0.44 and sigma = 0.63 ---
    for sigma in SIGMAS_TO_PREDICT:
        print(f"\n{'#'*70}\n### PREDICTING SMEARED DENSITY FOR SIGMA = {sigma} ###\n{'#'*70}")

        true_target_vector = true_smeared_rhos[sigma]
        current_model_dir_base = f"{TRAINED_MODELS_DIR_BASE}{int(sigma*100)}" # e.g., trained_models_sigma44

        # --- Loop over each trained job ensemble ---
        for job in training_jobs:
            job_label, architecture, nb, nrho = job.values()
            print(f"\n{'='*60}\nProcessing with ensemble: '{job_label}' for sigma={sigma}\n{'='*60}")

            # --- Step 4: Load Standardization Vectors ---
            # Mu/Gamma come from the original smeared density training dataset
            training_set_info_file = os.path.join(TRAINING_DATA_DIR, f"training_data_Nb{nb}_Nrho{nrho}.hdf5")
            try:
                with h5py.File(training_set_info_file, 'r') as f:
                    mu = f['mu_vector'][:]
                    gamma = f['gamma_vector'][:]
            except Exception as e:
                print(f"  ERROR: Could not load mu/gamma for job '{job_label}'. Skipping. Error: {e}")
                continue

            # --- Step 5: Load the Trained Model Ensemble ---
            ensemble_models = []
            model_dir = os.path.join(current_model_dir_base, job_label)
            print(f"  Loading {NR_REPLICAS} trained models from '{model_dir}'...")
            try:
                for r in range(1, NR_REPLICAS + 1):
                    model_path = os.path.join(model_dir, f"replica_{r}_best.pth")
                    # Use the smeared density architecture definitions (pytorch_models_t32.py)
                    model = architecture()
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.to(device)
                    ensemble_models.append(model)
            except FileNotFoundError as e:
                print(f"  ERROR: Could not load models for job '{job_label}', sigma={sigma}. Skipping. Details: {e}")
                continue

            # --- Step 6: Run Prediction ---
            predictions = predict_ensemble(ensemble_models, bootstrap_samples, mu, gamma, device)

            # --- Step 7: Store Results ---
            output_file = os.path.join(OUTPUT_DIR, f"O3_predictions_{job_label}_sigma{sigma}.hdf5")
            print(f"  Storing predictions in '{output_file}'...")
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('predictions', data=predictions)
                f.create_dataset('true_smeared_rho', data=true_target_vector)
                f.create_dataset('energy_grid_smeared', data=energy_grid_smeared)
                # Store the unsmeared spectral info for reference
                f.create_dataset('unsmeared_rho_o3', data=rho_o3)
                f.create_dataset('unsmeared_energy_grid', data=integration_grid)
                f.attrs['eth'] = O3_ETH # Store O3 parameter
            print("  Done.")

    print("\nAll O(3) mock data predictions for smeared densities are complete!")