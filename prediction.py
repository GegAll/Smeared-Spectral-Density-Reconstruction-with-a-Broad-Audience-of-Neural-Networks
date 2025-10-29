import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
# import matplotlib.pyplot as plt # Plotting removed for batch processing

# --- Import your project-specific files ---
# IMPORTANT: Use the models trained for the smeared density task
from pytorch_models_t32 import ArcS_t32, ArcL_t32 
# Utility to generate the mock data
from generate_delta_mock import create_delta_plus_continuum_mock_data 
# Utility to load noise characteristics
# Assuming H5 format for real data based on previous steps
# from read_correlator_dat import read_correlator_data 
# from read_covariance_dat import read_covariance_from_dat

# --- CONFIGURATION ---

# --- 1. Define the trained ensembles you want to use ---
training_jobs = [
    {"label": "Nmax", "architecture": ArcL_t32, "nb": 128, "nrho": 400000},
    {"label": "Nref_b", "architecture": ArcL_t32, "nb": 16, "nrho": 400000},
    {"label": "Nref_n", "architecture": ArcS_t32, "nb": 128, "nrho": 400000},
    {"label": "Nref_rho", "architecture": ArcL_t32, "nb": 128, "nrho": 25000},
]
NR_REPLICAS = 5

# --- 2. Paths to your data and models ---
REAL_DATA_FILE = "data/octet-psq0_nb1000_bin2.h5" 
# --- CHANGE 1: Use the correct dataset dir for smeared density training ---
TRAINING_DATA_DIR = "training_datasets_T32" # Assuming this dir holds data used for sigma models
# --- CHANGE 2: Define base paths for models and output ---
BASE_TRAINED_MODELS_DIR = "trained_models_sigma" # Will append 44 or 63
BASE_OUTPUT_DIR = "mock_predictions_sigma"       # Will append 44 or 63

# --- 3. Prediction parameters ---
# --- CHANGE 3: Define list of sigmas to predict ---
SIGMAS_TO_PREDICT = [0.44, 0.63] 
N_BOOTSTRAP_SAMPLES = 800
PEAK_MASS = 0.8
PEAK_AMPLITUDE = 1.5

# --- Utility Functions (unchanged) ---
def predict_ensemble(models, bootstrap_samples, mu, gamma, device):
    n_bootstrap, n_replicas = bootstrap_samples.shape[0], len(models)
    # --- IMPORTANT: Ensure output_features matches the smeared density model (47) ---
    output_features = 47 
    # Check the actual model output if unsure:
    # output_features = models[0].fully_connected_layers[-1].out_features 
    all_predictions = np.zeros((n_bootstrap, n_replicas, output_features))
    for model in models: model.eval()
    with torch.no_grad():
        for i in tqdm(range(n_bootstrap), desc="  Predicting on samples"):
            standardized_sample = (bootstrap_samples[i] - mu) / gamma
            input_tensor = torch.from_numpy(standardized_sample).float().unsqueeze(0).unsqueeze(0).to(device)
            for r, model in enumerate(models):
                prediction = model(input_tensor)
                all_predictions[i, r, :] = prediction.cpu().numpy()
    return all_predictions

# --- Main Execution ---
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Step 1: Load the real correlator mean and covariance from the H5 file ---
    print(f"--- Step 1: Loading Real Correlator Data from '{REAL_DATA_FILE}' ---")
    try:
        with h5py.File(REAL_DATA_FILE, 'r') as f:
            if 'octet(0)' in f: data = f['octet(0)']
            else: data = f
            mean_correlator = data['mean'][:]
            real_cov_matrix = data['cov'][:]
    except Exception as e:
        raise IOError(f"Could not read 'mean' or 'cov' from {REAL_DATA_FILE}. Error: {e}")
    
    TIME_EXTENT = len(mean_correlator)
    print(f"  Data loaded successfully. Detected TIME_EXTENT = {TIME_EXTENT}")

    # --- Step 2: Generate the clean "ground truth" mock data ---
    print("\n--- Step 2: Generating clean delta+continuum mock data ---")
    clean_mock_correlator, true_smeared_rhos, unsmeared_continuum, _, energy_grid_smeared, integration_grid = \
        create_delta_plus_continuum_mock_data(TIME_EXTENT, PEAK_MASS, PEAK_AMPLITUDE)
    
    # --- Step 3: Generate Bootstrap Samples for the mock correlator ---
    print(f"\n--- Step 3: Generating {N_BOOTSTRAP_SAMPLES} bootstrap samples from mock data ---")
    bootstrap_samples = np.random.multivariate_normal(clean_mock_correlator, real_cov_matrix, size=N_BOOTSTRAP_SAMPLES)

    # --- CHANGE 4: Loop over sigma values ---
    for sigma in SIGMAS_TO_PREDICT:
        print(f"\n{'#'*70}\n### PROCESSING FOR SIGMA = {sigma} ###\n{'#'*70}")
        
        # Construct sigma-specific paths
        current_model_dir_base = f"{BASE_TRAINED_MODELS_DIR}{int(sigma*100)}"
        current_output_dir = f"{BASE_OUTPUT_DIR}{int(sigma*100)}"
        os.makedirs(current_output_dir, exist_ok=True)

        # --- Loop over each trained job ensemble ---
        for job in training_jobs:
            job_label, architecture, nb, nrho = job.values()
            print(f"\n{'='*60}\nProcessing with ensemble: '{job_label}' for sigma={sigma}\n{'='*60}")

            # --- Step 4: Load Standardization Vectors for this job ---
            # Mu/Gamma come from the original training dataset (sigma-independent)
            training_set_info_file = os.path.join(TRAINING_DATA_DIR, f"training_data_Nb{nb}_Nrho{nrho}.hdf5") 
            try:
                with h5py.File(training_set_info_file, 'r') as f:
                    mu = f['mu_vector'][:]
                    gamma = f['gamma_vector'][:]
                print(f"  Loaded mu and gamma from {os.path.basename(training_set_info_file)}")
            except Exception as e:
                print(f"  ERROR: Could not load mu/gamma from {training_set_info_file}. Skipping. Error: {e}")
                continue

            # --- Step 5: Load the Trained Model Ensemble for this job AND sigma ---
            ensemble_models = []
            model_dir = os.path.join(current_model_dir_base, job_label) # Use sigma-specific model dir
            print(f"  Loading {NR_REPLICAS} trained models from '{model_dir}'...")
            try:
                for r in range(1, NR_REPLICAS + 1):
                    model_path = os.path.join(model_dir, f"replica_{r}_best.pth")
                    # --- IMPORTANT: Ensure you import the correct model definitions (pytorch_models_t32.py) ---
                    model = architecture() # architecture comes from training_jobs dict
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.to(device)
                    ensemble_models.append(model)
            except FileNotFoundError as e:
                print(f"  ERROR: Could not load models for job '{job_label}', sigma={sigma}. Skipping. Details: {e}")
                continue
            
            # --- Step 6: Run Prediction ---
            predictions = predict_ensemble(ensemble_models, bootstrap_samples, mu, gamma, device)
            
            # --- Step 7: Store Results ---
            output_file = os.path.join(current_output_dir, f"predictions_{job_label}_sigma{sigma}.hdf5")
            print(f"  Storing predictions in '{output_file}'...")
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('predictions', data=predictions)
                f.create_dataset('energy_grid_smeared', data=energy_grid_smeared)
                # Save the correct true smeared rho for this sigma
                f.create_dataset(f'true_smeared_rho', data=true_smeared_rhos[sigma])
                # Save unsmeared components (same for both sigmas)
                f.create_dataset('unsmeared_continuum', data=unsmeared_continuum)
                f.create_dataset('unsmeared_energy_grid', data=integration_grid)
                f.create_dataset('unsmeared_peak_mass', data=PEAK_MASS)
                f.create_dataset('unsmeared_peak_amplitude', data=PEAK_AMPLITUDE)
            print("  Done.")

    print("\nAll mock data predictions for all sigmas are complete!")