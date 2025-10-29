import os
import h5py
import numpy as np
from tqdm import tqdm
import re
from multiprocessing import Pool, cpu_count

# --- Import your data generation functions ---
from prepare_training_data_T12 import (
    generate_spectral_parameters,
    construct_spectral_density,
    generate_noisy_correlator,
    standardize_dataset,
    compute_smeared_spectral_density,
    read_covariance_from_dat,
    read_correlator_data
)

# --- ACCELERATED HELPER FUNCTIONS ---
def compute_correlator_vectorized(energy_grid, rho_E, time_extent):
    T_period = time_extent - 1
    t_values = np.arange(time_extent).reshape(-1, 1)
    omega = energy_grid.reshape(1, -1)
    factor = omega**2 / (12 * np.pi**2)
    exponentials = np.exp(-t_values * omega) + np.exp(-(T_period - t_values) * omega)
    integrand = factor * exponentials * rho_E
    correlator_vector = np.trapz(integrand, energy_grid, axis=1)
    return correlator_vector

def generate_single_clean_sample(args):
    nb, seed, time_extent, sigma_values = args
    integration_grid = np.linspace(0.0, 10.0, 4000)
    E0, coeffs = generate_spectral_parameters(nb, seed=seed)
    rho = construct_spectral_density(integration_grid, E0, coeffs)
    clean_c = compute_correlator_vectorized(integration_grid, rho, time_extent=time_extent)
    smeared_rhos = {}
    for sig in sigma_values:
        _, smeared_rho = compute_smeared_spectral_density(integration_grid, rho, sig)
        smeared_rhos[sig] = smeared_rho
    return clean_c, smeared_rhos

# --- CONFIGURATION ---
# --- CONFIGURATION ---
TIME_EXTENT = 32 # <-- The most important change

# Simulation parameters (ADAPTED for T=32)
NB_VALUES = [16, 32, 64, 128]
NRHO_VALUES = [25000, 50000, 100000, 200000, 400000]
REAL_DATA_FILE = "data/octet-psq0_nb1000_bin2.h5"
# The rest of the parameters can remain the same
SIGMA_VALUES = [0.44, 0.63]
NR_REPLICAS = 20
OUTPUT_DIR = "training_datasets_T32" 

# --- Main Script ---
if __name__ == '__main__':
    print("Loading real correlator data from .h5 file...")
    run00 = h5py.File(REAL_DATA_FILE, 'r')
    data = run00['octet(0)']
    
    real_cov_matrix = data['cov'][:]
    if real_cov_matrix is None: raise ValueError("Failed to load covariance matrix.")
    
    mean_correlator_data = data['mean'][:]
    if mean_correlator_data is None: raise ValueError("Failed to load mean correlator data.")
    
    real_C_latt_a = mean_correlator_data[1] 
    print(f"Successfully loaded real data: Covariance shape {real_cov_matrix.shape}, C(t=1) = {real_C_latt_a:.4e}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for nb in NB_VALUES:
        print(f"\n{'='*50}\nStarting generation for Nb = {nb}\n{'='*50}")
        
        max_nrho = max(NRHO_VALUES)
        num_cores = cpu_count()
        print(f"Generating {max_nrho} clean samples for Nb={nb} using {num_cores} cores...")
        
        tasks = [(nb, i, TIME_EXTENT, SIGMA_VALUES) for i in range(max_nrho)]
        results = []
        with Pool(processes=num_cores) as pool:
            results = list(tqdm(pool.imap(generate_single_clean_sample, tasks), total=max_nrho))

        # --- OPTIMIZATION 1: Use float32 to save space ---
        all_clean_correlators = np.array([res[0] for res in results], dtype=np.float32)
        all_clean_smeared_rhos = {sig: np.array([res[1][sig] for res in results], dtype=np.float32) for sig in SIGMA_VALUES}

        for nrho in NRHO_VALUES:
            print(f"\n--- Processing and saving dataset for (Nb={nb}, Nrho={nrho}) ---")
            
            clean_corrs_slice = all_clean_correlators[:nrho]
            
            print("Calculating standardization vectors (mu, gamma)...")
            mu = np.mean(clean_corrs_slice, axis=0, dtype=np.float32)
            gamma = np.std(clean_corrs_slice, axis=0, dtype=np.float32)
            gamma[gamma < 1e-9] = 1e-9 # Adjusted for float32 precision
            
            filename = f"training_data_Nb{nb}_Nrho{nrho}.hdf5"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            with h5py.File(filepath, 'w') as f:
                # Create datasets first, then fill them
                noisy_dset = f.create_dataset('standardized_noisy_correlators', shape=(nrho, NR_REPLICAS, TIME_EXTENT), dtype='f4') # 'f4' is float32
                
                print(f"Generating noisy replicas and saving to '{filepath}'...")
                for i in tqdm(range(nrho), desc=f"Injecting Noise & Saving (Nrho={nrho})"):
                    # --- OPTIMIZATION 2: Process and save one sample at a time ---
                    replicas_for_sample = np.zeros((NR_REPLICAS, TIME_EXTENT), dtype=np.float32)
                    for r in range(NR_REPLICAS):
                        noisy_c = generate_noisy_correlator(clean_corrs_slice[i], real_cov_matrix, real_C_latt_a)
                        replicas_for_sample[r, :] = (noisy_c - mu) / gamma
                    
                    # Write this chunk to the HDF5 file
                    noisy_dset[i, :, :] = replicas_for_sample

                # Save the other datasets (these are smaller and can be saved at once)
                for sig in SIGMA_VALUES:
                    smeared_slice = all_clean_smeared_rhos[sig][:nrho]
                    f.create_dataset(f'clean_smeared_rho_sigma_{sig}', data=smeared_slice)
                f.create_dataset('mu_vector', data=mu)
                f.create_dataset('gamma_vector', data=gamma)

            print(f"Successfully saved dataset for (Nb={nb}, Nrho={nrho}).")
            
    print("\nAll datasets have been generated!")