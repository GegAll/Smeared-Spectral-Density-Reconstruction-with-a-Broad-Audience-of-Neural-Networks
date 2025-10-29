import numpy as np
from scipy.special import erf # Needed for smearing, though not used here directly
import h5py # For completeness

# --- Functions from the first paper (Buzzicotti et al.) ---

def compute_correlator_vectorized_sigma(energy_grid, rho_E, time_extent):
    """
    Vectorized correlator calculation using the kernel from Eq. 32 
    of the Buzzicotti paper (for smeared density models).
    Kernel: (w^2 / 12pi^2) * [exp(-wt) + exp(-w(T-t))]
    """
    # T_period should be the total number of points, e.g., 32,
    # so (T-t) becomes (31-t).
    T_period = time_extent - 1 
    t_values = np.arange(time_extent).reshape(-1, 1) # Shape (time_extent, 1)
    omega = energy_grid.reshape(1, -1)              # Shape (1, n_energies)
    
    factor = omega**2 / (12 * np.pi**2)
    exponentials = np.exp(-t_values * omega) + np.exp(-(T_period - t_values) * omega)
    
    integrand = factor * exponentials * rho_E 
    
    correlator_vector = np.trapz(integrand, energy_grid, axis=1)
    return correlator_vector

def compute_smeared_spectral_density(integration_energy_grid, rho_E, sigma):
    """
    Computes the smeared spectral density vector rho_hat(E) from rho(E).
    (This function is from your prepare_training_data_T12.py)
    """
    MUON_MASS = 0.10566  # GeV
    E_min = MUON_MASS
    E_max = 24 * MUON_MASS
    N_E = 47  # Number of points
    output_energy_grid = np.linspace(E_min, E_max, N_E)
    smeared_rho_vector = np.zeros(N_E)

    def gaussian_kernel(E, omega, s):
        norm = 1.0 / (np.sqrt(2 * np.pi) * s)
        exponent = -((E - omega)**2) / (2 * s**2)
        return norm * np.exp(exponent)

    # Vectorized calculation
    E_vec = output_energy_grid.reshape(-1, 1) # Shape (47, 1)
    omega_vec = integration_energy_grid.reshape(1, -1) # Shape (1, 4000)
    
    kernel_matrix = gaussian_kernel(E_vec, omega_vec, sigma) # Shape (47, 4000)
    integrand = kernel_matrix * rho_E # Shape (47, 4000)
    
    smeared_rho_vector = np.trapz(integrand, integration_energy_grid, axis=1)
        
    return output_energy_grid, smeared_rho_vector