import numpy as np
from numpy.polynomial.chebyshev import chebval
import re

# --- Helper functions for creating the base data ---

def create_chebyshev_basis_function(E, E0, n):
    """Generates the n-th Chebyshev basis function B_n(E) as defined in Eq. (38)."""
    def exponential_map(energy):
        return 1 - 2 * np.exp(-energy)

    x_E = exponential_map(E)
    x_E0 = exponential_map(E0)

    coeffs = np.zeros(n + 1)
    coeffs[n] = 1
    
    Tn_x_E = chebval(x_E, coeffs)
    Tn_x_E0 = chebval(x_E0, coeffs)

    return Tn_x_E - Tn_x_E0

def generate_spectral_parameters(Nb, seed=None):
    """Generates the random parameters (E0, coefficients) for one spectral density."""
    if seed is not None:
        np.random.seed(seed)
        
    E0 = np.random.uniform(low=0.2, high=1.3)
    epsilon = 1e-7
    r_n = np.random.uniform(low=-1.0, high=1.0, size=Nb)
    
    coeffs = np.zeros(Nb)
    if Nb > 0:
        coeffs[0] = r_n[0]
    if Nb > 1:
        n_values = np.arange(1, Nb)
        denominators = np.power(n_values, 1 + epsilon)
        coeffs[1:] = r_n[1:] / denominators
        
    return E0, coeffs

def construct_spectral_density(energy_grid, E0, coeffs):
    """Constructs the unsmeared spectral density rho(E) as defined in Eq. (39)."""
    Nb = len(coeffs)
    rho_E = np.zeros_like(energy_grid)
    for n in range(Nb):
        basis_function = create_chebyshev_basis_function(energy_grid, E0, n)
        rho_E += coeffs[n] * basis_function
    rho_E[energy_grid < E0] = 0.0
    return rho_E

# --- Helper functions for noise and standardization ---

def generate_noisy_correlator(clean_correlator, cov_matrix, mock_Clatt_a):
    """Generates one noisy correlator from a clean one using Eq. (44)."""
    C_a = clean_correlator[0] # Note: Paper uses C(a), but C(t=0) is more stable for scaling.
    scaling_factor = (C_a / mock_Clatt_a)**2 if mock_Clatt_a != 0 else 1.0
    scaled_cov = cov_matrix * scaling_factor
    noisy_corr = np.random.multivariate_normal(mean=clean_correlator, cov=scaled_cov)
    return noisy_corr

def standardize_dataset(clean_correlators, noisy_correlators):
    """Standardizes the noisy correlators based on the clean ones (Eqs. 45-47)."""
    mu = np.mean(clean_correlators, axis=0)
    gamma = np.std(clean_correlators, axis=0)
    gamma[gamma < 1e-15] = 1e-15
    standardized_noisy = (noisy_correlators - mu) / gamma
    return standardized_noisy, mu, gamma

def compute_smeared_spectral_density(integration_energy_grid, rho_E, sigma):
    """Vectorized version of compute_smeared_spectral_density."""
    MUON_MASS = 0.10566
    E_min, E_max, N_E = MUON_MASS, 24 * MUON_MASS, 47
    output_energy_grid = np.linspace(E_min, E_max, N_E)
    
    # Use broadcasting for vectorization
    E_vals = output_energy_grid.reshape(-1, 1) # Shape: (47, 1)
    omega_vals = integration_energy_grid.reshape(1, -1) # Shape: (1, n_integration_points)
    
    norm = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    exponent = -((E_vals - omega_vals)**2) / (2 * sigma**2)
    kernel = norm * np.exp(exponent)
    
    integrand = kernel * rho_E
    
    smeared_rho_vector = np.trapz(integrand, integration_energy_grid, axis=1)
    
    return output_energy_grid, smeared_rho_vector

def read_covariance_from_dat(filepath, time_extent):
    """
    Reads a covariance matrix from a .dat file with a specific 'COV[i][j]' format.

    This function reads the file line by line, finds the covariance data block,
    and parses it to reconstruct the full matrix.

    Args:
        filepath (str): The path to the .dat file.
        time_extent (int): The dimension of the square covariance matrix (e.g., 64 or 65).

    Returns:
        np.ndarray: The reconstructed (time_extent x time_extent) covariance matrix,
                    or None if the data cannot be found.
    """
    print(f"Attempting to read covariance matrix from '{filepath}'...")
    
    # Initialize an empty matrix to hold the covariance values
    cov_matrix = np.zeros((time_extent, time_extent))
    
    # Regular expression to robustly find and parse lines like "COV[ 0][1]   +1.275e-12"
    # It captures the two integer indices and the floating point number.
    cov_pattern = re.compile(r"COV\[\s*(\d+)]\[\s*(\d+)]\s+([+\-]\d\.\d+e[+\-]\d+)")
    
    found_cov_data = False
    try:
        with open(filepath, 'r') as f:
            for line in f:
                match = cov_pattern.search(line)
                if match:
                    found_cov_data = True
                    # Extract the indices (i, j) and the value
                    i = int(match.group(1))
                    j = int(match.group(2))
                    value = float(match.group(3))
                    
                    # Populate the matrix, ensuring bounds are not exceeded
                    if i < time_extent and j < time_extent:
                        cov_matrix[i, j] = value
                        # The matrix is symmetric, so fill the other half
                        cov_matrix[j, i] = value
        
        if not found_cov_data:
            print("Warning: No lines matching the 'COV[i][j]' format were found.")
            return None
            
        print("Successfully read and reconstructed the covariance matrix!")
        print(f"  - Matrix shape: {cov_matrix.shape}")
        return cov_matrix

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

        return None

def read_correlator_data(filepath, time_extent):
    """
    Reads correlator data from a text-based .dat file with multiple columns.

    This function is now adapted to handle files where the first column is text
    and the subsequent columns are the mean correlator and its error. It also
    only reads the expected number of rows.

    Args:
        filepath (str): The path to the .dat file.
        time_extent (int): The number of time slices to read (e.g., 64).

    Returns:
        np.ndarray: A 2D NumPy array with shape (time_extent, 2),
                    containing the mean correlator and its error,
                    or None if the file cannot be read.
    """
    try:
        # Use numpy.loadtxt to read the data.
        # It automatically handles whitespace and scientific notation.
        print(f"Attempting to read data from '{filepath}'...")
        
        # `usecols=(2, 3)` tells numpy to read the third and fourth columns.
        # --- THE FIX IS HERE ---
        # `max_rows=time_extent` tells numpy to stop reading after 64 lines,
        # ignoring any summary text or malformed lines at the end of the file.
        data = np.loadtxt(filepath, comments='#', usecols=(2, 3), max_rows=time_extent)
        
        # --- Verification ---
        if data.shape != (time_extent, 2):
            print(f"Error: Expected data to have shape ({time_extent}, 2) but got {data.shape}.")
            return None
        
        n_time_slices, _ = data.shape
        print(f"Successfully read the data!")
        print(f"  - Found {n_time_slices} time slices (rows).")
        print(f"  - Extracted 2 columns (Correlator and Error).")
        
        return data

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


def compute_correlator_vectorized(energy_grid, rho_E, time_extent):
    """
    Vectorized correlator calculation using the kernel: C(t) = integral[ rho(w) * exp(-w*t) ]
    """
    t_values = np.arange(time_extent).reshape(-1, 1)
    omega = energy_grid.reshape(1, -1)
    exponentials = np.exp(-t_values * omega)
    integrand = exponentials * rho_E 
    correlator_vector = np.trapz(integrand, energy_grid, axis=1)
    return correlator_vector

def compute_momentum_vectorized(energy_grid, rho_E, alpha, time_extent):
    """
    Vectorized correlator momentum calculation using the kernel: 
    D(alpha, tau) = integral[ rho(w) * w^alpha * exp(-w*tau) ]
    """
    tau_values = np.arange(time_extent).reshape(-1, 1)
    omega = energy_grid.reshape(1, -1)
    if alpha == 0: E_pow_alpha = 1.0
    else: safe_E = np.maximum(omega, 0); E_pow_alpha = np.power(safe_E, alpha)
    exponentials = np.exp(-tau_values * omega)
    integrand = E_pow_alpha * exponentials * rho_E
    momentum_vector = np.trapz(integrand, energy_grid, axis=1)
    return momentum_vector

def generate_single_clean_sample(args):
    """
    Worker function for parallel generation of one clean sample.
    """
    nb, seed, time_extent = args
    integration_grid = np.linspace(0.0, 10.0, 4000)
    E0, coeffs = generate_spectral_parameters(nb, seed=seed)
    rho = construct_spectral_density(integration_grid, E0, coeffs)
    clean_c = compute_correlator_vectorized(integration_grid, rho, time_extent)
    momentum_vec_a0 = compute_momentum_vectorized(integration_grid, rho, 0, time_extent)
    momentum_vec_a1 = compute_momentum_vectorized(integration_grid, rho, 1, time_extent)
    return clean_c, momentum_vec_a0, momentum_vec_a1