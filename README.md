# Smeared Spectral Density Reconstruction with a Broad Audience of Neural Networks

This repository contains the code to reproduce the "Broad Audience" machine learning method for extracting smeared spectral densities $\hat{\rho}(E, \sigma)$ from noisy Euclidean correlator data $C(t)$.This work is based on the methodology presented in the paper "Teaching to extract spectral densities from lattice correlators to a broad audience of learning-machines" (**Buzzicotti et al.**). The project is built to be run on an HPC cluster using the Slurm scheduler.

## Methodology Overview
The core idea is to train neural networks to solve the ill-posed inverse problem of mapping a noisy correlator to its corresponding smeared spectral density. Instead of training a single model, we train a "broad audience" of models to get a robust estimate of the final result, including full statistical and systematic error bars.The relationship between the correlator $C(t)$ and the spectral density $\rho(\omega)$ is given by the integral equation: $C(t) = \int_{0}^{\infty} K(t, \omega) \rho(\omega) d\omega$
where $K(t, \omega)$ is the appropriate physical kernel. The target of the neural network is the smeared spectral density $\hat{\rho}_\sigma(E)$, which relates to $\rho(\omega)$ as:
$\hat{\rho}_\sigma(E)$

where $S_\sigma(E, \omega)$ is the smearing function. The networks are trained on noisy correlator data $C(t) + \delta C(t)$ to predict $\hat{\rho}_\sigma(E)$.

The workflow is divided into three main stages:
$\hat{\rho}*\sigma(E) = \int*{0}^{\infty} S\_\sigma(E, \omega) \rho(\omega) d\omega$  $\hat{\rho}_\sigma(E)$
  * **Data Generation:** A large, model-independent training set is generated. We create millions of mock spectral densities $\rho(\omega)$ using a **Chebyshev basis**, calculate their corresponding clean correlators $C(t)$ using the appropriate physical kernel, and compute the target smeared spectral densities $\hat{\rho}_\sigma(E)$. Realistic **noise**, derived from a real lattice QCD covariance matrix, is injected into the clean correlators.
  * **Model Training:** A **"broad audience"** of neural networks is trained on this dataset. This involves training multiple ensembles (for different $N_b$, $N_\rho$, $N_n$) of models, with each ensemble containing several replicas ($N_r=5$) to sample the space of initial conditions and noise instances.
  * **Prediction & Analysis:** The trained ensembles are used to make predictions on unseen mock data (e.g., O(3) model, Delta Peak + Tanh Continuum). The predictions from the different ensembles are combined to calculate the final result with a **full systematic error budget**.

-----

## 📁 File Structure

This repository is organized as follows:

```
.
├── data/
│   └── octet-psq0_nb1000_bin2.h5   # Real noise data (correlator mean + covariance)
│
├── training_datasets_T32/
│   ├── (This directory will be created by Step 1)
│   └── training_data_Nb...Nrho....hdf5
│
├── trained_models_sigma44/
│   ├── (This directory will be created by Step 2, but in this repository already trained versions are available)
│   ├── Nmax/
│   │   ├── replica_1_best.pth
│   │   └── replica_1_best_history.csv
│   └── ...
│
├── trained_models_sigma63/
│   ├── (This directory will be created by Step 2, but in this repository already trained versions are available)
│   └── ...
│
├── mock_predictions_sigma44/
│   ├── (This directory will be created by Step 3, but in this repository predictions done with the trained models are available)
│   └── O3_predictions_Nmax_sigma0.44.hdf5
│
├── mock_predictions_sigma63/
│   └── ...
│
├── (Main Python Scripts)
│   ├── generate_full_dataset_sigma.py  # Script for Step 1 (Data Generation)
│   ├── hpc_train_sigma.py              # Script for Step 2 (Training Worker)
│   ├── Analysis.ipynb                  # Script for Steps 3 and 4 (O(3), Delta+Tanh Validation and Analysis)
│
├── (Helper Python Scripts)
│   ├── sigma_model_utils.py            # Kernels for sigma models (Eq. 32, etc.)
│   ├── pytorch_models_t32.py           # NN architectures (47 outputs)
│   ├── prepare_training_data_T12.py    # Misc. helpers (Chebyshev functions)
│   └── generate_delta_mock.py          # Helper for delta+tanh mock data
│
├── slurm_scripts
│   ├── submit_generation_sigma.sh      # Slurm script for Step 1
│   ├── submit_training_sigma.sh        # Slurm script for Step 2
│   ├── submit_prediction_O3.sh         # Slurm script for Step 3 (O3)
│   └── submit_prediction_delta.sh      # Slurm script for Step 3 (Delta+Tanh)
│
└── README.md                           # This file
```

-----

## 🚀 How to Use: A Step-by-Step Workflow

### Step 0: Environment Setup (One-Time)

You must set up a Python virtual environment on the HPC to run the code.

1.  Log in to the HPC login node.
2.  Load the required system modules:
    ```bash
    module load python/3.11.7-watv22a
    module load cuda/12.6.1-jr4ru3u
    ```
3.  Create a Python virtual environment (e.g., in your home directory):
    ```bash
    python -m venv ml_gpu_env
    ```
4.  Activate the environment:
    ```bash
    source ml_gpu_env/bin/activate
    ```
5.  Install the necessary packages:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install h5py numpy matplotlib tqdm pandas
    ```

### Step 1: Generate the Training Datasets

This step generates the **HDF5 files** containing the (Noisy Input, Clean Target) pairs.

1.  **Edit `generate_full_dataset_sigma.sh`:**
      * Set `#SBATCH --account=...` to your project ID (from `rub-acclist`).
      * Set `source ${HDIR}/ml_gpu_env/bin/activate` to point to your venv.
2.  **Submit the Job:** This will launch 4 parallel CPU jobs (one for each $N_b$ value).
    ```bash
    sbatch submit_generation_sigma.sh
    ```
    > **Note:** This job is **CPU-only** and will take a long time (days) to complete. Upon completion, the `training_datasets_T32/` directory will be populated.

### Step 2: Train the "Broad Audience" Models

This step trains the models for both $\sigma=0.44$ and $\sigma=0.63$.

1.  **Edit `submit_training_sigma.sh`:**
      * Set `#SBATCH --account=...` to your project ID.
      * Set `source ${HDIR}/ml_gpu_env/bin/activate` to point to your venv.
2.  This script will submit **80 jobs** (4 job types $\times$ 5 replicas $\times$ 2 sigmas) to the **GPU queue**.
3.  **Submit the Job:**
    ```bash
    sbatch submit_training_sigma.sh
    ```
    > **Note:** This will run for several hours/days. You can monitor it with `squeue -u your_username`. When finished, the `trained_models_sigma44/` and `trained_models_sigma63/` directories will be populated with `.pth` models and `.csv` history files.

### Step 3: Run Validation Predictions

After training, you can test your models on the mock datasets.

1.  **Edit `submit_prediction_O3.sh`:**
      * This script runs `predict_on_O3_mock_sigma.py` to test the **O(3) model**.
      * Edit the `--account` and `source activate` lines as before.
      * **Submit the O(3) Test:**
        ```bash
        sbatch submit_prediction_O3.sh
        ```
2.  **Edit `submit_prediction_delta.sh`:**
      * This script runs `predict_on_delta_tanh.py` to test the **delta+continuum model**.
      * Edit the `--account` and `source activate` lines.
      * **Submit the Delta+Tanh Test:**
        ```bash
        sbatch submit_prediction_delta.sh
        ```
    > **Note:** When these jobs are finished, the `mock_predictions_sigma.../` directories will be populated with HDF5 files containing the prediction results.

### Step 4: Analyze the Results

This final step is done **locally** on your own computer (or the HPC login node) to generate the final plots.

1.  **Download Results:** Download the `mock_predictions_sigma.../` directories from the HPC to your local machine.
2.  **Run the O(3) Analysis:**
    ```bash
    python analyze_O3_mock_sigma.py
    ```
    This will generate the **`prediction_O3_mock_sigma_analysis.png`** plot and an HDF5 file with the final analysis data.
3.  **Run the Delta+Tanh Analysis:**
    ```bash
    python analyze_delta_tanh_mock.py
    ```
<img width="1000" height="600" alt="delta_tanh" src="https://github.com/user-attachments/assets/f41307c4-b6bc-4739-9984-a4733c2295dd" />

<img width="1200" height="1400" alt="prediction" src="https://github.com/user-attachments/assets/16784a89-315d-48b6-9715-c87050fc2b38" />


