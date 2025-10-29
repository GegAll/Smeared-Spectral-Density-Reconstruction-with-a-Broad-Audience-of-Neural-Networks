#!/bin/bash

# --- SLURM PREDICTION JOB SUBMISSION SCRIPT ---

# -- Job Details ---
#SBATCH --job-name=nn_prediction
#SBATCH --output=slurm_logs/prediction_%j.out
#SBATCH --error=slurm_logs/prediction_%j.err
#SBATCH --account=your_project_id_here      # *** CRITICAL: Replace with your project ID ***

# -- Resources ---
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00                 # Max runtime of 1 hour (prediction is fast)

# --- Your Environment Setup ---
echo "========================================================"
echo "Starting PREDICTION job on host: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================================"

# --- 1. Set up a Temporary Working Directory on the Node ---
HDIR=$(pwd)
WDIR=/tmp/$SLURM_JOB_ID
mkdir -p ${WDIR}
cd ${WDIR}
echo "Created temporary working directory at: ${WDIR}"

# --- 2. Copy Necessary Files to the Temporary Directory ---
# Copy all required python utility scripts
cp ${HDIR}/predict_on_mock.py .
cp ${HDIR}/pytorch_models_t32.py .
cp ${HDIR}/read_correlator_dat.py .
cp ${HDIR}/read_covariance_dat.py .
cp ${HDIR}/generate_delta_mock.py .
cp ${HDIR}/prepare_training_data_T12.py . # This is needed by generate_delta_mock.py

# Copy the entire directories for models and datasets
echo "Copying required data to local storage..."
cp -r ${HDIR}/trained_models .
cp -r ${HDIR}/training_datasets_T32 .
cp -r ${HDIR}/data . # Assuming your .dat file is in a 'data' subfolder

# --- 3. Load Required Modules ---
echo "Loading modules..."
module purge
module load python/3.11.7-watv22a
module load cuda/12.6.1-jr4ru3u

# Activate your python virtual environment
source ${HDIR}/ml_gpu_env/bin/activate

# --- 4. Perform the Calculation ---
echo "========================================================"
echo "Running the prediction script..."
echo "========================================================"

srun python predict_on_mock.py

# --- 5. Copy Output Back to Global File System ---
echo "Copying prediction results back to submission directory..."
# Create the final output directory if it doesn't exist and copy the whole folder
mkdir -p ${HDIR}/mock_predictions
cp -r mock_predictions/* ${HDIR}/mock_predictions/

# --- 6. Tidy Up ---
echo "Cleaning up temporary directory..."
rm -rf ${WDIR}

echo "========================================================"
echo "Prediction job finished."
echo "========================================================"