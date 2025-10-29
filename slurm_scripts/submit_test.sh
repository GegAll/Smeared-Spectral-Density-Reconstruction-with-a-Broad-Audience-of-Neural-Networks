#!/bin/bash

# --- SLURM TEST JOB SUBMISSION SCRIPT (Optimized for Elysium) ---

# -- Job Details ---
#SBATCH --job-name=nn_test_run
#SBATCH --output=slurm_logs/test_%j.out      # A single log file for this test job
#SBATCH --error=slurm_logs/test_%j.err       # A single error file
#SBATCH --account=bulavjzl_0001      # *** CRITICAL: Replace with your project ID ***

# -- Resources ---
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# --- Your Environment Setup ---
echo "========================================================"
echo "Starting TEST job on host: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================================"

# --- 1. Set up a Temporary Working Directory on the Node ---
HDIR=$(pwd)
WDIR=/tmp/$SLURM_JOB_ID
mkdir -p ${WDIR}
cd ${WDIR}
echo "Created temporary working directory at: ${WDIR}"

# --- 2. Copy Necessary Files to the Temporary Directory ---
cp ${HDIR}/hpc_train_worker.py .
cp ${HDIR}/pytorch_models_t32.py .
DATASET_FILENAME="training_data_Nb128_Nrho25000.hdf5"
echo "Copying dataset for test run: ${DATASTICK_FILENAME} to local storage..."
cp ${HDIR}/training_datasets_T32/${DATASET_FILENAME} .

# --- 3. Load Required Modules ---
echo "Loading modules..."
module purge
module load python/3.11.7-watv22a
module load cuda/12.6.1-jr4ru3u

# Activate your python virtual environment (venv)
source ${HDIR}/ml_gpu_env/bin/activate

# --- 4. Perform the Calculation (in Test Mode) ---
echo "========================================================"
echo "Running a short test (3 epochs) of the Nref_rho job, replica 1..."
echo "========================================================"

srun python hpc_train_worker.py --job_label Nref_rho --replica_num 1 --test_run

# --- 5. Copy Output Back to Global File System ---
echo "Copying output model and history back to submission directory..."
JOB_LABEL="Nref_rho" # Hard-coded for the test script
REPLICA_NUM=1

# Create the final output directory if it doesn't exist
mkdir -p ${HDIR}/trained_models/${JOB_LABEL}

# Define file paths
MODEL_FILE_TMP="trained_models/${JOB_LABEL}/replica_${REPLICA_NUM}_test.pth"
HISTORY_FILE_TMP="trained_models/${JOB_LABEL}/replica_${REPLICA_NUM}_test_history.csv"
MODEL_FILE_HOME="${HDIR}/trained_models/${JOB_LABEL}/"
HISTORY_FILE_HOME="${HDIR}/trained_models/${JOB_LABEL}/"

# Copy the model file if it exists
if [ -f "$MODEL_FILE_TMP" ]; then
    cp "$MODEL_FILE_TMP" "$MODEL_FILE_HOME"
fi

# --- THIS IS THE NEW LINE ---
# Copy the history file if it exists
if [ -f "$HISTORY_FILE_TMP" ]; then
    cp "$HISTORY_FILE_TMP" "$HISTORY_FILE_HOME"
fi

# --- 6. Tidy Up ---
echo "Cleaning up temporary directory..."
rm -rf ${WDIR}

echo "========================================================"
echo "Test job finished."
echo "========================================================"