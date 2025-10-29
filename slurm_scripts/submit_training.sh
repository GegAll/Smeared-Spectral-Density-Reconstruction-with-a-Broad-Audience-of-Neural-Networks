#!/bin/bash

# --- SLURM JOB SUBMISSION SCRIPT (Optimized for Elysium) ---

# -- Job Details ---
#SBATCH --job-name=nn_training_array
#SBATCH --output=slurm_logs/train_%x_%A_%a.out  # Log file for each task (%x=job-name, %A=job-ID, %a=task-ID)
#SBATCH --error=slurm_logs/train_%x_%A_%a.err   # Error file
#SBATCH --account=bulavjzl_0001      # *** CRITICAL: Replace with your project ID from 'rub-acclist' ***

# -- Resources ---
#SBATCH --partition=gpu
#SBATCH --nodes=1                         # Request 1 node (mandatory)
#SBATCH --gpus=1                          # Recommended syntax for requesting 1 GPU
#SBATCH --cpus-per-gpu=8                  # Request 8 CPU cores per GPU
#SBATCH --mem=32G                         # Request 32 GB of system RAM
#SBATCH --time=2-00:00:00                 # Max runtime of 2 days per job

# -- Job Array ---
# This creates an array of 20 jobs, indexed from 0 to 19
# Total jobs = 4 job_types * 5 replicas = 20
#SBATCH --array=0-19

# --- Your Environment Setup ---
echo "========================================================"
echo "Starting job on host: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "========================================================"

# --- 1. Set up a Temporary Working Directory on the Node ---
HDIR=$(pwd) # Your home/submission directory
WDIR=/tmp/$SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID # Unique temp dir for each task
mkdir -p ${WDIR}
cd ${WDIR}
echo "Created temporary working directory at: ${WDIR}"

# --- 2. Map Array Task ID to Job Label and Replica Number ---
JOB_LABELS=("Nmax" "Nref_b" "Nref_n" "Nref_rho")
REPLICAS_PER_JOB=5
JOB_INDEX=$((SLURM_ARRAY_TASK_ID / REPLICAS_PER_JOB))
REPLICA_NUM=$((SLURM_ARRAY_TASK_ID % REPLICAS_PER_JOB + 1))
JOB_LABEL=${JOB_LABELS[$JOB_INDEX]}

# --- 3. Copy Necessary Files to the Temporary Directory ---
cp ${HDIR}/hpc_train_worker.py .
cp ${HDIR}/pytorch_models_t32.py .

# Determine and copy the specific dataset file needed for this job
JOB_INFO_ARRAY=("128 400000" "16 400000" "128 400000" "128 25000") # Nb Nrho pairs matching JOB_LABELS
JOB_INFO=${JOB_INFO_ARRAY[$JOB_INDEX]}
read -r NB NRHO <<< "$JOB_INFO"
DATASET_FILENAME="training_data_Nb${NB}_Nrho${NRHO}.hdf5"
echo "Copying dataset: ${DATASET_FILENAME} to local storage..."
cp ${HDIR}/training_datasets_T32/${DATASET_FILENAME} .

# --- 4. Load Required Modules ---
echo "Loading modules..."
module purge # Best practice
module load python/3.11.7-watv22a
module load cuda/12.6.1-jr4ru3u

# Activate your python virtual environment (venv)
# IMPORTANT: Replace 'ml_gpu_env' with your actual environment name
source ${HDIR}/ml_gpu_env/bin/activate

# --- 5. Perform the Calculation ---
echo "========================================================"
echo "This task will run:"
echo "  - Job Label:   $JOB_LABEL"
echo "  - Replica Num: $REPLICA_NUM"
echo "========================================================"

# Run the python script without the --test_run flag for the full training
srun python hpc_train_worker.py --job_label $JOB_LABEL --replica_num $REPLICA_NUM

# --- 6. Copy Output Back to Global File System ---
echo "Copying output model and history back to submission directory..."
# Create the final output directory if it doesn't exist
mkdir -p ${HDIR}/trained_models/${JOB_LABEL}

# Define file paths for the full run
MODEL_FILE_TMP="trained_models/${JOB_LABEL}/replica_${REPLICA_NUM}_best.pth"
HISTORY_FILE_TMP="trained_models/${JOB_LABEL}/replica_${REPLICA_NUM}_best_history.csv"
OUTPUT_HOME_DIR="${HDIR}/trained_models/${JOB_LABEL}/"

# Copy the model file if it was created
if [ -f "$MODEL_FILE_TMP" ]; then
    cp "$MODEL_FILE_TMP" "$OUTPUT_HOME_DIR"
fi

# Copy the history file if it was created
if [ -f "$HISTORY_FILE_TMP" ]; then
    cp "$HISTORY_FILE_TMP" "$OUTPUT_HOME_DIR"
fi

# --- 7. Tidy Up ---
echo "Cleaning up temporary directory..."
rm -rf ${WDIR}

echo "========================================================"
echo "Job finished."
echo "========================================================"