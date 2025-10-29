#!/bin/bash

# --- SLURM JOB SUBMISSION SCRIPT (Momenta Training) ---

# -- Job Details ---
#SBATCH --job-name=nn_momenta_array
#SBATCH --output=slurm_logs/train_momenta_%x_%A_%a.out
#SBATCH --error=slurm_logs/train_momenta_%x_%A_%a.err
#SBATCH --account=bulavjzl_0001      # *** CRITICAL: Replace with your project ID ***

# -- Resources ---
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00                 # 2-day limit

# -- Job Array ---
# 4 jobs * 5 replicas * 2 alphas = 40 total tasks
# Array indices from 0 to 39
#SBATCH --array=0-39

# --- Your Environment Setup ---
echo "========================================================"
echo "Starting job on host: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "========================================================"

# --- 1. Set up a Temporary Working Directory on the Node ---
HDIR=$SLURM_SUBMIT_DIR
WDIR=/tmp/$SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID
mkdir -p ${WDIR}
cd ${WDIR}
echo "Created temporary working directory at: ${WDIR}"
echo "Copying files from home directory: ${HDIR}"

# --- 2. Map Array Task ID to Job Label, Replica, and Alpha ---
JOB_LABELS=("Nmax" "Nref_b" "Nref_n" "Nref_rho")
REPLICAS_PER_JOB=5
ALPHAS_PER_JOB=2 # (alpha=0, alpha=1)
TASKS_PER_JOB_LABEL=$((REPLICAS_PER_JOB * ALPHAS_PER_JOB))
JOB_INDEX=$((SLURM_ARRAY_TASK_ID / TASKS_PER_JOB_LABEL))
JOB_LABEL=${JOB_LABELS[$JOB_INDEX]}
REMAINDER=$((SLURM_ARRAY_TASK_ID % TASKS_PER_JOB_LABEL))
REPLICA_NUM=$((REMAINDER / ALPHAS_PER_JOB + 1))
ALPHA=$((REMAINDER % ALPHAS_PER_JOB))

# --- 3. Copy Necessary Files to the Temporary Directory ---
cp ${HDIR}/hpc_train_momenta.py .
cp ${HDIR}/pytorch_models_momenta.py .

JOB_INFO_ARRAY=("128 400000" "16 400000" "128 400000" "128 25000") # Nb Nrho pairs
JOB_INFO=${JOB_INFO_ARRAY[$JOB_INDEX]}
read -r NB NRHO <<< "$JOB_INFO"
DATASET_FILENAME="training_data_momenta_Nb${NB}_Nrho${NRHO}.hdf5"
echo "Copying dataset: ${DATASET_FILENAME} to local storage..."
cp ${HDIR}/training_datasets_momenta_T32/${DATASET_FILENAME} .

# --- 4. Load Required Modules ---
echo "Loading modules..."
module purge
module load python/3.11.7-watv22a
module load cuda/12.6.1-jr4ru3u

# Activate your python virtual environment
source ${HDIR}/ml_gpu_env/bin/activate

# --- 5. Perform the Calculation ---
echo "========================================================"
echo "This task will run:"
echo "  - Job Label:   $JOB_LABEL"
echo "  - Replica Num: $REPLICA_NUM"
echo "  - Alpha:       $ALPHA"
echo "========================================================"

srun python hpc_train_momenta.py --job_label $JOB_LABEL --replica_num $REPLICA_NUM --alpha $ALPHA

# --- 6. Copy Output Back to Global File System ---
echo "Copying output model and history back to submission directory..."
JOB_OUTPUT_DIR_NAME="${JOB_LABEL}_alpha${ALPHA}"
# Create the final output directory in your home area
mkdir -p ${HDIR}/trained_models_momenta/${JOB_OUTPUT_DIR_NAME}

# --- THIS IS THE FIX ---
# The paths below now correctly point to 'trained_models_momenta'
MODEL_FILE_TMP="trained_models_momenta/${JOB_OUTPUT_DIR_NAME}/replica_${REPLICA_NUM}_best.pth"
HISTORY_FILE_TMP="trained_models_momenta/${JOB_OUTPUT_DIR_NAME}/replica_${REPLICA_NUM}_best_history.csv"
OUTPUT_HOME_DIR="${HDIR}/trained_models_momenta/${JOB_OUTPUT_DIR_NAME}/"

if [ -f "$MODEL_FILE_TMP" ]; then
    cp "$MODEL_FILE_TMP" "$OUTPUT_HOME_DIR"
fi
if [ -f "$HISTORY_FILE_TMP" ]; then
    cp "$HISTORY_FILE_TMP" "$OUTPUT_HOME_DIR"
fi

# --- 7. Tidy Up ---
echo "Cleaning up temporary directory..."
rm -rf ${WDIR}

echo "========================================================"
echo "Job finished."
echo "========================================================"