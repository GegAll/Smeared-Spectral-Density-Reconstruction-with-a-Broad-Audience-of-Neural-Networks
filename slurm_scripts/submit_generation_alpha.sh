#!/bin/bash

# --- SLURM DATA GENERATION SCRIPT (CPU Job Array) ---

# -- Job Details ---
#SBATCH --job-name=nn_data_gen
#SBATCH --output=slurm_logs/gen_%x_%A_%a.out
#SBATCH --error=slurm_logs/gen_%x_%A_%a.err
#SBATCH --account=bulavjzl_0001      # *** CRITICAL: Replace with your project ID ***

# -- Resources ---
# We are requesting a CPU node, not a GPU node
#SBATCH --partition=cpu
#SBATCH --nodes=1                         # Request 1 full node
#SBATCH --ntasks=1                        # Run 1 main task (the python script) on that node
#SBATCH --cpus-per-task=48                # Request all 48 cores on the node for multiprocessing
#SBATCH --mem=32G                         # Request 32 GB of system RAM
#SBATCH --time=1-00:00:00                 # Max runtime of 1 day per Nb value

# -- Job Array ---
# This creates an array of 4 jobs, indexed from 0 to 3
# One for each Nb value: [16, 32, 64, 128]
#SBATCH --array=0-3

# --- Your Environment Setup ---
echo "========================================================"
echo "Starting DATA GEN job on host: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "========================================================"

# --- 1. Set up a Temporary Working Directory on the Node ---
HDIR=$(pwd)
WDIR=/tmp/$SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID
mkdir -p ${WDIR}
cd ${WDIR}
echo "Created temporary working directory at: ${WDIR}"

# --- 2. Map Array Task ID to Nb Value ---
NB_VALUES_ARRAY=(16 32 64 128)
NB_TO_RUN=${NB_VALUES_ARRAY[$SLURM_ARRAY_TASK_ID]}

# --- 3. Copy Necessary Files to the Temporary Directory ---
cp ${HDIR}/train_generation_alpha.py .
cp ${HDIR}/prepare_training_data_T12.py .
# Copy the real noise data file (assuming it's in a 'data' subfolder)
mkdir -p data
cp ${HDIR}/data/octet-psq0_nb1000_bin2.h5 ./data/

# --- 4. Load Required Modules ---
echo "Loading modules..."
module purge
module load python/3.11.7-watv22a
# No CUDA module needed for this CPU-only job

# Activate your python virtual environment
source ${HDIR}/ml_gpu_env/bin/activate

# --- 5. Perform the Calculation ---
echo "========================================================"
echo "This task will run data generation for:"
echo "  - Nb = $NB_TO_RUN"
echo "========================================================"

# Run the python script, passing the Nb value as an argument
srun python train_generation_alpha.py --nb $NB_TO_RUN

# --- 6. Copy Output Back to Global File System ---
echo "Copying output HDF5 files back to submission directory..."
# Create the final output directory if it doesn't exist
mkdir -p ${HDIR}/training_datasets_momenta_T32
# Copy all generated HDF5 files
cp training_datasets_momenta_T32/*.hdf5 ${HDIR}/training_datasets_momenta_T32/

# --- 7. Tidy Up ---
echo "Cleaning up temporary directory..."
rm -rf ${WDIR}

echo "========================================================"
echo "Data generation job finished for Nb = $NB_TO_RUN."
echo "========================================================"