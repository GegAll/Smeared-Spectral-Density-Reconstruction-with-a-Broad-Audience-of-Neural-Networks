import os
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import argparse
import numpy as np # Import numpy for saving

# Import your adapted architectures for T=32
from pytorch_models_t32 import ArcS_t32, ArcL_t32

# --- CONFIGURATION ---
training_jobs = {
    "Nmax": {"architecture": ArcL_t32, "nb": 128, "nrho": 400000},
    "Nref_b": {"architecture": ArcL_t32, "nb": 16, "nrho": 400000},
    "Nref_n": {"architecture": ArcS_t32, "nb": 128, "nrho": 400000},
    "Nref_rho": {"architecture": ArcL_t32, "nb": 128, "nrho": 25000},
}
SIGMA_FOR_TARGET = 0.63
OUTPUT_MODEL_DIR = "trained_models"
BATCH_SIZE = 32
INITIAL_LR = 2e-4
FULL_MAX_EPOCHS = 200
WEIGHT_INIT_MEAN = 0.0
WEIGHT_INIT_STD = 0.05
ES_PATIENCE = 15
ES_MIN_DELTA = 1e-5

# --- OPTIMIZED DATASET CLASS ---
class CorrelatorDataset(Dataset):
    """
    Optimized PyTorch Dataset that loads the entire dataset into RAM at once.
    """
    def __init__(self, hdf5_path, sigma):
        print(f"  Loading dataset '{os.path.basename(hdf5_path)}' into RAM... ", end="", flush=True)
        with h5py.File(hdf5_path, 'r') as f:
            self.inputs = f['standardized_noisy_correlators'][:]
            self.targets = f[f'clean_smeared_rho_sigma_{sigma}'][:]
        self.n_samples, self.n_replicas, _ = self.inputs.shape
        self.total_size = self.n_samples * self.n_replicas
        print("Done.")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        sample_idx, replica_idx = divmod(idx, self.n_replicas)
        inputs_data = self.inputs[sample_idx, replica_idx, :]
        targets_data = self.targets[sample_idx, :]
        return torch.from_numpy(inputs_data).float().unsqueeze(0), torch.from_numpy(targets_data).float()


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.normal_(m.weight, mean=WEIGHT_INIT_MEAN, std=WEIGHT_INIT_STD)
        if m.bias is not None: nn.init.zeros_(m.bias)

def train_model(model, train_loader, val_loader, model_save_path, job_label, replica_num, max_epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Training on device: {device}")
    model.to(device)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)
    lr_lambda = lambda epoch: 1.0 if epoch < 25 else 1.0 / (1 + (epoch - 24) * 4e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    best_val_loss, epochs_no_improve = float('inf'), 0
    
    # These lists will now be returned
    train_losses, val_losses = [], []
    
    epoch_pbar = tqdm(range(max_epochs), desc=f"Job '{job_label}' Replica {replica_num}", unit="epoch")
    for epoch in epoch_pbar:
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False)
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)
                loss = loss_fn(predictions, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        
        epoch_pbar.set_postfix(val_loss=f"{avg_val_loss:.4e}")
        scheduler.step()

        # Append losses for history
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss - ES_MIN_DELTA:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= ES_PATIENCE and max_epochs > ES_PATIENCE:
            epoch_pbar.write(f"  Early stopping triggered at epoch {epoch+1}.")
            epoch_pbar.close()
            break
            
    # --- CHANGE 1: Return the collected loss histories ---
    return train_losses, val_losses

# --- MAIN SCRIPT EXECUTION ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a single training job for the broad audience.")
    parser.add_argument("--job_label", type=str, required=True, choices=training_jobs.keys())
    parser.add_argument("--replica_num", type=int, required=True)
    parser.add_argument("--test_run", action='store_true', help="Run for a few epochs as a quick test.")
    args = parser.parse_args()

    MAX_EPOCHS = 3 if args.test_run else FULL_MAX_EPOCHS

    job_label = args.job_label
    replica_num = args.replica_num
    job_info = training_jobs[job_label]
    architecture = job_info["architecture"]
    nb = job_info["nb"]
    nrho = job_info["nrho"]

    print(f"\n{'='*70}")
    print(f"EXECUTING SINGLE JOB FROM HPC WORKER")
    print(f"  - Job Label:    '{job_label}'")
    print(f"  - Replica Num:  {replica_num}")
    print(f"  - Test Run:     {args.test_run} (Max Epochs: {MAX_EPOCHS})")
    print(f"{'='*70}")
    
    dataset_file = f"training_data_Nb{nb}_Nrho{nrho}.hdf5"
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    full_dataset = CorrelatorDataset(dataset_file, SIGMA_FOR_TARGET)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    num_workers = int(os.environ.get('SLURM_CPUS_PER_GPU', 0))
    print(f"Using {num_workers} workers for DataLoader.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    job_output_dir = os.path.join(OUTPUT_MODEL_DIR, job_label)
    os.makedirs(job_output_dir, exist_ok=True)
    
    model = architecture()
    model.apply(init_weights)
    
    save_path = os.path.join(job_output_dir, f"replica_{replica_num}_test.pth") if args.test_run else os.path.join(job_output_dir, f"replica_{replica_num}_best.pth")
    
    # --- CHANGE 2: Capture the returned history ---
    train_history, val_history = train_model(model, train_loader, val_loader, save_path, job_label, replica_num, MAX_EPOCHS)
    
    # --- CHANGE 3: Save the history to a .csv file ---
    history_save_path = save_path.replace(".pth", "_history.csv")
    print(f"  Saving loss history to '{history_save_path}'")
    # Combine the two lists into a 2-column array
    history_data = np.array([train_history, val_history]).T
    np.savetxt(history_save_path, history_data, delimiter=',', header='train_loss,val_loss', fmt='%.6e', comments='')

    print(f"\n--- Finished training for job '{job_label}', replica {replica_num} ---")