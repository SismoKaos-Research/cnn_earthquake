import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from cnn_train import ImprovedSeismicCNN
from sklearn.metrics import (classification_report, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)
from torch.utils.data import DataLoader, Dataset


# CUSTOM TENSOR DATASET
class SeismicTensorDataset(Dataset):
    """
    Loads PyTorch tensor (.pt) spectrograms directly from disk.
    Bypasses standard image loaders to preserve precision.
    """
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.files = sorted(list(self.root_dir.rglob("*.pt")))
        
        # Automatically map subfolder names to integer labels (e.g., 00_noise -> 0, 01_earthquake -> 1)
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for fpath in self.files:
            cls_name = fpath.parent.name
            if cls_name in self.class_to_idx:
                self.samples.append((fpath, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        # Load the saved tensor shape: (3, Freq, Time)
        tensor = torch.load(fpath, weights_only=True)
        return tensor, torch.tensor(label, dtype=torch.long)


# MAIN TRAINING SCRIPT
if __name__ == "__main__":
    # --- Configuration ---
    DATA_ROOT = "./dataset" 
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    PATIENCE = 7
    SAVE_DIR = Path("trained_model")
    SAVE_DIR.mkdir(exist_ok=True)
    SAVE_PATH = SAVE_DIR / "best_seismic_model_spectograp.pth"

    # --- Data Loading ---
    print("Loading datasets...")
    train_dataset = SeismicTensorDataset(f"{DATA_ROOT}/train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = SeismicTensorDataset(f"{DATA_ROOT}/val")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # --- Model & Hardware Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = ImprovedSeismicCNN().to(device)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2) 

    # --- Training State ---
    epochs_no_improve = 0
    best_val_loss = float('inf') 

    # --- Training Loop ---
    print("\nStarting Training...")
    for epoch in range(NUM_EPOCHS):
        
        # Training Phase
        model.train() 
        running_train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            # Apply Label Smoothing (0.0 -> 0.1, and 1.0 -> 0.9)
            labels = labels * 0.8 + 0.1
            
            # Forward pass in mixed precision
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass
            scaler.scale(loss).backward()
            
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_train_loss += loss.item() * inputs.size(0) 
            
        avg_train_loss = running_train_loss / len(train_loader.dataset)
        
        # Validation Phase
        model.eval() 
        running_val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        val_all_labels = []
        val_all_probs = []
        val_all_preds = []
        
        with torch.no_grad(): 
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * inputs.size(0)
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.60).float() # Threshold for prediction
                
                correct_preds += (preds == labels).sum().item()
                total_preds += labels.size(0)
                
                val_all_labels.extend(labels.cpu().squeeze(1).tolist())
                val_all_probs.extend(probs.cpu().squeeze(1).tolist())
                val_all_preds.extend(preds.cpu().squeeze(1).tolist())

        avg_val_loss = running_val_loss / len(val_loader.dataset)
        val_accuracy = correct_preds / total_preds
        
        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Metrics
        try:
            val_auc = roc_auc_score(val_all_labels, val_all_probs)
            val_mcc = matthews_corrcoef(val_all_labels, val_all_preds)
        except ValueError:
            val_auc = 0.0
            val_mcc = 0.0
        
        print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | AUC: {val_auc:.4f} | MCC: {val_mcc:.4f}")

        # Checkpoint Saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)
        else:
            epochs_no_improve += 1
            
        # Early Stopping
        if epochs_no_improve >= PATIENCE:
            print(f"\n[EARLY STOPPING] Validation loss hasn't improved in {PATIENCE} epochs.")
            break 
            
    print(f"\nTraining Complete! Best weights saved to {SAVE_PATH}.")
    torch.save(model, SAVE_DIR / "full_model.pth")   

    # FINAL TEST EVALUATION
    print("\nRunning Final Evaluation on Test Set...")

    test_dataset = SeismicTensorDataset(f"{DATA_ROOT}/test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Load best weights
    model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
    model.eval() 

    correct_preds = 0
    total_preds = 0
    all_labels = []
    all_preds = []

    with torch.no_grad(): 
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                
            probs = torch.sigmoid(outputs)
            preds = torch.round(probs) 
            
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

            all_labels.extend(labels.cpu().squeeze(1).tolist())
            all_preds.extend(preds.cpu().squeeze(1).tolist())
            
    test_accuracy = correct_preds / total_preds
    print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")
    
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix:")
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
