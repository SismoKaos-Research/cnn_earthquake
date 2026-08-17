import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score


def train_seismic_model(model, train_loader, val_loader, epochs=50, device="cuda"):
    """
    Executes the training loop, tracking ROC-AUC and PR-AUC.
    """
    model = model.to(device)
    
    train_labels = train_loader.dataset.labels
    n_pos = train_labels.sum().item()
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_auc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (cat_seq, wave_seq, labels) in enumerate(train_loader):
            cat_seq = cat_seq.to(device)
            labels = labels.unsqueeze(1).to(device)
            
            if model.use_waveform:
                wave_seq = wave_seq.to(device)
            else:
                wave_seq = None
                
            optimizer.zero_grad()
            logits = model(cat_seq, wave_seq)
            
            loss = criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for cat_seq, wave_seq, labels in val_loader:
                cat_seq = cat_seq.to(device)
                labels = labels.unsqueeze(1).to(device)
                
                if model.use_waveform:
                    wave_seq = wave_seq.to(device)
                else:
                    wave_seq = None
                    
                logits = model(cat_seq, wave_seq)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader)
        
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        
        if all_targets.sum() == 0:
            val_roc_auc, val_pr_auc = 0.0, 0.0
        else:
            val_roc_auc = roc_auc_score(all_targets, all_preds)
            val_pr_auc = average_precision_score(all_targets, all_preds)
            
        scheduler.step(val_roc_auc)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val ROC-AUC: {val_roc_auc:.4f} | "
              f"Val PR-AUC: {val_pr_auc:.4f}")
              
        if val_roc_auc > best_val_auc:
            best_val_auc = val_roc_auc
            torch.save(model.state_dict(), "best_seismic_fusion_model.pth")
            
    print(f"Training complete. Best Validation ROC-AUC: {best_val_auc:.4f}")
    return model
