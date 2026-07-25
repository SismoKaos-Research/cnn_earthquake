import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (classification_report, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        # We use BCEWithLogitsLoss internally because it is numerically stable
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # 1. Calculate standard cross entropy
        bce_loss = self.bce_with_logits(inputs, targets)
        
        # 2. Convert logits to probabilities 
        # (Math trick: since BCE = -log(pt), then pt = exp(-BCE))
        pt = torch.exp(-bce_loss)
        
        # 3. Apply the focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()

# Model Definition
class SeismicCNN(nn.Module):
    def __init__(self):
        super(SeismicCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((4, 4)) # Switched to MaxPool
        
        self.dropout2d = nn.Dropout2d(0.3) 
        self.dropout1d = nn.Dropout(0.5)
        
        # 256 channels * 4 * 4 = 4096 features
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.adaptive_pool(x)
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1d(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
# Initialization
    # Training Data
    train_dataset = datasets.ImageFolder('./dataset/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # Validation Data
    val_dataset = datasets.ImageFolder('./dataset/val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeismicCNN().to(device)
    scaler = torch.amp.GradScaler('cuda')

    criterion = BinaryFocalLoss(alpha=1.0, gamma=2.0)

    # Added weight_decay=1e-3 (You can tweak this between 1e-2 and 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Drops learning rate by 50% if val loss doesn't improve for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # The Training and Validation Loop
    num_epochs = 100
    patience = 7
    epochs_no_improve = 0
    best_val_loss = float('inf') # Track the best loss to know when to save
    save_path = "trained_model/best_seismic_model.pth"

    for epoch in range(num_epochs):
        
        model.train() 
        running_train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            
            # Forward pass in mixed precision
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # The backward pass and optimizer step!
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item() * inputs.size(0) 
            
        avg_train_loss = running_train_loss / len(train_loader.dataset)
        
        # VALIDATION PHASE
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
                preds = torch.round(probs) 
                
                correct_preds += (preds == labels).sum().item()
                total_preds += labels.size(0)
                
                val_all_labels.extend(labels.cpu().squeeze(1).tolist())
                val_all_probs.extend(probs.cpu().squeeze(1).tolist())
                val_all_preds.extend(preds.cpu().squeeze(1).tolist())
                
        avg_val_loss = running_val_loss / len(val_loader.dataset)
        val_accuracy = correct_preds / total_preds

        # Calculate Advanced Metrics
        try:
            val_auc = roc_auc_score(val_all_labels, val_all_probs)
            val_mcc = matthews_corrcoef(val_all_labels, val_all_preds)
        except ValueError:
            val_auc = 0.0
            val_mcc = 0.0

        # Tell the scheduler to check the validation loss
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Acc: {val_accuracy:.4f} | Val AUC: {val_auc:.4f} | Val MCC: {val_mcc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  => Best model saved to {save_path}!")
        else:
            epochs_no_improve += 1
            print(f"  => No improvement for {epochs_no_improve} epoch(s).")
            
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered! Validation loss hasn't improved in {patience} epochs.")
            print(f"The best weights from the run have been saved to {save_path}.")
            break 
    print(f"Saving full model at {save_path}...")
    torch.save(model, "trained_model/full_model.pth")   

    print("\nRunning Final Evaluation on Test Set...")

    # Setup the Test Dataloader
    test_dataset = datasets.ImageFolder('./dataset/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Load the BEST weights we saved during training
    model.load_state_dict(torch.load(save_path))
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
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix:")
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Optional: detailed metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
