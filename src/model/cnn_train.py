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


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        # Squeeze phase: squash spatial dimensions
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation phase: learn which channels to care about
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excite (get the volume knobs between 0 and 1)
        y = self.fc(y).view(b, c, 1, 1)
        # Multiply the original channels by their new volume knobs
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    """A standard Residual Block with an integrated SE Block."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Attention applied inside the block
        self.se = SEBlock(out_channels)
        
        # Skip connection to prevent degradation
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) # Excite channels before adding the residual
        out += self.shortcut(x)
        out = F.gelu(out)
        return out


class ImprovedSeismicCNN(nn.Module):
    def __init__(self):
        super(ImprovedSeismicCNN, self).__init__()
        
        # Initial Feature Extraction
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False), # Reduced from 32
            nn.BatchNorm2d(16),
            nn.GELU()
        )
        
        # Reduced Residual Stages 
        self.layer1 = ResBlock(16, 32, stride=2)
        self.layer2 = ResBlock(32, 64, stride=2)
        self.layer3 = ResBlock(64, 128, stride=2)
        self.layer4 = ResBlock(128, 256, stride=2)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Shrunk Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # Increased dropout here
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.3), # Increased dropout here
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.in_conv(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



if __name__ == "__main__":
    # Initialization & Hyperparameters
    
    # Increased Batch Size
    BATCH_SIZE = 128
    
    # Training Data
    train_dataset = datasets.ImageFolder('./dataset/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Validation Data
    val_dataset = datasets.ImageFolder('./dataset/val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedSeismicCNN().to(device)
    scaler = torch.amp.GradScaler('cuda')

    criterion = nn.BCEWithLogitsLoss()


    # Base optimizer with increased weight decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2) 

    num_epochs = 100
    patience = 7
    epochs_no_improve = 0
    best_val_loss = float('inf') 
    save_path = "trained_model/best_seismic_model.pth"

    # The Plateau Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2) 

    # The Training and Validation Loop
    for epoch in range(num_epochs):
        
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

            # The backward pass
            scaler.scale(loss).backward()
            
            # Unscale the gradients to apply the clipping speed limit
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step and update
            scaler.step(optimizer)
            scaler.update()
            
            # Zero gradients for the next step
            optimizer.zero_grad()

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
                preds = (probs > 0.60).float() # Forces the model to be 60% sure
                
                correct_preds += (preds == labels).sum().item()
                total_preds += labels.size(0)
                
                val_all_labels.extend(labels.cpu().squeeze(1).tolist())
                val_all_probs.extend(probs.cpu().squeeze(1).tolist())
                val_all_preds.extend(preds.cpu().squeeze(1).tolist())

        # FIX: Calculate average validation loss BEFORE stepping the scheduler
        avg_val_loss = running_val_loss / len(val_loader.dataset)
        val_accuracy = correct_preds / total_preds
        
        # Tell the plateau scheduler to check the validation loss
        scheduler.step(avg_val_loss)

        # Calculate Advanced Metrics
        try:
            val_auc = roc_auc_score(val_all_labels, val_all_probs)
            val_mcc = matthews_corrcoef(val_all_labels, val_all_preds)
        except ValueError:
            val_auc = 0.0
            val_mcc = 0.0
        
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

    # Final Evaluation on Test Set
    print("\nRunning Final Evaluation on Test Set...")

    # Setup the Test Dataloader (using same BATCH_SIZE)
    test_dataset = datasets.ImageFolder('./dataset/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

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

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
