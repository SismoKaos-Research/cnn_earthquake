import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

# Training Data
train_dataset = datasets.ImageFolder('./dataset/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

# Validation Data
val_dataset = datasets.ImageFolder('./dataset/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Model Definition
class SeismicCNN(nn.Module):
    def __init__(self):
        super(SeismicCNN, self).__init__()
        
        # Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Spatial Reduction
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adaptive pooling guarantees the output is exactly 4x4, regardless of 
        # whether your input images are 94x94, 64x64, or 128x128.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Regularization
        # We use a 30% dropout between layers to prevent spatial memorization
        self.dropout2d = nn.Dropout2d(p=0.3) 
        self.dropout1d = nn.Dropout(p=0.5)
        
        # Fully Connected Block
        # 64 channels * 4 * 4 adaptive pool size = 1024 flat features
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 1) # Binary output

    def forward(self, x):
        # Pass through Conv1 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv1(x)))
        
        # Pass through Conv2 -> ReLU -> MaxPool -> Spatial Dropout
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2d(x)
        
        # Pass through Conv3 -> ReLU -> Adaptive Pool
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        
        # Flatten the 2D matrices into a 1D array for the dense layers
        x = torch.flatten(x, 1)
        
        # Pass through Dense -> ReLU -> Heavy Dropout -> Output
        x = F.relu(self.fc1(x))
        x = self.dropout1d(x)
        x = self.fc2(x)
        
        return x

# Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SeismicCNN().to(device)
scaler = torch.amp.GradScaler('cuda')

criterion = nn.BCEWithLogitsLoss() 
# Added weight_decay=1e-3 (You can tweak this between 1e-2 and 1e-4)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

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
    
    model.eval() 
    running_val_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    with torch.no_grad(): 
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            # Using autocast here speeds up validation too!
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_val_loss += loss.item() * inputs.size(0)
            
            probs = torch.sigmoid(outputs)
            preds = torch.round(probs) 
            
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)
            
    avg_val_loss = running_val_loss / len(val_loader.dataset)

    # Tell the scheduler to check the validation loss
    scheduler.step(avg_val_loss)
    val_accuracy = correct_preds / total_preds
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

    # EARLY STOPPING & SAVE LOGIC
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
