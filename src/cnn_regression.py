import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) 
        out += self.shortcut(x)
        return F.gelu(out)


class RegressionSeismicCNN(nn.Module):
    def __init__(self):
        super(RegressionSeismicCNN, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU()
        )
        self.layer1 = ResBlock(16, 32, stride=2)
        self.layer2 = ResBlock(32, 64, stride=2)
        self.layer3 = ResBlock(64, 128, stride=2)
        self.layer4 = ResBlock(128, 256, stride=2)
        
        # FIXED: Replaced (1,1) GAP with an 8x8 adaptive pool to preserve RAM spatial textures
        self.texture_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten = nn.Flatten()
        
        # FIXED: Adjusted linear dimensions to accept the 8x8 flattened grid
        self.regressor = nn.Sequential(
            nn.Dropout(0.3),             
            nn.Linear(256 * 8 * 8, 256),
            nn.BatchNorm1d(256),         
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)            
        )

    def forward(self, x):
        x = self.in_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.texture_pool(x)
        x = self.flatten(x)
        x = self.regressor(x)
        return x


class MagnitudeDataset(Dataset):
    def __init__(self, image_dir, labels_csv, transform=None):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(labels_csv) 
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx]['filename']
        img_path = os.path.join(self.image_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        magnitude = float(self.labels_df.iloc[idx]['magnitude'])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(magnitude, dtype=torch.float32)


# ==========================================
# SAMPLER HELPER FUNCTION
# ==========================================
def get_balanced_sampler(labels_csv):
    df = pd.read_csv(labels_csv)
    mags = df['magnitude'].values
    
    bins = np.arange(0, 10.5, 0.5)
    binned_mags = np.digitize(mags, bins)
    
    class_counts = np.bincount(binned_mags)
    class_weights = 1.0 / np.where(class_counts == 0, 1, class_counts)
    
    sample_weights = class_weights[binned_mags]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    return sampler


if __name__ == "__main__":
    
    BATCH_SIZE = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # FIXED: Added RandomErasing to combat overfitting on oversampled RAM matrices
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_csv = './dataset/train/labels.csv'
    train_dataset = MagnitudeDataset(image_dir='./dataset/train/earthquakes', labels_csv=train_csv, transform=train_transform)
    train_sampler = get_balanced_sampler(train_csv)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True)

    val_dataset = MagnitudeDataset(image_dir='./dataset/val/earthquakes', labels_csv='./dataset/val/labels.csv', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = RegressionSeismicCNN().to(device)
    scaler = torch.amp.GradScaler('cuda')

    # FIXED: Changed to L1Loss to directly optimize for MAE
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2) 

    num_epochs = 100
    patience = 10
    epochs_no_improve = 0
    best_val_loss = float('inf') 
    save_path = "trained_model/best_regression_model.pth"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) 

    for epoch in range(num_epochs):
        
        # --- TRAINING PHASE ---
        model.train() 
        running_train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.unsqueeze(1).to(device)
            
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_train_loss += loss.item() * inputs.size(0) 
            
        avg_train_loss = running_train_loss / len(train_loader.dataset)
        
        # --- VALIDATION PHASE ---
        model.eval() 
        running_val_loss = 0.0
        running_val_mse = 0.0 # Track actual MSE for RMSE calculation
        
        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.unsqueeze(1).to(device)
                
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    mse_loss = F.mse_loss(outputs, targets, reduction='mean')
                
                running_val_loss += loss.item() * inputs.size(0)
                running_val_mse += mse_loss.item() * inputs.size(0)

        # FIXED: Metrics now reflect accurate MAE and true RMSE
        avg_val_loss = running_val_loss / len(val_loader.dataset) # This is now pure MAE
        val_rmse = np.sqrt(running_val_mse / len(val_loader.dataset)) 
        
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train MAE Loss: {avg_train_loss:.4f} | Val MAE Loss: {avg_val_loss:.4f}")
        print(f"  Val RMSE: {val_rmse:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  => Best model saved! (Val MAE: {avg_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  => No improvement for {epochs_no_improve} epoch(s).")
            
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered! Validation loss hasn't improved in {patience} epochs.")
            break 
            
    # --- TESTING PHASE ---
    print("\nRunning Final Evaluation on Test Set...")
    test_dataset = MagnitudeDataset(image_dir='./dataset/test/earthquakes', labels_csv='./dataset/test/labels.csv', transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model.load_state_dict(torch.load(save_path))
    model.eval() 

    test_loss = 0.0
    test_mse = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad(): 
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.unsqueeze(1).to(device)
            
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                mse_loss = F.mse_loss(outputs, targets, reduction='mean')
                
            test_loss += loss.item() * inputs.size(0)
            test_mse += mse_loss.item() * inputs.size(0)

            all_targets.extend(targets.cpu().squeeze(1).tolist())
            all_preds.extend(outputs.cpu().squeeze(1).tolist())
            
    final_mae = test_loss / len(test_loader.dataset)
    final_rmse = np.sqrt(test_mse / len(test_loader.dataset))

    print(f"Final Test MAE: {final_mae:.4f}")
    print(f"Final Test RMSE: {final_rmse:.4f}")
    
    print("\nSample Predictions vs Actual:")
    for i in range(min(10, len(all_preds))):
        print(f"  Predicted: {all_preds[i]:.2f} | Actual: {all_targets[i]:.2f} | Diff: {abs(all_preds[i]-all_targets[i]):.2f}")
