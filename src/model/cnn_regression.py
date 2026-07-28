import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1) # Outputs a single continuous magnitude value
        )

    def forward(self, x):
        x = self.in_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x


class MagnitudeDataset(Dataset):
    """
    Since ImageFolder only works for categorical sub-folders, 
    we need a custom dataset to map images to continuous magnitude floats.
    """
    def __init__(self, image_dir, labels_csv, transform=None):
        self.image_dir = image_dir
        # Assumes a CSV with columns: 'filename' and 'magnitude'
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

        # Return the magnitude as a float32 tensor
        return image, torch.tensor(magnitude, dtype=torch.float32)


if __name__ == "__main__":
    
    BATCH_SIZE = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Placeholder paths for when the preprocessor is updated to spit out CSVs
    train_dataset = MagnitudeDataset(image_dir='./dataset/dataset_60s/train_images', labels_csv='./dataset/dataset_60s/train_labels.csv', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = MagnitudeDataset(image_dir='./dataset/dataset_60s/val_images', labels_csv='./dataset/dataset_60s/val_labels.csv', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = RegressionSeismicCNN().to(device)
    scaler = torch.amp.GradScaler('cuda')

    # MSE Loss is standard for regression
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2) 

    num_epochs = 100
    patience = 10
    epochs_no_improve = 0
    best_val_loss = float('inf') 
    save_path = "trained_model/best_regression_model.pth"

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) 

    for epoch in range(num_epochs):
        
        # --- TRAINING PHASE ---
        model.train() 
        running_train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            # Targets need to match the output shape of the model: [Batch, 1]
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
        
        # We track Absolute Error for humans, Squared Error for the math
        total_absolute_error = 0.0
        
        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.unsqueeze(1).to(device)
                
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                running_val_loss += loss.item() * inputs.size(0)
                
                # Calculate MAE (Mean Absolute Error) for this batch
                abs_error = torch.abs(outputs - targets).sum().item()
                total_absolute_error += abs_error

        avg_val_loss = running_val_loss / len(val_loader.dataset) # This is your MSE
        val_rmse = np.sqrt(avg_val_loss) # Root Mean Squared Error
        val_mae = total_absolute_error / len(val_loader.dataset) # Mean Absolute Error
        
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")
        print(f"  Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f} (Avg. magnitude error)")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  => Best model saved! (RMSE: {val_rmse:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  => No improvement for {epochs_no_improve} epoch(s).")
            
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered! Validation loss hasn't improved in {patience} epochs.")
            break 
            
    print("\nRunning Final Evaluation on Test Set...")

    test_dataset = MagnitudeDataset(image_dir='./dataset/dataset_60s/test_images', labels_csv='./dataset/dataset_60s/test_labels.csv', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model.load_state_dict(torch.load(save_path))
    model.eval() 

    test_mse_loss = 0.0
    test_absolute_error = 0.0

    all_targets = []
    all_preds = []

    with torch.no_grad(): 
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.unsqueeze(1).to(device)
            
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            test_mse_loss += loss.item() * inputs.size(0)
            test_absolute_error += torch.abs(outputs - targets).sum().item()

            all_targets.extend(targets.cpu().squeeze(1).tolist())
            all_preds.extend(outputs.cpu().squeeze(1).tolist())
            
    final_mse = test_mse_loss / len(test_loader.dataset)
    final_rmse = np.sqrt(final_mse)
    final_mae = test_absolute_error / len(test_loader.dataset)

    print(f"Final Test MSE:  {final_mse:.4f}")
    print(f"Final Test RMSE: {final_rmse:.4f}")
    print(f"Final Test MAE:  {final_mae:.4f}")
    
    # Quick sanity check sample
    print("\nSample Predictions vs Actual:")
    for i in range(min(5, len(all_preds))):
        print(f"  Predicted: {all_preds[i]:.2f} | Actual: {all_targets[i]:.2f} | Diff: {abs(all_preds[i]-all_targets[i]):.2f}")
