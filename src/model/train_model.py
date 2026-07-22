import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# --- 1. DATASET DEFINITION ---
class SeismicNpyDataset(Dataset):
    def __init__(self, data_dir, dataframe):
        self.data_dir = data_dir
        self.dataframe = dataframe
        
    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
        filename = self.dataframe.iloc[idx]['filename']
        label = self.dataframe.iloc[idx]['label']
        
        file_path = os.path.join(self.data_dir, filename)
        data = np.load(file_path)
        
        x_tensor = torch.from_numpy(data)
        y_tensor = torch.tensor(label, dtype=torch.long)
        return x_tensor, y_tensor

# --- 2. MODEL DEFINITION ---
class Seismic_CNN_LSTM(nn.Module):
    def __init__(self, input_channels=3, cnn_filters=32, lstm_hidden=64, num_classes=2):
        super(Seismic_CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(16, cnn_filters, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(lstm_hidden, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, (hn, cn) = self.lstm(x)
        x = self.dropout(hn[-1, :, :])
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    DATA_DIR = "data/24012020_M6.8_Sivrice__Elazig_/2020_01_24/"
    CSV_PATH = os.path.join(DATA_DIR, "forecast_labels.csv")
    
    print("Loading labels...")
    df = pd.read_csv(CSV_PATH)
    
    # Calculate Imbalance Weights
    num_noise = len(df[df['label'] == 0])
    num_quakes = len(df[df['label'] == 1])
    print(f"Data Distribution -> Noise: {num_noise}, Quakes: {num_quakes}")
    
    imbalance_ratio = num_noise / max(1, num_quakes)
    class_weights = torch.tensor([1.0, imbalance_ratio])
    
    # Initialize Dataset and DataLoader
    dataset = SeismicNpyDataset(DATA_DIR, df)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    
    # Setup Device and Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = Seismic_CNN_LSTM().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    EPOCHS = 20
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {epoch_loss:.4f}")
        
    print("Training complete!")
    torch.save(model.state_dict(), "seismic_forecaster.pth")
