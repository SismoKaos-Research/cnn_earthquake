import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# --- 1. UNLABELED DATASET ---
class UnlabeledSeismicDataset(Dataset):
    def __init__(self, data_dir):
        """
        Loads all .npy files in the directory.
        No labels are required.
        """
        self.file_paths = glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True)
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        # Load the (3, N) numpy array
        data = np.load(self.file_paths[idx])
        
        # Convert to PyTorch float tensor
        x_tensor = torch.from_numpy(data).float()
        
        # Return x_tensor as both the input AND the target
        return x_tensor, x_tensor

# --- 2. AUTOENCODER ARCHITECTURE ---
class SeismicAutoencoder(nn.Module):
    def __init__(self, input_channels=3):
        super(SeismicAutoencoder, self).__init__()
        
        # ENCODER: Compresses the waveform into a latent representation
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(True),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(True),
            
            # The Bottleneck (Latent Space)
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(True)
        )
        
        # DECODER: Rebuilds the waveform from the latent representation
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(True),
            
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(True),
            
            # Final output layer uses the original channel count (3)
            # No activation here because seismic waveforms contain negative amplitudes
            nn.ConvTranspose1d(16, input_channels, kernel_size=7, stride=2, padding=3, output_padding=1)
        )

    def forward(self, x):
        latent_features = self.encoder(x)
        reconstructed = self.decoder(latent_features)
        return reconstructed

# --- 3. TRAINING EXECUTION ---
if __name__ == "__main__":
    # Settings
    DATA_DIR = "data"
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 1e-3

    print(f"Loading unlabeled data from {DATA_DIR}...")
    dataset = UnlabeledSeismicDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"Found {len(dataset)} windows for unsupervised pre-training.")

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    autoencoder = SeismicAutoencoder(input_channels=3).to(device)
    
    # Use Mean Squared Error (MSE) to measure reconstruction accuracy
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Training Loop
    for epoch in range(EPOCHS):
        autoencoder.train()
        running_loss = 0.0
        
        for batch_x, batch_target in dataloader:
            batch_x = batch_x.to(device)
            batch_target = batch_target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed = autoencoder(batch_x)
            
            # Calculate reconstruction error
            loss = criterion(reconstructed, batch_target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Reconstruction Loss (MSE): {epoch_loss:.6f}")

    print("\nPre-training complete!")
    
    # Save the FULL autoencoder (optional, for visualization later)
    torch.save(autoencoder.state_dict(), "model/full_autoencoder.pth")
    
    # Save ONLY THE ENCODER (this is what you need for the CNN-LSTM step)
    torch.save(autoencoder.encoder.state_dict(), "model/pretrained_encoder.pth")
    print("Saved 'pretrained_encoder.pth' for the forecasting model.")
