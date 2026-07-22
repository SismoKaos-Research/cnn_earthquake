import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_dual_channel_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    num_epochs: int = 10, 
    learning_rate: float = 1e-3, 
    device: str = "cuda"
):
    """
    Standard training loop for the 1D2D-EDL phase-picking classifier.
    """
    model = model.to(device)
    
    # CrossEntropyLoss is the standard for multi-class classification 
    # (e.g., Class 0: Noise, Class 1: P-Wave, Class 2: S-Wave)
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer generally performs best out-of-the-box for CNN/LSTM combos
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Beginning training on device: {device}")
    
    for epoch in range(num_epochs):
        # -----------------------
        # TRAINING PHASE
        # -----------------------
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (raw_1d, img_2d, labels) in enumerate(train_loader):
            # Move data to GPU. 
            # Note: We cast inputs to float() because PyTorch expects 32-bit floats 
            # for weights, and your RAM images might natively load as uint8.
            raw_1d = raw_1d.float().to(device)
            img_2d = img_2d.float().to(device)
            labels = labels.long().to(device)
            
            # 1. Zero the parameter gradients
            optimizer.zero_grad()
            
            # 2. Forward pass: Feed BOTH inputs to the model
            outputs = model(raw_1d, img_2d)
            loss = criterion(outputs, labels)
            
            # 3. Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # 4. Track metrics
            running_train_loss += loss.item() * raw_1d.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_epoch_loss = running_train_loss / len(train_loader.dataset)
        train_epoch_acc = (correct_train / total_train) * 100.0
        
        # -----------------------
        # VALIDATION PHASE
        # -----------------------
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        # Disable gradient tracking for validation to save memory and speed up computation
        with torch.no_grad():
            for raw_1d, img_2d, labels in val_loader:
                raw_1d = raw_1d.float().to(device)
                img_2d = img_2d.float().to(device)
                labels = labels.long().to(device)
                
                # Forward pass
                outputs = model(raw_1d, img_2d)
                loss = criterion(outputs, labels)
                
                # Track metrics
                running_val_loss += loss.item() * raw_1d.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_epoch_loss = running_val_loss / len(val_loader.dataset)
        val_epoch_acc = (correct_val / total_val) * 100.0
        
        # Print epoch summary
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.2f}% | "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")
              
    print("Training Complete!")
    return model

# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    # Assuming you have instantiated your dataset and data loaders:
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize the model (using the class we built previously)
    # model = EDL1D2D(num_classes=3)
    
    # Check for GPU
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Run the loop
    # trained_model = train_dual_channel_model(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_epochs=15,
    #     learning_rate=0.001,
    #     device=device
    # )
