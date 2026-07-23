import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

# Use ImageFolder. It expects a standard directory structure:
# dataset/train/earthquake/
# dataset/train/noise/
train_dataset = datasets.ImageFolder('./dataset/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class SeismicClassifier(nn.Module):
    def __init__(self):
        super(SeismicClassifier, self).__init__()
        
        # Initialize an untrained ResNet-18
        self.resnet = models.resnet18(weights=None)
        
        
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1) # 1 output node for a binary prediction

    def forward(self, x):
        return self.resnet(x)

# Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SeismicClassifier().to(device)

# Binary Cross Entropy with Logits is the standard loss function for binary classification
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        
        # Reshape labels to match the output dimensions [batch_size, 1] and convert to float
        labels = labels.float().unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {running_loss/len(train_loader):.4f}")
