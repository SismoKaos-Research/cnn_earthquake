import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("trained_model/full_model.pth", map_location=device, weights_only=False)
model.to(device)
model.eval()

# Data
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.ImageFolder("./dataset/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Inference
all_preds = []
all_probs = []

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= 0.5).long()
        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

print("Done. Num predictions:", len(all_preds))
