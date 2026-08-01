import torch
# Imported so torch.load can resolve the pickled class in full_model.pth.
# (This said `SeismicCNN`, which no longer exists -- the script raised
# ImportError before it could load anything.)
from cnn_train import ImprovedSeismicCNN  # noqa: F401
from sklearn.metrics import (brier_score_loss, classification_report,
                             confusion_matrix, matthews_corrcoef,
                             roc_auc_score)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
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
all_labels = [] # List to hold the ground truth labels

with torch.no_grad():
    # Modified to unpack both images and labels
    for images, labels in test_loader: 
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= 0.5).long()
        
        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        
        # Save the actual labels to compare against predictions
        all_labels.extend(labels.tolist()) 

print("Done. Num predictions:", len(all_preds))

print("\n" + "="*30)
print("EVALUATION METRICS")
print("="*30)

cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

if cm.size == 4:
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))

print("\n" + "="*30)
print("ADVANCED METRICS")
print("="*30)

# ROC-AUC uses the probabilities, NOT the hard predictions
auc_score = roc_auc_score(all_labels, all_probs)
print(f"ROC-AUC Score: {auc_score:.4f}")

# MCC uses the hard predictions
mcc_score = matthews_corrcoef(all_labels, all_preds)
print(f"Matthews Correlation Coefficient: {mcc_score:.4f}")

# Brier Score uses probabilities
brier_score = brier_score_loss(all_labels, all_probs)
print(f"Brier Score Loss: {brier_score:.4f}")
