"""
Runs inference with a trained seismic classifier loaded from the FULL
pickled model object (`trained_model/full_model.pth`), as saved by
`training.run_training` via `torch.save(model, ...)` -- not from a
state-dict checkpoint (see `cnn_run_from_state.py` for that variant).

`torch.load(..., weights_only=False)` below unpickles the saved object by
walking its exact module path, `training.ImprovedSeismicCNN`, and from
there into `model.trunk2d.SETrunk2D` and `model/blocks.py`'s `ResBlock`/
`SEBlock` via the class's MRO. That resolution is by qualified name, not by
structural shape, so `ImprovedSeismicCNN`, `ResBlock`, and `SEBlock` must
stay defined at exactly the module paths they're imported from below --
renaming, moving, or re-defining any of them (here or in `training.py`)
would make `full_model.pth` fail to unpickle. The `ResBlock, SEBlock`
import is otherwise unused in this script; it is kept (with the
`# noqa: F401` below) as the same backward-compatibility re-export
`cnn_train.py`'s own docstring describes -- older checkpoints pickled
before the `training.py` refactor may reference the legacy
`cnn_train.ImprovedSeismicCNN` module path rather than `training`'s.

Loads the ImageFolder test split from `./dataset/test`, runs inference, and
prints a confusion matrix, classification report, and ROC-AUC/MCC/Brier
score. No CLI flags -- every path is a hardcoded constant inline below.
This is a flat top-to-bottom script (no `main()`); it runs immediately on
import, so it is only ever meant to be run as `__main__`, never imported.

Usage:
    python cnn_run.py

Not imported by anything else -- standalone script.
"""

import torch
from sklearn.metrics import (brier_score_loss, classification_report,
                             confusion_matrix, matthews_corrcoef,
                             roc_auc_score)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn_train import ImprovedSeismicCNN, ResBlock, SEBlock  # noqa: F401

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
