"""
Runs inference with a trained seismic classifier loaded from the FULL
pickled model object (`trained_model/full_model.pth`), as saved by
`training.run_training` via `torch.save(model, ...)` -- not from a
state-dict checkpoint (see `cnn_run_from_state.py` for that variant).

`torch.load(..., weights_only=False)` unpickles by qualified name rather
than by structural shape -- but **the name it asks for is `__main__`, not
`training`.** The object was saved from a script, so the pickle references
`__main__.ImprovedSeismicCNN`, `__main__.ResBlock` and `__main__.SEBlock`,
which is why the imports below are into THIS module and why this file only
works when run as a script (`tests/test_imports.py` skips it for that
reason).

**Corrected 2026-09-06.** This docstring used to say the classes "must stay
defined at exactly the module paths they're imported from" and that moving
them "would make `full_model.pth` fail to unpickle". That is not so, and it
was checked: loading fails with `Can't get attribute 'ImprovedSeismicCNN' on
<module '__main__'>` no matter where the class lives, and succeeds as soon as
the three names are bound into `__main__` from wherever they now are. The
constraint is on the imports in this file, not on the package layout -- and
the claim had been shaping that layout.

Converting the file to a state dict removes even that: verified to
round-trip into a fresh `ImprovedSeismicCNN` and reproduce identical outputs
on the same input (1,249,297 parameters, 90 state-dict keys). See
`cnn_run_from_state.py`, which already takes that path.

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

from detection.cnn_train import (ImprovedSeismicCNN, ResBlock,  # noqa: F401
                                 SEBlock)

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
