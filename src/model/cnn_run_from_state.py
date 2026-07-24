import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

WEIGHTS_PATH = "trained_model/best_seismic_model.pth"
DATA_DIR = "./dataset/test"              # ImageFolder root
BATCH_SIZE = 64
THRESHOLD = 0.5
NUM_WORKERS = 4
OUT_CSV = "trained_model/test_predictions.csv"

class SeismicCNN(nn.Module):
    def __init__(self):
        super(SeismicCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.dropout2d = nn.Dropout2d(p=0.3)
        self.dropout1d = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2d(x)
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout1d(x)
        x = self.fc2(x)
        return x


def build_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def run_inference():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")
    if not os.path.isdir(DATA_DIR):
        raise NotADirectoryError(f"Data directory not found: {DATA_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    transform = build_transform()
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,                   # IMPORTANT: keeps order aligned with dataset.samples
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    print(f"[INFO] Class mapping: {dataset.class_to_idx}")
    print(f"[INFO] Total test images: {len(dataset)}")

    model = SeismicCNN().to(device)
    state = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    results = []
    all_labels = []
    all_preds = []

    sample_ptr = 0  # tracks absolute sample index across batches

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)

            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                logits = model(inputs)

            probs = torch.sigmoid(logits).squeeze(1)        # [B]
            preds = (probs >= THRESHOLD).long().cpu()       # [B]
            labels_cpu = labels.long().cpu()                # [B]

            # accumulate for confusion matrix
            all_labels.extend(labels_cpu.tolist())
            all_preds.extend(preds.tolist())

            # save per-image rows
            batch_size_actual = inputs.size(0)
            for i in range(batch_size_actual):
                img_path, true_idx = dataset.samples[sample_ptr]
                pred_idx = int(preds[i].item())
                prob = float(probs[i].item())

                results.append({
                    "image_path": img_path,
                    "true_label_idx": int(true_idx),
                    "true_label_name": idx_to_class.get(int(true_idx), str(true_idx)),
                    "pred_label_idx": pred_idx,
                    "pred_label_name": idx_to_class.get(pred_idx, str(pred_idx)),
                    "prob_positive": prob
                })
                sample_ptr += 1

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
                print(f"[INFO] Processed batch {batch_idx + 1}/{len(loader)} "
                      f"({sample_ptr}/{len(dataset)} images)")

    # sanity check
    if sample_ptr != len(dataset):
        print(f"[WARN] Processed {sample_ptr} samples, expected {len(dataset)}")

    # save CSV
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"[INFO] Saved predictions: {OUT_CSV}")

    # accuracy
    acc = (df["true_label_idx"] == df["pred_label_idx"]).mean() if len(df) > 0 else 0.0
    print(f"[INFO] Accuracy on '{DATA_DIR}': {acc * 100:.2f}%")

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        labels=[0, 1],
        target_names=[idx_to_class.get(0, "class_0"), idx_to_class.get(1, "class_1")],
        digits=4,
        zero_division=0
    ))


if __name__ == "__main__":
    run_inference()
