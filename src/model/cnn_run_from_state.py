import os

import pandas as pd
import torch
from cnn_train import SeismicCNN
from sklearn.metrics import (brier_score_loss, classification_report,
                             confusion_matrix, matthews_corrcoef,
                             roc_auc_score)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

WEIGHTS_PATH = "trained_model/best_seismic_model.pth"
DATA_DIR = "./dataset/test"              # ImageFolder root
BATCH_SIZE = 64
THRESHOLD = 0.5
NUM_WORKERS = 4
OUT_CSV = "trained_model/test_predictions.csv"

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
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    print(f"[INFO] Class mapping: {dataset.class_to_idx}")
    print(f"[INFO] Total test images: {len(dataset)}")

    model = SeismicCNN().to(device)
    state = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    results = []
    all_labels = []
    all_preds = []
    all_probs = [] # Added to track probabilities for AUC/Brier metrics

    sample_ptr = 0  # tracks absolute sample index across batches

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)

            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                logits = model(inputs)

            probs = torch.sigmoid(logits).squeeze(1)        # [B]
            preds = (probs >= THRESHOLD).long().cpu()       # [B]
            labels_cpu = labels.long().cpu()                # [B]

            # accumulate for confusion matrix and advanced metrics
            all_labels.extend(labels_cpu.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.cpu().tolist())

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

    #ADVANCED METRICS
    print("\n" + "="*30)
    print("ADVANCED METRICS")
    print("="*30)
    
    try:
        auc_score = roc_auc_score(all_labels, all_probs)
        mcc_score = matthews_corrcoef(all_labels, all_preds)
        brier_score = brier_score_loss(all_labels, all_probs)
        
        print(f"ROC-AUC Score: {auc_score:.4f}")
        print(f"Matthews Correlation Coefficient: {mcc_score:.4f}")
        print(f"Brier Score Loss: {brier_score:.4f}")
    except ValueError as e:
        print(f"[WARN] Could not compute advanced metrics: {e}")

if __name__ == "__main__":
    run_inference()
