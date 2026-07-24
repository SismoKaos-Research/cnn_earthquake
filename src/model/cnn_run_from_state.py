import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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


def run_inference(weights_path, data_dir, batch_size=64, threshold=0.5, num_workers=4, out_csv="predictions.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    transform = build_transform()
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Reverse map class index -> class name
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    # Load model
    model = SeismicCNN().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    results = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)

            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                logits = model(inputs)

            probs = torch.sigmoid(logits).squeeze(1)  # shape [B]
            preds = (probs >= threshold).long()

            labels = labels.long()  # shape [B]

            for i in range(inputs.size(0)):
                # dataset.samples gives (filepath, class_idx) in dataloader order when shuffle=False
                sample_idx = len(results)
                img_path, true_idx = dataset.samples[sample_idx]

                pred_idx = int(preds[i].item())
                prob = float(probs[i].item())

                # Predicted class name by predicted index:
                pred_class_name = idx_to_class.get(pred_idx, str(pred_idx))
                true_class_name = idx_to_class.get(int(true_idx), str(true_idx))

                results.append({
                    "image_path": img_path,
                    "true_label_idx": int(true_idx),
                    "true_label_name": true_class_name,
                    "pred_label_idx": pred_idx,
                    "pred_label_name": pred_class_name,
                    "prob_positive": prob
                })

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved predictions: {out_csv}")

    if len(df) > 0:
        acc = (df["true_label_idx"] == df["pred_label_idx"]).mean()
        print(f"[INFO] Accuracy on '{data_dir}': {acc * 100:.2f}%")
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SeismicCNN inference on folder data (ImageFolder format).")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights (.pth)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to inference data root (ImageFolder structure)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for binary classification")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--out_csv", type=str, default="predictions.csv", help="Output CSV path")

    args = parser.parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not os.path.isdir(args.data_dir):
        raise NotADirectoryError(f"Data directory not found: {args.data_dir}")

    run_inference(
        weights_path=args.weights,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        threshold=args.threshold,
        num_workers=args.num_workers,
        out_csv=args.out_csv
    )
