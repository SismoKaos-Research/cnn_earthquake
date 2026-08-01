import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (classification_report, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        # Squeeze phase: squash spatial dimensions
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation phase: learn which channels to care about
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excite (get the volume knobs between 0 and 1)
        y = self.fc(y).view(b, c, 1, 1)
        # Multiply the original channels by their new volume knobs
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    """A standard Residual Block with an integrated SE Block."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Attention applied inside the block
        self.se = SEBlock(out_channels)

        # Skip connection to prevent degradation
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) # Excite channels before adding the residual
        out += self.shortcut(x)
        out = F.gelu(out)
        return out


class ImprovedSeismicCNN(nn.Module):
    def __init__(self, dropout1=0.5, dropout2=0.3, hidden_dim=64, num_stages=4):
        super(ImprovedSeismicCNN, self).__init__()

        # Initial Feature Extraction
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU()
        )

        # Residual stages. num_stages=4 keeps the original layer1..layer4
        # naming (old checkpoints load unchanged); num_stages=3 swaps layer4
        # for an Identity, cutting parameters from ~1.25M to ~0.3M -- sized
        # for the much smaller short-window datasets.
        self.layer1 = ResBlock(16, 32, stride=2)
        self.layer2 = ResBlock(32, 64, stride=2)
        self.layer3 = ResBlock(64, 128, stride=2)
        if num_stages >= 4:
            self.layer4 = ResBlock(128, 256, stride=2)
            final_channels = 256
        else:
            self.layer4 = nn.Identity()
            final_channels = 128

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(final_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.in_conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# --- WINDOW-LENGTH PRESETS ---
#
# Long windows (60s) have more effective signal diversity and tolerate the
# full-capacity network; short windows (3s/6s) overfit it within ~10 epochs.
# The short preset therefore shrinks the model, regularizes harder, augments,
# and checkpoints on val AUC (ranking quality) instead of val loss, which
# degrades earlier than AUC as the model loses calibration.
#
# Any value passed explicitly on the command line overrides its preset value.
PRESETS = {
    "long": dict(batch_size=128, num_epochs=100, patience=7, lr=1e-4,
                 weight_decay=1e-2, dropout1=0.5, dropout2=0.3, hidden_dim=64,
                 num_stages=4, monitor="loss", scheduler="plateau",
                 random_erasing=0.0),
    "short": dict(batch_size=64, num_epochs=80, patience=10, lr=2e-4,
                  weight_decay=3e-2, dropout1=0.6, dropout2=0.4, hidden_dim=32,
                  num_stages=3, monitor="auc", scheduler="cosine",
                  random_erasing=0.25),
}
SHORT_WINDOW_THRESHOLD_SEC = 12.0


def parse_args():
    parser = argparse.ArgumentParser(description="Train ImprovedSeismicCNN on RAM images.")
    parser.add_argument("--dataset-dir", type=str, default="./dataset",
                         help="Directory containing train/val/test subfolders (ImageFolder layout).")
    parser.add_argument("--save-dir", type=str, default="trained_model",
                         help="Directory to save checkpoints into.")
    parser.add_argument("--window-seconds", type=float, default=None,
                         help="Window length of the dataset being trained on. Selects the "
                              f"'short' preset when <= {SHORT_WINDOW_THRESHOLD_SEC:.0f}s, else 'long'. "
                              "Omit to use the long preset (original behavior). "
                              "Any explicitly passed flag overrides its preset value.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed (python/numpy/torch) for run-to-run comparability.")
    # Tunables: default None means "take it from the preset".
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None,
                         help="Early-stopping patience, in epochs without improvement of the monitored metric.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout1", type=float, default=None,
                         help="Dropout before the first classifier Linear layer.")
    parser.add_argument("--dropout2", type=float, default=None,
                         help="Dropout before the final classifier Linear layer.")
    parser.add_argument("--hidden-dim", type=int, default=None,
                         help="Width of the classifier's hidden layer.")
    parser.add_argument("--num-stages", type=int, default=None, choices=[3, 4],
                         help="Number of residual stages (3 = ~0.3M params, 4 = ~1.25M).")
    parser.add_argument("--monitor", type=str, default=None, choices=["loss", "auc"],
                         help="Metric used for checkpointing and early stopping.")
    parser.add_argument("--scheduler", type=str, default=None, choices=["plateau", "cosine"],
                         help="LR schedule: ReduceLROnPlateau on val loss, or cosine annealing.")
    parser.add_argument("--random-erasing", type=float, default=None,
                         help="Probability of RandomErasing augmentation on training images (0 disables).")
    args = parser.parse_args()

    preset_name = "long"
    if args.window_seconds is not None and args.window_seconds <= SHORT_WINDOW_THRESHOLD_SEC:
        preset_name = "short"
    preset = PRESETS[preset_name]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.preset_name = preset_name
    return args


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print(f"Preset: '{args.preset_name}'"
          + (f" (window_seconds={args.window_seconds})" if args.window_seconds else " (no --window-seconds given)"))
    for key in ["batch_size", "num_epochs", "patience", "lr", "weight_decay",
                "dropout1", "dropout2", "hidden_dim", "num_stages", "monitor",
                "scheduler", "random_erasing", "seed"]:
        print(f"  {key:15s} = {getattr(args, key)}")
    print("=" * 60)

    BATCH_SIZE = args.batch_size

    # RandomErasing operates on tensors, so it sits after ToTensor. It's the
    # one label-safe image augmentation here: geometric flips/rotations would
    # scramble the RAM matrix's temporal structure.
    train_tf_list = [transforms.ToTensor()]
    if args.random_erasing > 0:
        train_tf_list.append(transforms.RandomErasing(p=args.random_erasing, scale=(0.02, 0.15)))
    train_transform = transforms.Compose(train_tf_list)
    eval_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.ImageFolder(f"{args.dataset_dir}/train", transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = datasets.ImageFolder(f"{args.dataset_dir}/val", transform=eval_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedSeismicCNN(
        dropout1=args.dropout1, dropout2=args.dropout2,
        hidden_dim=args.hidden_dim, num_stages=args.num_stages,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} | Train samples: {len(train_dataset)} "
          f"({n_params / max(1, len(train_dataset)):.0f} params/sample)")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_epochs = args.num_epochs
    patience = args.patience
    epochs_no_improve = 0
    # monitor='loss' minimizes val loss; monitor='auc' maximizes val AUC.
    best_metric = float('inf') if args.monitor == "loss" else float('-inf')
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = f"{args.save_dir}/best_seismic_model.pth"

    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # The Training and Validation Loop
    for epoch in range(num_epochs):

        model.train()
        running_train_loss = 0.0
        running_train_loss_raw = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels_raw = labels.float().unsqueeze(1).to(device)

            # Apply Label Smoothing (0.0 -> 0.1, and 1.0 -> 0.9)
            labels = labels_raw * 0.8 + 0.1

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            # Unscale the gradients to apply the clipping speed limit
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_train_loss += loss.item() * inputs.size(0)

            # Diagnostic-only: BCE against the true (unsmoothed) labels, so this
            # is comparable to Val Loss below (which is also unsmoothed). Smoothed
            # train loss alone has a nonzero floor (~0.32) that unsmoothed val loss
            # doesn't, making the two misleading to compare directly otherwise.
            with torch.no_grad():
                raw_loss = F.binary_cross_entropy_with_logits(outputs.detach().float(), labels_raw)
            running_train_loss_raw += raw_loss.item() * inputs.size(0)

        avg_train_loss = running_train_loss / len(train_loader.dataset)
        avg_train_loss_raw = running_train_loss_raw / len(train_loader.dataset)

        # VALIDATION PHASE
        model.eval()
        running_val_loss = 0.0
        correct_preds = 0
        total_preds = 0

        val_all_labels = []
        val_all_probs = []
        val_all_preds = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.50).float() # Standard 0.5 decision threshold (matches final test-set eval's torch.round())

                correct_preds += (preds == labels).sum().item()
                total_preds += labels.size(0)

                val_all_labels.extend(labels.cpu().squeeze(1).tolist())
                val_all_probs.extend(probs.cpu().squeeze(1).tolist())
                val_all_preds.extend(preds.cpu().squeeze(1).tolist())

        avg_val_loss = running_val_loss / len(val_loader.dataset)
        val_accuracy = correct_preds / total_preds

        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(avg_val_loss)

        try:
            val_auc = roc_auc_score(val_all_labels, val_all_probs)
            val_mcc = matthews_corrcoef(val_all_labels, val_all_preds)
        except ValueError:
            val_auc = 0.0
            val_mcc = 0.0

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} (unsmoothed: {avg_train_loss_raw:.4f}) | Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Acc: {val_accuracy:.4f} | Val AUC: {val_auc:.4f} | Val MCC: {val_mcc:.4f}")

        if args.monitor == "loss":
            improved = avg_val_loss < best_metric
            current = avg_val_loss
        else:
            improved = val_auc > best_metric
            current = val_auc

        if improved:
            best_metric = current
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  => Best model saved to {save_path}! (val {args.monitor} = {current:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  => No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered! Validation {args.monitor} hasn't improved in {patience} epochs.")
            print(f"The best weights from the run have been saved to {save_path}.")
            break

    print(f"Saving full model at {save_path}...")
    torch.save(model, f"{args.save_dir}/full_model.pth")

    # Final Evaluation on Test Set
    print("\nRunning Final Evaluation on Test Set...")

    test_dataset = datasets.ImageFolder(f"{args.dataset_dir}/test", transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model.load_state_dict(torch.load(save_path))
    model.eval()

    correct_preds = 0
    total_preds = 0

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)

            probs = torch.sigmoid(outputs)
            preds = torch.round(probs)

            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

            all_labels.extend(labels.cpu().squeeze(1).tolist())
            all_probs.extend(probs.cpu().squeeze(1).tolist())
            all_preds.extend(preds.cpu().squeeze(1).tolist())

    test_accuracy = correct_preds / total_preds
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
    try:
        test_auc = roc_auc_score(all_labels, all_probs)
        print(f"Final Test AUC:      {test_auc:.4f}")
    except ValueError:
        pass

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix:")
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
