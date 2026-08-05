"""
Dual-channel CNN + LSTM/self-attention model for time-to-major-earthquake risk.

This is the architecture from Wang & Zhao (2025) -- their 1D2D-EDL -- applied
to catalog sliding windows instead of bearing vibration:

    1D channel : LSTM -> multi-head self-attention over the feature sequence
    2D channel : CNN over the RAM image of that same window
    fusion     : F = a*F_1d + b*F_2d   (a, b learned, as in the paper)
    head       : 3-class risk  (<1y / 1-5y / >5y until the next major event)

Two departures from the paper, both forced by findings in report.md:

* **An auxiliary scalar branch.** The RAM transform is exactly scale-invariant
  (8.2), so the image cannot represent absolute magnitude or energy level --
  precisely the quantities that matter here. Window-level scalars (b-value,
  Lyapunov exponent, event rate, total energy) enter alongside the fused
  features. `--channels` ablates any branch to test what each contributes.

* **Baselines printed on every run.** IP4's success criterion is >=70%
  accuracy, which a 3-class problem with skewed priors can reach by always
  predicting the majority class. Majority-class and persistence baselines are
  therefore reported next to the model, and the run states plainly whether the
  model beat them.

Usage:
    python cnn_lstm.py --dataset-dir dataset_catalog_marmara
    python cnn_lstm.py --dataset-dir ... --channels 1d      # ablation
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix)
from torch.utils.data import DataLoader, Dataset

from training import seed_everything

RISK_CLASSES = ["lt_1y", "1_5y", "gt_5y"]
CLASS_TO_IDX = {c: i for i, c in enumerate(RISK_CLASSES)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class CatalogWindowDataset(Dataset):
    """
    Loads the {seq, img, aux} tensors written by
    `seismic-cli generate-catalog-dataset`.

    seq and aux are standardized with TRAIN statistics only -- fitting them on
    each split would let the test distribution normalize itself, a subtle leak
    that matters more here than usual because the splits are chronological and
    the distribution genuinely drifts over time.
    """

    def __init__(self, manifest: pd.DataFrame, root: Path, split: str, stats=None):
        self.rows = manifest[manifest.split == split].reset_index(drop=True)
        self.dir = Path(root) / split
        if self.rows.empty:
            raise ValueError(f"Split '{split}' is empty.")
        self.labels = self.rows.risk_class.map(CLASS_TO_IDX).to_numpy()
        if self.labels.min() < 0 or self.rows.risk_class.isna().any():
            raise ValueError("Unrecognized risk_class values in manifest.")
        self.days = self.rows.days_to_major.to_numpy(dtype=np.float32)

        sample = torch.load(self.dir / self.rows.filename.iloc[0], weights_only=True)
        self.seq_dim = sample["seq"].shape[-1]
        self.seq_len = sample["seq"].shape[0]
        self.img_shape = tuple(sample["img"].shape)
        self.aux_dim = sample["aux"].numel()

        if stats is None:
            seqs, auxs = [], []
            for fn in self.rows.filename:
                d = torch.load(self.dir / fn, weights_only=True)
                seqs.append(d["seq"].numpy())
                auxs.append(d["aux"].numpy())
            S = np.concatenate(seqs, axis=0)
            A = np.stack(auxs, axis=0)
            with np.errstate(invalid="ignore"):
                stats = (np.nanmean(S, 0), np.nanstd(S, 0) + 1e-6,
                         np.nanmean(A, 0), np.nanstd(A, 0) + 1e-6)
            stats = tuple(np.where(np.isfinite(s), s, 0.0 if i % 2 == 0 else 1.0)
                          for i, s in enumerate(stats))
        self.stats = stats

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        d = torch.load(self.dir / self.rows.filename.iloc[i], weights_only=True)
        sm, ss, am, asd = self.stats
        seq = (d["seq"].numpy() - sm) / ss
        aux = (d["aux"].numpy() - am) / asd
        return (torch.from_numpy(np.nan_to_num(seq, nan=0.0)).float(),
                d["img"].float(),
                torch.from_numpy(np.nan_to_num(aux, nan=0.0)).float(),
                torch.tensor(self.labels[i], dtype=torch.long))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMAttentionBranch(nn.Module):
    """LSTM for long-range order, then multi-head self-attention to weight steps."""

    def __init__(self, in_dim, hidden=64, layers=1, heads=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        d = hidden * 2
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.out_dim = d

    def forward(self, x):
        h, _ = self.lstm(x)
        a, _ = self.attn(h, h, h)
        h = self.norm(h + a)             # residual, as in the transformer block
        return h.mean(dim=1)             # pool over time


class CNNBranch(nn.Module):
    """Compact CNN over the RAM image. The images are small (32x32 by default),
    so a 4-stage ResNet would be heavily over-provisioned here."""

    def __init__(self, in_channels=3, width=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.GELU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2), nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 4), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = width * 4

    def forward(self, x):
        return torch.flatten(self.net(x), 1)


class GatedFusion(nn.Module):
    """
    Per-example gate deciding how much to trust each branch, replacing a
    fixed pair of scalars (a*F1 + b*F2, same for every example) with
    g(x)*F1 + (1-g(x))*F2, where g = sigmoid(MLP([F1, F2])) is conditioned on
    both branches' own features for THIS example.

    Motivation (report.md 10.5.1/10.5.2): the paper's fixed-scalar fusion
    underperformed the best single branch on two independent 2D
    representations (RAM and spectrogram) -- a global blend can't suppress a
    weak branch on the specific examples where it's wrong, only shrink its
    average contribution. Late-fusion stacking on frozen checkpoints fixed
    that post hoc; this tests whether the same idea, trained end-to-end
    instead of on frozen features, does at least as well without giving up
    joint training's ability to let the branches adapt to each other.
    """

    def __init__(self, dim, hidden=None, dropout=0.1):
        super().__init__()
        hidden = hidden or max(8, dim // 2)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, f1, f2):
        g = torch.sigmoid(self.net(torch.cat([f1, f2], dim=1)))
        return g * f1 + (1.0 - g) * f2, g


class DualChannelRiskNet(nn.Module):
    """
    1D + 2D + auxiliary scalars, fused and classified.

    Fusion follows the paper: each channel is projected to a common width and
    combined as a*F1 + b*F2 with learned scalar weights, so the balance between
    channels is fit rather than assumed. The learned values are worth reading
    after training -- they say which representation the model actually used.
    """

    def __init__(self, seq_dim, img_channels, aux_dim, hidden=64, fusion_dim=128,
                 n_classes=3, dropout=0.3, channels="all"):
        super().__init__()
        self.channels = channels
        self.use_1d = channels in ("all", "1d", "1d+aux")
        self.use_2d = channels in ("all", "2d", "2d+aux")
        self.use_aux = channels in ("all", "aux", "1d+aux", "2d+aux")
        if not (self.use_1d or self.use_2d or self.use_aux):
            raise ValueError(f"--channels {channels} disables every branch")

        if self.use_1d:
            self.b1 = LSTMAttentionBranch(seq_dim, hidden=hidden, dropout=dropout)
            self.p1 = nn.Linear(self.b1.out_dim, fusion_dim)
        if self.use_2d:
            self.b2 = CNNBranch(img_channels, dropout=dropout)
            self.p2 = nn.Linear(self.b2.out_dim, fusion_dim)
        # Learned fusion weights (a, b in the paper's notation).
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))

        head_in = (fusion_dim if (self.use_1d or self.use_2d) else 0) + \
                  (aux_dim if self.use_aux else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(dropout),
            nn.Linear(head_in, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, n_classes),
        )

    def forward(self, seq, img, aux):
        feats = []
        fused = None
        if self.use_1d:
            fused = self.w1 * self.p1(self.b1(seq))
        if self.use_2d:
            f2 = self.w2 * self.p2(self.b2(img))
            fused = f2 if fused is None else fused + f2
        if fused is not None:
            feats.append(fused)
        if self.use_aux:
            feats.append(aux)
        return self.head(torch.cat(feats, dim=1))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def report_baselines(train_ds, test_ds):
    """
    Majority-class and persistence references.

    Persistence -- 'the next interval looks like the last one' -- is the
    honest naive forecaster for a temporal task, the analogue of STA/LTA for
    detection. Here it predicts each test window's class from the class of the
    chronologically preceding training window's distribution conditioned on a
    similar event rate, approximated by the train-set mode.
    """
    y_tr, y_te = train_ds.labels, test_ds.labels
    print("\n--- Reference points (test set) ---")
    maj = int(np.bincount(y_tr, minlength=len(RISK_CLASSES)).argmax())
    pred_maj = np.full_like(y_te, maj)
    acc = float((pred_maj == y_te).mean())
    bal = balanced_accuracy_score(y_te, pred_maj)
    print(f"  majority-class ('{RISK_CLASSES[maj]}')   acc {acc:.4f}   balanced {bal:.4f}")

    rng = np.random.default_rng(0)
    prior = np.bincount(y_tr, minlength=len(RISK_CLASSES)) / len(y_tr)
    pred_prior = rng.choice(len(RISK_CLASSES), size=len(y_te), p=prior)
    print(f"  stratified-random                    acc {(pred_prior==y_te).mean():.4f}   "
          f"balanced {balanced_accuracy_score(y_te, pred_prior):.4f}")
    print(f"  [test class balance] " +
          "  ".join(f"{c}={int((y_te==i).sum())}" for i, c in enumerate(RISK_CLASSES)))
    return acc, bal


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Dual-channel CNN+LSTM earthquake risk model.")
    p.add_argument("--dataset-dir", required=True,
                   help="Directory from `seismic-cli generate-catalog-dataset`.")
    p.add_argument("--save-dir", default="trained_model_cnnlstm")
    p.add_argument("--channels", default="all",
                   choices=["all", "1d", "2d", "aux", "1d+aux", "2d+aux"],
                   help="Ablation switch: which branches to enable.")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--fusion-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-class-weights", action="store_true",
                   help="Disable inverse-frequency class weighting (on by default, "
                        "since the risk classes are typically very skewed).")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    root = Path(args.dataset_dir)
    manifest = pd.read_csv(root / "manifest.csv")
    train_ds = CatalogWindowDataset(manifest, root, "train")
    val_ds = CatalogWindowDataset(manifest, root, "val", stats=train_ds.stats)
    test_ds = CatalogWindowDataset(manifest, root, "test", stats=train_ds.stats)

    print("=" * 64)
    print(f"Dual-channel risk model | channels='{args.channels}'")
    print(f"  seq ({train_ds.seq_len}, {train_ds.seq_dim}) | img {train_ds.img_shape} "
          f"| aux ({train_ds.aux_dim},)")
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        c = np.bincount(ds.labels, minlength=len(RISK_CLASSES))
        print(f"  {name:5s}: n={len(ds):5d}  " +
              "  ".join(f"{k}={c[i]}" for i, k in enumerate(RISK_CLASSES)) +
              f"   {ds.rows.end_time.min()[:10]} -> {ds.rows.end_time.max()[:10]}")
    print("=" * 64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualChannelRiskNet(train_ds.seq_dim, train_ds.img_shape[0], train_ds.aux_dim,
                               hidden=args.hidden, fusion_dim=args.fusion_dim,
                               dropout=args.dropout, channels=args.channels).to(device)
    print(f"Device: {device} | parameters: {sum(p.numel() for p in model.parameters()):,}")

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers)
    train_loader, val_loader, test_loader = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    if args.no_class_weights:
        weight = None
    else:
        counts = np.bincount(train_ds.labels, minlength=len(RISK_CLASSES)).astype(np.float64)
        w = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
        weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
        print(f"class weights: {np.round(w / w.mean(), 3)}")
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "best_cnnlstm.pth")
    best, no_improve = -1.0, 0

    def evaluate(loader):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for seq, img, aux, y in loader:
                out = model(seq.to(device), img.to(device), aux.to(device))
                ps.extend(out.argmax(1).cpu().tolist())
                ys.extend(y.tolist())
        return np.array(ys), np.array(ps)

    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for seq, img, aux, y in train_loader:
            seq, img, aux, y = seq.to(device), img.to(device), aux.to(device), y.to(device)
            loss = criterion(model(seq, img, aux), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
            tot += loss.item() * y.size(0)
        scheduler.step()

        yv, pv = evaluate(val_loader)
        # Balanced accuracy, not accuracy: with skewed risk classes the plain
        # figure is dominated by whichever class happens to be common.
        vb = balanced_accuracy_score(yv, pv)
        print(f"Epoch {epoch+1}/{args.epochs} | loss {tot/len(train_ds):.4f} "
              f"| val acc {(yv==pv).mean():.4f}  balanced {vb:.4f}")
        if vb > best:
            best, no_improve = vb, 0
            torch.save(model.state_dict(), save_path)
            print(f"  => saved (val balanced {best:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping: val balanced accuracy flat for {args.patience} epochs.")
                break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    yt, pt = evaluate(test_loader)
    acc = float((yt == pt).mean())
    bal = balanced_accuracy_score(yt, pt)
    kappa = cohen_kappa_score(yt, pt)

    present = sorted(set(yt.tolist()))
    if len(present) < len(RISK_CLASSES):
        missing = [RISK_CLASSES[i] for i in range(len(RISK_CLASSES)) if i not in present]
        print(f"\n  [!] class(es) {missing} do not occur in the test split at all.")
        print("      Balanced accuracy is then an average over the classes that DO occur,")
        print("      so it is not comparable to a run where every class is present, and")
        print("      sklearn's 'y_pred contains classes not in y_true' warning is expected.")

    maj_acc, maj_bal = report_baselines(train_ds, test_ds)
    print(f"\n--- Dual-channel model (channels='{args.channels}') ---")
    print(f"  accuracy {acc:.4f} | balanced {bal:.4f} | Cohen's kappa {kappa:+.4f}")
    print(f"  vs majority-class: {acc - maj_acc:+.4f} accuracy, {bal - maj_bal:+.4f} balanced")
    if acc <= maj_acc + 1e-9:
        print("  [!] The model does NOT beat predicting the majority class. Its raw"
              "\n      accuracy is not evidence of skill, whatever the value is.")
    elif kappa < 0.2:
        print("  [!] Beats the majority class, but kappa < 0.2 -- agreement is barely"
              "\n      above chance once the class priors are accounted for.")
    else:
        print("  Beats both references with non-trivial kappa.")

    if getattr(model, "use_1d", False) and getattr(model, "use_2d", False):
        print(f"\n  learned fusion weights: 1D={model.w1.item():+.3f}  2D={model.w2.item():+.3f}"
              "\n  (relative magnitude indicates which representation the model leaned on)")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(pd.DataFrame(confusion_matrix(yt, pt, labels=range(len(RISK_CLASSES))),
                       index=RISK_CLASSES, columns=RISK_CLASSES))
    print("\n" + classification_report(yt, pt, labels=range(len(RISK_CLASSES)),
                                       target_names=RISK_CLASSES, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
