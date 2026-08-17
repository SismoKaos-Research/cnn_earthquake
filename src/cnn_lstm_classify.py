"""
Dual-channel CNN + LSTM/self-attention (1D2D-EDL, Wang & Zhao 2025) for
earthquake-vs-noise classification on short arrival-anchored windows, rather
than the catalog forecasting task in `cnn_lstm.py`.

The script is **representation-agnostic**: it loads whatever `{seq, img}`
tensor pair the dataset directory holds, so the 2D channel's meaning is set
by which dataset you point it at, not by anything here.

    - `dataset_specdual_*` -> log-power SPECTROGRAM (current default; what
      every result in spectrogram_classifier_report.md and
      MAGNITUDE_CNN_CHEATSHEET.md was produced with)
    - `dataset_dual_*`     -> RAM image (legacy; still loads, no longer used)

    1D channel : LSTM -> multi-head self-attention over the raw standardized
                 (m, 3) Z/N/E waveform `seismic-cli generate-spec-dual-dataset`
                 writes -- per the paper's Sec. 3.3.1, NOT a reshaped version
                 of the 2D image (see seismic_cli/ram_dual.py's docstring for
                 the design mistake this corrected).
    2D channel : CNN over the (3, freq, time) spectrogram -- or the
                 (3, target_n, target_n) RAM image on a legacy dataset -- built
                 from the same window independently of the 1D channel.
    fusion     : --fusion linear (paper's default): F = a*F_1d + b*F_2d,
                 a fixed pair of scalars learned jointly with both branches.
                 --fusion gate: F = g(x)*F_1d + (1-g(x))*F_2d, a per-example
                 gate (see cnn_lstm.GatedFusion) -- built because linear
                 fusion measurably underperformed the best single branch on
                 two independent datasets (report.md 10.5.1).
    head       : single logit, earthquake vs. noise (BCEWithLogitsLoss).

`--channels 1d`/`2d` ablate either branch (fusion is then a no-op).
`--channels 2d` alone is architecturally close to the existing single-branch
CNN classifier (`cnn_train.py`), so it doubles as that baseline's comparison
point here -- and on the spectrogram datasets it is also the BEST known
configuration: every fusion variant scored lower (0.9793 AUC for `2d` alone
vs 0.9743-0.9761 for the fusion variants, spectrogram_classifier_report.md).

Training conventions (label smoothing, unsmoothed-loss diagnostic, AMP, val
AUC/MCC, matched 0.5 threshold) match `training.py`, the shared core for the
image-only classifiers -- kept as its own loop rather than forced through
that module, since it assumes a single-tensor model, not two paired inputs.

Usage:
    # the headline detector: 0.9786 +/- 0.0014 AUC over seeds 42/43/44
    python cnn_lstm_classify.py --dataset-dir dataset_specdual_6s --channels 2d --batch-size 32
    python cnn_lstm_classify.py --dataset-dir dataset_specdual_6s --fusion gate  # gated fusion
    python cnn_lstm_classify.py --dataset-dir dataset_dual_6s                    # legacy RAM

Note `RamDualTensorDataset` and `RamDualEncoder` keep "Ram" in their names for
import compatibility (cnn_lstm_stack.py and seismic_cli/ram_dual.py both
reference them); the names are historical and neither class assumes a RAM
image -- both take whatever 2D tensor the dataset supplies.

Also imported (not just run standalone): cnn_lstm_stack.py imports
`DualChannelBinaryNet` and `RamDualTensorDataset` from this module, loading
this script's checkpoint (via `--ckpt-1d`/`--ckpt-2d`) as one branch of a
stacked ensemble.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (classification_report, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)
from torch.utils.data import DataLoader, Dataset

from seismolib.metrics import (binary_report, majority_class_baseline, print_report,
                     safe_auc)
from seismolib.model.dual_channel import DualChannelNet
from seismolib.training import seed_everything

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class RamDualTensorDataset(Dataset):
    """
    Loads the {seq, img} tensors written by `seismic-cli generate-dual-dataset`.

    Expects an ImageFolder-style layout: <root>/<class_name>/<name>.pt
    Class directories sort as "00_noise" < "01_earthquake", so label 1 is
    always earthquake -- matching BCEWithLogitsLoss's positive-class convention.
    """

    def __init__(self, root_dir):
        """Indexes every .pt sample under `root_dir`'s class subdirectories.

        Args:
            root_dir: Directory with one subdirectory per class, each
                containing .pt files (`seismic-cli generate-dual-dataset`
                output), e.g. `<root>/00_noise/`, `<root>/01_earthquake/`.

        Raises:
            FileNotFoundError: If `root_dir` doesn't exist.
            RuntimeError: If no .pt files are found under `root_dir`.
        """
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset split not found: {self.root_dir}")

        self.classes = sorted(d.name for d in self.root_dir.iterdir() if d.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for fpath in sorted(self.root_dir.rglob("*.pt")):
            cls_name = fpath.parent.name
            if cls_name in self.class_to_idx:
                self.samples.append((fpath, self.class_to_idx[cls_name]))
        if not self.samples:
            raise RuntimeError(f"No .pt tensors found under {self.root_dir}")

    def sample_shapes(self):
        """Returns (seq_shape, img_shape) of the first sample, as a quick check.

        Returns:
            Tuple of (seq tensor shape tuple, img tensor shape tuple).
        """
        d = torch.load(self.samples[0][0], weights_only=True)
        return tuple(d["seq"].shape), tuple(d["img"].shape)

    def validate_shapes(self, limit=None):
        """
        Every tensor must share one shape or the default collate throws
        mid-run -- see `SeismicTensorDataset.validate_shapes` in
        cnn_from_tensor.py for the spectrogram-side version of this same
        problem (mixed station sampling rates), which `RamDualEncoder`
        avoids the same way (resample to a nominal rate before reshaping).

        Args:
            limit: If given, only checks the first `limit` samples instead
                of the full dataset (cheaper, useful for a quick smoke test).

        Returns:
            Tuple of (seq_shape, img_shape) that every checked sample matched.

        Raises:
            ValueError: If any checked sample's seq or img shape differs
                from the first sample's.
        """
        seq_shape, img_shape = self.sample_shapes()
        paths = self.samples if limit is None else self.samples[:limit]
        for fpath, _ in paths:
            d = torch.load(fpath, weights_only=True)
            s, i = tuple(d["seq"].shape), tuple(d["img"].shape)
            if s != seq_shape or i != img_shape:
                raise ValueError(
                    f"Inconsistent tensor shapes at {fpath.name}: seq {s} (expected "
                    f"{seq_shape}), img {i} (expected {img_shape}). Regenerate with "
                    f"`seismic-cli generate-dual-dataset`.")
        return seq_shape, img_shape

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns one (seq, img, label) sample.

        Args:
            idx: Index into `self.samples`.

        Returns:
            Tuple of (float32 seq tensor, float32 img tensor, float32
            scalar label tensor).
        """
        fpath, label = self.samples[idx]
        d = torch.load(fpath, weights_only=True)
        return d["seq"].float(), d["img"].float(), torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DualChannelBinaryNet(DualChannelNet):
    """Same fusion design as `cnn_lstm.py`'s DualChannelRiskNet, minus the
    auxiliary-scalar branch (there are no catalog-style physical scalars
    here) and with a single-logit binary head instead of a 3-way softmax.

    `fusion="linear"` is the paper's a*F1+b*F2 (two global scalars).
    `fusion="gate"` replaces it with `model.blocks.GatedFusion`, a per-example
    gate -- only meaningful when both branches are active (`channels="all"`);
    single-branch ablations ignore `fusion` entirely since there is nothing
    to combine."""

    def __init__(self, seq_dim, img_channels, hidden=64, fusion_dim=128,
                dropout=0.3, channels="all", fusion="linear"):
        """See `DualChannelNet.__init__` (`aux_dim=0`, `n_classes=1`,
        `squeeze_output=False` always here -- the caller squeezes/unsqueezes
        as needed to match `BCEWithLogitsLoss`'s (batch, 1) convention)."""
        super().__init__(seq_dim, img_channels, aux_dim=0, hidden=hidden,
                         fusion_dim=fusion_dim, dropout=dropout, channels=channels,
                         fusion=fusion, n_classes=1, squeeze_output=False)

    def forward(self, seq, img):
        """See `DualChannelNet.forward` (`aux=None` always here).

        Returns:
            Tensor of shape (batch, 1) -- a single raw logit per sample.
        """
        return super().forward(seq, img)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    """Parses command-line arguments.

    Returns:
        argparse.Namespace with the script's CLI options.
    """
    p = argparse.ArgumentParser(description="Dual-channel CNN+LSTM earthquake/noise "
                                            "classifier (spectrogram or legacy RAM 2D "
                                            "channel + raw-waveform 1D channel).")
    p.add_argument("--dataset-dir", required=True,
                   help="Directory from `seismic-cli generate-spec-dual-dataset` "
                        "(spectrogram 2D channel, current default) or "
                        "`generate-dual-dataset` (legacy RAM images). The 2D "
                        "representation is decided here, not by any flag.")
    p.add_argument("--save-dir", default="trained_model_cnnlstm_classify")
    p.add_argument("--channels", default="all", choices=["all", "1d", "2d"],
                   help="Ablation switch: 'all' is the full dual-channel model, "
                        "'1d' is LSTM+attention only, '2d' is CNN-only (close to "
                        "the existing image-only classifier's architecture).")
    p.add_argument("--fusion", default="linear", choices=["linear", "gate"],
                   help="linear: paper's a*F1+b*F2 (two global scalars). gate: "
                        "per-example gate g(x)*F1+(1-g(x))*F2 (report.md 10.5.1/10.5.2 "
                        "-- linear fusion underperformed the best single branch on "
                        "both RAM and spectrogram 2D representations). Only affects "
                        "--channels all.")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=3e-2)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--fusion-dim", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42,
                   help="Single-seed shorthand. Ignored if --ensemble-seeds or "
                        "--random-seeds is given.")
    p.add_argument("--ensemble-seeds", type=str, default=None,
                   help="Comma-separated seeds to train and ensemble, e.g. '42,43,44'. "
                        "Reports each seed's test AUC, their spread, and the "
                        "probability-averaged ensemble. Matches "
                        "cnn_lstm_catalog_waveform_fusion.py's flag of the same name.")
    p.add_argument("--random-seeds", type=int, default=None, metavar="N",
                   help="Draw N random seeds instead of --ensemble-seeds. Fixed seeds "
                        "sample run-to-run variance exactly once and then hide it. The "
                        "drawn seeds print as a ready-to-paste --ensemble-seeds value.")
    p.add_argument("--num-workers", type=int, default=4)
    a = p.parse_args()
    if a.ensemble_seeds is None and a.random_seeds is None:
        a.ensemble_seeds = str(a.seed)
    return a


def trivial_amplitude_floor(test_ds, y_ref):
    """
    AUC of single amplitude statistics read straight off the test tensors.

    These windows are separated mostly by how loud they are, so "how loud is
    it" is the baseline a learned detector actually has to beat -- not the
    majority class. Measured on the 6s spectrogram corpus, `seq abs-max` alone
    reaches ~0.95 while the majority-class bar sits at 0.50, so quoting the
    latter overstates a model's edge by an order of magnitude.

    One pass over the test split (~9.5k tensors); trivial next to training.
    Returns {name: oriented AUC}.
    """
    seq_std, seq_absmax, img_mean = [], [], []
    for fpath, _lbl in test_ds.samples:
        d = torch.load(fpath, weights_only=True)
        s = d["seq"].float()
        seq_std.append(float(s.std()))
        seq_absmax.append(float(s.abs().max()))
        img_mean.append(float(d["img"].float().mean()))
    out = {}
    for name, vals in (("seq std", seq_std), ("seq abs-max", seq_absmax),
                       ("img mean dB", img_mean)):
        out[name] = safe_auc(y_ref, np.asarray(vals), oriented=True)
    return out


def train_one_seed(args, seed, train_ds, val_ds, test_ds, seq_shape, img_shape, device):
    """Trains and evaluates one seed, mirroring `cnn_lstm_catalog_waveform_fusion
    .train_one_seed` so both branches of the project have the same shape.

    The training body is unchanged from the single-seed version that produced
    0.9793 -- this only lifts it into a function so seeds can be looped and
    ensembled. Seed 42 must still reproduce 0.9793/0.8667/0.9328 exactly; if it
    does not, this refactor changed behaviour and is wrong.

    Returns:
        Tuple of (y_true, y_score, y_pred, gates, n_params) for the test split,
        from the best-val-AUC epoch's weights.
    """
    seed_everything(seed)
    model = DualChannelBinaryNet(seq_shape[-1], img_shape[0], hidden=args.hidden,
                                 fusion_dim=args.fusion_dim, dropout=args.dropout,
                                 channels=args.channels, fusion=args.fusion).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers, pin_memory=True)
    train_loader, val_loader, test_loader = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    # The checkpoint name must identify the RUN, not just the seed. Seed alone was
    # not enough: two runs launched concurrently with the same seeds wrote the same
    # files, and each then reloaded the other's weights at the end of training.
    # When the architectures differed that surfaced as a state_dict KeyError; when
    # they matched it was silent, and a seed scored 0.2480 -- an inverted model,
    # reported as if it were a training outcome. Config plus dataset identity plus
    # PID makes collision impossible even for two identical commands run at once.
    run_tag = (f"{args.channels}_{args.fusion}"
               f"_{os.path.basename(os.path.normpath(args.dataset_dir))}_pid{os.getpid()}")
    save_path = os.path.join(args.save_dir,
                             f"best_cnnlstm_classify_{run_tag}_seed{seed}.pth")
    best_auc, no_improve = -1.0, 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = running_loss_raw = 0.0
        for seq, img, labels in train_loader:
            seq, img = seq.to(device), img.to(device)
            labels_raw = labels.unsqueeze(1).to(device)
            labels_smooth = labels_raw * 0.8 + 0.1  # 0 -> 0.1, 1 -> 0.9

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(seq, img)
                loss = criterion(outputs, labels_smooth)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += loss.item() * labels.size(0)
            with torch.no_grad():
                raw = F.binary_cross_entropy_with_logits(outputs.detach().float(), labels_raw)
            running_loss_raw += raw.item() * labels.size(0)
        scheduler.step()

        avg_loss = running_loss / len(train_loader.dataset)
        avg_loss_raw = running_loss_raw / len(train_loader.dataset)

        model.eval()
        v_labels, v_probs, v_preds = [], [], []
        correct = total = 0
        with torch.no_grad():
            for seq, img, labels in val_loader:
                seq, img = seq.to(device), img.to(device)
                labels_dev = labels.unsqueeze(1).to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(seq, img)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.50).float()
                correct += (preds == labels_dev).sum().item()
                total += labels.size(0)
                v_labels.extend(labels.tolist())
                v_probs.extend(probs.float().cpu().squeeze(1).tolist())
                v_preds.extend(preds.float().cpu().squeeze(1).tolist())

        val_acc = correct / max(1, total)
        try:
            val_auc = roc_auc_score(v_labels, v_probs)
            val_mcc = matthews_corrcoef(v_labels, v_preds)
        except ValueError:
            val_auc = val_mcc = 0.0

        print(f"  [seed {seed}] epoch {epoch+1}/{args.epochs} val AUC {val_auc:.4f} "
              f"val loss {avg_loss_raw:.4f} val acc {val_acc:.4f} mcc {val_mcc:.4f}")

        if val_auc > best_auc:
            best_auc, no_improve = val_auc, 0
            torch.save(model.state_dict(), save_path)
            print(f"  => saved (val AUC {best_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping: val AUC flat for {args.patience} epochs.")
                break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()
    all_labels, all_probs, all_preds, all_gates = [], [], [], []
    with torch.no_grad():
        for seq, img, labels in test_loader:
            seq, img = seq.to(device), img.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(seq, img)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.50).float()
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.float().cpu().squeeze(1).tolist())
            all_preds.extend(preds.float().cpu().squeeze(1).tolist())
            if getattr(model, "last_gate", None) is not None:
                all_gates.extend(model.last_gate.float().cpu().squeeze(1).tolist())

    all_labels = np.array(all_labels); all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    print(f"  [seed {seed}] test AUC {safe_auc(all_labels, all_probs):.4f}")

    if getattr(model, "use_1d", False) and getattr(model, "use_2d", False):
        if args.fusion == "gate":
            g = np.array(all_gates)
            correct = (all_labels == all_preds)
            print(f"\ngate g (1 favors 1D, 0 favors 2D): mean {g.mean():.3f}  std {g.std():.3f}"
                  f"\n  earthquake windows: mean g {g[all_labels==1].mean():.3f}"
                  f"\n  noise windows:      mean g {g[all_labels==0].mean():.3f}"
                  f"\n  correct predictions: mean g {g[correct].mean():.3f}"
                  f"\n  wrong predictions:   mean g {g[~correct].mean() if (~correct).any() else float('nan'):.3f}")
        else:
            print(f"\nlearned fusion weights: 1D={model.w1.item():+.3f}  2D={model.w2.item():+.3f}"
                  "\n(relative magnitude indicates which representation the model leaned on)")

    return all_labels, all_probs, all_preds, all_gates, n_params


def main():
    """Loads the dual-tensor dataset once, trains every seed on it, and reports
    per-seed metrics, their spread, and the probability-averaged ensemble against
    the majority-class floor."""
    args = parse_args()

    train_ds = RamDualTensorDataset(f"{args.dataset_dir}/train")
    val_ds = RamDualTensorDataset(f"{args.dataset_dir}/val")
    test_ds = RamDualTensorDataset(f"{args.dataset_dir}/test")

    seq_shape, img_shape = train_ds.validate_shapes()
    for name, ds in (("val", val_ds), ("test", test_ds)):
        s, i = ds.sample_shapes()
        if s != seq_shape or i != img_shape:
            raise ValueError(f"{name} tensors are seq {s} img {i}, but train is "
                             f"seq {seq_shape} img {img_shape}.")

    print("=" * 64)
    print(f"Dual-channel event/noise classifier | channels='{args.channels}'")
    print(f"  seq {seq_shape} | img {img_shape}")
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        n_pos = sum(1 for _, lbl in ds.samples if lbl == ds.class_to_idx.get("01_earthquake", 1))
        print(f"  {name:5s}: n={len(ds):6d}  earthquake={n_pos}  noise={len(ds) - n_pos}")
    print("=" * 64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.random_seeds:
        # Fixed seeds sample run-to-run variance once and then hide it. Drawn seeds
        # are printed as a paste-ready --ensemble-seeds so the run stays replayable.
        seeds = [int(s) for s in
                 np.random.default_rng().integers(0, 2**31 - 1, size=args.random_seeds)]
        print(f"  [random-seeds] drew {len(seeds)} seeds: "
              f"--ensemble-seeds {','.join(str(s) for s in seeds)}")
    else:
        seeds = [int(s) for s in args.ensemble_seeds.split(",")]
    print(f"Device: {device} | training {len(seeds)} seed(s): {seeds}")

    per_seed_probs, per_seed_preds, y_ref, n_params = [], [], None, None
    for seed in seeds:
        y, probs, preds, _gates, n_params = train_one_seed(
            args, seed, train_ds, val_ds, test_ds, seq_shape, img_shape, device)
        if y_ref is None:
            y_ref = y
        per_seed_probs.append(probs)
        per_seed_preds.append(preds)

    print(f"\n  model parameters: {n_params:,} | train samples: {len(train_ds)} "
          f"({n_params / max(1, len(train_ds)):.1f} params/sample)")

    print("\n--- Floors (test set) ---")
    # Labels come from ds.samples rather than by indexing the dataset -- __getitem__
    # would load 50k tensors off disk just to read an int.
    train_labels = np.array([lbl for _, lbl in train_ds.samples])
    maj, maj_acc, maj_bal = majority_class_baseline(train_labels, y_ref)
    print(f"  majority-class ({maj})   accuracy {maj_acc:.4f}  balanced {maj_bal:.4f}  "
          f"AUC 0.5000   n={len(y_ref)}")

    # The majority-class bar is vacuous on a balanced set, and quoting it made an
    # earlier result look like +0.48 when it was worth +0.04. These windows are
    # separated mostly by loudness, so the honest bar is what a single amplitude
    # scalar achieves with no learning at all. Oriented (max(a, 1-a)) because an
    # anti-predictive rule is just as exploitable as a predictive one.
    amp_floor = trivial_amplitude_floor(test_ds, y_ref)
    for name, auc_val in amp_floor.items():
        print(f"  {name:<22s} AUC {auc_val:.4f}   (single scalar, no learning)")
    best_floor_name, best_floor = max(amp_floor.items(), key=lambda kv: kv[1])
    print(f"  -> strongest trivial floor: {best_floor_name} at {best_floor:.4f}")

    # Per-seed spread is the headline reliability number. A tight spread is what
    # separates a result from a lucky draw -- the forecasting branch's spread ran
    # ~0.17, which is how a single good seed carried an ensemble for a whole day.
    per_seed_aucs = [safe_auc(y_ref, p) for p in per_seed_probs]
    print(f"\n  per-seed test AUC: {[f'{a:.4f}' for a in per_seed_aucs]}")
    print(f"    mean {np.mean(per_seed_aucs):.4f}  std {np.std(per_seed_aucs):.4f}  "
          f"spread {max(per_seed_aucs) - min(per_seed_aucs):.4f}")

    # A seed below chance is anti-predictive, and averaging its probabilities into
    # the ensemble drags the ensemble below its own members -- which is how a run
    # once reported an ensemble AUC of 0.9108 while two of three seeds scored
    # above 0.94. Treat it as a failed run to investigate, never as a data point:
    # the usual causes are a corrupted/clobbered checkpoint or a diverged seed.
    inverted = [(s, a) for s, a in zip(seeds, per_seed_aucs) if a < 0.5]
    if inverted:
        print("\n  " + "!" * 60)
        for s, a in inverted:
            print(f"  !! seed {s} scored AUC {a:.4f} -- BELOW CHANCE (anti-predictive).")
        print(f"  !! {len(inverted)}/{len(seeds)} seed(s) failed. The ensemble below "
              f"averages them in and is NOT a valid result.")
        print(f"  !! Check for a clobbered checkpoint (another run sharing --save-dir) "
              f"or divergence in that seed's training curve.")
        print("  " + "!" * 60)

    ensemble_probs = np.mean(per_seed_probs, axis=0)
    ensemble_preds = (ensemble_probs > 0.5).astype(float)
    report = binary_report(y_ref, ensemble_probs, y_pred=ensemble_preds)
    print_report(f"Event/noise detector [channels={args.channels}, fusion={args.fusion}] "
                 f"({len(seeds)}-seed ensemble, test set)", report)
    edge = report["roc_auc"] - best_floor
    print(f"\n  ROC-AUC {report['roc_auc']:.4f}  vs majority-class floor 0.5000  "
          f"-> {'BEATS' if report['roc_auc'] > 0.5 else 'AT/BELOW'} floor")
    print(f"  ROC-AUC {report['roc_auc']:.4f}  vs {best_floor_name} floor {best_floor:.4f}  "
          f"-> {'+' if edge >= 0 else ''}{edge:.4f}   <- the number that matters! look at this!!!")

    print("\nClassification Report (ensemble):")
    print(classification_report(y_ref, ensemble_preds, digits=4))


if __name__ == "__main__":
    main()
