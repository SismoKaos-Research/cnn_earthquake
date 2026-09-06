"""
Trains the seismic classifier on spectrogram tensors (.pt) produced by
`seismic-cli generate-spectrogram-dataset`.

Shares its model and training loop with `cnn_train.py` via `training.py`, so
both entry points get the same fixes: matched val/test decision thresholds,
the unsmoothed train-loss diagnostic, val-AUC checkpointing, window-length
presets, seeding, and CPU support. Previously this script had its own
divergent copy of the loop and had fallen behind on all of them.

Usage:
    python cnn_from_tensor.py --dataset-dir dataset_spec_6s \\
        --save-dir trained_model_spec_6s --window-seconds 6

Not imported by anything else -- standalone script.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from sismokaos.training import (build_arg_parser, print_config, resolve_preset,
                                run_training)


class SeismicTensorDataset(Dataset):
    """
    Loads spectrogram tensors directly from disk, bypassing image loaders to
    preserve precision (the tensors are float dB values, not 8-bit pixels).

    Expects an ImageFolder-style layout:  <root>/<class_name>/<name>.pt
    """

    def __init__(self, root_dir, transform=None):
        """Indexes every .pt sample under `root_dir`'s class subdirectories.

        Args:
            root_dir: Directory with one subdirectory per class, each
                containing .pt spectrogram tensor files (`seismic-cli
                generate-spectrogram-dataset` output).
            transform: Optional callable applied to the loaded tensor
                before it's returned (e.g. RandomErasing for training).

        Raises:
            FileNotFoundError: If `root_dir` doesn't exist.
            RuntimeError: If no .pt files are found under `root_dir`.
        """
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset split not found: {self.root_dir}")
        self.transform = transform

        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for fpath in sorted(self.root_dir.rglob("*.pt")):
            cls_name = fpath.parent.name
            if cls_name in self.class_to_idx:
                self.samples.append((fpath, self.class_to_idx[cls_name]))

        if not self.samples:
            raise RuntimeError(f"No .pt tensors found under {self.root_dir}")

    def sample_shape(self):
        """Returns the tensor shape of the first sample, as a quick check.

        Returns:
            Shape tuple of the first sample's tensor.
        """
        return tuple(torch.load(self.samples[0][0], weights_only=True).shape)

    def validate_shapes(self, limit=None):
        """
        Every tensor must share one shape or the default collate throws mid-run.
        Datasets built before the generator resampled to a common rate mixed
        e.g. (3,129,94) and (3,129,47) -- stations recorded at different
        sampling rates -- so fail early here with a message that says why,
        rather than deep inside the first batch.

        Args:
            limit: If given, only checks the first `limit` samples instead
                of the full dataset (cheaper, useful for a quick smoke test).

        Returns:
            The shape tuple every checked sample matched.

        Raises:
            ValueError: If any checked sample's shape differs from the
                first sample's.
        """
        expected = self.sample_shape()
        paths = self.samples if limit is None else self.samples[:limit]
        for fpath, _ in paths:
            shape = tuple(torch.load(fpath, weights_only=True).shape)
            if shape != expected:
                raise ValueError(
                    f"Inconsistent tensor shapes: {self.samples[0][0].name} is {expected} "
                    f"but {fpath.name} is {shape}. Regenerate the dataset with "
                    f"`seismic-cli generate-spectrogram-dataset`, which resamples every "
                    f"window to a common rate so all tensors match."
                )
        return expected

    def __len__(self):
        """Returns the number of samples in this split."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns one (tensor, label) sample.

        Args:
            idx: Index into `self.samples`.

        Returns:
            Tuple of (spectrogram tensor, long label tensor).
        """
        fpath, label = self.samples[idx]
        tensor = torch.load(fpath, weights_only=True)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, torch.tensor(label, dtype=torch.long)


def main():
    """Loads the spectrogram-tensor dataset and runs `training.run_training`
    on `training.ImprovedSeismicCNN`."""
    parser = build_arg_parser(
        "Train ImprovedSeismicCNN on spectrogram tensors.",
        default_dataset_dir="./dataset_spectrogram",
        default_save_dir="trained_model_spectrogram",
    )
    args = resolve_preset(parser.parse_args())

    train_tf = None
    if args.random_erasing > 0:
        from torchvision import transforms

        # RandomErasing works directly on tensors. It is the only label-safe
        # augmentation here: flips would reverse time or invert frequency.
        train_tf = transforms.RandomErasing(p=args.random_erasing, scale=(0.02, 0.15))

    train_dataset = SeismicTensorDataset(f"{args.dataset_dir}/train", transform=train_tf)
    val_dataset = SeismicTensorDataset(f"{args.dataset_dir}/val")
    test_dataset = SeismicTensorDataset(f"{args.dataset_dir}/test")

    shape = train_dataset.validate_shapes()
    for name, ds in [("val", val_dataset), ("test", test_dataset)]:
        if ds.sample_shape() != shape:
            raise ValueError(f"{name} tensors are {ds.sample_shape()} but train is {shape}.")

    print_config(args, extra={"input_shape": shape, "classes": train_dataset.classes})
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    run_training(args, train_dataset, val_dataset, test_dataset, in_channels=shape[0])


if __name__ == "__main__":
    main()
