"""
Trains the seismic classifier on RAM images (PNG) produced by
`seismic-cli generate-dataset`.

The model and training loop live in `training.py`, shared with
`cnn_from_tensor.py` so the two entry points cannot drift apart again.
`ImprovedSeismicCNN`, `ResBlock` and `SEBlock` are re-exported here, so
existing imports (`from cnn_train import ImprovedSeismicCNN`) and existing
checkpoints keep working unchanged.

Usage:
    python cnn_train.py --dataset-dir dataset_6s_max \\
        --save-dir trained_model_6s --window-seconds 6

Also imported (not just run standalone): `cnn_run.py` imports
`ImprovedSeismicCNN`, `ResBlock`, `SEBlock` from this module (all
re-exported here from `training.py`, for the backward-compatibility reason
stated above -- older `full_model.pth` checkpoints may have been pickled
before the `training.py` refactor and so reference the legacy
`cnn_train.ImprovedSeismicCNN` module path); `cnn_run_from_state.py`
imports `ImprovedSeismicCNN` from this module to reconstruct the model
before loading a state-dict checkpoint into it.
"""

from torchvision import datasets, transforms

# Re-exported for backwards compatibility with cnn_run_from_state.py etc.
from seismolib.training import PRESETS  # noqa: F401
from seismolib.training import (SHORT_WINDOW_THRESHOLD_SEC, ImprovedSeismicCNN,
                                ResBlock, SEBlock, build_arg_parser,
                                print_config, resolve_preset, run_training)


def main():
    """Loads the RAM-image ImageFolder dataset and runs `training.run_training`
    on `training.ImprovedSeismicCNN`."""
    parser = build_arg_parser(
        "Train ImprovedSeismicCNN on RAM images.",
        default_dataset_dir="./dataset",
        default_save_dir="trained_model",
    )
    args = resolve_preset(parser.parse_args())

    # RandomErasing operates on tensors, so it sits after ToTensor. It is the
    # only label-safe image augmentation here: geometric flips/rotations would
    # scramble the RAM matrix's temporal ordering, since axis position IS time.
    train_tf = [transforms.ToTensor()]
    if args.random_erasing > 0:
        train_tf.append(transforms.RandomErasing(p=args.random_erasing, scale=(0.02, 0.15)))
    train_transform = transforms.Compose(train_tf)
    eval_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.ImageFolder(f"{args.dataset_dir}/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(f"{args.dataset_dir}/val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(f"{args.dataset_dir}/test", transform=eval_transform)

    print_config(args, extra={"classes": train_dataset.classes})
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    run_training(args, train_dataset, val_dataset, test_dataset, in_channels=3)


if __name__ == "__main__":
    main()
