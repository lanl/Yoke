import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def load_records(pattern):
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    arrays = []

    for fn in files:
        try:
            arr = np.loadtxt(fn, delimiter=",")
        except Exception as e:
            print(f"Skipping {fn}: {e}")
            continue

        if arr.size == 0:
            continue

        if arr.ndim == 1:
            arr = arr[None, :]

        arrays.append(arr)

    if len(arrays) == 0:
        raise RuntimeError(f"No valid data found for pattern: {pattern}")

    data = np.vstack(arrays)

    # columns: epoch, batch, loss
    epochs = data[:, 0].astype(int)
    batches = data[:, 1].astype(int)
    losses = data[:, 2]

    return epochs, batches, losses, files


def epoch_means(epochs, losses):
    unique_epochs = np.array(sorted(set(epochs)))
    mean_losses = np.array([losses[epochs == e].mean() for e in unique_epochs])
    std_losses = np.array([losses[epochs == e].std() for e in unique_epochs])
    return unique_epochs, mean_losses, std_losses


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--train_pattern",
    type=str,
    #default="runs/study_010/training_study010_epoch*.csv",
    default="runs/study_012/training_study012_epoch*.csv",
    )

    parser.add_argument(
        "--val_pattern",
        type=str,
        #default="runs/study_010/validation_study010_epoch*.csv",
        default="runs/study_012/validation_study012_epoch*.csv",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="loss_curves_study012.png",
    )

    parser.add_argument(
        "--logy",
        action="store_true",
        default=True,
        help="Use log scale on y-axis.",
    )

    args = parser.parse_args()

    train_epochs, train_batches, train_losses, train_files = load_records(args.train_pattern)

    print("Loaded training files:")
    for f in train_files:
        print("  ", f)

    train_ep, train_mean, train_std = epoch_means(train_epochs, train_losses)

    plt.figure(figsize=(8, 5))
    plt.plot(train_ep, train_mean, marker="o", label="Train")

    # Try validation, but do not fail if absent
    try:
        val_epochs, val_batches, val_losses, val_files = load_records(args.val_pattern)

        print("Loaded validation files:")
        for f in val_files:
            print("  ", f)

        val_ep, val_mean, val_std = epoch_means(val_epochs, val_losses)
        plt.plot(val_ep, val_mean, marker="s", label="Validation")

    except Exception as e:
        print(f"No validation curve plotted: {e}")

    plt.xlabel("Epoch")
    plt.ylabel("Mean loss")
    plt.title("Loss curves")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if args.logy:
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
