import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_STUDY = 24
DEFAULT_RUNS_ROOT = "runs"
DEFAULT_COLUMNS = "epoch,batch,loss"


def format_study(study):
    """Return both integer and zero-padded study strings."""
    study_int = int(study)
    return study_int, f"{study_int:03d}"


def default_patterns(study, runs_root):
    _, study_tag = format_study(study)
    run_dir = Path(runs_root) / f"study_{study_tag}"
    return {
        "train": str(run_dir / f"training_study{study_tag}_epoch*.csv"),
        "val": str(run_dir / f"validation_study{study_tag}_epoch*.csv"),
        "out": f"loss_curves_study{study_tag}.png",
    }


def parse_column_names(columns_arg, n_cols):
    """Build display names for columns in the CSV record files."""
    provided = [c.strip() for c in columns_arg.split(",") if c.strip()]

    if len(provided) < n_cols:
        provided.extend([f"col{i}" for i in range(len(provided), n_cols)])

    return provided[:n_cols]


def load_records(pattern, columns_arg=DEFAULT_COLUMNS):
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    arrays = []
    skipped = []

    for fn in files:
        try:
            arr = np.loadtxt(fn, delimiter=",")
        except Exception as exc:
            skipped.append((fn, str(exc)))
            continue

        if arr.size == 0:
            skipped.append((fn, "empty file"))
            continue

        if arr.ndim == 1:
            arr = arr[None, :]

        if arr.shape[1] < 3:
            skipped.append((fn, f"expected at least 3 columns, found {arr.shape[1]}"))
            continue

        arrays.append(arr)

    if len(arrays) == 0:
        details = "\n".join(f"  {fn}: {reason}" for fn, reason in skipped)
        raise RuntimeError(f"No valid data found for pattern: {pattern}\n{details}")

    # Keep only the common width if an interrupted run left mixed-width records.
    n_cols = min(arr.shape[1] for arr in arrays)
    if any(arr.shape[1] != n_cols for arr in arrays):
        print(f"Warning: mixed CSV widths found; using first {n_cols} columns.")
        arrays = [arr[:, :n_cols] for arr in arrays]

    data = np.vstack(arrays)

    # Sort by epoch, then batch.
    data = data[np.lexsort((data[:, 1], data[:, 0]))]

    names = parse_column_names(columns_arg, n_cols)

    epochs = data[:, 0].astype(int)
    batches = data[:, 1].astype(int)
    losses = data[:, 2:]
    loss_names = names[2:]

    return {
        "epochs": epochs,
        "batches": batches,
        "losses": losses,
        "loss_names": loss_names,
        "files": files,
        "skipped": skipped,
        "data": data,
    }


def epoch_stats(epochs, losses):
    unique_epochs = np.array(sorted(set(epochs)))
    mean_losses = np.vstack(
        [losses[epochs == e].mean(axis=0) for e in unique_epochs]
    )
    std_losses = np.vstack(
        [losses[epochs == e].std(axis=0) for e in unique_epochs]
    )
    return unique_epochs, mean_losses, std_losses


def infer_loss_labels(loss_names, n_loss_cols):
    """Make nicer labels for common scalar/GRI cases."""
    if n_loss_cols == 1:
        return [loss_names[0] if loss_names else "loss"]

    if n_loss_cols == 3 and loss_names == ["loss", "col3", "col4"]:
        return ["g", "r", "i"]

    if n_loss_cols == 4 and loss_names == ["loss", "col3", "col4", "col5"]:
        return ["total", "g", "r", "i"]

    return loss_names


def plot_epoch_curves(train, val, args):
    train_ep, train_mean, train_std = epoch_stats(
        train["epochs"],
        train["losses"],
    )

    n_loss_cols = train["losses"].shape[1]
    loss_labels = infer_loss_labels(train["loss_names"], n_loss_cols)

    if val is not None:
        val_ep, val_mean, val_std = epoch_stats(
            val["epochs"],
            val["losses"],
        )

        if val_mean.shape[1] != n_loss_cols:
            print(
                "Validation has a different number of loss columns "
                f"({val_mean.shape[1]}) than training ({n_loss_cols}); "
                "skipping validation."
            )
            val = None
            val_ep = None
            val_mean = None
            val_std = None
    else:
        val_ep = None
        val_mean = None
        val_std = None

    plt.figure(figsize=(9, 5.5))

    for idx, label in enumerate(loss_labels):
        suffix = "" if n_loss_cols == 1 else f" {label}"

        plt.plot(
            train_ep,
            train_mean[:, idx],
            marker="o",
            label=f"Train{suffix}",
        )

        if args.show_std:
            lo = np.maximum(train_mean[:, idx] - train_std[:, idx], 1e-30)
            hi = train_mean[:, idx] + train_std[:, idx]
            plt.fill_between(train_ep, lo, hi, alpha=0.15)

        if val is not None:
            plt.plot(
                val_ep,
                val_mean[:, idx],
                marker="s",
                linestyle="--",
                label=f"Validation{suffix}",
            )

            if args.show_std:
                lo = np.maximum(val_mean[:, idx] - val_std[:, idx], 1e-30)
                hi = val_mean[:, idx] + val_std[:, idx]
                plt.fill_between(val_ep, lo, hi, alpha=0.10)

    plt.xlabel("Epoch")
    plt.ylabel("Mean loss")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if args.logy:
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi)
    print(f"Saved {args.out}")


def print_loaded(label, record):
    print(f"Loaded {label} files:")
    for fn in record["files"]:
        print("  ", fn)

    for fn, reason in record["skipped"]:
        print(f"  skipped {fn}: {reason}")

    print(f"{label} rows: {record['data'].shape[0]}")
    print(f"{label} columns: {record['data'].shape[1]}")
    print(f"{label} loss columns: {', '.join(record['loss_names'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training/validation loss curves for scalar temporal LodeRunner GRI runs."
    )

    parser.add_argument("--study", type=int, default=DEFAULT_STUDY)
    parser.add_argument("--runs_root", type=str, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--train_pattern", type=str, default=None)
    parser.add_argument("--val_pattern", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)

    parser.add_argument(
        "--columns",
        type=str,
        default=DEFAULT_COLUMNS,
        help=(
            "Comma-separated CSV column names. The first two must be epoch,batch. "
            "Examples: epoch,batch,loss or epoch,batch,total,g,r,i."
        ),
    )

    parser.add_argument("--title", type=str, default="GRI loss curves")
    parser.add_argument("--dpi", type=int, default=200)

    parser.add_argument(
        "--logy",
        dest="logy",
        action="store_true",
        default=True,
        help="Use log scale on y-axis. This is the default.",
    )

    parser.add_argument(
        "--linear",
        dest="logy",
        action="store_false",
        help="Use linear y-axis.",
    )

    parser.add_argument(
        "--show_std",
        action="store_true",
        help="Shade +/- one epoch standard deviation.",
    )

    parser.add_argument(
        "--require_val",
        action="store_true",
        help="Fail instead of continuing when validation records are missing or invalid.",
    )

    args = parser.parse_args()

    defaults = default_patterns(args.study, args.runs_root)

    args.train_pattern = args.train_pattern or defaults["train"]
    args.val_pattern = args.val_pattern or defaults["val"]
    args.out = args.out or defaults["out"]

    train = load_records(args.train_pattern, args.columns)
    print_loaded("training", train)

    try:
        val = load_records(args.val_pattern, args.columns)
        print_loaded("validation", val)
    except Exception as exc:
        if args.require_val:
            raise
        print(f"No validation curve plotted: {exc}")
        val = None

    plot_epoch_curves(train, val, args)


if __name__ == "__main__":
    main()
