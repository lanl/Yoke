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


def epoch_stats(epochs, losses, central="mean"):
    """Per-epoch central tendency (+ spread) of the per-batch loss.

    Args:
        central (str): "mean" (default) or "median". The loss distribution is
            heavy-tailed (the blue-band minority drags the mean up ~an order of
            magnitude above the bulk), so "median" tracks the TYPICAL sample and
            is the more honest progress metric; "mean" reports the tail-weighted
            average. The returned spread column is the std either way (the
            percentile band is the better spread visual for the median).
    """
    unique_epochs = np.array(sorted(set(epochs)))
    if central == "median":
        central_losses = np.vstack(
            [np.median(losses[epochs == e], axis=0) for e in unique_epochs]
        )
    else:
        central_losses = np.vstack(
            [losses[epochs == e].mean(axis=0) for e in unique_epochs]
        )
    std_losses = np.vstack(
        [losses[epochs == e].std(axis=0) for e in unique_epochs]
    )
    return unique_epochs, central_losses, std_losses


def epoch_percentiles(
    epochs: np.ndarray,
    losses: np.ndarray,
    pct_lo: float = 5.0,
    pct_hi: float = 95.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-epoch lower/upper percentiles of the per-batch loss.

    Complements ``epoch_stats``: where the std band is symmetric and can dip
    below zero on a log axis, percentile bands capture the actual spread of the
    per-batch losses within each epoch (e.g. the 5th and 95th) and are robust to
    the heavy-tailed batches that make the raw loss curve jagged.

    Args:
        epochs (np.ndarray): Per-row epoch index, shape [N].
        losses (np.ndarray): Per-row loss columns, shape [N, n_loss_cols].
        pct_lo (float): Lower percentile in [0, 100]. Default 5.0.
        pct_hi (float): Upper percentile in [0, 100]. Default 95.0.

    Returns:
        unique_epochs (np.ndarray): Sorted unique epoch indices, shape [E].
        lo_losses (np.ndarray): Lower-percentile loss per epoch, [E, n_loss_cols].
        hi_losses (np.ndarray): Upper-percentile loss per epoch, [E, n_loss_cols].
    """
    unique_epochs = np.array(sorted(set(epochs)))
    lo_losses = np.vstack(
        [np.percentile(losses[epochs == e], pct_lo, axis=0) for e in unique_epochs]
    )
    hi_losses = np.vstack(
        [np.percentile(losses[epochs == e], pct_hi, axis=0) for e in unique_epochs]
    )
    return unique_epochs, lo_losses, hi_losses


def infer_loss_labels(loss_names, n_loss_cols):
    """Make nicer labels for common scalar/GRI cases."""
    if n_loss_cols == 1:
        return [loss_names[0] if loss_names else "loss"]

    if n_loss_cols == 3 and loss_names == ["loss", "col3", "col4"]:
        return ["g", "r", "i"]

    if n_loss_cols == 4 and loss_names == ["loss", "col3", "col4", "col5"]:
        return ["total", "g", "r", "i"]

    return loss_names


def shade_tf_regimes(ax, ramp_start, ramp_epochs):
    """Shade the scheduled-sampling teacher-forcing regimes on a loss axis.

    The teacher-forcing ratio p anneals from 1.0 (fully teacher-forced) down to
    0.0 (fully free-running) over training. This splits the run into three
    regimes, defined entirely by ``ramp_start`` and ``ramp_epochs``:

      - Warmup   [xmin, ramp_start]:              p = 1.0, equivalent to
                                                  single-step training.
      - Anneal   [ramp_start, ramp_start+epochs]: p ramps 1.0 -> 0.0.
      - Free-run [ramp_start+epochs, xmax]:       p = 0.0, matches inference.

    Only these training regimes are shaded; the validation curve is always pure
    free-run regardless of regime, so a val bump at the anneal onset is expected.
    """
    xmin, xmax = ax.get_xlim()
    anneal_end = ramp_start + ramp_epochs

    # Clip regime edges to the visible epoch range so a partial run still shades
    # sensibly (e.g. a plot that only reaches into the anneal phase).
    warmup_lo, warmup_hi = xmin, min(ramp_start, xmax)
    anneal_lo, anneal_hi = max(ramp_start, xmin), min(anneal_end, xmax)
    free_lo, free_hi = max(anneal_end, xmin), xmax

    regimes = [
        (warmup_lo, warmup_hi, "tab:blue", "warmup (p=1)"),
        (anneal_lo, anneal_hi, "tab:orange", "anneal (p:1→0)"),
        (free_lo, free_hi, "tab:green", "free-run (p=0)"),
    ]

    for lo, hi, color, label in regimes:
        if hi <= lo:
            continue
        ax.axvspan(lo, hi, color=color, alpha=0.08, zorder=0)
        # Place the label near the top of the axis, centered in the band.
        ax.text(
            0.5 * (lo + hi),
            0.97,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=color,
            alpha=0.9,
        )

    for boundary in (ramp_start, anneal_end):
        if xmin < boundary < xmax:
            ax.axvline(
                boundary, color="gray", linewidth=1, linestyle=":", alpha=0.6
            )


def plot_epoch_curves(train, val, args, diag=None):
    central = getattr(args, "central", "mean")
    train_ep, train_mean, train_std = epoch_stats(
        train["epochs"],
        train["losses"],
        central=central,
    )

    n_loss_cols = train["losses"].shape[1]
    loss_labels = infer_loss_labels(train["loss_names"], n_loss_cols)

    # Percentile bands (default 5th/95th) of the per-batch loss within each epoch.
    show_percentiles = getattr(args, "show_percentiles", False)
    pct_lo = getattr(args, "pct_lo", 5.0)
    pct_hi = getattr(args, "pct_hi", 95.0)
    if show_percentiles:
        _, train_plo, train_phi = epoch_percentiles(
            train["epochs"], train["losses"], pct_lo, pct_hi
        )
    else:
        train_plo = train_phi = None

    if val is not None:
        val_ep, val_mean, val_std = epoch_stats(
            val["epochs"],
            val["losses"],
            central=central,
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
            val_plo = val_phi = None
        elif show_percentiles:
            _, val_plo, val_phi = epoch_percentiles(
                val["epochs"], val["losses"], pct_lo, pct_hi
            )
        else:
            val_plo = val_phi = None
    else:
        val_ep = None
        val_mean = None
        val_std = None
        val_plo = val_phi = None

    # Fixed-difficulty train diagnostic (free-run rollout over train data). Same
    # task as validation, so it descends with real skill and is directly
    # comparable to the validation curve -- unlike the raw train loss, which is on
    # a moving (annealing) difficulty and looks flat.
    if diag is not None:
        diag_ep, diag_central, _ = epoch_stats(
            diag["epochs"], diag["losses"], central=central
        )
        if diag_central.shape[1] != n_loss_cols:
            print(
                "Train diagnostic has a different number of loss columns "
                f"({diag_central.shape[1]}) than training ({n_loss_cols}); "
                "skipping diagnostic."
            )
            diag = None
    else:
        diag_ep = diag_central = None

    plt.figure(figsize=(9, 5.5))

    band_label = f"{pct_lo:g}-{pct_hi:g} pct"

    for idx, label in enumerate(loss_labels):
        suffix = "" if n_loss_cols == 1 else f" {label}"

        (train_line,) = plt.plot(
            train_ep,
            train_mean[:, idx],
            marker="o",
            label=f"Train{suffix}",
        )

        if args.show_std:
            lo = np.maximum(train_mean[:, idx] - train_std[:, idx], 1e-30)
            hi = train_mean[:, idx] + train_std[:, idx]
            plt.fill_between(train_ep, lo, hi, alpha=0.15)

        if show_percentiles:
            lo = np.maximum(train_plo[:, idx], 1e-30)
            hi = train_phi[:, idx]
            plt.fill_between(
                train_ep,
                lo,
                hi,
                alpha=0.15,
                color=train_line.get_color(),
                label=f"Train {band_label}{suffix}",
            )

        if val is not None:
            (val_line,) = plt.plot(
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

            if show_percentiles:
                lo = np.maximum(val_plo[:, idx], 1e-30)
                hi = val_phi[:, idx]
                plt.fill_between(
                    val_ep,
                    lo,
                    hi,
                    alpha=0.10,
                    color=val_line.get_color(),
                    label=f"Validation {band_label}{suffix}",
                )

        if diag is not None:
            plt.plot(
                diag_ep,
                diag_central[:, idx],
                marker="^",
                linestyle="-.",
                color="tab:green",
                label=f"Train (free-run diag){suffix}",
            )

    plt.xlabel("Epoch")
    plt.ylabel(f"{central.capitalize()} loss")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if args.logy:
        plt.yscale("log")

    # Optionally shade the scheduled-sampling teacher-forcing regimes. Gated on
    # attributes that only the 9-band wrapper sets, so g/r/i runs (and any caller
    # that doesn't opt in) get the plain plot unchanged.
    if getattr(args, "shade_regimes", False):
        shade_tf_regimes(
            plt.gca(),
            ramp_start=getattr(args, "tf_ramp_start_epoch", 20),
            ramp_epochs=getattr(args, "tf_ramp_epochs", 20),
        )

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
        "--show_percentiles",
        action="store_true",
        help="Shade the per-epoch percentile band (default 5th-95th) of the "
        "per-batch loss.",
    )
    parser.add_argument(
        "--pct_lo",
        type=float,
        default=5.0,
        help="Lower percentile for --show_percentiles. Default 5.",
    )
    parser.add_argument(
        "--pct_hi",
        type=float,
        default=95.0,
        help="Upper percentile for --show_percentiles. Default 95.",
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
