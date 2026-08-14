"""Plot observation-count histograms for the 9-band kilonova light-curve data.

Each ``lc_*.npz`` file holds one light curve as a set of per-band arrays keyed by
``NINE_BAND_KEYS`` (``arr_ztfg`` ... ``arr_ps1__y``). Every row of a band array is
one observation with columns ``[MJD, value, error, ...]``. Matching the 9-band
dataset (``Kilonova_lc_scalar_context_DataSet_9band`` with
``drop_upper_limits=True``), a row is counted as a real **detection** when its
error column (col 2) is finite; a non-finite error flags an upper limit /
non-detection. This script summarises the data set as histograms:

  1. Total detections per band (bar chart) -- how much supervision each of the
     nine output heads actually gets.
  2. Total observations per band split into detections vs upper limits, when
     ``--include_upper_limits`` is set.
  3. Distribution of detections-per-light-curve, per band and summed over all
     bands -- how long/rich a typical object is.

Run directly, e.g.:
    python plot_observation_histograms.py
    python plot_observation_histograms.py --include_upper_limits
    python plot_observation_histograms.py \
        --data_glob '/path/to/lc_*.npz' --out obs_hist.png
"""

import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

# Reuse the canonical band ordering / keys and column conventions from the
# dataset so "observation" here means exactly what the model trains on.
from yoke.datasets.kilonova_dataset import NINE_BAND_KEYS


# Default to the same data the 9-band pipeline trains and computes norm stats on
# (the Rubin+ZTF set), so the histogram reflects the real training distribution.
DEFAULT_DATA_GLOB = (
    "/net/sescratch1/atoivonen/data/KN_lightcurves/"
    "rubin_ztf_10000_dataset/lc_*.npz"
)
DEFAULT_ERROR_COL = 2

# Short display labels for the bands, in NINE_BAND_KEYS order.
BAND_LABELS = tuple(k.replace("arr_", "") for k in NINE_BAND_KEYS)


def collect_counts(files, band_keys, error_col):
    """Count detections and upper limits per band across all files.

    Args:
        files (list[str]): npz light-curve files to read.
        band_keys (tuple[str, ...]): Band keys to count, in display order.
        error_col (int): Column whose finiteness distinguishes a detection
            (finite) from an upper limit / non-detection (non-finite).

    Returns:
        dict with:
            det_totals (np.ndarray): Total detections per band, shape [n_bands].
            lim_totals (np.ndarray): Total upper limits per band, shape [n_bands].
            det_per_curve (list[np.ndarray]): For each band, an array holding the
                detection count in each file that contains that band.
            total_det_per_curve (np.ndarray): Total detections (all bands) per
                file, one entry per file.
            n_files (int): Number of files successfully read.
    """
    n_bands = len(band_keys)
    det_totals = np.zeros(n_bands, dtype=np.int64)
    lim_totals = np.zeros(n_bands, dtype=np.int64)
    det_per_curve = [[] for _ in range(n_bands)]
    total_det_per_curve = []

    n_files = 0
    for fn in files:
        try:
            data = np.load(fn, allow_pickle=True)
        except Exception as exc:
            print(f"  skipped {fn}: {exc}")
            continue

        file_total_det = 0
        for b, key in enumerate(band_keys):
            if key not in data.files:
                continue

            arr = data[key]
            if arr.size == 0:
                continue

            errs = arr[:, error_col].astype(np.float64)
            detected = np.isfinite(errs)
            n_det = int(detected.sum())
            n_lim = int(detected.size - n_det)

            det_totals[b] += n_det
            lim_totals[b] += n_lim
            det_per_curve[b].append(n_det)
            file_total_det += n_det

        data.close()
        total_det_per_curve.append(file_total_det)
        n_files += 1

    return {
        "det_totals": det_totals,
        "lim_totals": lim_totals,
        "det_per_curve": [np.asarray(c, dtype=np.int64) for c in det_per_curve],
        "total_det_per_curve": np.asarray(total_det_per_curve, dtype=np.int64),
        "n_files": n_files,
    }


def plot_band_totals(ax, counts, include_upper_limits):
    """Bar chart of total observations per band."""
    n_bands = len(BAND_LABELS)
    x = np.arange(n_bands)
    det = counts["det_totals"]

    if include_upper_limits:
        lim = counts["lim_totals"]
        ax.bar(x, det, color="tab:blue", label="detections")
        ax.bar(x, lim, bottom=det, color="tab:gray", alpha=0.6, label="upper limits")
        ax.legend()
        title_extra = " (detections + upper limits)"
    else:
        ax.bar(x, det, color="tab:blue")
        title_extra = " (detections only)"

    # Annotate each bar with its detection count.
    for xi, di in zip(x, det):
        ax.text(xi, di, f"{int(di)}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS, rotation=45, ha="right")
    ax.set_ylabel("Number of observations")
    ax.set_title(f"Total observations per band{title_extra}")
    ax.grid(True, axis="y", alpha=0.3)


def plot_total_hist(ax, counts):
    """Histogram of total detections per light curve (summed over all bands)."""
    totals = counts["total_det_per_curve"]
    if totals.size == 0:
        ax.set_visible(False)
        return

    hi = int(totals.max())
    # One bin per integer count up to the max, capped so very long tails stay
    # readable.
    bins = np.arange(0, hi + 2) - 0.5 if hi <= 60 else 50
    ax.hist(totals, bins=bins, color="tab:green", alpha=0.8)
    ax.axvline(
        totals.mean(),
        color="k",
        linestyle="--",
        linewidth=1,
        label=f"mean {totals.mean():.1f}",
    )
    ax.set_xlabel("Detections per light curve (all bands)")
    ax.set_ylabel("Number of light curves")
    ax.set_title("Total detections per light curve")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_per_band_hist(ax, counts):
    """Overlaid step histograms of detections-per-curve for each band."""
    per_curve = counts["det_per_curve"]

    # Common integer bins across bands so the overlays are comparable.
    hi = max((c.max() if c.size else 0) for c in per_curve)
    hi = int(hi)
    bins = np.arange(0, max(hi, 1) + 2) - 0.5

    cmap = plt.get_cmap("tab10")
    for b, label in enumerate(BAND_LABELS):
        c = per_curve[b]
        if c.size == 0:
            continue
        ax.hist(
            c,
            bins=bins,
            histtype="step",
            linewidth=1.5,
            color=cmap(b % 10),
            label=label,
        )

    ax.set_xlabel("Detections per light curve")
    ax.set_ylabel("Number of light curves")
    ax.set_title("Detections per light curve, by band")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)


def print_summary(counts, include_upper_limits):
    """Print the per-band and overall counts to stdout."""
    det = counts["det_totals"]
    lim = counts["lim_totals"]

    print(f"Read {counts['n_files']} light-curve files.")
    print(f"{'band':<10} {'detections':>12} {'upper_limits':>14}")
    for b, label in enumerate(BAND_LABELS):
        print(f"{label:<10} {int(det[b]):>12} {int(lim[b]):>14}")

    print(f"{'TOTAL':<10} {int(det.sum()):>12} {int(lim.sum()):>14}")

    totals = counts["total_det_per_curve"]
    if totals.size:
        print(
            f"Detections per light curve: mean {totals.mean():.2f}, "
            f"median {np.median(totals):.0f}, "
            f"min {int(totals.min())}, max {int(totals.max())}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot observation-count histograms (per band and total) for the "
            "9-band kilonova light-curve data."
        )
    )

    parser.add_argument(
        "--data_glob",
        type=str,
        default=DEFAULT_DATA_GLOB,
        help="Glob for the light-curve npz files. Default: the Rubin+ZTF set.",
    )
    parser.add_argument(
        "--error_col",
        type=int,
        default=DEFAULT_ERROR_COL,
        help=(
            "Column whose finiteness marks a detection (finite) vs an upper "
            "limit / non-detection (non-finite). Default 2."
        ),
    )
    parser.add_argument(
        "--include_upper_limits",
        action="store_true",
        help=(
            "Also count and stack upper limits (non-detections) in the per-band "
            "totals. By default only real detections are counted, matching the "
            "9-band dataset with drop_upper_limits=True."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="observation_histograms.png",
        help="Output PNG path.",
    )
    parser.add_argument("--dpi", type=int, default=200)

    args = parser.parse_args()

    files = sorted(glob.glob(args.data_glob))
    if len(files) == 0:
        raise FileNotFoundError(f"No files matched glob: {args.data_glob}")

    print(f"Matched {len(files)} files for glob: {args.data_glob}")

    counts = collect_counts(files, NINE_BAND_KEYS, args.error_col)
    print_summary(counts, args.include_upper_limits)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plot_band_totals(axes[0], counts, args.include_upper_limits)
    plot_total_hist(axes[1], counts)
    plot_per_band_hist(axes[2], counts)

    fig.suptitle(
        f"KN light-curve observations ({counts['n_files']} light curves)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=args.dpi)
    print(f"Saved {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
