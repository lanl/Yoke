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
  4. A context-length sweep: how many training samples (context windows) and how
     many light curves survive as ``context_len`` grows. This matters because the
     9-band model bakes ``context_len`` into its first layer, so choosing a
     longer context means a retrain -- this panel shows the data cost before you
     pay for it.
  5. A time-window context sweep: context size (real detections) vs the trailing
     lookback ``W`` in days, for choosing ``context_window_days`` /
     ``max_context_len`` in window mode.
  6. Supervised lead-time ``Delta_t`` vs forecast horizon: the CDF of the
     gap-to-next-event (``h=1``, exactly the ``Delta_t`` training supervises)
     against the lead times reachable at larger event offsets. Its right tail is
     the coverage ceiling -- forecasts asked for a longer ``Delta_t`` than the
     ``h=1`` p99/max extrapolate beyond anything the model was trained on.

Run directly, e.g.:
    python plot_observation_histograms.py
    python plot_observation_histograms.py --include_upper_limits
    python plot_observation_histograms.py --sweep_max 20
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
            event_times_per_file (list[np.ndarray]): For each file, the merged,
                time-sorted, file-relative detection times (all bands), matching
                the event stream the 9-band dataset builds. Used by the
                time-window sweep.
            n_files (int): Number of files successfully read.
    """
    n_bands = len(band_keys)
    det_totals = np.zeros(n_bands, dtype=np.int64)
    lim_totals = np.zeros(n_bands, dtype=np.int64)
    det_per_curve = [[] for _ in range(n_bands)]
    total_det_per_curve = []
    event_times_per_file = []

    n_files = 0
    for fn in files:
        try:
            data = np.load(fn, allow_pickle=True)
        except Exception as exc:
            print(f"  skipped {fn}: {exc}")
            continue

        file_total_det = 0
        file_det_times = []
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

            # Collect detection times (col 0 = MJD) for the merged event stream.
            if n_det:
                file_det_times.append(arr[detected, 0].astype(np.float64))

        data.close()
        total_det_per_curve.append(file_total_det)

        # Merge all bands' detections into one time-sorted, file-relative stream,
        # exactly as Kilonova_lc_scalar_context_DataSet_9band does.
        if file_det_times:
            merged = np.concatenate(file_det_times)
            merged.sort(kind="stable")
            merged -= merged.min()
            event_times_per_file.append(merged)

        n_files += 1

    return {
        "det_totals": det_totals,
        "lim_totals": lim_totals,
        "det_per_curve": [np.asarray(c, dtype=np.int64) for c in det_per_curve],
        "total_det_per_curve": np.asarray(total_det_per_curve, dtype=np.int64),
        "event_times_per_file": event_times_per_file,
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


def context_len_sweep(events_per_file, sweep_max):
    """Compute surviving samples/curves as a function of ``context_len``.

    The 9-band dataset turns each light curve of ``n_events`` detections into
    ``max(0, n_events - context_len)`` training windows (one per start index,
    ``max_start = n_events - context_len - 1``, inclusive). A curve contributes
    at all only when ``n_events > context_len``. This mirrors
    ``Kilonova_lc_scalar_context_DataSet_9band`` exactly.

    Args:
        events_per_file (np.ndarray): Detections (== merged event count) per
            light curve, one entry per file.
        sweep_max (int): Largest context length to evaluate.

    Returns:
        context_lens (np.ndarray): Candidate context lengths, shape [sweep_max].
        n_samples (np.ndarray): Total training windows at each context length.
        n_curves (np.ndarray): Number of light curves that yield >=1 window.
    """
    context_lens = np.arange(1, sweep_max + 1)
    ev = events_per_file.astype(np.int64)

    n_samples = np.array(
        [np.maximum(ev - c, 0).sum() for c in context_lens], dtype=np.int64
    )
    n_curves = np.array(
        [int((ev > c).sum()) for c in context_lens], dtype=np.int64
    )
    return context_lens, n_samples, n_curves


def plot_context_sweep(ax, events_per_file, sweep_max):
    """Plot surviving training samples and light curves vs context length."""
    if events_per_file.size == 0:
        ax.set_visible(False)
        return

    context_lens, n_samples, n_curves = context_len_sweep(
        events_per_file, sweep_max
    )
    n_total = events_per_file.size

    # Samples on the left axis (can be large); fraction of curves kept on the
    # right axis so both trends are readable together.
    ax.plot(context_lens, n_samples, marker="o", color="tab:purple",
            label="training windows")
    ax.set_xlabel("context_len")
    ax.set_ylabel("Total training windows", color="tab:purple")
    ax.tick_params(axis="y", labelcolor="tab:purple")
    ax.set_title("Data cost of context length")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    frac_curves = n_curves / n_total
    ax2.plot(context_lens, frac_curves, marker="s", color="tab:red",
             linestyle="--", label="fraction of curves kept")
    ax2.set_ylabel("Fraction of light curves kept", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0, 1.02)

    # Mark the current default context_len=5 for reference.
    if context_lens[0] <= 5 <= context_lens[-1]:
        ax.axvline(5, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.text(5, ax.get_ylim()[1], " default 5", ha="left", va="top",
                fontsize=8, color="gray")

    # Combined legend from both axes.
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")


def time_window_counts(event_times_per_file, window_days):
    """Real detections per context window for a fixed lookback in days.

    Mirrors the planned window-mode selection in
    ``Kilonova_lc_scalar_context_DataSet_9band``: for each target event (every
    event after the first in a file's merged, time-sorted stream), the context
    is every earlier event ``j`` with
    ``times[target - 1] - times[j] <= window_days``. The window is anchored on
    the event immediately preceding the target (the most recent observation),
    so it always contains at least that one event.

    Args:
        event_times_per_file (list[np.ndarray]): Per-file merged, sorted,
            file-relative detection times.
        window_days (float): Trailing lookback length in days.

    Returns:
        np.ndarray: One entry per (file, target-event) sample: the number of
            real detections that fall inside the trailing window.
    """
    counts = []
    for times in event_times_per_file:
        n = times.shape[0]
        if n < 2:
            continue
        # target_idx runs over every event after the first; the window is
        # anchored at times[target_idx - 1] and looks back window_days.
        for target_idx in range(1, n):
            anchor = times[target_idx - 1]
            lo = anchor - window_days
            # Events strictly before the target that are within the window.
            in_window = times[:target_idx]
            counts.append(int((in_window >= lo).sum()))
    return np.asarray(counts, dtype=np.int64)


def time_window_sweep(event_times_per_file, window_grid):
    """Distribution of context size vs trailing window length.

    Args:
        event_times_per_file (list[np.ndarray]): Per-file merged detection times.
        window_grid (np.ndarray): Candidate window lengths in days.

    Returns:
        window_grid (np.ndarray): The evaluated window lengths.
        stats (dict): Percentile arrays keyed by label ("median", "p90", "p95",
            "p99", "max", "mean"), each shape [len(window_grid)].
        raw (list[np.ndarray]): Per-window arrays of per-sample context counts,
            for histogramming.
    """
    pct_labels = [("median", 50), ("p90", 90), ("p95", 95), ("p99", 99)]
    stats = {label: [] for label, _ in pct_labels}
    stats["max"] = []
    stats["mean"] = []
    raw = []

    for w in window_grid:
        c = time_window_counts(event_times_per_file, float(w))
        raw.append(c)
        if c.size == 0:
            for label, _ in pct_labels:
                stats[label].append(0.0)
            stats["max"].append(0.0)
            stats["mean"].append(0.0)
            continue
        for label, q in pct_labels:
            stats[label].append(float(np.percentile(c, q)))
        stats["max"].append(float(c.max()))
        stats["mean"].append(float(c.mean()))

    stats = {k: np.asarray(v) for k, v in stats.items()}
    return window_grid, stats, raw


def plot_time_window_sweep(ax, event_times_per_file, window_grid):
    """Plot context-size percentiles vs trailing window length."""
    if len(event_times_per_file) == 0:
        ax.set_visible(False)
        return

    window_grid, stats, _ = time_window_sweep(event_times_per_file, window_grid)

    ax.plot(window_grid, stats["median"], marker="o", label="median")
    ax.plot(window_grid, stats["p90"], marker="^", label="90th pct")
    ax.plot(window_grid, stats["p95"], marker="s", label="95th pct")
    ax.plot(window_grid, stats["p99"], marker="d", label="99th pct")
    ax.plot(window_grid, stats["max"], linestyle=":", color="gray", label="max")

    ax.set_xlabel("Context window length W (days)")
    ax.set_ylabel("Real detections in window")
    ax.set_title("Context size vs time window\n(pick max_context_len from a high pct)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def dt_horizon_stats(event_times_per_file, horizons):
    """Distribution of lead time Delta_t as a function of event horizon.

    Training supervises, for each context, the jump to the target event. When
    the target is the *immediate next* event (``horizon=1``) the supervised
    Delta_t is exactly the consecutive-event gap. This function collects, for
    each horizon ``h``, every reachable lead time ``times[i + h] - times[i]``
    across all light curves. Comparing ``h=1`` (what training actually sees)
    against the forecast horizons used at inference reveals whether the model is
    ever supervised at the Delta_t it is later asked to extrapolate to.

    Args:
        event_times_per_file (list[np.ndarray]): Per-file merged, sorted,
            file-relative detection times.
        horizons (iterable[int]): Event offsets ``h`` to evaluate. ``h=1`` is
            the training Delta_t (gap to next event).

    Returns:
        horizons (list[int]): The evaluated horizons.
        raw (dict[int, np.ndarray]): For each horizon, all reachable lead times
            (days) pooled over every light curve.
    """
    horizons = list(horizons)
    raw = {h: [] for h in horizons}
    for times in event_times_per_file:
        n = times.shape[0]
        for h in horizons:
            if n > h:
                raw[h].append(times[h:] - times[:-h])
    raw = {
        h: (np.concatenate(v) if v else np.zeros(0, dtype=np.float64))
        for h, v in raw.items()
    }
    return horizons, raw


def plot_dt_horizon(ax, event_times_per_file, horizons):
    """CDF of supervised lead time Delta_t vs event horizon.

    The ``h=1`` curve is the distribution of training Delta_t (gap to the next
    event); its right tail is where the model stops being supervised. Larger-``h``
    curves show how far ahead (in events) a target must be drawn to reach a given
    lead time -- the basis for horizon-covering target sampling. Percentile lines
    for ``h=1`` make the coverage ceiling explicit.
    """
    if len(event_times_per_file) == 0:
        ax.set_visible(False)
        return

    horizons, raw = dt_horizon_stats(event_times_per_file, horizons)
    cmap = plt.get_cmap("viridis")

    for j, h in enumerate(horizons):
        c = raw[h]
        if c.size == 0:
            continue
        xs = np.sort(c)
        ys = np.arange(1, xs.size + 1) / xs.size
        label = f"h={h}" + (" (train Δt)" if h == 1 else "")
        ax.plot(
            xs,
            ys,
            color=cmap(j / max(1, len(horizons) - 1)),
            linewidth=2.0 if h == 1 else 1.3,
            label=label,
        )

    # Percentile markers for the training Delta_t (h=1): the coverage ceiling.
    base = raw[horizons[0]]
    if base.size:
        for q in (95, 99):
            pv = float(np.percentile(base, q))
            ax.axvline(pv, color="tab:red", linestyle=":", linewidth=1, alpha=0.7)
            ax.text(
                pv,
                0.02,
                f" p{q}={pv:.1f}d",
                rotation=90,
                va="bottom",
                ha="right",
                fontsize=7,
                color="tab:red",
            )

    ax.set_xlabel("Lead time Δt (days)")
    ax.set_ylabel("Cumulative fraction of samples")
    ax.set_ylim(0, 1.02)
    ax.set_title(
        "Supervised Δt vs forecast horizon\n"
        "(h=1 is training Δt; right tail = uncovered)"
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)


def print_dt_horizon(event_times_per_file, horizons):
    """Print the lead-time-vs-horizon percentile table to stdout."""
    if len(event_times_per_file) == 0:
        return

    horizons, raw = dt_horizon_stats(event_times_per_file, horizons)

    print("\nSupervised lead time Δt by event horizon (days):")
    print(
        f"{'horizon':>8} {'n':>10} {'median':>8} {'p90':>7} {'p95':>7} "
        f"{'p99':>7} {'max':>7}"
    )
    for h in horizons:
        c = raw[h]
        if c.size == 0:
            print(f"{h:>8} {0:>10}")
            continue
        print(
            f"{h:>8} {c.size:>10} {np.median(c):>8.2f} "
            f"{np.percentile(c, 90):>7.2f} {np.percentile(c, 95):>7.2f} "
            f"{np.percentile(c, 99):>7.2f} {c.max():>7.2f}"
        )
    print(
        "h=1 is the Delta_t training actually supervises (target = next event). "
        "Its p99/max is the coverage ceiling; forecasts beyond it extrapolate."
    )


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


def print_sweep(events_per_file, sweep_max):
    """Print the context-length sweep table to stdout."""
    if events_per_file.size == 0:
        return

    context_lens, n_samples, n_curves = context_len_sweep(
        events_per_file, sweep_max
    )
    n_total = events_per_file.size

    print("\nContext-length sweep (samples = training windows):")
    print(f"{'context_len':>12} {'windows':>12} {'curves_kept':>12} {'frac':>7}")
    for c, s, k in zip(context_lens, n_samples, n_curves):
        print(f"{int(c):>12} {int(s):>12} {int(k):>12} {k / n_total:>7.2f}")


def print_time_window_sweep(event_times_per_file, window_grid):
    """Print the time-window context-size sweep table to stdout.

    This is the table to read when choosing the two window-mode hyperparameters:
    ``context_window_days`` (a W with enough baseline for real evolution) and
    ``max_context_len`` (a high percentile so padding rarely clips real events).
    """
    if len(event_times_per_file) == 0:
        return

    window_grid, stats, raw = time_window_sweep(event_times_per_file, window_grid)

    print("\nTime-window context sweep (real detections inside trailing W days):")
    print(
        f"{'W_days':>8} {'n_samples':>10} {'median':>8} {'p90':>6} "
        f"{'p95':>6} {'p99':>6} {'max':>6} {'mean':>7}"
    )
    for i, w in enumerate(window_grid):
        n_s = raw[i].size
        print(
            f"{float(w):>8.2f} {int(n_s):>10} {stats['median'][i]:>8.0f} "
            f"{stats['p90'][i]:>6.0f} {stats['p95'][i]:>6.0f} "
            f"{stats['p99'][i]:>6.0f} {stats['max'][i]:>6.0f} "
            f"{stats['mean'][i]:>7.1f}"
        )
    print(
        "Pick context_window_days from W with enough baseline; set "
        "max_context_len ~ the p95/p99 column at that W."
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
        "--sweep_max",
        type=int,
        default=20,
        help=(
            "Largest context_len to evaluate in the data-cost sweep panel. "
            "Default 20."
        ),
    )
    parser.add_argument(
        "--window_max",
        type=float,
        default=10.0,
        help=(
            "Largest trailing window length (days) in the time-window context "
            "sweep. Default 10."
        ),
    )
    parser.add_argument(
        "--window_step",
        type=float,
        default=1.0,
        help="Step (days) between evaluated window lengths. Default 1.0.",
    )
    parser.add_argument(
        "--dt_horizons",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5, 8],
        help=(
            "Event offsets h for the supervised-Delta_t panel. h=1 is the "
            "training Delta_t (gap to next event); larger h show how far ahead "
            "a target must be drawn to reach a given lead time. Default 1 2 3 5 8."
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
    print_sweep(counts["total_det_per_curve"], args.sweep_max)

    # Time-window sweep grid: window_step .. window_max (days).
    window_grid = np.arange(
        args.window_step, args.window_max + 0.5 * args.window_step, args.window_step
    )
    print_time_window_sweep(counts["event_times_per_file"], window_grid)
    print_dt_horizon(counts["event_times_per_file"], args.dt_horizons)

    fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    plot_band_totals(axes[0, 0], counts, args.include_upper_limits)
    plot_total_hist(axes[0, 1], counts)
    plot_per_band_hist(axes[0, 2], counts)
    plot_context_sweep(axes[1, 0], counts["total_det_per_curve"], args.sweep_max)
    plot_time_window_sweep(axes[1, 1], counts["event_times_per_file"], window_grid)
    plot_dt_horizon(axes[1, 2], counts["event_times_per_file"], args.dt_horizons)

    fig.suptitle(
        f"KN light-curve observations ({counts['n_files']} light curves)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=args.dpi)
    print(f"Saved {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
