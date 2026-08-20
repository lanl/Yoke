"""Dense late-time evaluation for the 9-band scalar temporal LodeRunner.

The model is trained on REALISTIC light curves (sparse, upper limits dropped),
which contain almost no late-time detections because kilonovae fade below the
detection limit. This eval measures how well the model, given a REALISTIC
observing context, forecasts the LATE-TIME behavior -- scored against a DENSE
companion set (the same objects, denser cadence, no limiting-mag cut, so all
late-time points are real detections). Realistic and dense views of an object
are paired by filename stem.

For each held-out (test-split) object:
  1. Build the model input from the realistic stream (trailing time-window
     context ending at the last realistic detection), exactly as in training.
  2. For every dense point in the late-time region (phase from the first
     realistic detection greater than ``--late_time_cutoff_days``), ask the model
     to predict all nine bands at that point's lead time and score the predicted
     magnitude of the dense point's band against the dense truth.
  3. Also sweep a smooth lead-time grid for a per-object forecast plot.

Only detections are used: realistic upper limits are dropped (matching
training); the dense set is all detections by construction.

IMPORTANT (time frames): the training dataset relativizes each stream to its own
first event, so the realistic and dense views of one object live in different
relative frames. Lead times (durations) are frame-independent and are what the
model's ``Dt`` consumes, so this script reads ABSOLUTE MJD (column 0) from the
raw npz files and works entirely in absolute-time differences.

Normalization uses the TRAIN-ONLY realistic stats the model was trained with
(``kilonova_9band_norm_stats_trainonly.npz`` by default) -- the exact encoding
the model saw. This is a read-only diagnostic: it writes plots and a CSV and
never trains.
"""

import argparse
import csv
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from yoke.datasets.kilonova_dataset import (
    EPS,
    NINE_BAND_KEYS,
    load_or_compute_band_normalization,
)

# Reuse the model loader and window/input helpers from the rollout diagnostics
# script that lives alongside this one. These harness scripts are run directly
# (not as an installed package), so make the script directory importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_pred_diagnostics_9band import (  # noqa: E402
    _select_window,
    build_context_input,
    load_9band_model,
)


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rc("font", family="serif")
plt.rcParams["figure.figsize"] = (7, 5)


BAND_KEYS = NINE_BAND_KEYS
BAND_NAMES = ("ztfg", "ztfr", "ztfi", "u", "g", "r", "i", "z", "y")
BAND_COLORS = (
    "#2A9D8F", "#E63946", "#F4A261", "#457B9D", "#1B9E77",
    "#D62828", "#E9C46A", "#8338EC", "#264653",
)
VALUE_COL = 1
ERROR_COL = 2
N_BANDS = len(BAND_KEYS)
DROP_UPPER_LIMITS = True  # matches training for the realistic (context) stream


def study_tag(study: int) -> str:
    """Zero-padded study id used in default paths."""
    return f"{int(study):03d}"


def _stem(path: str) -> str:
    """Return the object identifier: filename without directory or extension."""
    return os.path.splitext(os.path.basename(path))[0]


def read_merged_stream(
    npz_path: str, drop_upper_limits: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one file's merged, time-sorted event stream in ABSOLUTE MJD.

    Unlike the training dataset, times are NOT relativized here, so streams from
    two directories (realistic and dense) remain on a common absolute clock.

    Args:
        npz_path (str): Path to the light-curve npz.
        drop_upper_limits (bool): Drop non-detections (non-finite error) so the
            realistic context stream matches training. The dense set is all
            detections, so this is a no-op there.

    Returns:
        (times, values, bands): absolute MJD, raw magnitude, band index; each
        [N] and sorted by time. Empty arrays if the file has no usable events.
    """
    data = np.load(npz_path, allow_pickle=True)
    times, values, bands = [], [], []
    for band_idx, key in enumerate(BAND_KEYS):
        if key not in data.files:
            continue
        arr = data[key]
        if arr.size == 0:
            continue
        if drop_upper_limits:
            arr = arr[np.isfinite(arr[:, ERROR_COL])]
            if arr.shape[0] == 0:
                continue
        times.append(arr[:, 0].astype(np.float64))
        values.append(arr[:, VALUE_COL].astype(np.float32))
        bands.append(np.full(arr.shape[0], band_idx, dtype=np.int64))
    data.close()

    if not times:
        empty_f = np.empty(0, dtype=np.float64)
        return empty_f, empty_f.astype(np.float32), np.empty(0, dtype=np.int64)

    times = np.concatenate(times)
    values = np.concatenate(values)
    bands = np.concatenate(bands)
    order = np.argsort(times, kind="stable")
    return times[order], values[order], bands[order]


def _stem_to_path(data_glob: str) -> dict:
    """Map object stem -> file path for all files matched by a glob."""
    import glob

    return {_stem(f): f for f in glob.glob(data_glob)}


def _batched_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    lead_times: np.ndarray,
    device: torch.device,
    max_batch: int = 256,
) -> np.ndarray:
    """Predict all bands for many lead times in one (chunked) forward pass.

    The context ``x`` (shape [1, D]) is fixed; only the lead time varies. Tiling
    ``x`` to the batch dimension and passing a Dt vector runs every lead time
    together instead of one-at-a-time, which is dramatically faster on GPU and
    numerically identical to the per-point loop. Chunked at ``max_batch`` so a
    long lead-time sweep cannot exhaust GPU memory.

    Args:
        model: The 9-band scalar-temporal LodeRunner.
        x (torch.Tensor): Context input of shape [1, D].
        lead_times (np.ndarray): 1-D array of lead times (days).
        device (torch.device): Device to run on.
        max_batch (int): Maximum lead times evaluated per forward pass.

    Returns:
        np.ndarray: Predictions of shape [len(lead_times), N_BANDS] (normalized).
    """
    lead_times = np.asarray(lead_times, dtype=np.float32)
    out = np.zeros((lead_times.shape[0], N_BANDS), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, lead_times.shape[0], max_batch):
            chunk = lead_times[start : start + max_batch]
            x_batch = x.expand(chunk.shape[0], -1)
            Dt = torch.tensor(chunk, dtype=torch.float32, device=device)
            pred = model(x_batch, in_vars=None, out_vars=None, Dt=Dt)
            out[start : start + chunk.shape[0]] = (
                pred.reshape(chunk.shape[0], N_BANDS).detach().cpu().numpy()
            )
    return out


def eval_object(
    real_stream,
    dense_stream,
    model,
    device,
    means,
    stds,
    context_window_days,
    max_context_len,
    late_time_cutoff_days,
    late_time_max_days,
):
    """Score one object's late-time dense truth against a realistic-context forecast.

    Returns a dict with the scored late-time points and a smooth forecast curve,
    or None if the object cannot be evaluated (no realistic context, or no dense
    points in the late-time region).

    The scored/forecast region is the phase band
    ``late_time_cutoff_days < phase <= late_time_max_days`` (phase measured from
    the first realistic detection). Points beyond ``late_time_max_days`` are
    ignored so the forecast is only judged over a horizon we care about.
    """
    r_t, r_v, r_b = real_stream
    d_t, d_v, d_b = dense_stream

    if r_t.shape[0] < 1 or d_t.shape[0] < 1:
        return None

    # Phase zero = the first realistic detection (the observed trigger).
    t0 = float(r_t[0])

    # The cutoff splits context from forecast: the model may only see realistic
    # detections up to the cutoff phase, and must FORECAST everything after it
    # (scored against the dense truth). Truncating the context here -- rather
    # than feeding the whole realistic stream and only scoring late points --
    # makes every object forecast from the same phase boundary, instead of from
    # wherever its realistic coverage happens to end. (Without this, a
    # bright/well-covered object whose realistic detections run to ~14 d has an
    # almost-zero forecast horizon and the curve collapses to a stub.)
    ctx_mask = (r_t - t0) <= late_time_cutoff_days
    if not np.any(ctx_mask):
        return None
    r_t_ctx = r_t[ctx_mask]
    r_v_ctx = r_v[ctx_mask]
    r_b_ctx = r_b[ctx_mask]
    last_real_t = float(r_t_ctx[-1])

    # Score the forecast only within the phase band cutoff < phase <= max_days.
    d_phase = d_t - t0
    late_mask = (d_phase > late_time_cutoff_days) & (d_phase <= late_time_max_days)
    if not np.any(late_mask):
        return None

    # Seed context from the truncated realistic stream: trailing window ending
    # at the last pre-cutoff realistic detection, normalized as in training.
    # build_context_input subtracts win_t[0], so absolute times are fine here.
    r_v_norm = (r_v_ctx - means[r_b_ctx]) / (stds[r_b_ctx] + EPS)
    win_v, win_t, win_b = _select_window(
        ctx_t=list(r_t_ctx.astype(np.float32)),
        ctx_v=list(r_v_norm.astype(np.float32)),
        ctx_b=list(r_b_ctx),
        context_window_days=context_window_days,
        max_context_len=max_context_len,
    )
    x = build_context_input(
        win_v=win_v,
        win_t=win_t,
        win_b=win_b,
        context_len=max_context_len,
        n_bands=N_BANDS,
        device=device,
        window_mode=True,
    )

    # Score each late-time dense point at its true lead time from the last
    # realistic detection. The context ``x`` is fixed for this object, so all
    # lead times are evaluated in a SINGLE batched forward pass (tile x to the
    # batch dimension, pass a Dt vector) instead of one forward per point --
    # numerically identical, but far faster on GPU.
    late_idx = np.nonzero(late_mask)[0]
    lead_times = (d_t[late_idx] - last_real_t).astype(np.float32)
    # Only points strictly after the last realistic detection are forecasts.
    keep = lead_times > 0
    late_idx = late_idx[keep]
    lead_times = lead_times[keep]

    if late_idx.shape[0] == 0:
        return None

    pred_scored = _batched_forward(model, x, lead_times, device)  # [P, N_BANDS]

    scored = []
    for j, idx in enumerate(late_idx):
        band = int(d_b[idx])
        pred_mag = float(pred_scored[j, band] * (stds[band] + EPS) + means[band])
        true_mag = float(d_v[idx])
        scored.append(
            {
                "phase": float(d_t[idx]) - t0,
                "lead_time": float(lead_times[j]),
                "band": band,
                "pred_mag": pred_mag,
                "true_mag": true_mag,
                "residual_mag": pred_mag - true_mag,
            }
        )

    # Smooth forecast curve for plotting: sweep lead time from 0 to the farthest
    # scored late-time point, predicting all bands at each lead time -- also a
    # single batched forward pass.
    max_dt = float(lead_times.max())
    lead_grid = np.linspace(0.0, max_dt, 60).astype(np.float32)
    curve = _batched_forward(model, x, lead_grid, device)  # [60, N_BANDS]
    curve_mag = curve * (stds[None, :] + EPS) + means[None, :]

    return {
        "scored": scored,
        "t0": t0,
        "last_real_t": last_real_t,
        "curve_phase": (last_real_t - t0) + lead_grid,
        "curve_mag": curve_mag.astype(np.float32),
        # Only the pre-cutoff realistic detections were shown to the model, so
        # plot those as the context (not the full realistic stream).
        "real": (r_t_ctx - t0, r_v_ctx, r_b_ctx),
        "dense": (d_t - t0, d_v, d_b),
        # Right edge for plotting: the scored horizon. Beyond this the forecast
        # is unsupervised extrapolation, so it is not shown.
        "plot_max_phase": late_time_max_days,
    }


def plot_object(result, stem, outpath):
    """Plot realistic context, dense truth, and the late-time forecast per band."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    axes = axes.ravel()
    r_ph, r_v, r_b = result["real"]
    d_ph, d_v, d_b = result["dense"]
    # Show only the scored horizon; the forecast beyond it is unsupervised
    # extrapolation (where the late-time upturn artifact lives).
    plot_max = result.get("plot_max_phase")

    for b in range(N_BANDS):
        ax = axes[b]
        rm = r_b == b
        dm = d_b == b
        # Clip the dense-truth scatter to the plotted horizon as well.
        if plot_max is not None:
            dm = dm & (d_ph <= plot_max)
        if np.any(dm):
            ax.scatter(d_ph[dm], d_v[dm], s=14, c="0.6", label="dense truth")
        if np.any(rm):
            ax.scatter(
                r_ph[rm], r_v[rm], s=26, c=BAND_COLORS[b],
                edgecolor="k", linewidth=0.4, label="realistic ctx",
            )
        ax.plot(
            result["curve_phase"], result["curve_mag"][:, b],
            c=BAND_COLORS[b], lw=1.6, label="forecast",
        )
        ax.invert_yaxis()  # magnitudes: brighter is smaller
        if plot_max is not None:
            ax.set_xlim(right=plot_max)
        ax.set_title(BAND_NAMES[b], fontsize=9)
        if b == 0:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"Dense late-time forecast: {stem}")
    fig.supxlabel("Phase from first realistic detection [days]")
    fig.supylabel("Magnitude")
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def get_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=int, default=24)
    p.add_argument("--epoch", type=int, default=500)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument(
        "--realistic_glob",
        type=str,
        default=(
            "/net/sescratch1/atoivonen/data/KN_lightcurves/"
            "rubin_ztf_10000_dataset_same_seed/lc_*.npz"
        ),
        help="Glob for the realistic light-curve files (observing context).",
    )
    p.add_argument(
        "--dense_glob",
        type=str,
        required=True,
        help="Glob for the dense light-curve files (late-time truth).",
    )
    p.add_argument(
        "--test_filelist",
        type=str,
        default=None,
        help="Path to the test-split stem list (one object stem per line). If "
        "omitted, all objects present in BOTH globs are evaluated.",
    )
    p.add_argument(
        "--norm_stats_path",
        type=str,
        default="kilonova_9band_norm_stats_trainonly.npz",
        help="Train-only normalization stats the model was trained with.",
    )
    p.add_argument(
        "--late_time_cutoff_days",
        type=float,
        default=3.0,
        help="Splits context from forecast. The model sees realistic detections "
        "with phase (from first realistic detection) up to this value, and "
        "forecasts all dense points after it -- the late-time region scored here.",
    )
    p.add_argument(
        "--late_time_max_days",
        type=float,
        default=10.0,
        help="Upper bound (phase from first realistic detection) on the scored "
        "forecast region. Dense points beyond this are ignored, so the forecast "
        "is judged only over cutoff < phase <= this horizon.",
    )
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument(
        "--max_objects",
        type=int,
        default=0,
        help="Cap the number of objects evaluated (0 = all).",
    )
    p.add_argument(
        "--n_plots",
        type=int,
        default=12,
        help="Number of per-object forecast plots to write.",
    )
    return p.parse_args()


def main():
    """Run the dense late-time evaluation."""
    args = get_args()
    tag = study_tag(args.study)
    if args.ckpt is None:
        args.ckpt = (
            f"runs/study_{tag}/study{tag}_modelState_epoch{args.epoch:04d}.pth"
        )
    if args.outdir is None:
        args.outdir = f"runs/study_{tag}/dense_latetime_eval_9band"
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        model,
        context_len,
        n_bands,
        context_window_days,
        max_context_len,
    ) = load_9band_model(args.ckpt, device)

    if context_window_days is None:
        raise ValueError(
            "This eval requires a time-window checkpoint (context_window_days "
            "set); the loaded checkpoint is fixed-count."
        )

    # Load the TRAIN-ONLY stats the model was trained with (loaded if present;
    # no eval-set recomputation).
    means, stds = load_or_compute_band_normalization(
        stats_path=args.norm_stats_path,
        band_keys=BAND_KEYS,
        value_col=VALUE_COL,
        error_col=ERROR_COL,
        drop_upper_limits=DROP_UPPER_LIMITS,
    )
    means = np.asarray(means, dtype=np.float32)
    stds = np.asarray(stds, dtype=np.float32)

    # Pair realistic and dense objects by stem, restricted to the test split.
    real_map = _stem_to_path(args.realistic_glob)
    dense_map = _stem_to_path(args.dense_glob)
    stems = sorted(set(real_map) & set(dense_map))

    if args.test_filelist is not None:
        with open(args.test_filelist) as fh:
            test_stems = {line.strip() for line in fh if line.strip()}
        stems = [s for s in stems if s in test_stems]
        print(f"Restricted to {len(stems)} test-split objects.")

    print(
        f"Realistic files: {len(real_map)}; dense files: {len(dense_map)}; "
        f"paired & in-split: {len(stems)}"
    )
    if args.max_objects > 0:
        stems = stems[: args.max_objects]

    all_scored = []
    plotted = 0
    n_eval = 0
    for stem in stems:
        real_stream = read_merged_stream(real_map[stem], DROP_UPPER_LIMITS)
        dense_stream = read_merged_stream(dense_map[stem], drop_upper_limits=False)
        result = eval_object(
            real_stream,
            dense_stream,
            model,
            device,
            means,
            stds,
            context_window_days,
            max_context_len,
            args.late_time_cutoff_days,
            args.late_time_max_days,
        )
        if result is None:
            continue
        n_eval += 1
        for s in result["scored"]:
            s["stem"] = stem
            all_scored.append(s)
        if plotted < args.n_plots:
            plot_object(
                result, stem,
                os.path.join(args.outdir, f"latetime_{stem}.png"),
            )
            plotted += 1

    if not all_scored:
        print("No late-time points scored (check the cutoff and globs).")
        return

    # Per-band late-time error summary.
    resid = np.asarray([s["residual_mag"] for s in all_scored])
    bands = np.asarray([s["band"] for s in all_scored])
    print(f"\nEvaluated {n_eval} objects; {len(all_scored)} late-time points "
          f"(cutoff {args.late_time_cutoff_days} d).")
    print(f"Overall late-time RMSE (mag): {np.sqrt(np.mean(resid**2)):.4f}  "
          f"MAE: {np.mean(np.abs(resid)):.4f}")
    print("Per-band late-time error (mag):")
    for b in range(N_BANDS):
        m = bands == b
        if np.any(m):
            print(f"  {BAND_NAMES[b]:>5}: n={m.sum():5d}  "
                  f"RMSE={np.sqrt(np.mean(resid[m]**2)):.4f}  "
                  f"MAE={np.mean(np.abs(resid[m])):.4f}  "
                  f"bias={np.mean(resid[m]):+.4f}")

    # Error vs lead time (binned) plot.
    lead = np.asarray([s["lead_time"] for s in all_scored])
    fig, ax = plt.subplots()
    edges = np.linspace(0, lead.max(), 11)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rmse_bin = np.full(centers.shape[0], np.nan)
    for i in range(centers.shape[0]):
        m = (lead >= edges[i]) & (lead < edges[i + 1])
        if np.any(m):
            rmse_bin[i] = np.sqrt(np.mean(resid[m] ** 2))
    ax.plot(centers, rmse_bin, "o-")
    ax.set_xlabel("Lead time from last realistic detection [days]")
    ax.set_ylabel("Late-time forecast RMSE [mag]")
    ax.set_title("Dense late-time forecast error vs lead time")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "latetime_rmse_vs_lead.png"), dpi=130)
    plt.close(fig)

    # Full per-point CSV.
    csv_path = os.path.join(args.outdir, "latetime_scored_points.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["stem", "band", "phase_days", "lead_time_days",
             "pred_mag", "true_mag", "residual_mag"]
        )
        for s in all_scored:
            w.writerow([
                s["stem"], BAND_NAMES[s["band"]], f"{s['phase']:.4f}",
                f"{s['lead_time']:.4f}", f"{s['pred_mag']:.4f}",
                f"{s['true_mag']:.4f}", f"{s['residual_mag']:.4f}",
            ])

    print(f"\nWrote {plotted} per-object plots, the RMSE-vs-lead plot, and "
          f"{csv_path} in {args.outdir}")


if __name__ == "__main__":
    main()
