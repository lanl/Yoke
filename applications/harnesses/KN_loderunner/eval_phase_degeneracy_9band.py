"""Phase-degeneracy diagnostic for the 9-band scalar temporal LodeRunner.

MOTIVATION
----------
The model never sees the true time since merger. Its only notion of time is
``rel_t`` (relative to the first observation in the context window) and ``Dt``
(lead time to the target) -- both measured from the observed trigger, not the
physical explosion. See ``Kilonova_lc_scalar_context_DataSet_9band._getitem_window``:
``rel_t = ctx_t - ctx_t[0]``.

Kilonova light curves are strongly phase-dependent (fast blue rise, then a
reddening decline whose rate changes over the following week). Two objects with
identical observed context but DIFFERENT true times-since-merger are at different
physical phases and evolve differently. Because the model conditions only on the
observed clock, that mapping is many-to-one degenerate: the same input can map to
different correct outputs depending on the UNOBSERVED phase. A degenerate target
cannot be fit sharply -- the model averages over the ambiguity, which plausibly
contributes to the under-fade bias and the floored median loss.

This script tests whether that degeneracy actually hurts, empirically:

  For each held-out object we know (from the noise-free UNIFORM-grid companion
  set) an object-independent physical phase zero. The "detection lag" is how long
  after that zero the first REALISTIC detection occurred:

      detection_lag = t0_realistic - t_phasezero_proxy

  This is exactly the hidden phase offset the model cannot observe. We then ask:
  do the late-time forecast residuals depend on detection_lag? If objects first
  detected LATE (large lag -- already well past peak, unknowingly) are
  systematically mis-forecast (e.g. under-fade / positive-then-negative bias)
  while objects caught EARLY are not, the model is being confounded by the
  unknown phase, and an explicit phase treatment (auxiliary phase-regression
  head, or a phase feature with an inference-time estimator) is warranted.

PHASE-ZERO PROXY
----------------
The npz files carry no merger-time column, so we proxy physical phase zero by the
earliest sample of the UNIFORM-grid companion (``--uniform_glob``), which is
sampled on a regular phase grid from near explosion with no limiting-mag cut.
This is a common, object-independent zero point -- all that a correlation
diagnostic requires. If your uniform set's grid does not start at merger, the lag
is offset by a constant across ALL objects, which shifts the x-axis but does NOT
change whether a trend exists (the thing we test). Override with an explicit npz
key via ``--phasezero_key`` if your files store one.

This is a READ-ONLY diagnostic: it loads a trained checkpoint, reuses the exact
scoring path of ``eval_dense_latetime_9band`` (same context, normalization, and
Dt convention), and writes plots + a CSV. It never trains.
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
    NINE_BAND_KEYS,
    load_or_compute_band_normalization,
)

# Run directly (not installed), so make the script directory importable and reuse
# the trusted loader + the exact per-object scoring used by the dense eval. This
# guarantees the residuals here are the SAME residuals that eval measures, just
# re-indexed by detection lag.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_pred_diagnostics_9band import load_9band_model  # noqa: E402
from eval_dense_latetime_9band import (  # noqa: E402
    BAND_NAMES,
    DROP_UPPER_LIMITS,
    N_BANDS,
    VALUE_COL,
    ERROR_COL,
    _stem_to_path,
    eval_object,
    read_merged_stream,
    study_tag,
)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rc("font", family="serif")
plt.rcParams["figure.figsize"] = (7, 5)

BAND_KEYS = NINE_BAND_KEYS


def _phasezero_proxy(
    uniform_stream, phasezero_key: str | None, uniform_path: str | None
) -> float | None:
    """Object-independent physical phase-zero proxy in absolute MJD.

    Preference order:
      1. An explicit scalar npz key (``--phasezero_key``), if present -- the true
         merger/explosion time when the data provides it.
      2. The earliest UNIFORM-grid sample time -- a regular phase grid starting
         near explosion, so its minimum is a stable per-object zero point.

    Args:
        uniform_stream (tuple): (times, values, bands) of the uniform companion,
            absolute MJD; times may be empty.
        phasezero_key (str | None): Optional npz key holding an explicit
            phase-zero time (scalar). Checked first when given.
        uniform_path (str | None): Path to the uniform npz, used only to read
            ``phasezero_key`` if requested.

    Returns:
        float | None: Phase-zero time in absolute MJD, or None if unavailable.
    """
    if phasezero_key is not None and uniform_path is not None:
        data = np.load(uniform_path, allow_pickle=True)
        try:
            if phasezero_key in data.files:
                return float(np.asarray(data[phasezero_key]).ravel()[0])
        finally:
            data.close()

    u_t, _, _ = uniform_stream
    if u_t is not None and u_t.shape[0] > 0:
        return float(u_t.min())
    return None


def _binned_stats(x: np.ndarray, resid: np.ndarray, edges: np.ndarray):
    """Per-bin count, bias (mean residual), and RMSE over ``x`` binned by edges.

    Args:
        x (np.ndarray): Binning variable (e.g. detection lag) per point.
        resid (np.ndarray): Residual (pred - true) magnitude per point.
        edges (np.ndarray): Bin edges.

    Returns:
        (centers, counts, bias, rmse): each length len(edges) - 1; empty bins are
        count 0 with NaN bias/rmse.
    """
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = np.zeros(centers.shape[0], dtype=int)
    bias = np.full(centers.shape[0], np.nan)
    rmse = np.full(centers.shape[0], np.nan)
    for i in range(centers.shape[0]):
        m = (x >= edges[i]) & (x < edges[i + 1])
        counts[i] = int(m.sum())
        if counts[i] > 0:
            bias[i] = float(np.mean(resid[m]))
            rmse[i] = float(np.sqrt(np.mean(resid[m] ** 2)))
    return centers, counts, bias, rmse


def get_args():
    """Parse command-line arguments (mirrors eval_dense_latetime_9band)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=int, default=24)
    p.add_argument("--epoch", type=int, default=500)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument(
        "--use_ema",
        action="store_true",
        help="Overlay the EMA (Polyak) shadow instead of raw weights.",
    )
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
        default=(
            "/net/sescratch1/atoivonen/data/KN_lightcurves/"
            "rubin_ztf_dense_10000_dataset_same_seed/lc_*.npz"
        ),
        help="Glob for the dense light-curve files (late-time truth).",
    )
    p.add_argument(
        "--uniform_glob",
        type=str,
        default=(
            "/net/sescratch1/atoivonen/data/KN_lightcurves/"
            "rubin_ztf_uniform_10000_dataset_same_seed/lc_*.npz"
        ),
        help="Glob for the UNIFORM-grid companion set. REQUIRED here: its "
        "earliest sample time is the physical phase-zero proxy that defines the "
        "detection lag. Objects with no matching uniform file are skipped.",
    )
    p.add_argument(
        "--phasezero_key",
        type=str,
        default=None,
        help="Optional npz key holding an explicit phase-zero (merger/explosion) "
        "time per object. When present it is used instead of the uniform-grid "
        "minimum, giving a true (not proxied) detection lag.",
    )
    p.add_argument(
        "--test_filelist",
        type=str,
        default=None,
        help="Path to the test-split stem list (one object stem per line).",
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
        default=2.0,
        help="Splits context from forecast (phase from first realistic "
        "detection). Matches the dense eval default.",
    )
    p.add_argument(
        "--late_time_max_days",
        type=float,
        default=10.0,
        help="Upper bound on the scored forecast region (phase from first "
        "realistic detection).",
    )
    p.add_argument(
        "--rollout",
        action="store_true",
        help="Score the AUTOREGRESSIVE forecast (true inference path) instead of "
        "the DIRECT single-pass forecast. Passed through to eval_object.",
    )
    p.add_argument(
        "--n_lag_bins",
        type=int,
        default=6,
        help="Number of detection-lag bins for the trend summary/plots.",
    )
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument(
        "--max_objects",
        type=int,
        default=0,
        help="Cap the number of objects evaluated (0 = all).",
    )
    return p.parse_args()


def main():
    """Run the phase-degeneracy diagnostic."""
    args = get_args()
    tag = study_tag(args.study)
    if args.ckpt is None:
        args.ckpt = (
            f"runs/study_{tag}/study{tag}_modelState_epoch{args.epoch:04d}.pth"
        )
    if args.outdir is None:
        args.outdir = f"runs/study_{tag}/phase_degeneracy_9band"
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        model,
        context_len,
        n_bands,
        context_window_days,
        max_context_len,
    ) = load_9band_model(args.ckpt, device, use_ema=getattr(args, "use_ema", False))

    if context_window_days is None:
        raise ValueError(
            "This diagnostic requires a time-window checkpoint "
            "(context_window_days set); the loaded checkpoint is fixed-count."
        )

    means, stds = load_or_compute_band_normalization(
        stats_path=args.norm_stats_path,
        band_keys=BAND_KEYS,
        value_col=VALUE_COL,
        error_col=ERROR_COL,
        drop_upper_limits=DROP_UPPER_LIMITS,
    )
    means = np.asarray(means, dtype=np.float32)
    stds = np.asarray(stds, dtype=np.float32)

    real_map = _stem_to_path(args.realistic_glob)
    dense_map = _stem_to_path(args.dense_glob)
    uniform_map = _stem_to_path(args.uniform_glob) if args.uniform_glob else {}
    if not uniform_map:
        raise ValueError(
            "No uniform-grid files matched --uniform_glob. This diagnostic needs "
            "the uniform set (or --phasezero_key) to define the physical "
            "phase-zero proxy."
        )

    # Objects must appear in all three sets: realistic (context), dense (truth),
    # and uniform (phase-zero proxy).
    stems = sorted(set(real_map) & set(dense_map) & set(uniform_map))

    if args.test_filelist is not None:
        with open(args.test_filelist) as fh:
            test_stems = {line.strip() for line in fh if line.strip()}
        stems = [s for s in stems if s in test_stems]
        print(f"Restricted to {len(stems)} test-split objects.")

    print(
        f"Realistic: {len(real_map)}; dense: {len(dense_map)}; "
        f"uniform: {len(uniform_map)}; paired & in-split: {len(stems)}"
    )
    if args.max_objects > 0:
        stems = stems[: args.max_objects]

    # Per-point rows: each late-time scored point, tagged with the object's
    # (hidden) detection lag and its true phase from the physical zero.
    rows = []
    # Per-object rows: one detection lag + object-level error summary.
    obj_rows = []
    n_eval = 0
    n_no_phasezero = 0

    for stem in stems:
        real_stream = read_merged_stream(real_map[stem], DROP_UPPER_LIMITS)
        dense_stream = read_merged_stream(dense_map[stem], drop_upper_limits=False)
        uniform_stream = read_merged_stream(
            uniform_map[stem], drop_upper_limits=False
        )

        t_phasezero = _phasezero_proxy(
            uniform_stream, args.phasezero_key, uniform_map.get(stem)
        )
        if t_phasezero is None:
            n_no_phasezero += 1
            continue

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
            uniform_stream=uniform_stream,
            rollout=args.rollout,
        )
        if result is None:
            continue
        n_eval += 1

        # Detection lag: how long after physical phase zero the first realistic
        # detection occurred. This is the hidden phase offset the model cannot
        # observe -- the whole point of the diagnostic.
        t0_real = float(result["t0"])
        detection_lag = t0_real - t_phasezero

        obj_resid = []
        for s in result["scored"]:
            r = float(s["residual_mag"])
            obj_resid.append(r)
            rows.append(
                {
                    "stem": stem,
                    "band": int(s["band"]),
                    "detection_lag": detection_lag,
                    # Phase from the OBSERVED trigger (what the eval reports).
                    "obs_phase": float(s["phase"]),
                    # Phase from the PHYSICAL zero = observed phase + hidden lag.
                    "true_phase": float(s["phase"]) + detection_lag,
                    "lead_time": float(s["lead_time"]),
                    "pred_mag": float(s["pred_mag"]),
                    "true_mag": float(s["true_mag"]),
                    "residual_mag": r,
                }
            )

        if obj_resid:
            obj_resid = np.asarray(obj_resid)
            obj_rows.append(
                {
                    "stem": stem,
                    "detection_lag": detection_lag,
                    "n_points": obj_resid.shape[0],
                    "bias": float(np.mean(obj_resid)),
                    "rmse": float(np.sqrt(np.mean(obj_resid ** 2))),
                }
            )

    if not rows:
        print("No late-time points scored (check cutoffs, globs, phase-zero).")
        return
    if n_no_phasezero:
        print(f"Skipped {n_no_phasezero} objects with no phase-zero proxy.")

    lag = np.asarray([r["detection_lag"] for r in rows])
    resid = np.asarray([r["residual_mag"] for r in rows])
    bands = np.asarray([r["band"] for r in rows])
    mode = "AUTOREGRESSIVE rollout" if args.rollout else "DIRECT single-pass"

    print(f"\nForecast mode: {mode}")
    print(f"Evaluated {n_eval} objects; {len(rows)} late-time points.")
    print(
        f"Detection lag (days after phase-zero proxy): "
        f"min={lag.min():.2f} median={np.median(lag):.2f} max={lag.max():.2f}"
    )

    # --- Core test: does residual depend on the hidden detection lag? ----------
    # A near-zero, flat trend => the observed context already constrains phase
    # (the degeneracy is not hurting). A monotone bias trend => the model is
    # confounded by unknown phase, and explicit phase handling should help.
    if lag.max() > lag.min():
        # Pearson correlation of residual with lag (sign matters: positive means
        # late-detected objects are over-predicted in magnitude i.e. under-faded).
        corr = float(np.corrcoef(lag, resid)[0, 1])
        # Least-squares slope in mag per day of hidden lag.
        slope, intercept = np.polyfit(lag, resid, 1)
        print(
            f"residual ~ detection_lag: corr={corr:+.3f}  "
            f"slope={slope:+.4f} mag/day  intercept={intercept:+.4f} mag"
        )
    else:
        corr, slope, intercept = np.nan, np.nan, np.nan
        print("All objects share one detection lag; trend undefined.")

    edges = np.linspace(lag.min(), lag.max(), args.n_lag_bins + 1)
    centers, counts, bin_bias, bin_rmse = _binned_stats(lag, resid, edges)
    print("\nLate-time error binned by detection lag:")
    print(f"  {'lag[d]':>8}  {'n':>6}  {'bias':>8}  {'rmse':>8}")
    for c, n, b, r in zip(centers, counts, bin_bias, bin_rmse):
        if n > 0:
            print(f"  {c:8.2f}  {n:6d}  {b:+8.4f}  {r:8.4f}")

    # --- Plot 1: residual bias & RMSE vs detection lag -------------------------
    fig, ax = plt.subplots()
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
    ax.plot(centers, bin_bias, "o-", color="#E63946", label="bias (mean resid)")
    ax.plot(centers, bin_rmse, "s-", color="#457B9D", label="RMSE")
    ax.set_xlabel("Detection lag: first realistic detection − phase-zero [days]")
    ax.set_ylabel("Late-time residual [mag]  (pred − true)")
    ax.set_title(
        f"Phase degeneracy: forecast error vs hidden detection lag\n"
        f"corr={corr:+.3f}, slope={slope:+.4f} mag/day ({mode})"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "residual_vs_detection_lag.png"), dpi=130)
    plt.close(fig)

    # --- Plot 2: per-band bias vs detection lag --------------------------------
    # Different bands carry different phase information (blue bands fade fastest),
    # so a band-dependent lag trend localizes where the degeneracy bites.
    fig, ax = plt.subplots()
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
    for b in range(N_BANDS):
        mb = bands == b
        if mb.sum() < 5 or lag[mb].max() == lag[mb].min():
            continue
        _, cnts_b, bias_b, _ = _binned_stats(lag[mb], resid[mb], edges)
        ax.plot(centers, bias_b, "o-", ms=3, lw=1.0, label=BAND_NAMES[b])
    ax.set_xlabel("Detection lag [days]")
    ax.set_ylabel("Bias [mag]  (pred − true)")
    ax.set_title(f"Per-band late-time bias vs detection lag ({mode})")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "per_band_bias_vs_lag.png"), dpi=130)
    plt.close(fig)

    # --- Plot 3: object-level lag scatter --------------------------------------
    if obj_rows:
        o_lag = np.asarray([o["detection_lag"] for o in obj_rows])
        o_bias = np.asarray([o["bias"] for o in obj_rows])
        fig, ax = plt.subplots()
        ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
        ax.scatter(o_lag, o_bias, s=14, c="#2A9D8F", alpha=0.6)
        ax.set_xlabel("Detection lag [days]")
        ax.set_ylabel("Per-object mean residual [mag]")
        ax.set_title(f"Per-object forecast bias vs detection lag ({mode})")
        fig.tight_layout()
        fig.savefig(
            os.path.join(args.outdir, "per_object_bias_vs_lag.png"), dpi=130
        )
        plt.close(fig)

    # --- CSVs ------------------------------------------------------------------
    pt_csv = os.path.join(args.outdir, "phase_degeneracy_points.csv")
    with open(pt_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["stem", "band", "detection_lag_days", "obs_phase_days",
             "true_phase_days", "lead_time_days", "pred_mag", "true_mag",
             "residual_mag"]
        )
        for r in rows:
            w.writerow([
                r["stem"], BAND_NAMES[r["band"]], f"{r['detection_lag']:.4f}",
                f"{r['obs_phase']:.4f}", f"{r['true_phase']:.4f}",
                f"{r['lead_time']:.4f}", f"{r['pred_mag']:.4f}",
                f"{r['true_mag']:.4f}", f"{r['residual_mag']:.4f}",
            ])

    obj_csv = os.path.join(args.outdir, "phase_degeneracy_objects.csv")
    with open(obj_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "detection_lag_days", "n_points", "bias_mag", "rmse_mag"])
        for o in obj_rows:
            w.writerow([
                o["stem"], f"{o['detection_lag']:.4f}", o["n_points"],
                f"{o['bias']:.4f}", f"{o['rmse']:.4f}",
            ])

    print(
        f"\nWrote residual_vs_detection_lag.png, per_band_bias_vs_lag.png, "
        f"per_object_bias_vs_lag.png, {pt_csv}, and {obj_csv} in {args.outdir}"
    )
    print(
        "\nHow to read it: a flat, near-zero bias trend means the observed "
        "context already constrains phase (degeneracy not hurting). A trend that "
        "grows with detection lag means late-detected (unknowingly post-peak) "
        "objects are systematically mis-forecast -- evidence that adding explicit "
        "phase handling (auxiliary phase-regression head, or a phase feature with "
        "an inference-time estimator) should raise the ceiling."
    )


if __name__ == "__main__":
    main()
