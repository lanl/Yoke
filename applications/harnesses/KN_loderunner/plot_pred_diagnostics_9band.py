"""Autoregressive rollout diagnostics for the 9-band scalar temporal LodeRunner.

This is the 9-band analogue of ``plot_pred_diagnostics_gri.py``. Rather than
scoring a single next-event prediction, it produces an autoregressive forecast of
the next several observations and feeds the model's own predictions back into the
context, exactly like the g/r/i rollout.

The 9-band data is a merged, time-sorted event stream where each observation
belongs to a single filter. The model, however, emits a prediction for ALL nine
bands at any requested lead time. So a rollout proceeds as:

  1. Start from a context window of ``context_len`` true events.
  2. Predict all 9 bands at the next event's lead time Dt.
  3. The next true event is in one filter; take the model's prediction for that
     filter as the forecast value, append it (with the event's true time and
     band) back into the context, and drop the oldest event.
  4. Repeat for ``n_future_steps``, so later predictions are conditioned on
     earlier predictions.

Because the observation schedule (times + which filter is seen next) is taken
from the truth while the values are fed back from the model, this measures how
well the model forecasts future observations in each filter over a rollout.

At every step the model emits a prediction for ALL nine bands, including bands
with no context and bands not observed at that step. The per-series plots show
this full all-band forecast, overlaid with truth wherever an observation exists.
Only the observed band is fed back into the context and scored against truth.
"""

import argparse
import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from yoke.models.vit.swin.bomberman import (
    LodeRunner,
    ScalarTemporalConditionedLodeRunner_9band,
)
from yoke.datasets.kilonova_dataset import (
    EPS,
    NINE_BAND_KEYS,
    Kilonova_lc_scalar_context_DataSet_9band,
    load_or_compute_band_normalization,
)


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rc("font", family="serif")
plt.rcParams["figure.figsize"] = (7, 5)


BAND_KEYS = NINE_BAND_KEYS
BAND_NAMES = ("ztfg", "ztfr", "ztfi", "u", "g", "r", "i", "z", "y")
BAND_COLORS = (
    "#2A9D8F",  # ztfg
    "#E63946",  # ztfr
    "#F4A261",  # ztfi
    "#457B9D",  # u
    "#1B9E77",  # g
    "#D62828",  # r
    "#E9C46A",  # i
    "#8338EC",  # z
    "#264653",  # y
)
VALUE_COL = 1
ERROR_COL = 2
N_BANDS = len(BAND_KEYS)

# Match training: drop upper-limit (non-detection) observations, flagged by a
# non-finite uncertainty in ERROR_COL.
DROP_UPPER_LIMITS = True


def study_tag(study):
    return f"{int(study):03d}"


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Autoregressive rollout diagnostics for the scalar temporal "
            "LodeRunner 9-band model."
        )
    )

    parser.add_argument("--study", type=int, default=24)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument(
        "--N_imgs",
        type=int,
        default=50,
        help="Number of light-curve files to load into the eval dataset.",
    )
    parser.add_argument(
        "--n_future_steps",
        type=int,
        default=15,
        help="Number of future events to forecast autoregressively per series.",
    )
    parser.add_argument(
        "--n_series",
        type=int,
        default=10,
        help="Number of light curves to roll out.",
    )
    parser.add_argument(
        "--teacher_forced",
        action="store_true",
        help="Also compute a teacher-forced rollout (true values fed back "
        "instead of predictions) and overlay it on the free-running rollout to "
        "diagnose compounding rollout error. Off by default.",
    )

    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument(
        "--norm_stats_path",
        type=str,
        default="kilonova_9band_norm_stats_trainonly.npz",
        help="Train-only normalization stats the model was trained with. Must "
        "match training so plots use the exact encoding the model saw.",
    )
    parser.add_argument(
        "--data_glob",
        type=str,
        default=(
            "/net/sescratch1/atoivonen/data/KN_lightcurves/"
            "rubin_ztf_10000_dataset_same_seed/lc_*.npz"
        ),
        help="Glob for the realistic light-curve files to diagnose.",
    )
    parser.add_argument(
        "--test_filelist",
        type=str,
        default=None,
        help="Path to the test-split stem list (one object stem per line). When "
        "set, diagnostics run ONLY on held-out test objects, so plots are not "
        "leaked by training data. Omit to run over every object in --data_glob.",
    )

    return parser.parse_args()


def resolve_paths(args):
    tag = study_tag(args.study)

    if args.ckpt is None:
        args.ckpt = (
            f"runs/study_{tag}/study{tag}_modelState_epoch{args.epoch:04d}.pth"
        )

    if args.outdir is None:
        args.outdir = f"runs/study_{tag}/autoreg_diagnostics_9band"

    return tag


def strip_ddp_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
        }
    return state_dict


def load_9band_model(ckpt_path, device):
    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    model_args = ckpt["model_args"]
    context_len = ckpt.get("context_len", 5)

    n_bands = ckpt.get("n_bands", N_BANDS)
    backbone_channels = ckpt.get("backbone_channels", 8)
    hidden = ckpt.get("hidden", 64)
    noise_scale = ckpt.get("noise_scale", 0.0)

    # Time-window context mode. When set, the model's first layer is sized by the
    # padded width (max_context_len) and each event carries an extra validity
    # flag. Legacy fixed-count checkpoints fall through to None.
    context_window_days = ckpt.get("context_window_days", None)
    if context_window_days is not None:
        max_context_len = ckpt.get("max_context_len", context_len)
    else:
        max_context_len = context_len

    print("Loaded checkpoint:", ckpt_path)
    print("model_class:", ckpt.get("model_class", "unknown"))
    print("backbone_class:", ckpt.get("backbone_class", "LodeRunner"))
    print("target_type:", ckpt.get("target_type", "unknown"))
    print("context_len:", context_len)
    print("context_window_days:", context_window_days)
    print("max_context_len:", max_context_len)
    print("n_bands:", n_bands)
    print("band_keys:", ckpt.get("band_keys", list(BAND_KEYS)))
    print("backbone_channels:", backbone_channels)
    print("hidden:", hidden)

    backbone = LodeRunner(**model_args).to(device)
    backbone.noise_scale = noise_scale

    model = ScalarTemporalConditionedLodeRunner_9band(
        backbone=backbone,
        context_len=max_context_len,
        n_bands=n_bands,
        image_size=model_args["image_size"],
        backbone_channels=backbone_channels,
        hidden=hidden,
        context_window_days=context_window_days,
    ).to(device)

    state_dict = strip_ddp_prefix(ckpt["model_state_dict"])

    missing, unexpected = model.load_state_dict(state_dict, strict=True)

    print("Loaded ScalarTemporalConditionedLodeRunner_9band checkpoint")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.eval()

    return model, context_len, n_bands, context_window_days, max_context_len


def make_eval_dataset(
    args,
    context_len,
    context_window_days=None,
    max_context_len=None,
):
    band_means, band_stds = load_or_compute_band_normalization(
        stats_path=args.norm_stats_path,
        band_keys=BAND_KEYS,
        value_col=VALUE_COL,
        error_col=ERROR_COL,
        drop_upper_limits=DROP_UPPER_LIMITS,
    )

    print("Using band normalization:")
    print("band_means:", band_means)
    print("band_stds:", band_stds)

    # Restrict to held-out test objects when a split list is given, so the
    # diagnostics are not leaked by training data.
    object_ids = None
    if getattr(args, "test_filelist", None):
        with open(args.test_filelist) as fh:
            object_ids = {line.strip() for line in fh if line.strip()}
        print(f"Restricting diagnostics to {len(object_ids)} test-split objects.")

    dataset = Kilonova_lc_scalar_context_DataSet_9band(
        N_imgs=args.N_imgs,
        context_len=context_len,
        band_keys=BAND_KEYS,
        value_col=VALUE_COL,
        error_col=ERROR_COL,
        drop_upper_limits=DROP_UPPER_LIMITS,
        means=band_means,
        stds=band_stds,
        context_window_days=context_window_days,
        max_context_len=max_context_len,
        data_glob=getattr(args, "data_glob", None),
        object_ids=object_ids,
    )

    return dataset, np.asarray(band_means), np.asarray(band_stds)


def build_context_input(
    win_v,
    win_t,
    win_b,
    context_len,
    n_bands,
    device,
    window_mode=False,
):
    """Build the flattened per-event context input for the model.

    Fixed-count mode (``window_mode=False``): layout per event is
    ``[value, rel_t, one_hot_band(n_bands)]`` (width ``2 + n_bands``), with
    ``rel_t`` measured from the first event in the window and exactly
    ``context_len`` real events, matching the fixed-count dataset path.

    Time-window mode (``window_mode=True``): the (real) events are padded to
    ``context_len`` (= ``max_context_len``) rows and each event gains a validity
    flag, giving ``[value, rel_t, valid, one_hot_band(n_bands)]`` (width
    ``3 + n_bands``). Real events fill the leading rows in time order with
    ``rel_t`` relative to the first real event; padded rows are all-zero with
    ``valid = 0``. This matches ``_getitem_window`` in the dataset.
    """
    win_v = np.asarray(win_v, dtype=np.float32)
    win_t = np.asarray(win_t, dtype=np.float32)
    win_b = np.asarray(win_b, dtype=np.int64)

    if window_mode:
        n_real = win_v.shape[0]
        rel_t = (win_t - win_t[0]).astype(np.float32)

        per_event = np.zeros((context_len, 3 + n_bands), dtype=np.float32)
        per_event[:n_real, 0] = win_v
        per_event[:n_real, 1] = rel_t
        per_event[:n_real, 2] = 1.0  # validity flag for real events
        per_event[np.arange(n_real), 3 + win_b] = 1.0
    else:
        rel_t = (win_t - win_t[0]).astype(np.float32)

        band_onehot = np.zeros((context_len, n_bands), dtype=np.float32)
        band_onehot[np.arange(context_len), win_b] = 1.0

        per_event = np.concatenate(
            [win_v[:, None], rel_t[:, None], band_onehot],
            axis=1,
        )

    return torch.tensor(
        per_event.reshape(-1),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)


def _select_window(ctx_t, ctx_v, ctx_b, context_window_days, max_context_len):
    """Select the trailing time-window subset of a growing context.

    Mirrors ``_getitem_window`` in the dataset: keep every event within
    ``context_window_days`` of the most recent context event, then keep the most
    recent ``max_context_len`` if more qualify. Returns (win_v, win_t, win_b) as
    lists in time order (oldest first).
    """
    ct = np.asarray(ctx_t, dtype=np.float32)
    cv = np.asarray(ctx_v, dtype=np.float32)
    cb = np.asarray(ctx_b, dtype=np.int64)

    anchor_t = ct[-1]
    lo = anchor_t - context_window_days
    sel_idx = np.nonzero(ct >= lo)[0]
    if sel_idx.shape[0] > max_context_len:
        sel_idx = sel_idx[-max_context_len:]

    return (
        list(cv[sel_idx]),
        list(ct[sel_idx]),
        list(cb[sel_idx]),
    )


def get_rollout_from_stream(
    times,
    values,
    bands,
    model,
    device,
    start_idx,
    n_future_steps,
    context_len,
    n_bands,
    means,
    stds,
    teacher_forced=False,
    window_mode=False,
    context_window_days=None,
    max_context_len=None,
):
    """Autoregressively forecast the next events of one merged event stream.

    Args:
        times (np.ndarray): Relative event times for the file [N].
        values (np.ndarray): Normalized event values [N].
        bands (np.ndarray): Band index per event [N].
        start_idx (int): Index of the first context event.
        n_future_steps (int): Number of future events to forecast.
        context_len (int): Context window length.
        n_bands (int): Number of bands.
        means, stds (np.ndarray): Per-band normalization for denormalizing.
        teacher_forced (bool): If True, feed the TRUE observed value back into
            the context at each step instead of the model's own prediction. This
            isolates one-step-ahead skill from compounding rollout error: a model
            that tracks truth when teacher-forced but drifts when free-running is
            suffering from exposure bias, not a failure to learn the dynamics.

    Returns:
        dict describing the rollout (context, per-step forecasts, per-band MSE).
    """
    t_ref = float(times[start_idx])

    # Number of true events used to warm-start the running context. In window
    # mode we seed up to max_context_len so the trailing-W-days selection has
    # enough events to draw from; otherwise the fixed count. Clamp so at least
    # one true event remains past the seed for a future step (lets short curves,
    # now trainable in window mode, still roll out).
    requested_seed = max_context_len if window_mode else context_len
    seed_len = min(requested_seed, len(times) - start_idx - 1)
    seed_len = max(1, seed_len)

    # Running context window; values are fed back from predictions as we roll
    # out, while times and band identities follow the true observation schedule.
    ctx_t = list(times[start_idx : start_idx + seed_len].astype(np.float32))
    ctx_v = list(values[start_idx : start_idx + seed_len].astype(np.float32))
    ctx_b = list(bands[start_idx : start_idx + seed_len].astype(np.int64))

    context = {
        "t_rel": np.asarray(ctx_t, dtype=np.float32) - t_ref,
        "v_norm": np.asarray(ctx_v, dtype=np.float32),
        "band": np.asarray(ctx_b, dtype=np.int64),
    }

    steps = []
    band_sq_err = [[] for _ in range(n_bands)]

    with torch.no_grad():
        for step in range(n_future_steps):
            target_idx = start_idx + seed_len + step

            if target_idx >= len(times):
                break

            if window_mode:
                # Time-window context: trailing W days of all events observed
                # so far, padded to max_context_len (matches _getitem_window).
                win_v, win_t, win_b = _select_window(
                    ctx_t=ctx_t,
                    ctx_v=ctx_v,
                    ctx_b=ctx_b,
                    context_window_days=context_window_days,
                    max_context_len=max_context_len,
                )
                build_width = max_context_len
            else:
                win_v = ctx_v[-context_len:]
                win_t = ctx_t[-context_len:]
                win_b = ctx_b[-context_len:]
                build_width = context_len

            x = build_context_input(
                win_v=win_v,
                win_t=win_t,
                win_b=win_b,
                context_len=build_width,
                n_bands=n_bands,
                device=device,
                window_mode=window_mode,
            )

            # Lead time from the last context event to the next true event.
            Dt = torch.tensor(
                [float(times[target_idx]) - win_t[-1]],
                dtype=torch.float32,
                device=device,
            )

            pred_all = model(x, in_vars=None, out_vars=None, Dt=Dt)
            pred_all = pred_all.reshape(n_bands).detach().cpu().numpy()

            # The model predicts every band at this lead time, including bands
            # with no context and bands not observed at this step. Keep the full
            # 9-band prediction (normalized and denormalized) for plotting.
            pred_all_norm = pred_all.astype(np.float32)
            pred_all_mag = (
                pred_all_norm * (stds + EPS) + means
            ).astype(np.float32)

            target_band = int(bands[target_idx])

            pred_norm = float(pred_all[target_band])
            true_norm = float(values[target_idx])
            residual = pred_norm - true_norm

            pred_mag = pred_norm * (stds[target_band] + EPS) + means[target_band]
            true_mag = true_norm * (stds[target_band] + EPS) + means[target_band]

            band_sq_err[target_band].append(residual**2)

            steps.append(
                {
                    "step": step,
                    "t_rel": float(times[target_idx]) - t_ref,
                    "band": target_band,
                    "pred_norm": pred_norm,
                    "true_norm": true_norm,
                    "pred_mag": float(pred_mag),
                    "true_mag": float(true_mag),
                    "residual": residual,
                    # Full all-band prediction at this step's lead time.
                    "pred_all_norm": pred_all_norm,
                    "pred_all_mag": pred_all_mag,
                }
            )

            # Feed the newest context event, following the true schedule (time +
            # band) for the observation just forecast. Free-running rollout feeds
            # the model's own prediction back in; teacher forcing feeds the true
            # value instead, so errors do not compound down the rollout.
            ctx_t.append(float(times[target_idx]))
            ctx_v.append(true_norm if teacher_forced else pred_norm)
            ctx_b.append(target_band)

    residuals = np.asarray([s["residual"] for s in steps], dtype=np.float32)
    total_mse = float(np.mean(residuals**2)) if len(residuals) else np.nan

    band_mse = np.full(n_bands, np.nan, dtype=np.float32)
    for b in range(n_bands):
        if band_sq_err[b]:
            band_mse[b] = float(np.mean(band_sq_err[b]))

    # Smooth "forecast from now": hold the initial context window fixed and sweep
    # a monotonic lead-time grid, predicting all bands at each lead time. This is
    # the physically meaningful forecast (matches infer_9band.py) and, unlike the
    # per-step autoregressive predictions, does not sawtooth, because the lead
    # time increases monotonically instead of oscillating with the true event
    # cadence. Computed only for the free-run rollout (teacher forcing has no
    # meaning without feedback).
    fixed_forecast = None
    if not teacher_forced and steps:
        win_t0 = times[start_idx : start_idx + seed_len].astype(np.float32)
        win_v0 = values[start_idx : start_idx + seed_len].astype(np.float32)
        win_b0 = bands[start_idx : start_idx + seed_len].astype(np.int64)

        if window_mode:
            # Trailing W days of the seeded context, padded to max_context_len.
            win_v0, win_t0, win_b0 = _select_window(
                ctx_t=win_t0,
                ctx_v=win_v0,
                ctx_b=win_b0,
                context_window_days=context_window_days,
                max_context_len=max_context_len,
            )
            build_width0 = max_context_len
        else:
            build_width0 = context_len

        x0 = build_context_input(
            win_v=win_v0,
            win_t=win_t0,
            win_b=win_b0,
            context_len=build_width0,
            n_bands=n_bands,
            device=device,
            window_mode=window_mode,
        )

        last_ctx_t_rel = float(win_t0[-1]) - t_ref
        max_step_t_rel = max(s["t_rel"] for s in steps)
        horizon = max(1e-3, max_step_t_rel - last_ctx_t_rel)

        n_lead = 60
        lead_times = np.linspace(0.0, horizon, n_lead).astype(np.float32)
        preds_norm = np.zeros((n_lead, n_bands), dtype=np.float32)

        with torch.no_grad():
            for k, dt in enumerate(lead_times):
                Dt = torch.tensor([dt], dtype=torch.float32, device=device)
                pred = model(x0, in_vars=None, out_vars=None, Dt=Dt)
                preds_norm[k] = pred.reshape(n_bands).detach().cpu().numpy()

        preds_mag = preds_norm * (stds[None, :] + EPS) + means[None, :]

        fixed_forecast = {
            "t_rel": last_ctx_t_rel + lead_times,
            "preds_mag": preds_mag.astype(np.float32),
        }

    return {
        "start_idx": start_idx,
        "context": context,
        "steps": steps,
        "mse": total_mse,
        "band_mse": band_mse,
        "n_steps": len(steps),
        "fixed_forecast": fixed_forecast,
    }


def select_series(
    dataset,
    context_len,
    n_future_steps,
    n_series,
    seed_len=None,
):
    """Pick files with enough events for a rollout, longest first.

    Returns a list of (times, values, bands, start_idx) tuples.
    """
    # In window mode the context is seeded with up to max_context_len events
    # (passed as seed_len); otherwise the fixed count. Either way we need at
    # least one event past the seed for a future step.
    warm = seed_len if seed_len is not None else context_len
    min_events = warm + 1  # need at least one future step

    eligible = []
    for times, values, bands in dataset.events_per_file:
        if len(times) >= min_events:
            eligible.append((times, values, bands))

    # Prefer the longest streams so rollouts have the most future steps.
    eligible.sort(key=lambda tvb: len(tvb[0]), reverse=True)

    selected = []
    for times, values, bands in eligible[:n_series]:
        # Start at the beginning; the rollout naturally stops at the end of the
        # stream if fewer than n_future_steps events remain.
        selected.append((times, values, bands, 0))

    return selected


def _band_forecast_curve(steps, band_idx):
    """Return (t_rel, pred_mag) for a band across all steps, sorted by time."""
    t_all = np.asarray([s["t_rel"] for s in steps])
    pred_all = np.asarray([s["pred_all_mag"][band_idx] for s in steps])
    order = np.argsort(t_all)
    return t_all[order], pred_all[order]


def plot_series_lightcurves(rollout, means, stds, outpath, tf_rollout=None):
    """Plot one rollout's context, truth, and forecast per band (3x3 grid).

    If ``tf_rollout`` (the teacher-forced rollout for the same series) is given,
    its forecast is overlaid so free-running vs teacher-forced can be compared
    directly: divergence between the two indicates compounding rollout error
    (exposure bias) rather than a failure to learn one-step dynamics.
    """
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()

    context = rollout["context"]
    steps = rollout["steps"]
    tf_steps = tf_rollout["steps"] if tf_rollout is not None else None
    fixed_forecast = rollout.get("fixed_forecast")

    for band_idx in range(N_BANDS):
        ax = axes[band_idx]
        color = BAND_COLORS[band_idx]

        def denorm(v):
            return v * (stds[band_idx] + EPS) + means[band_idx]

        # Context observations that belong to this band.
        ctx_mask = context["band"] == band_idx
        if np.any(ctx_mask):
            ax.scatter(
                context["t_rel"][ctx_mask],
                denorm(context["v_norm"][ctx_mask]),
                s=30,
                color=color,
                marker="o",
                label="context",
            )

        # Headline forecast: smooth "forecast from now" over a monotonic lead
        # time grid, holding the initial context fixed. This replaces the old
        # per-step connected curve, whose sawtooth was an artifact of the lead
        # time Dt oscillating with the true (nightly-clustered) event cadence
        # rather than any model instability.
        if fixed_forecast is not None:
            ax.plot(
                fixed_forecast["t_rel"],
                fixed_forecast["preds_mag"][:, band_idx],
                "-",
                color="k",
                linewidth=1.8,
                alpha=0.85,
                label="forecast (fixed context)",
            )

        # Per-step autoregressive predictions as UNCONNECTED markers. Each is
        # made at that step's own lead time, so connecting them produces the
        # sawtooth; shown as points they still reveal free-run bias vs truth
        # without the misleading zigzag line.
        if steps:
            t_all, pred_all = _band_forecast_curve(steps, band_idx)
            ax.scatter(
                t_all,
                pred_all,
                s=22,
                facecolors="none",
                edgecolors="k",
                alpha=0.6,
                label="per-step pred (free-run)",
            )

        # Teacher-forced per-step predictions, also as unconnected markers.
        if tf_steps:
            t_tf, pred_tf = _band_forecast_curve(tf_steps, band_idx)
            ax.scatter(
                t_tf,
                pred_tf,
                s=22,
                marker="^",
                facecolors="none",
                edgecolors="tab:purple",
                alpha=0.6,
                label="per-step pred (teacher-forced)",
            )

        # Truth for this band, at the steps where it was actually observed.
        b_steps = [s for s in steps if s["band"] == band_idx]
        if b_steps:
            t = np.asarray([s["t_rel"] for s in b_steps])
            true_mag = np.asarray([s["true_mag"] for s in b_steps])

            order = np.argsort(t)
            t = t[order]
            true_mag = true_mag[order]

            ax.plot(
                t, true_mag, "-o", color=color, alpha=0.8, label="truth (obs)"
            )

        ax.axvline(0.0, color="gray", linewidth=1, linestyle=":", alpha=0.7)
        ax.invert_yaxis()
        ax.set_xlabel("Relative time (days)")
        ax.set_ylabel("Magnitude")
        ax.set_title(BAND_NAMES[band_idx])

        # Only add a legend if this band actually plotted labeled artists.
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"Autoregressive forecast (start_idx={rollout['start_idx']}, "
        f"{rollout['n_steps']} steps)",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_residuals_vs_step(rollouts, outpath):
    """Residual vs autoregressive step, colored by band, across all series."""
    plt.figure(figsize=(10, 6))

    for band_idx in range(N_BANDS):
        xs = []
        ys = []
        for rollout in rollouts:
            for s in rollout["steps"]:
                if s["band"] == band_idx:
                    xs.append(s["step"])
                    ys.append(s["residual"])
        if xs:
            plt.scatter(
                xs,
                ys,
                s=20,
                alpha=0.6,
                color=BAND_COLORS[band_idx],
                label=BAND_NAMES[band_idx],
            )

    plt.axhline(0.0, linestyle="--", linewidth=1, color="k")
    plt.xlabel("Autoregressive step")
    plt.ylabel("pred - truth (normalized)")
    plt.title("Autoregressive residuals vs step, by band")
    plt.legend(fontsize=7, ncol=3, loc="best")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def _per_band_mse(rollouts):
    """Aggregate per-band MSE over all steps and series."""
    band_sq = [[] for _ in range(N_BANDS)]
    for rollout in rollouts:
        for s in rollout["steps"]:
            band_sq[s["band"]].append(s["residual"] ** 2)
    return np.asarray(
        [float(np.mean(band_sq[b])) if band_sq[b] else 0.0 for b in range(N_BANDS)]
    )


def plot_band_mse_bar(rollouts, outpath, tf_rollouts=None):
    """Per-band MSE aggregated over all rollout steps and series.

    If ``tf_rollouts`` is given, free-running and teacher-forced MSE are shown as
    grouped bars per band. A large free-run bar next to a small teacher-forced
    bar is the signature of compounding rollout error (exposure bias).
    """
    mses = _per_band_mse(rollouts)

    plt.figure(figsize=(10, 5))
    x = np.arange(N_BANDS)

    if tf_rollouts is not None:
        tf_mses = _per_band_mse(tf_rollouts)
        width = 0.4
        plt.bar(x - width / 2, mses, width, color="tab:gray", label="free-run")
        plt.bar(
            x + width / 2,
            tf_mses,
            width,
            color="tab:purple",
            label="teacher-forced",
        )
        plt.legend(fontsize=9)
    else:
        plt.bar(x, mses, color=list(BAND_COLORS))

    plt.xticks(x, list(BAND_NAMES))
    plt.ylabel("Autoregressive MSE (normalized)")
    plt.xlabel("Band")
    plt.title("Per-band autoregressive rollout MSE")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_mse_csv(rollouts, outpath):
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["series", "start_idx", "n_steps", "mse_total"]
            + [f"mse_{n}" for n in BAND_NAMES]
        )
        for i, rollout in enumerate(rollouts):
            writer.writerow(
                [i, rollout["start_idx"], rollout["n_steps"], rollout["mse"]]
                + [rollout["band_mse"][b] for b in range(N_BANDS)]
            )


def save_step_csv(rollouts, outpath):
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "series",
                "start_idx",
                "step",
                "t_rel",
                "band_idx",
                "band_name",
                "pred_norm",
                "true_norm",
                "pred_mag",
                "true_mag",
                "residual",
            ]
            # Full all-band predicted magnitude at this step's lead time.
            + [f"pred_mag_{n}" for n in BAND_NAMES]
        )
        for i, rollout in enumerate(rollouts):
            for s in rollout["steps"]:
                writer.writerow(
                    [
                        i,
                        rollout["start_idx"],
                        s["step"],
                        s["t_rel"],
                        s["band"],
                        BAND_NAMES[s["band"]],
                        s["pred_norm"],
                        s["true_norm"],
                        s["pred_mag"],
                        s["true_mag"],
                        s["residual"],
                    ]
                    + [float(s["pred_all_mag"][b]) for b in range(N_BANDS)]
                )


def main():
    args = get_args()
    run_id = resolve_paths(args)

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    (
        model,
        context_len,
        n_bands,
        context_window_days,
        max_context_len,
    ) = load_9band_model(args.ckpt, device)

    window_mode = context_window_days is not None
    seed_len = max_context_len if window_mode else context_len

    eval_dataset, means, stds = make_eval_dataset(
        args=args,
        context_len=context_len,
        context_window_days=context_window_days,
        max_context_len=max_context_len if window_mode else None,
    )

    print("Dataset files with events:", len(eval_dataset.events_per_file))

    series = select_series(
        dataset=eval_dataset,
        context_len=context_len,
        n_future_steps=args.n_future_steps,
        n_series=args.n_series,
        seed_len=seed_len,
    )

    if not series:
        raise RuntimeError(
            "No eval series with enough events for a rollout. Check data path, "
            "N_imgs, and context_len."
        )

    print(f"Rolling out {len(series)} series, up to {args.n_future_steps} "
          f"steps each.")

    teacher_forced = args.teacher_forced

    rollouts = []
    tf_rollouts = [] if teacher_forced else None
    for i, (times, values, bands, start_idx) in enumerate(series):
        rollout = get_rollout_from_stream(
            times=times,
            values=values,
            bands=bands,
            model=model,
            device=device,
            start_idx=start_idx,
            n_future_steps=args.n_future_steps,
            context_len=context_len,
            n_bands=n_bands,
            means=means,
            stds=stds,
            teacher_forced=False,
            window_mode=window_mode,
            context_window_days=context_window_days,
            max_context_len=max_context_len,
        )
        rollouts.append(rollout)

        if teacher_forced:
            tf_rollout = get_rollout_from_stream(
                times=times,
                values=values,
                bands=bands,
                model=model,
                device=device,
                start_idx=start_idx,
                n_future_steps=args.n_future_steps,
                context_len=context_len,
                n_bands=n_bands,
                means=means,
                stds=stds,
                teacher_forced=True,
                window_mode=window_mode,
                context_window_days=context_window_days,
                max_context_len=max_context_len,
            )
            tf_rollouts.append(tf_rollout)

            print(
                f"  series {i + 1}/{len(series)}: {rollout['n_steps']} steps, "
                f"free-run mse={rollout['mse']:.4g}, "
                f"teacher-forced mse={tf_rollout['mse']:.4g}",
                flush=True,
            )
        else:
            print(
                f"  series {i + 1}/{len(series)}: {rollout['n_steps']} steps, "
                f"mse={rollout['mse']:.4g}",
                flush=True,
            )

    # Per-series forecast light curves.
    for i, rollout in enumerate(rollouts):
        series_path = os.path.join(
            args.outdir,
            f"study{run_id}_9band_autoreg_series{i:02d}.png",
        )
        tf_rollout = tf_rollouts[i] if teacher_forced else None
        plot_series_lightcurves(
            rollout, means, stds, series_path, tf_rollout=tf_rollout
        )

    residual_path = os.path.join(
        args.outdir, f"study{run_id}_9band_autoreg_residuals_vs_step.png"
    )
    mse_bar_path = os.path.join(
        args.outdir, f"study{run_id}_9band_autoreg_band_mse.png"
    )
    mse_csv_path = os.path.join(
        args.outdir, f"study{run_id}_9band_autoreg_mse_by_series.csv"
    )
    step_csv_path = os.path.join(
        args.outdir, f"study{run_id}_9band_autoreg_step_predictions.csv"
    )

    plot_residuals_vs_step(rollouts, residual_path)
    plot_band_mse_bar(rollouts, mse_bar_path, tf_rollouts=tf_rollouts)
    save_mse_csv(rollouts, mse_csv_path)
    save_step_csv(rollouts, step_csv_path)

    if teacher_forced:
        tf_mse_csv_path = os.path.join(
            args.outdir, f"study{run_id}_9band_autoreg_mse_by_series_tf.csv"
        )
        save_mse_csv(tf_rollouts, tf_mse_csv_path)

        overall_free = np.nanmean([r["mse"] for r in rollouts])
        overall_tf = np.nanmean([r["mse"] for r in tf_rollouts])
        print(
            f"Overall free-run MSE: {overall_free:.4g}  |  "
            f"teacher-forced MSE: {overall_tf:.4g}",
            flush=True,
        )

    print("Saved:")
    print("  per-series light curves in", args.outdir)
    print(" ", residual_path)
    print(" ", mse_bar_path)
    print(" ", mse_csv_path)
    print(" ", step_csv_path)


if __name__ == "__main__":
    main()
