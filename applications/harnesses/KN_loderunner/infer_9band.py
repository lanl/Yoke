"""Forecast all 9 bands into the future with the scalar temporal LodeRunner.

This is the intended production use of the 9-band model: given a light curve of
sparse, irregular, multi-band observations (real or simulated), take the most
recent ``context_len`` observations and predict the value in every band at a
grid of future lead times, measured from the last observation.

For each requested lead time Dt, the model consumes the same fixed context
window and emits a prediction for all 9 bands at that lead time. This directly
answers "what will each observatory see next?" without assuming any band is
observed at that future time.

Predictions are denormalized back to magnitudes using the per-band normalization
statistics. Results are plotted over the observed data and written to npz/csv.
"""

import argparse
import csv
import glob
import json
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
# non-finite uncertainty in ERROR_COL, from the context fed to the model.
DROP_UPPER_LIMITS = True


def study_tag(study):
    return f"{int(study):03d}"


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Forecast all 9 bands into the future from a light curve's most "
            "recent observations."
        )
    )

    parser.add_argument("--study", type=int, default=24)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument(
        "--data_glob",
        type=str,
        default=None,
        help="Glob for light-curve npz files to forecast. If omitted, uses the "
        "training data glob path.",
    )
    parser.add_argument(
        "--n_curves",
        type=int,
        default=5,
        help="Number of light-curve files to forecast.",
    )

    parser.add_argument(
        "--horizon",
        type=float,
        default=5.0,
        help="Forecast horizon in the same time units as the data (days), "
        "measured from the last observation.",
    )
    parser.add_argument(
        "--n_lead_times",
        type=int,
        default=25,
        help="Number of lead times sampled between 0 and --horizon.",
    )

    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument(
        "--norm_stats_path",
        type=str,
        default="kilonova_9band_norm_stats.npz",
    )

    return parser.parse_args()


def resolve_paths(args):
    tag = study_tag(args.study)

    if args.ckpt is None:
        args.ckpt = (
            f"runs/study_{tag}/study{tag}_modelState_epoch{args.epoch:04d}.pth"
        )

    if args.outdir is None:
        args.outdir = f"runs/study_{tag}/forecast_9band"

    if args.data_glob is None:
        args.data_glob = (
            "/net/sescratch1/atoivonen/data/KN_lightcurves/"
            "uniform_dataset_20000/lc_*.npz"
        )

    return tag


def strip_ddp_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
        }
    return state_dict


def load_9band_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

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
    print("target_type:", ckpt.get("target_type", "unknown"))
    print("context_len:", context_len)
    print("context_window_days:", context_window_days)
    print("max_context_len:", max_context_len)
    print("n_bands:", n_bands)

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

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.eval()

    return model, context_len, n_bands, context_window_days, max_context_len


def load_event_stream(fn, means, stds):
    """Load one npz file into a merged, time-sorted, normalized event stream.

    Mirrors the dataset's stream construction so inference matches training.

    Returns:
        times (np.ndarray): Relative observation times [N].
        values_norm (np.ndarray): Normalized values [N].
        bands (np.ndarray): Band index per event [N].
        raw (dict): Per-band raw (mjd, mag) arrays for plotting the observations.
        t0 (float): The earliest MJD, used to align forecast times.
    """
    data = np.load(fn, allow_pickle=True)

    times = []
    values = []
    bands = []
    raw = {}

    for band_idx, key in enumerate(BAND_KEYS):
        if key not in data.files:
            continue

        arr = data[key]
        if arr.size == 0:
            continue

        # Drop upper-limit (non-detection) rows, flagged by a non-finite
        # uncertainty in ERROR_COL, matching how the model was trained.
        if DROP_UPPER_LIMITS:
            detected = np.isfinite(arr[:, ERROR_COL])
            arr = arr[detected]
            if arr.shape[0] == 0:
                continue

        t = arr[:, 0].astype(np.float32)
        v = arr[:, VALUE_COL].astype(np.float32)

        times.append(t)
        values.append(v)
        bands.append(np.full(arr.shape[0], band_idx, dtype=np.int64))
        raw[band_idx] = (t, v)

    data.close()

    if not times:
        return None

    times = np.concatenate(times)
    values = np.concatenate(values)
    bands = np.concatenate(bands)

    order = np.argsort(times, kind="stable")
    times = times[order]
    values = values[order]
    bands = bands[order]

    t0 = float(times.min())
    times = times - t0

    values_norm = (values - means[bands]) / (stds[bands] + EPS)

    return times, values_norm.astype(np.float32), bands, raw, t0


def build_context_input(
    ctx_t, ctx_v, ctx_b, n_bands, device, window_mode=False, max_context_len=None
):
    """Build the flattened per-event context input for the model.

    Fixed-count mode (``window_mode=False``): layout per event is
    ``[value, rel_t, one_hot_band(n_bands)]`` (width ``2 + n_bands``).

    Time-window mode (``window_mode=True``): the real events are padded to
    ``max_context_len`` rows and each event gains a validity flag, giving
    ``[value, rel_t, valid, one_hot_band(n_bands)]`` (width ``3 + n_bands``).
    Real events fill the leading rows in time order with ``rel_t`` relative to
    the first real event; padded rows are all-zero with ``valid = 0``. Matches
    ``_getitem_window`` in the dataset.
    """
    ctx_t = np.asarray(ctx_t, dtype=np.float32)
    ctx_v = np.asarray(ctx_v, dtype=np.float32)
    ctx_b = np.asarray(ctx_b, dtype=np.int64)

    rel_t = (ctx_t - ctx_t[0]).astype(np.float32)

    if window_mode:
        n_real = ctx_t.shape[0]
        per_event = np.zeros((max_context_len, 3 + n_bands), dtype=np.float32)
        per_event[:n_real, 0] = ctx_v
        per_event[:n_real, 1] = rel_t
        per_event[:n_real, 2] = 1.0  # validity flag for real events
        per_event[np.arange(n_real), 3 + ctx_b] = 1.0
    else:
        context_len = ctx_t.shape[0]
        band_onehot = np.zeros((context_len, n_bands), dtype=np.float32)
        band_onehot[np.arange(context_len), ctx_b] = 1.0

        per_event = np.concatenate(
            [ctx_v[:, None], rel_t[:, None], band_onehot],
            axis=1,
        )

    x = torch.tensor(
        per_event.reshape(-1),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    return x


def forecast_curve(
    stream,
    model,
    device,
    context_len,
    n_bands,
    means,
    stds,
    lead_times,
    window_mode=False,
    context_window_days=None,
    max_context_len=None,
):
    """Forecast all bands at a grid of future lead times from the last context.

    Returns a [n_lead_times, n_bands] array of denormalized (magnitude)
    predictions and the absolute forecast times (in the last-observation frame).
    """
    times, values_norm, bands, _, _ = stream

    if window_mode:
        # Time-window context: all events within context_window_days of the last
        # observation, capped at max_context_len (matches _getitem_window).
        anchor_t = times[-1]
        lo = anchor_t - context_window_days
        sel_idx = np.nonzero(times >= lo)[0]
        if sel_idx.shape[0] > max_context_len:
            sel_idx = sel_idx[-max_context_len:]
        ctx_t = times[sel_idx]
        ctx_v = values_norm[sel_idx]
        ctx_b = bands[sel_idx]
    else:
        # Most recent context_len events.
        ctx_t = times[-context_len:]
        ctx_v = values_norm[-context_len:]
        ctx_b = bands[-context_len:]

    x = build_context_input(
        ctx_t,
        ctx_v,
        ctx_b,
        n_bands,
        device,
        window_mode=window_mode,
        max_context_len=max_context_len,
    )

    last_t = float(times[-1])

    preds_norm = np.zeros((len(lead_times), n_bands), dtype=np.float32)

    with torch.no_grad():
        for k, dt in enumerate(lead_times):
            Dt = torch.tensor([dt], dtype=torch.float32, device=device)
            pred = model(x, in_vars=None, out_vars=None, Dt=Dt)
            preds_norm[k] = pred.reshape(n_bands).detach().cpu().numpy()

    # Denormalize per band back to magnitudes.
    preds_mag = preds_norm * (stds[None, :] + EPS) + means[None, :]

    forecast_times = last_t + np.asarray(lead_times, dtype=np.float32)

    return preds_mag, forecast_times, last_t


def plot_forecast(stream, preds_mag, forecast_times, last_t, title, outpath):
    _, _, _, raw, t0 = stream

    plt.figure(figsize=(9, 6))

    for band_idx in range(N_BANDS):
        color = BAND_COLORS[band_idx]
        name = BAND_NAMES[band_idx]

        # Observed points for this band, aligned to the same relative-time
        # frame as the stream and forecast (raw MJDs shifted by global t0).
        if band_idx in raw:
            t_obs, v_obs = raw[band_idx]
            plt.scatter(
                t_obs - t0,
                v_obs,
                s=18,
                color=color,
                alpha=0.6,
                label=f"{name} obs",
            )

        # Forecast curve for this band.
        plt.plot(
            forecast_times,
            preds_mag[:, band_idx],
            linestyle="--",
            color=color,
            linewidth=1.6,
            label=f"{name} forecast",
        )

    plt.axvline(
        last_t,
        color="k",
        linewidth=1,
        linestyle=":",
        alpha=0.7,
        label="last observation",
    )

    plt.gca().invert_yaxis()
    plt.xlabel("Relative time (days)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.legend(fontsize=6, ncol=3, loc="best")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def save_forecast_csv(preds_mag, forecast_times, outpath):
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["forecast_time"] + [f"mag_{n}" for n in BAND_NAMES])
        for k in range(len(forecast_times)):
            writer.writerow(
                [forecast_times[k]] + [preds_mag[k, b] for b in range(N_BANDS)]
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

    means, stds = load_or_compute_band_normalization(
        stats_path=args.norm_stats_path,
        band_keys=BAND_KEYS,
        value_col=VALUE_COL,
        error_col=ERROR_COL,
        drop_upper_limits=DROP_UPPER_LIMITS,
    )
    means = np.asarray(means, dtype=np.float32)
    stds = np.asarray(stds, dtype=np.float32)

    files = sorted(glob.glob(args.data_glob))
    if not files:
        raise RuntimeError(f"No files matched data_glob: {args.data_glob}")

    files = files[: args.n_curves]
    print(f"Forecasting {len(files)} light curves.")

    lead_times = np.linspace(0.0, args.horizon, args.n_lead_times)

    for i, fn in enumerate(files):
        stream = load_event_stream(fn, means, stds)

        if stream is None:
            print(f"Skipping {fn}: no observations.")
            continue

        times = stream[0]
        # In window mode any curve with at least one detection can be forecast
        # (the trailing window is padded); otherwise we need a full context.
        min_events = 1 if window_mode else context_len
        if len(times) < min_events:
            print(
                f"Skipping {fn}: only {len(times)} events, "
                f"need at least {min_events}."
            )
            continue

        preds_mag, forecast_times, last_t = forecast_curve(
            stream=stream,
            model=model,
            device=device,
            context_len=context_len,
            n_bands=n_bands,
            means=means,
            stds=stds,
            lead_times=lead_times,
            window_mode=window_mode,
            context_window_days=context_window_days,
            max_context_len=max_context_len,
        )

        base = os.path.splitext(os.path.basename(fn))[0]

        png_path = os.path.join(
            args.outdir, f"study{run_id}_forecast_{base}.png"
        )
        csv_path = os.path.join(
            args.outdir, f"study{run_id}_forecast_{base}.csv"
        )
        npz_path = os.path.join(
            args.outdir, f"study{run_id}_forecast_{base}.npz"
        )

        plot_forecast(
            stream=stream,
            preds_mag=preds_mag,
            forecast_times=forecast_times,
            last_t=last_t,
            title=f"9-band forecast: {base}",
            outpath=png_path,
        )
        save_forecast_csv(preds_mag, forecast_times, csv_path)
        np.savez(
            npz_path,
            forecast_times=forecast_times,
            preds_mag=preds_mag,
            band_names=np.array(BAND_NAMES),
            last_observation_time=last_t,
        )

        print(f"[{i + 1}/{len(files)}] saved forecast for {base}")

    print("Done. Output directory:", args.outdir)


if __name__ == "__main__":
    main()
