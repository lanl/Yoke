"""Next-event prediction diagnostics for the 9-band scalar temporal LodeRunner.

The 9-band model is trained on a merged event stream: each sample is a window of
``context_len`` consecutive observations (across all bands) plus a lead time, and
the model predicts the value in every band at that lead time. Each training
target only observes one band, so diagnostics here compare the model's prediction
for the observed target band against the truth, aggregated per band.

Unlike the g/r/i diagnostics this script does NOT roll out autoregressively.
Autoregression is ill-defined for a mixed-band event stream (each future event
belongs to a single band), so we evaluate the direct one-step-ahead prediction
the model is actually trained on.
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
            "Next-event prediction diagnostics for scalar temporal LodeRunner "
            "9-band runs."
        )
    )

    parser.add_argument("--study", type=int, default=24)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument("--N_imgs", type=int, default=50)

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
        args.outdir = f"runs/study_{tag}/next_event_diagnostics_9band"

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

    print("Loaded checkpoint:", ckpt_path)
    print("model_class:", ckpt.get("model_class", "unknown"))
    print("backbone_class:", ckpt.get("backbone_class", "LodeRunner"))
    print("target_type:", ckpt.get("target_type", "unknown"))
    print("context_len:", context_len)
    print("n_bands:", n_bands)
    print("band_keys:", ckpt.get("band_keys", list(BAND_KEYS)))
    print("backbone_channels:", backbone_channels)
    print("hidden:", hidden)

    backbone = LodeRunner(**model_args).to(device)
    backbone.noise_scale = noise_scale

    model = ScalarTemporalConditionedLodeRunner_9band(
        backbone=backbone,
        context_len=context_len,
        n_bands=n_bands,
        image_size=model_args["image_size"],
        backbone_channels=backbone_channels,
        hidden=hidden,
    ).to(device)

    state_dict = strip_ddp_prefix(ckpt["model_state_dict"])

    missing, unexpected = model.load_state_dict(state_dict, strict=True)

    print("Loaded ScalarTemporalConditionedLodeRunner_9band checkpoint")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.eval()

    return model, context_len, n_bands


def make_eval_dataset(args, context_len):
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

    dataset = Kilonova_lc_scalar_context_DataSet_9band(
        N_imgs=args.N_imgs,
        context_len=context_len,
        band_keys=BAND_KEYS,
        value_col=VALUE_COL,
        error_col=ERROR_COL,
        drop_upper_limits=DROP_UPPER_LIMITS,
        means=band_means,
        stds=band_stds,
    )

    return dataset


def collect_next_event_predictions(dataset, model, device, n_bands):
    """Run the direct next-event prediction over every sample.

    Returns a dict of per-band arrays of (pred, truth) for the observed band of
    each sample, plus the residuals.
    """
    preds_by_band = [[] for _ in range(n_bands)]
    truths_by_band = [[] for _ in range(n_bands)]

    with torch.no_grad():
        for idx in range(len(dataset)):
            x, target, mask, Dt = dataset[idx]

            x = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
            target = torch.as_tensor(target, dtype=torch.float32)
            mask = torch.as_tensor(mask, dtype=torch.float32)
            Dt = torch.as_tensor(Dt, dtype=torch.float32, device=device)

            if Dt.ndim == 0:
                Dt = Dt.unsqueeze(0)

            pred = model(x, in_vars=None, out_vars=None, Dt=Dt)
            pred = pred.reshape(n_bands).detach().cpu()

            band_idx = int(torch.argmax(mask).item())

            preds_by_band[band_idx].append(float(pred[band_idx]))
            truths_by_band[band_idx].append(float(target[band_idx]))

    results = []
    for band_idx in range(n_bands):
        p = np.asarray(preds_by_band[band_idx], dtype=np.float32)
        t = np.asarray(truths_by_band[band_idx], dtype=np.float32)
        r = p - t

        if len(r) > 0:
            mse = float(np.mean(r**2))
        else:
            mse = np.nan

        results.append(
            {
                "band_idx": band_idx,
                "band_name": BAND_NAMES[band_idx],
                "pred": p,
                "truth": t,
                "residual": r,
                "n": len(r),
                "mse": mse,
            }
        )

    return results


def plot_pred_vs_truth(results, outpath):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for band_idx, res in enumerate(results):
        ax = axes[band_idx]

        if res["n"] == 0:
            ax.set_title(f"{res['band_name']} (no samples)")
            ax.axis("off")
            continue

        ax.scatter(res["truth"], res["pred"], s=8, alpha=0.5)

        lo = float(min(res["truth"].min(), res["pred"].min()))
        hi = float(max(res["truth"].max(), res["pred"].max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="k")

        ax.set_xlabel("truth (norm)")
        ax.set_ylabel("pred (norm)")
        ax.set_title(f"{res['band_name']}  (n={res['n']}, MSE={res['mse']:.3g})")

    fig.suptitle("Next-event prediction vs truth (normalized), per band", y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_residual_histograms(results, outpath):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for band_idx, res in enumerate(results):
        ax = axes[band_idx]

        if res["n"] == 0:
            ax.set_title(f"{res['band_name']} (no samples)")
            ax.axis("off")
            continue

        ax.hist(res["residual"], bins=min(30, max(1, res["n"])))
        ax.axvline(0.0, linestyle="--", linewidth=1, color="k")
        ax.set_xlabel("pred - truth (norm)")
        ax.set_ylabel("count")
        ax.set_title(f"{res['band_name']}  (n={res['n']})")

    fig.suptitle("Next-event residual distributions, per band", y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_band_mse_bar(results, outpath):
    names = [res["band_name"] for res in results]
    mses = [res["mse"] if np.isfinite(res["mse"]) else 0.0 for res in results]

    plt.figure(figsize=(9, 5))
    plt.bar(names, mses)
    plt.ylabel("Next-event MSE (normalized)")
    plt.xlabel("Band")
    plt.title("Per-band next-event MSE")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_band_mse_csv(results, outpath):
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["band_idx", "band_name", "n_samples", "mse"])
        for res in results:
            writer.writerow(
                [res["band_idx"], res["band_name"], res["n"], res["mse"]]
            )


def main():
    args = get_args()
    run_id = resolve_paths(args)

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model, context_len, n_bands = load_9band_model(args.ckpt, device)

    eval_dataset = make_eval_dataset(args=args, context_len=context_len)

    print("Dataset length:", len(eval_dataset))

    if len(eval_dataset) == 0:
        raise RuntimeError("Empty eval dataset. Check data path and N_imgs.")

    results = collect_next_event_predictions(
        dataset=eval_dataset,
        model=model,
        device=device,
        n_bands=n_bands,
    )

    pred_truth_path = os.path.join(
        args.outdir, f"study{run_id}_9band_next_event_pred_vs_truth.png"
    )
    resid_path = os.path.join(
        args.outdir, f"study{run_id}_9band_next_event_residual_hist.png"
    )
    mse_bar_path = os.path.join(
        args.outdir, f"study{run_id}_9band_next_event_band_mse.png"
    )
    csv_path = os.path.join(
        args.outdir, f"study{run_id}_9band_next_event_band_mse.csv"
    )

    plot_pred_vs_truth(results, pred_truth_path)
    plot_residual_histograms(results, resid_path)
    plot_band_mse_bar(results, mse_bar_path)
    save_band_mse_csv(results, csv_path)

    print("Saved:")
    print(" ", pred_truth_path)
    print(" ", resid_path)
    print(" ", mse_bar_path)
    print(" ", csv_path)


if __name__ == "__main__":
    main()
