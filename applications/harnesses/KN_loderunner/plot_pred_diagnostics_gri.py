import argparse
import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from yoke.models.vit.swin.bomberman import LodeRunner

from train_LodeRunner_ddp import (
    Kilonova_lc_scalar_context_DataSet_gri,
    ScalarTemporalConditionedLodeRunner_gri,
    load_or_compute_band_normalization,
)


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rc("font", family="serif")
plt.rcParams["figure.figsize"] = (7, 5)


BAND_KEYS = ("arr_ztfg", "arr_ztfr", "arr_ztfi")
BAND_NAMES = ("g", "r", "i")
VALUE_COL = 1


def study_tag(study):
    return f"{int(study):03d}"


def get_args():
    parser = argparse.ArgumentParser(
        description="Autoregressive prediction diagnostics for scalar temporal LodeRunner GRI runs."
    )

    parser.add_argument("--study", type=int, default=24)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument("--N_imgs", type=int, default=10)
    parser.add_argument("--n_future_steps", type=int, default=15)
    parser.add_argument("--n_series", type=int, default=10)

    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument(
        "--norm_stats_path",
        type=str,
        default="kilonova_gri_norm_stats.npz",
    )

    parser.add_argument(
        "--plot_all_series",
        action="store_true",
        help="Plot every rollout series. By default, plots all selected series too, but this flag is kept for compatibility.",
    )

    return parser.parse_args()


def resolve_paths(args):
    tag = study_tag(args.study)

    if args.ckpt is None:
        args.ckpt = (
            f"runs/study_{tag}/study{tag}_modelState_epoch{args.epoch:04d}.pth"
        )

    if args.outdir is None:
        args.outdir = f"runs/study_{tag}/autoreg_diagnostics_gri"

    return tag


def strip_ddp_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
        }
    return state_dict


def load_gri_model(ckpt_path, device):
    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    model_args = ckpt["model_args"]
    context_len = ckpt.get("context_len", 5)

    n_input_channels = ckpt.get("n_input_channels", 3)
    n_output_channels = ckpt.get("n_output_channels", 3)
    backbone_channels = ckpt.get("backbone_channels", 8)
    hidden = ckpt.get("hidden", 64)
    noise_scale = ckpt.get("noise_scale", 0.0)

    print("Loaded checkpoint:", ckpt_path)
    print("model_class:", ckpt.get("model_class", "unknown"))
    print("backbone_class:", ckpt.get("backbone_class", "LodeRunner"))
    print("predicts_delta:", ckpt.get("predicts_delta", False))
    print("target_type:", ckpt.get("target_type", "unknown"))
    print("context_len:", context_len)
    print("n_input_channels:", n_input_channels)
    print("n_output_channels:", n_output_channels)
    print("backbone_channels:", backbone_channels)
    print("hidden:", hidden)

    backbone = LodeRunner(**model_args).to(device)
    backbone.noise_scale = noise_scale

    model = ScalarTemporalConditionedLodeRunner_gri(
        backbone=backbone,
        context_len=context_len,
        n_input_channels=n_input_channels,
        n_output_channels=n_output_channels,
        image_size=model_args["image_size"],
        backbone_channels=backbone_channels,
        hidden=hidden,
    ).to(device)

    state_dict = strip_ddp_prefix(ckpt["model_state_dict"])

    missing, unexpected = model.load_state_dict(state_dict, strict=True)

    print("Loaded ScalarTemporalConditionedLodeRunner_gri checkpoint")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.eval()

    return model, context_len


def make_eval_dataset(args, context_len):
    band_means, band_stds = load_or_compute_band_normalization(
        stats_path=args.norm_stats_path,
        band_keys=BAND_KEYS,
        value_col=VALUE_COL,
    )

    print("Using band normalization:")
    print("band_means:", band_means)
    print("band_stds:", band_stds)

    dataset = Kilonova_lc_scalar_context_DataSet_gri(
        N_imgs=args.N_imgs,
        context_len=context_len,
        band_keys=BAND_KEYS,
        value_col=VALUE_COL,
        means=band_means,
        stds=band_stds,
        predicts_delta=True,
    )

    return dataset


def split_gri_context(x, context_len, n_bands=3):
    """
    Dataset x layout:
        [g0, r0, i0, g1, r1, i1, ..., gK, rK, iK, t0, t1, ..., tK]

    Returns:
        values: [context_len, 3]
        rel_t:  [context_len]
    """
    x = torch.as_tensor(x, dtype=torch.float32)

    value_count = context_len * n_bands
    values = x[:value_count].detach().cpu().numpy().reshape(context_len, n_bands)
    rel_t = x[value_count:].detach().cpu().numpy()

    return values.astype(np.float32), rel_t.astype(np.float32)


def build_gri_input(values, rel_t, device):
    """
    values:
        [context_len, 3]
    rel_t:
        [context_len]

    Returns:
        x: [1, context_len * 3 + context_len]
    """
    values = np.asarray(values, dtype=np.float32)
    rel_t = np.asarray(rel_t, dtype=np.float32)

    x = np.concatenate(
        [
            values.reshape(-1),
            rel_t,
        ],
        axis=0,
    )

    return torch.tensor(
        x,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)


def get_rollout_from_start_gri(
    dataset,
    model,
    device,
    start_idx,
    n_future_steps,
    context_len,
):
    x0, _, _ = dataset[start_idx]

    context_vals, _ = split_gri_context(
        x0,
        context_len=context_len,
        n_bands=3,
    )

    pred_vals = [row.copy() for row in context_vals]
    true_vals = [row.copy() for row in context_vals]

    pred_curve = []
    truth_curve = []
    residual_curve = []
    step_mses = []
    step_band_mses = []

    with torch.no_grad():
        for step in range(n_future_steps):
            future_idx = start_idx + step

            if future_idx >= len(dataset):
                break

            x_true_step, target_delta, future_Dt = dataset[future_idx]

            x_true_step = torch.as_tensor(x_true_step, dtype=torch.float32)
            target_delta = torch.as_tensor(
                target_delta,
                dtype=torch.float32,
                device=device,
            )
            future_Dt = torch.as_tensor(
                future_Dt,
                dtype=torch.float32,
                device=device,
            )

            if target_delta.ndim == 0:
                raise ValueError(
                    "Expected GRI target_delta shape [3], but got scalar target."
                )

            target_delta = target_delta.reshape(3)

            if future_Dt.ndim == 0:
                future_Dt = future_Dt.unsqueeze(0)

            _, current_rel_t = split_gri_context(
                x_true_step,
                context_len=context_len,
                n_bands=3,
            )

            current_pred_vals = np.asarray(
                pred_vals[-context_len:],
                dtype=np.float32,
            )

            x_pred = build_gri_input(
                values=current_pred_vals,
                rel_t=current_rel_t,
                device=device,
            )

            pred_delta = model(
                x_pred,
                in_vars=None,
                out_vars=None,
                Dt=future_Dt,
            )

            pred_delta = pred_delta.reshape(3)

            pred_next = (
                torch.as_tensor(pred_vals[-1], dtype=torch.float32, device=device)
                + pred_delta
            )
            true_next = (
                torch.as_tensor(true_vals[-1], dtype=torch.float32, device=device)
                + target_delta
            )

            residual = pred_next - true_next
            band_mse = residual.pow(2)
            total_mse = band_mse.mean()

            pred_next_np = pred_next.detach().cpu().numpy()
            true_next_np = true_next.detach().cpu().numpy()
            residual_np = residual.detach().cpu().numpy()
            band_mse_np = band_mse.detach().cpu().numpy()

            pred_curve.append(pred_next_np)
            truth_curve.append(true_next_np)
            residual_curve.append(residual_np)
            step_band_mses.append(band_mse_np)
            step_mses.append(float(total_mse.detach().cpu()))

            pred_vals.append(pred_next_np)
            true_vals.append(true_next_np)

    pred_curve = np.asarray(pred_curve, dtype=np.float32)
    truth_curve = np.asarray(truth_curve, dtype=np.float32)
    residual_curve = np.asarray(residual_curve, dtype=np.float32)
    step_mses = np.asarray(step_mses, dtype=np.float32)
    step_band_mses = np.asarray(step_band_mses, dtype=np.float32)

    total_mse = np.mean(step_mses) if len(step_mses) > 0 else np.nan

    if len(step_band_mses) > 0:
        band_mse = np.mean(step_band_mses, axis=0)
    else:
        band_mse = np.full(3, np.nan, dtype=np.float32)

    return {
        "start_idx": start_idx,
        "context": np.asarray(context_vals, dtype=np.float32),
        "pred": pred_curve,
        "truth": truth_curve,
        "residual": residual_curve,
        "step_mses": step_mses,
        "step_band_mses": step_band_mses,
        "mse": total_mse,
        "band_mse": band_mse,
    }


def plot_residuals_vs_step(rollouts, outpath):
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    for band_idx, band_name in enumerate(BAND_NAMES):
        ax = axes[band_idx]

        for rollout in rollouts:
            residual = rollout["residual"]
            steps = np.arange(len(residual))

            ax.plot(
                steps,
                residual[:, band_idx],
                marker="o",
                alpha=0.70,
                label=f"start {rollout['start_idx']}",
            )

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_ylabel(f"{band_name} residual")
        ax.set_title(f"{band_name}-band residuals")

    axes[-1].set_xlabel("Autoregressive step")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=7,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle("Autoregressive residuals vs time step", y=1.06)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_multiple_series_predictions(rollouts, outpath):
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    for band_idx, band_name in enumerate(BAND_NAMES):
        ax = axes[band_idx]

        for rollout in rollouts:
            start_idx = rollout["start_idx"]

            context_steps = np.arange(-len(rollout["context"]), 0)
            future_steps = np.arange(len(rollout["pred"]))

            ax.plot(
                context_steps,
                rollout["context"][:, band_idx],
                linestyle=":",
                alpha=0.45,
            )

            ax.plot(
                future_steps,
                rollout["truth"][:, band_idx],
                linewidth=1.5,
                alpha=0.75,
                label=f"truth start {start_idx}",
            )

            ax.plot(
                future_steps,
                rollout["pred"][:, band_idx],
                linestyle="--",
                linewidth=1.5,
                alpha=0.75,
                label=f"pred start {start_idx}",
            )

        ax.axvline(-0.5, linewidth=1, alpha=0.5)
        ax.invert_yaxis()
        ax.set_ylabel(f"{band_name} norm mag")
        ax.set_title(f"{band_name}-band autoregressive predictions")

    axes[-1].set_xlabel("Time step relative to forecast start")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=6,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle("Autoregressive predictions for validation curves", y=1.06)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mse_histogram(rollouts, outpath):
    mses = np.asarray([r["mse"] for r in rollouts], dtype=np.float32)
    mses = mses[np.isfinite(mses)]

    plt.figure(figsize=(7, 5))
    plt.hist(mses, bins=min(10, max(1, len(mses))))
    plt.xlabel("Mean autoregressive MSE per validation curve")
    plt.ylabel("Count")
    plt.title("Distribution of autoregressive rollout MSEs")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_band_mse_histograms(rollouts, outpath):
    band_mses = np.asarray([r["band_mse"] for r in rollouts], dtype=np.float32)

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=False)

    for band_idx, band_name in enumerate(BAND_NAMES):
        vals = band_mses[:, band_idx]
        vals = vals[np.isfinite(vals)]

        axes[band_idx].hist(vals, bins=min(10, max(1, len(vals))))
        axes[band_idx].set_xlabel(f"{band_name} mean autoregressive MSE")
        axes[band_idx].set_ylabel("Count")
        axes[band_idx].set_title(f"{band_name}-band rollout MSE distribution")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_mse_csv(rollouts, outpath):
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "start_idx",
                "mse_total",
                "mse_g",
                "mse_r",
                "mse_i",
                "n_steps",
            ]
        )

        for rollout in rollouts:
            writer.writerow(
                [
                    rollout["start_idx"],
                    rollout["mse"],
                    rollout["band_mse"][0],
                    rollout["band_mse"][1],
                    rollout["band_mse"][2],
                    len(rollout["residual"]),
                ]
            )


def save_step_csv(rollouts, outpath):
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "start_idx",
                "step",
                "pred_g",
                "pred_r",
                "pred_i",
                "truth_g",
                "truth_r",
                "truth_i",
                "residual_g",
                "residual_r",
                "residual_i",
                "mse_total",
                "mse_g",
                "mse_r",
                "mse_i",
            ]
        )

        for rollout in rollouts:
            for step in range(len(rollout["pred"])):
                writer.writerow(
                    [
                        rollout["start_idx"],
                        step,
                        rollout["pred"][step, 0],
                        rollout["pred"][step, 1],
                        rollout["pred"][step, 2],
                        rollout["truth"][step, 0],
                        rollout["truth"][step, 1],
                        rollout["truth"][step, 2],
                        rollout["residual"][step, 0],
                        rollout["residual"][step, 1],
                        rollout["residual"][step, 2],
                        rollout["step_mses"][step],
                        rollout["step_band_mses"][step, 0],
                        rollout["step_band_mses"][step, 1],
                        rollout["step_band_mses"][step, 2],
                    ]
                )


def main():
    args = get_args()
    run_id = resolve_paths(args)

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model, context_len = load_gri_model(args.ckpt, device)

    eval_dataset = make_eval_dataset(
        args=args,
        context_len=context_len,
    )

    max_start = max(0, len(eval_dataset) - args.n_future_steps)
    n_series = min(args.n_series, max_start + 1)

    print("Dataset length:", len(eval_dataset))
    print("Number of rollout series:", n_series)
    print("Autoregressive future steps:", args.n_future_steps)

    if n_series <= 0:
        raise RuntimeError("No rollout series available. Check dataset size and n_future_steps.")

    start_indices = np.linspace(0, max_start, n_series, dtype=int)

    rollouts = []

    for start_idx in start_indices:
        print(f"Rolling out validation curve starting at index {start_idx}")

        rollout = get_rollout_from_start_gri(
            dataset=eval_dataset,
            model=model,
            device=device,
            start_idx=int(start_idx),
            n_future_steps=args.n_future_steps,
            context_len=context_len,
        )

        rollouts.append(rollout)

    residual_path = os.path.join(
        args.outdir,
        f"study{run_id}_autoreg_gri_residuals_vs_step.png",
    )
    series_path = os.path.join(
        args.outdir,
        f"study{run_id}_autoreg_gri_multi_series_predictions.png",
    )
    hist_path = os.path.join(
        args.outdir,
        f"study{run_id}_autoreg_gri_mse_histogram.png",
    )
    band_hist_path = os.path.join(
        args.outdir,
        f"study{run_id}_autoreg_gri_band_mse_histograms.png",
    )
    csv_path = os.path.join(
        args.outdir,
        f"study{run_id}_autoreg_gri_mse_by_curve.csv",
    )
    step_csv_path = os.path.join(
        args.outdir,
        f"study{run_id}_autoreg_gri_step_predictions.csv",
    )

    plot_residuals_vs_step(rollouts, residual_path)
    plot_multiple_series_predictions(rollouts, series_path)
    plot_mse_histogram(rollouts, hist_path)
    plot_band_mse_histograms(rollouts, band_hist_path)
    save_mse_csv(rollouts, csv_path)
    save_step_csv(rollouts, step_csv_path)

    print("Saved:")
    print(" ", residual_path)
    print(" ", series_path)
    print(" ", hist_path)
    print(" ", band_hist_path)
    print(" ", csv_path)
    print(" ", step_csv_path)


if __name__ == "__main__":
    main()
