import argparse
import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from yoke.models.vit.swin.bomberman import LodeRunner
from train_LodeRunner_ddp import Kilonova_lc_img_DataSet_channels_context

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rc("font", family="serif")
plt.rcParams["figure.figsize"] = (7, 5)

RUN_ID = "021"


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default=f"runs/study_{RUN_ID}/study{RUN_ID}_modelState_epoch0300.pth",
    )
    parser.add_argument("--N_imgs", type=int, default=10)
    parser.add_argument("--n_future_steps", type=int, default=15)
    parser.add_argument("--n_series", type=int, default=10)
    parser.add_argument(
        "--outdir",
        type=str,
        default=f"runs/study_{RUN_ID}/autoreg_diagnostics",
    )

    return parser.parse_args()


def load_channel_model(ckpt_path, device):
    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    model_args = ckpt["model_args"]
    noise_scale = ckpt.get("noise_scale", 0.0)
    context_len = ckpt.get("context_len", 5)

    print("Loaded checkpoint:", ckpt_path)
    print("predicts_delta:", ckpt.get("predicts_delta", False))
    print("target_type:", ckpt.get("target_type", "absolute"))
    print("context_len:", context_len)

    model = LodeRunner(**model_args).to(device)

    state_dict = ckpt["model_state_dict"]
    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=True)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.noise_scale = noise_scale
    model.eval()

    return model, context_len


def ensure_batch(x):
    """
    Dataset item usually has shape:
        [T, H, W]

    Model expects:
        [B, T, H, W]
    """
    if x.ndim == 3:
        return x.unsqueeze(0)

    return x


def tensor_time_means(x):
    """
    Convert image sequence tensor to scalar light curve.

    Supports:
        [T, H, W]
        [B, T, H, W]
        [B, T, C, H, W]
    """
    if x.ndim == 3:
        return x.mean(dim=(1, 2)).detach().cpu().numpy().squeeze()

    if x.ndim == 4:
        return x.mean(dim=(2, 3)).detach().cpu().numpy().squeeze()

    if x.ndim == 5:
        return x.mean(dim=(2, 3, 4)).detach().cpu().numpy().squeeze()

    raise ValueError(f"Unexpected tensor shape: {x.shape}")


def get_rollout_from_start(
    dataset,
    model,
    device,
    start_idx,
    n_future_steps,
    in_vars,
    out_vars,
):
    """
    Clean autoregressive rollout.

    x_pred:
        model-generated autoregressive context

    x_true:
        ground-truth context used only to reconstruct true future values

    This avoids the bug where truth was reconstructed using the predicted
    previous frame.
    """

    context_img, _, _ = dataset[start_idx]

    x_pred = ensure_batch(context_img).to(device)
    x_true = ensure_batch(context_img).to(device)

    context_curve = tensor_time_means(x_true)

    pred_curve = []
    truth_curve = []
    residual_curve = []
    step_mses = []

    with torch.no_grad():
        for step in range(n_future_steps):
            future_idx = start_idx + step

            if future_idx >= len(dataset):
                break

            _, future_target_delta, future_Dt = dataset[future_idx]

            future_target_delta = ensure_batch(future_target_delta).to(device)

            future_Dt = torch.as_tensor(
                future_Dt,
                dtype=torch.float32,
                device=device,
            )

            if future_Dt.ndim == 0:
                future_Dt = future_Dt.unsqueeze(0)

            pred_delta_img = model(x_pred, in_vars, out_vars, future_Dt)

            pred_last_img = x_pred[:, -1:]
            true_last_img = x_true[:, -1:]

            pred_next_img = pred_last_img + pred_delta_img[:, -1:]
            true_next_img = true_last_img + future_target_delta[:, -1:]

            pred_scalar = pred_next_img.mean().item()
            true_scalar = true_next_img.mean().item()
            residual_scalar = pred_scalar - true_scalar

            step_mse = torch.mean((pred_next_img - true_next_img) ** 2).item()

            pred_curve.append(pred_scalar)
            truth_curve.append(true_scalar)
            residual_curve.append(residual_scalar)
            step_mses.append(step_mse)

            # Autoregressive model context gets the prediction.
            x_pred = torch.cat(
                [x_pred[:, 1:], pred_next_img.detach()],
                dim=1,
            )

            # Truth context gets the independently reconstructed truth.
            x_true = torch.cat(
                [x_true[:, 1:], true_next_img.detach()],
                dim=1,
            )

    pred_curve = np.asarray(pred_curve)
    truth_curve = np.asarray(truth_curve)
    residual_curve = np.asarray(residual_curve)
    step_mses = np.asarray(step_mses)

    total_mse = np.mean(step_mses) if len(step_mses) > 0 else np.nan

    return {
        "start_idx": start_idx,
        "context": context_curve,
        "pred": pred_curve,
        "truth": truth_curve,
        "residual": residual_curve,
        "step_mses": step_mses,
        "mse": total_mse,
    }


def plot_residuals_vs_step(rollouts, outpath):
    plt.figure(figsize=(8, 5))

    for rollout in rollouts:
        steps = np.arange(len(rollout["residual"]))
        plt.plot(
            steps,
            rollout["residual"],
            marker="o",
            alpha=0.75,
            label=f"start {rollout['start_idx']}",
        )

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Autoregressive step")
    plt.ylabel("Residual: prediction - truth")
    plt.title("Autoregressive residuals vs time step")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_multiple_series_predictions(rollouts, outpath):
    plt.figure(figsize=(9, 6))

    for rollout in rollouts:
        start_idx = rollout["start_idx"]

        context_steps = np.arange(-len(rollout["context"]), 0)
        future_steps = np.arange(len(rollout["pred"]))

        plt.plot(
            context_steps,
            rollout["context"],
            linestyle=":",
            alpha=0.45,
        )

        plt.plot(
            future_steps,
            rollout["truth"],
            linewidth=1.5,
            alpha=0.75,
            label=f"truth start {start_idx}",
        )

        plt.plot(
            future_steps,
            rollout["pred"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.75,
            label=f"pred start {start_idx}",
        )

    plt.axvline(-0.5, linewidth=1, alpha=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel("Time step relative to forecast start")
    plt.ylabel("Normalized magnitude")
    plt.title("Autoregressive predictions for validation curves")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_mse_histogram(rollouts, outpath):
    mses = np.asarray([r["mse"] for r in rollouts])
    mses = mses[np.isfinite(mses)]

    plt.figure(figsize=(7, 5))
    plt.hist(mses, bins=min(10, max(1, len(mses))))
    plt.xlabel("Mean autoregressive MSE per validation curve")
    plt.ylabel("Count")
    plt.title("Distribution of autoregressive rollout MSEs")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_mse_csv(rollouts, outpath):
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_idx", "mse", "n_steps"])

        for rollout in rollouts:
            writer.writerow(
                [
                    rollout["start_idx"],
                    rollout["mse"],
                    len(rollout["residual"]),
                ]
            )


def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model, context_len = load_channel_model(args.ckpt, device)

    eval_dataset = Kilonova_lc_img_DataSet_channels_context(
        half_image=False,
        N_imgs=args.N_imgs,
        context_len=context_len,
    )

    in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)
    out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)

    max_start = max(0, len(eval_dataset) - args.n_future_steps)
    n_series = min(args.n_series, max_start + 1)

    print("Dataset length:", len(eval_dataset))
    print("Number of rollout series:", n_series)
    print("Autoregressive future steps:", args.n_future_steps)

    start_indices = np.linspace(0, max_start, n_series, dtype=int)

    rollouts = []

    for start_idx in start_indices:
        print(f"Rolling out validation curve starting at index {start_idx}")

        rollout = get_rollout_from_start(
            dataset=eval_dataset,
            model=model,
            device=device,
            start_idx=int(start_idx),
            n_future_steps=args.n_future_steps,
            in_vars=in_vars,
            out_vars=out_vars,
        )

        rollouts.append(rollout)

    residual_path = os.path.join(
        args.outdir,
        f"study{RUN_ID}_autoreg_residuals_vs_step.png",
    )
    series_path = os.path.join(
        args.outdir,
        f"study{RUN_ID}_autoreg_multi_series_predictions.png",
    )
    hist_path = os.path.join(
        args.outdir,
        f"study{RUN_ID}_autoreg_mse_histogram.png",
    )
    csv_path = os.path.join(
        args.outdir,
        f"study{RUN_ID}_autoreg_mse_by_curve.csv",
    )

    plot_residuals_vs_step(rollouts, residual_path)
    plot_multiple_series_predictions(rollouts, series_path)
    plot_mse_histogram(rollouts, hist_path)
    save_mse_csv(rollouts, csv_path)

    print("Saved:")
    print(" ", residual_path)
    print(" ", series_path)
    print(" ", hist_path)
    print(" ", csv_path)


if __name__ == "__main__":
    main()
