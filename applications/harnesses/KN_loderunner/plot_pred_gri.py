import argparse
import os

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

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
        description="Plot one-step and autoregressive predictions for scalar GRI LodeRunner."
    )

    parser.add_argument("--study", type=int, default=24)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument("--N_imgs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_future_steps", type=int, default=15)

    parser.add_argument(
        "--norm_stats_path",
        type=str,
        default="kilonova_gri_norm_stats.npz",
    )

    parser.add_argument("--outdir", type=str, default=None)

    return parser.parse_args()


def resolve_paths(args):
    run_id = study_tag(args.study)

    if args.ckpt is None:
        args.ckpt = (
            f"runs/study_{run_id}/study{run_id}_modelState_epoch{args.epoch:04d}.pth"
        )

    if args.outdir is None:
        args.outdir = f"runs/study_{run_id}/pred_plots_gri"

    return run_id


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


def plot_onestep_predictions(
    idxs,
    preds,
    targets,
    prefix,
    outpath,
):
    preds = np.asarray(preds, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    prefix = np.asarray(prefix, dtype=np.float32)

    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    for band_idx, band_name in enumerate(BAND_NAMES):
        ax = axes[band_idx]

        ax.scatter(
            idxs,
            preds[:, band_idx],
            label="Predicted next magnitude",
        )

        ax.scatter(
            idxs,
            targets[:, band_idx],
            label="True next magnitude",
        )

        ax.scatter(
            np.arange(len(prefix)) - len(prefix),
            prefix[:, band_idx],
            label="Initial context window",
        )

        ax.invert_yaxis()
        ax.set_ylabel(f"{band_name} norm mag")
        ax.set_title(f"{band_name}-band one-step prediction")

    axes[-1].set_xlabel("Sample index")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=8,
        ncol=1,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_autoreg_rollout(
    idxs_seq,
    preds_seq,
    truth_seq,
    prefix,
    outpath,
):
    preds_seq = np.asarray(preds_seq, dtype=np.float32)
    truth_seq = np.asarray(truth_seq, dtype=np.float32)
    prefix = np.asarray(prefix, dtype=np.float32)

    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    for band_idx, band_name in enumerate(BAND_NAMES):
        ax = axes[band_idx]

        ax.scatter(
            idxs_seq,
            preds_seq[:, band_idx],
            label="Autoregressive predictions",
        )

        ax.scatter(
            idxs_seq,
            truth_seq[:, band_idx],
            label="Truth",
        )

        ax.scatter(
            np.arange(len(prefix)) - len(prefix),
            prefix[:, band_idx],
            label="Initial context window",
        )

        ax.invert_yaxis()
        ax.set_ylabel(f"{band_name} norm mag")
        ax.set_title(f"{band_name}-band autoregressive rollout")

    axes[-1].set_xlabel("Autoregressive step")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=8,
        ncol=1,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


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

    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ------------------------------------------------------------
    # One-step predictions
    # ------------------------------------------------------------
    preds = []
    targets = []
    idxs = []
    prefix = None

    for idx, (x, target_delta, Dt) in enumerate(loader):
        if x.shape[0] != 1:
            raise ValueError(
                "This plotting script assumes batch_size=1 so each point maps "
                "cleanly to a single light curve window."
            )

        x = x.to(torch.float32).to(device)
        target_delta = target_delta.to(torch.float32).to(device)
        Dt = Dt.to(torch.float32).to(device)

        if prefix is None:
            context_vals, _ = split_gri_context(
                x[0].detach().cpu(),
                context_len=context_len,
                n_bands=3,
            )
            prefix = context_vals

        with torch.no_grad():
            pred_delta = model(
                x,
                in_vars=None,
                out_vars=None,
                Dt=Dt,
            )

        context_vals, _ = split_gri_context(
            x[0].detach().cpu(),
            context_len=context_len,
            n_bands=3,
        )

        last_vals = torch.tensor(
            context_vals[-1],
            dtype=torch.float32,
            device=device,
        )

        pred_next = last_vals + pred_delta[0].reshape(3)
        true_next = last_vals + target_delta[0].reshape(3)

        preds.append(pred_next.detach().cpu().numpy())
        targets.append(true_next.detach().cpu().numpy())
        idxs.append(idx)

    onestep_path = os.path.join(
        args.outdir,
        f"study{run_id}_pred_vs_truth_gri_delta_onestep.png",
    )

    plot_onestep_predictions(
        idxs=idxs,
        preds=preds,
        targets=targets,
        prefix=prefix,
        outpath=onestep_path,
    )

    # ------------------------------------------------------------
    # Autoregressive rollout
    # ------------------------------------------------------------
    if len(eval_dataset) == 0:
        raise RuntimeError("Evaluation dataset is empty.")

    x0, _, _ = eval_dataset[0]

    context_vals, rel_t0 = split_gri_context(
        x0,
        context_len=context_len,
        n_bands=3,
    )

    pred_vals = [row.copy() for row in context_vals]
    true_vals = [row.copy() for row in context_vals]

    preds_seq = []
    truth_seq = []
    idxs_seq = []

    with torch.no_grad():
        for step in range(args.n_future_steps):
            if step >= len(eval_dataset):
                break

            x_true_step, target_delta, future_Dt = eval_dataset[step]

            x_true_step = torch.as_tensor(x_true_step, dtype=torch.float32)
            target_delta = torch.as_tensor(
                target_delta,
                dtype=torch.float32,
                device=device,
            ).reshape(3)

            future_Dt = torch.as_tensor(
                future_Dt,
                dtype=torch.float32,
                device=device,
            )

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
            ).reshape(3)

            pred_next = (
                torch.tensor(
                    pred_vals[-1],
                    dtype=torch.float32,
                    device=device,
                )
                + pred_delta
            )

            true_next = (
                torch.tensor(
                    true_vals[-1],
                    dtype=torch.float32,
                    device=device,
                )
                + target_delta
            )

            pred_next_np = pred_next.detach().cpu().numpy()
            true_next_np = true_next.detach().cpu().numpy()

            preds_seq.append(pred_next_np)
            truth_seq.append(true_next_np)
            idxs_seq.append(step)

            pred_vals.append(pred_next_np)
            true_vals.append(true_next_np)

    autoreg_path = os.path.join(
        args.outdir,
        f"study{run_id}_pred_vs_truth_gri_delta_autoreg_clean.png",
    )

    plot_autoreg_rollout(
        idxs_seq=idxs_seq,
        preds_seq=preds_seq,
        truth_seq=truth_seq,
        prefix=context_vals,
        outpath=autoreg_path,
    )

    print("Saved:")
    print(" ", onestep_path)
    print(" ", autoreg_path)


if __name__ == "__main__":
    main()
