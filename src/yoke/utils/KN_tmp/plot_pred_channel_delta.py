import argparse
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from yoke.models.vit.swin.bomberman import LodeRunner
from torch.utils.data import DataLoader

from train_LodeRunner_ddp import Kilonova_lc_img_DataSet_channels_context

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rc("font", family="serif")
plt.rcParams["figure.figsize"] = (6, 6)

# ============================================================
# RUN IDENTIFIER
# ============================================================
RUN_ID = "013"


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default=f"runs/study_{RUN_ID}/study{RUN_ID}_modelState_epoch0100.pth",
    )
    parser.add_argument("--N_imgs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_future_steps", type=int, default=15)

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

    model = LodeRunner(**model_args)
    model.to(device)

    state_dict = ckpt["model_state_dict"]

    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=True)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.noise_scale = noise_scale
    model.eval()

    return model, context_len


def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model, context_len = load_channel_model(args.ckpt, device)

    eval_dataset = Kilonova_lc_img_DataSet_channels_context(
        half_image=False,
        N_imgs=args.N_imgs,
        context_len=context_len,
    )

    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)
    out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)

    # ------------------------------------------------------------
    # One-step predictions
    # ------------------------------------------------------------
    preds = []
    targets = []
    idxs = []
    prefix = []

    for idx, (context_img, target_delta, Dt) in enumerate(loader):
        context_img = context_img.to(device)
        target_delta = target_delta.to(device)
        Dt = Dt.to(torch.float32).to(device)

        if idx == 0:
            context_means = context_img.mean(dim=(2, 3))[0].detach().cpu().numpy()
            for context in context_means:
                prefix.append(context.mean().item())

        with torch.no_grad():
            pred_delta_img = model(context_img, in_vars, out_vars, Dt)

        last_mag_img = context_img[:, -1:]

        pred_next_img = last_mag_img + pred_delta_img[:, -1:]
        true_next_img = last_mag_img + target_delta[:, -1:]

        preds.append(pred_next_img.mean().item())
        targets.append(true_next_img.mean().item())
        idxs.append(idx)

    plt.figure()
    plt.scatter(idxs, preds, label="Predicted next magnitude")
    plt.scatter(idxs, targets, label="True next magnitude")
    plt.scatter(
        np.arange(len(prefix)) - len(prefix),
        prefix,
        label="Initial context window",
    )

    plt.legend()
    plt.gca().invert_yaxis()
    plt.xlabel("Sample index")
    plt.ylabel("Normalized magnitude")
    plt.tight_layout()

    plt.savefig(
        f"study{RUN_ID}_pred_vs_truth_channel_delta_onestep.png",
        dpi=200,
    )

    # ------------------------------------------------------------
    # Autoregressive rollout
    # ------------------------------------------------------------
    context_seq, target_delta, Dt = next(iter(loader))

    x = context_seq.to(device)

    preds_seq = []
    truth_seq = []
    idxs_seq = []

    future_iter = iter(loader)

    pred_next_img = None
    true_next_img = None

    for step in range(args.n_future_steps):
        try:
            _, future_target_delta, future_Dt = next(future_iter)
        except StopIteration:
            break

        future_target_delta = future_target_delta.to(device)
        future_Dt = future_Dt.to(torch.float32).to(device)

        with torch.no_grad():
            pred_delta_img = model(x, in_vars, out_vars, future_Dt)

        last_mag_img = x[:, -1:]

        pred_next_img = last_mag_img + pred_delta_img[:, -1:]
        true_next_img = last_mag_img + future_target_delta[:, -1:]

        preds_seq.append(pred_next_img.mean().item())
        truth_seq.append(true_next_img.mean().item())
        idxs_seq.append(step)

        # Append predicted absolute next magnitude
        x = torch.cat([x[:, 1:], pred_next_img.detach()], dim=1)

    plt.figure()

    plt.scatter(idxs_seq, preds_seq, label="Autoregressive predictions")
    plt.scatter(idxs_seq, truth_seq, label="Truth")
    plt.scatter(
        np.arange(len(prefix)) - len(prefix),
        prefix,
        label="Initial context window",
    )

    plt.legend()
    plt.gca().invert_yaxis()
    plt.xlabel("Autoregressive step")
    plt.ylabel("Normalized magnitude")
    plt.tight_layout()

    plt.savefig(
        f"study{RUN_ID}_pred_vs_truth_channel_delta_autoreg.png",
        dpi=200,
    )

    # ------------------------------------------------------------
    # Image comparison
    # ------------------------------------------------------------
    if pred_next_img is not None and true_next_img is not None:
        pred_plot = pred_next_img[0, 0].detach().cpu().numpy()
        true_plot = true_next_img[0, 0].detach().cpu().numpy()
        error_plot = pred_plot - true_plot

        vmin = min(pred_plot.min(), true_plot.min())
        vmax = max(pred_plot.max(), true_plot.max())
        err_max = np.max(np.abs(error_plot))

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))

        im1 = ax1.imshow(pred_plot, origin="lower", vmin=vmin, vmax=vmax)
        ax1.set_title("Prediction")

        ax2.imshow(true_plot, origin="lower", vmin=vmin, vmax=vmax)
        ax2.set_title("Truth")

        im3 = ax3.imshow(
            error_plot,
            origin="lower",
            vmin=-err_max,
            vmax=err_max,
        )
        ax3.set_title("Error (Pred - Truth)")

        cbar = fig.colorbar(im1, ax=[ax1, ax2], shrink=0.8)
        cbar.set_label("Normalized magnitude")

        cbar_err = fig.colorbar(im3, ax=ax3, shrink=0.8)
        cbar_err.set_label("Error")

        for ax in (ax1, ax2, ax3):
            ax.axis("off")

        plt.tight_layout()

        plt.savefig(
            f"study{RUN_ID}_img_comp_channel_delta.png",
            bbox_inches="tight",
            dpi=200,
        )


if __name__ == "__main__":
    main()
