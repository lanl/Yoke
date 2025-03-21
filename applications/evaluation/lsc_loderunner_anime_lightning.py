"""Script to produce animation of LodeRunner prediction.

This script allows production of an animation of a single hydrodynamic field
within one lsc240420 simulation set of NPZ files.

Three types of images are produced:

    - Ground truth
    - LodeRunner Checkpoint prediction
    - Discrepancy

"""

import os
import glob
import argparse
import numpy as np

import torch

from yoke.models.vit.swin.bomberman import LodeRunner, Lightning_LodeRunner
import yoke.torch_training_utils as tr
from yoke.helpers import cli

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Use LaTeX font
font = {"family": "serif"}
plt.rc("font", **font)
plt.rcParams["figure.figsize"] = (6, 6)


###################################################################
# Define command line argument parser
descr_str = (
    "Create animation of single hydro-field for LodeRunner on lsc240420 "
    "simulation IDX."
)
parser = argparse.ArgumentParser(
    prog="Animation of LodeRunner",
    description=descr_str,
    fromfile_prefix_chars="@",
)

parser.add_argument(
    "--checkpoint",
    action="store",
    type=str,
    nargs="+",
    help="Paths to .ckpt model checkpoint(s) to evaluate.",
)

parser.add_argument(
    "--indir",
    "-D",
    action="store",
    type=str,
    default="/lustre/scratch5/exempt/artimis/mpmm/lsc240420/",
    help="Directory to find NPZ files.",
)

parser.add_argument(
    "--outdir",
    "-O",
    action="store",
    type=str,
    default="./",
    help="Directory to output images to.",
)

# run index
# Example: lsc240420_id00201_pvi_idx00100.npz
parser.add_argument(
    "--runID",
    "-R",
    action="store",
    type=int,
    default=201,
    help="Run identifier index.",
)

parser.add_argument(
    "--verbose", "-V", action="store_true", help="Flag to turn on debugging output."
)

parser = cli.add_model_args(parser=parser)


def print_NPZ_keys(npzfile: str = "./lsc240420_id00201_pvi_idx00100.npz") -> None:
    """Print keys of NPZ file."""
    NPZ = np.load(npzfile)
    print("NPZ file keys:")
    for key in NPZ.keys():
        print(key)

    NPZ.close()

    return


def singlePVIarray(
    npzfile: str = "./lsc240420_id00201_pvi_idx00100.npz", FIELD: str = "av_density"
) -> np.array:
    """Function to grab single array from NPZ.

    Args:
       npzfile (str): File name for NPZ.
       FIELD (str): Field to return array for.

    Returns:
       field (np.array): Array of hydro-dynamic field for plotting

    """
    NPZ = np.load(npzfile)
    arrays_dict = dict()
    for key in NPZ.keys():
        arrays_dict[key] = NPZ[key]

    NPZ.close()

    return arrays_dict[FIELD]


if __name__ == "__main__":
    # Parse commandline arguments
    args = parser.parse_args()

    # Assign command-line arguments
    VERBOSE = args.verbose

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Assemble filenames
    # Example: lsc240420_id00201_pvi_idx00100.npz
    npz_glob = os.path.join(
        args.indir, f"lsc240420_id{args.runID:05d}_pvi_idx?????.npz"
    )
    npz_list = sorted(glob.glob(npz_glob))
    if VERBOSE:
        print("NPZ files:", npz_list)

    # Prepare model.
    default_vars = [
        "density_case",
        "density_cushion",
        "density_maincharge",
        "density_outside_air",
        "density_striker",
        "density_throw",
        "Uvelocity",
        "Wvelocity",
    ]
    image_size = [1120, 400]
    patch_size = (5, 5)
    window_sizes = [(2, 2) for _ in range(4)]
    loderunner = LodeRunner(
        default_vars=default_vars,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=args.embed_dim,
        emb_factor=2,
        num_heads=8,
        block_structure=args.block_structure,
        window_sizes=window_sizes,
        patch_merge_scales=[
            (2, 2),
            (2, 2),
            (2, 2),
        ],
    )

    # Prepare some inputs.
    in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    Dt = torch.tensor([0.25])

    # Loop over checkpoints and evaluate.
    loss_per_image = torch.zeros(len(args.checkpoint), len(npz_list), dtype=torch.float)
    for m, ckpt in enumerate(args.checkpoint):
        model = Lightning_LodeRunner.load_from_checkpoint(
            ckpt,
            model=loderunner,
            in_vars=in_vars,
            out_vars=out_vars,
        )
        model.eval()

        # Loop through images and make predictions.
        loss = torch.nn.functional.mse_loss
        for k, npzfile in enumerate(npz_list):
            if args.verbose:
                print(f"Evaluating file {k+1} of {len(npz_list)}...")
            # Get index
            pviIDX = npzfile.split("idx")[1]
            pviIDX = int(pviIDX.split(".")[0])

            # Get the coordinates and time
            simtime = singlePVIarray(npzfile=npzfile, FIELD="sim_time")
            Rcoord = singlePVIarray(npzfile=npzfile, FIELD="Rcoord")
            Zcoord = singlePVIarray(npzfile=npzfile, FIELD="Zcoord")

            # Make prediction.
            true_img_list = []
            for hfield in default_vars:
                tmp_img = singlePVIarray(npzfile=npzfile, FIELD=hfield)
                tmp_img = np.nan_to_num(tmp_img, nan=0.0)
                true_img_list.append(tmp_img)

            # Concatenate images channel first.
            true_img = torch.tensor(np.stack(true_img_list, axis=0)).to(torch.float32)
            true_img = torch.unsqueeze(true_img, 0)
            true_img = true_img.to(model.device)
            Dt = Dt.to(model.device)

            if k == 0:
                # Make a prediction from ground truth input.
                pred_img = model(true_img, Dt)
            else:
                # Evaluate LodeRunner from last prediction
                pred_img = model(pred_img.to(model.device), Dt)

            pred_img = pred_img.detach().cpu()
            true_img = true_img.detach().cpu()
            Dt = Dt.detach().cpu()
            pred_rho = np.squeeze(pred_img.numpy())
            pred_rho = pred_rho[0:6, :, :].sum(0)

            # Compute loss.
            loss_per_image[m, k] = loss(
                pred_img.squeeze(), true_img.squeeze()
            )

            # Sum for true average density
            true_rho = true_img.numpy()[0]
            true_rho = true_rho[0:6, :, :].sum(0)

            # Plot Truth/Prediction/Discrepancy panel.
            fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
            fig1.suptitle(f"T={float(simtime):.2f}us", fontsize=18)
            img1 = ax1.imshow(
                true_rho,
                aspect="equal",
                extent=[0.0, Rcoord.max(), Zcoord.min(), Zcoord.max()],
                origin="lower",
                cmap="jet",
                vmin=true_rho.min(),
                vmax=true_rho.max(),
            )
            ax1.set_ylabel("Z-axis", fontsize=16)
            ax1.set_xlabel("R-axis", fontsize=16)
            ax1.set_title("True", fontsize=18)

            img2 = ax2.imshow(
                pred_rho,
                aspect="equal",
                extent=[0.0, Rcoord.max(), Zcoord.min(), Zcoord.max()],
                origin="lower",
                cmap="jet",
                vmin=true_rho.min(),
                vmax=true_rho.max(),
            )
            ax2.set_title("Predicted", fontsize=18)
            ax2.tick_params(axis="y", which="both", left=False, labelleft=False)

            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes("right", size="10%", pad=0.1)
            fig1.colorbar(img2, cax=cax2).set_label("Density (g/cc)", fontsize=14)

            discrepancy = np.abs(true_rho - pred_rho)
            img3 = ax3.imshow(
                discrepancy,
                aspect="equal",
                extent=[0.0, Rcoord.max(), Zcoord.min(), Zcoord.max()],
                origin="lower",
                cmap="hot",
                vmin=discrepancy.min(),
                vmax=0.3 * discrepancy.max(),
            )
            ax3.set_title("Discrepancy", fontsize=18)
            ax3.tick_params(axis="y", which="both", left=False, labelleft=False)

            divider3 = make_axes_locatable(ax3)
            cax3 = divider3.append_axes("right", size="10%", pad=0.1)
            fig1.colorbar(img3, cax=cax3).set_label("Discrepancy", fontsize=14)

            # Save images
            fig1.savefig(
                os.path.join(
                    args.outdir,
                    f"loderunner_prediction_ckpt{m}_id{args.runID:05d}_idx{pviIDX:05d}.png",
                ),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()
        model = model.cpu()  # free up GPU before next model load just in case
        fig, ax = plt.subplots()
        plt.plot(Dt * np.arange(len(npz_list)), loss_per_image[m], ".")
        # ax.set_yscale("log")
        plt.xlabel("time ($\mu$s)")
        plt.ylabel("MSE")
        fig.savefig(
            os.path.join(
                args.outdir,
                f"loss_per_timestep_ckpt{m}_id{args.runID:05d}.png",
            ),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    fig, ax = plt.subplots()
    for m, ckpt in enumerate(args.checkpoint):
        plt.plot(
            Dt * np.arange(len(npz_list)), loss_per_image[m], ".", label=f"ckpt{m}"
        )
    # ax.set_yscale("log")
    plt.legend()
    plt.xlabel("time ($\mu$s)")
    plt.ylabel("MSE")
    fig.savefig(
        os.path.join(
            args.outdir,
            f"loss_per_timestep_id{args.runID:05d}.png",
        ),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    if args.verbose:
        print(f"Done.")
