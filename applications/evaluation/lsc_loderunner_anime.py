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

from yoke.models.vit.swin.bomberman import LodeRunner
import yoke.torch_training_utils as tr

# Imports for plotting
# To view possible matplotlib backends use
# >>> import matplotlib
# >>> bklist = matplotlib.rcsetup.interactive_bk
# >>> print(bklist)
import matplotlib

# matplotlib.use('MacOSX')
# matplotlib.use('pdf')
# matplotlib.use('QtAgg')
# Get rid of type 3 fonts in figures
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Ensure LaTeX font
font = {"family": "serif"}
plt.rc("font", **font)
plt.rcParams["figure.figsize"] = (6, 6)


###################################################################
# Define command line argument parser
descr_str = (
    "Create animation of single hydro-field for LodeRunner on lsc240420 simulation IDX."
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
    default="./study018_modelState_epoch0100.hdf5",
    help="Name of HDF5 model checkpoint to evaluate output for.",
)

# indir
parser.add_argument(
    "--indir",
    "-D",
    action="store",
    type=str,
    default="/lustre/vescratch1/exempt/artimis/mpmm/lsc240420/",
    help="Directory to find NPZ files.",
)

# outdir
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

# run index
# Example: lsc240420_id00201_pvi_idx00100.npz
parser.add_argument(
    "--embed_dim",
    action="store",
    type=int,
    default=128,
    help="Embed dim.",
)

parser.add_argument(
    "--verbose", "-V", action="store_true", help="Flag to turn on debugging output."
)

"""
This argument can be a little confusing, so here is an example showing how this works.
Assume the initial image is at time zero.
Assume you are trying to predict the n'th timestep.
This would mean that you are most likely trying to predict time k * n, for some constant k.
In single mode, this prediction would be made by taking the TRUE input for the n-1 timestep.
    The model would then use that, along with timestep k to predict timestep n.
In chained mode, the following relation is used: P(n) = RM(I, n, k)
    P(n) is the prediction of the n'th timestep.
    RM(I, n, k) would mean run repeated predictions of the model.
        Here is a simplified version of the code for this function:
        def RM(inp, steps, dt):
            current = inp
            for _ in range(n): # underscore used bc it doesn't matter
                current = M(current, dt) # where M is the model.
            return current
In timestep mode, the initial image at time zero is used for ALL predictions.
    The timestep by itself is used to determine how far forward to predict.
Now, you're probably asking which mode should you use.
You must determine that, but here are some tips that may help:
    Single Mode
        Advantages:
            No propagating error.
            Requires one model call to predict. (O(1) model calls)
            Can be used on models trained to only predict one timestep ahead.
        Disadvantages:
            Cannot predict multiple timesteps ahead.
                This mode takes one true timestep and predicts the next.
            Uses a constant dt, so some parameters may be ignored.
    Chained Mode
        Advantages:
            Requires only the initial timestep to generate predictions.
            Can be used on models trained to only predict one timestep ahead.
        Disadvantages:
            Prediction of a timestep far ahead can be costly.
                Predicting timestep n requires n calls to the model. (O(n) model calls)
            Predictions lose accuracy across the rollout.
                The prediction quality decays significantly as the rollout progresses forward.
            Uses a constant dt, so some parameters may be ignored.
    Timestep Mode
        Advantages:
            Requires only the initial timestep to generate predictions.
            Does not cause the dt related parameters to be wasted.
            Allows for predictions of a specific timestep to only require one model call. (O(1) model calls)
        Consideration:
            Requires a model trained with a variety of timesteps.
                Training the model to predict one timestep ahead (exclusively) will NOT work effectively in this mode.
"""
parser.add_argument(
    "--mode",
    action="store",
    type=str,
    choices=["single", "chained", "timestep"],
    default="single",
    help=(
        "Determines the mode of how to do the prediction. "
        "Single mode does not propagate outputs. "
        "Chained mode propagates outputs. "
        "Timestep uses the initial image and just varies the timestep delta. "
    ),
)


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
    args_ns = parser.parse_args()

    # Assign command-line arguments
    checkpoint = args_ns.checkpoint
    indir = args_ns.indir
    outdir = args_ns.outdir
    runID = args_ns.runID
    embed_dim = args_ns.embed_dim
    VERBOSE = args_ns.verbose
    mode = args_ns.mode

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Assemble filenames
    # Example: lsc240420_id00201_pvi_idx00100.npz
    npz_glob = os.path.join(indir, f"lsc240420_id{runID:05d}_pvi_idx?????.npz")
    npz_list = sorted(glob.glob(npz_glob))

    if VERBOSE:
        print("NPZ files:", npz_list)

    # Instantiate model from checkpoint
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
    block_structure = (
        1,
        1,
        11,
        2,
    )
    model = LodeRunner(
        default_vars=default_vars,
        image_size=(1120, 400),
        patch_size=(10, 5),  # Since using half-image, halve patch size.
        embed_dim=embed_dim,
        emb_factor=2,
        num_heads=8,
        block_structure=block_structure,
        window_sizes=[
            (8, 8),
            (8, 8),
            (4, 4),
            (2, 2),
        ],
        patch_merge_scales=[
            (2, 2),
            (2, 2),
            (2, 2),
        ],
    )

    # Load weights from checkpoint
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-6,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
    )

    checkpoint_epoch = tr.load_model_and_optimizer_hdf5(model, optimizer, checkpoint)
    model.eval()

    # Build input data
    in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

    # Time offset
    # Loop through images
    for k, npzfile in enumerate(npz_list):
        # Get index
        Dt = torch.tensor([0.25]) if mode != "timestep" else torch.tensor([k * 0.25])
        pviIDX = npzfile.split("idx")[1]
        pviIDX = int(pviIDX.split(".")[0])

        # Get the coordinates and time
        simtime = singlePVIarray(npzfile=npzfile, FIELD="sim_time")
        Rcoord = singlePVIarray(npzfile=npzfile, FIELD="Rcoord")
        Zcoord = singlePVIarray(npzfile=npzfile, FIELD="Zcoord")

        # Step prediction
        input_img_list = []
        for hfield in default_vars:
            tmp_img = singlePVIarray(npzfile=npzfile, FIELD=hfield)

            # Remember to replace all NaNs with 0.0
            tmp_img = np.nan_to_num(tmp_img, nan=0.0)
            input_img_list.append(tmp_img)

        # Concatenate images channel first.
        temp_img = torch.tensor(np.stack(input_img_list, axis=0)).to(torch.float32)
        if mode == "timestep":
            if k == 0:
                input_img = temp_img
                initial_input = temp_img.clone()  # Save initial state
            else:
                input_img = initial_input  # Always use initial state as input
        else:
            input_img = temp_img

        # Sum for true average density
        true_rho = temp_img.detach().numpy()
        true_rho = true_rho[0:6, :, :].sum(0)

        # Make a prediction
        if mode != "chained":
            pred_img = model(torch.unsqueeze(input_img, 0), in_vars, out_vars, Dt)
            pred_rho = np.squeeze(pred_img.detach().numpy())
            pred_rho = pred_rho[0:6, :, :].sum(0)
        else:
            if k == 0:
                input_img_list = []
                for hfield in default_vars:
                    tmp_img = singlePVIarray(npzfile=npzfile, FIELD=hfield)

                    # Remember to replace all NaNs with 0.0
                    tmp_img = np.nan_to_num(tmp_img, nan=0.0)
                    input_img_list.append(tmp_img)

                # Concatenate images channel first.
                input_img = torch.tensor(
                    np.stack(
                        input_img_list,
                        axis=0)
                    ).to(torch.float32)

                # Sum for true average density
                true_rho = input_img.detach().numpy()
                true_rho = true_rho[0:6, :, :].sum(0)

                # Make a prediction
                pred_img = model(torch.unsqueeze(input_img, 0), in_vars, out_vars, Dt)
                pred_rho = np.squeeze(pred_img.detach().numpy())
                pred_rho = pred_rho[0:6, :, :].sum(0)

            else:
                # Get ground-truth average density
                true_img_list = []
                for hfield in default_vars:
                    tmp_img = singlePVIarray(npzfile=npzfile, FIELD=hfield)

                    # Remember to replace all NaNs with 0.0
                    tmp_img = np.nan_to_num(tmp_img, nan=0.0)
                    true_img_list.append(tmp_img)

                # Concatenate images channel and sum
                true_img = np.stack(true_img_list, axis=0)

                # Sum for true average density
                true_rho = true_img[0:6, :, :].sum(0)

                # Evaluate LodeRunner from last prediction
                pred_img = model(pred_img, in_vars, out_vars, Dt)
                pred_rho = np.squeeze(pred_img.detach().numpy())
                pred_rho = pred_rho[0:6, :, :].sum(0)

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
                outdir, f"loderunner_prediction_id{runID:05d}_idx{pviIDX:05d}.png"
            ),
            bbox_inches="tight",
        )
        plt.close()
