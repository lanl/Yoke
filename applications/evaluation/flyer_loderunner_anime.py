"""Script to produce animation of LodeRunner prediction.

This script allows production of an animation of a single field
within one flyer plate simulation set of NPZ files.

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
from yoke.utils.checkpointing import load_model_and_optimizer
from yoke.datasets.load_npz_dataset import LabeledData

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

TIMESTEP_DELTA = 0.25  # us, constant timestep for inference
DT_CONST = torch.tensor([TIMESTEP_DELTA])  # Constant timestep for inference


###################################################################
# Define command line argument parser
descr_str = (
    "Create animation of single field for LodeRunner on flyer plate simulation IDX."
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
    default="./study001_modelState_epoch0005.pth",
    help="Name of .pth model checkpoint to evaluate output for.",
)

# indir
parser.add_argument(
    "--indir",
    "-D",
    action="store",
    type=str,
    default="/net/sescratch1/exempt/artimis/users/rworley/flyer/",
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
# Example: flyer260625_id00001/flyer260625_id00001_pvi_idx00100.npz
parser.add_argument(
    "--runID",
    "-R",
    action="store",
    type=str,
    default="flyer260625_id00001",
    help="Flyerplate run prefix.",
)

# run index
# Example: flyer260625_id00001/flyer260625_id00001_pvi_idx00100.npz
parser.add_argument(
    "--embed_dim",
    action="store",
    type=int,
    default=128,
    help="Embed dim.",
)

parser.add_argument(
    "--csv_filepath",
    action="store",
    type=str,
    default="/usr/projects/artimis/mpmm/ryanw/design_flyer260625_id00001-01199.csv",
    help="Design CSV used to derive flyerplate channel mapping.",
)

parser.add_argument(
    "--num_pngs",
    action="store",
    type=int,
    default=None,
    help="Maximum number of PNG frames to generate. Default is all frames.",
)

parser.add_argument(
    "--verbose", "-V", action="store_true", help="Flag to turn on debugging output."
)

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


def singlePVIarray(
    npzfile: str = "./lsc240420_id00201_pvi_idx00100.npz", FIELD: str = "av_density"
) -> np.ndarray:
    """Function to grab single array from NPZ.

    Args:
       npzfile (str): File name for NPZ.
       FIELD (str): Field to return array for.

    Returns:
       field (np.ndarray): Array of hydro-dynamic field for plotting

    """
    NPZ = np.load(npzfile)
    arrays_dict = dict()
    for key in NPZ.keys():
        arrays_dict[key] = NPZ[key]

    NPZ.close()

    return arrays_dict[FIELD]


def scalarPVIarray(npzfile: str, field: str) -> float:
    """Return a scalar NPZ entry as a plain float."""
    value = singlePVIarray(npzfile=npzfile, FIELD=field)
    return float(np.asarray(value).reshape(-1)[0])


def loderunner_inference(
    model: torch.nn.Module,
    input_img: torch.Tensor,
    in_vars: torch.Tensor,
    out_vars: torch.Tensor,
    delta_t: torch.Tensor,
    present_vars: list[str],
) -> tuple[torch.Tensor, np.ndarray]:
    """Function to run prediction on a Yoke model and generate the density field.

    The input tensor is either the true state (from an NPZ file),
    or a predicted state (output from a previous prediction from the model).

    Args:
        model (torch.nn.Module): The model used for inferencing.
        input_img (torch.Tensor): The input tensor.
        in_vars (torch.Tensor): The list of input channels.
        out_vars (torch.Tensor): The list of channels the model should output.
        delta_t (torch.Tensor): The amount of time forward the model should predict.
        present_vars: (list[str]): Channel-aligned field names for active variables.

    Returns:
        output (tuple[torch.Tensor, np.ndarray]): The predicted output and density field.
    """
    pred_img = model(torch.unsqueeze(input_img, 0), in_vars, out_vars, delta_t)
    pred_rho = np.squeeze(pred_img.detach().numpy())
    density_idx = flyer_density_indices(present_vars)
    pred_rho = pred_rho[density_idx, :, :].sum(0)

    return pred_img, pred_rho


def prepare_input_images(
    npzfile: str,
    default_vars: list[str],
) -> tuple[torch.Tensor, list[str]]:
    """Prepare input images from NPZ file."""
    input_img_list = []
    present_vars = []

    NPZ = np.load(npzfile)

    for hfield in default_vars:
        if hfield not in NPZ:
            continue

        tmp_img = NPZ[hfield]

        # Keep only 2D image-like arrays
        if tmp_img.ndim != 2:
            continue

        tmp_img = np.nan_to_num(tmp_img, nan=0.0)
        input_img_list.append(tmp_img)
        present_vars.append(hfield)

    NPZ.close()

    return torch.tensor(np.stack(input_img_list, axis=0)).to(torch.float32), present_vars


def prepare_flyerplate_eval_sample(
    npzfile: str,
    csv_filepath: str,
    thermodynamic_variables: str = "density",
    kinematic_variables: str = "velocity",
) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    """Prepare flyerplate inference inputs consistent with training-time assembly."""
    ld = LabeledData(
        npzfile,
        csv_filepath,
        thermodynamic_variables=thermodynamic_variables,
        kinematic_variables=kinematic_variables,
    )

    active_npz_field_names = ld.get_active_npz_field_names()
    channel_map = ld.get_channel_map()

    input_img_list = []
    present_fields = []
    filtered_channel_map = []

    with np.load(npzfile) as npz:
        available = set(npz.files)
        for hfield, cm in zip(active_npz_field_names, channel_map):
            if hfield not in available:
                continue

            tmp_img = np.nan_to_num(npz[hfield], nan=0.0)
            if tmp_img.ndim != 2:
                continue

            input_img_list.append(tmp_img)
            present_fields.append(hfield)
            filtered_channel_map.append(cm)

    if not input_img_list:
        raise RuntimeError(f"No active 2D flyerplate fields found in {npzfile}")

    input_img = torch.tensor(np.stack(input_img_list, axis=0)).to(torch.float32)
    cm_tensor = torch.tensor(filtered_channel_map, dtype=torch.long)
    return input_img, present_fields, cm_tensor


MAX_FLYER_LAYERS = 6


def flyerplate_layer_fields(prefixes: list[str]) -> list[str]:
    """Return flyerplate field names for each layer/prefix combination."""
    fields: list[str] = []
    for layer_idx in range(MAX_FLYER_LAYERS):
        layer_name = f"layer{layer_idx:03d}"
        for prefix in prefixes:
            fields.append(f"{prefix}_{layer_name}")
    return fields


def flyer_density_indices(default_vars: list[str]) -> list[int]:
    """Return indices of density-layer fields in the provided channel list."""
    density_idx = []
    for idx, field in enumerate(default_vars):
        if field.startswith("density_layer"):
            density_idx.append(idx)
    return density_idx


def density_sum(img: torch.Tensor | np.ndarray, present_vars: list[str]) -> np.ndarray:
    """Return summed flyer-layer density image from active channels."""
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)

    density_idx = flyer_density_indices(present_vars)
    return arr[density_idx, :, :].sum(0)


if __name__ == "__main__":
    # Parse commandline arguments
    args_ns = parser.parse_args()

    # Assign command-line arguments
    checkpoint = args_ns.checkpoint
    indir = args_ns.indir
    outdir = args_ns.outdir
    runID = args_ns.runID
    embed_dim = args_ns.embed_dim
    csv_filepath = args_ns.csv_filepath
    num_pngs = args_ns.num_pngs
    VERBOSE = args_ns.verbose
    mode = args_ns.mode

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Assemble filenames
    # Example: flyer260625_id00001/flyer260625_id00001_pvi_idx00100.npz
    npz_glob = os.path.join(indir, runID, f"{runID}_pvi_idx?????.npz")
    npz_list = sorted(glob.glob(npz_glob))

    if VERBOSE:
        print("NPZ files:", npz_list)

    available_models = {"LodeRunner": LodeRunner}

    model, optimizer, checkpoint_epoch = load_model_and_optimizer(
        checkpoint,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={
            "lr": 1e-6,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0.01,
        },
        available_models=available_models,
        device=torch.device("cpu"),
    )

    model.eval()

    default_vars = [
        "sim_time",
        "av_density",
        "av_pressure",
        "av_temperature",
        "density_Air",
        "energy_Air",
        "pressure_Air",
        "sound_speed_Air",
        "vofm_Air",
        "Uvelocity",
        "Wvelocity",
        "Rcoord",
        "Zcoord",
    ] + flyerplate_layer_fields(
        [
            "density",
            "energy",
            "plst_strain",
            "pressure",
            "shear_modulus",
            "sound_speed",
            "strain_rate",
            "Sxxm",
            "Sxzm",
            "Syym",
            "Szzm",
            "vofm",
            "yield",
        ]
    )

    if len(npz_list) < 2:
        raise RuntimeError("Need at least two NPZ frames for flyerplate comparison.")

    initial_file = npz_list[0]
    initial_img, initial_present_vars, initial_channel_map = (
        prepare_flyerplate_eval_sample(
            initial_file,
            csv_filepath,
        )
    )
    initial_time = scalarPVIarray(initial_file, "sim_time")

    pred_img_chained: torch.Tensor | None = None

    # Loop over target frames. Frame 0 is only used as context.
    for k, target_file in enumerate(npz_list[1:], start=1):
        pviIDX = target_file.split("idx")[1]
        pviIDX = int(pviIDX.split(".")[0])

        simtime = scalarPVIarray(target_file, "sim_time")
        Rcoord = singlePVIarray(npzfile=target_file, FIELD="Rcoord")
        Zcoord = singlePVIarray(npzfile=target_file, FIELD="Zcoord")

        true_img, present_vars, channel_map = prepare_flyerplate_eval_sample(
            target_file,
            csv_filepath,
            thermodynamic_variables="density",
            kinematic_variables="velocity",
        )
        in_vars = channel_map
        out_vars = channel_map
        true_rho = density_sum(true_img, present_vars)

        if mode == "single":
            input_file = npz_list[k - 1]
            input_img, input_present_vars, input_channel_map = (
                prepare_flyerplate_eval_sample(
                    input_file,
                    csv_filepath,
                )
            )
            input_time = scalarPVIarray(input_file, "sim_time")
            Dt = torch.tensor([simtime - input_time], dtype=torch.float32)
            pred_img, pred_rho = loderunner_inference(
                model,
                input_img,
                input_channel_map,
                input_channel_map,
                Dt,
                input_present_vars,
            )

        elif mode == "timestep":
            Dt = torch.tensor([simtime - initial_time], dtype=torch.float32)
            pred_img, pred_rho = loderunner_inference(
                model,
                initial_img,
                initial_channel_map,
                initial_channel_map,
                Dt,
                initial_present_vars,
            )

        else:  # chained
            if k == 1:
                prev_time = initial_time
            else:
                prev_time = scalarPVIarray(npz_list[k - 1], "sim_time")

            Dt = torch.tensor([simtime - prev_time], dtype=torch.float32)

            if pred_img_chained is None:
                chained_input = initial_img
            else:
                chained_input = pred_img_chained

            pred_img, pred_rho = loderunner_inference(
                model,
                chained_input,
                initial_channel_map,
                initial_channel_map,
                Dt,
                initial_present_vars,
            )
            pred_img_chained = torch.squeeze(pred_img, 0)

        # Plot Truth/Prediction/Discrepancy panel.
        fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig1.suptitle(f"T={float(simtime):.2f}us", fontsize=18)

        discrepancy = np.abs(true_rho - pred_rho)

        true_rho_plot = np.rot90(true_rho, k=3)
        pred_rho_plot = np.rot90(pred_rho, k=3)
        discrepancy_plot = np.rot90(discrepancy, k=3)

        plot_extent = [Zcoord.max(), Zcoord.min(), 0.0, Rcoord.max()]

        img1 = ax1.imshow(
            true_rho_plot,
            aspect="equal",
            extent=plot_extent,
            origin="lower",
            cmap="jet",
            vmin=true_rho.min(),
            vmax=true_rho.max(),
        )
        ax1.set_xlabel("Z-axis", fontsize=16)
        ax1.set_ylabel("R-axis", fontsize=16)
        ax1.set_title("True", fontsize=18)

        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="10%", pad=0.1)
        fig1.colorbar(img1, cax=cax1).set_label("Density (g/cc)", fontsize=14)

        img2 = ax2.imshow(
            pred_rho_plot,
            aspect="equal",
            extent=plot_extent,
            origin="lower",
            cmap="jet",
            vmin=true_rho.min(),
            vmax=true_rho.max(),
        )
        ax2.set_xlabel("Z-axis", fontsize=16)
        ax2.set_title("Predicted", fontsize=18)
        ax2.tick_params(axis="y", which="both", left=False, labelleft=False)

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="10%", pad=0.1)
        fig1.colorbar(img2, cax=cax2).set_label("Density (g/cc)", fontsize=14)

        img3 = ax3.imshow(
            discrepancy_plot,
            aspect="equal",
            extent=plot_extent,
            origin="lower",
            cmap="hot",
            vmin=discrepancy.min(),
            vmax=0.3 * discrepancy.max(),
        )
        ax3.set_xlabel("Z-axis", fontsize=16)
        ax3.set_title("Discrepancy", fontsize=18)
        ax3.tick_params(axis="y", which="both", left=False, labelleft=False)

        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="10%", pad=0.1)
        fig1.colorbar(img3, cax=cax3).set_label("Discrepancy", fontsize=14)

        zmin, zmax = -1.0, 2.0
        rmin, rmax = 0.0, 3.0

        for ax in (ax1, ax2, ax3):
            ax.set_xlim(zmin, zmax)
            ax.set_ylim(rmin, rmax)

        # Save images
        fig1.savefig(
            os.path.join(outdir, f"loderunner_prediction_{runID}_idx{pviIDX:05d}.png"),
            bbox_inches="tight",
        )
        plt.close()

        if num_pngs is not None and k >= num_pngs:
            break
