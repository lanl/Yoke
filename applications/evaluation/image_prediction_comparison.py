"""Compares *true* and *predicted* images for a trained network.

For networks which take in scalars and output images this script plots a
comparison of the true and predicted image along with the inputs.

"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from yoke.models.surrogateCNNmodules import tCNNsurrogate
from yoke.datasets.lsc_dataset import LSCnpz2key, LSC_cntr2rho_DataSet
from yoke.utils.checkpointing import load_model_and_optimizer_hdf5

# Imports for plotting
# To view possible matplotlib backends use
# >>> import matplotlib
# >>> bklist = matplotlib.rcsetup.interactive_bk
# >>> print(bklist)
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# matplotlib.use('MacOSX')
# matplotlib.use('pdf')
# Get rid of type 3 fonts in figures
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# Ensure LaTeX font
font = {"family": "serif"}
plt.rc("font", **font)
plt.rcParams["figure.figsize"] = (6, 6)


###################################################################
# Define command line argument parser
descr_str = "Compare True and Predicted Images."
parser = argparse.ArgumentParser(
    prog="Image Comparison.", description=descr_str, fromfile_prefix_chars="@"
)

parser.add_argument(
    "--FILELIST_DIR",
    action="store",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "../filelists/"),
    help="Directory where filelists are located.",
)

parser.add_argument(
    "--eval_filelist",
    action="store",
    type=str,
    default="lsc240420_train_sample.txt",
    help="Name of filelist file to choose an evaluation sample from.",
)

parser.add_argument(
    "--LSC_DESIGN_DIR",
    action="store",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "../../data_examples/"),
    help="Directory in which LSC design.txt file lives.",
)

parser.add_argument(
    "--design_file",
    action="store",
    type=str,
    default="design_lsc240420_SAMPLE.csv",
    help=".csv file that contains the truth values for data files",
)

parser.add_argument(
    "--LSC_NPZ_DIR",
    action="store",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "../../data_examples/lsc240420/"),
    help="Directory in which LSC *.npz files lives.",
)

parser.add_argument(
    "--checkpoint",
    action="store",
    type=str,
    default="./study001_modelState_epoch0070.hdf5",
    help="Name of HDF5 model checkpoint to evaluate output for.",
)

parser.add_argument(
    "--sampIDX",
    action="store",
    type=int,
    default=689,
    help="Index within evaluation file list to generate plot for.",
)

parser.add_argument(
    "--dscale",
    action="store",
    type=float,
    default=0.3,
    help="Scaling factor for the maximum value of discrepancy.",
)

parser.add_argument(
    "--savedir",
    action="store",
    type=str,
    default="./",
    help="Directory for saving images.",
)

parser.add_argument("--savefig", "-S", action="store_true", help="Flag to save figures.")

args = parser.parse_args()

checkpoint = args.checkpoint

# Data Paths
design_file = os.path.abspath(args.LSC_DESIGN_DIR + args.design_file)
eval_filelist = args.FILELIST_DIR + args.eval_filelist

# Additional inpu variables
sampIDX = args.sampIDX
dscale = args.dscale
savedir = args.savedir
SAVEFIG = args.savefig

# Hardcode model hyperparameters for now.
#
# NOTE: This should be handled in a MUCH more elegant manner!!!
# kernel = [3, 3]
# featureList = [512, 512, 512, 512, 256, 128, 64, 32]
# linearFeatures = [4, 4]

# model = jekelCNNsurrogate(input_size=29,
#                           linear_features=linearFeatures,
#                           kernel=kernel,
#                           nfeature_list=featureList,
#                           output_image_size=(1120, 800),
#                           act_layer=nn.GELU)

# Set up for study001, study010, study11 respectively...
# For LSC usually have been using test IDX 689 or 3914
model = tCNNsurrogate(
    input_size=29,
    # linear_features=(7, 5, 256),
    # linear_features=(7, 5, 512),
    linear_features=(7, 5, 768),
    initial_tconv_kernel=(5, 5),
    initial_tconv_stride=(5, 5),
    initial_tconv_padding=(0, 0),
    initial_tconv_outpadding=(0, 0),
    initial_tconv_dilation=(1, 1),
    kernel=(3, 3),
    # nfeature_list=[256, 128, 64, 32, 16],
    # nfeature_list=[512, 512, 256, 128, 64],
    nfeature_list=[768, 512, 256, 128, 64],
    output_image_size=(1120, 800),
    act_layer=nn.GELU,
)


#############################################
# Initialize Optimizer
#############################################
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5.0e-03,  # Must be set to load checkpoint but irrelevant for this script.
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.01,
)

##############
# Load Model
##############
checkpoint_epoch = load_model_and_optimizer_hdf5(model, optimizer, checkpoint)

################
# Load dataset
################
eval_dataset = LSC_cntr2rho_DataSet(args.LSC_NPZ_DIR, eval_filelist, design_file)

# Load single image and parameters pair
sim_params, true_image = eval_dataset.__getitem__(sampIDX)

# Read filelist
with open(eval_filelist) as f:
    eval_filenames = [line.rstrip() for line in f]

# Get the input filename evaluated
eval_filename = eval_filenames[sampIDX]

# Get the simulation key associated with evaluation
eval_key = LSCnpz2key(args.LSC_NPZ_DIR + eval_filename)
print("Evaluation file key:", eval_key)

# Evaluate model
model.eval()
pred_image = model(sim_params)

# Reshape for plotting
sim_params = sim_params.numpy()
true_image = np.squeeze(true_image.numpy())
# Predictions from network must be detached from gradients in order to be
# written to numpy arrays.
pred_image = np.squeeze(pred_image.detach().numpy())
# print('Shape of image prediction:', pred_image.shape)

# Plot Truth/Prediction/Discrepancy panel.
fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
fig1.suptitle(f"Time={sim_params[-1]:.3f}us", fontsize=18)
img1 = ax1.imshow(
    true_image,
    aspect="equal",
    origin="lower",
    cmap="jet",
    vmin=true_image.min(),
    vmax=true_image.max(),
)
ax1.set_ylabel("Z-axis", fontsize=16)
ax1.set_xlabel("R-axis", fontsize=16)
ax1.set_title("True", fontsize=18)

# divider1 = make_axes_locatable(ax1)
# cax1 = divider1.append_axes('right', size='10%', pad=0.1)
# fig1.colorbar(img1,
#               cax=cax1).set_label('Density',
#                                   fontsize=14)

img2 = ax2.imshow(
    pred_image,
    aspect="equal",
    origin="lower",
    cmap="jet",
    vmin=true_image.min(),
    vmax=true_image.max(),
)
ax2.set_title("Predicted", fontsize=18)
ax2.tick_params(axis="y", which="both", left=False, labelleft=False)

divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="10%", pad=0.1)
fig1.colorbar(img2, cax=cax2).set_label("Density (g/cc)", fontsize=14)

discrepancy = np.abs(true_image - pred_image)
img3 = ax3.imshow(
    discrepancy,
    aspect="equal",
    origin="lower",
    cmap="hot",
    vmin=discrepancy.min(),
    vmax=dscale * discrepancy.max(),
)
ax3.set_title("Discrepancy", fontsize=18)
ax3.tick_params(axis="y", which="both", left=False, labelleft=False)

divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes("right", size="10%", pad=0.1)
fig1.colorbar(img3, cax=cax3).set_label("Discrepancy", fontsize=14)

# Save or plot images
if SAVEFIG:
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    plt.figure(fig1.number)
    filenameA = f"{savedir}/{eval_key}_image_compare.png"
    plt.savefig(filenameA, bbox_inches="tight")
else:
    plt.show()
