"""Shows networks predicted image.

Program to evaluate a trained neural network mapping a set of scalar parameters
to an output image or set of images and plot the result. Eventually this will
become a GUI interface to probe trained image generation models.

"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from yoke.models.surrogateCNNmodules import jekelCNNsurrogate
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
descr_str = "Evaluate a trained model on a set of inputs."
parser = argparse.ArgumentParser(
    prog="Image Prediction Slider.", description=descr_str, fromfile_prefix_chars="@"
)

parser.add_argument(
    "--checkpoint",
    action="store",
    type=str,
    default="./study001_modelState_epoch0070.hdf5",
    help="Name of HDF5 model checkpoint to evaluate output for.",
)

parser.add_argument(
    "--savedir",
    action="store",
    type=str,
    default="./",
    help="Directory for saving images.",
)

parser.add_argument("--savefig", "-S", action="store_true", help="Flag to save figures.")

parser.add_argument(
    "--image_name",
    action="store",
    type=str,
    default="param_prediction.png",
    help="Name of image file to save.",
)

args = parser.parse_args()

checkpoint = args.checkpoint

# Additional input variables
savedir = args.savedir
SAVEFIG = args.savefig
image_file_name = args.image_name

# Hardcode model hyperparameters for now.
kernel = [3, 3]
featureList = [512, 512, 512, 512, 256, 128, 64, 32]
linearFeatures = [4, 4]
initial_learningrate = 0.0007

model = jekelCNNsurrogate(
    input_size=29,
    linear_features=linearFeatures,
    kernel=kernel,
    nfeature_list=featureList,
    output_image_size=(1120, 800),
    act_layer=nn.GELU,
)

#############################################
# Initialize Optimizer
#############################################
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=initial_learningrate,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.01,
)

##############
# Load Model
##############
checkpoint_epoch = load_model_and_optimizer_hdf5(model, optimizer, checkpoint)

###################################
# Input parameters and evaluation
###################################
input_params = np.array(
    [
        [
            4.5,
            5.0,
            5.5,
            7.0,
            8.5,
            10.0,
            10.5,
            0.5,
            0.25,
            0.25,
            0.25,
            0.25,
            0.3,
            0.3,
            0.4,
            0.35,
            0.25,
            0.25,
            0.2,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.15,
            0.15,
            0.2,
            20.0,
        ]
    ],
    dtype=np.float32,
)
input_params = torch.from_numpy(input_params)

# Evaluate model
model.eval()
pred_image = model(input_params)

# print('Shape of inputs:', input_params.shape)
# print('Prediction shape:', pred_image.shape)

# Reshape for plotting
input_params = np.squeeze(input_params.numpy())
# Predictions from network must be detached from gradients in order to be
# written to numpy arrays.
pred_image = np.squeeze(pred_image.detach().numpy())


###################################
# Plotting Section
###################################
# Plot normalized radiograph and density field for diagnostics.
fig1, ax1 = plt.subplots(1, 1, figsize=(12, 12))
fig1.suptitle(f"Time={input_params[-1]:.3f}us", fontsize=18)
img1 = ax1.imshow(
    pred_image,
    aspect="equal",
    origin="lower",
    cmap="jet",
    vmin=pred_image.min(),
    vmax=pred_image.max(),
)
ax1.set_ylabel("Z-axis", fontsize=16)
ax1.set_xlabel("R-axis", fontsize=16)
ax1.set_title("Prediction", fontsize=18)

divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="10%", pad=0.1)
fig1.colorbar(img1, cax=cax1).set_label("Density (g/cc)", fontsize=14)

# Save or plot images
if SAVEFIG:
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    plt.figure(fig1.number)
    filenameA = f"{savedir}/{image_file_name}"
    plt.savefig(filenameA, bbox_inches="tight")
else:
    plt.show()
