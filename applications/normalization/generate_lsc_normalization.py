"""Normalization for the *lsc240420* dataset.

Functions and classes to calculate normalization values for the *lsc240420*
dataset. For example, finding the average density for each *av_density* image.

These quantities are saved in an NPZ file for use during training in a Torch
dataloader.

When run as a script an NPZ of calculated parameters is generated.

"""

####################################
# Packages
####################################
import os
import argparse
import numpy as np

from yoke.datasets.lsc_dataset import LSCnpz2key, LSCcsv2bspline_pts
from yoke.datasets.lsc_dataset import LSCread_npz

NoneStr = None | str


###################################################################
# Define command line argument parser
descr_str = "Generate lsc240420 normalization."
parser = argparse.ArgumentParser(
    prog="Generate Normalization", description=descr_str, fromfile_prefix_chars="@"
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
    help="Name of filelist file to evaluate normalizations for.",
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
    help=".csv file that contains the parameter values for data files",
)

parser.add_argument(
    "--LSC_NPZ_DIR",
    action="store",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "../../data_examples/lsc240420/"),
    help="Directory in which LSC *.npz files lives.",
)

parser.add_argument(
    "--fileout",
    action="store",
    type=str,
    default="./sample_norm.npz",
    help="Name of file containing computed normalization information",
)


# Main script
args = parser.parse_args()

# Data Paths
design_file = os.path.abspath(args.LSC_DESIGN_DIR + args.design_file)
eval_files = args.FILELIST_DIR + args.eval_filelist

# Create filelist
with open(eval_files) as f:
    eval_filelist = [line.rstrip() for line in f]

Nsamp = len(eval_filelist)
print("Number of samples:", Nsamp)

avg_time_dict = {}
nsamp_time_dict = {}

# Build arrays to normalize with
for k, filepath in enumerate(eval_filelist):
    # Get the input image
    npz = np.load(args.LSC_NPZ_DIR + filepath)

    true_image = LSCread_npz(npz, "av_density")
    true_image = np.concatenate((np.fliplr(true_image), true_image), axis=1)
    nY, nX = true_image.shape
    # print('True image shape:', nY, nX)

    # Get the contours and sim_time
    sim_key = LSCnpz2key(args.LSC_NPZ_DIR + filepath)
    Bspline_nodes = LSCcsv2bspline_pts(design_file, sim_key)
    # print('Shape of Bspline node array:', Bspline_nodes.shape)

    sim_time = npz["sim_time"]
    # print('Sim. Time:', sim_time)
    round_sim_time = str(round(4.0 * sim_time) / 4.0)
    # print('Nearest 0.25us Sim. Time:', round_sim_time)

    npz.close()

    if round_sim_time in avg_time_dict.keys():
        avg_time_dict[round_sim_time] += true_image
        nsamp_time_dict[round_sim_time] += 1
    else:
        avg_time_dict[round_sim_time] = true_image
        nsamp_time_dict[round_sim_time] = 1

    # Calculate normalization quantities
    if k == 0:
        image_avg = true_image
        image_min = np.min(true_image)
        image_max = np.max(true_image)
        Bspline_avg = Bspline_nodes
        Bspline_min = Bspline_nodes
        Bspline_max = Bspline_nodes
        # print('Image_min:', image_min)
        # print('Image_max:', image_max)
    else:
        image_avg += true_image
        image_min = min(image_min, np.min(true_image))
        image_max = max(image_max, np.max(true_image))
        Bspline_avg += Bspline_nodes
        Bspline_min = np.minimum(Bspline_nodes, Bspline_min)
        Bspline_max = np.maximum(Bspline_nodes, Bspline_max)
        # print('Bspline min array:', Bspline_min)
        # print('Bspline max array:', Bspline_max)
        # print('Image_min:', image_min)
        # print('Image_max:', image_max)

    # print('============')

# Calculate averages
image_avg = image_avg / Nsamp
Bspline_avg = Bspline_avg / Nsamp

for k, v in avg_time_dict.items():
    avg_time_dict[k] = v / nsamp_time_dict[k]

# Save normalization information
np.savez(
    args.fileout,
    image_avg=image_avg,
    image_min=image_min,
    image_max=image_max,
    Bspline_avg=Bspline_avg,
    Bspline_min=Bspline_min,
    Bspline_max=Bspline_max,
    **avg_time_dict,
)
