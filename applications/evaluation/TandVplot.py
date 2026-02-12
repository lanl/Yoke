"""This script reads `Yoke` record files and plots the network learning curves.

The data files created from `Yoke` training harnesses are CSV files which are
read with Pandas and only contain metric evaluation information. This script
associates plots with the *training* and *validation* sets of those files.

NOTE: For each network evaluation this script should be submitted using a
job-submission script.

"""

import os
import argparse
import glob
import numpy as np
import pandas as pd

# Imports for plotting
# To view possible matplotlib backends use
# >>> import matplotlib
# >>> bklist = matplotlib.rcsetup.interactive_bk
# >>> print(bklist)
import matplotlib
import matplotlib.pyplot as plt

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
descr_str = "Plot learning curves."
parser = argparse.ArgumentParser(
    prog="Training and validation curves.",
    description=descr_str,
    fromfile_prefix_chars="@",
)

parser.add_argument(
    "--basedir",
    action="store",
    type=str,
    default="./study_directory",
    help="Directory to look for studies.",
)

parser.add_argument(
    "--IDX",
    "-I",
    action="store",
    type=int,
    default=0,
    help="Index of study to plot curves for.",
)

parser.add_argument(
    "--Nsamps_per_trn_pt",
    "-Nt",
    action="store",
    type=int,
    default=2012,
    help="Number of samples per training loss plot point.",
)

parser.add_argument(
    "--Nsamps_per_val_pt",
    "-Nv",
    action="store",
    type=int,
    default=250,
    help="Number of samples per validation loss plot point.",
)

parser.add_argument(
    "--scatter",
    "-s",
    action="store_true",
    help="Plot each loss value as a scatter",
)

parser.add_argument(
    "--ylim",
    "-Y",
    action="store",
    type=float,
    default=1.0,
    help="Upper y-axis limit for plot.",
)

parser.add_argument(
    "--inprogress",
    "-P",
    action="store_true",
    help="If run is still training throw out last training CSV.",
)

parser.add_argument(
    "--savedir",
    action="store",
    type=str,
    default="./",
    help="Directory for saving images.",
)

parser.add_argument("--savefig", "-S", action="store_true", help="Flag to save figures.")

args_ns = parser.parse_args()

basedir = args_ns.basedir
IDX = args_ns.IDX
YLIM = args_ns.ylim
scatter = args_ns.scatter
INPROGRESS = args_ns.inprogress
savedir = args_ns.savedir
SAVEFIG = args_ns.savefig

# Number of batches to include in each plot point
Nsamps_per_trn_pt = args_ns.Nsamps_per_trn_pt
Nsamps_per_val_pt = args_ns.Nsamps_per_val_pt

# Read the data using Pandas
# Of the form: training_study009_epoch0100.csv
trn_glob = f"{basedir}/study_{IDX:03d}/training_study{IDX:03d}_epoch*.csv"
trn_csv_list = sorted(glob.glob(trn_glob))

# Throw out most recent training CSV if still in progress
if INPROGRESS:
    trn_csv_list.pop()

trn_file_epochs = []
for Tcsv in trn_csv_list:
    epoch = Tcsv.split("epoch")[1]
    epoch = int(epoch.split(".")[0])
    trn_file_epochs.append(epoch)

# Of the form: validation_study009_epoch0010.csv
val_glob = f"{basedir}/study_{IDX:03d}/validation_study{IDX:03d}_epoch*.csv"
val_csv_list = sorted(glob.glob(val_glob))
val_file_epochs = []
for Vcsv in val_csv_list:
    epoch = Vcsv.split("epoch")[1]
    epoch = int(epoch.split(".")[0])
    val_file_epochs.append(epoch)

# # Retrieve indexes
# trn_idxlist = trn_DF.index.values
# val_idxlist = val_DF.index.values

trn_csv_list = ["train.csv"]
val_csv_list = ["valcsv"]

# Plot loss for training over all steps and epochs
fig1 = plt.figure(num=1, figsize=(6, 6))
ax = plt.gca()
# Nepochs = min(trn_DF['Epoch'].max(), val_DF['Epoch'].max())
# startIDX = 0
vIDX = 0
startIDX = 0
for tIDX, Tcsv in enumerate(trn_csv_list):
    ptIDX_list = []
    trn_DF = pd.read_csv(
        Tcsv, sep=", ", header=None, names=["Epoch", "Batch", "Loss"], engine="python"
    )
    all_losses = trn_DF.loc[:, "Loss"].values
    all_idxs = trn_DF.loc[:, "Batch"].values
    scaled_trn_idxs = np.array(all_idxs) / Nsamps_per_trn_pt

    # print('Train IDX:', trn_file_epochs[tIDX])
    try:
        trn_epoch_loss = trn_DF.loc[:, "Loss"].values.reshape((Nsamps_per_trn_pt, -1))

    except ValueError:
        print("# of loss samples from training not divisible by Nsamps_per_trn_pt")
        print("Remainder of loss samples at the end will be excluded")
        trn_DF = trn_DF[: -(len(trn_DF) % Nsamps_per_trn_pt)]
        trn_epoch_loss = trn_DF.loc[:, "Loss"].values.reshape((Nsamps_per_trn_pt, -1))

    # print('Loss array shape:', trn_epoch_loss.shape)
    trn_positions = np.arange(trn_epoch_loss.shape[1])
    trn_qnts = np.quantile(trn_epoch_loss, [0.025, 0.5, 0.975], axis=0)
    # print('Quantiles shape:', trn_qnts.shape)

    if tIDX == 0:
        if scatter:
            plt.scatter(scaled_trn_idxs, all_losses, alpha=0.2)
        plt.plot(startIDX + trn_positions, trn_qnts[0, :], ":b")
        plt.plot(startIDX + trn_positions, trn_qnts[1, :], "-b", label="Training")
        plt.plot(startIDX + trn_positions, trn_qnts[2, :], ":b")
    else:
        plt.plot(startIDX + trn_positions, trn_qnts[0, :], ":b")
        plt.plot(startIDX + trn_positions, trn_qnts[1, :], "-b")
        plt.plot(startIDX + trn_positions, trn_qnts[2, :], ":b")

    startIDX += trn_positions[-1]

    # print('trn_file_epochs length:', len(trn_file_epochs))
    # print('val_file_epochs length:', len(val_file_epochs))
    # print('tIDX:', tIDX, 'vIDX:', vIDX)

    if vIDX < len(val_file_epochs):
        if trn_file_epochs[tIDX] == val_file_epochs[vIDX]:
            val_DF = pd.read_csv(
                val_csv_list[vIDX],
                sep=", ",
                header=None,
                names=["Epoch", "Batch", "Loss"],
                engine="python",
            )
            # print('Validation IDX:', val_file_epochs[vIDX])
            all_val_losses = val_DF.loc[:, "Loss"].values
            all_val_idxs = np.array(val_DF.loc[:, "Batch"].values)
            scaled_val_idxs = all_val_idxs / Nsamps_per_val_pt + np.array(
                scaled_trn_idxs[-1]
            )

            try:
                val_epoch_loss = val_DF.loc[:, "Loss"].values.reshape(
                    (Nsamps_per_val_pt, -1)
                )

            except ValueError:
                print(
                    "# of loss samples from validation not \
                      divisible by Nsamps_per_val_pt"
                )
                print("Remainder of loss samples at the end will be excluded")
                val_DF = val_DF[: -(len(val_DF) % Nsamps_per_val_pt)]
                val_epoch_loss = val_DF.loc[:, "Loss"].values.reshape(
                    (Nsamps_per_val_pt, -1)
                )

            # print('Loss array shape:', val_epoch_loss.shape)
            val_positions = np.arange(val_epoch_loss.shape[1])
            val_qnts = np.quantile(val_epoch_loss, [0.025, 0.5, 0.975], axis=0)
            # print('Quantiles shape:', val_qnts.shape)

            if vIDX == 0:
                if scatter:
                    plt.scatter(scaled_val_idxs, all_val_losses, alpha=0.2)
                plt.plot(startIDX + val_positions, val_qnts[0, :], ":r")
                plt.plot(
                    startIDX + val_positions, val_qnts[1, :], "-r", label="Validation"
                )
                plt.plot(startIDX + val_positions, val_qnts[2, :], ":r")
            else:
                plt.plot(startIDX + val_positions, val_qnts[0, :], ":r")
                plt.plot(startIDX + val_positions, val_qnts[1, :], "-r")
                plt.plot(startIDX + val_positions, val_qnts[2, :], ":r")

            vIDX += 1
            startIDX += val_positions[-1]

# Make legend
plt.legend(fontsize=16)

# No xlim
ax.set_ylim(0.0, YLIM)

# Set axis labels
ax.set_ylabel("Loss", fontsize=16)
ax.set_xlabel("Evaluation Index", fontsize=16)

# Save or plot images
if SAVEFIG:
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    plt.figure(fig1.number)
    filenameA = f"{savedir}/study{IDX:03d}_TandV_curve.png"
    plt.savefig(filenameA, bbox_inches="tight")
else:
    plt.show()
