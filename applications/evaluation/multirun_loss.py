"""Script to plot multiple network learning curves."""

import argparse
import glob
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Use LaTeX font
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
    "--idx_range",
    nargs="+",
    action="store",
    type=int,
    default=None,
    help="Index range of studies to plot curves for.",
)

parser.add_argument(
    "--version_range",
    nargs="+",
    action="store",
    type=int,
    default=None,
    help="Index range of lightning log versions to use for each run.",
)

parser.add_argument(
    "--savedir",
    action="store",
    type=str,
    default="./",
    help="Directory for saving images.",
)

parser.add_argument(
    "--savefig", "-S", action="store_true", help="Flag to save figures."
)

args = parser.parse_args()


# Search for results files of the form version_*/metrics.csv (For Lightning training).
csv_list = glob.glob(
    os.path.join(args.basedir, "study_*", "lightning_logs", "version_*", "metrics.csv")
)

# Filter files.
study_re = r"study_(?P<ind>\d+)"
study_inds = np.array([int(re.search(study_re, f)["ind"]) for f in csv_list])
version_re = r"version_(?P<ind>\d+)"
version_inds = np.array([int(re.search(version_re, f)["ind"]) for f in csv_list])
if args.idx_range is None:
    idx_range = [np.min(study_inds), np.max(study_inds)]
else:
    idx_range = args.idx_range
keep_inds = (study_inds >= idx_range[0]) & (study_inds <= idx_range[1])
csv_list = np.array(csv_list)[keep_inds]
study_inds = study_inds[keep_inds]
version_inds = version_inds[keep_inds]

if args.version_range is None:
    version_range = [np.min(version_inds), np.max(version_inds)]
else:
    version_range = args.version_range
keep_inds = (version_inds >= version_range[0]) & (version_inds <= version_range[1])
csv_list = np.array(csv_list)[keep_inds]
study_inds = study_inds[keep_inds]
version_inds = version_inds[keep_inds]

# Plot losses for each study.
fig, ax = plt.subplots()
studies = np.arange(idx_range[0], idx_range[1] + 1)
study_color = [
    matplotlib.colormaps["viridis"](v) for v in np.linspace(0, 1, len(studies))
]
trn_plt_properties = {
    "marker": ".",
    "markeredgewidth": 0.5,
    "linestyle": "",
    "alpha": 0.3,
}
val_plt_properties = {
    "marker": "x",
    "markeredgewidth": 2.0,
    "linestyle": "",
}
for n, study_idx in enumerate(studies):
    # Isolate current training files.
    current_study = study_inds == study_idx
    current_files = csv_list[current_study]

    # Load training losses.
    train_loss = []
    val_loss = []
    epoch = []
    for f in current_files:
        df = pd.read_csv(
            f,
            sep=",",
            engine="python",
        )
        if len(df) > 0:
            train_loss.append(df["train_loss"].to_numpy())
            val_loss.append(df["val_loss"].to_numpy())
            epoch.append(df["epoch"].to_numpy())

    # Plot losses.
    epoch = np.concatenate(epoch)
    train_loss = np.concatenate(train_loss)
    val_loss = np.concatenate(val_loss)
    ax.plot(
        epoch,
        train_loss,
        color=study_color[n],
        **trn_plt_properties,
        label=f"Training: study {study_idx}",
    )
    ax.plot(
        epoch,
        val_loss,
        color=study_color[n],
        **val_plt_properties,
        label=f"Validation: study {study_idx}",
    )

# Decorate plot
plt.legend()
ax.set_ylabel("Loss")
ax.set_xlabel("Epoch")
ax.set_yscale("log")

# Save or plot images
if args.savefig:
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    filename = os.path.join(
        args.savedir, f"studies_{args.idx_range[0]}_to_{args.idx_range[1]}_loss.png"
    )
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close(fig)
else:
    plt.show()
