"""Script to generate GIF animations from model output checkpoints.

This script finds the HDF5 checkpoint with the highest epoch for each
study directory (unless skipped), runs a visualization script to generate
PNGs, and then converts them into a GIF using ImageMagick's convert tool.
"""

import re
import subprocess
from pathlib import Path
import argparse

# Ex Usage 1: python3 lsc_loderunner_create_gif.py
#                   --runs-dir <path to runs dir that has study dirs>
#                   --npz-dir <path to the dir that has npz files>
#                   --skip-list 011,017,018
#                   --run-id <run id num>
#                   --embed-dim <embed dim value>
# OR
# Ex Usage 2: python3 lsc_loderunner_create_gif.py @lsc_loderunner_create_gif.input

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")


parser.add_argument(
    "--runs-dir",
    type=str,
    default="../harnesses/chicoma_lsc_loderunner/runs",
    help="Path to the directory containing the study_* folders",
)

parser.add_argument(
    "--npz-dir",
    type=str,
    default="/lustre/scratch5/exempt/artimis/mpmm/lsc240420/",
    help="Path to the directory containing NPZ input files",
)

parser.add_argument(
    "--skip-list",
    type=str,
    default="997,998,999",
    # default="011,017,018,044,053",
    help="Comma-separated list of study numbers to skip, e.g., '011,017'",
)

parser.add_argument(
    "--run-id", type=str, default="400", help="Run ID used when generating animations"
)

parser.add_argument(
    "--embed-dim",
    type=str,
    default="128",
    help="Embedding dimension used in the animation script",
)

args = parser.parse_args()
runs_dir = Path(args.runs_dir)

# Loop through all study_dir directories in runs_dir
for study_path in sorted(runs_dir.glob("study_*")):
    study_dir = study_path.name
    study_num = study_dir.split("_")[-1]
    print(f"\nNow processing {study_dir}")

    if study_num in set(args.skip_list.split(",")):
        print(f"\t***** Skipping {study_dir}. It is in the skip dir list ****")
        print("\t==========================================================")
        continue

    # Find the HDF5 file with the highest epoch
    max_epoch = -1
    hdf5_files = list(study_path.glob(f"study{study_num}_modelState_epoch*.hdf5"))
    hdf5_file_max_epoch = None

    for file in hdf5_files:
        match = re.search(r"_epoch(\d+)\.hdf5", str(file))
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                hdf5_file_max_epoch = file

    if hdf5_file_max_epoch is None:
        print(f"\tNo HDF5 files found for study_dir {study_dir}")
        continue
    else:
        print(f"\thdf5_file_max_epoch:\n\t{hdf5_file_max_epoch}")

    # Run the animation generation script
    outdir = Path(f"{runs_dir}/{study_dir}_gif")
    outdir.mkdir(parents=True, exist_ok=True)

    # Create PNG files from the true image, predicted image and discrepancy
    subprocess.run(
        [
            "python3",
            "lsc_loderunner_anime.py",
            "--checkpoint",
            str(hdf5_file_max_epoch),
            "--indir",
            args.npz_dir,
            "--outdir",
            str(outdir),
            "--runID",
            args.run_id,
            "--embed_dim",
            args.embed_dim,
        ],
        check=True,
    )

    # Convert PNG images to GIF
    png_pattern = str(outdir / "*.png")
    gif_path = outdir / f"{study_dir}.gif"

    subprocess.run(
        [
            "convert",
            "-delay",
            "20",
            "-loop",
            "0",
            *list(map(str, sorted(outdir.glob("*.png")))),
            str(gif_path),
        ],
        check=True,
    )

    # List the generated GIF
    subprocess.run(["ls", "-l", str(gif_path)])

    print(f"\tCompleted processing {study_dir}\n")
    print("\t==========================================================")
