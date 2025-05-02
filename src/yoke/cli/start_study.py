"""Yoke CLI: Launch a training or evaluation study from a harness directory.

This CLI tool prepares and configures job submission files for running a study
using a provided harness configuration (templates, scripts, and hyperparams).

Usage:
    yoke-start-study [--cpFile cp_files.txt] [--csv hyperparams.csv] [--rundir runs]

Expected Files in Harness Directory:
    - training_input.tmpl
    - training_slurm.tmpl
    - cp_files.txt
    - <study_parameters>.csv
    - <training_routine>.py
    - slurm_config.json (optional)

"""

import os
import shutil
import argparse
import pandas as pd

from yoke.helpers import cli, strings, create_slurm_files
from yoke.harnesses.base import HarnessStudy


def main():
    parser = argparse.ArgumentParser(
        prog="yoke-start-study",
        description="Starts execution of a Yoke training harness study.",
    )
    parser = cli.add_default_args(parser)
    args = parser.parse_args()

    harness = HarnessStudy(
        rundir=args.rundir, 
        template_dir=".", 
        cp_file=args.cpFile,
        dryrun=args.dryrun
        )
    study_list = harness.load_hyperparameters(args.csv)

    for study in study_list:
        harness.run_study(study)


if __name__ == "__main__":
    main()