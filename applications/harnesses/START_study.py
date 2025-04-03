"""Yoke CLI: Launch a training or evaluation study from a harness directory.

This CLI tool prepares and configures job submission files for running a study
using a provided harness configuration (templates, scripts, and hyperparams).

Usage:
    yoke-start-study [--cpFile cp_files.txt] [--csv hyperparams.csv] [--rundir runs]

Expected Files in Harness Directory:
    - training_START.input
    - training_START.slurm
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


def copy_required_files(cp_file_path: str, studydirname: str) -> None:
    """Copy files listed in text file to study directory."""
    with open(cp_file_path, "r") as cp_file:
        cp_files = [line.strip() for line in cp_file if line.strip()]
    
    # Copy files to study directory from list
    for file in cp_files:
        shutil.copy(file, studydirname)
        print(f"Copied: {file}")


def generate_slurm_if_configured(json_path: str):
    """Generate SLURM from JSON."""
    if os.path.exists(json_path):
        slrm_obj = create_slurm_files.MkSlurm(config_path=json_path)
        return slrm_obj.generateSlurm()
    return None


def create_study_list(csvfile: str) -> list[dict]:
    """Create list of studies from CSV."""
    # Process Hyperparameters File
    studyDF = pd.read_csv(
        csvfile, sep=",", header=0, index_col=0, comment="#", engine="python"
    )
    varnames = studyDF.columns.values
    idxlist = studyDF.index.values

    # Save Hyperparameters to list of dictionaries
    studylist = []
    for i in idxlist:
        studydict = {}
        studydict["studyIDX"] = int(i)

        for var in varnames:
            studydict[var] = studyDF.loc[i, var]

        studylist.append(studydict)

    return studylist


def main():
    parser = argparse.ArgumentParser(
        prog="yoke-start-study",
        description="Starts execution of a Yoke training harness study.",
    )
    parser = cli.add_default_args(parser)
    args = parser.parse_args()

    # Define expected template/config paths
    tmpl_paths = {
        "input_template": "training_input.tmpl",
        "slurm_template": "training_slurm.tmpl",
        "input_START": "training_START.input",
        "slurm_START": "training_START.slurm",
        "slurm_json": "slurm_config.json"
    }

    # Read json to SLURM
    slurm_tmpl_data = generate_slurm_if_configured(tmpl_paths["slurm_json"])

    # Read CSV to get list of studies.
    studylist = create_study_list(args.csv)
    
    ####################################
    # Run Studies
    ####################################
    # Iterate Through Dictionary List to Run Studies
    for k, study in enumerate(studylist):
        # Make Study Directory
        studydirname = args.rundir + "/study_{:03d}".format(study["studyIDX"])

        if not os.path.exists(studydirname):
            os.makedirs(studydirname)

        # Make new training_input.tmpl file
        with open(tmpl_paths["input_template"]) as f:
            training_input_data = f.read()

        training_input_data = strings.replace_keys(study, training_input_data)
        training_input_filepath = os.path.join(studydirname, "training_input.tmpl")

        with open(training_input_filepath, "w") as f:
            f.write(training_input_data)

        # Make new training_slurm.tmpl file
        if slurm_tmpl_data is None:
            with open(tmpl_paths["slurm_template"]) as f:
                training_slurm_data = f.read()
        else:
            training_slurm_data = slurm_tmpl_data

        training_slurm_data = strings.replace_keys(study, training_slurm_data)
        training_slurm_filepath = os.path.join(studydirname, "training_slurm.tmpl")

        with open(training_slurm_filepath, "w") as f:
            f.write(training_slurm_data)

        # Make new training_START.input file
        with open(tmpl_paths["input_START"]) as f:
            START_input_data = f.read()

        START_input_data = strings.replace_keys(study, START_input_data)
        START_input_name = "study{:03d}_START.input".format(study["studyIDX"])
        START_input_filepath = os.path.join(studydirname, START_input_name)

        with open(START_input_filepath, "w") as f:
            f.write(START_input_data)

        if slurm_tmpl_data is None:
            # Make a new training_START.slurm file
            with open(tmpl_paths["slurm_START"]) as f:
                START_slurm_data = f.read()

        if slurm_tmpl_data is not None:
            START_slurm_data = strings.replace_keys(study, slurm_tmpl_data).replace(
                "<epochIDX>", "0001"
            )
        else:
            START_slurm_data = strings.replace_keys(study, START_slurm_data)

        START_slurm_name = "study{:03d}_START.slurm".format(study["studyIDX"])
        START_slurm_filepath = os.path.join(studydirname, START_slurm_name)

        with open(START_slurm_filepath, "w") as f:
            f.write(START_slurm_data)

        # Copy training scripts.
        copy_required_files(args.cpFile, studydirname)

        # Submit Job
        submit_str = (
            f"cd {studydirname}; "
            f"sbatch {START_slurm_name}; "
            f"cd {os.path.dirname(__file__)}"
        )
        os.system(submit_str)


if __name__ == "__main__":
    main()
