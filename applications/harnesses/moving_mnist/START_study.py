"""Script to start training a study."""

####################################
# Packages
####################################
import os
import shutil
import argparse
import pandas as pd
import subprocess
from pathlib import Path

from yoke.helpers import cli, strings, create_slurm_files


####################################
# Process Hyperparameters
####################################
parser = argparse.ArgumentParser(
    prog="HARNESS START", description="Starts execution of training harness"
)
parser = cli.add_default_args(parser)
args = parser.parse_args()

training_START_input = "./training_START.input"
training_input_tmpl = "./training_input.tmpl"

if args.submissionType.lower() == "slurm":
    training_START_slurm = "./training_START.slurm"
    training_slurm_tmpl = "./training_slurm.tmpl"

elif args.submissionType.lower() == "flux":
    training_START_flux = "./training_START.flux"
    training_flux_tmpl = "./training_flux.tmpl"

elif args.submissionType.lower() == "shell":
    training_START_shell = "./training_START.sh"
    training_shell_tmpl = "./training_shell.tmpl"

elif args.submissionType.lower() == "batch":
    training_START_batch = "./training_START.bat"
    training_batch_tmpl = "./training_batch.tmpl"

training_json = "./slurm_config.json"

slurm_tmpl_data = None
if os.path.exists(training_json):
    slrm_obj = create_slurm_files.MkSlurm(config_path=training_json)
    slurm_tmpl_data = slrm_obj.generateSlurm()

# List of files to copy
with open(args.cpFile) as cp_text_file:
    cp_file_list = [line.strip() for line in cp_text_file]

# Process Hyperparameters File
studyDF = pd.read_csv(
    args.csv, sep=",", header=0, index_col=0, comment="#", engine="python"
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
    with open(training_input_tmpl) as f:
        training_input_data = f.read()

    training_input_data = strings.replace_keys(study, training_input_data)
    training_input_filepath = os.path.join(studydirname, "training_input.tmpl")

    with open(training_input_filepath, "w") as f:
        f.write(training_input_data)

    # Make new training_START.input file
    with open(training_START_input) as f:
        START_input_data = f.read()

    START_input_data = strings.replace_keys(study, START_input_data)
    START_input_name = "study{:03d}_START.input".format(study["studyIDX"])
    START_input_filepath = os.path.join(studydirname, START_input_name)

    with open(START_input_filepath, "w") as f:
        f.write(START_input_data)

    # Create submission scripts based on submission system.
    if args.submissionType.lower() == "slurm":
        # Make new training_slurm.tmpl file
        if slurm_tmpl_data is None:
            with open(training_slurm_tmpl) as f:
                training_slurm_data = f.read()

        else:
            training_slurm_data = slurm_tmpl_data

        training_slurm_data = strings.replace_keys(study, training_slurm_data)
        training_slurm_filepath = os.path.join(studydirname, "training_slurm.tmpl")

        with open(training_slurm_filepath, "w") as f:
            f.write(training_slurm_data)

        if slurm_tmpl_data is None:
            # Make a new training_START.slurm file
            with open(training_START_slurm) as f:
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

    elif args.submissionType.lower() == "flux":
        # Make new training_flux.tmpl file
        with open(training_flux_tmpl) as f:
            training_flux_data = f.read()

        training_flux_data = strings.replace_keys(study, training_flux_data)
        training_flux_filepath = os.path.join(studydirname, "training_flux.tmpl")

        with open(training_flux_filepath, "w") as f:
            f.write(training_flux_data)

        # Make a new training_START.slurm file
        with open(training_START_flux) as f:
            START_flux_data = f.read()

        START_flux_data = strings.replace_keys(study, START_flux_data)

        START_flux_name = "study{:03d}_START.flux".format(study["studyIDX"])
        START_flux_filepath = os.path.join(studydirname, START_flux_name)

        with open(START_flux_filepath, "w") as f:
            f.write(START_flux_data)

    elif args.submissionType.lower() == "shell":
        # Make new training_shell.tmpl file
        with open(training_shell_tmpl) as f:
            training_shell_data = f.read()

        training_shell_data = strings.replace_keys(study, training_shell_data)
        training_shell_filepath = os.path.join(studydirname, "training_shell.tmpl")

        with open(training_shell_filepath, "w") as f:
            f.write(training_shell_data)

        # Make a new training_START.sh file
        with open(training_START_shell) as f:
            START_shell_data = f.read()

        START_shell_data = strings.replace_keys(study, START_shell_data)

        START_shell_name = "study{:03d}_START.sh".format(study["studyIDX"])
        START_shell_filepath = os.path.join(studydirname, START_shell_name)

        with open(START_shell_filepath, "w") as f:
            f.write(START_shell_data)
        os.system(f"chmod +x {START_shell_filepath}")

    elif args.submissionType.lower() == "batch":
        # Make new training_batch.tmpl file
        with open(training_batch_tmpl) as f:
            training_batch_data = f.read()

        training_batch_data = strings.replace_keys(study, training_batch_data)
        training_batch_filepath = os.path.join(studydirname, "training_batch.tmpl")

        with open(training_batch_filepath, "w") as f:
            f.write(training_batch_data)

        # Make a new training_START.bat file
        with open(training_START_batch) as f:
            START_batch_data = f.read()

        START_batch_data = strings.replace_keys(study, START_batch_data)

        START_batch_name = "study{:03d}_START.bat".format(study["studyIDX"])
        START_batch_filepath = os.path.join(studydirname, START_batch_name)

        with open(START_batch_filepath, "w") as f:
            f.write(START_batch_data)

    # Copy files to study directory from list
    for f in cp_file_list:
        shutil.copy(f, studydirname)

    # Submit a job with the appropriate submission type
    if args.submissionType.lower() == "slurm":
        slurm_cmd = (
            f"cd {studydirname}; sbatch {START_slurm_name}; "
            f"cd {os.path.dirname(__file__)}"
        )
        os.system(slurm_cmd)

    elif args.submissionType.lower() == "flux":
        flux_cmd = (
            f"cd {studydirname}; flux batch {START_flux_name}; "
            f"cd {os.path.dirname(__file__)}"
        )
        os.system(flux_cmd)

    elif args.submissionType.lower() == "shell":
        shell_cmd = (
            f"cd {studydirname}; ./{START_shell_name}; "
            f"cd {os.path.dirname(__file__)}"
        )
        os.system(shell_cmd)

    elif args.submissionType.lower() == "batch":
        harness_dir = Path(args.csv).parent.resolve()
        wrapper = harness_dir / START_batch_name
        if not wrapper.exists():
            raise FileNotFoundError(f"Cannot find batch wrapper at {wrapper!r}")

        subprocess.run(
            ["cmd.exe", "/c", str(wrapper), study["train_script"], START_input_name],
            cwd=studydirname,
            check=True,
        )

    else:
        raise ValueError(f"Unknown submission type: {args.submissionType!r}")
