"""Script to start training a study."""

####################################
# Packages
####################################
import os
import shutil
import argparse
import pandas as pd

from yoke.helpers import cli, strings, create_slurm_files


####################################
# Process Hyperparameters
####################################
parser = argparse.ArgumentParser(
    prog="HARNESS START", description="Starts execution of training harness"
)
parser = cli.add_default_args(parser)
args = parser.parse_args()

training_input_tmpl = "./training_input.tmpl"
training_slurm_tmpl = "./training_slurm.tmpl"
training_START_input = "./training_START.input"
training_START_slurm = "./training_START.slurm"
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

    # Make new training_START.input file
    with open(training_START_input) as f:
        START_input_data = f.read()

    START_input_data = strings.replace_keys(study, START_input_data)
    START_input_name = "study{:03d}_START.input".format(study["studyIDX"])
    START_input_filepath = os.path.join(studydirname, START_input_name)

    with open(START_input_filepath, "w") as f:
        f.write(START_input_data)

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

    # Copy files to study directory from list
    for f in cp_file_list:
        shutil.copy(f, studydirname)

    # Submit Job
    os.system(
        f"cd {studydirname}; sbatch {START_slurm_name}; cd {os.path.dirname(__file__)}"
    )
