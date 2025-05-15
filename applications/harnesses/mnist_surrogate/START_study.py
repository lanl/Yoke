"""Script to start an MNIST study (no slurm)."""

# Running studies:
# \Yoke> set PYTHONPATH=src
# \Yoke> python -m applications.harnesses.mnist_surrogate.START_study

import os
import argparse
import numpy as np
import pandas as pd


def replace_keys(study_dict, data: str) -> str:
    """Replaces "key" values in a string with dictionary values."""
    for key, value in study_dict.items():
        if key == "studyIDX":
            data = data.replace(f"<{key}>", f"{value:03d}")
        elif type(value) is np.float64 or type(value) is float:
            data = data.replace(f"<{key}>", f"{value:5.4f}")
        elif type(value) is np.int64 or type(value) is int:
            data = data.replace(f"<{key}>", f"{value:d}")
        elif type(value) is str:
            data = data.replace(f"<{key}>", f"{value}")
        elif type(value) is np.bool_ or type(value) is bool:
            data = data.replace(f"<{key}>", f"{str(value)}")
        else:
            print("Key is", key, "with value of", value, "with type", type(value))
            raise ValueError("Unrecognized datatype in hyperparameter list.")
    return data


# Parse hyperparameters.csv
hyperparameters_csv = os.path.join(os.path.dirname(__file__), "hyperparameters.csv")
parser = argparse.ArgumentParser(description="Starts execution of training studies")
parser.add_argument(
    "--csv",
    type=str,
    default=hyperparameters_csv,
    help="CSV file containing study hyperparameters",
)
args = parser.parse_args()

# Define path to the training input template
training_input_tmpl = os.path.join(os.path.dirname(__file__), "training_input.tmpl")


# Ensure training input template exists
if not os.path.exists(training_input_tmpl):
    print(f"Template file {training_input_tmpl} not found.")
    exit()

# Read csv file into a dataframe
try:
    studyDF = pd.read_csv(
        args.csv, sep=",", header=0, index_col=0, comment="#", engine="python"
    )
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# Get column names and index values
varnames = studyDF.columns.values
idxlist = studyDF.index.values

# Create list to store study dictionaries
studylist = []

# Iterate over each index value to create study dictionaries
for i in idxlist:
    studydict = {}
    studydict["studyIDX"] = int(i)
    for var in varnames:
        studydict[var] = studyDF.loc[i, var]
    studylist.append(studydict)

print(f"\nTotal studies to run: {len(studylist)}")

# Iterate over each study in the list
for k, study in enumerate(studylist):
    # Create a name for the study ("studyXXX")
    studyname = "study_{:03d}".format(study["studyIDX"])

    # Define filepath for study directory
    studydirname = os.path.join(
        os.path.dirname(__file__), "study_{:03d}".format(study["studyIDX"])
    )

    # Create directory for the study if it doesn't exist
    if not os.path.exists(studydirname):
        os.makedirs(studydirname)

    # Read the template data
    with open(training_input_tmpl) as f:
        training_input_data = f.read()

    # Replace placeholders in the data
    training_input_data = replace_keys(study, training_input_data)

    print(f"\n{studydirname} arguments:\n")
    print(training_input_data)

    # Define path to the output file (training_input.txt)
    training_input_filepath = os.path.join(studydirname, "training_input.txt")

    # Write the modified data to the output file
    with open(training_input_filepath, "w") as f:
        f.write(training_input_data)
        # for key, value in study.items():
        #     f.write(f"{key}={value}\n")

    print(f"\nRunning {studyname} with script {study['train_script']}\n")

    # Run the study script with the configuration file
    os.system(f"python {study['train_script']} --config-file {training_input_filepath}")

    print(f"Completed {studyname}")

print("\nAll studies completed.")
