"""Functions to restart the training process after checkpointing."""

import os


def continuation_setup(
    checkpointpath: str, studyIDX: int, last_epoch: int, submissionType: str = "slurm"
) -> str:
    """Modify template submission files for training restart.

    Function to generate the training.input and training.slurm files for
    continuation of model training

    Args:
         checkpointpath (str): path to model checkpoint to load in model from
         studyIDX (int): study ID to include in file name
         last_epoch (int): numer of epochs completed at this checkpoint
         submissionType (str): Type of job-submission system to prepare files for.
                               Default 'slurm', either 'slurm', 'flux', 'shell',
                               or 'batch'

    Returns:
         new_training_filepath (str): Name of job-submission file defining job for
                                      continued training

    """
    # Input template is independent of submission sytem
    training_input_tmpl = "./training_input.tmpl"

    # Make new training.input file
    with open(training_input_tmpl) as f:
        training_input_data = f.read()

    new_training_input_data = training_input_data.replace("<CHECKPOINT>", checkpointpath)

    input_str = "study{0:03d}_restart_training_epoch{1:04d}.input"
    new_training_input_filepath = input_str.format(studyIDX, last_epoch + 1)

    with open(os.path.join("./", new_training_input_filepath), "w") as f:
        f.write(new_training_input_data)

    # Choose job-scheduling system
    if submissionType.lower() == "slurm":
        training_slurm_tmpl = "./training_slurm.tmpl"

        with open(training_slurm_tmpl) as f:
            training_slurm_data = f.read()

        slurm_str = "study{0:03d}_restart_training_epoch{1:04d}.slurm"
        new_training_filepath = slurm_str.format(studyIDX, last_epoch + 1)

        new_training_slurm_data = training_slurm_data.replace(
            "<INPUTFILE>", new_training_input_filepath
        )

        new_training_slurm_data = new_training_slurm_data.replace(
            "<epochIDX>", f"{last_epoch + 1:04d}"
        )

        with open(os.path.join("./", new_training_filepath), "w") as f:
            f.write(new_training_slurm_data)

    if submissionType.lower() == "flux":
        training_flux_tmpl = "./training_flux.tmpl"

        with open(training_flux_tmpl) as f:
            training_flux_data = f.read()

        flux_str = "study{0:03d}_restart_training_epoch{1:04d}.flux"
        new_training_filepath = flux_str.format(studyIDX, last_epoch + 1)

        new_training_flux_data = training_flux_data.replace(
            "<INPUTFILE>", new_training_input_filepath
        )

        new_training_flux_data = new_training_flux_data.replace(
            "<epochIDX>", f"{last_epoch + 1:04d}"
        )

        with open(os.path.join("./", new_training_filepath), "w") as f:
            f.write(new_training_flux_data)

    elif submissionType.lower() == "shell":
        training_shell_tmpl = "./training_shell.tmpl"

        with open(training_shell_tmpl) as f:
            training_shell_data = f.read()

        shell_str = "study{0:03d}_restart_training_epoch{1:04d}.sh"
        new_training_filepath = shell_str.format(studyIDX, last_epoch + 1)

        new_training_shell_data = training_shell_data.replace(
            "<INPUTFILE>", new_training_input_filepath
        )

        new_training_shell_data = new_training_shell_data.replace(
            "<epochIDX>", f"{last_epoch + 1:04d}"
        )

        with open(os.path.join("./", new_training_filepath), "w") as f:
            f.write(new_training_shell_data)

    return new_training_filepath
