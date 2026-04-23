Lode Runner Animation Guide
===========================

This guide explains how to make GIF animations of loderunner models.

Overview
--------

This code explains how to generate GIF animations that show the model's predictions over an entire simulation.

Code
-----

The code is available in the `applications/evaluation` folder. There are two scripts:

- `lsc_loderunner_anime.py`: This script generates the PNG files for each timestep of the simulation.
- `lsc_loderunner_create_gif.py`: This script uses `lsc_loderunner_anime.py` to generate PNG files, then generates a GIF from those PNG files.

Instructions
------------

.. caution::

   Generating a GIF requires many calls to the model. The number of model calls will be `O(steps)`, where `steps` are the number of timesteps in the simulation. With larger and more complex models, this can be **very** slow if run on CPU. You should probably use a **GPU** for this, given the potential for significant performance improvements. Even with 4 Nvidia A100 GPUs, this can still take ~10-15 minutes.

1. Setup the environment, making sure you have a good version of Python loaded as well as Yoke. As stated above, you probably should be using GPUs, so you may want to check that you have GPUs. For Nvidia GPUs you can use the `nvidia-smi` command to check that you have GPUs available. If you need help setting up the environment, please refer to one of the SLURM scripts in the `applications/evaluation/harnesses` folder. This was designed for a specific system, and may not work for you, so adjust as necessary.
2. Check the `lsc_loderunner_anime.py` script to make sure the command line arguments are set correctly and that the model parameters (especially the block structure) are set to the correct values. Please note: some things are hard coded for now (this should likely change in the future), so you may need to modify the code to suit your needs.
3. Check the `lsc_loderunner_create_gif.py` script to make sure you are using the correct arguments for `lsc_loderunner_anime.py`. These are currently hard coded, near the bottom of the script. Other than that, you should need little to no modifications to `lsc_loderunner_create_gif.py`.
4. Check the `lsc_loderunner_create_gif.input` file to make sure the parameters are set correctly. This file is an input file that gets fed into `lsc_loderunner_create_gif.py`.
5. Check your environment one more time to make sure that everything is set up correctly.
6. Run `python lsc_loderunner_create_gif.py @lsc_loderunner_create_gif.input` to generate the GIF.

Script CLI arguments
----------------------
This section details the command line arguments for both the `lsc_loderunner_anime.py` and `lsc_loderunner_create_gif.py` scripts.

lsc_loderunner_anime.py
~~~~~~~~~~~~~~~~~~~~~~~~
- `checkpoint`: Path to the model checkpoint file.
- `indir`: Directory for the input NPZ files.
- `outdir`: Directory for the output PNG files.
- `runID`: The index of the simulation to use.
- `embed_dim`: The size of the embedding dimension.
- `verbose` (`-V`): Flag to turn on debugging output.
- `mode`: The prediction mode to use (single, chained, or timestep). Please see the `"Prediction Modes Guide" document <prediction_modes.html>`_ for more details on these modes.


lsc_loderunner_create_gif.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- `runs-dir`: Directory containing the models (and other outputs) from the training runs. The directory will typically have the form of `${YOKE_ROOT}/applications/harnesses/${HARNESS}/runs`, where `${YOKE_ROOT}` is the root directory of Yoke and `${HARNESS}` is the name of the harness. This will eventually turn into the `checkpoint` argument for `lsc_loderunner_anime.py`, where each study in the runs directory will have a subdirectory with various checkpoints. The checkpoint for the largest epoch is chosen by default.
- `npz-dir`: Directory containing the input NPZ files. This gets directly passed to the `lsc_loderunner_anime.py` script as the `indir` argument.
- `skip-list`: A comma separated list of studies to skip. The purpose of this is if I already have the GIF for one or more studies, I do not need to rerun them.
- `run-id`: The index of the simulation to use. This gets directly passed to the `lsc_loderunner_anime.py` script as the `runID` argument.
- `embed-dim`: The size of the embedding dimension. This gets directly passed to the `lsc_loderunner_anime.py` script as the `embed_dim` argument.