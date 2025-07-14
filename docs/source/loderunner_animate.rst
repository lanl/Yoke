Lode Runner Animation Guide
=====================

This guide explains how to make GIF animations of loderunner models.

Overview
--------

This code explains how to generate GIF animations that shows the model's predictions over an entire simulation.

Code
-----

The code is available in the applications/evaluation folder. The `lsc_loderunner_create_gif.py` and `lsc_loderunner_create_gif.input` files are used to create the GIFs, with dependencies on the `lsc_loderunner_anime.py` script.

Instructions
------------

.. caution::

   Generating a GIF requires many calls to the model. The number of model calls will be `O(steps)`, where `steps` are the number of timesteps in the simulation. With larger and more complex models, this can be **very** slow if ran on CPU. You should probably use a **GPU** for this, given the potential for significant performance improvements. Even with 4 A100 GPUs, this can still take ~10-15 minutes.

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
`checkpoint`: Path to the model checkpoint file.
`indir`: Directory for the input NPZ files.
`outdir`: Directory for the output PNG files.
`runID`: The index of the simulation to use.
`embed_dim`: The size of the embedding dimension.
`verbose` (`-V`): Flag to turn on debugging output.
`mode`: The prediction mode to use (single, chained, or timestep). Please see below for more details on the prediction modes.

What is this `mode` argument, and how do I use it?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The `mode` argument specifies how the model will make predictions across multiple timesteps. The available modes are `single`, `chained`, and `timestep`. Each mode has its own advantages and disadvantages, which are detailed below.

Mode Descriptions
~~~~~~~~~~~~~~~~

Single Mode
~~~~~~~~~~~

In single mode, this prediction would be made by taking the TRUE input for the n-1 timestep.
The model would then use that, along with timestep k to predict timestep n.

**Advantages:**
- No propagating error.
- Requires one model call to predict. (O(1) model calls)
- Can be used on models trained to only predict one timestep ahead.

**Disadvantages:**
- Cannot predict multiple timesteps ahead.
- This mode takes one true timestep and predicts the next.
- Uses a constant dt, so some parameters may be ignored.

Chained Mode
~~~~~~~~~~~~

In chained mode, the following relation is used: P(n) = RM(I, n, k)

- P(n) is the prediction of the n'th timestep.
- RM(I, n, k) would mean run repeated predictions of the model.

Here is a simplified version of the code for this function::

    def RM(inp, steps, dt):
        current = inp
        for _ in range(n):  # underscore used because it doesn't matter
            current = M(current, dt)  # where M is the model.
        return current

**Advantages:**
- Requires only the initial timestep to generate predictions.
- Can be used on models trained to only predict one timestep ahead.

**Disadvantages:**
- Prediction of a timestep far ahead can be costly.
- Predicting timestep n requires n calls to the model. (O(n) model calls)
- Predictions lose accuracy across the rollout.
- The prediction quality decays significantly as the rollout progresses forward.
- Uses a constant dt, so some parameters may be ignored.

Timestep Mode
~~~~~~~~~~~~~

In timestep mode, the initial image at time zero is used for ALL predictions.
The timestep by itself is used to determine how far forward to predict.

**Advantages:**
- Requires only the initial timestep to generate predictions.
- Does not cause the dt related parameters to be wasted.
- Allows for predictions of a specific timestep to only require one model call. (O(1) model calls)

**Consideration:**
- Requires a model trained with a variety of timesteps.
- Training the model to predict one timestep ahead (exclusively) will NOT work effectively in this mode.


lsc_loderunner_create_gif.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`runs-dir`: Directory containing the models (and other outputs) from the training runs. The directory will typically have the form of `${YOKE_ROOT}/applications/harnesses/${HARNESS}/runs`, where `${YOKE_ROOT}` is the root directory of Yoke and `${HARNESS}` is the name of the harness. This will eventually turn into the `checkpoint` argument for `lsc_loderunner_anime.py`, where each study in the runs directory will have a subdirectory with various checkpoints. The checkpoint for the largest epoch is chosen by default.
`npz-dir`: Directory containing the input NPZ files. This gets directly passed to the `lsc_loderunner_anime.py` script as the `indir` argument.
`skip-list`: A comma separated list of studies to skip. The purpose of this is if I already have the GIF for one or more studies, I do not need to rerun them.
`run-id`: The index of the simulation to use. This gets directly passed to the `lsc_loderunner_anime.py` script as the `runID` argument.
`embed-dim`: The size of the embedding dimension. This gets directly passed to the `lsc_loderunner_anime.py` script as the `embed_dim` argument.

Basic Setup
-----------

Assume the initial image is at time zero.
Assume you are trying to predict the n'th timestep.
This would mean that you are most likely trying to predict time k * n, for some constant k.

Mode Descriptions
-----------------

Single Mode
~~~~~~~~~~~

In single mode, this prediction would be made by taking the TRUE input for the n-1 timestep.
The model would then use that, along with timestep k to predict timestep n.

**Advantages:**
- No propagating error.
- Requires one model call to predict. (O(1) model calls)
- Can be used on models trained to only predict one timestep ahead.

**Disadvantages:**
- Cannot predict multiple timesteps ahead.
- This mode takes one true timestep and predicts the next.
- Uses a constant dt, so some parameters may be ignored.

Chained Mode
~~~~~~~~~~~~

In chained mode, the following relation is used: P(n) = RM(I, n, k)

- P(n) is the prediction of the n'th timestep.
- RM(I, n, k) would mean run repeated predictions of the model.

Here is a simplified version of the code for this function::

    def RM(inp, steps, dt):
        current = inp
        for _ in range(n):  # underscore used because it doesn't matter
            current = M(current, dt)  # where M is the model.
        return current

**Advantages:**
- Requires only the initial timestep to generate predictions.
- Can be used on models trained to only predict one timestep ahead.

**Disadvantages:**
- Prediction of a timestep far ahead can be costly.
- Predicting timestep n requires n calls to the model. (O(n) model calls)
- Predictions lose accuracy across the rollout.
- The prediction quality decays significantly as the rollout progresses forward.
- Uses a constant dt, so some parameters may be ignored.

Timestep Mode
~~~~~~~~~~~~~

In timestep mode, the initial image at time zero is used for ALL predictions.
The timestep by itself is used to determine how far forward to predict.

**Advantages:**
- Requires only the initial timestep to generate predictions.
- Does not cause the dt related parameters to be wasted.
- Allows for predictions of a specific timestep to only require one model call. (O(1) model calls)

**Consideration:**
- Requires a model trained with a variety of timesteps.
- Training the model to predict one timestep ahead (exclusively) will NOT work effectively in this mode.

Which Mode Should You Use?
--------------------------

You must determine that, but here are some tips that may help:

Choose **Single Mode** when:
- You want to avoid propagating errors
- You need fast predictions (O(1) model calls)
- Your model was trained to predict only one timestep ahead
- You only need to predict the next immediate timestep

Choose **Chained Mode** when:
- You need to predict multiple timesteps ahead
- You only have the initial timestep available
- Your model was trained to predict one timestep ahead
- You can tolerate increasing prediction errors over time

Choose **Timestep Mode** when:
- Your model was trained with variable timesteps
- You want efficient predictions for specific future timesteps
- You want to preserve dt-related parameter information
- You can provide the initial timestep for all predictions
