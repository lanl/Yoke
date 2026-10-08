LodeRunner Chained Animation Guide
==================================

This guide explains how to make chained-rollout MP4 or GIF animations from
current LodeRunner checkpoints.

Overview
--------

``applications/evaluation/lsc_loderunner_chained_animation.py`` starts from
one or two ground-truth frames, predicts each subsequent frame
autoregressively, and compares every prediction with the corresponding LSC
simulation frame. It writes truth/prediction/discrepancy PNG panels for every
model channel and builds an animation for one selected field.

The script supports checkpoints created by
``yoke.utils.checkpointing.save_model_and_optimizer`` for ``LodeRunner`` and
``LodeRunnerViT``. Model architecture and field ordering come from checkpoint
metadata rather than command-line architecture parameters.

Usage
-----

The checked-in ``applications/evaluation/lsc_anime.input`` configures a
two-frame example. Run it from the repository root:

.. code-block:: console

   python applications/evaluation/lsc_loderunner_chained_animation.py \
       @applications/evaluation/lsc_anime.input

The output extension selects the animation format. MP4 output requires
FFmpeg; GIF output requires ImageMagick. PNG diagnostics are written under
``<output-stem>_frames/<channel>/`` and movie panels under
``<output-stem>_frames/movie/``.

Arguments
---------

``checkpoint``
   Path to a current ``.pth`` model and optimizer checkpoint.

``LSC_NPZ_DIR``
   Directory containing the LSC NPZ files.

``simulation_prefix``
   Simulation prefix such as ``lsc240420_id00201``. Rollouts always begin at
   index zero.

``prediction_length``
   Number of new autoregressive frames to predict, excluding seed frames.

``num_input_frames``
   ``1`` or ``2``. This must match the checkpoint model metadata.

``output_filename``
   Output filename ending in ``.mp4`` or ``.gif``.

``movie_field``
   Exact checkpoint channel name or ``sum_density``, ``sum_energy``, or
   ``sum_pressure``. Aggregate options sum all channels containing the
   corresponding term. The default is ``sum_density``.

``fps``
   Animation frames per second. The default is ``4``.

.. caution::

   Autoregressive inference is sequential, and every output step renders one
   panel per model channel. Long rollouts and models with many channels can be
   expensive. A CUDA device is used automatically when available.
