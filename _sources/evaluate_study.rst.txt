Post-training evaluation
========================

Yoke harnesses can run a **separate test-set evaluation job** for a trained
checkpoint. Evaluation is intentionally its own job, not a phase of training:
training needs DDP/NCCL resources and periodically resubmits itself, while
evaluation does not train, checkpoint, or need multiple GPUs. Keeping them
separate lets each harness pick appropriate evaluation resources and lets
evaluation be rerun independently.

There are two ways an evaluation job is created:

- **Automatically**, once, at the end of a *finished* training study (opt-in via
  :class:`yoke.harnesses.trainer.HarnessTrainer`).
- **Manually**, for any checkpoint, via the ``yoke-evaluate-study`` CLI.

Both paths share :meth:`yoke.harnesses.base.HarnessStudy.run_evaluation`, which
renders the checkpoint-specific input/submission files and submits them.

Enabling automatic evaluation
------------------------------

An evaluation-enabled harness adds these optional files alongside its training
files (the evaluator must be listed in ``cp_files.txt`` so it reaches each study
directory):

.. code-block:: none

    evaluation_input.tmpl
    evaluation_slurm.tmpl        # for --submissionType slurm
    evaluation_shell.tmpl        # for --submissionType shell
    eval_<harness>.py

The training script opts in by passing ``evaluate_after_training=True`` to
``HarnessTrainer``. When the final training cycle completes, rank 0 saves the
checkpoint and then submits exactly one evaluation job for it. If a harness
maintains a warmup-EMA shadow, pass ``evaluate_checkpoint="ema"`` to evaluate the
class-aware EMA companion (``..._ema.pth``) instead of the ordinary checkpoint.

.. code-block:: python

    HarnessTrainer(
        args,
        ...,
        evaluate_after_training=True,
        evaluate_checkpoint="main",   # or "ema"
    ).run()

Evaluation templates
--------------------

Evaluation templates are rendered at study creation with the study's CSV keys,
leaving three tokens late-bound for the specific checkpoint:

- ``<CHECKPOINT>`` — the checkpoint path being evaluated.
- ``<INPUTFILE>`` — the rendered evaluation input file the submission reads.
- ``<STEM>`` — the checkpoint filename stem, used to name all output artifacts.

Using the stem means an ordinary checkpoint and its EMA companion produce
distinct, non-colliding artifacts. For ``study005_modelState_epoch0100.pth``:

.. code-block:: none

    study005_evaluation_study005_modelState_epoch0100.input
    study005_evaluation_study005_modelState_epoch0100.slurm
    testing_study005_study005_modelState_epoch0100.csv

The evaluator reads the epoch it records from the checkpoint metadata itself;
the stem is only for naming.

The ``yoke-evaluate-study`` CLI
-------------------------------

Run from inside a harness directory to evaluate **any** checkpoint — an earlier
epoch, an EMA companion, a re-run, or a study produced before the evaluation
feature existed:

.. code-block:: bash

    cd applications/harnesses/ch_DDP_loderunner
    yoke-evaluate-study --studyIDX 5 \
        --checkpoint runs/study_005/study005_modelState_epoch0100.pth

The CLI resolves ``runs/study_005/`` and its CSV row. If that study directory
already contains rendered evaluation templates (a feature-era study) they are
used as-is. If not (an older study), the templates are re-rendered on demand from
the harness directory using the study's CSV row and the evaluator is copied in.
The checkpoint-specific files are then rendered and submitted.

Arguments
~~~~~~~~~

- ``--studyIDX`` — study index whose ``runs/study_###`` directory holds the
  checkpoint (required).
- ``--checkpoint`` — full path to the checkpoint to evaluate; may be an ordinary
  checkpoint or an EMA companion (``..._ema.pth``) (required).
- ``--rundir`` — directory containing the ``study_###`` directories
  (default ``./runs``).
- ``--csv`` — hyperparameter CSV used to re-render evaluation templates for
  studies that do not already contain them (default ``./hyperparameters.csv``).
- ``--cpFile`` — files to copy into the study directory when rendering on demand
  (default ``./cp_files.txt``).
- ``--submissionType`` — ``slurm`` or ``shell`` (default ``slurm``).
- ``--dryrun`` — render the evaluation files and print the submit command
  without submitting.

Dry runs
~~~~~~~~

.. code-block:: bash

    yoke-evaluate-study --studyIDX 5 \
        --checkpoint runs/study_005/study005_modelState_epoch0100.pth --dryrun

This writes the checkpoint-specific ``*.input`` and ``*.slurm`` (or ``*.sh``)
files into the study directory and prints the ``sbatch``/``source`` command that
*would* run, so you can inspect the rendered configuration before submitting.
