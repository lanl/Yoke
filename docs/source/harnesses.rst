Authoring a Yoke Harness
========================

A **harness** is a self-contained, reproducible configuration for training or
evaluating a model on a specific dataset. A harness is defined entirely by its
files — there is no per-harness Python launcher to copy or edit. The installed
``yoke-start-study`` command reads those files, constructs a
:class:`yoke.harnesses.base.HarnessStudy`, and drives the study.

This guide is a recipe for creating a new harness from scratch.

What a harness contains
-----------------------

A harness directory under ``applications/harnesses/<name>/`` contains:

.. code-block:: none

    applications/harnesses/<name>/
    ├── train_<something>.py     # training / evaluation script
    ├── training_input.tmpl      # single input-argument template
    ├── training_slurm.tmpl      # complete SLURM submission script
    │   (or training_shell.tmpl) # complete shell submission script
    ├── cp_files.txt             # files copied into each study directory
    ├── <hyperparameters>.csv    # one row per study
    └── README.md                # what the harness does and how to run it

Note there is **no** ``START_study.py`` and **no** ``training_START.*`` files.
A single input template and a single submission template generate both the
first-launch and the epoch-continuation forms.

Step-by-step
------------

**1. Create the directory.**

.. code-block:: bash

    mkdir applications/harnesses/<name>

**2. Add a training script.**

Write ``train_<something>.py``. It should:

- Build its argument parser with the builders in :mod:`yoke.helpers.cli` and read
  arguments from a file using ``fromfile_prefix_chars="@"`` (so it can be invoked
  as ``python train.py @some.input``).
- Import reusable logic from ``yoke.*`` rather than defining it inline.
- Call :meth:`yoke.harnesses.base.HarnessStudy.continuation_setup` when a training
  cycle finishes but the total epoch count has not been reached, to render and
  submit the next restart cycle.

**3. Write one input template** (``training_input.tmpl``).

List the training script's arguments, one per line, using ``<KEY>`` tokens for the
values that vary per study. Wrap continuation-only arguments in an optional block
(see `Template convention`_ below):

.. code-block:: none

    --studyIDX
    <studyIDX>
    --init_learnrate
    <LEARN_RATE>
    --total_epochs
    100
    --cycle_epochs
    2
    # <<optional:CONTINUATION>>
    --continuation
    --checkpoint
    <CHECKPOINT>
    # <<end>>

**4. Write one submission template** — a complete script.

For SLURM, name it ``training_slurm.tmpl``; for a local/dev shell run, name it
``training_shell.tmpl``. Use ``<INPUTFILE>`` where the script reads its input
arguments and ``<epochIDX>`` in output/error filenames:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=study<studyIDX>_e<epochIDX>
    #SBATCH --output=study<studyIDX>_epoch<epochIDX>.out
    #SBATCH --error=study<studyIDX>_epoch<epochIDX>.err
    ...
    srun python -u <train_script> @<INPUTFILE>

**5. Write** ``cp_files.txt`` — one path per line, listing files to copy into each
generated ``study_###`` directory (typically just the training script).

**6. Write a hyperparameter CSV.** The **first column must be** ``studyIDX`` and
becomes the study index. Each remaining column is a ``<KEY>`` substituted into the
templates. One row produces one ``study_###`` directory:

.. code-block:: none

    studyIDX,LEARN_RATE,BATCH_SIZE,train_script
    1,0.001,8,train_something.py
    2,0.002,8,train_something.py

Lines beginning with ``#`` are treated as comments and skipped, so alternate
configurations can be kept inline for reference.

**7. Launch** with the CLI (see :doc:`start_study`):

.. code-block:: bash

    cd applications/harnesses/<name>
    yoke-start-study --csv <hyperparameters>.csv --submissionType slurm --dryrun

Drop ``--dryrun`` to actually submit. Use ``--submissionType shell`` for local
runs.

.. _Template convention:

Template convention
-------------------

Templates are rendered with two mechanisms:

- ``<KEY>`` — literal substitution from the study row (and from the reserved keys
  below). See :func:`yoke.helpers.strings.replace_keys`.
- ``# <<optional:KEY>>`` ... ``# <<end>>`` — the enclosed lines are included in the
  rendered output **only** when ``KEY`` is present in the substitution dictionary.
  This is how continuation-only arguments (``--continuation`` / ``--checkpoint``)
  appear in the continuation form but not the first-launch form.

Reserved keys
-------------

These keys are injected automatically by ``HarnessStudy`` — do **not** put them in
the CSV:

- ``<studyIDX>`` — study index, zero-padded to **three** digits.
- ``<epochIDX>`` — epoch index, zero-padded to **four** digits. It is ``0001`` for
  the first launch and the next epoch (``last_epoch + 1``) on continuation.
- ``<INPUTFILE>`` — the input file the submission script should read; set to the
  first-launch input, or to the per-epoch restart input on continuation.
- ``<CONTINUATION>`` — present only in the continuation form; use it to gate the
  ``# <<optional:CONTINUATION>>`` block.
- ``<CHECKPOINT>`` — the checkpoint to resume from; filled in per epoch by
  :meth:`yoke.harnesses.base.HarnessStudy.continuation_setup`.

The continuation lifecycle
---------------------------

Training runs in cycles. When a cycle finishes without reaching the total epoch
count, the training script calls
:meth:`yoke.harnesses.base.HarnessStudy.continuation_setup` on the compute node.
That method reads the study directory's ``training_input.tmpl`` and submission
template, substitutes ``<CHECKPOINT>``/``<INPUTFILE>``/``<epochIDX>``, writes the
per-epoch restart files (``study###_restart_training_epoch####.{input,slurm|sh}``),
and returns the new submission-script name for the training script to resubmit.
Only ``slurm`` and ``shell`` submission systems are supported.

Data locations
--------------

Data is **not** stored in the repo. Dataset paths are passed as command-line
arguments (see the ``--*_DIR`` builders in :mod:`yoke.helpers.cli`); never hardcode
dataset locations into the installable package.
