The ``yoke-start-study`` CLI
============================

``yoke-start-study`` is the installed command-line tool used to launch a training
or evaluation **study** from a harness directory. It replaces the old,
copied-per-harness ``START_study.py`` script: the launcher logic now lives in the
installable package (:mod:`yoke.cli.start_study` and
:class:`yoke.harnesses.base.HarnessStudy`) and is invoked as a real command.

How it works
------------

Run from inside a harness directory, ``yoke-start-study``:

1. Parses arguments (:func:`yoke.helpers.cli.add_default_args`).
2. Constructs a :class:`yoke.harnesses.base.HarnessStudy` from the harness's
   configuration files (``training_input.tmpl``, the submission template, and
   ``cp_files.txt``).
3. Loads the hyperparameter CSV — one row per study.
4. For each row, creates ``runs/study_###/`` (``###`` = ``studyIDX``), renders the
   first-launch and continuation templates, copies the ``cp_files.txt`` entries,
   and submits the first job (``sbatch`` for SLURM, ``source`` for shell).

Because the CSV drives everything, re-using a harness means editing configuration
files, not rewriting a launcher.

Usage
-----

.. code-block:: bash

    cd applications/harnesses/ch_DDP_loderunner
    yoke-start-study --csv ddp_paper_study.csv --submissionType slurm

Arguments
---------

- ``--csv`` — hyperparameter CSV (default ``./hyperparameters.csv``). Its first
  column is ``studyIDX``; one row maps to one ``study_###`` directory.
- ``--rundir`` — directory to create the ``study_###`` directories in
  (default ``./runs``). This is typically a softlink to scratch space.
- ``--cpFile`` — text file listing files to copy into each study directory
  (default ``./cp_files.txt``).
- ``--submissionType`` — ``slurm`` (submit with ``sbatch``) or ``shell`` (submit
  with ``source``, for local/dev runs). Defaults to ``slurm``.
- ``--dryrun`` — prepare study directories and render all files, but print the
  submit command instead of executing it. Nothing is submitted.

Dry runs
--------

Use ``--dryrun`` to inspect exactly what a study would do without touching the
scheduler:

.. code-block:: bash

    yoke-start-study --csv ddp_paper_study.csv --submissionType slurm --dryrun

This renders each ``study_###`` directory and prints the ``sbatch``/``source``
command that *would* be run.

Selecting which studies to run
------------------------------

``yoke-start-study`` iterates **every** row of the CSV. To run a single study,
reduce the CSV to that one row (comment out the others with a leading ``#``), or
keep separate CSV files per study set.

Authoring a harness
-------------------

For how the templates, reserved keys, and continuation lifecycle work, see
:doc:`harnesses`.
