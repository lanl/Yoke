Using START_study.py
=====================

.. warning::

   🚧 **This section is under active construction!**  
   Expect rough edges, placeholder content, and ongoing changes.

``START_study.py`` is the common entry point script used across all harnesses in
Yoke to launch a full training or evaluation "study". It provides a consistent
interface for kicking off jobs using dataset-specific configurations, SLURM job
templates, and predefined model setups.

How It Works
------------

Each harness directory contains a copy of ``START_study.py``. Despite being
duplicated across folders, the script operates similarly in each case:

1. Loads harness-specific config files (e.g. ``hyperparameters.csv``, `.tmpl` templates)
2. Prepares SLURM job scripts and input files
3. Submits the job using `sbatch` or a similar SLURM-compatible mechanism
4. Optionally logs outputs, paths, or study parameters

This makes it easy to clone a harness, adjust just the configuration, and reuse the
same study launcher without rewriting training logic.

Typical Usage
-------------

.. code-block:: bash

    cd applications/harnesses/chicoma_lsc_loderunner_scheduled
    python ../START_study.py

This will:

- Prepare training input files using Jinja2 templates
- Configure job submission parameters
- Launch one or more training runs via SLURM

Customization
-------------

Harness-specific logic (e.g., model choice, dataset location) is controlled by:

- Values in ``hyperparameters.csv``
- Template files (e.g. ``training_input.tmpl``, ``training_slurm.tmpl``)
- CLI arguments passed to ``START_study.py`` (if supported)

You can modify or extend this script to support new clusters, workflows, or
non-SLURM systems.

