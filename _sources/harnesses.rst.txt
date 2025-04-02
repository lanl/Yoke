Yoke Harnesses
==============

.. warning::

   🚧 **This section is under active construction!**  
   Expect rough edges, placeholder content, and ongoing changes.

Harnesses are self-contained configurations and scripts for training and evaluating
models on specific datasets. They allow users to plug in datasets and kick off training
runs using predefined architectures and hyperparameters.

Each harness typically includes:

- One or more training scripts (e.g., ``train_density_LodeRunner_scheduled.py``)
- A ``START_study.py`` file defining the high-level training logic
- Dataset-specific hyperparameter files (e.g., ``hyperparameters.csv``)
- Templated input and SLURM job scripts

📂 Example Harness Directory:
-----------------------------

.. code-block:: none

    applications/harnesses/chicoma_lsc_loderunner_scheduled/
    ├── START_study.py
    ├── train_density_LodeRunner_scheduled.py
    ├── hyperparameters.csv
    ├── training_input.tmpl
    ├── training_slurm.tmpl
    └── training_START.input

🧠 What Makes a Harness?
-------------------------

Each harness is designed around:

- **A model variant** (e.g., LodeRunner)
- **A dataset** (external, user-provided)
- **Training strategies** like scheduled sampling, patch merging, etc.
- **SLURM batch compatibility** for cluster runs

Harnesses help ensure reproducibility and portability of experiments.

🚀 Running a Harness
---------------------

To run a harness, edit the relevant ``START_study.py`` or ``train_*.py`` script
with dataset paths and submit the job via SLURM using the provided templates.

Harnesses can be adapted or created from scratch for new datasets or tasks.

