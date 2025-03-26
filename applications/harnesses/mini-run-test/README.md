# Mini Run Test Harness

This folder contains the **Mini Run Test Harness**, a lightweight framework designed for an end-to-end test. This ensures a full-weight study is ran, just significantly faster.

## Purpose

This serves as an extra validation step when the user has already tested their code. This should be used after all automated testing has passed (all CI testing), but before using HPC to run a large-scale model.

## Usage

After making the desired modifications to the ML script, (currently `train_density_LodeRunner.py`), run the harness in the typical way. To do this, load the python environment then run:

```bash
python START_study.py
```