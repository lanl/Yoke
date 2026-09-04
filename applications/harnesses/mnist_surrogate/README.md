MNIST Surrogate Training
========================

A small, fast CNN classifier trained on MNIST. This harness is a lightweight
end-to-end functionality check for the Yoke training/harness workflow — it runs
quickly on a laptop or a single GPU and is a good smoke test after code changes
and before launching large HPC studies. It (together with `moving_mnist`)
replaces the former `mini-run-test` harness for quick functionality checks.

Files
-----

- `train_mnist_surrogate.py` — MNIST classifier training script.
- `training_input.tmpl` — single input template (with the
  `# <<optional:CONTINUATION>>` block for continuation runs).
- `training_slurm.tmpl` — complete SLURM submission script.
- `training_shell.tmpl` — complete shell submission script for local/dev runs.
- `cp_files.txt` — files copied into each `study_###` run directory.
- `hyperparameters.csv` — two studies varying the CNN channel widths.

Study
-----

`hyperparameters.csv` defines two studies that vary the convolution channel
widths (`conv1`–`conv4`) while holding batch size, epochs, learning rate, and LR
decay (`gamma`) fixed. `data_dir` points at the MNIST data location (downloaded
automatically if absent).

Launch
------

From within this directory, with the Yoke environment active. For a local/dev
run use the shell submission type:

```bash
yoke-start-study --csv hyperparameters.csv --submissionType shell
```

Or submit to SLURM:

```bash
yoke-start-study --csv hyperparameters.csv --submissionType slurm
```

Add `--dryrun` to render the study directories and print the submit commands
without executing them.
