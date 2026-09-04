Moving MNIST Training
=====================

Trains a LodeRunner model on the **Moving MNIST** video dataset — a small,
fast spatiotemporal prediction task. Like `mnist_surrogate`, this harness is a
lightweight end-to-end functionality check for the Yoke training/harness workflow
and is a good smoke test after code changes and before launching large HPC
studies. Together with `mnist_surrogate` it replaces the former `mini-run-test`
harness for quick functionality checks.

Files
-----

- `train_mnist_moving.py` — Moving MNIST LodeRunner training script.
- `training_input.tmpl` — single input template (with the
  `# <<optional:CONTINUATION>>` block for continuation runs).
- `training_shell.tmpl` — complete shell submission script for local/dev runs.
- `cp_files.txt` — files copied into each `study_###` run directory.
- `hyperparameters.csv` — two studies (identical parameters, distinct indices).

Study
-----

`hyperparameters.csv` configures batch size, epochs, and learning rate.
`data_dir` points at the Moving MNIST data location (downloaded automatically if
absent). This harness is shell/local-run oriented; it ships a
`training_shell.tmpl` submission script.

Launch
------

From within this directory, with the Yoke environment active:

```bash
yoke-start-study --csv hyperparameters.csv --submissionType shell
```

Add `--dryrun` to render the study directories and print the `source` commands
without executing them.
