# Utility Scripts

This directory contains standalone helper scripts that support Yoke workflows.
They are not intended to be imported as modules. Each script serves a specific
function useful during development or production runs.

## Contents

### archive_runs.sh
Archives run directories by timestamping and compressing output folders.
Useful for cleaning up after long training runs and saving logs efficiently.

### check_progress.py
Parses SLURM output files to assess training progress. Detects whether a run
is still active, has converged, or crashed based on output file patterns.

### run_yoke_study.sh
A wrapper script to launch Yoke training studies. Prepares inputs, submits a
job via SLURM, and manages naming and log directories automatically.

### split_csv.py
Splits a large CSV file into smaller chunks. Used to partition input data
across processes or nodes in distributed training workflows.

## Notes

These scripts are run directly and are not part of the Yoke importable code.
Place executable scripts here rather than inside `src/` to maintain clarity
between run-time tools and modular source code.