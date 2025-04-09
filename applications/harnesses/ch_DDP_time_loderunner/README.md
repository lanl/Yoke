# Update of Yoke Harness "LodeRunner Training - DDP - Chicoma": Dynamic max_timeIDX_offset

This document explains the modifications made to the Yoke training harness to allow dynamic control of the `max_timeIDX_offset` parameter using a CSV file. Previously, the parameter was hard-coded in the training script. With these updates, the parameter now comes from a CSV file, enabling hyperparameter studies and production-ready configurations.

## Overview of Changes

### CSV File: `ddp_study_time.csv`
- **Purpose:**  
  Contains a collection of hyperparameter settings.
- **Key Update:**  
  Includes **six different settings** for `max_timeIDX_offset` along with optimized settings for other hyperparameters.
- **Usage:**  
  The orchestrator (`START_study.py`) reads this CSV, converting each row into a dictionary of settings to be merged with configuration templates.

### Template Files
- **`training_input.tmpl`:**  
  - Modified to include a placeholder for `max_timeIDX_offset`.  
  - Placeholder added:
    ```
    --max_timeIDX_offset
    <MAX_TIMEIDX_OFFSET>
    ```
- **`training_START.input`:**  
  - Similarly updated to include a `<MAX_TIMEIDX_OFFSET>` placeholder so that the training script receives the correct value at runtime.
  
  These changes ensure that the dynamic value from the CSV will be substituted correctly when the orchestrator generates the run-specific configuration file.

### Training Script: `train_LodeRunner_ddp_time.py`
- **Modification:**  
  - Removed the hard-coded value of `max_timeIDX_offset` (previously set as 2).
  - Added a dedicated **Data Parameters** block to extract the `max_timeIDX_offset` value from the command-line arguments.
  - Code snippet added in the main function:
    ```python
    # Data Parameters
    max_timeIDX_offset = args.max_timeIDX_offset
    ```
  - The dataset is now initialized using:
    ```python
    train_dataset = LSC_rho2rho_temporal_DataSet(
        args.LSC_NPZ_DIR,
        file_prefix_list=train_filelist,
        max_timeIDX_offset=max_timeIDX_offset,
        max_file_checks=10,
        half_image=True,
    )
    ```
  - This update ensures that the training script uses the value provided from the CSV via the orchestrator.

### Updated `cp_files.txt` for consistency with the new name of the training script.

### Added `ddp_study_time_test.csv` with parameter settings intended to minimize runtime (for testing). 

### Files Left Unchanged
- **`START_study.py`:**  
  The overall study orchestration (including CSV parsing and template merging) remains the same.
- **`training_slurm.tmpl` & `training_START.slurm`:**  
  These files (used for job submission via Slurm) have not been modified because they do not directly handle training-specific parameter values.

## How to Use the Updated Harness

1. **Prepare the CSV File:**  
   Ensure that `ddp_study_time.csv` contains the desired settings for `max_timeIDX_offset` and the other hyperparameter values.

2. **Confirm Template Placeholders:**  
   Verify that both `training_input.tmpl` and `training_START.input` contain the `<MAX_TIMEIDX_OFFSET>` placeholder, matching the CSV header (or appropriately mapped by the orchestrator).

3. **Run the Orchestrator:**  
   Use the existing workflow (via `START_study.py`), which will:
   - Read the CSV file,
   - Replace placeholders in the template files with the corresponding CSV values (including `max_timeIDX_offset`), and
   - Generate study-specific configuration files for job submission.
   - Example command: `python START_study.py --csv ddp_study_time.csv.$USER`

4. **Execution of Jobs:**  
   Jobs will be submitted as usual with the unchanged `training_slurm.tmpl` and `training_START.slurm` files. The training script (`train_LodeRunner_ddp_time.py`) will now receive the dynamic value for `max_timeIDX_offset`.

## Summary

- **Dynamic Parameter Control:**  
  `max_timeIDX_offset` is now controlled by `ddp_study_time.csv` instead of being fixed at a hard-coded value.
- **Template Updates:**  
  Both `training_input.tmpl` and `training_START.input` have been updated with appropriate placeholders.
- **Training Script Adjustments:**  
  `train_LodeRunner_ddp_time.py` now retrieves `max_timeIDX_offset` from command-line arguments, enabling flexible data loading based on CSV settings.
- **Unchanged Components:**  
  The orchestrator (`START_study.py`) and job submission scripts (`training_slurm.tmpl`, `training_START.slurm`) remain unmodified.

These updates allow for expanded hyperparameter studies that include consideration of the `max_timeIDX_offset` parameter.


