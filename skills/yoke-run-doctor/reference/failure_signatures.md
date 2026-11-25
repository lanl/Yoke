# Failure Signatures

Common ways Yoke runs fail or go wrong, how to recognize them, and what to
propose. These are drawn from general PyTorch, DDP, and SLURM experience and
from Yoke's code. **They have not yet been checked against real failures on
LANL machines.** When a real failure turns out differently, update this file
with the actual message, the machine, and the date.

For each one: where it shows, the signature, the likely cause, and the fix to
propose. Fixes are proposals; any edit to harness files needs approval, and
any resubmission is run by the user.

## Job-level (SLURM)

### Wall time exceeded

- **Shows:** `sacct` `State=TIMEOUT`; `.err` ends with
  `*** JOB <id> ON <node> CANCELLED AT ... DUE TO TIME LIMIT ***`.
- **Check:** is there a checkpoint for this cycle? If not, the cycle took
  longer than `#SBATCH --time`, and nothing from it was saved.
- **Cause:** `cycle_epochs × (time per epoch)` is larger than the wall time.
  The time per epoch is in `.out` (`Epoch time (minutes): ...`).
- **Propose:** lower `cycle_epochs` (or `NTRN_BATCH`), or raise `--time` up to
  the partition or QOS limit (see the machine profile). Restart from the
  last checkpoint.

### Out of memory (GPU)

- **Shows:** `.err`: `torch.OutOfMemoryError: CUDA out of memory`, or
  `CUDA error: out of memory`.
- **Cause:** batch size, model size, or image size is too large for one GPU.
  `batch_size` is **per GPU** under DDP.
- **Propose:** a smaller `batch_size`, a smaller model (fewer or narrower
  layers), or a downscaled image size. Check the parameter count in the
  preflight report.

### Out of memory (host) or killed

- **Shows:** `sacct` `State=OUT_OF_MEMORY`, or `.err`
  `Killed` / `oom-kill event`, or dataloader workers dying
  (`DataLoader worker (pid ...) is killed by signal`).
- **Cause:** too many `num_workers` × large samples, or `--mem-per-gpu` too
  low.
- **Propose:** lower `NUM_WORKERS`, raise `--mem-per-gpu`, and check
  `MaxRSS` in `sacct`.

### sbatch rejects the QOS

- **Shows:** `yoke-start-study` output (no job ID):
  `sbatch: error: Batch job submission failed: Invalid qos specification`.
- **Cause:** the account lacks that QOS on that partition, or the QOS has the
  `RequiresReservation` flag (`sacctmgr -nP show qos <name> format=name,flags`)
  and the job gave no `--reservation`. Seen on rocinante's `debug` QOS,
  2026-11-13.
- **Propose:** add `#SBATCH --reservation=<name>` (see the machine profile),
  check it with `sbatch --test-only <script>`, then relaunch. The study dirs
  were already rendered, so relaunching the same CSV rewrites them.

### Job never started or was cancelled

- **Shows:** `squeue` shows `PD` with reason `(AssocGrpGRESMinutes)`,
  `(QOSMaxWallDurationPerJobLimit)`, `(InvalidAccount)`, or `(ReqNodeNotAvail)`;
  or `sacct` `State=CANCELLED` with no user action.
- **Cause:** allocation exhausted or invalid account, wall time above the QOS
  limit, partition unavailable.
- **Propose:** re-query allocations (`sacctmgr -nP show assoc user=$USER`),
  fix the `allocation` column or `--time`. These are the user's to resolve
  with the site if the account is out of hours.

## Distributed (DDP / NCCL)

### GPU-mode harness on a CPU-only node

- **Shows:** `.err` at start-up: `Torch not compiled with CUDA enabled`,
  `RuntimeError: No CUDA GPUs are available`, `invalid device ordinal`, or an
  NCCL init error, from `yoke.utils.parallel.setup_distributed`.
- **Cause:** the harness constructs `HarnessTrainer` with the default
  `device_type="cuda"`, but the job ran on a node without GPUs (a CPU-only
  cluster or partition).
- **Propose:** pass `device_type="cpu"` in the train script, and switch to
  the CPU-only SLURM template (one node, one task; see
  `skills/yoke-slurm-template/reference/slurm_template_skeleton.md`).

### CPU mode launched with several tasks

- **Shows:** `.err`: `RuntimeError: device_type='cpu' runs a single process,
  but SLURM_NTASKS=N`.
- **Cause:** the SLURM template asks for more than one task (`--ntasks`,
  `--ntasks-per-node`), or runs the script under a multi-task `srun`.
- **Propose:** `#SBATCH --nodes=1`, `#SBATCH --ntasks=1`, and launch with
  `python -u <train_script> @<INPUTFILE>`.

### NCCL cannot connect or times out

- **Shows:** `.err`: `NCCL error`, `ncclSystemError`, `Socket Timeout`,
  `Connection refused`, or a hang followed by
  `Watchdog caught collective operation timeout`.
- **Cause:** wrong `NCCL_SOCKET_IFNAME` for the compute nodes, a
  `MASTER_PORT` collision, or one rank crashing while the others wait.
- **Propose:** check the interface against the machine profile; check the
  other ranks' errors first (the real error is often on one rank only);
  set `NCCL_DEBUG=INFO` (commented out in the template) for one run.

### Ranks disagree

- **Shows:** hang at the first collective, or
  `Expected to have finished reduction in the prior iteration`.
- **Cause:** ranks take different code paths (for example a dataset raising
  on only one rank, or unused parameters in the model).
- **Propose:** look for an exception from one rank; for unused parameters,
  check the model's forward pass for branches.

## Data

### Missing files or empty filelist

- **Shows:** `.err`: `FileNotFoundError`, `ValueError: No prefixes found`,
  or the dataset's `RuntimeError: ... no usable sample found after 5
  prefixes`; `.out` full of `max_file_checks reached for prefix`.
- **Cause:** data paths in `training_input.tmpl` wrong for this machine,
  filelist out of date, or many simulations missing frames.
- **Propose:** re-run the path and filelist checks from
  `skills/yoke-launch-study/reference/launch_checks.md` (check 3); regenerate
  the filelists if needed.

## Resubmission chain

### Chain stopped without an error

- **Shows:** the last cycle has a checkpoint but no `restart_training_epoch`
  files, or has restart files but no next `.out`.
- **Causes, to check in order:**
  1. Training finished (`epoch ≥ total_epochs`): not a failure.
  2. Restart files exist but nothing ran: `sbatch` failed on the compute node.
     Look at the end of that cycle's `.out`/`.err` for `sbatch: error`.
     Compute nodes at some sites can't submit, or the account is out of hours.
  3. `--submissionType` mismatch: a shell-submitted study resubmits with
     `sbatch` unless its input file passes `--submissionType shell` (known
     issue 4 in `dev_plans/known_issues_from_skills_work.md`).
- **Propose:** fix the cause, then give the user the command to submit the
  existing restart script: `cd runs/study_NNN && sbatch
  studyNNN_restart_training_epochMMMM.slurm`.

### Restart from a checkpoint by hand

When a cycle died after its checkpoint was written but before resubmitting,
the restart files may be missing. The user can regenerate them, inside the
study directory, with:

```bash
<python> -c "from yoke.harnesses.base import HarnessStudy as H; \
print(H.continuation_setup('studyNNN_modelState_epochEEEE.pth', NNN, EEEE))"
```

`continuation_setup` writes the `restart_training_epoch{EEEE+1}` files and
prints the submission script name, which the user then submits. Propose this
command; do not run it, because it writes files inside `runs/`.

## Training behaviour (from the loss records)

### NaN or inf loss

- **Shows:** `nan`/`inf` in the record CSVs; loss suddenly jumps before.
- **Causes:** learning rate too high (check the warmup), NaNs in the data
  (fields NaN outside materials not replaced), division by small values in a
  relative loss (`eps`), or exploding gradients (no `grad_clip`).
- **Propose:** check the data's NaN handling (preflight on the offending
  sample), lower `ANCHOR_LR` or lengthen warmup, enable `grad_clip`, and
  restart from the last checkpoint before the jump.

### Loss flat from the start

- **Causes:** learning rate too low, or a scheduler stuck at zero LR; targets
  dominated by one channel (elementwise loss on unnormalized multi-scale data);
  frozen parameters.
- **Propose:** check the LR schedule against the step counts; consider
  `LpLoss` or normalization (`yoke-new-loss`).

### Validation diverges from training

- **Shows:** training loss falls while validation rises.
- **Cause:** overfitting (too little data or too large a model), or a
  train/validation split that differs in kind.
- **Propose:** a smaller model, more data, or regularization (weight decay);
  check the split by simulation.

### Loss scale looks "wrong"

- With `LpLoss` (no `1/M` averaging), values are often hundreds or more.
  This is expected; plot with a larger `--ylim`.

## Environment

### Import or version errors

- **Shows:** `.err`: `ModuleNotFoundError: No module named 'yoke'`, an
  `ImportError` from a renamed function, or a torch/CUDA mismatch
  (`CUDA driver version is insufficient`).
- **Cause:** the activation line in the SLURM template points at a different
  environment, or the study's copied script is older than the installed
  `yoke`.
- **Propose:** fix the activation line in the template (and in the study's
  `training_slurm.tmpl` copy for restarts); check `yoke.__file__` in that
  environment.
