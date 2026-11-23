---
name: yoke-launch-study
description: On the cluster login node, run read-only readiness checks for a Yoke harness (environment, template rendering, data paths, filelists), run the one-batch CPU --preflight, estimate GPU hours, give the user the exact yoke-start-study command, and - only if the user explicitly asks - run it for them (SLURM studies only). Never cancels or modifies jobs. Use when a harness is ready and the user wants to launch training, or asks "is this ready to run?".
---

# Prepare a Yoke Study for Launch

You check that a harness will run on this machine, hand the user the exact
command to launch it, and ask whether they want you to run it.

- **SLURM studies:** if the user explicitly says yes, you may run that one
  `yoke-start-study` command. It calls `sbatch` once per CSV row and returns.
  The jobs then live in the SLURM queue and resubmit themselves from the
  compute nodes (`HarnessStudy.submit_job`, `HarnessStudy.continuation_setup`),
  so nothing depends on this agent's shell or session.
- **Shell studies** (`--submissionType shell`): give the command only. Shell
  submission runs the job with `source`, in the calling shell, so the training
  would run inside the agent's shell and could die with the session.

Apart from that one command, you never run `sbatch`, `salloc`, `srun`,
`scancel`, `scontrol update`, or `yoke-start-study` without `--dryrun`.

Read first:

- `skills/shared/interview_conventions.md` (follow it throughout).
- `skills/yoke-launch-study/reference/launch_checks.md`.
- The progress file (`dev_plans/<project>_training.md`) and the harness's
  design notes and cards, if they exist.
- The machine profile in `skills/yoke-slurm-template/machines/`, if one
  exists for this cluster.

## Procedure

### 1. Orient

- Confirm which harness (`applications/harnesses/<name>/`), which CSV (start
  with the smoke CSV), and which Python environment.
- Confirm you are on the cluster where the job will run
  (`command -v sbatch`, `scontrol show config | grep -i ClusterName`). If not,
  say which checks can't be done here (data paths, environment, budget against
  allocations) and continue with the rest.

### 2. Readiness checks

Run checks 1–3 of `launch_checks.md` (environment, rendering, data and
filelists). Report each as pass or fail, with the evidence, and stop at the
first failure.

### 3. Preflight

Run check 4 of `launch_checks.md`: parameter count first, then ask where to run
it, then run it (or prepare the allocation command for the user), then read the
report. On failure, explain the cause in plain terms and route to the skill
in the failure table. Do not patch around it here.

### 4. Readiness summary

Show:

- checks passed, with the preflight report (batch shapes, parameters, loss);
- the GPU-hour (or, in CPU mode, node-hour) estimate (check 5) for the smoke
  CSV and the production CSV;
- what the run will produce, and where (`runs/study_NNN/`: `.out`/`.err` per
  cycle, record CSVs, `studyNNN_modelState_epochEEEE.pth` checkpoints,
  restart files);
- how to watch it: `squeue -u $USER`,
  `tail -f runs/study_001/study001_epoch*.out`, and
  `yoke-plot-loss --basedir ./runs --IDX 1`.

### 5. Give the command

Print the exact command, run from the harness directory:

```bash
cd applications/harnesses/<name>
yoke-start-study --csv <smoke.csv> --rundir ./runs --cpFile cp_files.txt \
    --submissionType slurm
```

Recommend the smoke CSV first, then the production CSV once the smoke runs
have completed at least one cycle and resubmitted.

Check that `./runs/study_NNN/` doesn't already exist for any of the CSV's
rows (check 7 of `launch_checks.md`). Relaunching into an existing study
directory overwrites its rendered files and mixes outputs. If any exist,
list them and stop until the user decides (a new CSV with new study IDs, or
a deliberate relaunch).

### 6. Offer to submit

For a **SLURM** study, ask whether the user wants you to submit it. Restate,
in the question:

- the exact command, and the directory it runs in;
- the CSV and its study IDs (one job per row);
- the GPU-hour or node-hour estimate, and the account (`allocation` column) to be
  charged;
- that you will run only this command, and won't cancel or change jobs
  afterwards.

Submit only on an explicit yes **to this question, now**. Approving the
design, the harness, or the preflight earlier doesn't count. If the answer is
no, or unclear, stop here; the user runs the command themselves. Each later
submission (for example the production CSV after the smoke CSV) needs its
own yes.

For a **shell** study, don't offer. Explain that shell submission would run the
training inside the agent's shell, and that the user should run the command in
a terminal they control (for example inside `tmux` or `screen`).

If the user says yes, follow check 7 of `launch_checks.md`:

1. Re-check that none of the study directories exist (step 5).
2. Run the command from the harness directory, exactly as shown, once. Do
   not retry on failure: report the error and stop.
3. Check the output: one `sbatch`-style `Submitted batch job <id>` line per
   row, and no errors. Report the job IDs.
4. Run `squeue -u $USER` once, filtered by the job-name prefix, and show
   the result. Don't poll. Point to `yoke-run-doctor` for later checks.

### 7. Write the Run card

Write the **Run card** (`skills/shared/cards.md`) into the progress file (or
the harness notes file, if there is no progress file): harness, CSV, run dir,
command, preflight result, job-name prefix (from the SLURM template), expected
outputs, budget, and today's date. Under `submitted:`, record either
"by agent at user's request, <date>, job IDs <ids>" or "not by agent: the
user will run the command".

Tell the user that in any later session they can say, for example, "check on
study 1 of `<harness>`" to start `yoke-run-doctor`, which reads this card.

### 8. Summarize

Checks run and their results, what was **not** checked (GPU and NCCL
behaviour and multi-node runs in GPU mode, thread performance in CPU mode,
and convergence: only a real job shows these),
the command, and whether it was submitted (by whom, and the job IDs).
