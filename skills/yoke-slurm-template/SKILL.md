---
name: yoke-slurm-template
description: On a SLURM cluster login node, query the machine with read-only SLURM commands (sinfo, scontrol, sacctmgr show), reuse or save a machine profile, interview the user, and write a Yoke training_slurm.tmpl (and evaluation_slurm.tmpl) for a harness. Use when a Yoke harness needs a SLURM submission template for a machine, or the user asks to set up SLURM settings for Yoke.
---

# Create a Yoke SLURM Submission Template

You find out what the cluster offers and what the user may use, agree the
job settings with the user, and write the harness's SLURM template. Machine
facts are saved in a profile for reuse. Allocations are always re-queried.

**You never run `sbatch`, `salloc`, `srun`, `scancel`, or any command that
changes SLURM state.** Only the read-only queries in
`skills/yoke-slurm-template/reference/slurm_queries.md`. Show each command
before running it.

Read first:

- `skills/shared/interview_conventions.md` (follow it throughout).
- `skills/yoke-slurm-template/reference/slurm_queries.md`.
- `skills/yoke-slurm-template/reference/slurm_template_skeleton.md`.

Load when saving a profile:

- `skills/yoke-slurm-template/reference/machine_profile_template.md`
- `skills/shared/cards.md`

## Procedure

### 1. Detect the machine

- Ask which harness directory the template is for
  (`applications/harnesses/<name>/`), unless `yoke-new-harness` already
  said.
- Run `command -v sinfo scontrol sacctmgr`.
- **SLURM available**: get `ClusterName` (`scontrol show config | grep -i
  ClusterName`) and go to step 2.
- **GPU or CPU-only?** Once the partitions are known, check their GRES
  (`sinfo -o "%P %G"`). If the partition the user will train on has no GPUs
  (GRES `(null)`), write the **CPU-only template** from
  `slurm_template_skeleton.md`. It is for harnesses with
  `HarnessTrainer(..., device_type="cpu")`; confirm the harness uses that
  mode.
- **SLURM not available** (for example a laptop): say so and offer:
  1. if the user names the target machine and
     `skills/yoke-slurm-template/machines/<cluster>.md` exists, build the
     template from the profile. Allocations then can't be checked, so keep
     `<allocation>` as a CSV column and flag it in the card;
  2. copy and adapt the template of an existing harness that targets the same
     machine (`grep -l` the harness templates for the cluster name or
     account);
  3. stop, and resume on the cluster's login node.

### 2. Look for a saved profile

Check `skills/yoke-slurm-template/machines/<cluster>.md`.

- **Found**: show its summary and `last verified` date and ask whether to
  reuse it. If yes, take the machine facts from it, and only spot-check:
  run `sinfo -s` and confirm the profile's partitions and time limits still
  exist. Report any differences and offer to update the profile.
- **Not found** (or the user declines it): run the full queries in
  `slurm_queries.md` (sections 1–3, 5–7).

### 3. Query allocations (every time)

Run section 4 of `slurm_queries.md` for the current user, even when a
profile is reused. Allocations differ between users and change over time.
Never take accounts or QOS from the profile without this check.

### 4. Interview

Present a short summary table (partitions, node hardware, the user's
accounts and QOS with limits, debug options) with suggested defaults, then
ask, in one or two rounds:

- Partition for production runs.
- Account: keep `<allocation>` as a study-CSV column (recommended), or fix
  one account in the template.
- QOS, if needed.
- Wall time per job. One job runs `--cycle_epochs` epochs, then resubmits
  itself. Check it against the partition `MaxTime` and QOS `maxwall`.
- GPU template only: GPUs per node (default: all on the node; this becomes
  the CSV `NGPUS`), typical node counts (CSV `KNODES`), memory per GPU,
  binding flags, and `NCCL_SOCKET_IFNAME` (from the profile, or ask; see
  section 6 of the queries file).
- `--cpus-per-task` and `OMP_NUM_THREADS` (defaults from the node hardware or
  profile). For a CPU-only template: one task on one node, so these set the
  training threads. Small models are often fastest with only a few threads,
  even on a large node; say so, and suggest timing the smoke run.
- Environment activation lines (modules, venv or conda).
- Include a commented-out debug block? (Existing templates do.)
- Evaluation template too? (Only if the harness opts into post-training
  evaluation; also ask for the evaluation script name and wall time.)

### 5. Design and approval

Fill in the skeleton from `slurm_template_skeleton.md` and show the
**complete** template text (and the evaluation template, if any) in the
design notes (`dev_plans/<harness>_slurm.md`, or the harness's notes file).
**Wait for approval.**

### 6. Write and check

- Write `applications/harnesses/<name>/training_slurm.tmpl` (and
  `evaluation_slurm.tmpl`). If a file already exists, show a diff and confirm
  before overwriting.
- Check:
  - no `@@...@@` fill-in markers remain;
  - the required tokens `<studyIDX>`, `<epochIDX>`, `<KNODES>`, `<NGPUS>`,
    `<train_script>`, `<INPUTFILE>` are present (and `<allocation>` if
    chosen); a CPU-only template needs no `<KNODES>`/`<NGPUS>` and must not
    contain `--gpus-per-node` or a multi-task `srun`; the evaluation template
    has no `<epochIDX>`;
  - bash syntax on a token-substituted copy:
    `sed -E 's/<[A-Za-z_]+>/X/g' training_slurm.tmpl > /tmp/x.sh && bash -n /tmp/x.sh`.
- The full render check is `yoke-start-study --dryrun`, done by
  `yoke-new-harness`.

### 7. Save or update the machine profile

Following `machine_profile_template.md`, write or update
`skills/yoke-slurm-template/machines/<cluster>.md` with:

- machine facts (partitions, node hardware, recommended settings, NCCL
  interface or "CPU-only: no GPUs", debug pattern, quirks);
- environment hints, with personal paths replaced by placeholders;
- a dated row under "Allocations seen", keeping the re-query warning;
- `last verified:` today's date and the SLURM version.

Show the new profile, or the diff for an update, and **get approval before
writing**. Never write credentials or personal paths.

### 8. Hand off

Write the **Slurm card** (`skills/shared/cards.md`): machine, device mode
(`cuda` or `cpu`), template paths,
partition/QOS, wall time, the CSV keys the template needs (`KNODES`, `NGPUS`,
`train_script`, and `allocation` if used), env lines, and what is **not
verified**. A real job is the only full test of the NCCL interface (GPU
mode), thread performance (CPU mode),
binding flags, and environment, and this skill does not submit one. Return
to `yoke-new-harness` if it called you.
