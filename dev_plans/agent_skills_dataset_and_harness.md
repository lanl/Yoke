# Agent Skills: Guided Dataset, Training-Step, SLURM, and Harness Creation

## Revision history

- **Rev 1** (2026-11-10): initial draft, two skills (dataset, harness).
- **Rev 2** (2026-11-10), after user review:
  - Skills live in an agent-neutral top-level `skills/` folder, with
    symlinks from `.claude/skills/` (and other agents' skill folders) so
    Claude Code, Codex, and opencode all read one copy.
  - The dataset skill is format-general (npz, HDF5, txt/dat, CSV, TIFF and
    other images, ...), not npz-focused.
  - No local training smoke run anywhere. Verification stops at unit tests,
    lint, and `yoke-start-study --dryrun` rendering.
  - Two new skills: **`yoke-new-training-step`** (build a datastep + epoch
    pair when no existing pair fits the batch contract) and
    **`yoke-slurm-template`** (query the local SLURM installation with
    read-only commands and write the SLURM template).
  - CI test for skills: kept as an open question, with what it would check
    spelled out (Section "Verification").
- **Rev 3** (2026-11-10), after user review:
  - Symlinks only for Claude Code (`.claude/skills/`) for now; Codex and
    opencode symlinks are deferred.
  - `yoke-slurm-template` saves **machine profiles** to
    `skills/yoke-slurm-template/machines/<cluster>.md` and reuses them, since
    the same company machines recur. Allocations (accounts/QOS) change per
    user and over time, so they are recorded only as dated hints and are
    always re-queried for the current user.
- **Rev 4** (2026-11-10), after user review:
  - The skills CI test is in scope: `tests/skills/test_skill_files.py`,
    run by the existing pytest step in CI (Section "Skills CI test"). No
    open questions remain.

## Goal

Give agentic coding harnesses (Claude Code, Codex, opencode, and any agent
that reads `AGENTS.md`) four **skills** that walk a user through adding new
training capability to Yoke by interview:

| Skill | Produces |
| --- | --- |
| `yoke-new-dataset` | A `Dataset` class in `src/yoke/datasets/` + tests |
| `yoke-new-training-step` | Train/eval datasteps + epoch function in `src/yoke/utils/training/` + tests |
| `yoke-slurm-template` | `training_slurm.tmpl` (and `evaluation_slurm.tmpl`) for the machine it runs on |
| `yoke-new-harness` | An `applications/harnesses/<name>/` directory built on `HarnessTrainer` |

The skills are agent instructions (Markdown), not Python code. They provide
(a) a fixed sequence of questions so key decisions are always asked rather
than guessed, and (b) accurate, Yoke-specific reference material so the agent
writes code that fits existing conventions on the first try.

Each skill can be used on its own. When used together, the harness skill is
the orchestrator: it calls the others when needed, and they pass results to
each other through short **cards** (Section "Handoffs").

```
yoke-new-dataset ──(dataset card)──┐
                                   ▼
               contract matches an existing epoch fn?
                    │ no                       │ yes
                    ▼                          │
     yoke-new-training-step ──(step card)──────┤
                                               ▼
                                       yoke-new-harness
                                               │ submission = slurm, new machine?
                                               ▼
                                     yoke-slurm-template ──(slurm card)──► harness
```

## Background: what the skills must know (verified in the repo)

- **Batch contracts.** A dataset is only usable if its `__getitem__` tuple
  matches what the datastep unpacks. Current contracts
  (`src/yoke/utils/training/datastep/*.py`):

  | Datastep family | Tuple unpacked from `data` | Line |
  | --- | --- | --- |
  | `array_output`, `scalar_output` | `(inpt, truth)` | `array_output.py:37`, `scalar_output.py:35` |
  | loderunner (1-step) | `(start_img, end_img, Dt)` | `loderunner.py:199` |
  | loderunner cylex | `(start_img, channel_map, end_img, channel_map, Dt)` | `loderunner.py:264` |
  | loderunner scheduled | `(img_seq, Dt)` | `loderunner.py:116` |
  | loderunner 2-frame / 2-frame LpLoss | `(input_frames, end_img, lead_times)` | `loderunner.py:351`, `:484` |
  | loderunner 2-frame varchan | `(input_frames, end_img, lead_times, channel_map, dataset_id)` | `loderunner.py:1025` |
  | diffusion loderunner | `(x, y_tau, noise, lead_times, tau)` | `diff_loderunner.py:48` |
  | lsc_reward | `(state_y, stateH, targetH, reward)` | `lsc_reward.py:37` |
  | lsc_policy | `(state_y, stateH, targetH, x_true)` | `lsc_policy.py:38` |

  Epoch functions (`src/yoke/utils/training/epoch/*.py`) select datasteps;
  the loderunner family dispatches through the `DATASTEP_FN` table keyed by
  the `dataset` tag (`epoch/loderunner.py:55`), and its datasteps must stay
  re-exported in `__all__` (`epoch/loderunner.py:31`).
- **Datastep/epoch tests** already exist under
  `tests/utils/training/{datastep,epoch}/` and are the model for new ones.
- **Dataset idioms.** Random-sampling datasets with `__len__ = int(1e6)` that
  draw a simulation prefix from a filelist and retry missing files
  (`pagosa_2frame.py:211`, `lsc_dataset.py:494`); fixed-index datasets
  (`lsc_dataset.py:134`); NaN-to-zero reads (`pagosa_2frame.py:125`);
  `MixedDataSet` for weighted mixtures (`mixed_dataset.py`). Filelists live in
  `applications/filelists/`, passed relative to `--FILELIST_DIR`.
- **Readers already available** (`pyproject.toml` dependencies): numpy,
  `h5py`, `pandas`, and image I/O via `scikit-image`/`matplotlib`. Anything
  else (for example `tifffile`, `netCDF4`) is a new dependency and needs the
  user's approval.
- **Harness idioms.** Thin `HarnessTrainer` wrapper
  (`src/yoke/harnesses/trainer.py:181`) injecting `model_builder` (usually
  `yoke.utils.builders.build_from_checkpoint`), `dataset_builder`,
  `epoch_fn`, and optionally `optimizer_builder`, `scheduler_builder`,
  `loss_builder`, `epoch_kwargs`, `TrainerHooks` (EMA via
  `yoke.utils.ema.make_ema_hooks`). Argparse comes from the `yoke.helpers.cli`
  builders. Newest reference: `applications/harnesses/ch_ldrViT_lsc_cyl/`.
- **SLURM templates.** Across the 19 existing submission templates the core
  is shared: `--nodes=<KNODES>`, `--ntasks-per-node=<NGPUS>`,
  `--gpus-per-node=<NGPUS>`, `--output/--error=study<studyIDX>_epoch<epochIDX>.*`,
  `MASTER_ADDR`/`MASTER_PORT` setup, `NCCL_SOCKET_IFNAME`, env activation, and
  `srun python -u <train_script> @<INPUTFILE>`
  (`ch_ldrViT_lsc_cyl/training_slurm.tmpl`). What varies by machine:
  `--account`, `--partition`, `--qos`, `--time`, `--cpus-per-task`,
  `--mem-per-gpu`, binding flags (`--gpu-bind`, `--mem-bind`,
  `--gres-flags`), the NCCL interface, and `OMP_NUM_THREADS`.
- **Models available** (`src/yoke/models/`): `LodeRunner`, `LodeRunnerViT`
  (`vit/swin/bomberman.py`), `DiffusionLodeRunner`, `Image2ScalarCNN`,
  `Image2VectorCNN`, `hybrid2vectorCNN`, `jekelCNNsurrogate`,
  `tCNNsurrogate`, `gaussian_policyCNN`, and others.

## Proposed file layout

```
skills/                                 # canonical, agent-neutral copy
  README.md                             # what the skills are, how agents find them
  shared/
    batch_contracts.md                  # epoch fn -> datastep -> tuple, shapes, dtypes
    interview_conventions.md            # ask in rounds, record answers in a
                                        #   dev_plans/ file, approval gate, cards
    cards.md                            # card formats (dataset, step, slurm)
  yoke-new-dataset/
    SKILL.md
    reference/
      format_inspection.md              # cheap inspection per format (below)
      dataset_patterns.md               # sampling, filelists, retries, NaNs,
                                        #   channel maps, normalization
      dataset_test_patterns.md          # synthetic files in tmp_path, per format
  yoke-new-training-step/
    SKILL.md
    reference/
      datastep_epoch_anatomy.md         # what a datastep/epoch pair must do:
                                        #   DDP gather, record CSVs, LR step,
                                        #   EMA/grad clip, train_per_val
      training_step_test_patterns.md    # tiny model, CPU, mocked dist
  yoke-slurm-template/
    SKILL.md
    reference/
      slurm_queries.md                  # read-only commands and how to read them
      slurm_template_skeleton.md        # required placeholders + shared core
      machine_profile_template.md       # format for machines/<cluster>.md
    machines/
      <cluster>.md                      # one saved profile per machine
                                        #   (created by the skill, not by hand)
  yoke-new-harness/
    SKILL.md
    reference/
      harness_checklist.md
      model_catalog.md                  # model -> ctor args, compatible epoch fns
      study_csv_and_templates.md        # <KEY> rules, optional blocks, tokens

.claude/skills/yoke-new-dataset        -> ../../skills/yoke-new-dataset   (symlink)
.claude/skills/yoke-new-training-step  -> ../../skills/yoke-new-training-step
.claude/skills/yoke-slurm-template     -> ../../skills/yoke-slurm-template
.claude/skills/yoke-new-harness        -> ../../skills/yoke-new-harness
                                        # Codex/opencode symlinks: deferred

AGENTS.md                               # new short "Agent skills" section

tests/skills/
  __init__.py
  test_skill_files.py                   # skills CI test (see below)
```

Notes on the layout:

- Symlinks are relative and committed to git, so they survive clones.
  Each skill directory is symlinked as a whole (not `shared/`), so every
  agent sees the same four skills.
- Because a skill may be read through a symlink, `SKILL.md` files refer to
  shared material by **repo-root paths** (`skills/shared/batch_contracts.md`),
  not `../shared/...`. Relative paths through symlinks are fragile.
- Each `SKILL.md` has YAML frontmatter with `name` (equal to the directory
  name) and a `description` saying when to use it. That is the Claude Code
  format; keeping the skills plain Markdown in `skills/` should make adding
  Codex/opencode symlinks later a small change.

## Common interview conventions (all four skills)

- Ask a few questions at a time, in rounds, and give a recommended default
  where the repo suggests one.
- Record answers as they come in a `dev_plans/<thing>_<skill>.md` file, so the
  decisions survive context compaction and the user can review them.
- Before writing any code, write the design into that file and **stop for
  user approval** (the user's global rule).
- Inspect data and machines only with cheap, read-only operations.
- Never add dependencies, submit jobs, or start training without approval.
  The skills themselves never submit or train.
- End with a card (Section "Handoffs") and a summary: files, verification,
  what was not verified.

## Skill 1: `yoke-new-dataset`

1. **Orient.** Read `AGENTS.md`, `skills/shared/batch_contracts.md`, and list
   `src/yoke/datasets/`. Ask whether an existing dataset nearly fits
   (subclassing is preferred, as `LSC_2frame_canonical_DataSet` does).
2. **Data inventory.** Ask: file format(s); directory layout and filename
   pattern (one file per sample, per time step, per simulation, or one big
   file); where a few sample files are (for example `data_subsamples/`); any
   design/metadata table; known gaps or corrupt files; approximate size per
   file and in total.
3. **Inspect samples cheaply**, using the recipe for the format from
   `format_inspection.md`, and report what was found:
   - **npz/npy**: `np.load(..., mmap_mode="r")` / `NpzFile.files`; keys,
     shapes, dtypes, NaN presence on a slice.
   - **HDF5**: `h5py` walk of groups/datasets, shapes, dtypes, attributes;
     read only small slices.
   - **CSV/txt/dat**: read the first lines raw (to find header, delimiter,
     comment character, fixed-width columns), then `pandas.read_csv(...,
     nrows=...)`; check for leading spaces in headers/values (seen in
     `design_cx241203_MASTER.csv`).
   - **TIFF/PNG/other images**: header-only metadata (size, bands, bit
     depth, page count for multi-page TIFF) with an installed reader;
     if none can read it, ask before adding one.
   - **Other/unknown** (netCDF, binary, VTK, ...): ask the user how it is
     read today; propose a reader; any new package needs approval.
   The agent confirms its reading of the data with the user before going on.
4. **Learning task.** Ask: input, target, time structure (single step,
   multi-frame, sequence; lead time from index or a stored time field), and
   the intended model family. **Propose a batch contract** from
   `batch_contracts.md` and the epoch function it implies. If none fits,
   record that and hand off to `yoke-new-training-step` after the dataset is
   designed (the new contract is part of the dataset card).
5. **Sampling and splits.** Random-sampling vs. indexed; how samples are
   split (existing filelists, or new ones, possibly using
   `applications/makefilelists.py`); offset rules; behaviour on missing
   files; per-worker RNG.
6. **Channels and preprocessing.** Which fields become channels and in what
   order (`channel_map` / `default_vars`); NaN handling; geometry (half vs.
   reflected image, cropping, resizing); normalization (precomputed stats
   file, as in `lsc_normalization.py`, or none); transforms
   (`datasets/transforms.py`).
7. **Design + approval.** Class name, module, ctor signature, return tuple
   with shapes and dtypes, reader choice, test plan.
8. **Implement** the module (Google docstrings, full annotations, 89-col,
   double quotes) and `tests/datasets/test_<name>.py`. Tests build tiny
   synthetic files **in the same format** in `tmp_path` and check shapes,
   dtypes, the contract, missing-file handling, and seeded determinism.
9. **Verify**: `pytest -Werror` on the new test file, `ruff check`,
   `ruff format --check`. With the user's OK, load 1–2 real samples and print
   shapes and value ranges (a data read, not training).
10. **Emit the dataset card.** Offer to continue with `yoke-new-harness`, or
    with `yoke-new-training-step` if no contract matched.

## Skill 2: `yoke-new-training-step`

Used when a dataset's batch contract (or a needed behaviour, such as a new
loss reduction) has no matching datastep + epoch pair. Called from the dataset
or harness skill, or directly.

1. **Orient.** Read `batch_contracts.md`, `datastep_epoch_anatomy.md`, and the
   closest existing pair the agent can find (it names it and explains how the
   new case differs). Prefer **extending** an existing family (a new
   `DATASTEP_FN` entry in `epoch/loderunner.py`, or a new datastep used by an
   existing epoch function) over a new epoch function; prefer a new epoch
   function over a new module.
2. **Interview.** Ask:
   - the batch tuple (from the dataset card if there is one);
   - how the model is called (forward signature, extra inputs such as
     `channel_map` or lead times);
   - loss shape and reduction: what is backpropagated, what one value per
     sample is recorded (as in the LpLoss amendment of
     `lsc_cyl_joint_ldrViT_harness.md`), and what to raise on a wrong shape;
   - DDP only, or also single-process; extra record columns (for example a
     dataset tag); EMA, grad clip, scheduled sampling, LR stepping per batch
     or per epoch;
   - what the eval step records.
3. **Design + approval.** Function names following the existing naming
   (`train_DDP_<family>_datastep`, `eval_DDP_<family>_datastep`,
   `train_DDP_<family>_epoch`), signatures, where each goes, which
   `DATASTEP_FN`/`__all__` entries change, and the record CSV columns.
4. **Implement** the datastep pair, the epoch function or table entry, and
   tests in `tests/utils/training/{datastep,epoch}/` modelled on the existing
   ones: a tiny model on CPU, mocked or single-rank `torch.distributed`,
   checks on the backward reduction, the recorded values, the CSV contents,
   and a wrong-shape error. These are unit tests, not a training run.
5. **Verify**: `pytest -Werror` on the new and neighbouring datastep/epoch
   tests (to catch regressions in shared tables), `ruff`.
6. **Emit the step card.**

## Skill 3: `yoke-slurm-template`

Run on a cluster login node. It finds out what the machine offers using
**read-only** SLURM queries, asks the user to choose, and writes the
template. It never runs `sbatch`, `salloc`, or `srun`.

1. **Detect.** Check that `sinfo`/`scontrol`/`sacctmgr` exist. If not (as on a
   laptop), say so and offer three fallbacks: use a saved machine profile
   (step 1a) if the user names the target machine; copy and adapt the
   closest existing harness template; or stop until it is run on the
   cluster.
1a. **Look for a saved profile.** Get the cluster name
   (`scontrol show config | grep -i ClusterName`) and check
   `skills/yoke-slurm-template/machines/<cluster>.md`. If it exists, show its
   summary and its `last verified` date and ask whether to reuse it. When it
   is reused, the **machine-level** facts (partitions, node hardware, NCCL
   interface, binding flags, typical env setup) are taken from the profile
   and only spot-checked (`sinfo -s`, to confirm the partitions still
   exist). **Allocations are always re-queried** for the current user
   (step 2, accounts and QOS), because they differ between users and change
   over time. If the spot check finds differences, the agent reports them and
   offers to update the profile.
2. **Query** (commands in `slurm_queries.md`; the agent shows each one before
   running it). With a reused profile, only the accounts/QOS queries run:
   - cluster name: `scontrol show config | grep -i ClusterName`;
   - partitions with time limits, nodes, GRES: `sinfo -s`,
     `sinfo -o "%P %a %l %D %c %m %G"`, `scontrol show partition`;
   - one node's hardware (GPUs/type, CPUs, sockets, memory):
     `scontrol show node <one node of the chosen partition>`;
   - the user's accounts and QOS:
     `sacctmgr -nP show assoc user=$USER format=account,partition,qos`,
     `sacctmgr -nP show qos format=name,maxwall,maxtrespu`;
   - reservations (for debug variants): `scontrol show reservation`;
   - network interfaces for `NCCL_SOCKET_IFNAME`: ask the user, or suggest
     `ip link show` (noting the login node may differ from compute nodes).
3. **Interview.** Present a short summary and ask: partition, account (or
   keep `<allocation>` as a CSV column, as `ch_ldrViT_lsc_cyl` does), qos,
   wall time per cycle (checked against the partition/QOS limit), GPUs per
   node (default: all on the node), CPUs per task and `OMP_NUM_THREADS`
   (default: node CPUs / GPUs), memory per GPU, binding flags, the
   environment activation lines (conda vs. venv path), and whether to include
   a commented-out debug block (as the existing templates do).
4. **Design + approval.** Show the full rendered template.
5. **Write** `training_slurm.tmpl` from `slurm_template_skeleton.md`, keeping
   the required placeholders: `<studyIDX>`, `<epochIDX>`, `<KNODES>`,
   `<NGPUS>`, `<train_script>`, `<INPUTFILE>`, and optionally
   `<allocation>`. If the harness uses evaluation, also write
   `evaluation_slurm.tmpl` (single GPU, `<STEM>` naming, no DDP).
6. **Verify**: `bash -n` on a copy with placeholders replaced by dummy
   values (syntax only), and a check that every placeholder the harness
   pipeline needs is present. The real render check happens in the harness
   skill's `--dryrun`.
7. **Save or update the machine profile** (with the user's approval, shown as
   a diff when updating) in
   `skills/yoke-slurm-template/machines/<cluster>.md`, following
   `machine_profile_template.md`:
   - **Machine facts** (stable; reused): cluster name and aliases; login vs.
     compute node notes; partitions with purpose (production/debug), time
     limits, and node counts; per-node hardware (GPU type and count, CPUs,
     sockets, memory); recommended `--cpus-per-task`, `OMP_NUM_THREADS`,
     `--mem-per-gpu`, binding flags; `NCCL_SOCKET_IFNAME`; the debug
     partition/qos/reservation pattern; known quirks.
   - **Environment hints** (semi-stable): module loads and the shape of the
     activation line (for example `source $HOME/<venv>/bin/activate`), with
     user-specific paths left as placeholders.
   - **Allocations seen** (volatile; never trusted): a dated list of
     account/QOS names observed, labelled "as of YYYY-MM-DD, for one user;
     re-query with `sacctmgr` before use". The template itself keeps
     `<allocation>` as a CSV column by default, so a changed allocation only
     means editing the CSV, not the template.
   - `last verified: YYYY-MM-DD` and the SLURM version (`sinfo --version`).
   No credentials, tokens, or personal paths are written to the profile.
8. **Emit the slurm card** (the `<KEY>`s the template needs from the CSV).

## Skill 4: `yoke-new-harness` (orchestrator)

1. **Orient.** Read `AGENTS.md` Sections 4–5a, `harness_checklist.md`, and any
   cards. Propose the closest existing harness to start from.
2. **Dataset.** Which class (card or existing); which ctor args become CLI
   args; filelists per split; mixing via `MixedDataSet`. If no class exists
   yet, offer `yoke-new-dataset`.
3. **Model.** Which model from `model_catalog.md`; which architecture args
   are fixed in `make_model_args`, exposed as CLI args, or swept in the CSV;
   fresh / continuation / pretrained-finetune (the last needs a bespoke
   `model_builder`).
4. **Training recipe.** Epoch function: the agent **checks the dataset's
   contract against it** using `batch_contracts.md`; on a mismatch it stops
   and offers `yoke-new-training-step`. Then loss, optimizer, scheduler, EMA,
   grad clip, `epoch_kwargs`, total/cycle epochs, batches per epoch, batch
   size.
5. **Submission.** SLURM or shell. For SLURM: reuse a template from an
   existing harness on the same machine, or run `yoke-slurm-template`.
   Machine-specific data/filelist paths go only into the harness's `.tmpl`
   files, never into `src/yoke`.
6. **Study design.** Which `<KEY>`s vary, the CSV rows (start with a small
   smoke-test CSV, as `ch_ldrViT_lsc_cyl` does), and whether to opt into
   post-training evaluation (eval templates + `evaluate_after_training=True`).
7. **Design + approval.**
8. **Implement**: train script, `training_input.tmpl` with the
   `# <<optional:CONTINUATION>>` block, submission template, `cp_files.txt`,
   CSV, `README.md`.
9. **Verify** (no jobs submitted, nothing trained):
   - `ruff check` / `ruff format --check` on the train script.
   - `yoke-start-study --csv <csv> --rundir <tmp dir> --cpFile cp_files.txt
     --submissionType <type> --dryrun`, then grep the rendered files for
     leftover `<[A-Za-z_]+>` placeholders.
   - Parse each rendered `@`-file with the script's `parser` (import the
     module, call `parser.parse_args(["@file"])`; the `__main__` guard means
     nothing runs) to catch argument typos and type errors.
   - Check every CSV column is used by some template, and every template
     `<KEY>` has a CSV column or is a known late-bound token.
10. **Summarize**, including the exact `yoke-start-study` command to launch
    for real.

## Handoffs (cards)

Formats live in `skills/shared/cards.md`. Each card is written into the
skill's dev_plans file and shown in chat.

```
Dataset card
- class:        yoke.datasets.<module>.<ClassName>
- ctor args:    name: type = default   (path args marked)
- returns:      (input_frames[2,C,H,W] float32, end_img[C,H,W], lead_times[2])
- contract:     loderunner 2-frame | NEW (see tuple above)
- channels:     ordered list / default_vars
- image size:   (H, W); half_image?
- __len__:      int(1e6) random sampling | N
- filelists:    applications/filelists/<...>_{train,validation,test}.txt
- constraints:  e.g. batch_size must be 1

Step card
- epoch fn:     yoke.utils.training.epoch.<module>.<fn>
- dataset tag:  <DATASTEP_FN key>, if table-dispatched
- loss shape:   e.g. (B, C) required
- epoch_kwargs: name: meaning
- records:      CSV columns

Slurm card
- machine:      <ClusterName>, partition, qos
- template:     path to training_slurm.tmpl (+ evaluation_slurm.tmpl)
- CSV keys:     KNODES, NGPUS, allocation, ...
- env:          activation line(s)
```

## Steps (implementation order)

1. `skills/shared/batch_contracts.md`: read every datastep and epoch function
   and fill in shapes, dtypes, and which epoch function calls which datastep.
   This is the most error-prone file, so it comes first and is checked
   against the source.
2. `skills/shared/interview_conventions.md`, `skills/shared/cards.md`.
3. `yoke-new-dataset` (reference files, then `SKILL.md`).
4. `yoke-new-training-step` (reference files from the existing datastep/epoch
   code and tests, then `SKILL.md`).
5. `yoke-slurm-template` (skeleton from the 19 existing templates; queries;
   `machine_profile_template.md`). No machine profiles are written in this
   step; the first ones are created when the skill is run on a cluster.
6. `yoke-new-harness` (checklist; model catalog with ctor signatures read
   from source; template/CSV rules from `helpers/strings.py` and
   `harnesses/base.py`).
7. `skills/README.md`, symlinks in `.claude/skills/` (Codex/opencode
   deferred), and the `AGENTS.md` section.
8. `tests/skills/test_skill_files.py` (Section "Skills CI test"). Written
   alongside steps 1–7 in practice, since it checks their output, and run
   after each step.
9. Rehearse (Verification) and revise.

## Skills CI test

`tests/skills/test_skill_files.py`, under `tests/` so the existing CI step
(`pytest -v ... -Werror` in `.github/workflows/yoke_install_test_lint.yml`)
runs it and the existing "Lint tests" step lints it. No workflow change is
needed. It does not check Markdown formatting (ruff does not lint Markdown);
it checks that the skills are well formed and still match the code.

Checks, each a separate parametrized test so a failure names the file:

1. **Skill discovery.** Every directory in `skills/` except `shared/` has a
   `SKILL.md`. The expected four skills are all present.
2. **Frontmatter.** Each `SKILL.md` starts with a `---` delimited block with a
   `name` equal to its directory name and a non-empty `description`.
   Parsed with a few lines of plain Python (key: value lines), not PyYAML:
   PyYAML is importable in the venv only as a transitive dependency of
   `lightning`, and the project does not declare it.
3. **Symlinks.** Each `.claude/skills/<name>` is a symlink, relative, and
   resolves to `skills/<name>`; every skill in `skills/` has one.
4. **Repo paths exist.** Every backticked repo-relative path in a skill
   Markdown file (`src/...`, `applications/...`, `tests/...`, `skills/...`,
   `docs/...`, `dev_plans/...`, `AGENTS.md`), with any `:line` suffix
   stripped, exists. Paths containing a placeholder (`<...>`, `*`, `###`)
   are skipped, since they describe files the skills create.
5. **Yoke names import.** Every backticked dotted name starting with
   `yoke.` resolves: import the longest module prefix with `importlib`, then
   `getattr` the rest. This is the main guard against the skills going stale
   when code is renamed.
6. **Batch contracts match the source.** For each row of
   `skills/shared/batch_contracts.md` the table also names the datastep
   function; the test reads that function's source (`inspect.getsource`) and
   checks that the tuple unpacked from `data` (for example
   `input_frames, end_img, lead_times = data`) matches the row. This keeps
   the most important reference file honest. The table format is fixed in
   step 1 so it can be parsed.
7. **Machine profiles** (only if any exist in `machines/`): each has a
   `last verified: YYYY-MM-DD` line and an "Allocations seen" section that
   carries the re-query warning.

Notes:

- Fast and CPU-only: reads files and imports modules; no data, no SLURM.
- Kept `-Werror` clean; Google docstrings and annotations as for all tests.
- The extraction regexes are kept simple and live at the top of the file with
  comments, so a false positive is easy to diagnose. A path that is
  intentionally illustrative should use a placeholder so it is skipped.

## Verification

- **Rehearsal of each skill** in a fresh agent session on a scratch branch or
  worktree; generated files are thrown away, and only lessons go back into
  the skills:
  - `yoke-new-dataset` on `data_subsamples/` as if the LSC 2-frame data were
    new; compare with `LSC_2frame_canonical_DataSet`. Plus a second
    rehearsal on a non-npz format (for example a small synthetic HDF5 or
    CSV file) to exercise the general path.
  - `yoke-new-training-step` re-deriving the varchan 2-frame pair; compare
    with `train_/eval_DDP_loderunner_2frame_varchan_datastep`.
  - `yoke-new-harness` re-deriving `ch_ldrViT_lsc_cyl`, through `--dryrun`.
  - `yoke-slurm-template`: **cannot be run here** (no SLURM on this Mac). You
    would rehearse it on a cluster login node (for example Chicoma) and
    compare its output with `ch_ldrViT_lsc_cyl/training_slurm.tmpl`. Run it
    twice: first to create the machine profile, then again to check that the
    profile is reused and only allocations are re-queried.
- Check that each skill is found by Claude Code through the symlinks.
- `pytest -Werror tests/skills` passes, and fails as expected when
  deliberately broken (temporarily misspell a `yoke.` name, a path, and a
  batch-contract tuple in a scratch copy, then restore).
- `ruff check`, `ruff check --preview`, `ruff format --check --diff` on
  `tests/skills`, and `pytest -Werror` on the whole repo.

## Open questions for the user

None. Resolved: Codex/opencode symlinks deferred (Rev 3); machine profiles
saved, with allocations treated as volatile (Rev 3); skills CI test added
(Rev 4).

## Implementation status (2026-11-10)

Steps 1–8 are implemented. Step 9 (rehearsals in fresh agent sessions) is
**not done yet**; it needs the user (and a cluster login node for
`yoke-slurm-template`).

Small additions beyond the plan:

- `skills/yoke-slurm-template/machines/README.md` explains the folder. The
  profile check skips it.
- An extra CI check, `test_all_public_datasteps_listed`, fails if a public
  datastep is added without a row in `batch_contracts.md`.
- The path check also covers the root files `AGENTS.md` and `pyproject.toml`.
- `slurm_template_skeleton.md` uses `@@NAME@@` for values the agent fills in,
  to keep them apart from Yoke's `<KEY>` tokens. The SLURM skill checks that
  none remain.

Verified:

- `pytest -Werror tests/skills`: 80 passed, 1 skipped (no machine profiles
  exist yet).
- Deliberately broken copies failed as expected, and the originals were
  restored: a swapped tuple, a misspelled `yoke.` name, a missing path, and a
  profile without its date or warning.
- Full suite `pytest -Werror`: 794 passed, 1 skipped (before
  `tests/skills/__init__.py` got its docstring; the skills tests were rerun
  after).
- `ruff check`, `ruff check --preview`, `ruff format --check` on
  `tests/skills`: clean.
- The harness skill's verification recipe (step 9.2–9.4) was run against
  `ch_ldrViT_lsc_cyl` with `--dryrun` into a temp dir: it rendered, left no
  tokens in the START files, and its parser accepted `study001_START.input`.
