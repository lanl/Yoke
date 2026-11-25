# AGENTS.md — Orientation for AI Agents Working on Yoke

This file brings an agent up to speed on Yoke's purpose, structure, conventions, and
workflows. Read it fully before making changes.

---

## 1. What Yoke is

**YOKE** (Yielding Optimal Knowledge Enhancement) is a PyTorch prototyping, training, and
testing harness developed at LANL for multi-physics / multi-material ML applications
(projects: ArtIMis, ASC-PEM-EADA).

**Core philosophy:** define models, datasets, losses, metrics, and training/eval routines as
*modular, reusable, installable* Python components under `src/yoke`, then *codify their use*
as **harnesses** under `applications/harnesses`. A harness pins together a training script,
its hyperparameters, and its job-submission configuration so that a training/eval study is
**reproducible, trackable, and easy to vary**.

Data is **not** stored in the repo. Data locations are passed as command-line arguments to
the training/eval scripts.

---

## 2. Repository layout

Only **`./src`** is the installable package. Everything else is project material but not an
installable module.

```
src/yoke/                      # THE installable package
  __init__.py
  datasets/                    # torch Dataset classes (lsc_, jwl_, nestedcyl_, diffusion_, load_npz_, transforms)
  models/                      # torch models (CNNmodules, hybridCNNmodules, policy/surrogate CNNs, mnist_model, vit/)
  losses/                      # masked_loss.py, NormMSE.py
  metrics/                     # shaped_charge_metrics.py
  helpers/                     # cli.py (argparse builders), strings.py (template subst),
                               #   logger.py, training_design.py
  utils/                       # checkpointing.py, dataload.py, parallel.py, parameters.py, ema.py
    training/
      datastep/                # per-batch step fns (loderunner, lsc_policy, lsc_reward, scalar_output, array_output, diff_loderunner)
      epoch/                   # per-epoch train/val fns (same family as datastep/)
    diffusion/
  lr_schedulers.py             # e.g. ConstantWithWarmupScheduler
  scheduled_sampling.py
  cli/                         # console-script entry points
    start_study.py             # `yoke-start-study` launcher
  harnesses/                   # HarnessStudy base class (study workflow + continuation)
    base.py

applications/                  # NOT installable; scripts + study definitions
  harnesses/                   # one subdir per study (see Section 4)
  evaluation/  viewers/  normalization/  filelists/
  makefilelists.py

tests/                         # pytest; mirrors src layout (datasets/, models/, helpers/, utils/, ...)
docs/                          # sphinx (source/, Makefile)
docker/                        # Dockerfile for containerized builds
dev_plans/                     # design/planning docs (markdown)
skills/                        # agent skills (Section 6a); .claude/skills/ links here
```

---

## 3. Environment, install, and tooling

- **Installed interpreter:** Yoke is installed in a dedicated Python environment (typically a
  conda env). **Ask the user for the path to the Python in which Yoke is installed** before
  running Yoke, its tests, or its CLI.
- **Build backend:** `flit` (`pyproject.toml`). Python `>=3.11`.
- **Note on PyTorch:** conda uses `pytorch`; pip/flit use `torch`/`torchvision`/`torchaudio`.
  Yoke is primarily used via conda envs, so torch deps are often handled separately.
- **Dev install:** `flit install --symlink --deps develop` (editable + test/dev deps).

### Testing (run from repo root)
```bash
pytest -Werror
pytest --cov --cov-report term-missing
pytest --cov=. --cov-report=html      # HTML report
```
CI (`.github/workflows/yoke_install_test_lint.yml`) runs, essentially:
`pytest -v --cov=yoke -Werror`, then `ruff` checks. **Tests must pass with `-Werror`** (no
warnings), and coverage is tracked via Coveralls.

### Linting / formatting (must pass in CI)
```bash
ruff check
ruff check --preview
ruff format --check --diff
```
Auto-fix with `ruff check --fix` and `ruff format`.

**Ruff config (pyproject.toml):** line-length **89**, 4-space indent, double quotes.
Enabled rule families: `E, F, D, UP, W, ANN`. Docstring convention: **google**. This means:
- **Every public module/class/function needs a docstring** (`D`).
- **Type annotations are required** on functions/args (`ANN`).
- Keep lines <= 89 chars.

CI lints `src` (non-blocking/`continue-on-error` for the main package preview) plus
`applications/evaluation`, `applications/filelists`, `applications/normalization`,
`applications/viewers`, `tests`, and scheduled sampling (blocking). Keep new code clean
everywhere.

---

## 4. How a harness works (the central workflow)

A harness directory (e.g. `applications/harnesses/ch_DDP_loderunner/`) contains:

- **A training/eval script** (e.g. `train_LodeRunner_ddp.py`) — imports from `yoke.*`, builds
  its argparse from `yoke.helpers.cli` builders, and reads args via
  `fromfile_prefix_chars="@"` (i.e. `python train.py @some.input`).
- **`training_input.tmpl`** — argument file template with `<KEY>` placeholders.
- **A submission template** — a single `training_slurm.tmpl` or `training_shell.tmpl` (a
  complete SLURM/shell script) with conditional blocks (`# <<optional:KEY>>` ... `# <<end>>`)
  for both first-launch and continuation.
- **`cp_files.txt`** — list of files to copy into each generated study directory.
- **A hyperparameter CSV** — first column is `studyIDX` (the index); remaining columns are
  `<KEY>` values varied per study.
- **`README.md`** — notes on the study.
- **(optional) evaluation files** — `evaluation_input.tmpl`, `evaluation_slurm.tmpl`
  (or `evaluation_shell.tmpl`), and a harness-local `eval_<harness>.py` (listed in
  `cp_files.txt`). Present only for harnesses that opt into post-training evaluation;
  see the evaluation subsection below.

### Launch flow (`yoke-start-study` / `HarnessStudy`)
The `yoke-start-study` CLI (`src/yoke/cli/start_study.py`) parses `--csv`, `--rundir`,
`--cpFile`, `--submissionType` (`slurm`/`shell`), and `--dryrun` via
`yoke.helpers.cli.add_default_args`, then constructs a generic `HarnessStudy`
(`src/yoke/harnesses/base.py`). `HarnessStudy.load_hyperparameters` reads the CSV (first
column becomes `studyIDX`), and for each row `run_study`:
1. Creates `runs/study_###/` (### = `studyIDX`, the CSV's first column; one row <-> one dir).
2. Renders `training_input.tmpl` and the submission template by literal `<KEY>` substitution
   (`yoke.helpers.strings.replace_keys`), honoring the `# <<optional:KEY>>` blocks.
3. Copies `cp_files.txt` entries into the study dir.
4. Submits the first job (`sbatch` for slurm, or runs the shell script), unless `--dryrun`.

Note: `--studyIDX` is NOT a `yoke-start-study` flag; it is sourced from the CSV. Train
scripts still receive `--studyIDX` in their rendered `training_input` `@`-file, where it is
declared by `yoke.helpers.cli.add_training_args` (not `add_default_args`).

### Continuation across epochs
Training runs in cycles. When a cycle finishes but total epochs aren't reached, the *train
script itself* calls `yoke.harnesses.base.HarnessStudy.continuation_setup` (a static method
taking `studyIDX`) on the compute node, which re-renders `training_input.tmpl` + submission
template into `study###_restart_training_epoch####.*` and re-submits (`sbatch`/shell). Key
substitution tokens: `<studyIDX>`, `<epochIDX>`, `<INPUTFILE>`, `<CONTINUATION>`,
`<CHECKPOINT>`.

### Template substitution rules (`helpers/strings.py:replace_keys`)
`<studyIDX>` -> zero-padded 3 digits; ints -> `%d`; floats -> str; bool/str -> str. Unknown
types raise `ValueError`.

### Post-training evaluation (opt-in, separate job)
A finished study can submit a **separate** test-set evaluation job for its final checkpoint.
This is opt-in and off by default. A harness enables it by shipping the optional evaluation
files (above) and passing `evaluate_after_training=True` to `HarnessTrainer`. Evaluation is a
distinct job, not a training phase: it does not train, checkpoint, or need DDP.

- `HarnessStudy.generate_evaluation_templates` renders the eval templates at study creation
  from the CSV row, leaving `<CHECKPOINT>`, `<INPUTFILE>`, and `<STEM>` late-bound.
- `HarnessStudy.evaluation_setup(checkpointpath, studyIDX, submission_type)` renders the
  checkpoint-specific `.input`/submission files. Names derive from the checkpoint **stem**
  (not an `epochIDX` arg): e.g. `study005_evaluation_study005_modelState_epoch0100.input`.
  The evaluator reads its record epoch from the checkpoint metadata.
- `HarnessStudy.run_evaluation(study_dir, checkpointpath, study=None)` is the shared
  render-and-submit path (honors `--dryrun`). It renders eval templates on demand from the
  harness dir + CSV row when a study directory lacks them (so pre-feature studies work), then
  submits. Returns `None` if the harness has no evaluation configured.
- `HarnessTrainer.finalize` calls `run_evaluation` on rank 0 exactly once when finished and
  `evaluate_after_training=True`. `evaluate_checkpoint="main"` (default) evaluates the primary
  `.pth`; `evaluate_checkpoint="ema"` evaluates the class-aware `..._ema.pth` companion.
- Manual interface: the installed `yoke-evaluate-study` CLI (`src/yoke/cli/evaluate_study.py`)
  evaluates any checkpoint of an existing study: `yoke-evaluate-study --studyIDX N
  --checkpoint <path> [--dryrun]`. Point `--checkpoint` at `..._ema.pth` to evaluate EMA.

The bespoke `moving_mnist`/`mnist_surrogate` demos and the Lightning harness are excluded.

---

## 5. The `start_study` design

Studies are launched by:
- an **installed CLI** `yoke-start-study` (`src/yoke/cli/start_study.py`, entry point in
  `pyproject.toml`), and
- a **generic `HarnessStudy` class** (`src/yoke/harnesses/base.py`) that encapsulates the
  per-study workflow, using a **single template per artifact** with conditional blocks
  (`# <<optional:KEY>>` ... `# <<end>>`) for both first-launch and continuation.

Design points baked into the code:
- Submission supports **SLURM + shell only**.
- There is **no CLI `--studyIDX` flag**; `studyIDX` is the CSV's first column and names
  `study_###` dirs (one row <-> one dir, for easy debugging). Train scripts receive
  `--studyIDX` through their rendered `training_input` `@`-file, declared by
  `yoke.helpers.cli.add_training_args`.
- A **single generic `HarnessStudy`** parameterized by template files — no per-harness
  subclasses/registry.
- Continuation logic lives **in `HarnessStudy.continuation_setup`** (static method).
- Post-training evaluation logic lives **in `HarnessStudy.run_evaluation` /
  `evaluation_setup`**; the installed `yoke-evaluate-study` CLI is the manual entry point.

---

## 5a. Authoring a training harness with `HarnessTrainer`

Non-Lightning DDP training harnesses should be thin wrappers around
`yoke.harnesses.trainer.HarnessTrainer` rather than copies of a ~450-line
`train_*.py`. The trainer owns the fixed orchestration (DDP setup via
`yoke.utils.parallel.setup_distributed`/`cleanup_distributed`, model/optimizer/
scheduler/dataloader construction, the timed epoch loop, `.pth` checkpointing
via `save_model_and_optimizer`, and continuation resubmission). It **always
writes `.pth`** (HDF5 checkpoint writing and vanilla `DataParallel`/`--multigpu`
are deprecated for **Dec-2026** removal) and trains through DDP in one of two
modes, chosen by `HarnessTrainer(..., device_type=...)`:

- `"cuda"` (default): multi-GPU DDP, one SLURM task per GPU, `nccl`.
- `"cpu"`: one process on one CPU node, in a one-rank `gloo` group
  (`yoke.utils.parallel.setup_single_process_cpu`), for CPU-only clusters
  (e.g. rocinante) and small models. Launch it as a single task
  (`--ntasks=1`, plain `python -u`); the trainer raises if `SLURM_NTASKS > 1`.
  Set threads with `OMP_NUM_THREADS` in the SLURM template.

A harness injects only what varies:

- `model_builder(args, device) -> (model, model_args, model_class, start_epoch, optimizer)`
  — owns *both* fresh and continuation paths; returns the model class so the
  saved `model_class` matches the built model (this killed the `vt_DDP_ldrViT`
  mismatch bug). Use `yoke.utils.builders.build_from_checkpoint(model_class,
  make_model_args, optimizer_kwargs=...)` for the common fresh-vs-continue case;
  write a bespoke builder for architectural surgery (pretrained backbone,
  strip/replace layers).
- `dataset_builder(args) -> (train_dataset, val_dataset)`.
- `epoch_fn` — an existing `yoke.utils.training.epoch` function; per-study extra
  kwargs (`channel_map`, `dataset` tag, EMA) go through `epoch_kwargs`.
- `optimizer_builder` (default `yoke.utils.builders.build_adamw`),
  `scheduler_builder` (optional), `loss_builder` (default
  `yoke.utils.builders.default_mse_loss`).

Dynamic in-loop behavior (EMA, grad-clip, progressive unfreeze) uses the
composable `TrainerHooks` (`on_after_ddp_wrap`, `on_epoch_start`,
`on_before_optimizer_step`, `on_after_step`, `on_before_save`). See
`applications/harnesses/se_DDP_loderunner/train_LodeRunner_ddp.py` for a
migrated example and `docs/source/harness_trainer.rst` for the full write-up.
`moving_mnist` and `mnist_surrogate` are intentionally kept as bespoke,
single-process demo scripts and do **not** use `HarnessTrainer`.

Warmup EMA is packaged as a reusable hook factory,
`yoke.utils.ema.make_ema_hooks`, which returns an
`(on_after_ddp_wrap, on_before_save)` pair (build/restore the EMA shadow +
`global_step`; write the `_ema.pth` companion and `_ema_weights.pth` production
checkpoints). The per-step EMA update and gradient clipping live in
`train_DDP_loderunner_epoch`; the harness passes `grad_clip` and
`ema_update_after_step` via `epoch_kwargs`. When `trainer.ema_model` is set, the
trainer threads it and the live `global_step` into each epoch call. See
`applications/harnesses/ch_ldrViT/train_ldrViT_ddp.py` (and `train_ldrViT_2frame.py`).

**Preflight.** Every `HarnessTrainer` script accepts `--preflight`
(`python train_<name>.py @<input> --preflight`). It builds everything in a
one-process CPU `gloo` group, runs one evaluation datastep through the real
`epoch_fn` and `epoch_kwargs`, prints a report, and exits. There is no training,
checkpointing, or resubmission (`HarnessTrainer.preflight`). The trainer also moves
`nn.Module` losses to the training device.

Shared builder helpers live in `yoke.utils.builders` (`build_adamw`,
`move_optimizer_state_to_device`, `compute_last_epoch`, `default_mse_loss`,
`checkpoint_name`, `build_from_checkpoint`).

---

## 6. Conventions & expectations for changes

- **Put reusable logic in `src/yoke`**, not in harness scripts. Harnesses should orchestrate
  and configure, importing from `yoke.*`.
- **Match existing style:** google docstrings, full type annotations, <=89 col lines, double
  quotes. Run `ruff check` and `ruff format` before finishing.
- **Add/extend tests** under `tests/` mirroring the `src` path of what you change. New
  `HarnessStudy`/CLI code should get `tests/harnesses/` and `tests/cli/`.
- **Keep `-Werror` clean** — no runtime warnings in tests.
- **Prefer editing existing files** over creating new ones. Don't add docs/markdown unless
  asked (this file and `dev_plans/` are the exceptions already established).
- **Git:** only commit/push when explicitly asked. `main` is the stable comparison point.
- **Data & external paths** are passed as CLI args (see `helpers/cli.py` `add_filepath_args`
  et al.); never hardcode dataset locations into the package.

---

## 6a. Agent skills

`skills/` holds guided, interview-style procedures for adding training
capability to Yoke. Each is a `SKILL.md` plus reference notes. Claude Code finds
them through `.claude/skills/` symlinks. Any other agent can follow them by
reading the `SKILL.md` directly. Start with `skills/README.md`.

| Skill | Use it to |
| --- | --- |
| `skills/yoke-train-my-data/` | **start here**: frame the problem and run the others in order |
| `skills/yoke-new-dataset/` | write a `Dataset` class (any file format) + tests |
| `skills/yoke-model/` | choose and size a model, or compose one from Yoke blocks |
| `skills/yoke-new-loss/` | choose or write a loss with the right output shape |
| `skills/yoke-new-training-step/` | add a datastep/epoch pair for a new batch contract |
| `skills/yoke-new-harness/` | build an `applications/harnesses/` directory |
| `skills/yoke-slurm-template/` | write a SLURM template from read-only cluster queries |
| `skills/yoke-launch-study/` | run readiness checks and preflight; give the launch command |
| `skills/yoke-run-doctor/` | diagnose a study the user already submitted |
| `skills/yoke-finetune-mcp/` | fine-tune a model a Nomad MCP server offers, through its tools only |

Agents never cancel or modify jobs. `yoke-launch-study` prints the launch command
and, only if the user explicitly asks, runs it for a SLURM study. The jobs then
live in the SLURM queue, not the agent's shell. Shell-type studies are never
agent-submitted. `yoke-run-doctor` picks the run up later from the files on disk,
and only proposes restart commands. `yoke-finetune-mcp` is the remote case: an
agent with no Yoke checkout fills in a fine-tune spec for a Nomad server, which
runs the training itself after the user's explicit yes.

`skills/shared/batch_contracts.md` maps dataset return tuples to datasteps and
epoch functions. `tests/skills/test_skill_files.py` keeps the skills in sync with
the code, so update the skills when you rename or change code they mention.

---

## 7. Quick command reference

```bash
# run tests / coverage / lint
pytest -Werror
pytest --cov --cov-report term-missing
ruff check && ruff format --check --diff

# the yoke-start-study CLI, run from inside a harness dir
yoke-start-study --csv <hyperparams.csv> --rundir ./runs --cpFile cp_files.txt \
                 --submissionType slurm   # or: shell

# manually evaluate a checkpoint of an existing study, from inside a harness dir
yoke-evaluate-study --studyIDX <N> --checkpoint runs/study_<NNN>/<checkpoint>.pth \
                    --dryrun   # render + print submit cmd without submitting

# compare current branch to stable
git diff main --stat
```
