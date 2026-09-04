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
                               #   create_slurm_files.py (DEPRECATED), logger.py, training_design.py, templates/
  utils/                       # checkpointing.py, dataload.py, parallel.py, parameters.py, restart.py
    training/
      datastep/                # per-batch step fns (loderunner, lsc_policy, lsc_reward, scalar_output, array_output, diff_loderunner)
      epoch/                   # per-epoch train/val fns (same family as datastep/)
    diffusion/
  lr_schedulers.py             # e.g. ConstantWithWarmupScheduler
  scheduled_sampling.py
  cli/                         # console-script entry points
    start_study.py             # `yoke-start-study` (in active development)
  harnesses/                   # NEW: HarnessStudy base class (in active development)
    base.py

applications/                  # NOT installable; scripts + study definitions
  harnesses/                   # one subdir per study (see Section 4)
    START_study.py             # legacy monolithic launcher (being replaced by yoke-start-study)
  evaluation/  viewers/  normalization/  filelists/
  makefilelists.py

tests/                         # pytest; mirrors src layout (datasets/, models/, helpers/, utils/, ...)
docs/                          # sphinx (source/, Makefile)
docker/                        # Dockerfile for containerized builds
dev_plans/                     # design/planning docs (markdown)
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
- **A submission template** — `training_slurm.tmpl` (a complete SLURM script) and, on `main`,
  a matching `training_START.slurm` first-launch variant.
- **`cp_files.txt`** — list of files to copy into each generated study directory.
- **A hyperparameter CSV** — first column is `studyIDX` (the index); remaining columns are
  `<KEY>` values varied per study.
- **`README.md`** — notes on the study.

### Launch flow (legacy `START_study.py`, being replaced)
For each CSV row, the launcher:
1. Creates `runs/study_###/` (### = `studyIDX`).
2. Renders `training_input.tmpl` and the submission template by literal `<KEY>` substitution
   (`yoke.helpers.strings.replace_keys`).
3. Copies `cp_files.txt` entries into the study dir.
4. Submits the first job (`sbatch`, etc.).

### Continuation across epochs
Training runs in cycles. When a cycle finishes but total epochs aren't reached, the *train
script itself* calls `yoke.utils.restart.continuation_setup` on the compute node, which
re-renders `training_input.tmpl` + submission template into
`study###_restart_training_epoch####.*` and re-submits (`sbatch`). Key substitution tokens:
`<studyIDX>`, `<epochIDX>`, `<INPUTFILE>`, `<CONTINUATION>`, `<CHECKPOINT>`.

### Template substitution rules (`helpers/strings.py:replace_keys`)
`<studyIDX>` -> zero-padded 3 digits; ints -> `%d`; floats -> str; bool/str -> str. Unknown
types raise `ValueError`.

---

## 5. Current in-development work: `start_study` upgrade

The active branch **`start_study_upgrade`** is turning the legacy per-harness
`START_study.py` copy-script into:
- an **installed CLI** `yoke-start-study` (`src/yoke/cli/start_study.py`, entry point in
  `pyproject.toml`), and
- a **generic `HarnessStudy` class** (`src/yoke/harnesses/base.py`) that encapsulates the
  per-study workflow, using a **single template per artifact** with conditional blocks
  (`# <<optional:KEY>>` ... `# <<end>>`) to collapse the old START/continuation file pairs.

**The full plan (read this before implementing):**
`dev_plans/start_study_upgrade_plan.md`. Its resolved decisions:
- **Q1:** Support **SLURM + shell only** (drop flux/batch).
- **Q2:** Remove the CLI `--studyIDX` flag; `studyIDX` is the CSV's first column and names
  `study_###` dirs (one row <-> one dir, for easy debugging).
- **Q3:** Keep a **single generic `HarnessStudy`** parameterized by template files — no
  per-harness subclasses/registry.
- **Q4:** Move continuation logic **into `HarnessStudy`** (static/class method), update all
  train scripts, and remove `yoke.utils.restart`.

**Known gaps to fix (per the plan):** `--dryrun` referenced but undefined in `cli.py`;
unused imports in `start_study.py`; SLURM-only `submit_job`; still importing the deprecated
`create_slurm_files`; missing docstrings/annotations; no tests for `HarnessStudy`/CLI.

---

## 6. Deprecations — do not build on these

- **JSON-to-SLURM generation is deprecated.** `yoke.helpers.create_slurm_files.MkSlurm` and
  `slurm_config.json` were an attempt to synthesize SLURM scripts from JSON. Abandoned — the
  space of valid SLURM scripts is too large for a schema to help. **Harnesses supply a
  complete SLURM script.** A dedicated follow-up task will remove this code entirely; for now
  stop using it and don't extend it.

---

## 7. Conventions & expectations for changes

- **Put reusable logic in `src/yoke`**, not in harness scripts. Harnesses should orchestrate
  and configure, importing from `yoke.*`.
- **Match existing style:** google docstrings, full type annotations, <=89 col lines, double
  quotes. Run `ruff check` and `ruff format` before finishing.
- **Add/extend tests** under `tests/` mirroring the `src` path of what you change. New
  `HarnessStudy`/CLI code should get `tests/harnesses/` and `tests/cli/`.
- **Keep `-Werror` clean** — no runtime warnings in tests.
- **Prefer editing existing files** over creating new ones. Don't add docs/markdown unless
  asked (this file and `dev_plans/` are the exceptions already established).
- **Git:** only commit/push when explicitly asked. The branch under development is
  `start_study_upgrade`; `main` is the stable comparison point.
- **Data & external paths** are passed as CLI args (see `helpers/cli.py` `add_filepath_args`
  et al.); never hardcode dataset locations into the package.

---

## 8. Quick command reference

```bash
# run tests / coverage / lint
pytest -Werror
pytest --cov --cov-report term-missing
ruff check && ruff format --check --diff

# the new CLI (in development), run from inside a harness dir
yoke-start-study --csv <hyperparams.csv> --rundir ./runs --cpFile cp_files.txt \
                 --submissionType slurm   # or: shell

# compare current branch to stable
git diff main...start_study_upgrade --stat
```
