# Plan: `start_study` as a CLI Tool and Class-Based Harness Creation

Status: **in development** on branch `start_study_upgrade`

## Progress log

Phases **A, B, C, D, and E are complete**. **Phase F is next** and not started.

- **Phase A — DONE.** `--dryrun` added to `cli.add_default_args`; commented-out `--studyIDX`
  block deleted; unused imports removed from `cli/start_study.py`; docstrings + type
  annotations added to all `HarnessStudy` methods; `ruff` clean.
- **Phase B — DONE.** `submission_type` threaded from CLI -> `HarnessStudy` -> `submit_job`;
  `SUBMISSION_SYSTEMS` dispatch table added (`slurm` -> `sbatch`/`training_slurm.tmpl`/`.slurm`,
  `shell` -> `source`/`training_shell.tmpl`/`.sh`); `_get_slurm_template` generalized to
  `_render_submission_template`; `generate_initial_inputs` emits `study###_START.<ext>`;
  `--submissionType` choices narrowed to `{slurm, shell}`.
- **Phase C — DONE (with a stronger action than originally planned).** Instead of just
  deprecating the JSON-to-SLURM code (original C2), we **deleted it outright** per a decision
  during implementation: removed `src/yoke/helpers/create_slurm_files.py`,
  `tests/helpers/test_slurm_create.py`, and the `src/yoke/helpers/templates/` dir
  (`chicoma.json`, `slurm.tmpl`). The `chicoma_lsc_loderunner-ch-subsampling` harness (the
  only one shipping `slurm_config.json`) was **removed entirely** (C3). The now-defunct JSON
  branch was also stripped out of the legacy `applications/harnesses/START_study.py` (which is
  still slated for deletion in Phase F).
- **Phase D — DONE.** `HarnessStudy.continuation_setup(...)` added as a `@staticmethod`
  (slurm + shell only; unsupported types raise `ValueError` — note the old `restart.py` raised
  `UnboundLocalError` for `batch`). The templating contract is documented in the `HarnessStudy`
  class docstring (D2). All 11 train scripts updated to
  `from yoke.harnesses.base import HarnessStudy` + `HarnessStudy.continuation_setup(...)` (D3).
  `src/yoke/utils/restart.py` and `tests/utils/test_restart.py` deleted (D4). `tests/harnesses/`
  package created with `test_base.py` containing the ported continuation tests **and** the
  parametrized (`slurm`/`shell`) round-trip test (D5). Full suite: **414 passed** with `-Werror`.

- **Phase E — DONE.** Added `tests/cli/` package with `test_start_study.py` exercising `main()`
  end-to-end under `--dryrun` for both `slurm` and `shell`, asserting study dirs / first-launch
  + continuation files / copied `cp_files.txt` entries are created and that the exact submit
  command (`sbatch ...` / `source ...`) is printed and never executed (E1, E4). Added
  `render_template` conditional-block unit tests (optional block present/absent) (E2) and a
  `load_hyperparameters` unit test (row->dict, `studyIDX` int index, comment/header handling)
  (E3). Also added constructor invalid-submission-type and non-dryrun `submit_job`
  (`os.system` monkeypatched) tests to close coverage gaps. **New modules at 100% coverage**
  (`yoke.cli.start_study`, `yoke.harnesses.base`). `ruff check` + `ruff format --check` clean.
  Full suite: **421 passed** with `-Werror` (E5).

- **Phase F1 — DONE.** Migrated all 13 maintained harnesses to single-template form. Merged
  each `training_START.input` into its `training_input.tmpl`, wrapping the continuation-only
  args in a `# <<optional:CONTINUATION>>` ... `# <<end>>` block; the submission `*.tmpl`
  (already carrying `<epochIDX>`/`<INPUTFILE>` placeholders) became the single authoritative
  submission template. Deleted all `training_START.{input,slurm,sh}`, `run_study.bat`, and the
  per-harness `START_study.py` copies/symlinks. Notable per-harness fixes:
    - **`base.py` padding fix:** `generate_initial_inputs` now sets the first-launch
      `<epochIDX>` to `0001` (4-digit) instead of the studyIDX value; padding rule locked as
      **studyIDX = 3-digit, epochIDX = 4-digit** everywhere.
    - **`ch_DDP_diffLDR`:** removed the vestigial `CHECKPOINT` column (all `none`) from
      `study_template.csv` so `<CHECKPOINT>` survives into the continuation template for
      `continuation_setup` to fill (the column was pre-consuming the reserved placeholder).
    - **`se_DDP_loderunner_finetune_cylex`:** finetune-only args (`--pretrained_model`,
      `--freeze_backbone_epochs`, `--warmup_lr`) now live unconditionally in the single input
      template; the train script already gates them on the first cycle, so no first-launch-only
      block syntax was needed.
    - **`mnist_surrogate`:** added a proper `training_slurm.tmpl` (from the stray
      `training_START.slurm`) so it supports both slurm and shell; normalized `<checkpoint>` ->
      `<CHECKPOINT>`. **`moving_mnist`:** shell-only, normalized `<checkpoint>` ->
      `<CHECKPOINT>`.
  All harness/submission-type combinations verified via a dryrun round-trip (correct
  `0001`/`study###_START.input` in first-launch files; `<epochIDX>`/`<INPUTFILE>`/`<CHECKPOINT>`
  preserved in continuation templates). Full suite still **421 passed** with `-Werror`; `ruff`
  clean.

- **Phase F2 — DONE.** Authored/updated a `README.md` for every harness (12 total), each with a
  purpose blurb, a file list, a study description, and the `yoke-start-study` invocation
  (`--submissionType slurm` or `shell`, with a `--dryrun` note). Removed **deprecated CSVs**
  whose columns no longer satisfy the current `training_input.tmpl` (missing `MAX_TIME_OFFSET`):
  six from `ch_DDP_loderunner` (`ddp_benchmark`, `ddp_chkpt_test`, `ddp_lrstudy1`,
  `ddp_lrstudy2`, `ddp_noise`, `ddp_production`) and one from `se_DDP_loderunner`
  (`ddp_production`); each harness keeps its working `ddp_paper_study.csv`. Deleted the
  redundant `lsc_action/training_slurm_debug.tmpl` (single authoritative `training_slurm.tmpl`
  only). **Removed the entire `mini-run-test` harness** — the `mnist_surrogate` and
  `moving_mnist` harnesses now serve as the quick end-to-end functionality checks (their READMEs
  say so). `se_` = Selene, `ch_` = Chicoma. All remaining CSVs render cleanly (dryrun, no
  unrendered non-reserved tokens); full suite **421 passed** with `-Werror`.

**Left off at:** F2 done. Remaining Phase F: **F3** (add an "Authoring a Harness" guide under
`docs/` and link from the main `README.md`), and **F4** (delete the legacy
`applications/harnesses/START_study.py` once CI is green).

---

This document lays out a plan for two intertwined goals:

1. Turn `start_study` into a proper, installed **command-line tool** (`yoke-start-study`).
2. Move harness creation from a **copy/edit-a-script** workflow to a **class-instantiation**
   workflow, so that harnesses become easier to author, more uniform, and less error-prone.

The plan is written against the current state of the branch and should be treated as a
living document. Update it as decisions are made.

---

## 1. Background: where we are coming from (`main`)

On `main`, every harness under `applications/harnesses/<name>/` is launched by copying a
monolithic driver script, `applications/harnesses/START_study.py` (248 lines), into (or
next to) the harness directory. That script:

- Parses args via `yoke.helpers.cli.add_default_args`.
- Reads a hyperparameter CSV into a list of per-study dicts.
- For each study, renders a family of template files by literal `<KEY>` string
  substitution (`yoke.helpers.strings.replace_keys`):
  - `training_input.tmpl` -> per-study `training_input.tmpl` (continuation form).
  - `training_START.input` -> `study###_START.input` (first-launch form).
  - `training_slurm.tmpl` / `training_START.slurm` (or `.flux` / `.sh` / `.bat`).
- Copies files listed in `cp_files.txt` into the study directory.
- Submits the first job (`sbatch` / `flux batch` / `source` / `cmd.exe`).

Continuation across epochs is handled separately by `yoke.utils.restart.continuation_setup`,
which re-renders `training_input.tmpl` and `training_slurm.tmpl` on the compute node and
re-submits via `sbatch`.

### Pain points on `main`

- **Copy-paste driver.** `START_study.py` is duplicated/copied per harness; fixes do not
  propagate. Logic lives outside the installable package (`src/yoke`), so it is untested
  and unversioned in a meaningful way.
- **Two near-identical file pairs.** Each harness must maintain both a "START" file and a
  "tmpl" file for both input and submission script (`training_START.input` +
  `training_input.tmpl`, `training_START.slurm` + `training_slurm.tmpl`). They differ only
  in a few substituted values (`<epochIDX>` -> `0001`, `<INPUTFILE>` hardcoded, and the
  presence/absence of `--continuation`/`--checkpoint`). This is a frequent source of drift
  and copy errors.
- **Four submission systems, four code paths.** slurm / flux / shell / batch are handled by
  large parallel `if/elif` blocks that are hard to keep in sync. (Decision: flux and batch
  are being dropped — see Section 3.4 / Section 8 Q1.)
- **JSON-to-SLURM generation (deprecated).** `slurm_config.json` +
  `yoke.helpers.create_slurm_files.MkSlurm` was an attempt to synthesize SLURM scripts from
  JSON. This has been **abandoned** — there are too many possible SLURM configurations for a
  JSON schema to add value. Harnesses should simply provide a complete SLURM script.

## 2. Background: what the branch has started (`start_study_upgrade`)

The branch introduces the scaffolding for a class-based, installed CLI:

- **`pyproject.toml`**: adds a console-script entry point
  `yoke-start-study = "yoke.cli.start_study:main"`.
- **`src/yoke/cli/start_study.py`**: a thin `main()` that builds a `HarnessStudy`, loads the
  CSV, and calls `run_study` per row.
- **`src/yoke/harnesses/base.py`**: a new `HarnessStudy` class encapsulating the
  per-study workflow (`load_hyperparameters`, `render_template`, `copy_files`,
  `generate_initial_inputs`, `generate_tmpl_inputs`, `submit_job`, `run_study`).
- **`src/yoke/helpers/cli.py`**: `--studyIDX` commented out (no longer needed since the CLI
  iterates all rows).
- Several stale/unmaintained harnesses removed (`adams_lsc_policy`,
  `burr_lsc_density_surrogate`, `nc_density_CNN`), and per-harness `START_study.py` shims
  deleted.

### Improvements already realized by the branch

- **Single template per artifact.** `HarnessStudy.render_template` supports conditional
  blocks (`# <<optional:KEY>>` ... `# <<end>>`), so one `training_input.tmpl` and one
  `training_slurm.tmpl` can generate *both* the first-launch and continuation forms. This
  eliminates the `training_START.*` duplicates.
- **Installed and importable.** Logic now lives in `src/yoke`, so it can be unit tested and
  reused, and is invoked as a real command rather than a copied script.

### Gaps / bugs in the current branch state

These are concrete issues found in the current checkout that the plan must address:

1. **`--dryrun` is referenced but never defined.** `start_study.py` reads `args.dryrun`, but
   `cli.add_default_args` does not add a `--dryrun` argument. Running the CLI as-is raises
   `AttributeError`.
2. **Submission is SLURM-only.** `HarnessStudy.submit_job` hardcodes `sbatch` and
   `submit_job`/`run_study` ignore `args.submissionType`. Per the decision in Section 8 Q1,
   the target set is **SLURM + shell**; `shell` support still needs to be added, and the
   `--submissionType` choices should be narrowed to `{slurm, shell}`.
3. **Still depends on deprecated JSON-to-SLURM.** `base.py._get_slurm_template` and
   `start_study.py` still import and use `yoke.helpers.create_slurm_files`. This should be
   removed per the deprecation decision (see Section 5).
4. **Unused imports** in `start_study.py` (`os`, `shutil`, `pd`, `strings`,
   `create_slurm_files`) — will trip `ruff` (F401).
5. **Continuation contract mismatch (to be consolidated).** `yoke.utils.restart.continuation_setup`
   (run on the compute node) hardcodes `<INPUTFILE>`, `<epochIDX>`, `<CHECKPOINT>`
   substitutions and the `training_input.tmpl` / `training_slurm.tmpl` filenames. The new
   `generate_tmpl_inputs` writes those same template names, so the two are implicitly coupled
   and untested. Per Decision Q4 this logic is being **moved into `HarnessStudy`** so the
   contract lives in one class (see Section 3.2 / Phase D).
6. **No tests** exist for `HarnessStudy` or the `yoke-start-study` entry point.
7. **Missing docstrings / type annotations** on several `HarnessStudy` methods
   (`load_hyperparameters`, `render_template`, etc.), which violates the repo's `ruff`
   `D`/`ANN` rules.

---

## 3. Target design

### 3.1 CLI surface

`yoke-start-study` should be runnable from within a harness directory:

```bash
cd applications/harnesses/ch_DDP_loderunner
yoke-start-study --csv ddp_paper_study.csv --rundir ./runs --cpFile cp_files.txt \
                 --submissionType slurm [--dryrun]
```

Responsibilities of `main()`:

- Parse args (`cli.add_default_args`), including a new `--dryrun` flag.
- Instantiate `HarnessStudy(...)` passing `submission_type` through.
- Load hyperparameters and iterate `run_study` per row.
- Exit non-zero with a clear message on missing required files (CSV, templates, cp list).

### 3.2 `HarnessStudy` class

**Design intent (Q3): a single generic class, no per-harness subclasses or registry.** A
harness is defined entirely by its configuration files (`training_input.tmpl`, submission
template, `cp_files.txt`, hyperparameter CSV). "Class instantiation" means `yoke-start-study`
constructs one `HarnessStudy` from that configuration and drives the study. Keep the class as
the single source of truth for the per-study workflow. Concretely:

- **Constructor** takes `rundir`, `template_dir`, `cp_file`, `submission_type`, `dryrun`.
  Validate existence of `training_input.tmpl` and the submission template up front and fail
  fast with actionable errors.
- **`load_hyperparameters(csv)`** -> `list[dict]` (already implemented; add annotations +
  docstring).
- **`render_template(path, subs)`** — keep the conditional-block mechanism. Consider making
  the block markers a documented public convention (see Section 3.3).
- **`copy_files(study_dir)`** — copy `cp_files.txt` entries; error clearly if a listed file
  is missing.
- **`generate_initial_inputs` / `generate_tmpl_inputs`** — produce the first-launch pair and
  the continuation template pair from the *single* source templates. Keep the emitted
  filenames (`training_input.tmpl`, submission template, `study###_START.*`) stable so the
  continuation path keeps working.
- **Continuation entry point (Q4)** — a `@staticmethod`/`@classmethod` that supersedes
  `yoke.utils.restart.continuation_setup`. Runs on the compute node from within a train
  script (no pre-built instance required): reads the local templates, substitutes
  `<CHECKPOINT>`/`<INPUTFILE>`/`<epochIDX>`, writes the restart files, returns the new submit
  script path. Supports `slurm` and `shell` only.
- **`submit_job(study_dir, script_path)`** — dispatch on `submission_type`:
  `slurm` -> `sbatch`, `shell` -> `source`. Honor `dryrun` by printing the command instead of
  executing.
- **`run_study(study)`** — orchestrate the above.

### 3.3 Template convention (single-file templates)

Document a single, minimal templating convention so harness authors do not need to maintain
duplicate START/continuation files:

- `<KEY>` — literal substitution from the study dict (existing `strings.replace_keys`).
- `# <<optional:KEY>>` ... `# <<end>>` — block included only when `KEY` is present in the
  substitution dict. Used to include `--continuation` / `--checkpoint` lines only in the
  continuation form.
- Reserved keys the harness injects automatically: `studyIDX`, `epochIDX`, `INPUTFILE`,
  `CONTINUATION`, `CHECKPOINT`.

This convention must be documented (Section 6) and covered by tests so authors can rely on
it.

### 3.4 Submission-system abstraction

**Decision (Q1): support SLURM + shell only.** flux and batch are dropped.

Replace the four parallel `if/elif` blocks with a small dispatch table keyed on
`submission_type`, mapping each supported system to (a) its template file name and (b) its
submit command:

- `slurm` -> template `training_slurm.tmpl`, submit `sbatch <script>`.
- `shell` -> template `training_shell.tmpl`, submit `source <script>` (for local/dev runs).

Narrow the `--submissionType` argument `choices` to `{slurm, shell}`. Removal of the flux and
batch code paths spans:

- `HarnessStudy` / `yoke.cli.start_study` (this branch).
- `applications/harnesses/START_study.py` (deleted at end of migration anyway).
- The continuation logic being moved into `HarnessStudy` per Q4 (keep only `slurm` / `shell`;
  the old `yoke.utils.restart.continuation_setup` is removed — see Phase D).

Also remove any `training_flux.tmpl` / `training_batch.tmpl` / `*.flux` / `*.bat` files from
harnesses during migration (Phase F).

---

## 4. Work breakdown

Ordered, each item small enough to review independently.

### Phase A — make the branch correct and runnable — DONE
- [x] A1. Add `--dryrun` (store_true) to `cli.add_default_args`. Permanently delete the
      commented-out CLI-level `--studyIDX` block (Decision Q2): `studyIDX` comes from the
      CSV's first column (`index_col=0`) and drives both the `study_###` directory names and
      the `<studyIDX>` template substitution passed to the train script. The old
      single-target CLI flag is redundant now that `yoke-start-study` iterates all rows.
- [x] A2. Remove unused imports from `src/yoke/cli/start_study.py`.
- [x] A3. Add docstrings + type annotations to all `HarnessStudy` methods (satisfy `D`/`ANN`).
- [x] A4. Run `ruff check` / `ruff format` and fix.

### Phase B — submission-system support (SLURM + shell) — DONE
- [x] B1. Thread `submission_type` from `args` -> `HarnessStudy` -> `submit_job`.
- [x] B2. Introduce a dispatch table for {template filename, submit command} for
      `slurm` and `shell` only. (Implemented as `HarnessStudy.SUBMISSION_SYSTEMS`.)
- [x] B3. Generalize `generate_initial_inputs` / `_get_slurm_template` to the selected
      submission template (slurm or shell). (`_get_slurm_template` renamed to
      `_render_submission_template`.)
- [x] B4. Narrow `--submissionType` choices to `{slurm, shell}` in `cli.add_default_args`.
- [x] B5. (See Phase D — flux/batch removal from the continuation logic happened as part of
      moving that logic into `HarnessStudy`.)

### Phase C — decouple from deprecated JSON-to-SLURM — DONE (removed, not just deprecated)
- [x] C1. Remove `create_slurm_files` usage from `base.py` and `start_study.py`; harnesses
      always supply a complete submission script template.
- [x] C2. **Superseded:** rather than marking `create_slurm_files` deprecated, we **deleted**
      it and its test entirely (`src/yoke/helpers/create_slurm_files.py`,
      `tests/helpers/test_slurm_create.py`) along with the `src/yoke/helpers/templates/` dir
      (`chicoma.json`, `slurm.tmpl`) that only it used.
- [x] C3. Remove `slurm_config.json` from the one harness that still shipped it — the whole
      `chicoma_lsc_loderunner-ch-subsampling` harness was removed.

### Phase D — move continuation logic into `HarnessStudy` (Q4) — DONE
- [x] D1. Add a continuation entry point to `HarnessStudy` (implemented as the
      `@staticmethod continuation_setup`) that reproduces the old
      `yoke.utils.restart.continuation_setup` behavior: read the local `training_input.tmpl` +
      submission template, substitute `<CHECKPOINT>`, `<INPUTFILE>`, `<epochIDX>`, write the
      `study###_restart_training_epoch####.{input,slurm|sh}` files, and return the new submit
      script path. Supports only `slurm` and `shell` (Q1); unsupported types raise
      `ValueError` (the old code raised `UnboundLocalError`).
- [x] D2. Document and lock the templating contract in one place (now entirely inside
      `HarnessStudy`'s class docstring): the keys `<studyIDX>`, `<epochIDX>`, `<INPUTFILE>`,
      `<CONTINUATION>`, `<CHECKPOINT>` and how `generate_tmpl_inputs` and the continuation
      method share them.
- [x] D3. Update every train script that imported `continuation_setup` to call the new
      `HarnessStudy` location. All 11 callers updated (all DDP / loderunner / policy /
      surrogate / diffusion / lightning train scripts).
- [x] D4. Remove `yoke.utils.restart` (and its flux/batch branches) — done; no train script
      imports it. `tests/utils/test_restart.py` removed (replaced by `tests/harnesses/`).
- [x] D5. Add a round-trip test: `generate_tmpl_inputs` then the continuation method in a temp
      dir; assert the restart `.input`/`.slurm` (and `.sh`) are well-formed (correct
      `epochIDX`, `INPUTFILE`, `CHECKPOINT`). Implemented in `tests/harnesses/test_base.py`,
      parametrized over `slurm`/`shell`.

### Phase E — tests — DONE
- [x] E1. Create `tests/harnesses/test_base.py` and `tests/cli/test_start_study.py`.
      (`tests/harnesses/test_base.py` created in Phase D; `tests/cli/test_start_study.py`
      added in Phase E.)
- [x] E2. Unit-test `render_template` conditional blocks (optional block present/absent).
- [x] E3. Unit-test `load_hyperparameters` (CSV -> list[dict], comment handling, index).
- [x] E4. Test `run_study` end-to-end in a temp dir with `dryrun=True` (assert files created,
      submit command printed, nothing executed). Done via the CLI `main()` dryrun tests, which
      assert the exact `sbatch`/`source` command is printed.
- [x] E5. Ensure `--cov` stays healthy for the new modules. `yoke.cli.start_study` and
      `yoke.harnesses.base` are at 100% coverage; full suite is 421 passed with `-Werror`.

### Phase F — migrate harnesses + docs
- [x] F1. Migrate the maintained harnesses to single-template form (drop `training_START.*`):
      `ch_DDP_loderunner`, `ch_lightning_loderunner`, `ch_lsc_policy`,
      `mnist_surrogate`,
      `moving_mnist`, `se_DDP_loderunner*`, `ch_lsc_inverse`, `ch_lsc_reward`,
      `ch_DDP_diffLDR`, `lsc_action`.
      (`chicoma_lsc_loderunner-ch-subsampling` was removed in Phase C rather than migrated.
      `mini-run-test` was migrated in F1 but then removed in F2 — see below.)
- [x] F2. Update each harness `README.md` to show the `yoke-start-study` invocation. Also
      removed deprecated CSVs (columns no longer matching the template), deleted the redundant
      `lsc_action/training_slurm_debug.tmpl`, and removed the `mini-run-test` harness entirely
      (superseded by the `mnist_surrogate`/`moving_mnist` quick-check harnesses).
- [ ] F3. Add a top-level "Authoring a harness" guide (see Section 6) under `docs/` and link
      from the main `README.md`.
- [ ] F4. Delete `applications/harnesses/START_study.py` once all harnesses are migrated and
      CI is green.

---

## 5. Deprecation: JSON-to-SLURM

Per project decision, the JSON-to-SLURM creation mechanism is deprecated:

- Modules/artifacts: `yoke.helpers.create_slurm_files` (`MkSlurm`), `slurm_config.json`,
  and `tests/helpers/test_slurm_create.py`.
- Rationale: the space of valid SLURM scripts is too large/varied for a JSON schema to
  provide value; harnesses will ship complete SLURM scripts instead.
- This plan **stops using** it (Phase C) but leaves final **removal** to the dedicated
  follow-up cleanup task, to keep this change set focused and reviewable.

---

## 6. Documentation deliverable: "Authoring a Harness"

A new guide should make harness creation a recipe rather than a copy job:

1. Create `applications/harnesses/<name>/`.
2. Add a training/eval script (referenced by `<train_script>` in templates).
3. Write one `training_input.tmpl` using `<KEY>` substitutions and `# <<optional:...>>`
   blocks for continuation-only args.
4. Write one submission template (e.g. `training_slurm.tmpl`) — a complete script.
5. Write `cp_files.txt` (files copied into each study dir) and a hyperparameter CSV.
6. Launch: `yoke-start-study --csv <csv> --submissionType slurm [--dryrun]`
   (or `--submissionType shell` for local/dev runs).
7. Document the reserved keys and the continuation lifecycle.

---

## 7. Acceptance criteria

- `yoke-start-study` runs from a harness dir with `--dryrun` and prints the exact submit
  commands without executing them.
- A single `training_input.tmpl` + single submission template generate both first-launch and
  continuation artifacts; no `training_START.*` files remain in migrated harnesses.
- All four submission systems (slurm/flux/shell/batch) work, OR a documented decision drops
  some of them. **(Decision Q1: only `slurm` and `shell` are supported; flux/batch removed.)**
- No runtime dependency on `create_slurm_files`; it and `slurm_config.json` are marked
  deprecated.
- Continuation logic lives in `HarnessStudy` (Q4); `yoke.utils.restart` is removed and no
  train script imports `continuation_setup` from the old location.
- New unit tests cover `HarnessStudy` and the CLI; `ruff check`, `ruff format --check`, and
  `pytest -Werror --cov` pass.
- `docs/` contains an "Authoring a Harness" guide; harness `README.md`s show the new CLI.

---

## 8. Resolved decisions

All open questions have been resolved. Summary:

- **Q1 — Submission systems:** SLURM + shell only; flux and batch dropped.
- **Q2 — CLI `--studyIDX`:** removed permanently; `studyIDX` is the CSV's first column.
- **Q3 — Class methodology:** single generic `HarnessStudy` + template files (no subclasses/registry).
- **Q4 — Continuation logic:** moved into `HarnessStudy`; all train scripts updated; `utils/restart.py` removed.

Details:

- **Keep flux/shell/batch?** **RESOLVED (Q1):** support **SLURM + shell only**. flux and
  batch are dropped from `HarnessStudy`, the `--submissionType` choices, and the continuation
  logic (which is itself being relocated into `HarnessStudy` per Q4). Their template files are
  removed during harness migration. `shell` is retained for local/dev testing.
- **CLI-level `--studyIDX`.** **RESOLVED (Q2):** remove it permanently. `studyIDX` is the
  CSV's first column (`index_col=0`); it names the `study_###` directory and is substituted
  into the train-script args via `<studyIDX>`. This one-to-one mapping between a CSV row and
  its `study_###` directory is intentional and supports debugging (locate the failed
  `study_###` dir, read the matching CSV row for its parameters). The old single-target CLI
  flag is redundant since the CLI iterates every row; to re-run one study, reduce the CSV to
  that single row.
- **Registry vs. convention.** **RESOLVED (Q3):** use the **generic class + template files**
  approach (Option A). There is a single generic `HarnessStudy` class; "class instantiation"
  means constructing a `HarnessStudy` from a harness's configuration (its `training_input.tmpl`,
  submission template, `cp_files.txt`, and hyperparameter CSV). Authoring a harness means
  writing those files, not writing a Python subclass. No per-harness subclasses or harness
  registry are introduced. SLURM/shell scripts remain first-class files supplied by the
  harness. This keeps per-harness code at zero and centralizes all orchestration logic in one
  tested class.
- **`continuation_setup` placement.** **RESOLVED (Q4):** move the compute-node continuation
  logic into `HarnessStudy` so the entire templating contract (generation *and* continuation)
  lives in one class. All train scripts are updated to call the new location instead of
  `from yoke.utils.restart import continuation_setup`. Since train scripts run standalone on
  compute nodes without a pre-built `HarnessStudy` instance, expose the continuation entry
  point as a `@staticmethod`/`@classmethod` (or a constructor cheap enough to build in-script
  from the local `./training_input.tmpl` + submission template). `yoke.utils.restart` is
  removed (or reduced to nothing) once all train scripts are migrated. Restrict the moved
  logic to the SLURM + shell systems per Q1.
