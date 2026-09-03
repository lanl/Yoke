# Plan: `start_study` as a CLI Tool and Class-Based Harness Creation

Status: **in development** on branch `start_study_upgrade`

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
  large parallel `if/elif` blocks that are hard to keep in sync.
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
   `submit_job`/`run_study` ignore `args.submissionType` (flux / shell / batch). The old
   multi-scheduler support has effectively been dropped.
3. **Still depends on deprecated JSON-to-SLURM.** `base.py._get_slurm_template` and
   `start_study.py` still import and use `yoke.helpers.create_slurm_files`. This should be
   removed per the deprecation decision (see Section 5).
4. **Unused imports** in `start_study.py` (`os`, `shutil`, `pd`, `strings`,
   `create_slurm_files`) — will trip `ruff` (F401).
5. **Continuation contract mismatch.** `yoke.utils.restart.continuation_setup` (run on the
   compute node) still hardcodes `<INPUTFILE>`, `<epochIDX>`, `<CHECKPOINT>` substitutions
   and the `training_input.tmpl` / `training_slurm.tmpl` filenames. The new
   `generate_tmpl_inputs` writes those same template names, so the two must be kept
   contract-compatible; this coupling is currently implicit and untested.
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

Keep the class as the single source of truth for the per-study workflow. Concretely:

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
  filenames (`training_input.tmpl`, `training_slurm.tmpl`, `study###_START.*`) stable so the
  compute-node continuation path keeps working.
- **`submit_job(study_dir, script_path)`** — dispatch on `submission_type`:
  slurm -> `sbatch`, flux -> `flux batch`, shell -> `source`, batch -> `cmd.exe /c`.
  Honor `dryrun` by printing the command instead of executing.
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

Replace the four parallel `if/elif` blocks with a small dispatch table keyed on
`submission_type`, mapping each system to (a) its template file name and (b) its submit
command. This keeps flux/shell/batch working without duplicating orchestration logic. If we
decide to drop non-slurm systems entirely, do so explicitly and document it — but the current
`main` behavior supports all four, so preserve unless a decision says otherwise (see Open
Questions).

---

## 4. Work breakdown

Ordered, each item small enough to review independently.

### Phase A — make the branch correct and runnable
- [ ] A1. Add `--dryrun` (store_true) to `cli.add_default_args`; remove the commented-out
      `--studyIDX` block or restore it if still needed by any harness input template
      (`training_input.tmpl` still emits `--studyIDX <studyIDX>`, so the *train script* arg
      is still needed — confirm the CLI-level `--studyIDX` truly is obsolete).
- [ ] A2. Remove unused imports from `src/yoke/cli/start_study.py`.
- [ ] A3. Add docstrings + type annotations to all `HarnessStudy` methods (satisfy `D`/`ANN`).
- [ ] A4. Run `ruff check` / `ruff format` and fix.

### Phase B — restore full submission-system support
- [ ] B1. Thread `submission_type` from `args` -> `HarnessStudy` -> `submit_job`.
- [ ] B2. Introduce a dispatch table for {template filename, submit command} per system.
- [ ] B3. Generalize `generate_initial_inputs` / `_get_slurm_template` to the selected
      submission template (not slurm-only).

### Phase C — decouple from deprecated JSON-to-SLURM
- [ ] C1. Remove `create_slurm_files` usage from `base.py` and `start_study.py`; harnesses
      always supply a complete submission script template.
- [ ] C2. Mark `yoke.helpers.create_slurm_files` and any `slurm_config.json` handling as
      **deprecated** (module docstring + `DeprecationWarning`), scheduled for removal in the
      follow-up task. Do not delete yet (its test `tests/helpers/test_slurm_create.py` still
      exists); coordinate removal with that cleanup task.
- [ ] C3. Remove `slurm_config.json` from the one harness that still ships it
      (`chicoma_lsc_loderunner-ch-subsampling`) as part of that harness's migration.

### Phase D — continuation contract
- [ ] D1. Document and lock the contract between `HarnessStudy.generate_tmpl_inputs` (writes
      `training_input.tmpl` / `training_slurm.tmpl`) and
      `yoke.utils.restart.continuation_setup` (consumes them on the compute node).
- [ ] D2. Add a round-trip test: generate templates, run `continuation_setup` against them in
      a temp dir, assert the restart `.input`/`.slurm` are well-formed (correct `epochIDX`,
      `INPUTFILE`, `CHECKPOINT`).

### Phase E — tests
- [ ] E1. Create `tests/harnesses/test_base.py` and `tests/cli/test_start_study.py`.
- [ ] E2. Unit-test `render_template` conditional blocks (optional block present/absent).
- [ ] E3. Unit-test `load_hyperparameters` (CSV -> list[dict], comment handling, index).
- [ ] E4. Test `run_study` end-to-end in a temp dir with `dryrun=True` (assert files created,
      submit command printed, nothing executed).
- [ ] E5. Ensure `--cov` stays healthy for the new modules.

### Phase F — migrate harnesses + docs
- [ ] F1. Migrate the maintained harnesses to single-template form (drop `training_START.*`):
      `ch_DDP_loderunner`, `ch_lightning_loderunner`, `ch_lsc_policy`,
      `chicoma_lsc_loderunner-ch-subsampling`, `mini-run-test`, `mnist_surrogate`,
      `moving_mnist`, `se_DDP_loderunner*`, `ch_lsc_inverse`, `ch_lsc_reward`,
      `ch_DDP_diffLDR`, `lsc_action`.
- [ ] F2. Update each harness `README.md` to show the `yoke-start-study` invocation.
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
6. Launch: `yoke-start-study --csv <csv> --submissionType slurm [--dryrun]`.
7. Document the reserved keys and the continuation lifecycle.

---

## 7. Acceptance criteria

- `yoke-start-study` runs from a harness dir with `--dryrun` and prints the exact submit
  commands without executing them.
- A single `training_input.tmpl` + single submission template generate both first-launch and
  continuation artifacts; no `training_START.*` files remain in migrated harnesses.
- All four submission systems (slurm/flux/shell/batch) work, OR a documented decision drops
  some of them.
- No runtime dependency on `create_slurm_files`; it and `slurm_config.json` are marked
  deprecated.
- New unit tests cover `HarnessStudy` and the CLI; `ruff check`, `ruff format --check`, and
  `pytest -Werror --cov` pass.
- `docs/` contains an "Authoring a Harness" guide; harness `README.md`s show the new CLI.

---

## 8. Open questions

- **Keep flux/shell/batch?** `main` supports all four; the branch's `base.py` currently only
  does slurm. Confirm whether non-slurm systems remain first-class or are dropped.
- **CLI-level `--studyIDX`.** It was commented out, but `training_input.tmpl` still passes
  `--studyIDX <studyIDX>` to the *train script*. Confirm the CLI-level flag is truly
  obsolete (it appears the studyIDX now always comes from the CSV index).
- **Registry vs. convention.** Should harnesses register themselves (e.g., a
  `Harness` subclass per harness) for stronger "class instantiation", or is the current
  "one generic `HarnessStudy` + per-harness template files" the intended end state? The task
  description mentions a "class instantiation methodology" — clarify how far to take it
  (generic class parameterized by files vs. subclass-per-harness).
- **`continuation_setup` placement.** Should the compute-node continuation logic move into
  `HarnessStudy` (as a method) so the templating contract lives in one class, rather than
  split between `harnesses/base.py` and `utils/restart.py`?
