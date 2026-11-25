# Known Issues Found While Writing the Agent Skills

Problems in existing Yoke code and docs, found on 2026-11-10 while building
`skills/` (see `dev_plans/agent_skills_dataset_and_harness.md`). Unless marked
FIXED, they are still open. Each was confirmed by reading the source. The skill
docs warn agents about issues 1–4, so update those notes when an issue is fixed.

## 1. Diffusion training steps hard-code channels 0–7

- **Where:** `src/yoke/utils/training/datastep/diff_loderunner.py:58`
  (`train_diffusion_loderunner_datastep`) and `:126`
  (`train_DDP_diffusion_loderunner_datastep`).
- **What:** both take `in_vars`/`out_vars` arguments but ignore them and build
  `vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])` for the model call.
- **Impact:** a diffusion model with a channel count other than 8, or a
  different channel subset, trains on the wrong variable indices without any
  error.
- **Fix idea:** use the passed `in_vars`/`out_vars`. Check whether the two
  eval steps already do, and add a test that a non-default `in_vars` reaches
  the model.
- **Skill note to update:** "diffusion loderunner" section of
  `skills/shared/batch_contracts.md`.

## 2. Non-DDP 2-frame eval entries in `DATASTEP_FN` point at the 1-step step

- **Where:** `src/yoke/utils/training/epoch/loderunner.py:64` and `:69`.
  These are the `"eval"` entries of `"pli_2frame"` and `"pli_2frame_lp"`.
- **What:** both map to `eval_loderunner_datastep`, which unpacks
  `(start_img, end_img, Dt)` and calls the model with one frame.
- **Impact:** `eval_loderunner_epoch(..., dataset="pli_2frame")` (single
  process, used by evaluation scripts) fails or misbehaves on 2-frame batches
  `(input_frames, end_img, lead_times)`.
- **Fix idea:** add non-DDP 2-frame eval datasteps (elementwise and LpLoss
  variants) and point the entries at them, with tests. Then add rows to
  `skills/shared/batch_contracts.md`; the skills CI test will require them.
- **Skill notes to update:** "The `DATASTEP_FN` table" caveat in
  `skills/shared/batch_contracts.md`, and "Gotchas" in
  `skills/yoke-new-training-step/reference/datastep_epoch_anatomy.md`.

## 3. Debug `print` in the cylex training step

- **Where:** `src/yoke/utils/training/datastep/loderunner.py:265`
  (`train_DDP_loderunner_datastep_cylex`).
- **What:** `print("In train_DDP_loderunner_datastep: channel_map =", ...)`
  runs on every batch on every rank. The message also names the wrong
  function.
- **Fix idea:** delete it, or use `yoke.helpers.logger` at debug level.

## 4. Shell-submitted harnesses resubmit continuations as SLURM

- **Where:** `src/yoke/harnesses/trainer.py:480` reads
  `getattr(self.args, "submissionType", "slurm")`. The flag is declared in
  `yoke.helpers.cli.add_default_args` (`src/yoke/helpers/cli.py:85`), but no
  `training_input.tmpl` passes it.
- **What:** `yoke-start-study --submissionType shell` affects only the first
  launch. On continuation the train script falls back to `slurm` and runs
  `sbatch` unless its input file contains `--submissionType shell`.
- **Fix idea:** have `HarnessStudy` inject the submission type into the
  rendered input (for example a `<submissionType>` substitution or an
  appended line), or record it in the study directory for
  `continuation_setup` to read. Add a test in `tests/harnesses/`.
- **Skill notes to update:** the "Submission template" section of
  `skills/yoke-new-harness/reference/study_csv_and_templates.md`, and step 5
  of `skills/yoke-new-harness/SKILL.md`.

## 5. `AGENTS.md` claims optional blocks work in submission templates

- **Where:** `AGENTS.md` §4 ("A submission template ... with conditional
  blocks") and §5 ("single template per artifact with conditional blocks").
- **What:** `HarnessStudy._render_submission_template`
  (`src/yoke/harnesses/base.py:288`) does plain `<KEY>` substitution with no
  `# <<optional:KEY>>` handling. Only `training_input.tmpl` and the evaluation
  templates (through `render_template`) support optional blocks. The
  `#PRODUCTION`/`#DEBUG` sections of existing SLURM templates are toggled by
  hand.
- **Fix idea:** either correct the `AGENTS.md` wording, or render the
  submission template through `render_template` too. In that case also note
  that unknown `optional:` keys drop their block.

## 6. `AGENTS.md` line lengths

- Most existing `AGENTS.md` lines exceed the 89-character limit used elsewhere
  in the repo. This is cosmetic. Reflow the file if anyone touches it broadly.

## 7. Loss not moved to the training device — FIXED 2026-11-11

- **Where:** `HarnessTrainer.setup()` in `src/yoke/harnesses/trainer.py`.
- **What:** the loss from `loss_builder()` was never moved to the device, so
  a loss with registered buffers (for example `MaskedLossMultiplicative`'s
  `mask`) stayed on the CPU while the predictions were on the GPU.
- **Fix:** `setup()` now calls `.to(self.device)` on `nn.Module` losses and
  leaves plain callables as they are. Tests are in
  `tests/harnesses/test_trainer_preflight.py`. Part of agent skills phase 2.

## 8. `make_distributed_dataloader` crashed with `num_workers=0` — FIXED 2026-11-11

- **Where:** `src/yoke/utils/dataload.py` (`make_distributed_dataloader`).
- **What:** it always passed `prefetch_factor=2`, which `torch.utils.data.DataLoader`
  rejects when `num_workers=0`. Any harness run with `--num_workers 0` failed
  at dataloader construction.
- **Fix:** pass `prefetch_factor=None` when `num_workers == 0`. Found while
  adding preflight, which uses no workers. A test was added in
  `tests/test_training_utils.py`.

## 9. `yoke-plot-loss` failed on array-epoch records — FIXED 2026-11-13

- **Where:** `yoke.plots.loss_curves.load_record_csv`
  (`src/yoke/plots/loss_curves.py`).
- **What:** `train_DDP_array_epoch` (and `train_lsc_policy_epoch`) write an
  `epoch,batch,loss` header at the start of every epoch. All first-cycle
  epochs append to one file, and the reader (`sep=", "`, no header handling)
  raised `ParserError: Expected 1 fields in line 2, saw 3`.
- **Fix:** skip lines starting with `epoch` before parsing. Test in
  `tests/plots/test_loss_curves.py`. Found during the TPBF harness work.

## 10. `train_DDP_array_epoch` needs a scheduler

- **Where:** `src/yoke/utils/training/epoch/array_output.py:165`.
- **What:** it calls `LRsched.step()` every batch, but `HarnessTrainer`'s
  `scheduler_builder` defaults to `None`, so a harness that omits it crashes
  with `AttributeError` on the first batch.
- **Fix idea:** skip the step when `LRsched is None` (as other epoch
  functions may need too). Until then, array harnesses must pass a
  `scheduler_builder`. Open.

## 11. LR schedulers restart their warmup on continuation

- **Where:** `src/yoke/lr_schedulers.py` (all three schedulers use
  `self._step_count`) and `yoke.utils.builders.compute_last_epoch`
  (one epoch short).
- **What:** PyTorch resets `_step_count` to 1 whenever a scheduler is built,
  so every continuation job restarts the LR schedule. Found 2026-11-13.
- **Details, reproduction, and fix plan:**
  `dev_plans/lr_scheduler_continuation_bug.md`. Open.

## Decision recorded: optional reader dependencies

`tifffile`, `imageio`, and `PIL` are importable in Yoke environments only
because `scikit-image` depends on them. They will **not** be added to
`pyproject.toml` until a real dataset needs one. Add the dependency then, in
the same change as that dataset. Until then, package code should use
`skimage.io` for images, as `skills/yoke-new-dataset/reference/format_inspection.md`
already says.
