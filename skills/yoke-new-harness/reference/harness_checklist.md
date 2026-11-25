# Harness Checklist

What a `HarnessTrainer`-based harness directory contains, and what each file
must do. Copy the structure of the closest existing harness. The newest and
cleanest example is `applications/harnesses/ch_ldrViT_lsc_cyl/`. For a 1-step
`LodeRunner`, see `applications/harnesses/se_DDP_loderunner/`. For one with
post-training evaluation, see `applications/harnesses/ch_DDP_loderunner/`.

Full write-up: `docs/source/harness_trainer.rst` and `AGENTS.md` §4–5a.

## Files

| File | Required | Purpose |
| --- | --- | --- |
| `train_<name>.py` | yes | Thin `HarnessTrainer` wrapper |
| `training_input.tmpl` | yes | Argument file template (see `study_csv_and_templates.md`) |
| `training_slurm.tmpl` or `training_shell.tmpl` | yes | Submission script (from `yoke-slurm-template`, or copied) |
| `cp_files.txt` | yes | Files copied into each study dir; at least the train script |
| `<study>.csv` | yes | Hyperparameter rows; first column `studyIDX` |
| `<study>_smoke.csv` | recommended | 2–3 tiny rows for a first test on the cluster |
| `README.md` | yes | What it trains, files, study columns, how to launch, caveats |
| `evaluation_input.tmpl` + `evaluation_<slurm,shell>.tmpl` + `eval_<name>.py` | optional | Post-training evaluation |

Directory naming: existing harnesses use a machine prefix (`ch_` Chicoma,
`vt_` Venado, `se_` ...) plus the model and data. Ask the user.

## The train script

Structure (see `train_ldrViT_2frame_lsc_cyl.py`):

1. Module docstring: what is trained, on what data, with which loss,
   batch-contract constraints (for example "batch size must be 1"), and
   which reference script it was based on.
2. Parser:

   ```python
   parser = argparse.ArgumentParser(prog=..., description=descr_str,
                                    fromfile_prefix_chars="@")
   parser = cli.add_default_args(parser=parser)    # also --submissionType
   parser = cli.add_filepath_args(parser=parser)
   parser = cli.add_computing_args(parser=parser)
   parser = cli.add_model_args(parser=parser)
   parser = cli.add_training_args(parser=parser)
   parser = cli.add_cosine_lr_scheduler_args(parser=parser)  # or step LR
   parser.add_argument(...)                        # harness-specific flags
   ```

   `HarnessTrainer` reads these from `args`: `train_batches`, `val_batches`,
   `batch_size`, `num_workers`, `TRAIN_PER_VAL`, `trn_rcrd_filename`,
   `val_rcrd_filename`, `cycle_epochs`, `total_epochs`, `studyIDX`,
   `continuation`, `checkpoint`, and `submissionType` (default `slurm`).
   The builders also read `init_learnrate` and `weight_decay` if present.
3. `make_model_args(args) -> dict`: the model's constructor kwargs (see
   `model_catalog.md`). Its keys are saved in the checkpoint and used again on
   continuation.
4. `build_dataset(args) -> (train_dataset, val_dataset)`: build the datasets
   from CLI paths and filelists (`args.FILELIST_DIR + args.train_filelist`).
5. Optional builders: `build_optimizer(model, args)` (default
   `yoke.utils.builders.build_adamw`), `build_scheduler(optimizer, args,
   last_epoch)`, and `build_loss()` (default
   `yoke.utils.builders.default_mse_loss`, i.e. `MSELoss(reduction="none")`;
   use `yoke.losses.lp_loss.LpLoss(..., reduction="none")` for per-channel
   contracts).
6. `if __name__ == "__main__":` parse args, check contract constraints
   (`parser.error(...)`), build `epoch_kwargs` and hooks, and run:

   ```python
   HarnessTrainer(
       args,
       model_builder=build_from_checkpoint(ModelClass, make_model_args,
                                           optimizer_kwargs={...}),
       dataset_builder=build_dataset,
       epoch_fn=<epoch function from batch_contracts.md>,
       optimizer_builder=build_optimizer,
       scheduler_builder=build_scheduler,
       loss_builder=build_loss,
       epoch_kwargs={...},      # channel_map, dataset tag, grad_clip, ...
       hooks=hooks,             # e.g. EMA via yoke.utils.ema.make_ema_hooks
       evaluate_after_training=False,
       device_type="cuda",      # "cpu" on a CPU-only machine (one process)
   ).run()
   ```

   `device_type="cpu"` runs one process on the CPU with no GPU or NCCL
   setup (see `docs/source/harness_trainer.rst`, "CPU-only machines"). Its
   SLURM template must launch a single task.

   All work happens under the `__main__` guard, so importing the module only
   builds `parser`. The verification step relies on this.

Key helpers:

- `yoke.utils.builders.build_from_checkpoint(model_class, make_model_args,
  optimizer_kwargs=...)` covers fresh starts and continuation.
  `yoke.utils.builders.build_from_pretrained` (same arguments) adds a start
  from another checkpoint's weights via `--pretrained_model`. Write a
  bespoke `model_builder` only for backbone surgery (see
  `applications/harnesses/se_DDP_loderunner_finetune_cylex/`).
- EMA: `on_after_ddp_wrap, on_before_save = make_ema_hooks(ModelClass,
  max_decay=..., inv_gamma=..., power=...)`, wrapped in
  `yoke.harnesses.trainer.TrainerHooks`, plus `ema_update_after_step` in
  `epoch_kwargs` (only for epoch functions that support EMA; see
  `batch_contracts.md`).
- `grad_clip` goes in `epoch_kwargs` for the loderunner epoch functions.

## `cp_files.txt`

One path per line, relative to the harness dir: the train script, the
evaluator (if any), and any harness-local helper modules or files. It
doesn't list data, which is read from the paths in `training_input.tmpl`.

## README.md

Cover: what the harness trains and on which data, the model and loss, the
files, the CSV columns and what each varies, any constraints (batch size,
channel sets), data and filelist paths that must exist on the machine, how
to launch (`yoke-start-study ...`), and how to plot loss
(`yoke-plot-loss --basedir ./runs --IDX N`).

## Lint and CI

CI does **not** lint `applications/harnesses`, but keep new scripts clean
with `ruff check`, `ruff check --preview`, and `ruff format --check`. Shared
logic belongs in `src/yoke` (with tests), not in the harness.
