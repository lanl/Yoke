LodeRunner Fine-tuning - DDP - Selene - CYLEX
=============================================

Selene (`se_`) DDP LodeRunner harness that **fine-tunes** a model pretrained on
`lsc240420` onto the `cx241203` (CYLEX) dataset. It shares the CYLEX training
script with `se_DDP_loderunner_cylex` but adds a fine-tuning phase: the model is
initialized from a pretrained checkpoint, its backbone is optionally frozen for
the first few epochs, and a reduced warmup learning rate is applied during that
frozen phase.

Files
-----

- `train_LodeRunner_ddp_cylex.py` — DDP training script with fine-tuning logic.
- `training_input.tmpl` — single input template. It always carries the
  fine-tuning arguments (`--pretrained_model`, `--freeze_backbone_epochs`,
  `--warmup_lr`); the training script applies them only on the first cycle and
  ignores them once a continuation checkpoint is loaded. The
  `# <<optional:CONTINUATION>>` block adds `--continuation`/`--checkpoint` on
  continuation runs.
- `training_slurm.tmpl` — complete SLURM submission script.
- `cp_files.txt` — files copied into each `study_###` run directory.
- `ddp_test.csv` — fine-tuning study configuration.

Fine-tuning behavior
---------------------

- `--pretrained_model` — path to the pretrained (weights-only) checkpoint used to
  initialize the model on the first cycle.
- `--freeze_backbone_epochs` — number of initial epochs to keep the backbone
  frozen.
- `--warmup_lr` — learning rate used while the backbone is frozen.

On continuation the script loads the full checkpoint (model + optimizer) and the
pretrained-init path is skipped.

Launch
------

From within this directory, with the Yoke environment active:

```bash
yoke-start-study --csv ddp_test.csv --submissionType slurm
```

Add `--dryrun` to render the study directories and print the `sbatch` commands
without submitting any jobs.
