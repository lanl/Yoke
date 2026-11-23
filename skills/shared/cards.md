# Handoff Cards

Each Yoke skill ends by writing a short **card** that the next skill reads.
Write the card in the skill's design notes file (`dev_plans/...`) under a
`## Card` heading, and show it in chat. Keep the field names exactly as
below, so a later skill (or a later session) can find them.

Which skill reads which card:

- `yoke-new-harness` reads the Dataset, Model, Loss, Step, and Slurm cards.
- `yoke-new-training-step` reads the Dataset, Model, and Loss cards.
- `yoke-new-loss` reads the Dataset, Model, and Step cards.
- `yoke-launch-study` reads all of them and writes the Run card.
- `yoke-run-doctor` reads the Run card.

When a progress file exists (`dev_plans/<project>_training.md`, see
`skills/shared/interview_conventions.md`), cards are also copied there.

## Dataset card (from `yoke-new-dataset`)

```
Dataset card
- class:        yoke.datasets.<module>.<ClassName>
- module file:  src/yoke/datasets/<module>.py
- test file:    tests/datasets/test_<module>.py
- ctor args:    one per line, "name: type = default"; mark data paths [path]
- returns:      the __getitem__ tuple, each element with shape and dtype,
                e.g. (input_frames[2,C,H,W] float32, end_img[C,H,W] float32,
                      lead_times[2] float32)
- contract:     a row name from skills/shared/batch_contracts.md, or
                "NEW" (then the tuple above is the new contract)
- channels:     ordered channel names (these become the model's
                default_vars or channel_map)
- sample size:  e.g. image (H, W) = (1120, 400), half image
- __len__:      int(1e6) (random sampling) | N (fixed index)
- filelists:    applications/filelists/<name>_{train,validation,test}.txt
- data paths:   which CLI args carry them (e.g. --NPZ_DIR, --FILELIST_DIR)
- constraints:  e.g. "batch_size must be 1", "needs design CSV"
```

## Model card (from `yoke-model`)

```
Model card
- class:         yoke.models.<module>.<ClassName>
- new module:    src/yoke/models/<module>.py, or "existing"
- test file:     tests/models/test_<module>.py, or "n/a"
- model args:    the make_model_args dict, with which keys come from CLI flags
                 or CSV columns
- forward call:  e.g. model(input_frames, in_vars, out_vars, lead_times)
- input/output:  shapes with B, matching the Dataset card
- constraints:   e.g. image size divisible by patch_size, num_input_frames=2
- parameters:    count for the chosen configuration
- memory note:   rough per-GPU (or per-CPU-node) estimate, or "not estimated"
```

## Loss card (from `yoke-new-loss`)

```
Loss card
- class:         yoke.losses.<module>.<ClassName>
- new module:    src/yoke/losses/<module>.py, or "existing"
- test file:     tests/losses/test_<module>.py, or "n/a"
- ctor args:     arguments and the values chosen
- output shape:  what forward returns with the chosen reduction,
                 e.g. (B, C) for reduction="none"
- datasteps:     which datasteps / DATASTEP_FN tags accept this shape
- loss_builder:  the harness snippet, e.g.
                 functools.partial(LpLoss, d=2, p=2, reduction="none")
- caveats:       e.g. "uses batch statistics; depends on batch size"
```

## Step card (from `yoke-new-training-step`)

```
Step card
- epoch fn:      yoke.utils.training.epoch.<module>.<function>
- datasteps:     train: yoke.utils.training.datastep.<module>.<function>
                 eval:  yoke.utils.training.datastep.<module>.<function>
- dataset tag:   DATASTEP_FN key passed as epoch_kwargs["dataset"], or "n/a"
- contract:      the batch tuple the datasteps unpack
- model call:    e.g. model(input_frames, in_vars, out_vars, lead_times)
- loss shape:    what loss_fn must return, e.g. (B, C) or elementwise
- loss builder:  suggested loss_builder for HarnessTrainer
- epoch_kwargs:  extra kwargs the harness must pass, with meaning
- records:       CSV columns written per batch, e.g. epoch, batch, loss
- tests:         test file paths
```

## Slurm card (from `yoke-slurm-template`)

```
Slurm card
- machine:       <ClusterName>
- device:        cuda (multi-GPU DDP) | cpu (HarnessTrainer device_type="cpu",
                 one task per job)
- profile:       skills/yoke-slurm-template/machines/<cluster>.md
- template:      path to training_slurm.tmpl (+ evaluation_slurm.tmpl if any)
- partition/qos: chosen values (and debug variant, if included)
- wall time:     per cycle
- CSV keys:      <KEY>s the template needs from the study CSV,
                 e.g. KNODES, NGPUS, allocation, train_script
                 (CPU mode: no KNODES/NGPUS)
- env:           environment activation line(s) used
- not verified:  anything that can only be checked by a real job
```

## Run card (from `yoke-launch-study`)

Written into the progress file. `yoke-run-doctor` starts from this card in a
later session.

```
Run card
- prepared:      YYYY-MM-DD, by yoke-launch-study
- machine:       <ClusterName>
- harness dir:   applications/harnesses/<name>/
- CSV:           <study.csv> (rows: studyIDX list)
- run dir:       applications/harnesses/<name>/runs/
- command:       the exact yoke-start-study command given to the user
- preflight:     passed YYYY-MM-DD (report summary), or "not run: <why>"
- job prefix:    the #SBATCH --job-name prefix, for finding jobs in sacct
- expected:      record CSVs, .out/.err per cycle, checkpoints
                 studyNNN_modelState_epochEEEE.pth, restart files
- budget:        GPU hours estimate (GPU mode) or node hours (CPU mode)
- submitted:     "by agent at user's request" or "by user"; date; job IDs
                 (filled in later by the user or yoke-run-doctor if unknown)
- doctor log:    dated findings from yoke-run-doctor sessions
```
