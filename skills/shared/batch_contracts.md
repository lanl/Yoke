# Batch Contracts: Dataset → Datastep → Epoch Function

A Yoke dataset works with a training loop only if its `__getitem__` returns
the tuple the **datastep** unpacks from each batch (`... = data`). The
**epoch function** calls the datasteps and is what a harness passes to
`HarnessTrainer` as `epoch_fn`. This file is the map between the three.

Shapes below are per **batch** (after the default `DataLoader` collation,
which adds the leading `B`). A dataset returns the same tuple **without** `B`.

`tests/skills/test_skill_files.py` checks every row of the table below
against the source. If you rename or change a datastep, update this table in
the same change.

## Contract table

One row per public datastep. `tuple` is exactly what the function unpacks
from `data`, in order.

<!-- contracts:begin -->
| contract | datastep | tuple |
| --- | --- | --- |
| array | `yoke.utils.training.datastep.array_output.train_array_datastep` | `inpt, truth` |
| array | `yoke.utils.training.datastep.array_output.eval_array_datastep` | `inpt, truth` |
| array | `yoke.utils.training.datastep.array_output.train_DDP_array_datastep` | `inpt, truth` |
| array | `yoke.utils.training.datastep.array_output.eval_DDP_array_datastep` | `inpt, truth` |
| scalar | `yoke.utils.training.datastep.scalar_output.train_scalar_datastep` | `inpt, truth` |
| scalar | `yoke.utils.training.datastep.scalar_output.eval_scalar_datastep` | `inpt, truth` |
| loderunner 1-step | `yoke.utils.training.datastep.loderunner.train_loderunner_datastep` | `start_img, end_img, Dt` |
| loderunner 1-step | `yoke.utils.training.datastep.loderunner.eval_loderunner_datastep` | `start_img, end_img, Dt` |
| loderunner 1-step | `yoke.utils.training.datastep.loderunner.train_DDP_loderunner_datastep` | `start_img, end_img, Dt` |
| loderunner 1-step | `yoke.utils.training.datastep.loderunner.eval_DDP_loderunner_datastep` | `start_img, end_img, Dt` |
| loderunner cylex | `yoke.utils.training.datastep.loderunner.train_DDP_loderunner_datastep_cylex` | `start_img, channel_map, end_img, channel_map, Dt` |
| loderunner cylex | `yoke.utils.training.datastep.loderunner.eval_DDP_loderunner_datastep_cylex` | `start_img, channel_map, end_img, channel_map, Dt` |
| loderunner cylex | `yoke.utils.training.datastep.loderunner.eval_loderunner_datastep_cylex` | `start_img, channel_map, end_img, channel_map, Dt` |
| loderunner scheduled | `yoke.utils.training.datastep.loderunner.train_scheduled_loderunner_datastep` | `img_seq, Dt` |
| loderunner scheduled | `yoke.utils.training.datastep.loderunner.eval_scheduled_loderunner_datastep` | `img_seq, Dt` |
| loderunner 2-frame | `yoke.utils.training.datastep.loderunner.train_DDP_loderunner_2frame_datastep` | `input_frames, end_img, lead_times` |
| loderunner 2-frame | `yoke.utils.training.datastep.loderunner.eval_DDP_loderunner_2frame_datastep` | `input_frames, end_img, lead_times` |
| loderunner 2-frame | `yoke.utils.training.datastep.loderunner.train_DDP_loderunner_2frame_lp_datastep` | `input_frames, end_img, lead_times` |
| loderunner 2-frame | `yoke.utils.training.datastep.loderunner.eval_DDP_loderunner_2frame_lp_datastep` | `input_frames, end_img, lead_times` |
| loderunner 2-frame varchan | `yoke.utils.training.datastep.loderunner.train_DDP_loderunner_2frame_varchan_datastep` | `input_frames, end_img, lead_times, channel_map, dataset_id` |
| loderunner 2-frame varchan | `yoke.utils.training.datastep.loderunner.eval_DDP_loderunner_2frame_varchan_datastep` | `input_frames, end_img, lead_times, channel_map, dataset_id` |
| diffusion loderunner | `yoke.utils.training.datastep.diff_loderunner.train_diffusion_loderunner_datastep` | `x, y_tau, noise, lead_times, tau` |
| diffusion loderunner | `yoke.utils.training.datastep.diff_loderunner.eval_diffusion_loderunner_datastep` | `x, y_tau, noise, lead_times, tau` |
| diffusion loderunner | `yoke.utils.training.datastep.diff_loderunner.train_DDP_diffusion_loderunner_datastep` | `x, y_tau, noise, lead_times, tau` |
| diffusion loderunner | `yoke.utils.training.datastep.diff_loderunner.eval_DDP_diffusion_loderunner_datastep` | `x, y_tau, noise, lead_times, tau` |
| lsc reward | `yoke.utils.training.datastep.lsc_reward.train_lsc_reward_datastep` | `state_y, stateH, targetH, reward` |
| lsc reward | `yoke.utils.training.datastep.lsc_reward.eval_lsc_reward_datastep` | `state_y, stateH, targetH, reward` |
| lsc policy | `yoke.utils.training.datastep.lsc_policy.train_lsc_policy_datastep` | `state_y, stateH, targetH, x_true` |
| lsc policy | `yoke.utils.training.datastep.lsc_policy.eval_lsc_policy_datastep` | `state_y, stateH, targetH, x_true` |
<!-- contracts:end -->

## Contracts in detail

The epoch function in **bold** is the one to use with `HarnessTrainer`.
`HarnessTrainer` always calls the epoch function with the same keywords
(`training_data`, `validation_data`, `num_train_batches`, `num_val_batches`,
`model`, `optimizer`, `loss_fn`, `LRsched`, `epochIDX`, `train_per_val`,
`train_rcrd_filename`, `val_rcrd_filename`, `device`, `rank`, `world_size`),
plus `epoch_kwargs`. Only DDP epoch functions accept all of these; the
non-DDP ones (`train_simple_*`, `train_array_epoch`, `train_scalar_epoch`)
cannot be used with `HarnessTrainer`.

### array — `(inpt, truth)`

- `inpt`: any `(B, ...)` model input; `truth`: `(B, ...)` target with the
  shape of the model output.
- Model call: `model(inpt)`.
- Loss: must be unreduced (`reduction="none"`); a scalar loss raises
  `ValueError` in the train steps. Backprop `loss.mean()`; recorded
  per-sample loss is the mean over all non-batch dims.
- Epoch functions: **`yoke.utils.training.epoch.array_output.train_DDP_array_epoch`**
  (no `channel_map`, no EMA, no grad clip; writes an `epoch,batch,loss`
  header in the training CSV only). Non-DDP:
  `yoke.utils.training.epoch.array_output.train_array_epoch`.
- Used for image → vector (`Image2VectorCNN`) and vector → image
  (`tCNNsurrogate`) models. The most general contract: if a task is "one
  input tensor → one target tensor", use it.

### scalar — `(inpt, truth)`

- `truth` is `(B,)`; the datastep unsqueezes it to `(B, 1)`.
- Epoch function: `yoke.utils.training.epoch.scalar_output.train_scalar_epoch`
  (single-process only, so not for `HarnessTrainer`). For DDP, return
  `truth` with shape `(1,)` per sample and use the array contract.

### loderunner 1-step — `(start_img, end_img, Dt)`

- `start_img`, `end_img`: `(B, C, H, W)` float32; `Dt`: `(B,)` lead time
  (cast to float32 by the datastep).
- Model call: `model(start_img, in_vars, out_vars, Dt)` with
  `in_vars = out_vars = torch.tensor(channel_map)`.
- Loss: elementwise, e.g. `nn.MSELoss(reduction="none")`; recorded loss is
  the mean over `(C, H, W)`.
- Epoch function: **`yoke.utils.training.epoch.loderunner.train_DDP_loderunner_epoch`**
  with `epoch_kwargs={"channel_map": [...], "dataset": "pli"}`. Supports
  EMA (`ema_update_after_step`) and `grad_clip`. Returns `global_step`.
- Example dataset: `yoke.datasets.lsc_dataset.LSC_rho2rho_temporal_DataSet`.

### loderunner cylex — `(start_img, channel_map, end_img, channel_map, Dt)`

- Like 1-step, but the channel map comes from the batch (both copies are
  the same). The `channel_map` argument of the datastep is ignored.
- Epoch function: **`train_DDP_loderunner_epoch`** with
  `epoch_kwargs={"dataset": "cylex", ...}`.

### loderunner scheduled — `(img_seq, Dt)`

- `img_seq`: `(B, S, C, H, W)` sequence; `Dt`: `(B, 1)`.
- Scheduled sampling over the sequence; channels fixed to 0–7 in the
  datastep.
- Epoch function: `yoke.utils.training.epoch.loderunner.train_scheduled_loderunner_epoch`
  (not DDP; not for `HarnessTrainer`).

### loderunner 2-frame — `(input_frames, end_img, lead_times)`

- `input_frames`: `(B, 2, C, H, W)` ordered `[x_{t-1}, x_t]`; `end_img`:
  `(B, C, H, W)`; `lead_times`: `(B, 2) = (dt_in, dt_out)`.
- Model: `LodeRunnerViT` with `num_input_frames=2`; call
  `model(input_frames, in_vars, out_vars, lead_times)`.
- Two loss conventions, chosen by the `dataset` tag:
  - `"pli_2frame"`: elementwise loss (MSE), recorded mean over `(C, H, W)`.
  - `"pli_2frame_lp"`: `loss_fn` returns `(B, C)` (e.g.
    `yoke.losses.lp_loss.LpLoss(..., reduction="none")`); backprop the mean
    over batch and channel; record the mean over channel.
- Epoch function: **`train_DDP_loderunner_epoch`** with
  `epoch_kwargs={"channel_map": [...], "dataset": "pli_2frame_lp"}`.
- Example dataset: `yoke.datasets.lsc_dataset.LSC_rho2rho_temporal_2frame_DataSet`.

### loderunner 2-frame varchan — `(input_frames, end_img, lead_times, channel_map, dataset_id)`

- As 2-frame, plus `channel_map` `(B, C)` int64 (indices into the model's
  `default_vars`) and `dataset_id` `(B,)` int64.
- All samples in a batch must share one channel map; in practice
  `batch_size=1`.
- `loss_fn` must return `(B, C)`; anything else raises `ValueError`.
- Epoch function: **`yoke.utils.training.epoch.loderunner.train_DDP_loderunner_multidataset_epoch`**
  (calls the varchan datasteps directly; no `channel_map`/`dataset` kwargs).
  Records gain a fourth column, the dataset name from
  `yoke.datasets.pagosa_fields.DATASET_NAMES`.
- Example datasets: `yoke.datasets.pagosa_2frame.LSC_2frame_canonical_DataSet`,
  `yoke.datasets.pagosa_2frame.CYL_2frame_canonical_DataSet`, mixed with
  `yoke.datasets.mixed_dataset.MixedDataSet`.

### diffusion loderunner — `(x, y_tau, noise, lead_times, tau)`

- `x`: `(B, C_in, H, W)` conditioning; `y_tau`: `(B, C_out, H, W)` noised
  target; `noise`: `(B, C_out, H, W)`; `lead_times`, `tau`: `(B,)`.
- Model: `DiffusionLodeRunner`. Loss on `(noise_pred, noise)`.
- Epoch function: `yoke.utils.training.epoch.diff_loderunner.train_DDP_diffusion_loderunner_epoch`,
  which also needs `in_vars`/`out_vars`; the existing harness wraps it
  (`run_epoch` in `applications/harnesses/ch_DDP_diffLDR/train_DDP_diffLDR.py`).
- Note: the DDP train datastep currently builds its own `vars` for channels
  0–7 rather than using `in_vars`/`out_vars`. Check before reusing it with a
  different channel count.

### lsc reward — `(state_y, stateH, targetH, reward)`

- `state_y`: `(B, N)` vector; `stateH`, `targetH`: `(B, C, H, W)`;
  `reward`: `(B,)` (unsqueezed to `(B, 1)`).
- Model call `model(state_y, stateH, targetH)` (`hybrid2vectorCNN`).
- The loss must already be per-sample: `loss_fn(pred, reward)` → `(B, 1)`.
- Epoch function: **`yoke.utils.training.epoch.lsc_reward.train_lsc_reward_epoch`**.

### lsc policy — `(state_y, stateH, targetH, x_true)`

- As reward, but the target `x_true` is `(B, N)` and the model returns a
  distribution; the loss uses its `.mean`.
- Epoch function: **`yoke.utils.training.epoch.lsc_policy.train_lsc_policy_epoch`**
  (optional `blocks` kwarg for per-block gradient monitoring).

## The `DATASTEP_FN` table

`train_DDP_loderunner_epoch` and `eval_loderunner_epoch` pick their datasteps
from `DATASTEP_FN` in `src/yoke/utils/training/epoch/loderunner.py`, keyed by
the `dataset` argument. Each entry has `train_ddp`, `eval_ddp`, and `eval`
names (strings, resolved through the module's globals so tests can patch
them; the functions must therefore be imported in that module and listed in
its `__all__`). Current keys: `pli`, `pli_2frame`, `pli_2frame_lp`, `cylex`.
An unknown key raises `ValueError("Unsupported dataset: ...")`.

Caveat: the `eval` (non-DDP) entries for `pli_2frame` and `pli_2frame_lp`
point at the 1-step `eval_loderunner_datastep`, so `eval_loderunner_epoch`
does not work for 2-frame data as is.

## Matching a new dataset

1. Write down the tuple the dataset will return (names, shapes without `B`,
   dtypes).
2. Find the row above with the same tuple *and* a model call that fits the
   intended model. Names need not match, but order, meaning, and shapes must.
3. If one fits, the harness uses the bold epoch function of that contract.
4. If none fits, either reshape the dataset output to an existing contract
   (often possible: for example, pack extra inputs into `inpt` for the array
   contract), or run `yoke-new-training-step` to add a new datastep/epoch
   pair. Prefer reshaping when it costs nothing in clarity.

## Record files written by epoch functions

- Filenames come from `--trn_rcrd_filename` / `--val_rcrd_filename`; the
  epoch function replaces a literal `<epochIDX>` with the 4-digit epoch.
  (At first launch `yoke-start-study` has already replaced `<epochIDX>` in
  the input file with `0001`, so the first cycle's epochs all append to the
  `epoch0001` files; continuation input files keep the literal token, so
  later epochs get their own files.)
- Only rank 0 writes. Rows are `epoch, batch, loss` (one row per sample);
  the multidataset epoch adds `, dataset`.
- Validation runs when `epochIDX % train_per_val == 0`.
- `LRsched.step()` is called after every training batch, not every epoch.
