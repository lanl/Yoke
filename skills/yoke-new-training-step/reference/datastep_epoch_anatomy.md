# Anatomy of a Datastep / Epoch Pair

How Yoke's training steps are built. Use this to write a new pair that
`HarnessTrainer` can drive without changes. Reference implementations:

- `yoke.utils.training.datastep.loderunner.train_DDP_loderunner_2frame_varchan_datastep`
  and `yoke.utils.training.datastep.loderunner.eval_DDP_loderunner_2frame_varchan_datastep`
  (the newest, with input checks and helpers);
- `yoke.utils.training.epoch.loderunner.train_DDP_loderunner_multidataset_epoch`
  (a newer epoch function with EMA, grad clip, and tagged records);
- `yoke.utils.training.epoch.loderunner.train_DDP_loderunner_epoch`
  (dispatch through `DATASTEP_FN`);
- `yoke.utils.training.datastep.array_output.train_DDP_array_datastep` and
  `yoke.utils.training.epoch.array_output.train_DDP_array_epoch` (the
  simplest DDP pair).

## Where the code goes

| Piece | File |
| --- | --- |
| datasteps | `src/yoke/utils/training/datastep/<family>.py` |
| epoch function | `src/yoke/utils/training/epoch/<family>.py` |
| datastep tests | `tests/utils/training/datastep/test_<family>_datastep.py` |
| epoch tests | `tests/utils/training/epoch/test_<family>_epoch.py` |

Prefer, in order:

1. **A new `DATASTEP_FN` entry** when the new datasteps take the same
   arguments as the loderunner DDP steps,
   `(data, model, optimizer, loss_fn, device, rank, world_size, channel_map,
   grad_clip=...)` for train and
   `(data, model, loss_fn, device, rank, world_size, channel_map)` for eval,
   and return `(truth, pred, all_losses)`. Then `train_DDP_loderunner_epoch`
   is reused as is (see "Extending `DATASTEP_FN`").
2. **A new epoch function in an existing module** when the loop differs (a
   different return tuple, extra record columns, no `channel_map`), as with
   `train_DDP_loderunner_multidataset_epoch`.
3. **A new module** only for a new model family.

## The `train` datastep

```python
def train_DDP_<family>_datastep(
    data: tuple,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
    grad_clip: float | None = None,      # plus channel_map etc. if needed
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
```

In order:

1. `model.train()`.
2. Unpack `data` in **one line** (`a, b, c = data`). The skills CI test
   reads this line to check `skills/shared/batch_contracts.md`.
3. Move tensors to `device` with `non_blocking=True`. Cast times to float32.
4. Forward pass.
5. Loss. Decide and document the shape `loss_fn` must return:
   - elementwise (for example `MSELoss(reduction="none")`): record
     `loss.mean(dim=<all non-batch dims>)`, backprop `loss.mean()`;
   - per-channel `(B, C)` (for example `LpLoss(reduction="none")`): record
     `.mean(dim=1)`, backprop `.mean()`;
   - per-sample `(B,)` or `(B, 1)`: record as is, backprop `.mean()`.

   **Check the shape and raise `ValueError` on a mismatch** (see
   `_varchan_lp_loss`), so a wrong loss fails at the first step instead of
   training on the wrong objective.
6. `optimizer.zero_grad(set_to_none=True)`, `backward()`, then
   `torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)` if
   `grad_clip` is not `None`, then `optimizer.step()`.
7. Gather per-sample losses for recording:

   ```python
   gathered = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
   dist.all_gather(gathered, per_sample_loss.detach())
   all_losses = torch.cat(gathered, dim=0) if rank == 0 else None
   ```

   All ranks must call `all_gather` (it is a collective). Extra per-sample
   tags (for example dataset ids) are gathered the same way (see
   `_gather_losses_and_ids`).
8. Return `(truth, pred, all_losses[, extras])`.

Gradients are averaged across ranks by DDP's backward. The datastep does not
reduce gradients itself.

## The `eval` datastep

Same as train, minus the optimizer: `model.eval()`, forward under
`torch.no_grad()`, the same loss and shape check, the same gather, and the
same return tuple. Its signature drops `optimizer` and `grad_clip`.

## The epoch function

`HarnessTrainer` calls `epoch_fn(**kwargs)` with **these keywords** (see
`_epoch_call_kwargs` in `src/yoke/harnesses/trainer.py`), so the epoch
function must accept all of them:

```
training_data, validation_data, num_train_batches, num_val_batches, model,
optimizer, loss_fn, LRsched, epochIDX, train_per_val, train_rcrd_filename,
val_rcrd_filename, device, rank, world_size
```

plus everything in the harness's `epoch_kwargs`, plus `ema_model` and
`global_step` when an EMA hook is set. If it returns an `int`, the trainer
stores it as the new `global_step`.

The loop (copy the structure of `train_DDP_loderunner_multidataset_epoch`):

1. `model.train()`. Replace `<epochIDX>` in `train_rcrd_filename` with
   `f"{epochIDX:04d}"`.
2. Open the record file in append mode **on rank 0 only**
   (`open(...) if rank == 0 else nullcontext()`).
3. For `trainbatch_ID, traindata in enumerate(training_data)`: stop at
   `num_train_batches`; call the train datastep; call `LRsched.step()` (every
   batch); `global_step += 1`; update the EMA if
   `ema_model is not None and global_step > ema_update_after_step` (on
   `model.module` if wrapped); on rank 0, write the records.
4. If `epochIDX % train_per_val == 0`: `model.eval()`, run the same loop
   over `validation_data` with the eval datastep under `torch.no_grad()`, and
   write records.
5. Return `global_step` (if the function supports EMA), else `None`.

Record rows are `epoch, batch, loss`, one row per sample, written with
`np.savetxt(..., fmt="%d, %d, %.8f")` or an f-string. Extra columns go at
the end (`, dataset`). Note that `yoke-plot-loss` reads these files. Check
that it handles any new column; the `--dataset` option already handles a
fourth column.

## Extending `DATASTEP_FN`

In `src/yoke/utils/training/epoch/loderunner.py`:

1. Import the new datasteps at the top.
2. Add their names to `__all__`. The table resolves names through
   `globals()`, and `__all__` keeps linters from flagging the imports.
3. Add an entry:

   ```python
   "<tag>": {
       "train_ddp": "train_DDP_<family>_datastep",
       "eval_ddp": "eval_DDP_<family>_datastep",
       "eval": "<a non-DDP eval datastep with the same contract>",
   },
   ```

4. The harness selects it with `epoch_kwargs={"dataset": "<tag>", ...}`.
5. Add a row per datastep to `skills/shared/batch_contracts.md`.

## Gotchas

- The `"eval"` (non-DDP) entries are used by `eval_loderunner_epoch` for
  post-training evaluation scripts. Give them a datastep with the same
  contract, or note in the card that non-DDP eval is unsupported. (The
  existing 2-frame entries point at a 1-step eval datastep.)
- Don't leave debug `print`s in steps. They run every batch on every rank.
- `torch.tensor(list_of_ints)` for `in_vars`/`out_vars` must be built on
  `device`.
- Keep the step free of file I/O. Recording belongs to the epoch function.
- With a variable channel map, all samples in a batch must share it. Check
  this and raise (see `_varchan_vars`).
