# Loss Catalog

The losses in `src/yoke/losses/`, plus the PyTorch losses Yoke harnesses use.
The most important property of each one is its **output shape**. Every Yoke
datastep reduces the loss itself, so it needs an unreduced loss of a particular
shape (see `skills/shared/batch_contracts.md`). A loss with the right maths
and the wrong shape either raises an error or trains on the wrong objective.

Shapes below are for a prediction and target of shape `(B, C, H, W)`. They were
checked by running each loss on random tensors (2026-11-11).

## Summary

| Loss | `reduction` | Output shape | Datasteps that accept it |
| --- | --- | --- | --- |
| `torch.nn.MSELoss` | `"none"` | `(B, C, H, W)` (elementwise) | loderunner 1-step / cylex / `pli_2frame`, array, diffusion |
| `torch.nn.L1Loss` | `"none"` | `(B, C, H, W)` (elementwise) | as MSE |
| `yoke.losses.lp_loss.LpLoss` | `"none"` | `(B, C)` (per channel) | `pli_2frame_lp`, 2-frame varchan, array |
| `yoke.losses.lp_loss.LpLoss` | `"channel_mean"` | `(B,)` (per sample) | array (records it as is) |
| `yoke.losses.NormMSE.NormalizedMSELoss` | `"none"` | `(B, C, H, W)` (elementwise) | as MSE |
| `yoke.losses.masked_loss.MaskedLossMultiplicative` | of the wrapped loss | same as the wrapped loss | as the wrapped loss |
| `yoke.losses.masked_loss.CroppedLoss2D` | of the wrapped loss | wrapped loss on the cropped `(h, w)` | as the wrapped loss |

Reductions to scalars (`"mean"`, `"sum"`) **cannot** be used with any Yoke
training step: the array datasteps raise `ValueError` on a scalar, and the
others fail or record nonsense.

The default `HarnessTrainer` loss is
`yoke.utils.builders.default_mse_loss`, which is `MSELoss(reduction="none")`.

## How each training step reduces the loss

- **Elementwise** loss `(B, C, H, W)`: the datastep records
  `loss.mean(dim=[1, 2, 3])` per sample and backpropagates `loss.mean()`.
  Every element counts equally, so channels with large values dominate.
- **Per-channel** loss `(B, C)` (`pli_2frame_lp`, varchan): it records
  `.mean(dim=1)` and backpropagates `.mean()`. Each channel counts equally,
  whatever its scale.
- **Array** datasteps accept any non-scalar shape. They record the mean over
  all non-batch dimensions.
- **lsc reward**: the loss must already be per sample, `(B, 1)`.

## Losses in detail

### `LpLoss` (`src/yoke/losses/lp_loss.py`)

```python
LpLoss(d: int = 2, p: int = 2, method: str = "abs", eps: float = 1e-4,
       reduction: str = "none")
```

- For each `(sample, channel)`, takes the Lp norm of the error over the last
  `d` dimensions (`d=2` for `H, W`). For `p=2` this is `sqrt(sum(e**2) + eps)`.
  There is **no** division by the number of elements, so values are much
  larger than MSE (plot loss curves with a larger `--ylim`).
- `method="rel"` divides by the target's norm, giving a relative error.
- Gradient per channel is `e / ||e||`, so channels of very different physical
  scale are balanced without normalizing the data. This is why the 2-frame
  LodeRunnerViT harnesses use it on raw fields.
- `reduction`: `"none"` → `(B, C)`, `"channel_mean"` → `(B,)`, `"mean"` /
  `"sum"` → scalar (do not use with Yoke datasteps).
- Used in `applications/harnesses/ch_ldrViT_lsc_cyl/` and
  `applications/harnesses/ch_ldrViT/train_ldrViT_2frame_8ch_lp.py`.

### `NormalizedMSELoss` (`src/yoke/losses/NormMSE.py`)

```python
NormalizedMSELoss(eps: float = 1e-8, reduction: str = "none")
```

- Normalizes prediction and target by the **target's** per-channel mean and
  standard deviation over dims `(0, 2, 3)`, then takes squared error.
- **Caveat:** the statistics come from the current **batch**, so the loss for
  one sample depends on the other samples in its batch and on the batch size.
  With small batches (or `batch_size=1`) the statistics are noisy. Under DDP
  each rank normalizes by its own batch. Prefer precomputed normalization
  (see `src/yoke/datasets/lsc_normalization.py`) or `LpLoss` for scale
  balancing.
- Expects 4-D input `(B, C, H, W)`.

### `MaskedLossMultiplicative` (`src/yoke/losses/masked_loss.py`)

```python
MaskedLossMultiplicative(loss_fxn: Callable, mask: torch.tensor = torch.tensor(1.0))
```

- Multiplies prediction and target by `mask` (broadcast), then calls
  `loss_fxn`. Masked-out regions give zero error. With an elementwise loss
  the zeros still count in the datastep's mean, so the recorded loss is
  diluted by the masked fraction.
- `mask` is a registered **buffer**. Since 2026-11-11, `HarnessTrainer` moves
  `nn.Module` losses to the training device, so the mask follows. Before that
  fix it stayed on the CPU.
- Output shape is that of `loss_fxn`.

### `CroppedLoss2D` (`src/yoke/losses/masked_loss.py`)

```python
CroppedLoss2D(loss_fxn: Callable, crop: Iterable)   # crop = (top, left, height, width)
```

- Crops both tensors with `torchvision.transforms.functional.crop` before
  `loss_fxn`. Use it to ignore padding added to meet a model's size
  constraints.
- Output shape: that of `loss_fxn` on the cropped region. It is used in the
  Lightning harness only.

## Wiring a loss into a harness

`HarnessTrainer` takes `loss_builder`, a callable with **no arguments** that
returns the loss. Pass arguments with `functools.partial` or a small
function:

```python
import functools
from yoke.losses.lp_loss import LpLoss

loss_builder = functools.partial(LpLoss, d=2, p=2, method="abs", eps=1e-4,
                                 reduction="none")
```

```python
def build_loss() -> nn.Module:
    """Masked MSE ignoring the padded border."""
    mask = torch.ones(1120, 400)
    mask[-20:, :] = 0.0
    return MaskedLossMultiplicative(nn.MSELoss(reduction="none"), mask)
```

- Tensors a loss needs (masks, weights) must be **registered buffers**
  (`self.register_buffer("weight", w)`), not plain attributes, so the trainer's
  `.to(device)` moves them.
- Loss settings that vary per study become CLI flags of the train script, and
  `<KEY>` columns in the study CSV if they are swept.
