# Testing Datasteps and Epoch Functions

Model new tests on:

- `tests/utils/training/datastep/test_loderunner_datastep.py`
- `tests/utils/training/datastep/test_ddp_array_datastep.py`
- `tests/utils/training/epoch/test_loderunner_epoch.py`

These are **unit tests on CPU**. They never start a process group, use a
GPU, or train a real model.

## Rules

- Must pass with `pytest -Werror`.
- Every test and fixture has a docstring and full annotations
  (`monkeypatch: pytest.MonkeyPatch`, `tmp_path: Path`, `-> None`).
- There is no shared `conftest.py`. Define dummy models and fakes in the test
  file.

## Faking `torch.distributed`

No test initializes `torch.distributed`. Patch `all_gather` with a stub that
copies the local tensor into every slot, as if all ranks had the same batch:

```python
import torch.distributed as dist

@pytest.fixture(autouse=True)
def fake_all_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace dist.all_gather with a single-process stand-in."""

    def _all_gather(output_list: list[torch.Tensor], tensor: torch.Tensor) -> None:
        for out in output_list:
            out.copy_(tensor)

    monkeypatch.setattr(dist, "all_gather", _all_gather)
```

Then `@pytest.mark.parametrize("rank", [0, 1])` with `world_size=3`. On rank 0,
check `all_losses.shape == (world_size * B,)`. On other ranks, check
`all_losses is None`.

## Tiny models

- A model with one scalar parameter whose forward matches the real call
  signature, for example:

  ```python
  class DummyModel(nn.Module):
      """Adds a learnable scalar to the input."""

      def __init__(self) -> None:
          """Create the single parameter."""
          super().__init__()
          self.param = nn.Parameter(torch.tensor(1.0))

      def forward(self, x: torch.Tensor, in_vars: torch.Tensor,
                  out_vars: torch.Tensor, lt: torch.Tensor) -> torch.Tensor:
          """Return x + param, ignoring the variable indices."""
          return x + self.param
  ```

- For two-frame input: `x.mean(dim=1) + self.param`.
- Optimizer: `optim.SGD(model.parameters(), lr=0.1)`.
- `device = torch.device("cpu")`.

## Datastep assertions

1. Returned truth equals the input target. The prediction equals what the
   dummy model must produce.
2. The recorded per-sample loss has shape `(B,)` and the expected values.
   For the reduction, compare with the loss computed by hand (for example
   `LpLoss(...)(pred, y).mean(dim=1)`).
3. The backward pass uses the documented reduction: compare the parameter's
   gradient with that of the hand-computed loss.
4. Train updates the parameters
   (`copy.deepcopy(list(model.parameters()))` before and after). Eval leaves
   them unchanged.
5. A wrong-shape loss raises `ValueError` (`pytest.raises(..., match=...)`).
6. `grad_clip` caps the gradient norm, if supported.
7. Rank 0 vs. other ranks (see above).

## Epoch-function assertions

Replace the datasteps with a counting stub **on the epoch module**, because
that's the name the function looks up:

```python
import yoke.utils.training.epoch.<family> as epoch_mod

class DummyStep:
    """Counts calls and returns a fixed one-sample loss."""

    def __init__(self) -> None:
        """Start the call counter at zero."""
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> tuple:
        """Return (truth, pred, losses) with a single 0.5 loss."""
        self.calls += 1
        return None, None, torch.tensor([0.5])

monkeypatch.setattr(epoch_mod, "train_DDP_<family>_datastep", DummyStep())
```

Use a `DataLoader` over two dummy items (`batch_size=1`) and an `nn.Linear`
model with SGD. A real `LRsched` such as `torch.optim.lr_scheduler.StepLR`
needs the module-level warning filter:

```python
# ruff: noqa: E402
import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:Detected call of.*lr_scheduler.step.*:UserWarning"
)
```

Check that:

- record files named with `<epochIDX>` are created with the 4-digit epoch
  (`train_0001.csv`) and have one line per sample;
- the datastep is called once per batch, capped by `num_train_batches`;
- validation runs only when `epochIDX % train_per_val == 0`;
- only rank 0 writes files;
- the returned `global_step` grows by the number of training batches, and the
  EMA updates only after `ema_update_after_step` (if supported);
- for `DATASTEP_FN` changes, the new tag dispatches to the new pair, and an
  unknown tag still raises `ValueError("Unsupported dataset ...")`.

## Running

```bash
<python> -m pytest -Werror tests/utils/training -q   # new and neighbouring tests
<ruff> check src/yoke/utils/training tests/utils/training
<ruff> format --check --diff src/yoke/utils/training tests/utils/training
```
