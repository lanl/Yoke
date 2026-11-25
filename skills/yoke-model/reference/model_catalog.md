# Model Catalog

Trainable models in `src/yoke/models/`, with the constructor arguments a
harness sets in `make_model_args`, the forward signature, and the epoch
function existing harnesses pair them with. Signatures were copied from the
source. Line numbers drift, so before relying on an argument, open the class
and check it.

Batch formats for each epoch function: `skills/shared/batch_contracts.md`.

## Summary

| Model | Input → output | Epoch function used by harnesses | Example harness |
| --- | --- | --- | --- |
| `yoke.models.vit.swin.bomberman.LodeRunner` | fields `(B,C,H,W)` + lead time → fields | `train_DDP_loderunner_epoch` | `applications/harnesses/se_DDP_loderunner/` |
| `yoke.models.vit.swin.bomberman.LodeRunnerViT` | 1 or 2 frames + lead time(s) → fields | `train_DDP_loderunner_epoch`, `train_DDP_loderunner_multidataset_epoch` | `applications/harnesses/ch_ldrViT/`, `applications/harnesses/ch_ldrViT_lsc_cyl/` |
| `yoke.models.vit.swin.diffusion_bomberman.DiffusionLodeRunner` | conditioning + noised target → noise | `train_DDP_diffusion_loderunner_epoch` (wrapped) | `applications/harnesses/ch_DDP_diffLDR/` |
| `yoke.models.CNNmodules.Image2VectorCNN` | image → vector | `train_DDP_array_epoch` | `applications/harnesses/ch_lsc_inverse/` |
| `yoke.models.CNNmodules.Image2ScalarCNN` | image → scalar | `train_scalar_epoch` (not DDP; no harness uses it) | none |
| `yoke.models.surrogateCNNmodules.tCNNsurrogate` | vector → image | `train_DDP_array_epoch` | `applications/harnesses/lsc_action/` |
| `yoke.models.surrogateCNNmodules.jekelCNNsurrogate` | vector → image | `train_DDP_array_epoch` would fit; no harness uses it | none |
| `yoke.models.hybridCNNmodules.hybrid2vectorCNN` | vector + 2 images → vector | `train_lsc_reward_epoch` | `applications/harnesses/ch_lsc_reward/` |
| `yoke.models.policyCNNmodules.gaussian_policyCNN` | vector + 2 images → Gaussian | `train_lsc_policy_epoch` | `applications/harnesses/ch_lsc_policy/` |
| `yoke.models.mnist_model.mnist_CNN` | `(N,1,28,28)` → 10 log-probs | bespoke loop (demo only) | `applications/harnesses/mnist_surrogate/` |

Not for `HarnessTrainer`: the Lightning wrappers
(`yoke.models.vit.swin.bomberman.Lightning_LodeRunner`,
`yoke.models.vit.swin.diffusion_bomberman.Lightning_DiffusionLodeRunner`),
and sub-modules such as backbones, embeddings, and patch layers.

## LodeRunner (Swin U-Net surrogate)

`yoke.models.vit.swin.bomberman.LodeRunner`

```python
LodeRunner(
    default_vars: list[str],
    image_size: Iterable[int, int] = (1120, 800),
    patch_size: Iterable[int, int] = (10, 10),
    embed_dim: int = 128,
    emb_factor: int = 2,
    num_heads: int = 8,
    block_structure: Iterable[int, int, int, int] = (1, 1, 3, 1),
    window_sizes = [(8, 8), (8, 8), (4, 4), (2, 2)],
    patch_merge_scales = [(2, 2), (2, 2), (2, 2)],
    noise_scale: float = 0.0,
    verbose: bool = False,
)
forward(x, in_vars, out_vars, lead_times) -> Tensor
```

- `default_vars`: the full list of channel **names** the model knows.
  `in_vars`/`out_vars` are integer index tensors into it (the datasteps
  build them from `channel_map`).
- `x` is `(B, len(in_vars), H, W)`, `lead_times` is `(B,)`, the output is
  `(B, len(out_vars), H, W)`.
- The constructor asserts
  `yoke.helpers.training_design.validate_patch_and_window`. `H, W` must be
  divisible by `patch_size`, and the patch grid must be divisible by
  `window_sizes[0]` and by every later merge scale and window size.
  `yoke.helpers.training_design.find_valid_pad` searches for a padding that
  works. Check this early for a new image size.
- `noise_scale > 0` adds input noise on every forward, including eval.

## LodeRunnerViT (plain ViT with RoPE)

`yoke.models.vit.swin.bomberman.LodeRunnerViT`

```python
LodeRunnerViT(
    default_vars: list[str],
    image_size: Iterable[int, int] = (1120, 800),
    patch_size: Iterable[int, int] = (10, 10),
    embed_dim: int = 128,
    num_heads: int = 8,             # variable-aggregation heads
    num_attention_heads: int = 8,   # ViT block heads
    attention_head_dim: int = 16,
    num_layers: int = 6,
    mlp_ratio: float = 4.0,
    rope_theta: float = 10000.0,
    rope_scale: Iterable[float, float] = (1.0, 1.0),
    concat_mlp: bool = True,
    eps: float = 1e-6,
    bias: bool = False,
    num_input_frames: int = 1,
    noise_scale: float = 0.0,
    verbose: bool = False,
)
forward(x, in_vars, out_vars, lead_times) -> Tensor
```

- Asserts `num_input_frames in (1, 2)` and
  `embed_dim == num_attention_heads * attention_head_dim`.
- `num_input_frames=1`: `x` is `(B, C, H, W)`, `lead_times` is `(B,)`.
- `num_input_frames=2`: `x` is `(B, 2, C, H, W)` ordered `[x_{t-1}, x_t]`,
  `lead_times` is `(B, 2) = (dt_in, dt_out)`. Use with the 2-frame
  datasteps.
- `H, W` should be divisible by `patch_size`. There is no Swin window check.
- Reference `make_model_args`:
  `applications/harnesses/ch_ldrViT_lsc_cyl/train_ldrViT_2frame_lsc_cyl.py`
  (`image_size=(1120, 400)`, `patch_size=(10, 5)`, `rope_scale=(80, 224)`).

## DiffusionLodeRunner

`yoke.models.vit.swin.diffusion_bomberman.DiffusionLodeRunner`

Constructor as `LodeRunner`, without `noise_scale`.

```python
forward(x, y_tau, in_vars, out_vars, lead_times, diffusion_time) -> Tensor
```

- `x` is `(B, C_in, H, W)` (conditioning), `y_tau` is `(B, C_out, H, W)`
  (noised target), `lead_times` and `diffusion_time` are `(B,)`. It returns
  predicted noise `(B, C_out, H, W)`.
- Same `validate_patch_and_window` assert as `LodeRunner`.
- The harness wraps the epoch function to supply `in_vars`/`out_vars` and the
  noise schedule: see `run_epoch` in
  `applications/harnesses/ch_DDP_diffLDR/train_DDP_diffLDR.py`.

## Image2VectorCNN / Image2ScalarCNN

`yoke.models.CNNmodules.Image2VectorCNN`

```python
Image2VectorCNN(
    img_size: tuple[int, int, int] = (1, 1120, 400),   # (C, H, W)
    output_dim: int = 29,
    size_threshold: tuple[int, int] = (12, 12),
    kernel: int = 3,
    features: int = 16,
    interp_depth: int = 12,
    conv_onlyweights: bool = True,
    batchnorm_onlybias: bool = True,
    act_layer: nn.Module = nn.GELU,
    norm_layer: nn.Module = nn.LayerNorm,
    hidden_features: int = 32,
)
forward(x) -> Tensor   # (B, C, H, W) -> (B, output_dim)
```

`yoke.models.CNNmodules.Image2ScalarCNN`: the same idea with
`img_size=(1, 1700, 500)`, `size_threshold=(8, 8)`, `kernel=5`,
`features=12`, `hidden_features=20`. It has no `output_dim` or `norm_layer`,
and outputs `(B, 1)`. The only matching epoch function,
`train_scalar_epoch`, is single-process. A DDP harness needs
`yoke-new-training-step`, or `train_DDP_array_epoch` with a `(B, 1)` target.

## tCNNsurrogate / jekelCNNsurrogate (vector → image)

`yoke.models.surrogateCNNmodules.tCNNsurrogate`

```python
tCNNsurrogate(
    input_size: int = 29,
    linear_features: tuple[int, int] = (7, 5, 256),
    initial_tconv_kernel: tuple[int, int] = (5, 5),
    initial_tconv_stride: tuple[int, int] = (5, 5),
    initial_tconv_padding: tuple[int, int] = (0, 0),
    initial_tconv_outpadding: tuple[int, int] = (0, 0),
    initial_tconv_dilation: tuple[int, int] = (1, 1),
    kernel: tuple[int, int] = (3, 3),
    nfeature_list: list[int] = [256, 128, 64, 32, 16],
    output_image_size: tuple[int, int] = (1120, 800),
    output_image_channels: int = 1,
    act_layer: nn.Module = nn.GELU,
)
forward(x) -> Tensor   # (B, input_size) -> (B, output_image_channels, H, W)
```

- `linear_features[2]` must equal `nfeature_list[0]`.
- Nothing checks the output size. The size after the initial transposed
  convolution, times `2 ** len(nfeature_list)`, must reach
  `output_image_size`. Work this out in the design and add a shape test.

`yoke.models.surrogateCNNmodules.jekelCNNsurrogate`

```python
jekelCNNsurrogate(
    input_size: int = 29,
    linear_features: tuple[int, int] = (4, 4),
    kernel: tuple[int, int] = (3, 3),
    nfeature_list: list[int] = [512, 512, 512, 512, 256, 128, 64, 32],
    output_image_size: tuple[int, int] = (1120, 800),
    act_layer: nn.Module = nn.GELU,
)
forward(x) -> Tensor   # (B, input_size) -> (B, 1, H, W), interpolated to size
```

## hybrid2vectorCNN / gaussian_policyCNN (vector + two images)

`yoke.models.hybridCNNmodules.hybrid2vectorCNN`

```python
hybrid2vectorCNN(
    img_size: tuple[int, int, int] = (1, 1120, 400),
    input_vector_size: int = 28,
    output_dim: int = 1,
    features: int = 12,
    depth: int = 12,
    kernel: int = 3,
    img_embed_dim: int = 32,
    vector_embed_dim: int = 32,
    size_reduce_threshold: tuple[int, int] = (8, 8),
    vector_feature_list: tuple[int, ...] = (32, 32, 64, 64),
    output_feature_list: tuple[int, ...] = (64, 128, 128, 64),
    act_layer: nn.Module = nn.GELU,
    norm_layer: nn.Module = nn.LayerNorm,
)
forward(y, h1, h2) -> Tensor   # y (B, input_vector_size), h1/h2 (B, C, H, W)
```

`yoke.models.policyCNNmodules.gaussian_policyCNN` has the same inputs, plus
`output_dim=28` and `min_variance=1e-6`. Its forward returns a
`torch.distributions.MultivariateNormal`, not a tensor, and the policy
datastep uses its `.mean`.

## mnist_CNN

`yoke.models.mnist_model.mnist_CNN(conv1_size=32, conv2_size=64,
conv3_size=128, conv4_size=128)` is hard-wired to `(N, 1, 28, 28)` input.
It is a demo only and not used with `HarnessTrainer`.

## Choosing a model: questions to ask

1. What are the input and output: fields to fields (surrogate), image to
   vector (inverse), vector to image (forward/action), or vector + images to
   vector (reward/policy)?
2. For field-to-field models: fixed channels, or a per-sample channel map?
   One input frame or two? Does the image size fit the patch/window
   constraints?
3. Which architecture arguments are fixed in `make_model_args`, exposed as CLI
   flags, or swept in the study CSV (`<KEY>` columns)?
4. Fresh training only, or also starting from a pretrained checkpoint? For
   the same architecture, use `yoke.utils.builders.build_from_pretrained`
   (as in `applications/harnesses/ch_ldrViT_lsc_cyl/`). Changing the
   architecture needs a bespoke `model_builder` (see
   `applications/harnesses/se_DDP_loderunner_finetune_cylex/`).
