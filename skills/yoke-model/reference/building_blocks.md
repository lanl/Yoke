# Model Building Blocks

Reusable `nn.Module` pieces in `src/yoke/models/` for composing a new model
when no model in `model_catalog.md` fits. Signatures were read from the source,
and the import paths were checked (2026-11-11). Shapes marked *(inferred)* come
from reading the code, not from a docstring or a run.

Notation: `B` batch, `C`/`V` variables (channels), `(H, W)` image,
`(ph, pw)` patch size, `(Hp, Wp) = (H/ph, W/pw)` patch grid,
`N = Hp*Wp` tokens, `D` embedding dimension.

## How the existing models compose them

These are the best templates for a new model.

**`LodeRunner`** (`yoke.models.vit.swin.bomberman.LodeRunner`): fields to
fields, Swin U-Net.

| Step | Block | Shape |
| --- | --- | --- |
| in | | `x (B, C, H, W)`, `in_vars`/`out_vars` 1-D index tensors, `lead_times (B,)` |
| 1 | `ParallelVarPatchEmbed(x, in_vars)` | `(B, C, N, D)` |
| 2 | `VarEmbed(x, in_vars)` | `(B, C, N, D)` |
| 3 | `AggVars` | `(B, N, D)` |
| 4 | `PosEmbed` | `(B, N, D)` |
| 5 | `TimeEmbed(x, lead_times)` | `(B, N, D)` |
| 6 | `SwinUnetBackbone` | `(B, N, D)` |
| 7 | `nn.Linear(D, V*ph*pw)` | `(B, N, V*ph*pw)` |
| 8 | `Unpatchify` | `(B, V, H, W)` |
| 9 | `x[:, out_vars]` | `(B, len(out_vars), H, W)` |

**`LodeRunnerViT`**: the same, but with `PlainViTBackbone` (RoPE, so no
`PosEmbed`) instead of the U-Net. With `num_input_frames=2`, each frame is
embedded (steps 1–3 plus a time tag), the two `(B, N, D)` grids are
concatenated, and `nn.Linear(2D, D)` fuses them before the backbone.

**`DiffusionLodeRunner`**: two embedding streams (conditioning `x` and noised
target `y_tau`, steps 1–4 each) added together, then `TimeEmbed` and
`DiffusionTimeEmbed`, then the U-Net. The output is predicted noise.

**CNN family** (`Image2VectorCNN`, `hybrid2vectorCNN`, `gaussian_policyCNN`):
per image, `CNN_Interpretability_Module` → `CNN_Reduction_Module` → flatten →
`generalMLP`. Vector inputs get their own `generalMLP`. The embeddings are
concatenated and passed through a final `generalMLP` head.

## Token pipeline blocks (`src/yoke/models/vit/`)

### `yoke.models.vit.patch_embed.ParallelVarPatchEmbed`

```python
ParallelVarPatchEmbed(max_vars: int = 5, img_size: tuple[int, int] = (128, 128),
                      patch_size: tuple[int, int] = (16, 16), embed_dim: int = 64,
                      norm_layer: nn.Module | None = None)
forward(x, in_vars) -> Tensor      # (B, C, H, W), in_vars (C,) -> (B, C, N, D)
```

- One patch-embedding conv **per variable**. `in_vars` indexes into
  `max_vars`, and every sample in a batch uses the same variables.
- Asserts that `H % ph == 0` and `W % pw == 0`. Sets `.grid_size = (Hp, Wp)`,
  which the backbone and `Unpatchify` use.
- Pass `in_vars` as a 1-D tensor of length C (the docstring's `(1, C)` is
  out of date).

### `yoke.models.vit.patch_embed.SwinEmbedding`

`SwinEmbedding(num_vars=5, img_size=(128, 128), patch_size=(16, 16),
embed_dim=64, norm_layer=None)`: a single conv patch embedding,
`(B, num_vars, H, W) -> (B, N, D)`. Use it for fixed-channel inputs (it is
used by the `Swin` classifiers).

### `yoke.models.vit.embedding_encoders.VarEmbed`

`VarEmbed(default_vars: list[str], embed_dim: int)`;
`forward(x, in_vars)`: `(B, V, N, D) -> (B, V, N, D)`. Adds a learnable,
sin/cos-initialized tag per variable name.

### `yoke.models.vit.aggregate_variables.AggVars`

`AggVars(embed_dim: int, num_heads: int)`: `(B, V, N, D) -> (B, N, D)`.
Attention over the variable axis with a learnable query. Requires
`embed_dim % num_heads == 0`.

### `yoke.models.vit.embedding_encoders.PosEmbed`

`PosEmbed(embed_dim, patch_size, image_size, num_patches)`:
`(B, N, D) -> (B, N, D)`. Adds a learnable 2-D sin/cos position embedding
(`num_patches == N`). Not needed with RoPE backbones.

### `yoke.models.vit.embedding_encoders.TimeEmbed` / `DiffusionTimeEmbed`

`TimeEmbed(embed_dim)`; `forward(x, lead_times)`: `(B, N, D)` and `(B,)` →
`(B, N, D)`, using a `Linear(1, D)` of the time, broadcast over tokens.
`DiffusionTimeEmbed` is the same for the diffusion time.

### `yoke.models.vit.patch_manipulation.Unpatchify`

`Unpatchify(total_num_vars=5, patch_grid_size=(64, 64), patch_size=(8, 8))`:
`(B, N, V*ph*pw) -> (B, V, H, W)`. Put `nn.Linear(D, V*ph*pw)` in front. It
asserts both sizes.

### `yoke.models.vit.patch_manipulation.PatchMerge` / `PatchExpand`

`PatchMerge(emb_size=64, emb_factor=2, patch_grid_size=(64, 64), s1=2, s2=2)`:
`(B, Hp*Wp, D) -> (B, (Hp/s1)*(Wp/s2), emb_factor*D)`. `PatchExpand` (same
arguments) is the inverse direction,
`(B, Hp*Wp, D) -> (B, (Hp*s1)*(Wp*s2), emb_factor*D/(s1*s2))`. Both expose
`.out_patch_grid_size` and `.out_emb_size`.

## Backbones (token in, token out)

### `yoke.models.vit.plain_vit.PlainViTBackbone`

```python
PlainViTBackbone(patch_grid_size: tuple[int, int] = (112, 80),
                 num_attention_heads: int = 8, attention_head_dim: int = 16,
                 num_layers: int = 6, mlp_ratio: float = 4.0,
                 rope_theta: float = 10000.0,
                 rope_scale: tuple[float, float] = (1.0, 1.0),
                 concat_mlp: bool = True, eps: float = 1e-6, bias: bool = False,
                 verbose: bool = False)
forward(x) -> Tensor      # (B, N, D) -> (B, N, D)
```

- Global attention with 2-D RoPE, isotropic (no down-sampling).
- Requires `D == num_attention_heads * attention_head_dim`,
  `N == Hp * Wp`, and an even `attention_head_dim`.
- Cost grows with `N²`, so keep the token count moderate (larger patches).
- Built from `yoke.models.vit.plain_vit.BasicTransformerBlock`
  (`(num_attention_heads, attention_head_dim, mlp_ratio, bias, eps, concat)`)
  and `yoke.models.vit.plain_vit.RopeAttention`.

### `yoke.models.vit.swin.unet.SwinUnetBackbone`

```python
SwinUnetBackbone(emb_size: int = 96, emb_factor: int = 2,
                 patch_grid_size: (int, int) = (112, 80),
                 block_structure: (int, int, int, int) = (1, 1, 3, 1),
                 num_heads: int = 10,
                 window_sizes = [(8, 8), (8, 8), (4, 4), (2, 2)],
                 patch_merge_scales = [(2, 2), (2, 2), (2, 2)],
                 num_output_classes: int = 5, verbose: bool = False)
forward(x) -> Tensor      # (B, N, D) -> (B, N, D)
```

- A windowed-attention U-Net: three down stages with skips, a bottleneck, and
  three up stages. Cost is linear in `N`, which suits large images.
- Every stage's grid must be divisible by its window size, and each merge
  must divide evenly. Check a configuration with
  `yoke.helpers.training_design.validate_patch_and_window` (all `True`), and
  search for padding with `yoke.helpers.training_design.find_valid_pad`.
- `emb_size * emb_factor**i` must be divisible by `num_heads` at every stage
  *(inferred)*.
- `num_output_classes` appears unused *(inferred)*.
- Built from `yoke.models.vit.swin.encoder.SwinEncoder2`,
  `SwinConnectEncoder`, `SwinConnectDecoder`, and the window attention classes
  in `yoke.models.vit.swin.windowed_msa` (`WindowCosMSA`,
  `ShiftedWindowCosMSA`; shifted windows need even window sizes).

### Classifiers: `yoke.models.vit.swin.transformer.Swin` / `SwinV2`

`(B, input_channels, H, W) -> (B, num_output_classes)`. Complete image-to-vector
models (`SwinEmbedding`, Swin stages with `PatchMerge`, pooling, `Linear`).
Usable as an image → vector model or as a starting point.

## CNN blocks (`src/yoke/models/`)

### `yoke.models.CNNmodules.CNN_Interpretability_Module`

```python
CNN_Interpretability_Module(img_size: tuple[int, int, int] = (1, 1700, 500),
                            kernel: int = 5, features: int = 12, depth: int = 12,
                            conv_onlyweights: bool = True,
                            batchnorm_onlybias: bool = True,
                            act_layer: nn.Module = nn.GELU)
```

`(B, C, H, W) -> (B, features, H, W)`: `depth` stride-1 "same" convolutions
(`img_size` is `(C, H, W)`; only `C` is used).

### `yoke.models.CNNmodules.CNN_Reduction_Module`

```python
CNN_Reduction_Module(img_size=(1, 1700, 500), size_threshold=(8, 8), kernel=5,
                     stride=2, features=12, conv_onlyweights=True,
                     batchnorm_onlybias=True, act_layer=nn.GELU)
```

`(B, C, H, W) -> (B, features, finalH, finalW)`. It adds strided convolutions
until both sizes are at most `size_threshold`. Read `.finalH`, `.finalW` after
construction to size the next `Linear` (`features * finalH * finalW`). To
chain after the interpretability module, pass `img_size=(features, H, W)`.

### `yoke.models.cnn_utils.generalMLP`

```python
generalMLP(input_dim: int = 64, output_dim: int = 16,
           hidden_feature_list: tuple[int, ...] = (16, 32, 32, 16),
           act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm)
```

`(..., input_dim) -> (..., output_dim)`. Hidden layers are Linear → norm →
activation. The last is a plain Linear. `act_layer`/`norm_layer` are classes,
and `hidden_feature_list` must be a tuple.

### Shape helpers

`yoke.models.cnn_utils.conv2d_shape(w, h, k, s_w, s_h, p_w, p_h)` and
`yoke.models.cnn_utils.convtranspose2d_shape(w, h, k_w, k_h, s_w, s_h, p_w,
p_h, op_w, op_h, d_w, d_h)` return `(new_w, new_h, total)`. Note the order:
**width first**. Use them in a shape walk-through for transposed-conv decoders
(the vector → image models in `src/yoke/models/surrogateCNNmodules.py` use raw
`nn` layers and do not check their sizes).

## Composition rules

1. **Match a batch contract.** The new model's `forward` must accept what the
   chosen datastep calls it with (see `skills/shared/batch_contracts.md`).
   For field-to-field models, that is `forward(x, in_vars, out_vars,
   lead_times)`, so it works with the loderunner datasteps unchanged.
2. **Plain constructor arguments.** `model_args` is saved in every
   checkpoint and the model is rebuilt from it, so use ints, floats, strings,
   bools, and lists or tuples of those. Pass classes such as `act_layer` only
   if the default is fine; don't put them in `make_model_args`.
3. **Walk the shapes.** In the design, write the tensor shape after every
   block for the real image size, and the token count `N`.
4. **Check constraints in `__init__`** with clear `ValueError`s (or the
   existing asserts), so a bad configuration fails at construction, not
   mid-run.
5. **Count parameters** for the real configuration, and estimate activation
   memory as roughly `B × N × D × layers × (a few)` floats. For ViTs, also
   check the `N²` attention cost.
6. **Reuse, don't copy.** Import blocks from `yoke.models...`, and do not
   paste their code into the new module.
