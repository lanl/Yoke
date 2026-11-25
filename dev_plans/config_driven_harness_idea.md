# Idea: Config-Driven Harnesses

Status: **idea only**, not a plan. Written 2026-11-10. To pick up after agent
skills phase 2 (`dev_plans/agent_skills_phase2.md`) is implemented and tested
on HPC.

## The problem

A new harness today means writing a train script. The script is short,
because `HarnessTrainer` does the heavy lifting, but it is still custom code
that has to wire many parts together correctly:

- the argparse builders and new flags;
- `make_model_args`;
- the dataset, optimizer, scheduler, and loss builders;
- the epoch function and its `epoch_kwargs`;
- the hooks (EMA);
- and a `training_input.tmpl` whose flags must match that parser.

Most harnesses differ only in *which* dataset, model, loss, and epoch
function they use, and with *what* arguments. Agent skills phase 2 teaches
agents to write this wiring. A lot of what the skills explain (batch
contracts, which epoch function goes with which tuple, how to pass loss
arguments) exists only because every harness repeats it by hand.

## The idea

One generic, installed train entry point (for example `yoke-train`), driven
by a config file that **names** components and gives their arguments:

```yaml
# training_config.tmpl  (rendered per study, like training_input.tmpl)
dataset:
  name: pagosa_2frame.LSC_2frame_canonical_DataSet
  train: {npz_dir: /lustre/.../lsc240420/, file_prefix_list: <TRAIN_LIST>,
          max_timeIDX_offset: <MAX_TIME_OFFSET>, max_file_checks: 10}
  validation: {...}
model:
  name: LodeRunnerViT
  args: {embed_dim: <VIT_EMBED_DIM>, num_layers: <VIT_NUM_LAYERS>, ...}
loss:
  name: LpLoss
  args: {d: 2, p: 2, method: abs, eps: 1.0e-4, reduction: none}
training:
  epoch_fn: loderunner.multidataset
  optimizer: {name: adamw, lr: <ANCHOR_LR>, weight_decay: 0.01}
  scheduler: {name: cosine_warmup, warmup_steps: <WARMUP_STEPS>, ...}
  ema: {max_decay: 0.9999, update_after_step: 1000}
  batch_size: 1
  epochs: {total: 100, cycle: 3}
  batches: {train: <NTRN_BATCH>, val: <NVAL_BATCH>, train_per_val: 4}
```

Small **registries** in `src/yoke` map names to classes or functions and
record what each component needs and provides:

- each dataset registers its **batch contract**;
- each epoch function registers the contract it consumes and the loss shape
  it expects;
- each model registers its forward call;
- each loss registers its output shape for each `reduction`.

Before any GPU work starts, the generic entry point builds everything from
the config and **checks compatibility from those declarations**: dataset
contract to epoch function, loss shape to epoch function, model forward
signature to contract. It also runs the phase 2 preflight.

## What it would change

- A new harness becomes a **config template + study CSV + submission
  template**. No Python, unless a genuinely new component is needed, and
  then it goes in `src/yoke` with its registry entry and tests.
- `skills/shared/batch_contracts.md` could be **generated** from the
  registries instead of hand-maintained (the skills CI test already checks it
  against source; registries make the source the single truth).
- Agent skills get simpler and more reliable: `yoke-new-harness` mostly
  fills in a config, and mismatches surface as clear config errors instead
  of runtime shape errors.
- Humans benefit equally: reading one config shows the whole experiment.

## What it would keep

- `HarnessTrainer`, `HarnessStudy`, `yoke-start-study`, the CSV and `<KEY>`
  rendering, continuation, and checkpoint formats are unchanged. The config
  is just another rendered template.
- Bespoke train scripts remain possible for unusual cases (architecture
  surgery, custom loops, the demos). Config-driven is an option, not a
  replacement.
- This is about registering **components**, not per-harness subclasses, so
  it is consistent with the existing "single generic `HarnessStudy`, no
  per-harness registry" decision (`AGENTS.md` §5).

## Open design questions (for the real plan)

- **Config format**: YAML (needs PyYAML, a new declared dependency), TOML
  (`tomllib` is in the standard library on Python ≥3.11, which Yoke
  requires), or keep argparse `@`-files with a `--config` section. TOML
  avoids a new dependency.
- **Registry mechanism**: decorators (`@register_dataset("...")`) vs. an
  explicit table module vs. dotted import paths in the config (no registry,
  but no declared metadata either).
- **How much to declare**: tuple names only, or shapes and dtypes too, and how
  strictly to check them.
- **Bespoke builders**: pretrained or finetune model builders, mixed datasets
  (`MixedDataSet`), EMA hooks. These need config-expressible forms or an
  escape hatch to a Python callable.
- **Migration**: which existing harnesses to convert first (likely
  `ch_ldrViT_lsc_cyl` and `se_DDP_loderunner`) to prove the design, and
  whether to convert the rest at all.
- **Checkpoint compatibility**: checkpoints store `model_class` and
  `model_args`; config-built models must save the same, so existing
  evaluation tools keep working.

## Rough sequence (when picked up)

1. Write a real plan resolving the questions above.
2. Registries plus compatibility checks, with tests, and no behaviour change.
3. The generic `yoke-train` entry point and config rendering through
   `HarnessStudy`.
4. Convert one or two harnesses side by side with the originals and compare.
5. Update the agent skills to prefer config-driven harnesses, and generate
   `batch_contracts.md` from the registries.
