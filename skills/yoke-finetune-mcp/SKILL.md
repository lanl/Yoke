---
name: yoke-finetune-mcp
description: Fine-tune a Yoke model that a Nomad MCP server offers, on data stored on that server, using only the server's yoke_finetune_* tools - no Yoke checkout or Yoke Python on the agent's side. Interviews the user, checks the data, drafts a declarative fine-tune spec, runs the server-side preflight, and starts training only after the user's explicit yes. Use when the agent is connected to a Nomad server with these tools and the user wants to adapt one of its models to their data, or asks about such a run later.
---

# Fine-tune a Served Yoke Model Through Nomad

You help a user adapt a Yoke model that a Nomad server offers to their own
data. You have **no Yoke repo and no Yoke Python**: everything goes through
the server's MCP tools. You never write code. You fill in a declarative
**fine-tune spec** with the user, and the server checks it and runs Yoke's
own training harness.

The tools:

| Tool | Use it to |
| --- | --- |
| `yoke_finetune_list_models` | see the models, the data each accepts, the overrides, the spec schema |
| `get_yoke_skill` | read this skill and the shared files below |
| `yoke_finetune_inspect_data` | check a data directory on the server (read-only) |
| `yoke_finetune_prepare` | send the spec; get a `run_id` (nothing runs yet) |
| `yoke_finetune_start` | start `mode="preflight"`, then `mode="train"` |
| `yoke_finetune_status` | follow a run: jobs, losses, checkpoints, logs, run card |

Read first, with `get_yoke_skill`:

- `shared/interview_conventions.md`: follow it throughout, in its **remote
  (MCP) mode**. You have no files, so the server's run card replaces the
  progress and design notes files.
- `shared/cards.md`: the Fine-tune card.

## Background to explain when it comes up

- **Fine-tuning**: start from a trained model's weights and keep training on
  new data. The model's architecture, loss, and training step stay as they
  were; only the data and a few training settings change.
- **Preflight**: one batch through the model on the server's CPU. It checks
  that the model, the data, and the loss fit together. Nothing is trained or
  saved.
- **Batches and epochs**: training sees `train_batches` examples per epoch
  (batch size 1 for these models) for `total_epochs` epochs; validation
  runs every `train_per_val` epochs.

## Procedure

### 0. Resume?

If the user mentions an earlier run, ask for its `run_id` (or look for one
in the conversation) and call `yoke_finetune_status`. Summarize the run card
and the job states, and continue from the first unfinished step below. A run
that has trained cannot be trained again; a new attempt is a new run.

### 1. Orient

Call `yoke_finetune_list_models`. If the tools are missing, tell the user this
server does not offer fine-tuning and stop. Otherwise, describe the models in
plain words and which kinds of data each accepts.

### 2. Frame the problem (round 1)

Ask, in plain language:

- What is the data, and where is it **on the server**? (Paths must be under
  the `data_roots` that `yoke_finetune_list_models` lists. If the data is
  elsewhere, the server's operator has to place it there.)
- Which model do they want to adapt, and to what end? (For example "the LSC
  model, so it predicts Cylex simulations".)
- What counts as success, and how much time can the run take?

### 3. Check the data

Call `yoke_finetune_inspect_data` on the data directory (and the design CSV,
if the data kind needs one). Report, per series: frame count, gaps, and the
design values. Gaps are tolerated; a series needs at least three frames.

If the data does not match any data kind the model accepts (file names,
layout, a missing design CSV), explain what format the model needs and
**stop**. Converting data, or writing a new dataset, is out of scope for this
skill.

### 4. Draft the spec (round 2)

Propose a spec and explain each choice:

- `base_model` and `data.kind`.
- `train_series` and `validation_series`: disjoint; keep at least one series
  for validation.
- `training` overrides, starting from the defaults the server lists. For a
  fine-tune, keep `anchor_lr` well below the original run's (the server's
  default is already lowered). Size `total_epochs` and `train_batches` to the
  user's time budget; on a CPU server, one step of a large model can take
  minutes, so start small.

Show the full spec as YAML and **ask for approval**. Do not call
`yoke_finetune_prepare` before the user approves.

### 5. Prepare and preflight

1. Call `yoke_finetune_prepare(spec)`. If it fails, the error lists every
   problem; explain them in plain words, fix the spec with the user, and try
   again.
2. Tell the user the `run_id`; it is how any later session finds the run.
3. Call `yoke_finetune_start(run_id, mode="preflight")`, then poll
   `yoke_finetune_status` every minute or so until the preflight is no longer
   `running`.
4. Report the preflight result: the report (model, parameter count, batch
   shapes, loss) on success, or the end of the error log on failure, with
   the likely cause in plain terms. A spec cannot be edited once prepared:
   to change it, prepare a new run.

### 6. Ask before training

Training uses the server's compute for as long as it runs. Ask the user
whether to start it, and restate in the question:

- the `run_id`, the base model, and the train and validation series;
- `total_epochs`, `train_batches`, `val_batches`, and `anchor_lr`;
- what the preflight showed about the time per batch, if anything, and that
  the run is a background process on the server that keeps going if this
  session ends.

Start only on an explicit yes **to this question, now**. Approving the spec
or the preflight earlier does not count. Then call
`yoke_finetune_start(run_id, mode="train", user_confirmed=true)`. Never set
`user_confirmed=true` otherwise. If the answer is no or unclear, stop here;
the run stays prepared and can be started later.

There is no tool to stop a run, and you never try to stop one some other way.
If the user wants it stopped, tell them the server's operator can end the
process (its `pid` is in the job status).

### 7. Follow the run

Poll `yoke_finetune_status` at a pace that suits the run (do not poll in a
tight loop). Report the training and validation loss rows as they appear,
then the checkpoints when the run ends. On failure, read the error log tail
and explain it.

Write the **Fine-tune card** (`shared/cards.md`) in chat from the run card
and the status. Tell the user they can come back in any later session with
the `run_id`.

### 8. Summarize

What was run (spec and `run_id`), what the preflight and training showed,
which checkpoints the server wrote, and what was **not** checked: whether the
fine-tuned model is better (a few batches say little about that), and
anything the status could not show. Serving the fine-tuned model for
predictions is a separate step for the server's operator.
