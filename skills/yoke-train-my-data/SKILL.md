---
name: yoke-train-my-data
description: Start here to train a neural network on your own data with Yoke, even if you do not know Yoke. Frames the problem in plain terms, maps it onto Yoke (dataset, model, loss, training step, harness, SLURM, launch), runs the other yoke-* skills in order, and keeps one progress file so the work can resume in any later session. Use when a user has data and wants a Yoke training run, or does not know where to start.
---

# Train a Model on Your Data with Yoke

You guide a user who has data and a goal, but probably doesn't know Yoke's
internals, to a training run on their cluster. You frame the
problem, translate it into Yoke's pieces, and run the specialized skills in
order. One progress file holds the state, so any later session can pick it up.

Read first:

- `skills/shared/interview_conventions.md` (follow it throughout, including
  the progress-file and submission rules).
- `skills/README.md` (the skills and how they fit together).
- `skills/shared/cards.md`.

## Procedure

### 0. Resume?

Look for an existing progress file (`dev_plans/*_training.md`) that matches
what the user describes. If there is one, summarize where things stand: stages
done, the cards, the next stage. Ask whether to continue from the first
unfinished stage. If a run was already launched (there is a Run card) and the
user is asking about it, switch to `yoke-run-doctor`.

### 1. Frame the problem (plain language, round 1)

Ask, without Yoke terms:

- What is the data? Where does it come from, and where is a sample of it?
- What do you want to predict, and from what? (For example "the field at the
  next time step from the current one", "design parameters from an image",
  "an image from design parameters".)
- How will you judge success: a number, a visual check, use in a later
  workflow?
- How much data is there (number of simulations or samples, size on disk)?
- Where will you train (which cluster)? Does it have GPUs, or is it
  CPU-only? How much compute time do you have (GPU hours or node hours), and
  is there a deadline?
- Have you, or a colleague, trained something like this before?

Ask which Python environment has Yoke installed.

### 2. Translate into Yoke terms

Explain, briefly and in plain words, and confirm with the user:

- **Task type**: surrogate or time stepping (fields to fields), inverse
  (image to numbers), forward (numbers to image), or reward/policy.
- **Candidate model family** (from `skills/yoke-model/reference/model_catalog.md`).
- **Candidate batch contract**: what one training example will look like (see
  `skills/shared/batch_contracts.md`). Explain it as "each example will be
  (input, target, ...)".
- **What already exists** in Yoke for each piece, and what will need to be
  built.
- **Device mode**: GPU clusters use multi-GPU DDP (the default). A CPU-only
  cluster, or a model small enough that GPUs don't help, uses
  `HarnessTrainer(..., device_type="cpu")`: one process per job on one CPU
  node. Record the choice in the progress file; the harness and SLURM
  stages need it.

Introduce each Yoke concept in one or two sentences the first time it comes
up (dataset class, harness, study CSV, continuation, preflight).

### 3. Create the progress file

Create `dev_plans/<project>_training.md`:

```markdown
# <Project>: training <model family> on <data>

Started: YYYY-MM-DD. Goal: <one line>. Machine: <cluster>.

## Stages

| Stage | Skill | Status | Notes file |
| --- | --- | --- | --- |
| Dataset | yoke-new-dataset | todo / existing: <class> / done | |
| Model | yoke-model | | |
| Loss | yoke-new-loss | | |
| Training step | yoke-new-training-step | only if needed | |
| Harness | yoke-new-harness | | |
| SLURM template | yoke-slurm-template | only if needed | |
| Launch | yoke-launch-study | | |

## Problem framing

<answers from step 1 and the translation from step 2>

## Cards

<each skill's card, pasted when it finishes>

## Decisions and open questions
```

Mark stages that existing Yoke parts already cover (an existing dataset class,
model, or loss) as `existing: <name>`.

### 4. Run the stages

In order, skipping stages that are covered:

1. `yoke-new-dataset`: unless an existing class loads the data.
2. `yoke-model`: choose a model, or compose one.
3. `yoke-new-loss`: choose or write the loss. This can be quick if the
   default fits; still record the Loss card.
4. `yoke-new-training-step`: only if the batch contract, model call, or
   loss shape fits no existing training step.
5. `yoke-new-harness`: build the harness directory.
6. `yoke-slurm-template`: only if no current template exists for the
   machine.
7. `yoke-launch-study`: checks, preflight, the exact command, and, if the
   user asks, submission.

For each stage: tell the user what it does and why, in one or two sentences;
follow that skill's procedure (including its approval gates); then update the
progress file (status, notes-file link, card); then give the user a short
plain-language summary of where things stand.

Several stages may happen in one session or across many. At the end of each
session, make sure the progress file says what is next.

### 5. After launch

`yoke-launch-study` gives the command and asks whether to submit it. The user
submits, or the agent does if the user explicitly asks (SLURM only). Then
tell them how to come back later: open the agent in the repo and say, for example,
"check on my `<project>` run". That starts `yoke-run-doctor` from the Run card
in the progress file.

### 6. Finish

Summarize the whole path: what was built (files), what was checked, what
wasn't, the command, and the progress file's location.
