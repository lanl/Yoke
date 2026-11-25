# Interview Conventions (all Yoke skills)

Every Yoke skill is a guided conversation. The agent asks; the user decides.
These rules apply to every skill in `skills/` (listed in `skills/README.md`).

Most users of these skills do **not** know Yoke's internals. When a Yoke
concept first comes up (batch contract, harness, study CSV, continuation,
preflight), explain it in one or two plain sentences before asking about it.

## Remote (MCP) mode

Some skills (`yoke-finetune-mcp`) are used by an agent that reaches Yoke only
through a Nomad MCP server's tools, with no Yoke repo, files, or Python of its
own. In that mode:

- Skip the repo steps below (reading `AGENTS.md`, asking for a Python
  environment, design notes and progress files). The server keeps each run's
  state and **run card**; a later session resumes from the run's id.
- You never write code. Work is sent as declarative input the server checks
  (a fine-tune spec), after the user approves it.
- Training starts only through the server's start tool, after the user's
  explicit yes to that run, as for SLURM submission below.
- Everything else here still applies: rounds of questions, plain-language
  explanations, recommended defaults, and the finishing summary.

## Before starting

1. Read `AGENTS.md` at the repo root, and any `CLAUDE.md` / `CONTRIBUTING.md`
   in the folders being changed. Project rules win over skill defaults.
2. Ask which Python interpreter or environment has Yoke installed before
   running any Python, tests, `ruff`, or the `yoke-*` CLIs. Do not install,
   upgrade, or remove packages.

## Asking questions

- Ask in **rounds** of two to five related questions, not one long form.
  Wait for the answers before the next round.
- With each question, offer a **recommended default** when the repo suggests
  one, and say where it comes from (for example "as in
  `applications/harnesses/ch_ldrViT_lsc_cyl/`").
- When an answer is ambiguous and changes the design, ask again. For minor
  choices, pick the default and note it in the design notes.
- Do not guess facts the user or the data can provide. Check the data or ask.
- Summarize what you understood at the end of each round, in one or two
  sentences, so the user can correct it early.

## Progress file (multi-skill work)

When the work spans several skills (normally started by `yoke-train-my-data`),
one **progress file**, `dev_plans/<project>_training.md`, ties it together.
It holds a checklist of stages, each with its status and a link to that
stage's design notes file, and all the cards (`skills/shared/cards.md`).

- At the start of any skill, look for a progress file that mentions this
  work. If there is one, read it and update it when the skill finishes:
  mark the stage done, paste the card, and add a link to the notes file.
- A later session resumes from the progress file, not from memory.

## Design notes file

- At the start of the interview, create
  `dev_plans/<short_name>_<skill_topic>.md` (for example
  `dev_plans/<cx250101>_dataset.md` for a new `cx250101` dataset). Use the
  repo's `dev_plans/` folder.
- Record each round's answers there as they come in, under a
  `## Decisions` heading. This keeps the decisions safe if the conversation
  is compacted, and lets the user review them.
- When the interview is done, write a `## Design` section (what will be
  written, where, and how it will be tested) and a `## Verification`
  section. Then **stop and ask for approval** before writing any code or
  templates. Writing the design notes file itself needs no approval.
- If the design changes later, update the file and tell the user what
  changed before going on.

## Safety rules

- Inspect data files and machines only with **cheap, read-only** operations:
  headers, keys, shapes, small slices, `--help`, `sinfo`. Never load a
  large file fully into memory to look at it.
- Never delete, move, or overwrite data files, checkpoints, or run outputs
  (`.npz`, `.pth`, `.h5`, anything under `runs/`).
- Never add a dependency to `pyproject.toml`, or use a package that is not
  already installed, without approval.
- **Never submit jobs** (`sbatch`, `salloc`, `srun`), start training, or
  launch GPU work, and never cancel or modify a job (`scancel`, `scontrol
  update`). Skills prepare the exact command for the user. **One exception:**
  `yoke-launch-study` may run its `yoke-start-study` command for a **SLURM**
  study, once, when the user explicitly asks at launch time. That is safe
  because the jobs then live in the SLURM queue, independent of the agent's
  shell. The same holds for `yoke-finetune-mcp`, which may start a
  fine-tune on a Nomad server through its start tool after an explicit yes:
  the server owns the process. No other skill submits; `yoke-run-doctor`
  only proposes restart commands. The one exception to "no compute" is the
  CPU **preflight** (`--preflight`), a single forward pass that the user has
  approved for that run.
- Never write credentials, tokens, or secrets into any file.
- Never run `git commit` or `git push`. Leave changes in the working tree.

## Code style for anything written

- Google-style docstrings on every public module, class, and function.
- Full type annotations on every function (arguments and return).
- Lines at most 89 characters, 4-space indentation, double quotes.
- Comment intent and non-obvious choices, matching the surrounding code.
- Put reusable logic in `src/yoke`; harness scripts only configure and
  orchestrate.
- Tests go under `tests/`, mirroring the `src/yoke` path, and must pass with
  `pytest -Werror`.

## Finishing

1. Run the checks the skill lists (tests, `ruff check`,
   `ruff format --check`) and report the real results, including failures.
2. Write the skill's **card** (see `skills/shared/cards.md`) into the design
   notes file and show it in chat.
3. End with a short summary: files changed, how they were checked, what was
   *not* checked, and open follow-ups.
