# Yoke Agent Skills

Guided, interview-style procedures that an AI coding agent follows to add
training capability to Yoke. Each skill is a folder with a `SKILL.md` (YAML
frontmatter `name` + `description`, then the procedure) and a `reference/`
folder of Yoke-specific notes it loads as needed.

| Skill | Produces |
| --- | --- |
| `yoke-train-my-data` | **Start here.** Frames the problem, runs the others in order, keeps a progress file |
| `yoke-new-dataset` | A `Dataset` class in `src/yoke/datasets/` + tests |
| `yoke-model` | A chosen, sized Yoke model, or a new one composed from Yoke blocks + tests |
| `yoke-new-loss` | A chosen or new loss in `src/yoke/losses/` with the right output shape + tests |
| `yoke-new-training-step` | Train/eval datasteps + epoch function in `src/yoke/utils/training/` + tests |
| `yoke-new-harness` | An `applications/harnesses/<name>/` directory on `HarnessTrainer` |
| `yoke-slurm-template` | `training_slurm.tmpl` (+ `evaluation_slurm.tmpl`) for the current cluster; saves a machine profile |
| `yoke-launch-study` | Readiness checks, CPU preflight, the exact launch command, and submission if the user asks (SLURM) |
| `yoke-run-doctor` | Diagnosis of a submitted run, in any later session |
| `yoke-finetune-mcp` | A fine-tune of a model a Nomad server offers, through its MCP tools only |

Typical flow:

```
                 yoke-train-my-data   (progress file: dev_plans/<project>_training.md)
                          │
   yoke-new-dataset → yoke-model → yoke-new-loss → [yoke-new-training-step]
                          │
                  yoke-new-harness → [yoke-slurm-template]
                          │
                  yoke-launch-study → prints command; user submits, or asks the agent to
                          ⋮   hours or days later, any session
                  yoke-run-doctor
```

Skills hand results to each other through short **cards**
(`shared/cards.md`) written into a `dev_plans/` notes file.

## Using the skills through Nomad

`yoke-finetune-mcp` is for an agent with no Yoke checkout, connected to a
Nomad MCP server that offers Yoke models (the `nomad_yoke` package's
`yoke_finetune_*` tools). The server serves the skill text itself
(`get_yoke_skill`). The agent writes no code: it fills in a declarative
**fine-tune spec** (base model, data, training overrides), and the server
checks it against the model's recipe and runs the Yoke harness. This is the
first slice of the config-driven direction in
`dev_plans/config_driven_harness_idea.md`. Pretrained starts in a harness use
`yoke.utils.builders.build_from_pretrained` (`--pretrained_model`).

## Layout

```
skills/
  shared/                 # used by all skills
    interview_conventions.md
    batch_contracts.md    # dataset tuple -> datastep -> epoch function
    cards.md
  yoke-train-my-data/
  yoke-new-dataset/
  yoke-model/             # model_catalog.md, building_blocks.md
  yoke-new-loss/          # loss_catalog.md
  yoke-new-training-step/
  yoke-new-harness/
  yoke-slurm-template/
    machines/             # saved per-cluster profiles (created by the skill)
  yoke-launch-study/
  yoke-run-doctor/        # run_layout.md, failure_signatures.md
  yoke-finetune-mcp/      # remote: fine-tune through a Nomad server's tools
```

This folder is the single, agent-neutral copy. Agents find the skills
through symlinks:

- Claude Code: `.claude/skills/<skill>` → `../../skills/<skill>`.
- Other agents (Codex, opencode): not linked yet. Agents that read
  `AGENTS.md` are pointed here from its "Agent skills" section.

Skills refer to each other's files by **repo-root paths**
(`skills/shared/...`) so they work through symlinks.

## Ground rules baked into every skill

- Ask, don't guess; record decisions in `dev_plans/`; get approval before
  writing code.
- Read-only inspection of data and machines.
- Never cancel or modify jobs, train, add dependencies, or commit. Submission
  happens only through `yoke-launch-study`. It gives the command, and runs it
  only if the user explicitly asks, for SLURM studies. The jobs then live in
  the SLURM queue and resubmit themselves, independent of any agent session.
  The only other computation is the one-batch CPU `--preflight`.
- Later sessions resume from files (the progress file and cards), not from
  memory.

## Keeping the skills accurate

`tests/skills/test_skill_files.py` runs in CI and checks:

- frontmatter;
- symlinks;
- that every repo path a skill mentions exists;
- that every `yoke.` dotted name imports;
- that each row of `shared/batch_contracts.md` matches the tuple its
  datastep unpacks;
- that saved machine profiles are dated and keep the allocation warning.

When you rename or change code that the skills mention, update them in the
same change.
