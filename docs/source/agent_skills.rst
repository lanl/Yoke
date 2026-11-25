Using Yoke with an AI agent
===========================

You don't need to know Yoke's internals to train a model on your own data.
Yoke ships a set of **agent skills**: guided procedures that an AI coding
agent (such as Claude Code) follows to take you from "I have data" to a
training run on your cluster, and to help debug it afterwards.

What you need
-------------

- Yoke installed in a Python environment, and the path to that Python.
- One or two sample files of your data that the agent can look at.
- The cluster you will train on, and a SLURM account with GPU time.
- An AI coding agent running in a checkout of the Yoke repository. Claude Code
  finds the skills automatically. Other agents can be pointed at ``AGENTS.md``
  and ``skills/README.md``.

How to start
------------

Open the agent in the Yoke repository and describe what you want, for
example::

    I want to train a model on my simulation data in /path/to/data.
    I want to predict the fields at the next time step.

In Claude Code you can also start the front-door skill directly with
``/yoke-train-my-data``. If you only need one step, ask for it, or use that
skill by name:

========================== =====================================================
Skill                      What it does
========================== =====================================================
``yoke-train-my-data``     Start here: frames the problem and runs the others
``yoke-new-dataset``       Writes a dataset class for your files (any format)
``yoke-model``             Chooses and sizes a model, or builds one from Yoke parts
``yoke-new-loss``          Chooses or writes the training loss
``yoke-new-training-step`` Adds a training step when no existing one fits
``yoke-new-harness``       Builds the training harness (script, templates, CSV)
``yoke-slurm-template``    Writes the SLURM job script for your cluster
``yoke-launch-study``      Checks everything and gives you the launch command
``yoke-run-doctor``        Looks at a run you submitted and diagnoses problems
========================== =====================================================

What to expect
--------------

- **Questions, a few at a time.** The agent asks about your data, your goal,
  and your compute, and suggests defaults. It explains Yoke concepts (dataset
  class, harness, study CSV) the first time they come up.
- **A record of decisions.** Answers and designs are written to
  ``dev_plans/<project>_training.md`` and related notes files, so you can
  review them and pick the work up in a later session.
- **Approval before code.** The agent shows each design and waits for your
  go-ahead before writing code or templates.
- **A check before you spend GPU time.** Before launch, the agent runs a
  one-batch *preflight* on the CPU (``--preflight``; see
  :doc:`harness_trainer`), which catches most mismatches between your data,
  the model, and the loss.
- **You decide when to submit.** The agent gives you the exact
  ``yoke-start-study`` command and asks whether you want it to submit the
  study for you. Either way, the jobs go into the SLURM queue and resubmit
  themselves, so they don't depend on the agent's session: you can close it
  straight away. (Studies run with ``--submissionType shell`` run in the
  shell that starts them, so the agent only gives you that command.)
- **Come back any time.** Later, in a new session, ask for example "check on
  my run in ``applications/harnesses/<name>``". The agent reads the job status,
  logs, and loss records and tells you what is happening.

What the agent will not do
--------------------------

It submits a study only when you ask it to, and it never cancels or changes
jobs, installs packages, commits to git, or modifies your data or run outputs.
It inspects your data and the cluster with read-only commands.

Where things end up
-------------------

- New dataset, model, and loss code goes in ``src/yoke/`` (with tests).
- The training harness goes in ``applications/harnesses/<name>/``. See
  :doc:`harnesses`.
- Run outputs go in ``applications/harnesses/<name>/runs/study_NNN/``. See
  :doc:`start_study`.

Limits
------

The agent writes real code, so review it as you would a colleague's. Start
with the small smoke-test study the agent prepares before launching the full
sweep. The skills are new: if one gives poor guidance, tell the Yoke
maintainers, or improve the files under ``skills/``.
