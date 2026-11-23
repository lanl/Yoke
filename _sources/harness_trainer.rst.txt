Authoring a harness with ``HarnessTrainer``
===========================================

Historically every training harness under ``applications/harnesses/`` shipped a
``train_*.py`` script that was, in the large, the same program copied and
lightly edited. :class:`yoke.harnesses.trainer.HarnessTrainer` removes that
copy-paste by owning the fixed *orchestration* (DDP setup, model/optimizer/
scheduler/dataloader construction, a timed epoch loop, ``.pth`` checkpointing,
and job resubmission) while a thin harness script injects only the parts that
genuinely vary.

The trainer is **always DDP** and **always writes** ``.pth`` checkpoints via
:func:`yoke.utils.checkpointing.save_model_and_optimizer`. Vanilla
``DataParallel`` and HDF5 checkpoint *writing* are deprecated (targeted for
removal in December 2026).

What you inject
---------------

- ``model_builder(args, device) -> (model, model_args, model_class, start_epoch, optimizer)``
  — constructs the model for **both** the fresh and continuation paths. On a
  fresh run it returns the constructed model (already on ``device``), the
  ``model_args`` dict, the model *class* (so the checkpoint records a matching
  ``model_class``), ``0`` for the starting epoch, and ``None`` for the optimizer
  (the trainer then builds one). On continuation it returns a *restored*
  optimizer so its state is preserved. For the common fresh-vs-continue case use
  the :func:`yoke.utils.builders.build_from_checkpoint` factory; write a bespoke
  builder when performing architectural surgery (loading a pretrained backbone,
  stripping/replacing layers, etc.).
- ``dataset_builder(args) -> (train_dataset, val_dataset)``.
- ``epoch_fn`` — one of the existing per-epoch functions in
  :mod:`yoke.utils.training.epoch`. Study-specific extra keyword arguments (e.g.
  ``channel_map`` or a ``dataset`` tag) are supplied via ``epoch_kwargs``.
- ``optimizer_builder(model, args) -> Optimizer`` — defaults to
  :func:`yoke.utils.builders.build_adamw`.
- ``scheduler_builder(optimizer, args, last_epoch) -> scheduler | None`` —
  optional.
- ``loss_builder() -> nn.Module`` — defaults to
  :func:`yoke.utils.builders.default_mse_loss`.

Dynamic, in-loop customization
-------------------------------

EMA, gradient clipping, and progressive (un)freezing are expressed through a
small :class:`yoke.harnesses.trainer.TrainerHooks` object of optional,
composable callables: ``on_after_ddp_wrap``, ``on_epoch_start``,
``on_before_optimizer_step``, ``on_after_step``, and ``on_before_save`` (which
may return a ``dict`` merged into the checkpoint's ``extra_state``). Because
hooks compose, a study needing EMA *and* grad-clip *and* scheduled unfreezing
combines them without any multiple-inheritance mess.

Warmup EMA via ``make_ema_hooks``
---------------------------------

The Diffusers-style warmup EMA recipe is packaged as a reusable hook factory,
:func:`yoke.utils.ema.make_ema_hooks`, so harnesses do not re-implement it. It
returns an ``(on_after_ddp_wrap, on_before_save)`` pair that builds the EMA
shadow from the DDP-wrapped module, restores the shadow plus the persisted
``global_step`` from an ``_ema.pth`` companion checkpoint on continuation, and
writes the ``_ema.pth`` companion and ``_ema_weights.pth`` production checkpoints
at save time (returning ``{"global_step": ...}`` into the main checkpoint's
``extra_state``).

The per-step EMA update and gradient clipping themselves live in the shared
``train_DDP_loderunner_epoch`` function; the harness just passes ``grad_clip``
and ``ema_update_after_step`` through ``epoch_kwargs``. When an EMA shadow is
registered (``trainer.ema_model``), the trainer threads it and the live
``global_step`` into each epoch call and captures the advanced ``global_step``
returned by the epoch function.

.. code-block:: python

    from yoke.harnesses.trainer import HarnessTrainer, TrainerHooks
    from yoke.utils.ema import make_ema_hooks

    on_after_ddp_wrap, on_before_save = make_ema_hooks(LodeRunnerViT)
    hooks = TrainerHooks(
        on_after_ddp_wrap=on_after_ddp_wrap,
        on_before_save=on_before_save,
    )
    HarnessTrainer(
        args,
        model_builder=build_from_checkpoint(LodeRunnerViT, make_model_args),
        dataset_builder=build_dataset,
        epoch_fn=train_DDP_loderunner_epoch,
        scheduler_builder=build_scheduler,
        epoch_kwargs={
            "channel_map": list(range(len(CHANNEL_LIST))),
            "grad_clip": grad_clip,
            "ema_update_after_step": args.ema_update_after_step,
        },
        hooks=hooks,
    ).run()

See ``applications/harnesses/ch_ldrViT/train_ldrViT_ddp.py`` (and its two-frame
sibling ``train_ldrViT_2frame.py``) for a complete migrated EMA + grad-clip
example.

Opt-in post-training evaluation
-------------------------------

Passing ``evaluate_after_training=True`` makes a **finished** study submit one
separate test-set evaluation job for its final checkpoint (nothing is submitted
for unfinished continuation cycles). The trainer routes this through
:meth:`yoke.harnesses.base.HarnessStudy.run_evaluation`, the same render-and-
submit path used by the manual ``yoke-evaluate-study`` CLI, so automatic and
manual evaluation behave identically.

``evaluate_checkpoint`` chooses which checkpoint that job targets:

- ``"main"`` (default) — the ordinary ``.pth`` checkpoint just written.
- ``"ema"`` — the class-aware EMA companion (``..._ema.pth``) written by
  ``make_ema_hooks``. It loads exactly like the main checkpoint, and stem-based
  artifact naming keeps its outputs distinct. Selecting ``"ema"`` when no
  companion exists raises ``FileNotFoundError``.

The harness must ship the optional evaluation files (``evaluation_input.tmpl``,
the submission template, and the evaluator, listed in ``cp_files.txt``); see
:doc:`evaluate_study` for the full workflow and the manual CLI.

.. code-block:: python

    HarnessTrainer(
        args,
        model_builder=build_from_checkpoint(LodeRunner, make_model_args),
        dataset_builder=build_dataset,
        epoch_fn=train_DDP_loderunner_epoch,
        epoch_kwargs={"channel_map": list(range(len(CHANNEL_LIST))), "dataset": "pli"},
        evaluate_after_training=True,   # opt in; default is False
        evaluate_checkpoint="main",     # or "ema" for EMA-enabled harnesses
    ).run()

A thin harness script
----------------------

.. code-block:: python

    if __name__ == "__main__":
        args = parser.parse_args()

        HarnessTrainer(
            args,
            model_builder=build_from_checkpoint(LodeRunner, make_model_args),
            dataset_builder=build_dataset,
            epoch_fn=train_DDP_loderunner_epoch,
            scheduler_builder=build_scheduler,
            epoch_kwargs={"channel_map": list(range(len(CHANNEL_LIST)))},
        ).run()

See ``applications/harnesses/se_DDP_loderunner/train_LodeRunner_ddp.py`` for a
complete migrated example.

Bespoke model builders: freezing and fine-tuning
------------------------------------------------

When a study needs parameter freezing, per-block optimizer groups, or
pretrained-weight initialization, put that logic in a **bespoke**
``model_builder``/``optimizer_builder`` rather than a hook:

- ``applications/harnesses/ch_lsc_policy/train_lsc_policy.py`` freezes all
  parameters and unfreezes eight named sub-blocks in its ``model_builder``, then
  builds AdamW with one parameter group per block (each with its own LR) in a
  custom ``optimizer_builder``.
- ``applications/harnesses/se_DDP_loderunner_finetune_cylex/train_LodeRunner_ddp_cylex.py``
  optionally loads pretrained weights (shape-safe, weights-only) and freezes the
  transformer backbone for a warmup phase, keeping only the variable-embedding
  and unpatch head trainable.

**Freeze before the DDP wrap.** Apply ``requires_grad`` changes inside the
``model_builder`` (which returns the model *before* the trainer wraps it in
``DistributedDataParallel``). Toggling ``requires_grad`` *after* the wrap
desynchronizes DDP's gradient reducer from the trainable-parameter set and, with
the default ``find_unused_parameters=False``, can hang or error. Because a
harness job trains only ``cycle_epochs`` and resubmits, a job-granular freeze
decision (based on the job's first/upcoming epoch) covers the epoch-scheduled
fine-tuning recipes without needing a mid-loop ``on_epoch_start`` toggle.

Demo harnesses
--------------

``moving_mnist`` and ``mnist_surrogate`` are intentionally kept as small,
single-process, bespoke reference scripts and do **not** use ``HarnessTrainer``.
They are minimal onboarding examples; migrating them would reintroduce a
single-process path that the "always DDP" design deliberately removes.
