The ``yoke-plot-loss`` CLI
==========================

``yoke-plot-loss`` is the installed command-line tool used to plot **training vs.
validation loss curves** for a single study. It replaces the old, ad-hoc
``applications/evaluation/TandVplot.py`` script: the plotting logic now lives in
the installable package (:mod:`yoke.cli.plot_loss` and
:mod:`yoke.plots.loss_curves`) and is invoked as a real command. The old script
is retained as a thin shim for backwards compatibility.

Purpose
-------

Rather than collapsing each evaluation window to a single average loss, this
tool is meant to give the researcher a sense of the **distribution** of training
and validation errors over the course of a run. It groups the per-batch losses
into fixed-size blocks (set by ``--Nsamps_per_trn_pt`` / ``--Nsamps_per_val_pt``)
and, for each block, plots a **per-block 95% confidence interval**: the median
loss as a solid line, bracketed by the 2.5th and 97.5th percentiles as dotted
lines. The spread between those bounds shows how variable the loss is within each
window, which is often more informative than the mean alone — for example, it can
reveal noisy or unstable training that a smoothed average would hide, and it lets
you compare the *width* of the training and validation distributions directly.


How it works
------------

Given a study index, ``yoke-plot-loss``:

1. Parses arguments (:func:`yoke.helpers.cli.add_plot_loss_args`).
2. Locates the training and validation record CSVs for the study under
   ``<basedir>/study_###/`` (:func:`yoke.plots.loss_curves.find_record_csvs`).
   These are the ``training_study###_epoch*.csv`` and
   ``validation_study###_epoch*.csv`` files written by a training harness.
3. Aggregates the per-batch losses into evaluation blocks and, for each block,
   computes the median together with the 2.5th and 97.5th percentiles that form
   the per-block 95% confidence interval
   (:func:`yoke.plots.loss_curves.compute_quantile_bands`).
4. Builds a figure with the training curve in blue and the validation curve in
   red, plotted against a running "Evaluation Index"
   (:func:`yoke.plots.loss_curves.plot_loss_curves`).
5. Either displays the figure interactively or saves it as
   ``study###_TandV_curve.png`` (:func:`yoke.plots.loss_curves.save_or_show_figure`).

Usage
-----

Display the curves for study ``1`` (looks under ``./runs/study_001/``):

.. code-block:: bash

    yoke-plot-loss --basedir ./runs --IDX 1

Save the figure to disk instead of displaying it:

.. code-block:: bash

    yoke-plot-loss --basedir ./runs --IDX 1 --savefig --savedir ./images

Like other Yoke CLIs, arguments can also be supplied from a file using the
``@`` prefix:

.. code-block:: bash

    yoke-plot-loss @plot_args.input

Arguments
---------

- ``--basedir`` — directory containing the ``study_###`` subdirectories
  (default ``./runs``, matching the ``yoke-start-study`` ``--rundir`` default).
- ``--IDX`` / ``-I`` — index of the study to plot (default ``0``).
- ``--Nsamps_per_trn_pt`` / ``-Nt`` — number of training samples aggregated into
  each plotted point (default ``2012``).
- ``--Nsamps_per_val_pt`` / ``-Nv`` — number of validation samples aggregated
  into each plotted point (default ``250``).
- ``--scatter`` / ``-s`` — overlay the raw loss values as a scatter plot.
- ``--ylim`` / ``-Y`` — upper y-axis limit for the plot (default ``1.0``).
- ``--inprogress`` / ``-P`` — drop the most recent (incomplete) training record
  CSV, useful when the run is still training.
- ``--savedir`` — directory in which to save the figure (default ``./``); created
  if it does not exist.
- ``--savefig`` / ``-S`` — save the figure as a PNG instead of displaying it.

Notes
-----

- If the number of loss samples in a record file is not evenly divisible by the
  ``Nsamps_per_*_pt`` block size, the trailing remainder is dropped (a warning is
  emitted) so the aggregation can proceed.
- Launching a study and producing these record files is handled by
  :doc:`start_study`.
