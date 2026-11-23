"""Yoke CLI: Plot training vs. validation loss curves for a study.

This CLI reads the training and validation record CSVs produced by a Yoke
training harness for a single study and renders a learning-curve figure,
either displaying it or saving it as a PNG.

Usage:
    yoke-plot-loss [--basedir ./runs] [--IDX 1] [--savefig --savedir ./]

"""

import argparse

from yoke.helpers import cli
from yoke.plots.loss_curves import plot_loss_curves, save_or_show_figure


def main() -> None:
    """Entry point for the ``yoke-plot-loss`` console script.

    Parses command-line arguments, builds the training/validation loss-curve
    figure for the requested study, and either saves or displays it.
    """
    parser = argparse.ArgumentParser(
        prog="yoke-plot-loss",
        description="Plot Yoke training vs. validation learning curves.",
        fromfile_prefix_chars="@",
    )
    parser = cli.add_plot_loss_args(parser)
    args = parser.parse_args()

    fig = plot_loss_curves(
        args.basedir,
        args.IDX,
        nsamps_per_trn_pt=args.Nsamps_per_trn_pt,
        nsamps_per_val_pt=args.Nsamps_per_val_pt,
        scatter=args.scatter,
        ylim=args.ylim,
        inprogress=args.inprogress,
        dataset=args.dataset,
    )
    save_or_show_figure(fig, args.IDX, args.savefig, args.savedir)


if __name__ == "__main__":
    main()
