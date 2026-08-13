"""Plot training/validation loss curves for the 9-band scalar temporal LodeRunner.

The 9-band training loop writes the same per-batch CSV records as the g/r/i runs
(``training_study<studyIDX>_epoch<epochIDX>.csv`` and the matching validation
files), with the format ``epoch, batch, loss`` where ``loss`` is the single
masked scalar loss per sample. The loss-curve plotting logic is therefore
identical; this module just reuses ``plot_loss_curves_gri`` with a 9-band title.

Run directly, e.g.:
    python plot_loss_curves_9band.py --study 24
"""

import plot_loss_curves_gri as base


def main():
    parser = base.argparse.ArgumentParser(
        description=(
            "Plot training/validation loss curves for scalar temporal "
            "LodeRunner 9-band runs."
        )
    )

    parser.add_argument("--study", type=int, default=base.DEFAULT_STUDY)
    parser.add_argument("--runs_root", type=str, default=base.DEFAULT_RUNS_ROOT)
    parser.add_argument("--train_pattern", type=str, default=None)
    parser.add_argument("--val_pattern", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)

    parser.add_argument(
        "--columns",
        type=str,
        default=base.DEFAULT_COLUMNS,
        help=(
            "Comma-separated CSV column names. The first two must be "
            "epoch,batch. For 9-band runs the loss is a single scalar column, "
            "so the default epoch,batch,loss is correct."
        ),
    )

    parser.add_argument("--title", type=str, default="9-band loss curves")
    parser.add_argument("--dpi", type=int, default=200)

    parser.add_argument(
        "--logy",
        dest="logy",
        action="store_true",
        default=True,
        help="Use log scale on y-axis. This is the default.",
    )
    parser.add_argument(
        "--linear",
        dest="logy",
        action="store_false",
        help="Use linear y-axis.",
    )

    parser.add_argument(
        "--show_std",
        action="store_true",
        help="Shade +/- one epoch standard deviation.",
    )
    parser.add_argument(
        "--require_val",
        action="store_true",
        help=(
            "Fail instead of continuing when validation records are missing "
            "or invalid."
        ),
    )

    # Scheduled-sampling regime shading. 9-band runs anneal the teacher-forcing
    # ratio, so shade the warmup / anneal / free-run regimes by default. These
    # must match the schedule in training_input.tmpl / training_START.input.
    parser.add_argument(
        "--shade_regimes",
        dest="shade_regimes",
        action="store_true",
        default=True,
        help="Shade the teacher-forcing warmup/anneal/free-run regimes. Default.",
    )
    parser.add_argument(
        "--no_shade_regimes",
        dest="shade_regimes",
        action="store_false",
        help="Disable teacher-forcing regime shading.",
    )
    parser.add_argument(
        "--tf_ramp_start_epoch",
        type=int,
        default=20,
        help="Absolute epoch at which the teacher-forcing anneal begins. Must "
        "match the training schedule. Default 20.",
    )
    parser.add_argument(
        "--tf_ramp_epochs",
        type=int,
        default=20,
        help="Number of epochs the teacher-forcing ratio anneals over. Must "
        "match the training schedule. Default 20.",
    )

    args = parser.parse_args()

    defaults = base.default_patterns(args.study, args.runs_root)

    args.train_pattern = args.train_pattern or defaults["train"]
    args.val_pattern = args.val_pattern or defaults["val"]
    args.out = args.out or defaults["out"]

    train = base.load_records(args.train_pattern, args.columns)
    base.print_loaded("training", train)

    try:
        val = base.load_records(args.val_pattern, args.columns)
        base.print_loaded("validation", val)
    except Exception as exc:
        if args.require_val:
            raise
        print(f"No validation curve plotted: {exc}")
        val = None

    base.plot_epoch_curves(train, val, args)


if __name__ == "__main__":
    main()
