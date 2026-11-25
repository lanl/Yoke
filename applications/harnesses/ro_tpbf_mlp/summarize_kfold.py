"""Score every k-fold study of the TPBF MLP harness on its full held-out fold.

Run from the harness directory after the studies finish (CPU, a few seconds)::

    python summarize_kfold.py --rundir ./runs --csv tpbf_kfold.csv

For each study (CSV row), this reads the study's ``studyNNN_START.input`` with
the train script's own parser, rebuilds the held-out fold with
:class:`yoke.datasets.tabular_csv.TabularCSV_DataSet` (every row; nothing is
dropped), loads the study's latest checkpoint, and predicts. Predictions are
mapped back to physical units with the training fold's target statistics.

It prints, per target, the RMSE, MAE, and median relative error of each study,
then the mean and standard deviation over studies. It writes:

- ``<out_prefix>_metrics.csv``: one row per (study, target);
- ``<out_prefix>_predictions.csv``: one row per (study, held-out simulation),
  with truth and prediction per target, for per-simulation analysis.

Studies with no checkpoint yet are reported and skipped.
"""

import argparse
import glob
import importlib.util
import os
import re

import numpy as np
import pandas as pd
import torch

from yoke.datasets.tabular_csv import TabularCSV_DataSet
from yoke.models.mlp import SwiGLU_mlp, model_args_for_state_dict

HERE = os.path.dirname(os.path.abspath(__file__))


def load_train_parser(script: str) -> argparse.ArgumentParser:
    """Import the train script as a module and return its ``parser``.

    The script does all its work under ``__main__``, so importing only builds
    the parser.

    Args:
        script (str): Path to ``train_tpbf_mlp.py``.

    Returns:
        argparse.ArgumentParser: The script's parser.
    """
    spec = importlib.util.spec_from_file_location("train_tpbf_mlp", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parser


def latest_checkpoint(study_dir: str, studyIDX: int) -> str | None:
    """Return the highest-epoch main checkpoint in ``study_dir``, if any.

    Args:
        study_dir (str): The ``study_NNN`` directory.
        studyIDX (int): Study index.

    Returns:
        str | None: Path of ``studyNNN_modelState_epochEEEE.pth``, or ``None``.
    """
    pattern = os.path.join(study_dir, f"study{studyIDX:03d}_modelState_epoch*.pth")
    found = [
        p for p in glob.glob(pattern) if re.search(r"_epoch\d{4}\.pth$", p) is not None
    ]
    return max(found) if found else None


def score_study(
    parser: argparse.ArgumentParser, study_dir: str, studyIDX: int
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Predict the full held-out fold of one study.

    Args:
        parser (argparse.ArgumentParser): The train script's parser.
        study_dir (str): The ``study_NNN`` directory.
        studyIDX (int): Study index.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame] | None: ``(metrics, predictions)``,
        or ``None`` if the study has no checkpoint.
    """
    ckpt_path = latest_checkpoint(study_dir, studyIDX)
    if ckpt_path is None:
        print(f"study {studyIDX:03d}: no checkpoint yet, skipped")
        return None

    # Parse the study's own input file so the split and columns match training.
    # Relative paths in it (the table CSV) are relative to the study dir.
    start_input = os.path.join(study_dir, f"study{studyIDX:03d}_START.input")
    args = parser.parse_args([f"@{start_input}"])
    table_csv = os.path.join(study_dir, args.table_csv)

    holdout = TabularCSV_DataSet(
        table_csv,
        args.input_cols,
        args.target_cols,
        split="holdout",
        n_folds=args.n_folds,
        fold=args.fold,
        split_seed=args.split_seed,
        require_true=args.require_true,
        stratify_col=args.stratify_col or None,
        log_cols=args.log_cols,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Checkpoints from before the input-LayerNorm fix (2026-11-18) need the
    # legacy input norm switched back on to load; newer ones load as saved.
    model_args = model_args_for_state_dict(
        checkpoint["model_args"], checkpoint["model_state_dict"]
    )
    model = SwiGLU_mlp(**model_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pred_z = model(holdout.inputs)

    # Undo the z-score; targets are not log-transformed in this harness, but
    # handle it anyway so other target choices still come back in units.
    pred = pred_z * holdout.target_std + holdout.target_mean
    truth = holdout.targets * holdout.target_std + holdout.target_mean
    for j, col in enumerate(args.target_cols):
        if col in args.log_cols:
            pred[:, j] = 10.0 ** pred[:, j]
            truth[:, j] = 10.0 ** truth[:, j]
    pred, truth = pred.numpy(), truth.numpy()

    err = pred - truth
    metrics = pd.DataFrame(
        {
            "studyIDX": studyIDX,
            "fold": args.fold,
            "split_seed": args.split_seed,
            "epoch": checkpoint["epoch"],
            "target": args.target_cols,
            "n_holdout": len(holdout),
            "rmse": np.sqrt((err**2).mean(axis=0)),
            "mae": np.abs(err).mean(axis=0),
            "median_rel_err": np.median(np.abs(err) / np.abs(truth), axis=0),
        }
    )
    preds = pd.DataFrame({"studyIDX": studyIDX, "name": holdout.row_names})
    for j, col in enumerate(args.target_cols):
        preds[f"{col}_true"] = truth[:, j]
        preds[f"{col}_pred"] = pred[:, j]
    return metrics, preds


def main() -> None:
    """Score all studies in the CSV and print/write the summary."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rundir", default="./runs", help="Directory of study_NNN dirs.")
    p.add_argument("--csv", default="tpbf_kfold.csv", help="Study CSV.")
    p.add_argument(
        "--train_script",
        default=os.path.join(HERE, "train_tpbf_mlp.py"),
        help="Train script whose parser reads the study input files.",
    )
    p.add_argument(
        "--out_prefix", default="kfold_summary", help="Prefix for output CSVs."
    )
    opts = p.parse_args()

    parser = load_train_parser(opts.train_script)
    studies = pd.read_csv(opts.csv, index_col=0, comment="#").index

    all_metrics, all_preds = [], []
    for studyIDX in studies:
        result = score_study(
            parser, os.path.join(opts.rundir, f"study_{studyIDX:03d}"), int(studyIDX)
        )
        if result is not None:
            all_metrics.append(result[0])
            all_preds.append(result[1])
    if not all_metrics:
        print("No studies with checkpoints; nothing to summarize.")
        return

    metrics = pd.concat(all_metrics, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    metrics.to_csv(f"{opts.out_prefix}_metrics.csv", index=False)
    preds.to_csv(f"{opts.out_prefix}_predictions.csv", index=False)

    pd.set_option("display.width", 120)
    print(metrics.to_string(index=False, float_format="%.4g"))
    summary = metrics.groupby("target", sort=False)[
        ["rmse", "mae", "median_rel_err"]
    ].agg(["mean", "std"])
    print(f"\nOver {metrics['studyIDX'].nunique()} studies (mean, std):")
    print(summary.to_string(float_format="%.4g"))
    print(f"\nWrote {opts.out_prefix}_metrics.csv and {opts.out_prefix}_predictions.csv")


if __name__ == "__main__":
    main()
