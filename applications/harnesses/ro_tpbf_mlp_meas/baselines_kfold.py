r"""Simple k-fold baselines to compare with the TPBF MLP (``summarize_kfold.py``).

Run from the harness directory (CPU, about a second; no training jobs)::

    python baselines_kfold.py --rundir ./runs --csv tpbf_kfold.csv \
        --compare mlp=kfold_summary_metrics.csv

For each study (CSV row) this reads the study's ``studyNNN_START.input`` with
the train script's parser, so it uses exactly the same split, columns,
filtering, ``log10`` transform, and train-fold z-scoring as the MLP. It then
fits three baselines on the training fold and scores the **whole** held-out
fold:

- ``mean``: predict the training-fold mean of every target;
- ``linear``: least squares on ``[1, x]``;
- ``quadratic``: least squares on ``[1, x, x_i * x_j (i <= j)]``.

The fits are in z-scored space (the inputs as the MLP sees them), and the
predictions are mapped back to physical units with the training fold's
statistics. Only the study input file and the table CSV are needed, so the
studies need not have been trained, and no checkpoints are read.

Outputs:

- ``<out_prefix>_metrics.csv``: one row per (model, study, target), with the
  same metric columns as ``summarize_kfold.py`` (``rmse``, ``mae``,
  ``median_rel_err``);
- a printed table: the mean and std over studies of each metric, per model and
  target. Each ``--compare label=path`` adds a ``summarize_kfold.py`` metrics
  CSV as another model row.
"""

import argparse
import os

import numpy as np
import pandas as pd

from summarize_kfold import load_train_parser
from yoke.datasets.tabular_csv import TabularCSV_DataSet

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = ["rmse", "mae", "median_rel_err"]


def design_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial features of degree 1 or 2, with an intercept column.

    Args:
        x (np.ndarray): Inputs, ``(n, d)``.
        degree (int): ``1`` for ``[1, x]``, ``2`` to add every product
            ``x_i * x_j`` with ``i <= j`` (squares and pairwise interactions).

    Returns:
        np.ndarray: ``(n, 1 + d)`` or ``(n, 1 + d + d(d+1)/2)``.
    """
    cols = [np.ones(len(x)), *x.T]
    if degree == 2:
        d = x.shape[1]
        cols += [x[:, i] * x[:, j] for i in range(d) for j in range(i, d)]
    return np.stack(cols, axis=1)


def fit_predict(
    model: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
) -> np.ndarray:
    """Fit one baseline on the training fold and predict the test inputs.

    Args:
        model (str): ``"mean"``, ``"linear"``, or ``"quadratic"``.
        x_train (np.ndarray): ``(n_train, d)`` z-scored inputs.
        y_train (np.ndarray): ``(n_train, k)`` z-scored targets.
        x_test (np.ndarray): ``(n_test, d)`` z-scored inputs.

    Returns:
        np.ndarray: ``(n_test, k)`` z-scored predictions.

    Raises:
        ValueError: On an unknown ``model``.
    """
    if model == "mean":
        # Targets are z-scored on the training fold, so its mean is 0.
        return np.tile(y_train.mean(axis=0), (len(x_test), 1))
    degrees = {"linear": 1, "quadratic": 2}
    if model not in degrees:
        raise ValueError(f"Unknown baseline {model!r}.")
    a = design_matrix(x_train, degrees[model])
    coef, *_ = np.linalg.lstsq(a, y_train, rcond=None)
    return design_matrix(x_test, degrees[model]) @ coef


def score_study(
    parser: argparse.ArgumentParser, study_dir: str, studyIDX: int, models: list[str]
) -> pd.DataFrame:
    """Score every baseline on one study's held-out fold.

    Args:
        parser (argparse.ArgumentParser): The train script's parser.
        study_dir (str): The ``study_NNN`` directory.
        studyIDX (int): Study index.
        models (list[str]): Baselines to fit.

    Returns:
        pd.DataFrame: One row per (model, target).
    """
    start_input = os.path.join(study_dir, f"study{studyIDX:03d}_START.input")
    args = parser.parse_args([f"@{start_input}"])
    kwargs = {
        "csv_path": os.path.join(study_dir, args.table_csv),
        "input_cols": args.input_cols,
        "target_cols": args.target_cols,
        "n_folds": args.n_folds,
        "fold": args.fold,
        "split_seed": args.split_seed,
        "require_true": args.require_true,
        "stratify_col": args.stratify_col or None,
        "log_cols": args.log_cols,
    }
    train = TabularCSV_DataSet(split="train", **kwargs)
    holdout = TabularCSV_DataSet(split="holdout", **kwargs)

    x_train = train.inputs.numpy().astype(np.float64)
    y_train = train.targets.numpy().astype(np.float64)
    x_test = holdout.inputs.numpy().astype(np.float64)
    mean = holdout.target_mean.numpy().astype(np.float64)
    std = holdout.target_std.numpy().astype(np.float64)
    truth = holdout.targets.numpy().astype(np.float64) * std + mean
    log_targets = [j for j, c in enumerate(args.target_cols) if c in args.log_cols]
    truth[:, log_targets] = 10.0 ** truth[:, log_targets]

    frames = []
    for model in models:
        pred = fit_predict(model, x_train, y_train, x_test) * std + mean
        pred[:, log_targets] = 10.0 ** pred[:, log_targets]
        err = pred - truth
        frames.append(
            pd.DataFrame(
                {
                    "model": model,
                    "studyIDX": studyIDX,
                    "fold": args.fold,
                    "split_seed": args.split_seed,
                    "target": args.target_cols,
                    "n_holdout": len(holdout),
                    "rmse": np.sqrt((err**2).mean(axis=0)),
                    "mae": np.abs(err).mean(axis=0),
                    "median_rel_err": np.median(np.abs(err) / np.abs(truth), axis=0),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def summary_table(metrics: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """Mean and std over studies of each metric, per model and target.

    Args:
        metrics (pd.DataFrame): Per-study rows with ``model`` and ``target``.
        targets (list[str]): Target order for the table rows.

    Returns:
        pd.DataFrame: Indexed by ``(model, target)``, with ``(metric, mean|std)``
        columns.
    """
    table = metrics.groupby(["model", "target"], sort=False)[METRICS].agg(
        ["mean", "std"]
    )
    order = list(dict.fromkeys(metrics["model"]))
    return table.reindex(pd.MultiIndex.from_product([order, targets]))


def main() -> None:
    """Fit and score the baselines for every study, then print the comparison."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rundir", default="./runs", help="Directory of study_NNN dirs.")
    p.add_argument("--csv", default="tpbf_kfold.csv", help="Study CSV.")
    p.add_argument(
        "--train_script",
        default=os.path.join(HERE, "train_tpbf_mlp.py"),
        help="Train script whose parser reads the study input files.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["mean", "linear", "quadratic"],
        choices=["mean", "linear", "quadratic"],
        help="Baselines to fit.",
    )
    p.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="A summarize_kfold.py metrics CSV to include as model LABEL (repeatable).",
    )
    p.add_argument(
        "--out_prefix", default="kfold_baselines", help="Prefix for the output CSV."
    )
    opts = p.parse_args()

    parser = load_train_parser(opts.train_script)
    studies = pd.read_csv(opts.csv, index_col=0, comment="#").index
    metrics = pd.concat(
        [
            score_study(
                parser,
                os.path.join(opts.rundir, f"study_{int(i):03d}"),
                int(i),
                opts.models,
            )
            for i in studies
        ],
        ignore_index=True,
    )
    metrics.to_csv(f"{opts.out_prefix}_metrics.csv", index=False)
    targets = list(dict.fromkeys(metrics["target"]))

    # Append MLP results; keep only the studies scored here, so every model is
    # averaged over the same splits.
    for spec in opts.compare:
        label, _, path = spec.partition("=")
        if not path:
            p.error(f"--compare expects LABEL=PATH, got {spec!r}")
        other = pd.read_csv(path)
        other = other.assign(model=label)
        splits = metrics[["fold", "split_seed"]].drop_duplicates()
        other = other.merge(splits, on=["fold", "split_seed"])
        metrics = pd.concat([metrics, other[metrics.columns]], ignore_index=True)

    pd.set_option("display.width", 140)
    n = metrics.groupby("model")["studyIDX"].nunique().to_dict()
    print(f"Studies per model: {n}")
    print("Held-out metrics, mean and std over studies:")
    print(summary_table(metrics, targets).to_string(float_format="%.4g"))
    print(f"\nWrote {opts.out_prefix}_metrics.csv")


if __name__ == "__main__":
    main()
