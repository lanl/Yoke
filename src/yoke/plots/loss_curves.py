"""Reusable functions for plotting Yoke training/validation loss curves.

The record files created by Yoke training harnesses are CSV files read with
pandas. They contain only metric-evaluation information for the *training* and
*validation* sets. This module locates those record files for a given study,
computes quantile bands of the loss over evaluation blocks, and produces a
matplotlib figure of the learning curves.

The functions here are intentionally free of interactive side effects (no
``plt.show``/``plt.savefig`` at import) so that the numeric logic can be unit
tested with a non-interactive backend.
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


# Column names used in Yoke record CSV files. Multi-dataset epoch functions
# append a fourth column naming the dataset each sample came from.
_RECORD_COLUMNS = ["Epoch", "Batch", "Loss"]
_TAGGED_RECORD_COLUMNS = [*_RECORD_COLUMNS, "Dataset"]

# Default quantiles for the plotted loss bands (lower, median, upper).
_DEFAULT_QUANTILES = (0.025, 0.5, 0.975)


def _apply_plot_style() -> None:
    """Apply Yoke's default matplotlib style for loss-curve figures.

    Sets Type-42 fonts (avoids Type-3 fonts in output figures), a serif font
    family, and a square default figure size.
    """
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rc("font", **{"family": "serif"})
    plt.rcParams["figure.figsize"] = (6, 6)


def find_record_csvs(
    basedir: str,
    studyIDX: int,
    kind: str,
    inprogress: bool = False,
) -> tuple[list[str], list[int]]:
    """Locate Yoke record CSVs for a single study and parse their epochs.

    Args:
        basedir (str): Directory containing ``study_###`` subdirectories.
        studyIDX (int): Index of the study whose records to find.
        kind (str): Record type, either ``"training"`` or ``"validation"``.
        inprogress (bool): If True, drop the most recent CSV (useful when a run
            is still training and the last record file is incomplete). Ignored
            for validation records and when no files are found.

    Returns:
        tuple[list[str], list[int]]: A sorted list of CSV file paths and the
        corresponding list of parsed epoch integers.

    Raises:
        ValueError: If ``kind`` is not ``"training"`` or ``"validation"``.
    """
    if kind not in ("training", "validation"):
        msg = f"kind must be 'training' or 'validation', got {kind!r}"
        raise ValueError(msg)

    pattern = f"{basedir}/study_{studyIDX:03d}/{kind}_study{studyIDX:03d}_epoch*.csv"
    csv_list = sorted(glob.glob(pattern))

    # Throw out the most recent training CSV if the run is still in progress.
    if inprogress and kind == "training" and csv_list:
        csv_list.pop()

    epochs = []
    for csv_path in csv_list:
        epoch_str = csv_path.split("epoch")[1]
        epochs.append(int(epoch_str.split(".")[0]))

    return csv_list, epochs


def load_record_csv(path: str) -> pd.DataFrame:
    """Read a single Yoke record CSV into a DataFrame.

    Args:
        path (str): Path to a record CSV with ``Epoch, Batch, Loss`` columns,
            optionally followed by a ``Dataset`` tag column.

    Returns:
        pandas.DataFrame: DataFrame with columns ``Epoch``, ``Batch``, ``Loss``
        and, for tagged records, ``Dataset``.

    Raises:
        ValueError: If the file has neither 3 nor 4 columns.
    """
    df = pd.read_csv(path, sep=", ", header=None, engine="python")
    if df.shape[1] == len(_RECORD_COLUMNS):
        df.columns = _RECORD_COLUMNS
    elif df.shape[1] == len(_TAGGED_RECORD_COLUMNS):
        df.columns = _TAGGED_RECORD_COLUMNS
    else:
        raise ValueError(f"{path}: expected 3 or 4 columns, found {df.shape[1]}.")
    return df


def _select_dataset(df: pd.DataFrame, dataset: str | None, path: str) -> pd.DataFrame:
    """Keep only the records of one dataset tag, or all when ``dataset`` is None.

    Args:
        df (pandas.DataFrame): Records from :func:`load_record_csv`.
        dataset (str | None): Dataset tag to keep.
        path (str): Source file, for error messages.

    Returns:
        pandas.DataFrame: The selected records.

    Raises:
        ValueError: If a dataset is requested but the file has no tags.
    """
    if dataset is None:
        return df
    if "Dataset" not in df.columns:
        raise ValueError(f"{path} has no Dataset column; cannot select {dataset!r}.")
    return df[df["Dataset"] == dataset]


def compute_quantile_bands(
    losses: np.ndarray,
    nsamps_per_pt: int,
    quantiles: tuple[float, float, float] = _DEFAULT_QUANTILES,
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape a loss vector into blocks and compute per-block quantiles.

    The loss vector is reshaped into ``(nsamps_per_pt, -1)`` columns; each
    column becomes one plotted evaluation point. If the number of losses is not
    divisible by ``nsamps_per_pt``, the trailing remainder is dropped (with a
    warning) so the reshape succeeds.

    Args:
        losses (numpy.ndarray): 1D array of loss values.
        nsamps_per_pt (int): Number of samples aggregated into each plot point.
        quantiles (tuple[float, float, float]): Lower, median, and upper
            quantiles to compute across each block.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Integer positions for each plot
        point and a ``(3, n_points)`` array of quantile bands.
    """
    losses = np.asarray(losses)
    remainder = len(losses) % nsamps_per_pt
    if remainder != 0:
        warnings.warn(
            "# of loss samples not divisible by nsamps_per_pt; "
            "remainder of loss samples at the end will be excluded.",
            stacklevel=2,
        )
        losses = losses[: len(losses) - remainder]

    block_loss = losses.reshape((nsamps_per_pt, -1))
    positions = np.arange(block_loss.shape[1])
    bands = np.quantile(block_loss, list(quantiles), axis=0)

    return positions, bands


def plot_loss_curves(
    basedir: str,
    studyIDX: int,
    *,
    nsamps_per_trn_pt: int,
    nsamps_per_val_pt: int,
    scatter: bool = False,
    ylim: float = 1.0,
    inprogress: bool = False,
    dataset: str | None = None,
) -> "matplotlib.figure.Figure":
    """Build a training/validation loss-curve figure for one study.

    Training curves are drawn in blue and validation curves in red, each as a
    median line with lower/upper quantile bands, plotted against a running
    "Evaluation Index". Optionally overlays raw losses as a scatter.

    Args:
        basedir (str): Directory containing ``study_###`` subdirectories.
        studyIDX (int): Index of the study to plot.
        nsamps_per_trn_pt (int): Samples aggregated per training plot point.
        nsamps_per_val_pt (int): Samples aggregated per validation plot point.
        scatter (bool): If True, scatter the raw loss values.
        ylim (float): Upper y-axis limit for the plot.
        inprogress (bool): If True, drop the most recent (incomplete) training
            record CSV.
        dataset (str | None): If given, plot only records tagged with this
            dataset (e.g. ``"lsc"``); requires tagged 4-column records.

    Returns:
        matplotlib.figure.Figure: The figure containing the loss curves. The
        caller is responsible for saving or displaying it.
    """
    _apply_plot_style()

    trn_csv_list, trn_file_epochs = find_record_csvs(
        basedir, studyIDX, "training", inprogress=inprogress
    )
    val_csv_list, val_file_epochs = find_record_csvs(basedir, studyIDX, "validation")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()

    vIDX = 0
    startIDX = 0
    for tIDX, trn_csv in enumerate(trn_csv_list):
        trn_DF = _select_dataset(load_record_csv(trn_csv), dataset, trn_csv)
        all_losses = trn_DF.loc[:, "Loss"].values
        all_idxs = trn_DF.loc[:, "Batch"].values
        scaled_trn_idxs = np.array(all_idxs) / nsamps_per_trn_pt

        trn_positions, trn_qnts = compute_quantile_bands(all_losses, nsamps_per_trn_pt)

        if tIDX == 0 and scatter:
            plt.scatter(scaled_trn_idxs, all_losses, alpha=0.2)
        plt.plot(startIDX + trn_positions, trn_qnts[0, :], ":b")
        plt.plot(
            startIDX + trn_positions,
            trn_qnts[1, :],
            "-b",
            label="Training" if tIDX == 0 else None,
        )
        plt.plot(startIDX + trn_positions, trn_qnts[2, :], ":b")

        startIDX += trn_positions[-1]

        if vIDX < len(val_file_epochs) and (
            trn_file_epochs[tIDX] == val_file_epochs[vIDX]
        ):
            val_DF = _select_dataset(
                load_record_csv(val_csv_list[vIDX]), dataset, val_csv_list[vIDX]
            )
            all_val_losses = val_DF.loc[:, "Loss"].values
            all_val_idxs = np.array(val_DF.loc[:, "Batch"].values)
            scaled_val_idxs = all_val_idxs / nsamps_per_val_pt + np.array(
                scaled_trn_idxs[-1]
            )

            val_positions, val_qnts = compute_quantile_bands(
                all_val_losses, nsamps_per_val_pt
            )

            if vIDX == 0 and scatter:
                plt.scatter(scaled_val_idxs, all_val_losses, alpha=0.2)
            plt.plot(startIDX + val_positions, val_qnts[0, :], ":r")
            plt.plot(
                startIDX + val_positions,
                val_qnts[1, :],
                "-r",
                label="Validation" if vIDX == 0 else None,
            )
            plt.plot(startIDX + val_positions, val_qnts[2, :], ":r")

            vIDX += 1
            startIDX += val_positions[-1]

    plt.legend(fontsize=16)
    ax.set_ylim(0.0, ylim)
    ax.set_ylabel("Loss", fontsize=16)
    ax.set_xlabel("Evaluation Index", fontsize=16)

    return fig


def save_or_show_figure(
    fig: "matplotlib.figure.Figure",
    studyIDX: int,
    savefig: bool,
    savedir: str,
) -> str | None:
    """Save the figure to disk or display it interactively.

    Args:
        fig (matplotlib.figure.Figure): Figure to save or display.
        studyIDX (int): Study index, used to name the saved file.
        savefig (bool): If True, save the figure as a PNG; otherwise display it.
        savedir (str): Directory in which to save the figure (created if
            necessary).

    Returns:
        str | None: The saved file path if ``savefig`` is True, otherwise None.
    """
    if savefig:
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        filename = f"{savedir}/study{studyIDX:03d}_TandV_curve.png"
        fig.savefig(filename, bbox_inches="tight")
        return filename

    plt.show()
    return None
