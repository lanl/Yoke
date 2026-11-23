"""Vector-to-vector datasets built from one row-per-sample CSV table.

Many small studies (for example melt-pool simulations, where each row holds a
simulation's input parameters and its scalar outputs) fit a whole dataset in
one CSV. :class:`TabularCSV_DataSet` turns selected columns of such a table
into ``(input, target)`` pairs for Yoke's ``array`` batch contract, which the
DDP array datastep / epoch functions in :mod:`yoke.utils.training` consume.

Every sample is a 2-tuple::

    (inpt, truth)

- ``inpt``: ``(n_inputs,)`` float32, the ``input_cols`` values in order.
- ``truth``: ``(n_targets,)`` float32, the ``target_cols`` values in order.

Splitting is done by seeded, optionally stratified k-fold
(:func:`kfold_indices`), so a training run is fully described by
``(n_folds, fold, split_seed)`` and no filelists are needed. A ``"train"`` and
a ``"holdout"`` dataset built with the same arguments are exact complements.

Normalization (optional ``log10`` of chosen columns, then a z-score per column)
is computed at construction from the **training fold only** and applied to
whichever split is requested, so no holdout information leaks into training.
The statistics are kept as attributes so predictions can be mapped back to
physical units.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

#: Accepted values of the ``split`` argument.
SPLITS = ("train", "holdout")


def kfold_indices(
    n_rows: int,
    n_folds: int,
    fold: int,
    seed: int,
    strata: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of one fold of a seeded, optionally stratified k-fold split.

    The rows of each stratum are shuffled with one ``numpy`` generator seeded
    by ``seed``. The shuffled strata are then concatenated and dealt to the
    folds round-robin. Because the dealing continues across stratum
    boundaries, both the fold sizes and every stratum's count per fold differ
    by at most one between folds.

    Args:
        n_rows (int): Number of rows to split.
        n_folds (int): Number of folds, at least 2.
        fold (int): Fold held out, in ``[0, n_folds)``.
        seed (int): Seed for the shuffle. Different seeds give different
            repeats of the k-fold.
        strata (np.ndarray | None): Optional per-row stratum labels, of
            length ``n_rows``. ``None`` treats all rows as one stratum.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(train_idx, holdout_idx)``, sorted
        int64 arrays that partition ``range(n_rows)``.

    Raises:
        ValueError: If ``n_folds < 2``, ``fold`` is out of range, there are
            fewer rows than folds, or ``strata`` has the wrong length.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}.")
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds}), got {fold}.")
    if n_rows < n_folds:
        raise ValueError(f"Cannot split {n_rows} rows into {n_folds} folds.")
    if strata is None:
        strata = np.zeros(n_rows, dtype=np.int64)
    strata = np.asarray(strata)
    if len(strata) != n_rows:
        raise ValueError(f"strata has length {len(strata)}, expected n_rows={n_rows}.")

    rng = np.random.default_rng(seed)
    # np.unique sorts the labels, so the stratum order (and hence the split)
    # does not depend on row order within the table.
    order = np.concatenate(
        [rng.permutation(np.flatnonzero(strata == s)) for s in np.unique(strata)]
    )
    fold_of_row = np.empty(n_rows, dtype=np.int64)
    fold_of_row[order] = np.arange(n_rows) % n_folds

    holdout_idx = np.flatnonzero(fold_of_row == fold)
    train_idx = np.flatnonzero(fold_of_row != fold)
    return train_idx, holdout_idx


class TabularCSV_DataSet(Dataset):
    """Fixed-index ``(input, target)`` vectors from selected columns of a CSV.

    Construction reads and filters the table, selects this run's fold, applies
    ``log10`` to ``log_cols``, and z-scores every input and target column with
    the training fold's mean and standard deviation. The (tiny) result is held
    in memory, so ``__getitem__`` is a lookup.

    Args:
        csv_path (str): Path to the CSV file, one sample per row.
        input_cols (Sequence[str]): Columns forming the input vector, in order.
        target_cols (Sequence[str]): Columns forming the target vector, in
            order.
        split (str): ``"train"`` (the other ``n_folds - 1`` folds) or
            ``"holdout"`` (fold ``fold``).
        n_folds (int): Number of k-fold folds.
        fold (int): Index of the held-out fold, in ``[0, n_folds)``.
        split_seed (int): Seed of the fold shuffle; use one seed per repeat.
        require_true (Sequence[str] | None): Boolean columns that must be
            ``True`` for a row to be kept (for example ``["melted"]``).
        stratify_col (str | None): Column whose values stratify the folds.
        log_cols (Sequence[str] | None): Input or target columns replaced by
            their ``log10`` before standardizing. Values must be positive.
        normalize (bool): If ``True``, z-score inputs and targets with the
            training-fold statistics. If ``False``, return the (possibly
            log-transformed) values unchanged; the statistics attributes are
            still computed.
        id_col (str | None): Column holding a per-row identifier, exposed as
            :attr:`row_names` if it exists in the table.

    Attributes:
        input_mean (torch.Tensor): ``(n_inputs,)`` training-fold mean, after
            any ``log10``.
        input_std (torch.Tensor): ``(n_inputs,)`` training-fold standard
            deviation (``ddof=0``).
        target_mean (torch.Tensor): ``(n_targets,)`` training-fold mean.
        target_std (torch.Tensor): ``(n_targets,)`` training-fold standard
            deviation.
        row_names (list[str]): ``id_col`` values of this split's rows, or
            their row numbers in the filtered table if ``id_col`` is absent.

    Raises:
        KeyError: If a named column is missing from the CSV.
        ValueError: On a bad ``split`` or fold, a non-numeric input/target
            column, a non-boolean ``require_true`` column, no rows left after
            filtering, a non-positive value in a ``log_cols`` column, or a
            column with zero training-fold standard deviation.
    """

    def __init__(
        self,
        csv_path: str,
        input_cols: Sequence[str],
        target_cols: Sequence[str],
        split: str = "train",
        n_folds: int = 5,
        fold: int = 0,
        split_seed: int = 0,
        require_true: Sequence[str] | None = None,
        stratify_col: str | None = None,
        log_cols: Sequence[str] | None = None,
        normalize: bool = True,
        id_col: str | None = "name",
    ) -> None:
        """Read, filter, split, and normalize the table."""
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}.")
        self.input_cols = list(input_cols)
        self.target_cols = list(target_cols)
        self.split = split
        require_true = list(require_true or [])
        log_cols = list(log_cols or [])

        df = pd.read_csv(csv_path)

        # --- Column checks ---
        value_cols = self.input_cols + self.target_cols
        named = value_cols + require_true + log_cols
        if stratify_col is not None:
            named.append(stratify_col)
        missing = [c for c in dict.fromkeys(named) if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not in {csv_path}: {missing}")
        not_numeric = [c for c in value_cols if not pd.api.types.is_numeric_dtype(df[c])]
        # Booleans count as numeric to pandas but are not meaningful values.
        not_numeric += [c for c in value_cols if pd.api.types.is_bool_dtype(df[c])]
        if not_numeric:
            raise ValueError(f"Input/target columns must be numeric: {not_numeric}")
        stray_logs = [c for c in log_cols if c not in value_cols]
        if stray_logs:
            raise ValueError(f"log_cols must be input or target columns: {stray_logs}")

        # --- Row filter ---
        for col in require_true:
            # pandas parses True/False text as bool; anything else (strings,
            # 0/1 ints) is rejected rather than guessed at.
            if not pd.api.types.is_bool_dtype(df[col]):
                raise ValueError(f"require_true column {col!r} is not boolean.")
            df = df[df[col]]
        df = df.reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No rows left after require_true={require_true}.")

        # --- Fold selection ---
        strata = None if stratify_col is None else df[stratify_col].to_numpy()
        train_idx, holdout_idx = kfold_indices(
            len(df), n_folds, fold, split_seed, strata=strata
        )
        split_idx = train_idx if split == "train" else holdout_idx

        # --- log10 transform (all rows, so both splits get the same one) ---
        values = df[value_cols].to_numpy(dtype=np.float64)
        for col in log_cols:
            j = value_cols.index(col)
            if np.any(values[:, j] <= 0.0):
                raise ValueError(f"log_cols column {col!r} has values <= 0.")
            values[:, j] = np.log10(values[:, j])

        # --- Training-fold statistics ---
        mean = values[train_idx].mean(axis=0)
        std = values[train_idx].std(axis=0)
        constant = [c for c, s in zip(value_cols, std, strict=True) if s == 0.0]
        if constant:
            # A constant column carries no information and can't be z-scored
            # (e.g. a composition that is zero in every row of this table).
            raise ValueError(f"Columns constant over the training fold: {constant}")
        if normalize:
            values = (values - mean) / std

        n_in = len(self.input_cols)
        self.input_mean = torch.tensor(mean[:n_in], dtype=torch.float32)
        self.input_std = torch.tensor(std[:n_in], dtype=torch.float32)
        self.target_mean = torch.tensor(mean[n_in:], dtype=torch.float32)
        self.target_std = torch.tensor(std[n_in:], dtype=torch.float32)

        split_values = torch.tensor(values[split_idx], dtype=torch.float32)
        self.inputs = split_values[:, :n_in].contiguous()
        self.targets = split_values[:, n_in:].contiguous()

        if id_col is not None and id_col in df.columns:
            self.row_names = [str(x) for x in df[id_col].to_numpy()[split_idx]]
        else:
            self.row_names = [str(i) for i in split_idx]

    def __len__(self) -> int:
        """Return the number of rows in this split."""
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(inpt, truth)`` pair of row ``index`` of this split.

        Args:
            index (int): Row index within the split.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(n_inputs,)`` and
            ``(n_targets,)`` float32 tensors.
        """
        return self.inputs[index], self.targets[index]
