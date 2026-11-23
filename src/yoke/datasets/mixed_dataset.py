"""Weighted mixture of several map-style datasets.

Used to train one model on several datasets at once (e.g. LSC and CYL PAGOSA
simulations). The mixture's index range is split into contiguous blocks, one
per dataset, sized in proportion to the requested weights. A shuffling
``DistributedSampler`` then draws indices uniformly from the whole range, so
each batch comes from dataset ``i`` with probability ``weight_i``. The split
depends only on the constructor arguments, so it is identical on every rank
and worker.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from torch.utils.data import Dataset


class MixedDataSet(Dataset):
    """Draw samples from several datasets in fixed proportions.

    Args:
        datasets (Sequence[Dataset]): Datasets to mix. Each must accept any
            non-negative integer index (the Yoke datasets wrap indices).
        weights (Sequence[float]): Positive relative weight per dataset.
        length (int): Total number of samples in the mixture.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        weights: Sequence[float],
        length: int = int(1e6),
    ) -> None:
        """Validate inputs and compute the index block boundaries."""
        if not datasets or len(datasets) != len(weights):
            raise ValueError("datasets and weights must be non-empty, equal length.")
        weights = np.asarray(weights, dtype=float)
        if np.any(weights <= 0):
            raise ValueError(
                "All weights must be positive; leave out datasets with weight 0."
            )
        if length < len(datasets):
            raise ValueError("length must be at least the number of datasets.")

        self.datasets = list(datasets)
        self.length = int(length)
        # Cumulative block ends, e.g. weights (3, 1) and length 8 -> [6, 8].
        ends = np.round(np.cumsum(weights / weights.sum()) * self.length).astype(int)
        ends[-1] = self.length
        self.block_ends = ends

    def __len__(self) -> int:
        """Return the total number of samples in the mixture."""
        return self.length

    def __getitem__(self, index: int) -> Any:  # noqa: ANN401
        """Return the sample at ``index`` from the dataset owning that block.

        Args:
            index (int): Index in ``[0, len(self))``.

        Returns:
            Any: The selected dataset's sample.
        """
        if not 0 <= index < self.length:
            raise IndexError(f"index {index} out of range for length {self.length}")
        which = int(np.searchsorted(self.block_ends, index, side="right"))
        start = 0 if which == 0 else int(self.block_ends[which - 1])
        return self.datasets[which][index - start]
