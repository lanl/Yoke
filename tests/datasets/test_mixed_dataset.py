"""Tests for :class:`yoke.datasets.mixed_dataset.MixedDataSet`."""

import pytest
from torch.utils.data import Dataset

from yoke.datasets.mixed_dataset import MixedDataSet


class _Tagged(Dataset):
    """Dataset returning ``(tag, index)`` for any index."""

    def __init__(self, tag: str) -> None:
        """Store the tag."""
        self.tag = tag

    def __len__(self) -> int:
        """Return a nominal length."""
        return 1

    def __getitem__(self, index: int) -> tuple[str, int]:
        """Return the tag and the local index."""
        return self.tag, index


def test_blocks_follow_weights() -> None:
    """Index blocks are proportional to the weights and locally re-indexed."""
    mixed = MixedDataSet([_Tagged("a"), _Tagged("b")], [3.0, 1.0], length=8)

    tags = [mixed[i][0] for i in range(len(mixed))]
    assert tags == ["a"] * 6 + ["b"] * 2
    assert mixed[6] == ("b", 0)
    assert mixed[5] == ("a", 5)


def test_single_dataset_passthrough() -> None:
    """A single dataset owns the whole range."""
    mixed = MixedDataSet([_Tagged("a")], [0.5], length=4)
    assert [mixed[i] for i in range(4)] == [("a", i) for i in range(4)]


def test_out_of_range_raises() -> None:
    """Indices outside the mixture raise IndexError."""
    mixed = MixedDataSet([_Tagged("a")], [1.0], length=2)
    with pytest.raises(IndexError):
        mixed[2]


@pytest.mark.parametrize(
    ("datasets", "weights", "length", "match"),
    [
        ([], [], 10, "non-empty"),
        ([_Tagged("a")], [1.0, 2.0], 10, "non-empty"),
        ([_Tagged("a"), _Tagged("b")], [1.0, 0.0], 10, "positive"),
        ([_Tagged("a"), _Tagged("b")], [1.0, 1.0], 1, "length"),
    ],
)
def test_invalid_arguments(
    datasets: list[Dataset], weights: list[float], length: int, match: str
) -> None:
    """Bad dataset/weight/length combinations raise ValueError."""
    with pytest.raises(ValueError, match=match):
        MixedDataSet(datasets, weights, length=length)
