"""Custom dataset class specialized for a new data source (PDEBench).

Currently, it only handles compNS dataset.
"""

from typing import Optional, Callable
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from yoke.datasets.hdf5_datasets import CompNSDataset


class CompNSLodeRunnerDataset(Dataset):
    """PyTorch dataset for loading CompNS data from HDF5 files."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        n_steps: int = 2,
        dt: int = 1,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            root_dir (str): Path to folder containing HDF5 files.
            split (str): 'train', 'val', or 'test'
            n_steps (int): Number of input steps (T).
            dt (int): Time delta between steps.
            transform (Callable, optional): Optional transform (e.g., ResizePadCrop).
        """
        self.dataset = CompNSDataset(
            path=root_dir,
            include_string="",  # Match all
            n_steps=n_steps,
            dt=dt,
            split=split,
            train_val_test=(0.8, 0.1, 0.1),
        )
        self.transform = transform

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get one sequence sample and target frame by index."""
        img_seq, _, target = self.dataset[idx]  # [T, C, H, W], target: [C, H, W]
        full_seq = torch.cat([img_seq, target.unsqueeze(0)], dim=0)

        if self.transform:
            full_seq = self.transform(full_seq)

        return full_seq[:-1], full_seq[-1]    #input: [T, C, H, W], target: [C, H, W]


class CompNSDataModule(pl.LightningDataModule):
    """Lightning DataModule for CompNS dataset."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize the data module."""
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage: Optional[str] = None) -> None:
        """Split the dataset into training and validation sets."""
        self.train_set = CompNSLodeRunnerDataset(
            self.data_dir, split="train", transform=self.transform
        )
        self.val_set = CompNSLodeRunnerDataset(
            self.data_dir, split="val", transform=self.transform
        )

    def train_dataloader(self) -> DataLoader:
        """Return training data loader."""
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation data loader."""
        return DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=4
        )
