"""Utility functions for creating pytorch dataloaders."""

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler


def make_distributed_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    rank: int,
    world_size: int,
) -> DataLoader:
    """Creates a DataLoader with a DistributedSampler.

    Args:
        dataset (torch.utils.data.Dataset): dataset to sample from for data loader
        batch_size (int): batch size
        shuffle (bool): Switch to shuffle dataset
        num_workers (int): Number of processes to load data in parallel
        rank (int): Rank of device for distribution
        world_size (int): Number of DDP processes

    Returns:
        dataloader (torch.utils.data.DataLoader): pytorch dataloader

    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
    )

    pin_memory = True if torch.cuda.is_available() else False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,  # Ensures uniform batch size
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2,
    )


def make_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 8,
    num_batches: int = 100,
    num_workers: int = 4,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Function to create a dataloader from a dataset.

    Each dataloader has batch_size*num_batches samples randomly selected
    from the dataset

    Args:
        dataset(torch.utils.data.Dataset): dataset to sample from for data loader
        batch_size (int): batch size
        num_batches (int): number of batches to include in data loader
        num_workers (int): Number of processes to load data in parallel
        prefetch_factor (int): Specifies the number of batches each worker preloads

    Returns:
        dataloader (torch.utils.data.DataLoader): pytorch dataloader

    """
    # Use randomsampler instead of just shuffle=True so we can specify the
    # number of batchs during an epoch.
    randomsampler = RandomSampler(dataset, num_samples=batch_size * num_batches)
    pin_memory = True if torch.cuda.is_available() else False
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=randomsampler,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )

    return dataloader
