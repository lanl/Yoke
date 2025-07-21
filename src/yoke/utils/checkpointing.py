"""Functions for checkpointing models and optimizers in PyTorch.

The functions provided here offer checkpointing capabilities that have behavior specific
to the Yoke framework. They are designed to work seamlessly with the Yoke training and
evaluation processes, ensuring that model states can be saved and restored effectively.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import h5py


def save_model_and_optimizer_hdf5(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    filepath: str,
    compiled: bool = False,
) -> None:
    """Saves the state of a model and optimizer in portable hdf5 format.

    Model and optimizer should be moved to the CPU prior to using this function.

    Args:
        model (torch.nn.Module): Pytorch model to save
        optimizer (torch.optim.Optimizer): Pytorch optimizer to save
        epoch (int): Epoch associated with training
        filepath (str): Where to save
        compiled (bool): Flag to extract original model if model being saved
                         was compiled.

    """
    # If model is wrapped in DataParallel, access the underlying module
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # If the model is a `torch.compiled` version the original model must be
    # extracted first.
    if compiled:
        model = model._orig_mod

    with h5py.File(filepath, "w") as h5f:
        # Save epoch number
        h5f.attrs["epoch"] = epoch

        # Save model parameters and buffers
        for name, param in model.named_parameters():
            data = param.detach().cpu().numpy()
            if data.ndim == 0:  # It's a scalar!
                h5f.attrs["model/parameters/" + name] = data
            else:
                h5f.create_dataset("model/parameters/" + name, data=data)

        for name, buffer in model.named_buffers():
            data = buffer.cpu().numpy()
            if data.ndim == 0:  # It's a scalar!
                h5f.attrs["model/buffers/" + name] = data
            else:
                h5f.create_dataset("model/buffers/" + name, data=data)

        # Save optimizer state
        optimizer_state = optimizer.state_dict()
        for idx, group in enumerate(optimizer_state["param_groups"]):
            group_name = f"optimizer/group{idx}"
            for k, v in group.items():
                # print('group_name:', group_name, k)
                if isinstance(v, (int, float)):
                    h5f.attrs[group_name + "/" + k] = v
                elif isinstance(v, list):
                    h5f.create_dataset(group_name + "/" + k, data=v)

        # Save state values, like momentums
        for idx, state in enumerate(optimizer_state["state"].items()):
            state_name = f"optimizer/state{idx}"
            for k, v in state[1].items():
                # print('state_name:', state_name, k)
                if isinstance(v, torch.Tensor):
                    h5f.create_dataset(
                        state_name + "/" + k, data=v.detach().cpu().numpy()
                    )


def load_model_and_optimizer_hdf5(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str
) -> int:
    """Loads state of model and optimizer stored in an hdf5 format.

    Args:
        model (torch.nn.Module): Pytorch model to load state into.
        optimizer (torch.optim.Optimizer): Pytorch optimizer to load state into.
        filepath (str): Path to the hdf5 checkpoint file.

    Returns:
        epoch (int): Epoch associated with training

    """
    # If model is wrapped in DataParallel, access the underlying module
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    with h5py.File(filepath, "r") as h5f:
        # Get epoch number
        epoch = h5f.attrs["epoch"]

        # Load model parameters and buffers
        for name in h5f.get("model/parameters", []):  # Get the group
            if isinstance(h5f["model/parameters/" + name], h5py.Dataset):
                data = torch.from_numpy(h5f["model/parameters/" + name][:])
            else:
                data = torch.tensor(h5f.attrs["model/parameters/" + name])

            name_list = name.split(".")
            param_name = name_list.pop()
            submod_name = ".".join(name_list)

            model.get_submodule(submod_name)._parameters[param_name].data.copy_(data)

        for name in h5f.get("model/buffers", []):
            if isinstance(h5f["model/buffers/" + name], h5py.Dataset):
                buffer = torch.from_numpy(h5f["model/buffers/" + name][:])
            else:
                buffer = torch.tensor(h5f.attrs["model/buffers/" + name])

            name_list = name.split(".")
            param_name = name_list.pop()
            submod_name = ".".join(name_list)
            model.get_submodule(submod_name)._buffers[param_name].data.copy_(buffer)

        # Rebuild optimizer state (need to call this before loading state)
        optimizer_state = optimizer.state_dict()

        # Load optimizer parameter groups
        for k in h5f.attrs:
            if "optimizer/group" in k:
                # print('k-string:', k)
                idx, param = k.split("/")[1:]
                optimizer_state["param_groups"][int(idx.lstrip("group"))][param] = (
                    h5f.attrs[k]
                )

        # Load state values, like momentums
        for name, group in h5f.items():
            if "optimizer/state" in name:
                state_idx = int(name.split("state")[1])
                param_idx, param_state = list(optimizer_state["state"].items())[
                    state_idx
                ]
                for k in group:
                    optimizer_state["state"][param_idx][k] = torch.from_numpy(
                        group[k][:]
                    )

        # Load optimizer state
        optimizer.load_state_dict(optimizer_state)

    return epoch


def save_model_and_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    filepath: str,
    model_class: type,
    model_args: dict,
) -> None:
    """Class-aware torch checkpointing.

    Saves model & optimizer state along with model-class information using torch.save.
    Works for both DDP and non-DDP training. Model's saved in this way should not be
    considered *deployable*. For deployment the model should be converted to ONNX format.

    - Stores the model's class name and initialization args.
    - Works for both DDP and non-DDP training.
    - If model is wrapped in DDP (`model.module` exists), saves
      `model.module.state_dict()`.
    - If model is NOT using DDP, saves `model.state_dict()`.
    - Moves model and optimizer to CPU to avoid CUDA-specific issues.
    - Saves only on rank 0 when using DDP to prevent redundant writes.
    - If using DDP, synchronizes all processes after saving to ensure consistency.

    Args:
        model (torch.nn.Module): Torch nn.Module instance or DDP version thereof.
        optimizer (torch.optim): Torch optimizer instance
        epoch (int): Epoch index being checkpointed.
        filepath (str): Checkpoint filename.
        model_class (torch.nn.Module class): Class of model being checkpointed.
        model_args (dict): Dictionary of model parameters.
    """
    is_ddp = isinstance(model, nn.parallel.DistributedDataParallel)

    # Get rank if in DDP, else assume single process
    if dist.is_initialized():
        save_rank = dist.get_rank()
    else:
        save_rank = 0

    # Save only on rank 0 in DDP or always in single-GPU mode
    if save_rank == 0:
        if is_ddp:
            model_cpu = model.module.to("cpu")
        else:
            model_cpu = model.to("cpu")

        optimizer_cpu = optimizer.state_dict()
        for state in optimizer_cpu["state"].values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu")

        checkpoint = {
            "epoch": epoch,
            "model_class": model_class.__name__,  # Store model class as a string
            "model_args": model_args,  # Store model init arguments
            "model_state_dict": model_cpu.state_dict(),
            "optimizer_state_dict": optimizer_cpu,
        }

        torch.save(checkpoint, filepath)
        print(f"[Rank {save_rank}] Saved checkpoint at epoch {epoch} -> {filepath}")

    # Ensure all processes synchronize before moving on (only if using DDP)
    if dist.is_initialized():
        dist.barrier()


def load_model_and_optimizer(
    filepath: str,
    optimizer: torch.optim.Optimizer,
    available_models: dict,
    device: str = "cuda",
) -> tuple[torch.nn.Module, int]:
    """Dynamically load model & optimizer state from checkpoint.

    NOTE: This function only works while loading checkpoints created by
    `save_model_and_optimizer`

    - Working for both DDP and non-DDP training.
    - Loads the checkpoint only on rank 0 when in DDP.
    - If using DDP, broadcasts the checkpoint to all other ranks.
    - Handles models both inside and outside of `DistributedDataParallel`.

    Args:
        filepath (str): Checkpoint filename.
        optimizer (torch.optim): Torch optimizer instance
        available_models (dict): Dictionary mapping class names to class references.
        device (torch.device): String or device specifier.

    """
    # Get rank if in DDP, else assume single process
    if dist.is_initialized():
        load_rank = dist.get_rank()
    else:
        load_rank = 0

    checkpoint = None

    if load_rank == 0:
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        epochIDX = checkpoint["epoch"]
        print(f"[Rank {load_rank}] Loaded checkpoint from epoch {epochIDX}")

    # If in DDP, broadcast checkpoint to all ranks
    if dist.is_initialized():
        checkpoint_list = [checkpoint]
        dist.broadcast_object_list(checkpoint_list, src=0)
        checkpoint = checkpoint_list[0]  # Unpack checkpoint on all ranks

    # Retrieve model class and arguments
    model_class_name = checkpoint["model_class"]
    model_args = checkpoint["model_args"]

    # Ensure model class exists
    if model_class_name not in available_models:
        raise ValueError(
            f"Unknown model class: {model_class_name}. Add it to `available_models`."
        )

    # Dynamically create the model
    model = available_models[model_class_name](**model_args)

    # Load state
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Move model to GPU if necessary
    model.to(device)

    # Synchronize all processes in DDP
    if dist.is_initialized():
        dist.barrier()

    return model, checkpoint["epoch"]
