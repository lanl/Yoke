"""Contains functions for training, validating, and testing a pytorch model."""

####################################
# Packages
####################################
import os
from contextlib import nullcontext
import time
import h5py
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


def count_torch_params(model, trainable=True):
    """Count parameters in a pytorch model.

    Args:
        model (nn.Module): Model to count parameters for.
        trainable (bool): If TRUE, count only trainable parameters.

    """
    plist = []
    for p in model.parameters():
        if trainable:
            if p.requires_grad:
                plist.append(p.numel())
            else:
                pass
        else:
            plist.append(p.numel())

    return sum(plist)


######################################################
# Helper function for model/optimizer saving/loading
######################################################
def save_model_and_optimizer_hdf5(model, optimizer, epoch, filepath, compiled=False):
    """Saves the state of a model and optimizer in portable hdf5 format. Model and
    optimizer should be moved to the CPU prior to using this function.

    Args:
        model (torch model): Pytorch model to save
        optimizer (torch optimizer: Pytorch optimizer to save
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


def load_model_and_optimizer_hdf5(model, optimizer, filepath):
    """Loads state of model and optimizer stored in an hdf5 format.

    Args:
        model (torch model): Pytorch model to save
        optimizer (torch optimizer: Pytorch optimizer to save
        filepath (str): Where to save

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


####################################
# Make Dataloader form DataSet
####################################
def make_distributed_dataloader(
        dataset,
        batch_size,
        shuffle,
        num_workers,
        rank,
        world_size
    ) -> torch.utils.data.DataLoader:
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

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,  # Ensures uniform batch size
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )


def make_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 8,
    num_batches: int = 100,
    num_workers: int = 4,
    prefetch_factor: int = 2,
):
    """Function to create a pytorch dataloader from a pytorch dataset
    **https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader**
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
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=randomsampler,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
    )

    return dataloader


####################################
# Saving Results
####################################
def save_append_df(path: str, df: pd.DataFrame, START: bool):
    """Function to save/append dataframe contents to a csv file

    Args:
        path (str): path of csv file
        df (pd.DataFrame): pandas dataframe to save
        START (bool): indicates if the file path needs to be initiated

    Returns:
        No Return Objects

    """
    if START:
        assert not os.path.isfile(path), (
            "If starting training, " + path + " should not exist."
        )
        df.to_csv(path, header=True, index=True, mode="x")
    else:
        assert os.path.isfile(path), "If continuing training, " + path + " should exist."
        df.to_csv(path, header=False, index=True, mode="a")


def append_to_dict(dictt: dict, batch_ID: int, truth, pred, loss):
    """Function to appending sample information to a dictionary Dictionary must
    be initialized with correct keys

    Args:
        dictt (dict): dictionary to append sample information to
        batch_ID (int): batch ID number for samples
        truth (): array of truth values for batch of samples
        pred (): array of prediction values for batch of samples
        loss (): array of loss values for batch of samples

    Returns:
        dictt (dict): dictionary with appended sample information

    """
    batchsize = np.shape(truth.cpu().detach().numpy().flatten())[0]

    for i in range(batchsize):
        dictt["epoch"].append(0)  # To be easily identified later
        dictt["batch"].append(batch_ID)
        dictt["truth"].append(truth.cpu().detach().numpy().flatten()[i])
        dictt["prediction"].append(pred.cpu().detach().numpy().flatten()[i])
        dictt["loss"].append(loss.cpu().detach().numpy().flatten()[i])

    return dictt


####################################
# Continue Slurm Study
####################################
def continuation_setup(checkpointpath, studyIDX, last_epoch):
    """Function to generate the training.input and training.slurm files for
    continuation of model training

    Args:
         checkpointpath (str): path to model checkpoint to load in model from
         studyIDX (int): study ID to include in file name
         last_epoch (int): numer of epochs completed at this checkpoint

    Returns:
         new_training_slurm_filepath (str): Name of slurm file to submit job for
                                            continued training

    """
    # Identify Template Files
    training_input_tmpl = "./training_input.tmpl"
    training_slurm_tmpl = "./training_slurm.tmpl"

    # Make new training.input file
    with open(training_input_tmpl) as f:
        training_input_data = f.read()

    new_training_input_data = training_input_data.replace("<CHECKPOINT>", checkpointpath)

    input_str = "study{0:03d}_restart_training_epoch{1:04d}.input"
    new_training_input_filepath = input_str.format(studyIDX, last_epoch + 1)

    with open(os.path.join("./", new_training_input_filepath), "w") as f:
        f.write(new_training_input_data)

    with open(training_slurm_tmpl) as f:
        training_slurm_data = f.read()

    slurm_str = "study{0:03d}_restart_training_epoch{1:04d}.slurm"
    new_training_slurm_filepath = slurm_str.format(studyIDX, last_epoch + 1)

    new_training_slurm_data = training_slurm_data.replace(
        "<INPUTFILE>", new_training_input_filepath
    )

    new_training_slurm_data = new_training_slurm_data.replace(
        "<epochIDX>", f"{last_epoch + 1:04d}"
    )

    with open(os.path.join("./", new_training_slurm_filepath), "w") as f:
        f.write(new_training_slurm_data)

    return new_training_slurm_filepath


####################################
# Training on a Datastep
####################################
def train_scalar_datastep(data: tuple, model, optimizer, loss_fn, device: torch.device):
    """Function to complete a training step on a single sample in which the
    network's output is a scalar.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select

    Returns:
        loss (): evaluated loss for the data sample

    """
    # Set model to train
    model.train()

    # Extract data
    (inpt, truth) = data
    inpt = inpt.to(device, non_blocking=True)
    # Unsqueeze is necessary for scalar ground-truth output
    truth = truth.to(torch.float32).unsqueeze(-1).to(device, non_blocking=True)

    # Perform a forward pass
    # NOTE: If training on GPU model should have already been moved to GPU
    # prior to initalizing optimizer.
    pred = model(inpt)
    loss = loss_fn(pred, truth)

    # Perform backpropagation and update the weights
    optimizer.zero_grad(set_to_none=True)
    loss.mean().backward()
    optimizer.step()

    return truth, pred, loss


def train_array_datastep(data: tuple, model, optimizer, loss_fn, device: torch.device):
    """Function to complete a training step on a single sample in which the
    network's output is an array.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select

    Returns:
        loss (): evaluated loss for the data sample

    """
    # Set model to train
    model.train()

    # Extract data
    (inpt, truth) = data
    inpt = inpt.to(device, non_blocking=True)
    truth = truth.to(device, non_blocking=True)

    # Perform a forward pass
    # NOTE: If training on GPU model should have already been moved to GPU
    # prior to initalizing optimizer.
    pred = model(inpt)
    loss = loss_fn(pred, truth)

    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Shape: (batch_size,)

    # Perform backpropagation and update the weights
    # optimizer.zero_grad()
    optimizer.zero_grad(set_to_none=True)  # Possible speed-up
    loss.mean().backward()
    optimizer.step()

    return truth, pred, per_sample_loss


def train_loderunner_datastep(
        data: tuple,
        model,
        optimizer,
        loss_fn,
        device: torch.device,
        channel_map: list
        ):
    """A training step for which the data is of multi-input, multi-output type.

    This is currently a proto-type function to get the LodeRunner architecture
    training on a non-variable set of channels.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select

    Returns:
        loss (): evaluated loss for the data sample

    """
    # Set model to train
    model.train()

    # Extract data
    (start_img, end_img, Dt) = data

    start_img = start_img.to(device, non_blocking=True)
    Dt = Dt.to(torch.float32).to(device, non_blocking=True)
    end_img = end_img.to(device, non_blocking=True)

    # For our first LodeRunner training on the lsc240420 dataset the input and
    # output prediction variables are fixed.
    #
    # Both in_vars and out_vars correspond to indices for every variable in
    # this training setup...
    #
    # in_vars = ['density_case',
    #            'density_cushion',
    #            'density_maincharge',
    #            'density_outside_air',
    #            'density_striker',
    #            'density_throw',
    #            'Uvelocity',
    #            'Wvelocity']
    in_vars = torch.tensor(channel_map).to(device, non_blocking=True)
    out_vars = torch.tensor(channel_map).to(device, non_blocking=True)

    # Perform a forward pass
    # NOTE: If training on GPU model should have already been moved to GPU
    # prior to initalizing optimizer.
    pred_img = model(start_img, in_vars, out_vars, Dt)

    # Expecting to use a *reduction="none"* loss function so we can track loss
    # between individual samples. However, this will make the loss be computed
    # element-wise so we need to still average over the (channel, height,
    # width) dimensions to get the per-sample loss.
    loss = loss_fn(pred_img, end_img)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Shape: (batch_size,)

    # Perform backpropagation and update the weights
    optimizer.zero_grad(set_to_none=True)  # Possible speed-up
    loss.mean().backward()
    optimizer.step()

    # Delete created tensors to free memory
    del in_vars
    del out_vars

    # Clear GPU memory after each deallocation
    torch.cuda.empty_cache()

    return end_img, pred_img, per_sample_loss


def train_scheduled_loderunner_datastep(
    data: tuple, model, optimizer, loss_fn, device: torch.device, scheduled_prob: float
):
    """
    A training step for the LodeRunner architecture with scheduled sampling
    using a decayed scheduled_prob.

    Args:
        data (tuple): Sequence of images in (img_seq, Dt) tuple.
        model (loaded pytorch model): model to train.
        optimizer (torch.optim): optimizer for training set.
        loss_fn (torch.nn Loss Function): loss function for training set.
        device (torch.device): device index to select.
        scheduled_prob (float): Probability of using the ground truth as input.

    Returns:
        tuple: (end_img, pred_seq, per_sample_loss, updated_scheduled_prob)

    """
    # Set model to train
    model.train()

    # Extract data
    img_seq, Dt = data

    # [B, S, C, H, W] where S=seq-length
    img_seq = img_seq.to(device, non_blocking=True)
    # [B, 1]
    Dt = Dt.to(device, non_blocking=True)

    # Input and output variable indices
    in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device, non_blocking=True)
    out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device, non_blocking=True)

    # Storage for predictions at each timestep
    pred_seq = []

    # Unbind and iterate over slices in sequence-length dimension
    # NOTE: we exclude img_seq[:, :-1] since we don't have the next
    #   timestep to compare to.
    for k, k_img in enumerate(torch.unbind(img_seq[:, :-1], dim=1)):
        if k == 0:
            # Forward pass for the initial step
            pred_img = model(k_img, in_vars, out_vars, Dt)
        else:
            # Apply scheduled sampling
            if random.random() < scheduled_prob:
                current_input = k_img
            else:
                current_input = pred_img

            pred_img = model(current_input, in_vars, out_vars, Dt)

        # Store the prediction
        pred_seq.append(pred_img)

    # Combine predictions into a tensor of shape [B, SeqLength, C, H, W]
    pred_seq = torch.stack(pred_seq, dim=1)

    # Compute loss
    loss = loss_fn(pred_seq, img_seq[:, 1:])
    per_sample_loss = loss.mean(dim=[1, 2, 3, 4])  # Shape: (batch_size,)

    # Perform backpropagation and update the weights
    optimizer.zero_grad(set_to_none=True)  # Possible speed-up
    loss.mean().backward()
    optimizer.step()

    # Delete created tensors to free memory
    del in_vars
    del out_vars

    # Clear GPU memory after each deallocation
    torch.cuda.empty_cache()

    return img_seq[:, 1:], pred_seq, per_sample_loss


def train_DDP_loderunner_datastep(
    data: tuple,
    model,
    optimizer,
    loss_fn,
    device: torch.device,
    rank: int,
    world_size: int,
):
    """A DDP-compatible training step for multi-input, multi-output data.

        Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select
        rank (int): Rank of device
        world_size (int): Number of total DDP processes

    """
    # Set model to train mode
    model.train()

    # Extract data
    start_img, end_img, Dt = data
    start_img = start_img.to(device, non_blocking=True)
    Dt = Dt.to(device, non_blocking=True)
    end_img = end_img.to(device, non_blocking=True)

    # Fixed input and output variable indices
    in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device, non_blocking=True)
    out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device, non_blocking=True)

    # Forward pass
    pred_img = model(start_img, in_vars, out_vars, Dt)

    # Compute loss
    loss = loss_fn(pred_img, end_img)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Per-sample loss

    # Backward pass and optimization
    optimizer.zero_grad(set_to_none=True)
    loss.mean().backward()
    optimizer.step()

    # Gather per-sample losses from all processes
    gathered_losses = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_loss)

    # Rank 0 concatenates and saves or returns all losses
    if rank == 0:
        all_losses = torch.cat(gathered_losses, dim=0)  # Shape: (total_batch_size,)
    else:
        all_losses = None

    # Free memory
    del in_vars, out_vars
    torch.cuda.empty_cache()

    return end_img, pred_img, all_losses


def train_loderunner_fabric_datastep(
        fabric,
        data: tuple,
        model,
        optimizer,
        loss_fn):
    """A training step for which the data is of multi-input, multi-output type.

    This is currently a proto-type function to get the LodeRunner architecture
    training on a non-variable set of channels.

    Args:
        fabric (lightning.fabric.Fabric): Fabric instance.
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set

    Returns:
        loss (): evaluated loss for the data sample

    """
    # Set model to train
    model.train()

    # Distribute with fabric
    start_img, end_img, Dt = data

    # For our first LodeRunner training on the lsc240420 dataset the input and
    # output prediction variables are fixed.
    #
    # Both in_vars and out_vars correspond to indices for every variable in
    # this training setup...
    #
    # in_vars = ['density_case',
    #            'density_cushion',
    #            'density_maincharge',
    #            'density_outside_air',
    #            'density_striker',
    #            'density_throw',
    #            'Uvelocity',
    #            'Wvelocity']
    in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    in_vars = fabric.to_device(in_vars)
    out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    out_vars = fabric.to_device(out_vars)

    # Perform a forward pass
    # NOTE: If training on GPU model should have already been moved to GPU
    # prior to initalizing optimizer.
    pred_img = model(start_img, in_vars, out_vars, Dt)

    # Expecting to use a *reduction="none"* loss function so we can track loss
    # between individual samples. However, this will make the loss be computed
    # element-wise so we need to still average over the (channel, height,
    # width) dimensions to get the per-sample loss.
    loss = loss_fn(pred_img, end_img)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Shape: (batch_size,)

    # Perform backpropagation and update the weights
    optimizer.zero_grad(set_to_none=True)  # Possible speed-up
    fabric.backward(loss.mean())
    optimizer.step()

    # Gather per-sample loss across all processes
    global_per_sample_loss = fabric.all_gather(per_sample_loss)
    global_per_sample_loss = global_per_sample_loss.flatten()

    # Delete created tensors to free memory
    del in_vars
    del out_vars

    # Clear GPU memory after each deallocation
    torch.cuda.empty_cache()

    return end_img, pred_img, global_per_sample_loss


####################################
# Evaluating on a Datastep
####################################
def eval_scalar_datastep(data: tuple, model, loss_fn, device: torch.device):
    """Function to complete a validation step on a single sample for which the
    network output is a scalar.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model evaluate
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select

    Returns:
        loss (): evaluated loss for the data sample

    """
    # Set model to eval
    model.eval()

    # Extract data
    (inpt, truth) = data
    inpt = inpt.to(device, non_blocking=True)
    truth = truth.to(torch.float32).unsqueeze(-1).to(device, non_blocking=True)

    # Perform a forward pass
    pred = model(inpt)
    loss = loss_fn(pred, truth)

    return truth, pred, loss


def eval_array_datastep(data: tuple, model, loss_fn, device: torch.device):
    """Function to complete a validation step on a single sample in which network
    output is an array.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model evaluate
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select

    Returns:
        loss (): evaluated loss for the data sample

    """
    # Set model to eval
    model.eval()

    # Extract data
    (inpt, truth) = data
    inpt = inpt.to(device, non_blocking=True)
    truth = truth.to(device, non_blocking=True)

    # Perform a forward pass
    pred = model(inpt)
    loss = loss_fn(pred, truth)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Shape: (batch_size,)

    return truth, pred, per_sample_loss


def eval_loderunner_datastep(
        data: tuple,
        model,
        loss_fn,
        device: torch.device,
        channel_map: list):
    """An evaluation step for which the data is of multi-input, multi-output type.

    This is currently a proto-type function to get the LodeRunner architecture
    training on a non-variable set of channels.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select

    Returns:
        loss (): evaluated loss for the data sample

    """
    # Set model to train
    model.eval()

    # Extract data
    (start_img, end_img, Dt) = data
    start_img = start_img.to(device, non_blocking=True)
    Dt = Dt.to(device, non_blocking=True)

    end_img = end_img.to(device, non_blocking=True)

    # For our first LodeRunner training on the lsc240420 dataset the input and
    # output prediction variables are fixed.
    #
    # Both in_vars and out_vars correspond to indices for every variable in
    # this training setup...
    #
    # in_vars = ['density_case',
    #            'density_cushion',
    #            'density_maincharge',
    #            'density_outside_air',
    #            'density_striker',
    #            'density_throw',
    #            'Uvelocity',
    #            'Wvelocity']
    in_vars = torch.tensor(channel_map).to(device, non_blocking=True)
    out_vars = torch.tensor(channel_map).to(device, non_blocking=True)

    # Perform a forward pass
    # NOTE: If training on GPU model should have already been moved to GPU
    # prior to initalizing optimizer.
    pred_img = model(start_img, in_vars, out_vars, Dt)

    # Expecting to use a *reduction="none"* loss function so we can track loss
    # between individual samples. However, this will make the loss be computed
    # element-wise so we need to still average over the (channel, height,
    # width) dimensions to get the per-sample loss.
    loss = loss_fn(pred_img, end_img)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Shape: (batch_size,)

    # Delete created tensors to free memory
    del in_vars
    del out_vars

    # Clear GPU memory after each deallocation
    torch.cuda.empty_cache()

    return end_img, pred_img, per_sample_loss


def eval_scheduled_loderunner_datastep(
        data: tuple,
        model,
        optimizer,
        loss_fn,
        device: torch.device,
        scheduled_prob: float):
    """
    A training step for the LodeRunner architecture with scheduled sampling
    using a decayed scheduled_prob.

    Args:
        data (tuple): Sequence of images in (img_seq, Dt) tuple.
        model (loaded pytorch model): model to train.
        optimizer (torch.optim): optimizer for training set.
        loss_fn (torch.nn Loss Function): loss function for training set.
        device (torch.device): device index to select.
        scheduled_prob (float): Probability of using the ground truth as input.

    Returns:
        tuple: (end_img, pred_seq, per_sample_loss, updated_scheduled_prob)

    """
    # Set model to evaluation
    model.eval()

    # Extract data
    img_seq, Dt = data

    # [B, S, C, H, W] where S=seq-length
    img_seq = img_seq.to(device, non_blocking=True)
    # [B, 1]
    Dt = Dt.to(device, non_blocking=True)

    # Input and output variable indices
    in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device, non_blocking=True)
    out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device, non_blocking=True)

    # Storage for predictions at each timestep
    pred_seq = []

    # Unbind and iterate over slices in sequence-length dimension
    # NOTE: we exclude img_seq[:, :-1] since we don't have the next
    #   timestep to compare to.
    for k, k_img in enumerate(torch.unbind(img_seq[:, :-1], dim=1)):
        if k == 0:
            # Forward pass for the initial step
            pred_img = model(k_img, in_vars, out_vars, Dt)
        else:
            # Apply scheduled sampling
            if random.random() < scheduled_prob:
                current_input = k_img
            else:
                current_input = pred_img

            pred_img = model(current_input, in_vars, out_vars, Dt)

        # Store the prediction
        pred_seq.append(pred_img)

    # Combine predictions into a tensor of shape [B, SeqLength, C, H, W]
    pred_seq = torch.stack(pred_seq, dim=1)

    # Compute loss
    loss = loss_fn(pred_seq, img_seq[:, 1:])
    per_sample_loss = loss.mean(dim=[1, 2, 3, 4])  # Shape: (batch_size,)

    # Delete created tensors to free memory
    del in_vars
    del out_vars

    # Clear GPU memory after each deallocation
    torch.cuda.empty_cache()

    return img_seq[:, 1:], pred_seq, per_sample_loss


def eval_DDP_loderunner_datastep(
    data: tuple,
    model,
    loss_fn,
    device: torch.device,
    rank: int,
    world_size: int,
):
    """A DDP-compatible evaluation step.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select
        rank (int): Rank of device
        world_size (int): Total number of DDP processes

    """
    # Set model to evaluation mode
    model.eval()

    # Extract data
    start_img, end_img, Dt = data
    start_img = start_img.to(device, non_blocking=True)
    Dt = Dt.to(device, non_blocking=True)
    end_img = end_img.to(device, non_blocking=True)

    # Fixed input and output variable indices
    in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device, non_blocking=True)
    out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device, non_blocking=True)

    # Forward pass
    with torch.no_grad():
        pred_img = model(start_img, in_vars, out_vars, Dt)

    # Compute loss
    loss = loss_fn(pred_img, end_img)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Per-sample loss

    # Gather per-sample losses from all processes
    gathered_losses = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_loss)

    # Rank 0 concatenates and saves or returns all losses
    if rank == 0:
        all_losses = torch.cat(gathered_losses, dim=0)  # Shape: (total_batch_size,)
    else:
        all_losses = None

    # Free memory
    del in_vars, out_vars
    torch.cuda.empty_cache()

    return end_img, pred_img, all_losses


def eval_loderunner_fabric_datastep(
        fabric,
        data: tuple,
        model,
        loss_fn):
    """An evaluation step for which the data is of multi-input, multi-output type.

    This is currently a proto-type function to get the LodeRunner architecture
    training on a non-variable set of channels.

    Args:
        fabric (lightning.fabric.Fabric): Fabric instance.
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select

    Returns:
        loss (): evaluated loss for the data sample

    """
    # Set model to train
    model.eval()

    # Extract data
    start_img, end_img, Dt = data

    # For our first LodeRunner training on the lsc240420 dataset the input and
    # output prediction variables are fixed.
    #
    # Both in_vars and out_vars correspond to indices for every variable in
    # this training setup...
    #
    # in_vars = ['density_case',
    #            'density_cushion',
    #            'density_maincharge',
    #            'density_outside_air',
    #            'density_striker',
    #            'density_throw',
    #            'Uvelocity',
    #            'Wvelocity']
    in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    in_vars = fabric.to_device(in_vars)
    out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    out_vars = fabric.to_device(out_vars)

    # Perform a forward pass
    # NOTE: If training on GPU model should have already been moved to GPU
    # prior to initalizing optimizer.
    pred_img = model(start_img, in_vars, out_vars, Dt)

    # Expecting to use a *reduction="none"* loss function so we can track loss
    # between individual samples. However, this will make the loss be computed
    # element-wise so we need to still average over the (channel, height,
    # width) dimensions to get the per-sample loss.
    loss = loss_fn(pred_img, end_img)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Shape: (batch_size,)

    # Gather per-sample loss across all processes
    global_per_sample_loss = fabric.all_gather(per_sample_loss)
    global_per_sample_loss = global_per_sample_loss.flatten()

    # Delete created tensors to free memory
    del in_vars
    del out_vars

    # Clear GPU memory after each deallocation
    torch.cuda.empty_cache()

    return end_img, pred_img, global_per_sample_loss


######################################
# Training & Validation for an Epoch
######################################
def train_scalar_dict_epoch(
    training_data,
    validation_data,
    model,
    optimizer,
    loss_fn,
    summary_dict: dict,
    train_sample_dict: dict,
    val_sample_dict: dict,
    device: torch.device,
):
    """Function to complete a training step on a single sample for a network in
    which the output is a single scalar. Training, Validation, and Summary
    information are saved to dictionaries.

    Args:
        training_data (torch.dataloader): dataloader containing the training samples
        validation_data (torch.dataloader): dataloader containing the validation samples
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        summary_dict (dict): dictionary to save epoch stats to
        train_sample_dict (dict): dictionary to save training sample stats to
        val_sample_dict (dict): dictionary to save validation sample stats to
        device (torch.device): device index to select

    Returns:
        summary_dict (dict): dictionary with epoch stats
        train_sample_dict (dict): dictionary with training sample stats
        val_sample_dict (dict): dictionary with validation sample stats

    """
    # Initialize things to save
    startTime = time.time()
    trainbatches = len(training_data)
    valbatches = len(validation_data)
    trainbatch_ID = 0
    valbatch_ID = 0

    # Train on all training samples
    for traindata in training_data:
        trainbatch_ID += 1
        truth, pred, train_loss = train_scalar_datastep(
            traindata, model, optimizer, loss_fn, device
        )

        train_sample_dict = append_to_dict(
            train_sample_dict, trainbatch_ID, truth, pred, train_loss
        )

    train_batchsize = np.shape(truth.cpu().detach().numpy().flatten())[0]

    # Calcuate the Epoch Average Loss
    train_samples = train_batchsize * trainbatches
    avgTrainLoss = np.sum(train_sample_dict["loss"][-train_samples:]) / train_samples
    summary_dict["train_loss"].append(avgTrainLoss)

    # Evaluate on all validation samples
    with torch.no_grad():
        for valdata in validation_data:
            valbatch_ID += 1
            truth, pred, val_loss = eval_scalar_datastep(valdata, model, loss_fn, device)

            val_sample_dict = append_to_dict(
                val_sample_dict, valbatch_ID, truth, pred, val_loss
            )

    val_batchsize = np.shape(truth.cpu().detach().numpy().flatten())[0]

    # Calcuate the Epoch Average Loss
    val_samples = val_batchsize * valbatches
    avgValLoss = np.sum(val_sample_dict["loss"][-val_samples:]) / val_samples

    summary_dict["val_loss"].append(avgValLoss)

    # Calculate Time
    endTime = time.time()
    epoch_time = (endTime - startTime) / 60
    summary_dict["epoch_time"].append(epoch_time)

    return summary_dict, train_sample_dict, val_sample_dict


def train_scalar_csv_epoch(
    training_data,
    validation_data,
    model,
    optimizer,
    loss_fn,
    epochIDX,
    train_per_val,
    train_rcrd_filename: str,
    val_rcrd_filename: str,
    device: torch.device,
):
    """Function to complete a training epoch on a network which has a single scalar
    as output. Training and validation information is saved to successive CSV files.

    Args:
        training_data (torch.dataloader): dataloader containing the training samples
        validation_data (torch.dataloader): dataloader containing the validation samples
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        epochIDX (int): Index of current training epoch
        train_per_val (int): Number of Training epochs between each validation
        train_rcrd_filename (str): Name of CSV file to save training sample stats to
        val_rcrd_filename (str): Name of CSV file to save validation sample stats to
        device (torch.device): device index to select

    """
    # Initialize things to save
    trainbatches = len(training_data)
    valbatches = len(validation_data)
    trainbatch_ID = 0
    valbatch_ID = 0

    train_batchsize = training_data.batch_size
    val_batchsize = validation_data.batch_size

    train_rcrd_filename = train_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
    # Train on all training samples
    with open(train_rcrd_filename, "a") as train_rcrd_file:
        for traindata in training_data:
            trainbatch_ID += 1
            truth, pred, train_loss = train_scalar_datastep(
                traindata, model, optimizer, loss_fn, device
            )

            template = "{}, {}, {}"
            for i in range(train_batchsize):
                print(
                    template.format(
                        epochIDX,
                        trainbatch_ID,
                        train_loss.cpu().detach().numpy().flatten()[i],
                    ),
                    file=train_rcrd_file,
                )

    # Evaluate on all validation samples
    if epochIDX % train_per_val == 0:
        print("Validating...", epochIDX)
        val_rcrd_filename = val_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
        with open(val_rcrd_filename, "a") as val_rcrd_file:
            with torch.no_grad():
                for valdata in validation_data:
                    valbatch_ID += 1
                    truth, pred, val_loss = eval_scalar_datastep(
                        valdata, model, loss_fn, device
                    )

                    template = "{}, {}, {}"
                    for i in range(val_batchsize):
                        print(
                            template.format(
                                epochIDX,
                                valbatch_ID,
                                val_loss.cpu().detach().numpy().flatten()[i],
                            ),
                            file=val_rcrd_file,
                        )


def train_array_csv_epoch(
    training_data,
    validation_data,
    model,
    optimizer,
    loss_fn,
    epochIDX,
    train_per_val,
    train_rcrd_filename: str,
    val_rcrd_filename: str,
    device: torch.device,
):
    """Function to complete a training epoch on a network which has an array
    as output. Training and validation information is saved to successive CSV
    files.

    Args:
        training_data (torch.dataloader): dataloader containing the training samples
        validation_data (torch.dataloader): dataloader containing the validation samples
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        epochIDX (int): Index of current training epoch
        train_per_val (int): Number of Training epochs between each validation
        train_rcrd_filename (str): Name of CSV file to save training sample stats to
        val_rcrd_filename (str): Name of CSV file to save validation sample stats to
        device (torch.device): device index to select

    """
    # Initialize things to save
    trainbatches = len(training_data)
    valbatches = len(validation_data)
    trainbatch_ID = 0
    valbatch_ID = 0

    train_batchsize = training_data.batch_size
    val_batchsize = validation_data.batch_size

    train_rcrd_filename = train_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
    # Train on all training samples
    with open(train_rcrd_filename, "a") as train_rcrd_file:
        for traindata in training_data:
            trainbatch_ID += 1
            truth, pred, train_losses = train_array_datastep(
                traindata, model, optimizer, loss_fn, device
            )

            # Save batch records to the training record file
            batch_records = np.column_stack([
                np.full(len(train_losses), epochIDX),
                np.full(len(train_losses), trainbatch_ID),
                train_losses.detach().cpu().numpy().flatten()
            ])
            np.savetxt(train_rcrd_file, batch_records, fmt="%d, %d, %.8f")

    # Evaluate on all validation samples
    if epochIDX % train_per_val == 0:
        print("Validating...", epochIDX)
        val_rcrd_filename = val_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
        with open(val_rcrd_filename, "a") as val_rcrd_file:
            with torch.no_grad():
                for valdata in validation_data:
                    valbatch_ID += 1
                    truth, pred, val_losses = eval_array_datastep(
                        valdata, model, loss_fn, device
                    )

                    # Save validation batch records
                    batch_records = np.column_stack([
                        np.full(len(val_losses), epochIDX),
                        np.full(len(val_losses), valbatch_ID),
                        val_losses.detach().cpu().numpy().flatten()
                    ])
                    np.savetxt(val_rcrd_file, batch_records, fmt="%d, %d, %.8f")

    return


def train_simple_loderunner_epoch(
    channel_map: list,
    training_data,
    validation_data,
    model,
    optimizer,
    loss_fn,
    epochIDX,
    train_per_val,
    train_rcrd_filename: str,
    val_rcrd_filename: str,
    device: torch.device,
    verbose: bool=False,
):
    """Function to complete a training epoch on the LodeRunner architecture with
    fixed channels in the input and output. Training and validation information
    is saved to successive CSV files.

    Args:
        training_data (torch.dataloader): dataloader containing the training samples
        validation_data (torch.dataloader): dataloader containing the validation samples
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        epochIDX (int): Index of current training epoch
        train_per_val (int): Number of Training epochs between each validation
        train_rcrd_filename (str): Name of CSV file to save training sample stats to
        val_rcrd_filename (str): Name of CSV file to save validation sample stats to
        device (torch.device): device index to select
        verbose (boolean): Flag to print diagnostic output.

    """
    # Initialize things to save
    trainbatch_ID = 0
    valbatch_ID = 0

    train_batchsize = training_data.batch_size
    val_batchsize = validation_data.batch_size

    train_rcrd_filename = train_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
    # Train on all training samples
    with open(train_rcrd_filename, "a") as train_rcrd_file:
        for traindata in training_data:
            trainbatch_ID += 1

            # Time each epoch and print to stdout
            if verbose:
                startTime = time.time()

            truth, pred, train_loss = train_loderunner_datastep(
                traindata, model, optimizer, loss_fn, device, channel_map
            )

            if verbose:
                endTime = time.time()
                batch_time = endTime - startTime
                print(f"Batch {trainbatch_ID} time (seconds): {batch_time:.5f}",
                      flush=True)

            if verbose:
                startTime = time.time()

            # Stack loss record and write using numpy
            batch_records = np.column_stack([
                np.full(train_batchsize, epochIDX),
                np.full(train_batchsize, trainbatch_ID),
                train_loss.detach().cpu().numpy().flatten()
            ])

            np.savetxt(train_rcrd_file, batch_records, fmt="%d, %d, %.8f")

            if verbose:
                endTime = time.time()
                record_time = endTime - startTime
                print(f"Batch {trainbatch_ID} record time: {record_time:.5f}",
                      flush=True)

            # Explictly delete produced tensors to free memory
            del truth
            del pred
            del train_loss

            # Clear GPU memory after each batch
            torch.cuda.empty_cache()

    # Evaluate on all validation samples
    if epochIDX % train_per_val == 0:
        print("Validating...", epochIDX)
        val_rcrd_filename = val_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
        with open(val_rcrd_filename, "a") as val_rcrd_file:
            with torch.no_grad():
                for valdata in validation_data:
                    valbatch_ID += 1
                    truth, pred, val_loss = eval_loderunner_datastep(
                        valdata, model, loss_fn, device, channel_map
                    )

                    # Stack loss record and write using numpy
                    batch_records = np.column_stack([
                        np.full(val_batchsize, epochIDX),
                        np.full(val_batchsize, valbatch_ID),
                        val_loss.detach().cpu().numpy().flatten()
                    ])

                    np.savetxt(val_rcrd_file, batch_records, fmt="%d, %d, %.8f")

                    # Explictly delete produced tensors to free memory
                    del truth
                    del pred
                    del val_loss

                    # Clear GPU memory after each batch
                    torch.cuda.empty_cache()


def train_scheduled_loderunner_epoch(
    training_data,
    validation_data,
    model,
    optimizer,
    LRsched,
    loss_fn,
    epochIDX,
    train_per_val,
    train_rcrd_filename: str,
    val_rcrd_filename: str,
    device: torch.device,
    scheduled_prob: float,
):
    """
    Function to complete a training epoch on the LodeRunner architecture using
    scheduled sampling. Updates the scheduled probability over time.

    Args:
        training_data (torch.dataloader): dataloader containing the training samples.
        validation_data (torch.dataloader): dataloader containing the validation samples.
        model (loaded pytorch model): model to train.
        optimizer (torch.optim): optimizer for training set.
        LRsched (torch.optim.lr_scheduler): Learning-rate scheduler that will be called
                                            every training step.
        loss_fn (torch.nn Loss Function): loss function for training set.
        epochIDX (int): Index of current training epoch.
        train_per_val (int): Number of training epochs between each validation.
        train_rcrd_filename (str): Name of CSV file to save training sample stats to.
        val_rcrd_filename (str): Name of CSV file to save validation sample stats to.
        device (torch.device): device index to select.
        scheduled_prob (float): Initial probability of using ground truth as input.

    """
    # Initialize variables for tracking batches
    trainbatch_ID = 0
    valbatch_ID = 0

    train_rcrd_filename = train_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")

    # Train on all training samples
    with open(train_rcrd_filename, "a") as train_rcrd_file:
        for traindata in training_data:
            trainbatch_ID += 1

            # Training step with scheduled sampling
            true_seq, pred_seq, train_losses = train_scheduled_loderunner_datastep(
                data=traindata,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                scheduled_prob=scheduled_prob
            )

            # Increment the learning-rate scheduler
            LRsched.step()

            # Save batch records to the training record file
            batch_records = np.column_stack([
                np.full(len(train_losses), epochIDX),
                np.full(len(train_losses), trainbatch_ID),
                train_losses.detach().cpu().numpy().flatten()
            ])
            np.savetxt(train_rcrd_file, batch_records, fmt="%d, %d, %.8f")

            # Clear memory
            del true_seq, pred_seq, train_losses
            torch.cuda.empty_cache()

    # Evaluate on all validation samples
    if epochIDX % train_per_val == 0:
        print("Validating...", epochIDX)
        val_rcrd_filename = val_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
        with open(val_rcrd_filename, "a") as val_rcrd_file:
            with torch.no_grad():
                for valdata in validation_data:
                    valbatch_ID += 1
                    truth, pred, val_losses = eval_scheduled_loderunner_datastep(
                        data=valdata,
                        model=model,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        device=device,
                        scheduled_prob=scheduled_prob
                    )

                    # Save validation batch records
                    batch_records = np.column_stack([
                        np.full(len(val_losses), epochIDX),
                        np.full(len(val_losses), valbatch_ID),
                        val_losses.detach().cpu().numpy().flatten()
                    ])
                    np.savetxt(val_rcrd_file, batch_records, fmt="%d, %d, %.8f")

                    # Clear memory
                    del truth, pred, val_losses
                    torch.cuda.empty_cache()

    # Return the updated scheduled probability
    return scheduled_prob


def train_LRsched_loderunner_epoch(
    channel_map: list,
    training_data,
    validation_data,
    model,
    optimizer,
    loss_fn,
    LRsched,
    epochIDX,
    train_per_val,
    train_rcrd_filename: str,
    val_rcrd_filename: str,
    device: torch.device,
    verbose: bool=False,
):
    """Function to complete a training epoch on the LodeRunner architecture with
    fixed channels in the input and output. Training and validation information
    is saved to successive CSV files.

    Args:
        training_data (torch.dataloader): dataloader containing the training samples
        validation_data (torch.dataloader): dataloader containing the validation samples
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        LRsched (torch.optim.lr_scheduler): Learning-rate scheduler that will be called
                                            every training step.
        epochIDX (int): Index of current training epoch
        train_per_val (int): Number of Training epochs between each validation
        train_rcrd_filename (str): Name of CSV file to save training sample stats to
        val_rcrd_filename (str): Name of CSV file to save validation sample stats to
        device (torch.device): device index to select
        verbose (boolean): Flag to print diagnostic output.

    """
    # Initialize things to save
    trainbatch_ID = 0
    valbatch_ID = 0

    train_batchsize = training_data.batch_size
    val_batchsize = validation_data.batch_size

    train_rcrd_filename = train_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
    # Train on all training samples
    with open(train_rcrd_filename, "a") as train_rcrd_file:
        for traindata in training_data:
            trainbatch_ID += 1

            # Time each epoch and print to stdout
            if verbose:
                startTime = time.time()

            truth, pred, train_loss = train_loderunner_datastep(
                traindata, model, optimizer, loss_fn, device, channel_map
            )

            # Increment the learning-rate scheduler
            LRsched.step()

            if verbose:
                endTime = time.time()
                batch_time = endTime - startTime
                print(f"Batch {trainbatch_ID} time (seconds): {batch_time:.5f}",
                      flush=True)

            if verbose:
                startTime = time.time()

            # Stack loss record and write using numpy
            batch_records = np.column_stack([
                np.full(train_batchsize, epochIDX),
                np.full(train_batchsize, trainbatch_ID),
                train_loss.detach().cpu().numpy().flatten()
            ])

            np.savetxt(train_rcrd_file, batch_records, fmt="%d, %d, %.8f")

            if verbose:
                endTime = time.time()
                record_time = endTime - startTime
                print(f"Batch {trainbatch_ID} record time: {record_time:.5f}",
                      flush=True)

            # Explictly delete produced tensors to free memory
            del truth
            del pred
            del train_loss

            # Clear GPU memory after each batch
            torch.cuda.empty_cache()

    # Evaluate on all validation samples
    if epochIDX % train_per_val == 0:
        print("Validating...", epochIDX)
        val_rcrd_filename = val_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
        with open(val_rcrd_filename, "a") as val_rcrd_file:
            with torch.no_grad():
                for valdata in validation_data:
                    valbatch_ID += 1
                    truth, pred, val_loss = eval_loderunner_datastep(
                        valdata, model, loss_fn, device, channel_map
                    )

                    # Stack loss record and write using numpy
                    batch_records = np.column_stack([
                        np.full(val_batchsize, epochIDX),
                        np.full(val_batchsize, valbatch_ID),
                        val_loss.detach().cpu().numpy().flatten()
                    ])

                    np.savetxt(val_rcrd_file, batch_records, fmt="%d, %d, %.8f")

                    # Explictly delete produced tensors to free memory
                    del truth
                    del pred
                    del val_loss

                    # Clear GPU memory after each batch
                    torch.cuda.empty_cache()


def train_DDP_loderunner_epoch(
    training_data,
    validation_data,
    num_train_batches,
    num_val_batches,
    model,
    optimizer,
    loss_fn,
    LRsched,
    epochIDX,
    train_per_val,
    train_rcrd_filename: str,
    val_rcrd_filename: str,
    device: torch.device,
    rank: int,
    world_size: int,
):
    """Function to complete a training epoch on the LodeRunner architecture with
    fixed channels in the input and output. Training and validation information
    is saved to successive CSV files.

    Args:
        training_data (torch.dataloader): dataloader containing the training samples
        validation_data (torch.dataloader): dataloader containing the validation samples
        num_train_batches (int): Number of batches in training epoch
        num_val_batches (int): Number of batches in validation epoch
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        LRsched (torch.optim.lr_scheduler): Learning-rate scheduler that will be called
                                            every training step.
        epochIDX (int): Index of current training epoch
        train_per_val (int): Number of Training epochs between each validation
        train_rcrd_filename (str): Name of CSV file to save training sample stats to
        val_rcrd_filename (str): Name of CSV file to save validation sample stats to
        device (torch.device): device index to select
        rank (int): rank of process
        world_size (int): number of total processes

    """
    # Initialize things to save
    trainbatch_ID = 0
    valbatch_ID = 0

    # Training loop
    model.train()
    train_rcrd_filename = train_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
    with open(train_rcrd_filename, "a") if rank == 0 else nullcontext() as train_rcrd_file:
        for trainbatch_ID, traindata in enumerate(training_data):
            # Stop when number of training batches is reached
            if trainbatch_ID >= num_train_batches:
                break

            # Perform a single training step
            truth, pred, train_losses = train_DDP_loderunner_datastep(
                traindata, model, optimizer, loss_fn, device, rank, world_size
            )

            # Increment the learning-rate scheduler
            LRsched.step()

            # Save training record (rank 0 only)
            if rank == 0:
                batch_records = np.column_stack([
                    np.full(len(train_losses), epochIDX),
                    np.full(len(train_losses), trainbatch_ID),
                    train_losses.cpu().numpy().flatten()
                ])
                np.savetxt(train_rcrd_file, batch_records, fmt="%d, %d, %.8f")

            # Free memory
            del truth, pred, train_losses
            torch.cuda.empty_cache()

    # Validation loop
    if epochIDX % train_per_val == 0:
        print("Validating...", epochIDX)
        val_rcrd_filename = val_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
        model.eval()
        with open(val_rcrd_filename, "a") if rank == 0 else nullcontext() as val_rcrd_file:
            with torch.no_grad():
                for valbatch_ID, valdata in enumerate(validation_data):
                    # Stop when number of training batches is reached
                    if valbatch_ID >= num_val_batches:
                        break

                    end_img, pred_img, val_losses = eval_DDP_loderunner_datastep(
                        valdata, model, loss_fn, device, rank, world_size,
                    )

                    # Save validation record (rank 0 only)
                    if rank == 0:
                        batch_records = np.column_stack([
                            np.full(len(val_losses), epochIDX),
                            np.full(len(val_losses), valbatch_ID),
                            val_losses.cpu().numpy().flatten()
                        ])
                        np.savetxt(val_rcrd_file, batch_records, fmt="%d, %d, %.8f")

                    # Free memory
                    del end_img, pred_img, val_losses
                    torch.cuda.empty_cache()


def train_fabric_loderunner_epoch(
    fabric,
    training_data,
    validation_data,
    model,
    optimizer,
    loss_fn,
    LRsched,
    epochIDX,
    train_per_val,
    train_rcrd_filename: str,
    val_rcrd_filename: str,
):
    """Function to complete a training epoch on the LodeRunner architecture with
    fixed channels in the input and output. Training and validation information
    is saved to successive CSV files.

    Args:
        fabric (lightning.fabric.Fabric): Fabric instance to take care of distributed
                                          data-parallel training
        training_data (torch.dataloader): dataloader containing the training samples
        validation_data (torch.dataloader): dataloader containing the validation samples
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        LRsched (torch.optim.lr_scheduler): Learning-rate scheduler that will be called
                                            every training step.
        epochIDX (int): Index of current training epoch
        train_per_val (int): Number of Training epochs between each validation
        train_rcrd_filename (str): Name of CSV file to save training sample stats to
        val_rcrd_filename (str): Name of CSV file to save validation sample stats to

    """
    # Initialize things to save
    trainbatch_ID = 0
    valbatch_ID = 0

    train_rcrd_filename = train_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
    # Train on all training samples
    with open(train_rcrd_filename, "a") as train_rcrd_file:
        for traindata in training_data:
            trainbatch_ID += 1

            truth, pred, train_losses = train_loderunner_fabric_datastep(
                fabric, traindata, model, optimizer, loss_fn,
            )

            # Increment the learning-rate scheduler
            LRsched.step()

            if fabric.global_rank == 0:
                # Stack loss record and write using numpy
                batch_records = np.column_stack([
                    np.full(len(train_losses), epochIDX),
                    np.full(len(train_losses), trainbatch_ID),
                    train_losses.detach().cpu().numpy().flatten()
                ])
                np.savetxt(train_rcrd_file, batch_records, fmt="%d, %d, %.8f")

            # Explictly delete produced tensors to free memory
            del truth
            del pred
            del train_losses

            # Clear GPU memory after each batch
            torch.cuda.empty_cache()

    # Evaluate on all validation samples
    if epochIDX % train_per_val == 0:
        print("Validating...", epochIDX)
        val_rcrd_filename = val_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
        with open(val_rcrd_filename, "a") as val_rcrd_file:
            with torch.no_grad():
                for valdata in validation_data:
                    valbatch_ID += 1
                    truth, pred, val_losses = eval_loderunner_fabric_datastep(
                        fabric, valdata, model, loss_fn,
                    )

                    if fabric.global_rank == 0:
                        # Stack loss record and write using numpy
                        batch_records = np.column_stack([
                            np.full(len(val_losses), epochIDX),
                            np.full(len(val_losses), valbatch_ID),
                            val_losses.detach().cpu().numpy().flatten()
                        ])
                        np.savetxt(val_rcrd_file, batch_records, fmt="%d, %d, %.8f")

                    # Explictly delete produced tensors to free memory
                    del truth
                    del pred
                    del val_losses

                    # Clear GPU memory after each batch
                    torch.cuda.empty_cache()
