"""Yoke module to assist GPU-parallel training.

Some models within Yoke require specific modifications to PyTorch multi-GPU
training utilities.

"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist


def setup_distributed() -> tuple[int, int, int, torch.device]:
    """Setup distributed training environment using SLURM environment variables.

    Required Environment Variables:
        SLURM_PROCID: Global rank of this process across all nodes
        SLURM_NTASKS: Total number of processes (world size)
        SLURM_LOCALID: Local rank on this node (GPU index)
        MASTER_ADDR: Address of the master node for communication
        MASTER_PORT: Port for distributed communication

    Returns:
        tuple: A 4-tuple containing:
            - rank (int): Global rank of this process (0 to world_size-1)
            - world_size (int): Total number of processes across all nodes
            - local_rank (int): Local rank on this node (GPU index)
            - device (torch.device): CUDA device object for this process
    """
    # ----- 1) Basic setup & environment variables -----
    rank = int(os.environ["SLURM_PROCID"])  # global rank
    world_size = int(os.environ["SLURM_NTASKS"])  # total number of processes
    local_rank = int(os.environ["SLURM_LOCALID"])  # local rank (GPU index on this node)

    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    print("============================", flush=True)
    print(f"[Rank {rank}] DDP setup, master_addr: {master_addr}", flush=True)
    print(f"[Rank {rank}] DDP setup, master_port: {master_port}", flush=True)
    print(f"[Rank {rank}] DDP setup, rank: {rank}", flush=True)
    print(f"[Rank {rank}] DDP setup, local_rank: {local_rank}", flush=True)
    print(f"[Rank {rank}] DDP setup, world_size: {world_size}", flush=True)
    print("============================", flush=True)

    # ----- 2) Set the current GPU device for this process -----
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # ----- 3) Initialize the process group -----
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )

    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    """Clean up distributed training environment.

    This function destroys the PyTorch distributed process group that was
    initialized by setup_distributed().
    """
    dist.destroy_process_group()

# Custom nn.DataParallel class to handle input to LodeRunner that should not be
# split by batch.
class LodeRunner_DataParallel(nn.DataParallel):
    """Handle unique GPU splitting of LodeRunner inputs.

    Since LodeRunner's *forward* method has multiple inputs consisting of
    several different shapes, some of which include a batch dimension and some
    of which do not, we must handle the splitting of data across multiple GPUs
    explicitly.

    """

    def __init__(self, model: nn.Module) -> None:
        """Get it initialized using parent."""
        super().__init__(model)

    def forward(self, *inputs: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Handle explicit GPU splitting."""
        # Input is (start_img, in_vars, out_vars, Dt)
        image_input = inputs[0]
        in_vars = inputs[1]
        out_vars = inputs[2]
        Dt_input = inputs[3]

        # Split batchsize-dependent inputs and replicate fixed inputs
        if self.device_ids:
            # Copy model to device
            replicas = self.replicate(self.module, self.device_ids)

            # Split batchsize-dependent inputs
            inputs_split = nn.parallel.scatter((image_input, Dt_input), self.device_ids)

            # Replicate non-batchsize-dependent inputs
            in_vars_replicas = [in_vars.to(device) for device in self.device_ids]

            out_vars_replicas = [out_vars.to(device) for device in self.device_ids]

            # Combine splits and replicas
            inputs_combined = [
                (split_inputs[0], in_vars, out_vars, split_inputs[1])
                for split_inputs, in_vars, out_vars in zip(
                    inputs_split, in_vars_replicas, out_vars_replicas
                )
            ]

            # Forward pass with replicas and custom splits
            outputs = nn.parallel.parallel_apply(replicas, inputs_combined)

            return nn.parallel.gather(outputs, self.output_device)
        else:
            return self.module(*inputs, **kwargs)
