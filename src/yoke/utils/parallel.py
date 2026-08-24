"""Yoke module to assist GPU-parallel training.

Some models within Yoke require specific modifications to PyTorch multi-GPU
training utilities.

"""

import os

import torch
import torch.nn as nn
import torch.distributed as dist


def setup_distributed() -> tuple[int, int, int, torch.device]:
    """Initialize the DDP process group from Slurm environment variables.

    Relies on the Slurm-provided variables SLURM_PROCID, SLURM_NTASKS, and
    SLURM_LOCALID, along with MASTER_ADDR and MASTER_PORT, to set up an NCCL
    process group and select this process's GPU.

    Returns:
        rank (int): Global rank of this process.
        world_size (int): Total number of processes.
        local_rank (int): Local rank (GPU index) on this node.
        device (torch.device): CUDA device for this process.
    """
    # ----- 1) Basic setup & environment variables -----
    # Rely on Slurm variables: SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID, etc.
    rank = int(os.environ["SLURM_PROCID"])  # global rank
    world_size = int(os.environ["SLURM_NTASKS"])  # total number of processes
    local_rank = int(os.environ["SLURM_LOCALID"])  # local rank (GPU index on node)

    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    # ----- 2) Set the current GPU device for this process -----
    # Map local rank onto the GPUs this process can actually see. When Slurm
    # binds one GPU per task (cgroup isolation), each rank sees a single device
    # renumbered to 0, so device_count() == 1 and local_rank must fold to 0.
    # When every rank sees all node GPUs, device_count() == NGPUS and the modulo
    # is a no-op. Guards against "invalid device ordinal" under per-task binding.
    n_visible = torch.cuda.device_count()
    if n_visible > 0:
        local_rank = local_rank % n_visible
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
    """Destroy the DDP process group."""
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
