#!/bin/bash

# This is a setup for GPU training on Chicoma. Find out how much
# memory per node, number of CPUs/node.

# NOTE: Number of CPUs per GPU must be an even number since there are
# 2 threads per core. If an odd number is requested the next higher
# even number gets used. 

# The following are one set of SBATCH options for the Chicoma GPU
# partition. There are optional other constraints. Namely, `gpu80` and
# `gpu40only`. 

#SBATCH --job-name=lscreward_s<studyIDX>_e0001
#SBATCH --account=w25_artimis_g
#SBATCH --nodes=<KNODES>
#SBATCH --ntasks-per-node=<NGPUS>
#SBATCH --gpus-per-node=<NGPUS>
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --time=1:20:00
#SBATCH --output=study<studyIDX>_epoch0001.out
#SBATCH --error=study<studyIDX>_epoch0001.err
#SBATCH -vvv

# Set the master node's address and port
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$(shuf -i 1024-65535 -n 1)  # Choose a random port
export MASTER_ADDR MASTER_PORT

# Check available GPUs
sinfo  -o "%P %.24G %N"
srun /usr/bin/echo $CUDA_AVAILABLE_DEVICES
nvidia-smi

# Specify NCCL communication
export NCCL_SOCKET_IFNAME=hsn0  # Check possible interfaces with `ip link show`

# Load correct conda environment
module load python/3.11-anaconda-2023.07
source activate
conda activate <YOKE_TORCH_ENV>

# Set number of threads per GPU
export OMP_NUM_THREADS=10

# Currently need to set an environment variable for MKL
# threading. Believed to be related to Numpy.
export MKL_SERVICE_FORCE_INTEL=1

# Get start time
export date00=`date`

# # Start GPU monitoring
# nvidia-smi --loop 5 --query-gpu=timestamp,gpu_bus_id,memory.used,memory.free,memory.total --format=csv > nvidia_smi_epoch0001.out &
# nvpid=$!

# Start the Code
srun python -u <train_script> @study<studyIDX>_START.input
#pypid=$!

# Kill GPU monitoring after python process finishes
#wait $pypid
#kill $nvpid

# Get end time and print to stdout
export date01=`date`

echo "===================TIME STARTED==================="
echo $date00
echo "===================TIME FINISHED==================="
echo $date01
