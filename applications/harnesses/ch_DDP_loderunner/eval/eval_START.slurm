#!/bin/bash

# This is a setup for GPU training on Venado. Find out how much
# memory per node, number of CPUs/node.

# NOTE: Number of CPUs per GPU must be an even number since there are
# 2 threads per core. If an odd number is requested the next higher
# even number gets used. 

# The following are one set of SBATCH options for the Venado GPU
# partition. There are optional other constraints.

#SBATCH --job-name=ddp_eval
#SBATCH --account=y26_artimis-fmod_g
#PRODUCTION  #SBATCH --partition=gpu
#PRODUCTION  #SBATCH --time=16:00:00
#SBATCH --time=1:00:00
#SBATCH --qos=debug
#SBATCH --reservation=gpu_debug
#SBATCH --partition=gpu_debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=eval_test.out
#SBATCH --error=eval_test.err
#SBATCH -vvv


# # Set the master node's address and port
# MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# MASTER_PORT=$(shuf -i 1024-65535 -n 1)  # Choose a random port
# export MASTER_ADDR MASTER_PORT

# # Check available GPUs
# sinfo  -o "%P %.24G %N"
# srun /usr/bin/echo $CUDA_AVAILABLE_DEVICES
# nvidia-smi

# # Specify NCCL communication
# export NCCL_SOCKET_IFNAME=hsn0  # Check possible interfaces with `ip link show`

# Debugging distributed data parallel
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT

# Load correct environment
source $HOME/yoke_260202/bin/activate

# Set number of threads per GPU
#export OMP_NUM_THREADS=10

# Get start time
export date00=`date`

# Start the Code
python eval_LodeRunner.py @eval_START.input

# Get end time and print to stdout
export date01=`date`

echo "===================TIME STARTED==================="
echo $date00
echo "===================TIME FINISHED==================="
echo $date01
