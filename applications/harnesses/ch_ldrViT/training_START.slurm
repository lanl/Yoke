#!/bin/bash

#SBATCH --job-name=ldr2frm_s<studyIDX>_e0001
#SBATCH --account=<allocation>
#PRODUCTION  
# #SBATCH --partition=gpu
# #SBATCH --time=08:00:00
#DEBUG  
#SBATCH --time=2:00:00
#SBATCH --qos=debug
#SBATCH --reservation=gpu_debug
#SBATCH --partition=gpu_debug
#SBATCH --nodes=<KNODES>
#SBATCH --ntasks-per-node=<NGPUS>
#SBATCH --gpus-per-node=<NGPUS>
#SBATCH --cpus-per-task=16
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

# Debugging distributed data parallel
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT

# Load correct python environment
source $HOME/yoke_260202/bin/activate

# Set number of threads per GPU
export OMP_NUM_THREADS=10

# Get start time
export date00=`date`

# Start the Code
# Explicitly set TCP environment for the following...
srun python -u <train_script> @study<studyIDX>_START.input

# Get end time and print to stdout
export date01=`date`

echo "===================TIME STARTED==================="
echo $date00
echo "===================TIME FINISHED==================="
echo $date01
