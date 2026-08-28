#!/bin/bash

#SBATCH --job-name=ddp_s<studyIDX>_e0001
#SBATCH --account=y26_artimis-test_g
#SBATCH --partition=standard
#SBATCH --time=04:00:00
#SBATCH --nodes=<KNODES>
#SBATCH --ntasks-per-node=<NGPUS>
#SBATCH --gpus-per-node=<NGPUS>
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=120G
#SBATCH --output=study<studyIDX>_epoch0001.out
#SBATCH --error=study<studyIDX>_epoch0001.err
#SBATCH -vvv

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$(shuf -i 1024-65535 -n 1)
export MASTER_ADDR
export MASTER_PORT

set -xv

sinfo -o "%P %.24G %N"
srun -vv --cpu-bind=verbose /usr/bin/echo $CUDA_AVAILABLE_DEVICES
nvidia-smi

export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3
export SLURM_CPU_BIND=verbose

source /usr/projects/artimis/mpmm/ryanw/Yoke/.venv/bin/activate

export OMP_NUM_THREADS=10
export date00=`date`

srun -vv --cpu-bind=verbose python -u <train_script> @study<studyIDX>_START.input

export date01=`date`

echo "===================TIME STARTED==================="
echo $date00
echo "===================TIME FINISHED==================="
echo $date01
