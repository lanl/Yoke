#!/bin/bash

# NOTE: Number of CPUs per GPU must be an even number since there are
# 2 threads per core. If an odd number is requested the next higher
# even number gets used. Also, there are 120Gb of memory on the
# node. However, if 30Gb per-job are requested for 4 jobs there are
# not enough resources. You may request up to 29G per-job.

#SBATCH --job-name=lsc_tCNN2_study<studyIDX>
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-gpu=8
#SBATCH --gpu-bind=closest
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem-bind=local
#SBATCH --mem=29G
#SBATCH --time=10:00:00
#SBATCH --output=study<studyIDX>_epoch0001.out
#SBATCH --error=study<studyIDX>_epoch0001.err
#SBATCH -vvv

# Check available GPUs
sinfo  -o "%P %.24G %N"
srun /usr/bin/echo $CUDA_AVAILABLE_DEVICES

# Load correct conda environment
source <YOKE_CONDA_BASE>/bin/activate <YOKE_TORCH_ENV>

# Get start time
export date00=`date`

# Start the Code
python <train_script> @study<studyIDX>_START.input

# Get end time and print to stdout
export date01=`date`

echo "===================TIME STARTED==================="
echo $date00
echo "===================TIME FINISHED==================="
echo $date01
