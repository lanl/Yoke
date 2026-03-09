#!/bin/bash

# This is a setup for GPU training on RZadams.

# The following are one set of Flux options for the RZadams partitions.

# RZadams has 96 CPUs/node and 4 GPUs/node

#flux: --job-name=lscpolicy_s<studyIDX>_e0001
#flux: --time=2h
#flux: --queue=plarge
#flux: --nodes=1
#flux: -n 4  # total number of tasks across all nodes.
#flux: -g 1  # --gpus-per-task=1
#flux: -c 24  # --cores-per-task=24
#flux: --output=lscpolicy_s<studyIDX>_e0001.out
#flux: --error=lscpolicy_s<studyIDX>_e0001.err


# ——————————————————————————————————————————————————————————————
# 1) Set up MASTER_ADDR / MASTER_PORT for torch.distributed (env://)
# ——————————————————————————————————————————————————————————————
# Grab the list of hosts from Flux, take the first as master
HOSTLIST=$(flux hostlist instance)             # full comma-sep list of nodes
MASTER_ADDR=${HOSTLIST%%,*}                    # text before first comma
MASTER_PORT=$(shuf -i 10000-65535 -n 1)        # random open port

export MASTER_ADDR MASTER_PORT
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"

# ——————————————————————————————————————————————————————————————
# 2) Show GPU topology
# ——————————————————————————————————————————————————————————————
# What GPUs do we have?
flux --parent resource list

# ——————————————————————————————————————————————————————————————
# 3) NCCL (ROCm-NCCL) / OpenMP tuning
# ——————————————————————————————————————————————————————————————
# Check possible interfaces (IFNAME) with `ip link show`? 
export NCCL_SOCKET_IFNAME=hsi0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export OMP_NUM_THREADS=10

# Load correct conda environment
export ROCMVERSION=6.2.4
module load rocm/$ROCMVERSION
module load cray-python/3.11.7
source /g/g12/hickmann/yoke_250415/bin/activate
export adams_python=/g/g12/hickmann/yoke_250415/bin/python

# AMD’s equivalent of nvidia-smi
rocm-smi

# Get start time
export date00=`date`

# Start the Code
# Explicitly set TCP environment for the following...
# flux run python -c "import torch; x = torch.rand(3, 3); print(x)"
# flux run python -c "import torch; print(torch.cuda.is_available())"

flux run $adams_python -u <train_script> @study<studyIDX>_START.input

# Get end time and print to stdout
export date01=`date`

echo "===================TIME STARTED==================="
echo $date00
echo "===================TIME FINISHED==================="
echo $date01
