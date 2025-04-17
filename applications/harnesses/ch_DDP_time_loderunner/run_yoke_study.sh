#!/bin/bash

# ----------------------------------------------------------------------
# How to run:
#   ./run_yoke_study.sh
#
# ⚠️ WARNING: This script must be run with the Bash shell.
# ----------------------------------------------------------------------

# ⚠️ IMPORTANT:
# Before running this script, type:
#     newgrp artimis
# to make sure files are group-owned by 'artimis'
# Otherwise, they will default to your personal group and won't be accessible to teammates.

# --- Warn if not running under Bash ---
if [ -z "$BASH_VERSION" ]; then
    echo "❌ This script must be run with bash."
    echo "💡 Tip: Run 'bash' to start a bash shell, then re-run this script."
    exit 1
fi

# --- Require active group to be 'artimis' ---
current_group=$(id -gn)
if [[ "$current_group" != "artimis" ]]; then
    echo "❌ Your active group is '$current_group', not 'artimis'."
    echo "💡 Please run: newgrp artimis"
    echo "Then re-run this script."
    exit 1
fi

# === Set file permissions ===
# Ensure files are created as rw-rw---- for the 'artimis' group
umask 007

# === Load module and activate environment ===
module load python/3.11-anaconda-2023.07
source activate
conda activate torch_ch_gpu_241112

# === Set PYTHONPATH to Yoke repo ===
export PYTHONPATH=/usr/projects/artimis/mpmm/wish/Yoke/src:$PYTHONPATH

# === Navigate to harness directory ===
cd /usr/projects/artimis/mpmm/wish/Yoke/applications/harnesses/ch_DDP_time_loderunner

# === Run training study using user-specific CSV ===
CSV_FILE="hyperparameters.${USER}.csv"
if [[ -f "$CSV_FILE" ]]; then
    echo "Launching jobs for $USER using $CSV_FILE..."
    python START_study.py --csv "$CSV_FILE"
else
    echo "❌ ERROR: Could not find $CSV_FILE"
    exit 1
fi

# === Run the check progress script ===
echo "Waiting 5 seconds before summarizing progress..."
sleep 5
python check_progress.py
