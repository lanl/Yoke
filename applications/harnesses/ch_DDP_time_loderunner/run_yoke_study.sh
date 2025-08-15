#!/bin/bash

# ----------------------------------------------------------------------
# run_yoke_study.sh
#
# Description:
#   This script navigates to a specified Yoke harness directory, sets up
#   the PYTHONPATH appropriately, and launches a Yoke training study using
#   a user-specific CSV file (hyperparameters.<USER>.csv).
#
# Usage:
#   ./run_yoke_study.sh [path_to_harness_directory]
#
#   [path_to_harness_directory] should point to a Yoke harness, e.g.:
#     /usr/projects/artimis/mpmm/wish/Yoke/applications/harnesses/ch_DDP_time_loderunner
#   If you omit the path, the script assumes the current directory is the harness.
#
#   The script will:
#     1. Verify that the harness directory exists.
#     2. Derive YOKE_ROOT by stripping everything after “/applications” in the
#        provided path. For example, given
#        /usr/projects/artimis/mpmm/wish/Yoke/applications/harnesses/ch_DDP_time_loderunner,
#        YOKE_ROOT becomes /usr/projects/artimis/mpmm/wish/Yoke.
#     3. Export PYTHONPATH=${YOKE_ROOT}/src.
#     4. cd into the harness directory.
#     5. Look for hyperparameters.<USER>.csv in that directory.
#     6. If found, run: python START_study.py --csv hyperparameters.<USER>.csv
#     7. Otherwise, emit an error and exit.
#
# ⚠️ IMPORTANT:
#   • This script must be run under Bash.
#   • Before running, execute:
#         newgrp artimis
#     so that any generated files are group-owned by “artimis” (otherwise
#     files will default to your personal group and not be accessible to teammates).
# ----------------------------------------------------------------------

# --- Ensure we’re running under Bash ---
if [ -z "$BASH_VERSION" ]; then
    echo "❌ ERROR: This script must be run under Bash. Exiting."
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

# --- Determine HARNESS_DIR: use argument if provided; else use current directory ---
if [ "$#" -gt 1 ]; then
    echo "Usage: $0 [path_to_harness_directory]"
    echo "If no path is provided, the current directory is assumed to be the harness."
    exit 1
elif [ "$#" -eq 1 ]; then
    HARNESS_DIR="$1"
else
    HARNESS_DIR="$(pwd)"
fi

# --- Verify that HARNESS_DIR exists and is a directory ---
if [ ! -d "$HARNESS_DIR" ]; then
    echo "❌ ERROR: Harness directory '$HARNESS_DIR' does not exist or is not a directory."
    exit 1
fi

# --- Derive YOKE_ROOT from the harness path ---
#
# We assume the harness path contains “/applications”. Everything up to (but
# not including) “/applications” is the Yoke root directory.
#
# Example:
#   HARNESS_DIR="/usr/projects/artimis/mpmm/wish/Yoke/applications/harnesses/ch_DDP_time_loderunner"
#   YOKE_ROOT="${HARNESS_DIR%%/applications*}"
#   → YOKE_ROOT="/usr/projects/artimis/mpmm/wish/Yoke"
#
YOKE_ROOT="${HARNESS_DIR%%/applications*}"

if [ -z "$YOKE_ROOT" ] || [ ! -d "$YOKE_ROOT" ]; then
    echo "❌ ERROR: Unable to determine valid YOKE_ROOT from '$HARNESS_DIR'."
    exit 1
fi

# --- Export PYTHONPATH to include Yoke’s src directory ---
export PYTHONPATH="${YOKE_ROOT}/src"
echo "PYTHONPATH set to: $PYTHONPATH"

# --- Navigate to the harness directory ---
cd "$HARNESS_DIR" || {
    echo "❌ ERROR: Failed to cd to '$HARNESS_DIR'."
    exit 1
}

# === Load module and activate environment ===
module load python/3.11-anaconda-2023.07
source /usr/projects/hpcsoft/common/x86_64/anaconda/2023.07-python-3.11/etc/profile.d/conda.sh
conda activate torch_ch_gpu_241112

# --- Run training study using user-specific CSV ---
CSV_FILE="hyperparameters.${USER}.csv"

if [ -f "$CSV_FILE" ]; then
    echo "Launching jobs for user '$USER' using '$CSV_FILE'..."
    python START_study.py --csv "$CSV_FILE"
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ ERROR: START_study.py exited with status $EXIT_CODE."
        exit $EXIT_CODE
    fi
else
    echo "❌ ERROR: Could not find CSV file '$CSV_FILE' in '$HARNESS_DIR'."
    exit 1
fi

# Optionally, you can add any cleanup or group‐permission adjustments here.

exit 0
