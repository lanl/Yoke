#!/bin/bash

# ----------------------------------------------------------------------
# archive_runs.sh
#
# Description:
#   Rename the 'runs/' directory under a Yoke harness into a timestamped
#   backup (runs_backup_<timestamp>/). Accepts an optional path to a
#   harness directory; if none is provided, assumes the current directory
#   is the harness.
#
# Usage:
#   ./archive_runs.sh [path_to_harness_directory]
#
#   [path_to_harness_directory] should point to a Yoke harness, e.g.:
#     /usr/projects/artimis/mpmm/wish/Yoke/applications/harnesses/ch_DDP_time_loderunner
#   If you omit the path, the script assumes the current directory is the harness.
#
#   The script will:
#     1. Determine HARNESS_DIR (argument or current directory).
#     2. Verify that HARNESS_DIR exists and contains a 'runs/' subdirectory.
#     3. Generate a timestamp (YYYYMMDD_HHMM).
#     4. Rename 'runs/' → 'runs_backup_<timestamp>/' inside HARNESS_DIR.
#     5. Print a success message.
# ----------------------------------------------------------------------

# --- Ensure we’re running under Bash ---
if [ -z "$BASH_VERSION" ]; then
    echo "❌ ERROR: This script must be run under Bash. Exiting."
    exit 1
fi

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

# --- Navigate into the harness directory ---
cd "$HARNESS_DIR" || {
    echo "❌ ERROR: Failed to cd into '$HARNESS_DIR'."
    exit 1
}

# --- Check if the runs directory exists under HARNESS_DIR ---
if [[ ! -d "runs" ]]; then
    echo "❌ No 'runs/' directory found under '$HARNESS_DIR'. Nothing to archive."
    exit 1
fi

# --- Generate timestamped backup directory name ---
timestamp=$(date +%Y%m%d_%H%M)
backup_dir="runs_backup_$timestamp"

# --- Rename the runs directory to the timestamped backup ---
mv runs "$backup_dir"

echo "✅ Archived 'runs/' to '$backup_dir' under '$HARNESS_DIR'."
exit 0
