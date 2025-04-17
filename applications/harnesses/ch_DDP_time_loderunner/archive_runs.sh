#!/bin/bash

# ----------------------------------------------------------------------
# archive_runs.sh
#
# Renames the 'runs/' directory to 'runs_backup_<timestamp>/'
#
# Usage:
#   ./archive_runs.sh
# ----------------------------------------------------------------------

# === Check if the runs directory exists ===
if [[ ! -d "runs" ]]; then
    echo "❌ No 'runs/' directory found. Nothing to archive."
    exit 1
fi

# === Generate timestamped backup directory name ===
timestamp=$(date +%Y%m%d_%H%M)
backup_dir="runs_backup_$timestamp"

# === Rename the runs directory ===
mv runs "$backup_dir"

echo "✅ Archived 'runs/' to '$backup_dir'"
