#!/usr/bin/env bash
set -euo pipefail

# run_study.sh <train_script> <config_file>

# 1) Export PYTHONPATH pointing back to src
#    $(dirname "$0") is the harness dir; ../../.. => repo root/src
export PYTHONPATH="$(dirname "$0")/../../../src"

# 2) Read args
train_script="$1"
config_file="$2"

# 3) Invoke Python in the current working directory (study dir)
echo "▶️ Running: python \"$train_script\" @\"$config_file\""
python "$train_script" @"$config_file"