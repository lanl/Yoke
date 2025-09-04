#!/bin/bash -v

# B. Kaiser, 9/3/2025

# This will run a data integrity verification script if that script requires:
# a) a csv that lists material properties / catalogs all simulations
# b) simulations ids from min_id to max_id (integers)
# c) the data are .npz files in npz_search_dir
# d) the report (a text file) print to output_dir

python data_integrity_cx.py \
  --csv_path /lustre/scratch5/exempt/artimis/mpmm/bkaiser/design_cx241203_MASTER.csv \
  --npz_search_dir /lustre/scratch5/exempt/artimis/data/cx241203_fp16_full/ \
  --name "cx241203" \
  --save_dir "./"
