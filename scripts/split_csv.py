#!/usr/bin/env python3
"""
Split a CSV into N smaller CSVs based on a provided list of usernames.

Usage:
    python split_csv.py <csv_file> <usernames>

Arguments:
    csv_file   Path to the input CSV file that will be split.
    usernames  A Python-list literal of usernames, e.g.
               "['wish','hickmank','dschodt','spandit','galgal']"

The script will:
  1. Read the entire CSV into a pandas.DataFrame.
  2. Compute N = len(usernames) and split the DataFrame into N chunks
     (each chunk has roughly len(df)/N rows; the last chunk may be smaller
     if len(df) is not perfectly divisible by N).
  3. Write each chunk to a new file named:
         hyperparameters.<username>.csv

Example:
    Given an input file named "ddp_study_time.csv" and a list
    of users `['wish','hickmank','dschodt','spandit','galgal']`, run:

        python split_csv.py ddp_study_time.csv "['wish','hickmank','dschodt','spandit','galgal']"

    This will produce:
        hyperparameters.wish.csv
        hyperparameters.hickmank.csv
        hyperparameters.dschodt.csv
        hyperparameters.spandit.csv
        hyperparameters.galgal.csv
"""

import argparse
import ast
import os
import sys
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split a CSV into multiple CSV files, one per username."
        ),
        epilog=(
            "Example:\n"
            "  python split_csv.py ddp_study_time.csv "
            "\"['wish','hickmank','dschodt','spandit','galgal']\"\n\n"
            "This yields hyperparameters.wish.csv, "
            "hyperparameters.hickmank.csv, hyperparameters.dschodt.csv, "
            "hyperparameters.spandit.csv, hyperparameters.galgal.csv.\n"
        )
    )
    parser.add_argument(
        "csv_file",
        help="Path to the input CSV file to split."
    )
    parser.add_argument(
        "usernames",
        help=(
            "List of usernames as a Python literal. "
            "E.g. \"['wish','hickmank','dschodt','spandit','galgal']\"."
        )
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Verify that the CSV file exists
    if not os.path.isfile(args.csv_file):
        print(f"ERROR: Input file '{args.csv_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Safely parse the usernames list literal
    try:
        usernames = ast.literal_eval(args.usernames)
    except Exception as e:
        print(
            f"ERROR: Failed to parse usernames list. Ensure it's a valid Python list literal.\n"
            f"Exception: {e}",
            file=sys.stderr
        )
        sys.exit(1)

    if not isinstance(usernames, list) or not all(isinstance(u, str) for u in usernames):
        print(
            "ERROR: 'usernames' must be a Python list of strings, e.g. "
            "['wish','hickmank','dschodt','spandit','galgal'].",
            file=sys.stderr
        )
        sys.exit(1)

    # Read the CSV (ignoring lines that start with '#')
    try:
        df = pd.read_csv(args.csv_file, comment="#")
    except Exception as e:
        print(f"ERROR: Could not read CSV '{args.csv_file}': {e}", file=sys.stderr)
        sys.exit(1)

    # Determine directory of the input CSV to write outputs there
    csv_dir = os.path.dirname(args.csv_file)

    total_rows = len(df)
    n_splits = len(usernames)
    if n_splits == 0:
        print("ERROR: The provided username list is empty.", file=sys.stderr)
        sys.exit(1)

    # Determine chunk size (last chunk may be smaller)
    chunk_size = total_rows // n_splits + (total_rows % n_splits > 0)

    # Perform splitting and write out each file
    for idx, uname in enumerate(usernames):
        start_idx = idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk = df.iloc[start_idx:end_idx]

        if chunk.empty:
            print(
                f"WARNING: No rows assigned to user '{uname}' "
                f"(indices {start_idx}:{end_idx})."
            )
            continue

        output_filename = f"hyperparameters.{uname}.csv"
        output_path = os.path.join(csv_dir, output_filename)
        try:
            chunk.to_csv(output_path, index=False)
        except Exception as e:
            print(f"ERROR: Could not write '{output_filename}': {e}", file=sys.stderr)
            sys.exit(1)

        print(f"Wrote {len(chunk)} rows to {output_filename}")

if __name__ == "__main__":
    main()
