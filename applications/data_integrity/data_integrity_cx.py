"""Generate a data-integrity report for a simulation dataset.

This script contains functions that generate a report regarding missing files and
bad files for a dataset consisting of simulation files where id is the simulation
number and idx is the time stamp.
"""

import os
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd


def verify_pvi_backmat_keys_in_npz(
    df_filtered: pd.DataFrame,
    npz_search_dir: str | Path,
    n_ts: int,
    name: str,
    verbose: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:  # pylint: disable=too-many-locals, too-many-statements
    """Checks that for each simulation ID.

      1. All n_ts expected PVI index .npz files exist (idx00000 to idx00056 for cylex).
      2. Each file contains the expected 'density_<BACKMAT>' key.

    Parameters:
    - df_filtered: DataFrame with 'key', 'sim_id', and 'backMat'
                  (assumed one row per sim_id; if not, it still works per-row)
    - npz_search_dir: directory to look for .npz files
    - verbose: whether to print progress

    Returns:
    - issues: list of dicts with:
        - 'sim_id'
        - 'missing_files': list of filenames that are missing
        - 'missing_keys': list of {file, expected_key, found_keys}
    - report: dict with roll-up stats:
        - 'total_simulations'
        - 'missing_simulations'            (count sim_ids with n_ts/n_ts files missing)
        - 'found_simulations'              (total_simulations - missing_simulations)
        - 'sims_with_any_missing_files'    (how many have missing idx indices)
        - 'total_missing_files_over_found'
        - 'avg_missing_files_per_found_sim'(float; 0.0 if none found)
        - 'max_missing_files_per_sim'      (int; 0 if none found)
        - 'sim_ids_with_max_missing_files' (list)
        - 'total_missing_key_occurrences'  (count of files missing the expected key)

    """
    issues: list[dict[str, object]] = []

    # Per-sim counters to compute the report
    per_sim_missing_files: dict[
        str, int
    ] = {}  # sim_id -> count of missing files (0..n_ts)
    per_sim_missing_keys: dict[
        str, int
    ] = {}  # sim_id -> count of files missing expected key

    for _, row in df_filtered.iterrows():
        sim_id = str(row["sim_id"]).zfill(5)
        base_name = f"{name}_id{sim_id}"
        expected_key = f"density_{str(row['backMat']).strip()}"
        missing_files: list[str] = []
        missing_keys: list[dict[str, object]] = []

        # n_ts time stamps per simulation id
        for i in range(n_ts):  # idx00000 ... idx{n_ts}
            idx_str = f"idx{i:05d}"
            filename = f"{base_name}_pvi_{idx_str}.npz"
            filepath = os.path.join(str(npz_search_dir), filename)

            if not os.path.exists(filepath):
                missing_files.append(filename)
                continue

            try:
                with np.load(filepath) as data:
                    if expected_key not in data:
                        mk: dict[str, object] = {
                            "file": filename,
                            "expected_key": expected_key,
                            "found_keys": list(data.keys()),
                        }
                        missing_keys.append(mk)
                        if verbose:
                            print(f" ❌ Missing '{expected_key}' in {filename}")
            except (OSError, ValueError) as e:
                if verbose:
                    print(f" ❗ Error loading {filename}: {e}")
                missing_files.append(filename)

        # Save per-sim tallies
        per_sim_missing_files[sim_id] = len(missing_files)
        per_sim_missing_keys[sim_id] = len(missing_keys)

        if missing_files or missing_keys:
            issues.append(
                {
                    "sim_id": sim_id,
                    "missing_files": missing_files,
                    "missing_keys": missing_keys,
                }
            )

        if verbose and (missing_files or missing_keys):
            print(
                f" ⚠️ Issues for sim_id={sim_id}: "
                f"{len(missing_files)} file(s) missing, "
                f"{len(missing_keys)} file(s) missing expected key."
            )

    # --- Build the roll-up report ---
    total_simulations = len(per_sim_missing_files)
    missing_simulations = sum(1 for c in per_sim_missing_files.values() if c == n_ts)
    found_simulations = total_simulations - missing_simulations

    # Only consider found simulations when computing file-missing averages/max
    found_counts = [c for c in per_sim_missing_files.values() if c < n_ts]
    sims_with_any_missing_files = sum(1 for c in found_counts if c > 0)
    total_missing_files_over_found = sum(found_counts) if found_counts else 0
    avg_missing_files_per_found_sim = mean(found_counts) if found_counts else 0.0
    max_missing_files_per_sim = max(found_counts) if found_counts else 0
    sim_ids_with_max_missing_files = (
        [
            sid
            for sid, c in per_sim_missing_files.items()
            if c == max_missing_files_per_sim and c < n_ts
        ]
        if found_counts
        else []
    )

    total_missing_key_occurrences = sum(per_sim_missing_keys.values())

    report: dict[str, object] = {
        "total_simulations": total_simulations,
        "missing_simulations": missing_simulations,
        "found_simulations": found_simulations,
        "sims_with_any_missing_files": sims_with_any_missing_files,
        "total_missing_files_over_found": total_missing_files_over_found,
        "avg_missing_files_per_found_sim": float(avg_missing_files_per_found_sim),
        "max_missing_files_per_sim": int(max_missing_files_per_sim),
        "sim_ids_with_max_missing_files": sim_ids_with_max_missing_files,
        "total_missing_key_occurrences": int(total_missing_key_occurrences),
    }

    if verbose:
        print("\n================ Report ================")
        print(f"Total simulations examined     : {total_simulations}")
        print(f"Missing simulations (n_ts/n_ts miss): {missing_simulations}")
        print(f"Found simulations               : {found_simulations}")
        print(f"Sims with any missing files     : {sims_with_any_missing_files}")
        print(f"Total missing files (found sims): {total_missing_files_over_found}")
        print(f"Avg missing files / found sim   : {avg_missing_files_per_found_sim:.2f}")
        print(f"Max missing files in a sim      : {max_missing_files_per_sim}")
        if sim_ids_with_max_missing_files:
            print(
                f"Sim(s) with that max            : "
                f"{', '.join(sim_ids_with_max_missing_files)}"
            )
        print(f"Missing expected-key occurrences: {total_missing_key_occurrences}")
        print("=======================================\n")

        print(f" 🔍 Final Summary: {len(issues)} sim_id(s) had missing files or keys.")

    return issues, report
