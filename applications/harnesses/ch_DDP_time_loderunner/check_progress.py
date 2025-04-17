#!/usr/bin/env python3
"""
check_progress.py

Monitors the progress of Yoke-based training studies across users.

Features:
- Tracks epochs, checkpoints, runtime, and status
- Fetches TOTAL_EPOCHS from training_input.tmpl
- Supports multiple users
- Adds JobID and JobName to output
- Outputs to terminal and CSV

Usage:
    python check_progress.py               # Monitor current user's jobs
    python check_progress.py alice bob     # Monitor jobs for multiple users
"""

import os
import re
import sys
import subprocess
from glob import glob
import pandas as pd

def read_total_epochs(tmpl_path="training_input.tmpl"):
    try:
        with open(tmpl_path) as f:
            for line in f:
                if line.strip().lower() == "--total_epochs":
                    next_line = next(f).strip()
                    return int(next_line)
    except Exception as e:
        print(f"⚠️ Could not read TOTAL_EPOCHS from {tmpl_path}: {e}")
    return 2000  # Default fallback

def extract_epoch_times(log_file):
    times = []
    with open(log_file) as f:
        for line in f:
            match = re.search(r"Epoch time \(minutes\): ([\d\.]+)", line)
            if match:
                times.append(float(match.group(1)))
    return times

def get_active_jobs(user_list):
    """Return a dict mapping study_id → job info (JobID, Status, JobName)."""
    active_jobs = {}
    for user in user_list:
        try:
            output = subprocess.check_output(["squeue", "-u", user], text=True)
            for line in output.strip().split("\n")[1:]:
                parts = line.split()
                if len(parts) < 5:
                    continue
                job_id, partition, name, user, status = parts[:5]
                match = re.search(r"ddp_s(\d+)", name)
                if match:
                    study_idx = int(match.group(1))
                    study_id = f"study_{study_idx:03d}"
                    active_jobs[study_id] = {
                        "jobid": job_id,
                        "status": status,
                        "jobname": name
                    }
        except Exception as e:
            print(f"⚠️ Warning: could not read squeue for user '{user}': {e}")
    return active_jobs

def get_job_usage(jobid):
    """Query sacct for wall-clock and CPU time for a Slurm job."""
    try:
        output = subprocess.check_output(
            ["sacct", "-j", jobid, "--format=JobID,Elapsed,TotalCPU,State", "--parsable2"],
            text=True
        )
        lines = output.strip().split("\n")[1:]
        for line in lines:
            fields = line.strip().split("|")
            if jobid in fields[0]:
                return (fields[1], fields[2])  # Elapsed, TotalCPU
        return ("N/A", "N/A")
    except subprocess.CalledProcessError:
        return ("N/A", "N/A")

def summarize_study(study_dir, job_lookup, total_epochs):
    study_id = os.path.basename(study_dir)
    study_index = int(study_id.split("_")[1])

    training_csvs = sorted(glob(os.path.join(study_dir, "training_*.csv")))
    latest_epoch = len(training_csvs)

    checkpoints = sorted(glob(os.path.join(study_dir, "*.pth")))
    latest_ckpt = os.path.basename(checkpoints[-1]) if checkpoints else "None"

    log_files = sorted(glob(os.path.join(study_dir, "*_epoch*.out")))
    epoch_times = []
    for log in log_files:
        epoch_times.extend(extract_epoch_times(log))
    total_minutes = sum(epoch_times)

    job_info = job_lookup.get(study_id, {})
    job_status = job_info.get("status", "DONE")
    jobid = job_info.get("jobid", "N/A")
    jobname = job_info.get("jobname", "N/A")
    elapsed, cpu_time = get_job_usage(jobid) if jobid != "N/A" else ("N/A", "N/A")
    done_flag = "✅" if latest_epoch >= total_epochs else ""

    return {
        "study": study_id,
        "index": study_index,
        "epoch": latest_epoch,
        "checkpoint": latest_ckpt,
        "train_time_min": round(total_minutes, 2),
        "status": job_status,
        "jobid": jobid,
        "jobname": jobname,
        "elapsed": elapsed,
        "cpu_time": cpu_time,
        "done": done_flag
    }

def main():
    # CLI: multiple users or default to $USER
    users = sys.argv[1:] if len(sys.argv) > 1 else [os.getenv("USER")]
    total_epochs = read_total_epochs()

    runs_dir = "runs"
    all_studies = sorted(glob(os.path.join(runs_dir, "study_*")))
    job_lookup = get_active_jobs(users)

    results = []
    header = (
        f"{'Study':<12} {'Epoch':<8} {'Last Checkpoint':<35} {'Train Time (min)':>16} "
        f"{'Status':>10} {'JobID':>10} {'JobName':>20} {'Elapsed (HH:MM:SS)':>20} "
        f"{'CPU Time (HH:MM:SS)':>20} {'Done':>6}"
    )
    print(header)
    print("-" * len(header))

    for study_dir in all_studies:
        summary = summarize_study(study_dir, job_lookup, total_epochs)
        results.append(summary)
        print(
            f"{summary['study']:<12} {summary['epoch']:<8} {summary['checkpoint']:<35} "
            f"{summary['train_time_min']:>16.2f} {summary['status']:>10} {summary['jobid']:>10} "
            f"{summary['jobname']:>20} {summary['elapsed']:>20} {summary['cpu_time']:>20} {summary['done']:>6}"
        )

    # Sort results by study index and export to CSV
    df = pd.DataFrame(results)
    df.sort_values(by="index", inplace=True)
    df.drop(columns=["index"], inplace=True)
    df.to_csv("study_status_summary.csv", index=False)

    # Footer legend
    print("\nLegend:")
    print("  Status = R       → Running")
    print("  Status = PD      → Pending in queue")
    print("  Status = DONE    → Job no longer in queue")
    print("  Done ✅           → All expected epochs completed (>= total_epochs)")
    print("  Elapsed          → Wall-clock time used for current job (HH:MM:SS)")
    print("  CPU Time         → Cumulative CPU time used (HH:MM:SS)")
    print("  Train Time (min) → Sum of epoch durations from training logs")
    print(f"\n  TOTAL_EPOCHS is set to: {total_epochs}")

if __name__ == "__main__":
    main()
