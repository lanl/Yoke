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
    python check_progress.py                         # Monitor current user's jobs (assumes CWD is Yoke harness dir)
    python check_progress.py /path/to/harness        # Monitor current user's jobs in specified harness
    python check_progress.py alice bob               # Monitor jobs for multiple users in CWD harness
    python check_progress.py /path/to/harness alice  # Monitor jobs for user 'alice' in specified harness
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
        except subprocess.CalledProcessError:
            continue
    return active_jobs

def summarize_study(study_dir, job_lookup, total_epochs):
    study_id = os.path.basename(study_dir)
    try:
        study_index = int(study_id.split("_")[1])
    except Exception:
        study_index = -1

    # Find latest checkpoint
    ckpt_files = sorted(glob(os.path.join(study_dir, "checkpoints", "*.ckpt")))
    latest_ckpt = os.path.basename(ckpt_files[-1]) if ckpt_files else "N/A"

    # Determine latest epoch from training log
    log_file = os.path.join(study_dir, "training.log")
    if os.path.isfile(log_file):
        epoch_times = extract_epoch_times(log_file)
        latest_epoch = len(epoch_times)
        total_minutes = sum(epoch_times)
    else:
        latest_epoch = 0
        total_minutes = 0.0

    # Fetch job info if active
    job_info = job_lookup.get(study_id, {})
    jobid = job_info.get("jobid", "N/A")
    jobname = job_info.get("jobname", "N/A")
    job_status = job_info.get("status", "DONE")

    # Get wall-clock and CPU time
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
    # Accept an optional harness path. If provided and valid, use it as the Yoke harness directory.
    args = sys.argv[1:]
    if args and os.path.isdir(args[0]) and os.path.isdir(os.path.join(args[0], "runs")):
        harness_dir = args[0]
        users = args[1:] if len(args) > 1 else [os.getenv("USER")]
    else:
        harness_dir = os.getcwd()
        users = args if args else [os.getenv("USER")]

    # Switch to the harness directory so that relative paths (runs/, training_input.tmpl) work correctly
    try:
        os.chdir(harness_dir)
    except Exception as e:
        print(f"⚠️ Could not change directory to {harness_dir}: {e}")
        sys.exit(1)

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
    df = pd.DataFrame(sorted(results, key=lambda x: x["index"]))
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
