"""Tests for the ``yoke-start-study`` CLI entry point."""

from pathlib import Path

import pytest

from yoke.cli import start_study


def _write_harness(harness_dir: Path) -> None:
    """Populate a minimal single-template SLURM harness.

    Args:
        harness_dir (Path): Directory to populate with harness config files.
    """
    (harness_dir / "hyperparameters.csv").write_text(
        "studyIDX,init_learnrate\n1,0.001\n2,0.002\n"
    )
    (harness_dir / "cp_files.txt").write_text("train.py\n")
    (harness_dir / "train.py").write_text("print('train')\n")
    (harness_dir / "training_input.tmpl").write_text(
        "--init_learnrate=<init_learnrate>\n"
        "--studyIDX=<studyIDX>\n"
        "# <<optional:CONTINUATION>>\n"
        "--continuation\n"
        "--checkpoint=<CHECKPOINT>\n"
        "# <<end>>\n"
    )
    (harness_dir / "training_slurm.tmpl").write_text(
        "#!/bin/bash\n"
        "#JOB study<studyIDX> epoch <epochIDX>\n"
        "python train.py @<INPUTFILE>\n"
    )


def test_main_dryrun_creates_files_and_prints_submit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``main`` with ``--dryrun`` creates study dirs and prints submit commands."""
    _write_harness(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "yoke-start-study",
            "--csv",
            "hyperparameters.csv",
            "--rundir",
            "./runs",
            "--cpFile",
            "cp_files.txt",
            "--submissionType",
            "slurm",
            "--dryrun",
        ],
    )

    start_study.main()

    out = capsys.readouterr().out

    # Both study directories should exist with their first-launch files.
    for sid in (1, 2):
        study_dir = tmp_path / "runs" / f"study_{sid:03d}"
        assert study_dir.is_dir()
        assert (study_dir / f"study{sid:03d}_START.input").exists()
        assert (study_dir / f"study{sid:03d}_START.slurm").exists()
        # Continuation templates should also be present.
        assert (study_dir / "training_input.tmpl").exists()
        assert (study_dir / "training_slurm.tmpl").exists()
        # cp_files.txt entries copied in.
        assert (study_dir / "train.py").exists()

    # Dryrun should print the exact submit commands and never run them.
    assert "[DRY RUN]" in out
    assert "sbatch study001_START.slurm" in out
    assert "sbatch study002_START.slurm" in out


def test_main_shell_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``main`` honors ``--submissionType shell`` and prints a source command."""
    (tmp_path / "hyperparameters.csv").write_text("studyIDX,lr\n1,0.1\n")
    (tmp_path / "cp_files.txt").write_text("")
    (tmp_path / "training_input.tmpl").write_text("--lr=<lr>\n")
    (tmp_path / "training_shell.tmpl").write_text(
        "python train.py @<INPUTFILE> <epochIDX>\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "yoke-start-study",
            "--csv",
            "hyperparameters.csv",
            "--rundir",
            "./runs",
            "--cpFile",
            "cp_files.txt",
            "--submissionType",
            "shell",
            "--dryrun",
        ],
    )

    start_study.main()

    out = capsys.readouterr().out
    study_dir = tmp_path / "runs" / "study_001"
    assert (study_dir / "study001_START.sh").exists()
    assert "[DRY RUN]" in out
    assert "source study001_START.sh" in out
