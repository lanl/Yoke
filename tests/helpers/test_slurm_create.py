"""Tests for Slurm create functionality."""

# test_mkslurm.py

import json
from pathlib import Path
from typing import Any

import pytest

import yoke.helpers.create_slurm_files as crSlurm_loader


def test_generateSingleRowSlurm_returns_row() -> None:
    """Test that generateSingleRowSlurm returns a valid SBATCH row."""
    row = crSlurm_loader.generateSingleRowSlurm("time", "01:00:00")
    assert row == "#SBATCH --time=01:00:00\n"


def test_generateSingleRowSlurm_empty_for_none() -> None:
    """Test that generateSingleRowSlurm returns empty when passed None."""
    value: Any = None
    row = crSlurm_loader.generateSingleRowSlurm("time", value)  # type: ignore[arg-type]
    assert row == ""


def test_default_generate_slurm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generateSlurm uses template defaults when no overrides are set."""
    # Point loader to look in our tmp_path for the "templates" folder
    dummy = tmp_path / "dummy.py"
    dummy.write_text("# placeholder")
    monkeypatch.setattr(crSlurm_loader, "__file__", str(dummy))

    # Build templates dir
    templates = tmp_path / "templates"
    templates.mkdir()

    # System config JSON
    system = "mysys"
    sysconfig = {
        "scheduler": "sbatch",
        "generated-params": {
            "run-config": {"default-mode": "mode1", "mode1": {"A": "v1"}},
            "log": "jobname",
            "verbose": 0,
        },
        "custom-params": {"time": "00:10:00"},
    }
    (templates / f"{system}.json").write_text(json.dumps(sysconfig))
    (templates / "sbatch.tmpl").write_text("#!/bin/bash\n[SLURM-PARAMS]\n")

    # Minimal config with only "system"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"system": system}))

    m = crSlurm_loader.MkSlurm(str(config_path))
    out = m.generateSlurm()

    # Header and default run-config param
    assert out.startswith("#!/bin/bash\n")
    assert "#SBATCH --A=v1\n" in out

    # Default log-based output/error
    assert "#SBATCH --output=jobname.out\n" in out
    assert "#SBATCH --error=jobname.err\n" in out

    # Default custom-params
    assert "#SBATCH --time=00:10:00\n" in out


def test_full_generate_slurm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generateSlurm applies run-config, email, verbose, and custom overrides."""
    # Monkeypatch module __file__
    dummy = tmp_path / "dummy.py"
    dummy.write_text("# placeholder")
    monkeypatch.setattr(crSlurm_loader, "__file__", str(dummy))

    # Create templates dir
    templates = tmp_path / "templates"
    templates.mkdir()

    # Detailed system config JSON
    system = "sysX"
    sysconfig = {
        "scheduler": "sbatch",
        "generated-params": {
            "run-config": {
                "default-mode": "m1",
                "m1": {"X": "x1"},
                "m2": {"Y": "y2"},
            },
            "log": "tmpljob",
            "verbose": 0,
        },
        "custom-params": {"time": "01:00:00", "mem": "1G"},
    }
    (templates / f"{system}.json").write_text(json.dumps(sysconfig))
    (templates / "sbatch.tmpl").write_text("#!/bin/bash\n[SLURM-PARAMS]\n")

    # Config with all overrides
    config = {
        "system": system,
        "generated-params": {
            "run-config": "m2",
            "log": "mylog",
            "email": ["a@b", "c@d"],
            "verbose": 2,
        },
        "custom-params": {"mem": "4G"},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    m = crSlurm_loader.MkSlurm(str(config_path))
    out = m.generateSlurm()

    # run-config override
    assert "#SBATCH --Y=y2\n" in out

    # log override
    assert "#SBATCH --output=mylog.out\n" in out
    assert "#SBATCH --error=mylog.err\n" in out

    # email override
    assert "#SBATCH --mail-user=a@b,c@d\n" in out
    assert "#SBATCH --mail-type=ALL\n" in out

    # verbose override: two v's
    assert "#SBATCH -vv\n" in out

    # custom-params override and default merge
    assert "#SBATCH --mem=4G\n" in out
    assert "#SBATCH --time=01:00:00\n" in out
