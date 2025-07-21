"""Test for restart functionality."""

from pathlib import Path

import pytest

from yoke.utils.restart import continuation_setup


def test_slurm_continuation_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test continuation_setup for SLURM submission type.

    This test creates template files with placeholders and verifies that
    continuation_setup replaces placeholders and writes training input and
    slurm files.

    Args:
        tmp_path (Path): Temporary directory fixture.
        monkeypatch (Any): Fixture to change working directory.

    Returns:
        None
    """
    monkeypatch.chdir(tmp_path)
    # Create input template
    (tmp_path / "training_input.tmpl").write_text("load <CHECKPOINT>")
    # Create SLURM template
    (tmp_path / "training_slurm.tmpl").write_text(
        "execute <INPUTFILE> at epoch <epochIDX>"
    )

    checkpoint = "path/to/check"
    study_idx = 2
    last_epoch = 5
    out_file = continuation_setup(checkpoint, study_idx, last_epoch, "slurm")

    expected_input = "study002_restart_training_epoch0006.input"
    expected_slurm = "study002_restart_training_epoch0006.slurm"
    assert out_file == expected_slurm

    inp = tmp_path / expected_input
    slurm = tmp_path / expected_slurm
    assert inp.exists() and slurm.exists()

    inp_text = inp.read_text()
    slurm_text = slurm.read_text()
    assert "<CHECKPOINT>" not in inp_text and checkpoint in inp_text
    assert "<INPUTFILE>" not in slurm_text and expected_input in slurm_text
    assert "<epochIDX>" not in slurm_text and f"{last_epoch + 1:04d}" in slurm_text


def test_flux_continuation_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test continuation_setup for FLUX submission type.

    This test creates template files with placeholders and verifies that
    continuation_setup replaces placeholders and writes training input and
    flux files.

    Args:
        tmp_path (Path): Temporary directory fixture.
        monkeypatch (Any): Fixture to change working directory.

    Returns:
        None
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "training_input.tmpl").write_text("in <CHECKPOINT>")
    (tmp_path / "training_flux.tmpl").write_text("launch <INPUTFILE> epoch <epochIDX>")

    checkpoint = "chk.pt"
    study_idx = 1
    last_epoch = 0
    out_file = continuation_setup(checkpoint, study_idx, last_epoch, "FLUx")

    expected_input = "study001_restart_training_epoch0001.input"
    expected_flux = "study001_restart_training_epoch0001.flux"
    assert out_file == expected_flux

    inp = tmp_path / expected_input
    flux = tmp_path / expected_flux
    assert inp.exists() and flux.exists()
    assert checkpoint in inp.read_text()

    flux_text = flux.read_text()
    assert expected_input in flux_text
    assert f"{last_epoch + 1:04d}" in flux_text


def test_shell_continuation_setup_case_insensitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test continuation_setup for SHELL submission type.

    This test creates template files with placeholders and verifies that
    continuation_setup replaces placeholders and writes training input and
    shell scripts in a case-insensitive manner.

    Args:
        tmp_path (Path): Temporary directory fixture.
        monkeypatch (Any): Fixture to change working directory.

    Returns:
        None
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "training_input.tmpl").write_text("run <CHECKPOINT>")
    (tmp_path / "training_shell.tmpl").write_text("bash <INPUTFILE> resume <epochIDX>")

    checkpoint = "resume.ckpt"
    study_idx = 3
    last_epoch = 7
    out_file = continuation_setup(checkpoint, study_idx, last_epoch, "ShElL")

    expected_input = "study003_restart_training_epoch0008.input"
    expected_shell = "study003_restart_training_epoch0008.sh"
    assert out_file == expected_shell

    inp = tmp_path / expected_input
    shell = tmp_path / expected_shell
    assert inp.exists() and shell.exists()
    assert checkpoint in inp.read_text()

    shell_text = shell.read_text()
    assert expected_input in shell_text
    assert f"{last_epoch + 1:04d}" in shell_text


def test_default_submission_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test default submission type is SLURM when none is specified.

    Args:
        tmp_path (Path): Temporary directory fixture.
        monkeypatch (Any): Fixture to change working directory.

    Returns:
        None
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "training_input.tmpl").write_text("dflt <CHECKPOINT>")
    (tmp_path / "training_slurm.tmpl").write_text("job <INPUTFILE> <epochIDX>")

    checkpoint = "default.ckpt"
    study_idx = 4
    last_epoch = 1
    out_file = continuation_setup(checkpoint, study_idx, last_epoch)

    expected_slurm = "study004_restart_training_epoch0002.slurm"
    assert out_file == expected_slurm
    assert (tmp_path / expected_slurm).exists()


def test_invalid_submission_type_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that invalid submission types raise an UnboundLocalError.

    Args:
        tmp_path (Path): Temporary directory fixture.
        monkeypatch (Any): Fixture to change working directory.

    Returns:
        None
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "training_input.tmpl").write_text("fail <CHECKPOINT>")

    with pytest.raises(UnboundLocalError):
        continuation_setup("x", 0, 0, "batch")
