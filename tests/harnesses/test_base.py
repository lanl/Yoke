"""Tests for yoke.harnesses.base.HarnessStudy."""

from pathlib import Path

import pytest

from yoke.harnesses.base import HarnessStudy


def test_slurm_continuation_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """continuation_setup writes restart input/slurm files for SLURM."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "training_input.tmpl").write_text("load <CHECKPOINT>")
    (tmp_path / "training_slurm.tmpl").write_text(
        "execute <INPUTFILE> at epoch <epochIDX>"
    )

    checkpoint = "path/to/check"
    study_idx = 2
    last_epoch = 5
    out_file = HarnessStudy.continuation_setup(
        checkpoint, study_idx, last_epoch, "slurm"
    )

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


def test_shell_continuation_setup_case_insensitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """continuation_setup writes restart input/shell files, case-insensitive."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "training_input.tmpl").write_text("run <CHECKPOINT>")
    (tmp_path / "training_shell.tmpl").write_text("bash <INPUTFILE> resume <epochIDX>")

    checkpoint = "resume.ckpt"
    study_idx = 3
    last_epoch = 7
    out_file = HarnessStudy.continuation_setup(
        checkpoint, study_idx, last_epoch, "ShElL"
    )

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
    """continuation_setup defaults to SLURM when no type is specified."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "training_input.tmpl").write_text("dflt <CHECKPOINT>")
    (tmp_path / "training_slurm.tmpl").write_text("job <INPUTFILE> <epochIDX>")

    checkpoint = "default.ckpt"
    study_idx = 4
    last_epoch = 1
    out_file = HarnessStudy.continuation_setup(checkpoint, study_idx, last_epoch)

    expected_slurm = "study004_restart_training_epoch0002.slurm"
    assert out_file == expected_slurm
    assert (tmp_path / expected_slurm).exists()


def test_invalid_submission_type_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """continuation_setup rejects unsupported submission systems."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "training_input.tmpl").write_text("fail <CHECKPOINT>")

    with pytest.raises(ValueError):
        HarnessStudy.continuation_setup("x", 0, 0, "batch")


def _write_single_template_harness(harness_dir: Path, submission_type: str) -> None:
    """Write a minimal single-template harness for round-trip testing.

    Args:
        harness_dir (Path): Directory to populate with harness config files.
        submission_type (str): Submission system to build templates for.
    """
    config = HarnessStudy.SUBMISSION_SYSTEMS[submission_type]

    (harness_dir / "hyperparameters.csv").write_text(
        "studyIDX,init_learnrate\n1,0.001\n"
    )
    (harness_dir / "cp_files.txt").write_text("train.py\n")
    (harness_dir / "train.py").write_text("print('train')\n")

    # Single input template with an optional continuation block.
    (harness_dir / "training_input.tmpl").write_text(
        "--init_learnrate=<init_learnrate>\n"
        "--studyIDX=<studyIDX>\n"
        "# <<optional:CONTINUATION>>\n"
        "--continuation\n"
        "--checkpoint=<CHECKPOINT>\n"
        "# <<end>>\n"
    )
    # Single submission template referencing INPUTFILE and epochIDX.
    (harness_dir / config["template"]).write_text(
        "#!/bin/bash\n"
        "#JOB study<studyIDX> epoch <epochIDX>\n"
        "python train.py @<INPUTFILE>\n"
    )


@pytest.mark.parametrize("submission_type", ["slurm", "shell"])
def test_generate_then_continuation_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    submission_type: str,
) -> None:
    """generate_tmpl_inputs then continuation_setup produce valid restart files.

    Renders a study's continuation templates via ``run_study`` (dryrun), then
    runs the compute-node ``continuation_setup`` from the study directory and
    asserts the restart input/submission files are well-formed.
    """
    config = HarnessStudy.SUBMISSION_SYSTEMS[submission_type]

    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    _write_single_template_harness(harness_dir, submission_type)

    monkeypatch.chdir(harness_dir)

    harness = HarnessStudy(
        rundir="./runs",
        template_dir=".",
        cp_file="cp_files.txt",
        submission_type=submission_type,
        dryrun=True,
    )
    studies = harness.load_hyperparameters("hyperparameters.csv")
    for study in studies:
        harness.run_study(study)

    study_dir = harness_dir / "runs" / "study_001"

    # The continuation templates should be written and carry placeholders.
    cont_input_tmpl = (study_dir / "training_input.tmpl").read_text()
    assert "--continuation" in cont_input_tmpl
    assert "<CHECKPOINT>" in cont_input_tmpl

    cont_submit_tmpl = (study_dir / config["template"]).read_text()
    assert "<INPUTFILE>" in cont_submit_tmpl
    assert "<epochIDX>" in cont_submit_tmpl

    # Now emulate the compute-node continuation from within the study dir.
    monkeypatch.chdir(study_dir)
    checkpoint = "checkpoints/model_epoch0003.pt"
    study_idx = 1
    last_epoch = 3
    out_file = HarnessStudy.continuation_setup(
        checkpoint, study_idx, last_epoch, submission_type
    )

    ext = config["ext"]
    expected_input = "study001_restart_training_epoch0004.input"
    expected_submit = f"study001_restart_training_epoch0004.{ext}"
    assert out_file == expected_submit

    restart_input = (study_dir / expected_input).read_text()
    assert checkpoint in restart_input
    assert "<CHECKPOINT>" not in restart_input

    restart_submit = (study_dir / expected_submit).read_text()
    assert expected_input in restart_submit
    assert "0004" in restart_submit
    assert "<INPUTFILE>" not in restart_submit
    assert "<epochIDX>" not in restart_submit
