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


def test_constructor_rejects_invalid_submission_type(tmp_path: Path) -> None:
    """The HarnessStudy constructor rejects unsupported submission systems."""
    with pytest.raises(ValueError):
        HarnessStudy(
            rundir=str(tmp_path / "runs"),
            template_dir=str(tmp_path),
            cp_file=str(tmp_path / "cp.txt"),
            submission_type="flux",
        )


def test_submit_job_executes_when_not_dryrun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """submit_job invokes os.system with the configured submit command."""
    (tmp_path / "training_input.tmpl").write_text("x\n")
    (tmp_path / "training_slurm.tmpl").write_text("y\n")
    harness = HarnessStudy(
        rundir=str(tmp_path / "runs"),
        template_dir=str(tmp_path),
        cp_file=str(tmp_path / "cp.txt"),
        submission_type="slurm",
        dryrun=False,
    )

    calls: list[str] = []
    monkeypatch.setattr("os.system", lambda cmd: calls.append(cmd) or 0)

    study_dir = tmp_path / "runs" / "study_001"
    study_dir.mkdir(parents=True)
    harness.submit_job(study_dir, study_dir / "study001_START.slurm")

    assert len(calls) == 1
    assert "sbatch study001_START.slurm" in calls[0]


def test_render_template_includes_optional_block_when_key_present(
    tmp_path: Path,
) -> None:
    """render_template keeps an optional block when its key is in the subs."""
    tmpl = tmp_path / "training_input.tmpl"
    tmpl.write_text(
        "--studyIDX=<studyIDX>\n"
        "# <<optional:CONTINUATION>>\n"
        "--continuation\n"
        "--checkpoint=<CHECKPOINT>\n"
        "# <<end>>\n"
        "--done\n"
    )
    harness = HarnessStudy(
        rundir=str(tmp_path / "runs"),
        template_dir=str(tmp_path),
        cp_file=str(tmp_path / "cp.txt"),
        submission_type="slurm",
    )

    rendered = harness.render_template(
        tmpl, {"studyIDX": 3, "CONTINUATION": True, "CHECKPOINT": "ckpt.pt"}
    )

    assert "--studyIDX=003" in rendered
    assert "--continuation" in rendered
    assert "--checkpoint=ckpt.pt" in rendered
    assert "--done" in rendered
    # Marker lines are stripped from output.
    assert "<<optional:" not in rendered
    assert "<<end>>" not in rendered


def test_render_template_omits_optional_block_when_key_absent(
    tmp_path: Path,
) -> None:
    """render_template drops an optional block when its key is missing."""
    tmpl = tmp_path / "training_input.tmpl"
    tmpl.write_text(
        "--studyIDX=<studyIDX>\n"
        "# <<optional:CONTINUATION>>\n"
        "--continuation\n"
        "--checkpoint=<CHECKPOINT>\n"
        "# <<end>>\n"
        "--done\n"
    )
    harness = HarnessStudy(
        rundir=str(tmp_path / "runs"),
        template_dir=str(tmp_path),
        cp_file=str(tmp_path / "cp.txt"),
        submission_type="slurm",
    )

    rendered = harness.render_template(tmpl, {"studyIDX": 3})

    assert "--studyIDX=003" in rendered
    assert "--continuation" not in rendered
    # The <CHECKPOINT> token never appears because the whole block is skipped.
    assert "<CHECKPOINT>" not in rendered
    assert "--checkpoint" not in rendered
    assert "--done" in rendered


def test_load_hyperparameters_parses_rows_index_and_comments(
    tmp_path: Path,
) -> None:
    """load_hyperparameters returns one dict per row, keyed by studyIDX."""
    csv = tmp_path / "hyperparameters.csv"
    csv.write_text(
        "# a comment line describing the study\n"
        "studyIDX,init_learnrate,batch_size\n"
        "1,0.001,8\n"
        "# inline comment / ignored row description\n"
        "2,0.002,16\n"
    )
    harness = HarnessStudy(
        rundir=str(tmp_path / "runs"),
        template_dir=str(tmp_path),
        cp_file=str(tmp_path / "cp.txt"),
        submission_type="slurm",
    )

    studies = harness.load_hyperparameters(str(csv))

    assert len(studies) == 2
    assert studies[0]["studyIDX"] == 1
    assert isinstance(studies[0]["studyIDX"], int)
    assert studies[0]["init_learnrate"] == pytest.approx(0.001)
    assert studies[0]["batch_size"] == 8
    assert studies[1]["studyIDX"] == 2
    assert studies[1]["init_learnrate"] == pytest.approx(0.002)
    assert studies[1]["batch_size"] == 16


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
