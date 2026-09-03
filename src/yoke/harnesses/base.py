"""Base class for a Yoke harness."""

import os
import shutil
import pandas as pd
from pathlib import Path
from yoke.helpers import strings, create_slurm_files


class HarnessStudy:
    """HarnessStudy class.

    Defines class containing attributes for a Yoke Harness object. Methods of this
    class are then used to submit entries of a Yoke study.

    Args:
        rundir (str or Path): Output directory for the study run
        template_dir (str or Path): Directory containing .tmpl files
        cp_file (str or Path): File listing training files to copy per study
        submission_type (str): Job-submission system to prepare files for. One of
            ``"slurm"`` or ``"shell"``.
        dryrun (bool): Flag to turn off job submission.

    """

    #: Dispatch table mapping a submission system to its template filename, the
    #: output-script extension, and the command used to submit the script.
    SUBMISSION_SYSTEMS: dict[str, dict[str, str]] = {
        "slurm": {
            "template": "training_slurm.tmpl",
            "ext": "slurm",
            "submit": "sbatch",
        },
        "shell": {
            "template": "training_shell.tmpl",
            "ext": "sh",
            "submit": "source",
        },
    }

    def __init__(
        self,
        rundir: str = "./runs",
        template_dir: str = ".",
        cp_file: str = "cp_files.txt",
        submission_type: str = "slurm",
        dryrun: bool = False,
    ) -> None:
        """Initialization for HarnessStudy."""
        self.rundir = Path(rundir)
        self.template_dir = Path(template_dir)
        self.cp_file = Path(cp_file)
        self.DRYRUN = dryrun

        submission_type = submission_type.lower()
        if submission_type not in self.SUBMISSION_SYSTEMS:
            valid = ", ".join(sorted(self.SUBMISSION_SYSTEMS))
            raise ValueError(
                f"Unknown submission type {submission_type!r}. "
                f"Supported types are: {valid}."
            )
        self.submission_type = submission_type
        self.submission_config = self.SUBMISSION_SYSTEMS[submission_type]

        # Template and base files
        self.input_template = self.template_dir / "training_input.tmpl"
        self.submission_template = self.template_dir / self.submission_config["template"]
        self.slurm_json = self.template_dir / "slurm_config.json"

        self.rundir.mkdir(parents=True, exist_ok=True)

    def load_hyperparameters(self, csv_path: str) -> list[dict]:
        """Read hyperparameters from a CSV into a list of dicts.

        Args:
            csv_path (str): Path to the hyperparameter CSV. The first column is
                treated as the study index (``studyIDX``).

        Returns:
            list[dict]: One dictionary per study row, with the study index stored
            under the ``studyIDX`` key.
        """
        df = pd.read_csv(
            csv_path,
            sep=",",
            header=0,
            index_col=0,
            comment="#",
            engine="python",
        )

        study_list = []
        for idx in df.index.values:
            study = df.loc[idx].to_dict()
            study["studyIDX"] = int(idx)
            study_list.append(study)

        return study_list

    def render_template(self, template_path: Path, substitutions: dict) -> str:
        """Render a template file, honoring optional conditional blocks.

        Lines bounded by ``# <<optional:KEY>>`` and ``# <<end>>`` are included in
        the rendered output only when ``KEY`` is present in ``substitutions``.
        All remaining ``<KEY>`` tokens are substituted via
        :func:`yoke.helpers.strings.replace_keys`.

        Args:
            template_path (Path): Path to the template file to render.
            substitutions (dict): Mapping of keys to values for substitution.

        Returns:
            str: The rendered template contents.
        """
        with open(template_path) as f:
            lines = f.readlines()

        rendered = []
        skip_block = False
        for line in lines:
            if line.strip().startswith("# <<optional:"):
                key = line.strip().split(":")[1].rstrip(">>")
                skip_block = key not in substitutions
                continue
            if line.strip() == "# <<end>>":
                skip_block = False
                continue
            if skip_block:
                continue
            rendered.append(strings.replace_keys(substitutions, line))

        return "".join(rendered)

    def copy_files(self, study_dir: Path) -> None:
        """Copy the files listed in ``cp_file`` into the study directory.

        Args:
            study_dir (Path): Destination directory for the copied files.
        """
        with open(self.cp_file) as f:
            for line in f:
                file_path = line.strip()
                if file_path:
                    shutil.copy(file_path, study_dir)
                    print(f"[COPY] {file_path} -> {study_dir}")

    def generate_initial_inputs(self, study_dir: Path, study: dict) -> Path:
        """Generate the input and submission scripts for the first submission.

        Args:
            study_dir (Path): Directory in which to write the first-launch files.
            study (dict): Study substitution dictionary.

        Returns:
            Path: Path to the generated first-launch submission script.
        """
        sid = study["studyIDX"]
        study["epochIDX"] = f"{sid:03d}"
        study["INPUTFILE"] = f"study{sid:03d}_START.input"

        # Ensure that the continuation and checkpoint arguments do not appear in the
        # intialization inputs.
        study.pop("CONTINUATION", None)

        # Render input and submission templates
        input_rendered = self.render_template(self.input_template, study)
        submit_rendered = self._render_submission_template(study)

        # Modify START files with substitutions
        ext = self.submission_config["ext"]
        input_path = study_dir / f"study{sid:03d}_START.input"
        submit_path = study_dir / f"study{sid:03d}_START.{ext}"

        with open(input_path, "w") as f:
            f.write(input_rendered)
        with open(submit_path, "w") as f:
            f.write(submit_rendered)

        return submit_path

    def generate_tmpl_inputs(self, study_dir: Path, study: dict) -> None:
        """Generate the input and submission templates for job continuation.

        Args:
            study_dir (Path): Directory in which to write the continuation
                templates.
            study (dict): Study substitution dictionary.
        """
        # For templates epochIDX and INPUTFILE should be left as variables.
        study.pop("epochIDX", None)
        study.pop("INPUTFILE", None)
        study["CONTINUATION"] = True

        # Render input and submission templates
        input_rendered = self.render_template(self.input_template, study)
        submit_rendered = self._render_submission_template(study)

        # Modify template files with substitutions
        input_path = study_dir / "training_input.tmpl"
        submit_path = study_dir / self.submission_config["template"]

        with open(input_path, "w") as f:
            f.write(input_rendered)
        with open(submit_path, "w") as f:
            f.write(submit_rendered)

    def _render_submission_template(self, study: dict) -> str:
        """Return the rendered submission script for the selected system.

        For SLURM, a ``slurm_config.json`` in the template directory is used to
        synthesize the script if present; otherwise the submission template file
        is used directly.

        Args:
            study (dict): Study substitution dictionary.

        Returns:
            str: The rendered submission script.
        """
        if self.submission_type == "slurm" and self.slurm_json.exists():
            slurm_obj = create_slurm_files.MkSlurm(config_path=str(self.slurm_json))
            tmpl = slurm_obj.generateSlurm()
        else:
            with open(self.submission_template) as f:
                tmpl = f.read()

        return strings.replace_keys(study, tmpl)

    def submit_job(self, study_dir: Path, submit_path: Path) -> None:
        """Submit a job using the configured submission system.

        Args:
            study_dir (Path): Directory containing the submission script.
            submit_path (Path): Path to the submission script to submit.
        """
        submit_cmd = self.submission_config["submit"]
        submit_str = f"cd {study_dir}; {submit_cmd} {submit_path.name}; cd .."

        if self.DRYRUN:
            # Just print what would be executed
            print(f"[DRY RUN] Would execute: {submit_str}.")
        else:
            # Submit Job
            os.system(submit_str)

    def run_study(self, study: dict) -> None:
        """Run a single study: generate inputs, copy files, and submit job.

        Args:
            study (dict): Study substitution dictionary for a single CSV row.
        """
        # Make Study Directory
        study_dir = self.rundir / "study_{:03d}".format(study["studyIDX"])
        study_dir.mkdir(parents=True, exist_ok=True)

        self.copy_files(study_dir)
        self.generate_tmpl_inputs(study_dir, study)
        submit_path = self.generate_initial_inputs(study_dir, study)
        self.submit_job(study_dir, submit_path)
