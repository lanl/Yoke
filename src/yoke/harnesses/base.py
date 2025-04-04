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
        dryrun (bool): Flag to turn off job submission.

    """
    def __init__(
            self, 
            rundir: str="./runs", 
            template_dir: str=".", 
            cp_file: str="cp_files.txt",
            dryrun: bool=False
            ):
        """Initialization for HarnessStudy."""

        self.rundir = Path(rundir)
        self.template_dir = Path(template_dir)
        self.cp_file = Path(cp_file)
        self.DRYRUN = dryrun

        # Template and base files
        self.input_template = self.template_dir / "training_input.tmpl"
        self.slurm_template = self.template_dir / "training_slurm.tmpl"
        self.slurm_json = self.template_dir / "slurm_config.json"

        self.rundir.mkdir(parents=True, exist_ok=True)

    def load_hyperparameters(self, csv_path):
        """Read hyperparameters from a CSV into a list of dicts."""
        df = pd.read_csv(
            csv_path, 
            sep=",", 
            header=0, 
            index_col=0, 
            comment="#", 
            engine="python"
            )
        
        study_list = []
        for idx in df.index.values:
            study = df.loc[idx].to_dict()
            study["studyIDX"] = int(idx)
            study_list.append(study)

        return study_list

    def render_template(self, template_path, substitutions):
        """Render template with conditional optional blocks."""
        with open(template_path, "r") as f:
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

    def copy_files(self, study_dir):
        """Copy the files listed in cp_file into the study directory."""
        with open(self.cp_file, "r") as f:
            for line in f:
                file_path = line.strip()
                if file_path:
                    shutil.copy(file_path, study_dir)
                    print(f"[COPY] {file_path} -> {study_dir}")

    def generate_initial_inputs(self, study_dir, study):
        """Generate the input and SLURM scripts for the first submission."""
        sid = study["studyIDX"]
        study["epochIDX"] = f"{sid:03d}"
        study["INPUTFILE"] = f"study{sid:03d}_START.input"

        # Ensure that the continuation and checkpoint arguments do not appear in the
        # intialization inputs.
        study.pop("CONTINUATION", None)

        # Render input and SLURM templates
        input_rendered = self.render_template(self.input_template, study)
        slurm_rendered = self._get_slurm_template(study)

        # Modify START files with substitutions
        input_path = study_dir / f"study{sid:03d}_START.input"
        slurm_path = study_dir / f"study{sid:03d}_START.slurm"

        with open(input_path, "w") as f:
            f.write(input_rendered)
        with open(slurm_path, "w") as f:
            f.write(slurm_rendered)

        return slurm_path

    def generate_tmpl_inputs(self, study_dir, study):
        """Generate the input and SLURM templates for job continuation."""
        sid = study["studyIDX"]

        # For templates epochIDX and INPUTFILE should be left as variables.
        study.pop("epochIDX", None)
        study.pop("INPUTFILE", None)
        study["CONTINUATION"] = True

        # Render input and SLURM templates
        input_rendered = self.render_template(self.input_template, study)
        slurm_rendered = self._get_slurm_template(study)

        # Modify START files with substitutions
        input_path = study_dir / "training_input.tmpl"
        slurm_path = study_dir / "training_slurm.tmpl"

        with open(input_path, "w") as f:
            f.write(input_rendered)
        with open(slurm_path, "w") as f:
            f.write(slurm_rendered)
    
    def _get_slurm_template(self, study):
        """Return rendered SLURM script, either from JSON or template."""
        if self.slurm_json.exists():
            slurm_obj = create_slurm_files.MkSlurm(config_path=str(self.slurm_json))
            tmpl = slurm_obj.generateSlurm()
        else:
            with open(self.slurm_template, "r") as f:
                tmpl = f.read()

        return strings.replace_keys(study, tmpl)

    def submit_job(self, study_dir: str, slurm_path: str) -> None:
        """Submit a SLURM job."""
        submit_str = (
            f"cd {study_dir}; "
            f"sbatch {slurm_path.name}; "
            f"cd .."
        )

        if self.DRYRUN:
            # Just print what would be executed
            print(f"[DRY RUN] Would execute: {submit_str}.")
        else:
            # Submit Job
            os.system(submit_str)

    def run_study(self, study):
        """Run a single study: generate inputs, copy files, and submit job."""
        # Make Study Directory
        study_dir = self.rundir / "study_{:03d}".format(study["studyIDX"])
        study_dir.mkdir(parents=True, exist_ok=True)

        self.copy_files(study_dir)
        self.generate_tmpl_inputs(study_dir, study)
        slurm_path = self.generate_initial_inputs(study_dir, study)
        self.submit_job(study_dir, slurm_path)
