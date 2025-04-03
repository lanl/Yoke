"""Base class for a Yoke harness."""

import os
import shutil
import pandas as pd
from pathlib import Path
from yoke.helpers import strings, create_slurm_files


class HarnessStudy:
    def __init__(self, study_dir, template_dir=".", cp_file="cp_files.txt"):
        """
        Args:
            study_dir (str or Path): Output directory for the study run
            template_dir (str or Path): Directory containing .tmpl files
            cp_file (str or Path): File listing training files to copy per study

        """
        self.study_dir = Path(study_dir)
        self.template_dir = Path(template_dir)
        self.cp_file = Path(cp_file)

        # Template and base files
        self.input_template = self.template_dir / "training_input.tmpl"
        self.slurm_template = self.template_dir / "training_slurm.tmpl"
        self.slurm_json = self.template_dir / "slurm_config.json"

        self.study_dir.mkdir(parents=True, exist_ok=True)

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
        """Substitute placeholders in a template string."""
        with open(template_path, "r") as f:
            content = f.read()
        return strings.replace_keys(substitutions, content)

    def copy_files(self):
        """Copy the files listed in cp_file into the study directory."""
        with open(self.cp_file, "r") as f:
            for line in f:
                file_path = line.strip()
                if file_path:
                    shutil.copy(file_path, self.study_dir)
                    print(f"[COPY] {file_path} -> {self.study_dir}")

    def generate_initial_inputs(self, study):
        """Generate the input and SLURM scripts for the first submission."""
        sid = study["studyIDX"]
        study["epochIDX"] = 1

        # Render input and SLURM templates
        input_rendered = self.render_template(self.input_template, study)
        slurm_rendered = self._get_slurm_template(study)

        # Modify START files with substitutions
        input_path = self.study_dir / f"study{sid:03d}_START.input"
        slurm_path = self.study_dir / f"study{sid:03d}_START.slurm"

        with open(input_path, "w") as f:
            f.write(input_rendered)
        with open(slurm_path, "w") as f:
            f.write(slurm_rendered)

        return slurm_path

    def generate_continuation(self, checkpoint_path, studyIDX, last_epoch):
        """Generate a continuation SLURM job from template."""
        epochIDX = last_epoch + 1

        # Render input with checkpoint
        with open(self.input_template) as f:
            content = f.read()
        content = content.replace("<CHECKPOINT>", checkpoint_path)

        input_name = f"study{studyIDX:03d}_restart_training_epoch{epochIDX:04d}.input"
        input_path = self.study_dir / input_name
        with open(input_path, "w") as f:
            f.write(content)

        # Render SLURM with input file reference
        with open(self.slurm_template) as f:
            slurm_data = f.read()
        slurm_data = slurm_data.replace("<INPUTFILE>", input_name)
        slurm_data = slurm_data.replace("<epochIDX>", f"{epochIDX:04d}")

        slurm_name = f"study{studyIDX:03d}_restart_training_epoch{epochIDX:04d}.slurm"
        slurm_path = self.study_dir / slurm_name
        with open(slurm_path, "w") as f:
            f.write(slurm_data)

        return slurm_path

    def _get_slurm_template(self, study):
        """Return rendered SLURM script, either from JSON or template."""
        if self.slurm_json.exists():
            slurm_obj = create_slurm_files.MkSlurm(config_path=str(self.slurm_json))
            tmpl = slurm_obj.generateSlurm()
        else:
            with open(self.slurm_template, "r") as f:
                tmpl = f.read()
        return strings.replace_keys(study, tmpl)

    def submit_job(self, slurm_path):
        """Submit a SLURM job."""
        submit_str = f"cd {self.study_dir} && sbatch {slurm_path.name}"
        print(f"[SUBMIT] {submit_str}")
        os.system(submit_str)

    def run_study(self, study):
        """Run a single study: generate inputs, copy files, and submit job."""
        self.copy_files()
        slurm_path = self.generate_initial_inputs(study)
        self.submit_job(slurm_path)
