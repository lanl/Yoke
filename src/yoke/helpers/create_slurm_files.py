import json
import os
from collections import defaultdict
from pathlib import Path

class MkSlurm:
    def __init__(self, config_path, output_path, template_path = "templates/training_slurm.tmpl"):
        scriptdir = Path(__file__).parent
        tPath = scriptdir / template_path
        with open(config_path, 'r') as file:
            self._config = defaultdict(None, json.load(file))
        self._output_path = output_path
        with open(tPath, 'r') as file:
            self._template = file.read()

    def generateSlurm(self):
        template = self._template
        template = template.replace("[JOBNAME]", self._config["jobname"])
        template = template.replace("[LOG]", self._config["log"])
        if self._config['debug']:
            template = template.replace(
                "[RUNINFO]",
                "#SBATCH --partition=gpu_debug\n#SBATCH --reservation=gpu_debug\n#SBATCH --time=02:00:00")
        else:
            template = template.replace(
                "[RUNINFO]",
                "#SBATCH --partition=gpu\n#SBATCH --time=16:00:00")
        if self._config["email"] and len(self._config["email"]) > 0:
            eList = ','.join(self._config["email"])
            template = template.replace("[EMAIL]", f"#SBATCH --mail-user={eList}\n#SBATCH --mail-type=ALL")
        else:
            template = template.replace("[EMAIL]", "")
        if self._config['local']:
            template = template.replace("[LOCAL]", "export PYTHONPATH=../../../../../src:../../../src:$PYTHONPATH")
        else:
            template = template.replace("[LOCAL]", "")
        return template