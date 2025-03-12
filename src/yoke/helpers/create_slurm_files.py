import json
import os
from collections import defaultdict

class MkSlurm:
    def __init__(self, config_path, output_path, template_path = "templates/training_START_slurm.tmpl"):
        with open(config_path, 'r') as file:
            self._config = defaultdict(None, json.load(file))
        self._output_path = output_path
        with open(template_path, 'r') as file:
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
        self._slurmFile = template

    def writeSlurm(self):
        with open(os.path.join(self._output_path, "training_slurm.tmpl"), 'w') as file:
            file.write(self._slurmFile)