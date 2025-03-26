import json
import os
from collections import defaultdict
from pathlib import Path

def generateSingleRowSlurm(key, value):
    if value is not None:
        return f"#SBATCH --{key}={value}\n"
    return ""

class MkSlurm:
    def __init__(self, config_path, template_dir="templates"):
        scriptdir = Path(__file__).parent
        template_dir = scriptdir / template_dir

        with open(config_path, 'r') as file:
            self._config = json.load(file)

        system_name = self._config['system']
        system_config_file = template_dir / f"{system_name}.json"
        with open(system_config_file, 'r') as file:
            self._sysconfig = json.load(file)

        template_file = template_dir / f"{self._sysconfig['scheduler']}.tmpl"
        with open(template_file, 'r') as file:
            self._template = file.read()

    def generateSlurm(self):
        template = self._sysconfig
        slurm_args_string = ""

        if 'generated-params' in self._config and 'run-config' in self._config['generated-params']:
            run_config = self._config['generated-params']['run-config']
        else:
            run_config = template['generated-params']['run-config']['default-mode']

        run_config_params = template['generated-params']['run-config'][run_config]

        run_config_slurm_lines = map(lambda x: generateSingleRowSlurm(x[0], x[1]), run_config_params.items())
        slurm_args_string += "".join(run_config_slurm_lines)

        if 'generated-params' in self._config and 'log' in self._config['generated-params']:
            log = self._config['generated-params']['log']
        else:
            log = template['generated-params']['log']
        log_slurm_lines = map(lambda x: generateSingleRowSlurm(x[0], f'{log}{x[1]}'), [('output', '.out'), ('error', '.err')])
        slurm_args_string += "".join(log_slurm_lines)

        if 'generated-params' in self._config and 'email' in self._config['generated-params']:
            emails = self._config['generated-params']['email']
            if emails is not None and len(emails) > 0:
                email_slurm_lines = map(lambda x: generateSingleRowSlurm(x[0], x[1]), [('mail-user', ','.join(emails)), ('mail-type', 'ALL')])
                slurm_args_string += "".join(email_slurm_lines)


        if 'generated-params' in self._config and 'verbose' in self._config['generated-params']:
            verbose = self._config['generated-params']['verbose']
        else:
            verbose = template['generated-params']['verbose']
        if verbose > 0:
            verbose_slurm_lines = '#SBATCH -' + 'v' * verbose + '\n'
        slurm_args_string += verbose_slurm_lines

        custom_params = {}
        if 'custom-params' in self._config:
            custom_params = self._config['custom-params']
        default_params = template['custom-params']
        for k in default_params:
            if k in custom_params:
                slurm_args_string += generateSingleRowSlurm(k, custom_params[k])
            else:
                slurm_args_string += generateSingleRowSlurm(k, default_params[k])
        return self._template.replace("[SLURM-PARAMS]", slurm_args_string)