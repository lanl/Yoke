"""A file to auto-generate the SLURM files used in harnesses.

This is one step moving the repo closer to the harness as config.
"""

import json
from pathlib import Path


def generateSingleRowSlurm(key: str, value: str) -> str:
    """Generates a single SLURM row of config of the form #SBATCH --key=value.

    params:
        key (str): the key/parameter name for SLURM.
        value (str): the value that gets assigned to the key within the SLURM config.
    """
    if value is not None:
        return f"#SBATCH --{key}={value}\n"
    return ""


class MkSlurm:
    """This is where the slurm file is generated.

    It uses the generateSingleRowSlurm as a helper function.
    """

    def __init__(self, config_path: str, template_dir: str = "templates") -> None:
        """Initializes the MkSlurm object.

        params:
            config_path (str): where the config.json lives.
            template_dir (str): where the directory containing the template files live.
        """
        scriptdir = Path(__file__).parent
        template_dir = scriptdir / template_dir

        with open(config_path) as file:
            self._config = json.load(file)

        system_name = self._config["system"]
        system_config_file = template_dir / f"{system_name}.json"
        with open(system_config_file) as file:
            self._sysconfig = json.load(file)

        template_file = template_dir / f"{self._sysconfig['scheduler']}.tmpl"
        with open(template_file) as file:
            self._template = file.read()

    def generateSlurm(self) -> str:
        """Generates the SLURM file using the config.

        Returns:
            str: the SLURM file as a string.
        """
        template = self._sysconfig
        slurm_arg_params = {}

        if (
            "generated-params" in self._config
            and "run-config" in self._config["generated-params"]
        ):
            run_config = self._config["generated-params"]["run-config"]
        else:
            run_config = template["generated-params"]["run-config"]["default-mode"]

        run_config_params = template["generated-params"]["run-config"][run_config]
        for arg in run_config_params:
            if arg not in slurm_arg_params:
                slurm_arg_params[arg] = run_config_params[arg]

        if (
            "generated-params" in self._config
            and "log" in self._config["generated-params"]
        ):
            log = self._config["generated-params"]["log"]
        else:
            log = template["generated-params"]["log"]
        # construct [('output', f'{log}.out'), ('error', f'{log}.err')] using map
        if "output" not in log:
            slurm_arg_params["output"] = f"{log}.out"
        if "error" not in log:
            slurm_arg_params["error"] = f"{log}.err"

        if (
            "generated-params" in self._config
            and "email" in self._config["generated-params"]
        ):
            emails = self._config["generated-params"]["email"]
            if emails is not None and len(emails) > 0:
                if "mail-user" not in slurm_arg_params:
                    slurm_arg_params["mail-user"] = ",".join(emails)
                if "mail-type" not in slurm_arg_params:
                    slurm_arg_params["mail-type"] = "ALL"

        if (
            "generated-params" in self._config
            and "verbose" in self._config["generated-params"]
        ):
            verbose = self._config["generated-params"]["verbose"]
        else:
            verbose = template["generated-params"]["verbose"]

        if verbose > 0:
            verbose_slurm_lines = "#SBATCH -" + "v" * verbose + "\n"
        else:
            verbose_slurm_lines = ""

        custom_params = {}
        if "custom-params" in self._config:
            custom_params = self._config["custom-params"]
        default_params = template["custom-params"]
        for k in default_params:
            if k not in custom_params:
                custom_params[k] = default_params[k]
        # override slurm_arg_params with custom_params
        for k in custom_params:
            slurm_arg_params[k] = custom_params[k]
        # generate slurm_args_string
        slurm_args_string = (
            "".join(
                [
                    generateSingleRowSlurm(k, slurm_arg_params[k])
                    for k in slurm_arg_params
                ]
            )
            + verbose_slurm_lines
        )
        return self._template.replace("[SLURM-PARAMS]", slurm_args_string)
