"""Collection of helper functions for processing strings."""

import numpy as np


def replace_keys(study_dict: dict, data: str) -> str:
    """Replace "key" values in a string with dictionary values.

    Args:
        study_dict (dict): Dictionary of keys and values to replace.
        data (str): Data to replace keys in.

    Returns:
        str: Data with keys replaced.

    Raises:
        ValueError: If an unrecognized datatype is encountered in the list.
    """
    for key, value in study_dict.items():
        if key == "studyIDX":
            data = data.replace(f"<{key}>", f"{value:03d}")
        elif isinstance(value, np.float64) or isinstance(value, float):
            data = data.replace(f"<{key}>", f"{value:5.4f}")
        elif isinstance(value, np.int64) or isinstance(value, int):
            data = data.replace(f"<{key}>", f"{value:d}")
        elif isinstance(value, str):
            data = data.replace(f"<{key}>", f"{value}")
        elif isinstance(value, np.bool_) or isinstance(value, bool):
            data = data.replace(f"<{key}>", f"{str(value)}")
        else:
            print("Key is", key, "with value of", value, "with type", type(value))
            raise ValueError("Unrecognized datatype in hyperparameter list.")

    return data
