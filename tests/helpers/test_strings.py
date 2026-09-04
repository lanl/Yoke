"""Test strings module."""

import numpy as np
import pytest

from yoke.helpers import strings


@pytest.mark.parametrize(
    "study_dict, data, expected",
    [
        # Case 1: integer studyIDX
        ({"studyIDX": 1}, "<studyIDX>", "001"),
        # Case 2: float values
        ({"float_test": 1.23e-6}, "<float_test>", "1.23e-06"),
        # Case 3: int values
        ({"int_test": 12}, "<int_test>", "12"),
        # Case 4: str values
        ({"str_test": "test"}, "<str_test>", "test"),
        # Case 5: bool values
        ({"bool_test": True}, "<bool_test>", "1"),
        # Case 6: numpy float
        ({"npfloat_test": np.float64(2.5)}, "<npfloat_test>", "2.5"),
        # Case 7: numpy int
        ({"npint_test": np.int64(7)}, "<npint_test>", "7"),
        # Case 8: numpy bool (hits the dedicated np.bool_ branch)
        ({"npbool_test": np.bool_(True)}, "<npbool_test>", "True"),
    ],
)
def test_replace_keys(study_dict: dict, data: str, expected: str) -> None:
    """Ensure replace_keys() works on some hardcoded test cases."""
    result = strings.replace_keys(study_dict=study_dict, data=data)
    assert result == expected


def test_replace_keys_unrecognized_type_raises() -> None:
    """replace_keys() raises ValueError for an unsupported value type."""
    with pytest.raises(ValueError, match="Unrecognized datatype"):
        strings.replace_keys(study_dict={"bad": [1, 2, 3]}, data="<bad>")
