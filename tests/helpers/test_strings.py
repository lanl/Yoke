"""Test strings module."""

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
    ],
)
def test_replace_keys(study_dict: dict, data: str, expected: str) -> None:
    """Ensure replace_keys() works on some hardcoded test cases."""
    result = strings.replace_keys(study_dict=study_dict, data=data)
    assert result == expected
