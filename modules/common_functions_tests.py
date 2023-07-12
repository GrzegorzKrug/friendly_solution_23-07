import pytest

from common_functions import NamingClass


check_args = [
        (1, 1, 2, 2, 2, 3, 4, 5, 6, 7),  # Base
        (1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 6, 7, 8),  # With optional
        (1, 1, 2, 2, 2, 3, 4, 5, 6, 7, "", 7, 8),  # With optional
        (1, 1, 2, 2, 2, 3, 4, 5, 6, 7, "", None, 8),  # With optional
        (1, 1, 2, 2, 2, 3, 4, 5, 6, 7, None, None, 8),  # With optional
        (1, 1, 2, 2, 2, 3, 4, 5, 6, 7, None, None, None,),  # With optional
]


@pytest.mark.parametrize("pos_args", check_args)
def test_1_(pos_args):
    inst1 = NamingClass(*pos_args)
    print(f"Path: {inst1.path}")
    inst2 = NamingClass.from_path(inst1.path)

    assert inst1.path == inst2.path, "Paths do not match"


@pytest.mark.parametrize("pos_args", check_args)
def test_2_(pos_args):
    inst1 = NamingClass(*pos_args)

    inst2 = inst1.copy()

    assert inst1.path == inst2.path, "Paths do not match"
