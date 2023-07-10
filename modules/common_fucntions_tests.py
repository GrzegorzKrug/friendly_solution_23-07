import pytest

from common_functions import NamingClass


check_args = [
        (1, 1, 2, 2, 2, 3, 4),  # Base
        (1, 1, 2, 2, 2, 3, 4, 6, 7, 8),  # With optional
        (1, 1, 2, 2, 2, 3, 4, "", 7, 8),  # With optional
        (1, 1, 2, 2, 2, 3, 4, "", None, 8),  # With optional
        (1, 1, 2, 2, 2, 3, 4, None, None, 8),  # With optional
        (1, 1, 2, 2, 2, 3, 4, None, None, None,),  # With optional
]


@pytest.mark.parametrize("pos_args", check_args)
def test_1_(pos_args):
    inst1 = NamingClass(*pos_args)
    print(f"Path: {inst1.path}")
    inst2 = NamingClass.from_path(inst1.path)

    assert inst1.path == inst2.model_dir, "Paths do not match"
