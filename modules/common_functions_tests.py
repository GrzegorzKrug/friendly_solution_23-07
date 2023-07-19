import pytest

from common_functions import NamingClass, get_eps


# from common_functions import get_eps


check_args = [
        (1, 1, 2, 2, 2, 3, 4, 5, 6, 7),  # Base
        (1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 6, 7, 8),  # With optional
        (1, 1, 2, 2, 2, 3, 4, 5, 6, 7, "", 7, 8),  # With optional
        (1, 1, 2, 2, 2, 3, 4, 5, 6, 7, "", None, 8),  # With optional
        (1, 1, 2, 2, 2, 3, 4, 5, 6, 7, None, None, 8),  # With optional
        (1, 1, 2, 2, 2, 3, 4, 5, 6, 7, None, None, None,),  # With optional
]


@pytest.mark.parametrize("pos_args", check_args)
def test_1_Naming_from_Path(pos_args):
    inst1 = NamingClass(*pos_args)
    print(f"Path: {inst1.path}")
    inst2 = NamingClass.from_path(inst1.path)

    assert inst1.path == inst2.path, "Paths do not match"


@pytest.mark.parametrize("pos_args", check_args)
def test_2_NamingCopy(pos_args):
    inst1 = NamingClass(*pos_args)

    inst2 = inst1.copy()

    assert inst1.path == inst2.path, "Paths do not match"


test3_args = [
        (n, repit, power, maxexp)
        for n in [100, 500]
        for repit in [1, 2, 4, 5, 25, 50]
        for power in [0.5, 1, 1.5, 2]
        for maxexp in [0.5, 0.8, 1]
]


@pytest.mark.parametrize("M,repeat,power,max_explore", test3_args)
def test_3_eps_start(M, repeat, power, max_explore):
    val = get_eps(0, M, repeat, power, max_explore)
    assert val == max_explore


@pytest.mark.parametrize("M,repeat,power,max_explore", test3_args)
def test_3_eps_end(M, repeat, power, max_explore):
    val = get_eps(M - 1, M, repeat, power, max_explore)
    diff = abs(val)
    assert diff < 1e-4, f"Last val {M - 1} of {M} should have 0"


test4_args = [
        (m, repit, power, maxexp)
        for m in [100]
        for repit in [2]
        for power in [0.5, 1, 1.5, 2]
        for maxexp in [0.5, 0.8, 1]
]


@pytest.mark.parametrize("M,repeat,power,max_explore", test4_args)
def test_4_eps_repeat_2(M, repeat, power, max_explore):
    val = get_eps(0, M, repeat, power, max_explore)
    assert val == max_explore, f"1 should have: {max_explore}"

    val = get_eps(49, M, repeat, power, max_explore)
    assert val == 0, f"49 should have 0"

    val = get_eps(50, M, repeat, power, max_explore)
    assert val == max_explore, f"50 should have: {max_explore}"

    val = get_eps(99, M, repeat, power, max_explore)
    assert val == 0, f"99 should have 0"
