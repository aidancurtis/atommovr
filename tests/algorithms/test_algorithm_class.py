import numpy as np
import pytest

from atommover.algorithms.Algorithm_class import Algorithm, get_effective_target_grid


def test_get_success_flag():
    # Check True checkerboard configuration
    matrix = np.array([[0, 1, 0, 1], [1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 0]]).reshape(
        4, 4, 1
    )
    target = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]).reshape(
        4, 4, 1
    )
    success_flag = Algorithm.get_success_flag(matrix, target, do_ejection=False)
    assert success_flag

    # Check False checkerboard configuration
    matrix = np.array([[0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0]]).reshape(
        4, 4, 1
    )
    target = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]).reshape(
        4, 4, 1
    )
    success_flag = Algorithm.get_success_flag(matrix, target, do_ejection=False)
    assert not success_flag

    # Check True middle_fill configuration
    matrix = np.array([[0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0]]).reshape(
        4, 4, 1
    )
    target = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]).reshape(
        4, 4, 1
    )
    success_flag = Algorithm.get_success_flag(matrix, target, do_ejection=False)
    assert success_flag

    # Check False middle_fill configuration
    matrix = np.array([[0, 1, 0, 1], [0, 1, 0, 0], [0, 1, 1, 1], [1, 0, 0, 0]]).reshape(
        4, 4, 1
    )
    target = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]).reshape(
        4, 4, 1
    )
    success_flag = Algorithm.get_success_flag(matrix, target, do_ejection=False)
    assert not success_flag

    # Check False middle_fill configuration with do_ejection
    matrix = np.array([[0, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]).reshape(
        4, 4, 1
    )
    target = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]).reshape(
        4, 4, 1
    )
    success_flag = Algorithm.get_success_flag(matrix, target, do_ejection=True)
    assert not success_flag

    # Check True middle_fill configuration with do_ejection
    matrix = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]).reshape(
        4, 4, 1
    )
    target = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]).reshape(
        4, 4, 1
    )
    success_flag = Algorithm.get_success_flag(matrix, target, do_ejection=True)
    assert success_flag


def test_get_effective_target_grid():
    target = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]).reshape(
        4, 4, 1
    )
    start_row, end_row, start_col, end_col = get_effective_target_grid(target)
    assert start_row == 1
    assert start_col == 1
    assert end_row == 2
    assert end_col == 2

    target = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]).reshape(
        4, 4, 1
    )
    start_row, end_row, start_col, end_col = get_effective_target_grid(target)
    assert start_row == 2
    assert start_col == 2
    assert end_row == 2
    assert end_col == 2

    target = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]).reshape(
        4, 4, 1
    )
    with pytest.raises(Exception):
        get_effective_target_grid(target)
