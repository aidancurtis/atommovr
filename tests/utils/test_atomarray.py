import numpy as np
import pytest

from atommover.utils import Move
from atommover.utils.AtomArray import AtomArray
from atommover.utils.core import Configurations
from atommover.utils.errormodels import UniformVacuumTweezerError


def test_atom_array_initialization():
    array = AtomArray(shape=[3, 3], n_species=1)
    array.matrix = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]]).reshape(3, 3, 1)
    assert array.matrix.shape == (3, 3, 1)
    assert array.target.shape == (3, 3, 1)
    assert np.array_equal(
        array.matrix, np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]]).reshape(3, 3, 1)
    )


def test_load_tweezers():
    array = AtomArray(shape=[3, 3], n_species=1)
    array.load_tweezers()
    assert not np.array_equal(array.matrix, np.zeros(shape=(3, 3, 1)))


def test_generate_checkerboard_target():
    array = AtomArray(shape=[4, 4])
    array.generate_target()
    assert np.array_equal(
        array.target,
        np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]).reshape(
            4, 4, 1
        ),
    )


def test_generate_middle_fill_target():
    array = AtomArray(shape=[4, 4])
    array.generate_target(Configurations.MIDDLE_FILL)
    assert np.array_equal(
        array.target,
        np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]).reshape(
            4, 4, 1
        ),
    )


def test_move_atoms():
    array = AtomArray(shape=[3, 3], n_species=1)
    array.matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).reshape(3, 3, 1)
    # 0 1 0
    # 1 0 1
    # 0 1 0
    move_list = [
        [Move(0, 1, 0, 2)],
        [Move(1, 2, 2, 2)],
        [Move(2, 1, 2, 0)],
        [Move(1, 0, 0, 0)],
    ]
    # 1 0 1
    # 0 0 0
    # 1 0 1
    total_time, n_parallel_moves, n_total_moves = array.move_atoms(move_list)
    assert n_parallel_moves == 1
    assert n_total_moves == 4


def test_get_effective_target_grid():
    array = AtomArray(shape=[3, 3], n_species=1)
    array.matrix = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]]).reshape(3, 3, 1)
    array.target = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]).reshape(3, 3, 1)
    bounding_box = array.get_effective_target_grid()
    assert bounding_box == (0, 2, 1, 1)

    array = AtomArray(shape=[3, 3], n_species=1)
    array.matrix = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]]).reshape(3, 3, 1)
    array.target = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).reshape(3, 3, 1)
    bounding_box = array.get_effective_target_grid()
    assert bounding_box == (1, 1, 1, 1)


def test_is_target_loaded():
    array = AtomArray(shape=[3, 3], n_species=1)
    array.matrix = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]]).reshape(3, 3, 1)
    array.target = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]).reshape(3, 3, 1)
    success_flag = array.is_target_loaded()
    assert success_flag == False

    array = AtomArray(shape=[3, 3], n_species=1)
    array.matrix = np.array([[0, 1, 1], [1, 1, 0], [0, 0, 1]]).reshape(3, 3, 1)
    array.target = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]]).reshape(3, 3, 1)
    success_flag = array.is_target_loaded()
    assert success_flag == True

    array = AtomArray(shape=[3, 3], n_species=1)
    array.matrix = np.array([[0, 1, 1], [1, 1, 0], [0, 0, 1]]).reshape(3, 3, 1)
    array.target = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]]).reshape(3, 3, 1)
    success_flag = array.is_target_loaded(do_ejection=True)
    assert success_flag == False

    array = AtomArray(shape=[3, 3], n_species=1)
    array.matrix = np.array([[0, 1, 1], [1, 1, 0], [0, 1, 1]]).reshape(3, 3, 1)
    target = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]).reshape(3, 3, 1)
    success_flag = array.is_target_loaded(target=target)
    assert success_flag == True


def test_evaluate_moves():
    array = AtomArray(shape=[3, 3], n_species=1)
    array.matrix = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]]).reshape(3, 3, 1)
    # 0 1 1
    # 1 0 0
    # 0 0 1
    move_list = [
        [[Move(0, 1, 0, 0), Move(0, 0, 1, 0)], [Move(1, 0, 1, 1)], [Move(2, 2, 1, 2)]]
    ]
    # 0 0 1
    # 1 1 1
    # 0 0 0
    total_time, n_parallel_moves, n_total_moves = array.evaluate_moves(move_list)
    assert n_parallel_moves == 2
    assert n_total_moves == 4
