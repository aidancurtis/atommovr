import numpy as np
import pytest

from atommover.utils import Move
from atommover.utils.AtomArray import AtomArray
from atommover.utils.core import Configurations
from atommover.utils.errormodels import UniformVacuumTweezerError


def test_atom_array_initialization():
    array = AtomArray(shape=[3,3], n_species=1)
    array.matrix = np.array([[0,1,1],
                             [1,0,0],
                             [0,0,1]]).reshape(3,3,1)
    assert array.matrix.shape == (3, 3, 1)
    assert array.target.shape == (3, 3, 1)
    assert np.array_equal(array.matrix, np.array([[0,1,1],
                                                  [1,0,0],
                                                  [0,0,1]]).reshape(3,3,1))

def test_load_tweezers():
    array = AtomArray(shape=[3,3], n_species=1)
    array.load_tweezers()
    assert not np.array_equal(array.matrix, np.zeros(shape=(3,3,1)))

def test_generate_checkerboard_target():
    array = AtomArray(shape=[4,4])
    array.generate_target()
    assert np.array_equal(array.target, np.array([[0,1,0,1],
                                                  [1,0,1,0],
                                                  [0,1,0,1],
                                                  [1,0,1,0]]).reshape(4,4,1))

def test_generate_middle_fill_target():
    array = AtomArray(shape=[4,4])
    array.generate_target(Configurations.MIDDLE_FILL)
    assert np.array_equal(array.target, np.array([[0,0,0,0],
                                                  [0,1,1,0],
                                                  [0,1,1,0],
                                                  [0,0,0,0]]).reshape(4,4,1))

def test_move_atoms():
    array = AtomArray(shape=[3,3], n_species=1)
    array.matrix = np.array([[0,1,1],
                             [1,0,0],
                             [0,0,1]]).reshape(3,3,1)
 
    # check that parallel moves work
    array.move_atoms([Move(1,0,1,1), Move(0,2,1,2)])
    assert np.array_equal(array.matrix, np.array([[0,1,0],
                                                  [0,1,1],
                                                  [0,0,1]]).reshape(3,3,1))
    # check that collisions expel atoms
    [failed_moves, flags], move_time = array.move_atoms([Move(0,1,1,1)])
    assert np.array_equal(array.matrix, np.array([[0,0,0],
                                                  [0,0,1],
                                                  [0,0,1]]).reshape(3,3,1))
    assert failed_moves == []
    assert flags == []
    assert move_time == 5e-5

    array.matrix = np.array([[0,1,1],
                             [1,0,0],
                             [0,0,1]]).reshape(3,3,1)
 
    error_model = UniformVacuumTweezerError(putdown_fail_rate=1, pickup_fail_rate=0)
    array = AtomArray(shape=[3,3], error_model=error_model, n_species=1)
    array.matrix = np.array([[0,1,1],
                             [1,0,0],
                             [0,0,1]]).reshape(3,3,1)
    # check that moving an empty spot fails
    [failed_moves, flags], move_time = array.move_atoms([Move(0,0,0,1)])
    assert np.array_equal(array.matrix, np.array([[0,1,1],
                                                  [1,0,0],
                                                  [0,0,1]]).reshape(3,3,1))
    assert failed_moves == [0]
    assert flags == [3]
    assert move_time == 5e-5

    # Check that destinations conflicts get sorted
    array = AtomArray(shape=[3,3], n_species=1)
    array.matrix = np.array([[0,1,0],
                             [1,0,1],
                             [0,0,1]]).reshape(3,3,1)
    moves = [Move(0,1,1,1), Move(1,0,1,1), Move(2,2,1,2)]
    [failed_moves, flags], move_time = array.move_atoms(moves)
    assert np.array_equal(array.matrix, np.array([[0,0,0],
                                                  [0,0,0],
                                                  [0,0,0]]).reshape(3,3,1))
    assert failed_moves == [0,1]
    assert flags == [3, 3]
    assert move_time == 5e-5

    error_model = UniformVacuumTweezerError(putdown_fail_rate=0, pickup_fail_rate=1)
    array = AtomArray(shape=[3,3], error_model=error_model, n_species=1)
    array.matrix = np.array([[0,1,0],
                             [1,0,1],
                             [0,0,1]]).reshape(3,3,1)
    moves = [Move(0,1,1,1), Move(1,0,1,1), Move(2,2,1,2)]
    [failed_moves, flags], move_time = array.move_atoms(moves)
    assert np.array_equal(array.matrix, np.array([[0,1,0],
                                                  [1,0,1],
                                                  [0,0,1]]).reshape(3,3,1))
    assert failed_moves == [0,1,2]
    assert flags == [1,1,1]
    assert move_time == 5e-5

def test_find_and_resolve_crossed_moves():
    array = AtomArray(shape=[3,3], n_species=1)
    array.matrix = np.array([[0,1,1],
                             [1,0,0],
                             [0,0,1]]).reshape(3,3,1)
    moves = [Move(0,1,0,2), Move(0,2,0,1)]
    moves, dup_move_idxs = array._find_and_resolve_crossed_moves(moves)
    for move in moves:
        assert move.failure_flag == 4
    assert dup_move_idxs == [0,1]
    assert np.array_equal(array.matrix, np.array([[0,0,0],
                                                  [1,0,0],
                                                  [0,0,1]]).reshape(3,3,1))

def test_find_and_resolve_same_dest():
    # check to make sure same destination errors work
    array = AtomArray(shape=[3,3], n_species=1)
    array.matrix = np.array([[0,1,1],
                             [1,0,0],
                             [0,0,1]]).reshape(3,3,1)
    moves = [Move(0,1,1,1), Move(1,0,1,1), Move(2,2,2,1)]
    moves, dup_move_idxs = array._find_and_resolve_same_dest_moves(moves)
    assert moves[0].failure_flag == 5
    assert moves[1].failure_flag == 5
    assert moves[2].failure_flag == 0
    assert dup_move_idxs == [0,1]
    assert np.array_equal(array.matrix, np.array([[0,0,1],
                                                  [0,0,0],
                                                  [0,0,1]]).reshape(3,3,1))

    array.matrix = np.array([[0,1,0],
                             [1,0,1],
                             [0,0,1]]).reshape(3,3,1)
    moves = [Move(0,1,1,1), Move(1,0,1,1), Move(1,2,1,1)]
    moves, dup_move_idxs = array._find_and_resolve_same_dest_moves(moves)
    for move in moves:
        assert move.failure_flag == 5
    assert dup_move_idxs == [0,1,2]
    assert np.array_equal(array.matrix, np.array([[0,0,0],
                                                  [0,0,0],
                                                  [0,0,1]]).reshape(3,3,1))

    array = AtomArray(shape=[3,3], n_species=1)
    array.matrix = np.array([[0,1,1],
                             [1,0,0],
                             [0,0,1]]).reshape(3,3,1)
    moves = [Move(0,1,1,1), Move(1,0,1,1), Move(2,2,2,1)]
    moves[1].failure_flag = 1
    moves, dup_move_idxs = array._find_and_resolve_same_dest_moves(moves)
    assert moves[0].failure_flag == 0
    assert moves[1].failure_flag == 1
    assert moves[2].failure_flag == 0
    assert dup_move_idxs == [0,1]
    assert np.array_equal(array.matrix, np.array([[0,1,1],
                                                  [1,0,0],
                                                  [0,0,1]]).reshape(3,3,1))

def test_evaluate_moves():
    array = AtomArray(shape=[3,3], n_species=1)
    array.matrix = np.array([[0,1,1],
                             [1,0,0],
                             [0,0,1]]).reshape(3,3,1)
    total_time, [N_parallel_moves, N_non_parallel_moves] = array.evaluate_moves([[Move(1,0,2,0), Move(0,1,1,1)], [Move(0,2,1,2)]])
    assert total_time == 1e-4
    assert N_parallel_moves == 2
    assert N_non_parallel_moves == 3
