import copy

import numpy as np

from atommover.utils.Move import Move
from atommover.utils.Tweezer import Tweezer, TweezerLossFlags


def test_simulate_move_sequence():
    moves = [Move(0, 0, 0, 1), Move(0, 1, 0, 2), Move(0, 2, 1, 2)]
    tweezer = Tweezer(moves=moves)
    matrix = np.array([[1, 0, 0], [0, 0, 0], [1, 1, 1]])
    move_time, num_moves, success_flag, error_flags = tweezer.simulate_move_sequence(
        matrix
    )
    assert move_time == float(
        3 * ((moves[0].distance * tweezer.array_spacing) / tweezer.speed)
    )
    assert num_moves == 3
    assert success_flag == 1
    assert error_flags == [0, 0, 0]


def test_make_move():
    moves = [Move(0, 0, 0, 1), Move(0, 1, 0, 2), Move(0, 2, 1, 2)]
    tweezer = Tweezer(moves=moves)
    matrix = np.array([[1, 0, 0], [0, 0, 0], [1, 1, 1]])
    for i in range(3):
        move, flag = tweezer.make_move(matrix, copy.deepcopy(matrix))
        assert move == moves[i]
        assert flag == TweezerLossFlags.SUCCESS
    assert np.array_equal(matrix, np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1]]))

    moves = [Move(0, 0, 0, 1), Move(0, 1, 0, 2), Move(0, 2, 1, 2)]
    tweezer = Tweezer(moves=moves)
    matrix = np.array([[1, 0, 0], [0, 0, 0], [1, 1, 1]])
    for i in range(10):
        try:
            move, flag = tweezer.make_move(matrix, copy.deepcopy(matrix))
            assert move == moves[i]
            assert flag == TweezerLossFlags.SUCCESS
        except:
            pass
    assert np.array_equal(matrix, np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1]]))

    # COLLISION_ERRORS flags currently not raised, but resulting array is correct
    moves = [Move(0, 0, 0, 1), Move(0, 1, 0, 2), Move(0, 2, 1, 2)]
    flags = [0, TweezerLossFlags.COLLISION_ERROR, TweezerLossFlags.PICKUP_ERROR]
    tweezer = Tweezer(moves=moves)
    matrix = np.array([[1, 0, 1], [0, 0, 0], [1, 1, 1]])
    for i in range(3):
        move, flag = tweezer.make_move(matrix, copy.deepcopy(matrix))
        assert move == moves[i]
        assert flag == flags[i]
    assert np.array_equal(matrix, np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]]))


def test_reset():
    moves = [Move(0, 0, 1, 0), Move(1, 0, 1, 0), Move(1, 0, 2, 0)]
    tweezer = Tweezer(moves=moves)
    matrix = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 1]])
    array = copy.deepcopy(matrix)
    for i in range(3):
        move, _ = tweezer.make_move(array, copy.deepcopy(array))
        assert move == moves[i]
    assert np.array_equal(array, np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]]))

    tweezer.reset()

    matrix = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 1]])
    array = copy.deepcopy(matrix)
    for i in range(3):
        move, _ = tweezer.make_move(array, copy.deepcopy(array))
        assert move == moves[i]
    assert np.array_equal(array, np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]]))
