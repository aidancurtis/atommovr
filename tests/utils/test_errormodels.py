import numpy as np

from atommover.move import Move
from atommover.utils.errormodels import UniformVacuumTweezerError, ZeroNoise


def test_zero_noise():
    zero_noise = ZeroNoise()
    assert zero_noise.putdown_time == 0
    assert zero_noise.pickup_time == 0

    # test get_move_errors
    matrix = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]]).reshape(3, 3, 1)
    moves = [Move(2, 2, 2, 1), Move(1, 0, 2, 0), Move(0, 2, 1, 2)]
    moves = zero_noise.get_move_errors(state=matrix, moves=moves)
    for move in moves:
        assert move.failure_flag == 0

    # test get_atom_loss
    matrix = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]]).reshape(3, 3, 1)
    new_state, loss_flag = zero_noise.get_atom_loss(state=matrix, evolution_time=1e6)
    assert loss_flag == 0
    assert np.array_equal(new_state, matrix)


def test_uniform_tweezer_error():
    uniform_error = UniformVacuumTweezerError(pickup_fail_rate=0, putdown_fail_rate=0)
    assert uniform_error.putdown_time == 0
    assert uniform_error.pickup_time == 0
    assert uniform_error.pickup_fail_rate == 0.00
    assert uniform_error.pickup_fail_rate == 0.00
    assert uniform_error.lifetime == 30

    # test get_move_errors
    matrix = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]]).reshape(3, 3, 1)
    moves = [Move(2, 2, 2, 1), Move(1, 0, 2, 0), Move(0, 2, 1, 2)]
    moves = uniform_error.get_move_errors(state=matrix, moves=moves)
    for move in moves:
        assert move.failure_flag == 0

    uniform_error.pickup_fail_rate = 1
    uniform_error.putdown_fail_rate = 0
    matrix = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]]).reshape(3, 3, 1)
    moves = [Move(2, 2, 2, 1), Move(1, 0, 2, 0), Move(0, 2, 1, 2)]
    moves = uniform_error.get_move_errors(state=matrix, moves=moves)
    for move in moves:
        assert move.failure_flag == 1

    uniform_error.pickup_fail_rate = 0
    uniform_error.putdown_fail_rate = 1
    matrix = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]]).reshape(3, 3, 1)
    moves = [Move(2, 2, 2, 1), Move(1, 0, 2, 0), Move(0, 2, 1, 2)]
    moves = uniform_error.get_move_errors(state=matrix, moves=moves)
    for move in moves:
        assert move.failure_flag == 2

    # test get_atom_loss
    matrix = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]]).reshape(3, 3, 1)
    new_state, loss_flag = uniform_error.get_atom_loss(state=matrix, evolution_time=1e6)

    assert loss_flag == 1
    assert np.array_equal(new_state, np.zeros(shape=(3, 3, 1)))
