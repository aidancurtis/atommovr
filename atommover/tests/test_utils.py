# Tests for utility functions (in progress)

import numpy as np

from atommover.utils.AtomArray import AtomArray
from atommover.utils.Move import Move
from atommover.utils.move_utils import (
    get_AOD_cmds_from_move_list,
    get_move_list_from_AOD_cmds,
)


def test_move():
    move = Move(1, 0, 2, 3)
    assert move.move_str() == "(1, 0) -> (2, 3)"
    assert move.to_col == 3


def test_AtomArray():
    ## move_atoms ##

    # 1. Single species
    array = AtomArray(shape=[3, 3])
    array.matrix = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]]).reshape(3, 3, 1)
    # checking that moves get row/col assignments correct
    _ = array.move_atoms(moves=[Move(0, 1, 1, 1)])
    assert np.array_equal(
        array.matrix, np.array([[0, 0, 1], [1, 1, 0], [0, 0, 1]]).reshape(3, 3, 1)
    )

    # checking that moves of sites next to one another work properly
    _ = array.move_atoms(moves=[Move(1, 0, 1, 1), Move(1, 1, 1, 2)])
    assert np.array_equal(
        array.matrix, np.array([[0, 0, 1], [0, 1, 1], [0, 0, 1]]).reshape(3, 3, 1)
    )
    # checking that atoms are not used in two moves
    _ = array.move_atoms(moves=[Move(2, 2, 2, 1), Move(2, 1, 2, 0)])
    assert np.array_equal(
        array.matrix, np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0]]).reshape(3, 3, 1)
    )
    # checking that collisions expel both atoms
    _ = array.move_atoms(moves=[Move(1, 2, 0, 2)])
    assert np.array_equal(
        array.matrix, np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]]).reshape(3, 3, 1)
    )
    # checking that crossed tweezers expel atoms
    _ = array.move_atoms(moves=[Move(1, 1, 2, 0), Move(2, 1, 1, 0)])
    assert np.array_equal(
        array.matrix, np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).reshape(3, 3, 1)
    )

    # 2. Dual species
    # TODO


def test_move_list_AOD_conversion():
    move = [Move(3, 5, 4, 6)]
    horiz_AOD_cmds, vert_AOD_cmds, parallel_success_flag = get_AOD_cmds_from_move_list(
        move
    )
    move_list = get_move_list_from_AOD_cmds(horiz_AOD_cmds, vert_AOD_cmds)
    assert move[0].to_col == move_list[0].to_col
    assert move[0].to_row == move_list[0].to_row
    assert move[0].from_col == move_list[0].from_col
    assert move[0].from_row == move_list[0].from_row
