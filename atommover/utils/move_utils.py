# Core functions and classes for moving atoms

import copy
from collections import Counter
from enum import IntEnum

import numpy as np

from atommover.move import Move
from atommover.utils.core import PhysicalParams


class MoveType(IntEnum):
    """
    Class to be used in conjunction with `move_atoms()`
    """

    ILLEGAL_MOVE = 0
    LEGAL_MOVE = 1
    EJECT_MOVE = 2
    NO_ATOM_TO_MOVE = 3


## AOD cmd functions ##

AOD_cmd_to_pos_shift = {"1": 0, "2": 1, "3": -1}


def get_move_list_from_AOD_cmds(horiz_AOD_cmds, vert_AOD_cmds):
    move_list = []
    for row_ind in range(len(vert_AOD_cmds)):
        row_cmd = int(vert_AOD_cmds[row_ind])
        if row_cmd == 0:
            pass
        else:
            row_shift = AOD_cmd_to_pos_shift["{}".format(row_cmd)]
            for col_ind in range(len(horiz_AOD_cmds)):
                col_cmd = int(horiz_AOD_cmds[col_ind])
                if col_cmd != 0 and col_cmd * row_cmd != 1:
                    # make a move
                    col_shift = AOD_cmd_to_pos_shift["{}".format(col_cmd)]
                    move_list.append(
                        Move(col_ind, row_ind, col_ind + col_shift, row_ind + row_shift)
                    )
    return move_list


def get_AOD_cmds_from_move_list(matrix, move_seq):
    row_num = len(matrix)
    col_num = len(matrix[0])
    horiz_AOD_cmds = np.zeros([row_num])
    vert_AOD_cmds = np.zeros([col_num])
    parallel_success_flag = True
    op_matrix = copy.deepcopy(matrix)

    # Generate AOD commands for a given row and column number
    for move in move_seq:
        # Chnage the status of vertical AOD commands
        if move.from_row > move.to_row:
            if vert_AOD_cmds[move.from_row] == 0:
                vert_AOD_cmds[move.from_row] = 3
            elif vert_AOD_cmds[move.from_row] != 3:
                parallel_success_flag = False
                break
        elif move.from_row < move.to_row:
            if vert_AOD_cmds[move.from_row] == 0:
                vert_AOD_cmds[move.from_row] = 2
            elif vert_AOD_cmds[move.from_row] != 2:
                parallel_success_flag = False
                break
        else:
            if vert_AOD_cmds[move.from_row] == 0:
                vert_AOD_cmds[move.from_row] = 1
            elif vert_AOD_cmds[move.from_row] != 1:
                parallel_success_flag = False
                break

        # Change the status of horizontal AOD commands
        if move.from_col > move.to_col:
            if horiz_AOD_cmds[move.from_col] == 0:
                horiz_AOD_cmds[move.from_col] = 3
            elif horiz_AOD_cmds[move.from_col] != 3:
                parallel_success_flag = False
                break
        elif move.from_col < move.to_col:
            if horiz_AOD_cmds[move.from_col] == 0:
                horiz_AOD_cmds[move.from_col] = 2
            elif horiz_AOD_cmds[move.from_col] != 2:
                parallel_success_flag = False
                break
        else:
            if horiz_AOD_cmds[move.from_col] == 0:
                horiz_AOD_cmds[move.from_col] = 1
            elif horiz_AOD_cmds[move.from_col] != 1:
                parallel_success_flag = False
                break

        # Check if there is an atom from source position
        if op_matrix[move.from_row][move.from_col] == 0:
            parallel_success_flag = False
            break

    if parallel_success_flag:
        move_list = get_move_list_from_AOD_cmds(vert_AOD_cmds, horiz_AOD_cmds)
        matrix_from_AOD, _ = move_atoms(copy.deepcopy(matrix), move_list)
        matrix_from_seq, _ = move_atoms(copy.deepcopy(matrix), move_seq)

        if not np.array_equal(matrix_from_AOD, matrix_from_seq):
            parallel_success_flag = False

    return horiz_AOD_cmds, vert_AOD_cmds, parallel_success_flag
