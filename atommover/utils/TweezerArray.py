import copy
import math
import random
from collections import Counter

import numpy as np
from atommover.utils import Move
from atommover.utils.Tweezer import Tweezer


class TweezerArray:
    """
    A parent class representing a moving tweezer array
    """

    def __init__(
        self,
        tweezers: list[Tweezer],
        speed: float = 1e6,
        array_spacing: float = 5e-6,
        pickup_time: float = 1e-6,
        putdown_time: float = 1e-6,
    ):
        self.tweezers: list[Tweezer] = tweezers
        self.move_list: list[list[Move]] = []
        self.max_moves = len(move_list)
        self.speed = speed
        self.array_spacing = array_spacing
        self.pickup_time = pickup_time
        self.putdown_time = putdown_time

    def _get_duplicate_vals_from_list(self, l: list) -> list:
        """
        Returns duplicate values in list
        """
        return [k for k, v in Counter(l).items() if v > 1]

    def _set_move_failure_flags(self):
        """
        Sets move failure flag for moves that collide
        """
        for parallel_move_seq in self.move_list:
            midpoints_seq = []
            for move in parallel_move_seq:
                midpoints_seq.append((move.midx, move.midy))
            keys = self._get_duplicate_vals_from_list(midpoints_seq)

            for move in parallel_move_seq:
                idx = (move.midx, move.midy)
                if idx in keys:
                    move.failure_flag = 4

    def _get_tweezer_off_list(self) -> list[list[int]]:
        """
        Returns list of tweezers ids to turn off at during each move sequence
        based on tweezers that have the same starting location

        Returns:
            off_list (list[list[int]]): list of list of tweezer ids to turn off
        """
        same = []
        for parallel_move_seq in self.move_list:
            same_seq = []
            for move in parallel_move_seq:
                same_seq.append((move.from_row, move.from_col))
            keys = self._get_duplicate_vals_from_list(same_seq)
            ls = {}
            for move in parallel_move_seq:
                idx = (move.from_row, move.from_col)
                if idx in keys:
                    try:
                        ls[idx].append(move.tweezer_id)
                    except:
                        ls[idx] = [move.tweezer_id]
            same.append(ls)

        off_list = []
        for idx, same_val_dict in enumerate(same):
            off_list.append([])
            for lst in same_val_dict.values():
                lst.pop(random.randrange(len(lst)))
                off_list[idx].extend(lst)

        return off_list

    def move_atoms(self, array: np.ndarray) -> tuple[float, int, int]:
        self._set_move_failure_flags()
        off_list = self._get_tweezer_off_list()

        n_parallel_moves = 0
        n_total_moves = 0
        for i in range(len(self.move_list)):
            past_array = copy.deepcopy(array)
            move_in_seq = 0
            for idx, tweezer in enumerate(self.tweezers):
                try:
                    on = idx not in off_list[i]
                    move, flag = tweezer.make_move(array, past_array, on=on)
                    move_in_seq += 1
                except:
                    pass

            array[np.where(array > 1)] = 0
            array[np.where(array < 0)] = 0

            n_parallel_moves += 1
            n_total_moves += move_in_seq

        total_time = (
            self.pickup_time
            + (n_parallel_moves * self.array_spacing) / self.speed
            + self.putdown_time
        )

        return total_time, n_parallel_moves, n_total_moves


if __name__ == "__main__":
    array = np.array([[1, 0, 0], [0, 0, 1], [1, 1, 0]])
    print(array)
    # 1 0 0
    # 0 0 1
    # 1 1 0
    moves1 = [Move(0, 0, 1, 0), Move(1, 0, 2, 0), Move(2, 0, 1, 0)]
    tw1 = Tweezer(moves=moves1, pickup_error_prob=0)
    print(tw1)

    moves2 = [Move(1, 2, 1, 1), Move(1, 1, 1, 0), Move(1, 0, 0, 0)]
    tw2 = Tweezer(moves=moves2, pickup_error_prob=0)
    print(tw2)

    moves3 = [Move(2, 1, 2, 2), Move(2, 2, 1, 2), Move(1, 2, 0, 2)]
    tw3 = Tweezer(moves=moves3, pickup_error_prob=0)
    print(tw3)

    # moves4 = [Move(1, 0, 1, 0), Move(2, 0, 2, 0), Move(1, 1, 1, 1)]
    # tw4 = Tweezer(moves=moves4)
    #
    move_list = []
    for i in range(3):
        moves1[i].tweezer_id = 0
        moves2[i].tweezer_id = 1
        moves3[i].tweezer_id = 2
        move_list.append([moves1[i], moves2[i], moves3[i]])

        # moves4[i].tweezer_id = 4
        # move_list.append([moves1[i], moves2[i], moves3[i], moves4[i]])

    t_array = TweezerArray([tw1, tw2, tw3])
    t_array.move_list = move_list
    t_array.move_atoms(array)
    print(array)
