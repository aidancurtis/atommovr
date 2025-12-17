import copy
from enum import IntEnum

import numpy as np
from atommover.utils import Move


class TweezerLossFlags(IntEnum):
    SUCCESS = 0
    PICKUP_ERROR = 1
    PUTDOWN_ERROR = 2
    VACUUM_ERROR = 3
    COLLISION_ERROR = 4
    NO_ATOM_ERROR = 5


class Tweezer:
    """
    Represents a moving tweezer

    Attributes:
        speed (int): speed of tweezer [m/s]
        array_spacing (float): spacing between static tweezers [meters]
        vacuum_lifetime (float): vacuum lifetime of array [seconds]
        pickup_error_prob (float): probability of pickup error
        putdown_error_prob (float): probability of putdown error
        moves: list[Move]: list of moves
    """

    def __init__(
        self,
        speed: float = 1e6,
        array_spacing: float = 5e-6,
        vacuum_lifetime: float = 1e6,
        pickup_error_prob: float = 0,
        putdown_error_prob: float = 0,
        moves: list[Move] = [],
    ):
        self.speed = speed
        self.array_spacing = array_spacing
        self.vacuum_lifetime = vacuum_lifetime
        if not (0 <= pickup_error_prob <= 1 and 0 <= putdown_error_prob <= 1):
            raise ValueError(
                "pickup_error_prob and putdown_error_prob must between 0 and 1"
            )
        self.pickup_error_prob = pickup_error_prob
        self.putdown_error_prob = putdown_error_prob
        self.moves = moves
        self.move_num = 0
        self.move_time = 0
        self.occupied = False

    def __str__(self):
        moves = []
        for move in self.moves:
            moves.append((move.from_row, move.from_col))
        moves.append((self.moves[-1].to_row, self.moves[-1].to_col))
        return " -> ".join(str(move) for move in moves)

    def simulate_move_sequence(
        self, array: np.ndarray
    ) -> tuple[float, int, bool, list[int]]:
        """
        Simulate a tweezer move sequence

        Returns:
            tuple(float, int, bool, int): (total_move_time, num_moves, success_flag, error_flags)
        """
        success_flag = True
        error_flags = [0 for _ in self.moves]
        if not self.moves:
            raise ValueError("No moves")

        pickup_error = np.random.random() < self.pickup_error_prob
        if pickup_error:
            success_flag = False
            error_flags[0] = TweezerLossFlags.PICKUP_ERROR
            atom_state = 0
        else:
            atom_state = array[self.moves[0].from_row, self.moves[0].from_col]
            array[self.moves[0].from_row, self.moves[0].from_col] = 0

            # Check for collision
            if array[self.moves[0].to_row, self.moves[0].to_col] == 0:
                array[self.moves[0].to_row, self.moves[0].to_col] = atom_state
            else:
                atom_state = 0
                error_flags[0] = TweezerLossFlags.COLLISION_ERROR
                array[self.moves[0].to_row, self.moves[0].to_col] = 0

        move_time = 0
        pickup_flag = True
        for i in range(1, len(self.moves)):
            move = self.moves[i]
            if pickup_flag:
                atom_state = copy.deepcopy(array[move.from_row, move.from_col])
                array[move.from_row, move.from_col] = 0

            pickup_flag = True

            loss_prob = 1 - np.exp(-move_time / self.vacuum_lifetime)
            atom_loss_error = np.random.random() < loss_prob
            if atom_loss_error:
                error_flags[i] = TweezerLossFlags.VACUUM_ERROR
                atom_state = 0

            # if next spot is empty, fill it with tweezer
            if array[move.to_row, move.to_col] == 0:
                array[move.to_row, move.to_col] = atom_state
            else:
                # if next spot is occupied and tweezer is occupied, clear both
                if atom_state:
                    success_flag = False
                    error_flags[i] = TweezerLossFlags.COLLISION_ERROR
                    array[move.to_row, move.to_col] = 0
                # if next spot is occupied and tweezer is empty
                else:
                    # calculate whether tweezer is picked up
                    pickup_flag = np.random.random() > self.pickup_error_prob

            move_time += (move.distance * self.array_spacing) / self.speed

        putdown_error = 1 if np.random.random() < self.putdown_error_prob else 0
        if (
            putdown_error
            and array[self.moves[-1].to_row, self.moves[-1].to_col] == 1
            and error_flags[-1] == 0
        ):
            success_flag = False
            error_flags[-1] = TweezerLossFlags.PUTDOWN_ERROR
            array[self.moves[-1].to_row, self.moves[-1].to_col] = 0

        return move_time, len(self.moves), success_flag, error_flags

    def make_move(
        self, array: np.ndarray, past_array: np.ndarray
    ) -> tuple[Move, TweezerLossFlags]:
        """
        Make self.moves[self.move_num] move

        Return:
            tuple(Move, bool): (move, error_flag)
        """
        if self.move_num >= len(self.moves):
            raise ValueError(f"All tweezer moves have been made")

        index = self.move_num
        move = self.moves[index]
        self.move_time += (self.moves[index].distance * self.array_spacing) / self.speed
        self.move_num += 1

        if move.failure_flag == TweezerLossFlags.PICKUP_ERROR:
            self.occupied = False
            self.move_time = 0
            return move, TweezerLossFlags.PICKUP_ERROR

        if past_array[move.from_row, move.from_col] == 0:
            self.occupied = False
            self.move_time = 0
            return move, TweezerLossFlags.PICKUP_ERROR

        # at this point this tweezer is on and there is an atom in src
        # check whether this is the first move or that the tweezer is not previously occupied
        if index == 0 or not self.occupied:
            pickup_error = np.random.random() < self.pickup_error_prob
            if pickup_error:
                self.occupied = False
                self.move_time = 0
                return move, TweezerLossFlags.NO_ATOM_ERROR
            else:
                self.occupied = True

        array[move.to_row, move.to_col] += 1
        array[move.from_row, move.from_col] -= 1

        loss_prob = 1 - np.exp(-self.move_time / self.vacuum_lifetime)
        atom_loss_error = np.random.random() < loss_prob
        if atom_loss_error:
            self.occupied = False
            self.move_time = 0
            array[move.to_row, move.to_col] -= 1
            return move, TweezerLossFlags.VACUUM_ERROR

        if move.failure_flag == TweezerLossFlags.COLLISION_ERROR:
            self.occupied = False
            self.move_time = 0
            array[move.to_row, move.to_col] -= 1
            return move, TweezerLossFlags.COLLISION_ERROR

        if index == len(self.moves) - 1:
            putdown_error = np.random.random() < self.putdown_error_prob
            if putdown_error:
                self.occupied = False
                self.move_time = 0
                array[move.to_row, move.to_col] -= 1
                return move, TweezerLossFlags.PUTDOWN_ERROR

        return move, TweezerLossFlags.SUCCESS

    def reset(self):
        self.occupied = False
        self.move_num = 0
        self.move_time = 0


# if __name__ == "__main__":
#     array = np.array([[0, 0, 1], [1, 0, 0], [1, 1, 0]])
#     print(array)
#     moves = [Move(1, 0, 2, 0), Move(2, 0, 2, 1), Move(2, 1, 1, 1)]
#     print(moves)
#     print()
#     tweezer = Tweezer(
#         speed=1e6,
#         array_spacing=5e-6,
#         vacuum_lifetime=1e8,
#         pickup_error_prob=0,
#         putdown_error_prob=0,
#         moves=moves,
#     )
#     # total_time, n_moves, success_flag, error_flags = tweezer.make_move_sequence(array)
#     move, flag = tweezer.make_move(array, on=True)
#     print(array)
#     print(f"move: {move}")
#     print(f"flag: {flag}")
#
#     move, flag = tweezer.make_move(array, on=False)
#     print(array)
#     print(f"move: {move}")
#     print(f"flag: {flag}")
#
#     move, flag = tweezer.make_move(array, on=True)
#     print(array)
#     print(f"move: {move}")
#     print(f"flag: {flag}")
#
#     # print(f"total_time: {total_time}")
#     # print(f"n_moves: {n_moves}")
#     # print(f"success_flag: {success_flag}")
#     # print(f"error_flags: {error_flags}")
