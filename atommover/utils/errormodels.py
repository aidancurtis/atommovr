# Collection of ErrorModel objects representing various loss processes

import copy
import random

import numpy as np

from atommover.error_model import ErrorModel
from atommover.move import Move
from atommover.utils.core import atom_loss, atom_loss_dual


class ZeroNoise(ErrorModel):
    """
    Simulates errorless rearrangement (assumes perfect tweezers
    and an infinitely long vacuum-limited lifetime).
    """

    def __init__(self):
        self.name = "ZeroNoise"
        self.putdown_time = 0  # seconds
        self.pickup_time = 0  # seconds

    def __repr__(self) -> str:
        return self.name

    def get_move_errors(self, state: np.ndarray, moves: list[Move]) -> list:
        """
        Given a set of moves and the current state, assigns
        an attribute `failure_flag` to the move.
        - If the move suceeds, `move_failure_flag = 0`.
        - If the move fails but the atom remains in its original position,
        `move.failure_flag = 1`.
        - If the move fails and the atom is also ejected,
        `move_failure_flag = 2`.

        For this noise model, it sets all failure flags to 0.
        """
        for move in moves:
            move.failure_flag = 0

        return moves

    def get_atom_loss(
        self, state: np.ndarray, evolution_time: float, n_species: int = 1
    ) -> tuple[np.ndarray, bool]:
        """
        Given the current state of the atom array, applies any general loss process
        over the period Δt = evolution_time.

        For this error model, it just returns the same state.

        ## Parameters
        state : np.ndarray
            the current state of the atom array.
        evolution_time : float
            the time over which we calculate the loss process (usually the time
            for a single move or set of parallel moves).
        - n_species : int, optional (default = 1)
            the number of atomic species (single, dual).

        ## Returns
        new_state : np.ndarray
            the state after the loss process.
        loss_flag : bool
            1 if any atom loss occurred, 0 if not.
        """
        loss_flag = 0
        new_state = copy.deepcopy(state)
        return new_state, loss_flag


class UniformVacuumTweezerError(ErrorModel):
    """
    Considers atom loss due to imperfect vacuum
    (i.e. collisions with background gas particles)
    and uniform tweezer failure rates.

    ## Parameters
     - `pickup_fail_rate` (optional): float, between 0 and 1.
     Probability that an atom to be moved will be not picked up by the moving tweezer
    (in this case, the atom is not lost but just stays in its original spot). Default
    value is 0.01 (1%).
     - `putdown_fail_rate` (optional): float, between 0 and 1.
    Probability that an atom to be moved will be picked up by the moving tweezer, but
    will be subsequently lost in the transfer to the new tweezer. Default value is
    0.01 (1%).
     - `lifetime` (optional): float.
    Vacuum limited lifetime of an individual atom (assumed to be uniform for all atoms),
    in seconds. Default value is 30.
    """

    def __init__(
        self,
        pickup_fail_rate: float = 0.01,
        putdown_fail_rate: float = 0.01,
        lifetime: float = 30,
        pickup_time: float = 0,
        putdown_time: float = 0,
    ):
        self.name = "UniformVacuumTweezerError"
        self.pickup_fail_rate = pickup_fail_rate
        self.putdown_fail_rate = putdown_fail_rate
        self.lifetime = lifetime
        self.pickup_time = pickup_time
        self.putdown_time = putdown_time

    def __repr__(self) -> str:
        return self.name

    def get_move_errors(self, state: np.ndarray, moves: list[Move]) -> list[Move]:
        """
        Given a set of moves and the current state, assigns
        an attribute `failure_flag` to the move.
        - If the move suceeds, `move_failure_flag = 0`.
        - If the move fails but the atom remains in its original position,
        `move.failure_flag = 1`.
        - If the move fails and the atom is also ejected,
        `move_failure_flag = 2`.

        In this error model, we uniformly sample from a probability distribution
        specified by the class attributes `pickup_fail_rate` and
        `putdown_fail_rate`.
        """

        move_fails = random.choices(
            [0, 1, 2],
            weights=[
                1 - self.pickup_fail_rate - self.putdown_fail_rate,
                self.pickup_fail_rate,
                self.putdown_fail_rate,
            ],
            k=len(moves),
        )
        for move_index, move in enumerate(moves):
            move.failure_flag = move_fails[move_index]
        return moves

    def get_atom_loss(
        self, state: np.ndarray, evolution_time: float, n_species: int = 1
    ) -> tuple[np.ndarray, bool]:
        """
        Given the current state of the atom array, applies any general loss process
        over the period Δt = evolution_time.

        For this error model, we consider uniform loss from background gas particles
        knocking atoms out of their traps.

        ## Parameters
        - state (np.ndarray). The current state of the atom array.
        - evolution_time (float). The time over which we calculate the loss process
        (usually the time for a single move).
        - n_species (int, must be 1 or 2). The number of atomic species (single, dual).

        ## Returns
        - new_state (np.ndarray). The state after the loss process.
        - loss_flag (bool). 1 if any atom loss occurred, 0 if not.
        """
        evolution_time = evolution_time
        if n_species == 1:
            new_state, loss_flag = atom_loss(state, evolution_time, self.lifetime)
        elif n_species == 2:
            new_state, loss_flag = atom_loss_dual(state, evolution_time, self.lifetime)
        return new_state, loss_flag

    # def evaluate_moves(self, state: AtomArray, moves: list[Move]) -> tuple[AtomArray, str, str]:
    #     """
    #     Given an AtomArray object representing the current state of the
    #     array, and a set of moves to execute in *parallel*, applies the
    #     moves to the array and returns the resulting state after
    #     error processes have occured, as well as flags to indicate the
    #     presence of any errors.

    #     Inputs:
    #     - `state`: `AtomArray`. Current state of the atom array.
    #     - `moves`: list[`Move`, `Move`, ...]. List of parallel moves to execute.

    #     Outputs:
    #     - `state`: `AtomArray`. Modified state object.
    #     - `MoveFailureFlag`: bool. 1 if any move has failed, 0 if not.
    #     - `AtomLossFlag`: bool. 1 if any atom has been lost, 0 if not.
    #     """
    #     MoveFailureFlag = 0
    #     AtomLossFlag = 0

    #     for move in moves:
    #         pass

    #     return state, MoveFailureFlag, AtomLossFlag
