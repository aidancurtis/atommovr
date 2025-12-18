# Core object for simulating noise and loss processes

import numpy as np

from atommover.move import Move


class ErrorModel:
    """
    Generic object for modelling error processes during rearrangement.
    (e.g. atom loss due to vacuum collisions or heating from tweezers,
    failure to pick up or put down atoms in tweezer handoff, etc.).

    Takes as input an AtomArray object representing the current state
    of the array and containing information about physical parameters
    (such as AOD move speed, vacuum-limited lifetime), and a list of
    Move objects representing moves to execute.

    Returns as output the modified AtomArray object after the moves
    have been applied, and two flags, MoveFailureFlag and AtomLossFlag,
    which indicate if any moves have failed or if any atoms have been
    lost, respectively.

    ## Parameters:
    putdown_time : float, optional (default = 0)
        the time it takes to set atoms down, in s.
    pickup_time : float, optional (default = 0)
        the time it takes to pick atoms up, in s.
    """

    def __init__(self, putdown_time: float = 0, pickup_time: float = 0):
        self.name = "Generic ErrorModel object"
        self.putdown_time = 0  # seconds
        self.pickup_time = 0  # seconds

    def __repr__(self) -> str:
        return self.name

    def get_move_errors(self, state: np.ndarray, moves: list[Move]):
        """
        Given a set of moves and the current state, assigns
        an attribute `failure_flag` to the move.
        - If the move suceeds, `move_failure_flag = 0`.
        - If the move fails but the atom remains in its original position,
        `move.failure_flag = 1`.
        - If the move fails and the atom is also ejected,
        `move.failure_flag = 2`.

        ## Parameters
        state : np.ndarray
            the current state of the array that the moves are being applied to.
        moves : list[Move]
            the moves to be executed on the state in parallel.

        ## Returns
        moves_w_flags : list[Move]
            a list of the same moves, but where each move now has an attribute
            `failure_flag` added.
        """

        return moves

    def get_atom_loss(
        self, state: np.ndarray, evolution_time: float, n_species: int = 1
    ) -> tuple[np.ndarray, bool]:
        """
        Simulates a general loss process and returns the modified state of
        the array.

        ## Parameters
        state : np.ndarray
            the current state of the array
        evolution_time : float
            the time it took to execute the last time of moves, or the time
            since the error syndrome was last simulated.
        n_species : int, optional (default = 1)
            the number of atomic species in the array.
        """

        return state

    def evaluate_moves(
        self, state: np.ndarray, moves: list[Move]
    ) -> tuple[np.ndarray, str, str]:
        """
        Given an AtomArray object representing the current state of the
        array, and a set of moves to execute in *parallel*, applies the
        moves to the array and returns the resulting state after
        error processes have occured, as well as flags to indicate the
        presence of any errors.

        ## Parameters
        state: AtomArray.
            current state of the atom array.
        moves: list[Move, Move, ...]
            list of parallel moves to execute.

        ## Returns
        mod_state: AtomArray
            modified state object.
        MoveFailureFlag: bool
            1 if any move has failed, 0 if not.
        AtomLossFlag: bool
            1 if any atom has been lost, 0 if not.
        """
        return state, False, False
