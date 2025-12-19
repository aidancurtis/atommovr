import numpy as np

from atommover.algorithms.algorithm import Algorithm
from atommover.algorithms.source.hungarian_works import (
    parallel_hungarian_algorithm_works,
)
from atommover.atom_array import AtomArray
from atommover.move import Move


class ParallelHungarian(Algorithm):
    """A variant on the Hungarian matching algorithm that parallelizes the moves
    instead of executing them sequentially (one by one).

    Supported configurations: all."""

    def __repr__(self):
        return "Parallel Hungarian"

    def get_moves(
        self,
        atom_array: AtomArray,
        do_ejection: bool = False,
        final_size: list = [],
        round_lim: int = 0,
    ) -> tuple[np.ndarray, list[list[list[Move]]], bool]:
        if atom_array.n_species != 1:
            raise ValueError(
                f"Single-species algorithm cannot process atom array with {atom_array.n_species} species."
            )
        if round_lim == 0:
            round_lim = int(np.sum(atom_array.target))

        return parallel_hungarian_algorithm_works(
            atom_array.matrix, atom_array.target, do_ejection, final_size, round_lim
        )
