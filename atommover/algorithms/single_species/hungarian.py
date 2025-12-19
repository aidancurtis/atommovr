# Hungarian
from atommover.algorithms.algorithm import Algorithm
from atommover.algorithms.source.hungarian_works import hungarian_algorithm_works
from atommover.atom_array import AtomArray


class Hungarian(Algorithm):
    """Implements the Hungarian matching algorithm, which generates a cost
    matrix mapping available atoms to the target spots, and solves the
    linear assignment problem to find an efficient set of moves.

    Supported configurations: all."""

    def __repr__(self):
        return "Hungarian"

    def get_moves(self, atom_array: AtomArray, do_ejection: bool = False):
        if atom_array.n_species != 1:
            raise ValueError(
                f"Single-species algorithm cannot process atom array with {atom_array.n_species} species."
            )

        return hungarian_algorithm_works(
            atom_array.matrix[:, :, 0], atom_array.target, do_ejection
        )
