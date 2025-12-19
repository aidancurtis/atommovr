from atommover.algorithms.algorithm import Algorithm
from atommover.algorithms.source.generalized_balance import generalized_balance
from atommover.atom_array import AtomArray


# Generalized Balance
class GeneralizedBalance(Algorithm):
    """Implements the generalized balance algorithm, which alternatively operates
    row balance and column balance algorithms, as originally described by Bo-Yu
    and Nikhil in the Bernien lab meeting GM 268.

    Supported configurations: all."""

    def __repr__(self):
        return "Generalized Balance"

    def get_moves(self, atom_array: AtomArray, do_ejection: bool = False):
        if atom_array.n_species != 1:
            raise ValueError(
                f"Single-species algorithm cannot process atom array with {atom_array.n_species} species."
            )
        return generalized_balance(
            atom_array.matrix[:, :, 0], atom_array.target, do_ejection
        )
