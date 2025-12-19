from atommover.algorithms.algorithm import Algorithm
from atommover.algorithms.source.balance_compact import balance_and_compact
from atommover.atom_array import AtomArray


# Balance and Compact
class BalanceAndCompact(Algorithm):
    """NOTE: we recommend that you use the (faster) BCv2 algorithm.
    This is an older version that we used to make Fig. 2 in the paper.

    A slow implementation of the Balance and Compact algorithm, as originally described
    in [PRA 70, 040302(R) (2004)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.70.040302)

    Supported configurations: `Configurations.MIDDLE_FILL`"""

    def __repr__(self):
        return "Balance & Compact (slow)"

    def get_moves(self, atom_array: AtomArray, do_ejection: bool = False):
        if atom_array.n_species != 1:
            raise ValueError(
                f"Single-species algorithm cannot process atom array with {atom_array.n_species} species."
            )
        return balance_and_compact(
            atom_array.matrix[:, :, 0], atom_array.target, do_ejection
        )
