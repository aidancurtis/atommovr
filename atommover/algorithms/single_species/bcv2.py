# Balance and Compact
from atommover.algorithms.algorithm import Algorithm
from atommover.algorithms.source.bc_new import bcv2
from atommover.atom_array import AtomArray


# Balance and Compact
class BCv2(Algorithm):
    """Implements the Balance and Compact algorithm, as originally described
    in [PRA 70, 040302(R) (2004)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.70.040302)

    Supported configurations: `Configurations.MIDDLE_FILL`"""

    def __repr__(self):
        return "Balance & Compact"

    def get_moves(self, atom_array: AtomArray, do_ejection: bool = False):
        if atom_array.n_species != 1:
            raise ValueError(
                f"Single-species algorithm cannot process atom array with {atom_array.n_species} species."
            )
        return bcv2(atom_array, do_ejection)
