# Base object to represent atom arrays

import copy
import math
import random

import numpy as np

from atommover.animation import dual_species_image, single_species_image
from atommover.move import Move
from atommover.tweezer_array import TweezerArrayModel
from atommover.utils.core import (
    ArrayGeometry,
    Configurations,
    PhysicalParams,
    generate_middle_fifty,
    random_loading,
)
from atommover.utils.customize import SPECIES1NAME, SPECIES2NAME


class AtomArray:
    """
    Base object representing the state of the atom array.

    ## Parameters
    shape : list[int,int]
        the number of columns and rows in the atom array.
    n_species : int
        whether the atom array is single- or dual-species. must be in [1,2].
    params : PhysicalParams
        the physical parameters describing tweezer and array properties.
    geom : ArrayGeometry
        the geometry of the array.

    ## Example usage
    ```
    n_cols, n_rows = 10,10
    n_species = 1
    tweezer_array = AtomArray([n_cols, n_rows], n_species)
    ```
    """

    def __init__(
        self,
        shape: list = [10, 10],
        n_species: int = 1,
        params: PhysicalParams = PhysicalParams(),
        geom: ArrayGeometry = ArrayGeometry.RECTANGULAR,
    ):
        self.geom = geom
        if n_species in [1, 2] and type(n_species) == int:
            self.n_species = n_species
        else:
            raise ValueError(
                f"Invalid entry for parameter `n_species`: {n_species}. The simulator only supports single and dual species arrays. "
            )
        self.shape = shape
        self.params = params

        self.matrix: np.ndarray = np.zeros(
            [self.shape[0], self.shape[1], self.n_species]
        )
        self.target: np.ndarray = np.zeros(
            [self.shape[0], self.shape[1], self.n_species]
        )
        self.target_Rb: np.ndarray = np.zeros([self.shape[0], self.shape[1]])
        self.target_Cs: np.ndarray = np.zeros([self.shape[0], self.shape[1]])

    def __setattr__(self, key, value):
        if key == "shape":
            self.matrix = np.zeros([value[0], value[1], self.n_species])
            self.target = np.zeros([value[0], value[1], self.n_species])
            self.target_Rb = np.zeros([value[0], value[1]])
            self.target_Cs = np.zeros([value[0], value[1]])
            # (Optional) run any custom logic here
        # Always delegate to superclass to avoid recursion
        super().__setattr__(key, value)

    def load_tweezers(self):
        """
        Simulates uniform stochastic loading for single- or dual-species atom arrays.
        Loading probability (default 60%) can be set with `AtomArray.params.loading_prob`
        """
        if self.n_species == 1:
            self.matrix[:, :, :] = random_loading(
                self.shape, self.params.loading_prob
            ).reshape(self.shape[0], self.shape[1], 1)
        if self.n_species == 2:
            dual_species_prob = 2 - 2 * math.sqrt(1 - self.params.loading_prob)
            self.matrix[:, :, 0] = random_loading(self.shape, dual_species_prob / 2)
            self.matrix[:, :, 1] = random_loading(self.shape, dual_species_prob / 2)

            # Randomly leave one atom if there are two atoms share the same (x,y) coordinate
            for i in range(len(self.matrix)):
                for j in range(len(self.matrix[0])):
                    if self.matrix[i][j][0] == 1 and self.matrix[i][j][1] == 1:
                        random_index = random.randint(0, 1)
                        self.matrix[i][j][random_index] = 0

        self.last_loaded_config = copy.deepcopy(self.matrix)

    def generate_target(
        self,
        pattern: Configurations = Configurations.CHECKERBOARD,
        middle_size: list = [],
        occupation_prob: float = 0.5,
    ):
        if self.n_species == 1:
            self._generate_single_species_target(
                pattern, middle_size=middle_size, occupation_prob=occupation_prob
            )
        elif self.n_species == 2:
            self._generate_dual_species_target(
                pattern, middle_size=middle_size, occupation_prob=occupation_prob
            )
        else:
            raise ValueError(
                f"Unrecognized entry '{self.n_species}' for parameter `n_species`. The simulator only supports single and dual species arrays."
            )

    def _generate_single_species_target(
        self,
        pattern: Configurations = Configurations.MIDDLE_FILL,
        middle_size: list = [],
        occupation_prob: float = 0.5,
    ):
        """
        A function for generating common target configurations,
        such as checkerboard, zebra stripes, and middle fill.
        """
        array = np.zeros([self.shape[0], self.shape[1]])

        if len(middle_size) == 0:
            middle_size = generate_middle_fifty(self.shape[0], occupation_prob)

        if pattern == Configurations.ZEBRA_HORIZONTAL:  # every other row
            for i in range(0, self.shape[0], 2):
                array[i, :] = 1
        elif pattern == Configurations.ZEBRA_VERTICAL:  # every other col
            for i in range(0, self.shape[1], 2):
                array[:, i] = 1
        elif pattern == Configurations.CHECKERBOARD:  # checkerboard
            array = np.indices(self.shape).sum(axis=0) % 2
        elif pattern == Configurations.MIDDLE_FILL:  # middle fill
            mrow = np.zeros([1, self.shape[1]])
            mrow[
                0,
                int(self.shape[1] / 2 - middle_size[1] / 2) : int(
                    self.shape[1] / 2 - middle_size[1] / 2
                )
                + middle_size[1],
            ] = 1
            for i in range(
                int(self.shape[0] / 2 - middle_size[0] / 2),
                int(self.shape[0] / 2 - middle_size[0] / 2) + middle_size[0],
            ):
                array[i, :] = mrow
        elif pattern == Configurations.Left_Sweep:
            for i in range(middle_size[0]):
                array[:, i] = 1
        elif pattern == Configurations.RANDOM:
            array = random_loading(
                self.shape, probability=self.params.target_occup_prob
            )
        self.target = array.reshape([self.shape[0], self.shape[1], 1])

    def _generate_dual_species_target(
        self,
        pattern: Configurations = Configurations.ZEBRA_HORIZONTAL,
        middle_size: list = [],
        occupation_prob: float = 0.5,
    ):
        """
        A function for generating common target configurations,
        such as checkerboard, zebra stripes, and middle fill,
        for dual-species.
        """
        self.target_Rb = np.zeros([self.shape[0], self.shape[1]])
        self.target_Cs = np.zeros([self.shape[0], self.shape[1]])

        if len(middle_size) == 0:
            middle_size = generate_middle_fifty(self.shape[0], occupation_prob)

        # Horizontal zebra stripes mixed species pattern
        if pattern == Configurations.ZEBRA_HORIZONTAL:
            for i in range(
                int(self.shape[0] / 2 - middle_size[0] / 2),
                int(self.shape[0] / 2 - middle_size[0] / 2) + middle_size[0],
            ):
                for j in range(
                    int(self.shape[1] / 2 - middle_size[1] / 2),
                    int(self.shape[1] / 2 - middle_size[1] / 2) + middle_size[1],
                ):
                    if i % 2 == 0:
                        self.target_Cs[i, j] = 1
                    else:
                        self.target_Rb[i, j] = 1

        # Vertical zebra stripes mixed species pattern
        if pattern == Configurations.ZEBRA_VERTICAL:
            for i in range(
                int(self.shape[0] / 2 - middle_size[0] / 2),
                int(self.shape[0] / 2 - middle_size[0] / 2) + middle_size[0],
            ):
                for j in range(
                    int(self.shape[1] / 2 - middle_size[1] / 2),
                    int(self.shape[1] / 2 - middle_size[1] / 2) + middle_size[1],
                ):
                    if j % 2 == 0:
                        self.target_Cs[i, j] = 1
                    else:
                        self.target_Rb[i, j] = 1

        if pattern == Configurations.CHECKERBOARD:
            for i in range(
                int(self.shape[0] / 2 - middle_size[0] / 2),
                int(self.shape[0] / 2 - middle_size[0] / 2) + middle_size[0],
            ):
                for j in range(
                    int(self.shape[1] / 2 - middle_size[1] / 2),
                    int(self.shape[1] / 2 - middle_size[1] / 2) + middle_size[1],
                ):
                    if (i + j) % 2 == 0:
                        self.target_Rb[i, j] = 1
                    else:
                        self.target_Cs[i, j] = 1

        if pattern == Configurations.SEPARATE:
            for i in range(
                int(self.shape[0] / 2 - middle_size[0] / 2),
                int(self.shape[0] / 2 - middle_size[0] / 2) + middle_size[0],
            ):
                for j in range(
                    int(self.shape[1] / 2 - middle_size[1] / 2),
                    int(self.shape[1] / 2 - middle_size[1] / 2) + middle_size[1],
                ):
                    if j < int(self.shape[1] / 2):
                        self.target_Cs[i, j] = 1
                    else:
                        self.target_Rb[i, j] = 1

        self.target = np.stack([self.target_Rb, self.target_Cs], axis=2)

    def move_atoms(self, moves: list[list[Move]]) -> tuple[float, int, int]:
        """
        Move atoms in tweezers according to moves
        Each list in moves should include a list of sequential atom moves for a single tweezer

        Returns:
            (move_time, n_parallel_moves, n_total_moves): tuple[float, int, int]
        """
        if np.max(self.matrix) > 1:
            raise Exception("Atom array cannot have values outside of {0,1}.")

        tweezers = TweezerArrayModel(move_list=moves)
        move_time, n_parallel_moves, n_total_moves = tweezers.move_atoms(self.matrix)

        return move_time, n_parallel_moves, n_total_moves

    def get_effective_target_grid(
        self, target: np.ndarray | None = None
    ) -> tuple[int, int, int, int]:
        """
        Returns the minimal bounding box around all atoms in the target configuration.

        ## Returns
        start_row, end_row, start_col, end_col : int
            Indices defining the minimal rectangle containing all target atoms.
        """
        if not isinstance(target, np.ndarray):
            target = self.target

        # Flatten target array to 2D mask
        if self.n_species == 1:
            target_mask = target != 0
        else:
            target_mask = np.any(target != 0, axis=2)

        # Boolean arrays indicating which rows/cols are occupied by target
        rows = np.any(target_mask, axis=1)
        cols = np.any(target_mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            raise Exception(
                "Could not find atoms. Did you initialize a target configuration with AtomArray.generate_target()?"
            )

        # Convert boolean arrats into indices and get first and last cols
        start_row, end_row = np.where(rows)[0][[0, -1]]
        start_col, end_col = np.where(cols)[0][[0, -1]]

        return start_row, end_row, start_col, end_col

    def is_target_loaded(
        self,
        target: np.ndarray | None = None,
        do_ejection: bool = False,
        n_species: int = 1,
    ) -> bool:
        """
        Checks whether the target configuration has been successfully prepared.

        The desired target configuration. Must have the same shape as `self.matrix`.

        **do_ejection** : bool, optional (default = False)
            If True, the function checks the entire `self.matrix` array against `target`.
            If False, only the minimal bounding square around target atoms is checked.

        **n_species** : int, optional (default = 1)
            Number of species in the system.

        ## Returns
        **success_flag** : bool
            True if the relevant part of `self.matrix` matches the `target` configuration,
            False otherwise. This flag helps verify whether the algorithm successfully
            prepared the desired configuration.
        """
        success_flag = False

        if not isinstance(target, np.ndarray):
            target = self.target

        if self.matrix.shape != target.shape:
            print(
                f"Mismatch in shapes {self.matrix.shape} and {target.shape}. Reshaping."
            )
            self.matrix = self.matrix.reshape(target.shape)

        if do_ejection:
            return np.array_equal(self.matrix, target)

        start_row, end_row, start_col, end_col = self.get_effective_target_grid(target)
        if n_species == 1:
            relevant_state = self.matrix[
                start_row : end_row + 1, start_col : end_col + 1
            ]
            relevant_target = target[start_row : end_row + 1, start_col : end_col + 1]
        else:
            relevant_state = self.matrix[
                start_row : end_row + 1, start_col : end_col + 1, :
            ]
            relevant_target = target[
                start_row : end_row + 1, start_col : end_col + 1, :
            ]

        target_mask = relevant_target.astype(bool)
        success_flag = np.sum(relevant_state[target_mask]) == np.sum(relevant_target)

        return success_flag

    def image(
        self, move_list: list[Move] = [], plotted_species: str = "all", savename=""
    ):
        """
        Takes a snapshot of the atom array.

        Parameters
        move_list : list
            any moves that you want to plot
        plotted_species : str
            the atomic species you want to plot. You can choose from ['{SPECIES1NAME}','{SPECIES2NAME}','all'].
        """

        if self.n_species == 1:
            single_species_image(self.matrix, move_list=move_list, savename=savename)
        elif self.n_species == 2:
            if type(self) != np.ndarray:
                plotted_arrays = self.matrix
            else:
                plotted_arrays = self

            if (
                plotted_species.lower() == "all"
                or plotted_species.lower() == SPECIES1NAME.lower()
                or plotted_species.lower() == SPECIES2NAME.lower()
            ):
                dual_species_image(
                    plotted_arrays, move_list=move_list, savename=savename
                )
            else:
                raise ValueError(
                    f"Invalid entry for parameter 'plotted_species': {plotted_species}. Please choose from ['{SPECIES1NAME}','{SPECIES2NAME}', 'all']."
                )

    def plot_target_config(self):
        if self.n_species == 1:
            single_species_image(self.target)
        elif self.n_species == 2:
            dual_species_image(self.target)

    def evaluate_moves(
        self, move_list: list[list[list[Move]]]
    ) -> tuple[float, int, int]:
        """
        Move atoms in tweezers according to move_list
        Each list in move_list represents a different tweezer move sequence
        Each tweezer move sequence (list) is a list of sequential atom moves for a single tweezer

        Returns:
            (move_time, n_parallel_moves, n_total_moves): tuple[float, int, int]
        """
        # making reference time
        t_total = 0
        n_parallel_moves = 0
        n_non_parallel_moves = 0

        # iterating through moves and updating internal state matrix
        for _, move_set in enumerate(move_list):
            # performing the move
            move_time, n_par_move, n_moves = self.move_atoms(move_set)

            n_parallel_moves += n_par_move
            n_non_parallel_moves += n_moves
            t_total += move_time

        return t_total, n_parallel_moves, n_non_parallel_moves
