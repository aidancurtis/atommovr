# Parent class for all algorithms.
# For developers: 
#   Feel free to use this as a template for new algorithms.
#   Each function below describes the requirements/what you need to put in it.

# Author: Nikhil Harle, Aidan Curtis

import numpy as np


class Algorithm:
    """ 
    Parent class for all algorithms. 
    
    NB: The following functions are placeholders for illustrative purposes only and 
    should be overwritten for your particular algorithm.
    
    If your algorithm can only prepare select target configurations, please list them here.
    
    e.g:
    
    Supported configurations: Middle Fill (see `atommover.utils.core.Configurations`
    for a list of configurations). 
    """
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return 'Insert the name of your algorithm here. This is what will show up on your benchmarking plots.'

    def get_moves(self, atom_array, do_ejection: bool = False) -> tuple[np.ndarray, list, bool]:
        """ 
        This is the main function for the algorithm. 
        
        ## Parameters
        **atom_array** : AtomArray 
            object containing the initial configuration `atom_array.matrix`
            and the target configuration `atom_array.target`.

        **do_ejection** : bool, optional (default = False) 
            argument to run an ejection subroutine(see 
            `atommover.algorithms.source.ejection.py` for the protocol).
        
        any other (optional!) kwargs you see fit to include :)
        
        ## Returns
        **config** : np.ndarray
            the final configuration after all moves have been applied
            (ideally, this should just be the target configuration)

        **move_set** : list[list[Move, Move...], list[Move], ...]
            contains all the moves to transform the initial configuration into the final 
            configuration. 
            each list inside `move_set` is a set of moves that will be done in parallel.
            If you're confused by lists inside of lists, consider the following example:
                `small_move_list = [Move1]`
                `small_move_list1 = [Move2, Move3]`
                `small_move_list2 = [Move4, Move5]`
                `move_set = [small_move_list, small_move_list1, small_move_list2]`
            When this is read by the framework, it will first execute Move1, then will execute
            Move2 and Move3 in parallel, then after that Move4 and Move5 will be executed in 
            parallel.
        **success_flag** : bool
            simple sanity check. This should be set to True if the algorithm prepares the 
            final configuration and `False` if it does not. This is helpful during benchmarking.
        """
        config = np.zeros(atom_array.shape)
        move_set = []
        success_flag = False

        # your code here #

        return config, move_set, success_flag
    
    # Utility function common to all algorithms
    @staticmethod
    def get_success_flag(state: np.ndarray, target: np.ndarray, do_ejection: bool = False, n_species: int = 1) -> bool:
        """
        Checks whether the target configuration has been successfully prepared.

        ## Parameters
        **state** : np.ndarray
            The current configuration of the system. This can be a 2D array for a single species
            or a 3D array if multiple species are present (rows x cols x species).

        **target** : np.ndarray
            The desired target configuration. Must have the same shape as `state`.

        **do_ejection** : bool, optional (default = False)
            If True, the function checks the entire `state` array against `target`.
            If False, only the minimal bounding square around target atoms is checked.

        **n_species** : int, optional (default = 1)
            Number of species in the system.

        ## Returns
        **success_flag** : bool
            True if the relevant part of `state` matches the `target` configuration, 
            False otherwise. This flag helps verify whether the algorithm successfully 
            prepared the desired configuration.
        """
        success_flag = False

        if state.shape != target.shape:
            print(f'Mismatch in shapes {state.shape} and {target.shape}. Reshaping.')
            state = state.reshape(target.shape)

        if do_ejection:
            return np.array_equal(state, target)

        start_row, end_row, start_col, end_col = get_effective_target_grid(target, n_species)
        if n_species == 1:
            relevant_state = state[start_row:end_row + 1, start_col:end_col + 1]
            relevant_target = target[start_row:end_row + 1, start_col:end_col + 1]
        else:
            relevant_state = state[start_row:end_row + 1, start_col:end_col + 1, :]
            relevant_target = target[start_row:end_row + 1, start_col:end_col + 1, :]

        target_mask = relevant_target.astype(bool)
        success_flag = np.sum(relevant_state[target_mask]) == np.sum(relevant_target)

        return success_flag

def get_effective_target_grid(target: np.ndarray, n_species: int = 1) -> tuple[int, int, int, int]:
    """
    Returns the minimal bounding box around all atoms in the target configuration.

    ## Parameters
    **target** : np.ndarray
        Target configuration array (2D for single species, 3D for multiple species).
    **n_species** : int, optional (default = 1)
        Number of species in the system.

    ## Returns
    start_row, end_row, start_col, end_col : int
        Indices defining the minimal rectangle containing all target atoms.
    """
    # Flatten target array to 2D mask
    if n_species == 1:
        target_mask = target != 0
    else:
        target_mask = np.any(target != 0, axis=2)

    # Boolean arrays indicating which rows/cols are occupied by target
    rows = np.any(target_mask, axis=1)
    cols = np.any(target_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        raise Exception("Could not find atoms. Did you initialize a target configuration with AtomArray.generate_target()?")

    # Convert boolean arrats into indices and get first and last cols
    start_row, end_row = np.where(rows)[0][[0, -1]]
    start_col, end_col = np.where(cols)[0][[0, -1]]

    return start_row, end_row, start_col, end_col
