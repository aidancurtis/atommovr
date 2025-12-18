import numpy as np


def get_effective_target_grid(
    target: np.ndarray, n_species: int = 1
) -> tuple[int, int, int, int]:
    """
    Returns the minimal bounding box around all atoms in the target configuration.

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
        raise Exception(
            "Could not find atoms. Did you initialize a target configuration with AtomArray.generate_target()?"
        )

    # Convert boolean arrats into indices and get first and last cols
    start_row, end_row = np.where(rows)[0][[0, -1]]
    start_col, end_col = np.where(cols)[0][[0, -1]]

    return start_row, end_row, start_col, end_col


def is_target_loaded(
    matrix: np.ndarray,
    target: np.ndarray,
    do_ejection: bool = False,
    n_species: int = 1,
) -> bool:
    """
    Checks whether the target configuration has been successfully prepared.

        The desired target configuration. Must have the same shape as `matrix`.

    **do_ejection** : bool, optional (default = False)
        If True, the function checks the entire `matrix` array against `target`.
        If False, only the minimal bounding square around target atoms is checked.

    **n_species** : int, optional (default = 1)
        Number of species in the system.

    ## Returns
    **success_flag** : bool
        True if the relevant part of `matrix` matches the `target` configuration,
        False otherwise. This flag helps verify whether the algorithm successfully
        prepared the desired configuration.
    """
    success_flag = False

    if matrix.shape != target.shape:
        print(f"Mismatch in shapes {matrix.shape} and {target.shape}. Reshaping.")
        matrix = matrix.reshape(target.shape)

    if do_ejection:
        return np.array_equal(matrix, target)

    start_row, end_row, start_col, end_col = get_effective_target_grid(target)
    if n_species == 1:
        relevant_state = matrix[start_row : end_row + 1, start_col : end_col + 1]
        relevant_target = target[start_row : end_row + 1, start_col : end_col + 1]
    else:
        relevant_state = matrix[start_row : end_row + 1, start_col : end_col + 1, :]
        relevant_target = target[start_row : end_row + 1, start_col : end_col + 1, :]

    target_mask = relevant_target.astype(bool)
    success_flag = np.sum(relevant_state[target_mask]) == np.sum(relevant_target)

    return success_flag
