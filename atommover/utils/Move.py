# Object representing a single atom move

import numpy as np


class Move:
    """
    Generic class for atom moves.
    """

    def __init__(self, from_row: int, from_col: int, to_row: int, to_col: int) -> None:
        self.from_row = from_row
        self.from_col = from_col
        self.to_row = to_row
        self.to_col = to_col
        self.dx = to_col - from_col
        self.dy = to_row - from_row
        self.distance = self._get_distance()
        self.midx, self.midy = self._get_move_midpoint()
        self.failure_flag = 0

    def __repr__(self) -> str:
        return self.move_str()

    def _get_distance(self) -> int:
        return np.sqrt((self.dx) ** 2 + (self.dy) ** 2)

    def move_str(self) -> str:
        return f"({self.from_row}, {self.from_col}) -> ({self.to_row}, {self.to_col})"

    def _get_move_midpoint(self):
        self.midx = self.from_col + self.dx / 2
        self.midy = self.from_row + self.dy / 2
        return self.midx, self.midy

    def __eq__(self, other) -> bool:
        if isinstance(other, Move):
            return (
                self.from_row == other.from_row
                and self.from_col == other.from_col
                and self.to_row == other.to_row
                and self.to_col == other.to_col
            )
        return False
