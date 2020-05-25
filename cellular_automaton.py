import jax
import jax.numpy as np
from typing import Callable, List


class CellularAutomaton2D:
    def __init__(
        self, rule: Callable, neighborhood: List[List[bool]], grid,
    ):

        if not self._is_valid_rule(rule):
            raise Exception("Invalid rule")

        if not self._is_valid_grid(grid):
            raise Exception("Invalid grid")

        if not self._is_valid_neighborhood(neighborhood):
            raise Exception("Invalid neighborhood")

        self.rule = rule
        self.neighborhood = neighborhood
        self.grid = np.array(grid)

        # Create a shifted matrix with the same shape as `grid` for each neighbor cell
        # that must be included, as defined by the neighborhood.
        self._shifts = self._generate_shifts()

        self._generate_next_grid_jit = jax.jit(
            self._generate_next_grid, static_argnums=[1, 2]
        )

    @staticmethod
    def _generate_next_grid(grid, shifts, rule):
        # TODO: handle non-cyclic and pad-cell
        shift_matrices = []
        for (shift_x, shift_y) in shifts:
            shift_matrices.append(np.roll(grid, (shift_x, shift_y), axis=(1, 0)))

        # Stack these matrices to create a 3D block.
        shift_block = np.array(shift_matrices)

        # Reshape so that all neighborhoods are loaded into vectors
        loaded = np.reshape(
            np.stack(shift_block, axis=2),
            (shift_block.shape[1] * shift_block.shape[2], shift_block.shape[0], -1),
        )

        # map the cellular automata's rule to a vectorized function
        # which acts on the loaded neighborhoods
        # this also effectively wraps the rule in "jax.jit"
        vrule = jax.vmap(rule, in_axes=0, out_axes=0)

        # create the next grid by running each loaded neighborhood to the next cell
        next_grid = np.reshape(
            vrule(loaded), (shift_block.shape[1], shift_block.shape[2], -1)
        )
        return next_grid

    def step(self, n=1):
        for i in range(n):
            self.grid = self._generate_next_grid_jit(self.grid, self._shifts, self.rule)

    def _generate_shifts(self):
        height = len(self.neighborhood)
        width = len(self.neighborhood[0])

        center_y = height // 2
        center_x = width // 2

        shifts = []
        for i in range(len(self.neighborhood)):
            for j in range(len(self.neighborhood[0])):
                if self.neighborhood[i][j]:
                    shifts.append((i - center_x, j - center_y))

        return shifts

    def _is_valid_rule(self, rule):
        return True

    def _is_valid_neighborhood(self, neighborhood):
        if len(neighborhood) < 1:
            return False
        if len(neighborhood) % 2 != 1:
            return False
        # TODO: check that it's a np or onp array
        # TODO: check rectangleness
        return True

    def _is_valid_grid(self, grid):
        if len(grid.shape) < 3:
            # TODO: Reply with a good error message here, if cell values are not vectors
            return False
        if len(grid) < 1:
            return False
        if len(grid[0]) < 1:
            return False
        return True
