import cv2
import jax.numpy as np
import numpy as onp
import jax

from cellular_automaton import CellularAutomaton2D
from neighborhood import von_neumann

if __name__ == "__main__":
    # cells are [uint8, uint8, uint8] color vectors

    width = 1200 * 2
    height = 900 * 2
    grid = onp.zeros((height, width, 3), dtype="uint8")

    grid[(1 * height) // 3][(2 * width) // 3] = [0, 0, 255]
    grid[(1 * height) // 3][(1 * width) // 3] = [0, 0, 255]
    grid[(2 * height) // 3][(1 * width) // 2] = [0, 255, 0]

    grid = np.array(grid)

    blend_coefficient = 0.008

    # list of cells -> cell
    def shiva_rule(neighbors):
        center = neighbors[len(neighbors) // 2]

        color_add = (np.argmax(np.average(neighbors, axis=0), axis=0) + 1) % 3

        arise = jax.nn.one_hot(np.array([color_add], dtype="uint8"), 3)[0] * 255

        updated_center = np.floor(
            np.average(
                np.stack([center, arise]),
                axis=0,
                weights=[1.0 - blend_coefficient, blend_coefficient],
            )
        )

        return np.array(updated_center, dtype="uint8")

    ca = CellularAutomaton2D(rule=shiva_rule, neighborhood=von_neumann(1), grid=grid)

    ca.step()
    while True:
        subsampled = ca.grid[::2, ::2]
        cv2.imshow("Shiva", onp.array(subsampled))
        ca.step(2)
        cv2.waitKey(20)
