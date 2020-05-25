import cv2
import jax.numpy as np
import numpy as onp

from cellular_automaton import CellularAutomaton2D
from neighborhood import moore

if __name__ == "__main__":
    # cells are [ bool ] vectors

    width = 1900
    height = 1200
    grid = onp.round(onp.random.random((height, width, 1)))
    grid = np.array(grid, dtype="bool")

    def life_rule(neighbors):
        center = neighbors[len(neighbors) // 2]

        alive = np.sum(neighbors) - center

        return np.where(
            alive == 3,
            np.array([1]),
            np.where(alive == 2, np.array([center]), np.array([0])),
        )[0]

    ca = CellularAutomaton2D(rule=life_rule, neighborhood=moore(1), grid=grid)

    ca.step()
    while True:
        cv2.imshow("Life", onp.array(ca.grid, dtype="float32"))
        ca.step(1)
        cv2.waitKey(20)
