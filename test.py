from cellular_automaton import CellularAutomaton2D
from neighborhood import moore
import jax.numpy as np
import math


def rule(neighborhood):
    return np.average(neighborhood, axis=0)


grid = np.array([[[0.0], [1.0], [0.0]], [[1.0], [0.0], [1.0]], [[0.0], [1.0], [0.0]]])

ca = CellularAutomaton2D(rule, moore(1), grid)

if __name__ == "__main__":
    ca.step()
    for i in range(100):
        ca.step()
    assert math.isclose(ca.grid[0][0], np.average(np.ravel(grid), axis=0))

    print("Test passed.")
