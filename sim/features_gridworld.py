'''Logic for features of gridworld simulation'''

import numpy as np
from mdp import coord_to_state

def L1(p1: tuple[int, int], p2: tuple[int, int]) -> int:
	return np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])

def compute_gridworld_features(width: int, height: int, obstacles: list[tuple[int, int]], goal_pos: tuple[int, int]) -> np.ndarray:
	"""Features per state, in this order:
        0: row
        1: col
        2: L1 distance to goal
        3: L1 distance to closest obstacle
    """
	features = np.zeros(shape=(width * height, 4))

	for i in range(height):
		for j in range(width):
			features[coord_to_state(i, j, width)] = [i, j, L1((i, j), goal_pos), np.min([L1((i, j), obs) for obs in obstacles])]

	return features