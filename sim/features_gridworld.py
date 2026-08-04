import numpy as np
from mdp import coord_to_state

def L1(p1, p2):
    return np.abs(p1[0]-p2[0]) + np.abs(p1[1]-p2[1])

def compute_gridworld_features(width, height, obstacles, goal_pos):
    """Features per state, in this order:
        0: L1 distance to goal
        1: L1 distance to closest obstacle

    row/col were dropped: they were the actual source of the collinearity
    with dist_goal found during Phase 0a validation (a structural,
    geometric correlation with a fixed corner goal -- randomizing the
    start state only partially fixes it, see README Limitations). They
    also don't correspond to anything in the real worm project.
    """
    features = np.zeros(shape=(width * height, 2))
    for i in range(height):
        for j in range(width):
            features[coord_to_state(i, j, width)] = [
                L1((i, j), goal_pos),
                np.min([L1((i, j), obs) for obs in obstacles]),
            ]
    return features