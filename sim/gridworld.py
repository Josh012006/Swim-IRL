'''Tabular simulated swimmer for Phase 0a (linear MaxEnt IRL validation).'''
import numpy as np
from mdp import TabularMDP, coord_to_state, state_to_coord
from sim.features_gridworld import compute_gridworld_features

def build_transition(width: int, height: int, obstacles: list[tuple[int, int]]) -> np.ndarray:
    # returns shape (n_states, n_actions, n_states)
    # actions: 0=stay, 1=up, 2=down, 3=left, 4=right

	n_states = width * height
	transition = np.zeros(shape=(n_states, 5, n_states))

	for i in range(height):
		for j in range(width):
    		# action stay
			transition[coord_to_state(i, j, width), 0, coord_to_state(i, j, width)] = 1.0

			# action up
			if(i == 0 or ((i-1, j) in obstacles)):
				transition[coord_to_state(i, j, width), 1, coord_to_state(i, j, width)] = 1.0
			else:
				transition[coord_to_state(i, j, width), 1, coord_to_state(i-1, j, width)] = 1.0

    		# action down
			if(i == height-1 or ((i+1, j) in obstacles)):
				transition[coord_to_state(i, j, width), 2, coord_to_state(i, j, width)] = 1.0
			else:
				transition[coord_to_state(i, j, width), 2, coord_to_state(i+1, j, width)] = 1.0

			# action left
			if(j == 0 or ((i, j-1) in obstacles)):
				transition[coord_to_state(i, j, width), 3, coord_to_state(i, j, width)] = 1.0
			else:
				transition[coord_to_state(i, j, width), 3, coord_to_state(i, j-1, width)] = 1.0

			# action right
			if(j == width-1 or ((i, j+1) in obstacles)):
				transition[coord_to_state(i, j, width), 4, coord_to_state(i, j, width)] = 1.0
			else:
				transition[coord_to_state(i, j, width), 4, coord_to_state(i, j+1, width)] = 1.0

	return transition



def build_gridworld_mdp(width: int, height: int, obstacles: list[tuple[int, int]], goal_pos: tuple[int, int]) -> TabularMDP:
    # features: shape (n_states, n_features), already calculated (via features_gridworld.py)
    # simply combines the geometry and the given features into a complete TabularMDP
    return TabularMDP(
        n_states=width * height,
        n_actions=5,
        transition=build_transition(width, height, obstacles),
        features=compute_gridworld_features(width, height, obstacles, goal_pos),
    )
