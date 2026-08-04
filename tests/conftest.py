"""Shared pytest fixtures for the Swim-IRL test suite.

reference_mdp is the tiny 5-state MDP (A -> B/C -> D/E) worked out by hand
early in the project: A branches into B or C (the only real decision
point), B leads deterministically to D, C leads deterministically to E.
Features: f_B = f_D = 1, everything else 0.

See the module docstring in test_maxent_linear.py for the full set of
hand-derived reference values this MDP was originally built to validate.
Factored out here since it's now shared by test_maxent_linear.py and
test_recovery.py.
"""
import numpy as np
import pytest

from mdp import TabularMDP

N_STATES = 5
N_ACTIONS = 2
A, B, C, D, E = 0, 1, 2, 3, 4


@pytest.fixture
def reference_mdp() -> TabularMDP:
    transition = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    transition[A, 0, B] = 1.0  # A --up--> B
    transition[A, 1, C] = 1.0  # A --down--> C
    transition[B, 0, D] = 1.0  # B's only real action --> D
    transition[B, 1, D] = 1.0  # spare slot, duplicate
    transition[C, 0, E] = 1.0  # C's only real action --> E
    transition[C, 1, E] = 1.0  # spare slot, duplicate
    transition[D, 0, D] = 1.0  # D is absorbing
    transition[D, 1, D] = 1.0
    transition[E, 0, E] = 1.0  # E is absorbing
    transition[E, 1, E] = 1.0

    features = np.zeros((N_STATES, 1))
    features[B, 0] = 1.0
    features[D, 0] = 1.0

    return TabularMDP(
        n_states=N_STATES,
        n_actions=N_ACTIONS,
        transition=transition,
        features=features,
    )