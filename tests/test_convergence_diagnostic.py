"""Tests for eval/convergence_diagnostic.py.

Uses a small 4x4 gridworld (not the tiny 5-state reference_mdp -- that
topology turns out to be too simple, with only one real decision point,
to exhibit non-monotonic log-likelihood even at very large learning
rates; the pathology needs enough decision points and feature
interaction to actually show up). Small enough to run in a few seconds,
but a real reproduction of the exact failure found in the project: at
learning_rate=0.3 the log-likelihood is provably NOT monotonically
increasing, while at learning_rate=0.02 it is -- this is the
regression test for that finding, so learning_rate creeping back up
gets caught automatically instead of silently producing a bad theta_hat.
"""
import numpy as np
import pytest

from mdp import state_to_coord
from sim.gridworld import build_gridworld_mdp
from data.simulate import generate_demonstrations
from eval.convergence_diagnostic import (
    compute_loglik_history,
    assert_monotonically_increasing,
)

WIDTH, HEIGHT = 4, 4
GOAL_POS = (3, 3)
OBSTACLES = [(1, 1)]
HORIZON = 6
THETA_TRUE = np.array([0.0, 1.0])


@pytest.fixture
def small_grid_demonstrations():
    mdp = build_gridworld_mdp(WIDTH, HEIGHT, OBSTACLES, GOAL_POS)
    valid_states = [
        s for s in range(mdp.n_states) if state_to_coord(s, WIDTH) not in OBSTACLES
    ]
    start_distribution = np.zeros(mdp.n_states)
    start_distribution[valid_states] = 1.0 / len(valid_states)

    rng = np.random.default_rng(0)
    demonstrations = generate_demonstrations(
        mdp, THETA_TRUE, start_distribution, HORIZON, n_demonstrations=400, rng=rng
    )
    return mdp, start_distribution, demonstrations


def test_small_learning_rate_is_monotonically_increasing(small_grid_demonstrations):
    mdp, start_distribution, demonstrations = small_grid_demonstrations
    loglik_history = compute_loglik_history(
        mdp, demonstrations, start_distribution, HORIZON,
        learning_rate=0.02, n_iterations=60,
    )
    assert_monotonically_increasing(loglik_history)  # must not raise


def test_large_learning_rate_is_caught_as_non_monotonic(small_grid_demonstrations):
    """This is the exact pathology found during the project: with too
    large a learning_rate, gradient ascent overshoots and the concave
    log-likelihood decreases at some iteration -- assert_monotonically_
    increasing must catch this, not silently let it through."""
    mdp, start_distribution, demonstrations = small_grid_demonstrations
    loglik_history = compute_loglik_history(
        mdp, demonstrations, start_distribution, HORIZON,
        learning_rate=0.3, n_iterations=60,
    )
    with pytest.raises(AssertionError, match="Log-likelihood decreased"):
        assert_monotonically_increasing(loglik_history)