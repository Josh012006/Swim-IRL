"""Tests for eval/recovery.py: hard_value_iteration and expected_value_diff.

Uses the same 5-state reference MDP as test_maxent_linear.py (see
conftest.py). Two things checked here that don't show up in the MaxEnt
tests: hard_value_iteration is DETERMINISTIC/greedy, not the soft MaxEnt
policy -- with theta=1 the reference action probability is 1.0, not the
~0.8808 from backward_pass. And expected_value_diff must never go
negative -- that guarantee is the entire reason EVD is trustworthy as a
metric.
"""
import numpy as np
import pytest

from eval.recovery import hard_value_iteration, expected_value_diff

A, B, C, D, E = 0, 1, 2, 3, 4
HORIZON = 2
THETA = np.array([1.0])
START_DISTRIBUTION = np.array([1.0, 0.0, 0.0, 0.0, 0.0])


def test_hard_value_iteration_is_deterministic_and_optimal(reference_mdp):
    action_probs, V = hard_value_iteration(reference_mdp, THETA, HORIZON)
    # "up" (A->B->D) totals f_B+f_D=2; "down" (A->C->E) totals 0 -- greedy
    # value iteration must pick "up" with probability EXACTLY 1
    assert action_probs[0, A, 0] == pytest.approx(1.0)
    assert action_probs[0, A, 1] == pytest.approx(0.0)
    assert V[0, A] == pytest.approx(2.0)


def test_evd_is_zero_when_comparing_theta_to_itself(reference_mdp):
    evd = expected_value_diff(
        reference_mdp, THETA, THETA, START_DISTRIBUTION, HORIZON
    )
    assert evd == pytest.approx(0.0, abs=1e-8)


def test_evd_is_zero_for_a_positive_rescaling_of_theta(reference_mdp):
    """The point this metric exists for: theta_hat = 2*theta_true induces
    the exact same greedy policy (positive scaling never flips an argmax),
    so EVD is 0 even though ||theta_true - theta_hat|| isn't -- this is
    exactly why EVD is used instead of raw parameter distance (see the
    module docstring in eval/recovery.py)."""
    theta_hat = 2.0 * THETA
    evd = expected_value_diff(
        reference_mdp, THETA, theta_hat, START_DISTRIBUTION, HORIZON
    )
    assert evd == pytest.approx(0.0, abs=1e-8)


def test_evd_is_never_negative(reference_mdp):
    rng = np.random.default_rng(0)
    for _ in range(20):
        theta_hat = rng.normal(loc=0.0, scale=2.0, size=THETA.shape)
        evd = expected_value_diff(
            reference_mdp, THETA, theta_hat, START_DISTRIBUTION, HORIZON
        )
        assert evd >= -1e-10  # tiny slack for floating point only