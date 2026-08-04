"""Reference-case tests for irl/maxent_linear.py.

Uses the tiny 5-state MDP (A -> B/C -> D/E) worked out by hand earlier in
the project: A branches into B or C (the only real decision point), B
leads deterministically to D, C leads deterministically to E. Features:
f_B = f_D = 1, everything else 0. With theta = 1, the hand-derived
reference values are:

    P(up | A)                    ~= 0.8808
    D_B  (visitation of B at t=1) ~= 0.8808
    expected feature count       ~= 1.7616

States are indexed 0=A, 1=B, 2=C, 3=D, 4=E. Actions are indexed 0=up,
1=down.

Note on the transition matrix: B, C, D, and E each only have one
*meaningful* action in this story, but TabularMDP requires a fixed
n_actions across every state, so each state's spare action slot is filled
with a duplicate of its real one (same target, same probability). This is
harmless for everything the algorithm actually consumes downstream --
P(a|s,t), state visitation, and feature counts are all unaffected, since a
duplicated action shifts every branch of a softmax by the same constant,
which cancels out in the ratio. It DOES inflate the raw log-partition
value V[0, A] by a known combinatorial factor (here, roughly log(2) per
artificially-duplicated decision point along the way) -- so V/Z itself is
deliberately not asserted here, since nothing downstream depends on its
raw value anyway.
"""
import numpy as np
import pytest

from mdp import TabularMDP
from irl.maxent_linear import (
    backward_pass,
    forward_pass,
    expected_feature_counts,
    fit,
)

N_STATES = 5
N_ACTIONS = 2
A, B, C, D, E = 0, 1, 2, 3, 4
HORIZON = 2
THETA = np.array([1.0])
START_DISTRIBUTION = np.array([1.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def reference_mdp() -> TabularMDP:
    transition = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    transition[A, 0, B] = 1.0  # A --up--> B
    transition[A, 1, C] = 1.0  # A --down--> C
    transition[B, 0, D] = 1.0  # B's only real action --> D
    transition[B, 1, D] = 1.0  # spare slot, duplicate (see module docstring)
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


def test_backward_pass_matches_hand_derivation(reference_mdp):
    action_probs, _V = backward_pass(reference_mdp, THETA, HORIZON)
    assert action_probs[0, A, 0] == pytest.approx(0.8808, abs=1e-4)  # P(up|A)
    assert action_probs[0, A, 1] == pytest.approx(0.1192, abs=1e-4)  # P(down|A)


def test_forward_pass_matches_hand_derivation(reference_mdp):
    action_probs, _V = backward_pass(reference_mdp, THETA, HORIZON)
    state_visitation = forward_pass(
        reference_mdp, action_probs, START_DISTRIBUTION, HORIZON
    )
    assert state_visitation[1, B] == pytest.approx(0.8808, abs=1e-4)
    assert state_visitation[1, C] == pytest.approx(0.1192, abs=1e-4)
    assert state_visitation[2, D] == pytest.approx(0.8808, abs=1e-4)
    assert state_visitation[2, E] == pytest.approx(0.1192, abs=1e-4)


def test_expected_feature_counts_matches_hand_derivation(reference_mdp):
    action_probs, _V = backward_pass(reference_mdp, THETA, HORIZON)
    state_visitation = forward_pass(
        reference_mdp, action_probs, START_DISTRIBUTION, HORIZON
    )
    expected = expected_feature_counts(reference_mdp, state_visitation)
    assert expected[0] == pytest.approx(1.7616, abs=1e-4)


def test_fit_recovers_a_known_theta(reference_mdp):
    """End-to-end sanity check: generate demonstrations from a KNOWN
    theta, then verify fit() recovers something close to it. Uses a fixed
    seed for reproducibility."""
    true_theta = np.array([2.0])
    action_probs, _V = backward_pass(reference_mdp, true_theta, HORIZON)

    rng = np.random.default_rng(0)
    demonstrations = []
    for _ in range(500):
        s = A
        trajectory = [s]
        for t in range(HORIZON):
            a = rng.choice(N_ACTIONS, p=action_probs[t, s])
            s = int(np.argmax(reference_mdp.transition[s, a]))
            trajectory.append(s)
        demonstrations.append(np.array(trajectory))

    theta_hat = fit(
        reference_mdp,
        demonstrations,
        START_DISTRIBUTION,
        HORIZON,
        learning_rate=0.5,
        n_iterations=300,
    )
    assert theta_hat[0] == pytest.approx(2.0, abs=0.05)