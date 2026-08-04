"""Convergence diagnostic for irl.maxent_linear.fit(): track the MaxEnt
log-likelihood across gradient ascent iterations.

Ziebart's MaxEnt objective is provably concave (see Ziebart 2008), so
log-likelihood must increase monotonically -- up to floating-point noise
-- for a correctly-behaving learning_rate. If it ever decreases, that's a
direct signal that learning_rate is too large for this data, not ordinary
optimizer noise to shrug off.

This diagnoses a DIFFERENT failure mode than eval/feature_correlation.py:
that one catches identifiability/collinearity issues that persist even
with a perfectly-converged fit; this one catches optimization instability
that produces a garbage theta_hat regardless of how identifiable the
features are. Run this whenever a recovered theta_hat looks suspicious.
"""
import numpy as np
from mdp import TabularMDP
from irl.maxent_linear import backward_pass, forward_pass, expected_feature_counts


def compute_loglik_history(
    mdp: TabularMDP,
    demonstrations: list[np.ndarray],
    start_distribution: np.ndarray,
    horizon: int,
    learning_rate: float = 0.02,
    n_iterations: int = 200,
    theta_init: np.ndarray | None = None,
) -> np.ndarray:
    """Re-runs fit()'s gradient ascent loop, recording the MaxEnt
    log-likelihood of the demonstrations at theta_hat BEFORE each
    gradient step (so index 0 is the log-likelihood at theta_init).

    Returns loglik_history, shape (n_iterations,).
    """
    theta_hat = (
        np.zeros(mdp.features.shape[1]) if theta_init is None else theta_init.copy()
    )
    empirical_features = np.zeros(shape=(len(demonstrations), mdp.features.shape[1]))
    for i, demo in enumerate(demonstrations):
        empirical_features[i] = mdp.features[demo].sum(axis=0)
    empirical = np.mean(empirical_features, axis=0)

    loglik_history = np.zeros(n_iterations)
    for it in range(n_iterations):
        action_probs, V = backward_pass(mdp, theta_hat, horizon)
        state_visitation = forward_pass(mdp, action_probs, start_distribution, horizon)
        expected = expected_feature_counts(mdp, state_visitation)
        loglik_history[it] = theta_hat @ empirical - (start_distribution @ V[0])
        gradient = empirical - expected
        theta_hat = theta_hat + learning_rate * gradient
    return loglik_history


def assert_monotonically_increasing(loglik_history: np.ndarray, tol: float = 1e-6) -> None:
    """Raises AssertionError, naming the first offending iteration, if
    log-likelihood ever decreases by more than floating-point noise.
    A concave objective under a well-behaved learning rate can never do
    this -- if it happens, learning_rate is too large for this data.
    """
    diffs = np.diff(loglik_history)
    bad_iterations = np.where(diffs < -tol)[0]
    if len(bad_iterations) > 0:
        first_bad = bad_iterations[0]
        raise AssertionError(
            f"Log-likelihood decreased at iteration {first_bad} -> {first_bad + 1} "
            f"({loglik_history[first_bad]:.4f} -> {loglik_history[first_bad + 1]:.4f}). "
            f"MaxEnt's objective is concave -- this means learning_rate is too "
            f"large for this data, not normal optimizer noise."
        )