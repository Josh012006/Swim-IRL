'''Reward recovery metrics for Phase 0a/0b: Expected Value Difference (EVD).

Comparing raw theta values directly (||theta_true - theta_hat||) is
misleading: several different theta vectors can induce near-identical
behavior -- the whole reason IRL is ill-posed (see README Limitations).
EVD compares *behavior* instead of parameters: how much expected reward
(under the TRUE reward) is lost by following the policy optimized for
theta_hat, instead of the truly-optimal policy for theta_true.
EVD = 0 means the recovered policy is behaviorally indistinguishable from
optimal, even if theta_hat != theta_true numerically -- exactly the point.
'''
import numpy as np
from mdp import TabularMDP
from irl.maxent_linear import forward_pass, expected_feature_counts


def hard_value_iteration(
    mdp: TabularMDP,
    theta: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    # returns (action_probs, V), same shapes as backward_pass -- but this
    # is the DETERMINISTIC, reward-maximizing optimal policy (argmax),
    # not the MaxEnt soft policy. Needed so that "value_true" below is
    # genuinely the best-achievable value under theta_true -- the MaxEnt
    # soft policy generally achieves LESS than this by design, which
    # would break EVD's >= 0 guarantee if reused here.
    #
    # Same recursion skeleton as backward_pass, softmax swapped for max:
    #   V_T(s) = theta^T f_s
    #   Q_t(s,a) = theta^T f_s + sum_s' P(s'|s,a) V_{t+1}(s')   (no log!)
    #   V_t(s) = max_a Q_t(s,a)
    #   action_probs[t,s,:] = one-hot at argmax_a Q_t(s,a)
    V = np.zeros(shape=(horizon+1, mdp.n_states))
    action_probs = np.zeros(shape=(horizon, mdp.n_states, mdp.n_actions))

    state_reward = mdp.features @ theta
    V[horizon] = state_reward

    for t in range(horizon-1, -1, -1):
    	for s in range(0, mdp.n_states):
            Q = np.zeros(shape=mdp.n_actions)
            for a in range(0, mdp.n_actions):
                Q[a] = state_reward[s] + V[t+1] @ mdp.transition[s, a]
            V[t, s] = np.max(Q)
            action_probs[t, s, np.argmax(Q)] = 1.0

    return (action_probs, V)


def expected_value_diff(
    mdp: TabularMDP,
    theta_true: np.ndarray,
    theta_hat: np.ndarray,
    start_distribution: np.ndarray,
    horizon: int,
) -> float:
    # returns EVD >= 0 (0 = perfect recovery; lower is better)
    pi_true = hard_value_iteration(mdp, theta_true, horizon)
    pi_hat = hard_value_iteration(mdp, theta_hat, horizon)
    D_true = forward_pass(mdp, pi_true[0], start_distribution, horizon)
    D_hat   = forward_pass(mdp, pi_hat[0],  start_distribution, horizon)
    value_true = expected_feature_counts(mdp, D_true) @ theta_true
    value_hat  = expected_feature_counts(mdp, D_hat)  @ theta_true
    return float(value_true - value_hat)