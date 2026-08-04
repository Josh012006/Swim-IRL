'''Linear MaxEnt IRL (Ziebart et al. 2008): forward-backward DP, Phase 0a.'''
from mdp import TabularMDP
import numpy as np
import scipy as sp

def backward_pass(
    mdp: TabularMDP,
    theta: np.ndarray,      # shape (n_features,)
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    # returns (action_probs, V)
    # action_probs: shape (horizon, n_states, n_actions) — P(a|s,t)
    # V: shape (horizon+1, n_states) — log Z_{s,t}, le soft value
    
    V = np.zeros(shape=(horizon+1, mdp.n_states))
    action_probs = np.zeros(shape=(horizon, mdp.n_states, mdp.n_actions))

    state_reward = mdp.features @ theta
    V[horizon] = state_reward

    for t in range(horizon-1, -1, -1):
    	for s in range(0, mdp.n_states):
            Q = np.zeros(shape=mdp.n_actions)
            for a in range(0, mdp.n_actions):
                Q[a] = state_reward[s] + sp.special.logsumexp(V[t+1], b=mdp.transition[s, a])
            V[t, s] = sp.special.logsumexp(Q)
            action_probs[t, s] = np.exp(Q - V[t, s])

    return (action_probs, V)





def forward_pass(
    mdp: TabularMDP,
    action_probs: np.ndarray,        # shape (horizon, n_states, n_actions)
    start_distribution: np.ndarray,  # shape (n_states,)
    horizon: int,
) -> np.ndarray:
    # returns state_visitation, shape (horizon+1, n_states) — D_{s,t}
    state_visitation = np.zeros(shape=(horizon+1, mdp.n_states))

    state_visitation[0] = start_distribution
    for t in range(1, horizon+1):
        for sf in range(0, mdp.n_states):
            state_visitation[t, sf] = np.sum(
                [
                    state_visitation[t-1, s] 
                    * action_probs[t-1, s, a] 
                    * mdp.transition[s, a, sf]
                    for s in range(mdp.n_states)
                    for a in range(mdp.n_actions)
                ]
            )

    return state_visitation



def expected_feature_counts(
    mdp: TabularMDP,
    state_visitation: np.ndarray,  # shape (horizon+1, n_states)
) -> np.ndarray:
    # returns shape (n_features,)
    result = np.zeros(shape=mdp.features.shape[1])

    for t in range(0, state_visitation.shape[0]):
        result += mdp.features.T @ state_visitation[t]

    return result




def fit(
    mdp: TabularMDP,
    demonstrations: list[np.ndarray],   # each shape (horizon+1,), dtype=int — état à chaque t
    start_distribution: np.ndarray,     # shape (n_states,)
    horizon: int,
    learning_rate: float = 0.1,
    n_iterations: int = 200,
    theta_init: np.ndarray | None = None,  # défaut : zeros(n_features)
) -> np.ndarray:
    # returns theta_hat, shape (n_features,)
    theta_hat = np.zeros(shape=mdp.features.shape[1]) if theta_init is None else theta_init.copy()

    empirical_features = np.zeros(shape=(len(demonstrations), mdp.features.shape[1]))
    for i, demo in enumerate(demonstrations):
        state_visitation_demo = np.zeros(shape=(horizon+1, mdp.n_states))
        for j in range(len(demo)):
            state_visitation_demo[j, demo[j]] = 1.0
        empirical_features[i] = expected_feature_counts(mdp, state_visitation_demo)
    empirical = np.mean(empirical_features, axis=0)

    for _ in range(0, n_iterations):
        action_probs, V = backward_pass(mdp, theta_hat, horizon)
        state_visitation = forward_pass(mdp, action_probs, start_distribution, horizon)
        expected = expected_feature_counts(mdp, state_visitation)

        gradient = empirical - expected
        theta_hat += learning_rate * gradient

    return theta_hat


