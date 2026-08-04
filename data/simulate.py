'''Ground-truth trajectory generation for Phase 0a.

Given an MDP and a CHOSEN ground-truth theta, generates synthetic expert
demonstrations by (1) running backward_pass to get the MaxEnt-optimal
policy for that theta, then (2) sampling trajectories from that policy.
Because the demonstrations come directly from the model backward_pass
assumes, this isolates "does fit() recover the theta that generated the
data" from "is MaxEnt IRL a good model of real behavior" -- only the
former question belongs in Phase 0a.
'''
import numpy as np
from mdp import TabularMDP
from irl.maxent_linear import backward_pass


def sample_trajectory(
    mdp: TabularMDP,
    action_probs: np.ndarray,        # shape (horizon, n_states, n_actions)
    start_distribution: np.ndarray,  # shape (n_states,)
    horizon: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    # returns (states, actions)
    # states:  shape (horizon+1,), dtype=int -- state visited at each t
    # actions: shape (horizon,),   dtype=int -- action taken at each t
    #
    # s_0 ~ start_distribution
    # for t in 0..horizon-1:
    #   a_t ~ action_probs[t, s_t]           (Categorical over n_actions)
	#   s_{t+1} ~ mdp.transition[s_t, a_t]   (Categorical over n_states)
	states = np.zeros(shape=horizon+1, dtype=int)
	actions = np.zeros(shape=horizon, dtype=int)

	init_state = int(rng.choice(mdp.n_states, p=start_distribution))
	states[0] = init_state
	for t in range(horizon):
		action = int(rng.choice(mdp.n_actions, p=action_probs[t, states[t]]))
		actions[t] = action
		new_state = int(rng.choice(mdp.n_states, p=mdp.transition[states[t], action]))
		states[t+1] = new_state

	return (states, actions)



def generate_demonstrations(
    mdp: TabularMDP,
    theta: np.ndarray,                # shape (n_features,) -- ground truth
    start_distribution: np.ndarray,   # shape (n_states,)
    horizon: int,
    n_demonstrations: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    # returns: list of length n_demonstrations, each shape (horizon+1,),
    # dtype=int -- directly pluggable into irl.maxent_linear.fit()'s
    # `demonstrations` argument.
    #
    # action_probs, _ = backward_pass(mdp, theta, horizon)   -- une seule fois
    # puis n_demonstrations appels à sample_trajectory, ne garder que .states
	demonstrations = []

	action_probs, _ = backward_pass(mdp, theta, horizon)
	for _ in range(n_demonstrations):
		states, _ = sample_trajectory(mdp, action_probs, start_distribution, horizon, rng)
		demonstrations.append(states)
	
	return demonstrations
    