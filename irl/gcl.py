'''Guided Cost Learning (Finn, Levine & Abbeel, 2016): sample-based
MaxEnt IRL for continuous state/action spaces, replacing exact tabular DP
with importance-weighted policy rollouts.

Alternates two steps every iteration: (1) update the reward network using
demonstrations AND importance-weighted samples from the current policy,
(2) update the policy (SB3 PPO) to better optimize the current reward
estimate. Step (2) is what makes this "guided" -- it keeps the sampling
distribution tracking the current reward, so the importance weights in
step (1) don't blow up the way they would against a fixed, unrelated
sampling policy.

Reward is state-only (r_theta(s), not r_theta(s,a)) -- consistent with
AIRL's state-only design (see irl/airl_wrapper.py, once written) and with
this project's existing feature philosophy. Known simplification: see
README Limitations for why this can't represent NanoGoal-RL's actual
action-dependent effort/spinning penalty.

The policy trained here starts from SCRATCH (random init) -- it is NOT
warm-started from the loaded easy/medium/hard checkpoints. Those are only
used to GENERATE demonstrations (see data/simulate_nanogoal.py); using
them to also seed GCL's own learner would quietly hand it a policy that
already solves the task, defeating the point of testing whether GCL can
recover a competent policy from demonstrations alone.
'''
import numpy as np
import scipy.special
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO

from sim.nanogoal_adapter import flatten_observation, create_env, rollout


class RewardNetwork(nn.Module):
    # small MLP: flattened observation (15,) -> scalar reward.
    def __init__(self, obs_dim: int = 15, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: shape (batch, 15)
        # returns: shape (batch,) -- squeeze the trailing singleton dim
        # from the last Linear(hidden_dim, 1) layer, or every batched
        # use downstream (importance weights, reward_loss) silently
        # broadcasts wrong via (batch,1) * (batch,) -> (batch,batch)
        return self.network(obs).squeeze(-1)


class RewardWrappedEnv(gym.Wrapper):
    """Same observation/action space as the real NanoGoal env (inherited
    from gym.Wrapper -- reset(), observation_space, action_space, etc.
    all pass through automatically), but step() returns reward_net(obs)
    instead of the environment's own true reward. This is what lets
    SB3's ordinary PPO.learn() train the policy against the CURRENT
    reward estimate, with zero changes to SB3 itself.
    """
    def __init__(self, env, reward_net: RewardNetwork):
        super().__init__(env)
        self.reward_net = reward_net

    def step(self, action):
        obs, _true_reward, terminated, truncated, info = self.env.step(action)

        obs_tensor = torch.from_numpy(flatten_observation(obs)).float().unsqueeze(0)
        with torch.no_grad():
            reward = self.reward_net(obs_tensor).item()

        return obs, reward, terminated, truncated, info


def _unflatten_observations_batch(flat_obs_batch: np.ndarray) -> dict:
    """Inverse of sim.nanogoal_adapter.flatten_observation, batched.
    Needed because rollout() only stores the flattened form, but SB3's
    MultiInputPolicy needs the original Dict structure to evaluate
    log-probabilities of actions. flat_obs_batch: shape (N, 15)."""
    return {
        "agent": torch.from_numpy(flat_obs_batch[:, 0:2]).float(),
        "mvt": torch.from_numpy(flat_obs_batch[:, 2:5]).float(),
        "delta_goal": torch.from_numpy(flat_obs_batch[:, 5:7]).float(),
        "lidar": torch.from_numpy(flat_obs_batch[:, 7:15]).float(),
    }


def compute_importance_weights(
    reward_net: RewardNetwork,
    background_trajectories: list[dict],
    policy,
) -> np.ndarray:
    """Self-normalized importance weights, one per background trajectory:

        w_i = softmax_i( sum_t r_theta(s_t) - sum_t log pi_phi(a_t|s_t) )

    Same log-space computation as backward_pass's logsumexp, for the same
    numerical-stability reason -- never exponentiate the raw (unbounded)
    sum_t r_theta(s_t) directly.

    Returns shape (len(background_trajectories),), summing to 1.
    """
    log_ratios = []
    for traj in background_trajectories:
        obs_all = torch.from_numpy(traj["observations"]).float()       # (T+1, 15)
        obs_for_actions = traj["observations"][:-1]                     # (T, 15)
        actions = torch.from_numpy(traj["actions"]).float()             # (T, 2)

        with torch.no_grad():
            trajectory_reward = reward_net(obs_all).sum()

            obs_dict = _unflatten_observations_batch(obs_for_actions)
            _, log_probs, _ = policy.policy.evaluate_actions(obs_dict, actions)
            log_q = log_probs.sum()

        log_ratios.append((trajectory_reward - log_q).item())

    log_ratios = np.array(log_ratios)
    log_Z = scipy.special.logsumexp(log_ratios)
    return np.exp(log_ratios - log_Z)


def reward_loss(
    reward_net: RewardNetwork,
    demonstrations: list[dict],
    background_trajectories: list[dict],
    importance_weights: np.ndarray,
) -> torch.Tensor:
    """GCL's per-iteration loss -- negative log-likelihood of the
    demonstrations under the current reward, sample-estimated:

        L(theta) = -mean_demo[ r_theta(tau) ] + sum_i w_i * r_theta(sigma_i)

    First term (gradient flows -- no torch.no_grad here, unlike
    compute_importance_weights) pushes reward up on demonstrated states.
    Second term pushes it down on states the importance-weighted
    background samples suggest the current reward over-favors.
    """
    demo_rewards = [
        reward_net(torch.from_numpy(demo["observations"]).float()).sum()
        for demo in demonstrations
    ]
    demo_term = torch.stack(demo_rewards).mean()

    background_rewards = torch.stack([
        reward_net(torch.from_numpy(traj["observations"]).float()).sum()
        for traj in background_trajectories
    ])
    weights_tensor = torch.from_numpy(importance_weights).float()
    background_term = (weights_tensor * background_rewards).sum()

    return -demo_term + background_term


def train_gcl(
    nanogoal_path: str,
    demonstrations: list[dict],
    n_iterations: int = 100,
    n_background_trajectories_per_iteration: int = 20,
    policy_update_steps_per_iteration: int = 2048,
    reward_learning_rate: float = 1e-3,
    seed: int = 0,
) -> tuple[RewardNetwork, PPO, dict]:
    """The alternating loop, once per iteration:
        1. collect n_background_trajectories_per_iteration rollouts from
           the CURRENT policy (deterministic=False -- exploration matters
           here, unlike demonstration generation, since these samples
           estimate the partition function across the state space, not a
           record of best-known behavior)
        2. importance_weights = compute_importance_weights(...)
        3. one gradient step on reward_loss(...)
        4. policy.learn(...) against the SAME wrapped_env object, whose
           reward_net reference is updated in place -- no need to
           recreate the wrapper each iteration

    Returns (reward_net, policy, history), where history is
    {"loss": np.ndarray shape (n_iterations,),
     "success_rate": np.ndarray shape (n_iterations,)} -- the background
    rollouts' own success rate each iteration, a cheap way to see whether
    the policy is learning to solve the task as the recovered reward
    improves. Used by experiments/plotting_phase0b.py.
    """
    background_env = create_env(nanogoal_path)
    reward_net = RewardNetwork()
    optimizer = torch.optim.Adam(reward_net.parameters(), lr=reward_learning_rate)

    wrapped_env = RewardWrappedEnv(create_env(nanogoal_path), reward_net)
    policy = PPO("MultiInputPolicy", wrapped_env, verbose=0, seed=seed)

    rng = np.random.default_rng(seed)

    loss_history = np.zeros(n_iterations)
    success_rate_history = np.zeros(n_iterations)

    for iteration in range(n_iterations):
        background_trajectories = [
            rollout(
                background_env, policy,
                seed=int(rng.integers(0, 1_000_000)),
                deterministic=False,
            )
            for _ in range(n_background_trajectories_per_iteration)
        ]

        importance_weights = compute_importance_weights(
            reward_net, background_trajectories, policy
        )

        optimizer.zero_grad()
        loss = reward_loss(reward_net, demonstrations, background_trajectories, importance_weights)
        loss.backward()
        optimizer.step()

        policy.learn(
            total_timesteps=policy_update_steps_per_iteration,
            reset_num_timesteps=False,
        )

        loss_history[iteration] = loss.item()
        success_rate_history[iteration] = np.mean(
            [t["is_success"] for t in background_trajectories]
        )

        print(
            f"iteration {iteration}: reward_loss={loss.item():.4f}  "
            f"background_success_rate={success_rate_history[iteration]:.2f}"
        )

    history = {"loss": loss_history, "success_rate": success_rate_history}
    return reward_net, policy, history