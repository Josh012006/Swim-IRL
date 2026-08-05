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
this project's existing feature philosophy (all five worm-feature
candidates in the README are state functions too).

The policy trained here starts from SCRATCH (random init) -- it is NOT
warm-started from the loaded easy/medium/hard checkpoints. Those are only
used to GENERATE demonstrations (see data/simulate_nanogoal.py); using
them to also seed GCL's own learner would quietly hand it a policy that
already solves the task, defeating the point of testing whether GCL can
recover a competent policy from demonstrations alone.
'''
import numpy as np
import torch
import torch.nn as nn


class RewardNetwork(nn.Module):
    # small MLP: flattened observation (15,) -> scalar reward.
    # Architecture is your call -- 2-3 hidden layers, ~32-64 units each,
    # is a reasonable starting point, not something to over-engineer yet.
    def __init__(self, obs_dim: int = 15, hidden_dim: int = 64):
        ...

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: shape (batch, 15)
        # returns: shape (batch,) -- scalar reward per state
        ...


class RewardWrappedEnv:
    """Gym wrapper: same observation/action space as the real NanoGoal
    env, but step() returns reward_net(obs) instead of the environment's
    own true reward. This is the mechanism that lets SB3's ordinary
    PPO.learn() train the policy against the CURRENT reward estimate,
    with zero changes to SB3 itself -- exactly the 'hand policy
    optimization off to an existing RL library' plan from the README.
    """
    def __init__(self, env, reward_net: RewardNetwork):
        ...

    def step(self, action):
        # calls the real env.step(action) for dynamics/termination, but
        # replaces the returned reward with
        # reward_net(flatten_observation(obs)).item()
        ...


def compute_importance_weights(
    reward_net: RewardNetwork,
    background_trajectories: list[dict],  # same format as
                                           # sim.nanogoal_adapter.rollout's
                                           # return value
    policy,  # the SB3 PPO model that generated background_trajectories
) -> np.ndarray:
    """Self-normalized importance weights, one per background trajectory:

        w_i = [exp(sum_t r_theta(s_t)) / q(trajectory_i)]
              -------------------------------------------
                   sum_j [exp(sum_t r_theta(s_t)) / q(trajectory_j)]

    where q(trajectory) = prod_t pi_phi(a_t | s_t), evaluated via the
    policy's own action-distribution log-prob (SB3:
    policy.policy.get_distribution(obs).log_prob(action), summed over t
    in log-space then exponentiated -- same logsumexp-style numerical
    care as backward_pass, for the same reason).

    Returns shape (len(background_trajectories),), summing to 1.
    """
    ...


def reward_loss(
    reward_net: RewardNetwork,
    demonstrations: list[dict],
    background_trajectories: list[dict],
    importance_weights: np.ndarray,
) -> torch.Tensor:
    """GCL's per-iteration loss -- negative log-likelihood of the
    demonstrations under the current reward, sample-estimated:

        L(theta) = -mean_demo[ r_theta(tau) ] + sum_i w_i * r_theta(sigma_i)

    where r_theta(tau) = sum_t r_theta(s_t) (summed along a whole
    trajectory). First term pushes reward up on demonstrated states --
    the direct analogue of Ziebart's "empirical" term. Second term pushes
    it down on states the importance-weighted background samples suggest
    the current reward over-favors -- the analogue of Ziebart's
    "expected" term, now Monte-Carlo-estimated instead of exact-DP-summed.
    """
    ...


def train_gcl(
    nanogoal_path: str,
    demonstrations: list[dict],
    n_iterations: int = 100,
    n_background_trajectories_per_iteration: int = 20,
    policy_update_steps_per_iteration: int = 2048,
    reward_learning_rate: float = 1e-3,
) -> tuple[RewardNetwork, "PPO"]:
    """The alternating loop, once per iteration:
        1. collect n_background_trajectories_per_iteration rollouts from
           the CURRENT policy (sim.nanogoal_adapter.rollout,
           deterministic=False this time -- exploration matters here,
           unlike demonstration generation, since these samples are what
           estimate the partition function across the whole state space,
           not a record of best-known behavior)
        2. importance_weights = compute_importance_weights(...)
        3. one (or a few) gradient step(s) on
           reward_loss(reward_net, demonstrations, background, weights)
        4. policy.learn(total_timesteps=policy_update_steps_per_iteration,
           env=RewardWrappedEnv(create_env(nanogoal_path), reward_net))
           -- policy tracks the CURRENT reward estimate

    Returns the trained (reward_net, policy).
    """
    ...