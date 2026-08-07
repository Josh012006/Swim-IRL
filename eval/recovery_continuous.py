"""Sampling-based reward recovery metric for Phase 0b (continuous state,
no exact DP available -- see eval/recovery.py for the tabular Phase 0a
version this is the continuous analogue of).

Same idea as Phase 0a's EVD, realized via rollouts instead of exact value
iteration: compare the TRUE-reward value achieved by the policy trained
against the RECOVERED reward, to the TRUE-reward value achieved by the
expert policy used to generate demonstrations in the first place
(already near-optimal under the true reward, by construction -- that's
what it was trained for). Both policies are evaluated under NanoGoal-RL's
own true reward (returned, but otherwise unused, in every rollout's
"rewards" field), on the SAME held-out seed set, so the comparison is
apples to apples regardless of the affine-transform ambiguity in what the
recovered reward numerically looks like.

IMPORTANT DIFFERENCE FROM PHASE 0A'S EVD: this is NOT guaranteed >= 0.
Phase 0a's EVD used hard_value_iteration -- a policy PROVABLY optimal
under its reward. The "expert" here is just a trained PPO policy, not a
provable optimum, so the recovered-reward policy slightly OUTPERFORMING
it (a small negative gap) is possible and not a bug -- unlike a negative
EVD in Phase 0a, which would have been a real red flag.
"""
import numpy as np

from sim.nanogoal_adapter import rollout


def evaluate_policy_under_true_reward(
    env,
    policy,
    seeds: list[int],
    deterministic: bool = True,
) -> dict:
    """Rolls out `policy` on every seed in `seeds`, scoring it with
    NanoGoal-RL's own true reward (env.py's step() reward -- never any
    recovered reward_net, regardless of which policy is being evaluated).

    Returns:
        {
            "mean_return": float,    -- average sum of true reward per episode
            "success_rate": float,   -- fraction of episodes with is_success
            "returns": np.ndarray,   -- shape (len(seeds),), per-episode
            "successes": np.ndarray, -- shape (len(seeds),), bool per episode
        }
    """
    returns = []
    successes = []
    for seed in seeds:
        trajectory = rollout(env, policy, seed, deterministic=deterministic)
        returns.append(trajectory["rewards"].sum())
        successes.append(trajectory["is_success"])

    returns = np.array(returns)
    successes = np.array(successes)
    return {
        "mean_return": float(returns.mean()),
        "success_rate": float(successes.mean()),
        "returns": returns,
        "successes": successes,
    }


def sampled_recovery_gap(
    expert_env,
    expert_policy,
    recovered_env,
    recovered_policy,
    seeds: list[int],
    deterministic: bool = True,
) -> dict:
    """The continuous analogue of Phase 0a's EVD: how much TRUE-reward
    value is lost by the policy trained against the RECOVERED reward,
    relative to the expert policy used to generate demonstrations.
    0 = behaviorally indistinguishable from the expert under the true
    reward; positive = the recovered-reward policy underperforms; a small
    negative value is possible (see module docstring) and not a bug.

    expert_env and recovered_env are separate parameters, not one shared
    env: expert_policy and the GCL-recovered policy both use NanoGoal-RL's
    native Dict observation (MultiInputPolicy), but AIRL's recovered
    policy is trained on the flattened Box(15,) representation
    (irl/airl_wrapper.py's FlattenedNanoGoalEnv) -- passing the wrong env
    to the wrong policy fails immediately with a clear assertion error
    from SB3 itself (verified), not a silent wrong number, but the two
    envs still need to be supplied correctly by the caller.

    Returns:
        {
            "expert": evaluate_policy_under_true_reward(...) result,
            "recovered": evaluate_policy_under_true_reward(...) result,
            "return_gap": float,       -- expert mean_return - recovered mean_return
            "success_rate_gap": float, -- expert success_rate - recovered success_rate
        }
    """
    expert_results = evaluate_policy_under_true_reward(
        expert_env, expert_policy, seeds, deterministic
    )
    recovered_results = evaluate_policy_under_true_reward(
        recovered_env, recovered_policy, seeds, deterministic
    )
    return {
        "expert": expert_results,
        "recovered": recovered_results,
        "return_gap": expert_results["mean_return"] - recovered_results["mean_return"],
        "success_rate_gap": expert_results["success_rate"] - recovered_results["success_rate"],
    }