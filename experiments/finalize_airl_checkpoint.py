"""Finalize a chosen AIRL checkpoint (by timesteps_done) as a cell's
official result -- for when training is stopped EARLY, mirroring
experiments/finalize_gcl_checkpoint.py's role and reasoning exactly
(see that file's module docstring for why simply killing the systemd
service leaves phase0b_airl_training.py's run_cell unable to do its own
finalization).

Copies the chosen checkpoint's reward_net/gen_algo into models/ (same
naming convention as run_cell's own finalization) and runs the SAME
sampled_recovery_gap evaluation methodology run_cell uses, PLUS the
same reward-component correlation and partial dependence checks
finalize_gcl_checkpoint.py runs for GCL -- reusing the exact same
plotting_phase0b functions via AIRLRewardNetAdapter below, so both
methods are inspected identically rather than with two different,
harder-to-compare diagnostics.

AIRL's reward_net (imitation's BasicShapedRewardNet) and policy
(gen_algo, an MlpPolicy) differ from GCL's in two ways this script
handles: reward_net.predict(state, action, next_state, done) takes four
arguments, not one, so AIRLRewardNetAdapter wraps it into the same
single-argument reward_net(obs) -> rewards interface
plotting_phase0b.py's functions expect (a zero action and done=False
for every call -- these two checks only vary STATE, matching AIRL's own
use_action=False training). And gen_algo is trained on FLATTENED
Box(15,) observations (irl.airl_wrapper.FlattenedNanoGoalEnv), not
NanoGoal-RL's native Dict -- rollouts and sampled_recovery_gap's
"recovered" side both need that wrapper; the expert side does not.

Usage:
    python -m experiments.finalize_airl_checkpoint --cell easy_easy --timesteps 90000000
"""
import argparse
import os
import shutil
import subprocess

import numpy as np
import torch
from stable_baselines3 import PPO
from imitation.rewards.reward_nets import BasicShapedRewardNet

from sim.nanogoal_adapter import create_env, load_policy, rollout
from irl.airl_wrapper import FlattenedNanoGoalEnv
from eval.recovery_continuous import sampled_recovery_gap
from experiments.plotting_phase0b import (
    collect_reward_component_data,
    compute_reward_component_correlations,
    plot_reward_components,
    plot_partial_dependence,
)

N_EVAL_SEEDS = 30  # matches experiments/phase0b_airl_training.py's own constant


class AIRLRewardNetAdapter:
    """Wraps imitation's BasicShapedRewardNet (state, action, next_state,
    done) calling convention into the same simple reward_net(obs) ->
    rewards interface irl.gcl.RewardNetwork and plotting_phase0b's
    functions expect -- lets the SAME reward-component correlation and
    partial dependence code serve both GCL and AIRL without duplicating
    it. Uses a zero action and done=False for every call: these two
    checks only ever vary STATE, matching AIRL's own use_action=False
    training (see irl/airl_wrapper.py) -- the action argument is never
    meaningfully used by reward_net itself in that mode, only required
    by predict()'s signature.
    """
    def __init__(self, airl_reward_net: BasicShapedRewardNet):
        self.airl_reward_net = airl_reward_net

    def __call__(self, obs_batch: torch.Tensor) -> torch.Tensor:
        obs_np = obs_batch.numpy()
        n = obs_np.shape[0]
        dummy_action = np.zeros((n, 2), dtype=np.float32)
        dummy_done = np.zeros(n, dtype=bool)
        rewards = self.airl_reward_net.predict(obs_np, dummy_action, obs_np, dummy_done)
        return torch.from_numpy(rewards).float()


def commit_and_push(cell: str, timesteps: int, extra_paths: list[str]) -> None:
    """Mirrors finalize_gcl_checkpoint.py's own commit_and_push -- same
    [skip ci] convention and same targeted-paths reasoning (never the
    whole experiments/results/ directory, to avoid sweeping in an
    unrelated stray file)."""
    paths = ["models/"] + extra_paths
    subprocess.run(["git", "add"] + paths, check=True)
    status = subprocess.run(
        ["git", "status", "--porcelain"] + paths,
        capture_output=True, text=True, check=True,
    )
    if not status.stdout.strip():
        print("\nNo changes to commit (models/ already up to date).")
        return

    subprocess.run(
        ["git", "commit", "-m",
         f"feat: finalize AIRL {cell} at {timesteps} timesteps [skip ci]"],
        check=True,
    )
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    subprocess.run(["git", "push", "origin", branch], check=True)
    print(f"\nCommitted and pushed models/ to branch '{branch}'.")


def main(cell: str, timesteps: int, seed: int, commit: bool, n_component_episodes: int) -> None:
    nanogoal_path = "external/NanoGoal-RL"
    model_difficulty, seed_mode = cell.split("_", 1)
    checkpoint_dir = f"experiments/results/checkpoints/airl_{cell}"

    reward_net_src = f"{checkpoint_dir}/reward_net_ts{timesteps}.pt"
    gen_algo_src = f"{checkpoint_dir}/gen_algo_ts{timesteps}.zip"
    if not os.path.exists(reward_net_src):
        raise FileNotFoundError(f"No checkpoint found at {reward_net_src}")

    os.makedirs("models", exist_ok=True)
    reward_net_dst = f"models/phase0b_airl_{cell}_reward_net.pt"
    policy_dst = f"models/phase0b_airl_{cell}_policy.zip"
    shutil.copy(reward_net_src, reward_net_dst)
    shutil.copy(gen_algo_src, policy_dst)
    print(f"Promoted checkpoint at {timesteps} timesteps to:")
    print(f"  {reward_net_dst}")
    print(f"  {policy_dst}")

    # recovered_policy is an MlpPolicy trained on flattened Box(15,)
    # observations -- must be evaluated through the same wrapper it was
    # trained with (see module docstring); the expert stays on the
    # native Dict env, matching phase0b_airl_training.py's own run_cell.
    expert_eval_env = create_env(nanogoal_path)
    recovered_eval_env = FlattenedNanoGoalEnv(create_env(nanogoal_path))
    recovered_policy = PPO.load(f"{checkpoint_dir}/gen_algo_ts{timesteps}", env=recovered_eval_env, device="cpu")
    expert_policy = load_policy(nanogoal_path, model_difficulty, create_env(nanogoal_path))

    rng = np.random.default_rng(seed)
    eval_seeds = [int(s) for s in rng.integers(0, 1_000_000, size=N_EVAL_SEEDS)]

    gap = sampled_recovery_gap(
        expert_eval_env, expert_policy, recovered_eval_env, recovered_policy, eval_seeds
    )

    print(
        f"\nFinal evaluation for '{cell}' at {timesteps} timesteps "
        f"({N_EVAL_SEEDS} held-out seeds):"
    )
    print(f"  expert success_rate:    {gap['expert']['success_rate']:.3f}")
    print(f"  recovered success_rate: {gap['recovered']['success_rate']:.3f}")
    print(f"  return_gap:       {gap['return_gap']:.3f}")
    print(f"  success_rate_gap: {gap['success_rate_gap']:.3f}")

    # Reward-component analysis -- same two checks as
    # finalize_gcl_checkpoint.py, via AIRLRewardNetAdapter so the SAME
    # plotting_phase0b functions run unchanged for AIRL. Rollouts use
    # the FLATTENED env (recovered_policy's own observation space) --
    # collect_reward_component_data/plot_partial_dependence both index
    # into flattened Box(15,) states regardless of algorithm, so this
    # is the correct env for both GCL and AIRL's own diagnostics.
    airl_reward_net = BasicShapedRewardNet(
        observation_space=recovered_eval_env.observation_space,
        action_space=recovered_eval_env.action_space,
        use_action=False,
    )
    airl_reward_net.load_state_dict(torch.load(reward_net_src))
    airl_reward_net.eval()
    reward_net_adapter = AIRLRewardNetAdapter(airl_reward_net)

    component_rng = np.random.default_rng(seed + 777_777)
    component_trajectories = [
        rollout(
            recovered_eval_env, recovered_policy,
            seed=int(component_rng.integers(0, 1_000_000)),
            deterministic=True,
        )
        for _ in range(n_component_episodes)
    ]
    component_data = collect_reward_component_data(reward_net_adapter, component_trajectories)
    correlations = compute_reward_component_correlations(component_data)

    print(
        f"\nReward-component correlations ({len(component_data['predicted'])} "
        f"pooled states across {n_component_episodes} episodes):"
    )
    print(f"  recovered vs dist_goal:  {correlations['predicted_vs_dist_goal']:>7.3f}  "
          f"(true reward: {correlations['true_vs_dist_goal']:>7.3f}, expect negative)")
    print(f"  recovered vs min_lidar:  {correlations['predicted_vs_min_lidar']:>7.3f}  "
          f"(true reward: {correlations['true_vs_min_lidar']:>7.3f}, expect positive)")

    components_output_path = f"experiments/results/phase0b_airl_{cell}_ts{timesteps}_reward_components.png"
    plot_reward_components(component_data, correlations, components_output_path)
    print(f"Reward-component scatter plots saved to {components_output_path}")

    partial_dep_output_path = f"experiments/results/phase0b_airl_{cell}_ts{timesteps}_partial_dependence.png"
    plot_partial_dependence(reward_net_adapter, component_data["all_states"], partial_dep_output_path)
    print(f"Partial dependence plots saved to {partial_dep_output_path}")

    if commit:
        commit_and_push(cell, timesteps, [components_output_path, partial_dep_output_path])
    else:
        print("\n--no-commit: models/ left uncommitted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=str, required=True, help="e.g. easy_easy")
    parser.add_argument("--timesteps", type=int, required=True,
                         help="matches a checkpoint_dir's reward_net_ts{N}.pt suffix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-commit", action="store_true",
        help="skip the git commit/push of models/ -- promote the checkpoint "
             "and print the evaluation, but leave committing to you",
    )
    parser.add_argument(
        "--n-component-episodes", type=int, default=20,
        help="episodes rolled out for the reward-component correlation and "
             "partial dependence checks",
    )
    args = parser.parse_args()
    main(args.cell, args.timesteps, args.seed, not args.no_commit, args.n_component_episodes)