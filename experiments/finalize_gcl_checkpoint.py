"""Finalize a chosen GCL checkpoint iteration as a cell's official
result -- for when training is stopped EARLY (a checkpoint iteration is
judged "good enough" before BUDGET's full n_iterations completes).

Simply killing the systemd service mid-run leaves
phase0b_gcl_training.py's run_cell unable to do its own finalization
step (saving to models/, computing the final sampled_recovery_gap) --
that code lives AFTER the train_gcl(...) call and never runs if
train_gcl is interrupted rather than returning normally. This script
does the same finalization work run_cell would have, on whichever
checkpoint iteration you choose instead of whatever iteration training
happened to reach when BUDGET ran out.

Copies the chosen checkpoint's reward_net/policy into models/ (same
naming convention as run_cell's own finalization -- so
query_reward_model.py --final and any other tool expecting models/
works normally afterward) and runs the SAME sampled_recovery_gap
evaluation methodology run_cell uses (same N_EVAL_SEEDS, same expert
policy, same eval env). Not necessarily bit-identical held-out seeds to
what a full run would have used (those depend on exactly how many
random draws demo generation happened to consume, itself stochastic --
regenerating that exactly wasn't worth the complexity here), but
methodologically equivalent and directly comparable.

Usage:
    python -m experiments.finalize_gcl_checkpoint --cell easy_easy --iteration 600
"""
import argparse
import os
import shutil
import subprocess

import numpy as np
import torch
from stable_baselines3 import PPO

from sim.nanogoal_adapter import create_env, load_policy, rollout
from irl.gcl import RewardNetwork
from eval.recovery_continuous import sampled_recovery_gap
from experiments.plotting_phase0b import (
    collect_reward_component_data,
    compute_reward_component_correlations,
    plot_reward_components,
    plot_partial_dependence,
)

N_EVAL_SEEDS = 30  # matches experiments/phase0b_gcl_training.py's own constant


def commit_and_push(cell: str, iteration: int, extra_paths: list[str]) -> None:
    """Mirrors .github/ci/train_phase0b.sh's own commit pattern -- same
    [skip ci] convention, so this doesn't trigger a pointless new
    workflow run (the flag file wasn't touched, so it would just read
    train=false and stop immediately anyway, but no reason to spend a
    CI run confirming that). Only commits if there's actually something
    to commit -- git status --porcelain being empty means this exact
    content was already committed (e.g. running this script twice on
    the same iteration). extra_paths are the specific new plot files
    this run produced -- added by exact path, not the whole
    experiments/results/ directory, so an unrelated stray file sitting
    there doesn't get swept into this commit.
    """
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
         f"feat: finalize GCL {cell} at iteration {iteration} [skip ci]"],
        check=True,
    )
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    subprocess.run(["git", "push", "origin", branch], check=True)
    print(f"\nCommitted and pushed models/ to branch '{branch}'.")


def main(cell: str, iteration: int, seed: int, commit: bool, n_component_episodes: int) -> None:
    nanogoal_path = "external/NanoGoal-RL"
    model_difficulty, seed_mode = cell.split("_", 1)
    checkpoint_dir = f"experiments/results/checkpoints/gcl_{cell}"

    reward_net_src = f"{checkpoint_dir}/reward_net_iter{iteration}.pt"
    policy_src = f"{checkpoint_dir}/policy_iter{iteration}.zip"
    if not os.path.exists(reward_net_src):
        raise FileNotFoundError(f"No checkpoint found at {reward_net_src}")

    os.makedirs("models", exist_ok=True)
    reward_net_dst = f"models/phase0b_gcl_{cell}_reward_net.pt"
    policy_dst = f"models/phase0b_gcl_{cell}_policy.zip"
    shutil.copy(reward_net_src, reward_net_dst)
    shutil.copy(policy_src, policy_dst)
    print(f"Promoted checkpoint iteration {iteration} to:")
    print(f"  {reward_net_dst}")
    print(f"  {policy_dst}")

    env = create_env(nanogoal_path)
    recovered_policy = PPO.load(f"{checkpoint_dir}/policy_iter{iteration}", env=env, device="cpu")
    expert_policy = load_policy(nanogoal_path, model_difficulty, create_env(nanogoal_path))

    rng = np.random.default_rng(seed)
    eval_seeds = [int(s) for s in rng.integers(0, 1_000_000, size=N_EVAL_SEEDS)]

    gap = sampled_recovery_gap(env, expert_policy, env, recovered_policy, eval_seeds)

    print(
        f"\nFinal evaluation for '{cell}' at iteration {iteration} "
        f"({N_EVAL_SEEDS} held-out seeds):"
    )
    print(f"  expert success_rate:    {gap['expert']['success_rate']:.3f}")
    print(f"  recovered success_rate: {gap['recovered']['success_rate']:.3f}")
    print(f"  return_gap:       {gap['return_gap']:.3f}")
    print(f"  success_rate_gap: {gap['success_rate_gap']:.3f}")

    # Reward-component analysis -- the neural-net analogue of reading
    # off individual theta_i weights in Phase 0a's linear MaxEnt IRL
    # (see plotting_phase0b.collect_reward_component_data's docstring
    # for the full reasoning). Uses n_component_episodes fresh
    # deterministic rollouts from THIS checkpoint's own policy -- more
    # episodes than sampled_recovery_gap's own eval_seeds by default,
    # since these checks need enough POOLED STATES (not just enough
    # episodes) for the correlations and the observed feature ranges to
    # be reliable rather than dominated by one or two episodes.
    reward_net = RewardNetwork()
    reward_net.load_state_dict(torch.load(reward_net_src))
    reward_net.eval()

    component_rng = np.random.default_rng(seed + 777_777)
    component_trajectories = [
        rollout(
            env, recovered_policy,
            seed=int(component_rng.integers(0, 1_000_000)),
            deterministic=True,
        )
        for _ in range(n_component_episodes)
    ]
    component_data = collect_reward_component_data(reward_net, component_trajectories)
    correlations = compute_reward_component_correlations(component_data)

    print(
        f"\nReward-component correlations ({len(component_data['predicted'])} "
        f"pooled states across {n_component_episodes} episodes):"
    )
    print(f"  recovered vs dist_goal:  {correlations['predicted_vs_dist_goal']:>7.3f}  "
          f"(true reward: {correlations['true_vs_dist_goal']:>7.3f}, expect negative)")
    print(f"  recovered vs min_lidar:  {correlations['predicted_vs_min_lidar']:>7.3f}  "
          f"(true reward: {correlations['true_vs_min_lidar']:>7.3f}, expect positive)")

    components_output_path = f"experiments/results/phase0b_gcl_{cell}_iter{iteration}_reward_components.png"
    plot_reward_components(component_data, correlations, components_output_path)
    print(f"Reward-component scatter plots saved to {components_output_path}")

    partial_dep_output_path = f"experiments/results/phase0b_gcl_{cell}_iter{iteration}_partial_dependence.png"
    plot_partial_dependence(reward_net, component_data["all_states"], partial_dep_output_path)
    print(f"Partial dependence plots saved to {partial_dep_output_path}")

    if commit:
        commit_and_push(cell, iteration, [components_output_path, partial_dep_output_path])
    else:
        print("\n--no-commit: models/ left uncommitted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=str, required=True, help="e.g. easy_easy")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-commit", action="store_true",
        help="skip the git commit/push of models/ -- promote the checkpoint "
             "and print the evaluation, but leave committing to you",
    )
    parser.add_argument(
        "--n-component-episodes", type=int, default=20,
        help="episodes rolled out for the reward-component correlation and "
             "partial dependence checks -- more than sampled_recovery_gap's "
             "own eval seeds by default, since these need enough POOLED "
             "STATES (not just enough episodes) for reliable correlations "
             "and observed feature ranges",
    )
    args = parser.parse_args()
    main(args.cell, args.iteration, args.seed, not args.no_commit, args.n_component_episodes)
