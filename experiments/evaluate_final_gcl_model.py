"""Full evaluation of a GCL cell's FINAL result -- everything
finalize_gcl_checkpoint.py runs (sampled_recovery_gap, reward-component
correlations, partial dependence), but sourced from models/ instead of
experiments/results/checkpoints/gcl_{cell}/.

This matters for WHERE it can run: experiments/results/checkpoints/ is
gitignored (large, frequently-changing binary files, never meant to be
permanent) and only ever exists on whichever machine actually trained
the cell -- a `git pull` on any other machine (e.g. a local laptop)
cannot retrieve it, only ever a FileNotFoundError. models/ IS committed
(see .github/ci/train_phase0b.sh's git add and
finalize_gcl_checkpoint.py's own commit step) -- so this script can run
anywhere the repo has been pulled, no access to the original training
machine required. Purely local: never touches git -- unlike
finalize_gcl_checkpoint.py (which promotes a NEW checkpoint into
models/ and so has something worth committing), this script only reads
an ALREADY-committed models/ and writes plots/summary for local
inspection.

Also writes results to a small JSON summary file (unlike
finalize_gcl_checkpoint.py, which only prints to the console) -- so a
number you look at once isn't lost the moment the terminal scrolls
past it; rerunning later, or reading the JSON, gets you back to it
without repeating the computation from memory.

Usage:
    python -m experiments.evaluate_final_gcl_model --cell easy_easy
"""
import argparse
import json

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


def load_gcl_final_artifacts(cell: str, env) -> tuple[RewardNetwork, PPO]:
    reward_net = RewardNetwork()
    reward_net.load_state_dict(torch.load(f"models/phase0b_gcl_{cell}_reward_net.pt"))
    reward_net.eval()
    policy = PPO.load(f"models/phase0b_gcl_{cell}_policy", env=env, device="cpu")
    return reward_net, policy


def main(cell: str, seed: int, n_component_episodes: int) -> None:
    nanogoal_path = "external/NanoGoal-RL"
    model_difficulty, seed_mode = cell.split("_", 1)

    env = create_env(nanogoal_path)
    recovered_policy_path = f"models/phase0b_gcl_{cell}_policy"
    reward_net, recovered_policy = load_gcl_final_artifacts(cell, env)

    expert_policy = load_policy(nanogoal_path, model_difficulty, create_env(nanogoal_path))
    rng = np.random.default_rng(seed)
    eval_seeds = [int(s) for s in rng.integers(0, 1_000_000, size=N_EVAL_SEEDS)]
    gap = sampled_recovery_gap(env, expert_policy, env, recovered_policy, eval_seeds)

    print(f"\nFinal-model evaluation for '{cell}' ({N_EVAL_SEEDS} held-out seeds):")
    print(f"  expert success_rate:    {gap['expert']['success_rate']:.3f}")
    print(f"  recovered success_rate: {gap['recovered']['success_rate']:.3f}")
    print(f"  return_gap:       {gap['return_gap']:.3f}")
    print(f"  success_rate_gap: {gap['success_rate_gap']:.3f}")

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

    components_output_path = f"experiments/results/phase0b_gcl_{cell}_final_reward_components.png"
    plot_reward_components(component_data, correlations, components_output_path)
    print(f"Reward-component scatter plots saved to {components_output_path}")

    partial_dep_output_path = f"experiments/results/phase0b_gcl_{cell}_final_partial_dependence.png"
    plot_partial_dependence(reward_net, component_data["all_states"], partial_dep_output_path)
    print(f"Partial dependence plots saved to {partial_dep_output_path}")

    summary = {
        "cell": cell,
        "n_eval_seeds": N_EVAL_SEEDS,
        "n_component_episodes": n_component_episodes,
        "expert_success_rate": gap["expert"]["success_rate"],
        "recovered_success_rate": gap["recovered"]["success_rate"],
        "return_gap": gap["return_gap"],
        "success_rate_gap": gap["success_rate_gap"],
        "correlations": correlations,
    }
    summary_path = f"experiments/results/phase0b_gcl_{cell}_final_evaluation.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=str, required=True, help="e.g. easy_easy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-component-episodes", type=int, default=20,
        help="episodes rolled out for the reward-component correlation and "
             "partial dependence checks",
    )
    args = parser.parse_args()
    main(args.cell, args.seed, args.n_component_episodes)