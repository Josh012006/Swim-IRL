"""Load a saved Phase 0b GCL reward network + policy from models/ (see
phase0b_gcl_training.py's run_cell) and actually query them, instead of
having a trained reward_net sitting on disk with no way to inspect what
it learned.

Rolls out n_episodes with the SAVED policy, pairs the reward_net's
prediction against NanoGoal-RL's own true reward at every step (reusing
experiments.plotting_phase0b.collect_reward_comparison_data -- the same
function the training pipeline never actually called, so this is also
the first real use of it), prints summary statistics + a sample of raw
pairs, and saves the scatter plot.

Usage:
    python -m experiments.query_reward_model --cell easy_easy --n-episodes 10
"""
import argparse

import numpy as np
import torch
from stable_baselines3 import PPO

from sim.nanogoal_adapter import create_env, rollout
from irl.gcl import RewardNetwork
from experiments.plotting_phase0b import collect_reward_comparison_data, plot_reward_comparison


def load_gcl_artifacts(cell: str, env) -> tuple[RewardNetwork, PPO]:
    reward_net = RewardNetwork()
    reward_net.load_state_dict(torch.load(f"models/phase0b_gcl_{cell}_reward_net.pt"))
    reward_net.eval()
    policy = PPO.load(f"models/phase0b_gcl_{cell}_policy", env=env, device="cpu")
    return reward_net, policy


def main(cell: str, seed: int, n_episodes: int) -> None:
    nanogoal_path = "external/NanoGoal-RL"
    env = create_env(nanogoal_path)
    reward_net, policy = load_gcl_artifacts(cell, env)

    rng = np.random.default_rng(seed)
    trajectories = [
        rollout(env, policy, seed=int(rng.integers(0, 1_000_000)), deterministic=True)
        for _ in range(n_episodes)
    ]

    predicted, true = collect_reward_comparison_data(reward_net, trajectories)
    correlation = np.corrcoef(predicted, true)[0, 1]

    print(f"GCL reward_net for cell '{cell}': {len(predicted)} state-reward pairs "
          f"across {n_episodes} episodes")
    print(f"  correlation with true reward: {correlation:.3f}")
    print(f"  predicted reward: mean={predicted.mean():.3f}  std={predicted.std():.3f}")
    print(f"  true reward:      mean={true.mean():.3f}  std={true.std():.3f}")
    print()
    print("First 10 (predicted, true) pairs:")
    for p, t in zip(predicted[:10], true[:10]):
        print(f"  predicted={p:.4f}   true={t:.4f}")

    output_path = f"experiments/results/phase0b_gcl_{cell}_reward_comparison.png"
    plot_reward_comparison(predicted, true, output_path=output_path)
    print(f"\nScatter plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=str, required=True, help="e.g. easy_easy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=10)
    args = parser.parse_args()
    main(args.cell, args.seed, args.n_episodes)