"""Importance-weight degeneracy diagnostic for GCL.

Computes the Effective Sample Size (ESS = 1 / sum(w_i^2)) of
compute_importance_weights' output, at several saved checkpoint
iterations, comparing RAW (unclipped, clip_percentile=100.0) against
CLIPPED (the percentile actually used in training) weights side by side
on the SAME sampled background trajectories -- so the comparison isn't
confounded by two different random draws. This is what originally
diagnosed the degeneracy problem (raw ESS collapsing to ~1) and what
later confirmed log-ratio clipping fixes most of it -- both numbers stay
useful going forward, not just one: raw ESS tracks whether the
UNDERLYING estimation problem is getting better or worse as training
progresses (e.g. as train/std or reward_net's own behavior changes),
while clipped ESS tracks what reward_loss's gradient is ACTUALLY
computed from at each checkpoint.

ESS is the standard diagnostic for importance-sampling weight
degeneracy, not something invented for this project: uniform weights
(w_i = 1/N for all i) give ESS = N (every background trajectory
contributes roughly equally to reward_loss's gradient); one dominant
weight (w_i near 1, rest near 0) gives ESS near 1 (degenerate --
reward_loss's gradient is effectively estimated from a SINGLE
trajectory, however many N were actually sampled). See irl/gcl.py's
compute_importance_weights
docstring for the full diagnosis and the clipping fix's citation trail
(Ionides 2008, per-decision pooling per Precup 2000).

Usage:
    python -m experiments.diagnose_importance_weights \
        --cell easy_easy --iterations 100 200 300 400 500 600
    python -m experiments.diagnose_importance_weights \
        --cell easy_easy --iterations 300 550 --clip-percentile 90
"""
import argparse

import numpy as np
from stable_baselines3 import PPO

from sim.nanogoal_adapter import create_env, load_test_seeds, sample_seed_from_mode, rollout
from irl.gcl import RewardNetwork, compute_importance_weights
import torch


def load_checkpoint(checkpoint_dir: str, iteration: int, env) -> tuple[RewardNetwork, PPO]:
    reward_net = RewardNetwork()
    reward_net.load_state_dict(
        torch.load(f"{checkpoint_dir}/reward_net_iter{iteration}.pt")
    )
    reward_net.eval()
    policy = PPO.load(f"{checkpoint_dir}/policy_iter{iteration}", env=env, device="cpu")
    return reward_net, policy


def effective_sample_size(weights: np.ndarray) -> float:
    return 1.0 / np.sum(weights ** 2)


def main(
    cell: str,
    iterations: list[int],
    seed_mode: str,
    n_background: int,
    clip_percentile: float,
    seed: int,
) -> None:
    nanogoal_path = "external/NanoGoal-RL"
    checkpoint_dir = f"experiments/results/checkpoints/gcl_{cell}"

    test_seeds = load_test_seeds(nanogoal_path)
    env = create_env(nanogoal_path)
    rng = np.random.default_rng(seed)

    header = (
        f"{'iteration':>10}  {'raw ESS':>9}  {'raw ESS/N':>10}  "
        f"{'clip'+str(clip_percentile)+' ESS':>13}  {'clip ESS/N':>11}  {'N_pool':>7}"
    )
    print(header)
    print("-" * len(header))
    for iteration in iterations:
        try:
            reward_net, policy = load_checkpoint(checkpoint_dir, iteration, env)
        except FileNotFoundError:
            print(f"{iteration:>10}  -- checkpoint not found, skipping --")
            continue

        background_trajectories = [
            rollout(
                env, policy,
                seed=sample_seed_from_mode(test_seeds, seed_mode, rng),
                deterministic=False,
            )
            for _ in range(n_background)
        ]

        # Same background_trajectories for both calls -- clip_percentile=100.0
        # is a no-op (np.percentile at 100 = the max value, clipping to it
        # clips nothing), giving the RAW, unclipped weights for direct
        # comparison against the CLIPPED ones on identical data.
        raw_weights = compute_importance_weights(
            reward_net, background_trajectories, policy, clip_percentile=100.0
        )
        clipped_weights = compute_importance_weights(
            reward_net, background_trajectories, policy, clip_percentile=clip_percentile
        )

        n_pool = len(raw_weights)  # actual trajectory count (v3: back to
                                    # per-trajectory weighting, see
                                    # irl/gcl.py's docstring) -- what ESS
                                    # is meaningfully a fraction OF
        raw_ess = effective_sample_size(raw_weights)
        clipped_ess = effective_sample_size(clipped_weights)

        print(
            f"{iteration:>10}  {raw_ess:>9.2f}  {raw_ess / n_pool:>9.2%}  "
            f"{clipped_ess:>13.2f}  {clipped_ess / n_pool:>10.2%}  {n_pool:>7}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=str, required=True, help="e.g. easy_easy")
    parser.add_argument("--iterations", type=int, nargs="+", required=True)
    parser.add_argument("--seed-mode", type=str, default="easy")
    parser.add_argument("--n-background", type=int, default=20)
    parser.add_argument(
        "--clip-percentile", type=float, default=50.0,
        help="should match experiments/phase0b_gcl_training.py's "
             "IMPORTANCE_WEIGHT_CLIP_PERCENTILE to reflect the real "
             "training config -- pass a different value to explore "
             "other percentiles on a saved checkpoint",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(
        args.cell, args.iterations, args.seed_mode, args.n_background,
        args.clip_percentile, args.seed,
    )