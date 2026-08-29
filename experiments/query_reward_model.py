"""Load saved Phase 0b GCL reward networks + policies and actually query
them, instead of having trained reward_nets sitting on disk with no way
to inspect what they learned.

By default, scans EVERY checkpoint iteration found in
experiments/results/checkpoints/gcl_{cell}/ (see irl/gcl.py's numbered
checkpointing) and reports how the recovered reward's correlation with
NanoGoal-RL's own true reward evolves ACROSS training -- a single
checkpoint can't answer "is there real improvement over time", only a
comparison across iterations can. Pass --iterations to restrict to a
specific subset instead (faster, e.g. for a quick spot-check), or
--final to load the final trained result from models/ (only available
once a cell has fully finished training).

Rolls out n_episodes with each checkpoint's OWN saved policy (the SAME
episode seeds are reused across every iteration checked, for a fair
apples-to-apples comparison -- differences in the numbers reflect
training progress, not which episodes happened to be sampled), pairs
reward_net's prediction against the true reward at every step (reusing
experiments.plotting_phase0b.collect_reward_comparison_data), and plots
the correlation trend across iterations.

Usage (scan every available checkpoint):
    python -m experiments.query_reward_model --cell easy_easy

Usage (restrict to specific iterations, faster):
    python -m experiments.query_reward_model --cell easy_easy --iterations 10 50 90

Usage (final result, after a cell finishes):
    python -m experiments.query_reward_model --cell easy_easy --final
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO

from sim.nanogoal_adapter import create_env, rollout, load_test_seeds, sample_seed_from_mode
from data.simulate_nanogoal import generate_nanogoal_demonstrations
from irl.gcl import RewardNetwork
from experiments.plotting_phase0b import collect_reward_comparison_data, plot_reward_comparison


def discover_checkpoint_iterations(checkpoint_dir: str) -> list[int]:
    """Every iteration with a saved state_iter{N}.json, sorted ascending
    -- mirrors irl.gcl._find_latest_checkpoint_iteration's detection
    pattern, but returns ALL of them instead of just the max."""
    iterations = []
    for path in glob.glob(os.path.join(checkpoint_dir, "state_iter*.json")):
        match = re.search(r"state_iter(\d+)\.json$", path)
        if match:
            iterations.append(int(match.group(1)))
    return sorted(iterations)


def load_gcl_artifacts_at_iteration(
    checkpoint_dir: str, iteration: int, env
) -> tuple[RewardNetwork, PPO]:
    reward_net = RewardNetwork()
    reward_net.load_state_dict(
        torch.load(f"{checkpoint_dir}/reward_net_iter{iteration}.pt")
    )
    reward_net.eval()
    policy = PPO.load(f"{checkpoint_dir}/policy_iter{iteration}", env=env, device="cpu")
    return reward_net, policy


def load_gcl_final_artifacts(cell: str, env) -> tuple[RewardNetwork, PPO]:
    reward_net = RewardNetwork()
    reward_net.load_state_dict(torch.load(f"models/phase0b_gcl_{cell}_reward_net.pt"))
    reward_net.eval()
    policy = PPO.load(f"models/phase0b_gcl_{cell}_policy", env=env, device="cpu")
    return reward_net, policy


def evaluate(reward_net, policy, env, seed: int, n_episodes: int):
    rng = np.random.default_rng(seed)
    trajectories = [
        rollout(env, policy, seed=int(rng.integers(0, 1_000_000)), deterministic=True)
        for _ in range(n_episodes)
    ]
    predicted, true = collect_reward_comparison_data(reward_net, trajectories)
    correlation = float(np.corrcoef(predicted, true)[0, 1]) if len(predicted) > 1 else float("nan")
    return predicted, true, correlation


def compute_demo_and_background_terms(
    reward_net: RewardNetwork,
    demonstrations: list[dict],
    policy: PPO,
    env,
    test_seeds: dict,
    seed_mode: str,
    rng: np.random.Generator,
    n_background: int,
) -> tuple[float, float]:
    """Reports the SAME two magnitudes reward_loss actually adds/subtracts
    internally: demo_term (mean over demonstrations of the FULL TRAJECTORY
    SUM of reward_net(s_t)) and a background per-state mean (reward_net's
    average output over a fresh batch of this checkpoint's OWN stochastic
    background rollouts). Not a re-derivation of reward_loss's importance-
    weighted background_term itself (that depends on compute_importance_weights'
    per-decision weights, specific to the exact batch drawn during
    training) -- this is a simpler, DIRECT check for the scale mismatch
    it can be exploited by: demo_term sums over ~T timesteps per
    trajectory, while the pooled per-decision background weights sum to
    1 across the WHOLE POOL (thousands of transitions) -- so a uniform,
    non-discriminative rise in reward_net's output inflates demo_term by
    roughly (T x the rise) but the weighted background term by only
    roughly (1 x the rise). If demo_term grows sharply relative to
    background_mean_per_state across checkpoints, that's this mismatch
    being exploited, not real discriminative learning.
    """
    with torch.no_grad():
        demo_rewards = [
            reward_net(torch.from_numpy(demo["observations"]).float()).sum().item()
            for demo in demonstrations
        ]
        demo_term = float(np.mean(demo_rewards))

        background_trajectories = [
            rollout(
                env, policy,
                seed=sample_seed_from_mode(test_seeds, seed_mode, rng),
                deterministic=False,
            )
            for _ in range(n_background)
        ]
        pooled_background_states = np.concatenate([
            traj["observations"][:-1] for traj in background_trajectories
        ])
        background_state_rewards = reward_net(
            torch.from_numpy(pooled_background_states).float()
        )
        background_mean_per_state = float(background_state_rewards.mean())

    return demo_term, background_mean_per_state


def plot_correlation_trend(iterations: list[int], correlations: list[float], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(iterations, correlations, marker="o")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("checkpoint iteration")
    ax.set_ylabel("correlation(predicted reward, true reward)")
    ax.set_title("GCL recovered-reward quality over training")
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_demo_vs_background_trend(
    iterations: list[int],
    demo_terms: list[float],
    background_means: list[float],
    output_path: str,
) -> None:
    """demo_term (a trajectory SUM) and background_mean_per_state (a
    per-state average) aren't literally comparable in scale -- this
    isn't "are these two lines close together", it's "does demo_term
    keep climbing while background_mean stays flat", which is what a
    uniform, non-discriminative rise in reward_net's output looks like
    (see compute_demo_and_background_terms' docstring)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(iterations, demo_terms, marker="o", color="tab:blue")
    ax1.set_xlabel("checkpoint iteration")
    ax1.set_ylabel("demo_term (mean trajectory-sum reward on demonstrations)")
    ax1.set_title("Demo term")

    ax2.plot(iterations, background_means, marker="o", color="tab:orange")
    ax2.set_xlabel("checkpoint iteration")
    ax2.set_ylabel("mean reward_net(s) over background rollout states")
    ax2.set_title("Background per-state mean")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def main(
    cell: str,
    seed: int,
    n_episodes: int,
    iterations: list[int] | None,
    final: bool,
    n_demos: int,
    n_background: int,
) -> None:
    nanogoal_path = "external/NanoGoal-RL"
    env = create_env(nanogoal_path)
    checkpoint_dir = f"experiments/results/checkpoints/gcl_{cell}"
    model_difficulty, seed_mode = cell.split("_", 1)

    if final:
        reward_net, policy = load_gcl_final_artifacts(cell, env)
        predicted, true, correlation = evaluate(reward_net, policy, env, seed, n_episodes)
        print(
            f"GCL reward_net for cell '{cell}' (final models/): "
            f"{len(predicted)} state-reward pairs across {n_episodes} episodes"
        )
        print(f"  correlation with true reward: {correlation:.3f}")
        print(f"  predicted reward: mean={predicted.mean():.3f}  std={predicted.std():.3f}")
        print(f"  true reward:      mean={true.mean():.3f}  std={true.std():.3f}")
        output_path = f"experiments/results/phase0b_gcl_{cell}_reward_comparison.png"
        plot_reward_comparison(predicted, true, output_path=output_path)
        print(f"\nScatter plot saved to {output_path}")
        return

    if iterations is None:
        iterations = discover_checkpoint_iterations(checkpoint_dir)
        if not iterations:
            print(f"No checkpoints found in {checkpoint_dir}")
            return
        print(f"Found {len(iterations)} checkpoints: {iterations}")

    # Demonstrations are generated once, from the fixed expert policy --
    # unlike background rollouts, they never change across checkpoints,
    # so regenerating them once here (rather than per-iteration) is both
    # correct and cheaper. Same expert used throughout training (the
    # actual demos used during training aren't saved to disk anywhere,
    # so this regenerates an equivalent set from the same expert/seed_mode
    # -- n_demos should match what the real run used, e.g. 150 for a
    # full run's BUDGET or 5 for --quick, if an exact match matters).
    demo_rng = np.random.default_rng(seed)
    demonstrations, _ = generate_nanogoal_demonstrations(
        nanogoal_path, model_difficulty, seed_mode,
        n_target_successes=n_demos, rng=demo_rng,
    )
    test_seeds = load_test_seeds(nanogoal_path)
    background_rng = np.random.default_rng(seed + 999_999)  # separate stream,
                                                               # mirrors train_gcl's own

    header = (
        f"{'iteration':>10}  {'correlation':>11}  {'pred mean':>10}  "
        f"{'pred std':>9}  {'true mean':>10}  {'true std':>9}  "
        f"{'demo_term':>10}  {'bg_mean':>9}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for iteration in iterations:
        try:
            reward_net, policy = load_gcl_artifacts_at_iteration(checkpoint_dir, iteration, env)
        except FileNotFoundError:
            print(f"{iteration:>10}  -- checkpoint not found, skipping --")
            continue

        predicted, true, correlation = evaluate(reward_net, policy, env, seed, n_episodes)
        demo_term, background_mean = compute_demo_and_background_terms(
            reward_net, demonstrations, policy, env, test_seeds, seed_mode,
            background_rng, n_background,
        )
        results.append((iteration, correlation, demo_term, background_mean))
        print(
            f"{iteration:>10}  {correlation:>11.3f}  {predicted.mean():>10.3f}  "
            f"{predicted.std():>9.3f}  {true.mean():>10.3f}  {true.std():>9.3f}  "
            f"{demo_term:>10.3f}  {background_mean:>9.4f}"
        )

    if len(results) >= 2:
        iters, corrs, demo_terms, bg_means = zip(*results)

        trend_output_path = f"experiments/results/phase0b_gcl_{cell}_correlation_trend.png"
        plot_correlation_trend(list(iters), list(corrs), trend_output_path)
        print(f"\nCorrelation trend plot saved to {trend_output_path}")

        demo_bg_output_path = f"experiments/results/phase0b_gcl_{cell}_demo_vs_background_trend.png"
        plot_demo_vs_background_trend(
            list(iters), list(demo_terms), list(bg_means), demo_bg_output_path
        )
        print(f"Demo-vs-background trend plot saved to {demo_bg_output_path}")
    elif len(results) == 1:
        print("\nOnly one checkpoint found -- pass more --iterations or wait "
              "for more checkpoints to see a trend.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=str, required=True, help="e.g. easy_easy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument(
        "--iterations", type=int, nargs="+", default=None,
        help="specific checkpoint iterations to check -- default: every "
             "checkpoint found in experiments/results/checkpoints/gcl_{cell}/",
    )
    parser.add_argument(
        "--final", action="store_true",
        help="load the final trained result from models/ instead of "
             "scanning checkpoint iterations (only works after a cell "
             "has fully finished training)",
    )
    parser.add_argument(
        "--n-demos", type=int, default=150,
        help="regenerate this many demonstrations from the expert policy "
             "for the demo_term check -- match the real run's BUDGET "
             "n_target_successes for an exact comparison (150 for a full "
             "run, 5 for --quick)",
    )
    parser.add_argument(
        "--n-background", type=int, default=20,
        help="background rollouts per checkpoint for the background_mean "
             "check -- match N_BACKGROUND_PER_ITER (20) for consistency",
    )
    args = parser.parse_args()
    main(
        args.cell, args.seed, args.n_episodes, args.iterations, args.final,
        args.n_demos, args.n_background,
    )