"""Phase 0b GCL experiment: the 3x3 grid (agent trained on
{easy, easy+medium, all three} x demonstrations drawn from
{easy-only, easy+medium, fully-mixed} seed distributions), training GCL
on each of the 9 cells and evaluating the recovered reward/policy via
eval.recovery_continuous.sampled_recovery_gap.

Usage:
    python -m experiments.phase0b_gcl_training --seed 0 --quick

--quick uses tiny parameters (a handful of GCL iterations, short PPO
updates) -- enough to confirm the full 9-cell pipeline runs end to end,
NOT enough for a meaningful result. Drop --quick for a real run, but be
aware of the cost: a single cell's smoke-test configuration already took
~10s in verification; REAL_PARAMS scales iterations 50x and PPO steps 8x
per iteration on top of that -- budget accordingly (likely hours, not
minutes, per cell) before launching the full grid unattended.

The "hard" row is skipped automatically with a warning if
models/ppo_nanogoal_hard.zip doesn't exist yet (training in progress as
of this writing -- see README).
"""
import argparse
import os

import numpy as np

from data.simulate_nanogoal import generate_nanogoal_demonstrations
from sim.nanogoal_adapter import create_env, load_policy
from irl.gcl import train_gcl
from eval.recovery_continuous import sampled_recovery_gap
from experiments.plotting_phase0b import (
    plot_training_diagnostics,
    collect_reward_comparison_data,
    plot_reward_comparison,
    plot_recovery_grid_heatmap,
)

NANOGOAL_PATH = "external/NanoGoal-RL"
MODEL_DIFFICULTIES = ["easy", "medium", "hard"]
SEED_MODES = ["easy", "easy_medium", "mixed"]

QUICK_PARAMS = dict(
    n_target_successes=5,
    n_iterations=2,
    n_background_trajectories_per_iteration=3,
    policy_update_steps_per_iteration=256,
)
REAL_PARAMS = dict(
    n_target_successes=150,
    n_iterations=100,
    n_background_trajectories_per_iteration=20,
    policy_update_steps_per_iteration=2048,
)
N_EVAL_SEEDS = 30  # held-out seeds used for sampled_recovery_gap, per cell


def run_cell(model_difficulty: str, seed_mode: str, params: dict, rng: np.random.Generator) -> dict:
    demonstrations, demo_stats = generate_nanogoal_demonstrations(
        NANOGOAL_PATH, model_difficulty, seed_mode,
        n_target_successes=params["n_target_successes"], rng=rng,
    )

    reward_net, recovered_policy, history = train_gcl(
        NANOGOAL_PATH, demonstrations,
        n_iterations=params["n_iterations"],
        n_background_trajectories_per_iteration=params["n_background_trajectories_per_iteration"],
        policy_update_steps_per_iteration=params["policy_update_steps_per_iteration"],
    )

    expert_policy = load_policy(NANOGOAL_PATH, model_difficulty, create_env(NANOGOAL_PATH))
    eval_env = create_env(NANOGOAL_PATH)
    eval_seeds = [int(s) for s in rng.integers(0, 1_000_000, size=N_EVAL_SEEDS)]
    gap = sampled_recovery_gap(eval_env, expert_policy, recovered_policy, eval_seeds)

    return {
        "demo_stats": demo_stats,
        "history": history,
        "reward_net": reward_net,
        "gap": gap,
    }


def main(seed: int, quick: bool) -> None:
    rng = np.random.default_rng(seed)
    params = QUICK_PARAMS if quick else REAL_PARAMS

    os.makedirs("experiments/results", exist_ok=True)

    return_gap_grid = np.full((3, 3), np.nan)
    success_gap_grid = np.full((3, 3), np.nan)

    for i, model_difficulty in enumerate(MODEL_DIFFICULTIES):
        model_path = f"{NANOGOAL_PATH}/models/ppo_nanogoal_{model_difficulty}.zip"
        if not os.path.exists(model_path):
            print(f"WARNING: {model_path} not found -- skipping row '{model_difficulty}'")
            continue

        for j, seed_mode in enumerate(SEED_MODES):
            cell_name = f"{model_difficulty}_{seed_mode}"
            print(f"=== cell {cell_name} ===")

            result = run_cell(model_difficulty, seed_mode, params, rng)

            return_gap_grid[i, j] = result["gap"]["return_gap"]
            success_gap_grid[i, j] = result["gap"]["success_rate_gap"]

            print(
                f"{cell_name}: demo success rates={result['demo_stats']['success_rate']}  "
                f"return_gap={result['gap']['return_gap']:.3f}  "
                f"success_rate_gap={result['gap']['success_rate_gap']:.3f}"
            )

            plot_training_diagnostics(
                result["history"],
                output_path=f"experiments/results/phase0b_training_{cell_name}.png",
            )

    plot_recovery_grid_heatmap(
        return_gap_grid, MODEL_DIFFICULTIES, SEED_MODES,
        metric_name="Return gap (expert - recovered, under true reward)",
        output_path="experiments/results/phase0b_return_gap_heatmap.png",
    )
    plot_recovery_grid_heatmap(
        success_gap_grid, MODEL_DIFFICULTIES, SEED_MODES,
        metric_name="Success rate gap (expert - recovered)",
        output_path="experiments/results/phase0b_success_gap_heatmap.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(args.seed, args.quick)