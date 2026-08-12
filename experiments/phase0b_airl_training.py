"""Phase 0b AIRL experiment: same 2x2 grid as
experiments/phase0b_gcl_training.py, same BUDGET calibration (see that
file's docstring), same --cell/--quick/--n-envs interface -- kept as a
SEPARATE script rather than folded into the GCL one, since GCL and AIRL
are independently-validated methods per the Roadmap (Phase 0b's
completion criterion is both, evaluated separately, not a combined
average).

"hard" was dropped from both axes (model_difficulty and seed_mode) --
see phase0b_gcl_training.py's module docstring and README Limitations.

Usage (local smoke test):
    python -m experiments.phase0b_airl_training --seed 0 --quick

Usage (real run, one cell, meant for a remote/background machine):
    python -m experiments.phase0b_airl_training --seed 0 --cell easy_easy --n-envs 4
"""
import argparse
import os

import numpy as np

from data.simulate_nanogoal import generate_nanogoal_demonstrations
from sim.nanogoal_adapter import create_env, load_policy
from irl.airl_wrapper import train_airl, FlattenedNanoGoalEnv
from eval.recovery_continuous import sampled_recovery_gap
from experiments.plotting_phase0b import plot_recovery_grid_heatmap
from experiments.phase0b_gcl_training import BUDGET, QUICK_PARAMS, MODEL_DIFFICULTIES, SEED_MODES

NANOGOAL_PATH = "external/NanoGoal-RL"
N_EVAL_SEEDS = 30
CHECKPOINT_EVERY_TIMESTEPS = 1_000_000  # must be >= gen_algo's own n_steps
                                          # (2048 default) -- see
                                          # irl/airl_wrapper.py docstring


def run_cell(
    model_difficulty: str,
    seed_mode: str,
    quick: bool,
    n_envs: int,
    seed: int,
) -> dict:
    params = QUICK_PARAMS if quick else BUDGET[seed_mode]
    rng = np.random.default_rng(seed)

    demonstrations, demo_stats = generate_nanogoal_demonstrations(
        NANOGOAL_PATH, model_difficulty, seed_mode,
        n_target_successes=params["n_target_successes"], rng=rng,
    )

    cell_name = f"{model_difficulty}_{seed_mode}"
    checkpoint_dir = f"experiments/results/checkpoints/airl_{cell_name}"
    reward_net, recovered_policy = train_airl(
        NANOGOAL_PATH, demonstrations, seed_mode,
        n_training_steps=params["total_timesteps"],
        n_envs=n_envs,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=2048 if quick else CHECKPOINT_EVERY_TIMESTEPS,
        tb_log_dir="logs/tensorboard/phase0b_airl",
        tb_log_name=cell_name,
    )

    expert_policy = load_policy(NANOGOAL_PATH, model_difficulty, create_env(NANOGOAL_PATH))
    expert_eval_env = create_env(NANOGOAL_PATH)
    # recovered_policy (AIRL's gen_algo) is an MlpPolicy trained on the
    # flattened Box(15,) representation, NOT the raw Dict env the expert
    # uses -- must evaluate it through the same FlattenedNanoGoalEnv
    # wrapper it was trained with, or SB3 raises an assertion error
    # immediately (verified) rather than silently misbehaving
    recovered_eval_env = FlattenedNanoGoalEnv(create_env(NANOGOAL_PATH))
    eval_seeds = [int(s) for s in rng.integers(0, 1_000_000, size=N_EVAL_SEEDS)]
    gap = sampled_recovery_gap(
        expert_eval_env, expert_policy, recovered_eval_env, recovered_policy, eval_seeds
    )

    return {"demo_stats": demo_stats, "reward_net": reward_net, "gap": gap}


def main(seed: int, quick: bool, n_envs: int, single_cell: str | None) -> None:
    os.makedirs("experiments/results", exist_ok=True)

    if single_cell is not None:
        model_difficulty, seed_mode = single_cell.split("_", 1)
        cells = [(model_difficulty, seed_mode)]
    else:
        cells = [(m, s) for m in MODEL_DIFFICULTIES for s in SEED_MODES]

    return_gap_grid = np.full((len(MODEL_DIFFICULTIES), len(SEED_MODES)), np.nan)
    success_gap_grid = np.full((len(MODEL_DIFFICULTIES), len(SEED_MODES)), np.nan)

    for model_difficulty, seed_mode in cells:
        model_path = f"{NANOGOAL_PATH}/models/ppo_nanogoal_{model_difficulty}.zip"
        if not os.path.exists(model_path):
            print(f"WARNING: {model_path} not found -- skipping '{model_difficulty}'")
            continue

        cell_name = f"{model_difficulty}_{seed_mode}"
        print(f"=== cell {cell_name} ===")
        result = run_cell(model_difficulty, seed_mode, quick, n_envs, seed)

        i = MODEL_DIFFICULTIES.index(model_difficulty)
        j = SEED_MODES.index(seed_mode)
        return_gap_grid[i, j] = result["gap"]["return_gap"]
        success_gap_grid[i, j] = result["gap"]["success_rate_gap"]

        print(
            f"{cell_name}: demo success rates={result['demo_stats']['success_rate']}  "
            f"return_gap={result['gap']['return_gap']:.3f}  "
            f"success_rate_gap={result['gap']['success_rate_gap']:.3f}"
        )

    if single_cell is None:
        plot_recovery_grid_heatmap(
            return_gap_grid, MODEL_DIFFICULTIES, SEED_MODES,
            metric_name="AIRL: return gap (expert - recovered, under true reward)",
            output_path="experiments/results/phase0b_airl_return_gap_heatmap.png",
        )
        plot_recovery_grid_heatmap(
            success_gap_grid, MODEL_DIFFICULTIES, SEED_MODES,
            metric_name="AIRL: success rate gap (expert - recovered)",
            output_path="experiments/results/phase0b_airl_success_gap_heatmap.png",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument(
        "--cell", type=str, default=None,
        help="e.g. easy_easy, medium_easy_medium -- run exactly one cell instead of the full grid",
    )
    args = parser.parse_args()
    main(args.seed, args.quick, args.n_envs, args.cell)