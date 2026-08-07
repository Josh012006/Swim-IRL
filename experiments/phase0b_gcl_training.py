"""Phase 0b GCL experiment: the 3x3 grid (agent trained on
{easy, easy+medium, all three} x demonstrations drawn from
{easy-only, easy+medium, fully-mixed} seed distributions), training GCL
on each of the 9 cells and evaluating the recovered reward/policy via
eval.recovery_continuous.sampled_recovery_gap.

Usage (local smoke test, minutes not hours):
    python -m experiments.phase0b_gcl_training --seed 0 --quick

Usage (real run, meant for a remote/background machine -- see
.github/ci/train_phase0b.sh for the unattended version of this):
    python -m experiments.phase0b_gcl_training --seed 0 --cell easy_easy --n-envs 4

BUDGET is calibrated against NanoGoal-RL's OWN reported timesteps to
reach a working policy at each difficulty (12M / 150M / 400M for
easy / medium / hard -- see that project's README), scaled x1.5 as a
starting hypothesis for GCL's harder learning problem (reward AND policy,
not policy alone). Per seed_mode, not per model_difficulty -- what the
generator policy inside train_gcl has to learn to solve is set by
seed_mode's difficulty mix, not by which expert produced the
demonstrations. THIS IS EXPENSIVE: even with parallelism, expect this to
take from tens of minutes (easy) to multiple hours (mixed) PER CELL on a
personal machine -- always pass --checkpoint-dir (done automatically
below) and run in the background, never interactively for anything but
--quick.

--cell trains exactly ONE (model_difficulty, seed_mode) pair -- the
remote pipeline launches 6 separate invocations of this (hard is skipped
automatically, its model doesn't exist yet), rather than one process
looping over all 9/6 cells, so a crash in one cell doesn't take the
others down with it and each can checkpoint/resume independently.
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
    plot_recovery_grid_heatmap,
)

NANOGOAL_PATH = "external/NanoGoal-RL"
MODEL_DIFFICULTIES = ["easy", "medium", "hard"]
SEED_MODES = ["easy", "easy_medium", "mixed"]

# Real, per-seed_mode budgets -- see module docstring for the calibration.
BUDGET = {
    "easy":        {"total_timesteps": 18_000_000,  "n_target_successes": 150},
    "easy_medium": {"total_timesteps": 225_000_000, "n_target_successes": 150},
    "mixed":       {"total_timesteps": 600_000_000, "n_target_successes": 150},
}
QUICK_PARAMS = dict(total_timesteps=4096, n_target_successes=5)
# 4096, not smaller: imitation's AIRL.train() asserts total_timesteps must
# be >= gen_algo's own n_steps (2048 by default) -- see
# irl/airl_wrapper.py's docstring. This constant is shared by both
# training scripts (phase0b_gcl_training.py and phase0b_airl_training.py),
# so it has to satisfy AIRL's harder constraint even though GCL itself
# would tolerate a smaller value.

N_ITERATIONS = 100          # GCL alternation steps, same across all seed_modes
N_BACKGROUND_PER_ITER = 20
N_EVAL_SEEDS = 30
CHECKPOINT_EVERY = 10        # GCL iterations, not timesteps -- see irl/gcl.py


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

    checkpoint_dir = f"experiments/results/checkpoints/gcl_{model_difficulty}_{seed_mode}"
    reward_net, recovered_policy, history = train_gcl(
        NANOGOAL_PATH, demonstrations, seed_mode,
        total_timesteps=params["total_timesteps"],
        n_iterations=2 if quick else N_ITERATIONS,
        n_background_trajectories_per_iteration=3 if quick else N_BACKGROUND_PER_ITER,
        n_envs=n_envs,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=1 if quick else CHECKPOINT_EVERY,
    )

    expert_policy = load_policy(NANOGOAL_PATH, model_difficulty, create_env(NANOGOAL_PATH))
    eval_env = create_env(NANOGOAL_PATH)
    eval_seeds = [int(s) for s in rng.integers(0, 1_000_000, size=N_EVAL_SEEDS)]
    # GCL's recovered_policy also uses MultiInputPolicy (Dict obs), same
    # as the expert -- one shared env is correct here, unlike AIRL (see
    # eval/recovery_continuous.py's docstring)
    gap = sampled_recovery_gap(eval_env, expert_policy, eval_env, recovered_policy, eval_seeds)

    return {"demo_stats": demo_stats, "history": history, "reward_net": reward_net, "gap": gap}


def main(seed: int, quick: bool, n_envs: int, single_cell: str | None) -> None:
    os.makedirs("experiments/results", exist_ok=True)

    if single_cell is not None:
        model_difficulty, seed_mode = single_cell.split("_", 1)
        cells = [(model_difficulty, seed_mode)]
    else:
        cells = [(m, s) for m in MODEL_DIFFICULTIES for s in SEED_MODES]

    return_gap_grid = np.full((3, 3), np.nan)
    success_gap_grid = np.full((3, 3), np.nan)

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
        plot_training_diagnostics(
            result["history"],
            output_path=f"experiments/results/phase0b_gcl_training_{cell_name}.png",
        )

    if single_cell is None:
        plot_recovery_grid_heatmap(
            return_gap_grid, MODEL_DIFFICULTIES, SEED_MODES,
            metric_name="GCL: return gap (expert - recovered, under true reward)",
            output_path="experiments/results/phase0b_gcl_return_gap_heatmap.png",
        )
        plot_recovery_grid_heatmap(
            success_gap_grid, MODEL_DIFFICULTIES, SEED_MODES,
            metric_name="GCL: success rate gap (expert - recovered)",
            output_path="experiments/results/phase0b_gcl_success_gap_heatmap.png",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument(
        "--cell", type=str, default=None,
        help="e.g. easy_easy, medium_mixed -- run exactly one cell instead of the full grid",
    )
    args = parser.parse_args()
    main(args.seed, args.quick, args.n_envs, args.cell)