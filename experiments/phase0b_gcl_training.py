"""Phase 0b GCL experiment: the 2x2 grid (agent trained on
{easy, easy+medium} x demonstrations drawn from {easy-only, easy+medium}
seed distributions), training GCL on each of the 4 cells and evaluating
the recovered reward/policy via eval.recovery_continuous.sampled_recovery_gap.

"hard" was dropped from both axes (model_difficulty and seed_mode) after
hard training completed but did not converge to an optimal policy -- see
README Limitations. The grid went from 3x3 to 2x2, not just from 3x3 to
"3x3 with a skipped row": seed_mode's "mixed" tier (which included hard
seeds) was removed entirely, not renamed, since a mix that no longer
contains its hardest category isn't the same experimental condition.

Usage (local smoke test, minutes not hours):
    python -m experiments.phase0b_gcl_training --seed 0 --quick

Usage (real run, meant for a remote/background machine -- see
.github/ci/train_phase0b_gcl.sh for the unattended version of this):
    python -m experiments.phase0b_gcl_training --seed 0 --cell easy_easy --n-envs 4

BUDGET is originally calibrated against NanoGoal-RL's OWN reported
timesteps to reach a working policy at each difficulty (12M / 150M for
easy / medium -- see that project's README), scaled x1.5 as a starting
hypothesis for GCL's harder learning problem (reward AND policy, not
policy alone). Per seed_mode, not per model_difficulty -- what the
generator policy inside train_gcl has to learn to solve is set by
seed_mode's difficulty mix, not by which expert produced the
demonstrations. "easy" was later revised to 180M/1000 iterations (10x
the original 18M/100) after a real run showed a late-training collapse
in the recovered reward alongside a still-climbing success rate -- see
BUDGET's own comment for the full diagnosis. THIS IS EXPENSIVE: even
with parallelism, expect this to take from hours (easy, at the revised
budget) to a few hours (easy_medium, still at the original budget) PER
CELL on a personal machine -- always pass --checkpoint-dir (done
automatically below) and run in the background, never interactively for
anything but --quick.

--cell trains exactly ONE (model_difficulty, seed_mode) pair -- the
remote pipeline launches 4 separate invocations of this, rather than one
process looping over all 4 cells, so a crash in one cell doesn't take the
others down with it and each can checkpoint/resume independently.
"""
import argparse
import os
import shutil

import numpy as np
import torch

from data.simulate_nanogoal import generate_nanogoal_demonstrations
from sim.nanogoal_adapter import create_env, load_policy
from irl.gcl import train_gcl
from eval.recovery_continuous import sampled_recovery_gap
from experiments.plotting_phase0b import (
    plot_training_diagnostics,
    plot_recovery_grid_heatmap,
)

NANOGOAL_PATH = "external/NanoGoal-RL"
MODEL_DIFFICULTIES = ["easy", "medium"]
SEED_MODES = ["easy", "easy_medium"]

# Real, per-seed_mode budgets -- see module docstring for the original
# calibration. n_iterations now lives HERE, per seed_mode, rather than as
# a single global constant -- "easy" was scaled up from 100 to 1000
# iterations (180M timesteps, 10x, keeping the SAME 180K-timesteps-per-
# iteration ratio as before -- not just 10x more iterations on the old,
# smaller total_timesteps, which would instead make reward_net update
# 10x MORE often per unit of policy training, the opposite of what's
# needed here) after a real run on easy/easy showed a late-training
# collapse: rollout/ep_rew_mean (the RECOVERED reward, from reward_net)
# fell sharply in the final ~15% of iterations while rollout/success_rate
# (genuine task performance) kept climbing over that same window, and
# train/value_loss spiked in lockstep -- PPO's value function tracking a
# reward_net that had not yet stabilized. Only "easy" is scaled here;
# "easy_medium" keeps its original budget, not touched by this specific
# diagnosis until it's been checked separately.
BUDGET = {
    "easy":        {"total_timesteps": 180_000_000, "n_iterations": 1000, "n_target_successes": 150},
    "easy_medium": {"total_timesteps": 225_000_000,  "n_iterations": 100,  "n_target_successes": 150},
}
QUICK_PARAMS = dict(total_timesteps=4096, n_iterations=4, n_target_successes=5)
# 4096, not smaller: imitation's AIRL.train() asserts total_timesteps must
# be >= gen_algo's own n_steps (2048 by default) -- see
# irl/airl_wrapper.py's docstring. This constant is shared by both
# training scripts (phase0b_gcl_training.py and phase0b_airl_training.py),
# so it has to satisfy AIRL's harder constraint even though GCL itself
# would tolerate a smaller value.

# Lowered from 1e-3 (irl.gcl.train_gcl's own default) after the same
# easy/easy run's diagnosis above: 1e-3 let reward_net's parameters move
# far enough per single gradient step that PPO's value function -- which
# tracks the reward AS IT WAS INSTEAD of as it currently is -- couldn't
# keep up, especially once policy performance started actually improving
# late in training (the exact window value_loss spiked in). A smaller
# reward-side learning rate slows how fast the target itself moves per
# iteration, independent of how many iterations or how much policy
# training time is given -- a different lever from total_timesteps/
# n_iterations, addressing a different failure mode (see README
# Limitations for the full reasoning, both hypotheses considered).
REWARD_LEARNING_RATE = 1e-4

# Log-ratio clipping for compute_importance_weights. Went through three
# versions -- see irl/gcl.py's compute_importance_weights docstring for
# the full history (per-trajectory -> per-decision pooling -> back to
# per-trajectory + clipping). This is v3: PURE per-trajectory GCL
# weighting (staying faithful to Ziebart/Finn's own derivation, and
# keeping the background term at the same trajectory-summed scale as
# demo_term -- v2's per-decision pooling silently broke that scale
# match, letting reward_net shrink reward_loss by uniformly inflating
# its output with no discriminative learning, confirmed on a real run
# via query_reward_model.py's demo_term/background_mean tracking),
# with clipping applied to the N per-trajectory log-ratios as a pure
# variance-reduction step on top.
#
# 50.0 here, NOT v2's 95.0 -- verified directly: with only ~20
# background trajectories (not v2's thousands of pooled transitions), a
# 95th-percentile clip barely caps anything (ESS improved from 1.00/20
# to just 1.43/20). 50.0 (the median) raised it to 10.01/20 -- 50% of
# the pool -- on the same real batch. Re-tune this again if
# N_BACKGROUND_PER_ITER changes materially from ~20.
IMPORTANCE_WEIGHT_CLIP_PERCENTILE = 50.0

N_BACKGROUND_PER_ITER = 20
N_EVAL_SEEDS = 30
CHECKPOINT_EVERY = 10        # GCL iterations, not timesteps -- see irl/gcl.py


def run_cell(
    model_difficulty: str,
    seed_mode: str,
    quick: bool,
    n_envs: int,
    seed: int,
    fresh: bool = False,
) -> dict:
    params = QUICK_PARAMS if quick else BUDGET[seed_mode]
    rng = np.random.default_rng(seed)

    demonstrations, demo_stats = generate_nanogoal_demonstrations(
        NANOGOAL_PATH, model_difficulty, seed_mode,
        n_target_successes=params["n_target_successes"], rng=rng,
    )

    cell_name = f"{model_difficulty}_{seed_mode}"
    checkpoint_dir = f"experiments/results/checkpoints/gcl_{cell_name}"

    # train_gcl auto-resumes from checkpoint_dir whenever a numbered
    # checkpoint is found there (see irl/gcl.py) -- deliberately, so an
    # interrupted multi-day run can continue after a crash. But
    # checkpoint_dir also SURVIVES every workflow checkout by design
    # (backed up/restored around actions/checkout's git clean -- see
    # .github/workflows/train_phase0b.yml), so it's ALSO still there the
    # next time this cell is triggered with genuinely different
    # settings (a new reward_learning_rate, a new
    # importance_weight_clip_percentile, etc.) -- silently resuming a
    # run that isn't the one being started, including its
    # policy.num_timesteps, which is what TensorBoard's step axis
    # actually is (PPO.load restores it, and policy.learn(...,
    # reset_num_timesteps=False) continues from there). fresh=True
    # clears checkpoint_dir first, an explicit "no, really start over"
    # signal instead of relying on implicit file-presence detection.
    if fresh and os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print(f"--fresh: cleared {checkpoint_dir} before starting")

    reward_net, recovered_policy, history = train_gcl(
        NANOGOAL_PATH, demonstrations, seed_mode,
        total_timesteps=params["total_timesteps"],
        n_iterations=params["n_iterations"],
        reward_learning_rate=REWARD_LEARNING_RATE,
        importance_weight_clip_percentile=IMPORTANCE_WEIGHT_CLIP_PERCENTILE,
        n_background_trajectories_per_iteration=3 if quick else N_BACKGROUND_PER_ITER,
        n_envs=n_envs,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=1 if quick else CHECKPOINT_EVERY,
        tb_log_dir="logs/tensorboard/phase0b_gcl",
        tb_log_name=cell_name,
    )

    # Final artifacts, separate from experiments/results/checkpoints/'s
    # numbered per-iteration snapshots (those are for resumption/
    # diagnostics during a run -- see irl/gcl.py). models/ is the
    # single, discoverable place for "the trained result of this cell",
    # mirroring NanoGoal-RL's own top-level models/ convention. Load
    # reward_net back with irl.gcl.RewardNetwork() + load_state_dict(),
    # or query it directly via experiments/query_reward_model.py.
    os.makedirs("models", exist_ok=True)
    torch.save(reward_net.state_dict(), f"models/phase0b_gcl_{cell_name}_reward_net.pt")
    recovered_policy.save(f"models/phase0b_gcl_{cell_name}_policy")

    expert_policy = load_policy(NANOGOAL_PATH, model_difficulty, create_env(NANOGOAL_PATH))
    eval_env = create_env(NANOGOAL_PATH)
    eval_seeds = [int(s) for s in rng.integers(0, 1_000_000, size=N_EVAL_SEEDS)]
    # GCL's recovered_policy also uses MultiInputPolicy (Dict obs), same
    # as the expert -- one shared env is correct here, unlike AIRL (see
    # eval/recovery_continuous.py's docstring)
    gap = sampled_recovery_gap(eval_env, expert_policy, eval_env, recovered_policy, eval_seeds)

    return {"demo_stats": demo_stats, "history": history, "reward_net": reward_net, "gap": gap}


def main(seed: int, quick: bool, n_envs: int, single_cell: str | None, fresh: bool) -> None:
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
        result = run_cell(model_difficulty, seed_mode, quick, n_envs, seed, fresh=fresh)

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
        help="e.g. easy_easy, medium_easy_medium -- run exactly one cell instead of the full grid",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="clear any existing checkpoint for the cell(s) being run before "
             "starting, instead of auto-resuming from it -- use this whenever "
             "hyperparameters changed since the last run of this cell",
    )
    args = parser.parse_args()
    main(args.seed, args.quick, args.n_envs, args.cell, args.fresh)