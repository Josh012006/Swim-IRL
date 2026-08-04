"""Phase 0a recovery experiment: for >=3 ground-truth reward modes on the
gridworld, generate synthetic demonstrations, run linear MaxEnt IRL, and
report + plot the recovered theta against ground truth (Expected Value
Difference as the primary metric -- see eval/recovery.py for why raw
parameter distance isn't used).

Usage:
    python -m experiments.phase0a_recovery --seed 0
"""
import argparse
import os

import numpy as np

from mdp import state_to_coord
from sim.gridworld import build_gridworld_mdp
from data.simulate import generate_demonstrations
from irl.maxent_linear import fit
from eval.recovery import expected_value_diff
from experiments.plotting import plot_recovery_comparison

# --- fixed environment for this experiment ---
WIDTH, HEIGHT = 6, 6
GOAL_POS = (5, 5)
OBSTACLES = [(1, 4), (3, 2), (4, 4)]
HORIZON = 15
N_DEMONSTRATIONS = 3000
N_ITERATIONS = 300
LEARNING_RATE = 0.02  # was 0.1 -- too large, caused non-monotonic
                       # log-likelihood and bad theta_hat (see
                       # eval/convergence_diagnostic.py)

FEATURE_NAMES = ["dist_goal", "dist_obstacle"]  # row/col dropped -- see
                                                 # sim/features_gridworld.py

# --- the >=3 ground-truth reward modes we chose ---
REWARD_MODES = [
    {
        # only distance-to-goal matters: pure goal-seeking
        "name": "A_distance_to_goal",
        "theta_true": np.array([-1.0, 0.0]),
    },
    {
        # only distance-to-obstacle matters, no goal at all: tests whether
        # IRL correctly assigns ~0 weight to dist_goal when it genuinely
        # doesn't drive behavior, not just whether it can detect a signal
        "name": "B_obstacle_avoidance",
        "theta_true": np.array([0.0, 1.0]),
    },
    {
        # both features contribute: the harder, more realistic case
        "name": "C_mixed",
        "theta_true": np.array([-1.0, 0.4]),
    },
]


def build_start_distribution(mdp, width, obstacles):
    """Uniform over every non-obstacle cell. A single fixed start state
    made row/col (when they still existed) collinear with dist_goal along
    the one demonstrated path; a broad start distribution reduces (but,
    for a fixed corner goal, doesn't fully eliminate -- see README
    Limitations) that collinearity, and is also a stronger test: behavior
    must be near-optimal from anywhere, not just one starting point."""
    valid_states = [
        s for s in range(mdp.n_states) if state_to_coord(s, width) not in obstacles
    ]
    start_distribution = np.zeros(mdp.n_states)
    start_distribution[valid_states] = 1.0 / len(valid_states)
    return start_distribution


def main(seed: int) -> None:
    rng = np.random.default_rng(seed)

    mdp = build_gridworld_mdp(WIDTH, HEIGHT, OBSTACLES, GOAL_POS)
    start_distribution = build_start_distribution(mdp, WIDTH, OBSTACLES)

    results = []
    for mode in REWARD_MODES:
        theta_true = mode["theta_true"]

        demonstrations = generate_demonstrations(
            mdp, theta_true, start_distribution, HORIZON,
            n_demonstrations=N_DEMONSTRATIONS, rng=rng,
        )
        theta_hat = fit(
            mdp, demonstrations, start_distribution, HORIZON,
            learning_rate=LEARNING_RATE, n_iterations=N_ITERATIONS,
        )
        evd = expected_value_diff(
            mdp, theta_true, theta_hat, start_distribution, HORIZON,
        )

        print(
            f"{mode['name']}: theta_true={theta_true}  "
            f"theta_hat={np.round(theta_hat, 3)}  EVD={evd:.4f}"
        )

        results.append({
            "name": mode["name"],
            "theta_true": theta_true,
            "theta_hat": theta_hat,
            "evd": evd,
            "feature_names": FEATURE_NAMES,
        })

    os.makedirs("experiments/results", exist_ok=True)
    plot_recovery_comparison(
        results, output_path="experiments/results/phase0a_recovery.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.seed)