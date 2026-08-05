"""Demonstration generation for Phase 0b: build the 3x3 experiment grid
(agent trained on {easy, easy+medium, all three} x evaluated on
{easy-only, easy+medium, fully-mixed} seed distributions), using
sim/nanogoal_adapter.py's rollout mechanics.

Only successful episodes (info["is_success"] == True) are kept as
demonstrations -- a failed rollout reflects a policy failure, not a
considered reward trade-off, and including it would corrupt the
recovered reward.

Equity target: seed mode proportions apply to the final count of
SUCCESSFUL demonstrations, not just to sampling attempts. If success
rates differ across difficulty categories (expected, especially for
weaker/out-of-distribution agents), sampling more attempts from an
under-performing category keeps the final demonstration set at the
intended proportions, instead of letting it silently skew toward
whichever category the policy happens to succeed at more often.
"""
import numpy as np

from sim.nanogoal_adapter import create_env, load_test_seeds, load_policy, rollout

SEED_MODE_PROPORTIONS = {
    "easy": {"easy": 1.0},
    "easy_medium": {"easy": 0.5, "medium": 0.5},
    "mixed": {"easy": 1 / 3, "medium": 1 / 3, "hard": 1 / 3},
}


def generate_nanogoal_demonstrations(
    nanogoal_path: str,
    model_difficulty: str,  # "easy", "medium", or "hard" -- which trained agent
    seed_mode: str,         # "easy", "easy_medium", or "mixed" -- which environment distribution
    n_target_successes: int,
    rng: np.random.Generator,
    max_attempts_per_category: int = 2000,
) -> tuple[list[dict], dict]:
    """Returns (demonstrations, stats).

    demonstrations: list of successful rollout dicts (see
    sim.nanogoal_adapter.rollout), length up to n_target_successes --
    possibly less if a category's held-out seed pool is exhausted, or
    max_attempts_per_category is reached, before hitting its target share.
    stats always reports this explicitly; it is never silently swallowed.

    stats: {"attempts": {cat: n}, "successes": {cat: n},
            "success_rate": {cat: float}} -- per-difficulty-category
    breakdown. This is itself a first-class result of the 3x3 grid: e.g.
    whether the easy-only agent's success rate really collapses on hard
    seeds, and by how much, independent of the reward recovery question.
    """
    if seed_mode not in SEED_MODE_PROPORTIONS:
        raise ValueError(
            f"Unknown seed_mode '{seed_mode}', expected one of "
            f"{list(SEED_MODE_PROPORTIONS)}"
        )

    proportions = SEED_MODE_PROPORTIONS[seed_mode]
    test_seeds = load_test_seeds(nanogoal_path)
    env = create_env(nanogoal_path)
    policy = load_policy(nanogoal_path, model_difficulty, env)

    demonstrations = []
    stats = {"attempts": {}, "successes": {}, "success_rate": {}}

    for category, share in proportions.items():
        target = round(n_target_successes * share)
        pool = list(test_seeds[category])
        rng.shuffle(pool)

        successes = 0
        attempts = 0
        pool_idx = 0
        while successes < target and attempts < max_attempts_per_category:
            if pool_idx >= len(pool):
                break  # pool exhausted -- report below, never loop forever
            seed = pool[pool_idx]
            pool_idx += 1
            attempts += 1

            trajectory = rollout(env, policy, seed, deterministic=True)
            if trajectory["is_success"]:
                demonstrations.append(trajectory)
                successes += 1

        stats["attempts"][category] = attempts
        stats["successes"][category] = successes
        stats["success_rate"][category] = successes / attempts if attempts > 0 else 0.0

        if successes < target:
            print(
                f"WARNING: only {successes}/{target} successful '{category}' "
                f"episodes found for model={model_difficulty}, "
                f"seed_mode={seed_mode} (pool exhausted or too many failures)"
            )

    return demonstrations, stats