'''Thin wrapper exposing NanoGoal-RL's env and trained policies as-is,
for Phase 0b's GCL/AIRL validation. No reward override -- the reward
recovered here is compared against NanoGoal-RL's own, already-known,
ground-truth reward exactly as defined in env.py.
'''
import json
import os
import sys
import importlib
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO


def create_env(nanogoal_path: str):
    # append, NOT insert(0, ...): NanoGoal-RL's flat eval.py collides by
    # name with this project's own eval/ package. insert(0, ...) gives
    # NanoGoal-RL's directory PRIORITY over our own project root, so any
    # subsequent `eval.something` import risks resolving to NanoGoal-RL's
    # eval.py instead -- and that file runs its own argparse at import
    # time, crashing on OUR argv. Confirmed: this specifically broke
    # SubprocVecEnv workers (which use "spawn", inheriting the polluted
    # sys.path) as soon as anything imported eval.recovery_continuous
    # after create_env() had already run once. append() makes our own
    # project's modules resolve first, matching normal expectations,
    # with NanoGoal-RL's directory only consulted as a fallback.
    sys.path.append(os.path.abspath(nanogoal_path))
    env = importlib.import_module("env")  # using importlib instead of just
                                           # 'import env' so the linter can
                                           # recognize it
    return env.NanoEnv()


SEED_MODE_PROPORTIONS = {
    "easy": {"easy": 1.0},
    "easy_medium": {"easy": 0.5, "medium": 0.5},
}
# "mixed" (1/3 easy + 1/3 medium + 1/3 hard) was dropped along with the
# hard model itself -- see README Limitations for why (hard training did
# not converge to an optimal policy, so it was excluded from the grid
# entirely, not just its own row: any seed_mode that included hard seeds
# no longer has a meaningful role once hard is out of scope).


def sample_seed_from_mode(
    test_seeds: dict[str, list[int]],
    seed_mode: str,
    rng: np.random.Generator,
) -> int:
    """Draws ONE seed respecting seed_mode's category proportions (see
    SEED_MODE_PROPORTIONS). Shared by data.simulate_nanogoal (demonstration
    generation), irl.gcl (background rollout sampling) and
    irl.airl_wrapper/SeedModeEnv below (generator rollout sampling) -- one
    definition, so all three stay consistent with each other by
    construction rather than by three separately-maintained copies.
    """
    proportions = SEED_MODE_PROPORTIONS[seed_mode]
    categories = list(proportions.keys())
    weights = [proportions[c] for c in categories]
    category = rng.choice(categories, p=weights)
    return int(rng.choice(test_seeds[category]))


class SeedModeEnv(gym.Wrapper):
    """Wraps an env so every reset() draws its OWN seed from a seed_mode's
    difficulty-respecting pool, ignoring whatever seed (if any) the caller
    passes in.

    Why this exists: earlier versions of irl.gcl.train_gcl's background
    rollouts used a fully random seed (`rng.integers(0, 1_000_000)`),
    completely ignoring which seed_mode the cell under test was supposed
    to represent -- every cell ended up training against the hardest
    possible mix regardless of its nominal seed_mode, exactly the
    "environments changing too much episode to episode" problem
    NanoGoal-RL's own README describes solving with its curriculum. We
    don't always have fine-grained control over exactly when/how a
    training loop calls reset() -- SB3's PPO.learn() and imitation's AIRL
    trainer both call it internally, not through code we write per-call --
    so fixing it at the env level, once, makes it automatic and impossible
    to bypass by accident, rather than something every caller has to
    remember to do correctly.
    """
    def __init__(
        self,
        env,
        test_seeds: dict[str, list[int]],
        seed_mode: str,
        rng: np.random.Generator,
    ):
        super().__init__(env)
        self.test_seeds = test_seeds
        self.seed_mode = seed_mode
        self.rng = rng

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        drawn_seed = sample_seed_from_mode(self.test_seeds, self.seed_mode, self.rng)
        return self.env.reset(seed=drawn_seed, options=options)


def _sample_category(seeds_list, rng: np.random.Generator, pct: float = 0.40):
    """Reimplementation of eval.py's _sample_category -- not imported
    directly, since eval.py has import-time side effects (argparse,
    env/model loading) that make it unsafe to import as a library. See
    module docstring in load_test_seeds for the pct=0.40-for-all-three
    caveat."""
    arr = np.array(seeds_list)
    k = max(1, int(len(arr) * pct))
    idx = rng.choice(len(arr), size=k, replace=False)
    return set(arr[idx].tolist())


def _build_test_set(all_category, train_set, rng: np.random.Generator, n: int = 500):
    """Reimplementation of eval.py's _build_test_set -- see _sample_category."""
    candidates = np.array([s for s in all_category if s not in train_set])
    k = min(n, len(candidates))
    idx = rng.choice(len(candidates), size=k, replace=False)
    return candidates[idx].tolist()


def load_test_seeds(nanogoal_path: str) -> dict[str, list[int]]:
    """Reproduces eval.py's train/test split exactly -- same seeds.json,
    same RNG seeds (99999 for the training-split sampling, 77777 for
    building the held-out test set), same call order (easy, medium,
    hard) for each -- so demonstrations here are guaranteed to come from
    seeds the loaded policy never saw during training. Skip eval.py's own
    24680 shuffle-order RNG -- we do our own sampling on top in
    data/simulate_nanogoal.py, so the seeds' internal order doesn't
    matter here.

    Still computes the hard split too, even though Phase 0b's grid no
    longer uses it -- keeps this function a faithful, complete
    reproduction of eval.py's actual split regardless of which categories
    the rest of the project currently draws from, and preserves the exact
    RNG call order (easy, medium, hard) that the real seed values depend on.

    NOTE: eval.py's own _sample_category calls
    (train_easy/medium/hard = _sample_category(_all_seeds[...])) never
    pass an explicit pct, so all three default to pct=0.40 in eval.py's
    own code -- even though the README describes an asymmetric 40/60/60
    training split. This reproduces eval.py's actual behavior (0.40 for
    all three) for exact consistency with it; worth checking on the
    NanoGoal-RL side whether that's intentional.

    Returns {"easy": [...], "medium": [...], "hard": [...]}.
    """
    seeds_path = Path(nanogoal_path) / "seeds.json"
    with open(seeds_path) as f:
        all_seeds = json.load(f)

    loader_rng = np.random.default_rng(99999)
    train_easy = _sample_category(all_seeds["easy"], loader_rng)
    train_medium = _sample_category(all_seeds["medium"], loader_rng)
    train_hard = _sample_category(all_seeds["hard"], loader_rng)

    test_rng = np.random.default_rng(77777)
    test_easy_seeds = _build_test_set(all_seeds["easy"], train_easy, test_rng)
    test_medium_seeds = _build_test_set(all_seeds["medium"], train_medium, test_rng)
    test_hard_seeds = _build_test_set(all_seeds["hard"], train_hard, test_rng)

    return {
        "easy": test_easy_seeds,
        "medium": test_medium_seeds,
        "hard": test_hard_seeds,
    }


def load_policy(nanogoal_path: str, model_difficulty: str, env):
    # model_difficulty: "easy" or "medium" (Phase 0b's grid; "hard" is a
    # valid model_difficulty for this function itself -- it still exists
    # and can be loaded -- but the grid no longer includes it, see README
    # Limitations)
    torch.set_num_threads(1)  # matching eval.py's determinism settings

    model_path = Path(nanogoal_path) / "models" / f"ppo_nanogoal_{model_difficulty}"
    model = PPO.load(model_path, env=env, device="cpu")

    return model


def flatten_observation(obs: dict) -> np.ndarray:
    # fixed concatenation order: agent(2) + mvt(3) + delta_goal(2) + lidar(8)
    return np.concatenate([
        obs["agent"],
        obs["mvt"],
        obs["delta_goal"],
        obs["lidar"],
    ])


def rollout(
    env,
    policy,
    seed: int,
    deterministic: bool = True,
) -> dict:
    """Runs one full episode via env.reset(seed=seed), then
    policy.predict(obs, deterministic=deterministic) + env.step(action)
    until terminated or truncated -- T is the ACTUAL number of steps this
    specific episode took, NOT a fixed constant across calls.

    Returns:
        {
            "observations": np.ndarray, shape (T+1, 15)
            "actions": np.ndarray, shape (T, 2)
            "rewards": np.ndarray, shape (T,) -- NanoGoal-RL's own reward,
                diagnostics only, not consumed by GCL/AIRL
            "is_success": bool
            "seed": int
        }
    """
    observations = []
    actions = []
    rewards = []

    obs, info = env.reset(seed=seed)
    observations.append(obs if not isinstance(obs, dict) else flatten_observation(obs))

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        actions.append(action)
        rewards.append(reward)
        # env may already return a flattened Box(15,) observation (e.g.
        # irl.airl_wrapper.FlattenedNanoGoalEnv) instead of the raw Dict
        # -- flatten_observation only applies to the raw Dict form, so
        # skip it when the env has already done that itself. Verified:
        # calling it unconditionally on an already-flat array crashes
        # with an IndexError from obs["agent"], not a silent wrong value.
        observations.append(obs if not isinstance(obs, dict) else flatten_observation(obs))

    return {
        "observations": np.array(observations),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "is_success": info["is_success"],
        "seed": seed,
    }