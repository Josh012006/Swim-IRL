'''Thin wrapper exposing NanoGoal-RL's env and trained policies as-is,
for Phase 0b's GCL/AIRL validation. No reward override -- the reward
recovered here is compared against NanoGoal-RL's own, already-known,
ground-truth reward exactly as defined in env.py.
'''
import numpy as np
import sys
import os
import importlib
import json
from stable_baselines3 import PPO
from pathlib import Path


def create_env(nanogoal_path: str):
    # Handles importing NanoGoal-RL's env.py (a standalone module living
    # in the submodule, not an installed package) -- e.g. sys.path.insert
    # to nanogoal_path, then `import env`, then `env.NanoEnv()`. All the
    # import-path mess stays contained here so nothing else in Swim-IRL
    # needs to know NanoGoal-RL isn't pip-installable.
    # returns: a NanoEnv instance

    sys.path.insert(0, os.path.abspath(nanogoal_path))

    env = importlib.import_module("env") # using importlib instead of just 'import env' so the linter can recognize it
    return env.NanoEnv()


def load_test_seeds(nanogoal_path: str) -> dict[str, list[int]]:
    """Reproduces eval.py's train/test split exactly -- same seeds.json,
    same RNG seeds (99999 for the training-split sampling, 77777 for
    building the held-out test set) -- so demonstrations here are
    guaranteed to come from seeds the loaded policy never saw during
    training. Skip eval.py's own 24680 shuffle-order RNG -- we do our own
    sampling on top in data/simulate_nanogoal.py, so the seeds' internal
    order doesn't matter here.

    Returns {"easy": [...], "medium": [...], "hard": [...]}.
    """
    sys.path.insert(0, os.path.abspath(nanogoal_path))

    env = importlib.import_module("env")

    # Using the same seeds as the NanoGoal project for perfect reproductibility
    _loader_rng = np.random.default_rng(99999)
    _test_rng = np.random.default_rng(77777)

    json_path = Path(nanogoal_path) / "seeds.json"
    with open(json_path) as f:
        _all_seeds = json.load(f)

    def _sample_category(seeds_list, pct=0.40):
        arr = np.array(seeds_list)
        k   = max(1, int(len(arr) * pct))
        idx = _loader_rng.choice(len(arr), size=k, replace=False)
        return set(arr[idx].tolist())

    train_easy   = _sample_category(_all_seeds["easy"], pct=0.40)
    train_medium = _sample_category(_all_seeds["medium"], pct=0.60)
    train_hard   = _sample_category(_all_seeds["hard"], pct=0.60)

    def _build_test_set(all_category, train_set, n=500):
        candidates = np.array([s for s in all_category if s not in train_set])
        k          = min(n, len(candidates))
        idx        = _test_rng.choice(len(candidates), size=k, replace=False)
        return candidates[idx].tolist()

    test_easy_seeds   = _build_test_set(_all_seeds["easy"],   train_easy)
    test_medium_seeds = _build_test_set(_all_seeds["medium"], train_medium)
    test_hard_seeds   = _build_test_set(_all_seeds["hard"],   train_hard)

    test_sets = {
        "easy":   test_easy_seeds,
        "medium": test_medium_seeds,
        "hard":   test_hard_seeds
    }

    return test_sets


def load_policy(nanogoal_path: str, model_difficulty: str, env):
    # model_difficulty: "easy", "medium", or "hard"
    # Loads models/ppo_nanogoal_<model_difficulty>.zip via PPO.load(...).
    # Match eval.py's determinism settings: torch.set_num_threads(1)
    # before loading, device="cpu".
    # returns: the loaded SB3 PPO model
    import torch
    torch.set_num_threads(1) # matching eval.py's determinism settings

    model_path = Path(nanogoal_path) / "models" / f"ppo_nanogoal_{model_difficulty}"
    model = PPO.load(model_path, env=env, device="cpu")

    return model


def flatten_observation(obs: dict) -> np.ndarray:
    # obs: the raw Dict observation from env.py
    #   (keys: "agent", "mvt", "delta_goal", "lidar")
    # returns: shape (15,) -- fixed concatenation order:
    #   agent(2) + mvt(3) + delta_goal(2) + lidar(8)
    result = np.concatenate([
        obs["agent"],
        obs["mvt"],
        obs["delta_goal"],
        obs["lidar"]
    ])
    return result


def rollout(
    env,
    policy,
    seed: int,
    deterministic: bool = True,
) -> dict:
    """Runs one full episode via env.reset(seed=seed), then
    policy.predict(obs, deterministic=deterministic) + env.step(action)
    until terminated or truncated -- T is the ACTUAL number of steps this
    specific episode took (bounded by env.py's own timelimit, usually far
    shorter), NOT a fixed constant across calls. Different seeds/episodes
    naturally return different T; callers must never assume otherwise.

    Returns:
        {
            "observations": np.ndarray, shape (T+1, 15) -- flattened via
                flatten_observation, one row per timestep including the
                final observation
            "actions": np.ndarray, shape (T, 2)
            "rewards": np.ndarray, shape (T,) -- NanoGoal-RL's own reward,
                kept for sanity-checking/diagnostics only, not consumed
                by GCL/AIRL themselves
            "is_success": bool -- from info["is_success"] at episode end
            "seed": int -- for traceability back to which world this was
        }
    """
    result = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "is_success": False,
        "seed": seed
    }

    obs, info = env.reset(seed=seed)
    result["observations"].append(flatten_observation(obs))

    terminated = False
    truncated = False

    # Run the episode
    while not (terminated or truncated):
        action, _ = policy.predict(obs, deterministic=deterministic)
        result["actions"].append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        result["observations"].append(flatten_observation(obs))
        result["rewards"].append(reward)


    result["is_success"] = info["is_success"]
    result["observations"] = np.array(result["observations"])
    result["actions"] = np.array(result["actions"])
    result["rewards"] = np.array(result["rewards"])

    return result