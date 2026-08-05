"""Integration test for the NanoGoal-RL demonstration pipeline
(sim/nanogoal_adapter.py + data/simulate_nanogoal.py), against the real
submodule and real trained models -- unlike the Phase 0a suite, this
genuinely depends on external resources that don't live inside this
repo's own fixtures: the NanoGoal-RL submodule, its seeds.json, and real
trained PPO checkpoints.

Skips automatically (rather than failing) if the submodule/models aren't
available -- e.g. before `git submodule update --init` has been run, or
in a CI environment that doesn't install Phase 0b's heavy dependencies
(torch, stable-baselines3, pygame) on purpose (see README Installation).
"""
import json
from pathlib import Path

import numpy as np
import pytest

NANOGOAL_PATH = str(Path(__file__).resolve().parents[1] / "external" / "NanoGoal-RL")

pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")

if not (Path(NANOGOAL_PATH) / "models" / "ppo_nanogoal_easy.zip").exists():
    pytest.skip(
        "NanoGoal-RL submodule/models not available -- run "
        "`git submodule update --init` and confirm models/ exists",
        allow_module_level=True,
    )

from data.simulate_nanogoal import generate_nanogoal_demonstrations
from sim.nanogoal_adapter import load_test_seeds, _sample_category

# env.py: timelimit is min(3 + 2*initial_distance, 40), in units of
# __timestep=0.05 -- so the max number of env.step() calls per episode
# is 40 / 0.05 = 800, NOT 40. Verified against real rollouts (T up to
# ~370 observed) before pinning this bound.
MAX_STEPS_PER_EPISODE = 800


def test_generate_demonstrations_easy_model_easy_seeds_runs_and_looks_sane():
    rng = np.random.default_rng(0)
    demonstrations, stats = generate_nanogoal_demonstrations(
        NANOGOAL_PATH,
        model_difficulty="easy",
        seed_mode="easy",
        n_target_successes=10,
        rng=rng,
    )

    assert len(demonstrations) > 0

    # loose sanity bound on success rate, not a pinned exact value --
    # the specific figure depends on which seeds get drawn, but the easy
    # model evaluated on easy (in-distribution) seeds should clearly
    # succeed far more often than not
    assert stats["success_rate"]["easy"] > 0.2, (
        f"success rate {stats['success_rate']['easy']:.2f} looks too low "
        f"for the easy model on easy seeds -- check the adapter/model "
        f"path before assuming this is just bad luck"
    )

    for demo in demonstrations:
        assert demo["is_success"] is True
        T = demo["actions"].shape[0]
        assert demo["observations"].shape == (T + 1, 15)
        assert demo["actions"].shape == (T, 2)
        assert demo["rewards"].shape == (T,)
        assert 1 <= T <= MAX_STEPS_PER_EPISODE


def test_demonstration_seeds_are_disjoint_from_training_pool():
    """The whole point of load_test_seeds() -- confirm none of the seeds
    it returns were ever in the easy model's training pool. Regression
    test: guards against a future change to the RNG seeds/pct values
    silently reintroducing train/test contamination."""
    with open(Path(NANOGOAL_PATH) / "seeds.json") as f:
        all_seeds = json.load(f)

    loader_rng = np.random.default_rng(99999)
    train_easy = _sample_category(all_seeds["easy"], loader_rng, pct=0.40)

    test_seeds = load_test_seeds(NANOGOAL_PATH)
    overlap = set(test_seeds["easy"]) & train_easy
    assert len(overlap) == 0, f"{len(overlap)} test seeds overlap with the training pool"