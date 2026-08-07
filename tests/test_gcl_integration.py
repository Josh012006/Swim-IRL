"""Integration test for irl/gcl.py's train_gcl -- against the real
NanoGoal-RL submodule/models, not mocked. Tiny parameters (a handful of
iterations, a short PPO update per iteration) -- this only confirms the
loop runs end to end and produces a well-formed, finite history, NOT
that GCL has converged to anything useful; that's what
experiments/phase0b_gcl_training.py is for. Skips cleanly if the
submodule/models/heavy deps aren't available (same pattern as
test_nanogoal_integration.py).
"""
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
from irl.gcl import train_gcl


def test_train_gcl_runs_and_produces_finite_history():
    rng = np.random.default_rng(0)
    demonstrations, _stats = generate_nanogoal_demonstrations(
        NANOGOAL_PATH, model_difficulty="easy", seed_mode="easy",
        n_target_successes=5, rng=rng,
    )

    _reward_net, _policy, history = train_gcl(
        NANOGOAL_PATH, demonstrations, seed_mode="easy",
        total_timesteps=4096,
        n_iterations=2,
        n_background_trajectories_per_iteration=3,
    )

    assert len(history["loss"]) == 2
    assert len(history["success_rate"]) == 2
    assert np.all(np.isfinite(history["loss"])), (
        "non-finite reward_loss -- likely a diverging/unstable update, "
        "same failure family as the learning_rate bug found in Phase 0a"
    )
    assert np.all((history["success_rate"] >= 0) & (history["success_rate"] <= 1))