"""Integration test for irl/airl_wrapper.py's train_airl -- against the
real NanoGoal-RL submodule/models, not mocked. Same pattern as
test_gcl_integration.py: tiny parameters, confirms the loop runs end to
end and produces a well-formed output, NOT that AIRL has converged.
Skips cleanly if the submodule/models/heavy deps aren't available.

Note: AIRL uses imitation's own training loop, so unlike test_gcl.py
there are no unit tests for the reward network or importance weights
themselves -- those belong to imitation's own test suite. What's worth
testing here is that our adapter wiring (FlattenedNanoGoalEnv,
SeedModeEnv, _to_imitation_trajectory, allow_variable_horizon) actually
assembles into a working pipeline against the real environment.
"""
from pathlib import Path

import numpy as np
import pytest

NANOGOAL_PATH = str(Path(__file__).resolve().parents[1] / "external" / "NanoGoal-RL")

pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")
pytest.importorskip("imitation")

if not (Path(NANOGOAL_PATH) / "models" / "ppo_nanogoal_easy.zip").exists():
    pytest.skip(
        "NanoGoal-RL submodule/models not available -- run "
        "`git submodule update --init` and confirm models/ exists",
        allow_module_level=True,
    )

from data.simulate_nanogoal import generate_nanogoal_demonstrations
from irl.airl_wrapper import train_airl


def test_train_airl_runs_and_returns_correct_types():
    rng = np.random.default_rng(0)
    demonstrations, _stats = generate_nanogoal_demonstrations(
        NANOGOAL_PATH, model_difficulty="easy", seed_mode="easy",
        n_target_successes=5, rng=rng,
    )

    # 4096 is the minimum budget that satisfies imitation's own assertion
    # (total_timesteps >= gen_algo's n_steps = 2048), verified directly --
    # anything smaller raises "No updates (need at least 2048 timesteps)"
    reward_net, gen_algo = train_airl(
        NANOGOAL_PATH, demonstrations, seed_mode="easy",
        n_training_steps=4096, demo_batch_size=8,
    )

    # reward_net must be a real nn.Module with callable forward()
    import torch
    from imitation.rewards.reward_nets import RewardNet
    assert isinstance(reward_net, RewardNet)
    dummy_obs = torch.zeros(1, 15)
    dummy_act = torch.zeros(1, 2)
    dummy_next = torch.zeros(1, 15)
    reward_out = reward_net(dummy_obs, dummy_act, dummy_next, torch.zeros(1, dtype=torch.bool))
    assert reward_out.shape == (1,), (
        f"reward_net output shape {reward_out.shape} -- expected (1,)"
    )

    # gen_algo must have trained for exactly the requested number of steps
    assert gen_algo.num_timesteps == 4096, (
        f"gen_algo.num_timesteps={gen_algo.num_timesteps}, expected 4096"
    )


def test_seed_mode_is_respected_in_airl_env():
    """SeedModeEnv is what keeps AIRL's internal rollout collection
    within the cell's difficulty pool -- verify it's actually wired in
    by checking that reset() on the AIRL training env only draws seeds
    present in the easy test set, not arbitrary integers."""
    from sim.nanogoal_adapter import load_test_seeds, create_env, SeedModeEnv
    from irl.airl_wrapper import FlattenedNanoGoalEnv

    test_seeds = load_test_seeds(NANOGOAL_PATH)
    easy_seed_set = set(test_seeds["easy"])

    raw_env = create_env(NANOGOAL_PATH)
    flat_env = FlattenedNanoGoalEnv(raw_env)
    seeded_env = SeedModeEnv(
        flat_env, test_seeds, seed_mode="easy", rng=np.random.default_rng(42)
    )

    drawn_seeds = set()
    for _ in range(20):
        seeded_env.reset()
        # env.py logs self._ep and the seed used; instead read the internal
        # state directly -- _last_seed is set by reset() for traceability
        if hasattr(seeded_env.env.env, '_ep'):
            # can't easily read the seed post-hoc without patching env.py,
            # so we verify the SeedModeEnv wrapper property directly:
            pass

    # What we can verify without patching env.py: that SeedModeEnv.reset()
    # calls self.env.reset(seed=drawn_seed) for a seed from the easy pool.
    # Patch SeedModeEnv.reset to capture seeds instead.
    captured = []
    original_reset = seeded_env.env.reset

    def capturing_reset(seed=None, options=None):
        captured.append(seed)
        return original_reset(seed=seed, options=options)

    seeded_env.env.reset = capturing_reset
    for _ in range(10):
        seeded_env.reset()

    assert all(s in easy_seed_set for s in captured), (
        f"SeedModeEnv drew seeds outside the easy pool: "
        f"{[s for s in captured if s not in easy_seed_set]}"
    )