"""Unit tests for irl/gcl.py -- pure shape/math checks with synthetic
data, no dependency on the actual NanoGoal-RL submodule or trained
models (unlike test_gcl_integration.py). Still needs torch/stable-
baselines3 installed, since irl/gcl.py imports them at module level via
sim/nanogoal_adapter.py -- skips cleanly if they're missing.
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

import torch

from irl.gcl import RewardNetwork, compute_importance_weights, reward_loss


def test_reward_network_output_is_unbatched_not_a_column_vector():
    """Regression test for the exact bug found during review: a missing
    .squeeze(-1) silently produced shape (batch, 1) instead of (batch,),
    which broadcasts WRONG (no error, just a wrong number) against
    (batch,) importance weights -- (batch,1) * (batch,) gives
    (batch,batch), not an elementwise product."""
    net = RewardNetwork()
    obs = torch.randn(5, 15)
    out = net(obs)
    assert out.shape == (5,)


def test_compute_importance_weights_sums_to_one():
    net = RewardNetwork()

    class FakeInnerPolicy:
        def evaluate_actions(self, obs_dict, actions):
            T = actions.shape[0]
            return None, torch.full((T,), -0.5), None

    class FakePolicy:
        policy = FakeInnerPolicy()

    trajectories = [
        {
            "observations": np.random.randn(T + 1, 15).astype(np.float32),
            "actions": np.random.randn(T, 2).astype(np.float32),
        }
        for T in [10, 15, 8]
    ]

    weights = compute_importance_weights(net, trajectories, FakePolicy())
    assert weights.shape == (3,)
    assert weights.min() >= 0.0
    assert np.isclose(weights.sum(), 1.0)


def test_reward_loss_produces_a_real_gradient():
    net = RewardNetwork()
    demos = [
        {"observations": np.random.randn(12, 15).astype(np.float32)}
        for _ in range(5)
    ]
    background = [
        {"observations": np.random.randn(9, 15).astype(np.float32)}
        for _ in range(3)
    ]
    weights = np.array([0.2, 0.5, 0.3])

    loss = reward_loss(net, demos, background, weights)
    assert loss.requires_grad

    loss.backward()
    first_layer = net.network[0]
    assert isinstance(first_layer, torch.nn.Linear)


    grad = first_layer.weight.grad
    assert grad is not None
    assert torch.any(grad != 0)