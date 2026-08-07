'''AIRL (Fu, Luo & Levine, 2018) via the `imitation` library -- unlike
GCL, the adversarial training loop itself is NOT hand-implemented here;
imitation's maintained AIRL trainer handles it (verified against
imitation==1.0.1). Our job is the adapter work: converting our own
demonstration format into imitation's TrajectoryWithRew, wrapping
NanoGoal-RL's Dict observation space into a Box(15,) representation, and
making sure the generator's rollouts respect seed_mode the same way
GCL's do.

IMPORTANT: this deliberately does NOT use gymnasium's built-in
FlattenObservation wrapper. Verified: it flattens Dict spaces
ALPHABETICALLY by key (agent, delta_goal, lidar, mvt) -- a different
order than sim.nanogoal_adapter.flatten_observation's (agent, mvt,
delta_goal, lidar), which GCL's reward network was built around. Using
the built-in wrapper here would silently make AIRL's state representation
inconsistent with GCL's, even though both would nominally be "the same
15-dim vector" -- breaking any later Phase 3 comparison. FlattenedNanoGoalEnv
below reuses our own flatten_observation instead.

SEED_MODE: unlike GCL (where WE own the rollout-collection loop and
explicitly draw seeds respecting seed_mode), AIRL's generator rollouts are
collected internally by imitation/SB3, on a schedule we don't control
call-by-call. sim.nanogoal_adapter.SeedModeEnv fixes this the same way as
for GCL, at the env level: every reset() -- no matter which internal code
path triggers it -- draws its own seed from seed_mode's pool, ignoring
whatever seed (if any) the caller passed in.

State-only reward (use_action=False): AIRL's strongest disentanglement
guarantee (dynamics-invariance) applies specifically to state-only reward
under deterministic dynamics -- consistent with GCL's own state-only
design. Same known simplification re: NanoGoal-RL's actual action-dependent
effort term applies here too (see README Limitations) -- not just to GCL.

PARALLELISM: unlike GCL, no manual weight-sync is needed here -- imitation
owns the entire training loop (reward net + generator policy updates)
within ONE process, so a SubprocVecEnv passed as `venv` works with
ordinary SB3 semantics, no extra plumbing required on our side.
'''
import json
import os

import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet, RewardNet
from imitation.data.types import TrajectoryWithRew

from sim.nanogoal_adapter import create_env, flatten_observation, load_test_seeds, SeedModeEnv


class FlattenedNanoGoalEnv(gym.ObservationWrapper):
    """Wraps NanoGoal-RL's Dict observation into the same flat Box(15,)
    representation used throughout Phase 0b -- see module docstring for
    why this can't be gymnasium's built-in FlattenObservation.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

    def observation(self, observation):
        return flatten_observation(observation)


def _to_imitation_trajectory(traj: dict) -> TrajectoryWithRew:
    """Converts one of our own rollout()-format dicts into imitation's
    own trajectory type. terminal=True always: demonstrations passed to
    train_airl are pre-filtered to is_success=True by
    data.simulate_nanogoal, and NanoGoal-RL's success is a genuine
    terminated=True (goal reached), not a time truncation."""
    return TrajectoryWithRew(
        obs=traj["observations"],
        acts=traj["actions"],
        infos=None,
        terminal=True,
        rews=traj["rewards"].astype(np.float32),
    )


def _make_env(nanogoal_path, test_seeds, seed_mode, worker_idx, seed):
    def _init():
        base_env = create_env(nanogoal_path)
        flat_env = FlattenedNanoGoalEnv(base_env)
        seeded = SeedModeEnv(
            flat_env, test_seeds, seed_mode,
            rng=np.random.default_rng(seed + 1000 * worker_idx),
        )
        return Monitor(seeded)
    return _init


def _save_checkpoint(checkpoint_dir: str, reward_net, gen_algo, timesteps_done):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(reward_net.state_dict(), os.path.join(checkpoint_dir, "reward_net.pt"))
    gen_algo.save(os.path.join(checkpoint_dir, "gen_algo"))
    with open(os.path.join(checkpoint_dir, "state.json"), "w") as f:
        json.dump({"timesteps_done": timesteps_done}, f)
    print(f"[checkpoint] saved at {timesteps_done} timesteps -> {checkpoint_dir}")


def train_airl(
    nanogoal_path: str,
    demonstrations: list[dict],
    seed_mode: str,
    n_training_steps: int,
    demo_batch_size: int = 64,
    n_envs: int = 1,
    seed: int = 0,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 1_000_000,
) -> tuple[RewardNet, PPO]:
    """Wires our environment/demonstrations into imitation's AIRL
    trainer. The policy (gen_algo) starts from scratch, same reasoning as
    GCL's train_gcl -- the pretrained easy/medium/hard checkpoints are
    only used to GENERATE demonstrations, never to warm-start the
    algorithm being validated.

    n_training_steps is the OVERALL budget for this seed_mode (see
    experiments/phase0b_gcl_training.py's BUDGET table -- the same table
    applies here, since both methods are being validated against the same
    calibration). Run in checkpoint_every-sized chunks internally
    (verified: repeated AIRL.train() calls on the same trainer continue
    training rather than restarting) so unattended multi-hour runs can be
    resumed after an interruption. checkpoint_every must be at least
    gen_algo's own n_steps (2048 by default here, unset explicitly) --
    imitation's AIRL.train() asserts on this and fails loudly if not (
    verified: total_timesteps=128 raises "No updates (need at least 2048
    timesteps)"), never silently under-runs the requested chunk the way
    SB3's own policy.learn() does in irl/gcl.py.

    Returns (reward_net, gen_algo) -- reward_net.predict(...) gives the
    recovered state-only reward; gen_algo is the trained policy, usable
    directly with eval.recovery_continuous.evaluate_policy_under_true_reward.
    """
    test_seeds = load_test_seeds(nanogoal_path)

    if n_envs > 1:
        venv = SubprocVecEnv(
            [_make_env(nanogoal_path, test_seeds, seed_mode, i, seed) for i in range(n_envs)]
        )
    else:
        venv = DummyVecEnv([_make_env(nanogoal_path, test_seeds, seed_mode, 0, seed)])

    imitation_demonstrations = [_to_imitation_trajectory(t) for t in demonstrations]

    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        use_action=False,  # state-only, see module docstring
    )

    timesteps_done = 0
    resuming = bool(checkpoint_dir) and os.path.exists(
        os.path.join(checkpoint_dir, "state.json")  # type: ignore[arg-type]
    )
    if resuming:
        assert checkpoint_dir is not None  # narrowing: bool(checkpoint_dir) guarantees str
        reward_net.load_state_dict(torch.load(os.path.join(checkpoint_dir, "reward_net.pt")))
        gen_algo = PPO.load(os.path.join(checkpoint_dir, "gen_algo"), env=venv, device="cpu")
        with open(os.path.join(checkpoint_dir, "state.json")) as f:
            timesteps_done = json.load(f)["timesteps_done"]
        print(f"[checkpoint] resuming from {timesteps_done}/{n_training_steps} timesteps")
    else:
        gen_algo = PPO("MlpPolicy", venv, seed=seed, verbose=0, device="cpu")

    airl_trainer = AIRL(
        demonstrations=imitation_demonstrations,
        demo_batch_size=demo_batch_size,
        venv=venv,
        gen_algo=gen_algo,
        reward_net=reward_net,
        # NanoGoal-RL's episodes are inherently variable-length (up to 800
        # steps, see env.py's timelimit). imitation's default check rejects
        # this because episode length can itself leak reward-relevant
        # information -- concretely here: our demonstrations are ALL
        # successes (short-to-medium length), while the generator's early
        # rollouts are often failures that run to the truncation cap, so
        # the discriminator could learn "short = expert" as a shortcut
        # instead of the real state-based reward structure. Accepting this
        # documented risk rather than resolving it -- see README
        # Limitations. GCL has no equivalent built-in check surfacing the
        # same risk, but is not necessarily immune to it either.
        allow_variable_horizon=True,
    )

    while timesteps_done < n_training_steps:
        chunk = min(checkpoint_every, n_training_steps - timesteps_done)
        airl_trainer.train(total_timesteps=chunk)
        timesteps_done += chunk
        print(f"[train_airl] {timesteps_done}/{n_training_steps} timesteps")

        if checkpoint_dir:
            assert checkpoint_dir is not None  # narrowing: bool(checkpoint_dir) guarantees str
            _save_checkpoint(checkpoint_dir, reward_net, gen_algo, timesteps_done)

    venv.close()
    return reward_net, gen_algo