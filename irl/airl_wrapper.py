'''AIRL (Fu, Luo & Levine, 2018) via the `imitation` library -- unlike
GCL, the adversarial training loop itself is NOT hand-implemented here;
imitation's maintained AIRL trainer handles it (verified against
imitation==1.0.1). Our job is the adapter work: converting our own
demonstration format into imitation's TrajectoryWithRew, and wrapping
NanoGoal-RL's Dict observation space into a Box(15,) representation.

IMPORTANT: this deliberately does NOT use gymnasium's built-in
FlattenObservation wrapper. Verified: it flattens Dict spaces
ALPHABETICALLY by key (agent, delta_goal, lidar, mvt) -- a different
order than sim.nanogoal_adapter.flatten_observation's (agent, mvt,
delta_goal, lidar), which irl/gcl.py's reward network was built around.
Using the built-in wrapper here would silently make AIRL's state
representation inconsistent with GCL's, even though both would nominally
be "the same 15-dim vector" -- breaking any later Phase 3 comparison
between the two recovered rewards. FlattenedNanoGoalEnv below reuses our
own flatten_observation instead, guaranteeing identical ordering.

State-only reward (use_action=False): AIRL's strongest disentanglement
guarantee (dynamics-invariance) applies specifically to state-only reward
under deterministic dynamics -- consistent with GCL's own state-only
design and this project's feature philosophy. Same known simplification
re: NanoGoal-RL's actual action-dependent effort term applies here too
(see README Limitations) -- not just to GCL.
'''
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet, RewardNet
from imitation.data.types import TrajectoryWithRew

from sim.nanogoal_adapter import create_env, flatten_observation


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


def train_airl(
    nanogoal_path: str,
    demonstrations: list[dict],
    n_training_steps: int = 200_000,
    demo_batch_size: int = 64,
    seed: int = 0,
) -> tuple[RewardNet, PPO]:
    """Wires our environment/demonstrations into imitation's AIRL
    trainer. The policy (gen_algo) starts from scratch, same reasoning
    as GCL's train_gcl -- the pretrained easy/medium/hard checkpoints are
    only used to GENERATE demonstrations, never to warm-start the
    algorithm being validated.

    Returns (reward_net, gen_algo) -- reward_net.predict(...) gives the
    recovered state-only reward; gen_algo is the trained policy, usable
    directly with eval.recovery_continuous.evaluate_policy_under_true_reward.
    """
    raw_env = create_env(nanogoal_path)
    flat_env = FlattenedNanoGoalEnv(raw_env)
    venv = DummyVecEnv([lambda: flat_env])

    imitation_demonstrations = [_to_imitation_trajectory(t) for t in demonstrations]

    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        use_action=False,  # state-only, see module docstring
    )

    gen_algo = PPO("MlpPolicy", venv, seed=seed, verbose=0)

    airl_trainer = AIRL(
        demonstrations=imitation_demonstrations,
        demo_batch_size=demo_batch_size,
        venv=venv,
        gen_algo=gen_algo,
        reward_net=reward_net,
    )

    airl_trainer.train(total_timesteps=n_training_steps)

    return reward_net, gen_algo