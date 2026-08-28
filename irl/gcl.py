'''Guided Cost Learning (Finn, Levine & Abbeel, 2016): sample-based
MaxEnt IRL for continuous state/action spaces, replacing exact tabular DP
with importance-weighted policy rollouts.

Alternates two steps every iteration: (1) update the reward network using
demonstrations AND importance-weighted samples from the current policy,
(2) update the policy (SB3 PPO) to better optimize the current reward
estimate. Step (2) is what makes this "guided" -- it keeps the sampling
distribution tracking the current reward, so the importance weights in
step (1) don't blow up the way they would against a fixed, unrelated
sampling policy.

Reward is state-only (r_theta(s), not r_theta(s,a)) -- consistent with
AIRL's state-only design (see irl/airl_wrapper.py) and this project's
existing feature philosophy. Known simplification: see README Limitations
for why this can't represent NanoGoal-RL's actual action-dependent
effort/spinning penalty.

The policy trained here starts from SCRATCH (random init) -- it is NOT
warm-started from the loaded easy/medium checkpoints. Those are only
used to GENERATE demonstrations (see data/simulate_nanogoal.py); using
them to also seed GCL's own learner would quietly hand it a policy that
already solves the task, defeating the point of testing whether GCL can
recover a competent policy from demonstrations alone.

PARALLELISM -- read before changing n_envs: SubprocVecEnv spawns each
sub-environment in a SEPARATE OS PROCESS. RewardWrappedEnv's reward_net
gets PICKLED into every worker at construction time, becoming an
INDEPENDENT COPY in each process from that point on -- updating reward_net
in the main process via optimizer.step() does NOT propagate to the
workers' copies automatically (verified directly: a worker kept computing
reward from iteration-0 weights indefinitely without this fix). Every
iteration, after optimizer.step(), we push the updated weights across the
process boundary explicitly via
`vec_env.env_method("set_reward_net_state", ...)`, which SB3's VecEnv
mechanism resolves through Monitor -> SeedModeEnv -> RewardWrappedEnv
(verified) -- currently raises a gymnasium deprecation warning
(`env.method_name` wrapper-attribute delegation is deprecated in favor of
`get_wrapper_attr`), harmless today but SB3's own internal implementation,
not something patchable from here -- worth re-checking on a future SB3
upgrade.
'''
import json
import os

import numpy as np
import scipy.special
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from sim.nanogoal_adapter import (
    flatten_observation,
    create_env,
    rollout,
    load_test_seeds,
    sample_seed_from_mode,
    SeedModeEnv,
)


class RewardNetwork(nn.Module):
    # small MLP: flattened observation (15,) -> scalar reward.
    def __init__(self, obs_dim: int = 15, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: shape (batch, 15)
        # returns: shape (batch,) -- squeeze the trailing singleton dim
        # from the last Linear(hidden_dim, 1) layer, or every batched
        # use downstream (importance weights, reward_loss) silently
        # broadcasts wrong via (batch,1) * (batch,) -> (batch,batch)
        return self.network(obs).squeeze(-1)


class RewardWrappedEnv(gym.Wrapper):
    """Same observation/action space as the real NanoGoal env (inherited
    from gym.Wrapper), but step() returns reward_net(obs) instead of the
    environment's own true reward. This is what lets SB3's ordinary
    PPO.learn() train the policy against the CURRENT reward estimate,
    with zero changes to SB3 itself.
    """
    def __init__(self, env, reward_net: RewardNetwork):
        super().__init__(env)
        self.reward_net = reward_net

    def step(self, action):
        obs, _true_reward, terminated, truncated, info = self.env.step(action)

        obs_tensor = torch.from_numpy(flatten_observation(obs)).float().unsqueeze(0)
        with torch.no_grad():
            reward = self.reward_net(obs_tensor).item()

        return obs, reward, terminated, truncated, info

    def set_reward_net_state(self, state_dict):
        """Called via vec_env.env_method() from the main process after
        every optimizer.step() -- see module docstring PARALLELISM note.
        Never called directly for n_envs=1 (no subprocess boundary to
        cross there), but harmless either way."""
        self.reward_net.load_state_dict(state_dict)


def _unflatten_observations_batch(flat_obs_batch: np.ndarray) -> dict:
    """Inverse of sim.nanogoal_adapter.flatten_observation, batched.
    Needed because rollout() only stores the flattened form, but SB3's
    MultiInputPolicy needs the original Dict structure to evaluate
    log-probabilities of actions. flat_obs_batch: shape (N, 15)."""
    return {
        "agent": torch.from_numpy(flat_obs_batch[:, 0:2]).float(),
        "mvt": torch.from_numpy(flat_obs_batch[:, 2:5]).float(),
        "delta_goal": torch.from_numpy(flat_obs_batch[:, 5:7]).float(),
        "lidar": torch.from_numpy(flat_obs_batch[:, 7:15]).float(),
    }


def compute_importance_weights(
    reward_net: RewardNetwork,
    background_trajectories: list[dict],
    policy: PPO,
    clip_percentile: float = 95.0,
) -> np.ndarray:
    """Per-decision (per-transition) self-normalized importance weights,
    POOLED across every timestep of every background trajectory, with
    log-ratio clipping to bound concentration:

        ell_{i,t}       = r_theta(s_t^(i)) - log pi_phi(a_t^(i)|s_t^(i))
        ell_{i,t}_clip  = min(ell_{i,t}, percentile(all ell, clip_percentile))
        w_{i,t}         = softmax_{(i,t)}( ell_{i,t}_clip )

    Replaces an earlier per-WHOLE-TRAJECTORY version (one weight per
    trajectory, log_ratio = sum_t r_theta(s_t) - sum_t log pi_phi(a_t|s_t))
    after that version was found, empirically, to degenerate almost
    completely on this environment: Effective Sample Size (1 /
    sum(w_i^2)) collapsed to ~1 -- meaning reward_loss's gradient was
    effectively estimated from a SINGLE background trajectory every
    iteration, not the 20+ actually sampled -- and stayed there even at
    100 background trajectories, ruling out "not enough samples" as the
    explanation (see README Limitations for the full diagnosis,
    including the diagnostic script that measured this directly).

    Moving to per-decision pooling (below) alone was a real but PARTIAL
    fix -- ESS improved from ~1 to ~3.5 out of thousands of pooled
    transitions on a real check, still heavily concentrated. Log-ratio
    clipping is added on top of it for exactly this reason: GCL's own
    "guided" mechanism (Finn et al. 2016 -- the policy.learn() call in
    train_gcl below, keeping the sampling distribution close to the
    current reward's induced optimum) is the paper's own answer to
    importance-weight degeneracy, but doesn't fully prevent it,
    especially mid-training before the policy has caught up to a
    still-shifting reward. Weight clipping/truncation (Ionides 2008,
    "Truncated Importance Sampling") is the standard complement across
    the broader importance-sampling literature (particle filtering,
    off-policy RL, likelihood-free inference) for exactly this residual
    problem -- directly analogous to PPO's own clip_range, which bounds
    its policy-side importance ratio the same way, just applied here to
    GCL's reward-side weights instead. Trades a controlled, bounded
    amount of bias for a large reduction in variance -- the clip
    threshold is a PERCENTILE of the current batch's own log-ratios,
    recomputed fresh every iteration, not a fixed absolute value that
    would need re-tuning as reward_net and the policy evolve over a run.

    The per-trajectory version's log_ratio summed log pi(a_t|s_t) over
    an ENTIRE episode (300-800 steps here) before ever reaching the
    softmax -- variance in a sum of that many per-step terms compounds
    with trajectory length, a well-documented failure mode of
    trajectory-level importance sampling (per-decision importance
    sampling, Precup 2000, exists in the broader off-policy RL
    literature specifically to avoid it). Pooling every (trajectory,
    timestep) pair into ONE softmax, instead of one softmax per
    trajectory's fully-summed ratio, means an outlier at a single
    timestep no longer drags every OTHER timestep of that same
    trajectory down with it, and the effective normalizing pool is on
    the order of (n_background_trajectories x average episode length),
    not just n_background_trajectories.

    This is arguably a MORE faithful sample-based approximation of
    Ziebart's own exact formulation (irl/maxent_linear.py's
    expected_feature_counts, weighted by the per-STATE visitation
    distribution D_{s,t} at each timestep) than the per-trajectory
    version was -- not a departure from the theory this project started
    from, a closer sample-based approximation of it.

    Only states 0..T-1 of each trajectory are included (states an
    action was actually taken from, needed for log pi(a_t|s_t)) -- the
    final post-episode state (index T) has no associated action and is
    excluded here, unlike the per-trajectory version which included it
    in the trajectory-level reward sum. reward_loss's background term
    must pool states the SAME way for its weighted sum to line up
    index-for-index with what this function returns.

    Returns shape (total number of (trajectory, timestep) pairs summed
    across all background_trajectories,), summing to 1.
    """
    log_ratios = []
    for traj in background_trajectories:
        obs_for_actions = traj["observations"][:-1]                     # (T, 15)
        actions = torch.from_numpy(traj["actions"]).float()             # (T, 2)

        with torch.no_grad():
            state_rewards = reward_net(torch.from_numpy(obs_for_actions).float())  # (T,)

            obs_dict = _unflatten_observations_batch(obs_for_actions)
            _, log_probs, _ = policy.policy.evaluate_actions(obs_dict, actions)    # (T,)

        log_ratios.append((state_rewards - log_probs).numpy())

    log_ratios = np.concatenate(log_ratios)  # (sum of T over all trajectories,)

    clip_value = np.percentile(log_ratios, clip_percentile)
    log_ratios = np.minimum(log_ratios, clip_value)

    log_Z = scipy.special.logsumexp(log_ratios)
    return np.exp(log_ratios - log_Z)


def reward_loss(
    reward_net: RewardNetwork,
    demonstrations: list[dict],
    background_trajectories: list[dict],
    importance_weights: np.ndarray,
) -> torch.Tensor:
    """GCL's per-iteration loss -- negative log-likelihood of the
    demonstrations under the current reward, sample-estimated:

        L(theta) = -mean_demo[ r_theta(tau) ] + sum_{i,t} w_{i,t} * r_theta(s_t^(i))

    where the first sum is per-trajectory (r_theta(tau) = sum_t
    r_theta(s_t), unaffected by the per-decision change below -- demo
    trajectories aren't importance-weighted at all, this term just
    averages over demonstrations). First term (gradient flows -- no
    torch.no_grad here, unlike compute_importance_weights) pushes reward
    up on demonstrated states. Second term pushes it down on the
    INDIVIDUAL background states the importance weights suggest the
    current reward over-favors.

    importance_weights must come from THIS module's compute_importance_weights
    (per-decision, pooled across (trajectory, timestep) pairs) -- shape
    must equal the total number of pooled background states below, not
    len(background_trajectories).
    """
    demo_rewards = [
        reward_net(torch.from_numpy(demo["observations"]).float()).sum()
        for demo in demonstrations
    ]
    demo_term = torch.stack(demo_rewards).mean()

    # Pool background states the SAME way compute_importance_weights does
    # -- states 0..T-1 of each trajectory, concatenated in the same
    # trajectory order -- so importance_weights[k] lines up with
    # pooled_states[k] index-for-index.
    pooled_states = np.concatenate([
        traj["observations"][:-1] for traj in background_trajectories
    ])
    background_state_rewards = reward_net(torch.from_numpy(pooled_states).float())  # (total_T,)
    weights_tensor = torch.from_numpy(importance_weights).float()
    background_term = (weights_tensor * background_state_rewards).sum()

    return -demo_term + background_term


def _save_checkpoint(checkpoint_dir: str, reward_net, optimizer, policy, iteration,
                      loss_history, success_rate_history):
    # Numbered per iteration (reward_net_iter{N}.pt, ...), NOT overwritten
    # in place -- the previous version overwrote the same fixed filenames
    # every time, which meant a run's full checkpoint history was gone
    # the moment it ended, with only the final snapshot ever available.
    # That made it impossible to check, after the fact, how much
    # reward_net actually moved between iterations -- exactly the
    # diagnostic needed when policy performance and recovered-reward
    # value diverge near the end of a run (see README Limitations).
    os.makedirs(checkpoint_dir, exist_ok=True)
    suffix = f"iter{iteration}"
    torch.save(reward_net.state_dict(), os.path.join(checkpoint_dir, f"reward_net_{suffix}.pt"))
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, f"optimizer_{suffix}.pt"))
    policy.save(os.path.join(checkpoint_dir, f"policy_{suffix}"))
    with open(os.path.join(checkpoint_dir, f"state_{suffix}.json"), "w") as f:
        json.dump({
            "iteration": iteration,
            "loss_history": loss_history,
            "success_rate_history": success_rate_history,
        }, f)
    print(f"[checkpoint] saved at iteration {iteration} -> {checkpoint_dir}")


def _find_latest_checkpoint_iteration(checkpoint_dir: str) -> int | None:
    """Scans checkpoint_dir for state_iter*.json files and returns the
    highest iteration number found, or None if there are none (e.g. a
    fresh checkpoint_dir, or one written by the old overwrite-in-place
    format -- that older format is NOT auto-detected here, so a
    checkpoint_dir carried over from before this change starts a fresh
    run rather than silently resuming from a differently-shaped state).
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    iterations = []
    for name in os.listdir(checkpoint_dir):
        if name.startswith("state_iter") and name.endswith(".json"):
            try:
                iterations.append(int(name[len("state_iter"):-len(".json")]))
            except ValueError:
                continue
    return max(iterations) if iterations else None


def _load_checkpoint(checkpoint_dir: str, reward_net, optimizer, iteration: int):
    suffix = f"iter{iteration}"
    reward_net.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"reward_net_{suffix}.pt")))
    optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"optimizer_{suffix}.pt")))
    with open(os.path.join(checkpoint_dir, f"state_{suffix}.json")) as f:
        state = json.load(f)
    return state["iteration"], state["loss_history"], state["success_rate_history"]


def train_gcl(
    nanogoal_path: str,
    demonstrations: list[dict],
    seed_mode: str,
    total_timesteps: int,
    n_iterations: int = 100,
    n_background_trajectories_per_iteration: int = 20,
    reward_learning_rate: float = 1e-3,
    importance_weight_clip_percentile: float = 95.0,
    n_envs: int = 1,
    seed: int = 0,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 10,
    tb_log_dir: str | None = None,
    tb_log_name: str = "gcl",
) -> tuple[RewardNetwork, PPO, dict]:
    """The alternating loop, once per iteration:
        1. collect n_background_trajectories_per_iteration rollouts from
           the CURRENT policy, with seeds drawn from seed_mode's own
           difficulty-respecting pool (deterministic=False -- exploration
           matters here, unlike demonstration generation)
        2. importance_weights = compute_importance_weights(...)
        3. one gradient step on reward_loss(...)
        4. push the updated reward_net weights to every parallel worker
           (see module docstring PARALLELISM note)
        5. policy.learn(total_timesteps // n_iterations, ...)

    total_timesteps is the OVERALL policy-training budget for this
    seed_mode (see experiments/phase0b_gcl_training.py's BUDGET table --
    calibrated against NanoGoal-RL's own reported easy/medium
    timesteps-to-convergence, not picked arbitrarily). Divided evenly
    across n_iterations; each chunk runs via one policy.learn() call.

    n_envs > 1 uses SubprocVecEnv for real parallelism (mirrors
    NanoGoal-RL's own train_*.py pattern) -- each worker gets an
    independently-seeded SeedModeEnv (seed + 1000*worker_idx) so parallel
    workers explore different episodes instead of following an identical
    sequence in lockstep, the same worker_seed_offset problem NanoGoal-RL
    itself documents having fixed.

    checkpoint_dir, if given, saves reward_net/optimizer/policy state
    every checkpoint_every iterations, and resumes automatically from
    there if a checkpoint already exists at that path -- unattended
    multi-hour runs should always set this.

    tb_log_dir/tb_log_name: if given, writes GCL-specific scalars
    (reward_loss, background_success_rate, policy_steps) to TensorBoard
    via SummaryWriter. PPO's OWN internal metrics (entropy, value_loss,
    policy_gradient_loss, etc.) are written separately by SB3 to the same
    directory automatically via policy.learn(tensorboard_log=...) -- both
    streams appear together in the same TensorBoard run.

    Returns (reward_net, policy, history), where history is
    {"loss": np.ndarray, "success_rate": np.ndarray} -- one entry per
    completed iteration (including any before a resume).
    """
    from torch.utils.tensorboard import SummaryWriter

    policy_update_steps_per_iteration = total_timesteps // n_iterations

    test_seeds = load_test_seeds(nanogoal_path)
    background_env = create_env(nanogoal_path)  # sequential, single-process --
                                                  # background sampling volume
                                                  # (tens per iteration) is
                                                  # negligible next to
                                                  # policy.learn()'s own
                                                  # rollout collection, not
                                                  # worth parallelizing too

    reward_net = RewardNetwork()
    optimizer = torch.optim.Adam(reward_net.parameters(), lr=reward_learning_rate)

    start_iteration = 0
    loss_history = []
    success_rate_history = []

    latest_checkpoint_iteration = (
        _find_latest_checkpoint_iteration(checkpoint_dir) if checkpoint_dir else None
    )
    resuming = latest_checkpoint_iteration is not None
    if resuming:
        assert checkpoint_dir is not None  # narrowing: latest_checkpoint_iteration is
                                            # only non-None when checkpoint_dir was given
        start_iteration, loss_history, success_rate_history = _load_checkpoint(
            checkpoint_dir, reward_net, optimizer, latest_checkpoint_iteration
        )
        print(f"[checkpoint] resuming from iteration {start_iteration}/{n_iterations}")

    def make_env(worker_idx):
        def _init():
            base_env = create_env(nanogoal_path)
            wrapped = RewardWrappedEnv(base_env, reward_net)
            seeded = SeedModeEnv(
                wrapped, test_seeds, seed_mode,
                rng=np.random.default_rng(seed + 1000 * worker_idx),
            )
            return Monitor(seeded)
        return _init

    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # SummaryWriter for GCL-specific metrics (reward_loss,
    # background_success_rate). PPO's own metrics (entropy, value_loss,
    # policy_gradient_loss, clip_fraction, etc.) are written separately
    # by SB3 to the same directory via policy.learn(tensorboard_log=...)
    # -- both streams appear together in the same TensorBoard run.
    writer: "SummaryWriter | None" = (
        SummaryWriter(log_dir=os.path.join(tb_log_dir, tb_log_name))
        if tb_log_dir else None
    )

    if resuming:
        assert checkpoint_dir is not None  # narrowing: same guard as above
        policy = PPO.load(
            os.path.join(checkpoint_dir, f"policy_iter{latest_checkpoint_iteration}"),
            env=vec_env, device="cpu", tensorboard_log=tb_log_dir,
        )
    else:
        policy = PPO(
            "MultiInputPolicy", vec_env, verbose=1, seed=seed, device="cpu",
            n_steps=max(1, 20_000 // n_envs),  # mirrors NanoGoal-RL's own
                                                 # train_*.py n_steps sizing
            tensorboard_log=tb_log_dir,
        )

    rng = np.random.default_rng(seed + 999_999)  # separate stream from
                                                    # workers' own, for
                                                    # background sampling

    for iteration in range(start_iteration, n_iterations):
        background_trajectories = [
            rollout(
                background_env, policy,
                seed=sample_seed_from_mode(test_seeds, seed_mode, rng),
                deterministic=False,
            )
            for _ in range(n_background_trajectories_per_iteration)
        ]

        importance_weights = compute_importance_weights(
            reward_net, background_trajectories, policy,
            clip_percentile=importance_weight_clip_percentile,
        )

        optimizer.zero_grad()
        loss = reward_loss(reward_net, demonstrations, background_trajectories, importance_weights)
        loss.backward()
        optimizer.step()

        # push updated weights across the process boundary -- see module
        # docstring PARALLELISM note; a no-op in effect for n_envs=1 but
        # always called for a single code path regardless of n_envs
        vec_env.env_method("set_reward_net_state", reward_net.state_dict())

        policy.learn(
            total_timesteps=policy_update_steps_per_iteration,
            reset_num_timesteps=False,
            tb_log_name=tb_log_name,
        )

        bg_success_rate = float(np.mean([t["is_success"] for t in background_trajectories]))
        loss_history.append(loss.item())
        success_rate_history.append(bg_success_rate)

        if writer is not None:
            writer.add_scalar("gcl/reward_loss", loss.item(), policy.num_timesteps)
            writer.add_scalar("gcl/background_success_rate", bg_success_rate, policy.num_timesteps)
            writer.add_scalar("gcl/iteration", iteration + 1, policy.num_timesteps)
            writer.flush()

        print(
            f"iteration {iteration + 1}/{n_iterations}: "
            f"reward_loss={loss.item():.4f}  "
            f"background_success_rate={bg_success_rate:.2f}  "
            f"policy_steps_so_far={policy.num_timesteps}"
        )

        if checkpoint_dir and (iteration + 1) % checkpoint_every == 0:
            assert checkpoint_dir is not None  # narrowing: bool(checkpoint_dir) guarantees str
            _save_checkpoint(
                checkpoint_dir, reward_net, optimizer, policy, iteration + 1,
                loss_history, success_rate_history,
            )

    if writer is not None:
        writer.close()
    vec_env.close()

    history = {
        "loss": np.array(loss_history),
        "success_rate": np.array(success_rate_history),
    }
    return reward_net, policy, history