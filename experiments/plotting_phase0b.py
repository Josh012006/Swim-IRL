"""Plotting helpers for Phase 0b (GCL/AIRL) experiment reports -- parallel
to experiments/plotting.py's role for Phase 0a. Lives here, not in eval/,
for the same reason as that file: these are reported RESULTS, not metrics
themselves (see README Reproducibility).
"""
import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_training_diagnostics(history: dict, output_path: str | None = None) -> None:
    """history: {"loss": array, "success_rate": array} -- train_gcl's
    return value. No concavity guarantee here (unlike Phase 0a's
    log-likelihood) -- a generally-decreasing loss is an informal health
    signal, not a proof; a rising background success rate alongside it is
    a second, independent signal that the recovered reward is actually
    guiding the policy somewhere useful, not just numerically shrinking.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["loss"])
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("reward_loss")
    axes[0].set_title("GCL reward loss (informal signal only -- no concavity guarantee)")
    axes[0].axhline(0, color="black", linewidth=0.5)

    axes[1].plot(history["success_rate"])
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("background rollout success rate")
    axes[1].set_title("Policy success rate during training")
    axes[1].set_ylim(-0.05, 1.05)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def collect_reward_comparison_data(
    reward_net,
    trajectories: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Pairs recovered reward_net(s_t) against NanoGoal-RL's own true
    per-step reward, across every step of every given trajectory (e.g.
    the demonstrations, or a batch of fresh evaluation rollouts).

    The true reward includes NanoGoal-RL's action-dependent
    spinning-penalty term (see README Limitations); reward_net(s_t)
    structurally cannot represent it, so some scatter is EXPECTED even
    under an otherwise-good state-only recovery. This plot is a direct
    way to see how much that mismatch actually costs -- not a claim that
    a perfect correlation is the bar for success.

    Returns (predicted, true), each shape (total steps across all
    trajectories,).
    """
    predicted = []
    true = []
    for traj in trajectories:
        obs = torch.from_numpy(traj["observations"][:-1]).float()  # states with a recorded true reward
        with torch.no_grad():
            predicted.append(reward_net(obs).numpy())
        true.append(traj["rewards"])
    return np.concatenate(predicted), np.concatenate(true)


def plot_reward_comparison(
    predicted: np.ndarray,
    true: np.ndarray,
    output_path: str | None = None,
) -> None:
    """Scatter of recovered vs. true reward, one point per state. IRL
    recovers reward up to an unknown positive affine transform (see
    Objective's ill-posedness framing) -- a good recovery shows points
    aligned along SOME line, not necessarily y=x. Pearson correlation is
    the honest summary statistic here, not RMSE against y=x.
    """
    correlation = np.corrcoef(predicted, true)[0, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true, predicted, alpha=0.3, s=10)
    ax.set_xlabel("true reward (NanoGoal-RL)")
    ax.set_ylabel("recovered reward (reward_net)")
    ax.set_title(f"Recovered vs. true reward (r = {correlation:.3f})")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_recovery_grid_heatmap(
    values: np.ndarray,           # shape (n_models, n_seed_modes)
    row_labels: list[str],        # model_difficulty values, e.g. ["easy", "medium"]
    col_labels: list[str],        # seed_mode values, e.g. ["easy", "easy_medium"]
    metric_name: str,
    output_path: str | None = None,
) -> None:
    """The agent-competence x environment-mix grid, as a heatmap. Rows =
    which trained agent generated the demonstrations, columns = which
    seed-mode distribution the demonstrations were drawn from. Shape is
    inferred from `values` itself, not hardcoded -- works the same for
    the current 2x2 grid as it did for the original 3x3.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(values, cmap="viridis")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("seed_mode (environment mix)")
    ax.set_ylabel("model_difficulty (agent)")
    ax.set_title(metric_name)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color="white")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


# Fixed concatenation order, matching sim.nanogoal_adapter.flatten_observation
# exactly: agent(2) + mvt(3) + delta_goal(2) + lidar(8). Used by both
# evaluation modes below so a feature index always means the same thing
# in both. "mvt"'s 3 components and "agent"'s 2 aren't given more specific
# names here -- NanoGoal-RL's own Dict observation space doesn't document
# their individual semantics beyond the key name, so labeling them further
# would be guessing, not reporting.
FEATURE_NAMES = (
    ["agent_0", "agent_1"]
    + ["mvt_0", "mvt_1", "mvt_2"]
    + ["delta_goal_0", "delta_goal_1"]
    + [f"lidar_{i}" for i in range(8)]
)


def collect_reward_component_data(reward_net, trajectories: list[dict]) -> dict:
    """For every pooled state across trajectories (same pooling
    collect_reward_comparison_data uses -- states 0..T-1, where a
    recorded true reward exists), extracts two quantities computable
    DIRECTLY from the observation's own structure, independent of
    knowing NanoGoal-RL's exact reward formula:

        dist_goal = ||delta_goal||     -- distance to goal
        min_lidar = min(lidar)         -- how close the nearest
                                           obstacle is, in whichever
                                           direction it's closest
                                           (smaller = an obstacle is near)

    This is the neural-network analogue of reading off individual
    theta_i weights in Phase 0a's linear MaxEnt IRL: we can't read a
    learned weight directly from a nonlinear reward_net, but since we
    know the OBSERVATION's own structure, we can still ask whether the
    recovered reward's output actually correlates with the same
    interpretable, problem-relevant quantities the true reward is known
    to depend on (the state-dependent goal-distance and collision terms
    -- see README Approach/Limitations for why only those two, not the
    action-dependent effort term, are checkable this way).

    Returns predicted (reward_net's own output) AND true reward
    alongside dist_goal/min_lidar for every pooled state, so the SAME
    correlation check can be run on both and compared directly -- if
    the true reward's own correlation with dist_goal is strongly
    negative but reward_net's isn't, that names a specific, concrete
    way recovery fell short, not just "the fit isn't perfect".

    Uses however many states `trajectories` contains -- pass enough
    episodes (10+ recommended) for the correlations and the observed
    feature ranges (see plot_partial_dependence) to be stable rather
    than dominated by one or two episodes' idiosyncrasies.
    """
    predicted_list, true_list = [], []
    dist_goal_list, min_lidar_list = [], []

    for traj in trajectories:
        obs = traj["observations"][:-1]  # states with a recorded true reward
        with torch.no_grad():
            predicted = reward_net(torch.from_numpy(obs).float()).numpy()
        true = traj["rewards"]

        dist_goal = np.linalg.norm(obs[:, 5:7], axis=1)
        min_lidar = obs[:, 7:15].min(axis=1)

        predicted_list.append(predicted)
        true_list.append(true)
        dist_goal_list.append(dist_goal)
        min_lidar_list.append(min_lidar)

    return {
        "predicted": np.concatenate(predicted_list),
        "true": np.concatenate(true_list),
        "dist_goal": np.concatenate(dist_goal_list),
        "min_lidar": np.concatenate(min_lidar_list),
        "all_states": np.concatenate([t["observations"][:-1] for t in trajectories]),
    }


def compute_reward_component_correlations(data: dict) -> dict:
    """Four correlations from collect_reward_component_data's output --
    expected signs, given NanoGoal-RL's known reward structure (goal-
    distance + collision terms): NEGATIVE for dist_goal (closer to goal
    should mean higher reward), POSITIVE for min_lidar (farther from
    the nearest obstacle should mean higher reward). These are
    predictions from the problem's known structure, not something this
    project verified independently -- worth keeping in mind when
    reading the numbers, not treated as certain ground truth.
    """
    def safe_corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float("nan")

    return {
        "predicted_vs_dist_goal": safe_corr(data["predicted"], data["dist_goal"]),
        "true_vs_dist_goal": safe_corr(data["true"], data["dist_goal"]),
        "predicted_vs_min_lidar": safe_corr(data["predicted"], data["min_lidar"]),
        "true_vs_min_lidar": safe_corr(data["true"], data["min_lidar"]),
    }


def plot_reward_components(data: dict, correlations: dict, output_path: str) -> None:
    """Recovered reward and true reward, each against dist_goal and
    min_lidar, as 4 side-by-side panels sharing the same x-axes pairwise
    -- not overlaid on one axis, since predicted and true reward live on
    different scales (IRL recovers reward up to an affine transform, see
    Objective) and forcing them onto one y-axis would visually
    exaggerate or hide differences that are just a scale artifact.

    True reward's y-axis uses symlog, not linear: NanoGoal-RL's true
    reward is DENSE (every single step has a small nonzero value, e.g.
    around -0.03, from the state-dependent goal-distance/collision
    terms) but occasionally spikes much larger in magnitude (a terminal
    penalty on collision/out-of-bounds, e.g. -13 or lower) -- verified
    directly on a real rollout: 0 exactly-zero steps out of 802. On a
    LINEAR y-axis spanning both scales, the dense small-magnitude signal
    visually collapses against y=0 and looks sparse, which it verifiably
    is not -- symlog keeps the near-zero region linear (where the dense
    signal actually lives) while still showing the rare large spikes,
    instead of letting them dominate the axis and hide everything else.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    axes[0, 0].scatter(data["dist_goal"], data["predicted"], alpha=0.2, s=8)
    axes[0, 0].set_xlabel("dist_goal")
    axes[0, 0].set_ylabel("recovered reward")
    axes[0, 0].set_title(f"Recovered vs dist_goal (r={correlations['predicted_vs_dist_goal']:.3f})")

    axes[0, 1].scatter(data["dist_goal"], data["true"], alpha=0.2, s=8, color="tab:orange")
    axes[0, 1].set_yscale("symlog", linthresh=0.1)
    axes[0, 1].set_xlabel("dist_goal")
    axes[0, 1].set_ylabel("true reward (symlog)")
    axes[0, 1].set_title(f"True vs dist_goal (r={correlations['true_vs_dist_goal']:.3f})")

    axes[1, 0].scatter(data["min_lidar"], data["predicted"], alpha=0.2, s=8)
    axes[1, 0].set_xlabel("min_lidar (obstacle proximity)")
    axes[1, 0].set_ylabel("recovered reward")
    axes[1, 0].set_title(f"Recovered vs min_lidar (r={correlations['predicted_vs_min_lidar']:.3f})")

    axes[1, 1].scatter(data["min_lidar"], data["true"], alpha=0.2, s=8, color="tab:orange")
    axes[1, 1].set_yscale("symlog", linthresh=0.1)
    axes[1, 1].set_xlabel("min_lidar (obstacle proximity)")
    axes[1, 1].set_ylabel("true reward (symlog)")
    axes[1, 1].set_title(f"True vs min_lidar (r={correlations['true_vs_min_lidar']:.3f})")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_partial_dependence(
    reward_net,
    reference_states: np.ndarray,
    output_path: str,
    n_points: int = 50,
) -> None:
    """For each of the 15 input dimensions, varies ONLY that dimension
    across its OBSERVED range in reference_states (never an arbitrary
    made-up range -- e.g. lidar's real range on this environment, not
    an assumed [0, 1]), holding the other 14 at reference_states' own
    mean, and plots reward_net's output as a function of it. The
    standard way to inspect what shape a black-box model learned along
    one input dimension at a time -- reveals shape (monotone, peaked,
    flat, non-monotone), which a single correlation coefficient cannot.

    reference_states should come from a real batch of rollout states
    (pass the same pooled states collect_reward_component_data used,
    via data["all_states"]) -- a small or unrepresentative reference
    batch would give an unreliable observed range and an unrealistic
    baseline for the 14 held-out dimensions.
    """
    baseline = reference_states.mean(axis=0)  # (15,)

    fig, axes = plt.subplots(3, 5, figsize=(20, 11))
    axes = axes.flatten()

    for i, name in enumerate(FEATURE_NAMES):
        low, high = reference_states[:, i].min(), reference_states[:, i].max()
        sweep_values = np.linspace(low, high, n_points)

        batch = np.tile(baseline, (n_points, 1))
        batch[:, i] = sweep_values

        with torch.no_grad():
            outputs = reward_net(torch.from_numpy(batch).float()).numpy()

        axes[i].plot(sweep_values, outputs)
        axes[i].set_title(name, fontsize=9)
        axes[i].tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
