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
    values: np.ndarray,           # shape (3, 3)
    row_labels: list[str],        # e.g. ["easy", "easy_medium", "mixed"] agent
    col_labels: list[str],        # e.g. ["easy", "easy_medium", "mixed"] seed_mode
    metric_name: str,
    output_path: str | None = None,
) -> None:
    """The 3x3 agent-competence x environment-mix grid, as a heatmap.
    Rows = which trained agent generated the demonstrations, columns =
    which seed-mode distribution the demonstrations were drawn from.
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