'''Plot theta_true vs theta_hat per ground-truth reward, with the EVD
displayed for each. Lives here (not in eval/recovery.py) because it's a
reported RESULT of the experiment, not a metric -- see README
Reproducibility.'''
import matplotlib.pyplot as plt
import numpy as np


def plot_recovery_comparison(
    results: list[dict],
    # each dict: {"name": str, "theta_true": np.ndarray, "theta_hat": np.ndarray,
    #             "evd": float, "feature_names": list[str]}
    output_path: str | None = None,
) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        x = np.arange(len(result["feature_names"]))
        width = 0.35
        ax.bar(x - width / 2, result["theta_true"], width, label="theta_true")
        ax.bar(x + width / 2, result["theta_hat"], width, label="theta_hat")
        ax.set_xticks(x)
        ax.set_xticklabels(result["feature_names"], rotation=30, ha="right")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(result["name"])
        ax.legend()
        ax.text(
            0.5, -0.3, f"EVD = {result['evd']:.4f}",
            transform=ax.transAxes, ha="center", fontsize=11,
        )

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()