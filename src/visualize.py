"""Visualization module for RA-RL-IDS.

Generates five publication-ready matplotlib figures:
  1. ``per_class_recall_comparison.png`` — bar chart comparing flat vs weighted.
  2. ``compression_comparison.png`` — grouped bar chart: size, latency, memory.
  3. ``confusion_matrix_baseline.png`` — baseline DQN confusion matrix heatmap.
  4. ``confusion_matrix_final.png`` — reward-shaped DQN confusion matrix heatmap.
  5. ``training_curve.png`` — episode rewards with moving average.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/headless environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils import load_config, setup_logging

logger = logging.getLogger("ra_rl_ids")

# Consistent styling
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})


def plot_per_class_recall_comparison(
    baseline_metrics: Dict[str, Any],
    reward_shaped_metrics: Dict[str, Any],
    class_counts: Optional[Dict[str, int]] = None,
    save_path: str = "results/figures/per_class_recall_comparison.png",
) -> None:
    """Bar chart comparing per-class recall between baseline and reward-shaped.

    Classes are sorted by sample count (rarest first) if class_counts provided.

    Args:
        baseline_metrics: Baseline model metrics dict.
        reward_shaped_metrics: Reward-shaped model metrics dict.
        class_counts: Optional dict mapping class names to sample counts.
        save_path: Output file path.
    """
    baseline_recall = baseline_metrics["per_class_recall"]
    rs_recall = reward_shaped_metrics["per_class_recall"]

    classes = list(baseline_recall.keys())

    # Sort by rarest first
    if class_counts:
        classes = sorted(classes, key=lambda c: class_counts.get(c, 0))

    baseline_vals = [baseline_recall.get(c, 0) for c in classes]
    rs_vals = [rs_recall.get(c, 0) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width / 2, baseline_vals, width, label="Baseline (Flat)", color="#4c72b0", alpha=0.85)
    bars2 = ax.bar(x + width / 2, rs_vals, width, label="Reward-Shaped (Weighted)", color="#dd8452", alpha=0.85)

    ax.set_xlabel("Attack Class (sorted by rarity ← rarest first)")
    ax.set_ylabel("Recall")
    ax.set_title("Per-Class Recall: Baseline vs. Reward-Shaped DQN (Gap B)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=6)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=6)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", save_path)


def plot_compression_comparison(
    benchmark: Dict[str, Any],
    save_path: str = "results/figures/compression_comparison.png",
) -> None:
    """Grouped bar chart comparing uncompressed vs quantized model metrics.

    Shows model size (MB), average latency (ms), and peak memory (MB).

    Args:
        benchmark: Compression benchmark dictionary.
        save_path: Output file path.
    """
    metrics = ["Model Size (MB)", "Avg Latency (ms)", "Peak Memory (MB)"]
    uncomp = benchmark["uncompressed"]
    quant = benchmark["quantized"]

    uncomp_vals = [
        uncomp["model_size_mb"],
        uncomp["latency"]["avg_latency_ms"],
        uncomp["peak_memory_mb"],
    ]
    quant_vals = [
        quant["model_size_mb"],
        quant["latency"]["avg_latency_ms"],
        quant["peak_memory_mb"],
    ]

    x = np.arange(len(metrics))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, uncomp_vals, width, label="Uncompressed", color="#4c72b0")
    bars2 = ax.bar(x + width / 2, quant_vals, width, label="Quantized (INT8)", color="#55a868")

    ax.set_ylabel("Value")
    ax.set_title("Resource Cost: Uncompressed vs. Quantized (Gap A)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # Add reduction percentages
    comp = benchmark.get("comparison", {})
    reduction_text = (
        f"Size ↓ {comp.get('size_reduction_pct', 0):.1f}%  |  "
        f"Latency ↓ {comp.get('latency_reduction_pct', 0):.1f}%  |  "
        f"Acc drop: {comp.get('accuracy_drop_pct', 0):.2f}%"
    )
    ax.text(0.5, -0.12, reduction_text, transform=ax.transAxes,
            ha="center", fontsize=9, style="italic")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", save_path)


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    title: str,
    save_path: str,
) -> None:
    """Heatmap confusion matrix.

    Args:
        cm: Confusion matrix as a list of lists.
        class_names: Class label names.
        title: Plot title.
        save_path: Output file path.
    """
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", save_path)


def plot_training_curves(
    baseline_ckpt_path: str,
    reward_shaped_ckpt_path: str,
    save_path: str = "results/figures/training_curve.png",
    window: int = 10,
) -> None:
    """Plot episode reward curves for baseline and reward-shaped agents.

    Args:
        baseline_ckpt_path: Path to baseline checkpoint.
        reward_shaped_ckpt_path: Path to reward-shaped checkpoint.
        save_path: Output file path.
        window: Moving average window size.
    """
    import torch

    fig, ax = plt.subplots(figsize=(10, 5))

    for label, ckpt_path, color in [
        ("Baseline (Flat)", baseline_ckpt_path, "#4c72b0"),
        ("Reward-Shaped (Weighted)", reward_shaped_ckpt_path, "#dd8452"),
    ]:
        if not Path(ckpt_path).exists():
            logger.warning("Checkpoint not found: %s — skipping", ckpt_path)
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        rewards = ckpt.get("episode_rewards", [])
        if not rewards:
            continue

        episodes = np.arange(1, len(rewards) + 1)
        rewards_arr = np.array(rewards)

        # Raw rewards (faded)
        ax.plot(episodes, rewards_arr, alpha=0.2, color=color, linewidth=0.8)

        # Moving average
        if len(rewards_arr) >= window:
            ma = np.convolve(rewards_arr, np.ones(window) / window, mode="valid")
            ax.plot(
                episodes[window - 1:],
                ma,
                label=f"{label} (MA-{window})",
                color=color,
                linewidth=2,
            )
        else:
            ax.plot(episodes, rewards_arr, label=label, color=color, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Episode Reward")
    ax.set_title("Training Curves: Baseline vs. Reward-Shaped DQN")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", save_path)


def generate_all_figures(config: Dict[str, Any]) -> None:
    """Generate all 5 visualization figures.

    Reads metrics JSON files and checkpoint files from the results and
    checkpoints directories.

    Args:
        config: Parsed configuration dictionary.
    """
    paths = config["paths"]
    results_dir = Path(paths["results"])
    figures_dir = Path(paths["figures"])
    ckpt_dir = Path(paths["checkpoints"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics
    baseline_path = results_dir / "baseline_metrics.json"
    rs_path = results_dir / "reward_shaped_metrics.json"
    comp_path = results_dir / "compression_benchmark.json"

    # Load class distribution for sorting
    class_counts: Dict[str, int] = {}
    dist_path = results_dir / "class_distribution.csv"
    if dist_path.exists():
        import pandas as pd
        dist_df = pd.read_csv(str(dist_path))
        class_counts = dict(zip(dist_df["class_name"], dist_df["total_count"]))

    # 1. Per-class recall comparison
    if baseline_path.exists() and rs_path.exists():
        with open(str(baseline_path)) as f:
            baseline = json.load(f)
        with open(str(rs_path)) as f:
            rs = json.load(f)

        plot_per_class_recall_comparison(
            baseline, rs, class_counts,
            save_path=str(figures_dir / "per_class_recall_comparison.png"),
        )

        # 3 & 4. Confusion matrices
        if "confusion_matrix" in baseline:
            class_names = list(baseline["per_class_recall"].keys())
            plot_confusion_matrix(
                baseline["confusion_matrix"],
                class_names,
                "Confusion Matrix — Baseline DQN (Flat Reward)",
                str(figures_dir / "confusion_matrix_baseline.png"),
            )
        if "confusion_matrix" in rs:
            class_names = list(rs["per_class_recall"].keys())
            plot_confusion_matrix(
                rs["confusion_matrix"],
                class_names,
                "Confusion Matrix — Reward-Shaped DQN (Weighted)",
                str(figures_dir / "confusion_matrix_final.png"),
            )
    else:
        logger.warning("Metrics files not found — skipping recall & confusion plots")

    # 2. Compression comparison
    if comp_path.exists():
        with open(str(comp_path)) as f:
            comp = json.load(f)
        plot_compression_comparison(
            comp,
            save_path=str(figures_dir / "compression_comparison.png"),
        )
    else:
        logger.warning("Compression benchmark not found — skipping compression plot")

    # 5. Training curves
    plot_training_curves(
        baseline_ckpt_path=str(ckpt_dir / "baseline_dqn.pt"),
        reward_shaped_ckpt_path=str(ckpt_dir / "reward_shaped_dqn.pt"),
        save_path=str(figures_dir / "training_curve.png"),
    )

    logger.info("All figures generated in %s", figures_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point for visualization generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate RA-RL-IDS figures")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config["paths"]["log_file"], config["logging"]["level"])

    generate_all_figures(config)


if __name__ == "__main__":
    main()
