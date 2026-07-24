"""Evaluation module for RA-RL-IDS.

Computes and exports:
  - Accuracy, precision, recall, F1 (macro + weighted), AUC (OvR).
  - Per-class recall breakdown.
  - Confusion matrix.
  - Exports metrics to JSON files in ``results/``.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_pipeline import run_pipeline
from src.env import IoMTIDSEnv
from src.model import DQNNetwork
from src.utils import ensure_dirs, load_config, set_seeds, setup_logging

logger = logging.getLogger("ra_rl_ids")


def evaluate_model(
    model: DQNNetwork,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run greedy inference and compute full classification metrics.

    Args:
        model: Trained DQN network.
        X: Feature matrix ``(n_samples, n_features)``.
        y: True labels ``(n_samples,)``.
        device: Torch device for inference.
        label_names: Optional list of human-readable class names.

    Returns:
        Dictionary containing all metrics, per-class recall, and confusion matrix.
    """
    model.eval()
    model.to(device)

    y_pred: List[int] = []

    # Batch inference for efficiency
    batch_size = 256
    n = len(X)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.FloatTensor(X[start:end]).to(device)
        with torch.no_grad():
            q_values = model(batch)
        preds = q_values.argmax(dim=-1).cpu().numpy().tolist()
        y_pred.extend(preds)

    y_pred_arr = np.array(y_pred, dtype=np.int64)
    y_true = y.astype(np.int64)

    num_classes = int(max(y_true.max(), y_pred_arr.max())) + 1

    # Overall metrics
    acc = float(accuracy_score(y_true, y_pred_arr))
    prec_macro = float(precision_score(y_true, y_pred_arr, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred_arr, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred_arr, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred_arr, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred_arr, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred_arr, average="weighted", zero_division=0))

    # Per-class recall
    per_class_recall: Dict[str, float] = {}
    for cls_idx in range(num_classes):
        mask = y_true == cls_idx
        if mask.sum() > 0:
            cls_recall = float((y_pred_arr[mask] == cls_idx).sum() / mask.sum())
        else:
            cls_recall = 0.0
        cls_name = label_names[cls_idx] if label_names else str(cls_idx)
        per_class_recall[cls_name] = round(cls_recall, 4)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_arr, labels=list(range(num_classes)))

    metrics = {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_macro": round(rec_macro, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred_arr.tolist(),
    }

    logger.info(
        "Evaluation — Acc: %.4f | Prec(M): %.4f | Rec(M): %.4f | F1(M): %.4f | F1(W): %.4f",
        acc, prec_macro, rec_macro, f1_macro, f1_weighted,
    )
    return metrics


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> DQNNetwork:
    """Load a trained DQN model from a checkpoint file.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint.
        device: Target torch device.

    Returns:
        Loaded ``DQNNetwork`` in eval mode.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model_cfg = config["model"]

    model = DQNNetwork(
        n_features=ckpt["n_features"],
        n_actions=ckpt["n_actions"],
        cnn_out_channels=model_cfg["cnn_out_channels"],
        cnn_kernel_size=model_cfg["cnn_kernel_size"],
        lstm_hidden_size=model_cfg["lstm_hidden_size"],
        lstm_num_layers=model_cfg["lstm_num_layers"],
        dropout=model_cfg["dropout"],
        dqn_hidden_size=model_cfg["dqn_hidden_size"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def run_evaluation(
    config: Dict[str, Any],
    pipeline_data: Dict[str, Any],
) -> None:
    """Evaluate both baseline and reward-shaped models on the test set.

    Args:
        config: Parsed configuration dictionary.
        pipeline_data: Data from ``run_pipeline()``.
    """
    device = torch.device("cpu")  # Evaluate on CPU for consistency
    results_dir = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(config["paths"]["checkpoints"])

    label_names = list(pipeline_data["label_encoder"].classes_)

    # Evaluate baseline
    baseline_ckpt = ckpt_dir / "baseline_dqn.pt"
    if baseline_ckpt.exists():
        logger.info("Evaluating baseline model...")
        model = load_model_from_checkpoint(str(baseline_ckpt), device)
        baseline_metrics = evaluate_model(
            model,
            pipeline_data["X_test"],
            pipeline_data["y_test"],
            device,
            label_names,
        )
        baseline_metrics["reward_mode"] = "flat"
        with open(str(results_dir / "baseline_metrics.json"), "w") as f:
            json.dump(baseline_metrics, f, indent=2)
        logger.info("Saved baseline_metrics.json")
    else:
        logger.warning("Baseline checkpoint not found at %s", baseline_ckpt)

    # Evaluate reward-shaped
    rs_ckpt = ckpt_dir / "reward_shaped_dqn.pt"
    if rs_ckpt.exists():
        logger.info("Evaluating reward-shaped model...")
        model = load_model_from_checkpoint(str(rs_ckpt), device)
        rs_metrics = evaluate_model(
            model,
            pipeline_data["X_test"],
            pipeline_data["y_test"],
            device,
            label_names,
        )
        rs_metrics["reward_mode"] = "weighted"
        with open(str(results_dir / "reward_shaped_metrics.json"), "w") as f:
            json.dump(rs_metrics, f, indent=2)
        logger.info("Saved reward_shaped_metrics.json")
    else:
        logger.warning("Reward-shaped checkpoint not found at %s", rs_ckpt)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RA-RL-IDS models")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config["paths"]["log_file"], config["logging"]["level"])
    set_seeds(config["seed"])
    ensure_dirs(config)

    pipeline_data = run_pipeline(config)
    run_evaluation(config, pipeline_data)


if __name__ == "__main__":
    main()
