"""Compression and edge-device benchmarking for RA-RL-IDS.

Applies PyTorch dynamic INT8 quantization to the trained reward-shaped
DQN model and measures:
  - Model binary file size (MB).
  - Average per-sample inference latency (ms) — with warmup.
  - Peak memory footprint (MB) via ``tracemalloc``.
  - Classification metrics (accuracy, macro F1, per-class recall).

All benchmarking runs strictly on **CPU** to simulate IoMT edge-device
conditions (Gap A from the project specification).
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
import os
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.ao.quantization as quantization

from src.data_pipeline import run_pipeline
from src.evaluate import evaluate_model, load_model_from_checkpoint
from src.model import DQNNetwork
from src.utils import ensure_dirs, load_config, set_seeds, setup_logging

logger = logging.getLogger("ra_rl_ids")


def get_model_size_mb(model: nn.Module) -> float:
    """Measure model file size in MB by saving to a temp file.

    Args:
        model: PyTorch model.

    Returns:
        File size in megabytes.
    """
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size_bytes = os.path.getsize(f.name)
    os.unlink(f.name)
    return size_bytes / (1024 * 1024)


def measure_latency(
    model: nn.Module,
    X: np.ndarray,
    num_samples: int = 1000,
    warmup: int = 50,
) -> Dict[str, float]:
    """Measure per-sample inference latency on CPU.

    Performs ``warmup`` runs before timing, then measures ``num_samples``
    individual forward passes.

    Args:
        model: PyTorch model in eval mode, on CPU.
        X: Feature matrix to sample from.
        num_samples: Number of inference runs to time.
        warmup: Number of warmup runs to discard.

    Returns:
        Dictionary with ``avg_latency_ms``, ``std_latency_ms``,
        ``min_latency_ms``, ``max_latency_ms``, ``p95_latency_ms``.
    """
    model.eval()
    indices = np.random.choice(len(X), size=num_samples + warmup, replace=True)

    # Warmup
    for i in range(warmup):
        sample = torch.FloatTensor(X[indices[i]]).unsqueeze(0)
        with torch.no_grad():
            _ = model(sample)

    # Timed runs
    latencies: List[float] = []
    for i in range(warmup, warmup + num_samples):
        sample = torch.FloatTensor(X[indices[i]]).unsqueeze(0)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(sample)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    latencies_arr = np.array(latencies)
    return {
        "avg_latency_ms": round(float(latencies_arr.mean()), 4),
        "std_latency_ms": round(float(latencies_arr.std()), 4),
        "min_latency_ms": round(float(latencies_arr.min()), 4),
        "max_latency_ms": round(float(latencies_arr.max()), 4),
        "p95_latency_ms": round(float(np.percentile(latencies_arr, 95)), 4),
    }


def measure_peak_memory_mb(
    model: nn.Module,
    X: np.ndarray,
    num_samples: int = 100,
) -> float:
    """Measure peak memory usage during inference using tracemalloc.

    Args:
        model: PyTorch model in eval mode.
        X: Feature matrix.
        num_samples: Number of inference runs.

    Returns:
        Peak memory in megabytes.
    """
    model.eval()
    tracemalloc.start()

    indices = np.random.choice(len(X), size=num_samples, replace=True)
    for i in range(num_samples):
        sample = torch.FloatTensor(X[indices[i]]).unsqueeze(0)
        with torch.no_grad():
            _ = model(sample)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def quantize_model(model: DQNNetwork) -> nn.Module:
    """Apply dynamic INT8 quantization to the model.

    Targets ``nn.Linear`` and ``nn.LSTM`` layers, which are well-supported
    by PyTorch dynamic quantization. Conv1d is left as-is since dynamic
    quantization support is limited for Conv layers.

    Args:
        model: Trained DQN model on CPU.

    Returns:
        Quantized model.
    """
    model.cpu()
    model.eval()

    quantized = quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM},
        dtype=torch.qint8,
    )
    logger.info("Applied dynamic INT8 quantization (Linear + LSTM layers)")
    return quantized


def run_compression_benchmark(
    config: Dict[str, Any],
    pipeline_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute the full compression and benchmarking pipeline.

    Steps:
      1. Load the trained reward-shaped model.
      2. Measure uncompressed metrics (size, latency, memory, accuracy).
      3. Quantize the model.
      4. Measure compressed metrics.
      5. Save quantized model checkpoint.
      6. Export comparison to ``results/compression_benchmark.json``.

    Args:
        config: Parsed configuration dictionary.
        pipeline_data: Data from ``run_pipeline()``.

    Returns:
        Benchmark results dictionary.
    """
    comp_cfg = config["compression"]
    paths_cfg = config["paths"]
    device = torch.device("cpu")  # All benchmarking on CPU

    ckpt_dir = Path(paths_cfg["checkpoints"])
    results_dir = Path(paths_cfg["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    label_names = list(pipeline_data["label_encoder"].classes_)
    X_test = pipeline_data["X_test"]
    y_test = pipeline_data["y_test"]

    # Load reward-shaped model
    rs_ckpt = ckpt_dir / "reward_shaped_dqn.pt"
    if not rs_ckpt.exists():
        logger.error("Reward-shaped checkpoint not found: %s", rs_ckpt)
        raise FileNotFoundError(f"Checkpoint not found: {rs_ckpt}")

    model = load_model_from_checkpoint(str(rs_ckpt), device)

    logger.info("=" * 60)
    logger.info("Compression Benchmark — CPU-only edge simulation")
    logger.info("=" * 60)

    # ---- Uncompressed metrics ----
    logger.info("Measuring UNCOMPRESSED model...")
    uncompressed_size = get_model_size_mb(model)
    uncompressed_latency = measure_latency(
        model, X_test,
        num_samples=comp_cfg["num_inference_samples"],
        warmup=comp_cfg["warmup_samples"],
    )
    uncompressed_memory = measure_peak_memory_mb(model, X_test)
    uncompressed_metrics = evaluate_model(model, X_test, y_test, device, label_names)

    logger.info(
        "Uncompressed — Size: %.2f MB | Latency: %.4f ms | Memory: %.2f MB | Acc: %.4f",
        uncompressed_size,
        uncompressed_latency["avg_latency_ms"],
        uncompressed_memory,
        uncompressed_metrics["accuracy"],
    )

    # ---- Quantize ----
    quantized_model = quantize_model(model)

    # ---- Compressed metrics ----
    logger.info("Measuring QUANTIZED model...")
    compressed_size = get_model_size_mb(quantized_model)
    compressed_latency = measure_latency(
        quantized_model, X_test,
        num_samples=comp_cfg["num_inference_samples"],
        warmup=comp_cfg["warmup_samples"],
    )
    compressed_memory = measure_peak_memory_mb(quantized_model, X_test)
    compressed_metrics = evaluate_model(
        quantized_model, X_test, y_test, device, label_names
    )

    logger.info(
        "Quantized   — Size: %.2f MB | Latency: %.4f ms | Memory: %.2f MB | Acc: %.4f",
        compressed_size,
        compressed_latency["avg_latency_ms"],
        compressed_memory,
        compressed_metrics["accuracy"],
    )

    # ---- Save quantized model ----
    quantized_ckpt = ckpt_dir / "reward_shaped_dqn_quantized.pt"
    torch.save(quantized_model.state_dict(), str(quantized_ckpt))
    logger.info("Saved quantized checkpoint: %s", quantized_ckpt)

    # ---- Build comparison ----
    size_reduction = (1 - compressed_size / uncompressed_size) * 100 if uncompressed_size > 0 else 0
    latency_reduction = (
        (1 - compressed_latency["avg_latency_ms"] / uncompressed_latency["avg_latency_ms"]) * 100
        if uncompressed_latency["avg_latency_ms"] > 0
        else 0
    )
    accuracy_drop = (uncompressed_metrics["accuracy"] - compressed_metrics["accuracy"]) * 100

    benchmark = {
        "uncompressed": {
            "model_size_mb": round(uncompressed_size, 4),
            "latency": uncompressed_latency,
            "peak_memory_mb": round(uncompressed_memory, 4),
            "accuracy": uncompressed_metrics["accuracy"],
            "f1_macro": uncompressed_metrics["f1_macro"],
            "per_class_recall": uncompressed_metrics["per_class_recall"],
        },
        "quantized": {
            "model_size_mb": round(compressed_size, 4),
            "latency": compressed_latency,
            "peak_memory_mb": round(compressed_memory, 4),
            "accuracy": compressed_metrics["accuracy"],
            "f1_macro": compressed_metrics["f1_macro"],
            "per_class_recall": compressed_metrics["per_class_recall"],
        },
        "comparison": {
            "size_reduction_pct": round(size_reduction, 2),
            "latency_reduction_pct": round(latency_reduction, 2),
            "accuracy_drop_pct": round(accuracy_drop, 2),
            "meets_latency_budget": (
                compressed_latency["avg_latency_ms"] < comp_cfg["latency_budget_ms"]
            ),
            "latency_budget_ms": comp_cfg["latency_budget_ms"],
        },
    }

    # Save
    benchmark_path = results_dir / "compression_benchmark.json"
    with open(str(benchmark_path), "w") as f:
        json.dump(benchmark, f, indent=2)
    logger.info("Saved compression benchmark: %s", benchmark_path)

    return benchmark


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point for compression benchmarking."""
    import argparse

    parser = argparse.ArgumentParser(description="RA-RL-IDS compression benchmark")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config["paths"]["log_file"], config["logging"]["level"])
    set_seeds(config["seed"])
    ensure_dirs(config)

    pipeline_data = run_pipeline(config)
    run_compression_benchmark(config, pipeline_data)


if __name__ == "__main__":
    main()
