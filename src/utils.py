"""Utility functions for RA-RL-IDS.

Provides:
  - Deterministic seeding across Python, NumPy, and PyTorch.
  - Rotating-file + console logging setup via the standard logging module.
  - YAML configuration loader with basic validation.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seeds(seed: int) -> None:
    """Set random seeds for full reproducibility.

    Fixes seeds for:
      - Python's built-in ``random``
      - NumPy
      - PyTorch (CPU and CUDA)

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logging(
    log_file: str = "ra-rl-ids.log",
    level: str = "INFO",
    max_bytes: int = 5_242_880,
    backup_count: int = 3,
) -> logging.Logger:
    """Configure project-wide logging with console and rotating file output.

    Args:
        log_file: Path to the log file.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        max_bytes: Maximum size in bytes before the log file rotates.
        backup_count: Number of rotated backup log files to keep.

    Returns:
        The root logger configured for the project.
    """
    logger = logging.getLogger("ra_rl_ids")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # Rotating file handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load and return the YAML configuration dictionary.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is empty or malformed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None or not isinstance(config, dict):
        raise ValueError(f"Configuration file is empty or malformed: {config_path}")

    return config


def get_device() -> torch.device:
    """Detect and return the best available torch device.

    Returns:
        ``torch.device("cuda")`` if a CUDA GPU is available, otherwise
        ``torch.device("cpu")``.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.getLogger("ra_rl_ids").info(
            "CUDA available — using GPU: %s", torch.cuda.get_device_name(0)
        )
    else:
        device = torch.device("cpu")
        logging.getLogger("ra_rl_ids").info("CUDA not available — using CPU")
    return device


def ensure_dirs(config: Dict[str, Any]) -> None:
    """Create all output directories specified in the config if they don't exist.

    Args:
        config: Parsed configuration dictionary (expects a ``paths`` sub-dict).
    """
    paths = config.get("paths", {})
    for key in ("data_processed", "results", "figures", "checkpoints"):
        dir_path = paths.get(key, "")
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    # Also ensure data/raw exists (even though it's gitignored)
    Path(paths.get("data_raw", "data/raw")).mkdir(parents=True, exist_ok=True)
