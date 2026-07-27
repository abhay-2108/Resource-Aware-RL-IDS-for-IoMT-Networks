"""Data pipeline for RA-RL-IDS.

Handles:
  - Loading and merging CICIoMT2024 CSV files from ``data/raw/``.
  - Synthetic IoMT dataset generation as a fallback when real CSVs are absent.
  - Cleaning, scaling, and mutual-information-based feature selection.
  - Stratified train / validation / test splitting.
  - Exporting class distribution to ``results/class_distribution.csv``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger("ra_rl_ids")

# ---------------------------------------------------------------------------
# CICIoMT2024 attack taxonomy (used by the synthetic generator)
# ---------------------------------------------------------------------------
IOMT_CLASSES: List[str] = [
    "Benign",
    "DDoS-ICMP_Flood",
    "DDoS-TCP_Flood",
    "DDoS-UDP_Flood",
    "DDoS-PSHACK_Flood",
    "DDoS-SYN_Flood",
    "DDoS-SynonymousIP_Flood",
    "DoS-TCP_Flood",
    "DoS-UDP_Flood",
    "DoS-SYN_Flood",
    "Recon-HostDiscovery",
    "Recon-PortScan",
    "Recon-OSScan",
    "MQTT-Publish",
    "Spoofing-ARP",
    "Spoofing-DNS",
]

# Realistic imbalanced distribution weights (sums to ~1.0)
# Benign is majority; MQTT/Spoofing/Recon-OS are rare
_CLASS_WEIGHTS: List[float] = [
    0.30,   # Benign
    0.10,   # DDoS-ICMP_Flood
    0.09,   # DDoS-TCP_Flood
    0.08,   # DDoS-UDP_Flood
    0.07,   # DDoS-PSHACK_Flood
    0.06,   # DDoS-SYN_Flood
    0.04,   # DDoS-SynonymousIP_Flood
    0.05,   # DoS-TCP_Flood
    0.04,   # DoS-UDP_Flood
    0.03,   # DoS-SYN_Flood
    0.03,   # Recon-HostDiscovery
    0.03,   # Recon-PortScan
    0.015,  # Recon-OSScan
    0.02,   # MQTT-Publish
    0.015,  # Spoofing-ARP
    0.01,   # Spoofing-DNS
]

# Typical CICIoMT2024-like flow feature names (46 features)
_SYNTHETIC_FEATURE_NAMES: List[str] = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration",
    "Rate", "Srate", "Drate", "fin_flag_number", "syn_flag_number",
    "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "ece_flag_number", "cwr_flag_number", "ack_count",
    "syn_count", "fin_count", "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size",
    "IAT", "Number", "Magnitue", "Radius", "Covariance",
    "Variance", "Weight",
]


def _generate_synthetic_dataset(
    total_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Generate a realistic synthetic IoMT network flow dataset.

    The generator creates class-specific feature distributions so that
    attack types have distinguishable statistical signatures, mimicking
    the structure of CICIoMT2024 flow data.

    Args:
        total_samples: Total number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        A ``pandas.DataFrame`` with feature columns and a ``label`` column.
    """
    rng = np.random.RandomState(seed)
    frames: List[pd.DataFrame] = []

    for cls_idx, (cls_name, cls_weight) in enumerate(
        zip(IOMT_CLASSES, _CLASS_WEIGHTS)
    ):
        n = max(int(total_samples * cls_weight), 10)  # at least 10 samples
        num_features = len(_SYNTHETIC_FEATURE_NAMES)

        # Base feature generation with class-specific mean shifts
        # so that a classifier can learn distinguishing patterns
        mean_shift = cls_idx * 0.5
        noise_scale = 1.0 + cls_idx * 0.1

        data = rng.randn(n, num_features) * noise_scale + mean_shift

        # Add class-specific feature patterns
        if "DDoS" in cls_name:
            # DDoS: high rate, high packet counts
            data[:, 4] += 5.0   # Rate
            data[:, 5] += 4.0   # Srate
            data[:, 37] += 3.0  # Std
        elif "DoS" in cls_name:
            # DoS: moderate rate, specific flag patterns
            data[:, 4] += 3.0   # Rate
            data[:, 8] += 2.0   # syn_flag_number
        elif "Recon" in cls_name:
            # Recon: low duration, scanning patterns
            data[:, 0] -= 2.0   # flow_duration (short)
            data[:, 39] += 2.0  # IAT (varied)
        elif "MQTT" in cls_name:
            # MQTT: protocol-specific patterns
            data[:, 2] += 3.0   # Protocol Type
            data[:, 33] += 2.0  # Tot sum
        elif "Spoofing" in cls_name:
            # Spoofing: ARP/DNS specific
            data[:, 29] += 3.0  # ARP
            data[:, 21] += 2.0  # DNS

        # Benign traffic is lower magnitude overall
        if cls_name == "Benign":
            data *= 0.5

        df = pd.DataFrame(data, columns=_SYNTHETIC_FEATURE_NAMES)
        df["label"] = cls_name
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    # Shuffle
    result = result.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    logger.info(
        "Generated synthetic IoMT dataset: %d samples, %d classes",
        len(result),
        len(IOMT_CLASSES),
    )
    return result


def load_raw_data(raw_dir: str, label_column: str = "label") -> Optional[pd.DataFrame]:
    """Attempt to load and merge all CSV files from the raw data directory.

    Args:
        raw_dir: Path to directory containing CICIoMT2024 CSV files.
        label_column: Name of the label/target column.

    Returns:
        Merged ``DataFrame`` or ``None`` if no CSVs are found.
    """
    raw_path = Path(raw_dir)
    csv_files = sorted(raw_path.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in %s — will use synthetic data", raw_dir)
        return None

    frames: List[pd.DataFrame] = []
    for csv_file in csv_files:
        logger.info("Loading %s ...", csv_file.name)
        df = pd.read_csv(csv_file, low_memory=False)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded %d CSV files → %d total samples, %d columns",
        len(csv_files),
        len(merged),
        len(merged.columns),
    )

    # Ensure the label column exists
    if label_column not in merged.columns:
        # Try common alternative names
        for alt in ("Label", "class", "Class", "attack_type", "Attack"):
            if alt in merged.columns:
                merged.rename(columns={alt: label_column}, inplace=True)
                logger.info("Renamed column '%s' → '%s'", alt, label_column)
                break
        else:
            raise ValueError(
                f"Label column '{label_column}' not found. "
                f"Available columns: {list(merged.columns)}"
            )
    return merged


def clean_dataframe(df: pd.DataFrame, label_column: str = "label") -> pd.DataFrame:
    """Clean the dataframe: handle missing values, infinities, and non-numeric cols.

    Args:
        df: Raw dataframe.
        label_column: Name of the label column (excluded from cleaning).

    Returns:
        Cleaned dataframe with only numeric features + label column.
    """
    initial_shape = df.shape
    # Separate label
    labels = df[label_column].copy()
    features = df.drop(columns=[label_column])

    # Keep only numeric columns
    features = features.select_dtypes(include=[np.number])

    # Replace infinities with NaN, then fill NaN with column median, fallback to 0.0
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median()).fillna(0.0)

    # Drop columns that are constant (zero variance)
    non_constant = features.columns[features.std() > 1e-10]
    features = features[non_constant]

    # Reattach labels
    features[label_column] = labels.values
    logger.info(
        "Cleaned dataframe: %s -> %s (dropped %d columns)",
        initial_shape,
        features.shape,
        initial_shape[1] - features.shape[1],
    )
    return features


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    num_features: int = 25,
    seed: int = 42,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Select top-K features by mutual information with the label.

    Args:
        X: Feature matrix ``(n_samples, n_features)``.
        y: Encoded label array ``(n_samples,)``.
        feature_names: List of feature column names.
        num_features: Number of top features to keep.
        seed: Random seed for MI estimation.

    Returns:
        Tuple of (selected feature matrix, selected feature names, MI scores).
    """
    num_features = min(num_features, X.shape[1])
    mi_scores = mutual_info_classif(X, y, random_state=seed)
    top_indices = np.argsort(mi_scores)[::-1][:num_features]
    selected_names = [feature_names[i] for i in top_indices]

    logger.info(
        "Selected top %d features by mutual information (top-3: %s)",
        num_features,
        selected_names[:3],
    )
    return X[:, top_indices], selected_names, mi_scores


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the full data pipeline.

    Steps:
      1. Load real CSVs or generate synthetic data.
      2. Clean and encode labels.
      3. Scale features with ``StandardScaler``.
      4. Select top-K features by mutual information.
      5. Stratified train/val/test split.
      6. Export class distribution and processed data.

    Args:
        config: Parsed configuration dictionary.

    Returns:
        Dictionary with keys: ``X_train``, ``X_val``, ``X_test``, ``y_train``,
        ``y_val``, ``y_test``, ``label_encoder``, ``scaler``,
        ``feature_names``, ``class_weights``, ``num_classes``.
    """
    ds_cfg = config["dataset"]
    paths_cfg = config["paths"]
    seed = config["seed"]

    # ---- 1. Load or generate ----
    raw_dir = paths_cfg["data_raw"]
    label_col = ds_cfg["label_column"]

    df = load_raw_data(raw_dir, label_col)
    if df is None:
        df = _generate_synthetic_dataset(
            total_samples=ds_cfg["synthetic_total_samples"],
            seed=seed,
        )

    # ---- 2. Clean ----
    df = clean_dataframe(df, label_col)

    # ---- 3. Encode labels ----
    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col])
    num_classes = len(le.classes_)
    logger.info("Encoded %d classes: %s", num_classes, list(le.classes_))

    # ---- 4. Separate features / labels ----
    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int64)

    # ---- 5. Stratified split (Before scaling/feature selection to avoid data leakage) ----
    train_ratio = ds_cfg["train_ratio"]
    val_ratio = ds_cfg["val_ratio"]
    test_ratio = ds_cfg["test_ratio"]

    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), stratify=y, random_state=seed
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=relative_test, stratify=y_temp, random_state=seed
    )

    # ---- 6. Scale (fit ONLY on X_train to prevent data leakage) ----
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    # ---- 7. Feature selection (fit ONLY on X_train, y_train) ----
    num_features_to_select = min(ds_cfg["num_features"], X_train.shape[1])
    mi_scores = mutual_info_classif(X_train, y_train, random_state=seed)
    top_indices = np.argsort(mi_scores)[::-1][:num_features_to_select]
    selected_features = [feature_cols[i] for i in top_indices]

    X_train = X_train[:, top_indices]
    X_val = X_val[:, top_indices]
    X_test = X_test[:, top_indices]

    logger.info(
        "Selected top %d features by mutual information (top-3: %s)",
        num_features_to_select,
        selected_features[:3],
    )

    logger.info(
        "Split sizes - train: %d, val: %d, test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    # ---- 8. Compute inverse-frequency class weights ----
    class_counts = np.bincount(y_train, minlength=num_classes)
    total = len(y_train)
    class_weights = total / (num_classes * class_counts.astype(np.float32) + 1e-8)
    # Normalize so mean weight = 1.0
    class_weights = class_weights / class_weights.mean()

    logger.info("Class weights (normalized): %s", dict(enumerate(class_weights.round(3))))

    # ---- 9. Save processed data ----
    processed_dir = Path(paths_cfg["data_processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(processed_dir / "X_train.npy"), X_train)
    np.save(str(processed_dir / "X_val.npy"), X_val)
    np.save(str(processed_dir / "X_test.npy"), X_test)
    np.save(str(processed_dir / "y_train.npy"), y_train)
    np.save(str(processed_dir / "y_val.npy"), y_val)
    np.save(str(processed_dir / "y_test.npy"), y_test)
    np.save(str(processed_dir / "class_weights.npy"), class_weights)

    logger.info("Saved processed data to %s", processed_dir)

    # ---- 10. Export class distribution ----
    results_dir = Path(paths_cfg["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    dist_df = pd.DataFrame({
        "class_index": range(num_classes),
        "class_name": le.classes_,
        "train_count": np.bincount(y_train, minlength=num_classes),
        "val_count": np.bincount(y_val, minlength=num_classes),
        "test_count": np.bincount(y_test, minlength=num_classes),
        "total_count": np.bincount(y, minlength=num_classes),
        "class_weight": class_weights.round(4),
    })
    dist_df = dist_df.sort_values("total_count", ascending=True).reset_index(drop=True)
    dist_path = results_dir / "class_distribution.csv"
    dist_df.to_csv(str(dist_path), index=False)
    logger.info("Saved class distribution to %s", dist_path)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "label_encoder": le,
        "scaler": scaler,
        "feature_names": selected_features,
        "class_weights": class_weights,
        "num_classes": num_classes,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.utils import load_config, set_seeds, setup_logging, ensure_dirs

    config = load_config("config.yaml")
    setup_logging(
        config["paths"]["log_file"],
        config["logging"]["level"],
    )
    set_seeds(config["seed"])
    ensure_dirs(config)
    pipeline_data = run_pipeline(config)
    print(f"Pipeline complete. {pipeline_data['num_classes']} classes, "
          f"{len(pipeline_data['X_train'])} training samples.")
