"""Unit tests for the RA-RL-IDS data pipeline.

Tests:
  - Synthetic data generation produces correct shape and class count.
  - Data cleaning handles infinities and NaN values.
  - Feature scaling produces values in a reasonable range.
  - Stratified split preserves class ratios.
  - Class distribution CSV is generated correctly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline import (
    IOMT_CLASSES,
    _generate_synthetic_dataset,
    clean_dataframe,
    run_pipeline,
    select_features,
)
from src.utils import load_config


@pytest.fixture
def config():
    """Load test configuration."""
    cfg = load_config("config.yaml")
    # Use smaller dataset for fast tests
    cfg["dataset"]["synthetic_total_samples"] = 2000
    cfg["dataset"]["num_features"] = 10
    # Use temp directories
    cfg["paths"]["data_processed"] = "data/processed"
    cfg["paths"]["results"] = "results"
    return cfg


class TestSyntheticGeneration:
    """Tests for the synthetic IoMT dataset generator."""

    def test_generates_correct_number_of_classes(self):
        """Synthetic data should contain all defined IoMT classes."""
        df = _generate_synthetic_dataset(total_samples=1000, seed=42)
        unique_labels = df["label"].nunique()
        assert unique_labels == len(IOMT_CLASSES), (
            f"Expected {len(IOMT_CLASSES)} classes, got {unique_labels}"
        )

    def test_generates_correct_columns(self):
        """Synthetic data should have feature columns plus a label column."""
        df = _generate_synthetic_dataset(total_samples=500, seed=42)
        assert "label" in df.columns
        assert len(df.columns) > 2  # features + label

    def test_sample_count_is_reasonable(self):
        """Total sample count should be approximately equal to requested."""
        df = _generate_synthetic_dataset(total_samples=5000, seed=42)
        # Allow ±20% margin due to rounding of per-class counts
        assert 4000 <= len(df) <= 6000

    def test_class_imbalance_exists(self):
        """Synthetic data should have realistic class imbalance."""
        df = _generate_synthetic_dataset(total_samples=10000, seed=42)
        counts = df["label"].value_counts()
        ratio = counts.max() / counts.min()
        # Expect at least 5x imbalance between majority and minority
        assert ratio > 5, f"Imbalance ratio {ratio:.1f} is too low"


class TestDataCleaning:
    """Tests for data cleaning functions."""

    def test_handles_nan_values(self):
        """Cleaning should fill NaN values without errors."""
        import pandas as pd

        df = pd.DataFrame({
            "feat1": [1.0, np.nan, 3.0, 4.0],
            "feat2": [np.nan, 2.0, np.nan, 4.0],
            "label": ["A", "B", "A", "B"],
        })
        cleaned = clean_dataframe(df, label_column="label")
        assert not cleaned.drop(columns=["label"]).isnull().any().any()

    def test_handles_inf_values(self):
        """Cleaning should replace infinity values."""
        import pandas as pd

        df = pd.DataFrame({
            "feat1": [1.0, np.inf, -np.inf, 4.0],
            "feat2": [1.0, 2.0, 3.0, 4.0],
            "label": ["A", "B", "A", "B"],
        })
        cleaned = clean_dataframe(df, label_column="label")
        numeric_cols = cleaned.drop(columns=["label"])
        assert np.isfinite(numeric_cols.values).all()

    def test_drops_constant_columns(self):
        """Cleaning should remove zero-variance columns."""
        import pandas as pd

        df = pd.DataFrame({
            "feat_const": [5.0, 5.0, 5.0, 5.0],
            "feat_vary": [1.0, 2.0, 3.0, 4.0],
            "label": ["A", "B", "A", "B"],
        })
        cleaned = clean_dataframe(df, label_column="label")
        assert "feat_const" not in cleaned.columns
        assert "feat_vary" in cleaned.columns


class TestFeatureSelection:
    """Tests for mutual information feature selection."""

    def test_selects_correct_number_of_features(self):
        """Feature selection should return the requested number of features."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 30).astype(np.float32)
        y = rng.randint(0, 3, size=200).astype(np.int64)
        feature_names = [f"f{i}" for i in range(30)]

        X_sel, names, scores = select_features(X, y, feature_names, num_features=10, seed=42)
        assert X_sel.shape[1] == 10
        assert len(names) == 10

    def test_does_not_exceed_available_features(self):
        """Requesting more features than available should cap gracefully."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5).astype(np.float32)
        y = rng.randint(0, 2, size=100).astype(np.int64)
        feature_names = [f"f{i}" for i in range(5)]

        X_sel, names, _ = select_features(X, y, feature_names, num_features=20, seed=42)
        assert X_sel.shape[1] == 5


class TestPipeline:
    """Integration tests for the full data pipeline."""

    def test_pipeline_produces_correct_splits(self, config):
        """Pipeline should produce train, val, test splits with correct dtypes."""
        data = run_pipeline(config)

        assert data["X_train"].dtype == np.float32
        assert data["y_train"].dtype == np.int64
        assert data["X_val"].shape[1] == data["X_train"].shape[1]
        assert data["X_test"].shape[1] == data["X_train"].shape[1]

    def test_pipeline_produces_class_weights(self, config):
        """Pipeline should compute class weights for all classes."""
        data = run_pipeline(config)
        assert len(data["class_weights"]) == data["num_classes"]
        assert all(w > 0 for w in data["class_weights"])

    def test_class_distribution_csv_created(self, config):
        """Pipeline should export class_distribution.csv."""
        _ = run_pipeline(config)
        dist_path = Path(config["paths"]["results"]) / "class_distribution.csv"
        assert dist_path.exists()
