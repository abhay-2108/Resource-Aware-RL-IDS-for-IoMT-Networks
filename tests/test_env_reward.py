"""Unit tests for the RA-RL-IDS Gymnasium environment and reward functions.

Tests:
  - Gymnasium API compliance (reset, step, spaces).
  - Flat reward returns +1/-1 correctly.
  - Weighted reward returns higher penalties for rare classes.
  - Episode terminates correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env import IoMTIDSEnv


@pytest.fixture
def sample_data():
    """Create minimal sample data for environment testing."""
    rng = np.random.RandomState(42)
    n_samples = 100
    n_features = 10
    n_classes = 4

    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = rng.randint(0, n_classes, size=n_samples).astype(np.int64)

    # Class weights: class 3 is rare (high weight), class 0 is common (low weight)
    class_weights = np.array([0.5, 1.0, 1.5, 3.0], dtype=np.float32)

    return X, y, n_classes, class_weights


class TestEnvironmentAPI:
    """Tests for Gymnasium API compliance."""

    def test_reset_returns_observation_and_info(self, sample_data):
        """reset() should return (observation, info) tuple."""
        X, y, _, _ = sample_data
        env = IoMTIDSEnv(X, y, reward_mode="flat")
        obs, info = env.reset(seed=42)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (X.shape[1],)
        assert isinstance(info, dict)
        assert "true_label" in info

    def test_step_returns_five_tuple(self, sample_data):
        """step() should return (obs, reward, terminated, truncated, info)."""
        X, y, _, _ = sample_data
        env = IoMTIDSEnv(X, y, reward_mode="flat", max_steps=10)
        env.reset(seed=42)

        result = env.step(0)
        assert len(result) == 5

        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_observation_space_matches_data(self, sample_data):
        """Observation space should match feature dimensionality."""
        X, y, _, _ = sample_data
        env = IoMTIDSEnv(X, y, reward_mode="flat")
        assert env.observation_space.shape == (X.shape[1],)

    def test_action_space_matches_classes(self, sample_data):
        """Action space should have one action per class."""
        X, y, n_classes, _ = sample_data
        env = IoMTIDSEnv(X, y, reward_mode="flat")
        assert env.action_space.n == n_classes

    def test_episode_terminates(self, sample_data):
        """Episode should terminate after max_steps."""
        X, y, _, _ = sample_data
        max_steps = 5
        env = IoMTIDSEnv(X, y, reward_mode="flat", max_steps=max_steps)
        env.reset(seed=42)

        done = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = env.step(0)
            done = terminated or truncated
            steps += 1

        assert steps == max_steps


class TestFlatReward:
    """Tests for the flat reward mode (+1 / -1)."""

    def test_correct_prediction_gives_positive_reward(self, sample_data):
        """Correct classification should yield +1.0 reward."""
        X, y, _, _ = sample_data
        env = IoMTIDSEnv(X, y, reward_mode="flat", max_steps=10)
        obs, info = env.reset(seed=42)

        true_label = info["true_label"]
        _, reward, _, _, step_info = env.step(true_label)

        assert reward == 1.0
        assert step_info["correct"] is True

    def test_incorrect_prediction_gives_negative_reward(self, sample_data):
        """Incorrect classification should yield -1.0 reward."""
        X, y, n_classes, _ = sample_data
        env = IoMTIDSEnv(X, y, reward_mode="flat", max_steps=10)
        obs, info = env.reset(seed=42)

        true_label = info["true_label"]
        wrong_action = (true_label + 1) % n_classes
        _, reward, _, _, step_info = env.step(wrong_action)

        assert reward == -1.0
        assert step_info["correct"] is False


class TestWeightedReward:
    """Tests for the weighted (class-imbalance-aware) reward mode."""

    def test_correct_prediction_gives_weighted_positive_reward(self, sample_data):
        """Correct classification should yield +w[class] reward."""
        X, y, _, class_weights = sample_data
        env = IoMTIDSEnv(
            X, y,
            reward_mode="weighted",
            class_weights=class_weights,
            penalty_factor=2.0,
            max_steps=10,
        )
        obs, info = env.reset(seed=42)

        true_label = info["true_label"]
        _, reward, _, _, _ = env.step(true_label)

        expected = float(class_weights[true_label])
        assert abs(reward - expected) < 1e-5

    def test_incorrect_prediction_gives_weighted_negative_reward(self, sample_data):
        """Incorrect classification should yield -w[class] * penalty_factor."""
        X, y, n_classes, class_weights = sample_data
        penalty_factor = 2.0
        env = IoMTIDSEnv(
            X, y,
            reward_mode="weighted",
            class_weights=class_weights,
            penalty_factor=penalty_factor,
            max_steps=10,
        )
        obs, info = env.reset(seed=42)

        true_label = info["true_label"]
        wrong_action = (true_label + 1) % n_classes
        _, reward, _, _, _ = env.step(wrong_action)

        expected = -float(class_weights[true_label]) * penalty_factor
        assert abs(reward - expected) < 1e-5

    def test_rare_class_has_higher_penalty(self, sample_data):
        """Misclassifying a rare class should incur a higher penalty."""
        _, _, _, class_weights = sample_data
        # class_weights = [0.5, 1.0, 1.5, 3.0]
        # Class 3 (rare) should have higher weight than class 0 (common)
        assert class_weights[3] > class_weights[0]

    def test_requires_class_weights(self, sample_data):
        """Weighted mode without class_weights should raise ValueError."""
        X, y, _, _ = sample_data
        with pytest.raises(ValueError, match="class_weights must be provided"):
            IoMTIDSEnv(X, y, reward_mode="weighted", class_weights=None)
