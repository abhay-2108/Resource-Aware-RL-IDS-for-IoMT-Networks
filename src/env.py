"""Custom Gymnasium environment for RA-RL-IDS.

Wraps the IoMT intrusion detection classification task as a sequential
Reinforcement Learning problem:

  - **State**: One network flow's feature vector (scaled, selected features).
  - **Action**: Predicted class index ∈ {0, 1, ..., num_classes - 1}.
  - **Reward**: Configurable — flat (+1 / -1) or inverse-frequency weighted.
  - **Episode**: Sequential presentation of all samples in a data partition.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger("ra_rl_ids")


class IoMTIDSEnv(gym.Env):
    """Gymnasium environment for IoMT intrusion detection via RL.

    Each ``step()`` presents one network flow to the agent; the agent must
    classify it.  The reward depends on the configured reward mode.

    Args:
        X: Feature matrix ``(n_samples, n_features)``, float32.
        y: Label array ``(n_samples,)``, int64.
        reward_mode: ``"flat"`` for +1/-1, ``"weighted"`` for class-frequency
            weighted rewards.
        class_weights: Array of per-class inverse-frequency weights (required
            when ``reward_mode="weighted"``).
        penalty_factor: Multiplier applied to the negative reward for
            misclassifications in weighted mode.
        max_steps: Maximum steps per episode (``None`` = use all samples).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        reward_mode: str = "flat",
        class_weights: Optional[np.ndarray] = None,
        penalty_factor: float = 2.0,
        max_steps: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.n_samples = len(self.X)
        self.n_features = self.X.shape[1]
        self.num_classes = int(self.y.max()) + 1

        # Reward configuration
        self.reward_mode = reward_mode
        self.penalty_factor = penalty_factor

        if reward_mode == "weighted":
            if class_weights is None:
                raise ValueError(
                    "class_weights must be provided for weighted reward mode"
                )
            self.class_weights = class_weights.astype(np.float32)
        else:
            self.class_weights = np.ones(self.num_classes, dtype=np.float32)

        # Episode configuration
        self.max_steps = max_steps or self.n_samples

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.num_classes)

        # Internal state
        self._indices: np.ndarray = np.arange(self.n_samples)
        self._current_step: int = 0
        self._current_idx: int = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode.

        Shuffles the sample presentation order and returns the first
        observation.

        Returns:
            Tuple of (first observation, info dict).
        """
        super().reset(seed=seed)

        # Shuffle presentation order
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        self._indices = rng.permutation(self.n_samples)

        self._current_step = 0
        self._current_idx = int(self._indices[0])

        obs = self.X[self._current_idx].copy()
        info = {"true_label": int(self.y[self._current_idx])}
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take an action (classify the current flow) and advance.

        Args:
            action: Predicted class index.

        Returns:
            Tuple of (next_obs, reward, terminated, truncated, info).
        """
        true_label = int(self.y[self._current_idx])
        correct = int(action) == true_label

        # Compute reward
        reward = self._compute_reward(correct, true_label)

        # Advance
        self._current_step += 1
        terminated = self._current_step >= self.max_steps
        truncated = self._current_step >= self.n_samples

        # Next observation
        if not (terminated or truncated):
            self._current_idx = int(
                self._indices[self._current_step % self.n_samples]
            )
            next_obs = self.X[self._current_idx].copy()
        else:
            next_obs = np.zeros(self.n_features, dtype=np.float32)

        info = {
            "true_label": true_label,
            "predicted_label": int(action),
            "correct": correct,
            "step": self._current_step,
        }

        return next_obs, reward, terminated, truncated, info

    def _compute_reward(self, correct: bool, true_label: int) -> float:
        """Compute the reward for the current classification decision.

        Flat mode:
            +1.0 for correct, -1.0 for incorrect.

        Weighted mode (per spec §7.3):
            Correct:   +w[true_label]
            Incorrect: -w[true_label] * penalty_factor

        This incentivises the agent to correctly classify rare attack
        classes (which carry higher inverse-frequency weights), directly
        addressing Gap B (class-imbalance-aware reward shaping).

        Args:
            correct: Whether the prediction matched the true label.
            true_label: Ground truth class index.

        Returns:
            Scalar reward value.
        """
        if self.reward_mode == "flat":
            return 1.0 if correct else -1.0

        # Weighted mode
        w = float(self.class_weights[true_label])
        if correct:
            return w
        else:
            return -w * self.penalty_factor
