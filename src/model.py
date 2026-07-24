"""Neural network architectures for RA-RL-IDS.

Provides:
  - ``CNNLSTMFeatureExtractor``: 1-D CNN → BatchNorm → ReLU → LSTM → dense
    embedding — captures spatial (feature-level) and temporal (flow-sequence)
    patterns in IoMT network traffic.
  - ``DQNNetwork``: Full Deep Q-Network wrapping the feature extractor with
    a Q-value head over the discrete action space.
  - ``ReplayBuffer``: Fixed-capacity experience-replay buffer used by the
    DQN training loop.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Experience tuple for replay buffer
# ---------------------------------------------------------------------------
class Experience(NamedTuple):
    """Single transition stored in the replay buffer."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer for DQN experience replay.

    Args:
        capacity: Maximum number of transitions to store.
    """

    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Experience] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition."""
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a random mini-batch of transitions."""
        return random.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# CNN-LSTM Feature Extractor
# ---------------------------------------------------------------------------
class CNNLSTMFeatureExtractor(nn.Module):
    """1-D CNN + LSTM feature extractor for IoMT network flow data.

    The input feature vector is reshaped into a short 1-D "sequence" so
    that the CNN can learn local correlations among adjacent features, and
    the LSTM captures sequential dependencies.

    Architecture:
        Input (n_features,) → reshape to (1, n_features) →
        Conv1d → BatchNorm1d → ReLU → Dropout →
        permute → LSTM → take last hidden state →
        Linear → ReLU → embedding

    Args:
        n_features: Number of input features.
        cnn_out_channels: Number of CNN output channels.
        cnn_kernel_size: CNN kernel size (along feature dim).
        lstm_hidden_size: LSTM hidden state dimensionality.
        lstm_num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability.
        embedding_dim: Output embedding dimensionality.
    """

    def __init__(
        self,
        n_features: int,
        cnn_out_channels: int = 64,
        cnn_kernel_size: int = 3,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
        dropout: float = 0.2,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()

        self.n_features = n_features

        # 1-D CNN block
        # Input shape: (batch, 1, n_features)
        padding = cnn_kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=cnn_out_channels,
                kernel_size=cnn_kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # LSTM block
        # Input shape: (batch, seq_len=n_features, input_size=cnn_out_channels)
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(lstm_hidden_size, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, n_features)``.

        Returns:
            Embedding tensor of shape ``(batch, embedding_dim)``.
        """
        # (batch, n_features) → (batch, 1, n_features)
        x = x.unsqueeze(1)

        # CNN: (batch, 1, n_features) → (batch, cnn_out_channels, n_features)
        x = self.cnn(x)

        # Permute for LSTM: (batch, n_features, cnn_out_channels)
        x = x.permute(0, 2, 1)

        # LSTM: take the last hidden state
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, lstm_hidden_size)
        x = lstm_out[:, -1, :]  # last time step

        # Project
        x = self.projection(x)
        return x


# ---------------------------------------------------------------------------
# DQN Network
# ---------------------------------------------------------------------------
class DQNNetwork(nn.Module):
    """Deep Q-Network for IoMT intrusion detection.

    Combines the CNN-LSTM feature extractor with a Q-value head that
    outputs one Q-value per discrete action (class).

    Args:
        n_features: Number of input features.
        n_actions: Number of discrete actions (classes).
        cnn_out_channels: CNN output channels.
        cnn_kernel_size: CNN kernel size.
        lstm_hidden_size: LSTM hidden size.
        lstm_num_layers: Number of LSTM layers.
        dropout: Dropout probability.
        dqn_hidden_size: Hidden layer size in the Q-head.
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        cnn_out_channels: int = 64,
        cnn_kernel_size: int = 3,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
        dropout: float = 0.2,
        dqn_hidden_size: int = 128,
    ) -> None:
        super().__init__()

        self.feature_extractor = CNNLSTMFeatureExtractor(
            n_features=n_features,
            cnn_out_channels=cnn_out_channels,
            cnn_kernel_size=cnn_kernel_size,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            dropout=dropout,
            embedding_dim=dqn_hidden_size,
        )

        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(dqn_hidden_size, dqn_hidden_size),
            nn.ReLU(),
            nn.Linear(dqn_hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions.

        Args:
            x: Input features ``(batch, n_features)``.

        Returns:
            Q-values ``(batch, n_actions)``.
        """
        embedding = self.feature_extractor(x)
        q_values = self.q_head(embedding)
        return q_values


# ---------------------------------------------------------------------------
# Epsilon-greedy action selection
# ---------------------------------------------------------------------------
def epsilon_greedy_action(
    q_values: torch.Tensor,
    epsilon: float,
    n_actions: int,
) -> int:
    """Select an action using epsilon-greedy policy.

    Args:
        q_values: Q-values for the current state ``(1, n_actions)`` or
            ``(n_actions,)``.
        epsilon: Current exploration probability.
        n_actions: Total number of actions.

    Returns:
        Selected action index.
    """
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        return int(q_values.argmax(dim=-1).item())
