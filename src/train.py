"""Training loop for RA-RL-IDS DQN agent.

Supports two reward modes:
  - ``--reward_mode flat``:     Baseline DQN with +1/-1 reward.
  - ``--reward_mode weighted``: Reward-shaped DQN with class-frequency weights.

Usage::

    python -m src.train --reward_mode flat
    python -m src.train --reward_mode weighted
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data_pipeline import run_pipeline
from src.env import IoMTIDSEnv
from src.model import DQNNetwork, ReplayBuffer, epsilon_greedy_action
from src.utils import ensure_dirs, get_device, load_config, set_seeds, setup_logging

logger = logging.getLogger("ra_rl_ids")


def train_dqn(
    config: Dict[str, Any],
    reward_mode: str,
    device: torch.device,
    pipeline_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Train a DQN agent on the IoMT IDS environment.

    Args:
        config: Parsed configuration dictionary.
        reward_mode: ``"flat"`` or ``"weighted"``.
        device: Torch device (CUDA or CPU).
        pipeline_data: Dictionary from ``run_pipeline()`` containing
            splits, class weights, etc.

    Returns:
        Dictionary with training history and trained model.
    """
    train_cfg = config["training"]
    model_cfg = config["model"]
    env_cfg = config["environment"]

    n_features = pipeline_data["X_train"].shape[1]
    n_actions = pipeline_data["num_classes"]
    class_weights = pipeline_data["class_weights"] if reward_mode == "weighted" else None

    logger.info("=" * 60)
    logger.info("Training DQN — reward_mode=%s, device=%s", reward_mode, device)
    logger.info("Features=%d, Classes=%d", n_features, n_actions)
    logger.info("=" * 60)

    # ---- Environment ----
    env = IoMTIDSEnv(
        X=pipeline_data["X_train"],
        y=pipeline_data["y_train"],
        reward_mode=reward_mode,
        class_weights=class_weights,
        penalty_factor=env_cfg["penalty_factor"],
        max_steps=train_cfg["max_steps_per_episode"],
    )

    # ---- Validation environment ----
    val_env = IoMTIDSEnv(
        X=pipeline_data["X_val"],
        y=pipeline_data["y_val"],
        reward_mode="flat",  # evaluate with flat reward for comparable metrics
        max_steps=len(pipeline_data["X_val"]),
    )

    # ---- Networks ----
    policy_net = DQNNetwork(
        n_features=n_features,
        n_actions=n_actions,
        cnn_out_channels=model_cfg["cnn_out_channels"],
        cnn_kernel_size=model_cfg["cnn_kernel_size"],
        lstm_hidden_size=model_cfg["lstm_hidden_size"],
        lstm_num_layers=model_cfg["lstm_num_layers"],
        dropout=model_cfg["dropout"],
        dqn_hidden_size=model_cfg["dqn_hidden_size"],
    ).to(device)

    target_net = DQNNetwork(
        n_features=n_features,
        n_actions=n_actions,
        cnn_out_channels=model_cfg["cnn_out_channels"],
        cnn_kernel_size=model_cfg["cnn_kernel_size"],
        lstm_hidden_size=model_cfg["lstm_hidden_size"],
        lstm_num_layers=model_cfg["lstm_num_layers"],
        dropout=model_cfg["dropout"],
        dqn_hidden_size=model_cfg["dqn_hidden_size"],
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # ---- Optimizer & replay buffer ----
    optimizer = optim.Adam(policy_net.parameters(), lr=train_cfg["learning_rate"])
    replay_buffer = ReplayBuffer(train_cfg["replay_buffer_capacity"])

    # ---- Training hyperparameters ----
    num_episodes = train_cfg["num_episodes"]
    batch_size = train_cfg["batch_size"]
    gamma = train_cfg["gamma"]
    epsilon = train_cfg["epsilon_start"]
    epsilon_end = train_cfg["epsilon_end"]
    epsilon_decay = train_cfg["epsilon_decay"]
    target_update_freq = train_cfg["target_update_frequency"]
    eval_freq = train_cfg["eval_frequency"]

    # ---- History tracking ----
    episode_rewards: List[float] = []
    episode_accuracies: List[float] = []
    val_accuracies: List[float] = []
    losses: List[float] = []

    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        state, info = env.reset(seed=config["seed"] + episode)
        total_reward = 0.0
        correct_count = 0
        step_count = 0

        policy_net.train()
        done = False

        while not done:
            # Select action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            action = epsilon_greedy_action(q_values, epsilon, n_actions)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store experience
            replay_buffer.push(state, action, reward, next_state, done)

            total_reward += reward
            if info.get("correct", False):
                correct_count += 1
            step_count += 1
            state = next_state

            # ---- DQN update ----
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                states_b = torch.FloatTensor(
                    np.array([e.state for e in batch])
                ).to(device)
                actions_b = torch.LongTensor(
                    [e.action for e in batch]
                ).to(device)
                rewards_b = torch.FloatTensor(
                    [e.reward for e in batch]
                ).to(device)
                next_states_b = torch.FloatTensor(
                    np.array([e.next_state for e in batch])
                ).to(device)
                dones_b = torch.BoolTensor(
                    [e.done for e in batch]
                ).to(device)

                # Current Q-values
                current_q = policy_net(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)

                # Target Q-values (no grad)
                with torch.no_grad():
                    next_q = target_net(next_states_b).max(dim=1)[0]
                    next_q[dones_b] = 0.0
                    target_q = rewards_b + gamma * next_q

                # Huber loss (smooth L1)
                loss = nn.SmoothL1Loss()(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()

                losses.append(loss.item())

        # Epsilon decay
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Track metrics
        accuracy = correct_count / max(step_count, 1)
        episode_rewards.append(total_reward)
        episode_accuracies.append(accuracy)

        # Target network update
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Periodic validation
        if episode % eval_freq == 0:
            val_acc = _evaluate_on_env(policy_net, val_env, device, config["seed"])
            val_accuracies.append(val_acc)
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            elapsed = time.time() - start_time
            logger.info(
                "Episode %d/%d | Reward: %.1f | Train Acc: %.3f | "
                "Val Acc: %.3f | Eps: %.3f | Loss: %.4f | Time: %.1fs",
                episode,
                num_episodes,
                total_reward,
                accuracy,
                val_acc,
                epsilon,
                avg_loss,
                elapsed,
            )

    # ---- Save checkpoint ----
    ckpt_dir = Path(config["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if reward_mode == "flat":
        ckpt_path = ckpt_dir / "baseline_dqn.pt"
    else:
        ckpt_path = ckpt_dir / "reward_shaped_dqn.pt"

    torch.save(
        {
            "model_state_dict": policy_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "reward_mode": reward_mode,
            "n_features": n_features,
            "n_actions": n_actions,
            "episode_rewards": episode_rewards,
            "episode_accuracies": episode_accuracies,
            "val_accuracies": val_accuracies,
        },
        str(ckpt_path),
    )
    logger.info("Saved checkpoint: %s", ckpt_path)

    elapsed = time.time() - start_time
    logger.info("Training complete in %.1f seconds", elapsed)

    return {
        "model": policy_net,
        "episode_rewards": episode_rewards,
        "episode_accuracies": episode_accuracies,
        "val_accuracies": val_accuracies,
        "losses": losses,
        "checkpoint_path": str(ckpt_path),
    }


def _evaluate_on_env(
    model: DQNNetwork,
    env: IoMTIDSEnv,
    device: torch.device,
    seed: int,
) -> float:
    """Evaluate model accuracy on an environment (greedy policy).

    Args:
        model: Trained DQN network.
        env: Evaluation environment.
        device: Torch device.
        seed: Random seed for env reset.

    Returns:
        Classification accuracy as a float.
    """
    model.eval()
    state, _ = env.reset(seed=seed)
    correct = 0
    total = 0
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
        action = int(q_values.argmax(dim=-1).item())

        state, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if info.get("correct", False):
            correct += 1
        total += 1

    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train RA-RL-IDS DQN agent")
    parser.add_argument(
        "--reward_mode",
        type=str,
        choices=["flat", "weighted"],
        default=None,
        help="Reward mode: 'flat' for baseline, 'weighted' for reward-shaped",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    reward_mode = args.reward_mode or config["environment"]["reward_mode"]

    setup_logging(
        config["paths"]["log_file"],
        config["logging"]["level"],
    )
    set_seeds(config["seed"])
    ensure_dirs(config)

    device = get_device()

    # Run data pipeline
    logger.info("Running data pipeline...")
    pipeline_data = run_pipeline(config)

    # Train
    result = train_dqn(config, reward_mode, device, pipeline_data)

    logger.info(
        "Training finished. Checkpoint saved to: %s", result["checkpoint_path"]
    )


if __name__ == "__main__":
    main()
