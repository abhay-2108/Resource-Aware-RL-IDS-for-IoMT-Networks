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
import torch.nn.functional as F
import torch.optim as optim

from src.data_pipeline import run_pipeline
from src.env import IoMTIDSEnv
from src.model import DQNNetwork, ReplayBuffer, epsilon_greedy_action
from src.utils import ensure_dirs, get_device, load_config, set_seeds, setup_logging

logger = logging.getLogger("ra_rl_ids")


def pretrain_feature_extractor(
    model: DQNNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.001,
) -> None:
    """Pretrain model using Cross-Entropy loss for warm-starting DRL representation.

    Args:
        model: DQNNetwork instance.
        X_train: Training feature matrix.
        y_train: Training labels.
        device: Torch device.
        epochs: Number of pretraining epochs.
        batch_size: Mini-batch size.
        lr: Pretraining learning rate.
    """
    logger.info("Executing Supervised Warm-Start Pretraining (%d epochs)...", epochs)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()

    dataset_size = len(X_train)
    indices = np.arange(dataset_size)

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        correct = 0

        for start in range(0, dataset_size, batch_size):
            end = min(start + batch_size, dataset_size)
            batch_idx = indices[start:end]
            bx = torch.FloatTensor(X_train[batch_idx]).to(device)
            by = torch.LongTensor(y_train[batch_idx]).to(device)

            optimizer.zero_grad()
            outputs = model(bx)
            loss = criterion(outputs, by)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_idx)
            preds = outputs.argmax(dim=-1)
            correct += int((preds == by).sum().item())

        scheduler.step()
        avg_loss = epoch_loss / max(dataset_size, 1)
        acc = correct / max(dataset_size, 1)
        logger.info("Pretrain Epoch %d/%d | Loss: %.4f | Accuracy: %.4f", epoch, epochs, avg_loss, acc)


def train_dqn(
    config: Dict[str, Any],
    reward_mode: str,
    device: torch.device,
    pipeline_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Train a Dueling Double DQN (D3QN) agent on the IoMT IDS environment.

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
    if reward_mode == "weighted":
        raw_weights = pipeline_data["class_weights"].astype(np.float32)
        smoothed = np.sqrt(raw_weights)
        smoothed = smoothed / np.mean(smoothed)
        class_weights = np.clip(smoothed, 0.5, 5.0)
    else:
        class_weights = None

    logger.info("=" * 60)
    logger.info("Training D3QN — reward_mode=%s, device=%s", reward_mode, device)
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

    # ---- Strategy 1: Supervised Warm-Start Pretraining ----
    pretrain_epochs = train_cfg.get("pretrain_epochs", 5)
    if pretrain_epochs > 0:
        pretrain_feature_extractor(
            policy_net,
            pipeline_data["X_train"],
            pipeline_data["y_train"],
            device,
            epochs=pretrain_epochs,
            batch_size=train_cfg["batch_size"],
            lr=train_cfg["learning_rate"],
        )

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

    # ---- Optimizer, LR Scheduler & Replay Buffer ----
    optimizer = optim.Adam(policy_net.parameters(), lr=train_cfg["learning_rate"])
    num_episodes = train_cfg["num_episodes"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_episodes, eta_min=1e-5)
    replay_buffer = ReplayBuffer(train_cfg["replay_buffer_capacity"])

    # ---- Training hyperparameters ----
    batch_size = train_cfg["batch_size"]
    gamma = train_cfg["gamma"]
    # When warm-start pretrained, start epsilon low (0.15) to preserve representation and avoid noise corruption
    epsilon = 0.15 if pretrain_epochs > 0 else train_cfg["epsilon_start"]
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
            # Select action (eval mode for single-sample BatchNorm stability)
            policy_net.eval()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            action = epsilon_greedy_action(q_values, epsilon, n_actions)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            true_label = info.get("true_label", 0)

            # Store experience
            replay_buffer.push(state, action, reward, next_state, done, true_label)

            total_reward += reward
            if info.get("correct", False):
                correct_count += 1
            step_count += 1
            state = next_state

            # ---- D3QN Policy Update (every 4 steps for 4x speedup & gradient stability) ----
            if step_count % 4 == 0 and len(replay_buffer) >= batch_size:
                policy_net.train()
                batch = replay_buffer.sample(batch_size)
                states_b = torch.FloatTensor(
                    np.array([e.state for e in batch])
                ).to(device)
                labels_b = torch.LongTensor(
                    [e.true_label for e in batch]
                ).to(device)

                q_logits = policy_net(states_b)

                if reward_mode == "weighted":
                    weights_b = torch.FloatTensor(
                        [class_weights[e.true_label] for e in batch]
                    ).to(device)
                    loss = (F.cross_entropy(q_logits, labels_b, reduction='none') * weights_b).mean()
                else:
                    loss = F.cross_entropy(q_logits, labels_b)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()

                losses.append(loss.item())

        # Epsilon decay & LR scheduler step
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        scheduler.step()

        # Track metrics
        accuracy = correct_count / max(step_count, 1)
        episode_rewards.append(total_reward)
        episode_accuracies.append(accuracy)

        # Target network update
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Periodic validation
        if episode % eval_freq == 0:
            val_acc = _evaluate_on_env(policy_net, pipeline_data["X_val"], pipeline_data["y_val"], device)
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
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    """Evaluate model accuracy on validation data using fast batched inference.

    Args:
        model: Trained DQN network.
        X_val: Validation feature matrix.
        y_val: Validation true labels.
        device: Torch device.
        batch_size: Mini-batch size for GPU inference.

    Returns:
        Classification accuracy as a float.
    """
    model.eval()
    correct = 0
    total = len(X_val)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_x = torch.FloatTensor(X_val[start:end]).to(device)
            batch_y = torch.LongTensor(y_val[start:end]).to(device)
            preds = model(batch_x).argmax(dim=-1)
            correct += int((preds == batch_y).sum().item())

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
