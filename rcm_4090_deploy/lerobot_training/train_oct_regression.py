"""Deterministic state + OCT action baseline without image observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from train_diffusion_policy import (
    DEFAULT_EPOCHS,
    RCMDataset,
    configure_stream_logging,
    image_transform_for_dataset,
)


class OCTStateRegressionPolicy(nn.Module):
    """Predict the teacher delta action from current state and OCT error."""

    def __init__(
        self,
        hidden_dim=128,
        history_mode="none",
        state_mean=None,
        state_std=None,
        oct_mean=None,
        oct_std=None,
        velocity_mean=None,
        velocity_std=None,
        previous_action_mean=None,
        previous_action_std=None,
        action_mean=None,
        action_std=None,
        history_noise_std=0.0,
        history_dropout=0.0,
    ):
        super().__init__()
        if history_mode not in ("none", "velocity", "velocity_previous_action"):
            raise ValueError(f"Unknown history_mode: {history_mode}")
        self.history_mode = history_mode
        self.history_noise_std = float(history_noise_std)
        self.history_dropout = float(history_dropout)
        if self.history_noise_std < 0 or not 0 <= self.history_dropout <= 1:
            raise ValueError("Invalid history noise/dropout settings")
        for name, value, fallback in (
            ('state_mean', state_mean, [0.0] * 3),
            ('state_std', state_std, [1.0] * 3),
            ('oct_mean', oct_mean, [0.0] * 3),
            ('oct_std', oct_std, [1.0] * 3),
            ('velocity_mean', velocity_mean, [0.0] * 3),
            ('velocity_std', velocity_std, [1.0] * 3),
            ('previous_action_mean', previous_action_mean, [0.0] * 3),
            ('previous_action_std', previous_action_std, [1.0] * 3),
            ('action_mean', action_mean, [0.0] * 3),
            ('action_std', action_std, [1.0] * 3),
        ):
            self.register_buffer(
                name,
                torch.as_tensor(fallback if value is None else value, dtype=torch.float32),
                persistent=False,
            )
        input_dim = 7
        if history_mode == "velocity":
            input_dim += 3
        elif history_mode == "velocity_previous_action":
            input_dim += 6
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, state, oct_navigation, velocity=None, previous_action=None):
        valid = oct_navigation[..., 3:4]
        oct_xyz = (
            oct_navigation[..., :3] - self.oct_mean.reshape(1, -1)
        ) / self.oct_std.reshape(1, -1)
        normalized_state = (
            state - self.state_mean.reshape(1, -1)
        ) / self.state_std.reshape(1, -1)
        features = [normalized_state, oct_xyz, valid]
        if self.history_mode != "none":
            if velocity is None:
                raise ValueError("velocity is required by the configured history mode")
            normalized_velocity = (
                velocity - self.velocity_mean.reshape(1, -1)
            ) / self.velocity_std.reshape(1, -1)
            if self.training:
                normalized_velocity = (
                    normalized_velocity
                    + torch.randn_like(normalized_velocity) * self.history_noise_std
                )
            features.append(normalized_velocity)
        if self.history_mode == "velocity_previous_action":
            if previous_action is None:
                raise ValueError(
                    "previous_action is required by the configured history mode"
                )
            normalized_previous_action = (
                previous_action - self.previous_action_mean.reshape(1, -1)
            ) / self.previous_action_std.reshape(1, -1)
            if self.training and self.history_noise_std:
                normalized_previous_action = (
                    normalized_previous_action
                    + torch.randn_like(normalized_previous_action)
                    * self.history_noise_std
                )
            if self.training and self.history_dropout:
                keep = (
                    torch.rand(
                        (normalized_previous_action.shape[0], 1),
                        device=normalized_previous_action.device,
                    )
                    >= self.history_dropout
                ).to(normalized_previous_action.dtype)
                normalized_previous_action = normalized_previous_action * keep
            features.append(normalized_previous_action)
        prediction = self.network(torch.cat(features, dim=-1))
        return prediction * self.action_std.reshape(1, -1) + self.action_mean.reshape(1, -1)


def _features(dataset, history_mode="none", control_period_s=0.05):
    rows = np.asarray(dataset.valid_indices, dtype=np.int64)
    state = dataset.states[rows].astype(np.float32)
    oct_navigation = np.concatenate([
        dataset.oct_navigation[rows].astype(np.float32),
        dataset.oct_navigation_valid[rows].astype(np.float32)[:, None],
    ], axis=1)
    has_previous = (
        (rows > 0)
        & (dataset.episode_indices[rows - 1] == dataset.episode_indices[rows])
    )
    previous_state = state.copy()
    previous_state[has_previous] = dataset.states[
        rows[has_previous] - 1
    ].astype(np.float32)
    velocity = (state - previous_state) / float(control_period_s)
    previous_action = np.zeros_like(state, dtype=np.float32)
    previous_action[has_previous] = dataset.actions[
        rows[has_previous] - 1
    ].astype(np.float32)
    actions = dataset.actions[rows].astype(np.float32)
    return state, oct_navigation, velocity, previous_action, actions


def _evaluate(model, loader, device):
    model.eval()
    predicted, target = [], []
    with torch.no_grad():
        for state, oct_navigation, velocity, previous_action, actions in loader:
            output = model(
                state.to(device, non_blocking=True),
                oct_navigation.to(device, non_blocking=True),
                velocity=velocity.to(device, non_blocking=True),
                previous_action=previous_action.to(device, non_blocking=True),
            )
            predicted.append(output.cpu().numpy())
            target.append(actions.numpy())
    error = np.concatenate(predicted) - np.concatenate(target)
    mae = np.abs(error).mean(axis=0)
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    return {
        'mae_deg_deg_mm': mae.tolist(),
        'rmse_deg_deg_mm': rmse.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train state + OCT MSE baseline")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--val-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument(
        "--history-mode",
        choices=("none", "velocity", "velocity_previous_action"),
        default="velocity_previous_action",
    )
    parser.add_argument("--history-noise-std", type=float, default=0.05)
    parser.add_argument("--history-dropout", type=float, default=0.1)
    parser.add_argument("--control-period-s", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--min-delta-mae", type=float, default=1e-8)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_stream_logging(output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(f"使用设备: {device}", flush=True)

    train_dataset = RCMDataset(
        Path(args.data_dir),
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
    )
    val_dataset = RCMDataset(
        Path(args.val_data_dir),
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
    )
    train_state, train_oct, train_velocity, train_previous_action, train_action = _features(
        train_dataset,
        args.history_mode,
        args.control_period_s,
    )
    val_state, val_oct, val_velocity, val_previous_action, val_action = _features(
        val_dataset,
        args.history_mode,
        args.control_period_s,
    )

    valid = train_oct[:, 3] > 0.5
    state_mean = train_state[valid].mean(axis=0)
    state_std = np.maximum(train_state[valid].std(axis=0), 1e-6)
    oct_mean = train_oct[valid, :3].mean(axis=0)
    oct_std = np.maximum(train_oct[valid, :3].std(axis=0), 1e-6)
    action_mean = train_action[valid].mean(axis=0)
    action_std = np.maximum(train_action[valid].std(axis=0), 1e-6)
    velocity_mean = train_velocity[valid].mean(axis=0)
    velocity_std = np.maximum(train_velocity[valid].std(axis=0), 1e-6)
    previous_action_mean = train_previous_action[valid].mean(axis=0)
    previous_action_std = np.maximum(train_previous_action[valid].std(axis=0), 1e-6)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_state),
            torch.from_numpy(train_oct),
            torch.from_numpy(train_velocity),
            torch.from_numpy(train_previous_action),
            torch.from_numpy(train_action),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(val_state),
            torch.from_numpy(val_oct),
            torch.from_numpy(val_velocity),
            torch.from_numpy(val_previous_action),
            torch.from_numpy(val_action),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    model = OCTStateRegressionPolicy(
        state_mean=state_mean,
        state_std=state_std,
        oct_mean=oct_mean,
        oct_std=oct_std,
        history_mode=args.history_mode,
        velocity_mean=velocity_mean,
        velocity_std=velocity_std,
        previous_action_mean=previous_action_mean,
        previous_action_std=previous_action_std,
        action_mean=action_mean,
        action_std=action_std,
        history_noise_std=args.history_noise_std,
        history_dropout=args.history_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    print(
        f"训练样本={len(train_action)}, 验证样本={len(val_action)}, "
        f"模型参数={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )

    history = []
    best_mae = float("inf")
    best_epoch = -1
    stale_epochs = 0
    history_path = output_dir / "training_history.json"
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for state, oct_navigation, velocity, previous_action, actions in tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
        ):
            state = state.to(device, non_blocking=True)
            oct_navigation = oct_navigation.to(device, non_blocking=True)
            velocity = velocity.to(device, non_blocking=True)
            previous_action = previous_action.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            prediction = model(
                state,
                oct_navigation,
                velocity=velocity,
                previous_action=previous_action,
            )
            normalized_prediction = (
                prediction - model.action_mean.reshape(1, -1)
            ) / model.action_std.reshape(1, -1)
            normalized_target = (
                actions - model.action_mean.reshape(1, -1)
            ) / model.action_std.reshape(1, -1)
            loss = torch.mean((normalized_prediction - normalized_target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss) * len(actions)
            sample_count += len(actions)
        scheduler.step()
        train_loss = total_loss / sample_count
        metrics = _evaluate(model, val_loader, device)
        normalized_mae = float(np.mean(
            np.asarray(metrics['mae_deg_deg_mm']) / action_std
        ))
        improved = normalized_mae < best_mae - args.min_delta_mae
        if improved:
            best_mae = normalized_mae
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                'checkpoint_schema': 'state_oct_mse_regression_v1',
                'policy_type': 'state_oct_mse_regression',
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_normalized_mae': best_mae,
                'config': {
                    'state_mean': state_mean.tolist(),
                    'state_std': state_std.tolist(),
                    'oct_mean': oct_mean.tolist(),
                    'oct_std': oct_std.tolist(),
                    'history_mode': args.history_mode,
                    'velocity_mean': velocity_mean.tolist(),
                    'velocity_std': velocity_std.tolist(),
                    'previous_action_mean': previous_action_mean.tolist(),
                    'previous_action_std': previous_action_std.tolist(),
                    'history_noise_std': args.history_noise_std,
                    'history_dropout': args.history_dropout,
                    'action_mean': action_mean.tolist(),
                    'action_std': action_std.tolist(),
                },
            }, output_dir / "best_model.pth")
        else:
            stale_epochs += 1
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_mae': metrics['mae_deg_deg_mm'],
            'val_rmse': metrics['rmse_deg_deg_mm'],
            'normalized_mae': normalized_mae,
            'best_normalized_mae': best_mae,
            'improved': improved,
        })
        history_tmp = history_path.with_name(history_path.name + ".tmp")
        with open(history_tmp, "w", encoding="utf-8") as stream:
            json.dump(history, stream, ensure_ascii=False, indent=2)
        history_tmp.replace(history_path)
        print(
            f"Epoch {epoch}: train_mse={train_loss:.8f} "
            f"val_mae_A/B/C={metrics['mae_deg_deg_mm']} "
            f"normalized_mae={normalized_mae:.8f}",
            flush=True,
        )
        if stale_epochs >= args.early_stop_patience:
            print("早停: 验证动作 MAE 未继续改善", flush=True)
            break

    print(f"训练完成! 最佳 epoch={best_epoch}, normalized_mae={best_mae:.8f}")


if __name__ == "__main__":
    main()
