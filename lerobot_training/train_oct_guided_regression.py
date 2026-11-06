"""Cheap state + OCT regression with direct proportional OCT guidance.

The OCT target-minus-tip vector no longer enters a generic network as a
globally z-scored vector.  A damped RCM Jacobian first converts it into a
proportional task-space action.  A small network learns only a bounded residual
from the OCT direction and log-distance.  This file deliberately omits images
and diffusion so the OCT interface can be validated quickly.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rcm_reference import Reference
try:
    from train_diffusion_policy import RCMDataset, configure_stream_logging
except ModuleNotFoundError:  # Imported as ``lerobot_training.*`` in tests.
    from .train_diffusion_policy import RCMDataset, configure_stream_logging


TASK_TO_SI = np.array(
    [math.pi / 180.0, math.pi / 180.0, 0.001],
    dtype=np.float64,
)
SI_TO_ACTION = np.array(
    [180.0 / math.pi, 180.0 / math.pi, 1000.0],
    dtype=np.float64,
)
CAMERA_SIGN = np.array([1.0, -1.0, -1.0], dtype=np.float64)
DEFAULT_DISTANCE_EDGES_MM = (0.0, 0.02, 0.1, 0.5, 1.0, 5.0, 10.0, np.inf)


def camera_error_to_world(camera_error_mm, camera_rotation):
    """Convert the OCT camera-optical delta back to world millimetres."""
    camera_error = np.asarray(camera_error_mm, dtype=np.float64)
    rotation = np.asarray(camera_rotation, dtype=np.float64)
    if (
        camera_error.shape != (3,)
        or rotation.shape != (3, 3)
        or not np.all(np.isfinite(camera_error))
        or not np.all(np.isfinite(rotation))
    ):
        raise ValueError("camera error and rotation must be finite 3/3x3 arrays")
    return rotation @ (camera_error / CAMERA_SIGN)


def task_jacobian(
    reference,
    q_si,
    finite_difference_rad=(1e-4, 1e-4),
    finite_difference_m=1e-5,
):
    """Numerical needle-tip Jacobian in world millimetres per SI task unit."""
    q_si = np.asarray(q_si, dtype=np.float64)
    if q_si.shape != (3,) or not np.all(np.isfinite(q_si)):
        raise ValueError("q_si must be a finite A/B/C vector in rad/rad/m")
    steps = np.array(
        [
            finite_difference_rad[0],
            finite_difference_rad[1],
            finite_difference_m,
        ],
        dtype=np.float64,
    )
    jacobian = np.empty((3, 3), dtype=np.float64)
    for axis, step in enumerate(steps):
        points = []
        for sign in (1.0, -1.0):
            trial = q_si.copy()
            trial[axis] += sign * step
            tip_robot_mm = reference.forward_kinematics(trial)
            points.append(reference.robot_to_world(tip_robot_mm))
        jacobian[:, axis] = (points[0] - points[1]) / (2.0 * step)
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("non-finite task Jacobian")
    return jacobian


def damped_pseudoinverse_solve(jacobian, error_world_mm, relative_damping=1e-3):
    """Return the least-squares task increment for a small world-space error."""
    matrix = np.asarray(jacobian, dtype=np.float64)
    error = np.asarray(error_world_mm, dtype=np.float64)
    if matrix.shape != (3, 3) or error.shape != (3,):
        raise ValueError("jacobian and error must have shapes (3, 3) and (3,)")
    try:
        u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("task Jacobian SVD failed") from exc
    if (
        not len(singular)
        or not np.all(np.isfinite(singular))
        or singular[0] <= 1e-12
    ):
        raise ValueError("singular task Jacobian")
    if relative_damping < 0 or not math.isfinite(relative_damping):
        raise ValueError("relative_damping must be finite and nonnegative")
    damping = max(singular[0] * float(relative_damping), 1e-12)
    return vt.T @ (
        singular / (singular ** 2 + damping ** 2) * (u.T @ error)
    )


def proportional_guided_action(
    reference,
    state_deg_deg_mm,
    oct_error_camera_mm,
    camera_rotation,
    *,
    valid=True,
    gain=0.15,
    saturation=None,
    relative_damping=1e-3,
):
    """Compute ``gain * sat(J^+ e)`` in A/B/C deg/deg/mm units."""
    state = np.asarray(state_deg_deg_mm, dtype=np.float64)
    camera_error = np.asarray(oct_error_camera_mm, dtype=np.float64)
    if state.shape != (3,) or camera_error.shape != (3,):
        raise ValueError("state and OCT error must each contain three values")
    if not math.isfinite(gain) or gain < 0:
        raise ValueError("gain must be finite and nonnegative")
    saturation = (
        np.array([0.29, 0.15, 0.05], dtype=np.float64)
        if saturation is None
        else np.asarray(saturation, dtype=np.float64)
    )
    if (
        saturation.shape != (3,)
        or not np.all(np.isfinite(saturation))
        or np.any(saturation <= 0)
    ):
        raise ValueError("saturation must contain three positive values")
    if not valid or np.linalg.norm(camera_error) <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    world_error = camera_error_to_world(camera_error, camera_rotation)
    jacobian = task_jacobian(reference, state * TASK_TO_SI)
    task_delta_si = damped_pseudoinverse_solve(
        jacobian,
        world_error,
        relative_damping=relative_damping,
    )
    task_delta_action_units = task_delta_si * SI_TO_ACTION
    return (
        float(gain)
        * np.clip(
            task_delta_action_units,
            -saturation,
            saturation,
        )
    )


def distance_aware_oct_features(oct_navigation):
    """Return unit direction, log distance and distance from OCT navigation."""
    oct_navigation = np.asarray(oct_navigation, dtype=np.float64)
    if oct_navigation.shape[-1] != 4:
        raise ValueError("OCT navigation must end with xyz+valid")
    xyz = oct_navigation[..., :3]
    distance = np.linalg.norm(xyz, axis=-1)
    direction = np.zeros_like(xyz)
    nonzero = distance > 1e-9
    direction[nonzero] = xyz[nonzero] / distance[nonzero, None]
    log_distance = np.log(np.maximum(distance, 1e-4))
    return direction, log_distance, distance


def residual_limit_for_distance(
    distance_mm,
    valid,
    *,
    far_limit,
    full_effect_distance_mm=1.0,
    near_fraction=0.1,
):
    """Return a residual bound that contracts monotonically near the target."""
    distance = np.asarray(distance_mm, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    far_limit = np.asarray(far_limit, dtype=np.float64)
    if far_limit.shape != (3,) or np.any(far_limit <= 0):
        raise ValueError("far_limit must contain three positive values")
    if (
        full_effect_distance_mm <= 0
        or not 0 <= near_fraction <= 1
    ):
        raise ValueError("invalid residual contraction settings")
    scale = np.clip(
        distance / float(full_effect_distance_mm),
        float(near_fraction),
        1.0,
    )
    scale = np.where(distance > 1e-9, scale, 0.0)
    scale = np.where(valid, scale, 0.0)
    if distance.ndim == 0:
        return far_limit * scale
    return scale[..., None] * far_limit


def _distance_features_tensor(oct_navigation):
    xyz = oct_navigation[..., :3]
    distance = torch.linalg.norm(xyz, dim=-1)
    direction = torch.zeros_like(xyz)
    nonzero = distance > 1e-9
    direction = torch.where(
        nonzero[..., None],
        xyz / torch.clamp(distance[..., None], min=1e-9),
        direction,
    )
    log_distance = torch.log(torch.clamp(distance, min=1e-4))
    return direction, log_distance, distance


def _residual_limit_tensor(
    distance,
    valid,
    far_limit,
    full_effect_distance_mm,
    near_fraction,
):
    scale = torch.clamp(
        distance / float(full_effect_distance_mm),
        min=float(near_fraction),
        max=1.0,
    )
    scale = torch.where(
        distance > 1e-9,
        scale,
        torch.zeros_like(scale),
    )
    scale = torch.where(
        valid,
        scale,
        torch.zeros_like(scale),
    )
    return scale[..., None] * far_limit.to(distance.device).reshape(1, 3)


class GuidedOCTResidualPolicy(nn.Module):
    """Predict the teacher action as guided proportional action + residual."""

    def __init__(
        self,
        *,
        state_mean,
        state_std,
        log_distance_mean,
        log_distance_std,
        base_mean,
        base_std,
        far_residual_limit,
        full_effect_distance_mm=1.0,
        near_fraction=0.1,
        hidden_dim=128,
    ):
        super().__init__()
        for name, value, expected in (
            ("state_mean", state_mean, 3),
            ("state_std", state_std, 3),
            ("base_mean", base_mean, 3),
            ("base_std", base_std, 3),
            ("far_residual_limit", far_residual_limit, 3),
        ):
            array = torch.as_tensor(value, dtype=torch.float32)
            if array.shape != (expected,) or not torch.all(torch.isfinite(array)):
                raise ValueError(f"invalid {name}")
            self.register_buffer(name, array, persistent=False)
        self.register_buffer(
            "log_distance_mean",
            torch.as_tensor(log_distance_mean, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "log_distance_std",
            torch.as_tensor(log_distance_std, dtype=torch.float32),
            persistent=False,
        )
        self.full_effect_distance_mm = float(full_effect_distance_mm)
        self.near_fraction = float(near_fraction)
        self.network = nn.Sequential(
            nn.Linear(11, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, state, oct_navigation, base_action):
        direction, log_distance, distance = _distance_features_tensor(
            oct_navigation
        )
        valid = oct_navigation[..., 3] > 0.5
        normalized_state = (
            state - self.state_mean.reshape(1, 3)
        ) / self.state_std.reshape(1, 3)
        normalized_log_distance = (
            log_distance - self.log_distance_mean
        ) / self.log_distance_std
        normalized_base = (
            base_action - self.base_mean.reshape(1, 3)
        ) / self.base_std.reshape(1, 3)
        features = torch.cat(
            [
                normalized_state,
                direction,
                normalized_log_distance[..., None],
                valid.to(state.dtype)[..., None],
                normalized_base,
            ],
            dim=-1,
        )
        limits = _residual_limit_tensor(
            distance,
            valid,
            self.far_residual_limit,
            self.full_effect_distance_mm,
            self.near_fraction,
        )
        residual = torch.tanh(self.network(features)) * limits
        return base_action + residual, residual, limits, base_action


def _sample_weights(distance, valid):
    weights = np.ones_like(distance, dtype=np.float64)
    weights[distance < 0.5] = 2.0
    weights[distance < 0.1] = 4.0
    weights[distance < 0.02] = 8.0
    weights[~valid] = 0.0
    if not np.any(weights > 0):
        raise ValueError("no valid OCT samples for training")
    return weights


def _build_arrays(
    dataset,
    reference,
    camera_rotation,
    gain,
    saturation,
    relative_damping,
    far_residual_limit,
    full_effect_distance_mm,
    near_fraction,
):
    rows = np.asarray(dataset.valid_indices, dtype=np.int64)
    states = dataset.states[rows].astype(np.float64)
    oct_navigation = np.concatenate(
        [
            dataset.oct_navigation[rows].astype(np.float64),
            dataset.oct_navigation_valid[rows].astype(np.float64)[:, None],
        ],
        axis=1,
    )
    actions = dataset.actions[rows].astype(np.float64)
    episodes = dataset.episode_indices[rows].astype(np.int64)
    base_actions = np.zeros_like(actions)
    for index, (state, oct_row) in enumerate(
        tqdm(
            zip(states, oct_navigation),
            total=len(states),
            desc="Computing proportional OCT guidance",
        )
    ):
        base_actions[index] = proportional_guided_action(
            reference,
            state,
            oct_row[:3],
            camera_rotation,
            valid=bool(oct_row[3] > 0.5),
            gain=gain,
            saturation=saturation,
            relative_damping=relative_damping,
        )
    direction, log_distance, distance = distance_aware_oct_features(
        oct_navigation
    )
    valid = oct_navigation[:, 3] > 0.5
    residual_limit = residual_limit_for_distance(
        distance,
        valid,
        far_limit=far_residual_limit,
        full_effect_distance_mm=full_effect_distance_mm,
        near_fraction=near_fraction,
    )
    residual_target = np.clip(
        actions - base_actions,
        -residual_limit,
        residual_limit,
    )
    return {
        "states": states,
        "oct_navigation": oct_navigation,
        "actions": actions,
        "episodes": episodes,
        "base_actions": base_actions,
        "residual_targets": residual_target,
        "direction": direction,
        "log_distance": log_distance,
        "distance": distance,
        "valid": valid,
    }


def _predict(model, state, oct_navigation, base, batch_size, device):
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(state, dtype=torch.float32),
            torch.as_tensor(oct_navigation, dtype=torch.float32),
            torch.as_tensor(base, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    predicted = []
    model.eval()
    with torch.no_grad():
        for batch_state, batch_oct, batch_base in loader:
            output, _, _, _ = model(
                batch_state.to(device, non_blocking=True),
                batch_oct.to(device, non_blocking=True),
                batch_base.to(device, non_blocking=True),
            )
            predicted.append(output.cpu().numpy())
    return np.concatenate(predicted)


def _metric_block(prediction, target):
    error = prediction - target
    mae = np.abs(error).mean(axis=0)
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    return {
        "samples": int(len(target)),
        "mae_deg_deg_mm": mae.tolist(),
        "rmse_deg_deg_mm": rmse.tolist(),
    }


def _stratified_indices(distance, valid, per_bin, seed):
    rng = np.random.default_rng(seed)
    selected = []
    lower = 0.0
    for upper in DEFAULT_DISTANCE_EDGES_MM[1:]:
        candidates = np.flatnonzero(
            valid
            & (distance >= lower)
            & (distance < upper)
        )
        if len(candidates):
            take = min(int(per_bin), len(candidates))
            selected.extend(rng.choice(candidates, take, replace=False).tolist())
        lower = upper
    return np.asarray(selected, dtype=np.int64)


def _shuffle_oct_by_distance(
    oct_navigation,
    distance,
    episodes,
    seed,
):
    """Replace each OCT vector by a seeded same-distance different-episode one."""
    rng = np.random.default_rng(seed)
    shuffled = oct_navigation.copy()
    lower = 0.0
    for upper in DEFAULT_DISTANCE_EDGES_MM[1:]:
        rows = np.flatnonzero((distance >= lower) & (distance < upper))
        if len(rows):
            for row in rows:
                candidates = rows[episodes[rows] != episodes[row]]
                if len(candidates):
                    chosen = int(rng.choice(candidates))
                    shuffled[row] = oct_navigation[chosen]
        lower = upper
    return shuffled


def _stratified_metrics(prediction, target, distance):
    result = {}
    lower = 0.0
    for upper in DEFAULT_DISTANCE_EDGES_MM[1:]:
        mask = (distance >= lower) & (distance < upper)
        if np.any(mask):
            upper_text = "inf" if math.isinf(upper) else f"{upper:g}"
            result[f"[{lower:g},{upper_text})"] = _metric_block(
                prediction[mask],
                target[mask],
            )
        lower = upper
    return result


def _causal_evaluation(
    model,
    arrays,
    batch_size,
    device,
    seed,
    per_bin,
):
    rows = _stratified_indices(
        arrays["distance"],
        arrays["valid"],
        per_bin,
        seed,
    )
    state = arrays["states"][rows]
    oct_navigation = arrays["oct_navigation"][rows]
    base = arrays["base_actions"][rows]
    target = arrays["actions"][rows]
    distance = arrays["distance"][rows]
    episodes = arrays["episodes"][rows]
    zero = oct_navigation.copy()
    zero[:, :3] = 0.0
    shuffled = _shuffle_oct_by_distance(
        oct_navigation,
        distance,
        episodes,
        seed + 1,
    )
    conditions = {}
    for name, condition_oct in (
        ("real", oct_navigation),
        ("zero_vector", zero),
        ("episode_shuffled", shuffled),
    ):
        prediction = _predict(
            model,
            state,
            condition_oct,
            base,
            batch_size,
            device,
        )
        conditions[name] = {
            "overall": _metric_block(prediction, target),
            "distance_bins": _stratified_metrics(
                prediction,
                target,
                distance,
            ),
        }
    near_key = "[0,0.02)"
    real_near_c = conditions["real"]["distance_bins"][near_key][
        "mae_deg_deg_mm"
    ][2]
    zero_near_c = conditions["zero_vector"]["distance_bins"][near_key][
        "mae_deg_deg_mm"
    ][2]
    shuffle_near_c = conditions["episode_shuffled"]["distance_bins"][
        near_key
    ]["mae_deg_deg_mm"][2]
    zero_gain = (zero_near_c - real_near_c) / max(zero_near_c, 1e-12)
    shuffle_gain = (
        shuffle_near_c - real_near_c
    ) / max(shuffle_near_c, 1e-12)
    return {
        "samples": int(len(rows)),
        "distance_bin_counts": {
            key: value["samples"]
            for key, value in conditions["real"]["distance_bins"].items()
        },
        "conditions": conditions,
        "near_target_C": {
            "real_mae_mm": real_near_c,
            "zero_mae_mm": zero_near_c,
            "shuffled_mae_mm": shuffle_near_c,
            "gain_vs_zero": zero_gain,
            "gain_vs_shuffled": shuffle_gain,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train direct-OCT proportional + bounded residual regression"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--val-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gain", type=float, default=0.15)
    parser.add_argument(
        "--base-saturation",
        type=float,
        nargs=3,
        default=None,
        metavar=("A_DEG", "B_DEG", "C_MM"),
    )
    parser.add_argument("--relative-damping", type=float, default=1e-3)
    parser.add_argument("--full-effect-distance-mm", type=float, default=1.0)
    parser.add_argument("--near-residual-fraction", type=float, default=0.1)
    parser.add_argument(
        "--far-residual-limit",
        type=float,
        nargs=3,
        default=None,
        metavar=("A_DEG", "B_DEG", "C_MM"),
        help="Defaults to the training action saturation limits.",
    )
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--early-stop-patience", type=int, default=25)
    parser.add_argument("--min-delta-mae", type=float, default=1e-7)
    parser.add_argument("--causal-per-bin", type=int, default=64)
    parser.add_argument("--min-causal-gain", type=float, default=0.10)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    if args.gain < 0 or args.relative_damping < 0:
        parser.error("gain and damping must be nonnegative")
    if args.full_effect_distance_mm <= 0:
        parser.error("full effect distance must be positive")
    if not 0 <= args.near_residual_fraction <= 1:
        parser.error("near residual fraction must be in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_stream_logging(args.output_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(f"使用设备: {device}", flush=True)

    train_dataset = RCMDataset(
        args.data_dir,
        n_obs_steps=1,
        n_action_steps=1,
    )
    val_dataset = RCMDataset(
        args.val_data_dir,
        n_obs_steps=1,
        n_action_steps=1,
    )
    reference = Reference()
    _, camera_rotation = reference.camera_pose()
    raw_train_actions = train_dataset.actions[
        np.asarray(train_dataset.valid_indices, dtype=np.int64)
    ]
    saturation = (
        np.max(np.abs(raw_train_actions.astype(np.float64)), axis=0)
        if args.base_saturation is None
        else np.asarray(args.base_saturation, dtype=np.float64)
    )
    far_residual_limit = (
        saturation.copy()
        if args.far_residual_limit is None
        else np.asarray(args.far_residual_limit, dtype=np.float64)
    )
    print(
        "OCT guided action: "
        f"gain={args.gain}, saturation={saturation.tolist()}, "
        f"residual_far={far_residual_limit.tolist()}, "
        f"full_distance_mm={args.full_effect_distance_mm}",
        flush=True,
    )

    train_arrays = _build_arrays(
        train_dataset,
        reference,
        camera_rotation,
        args.gain,
        saturation,
        args.relative_damping,
        far_residual_limit,
        args.full_effect_distance_mm,
        args.near_residual_fraction,
    )
    val_arrays = _build_arrays(
        val_dataset,
        reference,
        camera_rotation,
        args.gain,
        saturation,
        args.relative_damping,
        far_residual_limit,
        args.full_effect_distance_mm,
        args.near_residual_fraction,
    )
    valid_train = train_arrays["valid"]
    state_mean = train_arrays["states"][valid_train].mean(axis=0)
    state_std = np.maximum(
        train_arrays["states"][valid_train].std(axis=0),
        1e-6,
    )
    log_mean = float(train_arrays["log_distance"][valid_train].mean())
    log_std = max(
        float(train_arrays["log_distance"][valid_train].std()),
        1e-6,
    )
    base_mean = train_arrays["base_actions"].mean(axis=0)
    base_std = np.maximum(
        train_arrays["base_actions"].std(axis=0),
        1e-6,
    )
    action_std = np.maximum(
        train_arrays["actions"][valid_train].std(axis=0),
        1e-6,
    )
    sample_weights = _sample_weights(
        train_arrays["distance"],
        valid_train,
    )
    train_tensors = TensorDataset(
        torch.from_numpy(train_arrays["states"].astype(np.float32)),
        torch.from_numpy(
            train_arrays["oct_navigation"].astype(np.float32)
        ),
        torch.from_numpy(
            train_arrays["base_actions"].astype(np.float32)
        ),
        torch.from_numpy(
            train_arrays["actions"].astype(np.float32)
        ),
        torch.from_numpy(
            train_arrays["residual_targets"].astype(np.float32)
        ),
    )
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_tensors,
        batch_size=args.batch_size,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
    )
    model = GuidedOCTResidualPolicy(
        state_mean=state_mean,
        state_std=state_std,
        log_distance_mean=log_mean,
        log_distance_std=log_std,
        base_mean=base_mean,
        base_std=base_std,
        far_residual_limit=far_residual_limit,
        full_effect_distance_mm=args.full_effect_distance_mm,
        near_fraction=args.near_residual_fraction,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.epochs,
    )
    normalized_action_std = torch.as_tensor(
        action_std,
        dtype=torch.float32,
        device=device,
    ).reshape(1, 3)
    print(
        f"训练样本={len(train_tensors)}, 验证样本={len(val_arrays['actions'])}, "
        f"模型参数={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )
    print(
        "distance sampling: "
        f"<0.02mm={int(np.sum(valid_train & (train_arrays['distance'] < 0.02)))} "
        f"<0.1mm={int(np.sum(valid_train & (train_arrays['distance'] < 0.1)))} "
        f"<0.5mm={int(np.sum(valid_train & (train_arrays['distance'] < 0.5)))}",
        flush=True,
    )

    history = []
    best_mae = float("inf")
    best_epoch = -1
    stale_epochs = 0
    checkpoint_path = args.output_dir / "best_model.pth"
    history_path = args.output_dir / "training_history.json"
    val_prediction = np.zeros_like(val_arrays["actions"])
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for state, oct_navigation, base, actions, residual_target in tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
        ):
            state = state.to(device, non_blocking=True)
            oct_navigation = oct_navigation.to(device, non_blocking=True)
            base = base.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            residual_target = residual_target.to(device, non_blocking=True)
            prediction, residual, limits, _ = model(
                state,
                oct_navigation,
                base,
            )
            prediction_loss = torch.mean(
                ((prediction - actions) / normalized_action_std) ** 2
            )
            residual_loss = torch.mean(
                ((residual - residual_target) / normalized_action_std) ** 2
            )
            loss = prediction_loss + 0.25 * residual_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss) * len(actions)
            sample_count += len(actions)
        scheduler.step()
        val_prediction = _predict(
            model,
            val_arrays["states"].astype(np.float32),
            val_arrays["oct_navigation"].astype(np.float32),
            val_arrays["base_actions"].astype(np.float32),
            args.batch_size,
            device,
        )
        val_mae = np.abs(
            val_prediction - val_arrays["actions"]
        ).mean(axis=0)
        normalized_mae = float(np.mean(val_mae / action_std))
        improved = normalized_mae < best_mae - args.min_delta_mae
        if improved:
            best_mae = normalized_mae
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "checkpoint_schema": "oct_guided_residual_regression_v1",
                    "policy_type": "oct_guided_residual_regression",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_normalized_mae": best_mae,
                    "val_mae_deg_deg_mm": val_mae.tolist(),
                    "config": {
                        "state_mean": state_mean.tolist(),
                        "state_std": state_std.tolist(),
                        "log_distance_mean": log_mean,
                        "log_distance_std": log_std,
                        "base_mean": base_mean.tolist(),
                        "base_std": base_std.tolist(),
                        "far_residual_limit": far_residual_limit.tolist(),
                        "full_effect_distance_mm": args.full_effect_distance_mm,
                        "near_fraction": args.near_residual_fraction,
                        "hidden_dim": args.hidden_dim,
                        "gain": args.gain,
                        "base_saturation": saturation.tolist(),
                        "relative_damping": args.relative_damping,
                        "camera_sign": CAMERA_SIGN.tolist(),
                        "oct_feature_definition": (
                            "camera_to_world_jacobian_proportional_plus_"
                            "bounded_residual_direction_log_distance"
                        ),
                    },
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / sample_count,
                "val_mae_deg_deg_mm": val_mae.tolist(),
                "normalized_mae": normalized_mae,
                "best_normalized_mae": best_mae,
                "improved": improved,
            }
        )
        history_tmp = history_path.with_name(history_path.name + ".tmp")
        history_tmp.write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        history_tmp.replace(history_path)
        print(
            f"Epoch {epoch}: train_loss={total_loss / sample_count:.8f} "
            f"val_mae_A/B/C={val_mae.tolist()} "
            f"normalized_mae={normalized_mae:.8f}",
            flush=True,
        )
        if stale_epochs >= args.early_stop_patience:
            print("早停: 验证动作 MAE 未继续改善", flush=True)
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    causal = _causal_evaluation(
        model,
        val_arrays,
        args.batch_size,
        device,
        args.seed + 10000,
        args.causal_per_bin,
    )
    near = causal["near_target_C"]
    passed = bool(
        near["gain_vs_zero"] >= args.min_causal_gain
        and near["gain_vs_shuffled"] >= args.min_causal_gain
    )
    report = {
        "result": "PASS" if passed else "FAIL",
        "best_epoch": best_epoch,
        "best_normalized_mae": best_mae,
        "causal_evaluation": causal,
        "acceptance": {
            "min_causal_gain": args.min_causal_gain,
            "near_target_C_real_better_than_zero": passed,
            "near_target_C_real_better_than_shuffled": passed,
        },
        "config": checkpoint["config"],
    }
    (args.output_dir / "causal_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"训练完成! best_epoch={best_epoch}, "
        f"normalized_mae={best_mae:.8f}",
        flush=True,
    )
    print(
        "近目标 OCT 因果验证: "
        f"real={near['real_mae_mm']:.8f} mm, "
        f"zero={near['zero_mae_mm']:.8f} mm, "
        f"shuffled={near['shuffled_mae_mm']:.8f} mm, "
        f"gain_zero={near['gain_vs_zero']:.2%}, "
        f"gain_shuffle={near['gain_vs_shuffled']:.2%}, "
        f"result={report['result']}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
