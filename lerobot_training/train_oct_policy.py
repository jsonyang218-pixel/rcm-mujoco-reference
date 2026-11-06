"""Train an RGB + robot-state + OCT-navigation diffusion policy.

The depth condition is represented by per-frame OCT target-minus-tip vectors
in camera-optical millimetres. Camera depth maps are not model inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from train_diffusion_policy import (
    DEFAULT_EPOCHS,
    RCMDataset,
    SimpleDiffusionPolicy,
    ThreadBatchLoader,
    configure_stream_logging,
    evaluate_epoch,
    image_transform_for_dataset,
    seed_worker,
    train_epoch,
)


def atomic_torch_save(payload, destination: Path):
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    train_loss,
    val_loss,
    args,
    action_loss_weights,
    oct_scale,
    normalization,
    val_action_mae,
):
    return {
        'checkpoint_schema': 'oct_diffusion_policy_v1',
        'epoch': int(epoch),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),
        # Backward-compatible alias used by older tools.
        'loss': float(train_loss),
        'action_loss_weights': action_loss_weights.tolist(),
        'val_action_mae': val_action_mae,
        'best_normalized_mae': float(
            val_action_mae['normalized_mae']
            if val_action_mae is not None else float('nan')
        ),
        'config': {
            **vars(args),
            'use_oct': bool(args.use_oct),
            'oct_feature_dim': 4 if args.use_oct else 0,
            'oct_scale': oct_scale.tolist(),
            **normalization,
            'oct_feature_definition': (
                (
                    'oct_direction_log_distance_valid_and_'
                    'proportional_base_action_with_bounded_residual'
                    if args.guided_oct
                    else 'target_minus_tip_camera_optical_mm_plus_valid'
                )
                if args.use_oct else 'disabled_for_ablation'
            ),
        },
    }


def build_loader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    thread_workers,
    seed,
    pin_memory,
    prefetch_factor,
    prefetch_batches,
    sample_weights=None,
):
    if num_workers > 0:
        if sample_weights is not None:
            sampler = WeightedRandomSampler(
                torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=True,
                worker_init_fn=seed_worker,
            )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed) if shuffle else None,
        )
    return ThreadBatchLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_threads=thread_workers,
        seed=seed,
        pin_memory=pin_memory,
        prefetch_batches=prefetch_batches,
        sample_weights=sample_weights,
    )


def distance_sampling_weights(
    dataset,
    near_target_weight=8.0,
    close_target_weight=4.0,
    mid_target_weight=2.0,
):
    """Increase sampling probability for the sparse braking region."""
    weights = np.ones(len(dataset), dtype=np.float64)
    rows = np.asarray(dataset.valid_indices, dtype=np.int64)
    distances = np.linalg.norm(dataset.oct_navigation[rows], axis=1)
    valid = dataset.oct_navigation_valid[rows]
    weights[distances < 0.5] = float(mid_target_weight)
    weights[distances < 0.1] = float(close_target_weight)
    weights[distances < 0.02] = float(near_target_weight)
    weights[~valid] = 0.0
    if not np.any(weights > 0):
        raise ValueError("No training samples have valid OCT navigation")
    return weights, distances


def evaluate_action_mae(
    model,
    dataset,
    device,
    batch_size,
    thread_workers,
    seed,
    max_samples,
    action_std,
):
    """Estimate validation action MAE with balanced distance-bin coverage."""
    rows = np.asarray(dataset.valid_indices, dtype=np.int64)
    distances = np.linalg.norm(dataset.oct_navigation[rows], axis=1)
    edges = (0.02, 0.1, 1.0, 5.0, 10.0, np.inf)
    rng = np.random.default_rng(seed)
    bin_count = len(edges)
    per_bin = max(1, int(max_samples) // bin_count)
    selected = []
    lower = -np.inf
    for upper in edges:
        candidates = np.flatnonzero(
            (distances >= lower)
            & (distances < upper)
            & dataset.oct_navigation_valid[rows]
        )
        if len(candidates):
            take = min(per_bin, len(candidates))
            selected.extend(rng.choice(candidates, take, replace=False).tolist())
        lower = upper
    if not selected:
        raise RuntimeError("No validation samples available for action MAE")

    loader = ThreadBatchLoader(
        Subset(dataset, selected),
        batch_size=batch_size,
        shuffle=False,
        num_threads=max(1, min(thread_workers, 8)),
        seed=seed,
        pin_memory=device.type == "cuda",
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    predicted = []
    target = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation action MAE"):
            obs_image = (
                batch["observation.image"].to(
                    device,
                    non_blocking=True,
                )
                if "observation.image" in batch
                else None
            )
            obs_state = batch["observation.state"].to(device, non_blocking=True)
            obs_oct = batch["observation.oct"].to(device, non_blocking=True)
            guided_base = (
                batch["observation.guided_base"].to(
                    device,
                    non_blocking=True,
                )
                if model.guided_oct_mode else None
            )
            obs_vision_feature = (
                batch["observation.vision_feature"].to(
                    device,
                    non_blocking=True,
                )
                if model.use_cached_vision_features else None
            )
            actions = model(
                obs_image,
                obs_state,
                obs_oct=obs_oct,
                generator=generator,
                guided_base=guided_base,
                obs_vision_feature=obs_vision_feature,
            )
            predicted.append(actions[:, 0].cpu().numpy())
            target.append(batch["action"][:, 0].numpy())
    predicted = np.concatenate(predicted)
    target = np.concatenate(target)
    error = predicted - target
    mae = np.abs(error).mean(axis=0)
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    normalized_mae = float(np.mean(mae / action_std))
    return {
        'samples': int(len(predicted)),
        'mae_deg_deg_mm': mae.tolist(),
        'rmse_deg_deg_mm': rmse.tolist(),
        'normalized_mae': normalized_mae,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train RGB + state + OCT-navigation diffusion policy"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--val-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--resume-history", default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--thread-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--num-diffusion-iters", type=int, default=100)
    parser.add_argument(
        "--diffusion-schedule",
        choices=("linear", "cosine"),
        default="cosine",
    )
    parser.add_argument("--pred-x0-clip", type=float, default=4.0)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--min-delta-mae", type=float, default=1e-6)
    parser.add_argument("--val-mae-samples", type=int, default=384)
    parser.add_argument(
        "--vision-mode",
        choices=("global", "spatial_softmax"),
        default="spatial_softmax",
    )
    parser.add_argument(
        "--image-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--geometric-augmentation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable RandomResizedCrop; disabled by default for metric localization",
    )
    parser.add_argument(
        "--freeze-vision-encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--vision-feature-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--use-oct",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--guided-oct",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--oct-noise-std-mm", type=float, default=0.05)
    parser.add_argument("--oct-feature-dropout", type=float, default=0.10)
    parser.add_argument("--oct-invalid-prob", type=float, default=0.05)
    parser.add_argument(
        "--distance-balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--near-target-weight", type=float, default=8.0)
    parser.add_argument("--close-target-weight", type=float, default=4.0)
    parser.add_argument("--mid-target-weight", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--val-seed", type=int, default=20261103)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("fp16", "bf16"),
        default="fp16",
    )
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.guided_oct and not args.use_oct:
        parser.error("--guided-oct requires --use-oct")
    if min(
        args.near_target_weight,
        args.close_target_weight,
        args.mid_target_weight,
    ) <= 0:
        parser.error("distance sampling weights must be positive")
    if args.oct_noise_std_mm < 0:
        parser.error("--oct-noise-std-mm must be nonnegative")
    if not 0 <= args.oct_feature_dropout <= 1:
        parser.error("--oct-feature-dropout must be in [0, 1]")
    if not 0 <= args.oct_invalid_prob <= 1:
        parser.error("--oct-invalid-prob must be in [0, 1]")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_stream_logging(output_dir, args.log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    if use_amp and args.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 CUDA 设备不支持 bfloat16")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    print(f"使用设备: {device}", flush=True)
    print(f"混合精度: {args.amp_dtype if use_amp else 'disabled'}", flush=True)

    train_transform = image_transform_for_dataset(
        args.data_dir,
        augment=args.image_augmentation,
        geometric_augment=args.geometric_augmentation,
    )
    val_transform = image_transform_for_dataset(args.val_data_dir)
    train_dataset = RCMDataset(
        Path(args.data_dir),
        train_transform,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        use_vision_feature_cache=args.vision_feature_cache,
    )
    val_dataset = RCMDataset(
        Path(args.val_data_dir),
        val_transform,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        use_vision_feature_cache=args.vision_feature_cache,
    )
    if not (train_dataset.has_oct_navigation and val_dataset.has_oct_navigation):
        raise ValueError("Both datasets must contain OCT navigation fields")
    if not np.any(train_dataset.oct_navigation_valid):
        raise ValueError("Training dataset has no valid OCT navigation")
    if args.guided_oct and not (
        train_dataset.has_guided_oct_action
        and val_dataset.has_guided_oct_action
    ):
        raise ValueError(
            "guided OCT datasets require guided_base_action.npy and "
            "guided_residual_limit.npy sidecars"
        )

    valid_oct = train_dataset.oct_navigation[
        train_dataset.oct_navigation_valid
    ].astype(np.float64)
    oct_mean = valid_oct.mean(axis=0)
    oct_std = valid_oct.std(axis=0)
    if not np.all(np.isfinite(oct_std)) or np.any(oct_std <= 1e-9):
        raise ValueError(f"Invalid OCT navigation std: {oct_std.tolist()}")
    oct_scale = np.ones(4, dtype=np.float32)

    state_mean = train_dataset.states.mean(axis=0).astype(np.float64)
    state_std = train_dataset.states.std(axis=0).astype(np.float64)
    if not np.all(np.isfinite(state_std)) or np.any(state_std <= 1e-9):
        raise ValueError(f"Invalid state std: {state_std.tolist()}")
    if args.guided_oct:
        train_rows = np.asarray(
            train_dataset.valid_indices,
            dtype=np.int64,
        )
        residual_targets = np.stack(
            [
                train_dataset.actions[train_rows + step]
                - train_dataset.guided_base_action[train_rows]
                for step in range(args.n_action_steps)
            ],
            axis=1,
        )
        residual_limits = train_dataset.guided_residual_limit[
            train_rows
        ][:, None, :]
        residual_targets = np.clip(
            residual_targets,
            -residual_limits,
            residual_limits,
        )
        action_mean = residual_targets.reshape(-1, 3).mean(axis=0)
        action_std = residual_targets.reshape(-1, 3).std(axis=0)
        guided_meta = json.loads(
            (
                Path(args.data_dir)
                / "data"
                / "guided_oct_metadata.json"
            ).read_text(encoding="utf-8")
        )
        guided_base_mean = np.asarray(
            guided_meta["base_mean_deg_deg_mm"],
            dtype=np.float64,
        )
        guided_base_std = np.asarray(
            guided_meta["base_std_deg_deg_mm"],
            dtype=np.float64,
        )
        log_distance_mean = float(guided_meta["log_distance_mean"])
        log_distance_std = float(guided_meta["log_distance_std"])
        far_residual_limit = np.asarray(
            guided_meta["far_residual_limit_deg_deg_mm"],
            dtype=np.float64,
        )
        full_effect_distance_mm = float(
            guided_meta["full_effect_distance_mm"]
        )
        near_fraction = float(guided_meta["near_fraction"])
        guided_gain = float(guided_meta["gain"])
        guided_saturation = np.asarray(
            guided_meta["base_saturation_deg_deg_mm"],
            dtype=np.float64,
        )
        guided_relative_damping = float(
            guided_meta["relative_damping"]
        )
    else:
        action_mean = train_dataset.actions.mean(axis=0).astype(np.float64)
        action_std = train_dataset.actions.std(axis=0).astype(np.float64)
        guided_base_mean = np.zeros(3, dtype=np.float64)
        guided_base_std = np.ones(3, dtype=np.float64)
        log_distance_mean = 0.0
        log_distance_std = 1.0
        far_residual_limit = np.ones(3, dtype=np.float64)
        full_effect_distance_mm = 1.0
        near_fraction = 0.1
        guided_gain = 0.0
        guided_saturation = np.ones(3, dtype=np.float64)
        guided_relative_damping = 0.0
    if not np.all(np.isfinite(action_std)) or np.any(action_std <= 1e-9):
        raise ValueError(f"Invalid action std: {action_std.tolist()}")
    action_loss_weights = np.ones(3, dtype=np.float32)

    print(
        f"OCT valid: train={int(train_dataset.oct_navigation_valid.sum())}/"
        f"{len(train_dataset.states)}, "
        f"val={int(val_dataset.oct_navigation_valid.sum())}/"
        f"{len(val_dataset.states)}",
        flush=True,
    )
    print(
        f"State mean/std: {state_mean.tolist()} / {state_std.tolist()}",
        flush=True,
    )
    print(
        f"OCT mean/std [mm]: {oct_mean.tolist()} / {oct_std.tolist()}",
        flush=True,
    )
    print(
        f"Action mean/std [deg,deg,mm]: {action_mean.tolist()} / "
        f"{action_std.tolist()}",
        flush=True,
    )
    print(
        f"Action loss weights [A,B,C]: {action_loss_weights.tolist()}",
        flush=True,
    )
    sample_weights = None
    if args.distance_balanced_sampling:
        sample_weights, train_distances = distance_sampling_weights(
            train_dataset,
            near_target_weight=args.near_target_weight,
            close_target_weight=args.close_target_weight,
            mid_target_weight=args.mid_target_weight,
        )
        print(
            "distance sampling: "
            f"<0.02mm={int(np.sum(train_distances < 0.02))} "
            f"<0.1mm={int(np.sum(train_distances < 0.1))} "
            f"<0.5mm={int(np.sum(train_distances < 0.5))} "
            f"weights=(near={args.near_target_weight:g},"
            f"close={args.close_target_weight:g},"
            f"mid={args.mid_target_weight:g})",
            flush=True,
        )

    pin_memory = device.type == "cuda"
    train_loader = build_loader(
        train_dataset,
        args.batch_size,
        True,
        args.num_workers,
        args.thread_workers,
        args.seed,
        pin_memory,
        args.prefetch_factor,
        args.prefetch_factor,
        sample_weights=sample_weights,
    )
    val_loader = build_loader(
        val_dataset,
        args.val_batch_size,
        False,
        min(args.num_workers, 2),
        max(1, min(args.thread_workers, 4)),
        args.val_seed,
        pin_memory,
        args.prefetch_factor,
        args.prefetch_factor,
    )
    loader_mode = (
        f"process workers={args.num_workers}"
        if args.num_workers > 0
        else f"thread workers={args.thread_workers}"
    )
    print(f"数据加载: {loader_mode}", flush=True)

    model = SimpleDiffusionPolicy(
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        num_diffusion_iters=args.num_diffusion_iters,
        diffusion_schedule=args.diffusion_schedule,
        pred_x0_clip=args.pred_x0_clip,
        oct_dim=4 if args.use_oct else 0,
        oct_scale=oct_scale,
        state_mean=state_mean,
        state_std=state_std,
        oct_mean=oct_mean,
        oct_std=oct_std,
        action_mean=action_mean,
        action_std=action_std,
        loss_weights=action_loss_weights,
        vision_mode=args.vision_mode,
        oct_noise_std_mm=args.oct_noise_std_mm,
        oct_feature_dropout=args.oct_feature_dropout,
        oct_invalid_prob=args.oct_invalid_prob,
        guided_oct_mode=args.guided_oct,
        guided_base_mean=guided_base_mean,
        guided_base_std=guided_base_std,
        log_distance_mean=log_distance_mean,
        log_distance_std=log_distance_std,
        far_residual_limit=far_residual_limit,
        full_effect_distance_mm=full_effect_distance_mm,
        near_fraction=near_fraction,
        use_cached_vision_features=args.vision_feature_cache,
    ).to(device)
    if args.init_from:
        initial_checkpoint = torch.load(
            args.init_from,
            map_location=device,
            weights_only=False,
        )
        current_state = model.state_dict()
        compatible = {
            key: value
            for key, value in initial_checkpoint["model_state_dict"].items()
            if key in current_state
            and current_state[key].shape == value.shape
        }
        load_result = model.load_state_dict(compatible, strict=False)
        print(
            f"从 {args.init_from} 初始化兼容权重 "
            f"{len(compatible)} 项；未加载 "
            f"{len(load_result.missing_keys)} 项",
            flush=True,
        )
    if args.freeze_vision_encoder:
        model.vision_encoder.freeze_features = True
        for parameter in model.vision_encoder.encoder.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp and amp_dtype == torch.float16,
    )
    print(
        f"模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
        flush=True,
    )

    start_epoch = 0
    history = []
    best_val_mae = float("inf")
    normalization = {
        'state_mean': state_mean.tolist(),
        'state_std': state_std.tolist(),
        'oct_mean': oct_mean.tolist(),
        'oct_std': oct_std.tolist(),
        'action_mean': action_mean.tolist(),
        'action_std': action_std.tolist(),
        'guided_oct_mode': bool(args.guided_oct),
        'guided_base_mean': guided_base_mean.tolist(),
        'guided_base_std': guided_base_std.tolist(),
        'log_distance_mean': log_distance_mean,
        'log_distance_std': log_distance_std,
        'far_residual_limit': far_residual_limit.tolist(),
        'full_effect_distance_mm': full_effect_distance_mm,
        'near_fraction': near_fraction,
        'guided_gain': guided_gain,
        'guided_saturation': guided_saturation.tolist(),
        'guided_relative_damping': guided_relative_damping,
        'use_cached_vision_features': bool(args.vision_feature_cache),
    }
    if args.resume_from:
        resume_path = Path(args.resume_from)
        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        checkpoint_config = checkpoint.get('config', {})
        for name in ('n_obs_steps', 'n_action_steps', 'num_diffusion_iters'):
            saved = checkpoint_config.get(name)
            if saved is not None and int(saved) != int(getattr(args, name)):
                raise ValueError(
                    f"Resume config mismatch for {name}: {saved} != "
                    f"{getattr(args, name)}"
                )
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = int(checkpoint['epoch']) + 1
        checkpoint_mae = checkpoint.get('val_action_mae') or {}
        best_val_mae = float(checkpoint.get(
            'best_normalized_mae',
            checkpoint_mae.get('normalized_mae', float('inf')),
        ))
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            for _ in range(start_epoch):
                scheduler.step()
        if 'scaler_state_dict' in checkpoint and scaler.is_enabled():
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        history_source = (
            Path(args.resume_history)
            if args.resume_history
            else resume_path.parent / 'training_history.json'
        )
        if history_source.is_file():
            history = json.loads(history_source.read_text(encoding='utf-8'))
        print(
            f"从 {resume_path} 恢复；下一轮 epoch {start_epoch}，"
            f"best normalized MAE={best_val_mae:.6f}",
            flush=True,
        )

    history_path = output_dir / "training_history.json"
    if history:
        with open(history_path, "w", encoding="utf-8") as stream:
            json.dump(history, stream, ensure_ascii=False, indent=2)
    if args.resume_from and not (output_dir / "best_model.pth").exists():
        atomic_torch_save(
            checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                int(checkpoint['epoch']),
                float(checkpoint.get('train_loss', checkpoint.get('loss', 0.0))),
                float(checkpoint.get('val_loss', 0.0)),
                args,
                action_loss_weights,
                oct_scale,
                normalization,
                checkpoint.get('val_action_mae'),
            ),
            output_dir / "best_model.pth",
        )

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        val_loss = evaluate_epoch(
            model,
            val_loader,
            device,
            seed=args.val_seed,
            epoch=epoch,
            amp=use_amp,
            amp_dtype=amp_dtype,
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        mae_metrics = evaluate_action_mae(
            model,
            val_dataset,
            device,
            max(1, args.val_batch_size),
            args.thread_workers,
            args.val_seed + epoch,
            args.val_mae_samples,
            action_std,
        )
        normalized_mae = mae_metrics["normalized_mae"]
        improved = normalized_mae < best_val_mae - args.min_delta_mae
        if improved:
            best_val_mae = normalized_mae
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': learning_rate,
            'best_val_mae': best_val_mae,
            'val_action_mae': mae_metrics,
            'improved': improved,
        })
        history_tmp = history_path.with_name(history_path.name + ".tmp")
        with open(history_tmp, "w", encoding="utf-8") as stream:
            json.dump(history, stream, ensure_ascii=False, indent=2)
        history_tmp.replace(history_path)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_normalized_mae={normalized_mae:.6f} "
            f"best_val_mae={best_val_mae:.6f}",
            flush=True,
        )
        if improved:
            atomic_torch_save(
                checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    train_loss,
                    val_loss,
                    args,
                    action_loss_weights,
                    oct_scale,
                    normalization,
                    mae_metrics,
                ),
                output_dir / "best_model.pth",
            )
        elif epoch >= args.early_stop_patience:
            recent = history[-args.early_stop_patience:]
            if not any(row["improved"] for row in recent):
                print(
                    f"早停: 连续 {args.early_stop_patience} 轮验证动作 "
                    "MAE 未改善",
                    flush=True,
                )
                break

    print(f"训练完成! 最佳验证动作 MAE: {best_val_mae:.6f}", flush=True)
    print(f"模型保存在: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
