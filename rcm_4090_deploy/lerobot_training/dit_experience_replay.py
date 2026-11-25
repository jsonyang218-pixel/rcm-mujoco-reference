#!/usr/bin/env python3
"""State-attention DiT-Flow training with expert-corrected experience replay.

The experiment keeps the existing Flow-matching objective and action-block
contract. It does not train a critic and does not use reward in the loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

try:
    from . import dit_state_attention as attention
    from .experience_replay import ACTION_LIMIT
    from .dataset_contract import valid_window_indices
    from .train_diffusion_policy import (
        RCMDataset,
        configure_stream_logging,
        image_transform_for_dataset,
        seed_worker,
    )
    from .train_dit_rectified_flow import _atomic_torch_save, _loss, _move_batch
    from .navigation_overlay import file_sha256
except ImportError:
    import dit_state_attention as attention
    from experience_replay import ACTION_LIMIT
    from dataset_contract import valid_window_indices
    from train_diffusion_policy import (
        RCMDataset,
        configure_stream_logging,
        image_transform_for_dataset,
        seed_worker,
    )
    from train_dit_rectified_flow import _atomic_torch_save, _loss, _move_batch
    from navigation_overlay import file_sha256


TRAINING_MODULE = "state_attention_experience_replay_flow_v1"
STAT_NAMES = (
    "state_mean",
    "state_std",
    "oct_mean",
    "oct_std",
    "action_mean",
    "action_std",
    "velocity_mean",
    "velocity_std",
    "previous_action_mean",
    "previous_action_std",
)
INTEGER_POLICY_NAMES = (
    "n_obs_steps",
    "n_action_steps",
    "dit_dim",
    "dit_depth",
    "dit_heads",
    "flow_integration_steps",
)


class IndexSubset(Dataset):
    """Expose selected dataset sample indices without copying observations."""

    def __init__(
        self,
        dataset,
        indices,
        source=0,
        stationary_frame_counts=None,
        correction_weights=None,
    ):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.source = int(source)
        if self.indices.ndim != 1:
            raise ValueError("subset indices must be one-dimensional")
        if len(self.indices) and (
            self.indices.min() < 0 or self.indices.max() >= len(dataset)
        ):
            raise IndexError("subset index outside source dataset")
        self.stationary_frame_counts = (
            np.asarray(stationary_frame_counts, dtype=np.int64)
            if stationary_frame_counts is not None
            else np.zeros(len(self.indices), dtype=np.int64)
        )
        self.correction_weights = (
            np.asarray(correction_weights, dtype=np.float64)
            if correction_weights is not None
            else np.ones(len(self.indices), dtype=np.float64)
        )
        if self.stationary_frame_counts.shape != (len(self.indices),):
            raise ValueError(
                "stationary frame counts must match subset length"
            )
        if self.correction_weights.shape != (len(self.indices),):
            raise ValueError(
                "subset correction weights must match subset length"
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample = dict(self.dataset[int(self.indices[int(index)])])
        sample["sample.source"] = torch.tensor(self.source, dtype=torch.long)
        sample["sample.index"] = torch.tensor(int(index), dtype=torch.long)
        sample["sample.stationary_frame_count"] = torch.tensor(
            int(self.stationary_frame_counts[int(index)]),
            dtype=torch.long,
        )
        sample["sample.correction_weight"] = torch.tensor(
            float(self.correction_weights[int(index)]),
            dtype=torch.float32,
        )
        return sample


class ReplacementBatchSampler:
    """Yield a fixed number of uniformly sampled batches with replacement."""

    def __init__(self, sample_count, batches, batch_size, seed):
        self.sample_count = int(sample_count)
        self.batches = int(batches)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        if min(self.sample_count, self.batches, self.batch_size) < 1:
            raise ValueError("sampler counts must be positive")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.batches):
            yield rng.integers(
                self.sample_count, size=self.batch_size
            ).tolist()

    def __len__(self):
        return self.batches


class BalancedReplacementBatchSampler:
    """Yield batches with an exact original/correction sample split."""

    def __init__(
        self,
        original_count,
        correction_count,
        batches,
        batch_size,
        seed,
    ):
        self.original_count = int(original_count)
        self.correction_count = int(correction_count)
        self.batches = int(batches)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        if min(
            self.original_count,
            self.correction_count,
            self.batches,
            self.batch_size,
        ) < 1:
            raise ValueError("sampler counts must be positive")
        if self.batch_size % 2:
            raise ValueError("balanced batch size must be even")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        half = self.batch_size // 2
        for _ in range(self.batches):
            original = rng.integers(
                self.original_count, size=half
            ).tolist()
            correction = (
                self.original_count
                + rng.integers(self.correction_count, size=half)
            ).tolist()
            yield original + correction

    def __len__(self):
        return self.batches


class RatioReplacementBatchSampler:
    """Shuffle a fixed epoch quota; sample within each pool with replacement."""

    def __init__(
        self,
        original_count,
        correction_count,
        correction_ratio,
        batches,
        batch_size,
        seed,
        correction_weights=None,
        near_target_mask=None,
        near_target_weight=1.0,
    ):
        self.original_count = int(original_count)
        self.correction_count = int(correction_count)
        self.correction_ratio = float(correction_ratio)
        self.batches = int(batches)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.near_target_weight = float(near_target_weight)
        if min(
            self.original_count,
            self.correction_count,
            self.batches,
            self.batch_size,
        ) < 1:
            raise ValueError("sampler counts must be positive")
        if not 0.0 < self.correction_ratio < 1.0:
            raise ValueError("correction ratio must be in (0, 1)")
        if not math.isfinite(self.near_target_weight) or self.near_target_weight < 0:
            raise ValueError("near-target weight must be finite and nonnegative")
        if correction_weights is None:
            correction_weights = np.ones(self.correction_count, dtype=np.float64)
        correction_weights = np.asarray(correction_weights, dtype=np.float64)
        if correction_weights.shape != (self.correction_count,):
            raise ValueError("correction weights must match the correction pool")
        if not np.isfinite(correction_weights).all() or np.any(correction_weights < 0):
            raise ValueError("correction weights must be finite and nonnegative")
        if near_target_mask is None:
            near_target_mask = np.zeros(self.correction_count, dtype=np.bool_)
        near_target_mask = np.asarray(near_target_mask, dtype=np.bool_)
        if near_target_mask.shape != (self.correction_count,):
            raise ValueError("near-target mask must match the correction pool")
        correction_weights = correction_weights.copy()
        correction_weights[near_target_mask] *= self.near_target_weight
        if correction_weights.sum() <= 0:
            raise ValueError("correction weight pool is empty or all zero")
        self.correction_weights = correction_weights
        self.correction_probabilities = (
            correction_weights / correction_weights.sum()
        )

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        total = self.batches * self.batch_size
        correction_total = int(round(self.correction_ratio * total))
        correction_mask = np.zeros(total, dtype=np.bool_)
        correction_mask[:correction_total] = True
        rng.shuffle(correction_mask)
        for batch_index in range(self.batches):
            start = batch_index * self.batch_size
            choose_correction = correction_mask[
                start : start + self.batch_size
            ]
            indices = np.empty(self.batch_size, dtype=np.int64)
            original_rows = np.flatnonzero(~choose_correction)
            correction_rows = np.flatnonzero(choose_correction)
            if len(original_rows):
                indices[original_rows] = rng.integers(
                    self.original_count, size=len(original_rows)
                )
            if len(correction_rows):
                indices[correction_rows] = (
                    self.original_count
                    + rng.choice(
                        self.correction_count,
                        size=len(correction_rows),
                        replace=True,
                        p=self.correction_probabilities,
                    )
                )
            yield indices.tolist()

    def __len__(self):
        return self.batches


def split_experience_indices(dataset, correction_source="teacher"):
    """Split valid action windows by the episode's supervision source."""
    source_metadata = {}
    episodes = {}
    for item in dataset.metadata.get("episodes", []):
        source = item.get("action_source", "executed")
        if source == "per_episode":
            source_name = Path(item["source_dataset"]).name
            source_index = int(item["source_episode_index"])
            source_dir = dataset.data_dir.parent / source_name
            if source_name not in source_metadata:
                metadata = json.loads(
                    (source_dir / "metadata.json").read_text(
                        encoding="utf-8"
                    )
                )
                source_metadata[source_name] = {
                    int(row["episode_index"]): row for row in metadata["episodes"]
                }
            source_episodes = source_metadata[source_name]
            if source_index not in source_episodes:
                raise ValueError(
                    f"source episode {source_index} is missing from {source_name}"
                )
            source = source_episodes[source_index].get(
                "action_source", "executed"
            )
        episodes[int(item["episode_index"])] = source
    original, correction = [], []
    for sample_index, row in enumerate(dataset.valid_indices):
        episode = int(dataset.episode_indices[int(row)])
        if episode not in episodes:
            raise ValueError(
                f"episode {episode} has no declared action source"
            )
        source = episodes[episode]
        if source == correction_source:
            correction.append(sample_index)
        elif source == "executed":
            original.append(sample_index)
        else:
            raise ValueError(
                f"episode {episode} has unsupported action source {source!r}"
            )
    return np.asarray(original, dtype=np.int64), np.asarray(
        correction, dtype=np.int64
    )


def _loader_options(args):
    return {
        "num_workers": args.num_workers,
        "pin_memory": args.device == "cuda" and torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "persistent_workers": args.num_workers > 0,
        "prefetch_factor": 2 if args.num_workers > 0 else None,
    }


def _index_sha256(indices):
    return hashlib.sha256(
        np.asarray(indices, dtype="<i8").tobytes()
    ).hexdigest()


def _train_loader(
    args,
    dataset,
    mode,
    original_count,
    correction_count,
    correction_ratio,
    epoch,
    correction_weights=None,
    near_target_mask=None,
):
    if mode == "balanced":
        sampler = BalancedReplacementBatchSampler(
            original_count,
            correction_count,
            args.steps_per_epoch,
            args.batch_size,
            args.seed + epoch,
        )
    elif mode == "ratio":
        sampler = RatioReplacementBatchSampler(
            original_count,
            correction_count,
            correction_ratio,
            args.steps_per_epoch,
            args.batch_size,
            args.seed + epoch,
            correction_weights=correction_weights,
            near_target_mask=near_target_mask,
            near_target_weight=getattr(args, "near_target_weight", 1.0),
        )
    else:
        sampler = ReplacementBatchSampler(
            len(dataset),
            args.steps_per_epoch,
            args.batch_size,
            args.seed + epoch,
        )
    options = _loader_options(args)
    if not options["persistent_workers"]:
        options.pop("persistent_workers")
    if options["prefetch_factor"] is None:
        options.pop("prefetch_factor")
    return DataLoader(dataset, batch_sampler=sampler, **options)


def _validation_loader(args, dataset):
    generator = torch.Generator().manual_seed(args.seed)
    options = _loader_options(args)
    if not options["persistent_workers"]:
        options.pop("persistent_workers")
    if options["prefetch_factor"] is None:
        options.pop("prefetch_factor")
    return DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        generator=generator,
        **options,
    )


def _load_policy(init_from, device, oct_symlog_enabled=None, use_local_oct_branch=None):
    checkpoint = torch.load(init_from, map_location="cpu", weights_only=False)
    if checkpoint.get("policy_type") != "dit_rectified_flow":
        raise ValueError("initial checkpoint is not a DiT rectified-Flow policy")
    if checkpoint.get("architecture") != attention.ARCHITECTURE:
        raise ValueError(
            "initial checkpoint is not the state-attention v1 architecture"
        )
    config = dict(checkpoint["config"])
    policy = attention.StateConditionedDiTRectifiedFlowPolicy(
        **{name: config[name] for name in STAT_NAMES},
        **{name: int(config[name]) for name in INTEGER_POLICY_NAMES},
        oct_symlog_enabled=(
            bool(config.get("oct_symlog_enabled", False))
            if oct_symlog_enabled is None
            else bool(oct_symlog_enabled)
        ),
        use_local_oct_branch=(bool(config.get("use_local_oct_branch", False))
                              if use_local_oct_branch is None else bool(use_local_oct_branch)),
        local_oct_tanh_scale_mm=float(config.get("local_oct_tanh_scale_mm", 0.05)),
        local_oct_log_scale_mm=float(config.get("local_oct_log_scale_mm", 0.005)),
        local_oct_log_reference_mm=float(config.get("local_oct_log_reference_mm", 0.5)),
    ).to(device)
    state = dict(checkpoint["model_state_dict"])
    if policy.oct_symlog_enabled and not config.get("oct_symlog_enabled", False):
        if any(name.startswith("oct_symlog_") for name in state):
            raise ValueError("unexpected symlog weights in a baseline checkpoint")
        state.update({
            name: value for name, value in policy.state_dict().items()
            if name.startswith("oct_symlog_")
        })
    if policy.use_local_oct_branch and not config.get("use_local_oct_branch", False):
        if any(name.startswith("local_oct_mlp.") for name in state):
            raise ValueError("unexpected local OCT weights in baseline checkpoint")
        state.update({name: value for name, value in policy.state_dict().items()
                      if name.startswith("local_oct_mlp.")})
    policy.load_state_dict(state)
    return policy, config, int(checkpoint.get("epoch", -1))


def _optimizer(model, args):
    backbone, attention_parameters = [], []
    for name, parameter in model.named_parameters():
        (
            attention_parameters
            if "temporal_bias" in name
            else backbone
        ).append(parameter)
    if args.backbone_lr == 0:
        for parameter in backbone:
            parameter.requires_grad_(False)
    active_backbone = [parameter for parameter in backbone if parameter.requires_grad]
    active_attention = [
        parameter for parameter in attention_parameters if parameter.requires_grad
    ]
    groups = []
    if active_backbone:
        groups.append({"params": active_backbone, "lr": args.backbone_lr})
    if active_attention:
        groups.append({"params": active_attention, "lr": args.attention_lr})
    if not groups:
        raise ValueError("optimizer has no trainable parameters")
    return torch.optim.AdamW(
        groups,
        weight_decay=args.weight_decay,
    )


def assert_temporal_bias_only(model, args, *, freeze=False):
    """Enforce the frozen-backbone experiment contract when requested."""
    if not bool(getattr(args, "require_temporal_bias_only", False)):
        return
    if float(args.backbone_lr) != 0.0:
        raise ValueError(
            "strict frozen-backbone training requires backbone_lr=0"
        )
    if freeze:
        for name, parameter in model.named_parameters():
            if "temporal_bias" not in name:
                parameter.requires_grad_(False)
    offenders = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "temporal_bias" not in name
    ]
    if offenders:
        raise ValueError(
            "strict frozen-backbone training permits only temporal_bias "
            f"parameters; trainable: {offenders}"
        )
    temporal = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "temporal_bias" in name
    ]
    if not temporal:
        raise ValueError("strict frozen-backbone training has no temporal_bias")


def _scheduler(optimizer, args, total_batches):
    if args.lr_schedule == "constant":
        return None
    if args.lr_schedule != "cosine":
        raise ValueError(f"unsupported learning-rate schedule: {args.lr_schedule}")
    lr_min = float(getattr(args, "lr_min", 0.0))
    if not math.isfinite(lr_min) or lr_min < 0:
        raise ValueError("cosine minimum learning rate must be finite and nonnegative")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(total_batches)),
        eta_min=lr_min,
    )


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def load_correction_window_weights(
    data_dir,
    valid_indices,
    *,
    near_target_um=None,
):
    """Resolve per-window correction weights and the optional near-target mask."""
    data_dir = Path(data_dir)
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    overlay_path = data_dir / "data" / "label_repair_overlay.npz"
    if overlay_path.is_file():
        with np.load(overlay_path, allow_pickle=False) as overlay:
            overlay_rows = overlay["correction_window_indices"].astype(np.int64)
            overlay_weights = overlay["correction_window_weight"].astype(np.float64)
        if not np.array_equal(overlay_rows, valid_indices):
            raise ValueError(
                "correction window weights do not match unchanged valid indices"
            )
        weights = overlay_weights
        source = str(overlay_path)
    else:
        weights = np.ones(len(valid_indices), dtype=np.float64)
        source = "implicit_uniform"
    near = np.zeros(len(valid_indices), dtype=np.bool_)
    if near_target_um is not None:
        with np.load(data_dir / "data" / "dataset.npz", allow_pickle=False) as data:
            if "truth_navigation_mm" not in data:
                raise ValueError(
                    "near-target sampling requires truth_navigation_mm"
                )
            truth = data["truth_navigation_mm"][valid_indices].astype(np.float64)
        distance_um = np.linalg.norm(truth, axis=1) * 1000.0
        near = distance_um <= float(near_target_um)
    return weights, near, {
        "source": source,
        "window_count": len(valid_indices),
        "weight_min": float(weights.min()) if len(weights) else None,
        "weight_max": float(weights.max()) if len(weights) else None,
        "weight_mean": float(weights.mean()) if len(weights) else None,
        "near_target_um": near_target_um,
        "near_target_windows": int(near.sum()),
    }


def load_stationary_window_counts(
    data_dir,
    valid_indices,
    *,
    action_steps=16,
):
    """Count configured T3-stationary frames in every correction window."""
    data_dir = Path(data_dir)
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    overlay_path = data_dir / "data" / "label_repair_overlay.npz"
    if not overlay_path.is_file():
        return (
            np.zeros(len(valid_indices), dtype=np.int64),
            {
                "available": False,
                "window_count": len(valid_indices),
                "stationary_window_count": 0,
                "stationary_action_frames": 0,
                "action_frames": int(len(valid_indices) * action_steps),
                "t3_stationary_weight": None,
            },
        )
    with np.load(overlay_path, allow_pickle=False) as overlay:
        stationary = np.asarray(
            overlay["t3_stationary"], dtype=np.bool_
        )
        weight = (
            float(np.asarray(overlay["t3_stationary_weight"]).reshape(-1)[0])
            if "t3_stationary_weight" in overlay
            else None
        )
    counts = np.zeros(len(valid_indices), dtype=np.int64)
    for row, index in enumerate(valid_indices):
        start = int(index)
        stop = start + int(action_steps)
        if stop > len(stationary):
            raise ValueError(
                "correction window extends beyond stationary frame metadata"
            )
        counts[row] = int(np.count_nonzero(stationary[start:stop]))
    return counts, {
        "available": True,
        "overlay": str(overlay_path),
        "window_count": len(valid_indices),
        "stationary_window_count": int(np.count_nonzero(counts)),
        "stationary_action_frames": int(counts.sum()),
        "action_frames": int(len(valid_indices) * action_steps),
        "t3_stationary_weight": weight,
    }


def stationary_sampling_summary(
    epoch_counts,
    *,
    weights,
    stationary_counts,
    correction_ratio,
    action_steps=16,
):
    """Combine sampled and theoretical stationary-frame proportions."""
    stationary_counts = np.asarray(stationary_counts, dtype=np.float64)
    weights = (
        np.ones(len(stationary_counts), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if weights.shape != stationary_counts.shape:
        raise ValueError(
            "stationary counts and correction weights must align"
        )
    total_weight = float(weights.sum())
    weighted_stationary_fraction = (
        float(np.sum(weights * stationary_counts))
        / max(1e-12, total_weight * float(action_steps))
        if total_weight > 0
        else 0.0
    )
    correction_samples = int(epoch_counts.get("correction_samples", 0))
    correction_action_frames = int(
        epoch_counts.get("correction_action_frames", 0)
    )
    stationary_frames = int(
        epoch_counts.get("t3_stationary_action_frames", 0)
    )
    total_samples = int(epoch_counts.get("total_samples", 0))
    return {
        "correction_samples": correction_samples,
        "correction_action_frames": correction_action_frames,
        "t3_stationary_action_frames": stationary_frames,
        "stationary_windows": int(epoch_counts.get("stationary_windows", 0)),
        "actual_t3_stationary_action_fraction": (
            stationary_frames / correction_action_frames
            if correction_action_frames
            else 0.0
        ),
        "theoretical_weighted_t3_stationary_action_fraction": (
            weighted_stationary_fraction
        ),
        "all_batch_theoretical_t3_stationary_action_fraction": (
            float(correction_ratio) * weighted_stationary_fraction
        ),
        "all_batch_actual_t3_stationary_action_fraction": (
            stationary_frames / (total_samples * int(action_steps))
            if total_samples
            else 0.0
        ),
    }


def _validate(model, loader, device, args):
    model.eval()
    total, count = 0.0, 0
    autocast = args.amp and device.type == "cuda"
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(args.validation_seed)
        with torch.no_grad():
            for batch in tqdm(
                loader, desc="Experience replay validation"
            ):
                batch = _move_batch(batch, device)
                with torch.autocast(
                    device_type=device.type,
                    enabled=autocast,
                    dtype=torch.float16,
                ):
                    loss = _loss(model, batch)
                total += float(loss.detach())
                count += 1
                if (
                    args.validation_max_batches is not None
                    and count >= args.validation_max_batches
                ):
                    break
    return total / max(count, 1)


def _select_batch(batch, mask):
    return {
        name: value[mask]
        for name, value in batch.items()
        if not name.startswith("sample.")
    }


def _log_metrics(
    model,
    batch,
    device,
    args,
    global_step,
    gradient_norm,
    optimizer_updated,
    amp_scale,
    original_seen,
    correction_seen,
    sampling_counts,
    stationary_telemetry=None,
    stationary_telemetry_cumulative=None,
):
    source = batch["sample.source"]
    index = batch["sample.index"]
    correction_mask = source == 1
    original_mask = ~correction_mask
    group_loss = {}
    model.eval()
    with torch.no_grad():
        for name, mask in (
            ("original", original_mask),
            ("correction", correction_mask),
        ):
            if not bool(mask.any()):
                group_loss[name] = None
                continue
            with torch.autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda",
                dtype=torch.float16,
            ):
                group_loss[name] = float(
                    _loss(model, _select_batch(batch, mask)).detach()
                )
        generator = torch.Generator(device=device).manual_seed(
            args.validation_seed + global_step
        )
        actions = model.sample(
            batch["observation.image"],
            batch["observation.state"],
            batch["observation.oct"],
            batch["observation.velocity"],
            batch["observation.previous_action"],
            generator=generator,
        )
        action_error = torch.abs(
            actions[:, 0] - batch["action"][:, 0]
        )
        distance_um = torch.linalg.norm(
            batch["observation.oct"][:, -1, :3], dim=-1
        ) * 1000.0
        buckets = {
            "<30": distance_um < 30.0,
            "30-100": (distance_um >= 30.0) & (distance_um < 100.0),
            "100-300": (distance_um >= 100.0) & (distance_um < 300.0),
            "300-1000": (distance_um >= 300.0) & (distance_um < 1000.0),
            ">=1000": distance_um >= 1000.0,
        }
        bucket_action_error = {
            name: (
                action_error[mask].mean().tolist()
                if bool(mask.any())
                else None
            )
            for name, mask in buckets.items()
        }
    model.train()
    return {
        "step": int(global_step),
        "batch_correction_ratio": float(correction_mask.float().mean()),
        "cumulative_correction_ratio": (
            sampling_counts["correction"]
            / (
                sampling_counts["original"]
                + sampling_counts["correction"]
            )
        ),
        "original_unique_seen": len(original_seen),
        "correction_unique_seen": len(correction_seen),
        "original_loss": group_loss["original"],
        "correction_loss": group_loss["correction"],
        "t3_stationary_telemetry": stationary_telemetry,
        "t3_stationary_telemetry_cumulative": (
            stationary_telemetry_cumulative
        ),
        "distance_bucket_first_action_abs_error": bucket_action_error,
        "first_action_mean_abs_error_deg_deg_mm": (
            action_error.mean(dim=0).tolist()
        ),
        "first_action_limited_fraction": float(
            torch.any(
                torch.abs(actions[:, 0])
                > torch.as_tensor(ACTION_LIMIT, device=actions.device),
                dim=-1,
            ).float().mean()
        ),
        "local_oct_gradient_norm_after_clip": (
            float(torch.linalg.vector_norm(torch.stack([
                parameter.grad.detach().float().norm()
                for name, parameter in model.named_parameters()
                if name.startswith("local_oct_mlp.") and parameter.grad is not None
            ]))) if getattr(model, "use_local_oct_branch", False) else None
        ),
        "gradient_norm_before_clip": float(gradient_norm),
        "amp_optimizer_updated": bool(optimizer_updated),
        "amp_scale_after_update": float(amp_scale),
    }


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def train(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--correction-data-dir",
        type=Path,
        help="Optional independent correction dataset; all valid windows enter its pool",
    )
    parser.add_argument("--original-max-episode", type=int, default=None)
    parser.add_argument("--val-data-dir", type=Path, required=True)
    parser.add_argument("--init-from", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("original", "static", "balanced", "ratio"),
        required=True,
    )
    parser.add_argument(
        "--correction-ratio",
        type=float,
        default=-1.0,
        help="Correction sampling probability; negative uses the natural ratio",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument(
        "--total-batches",
        type=int,
        default=None,
        help="Optimizer batches across the run; defaults to epochs * steps-per-epoch",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--validation-max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--attention-lr", type=float, default=1e-5)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine"),
        default="constant",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=0.0,
        help="Cosine annealing floor; ignored for constant schedules",
    )
    parser.add_argument(
        "--near-target-um",
        type=float,
        default=None,
        help="Apply near-target correction weighting at or below this truth error",
    )
    parser.add_argument("--near-target-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--correction-source", default="teacher")
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--validation-seed", type=int, default=20261119)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--checkpoint-every-steps", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--require-temporal-bias-only",
        action="store_true",
        help=(
            "Fail unless backbone_lr=0 and every trainable parameter belongs "
            "to temporal_bias"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--oct-symlog", action="store_true")
    parser.add_argument("--use-local-oct-branch", action="store_true")
    parser.add_argument("--label-overlay", type=Path)
    parser.add_argument("--oct-overlay", type=Path)
    parser.add_argument("--c-label-overlay", type=Path)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)
    if args.label_overlay is not None and args.c_label_overlay is not None:
        parser.error("use either legacy label-overlay or composed c-label-overlay")
    if args.output_dir.exists() and not args.resume:
        parser.error("output directory already exists")
    if min(
        args.epochs,
        args.steps_per_epoch,
        args.batch_size,
        args.val_batch_size,
    ) < 1 or args.num_workers < 0:
        parser.error("counts must be positive and workers nonnegative")
    if (
        args.validation_max_batches is not None
        and args.validation_max_batches < 1
    ):
        parser.error("validation max batches must be positive")
    if args.batch_size % 2:
        parser.error("batch size must be even")
    if args.total_batches is None:
        args.total_batches = args.epochs * args.steps_per_epoch
    if args.total_batches < 1:
        parser.error("total batches must be positive")
    if args.correction_ratio > 1.0:
        parser.error("correction ratio must be at most 1 or negative for natural")
    if min(args.log_every_steps, args.checkpoint_every_steps) < 1:
        parser.error("logging and checkpoint intervals must be positive")
    if min(
        args.backbone_lr,
        args.attention_lr,
        args.weight_decay,
    ) < 0:
        parser.error("learning rates and weight decay must be nonnegative")
    if not math.isfinite(args.lr_min) or args.lr_min < 0:
        parser.error("lr-min must be finite and nonnegative")
    if args.near_target_weight < 0 or not math.isfinite(args.near_target_weight):
        parser.error("near-target weight must be finite and nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu only for a smoke run")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    transform = image_transform_for_dataset(args.data_dir)
    combined = RCMDataset(
        args.data_dir,
        transform=transform,
        n_obs_steps=2,
        n_action_steps=16,
        include_velocity=True,
        include_previous_action=True,
    )
    validation = RCMDataset(
        args.val_data_dir,
        transform=transform,
        n_obs_steps=combined.n_obs_steps,
        n_action_steps=combined.n_action_steps,
        include_velocity=True,
        include_previous_action=True,
    )
    from lerobot_training.navigation_overlay import apply_dataset_overlays

    overlay_receipt = apply_dataset_overlays(
        combined, oct_overlay=args.oct_overlay, c_overlay=args.c_label_overlay,
    )
    correction_weight_receipt = {
        "source": "not_applicable",
        "window_count": 0,
        "near_target_windows": 0,
    }
    correction_weights = near_target_mask = None
    correction_owner = combined
    stationary_window_counts = np.zeros(0, dtype=np.int64)
    stationary_pool_receipt = {
        "available": False,
        "window_count": 0,
        "stationary_window_count": 0,
        "stationary_action_frames": 0,
        "action_frames": 0,
        "t3_stationary_weight": None,
    }
    if args.correction_data_dir is not None:
        correction_owner = RCMDataset(
            args.correction_data_dir,
            transform=transform,
            n_obs_steps=combined.n_obs_steps,
            n_action_steps=combined.n_action_steps,
            include_velocity=True,
            include_previous_action=True,
        )
        all_original = np.arange(len(combined), dtype=np.int64)
        if args.original_max_episode is not None:
            rows = np.asarray(combined.valid_indices, dtype=np.int64)
            all_original = np.flatnonzero(
                combined.episode_indices[rows] <= args.original_max_episode
            )
        original_indices = all_original
        correction_indices = np.arange(len(correction_owner), dtype=np.int64)
        correction_weights, near_target_mask, correction_weight_receipt = (
            load_correction_window_weights(
                args.correction_data_dir,
                np.asarray(correction_owner.valid_indices, dtype=np.int64),
                near_target_um=args.near_target_um,
            )
        )
        stationary_window_counts, stationary_pool_receipt = (
            load_stationary_window_counts(
                args.correction_data_dir,
                np.asarray(correction_owner.valid_indices, dtype=np.int64),
                action_steps=combined.n_action_steps,
            )
        )
    else:
        original_indices, correction_indices = split_experience_indices(
            combined, args.correction_source
        )
    if args.label_overlay is not None:
        from scripts.research.overnight_cfix_data import apply_overlay

        combined.actions = apply_overlay(
            args.data_dir, combined.actions, args.label_overlay
        )
    if not len(original_indices):
        raise ValueError("no original action windows after source split")
    if args.mode != "original" and not len(correction_indices):
        raise ValueError(
            "static/balanced modes require expert correction windows"
        )
    original = IndexSubset(combined, original_indices, source=0)
    correction = IndexSubset(
        correction_owner,
        correction_indices,
        source=1,
        stationary_frame_counts=(
            stationary_window_counts[
                np.asarray(correction_indices, dtype=np.int64)
            ]
            if len(stationary_window_counts)
            else np.zeros(len(correction_indices), dtype=np.int64)
        ),
        correction_weights=(
            correction_weights[
                np.asarray(correction_indices, dtype=np.int64)
            ]
            if correction_weights is not None
            else np.ones(len(correction_indices), dtype=np.float64)
        ),
    )
    natural_correction_ratio = len(correction_indices) / (
        len(original_indices) + len(correction_indices)
    )
    correction_ratio = (
        natural_correction_ratio
        if args.correction_ratio < 0
        else args.correction_ratio
    )
    if args.mode == "original":
        training = original
    else:
        training = ConcatDataset([original, correction])
    if args.mode == "ratio" and not 0.0 < correction_ratio < 1.0:
        raise ValueError(
            "ratio mode requires a correction probability in (0, 1)"
        )

    resume_path = args.output_dir / "resume_state.pth"
    if args.output_dir.exists():
        if not args.resume or not resume_path.is_file():
            raise FileExistsError(
                f"non-resumable training output directory: {args.output_dir}"
            )
    else:
        args.output_dir.mkdir(parents=True)
    configure_stream_logging(args.output_dir)
    model, initial_config, initial_epoch = _load_policy(
        args.init_from,
        device,
        oct_symlog_enabled=args.oct_symlog,
        use_local_oct_branch=args.use_local_oct_branch,
    )
    assert_temporal_bias_only(model, args, freeze=True)
    optimizer = _optimizer(model, args)
    assert_temporal_bias_only(model, args)
    if bool(getattr(args, "require_temporal_bias_only", False)):
        non_temporal = [
            name
            for name, _ in model.named_parameters()
            if "temporal_bias" not in name
        ]
        if any(
            any(
                parameter is model.get_parameter(name)
                for group in optimizer.param_groups
                for parameter in group["params"]
            )
            for name in non_temporal
        ):
            raise ValueError(
                "strict frozen-backbone optimizer contains non-temporal_bias "
                "parameters"
            )
    scheduler = _scheduler(optimizer, args, args.total_batches)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and device.type == "cuda",
    )
    val_loader = _validation_loader(args, validation)
    run_config = {
        **initial_config,
        **{
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "policy_type": model.policy_type,
        "architecture": attention.ARCHITECTURE,
        "training_module": TRAINING_MODULE,
        "objective": "rectified_flow_only",
        "reward_in_loss": False,
        "history_action_source": "executed_action",
        "supervision_action_source": args.correction_source,
        "initial_checkpoint_epoch": initial_epoch,
        "oct_symlog_enabled": bool(args.oct_symlog),
        "use_local_oct_branch": bool(args.use_local_oct_branch),
        "local_oct_tanh_scale_mm": 0.05,
        "local_oct_log_scale_mm": 0.005,
        "local_oct_log_reference_mm": 0.5,
        "dagger_repeat": 1,
        "overlay_receipt": overlay_receipt,
        "correction_weight_receipt": correction_weight_receipt,
        "freeze_config": {
            "backbone_frozen": args.backbone_lr == 0,
            "backbone_lr": args.backbone_lr,
            "attention_lr": args.attention_lr,
            "trainable_parameters": [
                name for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ],
            "frozen_parameters": [
                name for name, parameter in model.named_parameters()
                if not parameter.requires_grad
            ],
        },
        "lr_schedule": args.lr_schedule,
        "lr_min": args.lr_min,
        "require_temporal_bias_only": bool(
            getattr(args, "require_temporal_bias_only", False)
        ),
        "total_batches": args.total_batches,
        "near_target_um": args.near_target_um,
        "near_target_weight": args.near_target_weight,
    }
    manifest = {
        "training_module": TRAINING_MODULE,
        "mode": args.mode,
        "overlay_receipt": overlay_receipt,
        "initial_checkpoint": str(args.init_from),
        "initial_checkpoint_epoch": initial_epoch,
        "original_windows": len(original),
        "correction_windows": len(correction),
        "natural_correction_ratio": natural_correction_ratio,
        "correction_ratio": correction_ratio,
        "correction_weight_receipt": correction_weight_receipt,
        "t3_stationary_sampling_pool": stationary_pool_receipt,
        "correction_sample_weight": (
            {
                "path": str(
                    args.correction_data_dir / "data" / "label_repair_overlay.npz"
                ),
                "near_target_um": args.near_target_um,
                "near_target_weight": args.near_target_weight,
            }
            if correction_weights is not None
            else None
        ),
        "batch_telemetry_fields": [
            "sample.stationary_frame_count",
            "sample.correction_weight",
        ],
        "freeze_config": run_config["freeze_config"],
        "lr_schedule": args.lr_schedule,
        "lr_min": args.lr_min,
        "total_batches": args.total_batches,
        "valid_window_index_sha256": _index_sha256(
            combined.valid_indices
        ),
        "original_index_sha256": _index_sha256(original_indices),
        "correction_index_sha256": _index_sha256(correction_indices),
        "window_diagnostics": combined.window_diagnostics,
        "validation_windows": len(validation),
        "validation_index_sha256": _index_sha256(
            validation.valid_indices
        ),
        "batch_size": args.batch_size,
        "batch_mix": (
            {
                "correction_probability": correction_ratio,
            }
            if args.mode == "ratio"
            else {
                "original": args.batch_size // 2,
                "correction": args.batch_size // 2,
            }
            if args.mode == "balanced"
            else {
                "uniform_over": (
                    "original"
                    if args.mode == "original"
                    else "original+correction"
                )
            }
        ),
        "sampling": (
            "fixed_original_correction_ratio_without_replacement_quota;"
            "correction_pool_weighted_with_replacement"
            if args.mode == "ratio"
            else "uniform_with_replacement"
        ),
        "steps_per_epoch": args.steps_per_epoch,
        "t3_stationary_sampling_pool": stationary_pool_receipt,
        "t3_stationary_telemetry": [],
        "epochs": args.epochs,
        "total_batches": args.total_batches,
        "flow_loss_weights": {
            "original": (
                0.5 if args.mode == "balanced" else None
            ),
            "correction": (
                0.5 if args.mode == "balanced" else None
            ),
        },
    }
    _write_json(args.output_dir / "training_manifest.json", manifest)
    _write_json(args.output_dir / "training_config.json", run_config)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)

    history, best, started = [], float("inf"), time.monotonic()
    metrics_path = args.output_dir / "training_metrics.jsonl"
    metrics_stream = metrics_path.open(
        "a" if resume_path.is_file() else "w", encoding="utf-8"
    )
    original_seen, correction_seen = set(), set()
    sampling_counts = {"original": 0, "correction": 0}
    empty_stationary_counts = {
        "correction_samples": 0,
        "correction_action_frames": 0,
        "t3_stationary_action_frames": 0,
        "stationary_windows": 0,
        "total_samples": 0,
    }
    epoch_stationary_counts = dict(empty_stationary_counts)
    cumulative_stationary_counts = dict(empty_stationary_counts)
    global_step = optimizer_updates = 0
    resumed_rng = None
    if resume_path.is_file():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("training_module") != TRAINING_MODULE:
            raise ValueError("resume state belongs to another training module")
        if saved.get("dataset_sha256") != file_sha256(
            args.data_dir / "data" / "dataset.npz"
        ):
            raise ValueError("resume source dataset changed")
        if args.correction_data_dir is not None and saved.get(
            "correction_dataset_sha256"
        ) != file_sha256(
            args.correction_data_dir / "data" / "dataset.npz"
        ):
            raise ValueError("resume correction dataset changed")
        if saved.get("sampling_contract", {}).get(
            "correction_ratio"
        ) != correction_ratio:
            raise ValueError("resume correction ratio differs")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scaler.load_state_dict(saved["scaler_state_dict"])
        if scheduler is not None:
            if saved.get("scheduler_state_dict") is None:
                raise ValueError("resume checkpoint lacks scheduler state")
            scheduler.load_state_dict(saved["scheduler_state_dict"])
        progress = saved["progress"]
        global_step = int(progress["global_step"])
        optimizer_updates = int(progress["optimizer_updates"])
        history = list(progress.get("history", []))
        best = float(progress.get("best", float("inf")))
        sampling_counts = {
            "original": int(progress["sampling_counts"][0]),
            "correction": int(progress["sampling_counts"][1]),
        }
        epoch_stationary_counts = {
            key: int(value)
            for key, value in progress.get(
                "epoch_stationary_counts", empty_stationary_counts
            ).items()
        }
        cumulative_stationary_counts = {
            key: int(value)
            for key, value in progress.get(
                "cumulative_stationary_counts", empty_stationary_counts
            ).items()
        }
        original_seen = set(progress.get("original_seen", []))
        correction_seen = set(progress.get("correction_seen", []))
        resumed_rng = saved["rng_state"]
        started -= float(progress.get("elapsed_s", 0.0))
        print(
            f"Resume exactly after batch {global_step}/{args.total_batches}",
            flush=True,
        )
        del saved

    def sampling_contract():
        return {
            "mode": args.mode,
            "correction_ratio": correction_ratio,
            "near_target_um": args.near_target_um,
            "near_target_weight": args.near_target_weight,
            "correction_weight_receipt": correction_weight_receipt,
        }

    def save_checkpoint(step, epoch, train_loss, val_loss):
        nonlocal best
        improved = val_loss is not None and val_loss < best
        if improved:
            best = float(val_loss)
        payload = {
            "schema_version": 2,
            "policy_type": model.policy_type,
            "architecture": attention.ARCHITECTURE,
            "training_module": TRAINING_MODULE,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "scaler_state_dict": scaler.state_dict(),
            "rng_state": _rng_state(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": run_config,
            "dataset_sha256": file_sha256(
                args.data_dir / "data" / "dataset.npz"
            ),
            "correction_dataset_sha256": (
                file_sha256(
                    args.correction_data_dir / "data" / "dataset.npz"
                )
                if args.correction_data_dir is not None
                else None
            ),
            "sampling_contract": sampling_contract(),
            "freeze_config": run_config["freeze_config"],
            "progress": {
                "global_step": step,
                "optimizer_updates": optimizer_updates,
                "history": history,
                "best": best,
                "sampling_counts": [
                    sampling_counts["original"],
                    sampling_counts["correction"],
                ],
                "epoch_stationary_counts": dict(epoch_stationary_counts),
                "cumulative_stationary_counts": dict(
                    cumulative_stationary_counts
                ),
                "original_seen": sorted(original_seen),
                "correction_seen": sorted(correction_seen),
                "elapsed_s": time.monotonic() - started,
            },
        }
        _atomic_torch_save(payload, args.output_dir / "resume_state.pth")
        _atomic_torch_save(payload, args.output_dir / "last_model.pth")
        _atomic_torch_save(
            payload, args.output_dir / f"step_{step:06d}.pth"
        )
        if improved:
            _atomic_torch_save(payload, args.output_dir / "best_model.pth")

    last_train_loss = float("inf")
    while global_step < args.total_batches:
        epoch = global_step // args.steps_per_epoch
        batch_offset = global_step % args.steps_per_epoch
        if batch_offset == 0:
            epoch_stationary_counts = dict(empty_stationary_counts)
        loader = _train_loader(
            args,
            training,
            args.mode,
            len(original),
            len(correction),
            correction_ratio,
            epoch,
            correction_weights=correction_weights,
            near_target_mask=near_target_mask,
        )
        model.train()
        epoch_loss = 0.0
        epoch_count = 0
        for batch_index, batch in enumerate(tqdm(
            loader,
            total=args.steps_per_epoch,
            desc=f"Experience replay epoch {epoch}",
        )):
            if batch_index < batch_offset:
                continue
            if global_step >= args.total_batches:
                break
            if resumed_rng is not None:
                _restore_rng_state(resumed_rng)
                resumed_rng = None
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda",
                dtype=torch.float16,
            ):
                loss = _loss(model, batch)
            loss_value = float(loss.detach())
            if not math.isfinite(loss_value):
                raise FloatingPointError(
                    f"nonfinite loss at batch {global_step + 1}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters()
                 if parameter.requires_grad],
                1.0,
            )
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after_step = scaler.get_scale()
            optimizer_updated = (
                not (args.amp and device.type == "cuda")
                or scale_after_step >= scale_before_step
            )
            if not math.isfinite(float(gradient_norm)) and optimizer_updated:
                raise FloatingPointError(
                    f"nonfinite gradient at batch {global_step + 1}"
                )
            if optimizer_updated and scheduler is not None:
                scheduler.step()
            optimizer_updates += int(optimizer_updated)
            epoch_loss += loss_value
            last_train_loss = loss_value
            epoch_count += 1
            global_step += 1
            source = batch["sample.source"]
            index = batch["sample.index"]
            original_seen.update(
                int(value)
                for value in index[source == 0].cpu().tolist()
            )
            correction_seen.update(
                int(value)
                for value in index[source == 1].cpu().tolist()
            )
            sampling_counts["original"] += int((source == 0).sum())
            sampling_counts["correction"] += int((source == 1).sum())
            epoch_stationary_counts["total_samples"] += int(len(source))
            cumulative_stationary_counts["total_samples"] += int(len(source))
            correction_positions = index[source == 1].cpu().numpy()
            if len(correction_positions):
                if len(stationary_window_counts):
                    selected_stationary = (
                        batch["sample.stationary_frame_count"][source == 1]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    expected_stationary = stationary_window_counts[
                        np.asarray(
                            correction_indices[correction_positions],
                            dtype=np.int64,
                        )
                    ]
                    if not np.array_equal(
                        selected_stationary, expected_stationary
                    ):
                        raise ValueError(
                            "training batch stationary-count telemetry mismatch"
                        )
                    epoch_stationary_counts["correction_samples"] += int(
                        len(correction_positions)
                    )
                    epoch_stationary_counts["correction_action_frames"] += int(
                        len(correction_positions) * combined.n_action_steps
                    )
                    epoch_stationary_counts["t3_stationary_action_frames"] += int(
                        selected_stationary.sum()
                    )
                    epoch_stationary_counts["stationary_windows"] += int(
                        np.count_nonzero(selected_stationary)
                    )
                    cumulative_stationary_counts["correction_samples"] += int(
                        len(correction_positions)
                    )
                    cumulative_stationary_counts["correction_action_frames"] += int(
                        len(correction_positions) * combined.n_action_steps
                    )
                    cumulative_stationary_counts[
                        "t3_stationary_action_frames"
                    ] += int(selected_stationary.sum())
                    cumulative_stationary_counts["stationary_windows"] += int(
                        np.count_nonzero(selected_stationary)
                    )
            if global_step % args.log_every_steps == 0:
                stationary_telemetry = stationary_sampling_summary(
                    epoch_stationary_counts,
                    weights=correction_weights,
                    stationary_counts=stationary_window_counts,
                    correction_ratio=correction_ratio,
                    action_steps=combined.n_action_steps,
                )
                cumulative_stationary_telemetry = stationary_sampling_summary(
                    cumulative_stationary_counts,
                    weights=correction_weights,
                    stationary_counts=stationary_window_counts,
                    correction_ratio=correction_ratio,
                    action_steps=combined.n_action_steps,
                )
                metrics_stream.write(
                    json.dumps(
                        _log_metrics(
                            model,
                            batch,
                            device,
                            args,
                            global_step,
                            float(gradient_norm),
                            optimizer_updated,
                            scale_after_step,
                            original_seen,
                            correction_seen,
                            sampling_counts,
                            stationary_telemetry,
                            cumulative_stationary_telemetry,
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                metrics_stream.flush()
            if global_step % args.checkpoint_every_steps == 0:
                save_checkpoint(
                    global_step, epoch,
                    epoch_loss / max(epoch_count, 1), None,
                )
                print(
                    json.dumps({
                        "step": global_step,
                        "epoch": epoch,
                        "lr_groups": [
                            group["lr"] for group in optimizer.param_groups
                        ],
                        "checkpoint": str(
                            args.output_dir / f"step_{global_step:06d}.pth"
                        ),
                    }),
                    flush=True,
                )
        completed_epoch = (
            batch_offset + epoch_count == args.steps_per_epoch
        )
        if completed_epoch:
            train_loss = epoch_loss / max(epoch_count, 1)
            val_loss = _validate(model, val_loader, device, args)
            if not math.isfinite(val_loss):
                raise FloatingPointError(
                    f"nonfinite validation loss at epoch {epoch}"
                )
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "steps": epoch_count,
                "batches_seen": global_step,
                "optimizer_updates": optimizer_updates,
                "amp_skipped_batches": global_step - optimizer_updates,
                "elapsed_s": time.monotonic() - started,
                "sampling_counts": dict(sampling_counts),
                "t3_stationary_telemetry": stationary_sampling_summary(
                    epoch_stationary_counts,
                    weights=correction_weights,
                    stationary_counts=stationary_window_counts,
                    correction_ratio=correction_ratio,
                    action_steps=combined.n_action_steps,
                ),
                "t3_stationary_telemetry_cumulative": (
                    stationary_sampling_summary(
                        cumulative_stationary_counts,
                        weights=correction_weights,
                        stationary_counts=stationary_window_counts,
                        correction_ratio=correction_ratio,
                        action_steps=combined.n_action_steps,
                    )
                ),
            }
            history.append(row)
            _write_json(args.output_dir / "training_history.json", history)
            save_checkpoint(global_step, epoch, train_loss, val_loss)
            print(json.dumps(row), flush=True)
        elif global_step < args.total_batches:
            raise RuntimeError(
                "training data loader ended before the configured batch count"
            )
    if global_step % args.checkpoint_every_steps:
        save_checkpoint(
            global_step,
            global_step // args.steps_per_epoch,
            last_train_loss,
            None,
        )
    _write_json(
        args.output_dir / "training_receipt.json",
        {
            "status": "completed",
            "total_batches": global_step,
            "optimizer_updates": optimizer_updates,
            "sampling_counts": sampling_counts,
            "sampling_contract": sampling_contract(),
            "t3_stationary_sampling_pool": stationary_pool_receipt,
            "t3_stationary_telemetry": [
                row["t3_stationary_telemetry"] for row in history
            ],
            "t3_stationary_telemetry_cumulative": (
                stationary_sampling_summary(
                    cumulative_stationary_counts,
                    weights=correction_weights,
                    stationary_counts=stationary_window_counts,
                    correction_ratio=correction_ratio,
                    action_steps=combined.n_action_steps,
                )
            ),
            "freeze_config": run_config["freeze_config"],
            "scheduler": (
                scheduler.state_dict() if scheduler is not None else None
            ),
        },
    )
    manifest["t3_stationary_telemetry"] = [
        row["t3_stationary_telemetry"] for row in history
    ]
    manifest["t3_stationary_telemetry_cumulative"] = (
        stationary_sampling_summary(
            cumulative_stationary_counts,
            weights=correction_weights,
            stationary_counts=stationary_window_counts,
            correction_ratio=correction_ratio,
            action_steps=combined.n_action_steps,
        )
    )
    _write_json(args.output_dir / "training_manifest.json", manifest)
    metrics_stream.close()
    return 0


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "eval"))
    mode = parser.parse_args(arguments[:1]).mode
    if mode == "train":
        return train(arguments[1:])
    evaluation_parser = argparse.ArgumentParser(add_help=False)
    evaluation_parser.add_argument("--model", type=Path)
    options, _ = evaluation_parser.parse_known_args(arguments[1:])
    if options.model is not None:
        checkpoint = torch.load(
            options.model, map_location="cpu", weights_only=False
        )
        if checkpoint.get("training_module") != TRAINING_MODULE:
            raise ValueError(
                "evaluation requires an experience-replay Flow checkpoint"
            )
        del checkpoint
    import dit_rectified_flow as flow

    flow.DiTRectifiedFlowPolicy = attention.StateConditionedDiTRectifiedFlowPolicy
    import eval_policy

    sys.argv = [eval_policy.__file__, *arguments[1:]]
    return eval_policy.main()


if __name__ == "__main__":
    raise SystemExit(main())
