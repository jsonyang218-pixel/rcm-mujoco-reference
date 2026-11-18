#!/usr/bin/env python3
"""State-attention DiT-Flow training with expert-corrected experience replay.

The experiment keeps the existing Flow-matching objective and action-block
contract. It does not train a critic and does not use reward in the loss.
"""
from __future__ import annotations

import argparse
import json
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
    from .train_diffusion_policy import (
        RCMDataset,
        configure_stream_logging,
        image_transform_for_dataset,
        seed_worker,
    )
    from .train_dit_rectified_flow import _atomic_torch_save, _loss, _move_batch
except ImportError:
    import dit_state_attention as attention
    from train_diffusion_policy import (
        RCMDataset,
        configure_stream_logging,
        image_transform_for_dataset,
        seed_worker,
    )
    from train_dit_rectified_flow import _atomic_torch_save, _loss, _move_batch


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

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1:
            raise ValueError("subset indices must be one-dimensional")
        if len(self.indices) and (
            self.indices.min() < 0 or self.indices.max() >= len(dataset)
        ):
            raise IndexError("subset index outside source dataset")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[int(self.indices[int(index)])]


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


def split_experience_indices(dataset, correction_source="teacher"):
    """Split valid action windows by the episode's supervision source."""
    episodes = {
        int(item["episode_index"]): item.get("action_source", "executed")
        for item in dataset.metadata.get("episodes", [])
    }
    original, correction = [], []
    for sample_index, row in enumerate(dataset.valid_indices):
        episode = int(dataset.episode_indices[int(row)])
        source = episodes.get(episode)
        if source == correction_source:
            correction.append(sample_index)
        else:
            original.append(sample_index)
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


def _train_loader(args, dataset, mode, original_count, correction_count, epoch):
    if mode == "balanced":
        sampler = BalancedReplacementBatchSampler(
            original_count,
            correction_count,
            args.steps_per_epoch,
            args.batch_size,
            args.seed + epoch,
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


def _load_policy(init_from, device):
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
    ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    return policy, config, int(checkpoint.get("epoch", -1))


def _optimizer(model, args):
    backbone, attention_parameters = [], []
    for name, parameter in model.named_parameters():
        (
            attention_parameters
            if ".temporal_bias." in name
            else backbone
        ).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": args.backbone_lr},
            {"params": attention_parameters, "lr": args.attention_lr},
        ],
        weight_decay=args.weight_decay,
    )


def _validate(model, loader, device, args):
    model.eval()
    total, count = 0.0, 0
    autocast = args.amp and device.type == "cuda"
    with torch.no_grad():
        for batch in tqdm(loader, desc="Experience replay validation"):
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
    parser.add_argument("--val-data-dir", type=Path, required=True)
    parser.add_argument("--init-from", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("original", "static", "balanced"),
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--validation-max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--attention-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--correction-source", default="teacher")
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)
    if args.output_dir.exists():
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
    if min(
        args.backbone_lr,
        args.attention_lr,
        args.weight_decay,
    ) < 0:
        parser.error("learning rates and weight decay must be nonnegative")
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
    original_indices, correction_indices = split_experience_indices(
        combined, args.correction_source
    )
    if not len(original_indices):
        raise ValueError("no original action windows after source split")
    if args.mode != "original" and not len(correction_indices):
        raise ValueError(
            "static/balanced modes require expert correction windows"
        )
    original = IndexSubset(combined, original_indices)
    correction = IndexSubset(combined, correction_indices)
    if args.mode == "original":
        training = original
    else:
        training = ConcatDataset([original, correction])

    args.output_dir.mkdir(parents=True)
    configure_stream_logging(args.output_dir)
    model, initial_config, initial_epoch = _load_policy(
        args.init_from, device
    )
    optimizer = _optimizer(model, args)
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
    }
    manifest = {
        "training_module": TRAINING_MODULE,
        "mode": args.mode,
        "initial_checkpoint": str(args.init_from),
        "initial_checkpoint_epoch": initial_epoch,
        "original_windows": len(original),
        "correction_windows": len(correction),
        "validation_windows": len(validation),
        "batch_size": args.batch_size,
        "batch_mix": (
            {
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
        "sampling": "uniform_with_replacement",
        "steps_per_epoch": args.steps_per_epoch,
        "epochs": args.epochs,
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
    for epoch in range(args.epochs):
        model.train()
        loader = _train_loader(
            args,
            training,
            args.mode,
            len(original),
            len(correction),
            epoch,
        )
        total, count = 0.0, 0
        for batch in tqdm(
            loader,
            total=args.steps_per_epoch,
            desc=f"Experience replay epoch {epoch}",
        ):
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda",
                dtype=torch.float16,
            ):
                loss = _loss(model, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            count += 1
        train_loss = total / max(count, 1)
        val_loss = _validate(model, val_loader, device, args)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "steps": count,
            "elapsed_s": time.monotonic() - started,
        }
        history.append(row)
        _write_json(args.output_dir / "training_history.json", history)
        payload = {
            "schema_version": 1,
            "policy_type": model.policy_type,
            "architecture": attention.ARCHITECTURE,
            "training_module": TRAINING_MODULE,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": run_config,
        }
        _atomic_torch_save(payload, args.output_dir / "last_model.pth")
        if val_loss < best:
            best = val_loss
            _atomic_torch_save(payload, args.output_dir / "best_model.pth")
        print(json.dumps(row), flush=True)
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
