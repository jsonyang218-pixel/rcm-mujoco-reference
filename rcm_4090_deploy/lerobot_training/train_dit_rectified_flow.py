"""Train a DiT action policy with rectified-flow matching."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

try:
    from .dit_rectified_flow import DiTRectifiedFlowPolicy
    from .train_diffusion_policy import (
        RCMDataset,
        configure_stream_logging,
        image_transform_for_dataset,
        seed_worker,
    )
except ImportError:
    from dit_rectified_flow import DiTRectifiedFlowPolicy
    from train_diffusion_policy import (
        RCMDataset,
        configure_stream_logging,
        image_transform_for_dataset,
        seed_worker,
    )


def _atomic_torch_save(payload, destination):
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def _motion_stats(dataset):
    velocities, previous_actions = dataset.motion_samples()
    if len(velocities) == 0:
        raise ValueError("dataset lacks within-episode motion samples")
    return velocities, previous_actions


def _dataset_statistics(dataset):
    velocities, previous_actions = _motion_stats(dataset)
    if not np.any(dataset.oct_navigation_valid):
        raise ValueError("dataset has no valid OCT samples")
    oct_values = np.concatenate(
        [
            dataset.oct_navigation,
            dataset.oct_navigation_valid[:, None].astype(np.float32),
        ],
        axis=1,
    )
    return {
        "state_mean": dataset.states.mean(axis=0),
        "state_std": np.maximum(dataset.states.std(axis=0), 1e-6),
        "oct_mean": oct_values.mean(axis=0),
        "oct_std": np.maximum(oct_values.std(axis=0), 1e-6),
        "action_mean": dataset.actions.mean(axis=0),
        "action_std": np.maximum(dataset.actions.std(axis=0), 1e-6),
        "velocity_mean": velocities.mean(axis=0),
        "velocity_std": np.maximum(velocities.std(axis=0), 1e-6),
        "previous_action_mean": previous_actions.mean(axis=0),
        "previous_action_std": np.maximum(
            previous_actions.std(axis=0), 1e-6
        ),
    }


def _move_batch(batch, device):
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
    }


def _loss(model, batch):
    return model.rectified_flow_loss(
        batch["observation.image"],
        batch["observation.state"],
        batch["observation.oct"],
        batch["observation.velocity"],
        batch["observation.previous_action"],
        batch["action"],
    )


def _sampling_weights(dataset, dagger_start, dagger_repeat):
    if dagger_start < 0:
        return None
    episodes = dataset.episode_indices[dataset.valid_indices]
    return torch.as_tensor(
        np.where(
            episodes >= int(dagger_start),
            float(dagger_repeat),
            1.0,
        ),
        dtype=torch.double,
    )


def train_epoch(model, loader, optimizer, scaler, device, epoch):
    model.train()
    total = 0.0
    count = 0
    for batch in tqdm(loader, desc=f"DiT-RF epoch {epoch}"):
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = _loss(model, batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach())
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def validate_epoch(model, loader, device, epoch):
    model.eval()
    total = 0.0
    count = 0
    for batch in tqdm(loader, desc=f"DiT-RF validation {epoch}"):
        batch = _move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = _loss(model, batch)
        total += float(loss.detach())
        count += 1
    return total / max(count, 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--val-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--flow-integration-steps", type=int, default=4)
    parser.add_argument("--dit-dim", type=int, default=384)
    parser.add_argument("--dit-depth", type=int, default=6)
    parser.add_argument("--dit-heads", type=int, default=6)
    parser.add_argument("--dagger-episode-start", type=int, default=-1)
    parser.add_argument("--dagger-repeat", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261116)
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
        args.batch_size,
        args.val_batch_size,
        args.n_obs_steps,
        args.n_action_steps,
        args.flow_integration_steps,
        args.dit_dim,
        args.dit_depth,
        args.dit_heads,
    ) < 1:
        parser.error("counts and dimensions must be positive")
    if args.dagger_repeat < 1:
        parser.error("--dagger-repeat must be at least 1")
    args.output_dir.mkdir(parents=True)
    configure_stream_logging(args.output_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    transform = image_transform_for_dataset(str(args.data_dir))
    train_dataset = RCMDataset(
        args.data_dir,
        transform=transform,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        include_velocity=True,
        include_previous_action=True,
    )
    val_dataset = RCMDataset(
        args.val_data_dir,
        transform=transform,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        include_velocity=True,
        include_previous_action=True,
    )
    stats = _dataset_statistics(train_dataset)
    generator = torch.Generator().manual_seed(args.seed)
    weights = _sampling_weights(
        train_dataset,
        args.dagger_episode_start,
        args.dagger_repeat,
    )
    sampler = (
        None
        if weights is None
        else WeightedRandomSampler(
            weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=min(args.num_workers, 4),
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
    )
    device = torch.device("cuda")
    model = DiTRectifiedFlowPolicy(
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        dit_dim=args.dit_dim,
        dit_depth=args.dit_depth,
        dit_heads=args.dit_heads,
        flow_integration_steps=args.flow_integration_steps,
        **stats,
    ).to(device)
    if args.init_from is not None:
        initial_checkpoint = torch.load(
            args.init_from,
            map_location="cpu",
            weights_only=False,
        )
        if (
            initial_checkpoint.get("policy_type")
            != DiTRectifiedFlowPolicy.policy_type
        ):
            raise ValueError(
                "initial checkpoint is not a DiTRectifiedFlowPolicy"
            )
        initial_config = initial_checkpoint.get("config", {})
        for name in (
            "n_obs_steps",
            "n_action_steps",
            "dit_dim",
            "dit_depth",
            "dit_heads",
        ):
            if int(initial_config.get(name, -1)) != int(
                getattr(args, name)
            ):
                raise ValueError(
                    f"initial checkpoint {name} does not match"
                )
        model.load_state_dict(
            initial_checkpoint["model_state_dict"]
        )
        print(
            json.dumps({
                "initialized_from": str(args.init_from.resolve()),
                "initial_epoch": int(initial_checkpoint["epoch"]),
            }),
            flush=True,
        )
    print(
        json.dumps({
            "policy_type": model.policy_type,
            "parameters_m": (
                sum(parameter.numel() for parameter in model.parameters())
                / 1e6
            ),
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "backbone": "DiT",
            "objective": "rectified_flow",
        }),
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    history = []
    best = float("inf")
    started = time.monotonic()
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
        )
        val_loss = validate_epoch(
            model,
            val_loader,
            device,
            epoch,
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "elapsed_s": time.monotonic() - started,
        })
        (args.output_dir / "training_history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )
        payload = {
            "schema_version": 1,
            "policy_type": model.policy_type,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": {
                **vars(args),
                "policy_type": model.policy_type,
                "use_oct": True,
                "include_velocity": True,
                "include_previous_action": True,
                **{
                    name: np.asarray(value, dtype=np.float32).tolist()
                    for name, value in stats.items()
                },
            },
        }
        _atomic_torch_save(
            payload,
            args.output_dir / "last_model.pth",
        )
        if val_loss < best:
            best = val_loss
            _atomic_torch_save(
                payload,
                args.output_dir / "best_model.pth",
            )
        print(
            f"Epoch {epoch}: train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}",
            flush=True,
        )
    print(f"Training complete; best val loss={best:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
