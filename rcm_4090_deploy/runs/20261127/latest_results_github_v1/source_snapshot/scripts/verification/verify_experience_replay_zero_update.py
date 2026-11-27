#!/usr/bin/env python3
"""Verify zero-update checkpoint identity and fixed-noise inference parity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "lerobot_training"))

import dit_state_attention as attention
from train_diffusion_policy import (
    RCMDataset,
    image_transform_for_dataset,
)
from train_dit_rectified_flow import _move_batch


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


def build_policy(checkpoint, device):
    config = checkpoint["config"]
    model = attention.StateConditionedDiTRectifiedFlowPolicy(
        **{name: config[name] for name in STAT_NAMES},
        **{name: int(config[name]) for name in INTEGER_POLICY_NAMES},
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def fixed_action(model, batch, seed):
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        return model.sample(
            batch["observation.image"],
            batch["observation.state"],
            batch["observation.oct"],
            batch["observation.velocity"],
            batch["observation.previous_action"],
            generator=generator,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--zero-update", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory already exists")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)

    initial = torch.load(
        args.initial, map_location="cpu", weights_only=False
    )
    zero_update = torch.load(
        args.zero_update, map_location="cpu", weights_only=False
    )
    initial_state = initial["model_state_dict"]
    zero_state = zero_update["model_state_dict"]
    if set(initial_state) != set(zero_state):
        raise ValueError("initial and zero-update state keys differ")
    mismatches = [
        name
        for name in initial_state
        if not torch.equal(initial_state[name], zero_state[name])
    ]
    if mismatches:
        raise ValueError(f"zero update changed parameters: {mismatches}")

    initial_model = build_policy(initial, device)
    zero_model = build_policy(zero_update, device)
    dataset = RCMDataset(
        args.data_dir,
        transform=image_transform_for_dataset(str(args.data_dir)),
        n_obs_steps=initial_model.n_obs_steps,
        n_action_steps=initial_model.n_action_steps,
        include_velocity=True,
        include_previous_action=True,
    )
    sample = _move_batch(default_collate([dataset[0]]), device)
    first = fixed_action(initial_model, sample, args.seed)
    second = fixed_action(zero_model, sample, args.seed)
    if not torch.equal(first, second):
        maximum = float((first - second).abs().max())
        raise ValueError(f"fixed-noise action mismatch: {maximum}")

    result = {
        "status": "PASS",
        "weights_exact": True,
        "tensor_count": len(initial_state),
        "fixed_noise_seed": args.seed,
        "action_exact": True,
        "action_sha256": hashlib.sha256(
            first.detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "initial_checkpoint": str(args.initial.resolve()),
        "zero_update_checkpoint": str(args.zero_update.resolve()),
    }
    (args.output_dir / "zero_update_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
