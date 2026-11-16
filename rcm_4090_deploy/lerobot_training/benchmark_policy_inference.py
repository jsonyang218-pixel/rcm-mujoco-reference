"""Benchmark action-chunk inference for supported policy checkpoints."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from dit_rectified_flow import DiTRectifiedFlowPolicy
from flow_temporal_unet import FlowTemporalUNetPolicy
from train_diffusion_policy import SimpleDiffusionPolicy


def _statistics(config, prefix):
    names = (
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
    return {name: config[f"{prefix}{name}"] for name in names}


def _load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    policy_type = checkpoint.get("policy_type")
    common = {
        "n_obs_steps": int(config.get("n_obs_steps", 2)),
        "n_action_steps": int(config.get("n_action_steps", 16)),
    }
    if policy_type == DiTRectifiedFlowPolicy.policy_type:
        model = DiTRectifiedFlowPolicy(
            **common,
            dit_dim=int(config.get("dit_dim", 384)),
            dit_depth=int(config.get("dit_depth", 6)),
            dit_heads=int(config.get("dit_heads", 6)),
            flow_integration_steps=int(
                config.get("flow_integration_steps", 4)
            ),
            **_statistics(config, ""),
        )
        steps = int(config.get("flow_integration_steps", 4))
        family = "DiT + Rectified Flow"
    elif policy_type == FlowTemporalUNetPolicy.policy_type:
        model = FlowTemporalUNetPolicy(
            **common,
            base_channels=int(config.get("base_channels", 256)),
            flow_integration_steps=int(
                config.get("flow_integration_steps", 4)
            ),
            **_statistics(config, ""),
        )
        steps = int(config.get("flow_integration_steps", 4))
        family = "1D U-Net + Flow Matching"
    elif policy_type in (None, "oct_diffusion_policy"):
        model = SimpleDiffusionPolicy(
            **common,
            num_diffusion_iters=int(
                config.get("num_diffusion_iters", 10)
            ),
            diffusion_schedule=config.get(
                "diffusion_schedule",
                "linear",
            ),
            pred_x0_clip=config.get("pred_x0_clip"),
            oct_dim=4,
            oct_scale=config.get("oct_scale", [1, 1, 1, 1]),
            state_mean=config.get("state_mean", [0, 0, 0]),
            state_std=config.get("state_std", [1, 1, 1]),
            oct_mean=config.get("oct_mean", [0, 0, 0, 0]),
            oct_std=config.get("oct_std", [1, 1, 1, 1]),
            action_mean=config.get("action_mean", [0, 0, 0]),
            action_std=config.get("action_std", [1, 1, 1]),
            vision_mode=config.get("vision_mode", "global"),
            guided_oct_mode=False,
            include_velocity=bool(
                config.get("include_velocity", False)
            ),
            include_previous_action=bool(
                config.get("include_previous_action", False)
            ),
            velocity_mean=config.get("velocity_mean"),
            velocity_std=config.get("velocity_std"),
            previous_action_mean=config.get("previous_action_mean"),
            previous_action_std=config.get("previous_action_std"),
        )
        steps = int(config.get("num_diffusion_iters", 10))
        family = "U-Net + DDPM"
    else:
        raise ValueError(f"unsupported policy type: {policy_type}")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint, family, steps


def _inputs(policy_type, config, batch, device):
    n_obs = int(config.get("n_obs_steps", 2))
    image = torch.rand(
        batch,
        n_obs,
        3,
        224,
        224,
        device=device,
    )
    state = torch.randn(batch, n_obs, 3, device=device)
    oct_feature = torch.randn(batch, n_obs, 4, device=device)
    velocity = torch.randn(batch, 3, device=device)
    previous_action = torch.randn(batch, 3, device=device)
    return image, state, oct_feature, velocity, previous_action


def _infer(model, policy_type, inputs, generator):
    image, state, oct_feature, velocity, previous_action = inputs
    if policy_type in (
        DiTRectifiedFlowPolicy.policy_type,
        FlowTemporalUNetPolicy.policy_type,
    ):
        return model.sample(
            image,
            state,
            oct_feature,
            velocity,
            previous_action,
            generator=generator,
        )
    return model(
        image,
        state,
        obs_oct=oct_feature,
        generator=generator,
        guided_base=None,
        velocity=velocity,
        previous_action=previous_action,
    )


def _benchmark(model, policy_type, config, device, batch, repeats, warmup):
    inputs = _inputs(policy_type, config, batch, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(20261116)
    torch.cuda.synchronize(device)
    for _ in range(warmup):
        _infer(model, policy_type, inputs, generator)
    torch.cuda.synchronize(device)
    with FlopCounterMode(display=False) as flop_counter:
        _infer(model, policy_type, inputs, generator)
    torch.cuda.synchronize(device)
    total_flops = int(flop_counter.get_total_flops())
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    latencies = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _infer(model, policy_type, inputs, generator)
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end))
    latencies.sort()
    return {
        "batch_size": int(batch),
        "mean_ms": statistics.fmean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": latencies[
            min(len(latencies) - 1, int(0.95 * len(latencies)))
        ],
        "samples_per_s": batch * 1000.0 / statistics.fmean(latencies),
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        ),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / (1024 * 1024)
        ),
        "total_flops": total_flops,
        "gflops_per_sample": total_flops / batch / 1e9,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 32],
    )
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sampling-steps", type=int, default=None)
    args = parser.parse_args()
    device = torch.device("cuda")
    model, checkpoint, family, steps = _load_model(
        args.checkpoint,
        device,
    )
    if args.sampling_steps is not None:
        if not hasattr(model, "flow_integration_steps"):
            raise ValueError(
                "sampling-step override is only supported for flow policies"
            )
        if args.sampling_steps < 1:
            raise ValueError("sampling steps must be positive")
        model.flow_integration_steps = int(args.sampling_steps)
        steps = int(args.sampling_steps)
    policy_type = checkpoint.get(
        "policy_type",
        "oct_diffusion_policy",
    )
    results = {
        "checkpoint": str(args.checkpoint.resolve()),
        "family": family,
        "policy_type": policy_type,
        "parameters_m": (
            sum(parameter.numel() for parameter in model.parameters())
            / 1e6
        ),
        "model_size_mib": (
            args.checkpoint.stat().st_size / (1024 * 1024)
        ),
        "sampling_steps": steps,
        "batch_results": [
            _benchmark(
                model,
                policy_type,
                checkpoint["config"],
                device,
                int(batch),
                args.repeats,
                args.warmup,
            )
            for batch in args.batch_sizes
        ],
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
