"""Benchmark distilled single-step and diffusion inference on CPU and GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from restricted_perception.stage2_policy import Stage2ActionPolicy
from restricted_perception.stage3_policy import Stage3ResidualActionPolicy


ROOT = Path(__file__).resolve().parent
if str(ROOT / "lerobot_training") not in sys.path:
    sys.path.insert(0, str(ROOT / "lerobot_training"))

from eval_policy import PolicyEvaluator  # noqa: E402


def _percentiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float) * 1000.0
    return {
        "p50_ms": float(np.percentile(array, 50.0)),
        "p95_ms": float(np.percentile(array, 95.0)),
    }


def _benchmark_student(
    checkpoint: Path,
    device: torch.device,
    *,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("policy_type") == "restricted_stage3_residual_distilled":
        model = Stage3ResidualActionPolicy(
            vision_dim=512,
            state_dim=3,
            restricted_dim=18,
            use_velocity=bool(payload["config"].get("use_velocity", False)),
            residual_limit=payload["config"].get(
                "residual_limit",
                [0.005, 0.003, 0.001],
            ),
        ).to(device)
        model.base.load_state_dict(payload["base_model_state_dict"])
        model.base.set_normalization(**payload["normalization"])
        model.residual.load_state_dict(payload["residual_state_dict"])
    else:
        model = Stage2ActionPolicy(
            vision_dim=512,
            state_dim=3,
            restricted_dim=18,
            use_velocity=bool(payload["config"].get("use_velocity", False)),
        ).to(device)
        model.set_normalization(**payload["normalization"])
        model.load_state_dict(payload["model_state_dict"])
    feature_dim = int(payload["normalization"]["feature_mean"].__len__())
    model.eval()
    features = torch.randn(batch_size, feature_dim, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(features)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations = []
        for _ in range(repeats):
            started = time.perf_counter()
            model(features)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - started)
    return _percentiles(durations)


def _benchmark_diffusion(
    checkpoint: Path,
    device: torch.device,
    *,
    batch_size: int,
    samples: int,
    warmup: int,
    repeats: int,
) -> dict:
    evaluator = PolicyEvaluator(
        str(checkpoint),
        device=str(device),
        num_action_samples=samples,
    )
    model = evaluator.model
    images = torch.randn(batch_size, 2, 3, 224, 224, device=device)
    states = torch.randn(batch_size, 2, 3, device=device)
    oct_features = torch.randn(batch_size, 2, 4, device=device)
    oct_features[:, :, 3] = 1.0
    guided_base = torch.zeros(batch_size, 3, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(20270301)
    with torch.inference_mode():
        for _ in range(warmup):
            for _ in range(samples):
                model(
                    images,
                    states,
                    obs_oct=oct_features,
                    generator=generator,
                    guided_base=guided_base,
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations = []
        for _ in range(repeats):
            started = time.perf_counter()
            first_actions = []
            for _ in range(samples):
                output = model(
                    images,
                    states,
                    obs_oct=oct_features,
                    generator=generator,
                    guided_base=guided_base,
                )
                first_actions.append(output[:, 0])
            if samples > 1:
                torch.stack(first_actions).median(dim=0).values
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - started)
    return _percentiles(durations)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=ROOT / "outputs/restricted_stage3_distilled_v2_20261106/best_model.pth",
    )
    parser.add_argument(
        "--diffusion-checkpoint",
        type=Path,
        default=ROOT / "outputs/diffusion_policy_v7_guided_dagger1_cached/best_model.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/restricted_stage3_latency_20261106.json",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    report = {
        "schema_version": 1,
        "definition": (
            "wall-clock model inference with one control observation; "
            "CUDA timings synchronize before and after"
        ),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "devices": {},
    }
    for device in devices:
        report["devices"][device.type] = {
            "student_single_step_batch_1": _benchmark_student(
                args.student_checkpoint,
                device,
                batch_size=1,
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "student_single_step_batch_32": _benchmark_student(
                args.student_checkpoint,
                device,
                batch_size=32,
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "diffusion_single_sample_batch_1": _benchmark_diffusion(
                args.diffusion_checkpoint,
                device,
                batch_size=1,
                samples=1,
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "diffusion_single_sample_batch_8": _benchmark_diffusion(
                args.diffusion_checkpoint,
                device,
                batch_size=8,
                samples=1,
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "diffusion_seven_sample_median_batch_1": _benchmark_diffusion(
                args.diffusion_checkpoint,
                device,
                batch_size=1,
                samples=7,
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "diffusion_seven_sample_median_batch_8": _benchmark_diffusion(
                args.diffusion_checkpoint,
                device,
                batch_size=8,
                samples=7,
                warmup=args.warmup,
                repeats=args.repeats,
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
