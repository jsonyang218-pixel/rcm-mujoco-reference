"""CUDA smoke test for the v13 spatial FiLM U-Net diffusion policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from lerobot_training.train_diffusion_policy import (
    RCMDataset,
    image_transform_for_dataset,
)
from lerobot_training.v13_policy import (
    SpatialFiLMUNetDiffusionPolicy,
)


def _gradient_norm(parameters) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().pow(2).sum())
    return squared ** 0.5


def _select_sample(dataset: RCMDataset) -> int:
    distances = np.linalg.norm(dataset.oct_navigation, axis=1)
    for sample_index, data_index in enumerate(dataset.valid_indices):
        if (
            dataset.oct_navigation_valid[data_index]
            and 2.0 <= distances[data_index] <= 5.0
        ):
            return sample_index
    raise RuntimeError("no valid sample found with OCT distance in [2, 5] mm")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data/lerobot_v12_oct_dagger1_fixed_224",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-diffusion-iters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "RUNNING",
        "data_dir": str(Path(args.data_dir).resolve()),
        "requested_device": args.device,
        "seed": args.seed,
        "timing_s": {},
    }
    result_path = output_dir / "result.json"
    started = time.perf_counter()

    try:
        cuda_available = torch.cuda.is_available()
        if args.device == "cuda" and not cuda_available:
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device(
            "cuda"
            if args.device == "cuda"
            or (args.device == "auto" and cuda_available)
            else "cpu"
        )
        result["device"] = str(device)
        result["cuda_available"] = cuda_available
        if device.type == "cuda":
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_capability"] = list(
                torch.cuda.get_device_capability(0)
            )
        else:
            result["cpu_fallback_reason"] = (
                "torch.cuda.is_available() returned false; "
                "see host GPU access diagnostics"
            )

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

        dataset = RCMDataset(
            Path(args.data_dir),
            transform=image_transform_for_dataset(args.data_dir),
            n_obs_steps=4,
            n_action_steps=16,
            include_velocity=True,
            include_previous_action=True,
        )
        sample_index = _select_sample(dataset)
        sample = dataset[sample_index]
        batch = default_collate([sample])
        result["sample"] = {
            "dataset_index": int(dataset.valid_indices[sample_index]),
            "image_shape": list(batch["observation.image"].shape),
            "state_shape": list(batch["observation.state"].shape),
            "oct_shape": list(batch["observation.oct"].shape),
            "velocity_shape": list(
                batch["observation.velocity"].shape
            ),
            "previous_action_shape": list(
                batch["observation.previous_action"].shape
            ),
            "action_shape": list(batch["action"].shape),
            "oct_distance_mm": float(
                np.linalg.norm(
                    dataset.oct_navigation[
                        dataset.valid_indices[sample_index]
                    ]
                )
            ),
        }

        tensors = {
            name: batch[name].to(device, non_blocking=True)
            for name in (
                "observation.image",
                "observation.state",
                "observation.oct",
                "observation.velocity",
                "observation.previous_action",
                "action",
            )
        }
        model = SpatialFiLMUNetDiffusionPolicy(
            n_obs_steps=4,
            n_action_steps=16,
            num_diffusion_iters=args.num_diffusion_iters,
            diffusion_schedule="cosine",
            pred_x0_clip=4.0,
            oct_dim=4,
            state_mean=dataset.states.mean(axis=0),
            state_std=np.maximum(dataset.states.std(axis=0), 1e-6),
            oct_mean=np.concatenate([
                dataset.oct_navigation[
                    dataset.oct_navigation_valid
                ].mean(axis=0),
                [0.0],
            ]),
            oct_std=np.concatenate([
                np.maximum(
                    dataset.oct_navigation[
                        dataset.oct_navigation_valid
                    ].std(axis=0),
                    1e-6,
                ),
                [1.0],
            ]),
            action_mean=dataset.actions.mean(axis=0),
            action_std=np.maximum(
                dataset.actions.std(axis=0),
                1e-6,
            ),
            include_velocity=True,
            include_previous_action=True,
            keypoint_channels=32,
            token_dim=256,
            condition_dim=256,
            base_channels=64,
            middle_channels=128,
            top_channels=256,
            pretrained_vision=False,
        ).to(device)
        result["architecture"] = {
            "name": "spatial_film_unet_diffusion",
            "observation_steps": 4,
            "action_steps": 16,
            "keypoint_channels": 32,
            "keypoint_coordinates_dim": 64,
            "token_dim": 256,
            "condition_dim": 256,
            "unet_channels": [64, 128, 256],
            "num_diffusion_iters": args.num_diffusion_iters,
            "pretrained_vision": False,
            "parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
            ),
        }

        model.train()
        encoding_started = time.perf_counter()
        with torch.no_grad():
            condition, keypoints, attention = model.encode_observation(
                tensors["observation.image"],
                tensors["observation.state"],
                obs_oct=tensors["observation.oct"],
                velocity=tensors["observation.velocity"],
                previous_action=tensors[
                    "observation.previous_action"
                ],
            )
        result["forward_shapes"] = {
            "condition": list(condition.shape),
            "keypoints": list(keypoints.shape),
            "attention": list(attention.shape),
        }
        result["timing_s"]["encoding"] = time.perf_counter() - encoding_started

        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        train_started = time.perf_counter()
        loss = model(
            tensors["observation.image"],
            tensors["observation.state"],
            tensors["action"],
            obs_oct=tensors["observation.oct"],
            velocity=tensors["observation.velocity"],
            previous_action=tensors["observation.previous_action"],
            generator=generator,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: {loss.item()}")
        loss.backward()
        result["timing_s"]["train_forward_backward"] = (
            time.perf_counter() - train_started
        )
        result["loss"] = float(loss.detach().cpu())
        result["gradient_norms"] = {
            "vision": _gradient_norm(model.vision_encoder.parameters()),
            "cross_attention": _gradient_norm(
                model.cross_attention.parameters()
            ),
            "unet": _gradient_norm(model.denoiser.parameters()),
        }
        if any(
            not np.isfinite(value) or value <= 0
            for value in result["gradient_norms"].values()
        ):
            raise RuntimeError("one or more architecture branches lack gradients")

        model.eval()
        sample_started = time.perf_counter()
        with torch.no_grad():
            sampled = model(
                tensors["observation.image"],
                tensors["observation.state"],
                obs_oct=tensors["observation.oct"],
                velocity=tensors["observation.velocity"],
                previous_action=tensors[
                    "observation.previous_action"
                ],
                generator=torch.Generator(device=device).manual_seed(
                    args.seed
                ),
            )
        if sampled.shape != (1, 16, 3):
            raise RuntimeError(
                f"unexpected sample shape: {tuple(sampled.shape)}"
            )
        if not torch.isfinite(sampled).all():
            raise RuntimeError("sampled actions contain non-finite values")
        result["timing_s"]["ddim_sample"] = time.perf_counter() - sample_started
        result["sample_output"] = {
            "shape": list(sampled.shape),
            "minimum": float(sampled.min().cpu()),
            "maximum": float(sampled.max().cpu()),
            "mean": float(sampled.mean().cpu()),
            "std": float(sampled.std().cpu()),
        }
        result["status"] = "PASS"
    except Exception as error:
        result["status"] = "FAIL"
        result["error"] = f"{type(error).__name__}: {error}"
        result["elapsed_s"] = time.perf_counter() - started
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise

    result["elapsed_s"] = time.perf_counter() - started
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
