"""Offline counterfactual diagnostics for a trained restricted action policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from restricted_perception.policy import RestrictedActionMLP
from train_restricted_policy import load_split


def _distance_bins(distance_mm: np.ndarray, width_mm: float = 0.1) -> np.ndarray:
    return np.floor(np.maximum(distance_mm, 0.0) / width_mm).astype(np.int64)


def _mismatch_indices(
    episodes: np.ndarray,
    distance_mm: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    bins = _distance_bins(distance_mm)
    result = np.empty(len(episodes), dtype=np.int64)
    for index in range(len(episodes)):
        candidates = np.flatnonzero(
            (bins == bins[index]) & (episodes != episodes[index])
        )
        if not len(candidates):
            adjacent = np.argsort(np.abs(distance_mm - distance_mm[index]))
            candidates = np.asarray(
                [candidate for candidate in adjacent if episodes[candidate] != episodes[index]][:32],
                dtype=np.int64,
            )
        result[index] = int(rng.choice(candidates))
    return result


def _predict(model, features: np.ndarray, device, batch_size: int) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start + batch_size]).to(device)
            outputs.append(model(batch)[:, 0].cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _mae(prediction, target) -> dict:
    error = np.abs(prediction - target)
    return {
        "samples": int(len(prediction)),
        "mean_mae_deg_deg_mm": error.mean(axis=0).tolist(),
        "normalized_mae": float(np.mean(error / np.maximum(np.std(target, axis=0), 1e-6))),
        "mean_action_offset_norm": float(np.linalg.norm(prediction - target, axis=1).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261205)
    parser.add_argument("--vision-feature-file", type=Path, default=None)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if not config.get("use_image", False):
        raise ValueError("offline image causal gate requires an image policy")
    features, targets, metadata = load_split(
        args.data_dir,
        use_image=True,
        include_dagger=True,
        n_action_steps=int(config["n_action_steps"]),
        max_rows=args.max_rows,
        seed=args.seed,
        vision_feature_file=args.vision_feature_file,
    )
    valid = metadata["valid_indices"]
    with np.load(args.data_dir / "data" / "dataset.npz") as dataset:
        episodes = dataset["episode_index"][valid]
        fine = dataset["oct_navigation"][valid]
        fine_valid = dataset["oct_navigation_valid"][valid]
    with np.load(args.data_dir / "restricted_sidecar" / "restricted_sidecar.npz") as sidecar:
        fused = sidecar["fused_goal_vector_robot_mm"][valid]
        fused_valid = np.isfinite(fused).all(axis=1)
    distances_mm = np.where(
        fine_valid,
        np.linalg.norm(fine, axis=1),
        np.where(fused_valid, np.linalg.norm(np.nan_to_num(fused), axis=1), np.inf),
    )
    rng = np.random.default_rng(args.seed)
    mismatch = _mismatch_indices(episodes, distances_mm, rng)
    image_dim = int(config["image_dim"])
    vector_start = image_dim + int(config["state_dim"])
    vector_end = vector_start + 6
    zero_image = features.copy()
    zero_image[:, :image_dim] = 0.0
    mismatch_image = features.copy()
    mismatch_image[:, :image_dim] = features[mismatch, :image_dim]
    zero_vector = features.copy()
    zero_vector[:, vector_start:vector_end] = 0.0
    mismatch_vector = features.copy()
    mismatch_vector[:, vector_start:vector_end] = features[mismatch, vector_start:vector_end]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RestrictedActionMLP(
        vision_dim=image_dim,
        state_dim=int(config["state_dim"]),
        restricted_dim=int(config["restricted_dim"]),
        n_action_steps=int(config["n_action_steps"]),
        use_image=True,
    ).to(device)
    model.set_normalization(**checkpoint["normalization"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    target = targets[:, 0]
    variants = {
        "normal": _predict(model, features, device, args.batch_size),
        "zero_image": _predict(model, zero_image, device, args.batch_size),
        "mismatch_image": _predict(model, mismatch_image, device, args.batch_size),
        "zero_vector": _predict(model, zero_vector, device, args.batch_size),
        "mismatch_vector": _predict(model, mismatch_vector, device, args.batch_size),
    }
    near = distances_mm < 2.0
    report = {
        "result": "OFFLINE_ONLY",
        "scope": "No closed-loop success claim; inspect near-target action causality before rollout.",
        "samples": int(len(features)),
        "near_target_samples": int(near.sum()),
        "image_dim": image_dim,
        "vector_slice": [vector_start, vector_end],
        "normal_action": _mae(variants["normal"], target),
        "zero_image": _mae(variants["zero_image"], target),
        "mismatch_image": _mae(variants["mismatch_image"], target),
        "zero_vector": _mae(variants["zero_vector"], target),
        "mismatch_vector": _mae(variants["mismatch_vector"], target),
        "near_target": {
            name: _mae(prediction[near], target[near])
            for name, prediction in variants.items()
        },
        "normal_prediction_shift": {
            name: {
                "nearest_neighbor_offset_norm": float(
                    np.linalg.norm(prediction - variants["normal"], axis=1).mean()
                ),
                "p95_offset_norm": float(
                    np.percentile(
                        np.linalg.norm(prediction - variants["normal"], axis=1),
                        95.0,
                    )
                ),
            }
            for name, prediction in variants.items()
            if name != "normal"
        },
        "closed_loop_gate": "NOT_EVALUATED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": report["result"], "near_target": report["near_target"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
