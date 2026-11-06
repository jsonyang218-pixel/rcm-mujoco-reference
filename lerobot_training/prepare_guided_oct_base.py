"""Precompute direct-OCT guided base actions for LeRobot datasets."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rcm_reference import Reference
from train_oct_guided_regression import (
    distance_aware_oct_features,
    proportional_guided_action,
    residual_limit_for_distance,
)


def prepare_dataset(
    data_dir,
    reference,
    camera_rotation,
    saturation,
    far_residual_limit,
    gain,
    relative_damping,
    full_effect_distance_mm,
    near_fraction,
):
    with np.load(data_dir / "data" / "dataset.npz") as data:
        states = data["state"].astype(np.float64)
        oct_navigation = np.concatenate(
            [
                data["oct_navigation"].astype(np.float64),
                data["oct_navigation_valid"].astype(np.float64)[:, None],
            ],
            axis=1,
        )
    guided_base = np.zeros((len(states), 3), dtype=np.float32)
    for row, (state, oct_row) in enumerate(
        tqdm(
            zip(states, oct_navigation),
            total=len(states),
            desc=f"Guided base {data_dir.name}",
        )
    ):
        guided_base[row] = proportional_guided_action(
            reference,
            state,
            oct_row[:3],
            camera_rotation,
            valid=bool(oct_row[3] > 0.5),
            gain=gain,
            saturation=saturation,
            relative_damping=relative_damping,
        )
    _, log_distance, distance = distance_aware_oct_features(
        oct_navigation
    )
    valid = oct_navigation[:, 3] > 0.5
    limit = residual_limit_for_distance(
        distance,
        valid,
        far_limit=far_residual_limit,
        full_effect_distance_mm=full_effect_distance_mm,
        near_fraction=near_fraction,
    ).astype(np.float32)
    data_output = data_dir / "data"
    base_path = data_output / "guided_base_action.npy"
    limit_path = data_output / "guided_residual_limit.npy"
    for path, array in (
        (base_path, guided_base),
        (limit_path, limit),
    ):
        temporary = path.with_name(path.name + ".tmp.npy")
        np.save(temporary, array, allow_pickle=False)
        temporary.replace(path)
    metadata = {
        "definition": (
            "gain * sat(damped_pseudoinverse(J_world) @ "
            "camera_to_world(OCT target-minus-tip))"
        ),
        "num_frames": int(len(states)),
        "gain": float(gain),
        "base_saturation_deg_deg_mm": saturation.tolist(),
        "far_residual_limit_deg_deg_mm": far_residual_limit.tolist(),
        "full_effect_distance_mm": float(full_effect_distance_mm),
        "near_fraction": float(near_fraction),
        "relative_damping": float(relative_damping),
        "camera_sign": [1.0, -1.0, -1.0],
        "valid_frames": int(valid.sum()),
        "log_distance_mean": float(log_distance[valid].mean()),
        "log_distance_std": float(max(log_distance[valid].std(), 1e-6)),
        "base_mean_deg_deg_mm": guided_base.mean(axis=0).tolist(),
        "base_std_deg_deg_mm": np.maximum(
            guided_base.std(axis=0),
            1e-6,
        ).tolist(),
    }
    metadata_path = data_output / "guided_oct_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "data_dir": str(data_dir),
        "base_path": str(base_path),
        "limit_path": str(limit_path),
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        required=True,
        help="Repeat for train and validation datasets.",
    )
    parser.add_argument("--gain", type=float, default=0.15)
    parser.add_argument("--relative-damping", type=float, default=1e-3)
    parser.add_argument("--full-effect-distance-mm", type=float, default=1.0)
    parser.add_argument("--near-residual-fraction", type=float, default=0.1)
    args = parser.parse_args()
    if args.gain < 0 or args.relative_damping < 0:
        parser.error("gain and damping must be nonnegative")
    if args.full_effect_distance_mm <= 0:
        parser.error("full effect distance must be positive")
    if not 0 <= args.near_residual_fraction <= 1:
        parser.error("near residual fraction must be in [0, 1]")

    reference = Reference()
    _, camera_rotation = reference.camera_pose()
    with np.load(args.data_dir[0] / "data" / "dataset.npz") as data:
        actions = data["action"].astype(np.float64)
    saturation = np.max(np.abs(actions), axis=0)
    far_residual_limit = saturation.copy()
    print(
        "guided OCT limits: "
        f"saturation={saturation.tolist()}, "
        f"far_residual={far_residual_limit.tolist()}",
        flush=True,
    )
    summaries = []
    for data_dir in args.data_dir:
        if not (data_dir / "data" / "dataset.npz").is_file():
            raise FileNotFoundError(data_dir / "data" / "dataset.npz")
        summaries.append(
            prepare_dataset(
                data_dir.resolve(),
                reference,
                camera_rotation,
                saturation,
                far_residual_limit,
                args.gain,
                args.relative_damping,
                args.full_effect_distance_mm,
                args.near_residual_fraction,
            )
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
