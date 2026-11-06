"""Evaluate real, zero and shuffled OCT on a guided residual checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_diffusion_policy import RCMDataset
from train_oct_guided_regression import (
    GuidedOCTResidualPolicy,
    _build_arrays,
    _causal_evaluation,
)
from rcm_reference import Reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--per-bin", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20271105)
    parser.add_argument("--min-causal-gain", type=float, default=0.10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["config"]
    dataset = RCMDataset(
        args.data_dir,
        n_obs_steps=1,
        n_action_steps=1,
    )
    reference = Reference()
    _, camera_rotation = reference.camera_pose()
    arrays = _build_arrays(
        dataset,
        reference,
        camera_rotation,
        config["gain"],
        np.asarray(config["base_saturation"], dtype=np.float64),
        config["relative_damping"],
        np.asarray(config["far_residual_limit"], dtype=np.float64),
        config["full_effect_distance_mm"],
        config["near_fraction"],
    )
    model = GuidedOCTResidualPolicy(
        state_mean=config["state_mean"],
        state_std=config["state_std"],
        log_distance_mean=config["log_distance_mean"],
        log_distance_std=config["log_distance_std"],
        base_mean=config["base_mean"],
        base_std=config["base_std"],
        far_residual_limit=config["far_residual_limit"],
        full_effect_distance_mm=config["full_effect_distance_mm"],
        near_fraction=config["near_fraction"],
        hidden_dim=config["hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    causal = _causal_evaluation(
        model,
        arrays,
        args.batch_size,
        device,
        args.seed,
        args.per_bin,
    )
    near = causal["near_target_C"]
    passed = bool(
        near["gain_vs_zero"] >= args.min_causal_gain
        and near["gain_vs_shuffled"] >= args.min_causal_gain
    )
    report = {
        "result": "PASS" if passed else "FAIL",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_val_mae_deg_deg_mm": checkpoint.get(
            "val_mae_deg_deg_mm"
        ),
        "causal_evaluation": causal,
        "acceptance": {
            "min_causal_gain": args.min_causal_gain,
            "near_target_C_real_better_than_zero": (
                near["gain_vs_zero"] >= args.min_causal_gain
            ),
            "near_target_C_real_better_than_shuffled": (
                near["gain_vs_shuffled"] >= args.min_causal_gain
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": report["result"],
                "checkpoint_epoch": report["checkpoint_epoch"],
                "checkpoint_val_mae_deg_deg_mm": (
                    report["checkpoint_val_mae_deg_deg_mm"]
                ),
                "near_target_C": near,
                "overall_mae": {
                    name: row["overall"]["mae_deg_deg_mm"]
                    for name, row in causal["conditions"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
