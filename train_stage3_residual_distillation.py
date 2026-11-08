"""Train a bounded teacher residual on top of the selected restricted student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, TensorDataset

from restricted_perception.stage3_policy import Stage3ResidualActionPolicy
from restricted_perception.stage34 import (
    load_stage34_protocol,
    sha256_file,
)


ROOT = Path(__file__).resolve().parent


def _load(data_dir: Path):
    metadata = json.loads(
        (data_dir / "metadata.json").read_text(encoding="utf-8")
    )
    features = []
    actions = []
    targets = []
    for run in metadata["runs"]:
        with np.load(run["run_npz"]) as payload:
            x = payload["features"].astype(np.float32)
            y = payload["teacher_actions"].astype(np.float32)
        features.append(x)
        actions.append(y)
        targets.append(np.full(len(x), run["target_id"], dtype=object))
    return (
        np.concatenate(features),
        np.concatenate(actions),
        np.concatenate(targets),
    )


def _isolation(target_manifest: Path) -> dict:
    targets = json.loads(target_manifest.read_text(encoding="utf-8"))
    split = json.loads(
        (ROOT / "configs/target_split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    train = np.asarray(
        [target["point_mm"] for target in targets["targets"]],
        dtype=float,
    )
    excluded = np.asarray(
        [
            item["point_mm"]
            for key in ("development_targets", "final_test_targets")
            for item in split[key]
        ],
        dtype=float,
    )
    distance, _ = cKDTree(excluded).query(train)
    return {
        "minimum_distance_to_development_or_final_mm": float(distance.min()),
        "pass": bool(distance.min() >= 1.0 - 1e-12),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/stage34_protocol.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data/stage3_distillation_v2_20261106",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=ROOT / "outputs/restricted_stage2_variants_v5_20261106/dagger_only/best_model.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/restricted_stage3_residual_distilled_v7_20261106",
    )
    parser.add_argument("--seed", type=int, default=20270301)
    args = parser.parse_args()
    protocol = load_stage34_protocol(args.protocol)
    settings = protocol["stage3"]["distillation"]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    isolation = _isolation(ROOT / settings["training_target_manifest"])
    if not isolation["pass"]:
        raise ValueError("target isolation failed")
    features, teacher_actions, target_ids = _load(args.data_dir)
    unique = np.asarray(sorted(set(target_ids.tolist())))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique)
    validation_targets = set(
        unique[: int(settings["validation_targets"])].tolist()
    )
    validation_mask = np.asarray(
        [target in validation_targets for target in target_ids]
    )
    train_mask = ~validation_mask
    base_checkpoint = torch.load(
        args.base_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_velocity = bool(base_checkpoint["config"].get("use_velocity", False))
    model = Stage3ResidualActionPolicy(
        vision_dim=512,
        state_dim=3,
        restricted_dim=18,
        use_velocity=use_velocity,
    ).to(device)
    model.base.load_state_dict(base_checkpoint["model_state_dict"])
    model.base.set_normalization(**base_checkpoint["normalization"])
    model.base.eval()
    for parameter in model.base.parameters():
        parameter.requires_grad_(False)
    base_features = torch.from_numpy(features).to(device)
    with torch.no_grad():
        base_actions = model.base(base_features)[0].float().cpu().numpy()
    residual_targets = (teacher_actions - base_actions).astype(np.float32)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(features[train_mask]),
            torch.from_numpy(residual_targets[train_mask]),
        ),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
    )
    val_x = torch.from_numpy(features[validation_mask]).to(device)
    val_y = torch.from_numpy(residual_targets[validation_mask]).to(device)
    optimizer = torch.optim.AdamW(
        model.residual.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=1e-4,
    )
    history = []
    best_loss = float("inf")
    best_residual_state = None
    for epoch in range(int(settings["epochs"])):
        model.residual.train()
        losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            full_action, _ = model(batch_x)
            base_action, _ = model.base(batch_x)
            correction = full_action - base_action
            loss = torch.nn.functional.smooth_l1_loss(
                correction,
                batch_y,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.residual.parameters(),
                1.0,
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.residual.eval()
        with torch.no_grad():
            full_action, _ = model(val_x)
            base_action, _ = model.base(val_x)
            correction = full_action - base_action
            val_loss = float(
                torch.nn.functional.smooth_l1_loss(
                    correction,
                    val_y,
                ).cpu()
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
            }
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_residual_state = {
                key: value.detach().cpu()
                for key, value in model.residual.state_dict().items()
            }
        print(
            f"epoch={epoch} train={np.mean(losses):.7f} "
            f"val={val_loss:.7f}",
            flush=True,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "policy_type": "restricted_stage3_residual_distilled",
        "variant": "bounded_residual_from_seven_sample_median",
        "base_model_state_dict": base_checkpoint["model_state_dict"],
        "normalization": base_checkpoint["normalization"],
        "residual_state_dict": best_residual_state,
        "config": {
            "vision_dim": 512,
            "state_dim": 3,
            "restricted_dim": 18,
            "use_velocity": use_velocity,
            "residual_distance_scale": None,
            "use_stop_head": False,
            "n_action_steps": 1,
            "residual_limit": [0.005, 0.003, 0.001],
        },
        "source": {
            "distillation_data": str(args.data_dir.resolve()),
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": sha256_file(args.protocol),
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
            "teacher_checkpoint": settings["teacher_checkpoint"],
            "training_target_manifest": settings["training_target_manifest"],
            "training_target_isolation": isolation,
            "teacher_samples": int(settings["teacher_samples"]),
            "teacher_aggregation": "median",
            "total_samples": int(len(features)),
            "train_samples": int(train_mask.sum()),
            "validation_samples": int(validation_mask.sum()),
        },
        "history": history,
        "best_val_loss": best_loss,
    }
    torch.save(checkpoint, args.output_dir / "best_model.pth")
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": "PASS",
                "best_val_loss": best_loss,
                "checkpoint": str(
                    (args.output_dir / "best_model.pth").resolve()
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
