"""Train the deterministic single-step head from seven-sample teacher medians."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, TensorDataset

from restricted_perception.stage2_policy import Stage2ActionPolicy
from restricted_perception.stage34 import (
    load_stage34_protocol,
    sha256_file,
)


ROOT = Path(__file__).resolve().parent


def _load_distillation(
    data_dir: Path,
    *,
    append_velocity: bool,
    sample_every_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metadata = json.loads(
        (data_dir / "metadata.json").read_text(encoding="utf-8")
    )
    feature_rows = []
    action_rows = []
    target_rows = []
    for run in metadata["runs"]:
        with np.load(run["run_npz"]) as payload:
            features = payload["features"].astype(np.float32)
            actions = payload["teacher_actions"].astype(np.float32)
            if append_velocity:
                states = payload["states"].astype(np.float32)
                velocity = np.zeros_like(states)
                if len(states) > 1:
                    velocity[1:] = np.diff(states, axis=0) / (
                        0.05 * float(sample_every_frames)
                    )
                features = np.concatenate([features, velocity], axis=1)
        if len(features) != len(actions) or len(features) == 0:
            raise ValueError(f"invalid distillation run: {run['run_npz']}")
        feature_rows.append(features)
        action_rows.append(actions)
        target_rows.append(
            np.full(len(features), run["target_id"], dtype=object)
        )
    return (
        np.concatenate(feature_rows, axis=0),
        np.concatenate(action_rows, axis=0),
        np.concatenate(target_rows, axis=0),
    )


def _normalization(features: np.ndarray, actions: np.ndarray) -> dict:
    feature_mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    action_mean = actions.mean(axis=0, dtype=np.float64).astype(np.float32)
    action_std = actions.std(axis=0, dtype=np.float64).astype(np.float32)
    action_std[action_std < 1e-6] = 1.0
    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }


def _verify_training_target_isolation(
    root: Path,
    target_manifest_path: Path,
) -> dict:
    targets = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    split = json.loads(
        (root / "configs/target_split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    training = np.asarray(
        [target["point_mm"] for target in targets["targets"]],
        dtype=float,
    )
    excluded = np.asarray(
        [
            entry["point_mm"]
            for key in ("development_targets", "final_test_targets")
            for entry in split[key]
        ],
        dtype=float,
    )
    distances, _ = cKDTree(excluded).query(training)
    return {
        "minimum_distance_to_development_or_final_mm": float(distances.min()),
        "pass": bool(distances.min() >= 1.0 - 1e-12),
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
        "--init-checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/restricted_stage2_variants_v5_20261106/dagger_only/best_model.pth"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/restricted_stage3_distilled_v2_20261106",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--append-velocity", action="store_true")
    parser.add_argument("--student-anchor-weight", type=float, default=0.0)
    args = parser.parse_args()
    protocol = load_stage34_protocol(args.protocol)
    settings = protocol["stage3"]["distillation"]
    seed = int(protocol["seed"] if args.seed is None else args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    isolation = _verify_training_target_isolation(
        ROOT,
        ROOT / settings["training_target_manifest"],
    )
    if not isolation["pass"]:
        raise ValueError("distillation training target isolation failed")
    if args.student_anchor_weight < 0:
        raise ValueError("student anchor weight must be nonnegative")
    features, teacher_actions, target_ids = _load_distillation(
        args.data_dir,
        append_velocity=args.append_velocity,
        sample_every_frames=int(settings["sample_every_frames"]),
    )
    if features.ndim != 2 or features.shape[1] not in (533, 536):
        raise ValueError(
            "distillation feature dimension must be 533 or 536"
        )
    use_velocity = features.shape[1] == 536
    unique_targets = np.asarray(sorted(set(target_ids.tolist())))
    rng = np.random.default_rng(seed)
    shuffled = unique_targets.copy()
    rng.shuffle(shuffled)
    validation_targets = set(
        shuffled[: int(settings["validation_targets"])].tolist()
    )
    validation_mask = np.asarray(
        [target in validation_targets for target in target_ids]
    )
    train_mask = ~validation_mask
    stats = _normalization(features[train_mask], teacher_actions[train_mask])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_checkpoint = torch.load(
        args.init_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    student_model = Stage2ActionPolicy(
        vision_dim=512,
        state_dim=3,
        restricted_dim=18,
        use_velocity=bool(
            student_checkpoint["config"].get("use_velocity", False)
        ),
        residual_distance_scale=student_checkpoint["config"].get(
            "residual_distance_scale"
        ),
        use_stop_head=False,
        action_limit=[0.2898, 0.14434, 0.04271],
    ).to(device)
    student_model.load_state_dict(student_checkpoint["model_state_dict"])
    student_model.set_normalization(**student_checkpoint["normalization"])
    student_model.eval()
    with torch.no_grad():
        student_actions = student_model(
            torch.from_numpy(features[:, :533]).to(device)
        )[0].cpu().numpy()
    model = Stage2ActionPolicy(
        vision_dim=512,
        state_dim=3,
        restricted_dim=18,
        use_velocity=use_velocity,
        residual_distance_scale=None,
        use_stop_head=False,
        action_limit=[0.2898, 0.14434, 0.04271],
    ).to(device)
    compatible = {
        key: value
        for key, value in student_checkpoint["model_state_dict"].items()
        if key in model.state_dict()
        and model.state_dict()[key].shape == value.shape
        and not key.startswith("feature_")
        and not key.startswith("action_")
    }
    model.load_state_dict(compatible, strict=False)
    model.set_normalization(**stats)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(features[train_mask]),
            torch.from_numpy(teacher_actions[train_mask]),
            torch.from_numpy(student_actions[train_mask]),
        ),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
    )
    val_x = torch.from_numpy(features[validation_mask]).to(device)
    val_y = torch.from_numpy(teacher_actions[validation_mask]).to(device)
    val_student = torch.from_numpy(
        student_actions[validation_mask]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=1e-4,
    )
    history = []
    best_loss = float("inf")
    best_state = None
    for epoch in range(int(settings["epochs"])):
        model.train()
        train_losses = []
        for batch_features, batch_actions, batch_student in train_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_actions = batch_actions.to(device, non_blocking=True)
            batch_student = batch_student.to(device, non_blocking=True)
            predicted, _ = model(batch_features)
            target = (batch_actions - model.action_mean) / model.action_std
            prediction = (
                predicted - model.action_mean
            ) / model.action_std
            loss = torch.nn.functional.smooth_l1_loss(prediction, target)
            if args.student_anchor_weight:
                student_target = (
                    batch_student - model.action_mean
                ) / model.action_std
                loss = loss + float(args.student_anchor_weight) * (
                    torch.nn.functional.smooth_l1_loss(
                        prediction,
                        student_target,
                    )
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            predicted, _ = model(val_x)
            target = (val_y - model.action_mean) / model.action_std
            prediction = (
                predicted - model.action_mean
            ) / model.action_std
            val_loss = float(
                torch.nn.functional.smooth_l1_loss(
                    prediction,
                    target,
                ).cpu()
            )
            if args.student_anchor_weight:
                student_target = (
                    val_student - model.action_mean
                ) / model.action_std
                val_loss += float(args.student_anchor_weight) * float(
                    torch.nn.functional.smooth_l1_loss(
                        prediction,
                        student_target,
                    ).cpu()
                )
            raw_mae = torch.mean(
                torch.abs(predicted - val_y),
                dim=0,
            ).cpu().numpy()
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
                "val_mae_deg_deg_mm": raw_mae.tolist(),
            }
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }
        print(
            f"epoch={epoch} train={np.mean(train_losses):.6f} "
            f"val={val_loss:.6f}",
            flush=True,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "policy_type": "restricted_stage3_distilled",
        "variant": "distilled_single_step_seven_median",
        "model_state_dict": best_state,
        "normalization": {
            key: value.tolist() for key, value in stats.items()
        },
        "config": {
            "vision_dim": 512,
            "state_dim": 3,
            "restricted_dim": 18,
            "use_velocity": use_velocity,
            "residual_distance_scale": None,
            "use_stop_head": False,
            "n_action_steps": 1,
        },
        "source": {
            "distillation_data": str(args.data_dir.resolve()),
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": sha256_file(args.protocol),
            "init_checkpoint": str(args.init_checkpoint.resolve()),
            "init_checkpoint_sha256": sha256_file(args.init_checkpoint),
            "training_target_manifest": settings["training_target_manifest"],
            "training_target_isolation": isolation,
            "teacher_samples": int(settings["teacher_samples"]),
            "teacher_aggregation": settings["teacher_aggregation"],
            "total_samples": int(len(features)),
            "feature_dim": int(features.shape[1]),
            "student_anchor_weight": float(args.student_anchor_weight),
            "append_velocity": bool(args.append_velocity),
            "train_samples": int(train_mask.sum()),
            "validation_samples": int(validation_mask.sum()),
            "training_targets": int(
                len(set(target_ids[train_mask].tolist()))
            ),
            "validation_targets": int(
                len(set(target_ids[validation_mask].tolist()))
            ),
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
