"""Train the ordered Stage 2 DAgger single-factor ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from restricted_perception.config import load_config
from restricted_perception.stage2_policy import Stage2ActionPolicy
from restricted_perception.sidecar import restricted_feature_matrix


VARIANTS = {
    "dagger_only": {"use_velocity": False, "residual_distance_scale": None, "use_stop_head": False},
    "velocity": {"use_velocity": True, "residual_distance_scale": None, "use_stop_head": False},
    "velocity_residual_cap": {
        "use_velocity": True,
        "residual_distance_scale": 0.010,
        "use_stop_head": False,
    },
    "velocity_residual_cap_stop": {
        "use_velocity": True,
        "residual_distance_scale": 0.010,
        "use_stop_head": True,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_dataset(data_dir: Path, use_velocity: bool):
    with np.load(data_dir / "dataset.npz") as dataset:
        arrays = {key: dataset[key].copy() for key in dataset.files}
    vision = np.load(data_dir / "vision_features.npy", mmap_mode="r")
    if len(vision) != len(arrays["state"]):
        raise ValueError("vision feature count mismatch")
    parts = [np.asarray(vision, dtype=np.float32), arrays["state"], arrays["restricted"]]
    if use_velocity:
        parts.append(arrays["velocity"])
    features = np.concatenate(parts, axis=1).astype(np.float32)
    action = np.where(arrays["teacher_valid"][:, None], arrays["action_teacher"], 0.0)
    fine = arrays["restricted"][:, 3:6]
    fine_valid = arrays["restricted"][:, 7] > 0.5
    precision = arrays["restricted"][:, 17] > 0.5
    stop = (
        fine_valid
        & precision
        & (np.linalg.norm(fine, axis=1) <= 0.020)
    ).astype(np.float32)
    valid = np.ones(len(action), dtype=bool)
    return features, action.astype(np.float32), stop, valid, arrays


def _split_episodes(episode_indices: np.ndarray, seed: int, val_fraction=0.2):
    episodes = np.unique(episode_indices)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    val_count = max(1, int(round(len(episodes) * val_fraction)))
    validation = set(episodes[:val_count].tolist())
    train_mask = np.asarray([item not in validation for item in episode_indices])
    return train_mask, ~train_mask


def _load_base_dataset(
    data_dir: Path,
    vision_feature_file: Path,
    use_velocity: bool,
    sidecar_dir: Path | None = None,
):
    with np.load(data_dir / "data" / "dataset.npz") as dataset:
        episodes = dataset["episode_index"].copy()
        states = dataset["state"].astype(np.float32)
        actions = dataset["action"].astype(np.float32)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    allowed = np.ones(len(episodes), dtype=bool)
    for source in metadata.get("source_datasets", []):
        if "dagger" in Path(source["directory"]).name.lower():
            start = int(source["episode_offset"])
            end = start + int(source["episodes"])
            allowed &= ~((episodes >= start) & (episodes < end))
    sidecar_path = (
        data_dir / "restricted_sidecar" / "restricted_sidecar.npz"
        if sidecar_dir is None
        else sidecar_dir / "restricted_sidecar.npz"
    )
    with np.load(sidecar_path) as sidecar:
        restricted = restricted_feature_matrix(sidecar)
    vision = np.load(vision_feature_file, mmap_mode="r")
    velocity = np.zeros_like(states)
    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        if len(rows) > 1:
            velocity[rows[1:]] = np.diff(states[rows], axis=0) / 0.05
    vision_array = np.asarray(vision, dtype=np.float32)
    parts = [vision_array[allowed], states[allowed], restricted[allowed]]
    if use_velocity:
        parts.append(velocity[allowed])
    features = np.concatenate(parts, axis=1).astype(np.float32)
    fine = restricted[allowed, 3:6]
    fine_valid = restricted[allowed, 7] > 0.5
    precision = restricted[allowed, 17] > 0.5
    stop = (
        fine_valid
        & precision
        & (np.linalg.norm(fine, axis=1) <= 0.020)
    ).astype(np.float32)
    return features, actions[allowed], stop, episodes[allowed]


def _normalization(features, actions):
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    action_mean = actions.mean(axis=0)
    action_std = actions.std(axis=0)
    action_std[action_std < 1e-6] = 1.0
    return {
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "action_mean": action_mean.astype(np.float32),
        "action_std": action_std.astype(np.float32),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--base-data-dir", type=Path, required=True)
    parser.add_argument("--base-val-data-dir", type=Path, required=True)
    parser.add_argument("--base-vision-feature-file", type=Path, required=True)
    parser.add_argument("--val-vision-feature-file", type=Path, required=True)
    parser.add_argument("--base-sidecar-dir", type=Path, default=None)
    parser.add_argument("--val-sidecar-dir", type=Path, default=None)
    parser.add_argument("--dagger-replay-weight", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20270102)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=list(VARIANTS),
    )
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perception_config = load_config(
        Path(__file__).resolve().parent / "configs/restricted_perception.json"
    )
    stage1 = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    histories = {}
    for variant in args.variants:
        options = VARIANTS[variant]
        features, actions, stop, valid, arrays = _load_dataset(
            args.data_dir,
            options["use_velocity"],
        )
        features, actions, stop, valid = (
            features[valid],
            actions[valid],
            stop[valid],
            valid[valid],
        )
        dagger_count = len(features)
        base_features, base_actions, base_stop, base_episodes = _load_base_dataset(
            args.base_data_dir,
            args.base_vision_feature_file,
            options["use_velocity"],
            args.base_sidecar_dir,
        )
        train_features = np.concatenate([base_features, features], axis=0)
        train_actions = np.concatenate([base_actions, actions], axis=0)
        train_stop = np.concatenate([base_stop, stop], axis=0)
        sample_weights = np.concatenate(
            [
                np.ones(len(base_features), dtype=np.float64),
                np.full(dagger_count, args.dagger_replay_weight, dtype=np.float64),
            ]
        )
        val_features, val_actions, val_stop, _ = _load_base_dataset(
            args.base_val_data_dir,
            args.val_vision_feature_file,
            options["use_velocity"],
            args.val_sidecar_dir,
        )
        stats = _normalization(train_features, train_actions)
        model = Stage2ActionPolicy(
            vision_dim=512,
            state_dim=3,
            restricted_dim=18,
            use_velocity=options["use_velocity"],
            residual_distance_scale=None,
            use_stop_head=options["use_stop_head"],
            action_limit=[0.2898, 0.14434, 0.04271],
        ).to(device)
        model.set_normalization(**stats)
        base_state = stage1["model_state_dict"]
        if "network.0.weight" in base_state:
            with torch.no_grad():
                columns = min(
                    base_state["network.0.weight"].shape[1],
                    model.network[0].weight.shape[1],
                )
                model.network[0].weight[:, :columns] = base_state[
                    "network.0.weight"
                ][:, :columns].to(device)
                model.network[0].bias.copy_(base_state["network.0.bias"].to(device))
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(train_features),
                torch.from_numpy(train_actions),
                torch.from_numpy(train_stop),
            ),
            batch_size=args.batch_size,
            sampler=WeightedRandomSampler(
                torch.from_numpy(sample_weights),
                num_samples=len(sample_weights),
                replacement=True,
            ),
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(val_features),
                torch.from_numpy(val_actions),
                torch.from_numpy(val_stop),
            ),
            batch_size=args.batch_size,
            shuffle=False,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        history = []
        best_loss = float("inf")
        best_state = None
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for batch_features, batch_action, batch_stop in train_loader:
                batch_features = batch_features.to(device)
                batch_action = batch_action.to(device)
                batch_stop = batch_stop.to(device)
                predicted, stop_logit = model(batch_features)
                target = (batch_action - model.action_mean) / model.action_std
                prediction = (predicted - model.action_mean) / model.action_std
                loss = torch.nn.functional.smooth_l1_loss(prediction, target)
                if stop_logit is not None:
                    loss = loss + 0.2 * torch.nn.functional.binary_cross_entropy_with_logits(
                        stop_logit[:, 0],
                        batch_stop,
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            epoch_loss = float(np.mean(losses))
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_features, batch_action, batch_stop in val_loader:
                    batch_features = batch_features.to(device)
                    batch_action = batch_action.to(device)
                    batch_stop = batch_stop.to(device)
                    predicted, stop_logit = model(batch_features)
                    target = (batch_action - model.action_mean) / model.action_std
                    prediction = (predicted - model.action_mean) / model.action_std
                    loss = torch.nn.functional.smooth_l1_loss(prediction, target)
                    if stop_logit is not None:
                        loss = loss + 0.2 * torch.nn.functional.binary_cross_entropy_with_logits(
                            stop_logit[:, 0],
                            batch_stop,
                        )
                    val_losses.append(float(loss.cpu()))
            val_loss = float(np.mean(val_losses))
            history.append({"epoch": epoch, "loss": epoch_loss, "val_loss": val_loss})
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                }
            print(f"{variant} epoch={epoch} loss={epoch_loss:.6f} val={val_loss:.6f}")
        output = args.output_dir / variant
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "schema_version": 1,
            "policy_type": "restricted_stage2",
            "variant": variant,
            "model_state_dict": best_state,
            "normalization": {
                key: value.tolist() for key, value in stats.items()
            },
            "config": {
                "vision_dim": 512,
                "state_dim": 3,
                "restricted_dim": 18,
                **options,
            },
            "source": {
                "stage2_data": str(args.data_dir.resolve()),
                "stage2_data_sha256": _sha256(args.data_dir / "dataset.npz"),
                "stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
                "stage1_checkpoint_sha256": _sha256(args.stage1_checkpoint),
                "target_manifest": metadata["targets"],
                "episodes": metadata["num_episodes"],
                "frames": metadata["num_frames"],
            },
            "history": history,
            "best_loss": best_loss,
        }
        torch.save(checkpoint, output / "best_model.pth")
        (output / "training_history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        histories[variant] = best_loss
    print(json.dumps(histories, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
