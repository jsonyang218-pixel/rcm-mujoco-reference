"""Train a single-step action policy on restricted coarse/fine observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from restricted_perception.config import load_config
from restricted_perception.policy import RestrictedActionMLP
from restricted_perception.sidecar import (
    RESTRICTED_FEATURE_NAMES,
    restricted_feature_matrix,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _valid_indices(episodes: np.ndarray, frames: np.ndarray, n_action_steps: int) -> np.ndarray:
    result = []
    for index in range(len(frames) - n_action_steps + 1):
        if frames[index] < 1:
            continue
        if index == 0 or episodes[index - 1] != episodes[index]:
            continue
        future = np.arange(index, index + n_action_steps)
        if np.all(episodes[future] == episodes[index]):
            result.append(index)
    return np.asarray(result, dtype=np.int64)


def _source_allowed(mask: np.ndarray, metadata: dict, include_dagger: bool) -> np.ndarray:
    if include_dagger:
        return np.ones(len(mask), dtype=bool)
    allowed = np.ones(len(mask), dtype=bool)
    for source in metadata.get("source_datasets", []):
        if "dagger" in Path(source["directory"]).name.lower():
            start = int(source["episode_offset"])
            end = start + int(source["episodes"])
            allowed &= ~((mask >= start) & (mask < end))
    return allowed


def load_split(
    data_dir: Path,
    *,
    use_image: bool,
    include_dagger: bool,
    n_action_steps: int,
    max_rows: int | None,
    seed: int,
    vision_feature_file: Path | None = None,
    sidecar_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    data_dir = data_dir.resolve()
    with np.load(data_dir / "data" / "dataset.npz") as dataset:
        episodes = dataset["episode_index"].copy()
        frames = dataset["frame_index"].copy()
        states = dataset["state"].astype(np.float32)
        actions = dataset["action"].astype(np.float32)
    sidecar_path = (
        data_dir / "restricted_sidecar" / "restricted_sidecar.npz"
        if sidecar_dir is None
        else Path(sidecar_dir) / "restricted_sidecar.npz"
    )
    with np.load(sidecar_path) as sidecar:
        if (
            not np.array_equal(episodes, sidecar["episode_index"])
            or not np.array_equal(frames, sidecar["frame_index"])
        ):
            raise ValueError("restricted sidecar is not row-aligned")
        restricted = restricted_feature_matrix(sidecar)
    valid = _valid_indices(episodes, frames, n_action_steps)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    valid = valid[_source_allowed(episodes[valid], metadata, include_dagger)]
    if len(valid) == 0:
        raise RuntimeError("no valid training rows")
    rng = np.random.default_rng(seed)
    if max_rows is not None and len(valid) > max_rows:
        valid = np.sort(rng.choice(valid, max_rows, replace=False))
    parts = [states[valid], restricted[valid]]
    image_dim = 0
    if use_image:
        image_path = (
            vision_feature_file
            if vision_feature_file is not None
            else data_dir / "data" / "vision_features.npy"
        )
        if not image_path.is_absolute():
            cwd_path = Path.cwd() / image_path
            image_path = cwd_path if cwd_path.is_file() else data_dir / image_path
        image = np.load(image_path, mmap_mode="r")
        image_dim = int(image.shape[1])
        parts.insert(0, np.asarray(image[valid], dtype=np.float32))
    features = np.concatenate(parts, axis=1).astype(np.float32)
    targets = np.stack(
        [
            actions[valid + offset]
            for offset in range(n_action_steps)
        ],
        axis=1,
    ).astype(np.float32)
    return features, targets, {
        "rows": len(valid),
        "image_dim": image_dim,
        "episodes": int(len(np.unique(episodes[valid]))),
        "valid_indices": valid,
    }


def standardization(features: np.ndarray, targets: np.ndarray) -> dict:
    feature_mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    action_mean = targets.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    action_std = targets.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    action_std[action_std < 1e-6] = 1.0
    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }


def normalized_mse(outputs: torch.Tensor, targets: torch.Tensor, stats: dict, device) -> torch.Tensor:
    mean = torch.as_tensor(stats["action_mean"], device=device)
    std = torch.as_tensor(stats["action_std"], device=device)
    return torch.nn.functional.smooth_l1_loss(
        (outputs - mean) / std,
        (targets - mean) / std,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--val-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20261202)
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--exclude-dagger", action="store_true")
    parser.add_argument("--train-vision-feature-file", type=Path, default=None)
    parser.add_argument("--val-vision-feature-file", type=Path, default=None)
    parser.add_argument("--train-sidecar-dir", type=Path, default=None)
    parser.add_argument("--val-sidecar-dir", type=Path, default=None)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_features, train_actions, train_meta = load_split(
        args.data_dir,
        use_image=not args.no_image,
        include_dagger=not args.exclude_dagger,
        n_action_steps=args.n_action_steps,
        max_rows=args.max_train_rows,
        seed=args.seed,
        vision_feature_file=args.train_vision_feature_file,
        sidecar_dir=args.train_sidecar_dir,
    )
    val_features, val_actions, val_meta = load_split(
        args.val_data_dir,
        use_image=not args.no_image,
        include_dagger=True,
        n_action_steps=args.n_action_steps,
        max_rows=args.max_val_rows,
        seed=args.seed + 1,
        vision_feature_file=args.val_vision_feature_file,
        sidecar_dir=args.val_sidecar_dir,
    )
    stats = standardization(train_features, train_actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RestrictedActionMLP(
        vision_dim=train_meta["image_dim"],
        state_dim=3,
        restricted_dim=len(RESTRICTED_FEATURE_NAMES),
        n_action_steps=args.n_action_steps,
        use_image=not args.no_image,
    ).to(device)
    model.set_normalization(**stats)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_features),
            torch.from_numpy(train_actions),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(val_features),
            torch.from_numpy(val_actions),
        ),
        batch_size=args.batch_size * 2,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for features, actions in train_loader:
            features = features.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features)
            loss = normalized_mse(outputs, actions, stats, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for features, actions in val_loader:
                features = features.to(device, non_blocking=True)
                actions = actions.to(device, non_blocking=True)
                val_losses.append(
                    float(normalized_mse(model(features), actions, stats, device).cpu())
                )
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch} train={train_loss:.6f} val={val_loss:.6f}")
        if val_loss < best:
            best = val_loss
            best_state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "policy_type": "restricted_action_mlp",
        "model_state_dict": best_state,
        "normalization": {
            key: value.tolist() for key, value in stats.items()
        },
        "config": {
            "image_dim": train_meta["image_dim"],
            "state_dim": 3,
            "restricted_dim": len(RESTRICTED_FEATURE_NAMES),
            "restricted_feature_names": list(RESTRICTED_FEATURE_NAMES),
            "n_action_steps": args.n_action_steps,
            "use_image": not args.no_image,
        },
        "data": {
            "train_path": str(args.data_dir.resolve()),
            "val_path": str(args.val_data_dir.resolve()),
            "source_sha256": {
                "train_dataset": _sha256(args.data_dir.resolve() / "data" / "dataset.npz"),
                "val_dataset": _sha256(args.val_data_dir.resolve() / "data" / "dataset.npz"),
                "restricted_config": _sha256(
                    Path(__file__).resolve().parent / "configs" / "restricted_perception.json"
                ),
            },
            "train_summary": {
                key: value for key, value in train_meta.items() if key != "valid_indices"
            },
            "val_summary": {
                key: value for key, value in val_meta.items() if key != "valid_indices"
            },
        },
        "history": history,
        "best_val_loss": best,
    }
    torch.save(checkpoint, output / "best_model.pth")
    (output / "training_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"PASS best_val_loss={best:.6f} checkpoint={output / 'best_model.pth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
