"""Build a compact training dataset from Stage 2 restricted DAgger episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
if str(ROOT / "lerobot_training") not in sys.path:
    sys.path.insert(0, str(ROOT / "lerobot_training"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--dagger-root", type=Path, required=True)
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        default=ROOT / "outputs/diffusion_policy_v6_guided_oct_finetune/best_model.pth",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    from prepare_vision_feature_cache import load_feature_encoder

    target_manifest = json.loads(args.targets.read_text(encoding="utf-8"))
    ordered = []
    for target in target_manifest["targets"]:
        for start in target["starts"]:
            path = (
                args.dagger_root / start["shard"] / f"{start['start_id']}.npz"
                if "shard" in start
                else None
            )
            if path is None or not path.is_file():
                matches = sorted(args.dagger_root.glob(f"*/{start['start_id']}.npz"))
                path = matches[0] if matches else None
            if path is None:
                raise FileNotFoundError(start["start_id"])
            ordered.append((target, start, path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _ = load_feature_encoder(args.encoder_checkpoint, device)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    feature_chunks = []
    arrays = {
        "episode_index": [],
        "frame_index": [],
        "state": [],
        "velocity": [],
        "restricted": [],
        "action_student": [],
        "action_teacher": [],
        "action_executed": [],
        "teacher_valid": [],
        "error_um": [],
        "fallback_state": [],
    }
    metadata_episodes = []
    pending_images = []
    pending_count = 0

    def flush_images():
        nonlocal pending_images, pending_count
        if not pending_images:
            return
        tensor = torch.stack(pending_images).to(device)
        with torch.no_grad():
            feature_chunks.append(encoder(tensor).cpu().numpy().astype(np.float16))
        pending_images = []
        pending_count = 0

    for episode_index, (target, start, path) in enumerate(ordered):
        with np.load(path) as episode:
            frame_count = len(episode["state"])
            for key in arrays:
                if key == "episode_index":
                    arrays[key].extend([episode_index] * frame_count)
                elif key == "frame_index":
                    arrays[key].extend(range(frame_count))
                else:
                    arrays[key].append(np.asarray(episode[key]))
            for image in episode["image"]:
                pending_images.append(transform(Image.fromarray(image, mode="RGB")))
                pending_count += 1
                if pending_count >= args.batch_size:
                    flush_images()
        metadata_episodes.append(
            {
                "episode_index": episode_index,
                "target_id": target["id"],
                "start_id": start["start_id"],
                "target_robot_mm": target["point_mm"],
                "initial_error_um": start["initial_error_um"],
                "seed": start["seed"],
                "source": str(path.resolve()),
                "frames": frame_count,
            }
        )
    flush_images()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: (
            np.asarray(values, dtype=np.int64)
            if key in ("episode_index", "frame_index")
            else np.concatenate(values, axis=0)
        )
        for key, values in arrays.items()
    }
    vision = np.concatenate(feature_chunks, axis=0)
    if len(vision) != len(arrays["state"]):
        raise ValueError("vision feature count does not match frame count")
    np.savez_compressed(output / "dataset.npz", **arrays)
    np.save(output / "vision_features.npy", vision, allow_pickle=False)
    metadata = {
        "schema_version": 1,
        "targets": str(args.targets.resolve()),
        "dagger_root": str(args.dagger_root.resolve()),
        "encoder_checkpoint": str(args.encoder_checkpoint.resolve()),
        "encoder_checkpoint_sha256": _sha256(args.encoder_checkpoint),
        "episodes": metadata_episodes,
        "num_episodes": len(metadata_episodes),
        "num_frames": int(len(arrays["state"])),
        "image_resolution": 112,
        "preprocessing": "student degraded RGB saved at collection, then frozen ResNet features",
        "teacher_label_scope": "full RGB-D geometry teacher at the same state",
        "truth_used_for_student_input": False,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "episodes": len(metadata_episodes),
                "frames": len(arrays["state"]),
                "vision_features": list(vision.shape),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
