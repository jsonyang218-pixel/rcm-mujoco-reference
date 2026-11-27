"""Precompute frozen ResNet features for fast policy fine-tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train_diffusion_policy import ResNetEncoder, ThreadBatchLoader


class ImagePathDataset(Dataset):
    def __init__(self, data_dir, image_paths, transform):
        self.data_dir = data_dir
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.data_dir / self.image_paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {"image": tensor}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_feature_encoder(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint.get("config", {})
    encoder = ResNetEncoder(
        output_dim=512,
        vision_mode=config.get("vision_mode", "global"),
    ).to(device)
    prefix = "vision_encoder."
    state = {
        key[len(prefix):]: value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith(prefix)
    }
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"vision encoder state mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder, checkpoint


def prepare_dataset(
    data_dir,
    encoder,
    checkpoint_path,
    checkpoint_sha256,
    checkpoint_epoch,
    device,
    batch_size,
    workers,
    force,
    limit,
):
    from train_diffusion_policy import image_transform_for_dataset

    output_path = data_dir / "data" / "vision_features.npy"
    metadata_path = data_dir / "data" / "vision_features.json"
    with np.load(
        data_dir / "data" / "dataset.npz",
        allow_pickle=False,
    ) as data:
        frame_count = int(len(data["action"]))
    image_paths = json.loads(
        (data_dir / "data" / "image_paths.json").read_text(
            encoding="utf-8"
        )
    )
    if len(image_paths) != frame_count:
        raise ValueError(
            f"{data_dir}: image count {len(image_paths)} != frames {frame_count}"
        )
    selected_count = frame_count if limit is None else min(limit, frame_count)
    if (
        not force
        and output_path.is_file()
        and metadata_path.is_file()
    ):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            int(metadata.get("num_frames", -1)) == selected_count
            and metadata.get("checkpoint_sha256") == checkpoint_sha256
        ):
            print(f"复用视觉特征缓存: {output_path}")
            return metadata

    dataset = ImagePathDataset(
        data_dir,
        image_paths[:selected_count],
        image_transform_for_dataset(data_dir, augment=False),
    )
    loader = ThreadBatchLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_threads=max(1, workers),
        seed=20261106,
        pin_memory=device.type == "cuda",
        prefetch_batches=4,
    )
    batches = []
    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc=f"Vision features {data_dir.name}",
            total=len(loader),
        ):
            features = encoder(
                batch["image"].to(device, non_blocking=True)
            )
            batches.append(features.cpu().numpy().astype(np.float16))
    features = np.concatenate(batches, axis=0)
    if features.shape != (selected_count, 512):
        raise RuntimeError(
            f"unexpected feature shape {features.shape}, "
            f"expected ({selected_count}, 512)"
        )
    if not np.all(np.isfinite(features)):
        raise RuntimeError("vision feature cache contains non-finite values")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp.npy")
    np.save(temporary, features, allow_pickle=False)
    temporary.replace(output_path)
    metadata = {
        "schema_version": 1,
        "num_frames": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "dtype": str(features.dtype),
        "source_checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": int(checkpoint_epoch),
        "preprocessing": "RGB, deterministic bilinear/ToTensor/ImageNet normalize",
        "augmentation": "disabled; frozen features are precomputed once",
    }
    metadata_temporary = metadata_path.with_name(
        metadata_path.name + ".tmp"
    )
    metadata_temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata_temporary.replace(metadata_path)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        required=True,
        help="Repeat for train and validation datasets.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Image-loader threads; CUDA feature encoding remains batched.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 1:
        parser.error("batch-size and workers must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, checkpoint = load_feature_encoder(
        args.checkpoint,
        device,
    )
    checkpoint_sha256 = sha256_file(args.checkpoint)
    summaries = []
    for data_dir in args.data_dir:
        summaries.append(
            prepare_dataset(
                data_dir.resolve(),
                encoder,
                args.checkpoint,
                checkpoint_sha256,
                int(checkpoint["epoch"]),
                device,
                args.batch_size,
                args.workers,
                args.force,
                args.limit,
            )
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
