"""Encode one-factor degraded images with the frozen v6 visual encoder."""

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
from torchvision import transforms
from tqdm import tqdm

from restricted_perception.image_ops import degrade_image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


class DegradedPathDataset(Dataset):
    def __init__(
        self,
        root: Path,
        image_paths: list[str],
        *,
        resolution: int,
        blur: float,
        occlusion: float,
    ):
        self.root = root
        self.image_paths = image_paths
        self.resolution = int(resolution)
        self.blur = float(blur)
        self.occlusion = float(occlusion)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with Image.open(self.root / self.image_paths[index]) as source:
            image = degrade_image(
                source,
                resolution=self.resolution,
                gaussian_blur_radius=self.blur,
                center_occlusion_fraction=self.occlusion,
            )
        return {"image": self.transform(image)}


def prepare(
    data_dir: Path,
    encoder,
    checkpoint_path: Path,
    resolution: int,
    blur: float,
    occlusion: float,
    batch_size: int,
    workers: int,
    device,
) -> dict:
    from train_diffusion_policy import ThreadBatchLoader

    data_dir = data_dir.resolve()
    image_paths = json.loads(
        (data_dir / "data" / "image_paths.json").read_text(encoding="utf-8")
    )
    dataset = DegradedPathDataset(
        data_dir,
        image_paths,
        resolution=resolution,
        blur=blur,
        occlusion=occlusion,
    )
    loader = ThreadBatchLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_threads=workers,
        seed=20261206,
        pin_memory=device.type == "cuda",
        prefetch_batches=4,
    )
    batches = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Degraded features {data_dir.name}"):
            batches.append(
                encoder(batch["image"].to(device, non_blocking=True))
                .cpu()
                .numpy()
                .astype(np.float16)
            )
    features = np.concatenate(batches, axis=0)
    suffix = f"r{resolution}_b{blur:g}_o{occlusion:g}".replace(".", "p")
    output = data_dir / "data" / f"restricted_vision_features_{suffix}.npy"
    metadata = {
        "schema_version": 1,
        "num_frames": int(len(features)),
        "feature_dim": int(features.shape[1]),
        "resolution": int(resolution),
        "gaussian_blur_radius": float(blur),
        "center_occlusion_fraction": float(occlusion),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "preprocessing": "degrade one factor -> ToTensor -> ImageNet normalize",
        "source_image_paths": str((data_dir / "data" / "image_paths.json").resolve()),
    }
    np.save(output, features, allow_pickle=False)
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "features": str(output),
        "shape": list(features.shape),
        **{key: value for key, value in metadata.items() if key not in ("schema_version", "preprocessing")},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=112)
    parser.add_argument("--gaussian-blur-radius", type=float, default=0.0)
    parser.add_argument("--center-occlusion-fraction", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if str(root / "lerobot_training") not in sys.path:
        sys.path.insert(0, str(root / "lerobot_training"))
    from prepare_vision_feature_cache import load_feature_encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _ = load_feature_encoder(args.checkpoint, device)
    summaries = [
        prepare(
            data_dir,
            encoder,
            args.checkpoint,
            args.resolution,
            args.gaussian_blur_radius,
            args.center_occlusion_fraction,
            args.batch_size,
            args.workers,
            device,
        )
        for data_dir in args.data_dir
    ]
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
