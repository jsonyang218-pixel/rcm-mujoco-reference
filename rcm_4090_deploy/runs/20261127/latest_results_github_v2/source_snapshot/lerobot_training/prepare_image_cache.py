"""Create a 224x224 RGB image cache for a converted LeRobot dataset."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

from PIL import Image
from tqdm import tqdm


def resize_one(source: Path, destination: Path, size: int):
    with Image.open(source) as image:
        resized = image.convert("RGB").resize(
            (size, size),
            Image.Resampling.BILINEAR,
        )
        resized.save(destination)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    source = Path(args.input_dir)
    destination = Path(args.output_dir)
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    if args.size < 1 or args.workers < 1:
        raise ValueError("size and workers must be positive")

    paths = json.loads(
        (source / "data" / "image_paths.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (source / "metadata.json").read_text(encoding="utf-8")
    )
    (destination / "data").mkdir(parents=True)
    (destination / "images").mkdir()

    shutil.copy2(source / "data" / "dataset.npz", destination / "data" / "dataset.npz")
    metadata["image_cache"] = {
        "source_directory": str(source.resolve()),
        "size_px": [args.size, args.size],
        "interpolation": "bilinear",
    }
    image_spec = metadata.setdefault("observation_space", {}).setdefault(
        "image", {}
    )
    image_spec["shape"] = [
        args.size,
        args.size,
        3,
    ]
    with (destination / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)

    def convert(job):
        index, relative_path = job
        source_path = source / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        relative_output = Path("images") / f"frame_{index:09d}.png"
        output_path = destination / relative_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resize_one(source_path, output_path, args.size)
        return str(relative_output)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        cached_paths = list(tqdm(
            executor.map(convert, enumerate(paths)),
            total=len(paths),
            desc=f"Resize {source.name}",
        ))
    with (destination / "data" / "image_paths.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(cached_paths, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"完成: {source} -> {destination} ({len(paths)} images)")


if __name__ == "__main__":
    main()
