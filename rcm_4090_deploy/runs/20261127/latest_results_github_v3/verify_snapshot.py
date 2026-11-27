"""Verify the frozen publication without loading models or running experiments."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


def verify(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_bytes())
    expected = {item["path"]: item for item in manifest["files"]}
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != root / "manifest.json"
    }
    if set(expected) != actual:
        raise ValueError("Manifest does not match snapshot paths")
    counts = {"files": 0, "bytes": 0, "json": 0, "jsonl": 0, "python": 0}
    forbidden_suffixes = {
        ".pth", ".pt", ".ckpt", ".safetensors", ".npy", ".npz", ".png",
        ".jpg", ".jpeg", ".mp4", ".avi", ".zip", ".pyc", ".pem", ".key",
    }
    token_patterns = [
        re.compile(rb"-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----"),
        re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
        re.compile(rb"github_pat_[A-Za-z0-9_]{30,}"),
    ]
    for relative, item in sorted(expected.items()):
        path = root / relative
        if path.is_symlink() or path.suffix.lower() in forbidden_suffixes:
            raise ValueError(f"Disallowed file: {relative}")
        if ".env" in path.name or path.name in {"id_rsa", "id_ed25519"}:
            raise ValueError(f"Private filename: {relative}")
        data = path.read_bytes()
        if len(data) != item["bytes"]:
            raise ValueError(f"Size mismatch: {relative}")
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {relative}")
        if any(pattern.search(data) for pattern in token_patterns):
            raise ValueError(f"Sensitive token pattern: {relative}")
        if path.suffix == ".json":
            json.loads(data)
            counts["json"] += 1
        elif path.suffix == ".jsonl":
            for line in data.splitlines():
                if line.strip():
                    json.loads(line)
            counts["jsonl"] += 1
        elif path.suffix == ".py":
            ast.parse(data, filename=relative)
            counts["python"] += 1
        counts["files"] += 1
        counts["bytes"] += len(data)
    inventory = json.loads((root / "capture_inventory.json").read_bytes())
    for item in inventory["files"]:
        archived = expected[item["path"]]
        if (archived["sha256"], archived["bytes"]) != (item["sha256"], item["bytes"]):
            raise ValueError(f"Source inventory mismatch: {item['path']}")
        if item["source_changed_during_read"]:
            raise ValueError(f"Source changed during capture: {item['path']}")
    return {"status": "passed", "counts": counts, "snapshot_at": inventory["snapshot_at"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.snapshot_dir), ensure_ascii=False, indent=2))
