"""Build the frozen target split manifest and write a verified report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from restricted_perception.target_split import build_manifest, verify_manifest


ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--final-seed", type=int, default=20261215)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "configs" / "target_split_manifest.json",
    )
    args = parser.parse_args()
    manifest = build_manifest(args.root, final_seed=args.final_seed)
    verification = verify_manifest(manifest)
    manifest["verification"] = verification
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hashes = {
        args.output.name: _sha256(args.output),
    }
    (args.output.parent / "target_split_manifest.sha256.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"result": verification["result"], **manifest["counts"]}, indent=2))
    return 0 if verification["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
