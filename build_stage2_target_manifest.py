"""Build 60 fresh Stage 2 training targets with two near-target starts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from restricted_perception.target_split import generate_training_target_set


ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "configs/stage2_training_targets.json")
    parser.add_argument("--seed", type=int, default=20270101)
    args = parser.parse_args()
    manifest = generate_training_target_set(ROOT, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(manifest["targets"]), "sha256": _sha256(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
