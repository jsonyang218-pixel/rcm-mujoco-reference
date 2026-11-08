"""Build restricted coarse/fine sidecars for existing recovery datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from restricted_perception.config import load_config
from restricted_perception.sidecar import build_restricted_sidecar


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "restricted_perception.json",
    )
    parser.add_argument("--seed", type=int, default=20261201)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()
    metadata = build_restricted_sidecar(
        args.dataset,
        load_config(args.config),
        args.output_dir,
        seed=args.seed,
        max_episodes=args.max_episodes,
    )
    print(
        f"PASS samples={metadata['samples']} included_episodes={metadata['included_episodes']} "
        f"excluded_episodes={len(metadata['excluded_episodes'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
