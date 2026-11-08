"""Run Stage 0 restricted perception acceptance and write a frozen report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

from restricted_perception.config import load_config
from restricted_perception.experiments import (
    compare_hard_blend_arrival,
    run_precision_scan,
    run_switch_scan,
)


ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "restricted_perception.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "restricted_perception_stage0_20261106",
    )
    parser.add_argument("--switch-episodes", type=int, default=100)
    parser.add_argument("--closed-loop-seeds-per-target", type=int, default=10)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    switch = run_switch_scan(config, episode_count=args.switch_episodes)
    precision = run_precision_scan(
        config,
        seeds_per_target=args.closed_loop_seeds_per_target,
    )
    arrival = compare_hard_blend_arrival(config, precision)
    checks = {
        "switch_step_reduction": switch["pass"],
        "precision_floor": precision["pass"],
        "blend_arrival_non_inferior": arrival["pass"],
    }
    report = {
        "schema_version": 1,
        "stage": 0,
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config.resolve()),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "switch_scan": switch,
        "precision_floor": precision,
        "hard_blend_arrival": arrival,
        "scope": (
            "Synthetic deterministic restricted-perception acceptance only; "
            "no learned policy or real OCT performance is claimed."
        ),
    }
    report_path = output / "stage0_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hashes = {
        "stage0_report.json": _sha256(report_path),
        "configs/restricted_perception.json": _sha256(
            ROOT / "configs" / "restricted_perception.json"
        ),
    }
    (output / "SHA256SUMS.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"result": report["result"], "checks": checks}, indent=2))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
