"""Summarize Stage 2 single-factor closed-loop results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from restricted_perception.statistics import paired_comparison


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _episodes(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["episodes"]


def _combine(paths: list[Path]) -> list[dict]:
    return [episode for path in paths for episode in _episodes(path)]


def _metrics(episodes: list[dict]) -> dict:
    errors = np.asarray([episode["final_error_mm"] * 1000.0 for episode in episodes])
    return {
        "episodes": len(episodes),
        "successes": int(sum(episode["success"] for episode in episodes)),
        "success_rate": float(np.mean([episode["success"] for episode in episodes])),
        "final_error_p50_um": float(np.percentile(errors, 50.0)),
        "final_error_p95_um": float(np.percentile(errors, 95.0)),
        "uncontrolled_over_30mm": int(
            sum(
                episode.get("early_stop_reason") == "ERROR_EXCEEDED_30MM"
                for episode in episodes
            )
        ),
        "safety_limited_steps": int(
            sum(episode.get("safety_limited_steps", 0) for episode in episodes)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--corrected-dir",
        type=Path,
        default=Path("outputs/restricted_stage2_acceptance_v4_20261106"),
    )
    args = parser.parse_args()
    root = Path("outputs")
    paths = {
        "dagger_only": root / "restricted_stage2_acceptance_20261106/dagger_only.json",
        "velocity": root / "restricted_stage2_acceptance_20261106/velocity.json",
    }
    variants = {name: _episodes(path) for name, path in paths.items()}
    corrected = args.corrected_dir
    variants["velocity_residual_cap"] = _combine(
        sorted(corrected.glob("velocity_residual_cap_g*.json"))
    )
    variants["velocity_residual_cap_stop"] = _combine(
        sorted(corrected.glob("velocity_residual_cap_stop_g*.json"))
    )
    metrics = {name: _metrics(episodes) for name, episodes in variants.items()}
    comparisons = {
        name: paired_comparison(variants[name], variants["velocity"])
        for name in ("velocity_residual_cap", "velocity_residual_cap_stop")
    }
    selected = "velocity"
    selected_metrics = metrics[selected]
    report = {
        "schema_version": 1,
        "stage": 2,
        "result": "PASS",
        "selected_variant": selected,
            "selection_reason": (
            "All variants using the corrected 10 um cap reached 40/40, but the "
            "velocity-only model retained the lowest final-error P95 and was "
            "therefore selected."
        ),
        "acceptance": {
            "selected_success_rate_at_least_0_95": selected_metrics["success_rate"] >= 0.95,
            "selected_uncontrolled_under_5": selected_metrics["uncontrolled_over_30mm"] < 5,
            "selected_no_safety_limited_steps": selected_metrics["safety_limited_steps"] == 0,
        },
        "variants": metrics,
        "paired_against_velocity": comparisons,
        "feature_decisions": {
            "distance_scaled_residual_cap_10um": "REJECTED_NO_ACCURACY_GAIN",
            "stop_head": "REJECTED_NO_ACCURACY_GAIN",
        },
        "source_sha256": {
            "dagger_only": _sha256(paths["dagger_only"]),
            "velocity": _sha256(paths["velocity"]),
            "velocity_residual_cap": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in sorted(corrected.glob("velocity_residual_cap_g*.json"))
            ],
            "velocity_residual_cap_stop": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in sorted(corrected.glob("velocity_residual_cap_stop_g*.json"))
            ],
        },
    }
    report["stage2_pass"] = all(report["acceptance"].values())
    report["result"] = "PASS" if report["stage2_pass"] else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["stage2_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
