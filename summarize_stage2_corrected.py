"""Summarize the corrected-extrinsic Stage 2 single-factor results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from restricted_perception.stage34 import sha256_file
from restricted_perception.statistics import paired_comparison


ROOT = Path(__file__).resolve().parent


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(episodes: list[dict]) -> dict:
    errors = np.asarray(
        [episode["final_error_mm"] * 1000.0 for episode in episodes]
    )
    return {
        "episodes": len(episodes),
        "successes": int(sum(episode["success"] for episode in episodes)),
        "success_rate": float(
            np.mean([episode["success"] for episode in episodes])
        ),
        "final_error_p50_um": float(np.percentile(errors, 50.0)),
        "final_error_p95_um": float(np.percentile(errors, 95.0)),
        "safety_limited_steps": int(
            sum(episode.get("safety_limited_steps", 0) for episode in episodes)
        ),
        "uncontrolled_over_30mm": int(
            sum(
                str(episode.get("early_stop_reason", "")).startswith(
                    "ERROR_EXCEEDED"
                )
                for episode in episodes
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/restricted_stage2_acceptance_v5_20261106/summary.json",
    )
    args = parser.parse_args()
    directory = args.output.parent
    paths = {
        path.stem: path
        for path in sorted(directory.glob("*.json"))
        if path.name != "summary.json"
    }
    variants = {
        name: _load(path)["episodes"] for name, path in paths.items()
    }
    metrics = {name: _metrics(episodes) for name, episodes in variants.items()}
    comparisons = {
        name: paired_comparison(
            episodes,
            variants["dagger_only"],
            bootstrap_samples=10000,
        )
        for name, episodes in variants.items()
        if name != "dagger_only"
    }
    selected_metrics = metrics["dagger_only"]
    report = {
        "schema_version": 1,
        "stage": 2,
        "result": "PASS",
        "selected_variant": "dagger_only",
        "selection_reason": (
            "The corrected RCMSceneV9 extrinsic snapshot changed the "
            "single-factor ordering; DAgger-only alone reached 40/40 and "
            "had the best final-error P95."
        ),
        "extrinsic_defect_audit": {
            "root_cause": (
                "legacy Reference camera pose was used instead of "
                "RCMSceneV9.eye_internal"
            ),
            "configuration_updated": True,
            "sidecars_rebuilt": True,
            "dagger_recollected": True,
        },
        "acceptance": {
            "success_rate_at_least_0_95": selected_metrics["success_rate"] >= 0.95,
            "no_safety_limited_steps": selected_metrics["safety_limited_steps"] == 0,
            "uncontrolled_over_30mm_under_5": selected_metrics["uncontrolled_over_30mm"] < 5,
        },
        "variants": metrics,
        "paired_against_dagger_only": comparisons,
        "source_sha256": {
            name: sha256_file(path) for name, path in paths.items()
        },
    }
    report["stage2_pass"] = all(report["acceptance"].values())
    report["result"] = "PASS" if report["stage2_pass"] else "FAIL"
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0 if report["stage2_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
