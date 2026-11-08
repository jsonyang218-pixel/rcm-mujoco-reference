"""Summarize corrected-extrinsic Stage 1 causal and paired comparisons."""

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


def _combine(paths: list[Path]) -> list[dict]:
    return [episode for path in paths for episode in _load(path)["episodes"]]


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
        "minimum_error_p95_um": float(
            np.percentile(
                [
                    episode.get(
                        "minimum_error_mm",
                        episode["final_error_mm"],
                    )
                    * 1000.0
                    for episode in episodes
                ],
                95.0,
            )
        ),
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
        default=ROOT / "outputs/restricted_stage1_summary_v2_20261106.json",
    )
    args = parser.parse_args()
    directory = ROOT / "outputs/restricted_stage1_closed_loop_v2_20261106"
    normal = _combine(sorted(directory.glob("normal_g*.json")))
    zero = _combine(sorted(directory.glob("zero_g*.json")))
    mismatch = _combine(sorted(directory.glob("mismatch_g*.json")))
    baseline = _combine(
        sorted(
            (ROOT / "outputs/restricted_stage1_baseline_v4_20261106").glob(
                "g*/evaluation_results.json"
            )
        )
    )
    variants = {
        "normal": normal,
        "zero": zero,
        "mismatch": mismatch,
        "v4_no_dagger_baseline": baseline,
    }
    metrics = {name: _metrics(episodes) for name, episodes in variants.items()}
    comparisons = {
        "normal_vs_zero": paired_comparison(
            normal,
            zero,
            bootstrap_samples=10000,
        ),
        "normal_vs_mismatch": paired_comparison(
            normal,
            mismatch,
            bootstrap_samples=10000,
        ),
        "normal_vs_v4_no_dagger": paired_comparison(
            normal,
            baseline,
            bootstrap_samples=10000,
        ),
    }
    zero_drop = metrics["normal"]["success_rate"] - metrics["zero"]["success_rate"]
    mismatch_drop = (
        metrics["normal"]["success_rate"] - metrics["mismatch"]["success_rate"]
    )
    causal_pass = zero_drop >= 0.10 and mismatch_drop >= 0.10
    primary = comparisons["normal_vs_v4_no_dagger"]
    primary_pass = bool(
        primary["success_difference_bootstrap_95"][0] > 0.0
        or (
            primary["candidate_success_rate"] >= 0.95
            and primary["relative_p95_error_improvement"] >= 0.10
            and primary["paired_error_difference_bootstrap_95_um"][1] < 0.0
        )
    )
    report = {
        "schema_version": 1,
        "stage": 1,
        "result": "PASS" if causal_pass and primary_pass else "FAIL",
        "extrinsic_audit": {
            "reason": (
                "Stage 1 rerun after replacing the legacy Reference camera "
                "pose with the actual RCMSceneV9 eye_internal optical pose"
            ),
            "offline_causal_ladder_from_prior_extrinsic_is_superseded": True,
        },
        "causal_closed_loop_gate": {
            "zero_success_drop": zero_drop,
            "mismatch_success_drop": mismatch_drop,
            "minimum_success_drop": 0.10,
            "pass": causal_pass,
        },
        "primary_gate": {
            "pass": primary_pass,
            "comparison": primary,
        },
        "variants": metrics,
        "comparisons": comparisons,
        "source_sha256": {
            path.name: sha256_file(path)
            for path in sorted(directory.glob("*_g*.json"))
        },
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
