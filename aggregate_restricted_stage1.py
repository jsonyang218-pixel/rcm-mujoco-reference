"""Aggregate paired Stage 1 closed-loop normal and vector-ablation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from restricted_perception.causal import image_leakage_gate, summarize_counterfactual
from restricted_perception.statistics import paired_comparison


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load(pattern: str) -> tuple[dict, list[Path]]:
    paths = sorted(Path(".").glob(pattern))
    episodes = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for episode in payload.get("episodes", []):
            key = (episode["target_id"], int(episode["seed"]))
            if key in episodes:
                raise ValueError(f"duplicate paired episode {key}")
            episodes[key] = episode
    if not episodes:
        raise FileNotFoundError(pattern)
    return episodes, paths


def _summary(candidate: dict, baseline: dict) -> dict:
    keys = sorted(set(candidate).intersection(baseline))
    if len(keys) != 40:
        raise ValueError(f"expected 40 paired targets, found {len(keys)}")
    return summarize_counterfactual(
        np.asarray([candidate[key]["success"] for key in keys], dtype=bool),
        np.asarray([baseline[key]["success"] for key in keys], dtype=bool),
        np.asarray([candidate[key]["final_error_mm"] * 1000.0 for key in keys]),
        np.asarray([baseline[key]["final_error_mm"] * 1000.0 for key in keys]),
        target_ids=[key[0] for key in keys],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    normal, normal_paths = _load(str(args.input_dir / "normal_g*.json"))
    zero, zero_paths = _load(str(args.input_dir / "zero_g*.json"))
    mismatch, mismatch_paths = _load(str(args.input_dir / "mismatch_g*.json"))
    baseline, baseline_paths = _load(
        str(args.baseline_dir / "g*" / "evaluation_results.json")
    )
    normal_keys = set(normal)
    if not normal_keys == set(zero) == set(mismatch) == set(baseline):
        raise ValueError("normal, zero, mismatch, and baseline target sets differ")
    zero_summary = summarize_counterfactual(
        np.asarray([normal[key]["success"] for key in sorted(normal_keys)], dtype=bool),
        np.asarray([zero[key]["success"] for key in sorted(normal_keys)], dtype=bool),
        np.asarray([normal[key]["final_error_mm"] * 1000.0 for key in sorted(normal_keys)]),
        np.asarray([zero[key]["final_error_mm"] * 1000.0 for key in sorted(normal_keys)]),
        target_ids=[key[0] for key in sorted(normal_keys)],
    )
    mismatch_summary = summarize_counterfactual(
        np.asarray([normal[key]["success"] for key in sorted(normal_keys)], dtype=bool),
        np.asarray([mismatch[key]["success"] for key in sorted(normal_keys)], dtype=bool),
        np.asarray([normal[key]["final_error_mm"] * 1000.0 for key in sorted(normal_keys)]),
        np.asarray([mismatch[key]["final_error_mm"] * 1000.0 for key in sorted(normal_keys)]),
        target_ids=[key[0] for key in sorted(normal_keys)],
    )
    image_gate = image_leakage_gate(zero_summary, mismatch_summary)
    primary = paired_comparison(
        list(normal.values()),
        list(baseline.values()),
    )
    uncontrolled = {
        name: sum(
            int(episode.get("early_stop_reason") == "ERROR_EXCEEDED_30MM")
            for episode in episodes.values()
        )
        for name, episodes in (
            ("normal", normal),
            ("zero", zero),
            ("mismatch", mismatch),
            ("baseline", baseline),
        )
    }
    safety_limited = {
        name: sum(int(episode.get("safety_limited_steps", 0)) for episode in episodes.values())
        for name, episodes in (
            ("normal", normal),
            ("zero", zero),
            ("mismatch", mismatch),
            ("baseline", baseline),
        )
    }
    report = {
        "schema_version": 1,
        "stage": 1,
        "input_dir": str(args.input_dir.resolve()),
        "paired_targets": len(normal_keys),
        "image_causal_gate": image_gate,
        "primary_vs_no_dagger_baseline": primary,
        "zero_vector": zero_summary,
        "mismatch_vector": mismatch_summary,
        "uncontrolled_over_30mm": uncontrolled,
        "safety_limited_steps": safety_limited,
        "source_files": {
            "normal": [
                {"path": str(path), "sha256": _sha256(path)} for path in normal_paths
            ],
            "zero": [
                {"path": str(path), "sha256": _sha256(path)} for path in zero_paths
            ],
            "mismatch": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in mismatch_paths
            ],
            "baseline": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in baseline_paths
            ],
        },
        "remaining_gate": {},
        "stage1_pass": False,
    }
    primary_passes = (
        primary["success_difference_bootstrap_95"][0] > 0.0
        or (
            primary["candidate_success_rate"] == primary["baseline_success_rate"] == 1.0
            and primary["paired_error_difference_bootstrap_95_um"][1] < 0.0
            and primary["relative_p95_error_improvement"] >= 0.10
        )
    )
    no_uncontrolled = all(value == 0 for value in uncontrolled.values())
    report["gates"] = {
        "image_causal": image_gate["result"] == "PASS",
        "primary_success_risk_difference_lower_gt_zero": bool(
            primary["success_difference_bootstrap_95"][0] > 0.0
        ),
        "primary_paired_rule": bool(primary_passes),
        "no_uncontrolled_over_30mm": bool(no_uncontrolled),
    }
    report["stage1_pass"] = all(report["gates"].values())
    report["result"] = "PASS" if report["stage1_pass"] else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": report["result"], **image_gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
