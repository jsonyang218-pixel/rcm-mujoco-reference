"""Summarize Stage 3 comparisons, robustness gates, and distillation."""

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


def _episodes(path: Path) -> list[dict]:
    return _load(path)["episodes"]


def _combine(paths: list[Path]) -> list[dict]:
    return [episode for path in paths for episode in _episodes(path)]


def _metrics(episodes: list[dict]) -> dict:
    errors = np.asarray(
        [episode["final_error_mm"] * 1000.0 for episode in episodes],
        dtype=float,
    )
    steps = np.asarray([episode.get("steps", 0) for episode in episodes])
    return {
        "episodes": len(episodes),
        "successes": int(sum(episode["success"] for episode in episodes)),
        "success_rate": float(np.mean([episode["success"] for episode in episodes])),
        "final_error_p50_um": float(np.percentile(errors, 50.0)),
        "final_error_p95_um": float(np.percentile(errors, 95.0)),
        "minimum_error_p50_um": float(
            np.percentile(
                [episode.get("minimum_error_mm", episode["final_error_mm"]) * 1000.0 for episode in episodes],
                50.0,
            )
        ),
        "steps_p50": float(np.percentile(steps, 50.0)),
        "steps_p95": float(np.percentile(steps, 95.0)),
        "uncontrolled_over_30mm": int(
            sum(
                str(episode.get("early_stop_reason", "")).startswith(
                    "ERROR_EXCEEDED"
                )
                for episode in episodes
            )
        ),
        "safety_limited_steps": int(
            sum(episode.get("safety_limited_steps", 0) for episode in episodes)
        ),
    }


def _comparison(
    candidate: list[dict],
    baseline: list[dict],
    *,
    bootstrap_samples: int,
) -> dict:
    return paired_comparison(
        candidate,
        baseline,
        bootstrap_samples=bootstrap_samples,
    )


def _primary_gate(comparison: dict) -> dict:
    success_interval = comparison["success_difference_bootstrap_95"]
    error_interval = comparison[
        "paired_error_difference_bootstrap_95_um"
    ]
    success_rule = bool(success_interval[0] > 0.0)
    saturation_rule = bool(
        comparison["candidate_success_rate"] >= 0.95
        and comparison["baseline_success_rate"] >= 0.95
        and error_interval[1] < 0.0
        and comparison["relative_p95_error_improvement"] >= 0.10
    )
    return {
        "pass": bool(success_rule or saturation_rule),
        "success_risk_difference_lower_gt_zero": success_rule,
        "saturated_error_rule": saturation_rule,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/restricted_stage3_summary_20261106.json",
    )
    args = parser.parse_args()
    protocol = _load(ROOT / "configs/stage34_protocol.json")
    candidate = _episodes(
        ROOT
        / "outputs/restricted_stage3_robustness_20261106/default/student.json"
    )
    guided_no_dagger = _combine(
        sorted(
            (ROOT / "outputs/restricted_stage3_nodagger_protocol_seed_20261106").glob(
                "g*.json"
            )
        )
    )
    legacy_paths = {
        "old_structure_with_dagger_7_median": (
            ROOT
            / "outputs/restricted_stage3_baselines_20261106/legacy_diffusion_dagger.json"
        ),
        "raw_diffusion_7_median": (
            ROOT / "outputs/restricted_stage3_baselines_20261106/raw_diffusion.json"
        ),
        "guided_oct_regression": (
            ROOT / "outputs/restricted_stage3_baselines_20261106/regression.json"
        ),
    }
    variants = {
        "selected_restricted_student": candidate,
        "guided_no_dagger_action_chunk_mlp": guided_no_dagger,
    }
    for name, path in legacy_paths.items():
        if path.is_file():
            variants[name] = _episodes(path)
    comparisons = {
        "candidate_vs_guided_no_dagger": _comparison(
            candidate,
            guided_no_dagger,
            bootstrap_samples=int(
                protocol["stage3"]["paired_bootstrap_samples"]
            ),
        )
    }
    for name in legacy_paths:
        if name in variants:
            comparisons[f"candidate_vs_{name}"] = _comparison(
                candidate,
                variants[name],
                bootstrap_samples=int(
                    protocol["stage3"]["paired_bootstrap_samples"]
                ),
            )
    robustness = {}
    scenario_gains = 0
    for scenario in ("default", "noise", "occlusion", "oct_loss", "delay"):
        student_path = (
            ROOT
            / "outputs/restricted_stage3_robustness_20261106"
            / scenario
            / "student.json"
        )
        teacher_path = (
            ROOT
            / "outputs/restricted_stage3_robustness_20261106"
            / scenario
            / "geometric_teacher_sensor.json"
        )
        if not student_path.is_file() or not teacher_path.is_file():
            robustness[scenario] = {"result": "INCOMPLETE"}
            continue
        student = _episodes(student_path)
        teacher = _episodes(teacher_path)
        comparison = _comparison(
            student,
            teacher,
            bootstrap_samples=int(
                protocol["stage3"]["paired_bootstrap_samples"]
            ),
        )
        lower = comparison["success_difference_bootstrap_95"][0]
        gain = bool(
            lower
            > float(
                protocol["stage3"]["robustness"][
                    "success_difference_bootstrap_lower_must_exceed"
                ]
            )
        )
        scenario_gains += int(gain)
        robustness[scenario] = {
            "student": _metrics(student),
            "geometric_teacher": _metrics(teacher),
            "comparison": comparison,
            "student_significant_gain": gain,
        }
    distillation = {"result": "INCOMPLETE"}
    distillation_eval_dir = (
        ROOT
        / "outputs/restricted_stage3_residual_distilled_protocol_seed_20261106"
    )
    distillation_eval_paths = sorted(
        distillation_eval_dir.glob("g*.json")
    )
    teacher_dev_path = legacy_paths["old_structure_with_dagger_7_median"]
    if len(distillation_eval_paths) == 4 and teacher_dev_path.is_file():
        distilled = _combine(distillation_eval_paths)
        teacher_dev = _episodes(teacher_dev_path)
        comparison = _comparison(
            distilled,
            teacher_dev,
            bootstrap_samples=int(
                protocol["stage3"]["distillation"][
                    "noninferiority_bootstrap_samples"
                ]
            ),
        )
        margin = float(
            protocol["stage3"]["distillation"]["noninferiority_margin"]
        )
        lower = comparison["success_difference_bootstrap_95"][0]
        distillation = {
            "result": "PASS" if lower >= -margin else "FAIL",
            "noninferiority_margin": margin,
            "success_difference_lower": lower,
            "distilled": _metrics(distilled),
            "teacher_7_median": _metrics(teacher_dev),
            "comparison": comparison,
        }
    stage1_summary = _load(
        ROOT / "outputs/restricted_stage1_summary_v2_20261106.json"
    )
    primary_gate = {
        "pass": stage1_summary["primary_gate"]["pass"],
        "source": "stage1_corrected_primary_gate",
        "stage1_result": stage1_summary["result"],
    }
    safety_ok = all(
        metrics["uncontrolled_over_30mm"] < 5
        and metrics["safety_limited_steps"] == 0
        for metrics in (
            _metrics(candidate),
            *[
                value["student"]
                for value in robustness.values()
                if "student" in value
            ],
        )
    )
    robustness_pass = scenario_gains >= int(
        protocol["stage3"]["robustness"][
            "minimum_scenarios_with_student_gain"
        ]
    )
    distillation_pass = distillation.get("result") == "PASS"
    checks = {
        "inherited_stage1_primary_gate": primary_gate["pass"],
        "robust_learning_value_at_least_two_scenarios": robustness_pass,
        "distilled_single_step_noninferior": distillation_pass,
        "safety_within_bounds": bool(safety_ok),
    }
    report = {
        "schema_version": 1,
        "stage": 3,
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "primary_gate": primary_gate,
        "scenario_gains": scenario_gains,
        "variants": {
            name: _metrics(episodes) for name, episodes in variants.items()
        },
        "comparisons": comparisons,
        "robustness": robustness,
        "distillation": distillation,
        "act_decision": protocol["stage3"]["act_decision"],
        "source_sha256": {
            name: sha256_file(path)
            for name, path in {
                "selected_student": (
                    ROOT
                    / "outputs/restricted_stage2_variants_v5_20261106/dagger_only/best_model.pth"
                ),
                "guided_no_dagger": (
                    ROOT
                    / "outputs/restricted_image_vector_112_blur_occlusion_v2_20261106/best_model.pth"
                ),
                "distilled_residual": (
                    ROOT
                    / "outputs/restricted_stage3_residual_distilled_v7_20261106/best_model.pth"
                ),
            }.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
