"""Analyze final-target distribution shift and Stage 4 failure modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from restricted_perception.config import load_config
from restricted_perception.stage34 import (
    load_stage34_protocol,
    sha256_file,
    stage4_run_specs,
)
from restricted_perception.stage4_analysis import (
    bias_summary,
    failure_mode_summary,
    ks_comparison,
    load_stage4_episodes,
    target_geometry,
)
from rcm_fundus import eye_frame
from rcm_reference import Reference


ROOT = Path(__file__).resolve().parent


def _strip_vectors(geometry: dict) -> dict:
    return {
        key: value
        for key, value in geometry.items()
        if key != "_vectors"
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/stage34_protocol.json",
    )
    parser.add_argument(
        "--target-manifest",
        type=Path,
        default=ROOT / "configs/target_split_manifest.json",
    )
    parser.add_argument(
        "--stage4-dir",
        type=Path,
        default=ROOT / "outputs/restricted_stage4_20261106",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/restricted_stage4_failure_analysis_20261107.json",
    )
    args = parser.parse_args()
    protocol = load_stage34_protocol(args.protocol)
    manifest = json.loads(
        args.target_manifest.read_text(encoding="utf-8")
    )
    config = load_config(
        ROOT / "configs/restricted_perception.json"
    )
    reference = Reference()
    center, rotation, radii = eye_frame(reference)
    development = np.asarray(
        [
            target["point_mm"]
            for target in manifest["development_targets"]
        ],
        dtype=float,
    )
    final = np.asarray(
        [
            target["point_mm"]
            for target in manifest["final_test_targets"]
        ],
        dtype=float,
    )
    dev_geometry = target_geometry(
        development,
        center_world_mm=center,
        rotation=rotation,
        radii_mm=radii,
        robot_to_world=reference.robot_to_world,
    )
    test_geometry = target_geometry(
        final,
        center_world_mm=center,
        rotation=rotation,
        radii_mm=radii,
        robot_to_world=reference.robot_to_world,
    )
    distribution_tests = {
        key: ks_comparison(
            dev_geometry["_vectors"][key],
            test_geometry["_vectors"][key],
        )
        for key in (
            "normalized_z",
            "polar_angle_deg",
            "surface_error_um",
            "x_mm",
            "y_mm",
            "z_mm",
        )
    }
    dev_to_test, _ = cKDTree(final).query(development)
    dev_dev, _ = cKDTree(development).query(development, k=2)
    test_test, _ = cKDTree(final).query(final, k=2)
    development_episode_specs = [
        {
            "target_index": index,
            "seed": int(protocol["seed"]) + index,
            "target_id": target["id"],
        }
        for index, target in enumerate(manifest["development_targets"])
    ]
    test_episode_specs = stage4_run_specs(manifest, protocol)
    dev_bias = bias_summary(
        config,
        episodes=development_episode_specs,
        episode_id_prefix="stage3_default_student",
    )
    test_bias = bias_summary(
        config,
        episodes=test_episode_specs,
        episode_id_prefix="stage3_stage4_student",
    )
    bias_tests = {
        "coarse_bias_norm_um": ks_comparison(
            [row["coarse_bias_norm_um"] for row in dev_bias["rows"]],
            [row["coarse_bias_norm_um"] for row in test_bias["rows"]],
        ),
        "fine_bias_norm_um": ks_comparison(
            [row["fine_bias_norm_um"] for row in dev_bias["rows"]],
            [row["fine_bias_norm_um"] for row in test_bias["rows"]],
        ),
    }
    episodes = load_stage4_episodes(
        sorted(args.stage4_dir.glob("distilled_shard*.json"))
    )
    if len(episodes) != 180:
        raise ValueError(f"expected 180 Stage 4 episodes, got {len(episodes)}")
    modes = failure_mode_summary(episodes)
    target_outcomes = {}
    for episode in episodes:
        target_outcomes.setdefault(int(episode["target_index"]), []).append(
            episode
        )
    final_angles = test_geometry["_vectors"]["polar_angle_deg"]
    target_rows = []
    for target_index in sorted(target_outcomes):
        target_episodes = target_outcomes[target_index]
        target_rows.append(
            {
                "target_index": target_index,
                "target_id": target_episodes[0]["target_id"],
                "polar_angle_deg": float(final_angles[target_index]),
                "success_rate_20um": float(
                    np.mean(
                        [
                            episode["success_by_threshold_um"]["20"]
                            for episode in target_episodes
                        ]
                    )
                ),
                "minimum_error_p50_um": float(
                    np.median(
                        [
                            episode["minimum_error_mm"] * 1000.0
                            for episode in target_episodes
                        ]
                    )
                ),
            }
        )
    angle_bins = []
    for lower, upper in (
        (28.0, 31.0),
        (31.0, 33.0),
        (33.0, 35.0),
        (35.0, 38.0),
    ):
        selected = [
            row
            for row in target_rows
            if lower <= row["polar_angle_deg"] < upper
        ]
        angle_bins.append(
            {
                "angle_range_deg": [lower, upper],
                "targets": len(selected),
                "mean_target_success_rate_20um": (
                    float(
                        np.mean(
                            [
                                row["success_rate_20um"]
                                for row in selected
                            ]
                        )
                    )
                    if selected
                    else None
                ),
            }
        )
    target_angles = np.asarray(
        [row["polar_angle_deg"] for row in target_rows]
    )
    target_success = np.asarray(
        [row["success_rate_20um"] for row in target_rows]
    )
    target_minimum = np.asarray(
        [row["minimum_error_p50_um"] for row in target_rows]
    )
    angle_success_correlation = spearmanr(
        target_angles,
        target_success,
    )
    angle_error_correlation = spearmanr(
        target_angles,
        target_minimum,
    )
    stage0 = json.loads(
        (
            ROOT
            / "outputs/restricted_perception_stage0_v2_20261106/stage0_report.json"
        ).read_text(encoding="utf-8")
    )
    stage0_20 = stage0["precision_floor"]["runs"]["20.0"]
    report = {
        "schema_version": 1,
        "question": (
            "Is the final-test failure explained by target distribution "
            "shift, fine-channel bias above 20 um, or hold failure?"
        ),
        "target_distribution": {
            "development": _strip_vectors(dev_geometry),
            "final_test": _strip_vectors(test_geometry),
            "ks_tests_development_vs_final": distribution_tests,
            "development_to_final_nearest_mm": {
                "minimum": float(dev_to_test.min()),
                "p50": float(np.median(dev_to_test)),
                "p95": float(np.percentile(dev_to_test, 95.0)),
            },
            "within_development_nearest_mm": {
                "minimum": float(dev_dev[:, 1].min()),
                "p50": float(np.median(dev_dev[:, 1])),
                "p95": float(np.percentile(dev_dev[:, 1], 95.0)),
            },
            "within_final_nearest_mm": {
                "minimum": float(test_test[:, 1].min()),
                "p50": float(np.median(test_test[:, 1])),
                "p95": float(np.percentile(test_test[:, 1], 95.0)),
            },
            "region_separation": {
                "final_all_below_development_minimum_normalized_z": bool(
                    float(
                        test_geometry["normalized_eye_local_z"]["max"]
                    )
                    < float(
                        dev_geometry["normalized_eye_local_z"]["min"]
                    )
                ),
                "final_all_above_development_maximum_polar_angle": bool(
                    float(
                        test_geometry[
                            "polar_angle_from_posterior_axis_deg"
                        ]["min"]
                    )
                    > float(
                        dev_geometry[
                            "polar_angle_from_posterior_axis_deg"
                        ]["max"]
                    )
                ),
            },
            "within_final_target_gradient": {
                "angle_bins": angle_bins,
                "spearman_angle_vs_target_success_rate_20um": {
                    "rho": float(angle_success_correlation.statistic),
                    "pvalue": float(angle_success_correlation.pvalue),
                },
                "spearman_angle_vs_target_minimum_error_p50_um": {
                    "rho": float(angle_error_correlation.statistic),
                    "pvalue": float(angle_error_correlation.pvalue),
                },
            },
        },
        "restricted_bias": {
            "development": {
                key: value
                for key, value in dev_bias.items()
                if key != "rows"
            },
            "final_test": {
                key: value
                for key, value in test_bias.items()
                if key != "rows"
            },
            "ks_tests": bias_tests,
        },
        "stage0_perception_floor": {
            "near_target_p95_um": float(
                stage0_20["perception_error_p95_um"][
                    "fused_near_target"
                ]
            ),
            "fine_p95_um": float(
                stage0_20["perception_error_p95_um"]["fine"]
            ),
            "closed_loop_success_rate_at_20um": float(
                stage0_20["success_rate"]
            ),
            "selected_threshold_um": float(
                stage0["precision_floor"]["selected_threshold_um"]
            ),
        },
        "stage4_failure_modes": modes,
        "interpretation": {
            "target_location_is_out_of_development_distribution": bool(
                distribution_tests["normalized_z"]["statistic"] == 1.0
                and distribution_tests["polar_angle_deg"]["statistic"] == 1.0
            ),
            "fine_channel_bias_is_above_20um": bool(
                test_bias["fine_bias_above_20um_episodes"] > 0
            ),
            "fine_channel_was_available_inside_episodes": bool(
                modes.get("REACHED_20UM_BUT_DID_NOT_HOLD", {})
                .get("fallback_fraction", {})
                .get("FINE_TRACKING", {})
                .get("median_fraction", 0.0)
                > 0.9
            ),
            "primary_issue_is_policy_hold_stability": bool(
                modes.get("REACHED_20UM_BUT_DID_NOT_HOLD", {}).get(
                    "episodes",
                    0,
                )
                > modes.get("SUCCESS_20UM", {}).get("episodes", 0)
            ),
        },
        "scope": (
            "Simulation-only post-hoc failure analysis. It does not tune or "
            "retest the already-used final targets."
        ),
        "source_sha256": {
            "target_manifest": sha256_file(args.target_manifest),
            "stage4_summary": sha256_file(
                ROOT / "outputs/restricted_stage4_summary_20261106.json"
            ),
            "stage4_episodes": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                }
                for path in sorted(
                    args.stage4_dir.glob("distilled_shard*.json")
                )
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
