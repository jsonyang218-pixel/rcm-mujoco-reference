"""Reproduce two parameter comparisons without changing the project config."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from rcm_reference import Reference
from rcm_visual_servo import VisualServoController, visual_servo_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT / "runs" / "历史记录" / "03_studies" / f"visual_parameter_study_{datetime.now():%Y%m%d_%H%M%S_%f}",
    )
    args = parser.parse_args()
    variants = {
        "small_angular_difference": {
            "finite_difference_A_deg": 0.05,
            "finite_difference_B_deg": 0.05,
        },
        "lower_gain": {"step_gain": 0.3},
    }
    summary = []
    for name, overrides in variants.items():
        reference = Reference()
        settings = {**visual_servo_settings(reference), **overrides}
        directory = args.output_dir / name
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "effective_config.json").write_text(
            json.dumps(reference.config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        (directory / "effective_servo_settings.json").write_text(
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        controller = VisualServoController(reference, settings=settings, output_dir=directory)
        started = time.perf_counter()
        try:
            report = controller.run()
            controller.write(report, directory)
        except Exception:
            controller.sensor.close()
            raise
        summary.append({
            "variant": name,
            "overrides": overrides,
            "result": report["result"],
            "reason": report["reason"],
            "iterations": report["iterations"],
            "nominal_last_observation_time_s": controller.rows[-1]["time_s"] if controller.rows else None,
            "wall_time_s": time.perf_counter() - started,
            "final_tip_target_error_um": report["final_tip_target_error_um"],
            "endpoint_acceptance_pass": report["final_tip_target_error_um"] <= settings["endpoint_tolerance_um"],
            "trajectory_rate_limits_pass": report["trajectory_rate_limits_pass"],
        })
        print(json.dumps(summary[-1], ensure_ascii=False, indent=2), flush=True)
    (args.output_dir / "parameter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
