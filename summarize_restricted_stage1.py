"""Summarize the one-factor image degradation ladder without overclaiming."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_increase(value: float, reference: float) -> float:
    return float(value / reference - 1.0) if reference > 0.0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "restricted_stage1_causal_ladder_20261106.json",
    )
    args = parser.parse_args()
    rows = [
        (
            "full_224",
            ROOT / "outputs/restricted_image_vector_v1_20261106",
            224,
            0.0,
            0.0,
        ),
        (
            "reduced_112",
            ROOT / "outputs/restricted_image_vector_112_v1_20261106",
            112,
            0.0,
            0.0,
        ),
        (
            "reduced_112_blur_1p5",
            ROOT / "outputs/restricted_image_vector_112_blur_v1_20261106",
            112,
            1.5,
            0.0,
        ),
        (
            "reduced_112_blur_1p5_center_occlusion_0p25",
            ROOT / "outputs/restricted_image_vector_112_blur_occlusion_v1_20261106",
            112,
            1.5,
            0.25,
        ),
    ]
    summaries = []
    for name, directory, resolution, blur, occlusion in rows:
        causal = _load(directory / "offline_causal.json")
        history = _load(directory / "training_history.json")
        normal = causal["near_target"]["normal"]["mean_action_offset_norm"]
        zero_vector = causal["near_target"]["zero_vector"]["mean_action_offset_norm"]
        mismatch_vector = causal["near_target"]["mismatch_vector"][
            "mean_action_offset_norm"
        ]
        zero_image = causal["near_target"]["zero_image"]["mean_action_offset_norm"]
        summaries.append(
            {
                "name": name,
                "image_resolution": resolution,
                "gaussian_blur_radius": blur,
                "center_occlusion_fraction": occlusion,
                "checkpoint": str((directory / "best_model.pth").resolve()),
                "best_val_loss": min(row["val_loss"] for row in history),
                "near_target_normal_action_error": normal,
                "zero_vector_relative_increase": _relative_increase(
                    zero_vector, normal
                ),
                "mismatch_vector_relative_increase": _relative_increase(
                    mismatch_vector, normal
                ),
                "zero_image_relative_increase": _relative_increase(
                    zero_image, normal
                ),
            }
        )
    report = {
        "schema_version": 1,
        "scope": "Offline near-target action causality; not closed-loop success.",
        "levels": summaries,
        "closed_loop_gate": "NOT_EVALUATED",
        "stage1_pass": False,
        "next_required_evidence": (
            "Run paired closed-loop normal/zero/mismatch vector evaluations on the "
            "40 development targets and apply target-paired bootstrap plus exact McNemar."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
