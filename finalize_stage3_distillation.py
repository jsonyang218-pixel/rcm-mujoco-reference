"""Finalize and validate a sharded Stage 3 distillation collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from restricted_perception.stage34 import sha256_file


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data/stage3_distillation_v2_20261106",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/stage34_protocol.json",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.data_dir / "runs").glob("*.json"))
    ]
    expected = 120
    if len(runs) != expected:
        raise ValueError(f"expected {expected} distillation runs, got {len(runs)}")
    ids = [run["start_id"] for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate distillation start IDs")
    metadata = {
        "schema_version": 1,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "student_checkpoint": str(
            (
                ROOT
                / "outputs/restricted_stage2_variants_v5_20261106/dagger_only/best_model.pth"
            ).resolve()
        ),
        "student_checkpoint_sha256": sha256_file(
            ROOT
            / "outputs/restricted_stage2_variants_v5_20261106/dagger_only/best_model.pth"
        ),
        "teacher_checkpoint": str(
            (
                ROOT
                / "outputs/diffusion_policy_v7_guided_dagger1_cached/best_model.pth"
            ).resolve()
        ),
        "teacher_checkpoint_sha256": sha256_file(
            ROOT
            / "outputs/diffusion_policy_v7_guided_dagger1_cached/best_model.pth"
        ),
        "teacher_samples": int(
            protocol["stage3"]["distillation"]["teacher_samples"]
        ),
        "aggregation": "median",
        "training_target_manifest": str(
            (
                ROOT
                / protocol["stage3"]["distillation"]["training_target_manifest"]
            ).resolve()
        ),
        "training_target_manifest_sha256": sha256_file(
            ROOT
            / protocol["stage3"]["distillation"]["training_target_manifest"]
        ),
        "runs": runs,
        "total_runs": len(runs),
        "total_samples": int(sum(run["samples"] for run in runs)),
        "training_split_only": True,
        "development_or_final_targets_used": False,
    }
    output = args.data_dir / "metadata.json"
    output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": "PASS",
                "runs": metadata["total_runs"],
                "samples": metadata["total_samples"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
