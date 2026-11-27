#!/usr/bin/env python3
"""Validate the portable RCM deployment layout without importing project deps."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DATASETS = (
    "lerobot_rcm_train_guided_dagger1_full_224",
    "lerobot_rcm_val_recovery_full_224",
    "lerobot_rcm_test_recovery_full_224",
    "lerobot_rcm_train_recovery_full_224",
    "lerobot_rcm_train_224",
    "lerobot_rcm_val_224",
    "lerobot_rcm_test_224",
    "dagger_guided_v6_round1_all_224",
    "recovery_wave2_train_main_224",
    "recovery_wave2_train_a_224",
    "recovery_wave2_train_b_224",
    "recovery_wave2_val_224",
    "recovery_wave2_test_224",
    "recovery_pilot_train_224",
    "recovery_pilot_val_224",
    "recovery_pilot_test_224",
    "restricted_stage2_dagger_v2_20261106",
    "stage3_distillation_v2_20261106",
)

REQUIRED_ROOT_FILES = (
    "README.md",
    "FREEZE_MANIFEST_20261106.json",
    "TEST_LOG.md",
    "requirements.txt",
    "requirements-mesh.txt",
    "config.json",
    "v9_eye_scene.xml",
    "examples/three_waypoints.json",
    "examples/targets.json",
    "examples/rejected_targets.json",
)

REQUIRED_CODE_FILES = (
    "run_reference.py",
    "rcm/reference.py",
    "rcm/scene_v9.py",
    "scripts/applications/run_visual_servo.py",
    "scripts/tools/export_v9_eye_scene.py",
    "scripts/research/restricted_perception/train_stage3_residual_distillation.py",
    "scripts/research/restricted_perception/evaluate_restricted_policy.py",
    "scripts/datasets/collect_v9_dataset.py",
    "scripts/research/restricted_perception/run_stage4_distilled.py",
    "scripts/research/restricted_perception/run_stage3_robustness.py",
    "scripts/applications/run_waypoint_comparison.py",
    "scripts/tools/plot_waypoint_results.py",
    "rcm/front_tool.py",
    "scripts/verification/verify_v9_dataset.py",
    "lerobot_training/train_diffusion_policy.py",
)

REQUIRED_OUTPUTS = (
    "diffusion_policy_v7_guided_dagger1_cached/best_model.pth",
    "evaluation_v7_guided_dagger1_cached_20261106",
    "evaluation_v7_guided_20mm_repeat_20261106",
    "restricted_stage3_residual_distilled_v7_20261106/best_model.pth",
    "restricted_stage3_distilled_preserve_v6_20261106/best_model.pth",
    "restricted_stage2_variants_v5_20261106/dagger_only/best_model.pth",
    "diffusion_policy_v6_guided_oct_finetune/best_model.pth",
    "oct_guided_regression_v2_20261105",
    "restricted_stage3_residual_distilled_eval_v7_20261106",
)


def _exists(path: Path) -> bool:
    return path.exists()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_manifest(
    dataset_dir: Path,
    sample_size: int = 0,
) -> tuple[int, list[str]]:
    manifest = dataset_dir / "data" / "image_paths.json"
    if not manifest.is_file():
        return 0, []
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        return 0, [f"{manifest}: expected JSON array"]
    if sample_size > 0:
        entries = entries[:sample_size]
    errors = []
    normalized = []
    for index, value in enumerate(entries):
        if not isinstance(value, str) or "\\" in value:
            errors.append(f"{manifest}:{index}: non-POSIX path {value!r}")
            if len(errors) >= 20:
                break
            continue
        normalized.append((index, dataset_dir / value))

    def check(item: tuple[int, Path]) -> str | None:
        index, path = item
        return None if path.is_file() else f"{manifest}:{index}: missing {path}"

    with ThreadPoolExecutor(max_workers=32) as pool:
        for error in pool.map(check, normalized, chunksize=256):
            if error is not None:
                errors.append(error)
                if len(errors) >= 20:
                    break
    return len(entries), errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Check only the first N image entries per manifest; 0 checks all.",
    )
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Validate required code, outputs, and frozen files without scanning datasets.",
    )
    args = parser.parse_args()

    errors = []
    for relative in REQUIRED_ROOT_FILES + REQUIRED_CODE_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"missing root file: {relative}")
    for relative in REQUIRED_OUTPUTS:
        if not (ROOT / "outputs" / relative).exists():
            errors.append(f"missing output asset: outputs/{relative}")

    freeze = json.loads(
        (ROOT / "FREEZE_MANIFEST_20261106.json").read_text(encoding="utf-8")
    )
    checkpoint = ROOT / freeze["checkpoint"]["path"]
    if not checkpoint.is_file():
        errors.append(f"missing frozen checkpoint: {freeze['checkpoint']['path']}")
    else:
        if checkpoint.stat().st_size != int(freeze["checkpoint"]["size_bytes"]):
            errors.append("frozen checkpoint size mismatch")
        if _sha256(checkpoint) != freeze["checkpoint"]["sha256"]:
            errors.append("frozen checkpoint SHA-256 mismatch")
    for evaluation in freeze["evaluations"]:
        evaluation_root = ROOT / evaluation["path"]
        files = [
            path
            for path in evaluation_root.rglob("*")
            if path.is_file()
        ]
        if len(files) != int(evaluation["file_count"]):
            errors.append(
                f"{evaluation['path']}: expected {evaluation['file_count']} files, "
                f"found {len(files)}"
            )
        for entry in evaluation["files"]:
            path = evaluation_root / entry["path"]
            if not path.is_file():
                errors.append(f"missing evaluation file: {path}")
            elif _sha256(path) != entry["sha256"]:
                errors.append(f"evaluation SHA-256 mismatch: {path}")

    dataset_counts = {}
    manifest_paths = 0
    checked_paths = 0
    absolute_markers = 0
    json_count = 0
    if not args.code_only:
        for name in DATASETS:
            dataset = ROOT / "data" / name
            if not dataset.is_dir():
                errors.append(f"missing dataset: data/{name}")
                continue
            manifest = dataset / "data" / "image_paths.json"
            if not manifest.is_file():
                continue
            manifest_paths += 1
            total, manifest_errors = validate_manifest(dataset, args.sample_size)
            dataset_counts[name] = total
            checked_paths += total
            errors.extend(manifest_errors[:20])

        for path in (ROOT / "data").rglob("*.json"):
            text = path.read_text(encoding="utf-8")
            json.loads(text)
            json_count += 1
            absolute_markers += text.count("E:\\\\") + text.count("E:/")

    frozen_v7 = ROOT / "outputs" / "diffusion_policy_v7_guided_dagger1_cached"
    report = {
        "status": "PASS" if not errors and absolute_markers == 0 else "FAIL",
        "root": str(ROOT),
        "dataset_count": len(dataset_counts),
        "manifest_datasets": manifest_paths,
        "checked_image_entries": checked_paths,
        "json_files_parsed": json_count,
        "absolute_windows_path_markers": absolute_markers,
        "data_checks_enabled": not args.code_only,
        "frozen_v7_checkpoint_present": frozen_v7.exists(),
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
