"""Read-only integrity audit for the OCT three-group supplement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> None:
    expected = [
        "README.md",
        "provenance.json",
        "protocol.json",
        "metrics.json",
        "summary.csv",
        "endpoint_metrics.csv",
        "oct_audit.json",
        "raw_rgb_manifest.json",
        "slides_oct_update.html",
        "SHA256SUMS.json",
        "TEST_LOG_tail.txt",
    ]
    expected += [f"figures/figure_{index:02d}_{name}.{suffix}" for index, name in (
        (1, "A_scene_overview"),
        (2, "A_executed_trajectory"),
        (3, "A_error_convergence"),
        (4, "endpoint_status"),
        (5, "C_P2_recovery_timeline"),
        (6, "P3_depth_audit"),
    ) for suffix in ("png", "pdf")]
    missing = [path for path in expected if not (ROOT / path).is_file()]
    if missing:
        fail(f"missing expected files: {missing}")

    checksum_data = json.loads((ROOT / "SHA256SUMS.json").read_text(encoding="utf-8"))
    checksum_count = 0
    for relative, metadata in checksum_data["files"].items():
        path = ROOT / relative
        if not path.is_file():
            fail(f"checksum entry missing: {relative}")
        if path.stat().st_size != metadata["bytes"]:
            fail(f"size mismatch: {relative}")
        if sha256(path) != metadata["sha256"]:
            fail(f"sha256 mismatch: {relative}")
        checksum_count += 1

    raw_manifest = json.loads((ROOT / "raw_rgb_manifest.json").read_text(encoding="utf-8"))
    raw_count = 0
    for item in raw_manifest["items"]:
        source = REPO / item["source"]
        copied = ROOT / item["copied_path"]
        if not source.is_file() or not copied.is_file():
            fail(f"raw RGB path missing: {item['label']}")
        if source.read_bytes() != copied.read_bytes():
            fail(f"raw RGB bytes changed: {item['label']}")
        if sha256(source) != item["source_sha256"] or sha256(copied) != item["copied_sha256"]:
            fail(f"raw RGB hash mismatch: {item['label']}")
        if not item["byte_identical"]:
            fail(f"manifest did not record byte identity: {item['label']}")
        raw_count += 1

    oct_audit = json.loads((ROOT / "oct_audit.json").read_text(encoding="utf-8"))
    phase_count = 0
    npz_count = 0
    for phase in oct_audit["local_phase_audits"]:
        for key in ("unique_frame_ids", "sequence_strictly_increasing", "timestamp_strictly_increasing"):
            if not phase[key]:
                fail(f"OCT audit failed {key}: {phase['group']} {phase['phase']}")
        if phase["source"] != ["mujoco_ideal_registered_oct_simulation"]:
            fail(f"unexpected OCT source: {phase['source']}")
        if phase["depth_units"] != ["m"] or phase["navigation_depth_units"] not in (["m"], []):
            fail(f"unexpected OCT units: {phase['depth_units']} / {phase['navigation_depth_units']}")
        npz = phase["npz"]
        if npz is None or not npz["allow_pickle_false_loaded"]:
            fail(f"NPZ audit missing: {phase['group']} {phase['phase']}")
        with np.load(REPO / npz["path"], allow_pickle=False) as archive:
            if "feature_valid_mask" not in archive.files:
                fail(f"feature mask missing: {npz['path']}")
        phase_count += 1
        npz_count += 1

    metrics = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
    primary = metrics["primary"]
    if primary["A_no_occlusion"]["successful_targets"] != 3 or primary["C_occlusion_recovery"]["successful_targets"] != 3:
        fail("primary A/C success counts changed")
    if primary["B_occlusion_stop"]["successful_targets"] != 0 or primary["B_occlusion_stop"]["not_run_targets"] != 2:
        fail("primary B failure/NOT_RUN accounting changed")
    local = metrics["local_diagnostic"]
    if local["A_no_occlusion"]["attempted_targets"] != 1 or local["B_occlusion_stop"]["attempted_targets"] != 1:
        fail("local A/B attempted accounting is incorrect")
    if local["C_occlusion_recovery"]["attempted_targets"] != 3:
        fail("local C attempted accounting is incorrect")

    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    if protocol["strict_config_structure_diff_stop_vs_recovery"] != ["/navigation/visual_servo/occlusion_handling"]:
        fail("stop/recovery configuration diff is not isolated")

    html = (ROOT / "slides_oct_update.html").read_text(encoding="utf-8")
    for marker in ("79afacc", "not physical OCT", "NOT_RUN", "figure_01", "figure_06", "local fallback"):
        if marker not in html:
            fail(f"HTML marker missing: {marker}")

    image_count = 0
    for image_path in sorted((ROOT / "figures").glob("*.png")):
        with Image.open(image_path) as image:
            if image.width < 100 or image.height < 100:
                fail(f"figure raster is unexpectedly small: {image_path}")
        image_count += 1

    print(json.dumps({
        "result": "PASS",
        "expected_files": len(expected),
        "checksum_files": checksum_count,
        "raw_rgb_items": raw_count,
        "oct_phase_audits": phase_count,
        "oct_npz_checks": npz_count,
        "figure_pngs": image_count,
        "html_markers": 6,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
