"""Annotate and verify strict-zero metadata without changing dataset arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tolerance-um", type=float, default=0.1)
    args = parser.parse_args()

    metadata_path = args.dataset_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("metadata.json lacks episode records")

    inferred = []
    failures = []
    for episode in episodes:
        final_error_um = episode.get("final_error_um")
        strict = episode.get("strict_zero_error")
        if strict is None:
            strict = bool(
                episode.get("success") is True
                and final_error_um is not None
                and float(final_error_um) <= args.tolerance_um
            )
            if strict:
                inferred.append(int(episode["episode_index"]))
        episode["strict_zero_error"] = bool(strict)
        if not strict:
            failures.append({
                "episode_index": episode.get("episode_index"),
                "success": episode.get("success"),
                "final_error_um": final_error_um,
            })
    if failures:
        raise ValueError(
            "strict-zero audit failed: "
            + json.dumps(failures[:10], ensure_ascii=False)
        )

    metadata["strict_zero_error"] = True
    metadata["strict_zero_error_episodes"] = len(episodes)
    metadata["strict_zero_error_tolerance_um"] = float(
        args.tolerance_um
    )
    metadata["strict_zero_error_metadata_inferred_episodes"] = inferred
    report = {
        "result": "PASS",
        "episodes": len(episodes),
        "frames": int(metadata["num_frames"]),
        "tolerance_um": float(args.tolerance_um),
        "maximum_final_error_um": max(
            float(episode["final_error_um"]) for episode in episodes
        ),
        "inferred_episode_count": len(inferred),
    }
    backup_path = args.dataset_dir / "metadata.pre_strict_audit.json"
    if not backup_path.exists():
        backup_path.write_text(
            metadata_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (args.dataset_dir / "strict_zero_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
