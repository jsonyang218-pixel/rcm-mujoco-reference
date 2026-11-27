"""Finalize a completed or intentionally stopped DAgger directory prefix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--requested-episodes", type=int, required=True)
    parser.add_argument("--starts-per-episode", type=int, required=True)
    parser.add_argument("--guided-oct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reason", default="COMPLETED")
    args = parser.parse_args()

    reports = []
    for path in sorted(args.collect_dir.glob("episode_*/control_report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        reports.append(report)
    if not reports:
        raise ValueError("no completed episode reports found")
    indexes = [int(report["episode_index"]) for report in reports]
    if len(indexes) != len(set(indexes)):
        raise ValueError("duplicate episode indexes")
    expected = set(range(args.requested_episodes))
    missing = sorted(expected - set(indexes))
    extra = sorted(set(indexes) - expected)
    if extra:
        raise ValueError(f"episode indexes outside requested range: {extra}")
    manifest = {
        "schema_version": 1,
        "format": "DAgger episodes with per-frame policy, teacher and executed actions",
        "model": str(args.model.resolve()),
        "policy_type": "oct_diffusion_policy",
        "policy_checkpoint_epoch": int(args.checkpoint_epoch),
        "policy_guided_oct": bool(args.guided_oct),
        "seed": int(args.seed),
        "requested_episodes": int(args.requested_episodes),
        "completed_episodes": len(reports),
        "missing_episode_indexes": missing,
        "finalization_reason": args.reason,
        "episodes": reports,
        "total_frames": sum(int(report["frames"]) for report in reports),
        "teacher_invalid_frames": sum(
            int(report.get("teacher_invalid_frames", 0))
            for report in reports
        ),
    }
    output = args.collect_dir / "manifest.json"
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "result": "PASS",
                "manifest": str(output),
                "episodes": len(reports),
                "missing_episode_indexes": missing,
                "total_frames": manifest["total_frames"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
