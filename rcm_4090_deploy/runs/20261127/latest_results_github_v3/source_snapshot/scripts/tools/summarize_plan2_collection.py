#!/usr/bin/env python3
"""Summarize plan-2 raw collection runs without copying images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize_manifest(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes", [])
    finals = [
        float(row["final_error_um"])
        for row in episodes
        if row.get("final_error_um") is not None
    ]
    return {
        "root": str(path.parents[2]),
        "run": path.parent.name,
        "category": path.parent.name.split("_", 1)[0].upper(),
        "requested": int(payload.get("requested_episodes", 0)),
        "completed": int(payload.get("completed_episodes", len(episodes))),
        "frames": int(payload.get("total_frames", 0)),
        "teacher_invalid_frames": int(
            payload.get("teacher_invalid_frames", 0)
        ),
        "strict_zero_error_episodes": int(
            payload.get("strict_zero_error_episodes", 0)
        ),
        "final_error_um": {
            "count": len(finals),
            "mean": sum(finals) / len(finals) if finals else None,
            "minimum": min(finals) if finals else None,
            "maximum": max(finals) if finals else None,
        },
        "skipped_empty_episodes": payload.get(
            "skipped_empty_episodes", []
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = []
    roots = []
    for root in args.root:
        plan_path = root / "collection_plan.json"
        plan = (
            json.loads(plan_path.read_text(encoding="utf-8"))
            if plan_path.is_file()
            else {}
        )
        roots.append({
            "root": str(root),
            "stage": plan.get("stage"),
            "expected_trajectories": plan.get("expected_trajectories"),
            "expected_counts": plan.get("expected_counts"),
        })
        for path in sorted((root / "raw").glob("*/manifest.json")):
            runs.append(summarize_manifest(path))
    result = {
        "runs": runs,
        "roots": roots,
        "completed_trajectories": sum(row["completed"] for row in runs),
        "completed_frames": sum(row["frames"] for row in runs),
        "teacher_invalid_frames": sum(
            row["teacher_invalid_frames"] for row in runs
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: result[key]
        for key in (
            "completed_trajectories",
            "completed_frames",
            "teacher_invalid_frames",
        )
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
