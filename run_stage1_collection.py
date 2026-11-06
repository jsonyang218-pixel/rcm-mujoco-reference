#!/usr/bin/env python3
"""Run the resumable single-target train/validation/test collection plan."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SPLITS = (
    ("train", 200, 20261031),
    ("val", 40, 20261231),
    ("test", 40, 20270131),
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "runs" / "paper_stage1")
    parser.add_argument("--min-success-rate", type=float, default=.90)
    parser.add_argument("--max-iterations", type=int, default=1600)
    parser.add_argument("--splits", nargs="+", choices=[name for name, _, _ in DEFAULT_SPLITS],
                        default=[name for name, _, _ in DEFAULT_SPLITS])
    args = parser.parse_args(argv)
    if not 0 <= args.min_success_rate <= 1:
        parser.error("--min-success-rate must be between 0 and 1")
    if args.max_iterations < 1:
        parser.error("--max-iterations must be positive")

    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    status_path = root / "stage1_status.json"
    status = {
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "output_root": str(root),
        "min_success_rate": args.min_success_rate,
        "splits": {},
    }
    write_json(status_path, status)

    selected = [entry for entry in DEFAULT_SPLITS if entry[0] in args.splits]
    for name, episodes, seed in selected:
        out = root / name
        item = {
            "status": "running",
            "episodes": episodes,
            "seed": seed,
            "output_dir": str(out),
            "started_at": datetime.now().astimezone().isoformat(),
        }
        status["splits"][name] = item
        write_json(status_path, status)
        command = [
            sys.executable, str(ROOT / "collect_v9_dataset.py"),
            "--output-dir", str(out),
            "--episodes", str(episodes),
            "--seed", str(seed),
            "--random-targets",
            "--max-iterations", str(args.max_iterations),
        ]
        if out.exists():
            command.append("--resume")
        started = time.monotonic()
        log_path = logs / f"{name}.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"\n[{datetime.now().astimezone().isoformat()}] "
                + " ".join(command) + "\n"
            )
            log.flush()
            completed = subprocess.run(
                command, cwd=ROOT, text=True, stdout=log,
                stderr=subprocess.STDOUT,
            )
        elapsed = time.monotonic() - started
        verification_path = out / "dataset_verification.json"
        if not verification_path.is_file():
            item.update(
                status="FAILED",
                returncode=completed.returncode,
                wall_s=elapsed,
                reason="dataset_verification.json missing",
                finished_at=datetime.now().astimezone().isoformat(),
            )
            status.update(status="FAILED", finished_at=datetime.now().astimezone().isoformat())
            write_json(status_path, status)
            return 1
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        successes = int(verification.get("navigation_successes", 0))
        count = int(verification.get("episodes", 0))
        success_rate = successes / count if count else 0.
        item.update(
            returncode=completed.returncode,
            wall_s=elapsed,
            verification_result=verification.get("result"),
            navigation_successes=successes,
            navigation_success_rate=success_rate,
            total_frames=verification.get("total_frames"),
            completed_at=datetime.now().astimezone().isoformat(),
        )
        if verification.get("result") != "PASS":
            item.update(status="FAILED", reason="dataset verification failed")
            status.update(status="FAILED", finished_at=datetime.now().astimezone().isoformat())
            write_json(status_path, status)
            return 1
        if success_rate < args.min_success_rate:
            item.update(status="LOW_SUCCESS_RATE", reason="navigation success rate below threshold")
            status.update(status="LOW_SUCCESS_RATE",
                          finished_at=datetime.now().astimezone().isoformat())
            write_json(status_path, status)
            return 2
        item["status"] = "PASS"
        write_json(status_path, status)

    status.update(
        status="PASS",
        finished_at=datetime.now().astimezone().isoformat(),
    )
    total_episodes = sum(item["episodes"] for item in status["splits"].values())
    total_successes = sum(item.get("navigation_successes", 0)
                          for item in status["splits"].values())
    status["overall_navigation_success_rate"] = (
        total_successes / total_episodes if total_episodes else 0.
    )
    write_json(status_path, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
