"""Read-only unified 20 um hold metrics for completed raw trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "runs/20261125/local_oct_branch_v1"
FAST5 = (0, 2, 6, 8, 10)
DEV20 = tuple(range(20))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def runs(values):
    values = np.asarray(values, dtype=np.bool_)
    edges = np.diff(np.r_[False, values, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def trajectory_metrics(path):
    document = read(path)
    if len(document.get("episodes", [])) != 1:
        return {"status": "invalid_raw", "reason": "episode_count"}
    episode = document["episodes"][0]
    trajectory = episode.get("trajectory", {})
    errors = np.asarray(trajectory.get("errors", []), dtype=np.float64) * 1000.
    records = episode.get("motor_records", [])
    if len(errors) < 1600 or len(records) < 1600:
        return {
            "status": "short_trajectory",
            "frames": int(min(len(errors), len(records))),
        }
    speed = np.asarray([
        float(record["tip_displacement_um"])
        / max(float(record["duration_s"]), 1e-12)
        for record in records[:1600]
    ])
    at20 = errors[:1600] <= 20.
    segments = runs(at20)
    lengths = [end - start + 1 for start, end in segments]
    max_cycles = int(max(lengths) if lengths else 0)
    best = int(np.argmax(lengths)) if lengths else None
    hold_start = segments[best][0] if best is not None else None
    hold_end = segments[best][1] if best is not None else None
    return {
        "status": "completed",
        "frames": int(len(errors[:1600])),
        "min_error_um": float(errors[:1600].min()),
        "final_error_um": float(episode["final_error_mm"] * 1000.),
        "entries_le_20_um": int(len(segments)),
        "max_consecutive_le_20_um_cycles": max_cycles,
        "max_consecutive_le_20_um_seconds": max_cycles * .05,
        "sustained_200_pass": bool(max_cycles >= 200),
        "last_200_max_error_um": float(errors[1400:1600].max()),
        "actual_speed_um_per_s_overall": float(speed[:1600].mean()),
        "actual_speed_um_per_s_last_200": float(speed[1400:1600].mean()),
        "hold_start_step": hold_start,
        "hold_end_step": hold_end,
        "raw_result": str(path),
    }


def expected_targets(group):
    if "fast5" in group:
        return FAST5
    if "dev20" in group:
        return DEV20
    return None


def discover_groups():
    groups = {}

    def add(name, root, expected):
        groups[name] = {
            "root": Path(root),
            "expected": expected,
        }

    add(
        "baseline_task2_step4000_fast5",
        ROOT / "github_reports/homog_label_fix_20261123_v3_s3_fast5_1600"
        / "baseline_task2_step4000/shards/tasks",
        FAST5,
    )
    add(
        "baseline_task2_step4000_dev20",
        ROOT / "runs/20261124/homog_label_fix_20261124_v5_3h/reports"
        / "homog_label_fix_20261124_v5_phase0_baseline_dev20/shards/tasks",
        DEV20,
    )
    add(
        "v4_fallback_F_step4000_fast5",
        ROOT / "runs/20261124/homog_label_fix_20261124_v4_6h/reports"
        / "homog_label_fix_20261124_v4_fallback_F_fast5/shards/tasks",
        FAST5,
    )
    add(
        "v4_fallback_F_step4000_dev20",
        ROOT / "runs/20261124/homog_label_fix_20261124_v5_3h/reports"
        / "homog_label_fix_20261124_v5_phase0_fallback_F_dev20/shards/tasks",
        DEV20,
    )
    v5_root = (
        ROOT / "runs/20261124/homog_label_fix_20261124_v5_3h/reports"
        / "homog_label_fix_20261124_v5_phase1"
    )
    for arm in ("A", "B"):
        for root in sorted(v5_root.glob(f"{arm}_step*_fast5")):
            step = root.name.split("_step", 1)[1].split("_", 1)[0]
            add(
                f"v5_{arm}_step{step}_fast5",
                root / "shards/tasks",
                FAST5,
            )
    return groups


def collect(groups):
    result = {}
    for name, spec in groups.items():
        targets = {}
        for index in spec["expected"]:
            path = (
                spec["root"]
                / f"eval_E{index:02d}/raw/evaluation_results.json"
            )
            if not path.is_file():
                targets[f"E{index:02d}"] = {
                    "status": "missing_raw",
                    "raw_result": str(path),
                }
                continue
            targets[f"E{index:02d}"] = trajectory_metrics(path)
        result[name] = targets
    return result


def markdown_report(metrics):
    lines = [
        "# Unified 20 um sustained-hold metrics",
        "",
        "Read-only metrics from completed raw trajectories. No extrapolation.",
        "",
        "| Model | Target | status | min um | final um | entries <=20 | "
        "max <=20 cycles | max <=20 s | sustained 200 | last-200 max um | "
        "speed overall um/s | speed last-200 um/s |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for model, targets in metrics.items():
        for target, row in targets.items():
            if row["status"] != "completed":
                lines.append(
                    f"| {model} | {target} | {row['status']} | | | | | | | | | |"
                )
            else:
                lines.append(
                    f"| {model} | {target} | completed | "
                    f"{row['min_error_um']:.3f} | "
                    f"{row['final_error_um']:.3f} | "
                    f"{row['entries_le_20_um']} | "
                    f"{row['max_consecutive_le_20_um_cycles']} | "
                    f"{row['max_consecutive_le_20_um_seconds']:.2f} | "
                    f"{row['sustained_200_pass']} | "
                    f"{row['last_200_max_error_um']:.3f} | "
                    f"{row['actual_speed_um_per_s_overall']:.3f} | "
                    f"{row['actual_speed_um_per_s_last_200']:.3f} |"
                )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = collect(discover_groups())
    write_json(output / "step1_existing_20um_metrics.json", {
        "status": "completed",
        "control_period_s": .05,
        "definitions": {
            "entry": "contiguous true_error_um <= 20 segment",
            "sustained_200_pass": "maximum contiguous cycles >= 200",
            "last_200": "steps 1400:1600",
            "actual_speed": "mean tip_displacement_um / duration_s per cycle",
        },
        "models": metrics,
    })
    (output / "step1_existing_20um_metrics.md").write_text(
        markdown_report(metrics), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "models": {
            name: len(values) for name, values in metrics.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
