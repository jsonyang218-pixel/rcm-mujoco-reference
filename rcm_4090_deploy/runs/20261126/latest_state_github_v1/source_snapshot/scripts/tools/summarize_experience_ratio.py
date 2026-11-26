#!/usr/bin/env python3
"""Summarize natural/10%/20% correction-ratio closed-loop experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_experience_replay_abc import (
    aggregate,
    load_episodes,
    paired,
)


ARMS = {
    "R0": "r0_natural",
    "R10": "r10",
    "R20": "r20",
}


def markdown(results):
    lines = [
        "# Correction Sampling Ratios",
        "",
        "| Metric | R0 natural | R10 | R20 |",
        "|---|---:|---:|---:|",
    ]
    fields = (
        ("Successes", lambda row: f"{row['successes']}/{row['episodes']}"),
        ("Final mean um", lambda row: f"{row['final_error_um']['mean']:.3f}"),
        ("Final median um", lambda row: f"{row['final_error_um']['median']:.3f}"),
        ("Final P90 um", lambda row: f"{row['final_error_um']['p90']:.3f}"),
        ("Final maximum um", lambda row: f"{row['final_error_um']['max']:.3f}"),
        ("Final error <=300 um", lambda row: str(
            sum(
                item["final_error_um"] <= 300.0
                for item in row["episodes_detail"]
            )
        )),
        (
            "Encoder settled at exit",
            lambda row: f"{row['encoder_settled_at_exit']}/{row['episodes']}",
        ),
        ("Safety-limited steps", lambda row: str(row["safety_limited_steps"])),
    )
    for label, getter in fields:
        lines.append(
            f"| {label} | {getter(results['R0'])} | "
            f"{getter(results['R10'])} | {getter(results['R20'])} |"
        )
    lines.extend(["", "## Paired final-error improvements", ""])
    for item in results["paired"]:
        lines.append(
            f"- `{item['candidate']}` vs `{item['reference']}`: "
            f"{item['improved']} improved, {item['worsened']} worsened, "
            f"mean improvement {item['mean_improvement_um']:+.3f} um, "
            f"median improvement {item['median_improvement_um']:+.3f} um."
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory already exists")
    args.output_dir.mkdir(parents=True)
    results = {
        label: aggregate(load_episodes(args.root / directory))
        for label, directory in ARMS.items()
    }
    results["paired"] = [
        paired(results["R0"], results["R10"], "R0", "R10"),
        paired(results["R0"], results["R20"], "R0", "R20"),
        paired(results["R10"], results["R20"], "R10", "R20"),
    ]
    (args.output_dir / "ratio_comparison.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RATIO_COMPARISON.md").write_text(
        markdown(results), encoding="utf-8"
    )
    print(json.dumps({
        label: {
            "successes": results[label]["successes"],
            "final_error_um": results[label]["final_error_um"],
            "safety_limited_steps": results[label]["safety_limited_steps"],
        }
        for label in ARMS
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
