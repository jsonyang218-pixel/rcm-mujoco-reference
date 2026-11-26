#!/usr/bin/env python3
"""Collect the 280-trajectory expert-correction dataset in plan 2."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_SEEDS = {
    "train": list(range(20310000, 20310050)),
    "validation": list(range(20320000, 20320010)),
    "test": list(range(20330000, 20330010)),
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def selected(seeds, indices):
    return [seeds[index] for index in indices]


def run_spec(
    name,
    seeds,
    *,
    category,
    max_steps,
    hold_frames,
    teacher_takeover_steps=40,
    beta=0.5,
    profiles=None,
    start_error_mm=None,
    max_error_mm=15.0,
):
    if profiles is not None and start_error_mm is not None:
        raise ValueError("profile and pre-roll start error are exclusive")
    return {
        "name": name,
        "category": category,
        "seeds": list(seeds),
        "max_steps": int(max_steps),
        "hold_frames": int(hold_frames),
        "teacher_takeover_steps": int(teacher_takeover_steps),
        "beta": float(beta),
        "profiles": profiles,
        "start_error_mm": start_error_mm,
        "max_error_mm": float(max_error_mm),
    }


def split_runs(split):
    seeds = SPLIT_SEEDS[split]
    if split == "train":
        return train_runs(seeds)
    return validation_test_runs(split, seeds)


def pilot_runs():
    seeds = SPLIT_SEEDS["train"]
    return [
        run_spec(
            "d1_pilot_primary",
            selected(seeds, [0, 1, 2, 3, 4]),
            category="D1",
            max_steps=400,
            hold_frames=1,
            profiles=[11, 9, 8, 7, 6],
            max_error_mm=1.0,
        ),
        run_spec(
            "d1_pilot_extra",
            selected(seeds, [0, 1]),
            category="D1",
            max_steps=400,
            hold_frames=1,
            profiles=[13, 10],
            max_error_mm=1.0,
        ),
        run_spec(
            "d2_pilot_overshoot",
            selected(seeds, [0, 1, 2]),
            category="D2",
            max_steps=400,
            hold_frames=1,
            profiles=[9, 10, 11],
            max_error_mm=1.0,
        ),
        run_spec(
            "d2_pilot_oscillation",
            selected(seeds, [3, 4]),
            category="D2",
            max_steps=400,
            hold_frames=20,
            profiles=[16, 19],
            max_error_mm=1.0,
        ),
        run_spec(
            "d3_pilot_hold",
            selected(seeds, [0, 1, 2]),
            category="D3",
            max_steps=200,
            hold_frames=40,
            teacher_takeover_steps=20,
            beta=1.0,
            profiles=[0, 1, 7],
            max_error_mm=0.1,
        ),
        run_spec(
            "d3_pilot_perturb_hold",
            selected(seeds, [3, 4]),
            category="D3",
            max_steps=200,
            hold_frames=60,
            teacher_takeover_steps=20,
            beta=1.0,
            profiles=[12, 16],
            max_error_mm=0.1,
        ),
        run_spec(
            "d4_pilot_full",
            selected(seeds, [0, 1, 2]),
            category="D4",
            max_steps=1600,
            hold_frames=40,
            start_error_mm=10.0,
        ),
    ]


def d3_pilot_runs():
    seeds = SPLIT_SEEDS["train"]
    return [
        run_spec(
            "d3_pilot_hold_beta1",
            selected(seeds, [0, 1, 2]),
            category="D3",
            max_steps=200,
            hold_frames=40,
            teacher_takeover_steps=20,
            beta=1.0,
            profiles=[0, 1, 7],
            max_error_mm=0.1,
        ),
        run_spec(
            "d3_pilot_perturb_hold_beta1",
            selected(seeds, [3, 4]),
            category="D3",
            max_steps=200,
            hold_frames=60,
            teacher_takeover_steps=20,
            beta=1.0,
            profiles=[12, 16],
            max_error_mm=0.1,
        ),
    ]


def remaining_train_runs():
    seeds = SPLIT_SEEDS["train"]
    return [
        run_spec(
            "d1_300_1000um_remaining",
            selected(seeds, list(range(2, 20))),
            category="D1",
            max_steps=400,
            hold_frames=1,
            start_error_mm=0.50,
            max_error_mm=1.0,
        ),
        run_spec(
            "d1_100_300um_remaining",
            selected(seeds, list(range(20, 39))),
            category="D1",
            max_steps=400,
            hold_frames=1,
            start_error_mm=0.20,
            max_error_mm=0.5,
        ),
        run_spec(
            "d1_30_100um_remaining",
            selected(seeds, list(range(39, 50)) + list(range(2, 9))),
            category="D1",
            max_steps=400,
            hold_frames=1,
            start_error_mm=0.05,
            max_error_mm=0.15,
        ),
        run_spec(
            "d1_under_30um_remaining",
            selected(seeds, list(range(9, 17))),
            category="D1",
            max_steps=400,
            hold_frames=1,
            start_error_mm=0.02,
            max_error_mm=0.05,
        ),
        run_spec(
            "d2_overshoot_remaining",
            selected(seeds, list(range(3, 25))),
            category="D2",
            max_steps=400,
            hold_frames=1,
            profiles=[8, 9, 10, 11],
            max_error_mm=1.0,
        ),
        run_spec(
            "d2_oscillation_remaining",
            selected(seeds, list(range(5, 28))),
            category="D2",
            max_steps=400,
            hold_frames=20,
            profiles=[12, 13, 14, 15, 16, 17, 18, 19],
            max_error_mm=1.0,
        ),
        run_spec(
            "d3_hold_remaining",
            selected(seeds, list(range(3, 25))),
            category="D3",
            max_steps=200,
            hold_frames=40,
            teacher_takeover_steps=20,
            beta=1.0,
            profiles=[0, 1, 2, 6, 7],
            max_error_mm=0.1,
        ),
        run_spec(
            "d3_perturb_hold_remaining",
            selected(seeds, list(range(5, 28))),
            category="D3",
            max_steps=200,
            hold_frames=60,
            teacher_takeover_steps=20,
            beta=1.0,
            profiles=[12, 13, 14, 15, 16, 17, 18, 19],
            max_error_mm=0.1,
        ),
        run_spec(
            "d4_full_remaining",
            [
                seeds[unit * 5 + offset]
                for unit in range(1, 10)
                for offset in range(3)
            ],
            category="D4",
            max_steps=1600,
            hold_frames=40,
            start_error_mm=10.0,
        ),
    ]


def train_runs(seeds):
    return pilot_runs() + remaining_train_runs()


def validation_test_runs(split, seeds):
    return [
        run_spec(
            "d1_300_1000um",
            selected(seeds, [0, 1, 2, 3]),
            category="D1",
            max_steps=400,
            hold_frames=1,
            start_error_mm=0.50,
            max_error_mm=1.0,
        ),
        run_spec(
            "d1_100_300um",
            selected(seeds, [4, 5, 6, 7]),
            category="D1",
            max_steps=400,
            hold_frames=1,
            start_error_mm=0.20,
            max_error_mm=0.5,
        ),
        run_spec(
            "d1_30_100um",
            selected(seeds, [8, 9, 0, 1]),
            category="D1",
            max_steps=400,
            hold_frames=1,
            start_error_mm=0.05,
            max_error_mm=0.15,
        ),
        run_spec(
            "d1_under_30um",
            selected(seeds, [2, 3]),
            category="D1",
            max_steps=400,
            hold_frames=1,
            start_error_mm=0.02,
            max_error_mm=0.05,
        ),
        run_spec(
            "d2_overshoot",
            selected(seeds, [0, 1, 2, 3, 4]),
            category="D2",
            max_steps=400,
            hold_frames=1,
            profiles=[8, 9, 10, 11],
            max_error_mm=1.0,
        ),
        run_spec(
            "d2_oscillation",
            selected(seeds, [5, 6, 7, 8, 9]),
            category="D2",
            max_steps=400,
            hold_frames=20,
            profiles=[12, 13, 14, 15, 16],
            max_error_mm=1.0,
        ),
        run_spec(
            "d3_hold",
            selected(seeds, [0, 1, 2, 3, 4]),
            category="D3",
            max_steps=200,
            hold_frames=40,
            teacher_takeover_steps=20,
            beta=1.0,
            profiles=[0, 1, 2, 6, 7],
            max_error_mm=0.1,
        ),
        run_spec(
            "d3_perturb_hold",
            selected(seeds, [5, 6, 7, 8, 9]),
            category="D3",
            max_steps=200,
            hold_frames=60,
            teacher_takeover_steps=20,
            beta=1.0,
            profiles=[12, 13, 14, 15, 16],
            max_error_mm=0.1,
        ),
        run_spec(
            "d4_full",
            selected(seeds, [0, 1, 2, 5, 6, 7]),
            category="D4",
            max_steps=1600,
            hold_frames=40,
            start_error_mm=10.0,
        ),
    ]


def expected_counts(specs):
    counts = {"D1": 0, "D2": 0, "D3": 0, "D4": 0}
    for spec in specs:
        counts[spec["category"]] += len(spec["seeds"])
    return counts


def command_for(spec, model, dataset_root, seed, *, materialize):
    seed_path = dataset_root / "manifests" / f"{spec['name']}_seeds.json"
    if materialize:
        write_json(seed_path, {"seeds": spec["seeds"]})
    output = dataset_root / "raw" / spec["name"]
    command = [
        sys.executable,
        "-m",
        "scripts.datasets.collect_dagger_dataset",
        "--model",
        str(model),
        "--output-dir",
        str(output),
        "--episodes",
        str(len(spec["seeds"])),
        "--starts-per-episode",
        "1",
        "--seed",
        str(seed),
        "--beta",
        str(spec["beta"]),
        "--teacher-takeover-steps",
        str(spec["teacher_takeover_steps"]),
        "--teacher-takeover-max-steps",
        "400",
        "--hold-frames",
        str(spec["hold_frames"]),
        "--max-steps",
        str(spec["max_steps"]),
        "--max-error-mm",
        str(spec["max_error_mm"]),
        "--num-action-samples",
        "1",
        "--target-seeds-file",
        str(seed_path),
    ]
    if spec["profiles"] is not None:
        command.append("--terminal-start-profile-indices")
        command.extend(str(value) for value in spec["profiles"])
    else:
        command.extend([
            "--start-errors-mm",
            str(spec["start_error_mm"]),
        ])
    return command, output


def prepare_resume(item, interrupted_tag):
    output = Path(item["output"])
    manifest = output / "manifest.json"
    if manifest.is_file():
        item["status"] = "already_complete"
        return item
    if not output.exists():
        item["status"] = "pending"
        return item

    reports = sorted(output.glob("episode_*/control_report.json"))
    completed = sorted(
        int(path.parent.name.split("_")[-1]) for path in reports
    )
    if completed and completed != list(range(completed[-1] + 1)):
        raise ValueError(
            f"{output} completed episodes are not a contiguous prefix"
        )
    if not completed:
        destination = output.with_name(
            f"{output.name}_prestart_archived_{interrupted_tag}"
        )
        output.rename(destination)
        item["status"] = "fresh_after_archive"
        item["archived_output"] = str(destination)
        return item
    for episode_dir in sorted(output.glob("episode_*")):
        if not episode_dir.is_dir():
            continue
        if (episode_dir / "control_report.json").is_file():
            continue
        destination = output / (
            f"partial_{episode_dir.name}_{interrupted_tag}"
        )
        episode_dir.rename(destination)
        for name in (
            "images",
            "policy_inputs",
            "policy_inputs_no_image",
            "action_labels",
        ):
            partial = output / "dataset" / name / episode_dir.name
            if partial.is_dir():
                partial.rename(
                    output
                    / "dataset"
                    / name
                    / f"partial_{episode_dir.name}_{interrupted_tag}"
                )
    next_episode = completed[-1] + 1 if completed else 0
    if next_episode:
        item["command"].extend([
            "--resume-start-episode",
            str(next_episode),
        ])
    item["resume_start_episode"] = next_episode
    item["status"] = "resume"
    return item


def execute_item(item, dataset_root):
    logs = dataset_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{item['name']}.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as stream:
        print(
            json.dumps({
                "event": "start",
                "run": item["name"],
                "resume_start_episode": item.get(
                    "resume_start_episode", 0
                ),
            }),
            file=stream,
            flush=True,
        )
        code = subprocess.run(
            item["command"],
            cwd=PROJECT_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
        ).returncode
    return {
        "run": item["name"],
        "returncode": code,
        "log": str(log_path),
        "elapsed_s": time.monotonic() - started,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "pilot",
            "d3-pilot",
            "remaining",
            "validation",
            "test",
            "all",
        ),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20300101)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    model = args.model.resolve()
    if not model.is_file():
        parser.error("model checkpoint does not exist")
    if dataset_root.exists() and not args.dry_run and not args.resume:
        parser.error("dataset root already exists")
    if args.workers < 1:
        parser.error("workers must be positive")

    stages = {
        "pilot": pilot_runs(),
        "d3-pilot": d3_pilot_runs(),
        "remaining": remaining_train_runs(),
        "validation": split_runs("validation"),
        "test": split_runs("test"),
    }
    specs = (
        stages["pilot"]
        + stages["remaining"]
        + stages["validation"]
        + stages["test"]
        if args.stage == "all"
        else stages[args.stage]
    )
    plan = {
        "plan": "experience_replay_plan2_20261118",
        "stage": args.stage,
        "model": str(model),
        "dataset_root": str(dataset_root),
        "dry_run": args.dry_run,
        "workers": args.workers,
        "expected_counts": expected_counts(specs),
        "expected_trajectories": sum(
            len(spec["seeds"]) for spec in specs
        ),
        "runs": specs,
    }
    commands = []
    if not args.dry_run:
        dataset_root.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(specs):
        command, output = command_for(
            spec,
            model,
            dataset_root,
            args.seed + index * 100,
            materialize=not args.dry_run,
        )
        commands.append({
            "name": spec["name"],
            "category": spec["category"],
            "trajectories": len(spec["seeds"]),
            "output": str(output),
            "command": command,
            "status": "pending",
        })
    if args.resume and not args.dry_run:
        interrupted_tag = datetime.now().strftime(
            "interrupted_%Y%m%d_%H%M%S"
        )
        commands = [
            prepare_resume(item, interrupted_tag)
            for item in commands
        ]
        commands = [
            item for item in commands
            if item["status"] != "already_complete"
        ]
    plan["commands"] = commands
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    write_json(dataset_root / "collection_plan.json", plan)
    failures = []
    with ThreadPoolExecutor(
        max_workers=min(args.workers, max(len(commands), 1))
    ) as pool:
        futures = {
            pool.submit(execute_item, item, dataset_root): item
            for item in commands
        }
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result["returncode"]:
                failures.append(result)
    if failures:
        raise RuntimeError(
            f"{len(failures)} collection runs failed: {failures}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
