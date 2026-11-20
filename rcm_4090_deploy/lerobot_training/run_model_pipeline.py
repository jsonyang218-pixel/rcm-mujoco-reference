#!/usr/bin/env python3
"""Run training (including validation), closed-loop evaluation and recording."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def absolute(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def flags(options):
    result = []
    for key, value in options.items():
        if value is None:
            continue
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            result.append(option if value else "--no-" + option[2:])
        else:
            result.append(option)
            result.extend(map(str, value if isinstance(value, list) else [value]))
    return result


def start_logged(command, log_path, *, cwd, env, label):
    log = log_path.open("w", encoding="utf-8", buffering=1)
    try:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", start_new_session=True,
        )
    except BaseException:
        log.close()
        raise

    def tee():
        with process.stdout, log:
            for line in process.stdout:
                log.write(line)
                try:
                    print(f"[{label}] {line.rstrip()}", flush=True)
                except OSError:
                    pass  # Keep disk logging even if the output reader disconnects.

    reader = threading.Thread(target=tee, daemon=True)
    reader.start()
    return {"process": process, "reader": reader, "path": log_path}


def stop_logged(jobs, grace_seconds=5):
    active = [job["process"] for job in jobs if job["process"].poll() is None]
    for process in active:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    for process in active:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    for job in jobs:
        job["reader"].join(timeout=1)


class Pipeline:
    def __init__(self, root, output, config):
        self.root, self.output, self.config = root, output, config
        self.jobs = []
        self.state = {"status": "running", "stage": "preparing", "started_at": timestamp()}
        self.baseline = None
        self.environment = os.environ.copy()
        self.environment.update({key: str(value) for key, value in config.get("environment", {}).items()})
        self.environment["PYTHONDONTWRITEBYTECODE"] = "1"
        self.environment.setdefault("RCM_EVAL_PROGRESS_EVERY", "100")
        python_path = self.environment.get("PYTHONPATH", "")
        self.environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root), python_path]))

    def record(self, message):
        line = f"[{timestamp()}] {message}"
        try:
            print(line, flush=True)
        except OSError:
            pass
        with (self.output / "pipeline.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def stage(self, name, **values):
        self.state.update(stage=name, updated_at=timestamp(), **values)
        write_json(self.output / "status.json", self.state)

    def snapshot(self, evaluation_only):
        destination = self.output / "code"
        shutil.copytree(
            self.root / "lerobot_training", destination / "lerobot_training",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
        def freeze(value):
            entry = absolute(self.root, value)
            frozen = destination / entry.relative_to(self.root)
            if not frozen.is_file():
                frozen.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(entry, frozen)
            return entry, frozen

        entry, frozen_entry = freeze(self.config["model_entry"])
        training_entry = frozen_training_entry = None
        if not evaluation_only:
            training_entry, frozen_training_entry = freeze(
                self.config.get("training_entry", self.config["model_entry"])
            )
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.root, text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        metadata = {
            "model_name": self.config["model_name"],
            "model_entry": str(entry), "executed_model_entry": str(frozen_entry),
            "training_entry": str(training_entry) if training_entry else None,
            "executed_training_entry": str(frozen_training_entry) if frozen_training_entry else None,
            "git_commit": commit,
            "files": {str(p.relative_to(destination)): digest(p)
                      for p in sorted(destination.rglob("*")) if p.is_file()},
        }
        write_json(self.output / "code_version.json", metadata)
        return frozen_training_entry, frozen_entry

    def launch(self, command, log_path, **details):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with (self.output / "commands.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "started_at": timestamp(), "cwd": str(self.root),
                "argv": command, "log": str(log_path), **details,
            }, ensure_ascii=False) + "\n")
        self.record(shlex.join(command))
        self.record(f"日志: {log_path}")
        label = log_path.parent.name if details.get("stage") == "evaluation" else "train"
        job = start_logged(command, log_path, cwd=self.root, env=self.environment, label=label)
        job.update(details)
        self.jobs.append(job)
        return job

    def record_training_result(self, checkpoint):
        history_path = self.output / "training" / "training_history.json"
        if not history_path.exists():
            raise RuntimeError(
                f"训练完成但缺少训练历史: {history_path}"
            )
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not history:
            raise RuntimeError(f"训练历史为空: {history_path}")
        best = min(history, key=lambda row: row["val_loss"])
        last = history[-1]
        if not all(math.isfinite(float(row[key])) for row in history
                   for key in ("train_loss", "val_loss")):
            raise RuntimeError("训练历史包含非有限 loss")
        result = {
            "stage": "training_and_validation",
            "status": "completed",
            "model_name": self.config["model_name"],
            "finished_at": timestamp(),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "best_epoch": int(best["epoch"]),
            "best_train_loss": float(best["train_loss"]),
            "best_val_loss": float(best["val_loss"]),
            "last_epoch": int(last["epoch"]),
            "last_train_loss": float(last["train_loss"]),
            "last_val_loss": float(last["val_loss"]),
            "training_seconds": float(last["elapsed_s"]),
            "history": history,
        }
        write_json(self.output / "training" / "result.json", result)
        (self.output / "training" / "REPORT.md").write_text(
            f"# {self.config['model_name']} training result\n\n"
            "Training and validation are complete.\n\n"
            "| Metric | Result |\n|---|---:|\n"
            f"| Best epoch | {result['best_epoch']} |\n"
            f"| Best train loss | {result['best_train_loss']:.6f} |\n"
            f"| Best validation loss | {result['best_val_loss']:.6f} |\n"
            f"| Last epoch | {result['last_epoch']} |\n"
            f"| Last train loss | {result['last_train_loss']:.6f} |\n"
            f"| Last validation loss | {result['last_val_loss']:.6f} |\n"
            f"| Training seconds | {result['training_seconds']:.3f} |\n\n"
            f"Checkpoint: `{checkpoint}`\n\n"
            f"SHA-256: `{result['checkpoint_sha256']}`\n",
            encoding="utf-8",
        )
        self.record(
            "训练阶段完成；"
            f"best_epoch={result['best_epoch']} "
            f"best_val_loss={result['best_val_loss']:.6f} "
            f"checkpoint={checkpoint}；结果: {self.output / 'training' / 'result.json'}"
        )
        self.stage(
            "training_completed",
            training_status="completed",
            checkpoint=str(checkpoint),
            checkpoint_sha256=result["checkpoint_sha256"],
            best_epoch=result["best_epoch"],
            best_val_loss=result["best_val_loss"],
        )

    @staticmethod
    def episode_rows(episodes):
        rows = []
        for episode in episodes:
            terminal = episode.get("terminal_success", {})
            trajectory = episode.get("trajectory", {})
            errors = trajectory.get("errors", [])
            rows.append({
                "episode_index": episode["episode_index"],
                "target_id": episode.get("target_id"),
                "seed": episode.get("seed"),
                "success": bool(episode["success"]),
                "steps": episode["steps"],
                "final_error_um": episode["final_error_mm"] * 1000,
                "trajectory_minimum_error_um": (
                    min(errors or [episode["final_error_mm"]]) * 1000
                ),
                "low_speed_at_exit": terminal.get("low_speed_at_exit"),
                "encoder_settled_at_exit": terminal.get(
                    "encoder_settled_at_exit"
                ),
                "safety_limited_steps": episode.get(
                    "safety_limited_steps", 0
                ),
            })
        return rows

    @staticmethod
    def write_episode_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def record_evaluation_progress(
        self, shards, start, end, completed_indices
    ):
        episodes = []
        completed_shards = []
        for index in sorted(completed_indices):
            folder = self.output / "evaluation" / f"g{index + 1:02d}"
            path = folder / "evaluation_results.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("result") == "IN_PROGRESS" or not payload.get("evaluation_contract"):
                raise RuntimeError(f"评测分片未写出完整结果: {path}")
            shard_episodes = payload.get("episodes", [])
            episodes.extend(shard_episodes)
            completed_shards.append(f"g{index + 1:02d}")
            shard_result_path = folder / "result.json"
            if not shard_result_path.exists():
                shard_rows = self.episode_rows(shard_episodes)
                shard_result = {
                    "stage": "evaluation_shard",
                    "status": "completed",
                    "shard": f"g{index + 1:02d}",
                    "episodes": len(shard_rows),
                    "successes": sum(
                        row["success"] for row in shard_rows
                    ),
                    "episode_indices": [
                        row["episode_index"] for row in shard_rows
                    ],
                    "finished_at": timestamp(),
                    "source_result": str(path),
                }
                write_json(shard_result_path, shard_result)
        if not episodes:
            return
        episodes.sort(key=lambda row: row["episode_index"])
        rows = self.episode_rows(episodes)
        indices = [row["episode_index"] for row in rows]
        if len(indices) != len(set(indices)):
            raise RuntimeError("已完成的评测分片包含重复目标")
        finals = [row["final_error_um"] for row in rows]
        minimums = [
            row["trajectory_minimum_error_um"] for row in rows
        ]
        partial = {
            "stage": "evaluation",
            "status": "partial",
            "updated_at": timestamp(),
            "completed_shards": len(completed_shards),
            "total_shards": shards,
            "completed_shard_names": completed_shards,
            "completed_episodes": len(rows),
            "target_start": start,
            "target_end": end,
            "episode_indices": indices,
            "successes": sum(row["success"] for row in rows),
            "success_rate": sum(
                row["success"] for row in rows
            ) / len(rows),
            "encoder_settled_at_exit": sum(
                bool(row["encoder_settled_at_exit"]) for row in rows
            ),
            "final_error_um": {
                "mean": statistics.mean(finals),
                "median": statistics.median(finals),
                "min": min(finals),
                "max": max(finals),
            },
            "trajectory_minimum_error_mean_um": (
                statistics.mean(minimums)
            ),
            "safety_limited_steps": sum(
                int(row["safety_limited_steps"]) for row in rows
            ),
            "episodes": rows,
        }
        evaluation_dir = self.output / "evaluation"
        write_json(evaluation_dir / "partial_summary.json", partial)
        self.write_episode_csv(
            evaluation_dir / "partial_episodes.csv", rows
        )
        snapshot_dir = (
            evaluation_dir
            / "progress"
            / f"after_{len(completed_shards):02d}_shards"
        )
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        write_json(snapshot_dir / "summary.json", partial)
        self.write_episode_csv(
            snapshot_dir / "episodes.csv", rows
        )
        (snapshot_dir / "REPORT.md").write_text(
            "# Partial closed-loop result\n\n"
            f"Completed shards: "
            f"{len(completed_shards)}/{shards}\n\n"
            f"Completed episodes: {len(rows)}\n\n"
            f"Episode indices: {indices}\n\n"
            "| Metric | Result |\n|---|---:|\n"
            f"| Successes | {partial['successes']}/{len(rows)} |\n"
            f"| Final error mean | "
            f"{partial['final_error_um']['mean']:.3f} µm |\n"
            f"| Final error median | "
            f"{partial['final_error_um']['median']:.3f} µm |\n"
            f"| Minimum error mean | "
            f"{partial['trajectory_minimum_error_mean_um']:.3f} µm |\n\n"
            "This is a partial result. Do not treat it as the final "
            "evaluation.\n",
            encoding="utf-8",
        )
        self.record(
            f"评测阶段输出: {len(completed_shards)}/{shards} shards, "
            f"{len(rows)}/{end - start} episodes, "
            f"success={partial['successes']}/{len(rows)}, "
            f"{evaluation_dir / 'partial_summary.json'}"
        )

    def prepare_baseline(self, manifest_sha256, options, start, targets):
        reference = self.config.get("baseline_manifest")
        if reference is None:
            return
        baseline_path = absolute(self.root, reference)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        evaluation = baseline["evaluation"]
        expected_manifest_sha256 = evaluation["target_manifest"]["sha256"]
        if manifest_sha256 != expected_manifest_sha256:
            raise RuntimeError(
                "评测目标清单与 baseline 不一致: "
                f"{manifest_sha256} != {expected_manifest_sha256}"
            )
        protocol = evaluation["protocol"]
        range_keys = {"target_manifest", "target_start", "target_end", "episodes", "shards"}
        for key, expected in protocol.items():
            if key in range_keys:
                continue
            if options.get(key) != expected:
                raise RuntimeError(
                    f"评测协议与 baseline 不一致: {key}={options.get(key)!r}, "
                    f"baseline={expected!r}"
                )
        records = evaluation["records"]
        by_index = {int(row["episode_index"]): row for row in records}
        if len(by_index) != len(records):
            raise RuntimeError("baseline 含重复 episode 索引")
        selected = []
        for index, target in enumerate(targets, start):
            row = by_index.get(index)
            if row is None or not target.get("id") or row.get("target_id") != target["id"]:
                raise RuntimeError(f"baseline 缺少匹配的目标: episode {index}, target {target.get('id')}")
            seed = int(options.get("seed", 42)) + index
            if row.get("seed", seed) != seed:
                raise RuntimeError(f"baseline seed 不一致: episode {index}")
            selected.append(row)
        self.baseline = (baseline_path, baseline["baseline_id"], manifest_sha256, selected)

    def compare_with_baseline(self, rows, summary):
        if self.baseline is None:
            return None
        baseline_path, baseline_id, manifest_sha256, baseline_rows = self.baseline
        if [(row["episode_index"], row["target_id"]) for row in rows] != [
            (int(row["episode_index"]), row["target_id"]) for row in baseline_rows
        ]:
            raise RuntimeError("候选模型与 baseline 的 episode 索引或目标不一致")
        improvements = [
            float(baseline["final_error_um"]) - float(candidate["final_error_um"])
            for baseline, candidate in zip(baseline_rows, rows)
        ]
        baseline_finals = [float(row["final_error_um"]) for row in baseline_rows]
        baseline_mean = statistics.mean(baseline_finals)
        baseline_median = statistics.median(baseline_finals)
        baseline_minimum_mean = statistics.mean(float(row["minimum_error_um"]) for row in baseline_rows)
        baseline_successes = sum(bool(row["success"]) for row in baseline_rows)
        comparison = {
            "schema_version": 1,
            "status": "comparable",
            "baseline_id": baseline_id,
            "baseline_manifest": {
                "path": str(baseline_path),
                "sha256": digest(baseline_path),
            },
            "target_manifest_sha256": manifest_sha256,
            "episodes": len(rows),
            "episode_indices": [row["episode_index"] for row in rows],
            "baseline": {
                "successes": baseline_successes,
                "success_rate": baseline_successes / len(baseline_rows),
                "final_error_mean_um": baseline_mean,
                "final_error_median_um": baseline_median,
                "per_episode_minimum_error_mean_um": (
                    baseline_minimum_mean
                ),
                "safety_limited_steps": sum(int(row["safety_limited_steps"]) for row in baseline_rows),
            },
            "candidate": {
                "successes": int(summary["successes"]),
                "success_rate": float(summary["success_rate"]),
                "final_error_mean_um": float(
                    summary["final_error_um"]["mean"]
                ),
                "final_error_median_um": float(
                    summary["final_error_um"]["median"]
                ),
                "per_episode_minimum_error_mean_um": float(
                    summary["trajectory_minimum_error_mean_um"]
                ),
                "safety_limited_steps": int(sum(
                    row["safety_limited_steps"] for row in rows
                )),
            },
            "delta_candidate_minus_baseline": {
                "successes": int(summary["successes"]) - baseline_successes,
                "final_error_mean_um": float(
                    summary["final_error_um"]["mean"]
                ) - baseline_mean,
                "final_error_median_um": float(
                    summary["final_error_um"]["median"]
                ) - baseline_median,
                "per_episode_minimum_error_mean_um": float(
                    summary["trajectory_minimum_error_mean_um"]
                ) - baseline_minimum_mean,
            },
            "paired_final_error": {
                "definition": (
                    "baseline_final_error_um - "
                    "candidate_final_error_um"
                ),
                "improved": sum(value > 0 for value in improvements),
                "worsened": sum(value < 0 for value in improvements),
                "unchanged": sum(value == 0 for value in improvements),
                "mean_improvement_um": statistics.mean(improvements),
                "median_improvement_um": statistics.median(improvements),
            },
        }
        write_json(self.output / "comparison.json", comparison)
        (self.output / "COMPARISON.md").write_text(
            f"# Comparison against `{baseline_id}`\n\n"
            f"Target manifest SHA-256: "
            f"`{manifest_sha256}`\n\n"
            f"Paired targets: {len(rows)}\n\n"
            "| Metric | Baseline | Candidate | Candidate - baseline |\n"
            "|---|---:|---:|---:|\n"
            f"| Successes | {comparison['baseline']['successes']} | "
            f"{comparison['candidate']['successes']} | "
            f"{comparison['delta_candidate_minus_baseline']['successes']:+d} |\n"
            f"| Final error mean | "
            f"{baseline_mean:.3f} µm | "
            f"{comparison['candidate']['final_error_mean_um']:.3f} µm | "
            f"{comparison['delta_candidate_minus_baseline']['final_error_mean_um']:+.3f} µm |\n"
            f"| Final error median | "
            f"{baseline_median:.3f} µm | "
            f"{comparison['candidate']['final_error_median_um']:.3f} µm | "
            f"{comparison['delta_candidate_minus_baseline']['final_error_median_um']:+.3f} µm |\n"
            f"| Minimum error mean | "
            f"{baseline_minimum_mean:.3f} µm | "
            f"{comparison['candidate']['per_episode_minimum_error_mean_um']:.3f} µm | "
            f"{comparison['delta_candidate_minus_baseline']['per_episode_minimum_error_mean_um']:+.3f} µm |\n\n"
            f"Paired final error: "
            f"{comparison['paired_final_error']['improved']} improved, "
            f"{comparison['paired_final_error']['worsened']} worsened, "
            f"{comparison['paired_final_error']['unchanged']} unchanged. "
            f"Mean improvement "
            f"{comparison['paired_final_error']['mean_improvement_um']:+.3f} µm; "
            f"median improvement "
            f"{comparison['paired_final_error']['median_improvement_um']:+.3f} µm.\n",
            encoding="utf-8",
        )
        self.record(
            f"baseline 对照完成: {baseline_id}, "
            f"success_delta="
            f"{comparison['delta_candidate_minus_baseline']['successes']:+d}, "
            f"final_error_mean_delta="
            f"{comparison['delta_candidate_minus_baseline']['final_error_mean_um']:+.3f} um；"
            f"结果: {self.output / 'comparison.json'}"
        )
        return comparison

    def train(self, entry):
        checkpoint_name = self.config.get("checkpoint", "best_model.pth")
        if checkpoint_name in {"", ".", ".."} or Path(checkpoint_name).name != checkpoint_name:
            raise ValueError("checkpoint 必须是本次训练目录下的文件名")
        self.stage("training_and_validation")
        options = dict(self.config["training"])
        for key in ("data_dir", "val_data_dir", "init_from"):
            if options.get(key) is not None:
                options[key] = str(absolute(self.root, options[key]))
        options["output_dir"] = str(self.output / "training")
        mode = [] if "training_entry" in self.config else ["train"]
        command = [sys.executable, "-u", str(entry), *mode, *flags(options)]
        job = self.launch(command, self.output / "training.log", stage="training_and_validation")
        code = job["process"].wait()
        job["reader"].join(timeout=1)
        if code:
            raise RuntimeError(f"训练失败，退出码 {code}；见 {job['path']}")
        checkpoint = self.output / "training" / checkpoint_name
        if not checkpoint.is_file():
            raise RuntimeError(f"训练结束但缺少本次运行的 checkpoint: {checkpoint}")
        self.record_training_result(checkpoint)
        return checkpoint

    def prepare_evaluation(self, *, persist=True):
        options = dict(self.config["evaluation"])
        shards = int(options.pop("shards", 10))
        manifest_path = absolute(self.root, options["target_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split = options.get("target_split", "development")
        targets = manifest["targets"] if "targets" in manifest else manifest[f"{split}_targets"]
        start = int(options.get("target_start", 0))
        requested_end = options.get("target_end")
        end = min(len(targets) if requested_end is None else int(requested_end),
                  len(targets), start + int(options.get("episodes", 20)))
        count = end - start
        if start < 0 or count < 1 or shards < 1:
            raise ValueError("目标起点必须非负，评测目标数量和分片数量必须大于零")
        shards = min(shards, count)
        manifest_sha256 = digest(manifest_path)
        options.update(target_manifest=str(manifest_path), target_split=split)
        selected = targets[start:end]
        self.prepare_baseline(manifest_sha256, options, start, selected)
        if persist:
            write_json(self.output / "evaluation_manifest.json", {
                "source": str(manifest_path), "sha256": manifest_sha256, "target_split": split,
                "target_start": start, "target_end": end, "targets": selected,
            })
        return options, shards, start, end, selected

    def evaluate(self, entry, checkpoint, plan):
        options, shards, start, end, selected = plan
        self.stage("evaluation", completed_shards=0, total_shards=shards)
        count = end - start
        options = dict(options, model=str(checkpoint))
        pending = []
        for index in range(shards):
            lower = start + count * index // shards
            upper = start + count * (index + 1) // shards
            folder = self.output / "evaluation" / f"g{index + 1:02d}"
            shard_options = dict(options, target_start=lower, target_end=upper,
                                 episodes=upper - lower, output_dir=str(folder))
            command = [sys.executable, "-u", str(entry), "eval", *flags(shard_options)]
            pending.append(self.launch(
                command,
                folder / "eval.log",
                stage="evaluation",
                shard_index=index,
                target_start=lower,
                target_end=upper,
            ))
        completed = 0
        completed_indices = []
        while pending:
            for job in pending[:]:
                code = job["process"].poll()
                if code is None:
                    continue
                job["reader"].join(timeout=1)
                if code:
                    raise RuntimeError(f"评测分片失败，退出码 {code}；见 {job['path']}")
                path = job["path"].parent / "evaluation_results.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (payload.get("result") == "IN_PROGRESS" or not payload.get("evaluation_contract")
                        or sorted(row["episode_index"] for row in payload["episodes"])
                        != list(range(job["target_start"], job["target_end"]))):
                    raise RuntimeError(f"评测分片缺少完整目标结果: {path}")
                pending.remove(job)
                completed += 1
                completed_indices.append(int(job["shard_index"]))
                self.stage("evaluation", completed_shards=completed, total_shards=shards)
                self.record(f"评测完成 {completed}/{shards} 个分片")
                self.record_evaluation_progress(
                    shards, start, end, completed_indices
                )
            if pending:
                time.sleep(0.25)
        episodes, contract = [], None
        for index in range(shards):
            path = self.output / "evaluation" / f"g{index + 1:02d}" / "evaluation_results.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            episodes.extend(result["episodes"])
            if contract is None:
                contract = result.get("evaluation_contract", {})
            elif result.get("evaluation_contract", {}) != contract:
                raise RuntimeError(f"评测分片协议不一致: {path}")
        # Do not report a partial/duplicated evaluation as a complete experiment.
        episodes.sort(key=lambda row: row["episode_index"])
        if [row["episode_index"] for row in episodes] != list(range(start, end)):
            raise RuntimeError("评测结果缺少目标或包含重复目标；已保留分片日志和结果")
        for row, target in zip(episodes, selected):
            if row.get("target_id") != target.get("id"):
                raise RuntimeError(f"评测结果目标 ID 不一致: episode {row['episode_index']}")
            if row.get("seed") != int(options.get("seed", 42)) + row["episode_index"]:
                raise RuntimeError(f"评测结果 seed 不一致: episode {row['episode_index']}")
        return episodes, contract

    def summarize(self, episodes, contract, checkpoint, evaluation_only):
        self.stage("recording_results")
        rows = self.episode_rows(episodes)
        finals = [row["final_error_um"] for row in rows]
        history_path = self.output / "training" / "training_history.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else None
        summary = {
            "pipeline_status": "completed", "run_id": self.output.name,
            "model_name": self.config["model_name"], "model_entry": self.config["model_entry"],
            "started_at": self.state["started_at"], "finished_at": timestamp(),
            "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
            "checkpoint_source": "eval_only" if evaluation_only else "trained_in_this_run",
            "episodes": len(rows), "successes": sum(row["success"] for row in rows),
            "success_rate": statistics.mean(row["success"] for row in rows),
            "all_targets_successful": all(row["success"] for row in rows),
            "strict_successes": sum(row["success"] for row in rows),
            "final_error_um": {"mean": statistics.mean(finals), "median": statistics.median(finals),
                               "min": min(finals), "max": max(finals)},
            "trajectory_minimum_error_mean_um": statistics.mean(row["trajectory_minimum_error_um"] for row in rows),
            "encoder_settled_at_exit": sum(bool(row["encoder_settled_at_exit"]) for row in rows),
            "training_history": history, "evaluation_contract": contract,
            "episode_results": rows,
        }
        comparison = self.compare_with_baseline(rows, summary)
        if comparison is not None:
            summary["baseline_comparison"] = comparison
        write_json(self.output / "summary.json", summary)
        with (self.output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(self.output / "evaluation" / "evaluation_results.json", {
            "num_episodes": len(episodes), "success_rate": summary["success_rate"],
            "avg_steps": statistics.mean(row["steps"] for row in rows),
            "avg_final_error_mm": summary["final_error_um"]["mean"] / 1000,
            "median_final_error_mm": summary["final_error_um"]["median"] / 1000,
            "evaluation_contract": contract, "episodes": episodes,
        })
        last = history[-1] if history else {}
        report = (
            f"# {self.config['model_name']}\n\n"
            f"运行目录：`{self.output}`\n\n模型权重：`{checkpoint}`\n\n"
            "流程：所选范围全部完成；这不等于指标改善或严格成功。\n\n"
            "| 指标 | 结果 |\n|---|---:|\n"
            f"| 按评测协议严格成功 | {summary['strict_successes']}/{len(rows)} |\n"
            f"| 最终误差均值 | {statistics.mean(finals):.3f} μm |\n"
            f"| 最终误差中位数 | {statistics.median(finals):.3f} μm |\n"
            f"| 最坏最终误差 | {max(finals):.3f} μm |\n"
            f"| 退出时编码器停稳 | {summary['encoder_settled_at_exit']}/{len(rows)} |\n"
        )
        if comparison is not None:
            delta = comparison["delta_candidate_minus_baseline"]["final_error_mean_um"]
            report += f"\n指标对照：最终误差均值差 {delta:+.3f} μm（候选减基线），见 `COMPARISON.md`。\n"
        else:
            report += "\n指标改善：无配对基线，不判定。\n"
        if history:
            report += (f"| 最后一轮训练 loss | {last['train_loss']:.6f} |\n"
                       f"| 最后一轮验证 loss | {last['val_loss']:.6f} |\n")
        report += ("\n最后一轮 loss 不一定对应被选中的 best checkpoint；权重的 epoch 与 loss 见 "
                   "summary.json 的 evaluation_contract。轨迹最小误差沿用动作前记录口径。\n")
        (self.output / "REPORT.md").write_text(report, encoding="utf-8")
        self.record(f"成功 {summary['successes']}/{len(rows)}；最终平均误差 {statistics.mean(finals):.3f} μm")
        return summary

    def record_experiment(self, summary):
        training = self.config.get("training", {})
        evaluation = self.config.get("evaluation", {})
        state = self.state

        def relative(path):
            path = Path(path)
            return path.relative_to(self.root) if path.is_relative_to(self.root) else path

        checkpoint = state.get("checkpoint")
        lines = [
            f"\n- {state['finished_at']}：模型流水线 `{self.config['model_name']}` "
            f"**{state['status']}**（阶段 `{state['stage']}`；配置 `{relative(state['config'])}`）。",
            f"  运行目录 `{relative(self.output)}`；命令和状态见 `commands.jsonl`、`status.json`。",
        ]
        if state["mode"] == "train_and_evaluate":
            lines.append(f"  训练集 `{training.get('data_dir')}`，验证集 `{training.get('val_data_dir')}`，"
                         f"初始化 `{training.get('init_from')}`，训练 seed `{training.get('seed')}`。")
        if checkpoint:
            lines.append(f"  评测 checkpoint `{relative(checkpoint)}`；"
                         f"SHA-256 `{(summary or state).get('checkpoint_sha256', '未记录')}`。")
        lines.append(f"  评测清单 `{evaluation.get('target_manifest')}`，split `{evaluation.get('target_split', 'development')}`，"
                     f"seed `{evaluation.get('seed', 42)}`，协议见本次 `config.json`。")
        if "best_val_loss" in state:
            lines.append(f"  最佳验证 loss `{state['best_val_loss']:.6f}`（epoch {state['best_epoch']}）。")
        if summary:
            lines.append(f"  `summary.json`：成功 {summary['successes']}/{summary['episodes']}，"
                         f"最终误差均值 {summary['final_error_um']['mean']:.3f} µm。")
            if comparison := summary.get("baseline_comparison"):
                delta = comparison["delta_candidate_minus_baseline"]["final_error_mean_um"]
                lines.append(f"  相对 `{comparison['baseline_id']}` 的配对最终误差均值差 {delta:+.3f} µm，见 `comparison.json`。")
        if "error" in state:
            lines.append(f"  失败或中断原因：{state['error']}。")
            if (self.output / "evaluation" / "partial_summary.json").is_file():
                partial = json.loads((self.output / "evaluation" / "partial_summary.json").read_text(encoding="utf-8"))
                lines.append(f"  部分结果：{partial['completed_shards']}/{partial['total_shards']} 分片、"
                             f"{partial['completed_episodes']} 目标，见 `evaluation/partial_summary.json`。")
            elif (self.output / "training" / "training_history.json").is_file() and "best_epoch" not in state:
                lines.append("  部分训练结果见 `training/training_history.json`。")
        lines.append("  结果仅适用于 MuJoCo 仿真。\n")
        with (self.root / "TEST_LOG.md").open("a", encoding="utf-8") as stream:
            stream.write("\n".join(lines))

    def stop_children(self):
        stop_logged(self.jobs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/state_attention_pipeline.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--eval-only", type=Path, metavar="CHECKPOINT")
    parser.add_argument("--training-max-val-loss", type=float)
    args = parser.parse_args(argv)
    if args.training_max_val_loss is not None and (
        args.eval_only is not None or not math.isfinite(args.training_max_val_loss)
    ):
        parser.error("--training-max-val-loss requires training and a finite threshold")
    root = args.project_root.expanduser().resolve()
    config_path = absolute(root, args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_name = config["model_name"] + "_" + datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = absolute(root, args.output_dir or Path("outputs") / run_name)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    pipeline = Pipeline(root, output, config)
    evaluation_only = args.eval_only is not None
    summary = None
    try:
        pipeline.stage("preparing", config=str(config_path),
                       mode="eval_only" if evaluation_only else "train_and_evaluate")
        pipeline.record(f"运行目录: {output}")
        if evaluation_only:
            checkpoint = absolute(root, args.eval_only)
            pipeline.state["checkpoint"] = str(checkpoint)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"评测 checkpoint 不存在: {checkpoint}")
        elif not config.get("training"):
            raise ValueError("配置缺少训练参数；请选完整流水线配置，或通过 --eval-only 指定权重")
        plan = pipeline.prepare_evaluation()
        training_entry, evaluation_entry = pipeline.snapshot(evaluation_only)
        if not evaluation_only:
            checkpoint = pipeline.train(training_entry)
            if (args.training_max_val_loss is not None
                    and pipeline.state["best_val_loss"] > args.training_max_val_loss):
                raise RuntimeError(
                    f"验证 loss 门禁失败: {pipeline.state['best_val_loss']} > "
                    f"{args.training_max_val_loss}；未启动闭环评测"
                )
        episodes, contract = pipeline.evaluate(evaluation_entry, checkpoint, plan)
        summary = pipeline.summarize(episodes, contract, checkpoint, evaluation_only)
        pipeline.stage("completed", status="completed", finished_at=timestamp())
        pipeline.record(f"结果已记录: {output / 'REPORT.md'}")
        return 0
    except (Exception, KeyboardInterrupt) as error:
        pipeline.state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                              error=f"{type(error).__name__}: {error}", finished_at=timestamp())
        write_json(output / "status.json", pipeline.state)
        pipeline.record(f"流程停止: {error}")
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        pipeline.stop_children()
        pipeline.record_experiment(summary if pipeline.state["status"] == "completed" else None)


def interrupt_pipeline(signum, _frame):
    raise KeyboardInterrupt(signal.Signals(signum).name)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt_pipeline)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, interrupt_pipeline)
    raise SystemExit(main())
