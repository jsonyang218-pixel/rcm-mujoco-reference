#!/usr/bin/env python3
"""Run training (including validation), closed-loop evaluation and recording."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
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


class Pipeline:
    def __init__(self, root, output, config):
        self.root, self.output, self.config = root, output, config
        self.jobs = []
        self.state = {"status": "running", "stage": "preparing", "started_at": timestamp()}
        self.environment = os.environ.copy()
        self.environment.update({key: str(value) for key, value in config.get("environment", {}).items()})
        python_path = self.environment.get("PYTHONPATH", "")
        self.environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root), python_path]))

    def record(self, message):
        line = f"[{timestamp()}] {message}"
        print(line, flush=True)
        with (self.output / "pipeline.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def stage(self, name, **values):
        self.state.update(stage=name, updated_at=timestamp(), **values)
        write_json(self.output / "status.json", self.state)

    def snapshot(self):
        destination = self.output / "code"
        shutil.copytree(
            self.root / "lerobot_training", destination / "lerobot_training",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
        entry = absolute(self.root, self.config["model_entry"])
        relative = entry.relative_to(self.root)
        frozen_entry = destination / relative
        if not frozen_entry.exists():
            frozen_entry.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry, frozen_entry)
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
            "git_commit": commit,
            "files": {str(p.relative_to(destination)): digest(p)
                      for p in sorted(destination.rglob("*")) if p.is_file()},
        }
        write_json(self.output / "code_version.json", metadata)
        return frozen_entry

    def launch(self, command, log_path, **details):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with (self.output / "commands.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "started_at": timestamp(), "cwd": str(self.root),
                "argv": command, "log": str(log_path), **details,
            }, ensure_ascii=False) + "\n")
        self.record(shlex.join(command))
        self.record(f"日志: {log_path}")
        log = log_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command, cwd=self.root, env=self.environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        except BaseException:
            log.close()
            raise
        job = {"process": process, "log": log, "path": log_path, **details}
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
            f"checkpoint={checkpoint}"
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
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
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
            "20-target evaluation.\n",
            encoding="utf-8",
        )
        self.record(
            f"评测阶段输出: {len(completed_shards)}/{shards} shards, "
            f"{len(rows)}/{end - start} episodes, "
            f"success={partial['successes']}/{len(rows)}, "
            "partial_summary.json"
        )

    def compare_with_baseline(self, rows, summary):
        reference = self.config.get("baseline_manifest")
        if reference is None:
            return None
        baseline_path = absolute(self.root, reference)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        evaluation = baseline["evaluation"]
        expected_manifest_sha256 = evaluation["target_manifest"]["sha256"]
        manifest_path = absolute(
            self.root, self.config["evaluation"]["target_manifest"]
        )
        actual_manifest_sha256 = digest(manifest_path)
        if actual_manifest_sha256 != expected_manifest_sha256:
            raise RuntimeError(
                "评测目标清单与 baseline 不一致: "
                f"{actual_manifest_sha256} != {expected_manifest_sha256}"
            )
        baseline_records = {
            int(row["episode_index"]): row
            for row in evaluation["records"]
        }
        candidate_records = {
            int(row["episode_index"]): row for row in rows
        }
        if set(candidate_records) != set(baseline_records):
            raise RuntimeError(
                "候选模型与 baseline 的 episode 索引不一致"
            )
        for index in sorted(candidate_records):
            candidate_target = candidate_records[index].get("target_id")
            baseline_target = baseline_records[index].get("target_id")
            if candidate_target != baseline_target:
                raise RuntimeError(
                    f"episode {index} 目标不一致: "
                    f"{candidate_target} != {baseline_target}"
                )
        improvements = [
            float(baseline_records[index]["final_error_um"])
            - float(candidate_records[index]["final_error_um"])
            for index in sorted(candidate_records)
        ]
        baseline_mean = float(evaluation["final_error"]["mean_um"])
        baseline_median = float(evaluation["final_error"]["median_um"])
        baseline_minimum_mean = float(
            evaluation["per_episode_minimum_error"]["mean_um"]
        )
        comparison = {
            "schema_version": 1,
            "status": "comparable",
            "baseline_id": baseline["baseline_id"],
            "baseline_manifest": {
                "path": str(baseline_path),
                "sha256": digest(baseline_path),
            },
            "target_manifest_sha256": actual_manifest_sha256,
            "episodes": len(rows),
            "baseline": {
                "successes": int(evaluation["successes"]),
                "success_rate": float(evaluation["success_rate"]),
                "final_error_mean_um": baseline_mean,
                "final_error_median_um": baseline_median,
                "per_episode_minimum_error_mean_um": (
                    baseline_minimum_mean
                ),
                "safety_limited_steps": int(
                    evaluation["safety_limited_steps"]["total"]
                ),
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
                "successes": int(summary["successes"])
                - int(evaluation["successes"]),
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
            f"# Comparison against `{baseline['baseline_id']}`\n\n"
            f"Target manifest SHA-256: "
            f"`{actual_manifest_sha256}`\n\n"
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
            f"baseline 对照完成: {baseline['baseline_id']}, "
            f"success_delta="
            f"{comparison['delta_candidate_minus_baseline']['successes']:+d}, "
            f"final_error_mean_delta="
            f"{comparison['delta_candidate_minus_baseline']['final_error_mean_um']:+.3f} um"
        )
        return comparison

    def train(self, entry):
        self.stage("training_and_validation")
        options = dict(self.config["training"])
        for key in ("data_dir", "val_data_dir", "init_from"):
            if options.get(key) is not None:
                options[key] = str(absolute(self.root, options[key]))
        options["output_dir"] = str(self.output / "training")
        command = [sys.executable, "-u", str(entry), "train", *flags(options)]
        job = self.launch(command, self.output / "training.log", stage="training_and_validation")
        code = job["process"].wait()
        job["log"].close()
        if code:
            raise RuntimeError(f"训练失败，退出码 {code}；见 {job['path']}")
        checkpoint = (
            self.output
            / "training"
            / self.config.get("checkpoint", "best_model.pth")
        )
        self.record_training_result(checkpoint)
        return checkpoint

    def evaluate(self, entry, checkpoint):
        self.stage("evaluation", completed_shards=0)
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
        options.update(model=str(checkpoint), target_manifest=str(manifest_path))
        write_json(self.output / "evaluation_manifest.json", {
            "source": str(manifest_path), "sha256": digest(manifest_path),
            "target_start": start, "target_end": end, "targets": targets[start:end],
        })
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
                job["log"].close()
                if code:
                    raise RuntimeError(f"评测分片失败，退出码 {code}；见 {job['path']}")
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
        # Do not report a partial/duplicated evaluation as a complete experiment.
        if sorted(row["episode_index"] for row in episodes) != list(range(start, end)):
            raise RuntimeError("评测结果缺少目标或包含重复目标；已保留分片日志和结果")
        return sorted(episodes, key=lambda row: row["episode_index"]), contract

    def summarize(self, episodes, contract, checkpoint, evaluation_only):
        self.stage("recording_results")
        rows = []
        for episode in episodes:
            terminal = episode.get("terminal_success", {})
            trajectory = episode.get("trajectory", {})
            errors = trajectory.get("errors", [])
            rows.append({
                "episode_index": episode["episode_index"], "target_id": episode.get("target_id"),
                "seed": episode.get("seed"), "success": bool(episode["success"]),
                "steps": episode["steps"], "final_error_um": episode["final_error_mm"] * 1000,
                "trajectory_minimum_error_um": min(errors or [episode["final_error_mm"]]) * 1000,
                "low_speed_at_exit": terminal.get("low_speed_at_exit"),
                "encoder_settled_at_exit": terminal.get("encoder_settled_at_exit"),
                "safety_limited_steps": episode.get("safety_limited_steps", 0),
            })
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
            "| 指标 | 结果 |\n|---|---:|\n"
            f"| 成功目标 | {summary['successes']}/{len(rows)} |\n"
            f"| 最终误差均值 | {statistics.mean(finals):.3f} μm |\n"
            f"| 最终误差中位数 | {statistics.median(finals):.3f} μm |\n"
            f"| 最坏最终误差 | {max(finals):.3f} μm |\n"
            f"| 退出时编码器停稳 | {summary['encoder_settled_at_exit']}/{len(rows)} |\n"
        )
        if history:
            report += (f"| 最后一轮训练 loss | {last['train_loss']:.6f} |\n"
                       f"| 最后一轮验证 loss | {last['val_loss']:.6f} |\n")
        report += ("\n最后一轮 loss 不一定对应被选中的 best checkpoint；权重的 epoch 与 loss 见 "
                   "summary.json 的 evaluation_contract。轨迹最小误差沿用动作前记录口径。\n")
        (self.output / "REPORT.md").write_text(report, encoding="utf-8")
        self.record(f"成功 {summary['successes']}/{len(rows)}；最终平均误差 {statistics.mean(finals):.3f} μm")

    def stop_children(self):
        for job in self.jobs:
            process = job["process"]
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    pass
            job["log"].close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/state_attention_pipeline.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--eval-only", type=Path, metavar="CHECKPOINT")
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    config_path = absolute(root, args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_name = config["model_name"] + "_" + datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = absolute(root, args.output_dir or Path("outputs") / run_name)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    pipeline = Pipeline(root, output, config)
    try:
        pipeline.stage("preparing")
        pipeline.record(f"运行目录: {output}")
        entry = pipeline.snapshot()
        checkpoint = absolute(root, args.eval_only) if args.eval_only else pipeline.train(entry)
        pipeline.state["checkpoint"] = str(checkpoint)
        episodes, contract = pipeline.evaluate(entry, checkpoint)
        pipeline.summarize(episodes, contract, checkpoint, args.eval_only is not None)
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


if __name__ == "__main__":
    raise SystemExit(main())
