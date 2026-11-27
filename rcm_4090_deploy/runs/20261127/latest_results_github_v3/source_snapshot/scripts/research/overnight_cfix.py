"""Run the local C-label and OCT residual experiment without a chat session."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from lerobot_training.run_model_pipeline import start_logged, stop_logged

ROOT = Path(__file__).resolve().parents[2]
MODULE = "scripts.research.overnight_cfix"
DATA_MODULE = MODULE + "_data"
EVAL_MODULE = MODULE + "_eval"
DATA = Path("data/plan2_plus_original_train_224_20261119")
VAL = Path("data/lerobot_rcm_val_acthist_20261117_224")
R0_ROOT = Path("outputs/state_attention_experience_ratio_20261118_1/r0_natural")
R0 = R0_ROOT / "training/step_004000.pth"
V1 = Path("outputs/dit_rectified_flow_state_attention_v1_acthist_20261117/best_model.pth")
TARGETS = Path("configs/v12_reachable_development_targets_0p35um.json")
C_REFERENCE = Path("outputs/state_attention_v1_c_retraction_20261119_1/results.json")
ARM_HOURS = {"I0": 5, "I1": 5, "I3": 7}
CONTINUATION_HOURS = 30
ASSETS = {
    DATA / "data/dataset.npz": "9d80523f519faf4e00bbaef3d30e29dec1401181f3dcddd438c5087c05eb54db",
    DATA / "metadata.json": "7f02f265acadf965a9dc0de29b2058cfccc240ea39d06dc88050179d0ae2698d",
    R0: "b504146485c38c1c26bd960d8aa831beba0df7e564f1f20267862a79b7dc1733",
    V1: "d1e2527fe087ad1f619450bdd83eac24e63d1e3a0d96186dc2cdd849fdfbc424",
    TARGETS: "fcc4872a9d5b58bb7158b36d3c0ab6d7452e4d8d747317dfb077b6e7bbc815c7",
}


def now():
    return datetime.now(ZoneInfo("America/Phoenix"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def environment(output):
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES="0", MUJOCO_GL="egl", MPLBACKEND="Agg",
               OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
               MPLCONFIGDIR=str((output / "mpl").resolve()), TZ="America/Phoenix")
    return env


def step_zero(checkpoint, dataset, output):
    import torch
    from lerobot_training.dit_experience_replay import _load_policy

    device = torch.device("cuda:0")
    base, _, _ = _load_policy(checkpoint, device, oct_symlog_enabled=False)
    residual, _, _ = _load_policy(checkpoint, device, oct_symlog_enabled=True)
    base.eval()
    residual.eval()
    samples = [dataset[i] for i in (0, len(dataset) // 2, len(dataset) - 1)]
    keys = ["image", "state", "oct", "velocity", "previous_action"]
    batch = [torch.stack([row[f"observation.{key}"] for row in samples]).to(device) for key in keys]
    batch[2][0, 0] = 0  # Mixed valid/invalid history.
    batch[2][1] = 0     # Fully invalid history.
    results = {}
    with torch.no_grad():
        for amp in (False, True):
            with torch.autocast("cuda", enabled=amp, dtype=torch.float16):
                outputs = [model.sample(*batch, generator=torch.Generator(device=device).manual_seed(42))
                           for model in (base, residual)]
            if any(not bool(torch.isfinite(value).all()) for value in outputs):
                raise FloatingPointError("nonfinite step-zero output")
            results["amp_fp16" if amp else "fp32"] = float((outputs[0] - outputs[1]).abs().max())
    maximum = max(results.values())
    write(output, {"checkpoint": str(checkpoint), "max_abs_diff": maximum,
                   "by_mode": results, "limit": 1e-6, "status": "PASS" if maximum < 1e-6 else "FAILED"})
    if maximum >= 1e-6:
        raise ValueError(f"step-zero difference {maximum} >= 1e-6")


def dataset_for_check():
    from lerobot_training.train_diffusion_policy import RCMDataset, image_transform_for_dataset

    return RCMDataset(DATA, transform=image_transform_for_dataset(DATA), n_obs_steps=2,
                      n_action_steps=16, include_velocity=True, include_previous_action=True)


def preflight(output):
    import numpy as np
    import torch
    from lerobot_training.dit_experience_replay import RatioReplacementBatchSampler, split_experience_indices
    from scripts.research.overnight_cfix_data import apply_overlay, windows
    from scripts.research.overnight_cfix_eval import targets, validate_contract, validate_episode

    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("expected CUDA on the RTX 4090")
    hashes = {str(path): digest(path) for path in ASSETS}
    for path, expected in ASSETS.items():
        if hashes[str(path)] != expected:
            raise ValueError(f"frozen asset SHA-256 mismatch: {path}")
    for dataset_dir in (DATA, VAL):
        images = read(dataset_dir / "data/image_paths.json")
        if len(images) != read(dataset_dir / "metadata.json")["num_frames"]:
            raise ValueError(f"image/metadata length mismatch: {dataset_dir}")
        for image in images:
            if not (dataset_dir / image.replace("\\", "/")).is_file():
                raise FileNotFoundError(f"{dataset_dir}: {image}")
    original = dataset_for_check()
    clean, correction = split_experience_indices(original)
    if len(clean) + len(correction) != 301634 or not len(clean) or not len(correction):
        raise ValueError("unexpected replay window split")
    quota = []
    for epoch in range(4):
        sampler = RatioReplacementBatchSampler(len(clean), len(correction), 0.08, 1000, 16, 20261118 + epoch)
        quota.append(sum(index >= len(clean) for batch in sampler for index in batch))
    if quota != [1280] * 4:
        raise ValueError(f"8% sampler quota mismatch: {quota}")
    before = original.actions.copy()
    executed = original.executed_actions.copy()
    row = int(np.flatnonzero(before[:, 2] < 0)[0])
    overlay = output / "smoke_overlay.npz"
    np.savez(overlay, rows=np.asarray([row], dtype=np.int64), old_dC_mm=before[[row], 2],
             new_dC_mm=np.zeros(1, dtype=np.float32), dataset_sha256=hashes[str(DATA / "data/dataset.npz")])
    after = apply_overlay(DATA, original.actions, overlay)
    np.testing.assert_array_equal(original.actions, before)
    np.testing.assert_array_equal(original.executed_actions, executed)
    np.testing.assert_array_equal(after[:, :2], before[:, :2])
    np.testing.assert_array_equal(np.flatnonzero(after[:, 2] != before[:, 2]), [row])
    with np.load(DATA / "data/dataset.npz") as arrays:
        a, diagnostics = windows(arrays)
        b, _ = windows({**arrays, "action": after})
    np.testing.assert_array_equal(a, b)
    frozen = read(R0_ROOT / "evaluation/evaluation_results.json")
    validate_contract(frozen["evaluation_contract"], "I0")
    expected_targets = targets(TARGETS)
    for index, (episode, target) in enumerate(zip(frozen["episodes"], expected_targets, strict=True)):
        validate_episode(episode, target, index, require_latency=False)
    c_reference = {}
    comparison = read(C_REFERENCE)["comparison"]
    for label in ("r0", "plan2"):
        c_reference[label] = {}
        for index in (8, 0):
            row = comparison[label][str(index)]
            if row["seed"] != 42 + index or row["target_id"] != expected_targets[index]["id"]:
                raise ValueError(f"{label}: C reference target or seed mismatch")
            c_reference[label][f"E{index:02d}"] = row["C_residual_at_min_um"] - row["C_residual_80_steps_after_min_um"]
    step_zero(V1, original, output / "step_zero_v1.json")
    step_zero(R0, original, output / "step_zero_r0.json")
    write(output / "preflight.json", {
        "status": "PASS", "gpu": torch.cuda.get_device_name(0), "assets": hashes,
        "original_windows": len(clean), "correction_and_plan2_windows": len(correction),
        "window_diagnostics": diagnostics, "overlay_arrays_isolated": True,
        "correction_samples_per_epoch": quota, "total_samples_per_epoch": 16000,
        "r0_final_mean_um": sum(e["final_error_mm"] for e in frozen["episodes"]) * 50,
        "c_reference_path": str(C_REFERENCE), "c_reference_retreat_um": c_reference,
    })


def continuation_inputs(receipt_path):
    receipt = read(receipt_path)
    hashes = receipt["sha256"]
    checkpoint, overlay = (Path(receipt[key]) for key in ("checkpoint", "label_overlay"))
    required = {checkpoint, overlay, *ASSETS}
    for directory in (DATA, VAL):
        required.update(directory / name for name in ("metadata.json", "data/dataset.npz", "data/image_paths.json"))
    if (receipt["status"] != "passed" or not required.issubset(map(Path, hashes))
            or any(hashes[str(path)] != expected for path, expected in ASSETS.items())):
        raise ValueError("invalid or incomplete Task 4 asset preflight receipt")
    for path, expected in hashes.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"asset changed after preflight: {path}")
    return checkpoint, overlay


def continuation_identity():
    from scripts.tools.experiment_workflow import scene_and_code

    return {**scene_and_code(), "target_manifest_sha256": digest(TARGETS),
            "runner_sha256": digest(Path(__file__)),
            "evaluator_sha256": digest(Path(__file__).with_name("overnight_cfix_eval.py")),
            "overlay_code_sha256": digest(Path(__file__).with_name("overnight_cfix_data.py"))}


class Interrupted(Exception):
    pass


class Runner:
    def __init__(self, output, *, continuation=False):
        self.output = output
        self.continuation = continuation
        self.tag = output.parent.name if continuation else output.name
        self.checkpoints = Path("outputs") / self.tag
        self.overlay_dir = Path("data") / f"{self.tag}_overlay"
        self.reports = output / "reports" if continuation else Path("github_reports") / self.tag
        self.reports.mkdir(parents=True, exist_ok=False)
        self.started = now().isoformat()
        self.hours = CONTINUATION_HOURS if continuation else 7
        self.deadline = time.monotonic() + self.hours * 3600
        self.results = {}
        self.active = None
        self.status("running")

    def status(self, status):
        write(self.output / "status.json", {"status": status, "active_stage": self.active,
              "pid": os.getpid(), "started_at": self.started, "updated_at": now().isoformat(),
              "report_dir": str(self.reports), "stages": self.results})

    def record(self, name, status, detail):
        detail = dict(detail)
        if "status" in detail:
            detail["result_status"] = detail.pop("status")
        index = self.output / "commands.jsonl"
        commands = [json.loads(line) for line in index.read_text().splitlines()] if index.is_file() else []
        detail["commands"] = [command for command in commands if command["stage"] == name]
        self.results[name] = {"status": status, **detail}
        report = self.reports / name
        report.mkdir(exist_ok=True)
        (report / "STATUS.md").write_text(f"# {name}: {status}\n\n```json\n{json.dumps(detail, indent=2)}\n```\n")
        with Path("TEST_LOG.md").open("a", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(f"\n- {now().isoformat()}: `{self.tag}/{name}` **{status}**. "
                         f"{json.dumps(detail, ensure_ascii=False)}. MuJoCo simulation only.\n")
        self.active = None
        self.status("running")
        print(f"[{name}] {status}: {report / 'STATUS.md'}", flush=True)

    @contextmanager
    def stage(self, name, **details):
        self.active = name
        self.status("running")
        try:
            yield details
        except BaseException as error:
            state = "interrupted" if isinstance(error, (Interrupted, KeyboardInterrupt, subprocess.TimeoutExpired)) else "failed"
            if "output" in details:
                for filename in ("progress.json", "training_history.json"):
                    artifact = Path(details["output"]) / filename
                    if artifact.is_file():
                        details["partial_evidence"] = str(artifact)
                        if filename == "progress.json":
                            details["partial_progress"] = read(artifact)
                        break
            self.record(name, state, {**details, "reason": str(error)})
            raise
        else:
            self.record(name, "completed", details)

    def cmd(self, stage, module, *args, timeout_s=None):
        command = [sys.executable, "-B", "-u", "-m", module, *map(str, args)]
        log = self.output / f"{stage}.log"
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise Interrupted(f"{self.hours}-hour experiment deadline reached")
        budget = remaining if timeout_s is None else min(timeout_s, remaining)
        with (self.output / "commands.jsonl").open("a") as stream:
            stream.write(json.dumps({"stage": stage, "time": now().isoformat(),
                                     "argv": command, "timeout_s": budget, "log": str(log)}) + "\n")
        print(f"[{stage}] started; timeout {budget:.1f} s; log: {log}", flush=True)
        job = start_logged(command, log, cwd=ROOT, env=environment(self.output), label=stage)
        try:
            code = job["process"].wait(timeout=budget)
        finally:
            stop_logged([job], grace_seconds=5)
        if code in (130, -signal.SIGINT, -signal.SIGTERM, -signal.SIGHUP):
            raise Interrupted(f"{stage} interrupted (exit {code}); see {log}")
        return code, log

    @staticmethod
    def require_ok(result):
        code, log = result
        if code:
            raise RuntimeError(f"child exited {code}; see {log}")

    def evaluate(self, name, checkpoint, arm):
        out = self.output / name
        with self.stage(name, output=str(out), checkpoint=str(checkpoint), arm=arm,
                        protocol="development E00-E19, seed 42+index, 1600 steps, 0.35 um") as details:
            if self.continuation:
                current = continuation_identity()
                if current != self.source_identity:
                    raise ValueError("source or scene changed after continuation preflight")
                identity = {**current, "checkpoint_sha256": digest(checkpoint), "target_manifest": str(TARGETS),
                            "renderer": "egl", "cuda_visible_devices": "0"}
                details["identity"] = identity
            self.require_ok(self.cmd(name, EVAL_MODULE, "--checkpoint", checkpoint, "--target-manifest",
                                     TARGETS, "--output-dir", out, "--arm", arm,
                                     timeout_s=ARM_HOURS[arm] * 3600 if self.continuation else None))
            result = read(out / "summary.json")
            if (result["status"] != "completed" or result["target_count"] != 20
                    or result["arm"] != arm or Path(result["checkpoint"]) != checkpoint):
                raise ValueError("incomplete evaluation or mismatched arm/checkpoint")
            if self.continuation:
                if continuation_identity() != current or digest(checkpoint) != identity["checkpoint_sha256"]:
                    raise ValueError("evaluation inputs changed while running")
                write(out / "provenance.json", {**identity, "summary_sha256": digest(out / "summary.json"),
                                                "arm": arm, "base_seed": 42, "target_count": 20,
                                                "max_steps": 1600, "error_threshold_um": 0.35})
            details.update(result)
            for filename in ("summary.json", "episodes.csv", "error_curves_all.png", "best_and_final_error.png", "c_min_aligned.png"):
                dest = self.reports / name
                dest.mkdir(exist_ok=True)
                shutil.copy2(out / filename, dest / filename)
        return result

    def train(self, name, checkpoint, overlay, symlog=False):
        out = self.checkpoints / name
        with self.stage(name, checkpoint=str(checkpoint), dataset=str(DATA), validation=str(VAL),
                        overlay=str(overlay) if overlay else None, output=str(out), seed=20261118) as details:
            extra = ["--label-overlay", overlay] if overlay else []
            if symlog:
                extra.append("--oct-symlog")
            self.require_ok(self.cmd(name, "lerobot_training.dit_experience_replay", "train",
                                     "--data-dir", DATA, "--val-data-dir", VAL, "--init-from", checkpoint,
                                     "--output-dir", out, "--mode", "ratio", "--correction-ratio", 0.08,
                                     "--epochs", 4, "--steps-per-epoch", 1000, "--checkpoint-every-steps", 1000,
                                     *extra, timeout_s=2700 if self.continuation else None))
            history = read(out / "training_history.json")
            expected = {"original": 58880, "correction": 5120}
            if (len(history) != 4 or [row["batches_seen"] for row in history] != [1000, 2000, 3000, 4000]
                or history[-1]["sampling_counts"] != expected or any(
                not (out / f"step_{step:06d}.pth").is_file() for step in range(1000, 4001, 1000)
            )):
                raise ValueError("training did not meet the 4,000-step / 8% replay contract")
            details.update(sampling_counts=expected, train_loss=history[-1]["train_loss"],
                           val_loss=history[-1]["val_loss"], batches_seen=4000,
                           optimizer_updates=history[-1]["optimizer_updates"],
                           amp_skipped_batches=history[-1]["amp_skipped_batches"],
                           replay_labels="C-only overlay" if overlay else "uncleaned")
        return out

    def run(self):
        with self.stage("preflight", output=str(self.output / "preflight.json"), dataset=str(DATA),
                        checkpoints=[str(V1), str(R0)]) as details:
            self.require_ok(self.cmd("preflight", MODULE, "preflight", "--output-dir", self.output))
            details.update(read(self.output / "preflight.json"))
        with self.stage("eval_smoke", checkpoint=str(R0), output=str(self.output / "eval_smoke")) as details:
            self.require_ok(self.cmd("eval_smoke", EVAL_MODULE, "--checkpoint", R0, "--target-manifest", TARGETS,
                                     "--output-dir", self.output / "eval_smoke", "--arm", "I3", "--smoke"))
            details.update(read(self.output / "eval_smoke/smoke.json"))
        audit = self.output / "task1"
        with self.stage("task1", dataset=str(DATA), output=str(audit)) as details:
            child = self.cmd("task1", DATA_MODULE, "audit", "--dataset-dir", DATA, "--output-dir", audit)
            result = None
            if (audit / "contamination_audit.json").is_file():
                result = read(audit / "contamination_audit.json")
                for key in ("candidate_count", "replayed_count", "reproduced_count", "dragged_negative_fraction",
                            "threshold_reached", "failure_count"):
                    details[key] = result[key]
                dest = self.reports / "task1"
                dest.mkdir(exist_ok=True)
                for filename in ("contamination_audit.md", "contamination_audit.json", "anomalies.json"):
                    if (audit / filename).is_file():
                        shutil.copy2(audit / filename, dest / filename)
            self.require_ok(child)
            if result is None or result["status"] != "PASS" or result["threshold_reached"] is None:
                raise ValueError("original labels were not reproduced; no 60% decision")
        checkpoint, overlay = R0, None
        if result["threshold_reached"]:
            overlay = self.overlay_dir / "label_override.npz"
            with self.stage("task2_overlay", output=str(overlay), dataset=str(DATA)) as details:
                self.require_ok(self.cmd("task2_overlay", DATA_MODULE, "prepare-clean", "--dataset-dir", DATA,
                                         "--output-dir", self.overlay_dir, "--audit-csv", audit / "contamination_audit.csv",
                                         "--audit-json", audit / "contamination_audit.json"))
                details.update(read(self.overlay_dir / "label_override.json"))
            trained = self.train("task2_train", V1, overlay)
            scores = [self.evaluate(f"task2_step{step}", trained / f"step_{step:06d}.pth", "I0") for step in (2000, 4000)]
            best = min(scores, key=lambda entry: entry["final_mean_um"])
            r0_mean = read(self.output / "preflight.json")["r0_final_mean_um"]
            if best["final_mean_um"] < r0_mean:
                checkpoint = Path(best["checkpoint"])
            self.record("task2_selection", "completed", {"checkpoint": str(checkpoint), "r0_final_mean_um": r0_mean,
                        "best_replay_final_mean_um": best["final_mean_um"], "dataset": str(DATA),
                        "protocol": "paired development E00-E19 at 42+index, 1600 steps, I0"})
        else:
            self.record("task2", "skipped", {"reason": "reproduced drag fraction below 60%", "checkpoint": str(R0)})
        self.run_tasks_3_and_4(checkpoint, overlay)
        if self.deadline - time.monotonic() >= 20 * 60:
            with self.stage("task5", dataset=str(DATA), output=str(self.output / "task5")) as details:
                self.require_ok(self.cmd("task5", DATA_MODULE, "window-audit", "--dataset-dir", DATA,
                                         "--output-dir", self.output / "task5"))
                dest = self.reports / "task5"
                dest.mkdir(exist_ok=True)
                for artifact in (self.output / "task5").iterdir():
                    shutil.copy2(artifact, dest / artifact.name)
                details.update(read(self.output / "task5/window_retention.json"))
        else:
            self.record("task5", "skipped", {"reason": "less than 20 minutes remaining"})

    def run_continuation(self, receipt_path):
        with self.stage("continuation_preflight", asset_receipt=str(receipt_path)) as details:
            checkpoint, overlay = continuation_inputs(receipt_path)
            if self.checkpoints.exists():
                raise FileExistsError(self.checkpoints)
            self.source_identity = continuation_identity()
            details.update(checkpoint=str(checkpoint), overlay=str(overlay), total_hours=self.hours,
                           per_arm_hours=ARM_HOURS, source_identity=self.source_identity)
        self.run_tasks_3_and_4(checkpoint, overlay, receipt_path=receipt_path)

    def run_tasks_3_and_4(self, checkpoint, overlay, *, receipt_path=None):
        arms = [self.evaluate(f"task3_{arm}", checkpoint, arm) for arm in ("I0", "I1", "I3")]
        best_arm = min(arms, key=lambda entry: (entry["final_mean_um"], entry["inference_median_ms"]))["arm"]
        self.record("task3_selection", "completed", {"checkpoint": str(checkpoint), "arm": best_arm,
                    "rule": "minimum final mean error, tie-break by median inference latency",
                    "summaries": {row["arm"]: str(self.output / f"task3_{row['arm']}" / "summary.json") for row in arms}})
        gate = self.output / "step_zero_selected.json"
        with self.stage("task4_step_zero", checkpoint=str(checkpoint), dataset=str(DATA), output=str(gate)) as details:
            if receipt_path is not None:
                if continuation_inputs(receipt_path) != (checkpoint, overlay) or continuation_identity() != self.source_identity:
                    raise ValueError("continuation assets, code or scene changed before step-zero")
            child = self.cmd("task4_step_zero", MODULE, "step-zero", "--checkpoint", checkpoint,
                             "--output-dir", self.output, timeout_s=600 if self.continuation else None)
            if gate.is_file():
                details.update(read(gate))
                shutil.copy2(gate, self.reports / gate.name)
            self.require_ok(child)
            if details.get("status") != "PASS" or not 0 <= details.get("max_abs_diff", float("inf")) < 1e-6:
                raise ValueError("step-zero gate did not produce a passing result")
        trained = self.train("task4_train", checkpoint, overlay, symlog=True)
        for arm in dict.fromkeys(("I0", best_arm)):
            self.evaluate(f"task4_{arm}", trained / "step_004000.pth", arm)

    def finish(self, status):
        lines = ["# Overnight C-label / OCT residual experiment", "", f"Status: **{status}**",
                 f"Evidence: `{self.output}`", "Source: local deployment tree; no Git metadata or push.",
                 "MuJoCo simulation only. Unrun or partial evaluations are not results.", ""]
        for stage, row in self.results.items():
            lines.append(f"- [{stage}]({stage}/STATUS.md): {row['status']}")
            if "dragged_negative_fraction" in row:
                fraction = row["dragged_negative_fraction"]
                lines[-1] += f"; reproduced {row['reproduced_count']}/{row['candidate_count']} original labels"
                lines[-1] += f"; drag fraction {fraction:.2%}" if fraction is not None else "; drag fraction undetermined"
            if "max_abs_diff" in row:
                lines[-1] += f"; step-0 maximum difference {row['max_abs_diff']:.3g} (must be <1e-6)"
            if "replay_labels" in row:
                lines[-1] += f"; labels: {row['replay_labels']}; {row['optimizer_updates']}/4000 optimizer updates (AMP)"
            if stage == "task2_selection":
                lines[-1] += (f"; selected `{row['checkpoint']}` (replay best {row['best_replay_final_mean_um']:.3f} um; "
                              f"R0 {row['r0_final_mean_um']:.3f} um)")
            if stage == "task3_selection":
                lines[-1] += f"; selected {row['arm']}"
            if "reason" in row:
                lines[-1] += f"; {row['reason']}"
        evaluations = {stage: row for stage, row in self.results.items()
                       if row["status"] == "completed" and row.get("target_count") == 20}
        if evaluations:
            lines += ["", "## Complete closed-loop evaluations (E00-E19, seeds 42-61)", "",
                      "| Stage | Final mean / median / min (um) | Mean trajectory minimum (um) | A/B frame jitter (deg) | Settled at exit | Strict success | GPU median / P90 (ms) |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for stage, row in evaluations.items():
                ab = row["near_target_ab_mean_change_deg"]
                jitter = "/".join(f"{axis:.5f}" for axis in ab) if ab is not None else "n/a"
                lines.append(f"| {stage} | {row['final_mean_um']:.3f} / {row['final_median_um']:.3f} / {row['final_min_um']:.3f} | "
                             f"{row['minimum_mean_um']:.3f} | {jitter} | {row['encoder_settled_at_exit_count']}/20 | "
                             f"{row['canonical_success_count']}/20 | {row['inference_median_ms']:.2f} / {row['inference_p90_ms']:.2f} |")
            for arm in ARM_HOURS:
                baseline, trial = evaluations.get(f"task3_{arm}"), evaluations.get(f"task4_{arm}")
                if baseline and trial:
                    final_delta = trial["final_mean_um"] - baseline["final_mean_um"]
                    minimum_delta = trial["minimum_mean_um"] - baseline["minimum_mean_um"]
                    single_digits = "yes" if trial["minimum_mean_um"] < 10 else "no"
                    lines += ["", f"- {arm} (Task 4 minus Task 3): final mean {final_delta:+.3f} um; "
                               f"mean trajectory minimum {minimum_delta:+.3f} um; single-digit minimum mean: {single_digits}."]
            lines += ["", "## C retreat 80 steps after each trajectory minimum (um)", "",
                      "Positive means retraction. `n/a` means no comparable 80-step observation.", "",
                      "| Stage | E08 | E00 | E02 | E06 |", "|---|---:|---:|---:|---:|"]
            for name, values in self.results.get("preflight", {}).get("c_reference_retreat_um", {}).items():
                lines.append(f"| Frozen reference: {name} | {values['E08']:.2f} | {values['E00']:.2f} | n/a | n/a |")
            for stage, row in evaluations.items():
                values = [row["c_retraction"][ep]["C_retreat_after_80_steps_um"] for ep in ("E08", "E00", "E02", "E06")]
                lines.append(f"| {stage} | " + " | ".join(f"{v:.2f}" if v is not None else "n/a" for v in values) + " |")
        temporary = self.reports / "SUMMARY.md.tmp"
        temporary.write_text("\n".join(lines) + "\n")
        temporary.replace(self.reports / "SUMMARY.md")
        self.status(status)
        print(f"Experiment {status}: {self.reports / 'SUMMARY.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("launch", "run", "preflight", "step-zero", "run34"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--asset-receipt", type=Path)
    args = parser.parse_args()
    if Path.cwd() != ROOT:
        parser.error("run from rcm_4090_deploy")
    output = args.output_dir
    if args.command == "launch":
        output.mkdir(parents=True, exist_ok=False)
        with (output / "runner.log").open("x") as stream:
            child = subprocess.Popen([sys.executable, "-m", MODULE, "run", "--output-dir", str(output)],
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=environment(output))
        print(f"PID {child.pid}; status: {output / 'status.json'}")
    elif args.command == "preflight":
        preflight(output)
    elif args.command == "step-zero":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for step-zero")
        step_zero(args.checkpoint, dataset_for_check(), output / "step_zero_selected.json")
    else:
        continuation = args.command == "run34"
        if continuation and args.asset_receipt is None:
            parser.error("--asset-receipt is required for run34")

        def stop(signum, _frame):
            raise Interrupted(f"received {signal.Signals(signum).name}")

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, stop)
        runner = Runner(output, continuation=continuation)
        try:
            if continuation:
                runner.run_continuation(args.asset_receipt)
            else:
                runner.run()
        except (Interrupted, subprocess.TimeoutExpired):
            runner.finish("interrupted")
            if continuation:
                return 130
            raise
        except BaseException:
            runner.finish("failed")
            raise
        runner.finish("completed")


if __name__ == "__main__":
    sys.exit(main())
