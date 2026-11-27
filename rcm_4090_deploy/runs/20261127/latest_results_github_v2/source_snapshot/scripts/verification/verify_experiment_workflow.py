"""Check the experiment workflow without training or a full evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from lerobot_training.run_model_pipeline import write_json
from scripts.tools.experiment_workflow import MODULE, ROOT, TARGETS, now, output_path, read_json


FIXTURE = '''
import argparse
import json
import os
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument("mode")
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--fixture-slow", type=float, default=0)
parser.add_argument("--target-manifest", type=Path)
parser.add_argument("--target-start", type=int)
parser.add_argument("--target-end", type=int)
parser.add_argument("--seed", type=int)
options, _ = parser.parse_known_args()
output = options.output_dir
output.mkdir(parents=True, exist_ok=True)
if options.mode == "train":
    print("fixture val_loss=0.250000", flush=True)
    time.sleep(0.3)
    (output / "best_model.pth").write_bytes(b"fixture checkpoint")
    (output / "training_history.json").write_text(json.dumps([
        {"epoch": 0, "train_loss": 0.5, "val_loss": 0.25, "elapsed_s": 0.3}
    ]))
else:
    index = options.target_start
    targets = json.loads(options.target_manifest.read_text())["targets"]
    print(f"fixture closed-loop step 1/2, target={index}, pid={os.getpid()}", flush=True)
    if index:
        time.sleep(options.fixture_slow)
    episodes = [
        {"episode_index": i, "target_id": targets[i]["id"], "seed": options.seed + i,
         "steps": 2, "success": False, "final_error_mm": 0.01}
        for i in range(index, options.target_end)
    ]
    (output / "evaluation_results.json").write_text(json.dumps({
        "episodes": episodes, "evaluation_contract": {"fixture": True}
    }))
'''


def wait_for(check, timeout=25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.05)
    raise AssertionError("fixture did not reach expected state")


class WorkflowChecks:
    def __init__(self, output):
        self.output = output
        self.results = {}
        self.command = [sys.executable, "-B", "-m", MODULE]
        self.env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        self.entry = output / "fixture.py"
        self.entry.write_text(FIXTURE, encoding="utf-8")
        self.config = {
            "model_name": "sop_fixture", "model_entry": str(self.entry),
            "environment": {"MUJOCO_GL": "egl"},
            "training": {"data_dir": "data/model_pipeline_smoke_20261119/train",
                         "val_data_dir": "data/model_pipeline_smoke_20261119/val",
                         "epochs": 1, "device": "cpu"},
            "evaluation": {"target_manifest": TARGETS, "target_split": "development",
                           "target_start": 0, "target_end": 2, "episodes": 2, "shards": 2,
                           "max_steps": 2, "seed": 42, "error_threshold_mm": 0.00035,
                           "control_period_s": 0.05, "device": "cpu"},
        }

    def invoke(self, name, *args, expected=0, env=None):
        result = subprocess.run(self.command + list(map(str, args)), cwd=ROOT, env=env or self.env,
                                capture_output=True, text=True, timeout=45)
        (self.output / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        assert result.returncode == expected, (name, result.returncode, result.stderr)
        return result.stdout

    def card(self, name, *, max_val_loss=None, slow=0, stages=None):
        config = self.output / f"{name}_config.json"
        write_json(config, dict(self.config, evaluation=dict(self.config["evaluation"], fixture_slow=slow)))
        path = self.output / f"{name}_card.json"
        folder = self.output / name
        if stages is None:
            stages = [{"id": "trial", "type": "pipeline", "tier": "smoke", "config": str(config),
                       "timeout_s": 30, "max_val_loss": max_val_loss}]
        write_json(path, {"schema_version": 1, "id": name, "hypothesis": "workflow fixture",
                          "main_change": "no actual model training", "output_dir": str(folder), "stages": stages})
        return path, folder, config

    def verify_planning_and_gates(self):
        card, folder, config = self.card("gated", max_val_loss=0.5)
        content = read_json(card)
        content["stages"][0]["gates"] = [{"file": "summary.json", "key": "episodes", "op": "==", "value": 2}]
        content["stages"] += [
            {"id": "branch", "type": "pipeline", "tier": "smoke", "timeout_s": 30, "config": str(config),
             "checkpoint_from": "trial", "when": {"stage": "trial", "file": "summary.json",
                                                    "key": "successes", "op": "==", "value": 1}},
            {"id": "followup", "type": "pipeline", "tier": "smoke", "timeout_s": 30, "config": str(config),
             "checkpoint_from": "trial", "requires": ["trial"],
             "when": {"stage": "trial", "file": "summary.json", "key": "episodes", "op": "==", "value": 2}},
        ]
        write_json(card, content)
        planned = json.loads(self.invoke("planning", "preflight", "--card", card))
        assert planned["status"] == "preflight_passed" and not folder.exists()
        self.results["plan_only"] = "no run directory or child stage after read-only preflight"
        live = self.invoke("gated_launch", "launch", "--terminal", "current", "--card", card)
        state = read_json(folder / "workflow_status.json")
        assert state["status"] == "completed" and state["stages"]["branch"]["status"] == "skipped"
        assert state["stages"]["trial"]["status"] == state["stages"]["followup"]["status"] == "completed"
        assert not (folder / "branch").exists()
        assert "fixture val_loss=0.250000" in live and "closed-loop step 1/2" in live and "完成 2/2" in live
        assert "val_loss=0.25" in (folder / "trial/training.log").read_text()
        assert "closed-loop step 1/2" in (folder / "trial/evaluation/g01/eval.log").read_text()
        self.results["execute_plan"] = "gates pass, selected dependency runs, nonselected branch skipped; live/disk logs match"
        return folder / "followup", config

    def verify_denied_gate(self):
        card, folder, config = self.card("denied", max_val_loss=0.1)
        content = read_json(card)
        content["stages"].append({"id": "dependent", "type": "pipeline", "tier": "smoke", "config": str(config),
                                  "checkpoint_from": "trial", "timeout_s": 30})
        write_json(card, content)
        self.invoke("denied_launch", "launch", "--terminal", "current", "--card", card, expected=1)
        state = read_json(folder / "workflow_status.json")
        assert state["status"] == "failed" and state["stages"]["dependent"]["status"] == "skipped"
        assert (folder / "trial/training/result.json").is_file()
        assert not (folder / "trial/evaluation").exists() and not (folder / "dependent").exists()
        self.results["failed_gate"] = "training kept; no evaluation, dependent stage, or retry"

    def verify_terminal(self):
        directory = self.output / "fake_terminal"
        directory.mkdir()
        terminal = directory / "gnome-terminal"
        terminal.write_text(
            "#!/usr/bin/python3\nimport os, subprocess, sys\nfrom pathlib import Path\n"
            "assert '--window' in sys.argv and '--wait' not in sys.argv\n"
            "if os.environ.get('REFUSE_TERMINAL'): raise SystemExit(7)\n"
            "command = sys.argv[sys.argv.index('--')+1:]\n"
            "with Path(os.environ['TERMINAL_LOG']).open('x') as log:\n"
            "    subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)\n",
            encoding="utf-8",
        )
        terminal.chmod(0o755)
        env = dict(self.env, PATH=str(directory) + os.pathsep + self.env["PATH"],
                   TERMINAL_LOG=str(self.output / "fake_terminal_console.log"))
        card, folder, _ = self.card("terminal_failure")
        self.invoke("terminal_failure_launch", "launch", "--card", card, expected=1,
                    env=dict(env, REFUSE_TERMINAL="1"))
        assert read_json(folder / "workflow_status.json")["status"] == "failed"
        assert not (folder / "worker.claim").exists() and not (folder / "trial").exists()
        self.results["terminal_failure"] = "exit 7 logged; no worker and no background fallback"
        card, folder, _ = self.card("terminal_once", slow=1.5)
        output = self.invoke("terminal_once_launch", "launch", "--card", card, env=env)
        assert "GNOME Terminal accepted launch" in output
        path = folder / "workflow_status.json"
        assert read_json(path)["status"] in {"launching", "running"}
        wait_for(lambda: read_json(path)["status"] in {"completed", "failed", "interrupted"})
        assert read_json(path)["status"] == "completed"
        assert (folder / "worker.claim").is_file()
        self.results["single_terminal_launch"] = "one window command, returns during run, then train/validate/eval complete"

    def verify_interrupt(self):
        card, folder, _ = self.card("interrupted", slow=20)
        partial = folder / "trial/evaluation/partial_summary.json"
        with (self.output / "interrupt_console.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(self.command + ["launch", "--terminal", "current", "--card", str(card)],
                                       cwd=ROOT, env=self.env, stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_for(partial.is_file)
                before = json.loads(self.invoke("partial_status", "status", "--output-dir", folder))
                item = before["stages"]["trial"]
                assert item["partial_evaluation"]["completed_shards"] == 1 and "final_summary" not in item
                process.send_signal(signal.SIGTERM)
                assert process.wait(timeout=15) == 130
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    process.wait(timeout=15)
        after = json.loads(self.invoke("interrupted_status", "status", "--output-dir", folder))
        assert after["status"] == "interrupted" and "final_summary" not in after["stages"]["trial"]
        assert after["stages"]["trial"]["partial_evaluation"]["status"] == "partial"
        assert not (folder / "trial/summary.json").exists()
        self.results["interruption_and_snapshot"] = "SIGTERM; one of two shards retained; no final summary"

    def verify_reuse(self, source, config):
        checkpoint = source.parent / "trial/training/best_model.pth"
        spec = {"id": "trial", "type": "pipeline", "tier": "smoke", "config": str(config),
                "timeout_s": 30, "checkpoint": str(checkpoint), "reuse_from": str(source)}
        card, folder, _ = self.card("reuse", stages=[spec])
        self.invoke("reuse_launch", "launch", "--terminal", "current", "--card", card)
        assert read_json(folder / "workflow_status.json")["stages"]["trial"]["status"] == "reused"
        assert not (folder / "trial/evaluation").exists()
        fail_gate = {"file": "summary.json", "key": "successes", "op": ">", "value": 0}
        card, folder, _ = self.card("reuse_failed_gate", stages=[dict(spec, gates=[fail_gate])])
        self.invoke("reuse_gate_launch", "launch", "--terminal", "current", "--card", card, expected=1)
        state = json.loads(self.invoke("reuse_failed_gate_status", "status", "--output-dir", folder))
        item = state["stages"]["trial"]
        assert item["status"] == "failed" and "final_summary" in item and "partial_evaluation" not in item
        changed = read_json(config)
        changed["evaluation"]["seed"] += 1
        bad_config = self.output / "reuse_bad_seed.json"
        write_json(bad_config, changed)
        card, folder, _ = self.card("reuse_mismatch", stages=[dict(spec, config=str(bad_config))])
        self.invoke("reuse_refusal", "preflight", "--card", card, expected=1)
        assert not folder.exists()
        self.results["provenance_reuse"] = "complete matching receipt reused; different seed rejected"
        self.results["completed_evaluation_failed_gate"] = "failed metric gate does not relabel a complete evaluation as partial"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    output = output_path(parser.parse_args().output_dir)
    output.mkdir(parents=True, exist_ok=False)
    checks = WorkflowChecks(output)
    verification = {"status": "failed", "checks": checks.results}
    try:
        source, config = checks.verify_planning_and_gates()
        checks.verify_denied_gate()
        checks.verify_terminal()
        checks.verify_interrupt()
        checks.verify_reuse(source, config)
        verification["status"] = "PASS"
    except BaseException as error:
        verification["error"] = repr(error)
        raise
    finally:
        write_json(output / "verification.json", verification)
        with (ROOT / "TEST_LOG.md").open("a", encoding="utf-8") as log:
            log.write(f"\n- {now()}：实验 SOP 轻量验收 **{verification['status']}**。"
                      f"`python -B -m {__spec__.name} --output-dir {output.relative_to(ROOT)}`；"
                      f"{len(checks.results)} 项通过，证据 `{output.relative_to(ROOT) / 'verification.json'}`。"
                      f"{verification.get('error', '')} 使用 CPU 文件/进程夹具，未训练或闭环推理。\n")
    print(f"PASS: {output / 'verification.json'}")


if __name__ == "__main__":
    main()
