"""Preflight, launch, or read one snapshot of a versioned experiment card."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys

from lerobot_training.run_model_pipeline import (
    Pipeline, absolute, digest, start_logged, stop_logged, write_json,
)


ROOT = Path(__file__).resolve().parents[2]
MODULE = "scripts.tools.experiment_workflow"
PIPELINE = "lerobot_training/run_model_pipeline.py"
TARGETS = "configs/v12_reachable_development_targets_0p35um.json"
TIERS = {"screen10": 10, "dev20": 20}
OPS = {"<": operator.lt, "<=": operator.le, "==": operator.eq,
       ">=": operator.ge, ">": operator.gt}
SCENE_CONFIG = ("config.json", "v9_alignment.json", "v9_side_profile.json", "v9_tool_mount.json",
                "mujoco_model/installation_config.json", "mujoco_model/silver_motion_config.json")


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def display(message):
    try:
        print(message, flush=True)
    except OSError:
        pass  # A closed terminal must not prevent state and TEST_LOG writes.


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_path(value):
    path = absolute(ROOT, value)
    require(path.is_file() and os.access(path, os.R_OK), f"missing/unreadable file: {path}")
    return path


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def output_path(value):
    path = absolute(ROOT, value)
    parts = path.relative_to(ROOT).parts
    require((len(parts) >= 2 and parts[0] == "outputs") or
            (len(parts) >= 3 and parts[0] == "runs" and re.fullmatch(r"[0-9]{8}", parts[1])),
            f"use outputs/<experiment> or runs/YYYYMMDD/<purpose>: {path}")
    return path


def relative_result(value):
    path = Path(value)
    require(not path.is_absolute() and path.parts and ".." not in path.parts, f"invalid result path: {value}")
    return path


def protocol(options):
    excluded = {"target_manifest", "target_start", "target_end", "episodes", "shards"}
    return {key: value for key, value in options.items() if key not in excluded}


def baseline_identity(runner, options, selected):
    if runner.baseline is None:
        return None
    path, baseline_id, _, rows = runner.baseline
    baseline = read_json(path)
    checkpoint = baseline["model"]["checkpoint"]
    require(digest(file_path(checkpoint["path"])) == checkpoint["sha256"], "baseline checkpoint hash mismatch")
    evaluation = baseline["evaluation"]
    declared = evaluation["protocol"]
    expected_indices = list(range(declared["target_start"], declared["target_end"]))
    require(evaluation["episodes"] == len(expected_indices) and
            sorted(row["episode_index"] for row in evaluation["records"]) == expected_indices,
            "baseline evaluation is incomplete")
    require(protocol(options) == protocol(declared), "baseline protocol has extra or mismatched options")
    for row in rows:
        require(all(numeric(row[key]) for key in ("final_error_um", "minimum_error_um")),
                "invalid baseline error metric")
    source = evaluation.get("source_summary")
    if source:
        require(digest(file_path(source["path"])) == source["sha256"], "baseline source summary hash mismatch")
    return {"id": baseline_id, "manifest": str(path), "sha256": digest(path),
            "checkpoint": checkpoint, "paired_targets": len(selected),
            "source_summary": source,
            "historical_comparison_only": "evaluation_identity" not in evaluation}


def scene_and_code():
    def fingerprint(files):
        entries = {str(path.relative_to(ROOT)): digest(path) for path in sorted(set(files))}
        return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()

    code = list((ROOT / "lerobot_training").glob("*.py"))
    code += list((ROOT / "rcm").rglob("*.py")) + list((ROOT / "restricted_perception").glob("*.py"))
    code += list((ROOT / "mujoco_model").glob("*.py"))
    code += list((ROOT / "mujoco_model/vendor").rglob("*.py"))
    scene = [file_path(value) for value in SCENE_CONFIG] + [file_path("mujoco_model/assets/scan_template.xml")]
    scene += list((ROOT / "mujoco_model/assets").glob("*.obj"))
    return {"code_sha256": fingerprint(code), "scene_sha256": fingerprint(scene)}


def check_scene(config, checked):
    evaluation = config["evaluation"]
    environment = Pipeline(ROOT, None, config).environment
    device = evaluation.get("device", "cuda")
    training_device = config.get("training", {}).get("device", "cuda")
    cuda = device.startswith("cuda") or (training_device.startswith("cuda") and bool(config.get("training")))
    require(not cuda or (environment.get("CUDA_VISIBLE_DEVICES") == "0" and
                         environment.get("MUJOCO_GL") == "egl"), "CUDA requires GPU 0 and MUJOCO_GL=egl")
    key = (cuda, tuple(sorted(config.get("environment", {}).items())))
    if key in checked:
        return
    probe = "from rcm.scene_v9 import RCMSceneV9; scene=RCMSceneV9(); scene.close()"
    if cuda:
        probe = ("import torch; assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0), "
                 "'RTX 4090 unavailable'; " + probe)
    result = subprocess.run([sys.executable, "-B", "-c", probe], cwd=ROOT, env=environment,
                            capture_output=True, text=True, timeout=30)
    require(result.returncode == 0, f"MuJoCo/CUDA preflight failed: {result.stderr.strip()}")
    checked.add(key)


def pipeline_stage(spec, folder, previous, checked):
    config_path = file_path(spec["config"])
    config = read_json(config_path)
    require(isinstance(config["model_name"], str) and config["model_name"], "missing model name")
    evaluation = config["evaluation"]
    for key in ("output_dir", "model"):
        require(key not in evaluation and key not in config.get("training", {}), f"{key} is managed by the workflow")
    require(not evaluation.get("random_targets"), "workflow requires manifest targets")
    entry = file_path(config["model_entry"])
    require(entry.is_relative_to(ROOT), "model entry must be inside the project")
    eval_only = spec.get("checkpoint")
    from_stage = spec.get("checkpoint_from")
    require(not (eval_only and from_stage), "checkpoint and checkpoint_from are mutually exclusive")
    asset_hashes = {}
    if eval_only:
        asset = file_path(eval_only)
        asset_hashes[str(asset)] = digest(asset)
    elif from_stage:
        require(from_stage in spec["requires"] and previous[from_stage]["trains"],
                "checkpoint_from requires a training dependency")
        eval_only = previous[from_stage]["checkpoint"]
    else:
        training = config["training"]
        training_entry = file_path(config.get("training_entry", config["model_entry"]))
        require(training_entry.is_relative_to(ROOT), "training entry must be inside the project")
        asset_hashes[str(training_entry)] = digest(training_entry)
        require(type(training["epochs"]) is int and training["epochs"] > 0, "set a positive training epoch budget")
        require(absolute(ROOT, training["data_dir"]) != absolute(ROOT, training["val_data_dir"]),
                "training and validation datasets must differ")
        for key in ("data_dir", "val_data_dir"):
            dataset = absolute(ROOT, training[key])
            for name in ("metadata.json", "data/dataset.npz"):
                asset = file_path(dataset / name)
                asset_hashes[str(asset)] = digest(asset)
        if training.get("init_from"):
            asset = file_path(training["init_from"])
            asset_hashes[str(asset)] = digest(asset)
    if evaluation.get("fine_model"):
        fine = file_path(evaluation["fine_model"])
        asset_hashes[str(fine)] = digest(fine)
    require(bool(evaluation.get("fine_model")) == (evaluation.get("fine_switch_error_mm") is not None),
            "fine_model and fine_switch_error_mm must be configured together")
    runner = Pipeline(ROOT, folder, config)
    options, shards, start, end, selected = runner.prepare_evaluation(persist=False)
    require(type(evaluation["max_steps"]) is int and evaluation["max_steps"] > 0, "set a positive max_steps")
    require(type(evaluation["seed"]) is int, "set an explicit integer evaluation seed")
    require(positive(evaluation["error_threshold_mm"]) and positive(evaluation["control_period_s"]),
            "set positive success threshold and control period")
    require(type(evaluation["shards"]) is int and shards == evaluation["shards"], "invalid shard count")
    require(type(evaluation.get("target_start", 0)) is int and
            type(evaluation["episodes"]) is int and end - start == evaluation["episodes"] and
            evaluation.get("target_end", end) == end, "target range would be truncated")
    ids = [row["id"] for row in selected]
    require(all(ids) and len(set(ids)) == len(ids), "target IDs must be unique and nonempty")
    for row in selected:
        for vector in (row["point_mm"], row.get("reachable_target_robot_mm", row["point_mm"])):
            require(len(vector) == 3 and all(numeric(value) for value in vector), "invalid target coordinates")
    manifest = file_path(options["target_manifest"])
    asset_hashes[str(manifest)] = digest(manifest)
    if spec["tier"] in TIERS:
        require(manifest == file_path(TARGETS) and options["target_split"] == "development" and
                start == 0 and end == TIERS[spec["tier"]], "tier requires its fixed development target range")
        require(runner.baseline is not None, "formal tier requires a paired baseline")
    elif spec["tier"] == "smoke":
        require(end - start <= 3, "smoke tier must stay within three targets")
    else:
        raise ValueError(f"unknown evaluation tier: {spec['tier']}")
    baseline = baseline_identity(runner, options, selected)
    if baseline:
        asset_hashes[baseline["manifest"]] = baseline["sha256"]
        for asset in (baseline["checkpoint"], baseline["source_summary"]):
            if asset:
                asset_hashes[str(absolute(ROOT, asset["path"]))] = asset["sha256"]
    checkpoint = config.get("checkpoint", "best_model.pth")
    require(checkpoint not in {"", ".", ".."} and Path(checkpoint).name == checkpoint, "invalid checkpoint name")
    limit = spec.get("max_val_loss")
    require(limit is None or eval_only is None and numeric(limit), "max_val_loss needs a training stage")
    check_scene(config if not eval_only else dict(config, training={}), checked)
    command = [sys.executable, "-B", "-u", PIPELINE, "--config", str(config_path), "--output-dir", str(folder)]
    if eval_only:
        command += ["--eval-only", str(absolute(ROOT, eval_only))]
    if limit is not None:
        command += ["--training-max-val-loss", str(limit)]
    fingerprint = {
        "protocol": protocol(options), "targets": ids, "indices": list(range(start, end)),
        "seeds": [options["seed"] + index for index in range(start, end)],
        "target_sha256": asset_hashes[str(manifest)], "model_entry": config["model_entry"],
        "entry_sha256": digest(entry), "environment": config.get("environment", {}),
        "cuda_visible_devices": runner.environment.get("CUDA_VISIBLE_DEVICES"),
        "mujoco_gl": runner.environment.get("MUJOCO_GL"), "baseline": baseline,
        "fine_checkpoint_sha256": asset_hashes.get(str(absolute(ROOT, evaluation["fine_model"])))
        if evaluation.get("fine_model") else None,
    }
    asset_hashes[str(entry)] = fingerprint["entry_sha256"]
    return {"argv": command, "config": str(config_path), "config_sha256": digest(config_path),
            "assets": asset_hashes, "identity": fingerprint, "trains": not eval_only,
            "checkpoint": str(folder / "training" / checkpoint) if not eval_only else str(absolute(ROOT, eval_only))}


def validate_gate(gate, *, branch=False):
    path = relative_result(gate["file"])
    require("partial" not in path.name and "progress" not in path.parts,
            f"gate must read a completed result: {path}")
    require(isinstance(gate["key"], str) and gate["key"] and gate["op"] in OPS and numeric(gate["value"]),
            "invalid numeric gate")
    if branch:
        require(isinstance(gate["stage"], str), "branch needs a source stage")


def preflight(card_path):
    card = read_json(card_path)
    require(card["schema_version"] == 1, "unsupported card version")
    for key in ("id", "hypothesis", "main_change"):
        require(isinstance(card[key], str) and card[key].strip(), f"missing card field: {key}")
    output = output_path(card["output_dir"])
    require(not output.exists(), f"output already exists: {output}")
    parent = next(path for path in output.parents if path.exists())
    require(parent.is_dir() and os.access(parent, os.W_OK | os.X_OK) and os.access(ROOT / "TEST_LOG.md", os.W_OK),
            f"unwritable output/TEST_LOG: {output}")
    require(isinstance(card["stages"], list) and card["stages"], "no stages")
    planned, checked = {}, set()
    identities = scene_and_code() if any(stage["type"] == "pipeline" for stage in card["stages"]) else {}
    for spec in card["stages"]:
        name = spec["id"]
        require(re.fullmatch(r"[a-z][a-z0-9_-]*", name) and name not in planned, f"invalid/duplicate stage: {name}")
        kind = spec["type"]
        require(kind in ("pipeline", "module"), f"unsupported stage type: {kind}")
        require(positive(spec["timeout_s"]), f"missing positive stage budget: {name}")
        dependencies = spec.setdefault("requires", [next(reversed(planned))] if planned else [])
        require(isinstance(dependencies, list) and all(dep in planned for dep in dependencies),
                f"dependencies must precede {name}")
        for gate in spec.get("gates", []):
            validate_gate(gate)
        if branch := spec.get("when"):
            validate_gate(branch, branch=True)
            require(branch["stage"] in dependencies, f"branch source must be a dependency of {name}")
        folder = output / name
        assets = {str(path): digest(path) for path in map(file_path, spec.get("assets", []))}
        stage = {key: spec[key] for key in ("id", "type", "tier", "timeout_s", "requires", "gates", "when",
                                           "checkpoint_from", "reuse_from") if key in spec}
        stage.update(output_dir=str(folder), assets=assets, trains=False)
        if kind == "pipeline":
            probe = pipeline_stage(spec, folder, planned, checked)
            probe["assets"].update(assets)
            stage.update(probe)
            stage["identity"].update(identities)
            if spec.get("reuse_from"):
                require(spec.get("checkpoint"), "reuse_from requires an existing evaluation-only checkpoint")
                stage["reuse"] = eligible_reuse(stage)
        else:
            module = spec["module"]
            require(re.fullmatch(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+", module), f"invalid module: {module}")
            source = file_path(module.replace(".", "/") + ".py")
            stage["assets"][str(source)] = digest(source)
            args = spec.get("args", [])
            require(isinstance(args, list) and all(isinstance(arg, str) and not arg.startswith("--output-dir")
                                                   for arg in args), "module args must not override --output-dir")
            stage["argv"] = [sys.executable, "-B", "-u", "-m", module, *args, "--output-dir", str(folder)]
        planned[name] = stage
    return {"status": "preflight_passed", "card": str(card_path), "card_sha256": digest(card_path),
            "id": card["id"], "hypothesis": card["hypothesis"], "main_change": card["main_change"],
            "output_dir": str(output), "stages": list(planned.values())}


def full_evaluation(folder, identity):
    status = read_json(folder / "status.json")
    summary = read_json(folder / "summary.json")
    require(status["status"] == summary["pipeline_status"] == "completed", "evaluation was not completed")
    rows = summary["episode_results"]
    require(summary["episodes"] == len(rows) == len(identity["indices"]) and
            [(row["episode_index"], row["target_id"], row["seed"]) for row in rows] ==
            list(zip(identity["indices"], identity["targets"], identity["seeds"])),
            "evaluation has missing, reordered, or mismatched targets/seeds")
    require(bool(summary.get("evaluation_contract")) and summary["checkpoint_sha256"] == identity["checkpoint_sha256"],
            "evaluation checkpoint/contract mismatch")
    require(status["completed_shards"] == status["total_shards"] > 0 and
            all(numeric(row["final_error_um"]) for row in rows), "incomplete or invalid evaluation")
    return summary


def eligible_reuse(stage):
    source = output_path(stage["reuse_from"])
    receipt = read_json(source / "evaluation_receipt.json")
    identity = dict(stage["identity"], checkpoint_sha256=stage["assets"][stage["checkpoint"]])
    require(receipt["identity"] == identity, "reuse identity does not match weight/code/scene/targets/seeds/protocol")
    require(receipt["summary_sha256"] == digest(source / "summary.json"), "reuse summary hash mismatch")
    full_evaluation(source, identity)
    return str(source)


def gate_result(folder, gate):
    path = folder / relative_result(gate["file"])
    payload = read_json(path)
    require(all(payload.get(key) not in ("partial", "running", "failed", "interrupted", "IN_PROGRESS")
                for key in ("status", "pipeline_status", "result")), f"not final: {path}")
    value = payload
    for key in gate["key"].split("."):
        value = value[key]
    require(numeric(value), f"non-numeric gate metric: {gate['key']}")
    return OPS[gate["op"]](value, gate["value"]), f"{gate['key']}={value} {gate['op']} {gate['value']} ({path})"


def append_record(plan, status, *, stage=None, reason=None, result=None):
    name = f"/{stage['id']}" if stage else ""
    lines = [f"\n- {now()}：实验 SOP `{plan['id']}{name}` **{status}**；卡 `{plan['card']}`；"
             f"目录 `{plan['output_dir']}`。"]
    if stage:
        lines.append(f"  命令 `{shlex.join(stage['argv'])}`；配置 `{stage.get('config', '见实验卡')}`。")
        if stage["type"] == "pipeline":
            lines.append(f"  档位 `{stage['tier']}`；权重 `{stage['checkpoint']}`；"
                         f"目标、种子、基线和协议见 `preflight.json`。")
            partial = Path(stage["output_dir"]) / "evaluation" / "partial_summary.json"
            if status == "interrupted" and partial.is_file():
                data = read_json(partial)
                lines.append(f"  部分结果：{data['completed_shards']}/{data['total_shards']} 分片、"
                             f"{data['completed_episodes']} 目标，见 `{partial}`。")
    if result:
        lines.append(f"  结果：{result}。")
    if reason:
        lines.append(f"  原因：{reason}。")
    lines.append("  仅验证软件/仿真，不代表临床或硬件就绪。\n")
    with (ROOT / "TEST_LOG.md").open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def run_stage(stage, stages, output):
    folder = Path(stage["output_dir"])
    for path, checksum in stage["assets"].items():
        require(digest(file_path(path)) == checksum, f"asset changed after preflight: {path}")
    if stage["type"] == "pipeline":
        identity = stage["identity"]
        require(digest(file_path(stage["config"])) == stage["config_sha256"], "configuration changed after preflight")
        require(all(identity[key] == value for key, value in scene_and_code().items()), "code/scene changed after preflight")
        config = read_json(Path(stage["config"]))
        environment = Pipeline(ROOT, folder, config).environment
        require(all(identity[key.lower()] == environment.get(key) for key in ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL")),
                "device/renderer environment changed after preflight")
        if not stage["trains"]:
            identity["checkpoint_sha256"] = digest(file_path(stage["checkpoint"]))
            if source_stage := stage.get("checkpoint_from"):
                require(identity["checkpoint_sha256"] == stages[source_stage]["checkpoint_sha256"],
                        "upstream checkpoint changed")
        if stage.get("reuse"):
            source = Path(eligible_reuse(stage))
            folder.mkdir()
            for name in ("status.json", "summary.json", "evaluation_receipt.json"):
                shutil.copy2(source / name, folder / name)
            write_json(folder / "reuse.json", {"source": str(source)})
            display(f"[{stage['id']}] Reused complete result: {source / 'summary.json'}")
            return
    log = output / f"{stage['id']}.log"
    display(f"[{stage['id']}] {shlex.join(stage['argv'])}\nLog: {log}")
    job = start_logged(stage["argv"], log, cwd=ROOT, env=os.environ.copy(), label=stage["id"])
    try:
        code = job["process"].wait(timeout=stage["timeout_s"])
    finally:
        stop_logged([job], grace_seconds=10)
    if code in (130, -signal.SIGINT, -signal.SIGTERM, -signal.SIGHUP):
        raise KeyboardInterrupt(f"stage interrupted (exit {code}); see {log}")
    if code:
        detail = read_json(folder / "status.json").get("error") if (folder / "status.json").is_file() else None
        raise RuntimeError(detail or f"stage exit {code}; see {log}")
    require(folder.is_dir(), f"stage did not create its result directory: {folder}")
    if stage["type"] == "pipeline":
        checksum = digest(file_path(stage["checkpoint"]))
        if stage["trains"]:
            identity["checkpoint_sha256"] = checksum
            trained = read_json(folder / "training" / "result.json")
            require(trained["checkpoint_sha256"] == checksum, "trained checkpoint changed before evaluation finished")
        require(identity["checkpoint_sha256"] == checksum, "checkpoint changed during evaluation")
        for path, expected in stage["assets"].items():
            require(digest(file_path(path)) == expected, f"asset changed during evaluation: {path}")
        require(all(identity[key] == value for key, value in scene_and_code().items()), "code/scene changed during evaluation")
        frozen_entry = read_json(folder / "code_version.json")["executed_model_entry"]
        require(digest(file_path(frozen_entry)) == identity["entry_sha256"], "executed model entry differs from preflight")
        full_evaluation(folder, identity)
        write_json(folder / "evaluation_receipt.json", {
            "identity": identity, "summary_sha256": digest(folder / "summary.json"),
        })


def execute(plan):
    output = Path(plan["output_dir"])
    states = {stage["id"]: {"status": "pending", "type": stage["type"], "output_dir": stage["output_dir"]}
              for stage in plan["stages"]}
    state = {"id": plan["id"], "status": "running", "pid": os.getpid(), "stages": states}
    path = output / "workflow_status.json"

    def save():
        state["updated_at"] = now()
        write_json(path, state)

    save()
    display(f"Experiment: {plan['id']}\nOutput: {output}")
    try:
        for stage in plan["stages"]:
            name, folder = stage["id"], Path(stage["output_dir"])
            item = states[name]
            result = None
            try:
                blocked = [dep for dep in stage["requires"] if states[dep]["status"] not in {"completed", "reused"}]
                if blocked:
                    item.update(status="skipped", reason=f"dependency not passed: {', '.join(blocked)}")
                    continue
                if branch := stage.get("when"):
                    passed, detail = gate_result(Path(states[branch["stage"]]["output_dir"]), branch)
                    if not passed:
                        item.update(status="skipped", reason=f"branch not selected: {detail}")
                        continue
                item.update(status="running", started_at=now())
                save()
                run_stage(stage, states, output)
                if stage["type"] == "pipeline":
                    summary = read_json(folder / "summary.json")
                    item["checkpoint_sha256"] = summary["checkpoint_sha256"]
                    item["evaluation_complete"] = True
                    result = (f"{folder / 'summary.json'}; success {summary['successes']}/{summary['episodes']}; "
                              f"mean final error {summary['final_error_um']['mean']:.3f} um")
                for gate in stage.get("gates", []):
                    passed, detail = gate_result(folder, gate)
                    item.setdefault("gates", []).append({"passed": passed, "detail": detail})
                    require(passed, f"gate failed: {detail}")
                item.update(status="reused" if stage.get("reuse") else "completed")
            except (KeyboardInterrupt, subprocess.TimeoutExpired) as error:
                item.update(status="interrupted", reason=str(error) or "SIGINT")
                raise
            except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
                item.update(status="failed", reason=str(error))
            finally:
                item["finished_at"] = now()
                save()
                append_record(plan, item["status"], stage=stage, reason=item.get("reason"), result=result)
                display(f"[{name}] {item['status']}: {item.get('reason', result or folder)}")
        state["status"] = "failed" if any(item["status"] == "failed" for item in states.values()) else "completed"
    except (KeyboardInterrupt, subprocess.TimeoutExpired) as error:
        state.update(status="interrupted", reason=str(error) or "SIGINT")
        for item in states.values():
            if item["status"] == "pending":
                item.update(status="skipped", reason="workflow interrupted")
    except Exception as error:
        state.update(status="failed", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        append_record(plan, state["status"], reason=state.get("reason"), result=str(path))
        display(f"Workflow {state['status']}: {path}")
    return {"completed": 0, "interrupted": 130}.get(state["status"], 1)


def status_snapshot(output):
    workflow = output / "workflow_status.json"
    if not workflow.is_file():
        return {"status": "not_started", "output_dir": str(output)}
    state = read_json(workflow)
    for item in state["stages"].values():
        if item["type"] != "pipeline":
            continue
        folder = Path(item["output_dir"])
        child_status = folder / "status.json"
        if child_status.is_file():
            item["pipeline"] = read_json(child_status)
        if (folder / "training" / "result.json").is_file():
            item["training_result"] = str(folder / "training" / "result.json")
        partial = folder / "evaluation" / "partial_summary.json"
        if item.get("evaluation_complete"):
            item["final_summary"] = str(folder / "summary.json")
        elif partial.is_file():
            payload = read_json(partial)
            item["partial_evaluation"] = {key: payload[key] for key in (
                "status", "completed_shards", "total_shards", "completed_episodes")}
            item["partial_evaluation"]["path"] = str(partial)
    return state


def launch_terminal(plan):
    output = Path(plan["output_dir"])
    environment = os.environ.copy()
    # Do not inject AppImage Python/GTK libraries into the system GNOME launcher.
    appdir = environment.get("APPDIR")
    if appdir:
        for key in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH", "GTK_PATH", "GTK_EXE_PREFIX",
                    "GIO_MODULE_DIR", "GI_TYPELIB_PATH", "PATH"):
            kept = [value for value in environment.get(key, "").split(os.pathsep)
                    if value and not Path(value).is_relative_to(appdir)]
            if kept:
                environment[key] = os.pathsep.join(kept)
            else:
                environment.pop(key, None)
    terminal = shutil.which("gnome-terminal", path=environment.get("PATH"))
    require(terminal, "gnome-terminal not found; no background fallback")
    command = [terminal, "--window", f"--working-directory={ROOT}", f"--title=Experiment: {plan['id']}", "--",
               sys.executable, "-B", "-u", "-m", MODULE, "_run", "--output-dir", str(output)]
    with (output / "terminal_launch.log").open("w", encoding="utf-8") as log:
        log.write(shlex.join(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=15)
    require(not result.returncode, f"GNOME Terminal failed (exit {result.returncode}); see {output / 'terminal_launch.log'}")
    print(f"GNOME Terminal accepted launch: {output}", flush=True)


def stop(signum, _frame):
    raise KeyboardInterrupt(signal.Signals(signum).name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight", help="read-only card validation").add_argument("--card", required=True)
    launch = sub.add_parser("launch", help="preflight and execute an authorized card")
    launch.add_argument("--card", required=True)
    launch.add_argument("--terminal", choices=("gnome", "current"), default="gnome")
    sub.add_parser("status", help="read one snapshot (no polling)").add_argument("--output-dir", required=True)
    sub.add_parser("_run", help=argparse.SUPPRESS).add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        if args.command == "status":
            print(json.dumps(status_snapshot(output_path(args.output_dir)), ensure_ascii=False, indent=2))
            return 0
        if args.command == "_run":
            output = output_path(args.output_dir)
            plan_path = output / "preflight.json"
            plan = read_json(plan_path)
            require(Path(plan["output_dir"]) == output, "preflight output mismatch")
            with (output / "worker.claim").open("x", encoding="utf-8") as claim:
                claim.write(f"{os.getpid()}\n")
            return execute(plan)
        card_path = file_path(args.card)
        try:
            plan = preflight(card_path)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            if args.command == "launch":
                try:
                    card = read_json(card_path)
                    requested = card.get("output_dir", "未指定")
                    name = card.get("id", card_path.stem)
                except (OSError, ValueError, AttributeError):
                    requested, name = "未解析", card_path.stem
                append_record({"id": name, "card": str(card_path), "output_dir": "未启动"},
                              "interrupted" if isinstance(error, KeyboardInterrupt) else "preflight_failed",
                              reason=str(error), result=f"请求目录 `{requested}` 未写入")
            raise
        if args.command == "preflight":
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        output = Path(plan["output_dir"])
        output.mkdir(parents=True, exist_ok=False)
        shutil.copy2(card_path, output / "experiment_card.json")
        write_json(output / "preflight.json", plan)
        write_json(output / "workflow_status.json", {"id": plan["id"], "status": "launching", "stages": {}})
        if args.terminal == "current":
            return execute(plan)
        try:
            launch_terminal(plan)
        except (OSError, ValueError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            write_json(output / "workflow_status.json", {"id": plan["id"], "status": status,
                       "stages": {}, "reason": str(error), "updated_at": now()})
            append_record(plan, status, reason=str(error), result=str(output / "workflow_status.json"))
            raise
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        print(f"{args.command}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
