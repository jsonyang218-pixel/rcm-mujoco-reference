"""Persistent spawn workers, per-task deadlines and checked artifact reuse."""
from __future__ import annotations

from collections import deque
from datetime import datetime
import hashlib
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from lerobot_training.navigation_overlay import file_sha256

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def record(purpose, status, output, detail):
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with (ROOT / "TEST_LOG.md").open("ab") as stream:
        stream.write((f"\n- {stamp}: clean OCT/C-bias {purpose} **{status}**; "
                      f"command `{' '.join(sys.argv)}`; {detail}; output `{output}`. "
                      "Software/MuJoCo simulation only.\n").encode())


def artifact_hashes(folder):
    return {str(path.relative_to(folder)): file_sha256(path)
            for path in sorted(folder.rglob("*"))
            if path.is_file() and path.name != "receipt.json"}


def reuse(folder, expected):
    receipt = json.loads((folder / "receipt.json").read_text())
    if (receipt["status"] != "completed" or receipt["signature"] != expected
            or not receipt["files"] or artifact_hashes(folder) != receipt["files"]):
        raise ValueError(f"incomplete, changed or incompatible task receipt: {folder}")
    return receipt["result"]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def worker_main(connection, factory, spec, log_path):
    os.environ.update(CUDA_VISIBLE_DEVICES="0", MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID="0",
                      MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    handler = None
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with Path(log_path).open("x", buffering=1) as log:
        sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
        try:
            module, name = factory.split(":")
            handler = getattr(importlib.import_module(module), name)(spec)
            connection.send({"event": "ready"})
            while True:
                request = connection.recv()
                if request is None:
                    break
                result = handler.run(request["task"], Path(request["output_dir"]))
                connection.send({"event": "completed", "result": result})
        except BaseException as error:
            import traceback
            traceback.print_exc()
            try:
                connection.send({"event": "failed", "error": repr(error)})
            except (BrokenPipeError, EOFError):
                pass
        finally:
            try:
                if handler is not None:
                    handler.close()
            finally:
                connection.close()
                # Multiprocessing flushes streams again during process exit.
                sys.stdout, sys.stderr = original_stdout, original_stderr


def resources():
    gpu = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True,
    ).strip().split(",")
    memory = {line.split(":", 1)[0]: int(line.split()[1])
              for line in Path("/proc/meminfo").read_text().splitlines()}
    return {"gpu_free_mib": int(gpu[0]), "gpu_utilization_percent": int(gpu[1]),
            "ram_available_mib": memory["MemAvailable"] // 1024}


def run_pool(groups, output_dir, *, workers=4, max_workers=8, timeout_s=600,
             stage_timeout_s=2400, resume=False, initializer_timeout_s=120,
             execution_timeout_s=None, snapshot_every=None):
    """Groups contain factory/spec/tasks; no task is retried within a run."""
    if (not groups or not 1 <= workers <= max_workers <= 8
            or any(value is not None and value <= 0
                   for value in (timeout_s, stage_timeout_s, initializer_timeout_s, execution_timeout_s))):
        raise ValueError("invalid pool size or deadline")
    output = Path(output_dir).resolve()
    if output.exists() and not resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    contract = {"groups": groups, "workers": workers, "max_workers": max_workers,
                "timeout_s": timeout_s, "stage_timeout_s": stage_timeout_s,
                "pool_sha256": file_sha256(__file__)}
    if timeout_s is None or stage_timeout_s is None or snapshot_every is not None:
        contract.update(initializer_timeout_s=initializer_timeout_s,
                        execution_timeout_s=execution_timeout_s, snapshot_every=snapshot_every)
    if snapshot_every is not None and (not isinstance(snapshot_every, int) or snapshot_every < 1):
        raise ValueError("snapshot_every must be a positive integer")
    contract_sha = signature(contract)
    budgets = [v for v in (stage_timeout_s, execution_timeout_s) if v is not None]
    effective_budget = min(budgets) if budgets else None
    contract_path = output / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("resume contract differs; no completed evidence reused")
    else:
        if resume:
            raise ValueError("resume needs the original contract")
        write_json(contract_path, contract)
    queue, results, receipts = {}, {}, []
    tasks_root = output / "tasks"
    tasks_root.mkdir(exist_ok=True)
    for group_id, group in groups.items():
        queue[group_id] = deque()
        for task in group["tasks"]:
            key = f"{group_id}_{task['id']}"
            if key in results or any(item[0] == key for q in queue.values() for item in q):
                raise ValueError("duplicate task identifier")
            folder = tasks_root / key
            expected = signature({"contract": contract_sha, "group": group_id, "task": task})
            if (folder / "receipt.json").exists():
                results[key] = reuse(folder, expected)
                receipts.append({"task": key, "signature": expected, "reused": True})
            else:
                if folder.exists():
                    if not resume:
                        raise FileExistsError(folder)
                    archive = output / "interrupted"
                    archive.mkdir(exist_ok=True)
                    folder.rename(archive / f"{key}_{time.time_ns()}")
                queue[group_id].append((key, task, folder, expected))
    write_json(output / "reuse_receipt.json", receipts)
    started, next_heartbeat = time.monotonic(), time.monotonic()
    running, completed_times, last_scaled = [], deque(maxlen=32), None
    state = {"status": "running", "completed": len(results), "total": sum(len(g['tasks']) for g in groups.values()),
             "results": results, "workers": [], "contract_sha256": contract_sha}
    context = mp.get_context("spawn")

    def save():
        state["elapsed_s"] = time.monotonic() - started
        state["completed"] = len(results)
        write_json(output / "partial_summary.json", state)

    def spawn(group_id):
        parent, child = context.Pipe()
        log = output / f"worker_{time.time_ns()}.log"
        group = groups[group_id]
        process = context.Process(target=worker_main,
                                  args=(child, group["factory"], group["spec"], str(log)))
        process.start()
        child.close()
        item = {"process": process, "connection": parent, "group": group_id,
                "started": time.monotonic(), "active": None, "ready": False}
        running.append(item)
        state["workers"].append({"pid": process.pid, "group": group_id, "log": str(log)})

    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")

    old_handlers = {sig: signal.signal(sig, interrupt)
                    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        while any(queue.values()) or any(w["active"] or not w["ready"] for w in running):
            now = time.monotonic()
            if effective_budget is not None and now - started >= effective_budget:
                raise TimeoutError(f"stage budget {effective_budget}s exceeded")
            busy = len(running)
            desired = workers if last_scaled is None else last_scaled
            if busy < min(desired, sum(len(q) for q in queue.values())):
                group_id = max(queue, key=lambda key: len(queue[key]) / (1 + sum(w['group'] == key for w in running)))
                available = resources()
                if available['gpu_free_mib'] < 10240 or available['ram_available_mib'] < 10240:
                    raise MemoryError(f"insufficient worker admission reserve: {available}")
                spawn(group_id)
            for item in list(running):
                if item["connection"].poll():
                    event = item["connection"].recv()
                    if event["event"] == "failed":
                        raise RuntimeError(f"worker {item['process'].pid}: {event['error']}")
                    if event["event"] == "ready":
                        item["ready"] = True
                    elif event["event"] == "completed":
                        key, task, folder, expected = item["active"]
                        files = artifact_hashes(folder)
                        if not files:
                            raise ValueError(f"worker produced no evidence: {key}")
                        write_json(folder / "receipt.json", {"status": "completed", "signature": expected,
                                                             "files": files, "result": event["result"]})
                        results[key] = event["result"]
                        completed_times.append(now)
                        receipts.append({"task": key, "signature": expected, "reused": False})
                        write_json(output / "reuse_receipt.json", receipts)
                        item["active"] = None
                        record(key, "completed", folder, f"result={event['result']}; signature={expected}")
                        save()
                        if snapshot_every and len(results) % snapshot_every == 0:
                            write_json(output / f"snapshot_{len(results):03d}.json", state)
                if not item["process"].is_alive():
                    raise RuntimeError(f"worker {item['process'].pid} exited without a complete shutdown")
                if not item["ready"]:
                    if initializer_timeout_s is not None and now - item["started"] > initializer_timeout_s:
                        raise TimeoutError("worker initialization timed out")
                    continue
                if item["active"] is None and queue[item["group"]]:
                    item["active"] = queue[item["group"]].popleft()
                    key, task, folder, _ = item["active"]
                    folder.mkdir()
                    item["task_started"] = time.monotonic()
                    item["connection"].send({"task": task, "output_dir": str(folder)})
                elif item["active"] and timeout_s is not None and now - item["task_started"] > timeout_s:
                    raise TimeoutError(f"task {item['active'][0]} exceeded {timeout_s}s")
            if now >= next_heartbeat:
                available = resources()
                state["resources"] = available
                state["active_tasks"] = [w["active"][0] for w in running if w["active"]]
                print(f"GPU pool {len(results)}/{state['total']}; workers={len(running)}; {available}", flush=True)
                # Admit another worker only after completed real work proves progress.
                # Compare completion throughput over equal-size windows, not utilization alone.
                if len(completed_times) >= 8 and len(running) < max_workers and any(queue.values()):
                    prior = 3 / max(completed_times[3] - completed_times[0], .001)
                    recent = 3 / max(completed_times[-1] - completed_times[-4], .001)
                    if (recent >= prior and available['gpu_free_mib'] >= 12288
                            and available['ram_available_mib'] >= 12288):
                        last_scaled = len(running) + 1
                        completed_times.clear()
                next_heartbeat = now + 30
                save()
            time.sleep(.05)
        state["status"] = "completed"
        save()
        write_json(output / "summary.json", state)
        return results
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     failure=repr(error), pending=sum(len(q) for q in queue.values()),
                     active_tasks=[w['active'][0] for w in running if w['active']])
        save()
        record("pool", state['status'], output, f"completed={len(results)}/{state['total']}; {error!r}")
        raise
    finally:
        for item in running:
            if item['process'].is_alive():
                if item['active'] is None and state['status'] == 'completed':
                    item['connection'].send(None)
                else:
                    item['process'].terminate()
        deadline = time.monotonic() + 5
        for item in running:
            item['process'].join(max(0, deadline - time.monotonic()))
            if item['process'].is_alive():
                item['process'].kill()
                item['process'].join()
            item['connection'].close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
