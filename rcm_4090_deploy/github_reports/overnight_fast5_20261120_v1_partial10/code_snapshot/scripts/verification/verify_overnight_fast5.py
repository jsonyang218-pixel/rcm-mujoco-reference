"""Lightweight fast5 pairing/gating checks; no training or rollout."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.research.overnight_cfix import Interrupted
from scripts.research.overnight_cfix_fast import FAST5, FastRunner, comparison, paired_rows
from scripts.research.overnight_cfix_eval import parallel_evaluate, summarize, targets


class Fast5Tests(unittest.TestCase):
    output = None

    def row(self, index, final=10, minimum=5):
        return {"target_id": f"t{index}", "episode": f"E{index:02d}", "seed": 42+index,
                "final_error_um": final, "minimum_error_um": minimum}

    def test_pairs_subset_by_original_id_and_seed(self):
        base = {f"t{i}": self.row(i) for i in range(20)}
        trial = {f"t{i}": self.row(i, 9, 4) for i in reversed(FAST5)}
        pairs = paired_rows(base, trial)
        self.assertEqual({row["seed"] for row in pairs}, {42, 44, 48, 50, 52})
        trial["t8"]["seed"] = 45
        with self.assertRaises(ValueError):
            paired_rows(base, trial)

    def test_rejects_unknown_duplicate_and_nonfinite_results(self):
        with self.assertRaises(ValueError):
            paired_rows({"a": self.row(0)}, {"b": self.row(1)})
        with self.assertRaises(ValueError):
            paired_rows({"a": self.row(0)}, {"a": self.row(0, float("nan"))})

    def test_partial_summary_never_published(self):
        folder = self.output / "partial"
        folder.mkdir()
        with self.assertRaises(ValueError):
            summarize(folder, [{"episode_index": i} for i in FAST5[:-1]], Path("none"), "I0",
                      indices=FAST5, max_steps=1000, tier="fast5")
        self.assertFalse((folder / "summary.json").exists())

    def check_worker_cleanup(self, name, interrupt):
        from lerobot_training.run_model_pipeline import start_logged as real_start
        jobs = []
        out = self.output / name
        args = SimpleNamespace(output_dir=out, workers=3, target_manifest=Path(
            "configs/v12_reachable_development_targets_0p35um.json"), checkpoint=Path("fixture"),
            arm="I0", tier="fast5", smoke=True)

        def start(_command, log, **options):
            script = "import time; time.sleep(30)" if interrupt or jobs else "import time; time.sleep(.1); raise SystemExit(2)"
            job = real_start([sys.executable, "-c", script], log, **options)
            jobs.append(job)
            return job

        timer = threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM)) if interrupt else None
        if timer:
            timer.start()
        try:
            with patch("lerobot_training.run_model_pipeline.start_logged", side_effect=start):
                with self.assertRaises(KeyboardInterrupt if interrupt else RuntimeError):
                    parallel_evaluate(args, FAST5[:3], 3)
        finally:
            if timer:
                timer.cancel()
        self.assertEqual(len(jobs), 3)
        self.assertTrue(all(job["process"].poll() is not None for job in jobs))
        self.assertFalse((out / "summary.json").exists())
        state = json.loads((out / "progress.json").read_text())
        self.assertEqual(state["status"], "interrupted" if interrupt else "failed")
        self.assertEqual(state["validated_targets"], 0)

    def test_worker_failure_stops_sibling_processes(self):
        self.check_worker_cleanup("worker_failure", False)

    def test_worker_interruption_stops_sibling_processes(self):
        self.check_worker_cleanup("worker_interruption", True)

    def test_gate_threshold_and_protocol(self):
        base, trial = [self.output / name for name in ("base", "trial")]
        summary = {"status": "completed", "arm": "I0", "max_steps": 1000, "tier": "fast5",
                   "error_threshold_um": 0.35, "flow_steps": 4, "samples": 1, "target_count": 5}
        for folder, value in ((base, 10), (trial, 9)):
            folder.mkdir()
            (folder / "summary.json").write_text(json.dumps(summary))
            rows = [self.row(i, value) for i in FAST5]
            with (folder / "episodes.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        self.assertTrue(comparison(base, trial)["gate_passed"])
        summary["max_steps"] = 1600
        (trial / "summary.json").write_text(json.dumps(summary))
        with self.assertRaises(ValueError):
            comparison(base, trial)

    def test_frozen_manifest_original_indices(self):
        selected = targets(Path("configs/v12_reachable_development_targets_0p35um.json"))
        self.assertEqual(len({selected[index]["id"] for index in FAST5}), 5)
        self.assertEqual([42+i for i in FAST5], [42, 44, 48, 50, 52])

    def fixture(self, name, *, step_zero=True, improves=True):
        folder = self.output / name / "task34"
        folder.mkdir(parents=True)
        runner = FastRunner(folder, 5)
        runner.checkpoints = self.output / name / "checkpoints"
        calls = []
        runner.record = lambda stage, status, details: calls.append((stage, status, details))

        def evaluate(name, checkpoint, arm, tier="fast5"):
            calls.append((name, "eval", {"tier": tier}))
            return {"arm": arm, "checkpoint": str(checkpoint), "final_mean_um": 8 if "task4" in name else 10,
                    "minimum_mean_um": 4 if "task4" in name else 5,
                    "inference_median_ms": 55, "inference_p90_ms": 60,
                    "canonical_success_count": 0}

        runner.evaluate_tier = evaluate
        runner.cmd = lambda *args, **kwargs: (0, "mock")
        runner.train = lambda *args, **kwargs: calls.append(("train", "called", {})) or Path("trained")
        real_read = __import__("scripts.research.overnight_cfix_fast", fromlist=["read"]).read

        def fake_read(path):
            if path.name == "step_zero_selected.json":
                return {"status": "PASS" if step_zero else "FAILED", "max_abs_diff": 0 if step_zero else 1}
            return real_read(path)

        with patch("scripts.research.overnight_cfix_fast.continuation_inputs", return_value=(Path("base"), Path("overlay"))), \
             patch("scripts.research.overnight_cfix_fast.identity", return_value={"code": "fixed"}), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_name", return_value="RTX 4090"), \
             patch("scripts.research.overnight_cfix_fast.episode_rows", return_value={"t0": self.row(0)}), \
             patch("scripts.research.overnight_cfix_fast.read", side_effect=fake_read), \
             patch("scripts.research.overnight_cfix_fast.comparison", return_value={"gate_passed": improves}), \
             patch("scripts.research.overnight_cfix_fast.near_target_plot"):
            if not step_zero:
                with self.assertRaises(ValueError):
                    runner.run_fast(Path("mock_receipt"))
            elif not improves:
                runner.run_fast(Path("mock_receipt"))
            else:
                # Reaching dev20 report proves automatic propagation; stop fixture before real file reads.
                with patch("scripts.research.overnight_cfix_fast.near_target_plot", side_effect=Interrupted("fixture complete")):
                    with self.assertRaises(Interrupted):
                        runner.run_fast(Path("mock_receipt"))
        return calls

    def test_zero_update_gate_stops_training(self):
        calls = self.fixture("failed_zero", step_zero=False)
        self.assertFalse(any(stage == "train" for stage, _, _ in calls))

    def test_failed_screen_gate_skips_dev20(self):
        calls = self.fixture("failed_screen", improves=False)
        self.assertEqual(sum(state == "eval" and "task4_step" in stage for stage, state, _ in calls), 2)
        self.assertTrue(any(stage == "task4_dev20" and state == "skipped" for stage, state, _ in calls))

    def test_passed_screen_runs_one_dev20(self):
        calls = self.fixture("passed_screen")
        self.assertEqual(sum(state == "eval" and detail["tier"] == "dev20" for _, state, detail in calls), 1)
        selected = next(detail for stage, _, detail in calls if stage == "task3_selection")
        self.assertEqual(selected["arm"], "I0")  # Latency gate rejects both candidates.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    Fast5Tests.output = args.output_dir
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Fast5Tests))
    (args.output_dir / "verification.json").write_text(json.dumps({
        "status": "passed" if result.wasSuccessful() else "failed", "tests": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "scope": "fixtures only; no training/full evaluation"}, indent=2) + "\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
