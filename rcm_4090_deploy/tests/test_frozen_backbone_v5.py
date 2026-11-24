"""Contracts for the frozen-backbone v5 three-hour experiment."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lerobot_training.dit_experience_replay import (  # noqa: E402
    IndexSubset,
    RatioReplacementBatchSampler,
    _optimizer,
    _scheduler,
    assert_temporal_bias_only,
    stationary_sampling_summary,
)
from scripts.research.frozen_backbone_v5_3h import (  # noqa: E402
    can_start,
    checkpoint_freeze_evidence,
    diagnose_stall_or_oscillation,
    make_report,
    phase1_candidate,
    phase2_gate,
    run_managed,
    select_top1,
    should_skip_after_first_checkpoint,
)
from scripts.research.frozen_backbone_v5_publish import (  # noqa: E402
    _lightweight_files,
    scan_secret_material,
)
from scripts.research.homog_label_fix_repair import (  # noqa: E402
    build_fixed_dataset,
    repair_arrays,
    repair_sample_weights,
    window_weights,
)


def fixture():
    count = 5
    return {
        "episode_index": np.array([0, 20, 20, 40, 40], dtype=np.int64),
        "frame_index": np.arange(count, dtype=np.int32),
        "state": np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        "state_si": np.arange(count * 3, dtype=np.float64).reshape(count, 3),
        "action": np.array([
            [.1, .05, 0.],
            [0., .06, 0.],
            [-.04, .02, 0.],
            [.01, .02, .03],
            [.01, .02, .03],
        ], dtype=np.float32),
        "executed_action": np.array([
            [.05, 0., 0.],
            [0., .03, 0.],
            [.02, 0., 0.],
            [.02, .03, .004],
            [0., 0., 0.],
        ], dtype=np.float32),
        "zero": np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [1., 0., 0.],
            [.02, .03, .004],
            [.01, .02, .003],
        ]),
        "pre": np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [-1., 0., 0.],
            [.02, .03, .004],
            [.01, .02, .003],
        ]),
        "truth_navigation_mm": np.array([
            [.01, 0., 0.],
            [0., .01, 0.],
            [0., .01, 0.],
            [0., 0., .01],
            [0., 0., .004],
        ]),
        "source_frame_index": np.arange(count, dtype=np.int64),
        "timestamp_s": np.arange(count, dtype=np.float64) * .05,
        "oct_navigation": np.ones((count, 3), dtype=np.float32) * .01,
        "oct_navigation_valid": np.ones(count, dtype=np.bool_),
        "hold": np.array([False, False, False, False, True]),
    }


class DatasetWeightTests(unittest.TestCase):
    def test_action_selection_and_exact_v5_weights(self):
        arrays = fixture()
        actions, _, stationary = repair_arrays(arrays, "executed_only")
        np.testing.assert_array_equal(
            actions, arrays["executed_action"]
        )
        for weight in (.25, .10):
            values = repair_sample_weights(
                arrays,
                "executed_only",
                stationary,
                t3_stationary_weight=weight,
            )
            self.assertAlmostEqual(float(values[4]), weight, places=6)
            np.testing.assert_array_equal(values[:4], np.ones(4))
        actions, _, stationary = repair_arrays(arrays, "t3_only")
        np.testing.assert_array_equal(
            actions[:3], arrays["executed_action"][:3]
        )
        values = repair_sample_weights(
            arrays,
            "t3_only",
            stationary,
            t3_stationary_weight=.25,
        )
        np.testing.assert_array_equal(values[:3], np.zeros(3))
        self.assertEqual(float(values[3]), 1.)
        self.assertAlmostEqual(float(values[4]), .25, places=6)
        np.testing.assert_array_equal(window_weights([1], values, 2), [0.])

    def test_full_dataset_contract_preserves_everything_else(self):
        temporary = Path(tempfile.mkdtemp(prefix="frozen_v5_repair_"))
        self.addCleanup(shutil.rmtree, temporary, True)
        audit, receipt = build_fixed_dataset(
            temporary / "audit",
            temporary / "dataset",
            "t3_only",
            t3_stationary_weight=.10,
        )
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["t3_stationary_weight"], .10)
        self.assertIn("nonzero_weight_fraction", audit)
        self.assertIn(
            "empirical_t3_stationary_action_fraction", audit
        )
        source_path = (
            ROOT
            / "data/homog_clean_replay_20261123_mixed_1600_600_12_hold_v2"
            / "data/dataset.npz"
        )
        with np.load(source_path, allow_pickle=False) as source:
            with np.load(
                temporary / "dataset/data/dataset.npz",
                allow_pickle=False,
            ) as fixed:
                self.assertEqual(
                    fixed["correction_sample_weight"][
                        (fixed["episode_index"] >= 40)
                        & np.all(fixed["executed_action"] == 0, axis=1)
                        & (
                            np.linalg.norm(
                                fixed["truth_navigation_mm"].astype(float),
                                axis=1,
                            ) * 1000 < 5
                        )
                    ].min(),
                    np.float32(.10),
                )
                for key in (
                    "episode_index",
                    "frame_index",
                    "state",
                    "state_si",
                    "executed_action",
                    "zero",
                    "pre",
                    "truth_navigation_mm",
                    "source_frame_index",
                    "timestamp_s",
                    "oct_navigation",
                    "oct_navigation_valid",
                    "hold",
                ):
                    np.testing.assert_array_equal(fixed[key], source[key])
        with np.load(
            temporary / "dataset/data/valid_window_indices.npz",
            allow_pickle=False,
        ) as windows:
            rows = windows["rows"].copy()
        with np.load(
            ROOT
            / "data/homog_clean_replay_20261123_mixed_1600_600_12_hold_v2"
            / "data/valid_window_indices.npz",
            allow_pickle=False,
        ) as source_windows:
            np.testing.assert_array_equal(rows, source_windows["rows"])
        with np.load(
            temporary / "dataset/data/label_repair_overlay.npz",
            allow_pickle=False,
        ) as overlay:
            self.assertEqual(float(overlay["t3_stationary_weight"]), .10)
        self.assertEqual(receipt["t3_stationary_weight"], .10)


class SamplerTelemetryTests(unittest.TestCase):
    def test_ratio_is_deterministic_and_near_target_weight_is_exact(self):
        def build():
            return RatioReplacementBatchSampler(
                50, 5, .20, 10, 10, 7,
                correction_weights=[0., 1., 1., 1., 0.],
                near_target_mask=[False, True, False, False, False],
                near_target_weight=2.,
            )

        first = list(build())
        second = list(build())
        self.assertEqual(first, second)
        flat = np.asarray(first).reshape(-1)
        self.assertEqual(int(np.sum(flat >= 50)), 20)
        probabilities = build().correction_probabilities
        self.assertAlmostEqual(float(probabilities[1]), .5)
        self.assertAlmostEqual(float(probabilities[2]), .25)

    def test_stationary_counts_and_weights_enter_batch_and_hand_summary(self):
        class Tiny:
            def __len__(self):
                return 4

            def __getitem__(self, index):
                return {"action": np.array([index], dtype=np.int64)}

        subset = IndexSubset(
            Tiny(),
            [0, 1, 2, 3],
            source=1,
            stationary_frame_counts=[0, 2, 4, 0],
            correction_weights=[1., .5, .5, 1.],
        )
        sample = subset[2]
        self.assertEqual(int(sample["sample.stationary_frame_count"]), 4)
        self.assertEqual(float(sample["sample.correction_weight"]), .5)
        counts = {
            "correction_samples": 4,
            "correction_action_frames": 64,
            "t3_stationary_action_frames": 10,
            "stationary_windows": 3,
            "total_samples": 8,
        }
        summary = stationary_sampling_summary(
            counts,
            weights=np.array([1., .5, .5, 1.]),
            stationary_counts=np.array([0, 2, 4, 0]),
            correction_ratio=.20,
            action_steps=16,
        )
        self.assertAlmostEqual(
            summary["actual_t3_stationary_action_fraction"], 10 / 64
        )
        self.assertAlmostEqual(
            summary[
                "theoretical_weighted_t3_stationary_action_fraction"
            ],
            3 / 48,
        )
        self.assertAlmostEqual(
            summary[
                "all_batch_theoretical_t3_stationary_action_fraction"
            ],
            .2 * 3 / 48,
        )
        self.assertAlmostEqual(
            summary["all_batch_actual_t3_stationary_action_fraction"],
            10 / (8 * 16),
        )

class FrozenTrainingTests(unittest.TestCase):
    class Model(__import__("torch").nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = __import__("torch").nn.Linear(3, 3)
            self.temporal_bias = __import__("torch").nn.Linear(3, 3)

        def forward(self, value):
            return self.backbone(value) + self.temporal_bias(value)

    def test_only_temporal_bias_is_trainable_and_must_be_enforced(self):
        import torch

        model = self.Model()
        args = types.SimpleNamespace(
            backbone_lr=0.,
            attention_lr=3e-5,
            weight_decay=1e-4,
            lr_schedule="cosine",
            lr_min=1e-6,
            require_temporal_bias_only=True,
        )
        before = model.backbone.weight.detach().clone()
        optimizer = _optimizer(model, args)
        assert_temporal_bias_only(model, args)
        self.assertFalse(model.backbone.weight.requires_grad)
        self.assertTrue(model.temporal_bias.weight.requires_grad)
        self.assertEqual(len(optimizer.param_groups), 1)
        optimizer.zero_grad()
        sum(
            parameter.square().sum()
            for parameter in model.parameters()
            if parameter.requires_grad
        ).backward()
        optimizer.step()
        torch.testing.assert_close(model.backbone.weight, before)
        backbone_after = model.backbone.weight.detach().clone()
        model.backbone.weight.requires_grad_(True)
        optimizer.zero_grad()
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        with self.assertRaises(ValueError):
            assert_temporal_bias_only(model, args)
        torch.testing.assert_close(model.backbone.weight, backbone_after)
        bad = types.SimpleNamespace(
            **{**vars(args), "backbone_lr": 1e-6}
        )
        with self.assertRaises(ValueError):
            assert_temporal_bias_only(model, bad)

    def test_cosine_floor_and_constant_compatibility(self):
        import torch

        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=3e-5)
        args = types.SimpleNamespace(
            lr_schedule="cosine", lr_min=1e-6,
        )
        scheduler = _scheduler(optimizer, args, 8)
        for _ in range(8):
            optimizer.step()
            scheduler.step()
        self.assertGreaterEqual(optimizer.param_groups[0]["lr"], 1e-6)
        constant = types.SimpleNamespace(
            lr_schedule="constant", lr_min=1e-6,
        )
        self.assertIsNone(_scheduler(optimizer, constant, 8))

    def test_actual_checkpoint_freeze_evidence(self):
        import torch

        with tempfile.TemporaryDirectory(
            prefix="frozen_v5_checkpoint_freeze_"
        ) as temporary:
            root = Path(temporary)
            model = self.Model()
            base = root / "base.pth"
            candidate = root / "candidate.pth"
            torch.save({"model_state_dict": model.state_dict()}, base)
            with torch.no_grad():
                model.temporal_bias.weight.add_(1.)
            torch.save(
                {"model_state_dict": model.state_dict()}, candidate,
            )
            evidence = checkpoint_freeze_evidence(base, candidate)
            self.assertTrue(evidence["passed"])
            self.assertEqual(evidence["frozen_tensors_changed"], 0)
            with torch.no_grad():
                model.backbone.weight.add_(1.)
            torch.save(
                {"model_state_dict": model.state_dict()}, candidate,
            )
            with self.assertRaises(ValueError):
                checkpoint_freeze_evidence(base, candidate)


class OrchestratorGateTests(unittest.TestCase):
    def _fast(
        self,
        final=60.,
        minimum=20.,
        drift=5.,
        strict=0,
        hits=(1, 2, 3),
    ):
        return {
            "protocol": {"checkpoint": "/checkpoint.pth"},
            "completed_targets": 5,
            "final_mean_um": final,
            "min_mean_um": minimum,
            "strict_success_count": strict,
            "final_hits_le_5_um": hits[2],
            "final_hits_le_10_um": hits[1],
            "final_hits_le_30_um": hits[0],
            "absolute_drift_1000_to_1600_um": drift,
        }

    def _snapshot(self, toward=.5, reverse=2):
        return {
            "first_order_toward_fraction": toward,
            "axis_reverse_snapshot_counts": {"A": reverse, "B": 0, "C": 0},
        }

    def _candidate(self, arm, toward, final=60., minimum=20., drift=5.,
                   strict=0, step=2000, baseline=.5):
        return phase1_candidate(
            arm,
            f"/{arm}.pth",
            self._fast(final, minimum, drift, strict),
            self._snapshot(toward),
            baseline_toward_fraction=baseline,
            step=step,
        )

    def test_baseline_relative_gate_ranking_and_early_stop(self):
        failed = self._candidate("A", .47, final=40.)
        self.assertFalse(failed["direction_gate_passed"])
        passed = self._candidate("B", .49, final=60.)
        self.assertTrue(passed["direction_gate_passed"])
        self.assertGreater(should_skip_after_first_checkpoint(
            self._candidate("B", .6, final=75.01)
        ), 0)
        self.assertFalse(should_skip_after_first_checkpoint(
            self._candidate("B", .6, final=75.)
        ))
        better_strict = self._candidate(
            "C", .6, final=80., strict=1, step=4000,
        )
        better_final = self._candidate(
            "B", .6, final=50., minimum=40., step=6000,
        )
        best_min = self._candidate(
            "A", .6, final=50., minimum=20., step=2000,
        )
        selected = select_top1(
            [failed, passed, better_strict, better_final, best_min]
        )
        self.assertEqual(selected["arm"], "C")
        selected = select_top1([failed, better_final, best_min])
        self.assertEqual(selected["arm"], "A")
        self.assertIsNone(select_top1([failed]))

    def test_phase2_or_gate_and_time_capacity(self):
        self.assertTrue(phase2_gate(self._candidate(
            "A", .6, final=54.30,
        )))
        self.assertTrue(phase2_gate(self._candidate(
            "A", .6, final=90., minimum=29.99,
        )))
        self.assertTrue(phase2_gate(self._candidate(
            "A", .6, final=90., minimum=40., strict=1,
        )))
        self.assertFalse(phase2_gate(self._candidate(
            "A", .6, final=54.31, minimum=30., strict=0,
        )))
        now = time.time()
        self.assertTrue(can_start(now, 35, now=now))
        self.assertFalse(can_start(now - 121 * 60, 35, now=now))
        self.assertFalse(can_start(
            now - 165 * 60, 0, now=now, reserve=0,
        ))

    def test_process_group_timeout_and_report_on_interruption(self):
        with tempfile.TemporaryDirectory(prefix="frozen_v5_timeout_") as temporary:
            log = Path(temporary) / "sleep.log"
            with self.assertRaises(TimeoutError):
                run_managed(
                    ["bash", "-c", "sleep 30"],
                    log,
                    time.time() + .2,
                    cwd=temporary,
                )
            self.assertTrue(log.is_file())
            summary = {
                "elapsed_minutes": 180.,
                "stages": {},
            }
            report = make_report(
                Path(temporary),
                summary,
                "interrupted",
                "TimeoutError('hard deadline exceeded')",
            )
            text = report.read_text(encoding="utf-8")
            self.assertIn("hard deadline exceeded", text)
            self.assertIn("software and MuJoCo simulation only", text)

    def test_stall_vs_oscillation_diagnostic(self):
        def write_summary(root, actions, errors):
            root.mkdir(parents=True)
            result_path = root / "evaluation_results.json"
            result_path.write_text(json.dumps({
                "episodes": [{
                    "trajectory": {
                        "actions": actions.tolist(),
                        "errors": errors.tolist(),
                    }
                }]
            }))
            return {
                "episodes": [{
                    "result_path": str(result_path),
                    "result_sha256": __import__(
                        "hashlib"
                    ).sha256(result_path.read_bytes()).hexdigest(),
                }]
            }

        with tempfile.TemporaryDirectory(prefix="frozen_v5_stall_") as temporary:
            root = Path(temporary)
            errors = np.ones(1600) * .001
            stable = np.ones((1600, 3)) * .01
            small = stable * .25
            baseline = write_summary(root / "b", stable, errors)
            stalled = write_summary(root / "s", small, errors)
            value = diagnose_stall_or_oscillation(baseline, stalled)
            self.assertEqual(value["classification"], "stalling")
            alternating = stable.copy()
            alternating[1000::2, 0] *= -1
            oscillating = write_summary(
                root / "o", alternating, errors * .5,
            )
            value = diagnose_stall_or_oscillation(baseline, oscillating)
            self.assertEqual(value["classification"], "oscillation")


class PublicationTests(unittest.TestCase):
    def test_allowlist_excludes_weight_raw_and_shard_files(self):
        with tempfile.TemporaryDirectory(prefix="frozen_v5_publish_") as temporary:
            root = Path(temporary)
            (root / "reports/raw").mkdir(parents=True)
            (root / "reports/shards").mkdir(parents=True)
            for relative in (
                "summary.json",
                "plot.png",
                "weights.pth",
                "dataset.npz",
                "array.npy",
                "reports/raw/trajectory.json",
                "reports/shards/result.json",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"{}")
            names = {
                path.relative_to(root).as_posix()
                for path in _lightweight_files(root)
            }
            self.assertEqual(names, {"summary.json", "plot.png"})

    def test_secret_scanner_rejects_material_without_echoing_it(self):
        with tempfile.TemporaryDirectory(prefix="frozen_v5_secret_") as temporary:
            root = Path(temporary)
            safe = root / "safe.md"
            safe.write_text("safe", encoding="utf-8")
            scan_secret_material({"safe": safe})
            unsafe = root / "unsafe.txt"
            unsafe.write_bytes(
                b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----\n"
            )
            with self.assertRaises(ValueError) as raised:
                scan_secret_material({"unsafe": unsafe})
            self.assertNotIn(
                "OPENSSH PRIVATE KEY", str(raised.exception)
            )


if __name__ == "__main__":
    unittest.main()
