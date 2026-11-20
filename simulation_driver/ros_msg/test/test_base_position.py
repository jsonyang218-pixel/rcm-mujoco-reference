#!/usr/bin/env python3
"""Base targets, independent references, residual accumulation and stop paths."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from robot_control.base_position import (
    MAX_HANDLE_STEP,
    BasePositionExperiment,
    stationary_base_protocol_origin,
)
from robot_control.kinematics import (
    ENCODER_COUNTS_PER_MM,
    model_tip_from_robot_encoders,
    predicted_robot_encoders,
    validate_cartesian_delta,
)
from robot_control.response_experiments import ResponseControl, response_state_key, stationary_protocol_origin
from robot_control.response_signals import ResponseSignal
from robot_control.response_reporting import (
    export_response,
    load_response,
    response_metrics,
)

ROBOT = (-5, 1466022, 893518 + round(0.5 * ENCODER_COUNTS_PER_MM), 0)


def state(robot=ROBOT):
    return dict(
        robot=robot,
        handle=(9, 0, 1, 0),
        position=model_tip_from_robot_encoders(robot, 0),
        forward=False,
        backward=False,
    )


def engine(**kwargs):
    kwargs.setdefault("control", ResponseControl(state_timeout_s=0.5))
    return BasePositionExperiment(
        ResponseSignal(kind="sine", cycles=1, pretrial=True),
        state(),
        (9, 0, 1, 0),
        0,
        1000,
        offset_sign=-1,
        **kwargs
    )


def feed(run, t, value=None):
    run.observe(value or state(), 1000 + t, t, len(run.feedback) + 1)


class BasePositionTest(unittest.TestCase):
    def test_all_elapsed_time_signals_use_base_position_controller(self):
        for kind in ("step", "ramp", "pulse"):
            with self.subTest(kind=kind):
                run = BasePositionExperiment(
                    ResponseSignal(kind=kind, cycles=4),
                    state(),
                    (9, 0, 1, 0),
                    0,
                    1000,
                    xyz_amplitudes_um=(10, 6, 4),
                    offset_sign=-1,
                )
                self.assertEqual(run.mode, "junction_xyz")
                self.assertEqual(run.metadata["controller"], "base_position_residual")
                self.assertEqual(run.reference_at(4.999)["reference_xyz_um"], (0, 0, 0))
                expected_at_start = (0, 0, 0) if kind == "ramp" else (10, 6, 4)
                self.assertEqual(run.reference_at(5)["reference_xyz_um"], expected_at_start)
                self.assertTrue(any(run.reference_at(t)["phase"] not in ("baseline", "recovery")
                                    for t in (5.1, 6, 16)))

    def test_active_base_signals_reject_rcm_start_without_preparation_exception(self):
        rcm_robot = (-5, 1466022, 893518, 0)
        rcm_state = state(rcm_robot)
        for kind in ("step", "ramp", "pulse"):
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, "RCM"):
                    BasePositionExperiment(
                        ResponseSignal(kind=kind, cycles=4), rcm_state,
                        rcm_state["handle"], 0, 1000,
                        xyz_amplitudes_um=(10, 6, 4), offset_sign=-1,
                    )

    def test_active_base_signals_require_sine_workpoint(self):
        robot = (ROBOT[0], ROBOT[1], ROBOT[2] - round(0.2 * ENCODER_COUNTS_PER_MM), ROBOT[3])
        s = state(robot)
        with self.assertRaisesRegex(ValueError, "500±25"):
            BasePositionExperiment(
                ResponseSignal(kind="step"), s, s["handle"], 0, 1000,
                xyz_amplitudes_um=(10, 6, 4), offset_sign=-1,
            )

    def test_startup_accepts_bounded_hold_ripple_but_keeps_exact_protocol(self):
        history = []
        for i in range(61):
            s = state((ROBOT[0], ROBOT[1], ROBOT[2] + (7 if i % 2 else -7), 0))
            history.append((i * .02, response_state_key(s)))
        self.assertEqual(stationary_base_protocol_origin(history, s, 1.21, .5), s['handle'])
        with self.assertRaises(ValueError):
            stationary_protocol_origin(history, s, 1.21, .5)
        for field, value in (('handle', (10, 0, 1, 0)), ('forward', True),
                             ('robot', (*s['robot'][:3], 1)), ('position', (float('nan'), 0, 0))):
            changed = dict(s, **{field: value})
            with self.assertRaises(ValueError):
                stationary_base_protocol_origin(history, changed, 1.21, .5)

    def test_startup_rejects_motion_drift_gaps_and_tip_motion(self):
        for kind in ('c_motion', 'drift', 'gap', 'short', 'tip_motion'):
            history = []
            for i in range(61):
                s = state((ROBOT[0] + (2 if kind == 'tip_motion' and i % 2 else 0),
                           ROBOT[1], ROBOT[2] + (30 if kind == 'c_motion' and i % 2 else 0), 0))
                if kind == 'drift':
                    s['position'] = (s['position'][0] + i * .000002, *s['position'][1:])
                if kind != 'gap' or not 30 < i < 45:
                    history.append((i * .02, response_state_key(s)))
            if kind == 'short':
                history = history[30:]
            with self.assertRaises(ValueError, msg=kind):
                stationary_base_protocol_origin(history, s, 1.21, .5)

    def test_independent_amplitudes_use_base_conversion_and_wall_time(self):
        with patch(
            "robot_control.response_experiments.cartesian_handle_target",
            side_effect=AssertionError("tip inverse forbidden"),
        ):
            r = engine(xyz_amplitudes_um=(10, 6, -4))
            self.assertEqual(r.reference_at(30)["reference_xyz_um"], (10, 6, -4))
            self.assertEqual(r.reference_at(105)["reference_xyz_um"], (0, 0, 0))
            self.assertEqual(r.signal.duration_s, 120)
            for t in np.arange(0, 10, 0.05):
                feed(r, float(t))
                d = r.tick(float(t))
                if d and d["action"] == "send":
                    r.published(d, float(t), 1000 + float(t))
            self.assertGreater(len(r.commands), 3)
            self.assertNotEqual(r.commands[0]["handles"], r.commands[-1]["handles"])
            self.assertEqual(r.status, "running")

    def test_near_rcm_and_bad_amplitudes_reject(self):
        for amplitudes in ((0, 0, 0), (21, 6, 4), (10, float("nan"), 4), (1, 2)):
            with self.assertRaises(ValueError):
                engine(xyz_amplitudes_um=amplitudes)
        s = state((-5, 1466022, 893518, 0))
        with self.assertRaisesRegex(ValueError, "RCM"):
            BasePositionExperiment(ResponseSignal(kind="sine"), s, s["handle"], 0, 1000)
        with self.assertRaises(ValueError):
            ResponseSignal(kind="sine", cycles=1)
        with self.assertRaises(ValueError):
            ResponseSignal(kind="sine", cycles=4, pretrial=True)

    def test_one_packet_per_fresh_feedback_and_last_success_origin(self):
        r = engine()
        for t in np.arange(0, 10, 0.05):
            feed(r, float(t))
            r.tick(float(t))
        d = r.tick(10.1)
        self.assertIsNone(d)
        feed(r, 10.2)
        d = r.tick(10.2)
        self.assertEqual(d["action"], "send")
        r.published(d, 10.2, 1010.2)
        self.assertIsNone(r.tick(10.21))
        self.assertIsNone(r.tick(10.3))
        feed(r, 10.4)
        d2 = r.tick(10.4)
        self.assertEqual(
            tuple(a - b for a, b in zip(d2["handles"], d["handles"])), d2["delta"]
        )
        self.assertEqual(d2["delta"][3], 0)

    def test_dispatch_rejects_fault_timeout_and_stale_origin(self):
        for failure in ("fault", "timeout", "origin"):
            r = engine()
            for t in np.arange(0, 10, 0.05):
                feed(r, float(t))
                r.tick(float(t))
            feed(r, 10)
            d = r.tick(10)
            if failure == "fault":
                r.fault = "new_feedback_fault"
            if failure == "origin":
                r.handles = tuple(x + 1 for x in r.handles)
            self.assertFalse(
                r.validate_dispatch(d, 11 if failure == "timeout" else 10.001)
            )
            self.assertEqual(r.status, "stopped")
            self.assertFalse(r.commands)

    def test_feedback_gap_jump_direction_e_and_nan_stop(self):
        for failure in ("jump", "direction", "e", "nan", "gap"):
            r = engine()
            feed(r, 0)
            r.tick(0)
            s = state()
            if failure == "jump":
                s = state((ROBOT[0] + 30, *ROBOT[1:]))
            if failure == "direction":
                s["forward"] = True
            if failure == "e":
                s["robot"] = (*ROBOT[:3], 1)
            if failure == "nan":
                s["position"] = (float("nan"), 0, 0)
            feed(r, 1 if failure == "gap" else 0.02, s)
            self.assertIsNone(r.tick(1 if failure == "gap" else 0.02))
            self.assertEqual(r.status, "stopped", failure)

    def test_base_correction_respects_tip_packet_limit(self):
        r = engine()
        feed(r, 0)
        handles, delta = r.command_delta(r.latest, (0.02, 0.012, 0.008), {})
        validate_cartesian_delta(ROBOT, delta, 4, -1, max_step_mm=0.05)
        self.assertTrue(any(delta))
        self.assertEqual(tuple(a - b for a, b in zip(handles, r.handles)), delta)

    def test_base_correction_limits_native_handle_step(self):
        r = engine()
        feed(r, 0)
        handles, delta = r.command_delta(r.latest, (0.01, 0.006, 0.004), {})
        self.assertTrue(any(delta))
        self.assertTrue(all(abs(value) <= limit for value, limit in zip(delta, MAX_HANDLE_STEP)))
        self.assertEqual(tuple(a - b for a, b in zip(handles, r.handles)), delta)

    def test_stalled_a_is_not_hidden_by_c_motion(self):
        r = engine(control=ResponseControl(no_progress_s=2))
        for t in np.arange(0, 25, 0.05):
            robot = (ROBOT[0], ROBOT[1], ROBOT[2] + round(float(t) * 20), 0)
            feed(r, float(t), state(robot))
            d = r.tick(float(t))
            if r.status != "running":
                break
            if d and d["action"] == "send":
                r.published(d, float(t), 1000 + float(t))
        self.assertEqual(r.reason, "no_physical_progress_A")

    def test_full_pretrial_recovers_and_exports_without_formal_fit(self):
        r = engine()
        robot = list(ROBOT)
        goal = list(robot)
        for t in np.arange(0, 120.1, 0.02):
            for i, limit in enumerate((8, 500, 50, 0)):
                robot[i] += max(-limit, min(limit, goal[i] - robot[i]))
            feed(r, float(t), state(tuple(robot)))
            d = r.tick(float(t))
            if r.status != "running":
                break
            if d and d["action"] == "send":
                self.assertTrue(r.validate_dispatch(d, float(t)))
                r.published(d, float(t), 1000 + float(t))
                goal[:] = predicted_robot_encoders(robot, d["delta"])
        self.assertEqual(r.status, "completed", r.reason)
        snap = r.snapshot()
        self.assertTrue(snap["final_stable"])
        self.assertFalse(response_metrics(snap)["sine"]["available"])
        self.assertTrue(all(any(c["delta"][i] for c in r.commands) for i in range(3)))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "result"
            export_response(snap, root)
            loaded = load_response(root)
            self.assertEqual(loaded["drive_vector"], list(r.drive_vector))
            self.assertAlmostEqual(
                response_metrics(loaded)["position_3d_rmse_um"],
                response_metrics(snap)["position_3d_rmse_um"],
            )
            self.assertEqual(
                json.loads((root / "metadata.json").read_text())["controller"],
                "base_position_residual",
            )

    def test_zero_sine_output_has_no_phase(self):
        r = BasePositionExperiment(
            ResponseSignal(kind="sine"), state(), (9, 0, 1, 0), 0, 1000, offset_sign=-1
        )
        for t in np.arange(0, 420.1, 0.2):
            feed(r, float(t))
        snap = r.snapshot()
        fit = response_metrics(snap)["sine"]
        self.assertTrue(fit["available"])
        self.assertFalse(fit["phase_available"])
        self.assertIsNone(fit["phase_lag_deg"])

    def test_failed_publish_does_not_advance_origin(self):
        r = engine()
        origin = r.handles
        r.published(dict(handles=(10, 0, 1, 0), delta=(1, 0, 0, 0)), 0, 1000, False)
        self.assertEqual(r.handles, origin)
        self.assertEqual(r.reason, "publish_failed")


if __name__ == "__main__":
    unittest.main()
