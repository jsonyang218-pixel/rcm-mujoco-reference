#!/usr/bin/env python3
"""Protocol, safety and analytic response regressions for direct sine input."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robot_control.kinematics import model_tip_from_robot_encoders, predicted_robot_encoders
from robot_control.response_experiments import ResponseControl
from robot_control.response_signals import WireSineSignal
from robot_control.wire_sine import WireSineExperiment
from robot_control.wire_sine_reporting import (
    actual_inputs, build_wire_figures, export_wire_sine, wire_metrics,
)


def state(robot=(0, 1466021, 893518, 0), handles=(61, 12, 81, 0)):
    return dict(robot=robot, handle=handles, position=model_tip_from_robot_encoders(robot, 0),
                forward=False, backward=False)


def experiment(**kwargs):
    return WireSineExperiment(WireSineSignal(frequency_hz=.05), state(), (61, 12, 81, 0), 0., 1000., **kwargs)


class WireSineTests(unittest.TestCase):
    def test_signal_time_boundaries_and_invalid_configuration(self):
        signal = WireSineSignal(frequency_hz=.01)
        self.assertEqual(signal.duration_s, 420)
        self.assertAlmostEqual(signal.at(30)[0], 1)
        self.assertEqual(signal.at(405), (0., 'recovery'))
        for values in (dict(cycles=2), dict(frequency_hz=1), dict(amplitude_um=10)):
            with self.assertRaises(ValueError):
                WireSineSignal(**values)
        self.assertEqual(WireSineSignal(cycles=1).duration_s, 120)

    def test_nonzero_origin_difference_no_feedback_error_accumulation(self):
        run = experiment()
        run.observe(state(), 1010, 10, 1)
        first = run.tick(10)
        self.assertEqual(first['handles'], (64, 15, 89, 0))
        self.assertEqual(first['delta'], (3, 3, 8, 0))
        run.published(first, 10, 1010)
        # Stationary physical feedback and stale handle echo never add another sine offset.
        run.observe(state(), 1010.1, 10.1, 2)
        second = run.tick(10.1)
        self.assertEqual(second['action'], 'quantized_hold')
        self.assertEqual(second['delta'], (0, 0, 0, 0))
        self.assertIsNone(run.tick(10.2))

    def test_current_elapsed_reference_skips_old_samples_and_validates_wire_difference(self):
        run = experiment()
        run.observe(state(), 1010, 10, 1)
        first = run.tick(10)
        run.published(first, 10, 1010)
        # Keep feedback fresh while UI ticks are absent for half a cycle.
        for sequence, t in enumerate(np.arange(10.1, 20., .1), 2):
            run.observe(state(), 1000 + t, t, sequence)
        decision = run.tick(20)
        self.assertEqual(decision['handles'], (58, 9, 73, 0))
        self.assertEqual(decision['delta'], (-6, -6, -16, 0))

    def test_publish_failure_stops_and_keeps_last_success(self):
        run = experiment()
        run.observe(state(), 1010, 10, 1)
        decision = run.tick(10)
        run.published(decision, 10, 1010, success=False)
        self.assertEqual(run.handles, (61, 12, 81, 0))
        self.assertEqual(run.reason, 'publish_failed')
        self.assertIsNone(run.tick(10.1))

    def test_fault_or_stale_feedback_between_decision_and_publish_rejects_dispatch(self):
        for invalid in ('fault', 'timeout', 'origin'):
            run = experiment()
            run.observe(state(), 1010, 10, 1)
            decision = run.tick(10)
            self.assertTrue(run.validate_dispatch(decision, 10))
            if invalid == 'fault':
                run.observe(state((0, 1466021, 893518, 1)), 1010.01, 10.01, 2)
            elif invalid == 'origin':
                run.handles = (62, 12, 81, 0)
            now = 12 if invalid == 'timeout' else 10.02
            self.assertFalse(run.validate_dispatch(decision, now))
            self.assertEqual(run.status, 'stopped')
            self.assertEqual(run.commands, [])

    def test_watchdog_interval_manual_stop_and_gap_never_resume(self):
        for reason in ('feedback_timeout', 'operator_stop', 'feedback_gap_exceeded'):
            with self.subTest(reason=reason):
                run = experiment(control=ResponseControl(state_timeout_s=.5))
                run.observe(state(), 1000, 0, 1)
                run.tick(0)
                if reason == 'operator_stop':
                    run.stop(reason, .1)
                elif reason == 'feedback_gap_exceeded':
                    run.observe(state(), 1001, 1, 2)
                run.tick(1)
                self.assertEqual(run.reason, reason)
                run.observe(state(), 1001.01, 1.01, 3)
                self.assertIsNone(run.tick(1.01))

    def test_limits_nan_e_direction_jump_and_envelopes(self):
        bad_states = []
        bad = state(); bad['position'] = (float('nan'), 0, 0); bad_states.append((bad, 'invalid_feedback'))
        bad = state(); bad['forward'] = True; bad_states.append((bad, 'unexpected_direction_flag'))
        bad_states.append((state((0, 1466021, 893518, 1)), 'unexpected_e_motion'))
        bad = state(); bad['position'] = (1, 0, 0); bad_states.append((bad, 'base_envelope'))
        bad_states.append((state((1000, 1466021, 893518, 0)), 'joint_A_envelope'))
        for bad, reason in bad_states:
            with self.subTest(reason=reason):
                run = experiment()
                run.observe(bad, 1000, 0, 1)
                run.tick(0)
                self.assertIn(reason, run.reason)
        run = experiment()
        run.observe(state(), 1000, 0, 1)
        run.observe(state((20, 1466021, 893518, 0)), 1000.1, .1, 2)
        run.tick(.1)
        self.assertEqual(run.reason, 'joint_A_jump')
        for amplitudes in ((7, 0, 0), (0, 0, 18), (0, 0, 0), (1.5, 1, 8)):
            with self.assertRaises(ValueError):
                experiment(amplitudes=amplitudes)
        with self.assertRaises(ValueError):
            WireSineExperiment(WireSineSignal(), state((0, -1, 893518, 0)), (0, 0, 0, 0), 0, 0)
        with self.assertRaises(ValueError):
            WireSineExperiment(WireSineSignal(), state(), (2**31 - 1, 12, 81, 0), 0, 0)

    def test_sustained_excursion_without_motion_stops_even_across_plateaus(self):
        run = experiment(amplitudes=(3, 0, 0))
        for i in range(1800):
            t = i * .05
            run.observe(state(), 1000 + t, t, i)
            decision = run.tick(t)
            if decision and decision['action'] == 'send':
                run.published(decision, t, 1000 + t)
            if run.status != 'running':
                break
        self.assertEqual(run.reason, 'no_physical_progress_A')
        self.assertLess(run.stopped_mono, 30)

    def test_full_model_run_restores_wire_origin_and_requires_settle(self):
        run = experiment()
        robot = run.start_robot
        for i in range(5001):
            t = i * .02
            run.observe(state(robot), 1000 + t, t, i)
            decision = run.tick(t)
            if decision and decision['action'] == 'send':
                run.published(decision, t, 1000 + t)
                offset = tuple(a - b for a, b in zip(run.handles, run.origin_handles))
                robot = predicted_robot_encoders(run.start_robot, offset)
            if run.status != 'running':
                break
        self.assertEqual(run.status, 'completed')
        self.assertEqual(run.handles, run.origin_handles)
        self.assertTrue(run.final_stable)
        # Every packet equals its actual change from the last successful packet.
        previous = run.origin_handles
        for command in run.commands:
            self.assertEqual(tuple(a - b for a, b in zip(command['handles'], previous)), command['delta'])
            previous = command['handles']
        self.assertTrue(all(b['publish_mono'] - a['publish_mono'] >= .05 - 1e-9
                            for a, b in zip(run.commands, run.commands[1:])))
        self.assertGreaterEqual(wire_metrics(run.snapshot())['fitted_cycles'], 3)


class WireReportingTests(unittest.TestCase):
    def sample(self):
        run = experiment()
        snap = run.snapshot()
        snap['metadata'] = dict(data_source='simulation')
        snap['status'], snap['reason'] = 'completed', 'analytic fixture'
        t = np.arange(0, 100.001, .05)
        omega = 2 * np.pi * .05
        signal = run.signal
        snap['commands'] = [dict(publish_elapsed_s=float(x), publish_success=True, action='send',
                                 handles=(61 + round(3 * signal.at(x)[0]), 12 + round(3 * signal.at(x)[0]),
                                          81 + round(8 * signal.at(x)[0]), 0), delta=(0, 0, 0, 0)) for x in t]
        previous = tuple(snap['origin_handles'])
        for command in snap['commands']:
            command['delta'] = tuple(a - b for a, b in zip(command['handles'], previous))
            previous = command['handles']
        snap['feedback'] = [dict(elapsed_s=float(x), actual_xyz_um=(2 * np.sin(omega * (x - 5 - 2)),
                                                                      .4 * np.sin(omega * (x - 5 - 2)), 0.),
                                 joint_actual=(.01 * np.sin(omega * (x - 5)), .01 * np.sin(omega * (x - 5)),
                                               2 * np.sin(omega * (x - 5)))) for x in t]
        return snap

    def test_actual_publish_hold_failure_and_equal_timestamp_order(self):
        snap = self.sample()
        snap['commands'] = [dict(publish_elapsed_s=2., handles=(62, 12, 81, 0), publish_success=False),
                            dict(publish_elapsed_s=3., handles=(63, 12, 81, 0), publish_success=True)]
        np.testing.assert_array_equal(actual_inputs(snap, [1, 2, 3, 4])[:, 0], [0, 0, 2, 2])

    def test_analytic_gain_phase_and_missing_cycles_not_fabricated(self):
        snap = self.sample()
        metrics = wire_metrics(snap)
        self.assertEqual(metrics['fitted_cycles'], 3)
        self.assertAlmostEqual(metrics['base']['X']['gain_per_handle_count'], .25, delta=.004)
        self.assertAlmostEqual(metrics['base']['X']['lag_deg'], 36, delta=1)
        self.assertAlmostEqual(metrics['base']['X']['equivalent_lag_s'], 2, delta=.06)
        snap['feedback'] = snap['feedback'][:1000]
        self.assertEqual(wire_metrics(snap)['fitted_cycles'], 0)
        self.assertEqual(wire_metrics(snap)['base'], {})
        snap = self.sample()
        snap['feedback'] = [r for r in snap['feedback'] if not 40 < r['elapsed_s'] < 42]
        self.assertEqual(wire_metrics(snap)['fitted_cycles'], 0)

    def test_raw_export_and_all_four_figures_render_headless(self):
        snap = self.sample()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'result'
            export_wire_sine(snap, root)
            loaded = json.loads((root / 'snapshot.json').read_text())
            self.assertEqual(loaded['feedback'], json.loads(json.dumps(snap['feedback'])))
            self.assertTrue((root / 'commands.csv').is_file())
            figures = build_wire_figures(snap)
            self.assertEqual(len(figures), 4)
            for name, figure in figures.items():
                figure.canvas.draw()
                figure.savefig(str(root / (name + '.png')))
                self.assertGreater((root / (name + '.png')).stat().st_size, 1000)
                figure.clear()
            with self.assertRaises(FileExistsError):
                export_wire_sine(snap, root)


if __name__ == '__main__':
    unittest.main()
