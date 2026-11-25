"""CPU contract checks; no rendering, inference, training or rollout."""
import argparse
import io
import json
from pathlib import Path
import unittest

import numpy as np

from scripts.research.clean_oct_drift_diagnosis import (
    UNIT, reconstruction_check, segment_statistics,
)
from scripts.research.gpu_episode_pool import record, write_json


class DiagnosisTests(unittest.TestCase):
    def episode(self):
        accepted = np.array([[.002, -.004, .002], [-.002, .004, -.002], [0., 0., 0.]])
        before = np.array([10., 20., 4.])
        actual = np.array([[.001, -.002, .001], [0., .002, -.001], [0., 0., 0.]])
        records = []
        for command, movement in zip(accepted, actual):
            records.append({
                'actual_before_task_rad_rad_m': (before * UNIT).tolist(),
                'actual_after_task_rad_rad_m': ((before + movement) * UNIT).tolist(),
                'requested_task_rad_rad_m': ((before + command) * UNIT).tolist(),
                'commanded_task_rad_rad_m': ((before + command / 2) * UNIT).tolist(),
            })
        return {'trajectory': {'raw_actions': accepted.tolist(), 'actions': accepted.tolist(),
                               'velocities': np.zeros((3, 3)).tolist()}, 'motor_records': records}

    def test_conversion_stages_and_units(self):
        report = segment_statistics(self.episode(), 0, 3)
        np.testing.assert_allclose(report['statistics']['actual_delta']['sum'], [.001, 0., 0.], atol=1e-12)
        np.testing.assert_allclose(report['requested_minus_accepted_max_abs'], 0., atol=1e-12)
        np.testing.assert_allclose(report['statistics']['requested_delta']['sum'], 0., atol=1e-12)

    def test_sign_conditioning_and_zero_response(self):
        report = segment_statistics(self.episode(), 0, 3)
        a = report['conditioned']['A_deg']
        self.assertEqual(a['active_no_motion_fraction'], .5)
        self.assertEqual(a['bins']['negative_abs_0.001_0.003']['actual_zero_fraction'], 1.)
        self.assertEqual(a['bins']['zero_abs_0_0.001']['count'], 1)

    def test_segment_boundaries_exclude_outside_actions(self):
        report = segment_statistics(self.episode(), 1, 2)
        np.testing.assert_allclose(report['statistics']['raw']['sum'], [-.002, .004, -.002])
        self.assertEqual((report['start'], report['end_exclusive']), (1, 2))

    def test_incomplete_and_nonfinite_segments_fail(self):
        with self.assertRaises(ValueError):
            segment_statistics(self.episode(), 0, 4)
        episode = self.episode()
        episode['trajectory']['velocities'][1][0] = float('nan')
        with self.assertRaises(ValueError):
            segment_statistics(episode, 0, 3)

    def test_reconstruction_accepts_registered_tolerance(self):
        report = reconstruction_check(np.full((2, 3), 1e-8), np.zeros((2, 3)), [1e-8] * 3, np.zeros(3))
        self.assertTrue(report['passed'])
        self.assertIn('pixel equality is not claimed', report['rgb_limitation'])

    def test_reconstruction_rejects_oct_action_and_nonfinite(self):
        for oct_value, action in ((1e-4, 0.), (0., 1e-4), (float('nan'), 0.)):
            with self.subTest(oct=oct_value, action=action), self.assertRaises(ValueError):
                reconstruction_check(np.full((2, 3), oct_value), np.zeros((2, 3)), [action] * 3, np.zeros(3))

    def test_reconstruction_rejects_broadcastable_missing_frame(self):
        with self.assertRaises(ValueError):
            reconstruction_check(np.zeros(3), np.zeros((2, 3)), np.zeros(3), np.zeros(3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(DiagnosisTests))
    text = stream.getvalue()
    print(text, end='')
    (args.output_dir / 'tests.log').write_text(text)
    report = {'status': 'completed' if result.wasSuccessful() else 'failed',
              'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'scope': 'CPU synthetic contracts only; no experiment inference or rendering'}
    write_json(args.output_dir / 'summary.json', report)
    record('drift_diagnosis_contracts', report['status'], args.output_dir, json.dumps(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
