"""Verify paired gates and sampling-only protocol checks without GPU inference."""
import argparse
import copy
import io
import json
from pathlib import Path
import unittest

from scripts.research.clean_oct_sampling_ablation import compare_contract, episode_metrics, paired_metrics
from scripts.research.gpu_episode_pool import record, write_json


class SamplingAblationTests(unittest.TestCase):
    def contracts(self):
        baseline = {'num_action_samples': 1, 'action_aggregation': 'single_sample',
                    'flow_integration_steps': 4, 'control_period_s': .05,
                    'success_error_threshold_mm': .00035}
        return dict(baseline, num_action_samples=9, action_aggregation='median'), baseline

    def test_contract_allows_only_frozen_sampling_change(self):
        candidate, baseline = self.contracts()
        compare_contract(candidate, baseline)
        for key, value in [('control_period_s', .1), ('flow_integration_steps', 8),
                           ('success_error_threshold_mm', .005), ('num_action_samples', 8),
                           ('action_aggregation', 'mean')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare_contract(dict(candidate, **{key: value}), baseline)

    def test_contract_rejects_missing_non_sampling_key(self):
        candidate, baseline = self.contracts()
        del candidate['control_period_s']
        with self.assertRaises(ValueError):
            compare_contract(candidate, baseline)

    def rows(self, value):
        return [{'episode_index': i, 'seed': 42 + i, 'target_id': str(i),
                 'final_error_um': value, 'last200_mean_error_um': value,
                 'absolute_drift_1000_to_exit_um': value} for i in (0, 8, 10)]

    def test_paired_gate_requires_every_metric_improve(self):
        candidate, baseline = self.rows(1.), self.rows(2.)
        self.assertEqual(paired_metrics(candidate, baseline)['candidate_passed'], 1)
        for key in ('final_error_um', 'last200_mean_error_um', 'absolute_drift_1000_to_exit_um'):
            modified = copy.deepcopy(candidate)
            for row in modified:
                row[key] = 2.
            self.assertEqual(paired_metrics(modified, baseline)['candidate_passed'], 0)

    def test_paired_gate_rejects_partial_duplicate_or_mismatched_seed(self):
        candidate, baseline = self.rows(1.), self.rows(2.)
        for modified in (candidate[:2], [candidate[0]] * 3,
                         [dict(r, seed=0) for r in candidate]):
            with self.subTest(rows=modified), self.assertRaises(ValueError):
                paired_metrics(modified, baseline)

    def episode(self, steps=1600, success=False):
        errors = [.01] * steps
        if steps > 1000:
            errors[1000] = .02
        return {'trajectory': {'errors': errors}, 'final_error_mm': .03,
                'episode_index': 0, 'seed': 42, 'target_id': '0', 'steps': steps,
                'success': success, 'safety_limited_steps': 0}

    def test_drift_uses_index1000_and_post_action_final(self):
        metrics = episode_metrics(self.episode())
        self.assertAlmostEqual(metrics['drift_1000_to_exit_um'], 10.)
        self.assertAlmostEqual(metrics['last200_mean_error_um'], 10.)

    def test_early_success_preserved_and_partial_rejected(self):
        metrics = episode_metrics(self.episode(100, True))
        self.assertTrue(metrics['early_strict_success'])
        self.assertFalse(metrics['drift_interval_reached'])
        self.assertEqual(metrics['absolute_drift_1000_to_exit_um'], 0.)
        with self.assertRaises(ValueError):
            episode_metrics(self.episode(100, False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(SamplingAblationTests))
    print(stream.getvalue(), end='')
    (args.output_dir / 'tests.log').write_text(stream.getvalue())
    report = {'status': 'completed' if result.wasSuccessful() else 'failed', 'tests': result.testsRun,
              'errors': len(result.errors), 'failures': len(result.failures),
              'scope': 'CPU synthetic metric/protocol gates; no inference, rendering or rollout'}
    write_json(args.output_dir / 'summary.json', report)
    record('median9_contracts', report['status'], args.output_dir, json.dumps(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
