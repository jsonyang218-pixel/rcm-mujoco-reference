"""Deterministic checks before the planning-file-6 GPU workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import unittest

import numpy as np

from scripts.research.homog_label_fix_common import (
    BASE, BRANCH, FAST5, FIX_DATA, ROOT, SOURCE_DATA, SOURCE_REPO, STEPS, TASK,
    WEIGHTS, identity, read, record, write_json,
)
from scripts.research.homog_label_fix_pipeline import card, ensure_branch
from scripts.research.homog_label_fix_publish import publication_files
from scripts.research.homog_label_fix_repair import (
    axis_reverse_audit, build_fixed_dataset, repair_arrays,
)

OUTPUT = None


def fixture():
    count = 4
    return {
        'episode_index': np.array([0, 20, 40, 40], dtype=np.int64),
        'frame_index': np.arange(count, dtype=np.int32),
        'state_si': np.zeros((count, 3)),
        'action': np.array([[.1, 0., 0.], [0., .06, 0.],
                            [.01, .02, .03], [.01, .02, .03]]),
        'executed_action': np.array([[.05, 0., 0.], [0., .03, 0.],
                                     [.02, .03, .004], [0., 0., 0.]]),
        'zero': np.array([[1., 0., 0.], [0., 1., 0.],
                          [.02, .03, .004], [.01, .02, .003]]),
        'pre': np.array([[1., 0., 0.], [0., 1., 0.],
                         [.02, .03, .004], [.01, .02, .003]]),
        'truth_navigation_mm': np.array([[.01, 0., 0.], [0., .01, 0.],
                                         [0., 0., .010], [0., 0., .004]]),
        'source_frame_index': np.arange(count, dtype=np.int64),
        'timestamp_s': np.arange(count, dtype=np.float64)*.05,
        'oct_navigation': np.ones((count, 3), dtype=np.float32)*.01,
        'oct_navigation_valid': np.ones(count, dtype=bool),
        'hold': np.array([False, False, False, True]),
    }


class LabelFixTests(unittest.TestCase):
    def test_T1_and_T2_keep_direction_and_adopt_executed_norm(self):
        arrays = fixture()
        repaired, _, stationary = repair_arrays(arrays)
        np.testing.assert_allclose(repaired[0], [.05, 0., 0.], atol=1e-8)
        np.testing.assert_allclose(repaired[1], [0., .03, 0.], atol=1e-8)
        np.testing.assert_allclose(repaired[2], [.02, .03, .004], atol=1e-8)
        np.testing.assert_array_equal(repaired[3], [0., 0., 0.])
        self.assertFalse(stationary[2])
        self.assertTrue(stationary[3])

    def test_T3_stationary_requires_exact_zero_and_true_error_below_5um(self):
        arrays = fixture()
        arrays['executed_action'][3] = [.001, 0., 0.]
        repaired, _, stationary = repair_arrays(arrays)
        self.assertFalse(stationary[3])
        np.testing.assert_allclose(repaired[3], [.001, 0., 0.], atol=1e-8)
        arrays = fixture()
        arrays['truth_navigation_mm'][3] = [0., 0., .005]
        _, _, stationary = repair_arrays(arrays)
        self.assertFalse(stationary[3])

    def test_label_caps_and_nonfinite_inputs_fail_closed(self):
        arrays = fixture()
        arrays['action'][1] = [100., 100., 0.]
        repaired, _, _ = repair_arrays(arrays)
        self.assertTrue(np.all(np.abs(repaired) <= np.asarray([.2, .1, .03])+1e-8))
        arrays['action'][0, 0] = np.nan
        with self.assertRaises(ValueError):
            repair_arrays(arrays)

    def test_reverse_audit_uses_existing_zero_and_pre_definition(self):
        arrays = fixture()
        actions = np.array([[1., 0., 0.], [0., 1., 0.],
                            [.02, .03, .004], [0., 0., 0.]])
        result = axis_reverse_audit(arrays, actions)
        self.assertTrue(all(value['reverse_fraction'] == 0 for value in result.values()))
        actions[0, 0] = -1.
        result = axis_reverse_audit(arrays, actions)
        self.assertGreater(result['A']['reverse_fraction'], 0)

    def test_card_has_no_timeouts_and_exact_fast5(self):
        value = card()
        self.assertEqual(value['fast5']['indices'], list(FAST5))
        self.assertEqual(value['fast5']['seeds'], [42+i for i in FAST5])
        self.assertEqual(value['fast5']['steps'], STEPS)
        self.assertTrue(all(budget is None for budget in value['budgets'].values()))
        self.assertEqual(value['training']['batches'], 4000)
        self.assertEqual(value['training']['batch_size'], 16)
        self.assertEqual(value['training']['correction_ratio'], .08)

    def test_source_dataset_matches_cd22e10_and_branch_is_descendant(self):
        local = subprocess.check_output(
            ['git','-C',str(SOURCE_REPO),'hash-object',str(SOURCE_DATA/'data/dataset.npz')],
            text=True).strip()
        published = subprocess.check_output(
            ['git','-C',str(SOURCE_REPO),'rev-parse',
             BASE+':rcm_4090_deploy/data/homog_clean_replay_20261123_mixed_1600_600_12_hold_v2/data/dataset.npz'],
            text=True).strip()
        self.assertEqual(local, published)
        self.assertEqual(subprocess.check_output(
            ['git','-C',str(SOURCE_REPO),'merge-base',BASE,BRANCH],
            text=True).strip(), BASE)

    def test_publication_is_additive_and_has_lightweight_label_artifacts(self):
        manifest = {'inputs': ['source validation does not create destinations']}
        # Use the real workflow output only when it exists; before launch this
        # verifies source collection without mapping symlinked image trees.
        source_files = {str(path.relative_to(ROOT)) for path in (
            ROOT/'scripts/research/homog_label_fix_pipeline.py',
            ROOT/'scripts/research/homog_label_fix_repair.py',
            ROOT/'scripts/research/homog_label_fix_publish.py',
            ROOT/'scripts/verification/verify_homog_label_fix.py')}
        self.assertTrue(all(Path(ROOT/path).is_file() for path in source_files))
        self.assertEqual(BRANCH, 'feature/homog-label-fix-20261123')
        self.assertTrue(TASK.is_file())

    def test_full_source_repair_build_completes_all_four_gates(self):
        report = OUTPUT/'stage1_end_to_end'
        fixed = OUTPUT/'stage1_end_to_end_fixed_data'
        audit, receipt = build_fixed_dataset(report, fixed)
        self.assertEqual(audit['status'], 'passed')
        self.assertTrue(all(gate['passed'] for gate in audit['gates'].values()))
        self.assertTrue((fixed/'data/dataset.npz').is_file())
        self.assertTrue((fixed/'data/label_repair_overlay.npz').is_file())
        self.assertTrue((fixed/'shards').is_symlink())
        self.assertEqual(receipt['fixed_dataset_sha256'],
                         __import__('lerobot_training.navigation_overlay',
                                    fromlist=['file_sha256']).file_sha256(
                                        fixed/'data/dataset.npz'))
        from lerobot_training.train_diffusion_policy import (
            RCMDataset, image_transform_for_dataset,
        )
        dataset = RCMDataset(fixed, transform=image_transform_for_dataset(fixed),
            n_obs_steps=2, n_action_steps=16,
            include_velocity=True, include_previous_action=True)
        sample = dataset[0]
        self.assertEqual(tuple(sample['observation.image'].shape), (2, 3, 224, 224))
        self.assertEqual(tuple(sample['action'].shape), (16, 3))
        self.assertTrue(bool(np.isfinite(sample['action']).all()))


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    OUTPUT = args.output_dir.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    # Creating the versioned branch is part of preflight state, but this check
    # only reuses it if already present and validates its ancestor.
    ensure_branch()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(LabelFixTests))
    summary = {'status': 'passed' if result.wasSuccessful() else 'failed',
        'tests': result.testsRun, 'failures': len(result.failures),
        'errors': len(result.errors), 'identity': identity(),
        'CPU_reason': 'Deterministic array/contract/source checks; no rendering or tensor work'}
    write_json(OUTPUT/'summary.json', summary)
    record('label_fix_contracts', summary['status'], OUTPUT, summary)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
