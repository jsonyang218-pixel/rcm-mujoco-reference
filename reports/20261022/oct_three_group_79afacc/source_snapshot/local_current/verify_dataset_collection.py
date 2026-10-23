"""Fast collection regressions; no simulator rendering or navigation is run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import random_fundus_dataset as collection
from rcm_waypoint_dataset import (EpisodeWriter, NO_IMAGE_INPUT_FIELDS,
                                  PERCEPTION_INTEGER_FIELDS, PERCEPTION_NUMERIC_FIELDS,
                                  PERCEPTION_STRING_FIELDS,
                                  POLICY_INPUT_FIELDS, _normalize_perception, write_manifest)


class CollectionIntegrityTests(unittest.TestCase):
    def collect_fixture(self, root, reports, episodes, *, returncode=0,
                        methods=('visual_servo', 'RRT', 'PRM'), seeds=(7, 17, 27)):
        episode = root/'episode'
        config = root/'frozen.json'
        config.write_text('{}\n')

        def fake_process(command, **kwargs):
            episode.mkdir()
            (episode/'dataset').mkdir()
            (episode/'run_reports.json').write_text(json.dumps(reports))
            (episode/'dataset/manifest.json').write_text(json.dumps({'episodes': episodes}))
            self.assertEqual(Path(command[command.index('--config')+1]), config)
            return SimpleNamespace(returncode=returncode)

        with patch.object(collection.subprocess, 'run', side_effect=fake_process):
            return collection._collect_episode(root/'targets.json', episode, list(methods), list(seeds),
                                               'precision', config, 20., 'visual_current_target')

    def test_multiseed_capture_checks_all_expected_method_seed_pairs(self):
        runs = [{'method': method, 'seed': seed, 'result': 'PASS'}
                for method in ('visual_servo', 'RRT', 'PRM')
                for seed in ([None] if method == 'visual_servo' else [7, 17, 27])]
        cases = [('complete', runs, runs, True),
                 ('missing_report', runs[:-1], runs, False),
                 ('missing_manifest_episode', runs, runs[:-1], False),
                 ('duplicate_manifest_identity', runs, runs[:-1]+[runs[0]], False),
                 ('duplicate_report_identity', runs[:-1]+[runs[0]], runs, False),
                 ('wrong_manifest_seed', runs, runs[:-1]+[{**runs[-1], 'seed': 99}], False),
                 ('wrong_report_method', runs[:-1]+[{**runs[-1], 'method': 'other'}], runs, False)]
        for name, reports, episodes, expected in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix='rcm_collection_') as temporary:
                result = self.collect_fixture(Path(temporary), reports, episodes)
                self.assertEqual(result['capture_complete'], expected)
                self.assertEqual(result['expected_runs'], 7)
                self.assertEqual(result['manifest_episodes'], len(episodes))

    def test_visual_only_runs_once_and_complete_failure_capture_is_not_navigation_success(self):
        for methods, seeds, runs in [
                (('visual_servo',), (7, 17, 27), [{'method': 'visual_servo', 'seed': None, 'result': 'FAIL'}]),
                (('RRT',), (7, 17), [{'method': 'RRT', 'seed': seed, 'result': 'FAIL'} for seed in (7, 17)])]:
            with self.subTest(methods=methods), tempfile.TemporaryDirectory(prefix='rcm_capture_failure_') as temporary:
                result = self.collect_fixture(Path(temporary), runs, runs, returncode=1,
                                              methods=methods, seeds=seeds)
                self.assertTrue(result['capture_complete'])
                self.assertEqual(result['exit_code'], 1)
                self.assertEqual(result['expected_runs'], len(runs))
                self.assertTrue(all(run['result'] == 'FAIL' for run in result['reports']))

    def test_main_hashes_and_collects_frozen_config_after_original_changes(self):
        with tempfile.TemporaryDirectory(prefix='rcm_frozen_config_') as temporary:
            root = Path(temporary)
            original = root/'original.json'
            original_bytes = b'{"configuration_version": "before-generation"}\n'
            original.write_bytes(original_bytes)
            output = root/'collected'
            seen = []

            def write_targets(reference, path, seed, count, separation):
                payload = {'frame': 'world', 'targets': [
                    {'id': f'P{index}', 'point_mm': [float(index), 0., 20.]} for index in range(count)]}
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(payload))
                # Mutation between the initial read and child collection must
                # neither change the frozen hash nor leak into child commands.
                original.write_text('{"configuration_version":"changed-later"}\n')
                return payload, {'result': 'PASS'}

            def collect_episode(*args, **kwargs):
                frozen = Path(args[5])
                self.assertEqual(frozen, output/'collection_config.json')
                self.assertEqual(frozen.read_bytes(), original_bytes)
                self.assertNotEqual(original.read_bytes(), original_bytes)
                self.assertTrue(kwargs['strict_visual'])
                seen.append(frozen)
                return {'exit_code': 0, 'capture_complete': True, 'reports': [
                    {'method': 'visual_servo', 'seed': None, 'result': 'PASS'}]}

            with (patch.object(collection, 'Reference', return_value=object()),
                  patch.object(collection, '_write_targets', side_effect=write_targets),
                  patch.object(collection, '_collect_episode', side_effect=collect_episode)):
                result = collection.main(['--config', str(original), '--output-dir', str(output),
                                          '--episodes', '1', '0', '0', '--targets-per-episode', '3',
                                          '--methods', 'visual_servo', '--strict-visual', '--collect'])
            self.assertEqual(result, 0)
            self.assertEqual(len(seen), 1)
            manifest = json.loads((output/'scenario_manifest.json').read_text())
            index = json.loads((output/'dataset_index.json').read_text())
            digest = hashlib.sha256(original_bytes).hexdigest()
            self.assertEqual(manifest['config_sha256'], digest)
            self.assertEqual(index['config_sha256'], digest)
            self.assertEqual(manifest['config_snapshot'], 'collection_config.json')
            self.assertIn('set of 3', index['research_scope']['multi_target'])
            self.assertIn('given input order', index['research_scope']['multi_target'])
            self.assertIn('target set', index['research_scope']['split_unit'])

    def test_strict_visual_still_requires_visual_method_and_current_target_task(self):
        for options in [('--methods', 'RRT'),
                        ('--methods', 'visual_servo', '--task-regime', 'registered_coordinate_prior')]:
            with self.subTest(options=options), patch.object(collection, 'Reference') as reference:
                with self.assertRaises(SystemExit) as error:
                    collection.main(['--strict-visual', '--targets-per-episode', '3', *options])
                self.assertEqual(error.exception.code, 2)
                reference.assert_not_called()


class PerceptionAuditTests(unittest.TestCase):
    def test_optional_provenance_roundtrips_but_never_becomes_a_policy_input(self):
        with tempfile.TemporaryDirectory(prefix='rcm_perception_audit_') as temporary:
            root = Path(temporary)/'dataset'
            writer = EpisodeWriter(root, 0, '当前目标', 'visual_servo', None, ['P1'])
            rgb = np.zeros((2, 3, 3), dtype=np.uint8)
            states = [np.array([0., 0., 0.]), np.array([.1, -.2, .3]), np.array([.2, -.3, .4])]
            observed = {'target_source': 'current_rgbd', 'tip_source': 'shaft_end_plane',
                        'controller_state': 'APPROACH', 'cached_target_age_s': 0.,
                        'estimated_position_error_um': 8., 'true_position_error_um': 7.5,
                        'target_observed': True, 'control_frame_index': np.int64(42),
                        'occlusion_event_id': 'event_0001'}
            cached = {'target_source': 'static_target_memory', 'tip_source': 'shaft_end_plane',
                      'controller_state': 'APPROACH', 'cached_target_age_s': .05,
                      'estimated_position_error_um': 5., 'true_position_error_um': 4.5,
                      'target_observed': False, 'control_frame_index': 44,
                      'occlusion_event_id': 'event_0002'}
            writer.add(0., states[0], states[1], 'P1', rgb, perception=observed)
            writer.add(.05, states[1], states[2], 'P1', rgb, perception=MappingProxyType(cached))
            writer.add(.1, states[2], states[2], 'P1', rgb, terminal=True, phase='hold')
            observed['target_source'] = 'changed_outside_writer'
            episode = writer.finish('PASS')
            manifest = write_manifest(root, [episode], [2, 3, 3])
            records = [json.loads(line) for line in (root/episode['jsonl']).read_text().splitlines()]
            self.assertEqual(records[0]['perception']['target_source'], 'current_rgbd')
            self.assertEqual(records[1]['perception'], cached)
            self.assertEqual(records[2]['perception']['target_source'], 'unknown')
            self.assertEqual(records[2]['perception']['occlusion_event_id'], 'unknown')
            self.assertIsNone(records[2]['perception']['control_frame_index'])
            self.assertFalse(records[2]['perception']['target_observed'])
            with np.load(root/episode['data'], allow_pickle=False) as data:
                for field in PERCEPTION_STRING_FIELDS:
                    self.assertEqual(data[f'perception.{field}'].tolist(), [r['perception'][field] for r in records])
                for field in PERCEPTION_NUMERIC_FIELDS:
                    expected = [-1. if r['perception'][field] is None else r['perception'][field] for r in records]
                    np.testing.assert_array_equal(data[f'perception.{field}'], expected)
                    self.assertEqual(data[f'perception.{field}'].dtype, np.float64)
                for field in PERCEPTION_INTEGER_FIELDS:
                    expected = [-1 if r['perception'][field] is None else r['perception'][field] for r in records]
                    np.testing.assert_array_equal(data[f'perception.{field}'], expected)
                    self.assertEqual(data[f'perception.{field}'].dtype, np.int64)
                self.assertEqual(data['perception.target_observed'].dtype, np.bool_)
                self.assertEqual(data['perception.target_observed'].tolist(), [True, False, False])
                np.testing.assert_array_equal(data['action'][:-1], data['observation.state'][1:])
            for name, fields in [('policy_inputs', POLICY_INPUT_FIELDS),
                                 ('policy_inputs_no_image', NO_IMAGE_INPUT_FIELDS),
                                 ('action_labels', ('action',))]:
                with np.load(root/episode[name], allow_pickle=False) as bundle:
                    self.assertEqual(set(bundle.files), set(fields))
                    self.assertFalse(any(field.startswith('perception.') for field in bundle.files))
            self.assertIn('perception.*', manifest['policy_interface']['excluded_from_policy_inputs'])
            self.assertEqual(manifest['fields']['perception.control_frame_index']['dtype'], 'int64')
            self.assertEqual(manifest['fields']['perception.control_frame_index']['unknown_sentinel'], -1)
            self.assertEqual(manifest['fields']['perception.occlusion_event_id']['unknown_sentinel'], 'unknown')
            self.assertIn('stationary', manifest['target_memory_scope'])
            self.assertIn('does not validate', manifest['target_memory_scope'])

    def test_invalid_audit_numbers_and_flags_are_rejected(self):
        cases = [{'cached_target_age_s': -.01}, {'cached_target_age_s': float('nan')},
                 {'estimated_position_error_um': float('inf')}, {'true_position_error_um': -2.},
                 {'target_observed': 'false'}, {'tip_source': None}, {'occlusion_event_id': None},
                 {'control_frame_index': -1}, {'control_frame_index': .5},
                 {'control_frame_index': True}, {'control_frame_index': np.bool_(False)},
                 {'control_frame_index': 2**63}, []]
        for metadata in cases:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    _normalize_perception(metadata)


if __name__ == '__main__':
    unittest.main(verbosity=2)
