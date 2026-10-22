"""Image-only geometry regression: degeneracy, hidden apex and old failure poses."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from rcm_reference import Reference
from rcm_visual_geometry import NeedleGeometry
from rcm_visual_servo import VisualServoSensor


def verify(output):
    output.mkdir(parents=True, exist_ok=True)
    reference = Reference()
    sensor = VisualServoSensor(reference)
    geometry = sensor.geometry
    results = []
    goal = reference.target_task()
    try:
        poses = [('home', reference.home.copy())]
        for fraction in (.2, .4, .6, .7, .8, .9, 1.):
            q = goal.copy()
            q[2] *= fraction
            poses.append((f'approach_{fraction}', q))
        # The old visual PASS pose was approximately 866 um beyond the target.
        poses.append(('previous_false_success', np.array([
            np.radians(4.94351109859499), np.radians(-7.214802584150107), .02056125])))
        for name, q in poses:
            reference.set_task(q)
            observation = sensor.observe()
            expected_tip = reference.site_mm('needle_tip').copy()
            expected_target = np.asarray(reference.metrics()['target_world_mm'])
            tip, quality = geometry.tip(observation)
            target, _ = geometry.target(observation)
            tip_error = float(np.linalg.norm(tip-expected_tip)*1000)
            target_error = float(np.linalg.norm(target-expected_target)*1000)
            assert tip_error < 10., (name, tip_error)
            assert target_error < 5., (name, target_error)
            results.append({'pose': name, 'tip_error_um': tip_error,
                            'target_error_um': target_error, **quality})
            if name == 'home':
                home = copy.deepcopy(observation)
            if name == 'previous_false_success':
                frozen, frozen_tip, frozen_target = observation, tip.copy(), target.copy()
                assert quality['source'] == 'shaft_end_plane'
                assert abs(np.linalg.norm(tip-target)*1000-866.29833945) < 5.

        # A shaft axis without either an end plane or actual apex cannot
        # determine axial translation. Keeping the taper must not invent it.
        missing = copy.deepcopy(home)
        missing['segmentation'][missing['segmentation'][:, :, 0] == geometry.apex_id, 0] = -1
        try:
            geometry.tip(missing)
        except RuntimeError as exc:
            assert 'TIP_NOT_OBSERVABLE' in str(exc), str(exc)
        else:
            raise AssertionError('Uniform shaft plus visible taper invented a physical apex')

        no_shaft = copy.deepcopy(frozen)
        no_shaft['segmentation'][no_shaft['segmentation'][:, :, 0] == geometry.proximal_id, 0] = -1
        try:
            geometry.tip(no_shaft)
        except RuntimeError as exc:
            assert 'SHAFT' in str(exc)
        else:
            raise AssertionError('Missing geometric support was accepted')

        changed = copy.deepcopy(frozen)
        mask = changed['segmentation'][:, :, 0] == geometry.proximal_id
        changed['oct_depth'][mask] += .00002
        try:
            changed_tip, _ = geometry.tip(changed)
            assert np.linalg.norm(changed_tip-frozen_tip)*1000 > 5.
        except RuntimeError:
            pass  # Rejecting an inconsistent changed sensor is also causal.

        # Frozen arrays must suffice after all dynamic truth APIs are poisoned.
        def forbidden(*args, **kwargs):
            raise AssertionError('Dynamic truth used by the geometry estimator')
        reference.metrics = reference.site_mm = reference.target_task = reference.current_task = forbidden
        tip, _ = geometry.tip(frozen)
        target, _ = geometry.target(frozen)
        assert np.array_equal(tip, frozen_tip)
        assert np.array_equal(target, frozen_target)
        result = {'result': 'PASS', 'sampled_poses': len(results),
                  'max_tip_error_um': max(r['tip_error_um'] for r in results),
                  'max_target_error_um': max(r['target_error_um'] for r in results),
                  'uniform_shaft_axial_degeneracy_rejected': True,
                  'missing_shaft_rejected': True, 'depth_changes_estimate_or_is_rejected': True,
                  'dynamic_truth_interfaces_forbidden': True, 'poses': results,
                  'scope': 'Representative ideal RGB-D simulation poses, not exhaustive or hardware accuracy.'}
        (output/'geometry_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        return result
    finally:
        sensor.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('runs/geometry_verification'))
    print(json.dumps(verify(parser.parse_args().output_dir), ensure_ascii=False, indent=2))
