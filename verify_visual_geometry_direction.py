"""Synthetic regressions for oriented shaft ends and visible sphere roots."""
from __future__ import annotations

import json

import numpy as np

from rcm_visual_geometry import NeedleGeometry, fit_marker_center


class PointCloudGeometry(NeedleGeometry):
    """Supply already backprojected measurements to isolate sign/identity logic."""
    def points(self, observation, mask):
        return observation['points'][np.asarray(mask).ravel()]


def sample(camera_z, *, cap='rear', marker=False, directional=True,
           ambiguous_marker=False):
    z = np.repeat(np.linspace(-16.8, 4.8, 60), 8)
    theta = np.tile(np.linspace(-1.3, 1.3, 8), 60)
    shaft = np.column_stack([.25*np.cos(theta), .25*np.sin(theta), z])
    points = [shaft]
    ids = [np.full(len(shaft), 1)]
    if cap is not None:
        plane = {'rear': -17., 'front': 5., 'ambiguous': -6.}[cap]
        x, y = np.meshgrid(np.linspace(-.17, .17, 17), np.linspace(-.17, .17, 17))
        mask = x*x+y*y < .17**2
        disk = np.column_stack([x[mask], y[mask], np.full(mask.sum(), plane)])
        points.append(disk)
        ids.append(np.full(len(disk), 1))
    if directional:
        distal = np.array([[.02, 0., 5.5], [.02, 0., 7.], [.02, 0., 8.5]])
        points.append(distal)
        ids.append(np.full(len(distal), 2))
    if marker:
        sphere_point = np.array([[0., 0., 9.+np.sign(camera_z)*.02]])
        if ambiguous_marker:
            sphere_point = np.array([[.01, 0., 9.-np.sqrt(.02**2-.01**2)]])
        points.append(sphere_point)
        ids.append(np.full(len(sphere_point), 3))
    points, ids = np.concatenate(points), np.concatenate(ids)
    observation = {'points': points, 'segmentation': np.column_stack([ids, np.zeros(len(ids))])[None]}
    geometry = PointCloudGeometry(
        camera_position_mm=[12., 0., camera_z], camera_rotation=np.eye(3),
        width=len(ids), height=1, focal_px=100., rcm_mm=np.zeros(3),
        proximal_radius_mm=.25, proximal_length_mm=22., distal_length_mm=4.,
        apex_radius_mm=.02, target_radius_mm=.15,
        proximal_id=1, distal_ids=(2,), apex_id=3, target_id=4,
        target_rgb_min=[0, 0, 0], target_rgb_max=[255, 255, 255])
    if ambiguous_marker:
        geometry.camera_position_mm = np.array([40., 0., 9.])
    return geometry, observation


def verify():
    results = []
    for camera_z, end in [(-40., 'rear'), (40., 'front')]:
        geometry, observation = sample(camera_z, cap=end)
        tip, quality = geometry.tip(observation)
        error = float(np.linalg.norm(tip-[0., 0., 9.]))
        assert error < 1e-7, (end, tip)
        assert quality['plane_end'] == end, quality
        expected_source = 'shaft_end_plane' if end == 'rear' else 'shaft_front_plane'
        assert quality['source'] == expected_source, quality
        results.append({'case': end+'_cap', 'error_um': error*1000})
    for camera_z in (-40., 40.):
        geometry, observation = sample(camera_z, cap=None, marker=True)
        tip, quality = geometry.tip(observation)
        error = float(np.linalg.norm(tip-[0., 0., 9.]))
        assert error < 1e-7, (camera_z, tip)
        results.append({'case': f'sphere_camera_z_{camera_z}', 'error_um': error*1000})
    for kwargs, reason in [
            ({'cap': 'rear', 'directional': False}, 'NEEDLE_DIRECTION_NOT_OBSERVABLE'),
            ({'cap': 'ambiguous'}, 'NEEDLE_END_PLANE_NOT_OBSERVABLE'),
            ({'cap': None, 'marker': True, 'ambiguous_marker': True}, 'AMBIGUOUS_APEX_DEPTH')]:
        geometry, observation = sample(-40., **kwargs)
        try:
            geometry.tip(observation)
        except RuntimeError as exc:
            assert str(exc) == reason, (kwargs, str(exc))
            results.append({'case': reason, 'result': 'rejected'})
        else:
            raise AssertionError(f'Unobservable geometry accepted: {kwargs}')
    theta = np.linspace(0., 2.*np.pi, 8, endpoint=False)
    # The dominant sphere inliers form a planar circle, so refitting the center
    # is rank deficient even when a mixed RANSAC sample supplied a 3D candidate.
    sphere = np.vstack([np.column_stack([np.cos(theta), np.sin(theta), np.zeros(8)]),
                        [0., 0., 2.], [0., 0., -2.]])
    try:
        fit_marker_center(sphere, np.zeros_like(sphere), np.ones(len(sphere)))
    except RuntimeError as exc:
        assert str(exc) == 'ILL_CONDITIONED_VISUAL_DEPTH', str(exc)
        results.append({'case': 'rank_deficient_sphere_refit', 'result': 'rejected'})
    else:
        raise AssertionError('Rank-deficient sphere inliers accepted')
    return {'result': 'PASS', 'case_count': len(results), 'cases': results}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
