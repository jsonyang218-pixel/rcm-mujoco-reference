"""Calibrated RGB-D geometry; dynamic robot/target truth is never an input.

The fixed camera, RCM and needle dimensions are calibration inputs. Needle
orientation comes from visible shaft surface points. Axial position comes
from a visible shaft end plane, or the physical apex sphere. A uniform
shaft alone cannot constrain insertion and is explicitly rejected.
"""
from __future__ import annotations

import math

import numpy as np


def fit_marker_center(points_mm, offsets_mm, radii_mm):
    """Robust sphere-center fit with calibrated radii and conditioning checks."""
    points, offsets, radii = map(np.asarray, (points_mm, offsets_mm, radii_mm))
    if len(points) < 6:
        raise RuntimeError("INSUFFICIENT_VISUAL_DEPTH")
    shifted = points-offsets

    def algebraic(indices):
        origin = shifted[indices].mean(axis=0)
        relative = shifted[indices]-origin
        solution, _, rank, _ = np.linalg.lstsq(
            np.column_stack([2*relative, np.ones(len(relative))]),
            np.sum(relative**2, axis=1)-radii[indices]**2, rcond=None)
        return origin+solution[:3] if rank == 4 else None

    rng = np.random.default_rng(0)
    best, best_count = None, 0
    subsets = [np.arange(len(points)), *[rng.choice(len(points), 6, replace=False) for _ in range(48)]]
    for indices in subsets:
        center = algebraic(indices)
        if center is None:
            continue
        inliers = np.abs(np.linalg.norm(center-shifted, axis=1)-radii) < .015
        if int(inliers.sum()) > best_count:
            best, best_count = inliers, int(inliers.sum())
    if best_count < max(6, math.ceil(.7*len(points))):
        raise RuntimeError("INCONSISTENT_VISUAL_DEPTH")
    center = algebraic(np.flatnonzero(best))
    if center is None:
        raise RuntimeError("ILL_CONDITIONED_VISUAL_DEPTH")
    shifted, radii = shifted[best], radii[best]
    for _ in range(30):
        delta = center-shifted
        distance = np.linalg.norm(delta, axis=1)
        jacobian = delta/np.maximum(distance[:, None], 1e-12)
        step, _, rank, singular = np.linalg.lstsq(jacobian, distance-radii, rcond=None)
        if rank < 3 or singular[-1] < singular[0]*1e-3:
            raise RuntimeError("ILL_CONDITIONED_VISUAL_DEPTH")
        cost = float(np.sum((distance-radii)**2))
        scale = min(1., .1/max(float(np.linalg.norm(step)), 1e-12))
        for _ in range(12):
            candidate = center-scale*step
            if np.sum((np.linalg.norm(candidate-shifted, axis=1)-radii)**2) <= cost:
                center = candidate
                break
            scale *= .5
        if np.linalg.norm(step) < 1e-7:
            break
    residual = float(np.sqrt(np.mean((np.linalg.norm(center-shifted, axis=1)-radii)**2)))
    if not np.isfinite(center).all() or residual > .025:
        raise RuntimeError("INCONSISTENT_VISUAL_DEPTH")
    return center, residual


def _tangents(axis):
    basis = np.eye(3)[int(np.argmin(np.abs(axis)))]
    first = np.cross(axis, basis)
    first /= np.linalg.norm(first)
    return np.column_stack([first, np.cross(axis, first)])


def fit_shaft_axis(points_mm, rcm_mm, radius_mm, forward):
    """Fit a known-radius cylinder through a calibrated RCM, rejecting caps."""
    if len(points_mm) < 40:
        raise RuntimeError("INSUFFICIENT_NEEDLE_SHAFT_DEPTH")
    d = np.asarray(points_mm)-rcm_mm
    _, vectors = np.linalg.eigh(np.cov(d.T))
    axis = vectors[:, -1]
    if axis@forward < 0:
        axis = -axis
    for _ in range(25):
        tangent = _tangents(axis)
        axial = d@axis
        radial = np.sqrt(np.maximum(np.sum(d*d, axis=1)-axial*axial, 1e-16))
        residual = radial-radius_mm
        # Cap interiors and raster/ID boundary disagreements are not side walls.
        weights = np.minimum(1., .002/np.maximum(np.abs(residual), 1e-12))
        jacobian = -axial[:, None]*(d@tangent)/radial[:, None]
        step, _, rank, singular = np.linalg.lstsq(
            jacobian*np.sqrt(weights[:, None]), -residual*np.sqrt(weights), rcond=None)
        if rank < 2 or singular[-1] < singular[0]*1e-5:
            raise RuntimeError("ILL_CONDITIONED_NEEDLE_AXIS")
        step *= min(1., .05/max(np.linalg.norm(step), 1e-12))
        axis += tangent@step
        axis /= np.linalg.norm(axis)
        if np.linalg.norm(step) < 1e-10:
            break
    axial = d@axis
    radial = np.linalg.norm(d-axial[:, None]*axis, axis=1)
    inliers = np.abs(radial-radius_mm) < .004
    if inliers.sum() < 30 or np.ptp(axial[inliers]) < 1.:
        raise RuntimeError("INSUFFICIENT_NEEDLE_AXIS_SUPPORT")
    residual = float(np.sqrt(np.mean((radial[inliers]-radius_mm)**2)))
    # This is a fit-quality budget, not a calibrated statistical confidence bound.
    angle_quality = max(2e-5, residual/max(np.ptp(axial[inliers]), 1.))
    return axis, residual, angle_quality


class NeedleGeometry:
    def __init__(self, *, camera_position_mm, camera_rotation, width, height, focal_px,
                 rcm_mm, proximal_radius_mm, proximal_length_mm, distal_length_mm,
                 apex_radius_mm, target_radius_mm, proximal_id, apex_id, target_id,
                 target_rgb_min, target_rgb_max, distal_ids=()):
        self.camera_position_mm = np.asarray(camera_position_mm).copy()
        self.camera_rotation = np.asarray(camera_rotation).copy()
        self.rcm_mm = np.asarray(rcm_mm).copy()
        self.width, self.height, self.focal_px = width, height, focal_px
        self.proximal_radius_mm = float(proximal_radius_mm)
        self.proximal_length_mm = float(proximal_length_mm)
        self.distal_length_mm = float(distal_length_mm)
        self.apex_radius_mm = float(apex_radius_mm)
        self.target_radius_mm = float(target_radius_mm)
        self.proximal_id, self.apex_id, self.target_id = proximal_id, apex_id, target_id
        self.distal_ids = tuple(distal_ids)
        self.target_rgb_min, self.target_rgb_max = np.asarray(target_rgb_min), np.asarray(target_rgb_max)

    @classmethod
    def from_reference(cls, reference, width, height, camera_name, settings):
        # Read only fixed calibration and static shape/ID data, never moving poses.
        cid = reference.camera_id(camera_name)
        return cls(camera_position_mm=reference.data.cam_xpos[cid]*1000.,
                   camera_rotation=reference.data.cam_xmat[cid].reshape(3, 3),
                   width=width, height=height,
                   focal_px=.5*height/math.tan(math.radians(reference.model.cam_fovy[cid])/2),
                   rcm_mm=reference.rcm_mm,
                   proximal_radius_mm=reference.config['needle']['proximal_radius_mm'],
                   proximal_length_mm=reference.config['needle']['proximal_length_mm'],
                   distal_length_mm=reference.config['needle']['distal_length_mm'],
                   apex_radius_mm=reference.config['needle']['tip_marker_radius_mm'],
                   target_radius_mm=reference.model.geom('lesion_marker').size[0]*1000.,
                   proximal_id=int(reference.model.geom('needle_proximal').id),
                   apex_id=int(reference.model.geom('needle_tip_marker').id),
                   target_id=int(reference.model.geom('lesion_marker').id),
                   distal_ids=tuple(int(reference.model.geom(name).id)
                                    for name in ('needle_distal', 'needle_distal_tip')),
                   target_rgb_min=settings['target_rgb_min'], target_rgb_max=settings['target_rgb_max'])

    def points(self, observation, mask):
        rows, cols = np.where(mask)
        depth = observation.get('oct_depth', observation['depth'])[rows, cols]*1000.
        finite = np.isfinite(depth) & (depth > 0.)
        validity = observation.get('oct_depth_valid_mask')
        if validity is not None:
            finite &= np.asarray(validity, dtype=bool)[rows, cols]
        rows, cols, depth = rows[finite], cols[finite], depth[finite]
        camera = np.column_stack([(cols+.5-self.width/2)*depth/self.focal_px,
                                  -(rows+.5-self.height/2)*depth/self.focal_px, -depth])
        return camera@self.camera_rotation.T+self.camera_position_mm

    def project(self, world_mm):
        p = self.camera_rotation.T@(np.asarray(world_mm)-self.camera_position_mm)
        return [self.width/2-.5+self.focal_px*p[0]/-p[2],
                self.height/2-.5-self.focal_px*p[1]/-p[2]], float(-p[2]*.001)

    def target(self, observation):
        if observation['target_pixel'] is None:
            raise RuntimeError('TARGET_NOT_VISIBLE')
        mask = observation['segmentation'][:, :, 0] == self.target_id
        rgb = observation['rgb']
        mask &= np.all((rgb >= self.target_rgb_min) & (rgb <= self.target_rgb_max), axis=2)
        points = self.points(observation, mask)
        center, residual = fit_marker_center(points, np.zeros_like(points),
                                              np.full(len(points), self.target_radius_mm))
        return center, {'source': 'target_sphere_fit', 'support': len(points),
                        'residual_mm': residual, 'fit_quality_mm': max(.002, residual*3)}

    def tip(self, observation):
        ids = observation['segmentation'][:, :, 0]
        points = self.points(observation, ids == self.proximal_id)
        axis, side_residual, angle_quality = fit_shaft_axis(
            points, self.rcm_mm, self.proximal_radius_mm,
            self.rcm_mm-self.camera_position_mm)
        # A cylinder axis is an unoriented line. Camera position cannot tell
        # which end is the needle tip: viewing from the opposite side reverses
        # that heuristic. Resolve sign from ordered, observed physical parts.
        distal = self.points(observation, np.isin(ids, (*self.distal_ids, self.apex_id)))
        if not len(distal):
            raise RuntimeError('NEEDLE_DIRECTION_NOT_OBSERVABLE')
        shaft_middle = float(np.median((points-self.rcm_mm)@axis))
        direction_offsets = (distal-self.rcm_mm)@axis-shaft_middle
        if np.mean(direction_offsets > .25) >= .9:
            pass
        elif np.mean(direction_offsets < -.25) >= .9:
            axis = -axis
        else:
            raise RuntimeError('INCONSISTENT_NEEDLE_DIRECTION')
        d = points-self.rcm_mm
        axial = d@axis
        radial = np.linalg.norm(d-axial[:, None]*axis, axis=1)
        interior = points[radial < self.proximal_radius_mm*.75]
        if len(interior) >= 16:
            # Either physical end plane can be visible. Identify it from the
            # side-wall samples, rather than assuming the rear faces the camera.
            offsets = (interior-self.rcm_mm)@axis
            median = np.median(offsets)
            interior = interior[np.abs(offsets-median) < .02]
            if len(interior) >= 16:
                origin = np.mean(interior, axis=0)
                _, singular, vt = np.linalg.svd(interior-origin, full_matrices=False)
                normal = vt[-1]
                if normal@axis < 0:
                    normal = -normal
                residual = float(np.sqrt(np.mean(((interior-origin)@normal)**2)))
                if singular[-2] > .03 and residual < .002 and normal@axis > .995:
                    axis = normal
                    plane = float(np.median((interior-self.rcm_mm)@axis))
                    side_axial = (points-self.rcm_mm)@axis
                    side_radial = np.linalg.norm(
                        points-self.rcm_mm-side_axial[:, None]*axis, axis=1)
                    side_offsets = side_axial[np.abs(side_radial-self.proximal_radius_mm) < .004]-plane
                    side_offsets = side_offsets[np.abs(side_offsets) > .05]
                    end = None
                    if len(side_offsets) >= 30 and np.ptp(side_offsets) > 1.:
                        if np.mean(side_offsets > 0.) >= .9:
                            end = 'rear'
                        elif np.mean(side_offsets < 0.) >= .9:
                            end = 'front'
                    if end is None:
                        raise RuntimeError('NEEDLE_END_PLANE_NOT_OBSERVABLE')
                    length_from_plane = self.distal_length_mm
                    if end == 'rear':
                        length_from_plane += self.proximal_length_mm
                    tip = self.rcm_mm+axis*(plane+length_from_plane)
                    quality = .002+3*residual*(1.+(self.proximal_length_mm+self.distal_length_mm)/.1)
                    if quality <= .01:
                        return tip, {'source': ('shaft_end_plane' if end == 'rear' else 'shaft_front_plane'),
                                     'plane_end': end, 'support': len(interior),
                                     'residual_mm': residual, 'axis_residual_mm': side_residual,
                                     'fit_quality_mm': quality, 'axis_world': axis.tolist()}
        # A true sphere sample constrains position along the measured shaft
        # axis. A visible taper point does not, and is never substituted here.
        marker = self.points(observation, ids == self.apex_id)
        if not len(marker):
            raise RuntimeError('TIP_NOT_OBSERVABLE')
        d = marker-self.rcm_mm
        along = d@axis
        perpendicular2 = np.sum((d-along[:, None]*axis)**2, axis=1)
        # Near-tangent rays amplify tiny axis/depth errors. Reject those rays,
        # rather than rejecting an otherwise usable sphere observation.
        inside = perpendicular2 < (.9*self.apex_radius_mm)**2
        if not inside.any():
            raise RuntimeError('INCONSISTENT_APEX_DEPTH')
        height = np.sqrt(self.apex_radius_mm**2-perpendicular2[inside])
        # The two sphere/axis intersections exchange roles when the camera
        # views the needle from the other end. Choose the root whose sampled
        # surface faces the camera, and reject an ambiguous side view.
        seed = int(np.argmax(height))
        seed_point = marker[inside][seed]
        view = self.camera_position_mm-seed_point
        roots = [sign for sign in (1., -1.)
                 if (seed_point-(self.rcm_mm+axis*(along[inside][seed]+sign*height[seed])))@view > 0.]
        if len(roots) != 1:
            raise RuntimeError('AMBIGUOUS_APEX_DEPTH')
        centers = along[inside]+roots[0]*height
        # ID and depth passes can disagree on antialiased silhouette pixels.
        # Seed with the most central ray, then retain its consistent returns.
        # The edge ray must not shift the common center or poison a whole frame.
        consistent = np.abs(centers-centers[int(np.argmax(height))]) < .004
        centers = centers[consistent]
        length = float(np.median(centers))
        spread = float(np.max(np.abs(centers-length)))
        quality = .002+abs(length)*angle_quality+spread
        if spread > .008 or quality > .01:
            raise RuntimeError('ILL_CONDITIONED_APEX_DEPTH')
        return self.rcm_mm+axis*length, {
            'source': 'apex_sphere_on_shaft_axis', 'support': len(centers),
            'residual_mm': spread, 'axis_residual_mm': side_residual,
            'fit_quality_mm': quality, 'axis_world': axis.tolist()}
