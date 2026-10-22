"""Targets on the posterior compiled eye surface, a simulation fundus proxy."""
from __future__ import annotations

import numpy as np

from rcm_navigation import task_si_units


def eye_frame(reference):
    geom = reference.model.geom("eye_reference").id
    return (reference.data.geom_xpos[geom]*1000.,
            reference.data.geom_xmat[geom].reshape(3, 3),
            reference.model.geom_size[geom]*1000.)


def clearance_lower_bound_mm(reference, task):
    """Conservative tip-to-wall clearance from the ellipsoid affine map.

    For interior points, min(radius)*(1-||normalized local point||) is a
    lower bound on Euclidean wall distance. This does not cover the shaft.
    Task coordinates use degrees, degrees, millimeters.
    """
    q = task_si_units(task)
    robot = reference.nominal.forward(reference.state_from_task(q), enforce_limits=False)
    center, rotation, radii = eye_frame(reference)
    unit = rotation.T@(reference.robot_to_world(robot)-center)/radii
    return float(np.min(radii)*(1.-np.linalg.norm(unit)))


def standoff_task(task, retract_mm):
    result = np.asarray(task, dtype=float).copy()
    result[2] -= retract_mm
    return result


def no_surface_slide_edge(reference, start, end, clearance_mm):
    """Reject angular movement near the wall; always reject leaving the eye."""
    start, end = np.asarray(start), np.asarray(end)
    turning = np.max(np.abs(end[:2]-start[:2])) > 1e-9
    required = clearance_mm if turning else 0.
    count = max(1, int(np.ceil(np.max(np.abs(end-start)/[.1, .1, .02]))))
    return all(clearance_lower_bound_mm(reference, q) >= required-1e-8
               for q in np.linspace(start, end, count+1))


def make_fundus_targets(reference):
    center, rotation, radii = eye_frame(reference)
    targets = []
    for index, xy in enumerate(((.22, -.10), (-.08, .18), (.16, .26)), 1):
        unit = np.r_[xy, np.sqrt(1.-np.dot(xy, xy))]
        world = center+rotation@(radii*unit)
        targets.append({"id": f"P{index}", "point_mm": reference.world_to_robot(world).tolist()})
    return {
        "schema_version": 1, "frame": "robot",
        "target_region": {
            "kind": "posterior_eye_surface", "minimum_normalized_posterior_z": .9,
            "surface_tolerance_um": .001,
            "scope": "Compiled eye ellipsoid posterior cap as fundus proxy; no separate retinal layer",
        },
        "targets": targets,
    }


def make_random_fundus_targets(reference, rng, count, min_separation_mm=1.0,
                               minimum_normalized_posterior_z=.9, max_attempts=10000):
    """Generate a reproducible unordered set of posterior-surface targets.

    Sampling is uniform in the cap's normalized surface coordinates: z is
    uniform over [minimum_normalized_posterior_z, 1] and azimuth is uniform
    over [0, 2*pi).  The physical ellipsoid distance is used for rejection so
    the requested separation has millimetre semantics after scaling.
    """
    if not hasattr(rng, "uniform"):
        raise TypeError("rng must be a NumPy Generator")
    if int(count) != count or count < 1:
        raise ValueError("count must be a positive integer")
    if not np.isfinite(min_separation_mm) or min_separation_mm < 0:
        raise ValueError("min_separation_mm must be finite and nonnegative")
    z_min = float(minimum_normalized_posterior_z)
    if not 0.0 < z_min < 1.0:
        raise ValueError("minimum_normalized_posterior_z must be between 0 and 1")
    center, rotation, radii = eye_frame(reference)
    selected = []
    attempts = 0
    while len(selected) < int(count) and attempts < int(max_attempts):
        attempts += 1
        z = float(rng.uniform(z_min, 1.0))
        azimuth = float(rng.uniform(0.0, 2.0*np.pi))
        radial = np.sqrt(max(0.0, 1.0-z*z))
        unit = np.array([radial*np.cos(azimuth), radial*np.sin(azimuth), z])
        world = center + rotation @ (radii*unit)
        if all(float(np.linalg.norm(world-old)) >= float(min_separation_mm)-1e-12
               for old in selected):
            selected.append(world)
    if len(selected) != int(count):
        raise RuntimeError(
            f"Could not sample {count} fundus targets with {min_separation_mm:g} mm separation "
            f"after {max_attempts} attempts")
    targets = [{"id": f"P{index+1}", "point_mm": reference.world_to_robot(world).tolist()}
               for index, world in enumerate(selected)]
    return {
        "schema_version": 1, "frame": "robot", "targets": targets,
        "target_region": {
            "kind": "posterior_eye_surface",
            "minimum_normalized_posterior_z": z_min,
            "surface_tolerance_um": .001,
            "scope": "Compiled eye ellipsoid posterior cap as fundus proxy; no separate retinal layer",
        },
    }


def verify_fundus_targets(reference, solutions, region):
    if region.get("kind") != "posterior_eye_surface":
        raise ValueError("Unknown target region")
    center, rotation, radii = eye_frame(reference)
    rows = []
    for solution in solutions:
        local = rotation.T@(np.asarray(solution["requested_world_mm"])-center)
        unit = local/radii
        norm = float(np.linalg.norm(unit))
        radial_gap_um = float(np.linalg.norm(local-local/norm)*1000.) if norm > 0 else float(np.min(radii)*1000.)
        passed = (radial_gap_um <= region["surface_tolerance_um"]
                  and unit[2] >= region["minimum_normalized_posterior_z"])
        rows.append({"waypoint_id": solution["id"], "result": "PASS" if passed else "FAIL",
                     "eye_local_mm": local.tolist(), "ellipsoid_value": norm**2,
                     "normalized_posterior_z": float(unit[2]), "surface_radial_gap_um": radial_gap_um})
    return {"result": "PASS" if rows and all(r["result"] == "PASS" for r in rows) else "FAIL",
            "definition": region, "targets": rows}
