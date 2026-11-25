"""Match pre-action histories from one simulated training-target episode."""

import numpy as np


GATES = {
    "error_range_um": [20.0, 100.0],
    "max_radius_difference_um": 5.0,
    "max_true_vector_difference_um": 10.0,
    "max_oct_vector_difference_um": 10.0,
    "min_signed_position_cosine": 0.95,
    "min_abs_radial_speed_um_s": 5.0,
    "max_velocity_cosine": -0.25,
    "min_history_index": 2,
    "min_separation_steps": 3,
}


def _cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def _scene_tips(states):
    from rcm.scene_v9 import RCMSceneV9

    units = np.array([np.pi / 180, np.pi / 180, 0.001])
    scene = RCMSceneV9()
    try:
        tips = []
        for state in states:
            scene.set_task(state * units)
            tips.append(scene.site_mm("needle_tip").copy())
        return np.asarray(tips)
    finally:
        scene.close()


def select_pair(ep, tips_world_mm=None, horizon=20):
    """Return the best matched pair, using past tip displacement for motion.

    Inputs are pre-action observations. A zero horizon is appropriate when
    branches are simulated separately; a positive one reserves recorded steps.
    """
    tr = ep["trajectory"]
    states = np.asarray(tr["states"], dtype=float)
    velocities = np.asarray(tr["velocities"], dtype=float)
    times = np.asarray(tr["simulation_time_s"], dtype=float)
    errors = np.asarray(tr["errors"], dtype=float) * 1000
    oct_valid = np.asarray(tr["oct_navigation_valid"], dtype=bool)
    oct_um = np.asarray(tr["oct_navigation_mm"], dtype=float) * 1000
    n = len(times)
    if horizon < 0 or not isinstance(horizon, (int, np.integer)):
        raise ValueError("horizon must be a nonnegative integer")
    if states.shape != (n, 3) or velocities.shape != (n, 3) or oct_um.shape != (n, 3):
        raise ValueError("states, velocities and OCT must align to pre-action times")
    if errors.shape != (n,) or oct_valid.shape != (n,):
        raise ValueError("mismatched scalar trace lengths")
    tips = _scene_tips(states) if tips_world_mm is None else np.asarray(tips_world_mm, dtype=float)
    target = np.asarray(ep["target_world_mm"], dtype=float)
    if tips.shape != (n, 3) or target.shape != (3,):
        raise ValueError("expected N world-frame tip positions and one fixed target in mm")
    if not all(np.isfinite(x).all() for x in (target, tips, states, velocities, errors, times, oct_um[oct_valid])):
        raise ValueError("non-finite valid observation")
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("timestamps are not strictly increasing")
    truth = (tips - target) * 1000
    radii = np.linalg.norm(truth, axis=1)
    np.testing.assert_allclose(radii, errors, rtol=0, atol=0.05)
    tip_velocity = np.zeros_like(truth)
    tip_velocity[1:] = np.diff(truth, axis=0) / dt[:, None]
    radial = np.divide(np.sum(truth * tip_velocity, axis=1), radii,
                       out=np.zeros_like(radii), where=radii > 0)
    # Enforce actual decrease/increase as well as the signed velocity projection.
    change = np.r_[0., np.diff(radii)] if n else np.array([])
    lower, upper = GATES["error_range_um"]
    eligible = (oct_valid & (radii >= lower) & (radii <= upper)
                & (np.arange(n) >= GATES["min_history_index"])
                & (np.arange(n) + horizon <= n))
    cutoff = GATES["min_abs_radial_speed_um_s"]
    approach = np.flatnonzero(eligible & (radial <= -cutoff) & (change < 0))
    recede = np.flatnonzero(eligible & (radial >= cutoff) & (change > 0))
    counts = {"observations": n, "eligible": int(eligible.sum()),
              "approach": len(approach), "recede": len(recede), "pairs": 0}
    result = {"status": "no_pair", "gates": GATES.copy(), "counts": counts}
    best_key = None
    for i in approach:
        for j in recede:
            if abs(i - j) < GATES["min_separation_steps"]:
                continue
            dr = abs(errors[i] - errors[j])
            dv = np.linalg.norm(truth[i] - truth[j])
            do = np.linalg.norm(oct_um[i] - oct_um[j])
            if (dr > GATES["max_radius_difference_um"] or dv > GATES["max_true_vector_difference_um"]
                    or do > GATES["max_oct_vector_difference_um"]):
                continue
            true_cos, oct_cos = _cosine(truth[i], truth[j]), _cosine(oct_um[i], oct_um[j])
            vel_cos = _cosine(tip_velocity[i], tip_velocity[j])
            if min(true_cos, oct_cos) < GATES["min_signed_position_cosine"] or vel_cos > GATES["max_velocity_cosine"]:
                continue
            counts["pairs"] += 1
            # Minimise the largest position mismatch first; no action or outcome
            # from either continuation is part of the ranking.
            key = (max(dv, do), dv + do, dr, vel_cos, int(i), int(j))
            if best_key is None or key < best_key:
                best_key = key
                best_indices = (i, j)
                best_match = {"radius_difference_um": float(dr), "true_vector_difference_um": float(dv),
                              "oct_vector_difference_um": float(do), "true_vector_cosine": true_cos,
                              "oct_vector_cosine": oct_cos, "velocity_cosine": vel_cos}
    if best_key is None:
        return result
    result.update(status="matched", match=best_match)
    for name, i in zip(("approach", "recede"), best_indices):
        result[name] = {"step": int(i), "error_um": float(radii[i]),
                        "tip_minus_target_world_um": truth[i].tolist(),
                        "oct_target_minus_tip_camera_um": oct_um[i].tolist(),
                        "truth_radial_speed_um_s": float(radial[i]),
                        "tip_velocity_world_um_s": tip_velocity[i].tolist(),
                        "state_deg_deg_mm": states[i].tolist(),
                        "velocity_deg_deg_mm_s": velocities[i].tolist()}
    return result
