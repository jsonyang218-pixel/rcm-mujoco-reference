"""Layer-1 migration check: legacy recovered kinematics vs v9 scanned driver.

The two code paths share encoder constants and needle-direction algebra.  This
checks (a) that the local tip vector matches for identical A/B/C poses, and
(b) that the world-frame mapping used to place the legacy reference scene into
the v9 model reproduces the v9 needle-tip site.  The world frames themselves
differ by the installation transform, so only the shared local frame supports a
strict equality claim without real-machine calibration.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mujoco_model"))

from rcm_recovered_model import EncoderMap, Geometry, RCMModel  # noqa: E402
from rcm_reference import Reference  # noqa: E402
from driver_loader import load_driver  # noqa: E402
from model_sync import DriverModel, HOME  # noqa: E402


def _local_tip(model, ref, kin, counts_per_mm, alpha0, pose):
    a_deg, b_deg, c_mm = pose
    state = ref.state_from_task([math.radians(a_deg), math.radians(b_deg), c_mm * 0.001])
    legacy = np.asarray(model.forward(state, enforce_limits=False))
    a_counts_per_degree = ref.config["encoders"]["a_counts_per_degree"]
    c_reference_mm = ref.config["kinematics"]["c_reference_mm"]
    robot = (round(a_deg * a_counts_per_degree),
             round(counts_per_mm * kin.b_distance_from_alpha(alpha0 + b_deg)),
             round(counts_per_mm * (c_mm + c_reference_mm)), 0)
    v9 = np.asarray(kin.model_tip_from_robot_encoders(robot, 4.0, 1.0))
    return legacy, v9, robot


def main() -> int:
    ref = Reference()
    model = RCMModel(Geometry(**ref.config["kinematics"]), EncoderMap())
    kin = load_driver()
    scene = DriverModel(needle_profile="legacy")
    counts_per_mm = kin.ENCODER_COUNTS_PER_MM
    alpha0 = kin.b_alpha_from_robot_encoder(HOME[1])

    worst_local = 0.0
    worst_world = 0.0
    for a_deg in np.linspace(-10.0, 10.0, 9):
        for b_deg in np.linspace(-5.0, 5.0, 9):
            for c_mm in np.linspace(0.0, 8.0, 9):
                legacy, v9_local, robot = _local_tip(
                    model, ref, kin, counts_per_mm, alpha0, (a_deg, b_deg, c_mm))
                worst_local = max(worst_local, float(np.linalg.norm(legacy - v9_local)))
                scene.sync(robot)
                v9_world_mm = scene.data.site_xpos[scene.sites["needle_tip"]] * 1000.0
                mapped_mm = scene.rcm_world * 1000.0 + scene.robot_to_world @ legacy
                worst_world = max(worst_world, float(np.linalg.norm(mapped_mm - v9_world_mm)))

    result = {
        "worst_local_tip_diff_mm": worst_local,
        "worst_world_mapping_diff_mm": worst_world,
        "pass": worst_local < 1e-3 and worst_world < 1e-3,
    }
    print(result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
