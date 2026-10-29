"""Fixed twelve-case validation catalog and pure, guarded route planning."""

import math
from dataclasses import dataclass

from .kinematics import (
    inverse_robot_encoders_from_tip, model_tip_from_robot_encoders,
    refine_tip_path_for_axis_limit, validate_manual_jog,
)
from .trajectory import (
    MAX_EXECUTION_POINTS, build_circle_targets, build_rectangle_targets,
    build_sine_targets, densify_targets,
)

EXPERIMENT_STEP_MM = 0.050
EXPERIMENT_TOLERANCE_UM = 2.0


@dataclass(frozen=True)
class ValidationExperiment:
    identifier: str
    label: str
    kind: str
    plane: str = ""

    @property
    def parameter_text(self):
        return {
            "circle": "半径 800 µm · 128 点",
            "rectangle": "800 × 400 µm · 每边 16 段",
            "sine": "长 600 / 振幅 100 µm · 2 周期",
            "diagonal": "XYZ 各 +800 µm · 原路返回",
            "box": "800 × 800 × 800 µm · 闭环",
            "curve": "X ±800 / Y ±400 / Z +800 µm · 往返",
        }[self.kind]


VALIDATION_EXPERIMENTS = tuple(
    ValidationExperiment(plane + "_" + kind, plane.upper() + " " + label, kind, plane)
    for plane in ("xy", "xz", "yz")
    for kind, label in (("circle", "圆轨迹"), ("rectangle", "矩形轨迹"), ("sine", "正弦往返"))
) + (
    ValidationExperiment("space_diagonal", "空间斜线往返", "diagonal"),
    ValidationExperiment("space_box", "空间盒形闭环", "box"),
    ValidationExperiment("space_curve", "空间曲线往返", "curve"),
)


def get_experiment(identifier):
    for experiment in VALIDATION_EXPERIMENTS:
        if experiment.identifier == identifier:
            return experiment
    raise ValueError("未知实验项目：" + str(identifier))


def experiment_targets(identifier, start_tip):
    """Return samples, segment phases and immutable, reproducible parameters."""
    experiment = get_experiment(identifier)
    start = tuple(float(v) for v in start_tip)
    if len(start) != 3 or not all(math.isfinite(v) for v in start):
        raise ValueError("实验起点必须包含有限 XYZ")
    parameters = {"kind": experiment.kind, "plane": experiment.plane,
                  "step_um": 50.0, "waypoint_tolerance_um": 2.0,
                  "final_tolerance_um": 1.0, "start_tip_mm": start}
    if experiment.kind == "circle":
        targets = build_circle_targets(start, experiment.plane, 800, 128)
        phases = ["approach"] + ["contour"] * 128 + ["return"]
        parameters.update(radius_um=800, circle_points=128)
    elif experiment.kind == "rectangle":
        targets = build_rectangle_targets(start, experiment.plane, 800, 400, points_per_edge=16)
        phases = ["approach"] + ["contour"] * 64 + ["return"]
        parameters.update(width_um=800, height_um=400, points_per_edge=16)
    elif experiment.kind == "sine":
        targets = build_sine_targets(start, experiment.plane, 600, 100, 2, 64)
        phases = ["outward"] * 64 + ["return"] * 64
        parameters.update(length_um=600, amplitude_um=100, cycles=2, segments=64)
    elif experiment.kind == "diagonal":
        targets = [tuple(v + 0.8 for v in start), start]
        phases = ["outward", "return"]
        parameters.update(delta_um=(800, 800, 800))
    elif experiment.kind == "box":
        offsets = ((.8, 0, 0), (.8, .8, 0), (0, .8, 0), (0, .8, .8),
                   (.8, .8, .8), (.8, 0, .8), (0, 0, .8), (0, 0, 0))
        targets = [tuple(start[i] + p[i] for i in range(3)) for p in offsets]
        phases = ["edge_%d" % i for i in range(1, 9)]
        parameters.update(box_edge_um=800)
    else:
        outward = [start]
        for index in range(1, 257):
            t = index / 256.0
            offset = (0.8 * math.sin(2 * math.pi * t),
                      0.4 * math.sin(4 * math.pi * t), 0.8 * t)
            if index == 256:
                offset = (0.0, 0.0, 0.8)
            outward.append(tuple(start[i] + offset[i] for i in range(3)))
        targets = outward[1:] + list(reversed(outward[:-1]))
        phases = ["outward"] * 256 + ["return"] * 256
        parameters.update(x_amplitude_um=800, y_amplitude_um=400, z_length_um=800, segments=256)
    return [start] + targets, ["start"] + phases, parameters


def plan_validation_experiment(identifier, start_tip, current_robot, tip_offset_mm, offset_sign):
    """Validate every predicted pose before any UI motion can be started."""
    samples, phases, parameters = experiment_targets(identifier, start_tip)
    route = densify_targets(samples, EXPERIMENT_STEP_MM, max_points=MAX_EXECUTION_POINTS)
    route = refine_tip_path_for_axis_limit(
        route, current_robot, tip_offset_mm, offset_sign, max_points=MAX_EXECUTION_POINTS)
    robot = tuple(current_robot)
    validate_manual_jog(robot, (0, 0, 0, 0))
    model_start = model_tip_from_robot_encoders(robot, tip_offset_mm, offset_sign)
    for point in route[1:]:
        candidate = tuple(model_start[i] + point[i] - route[0][i] for i in range(3))
        robot = inverse_robot_encoders_from_tip(candidate, robot, tip_offset_mm, offset_sign) + (robot[3],)
        validate_manual_jog(robot, (0, 0, 0, 0))
    # Densification and axis refinement preserve every source vertex exactly.
    # Assign phases in traversal order even for overlapping outward/return legs.
    route_phases = ["start"]
    sample_index = 1
    for point in route[1:]:
        route_phases.append(phases[sample_index])
        if point == samples[sample_index] and sample_index < len(samples) - 1:
            sample_index += 1
    return route, {
        **parameters, "experiment_id": identifier, "waypoint_phases": route_phases,
        "source_points_mm": samples, "source_phases": phases,
    }
