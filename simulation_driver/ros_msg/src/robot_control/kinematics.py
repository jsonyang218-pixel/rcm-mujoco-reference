"""Pure kinematics and control-protocol helpers shared by both needle types.

All distances are millimetres and all robot encoder values are integer counts.
The functions in this module do not access ROS, Tk, files, or global robot
state, which makes them safe to use from tests and offline planners.
"""

import math

from ros_msg.msg import controlSignal


AXES = ("A", "B", "C", "E")
CONTROL_MODES = ("hold", "forward", "backward")
INT32_MIN = -(2 ** 31)
INT32_MAX = 2 ** 31 - 1
MIN_CARTESIAN_STEP_UM = 10
MAX_CARTESIAN_STEP_UM = 1000
DEFAULT_CARTESIAN_STEP_UM = 50
MIN_CARTESIAN_STEP_MM = MIN_CARTESIAN_STEP_UM / 1000.0
MAX_CARTESIAN_STEP_MM = MAX_CARTESIAN_STEP_UM / 1000.0
DEFAULT_CARTESIAN_STEP_MM = DEFAULT_CARTESIAN_STEP_UM / 1000.0
MAX_CARTESIAN_AXIS_CHANGE_DEG = 8.0
DEFAULT_TIP_OFFSET_MM = 4.0
NEW_NEEDLE_SOLID_LENGTH_MM = 1.0
LEGACY_NEEDLE_OFFSET_SIGN = 1.0
DEFAULT_NEEDLE_OFFSET_SIGN = LEGACY_NEEDLE_OFFSET_SIGN
NEW_NEEDLE_OFFSET_SIGN = -1.0
DEFAULT_NEEDLE_A_ANGLE_DEG = 76.23602856007702
ABSOLUTE_TARGET_TOL_MM = 0.001
ABSOLUTE_TARGET_WAYPOINT_STEP_MM = 0.300
# Intermediate route points shape the path; the final target keeps the tighter
# 1 um acceptance threshold while every intermediate correction stays within
# the requested 1.5 um error bound.
ABSOLUTE_TARGET_WAYPOINT_TOL_MM = 0.0015
MAX_PLOT_POINTS = 12_000
CONTROL_GEAR = 10
PROTOCOL_INITIAL_HANDLES = (0, 0, 0, 0)
ENCODER_COUNTS_PER_MM = 4 * 1024 * 15.5817175
C_ENCODER_COUNTS_PER_HANDLE = 3.0 * CONTROL_GEAR * 10000.0 / 4000.0
INSERTION_STEP_MM = 0.010
C_HANDLE_COUNTS_PER_MM = ENCODER_COUNTS_PER_MM / C_ENCODER_COUNTS_PER_HANDLE
INSERTION_STEP_HANDLE_COUNTS = INSERTION_STEP_MM * C_HANDLE_COUNTS_PER_MM
ROBOT_ENCODER_LIMITS = {
    "A": (-74_873, 73_302),
    "B": (223_159, 5_023_488),
    "C": (0, 2_952_362),
}
CARTESIAN_DIRECTIONS = {
    "left": (1.0, 0.0, 0.0),
    "right": (-1.0, 0.0, 0.0),
    "forward_y": (0.0, 1.0, 0.0),
    "backward_y": (0.0, -1.0, 0.0),
    "up": (0.0, 0.0, -1.0),
    "down": (0.0, 0.0, 1.0),
}


def direction_text(forward, backward):
    if forward and backward:
        return "非法：同时进针/退针"
    if forward:
        return "进针"
    if backward:
        return "退针"
    return "保持"


def control_direction_flags(mode):
    if mode == "forward":
        return True, False
    if mode == "backward":
        return False, True
    return False, False


def c_handle_counts_for_distance(distance_mm):
    return float(distance_mm) * C_HANDLE_COUNTS_PER_MM


def stepped_values(values, axes, step):
    result = list(values)
    amounts = list(step) if isinstance(step, (tuple, list)) else [step] * len(axes)
    if len(amounts) != len(axes):
        raise ValueError("轴和步长数量不一致")
    for axis, amount in zip(axes, amounts):
        index = AXES.index(axis)
        updated = result[index] + int(amount)
        if updated < INT32_MIN or updated > INT32_MAX:
            raise ValueError("%s 轴编码值超出 int32 范围" % axis)
        result[index] = updated
    return result


def make_control_message(values, mode):
    message = controlSignal()
    for axis, value in zip(AXES, values):
        setattr(message, "handleEncoder" + axis, int(value))
    message.stepForward, message.stepBackward = control_direction_flags(mode)
    return message


def b_distance_from_alpha(alpha_deg):
    alpha = math.radians(float(alpha_deg))
    radicand = 52.0 ** 2 - (60.0 * math.sin(alpha) - 16.0) ** 2
    if radicand < 0.0:
        raise ValueError("B 轴目标角超出连杆几何范围")
    return 98.48 - 60.0 * math.cos(alpha) - math.sqrt(radicand)


def b_alpha_from_robot_encoder(robot_encoder_b):
    distance = float(robot_encoder_b) / ENCODER_COUNTS_PER_MM
    ad = 98.48 - distance
    ac = math.hypot(ad, 16.0)
    if ac < 1e-12:
        raise ValueError("B 轴反馈无法反解连杆角")
    gamma = math.degrees(math.atan2(16.0, ad))
    beta_cos = (60.0 ** 2 + ac ** 2 - 52.0 ** 2) / (2.0 * 60.0 * ac)
    beta = math.degrees(math.acos(max(-1.0, min(1.0, beta_cos))))
    return beta + gamma


def needle_direction_from_robot_encoders(robot_values):
    robot_a, robot_b, _robot_c, _robot_e = (int(value) for value in robot_values)
    phi = math.radians(robot_a * 360.0 / (2000.0 * 100.0))
    alpha = math.radians(b_alpha_from_robot_encoder(robot_b))
    theta = alpha + math.radians(22.0)
    direction = (
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
        math.sin(theta) * math.cos(phi),
    )
    length = math.sqrt(sum(component * component for component in direction))
    if length <= 1e-12:
        raise ValueError("针轴方向无法归一化")
    return tuple(component / length for component in direction)


def inverse_robot_encoders_from_position(position, axial_offset_mm=0.0):
    x, y, z = (float(value) for value in position)
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 1e-9:
        raise ValueError("当前 XYZ 太接近 RCM，无法稳定计算姿态；请先离开奇异点")
    phi_deg = math.degrees(math.atan2(x, z))
    theta_deg = math.degrees(math.acos(max(-1.0, min(1.0, y / length))))
    alpha_deg = theta_deg - 22.0
    target_a = int(round(phi_deg * (2000.0 * 100.0 / 360.0)))
    target_b = int(round(ENCODER_COUNTS_PER_MM * b_distance_from_alpha(alpha_deg)))
    target_c = int(round(ENCODER_COUNTS_PER_MM * (14.0 + length - float(axial_offset_mm))))
    return target_a, target_b, target_c


def inverse_robot_encoders_from_tip(target_tip, current_robot,
                                    tip_offset_mm=DEFAULT_TIP_OFFSET_MM,
                                    tip_offset_sign=LEGACY_NEEDLE_OFFSET_SIGN):
    """Solve a tip in the encoder model's RCM frame (not a UI zero frame)."""
    target = tuple(float(value) for value in target_tip)
    if len(target) != 3:
        raise ValueError("针尖目标输入维度错误")
    offset = float(tip_offset_mm)
    sign = float(tip_offset_sign)
    if offset <= 0.0 or sign == 0.0:
        raise ValueError("针尖相对交接点的固定偏置必须大于零")
    radius = math.sqrt(sum(value * value for value in target))
    current_direction = needle_direction_from_robot_encoders(current_robot)
    if radius <= 1e-9:
        direction = current_direction
        junction_distance = -sign * offset
    else:
        target_unit = tuple(value / radius for value in target)
        if sum(current_direction[i] * target_unit[i] for i in range(3)) >= 0.0:
            direction = target_unit
            junction_distance = radius - sign * offset
        else:
            direction = tuple(-value for value in target_unit)
            junction_distance = -radius - sign * offset
    phi_deg = math.degrees(math.atan2(direction[0], direction[2]))
    theta_deg = math.degrees(math.acos(max(-1.0, min(1.0, direction[1]))))
    alpha_deg = theta_deg - 22.0
    target_a = int(round(phi_deg * (2000.0 * 100.0 / 360.0)))
    target_b = int(round(ENCODER_COUNTS_PER_MM * b_distance_from_alpha(alpha_deg)))
    target_c = int(round(ENCODER_COUNTS_PER_MM * (14.0 + junction_distance)))
    return target_a, target_b, target_c


def inverse_robot_encoders_from_new_needle_tip(target_tip, current_robot,
                                                tip_offset_mm=DEFAULT_TIP_OFFSET_MM):
    """Compatibility wrapper for the replacement-needle inverse solver."""
    return inverse_robot_encoders_from_tip(
        target_tip, current_robot, tip_offset_mm, NEW_NEEDLE_OFFSET_SIGN,
    )


def cartesian_direction_delta(direction, step_mm):
    try:
        vector = CARTESIAN_DIRECTIONS[direction]
    except KeyError:
        raise ValueError("未知笛卡尔方向：%s" % direction)
    step = float(step_mm)
    if not MIN_CARTESIAN_STEP_MM <= step <= MAX_CARTESIAN_STEP_MM:
        raise ValueError(
            "笛卡尔步长范围是 %.3f 到 %.3f mm（%d–%d µm）" %
            (MIN_CARTESIAN_STEP_MM, MAX_CARTESIAN_STEP_MM,
             MIN_CARTESIAN_STEP_UM, MAX_CARTESIAN_STEP_UM)
        )
    return tuple(component * step for component in vector)


def cartesian_step_mm_from_um(step_um):
    try:
        numeric_step = float(step_um)
    except (TypeError, ValueError):
        raise ValueError("XYZ 点动步长必须是整数微米")
    if not math.isfinite(numeric_step) or not numeric_step.is_integer():
        raise ValueError("XYZ 点动步长必须是整数微米")
    step = int(numeric_step)
    if step < MIN_CARTESIAN_STEP_UM or step > MAX_CARTESIAN_STEP_UM:
        raise ValueError("XYZ 点动步长范围是 %d 到 %d µm" %
                         (MIN_CARTESIAN_STEP_UM, MAX_CARTESIAN_STEP_UM))
    return step / 1000.0


def model_tip_from_robot_encoders(robot_values, tip_offset_mm=DEFAULT_TIP_OFFSET_MM,
                                 tip_offset_sign=LEGACY_NEEDLE_OFFSET_SIGN):
    """Reconstruct the tip in the same frame as the encoder inverse solver.

    A UI origin captured from RobotState may be translated from this frame.
    Displacements share the robot XYZ axes and can be applied in either frame.
    """
    direction = needle_direction_from_robot_encoders(robot_values)
    radius = (int(robot_values[2]) / ENCODER_COUNTS_PER_MM - 14.0 +
              float(tip_offset_mm) * float(tip_offset_sign))
    return tuple(radius * value for value in direction)


def cartesian_handle_target(robot_values, handle_values, position, delta_xyz,
                            gear=CONTROL_GEAR, tip_offset_mm=DEFAULT_TIP_OFFSET_MM,
                            tip_offset_sign=LEGACY_NEEDLE_OFFSET_SIGN):
    current_robot = tuple(int(value) for value in robot_values)
    current_handles = tuple(int(value) for value in handle_values)
    current_junction = tuple(float(value) for value in position)
    delta = tuple(float(value) for value in delta_xyz)
    if (len(current_robot) != 4 or len(current_handles) != 4 or
            len(current_junction) != 3 or len(delta) != 3):
        raise ValueError("笛卡尔点动输入维度错误")
    if not all(math.isfinite(value) for value in current_junction + delta):
        raise ValueError("XYZ 反馈和位移必须是有限数值")
    tip_direction = needle_direction_from_robot_encoders(current_robot)
    signed_offset = float(tip_offset_mm) * float(tip_offset_sign)
    current_tip = tuple(current_junction[i] + signed_offset * tip_direction[i] for i in range(3))
    target_tip = tuple(current_tip[i] + delta[i] for i in range(3))
    model_tip = model_tip_from_robot_encoders(
        current_robot, tip_offset_mm, tip_offset_sign)
    model_target = tuple(model_tip[i] + delta[i] for i in range(3))
    alpha = b_alpha_from_robot_encoder(current_robot[1])
    target_a, target_b, target_c = inverse_robot_encoders_from_tip(
        model_target, current_robot, tip_offset_mm, tip_offset_sign,
    )
    target_alpha = b_alpha_from_robot_encoder(target_b)
    target_direction = needle_direction_from_robot_encoders(
        (target_a, target_b, target_c, current_robot[3]))
    axis_dot = max(-1.0, min(1.0, sum(
        tip_direction[i] * target_direction[i] for i in range(3))))
    axis_change_deg = math.degrees(math.acos(axis_dot))
    if axis_change_deg > MAX_CARTESIAN_AXIS_CHANGE_DEG:
        raise ValueError(
            "该 XYZ 点动需要针轴姿态变化 %.2f°，超过单次 %.1f° 安全上限；"
            "RCM 附近只允许沿当前针轴进给或外延" %
            (axis_change_deg, MAX_CARTESIAN_AXIS_CHANGE_DEG))
    a_per_handle = (2.0 * gear * 3.0 / 4000.0) * (2000.0 * 100.0 / 360.0)
    b_per_handle = gear * 3.0 / 4000.0
    c_per_handle = 3.0 * gear * 10000.0 / 4000.0
    handle_delta = (
        int(round((target_a - current_robot[0]) / a_per_handle)),
        int(round((target_alpha - alpha) / b_per_handle)),
        int(round((target_c - current_robot[2]) / c_per_handle)),
        0,
    )
    target_handles = tuple(current_handles[i] + handle_delta[i] for i in range(4))
    for axis, value in zip(AXES, target_handles):
        if value < INT32_MIN or value > INT32_MAX:
            raise ValueError("%s 轴编码值超出 int32 范围" % axis)
    return target_tip, target_handles, (target_a, target_b, target_c, current_robot[3])


def plan_tip_segments(start_tip, target_tip, current_robot,
                      tip_offset_mm=DEFAULT_TIP_OFFSET_MM,
                      tip_offset_sign=LEGACY_NEEDLE_OFFSET_SIGN,
                      max_step_mm=ABSOLUTE_TARGET_WAYPOINT_STEP_MM,
                      max_axis_change_deg=MAX_CARTESIAN_AXIS_CHANGE_DEG):
    """Plan bounded RCM-frame tip waypoints for either needle variant."""
    start = tuple(float(value) for value in start_tip)
    target = tuple(float(value) for value in target_tip)
    if len(start) != 3 or len(target) != 3:
        raise ValueError("针尖目标输入维度错误")
    max_step = float(max_step_mm)
    max_angle = float(max_axis_change_deg)
    if not all(math.isfinite(value) for value in start + target + (max_step, max_angle)):
        raise ValueError("XYZ 反馈、目标和路点限制必须是有限数值")
    if max_step <= 0.0 or max_angle <= 0.0:
        raise ValueError("路点限制必须大于零")
    if math.sqrt(sum((target[i] - start[i]) ** 2 for i in range(3))) <= ABSOLUTE_TARGET_TOL_MM:
        return []
    points = []
    current_tip = start
    robot = tuple(int(value) for value in current_robot)
    model_start = model_tip_from_robot_encoders(robot, tip_offset_mm, tip_offset_sign)
    while True:
        remaining = tuple(target[i] - current_tip[i] for i in range(3))
        distance = math.sqrt(sum(value * value for value in remaining))
        if distance <= ABSOLUTE_TARGET_TOL_MM:
            break
        fraction = min(1.0, max_step / distance)
        current_direction = needle_direction_from_robot_encoders(robot)
        solved_robot = None
        candidate = None
        for _attempt in range(40):
            candidate = tuple(current_tip[i] + fraction * remaining[i] for i in range(3))
            # Waypoints stay in the UI frame; only the inverse solver sees the
            # translated encoder-model frame. Do not command the origin offset.
            model_candidate = tuple(model_start[i] + (candidate[i] - start[i])
                                    for i in range(3))
            solved_a, solved_b, solved_c = inverse_robot_encoders_from_tip(
                model_candidate, robot, tip_offset_mm, tip_offset_sign)
            solved_robot = (solved_a, solved_b, solved_c, robot[3])
            solved_direction = needle_direction_from_robot_encoders(solved_robot)
            dot = max(-1.0, min(1.0, sum(
                current_direction[i] * solved_direction[i] for i in range(3))))
            if math.degrees(math.acos(dot)) <= max_angle + 1e-9:
                break
            fraction *= 0.5
        if candidate is None or solved_robot is None:
            raise ValueError("无法生成连续的针轴路点")
        points.append(candidate)
        current_tip, robot = candidate, solved_robot
    return points


def refine_tip_path_for_axis_limit(targets, current_robot,
                                   tip_offset_mm=DEFAULT_TIP_OFFSET_MM,
                                   tip_offset_sign=LEGACY_NEEDLE_OFFSET_SIGN,
                                   max_points=None):
    """Insert collinear midpoints where a sampled path exceeds the axis limit.

    Keep all original vertices in the fixed UI frame. Translate to the encoder
    model only for the inverse solve, advancing the predicted pose each leg.
    No commands are published and no configured displacement bound is enlarged.
    """
    path = [tuple(float(value) for value in point) for point in targets]
    if not path or any(len(point) != 3 or
                       not all(math.isfinite(value) for value in point)
                       for point in path):
        raise ValueError("轨迹路点必须包含有限 XYZ 坐标")
    robot = tuple(int(value) for value in current_robot)
    model_start = model_tip_from_robot_encoders(robot, tip_offset_mm, tip_offset_sign)
    result = [path[0]]
    for endpoint in path[1:]:
        pending = [(endpoint, 0)]
        while pending:
            candidate, depth = pending.pop()
            model_candidate = tuple(model_start[i] + candidate[i] - path[0][i]
                                    for i in range(3))
            solved = inverse_robot_encoders_from_tip(
                model_candidate, robot, tip_offset_mm, tip_offset_sign)
            next_robot = tuple(solved) + (robot[3],)
            direction = needle_direction_from_robot_encoders(robot)
            next_direction = needle_direction_from_robot_encoders(next_robot)
            dot = max(-1.0, min(1.0, sum(direction[i] * next_direction[i]
                                        for i in range(3))))
            if math.degrees(math.acos(dot)) <= MAX_CARTESIAN_AXIS_CHANGE_DEG + 1e-9:
                if max_points is not None and len(result) >= max_points:
                    raise ValueError("Execution path exceeds %d waypoints" % max_points)
                result.append(candidate)
                robot = next_robot
                continue
            midpoint = tuple((result[-1][i] + candidate[i]) / 2.0 for i in range(3))
            if depth >= 40 or midpoint == result[-1] or midpoint == candidate:
                raise ValueError("无法在单段 %.0f° 姿态上限内生成连续轨迹" %
                                 MAX_CARTESIAN_AXIS_CHANGE_DEG)
            pending.extend(((candidate, depth + 1), (midpoint, depth + 1)))
    return result


def plan_new_needle_tip_segments(start_tip, target_tip, current_robot,
                                 tip_offset_mm=DEFAULT_TIP_OFFSET_MM,
                                 max_step_mm=ABSOLUTE_TARGET_WAYPOINT_STEP_MM,
                                 max_axis_change_deg=MAX_CARTESIAN_AXIS_CHANGE_DEG):
    """Compatibility wrapper for the replacement-needle planner."""
    return plan_tip_segments(
        start_tip, target_tip, current_robot, tip_offset_mm,
        NEW_NEEDLE_OFFSET_SIGN, max_step_mm, max_axis_change_deg,
    )


def predicted_robot_encoders(robot_values, handle_delta, gear=CONTROL_GEAR):
    robot_a, robot_b, robot_c, robot_e = (int(value) for value in robot_values)
    delta_a, delta_b, delta_c, delta_e = (int(value) for value in handle_delta)
    if delta_e:
        raise ValueError("E 轴尚无经标定的手柄换算公式，已禁止点动")
    a_per_handle = (2.0 * gear * 3.0 / 4000.0) * (2000.0 * 100.0 / 360.0)
    c_per_handle = 3.0 * gear * 10000.0 / 4000.0
    alpha = b_alpha_from_robot_encoder(robot_b)
    target_alpha = alpha + delta_b * gear * 3.0 / 4000.0
    target_b = int(round(ENCODER_COUNTS_PER_MM * b_distance_from_alpha(target_alpha)))
    return (int(round(robot_a + delta_a * a_per_handle)), target_b,
            int(round(robot_c + delta_c * c_per_handle)), robot_e)


def virtual_handles_from_robot_encoders(robot_values, zero_robot_values, gear=CONTROL_GEAR):
    robot_a, robot_b, robot_c, _robot_e = (int(value) for value in robot_values)
    zero_a, zero_b, zero_c, _zero_e = (int(value) for value in zero_robot_values)
    a_per_handle = (2.0 * gear * 3.0 / 4000.0) * (2000.0 * 100.0 / 360.0)
    c_per_handle = 3.0 * gear * 10000.0 / 4000.0
    alpha = b_alpha_from_robot_encoder(robot_b)
    zero_alpha = b_alpha_from_robot_encoder(zero_b)
    b_per_handle = gear * 3.0 / 4000.0
    return [
        int(round(PROTOCOL_INITIAL_HANDLES[0] + (robot_a - zero_a) / a_per_handle)),
        int(round(PROTOCOL_INITIAL_HANDLES[1] + (alpha - zero_alpha) / b_per_handle)),
        int(round(PROTOCOL_INITIAL_HANDLES[2] + (robot_c - zero_c) / c_per_handle)),
        PROTOCOL_INITIAL_HANDLES[3],
    ]


def validate_manual_jog(robot_values, handle_delta):
    for axis, value in zip(AXES[:3], robot_values[:3]):
        low, high = ROBOT_ENCODER_LIMITS[axis]
        if not low <= int(value) <= high:
            raise ValueError(
                "%s 轴当前机器人编码器 %d 不在已确认安全范围 [%d, %d]；请先完成零点/限位标定" %
                (axis, value, low, high))
    target = predicted_robot_encoders(robot_values, handle_delta)
    for axis, value in zip(AXES[:3], target[:3]):
        low, high = ROBOT_ENCODER_LIMITS[axis]
        if not low <= value <= high:
            raise ValueError("%s 轴预测目标编码器 %d 超出安全范围 [%d, %d]" %
                             (axis, value, low, high))
    return target


def validate_cartesian_delta(robot_values, handle_delta, tip_offset_mm,
                             tip_offset_sign,
                             max_step_mm=MAX_CARTESIAN_STEP_MM):
    """Check a quantized command against its caller's local step limit."""
    max_step = float(max_step_mm)
    if not math.isfinite(max_step) or max_step <= 0.0:
        raise ValueError("单段 XYZ 修正上限必须是正数")
    target = validate_manual_jog(robot_values, handle_delta)
    directions = [needle_direction_from_robot_encoders(values)
                  for values in (robot_values, target)]
    tips = []
    for values, direction in zip((robot_values, target), directions):
        radius = (values[2] / ENCODER_COUNTS_PER_MM - 14.0 +
                  tip_offset_mm * tip_offset_sign)
        tips.append(tuple(radius * component for component in direction))
    distance = math.sqrt(sum((tips[1][i] - tips[0][i]) ** 2 for i in range(3)))
    dot = max(-1.0, min(1.0, sum(directions[0][i] * directions[1][i]
                                for i in range(3))))
    if distance > max_step + 1e-9:
        raise ValueError("累计 XYZ 指令超过单段 %.0f µm 上限" % (max_step * 1000.0))
    if math.degrees(math.acos(dot)) > MAX_CARTESIAN_AXIS_CHANGE_DEG + 1e-9:
        raise ValueError("累计 XYZ 指令超过单段 %.0f° 姿态上限" %
                         MAX_CARTESIAN_AXIS_CHANGE_DEG)
    return target


def initial_observation_geometry(rcm, rcm_offset_mm, offset_sign=LEGACY_NEEDLE_OFFSET_SIGN):
    length = max(float(rcm_offset_mm), 0.1)
    angle = math.radians(DEFAULT_NEEDLE_A_ANGLE_DEG)
    direction = (0.0, math.cos(angle), math.sin(angle))
    tip = tuple(rcm[i] + float(offset_sign) * length * direction[i] for i in range(3))
    a_axis_center = tuple(rcm)
    half = max(0.9 * length, 1.0)
    a_axis_start = (a_axis_center[0], a_axis_center[1] - half, a_axis_center[2])
    a_axis_end = (a_axis_center[0], a_axis_center[1] + half, a_axis_center[2])
    arc_angles = [math.radians(-58.0 + 116.0 * i / 24.0) for i in range(25)]
    signed_length = float(offset_sign) * length
    axial_component = signed_length * math.cos(angle)
    lateral_component = signed_length * math.sin(angle)
    a_arc = [
        (rcm[0] + lateral_component * math.sin(rotation),
         rcm[1] + axial_component,
         rcm[2] + lateral_component * math.cos(rotation))
        for rotation in arc_angles
    ]
    return {
        "tip": tip,
        "needle_direction": tuple(float(offset_sign) * value for value in direction),
        "a_axis_center": a_axis_center,
        "a_axis_start": a_axis_start,
        "a_axis_end": a_axis_end,
        "a_arc": a_arc,
        "limit_points": [tip, a_axis_start, a_axis_end] + a_arc,
    }


def tip_to_display(point):
    return tuple(float(value) for value in point)


def relative_to_rcm(point, rcm):
    return tuple(float(value) - float(rcm[i]) for i, value in enumerate(point))


def trajectory_to_display(points):
    return [(point[0],) + tip_to_display(point[1:4]) for point in points]


def b_axis_display_direction(phi):
    return math.cos(float(phi)), 0.0, -math.sin(float(phi))


def needle_pose_from_robot_encoders(robot_values, junction,
                                    tip_offset_mm=DEFAULT_TIP_OFFSET_MM,
                                    tip_offset_sign=LEGACY_NEEDLE_OFFSET_SIGN):
    robot_a, robot_b, robot_c, _robot_e = (int(value) for value in robot_values)
    phi = math.radians(robot_a * 360.0 / (2000.0 * 100.0))
    alpha = math.radians(b_alpha_from_robot_encoder(robot_b))
    theta = alpha + math.radians(22.0)
    length = float(tip_offset_mm)
    sign = float(tip_offset_sign)
    if length <= 0.0 or sign == 0.0:
        raise ValueError("针尖相对交接点的固定偏置必须大于零")
    direction = needle_direction_from_robot_encoders(robot_values)
    tip = tuple(junction[i] + sign * length * direction[i] for i in range(3))
    return {
        "display_tip": tip,
        "length": length,
        "offset_sign": sign,
        "phi": phi,
        "theta": theta,
        "alpha": alpha,
        "axial_travel": robot_c / ENCODER_COUNTS_PER_MM - 14.0,
        "needle_direction": direction,
    }


def new_needle_pose_from_robot_encoders(robot_values, junction,
                                        tip_offset_mm=DEFAULT_TIP_OFFSET_MM):
    return needle_pose_from_robot_encoders(
        robot_values, junction, tip_offset_mm, NEW_NEEDLE_OFFSET_SIGN)


def absolute_tip_position_from_state(robot_values, junction, rcm,
                                     tip_offset_mm=DEFAULT_TIP_OFFSET_MM,
                                     tip_offset_sign=LEGACY_NEEDLE_OFFSET_SIGN):
    rcm_absolute = tuple(float(value) for value in rcm)
    relative_junction = relative_to_rcm(junction, rcm_absolute)
    pose = needle_pose_from_robot_encoders(
        robot_values, relative_junction, tip_offset_mm, tip_offset_sign)
    return tuple(rcm_absolute[i] + pose["display_tip"][i] for i in range(3))


def absolute_new_needle_position_from_state(robot_values, junction, rcm,
                                            tip_offset_mm=DEFAULT_TIP_OFFSET_MM):
    return absolute_tip_position_from_state(
        robot_values, junction, rcm, tip_offset_mm, NEW_NEEDLE_OFFSET_SIGN)


def new_needle_solid_segment(rcm, pose, solid_length_mm=NEW_NEEDLE_SOLID_LENGTH_MM):
    del rcm  # retained in the public signature for compatibility
    tip = tuple(float(value) for value in pose["display_tip"])
    length = float(solid_length_mm)
    if length <= 0.0:
        raise ValueError("实体针长度必须大于零")
    direction = tuple(float(value) for value in pose["needle_direction"])
    sign = float(pose.get("offset_sign", NEW_NEEDLE_OFFSET_SIGN))
    return tip, tuple(tip[i] + sign * length * direction[i] for i in range(3))
