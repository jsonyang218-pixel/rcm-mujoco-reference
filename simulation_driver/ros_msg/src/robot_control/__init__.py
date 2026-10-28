"""Shared, side-effect-free robot control building blocks.

The package is intentionally kept below ``scripts`` so it is importable both
from a catkin-installed node and when the scripts are run directly from a
checkout.  ROS/Tk entry points live next to it; geometry and protocol code does
not start a node or create widgets at import time.
"""

from .kinematics import (  # noqa: F401
    AXES,
    LEGACY_NEEDLE_OFFSET_SIGN,
    NEW_NEEDLE_OFFSET_SIGN,
    absolute_new_needle_position_from_state,
    absolute_tip_position_from_state,
    b_alpha_from_robot_encoder,
    b_distance_from_alpha,
    cartesian_direction_delta,
    cartesian_handle_target,
    cartesian_step_mm_from_um,
    control_direction_flags,
    c_handle_counts_for_distance,
    direction_text,
    initial_observation_geometry,
    inverse_robot_encoders_from_tip,
    make_control_message,
    needle_direction_from_robot_encoders,
    needle_pose_from_robot_encoders,
    new_needle_pose_from_robot_encoders,
    new_needle_solid_segment,
    plan_tip_segments,
    plan_new_needle_tip_segments,
    predicted_robot_encoders,
    relative_to_rcm,
    stepped_values,
    trajectory_to_display,
    validate_manual_jog,
    virtual_handles_from_robot_encoders,
)
from .profiles import (  # noqa: F401
    LEGACY_NEEDLE_PROFILE,
    NEW_NEEDLE_PROFILE,
    NeedleProfile,
)
