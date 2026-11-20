"""Resolve ROS remappings before admitting an isolated simulation loop."""

import rospy


def resolve_experiment_topics(feedback_topic, control_topic, simulation_only):
    """Check effective names, including launch/CLI remaps, before ROS I/O."""
    feedback = rospy.resolve_name(feedback_topic)
    control = rospy.resolve_name(control_topic)
    if simulation_only:
        for topic in (feedback, control):
            if not topic.startswith("/simulation/") or topic == "/simulation/":
                raise ValueError("--simulation-only requires both resolved topics under /simulation/")
    return feedback, control
