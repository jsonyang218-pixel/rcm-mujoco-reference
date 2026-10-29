#!/usr/bin/env python3
"""Read-only diagnostic for a remote ROS RobotState connection.

The probe never publishes control messages. It verifies that the local ROS
environment can resolve the master, subscribe to ``/robot/RobotState`` and
receive fresh ``ros_msg/Encoder`` samples.
"""

import argparse
import time

import rospy
from ros_msg.msg import Encoder


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/robot/RobotState")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args(rospy.myargv()[1:])
    if args.duration <= 0 or args.timeout <= 0:
        parser.error("--duration and --timeout must be positive")
    return args


def main():
    args = parse_args()
    rospy.init_node("ros_msg_feedback_probe", anonymous=True, disable_signals=True)
    samples = []
    started = time.monotonic()
    last_received = [None]

    def callback(message):
        now = time.monotonic()
        samples.append(message)
        last_received[0] = now

    subscriber = rospy.Subscriber(args.topic, Encoder, callback, queue_size=20)
    deadline = started + args.duration
    first_deadline = started + args.timeout
    rate = rospy.Rate(50)
    try:
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if not samples and time.monotonic() >= first_deadline:
                raise RuntimeError(
                    "no RobotState received; check ROS_MASTER_URI, ROS_IP, "
                    "topic type, hostname resolution and firewall"
                )
            rate.sleep()
    finally:
        subscriber.unregister()

    if not samples:
        raise RuntimeError("no RobotState received")
    elapsed = max(time.monotonic() - started, 1e-9)
    print("master=%s" % rospy.get_master().getUri())
    print("topic=%s type=%s" % (args.topic, Encoder._type))
    print("samples=%d rate_hz=%.2f" % (len(samples), len(samples) / elapsed))
    print(
        "last robot ABC E=%s position XYZ=(%.9g, %.9g, %.9g)"
        % (
            tuple(getattr(samples[-1], "robotEncoder" + axis) for axis in "ABCE"),
            samples[-1].posX,
            samples[-1].posY,
            samples[-1].posZ,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, rospy.ROSInterruptException) as error:
        print("feedback_probe: %s" % error)
        raise SystemExit(2)
