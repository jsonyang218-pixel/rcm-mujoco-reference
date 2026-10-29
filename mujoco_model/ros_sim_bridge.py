#!/usr/bin/env python3
"""Optional ROS1 simulation adapter, never a real motor connection.

Requires a sourced ROS1/catkin process with generated ros_msg messages. This
module is not exercised when running the ROS-free offline demo. It deliberately
uses a local ROS Master and never sources the uploaded ros_network.env.
"""
import argparse
import os
from pathlib import Path
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--control-topic',default='/simulation/controlSignal')
    parser.add_argument('--feedback-topic',default='/simulation/RobotState')
    parser.add_argument('--rate',type=float,default=100.)
    parser.add_argument('--needle-profile',choices=('new','legacy'),default='new')
    args = parser.parse_args()
    if args.control_topic != '/simulation/controlSignal' or args.feedback_topic != '/simulation/RobotState':
        parser.error('This adapter only supports the exact isolated /simulation/ topics')
    if not 1 <= args.rate <= 1000:
        parser.error('--rate must be between 1 and 1000 Hz')
    # Explicitly avoid inherited ROS_MASTER_URI pointing at the actual lab LAN.
    os.environ['ROS_MASTER_URI'] = 'http://127.0.0.1:11311'
    os.environ['ROS_IP'] = '127.0.0.1'
    os.environ.pop('ROS_HOSTNAME',None)
    from driver_loader import get_message_types
    ControlSignal,Encoder = get_message_types(require_ros=True)
    import rospy
    from motor_backend import MotorBackend
    from model_sync import DriverModel
    backend=MotorBackend(needle_profile=args.needle_profile)
    model=DriverModel(args.model_dir,needle_profile=args.needle_profile)
    lock=threading.RLock()
    rospy.init_node('mujoco_simulation_driver',anonymous=False)
    publisher=rospy.Publisher(args.feedback_topic,Encoder,queue_size=10)

    def command_callback(command):
        with lock:
            try:
                backend.consume(command,time.monotonic())
            except (TypeError,ValueError) as exc:
                rospy.logwarn('Rejected simulation command: %s',exc)

    subscriber=rospy.Subscriber(args.control_topic,ControlSignal,command_callback,queue_size=10)
    last=time.monotonic()
    rate=rospy.Rate(args.rate)
    while not rospy.is_shutdown():
        now=time.monotonic();dt=min(.1,max(1e-6,now-last));last=now
        with lock:
            message=backend.step(dt,now)
            model.sync(backend.robot)
            message=model.feedback_from_model(message)
        publisher.publish(message)
        rate.sleep()
    subscriber.unregister()


if __name__=='__main__':
    main()
