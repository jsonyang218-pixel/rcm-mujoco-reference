#!/usr/bin/env python3
"""Private loopback ROS validation of the standalone simulation package."""

import argparse
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import threading
import time
import xmlrpc.client

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))


def validate(output, cases=None):
    import rospy
    from ros_msg.msg import Encoder, controlSignal
    from robot_control.kinematics import (
        AXES,
        ENCODER_COUNTS_PER_MM,
        model_tip_from_robot_encoders,
        predicted_robot_encoders,
    )

    if "127.0.0.1" not in os.environ["ROS_MASTER_URI"]:
        raise RuntimeError("private master required")
    rospy.init_node("base_position_fixture", disable_signals=True)
    lock = threading.RLock()
    home = (-5, 1466022, 893518, 0)
    work = (home[0], home[1], home[2] + round(0.5 * ENCODER_COUNTS_PER_MM), 0)
    robot, goal, handles = list(work), list(work), [9, 0, 1, 0]
    mode = ["normal"]
    commands, packets = [], []
    running = threading.Event()
    running.set()
    enabled = threading.Event()
    enabled.set()
    publisher = rospy.Publisher("/simulation/RobotState", Encoder, queue_size=10)

    def command(message):
        with lock:
            target = tuple(getattr(message, "handleEncoder" + axis) for axis in AXES)
            delta = tuple(a - b for a, b in zip(target, handles))
            packets.append(dict(target=target, forward=message.stepForward, backward=message.stepBackward))
            handles[:] = target
            if any(delta):
                commands.append(dict(mono=time.monotonic(), target=target, delta=delta))
                goal[:] = predicted_robot_encoders(robot, delta)
                if mode[0] == "timeout":
                    enabled.clear()
                if mode[0] == "jump":
                    robot[0] += 50

    subscriber = rospy.Subscriber(
        "/simulation/controlSignal", controlSignal, command, queue_size=10
    )

    def feedback():
        frame = 0
        while running.is_set():
            if enabled.is_set():
                with lock:
                    for i, limit in enumerate((8, 500, 50, 0)):
                        robot[i] += max(-limit, min(limit, goal[i] - robot[i]))
                    message = Encoder()
                    for i, axis in enumerate(AXES):
                        setattr(message, "robotEncoder" + axis, robot[i])
                        setattr(message, "handleEncoder" + axis, handles[i])
                    measured = list(robot)
                    if mode[0] == 'ripple':
                        measured[2] += 7 if frame % 2 else -7
                        message.robotEncoderC = measured[2]
                    message.posX, message.posY, message.posZ = (
                        model_tip_from_robot_encoders(measured, 0)
                    )
                publisher.publish(message)
                frame += 1
            time.sleep(0.02)

    thread = threading.Thread(target=feedback, daemon=True)
    thread.start()
    results = {}
    try:
        for selected in cases or ("readonly", "normal", "ripple", "timeout", "jump", "stop", "preflight"):
            with lock:
                robot[:] = home if selected == "preflight" else work
                goal[:] = robot
                handles[:] = [9, 0, 1, 0]
                commands[:], packets[:] = [], []
                mode[0] = selected
                enabled.set()
            time.sleep(0.2)
            root = output / selected
            cmd = [
                "/usr/bin/python3",
                str(PACKAGE / "scripts/base_position_response.py"),
                "--output",
                str(root),
                "--baseline-s",
                "2",
                "--frequency",
                ".05",
                "--cycles",
                "1",
                "--simulation-only",
            ]
            if selected != "readonly":
                cmd += ["--execute", "--confirm-safe"]
            with (output / (selected + ".log")).open("w") as log:
                if selected == "stop":
                    process = subprocess.Popen(cmd, env=os.environ.copy(), stdout=log, stderr=log)
                    try:
                        deadline = time.monotonic() + 15
                        while not commands:
                            if process.poll() is not None or time.monotonic() > deadline:
                                raise RuntimeError("stop fixture never received motion")
                            time.sleep(0.01)
                        process.send_signal(signal.SIGINT)
                        process.wait(timeout=20)
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait()
                else:
                    process = subprocess.run(
                        cmd, env=os.environ.copy(), stdout=log, stderr=log, timeout=180
                    )
            r = json.loads((root / "result.json").read_text())
            with lock:
                sent, wire = list(commands), list(packets)
            assert all(not packet["forward"] and not packet["backward"] for packet in wire), wire
            assert all(packet["target"][3] == 0 for packet in wire), wire
            if selected == "readonly":
                assert r["status"] == "read_only_finished" and not wire, r
            elif selected in ("normal", "ripple"):
                assert (
                    process.returncode == 0
                    and r["status"] == "completed"
                    and not r.get("export_error")
                ), r
                assert all(c["delta"][3] == 0 for c in sent), sent
                assert all(any(c["delta"][i] for c in sent) for i in range(3)), sent
                snap = json.loads((root / "result/snapshot.json").read_text())
                assert (
                    snap["controller"] == "base_position_residual"
                    and snap["final_stable"]
                ), snap["reason"]
                assert snap["drive_vector"] == [1.0, 0.6, 0.4]
                assert len(list((root / "result").glob("*.pdf"))) == 5
                assert not json.loads((root / "result/metadata.json").read_text())[
                    "metrics"
                ]["sine"]["available"]
            elif selected == "stop":
                assert r["status"] == "stopped" and r["reason"] == "operator_stop" and sent, r
            elif selected == "preflight":
                assert r["status"] == "stopped" and "RCM" in r["reason"] and not wire, r
            else:
                # A feedback timeout is checked on the 50 ms control timer;
                # one in-flight fresh frame may legitimately produce a second
                # correction before the 0.5 s watchdog fires.  The invariant
                # is bounded publication followed by a stop, not an exact
                # packet count.
                assert r["status"] == "stopped" and 1 <= len(sent) <= 2, r
                assert ("feedback_timeout" if selected == "timeout" else "jump") in r[
                    "reason"
                ], r
            assert process.returncode == (0 if selected in ("readonly", "normal", "ripple") else 2), r
            if selected in ("normal", "ripple", "timeout", "jump", "stop"):
                snapshot = json.loads((root / "result/snapshot.json").read_text())
                assert snapshot["metadata"]["control_gear"] == 15, snapshot["metadata"]
                assert snapshot["metadata"]["data_source"] == "simulation", snapshot["metadata"]
                assert snapshot["metadata"]["feedback_topic"] == "/simulation/RobotState"
                assert snapshot["metadata"]["control_topic"] == "/simulation/controlSignal"
                assert snapshot["commands"][-1]["action"] == "hold", snapshot["commands"][-1]
                events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
                holds = [i for i, event in enumerate(events) if event["event"] == "hold"]
                assert holds and not any(event["event"] == "publish" for event in events[holds[-1] + 1:])
                assert not dict(rospy.get_master().getSystemState()[2][0]).get("/simulation/controlSignal")
            results[selected] = dict(
                status=r["status"],
                reason=r["reason"],
                changed_packets=len(sent),
                segments=r.get("segments"),
            )
            print(selected, results[selected], flush=True)
        code, message, system = rospy.get_master().getSystemState()
        assert code == 1, message
        assert not dict(system[0]).get("/robot/controlSignal"), system
        assert not dict(system[1]).get("/robot/RobotState"), system
        results["no_robot_topic_io"] = True
        results["hardware_connected"] = False
        (output / "result.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        running.clear()
        thread.join(timeout=2)
        subscriber.unregister()
        publisher.unregister()
        rospy.signal_shutdown("fixture ended")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", nargs="+", choices=("readonly", "normal", "ripple", "timeout", "jump", "stop", "preflight"))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(
        os.environ,
        ROS_MASTER_URI="http://127.0.0.1:%d" % port,
        ROS_IP="127.0.0.1",
        ROS_HOSTNAME="127.0.0.1",
    )
    with (output / "roscore.log").open("w") as log:
        master = subprocess.Popen(
            ["roscore", "-p", str(port)], env=env, stdout=log, stderr=log
        )
        try:
            deadline = time.monotonic() + 10
            while True:
                try:
                    if (
                        xmlrpc.client.ServerProxy(env["ROS_MASTER_URI"]).getPid(
                            "/fixture"
                        )[0]
                        == 1
                    ):
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("private master failed")
                time.sleep(0.1)
            process = subprocess.run(
                ["/usr/bin/python3", __file__, "--worker", str(output)] + (args.cases or []),
                env=env,
                timeout=500,
            )
            if process.returncode:
                raise RuntimeError("private fixture failed")
        finally:
            master.terminate()
            master.wait(timeout=8)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        validate(Path(sys.argv[2]), sys.argv[3:] or None)
    else:
        main()
