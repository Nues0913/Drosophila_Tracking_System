"""Scripted braking for the Gazebo agent's forward pitch (#140 E11; no policy).

Usage (ROS image, with the world launched and no policy container):
  python3 scripts/diagnostics/brake_pitch_probe.py OUTPUT.jsonl EPISODES
Stands in for the policy: serves /connectome_policy_node/reset (so scenario_manager runs its
normal reset gate) and, while an episode is active, repeats one cycle on /cmd_vel:
  drive at V_MAX for DRIVE_S, steering towards the arena centre (so the agent stays clear of the
  walls), then command zero speed for BRAKE_S. The drive plugin limits the deceleration to its
  max_linear_acceleration, so each stop is the hardest braking any policy can ask for.
Every agent pose received during an active episode is written to OUTPUT.jsonl with the phase,
the commanded speed and the pitch and roll from its orientation. OUTPUT.jsonl.done is written
(completed episode count) once EPISODES episodes have completed. summarize() reads the file back.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

V_MAX = 0.5
DRIVE_S = 2.0
BRAKE_S = 1.5
TURN_GAIN = 2.0
OMEGA_MAX = 1.0


def pitch_roll(q) -> tuple[float, float]:
    """Pitch (positive nose down, for a forward-x body) and roll of a quaternion, in radians."""
    w, x, y, z = q
    sin_pitch = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return math.asin(sin_pitch), roll


def phase_at(elapsed_s: float) -> str:
    return "drive" if elapsed_s % (DRIVE_S + BRAKE_S) < DRIVE_S else "brake"


def summarize(path: Path) -> dict:
    """Largest |pitch| and |roll| per episode and phase, and over the whole run, in degrees."""
    episodes: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        cell = episodes.setdefault(row["episode_id"], {"samples": 0})
        cell["samples"] += 1
        for key in ("pitch", "roll"):
            name = f"max_abs_{key}_deg_{row['phase']}"
            cell[name] = max(cell.get(name, 0.0), abs(math.degrees(row[key])))
    overall = {}
    for cell in episodes.values():
        for name, value in cell.items():
            if name.startswith("max_abs_"):
                overall[name] = max(overall.get(name, 0.0), value)
    return {
        "episodes": {str(k): v for k, v in sorted(episodes.items())},
        "overall": overall,
        "samples": sum(cell["samples"] for cell in episodes.values()),
        "cycle": {"v_max": V_MAX, "drive_s": DRIVE_S, "brake_s": BRAKE_S},
    }


def main() -> None:
    import rclpy
    from flytrack_interfaces.msg import ScenarioEvent
    from geometry_msgs.msg import PoseStamped, Twist
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from std_srvs.srv import Trigger

    output = Path(sys.argv[1])
    episodes = int(sys.argv[2])
    done_path = output.with_name(output.name + ".done")
    sink = output.open("a", encoding="utf-8")

    class Driver(Node):
        def __init__(self):
            super().__init__(
                "brake_pitch_probe",
                parameter_overrides=[Parameter("use_sim_time", value=True)],
            )
            self.cmd = self.create_publisher(Twist, "/cmd_vel", 1)
            self.create_service(Trigger, "/connectome_policy_node/reset", self._reset)
            self.create_subscription(ScenarioEvent, "/world/scenario", self._event, 100)
            self.create_subscription(PoseStamped, "/world/agent/pose", self._agent, 50)
            self.agent = None
            self.active_episode = None
            self.started_s = None
            self.v_cmd = 0.0
            self.completed = set()
            self.create_timer(0.05, self._tick)

        def _now(self) -> float:
            return self.get_clock().now().nanoseconds * 1e-9

        def _reset(self, _request, response):
            self.agent = None
            response.success = True
            return response

        def _event(self, message):
            if message.active and self.active_episode != message.episode_id:
                self.active_episode = message.episode_id
                self.started_s = self._now()
            elif not message.active:
                self.active_episode = None
                phase = json.loads(message.parameters_json).get("phase")
                if phase == "completed":
                    self.completed.add(message.episode_id)
                    if len(self.completed) >= episodes:
                        sink.flush()
                        done_path.write_text(f"{len(self.completed)}\n", encoding="utf-8")

        def _agent(self, message):
            p, q = message.pose.position, message.pose.orientation
            self.agent = (
                p.x,
                p.y,
                math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2)),
            )
            if self.active_episode is None or self.started_s is None:
                return
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            pitch, roll = pitch_roll((q.w, q.x, q.y, q.z))
            row = {
                "episode_id": self.active_episode,
                "t": stamp,
                "phase": phase_at(stamp - self.started_s),
                "v_cmd": self.v_cmd,
                "x": p.x,
                "y": p.y,
                "z": p.z,
                "pitch": pitch,
                "roll": roll,
            }
            sink.write(json.dumps(row) + "\n")

        def _tick(self):
            twist = Twist()
            if self.active_episode is not None and self.agent is not None:
                x, y, yaw = self.agent
                if phase_at(self._now() - self.started_s) == "drive":
                    error = math.atan2(-y, -x) - yaw
                    error = math.atan2(math.sin(error), math.cos(error))
                    twist.linear.x = V_MAX
                    twist.angular.z = max(-OMEGA_MAX, min(OMEGA_MAX, TURN_GAIN * error))
            self.v_cmd = float(twist.linear.x)
            self.cmd.publish(twist)

    rclpy.init()
    node = Driver()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        sink.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--summarize":
        print(json.dumps(summarize(Path(sys.argv[2])), indent=1))
    else:
        main()
