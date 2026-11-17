"""Scripted collisions for the Gazebo contact check (#116 G0f, run (b); no policy).

Usage (ROS image, with the world launched with contact_check:=true and no policy container):
  python3 scripts/diagnostics/contact_probe_driver.py PLAN.json [EPISODES]
Stands in for the policy: serves /connectome_policy_node/reset (so scenario_manager runs its
normal reset gate) and, while an episode is active, drives /cmd_vel by one manoeuvre per episode
from PLAN.json (default_plan() when the file does not exist; it is then written there):
  {"kind": "wall", "wall": "north", "incidence_deg": 60, "side": 1}
      drive at constant speed (never braking: the Gazebo agent tips forward onto its body when it
      decelerates) to a lead-in point LEAD_M before the wall, then follow the straight line that
      meets the wall at the given angle from its normal (side +1/-1 picks which way along it).
      wall_approach() has the geometry; the analysis reports the incidence actually reached.
  {"kind": "target", "offset_m": 0.0}
      pursue the target; offset_m != 0 aims beside it (perpendicular to the line of sight) so the
      agent's side, not its front, touches the target.
With EPISODES, PLAN.json.done is written (completed episode count) once that many episodes have
completed, so a runner can wait for it. Each manoeuvre and its staging point/heading are
logged as JSON lines on /sim/contact_probe.
The geometric rule ends the episode; whatever the agent touched first is what the analysis
reports (on the way to a staging point it can also touch the target).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

WALLS = ("north", "east", "south", "west")
NORMAL_INTO_WALL = {"north": (0, 1), "east": (1, 0), "south": (0, -1), "west": (-1, 0)}
SPEED_M_S = 0.3
TURN_GAIN = 2.0
OMEGA_MAX = 1.0
HEADING_TOLERANCE_RAD = math.radians(1.0)
LEAD_M = 1.5
LEAD_DEPTH_M = 1.0  # the lead-in point is at least this far inside the collision boundary
LEAD_TOLERANCE_M = 0.15
LINE_GAIN = 2.0  # rad per metre of cross-track error


def default_plan() -> list[dict]:
    """36 manoeuvres: 7 angles x 2 sides x 2 rounds at the walls (in turn), 8 at the target."""
    plan = []
    for _round in range(2):
        for angle in (0, 15, 30, 45, 60, 70, 80):
            for side in (1, -1):
                wall = WALLS[len(plan) % 4]
                plan.append({"kind": "wall", "wall": wall, "incidence_deg": angle, "side": side})
    plan += [{"kind": "target", "offset_m": 0.0}] * 4
    plan += [{"kind": "target", "offset_m": 0.25}, {"kind": "target", "offset_m": -0.25}] * 2
    return plan


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def first_wall(x: float, y: float, heading: float, half: float, radius: float) -> str | None:
    """The wall the agent's centre reaches first (its collision boundary is half - radius)."""
    limit = half - radius
    dx, dy = math.cos(heading), math.sin(heading)
    hits = []
    for wall, (nx, ny) in NORMAL_INTO_WALL.items():
        along = dx * nx + dy * ny
        if along <= 1e-9:
            continue
        hits.append(((limit - (x * nx + y * ny)) / along, wall))
    return min(hits)[1] if hits else None


def wall_approach(manoeuvre: dict, half: float = 5.0, radius: float = 0.18, lead=None):
    """(lead-in x, y, hit x, y, heading): the agent centre's straight path from the lead-in point
    to the wall's collision boundary, centred on the wall's midpoint. The path is LEAD_M long, or
    longer at grazing angles so the lead-in point stays LEAD_DEPTH_M from the boundary."""
    nx, ny = NORMAL_INTO_WALL[manoeuvre["wall"]]
    tx, ty = -ny, nx  # along the wall, counter-clockwise from the inward normal
    incidence = math.radians(manoeuvre["incidence_deg"]) * manoeuvre["side"]
    heading = wrap(math.atan2(ny, nx) + incidence)
    if lead is None:
        lead = max(LEAD_M, LEAD_DEPTH_M / math.cos(incidence))
    dx, dy = math.cos(heading), math.sin(heading)
    depth = half - radius
    along = lead * (dx * tx + dy * ty) / 2
    hx, hy = depth * nx + along * tx, depth * ny + along * ty
    return hx - lead * dx, hy - lead * dy, hx, hy, heading


def target_aim(agent, target, offset_m: float) -> float:
    """Heading to the target, or to a point offset_m beside it (left of the line of sight)."""
    dx, dy = target[0] - agent[0], target[1] - agent[1]
    distance = math.hypot(dx, dy)
    if distance < 1e-9:
        return agent[2]
    ax = target[0] - offset_m * dy / distance
    ay = target[1] + offset_m * dx / distance
    return math.atan2(ay - agent[1], ax - agent[0])


def command(agent_yaw: float, heading: float, drive: bool) -> tuple[float, float]:
    """(v, omega): proportional turn; forward speed only when ``drive``."""
    error = wrap(heading - agent_yaw)
    omega = max(-OMEGA_MAX, min(OMEGA_MAX, TURN_GAIN * error))
    return (SPEED_M_S if drive else 0.0), omega


def wall_step(state: dict, agent, manoeuvre: dict) -> tuple[float, float]:
    """One control step of a wall manoeuvre at constant forward speed; ``state`` keeps the phase."""
    lx, ly, hx, hy, heading = wall_approach(manoeuvre)
    if state.setdefault("phase", "lead") == "lead":
        if math.hypot(lx - agent[0], ly - agent[1]) > LEAD_TOLERANCE_M:
            return command(agent[2], math.atan2(ly - agent[1], lx - agent[0]), True)
        state["phase"] = "line"
    # Signed cross-track error (left of the line positive) steers back onto the line.
    cross = -(agent[0] - hx) * math.sin(heading) + (agent[1] - hy) * math.cos(heading)
    correction = max(-math.pi / 3, min(math.pi / 3, -LINE_GAIN * cross))
    return command(agent[2], heading + correction, True)


def main() -> None:
    import rclpy
    from flytrack_interfaces.msg import ScenarioEvent
    from geometry_msgs.msg import PoseStamped, Twist
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    plan_path = Path(sys.argv[1])
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else None
    done_path = plan_path.with_name(plan_path.name + ".done")
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        plan = default_plan()
        plan_path.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")

    def yaw(q):
        return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    class Driver(Node):
        def __init__(self):
            super().__init__(
                "contact_probe_driver",
                parameter_overrides=[Parameter("use_sim_time", value=True)],
            )
            self.cmd = self.create_publisher(Twist, "/cmd_vel", 1)
            self.log = self.create_publisher(String, "/sim/contact_probe", 100)
            self.create_service(Trigger, "/connectome_policy_node/reset", self._reset)
            self.create_subscription(ScenarioEvent, "/world/scenario", self._event, 100)
            self.create_subscription(PoseStamped, "/world/agent/pose", self._agent, 10)
            self.create_subscription(PoseStamped, "/world/target/pose", self._target, 10)
            self.agent = self.target = None
            self.active_episode = None
            self.state = {}
            self.completed = set()
            self.create_timer(0.05, self._tick)

        def _reset(self, _request, response):
            self.agent = self.target = None
            self.state = {}
            response.success = True
            return response

        def _event(self, message):
            if message.active and self.active_episode != message.episode_id:
                self.active_episode = message.episode_id
                self.state = {}
            elif not message.active:
                self.active_episode = None
                phase = json.loads(message.parameters_json).get("phase")
                if phase == "completed":
                    self.completed.add(message.episode_id)
                    if episodes is not None and len(self.completed) >= episodes:
                        done_path.write_text(f"{len(self.completed)}\n", encoding="utf-8")

        def _agent(self, message):
            p = message.pose
            self.agent = (p.position.x, p.position.y, yaw(p.orientation))

        def _target(self, message):
            p = message.pose
            self.target = (p.position.x, p.position.y, yaw(p.orientation))

        def _tick(self):
            if self.active_episode is None or self.agent is None or self.target is None:
                return
            manoeuvre = plan[self.active_episode % len(plan)]
            if not self.state.get("logged"):
                record = {
                    "episode_id": self.active_episode,
                    "manoeuvre": manoeuvre,
                    "start": self.agent,
                    "target_start": self.target,
                }
                if manoeuvre["kind"] == "wall":
                    record["lead_hit_heading"] = wall_approach(manoeuvre)
                self.log.publish(String(data=json.dumps(record)))
                self.state["logged"] = True
            if manoeuvre["kind"] == "wall":
                v, omega = wall_step(self.state, self.agent, manoeuvre)
            else:
                aim = target_aim(self.agent, self.target, manoeuvre["offset_m"])
                v, omega = command(self.agent[2], aim, True)
            twist = Twist()
            twist.linear.x, twist.angular.z = float(v), float(omega)
            self.cmd.publish(twist)

    rclpy.init()
    node = Driver()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
