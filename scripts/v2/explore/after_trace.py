"""#138 A2: speeds before, during and after the occlusion (described only).

cd <worktree> && OMP_NUM_THREADS=4 PYTHONPATH=python \
    python scripts/v2/explore/after_trace.py N LABEL=ARTIFACT ...

The wind world's occlusion scenario, both motions, the first N pilot_held_out episodes of
training seed 100; each artifact acts with its mode, the occlusion teacher labels the same
states (never acting). Per phase (before, during, after the occlusion; collided episodes stop
at the hit): the student's mean speed, the teacher label's, and the mean distance to the
target; after the occlusion also split by distance (<= 1 m, 1-2 m, > 2 m).
"""

import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, "scripts/v2")
import evaluate_v2 as ev  # noqa: E402
from flytrack_v2.eval.records import episode_numbers  # noqa: E402
from flytrack_v2.eval.runner import V2Actor  # noqa: E402
from flytrack_v2.model.export import load_policy  # noqa: E402
from flytrack_v2.world.teacher import OcclusionTeacher, PrivilegedTeacher  # noqa: E402

STEPS = 1205


def trace(policy, motion: str, n: int, seeds: list[int], protocol: dict) -> dict:
    world, _ = ev.build_world(
        policy.contract, "wind", motion, ["occlusion"], 0.75, protocol, n, "cpu"
    )
    teacher = OcclusionTeacher(PrivilegedTeacher(policy.contract), walls=world.walls)
    state = teacher.initial_state(n, "cpu")
    actor = V2Actor(policy)
    obs = world.reset(None, torch.as_tensor(seeds))
    actor.reset(n, "cpu")
    done = torch.zeros(n, dtype=torch.bool)
    seen = torch.zeros(n, dtype=torch.bool)
    sums: dict[str, list[float]] = {}
    for _ in range(STEPS):
        privileged = world.privileged()
        action = actor.act(obs, None)
        label, state = teacher.act(
            privileged, obs, privileged["agent_velocity"], state, limit_acceleration=False
        )
        occluded = privileged["occluded"].bool()
        seen |= occluded
        offset = privileged["target_pose"][:, :2] - privileged["agent_pose"][:, :2]
        distance = offset.norm(dim=-1)
        live = ~done
        after = live & ~occluded & seen
        masks = {
            "before": live & ~occluded & ~seen,
            "during": live & occluded,
            "after": after,
            "after d<=1": after & (distance <= 1),
            "after 1<d<=2": after & (distance > 1) & (distance <= 2),
            "after d>2": after & (distance > 2),
        }
        for key, mask in masks.items():
            if mask.any():
                total = sums.setdefault(key, [0.0, 0.0, 0.0, 0.0])
                total[0] += float(mask.sum())
                total[1] += float(action[mask, 0].sum())
                total[2] += float(label[mask, 0].sum())
                total[3] += float(distance[mask].sum())
        result = world.step(action)
        done |= (result.terminated | result.truncated) & live
        obs = result.obs
        if bool(done.all()):
            break
    return sums


def main() -> None:
    n = int(sys.argv[1])
    protocol = yaml.safe_load(Path("configs/v2/evaluation_protocol.yaml").read_text())
    seeds = list(episode_numbers(protocol, "pilot_held_out", 100, n))
    for label, path in (arg.split("=", 1) for arg in sys.argv[2:]):
        policy = load_policy(Path(path), device="cpu")
        for motion in ("curved", "wander"):
            print(f"== {label} {motion}")
            for key, (steps, student, teacher, distance) in trace(
                policy, motion, n, seeds, protocol
            ).items():
                print(
                    f"   {key:13s}: steps {int(steps):6d}  student {student / steps:.3f}  "
                    f"teacher {teacher / steps:.3f} m/s  distance {distance / steps:.2f} m"
                )


if __name__ == "__main__":
    main()
