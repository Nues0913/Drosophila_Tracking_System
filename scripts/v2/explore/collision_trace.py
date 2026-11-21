# ruff: noqa: E501, B023  (#138 exploratory analysis output lines)
"""#138 A2 diagnosis: where and how the occlusion collisions happen (described only).

cd <A2 worktree> && OMP_NUM_THREADS=4 PYTHONPATH=python python collision_trace.py N LABEL=ARTIFACT ...

The wind world's occlusion scenario, both motions, the first N pilot_held_out episodes of
training seed 100 (the evaluation's own episodes, so rows pair across artifacts). Each artifact
acts with its mode; beside it the DAgger teacher (OcclusionTeacher with the arena walls, the
labels' settings) is evaluated on the same observations and states, never acting. Per episode:
whether and when it collided (before, during or after the occlusion), the teacher's rule at the
collision (visible, odor present, odor lost, wall turn), the steps since the wall rule last
took over; and over the occluded steps, per rule: the share of steps, the student's turn sign
agreement with the teacher's label and the mean |turn difference|.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, "scripts/v2")
import evaluate_v2 as ev  # noqa: E402
from flytrack_v2.eval.records import episode_numbers  # noqa: E402
from flytrack_v2.eval.runner import V2Actor  # noqa: E402
from flytrack_v2.model.export import load_policy  # noqa: E402
from flytrack_v2.world.teacher import OcclusionTeacher, PrivilegedTeacher  # noqa: E402

N = int(sys.argv[1])
artifacts = [arg.split("=", 1) for arg in sys.argv[2:]]
protocol = yaml.safe_load(Path("configs/v2/evaluation_protocol.yaml").read_text())
seeds = list(episode_numbers(protocol, "pilot_held_out", 100, N))
device = torch.device("cpu")
CASES = {0: "visible", 1: "odor present", 2: "odor lost", 3: "wall turn"}

for label, path in artifacts:
    policy = load_policy(Path(path), device="cpu")
    for motion in ("curved", "wander"):
        world, _ = ev.build_world(policy.contract, "wind", motion, ["occlusion"], 0.75, protocol, N, device)
        teacher = OcclusionTeacher(PrivilegedTeacher(policy.contract), walls=world.walls)
        tstate = teacher.initial_state(N, device)
        actor = V2Actor(policy)
        obs = world.reset(None, torch.as_tensor(seeds))
        actor.reset(N, device)
        done = torch.zeros(N, dtype=torch.bool)
        hit_step = np.full(N, -1)
        hit_phase, hit_case, since_wall = [""] * N, [""] * N, np.full(N, -1)
        last_wall = np.full(N, -10_000)
        kind = [""] * N
        at_hit = np.full((N, 3), np.nan)  # student speed, teacher label speed, mean odor
        contact = world.agent_radius + world.target_radius
        seen_occlusion = torch.zeros(N, dtype=torch.bool)
        steps = {c: 0 for c in CASES}
        agree = {c: 0 for c in CASES}
        gap = {c: 0.0 for c in CASES}
        for step in range(1205):
            priv = world.privileged()
            action = actor.act(obs, None)
            label_t, tstate = teacher.act(
                priv, obs, priv["agent_velocity"], tstate, limit_acceleration=False
            )
            case = teacher.cases(priv, obs).long()
            live = ~done
            occluded = priv["occluded"].bool()
            seen_occlusion |= occluded
            for c in CASES:
                m = live & occluded & (case == c) if c else live & ~occluded
                if c and m.any():
                    steps[c] += int(m.sum())
                    s, t = action[m, 1], label_t[m, 1]
                    agree[c] += int((torch.sign(s) == torch.sign(t)).sum())
                    gap[c] += float((s - t).abs().sum())
            last_wall[(live & (case == 3)).numpy()] = step
            result = world.step(action)
            hit = (result.info["collision"] & live).numpy()
            for i in np.flatnonzero(hit):
                hit_step[i] = step
                hit_phase[i] = (
                    "during" if occluded[i] else ("after" if seen_occlusion[i] else "before")
                )
                hit_case[i] = CASES[int(case[i])]
                since_wall[i] = step - last_wall[i] if last_wall[i] >= 0 else -1
                touching = float(result.info["distance_m"][i]) <= contact + 0.02
                kind[i] = "target" if touching else "wall"
                at_hit[i] = (float(action[i, 0]), float(label_t[i, 0]), float(obs["odor"][i].mean()))
            done |= (result.terminated | result.truncated) & live
            obs = result.obs
            if bool(done.all()):
                break
        hits = hit_step >= 0
        phases = {p: sum(1 for i in np.flatnonzero(hits) if hit_phase[i] == p) for p in ("before", "during", "after")}
        cases = {c: sum(1 for i in np.flatnonzero(hits) if hit_case[i] == c) for c in CASES.values()}
        recent = since_wall[hits]
        for k in ("target", "wall"):
            idx = [i for i in np.flatnonzero(hits) if kind[i] == k and hit_phase[i] == "during"]
            if idx:
                v = at_hit[idx]
                print(f"   during occlusion, {k}: {len(idx)}; at the hit, student speed {np.median(v[:, 0]):.3f}, teacher label speed {np.median(v[:, 1]):.3f} m/s, mean odor {np.median(v[:, 2]):.3f}")
        print(f"== {label} {motion}: collisions {int(hits.sum())}/{N}, phase {phases}, teacher rule at collision {cases}")
        print(f"   steps since the wall rule last took over (collisions): median {np.median(recent[recent >= 0]) if (recent >= 0).any() else None}, never {int((recent < 0).sum())}")
        total = sum(steps[c] for c in (1, 2, 3))
        for c in (1, 2, 3):
            if steps[c]:
                print(f"   occluded, {CASES[c]:12s}: {steps[c] / total:5.1%} of steps, turn sign agrees {agree[c] / steps[c]:.2f}, mean |turn - teacher| {gap[c] / steps[c]:.3f} rad/s")
