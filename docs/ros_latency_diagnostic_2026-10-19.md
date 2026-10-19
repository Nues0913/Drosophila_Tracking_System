# Issue #22: same-clock latency diagnosis

This report re-analyzes the existing five-episode CPU-inference and GPU-inference
bags from the `83f57d1` validation run. Both inference configurations were run
on the same RTX 4070 host; this is not a comparison between two hardware hosts.
It does not change the 50 ms end-to-end gate, control frequency, sensor
contract, or evaluation rules. No simulation or GPU workload was started for
this analysis.

## Method and integrity

For every inference command, the analyzer selects the valid contributing sensor
sample with the oldest same-host `CLOCK_MONOTONIC` source timestamp. It measures
one continuous path from that sample to the `cmd_vel` publish timestamp and
splits it into adjacent intervals:

`source → ingress → forward → policy callback received → control callback start → command published`

The stage durations telescope to the end-to-end duration for each command; all
4,707 included traces passed this check and the clock-domain/order checks. Both
SQLite bags passed `PRAGMA quick_check`. The analysis retains all valid full-bag
traces, including traces outside an assigned episode, and reports active-window
traces separately. The 17 CPU and 15 GPU unassigned traces were retained in
full-bag statistics; none was active. There were no invalid causal traces.

Each table below reports exact bag-derived quantiles rounded to 0.01 ms. The
complete per-command rows and summaries are in the compressed JSON evidence.
The script is `scripts/diagnostics/analyze_ros_latency_bag.py`; run it from the
repository root in the existing ROS Lyrical/Python 3.14 analysis environment,
after sourcing ROS and the workspace:

```sh
source /opt/ros/lyrical/setup.bash
source ros2_ws/install/setup.bash
python3 scripts/diagnostics/analyze_ros_latency_bag.py \
  runs/ros_validation/issue22_83f57d1/multi5/bags/flytrack_20261019T031946_320499Z \
  /tmp/cpu-latency-diagnostic.json
python3 scripts/diagnostics/analyze_ros_latency_bag.py \
  runs/ros_validation/issue22_83f57d1/gpu5/bags/flytrack_20261019T034508_171480Z \
  /tmp/gpu-latency-diagnostic.json
```

Before choosing the oldest contributor, the script validates the clock domain,
presence, and ordering of all six monotonic points for **every** valid
contributing sensor. If any contributor is invalid, it rejects that command.
Regression tests cover a non-oldest cross-domain sample, reversed stage order,
missing timestamps, a command with no valid contributors, and selection of the
oldest source on a valid command.

## End-to-end distributions

| 推論設定 / 範圍 | n | p50 ms | p90 ms | p95 ms | p99 ms | max ms | ≥50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU full bag | 2,352 | 37.54 | 49.39 | 55.62 | 139.95 | 200.36 | 206 (8.8%) |
| CPU active | 2,335 | 37.51 | 49.09 | 54.13 | 112.73 | 173.04 | 190 (8.1%) |
| GPU full bag | 2,355 | 38.32 | 51.29 | 56.46 | 109.37 | 339.08 | 300 (12.7%) |
| GPU active | 2,340 | 38.23 | 51.09 | 55.09 | 96.56 | 339.08 | 285 (12.2%) |

Neither inference configuration meets the unchanged 50 ms p95 gate. These recomputed p95 values
match the values previously verified from the bags. Full-bag percentiles retain
all inference traces, including episode-boundary traces; active means the
command simulation timestamp lies inside a scenario's tracking-to-completed
window.

## Per-episode active distributions

| Episode | CPU n / p95 ms / ≥50 ms | GPU n / p95 ms / ≥50 ms |
| ---: | --- | --- |
| 0 | 495 / 58.14 / 50 (10.1%) | 495 / 52.89 / 61 (12.3%) |
| 1 | 452 / 59.29 / 43 (9.5%) | 456 / 54.31 / 60 (13.2%) |
| 2 | 452 / 53.02 / 32 (7.1%) | 452 / 53.99 / 70 (15.5%) |
| 3 | 561 / 51.26 / 35 (6.2%) | 562 / 67.05 / 57 (10.1%) |
| 4 | 375 / 52.57 / 30 (8.0%) | 375 / 56.55 / 37 (9.9%) |

The highest episode p95 is GPU episode 3 at 67.05 ms; CPU episode 1 is 59.29 ms.
Counts differ slightly because the captured CPU and GPU inference streams differ.

## Slowest five percent: stage composition

The slowest group is `ceil(5% × n)` commands ranked by each command's complete
latency. Stage shares below are computed by summing stages over that same group;
they are not sums of separate stage percentiles.

| 推論設定 / 範圍 | n | Source→ingress | Ingress→forward | Forward→received | Received→control | Control→published | Oldest source in slowest group |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CPU full bag | 118 | 1.33 ms (1.4%) | 0.32 (0.3%) | 29.09 (29.5%) | 34.29 (34.8%) | 33.61 (34.1%) | odor 61, left 46, right 11 |
| CPU active | 117 | 1.36 (1.5%) | 0.33 (0.4%) | 29.29 (33.3%) | 22.22 (25.3%) | 34.72 (39.5%) | odor 58, left 54, right 5 |
| GPU full bag | 118 | 1.18 (1.3%) | 0.29 (0.3%) | 27.65 (31.2%) | 30.13 (34.1%) | 29.23 (33.0%) | left 72, odor 41, right 5 |
| GPU active | 117 | 1.19 (1.5%) | 0.30 (0.4%) | 28.31 (36.5%) | 17.78 (22.9%) | 30.07 (38.7%) | left 66, odor 43, right 8 |

Across the two active slowest-five-percent groups, `forward→received` and
`control→published` dominate. `received→control` is also material, especially
on CPU. The first two sensor-relay stages together contribute under 2% of the
slowest-group wall time. A per-episode breakdown is retained in the JSON: the
largest active slow-tail stage was `control→published` for CPU episodes 0, 2,
3, and 4 and GPU episodes 0, 1, 2, and 4; it was `forward→received` for CPU
episode 1 and GPU episode 3.

## Cold-start commands

The first active command of each episode is listed separately here and remains
included in the official distributions above.

| 推論設定 | Episode 0 | Episode 1 | Episode 2 | Episode 3 | Episode 4 |
| --- | --- | --- | --- | --- | --- |
| CPU total ms | 111.23 | 31.94 | 31.17 | 31.07 | 37.67 |
| GPU total ms | 339.08 | 37.77 | 32.75 | 47.39 | 37.58 |

Episode 0 is a pronounced cold-start outlier on both hosts. CPU's 111.23 ms is
mostly `forward→received` (85.55 ms); GPU's 339.08 ms is mostly
`control→published` (251.21 ms), with another 86.58 ms before the policy
callback receives the oldest sample. These are retained, not filtered from
the reported p95 or maximum.

## Interpretation and next diagnostic

The trace localizes delay but cannot identify its internal cause. In particular,
`forward→received` includes both ROS delivery and policy-executor queueing. The
policy node currently uses a `SingleThreadedExecutor`; its sensor callbacks
decode/update buffers and its timer callback runs `PolicyRuntime.tick()` on the
same thread. `control→published` spans the complete tick, action handling, and
command publish. The existing `inference_latency_ms` metric measures only
`PolicyRuntime.tick()`'s tensor construction, model call, finite checks, and
device-to-CPU copies; its p95 was 24.76 ms CPU and 24.19 ms GPU. That metric
cannot be joined to individual trace rows, so it does not establish how much of
each slow command is model execution.

The current evidence supports investigating callback scheduling and the
policy-tick internals. It does not support selecting a code optimization yet.
PM approved this bounded diagnostic under Issue #49. Its default-off probes
may be prepared, but replay must wait until this PR's contributor-validation
fixes have been reviewed and merged. The approved replay budget is at most 10
minutes for each CPU/GPU inference configuration on the same RTX 4070 host,
including warmup and profile-off/on runs; total replay time is at most 20
minutes and engineering work is about 60 minutes. No executor, QoS, frequency,
or contract change is authorized by that profiler decision.

The bag firmware/code basis is `83f57d1`; inspection confirms the policy,
world sensor timing, and interface message implementations used here are
unchanged on current `main` `525db6a`. The later changes in those paths are the
pytest collection setup and the independent `target_mover` cancellation fix.

Evidence files (`gzip` of the complete JSON outputs, including every analyzed
command row):

| File | SHA-256 |
| --- | --- |
| `docs/evidence/issue22-latency/cpu.json.gz` | `1c5d743c585ce9b2631718fb55396c0d93ff6d1497ee537256e2a7302f41a401` |
| `docs/evidence/issue22-latency/gpu.json.gz` | `9f30e3e7986349a012afa4a9723c723accafc3890eeebb90fa428966e8bcd1dd` |
