# Project history

## Timeline

| Date | Milestone |
| --- | --- |
| 2026-10-15 | MaleCNS v1.0 tables downloaded |
| 2026-10-28 | [formal-v1](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/formal-v1): formal matrix, 2D and Gazebo evaluation, report ([formal_v1_report.md](formal_v1_report.md)) |
| 2026-11-15 | [research-35](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/research-35): follow-up research and engineering after formal-v1 |
| 2026-11-18 | [v1-final](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/v1-final): final state of v1 (also kept on the `release/v1` branch); the v2 plan is confirmed |
| 2026-11-18 to 11-23 | v2 implementation, pilot runs, and the occlusion exploration track that selected the A3 recipe |
| 2026-11-24 | [formal-v2](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/formal-v2): six variants × five seeds, gates G-A to G-E, Gazebo compatibility check ([v2/report.md](v2/report.md)) |
| 2026-11-28 | `main` reorganized as the public showcase; process documents archived |

## Releases

| Tag | Contents |
| --- | --- |
| `formal-v2` | Formal v2 results and deployment check (latest) |
| `research-v2-final` | Formal 2D evaluation records and five-seed statistics |
| `research-v2-formal` | Weights and training evidence of the formal matrix, one archive per host |
| `research-v2-pilot`, `research-v2-wp6` | v2 pilot runs; ROS 2 / Gazebo evidence |
| `v1-final`, `formal-v1`, `research-35`, `posthoc-gru-85`, `explore-4` | v1 results, follow-up work, and exploratory artifacts |

## Archived process documents

Before `main` was turned into the showcase, the repository also held the team's working documents: the agent-team workflow, progress logs, follow-up plans, requirement traceability, GPU host notes, diagnostic notes from October 2026, the v2 status log, and health checks. They are preserved unchanged at:

- tag [`team-archive-2026-11-28`](https://github.com/Nues0913/Drosophila_Tracking_System/tree/team-archive-2026-11-28)
- branch [`archive/team-docs`](https://github.com/Nues0913/Drosophila_Tracking_System/tree/archive/team-docs)

Some code comments, frozen configs, and protocol files still name these documents (for example `docs/status.md`, `docs/dn/…`, `docs/gpu_hosts.md`, `後續計畫.md`). Those files are hashed into model cards and frozen protocols, so they were left unchanged. Look such paths up in the archive tag. Links in the remaining Markdown documents already point to the archive.

The issue tracker and pull requests keep the full discussion behind each decision. Issue numbers such as #35, #138, or #144 in the documents refer to them.
