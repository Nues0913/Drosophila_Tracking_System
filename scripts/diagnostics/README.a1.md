# A1 diagnostics

These tools read trained artifacts and existing evaluations. They do not train,
change policies or modify evaluation inputs. Run from the repository root with
the project's Python environment, `PYTHONPATH=python`, `PYTHONNOUSERSITE=1`,
`OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`. Use one CPU process at a time and
the memory limit specified by the experiment driver.

## Episode replay

```bash
python scripts/diagnostics/episode_trace.py \
  --artifact <runs>/bc/VO-late/seed_0/actor \
  --config configs/train/evaluate.yaml --scenario occlusion \
  --lesion none --episodes 0,2,5-7 --device cpu \
  --output <runs>/traces/VO-late/seed_0/occlusion/none < /dev/null
```

Use a separate output directory for each invocation. `episodes.json` and
`episodes.csv` describe the episodes selected in that invocation. Step records
are streamed to `trace/{variant}_s{seed}_{scenario}_{lesion}_ep{episode}.jsonl.gz`.
`analysis_notes.json` records the input/source hashes, artifact card, conventions
and trace hashes. Episode numbers are sorted and deduplicated before replay.

| Output | Fields |
| --- | --- |
| Step JSONL | Zero-based `step`; decision `time_s`, `agent_pose`, `target_pose`, `teacher_action`, `target_distance_m`, `target_bearing_rad`, `target_visible`, `occluded`; policy `action`; resulting `applied_action`, `collision`, `post_time_s`, `post_agent_pose`, `post_target_pose`; non-GRU `current_mass` |
| Episode JSON/CSV | All lesion-analysis metrics; `collision_types`, `collision_time_s`, `descriptors`, `trace`; non-GRU fractions, `current_total`, `current_segments` |
| `current_total`, each current segment | `steps`, three-category `mass`, three-category `fraction` |
| `descriptors.occlusion` | `steps`, `mean_v_m_s`, `mean_abs_omega_rad_s`, `abs_net_turn_rad`, `total_abs_turn_rad`, `straightness`, `onset_heading_target_angle_rad`, `mean_heading_old_target_angle_rad`, `min_wall_clearance_m`, `windows` |
| `descriptors.whole_episode` | `mean_policy_minus_teacher_v_m_s`, `abs_omega_policy_teacher_ratio`, `mean_policy_minus_teacher_omega_rad_s`, `mean_signed_bearing_rad`, `median_distance_m`, `last_10s_mean_distance_m` |

Decision labels come from reset info for the first action and the preceding step
info thereafter. EpisodeMetrics consumes resulting step info exactly as in
`lesion_analysis.py`. Current segmentation follows the **decision** occlusion
flag. A boundary-crossing action is assigned to the flag seen at its start.

Speeds, turn descriptors and policy/teacher comparisons use the current step's
`info["applied_action"]`: it and the decision-time teacher action are clipped
using the same previous action, speed bounds and acceleration limits. The field
names containing `policy` therefore describe the applied policy action. Step
traces retain both raw policy `action` and `applied_action`. Turn integrals use
actual elapsed time. Straightness uses sampled
agent positions, including the terminal point. Wall clearance measures the
distance from the agent disk to the rectangular wall boundary, and can be
negative at contact. The last ten seconds use decision-time distance samples.
Signed bearing is an arithmetic mean of wrapped bearings, not a circular mean.
Angles are radians unless a field explicitly says degrees.

For multiple occlusion windows the old target position resets at each onset.
The aggregate straightness is total window endpoint displacement divided by total
window path length. The aggregate onset angle describes the first window;
`windows` retains all onset angles and per-window results. No observed occlusion
yields `descriptors.occlusion = null`. Zero denominators yield null. Unavailable
collision geometry yields the literal `"NOT AVAILABLE"`; simultaneous contacts
are retained in a list. Missing wall geometry yields null clearance. GRU outputs
have no current fields (and keep lesion-analysis's `analyzed_steps = 0`).

## Statistics and failure selection

```bash
python scripts/diagnostics/a1_summary.py select-failures \
  --f3 <runs>/evaluation/formal/episodes.json --output <runs>/a1/selection < /dev/null

python scripts/diagnostics/a1_summary.py lesion \
  --input <runs>/lesion_analysis \
  --f3 <runs>/evaluation/formal/episodes.json \
  --posthoc <runs>/evaluation/gru_posthoc/episodes.json \
  --output <runs>/a1/lesion < /dev/null

python scripts/diagnostics/a1_summary.py traces \
  --input <runs>/traces --failures <runs>/a1/selection/failure_cases.json \
  --output <runs>/a1/traces < /dev/null
```

`--input` accepts an episode JSON file or a directory searched recursively for
`episodes.json`. `--output` also accepts `-o`. Both statistics commands accept
`--allow-partial`; otherwise they require the complete declared matrix. Lesion
statistics expect VO, VO-late, RS and posthoc GRU; three scenarios (normal,
occlusion, conflict); three lesions; five seeds; episodes 0–19. Trace current
statistics expect the 300 occlusion/none episodes for VO, VO-late and RS. The
selected failure manifest identifies exactly which episodes enter failure tables,
so additional current replays do not change the failure sample. Only lesion none
traces enter either trace table. Identical overlapping replays are deduplicated;
conflicting records raise an error.

All commands write a primary JSON (`failure_cases.json`, `lesion.json`, or
`traces.json`), a CSV for each table, `<command>_summary.md` (three decimal
places), and `<command>_provenance.json`, where `<command>` is `select-failures`,
`lesion`, or `traces` (for example, `lesion_summary.md` and
`lesion_provenance.json`). We recommend a different `--output` directory for
each subcommand. Shared directories preserve each subcommand's reports;
`failure_cases.json` retains its name for downstream drivers and `traces` input.
JSON and CSV keep full precision for subsequent checks.
Nested fields are flattened with dots in CSV; lists remain JSON strings.

| Command | Tables |
| --- | --- |
| `select-failures` | `cases`: category, source identity, requested seed, replacement note, all source F3 metrics |
| `lesion` | `seed_means`, `lesion_summary`, `paired_seed_differences`, `paired_comparisons`, `consistency`, `missing_cells` |
| `traces` | `current_seed_means`, `current_summary`, `current_paired_seed_differences`, `current_paired_comparisons`, `failure_cases`, `failure_ranges`, `missing_current_cells`, `missing_failure_cases` |

Selection first reserves each available seed's minimum qualifying episode, then
fills empty slots from the next seed in cyclic ascending order, taking its next
unused minimum episode. It raises if a variant/category has fewer than five
eligible episodes overall. The manifest includes this rule and all replacements.

Each seed contributes one episode mean. Differences pair the same seeds, with
signs `none - vision`, `none - odor`, or `occluded - non_occluded`.
Every bootstrap output row calls the shared `bootstrap_mean` with a **fresh**
`np.random.default_rng(1301)` and 10,000 draws. This differs from the formal
evaluator's shared RNG stream and makes results invariant to row ordering.
Undefined values are excluded with explicit counts, including episodes ending
before occlusion. Partial results identify every missing cell/episode and report
the actual seed and episode counts. Do not treat them as the completed analysis.

Lesion-none agreement checks use formal F3 for VO/VO-late/RS and the posthoc
reference for GRU. Every observed row must match a reference identity; missing or
duplicate references raise. Legacy `collision`, `bearing_error_deg`, and
`tracking_fraction` aliases are accepted. Provenance records `program_version.git_head`
using `git rev-parse HEAD` in the script's repository, or `"NOT AVAILABLE"` if Git
or HEAD is unavailable. `program_version.files` always records SHA-256 for all
Python files under `python/flytrack_core/model/`, the metrics, teacher,
environment, worlds, configuration and evaluation modules, `scripts/lesion_analysis.py`,
and the three A1 Python scripts, using paths relative to the repository root.
Trace summaries also hash the replay manifests.
