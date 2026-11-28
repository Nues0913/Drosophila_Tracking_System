# Documentation

Most technical documents are written in Traditional Chinese. The [top-level README](../README.md) gives the English overview.

## v2 (current, formal-v2)

| Document | Contents |
| --- | --- |
| [v2/report.md](v2/report.md) | Formal-v2 report: methods, gates G-A to G-E, secondary analyses, exploration track, Gazebo check, limitations |
| [v2/大實驗計劃書.md](v2/大實驗計劃書.md) | The v2 study plan, confirmed before implementation; the frozen protocol cites its sections 3.9–3.11. It is a historical document, and the process documents it mentions are in the [archive](HISTORY.md#archived-process-documents) |
| [v2/subgraph.md](v2/subgraph.md) | MaleCNS subgraph extraction, thresholds, and checks |
| [v2/model.md](v2/model.md) | Connectome core, descending-neuron readout, and sensory encoding |
| [v2/interfaces.md](v2/interfaces.md) | Sensor contract v2 and the observation interface |
| [v2/world.md](v2/world.md) | Puff (`v1compat`) and wind worlds, in 2D and in Gazebo |
| [v2/train.md](v2/train.md) | DAgger and PPO training, the formal runner, packaging |
| [v2/metrics.md](v2/metrics.md) | Evaluation records and statistics |

The frozen evaluation protocol is [configs/v2/evaluation_protocol.yaml](../configs/v2/evaluation_protocol.yaml), and the formal training recipe is [configs/v2/train/a3_v15.yaml](../configs/v2/train/a3_v15.yaml).

## Shared and v1

| Document | Contents |
| --- | --- |
| [data_sources.md](data_sources.md) | MaleCNS download, file hashes, and how the data are interpreted |
| [biology_verification.md](biology_verification.md) | Literature check of the sensory and descending-neuron types used in the model |
| [formal_v1_report.md](formal_v1_report.md) | Formal-v1 report (the baseline for G-D) |

## Assets

`assets/` holds the figures used in the README. The architecture and result figures come from the course report, and the animation is cut from the project video.

## History

[HISTORY.md](HISTORY.md) describes the development timeline and where the process documents (plans, progress logs, team workflow, diagnostics) are archived.
