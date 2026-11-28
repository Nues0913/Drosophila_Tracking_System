# FlyTrack: Connectome-Constrained Multisensory Target Tracking

[![CI](https://github.com/Nues0913/Drosophila_Tracking_System/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Nues0913/Drosophila_Tracking_System/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-formal--v2-2ea44f.svg)](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/formal-v2)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/MaleCNS%20data-CC%20BY%204.0-lightgrey.svg)](THIRD_PARTY_NOTICES.md)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch 2.9](https://img.shields.io/badge/pytorch-2.9-ee4c2c.svg)
![ROS 2](https://img.shields.io/badge/ROS%202-Gazebo-22314e.svg)

**English** · [繁體中文](README.zh-TW.md) · [Report](docs/v2/report.md) · [Release](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/formal-v2) · [Pretrained weights](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/research-v2-formal)

FlyTrack is a target-tracking agent whose controller is wired from the connectome of the adult *Drosophila* male central nervous system (MaleCNS v1.0). A recurrent rate network built on an 8,792-neuron subgraph receives compound-eye images, bilateral odor, and antennal wind, and drives forward speed and turning through descending neurons. We train it with imitation learning and recurrent PPO in GPU-batched 2D worlds, evaluate it under a pre-registered protocol, and deploy the same policy in ROS 2 / Gazebo.

The question we ask: **when the target disappears from view, can odor and wind keep a connectome-constrained controller on track?**

<p align="center">
  <img src="docs/assets/vo_vs_v_occlusion.gif" width="820" alt="The same occluded episode replayed with the multisensory model (left) and the vision-only model (right)">
</p>
<p align="center"><sub>One wind-world episode with the target hidden from 8 s to 38 s. Left: VO2 (vision + odor + wind) stays near the target; right: V2 (vision only) loses it. This is a single episode. Across the 1,000 formal occlusion episodes, whole-episode success was 19.9% for VO2 and 17.7% for V2 (see <a href="#results-formal-v2">Results</a>).</sub></p>

## Highlights

- **Connectome controller.** A type-thresholded MaleCNS subgraph (8,792 neurons, 185,261 directed edges) links 803 lobula columnar visual neurons, 148 olfactory receptor neurons, and 115 Johnston's organ wind neurons to five bilateral descending-neuron types. Synaptic signs and counts come from the data. The dynamics are a simplified rate model.
- **Grounded sensing.** A frozen [flyvis](https://github.com/TuragaLab/flyvis) early visual system (144 features per eye), an olfactory receptor code with complete and partial adaptation, and an antennal wind-direction model.
- **Training for weak cues.** Paired-observation DAgger supervises the policy on counterfactual odor and wind inputs. Recurrent PPO with a privileged critic and kickstarting then fine-tunes it. The deployed policy never sees the target position.
- **Fast simulation.** GPU-batched worlds with puff and filament plumes run 1,024 environments in parallel. Training throughput is 6–11× that of our v1 trainer on the same GPU.
- **Pre-registered evaluation.** Six model variants × five training seeds, 76,000 primary evaluation episodes, paired one-sided tests with Holm correction, and negative results reported as such.

## Results (formal-v2)

The six variants share the sensory encoders and the training pipeline: VO2 (vision + odor + wind), V2 (vision only), O2 (odor + wind only), RW2 (degree-preserving rewired graph), ER2 (random graph), and GRU2 (parameter-matched GRU). The gates were frozen before formal training ([protocol](configs/v2/evaluation_protocol.yaml)).

| Gate | Comparison | Result | Outcome |
| --- | --- | --- | --- |
| G-A behavior | VO2 curved-target success, normal scenario, ≥ 80% | 100.0% (puff world) · 98.8% (wind world) | ✅ pass |
| G-B multisensory | VO2 − V2, wind-world occlusion, ≥ +10 pp | +2.2 pp (19.9% vs 17.7%), Holm p = 0.555 | ❌ fail |
| G-C structure | VO2 noninferior to RW2, ER2, GRU2 (margin −5 pp) | ER2 +16.0 pp (pass) · RW2 +9.7 pp (p = 0.072) · GRU2 −14.2 pp | ❌ fail |
| G-D vs. v1 | VO2 − v1 connectome policy (VO-late) | +51.8 pp (100.0% vs 48.2%), Holm p = 0.001 | ✅ pass |
| G-E engineering | Training throughput vs. the v1 trainer, same GPU; CI | 6.06–11.13×; CI green | ✅ pass |

<p align="center">
  <img src="docs/assets/gates.png" width="440" alt="Registered paired comparisons with 95% t intervals over five seeds">
  &nbsp;
  <img src="docs/assets/occlusion_attribution.png" width="470" alt="VO2 tracking fraction by occlusion phase with odor-swap and wind-rotation interventions">
</p>
<p align="center"><sub>Left: registered paired comparisons (mean and 95% t interval over five seeds); filled markers pass. Right: VO2 tracking fraction by occlusion phase, without intervention and with the odor sides swapped or the wind rotated by 90° during the occlusion.</sub></p>

- **The policy responds to chemical cues.** Swapping the left and right odor during the occlusion cut VO2's tracking fraction from 57.0% to 14.2% (−42.8 pp); rotating the wind by 90° cut it to 33.2% (−23.8 pp). Both drops are Holm-significant.
- **Closer is not safer.** During the occlusion VO2 tracked the hidden target more of the time than V2 (57.0% vs 40.6%) but collided more often (41.7% vs 15.0% of episodes), so whole-episode success barely moved.
- **Why the odor signal is weak.** In our diagnostics, olfactory projection neurons receive input from both antennae, which cancels about 79% of the left–right odor contrast at the first synapse. By the steering neurons the contrast is more than five orders of magnitude smaller. The residual odor steering runs almost entirely through a two-synapse shortcut to DNb05.
- **Deployment.** A ROS 2 / Gazebo compatibility check in the puff world (seed-100 VO2 and V2 actors, 48 episodes) matched the 2D outcome in 48 of 48 paired episodes. Sensor-to-command p95 latency was 42.4 ms (VO2) and 42.1 ms (V2).

The [v2 report](docs/v2/report.md) (in Traditional Chinese) has the full analysis, including the exploration track that selected the formal recipe.

## Method

<p align="center">
  <img src="docs/assets/architecture.png" width="500" alt="System architecture: GPU-batched world, sensory encoders, connectome core, descending-neuron readout, and the two training stages">
</p>

| Component | What it does | Details |
| --- | --- | --- |
| Subgraph extraction | Downstream search from sensory seeds intersected with upstream search from descending neurons, with type-level synapse thresholds. Every required DN keeps ≥ 40% of its whole-brain input. | [subgraph.md](docs/v2/subgraph.md) |
| Connectome core | Fan-in-normalized signed weights; type-shared gains, time constants and biases; 52 bounded plastic gains on chemical-sense paths | [model.md](docs/v2/model.md) |
| Sensing | 721-ommatidium hexagonal eyes, frozen flyvis frontend, dual-adaptation ORN code, antennal wind | [interfaces.md](docs/v2/interfaces.md) |
| Readout | Turning from left–right differences of DNa01, DNa02, DNa03, DNb05 and DNp09; forward speed from DNp09 and DNb05 | [model.md](docs/v2/model.md) |
| Worlds | Puff world (v1-compatible) and wind world with filament plumes; occlusion, sensor noise, and odor decoys | [world.md](docs/v2/world.md) |
| Training | Stage 1: paired-observation DAgger (3 M steps). Stage 2: recurrent PPO (10 M steps) with a privileged critic, kickstarting, and occlusion auxiliary losses | [train.md](docs/v2/train.md) |
| Evaluation | Per-episode records, seed-level paired statistics, sensory-attribution interventions | [metrics.md](docs/v2/metrics.md) |

In the code and configs the puff world is called `v1compat` and the wind world `wind`.

## Getting started

### Requirements

- Linux x86_64 (native or WSL2) and Python 3.12 through [uv](https://docs.astral.sh/uv/).
- For training, an NVIDIA GPU with CUDA 12.6. The formal runs used GPUs from an RTX 2060 (6 GB, with activation checkpointing) to an RTX 3090. Evaluation also runs on the CPU.
- For deployment, Docker with Compose 2.30 or later. NVIDIA container support is optional.

The extracted subgraphs, the converted flyvis frontend, and the input normalization ship with the repository, so no connectome download is needed.

### Install

```bash
git clone https://github.com/Nues0913/Drosophila_Tracking_System.git flytrack && cd flytrack
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --require-hashes -r requirements/ml-cu126-py312.lock  # CPU only: ml-cpu-py312.lock
uv pip install --python .venv/bin/python --no-deps -e .
source .venv/bin/activate
python -m pytest -m "not slow and not gpu and not ros and not legacy and not bench" -q
```

### Evaluate a released policy

The [research-v2-formal](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/research-v2-formal) release holds the weights of all 60 formal checkpoints in one archive per training host: `gputw3090` (seeds 1–2), `4060` (seed 3), `2060` (seed 4), and `4070` (seed 5).

```bash
gh release download research-v2-formal -p 'v2_formal_2060_weights.tar.gz' -p 'v2_formal_2060_SHA256SUMS'
sha256sum -c v2_formal_2060_SHA256SUMS
mkdir -p runs/formal_weights && tar xzf v2_formal_2060_weights.tar.gz -C runs/formal_weights
python scripts/v2/evaluate_v2.py \
  --artifacts runs/formal_weights/VO2/seed_4/stage2/final \
  --seeds formal_evaluation --worlds wind --motions curved --episodes 2 \
  --no-sensitivity --device cpu --output runs/eval_demo
```

Each archive holds `<variant>/seed_<n>/stage{1,2}/final/`. This example takes about a minute on a CPU; use `--device cuda` and more episodes for real runs. The evaluator writes `episodes.jsonl` and `secondary.jsonl`; [metrics.md](docs/v2/metrics.md) defines the record format.

### Train

`scripts/v2/run_formal.py` runs the whole chain for each seed and all six variants: Stage 1, held-out evaluation, Stage 2, held-out evaluation. With `--smoke` it runs at toy size on a pilot seed, which takes about 13 minutes on an RTX 2060:

```bash
python scripts/v2/run_formal.py --host 2060 --seeds 100 --sha "$(git rev-parse HEAD)" --smoke --out runs/smoke
```

`--host` names an entry in [hosts.yaml](configs/v2/train/hosts.yaml), which sets the micro-batch size. The runner refuses a dirty checkout and writes the commit into every model card. A Stage 1 that ends with the low-speed health warning exits with code 3; the runner records it and continues, and at smoke size most graph models do.

### Reproduce formal-v2

The formal matrix trained seeds 1–5 with the [A3 recipe](configs/v2/train/a3_v15.yaml) under the frozen protocol. Print the plan with `--dry-run` first:

```bash
python scripts/v2/run_formal.py --host <host> --seeds <seeds> --sha "$(git rev-parse HEAD)" --out runs/v2_formal --dry-run
```

One seed (six variants, both stages, held-out evaluation) took about 17 hours on an RTX 2060. The 2D evidence, statistics, and per-host provenance of the published run are attached to [formal-v2](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/formal-v2). [train.md](docs/v2/train.md) covers resuming, retries, and packaging.

### Deploy in ROS 2 / Gazebo

```bash
docker compose --profile sim build
export ARTIFACT=/workspace/runs/formal_weights/VO2/seed_4/stage2/final
export SENSOR_CONTRACT=$ARTIFACT/sensor_contract.json
docker compose --profile sim up sim policy eval viz
```

The policy node subscribes only to the simulated camera and odor topics (plus wind in the wind world) and publishes `/cmd_vel` every 50 ms. It refuses an artifact whose hashes, timing, geometry, or sensor contract do not match. A v2 model must be paired with its own `sensor_contract.json`. Foxglove Studio can connect to `ws://localhost:8765`. Without NVIDIA container support, add `-f compose.yaml -f compose.cpu.yaml`.

`scripts/run_gazebo_matrix.py` runs paired Gazebo cells for several variants and seeds, for example `--variants VO2 V2 --seeds 4 --scenarios normal occlusion --episodes 6 --artifact-template 'runs/formal_weights/{variant}/seed_{seed}/stage2/final' --cpu`. Formal-v2 validated only the puff world in Gazebo, which is the default (`FLYTRACK_WORLD=v1compat`). The wind world is implemented in Gazebo but was not evaluated there ([world.md](docs/v2/world.md) §8).

## Repository layout

```text
python/flytrack_v2/         v2 model, GPU-batched worlds, DAgger and PPO trainers, evaluation
python/flytrack_core/       shared contracts, connectome core, flyvis frontend
python/connectome_extract/  MaleCNS download, inventory, and subgraph extraction
python/flytrack_sim/        v1 Gymnasium environment (formal-v1 baseline)
python/flytrack_train/      v1 trainers and evaluation (formal-v1 baseline)
python/tests/               pytest suite
scripts/v2/                 formal runner, evaluation, statistics, benchmarks, audits
configs/v2/                 sensor contract, frozen evaluation protocol, model and training configs
data/subgraphs/             extracted MaleCNS subgraphs (CC BY 4.0)
artifacts/                  converted flyvis frontend and fitted input normalization
ros2_ws/                    ROS 2 packages: world, policy node, evaluator, bringup
docker/                     ML and ROS images (Compose files are at the repository root)
docs/                       technical documentation and reports
```

## Limitations

- The recipe was selected on VO2 alone, which favors VO2 in G-B and G-C. Five seeds give limited statistical power.
- Same-type, same-side visual neurons share one receptive field, because MaleCNS has no per-cell retinotopy for these types.
- Reading forward speed from DNp09 and DNb05 is a modeling choice, not an established physiological result.
- 15 of the 30 formal Stage 1 chains raised a low-speed-variability health warning before PPO. The Gazebo check covered the puff world only.

## Project history

| Version | Date | Summary |
| --- | --- | --- |
| [formal-v1](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/formal-v1) | 2026-10-28 | 4,502-neuron connectome policy, CPU worlds, imitation learning ([report](docs/formal_v1_report.md)) |
| [v1-final](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/v1-final) | 2026-11-18 | Final state of v1, including follow-up research ([research-35](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/research-35)) |
| [formal-v2](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/formal-v2) | 2026-11-24 | Larger subgraph with wind, dual-adaptation olfaction, GPU worlds, two-stage training ([report](docs/v2/report.md)) |

[HISTORY.md](docs/HISTORY.md) explains where the development record (plans, progress logs, team workflow) is archived.

## Team

[@Nues0913](https://github.com/Nues0913) · [@MiLong9487](https://github.com/MiLong9487) · [@cweiting731](https://github.com/cweiting731)

## Citation

If you use FlyTrack, please cite this repository ([CITATION.cff](CITATION.cff)) together with the MaleCNS connectome and flyvis:

```bibtex
@software{flytrack2026,
  title   = {FlyTrack: Connectome-Constrained Multisensory Target Tracking},
  author  = {{FlyTrack contributors}},
  year    = {2026},
  version = {formal-v2},
  url     = {https://github.com/Nues0913/Drosophila_Tracking_System}
}
```

- S. Berg et al. Sexual dimorphism in the complete *Drosophila* male central nervous system connectome. *Cell* 189(18):5504–5526.e15, 2026. [doi:10.1016/j.cell.2026.08.015](https://doi.org/10.1016/j.cell.2026.08.015)
- J. K. Lappalainen et al. Connectome-constrained networks predict neural activity across the fly visual system. *Nature*, 2024. [doi:10.1038/s41586-024-07939-3](https://doi.org/10.1038/s41586-024-07939-3)

## License

The code is released under the [MIT License](LICENSE). The subgraphs in `data/subgraphs/` are derived from MaleCNS v1.0 and remain under CC BY 4.0. The converted frontend in `artifacts/` is derived from the flyvis pretrained models (MIT). [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) has the attributions and license texts.

## Acknowledgements

MaleCNS is a collaboration of FlyEM (HHMI Janelia), the University of Cambridge Department of Zoology, the MRC Laboratory of Molecular Biology, and Google Research. We thank the flyvis authors for releasing their pretrained models.
