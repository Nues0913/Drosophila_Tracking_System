# Third-party notices

FlyTrack's own code is under the [MIT License](LICENSE). This repository also redistributes material derived from the two third-party works below, which keep their own licenses. The software dependencies listed at the end are installed separately and are not redistributed here.

## MaleCNS v1.0 connectome (CC BY 4.0)

**Used in:** `data/subgraphs/` (`main`, `v2_main`, `v2_cx_lat`, `olfactory_v2`, `olfactory_v2_nokc`) and the summary files in `artifacts/data_execution/`.

**Source:** MaleCNS v1.0, the connectome of the adult *Drosophila melanogaster* male central nervous system, from the [official download page](https://male-cns.janelia.org/download/). We used three of the flat-connectome tables released under `flyem-male-cns/v1.0/connectome-data/flat-connectome/`: body annotations (minconf 0.5), body neurotransmitters, and connectome weights (minconf 0.5).

**Creators:** the MaleCNS collaboration of FlyEM (HHMI Janelia Research Campus), the University of Cambridge Department of Zoology, the MRC Laboratory of Molecular Biology, and Google Research.

**License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). The material is provided as is, without warranties, as stated in the license.

**Changes we made:** we selected neurons by cell type, extracted bounded downstream and upstream subgraphs, applied type-level synapse-count thresholds, and saved the selected neurons, connections, synapse counts, and consensus neurotransmitter labels as NumPy and JSON files. The extraction code is in `python/connectome_extract/`, and every `manifest.json` records the SHA-256 of the source tables. The subgraphs are not the complete connectome and should not be used as a substitute for it.

**Citation:**

> S. Berg et al. Sexual dimorphism in the complete *Drosophila* male central nervous system connectome. *Cell* 189(18):5504–5526.e15, 2026. doi:[10.1016/j.cell.2026.08.015](https://doi.org/10.1016/j.cell.2026.08.015). Preprint: bioRxiv, doi:[10.1101/2025.10.09.680999](https://doi.org/10.1101/2025.10.09.680999).

The derived subgraphs in `data/subgraphs/` are distributed under CC BY 4.0 with this attribution.

## flyvis pretrained visual system (MIT)

**Used in:** `artifacts/frontend/frontend.npz` and `artifacts/frontend_v2/frontend.npz`, created by `scripts/export_flyvis.py` from the official flyvis 1.2.0 pretrained model `results/flow/0000/000` (checkpoint `chkpt_00000`). The parameters are converted to NumPy arrays without retraining. `artifacts/frontend_v2/verification.json` records the export command, the checkpoint SHA-256, and the numerical parity check against the flyvis network.

**Source:** [TuragaLab/flyvis](https://github.com/TuragaLab/flyvis).

**Citation:**

> J. K. Lappalainen, F. D. Tschopp, S. Prakhya, M. McGill, A. Nern, K. Shinomiya, S. Takemura, E. Gruntman, J. H. Macke, and S. C. Turaga. Connectome-constrained networks predict neural activity across the fly visual system. *Nature*, 2024. doi:[10.1038/s41586-024-07939-3](https://doi.org/10.1038/s41586-024-07939-3).

**License text:**

```text
MIT License

Copyright (c) 2023 Janne K. Lappalainen, Fabian D. Tschopp, Mason McGill, Jakob H. Macke, Srinivas C. Turaga

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Software dependencies (not redistributed)

These are installed from their own distributions by the lock files in `requirements/` or by the Docker images. The licenses below were read from the installed package metadata or the upstream repositories; check the upstream projects for the authoritative terms.

| Component | Version | License |
| --- | --- | --- |
| PyTorch | 2.9.1 | BSD-3-Clause |
| torchvision | 0.24.1 | BSD-3-Clause |
| NumPy | 2.x | BSD-3-Clause (bundled parts under 0BSD, MIT, Zlib, CC0-1.0) |
| pandas | 2.x | BSD-3-Clause |
| PyArrow | 18–23 | Apache-2.0 |
| PyYAML | 6.x | MIT |
| Gymnasium | 1.x | MIT |
| TensorBoard | 2.x | Apache-2.0 |
| flyvis | 1.2.0 | MIT |
| neuprint-python | 0.6.x | BSD-3-Clause (Copyright HHMI) |
| ROS 2 (rclpy and core packages) | Lyrical | Apache-2.0 |
| Gazebo (gz-sim) and ros_gz | — | Apache-2.0 |
| foxglove_bridge | — | MIT |
