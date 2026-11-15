# 已驗證 ML 環境的依賴鎖定

`ml-cu126-py312.lock` 固定 2026-10-15 本機實際安裝、`uv pip check` 通過的
ML 環境：Linux x86_64、Python 3.12.15、PyTorch 2.9.1+cu126、
torchvision 0.24.1+cu126、flyvis 1.2.0，以及 extract/train/visual/dev 的傳遞依賴。
這不是跨平台通用 lock，也不適用 ROS Python 3.14 環境。

`ml-cu126-py312.txt` 是安裝後版本快照；`.in` 把 torch/torchvision 換成官方
CUDA 12.6 wheel 的明確 URL，避免把一般套件交由 PyTorch 索引的舊版本解析。
`.lock` 由 uv 0.13.1 產生並包含下載雜湊。沒有私人索引或帳戶憑證。

重建到乾淨 Python 3.12 venv：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --require-hashes \
  -r requirements/ml-cu126-py312.lock
uv pip install --python .venv/bin/python --no-deps -e .
uv pip check --python .venv/bin/python
```

目前的相容性證據：`runs/validation/ml_environment.json`、
`runs/validation/pytest_with_frontend.xml`、`docs/frontend_execution.md`。
這組依賴已在本機通過相容性與真實前端測試；乾淨容器重建仍需以實際 Docker 建置結果核對。

# ROS Python 3.14 映像的版本快照（2026-11-16）

ROS 端沒有 uv 產生的雜湊鎖定，這裡記的是**安裝後的版本快照**，和 `ml-cu126-py312.txt` 同類，用來核對重建的映像是否相同。

- `ros-py314-cpu.txt`：`flytrack-ros:0.1`（從 main `8e19c9b` 建置，映像 `sha256:71b5992b…`），venv `/opt/flytrack-ros`（繼承系統套件）的 `pip freeze`，不含以 editable 安裝的本專案；`pip check` 沒有相依衝突。torch 是 `2.9.1+cpu`。
- `ros-py314-cu126.txt`：同一份程式改用 `--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126` 建置（main `28c7021`，映像 `sha256:aff01f79…`，#110），torch 是 `2.9.1+cu126`，其餘相同。
- `ros-apt.txt`：兩個映像相同的關鍵 apt 套件版本（ROS Lyrical、Gazebo、foxglove bridge、launch testing、numpy、PyYAML 等）。
- 基底映像：`osrf/ros:lyrical-desktop-full-resolute@sha256:956cc9eeb9f2dcae30c659ea1c2ad985dc8fc6f25a4a70d8f485106311b0c8c6`（取自建置紀錄）。

這些版本只在 Linux x86_64、上述基底映像下核對過。要逐位元重建，請固定基底映像的 digest。
