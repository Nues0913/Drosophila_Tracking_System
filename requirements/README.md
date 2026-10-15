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
