# 官方 flyvis 前端執行紀錄

## 公開來源與權重

使用 [TuragaLab/flyvis v1.2.0](https://github.com/TuragaLab/flyvis/releases/tag/v1.2.0)；
官方安裝說明的 `flyvis download-pretrained` 指向作者公開 Google Drive 下載。
本次為避免使用上游程式內附的公開 API key，直接使用無登入的公開檔案下載路徑，
沒有讀取本機帳戶、token、金鑰或其他私密資料。

- 官方資料夾：<https://drive.google.com/drive/folders/15_8gPaVVJV6wGAspwkrdN8r-2NxMY4Kj>
- 公開檔案：`results_pretrained_models.zip`，3,417,042 bytes。
- SHA-256：`71c78d4070556a536b13b23ee3139cd2788aa2a9d07d430a223b4edead281db1`。
- checksum 出處：[v1.2.0 官方 downloader](https://github.com/TuragaLab/flyvis/blob/v1.2.0/flyvis_cli/download_pretrained_models.py)。
- 本機檔案與驗證：`data/flyvis/results_pretrained_models.zip`、`data/flyvis/pretrained_download_verification.json`。
- 模型：`data/flyvis/results/flow/0000/000`；選定官方模型 000，不以本研究測試集挑選模型。

2026-10-15 已完成公開下載、官方 checksum 驗證和安全路徑檢查後解壓。
此壓縮包包含官方 ensemble 的 50 個模型；本計畫共用固定的第 000 模型作凍結前端。

## 轉換與驗證契約

`scripts/export_flyvis.py` 載入官方目錄設定及 pretrained state dict，
拒絕非 passive point neuron／instantaneous graded release／ReLU 的動力學，
依公開模型參數展開每個節點的 bias、tau 與每條邊的 sign × count × strength。
光感受器依本專案六角座標重新排列，不依賴原模型陣列索引順序。

檢查使用灰階、中心閃光與移動正弦圖樣，每類 24 個控制步，
每控制步再重複到實際神經步長；全節點輸出與官方 `Network.forward` 比對。
另外載入真正的 `FrozenFlyvis` 部署程式，逐控制步核對雙眼獨立狀態與池化結果；
包含 CSR 疏鬆矩陣、控制／神經步長及眼別狀態排列。只有兩層 parity 都通過才原子替換輸出檔。
容忍值預設為 `atol=2e-5, rtol=2e-5`。

執行命令：

```bash
PYTHON_DOTENV_DISABLED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  FLYVIS_ROOT_DIR="$PWD/data/flyvis" .venv/bin/python scripts/export_flyvis.py \
  --network-dir "$PWD/data/flyvis/results/flow/0000/000" \
  --sensor-contract configs/sensor_contract.json \
  --output artifacts/frontend/frontend.npz
PYTHON_DOTENV_DISABLED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  FLYVIS_FRONTEND="$PWD/artifacts/frontend/frontend.npz" \
  .venv/bin/python -m pytest python/tests/test_frontend.py -q
```

## 本次執行狀態

2026-10-15 已完成轉換、官方 parity 與部署測試，產物在
`artifacts/frontend/frontend.npz`，來源、雜湊、數值誤差與 benchmark 的完整紀錄在
`artifacts/frontend/verification.json`。

| 驗證 | 結果 |
| --- | --- |
| 官方網路對展開公式，全節點最大絕對誤差 | 1.9073486e-6 |
| 官方網路對實際 CSR 部署狀態，最大絕對誤差 | 1.9073486e-6 |
| 官方網路對實際雙眼池化特徵，最大絕對誤差 | 3.5762787e-7 |
| 灰階／閃光／移動圖樣，各 24 控制步 | 通過 |
| 真前端測試：凍結、缺失眼狀態隔離、雙眼獨立 | 3 passed，1.46 秒 |
| 前端相關 ruff check／format check | 通過 |

初次轉換發現 flyvis 在 import 時依 CUDA 可用性建立模組層級 generator，
會與 CPU 轉換產生 device 衝突。匯出程式現在先僅對自身程序設定
`CUDA_VISIBLE_DEVICES=''`，再 import torch／flyvis；未更改主機或父程序設定。
同時設 `PYTHON_DOTENV_DISABLED=1`，避免 flyvis 自動搜尋載入私人 `.env`。
首次 import 後僅檢查工作區及父目錄 `.env` 的存在性：四個位置均不存在，
沒有可供該搜尋鏈讀取的 `.env`；未讀取任何檔案內容。

## CPU 吞吐實測

測量完整雙眼、每眼完整官方網路、每控制步 5 個神經步；固定灰階輸入，
3 次暖機後測 15 次，使用 `torch.no_grad()`。測量未包含模型建構、
世界渲染、中央網路、反向傳播或 ROS 延遲，因此不能當成整個系統的延遲證據。

| CPU threads | 環境 batch | 平均控制步 ms | p95 ms | 環境步／秒 |
| --- | --- | --- | --- | --- |
| 1 | 1 | 21.44 | 23.30 | 46.65 |
| 1 | 4 | 32.19 | 34.56 | 124.24 |
| 1 | 8 | 59.73 | 65.91 | 133.95 |
| 2 | 1 | 16.70 | 20.77 | 59.87 |
| 2 | 4 | 24.62 | 26.17 | 162.46 |
| 2 | 8 | 45.95 | 50.08 | 174.12 |

前端 inference 以 2 threads／batch 8 粗估，每 240k 環境步至少約 23 分鐘；
batch 1 約 67 分鐘。這是此主機量測推算的前端下限，不是完整訓練時間承諾。
CPU 上可先採 batch 4–8；正式訓練還需量測世界與中央網路成本後決定。
