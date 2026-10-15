# 完整計畫執行紀錄

記錄實際執行過的環境與驗證證據；進度摘要見 [status.md](status.md)。不能以冒煙測試或單一模型代替正式實驗。下表 `runs/` 路徑只存在原執行主機（不進 git）。

## 執行環境

- Ubuntu 26.04.1 LTS，Linux 7.0.0-38-generic，x86_64。
- Intel Core i7-9700，8 核心；RAM 約 15 GiB，初查可用約 4.9 GiB；swap 約 19 GiB。
- NVIDIA GTX 1050 3 GiB，驅動 580.178.04，驅動通訊正常。初查顯存約 2955/3072 MiB 被既有應用占用；不終止使用者的桌面或遊戲。
- Docker daemon 29.9.0、Compose 5.6.0 可用。主機磁碟可用約 267 GiB。
- ML 使用工作區 `.tools/` 下管理的 Python 3.12.15 與 `.venv/`。公開 PyPI 安裝 uv；不修改系統 Python 或讀取私人套件索引設定。
- PyTorch 使用 2.9.1 的 CUDA 12.6 發行版，安裝後須實際核對 Pascal 架構與 CUDA 運算。CUDA 發行版依據：[PyTorch 官方安裝表](https://pytorch.org/get-started/previous-versions/)。

## 執行證據與當前工作

| 項目 | 證據／產物 | 狀態 |
| --- | --- | --- |
| MaleCNS 三份官方表格 | `data/raw/download_manifest.json`、`download_verification.json` | 下載及大小、ETag、框架、本地 SHA-256 核對完成 |
| 官方 flyvis 預訓練包 | `data/flyvis/pretrained_download_verification.json`、`artifacts/frontend/verification.json` | 下載、SHA-256 與轉換 parity 完成（見 `docs/frontend_execution.md`） |
| ML 依賴 | `.venv/`、`runs/validation/ml_freeze.txt` | 已安裝；ML／ROS 共存與鎖版尚未完成 |
| 真實 schema／型別盤點與圖 | `data/subgraphs/main/`、`artifacts/data_execution/graph_verification.log` | 萃取完成：2569 神經元、14165 邊；`scripts/verify_subgraph.py` 全部檢查通過；兩次萃取 10 個產物位元相同。匯流神經元僅 4 個；LC9、LC11 各只保留 1 個 |
| 測試與 lint | `FLYVIS_FRONTEND=artifacts/frontend/frontend.npz pytest python/tests` | 2026-10-15 14:2x：66 passed（含真實前端）；`ruff check` 通過；`ruff format --check` 通過 |
| 參數預算 | `runs/experiment_configs/parameter_budget.json` | VO=6794、V=6794、O=6794、VO-late=6766、GRU=6831、RS=6794，均在 ±5% 內 |
| 小規模流程 | `runs/validation/smoke_ppo.log`、`smoke_chain_*.log`、`runs/smoke_chain/` | VO seed 0 縮小版 pretrain(視覺/氣味)→BC→PPO 全部跑完，各階段權重有限且有更新；CPU 每階段約 15–60 s、RSS 約 2.4–2.7 GB。非正式比較 |
| ML／ROS 映像 | `docker images` | `flytrack-ml:0.1`、`flytrack-ros:0.1` 建置完成；colcon 與 Gazebo 閉環尚未執行 |

### 2026-10-15 修正：中央遞迴初始化發散

`configs/train/smoke.yaml` 的 PPO 在第一個 rollout 即因策略輸出 NaN 失敗。根因：邊權重 = 號誌 × 突觸數（萃取門檻 60，中位數 92）× α，固定 `alpha_init=0.01` 使真實圖的遞迴矩陣譜半徑約 32，ReLU 動態每控制步放大約 2×10⁴ 倍，第 9 步溢位。修正：新增 `init_spectral_radius`（預設 0.9），初始化時以冪迭代求 |W| 的 Perron 根，α 取 `min(alpha_init, 0.9/ρ)`；真實圖得 α≈2.56×10⁻⁴。參數數不變；六個變體同規則各自計算（RS 依其重連後的圖）。回歸測試 `test_initial_recurrence_is_contractive_for_large_synapse_counts`。

### 已知限制

- 教師校準（`runs/calibration/teacher.json`）20 回合結果逐位相同：`evaluate.yaml` 關閉 domain randomization，圓形軌跡只有起始相位不同，幾何上等價。這份資料不能用來校準方位解碼門檻，需改用有變異的校準情境。
- 正式訓練預算：PPO 在本機 CPU 實測約 25 步／秒，2M 步約 22 小時；GTX 1050 3 GB 在 PPO 更新時 OOM，30 個訓練（6 變體 × 5 種子）不適合在本機單機完成；需使用者決定 GPU 主機或縮減預算。

測試或長工作只能依實際輸出更新為通過；啟動命令與程序存在不等同驗收完成。每階段後續成果需補入本表及 `requirements_traceability.md`。

## 權限與資料處理

只使用公開資料、公開模型與匿名映像來源；沒有讀取 `.env`、本機雲端憑證或 Docker 帳戶憑證。計畫書 PDF 封面含作者個人資料，不放進 repo。`runs/` 下的紀錄與 `data/raw/` 只存在執行主機上，不進 git；repo 只附上小型已驗證產物（子圖、前端、稽核 JSON）。
