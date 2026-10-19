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

### 2026-10-17 整合修正：環境退化、成功判定、DN 尺度與 BC 驗收

分支 `integrate/tracking-fixes`（基於 `fix/motor-dn-diagnosis`）。完整問題對照、數據與待決事項見 [fixes_2026-10-17.md](fixes_2026-10-17.md)。執行環境同上（GTX 1050 3 GB、i7-9700）。

- 環境審計（`python -m flytrack_train audit-env`，60 秒回合）：舊世界＋舊判定中固定左轉 100% 成功（與教師相同）；新世界 curved 教師 100%、所有盲目策略 0%；BC 世界 `[curved, wander]` 教師 95%、盲目策略 0%，教師平均 ω 由 0.081 變為 −0.001。
- 教師校準重做（`runs/calibration/teacher_20261017.json`，種子 1.65e9 起）：20/20 成功，平均方位誤差 2.22°（1.48°～3.21°），20 個回合佈局不同。
- 譜半徑估計改為逐輪重新縮放的位移冪迭代，與稠密特徵值解差異約 1e-15；DN 輸入增益 100 下真實圖 ρ 不變（3517.85）。
- ROS：在 `flytrack-ros:0.1` 映像（Python 3.14、rclpy）掛載新程式做節點層級煙霧測試，佈局、事件、`target_mover` 軌跡重建與 legacy 參數均通過；Gazebo 閉環未執行。
- 小規模端到端驗證（VO seed 0，`runs/validation_20261017/`）：結果見 fixes 文件。

### 2026-10-18～19：兩台 GPU 主機與 #4、#19 的實驗

這段期間的重度工作都在 teammate 的主機上執行，證據放在 GitHub Issue 與 prerelease，不在本機 `runs/`：

| 項目 | 主機 | 結果 | 證據 |
| --- | --- | --- | --- |
| 主機基準與交叉驗證 | RTX 4060、RTX 4070 Laptop | 同版本指標相對差 < 1e-3；跨主機評估同一 artifact 相對差 < 3e-8；PPO 雙程序約 340–350 步／秒 | #21、release `host-calibration` |
| #4 縮小鏈 G0–G3（seed 0）與 G0–G2（seed 1） | 4060、4070 | 採用 G2：BC normal／occlusion 成功率 0.43／0.40（現行流程 0.25／0.15）；BC 後氣味／視覺比值仍 < 0.005 | #4、release `explore-4` |
| T3 重跑（V、seed 0，兩組 PPO 各 500k） | 4060 | 撞牆誘因修好（從 pretrain 開始，碰撞率 1.00 → 0.07），但 PPO 沒有改善 BC | #19、#35 |
| 遮蔽氣味 teacher 校準 | 4070（CPU） | 20%／50% 遮蔽：19/20、16/20，無碰撞 | #24、`docs/teacher/issue24/` |
| 測試 | 本機 i7-9700 | 2026-10-19 的 `main`：165 passed，`ruff check` 通過 | — |

### 2026-10-19：PM 接手後的 ROS 證據覆核

- [#44 的 PM review](https://github.com/Nues0913/Drosophila_Tracking_System/pull/44#pullrequestreview-5567304481)：完整差異審查後，在本機 `flytrack-ros:0.1`（映像 `ed4b8b4d8331`、Lyrical／Python 3.14.4）的一次性容器覆入 `916bed7` 的三個 Python 套件並重新 build。colcon 收集 4 項測試，正常 exit 0；只在容器副本改錯 3 處預期值後，3 failed／1 passed、exit 1。#44 已合併為 `a0c8cfb`，#42 關閉；policy/world 的空集合與獨立 launch_test 邊界不變。
- [#45 的 PM review](https://github.com/Nues0913/Drosophila_Tracking_System/pull/45#pullrequestreview-5567311087)：下載 release `explore-4` 的 CPU 首輪及 GPU 追加證據包，壓縮包 SHA-256 與全部 56＋36 個逐檔 hash 一致。以 rclpy 獨立解碼原始 db3、依最舊有效來源 sample 重算，CPU／GPU 完整端到端 p95 為 55.6233651／56.4599500 ms，active 為 54.1289413／55.08554345 ms；與報告一致，仍未達 50 ms。兩份五回合 bag 均 quick_check=ok、13 個 topic 有資料。
- 使用實際 rclpy.Future 及 production callback 獨立重現取消 future 的假 `NoneType.success` 錯誤，另開 #46，由原 ROS DRI 修正。這不代表真實服務失敗可以忽略。
- [#20 的後續 Decision](https://github.com/Nues0913/Drosophila_Tracking_System/issues/20#issuecomment-6260437085)：保留完整端到端門檻與頻率，先從既有 bag 做同 sample、同 wall-clock 的逐筆 stage 診斷；撤回先前將 simulation-clock age 與 wall time 相加的推論。
- 本次 **NOT RUN**：新增 Gazebo 閉環、GPU 推論、正式訓練、#23 一致性實驗與真值隔離 launch_test。上述延遲是既有資料的獨立重算，不是新增量測；本次容器均已移除。

### 已知限制

- ~~教師校準 20 回合結果逐位相同~~：2026-10-17 已在隨機化世界重做（見上）。
- 正式訓練預算：PPO 在本機 CPU 實測約 25 步／秒，2M 步約 22 小時；GTX 1050 3 GB 在 PPO 更新時 OOM，30 個訓練（6 變體 × 5 種子）不適合在本機單機完成；需使用者決定 GPU 主機或縮減預算。

測試或長工作只能依實際輸出更新為通過；啟動命令與程序存在不等同驗收完成。每階段後續成果需補入本表及 `requirements_traceability.md`。

## 權限與資料處理

只使用公開資料、公開模型與匿名映像來源；沒有讀取 `.env`、本機雲端憑證或 Docker 帳戶憑證。計畫書 PDF 封面含作者個人資料，不放進 repo。`runs/` 下的紀錄與 `data/raw/` 只存在執行主機上，不進 git；repo 只附上小型已驗證產物（子圖、前端、稽核 JSON）。
