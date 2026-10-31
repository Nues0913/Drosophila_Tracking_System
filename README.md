# 果蠅連接體多感官追蹤系統（FlyTrack）

以 flyvis 預訓練早期視覺前端（方案 A）加上 MaleCNS v1.0 中央子圖組成感測策略：先在 2D 模擬器訓練，再匯出模型，最後透過 ROS 2 Lyrical／Gazebo Jetty 做閉環評估。研究問題是比較六個變體（V、O、VO、VO-late、GRU、RS）在相同感測前端之後的追蹤表現。

> **目前進度（2026-10-31）：** 正式規格是 no-KC＋P2/B3、pretrain → BC/DAgger。團隊選 F：保留全部已登記比較，如實報告 VO 的有限氣味反應與負面結果；PPO／P3 留待後續研究。**正式訓練和 2D 正式評估都已在 main `35432c3` 完成並驗收**（release `formal-v1`）：VO、VO-late 對 V 的遮蔽優越性都不成立，沒有變體達到 G3 的 normal 成功率 0.8，GRU／RS 對照組退化（[#85](https://github.com/Nues0913/Drosophila_Tracking_System/issues/85)）。詳細進度見 [docs/status.md](docs/status.md)，總追蹤 [#28](https://github.com/Nues0913/Drosophila_Tracking_System/issues/28)，協作規則見 [AGENT_TEAM_WORKFLOW.md](AGENT_TEAM_WORKFLOW.md)。

## 組員快速上手

### 0. 需求

- Linux x86_64：原生 Ubuntu，或 Windows 上的 WSL2。Colab 也可以，見 [docs/gpu_hosts.md](docs/gpu_hosts.md)。
- NVIDIA GPU，驅動支援 CUDA 12.6。GTX 1050 3 GB 在 PPO 更新時會 OOM，建議 8 GB 以上（未實測下限）。
- 約 5 GB 磁碟空間；只有重新萃取子圖時才需要額外 1.2 GB 原始資料。

### 1. 取得程式與環境

```bash
git clone <repo-url> flytrack && cd flytrack
curl -LsSf https://astral.sh/uv/install.sh | sh     # 已安裝 uv 可略過
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --require-hashes -r requirements/ml-cu126-py312.lock
uv pip install --python .venv/bin/python --no-deps -e .
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

lock 檔固定 Linux x86_64、Python 3.12、PyTorch 2.9.1+cu126（細節見 [requirements/README.md](requirements/README.md)）。

### 2. 已隨 repo 附上的產物

不需要重新下載原始資料或轉換前端就能開始訓練：

| 路徑 | 內容 |
| --- | --- |
| `data/subgraphs/olfactory_v2_nokc/` | 正式 no-KC 子圖（4502 神經元、31682 邊）、來源 manifest 與盤點報表 |
| `data/subgraphs/main/` | 歷史子圖（2569 神經元、14165 邊），保留供舊證據使用 |
| `artifacts/frontend/` | 轉換後的 flyvis 前端 `frontend.npz` 與 parity 驗證 |
| `artifacts/data_execution/*.json` | 候選型別掃描、原始 schema、子圖獨立驗證結果 |

### 3. 確認環境正常

```bash
pytest python/tests                    # 全部應通過（自動使用 artifacts/frontend）
ruff check python scripts
python -m flytrack_train audit-env --config configs/train/evaluate.yaml      # teacher 須成功、盲目策略須失敗
python -m flytrack_train ppo --config configs/train/smoke.yaml --device cuda   # 約 10 秒的小流程
```

### 4. 正式訓練

正式啟動以 #28 的 PM Decision 與共同 main SHA 為準。#77 已修正有效預算：有效 VO 為 14,546，GRU hidden=16 加 bottleneck=18 的 effective 為 14,904（相對 VO +2.46%），六個變體都在 ±5% 內，不能把零學習率參數當成可更新參數。修正內容已由 PR #79 合併（main `35432c3`）；下列是正式矩陣的執行方式。

```bash
python scripts/prepare_experiments.py --device cuda --jobs 2
cat runs/experiment_configs/parameter_budget.json      # 六變體 BC effective（lr>0）須在有效 VO 的 ±5% 內（#77）
bash runs/experiment_configs/run_all.sh                # 6 變體 × 5 種子，每個為 pretrain → BC/DAgger
python -m flytrack_train evaluate --config configs/train/evaluate.yaml --device cuda
python -m flytrack_train evaluate --config configs/train/evaluate_wander.yaml --device cuda   # 次要 wander，另列
python -m flytrack_train evaluate --config configs/train/evaluate_registered.yaml --device cpu  # 補上 VO-late 的已登記比較，不重跑模擬
```

正式訓練鏈到 BC/DAgger 為止，最終 actor 是 `runs/bc/{variant}/seed_{seed}/actor`。PPO 依 [#35](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35) 的 Decision 移出正式鏈，保留為研究項目；`--algorithm ppo` 或 `es` 可以在 BC 之後加一個階段，但這種結果不屬於正式矩陣。

`run_all.sh` 呼叫 `scripts/run_experiments.py`，開始前先審計 BC（以及有加 RL 階段時的 PPO／ES）的世界（固定動作策略成功率須 ≤10%）。正式 BC 後的 G4 與 DN 失敗分類見下段。工具預設同時跑 3 組，正式分工則固定傳入 `--jobs 2`；每組內階段依序執行，中斷後會驗證既有產物與 gate，再決定略過或續跑。額外參數會傳下去，例如 `bash runs/experiment_configs/run_all.sh --seeds 2` 或 `--variants VO --seeds 0 --jobs 1`。環境模擬在 CPU 子行程平行執行（`--workers`，預設依 CPU 數分配），預訓練資料同一種子只收集一次（`runs/datasets/`）。

O／VO／VO-late 在 BC 後要求氣味左右交換的平均 |Δω| ≥1e−3 rad/s 且至少是同 chain odor-pretrain 起點的 3 倍；V／GRU／RS 只記錄模態診斷。終端 BC 的轉向品質失敗保留數值並繼續評估，工程失敗則讓整個矩陣在階段交界停止，`--keep-going` 也不能略過。完整規則與揭露要求見 [協定凍結依據](docs/protocol_freeze_2026-10-28.md)。

WSL2 預設只分到主機約一半記憶體；16 GB 筆電建議在 Windows 的 `C:\Users\<帳號>\.wslconfig` 寫入 `[wsl2]`、`memory=12GB`，再於 PowerShell 執行 `wsl --shutdown` 後重開 Ubuntu。筆電訓練時請插電並使用最佳效能電源模式。

先跑單一變體、單一種子確認穩定性與速度，再分工跑完整矩陣。分工方式與注意事項見 [docs/gpu_hosts.md](docs/gpu_hosts.md#分工跑正式訓練)，誰跑哪些組以 PM 開的 GPU 執行 Issue 為準（[AGENT_TEAM_WORKFLOW.md](AGENT_TEAM_WORKFLOW.md#3-算力與重度測試分配)）。`runs/` 不進 git；訓練完成後請把各自的 `runs/<variant>/seed_<n>/actor` 與 log 交回整合。

## 已確認的實作決策

- 主模態為視覺＋嗅覺；溫度感覺留待主模型完成後再評估。
- 早期視覺採方案 A，所有模型共用同一份凍結前端與固定 ORN 轉換。沒有真正的 flyvis 前端產物時，模型建立、訓練與部署會報錯。
- V／O 使用相同完整中央圖與相同參數數，只關閉另一模態輸入。VO-late 使用分離的遞迴狀態、模態子圖和晚期 DN 讀出，型別參數共享。
- GRU 與 RS 比較中央網路結構；研究結論限定在共同感測前端之後的比較。RS 保持入出度、密度和號誌比例，保留原邊的參數共享標籤。
- actor 只接收 `vision`、`odor`、`valid`；教師、critic、獎勵與評估可以讀真值，真值不進入 actor。
- 中央遞迴初始化把 |W| 的譜半徑壓到 `init_spectral_radius`（0.9），否則真實突觸數會讓 ReLU 動態發散；接進運動 DN 的邊另有固定增益 `motor_input_gain`（100），避免方位訊號到 DN 時被一起壓小（見 [MODEL_NOTES](python/flytrack_core/model/MODEL_NOTES.md)）。
- 每回合隨機化目標繞圈方向、半徑與起始位置（2D 與 Gazebo 共用），成功須在 2 m 內**且**朝向目標 45° 內；訓練與正式評估前以 `audit-env` 確認固定動作策略無法成功（見 [SIM_NOTES](python/flytrack_sim/SIM_NOTES.md)）。
- Gazebo 先評估直接移轉的 2D 模型；追加微調另立實驗。
- 核心只依賴 NumPy／PyTorch，ROS 與 Gymnasium 是外層，模組按責任拆分（見下方「目錄與責任」）。

## 目錄與責任

```text
configs/                         感測契約、萃取、模型、訓練與情境設定
python/connectome_extract/       官方表格下載、inventory、路徑式子圖萃取
python/flytrack_core/            契約、神經模型、感測、羽流、軌跡與評估計算
python/flytrack_sim/             2D Gymnasium 環境與特權教師
python/flytrack_train/           預訓練、BC／DAgger、PPO、ES、評估及匯出
python/tests/                    pytest
ros2_ws/src/flytrack_interfaces/ ROS 自訂訊息
ros2_ws/src/flytrack_policy/     感測訂閱、模型推論、看門狗及控制命令
ros2_ws/src/flytrack_world/      氣味、目標、情境、干擾和回合重置
ros2_ws/src/flytrack_eval/       唯讀評估、rosbag 與真值隔離測試
ros2_ws/src/flytrack_bringup/    launch、bridge、差速機器人與場地
docker/                          ML／ROS 兩個映像
scripts/                         flyvis 轉換、實驗設定產生、子圖驗證及主機轉移工具
docs/                            進度、執行紀錄、資料來源與 GPU 主機說明
data/ runs/ artifacts/           原始資料、執行紀錄、模型；只有小型已驗證產物進 git
```

```mermaid
flowchart LR
  D[MaleCNS 官方表格] --> E[離線子圖萃取]
  F[官方 flyvis checkpoint] --> X[原生前端轉換與輸出比對]
  E --> P[共用感測前端與中央策略]
  X --> P
  S[2D 世界] --> O[影像小眼取樣、左右氣味、有效遮罩]
  O --> P
  S --> T[教師、critic、獎勵與評估]
  P --> A[動作 v、omega]
  A --> S
  P --> C[權重、圖、感測契約、model_card]
  C --> R[ROS 感測策略節點]
  G[Gazebo 世界與干擾節點] --> R
  R --> G
  G --> V[唯讀評估與 rosbag]
```

## 觀測、動作與記憶契約

| 項目 | 形狀／單位 |
| --- | --- |
| `vision` | float32 `[2,721]`，左右眼灰階值 `[0,1]` |
| `odor` | float32 `[2]`，左右非負濃度，任意濃度單位 |
| `valid` | float32 `[3]`，左眼、右眼、成對氣味的有效旗標 |
| 動作 | float32 `[2]`，`v` 為 m/s、`omega` 為 rad/s |
| 控制步長 | 50 ms，20 Hz |
| 神經步長 | 10 ms，每個控制步做 5 個子步 |
| 遞迴狀態 | 包含雙眼前端、ORN 適應與中央網路；回合間重置 |

`history_steps=10` 保留觀測儲存／暖啟動規格，策略每次只讀一個重取樣後的新控制步，持續更新隱狀態，不重複灌入整段歷史。訓練使用 32 步的截斷反向傳播；記憶本身不被硬性限制為 0.5 秒。

感測契約在 [configs/sensor_contract.json](configs/sensor_contract.json)。小眼採半徑 15 六角晶格；方向與取樣順序共用同一演算法，前端轉換使用相同座標表。相機為 160×120、水平 120°、向外偏 45°；垂直視野 104.821821° 對應同像素焦距。相機與氣味感測器的前後偏移、左右間距應和 SDF 一併修改。

## 重建資料與前端（一般不需要）

repo 已附上驗證過的子圖與前端。只有要修改萃取設定或重新驗證時才需要以下步驟。

### 子圖萃取

```bash
python python/connectome_extract/download.py --output data/raw          # 約 1.11 GB，不需要 token
python -m connectome_extract inventory --config configs/extraction/main.yaml
python -m connectome_extract extract --config configs/extraction/main.yaml
python scripts/verify_subgraph.py                                       # 對照原始表格的獨立檢查
```

萃取採感覺下游與 DN 上游的有界搜尋交集、突觸門檻及固定點修剪。每個保留的種子型別需可到 DN，子圖需弱連通且在預算內。目前設定的選擇理由寫在 `configs/extraction/main.yaml` 的 `selection_rationale`；候選型別掃描可用 `scripts/scan_candidates.py` 重跑。同資料同設定重跑兩次，產物位元相同。

### flyvis 前端

```bash
python scripts/export_flyvis.py \
  --network-dir /path/to/flyvis/results/flow/0000/000 \
  --sensor-contract configs/sensor_contract.json \
  --output artifacts/frontend/frontend.npz
```

轉換程式展開已學習參數，以灰階、閃光與移動刺激比對官方與原生模型輸出，通過才寫出產物。結果見 [docs/frontend_execution.md](docs/frontend_execution.md)。前端到 LC 種子的空間池化是可配置的工程假設，不是實測感受野。

## 訓練流程細節

```bash
python -m flytrack_train pretrain --config configs/train/pretrain_vision.yaml
python -m flytrack_train pretrain --config configs/train/pretrain_odor.yaml
python -m flytrack_train bc --config configs/train/bc.yaml
python -m flytrack_train ppo --config configs/train/ppo.yaml
```

各階段透過 `init_artifact` 接續同變體模型。預訓練 decoder 留在研究 checkpoint，不隨 actor 匯出；BC 與三輪 DAgger 的標籤來自教師，學生仍只看感測。PPO 使用獨立真值 critic、動作機率轉換、回合重置、終止／截斷分別處理與序列更新。各 CLI 支援 `--seed`、`--device`、`--output`、`--variant`、`--resume`。

研究用的 RL 階段有 PPO 和 ES（`python -m flytrack_train es --config configs/train/es.yaml`）。ES 是鏡像擾動、秩正規化的 OpenAI ES 風格實作，不是 CMA-ES。不論用哪一種，要加回正式鏈都必須另發 Decision，而且六個變體要用同一個算法和預算。

正式比較的預算：預訓練 240,000 步、BC／DAgger 150,000 步（[configs/evaluation_protocol.yaml](configs/evaluation_protocol.yaml) 2026-10-24 的修訂把 PPO 改為 0）。評估輸出逐回合 CSV／JSON、種子平均、bootstrap 95% 信賴區間及預先指定的 VO、VO-late 對 V／GRU／RS 比較（[configs/evaluation_protocol.yaml](configs/evaluation_protocol.yaml)）。未重新捕獲以截尾資料記錄，不能填成零秒；不會只以差異不顯著就宣稱等效。

## 匯出與 ROS／Gazebo

```bash
python -m flytrack_train export --config configs/train/bc.yaml \
  --artifact runs/bc/VO/seed_0/actor --output artifacts/VO
cp .env.example .env
docker compose --profile ml --profile sim build
docker compose --profile sim up sim policy eval viz
```

匯出包含 `weights.pt`、`graph.npz`、`frontend.npz`、`sensor_contract.json`、`model_card.json` 和檢查碼；部署時核對雜湊、步長、幾何和形狀，不相符就拒絕載入。Compose 有六個服務、兩種映像：extract／train 使用 ML，sim／policy／eval／viz 使用 ROS。沒有 NVIDIA 容器支援的主機可加 `-f compose.cpu.yaml` 並設定 `TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`。

策略節點只訂閱 `/fly/*` 感測。單一感測超過 200 ms 以缺失輸入繼續，全部逾時停車；回合控制透過 `/connectome_policy_node/reset` 清除神經與感測緩衝。Foxglove bridge 使用 port 8765。

主要四模型的 Gazebo 批次評估與 lesion 分析：

```bash
python scripts/run_gazebo_matrix.py
python -m flytrack_train evaluate --config configs/train/evaluate_gazebo_recorded.yaml
python scripts/lesion_analysis.py --artifact artifacts/VO
```

lesion 的來源電流分析是描述性診斷，不能當成因果貢獻證明。ROS／Gazebo 部署鏈路已通過五回合 smoke 驗證，延遲仍未達標；這不是正式策略品質的驗收，證據與限制見 [docs/ros_execution.md](docs/ros_execution.md) 及 [#22](https://github.com/Nues0913/Drosophila_Tracking_System/issues/22)。

### 延遲指標的兩種時鐘

`use_sim_time` 用來計算回合、軌跡、感測過期和行為指標。`/fly/inference_latency_ms` 記錄實際推論的 wall time。完整 sensor-to-command 的因果計時已有契約與實作（`SensorTiming.msg`、`ControlTrace.msg`、`flytrack_eval.latency.CausalLatency`），已在 Gazebo smoke 中實測：CPU／GPU 的完整端到端 p95 為 55.62／56.46 ms（#22、PR #45），均未達計劃書的 50 ms。兩種時鐘不能相加，各階段的 p95 也不能相加代替端到端 p95；#20 維持原門檻，先以既有 bag 做逐筆、同時鐘的延遲診斷。

## 文件索引

- [docs/status.md](docs/status.md)：目前進度、已知問題、下一步（**先讀這份**）
- [AGENT_TEAM_WORKFLOW.md](AGENT_TEAM_WORKFLOW.md)：三人（Agent）分工、Issue／PR 與審查規範（參與開發前先讀）
- [後續計畫.md](後續計畫.md)：分階段檢查清單
- [docs/gpu_hosts.md](docs/gpu_hosts.md)：在 Linux／WSL2／Colab 跑訓練與分工
- [docs/execution_log.md](docs/execution_log.md)：執行環境與驗證證據
- [docs/data_sources.md](docs/data_sources.md)：資料來源、授權、雜湊
- [docs/requirements_traceability.md](docs/requirements_traceability.md)：計劃書驗收對照
- `python/*/…_NOTES.md`：各套件設計筆記

## 授權與引用

MaleCNS v1.0 資料為 CC-BY 4.0，報告須引用 FlyEM／Janelia 等來源（見 [docs/data_sources.md](docs/data_sources.md)）。前端參數來自 [flyvis](https://github.com/TuragaLab/flyvis) 官方預訓練模型。
