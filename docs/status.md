# 目前進度與已知問題

最後更新：2026-10-15。逐項檢查清單見 [後續計畫.md](../後續計畫.md)，驗證證據見 [execution_log.md](execution_log.md)。

## 一句話

資料 → 前端 → 子圖 → 模型 → 小規模訓練鏈都已實際跑通並驗證；**正式訓練（30 次）與 ROS／Gazebo 閉環還沒開始**，卡在需要更大的 GPU。

## 已完成並有證據

| 階段 | 結果 | 證據 |
| --- | --- | --- |
| MaleCNS v1.0 原始表格 | 三份官方表格下載，大小、ETag、Arrow 框架、SHA-256 核對 | `docs/data_sources.md` |
| flyvis 前端（方案 A） | 官方 checkpoint 作者 SHA-256 核對；原生轉換對官方輸出最大誤差 1.91e-6 | `artifacts/frontend/verification.json`、`docs/frontend_execution.md` |
| 中央子圖萃取 | 2569 神經元、14165 邊；弱連通、每個種子型別可達 DN、號誌、雜湊等 19 項獨立檢查全過；重跑兩次位元相同 | `artifacts/data_execution/graph_verification.json`、`scripts/verify_subgraph.py` |
| 六變體參數預算 | DN A＋C 後仍為 VO/V/O/RS=6794、VO-late=6766、GRU=6831，均在 ±5% 內 | `runs/dn_fix/experiment_configs/parameter_budget.json`（只產生設定，沒有執行 run_all.sh） |
| 測試與 lint | 98 passed（含真實前端）；`ruff check`、`ruff format --check` 通過 | `runs/dn_fix/pytest.txt`、`runs/dn_fix/REPORT.md` |
| 小規模訓練鏈 | VO seed 0 縮小版：視覺預訓練 → 氣味預訓練 → BC → PPO 全部跑完，各階段權重有限且有更新 | 可用 `configs/train/smoke.yaml` 重現 PPO 部分 |
| Docker 映像 | `flytrack-ml:0.1`、`flytrack-ros:0.1` 本機建置成功 | `docs/ros_execution.md` |

## 已修正的重大問題

**中央遞迴初始化發散（已修）。** 邊權重 = 號誌 × 突觸數 × α。萃取門檻讓每條邊至少 60 個突觸（中位數 92），原本固定 `alpha_init=0.01` 使真實圖的遞迴矩陣譜半徑約 32，ReLU 動態每控制步放大約 2×10⁴ 倍，第 9 步溢位成 NaN，任何訓練都無法進行。現在初始化時以冪迭代求 |W| 的 Perron 根，α 取 `min(alpha_init, init_spectral_radius / ρ)`，`init_spectral_radius=0.9`，真實圖得 α≈2.56×10⁻⁴。參數數不變，六變體各自依自己的圖套用同一規則。回歸測試：`test_initial_recurrence_is_contractive_for_large_synapse_counts`。

**訓練速度（2026-10-15，RTX 4060 Laptop 8 GB、Ryzen 7 7735HS）。** 原本 pretrain／BC 一次只處理一個序列、每步重算凍結的 flyvis 前端，環境只用一個 CPU 核心；V seed 0 視覺預訓練估計約 27 小時（收集 38 分鐘＋每 epoch 81 分鐘 × 20）。現在：凍結感覺碼每步只算一次並重用；前端 spmm 改 node-major 排列（批次 32–128 時快 3–7 倍）；中央核心每窗口只組一次矩陣、每子步 3 個 kernel；環境在子行程平行；pretrain／BC 一次 16 條回合的批次 TBPTT，以保留回合的驗證損失提早停止；評估批次執行，每格另以部署方式量 latency。實測 V seed 0 視覺預訓練（240k 步）**34 分鐘**（收集約 10 分鐘、每 epoch 約 2.4 分鐘、第 11 epoch 提早停止、取第 8 epoch），held-out 方位誤差 0.38°（門檻 15°）。PPO 單一工作約 104 步／秒，同時 3 個工作合計約 231 步／秒。新舊前向輸出差異僅浮點捨入（約 1e-7）。

**GRU 正式設定無法載入（已修）。** `prepare_experiments.py` 寫入已解析的 `gru_input_size: 9` 後，`GRUPolicy` 仍用直接公式重算 hidden size，參數數變 5691，超出預算而拒絕建立；正式 GRU 訓練會在開始時失敗。現在已有 `gru_input_size` 時沿用 `hidden_size`，得到 6831。

## 已知問題與科學上的限制（報告須揭露）

0. **【程式已修、待訓練驗證】運動 DN 沉默的 A＋C 修正。** 新連接體模型只對運動 DN 用 `softplus(10v)/10`，遞迴、activity、動作讀出共用；pretrain／BC／PPO 的直接活動正則排除 DN。GRU 動態與參數預算不變，舊 artifact 未記錄 β 時仍保留 ReLU。現有 smoke 權重的唯讀 CPU 反事實探針：換眼角速度差中位數由 `5.13e-7` 升到 `2.08e-4 rad/s`，有效 DN 由 0/6 升到 6/6（rate>1e-3）；仍未達 `0.05 rad/s` 的控制門檻，不能宣稱已通過端到端驗收。`dn_health.py` 已加入階段 gate，單元測試確認能偵測全 DN 沉默並回傳失敗。本次沒有實際訓練，預訓練／BC 後活性、控制誤差與 validate-pretrain 均**待訓練驗證**；完整數字、舊沉默 checkpoint 探針與風險見 [修正報告](dn/fix_A_C_2026-10-15.md)，原始證據見 [DN 檢查報告](dn/audit_2026-10-15.md)。活動正則是否造成沉默仍未做因果消融。

   同一組 held-out 種子重播現有中斷 V checkpoint（不更新權重）：舊 ReLU 的 6 個 DN 正活性比例均為 0、中央動作梯度範數為 0；只切換 β=10 後均為 100%，中央梯度恢復至 `1.74e-3`，但換眼差中位數仍為 0、最大僅 `5.38e-6 rad/s`。這證明本次探針解除 dead ReLU 的阻斷，尚未證明有足夠的感覺控制能力。

   **2026-10-16 更新：小規模端到端驗證未通過 BC gate。** DN 已不再沉默、梯度恢復，但 BC 只學到平均動作（固定轉向偏差），換眼差 0.017 rad/s。診斷顯示 pretrain 後方位資訊已傳到 DN 左右差（R² 0.66）但幅度僅約 3e-4，BC 因而改調偏置並破壞了這些資訊。完整數據、反事實測試與待決定事項見 [DN 診斷結論](dn/diagnosis_2026-10-16.md)；訓練時間見 [training_time.md](training_time.md)。

   **2026-10-17 更新：BC 只學到平均動作，主因是環境，不是 DN。**目標永遠以 0.1 rad/s 逆時針繞圈，蒼蠅永遠從正後方出發，所以一個不看輸入的固定左轉策略在 4 種評估情境都能 100% 成功，和 teacher 一樣。改成每回合隨機繞圈方向和起始偏移後，固定策略降到 ≤23%；原樣模型的 BC 也學會依方位轉向（ω 和 teacher 的相關 0.78，換眼差 0.156）。讀出層 ω 偏置、BC 固定神經元偏置、DN 輸入增益三項修正讓 BC 指標更好（唯一通過所有 BC gate 的一組），但不是必要的。環境、成功判定、gate 都需要組內決定；新選項已加入程式，預設關閉。見 [情境測試](dn/scenario_test_2026-10-17.md)。

1. **匯流神經元只有 4 個。** 視覺與嗅覺路徑在 3 跳內交會的神經元很少；依計劃書 G1，以 VO-late 為後備主要變體並明示，不能把晚期融合說成找到生物匯流。
2. **LC9、LC11 各只保留 1 個神經元**（原有 219、143 個）。視覺輸入實際上主要來自 LC10a（208）、LC10d（143）、LPLC2（112）。
3. **ORN_VA2 有 9 個（約 11%）側別未知，已依設定丟棄**；見 `data/subgraphs/main/excluded_odor_seed_sides.csv`。
4. **DNp09 作為前進讀出仍是候選標籤**，尚未有文獻核對。
5. **教師校準無效**（同一個根因：環境沒有變化，見 [情境測試](dn/scenario_test_2026-10-17.md)）。 `runs/calibration/teacher.json` 的 20 回合結果逐位相同：`evaluate.yaml` 關閉 domain randomization，圓形軌跡只差起始相位，幾何上等價。不能用它設定方位解碼門檻，需要改成有變異的校準情境。
6. **GPU 記憶體。** GTX 1050 3 GB 在 PPO 第一次更新時 OOM（PyTorch 已配置 1.75 GiB，加上桌面程式占用）。正式設定需要更大的 GPU；確切下限還沒量。
7. **速度。** 見上方「訓練速度」。PPO 仍是主要耗時：30 次 × 2M 步，同時 3 個工作約 3 天；GPU 端受 kernel 啟動延遲限制（torch.compile 只再快約 20%，未採用）。
8. **ROS／Gazebo** 只建好映像，`colcon build`、閉環回合、真值隔離 launch test、延遲量測都還沒做。
9. **評估的主要終點、非劣容許差** 已在 `configs/evaluation_protocol.yaml` 宣告，正式訓練前請組內確認一次。

## 下一步（建議順序）

0. **先決定訓練與評估環境、成功判定、BC gate**（見 [情境測試](dn/scenario_test_2026-10-17.md) 第 5 節）。在這之前，正式長時間訓練的結果無法解讀。
1. **在有 ≥8 GB GPU 的主機**照 README「組員快速上手」安裝，跑 `pytest` 與 smoke，回報速度與顯存。
2. 用 **VO seed 0** 跑完整預算，確認 PPO 穩定、回報有上升；需要調整的話，所有變體一起改。
3. 修正教師校準情境（第 5 點），固定方位門檻與成功率目標。
4. 依 [gpu_hosts.md](gpu_hosts.md#分工跑正式訓練) 分工跑 6 變體 × 5 種子。
5. 同時有人負責 ROS／Gazebo：`colcon build`、閉環回合、真值隔離測試（見 [ros_execution.md](ros_execution.md)）。
6. 2D 正式評估 → Gazebo 主要比較 → lesion 分析 → 報告與展示。
