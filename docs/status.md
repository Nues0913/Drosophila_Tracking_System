# 目前進度與已知問題

最後更新：2026-10-18。逐項檢查清單見 [後續計畫.md](../後續計畫.md)，驗證證據見 [execution_log.md](execution_log.md)，本次整合修正見 [fixes_2026-10-17.md](fixes_2026-10-17.md)。

## 一句話

資料 → 前端 → 子圖 → 模型 → 小規模訓練鏈都已實際跑通；**「只學會一直往同一邊彎」的根因（固定繞圈環境＋只看距離的成功判定）已修正**，訓練與評估前會自動檢查環境是否能被盲目策略破解。**正式訓練（30 次）與 ROS／Gazebo 閉環還沒開始**：先請組內確認新門檻，再到 ≥8 GB GPU 的主機執行。

## 已完成並有證據

| 階段 | 結果 | 證據 |
| --- | --- | --- |
| MaleCNS v1.0 原始表格 | 三份官方表格下載，大小、ETag、Arrow 框架、SHA-256 核對 | `docs/data_sources.md` |
| flyvis 前端（方案 A） | 官方 checkpoint 作者 SHA-256 核對；原生轉換對官方輸出最大誤差 1.91e-6 | `artifacts/frontend/verification.json`、`docs/frontend_execution.md` |
| 中央子圖萃取 | 2569 神經元、14165 邊；弱連通、每個種子型別可達 DN、號誌、雜湊等 19 項獨立檢查全過；重跑兩次位元相同 | `artifacts/data_execution/graph_verification.json`、`scripts/verify_subgraph.py` |
| 六變體參數預算 | DN A＋C 後仍為 VO/V/O/RS=6794、VO-late=6766、GRU=6831，均在 ±5% 內 | `runs/dn_fix/experiment_configs/parameter_budget.json`（只產生設定，沒有執行 run_all.sh） |
| 測試與 lint | 116 passed（含真實前端，2026-10-18 的 main）；`ruff check`、`ruff format --check` 通過 | `pytest python/tests` |
| 小規模訓練鏈 | VO seed 0 縮小版：視覺預訓練 → 氣味預訓練 → BC → PPO 全部跑完，各階段權重有限且有更新 | 可用 `configs/train/smoke.yaml` 重現 PPO 部分 |
| Docker 映像 | `flytrack-ml:0.1`、`flytrack-ros:0.1` 本機建置成功 | `docs/ros_execution.md` |

## 已修正的重大問題

**中央遞迴初始化發散（已修）。** 邊權重 = 號誌 × 突觸數 × α。萃取門檻讓每條邊至少 60 個突觸（中位數 92），原本固定 `alpha_init=0.01` 使真實圖的遞迴矩陣譜半徑約 32，ReLU 動態每控制步放大約 2×10⁴ 倍，第 9 步溢位成 NaN，任何訓練都無法進行。現在初始化時以冪迭代求 |W| 的 Perron 根，α 取 `min(alpha_init, init_spectral_radius / ρ)`，`init_spectral_radius=0.9`，真實圖得 α≈2.56×10⁻⁴。參數數不變，六變體各自依自己的圖套用同一規則。回歸測試：`test_initial_recurrence_is_contractive_for_large_synapse_counts`。

**訓練速度（2026-10-15，RTX 4060 Laptop 8 GB、Ryzen 7 7735HS）。** 原本 pretrain／BC 一次只處理一個序列、每步重算凍結的 flyvis 前端，環境只用一個 CPU 核心；V seed 0 視覺預訓練估計約 27 小時（收集 38 分鐘＋每 epoch 81 分鐘 × 20）。現在：凍結感覺碼每步只算一次並重用；前端 spmm 改 node-major 排列（批次 32–128 時快 3–7 倍）；中央核心每窗口只組一次矩陣、每子步 3 個 kernel；環境在子行程平行；pretrain／BC 一次 16 條回合的批次 TBPTT，以保留回合的驗證損失提早停止；評估批次執行，每格另以部署方式量 latency。實測 V seed 0 視覺預訓練（240k 步）**34 分鐘**（收集約 10 分鐘、每 epoch 約 2.4 分鐘、第 11 epoch 提早停止、取第 8 epoch），held-out 方位誤差 0.38°（門檻 15°）。PPO 單一工作約 104 步／秒，同時 3 個工作合計約 231 步／秒。新舊前向輸出差異僅浮點捨入（約 1e-7）。

**GRU 正式設定無法載入（已修）。** `prepare_experiments.py` 寫入已解析的 `gru_input_size: 9` 後，`GRUPolicy` 仍用直接公式重算 hidden size，參數數變 5691，超出預算而拒絕建立；正式 GRU 訓練會在開始時失敗。現在已有 `gru_input_size` 時沿用 `hidden_size`，得到 6831。

**環境退化：固定左轉就能「追蹤」（2026-10-17 已修）。** 目標永遠逆時針繞同一個圓、果蠅永遠從正後方出發，成功判定只看距離，所以不看輸入的固定左轉在 4 種情境都和 teacher 一樣 100% 成功，BC 只學到平均動作。現在每回合隨機化佈局（2D 與 ROS 共用 `flytrack_core.world.layout`）、新增 wander 目標、成功須 `|方位| ≤ 45°`，並以 `audit-env` 在訓練與評估前檢查：新世界中 teacher 100%、所有盲目策略 0%。DN 輸入增益改在初始化設定（`motor_input_gain=100`）、讀出加 ω 偏置、BC 固定神經元偏置並改用轉向相關／R²／幅度驗收。細節與數字見 [fixes_2026-10-17.md](fixes_2026-10-17.md)。

## 已知問題與科學上的限制（報告須揭露）

0. **【程式已修、待訓練驗證】運動 DN 沉默的 A＋C 修正。** 新連接體模型只對運動 DN 用 `softplus(10v)/10`，遞迴、activity、動作讀出共用；pretrain／BC／PPO 的直接活動正則排除 DN。GRU 動態與參數預算不變，舊 artifact 未記錄 β 時仍保留 ReLU。現有 smoke 權重的唯讀 CPU 反事實探針：換眼角速度差中位數由 `5.13e-7` 升到 `2.08e-4 rad/s`，有效 DN 由 0/6 升到 6/6（rate>1e-3）；仍未達 `0.05 rad/s` 的控制門檻，不能宣稱已通過端到端驗收。`dn_health.py` 已加入階段 gate，單元測試確認能偵測全 DN 沉默並回傳失敗。本次沒有實際訓練，預訓練／BC 後活性、控制誤差與 validate-pretrain 均**待訓練驗證**；完整數字、舊沉默 checkpoint 探針與風險見 [修正報告](dn/fix_A_C_2026-10-15.md)，原始證據見 [DN 檢查報告](dn/audit_2026-10-15.md)。活動正則是否造成沉默仍未做因果消融。

   同一組 held-out 種子重播現有中斷 V checkpoint（不更新權重）：舊 ReLU 的 6 個 DN 正活性比例均為 0、中央動作梯度範數為 0；只切換 β=10 後均為 100%，中央梯度恢復至 `1.74e-3`，但換眼差中位數仍為 0、最大僅 `5.38e-6 rad/s`。這證明本次探針解除 dead ReLU 的阻斷，尚未證明有足夠的感覺控制能力。

   **2026-10-16 更新：小規模端到端驗證未通過 BC gate。** DN 已不再沉默、梯度恢復，但 BC 只學到平均動作（固定轉向偏差），換眼差 0.017 rad/s。診斷顯示 pretrain 後方位資訊已傳到 DN 左右差（R² 0.66）但幅度僅約 3e-4，BC 因而改調偏置並破壞了這些資訊。完整數據、反事實測試與待決定事項見 [DN 診斷結論](dn/diagnosis_2026-10-16.md)；訓練時間見 [training_time.md](training_time.md)。

   **2026-10-17 更新：BC 只學到平均動作，主因是環境，不是 DN。**目標永遠以 0.1 rad/s 逆時針繞圈，蒼蠅永遠從正後方出發，所以一個不看輸入的固定左轉策略在 4 種評估情境都能 100% 成功，和 teacher 一樣。改成每回合隨機繞圈方向和起始偏移後，固定策略降到 ≤23%；原樣模型的 BC 也學會依方位轉向（ω 和 teacher 的相關 0.78，換眼差 0.156）。讀出層 ω 偏置、BC 固定神經元偏置、DN 輸入增益三項修正讓 BC 指標更好（唯一通過所有 BC gate 的一組），但不是必要的。環境、成功判定、gate 都需要組內決定；新選項已加入程式，預設關閉。見 [情境測試](dn/scenario_test_2026-10-17.md)。

   **2026-10-17 整合後本機驗證（限時，約 30 分鐘）：** 視覺 pretrain 後 DN 健康通過，中央梯度 5.8e-2，策略 ω 與 teacher 相關 0.55（尚未學動作）。短版 BC（18k 步，正式預算的 12%，沒有嗅覺 pretrain）**未過新的 BC gate**：相關 0.54、換眼反應 0.065 有過，但 R² −0.03、幅度只有 teacher 的 22%。DNa01 沉默，轉向只靠 DNa02。需要在 GPU 主機用完整預算確認，細節見 [fixes_2026-10-17.md](fixes_2026-10-17.md#小規模端到端驗證本機限時)。

1. **匯流神經元只有 4 個。** 視覺與嗅覺路徑在 3 跳內交會的神經元很少；依計劃書 G1，以 VO-late 為後備主要變體並明示，不能把晚期融合說成找到生物匯流。嗅覺到運動 DN 要 5–7 跳，視覺只要 2–3 跳，訊號強度因此天生偏向視覺（見第 13 點）。
2. **LC9、LC11 各只保留 1 個神經元**（原有 219、143 個）。視覺輸入實際上主要來自 LC10a（208）、LC10d（143）、LPLC2（112）。
3. **ORN_VA2 有 9 個（約 11%）側別未知，已依設定丟棄**；見 `data/subgraphs/main/excluded_odor_seed_sides.csv`。
4. **運動 DN 讀出的文獻依據（2026-10-17 核對）。** DNp09 即 P9：雙側活化驅動前進、單側活化產生同側轉向前進（Bidaye et al. 2020, *Neuron* 108:469–485，「This line labels DNp09 (henceforth referred to as P9)」）。DNa02 左右活性差與轉向速度呈線性，DNa01 追蹤緩慢持續的轉向（Rayshubskiy et al., *eLife* 102230）。模型以 DNp09 左右平均讀前進、DNa01／DNa02 的「左 − 右」讀轉向（側別 −1 為左），初始權重為正時左側較活躍即向左轉，與同側轉向一致。DNp09 本身也帶轉向成分，模型未使用，報告須說明。
5. **教師校準（2026-10-17 已重做）。** 舊版 20 回合逐位相同（環境沒有變化）。在隨機化世界、種子 1.65e9 起重跑：成功率 100%，平均方位誤差 2.22°（1.48°～3.21°），20 個回合佈局都不同；評估協定已更新。
6. **GPU 記憶體。** GTX 1050 3 GB 在 PPO 第一次更新時 OOM（PyTorch 已配置 1.75 GiB，加上桌面程式占用）。正式設定需要更大的 GPU；確切下限還沒量。
7. **速度。** 見上方「訓練速度」。PPO 仍是主要耗時：30 次 × 2M 步，同時 3 個工作約 3 天；GPU 端受 kernel 啟動延遲限制（torch.compile 只再快約 20%，未採用）。還沒處理的部分：
    - 縮短方案還沒決定，選項見 [training_time.md](training_time.md)。
    - CUDA Graphs 還沒試。
    - 第 13 點的修正如果要從 pretrain 重練，每條訓練鏈約多 1 小時。
8. **ROS／Gazebo**：映像內的 `colcon build` 成功；2026-10-18 用目前的程式碼在映像裡跑真值隔離 launch test（通過）和 latency 單元測試（通過）。同時發現策略節點關閉時送不出停止命令，已由 #15 修正（SIGINT／SIGTERM 都會先送出零速度命令再退出）。**閉環回合、Gazebo 物理與相機、延遲量測都還沒做。**
9. **評估的主要終點、非劣容許差** 已在 `configs/evaluation_protocol.yaml` 宣告（2026-10-17 修訂為 schema 2：隨機化世界、朝向條件、環境審計），正式訓練前請組內確認一次。
10. **新門檻是工程選擇**：朝向 45°、盲目策略 ≤10%、teacher ≥90%、BC 轉向相關 ≥0.5 與幅度 ≥25%、DN 輸入增益 100、wander 參數；請 reviewer 確認。
11. **noise 情境的大幅波動**（情境測試中 A 3%、B 77%）。有一部分是設定不一致造成的：情境參數原本有程式預設值和 YAML 兩個來源，bc_A 的 noise 碰撞率因此出現 0.667 對 0.267（#2，已由 #5 修正）。另外，noise 情境的結果對雜訊參數非常敏感：bc_B 的 noise 成功率在兩組參數下是 77% 對 0%。雜訊強度對成功率的曲線，以及 BC 是否要做雜訊擴增，都還沒做。
12. **linear 目標撞牆反彈會撞上跟隨者**（teacher 13% 碰撞），因此只留在 PPO 暖身課程。wander 目標撞上 teacher 的問題已在 #1 修正：總轉速加上限，teacher 也會閃避，1000 回合 0 次碰撞。正式評估要不要加入 wander 還沒決定。
13. **【未解決】策略不用氣味（#4）。** bc_A、bc_B 對去掉氣味、氣味 ×10、左右交換的動作差都是 0。原因是非運動節點的 ReLU 在工作點截斷了嗅覺訊號。**所以現有的 VO 結果實際上只用了視覺**，conflict 的數字等於 normal，VO／V／O 的比較和 lesion 結論都還不成立。
    - #7（已 merge）：可選的 ORN 輸入增益與嗅覺 softplus（預設都關閉；×100 會讓 odor pretrain 發散），pretrain 支援 `core_bias_learning_rate`，以及 `scripts/diagnostics/modality_sensitivity.py`。#12 讓 PPO／ES 也遵守這個設定，凍結的 bias 會一路保持到最後階段。
    - 我們在 4060 上的對照：
      - ×100 會讓嗅覺 rate 到 136（視覺 0.37），odor pretrain 因此發散。
      - vision pretrain 只更新 38 次，氣味對 ω 的影響就掉約 150 倍。
      - 在氣味通路不通時，用 20 秒長遮蔽混合訓練 BC，策略會崩潰成常數 ω（normal 成功率 0%，兩個 seed 都一樣）。
    - 診斷（2026-10-18）：訊號其實傳得到 DN，是 DN 的工作點被壓低了。
      - vision pretrain 預設會訓練 bias（BC 預設凍結；pretrain 現在可用 `core_bias_learning_rate: 0` 凍結），上游抑制性節點因此變得持續活躍，再經 DN 輸入增益 ×100 放大，把 DNa01 的電壓壓到 −0.875。
      - 在這個電壓下，softplus 的斜率只剩 1.6e-4，氣味造成的 ΔV 換不成 rate。
      - 氣味主要經 DNa01 影響轉向，所以「DNa01 沉默」（第 0 點）和 #4 是同一件事。
      - 只把 bias 換回初始化的值，氣味對 ω 的影響就恢復到初始化的 104%。
      - 正在試三個可選修正：pretrain 凍結 bias、DN 電壓下限正則、odor decoder 輸入標準化。
    - 修好之後，pretrain 要從頭重練，舊的 pretrain artifact 不能沿用。
14. **情境強度。**
    - conflict 目前等於 normal（第 13 點）。
    - occlusion 只遮 2 秒（佔回合 3%），太弱，測不出氣味的補位作用。但在氣味通路修好之前，改成長遮蔽反而會讓 BC 崩潰（第 13 點）。
15. **GPU 評估無法逐位元重現（#3，已修）。** 唯一的來源是 cuSPARSE。已由 #6 修正：評估 config 設 `evaluation.deterministic: true`，或診斷腳本設 `DETERMINISTIC=1`。在 4060 上兩次重跑的結果逐位元相同，耗時約 2 倍，只用在回歸測試和參考檔比對。
16. **teacher／BC 是否必要還沒驗證（T3）。** 計畫從同一個 pretrain 出發，比較「BC 加 PPO」和「直接 PPO」：先跑 1 個種子各 500k 步，再決定要不要跑 3 個種子 × 2M 步。

### 必須揭露的工程近似

以下原本記在 `implementation_status.md`（2026-10-15），報告須說明。

1. 方案 A 的早期視覺來自 flyvis 的資料與學習參數，只有中央子圖使用 MaleCNS。報告應區分這兩個來源。
2. 沒有真實 RF 座標時，LC 輸入映射採型別／側別的空間池化；ORN Hill 閾值也是模擬設定，不是特定化學氣味的實測受體曲線。
3. V／O 是同一張圖的輸入消融。VO-late 共享型別參數但隔離活性；RS 保留原邊參數槽的共享標籤。這些比較口徑都寫進 model_card。
4. 2D 使用 2.5D 光線投射，Gymnasium 向量環境是 CPU 上的平行子行程，沒有實作 GPU 張量化世界。
5. 2D 有大小／亮度的領域隨機化。Gazebo 使用設定中的固定幾何和材質，還沒證明兩者的感測分布一致，仍需量測與校準。
6. ROS 碰撞目前用場地、半徑與障礙幾何判斷，沒有接 Gazebo contact sensor 做獨立核對。reset 的 pose 服務、零速度與隱狀態清理仍需跨節點驗證。
7. ROS 的感測時間戳用模擬時鐘，推論耗時用 wall time。完整 sensor-to-command 因果延遲還沒有量測契約，不能宣稱 p95 < 50 ms。
8. 續訓：同一階段中斷後可完整接續（權重、optimizer、RNG、環境重播，由 `test_interrupted_training_matches_uninterrupted_weights_optimizer_and_rng` 驗證）；跨階段以 `init_artifact` 暖啟動，recovery 也綁定上游 actor 的內容（#14）。改了程式碼或 optimizer 參數組數之後，舊的 checkpoint 不能接續，要重跑該階段。

## 下一步（建議順序）

0. **reviewer 確認 [fixes_2026-10-17.md](fixes_2026-10-17.md)「仍待組內決定」的門檻**。環境、成功判定與 BC gate 已依情境測試的建議實作。
1. **在有 ≥8 GB GPU 的主機**照 README「組員快速上手」安裝，跑 `pytest`、`audit-env` 與 smoke，回報速度與顯存。
2. 用 **VO seed 0** 跑完整預算，確認 PPO 穩定、回報有上升；需要調整的話，所有變體一起改。**先修好 #4（第 13 點）**，否則 VO 實際上等於 V，正式訓練的多感官比較沒有意義。
3. 依新的教師校準（第 5 點）固定方位解碼門檻與成功率目標。
4. 依 [gpu_hosts.md](gpu_hosts.md#分工跑正式訓練) 分工跑 6 變體 × 5 種子；派工方式見 [AGENT_TEAM_WORKFLOW.md](../AGENT_TEAM_WORKFLOW.md#gpu-任務派工)。
5. 同時有人負責 ROS／Gazebo：`colcon build` 和真值隔離測試已通過（第 8 點），接著是閉環回合、reset 跨節點驗證與延遲量測（見 [ros_execution.md](ros_execution.md)）。
6. 2D 正式評估 → Gazebo 主要比較 → lesion 分析 → 報告與展示。
