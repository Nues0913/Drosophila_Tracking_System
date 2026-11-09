# 目前進度與已知問題

最後更新：2026-11-09（正式矩陣的 30 條訓練鏈和 2D 正式評估（curved、wander）都在 main `35432c3` 完成並驗收，交付在 release `formal-v1`；已登記比較依 PR #83 重新彙總。GRU 對照組已依 #85 修正（PR #86）並完成事後、未登記的重新訓練與評估（#87）。協定必交的分析 A1 已驗收（#90），發現 VO／VO-late 的前進速度是常數；接著的探索實驗 S1（速度讀出校準，#95）和 R1（BC 配方，#4）都判定無效；formal 配方原樣重訓（R0）顯示基準在重訓下穩定。D2 的探索到此結束，formal-v2 不觸發。Gazebo 的 wander 接線已合併（PR #94），G1 試跑發現 policy 跟不上 20 Hz、因果計時大多沒有對上，已由 G0b（PR #99）修正；2060 和 4070 的 G1 範圍都通過；G2 派工後，PM 量到 Gazebo 的視覺輸入和 2D 明暗相反，G2 暫停，修正 G0c 在 PR #102（#88）。2D 展示素材的工具已合併（PR #100），正式交付由 V1（#97）執行）。正式規格維持 no-KC＋P2/B3、pretrain → BC/DAgger（#35），團隊選 F：保留全部已登記比較，如實報告 VO 的有限氣味反應及負面結果。待辦與分工以 GitHub Issue 為準，總追蹤是 [#28](https://github.com/Nues0913/Drosophila_Tracking_System/issues/28)。協作規則見 [AGENT_TEAM_WORKFLOW.md](../AGENT_TEAM_WORKFLOW.md)，逐項檢查清單見 [後續計畫.md](../後續計畫.md)，驗證證據見 [execution_log.md](execution_log.md)。

## 2026-11-09 G0b 合併、兩台主機的 G1 範圍、G2 派工與暫停（#88）

- **G2 暫停**（[#88 Decision 6744333249](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6744333249)）：
  - **發現**：PM 錄 rosbag 做開環對照。在同一 pose 下，Gazebo 的 hex-eye 輸入（policy 實際收到的）和 2D ray cast 的逐幀相關中位數是 −0.97。
  - **原因**：舊世界的天空和被照亮的地板很亮，牆和目標反而暗，和 2D 的暗背景、亮目標相反。
  - 氣味兩邊一致。G1 範圍的 52 回合 success 全部是 0，同一批 actor 在 2D 是 0.39–0.48。
  - **修正 G0c**（PR #102）：平光材質，emissive 用 sRGB 解碼後的值。驗收（看行為之前就固定）通過：左右眼 r 中位數 0.979／0.968，MAD 0.013／0.011。同樣兩個環境 seed 的行為，逐回合和 2D 幾乎相同。
  - **G0c 合併之後**：4070、4060 重跑 G1 範圍，再重新發 G2。細節見 PR #102 的 `docs/gazebo_consistency.md`。

- **G0b**（PR #99，main `4a537af`）：
  - policy 預設用 1 個 torch 執行緒；
  - evaluator 自己訂閱 SensorTiming，補齊計時。
  - 控制路徑的數學、感測頻率、世界和材質都不變。
- **G1 範圍**（seed 0，normal 和 occlusion，curved 和 wander，每格 2 回合，`--cpu`）：
  - **2060**：52/52 valid，19.6–19.9 Hz，計時 100%，sensor→command p95 是 55–130 ms。
  - **4070**：52/52 valid，19.5–19.9 Hz，計時 100%，p95 是 48–53.5 ms。只有 GRU 和事後 GRU 低於 50 ms。
  - 兩台主機都可以跑 G2。2060 的延遲高很多，所以不用於 G2：使用者的規則是重度測試交給 teammate。
  - 每格只有 2 回合，這些都是工程紀錄。逐台的表和頻率定義見 [ros_execution.md](ros_execution.md) 的 2026-11-09 一節。
- **G2 派工**（已暫停，見上；[#88 Decision 6743008783](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6743008783)）：
  - 固定 main `4a537af`。
  - 一個區塊是「一個訓練 seed × 一種目標運動」：4070 跑 curved seed 0–2、wander seed 0–1；4060 先跑 G1 範圍，再跑其餘區塊。兩台各約 27 小時。
  - G3 延遲的判定在資料出來之前就固定：參考主機是 4070，取 curved normal 合併後的 p95。
  - 彙總設定和延遲腳本在 PR #101。
- **V1**（#97）：2D 展示素材的工具已合併（PR #100，main `2fb98a4`），正式交付等 MiLong 在 4060 上執行。V1 不能和 Gazebo 在同一台主機上同時跑。

## 2026-11-08 D2 結束、Issue 整理與 Gazebo G1（2060）

**D2 結束**（[#28 Decision 6718154301](https://github.com/Nues0913/Drosophila_Tracking_System/issues/28#issuecomment-6718154301)）
- R1、S1 判定無效，B1 不執行，R0 顯示基準在重訓下穩定。沒有任何探索配方通過事先固定的判讀。
- formal-v2 不觸發，formal-v1 就是要報告的版本。速度讀出要不要改結構，等使用者決定。

**Issue 整理**（[#28 Decision 6718297431](https://github.com/Nues0913/Drosophila_Tracking_System/issues/28#issuecomment-6718297431)）
- #4 結案，剩下的方向（P3／PPO、文獻上的候選做法）併入 #35。
- #92（B1）關閉，不執行。
- #23（2D／Gazebo 一致性）併入 #88，在 G1 之後量測。
- #35 改為 formal-v1 之後研究的唯一入口。文獻對照見 [related_work.md](related_work.md)。
- 新增 V1（#97）：報告和展示用的 2D 軌跡、DN 活性與動作圖。

**Gazebo G1**（PM 在 2060 上執行，[#88 6718765936](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6718765936)）
- **版本與交付**：main `c339c0c`，ROS 映像 `c877bc6351a3`；release `formal-v1` 的 `gazebo_g1_2060.tar.gz`，SHA-256 `2ceba677353a7cc6d3ea606aa4ac5a9d5760c9a9bdcf746d209ec6e2220a70a9`。
- **通過的項目**：50 回合全部 valid；每回合的 `target_motion` 和環境 seed 都正確；全部 48 分鐘。
- **問題 1：policy 只有 3–11 Hz**，應該是 20 Hz。
  - 離線量，VO-late 單執行緒的推論 p50 是 25.6 ms，模型本身夠快。
  - 原因是 torch 預設的執行緒和 Gazebo 的軟體渲染搶 CPU。policy 只用 1 個執行緒，就回到 19 Hz（[診斷 6718924373](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6718924373)，`gazebo_g1_2060_diag.tar.gz`）。
- **問題 2：超過 99% 的命令沒有感測時間**，算不出 G3 的完整延遲。推論是 policy 端的 SensorTiming 大多被覆蓋，數字吻合，但還沒直接驗證。
- **處理**：
  - G0b 提案：compose 讓 policy 的執行緒數可以設定，計時的對應改由 evaluator 補齊；
  - 4070 的 G1 等 G0b 之後再跑；G2 等 G1 驗收後另發 Decision；
  - 2060 不用於 G2 的正式格子，除非 G0b 之後每格都 ≥ 18 Hz。

## 2026-11-08 探索實驗 R1：遮蔽＋氣味 teacher 的 BC 配方（#4，判定無效）與重訓變異 R0（#28，穩定）

兩個實驗都用 seed 0、1，探索評估區段（`environment_seed` 1800000000），curved 四個情境，每格 100 回合，基準是同 seed 的 formal-v1 VO-late actor。

**R1**
- **做法**：VO-late 和 O 的 BC 改用 4-R8 的 B 配方（normal 50%，遮蔽 8 s／30 s 50%，氣味 teacher 增益 8），其他條件和 formal-v1 相同。
- **執行**：@MiLong9487 在 4060 主機上執行，main `8ae904c`。
- **交付**：release `formal-v1` 的 `r1_recipe.tar.gz`，SHA-256 `685801b6f50e1f2121518169849b79c005bfa9b247028cbfec9310ef5edd3f1d`。
- **驗收**：[Decision 6715133955](https://github.com/Nues0913/Drosophila_Tracking_System/issues/4#issuecomment-6715133955)。

| seed | occlusion 成功率（基準 → R1） | normal 成功率（基準 → R1） | 遮蔽時拿掉氣味的降幅，R1 減基準 |
| --- | --- | --- | --- |
| 0 | 0.00 → 0.07 | 0.71 → 0.35 | 0.00 |
| 1 | 0.14 → 0.09 | 0.48 → 0.52 | 0.00 |

- **判讀**：規則是事先固定的，兩個 seed 都要同時達到：occlusion +0.10、normal 下降不超過 0.05、拿掉氣味的降幅比基準多 0.05。兩個 seed 都有未達標的項目，判定為無效。
- **結論**：VO（4-R8）和 VO-late、O（R1）都試過在遮蔽時給氣味轉向標籤，都沒有讓模型用上氣味方向。BC 配方路線到此為止，之後只剩 P3／PPO（#35）。

**R0：formal 配方原樣重訓**
- **目的**：S1 和 R1 的改動完全不同，seed 0 卻都失去轉向，所以先量 formal 基準在重訓下的變異。
- **做法**：用 F1／F2 包裡的正式 `bc.yaml`，只改輸出路徑。
- **執行**：PM 在 RTX 2060 上執行，main `8ae904c`。formal-v1 的 seed 0、1 原本分別在 4060、4070 上訓練。
- **交付**：release `formal-v1` 的 `r0_retrain_variance.tar.gz`，SHA-256 `8762744c9a284a3360aed2a3cd5b2c7a31e9115c7eed9296c5fbe1a5e0591ce4`。
- **結果**：[#28 6716313879](https://github.com/Nues0913/Drosophila_Tracking_System/issues/28#issuecomment-6716313879)。

| seed | normal 成功率（formal → 重訓） | 判讀（事先固定：差 ≤ 0.10 為穩定，≥ 0.20 為不穩） | S1 的差 | R1 的差 |
| --- | --- | --- | --- | --- |
| 0 | 0.71 → 0.72 | 穩定 | −0.53 | −0.36 |
| 1 | 0.48 → 0.38 | 穩定（剛好在界線上） | +0.18 | +0.04 |

- **seed 0**：重訓幾乎和 formal 相同，100 回合裡有 99 回合結果一樣。所以 S1 和 R1 在 seed 0 的下降，是改動造成的，不是重訓的變異。
- **seed 1**：
  - 重訓從 DAgger 第 3 輪開始分岔，得到的是另一個比較差的模型。
  - occlusion 0.14 → 0.01，noise 0.39 → 0.27。
  - R1 在 seed 1 的 +0.04 小於重訓本身的差，不能歸因於 R1。
- **之後的探索規格**：如果每個 seed 的判讀門檻低於 0.10，規格要附至少 1 個重訓基準，或改用更多 seed。
- formal-v1 已登記的結果不變，S1 和 R1 的判讀也不變。

## 2026-11-08 探索實驗 S1：速度讀出校準（#95，判定無效）

- **目的**：測試讓速度讀出能調速，是否能改善 normal 的成功率。
- **做法**：
  - 對 DNp09 的速度讀出輸入做固定的中心化和 `1/std` 放大，做法比照 B3。常數取自 round 0 的 teacher 回合。
  - 其他條件和 formal-v1 的 VO-late BC 完全相同。
  - 範圍：seed 0、1；探索評估區段（`environment_seed` 1800000000）；curved 四個情境，每格 100 回合。
- **執行**：
  - PM 在使用者的 RTX 2060 主機上執行。程式是未合併的 PR #96 的 head `1796491`，PR 已關閉，分支保留。
  - 交付：release `formal-v1` 的 `s1_speed_readout.tar.gz`，SHA-256 `31676e00b240b3c957e686b55691c0256758a4c46b8edbc5077ae08e8ff3cb10`。
  - 驗收：[Decision 6714308434](https://github.com/Nues0913/Drosophila_Tracking_System/issues/95#issuecomment-6714308434)。

| seed | normal 成功率（基準 → S1） | occlusion 成功率（基準 → S1） | 回合內速度 std 中位數（門檻 0.01 m/s） |
| --- | --- | --- | --- |
| 0 | 0.71 → 0.18 | 0.00 → 0.04 | 0.0099 |
| 1 | 0.48 → 0.66 | 0.14 → 0.01 | 0.0094 |

- **判讀**：規則是事先固定的，兩個 seed 都要同時達到 normal +0.10、occlusion 下降不超過 0.05、速度 std ≥ 0.01。兩個 seed 都有未達標的項目，所以判定為無效。
- **速度確實開始變動**：
  - 排除起步加速，回合內速度 std 約 0.007 m/s，基準是 0。
  - 在 teacher 控制的回合上，S1 actor 的速度和 teacher 速度的相關：seed 1 是 0.54；seed 0 只有 0.03，速度會變，但和 teacher 無關。
- **seed 0 壞在轉向**：方位誤差從 12° 變成 58°。這是 trace 的 normal 第 0–19 回合；100 回合評估的平均是 9.6° → 53.1°。
- **原因**：放大倍數約 2.4e5，它同時放大了速度誤差往共用上游網路傳的梯度。
  - 在 round 0 量測，速度誤差的梯度是轉向誤差的 600–5,000 倍，基準只有 0.003–0.02 倍。
  - 結果是上游的更新被速度誤差主導。
- **處理**：
  - 不再接著試變體。在同樣 2 個 seed 和同一個評估區段上反覆試，判讀會失去意義。
  - formal-v2 要不要處理速度，由使用者決定。
  - B1（#92）後來關閉，不執行，見上方 D2 一節。
  - formal-v1 已登記的結果不變。

## 2026-11-08 協定必交的分析 A1 驗收（#90，描述性診斷）

- **交付**：[#90](https://github.com/Nues0913/Drosophila_Tracking_System/issues/90)（4060 CPU，@MiLong9487）。
  - 程式：A1-1 用 main `8ae904c` 上沒改過的 `lesion_analysis.py`；A1-2 和統計用 PR #93 的腳本（main `87ec9f6`）。
  - 封裝：release `formal-v1` 的 `formal_v1_analysis_a1.tar.gz`，SHA-256 `fa37e605a61c56823535b72e18e3df778c4090e3370b2e6d9280bb7381b5dfa0`。
- **PM 驗收**：[Decision 6711664543](https://github.com/Nues0913/Drosophila_Tracking_System/issues/90#issuecomment-6711664543)。
  - 用 PM 自己的腳本獨立重算，lesion 差值、一致率、分段電流都和交付逐位元相同；
  - `lesion=none` 和 F3／#87 的 success、collision 一致率都是 1.00。
- **定位**：全部是描述性的診斷，不是新的比較或主張。
- **範圍**：VO、VO-late、RS、事後 GRU；seed 0–4；normal、occlusion、conflict；lesion none／vision／odor；每格 20 回合，共 3,600 回合，環境 seed 和 F3 配對。

| curved 成功率的 Δ（none − lesion） | normal −vision | normal −odor | occlusion −vision | occlusion −odor |
| --- | --- | --- | --- | --- |
| VO | +0.43 [+0.31, +0.54] | −0.05 [−0.18, +0.03] | +0.17 [+0.04, +0.30] | +0.06 [−0.01, +0.12] |
| VO-late | +0.47 [+0.33, +0.62] | +0.13 [+0.02, +0.24] | +0.11 [+0.04, +0.21] | +0.11 [+0.04, +0.21] |
| 事後 GRU | +0.21 [+0.12, +0.29] | +0.21 [+0.12, +0.29] | +0.05 [+0.01, +0.09] | +0.05 [+0.01, +0.09] |

- **視覺**：四個變體拿掉視覺後，成功率都降到 0，方位誤差增加 25–59°。
- **氣味**：
  - VO 拿掉氣味，成功率沒有下降。
  - VO-late 拿掉氣味，成功率下降 0.11–0.13。但它的 odor lesion 會讓**整個氣味支路靜默**，連這條支路非方向性的貢獻也一起拿掉。
  - conflict（誘餌氣味）對 VO-late 完全沒有影響：五個 seed 的成功率都和 normal 相同。所以這個下降**不代表**它用了氣味的方向。
  - 事後 GRU 拿掉氣味會降到 0，原因也是輸入分布改變，不是用了氣味。
- **匯流電流**：
  - VO 和 VO-late 的匯流電流約九成是 odor_exclusive，遮蔽和非遮蔽兩段的差，CI 都含 0。
  - 但 VO 拿掉氣味後行為幾乎不變，所以電流質量的比例不能當成行為貢獻。
- **失敗案例**：各取 10 個回合。
  - **occlusion 的碰撞**：都是撞牆。延續遮蔽開始時的朝向直線前進，直線度中位數 0.96，沒有打轉，也沒有回頭找。
  - **normal 的距離失敗**：多數是跟在 2 m 門檻外一點，速度不夠。
- **前進速度是常數**（PM 補充發現）：
  - **現象**：VO 和 VO-late 每個 seed 的速度輸出都固定在 0.168–0.179 m/s，回合內標準差 ≤ 2.5e-6；RS 的速度會在 0.17–0.35 m/s 之間變化。curved 的目標速度固定是 0.200 m/s。
  - **原因**：structured readout 的速度只來自 DNp09。它在 no-KC 圖上只有 7–10 條輸入邊，轉向 DN 有 45–78 條；離視覺 3 跳、離嗅覺 5 跳，活動幾乎不變。
  - **反事實測試**：只在記憶體裡把 VO-late 的速度固定在 0.21 m/s，seed 0、1 各 10 回合。距離失敗大致消失，但多出的碰撞都是撞上目標，成功率沒有提高（0.80→0.50、0.30→0.30）。
  - **含意**：常數速度比目標慢就會落後，比目標快就會撞上目標。G3 的 normal 成功率要靠讀出能調速，加大 BC 預算解決不了。
  - **證據包**：`pm_a1_speed_check_Nues0913.tar.gz`（SHA-256 `17999cab45e23e55749543bcbda20c263a82905b2ea867529959524566a98aa5`）。
  - **處理方式**：讀出的設計會影響已登記的架構，所以另行決定，已登記的結果不變。

## 2026-11-04 GRU 修正與事後評估（#85，未登記）

- **修正**：PR #86 已合併（main `93ec7cd`）。
  - 只有 GRU 打開 `gru_odor_gain_compensation`，在輸入瓶頸前把氣味 seed features 乘上 `1/odor_input_gain`。
  - hidden 16、瓶頸 18、有效參數 14,904 都不變。
  - 舊的 card 載入時補成 `False`；5 個已登記的 GRU actor 在新程式上逐位元重播相同。
  - 訓練前門檻（torch seed 0–4）：|x|>3 比例 0.000，視覺／氣味 std 比 0.056–0.116。
- **重新訓練和評估**：[#87](https://github.com/Nues0913/Drosophila_Tracking_System/issues/87)（4060，@MiLong9487）。
  - 交付在 release [`posthoc-gru-85`](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/posthoc-gru-85) 的 `posthoc_gru_85.tar.gz`，SHA-256 `ed944a719e3926c67e8d6823feae439e14664870dda45b65b3ac7a572a952673`。
  - 驗收見 [Decision 6631062573](https://github.com/Nues0913/Drosophila_Tracking_System/issues/87#issuecomment-6631062573)。
  - 新 GRU 和已登記 GRU 的 `model_config` 只差這個欄位，預算帳本相同；評估回合的環境 seed 和 F3／F4 的 VO 逐回合相同。
- **事後彙總**：用 main `b2b0d99`（PR #89）的 `evaluate_gru_posthoc_registered.yaml` 和 `evaluate_wander_gru_posthoc_registered.yaml`，在 CPU 上重新彙總，不重跑模擬。
  - GRU 換成 #87 的 actor 和回合（`recorded_variant_files`），其他變體沿用 formal-v1 的 actor 和 F3／F4 回合。
  - 只採用含 GRU 的列。其他五個變體的 summary 列和比較列，都以已登記的結果為準：所有列共用同一條 bootstrap 串流，GRU 又排在最前面，它的資料一變，後面各列的 CI 都會小幅移動（成功率最多 0.004，方位誤差最多 0.78°），平均值則不變。
  - 結果在 release `posthoc-gru-85` 的 `posthoc_gru_85_summary.tar.gz`，SHA-256 `568c47ecb645461b3319a151b328582b5dfa41d73feb64143b67555afc312979`。和 #87 驗收時的本機彙總逐位元相同，MiLong 在 PR #89 的 review 也獨立重跑確認過。
- **定位**：結果一律標為**事後、未登記**，和已登記的結果並列；已登記的結論不改。
- **重現方式**：main 的 `variants.yaml` 從 `93ec7cd` 起產生的是事後 GRU；要重現已登記的 GRU，必須用 `35432c3`。

| curved 成功率 | VO | VO-late | 事後 GRU | 已登記 GRU |
| --- | --- | --- | --- | --- |
| normal | 0.424 | 0.484 | 0.294 [0.180, 0.384] | 0.000 |
| occlusion | 0.136 | 0.124 | 0.052 [0.012, 0.092] | 0.000 |
| noise | 0.176 | 0.234 | 0.138 [0.048, 0.260] | 0.000 |
| conflict | 0.408 | 0.484 | 0.290 [0.198, 0.372] | 0.000 |

| 不劣性（下界 > −0.05） | normal | occlusion | noise | conflict |
| --- | --- | --- | --- | --- |
| VO − 事後 GRU（curved） | +0.130 [+0.040, +0.214] | +0.084 [−0.018, +0.178] | **+0.038 [−0.114, +0.136]，不成立** | +0.118 [+0.044, +0.174] |
| VO-late − 事後 GRU（curved） | +0.190 [+0.064, +0.338] | +0.072 [−0.036, +0.166] | +0.096 [+0.020, +0.188] | +0.194 [+0.062, +0.336] |
| VO − 事後 GRU（wander） | **+0.006 [−0.080, +0.096]，不成立** | +0.010 [−0.004, +0.030] | +0.032 [−0.020, +0.080] | **−0.002 [−0.106, +0.098]，不成立** |
| VO-late − 事後 GRU（wander） | **+0.080 [−0.128, +0.268]，不成立** | +0.010 [−0.008, +0.024] | +0.146 [+0.070, +0.212] | **+0.060 [−0.138, +0.226]，不成立** |

- **curved（主要運動）**：主架構 VO-late 對事後 GRU 的四個情境都達成不劣性；原登記的 VO 在 noise 不成立。
- **wander（次要運動）**：VO 和 VO-late 在 normal、conflict 都不成立。wander normal 的成功率是事後 GRU 0.210、VO 0.216、VO-late 0.290。
- **對已登記結論的影響**：已登記的「對 GRU 不劣」在 curved 的 VO noise，以及 wander 的 normal、conflict，只是因為 GRU 退化才成立，不能據此主張。
- **不做的主張**：VO-late − 事後 GRU 有幾項 CI 下界大於 0，但協定沒有登記這項優越性檢定，不做主張。區間都是名目的，沒有做多重比較校正。
- **BC 後的瓶頸飽和**（只記錄，PM 量測，條件和訓練前門檻相同）：
  - 事後 GRU 的 |x|>3 比例是 0.26–0.58，視覺／氣味 std 比 0.057–0.079；拿掉視覺的 |Δω| 是 0.100–0.140。
  - 已登記的 GRU 是 0.93–0.94，std 比 0.0003–0.0005。
- **seed 間變異**：curved normal 的事後 GRU 五個 seed 是 0.24、0.36、0.09、0.36、0.42。

## 2026-10-31 2D 正式評估完成並驗收（F3 curved、F4 wander）

- **版本**：評估用凍結的 main `35432c3`，30 個 actor 和上一節相同，設定檔不加覆寫。
- **登記比較的重新彙總**：用 main `8ae904c` 的 `evaluate_registered.yaml`、`evaluate_wander_registered.yaml`，在 CPU 上重新彙總，不重跑模擬（PR #83）。
  - 補上主架構 VO-late 的已登記比較。
  - VO 的列、summary、seed_means 和評估器原始輸出逐位元相同。
  - 重新彙總結果在 release `formal-v1` 的 `formal_v1_registered.tar.gz`，SHA-256 `80629205a22a335e2ffd5812a9a01da921a3bb4bd517bde7fdb6de28bbaac685`。和 PR #83 合併前的 dry run 逐位元相同。

| 檔案 | 內容 | 主機／執行者 | SHA-256 |
| --- | --- | --- | --- |
| `formal_v1_eval.tar.gz` | F3：curved 四情境 × 30 個 actor × 100 回合 | 4060／@MiLong9487（#80） | `23ef7834e17ee3964809c0ad39e6dcdcb655ab53efdf54af5f552bd51ee51a90` |
| `formal_v1_eval_wander.tar.gz` | F4：wander 次要格子，其餘相同 | 4070／@cweiting731（#84） | `dc57e9fc7466932acad6d8888cbb5a72cd88351e4e9c793c5c6ef3aba6e15745` |

- **PM 驗收**：[F3 的 Decision](https://github.com/Nues0913/Drosophila_Tracking_System/issues/80#issuecomment-6497451002)、[F4 的 Decision](https://github.com/Nues0913/Drosophila_Tracking_System/issues/84#issuecomment-6541430822)。兩包都核對了：
  - fingerprint `e41c9aad…b68e`；
  - 環境審計：teacher 1.00、blind 0.00；
  - 12,000 回合的環境 seed；
  - 每回合的 model card 都等於 30 個已驗收的 actor。

### 主要終點：curved 的成功率
數字是五個 seed 平均，括號是 95% bootstrap CI。

| 情境 | V | O | VO | VO-late | GRU | RS |
| --- | --- | --- | --- | --- | --- | --- |
| normal | 0.390 [0.358, 0.416] | 0.002 | 0.424 [0.376, 0.480] | 0.484 [0.378, 0.598] | 0.000 | 0.004 |
| occlusion | 0.112 [0.066, 0.158] | 0.002 | 0.136 [0.040, 0.232] | 0.124 [0.052, 0.186] | 0.000 | 0.004 |
| noise | 0.158 [0.116, 0.204] | 0.006 | 0.176 [0.114, 0.256] | 0.234 [0.152, 0.324] | 0.000 | 0.000 |
| conflict | 0.390 [0.360, 0.416] | 0.002 | 0.408 [0.362, 0.452] | 0.484 [0.370, 0.598] | 0.000 | 0.006 |

| 已登記比較（curved） | 情境 | 差值 [95% CI] | 判讀 |
| --- | --- | --- | --- |
| VO − V（優越性） | occlusion | +0.024 [−0.112, +0.160] | 不成立，所以不做 VO 融合的主張 |
| VO-late − V | occlusion | +0.012 [−0.062, +0.082] | 不成立 |
| VO、VO-late − GRU／RS（不劣，界線 0.05） | 四個情境，共 16 項 | CI 下界最低 +0.032 | 名目上都達成，但對照組退化，見下方 |

### 判讀與揭露
- **對照組退化**（[#85](https://github.com/Nues0913/Drosophila_Tracking_System/issues/85)）：GRU 在 curved 和 wander 的成功率都是 0，RS 最高 0.006。所以對它們的「不劣」不能當成連接體結構的證據。
  - **GRU**：共用的 `odor_input_gain: 100` 讓輸入瓶頸的 tanh 有 92–93% 飽和，看不到視覺。這是工程缺陷。團隊選 C：修正後事後追加 GRU，結果標成「未登記」，已登記的結果不變。修正和事後結果見上方 2026-11-04 一節。
  - **RS**：照固定標準診斷，不是工程缺陷。隨機接線讓 ORN 直接接到轉向 DN；BC 後視覺的影響只有同 seed VO 的 1/66–1/248。RS 照登記結果報告；五個 seed 都用同一張隨機圖。
- **G3**（normal 成功率 ≥ 0.8）：2D 沒有變體達到。curved 最高是 VO-late 0.484，wander 最高是 0.290。Gazebo 還沒有跑（NOT RUN）。
- **O**：正式配方下 O 沒有用氣味追蹤。成功率 0.002–0.006，方位誤差 77–86°，碰撞率 0.81–0.88。4-R6～4-R8 的 O 證據只限長遮蔽加氣味 teacher 的配方。
- **seed 間變異大**：例如 curved normal 的 VO-late 五個 seed 是 0.71、0.50、0.34、0.36、0.51。
- **wander（次要格子）**：
  - VO − V 遮蔽是 +0.010 [−0.004, +0.024]。
  - VO-late − V 遮蔽是 +0.010 [+0.004, +0.016]。CI 下界大於 0，但兩邊成功率只有 0.020 和 0.010，貼近下限，也沒有多重比較校正，所以不做主張。
  - 完整數字見 #84 的 Decision。
- **延遲**：2D 的 actor 呼叫延遲是 wall time，每格量第 1 回合，p95 最高 curved 11.2 ms、wander 12.1 ms（都是 VO-late）。這不能替代協定 50 ms 的 Gazebo sensor-to-command 量測。
- **失敗分解**（PM 第一層，用 F3 回合，不重跑模擬）：
  - normal 的失分主要是距離：VO-late 的成功率 0.48、碰撞 0.06，「沒碰撞但 2 m 內的時間不到 80%」佔 0.44，方位只佔 0.01。
  - occlusion 的失分主要是碰撞：VO、VO-late、V 的碰撞率都是 0.46–0.48，碰撞時間的中位數約 29–31 s，落在第 8–38 s 的遮蔽期間。
  - VO 在 conflict 和 normal 的成功率幾乎一樣（0.41 對 0.42）。
  - 逐步軌跡和 lesion 分析已在 #90 完成並驗收，見上方 2026-11-08 一節。
- **NOT RUN**：Gazebo（#88）、展示。

## 2026-10-29 正式矩陣的 30 條訓練鏈完成並驗收

- **版本**：main `35432c3b62e20d1d69963d4a66dae1b6d41f89e4`，協定 schema 3（amended 2026-10-28）。凍結與派工見 [#28 的 Decision](https://github.com/Nues0913/Drosophila_Tracking_System/issues/28#issuecomment-6471391440)。
- **交付位置**：GitHub release [`formal-v1`](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/formal-v1)（prerelease）。`runs/` 和權重不進 Git。

| 檔案 | 內容 | 主機／執行者 | SHA-256 |
| --- | --- | --- | --- |
| `formal_v1_4060_F1.tar.gz` | seed 0、2、4 × 六變體，18 條鏈 | 4060／@MiLong9487（[#80](https://github.com/Nues0913/Drosophila_Tracking_System/issues/80)） | `2f0e2acd6adcf9add500e2b54f116661e1cffffa795daa2e47078d315cea44ac` |
| `formal_v1_4070_F2.tar.gz` | seed 1、3 × 六變體，12 條鏈 | 4070／@cweiting731（[#81](https://github.com/Nues0913/Drosophila_Tracking_System/issues/81)） | `be05b1c780e9531f601136b6b50dc156d8ca2a612e7721c5770f0597f80fcffa` |

- **PM 驗收**（[#80 的 Decision](https://github.com/Nues0913/Drosophila_Tracking_System/issues/80#issuecomment-6489237904)）：
  - 80 張 card（30 個 BC、50 個 pretrain）的 `source_code.sha256` 都等於 `35432c3` 的值 `e41c9aad…b68e`。
  - 在凍結程式上，`validate_artifact_matrix` 對 30 個 actor 通過：有效參數由 actor 重算、預算 ledger 一致、graph 和 frontend 共用。
  - 沒有終端 BC 的工程類失敗；O、VO、VO-late 的 15 個 G4 全部通過。
- **轉向品質類 dn_health 失敗**：依 schema 3 只記錄，actor 照常評估，報告逐組揭露。25 組裡有 17 組（GRU 沒有 DN，不適用）：

| 變體 | 有失敗的 seed 與項目 |
| --- | --- |
| V | s0：omega_r2；s3：omega_correlation |
| O | 五個 seed 都有 omega_correlation、omega_r2；s2、s3 另有 omega_amplitude |
| VO | s1、s4：omega_r2；s3：omega_correlation、omega_r2 |
| VO-late | s0：omega_r2；s3：omega_correlation、omega_r2 |
| RS | 五個 seed 都有 omega_correlation、omega_r2、eye_swap_response；s0、s1、s2 另有 omega_amplitude |

- **G4 只當 gate**：它只量一個 teacher 回合（seed 1,610,000,000）。同配方、同 seed 的 VO 重訓一次，在同一回合上的值就差 8 倍，所以不能拿 G4 比較效應大小（[#4 的補充](https://github.com/Nues0913/Drosophila_Tracking_System/issues/4#issuecomment-6489238229)）。
- **O 的證據範圍**：4-R6～4-R8 的 O 是在長遮蔽加氣味 teacher（k = 8）的世界訓練的，和正式 O（只用 normal、沒有氣味 teacher）配方不同（同上）。F3 顯示正式 O 沒有用氣味追蹤，見上方 2026-10-31 一節。
- **執行參數的差異**：worker 數由 `auto` 決定，4060 是 7、4070 是 8。收集器依回合編號提交，資料和 worker 數無關。同一個 seed 的六個變體在同一台主機上跑，跨主機依 #21。
- **時間**：4060 13.42 小時、4070 7.80 小時（`--jobs 2`），見 [training_time.md](training_time.md)。

## 2026-10-28 有效參數預算（#77）

正式 ±5% 以 **BC optimizer 的 lr > 0 且 requires_grad=True 的 actor 參數**為準，包含 B3 校準新增的讀出列，排除 lr=0 的核心 bias 與共同凍結前端。`declared` 只代表 requires_grad 容量，`zero_lr` 是其中 lr=0 的部分，`effective = declared - zero_lr`。

| 變體 | declared | zero_lr | effective | 相對有效 VO |
| --- | ---: | ---: | ---: | ---: |
| V | 16,242 | 1,697 | 14,545 | −0.0069% |
| O | 16,243 | 1,697 | 14,546 | 0% |
| VO | 16,243 | 1,697 | 14,546 | 0% |
| VO-late | 15,790 | 1,697 | 14,093 | −3.1143% |
| GRU | 14,904 | 0 | 14,904 | +2.4612% |
| RS | 16,242 | 1,697 | 14,545 | −0.0069% |

舊 GRU hidden=7、無 bottleneck，15,516 個有效參數相對 VO 為 +6.6685%，超出原 ±5%；這是 #77 修正前 main `79eff41` 的歷史配置。以有效 VO=14,546 走既有解析器後為 hidden=16、`gru_input_size=18`（有作用的輸入 bottleneck），effective 為 14,904（相對 VO +2.46%），六個變體都在 ±5% 內；此配置已獲 PM 接受。連接體的 `budget_target=null`，由產生器檢查有效預算；只有 GRU 以此鍵解析架構。`parameter_budget.json` 的舊鍵 `actor_trainable_parameters` 明確保留為 declared，新增 `parameter_counts`、BC lr 與逐組明細。card 的 `metadata.actor_parameter_budget` 記錄各自階段的分組計數；正式評估用載入的 actor 與保存的 `metadata.config.train` 重算，核對 declared／zero_lr／effective、每組 lr／numel／參數名稱與數量、zero_lr_parameters 及頂層 trainable_parameters，缺少、無效或不一致皆拒絕，±5% 使用重算值。修正內容已由 [PR #79](https://github.com/Nues0913/Drosophila_Tracking_System/pull/79) 合併（main `35432c3`）。初輪 CPU 證據在 `runs/r77/summary.md`，本輪 review 修正證據在 `<runs>/r77_fix/`。

## 2026-10-28：試跑已交付，協定記錄完成，正式矩陣仍待 #79 審查與版本凍結

- 兩個 seed 的 VO A 都已交付。seed 1 的 G4 通過，但 G3 原登記 100 對的 D=−26.09°，氣味移除反而降低方位誤差；A 不能記為複製成立。seed 0 的原始 G3 受轉向偏差補償干擾，任何引用都須並列去偏差結果。
- 依 [#4 選 F 的 Decision](https://github.com/Nues0913/Drosophila_Tracking_System/issues/4#issuecomment-6442266901)，BC 維持 normal-only、不用氣味 teacher、不做 noise 擴增、不做偏差置中；P3 留到正式矩陣之後。
- #9 的雙 seed 選點是 **0.75 倍**；正式遮蔽是第 8 秒起 30 秒。schema 3 記錄情境強度、#25 的 G4／終端 BC 失敗處理、wander 次要比較及報告範圍。所有原 VO／VO-late 對 V／GRU／RS 的已登記比較都保留。
- **#77 已修正（PR #79 已合併）**：BC-only 鏈全程凍結 1,697 個核心 bias，但舊預算仍將它們計入。固定 `79eff41` 的非零學習率計數為 VO 14,546、GRU 15,516，差距 6.67%，超過既有 ±5%；這是 #77 修正前的歷史版本。目前有效 VO 為 14,546，GRU hidden=16 加 bottleneck=18 的 effective 為 14,904（+2.46%），六個變體都在 ±5% 內。修正內容在 PR #79，已合併。
- #10 外推：按 VO 試跑估計訓練＋主要 2D 評估的串行等效約 4060 23.9 h、4070 12.0 h，**NOT VERIFIED**，不當成上限、不把 jobs=2 當兩倍加速。保留 48–72 小時工作目標，不減 seed 或步數。

原始包、逐回合重算、情境參數、限制與時間口徑見 [協定凍結依據](protocol_freeze_2026-10-28.md)。之後 PR #79 已合併，PM 在 #28 凍結 main `35432c3` 並開了執行 Issue，見上方 2026-10-29 一節。

## 2026-10-24 正式訓練鏈改為 pretrain → BC/DAgger（#35 選項 D）

從 BC 開始的 PPO 在 C9–C13 都沒有穩定改善 BC。C11–C13 一共六次（每輪兩個 seed），只有 C11 s0 通過 protocol100，換 seed 就不成立。C13 用 L2 錨定，把參數限制在離 BC 0.13 以內，兩個 seed 仍然比 BC 差。使用者團隊因此決定（Decision 6382764327）：

- 正式矩陣只跑 pretrain → BC/DAgger。最終 actor 是 `runs/bc/{variant}/seed_{seed}/actor`，`evaluate.yaml` 和 Gazebo 矩陣都讀這個路徑。
- PPO 不刪除，程式和測試都保留，#35 改成不擋 #28 的研究項目。研究方向等正式矩陣跑完一次後再決定。要加回正式鏈，必須另發 Decision，並在正式規格的所有變體、每個至少 2 個 seed 都通過 protocol100。
- 正式結果的範圍是模仿學習，上限是 teacher 的水準，不宣稱 RL 帶來改善。`configs/evaluation_protocol.yaml` 的 2026-10-24 修訂把 PPO 步數改為 0。
- 2026-10-24 的本機探索（只更新 readout、限制轉向的累積變化）在 curved/normal 有小幅、可重現的回報改善，但 protocol100 五個 seed 都不通過，碰撞率也上升。這記為 #35 研究的第一個候選方向。

## 2026-10-24 正式模型規格：no-KC 圖 + P2 + B3

#4 的 4-R7 在兩個 seed 上都複製成立：P2（左右共用的 ORN 適應）加上 B3（DNb05 轉向列）後，O 變體在長遮蔽下會用氣味轉向。使用者團隊因此決定把這個組合納入正式規格。驗證是在 no-KC 圖上完成的，所以正式預設圖也改成 no-KC（#55）。

- 實作方式見 `python/flytrack_train/TRAIN_NOTES.md` 的「正式模型規格」。B3 只用在 O、VO、VO-late。
- VO 試跑已交付，結果與後續範圍見上方 2026-10-28 摘要。
- 遮蔽期間 teacher 的標籤本身就是原始左右比例的函數，報告必須揭露。
- DNb05 的生物依據已由 #27 查證；相關性不代表因果擾動證據。

## #55 重新萃取候選（v2 已併入；no-KC 在 2026-10-24 改為正式圖）

`olfactory_v2` 是 PR #56 併入的歷史候選：6,244 個節點、35,627 條邊，嗅覺群 3,463、三跳內匯流 59。正式預設已由 PR #67 切換為 `olfactory_v2_nokc`；舊的 `data/subgraphs/main` 保留供歷史證據使用。

兩台 GPU 的有界校準都沒有 OOM。v2 時期的舊鏈外推為 4060 約 58 小時、4070 約 42 小時（NOT VERIFIED），不能替代目前 no-KC、BC-only 的時程。

no-KC 排除 Kenyon cell、APL 和 DPM 後是 4,502 個節點、31,682 條邊。共同 `dense_threshold=4000`，讓它走 sparse core。通路、活動隔離與初始化檢查見[候選報告](olfactory_reextraction_55.md)；初始化反應不等於 trained 氣味效益。候選時的參數容量檢查不能替代 BC-only 的有效 optimizer 預算（#77）。下方舊圖基線不冒稱 no-KC 的正式結果。

## 一句話

資料 → 前端 → 子圖 → 模型 → 訓練鏈都已實際跑通，兩台 GPU 主機也已校準完成。**正式訓練（6 變體 × 5 種子，pretrain → BC/DAgger）已在 main `35432c3` 完成並驗收**，30 個 actor 在 release [`formal-v1`](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/formal-v1)。2D 正式評估（curved 和 wander）也已完成並驗收：VO 和 VO-late 對 V 的遮蔽優越性都不成立，沒有變體達到 G3 的 normal 成功率 0.8；GRU、RS 對照組退化。GRU 依 [#85](https://github.com/Nues0913/Drosophila_Tracking_System/issues/85) 修正後的事後結果（未登記）：curved 的 VO-late 對它四個情境都不劣，VO 在 noise 不成立，wander 的 normal、conflict 兩者都不成立。下一步是 Gazebo 主要比較（[#88](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88)）。

PPO 已移出正式訓練鏈，改成研究項目（[#35](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35)），不再擋正式訓練。

ROS／Gazebo 部署鏈路已在 Gazebo 實跑通過（[#22](https://github.com/Nues0913/Drosophila_Tracking_System/issues/22)，2026-10-19，CPU／GPU 各 5 回合 invalid 0）；完整端到端延遲 p95 分別為 55.62／56.46 ms，均未達 50 ms。

## 舊圖基線與已驗證鏈路（依各列證據版本）

| 項目 | 結果 | 證據 |
| --- | --- | --- |
| MaleCNS v1.0 原始表格 | 三份官方表格下載，大小、ETag、Arrow 框架、SHA-256 核對 | `docs/data_sources.md` |
| flyvis 前端（方案 A） | 官方 checkpoint 作者 SHA-256 核對；原生轉換對官方輸出最大誤差 1.91e-6 | `artifacts/frontend/verification.json`、`docs/frontend_execution.md` |
| 中央子圖萃取 | 2569 神經元、14165 邊；19 項獨立檢查全過；重跑兩次位元相同 | `artifacts/data_execution/graph_verification.json` |
| 六變體參數預算 | no-KC + P2+B3 的 BC effective 為 14,093–14,904（參考 VO 14,546），全在 ±5% 內；分欄數值見上表。舊 G2 的 6795／6767／6831 為歷史宣告值 | #77（舊值 #36） |
| 測試與 lint | #77 初輪 Codex CPU（`CUDA_VISIBLE_DEVICES=''`）：434 passed、1 CUDA-only skipped，ruff check／format --check 通過；Claude 另在本機含 CUDA 重跑：435 passed，該次使用 GPU。歷史 main 83f57d1 為 165 passed | Codex：`runs/r77/tests.log`、`ruff_check.log`、`ruff_format.log`；Claude：另次執行回報 |
| #77 review 修正驗證 | 本輪 Codex CPU（`CUDA_VISIBLE_DEVICES=''`）：473 passed、1 CUDA-only skipped，新增 39 個參數化案例、改寫 3 項真實 artifact 評估測試；pytest 404.12 s，ruff check／format --check 通過。PM scalar 反例由修正前錯誤通過改為拒絕，正確往返通過 | `<runs>/r77_fix/summary.md`、`tests.log`、`tests_timing.txt`、`ruff_check.log`、`ruff_format.log` |
| 兩台 GPU 主機校準（#21） | 同版本訓練的指標相對差 < 1e-3；同一個 artifact 跨主機評估相對差 1.5e-8／2.7e-8 | release `host-calibration` |
| 環境審計 | 隨機化世界中 teacher 100%、所有盲目策略 0%；訓練與評估前自動檢查 | [fixes_2026-10-17.md](fixes_2026-10-17.md) |
| Docker 映像與 ROS 節點 | `flytrack-ros:0.1` 內 `colcon build`、真值隔離 launch test、latency 單元測試通過；策略節點關閉時會送零速度（#15） | `docs/ros_execution.md` |
| Gazebo 閉環部署鏈路（#22） | 5 回合 reset invalid 0/5、truth coverage 100%、bag 可讀；修正 rosbag `--topics` 與 SIGINT 關閉（#41）、`uint64` 序列化（#40） | `docs/ros_execution.md`、release `explore-4` |
| ROS Python 測試收集（#42） | #44 已合併；colcon 正常收集 4 項 latency 測試且回傳 0，臨時副本改錯預期值後 3 failed 並回傳 1。後續 #47 的 world 取消回呼 5 項、#50 的 policy profiler 4 項及 #53 擴充後 policy 10 項均分別由 colcon 驗證通過；真值隔離 launch_test 獨立執行 | #44 的 PM review、release `explore-4` 的 GPU 追加證據包 |

## 速度（2026-10-18，#21）

| 主機 | V seed 0 視覺 pretrain（240k 步） | PPO 單程序 | PPO 雙程序合計 |
| --- | --- | --- | --- |
| @MiLong9487：RTX 4060 Laptop、Ryzen 7 7735HS、WSL2 | 28.0 分（重測 fit 段 11.4 分） | 239 步／秒 | 351 步／秒 |
| @cweiting731：RTX 4070 Laptop、i7-13700HX、WSL2 | 36.6 分 | 307 步／秒 | 337 步／秒 |

- 4060 最初只有 103 步／秒，主因是 Windows 的電源模式「平衡」。改成「最佳效能」後約 2.2 倍，再關掉「硬體加速 GPU 排程」（HAGS）約再快 20%。設定方式見 [gpu_hosts.md](gpu_hosts.md#windowswsl2-的效能設定)。
- PPO 期間 GPU 使用率約 35–55%。瓶頸還沒定位：推測是每步很小的 GPU kernel 受啟動延遲限制，但沒有用 profiler 或環境 worker 對照驗證過。依 #21 的決定，兩台都用 `--jobs 2`（只量過 1 和 2 個程序）。
- 正式鏈不再跑 PPO（#35 選項 D），原本 2M 步 PPO 的估計不再適用。no-KC 試跑與只跑 pretrain／BC 的外推見[協定凍結依據](protocol_freeze_2026-10-28.md#10時間重估是外推不是完整矩陣實測)；不是全變體或雙程序實測。

## 已修正的重大問題

| 問題 | 修正 | 位置 |
| --- | --- | --- |
| 中央遞迴初始化發散：真實圖的譜半徑約 32，第 9 步就溢位 | 初始化時用冪迭代求 Perron 根，α 取 `min(alpha_init, 0.9/ρ)` | 2026-10-15 |
| GRU 正式設定無法載入（參數數重算成 5691） | 已有 `gru_input_size` 時沿用 | 2026-10-15 |
| 環境退化：固定左轉就能「追蹤」 | 每回合隨機化佈局、新增 wander 目標、成功須 `|方位| ≤ 45°`、`audit-env` 自動檢查 | [fixes_2026-10-17.md](fixes_2026-10-17.md)、#1 |
| 運動 DN 沉默（dead ReLU） | 運動 DN 改用 `softplus(10v)/10`，加上 `dn_health` gate | [dn/fix_A_C_2026-10-15.md](dn/fix_A_C_2026-10-15.md) |
| 情境參數有程式預設和 YAML 兩個來源 | 只用 `configs/scenarios/*.yaml`，缺檔就報錯 | #2、#5 |
| GPU 評估無法逐位元重現（cuSPARSE） | `evaluation.deterministic` 或 `DETERMINISTIC=1` | #3、#6 |
| recovery 只綁定上游路徑、不綁內容 | 加入上游 `model_card.json` 的 SHA-256 | #14 |
| 策略節點關閉時送不出停止命令 | 自行處理 SIGINT／SIGTERM，先送零速度再退出 | #15 |
| **氣味通路被 bias 漂移截斷**：vision pretrain 訓練 bias，把 DNa01 壓到 −0.875，softplus 斜率只剩 1.6e-4 | **G2 設定**：pretrain 凍結 bias、decoder 族群 RMS 縮放、`odor_input_gain: 100`。兩個 seed 合併後，BC 的 normal／occlusion 成功率從 0.25／0.15 升到 0.43／0.40 | #4、#34、#36 |
| **PPO 學會撞牆**：每步 reward ≤ 0，撞牆只扣 10 | 有界的存活獎勵、reward 正規化、critic 不做 value clipping、actor 和 critic 分開裁剪梯度；從 pretrain 開始的 PPO 碰撞率從 1.00 降到 0.07 | #19、PR #32（已合併） |
| 遮蔽時 teacher 用學生看不到的真值轉向，長遮蔽 BC 崩潰 | 新增可選的 `occlusion_odor_steering`：遮蔽期間用學生同一份氣味輸入轉向（預設關閉） | #24、#33 |

## 未解決的問題

### 正式訓練前的主要判定與阻礙

1. **#4 按團隊選 F 交付有限範圍的負面結果。** P2+B3 的 O 氣味追蹤已有複製證據，但只限長遮蔽加氣味 teacher 的訓練世界；正式 O 只用 normal BC，F3 顯示正式 O 沒有用氣味追蹤（成功率 ≤ 0.006）；VO 的氣味左右反應很弱，seed 0 原始 G3 受偏差補償干擾，seed 1 G3 反方向。B／C 不成立，A 不一致。正式 BC 保留 normal-only，主架構 VO-late；原登記比較全部報告，不能由 G4 通過直接宣稱融合成功。P3 留到正式矩陣之後。#4 已在 2026-11-08 結案，後續研究併入 #35。
2. **PPO 沒有改善 BC（#35），已移出正式訓練鏈。** C9–C13 的各種限制（凍結 readout bias、去掉課程、L2 錨定）都沒有在兩個 seed 上通過 protocol100。每次失敗的方式不同（繞圈、追太近而撞上、帶著轉向偏差），參數離 BC 很近也一樣會變差。本機探索（VO、舊圖、只更新讀出層、核心凍結）顯示，在那個設定下更新方向大致正確，崩潰主要來自步幅過衝；但 reward 的改善沒有換成 protocol100 要求的方位和碰撞表現。C13 是全部參數都在更新、離 BC 不到 0.13，兩個 seed 仍然變差，所以這個結論還不能推廣到正式規格。原因還沒拆清楚，留在 #35 研究，不擋正式訓練。
3. **協定交付與正式版本凍結（#20／#28，已完成）。** schema 3 記錄原有成功判定、審計、BC gate、DN 增益 100、主要終點與非劣界值，以及新完成的 noise=0.75 倍、30 秒遮蔽、#25 失敗處理與報告範圍。wander 仍是次要情境，主要終點維持 curved。#77 的有效預算已由 PR #79 修正，PM 已凍結 main `35432c3`（Decision 6471391440），30 條訓練鏈已完成。

### 其他未解決

4. **noise 情境對雜訊參數非常敏感**（bc_B 在兩組參數下 77% 對 0%）。雜訊曲線和 BC 雜訊擴增見 [#9](https://github.com/Nues0913/Drosophila_Tracking_System/issues/9)。
5. **情境強度與研究範圍**：正式 occlusion 已改成第 8 秒起遮 30 秒，noise 用雙 seed 曲線選出的 0.75 倍。這些設定不能保證所有變體都受益；conflict 與 normal 的差異依實際結果報告。
6. **模態 gate 與 BC 失敗分類已實作**（[#25](https://github.com/Nues0913/Drosophila_Tracking_System/issues/25)，PR #76）。O、VO、VO-late 在 BC 後要求 swap ≥1e−3 rad/s 且 ≥同 chain 起點的 3 倍；V／GRU／RS 模態診斷只記錄。終端 BC 品質失敗照樣評估並逐組揭露，工程失敗強制停止整個矩陣，keep-going 不能略過。schema 3 記錄相同規則。
7. ~~`lesion_analysis.py` 用了訓練範圍的種子，並繞過情境設定檔~~：已由 PR #69 修正（#29）。lesion 現在用 `evaluate.yaml` 的世界、card 的訓練 seed 和正式評估同一個種子公式，`lesion=none` 的回合和 `evaluate()` 逐回合相同。lesion 分析前還要注意兩件事（PR #69 review 的建議）：
   - 如果正式評估改成 `deterministic: true`（目前是 false），lesion 腳本也要呼叫 `enable_deterministic()`，GPU 上才會逐位元配對。
   - 可以把「`lesion=none` 等於 `evaluate()`」寫成回歸測試。
8. **linear 目標撞牆反彈會撞上跟隨者**（teacher 13% 碰撞），所以只留在 PPO 暖身課程（研究用）。
9. **ROS／Gazebo**（#22，2026-10-19，沿用舊的 smoke 模型、軟體渲染，只驗鏈路、不代表追蹤品質）：CPU 首輪單回合與 5 回合有效；GPU 追加 5 回合也 invalid 0/5、truth coverage 100%，沒有沿用前一回合的 sample；策略沒有訂閱真值 topic，bag 可以讀回。**CPU／GPU 完整 sensor-to-command p95 為 55.62／56.46 ms，均未達 50 ms**；active 窗口為 54.13／55.09 ms，inference wall p95 為 24.76／24.19 ms。GPU 報告已合併（PR #45），PM 獨立解碼原始 bag 重算相符。兩者各只有一輪，不能推斷 GPU 穩定較慢或唯一瓶頸。依 #20 決策保留門檻與頻率；[#48 的離線診斷](ros_latency_diagnostic_2026-10-19.md) 已核對 4,707 筆同一 sample 的逐段 wall time，慢尾主要落在 forward→received 與 control→published，但尚未定位內部原因。預設關閉的 profiler 探針已由 #50 合併；#49 已從固定 main 1563a93 完成一次有上限的短回放，[#51 報告](ros_policy_replay_profile_2026-10-19.md) 已合併並驗收結案。單次 off/on 分布不代表普遍 probe 開銷，reset 前 trace 缺口的可觀測性已由 #53 的 schema2 狀態與丟棄紀錄補強，#52 結案，歷史缺口保留；simulation-clock sensor age 不與 wall time 相加。#42 測試收集已由 #44 修正；取消 future 的誤報已由 #47 修正，#46 結案。2D／Gazebo 感測分布與碰撞判斷一致性仍未量（原 [#23](https://github.com/Nues0913/Drosophila_Tracking_System/issues/23)，已併入 #88）。ROS 的 `target_motion` 只接受單一字串，正式評估每個 cell 固定一種運動（#20）。但 `world.launch.py`、compose 和 `run_gazebo_matrix.py` 還沒有把 `target_motion` 接出來，目前只能跑 curved；orchestrator 也還沒實際執行過。這兩項已由 [#88](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88) 的 G0（PR #94）和 G1 補上，G1 的結果見上方 2026-11-08 一節。
10. ~~生物對應查證~~：已完成（PR #72，[#27](https://github.com/Nues0913/Drosophila_Tracking_System/issues/27) 已關閉），報告在 [biology_verification.md](biology_verification.md)。沒有需要改圖的發現，也不改規格；要揭露的事項列在下方科學限制的第 4、14、15 點。

## 報告須揭露的科學限制與工程近似

1. **舊圖的三跳匯流只有 4 個。** 這是原本採 VO-late 為主要變體的依據，schema 3 明確標記該數字屬於歷史圖，不能冒稱 no-KC 的盤點。新圖與活動隔離見 #55 報告；主架構及已登記比較仍依選 F 保留，不能把晚期融合說成找到生物匯流。
2. **舊圖的 LC9、LC11 各只保留 1 個神經元**（原有 219、143 個），視覺輸入主要來自 LC10a（208）、LC10d（143）、LPLC2（112）；這些是舊圖的盤點，正式 no-KC 圖依 #55 的產物報表。
3. **ORN_VA2 有 9 個（約 11%）側別未知，已丟棄**；見 `data/subgraphs/main/excluded_odor_seed_sides.csv`。
4. **運動 DN 讀出**：DNp09 即 P9，雙側活化驅動前進（Bidaye et al. 2020, *Neuron* 108:469–485）；DNa02 左右差與轉向速度呈線性，DNa01 追蹤緩慢持續的轉向（Rayshubskiy et al., *eLife* 102230）。模型以 DNp09 左右平均讀前進、DNa01／DNa02 的「左 − 右」讀轉向，B3 另外加入 DNb05 的左右差。DNp09 就是 P9，P9 也會驅動同側轉向（Bidaye et al. 2020），所以「純前進神經元」的說法和文獻不一致；只讀前進是工程選擇，模型沒有用到它的轉向成分（#27）。
5. **輸入增益**：運動 DN 的輸入增益 100、ORN 輸入增益 100 都是工程選擇，用來補償圖的跳數和尺度差，不是生物量測值。
6. **bias 的訓練範圍與有效參數預算**：舊 main 圖核心 bias 為 781 個；正式 no-KC 圖為 1,697 個。pretrain 和 BC 維持凍結，選配 PPO／ES 維持各自原設定。PPO 已移出正式鏈，不能沿用「最後靠 PPO 解凍」的公平性理由。固定 `79eff41` 的有效 VO 為 14,546、GRU 為 15,516（+6.67%）；這是 #77 修正前的歷史計數。#77 已直接排除 BC lr=0 的 bias，以有效 VO 14,546 重新解析 GRU 為 hidden=16、bottleneck=18，effective 14,904（+2.46%），六個變體都在既有 ±5% 內，不靠解凍 bias 達成。修正內容已由 PR #79 合併（main `35432c3`）。
7. **前端來源**：方案 A 的早期視覺來自 flyvis 的資料與學習參數，只有中央子圖使用 MaleCNS。
8. **輸入映射**：沒有真實 RF 座標，LC 輸入採型別／側別的空間池化；ORN Hill 閾值是模擬設定，不是特定氣味的實測受體曲線。
9. **變體口徑**：V／O 是同一張圖的輸入消融。VO-late 共享型別參數但隔離活性；RS 保留原邊參數槽的共享標籤。這些都寫進 model_card。
10. **2D 世界**：2.5D 光線投射，Gymnasium 向量環境是 CPU 上的平行子行程。2D 有大小／亮度的領域隨機化，Gazebo 用固定的幾何和材質，兩者的感測分布還沒證明一致（#23）。
11. **ROS 碰撞**用場地、半徑與障礙幾何判斷，沒有接 Gazebo contact sensor。
12. **時鐘與延遲**：行為指標用模擬時鐘，推論耗時用 wall time。sensor-to-command 的因果計時**已有契約與實作**：`SensorTiming.msg`／`ControlTrace.msg` 定義同一台主機上 `CLOCK_MONOTONIC` 的 source → ingress → forward → received → control → published 時間線，`flytrack_eval.latency.CausalLatency` 檢查時鐘域、因果順序和覆蓋率，有 `test_latency.py` 測試。2026-10-19 的 CPU／GPU 實測均未達 50 ms，詳見第 9 點與 [ros_execution.md](ros_execution.md)。端到端 p95 必須由各筆命令最舊的有效 contributing sample 至命令發布的完整差值計算；各階段 p95、不同 sensor 的階段最大值不可拼成一條因果路徑。
13. **續訓**：同一階段中斷後可以完整接續（權重、optimizer、RNG、環境重播）；跨階段以 `init_artifact` 暖啟動，recovery 綁定上游 actor 的內容（#14）。改了程式碼或設定之後，舊的 checkpoint 不能接續；#36 合併後，舊的 pretrain／BC artifact 都要重跑。
14. **號誌與 NT 信心**（#27）：
    - consensus 讀取不使用數值信心，`nt_confidence_min` 不起作用。
    - 號誌不明的 88 個候選細胞被移除，其中包括已知的單胺神經元，因為它們沒有固定極性。seed 到 DN 的可達性和最短跳數都不受影響，但調質連接不在模型中。
    - Glu 一律為負號、每個細胞的所有出邊同號，是模型假設。觸角葉的 GluClα 支持這個方向，但沒有逐邊驗證。
15. **LC 輸入的功能**（#27）：LC10a 有追蹤和轉向的行為證據；LC9、LC11 偵測小物體；LPLC2 偵測 looming；LC10d 的迴避功能還是候選。報告稱它們為「視覺物體特徵輸入」，不寫成追蹤通道。
16. **對照組退化**（#85）：正式矩陣的 GRU 因共用的 ×100 ORN 增益讓輸入瓶頸飽和，看不到視覺，屬於工程缺陷；RS 依固定標準不是缺陷，但隨機接線加上同一組增益讓視覺送不到轉向輸出，而且五個 seed 共用一張隨機圖。對兩者的不劣性都不能當成連接體結構的證據；修正後追加的 GRU 一律標成「未登記」。事後 GRU（#86／#87）的結果顯示，已登記的「對 GRU 不劣」在 curved 的 VO noise，以及 wander 的 normal、conflict（VO 和 VO-late），只是因為 GRU 退化才成立。
17. **G3 未達成**：2D 的 normal 成功率最高 0.484（curved，VO-late），低於登記門檻 0.8；Gazebo 延遲也還未達 50 ms（第 12 點）。
18. **前進速度是常數**（#90）：VO 和 VO-late 的速度只由 DNp09 讀出，DNp09 在圖上的輸入很少，所以速度固定在 0.168–0.179 m/s，低於目標速度 0.200 m/s。normal 的距離失敗和遮蔽期間直線撞牆都和這一點有關。報告不能把 G3 未達成歸因於訓練預算，也不能把 VO-late 拿掉氣味後的下降說成它用了氣味的方向。探索實驗 S1（#95）把速度讀出的輸入放大，速度開始會變，但轉向或遮蔽的表現變差，判定無效，所以這個限制在 formal-v1 仍然成立。
19. **沒有風的輸入**：果蠅在 turbulent plume 裡主要靠風向導航，遇到氣味就逆風走（Álvarez-Salvado et al. 2018、Demir et al. 2020、Matheson et al. 2022）。本模型只有左右 ORN 的濃度，沒有風或機械感覺的輸入。所以 VO 沒有用氣味方向、正式 O 沒有用氣味追蹤，不能完全歸因於模型或訓練（[related_work.md](related_work.md) 第 2 節）。
20. **記憶的時間尺度**：正式設定的神經時間常數初值是 50 ms。從 25 個連接體 actor 的權重讀出，每個 actor 的中位數都還是 50 ms，最長的是 0.06–0.52 s；BC 的序列是 32 步，也就是 1.6 秒。協定的遮蔽是 30 秒，遮蔽期間的表現要在這個限制下解讀（[related_work.md](related_work.md) 第 3 節）。
21. **子圖規模**：正式 no-KC 圖有 4,502 個節點、31,682 條邊。萃取時把門檻放寬到 50 突觸，保留 LC9／LC11 和更多中央嗅覺通路，同時移除原本的 3000 節點上限（`max_neurons: null`，#55），所以超出原計畫的 500–3000 節點預算。
22. **G3 的延遲跟主機有關**：同一份程式在 G1 範圍，2060 的 sensor→command p95 是 55–130 ms，4070 是 48–53.5 ms。G3 的延遲用事先固定的參考主機 4070 判定（#88 Decision 6743008783），其他主機的數字照實報告。
23. **Gazebo 的渲染改成符合 2D**（G0c，PR #102）：舊世界的視覺輸入和 2D 明暗相反，G0c 改用平光材質後才和 2D 一致。G1 範圍的行為（全部是 0）在修改前已經看過，但修改依據的是輸入分布的量測，驗收條件在看行為之前就固定了。

## 下一步

依 [#28](https://github.com/Nues0913/Drosophila_Tracking_System/issues/28) 的順序：

1. ~~#77、#20、#10、凍結與派工~~：已完成。main `35432c3` 上的 30 條訓練鏈已交付並驗收（#80、#81）。
2. ~~2D 正式評估（F3 curved、F4 wander）~~：已完成並驗收（#80、#84），結果見上方 2026-10-31 一節。
   - ~~**#85**~~：GRU 修正（PR #86）和事後重新訓練與評估（#87）都已完成，結果標為「未登記」，見上方 2026-11-04 一節。
3. ~~**協定必交的分析**（A1，#90）~~：已完成並驗收，結果見上方 2026-11-08 一節。之後並行的兩個探索實驗：BC 配方實驗 R1（[#4](https://github.com/Nues0913/Drosophila_Tracking_System/issues/4)，@MiLong9487，判讀第 3 點已在看到結果前修正為相對基準的配對差）和 BC 預算實驗 B1（[#92](https://github.com/Nues0913/Drosophila_Tracking_System/issues/92)，使用者）。A1 發現速度輸出是常數，所以 B1 先暫緩，後來關閉。改做的速度讀出校準 S1（[#95](https://github.com/Nues0913/Drosophila_Tracking_System/issues/95)）判定無效，見上方 S1 一節。R1 也判定無效，BC 配方路線到此為止；重訓變異 R0 顯示基準穩定，見上方 R1／R0 一節。D2 已結束，formal-v2 不觸發（[#28 Decision 6718154301](https://github.com/Nues0913/Drosophila_Tracking_System/issues/28#issuecomment-6718154301)）。#4 已結案；P3、PPO 和文獻上的候選解法都放在 #35，由使用者決定，不進正式矩陣。
4. **Gazebo 主要比較（[#88](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88)）**：G0 的 wander 接線已合併（PR #94，main `c339c0c`，PM 補跑 colcon test 通過）。G1 在 2060 上發現的兩個問題已由 G0b 修正（PR #99，main `4a537af`）；2060 和 4070 的 G1 範圍都通過。完整矩陣 G2（curved、wander 各 2,000 回合，加上事後 GRU 各 400 回合）派工後暫停：2D 與 Gazebo 的一致性（原 #23）量測發現，Gazebo 的視覺輸入和 2D 明暗相反。修正 G0c 在 PR #102，合併後兩台主機重跑 G1 範圍，再重新發 G2；彙總用 PR #101 的設定。見上方 2026-11-09 一節。
5. 並行：#49 的一次短回放已由 #51 報告驗收結案；#52 的 trace 狀態與 reset 清除 pending 可觀測性已由 #53 合併，PM colcon 10 項全過、無 skip；既有發布／reset 行為不變。下一個效能候選是 camera hex-eye 投影的等價優化設計，尚未核准實作或追加回放；2D／Gazebo 一致性（原 #23）已併入 #88，仍由 @cweiting731 處理。#42 已由 #44 結案。
6. Gazebo 主要比較（#88）→ 報告與展示。#22 已結案，延遲量測由 #88 接手。2D 的展示素材由 V1（#97）製作，工具已合併（PR #100）。Gazebo 的展示和 rosbag 排在 G2 之後。

舊版的逐日診斷紀錄保留在 [dn/](dn/)、[fixes_2026-10-17.md](fixes_2026-10-17.md) 和 [training_time.md](training_time.md)，這些是歷史紀錄，不代表目前狀態。
