# 目前進度與已知問題

最後更新：2026-10-28。正式規格是 no-KC＋P2/B3、pretrain → BC/DAgger，團隊選 F：保留全部已登記比較，如實報告 VO 的有限氣味反應及負面結果。待辦與分工以 GitHub Issue 為準，總追蹤是 [#28](https://github.com/Nues0913/Drosophila_Tracking_System/issues/28)。協作規則見 [AGENT_TEAM_WORKFLOW.md](../AGENT_TEAM_WORKFLOW.md)，逐項檢查清單見 [後續計畫.md](../後續計畫.md)，驗證證據見 [execution_log.md](execution_log.md)。

## 2026-10-28：試跑已交付，協定記錄完成，正式矩陣仍待有效預算與版本凍結

- 兩個 seed 的 VO A 都已交付。seed 1 的 G4 通過，但 G3 原登記 100 對的 D=−26.09°，氣味移除反而降低方位誤差；A 不能記為複製成立。seed 0 的原始 G3 受轉向偏差補償干擾，任何引用都須並列去偏差結果。
- 依 [#4 選 F 的 Decision](https://github.com/Nues0913/Drosophila_Tracking_System/issues/4#issuecomment-6442266901)，BC 維持 normal-only、不用氣味 teacher、不做 noise 擴增、不做偏差置中；P3 留到正式矩陣之後。
- #9 的雙 seed 選點是 **0.75 倍**；正式遮蔽是第 8 秒起 30 秒。schema 3 記錄情境強度、#25 的 G4／終端 BC 失敗處理、wander 次要比較及報告範圍。所有原 VO／VO-late 對 V／GRU／RS 的已登記比較都保留。
- **新阻礙 #77**：BC-only 鏈全程凍結 1,697 個核心 bias，但舊預算仍將它們計入。固定 `79eff41` 的非零學習率計數為 VO 14,546、GRU 15,516，差距 6.67%，超過既有 ±5%。模型 DRI 修正有效計數與 GRU 配置後，才能凍結正式版本。
- #10 外推：按 VO 試跑估計訓練＋主要 2D 評估的串行等效約 4060 23.9 h、4070 12.0 h，**NOT VERIFIED**，不當成上限、不把 jobs=2 當兩倍加速。保留 48–72 小時工作目標，不減 seed 或步數。

原始包、逐回合重算、情境參數、限制與時間口徑見 [協定凍結依據](protocol_freeze_2026-10-28.md)。#77 完成後，仍須由 PM 在 #28 凍結共同 main SHA 並開執行 Issue。

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

資料 → 前端 → 子圖 → 模型 → 訓練鏈都已實際跑通，兩台 GPU 主機也已校準完成。**正式訓練（6 變體 × 5 種子，pretrain → BC/DAgger）還沒開始**，開跑前必須完成：

1. [#77](https://github.com/Nues0913/Drosophila_Tracking_System/issues/77) 的有效參數預算修正與必要審查。
2. [#20](https://github.com/Nues0913/Drosophila_Tracking_System/issues/20) 的 schema 3 協定交付，再由 PM 在 [#28](https://github.com/Nues0913/Drosophila_Tracking_System/issues/28) 凍結共同版本、資源與派工。

PPO 已移出正式訓練鏈，改成研究項目（[#35](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35)），不再擋正式訓練。

ROS／Gazebo 部署鏈路已在 Gazebo 實跑通過（[#22](https://github.com/Nues0913/Drosophila_Tracking_System/issues/22)，2026-10-19，CPU／GPU 各 5 回合 invalid 0）；完整端到端延遲 p95 分別為 55.62／56.46 ms，均未達 50 ms。

## 舊圖基線與已驗證鏈路（依各列證據版本）

| 項目 | 結果 | 證據 |
| --- | --- | --- |
| MaleCNS v1.0 原始表格 | 三份官方表格下載，大小、ETag、Arrow 框架、SHA-256 核對 | `docs/data_sources.md` |
| flyvis 前端（方案 A） | 官方 checkpoint 作者 SHA-256 核對；原生轉換對官方輸出最大誤差 1.91e-6 | `artifacts/frontend/verification.json`、`docs/frontend_execution.md` |
| 中央子圖萃取 | 2569 神經元、14165 邊；19 項獨立檢查全過；重跑兩次位元相同 | `artifacts/data_execution/graph_verification.json` |
| 六變體參數預算 | G2 設定下 VO／V／O／RS 6795、VO-late 6767、GRU 6831，都在 ±5% 內 | #36 |
| 測試與 lint | 165 passed（`main` 83f57d1 的既有結果，含真實前端與 CUDA 測試）；`ruff check` 通過 | `pytest python/tests` |
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

1. **#4 按團隊選 F 交付有限範圍的負面結果。** P2+B3 的 O 氣味追蹤已有複製證據；VO 的氣味左右反應很弱，seed 0 原始 G3 受偏差補償干擾，seed 1 G3 反方向。B／C 不成立，A 不一致。正式 BC 保留 normal-only，主架構 VO-late；原登記比較全部報告，不能由 G4 通過直接宣稱融合成功。P3 留到正式矩陣之後。
2. **PPO 沒有改善 BC（#35），已移出正式訓練鏈。** C9–C13 的各種限制（凍結 readout bias、去掉課程、L2 錨定）都沒有在兩個 seed 上通過 protocol100。每次失敗的方式不同（繞圈、追太近而撞上、帶著轉向偏差），參數離 BC 很近也一樣會變差。本機探索（VO、舊圖、只更新讀出層、核心凍結）顯示，在那個設定下更新方向大致正確，崩潰主要來自步幅過衝；但 reward 的改善沒有換成 protocol100 要求的方位和碰撞表現。C13 是全部參數都在更新、離 BC 不到 0.13，兩個 seed 仍然變差，所以這個結論還不能推廣到正式規格。原因還沒拆清楚，留在 #35 研究，不擋正式訓練。
3. **協定交付與正式版本凍結（#20／#28）。** schema 3 記錄原有成功判定、審計、BC gate、DN 增益 100、主要終點與非劣界值，以及新完成的 noise=0.75 倍、30 秒遮蔽、#25 失敗處理與報告範圍。wander 仍是次要情境，主要終點維持 curved。#77 的有效預算修正完成後，才凍結共同 main SHA，不能把協定文件當成開跑授權。

### 其他未解決

4. **noise 情境對雜訊參數非常敏感**（bc_B 在兩組參數下 77% 對 0%）。雜訊曲線和 BC 雜訊擴增見 [#9](https://github.com/Nues0913/Drosophila_Tracking_System/issues/9)。
5. **情境強度與研究範圍**：正式 occlusion 已改成第 8 秒起遮 30 秒，noise 用雙 seed 曲線選出的 0.75 倍。這些設定不能保證所有變體都受益；conflict 與 normal 的差異依實際結果報告。
6. **模態 gate 與 BC 失敗分類已實作**（[#25](https://github.com/Nues0913/Drosophila_Tracking_System/issues/25)，PR #76）。O、VO、VO-late 在 BC 後要求 swap ≥1e−3 rad/s 且 ≥同 chain 起點的 3 倍；V／GRU／RS 模態診斷只記錄。終端 BC 品質失敗照樣評估並逐組揭露，工程失敗強制停止整個矩陣，keep-going 不能略過。schema 3 記錄相同規則。
7. ~~`lesion_analysis.py` 用了訓練範圍的種子，並繞過情境設定檔~~：已由 PR #69 修正（#29）。lesion 現在用 `evaluate.yaml` 的世界、card 的訓練 seed 和正式評估同一個種子公式，`lesion=none` 的回合和 `evaluate()` 逐回合相同。lesion 分析前還要注意兩件事（PR #69 review 的建議）：
   - 如果正式評估改成 `deterministic: true`（目前是 false），lesion 腳本也要呼叫 `enable_deterministic()`，GPU 上才會逐位元配對。
   - 可以把「`lesion=none` 等於 `evaluate()`」寫成回歸測試。
8. **linear 目標撞牆反彈會撞上跟隨者**（teacher 13% 碰撞），所以只留在 PPO 暖身課程（研究用）。
9. **ROS／Gazebo**（#22，2026-10-19，沿用舊的 smoke 模型、軟體渲染，只驗鏈路、不代表追蹤品質）：CPU 首輪單回合與 5 回合有效；GPU 追加 5 回合也 invalid 0/5、truth coverage 100%，沒有沿用前一回合的 sample；策略沒有訂閱真值 topic，bag 可以讀回。**CPU／GPU 完整 sensor-to-command p95 為 55.62／56.46 ms，均未達 50 ms**；active 窗口為 54.13／55.09 ms，inference wall p95 為 24.76／24.19 ms。GPU 報告已合併（PR #45），PM 獨立解碼原始 bag 重算相符。兩者各只有一輪，不能推斷 GPU 穩定較慢或唯一瓶頸。依 #20 決策保留門檻與頻率；[#48 的離線診斷](ros_latency_diagnostic_2026-10-19.md) 已核對 4,707 筆同一 sample 的逐段 wall time，慢尾主要落在 forward→received 與 control→published，但尚未定位內部原因。預設關閉的 profiler 探針已由 #50 合併；#49 已從固定 main 1563a93 完成一次有上限的短回放，[#51 報告](ros_policy_replay_profile_2026-10-19.md) 已合併並驗收結案。單次 off/on 分布不代表普遍 probe 開銷，reset 前 trace 缺口的可觀測性已由 #53 的 schema2 狀態與丟棄紀錄補強，#52 結案，歷史缺口保留；simulation-clock sensor age 不與 wall time 相加。#42 測試收集已由 #44 修正；取消 future 的誤報已由 #47 修正，#46 結案。2D／Gazebo 感測分布與碰撞判斷一致性仍未量（[#23](https://github.com/Nues0913/Drosophila_Tracking_System/issues/23)）。ROS 的 `target_motion` 只接受單一字串，正式評估每個 cell 固定一種運動（#20）。
10. ~~生物對應查證~~：已完成（PR #72，[#27](https://github.com/Nues0913/Drosophila_Tracking_System/issues/27) 已關閉），報告在 [biology_verification.md](biology_verification.md)。沒有需要改圖的發現，也不改規格；要揭露的事項列在下方科學限制的第 4、14、15 點。

## 報告須揭露的科學限制與工程近似

1. **舊圖的三跳匯流只有 4 個。** 這是原本採 VO-late 為主要變體的依據，schema 3 明確標記該數字屬於歷史圖，不能冒稱 no-KC 的盤點。新圖與活動隔離見 #55 報告；主架構及已登記比較仍依選 F 保留，不能把晚期融合說成找到生物匯流。
2. **舊圖的 LC9、LC11 各只保留 1 個神經元**（原有 219、143 個），視覺輸入主要來自 LC10a（208）、LC10d（143）、LPLC2（112）；這些是舊圖的盤點，正式 no-KC 圖依 #55 的產物報表。
3. **ORN_VA2 有 9 個（約 11%）側別未知，已丟棄**；見 `data/subgraphs/main/excluded_odor_seed_sides.csv`。
4. **運動 DN 讀出**：DNp09 即 P9，雙側活化驅動前進（Bidaye et al. 2020, *Neuron* 108:469–485）；DNa02 左右差與轉向速度呈線性，DNa01 追蹤緩慢持續的轉向（Rayshubskiy et al., *eLife* 102230）。模型以 DNp09 左右平均讀前進、DNa01／DNa02 的「左 − 右」讀轉向，B3 另外加入 DNb05 的左右差。DNp09 就是 P9，P9 也會驅動同側轉向（Bidaye et al. 2020），所以「純前進神經元」的說法和文獻不一致；只讀前進是工程選擇，模型沒有用到它的轉向成分（#27）。
5. **輸入增益**：運動 DN 的輸入增益 100、ORN 輸入增益 100 都是工程選擇，用來補償圖的跳數和尺度差，不是生物量測值。
6. **bias 與有效參數預算**：no-KC 有 1,697 個核心 bias，pretrain 與 BC 都凍結。PPO 已移出正式鏈，不能沿用「最後靠 PPO 解凍」的公平性理由。固定 `79eff41` 的有效 VO 為 14,546、GRU 為 15,516（+6.67%）；#77 必須修正有效計數與 GRU 配置，正式比較仍遵守既有 ±5%。
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

## 下一步

依 [#28](https://github.com/Nues0913/Drosophila_Tracking_System/issues/28) 的順序：

1. #77 修正 BC-only 有效參數預算，由模型 DRI 交付、PM Review。
2. #20 交付 schema 3、noise=0.75 倍、30 秒遮蔽、#25 規則與報告範圍；必要審查與當前完成狀態以 GitHub 為準。
3. #10 依已交付試跑的時間外推做資源規劃；不要求為此重跑矩陣。#27 已完成，P3／PPO 另列後續研究。
4. 全部完成後，PM 凍結共同 `main` commit 與協定，依 #21 開執行 Issue：4060 跑 seed 0、2、4，4070 跑 seed 1、3，兩台均 `--jobs 2 --keep-going`，工程失敗停止規則照 #25。
5. 並行：#49 的一次短回放已由 #51 報告驗收結案；#52 的 trace 狀態與 reset 清除 pending 可觀測性已由 #53 合併，PM colcon 10 項全過、無 skip；既有發布／reset 行為不變。下一個效能候選是 camera hex-eye 投影的等價優化設計，尚未核准實作或追加回放；#23 2D／Gazebo 一致性仍由 @cweiting731 處理。#42 已由 #44 結案。
6. 2D 正式評估 → Gazebo 主要比較 → lesion 分析 → 報告與展示。

舊版的逐日診斷紀錄保留在 [dn/](dn/)、[fixes_2026-10-17.md](fixes_2026-10-17.md) 和 [training_time.md](training_time.md)，這些是歷史紀錄，不代表目前狀態。
