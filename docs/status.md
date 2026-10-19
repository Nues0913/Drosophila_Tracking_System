# 目前進度與已知問題

最後更新：2026-10-19（已同步 `main` 388d481）。待辦與分工以 GitHub Issue 為準，總追蹤是 [#28](https://github.com/Nues0913/Drosophila_Tracking_System/issues/28)。協作規則見 [AGENT_TEAM_WORKFLOW.md](../AGENT_TEAM_WORKFLOW.md)，逐項檢查清單見 [後續計畫.md](../後續計畫.md)，驗證證據見 [execution_log.md](execution_log.md)。

## 一句話

資料 → 前端 → 子圖 → 模型 → 訓練鏈都已實際跑通，兩台 GPU 主機也已校準完成，結果可以互換。**正式訓練（6 變體 × 5 種子）還沒開始**，開跑前還有三個前提沒解決：

1. 策略還不會用氣味（[#4](https://github.com/Nues0913/Drosophila_Tracking_System/issues/4)）。
2. 目前實驗中 PPO 未改善 BC；觀察到 KL 偏小，但這只是診斷線索，尚不足以確立未改善的原因，仍在持續診斷（[#35](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35)）。
3. 評估協定還有兩項沒凍結：noise 強度和 occlusion 長度（[#20](https://github.com/Nues0913/Drosophila_Tracking_System/issues/20)）。

ROS／Gazebo 部署鏈路已在 Gazebo 實跑通過（[#22](https://github.com/Nues0913/Drosophila_Tracking_System/issues/22)，2026-10-19，CPU／GPU 各 5 回合 invalid 0）；完整端到端延遲 p95 分別為 55.62／56.46 ms，均未達 50 ms。

## 已完成並有證據

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
- 粗估：2M 步的正式 PPO 矩陣由兩台分攤，每台約 1.5～2 天。是否縮短見 [#10](https://github.com/Nues0913/Drosophila_Tracking_System/issues/10)，要等 #35。

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
| **PPO 學會撞牆**：每步 reward ≤ 0，撞牆只扣 10 | 有界的存活獎勵、reward 正規化、critic 不做 value clipping、actor 和 critic 分開裁剪梯度；從 pretrain 開始的 PPO 碰撞率從 1.00 降到 0.07 | #19、PR #32（等證據交付後合併） |
| 遮蔽時 teacher 用學生看不到的真值轉向，長遮蔽 BC 崩潰 | 新增可選的 `occlusion_odor_steering`：遮蔽期間用學生同一份氣味輸入轉向（預設關閉） | #24、#33 |

## 未解決的問題

### 正式訓練的前提（P0）

1. **策略還不會用氣味（#4）。** G2 之後，氣味的影響在 pretrain 後保住了，但 BC 只用 normal 情境，只靠視覺就能完成。BC 把視覺的影響拉高約 200 倍，氣味／視覺的 |Δω| 比值掉到 0.005 以下，odor lesion 幾乎沒有影響。**所以現有的 VO 結果實際上只用了視覺，VO／V／O 的比較和 lesion 結論都還不成立。** 下一步是方向 3：BC 世界混合長遮蔽，遮蔽時啟用 #33 的氣味 teacher，等 @MiLong9487 貼出設計後由 PM 核准。
2. **目前實驗中 PPO 未改善 BC（#35）。** 觀察到 KL 偏小，但這只是診斷線索，尚不足以確立未改善的原因，仍在持續診斷。reward 修好之後，T3 重跑（V、seed 0、各 500k）的 `approx_kl` 中位數只有 1e-4～2e-5。從 BC 開始的 PPO，碰撞率 0.60（BC 是 0.27），成功率 0.03（BC 是 0.20）；從 pretrain 開始的那組不撞牆，但方位誤差約 90°。正在做 CPU 診斷：各參數組的更新量、critic 暖身、learning rate。
3. **評估協定還沒完全凍結（#20）。** 2026-10-19 已凍結：成功判定（≤ 2 m、方位 ≤ 45°、≥ 80% 時間）、環境審計（teacher ≥ 90%、盲目策略 ≤ 10%）、BC gate、運動 DN 輸入增益 100、主要終點與非劣容許差，以及 wander 列為次要情境（主要終點維持 curved）。**還沒凍結：noise 強度（等 #9）、occlusion 長度（等 #4 方向 3）**。兩項定案後，會一起更新 `configs/evaluation_protocol.yaml`。

### 其他未解決

4. **noise 情境對雜訊參數非常敏感**（bc_B 在兩組參數下 77% 對 0%）。雜訊曲線和 BC 雜訊擴增見 [#9](https://github.com/Nues0913/Drosophila_Tracking_System/issues/9)。
5. **情境強度**：occlusion 只遮 2 秒（佔回合 3%），太弱，測不出氣味的補位作用。conflict 目前等於 normal（第 1 點）。
6. **模態使用 gate、BC 崩潰檢查**還沒加進 `run_experiments.py`（[#25](https://github.com/Nues0913/Drosophila_Tracking_System/issues/25)），門檻要等 #4 和 #35。
7. **`lesion_analysis.py` 用了訓練範圍的種子，並繞過情境設定檔**（[#29](https://github.com/Nues0913/Drosophila_Tracking_System/issues/29)）。正式結果前要修。
8. **linear 目標撞牆反彈會撞上跟隨者**（teacher 13% 碰撞），所以只留在 PPO 暖身課程。
9. **ROS／Gazebo**（#22，2026-10-19，沿用舊的 smoke 模型、軟體渲染，只驗鏈路、不代表追蹤品質）：CPU 首輪單回合與 5 回合有效；GPU 追加 5 回合也 invalid 0/5、truth coverage 100%，沒有沿用前一回合的 sample；策略沒有訂閱真值 topic，bag 可以讀回。**CPU／GPU 完整 sensor-to-command p95 為 55.62／56.46 ms，均未達 50 ms**；active 窗口為 54.13／55.09 ms，inference wall p95 為 24.76／24.19 ms。GPU 報告已合併（PR #45），PM 獨立解碼原始 bag 重算相符。兩者各只有一輪，不能推斷 GPU 穩定較慢或唯一瓶頸。依 #20 決策保留門檻與頻率；[#48 的離線診斷](ros_latency_diagnostic_2026-10-19.md) 已核對 4,707 筆同一 sample 的逐段 wall time，慢尾主要落在 forward→received 與 control→published，但尚未定位內部原因。預設關閉的 profiler 探針已由 #50 合併；#49 已從固定 main 1563a93 完成一次有上限的短回放，[#51 報告](ros_policy_replay_profile_2026-10-19.md) 已合併並驗收結案。單次 off/on 分布不代表普遍 probe 開銷，reset 前 trace 缺口的可觀測性已由 #53 的 schema2 狀態與丟棄紀錄補強，#52 結案，歷史缺口保留；simulation-clock sensor age 不與 wall time 相加。#42 測試收集已由 #44 修正；取消 future 的誤報已由 #47 修正，#46 結案。2D／Gazebo 感測分布與碰撞判斷一致性仍未量（[#23](https://github.com/Nues0913/Drosophila_Tracking_System/issues/23)）。ROS 的 `target_motion` 只接受單一字串，正式評估每個 cell 固定一種運動（#20）。
10. **生物對應查證**（ORN、LC、DN 標籤、號誌信心）還沒做（[#27](https://github.com/Nues0913/Drosophila_Tracking_System/issues/27)）。如果需要改圖，要在正式矩陣前完成。

## 報告須揭露的科學限制與工程近似

1. **匯流神經元只有 4 個。** 依計劃書 G1，以 VO-late 為後備主要變體並明示，不能把晚期融合說成找到生物匯流。嗅覺到運動 DN 要 5–7 跳，視覺只要 2–3 跳，訊號強度天生偏向視覺。
2. **LC9、LC11 各只保留 1 個神經元**（原有 219、143 個），視覺輸入主要來自 LC10a（208）、LC10d（143）、LPLC2（112）。
3. **ORN_VA2 有 9 個（約 11%）側別未知，已丟棄**；見 `data/subgraphs/main/excluded_odor_seed_sides.csv`。
4. **運動 DN 讀出**：DNp09 即 P9，雙側活化驅動前進（Bidaye et al. 2020, *Neuron* 108:469–485）；DNa02 左右差與轉向速度呈線性，DNa01 追蹤緩慢持續的轉向（Rayshubskiy et al., *eLife* 102230）。模型以 DNp09 左右平均讀前進、DNa01／DNa02 的「左 − 右」讀轉向。DNp09 本身也帶轉向成分，模型沒有使用。
5. **輸入增益**：運動 DN 的輸入增益 100、ORN 輸入增益 100 都是工程選擇，用來補償圖的跳數和尺度差，不是生物量測值。
6. **bias 的訓練範圍**：核心 bias 有 781 個參數（佔可訓練參數 11.5%）。只在 pretrain 和 BC 凍結，PPO／ES 階段可訓練，參數預算才能和 GRU 公平比較（#4 Decision）。
7. **前端來源**：方案 A 的早期視覺來自 flyvis 的資料與學習參數，只有中央子圖使用 MaleCNS。
8. **輸入映射**：沒有真實 RF 座標，LC 輸入採型別／側別的空間池化；ORN Hill 閾值是模擬設定，不是特定氣味的實測受體曲線。
9. **變體口徑**：V／O 是同一張圖的輸入消融。VO-late 共享型別參數但隔離活性；RS 保留原邊參數槽的共享標籤。這些都寫進 model_card。
10. **2D 世界**：2.5D 光線投射，Gymnasium 向量環境是 CPU 上的平行子行程。2D 有大小／亮度的領域隨機化，Gazebo 用固定的幾何和材質，兩者的感測分布還沒證明一致（#23）。
11. **ROS 碰撞**用場地、半徑與障礙幾何判斷，沒有接 Gazebo contact sensor。
12. **時鐘與延遲**：行為指標用模擬時鐘，推論耗時用 wall time。sensor-to-command 的因果計時**已有契約與實作**：`SensorTiming.msg`／`ControlTrace.msg` 定義同一台主機上 `CLOCK_MONOTONIC` 的 source → ingress → forward → received → control → published 時間線，`flytrack_eval.latency.CausalLatency` 檢查時鐘域、因果順序和覆蓋率，有 `test_latency.py` 測試。2026-10-19 的 CPU／GPU 實測均未達 50 ms，詳見第 9 點與 [ros_execution.md](ros_execution.md)。端到端 p95 必須由各筆命令最舊的有效 contributing sample 至命令發布的完整差值計算；各階段 p95、不同 sensor 的階段最大值不可拼成一條因果路徑。
13. **續訓**：同一階段中斷後可以完整接續（權重、optimizer、RNG、環境重播）；跨階段以 `init_artifact` 暖啟動，recovery 綁定上游 actor 的內容（#14）。改了程式碼或設定之後，舊的 checkpoint 不能接續；#36 合併後，舊的 pretrain／BC artifact 都要重跑。

## 下一步

依 [#28](https://github.com/Nues0913/Drosophila_Tracking_System/issues/28) 的順序：

1. #4 方向 3（@MiLong9487 設計，PM 核准後在兩台主機各跑一個 seed）。
2. #35 PPO 診斷與一次 T3 重跑（@MiLong9487）。
3. #20 凍結剩下的 noise 強度和 occlusion 長度，更新評估協定（PM）；#25 加入新 gate（@MiLong9487）。
4. 全部完成後，PM 凍結一個 `main` commit，依 #21 的分配開執行 Issue：4060 跑 seed 0、2、4，4070 跑 seed 1、3。
5. 並行：#49 的一次短回放已由 #51 報告驗收結案；#52 的 trace 狀態與 reset 清除 pending 可觀測性已由 #53 合併，PM colcon 10 項全過、無 skip；既有發布／reset 行為不變。下一個效能候選是 camera hex-eye 投影的等價優化設計，尚未核准實作或追加回放；#23 2D／Gazebo 一致性仍由 @cweiting731 處理。#42 已由 #44 結案。
6. 2D 正式評估 → Gazebo 主要比較 → lesion 分析 → 報告與展示。

舊版的逐日診斷紀錄保留在 [dn/](dn/)、[fixes_2026-10-17.md](fixes_2026-10-17.md) 和 [training_time.md](training_time.md)，這些是歷史紀錄，不代表目前狀態。
