# 原計畫與後續計畫驗收對照

最後更新：2026-11-18。

依據工作區兩份 PDF、`後續計畫.md` 及使用者 2026-10-15 的完整執行指示。此表保留完整研究範圍；沒有實際證據的項目均為未達成。G0–G3 依原始計畫的定義驗收，不能以局部檢查替代。

| 要求 | 足以證明完成的證據 | 當前狀態 |
| --- | --- | --- |
| G0：ML 與 ROS 容器都可啟動 | 兩映像 digest、啟動紀錄、依賴鎖定、ML 容器真實 CUDA 運算 | 部分：ML 依賴已鎖定（`requirements/ml-cu126-py312.lock`），正式訓練在 4060／4070 的 ML 容器以 CUDA 執行（#80、#81）；ROS 映像從 main `4a537af` 建置（映像 ID：2060 `7e744c4e8989`、4070 `f61921e3c157`），colcon 23 個測試通過（PR #99）。ROS 端沒有依賴鎖定清單；映像只在本機，沒有 registry digest |
| 原始資料與授權來源 | 官方 URL、版本、檔案大小、雜湊與取得紀錄 | 三份 MaleCNS 表格已取得；見 `docs/data_sources.md` |
| G0：子圖重跑位元相同 | 兩次獨立萃取的 nodes/edges/NPZ/manifest 雜湊逐一相同 | **達成**：10 個產物位元相同 |
| G1：真實生物子圖 | 實際 schema/型別/側別/NT 語意，500–3000 節點預算、弱連通、每種子可達 DN、轉向與前進群 | 部分：舊圖 2569 節點、14165 邊，`scripts/verify_subgraph.py` 19 項檢查通過。正式 no-KC 圖是 4,502 節點、31,682 邊，萃取時明確移除 3000 上限（`max_neurons: null`，[#55 報告](olfactory_reextraction_55.md)），所以超出原計畫的 500–3000 預算（[status.md](status.md) 揭露第 21 點） |
| 匯流定義及備案 | h=2–3 視嗅下游交集、數量與來源；不足時明列 VO-late 主變體 | 部分：舊圖的三跳匯流只有 4 個；正式 no-KC 圖有 59 個三跳匯流，另有 483 個同時屬於視覺和嗅覺群的神經元（[#55 報告](olfactory_reextraction_55.md)）。主架構登記為 VO-late（`configs/evaluation_protocol.yaml` 的 `reporting_scope`），報告不能把晚期融合說成生物匯流（[status.md](status.md) 揭露第 1 點） |
| 方案 A 官方前端 | 公開模型來源、作者校驗值、轉換 manifest、灰階/閃光/移動相同刺激誤差通過 | **達成**：最大誤差 1.91e-6（`docs/frontend_execution.md`） |
| 固定遮罩與 Dale 號誌、型別共享 | 訓練前後不新增邊、不改號誌、tau 下限、dense/CSR 梯度檢查 | **達成**（單元測試） |
| G1：六變體參數預算 | 真實圖/前端下六變體的 BC optimizer lr>0 且 requires_grad actor 參數（含 B3）在有效 VO ±5%；RS 度數/密度/號誌比例保持 | **達成（#77 CPU 計數）**：effective 14,093–14,904，參考 VO 14,546；declared／zero_lr／effective 分欄見 [status.md](status.md)。舊 6766–6831（參考 6794）是歷史圖的宣告值 |
| 感測與神經時間契約 | 六角取樣/FOV/感測位置/控制與神經步長/雜湊一致；2D 與 ROS 真實輸入比對 | 未達成 |
| 真值隔離與看門狗 | actor 輸入資料測試、策略實際訂閱清單、亂序/未來/逾時/重置與零速驗證 | 部分：策略訂閱清單的 launch test 通過、關閉時送零速度（#15）；Gazebo 5 回合跨節點 reset 通過、invalid 0（#22）；亂序／未來封包與逾時在 Gazebo 中還沒有專門測試 |
| 教師及學習實作 | 教師追蹤與安全性、終止/截斷 bootstrap、BPTT 狀態/梯度、BC/DAgger/PPO/ES 測試 | 部分：教師在隨機化世界 20/20 成功、方位誤差 2.22°；環境退化審計（盲目策略 0%）；GAE、BPTT、BC/PPO 單元測試通過；小規模端到端鏈見 [fixes_2026-10-17.md](fixes_2026-10-17.md)。PPO 沒有改善 BC，已移出正式鏈，正式鏈是 pretrain → BC/DAgger（#35 選項 D，Decision 6382764327）；PPO 留作研究項目。ES 未實跑 |
| G2：感官方位解碼 | 教師與獨立驗證資料校準後固定門檻；視/嗅真實解碼結果及誤差 | **達成**：正式 pretrain 的每條鏈都跑了 validate-pretrain，門檻固定（視覺 15°、嗅覺 30°）。視覺 25 個 actor 的方位 MAE 0.78–6.98°，嗅覺 25 個 1.99–4.28°，presence 正確率 ≥ 0.9915，全部通過（release `formal-v1` 的 F1／F2 包，`runs/pretrain/*/seed_*/{vision,odor}/validation/metrics.json`；[status.md](status.md) 2026-11-18 一節）。教師校準已在隨機化世界重做 |
| 小規模真實流程 | `smoke.yaml` 梯度更新、可重新載入 artifact、記憶體/吞吐量與真值隔離 | **達成**：縮小版全鏈跑通；兩台 GPU 主機的吞吐量、顯存與跨主機一致性已量（#21） |
| ROS／Gazebo 完整整合 | colcon/build/tests、SDF/bridge/QoS、clock、感測上線後 1 秒內 cmd_vel、兩回合 reset、rosbag 可重讀 | 部分：colcon build 與 23 個測試通過（PR #99）；`/clock`、感測上線後 1 秒內出現 cmd_vel、跨節點 reset、bag 可讀回（#22）；G1 範圍的 valid、頻率與計時在兩台主機都通過（[ros_execution.md](ros_execution.md) 2026-11-09 一節；G0c 之前的世界，行為因明暗相反沒有意義）；G0c 之後 2060 重跑，行為和 2D 一致。2D／Gazebo 輸入一致性（原 #23）已量測：修正前視覺相關 −0.97，G0c（PR #102，已合併）之後 0.97–0.98，氣味 0.99999。contact sensor 的碰撞一致率已核對（G0f，PR #120）：撞牆只差取樣延遲；撞目標的旗標延遲由 G0g 修正（PR #122），修正後旗標都在 contact ±0.022 s 內（[status.md](status.md) 揭露第 11 點）。SDF／bridge／QoS 的逐項核對還沒做 |
| G3：行為與延遲 | 正常 VO 成功率預定門檻、Gazebo 完整回合、因果 sensor-to-command wall-time p95<50 ms | 未達成：2D 的 normal 成功率最高是 0.484（curved，VO-late），低於門檻 0.8。Gazebo 的 G2（4070、4060，固定 `a58b7b2`）也未達成：curved normal 最高是 V 的 0.450；延遲照事先固定的規則，用參考主機 4070 的 curved normal 合併後的 p95（[#88 Decision 6743008783](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6743008783)），V 49.3、VO 49.0、VO-late 51.5、GRU 47.1、RS 50.5、事後 GRU 47.5 ms。延遲跟主機有關（揭露第 17、22 點）。見 [formal_v1_report.md](formal_v1_report.md) 5.3。formal-v1 之後：#110 在 4070 重測（main `e5a166b`，三項等價加速之後），CPU 版 p95 43.4–44.9 ms 低於 50 ms，GPU 版 52.3–53.1 ms；成功率沒有改善，判定維持未達成 |
| 共同正式訓練預算 | 六變體×五訓練種子，單模態預訓練、BC+DAgger、共同 PPO（或全部 ES）環境步數帳本與完整模型產物 | **達成**：六變體 × 五種子，pretrain → BC/DAgger，每條鏈 390,000 步；30 個 actor 和步數帳本在 release `formal-v1`（#80、#81，Decision 6489237904）。PPO 為 0 步（#35 選項 D） |
| 預定主要統計比較 | 正式評估前鎖定主要終點、非劣界值、VO–V 遮蔽、VO–GRU、VO–RS | **達成**：協定 schema 3 由 PR #78 凍結，正式矩陣固定在 main `35432c3`（#28 Decision 6471391440）；VO-late 的晚期融合比較也已登記 |
| 2D 正式評估 | 六變體×四情境×五模型種子×100 回合，配對環境種子、完整失敗/截尾/碰撞紀錄 | **達成**：F3 curved（#80）、F4 wander（#84），已登記比較依 PR #83 重新彙總（Decision 6489649944）。GRU 對照組退化，修正後的事後結果未登記（#85、#87） |
| Gazebo 正式評估 | 三組主要比較，每組五模型種子×20 回合，直接移轉優先，追加微調另報 | 已完成：G2（[#88 6872441153](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6872441153)），直接移轉、不微調，curved、wander 各五變體 × 四情境 × 五個 seed × 20 回合；已登記比較的判讀和 2D 完全相同，只用 4070 的 seed、漏記撞牆的保守重算都沒有改變判讀（[formal_v1_report.md](formal_v1_report.md) 5.2）。經過：G2 派工（#88 Decision 6743008783），彙總設定在 PR #101。之後發現 Gazebo 的視覺輸入和 2D 明暗相反，G2 暫停（Decision 6744333249），修正 G0c 已合併（PR #102）；2060、4070、4060 用 G0c 世界的 G1 範圍，行為都和 2D 一致，G2 已在 4070、4060 上開跑（11-09、11-10）。之後量到斜角撞牆記不到碰撞、轉向比指令快 12%：G2 照跑，處理方式在資料出來之前登記（#88 6763512170、6764426046），修正 G0d 已在 G2 結束後合併（PR #107） |
| 匯流與 lesion 分析 | 視/嗅切斷測試、來源電流描述性分析、種子層級 bootstrap 95% CI | **達成（2D）**：A1（#90，Decision 6711664543）做了視／嗅 lesion、分段來源電流、失敗案例，種子層級 bootstrap 95% CI；屬於描述性診斷 |
| 展示與可重現交付 | 軌跡/活性/動作圖與展示、rosbag、來源/參數/鎖版/操作文件，另一環境可重建 | 部分：2D 展示已交付並驗收（V1，#97，release `formal-v1` 的 `formal_v1_demo_2d.tar.gz`）。Gazebo 有兩份展示：G0d 世界的 11 回合重播和 rosbag（release `formal-v1` 的 `gazebo_demo_g0d_2060.tar.gz`）；3D 影片 V2（#123，PR #124），用跟拍和全場相機重播同樣的 11 回合，在 release `research-35` 的 `gazebo_v2_demo_4070*`，含 88 秒的精華片段。各 release 的包都附 `SHA256SUMS`。ROS 端沒有依賴鎖定清單（見 G0 一列），「另一環境可重建」沒有獨立驗證 |

溫度感覺、方案 B、精準 RF、CMA-ES、實體機器人與 Gazebo 線上 RL 依原後續計畫不列入主時程。研究假設不成立可如實交付結果；不可省略正式實驗或以不顯著差異當作非劣證明。
