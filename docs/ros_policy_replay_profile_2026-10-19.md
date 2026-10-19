# #49 ROS policy 短回放 profiler 結果

日期：2026-10-19。固定 main `1563a936bc75f4db2e82e1c9c9df3fd6b0929a13`，在同一台 RTX 4070 Laptop 上，以同一 policy artifact、sensor contract 和 #22 五回合 input bag，分別跑 CPU 與 CUDA；每種 device 各做一次 profiler 關閉／開啟配對。沒有啟動 Gazebo、訓練或正式評估。完整命令、原始 monitor／profiler JSONL 與 policy log 已上傳至 [explore-4 原始證據包](https://github.com/Nues0913/Drosophila_Tracking_System/releases/download/explore-4/issue49_replay_evidence_2026-10-19.tar.gz)；SHA-256：`7c58f954575cbbfb125a2e477f67d3ba03b87b9593ceb2a20a5a62a694277cef`（[校驗檔](https://github.com/Nues0913/Drosophila_Tracking_System/releases/download/explore-4/issue49_replay_evidence_2026-10-19.tar.gz.sha256)）。原始包保持 byte-for-byte 不變。

## 執行與覆蓋

| Device | profile off/on case elapsed | 完整 bag replay | bag sensor publish（left/right/odor） | reset ack | 新鮮輸入後首次有效推論 |
| --- | ---: | ---: | ---: | ---: | --- |
| CPU | 186.669／186.720 s | 171.762 s／case | 3554／3552／5861（兩次相同） | 每次 6/6 成功 | 5 個 active episode 均在三路新樣本到齊後開始；從 reset 到第一個全 valid 推論為 692–699 ms（bag sim time） |
| CUDA | 186.622／186.356 s | 171.762 s／case | 3554／3552／5861（兩次相同） | 每次 6/6 成功 | 5 個 active episode 均在三路新樣本到齊後開始；從 reset 到第一個全 valid 推論為 692–698 ms（bag sim time） |

每組另有 10 秒暖機，之後重播完整 bag；profile off／on 使用相同排程和 episode reset。每種 device 配對約 379 秒，低於各 600 秒上限；總正式配對約 12 分 38 秒，連同 CPU 預檢仍低於 20 分鐘總上限。控制頻率、QoS、executor、sensor contract、模型和 reset 語意均未更改。原 bag 的 monotonic 值沒有與本次時間相減。

CPU profile-on 的 measured command callback 有 3435 筆 timing row；CUDA 有 3423 筆。profile-on/off case elapsed 差異分別為 +0.050 秒及 −0.266 秒（絕對值均低於 0.15%）；單次配對只能說本次沒有觀察到明顯回放耗時增加，不能估計普遍 overhead。兩份 profile-on sidecar 約 10.8 MB。

## 時間分解

下表為 profile-on、command callback 進入 measurement 區間的 p50／p95／p99，單位 ms；分位數使用線性插值，位置為 `(n−1)×p`。Command callback 與 publish spans 的 n=CPU/CUDA 3435/3423；inference-only tick spans 的 n=2364/2352。tick 各段依本次 `CLOCK_MONOTONIC` 時間戳相減；不包含 ROS/Gazebo closed-loop、訊息網路或完整 sensor-to-command 端到端延遲。

| Span | CPU p50／p95／p99 | CUDA p50／p95／p99 |
| --- | ---: | ---: |
| 最近一個 sensor callback 結束至 control callback 開始（只含有 callback 的 command） | 1.700／11.969／12.879 | 1.106／11.904／13.161 |
| control callback entry → exit（所有 command） | 13.452／19.991／24.060 | 18.128／22.570／25.891 |
| entry → command publish start（所有 command） | 12.956／19.383／23.532 | 17.637／21.979／25.274 |
| command publish call（所有 command） | 0.105／0.140／0.207 | 0.091／0.131／0.182 |
| tick input conversion（inference） | 0.080／0.113／0.149 | 1.111／1.603／3.886 |
| device sync before forward（inference） | — | 0.035／0.051／0.077 |
| model forward and completion（inference） | 13.092／20.487／23.468 | 16.525／20.416／26.105 |
| finite check（inference） | 0.185／0.358／0.470 | 0.480／0.838／1.287 |
| device-to-host（inference） | 0.024／0.036／0.051 | 0.138／0.255／0.350 |
| action limiting（inference） | 0.060／0.080／0.110 | 0.052／0.073／0.103 |

本輪量測中，最近 sensor callback 結束至 control callback 開始的 p95 約 12 ms。原先容易把不同 cohort 混讀：全 command callback p95 是 CPU/CUDA 19.991/22.570 ms（n=3435/3423），model-forward p95 僅計實際 inference，是 20.487/20.416 ms（n=2364/2352）；inference-only 整個 callback p95 為 21.582/23.529 ms。CUDA model-forward p99 是 26.105 ms，並非 p95。CUDA input conversion、finite check、device-to-host 也比 CPU 高。這些是該 smoke artifact、該 bag 與單次重播的觀察值，不能外推為穩定硬體基準，也不能直接解釋舊 Gazebo p95。

## 離線補分析：同 cohort 配對、感測 callback 等待與處理

以下由原始 monitor／profiler JSONL 重算，沒有修改或重跑原始資料。量測區間以 monitor 的 `phase_start/phase_end(measurement)` 定義；同一 run 的 `CLOCK_MONOTONIC` 時間戳相減。分位數使用線性插值，位置為 `(n−1)×p`。Profile-on cohort 要求 command callback entry 在 measurement 內、三個 `valid` 都為 true，且 monitor 確實收到同 command sequence 的 `ControlTrace`；profile-off 使用 measurement 內 monitor 收到且三路皆 valid 的 ControlTrace。Profile-off start→publish 使用 `control_started_monotonic_ns` 至 `command_published_monotonic_ns`；profile-on 使用 profiler callback entry 至 publish done。這兩個量都表示策略 callback 開始到 command publish 完成，並非固定排程 replay 的 case elapsed。

| Device / profiler | Valid trace cohort n | Start → command publish p50 / p95 / p99 (ms) |
|---|---:|---:|
| CPU off | 2289 | 13.690 / 21.275 / 24.520 |
| CPU on | 2289 | 13.660 / 21.110 / 23.867 |
| CUDA off | 2270 | 18.089 / 22.186 / 33.884 |
| CUDA on | 2278 | 18.631 / 23.133 / 29.155 |

這是每種 device 各一次的配對觀察，不能據此推論 profiler 的普遍或因果 overhead。Profile-on sidecar 寫檔成本也沒有被這個 span 隔離；case elapsed 主要受固定 bag 發布時序支配，不能單獨估 probe overhead。總 measurement command callback 數為 CPU/CUDA 3435/3423，實際 inference 數為 2364/2352，收到 ControlTrace 數為 3361/3350；因此不能把完整 command 集或 callback 與 model span 當成同一個 cohort。舊報告的「最近 callback 結束→control」只涵蓋有一筆以上 accepted sensor callback since previous command 的 2350/2322 筆 command，不代表每筆 command 的等待時間。

Sensor publish 記錄依 sensor 與 `header_stamp` 配對到 profiler 收集的 accepted sensor callback；同一 identity 有多筆發布時，採 callback entry 前最近的一筆。表中保留發布、accepted callback 與配對數，能看出 CUDA best-effort 輸入有掉落；publish→callback entry 同時含 DDS 傳遞與 executor 排程，不是單獨的 executor 等待時間。

| Device | Sensor | Published | Accepted callback / matched | Publish → callback entry p50 / p95 / p99 (ms) | Decode p95 | Hex-eye sample p95 | Buffer update p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| CPU | left | 3554 | 3554 / 3554 | 0.293 / 11.706 / 15.324 | 0.057 | 5.207 | 0.046 |
| CPU | right | 3552 | 3552 / 3552 | 0.997 / 13.921 / 16.805 | 0.050 | 4.912 | 0.045 |
| CPU | odor | 5861 | 5861 / 5861 | 2.671 / 15.744 / 20.769 | — | — | 0.058 |
| CUDA | left | 3554 | 3499 / 3499 | 0.273 / 21.099 / 40.104 | 0.052 | 4.904 | 0.043 |
| CUDA | right | 3552 | 3486 / 3486 | 1.320 / 22.164 / 41.280 | 0.051 | 4.728 | 0.041 |
| CUDA | odor | 5861 | 5777 / 5777 | 4.240 / 38.027 / 74.728 | — | — | 0.045 |

可由 PR 內的 `scripts/diagnostics/analyze_policy_replay.py` 重算上述摘要。解開 release 原始包後執行：`python3 scripts/diagnostics/analyze_policy_replay.py /path/to/package --output analysis.md --json analysis.json`。此 analyzer 僅處理 ROS replay 證據，未改訓練工具或資料 fingerprint；原始包不包含 analyzer，避免改寫其固定 SHA。

CPU profile-on 記錄 3554／3552／5861 個 left／right／odor callback，均 accepted，與對應 sensor publish 數相同。CUDA 記錄 3499／3486／5777 個 accepted callbacks，約比發布數少 1.4–1.9%；ROS best-effort queue 下這次重播有少數 callback 未被策略處理。這不等於 active episode 失敗：每個 active episode 的第一個有效推論都等到三路新樣本。

## Cold start 與暖機

每個 process 的 10 秒暖機中，第一筆三路有效推論單獨列出；後續暖機有效推論作為 warm steady。CUDA 第一筆 model-forward span 為 275.06 ms、整個 policy callback 為 279.16 ms；後續暖機 forward p50／p95 為 15.70／18.08 ms，callback 為 18.18／20.44 ms。CPU 第一筆 forward 為 40.70 ms、callback 為 42.97 ms；後續暖機 forward p50／p95 為 12.59／19.76 ms，callback 為 13.60／20.94 ms。這確認本次 CUDA 首次有效推論明顯較慢，但 span 含模型呼叫與同步，沒有把 CUDA context 建立單獨計時，也不能單憑此結果定位唯一原因。正式 measurement 在暖機後進行，故上表沒有把 cold start 混入穩態分位數。

## ControlTrace 限制與後續

回放器只重播 camera、odor、`/clock`，不重播 bag 內舊的 `/fly/sensor_timing`。因此 policy 的 `ControlTracer` 會等不到對應 timing，將有效 trace 留在 pending；reset 會清掉 pending。獨立 monitor 實際收到的 trace 在四個 episode 邊界各少約 18–19 個 sequence：CPU 收到 3611 個 trace，對應 3685 個 profiler command sequence（少 74）；CUDA profile off 收到 3583／3657（少 74），profile on 收到 3594／3667（少 73）。monitor 的 `/cmd_vel` 計數另含每次 reset 發出的停車命令，所以數量較高。缺口均成群出現在 reset 前。profile sidecar 對 command sequence 連續記錄，並將每筆標為 `control_trace_status=emitted`，但本次證據顯示該欄位不能當成 subscriber 已收到 trace 的確認。

因此 callback／tick 時間分解可用 sidecar 的新程序 monotonic 時間與 header identity 覆核；ControlTrace 的獨立送達覆蓋不完整，不能宣稱每個 command 都有外部 trace 確認。若要補齊，需由 PM 核准後續處理舊 `SensorTiming` identity 或調整 profiler 的 pending／status 語意，再依新的明確預算決定是否重播。本次沒有超出一次核准回放的範圍再次執行。

這項診斷**不代表 G3 或完整端到端 50 ms 門檻通過**，也沒有改動 #20 的門檻、頻率或 sensor contract。

## 固定輸入

| Input | SHA-256 |
| --- | --- |
| main commit | `1563a936bc75f4db2e82e1c9c9df3fd6b0929a13` |
| #22 bag DB3 | `1d25294bb78123590167a51fbdebb70a3a600db49f54af0140ce11c820b60c92` |
| bag metadata | `e34ed68595c4021ff078763a34ebd1ab73f07d54a5356793b3252331dce46912` |
| model card | `7ccff60ad0620b4ec5275eba48f1d8a8aece59d6b73a8594d93fdd69360f5fdc` |
| sensor contract | `a78efde40508d298f2f1af444e163f0f9b7e46efb12158a642a159d8934dc5d5` |
| frontend | `47cccf33ad8a5ba708d557e500cf48b7f3946be1b0f9f6eecffec6349955ec69` |
| graph | `a3c70c2759b0c9af179bb40ef0ff6e46ff469c96df0155c8fbe82bc4715f2e71` |
| weights | `8a35664fca681b55022b27d3303ea6aff46eedb71d76eb40247ac7933f58acb3` |

主機為 NVIDIA GeForce RTX 4070 Laptop GPU、8188 MiB、driver 617.42。映像 `flytrack-ros:issue22-gpu-cu126` image ID：`sha256:792f9b3d4f09717020c33268ce379ff29a250cc81906d6d3881e3908a4aca39c`。
