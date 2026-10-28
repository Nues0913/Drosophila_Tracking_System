# 正式協定凍結依據（2026-10-28）

本文件記錄 #4 兩個 VO A 試跑、#9 的 noise 選點、#25 的 gate，以及 #10 的時間估計。[#77](https://github.com/Nues0913/Drosophila_Tracking_System/issues/77) 已修正有效參數預算、GRU 配置與評估端重算核對，內容在 PR #79。正式矩陣尚未開始；PR #79 仍須通過 PM review 並合併，再由 PM 在 #28 凍結共同 main SHA、協定版本及派工。

## 來源與 PM 核對範圍

兩台試跑都使用固定 main `f7c589142df4203aee504999a6f2ea26b3933728`，no-KC 圖、VO、normal-only BC。seed 1 只有 A，沒有 B／C／O 或正式矩陣；原 runner 因 BC 的 `omega_r2` 品質失敗退出 1，不代表訓練預算未完成。

| 交付包（explore-4） | SHA-256 | PM 通過的清單項目數 |
| --- | --- | ---: |
| [seed 0 4-R8](https://github.com/Nues0913/Drosophila_Tracking_System/releases/download/explore-4/issue4_r8_MiLong9487.tar.gz) | `58974908574761fe63db1997fc5d30d42f3a957ff098f89521569458fa9f1156` | 206 |
| [seed 0 窗口診斷及 noise](https://github.com/Nues0913/Drosophila_Tracking_System/releases/download/explore-4/issue4_r8w_MiLong9487.tar.gz) | `735eba72ede855f6f419fe03dccef456dccdbb921ac182c648eced6ac6ef7d94` | 40 |
| [seed 0 的 0.75 倍](https://github.com/Nues0913/Drosophila_Tracking_System/releases/download/explore-4/issue4_r8w_s075_MiLong9487.tar.gz) | `e4405b15e71dfd1985273dafb84e084cf5c5afc4661c2a8e8ffdfa25a5b5c2eb` | 5 |
| [seed 1 4-R8](https://github.com/Nues0913/Drosophila_Tracking_System/releases/download/explore-4/issue4_r8_cweiting731.tar.gz) | `8b687cc311c696e2f8947d669cbde15d9e89be8cde556b16df10b9971e4755db` | 34 |
| [seed 1 窗口補件](https://github.com/Nues0913/Drosophila_Tracking_System/releases/download/explore-4/issue4_r8_dn_health_seed1_windows.tar.gz) | `c8e79e32831b44968452d75c97ad8e0af675229498540db91700eeec2c27af1f` | 34 |

清單數不含 `SHA256SUMS` 本身。PM 另核對 seed 1 三份訓練 card 各 51 個 source hash 對應固定 commit、6 個輸入雜湊、完整逐回合聚合、G3 和 G4。匯出的感測契約 JSON 與 repo 設定是同一內容、不同格式，不能直接把兩者的 bytes hash 當成相同。

本輪 PM 只做離線資料核對、統計與靜態 optimizer 分組計數，未重播權重、重訓或啟動正式矩陣。封存包不含權重。

## #4：A 沒有複製氣味效益

判定的 D 是每個配對回合「拿掉氣味的遮蔽期方位誤差 − intact 誤差」。正值才代表氣味有幫助。原登記判定使用全部有效配對；完整窗口配對是另列的描述項，兩者不能替換。

| seed 1 actor／口徑 | 配對數 | D 平均 | 正／負配對 | 雙尾符號檢定 p | bootstrap 95% CI |
| --- | ---: | ---: | --- | ---: | --- |
| 原始 A，原登記口徑 | 100 | −26.09° | 18／82 | 6.15e−11 | [−32.10°, −20.23°] |
| 原始 A，只留完整窗口 | 60 | −36.07° | 10／50 | 1.62e−7 | [−44.34°, −27.80°] |
| 事後置中，只描述，全部配對 | 100 | −24.66° | 15／85 | 4.83e−13 | [−30.15°, −19.33°] |
| 事後置中，只描述，完整窗口 | 60 | −33.15° | 9／51 | 3.09e−8 | [−40.73°, −25.53°] |

CI 是以配對回合為單位的 10,000 次 bootstrap，重算 seed=1301。完整窗口有 599 個 post-step 樣本，另外 40 對因提前終止未滿窗口；沒有重播或補回合。

seed 1 的 G4 通過：BC swap=0.0259075722 rad/s、同 seed odor-pretrain 起點=1.85836639e−5，約 1,394 倍。終端 BC 只有 `omega_r2` 失敗，工程檢查通過，依 #25 繼續評估並揭露。normal 成功率 0.24、碰撞率 0.03；長遮蔽 intact 成功率 0.01、碰撞率 0.33。

seed 0 的原始 A 在先前驗收中通過 G3／G4，但置中後 G3 主要效應消失；seed 1 的 G3 是反方向。**A 不能記為兩個 seed 複製成立**。依使用者團隊選 F（[Decision 6442266901](https://github.com/Nues0913/Drosophila_Tracking_System/issues/4#issuecomment-6442266901)），維持 normal-only BC，保留全部預登記比較，並如實交付有限範圍的負面結果；不追加 P3 或置中訓練。

### 後續窗口補件：保留原 burn-in

[18:15 UTC 補交的診斷](https://github.com/Nues0913/Drosophila_Tracking_System/issues/4#issuecomment-6450055014)已由 PM 離線核對，取代初版「未交付／NOT RUN」狀態。作者在固定 f7c5891、同一個 seed 1 A actor 上用 CPU 跑兩次；每個窗口24個 held-out episodes，每回合200個計分步，環境seed為1,500,100,000–023。

| burn-in | Corr | R² | 平均 ω 偏差（rad/s） | Corr≥0.5 且 R²>0 |
| ---: | ---: | ---: | ---: | --- |
| 16 | 0.521714 | −0.512415 | +0.015684 | 不通過 |
| 200 | 0.763020 | 0.299381 | −0.028893 | 通過 |
| 400 | 0.441661 | 0.014822 | −0.041356 | 不通過 |
| 800 | 0.814674 | 0.518195 | −0.025194 | 通過 |

PM 核對34個 manifest 項目、actor／設定輸入與原R8包一致；本機診斷副本相對原腳本只增加 omega_mean／mean_omega_bias 兩個統計欄位。兩次執行除新增欄位及耗時外的所有輸出完全一致；burn-in 16 的原有指標也與終端 DN 報告一致。沒有重播權重或追加模型診斷。

seed 1 的400步窗口及seed 0的800步窗口都不符合預先條件，所以仍是「其他情況」，正式 BC burn-in 不改。這項補件只補齊描述，不改門檻或配方。#77 的有效預算另已修正，仍待 PR #79 通過 PM review 並合併後凍結正式版本。初始 PM 核對包保留為補件前的快照，窗口補件另外保存。

## #9／#20：noise 用原強度的 0.75 倍

沿用 Decision 6401298099 的選點規則，不增加回合數或改基準。每格 2 個訓練 seed、每 seed 30 回合，表中是成功回合數：

| 倍率 | seed 0／30 | seed 1／30 | 平均成功率 |
| --- | ---: | ---: | ---: |
| 0 | 11 | 5 | 0.2667 |
| 0.25 | 11 | 4 | 0.2500 |
| 0.5 | 10 | 2 | 0.2000 |
| 0.75 | 7 | 1 | 0.1333 |
| 1 | 4 | 0 | 0.0667 |
| 1.5 | 1 | 0 | 0.0167 |
| 2 | 2 | 0 | 0.0333 |

0 倍平均 16/60，一半是 8/60；0.75 倍剛好是 8/60，內插後再四捨五入到 0.05 仍是 **0.75**。兩邊的環境與五項 noise 參數一致。曲線輸出只有每格聚合，沒有逐回合 noise 明細；成功數由 30 × 原始成功率還原，不能宣稱重播過 noise 回合。

正式五項值是 image_noise_std=0.06、brightness_std≈0.15、image_dropout≈0.0375、odor_noise_std=0.0225、odor_dropout≈0.075。YAML 與 ScenarioSpec 的五項預設同步，保留原始乘法的浮點值，使新設定 1 倍與舊曲線 0.75 倍的五欄完全一致。既有 one-source 回歸測試會阻擋只改 YAML、漏改程式預設的情況。

限制：只有兩個訓練 seed，0 倍成功率分別 11/30 和 5/30，seed 差異大；每格平均只基於 60 回合。曲線不是正式評估結果，不保證換 seed 或模型後仍在一半成功率；正式矩陣開始後不再依結果調強度。

遮蔽固定為 60 秒回合中「第 8 秒起遮 30 秒」。不更改成功判定、主要比較、非劣界值、50 ms 延遲門檻或感測頻率。wander 依 #20 舊 Decision 列為次要情境，與 curved 分開報告。

## #25：終端 BC 的失敗處理

[PR #76](https://github.com/Nues0913/Drosophila_Tracking_System/pull/76) 已合併，固定 head 的 PM 相關 CPU 測試 142 passed。schema 3 記錄已實作的規則：

- O／VO／VO-late 的 G4 ≥1e−3 rad/s，且 ≥同 chain odor-pretrain 的 3 倍；兩個邊界含等號。V／GRU／RS 的模態診斷只記錄。
- G4 用 CPU、單執行緒 float32、200 步、診斷 seed 1,610,000,000；不使用正式評估區塊。比例與 best epoch 只記錄。
- 終端 BC 的工程失敗強制讓矩陣在階段交界停止，`--keep-going` 也不能略過；PM 查明、另作 Decision 後才重跑受影響的組。
- 品質失敗照樣保留 actor、通過模態 gate 後評估，逐組揭露，不重跑、不換 seed。GRU 沒有生物 DN gate。pretrain 與研究用 PPO／ES 鏈保留嚴格 DN gate。

## #10：時間重估是外推，不是完整矩陣實測

4060 的 A runner 原始 wall 為 **3,525.74 s**（58.8 分），已含環境審計、兩個 pretrain、B3 校準、BC、DN 與 decoder 驗證；另外約 128 s 的重複獨立 DN probe 不算正式鏈再加一次。600 回合的 odor reliance 評估為 1,779.55 s。

4070 的訓練 log 有效區間：vision 369.79 s、odor 352.90 s、BC 1,189.79 s，均不含 log 沒記到的 setup／recovery。odor 跨使用者暫停，不能直接減頭尾時間。作者的 357.7 s odor 數字包含稍多的啟動時間，不和 log 區間混稱同一量測。監督器 active 總量 5,890.6 s，扣除五項評估／noise／置中診斷共 3,304 s，得到 **2,586.6 s 的訓練、審計與恢復合計餘額**；這是帳務差值，不是純 fit 時間。600 回合 odor reliance 評估為 1,395 s。

按 VO 單 chain 外推全部變體，每條另預留 75 s 給新 G4，主要評估按 600→400 回合等比例估計：

| 主機／分工 | 訓練鏈串行等效 | 四情境主要 2D 評估外推 | 合計 |
| --- | ---: | ---: | ---: |
| 4060：seed 0／2／4，18 條 | 約 18.0 h | 約 5.9 h | 約 23.9 h |
| 4070：seed 1／3，12 條 | 約 8.9 h | 約 3.1 h | 約 12.0 h |

**NOT VERIFIED**：六變體的實際時間、兩個同時工作的競爭、較難 seed、CPU gate 與 artifact 交付的額外開銷。沒有把 `--jobs 2` 當成兩倍加速，也沒有把這些估計當成上限。GRU 的 #77 修正後時間也未量測。表格不含 Gazebo、額外 lesion／展示或重試；正式配置仍是兩台各 `--jobs 2 --keep-going`，保留 **48–72 小時工作目標**，不減 seed、步數或比較。

## 本次設定與文件的自查

相關 CPU 測試最終 **194 passed**；40 回合環境審計的 teacher 成功率 1.0、所有盲目策略 0。另核對原始主要比較、統計設定、種子、步數、G2／G3 門檻和 train/model 配置都未改，五項 noise 值與原始選點完全一致；Ruff 與 diff check 通過。

第一輪既有 one-source 測試有 1 項失敗（193 passed），原因是先改 YAML 後尚未同步 ScenarioSpec 的 noise 預設；已修正並重跑同一批測試。原始失敗與最終通過的紀錄一起保留。未執行 CUDA、正式訓練或全變體 runtime 校準。

ScenarioSpec 的程式變更會改變 training fingerprint。歷史 4-R8 的 checkpoint 留在原固定 worktree；正式矩陣使用日後凍結的新版本，不能跨來源雜湊接續舊 checkpoint。

## 有效參數預算：#77 已修正，待 PR #79 審查與合併

歷史版本 main `79eff41`（#77 修正前）的 declared VO 是 16,243，但 BC 對 `core.bias` 的 1,697 個參數設 lr=0；PPO 移出後，它們在正式鏈全程凍結。當時非零學習率參數是 VO 14,546、GRU 15,516，GRU 高出 **6.6685%**，超過原有 ±5%。這是該版本靜態建模／optimizer 分組的確定結果，不是訓練實驗。

目前 ±5% 以 BC optimizer 中 **lr>0 且 requires_grad=True 的 actor 參數**為準，包含 B3 新增讀出列。`declared` 是 requires_grad 容量，`zero_lr` 是其中 lr=0 的部分，`effective = declared - zero_lr`；共同凍結前端、pretrain decoder 和 PPO critic 不計入。

| 變體 | declared | zero_lr | effective | 相對有效 VO |
| --- | ---: | ---: | ---: | ---: |
| V | 16,242 | 1,697 | 14,545 | −0.0069% |
| O | 16,243 | 1,697 | 14,546 | 0% |
| VO | 16,243 | 1,697 | 14,546 | 0% |
| VO-late | 15,790 | 1,697 | 14,093 | −3.1143% |
| GRU | 14,904 | 0 | 14,904 | +2.4612% |
| RS | 16,242 | 1,697 | 14,545 | −0.0069% |

GRU 已解析為 hidden=16、input bottleneck=18，effective 14,904（相對有效 VO +2.46%），此配置已獲 PM 接受；本輪保留該配置、bias 凍結及 ±5% 門檻。

評估在 `require_parameter_match` 開啟時，重用已載入且已還原 B3 列的 actor，以該 artifact 保存的 `metadata.config.train` 呼叫共用 `count_effective_parameters`。逐欄核對 `metadata.actor_parameter_budget` 的 declared／zero_lr／effective、各組 lr／numel／參數名稱與 numel，以及 `zero_lr_parameters`；lr 使用 `math.isclose(rel_tol=1e-9, abs_tol=0)`，零學習率仍須完全為零。頂層 `trainable_parameters` 須等於重算的 declared；缺少、無效或不一致皆在評估前拒絕，±5% 使用重算的 effective。schema 3 的 2026-10-28 amendment 一併記錄此核對規則。

正式開跑仍須 PR #79 通過 PM review 並合併，由 PM 凍結共同 main SHA；協定文件本身不是啟動令。
