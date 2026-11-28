# FlyTrack formal-v1 報告

> 版本：formal-v1（協定 schema 3，main `35432c3` 訓練與 2D 評估；Gazebo 用 main `a58b7b2`）。
> 這份報告整理已驗收的結果。逐日的紀錄和證據在 [status.md](https://github.com/Nues0913/Drosophila_Tracking_System/blob/team-archive-2026-11-28/docs/status.md)，總追蹤在 [#28](https://github.com/Nues0913/Drosophila_Tracking_System/issues/28)。

## 摘要
- 研究問題：在同一個感測前端之後，比較六個變體（V、O、VO、VO-late、GRU、RS）的追蹤表現。中央網路使用 MaleCNS 的連接體子圖。
- 2D（curved，主要運動）：
  - 沒有變體達到 G3 的 normal 成功率 0.8，最高是 VO-late 的 0.484。
  - VO 和 VO-late 對 V 的遮蔽優越性都不成立。
  - 對 GRU、RS 的不劣性名目上都成立，但兩個對照組都退化。GRU 修正後的事後結果（未登記）顯示，VO-late 對它四個情境都不劣。
- 主要限制：VO 和 VO-late 的前進速度是常數（0.168–0.179 m/s，目標 0.200）。兩個探索實驗 S1、D3 都沒有解決。
- Gazebo（G2，直接移轉，curved、wander 各 2,000 回合）：
  - 已登記比較的判讀和 2D 完全相同；只用 4070 的 seed、漏記撞牆的保守重算，都沒有改變任何判讀。
  - G3 也未達成：curved normal 最高是 V 的 0.450；參考主機 4070 上，VO-late 的延遲 p95 是 51.5 ms，超過 50 ms。

## 1. 方法
### 1.1 資料、前端與子圖
- 早期視覺用 flyvis 的預訓練前端（方案 A）；中央子圖取自 MaleCNS v1.0。
- 正式圖是 no-KC：4,502 個節點、31,682 條邊（#55）。
- 來源與驗證：[data_sources.md](data_sources.md)、[frontend_execution.md](https://github.com/Nues0913/Drosophila_Tracking_System/blob/team-archive-2026-11-28/docs/frontend_execution.md)、[olfactory_reextraction_55.md](https://github.com/Nues0913/Drosophila_Tracking_System/blob/team-archive-2026-11-28/docs/olfactory_reextraction_55.md)。
- 生物對應查證：[biology_verification.md](biology_verification.md)（#27）。

### 1.2 模型與變體
- 正式規格：no-KC 圖 + P2（左右共用的 ORN 適應）+ B3（DNb05 轉向列，只用在 O、VO、VO-late）。
- 讀出：前進讀 DNp09，轉向讀 DNa01、DNa02 的左右差。這是工程選擇，限制見第 6 節。
- 變體的定義和口徑：status 科學限制第 9 點。
- 有效參數預算：六個變體都在有效 VO（14,546）的 ±5% 以內（#77、PR #79）。

### 1.3 訓練
- 每條鏈是 pretrain → BC/DAgger。PPO 已移出正式鏈（#35 選項 D）。
- 規模：6 個變體 × 5 個 seed，共 30 條鏈，每條 390,000 步。
- 主機：4060 跑 seed 0、2、4，4070 跑 seed 1、3。

### 1.4 協定（schema 3，看資料之前固定）
- **情境**：normal、occlusion（第 8 秒起遮 30 秒）、noise（0.75 倍）、conflict。
- **成功的定義**：
  - 80% 的時間在 2 m 以內，而且方位誤差 ≤ 45°；
  - 有碰撞就不算成功。
- **統計**：
  - 主要終點是每回合的成功；
  - 先在每個訓練 seed 內平均，再做 seed 層級的 bootstrap（10,000 次），報 95% CI。
- **已登記的比較**：
  - VO、VO-late 對 V 的遮蔽優越性；
  - VO、VO-late 對 GRU、RS 的不劣性，界線是成功率 0.05。
- **G3**：normal 成功率 ≥ 0.8，而且 Gazebo 的 sensor→command p95 ≤ 50 ms。
- **主架構**：VO-late；所有原本登記的 VO 比較都保留。

## 2. 2D 結果（F3 curved、F4 wander；#80、#84）
### 2.1 主要終點：curved 的成功率
數字是五個 seed 的平均，括號是 95% bootstrap CI。

| 情境 | V | O | VO | VO-late | GRU | RS |
| --- | --- | --- | --- | --- | --- | --- |
| normal | 0.390 [0.358, 0.416] | 0.002 | 0.424 [0.376, 0.480] | 0.484 [0.378, 0.598] | 0.000 | 0.004 |
| occlusion | 0.112 [0.066, 0.158] | 0.002 | 0.136 [0.040, 0.232] | 0.124 [0.052, 0.186] | 0.000 | 0.004 |
| noise | 0.158 [0.116, 0.204] | 0.006 | 0.176 [0.114, 0.256] | 0.234 [0.152, 0.324] | 0.000 | 0.000 |
| conflict | 0.390 [0.360, 0.416] | 0.002 | 0.408 [0.362, 0.452] | 0.484 [0.370, 0.598] | 0.000 | 0.006 |

### 2.2 已登記的比較（curved）
| 比較 | 情境 | 差值 [95% CI] | 判讀 |
| --- | --- | --- | --- |
| VO − V（優越性） | occlusion | +0.024 [−0.112, +0.160] | 不成立，所以不做 VO 融合的主張 |
| VO-late − V | occlusion | +0.012 [−0.062, +0.082] | 不成立 |
| VO、VO-late − GRU／RS（不劣，界線 0.05） | 四個情境，共 16 項 | CI 下界最低 +0.032 | 名目上都成立，但對照組退化 |

- wander（次要運動）的完整數字見 #84 的 Decision。

### 2.3 對照組退化與事後 GRU（#85、#87，未登記）
- GRU：共用的 ×100 ORN 增益讓輸入瓶頸飽和，GRU 看不到視覺，這是工程缺陷。修正後重新訓練，結果另外標為「未登記」。
- RS：照固定標準診斷，不是工程缺陷，照登記結果報告。
- curved：VO-late 對事後 GRU 的四個情境都不劣；VO 在 noise 不成立。
- wander：VO 和 VO-late 在 normal、conflict 都不成立。
- 完整表格見 status 的 2026-11-04 一節。

### 2.4 次要終點（curved，描述性）
協定列的次要終點（`configs/evaluation_protocol.yaml`）。數字是五個 seed 的平均（每個 seed 先平均回合），和評估器的 `summary.json` 相同，CI 在 `formal_v1_eval.tar.gz` 的 `summary.json`。延遲見 5.3。

| 變體 | 平均距離 normal（m） | 方位誤差 normal（°） | 平順度 normal（rad/s） | 碰撞率 normal／occlusion／noise | 重新捕獲 occlusion（s；完成／事件） |
| --- | --- | --- | --- | --- | --- |
| V | 2.78 | 39.3 | 0.0020 | 0.07／0.46／0.39 | 6.7（108／500） |
| O | 3.08 | 81.6 | 0.0004 | 0.81／0.81／0.84 | 7.9（13／500） |
| VO | 2.69 | 40.3 | 0.0019 | 0.06／0.48／0.44 | 8.1（138／500） |
| VO-late | 2.53 | 37.2 | 0.0015 | 0.06／0.48／0.51 | 9.3（152／500） |
| GRU | 2.58 | 75.5 | 0.0003 | 0.97／0.97／0.91 | 3.0（5／500） |
| RS | 2.71 | 69.3 | 0.0025 | 0.83／0.81／0.51 | 6.4（18／500） |

- **重新捕獲**：時間只算完成的事件（評估器的定義），所以旁邊列出完成／遮蔽事件數。V、VO、VO-late 的 500 次遮蔽裡，完成重新捕獲的只有 108–152 次（22%–30%），其餘是截尾。截尾的事件裡，59%–68% 是回合撞牆結束（V 230／392、VO 240／362、VO-late 238／348），和 A1 看到的遮蔽時直線撞牆一致。所以重新捕獲時間只代表完成的那些事件，不能拿來比較變體。
- **只算到碰撞為止**：評估器碰撞就結束回合，所以平均距離、方位誤差和平順度只平均到碰撞為止。O、GRU、RS 大多數回合提早結束（normal 的回合長度中位數約 650–800 步，滿長是 1,200 步；V、VO、VO-late 是 1,200 步），所以這三項不能和 V、VO、VO-late 比較。例如 GRU 的平均距離 2.58 m 比 V 的 2.78 m 小，但 GRU 沒有碰撞的 15 回合，平均距離是 3.10 m。
- **碰撞**：O、GRU、RS 在 normal、occlusion 的碰撞率是 0.81–0.97（RS 在 noise 是 0.51）。V、VO、VO-late 在 normal 是 0.06–0.07，在 occlusion、noise 是 0.39–0.51。
- **平順度**：noise 情境的數值比 normal 大一個數量級以上，例如 V 是 0.080 對 0.002。
- 四個情境的完整表格，以及 Gazebo 的表格，在 release `formal-v1` 的 `pm_secondary_endpoints.tar.gz`（`76248594…`）。

## 3. 分析（A1，#90，描述性診斷）
- **拿掉視覺**：四個變體的成功率都降到 0。
- **拿掉氣味**：
  - VO 的成功率沒有下降。
  - VO-late 下降 0.11–0.13，但 conflict 對它沒有影響，所以這個下降不代表它用了氣味的方向。
- **匯流電流**：約九成是 odor_exclusive，但電流的比例不能當成行為貢獻。
- **失敗案例**：
  - occlusion 失敗多半是延續遮蔽開始時的朝向，直線撞牆；
  - normal 失敗多半是跟在 2 m 門檻外一點，因為速度不夠。
- **前進速度是常數**：反事實測試把速度固定在 0.21 m/s 時，距離失敗大致消失，但改成撞上目標，成功率沒有提高。

## 4. 探索實驗（都不改 formal-v1 的結果）
| 實驗 | 改動 | 判讀 | 證據 |
| --- | --- | --- | --- |
| S1（#95） | DNp09 速度輸入固定放大 | 無效：放大後往上游的梯度太大，seed 0 的轉向壞掉 | `s1_speed_readout.tar.gz` |
| R1（#4） | 遮蔽時加氣味 teacher 的 BC 配方 | 無效 | `r1_recipe.tar.gz` |
| R0（#28） | formal 配方原樣重訓 | 穩定（normal 的差 ≤ 0.10） | `r0_retrain_variance.tar.gz` |
| D3（#104） | 速度讀出加上校準過、切斷梯度的 DN 列 | 不成立：速度只是整體變快，撞上目標的次數變多 | `d3_forward_rows.tar.gz` |

- formal-v2 沒有觸發。使用者 11-11 同意 PM 的建議，formal-v1 不再做速度讀出的實驗（[#35 6894303266](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6894303266)）；P3／PPO 留在 #35。
- formal-v1 交付之後的探索（#35，2026-11-16 起：S2、M1、R2、S4 等）都標為未登記，不改這份報告的結論；結果和結論見 [status.md](https://github.com/Nues0913/Drosophila_Tracking_System/blob/team-archive-2026-11-28/docs/status.md) 的 2026-11-18、2026-11-17、2026-11-16 三節，證據包在 release `research-35`。11-18 的 T1（可見度閘門轉向讀出）和 S8（衍生圖加閘門）以 wander 遮蔽為主要終點，照登記的規則判定都不成立。

## 5. Gazebo（#88）
### 5.1 平台的一致性
- **G0b**（PR #99）：policy 改用單執行緒，計時改由 evaluator 補齊。
- **G0c**（PR #102）：改用平光材質。修改前，Gazebo 的視覺輸入和 2D 明暗相反，逐幀相關係數 r 是 −0.97；修改後 r 是 0.97–0.98。細節見 [gazebo_consistency.md](https://github.com/Nues0913/Drosophila_Tracking_System/blob/team-archive-2026-11-28/docs/gazebo_consistency.md)。
- **G1 範圍**：2060、4070、4060 三台主機，和 2D 逐回合配對，三台都是 47/48 一致，判定 GO。
- **G2 期間發現、G2 之後才修的兩個問題**（G0d，PR #107，G2 結束後合併，main `8e19c9b`）：
  - 斜角撞牆記不到碰撞（6763512170）；
  - 轉向比指令快 12%（6764426046）。
  - 修正後在 2060 上重跑 G1 範圍，仍然是 GO，兩邊都撞上時，撞上時間的差距從 −18 到 +15 s 縮到 0 到 2.4 s。G2 用的是修正前的世界。
- **目標對比**（6783245712、6788230860）：Gazebo 的目標比 2D 暗約 10%，相當於 2D 亮度 0.80–0.85。
  - 在 2D 把 G2 範圍的目標調暗到 0.82（事後、描述性），每格成功率變化 0 到 −0.09，每格成功與否相同的比例 91%–100%，變體之間的差距最多移動 0.08。
  - 正式 GRU、RS 本來就退化（成功率 0），變暗也不變；事後 GRU 也會受影響，每格變化 +0.01 到 −0.07，和 VO-late 差不多（6845484730）。所有變體合計，每格一致比例最低 91%，變體差距最多移動 0.08。

### 5.2 G2：主要比較
- **規模**：curved、wander 各 5 個變體 × 4 個情境 × 5 個 seed × 20 回合，另外加上事後 GRU。程式固定在 main `a58b7b2`（G0c 世界），直接移轉、不微調。
- **分配**：4070 跑 curved 0–2、wander 0–1；4060 跑其餘的 seed。
- **交付**：4060 在 11-11、4070 在 11-14 交件，PM 的收件檢查都通過（雜湊、分配，用凍結的 kit v2 重算每個區塊；#88 6806858753、6872441153）。彙總照事先登記的方式，分析 checkout 是 main `04f1708`，重跑一次結果逐檔相同。

**curved 的成功率**（Gazebo，五個 seed 的平均，括號是 95% bootstrap CI）

| 情境 | V | VO | VO-late | GRU | RS |
| --- | --- | --- | --- | --- | --- |
| normal | 0.450 [0.380, 0.540] | 0.400 [0.300, 0.510] | 0.440 [0.310, 0.620] | 0.000 | 0.000 |
| occlusion | 0.100 [0.050, 0.140] | 0.130 [0.030, 0.240] | 0.100 [0.050, 0.180] | 0.000 | 0.000 |
| noise | 0.250 [0.190, 0.310] | 0.190 [0.090, 0.300] | 0.300 [0.210, 0.410] | 0.000 | 0.020 [0.000, 0.060] |
| conflict | 0.440 [0.360, 0.530] | 0.380 [0.280, 0.510] | 0.430 [0.340, 0.550] | 0.000 | 0.000 |

- wander 的 normal：V 0.160、VO 0.180、VO-late 0.200，GRU、RS 都是 0。完整的表在 PM 的分析包（`contrast/success_table.md`）。
- G2 每個 seed 只跑 20 回合（每格 100 回合），2D 的表（2.1）是每個 seed 100 回合（每格 500 回合），G2 的回合是其中的第 0–19 回合。所以兩個平台要用逐回合配對來比，不要直接比兩張表的數字。

**已登記的比較**

| 比較 | 情境 | curved [95% CI] | wander [95% CI] | 判讀 |
| --- | --- | --- | --- | --- |
| VO − V（優越性） | occlusion | +0.030 [−0.090, +0.150] | +0.010 [−0.030, +0.060] | 都不成立 |
| VO-late − V（優越性） | occlusion | +0.000 [−0.060, +0.060] | −0.010 [−0.030, +0.000] | 都不成立 |
| VO、VO-late − GRU／RS（不劣，界線 0.05） | 四個情境，每種運動 16 項 | CI 下界最低 +0.030 | CI 下界最低 +0.000 | 名目上都成立，但對照組退化 |

- **判讀和 2D 完全相同**（2.2，wander 見 #84）。
- **只用 4070 seed 的敏感度分析**（#88 6747441878）：36 項的判讀都沒有改變，沒有比較要標「受主機延遲影響」。
- **漏記撞牆**（#88 6763512170）：
  - 保守重算把 curved 8 回合、wander 32 回合改成失敗，36 項的判讀都沒有改變，沒有比較要標「受漏記撞牆影響」。
  - 和事先的預估對照（#88 6764426046；curved 的 V、VO、VO-late，2D 撞過的 303 回合）：2D 撞到、Gazebo 沒撞到的有 107 回合，預估是 130；反過來，Gazebo 撞到、2D 沒撞到的有 60 回合。軌跡分岔以後就不能一一對應，只能當成粗略的對照。
- **目標對比的判讀**（6788230860，G2 資料之前固定）：
  - 逐回合配對表裡，成功與否相同的比例 ≥ 91% 的格子，單靠目標對比就可能造成；
  - 和 2D 相差 ≤ 0.08 的變體差異，註明「可能受目標對比影響」。
  - 結果：
    - 36 項登記比較和 2D 配對回合的差距都 ≤ 0.08（最大 0.06），所以都註明「可能受目標對比影響」。
    - GRU、RS 每格一致 98%–100%。V、VO、VO-late 兩種運動各有 5／20 格低於 91%，不能只用目標對比解釋：curved 的 V noise（79%）、VO noise（87%）、VO-late normal（85%）、noise（86%）、conflict（88%），wander 的 V normal（88%）、conflict（90%）、VO normal（87%）、VO-late normal（86%）、noise（88%）。
    - 其中最大的是 curved V noise：Gazebo 0.25 對 2D 0.14，有 16 回合只有 Gazebo 成功。
- **事後 GRU**（未登記，描述性）：
  - Gazebo 的 curved 成功率是 normal 0.250、occlusion 0.010、noise 0.130、conflict 0.300。
  - 不劣性和 2D（2.3）判讀不同的有 4 項：curved 的 VO conflict（Gazebo 不成立），wander 的 VO normal、VO conflict、VO-late conflict（Gazebo 成立）。區間都很寬。
  - 主架構 VO-late 在 curved 的四個情境都不劣，和 2D 相同。
  - 事後 GRU 在 Gazebo 的 conflict，curved 是 0.30（2D 配對 0.24），wander 是 0.08（2D 配對 0.21），原因沒有查。
- **次要終點**（curved，描述性；括號是同一批回合在 2D 的值。完整表格在 `pm_secondary_endpoints.tar.gz`）：

| 變體 | 平均距離 normal（m） | 方位誤差 normal（°） | 碰撞率 normal／occlusion／noise（只列 Gazebo） | 重新捕獲 occlusion（s；完成／事件） |
| --- | --- | --- | --- | --- |
| V | 2.67（2.66） | 43.4（37.1） | 0.13／0.38／0.23 | 6.7，26／100（7.1，24／100） |
| VO | 2.73（2.65） | 47.4（39.0） | 0.07／0.44／0.32 | 7.3，32／100（7.2，32／100） |
| VO-late | 2.70（2.47） | 46.5（37.0） | 0.08／0.33／0.34 | 8.4，33／100（7.8，28／100） |
| GRU | 2.77（2.60） | 80.5（76.9） | 0.92／0.92／0.86 | —，0／100（2.3，1／100） |
| RS | 2.78（2.66） | 72.6（70.8） | 0.81／0.80／0.44 | 7.1，5／100（15.1，4／100） |

  - normal 的方位誤差，V、VO、VO-late 在 Gazebo 比 2D 大 6–10°。
  - 重新捕獲完成的比例和 2D 接近：V、VO、VO-late 是 26–33／100，2D 是 24–32／100。
  - Gazebo 的碰撞率會少算（揭露第 24 點），所以不和 2D 比較。
  - 和 2.4 一樣，平均距離和方位誤差只算到碰撞為止，GRU、RS 大多數回合提早結束（normal 的回合長度中位數約 650–670 步），不能和 V、VO、VO-late 比較。
  - 截尾的遮蔽事件裡，回合撞牆結束的有 V 38／74、VO 43／68、VO-late 32／67。Gazebo 的碰撞會少算，所以這是下限。
- **兩台主機的執行狀況**：
  - 主機中斷，照例外規則重跑一次（6743008783）：4070 的 curved 1（主機使用者中止）、curved 2（Windows Update 重開機）；4060 的 curved 4、wander 3（主機關機）。未完成的嘗試都不進彙總。
  - 4060 的 wander seed 2，最後三格 RS 和主機上一段無關的 GPU 負載重疊，沒有重跑（6798991965、6806858753）。延遲見 5.3。
  - 4070 的 wander seed 1 有一回合 invalid（V normal episode 7，coverage 0.782），保留，算成失敗（6836545772、6843506724）。這一格的成功率最多被壓低 0.05，它不在任何登記的比較裡。
  - 4070 的 posthoc 區塊中途照主機使用者的要求暫停約 4.5 小時，已完成的區塊沒有重跑。4070 在 G2 期間沒有記錄 GPU／CPU 的監控資料，4060 有。

### 5.3 G3
- 延遲：參考主機 4070，curved normal 的所有計時命令合併後算 p95，必須 ≤ 50 ms。
- 成功率：curved normal 五個 seed 的平均，必須 ≥ 0.8。
- 結果：五個變體都**未達成**。

| 變體 | 成功率 [95% CI] | 4070 的 p95 | G3 |
| --- | --- | --- | --- |
| V | 0.450 [0.380, 0.540] | 49.3 ms | 未達成 |
| VO | 0.400 [0.300, 0.510] | 49.0 ms | 未達成 |
| VO-late | 0.440 [0.310, 0.620] | 51.5 ms | 未達成（延遲也超過） |
| GRU | 0.000 | 47.1 ms | 未達成 |
| RS | 0.000 | 50.5 ms | 未達成（延遲也超過） |
| 事後 GRU（未登記） | 0.250 [0.160, 0.340] | 47.5 ms | 未達成 |

- **formal-v1 之後的延遲重測**（#110，4070，main `e5a166b`）：三項等價加速之後，CPU 版六組合併 p95 是 43.4–44.9 ms，低於 50 ms；GPU 版 52.3–53.1 ms。成功率沒有改善，上表的判定不變。取樣核和 G2 不同（subrays 對 gaussian），所以和上表的差只當參考。
- **4060 的延遲**（只報告，不用來判定）：sensor→command p50 分成 42.5–59.6 ms 和 107.7–126.6 ms 兩群，另有一格是 79.9 ms。分界是推論時間：每格推論時間的中位數 ≤ 24.3 ms 的都在低的那一群，≥ 25.0 ms 的除了一格都在高的那一群，大約多一個控制週期（50 ms）。
  - VO-late 每格推論時間的中位數是 26.2–31.5 ms，所以在 4060 上每一格都在高的那一群。
  - 有負載的 RS noise、conflict，推論時間是 28.6、27.8 ms，比沒有負載的 RS 格子都高（≤ 25.2 ms），可以歸因於負載把推論時間推過門檻；occlusion（25.1 ms）不能確定。
  - 分群用 PM 的 G2 分析包裡的 `latency_groups.py` 算，逐格的值在同一個包。
- **4070 的延遲**：沒有分群。每格推論時間是 12.8–19.9 ms，都在 4060 的門檻以下；sensor→command p50 是 33.0–42.6 ms，連續分布，最大的間隔只有 0.7 ms。

## 6. 必須揭露的限制
完整清單是 status 的「報告須揭露的科學限制與工程近似」第 1–26 點。和結論直接相關的有：
- 第 16 點：對照組退化；
- 第 17 點：G3 未達成（2D 和 Gazebo）；
- 第 18 點：前進速度是常數；
- 第 19 點：沒有風的輸入；
- 第 20 點：記憶的時間尺度；
- 第 22 點：延遲和主機有關；
- 第 23 點：Gazebo 的渲染改成符合 2D。
- 第 24 點：Gazebo 斜角撞牆記不到碰撞；第 25 點：G2 世界的轉向比指令快 12%（兩者都由 G0d 修正，G2 不重跑）。
- 第 26 點：Gazebo 的目標比 2D 暗約 10%（G2 之後已找到原因並修正，G0e／PR #118；G2 不重跑）。

## 7. 展示
- **2D**：V1（#97）的 `formal_v1_demo_2d.tar.gz`，內容是軌跡、DN 活性和動作圖。
- **Gazebo**：release `formal-v1` 的 `gazebo_demo_g0d_2060.tar.gz`（`4c2b74d7…`）。
  - V1 選的 11 個回合，用同一個 actor 和環境 seed，在 G0d 世界（main `8e19c9b`）重播，含 rosbag、四格圖和 2D 對照。
  - 成功與否和 2D 相同的有 11／11，G0d 之前的世界是 10／11。發現斜角撞牆漏記的那一回合（V occlusion ep2），現在和 2D 一樣在撞牆時結束。碰撞不是每個回合都一致：VO-late occlusion ep3 在 2D 撞牆結束，在 Gazebo 沒有撞牆，但同樣失敗。
  - 這是展示，不是量測；行為數字以 G2 為準。
- **Gazebo 3D 影片 V2**（#123，PR #124，formal-v1 之後）：release `research-35` 的 `gazebo_v2_demo_4070_highlights.mp4`（88 s 精華）和 `gazebo_v2_demo_4070.tar.gz`（`13ef05de…`）。
  - 用預設關閉的跟拍和全場相機，在 main `ed71149`（含 G0e、G0g）重播同樣的 11 個回合。
  - 成功與否和有沒有碰撞，11／11 和 2D 相同。VO-late occlusion ep3 這次和 2D 一樣撞牆，和 G0d 版不同。
  - 錄影時每格頻率最低 19.970 Hz；相機打開時，policy 的影像和關閉時逐位元相同。

## 8. 重現
release [`formal-v1`](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/formal-v1) 的主要檔案：

| 檔案 | 內容 |
| --- | --- |
| `formal_v1_4060_F1.tar.gz`、`formal_v1_4070_F2.tar.gz` | 30 條訓練鏈的 actor 和紀錄 |
| `formal_v1_pretrain_seed0/1.tar.gz` | seed 0、1 的 O、VO-late pretrain actor |
| `formal_v1_eval.tar.gz`、`formal_v1_eval_wander.tar.gz`、`formal_v1_registered.tar.gz` | 2D 評估，以及已登記比較的重新彙總 |
| `formal_v1_analysis_a1.tar.gz`、`pm_a1_speed_check_Nues0913.tar.gz` | A1 分析，以及速度的反事實測試 |
| `s1_speed_readout.tar.gz`、`r1_recipe.tar.gz`、`r0_retrain_variance.tar.gz`、`d3_forward_rows.tar.gz` | 探索實驗 |
| `formal_v1_demo_2d.tar.gz` | 2D 展示 |
| `gazebo_g0c_consistency_2060.tar.gz`、`gazebo_g1c_{2060,4070,4060}.tar.gz`、`gazebo_g2_kit_v2.tar.gz` | Gazebo 的一致性量測、G1 範圍和 G2 工具包 |
| `gazebo_g0d_2060.tar.gz`、`gazebo_g0d_2060_w15.tar.gz` | G0d 的量測和 G0d 世界的 G1 範圍 |
| `pm_2d_target_brightness_g2scope.tar.gz`（`b1ce5ecd…`） | 2D 目標亮度敏感度分析：腳本、設定、評估輸出、逐格表；參考回合用 `formal_v1_eval*.tar.gz` 和 `posthoc_gru_85.tar.gz` |
| `gazebo_g2_4070.tar.gz`（`f2c26c6e…`）、`gazebo_g2_4060.tar.gz`（`561ea8f4…`） | G2 的原始資料 |
| `gazebo_demo_g0d_2060.tar.gz`（`4c2b74d7…`） | Gazebo 展示和 rosbag（G0d 世界） |
| `pm_secondary_endpoints.tar.gz`（`76248594…`） | 協定的次要終點，curved，2D 和 Gazebo |
| `pm_g2_analysis.tar.gz`（`f716e708…`） | PM 的 G2 分析：收件檢查、登記的彙總、和 2D 逐回合配對、保守重算、G3 判定、只用 4070 的敏感度分析、目標對比的判讀、延遲分群 |

- 事後 GRU 在 release `posthoc-gru-85`。
- 要重現已登記的 GRU，必須用 `35432c3`；從 `93ec7cd` 起，main 產生的是事後 GRU。
