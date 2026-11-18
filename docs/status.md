# 目前進度與已知問題

最後更新：2026-11-18（formal-v1 之後的研究和工程都不改 formal-v1 的結論，見下方 2026-11-18、2026-11-17、2026-11-16 三節。以下是 formal-v1 的狀態：正式矩陣的 30 條訓練鏈和 2D 正式評估（curved、wander）都在 main `35432c3` 完成並驗收，交付在 release `formal-v1`；已登記比較依 PR #83 重新彙總。GRU 對照組已依 #85 修正（PR #86）並完成事後、未登記的重新訓練與評估（#87）。協定必交的分析 A1 已驗收（#90），發現 VO／VO-late 的前進速度是常數；接著的探索實驗 S1（速度讀出校準，#95）和 R1（BC 配方，#4）都判定無效；formal 配方原樣重訓（R0）顯示基準在重訓下穩定。D2 的探索到此結束，formal-v2 不觸發；使用者 11-10 要求再試速度讀出，探索實驗 D3（#104，切斷梯度的 DN 列）也判定不成立。Gazebo 的 wander 接線已合併（PR #94），G1 試跑發現 policy 跟不上 20 Hz、因果計時大多沒有對上，已由 G0b（PR #99）修正；2060 和 4070 的 G1 範圍，valid、頻率與計時都通過（行為因明暗相反沒有意義）；G2 派工後，PM 量到 Gazebo 的視覺輸入和 2D 明暗相反，G2 暫停；修正 G0c 已合併（PR #102），2060、4070、4060 用新世界的 G1 範圍，行為都和 2D 一致，G2 在 4070、4060 上跑完並通過收件（#88）：已登記比較的判讀和 2D 完全相同，只用 4070 的 seed、漏記撞牆的保守重算都沒有改變判讀；G3 在 Gazebo 也未達成（curved normal 最高 0.450，4070 上 VO-late 的延遲 p95 51.5 ms）。G2 期間 PM 在 G2 的世界量到兩個和 2D 的差異：斜角撞牆記不到碰撞、轉向比指令快 12%，處理方式在資料出來之前登記；修正 G0d 已在 G2 結束後合併（PR #107）。2D 展示素材 V1 已交付並驗收（#97）。正式規格維持 no-KC＋P2/B3、pretrain → BC/DAgger（#35），團隊選 F：保留全部已登記比較，如實報告 VO 的有限氣味反應及負面結果。待辦與分工以 GitHub Issue 為準，總追蹤是 [#28](https://github.com/Nues0913/Drosophila_Tracking_System/issues/28)。協作規則見 [AGENT_TEAM_WORKFLOW.md](../AGENT_TEAM_WORKFLOW.md)，逐項檢查清單見 [後續計畫.md](../後續計畫.md)，驗證證據見 [execution_log.md](execution_log.md)。

## 2026-11-18 T1／S8 結果、G3 延遲重測、G0g 與 Gazebo 3D 展示（#35、#110、#121、#123）

接續下面 2026-11-17 一節。都在 formal-v1 之後，formal-v1 的結論不變。證據包在 release `research-35`。

### 使用者的決定（11-17）
2026-11-17 一節列出四項變動。使用者決定照 3 → 2 → 1 的順序做，第 4 項（plume 衰減）不做：
- 3：後續研究的主要終點改成 2D wander 遮蔽的成功率，curved 照樣報告。
- 2：T1，加可見度閘門的轉向讀出。
- 1：S8，衍生圖（補回 LC9→DNp09）加上同一個閘門。

做法和判讀規則都在看資料之前登記（[#35 6948106054](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6948106054)）。

### T1／S8：都不成立（[#35 6972415415](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6972415415)）
- **做法**
  - **T1**
    - 正式 VO-late actor 不動。轉向改成 ω = (1 − p)·ω_formal + p·h_odor，速度用 actor 自己的。
    - h_odor 是正式 6 個轉向特徵上的 ridge 回歸，只用遮蔽時的步數，標籤是 R1 的氣味 teacher 轉向。
    - p 是 134 型雙側 DN 群體上的邏輯回歸，學的是「現在是否遮蔽」。
    - 在 wander 上做 4 輪 DAgger。
    - 讀出放在 artifact 外，所以這是探針層級的閉環評估。
  - **S8**：S6 的衍生圖，移植正式 BC actor；核心凍結，只擬合 DNp09 的速度讀出（S7 的做法），再加上 T1 的閘門。
- **判讀**
  - VO-late 的 5 個正式 seed，每個 seed 100 回合，評估區段是 1,990,000,000 + i，和正式 actor（base）照環境 seed 配對。
  - 四項規則都要過：
    1. 遮蔽成功率 ≥ base + 0.10，而且 bootstrap 95% CI 的下限 > 0；
    2. normal 成功率 ≥ base − 0.05；
    3. normal 方位誤差 ≤ base + 5°；
    4. normal 碰撞率 ≤ base + 0.05。

| 候選 | wander 遮蔽成功率 vs base | 差 [95% CI] | normal 成功率差 | normal 方位誤差差 | normal 碰撞差 | 規則 1／2／3／4 |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | 0.034 vs 0.014 | +0.020 [+0.002, +0.038] | +0.002 | +2.8° | −0.036 | ✗／✓／✓／✓ |
| S8 | 0.000 vs 0.014 | −0.014 [−0.026, −0.004] | −0.122 | +29.0° | −0.054 | ✗／✗／✗／✓ |
| S8 自己的轉向 | 0.000 vs 0.014 | −0.014 [−0.026, −0.004] | −0.122 | +28.9° | −0.060 | ✗／✗／✗／✓ |

- **T1**
  - 閘門能分辨遮蔽與否：wander 遮蔽情境的閘門正確率是 0.987。氣味轉向頭的擬合 R²（遮蔽時）是 0.55–0.76。
  - 遮蔽成功率只多了 0.02，遠低於登記的 +0.10。normal 不受影響；遮蔽時的碰撞率從 0.508 降到 0.366。
  - 瓶頸不在偵測遮蔽，而在遮蔽時只靠氣味轉向追不回目標。
  - 11-17 的零訓練探針（感覺碼直接讀轉向，加上 teacher 的速度）遮蔽可以到 0.70，T1 只有 0.034。兩者的差別在轉向特徵的來源和速度，這次沒有分開檢驗。
- **S8**
  - 移植之後，seed 1–4 的 normal 成功率掉了 0.13–0.29，只有 seed 0 變好（+0.20）。
  - S8 比 T1 的遮蔽成功率低 0.034，normal 成功率低 0.124。這次的移植方式不可行。
- **偏離登記**
  - 登記之後、評估之前，S8 的閘門 DN 池改成直接建，因為 S8 actor 已經帶有 DNp09 的 forward rows。T1 的池和修正前逐位元相同。
  - curved 原本要跑 5 個 seed。使用者要求直接下結論，所以停在 seed 0–2（seed 2 只跑了一部分）。curved 不是判讀終點，不影響結論。
- **curved（探索性，沒有登記）**
  - T1 的遮蔽成功率是 0.128 vs 0.034（3 個 seed 方向一致），normal 和 base 相同；S8 的遮蔽成功率是 0。
  - 只有 3 個 seed，也沒有登記，只能當假說。
- 證據包：`research_t1_s8_2060.tar.gz`（`67eebeb2…`），不含權重。
- 閘門不做進 artifact，formal-v1 的 artifact 不變。

### G3 延遲重測（#110，4070，main `e5a166b`）
- 在三項等價加速（#109、#115、#117）之後，cweiting 在 4070 上照 G2 的規則重測。
  - CPU 版六組合併 p95 是 43.4–44.9 ms，低於 50 ms；VO-late 是 44.9 ms，G2 時是 51.5 ms。
  - GPU 版 52.3–53.1 ms，只當參考。
  - PM 從 control traces 獨立重算，數字相符（[#110 6957460972](https://github.com/Nues0913/Drosophila_Tracking_System/issues/110#issuecomment-6957460972)）。
- **formal-v1 的 G3 判定照舊（未達成）**：成功率那一半沒有改善。另外這次用新的取樣核（subrays），G2 用的是 gaussian，所以和 G2 的差只當參考，不算同條件的配對差。

### G0g（#121，PR #122，main `daa6e5f`）
- 目標的 collision 圓柱從 0.15 m 縮到 0.145 m，視覺外形不變。兩個中心最近可以到 0.325 m，比判定距離 0.33 m 小。
- 用 contact sensor 核對：12 個回合都有旗標，旗標減第一次 contact 是 −0.022 到 +0.020 s；G1 範圍 48/48 和 2D 相同（[#121 6958797577](https://github.com/Nues0913/Drosophila_Tracking_System/issues/121#issuecomment-6958797577)）。
- 揭露第 11 點已經更新，G2 不重跑。

### Gazebo 3D 展示 V2（#123，PR #124，main `ed71149`）
- 加了預設關閉的展示相機（跟拍和全場），在目前的 main 上重播 V1 的 11 個回合；main 已經包含 G0e 和 G0g。
  - 成功與否和有沒有碰撞，11／11 和 2D 相同；方位誤差最多差 0.40°。
  - 和 G0d 版不同的只有 VO-late occlusion ep3：這次和 2D 一樣在約 40.8 s 撞牆，G0d 版沒有撞。三個版本都是失敗，原因（G0e 或 G0g）不追。
  - 錄影時每格頻率最低 19.970 Hz，計時比例 1.0000。
  - 相機打開時，policy 的影像和關閉時逐位元相同。
- 檔案：`gazebo_v2_demo_4070_highlights.mp4`（88 s 精華）、`gazebo_v2_demo_4070.tar.gz`（`13ef05de…`，11 支影片、對照表和證據）、`gazebo_v2_demo_4070_frames.tar.gz`（原始影格）。驗收在 [#123 6967114350](https://github.com/Nues0913/Drosophila_Tracking_System/issues/123#issuecomment-6967114350)。

### G2 感官方位解碼：更正驗收對照表
- [requirements_traceability.md](requirements_traceability.md) 原本寫「正式預算的解碼與門檻確認未完成」，這是 11-10 之後沒有更新。
- 實際上正式 pretrain 的每條鏈都做了 validate-pretrain，全部通過：
  - 視覺 25 個 actor 的方位 MAE 是 0.78–6.98°（門檻 15°）；
  - 嗅覺 25 個是 1.99–4.28°（門檻 30°）；
  - presence 正確率 ≥ 0.9915。
- 檔案在 release `formal-v1` 的 F1／F2 包，路徑是 `runs/pretrain/*/seed_*/{vision,odor}/validation/metrics.json`。對照表已改成「達成」。

## 2026-11-17 formal-v1 之後的研究：結果與結論（#35）

接續下面 2026-11-16 一節。全部是未登記的探索，formal-v1 的結論不變；證據包在 release `research-35`。

### 結論
1. **速度固定有結構上的原因：DNp09 的真實輸入在子圖裡只剩 11.6%。**
   - 真實連接體（MaleCNS v1.0）裡，DNp09 最大的輸入是 LC9 直接連過來的 1,977 個突觸（12.4%，全部同側）。子圖的門檻是「單一條邊 ≥ 50 突觸」，LC9→DNp09 是 122 顆細胞各十幾個突觸（最多 45），整組被濾掉；LC9 本身也只剩 21／219 顆。轉向的 DNa01／DNa02 保留了 45%／41%。
   - 把這些邊接回去（探索用的衍生圖，正式子圖不動）之後，DNp09 的電壓標準差從 6e-6 變成 0.01–0.04，和 teacher 期望速度的相關方向也對。
2. **但目前的訓練方式還是沒辦法把速度學出來。** S2、S4、S6、S5、S7 五個做法都沒有通過事先登記的五項規則（下表）。共同的卡點有兩個：讀出前的訊號幅度很小，靠梯度學增益太慢；直接擬合讀出時速度會變，但碰撞變多，加感測雜訊時成功率掉到 0。
3. **腳本與感覺碼探針顯示 wander 遮蔽適合檢驗氣味的幫助；尚未證明可部署模型的多感官融合成功。**
   - 遮蔽前由特權 teacher 控制、遮蔽後速度維持最後值的腳本中，curved 只維持最後動作的成功率是 0.59，改用氣味轉向是 0.55；wander 則是 0.07 → 0.52（100 回合）。這支持後續在 wander 檢驗氣味的作用，不能據此判定 curved 無法檢驗融合。
   - formal-v1 已評估 F3 curved 和 F4 wander 的四個情境，包含遮蔽；wander 是已登記的次要運動（[#84 驗收](https://github.com/Nues0913/Drosophila_Tracking_System/issues/84#issuecomment-6541430822)）。R1 才只在 curved 評估。formal-v1 的融合主張仍依原協定判讀。
   - 感覺碼裡的氣味差和原始的左右濃度對比相關 0.999。從感覺碼直接讀出轉向、再用視覺碼學可見度閘門，wander 遮蔽 0.70、normal 1.00（20 回合）。**這個探針繞過連接體核心，且全程用 teacher 的特權速度**；只能支持閘門的可行性。切換不足可能是 R1 的限制之一，還不能認定是唯一原因或視為正式 VO-late 的結果。
4. **Gazebo 的兩個平台差異都找到原因**：目標對比偏低是相機取樣核（G0e，已修正，揭露第 26 點）；撞目標的旗標會晚，是物理外形的半徑和剛好等於判定距離（G0f，修正排在 G0g／#121，揭露第 11 點）。**RS 退化的原因只找到一部分**：保留度數的隨機重接打亂了左右對應（2026-11-16 一節），但保留左右側的隨機接線（R2）訓練之後也完全退化（成功率 0），所以左右對應不是唯一的原因。
5. G3 仍未達成。延遲的部分三個等價的加速（#109、#115、#117）都已合併，4070 上的正式重測（#110，main `e5a166b`）還沒做。（11-18：重測已完成，CPU 版 p95 低於 50 ms，G3 判定不變，見上方 2026-11-18 一節。）

### 訓練實驗（都用 S2 的五項規則；VO-late seed 0、1，每格 100 回合）
| 實驗 | 做法 | normal 成功率（基準 → 候選，seed 0／1） | 速度回合內標準差／和 teacher 的相關 | 判讀 |
| --- | --- | --- | --- | --- |
| S2（[6906921113](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6906921113)） | BC 時平衡速度和轉向的梯度 | — | 0.0000 | 不成立 |
| S4（[6917660448](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6917660448)） | 速度改讀 134 型 DN，和核心一起訓練 | 0.67 → 0.09／0.46 → 0.52 | 0.019／0.032；−0.12／+0.24 | 不成立 |
| S6（[6939164635](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6939164635)） | 衍生圖（補回 LC9→DNp09）上做 BC，速度仍只讀 DNp09 | 0.77 → 0.40／0.42 → 0.58 | 0.0005／0.0004；+0.40／+0.50 | 不成立 |
| S5（[6940122400](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6940122400)） | 正式核心凍結，只擬合 DN 群體的速度讀出 | 0.77 → 0.90／0.42 → 0.42 | 0.042／0.072；+0.26／−0.17 | 不成立 |
| S7（[6940930007](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6940930007)） | S6 的核心凍結，只擬合 DNp09 的速度讀出 | 0.77 → 0.65／0.42 → 0.59 | 0.066／0.043；−0.10／−0.08 | 不成立 |
| M1 s128／s32（[6906691604](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6906691604)、[6920334616](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6920334616)） | BC 加遮蔽回合，遮蔽時維持遮蔽前的動作 | 遮蔽 0.00 → 0.22／0.07 → 0.26（s128）；0.02／0.02（s32） | — | 不成立（M1 自己的五項規則） |

- S5 的 seed 0 是 formal-v1 之後第一次有單一 seed 在標準評估的 normal 超過 0.8（0.90），但 seed 1 沒有改善、碰撞 0.47，noise 兩個 seed 都是 0。能讀出來的訊號在不同 seed 之間不一致。
- S4、S6 都用了 S3 的期望指令標籤和 S2 的梯度平衡，而「只有這兩項」的對照沒有跑過，所以這兩輪 normal 的變化不能歸給各自的改動。S5、S7 的讀出也擬合到期望指令（核心不訓練，沒有梯度平衡）。M1 沒有用這兩項，只改了資料（一半遮蔽回合）、遮蔽時的 teacher 和序列長度；它的 normal 變化可能來自 normal 資料減半，沒有查。
- **R2（保留左右側的隨機接線，#113，[6945720535](https://github.com/Nues0913/Drosophila_Tracking_System/issues/113#issuecomment-6945720535)，PM 驗收 [6946063362](https://github.com/Nues0913/Drosophila_Tracking_System/issues/113#issuecomment-6946063362)）**：主要判讀照規則成立（normal VO-late − R2 RS = +0.516 [+0.418, +0.634]），但 R2 的 RS 五個 seed、四個情境的成功率全部是 0，對照組本身退化，不能說明連接體特定的接線比較好。

### 零訓練的預檢和探針（核心不訓練；每格 20 回合，除非註明）
- **速度從哪一層讀得出來**（正式 VO-late 凍結，只換速度的線性讀出，curved normal；模型 0.80／0.35，換成 teacher 的速度 1.00／0.95）：感覺碼 0.95／0.80、LC 五型 1.00／0.50、DN 群體 1.00／0.90（加上 1e-4 的變異下限後 1.00／0.40）、只接回 LC9 的 DNp09 1.00／0.00（seed 1 的轉向被 DNp09 的下游弄壞）。
- **氣味帶了什麼**：目前的 plume（puff 留存 20 s、衰減 10 s）是目標走過的軌跡，左右濃度差和目標方位的相關只有 0.17–0.25，濃度幾乎不帶距離的資訊（R² ≤ 0.06）。衰減改短到 1.5 s，wander 遮蔽的氣味轉向從 0.52 到 0.87（環境的改動，沒有採用）。
- **遮蔽時的腳本策略**（100 回合；curved／wander）：維持遮蔽前的動作 0.59／0.07；氣味轉向、速度維持 0.55／0.52；氣味轉向、特權速度 0.87／0.70。
- **轉向的讀出**（wander，teacher 的速度）：
  - R1 的標籤（切換）下，重新擬合正式的 6 個轉向特徵或整個感覺碼，遮蔽只有 0.05–0.10。離線的解碼 R² 0.85–0.98 在閉環裡不成立。
  - 視覺讀出＋氣味項直接相加：遮蔽 0.65、normal 0.85；加可見度閘門：0.70、1.00。
  - 標籤改成相加的形式、只擬合正式的 6 個特徵：遮蔽 0.05／0.00 → 0.40／0.45，normal 0.95／0.35 → 0.55／0.55。速度改用 actor 自己的（固定 0.17 m/s）：遮蔽 0.00 → 0.15／0.20。
- 證據：`research_20261117_probes.tar.gz`、`research_20261117_probes2.tar.gz`（[6925466534](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6925466534)、[6940738752](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6940738752)）。

### 需要使用者決定的變動（11-17 時都還沒有做；使用者的決定和 T1／S8 的結果見上方 2026-11-18 一節）
1. **子圖抽取**：門檻改成對「細胞型到 DN 的突觸總和」套用，把 LC9 整群和它到 DNp09 的邊補回正式子圖（4,502 → 4,613 節點）。S6、S7 顯示光補圖還不夠，要配合速度讀出的訓練方式。
2. **讀出加可見度閘門**（轉向在看不到目標時切到氣味支路）。
3. **後續研究是否把 wander 遮蔽列為主要終點**（curved 仍報告）。formal-v1 已有 wander 次要評估，不更動既有終點和判讀。
4. **plume 的衰減**（改短會讓氣味更指向目標）。

### 工程
- PR #109（VO-late 支路合併）、#115（前端 CPU 加速約 36%，輸出逐位元相同）已合併；#117（部署時重複使用核心運算子；ES 連續改綁參數的問題依 review 修正）也已合併。三者疊加在 2060 上 VO-late 的推論 p50 約 20 → 13 ms（本機量測，4070 的正式重測在 #110）。
- **G0e（#111，PR #118，已合併）**：Gazebo 目標對比低約 10% 的原因是相機的取樣核（policy node 用完整高斯核，2D 是 7 條子射線的平均），和 Gazebo 世界無關。改用和 2D 相同的子射線之後，目標對比 Gazebo／2D 是 1.002／1.001（原本 0.89–0.91），G1 範圍 48/48 和 2D 相同。舊的核保留為 `gaussian`（`--vision-kernel gaussian` 可重現 G2）。
- **G0f（#116，PR #120，已合併）**：預設關閉的 contact sensor 核對。撞牆時 contact 和幾何條件成立相差 ≤ 1 個 pose 週期（只是取樣延遲），G0d 之後可視為和 2D 等價。撞目標時旗標晚 5.7–57 s，9 次裡 2 次沒有觸發：agent 0.18 m＋目標 0.15 m 的外形剛好等於判定距離 0.33 m，物理把中心擋在 0.329–0.335 m。修正（目標的 collision 外形縮到判定距離內）是 G0g（#121），排在 #110 之後。

## 2026-11-16 formal-v1 之後的研究與工程（#35）

使用者 11-16 授權 PM 全力推進所有未解的問題（#35 [6902162092](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6902162092)）。以下都是 formal-v1 之後的探索，標為未登記；formal-v1 的結論不變。證據包在 release `research-35`。

- **遮蔽時的記憶（E，[#35 6894989078](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6894989078)）**：
  - 2D 的遮蔽是整個相機變黑。正式 actor 在遮蔽後 0.4–0.9 s（V）或 0.6–3.5 s（VO-late）就收斂到每個 actor 固定的盲轉速，和遮蔽前目標在哪裡無關。
  - 視覺通路的時間常數沒有變長（V 最長 0.08 s）；長的時間常數只在嗅覺通路。圖裡沒有中央複合體和 TuBu（萃取規則排除）。主因是 BC 只用 normal 資料。
- **速度讀出的零訓練預檢**（A，[#35 6901989849](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6901989849)）：teacher 想要的速度平均 0.25–0.41、標準差 0.13–0.23 m/s，模型固定 0.17 m/s；DNp09 在回合內幾乎沒有變化（標準差約 1e-5）；DN 群體只有部分 seed 帶有速度資訊。DAgger 的速度標籤從模型上一步的動作出發、每步最多差 0.05 m/s，而模型觀測不到自己的動作。
- **S2：BC 時動態平衡速度和轉向的梯度**（[#35 6906921113](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6906921113)，判定不成立）：
  - 每一步把速度誤差往上游的梯度調到轉向的 0.3 倍（實際放大 300–850 倍），速度在回合內仍然完全沒有變化，讀出權重也幾乎沒變。
  - 原因是速度只從 DNp09 讀，而 DNp09 的輸入幾乎是常數：正式 VO-late 的 DNp09 電壓標準差 6e-6，轉向的 DNa02 是 0.012；嗅覺支路 DNp09 的上游完全沒有活躍過。所以問題在讀出的位置，不在梯度大小。
- **S3：速度標籤改用 teacher 想要的動作**（#112）：S2 之後暫緩。只改標籤，速度的讀出路徑一樣卡死；之後和新的讀出一起測。
- **S4：速度改從 DN 群體讀出**（[#35 6907680785](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6907680785) 登記；結果不成立，見 2026-11-17 一節）：
  - 零訓練預檢：核心的輸入（89 維感覺碼）可以預測 teacher 想要的速度（留出回合的 R² 0.51–0.88），進到中央網路和 DN 之後就不穩定或消失。問題是路由，不是輸入裡沒有。
  - 做法：速度讀出加 134 種兩側都有的 DN（268 列，梯度不切斷），每一輪用當下的核心重新標準化（不改輸出），配 S2 的梯度平衡和 S3 的標籤。判讀沿用 S2 的五項。
- **M1：BC 加入遮蔽回合，遮蔽時用只靠記憶的 teacher**（[#35 6902356202](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6902356202)）：零訓練預檢，curved occlusion 100 回合，遮蔽時維持最後動作 0.59、直走 0.00（碰撞 0.71）、氣味轉向＋特權速度 0.87、氣味轉向＋維持速度 0.55。訓練結果（[#35 6906691604](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6906691604)，判定不成立）：
  - 一半的 BC 資料換成遮蔽回合、TBPTT 128 步。遮蔽成功率兩個 seed 都提高（0.00 → 0.22、0.07 → 0.26），遮蔽時的轉向和遮蔽前的轉向相關（r +0.73、+0.51）。
  - 但不同回合之間的盲轉速差異（0.015–0.017 rad/s）沒到事先定的 0.02，normal 的碰撞也變多（0.00 → 0.16、0.03 → 0.09），seed 0 的 normal 成功率掉 0.17。
- **RS 退化的原因**（[#35 6902649027](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6902649027)）：不是視覺送不到（RS 圖反而更近），而是保留度數的隨機重接打亂了左右對應。加權的線性傳遞裡，連接體的 DNa02 左右差對左右眼刺激方向相反（+0.196／−0.240），5 張隨機圖只有 2 張保留、而且弱 2–5 倍。只在同側的邊之間交換的隨機重接，5 張全部保留方向性（強度是連接體的 15%–58%），所以事後的隨機對照 R2 改用這個規則、每個 seed 一張圖（#113，4060；結果見 2026-11-17 一節：也退化）。
- **G3 延遲**：推論的大頭是 flyvis 前端（2060 上 CPU 單執行緒約 21 ms、GPU 約 4 ms）。PR #109 把 VO-late 的兩個支路合併成一次計算（GPU 上核心 5.2 → 2.7 ms，閉環 8 回合逐位元相同）；#110 在 4070 上實測 policy 改用 GPU（@cweiting731；#109、#115、#117 都已合併，CPU、GPU 用 main `e5a166b` 一次量）。
- **Gazebo 的目標對比**（#111，G0e，@cweiting731）：已完成（PR #118），見 2026-11-17 一節和揭露第 26 點。

## 2026-11-14 Gazebo：G2 結果（#88）

- **交付和收件**：
  - 4060（MiLong）11-11 交件，`gazebo_g2_4060.tar.gz`（`561ea8f4…`，[#88 6798991965](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6798991965)）；4070（cweiting）11-14 交件，`gazebo_g2_4070.tar.gz`（`f2c26c6e…`，[#88 6869576802](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6869576802)）。兩個包都在 release `formal-v1`。
  - PM 的收件檢查都通過（[#88 6806858753](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6806858753)、[6872441153](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6872441153)）：雜湊相符、區塊和分配一致，用凍結的 kit v2 重算每個區塊。
- **執行狀況**（都照實揭露，不重跑）：
  - 主機中斷時，照事先的例外規則重跑一次（[#88 6743008783](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6743008783)）：4070 的 curved 1（主機使用者中止）、curved 2（Windows Update 重開機），4060 的 curved 4、wander 3（主機關機）。未完成的嘗試不進彙總。
  - 4060 的 wander seed 2，最後三格 RS 和主機上一段無關的 GPU 負載重疊（11-10 15:31–16:18Z），頻率和計時都通過，不重跑（Decision 6806858753）。延遲見揭露第 17 點。
  - 4070 的 wander seed 1：V normal episode 7 invalid（coverage 0.782），評估器把它算成失敗、留在分母裡，所以保留，不重跑（[#88 Decision 6843506724](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6843506724)）。這一格最多被壓低 0.05，不在任何登記的比較裡。原因沒有確認；主機使用者確認那段時間沒有人在前景使用。
  - 4070 的 posthoc 區塊中途照主機使用者的要求暫停約 4.5 小時，已完成的區塊沒有重跑。4070 在 G2 期間沒有記錄 GPU／CPU 的監控資料。
- **彙總**：照事先登記的方式，分析 checkout 是 main `04f1708`。PM 的分析包是 release `formal-v1` 的 `pm_g2_analysis.tar.gz`（`f716e708…`），重跑一次到別的目錄，93 個輸出檔逐檔相同。
- **G3**（參考主機 4070，curved normal）：五個變體都**未達成**。

| 變體 | 成功率 [95% CI] | 4070 的 p95 |
| --- | --- | --- |
| V | 0.450 [0.380, 0.540] | 49.3 ms |
| VO | 0.400 [0.300, 0.510] | 49.0 ms |
| VO-late | 0.440 [0.310, 0.620] | **51.5 ms** |
| GRU | 0.000 | 47.1 ms |
| RS | 0.000 | **50.5 ms** |
| 事後 GRU（未登記） | 0.250 [0.160, 0.340] | 47.5 ms |

- **已登記的比較**（curved、wander 各 18 項）：**判讀和 2D 完全相同**。
  - 遮蔽優越性都不成立：curved 的 VO − V 是 +0.030 [−0.090, +0.150]，VO-late − V 是 +0.000 [−0.060, +0.060]；wander 是 +0.010 [−0.030, +0.060]、−0.010 [−0.030, +0.000]。
  - 對 GRU、RS 的不劣性名目上都成立，CI 下界最低 curved +0.030、wander +0.000。兩個對照組在 Gazebo 也退化（成功率 0–0.02），和 2D 一樣不能當成證據（揭露第 16 點）。
  - **只用 4070 的 seed**（[#88 6747441878](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6747441878)）：36 項的判讀都沒有改變，沒有比較要標「受主機延遲影響」。
  - **保守重算**（[#88 6763512170](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6763512170)）：curved 有 8 回合、wander 有 32 回合改成失敗，36 項的判讀都沒有改變，沒有比較要標「受漏記撞牆影響」。
- **和 2D 逐回合配對**（同一個 actor 和環境 seed）：
  - 每格成功與否相同的比例：GRU、RS 是 98%–100%；V、VO、VO-late 是 79%–100%，兩種運動各有 5／20 格低於 91%，最低是 curved V noise（79%，Gazebo 0.25 對 2D 0.14）。依事先固定的規則（[#88 6788230860](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6788230860)），這些格子不能只用目標對比解釋。
  - 36 項登記比較和 2D 配對回合的差距都 ≤ 0.08（最大 0.06），所以都註明「可能受目標對比影響」。
  - 漏記撞牆，和事先的預估對照（curved 的 V、VO、VO-late）：2D 撞到、Gazebo 沒撞到的有 107 回合（預估 130）；Gazebo 撞到、2D 沒撞到的有 60 回合。軌跡分岔以後就不能一一對應，只能當成粗略的對照。
- **事後 GRU**（未登記，描述性）：Gazebo 的 curved 成功率 normal 0.250、occlusion 0.010、noise 0.130、conflict 0.300。和 2D 判讀不同的有 4 項：curved 的 VO conflict（Gazebo 不成立），wander 的 VO normal、VO conflict、VO-late conflict（Gazebo 成立），區間都很寬。事後 GRU 在 Gazebo 的 conflict，curved 是 0.30（2D 配對 0.24），wander 是 0.08（2D 配對 0.21），原因沒有查。
- **延遲**：4070 的推論時間每格是 12.8–19.9 ms，sensor→command p50 是 33.0–42.6 ms，連續分布。4060 的 p50 分成兩群，分界是推論時間約 25 ms，見揭露第 17 點。
- **G0d 已合併**（PR #107，main `8e19c9b`，11-14）。PM 用 `8e19c9b` 重建本機的 `flytrack-ros:0.1`（`71b5992b`），在 2060 上重錄 V1 選的 11 個回合，含 rosbag（release `formal-v1` 的 `gazebo_demo_g0d_2060.tar.gz`（`4c2b74d7…`））：成功與否和 2D 相同 11／11，G0d 之前的世界是 10／11。
- 完整的表在 [formal_v1_report.md](formal_v1_report.md) 第 5 節。

## 2026-11-11 Gazebo：G2 開跑、斜角撞牆與轉向倍率（#88）

- **三台主機用 G0c 世界的 G1 範圍都通過**（PM 用同一套腳本重算）：
  - 4070（[#88 6746745840](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6746745840)）：47/48 對結果相同，成功回合 Gazebo 7、2D 6，方位誤差中位數 62.1° 對 59.6°；
  - 4060（[#88 6747441878](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6747441878)）：47/48，7 對 6，61.1° 對 59.6°；
  - 2060 的結果見下方 11-10 一節。三台的行為對照幾乎一樣。
  - 4070 在 G0c 世界的 sensor→command p95：V 49.6–49.7、VO 49.4–50.0、VO-late 51.8、GRU 47.3、RS 50.8–51.0、事後 GRU 47.4–47.7 ms。每格只有 2 回合，是工程紀錄，G3 以 G2 的資料判定。
- **G2 開跑**：固定 main `a58b7b2`，用工具包 v2。4060（MiLong）從 11-09 17:48Z 開始（[#88 6747260444](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6747260444)），4070（cweiting）從 11-10 01:51Z 開始（[#88 6756668333](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6756668333)）。
- **延遲的敏感度分析**（事先登記，[#88 6747441878](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6747441878)）：
  - 4060 上 VO-late 的 sensor→command p50 是 116–119 ms，V 是 49 ms；4070 上兩者差不多（39、37 ms）。
  - 所以已登記的比較，除了照協定用 5 個 seed，另外只用 4070 跑的 seed（curved 0–2、wander 0–1）再算一次。兩種結果並列，結論不同就標「受主機延遲影響」。
- **斜角撞牆記不到碰撞**（[#88 6763512170](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6763512170)）：
  - PM 錄 Gazebo 展示時發現：V seed 0 的 curved occlusion（環境 seed 1700000002）在 2D 撞牆結束，在 Gazebo 卻從 t ≈ 31 s 開始貼著東牆滑行到回合結束。
  - 原因：碰撞的判定和 2D 相同，都是中心離牆 ≤ 0.18 m。但 Gazebo 模型的輪子前緣離中心約 0.195 m，斜著靠牆時牆先擋住輪子，中心到不了判定線。
  - 在資料出來之前登記的處理：G2 照跑；Gazebo 的碰撞率照實報告，不和 2D 比較；G2 和 F3／F4 逐回合配對；2D 撞牆、Gazebo 沒記到卻算成功的回合，另外保守重算成失敗，和原始結果並列。
  - G2 之前用 2D 重播所做的預估（[#88 6764426046](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6764426046)）：curved 的 V、VO、VO-late，在 G2 會跑到的回合（每格第 0–19 回合）裡，265 次撞牆有 130 次是斜角，主要在 noise（94/120）；主要比較所在的 occlusion 只有 26/126。
- **轉向比指令快 12%**（[#88 6764426046](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6764426046)）：輪子在內側邊緣接觸地面，有效輪距是 0.285 m，但差速驅動設定的是 0.32 m，所以實際角速度是指令的 1.123 倍。G2 照跑，報告照實揭露。
- **修正 G0d**：輪子和後輪移到 0.18 m 以內，`wheel_separation` 改成內側邊緣的間距。PM 在 2060 上用固定指令的駕駛節點驗證：直線開向牆的 8 回合都在第一次碰牆就記到碰撞，轉向倍率變成 1.000。G2 固定在 `a58b7b2`，不受影響；G0d 在 G2 結束後合併。

## 2026-11-10 探索實驗 D3：速度讀出加上校準過、切斷梯度的 DN 列（#104，判定不成立）

- **背景**：A1 發現 VO、VO-late 的前進速度是常數（見下方 A1 一節）。S1 把速度輸入放大，梯度也跟著放大，轉向因此壞掉。使用者 11-10 決定「如果能改善的話那就改」。
- **做法**（[#104](https://github.com/Nues0913/Drosophila_Tracking_System/issues/104)，設計和判讀規則都在看資料之前固定）：
  - 在 structured readout 的速度 latent 上，加 DNp09、DNa01、DNa02 三列，每列是該型別所有節點的平均 rate；VO-late 的兩個支路各一組，共 +6 個參數。
  - 每列先校準：(S − center) × scale。常數在 BC 第 0 輪，取 teacher 訓練回合 60 步之後的平均和 1／標準差。
  - 這些列的輸入先 detach，速度誤差不會傳回上游。這是為了避開 S1 的失敗原因。權重初值是 0。
- **範圍**：從 formal-v1 的 pretrain 跑 BC，VO-late 的 seed 0、1；新的評估區段 `environment_seed` 1,850,000,000；curved，四個情境，每格 100 回合。基準是同 seed 的 formal-v1 VO-late actor，在同一個區段上重新評估。V 只用來報告，判讀不成立之後就沒有跑。
- **執行**：PM 在使用者的 RTX 2060 上執行（使用者同意），用的是未合併的 PR #105 的 head `baab1ce`。PR 已關閉，分支保留。
- **交付**：release `formal-v1` 的 `d3_forward_rows.tar.gz`，SHA-256 `95a0e341a2cceffcdf13408722aa6894e3d5da3c90ea97bdd86ca9c7bf3cef1f`。
- **結果**：[#104 6761197648](https://github.com/Nues0913/Drosophila_Tracking_System/issues/104#issuecomment-6761197648)。

| seed | normal 成功率（基準 → D3，需 +0.10） | occlusion（需 ≥ −0.05） | 回合內速度 std 中位數（需 ≥ 0.01 m/s） | normal 方位誤差（需 ≤ +5°） |
| --- | --- | --- | --- | --- |
| 0 | 0.70 → 0.72 | 0.00 → 0.25 | 0.0101 | 10.6° → 13.4° |
| 1 | 0.45 → 0.02 | 0.16 → 0.00 | 0.0093 | 44.8° → 53.8° |

- **判讀**：兩個 seed 都要四項全部達到。seed 0 的 normal 沒有達到，seed 1 四項都沒有，所以判定不成立。formal-v2 不觸發。
- **事後診斷**（不改變判讀）：
  - **速度整體變快，沒有跟著距離調整**：seed 1 的 v 中位數是 0.234 m/s，seed 0 有幾個回合整段約 0.30 m/s。回合內的 std 只有 0.009–0.010。
  - **失敗主要是撞上目標**：normal 的碰撞率，seed 0 從 0.01 升到 0.23，seed 1 從 0.05 升到 0.46。seed 0 在距離內的比例其實從 0.80 升到 0.94，但被碰撞抵銷。這和 A1 的反事實測試一致：速度固定在 0.21 m/s 時，多出來的碰撞也都是撞上目標。
  - **校準在評估時已經失效**：校準後的特徵應該是平均 0、標準差 1。重播候選 actor 的 normal 回合時，seed 0 視覺支路的 DNa02 是 437，seed 1 視覺支路的 DNp09 是 31，嗅覺支路的 DNa02，seed 0 是 137，seed 1 是 242。DN rate 本身的 std 只有 1e-5 到 1e-3，而校準只在第 0 輪做一次；之後 BC 繼續訓練核心網路，DAgger 之後的狀態分布也和 teacher 不同。
  - **預檢量的目標不對**：預檢用 formal-v1 自己的 rollout 預測距離，R² 很高。但第 0 輪的閉式解顯示，加上新列以後，對 teacher 速度的預測沒有變好（留出回合的相關係數：seed 0 從 0.108 變 0.096，seed 1 從 0.608 變 0.547）。
  - 這次沒有跑「同一台主機、不加新列」的對照，所以 seed 1 的方位誤差多了 9°，不能完全歸因於新列。
- **處理**：
  - 記為負面結果。不在同樣的 seed 和這個評估區段上改試別的變體，這是 S1 的教訓。
  - 速度讀出留在下方的科學限制第 18 點。之後要不要再研究，連同 #35 由使用者決定。
  - formal-v1 已登記的結果不變。

## 2026-11-10 G0c 合併、2060 重跑 G1 範圍、V1 驗收（#88、#97）

- **G0c 已合併**（PR #102，main `a58b7b2`）：cweiting 用證據包獨立重算過驗收數字。PM 的檢查是 `python/tests` 586 passed、1 skipped；colcon 23 tests，0 errors。
- **2060 用 G0c 世界重跑 G1 範圍**（release `formal-v1` 的 `gazebo_g1c_2060.tar.gz`（SHA-256 `3e330bd0c17ddcda44354e89810e709704404215e4de53c902eb9955e1ab5b3d`））：
  - 48 回合全部 valid；每格 ≥ 18 Hz，計時 100%。
  - 和 2D 同一個 actor、同一個環境 seed 的回合配對（`g1_vs_2d.py`，開跑條件在看行為之前就固定）：48 對裡有 47 對結果相同；成功回合 Gazebo 7、2D 6；方位誤差中位數 62.0° 對 59.6°。判定 GO。
  - 這段時間主機同時有其他使用者程式在跑，推論從約 28 ms 變成約 38 ms，所以延遲數字不能用。同一天、條件相同的探針裡，舊世界和 G0c 的延遲差不多，所以不是 G0c 造成的。
- **G2 重新開跑的條件**（[#88 6745099162](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6745099162)）：
  - 固定 `a58b7b2`，用工具包 v2；
  - 4070、4060 各自跑 G1 範圍，valid、頻率、計時和行為對照都通過，就直接開跑自己的區塊；
  - 區塊、分配、停止規則和 G3 延遲的判定都不變。
- **G2 的彙總設定和延遲腳本已合併**（PR #101，main `a411831`）：計時比例的定義固定為 `causal_clock_valid` ÷ trace 筆數。
- **V1 驗收並結案**（[#97 6745120560](https://github.com/Nues0913/Drosophila_Tracking_System/issues/97#issuecomment-6745120560)）：
  - 交付是 release `formal-v1` 的 `formal_v1_demo_2d.tar.gz`；
  - PM 重算的回合選取 6 格全部一致，重播和 F3 11/11 一致。

## 2026-11-09 G0b 合併、兩台主機的 G1 範圍、G2 派工與暫停（#88）

- **G2 暫停**（[#88 Decision 6744333249](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6744333249)）：
  - **發現**：PM 錄 rosbag 做開環對照。在同一 pose 下，Gazebo 的 hex-eye 輸入（policy 實際收到的）和 2D ray cast 的逐幀相關中位數是 −0.97。
  - **原因**：舊世界的天空和被照亮的地板很亮，牆和目標反而暗，和 2D 的暗背景、亮目標相反。
  - 氣味兩邊一致。G1 範圍的 52 回合 success 全部是 0，同一批 actor 在 2D 是 0.39–0.48。
  - **修正 G0c**（PR #102，已合併）：平光材質，emissive 用 sRGB 解碼後的值。驗收（看行為之前就固定）通過：左右眼 r 中位數 0.979／0.968，MAD 0.013／0.011。同樣兩個環境 seed 的行為，逐回合和 2D 幾乎相同。
  - **G0c 合併之後**：4070、4060 重跑 G1 範圍，再重新發 G2。細節見 PR #102 的 `docs/gazebo_consistency.md`。

- **G0b**（PR #99，main `4a537af`）：
  - policy 預設用 1 個 torch 執行緒；
  - evaluator 自己訂閱 SensorTiming，補齊計時。
  - 控制路徑的數學、感測頻率、世界和材質都不變。
- **G1 範圍**（seed 0，normal 和 occlusion，curved 和 wander，每格 2 回合，`--cpu`）：
  - **2060**：52/52 valid，19.6–19.9 Hz，計時 100%，sensor→command p95 是 55–130 ms。
  - **4070**：52/52 valid，19.5–19.9 Hz，計時 100%，p95 是 48–53.5 ms（G0c 之前的世界）。只有 GRU 和事後 GRU 低於 50 ms。
  - 兩台主機的 G1 範圍，valid、頻率和計時都通過。但行為因為輸入明暗相反沒有意義（52 回合 success 全部是 0），G0c 之後要重跑。
  - 2060 的延遲高很多，而且使用者規定重度測試交給 teammate 的主機，所以 2060 不用於 G2。
  - 每格只有 2 回合，這些都是工程紀錄。逐台的表和頻率定義見 [ros_execution.md](ros_execution.md) 的 2026-11-09 一節。
- **G2 派工**（已暫停，見上；[#88 Decision 6743008783](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88#issuecomment-6743008783)）：
  - 固定 main `4a537af`。
  - 一個區塊是「一個訓練 seed × 一種目標運動」：4070 跑 curved seed 0–2、wander seed 0–1；4060 先跑 G1 範圍，再跑其餘區塊。兩台各約 27 小時。
  - G3 延遲的判定在資料出來之前就固定：參考主機是 4070，取 curved normal 合併後的 p95。
  - 彙總設定和延遲腳本在 PR #101。
- **V1**（#97）：2D 展示素材的工具已合併（PR #100，main `2fb98a4`）。正式交付已在 11-10 驗收，見上方一節。

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
    - G0f（#116，PR #120，formal-v1 之後）用預設關閉的 contact sensor 核對：撞牆的判定和實際接觸只差取樣延遲（≤ 1 個 pose 週期再加 20 Hz 的 tick）。
    - **撞目標共 9 次**：
      - 追目標時撞上 8 次，其中 **6 次旗標晚 5.7–57 s，2 次完全沒有觸發**；
      - 另 1 次是撞牆途中正面撞到目標，旗標只晚 0.04 s（拆法依 #119 的 review）。
    - 原因是 agent 和目標的外形半徑和（0.18＋0.15 m）剛好等於判定距離 0.33 m，物理把兩個中心擋在判定線附近。
    - G2 的保守重算已經把「2D 撞到、Gazebo 沒記到又算成功」的回合都改成失敗（不分撞牆或撞目標），判讀不變。只有「Gazebo 撞目標、而且沒記到」的回合，G2 沒有逐步距離，量不出來。
    - **修正是 G0g**（#121，PR #122，main `daa6e5f`）：目標的 collision 外形縮到 0.145 m，視覺外形不變。12 回合的旗標都在第一次 contact 的 ±0.022 s 內，G1 範圍 48/48 和 2D 相同。G2 不重跑。
12. **時鐘與延遲**：行為指標用模擬時鐘，推論耗時用 wall time。sensor-to-command 的因果計時**已有契約與實作**：`SensorTiming.msg`／`ControlTrace.msg` 定義同一台主機上 `CLOCK_MONOTONIC` 的 source → ingress → forward → received → control → published 時間線，`flytrack_eval.latency.CausalLatency` 檢查時鐘域、因果順序和覆蓋率，有 `test_latency.py` 測試。2026-10-19 的 CPU／GPU 實測均未達 50 ms，詳見第 9 點與 [ros_execution.md](ros_execution.md)。端到端 p95 必須由各筆命令最舊的有效 contributing sample 至命令發布的完整差值計算；各階段 p95、不同 sensor 的階段最大值不可拼成一條因果路徑。
13. **續訓**：同一階段中斷後可以完整接續（權重、optimizer、RNG、環境重播）；跨階段以 `init_artifact` 暖啟動，recovery 綁定上游 actor 的內容（#14）。改了程式碼或設定之後，舊的 checkpoint 不能接續；#36 合併後，舊的 pretrain／BC artifact 都要重跑。
14. **號誌與 NT 信心**（#27）：
    - consensus 讀取不使用數值信心，`nt_confidence_min` 不起作用。
    - 號誌不明的 88 個候選細胞被移除，其中包括已知的單胺神經元，因為它們沒有固定極性。seed 到 DN 的可達性和最短跳數都不受影響，但調質連接不在模型中。
    - Glu 一律為負號、每個細胞的所有出邊同號，是模型假設。觸角葉的 GluClα 支持這個方向，但沒有逐邊驗證。
15. **LC 輸入的功能**（#27）：LC10a 有追蹤和轉向的行為證據；LC9、LC11 偵測小物體；LPLC2 偵測 looming；LC10d 的迴避功能還是候選。報告稱它們為「視覺物體特徵輸入」，不寫成追蹤通道。
16. **對照組退化**（#85）：正式矩陣的 GRU 因共用的 ×100 ORN 增益讓輸入瓶頸飽和，看不到視覺，屬於工程缺陷；RS 依固定標準不是缺陷，五個 seed 共用一張隨機圖。原本寫「隨機接線讓視覺送不到轉向輸出」，2026-11-16 量過之後更正（#35 6902649027）：RS 圖裡視覺到轉向 DN 的路徑反而更短（1–2 跳，連接體 2–3 跳）；問題是保留度數的隨機重接打亂了左右對應。帶正負號、依突觸數加權的線性傳遞裡，連接體的 DNa02 左右差對左右眼刺激方向相反（+0.196／−0.240），同樣規則產生的 5 張隨機圖（seed 0–4，正式 RS 用 seed 0）只有 2 張保留方向性，而且弱 2–5 倍（線性近似，忽略整流和學到的參數）。formal-v1 之後的 R2（#113）改用保留左右側的隨機接線、每個 seed 一張圖，訓練之後四個情境的成功率也全部是 0，所以左右對應不是 RS 退化的唯一原因。對兩者的不劣性都不能當成連接體結構的證據；修正後追加的 GRU 一律標成「未登記」。事後 GRU（#86／#87）的結果顯示，已登記的「對 GRU 不劣」在 curved 的 VO noise，以及 wander 的 normal、conflict（VO 和 VO-late），只是因為 GRU 退化才成立。
17. **G3 未達成**：2D 的 normal 成功率最高 0.484（curved，VO-late），低於登記門檻 0.8。Gazebo（G2）也未達成：curved normal 最高是 V 的 0.450；參考主機 4070 的 p95，VO-late 51.5 ms、RS 50.5 ms 超過 50 ms，V、VO、GRU、事後 GRU 是 47.1–49.3 ms（上方 2026-11-14 一節）。
    - 4060 的延遲只報告，不用來判定。sensor→command p50 分成 42.5–59.6 ms 和 107.7–126.6 ms 兩群，另有一格是 79.9 ms；分界是推論時間：每格推論時間的中位數 ≤ 24.3 ms 的都在低的那一群，≥ 25.0 ms 的除了一格（curved RS seed 3 noise，79.9 ms）都在高的那一群，大約多一個控制週期（50 ms）。VO-late 每格推論時間的中位數是 26.2–31.5 ms，所以在 4060 上每一格都在高的那一群。
    - 4060 的 wander seed 2，最後三格 RS 和主機上一段無關的 GPU 負載重疊（11-10 15:31–16:18Z，平均 34%、最高 64%）。noise、conflict 的推論時間是 28.6、27.8 ms，比沒有負載的 RS 格子都高（≤ 25.2 ms），延遲變高可以歸因於負載把推論時間推過門檻；occlusion（25.1 ms）不能確定。不重跑（#88 Decision 6806858753；這一點修正了那則 Decision「拆不出來」的說法，決定不變）。
    - 4070 沒有分群：每格推論時間是 12.8–19.9 ms，p50 是 33.0–42.6 ms，連續分布。
    - **formal-v1 之後的延遲重測**（#110，4070，main `e5a166b`）：已含三項等價加速 #109、#115、#117，取樣核是 subrays。CPU 版六組合併 p95 是 43.4–44.9 ms，低於 50 ms；GPU 版是 52.3–53.1 ms。這是工程結果，成功率沒有改善，所以 G3 判定不變。
18. **前進速度是常數**（#90）：VO 和 VO-late 的速度只由 DNp09 讀出，DNp09 在圖上的輸入很少，所以速度固定在 0.168–0.179 m/s，低於目標速度 0.200 m/s。normal 的距離失敗和遮蔽期間直線撞牆都和這一點有關。報告不能把 G3 未達成歸因於訓練預算，也不能把 VO-late 拿掉氣味後的下降說成它用了氣味的方向。formal-v1 之後做了兩個探索實驗：
    - S1（#95）把速度讀出的輸入放大，速度開始會變，但轉向或遮蔽的表現變差；
    - D3（#104）改加切斷梯度的 DN 列，速度只是整體變快，接近目標時沒有減速，撞上目標的次數變多。

    兩個都判定無效，所以這個限制在 formal-v1 仍然成立。
19. **沒有風的輸入**：果蠅在 turbulent plume 裡主要靠風向導航，遇到氣味就逆風走（Álvarez-Salvado et al. 2018、Demir et al. 2020、Matheson et al. 2022）。本模型只有左右 ORN 的濃度，沒有風或機械感覺的輸入。所以 VO 沒有用氣味方向、正式 O 沒有用氣味追蹤，不能完全歸因於模型或訓練（[related_work.md](related_work.md) 第 2 節）。
20. **記憶的時間尺度**：正式設定裡，神經時間常數 τ 是每個 type 共用一個，初值是 50 ms。從 25 個連接體 actor 的權重讀出，每個 actor 的 type τ 中位數都還是 50 ms，最長的是 0.06–0.52 s；BC 的序列是 32 步，也就是 1.6 秒。協定的遮蔽是 30 秒，遮蔽期間的表現要在這個限制下解讀（[related_work.md](related_work.md) 第 3 節）。
21. **子圖規模**：正式 no-KC 圖有 4,502 個節點、31,682 條邊。萃取時把門檻放寬到 50 突觸，保留 LC9／LC11 和更多中央嗅覺通路，同時移除原本的 3000 節點上限（`max_neurons: null`，#55），所以超出原計畫的 500–3000 節點預算。
22. **G3 的延遲跟主機有關**：同一份程式在 G1 範圍（G0c 之前的世界），2060 的 sensor→command p95 是 55–130 ms，4070 是 48–53.5 ms；G0c 之後的延遲，以 G2 的資料為準（第 17 點）。G3 的延遲用事先固定的參考主機 4070 判定（#88 Decision 6743008783），其他主機的數字照實報告。
23. **Gazebo 的渲染改成符合 2D**（G0c，PR #102，已合併）：舊世界的視覺輸入和 2D 明暗相反，G0c 改用平光材質後才和 2D 一致。G1 範圍的行為（全部是 0）在修改前已經看過，但修改依據的是輸入分布的量測，驗收條件在看行為之前就固定了。
24. **Gazebo 斜角撞牆記不到碰撞**（#88 6763512170）：碰撞的判定和 2D 相同，都是中心離牆 ≤ 0.18 m。但 G2 世界的 agent 模型，輪子前緣離中心約 0.195 m，斜著靠牆時（車頭和牆的法線夾約 43°–89°，幾何推算）會被牆擋住，中心到不了判定線，於是貼著牆繼續走，不會記成碰撞。所以 Gazebo 的碰撞率會少算，不和 2D 比較。對成功率的可能影響，用 G2 和 F3／F4 的逐回合配對，以及保守重算來報告。G2 之前用 2D 重播所做的預估（curved 的 V、VO、VO-late）：265 次撞牆裡有 130 次是斜角，大多在 noise（94/120），occlusion 只有 26/126（#88 6764426046）。G2 的配對結果：同樣的範圍，2D 撞到、Gazebo 沒撞到的有 107 回合，Gazebo 撞到、2D 沒撞到的有 60 回合；保守重算把 curved 8 回合、wander 32 回合改成失敗，已登記比較的判讀都沒有改變。修正 G0d 已在 G2 結束後合併（PR #107，main `8e19c9b`）。
25. **G2 世界的轉向比指令快 12%**（#88 6764426046）：輪子在內側邊緣接觸地面，有效輪距是 0.285 m，但差速驅動設定的是 0.32 m，所以實際角速度是指令的 1.123 倍；2D 的運動學是理想的。G0d 一併修正，倍率變成 1.000。G2 用的是修正前的世界，不重跑；三台主機 G1 範圍的行為仍然是 47/48 和 2D 相同。
26. **Gazebo 的目標比 2D 暗約 10%**（#88 6783245712）：
    - G0c 之後，Gazebo 的目標對比仍比 2D 低 9%–11%（G0c 驗收表），相當於 2D 目標亮度 0.80–0.85，仍在訓練的隨機範圍 0.75–1.0 內。
    - 只是勉強追得上的回合，在 Gazebo 可能變成跟丟。起因是 wander normal 的環境 seed 1700000000：V、VO 在 Gazebo 跟丟；在 2D 只把目標調暗，V 在 0.8、VO 在 0.75 也會跟丟，VO-late 在這一回合不受影響。
    - 2D 敏感度分析（事後、描述性；#88 6788230860、6845484730；release `formal-v1` 的 `pm_2d_target_brightness_g2scope.tar.gz`）：G2 範圍的目標從 0.9 調到 0.82。V、VO、VO-late 共 2,400 回合，成功→失敗 74、失敗→成功 13；每格成功與否相同 91%–100%，每格成功率變化 0 到 −0.09。正式 GRU、RS 本來就退化，不受影響；事後 GRU 每格變化 +0.01 到 −0.07。變體之間的成功率差最多移動 0.08。整個範圍內，wander 受影響最多的是 VO-late（平均 −0.04），不是那一回合看到的 V、VO。
    - 判讀規則在 G2 資料之前固定：配對一致 ≥ 91% 的格子，單靠目標對比就可能造成；和 2D 相差 ≤ 0.08 的變體差異，註明「可能受目標對比影響」。
    - G2 的結果：36 項登記比較和 2D 的差距都 ≤ 0.08，都註明「可能受目標對比影響」；V、VO、VO-late 兩種運動各有 5／20 格一致低於 91%（最低 79%），這些格子不能只用目標對比解釋。
    - **原因和修正（G0e，#111，PR #118，G2 之後）**：差距來自 policy node 的相機取樣核（完整高斯核，比 2D 的 7 條子射線寬），和 Gazebo 世界無關。離線探針用 2D 場景的理想相機影像就重現了整個差距（r 0.99998）。改用和 2D 相同的子射線之後，目標對比 Gazebo／2D 是 1.002／1.001，逐幀相關 0.998／0.996。G2 是用舊的核跑的，不重跑，這一點對 G2 仍然適用；之後的 Gazebo 執行預設用新的核，`run_gazebo_matrix.py --vision-kernel gaussian` 可以重現 G2。

## 下一步

依 [#28](https://github.com/Nues0913/Drosophila_Tracking_System/issues/28) 的順序：

1. ~~#77、#20、#10、凍結與派工~~：已完成。main `35432c3` 上的 30 條訓練鏈已交付並驗收（#80、#81）。
2. ~~2D 正式評估（F3 curved、F4 wander）~~：已完成並驗收（#80、#84），結果見上方 2026-10-31 一節。
   - ~~**#85**~~：GRU 修正（PR #86）和事後重新訓練與評估（#87）都已完成，結果標為「未登記」，見上方 2026-11-04 一節。
3. ~~**協定必交的分析**（A1，#90）~~：已完成並驗收，結果見上方 2026-11-08 一節。之後並行的兩個探索實驗：BC 配方實驗 R1（[#4](https://github.com/Nues0913/Drosophila_Tracking_System/issues/4)，@MiLong9487，判讀第 3 點已在看到結果前修正為相對基準的配對差）和 BC 預算實驗 B1（[#92](https://github.com/Nues0913/Drosophila_Tracking_System/issues/92)，使用者）。A1 發現速度輸出是常數，所以 B1 先暫緩，後來關閉。改做的速度讀出校準 S1（[#95](https://github.com/Nues0913/Drosophila_Tracking_System/issues/95)）判定無效，見上方 S1 一節。R1 也判定無效，BC 配方路線到此為止；重訓變異 R0 顯示基準穩定，見上方 R1／R0 一節。D2 已結束，formal-v2 不觸發（[#28 Decision 6718154301](https://github.com/Nues0913/Drosophila_Tracking_System/issues/28#issuecomment-6718154301)）。使用者 11-10 要求再試速度讀出，D3（[#104](https://github.com/Nues0913/Drosophila_Tracking_System/issues/104)）判定不成立，見上方 D3 一節。#4 已結案；P3、PPO 和文獻上的候選解法都放在 #35，由使用者決定，不進正式矩陣。
4. ~~**Gazebo 主要比較（[#88](https://github.com/Nues0913/Drosophila_Tracking_System/issues/88)）**~~：已完成。G0 的 wander 接線已合併（PR #94，main `c339c0c`，PM 補跑 colcon test 通過）。G1 在 2060 上發現的兩個問題已由 G0b 修正（PR #99，main `4a537af`）；2060 和 4070 的 G1 範圍，valid、頻率和計時都通過（行為因為輸入明暗相反沒有意義）。完整矩陣 G2（curved、wander 各 2,000 回合，加上事後 GRU 各 400 回合）派工後暫停：2D 與 Gazebo 的一致性（原 #23）量測發現，Gazebo 的視覺輸入和 2D 明暗相反。修正 G0c 已合併（PR #102，main `a58b7b2`）；2060、4070、4060 的 G1 範圍，行為都和 2D 一致（GO），G2 已在 4070、4060 上跑完（工具包 v2），交付和收件見上方 2026-11-14 一節。已登記比較的判讀和 2D 完全相同，只用 4070 seed 的敏感度分析和漏記撞牆的保守重算都沒有改變判讀；G3 未達成。斜角撞牆和轉向倍率的修正 G0d 已在 G2 結束後合併（PR #107，main `8e19c9b`）。報告見 [formal_v1_report.md](formal_v1_report.md)。
5. 並行：#49 的一次短回放已由 #51 報告驗收結案；#52 的 trace 狀態與 reset 清除 pending 可觀測性已由 #53 合併，PM colcon 10 項全過、無 skip；既有發布／reset 行為不變。下一個效能候選是 camera hex-eye 投影的等價優化設計，尚未核准實作或追加回放；2D／Gazebo 一致性（原 #23）已併入 #88，仍由 @cweiting731 處理。#42 已由 #44 結案。
6. ~~Gazebo 主要比較（#88）→ 報告與展示~~：報告是 [formal_v1_report.md](formal_v1_report.md)。#22 已結案，延遲量測由 #88 接手。2D 的展示素材已由 V1（#97）交付並驗收。Gazebo 的展示和 rosbag：G0d 合併後重錄 11 個回合並交付（release `formal-v1` 的 `gazebo_demo_g0d_2060.tar.gz`（`4c2b74d7…`））。3D 影片 V2（#123）在 G0e、G0g 之後的 main 上重播同樣的 11 個回合，交付在 release `research-35`（2026-11-18 一節）。

7. **formal-v1 之後的研究（[#35](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35)）**：使用者 11-16 授權 PM 全力推進，結果和結論見上方 2026-11-17、2026-11-16 兩節。速度（S2、S4–S7）和遮蔽記憶（M1）的訓練實驗都不成立；找到的可能原因是 DNp09 的輸入被子圖門檻濾掉（結構上確認），以及氣味需要「看不見目標時切換」而晚期融合的讀出只能相加（零訓練探針支持，還不能認定是唯一原因）。R2（#113）主要判讀名目上成立，但對照組退化。使用者 11-17 決定做其中三項：T1（可見度閘門）和 S8（衍生圖加閘門），都照登記的規則判定不成立，見 2026-11-18 一節。延遲重測（#110）、G0g（#121）和 Gazebo 3D 展示影片 V2（#123）都已完成並驗收。S3（#112）以「未執行」關閉。

舊版的逐日診斷紀錄保留在 [dn/](dn/)、[fixes_2026-10-17.md](fixes_2026-10-17.md) 和 [training_time.md](training_time.md)，這些是歷史紀錄，不代表目前狀態。
