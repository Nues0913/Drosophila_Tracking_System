# v2 世界：`v1compat`（WP2a，#132）

> - 程式：`python/flytrack_v2/world/`（`vision.py`、`v1compat.py`、`draws.py`、`rng.py`、`kernels.py`）
> - 測試：`python/tests/v2/test_v2_world_vision.py`、`test_v2_world_v1compat.py`
> - bench：`scripts/v2/bench_world.py`
> - 介面：`docs/v2/interfaces.md`（M0 凍結）；設計：計劃書 3.3

## 1. 範圍

`V1CompatWorld` 把 v1 的 `FlyTrackEnv` 改成整批在 GPU 上步進的 `BatchedWorld`。

- **和 v1 一樣**：
  - 場地、agent 和目標的半徑、目標速度；
  - 目標運動 curved、wander；
  - 四種情境；
  - puff trail plume；
  - 感測時序和雜訊；
  - 碰撞、reward、成功判定。
- **參數來源**：建構時用同樣的選項建一個 v1 `FlyTrackEnv`，從它讀取參數。所以解析方式、預設值和 reward 的 survival 上界都直接是 v1 的。
- **沒有風**：`wind` 觀測和 valid 的風旗標都是 0。風和 filament plume 屬於 WP2b（#133）。
- **兩個和 v1 不同的預設**（#161 的 Decision）：
  - `reward="track_v2"`（D3）：計劃書 3.8 的 PPO reward，沒有單獨的速度項。
    - 追蹤中（2 m 內、45° 內）每步 +1；
    - −0.3 · min(|d − 1 m|, 3 m)；
    - 碰撞 −10；
    - −0.05 · ‖Δa‖（動作變化的歐氏範數）。
    - `reward="v1"` 是 `FlyTrackEnv` 的 reward，只給 parity 測試用。
  - `occlusion_view="hide_target"`（C2）：遮蔽時影像照常 render，只是看不到目標。
    - `occlusion_view="blackout"` 是 v1 的全黑影像，給 parity 測試，以及用全黑遮蔽訓練出來的 v1 actor 用。
    - 兩種模式下 `valid` 都維持 1：`valid = 0` 只代表感測資料遺失（noise 情境的 dropout），所以模型沒辦法從 valid 讀出「現在是遮蔽」。
- **v1 有、這裡沒做的選項**：遇到會直接報錯，不會默默忽略。
  - 障礙物；
  - Gaussian plume；
  - linear、stationary 目標；
  - 固定的起始姿態；
  - `occlusion_odor_steering`（teacher，屬於 #132 的第 3 部分）；
  - `sensing: false`。

## 2. 一個控制步

照 v1 的迴圈，但每一段都整批算完：

1. **時鐘和運動**：在每個 `neural_dt` 邊界，以及每個視覺、氣味取樣時間切子步，用的是 v1 的 float64 時鐘算術。
   - 位移照 v1 的型別計算：numpy 2 會把 `omega * dt`、`dt * v` 算成 float32，`v / omega` 也是 float32 除法。
   - 這一段先假設沒有碰撞，把每個子步都算出來。
2. **目標和碰撞**：算出每個子步時間的目標位置，找出第一次碰撞。之後的子步都不算，等於 v1 的 `break`。
3. **plume**：一個 tick 一個 tick 前進（integration_dt 0.01 s）。氣味在最後一個有效取樣的 tick 讀。
4. **視覺**：每個控制步最多 2 次取樣。每隻眼睛只 render 最後一次有效的取樣；更早的取樣在 v1 也會在被讀到之前就被覆寫。
5. **reward、指標、結束旗標**，然後對結束的列做自動重設。

## 3. 隨機數

| 來源 | 用在哪裡 | 說明 |
| --- | --- | --- |
| `HashDraws` | 正式使用 | 以 (episode seed, 流, 計數) 做 SplitMix64 雜湊。不存狀態、不需要 host 同步，也和 agent 的動作、其他列都無關。 |
| `V1Draws` | parity 測試 | 對同一個 episode seed，照 v1 的抽取順序重放它的 numpy 流（布局、軌跡相位、domain randomization、wander、puff 擾動、感測雜訊）。在 host 上逐列跑，只給測試用。 |

- **回合 seed**：環境 i 的第 k 回合用 hash(seed, i, k)。`reset(episode_seeds=...)` 可以直接指定。
- **v1 的一個特性**：用明確 seed 重設時，v1 的軌跡相位和 domain randomization 都從 `default_rng(seed)` 的第一個值開始抽，所以兩者相關。`V1Draws` 照樣重現，`HashDraws` 則用獨立的流。

## 4. 和 v1 的比對（parity）

- **方法**（計劃書 3.3、#145）：
  - 注入 v1 抽出的隨機量（`draws="v1"`），世界用 float64，`reward="v1"`、`occlusion_view="blackout"`；
  - 兩邊收到同一串動作，逐步比對觀測、reward、旗標和回合指標，用容差判定。
- **涵蓋範圍**：4 種情境 × 2 種目標運動。
  - 快速測試：40 步。遮蔽移到 0.5–1.5 s，讓重新捕捉的邏輯也會跑到。
  - `slow` 測試：完整回合（1,200 步），開 domain randomization，teacher 動作和隨機動作各一組。
- **結果**（CPU，float64）：

| 項目 | 最大誤差 |
| --- | --- |
| 視覺 | ≤ 1.8e-7 |
| 氣味、valid、旗標、回合步數、成功判定 | 0 |
| 姿態 | ≤ 4e-15 |
| reward | ≤ 5e-7（介面規定輸出 float32） |
| 回合的方位誤差 | ≤ 5e-14 |

- **視覺的容差**（PM 在 #158 的 Decision：預設用 float32）：子射線剛好擦過邊緣（牆和地板的交界、圓柱的邊緣）時，捨入可能讓它落到另一側，那個小眼會差一條子射線的份量（≤ 1/7）。
  - **條件**：float64 全部 ≤ 1e-4；float32 超過 1e-4 的比例 ≤ 1e-5，而且每個值的誤差 ≤ 1/7。
  - **理由**：3,000 個姿態、432 萬個值裡只有 2 個翻轉。v1 在交界上的取捨本身也是捨入決定的，float64 並沒有比較「對」；noise 情境的影像雜訊標準差（0.06）和一條子射線的份量（0.062）同量級；改用 float64 在消費級 GPU 上會明顯變慢（計劃書 0.2）。
  - 計劃書 3.3 的「≤ 1e-4」會由 PM 補上這個例外。

## 5. GPU 的快速路徑

在 CUDA 上用 `HashDraws` 時，有兩項加速：

- **融合 kernel**：熱迴圈改成 `torch.cuda.jiterator` 的融合 kernel（`kernels.py`）。jiterator 用 torch 內附的 NVRTC 編譯，不需要 gcc。涵蓋：
  - 時鐘、運動、wander；
  - plume tick，擾動的雜湊直接在 kernel 裡算；
  - 濃度、目標 render、隨機數。
- **CUDA graph**：每一步 replay 兩個 graph。
  - 第一個是物理部分；
  - 第二個是自動重設，只有在有列結束時才 replay。
  - 判斷要不要 replay 第二個 graph，是每一步唯一的一次 host 同步。
  - 重設 graph 只 render 前 K 個結束的列（K = max(32, B/16)）。同時結束的列比 K 多時（例如所有列同時被時間截斷），其餘的在 replay 後補 render。
- **和 eager 的差異**：eager 的 torch 程式仍然是參考實作，CUDA 測試會比較兩者。
  - 濃度、puff 擾動和影像雜訊的常態分布值，用 float32 運算（FP64 在消費級 GPU 上只有 1/64 的速度），所以最後幾個位元和 eager 不同。
  - 姿態、目標、reward 和 eager 完全相同（4070 實測）。
  - 視覺的翻轉比例 ≤ 1e-5；氣味的差異 ≤ 1e-6（絕對值）。

**速度**（4070，pursuit 動作，60 s 回合，`bench_world.py`，2026-11-18 的實測；不含模型）：

| B | 情境 | 沒有重設的步 | 有一列重設的步 | 預期平均 | 每秒環境步數 |
| --- | --- | --- | --- | --- | --- |
| 256 | normal | 2.97 ms | 3.54 ms | 3.08 ms | 83,072 |
| 256 | 四種混合 | 3.20 ms | 3.98 ms | 3.35 ms | 76,517 |
| 1,024 | normal | 4.58 ms | 5.40 ms | 5.05 ms | 202,645 |
| 1,024 | 四種混合 | 5.42 ms | 6.55 ms | 6.07 ms | 168,694 |
| 4,096 | normal | 12.41 ms | 13.92 ms | 13.87 ms | 295,375 |

- 預期平均：每一步至少有一列重設的機率是 1 − exp(−B / 1200)。
- 計劃書 3.3 的目標是「1,024 個環境每步 ≤ 5 ms」。normal 剛好在邊界上，四種混合超過約 1 ms；混合比較慢，主要因為 conflict 情境有兩個氣味源，所有列的 puff 槽都要加倍。
- 對照 v1：一個 `FlyTrackEnv` 在一個 CPU 執行緒上是每秒 343 步（4070，#139）。

## 6. Teacher 和 world audit（#132 第 3 部分）

- **`PrivilegedTeacher`**（`world/teacher.py`）：v1 `teacher_action` 的整批版本，實作 `interfaces.Teacher`，狀態寬度是 0。
  - 前進速度 = 0.7 ×（距離 − 1 m），乘上 max(0, cos 方位)，不倒退；轉向 = 2 × 方位。
  - 目標掉頭衝向 agent 時，照 v1 的規則往側邊閃避。
  - 可選 `occlusion_odor_gain`（v1 #33 的選項）：遮蔽時改用學生自己收到的原始氣味轉向，不用 plume 的真值。
  - 標籤照契約的速度和加速度上限，對前一個實際動作做 clip，和 v1 一樣。
  - 和 v1 比對：
    - 400 個隨機狀態（含閃避、遮蔽、沒有氣味或氣味失效），開關氣味選項都測，誤差 ≤ 1e-6；
    - parity 測試裡，每一步的 teacher 標籤也和 v1 env 的 `info["teacher_action"]` 比對。
- **world audit**（`world/audit.py`、`scripts/v2/audit_world.py`）：v1 `audit.py` 的 v2 版本。
  - 特權 teacher 的成功率要 ≥ 0.9；blind 策略（8 個固定動作，加上 teacher 實際動作的平均）都要 ≤ 0.1。
  - 和 v1 的差別：每個情境都檢查（計劃書 3.3），感測是開著的；每個策略的回合在同一批裡跑，所有策略用同一組回合 seed（配對）。
  - **4070 實測**（commit `5b2f1d5`，dirty = false，每格 100 回合、60 s；JSON 在本機 `runs/audit/world_4070_5b2f1d5.json`）：

| 運動 | 情境 | teacher 成功率 | 碰撞 | 方位誤差 | 最好的 blind 策略 |
| --- | --- | --- | --- | --- | --- |
| curved | 四種都一樣 | 1.00 | 0 | 2.23° | `left_0.05` 0.02 |
| wander | 四種都一樣 | 1.00 | 0 | 2.78° | 0.00 |

  - 四種情境的結果相同。原因是特權 teacher 和 blind 策略都不讀感測，情境只改變感測；noise 情境要三路同時失效才會觸發緊急停車，這 100 回合裡沒有發生。

## 7. `wind` 世界（WP2b，#133）

`V1CompatWorld(..., wind=WindConfig(...))` 就是 `wind` 世界。它和 `v1compat` 共用整個步進、事件、render 和 CUDA graph，只在下面幾處不同（`world/wind.py`）。

- **風**：均勻的風場，速度 U（預設 0.5 m/s；試跑的候選是 0.3、0.5）。
  - 風向每回合隨機。之後每個 plume tick（0.01 s）做一步 OU：平穩標準差 20°，相關時間 10 s。
- **filament plume**（Farrell 2002，2D）：
  - 目標每 0.1 s 釋放一個 filament；conflict 情境的 decoy 也會釋放。
  - filament 隨當下的風平流、做亂流隨機漫步，寬度 σ² = σ0² + 2 D age。
  - 壽命 20 s。每個環境最多 256 個，滿了就丟掉最舊的。
  - 預設不衰減。濃度是各 filament 2D 高斯的總和，形式和 v1 的 puff 相同。
- **觸角**：照介面契約，左 = (U/U_ref)·cos(θ − α)、右 = (U/U_ref)·cos(θ + α)，加上雜訊（標準差 0.05）。
  - θ 是風「吹來的方向」相對頭向的角度，左為正。
  - 取樣用氣味的時鐘（`wind_hz` = `odor_hz` = 50 Hz）；noise 情境的 dropout 和氣味相同。
  - 觀測放在 `wind[2]`、`valid[:, 3]`；特權的 `wind_velocity` 是當下的風。
- **雜湊的 seed**：回合 seed 限制在 2⁵² 以下（`rng.SEED_BITS`）。這樣一個 double 就能精確裝下 seed，融合的 plume kernel 才能帶著每列的風，又不超過 jiterator 8 個輸入的上限。
- **測試**（`test_v2_world_wind.py`）：
  - 觸角：正前、正左、正右、正後四個方向的讀值（#161 C1）、左右鏡像對稱，以及讀值反推 θ；
  - 世界的觀測和特權欄位一致；
  - 沒有亂流時，filament 每個 tick 剛好移動 wind·dt，而且在目標的下風處；
  - 不衰減時，濃度在平面上的積分等於 filament 數 × 質量（相對誤差 ≤ 1e-3）；
  - CUDA 上 fused 和 eager 一致。

### 只看觀測的遮蔽 teacher（`OcclusionTeacher`）

DAgger 在遮蔽期間用它的標籤（#133、#161 D4）；遮蔽以外則是特權 teacher。

- **遮蔽期間的規則**：
  - **有氣味**：逆風（觸角算出的角度 × 0.5），加上左右氣味差 × 4·(L − R)/(L + R)；氣味越強越慢，到平均濃度 2 時停下，避免撞上氣味源。
  - **沒有氣味**：維持遮蔽前一刻的轉向速度（dead reckoning）。
  - **速度**：遮蔽前一刻的實際速度。
  - **撞牆**：前方 0.8 m 的點如果離開安全範圍，就轉向場地中心。這只用 agent 自己的姿態，不用目標。
  - **v1compat（風無效）**：只剩氣味轉向。
- **不偷看目標**（#161 C3）：測試中，可觀測的歷史和前一個動作都相同，只改目標的位置和速度，連續 10 步的標籤和狀態都完全相同。
- **參數怎麼選**：
  - 調參只用 seed 1,630,000,000 起的回合，不碰 audit 的 seed（1,620,000,000 起）；
  - 每種目標運動 100 回合、U = 0.5。
  - 先試純氣味＋橫風搜尋：只有 0.06–0.09，碰撞 19–22%。之後加了記憶、近源減速和避牆。
- **audit**（`scripts/v2/audit_world.py --worlds wind`；4070，每格 100 回合、60 s，用 audit 的 seed）：

| U | 運動 | 遮蔽 teacher | 只有記憶（不看氣味、風） |
| --- | --- | --- | --- |
| 0.5 | curved | **0.83** | 0.51 |
| 0.5 | wander | **0.66** | 0.05 |
| 0.3 | curved | 0.64 | 0.51 |
| 0.3 | wander | 0.50 | 0.05 |

- 協定的門檻是 ≥ 0.5（`odor_teacher_min_occlusion_success`），U = 0.5 兩種運動都通過。
- **比較**：curved 的目標曲率固定，只靠記憶就有 0.51；wander 只靠記憶只有 0.05，加上氣味後是 0.66。這個世界裡，氣味對遮蔽中的追蹤確實有用。
- U = 0.3 時 wander 剛好落在門檻上（teacher 的參數是在 U = 0.5 調的）。
- 其他情境：特權 teacher 都是 1.00，最好的 blind 策略 ≤ 0.02，結果和 `v1compat` 相同。

**速度**（4070，`bench_world.py --worlds wind`）：B = 1,024，normal 沒有重設的步 5.48 ms、預期平均 5.99 ms；四種混合 6.57 ms。比 `v1compat` 慢約 0.9 ms，原因是 filament 比較多（203 對 163 個），而且每個 tick 都要更新風。
