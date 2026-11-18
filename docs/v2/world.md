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
  - 注入 v1 抽出的隨機量（`draws="v1"`），世界用 float64；
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
