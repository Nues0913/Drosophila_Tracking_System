# v2 評估輸出

> **M0 凍結**（計劃書 5.3）。程式是 `python/flytrack_v2/metrics.py`，用 `validate_episode_record` 檢查。

## 逐回合紀錄（JSON Lines，每行一個回合）

| 欄位 | 型別 | 內容 |
| --- | --- | --- |
| `run_id` | str | 評估工作的識別碼 |
| `variant` | str | VO2、V2、O2、RW2、ER2、GRU2；formal-v1 的 actor 寫成 `v1:<變體>`，例如 `v1:VO-late`（T-4） |
| `train_seed` | int | 訓練的 seed |
| `world` | str | `v1compat` 或 `wind` |
| `scenario` | str | `normal`、`occlusion`、`noise`、`conflict` |
| `motion` | str | `curved` 或 `wander` |
| `episode_seed` | int | 回合 seed；配對比較用同一組 |
| `success` | bool | 照 `SUCCESS_RULE`：2 m 內、45° 內的步數 ≥ 80%，而且沒有碰撞 |
| `tracking_fraction` | float | 同時符合距離和方位條件的步數比例，範圍 [0, 1] |
| `within_distance_fraction` | float | 只看距離條件的步數比例，範圍 [0, 1]（v1 的指標） |
| `collision` | bool | 有沒有碰撞 |
| `steps` | int | 回合的步數 |
| `mean_distance_m` | float | 和目標的平均距離 |
| `bearing_error_deg` | float | 平均 \|方位\| |
| `mean_speed_m_s` | float | 實際套用的平均速度 |
| `speed_std_m_s` | float | 回合內速度的標準差，用來檢查 S2（速度固定） |
| `occlusion_tracking_fraction` | float 或 null | 遮蔽期間的追蹤比例；沒有遮蔽的情境填 null |
| `reacquisition_time_s` | float 或 null | 遮蔽結束後，重新對準目標（\|方位\| < 15°）所花的時間；沒有遮蔽，或沒有重新對準（截尾）時填 null |
| `reacquisition_censored` | bool 或 null | 遮蔽結束後沒有重新對準，表示這筆是截尾資料；沒有遮蔽時填 null。截尾的回合不能填成 0 秒。遮蔽還沒結束、回合就先結束（例如提早碰撞）時，這欄和 `reacquisition_time_s` 都填 null（#161 E3） |
| `noise_scale` | float 或 null | noise 情境的強度（協定的 0.75，以及敏感度的 0.5、1.0）；其他情境填 null |
| `artifact_sha256` | str | 被評估的 artifact 的 `SHA256SUMS` 雜湊 |
| `commit` | str | 評估程式的 git SHA |

**數值規則**
- 浮點數都必須是有限值。可以是 null 的欄位不能用 NaN 代替。
- `noise_scale` 只能在 noise 情境填值，其他情境一定是 null。

**解讀的限制**
- `mean_distance_m`、`bearing_error_deg` 只平均到回合結束。碰撞會讓回合提早結束，所以提早撞的變體只算到前段的步數。
- 因此碰撞率不同的變體，不能直接比較這兩項。要比較時，另外報告沒有碰撞的回合（#108 的 review）。
- **遮蔽還沒結束就結束的回合**（#161 E3）：`reacquisition_censored` 和 `reacquisition_time_s` 都是 null，不納入重新捕獲時間，也不算進截尾比例。報告另外列出它們的比例：occlusion 情境中，`reacquisition_censored` 是 null 而且 `collision` 是 true 的回合，就是「遮蔽結束前碰撞」。

## 評估執行器的次要輸出（`flytrack_v2/eval/runner.py`，不寫進逐回合紀錄）
- **速度對距離**（#161 A3）：每個回合，用「actor 下指令前看到的距離」對指令速度做最小平方回歸，得到斜率（m/s per m）。不用這一步之後的距離，因為它已經被這一步的速度改變，會讓斜率偏負。
- **遮蔽的四段追蹤比例**（#161 E1）：遮蔽前、遮蔽中、遮蔽結束到重新對準之前（\|方位\| < 15°，和世界的規則相同）、重新對準之後。
- **觀測消融**（#161 E1）：同一個 checkpoint、同一批回合 seed，只改 actor 看到的觀測。
  - `swap_odor`：左右氣味交換，總濃度不變；
  - `rotate_wind(Δ, α)`：把觸角讀值換成風從 θ + Δ 吹來時的讀值（線性轉換，所以雜訊也一起轉）。
  - 世界本身沒有改變；不過 actor 是照它看到的觀測行動，所以軌跡會不一樣。

**彙總 JSON**：格式（seed 平均、bootstrap 信賴區間、配對比較、Holm 校正）由 WP5（#139）定義，加在這份文件。
