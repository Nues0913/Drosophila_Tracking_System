# v2 介面契約

> **M0 凍結**（計劃書 5.3）。之後要改，須有 PM 的 Decision，並經兩位 reviewer 審查。
> - 程式：`python/flytrack_v2/` 的 `contracts.py`、`interfaces.py`、`artifacts.py`、`metrics.py`
> - 測試：`python/tests/v2/`
> - 計劃書：`docs/v2/大實驗計劃書.md`（#130）

## 1. 感測契約

檔案是 `configs/v2/sensor_contract.json`，類別是 `SensorContractV2`。
- **沿用 v1**：v1 的全部欄位，包括相機、六角眼（每眼 721 個小眼）、時序（控制 50 ms、神經 10 ms）和動作上限，數值都不變。
- **新增三個欄位**：
  - `wind_hz` = 50；
  - `antenna_angle_deg` = 45；
  - `wind_reference_m_s` = 0.5。
- **雜湊**：雜湊內容包含 schema 4 和 `spec_version: "v2"`，所以和 v1 的雜湊不同。

**觸角模型**（WP2b #133 實作；計劃書 3.3 和 #133 的內文以這裡為準）：
- 符號：θ 是風「吹來的方向」相對頭向的角度（左為正），α = `antenna_angle_deg`，U 是風速。
- 公式：
  - 左 = (U / `wind_reference_m_s`) · cos(θ − α)
  - 右 = (U / `wind_reference_m_s`) · cos(θ + α)
  - 再加上世界設定的雜訊。
- 性質：左 − 右和 sin θ 成正比，在正前方附近和方位角成線性（Suver et al. 2019）。

## 2. 觀測（actor 只看得到這些）

| key | 形狀 | 內容 |
| --- | --- | --- |
| `vision` | `[B, 2, 721]` | 左、右眼的灰階，範圍 [0, 1] |
| `odor` | `[B, 2]` | 左、右的氣味濃度，≥ 0，任意單位 |
| `wind` | `[B, 2]` | 左、右觸角的訊號（無因次） |
| `valid` | `[B, 4]` | 左眼、右眼、氣味、風的有效旗標，值是 0 或 1 |

- **資料型別與位置**：全部是 float32，放在同一個 device 上。
- **缺資料時**：數值填 0，對應的 `valid` 也填 0。不用 NaN。
- **v1compat 世界**：沒有風，所以 `wind` 是 0，`valid[:, 3]` 也是 0。
- **取樣時序**：和 v1 的 `FlyTrackEnv.step` 一樣，在感測器的時間點切分子步。
  - 每個控制步裡，積分的切點是兩組時間點的聯集：
    - 10 ms 的神經步長格點；
    - vision（1/30 s 的倍數）、odor 和 wind（1/50 s 的倍數）的取樣時間。
  - 兩次取樣之間，感測器保持上一次的值。
  - 不是把取樣時間取整到 10 ms 的格點。取整會讓位置最多差 5 mm，vision 的 parity 就過不了。
- **`validate_obs(obs, contract)`**：
  - 預設只檢查 key、型別、device 和形狀，不會和 GPU 同步，可以放在熱路徑；
  - 加上 `check_values=True` 會再檢查數值的範圍和有限性，需要同步，只在測試和除錯時用。

## 3. 特權狀態（只給 teacher、critic 和評估）

`world.privileged()` 回傳以下內容，actor 絕對拿不到：

| key | 形狀 | 內容 |
| --- | --- | --- |
| `agent_pose` | `[B, 3]` | x、y（m），yaw（rad） |
| `agent_velocity` | `[B, 2]` | 實際套用的 v（m/s）、ω（rad/s） |
| `target_pose` | `[B, 3]` | x、y（m），heading（rad） |
| `target_velocity` | `[B, 2]` | vx、vy（m/s），世界座標 |
| `occluded` | `[B]` bool | 情境的遮蔽中 |
| `target_visible` | `[B]` bool | 在任一眼的視野內，而且沒有被遮蔽 |
| `wind_velocity` | `[B, 2]` | vx、vy（m/s），世界座標；v1compat 世界是 0 |
| `decoy_active`、`decoy_position` | `[B]`、`[B, 2]` | conflict 情境的誘餌；沒有誘餌時位置是 0 |
| `time_s`、`episode_seed` | `[B]` | 回合時間、回合 seed |

## 4. 世界：`BatchedWorld`

所有環境在同一個 device 上一起步進。建構方式由 WP2a（#132）決定，但物件要有 `contract`、`num_envs`、`device` 三個屬性。

**`reset(mask=None, episode_seeds=None) -> obs`**
- `mask` 指定要重設哪幾列；`None` 表示全部重設。
- `episode_seeds` 可以明確指定每一列的回合 seed，評估用它來配對比較。

**`step(action) -> StepResult(obs, reward, terminated, truncated, info, final_obs, final_privileged)`**
- **動作**：
  - `action` 是 `[B, 2]`，表示要求的 (v, ω)；
  - 世界照 contract 裁剪速度和加速度。
- **停車與碰撞**：
  - 四個 `valid` 都是 0 時停車，和 v1 一樣；
  - 碰撞的那一列，在這個控制步剩下的時間不再移動。v1 是跳出迴圈，這裡用遮罩做到同樣的效果。
- **回合結束**：
  - `terminated` 表示碰撞而提早結束；
  - `truncated` 表示到了時間上限。
- **自動重設**：
  - 結束的列會自動重設，所以 `obs` 和 `privileged()` 已經是下一回合的狀態；
  - 下一次呼叫 policy 時，`reset_mask` 要設成 `terminated | truncated`。
- **`final_obs`、`final_privileged`**：
  - 這一步實際到達的狀態，是在自動重設**之前**的值；
  - 沒有結束的列，和 `obs`、`privileged()` 相同；
  - PPO 遇到 `truncated` 的列，critic 用 `final_privileged` 做 bootstrap。
- **回合 seed**：
  - 第 i 個環境的第 k 回合，seed 由 (seed, i, k) 推導；
  - 世界的隨機性（布局、目標路徑、plume、雜訊）**不能受 agent 的動作影響**，配對比較才能共用同一組抽樣；
  - 每一列有自己的隨機流，用 counter-based 的雜湊實作（#132）。
- **parity 測試**（計劃書 4.2）：
  - 逐步比對：把 v1 用 `np_random` 抽出的隨機量注入 v1compat，再用容差比對；
  - 分布比對：用世界自己的隨機流，比對 audit 的成功率、雜訊的均值和變異數等統計。
  - 注入的方式是世界的建構參數，不在這份介面裡。
- **reward**：
  - 預設是計劃書 3.8 的 v2 reward；
  - 世界設定 `reward: v1` 會改成 v1 的 reward，給 parity 測試用。
- **`info`**：
  - 每一步都有：`tracked`、`collision`、`distance_m`、`bearing_rad`、`occluded`；
  - 回合結束的列才有意義：
    - `episode_success`、`episode_tracking_fraction`、`episode_within_distance_fraction`；
    - `episode_collision`、`episode_steps`；
    - `episode_mean_distance_m`、`episode_bearing_error_deg`；
    - `episode_mean_speed_m_s`、`episode_speed_std_m_s`；
    - `episode_occlusion_tracking_fraction`；
    - `episode_reacquisition_time_s`、`episode_reacquisition_censored`；
    - `episode_seed`。
- **成功的定義**（`SUCCESS_RULE`，和 v1 協定 schema 3 相同）：
  - 在目標 2 m 內、方位誤差 45° 內的步數 ≥ 80%；
  - 而且沒有碰撞。

## 5. Policy

VO2、V2、O2、RW2、ER2、GRU2 共用這個介面。

**兩種狀態**
- **感覺狀態**（凍結前端、ORN 適應）：
  - 由 policy 自己管理，版面由 policy 決定；
  - v2 的前端用 node-major（`[節點, 2B]`），每一步不用轉置，比 batch-major 快 1.9 倍（WP3b #135）；
  - trainer 只做兩件事：用 `reset_sensory_state` 重設，再把它傳回 `sense()`；
  - 不存進 buffer。
- **中央狀態**：`[B, central_state_size]`，buffer 會存它，給 burn-in 和截斷反向傳播用。

**方法**
- `initial_sensory_state(B, device)`、`reset_sensory_state(state, reset_mask)`：建立和重設感覺狀態。
- `initial_central_state(B, device)`：中央狀態的初始值。
- `sense(obs, sensory_state) -> (code, sensory_state)`：
  - 只跑凍結的部分，碰不到任何可訓練的參數；
  - 所以 code（`[B, code_size]`）可以在 rollout 時算一次，給訓練重複使用。
- `forward_code(code, central_state) -> PolicyStep(dist, central_state, aux)`：所有可訓練的參數都在這裡。

**輔助函式**
- `policy_step(policy, obs, sensory_state, central_state, reset_mask)`：
  - 依序做三件事：重設結束的列、`sense`、`forward_code`；
  - 回傳 `(step, sensory_state, code)`。
- `reset_rows(state, initial, reset_mask)`：任何 `[B, …]` 的狀態，只把 `reset_mask` 為 True 的列換成初始值，不會和 GPU 同步。

**限制**：actor 只接收 `OBS_KEYS`，測試裡用 `validate_obs` 鎖住這一點。

**動作分布 `ActionDistribution`**
- 定義：在潛在變數 z 上的對角高斯，再轉換成 v = v_max · sigmoid(z₀)、ω = ω_max · tanh(z₁)。
- **PPO**：機率比用存下來的 z 計算 `log_prob`。轉換的 Jacobian 在比值裡會消掉，所以比值是精確的。
- **模仿學習**：損失是 teacher 動作的高斯負對數似然，算法是 `log_prob(unsquash(action))`。
  - `unsquash` 先把正規化後的動作截到 [ε, 1 − ε]（速度）和 [−1 + ε, 1 − ε]（轉向），再轉成 z；
  - ε = `ACTION_EPS` = 1e-6，和 v1 的 `distributions.py` 相同；
  - 在動作上限上的值也會得到有限的 z，訓練和評估都用同一個 ε。
- 確定性的動作用 `mode()`。

**`aux`**：放診斷用的資訊（例如 DN 活動），不會回饋給世界。

## 6. Teacher

- **介面**：
  - `initial_state(B, device) -> [B, state_size]`，`state_size` 可以是 0；
  - `act(privileged, obs, previous_action, state) -> (action, state)`，輸出在動作上限內。
- **為什麼要狀態**：遮蔽期間要維持遮蔽前的速度，橫風搜尋也要計時或記方向，這些都放在 teacher 的狀態裡。
- **重設**：世界自動重設的列（`terminated | truncated`），由呼叫端用 `reset_rows` 重設 teacher 的狀態，和 policy 的中央狀態一樣。
- **限制**：
  - teacher 的輸出只當訓練標籤，不能當成觀測；
  - 遮蔽期間，只看觀測的氣味 teacher（WP2b）只能使用 `obs`。

## 7. 產物 schema 4

**檔案**
- 所有變體都有：`weights.pt`、`frontend.npz`、`input_norm.npz`、`sensor_contract.json`、`model_card.json`、`SHA256SUMS`。
- graph 變體另外有 `graph.npz`；GRU2 沒有。
- `graph.npz` 存的是 policy **實際用的圖**。RW2、ER2 存接線之後的圖，載入時不重新接線（#161 C4）。
- 匯出後再載入，要和原本的 policy 在同一段觀測上逐步給出相同的動作和隱狀態（容差比對）。六個變體都要有這個往返測試（WP3c、WP6）。

**`model_card.json` 的必要欄位**

| 欄位 | 內容 |
| --- | --- |
| `spec_version` | `"v2"` |
| `schema_version` | 4 |
| `variant` | 變體名稱 |
| `sensor_contract_sha256` | 感測契約的雜湊 |
| `files` | 每個檔案的 SHA-256，model card 和 `SHA256SUMS` 除外 |
| `state` | `central_state_size`、`code_size`；感覺狀態是 policy 內部的物件，不列 |
| `control_dt`、`neural_dt`、`n_ommatidia` | 時序與小眼數 |
| `action_limits` | `v_max`、`omega_max` |
| `model_config` | 完整解析後的模型設定：讀出的型別清單、g 的參數化、`structure_seed`、輸入對應等。和 `graph.npz` 一起，不需要訓練設定就能重建 policy（#161 C4） |
| `provenance` | `git_sha`、`dirty`、`config_sha256`、`stages` |

`validate_model_card` 會拒絕 v1 的 card，因為 v1 的 card 沒有 `spec_version`，避免 v1 和 v2 混用。

## 8. 評估輸出

見 [metrics.md](metrics.md)。

## 9. 共同規則

- **數值比對用容差**（`torch.allclose`），不比位元，也不重跑比對位元（計劃書 0.2）。
- **熱路徑不和 GPU 同步**：不用 `.item()`、`bool(tensor)`、`.cpu()`。統計在 GPU 上累積，每次更新只同步一次（計劃書 4.4）。
