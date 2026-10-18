# BC 只學到平均動作：根因與情境測試（2026-10-16～17）

> **注意（2026-10-18，#4）：**本文件所有 VO 結果的動作都完全不受氣味影響（去掉氣味時 Δω 為 0），實際上只用了視覺；conflict 情境的數字因此等於 normal。VO、V-only、O-only 的比較與 lesion 結論在 #4 修好前都不成立。

> **後續（2026-10-17）：**第 5 節的建議已在 `integrate/tracking-fixes` 實作並設為預設：隨機化佈局（另加 wander 目標）、成功須朝向目標、BC gate 改用相關／R²／幅度、DN 輸入增益改在初始化設定、讀出 ω 偏置與 BC 分組學習率。環境另加盲目策略審計；conflict 情境的強度已量測。見 [fixes_2026-10-17.md](../fixes_2026-10-17.md)。本文件的數據是當時的設定（選項預設關閉）。重現原環境需在 env 加 `layout_preset: legacy` 與 `success_bearing_deg: null`；本文件的隨機方向環境需加 `randomize_target_layout: false` 與 `success_bearing_deg: null`（`configs/` 下的兩個設定檔已加上）。

接續 [diagnosis_2026-10-16.md](diagnosis_2026-10-16.md)。該文件的結論是「BC 開始時 DN 左右差幅度太小」，建議放大接進 DN 的連結。本文件記錄這個建議的測試，以及後續找到的真正原因。

**結論：主要原因在訓練與評估的環境，不在 DN 的訊號強度。**

- 原本的環境裡，目標永遠以固定角速度逆時針繞圈，蒼蠅永遠從目標正後方出發。「固定左轉」就是近乎完美的策略：成功率 100%，和 teacher 一樣。
- BC 學到固定左轉是合理結果，評估也分不出有沒有在追蹤。
- 把繞圈方向和起始位置隨機化後，原樣的模型 BC 就學會依方位轉向：ω 和 teacher 的相關從約 0.1 升到 0.76～0.78。

所有測試都是 VO、seed 0、單一訓練種子的小預算（BC teacher 30k＋DAgger 2×15k）。數據是診斷用，不是正式結果。

## 1. 放大接進 DN 的連結（在 BC 開始時 ×100）

做法：從嗅覺 pretrain 模型出發，用 `scripts/diagnostics/scale_dn_input.py` 把接進 6 個運動 DN 的 147 條連結（82 個標籤）的強度乘上 k，再跑 BC。k1 是對照組。數據：[gain_bc_dn_health.json](data/gain_bc_dn_health.json)、[gain_bc_behaviour.json](data/gain_bc_behaviour.json)、[gain_bc_layers.json](data/gain_bc_layers.json)。

**BC 開始前：增益有效。**

| | k1 | k100 |
|---|---|---|
| DN 左右差 R²（teacher ω） | 0.63 | 0.63 |
| DN 左右差變動幅度 | 0.0003 | 0.018 |
| 換眼後 DN 左右差（中位數） | 0.002 | 0.13 |

**BC 之後：兩組的資訊都消失。**

| | k1 | k100 |
|---|---|---|
| DN 左右差 R² | −0.16 | −0.24 |
| 策略 ω 的變動 | 0.008 | 0.00001 |
| 策略 ω 與 teacher 的相關 | 0.12 | 0.35 |
| Gate 換眼差（門檻 ≥0.05） | 0.047 | 0.0002 |
| Gate ω 誤差改善（24×200 步，門檻 ≥0.2） | 0.18 | **0.32（通過）** |

k100 在 BC 中把兩個 DNa02 壓到電壓為負（等於關閉），ω 固定在約 0.089。它的 ω 完全固定，卻通過「比零輸出好」的 gate，再次證明這個門檻太弱。

## 2. BC 損失與離線訓練

**teacher 動作統計**（BC 資料 6 萬步）：
- v：平均 0.171、標準差 0.035。
- ω：平均 0.081、標準差 0.079，幾乎全為正。
- BC 損失以 (v_max, ω_max) = (0.5, 1.5) 正規化，「只輸出平均動作」的驗證損失是 0.00496。

**離線 BC**（`scripts/diagnostics/offline_bc.py`）：同一份資料，2 個起點（k1、k100）× 3 種訓練方式，各 10 epoch。數據：[offline_bc_fits.json](data/offline_bc_fits.json)、[offline_bc_behaviour.json](data/offline_bc_behaviour.json)。

| 起點／方式 | 最佳驗證損失（平均動作 0.00496） | DN 左右差 R² | 策略 ω 與 teacher 相關 | 策略 ω 變動 |
|---|---|---|---|---|
| k100／全部訓練 | 0.00495 | −0.12 | 0.12 | 0.002 |
| k100／固定 `core.bias` | 0.00502 | **0.67** | **0.82** | 0.001 |
| k100／只訓練讀出層 | 0.00518 | 0.64 | −0.23 | 0.003 |
| k1／全部訓練 | 0.00494 | 0.08 | 0.12 | 0.001 |
| k1／固定 `core.bias` | 0.00513 | 0.67 | 0.21 | 0.004 |
| k1／只訓練讀出層 | 0.00638 | 0.63 | −0.11 | 0.000 |

- 沒有任何一組比「輸出平均動作」好。
- **破壞 DN 資訊的是神經元偏置 `core.bias` 的學習**：固定它，資訊就保住。偏置的梯度比連結強度大約 30～1000 倍。
- 固定偏置時轉向方向正確（相關 0.82），但幅度只有 teacher 的約 1/80。原因是讀出層的 ω 沒有偏置項，同一個權重要同時產生平均轉向和依方位調整。DN 左右差的平均值約是其變動量的 4 倍，權重被平均值拉著走。

**讀出層的上限**（`scripts/diagnostics/readout_closed_form.py`，閉式解，[readout_closed_form.json](data/readout_closed_form.json)）：用最佳線性讀出，ω 和 teacher 的相關也只有 0.43～0.50，ω 誤差只比平均值低 12～16%。在這份資料裡，學轉向能拿到的分數本來就很少。

## 3. 固定動作策略：環境本身的問題

**原因（程式碼）：**
- `python/flytrack_core/world/trajectories.py`：curved 目標以 0.2 m/s 沿半徑 2 m 的圓移動，角速度固定 0.1 rad/s，永遠逆時針。
- `python/flytrack_sim/env.py`：蒼蠅永遠從目標正後方 1.5 m、同方向出發。
- 唯一的隨機性是圓上的起始相位，對不看輸入的策略來說，旋轉後完全相同。

**訓練流程的情境：**
- pretrain、BC 只用 curved＋normal。
- 只有 PPO 的課程用到 stationary 和 linear，但兩者每回合也完全相同：stationary 是目標固定在中心，linear 是永遠從中心往 +x 走。

**實測**（每種策略 × 4 情境 × 30 回合；環境種子從 1.6e9 起，不和訓練、解碼器驗證、正式評估重疊）。數據：[fixed_action_original_env.json](data/fixed_action_original_env.json)、[fixed_action_varied_env.json](data/fixed_action_varied_env.json)。

| 策略 | 原環境成功率 | 隨機方向環境成功率 | 原環境方位誤差 |
|---|---|---|---|
| teacher | 100% | 100% | 2.6° |
| 固定 v=0.2, ω=0.1（左轉） | **100%** | 20% | 93.9° |
| 固定 v=0.17, ω=0.1 | **100%** | 23% | 93.9° |
| 固定 v=0.17, ω=0.08 | 0%（碰撞） | 17% | 51.6° |
| 固定右轉 ω=−0.1 | 0% | 20% | 106.2° |
| 直走 | 0% | 0% | 79.9° |

- 原環境中，固定左轉策略在 normal、occlusion、noise、conflict **4 種情境都是 100% 成功**：遮擋、雜訊、干擾源只影響感覺輸入，不影響目標運動。
- 成功判定只看「80% 時間在 2 m 內且沒碰撞」，所以方位誤差 94°（側對目標）也算成功。
- 隨機方向環境中，所有固定策略都降到 23% 以下，teacher 仍是 100%。新環境能區分真的追蹤和固定繞圈。

## 4. 情境測試：隨機方向環境中的 BC

**新增的環境選項**（預設全部關閉，原行為不變）：
- `randomize_target_direction`：每回合隨機選順時針或逆時針。
- `start_lateral_offset_m`：起始位置的左右偏移上限。
- `start_heading_offset_deg`：起始朝向的偏移上限。
- 這些隨機數用獨立的亂數流（以回合種子衍生），不改變其他隨機數的序列。

本測試設定：隨機方向、左右偏移 ±0.5 m、朝向 ±30°。

**兩組 BC**，都從同一個嗅覺 pretrain 模型出發，同一個環境，`early_stopping_min_delta: 0.005`。設定檔：[scenario_bc_A.yaml](configs/scenario_bc_A.yaml)、[scenario_bc_B.yaml](configs/scenario_bc_B.yaml)。

- **A：原樣模型。**
- **B：加上三項修正：**
  - 接進 DN 的連結 ×100（仍是在 BC 開始時套用）。
  - 讀出層加 ω 偏置：`model.readout_turn_bias: true`。
  - BC 固定神經元偏置（`train.core_bias_learning_rate: 0`），讀出層學習率提高到 0.01（`train.readout_learning_rate: 0.01`）。

**DN 健康與行為**（teacher 駕駛，24 回合 × 200 步；A、B 在隨機方向環境量測，對照組在原環境量測）。數據：[scenario_dn_health_A.json](data/scenario_dn_health_A.json)、[scenario_dn_health_B.json](data/scenario_dn_health_B.json)、[scenario_behaviour.json](data/scenario_behaviour.json)。

| | 原環境 BC（k1，對照） | A | B |
|---|---|---|---|
| 策略 ω 與 teacher 相關 | 0.12 | **0.78** | **0.75** |
| 策略 ω 的 R² | −3.99 | 0.12 | **0.32** |
| 策略 ω 變動（teacher 約 0.08～0.10） | 0.008 | 0.079 | 0.068 |
| 換眼差中位數（門檻 ≥0.05） | 0.047 | **0.156** | **0.257** |
| ω 誤差改善（門檻 ≥0.2） | 0.18 | 0.13 | **0.24** |
| DN 左右差 R² | −0.16 | 0.51 | 0.58 |
| 匯流神經元沉默比例 | 75% | 100% | 50% |
| BC 時間 | — | 46 分 | 27 分 |

**正式指標**（`scripts/diagnostics/policy_episode_eval.py`，4 情境 × 30 回合）。數據：[scenario_episodes_varied_env.json](data/scenario_episodes_varied_env.json)、[scenario_episodes_original_env.json](data/scenario_episodes_original_env.json)。

| 評估環境 | 策略 | normal | occlusion | noise | conflict | normal 方位誤差 |
|---|---|---|---|---|---|---|
| 隨機方向 | teacher | 100% | 100% | 100% | 100% | 2.7° |
| 隨機方向 | 固定左轉 | 20% | 20% | 20% | 20% | 95.8° |
| 隨機方向 | A | 57% | 27% | 3% | 57% | 33.3° |
| 隨機方向 | B | 40% | 30% | 30% | 40% | 51.8° |
| 原環境 | A | 70% | 0% | 0% | 70% | 23.7° |
| 原環境 | B | 23% | 23% | 77% | 23% | 56.2° |

**解讀：**

1. **只要資料需要看方位，原樣模型的 BC 就學會轉向。**A 的相關 0.78、換眼差 0.156，都是原環境 BC 的好幾倍。主要問題是環境。
2. **B 的修正有幫助，但不是必要。**
   - B 的 ω R²、換眼差、ω 誤差改善都比 A 好，匯流神經元也沒有全部沉默；它是唯一通過所有 BC gate 的一組。
   - BC 時間也較短（27 分 vs 46 分）。
   - 但正式指標的成功率互有勝負，而且只有 30 回合、單一種子，差距在雜訊範圍內。
3. **兩者都遠不如 teacher**（成功率 3～77%）。這是只有 BC 的小預算結果，後面還有 PPO，現在不能用來判斷最終能力。
4. **conflict 的結果和 normal 完全相同。**在一個抽查回合中，干擾源只讓嗅覺輸入改變不到 0.01，對策略幾乎沒有影響。這個情境可能太弱，需要另外檢查。
5. noise 的結果起伏很大（A 3%、B 77%），原因還沒查。

## 5. 建議（需要組內決定）

1. **改訓練和評估的環境：**至少打開隨機繞圈方向和起始偏移，並考慮讓 stationary、linear 目標也有隨機位置或方向。這會改到已登記的評估協定（`configs/evaluation_protocol.yaml`）。
2. **改成功判定：**加入方位條件，並在正式報告中加入固定動作策略作為基準。
3. **BC gate：**改用和 teacher ω 的相關係數或 R²，量 24 回合 × 200 步；`early_stopping_min_delta` 用約 0.005。
4. **模型修正（B 的三項）：**
   - 讀出層 ω 偏置、BC 分組學習率：已實作成可選設定，預設關閉。
   - DN 輸入增益：要採用的話，應改成在模型初始化時設定，再用多個種子比較。
5. **conflict 情境的強度**需要檢查。

## 這次的程式變更

全部預設關閉，舊設定和舊 artifact 的行為不變。

- `TrajectoryConfig.direction`（±1）。
- `FlyTrackEnv` 的 `randomize_target_direction`、`start_lateral_offset_m`、`start_heading_offset_deg`。
- 模型設定 `readout_turn_bias`。
- BC 設定 `core_bias_learning_rate`、`readout_learning_rate`（`flytrack_train.imitation.parameter_groups`）。
- 新增的診斷腳本（`scripts/diagnostics/`）：
  - `scale_dn_input.py`（新增 `--turn-bias`）
  - `offline_bc.py`
  - `readout_closed_form.py`
  - `policy_episode_eval.py`
- `dn_gain_counterfactual.py`：可用 `GAINS`、`SCOPES` 環境變數，並輸出策略 ω 的相關係數與 R²。
- 新增 3 個單元測試，全部 101 個測試通過。

## 重現方式

```bash
export PYTHONPATH=python
# 起點 B：DN 輸入 ×100 並加讀出 ω 偏置
.venv/bin/python scripts/diagnostics/scale_dn_input.py --source <嗅覺 pretrain actor> --output <start_B>/actor --gain 100 --turn-bias
# BC（設定檔中的 <runs>/ 改成本機路徑）
.venv/bin/python -m flytrack_train bc --config docs/dn/configs/scenario_bc_A.yaml --device cuda --workers 6
# DN 健康（使用 artifact 記錄的環境設定）
.venv/bin/python scripts/dn_health.py --artifact <bc_A>/actor --stage bc --episodes 24 --steps 200 --output out.json
# 正式指標；固定動作基準內建，環境可用 ENV_OVERRIDES 覆寫
ENV_OVERRIDES='{"randomize_target_direction": true, "start_lateral_offset_m": 0.5, "start_heading_offset_deg": 30}' \
  .venv/bin/python scripts/diagnostics/policy_episode_eval.py out.json docs/dn/configs/scenario_bc_A.yaml <bc_A>/actor
```

JSON 和設定檔中的 `<runs>/` 代表本機的 `runs/` 目錄（不納入版本控制）。原環境的固定動作數據是用較早的同等腳本產生，策略組合比 `policy_episode_eval.py` 的預設多。
