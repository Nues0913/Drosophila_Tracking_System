# 運動 DN 沉默問題修正計畫

狀態：**已執行，僅保留作歷史紀錄**（建立日期：2026-10-15）。A＋C 修正見 [dn/fix_A_C_2026-10-15.md](dn/fix_A_C_2026-10-15.md)，驗證與後續診斷見 [dn/diagnosis_2026-10-16.md](dn/diagnosis_2026-10-16.md)、[dn/scenario_test_2026-10-17.md](dn/scenario_test_2026-10-17.md)，整合後的修正見 [fixes_2026-10-17.md](fixes_2026-10-17.md)；目前狀態以 [status.md](status.md) 為準。原始證據 `runs/dn_audit/REPORT.md` 不在版本控制中，只存在當時的主機上。

## 1. 問題

連接體變體（V、O、VO、VO-late、RS）的動作只由 6 個運動 DN 讀出（DNa01、DNa02 各左右一個作轉向，DNp09 左右作前進；`MotorReadout`，`python/flytrack_core/model/policy.py`）。實測完整視覺預訓練（V seed 0，240k 步，提早停止於第 11 epoch）後：

- 6 個 DN 的型別 bias 漂成負值（−0.006 ～ −0.018），電位全程為負，`relu` 後活性為 0。
- 動作固定 `[0.25, 0]`（`v_max·sigmoid(forward_bias)`、`omega_max·tanh(0)`），與視覺輸入無關；換左右眼影像，角速度差為 0。
- 動作損失對 `core.raw_alpha / raw_tau / bias`、`readout.turn_weight / forward_weight` 的梯度全為 0（dead ReLU），BC 與 PPO 無法修復。
- 原始程式（`git show HEAD`）行為相同，與效能修改無關。
- 記憶體中把 DN bias 設為 0.01，DN 恢復活性、梯度恢復非零；只設為 0 則仍全零（PyTorch 的 `relu'(0)=0`）。

推測成因（需在第 4 步驗證，不得未經驗證就寫進報告）：預訓練 decoder 不需要經過 DN，activity penalty（`activity_coef`）與其他梯度把 DN 推到負電位後，DN 再也收不到修復梯度。

## 2. 限制（必須遵守）

1. 六個連接體變體採**同一方案**；GRU 不受影響但測試須照常通過。
2. 有效參數數不變，或仍在有效 VO 的 ±5% 內（#77：BC optimizer lr>0 且 requires_grad，含 B3；`assert_parameter_budget(..., effective_counts=...)`）；若參數數改變，須重跑 `scripts/prepare_experiments.py` 並更新 status.md 的 declared／zero_lr／effective 預算表。
3. 不改圖、號誌（Dale）、突觸數、edge mask；`test_training_cannot_change_dale_sign_or_create_edges` 等既有不變量必須通過。
4. 初始遞迴仍須收縮（`test_initial_recurrence_is_contractive_for_large_synapse_counts`）。若改變非 DN 神經元的 rate function，須重新論證譜半徑初始化。
5. 保留效能修改的介面：`sense()` / `forward_code()` 分離、`adjacency_cache()`、`MaskedRateCore` 的 `torch.lerp` / `torch.addmm` 子步寫法、批次 TBPTT（`recovery.fit_sequence_epochs`）。不要退回逐步單筆計算。
6. 部署語意不變：`policy(obs, state)` 仍是 `forward_code(sense(...))` 的組合，ROS 節點不需改動。
7. 修改前先確認工作樹已 commit（目前有大量未 commit 的效能修改），每個步驟獨立 commit。

## 3. 候選方案

| 方案 | 內容 | 優點 | 風險 |
| --- | --- | --- | --- |
| A. DN 平滑 rate（建議主方案） | 只對運動 DN 用 `softplus(β·v)/β`（β 例如 10、設定鍵 `motor_rate_beta`）取代 `relu`，同時用於讀出與 DN 的遞迴輸出；其他神經元維持 ReLU | 負電位仍有梯度，DN 不會永久死亡；改動局部；不新增參數 | DN 靜息時有小的正輸出（`ln2/β`），需確認不破壞收縮與左右對稱 |
| B. DN 正工作點 | DN 型別 bias 初始化為正值（例如 0.01），並以 `floor + softplus(raw)` 參數化保持為正 | 最接近稽核中已驗證的干預 | 只靠 bias 為正不保證不被輸入壓到負電位；仍可能死亡 |
| C. activity penalty 排除 DN | `activity_coef` 項不計入運動 DN | 去掉一個把 DN 推向 0 的力 | 單獨使用不夠：沉默後仍無梯度 |
| D. 預訓練加入動作路徑輔助項 | 預訓練 decoder 也讀 DN，或加小權重的「DN 須保持活性」項 | 防止預訓練留下無效運動路徑 | 改變預訓練目標的科學解釋，需在報告揭露 |

建議：**A 為主，加上 C**；B、D 只作為對照實驗。若 A＋C 未通過驗收，再評估 A＋B。不採用「改成線性 DN 讀出」：輸入若仍是六個零活性，問題不變。

## 4. 執行步驟

1. **重現（不改程式）**：寫 `scripts/dn_health.py`，輸入 actor artifact，用真實圖與 held-out 種子（`validation.environment_seed` 範圍，不可用訓練種子）回報：
   - 每個 DN 的正活性步數比例、電位分布；
   - 左右眼影像互換造成的角速度差（中位數、最大值）；
   - 動作損失（以教師動作為目標）對 `core.*` 與 `readout.*` 參數的梯度範數。
   以現有沉默 checkpoint（`runs/_stale/pretrain_V_seed_0_vision_interrupted_2026-10-15` 不完整，可直接重跑縮小版 V 預訓練得到）確認腳本能偵測出問題。
2. **實作 A＋C**：
   - `MaskedRateCore` 增加 motor mask buffer 與 rate function：`rates = where(motor, softplus(β v)/β, relu(v))`；dense 子步仍是 `addmm` + `lerp`。
   - `ConnectomePolicy.forward_code` 回傳的 `activity` 與 `MotorReadout` 輸入使用同一個 rate function。
   - VO-late 兩個分支都套用。
   - `activity_coef` 項排除運動 DN（pretrain、BC、PPO 三處的 activity loss 共用一個 helper，避免不一致）。
   - 新設定鍵加入 `DEFAULT_CONFIG` 與 `configs/model/variants.yaml` 的 `common`。
3. **單元測試**（加在 `python/tests/test_models.py`）：
   - DN 電位全為負時，動作對 `turn_weight`、`core.bias`（DN 型別）與上游 `raw_alpha` 的梯度非零；
   - 非 DN 神經元的動態與修改前完全相同（與舊公式比對）；
   - dense / CSR 一致、cached / uncached 一致、`forward == forward_code ∘ sense` 仍通過；
   - 初始化仍收縮。
4. **小規模端到端驗證**（必須在 GPU 上實跑，不得只看單元測試）：VO seed 0，視覺預訓練 60k → 嗅覺預訓練 60k → BC（teacher 30k、DAgger 2 輪 × 15k）→ PPO 200k。每個階段結束都跑 `dn_health.py`。對照組：同設定、未修正版本。
5. **驗收標準**（全部滿足才算完成，數字寫進 status.md）：
   - 預訓練與 BC 之後，6 個 DN 中至少 4 個在 ≥10% 的 held-out 步數有正活性；
   - 左右眼互換造成的角速度差中位數 ≥ 0.05 rad/s（未修正版本為 0）；
   - BC 的 held-out 角速度誤差低於「固定輸出 0」基準至少 20%；
   - 動作損失對中央參數的梯度範數 > 1e-6；
   - `validate-pretrain` 門檻仍通過；
   - 全部測試、`ruff check`、`ruff format --check` 通過。
6. **文件**：更新 `MODEL_NOTES.md`（DN rate function 的定義與理由，報告需揭露它不是生物學測量）、`TRAIN_NOTES.md`、`docs/status.md`（把已知問題第 0 點改為已修並附數字）。把 `dn_health.py` 加入 `scripts/run_experiments.py` 的階段檢查：DN 健康度未過門檻時停止該組，避免再浪費數天訓練。
7. **重新產生設定**：`python scripts/prepare_experiments.py --algorithm ppo --device cuda`，確認六變體參數預算。

## 5. 交付

- 每步一個 commit，訊息說明改了什麼與驗證結果。
- 一份簡短報告（附在 status.md 或 `runs/dn_fix/REPORT.md`）：修正前後 `dn_health.py` 數字對照、端到端小規模結果、未解決的風險。
- 若 A＋C 未達驗收標準，不要自行擴大範圍改模型；記錄結果並停下等待決定。
