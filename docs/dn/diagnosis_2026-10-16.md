# 運動 DN 問題：小規模驗證與診斷結論（2026-10-16）

> **注意（2026-10-18，#4）：**本文件所有 VO 結果的動作都完全不受氣味影響（去掉氣味時 Δω 為 0），實際上只用了視覺；conflict 情境的數字因此等於 normal。VO、V-only、O-only 的比較與 lesion 結論在 #4 修好前都不成立。

> **2026-10-17 後續：**本文件第 4、6 節「BC 前放大 DN 增益」的建議已測試，單獨使用無效。真正的主因是訓練與評估環境：目標永遠以固定方向繞圈，固定左轉策略就能 100% 成功。見 [scenario_test_2026-10-17.md](scenario_test_2026-10-17.md)。

本文件記錄 A＋C 修正（見 [fix_A_C_2026-10-15.md](fix_A_C_2026-10-15.md)；原始問題見 [audit_2026-10-15.md](audit_2026-10-15.md)）的端到端小規模驗證，以及之後兩項唯讀診斷。**驗證沒有通過**；依 [dn_fix_plan.md](../dn_fix_plan.md)，沒有擴大修改範圍，停下來等待決定。

> 兩份 10-15 報告是從 `runs/dn_audit/`、`runs/dn_fix/` 複製進來的；其中指向同目錄 JSON／腳本的相對連結在 repo 中不存在（`runs/` 不納入版本控制）。

## 1. 小規模端到端驗證

設定：VO、seed 0，`run_experiments.py --variants VO --seeds 0 --jobs 1 --workers 7`。預算：視覺 pretrain 60k → 嗅覺 pretrain 60k → BC（teacher 30k＋DAgger 2×15k）→ PPO 200k。兩組：修正版（`motor_rate_beta: 10`）與對照組（`null`，原 ReLU）。

| 檢查點 | 修正版 | 對照組 |
|---|---|---|
| 視覺 pretrain 後 | 通過：6/6 DN 有效活化，中央梯度約 1e-3；validate-pretrain 方位誤差 0.33° | **失敗**：0/6 DN 活化，梯度 0，鏈停止 |
| 嗅覺 pretrain 後 | 通過 | — |
| BC 後 | **失敗** | — |
| PPO | 未執行（runner 依設計停止） | — |

BC gate 失敗細項（`dn_health` 預設 4 回合 × 64 步，數據：[data/validation_bc_dn_health.json](data/validation_bc_dn_health.json)）：

- 換眼角速度差中位數 0.0169 rad/s（門檻 ≥0.05）。
- ω MAE 0.0652，對固定零輸出 0.0254，改善 −1.57（門檻 ≥0.2）。
- 動作幾乎固定：v≈0.169 m/s、ω≈0.09 rad/s（固定的轉向偏差）。

BC 各輪：round 0（teacher）14 epoch、best 11；round 1 4 epoch、best 1；round 2 跑滿 30 epoch 上限——驗證損失每 epoch 只降約 0.1%，`early_stopping_min_delta: 0` 時永遠不觸發提早停止。整條修正版鏈 4774 秒，其中 BC 2074 秒。

## 2. 逐層資訊探針（`scripts/diagnostics/dn_layer_probe.py`）

方法：24 個 held-out 回合（環境種子 1.5e9 起）由 teacher 駕駛，burn-in 16 步後量 200 步，共 4800 步。對每一層以 ridge 迴歸解碼 teacher ω（偶數回合訓練、奇數回合測試），R² 越高表示該層越多方位資訊。另記錄每層沉默比例、時間變動量。數據：[data/layer_probe_3models.json](data/layer_probe_3models.json)、[data/layer_probe_presyn.json](data/layer_probe_presyn.json)。

| 層（神經元數） | 嗅覺 pretrain 後 R² | BC 後 R² | BC 後沉默比例 |
|---|---:|---:|---:|
| 感覺碼（80） | 0.87 | 0.87 | — |
| 視覺種子（465） | 0.51 | 0.42 | 44%（pretrain 後 0%） |
| DN 的直接上游（129） | 0.68 | 0.46 | 79% |
| 匯流神經元（4） | 0.63 | 0（全部不變） | **100%** |
| 運動 DN（6） | ≈0 | 0.30 | 0% |
| **DN 左右差（讀出實際使用的訊號）** | **0.66** | **0.04** | — |

量級：DN 活性隨時間的變動只有平均值的約千分之一（嗅覺 pretrain 後：平均 0.105、變動 6e-5；BC 後：平均 0.50、變動 8e-4）。

參數變化（嗅覺 pretrain → BC）：

- DN 偏置約 0.06 → 0.45；讀出 `turn_weight` 0.1 → 約 0.46，`forward_weight` 0.1 → −0.49。
- 連結強度 α 平均 2.568e-4 → 2.589e-4（幾乎不動）。設定 `alpha_init: 0.01`，但 `init_spectral_radius: 0.9` 把它壓到 2.6e-4（約 40 倍）。

## 3. 增益反事實測試（`scripts/diagnostics/dn_gain_counterfactual.py`）

方法：同上 24×200 步。teacher 軌跡與凍結前端都和 actor 無關，因此感覺碼（原始與換眼）只收集一次，各條件以批次重播中央網路。只在記憶體中把選定連結的突觸數乘上 k，不訓練、不寫回。數據：[data/gain_counterfactual.json](data/gain_counterfactual.json)。

嗅覺 pretrain 後，只放大**接進 DN 的連結**：

| k | DN 左右差變動 | 左右差 R² | 左右差的換眼變化 |
|---:|---:|---:|---:|
| 1 | 0.0003 | 0.66 | 0.002 |
| 100 | 0.020 | 0.66 | 0.14 |

- 只放大接進 DN 的連結：幅度隨 k 線性放大、資訊不變、數值穩定（k=100 仍有限）。
- 連 DN 上游一層一起放大：資訊被破壞（k≥3 時 R² 降到約 0）。
- 全網放大：k=1.5 已爆炸，k≥2 非有限值——`init_spectral_radius` 對整個網路確實必要。

BC 後，只放大接進 DN 的連結：左右差 R² 在 k=1～100 都只有 0.02～0.05；k=3 時換眼差 0.058 可以過門檻，但只是放大雜訊，資訊並沒有回來。

## 4. 結論

1. A＋C 解決了 DN 沉默與梯度阻斷（對照組證實原 ReLU 版本在 pretrain 後就全沉默）。
2. **嗅覺 pretrain 結束時，方位資訊已經傳到 DN 左右差（R² 0.66），但幅度只有約 3e-4。** 原因是 spectral radius 上限把每突觸權重壓到 2.6e-4，讀出又以 0.1 的權重讀取。
3. BC 開始時，這麼小的訊號產生的轉向梯度極小，優化器改以調高 DN 偏置與讀出權重去配合平均動作；工作點移動後，匯流神經元全部沉默、視覺種子 44% 沉默，DN 左右差的資訊從 0.66 掉到 0.04。
4. 因此修正必須在 **BC 開始前**讓 DN 左右差有足夠幅度。只放大接進 DN 的連結已被證實穩定且保留資訊。
5. 計畫的方案 B（正 DN 偏置）可以排除：BC 已經自行把偏置推到約 0.45，結果沒有改善。方案 D（輔助損失）單獨使用大概不夠，因為瓶頸在權重量級。

## 5. Gate 本身的問題

同一個 BC 模型：`dn_health` 預設的 4×64 步量得 ω 誤差「比零輸出差 157%」；24×200 步則量得「比零輸出好 30%」，可通過 ≥0.2 的門檻。held-out 回合裡 teacher ω 平均約 0.053 rad/s（不為零），所以零輸出是很弱的基準，固定偏差的策略也能通過。建議改用與 teacher ω 的 R² 或相關係數，並增加量測步數。

## 6. 待決定的下一步

1. **最小修正（建議）：** 接進 DN 的連結初始化為目前的 30～100 倍，也就是不受 spectral radius 上限壓縮（或在讀出層對 DN 左右差做增益／標準化）。從既有嗅覺 pretrain 模型直接重跑 BC（約 35 分鐘），看 DN 左右差 R² 是否保住、轉向是否學起來。
2. **修改 BC gate：** 改用 R²／相關係數，量測改為 24 回合 × 200 步。
3. **BC 提早停止：** `early_stopping_min_delta` 由 0 改為約 0.5% 相對改善。

這些都會改到模型設計或驗收標準，需要組內決定。

## 重現方式

```bash
# 逐層探針（每個 artifact 約 2 分鐘，GPU）
PYTHONPATH=python .venv/bin/python scripts/diagnostics/dn_layer_probe.py out.json <actor 目錄> [...]
# 增益反事實（兩個 artifact 合計約 2 分鐘）
PYTHONPATH=python .venv/bin/python scripts/diagnostics/dn_gain_counterfactual.py out.json <actor 目錄> [...]
```

驗證用的 artifact 沒有放進 repo（`runs/` 不納入版本控制）；JSON 裡的 `<validation>/` 代表本機的驗證輸出目錄。
