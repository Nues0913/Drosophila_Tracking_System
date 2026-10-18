# #24：遮蔽氣味 teacher 的本機校準

Refs #24。這是 teacher 的診斷校準，不是訓練結果或感官基準。

- 實作版本：`c43028050dedf43c81308b4983484fd61325b333`，基於 main `8c326f670c522940c66170315a4a0a4630652515`。
- 授權：#24 的 PM Decision（2026-10-18），預設關閉，只實作、測試與 teacher 校準；20:16 UTC 調升為 P1 第一順位。
- 執行端：cweiting731，本機 WSL2 Ubuntu，Python 3.12，CPU 模擬，未執行 GPU 訓練。
- 每組 20 個相同診斷 seeds：1650000000–1650000019；每回合 60 秒。
- 以 `configs/train/evaluate.yaml` 為基礎，完整 `configs/sensor_contract.json`，curved 目標、puff 羽流、domain_randomization=false；仍保留 20 種不同 layout。
- 遮蔽從第 8 秒開始，持續 12 秒或 30 秒，分別佔完整回合的 20%／50%。每組實際有效設定完整保存在同名 YAML 與 JSON 的 config 欄位。

| 遮蔽比例 | 氣味轉向 | 成功回合 | mean_success | mean_collision_rate | 平均方位誤差 |
| --- | --- | --- | --- | --- | --- |
| 20% | 關閉 | 20/20 | 1.00 | 0.00 | 2.216° |
| 20% | k=8 | 19/20 | 0.95 | 0.00 | 7.068° |
| 50% | 關閉 | 20/20 | 1.00 | 0.00 | 2.216° |
| 50% | k=8 | 16/20 | 0.80 | 0.00 | 12.974° |

四組皆完成、exit code 0、沒有任何回合碰撞。碰撞率採用既有 EpisodeMetrics 指標，沒有更改定義。關閉選項時 teacher 仍可讀真值方位，因此其成功率不能代表學生在遮蔽下的能力；開啟氣味轉向後，前進速度仍沿用原有 privileged 規則。

這些小樣本僅用來量出 teacher 在這組條件下的退化。沒有比較其他 gain、沒有啟用 BC 混合、沒有判定 #4 已解決，也沒有據此決定正式訓練參數。

## 實際執行命令

從固定實作版本的獨立 worktree 執行，明確讓 PYTHONPATH 指向該版本的 python/。原始產物在忽略的 `runs/teacher_occlusion/issue24/`；本目錄保存其完整複本供遠端審查。

```sh
export PYTHONPATH="$PWD/python"
.venv/bin/python -m pytest python/tests -q
.venv/bin/ruff check python scripts
.venv/bin/ruff format --check python scripts
```

完整測試：148 passed、7 warnings，33.60 秒。Ruff check 與 format check 通過（74 files）。沒有執行 ROS／Gazebo 或 GPU 訓練，因本變更只涉及 2D teacher。

四組校準使用相同命令，只更換 label：

```sh
label=occlusion_20_on  # 另三組：occlusion_20_off、occlusion_50_on、occlusion_50_off
PYTHONPATH="$PWD/python" .venv/bin/python scripts/calibrate_teacher.py \
  --config "docs/teacher/issue24/$label.yaml" --scenario occlusion \
  --episodes 20 --seed-start 1650000000 --output "runs/teacher_occlusion/issue24/$label.json"
```

JSON 保留逐回合 metrics、完整設定、完成狀態；log 保留每回合進度。`sha256.json` 對照原始複本的內容雜湊。

## 預設相容性及輸入隔離

除了單元測試，另用修改前 d1d6eac 的 env.py **與 teacher.py**（兩者明確載入舊版）對照新版預設與明確 null 兩種路徑。4 種目標運動 × 4 種情境 × 2 個 seeds，共 32 回合、2560 步、97280 個 observation 元素；observation、teacher action、其他 info、reward 和終止旗標逐位元一致。其結果見 `independent_parity.json`。此額外相容性重播是本機獨立驗證，不是 CI 測試。

新增單元測試涵蓋同一份 observation 的物件身分、禁止 plume 真值重採樣、左右方向、濃度尺度不變性、無有效氣味時中性轉向、動作與加速度限制，以及遮蔽前後不變。只有遮蔽期間的 requested omega 被替換；加速度限制仍可能讓轉向反轉需要數步。

## 待審查與 PM 決定

請 MiLong9487 複核 raw ORN 輸入、正規化與 BC 介面，並由 Nues0913 審查與決定合併。訓練是否啟用、gain 與遮蔽混合比例仍由 PM 在 #4 後決定。回滾方式為保持選項 null（預設），不需要變更共享訓練設定。
