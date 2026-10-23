# C11 執行 driver

正式 C11 沿用 `runs/c10/ppo_c10.yaml`：V、seed 0、BC 起跑、500,000 步、
core bias LR 0.001、readout weights LR 0.01、critic warmup 10 updates。
唯一實驗差異是新增 `train.readout_bias_learning_rate: 0`；另一個允許差異為 output。
比較設定前會把絕對路徑轉成等價的 repo 相對路徑，所有新 YAML、JSON、收據及紀錄
均使用相對路徑。smoke 的縮小設定另外列在 manifest，不當成正式結果。

正式執行前，由操作者將對應 `MAIN_REF`（預設 `origin/main`）的來源檔案雜湊清單
放入 `runs/c11/main_python.sha256`，每行是 `SHA256  python/相對檔名`，
並以 `MAIN_COMMIT` 或 `runs/c11/main_python.commit` 提供相應的完整 40 位 commit。
driver 記錄 ref、commit 及清單本身的 SHA256，全程不呼叫版本控制命令。
逐檔比對也會核對完整清單，排除 `python/tests/test_ppo35_*.py` 和執行時產生的
`__pycache__`／bytecode。正式清單缺失、不符、缺 commit 均停止；smoke 缺清單或
不符只警告。正式評估 seed 公告仍由操作者在 Issue 完成。

`BC_ACTOR`、`MEASURE_REPO`、`MEASURE_PY` 預設取自 C10 manifest，BC 的所有檔案
另與 C10 SHA 核對。可用同名環境變數指定等價來源。DN 沿用原 measure 程式、fixtures、
200 步及 seed 1610000000（smoke 8 步），優先載入本工作樹的 Python 套件，
使它能讀取合併後模型卡新增的設定鍵。沒有變更既有程式或測量資料。

CPU smoke 在 repo 根目錄執行：

```bash
PYTHONPATH=python:experiments/ppo35 .venv/bin/python -m pytest -q experiments/ppo35/c11 < /dev/null
.venv/bin/ruff check experiments/ppo35/c11 < /dev/null
.venv/bin/ruff format --check experiments/ppo35/c11 < /dev/null
SMOKE=1 bash experiments/ppo35/c11/run_c11.sh < /dev/null
SMOKE=1 bash experiments/ppo35/c11/run_c11.sh < /dev/null
touch runs/c11_smoke/STOP < /dev/null
SMOKE=1 bash experiments/ppo35/c11/run_c11.sh < /dev/null
```

smoke 強制 CPU、1 env、rollout 12、共 36 步（三次迭代）、每情境 1 回合、
每回合 0.2 秒；快照每迭代一個邊界。protocol smoke seed 1600000000，
snapshot smoke seed 1600001000。任何和兩個正式區塊重疊的 smoke seed 會在訓練前報錯。
正式 protocol100 是每情境 100 回合、seed 1600200000–1600200099；
快照和 BC 第 0 點皆固定 20 個 normal 回合、seed 1600300000–1600300019。
沒有執行正式設定的 smoke 結果永遠 `accepted: false`。

執行順序是 `c11_train` → `c11_eval` → `bc_eval` → `dn_c11` →
等待 snapshot chain → `acceptance` → `summary`。
訓練完成後另啟動 `snap_build` → `snap_eval` → `snap_dn` 的 CPU chain，
與兩個 GPU 評估同時跑；chain 使用 nice 10、最多兩顆 affinity CPU、單執行緒、workers=0。
因此三個快照階段不用 GPU，並分別記錄耗時。CPU Python 子程序限制 6 GiB。

sidecar 是獨立 CPU 程序，每秒讀 progress；以 `os.link` 留住當時 checkpoint inode，
從 link 的 checkpoint 核對實際步數，才命名 `snapshots/step_<實際步數>.pt`。
跨多個邊界只保留當時可取得的一份，較早的邊界記入 `missed_boundaries`。
快照錯誤留在 index，訓練繼續；sidecar 不讀寫訓練記憶體或訓練輸出。
啟動前等待 sidecar 就緒，結束時額外取最終快照，兩者耗時都算入 train。
重建沿用 `prepare_policy`／`save_actor`，嚴格載入 checkpoint actor state_dict；
重建結果和最終訓練 actor 逐 tensor 比較原始位元（包括正負零），
不比較帶有序列化容器資訊的 `.pt` 檔案 bytes。
訓練完成也會驗證 turn／forward bias 與 BC 的位元完全相同。

GPU 共用硬預算 4500 秒，涵蓋三個 GPU 階段的所有失敗、中斷及成功嘗試，
包含 train 的 sidecar。timeout 依當時剩餘時間設定並保留五秒清理餘裕。
不足預算時停止剩餘工作，`acceptance` 和尚未完成的階段均記為 `NOT RUN`。
CPU 階段獨立計時。`active_<stage>.json` 支援異常中止後補記帳，
GPU 最多補記當時剩餘預算，重跑不會重置 GPU 秒數。
同 OUT 使用 flock 防止重複 driver；preflight 另檢查其他 flytrack 訓練程序。

每個完成階段以 `*.complete.json` 綁定檔案 SHA；重跑會重驗 SHA 和內容才略過。
未完成訓練有 checkpoint 時使用 `--resume`。
STOP 在啟動前、階段邊界及執行期間皆可乾淨停止，保留續跑資料。
SIGINT／SIGTERM 清理程序群組與背景工作。
`MONITOR_SH` 沿 C10 呼叫介面啟動；smoke 不啟動 monitor。

`summary.json`／中文 `summary.md` 只列數值、門檻、PASS／FAIL 和最終判定，
驗收直接使用 `accept100.compare` 的四項條件。
追蹤比例取 normal 回合平均，ω 中位數取所有實際 applied omega 樣本，
turn_bias 欄為最大絕對值。C10 數字只讀 `runs/c10`；四門檻取自現存
`acceptance_c10_only.json`，C10 僅列終點。既有 C10 沒有儲存 omega 中位數，
該欄以 JSON null／表格「—」呈現。
