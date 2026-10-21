# C10 與 protocol100

本驅動不修改 C9 的結果或判定。正式候選只有 C10：V、seed 0、BC 起跑、
500,000 環境步、暖身 10 次 update；相對 `ppo_bc.yaml` 只將
`train.core_bias_learning_rate` 設為 0.001（執行路徑 `output`／`init_artifact` 除外）。
`manifest.json` 記錄差異；smoke 的縮小設定另外列出，不冒充正式候選。

所有命令在 repo 根目錄執行：

```bash
export BC_ACTOR=../flytrack-reward/runs/t3r/bc/actor
export MEASURE_REPO=../flytrack-odor
# MEASURE_PY 預設為 $MEASURE_REPO/experiments/odor_chain/measure.py。
# MONITOR_SH 可選；不要在 CPU smoke 設 GPU monitor。
CUDA_VISIBLE_DEVICES='' SMOKE=1 OUT=runs/c10_smoke bash experiments/ppo35/run_c10.sh
# 再跑同一命令會驗證並略過全部六個階段。
```

完成 CPU smoke 後，正式 GPU 命令供操作者執行：

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE=0 OUT=runs/c10 bash experiments/ppo35/run_c10.sh
```

正式階段強制使用 CUDA 且 device name 含 `4060`，不會自動退回 CPU。
訓練採非決定性模式（同 C9），子程序環境移除 `DETERMINISTIC` 和
`CUBLAS_WORKSPACE_CONFIG`，不呼叫 `enable_deterministic()`；`train_runtime.json`
記錄 `deterministic: false`。評估子程序固定 `DETERMINISTIC=1`、
`CUBLAS_WORKSPACE_CONFIG=:4096:8`，評估器呼叫 `enable_deterministic()`。
CPU Python 子程序套用 6 GiB `prlimit`。
smoke 的 PPO 是 8 步、每情境 1 回合、episode 0.2 秒、DN 8 步；永遠不通過正式驗收。
smoke 與正式 OUT 禁止混用；smoke 也禁止使用預設正式 `runs/c10`。

順序為 C10 PPO → BC 評估 → C10 評估 → C10 DN → C9 評估 → 驗收。
BC 和 C9 各 100 × 4 回合共 800 回合，C10 另外 400 回合；三者每情境均使用
`1600100000 + 0..99`，相同 seed 逐回合配對。`eval100.py` 只跑傳入的 actor，
不讀 `FIXED` 或 `ENV_OVERRIDES`；每情境保存 summary 和含 `environment_seed` 的 episodes。
只有未量測 latency、沒有重捕事件的 reacquisition 轉為 null；其餘非有限數值報錯。

GPU 預算按**整次獨立呼叫的牆鐘**計算，包含載入、worker 啟停、輸出與程序清理：

| 預算 | 階段 | 總額 |
| --- | --- | --- |
| (a) | `bc_eval` + `c9_eval` | 1,800 秒 |
| (b) | `c10_train` + `c10_eval` | 3,600 秒 |

每個 actor 的純評估耗時另存 JSON 的 `actor_wall_s`，總預算則以 `stages.tsv` 為準，
不會把 BC 的秒數算到 C10。失敗或中斷的嘗試也扣款，續跑不會重置預算。
每階段 TSV 欄位為 `stage / end_utc / exit / wall_s`；CPU smoke 同樣記錄時間，
但不當成正式 GPU 預算。每次 GPU 呼叫使用 GNU `timeout`，依剩餘時間扣除兩秒
TERM→KILL 清理寬限後啟動；不足兩秒就不啟動。timeout 後本階段與其後階段
都標示 `NOT RUN`，寫入 `budget_status.json`、`acceptance.json`、`summary.md`，exit 0。
其中 timeout 嘗試本身仍有非零 exit 與實際秒數的 TSV 紀錄。耗盡的 OUT 不再啟動 GPU。

建議預設 **LANES=100、WORKERS=12**（可在新的 OUT 開始前覆寫）。舊資料按回合數
與 deterministic 約兩倍推算，LANES=30 的 BC+C9 評估約 40 分鐘，超過 30 分鐘。
100 lanes 讓各情境只需一批同步 rollout，減少 GPU 推論批次；12 workers 沿用 C9
的 CPU 配置。每次只保留一個情境的 worlds，控制記憶體用量。這是未經 GPU 驗證的
起始值，非速度或顯存保證；worker 重啟、CPU 模擬與 deterministic gather 的顯存仍
可能成為瓶頸。正式速度／顯存由 GPU 操作者量測，硬預算無論如何都由 timeout 執行。

C9 評估執行或略過前，核對
`runs/t3c_pack/t3c_c9_7b6a722/trained_actor_SHA256SUMS` 中所有 `ppo_bc/actor/`
項目，且須覆蓋 model card 和 card 所列的全部檔案；任一不一致就報錯。
不執行 direct 組。C9 DN 使用已存在的 `runs/t3c/dn_ppo_bc.json`（CPU、200 步、
seed 1610000000）；該 JSON 的雜湊綁定到新 manifest，不重算或修改 C9 資料。

`accept100.py OUT` 只讀現有 JSON：四情境碰撞率各 ≤ BC+0.05、normal 成功率
≥ max(0.20, BC)、normal bearing 的候選勝回合 > BC 勝回合且雙尾精確二項
p < 0.05（平手剔除）、兩個 DN slope median ≥ 0.01。JSON 與 summary 列出
各條件的數字、門檻、PASS／FAIL。p 用整數與 Fraction 精確計算，再轉浮點作顯示；
有限回合的公平二項 p 是二進分數，無法恰等於 1/20，測試因此直接驗證 0.05 的
決策邊界。BC 本身無法定義、summary 與逐回合值不符或 seed 配對不全都報錯。

來源、BC actor、外部 measure 工具及 fixtures、環境變數、實際 YAML 都綁定於 manifest。
完成階段有輸出 SHA-256 receipt，略過前重新檢查內容與語意；PPO actor 還須逐 tensor
等於最終 checkpoint。未完成 PPO 有 checkpoint 即加 `--resume`。同 OUT 有 flock
避免重複執行；INT／TERM 清理整個程序群組與 monitor。若遭 SIGKILL／斷電，續跑會
將 `active_stage.json` 起始後的經過時間保守計入原預算（包含停機時間，上限為當時
剩餘預算），因此可能提早耗盡，避免憑空補回預算。

```bash
touch runs/c10/STOP
# 在下一階段邊界乾淨退出，保留 checkpoint。
rm runs/c10/STOP
# 同環境、同 SMOKE、同 OUT 重跑原命令。
```

驗證：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=python .venv/bin/python -m pytest -q \
  python/tests/test_ppo35_driver.py python/tests/test_ppo35_protocol100.py < /dev/null
.venv/bin/ruff check experiments python/tests
.venv/bin/ruff format --check experiments python/tests
```
