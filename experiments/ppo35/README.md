# #35 固定候選 T3 驅動

依賴基底 commit `1af1bdb`（PR #32 head）與本目錄的原生 PPO 修改。
DN 工具依賴 `exp/4-odor-gain` 的 `experiments/odor_chain/measure.py`、同目錄
`common.py`／`fixtures/*.npz` 與其 repo 的 modality sensitivity 工具；manifest
記錄外部工具、Python 原始碼及固定資料的 SHA-256，續跑時須保持一致。
不在此驅動中搜尋候選：只有一個設定，每組 V、seed 0、500,000 環境步。

必要環境變數：

| 變數 | 內容 |
| --- | --- |
| `BC_ACTOR` | BC actor 目錄（含 model_card.json） |
| `PRETRAIN_ACTOR` | vision pretrain actor 目錄 |
| `MEASURE_PY` | exp/4-odor-gain 的 measure.py 路徑 |
| `MEASURE_REPO` | 上述工具的 repo 根目錄 |

選用：`OUT` 預設 `runs/t3c`，`PY` 預設本 repo 的 `.venv/bin/python`，
`MONITOR_SH` 指向 repo 外的 monitor.sh（未設則不監控），`PPO_WORKERS` 預設 auto。
需可用的 `bash`、`prlimit`、`setsid`、PyTorch、NumPy、PyYAML，以及本 repo 的
graph、frontend 和 scenario 設定。Python 的工作目錄一律由驅動設定。

在本 repo 根目錄、相鄰工作樹採以下名稱時可直接執行：

```bash
export BC_ACTOR="$(realpath ../flytrack-reward/runs/t3r/bc/actor)"
export PRETRAIN_ACTOR="$(realpath ../flytrack-reward/runs/t3r/pretrain_vision/actor)"
export MEASURE_REPO="$(realpath ../flytrack-odor)"
export MEASURE_PY="$MEASURE_REPO/experiments/odor_chain/measure.py"
export MONITOR_SH="$(realpath ../flytrack/.claude/scripts/monitor.sh)"
export OUT=runs/t3c
CUDA_VISIBLE_DEVICES=0 SMOKE=0 bash experiments/ppo35/run.sh
```

4060 兩組 GPU PPO 並行預估約 1 小時，加上三個 actor 各 30 回合 × 4 情境評估。
DN 工作點與輸出驗證一律在 CPU，CPU Python 皆以 `prlimit --as=6442450944`
執行。GPU 訓練／評估不設此 CPU 位址空間限制。

候選：base lr=0.001、readout lr=0.01、core.bias lr=0、epochs=4，暖身 10 次
rollout update。暖身只把 actor lr 設為 0，保留非永久凍結組的 Adam moments，
與診斷 hook 相同；暖身後恢復設定的 lr。其餘 PPO YAML 沿用 t3r；
`init_artifact`、`output` 在執行時寫進 OUT 的實際設定，`eval.yaml` 原樣沿用 t3r。
正式 episode 評估沿用 `FIXED='{}' EPISODES=30 LANES=30 WORKERS=12`。

完整 CPU smoke（需相同的必要變數）：

```bash
CUDA_VISIBLE_DEVICES='' SMOKE=1 OUT=runs/ppo35_smoke bash experiments/ppo35/run.sh
```

Smoke 每組 2 個 rollout、共 8 步、1 env、4 步/rollout、TBPTT=2、epochs=1、
critic hidden=8、暖身 1 次、episode 0.2 秒；每情境評估 1 回合，DN 用工具內建的
8 步 smoke。Smoke 永遠 `accepted=false`，即使數值達標也不能當正式驗收。
不得在同一 OUT 混用 smoke 與正式模式。

輸出：`manifest.json`、實際 YAML、`ppo_{bc,direct}/actor/`、各組 checkpoint／
progress／training_metrics、`episode_eval.json`、四份 `dn_*.json`、
`acceptance.json`、`summary.md`、各階段 log，選用監控輸出在 `monitor/`。
每個階段完成及略過前都驗證檔案、來源及數值；完成的 actor 還會與最終 checkpoint
逐 tensor 比較。已存在但驗證失敗的結果會報錯，不當作已完成。
評估器的未量測 latency 與無可重捕事件的 reacquisition NaN 轉成 JSON null；
其餘數值均須有限，成功率／碰撞率必須在 [0,1]。

驗收門檻固定寫在 `check.py`：BC 起跑那組的 **normal** collision_rate ≤0.27、
normal success ≥0.20，且 BC 起跑那組須 DN 不沉默；direct 組另記診斷。0.27 對應 t3r BC normal 的
0.266667；不從本次 BC 評估重算門檻，也不跨情境平均碰撞率。

DN 不沉默定義：DNa01 與 DNa02 各自 `n>0`、電壓／斜率有限、
`0.01 <= slope_median <= 1`。measure.py 的 slope 是
`sigmoid(voltage * motor_rate_beta)`，亦即 motor softplus 對電壓的導數；
負電壓很低時接近零。要求至少 1% 的局部增益；正向高電壓 slope=1 不算沉默。
t3r BC 的 DNa01/DNa02 slope=0.01250177/1，pretrain=0.04353683/1，都通過。
這只是既有量測能支持的工作點檢查，沒有聲稱完成 DN health 的感覺調變／梯度 gate。
第一個候選未通過時，`next_action=return_to_issue_no_parameter_sweep`，回 Issue 討論。

停止與續跑：

```bash
touch "${OUT:-runs/t3c}/STOP"
# 等目前階段完成，驅動在下一個階段邊界乾淨退出，監控也停止。
rm "${OUT:-runs/t3c}/STOP"
# 使用相同環境變數與 SMOKE 值重跑上述 run.sh 命令。
```

兩組 PPO 同時啟動；等 BC 組完成後也會檢查 STOP，此時尚在執行的 direct 組會
收到 TERM。checkpoint 在每次 update 後保存，續跑時存在 checkpoint 就加
`--resume`；已完成的階段會檢查後略過。INT／TERM／階段失敗亦會清理背景 PPO
與監控程序。不要在工作進行中修改來源碼、設定或上游 actor；manifest 和 PPO
recovery identity 都會拒絕使用變更前的輸出。需要不同設定時使用新的 OUT。
