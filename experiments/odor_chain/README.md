# #4 odor chain 重複實驗

適用 `exp/4-odor-gain`，模型／訓練程式依賴基底 commit `753403a`。
執行前須取得**包含本目錄的後續 commit**；由 Claude commit、push 後把該 SHA
補到 #4 的認領留言。只有 `753403a` 還沒有可攜驅動；main 的 decoder PR 也不足以跑 G3。

從 repo 根目錄執行，使用專案 `.venv`（Python 3.12、PyTorch 2.9.1、NumPy、
Gymnasium、PyYAML），以及 repo 內的 graph、frontend、sensor contract。
正式訓練使用 CUDA，需可用的 `systemd-run --user`，以 cgroup 限制記憶體；
CPU smoke 使用 `prlimit`，上限 6 GiB，並隱藏 GPU。

## seed 1 認領命令

G0、G1、G2 為一個認領單位，請在同一台主機依序跑完各組的 vision → odor → BC：

```bash
SEED=1 CHAIN_GROUPS="G0 G1 G2" bash experiments/odor_chain/run.sh
```

4060 上每組約 1.5 小時，三組合計約 4.5 小時；4070 依實際結果回報。
正式訓練／診斷預算和 seed 0 相同。G0 使用模板原值；G1/G2 的 odor gain
分別為 10/100，pretrain 凍結 core bias 並使用零初始化 population-RMS decoder。
G3 的 DN floor=0.01 保留供探索，這次 seed 1 不跑 G3。

| 環境變數 | 預設值 | 用途 |
|---|---|---|
| `SEED` | `0` | 所有訓練階段的 seed |
| `CHAIN_GROUPS` | `G0 G1 G2 G3` | 選取並排序組別，避免 bash 保留的 `GROUPS` |
| `OUT` | `runs/odor_gain/chain_seed${SEED}` | repo 根目錄下的輸出目錄，必須位於 `runs/` |
| `WORKERS` | `auto` | 同 CLI：CPU 數減一，最多 8；可設 `0` 關閉子程序 |
| `SMOKE` | `0` | `1` 改用 CPU、最小預算，寫入 `OUT/smoke/` |

seed 1 正式輸出為 `<runs>/odor_gain/chain_seed1/`，包含各組／階段、
`configs/manifest.json`、`measure/`、`logs/`、`summary.md`、`summary.json`。
`measure/<group>_<stage>.json` 是 modality_sensitivity 與 DN 工作點輸出。
隨附 `fixtures/` 保存原 seed 0 的固定 200-step 診斷輸入，診斷 seed 不隨訓練 seed 改變。
只跑部分組別也能彙總；缺 G0 的比較或缺 G1/G2 的逐位元比較會記為 null／「—」。

## 停止與續跑

```bash
touch runs/odor_gain/chain_seed1/STOP
```

目前階段／量測完成後在下一個步驟停止，exit 0，保留 STOP 與 `stopped.txt`。
同一個 STOP 也會停止該 OUT 的 smoke。移除後用完全相同的命令續跑：

```bash
rm runs/odor_gain/chain_seed1/STOP
SEED=1 CHAIN_GROUPS="G0 G1 G2" bash experiments/odor_chain/run.sh
```

完成的階段／量測會跳過；有 `training_checkpoint.pt` 的未完成訓練使用 `--resume`；
沒有 checkpoint 的未完成階段從頭開始。續跑時保持程式、設定、輸出路徑與組別不變；
manifest 不相符會拒絕覆寫，可另指定新的 OUT。

## CPU smoke

```bash
CUDA_VISIBLE_DEVICES='' SMOKE=1 SEED=1 CHAIN_GROUPS="G0 G1" bash experiments/odor_chain/run.sh
```

每階段只收 24 steps、1 epoch；pretrain 保留完整 episode，最多 12 秒。
decoder validation 每組／模態 1 episode；BC normal／occlusion 各 2 episodes，
各跑 intact 和 odor lesion；DN health 使用 1 episode、16 steps，modality_sensitivity
與 DN 工作點量測各使用固定輸入的前 8 steps（正式仍為 200）。
最小訓練的指標不代表正式實驗結果，decoder／DN gate 未通過仍保留在報表中。

驗證強制中斷後續跑、STOP、再續跑的完整流程（OUT 必須是新目錄）：

```bash
SEED=1 CHAIN_GROUPS="G0 G1" bash experiments/odor_chain/smoke_test.sh
```

此測試預設輸出至 `<runs>/odor_gain/chain_seed1_resume_smoke/smoke/`。

## 打包交回

請交回各組的 `metrics.json`（包含 validation、DN health）、`training_metrics.json`、
`measure/` 的 modality_sensitivity 輸出、episode 結果、所有 log、設定、
`summary.md` 與 `summary.json`。不用傳權重或訓練資料。

```bash
bash experiments/odor_chain/pack.sh runs/odor_gain/chain_seed1 cweiting731
```

產生 `<runs>/odor_gain/explore4_seed1_cweiting731.tar.gz` 與同名 `.tar.gz.sha256`，
排除所有 `datasets/`、BC 的 `dataset/`、`*.pt`、`actor/`。打包 smoke 時將第一個參數改成該次的
`.../smoke`。回報主機／GPU、實際時間、所用 commit，並附上這兩個檔案。
