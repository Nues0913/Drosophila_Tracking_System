# DAgger-v2 訓練

在 repository 根目錄執行。六個變體共用 trainer。RW2 和 ER2 的結構 seed 同訓練 seed。Trainer 先載入 `artifacts/input_norm_v2/input_norm.npz`，再建立 policy，由模型用擬合後的中性 code 校準靜息狀態和讀出偏置。

```sh
PYTHONPATH=python python -m flytrack_v2.train.dagger \
  --config configs/v2/train/dagger.yaml --variant VO2 --seed 0 \
  --host 4060 --device cuda --out runs/v2/VO2/seed_0/dagger
```

`--device cuda` 使用 PyTorch 的目前 CUDA 裝置，也可用 `--device cuda:0` 指定編號。`--override` 接受一個或多個 `路徑=YAML值`，可重複使用。下方是小型 VO2 CPU 匯出檢查。少量更新可能使最終速度標準差低於門檻，此時仍會匯出診斷用產物，並回傳 exit 3。

```sh
OMP_NUM_THREADS=1 PYTHONPATH=python python -m flytrack_v2.train.dagger \
  --config configs/v2/train/dagger.yaml --variant VO2 --seed 0 \
  --host 4060 --device cpu --out runs/v2/vo2-cpu-smoke \
  --override 'rounds.beta=[1.0]' envs_per_instance=2 rounds.env_steps_per_round=400 \
             replay_ratio=1 batch_sequences=8 held_out.every_updates=1
```

## 收集與更新

- 四個 batched world 各跑一組世界與目標運動的組合，預設每組 256 列。各列的情境依最大餘數分配。同餘數依 normal、occlusion、noise、conflict 的順序優先。每個環境都有全域列號，初始、強制與自動重設都用 `episode_seed(seed, env_index, episode_index) | (1 << 40)`。初始序號為 0，每列進入下一回合時遞增。
- `desync_first_episode: true` 預設在首回合錯開各列相位。每列以訓練 seed 衍生的獨立 generator 均勻抽取 `r_i ∈ [0, episode_steps)`。若 `r_i > 0` 且首回合仍在進行，執行第 `r_i` 步後強制重設。`r_i = 0` 時首回合自然結束，後續回合也依世界的碰撞與時間上限自然結束。可用 `--override desync_first_episode=false` 關閉，恢復同步起跑的收集方式。
- 每步先取得 sensory code、學生 mode 與遮蔽 teacher 標籤，再用每列獨立的 Bernoulli 抽樣選擇動作。回合結束會重設 sensory、central 和 teacher 狀態。輪次交界保留這些狀態和世界狀態，後續 rollout 以更新過的學生參數接續。
- 每輪的 CPU replay 保存 float32 code、teacher 動作及中央狀態，另存 reset、遮蔽旗標和世界、運動、情境的列標籤。中央狀態存於全域控制步數的每個 16 步倍數。取樣會在分片內選擇完整視窗，並跨累積分片均勻抽取合格的時間與列組合。序列可以跨回合。
- 每輪至少收集一條完整視窗所需的時間步，之後以整批環境為計數單位。正式設定每輪 733 個控制步，共 750,592 個環境步。更新次數依實際環境步計算，每輪 184 次。前 16 步在 `no_grad` 下重建中央狀態，後 64 步計算 loss。reset 旗標作用於下一步。
- 標籤依 `label_margin` 向動作範圍內縮，再用 `unsquash` 算潛在高斯 NLL。活動懲罰對圖變體使用所有讀出 DN 以外的神經元速率平方平均，GRU2 使用隱狀態平方平均。梯度累積以 micro-batch 的序列數加權。有限的梯度做全域範數裁剪後交給 Adam。Cosine 排程涵蓋四輪，以參數群組的 LR 比例一起下降。
- `hosts.yaml` 設 4060、2060、4070 的 micro-batch 為 128、64、128。若縮小 effective batch 至主機值以下，整批作為一個 micro-batch。實際的 micro-batch 必須整除 effective batch。Activation checkpoint 預設關，並在設定與 model card 記錄實際值。

`replay_ratio=4` 及 held-out 每 50 次更新、至多 256 條序列、保留 5% 是計劃書交辦的 Claude 預設。固定保留的列以 seed 抽取，四輪皆保留同一批列，避免其狀態歷史進入訓練。列數四捨五入且至少保留 1 列。資料量小時，held-out 使用所有可用的不同視窗。

4060 的 smoke 使用 1,024 個同步起跑的環境，回合上限為 60 秒（1,200 步）。Teacher 很少撞牆，正式設定每輪約 36.7 秒，第一輪主要收集起步到 36.7 秒的資料，第二輪才收集回合尾段與下一回合起步。Smoke 每輪 80 步，四輪總共只看到每回合前 16 秒。第一輪 teacher 速度的平均與標準差為 0.254 和 0.055 m/s，第四輪（12–16 秒）則為 0.164 和 0.012 m/s。錯開首回合的結束時間可減少每輪資料集中在同一回合相位的情況。

## 監控與健康檢查

`monitor.jsonl` 記錄 loss、NLL、活動懲罰、梯度範數、LR、log std、更新吞吐、收集吞吐和 CUDA peak allocated bytes。飽和率是任何動作維度的絕對值在內縮前達到 `(1 − 1e-6) × limit` 的比例。飽和步驟的 NLL 份額以這些步驟的 NLL 總和除以全部步驟的 NLL 總和。定期 held-out NLL 於該次更新之前測量。每輪用有效讀值統計左右眼、氣味、風的正規化 `|z|` 最大值與精確的 99.9 百分位。沒有有效讀值的組別記為零。

每輪第一次更新記錄 `episode_time_mean_s` 和 `episode_time_std_s`，分別是該輪所有收集樣本在選擇動作前的回合內時間平均與母體標準差，以秒為單位。新回合的第一筆樣本時間為 0。逐步計數只用於這兩項監控統計。

每輪完成更新後，在 held-out 視窗以 mode 動作探測，並在 `monitor.jsonl` 新增一筆 `round_end=1` 的健康指標紀錄。圖變體每個讀出 DN 的平均速率都須大於 `1e-4`。同一 burn-in 狀態會分別跑原始及左右氣味互換的 code，並記錄兩條分支的平均 `|Δω|`。中間輪次發現 DN 沉默時立即以 exit 3 結束。中間輪次的速度標準差 ≤ `0.02 m/s` 時會記錄指標，整次訓練只在首次觸及門檻時發出警告，並繼續下一輪。

最後一輪的探測是 Stage 1 交界健康檢查。若出現 DN 沉默或速度標準差 ≤ `0.02 m/s`，trainer 會保存最後一輪 checkpoint 與 `failure/` 的診斷資料，仍用 `export_policy` 匯出至 `final/`，再以 exit 3 結束。匯出成功後，`failure/reason.json` 會註明產物已匯出且 Stage 1 健康檢查未通過。健康指標、loss 或梯度出現非有限值也會觸發 exit 3。`failure/` 保存 checkpoint、可用的探測或訓練 batch，以及失敗原因。

## 停止、續跑與匯出

在輸出目錄建立 `STOP`。Trainer 會在輪次邊界及每輪每 50 次更新查閱該檔，保存進度後以 0 結束。移除 `STOP`，以相同設定、主機、裝置、變體和 seed 加上 `--resume`。續跑讀取最後發布的完整或輪內 checkpoint。Checkpoint 保存 RNG、世界與三種 recurrent state，以及首回合的重設排程、已執行的列和收集步數。輪內 checkpoint 也保存下一次更新的位置，並讀取已收集的 replay。要續跑加入錯相設定之前的 checkpoint，請使用 `--override desync_first_episode=false`。Monitor 對齊 checkpoint 的環境步和日誌位置。

每輪 `round_<k>/replay.pt` 只寫入該輪的新資料。原子發布的 `checkpoint.pt` 指示該輪已完成，`progress.pt` 指示該輪可從中間續跑。`config.json` 記錄解析後設定。訓練完成時，`final/` 由 `export_policy` 產生，可直接用 `load_policy` 載入。Model card 的 `provenance.stages` 包含 DAgger 的設定、主機、實際 micro-batch、activation checkpoint 與以實際步數推得的四輪排程。來源版本可由執行環境提供 `GIT_COMMIT` 或 `GITHUB_SHA`，缺值時記為 null。

## 記憶體

正式 VO2 的 296 欄 code、兩欄動作與兩個旗標共約 3.58 GB。以 8,598 欄中央狀態及每 16 步存一次計算，快照另需 6.48 GB。四輪合計 10.06 GB（9.37 GiB）。Replay 逐輪落盤，取樣以 mmap 存取分片，維持 float32 精度。收集和統計時新分片約用 2.52 GB heap，單眼百分位的暫存上限約 0.43 GB。OS 會依可用記憶體保留 mmap 的檔案頁面。

GPU 收集每 32 個控制步批次拷回 CPU，VO2 的暫存約 105 MiB。強制重設在 graph replay 後執行，同一世界、同一步的列合成一個遮罩，並就地更新 graph 使用的張量。每輪收集完會釋放世界的 CUDA graph capture pool，下一輪用同一世界狀態重新 capture。無 activation checkpoint 時，VO2 的 recurrent core 計算圖主項估計在 128 條序列約 3.94 GiB、64 條序列約 1.97 GiB。連同讀出、CSR 工作區、模型緩衝及世界狀態，訓練峰值粗估約為 5–6 GiB 及 3–4 GiB，收集階段的峰值另受 CUDA graph 影響。實測值以 monitor 的 `max_memory_allocated` 為準。
