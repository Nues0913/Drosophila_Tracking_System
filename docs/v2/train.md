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

## 模型覆寫

`dagger.yaml` 的 `model: {}` 預設使用 `configs/v2/model/default.yaml` 的模型設定。可在 `model` 下填入要覆寫的 `ModelConfigV2` 欄位，或用 `--override model.<欄位>=<YAML 值>` 設定。CLI 優先於 YAML，例如 `--override model.speed_types=bilateral model.gain_parameterization=bounded` 會選用雙側速度讀出和 bounded gain。未知欄位會報錯。`init_seed`、`structure_seed` 和 `activation_checkpoint` 由 trainer 根據 `--seed`、變體及 DAgger 頂層的 `activation_checkpoint` 決定。

## 收集與更新

- 四個 batched world 各跑一組世界與目標運動的組合，預設每組 256 列。各列的情境依最大餘數分配。同餘數依 normal、occlusion、noise、conflict 的順序優先。每個環境都有全域列號，初始、強制與自動重設都用 `episode_seed(seed, env_index, episode_index) | (1 << 40)`。初始序號為 0，每列進入下一回合時遞增。
- `desync_first_episode: true` 預設在首回合錯開各列相位。每列以訓練 seed 衍生的獨立 generator 均勻抽取 `r_i ∈ [0, episode_steps)`。若 `r_i > 0` 且首回合仍在進行，執行第 `r_i` 步後強制重設。`r_i = 0` 時首回合自然結束，後續回合也依世界的碰撞與時間上限自然結束。可用 `--override desync_first_episode=false` 關閉，恢復同步起跑的收集方式。
- 每步先取得 sensory code、學生 mode 與遮蔽 teacher 標籤，再用每列獨立的 Bernoulli 抽樣選擇動作。回合結束會重設 sensory、central 和 teacher 狀態。輪次交界保留這些狀態和世界狀態，後續 rollout 以更新過的學生參數接續。
- 每輪的 CPU replay 跨累積分片均勻抽取合格的時間與列組合，視窗在同一分片內，可以跨回合。中央狀態存於全域控制步數的每個 16 步倍數。抽樣後按分片、環境列與起點排序讀取，最後還原抽樣順序。Code 和中央狀態在 CPU 還原成 float32，再傳到訓練裝置。
- 預設 `teacher_limit_acceleration: false`（#184 的 Decision）：teacher 的標籤只套用動作的上下限，是 teacher「想要」的動作；被執行時（β 的那一部分）照樣由世界套用加速度限制。設成 true 是 v1 的標籤：依上一步實際套用的動作做加速度限制。學生駕駛時，這會把標籤錨定在學生自己的動作附近（速度每步 ±0.05 m/s、轉向 ±0.15 rad/s），等於教「維持你現在的動作」。沒有這個 key 的舊設定檔和舊 checkpoint 照舊用 true。
- 預設 `readout_normalization: true`。圖變體在第一輪（`β = 1`）使用初始 policy，每四個控制步保存一次兩個世界訓練列的讀出 DN 速率。首輪 replay 保存這些樣本供擬合和續跑使用。收集結束後，trainer 擬合正規化、重設初始讀出權重及校準偏置，然後建立 Adam 與 cosine 排程，再執行第一次更新。GRU2 忽略此設定。 [#137 Decision](https://github.com/Nues0913/Drosophila_Tracking_System/issues/137)
- 每輪至少收集一條完整視窗所需的時間步，之後以整批環境為計數單位。正式設定每輪 733 個控制步，共 750,592 個環境步。更新次數依實際環境步計算，每輪 184 次。前 16 步在 `no_grad` 下重建中央狀態，後 64 步計算 loss。reset 旗標作用於下一步。
- 標籤依各維度的動作範圍內縮。設 `m = label_margin`，速度限制在 `[m × v_max, (1 − m) × v_max]`，轉向限制在 `[−(1 − m) × ω_max, (1 − m) × ω_max]`。內縮後用 `unsquash` 算潛在高斯 NLL。預設 `m = 1e-3` 時，速度的潛在值範圍為 `[logit(1e-3), logit(1 − 1e-3)] ≈ [−6.9, 6.9]`。 預設 `label_latent_bound: 3.0`（#184 Decision、#189 review）：速度的 m = sigmoid(−3) ≈ 0.0474、轉向的 m = 1 − tanh(3) ≈ 0.0049，所以每個標籤的潛在值都在 ±3 以內，貼在動作上下限的標籤不會主導 NLL；沒有這個 key 的設定照舊用 `label_margin`。
- 活動懲罰對圖變體使用所有讀出 DN 以外的神經元速率平方平均，GRU2 使用隱狀態平方平均。梯度累積以 micro-batch 的序列數加權。有限的梯度做全域範數裁剪後交給 Adam。Cosine 排程涵蓋四輪，以參數群組的 LR 比例一起下降。
- `hosts.yaml` 設 4060、2060、4070 的 micro-batch 為 128、64、128。若縮小 effective batch 至主機值以下，整批作為一個 micro-batch。實際的 micro-batch 必須整除 effective batch。Activation checkpoint 預設關，並在設定與 model card 記錄實際值。

`replay_ratio=4` 及 held-out 每 50 次更新、至多 256 條序列、保留 5% 是計劃書交辦的 Claude 預設。固定保留的列以 seed 抽取，四輪皆保留同一批列，避免其狀態歷史進入訓練。列數四捨五入且至少保留 1 列。資料量小時，held-out 使用所有可用的不同視窗。

4060 的 smoke 使用 1,024 個同步起跑的環境，回合上限為 60 秒（1,200 步）。Teacher 很少撞牆，正式設定每輪約 36.7 秒，第一輪主要收集起步到 36.7 秒的資料，第二輪才收集回合尾段與下一回合起步。Smoke 每輪 80 步，四輪總共只看到每回合前 16 秒。第一輪 teacher 速度的平均與標準差為 0.254 和 0.055 m/s，第四輪（12–16 秒）則為 0.164 和 0.012 m/s。錯開首回合的結束時間可減少每輪資料集中在同一回合相位的情況。

## 監控與健康檢查

`monitor.jsonl` 記錄 loss、NLL、活動懲罰、梯度範數、LR、log std、更新吞吐、收集吞吐和 CUDA peak allocated bytes。內縮前的速度 `v ≤ 1e-6 × v_max` 記為 `saturation_fraction_speed_low`，`v ≥ (1 − 1e-6) × v_max` 記為 `saturation_fraction_speed_high`，轉向 `|ω| ≥ (1 − 1e-6) × ω_max` 記為 `saturation_fraction_turn`。這三種情況任一成立就算一次飽和，合計比例記為 `saturation_fraction`。`saturated_nll_fraction` 是飽和步驟的 NLL 總和除以全部步驟的 NLL 總和。定期 held-out NLL 於該次更新之前測量。每輪用有效讀值統計左右眼、氣味、風的正規化 `|z|` 最大值與精確的 99.9 百分位。沒有有效讀值的組別記為零。

每輪結束時，使用該輪收集的所有 teacher 標籤，內縮並經過 `unsquash` 後，分別記錄速度和轉向潛在值的第 0.1、1、50、99、99.9 百分位。欄位以 `label_latent_speed_` 或 `label_latent_turn_` 開頭，後綴依序為 `p001`、`p010`、`p500`、`p990`、`p999`。

每輪第一次更新記錄 `episode_time_mean_s` 和 `episode_time_std_s`，分別是該輪所有收集樣本在選擇動作前的回合內時間平均與母體標準差，以秒為單位。新回合的第一筆樣本時間為 0。逐步計數只用於這兩項監控統計。

擬合事件在 `monitor.jsonl` 記為 `readout_norm_fitted=1`，使用 `readout_norm_*` 欄位。`round_1/readout_norm.json` 保存 DN 的 `body_ids`、`rate_mean`、`rate_std` 與 `rate_scale`，以及 `n_samples`、`scale_floor`、`floored_nodes`、`speed_weight` 和 `steer_weight`。

Trainer 在 checkpoint 時同步輸出 `diagnostics.json`。各項診斷也寫入 monitor。

- `first_batch_core_gradients` 使用第一次更新的同一批序列與初始核心，分別測量速度 NLL、轉向 NLL 對 `g`、`tau`、`bias`、`lc_rf`、`missing_bias` 及整個核心的 L2 梯度範數。比較 `normalized` 及具原始讀出權重的 `unnormalized`，各組記錄 `speed`、`turn` 和 `speed_to_turn`。轉向範數為零時比值記為 null。Monitor 欄位以 `first_batch_core_grad_` 開頭。探測只計算梯度，optimizer 隨後照常更新。
- `nll` 按更新保存總 NLL、`nll_speed`、`nll_turn`，以及定期探測的 `held_out_nll`、`held_out_nll_speed`、`held_out_nll_turn`。
- `readout_drift` 在每輪 held-out 健康探測時，使用原始觀測分支記錄 `|(r − μ) / σ̃|` 的 `mean_abs`、`max_abs` 及依 bodyId 索引的 `max_abs_per_dn`。Monitor 欄位以 `readout_drift_` 開頭。漂移數值只用於診斷。

每輪完成更新後，在 held-out 視窗以 mode 動作探測，並在 `monitor.jsonl` 新增一筆 `round_end=1` 的健康指標紀錄。圖變體每個讀出 DN 的平均速率都須大於 `1e-4`。同一 burn-in 狀態會分別跑原始及左右氣味互換的 code，並記錄兩條分支的平均 `|Δω|`。中間輪次發現 DN 沉默時立即以 exit 3 結束。中間輪次的速度標準差 ≤ `0.02 m/s` 時會記錄指標，整次訓練只在首次觸及門檻時發出警告，並繼續下一輪。

最後一輪的探測是 Stage 1 交界健康檢查。若出現 DN 沉默或速度標準差 ≤ `0.02 m/s`，trainer 會保存最後一輪 checkpoint 與 `failure/` 的診斷資料，仍用 `export_policy` 匯出至 `final/`，再以 exit 3 結束。匯出成功後，`failure/reason.json` 會註明產物已匯出且 Stage 1 健康檢查未通過。健康指標、loss 或梯度出現非有限值也會觸發 exit 3。`failure/` 保存 checkpoint、可用的探測或訓練 batch，以及失敗原因。

## 停止、續跑與匯出

在輸出目錄建立 `STOP`。Trainer 會在輪次邊界、讀出擬合前後及每輪每 50 次更新查閱該檔，保存進度後以 0 結束。移除 `STOP`，以相同設定、主機、裝置、變體和 seed 加上 `--resume`。續跑讀取最後發布的完整或輪內 checkpoint。Checkpoint 保存 RNG、世界與三種 recurrent state，以及首回合的重設排程、已執行的列和收集步數。輪內 checkpoint 也保存下一次更新的位置，並讀取已收集的 replay。首輪收集後的 checkpoint 可接續擬合，擬合後的 checkpoint 會還原正規化 buffer 及 Adam，直接接續更新。Monitor 與診斷輸出對齊 checkpoint 的環境步和日誌位置。舊 checkpoint 若在收集 replay 前建立，續跑可沿用當時的設定。錯相設定加入前使用 `--override desync_first_episode=false`，圖策略的 DN 正規化加入前使用 `--override readout_normalization=false`。

續跑沿用 `config.json` 中的 `model` 覆寫，CLI 可省略原先的 `model.*` 參數。續跑時若指定了新增或不同的模型覆寫，trainer 會報錯。

每輪 `round_<k>/replay.pt` 只寫入該輪的新資料。格式標記為 `layout: env_major_f16`。各步資料連續排列為 `[環境列, 步數, ...]`，中央狀態排列為 `[環境列, 快照, 狀態寬度]`，因此單一序列在各欄位內都是連續的 mmap 讀取。載入分片時，抽樣會用到的頁面設為 `MADV_RANDOM`，讓核心依缺頁需求讀取，避免每次跳到新位置時大量預讀。讀出 DN 速率保留預設的 mmap 行為，供首輪擬合循序讀取。Code 和中央狀態以 float16 保存，teacher 動作和讀出 DN 速率以 float32 保存，reset 和遮蔽旗標以 bool 保存。分片另存世界、運動、情境與 held-out 的列標籤。舊的時間優先 float32 分片會報錯，請使用新輸出目錄重新訓練。

原子發布的 `checkpoint.pt` 指示該輪已完成，`progress.pt` 指示該輪可從中間續跑。`config.json` 記錄解析後設定，`model` 保存覆寫，`model_config` 保存完整模型設定。訓練完成時，`final/` 由 `export_policy` 產生，可直接用 `load_policy` 載入。Model card 的 `provenance.stages` 包含 DAgger 的設定、模型覆寫、主機、實際 micro-batch、activation checkpoint 與以實際步數推得的四輪排程。Model card 頂層的 `model_config` 會將雙側讀出展開為實際的型別清單和 bodyId。來源版本可由執行環境提供 `GIT_COMMIT` 或 `GITHUB_SHA`，缺值時記為 null。

## 記憶體

以預設 VO2 四輪、每輪 1,024 列 × 733 步估算，296 欄 code、兩欄動作、兩個旗標及每 16 步存一次的 8,598 欄中央狀態，合計由 10.06 GB 降到 5.05 GB（未計讀出樣本、列標籤和封裝開銷）。以 4060 實際觀察到的約 27 GB 舊分片用量換算，相同收集量的新格式約 13.5 GB。依環境連續存放可讓抽樣沿著視窗讀取，減少分散缺頁引發的預讀量。

預設設定每輪的收集緩衝約 2.52 GB，單眼百分位的暫存上限約 0.43 GB。統計完成後，每 64 列轉置拷貝到連續的目的張量，先轉換 code 並釋放舊 code，再轉換較大的中央狀態快照。轉置時同時在記憶體中的 replay 張量峰值約 2.96 GB。OS 會依可用記憶體保留 mmap 的檔案頁面。

GPU 收集每 32 個控制步批次拷回 CPU，VO2 的暫存約 105 MiB。強制重設在 graph replay 後執行，同一世界、同一步的列合成一個遮罩，並就地更新 graph 使用的張量。每輪收集完會釋放世界的 CUDA graph capture pool，下一輪用同一世界狀態重新 capture。無 activation checkpoint 時，VO2 的 recurrent core 計算圖主項估計在 128 條序列約 3.94 GiB、64 條序列約 1.97 GiB。連同讀出、CSR 工作區、模型緩衝及世界狀態，訓練峰值粗估約為 5–6 GiB 及 3–4 GiB，收集階段的峰值另受 CUDA graph 影響。實測值以 monitor 的 `max_memory_allocated` 為準。

# PPO-v2（Stage 2，#138）

從 Stage 1 的產物（`final/`）開始微調。程式在 `python/flytrack_v2/train/ppo.py`，設定在 `configs/v2/train/ppo.yaml`。

```sh
PYTHONPATH=python python -m flytrack_v2.train.ppo \
  --init runs/v2/VO2/seed_0/dagger/final --seed 0 --host 4060 --device cuda \
  --out runs/v2/VO2/seed_0/ppo
```

## 演算法
- **世界**：沿用 WP4a 的四個世界實例（`v1compat`、`wind` × curved、wander），預設每個 256 列，共 1,024 個環境；情境混合和第一回合錯開也和 Stage 1 相同。
- **actor**：用 `load_policy` 載入 Stage 1 產物，只看觀測。每一步從高斯分布抽樣，存下潛在變數 z，PPO 的比值用 `log_prob(z)` 計算，所以 squash 的 Jacobian 會在比值裡抵消。
- **critic**：MLP（256、256，tanh），讀 24 個以 agent 為中心的特權特徵：
  - 目標的相對位置、距離、方位、速度；
  - 自己的速度、到四面牆的距離、朝向；
  - 遮蔽和可見旗標、風、decoy；
  - 回合時間。
  
  輸出的單位是 1/(1 − γ)，所以回報的數量級是 1。
- **遞迴的更新**：
  - 每次 rollout（32 步）存下第一步之前的中央狀態，以及每一步的感覺碼（前端是凍結的）；
  - 更新時，每個 minibatch 的列從存下的狀態重跑同樣的 32 步，回合結束的列換回 rollout 當時的靜息狀態；
  - minibatch 依環境切分（1,024 / 8 = 128 列），主機的 `micro_batch` 再切成小塊累積梯度。
- **advantage**：GAE（γ 0.99、λ 0.95），在整批上做正規化。
  - 碰撞和 60 s 的時間上限都結束回報（terminated）。60 s 是任務本身的終點（評估的回合也是 60 s），critic 又看得到回合時間，所以在有限時域下把它當成終止才是正確的值（#188 review；Pardo et al. 2018, *Time Limits in RL*）；
  - 只有第一回合被強制提早結束時（任務本身沒有這種結束），才從重設前那個狀態的 value bootstrap；
  - advantage 不跨回合累積。
- **暖身**（計劃書 3.8，#161 D2）：前 `critic_only_updates` 次只訓練 critic；之後 `actor_ramp_updates` 次 actor 的 lr 從 0 線性升到 3e-4，再線性降到結束。critic 的 lr 從 3e-4 線性下降。
- **kickstarting**（#161 D1）：loss 加上 λ·KL(π_teacher ‖ π_θ)。
  - teacher 是同一個 Stage 1 產物另外載入、凍結的一份，在 rollout 時用自己的中央狀態跑同樣的感覺碼；
  - KL 用兩個對角高斯的封閉形式；
  - λ 在 actor 更新的前 30% 從 1 線性降到 0，之後不再執行 teacher。
  - λ > 0 的期間（包括只訓練 critic 的暖身）teacher 每次 rollout 都跟著推進，所以開始算 KL 時，它和學生帶著同樣的歷史；暖身期間只是不計算 KL（#192 R7）。
- **其他**：
  - PPO clip 0.2、value 係數 0.5、entropy 0.001；
  - 活動正則 1e-4，算法和 Stage 1 相同；
  - actor、critic 分開用 Adam，各自做梯度裁剪 0.5。
- **KL**：每個 epoch 的平均 approx KL 超過 1.5 × target（0.02）時，提早結束這次更新的 epoch。approx KL 連續 3 次超過 4 × target 時，以 exit 3 結束（monitor 的 `watch`）。

## 監控、停止與續跑
- `metrics.jsonl` 每次更新一筆：
  - 結束的回合數、成功率、碰撞率、每步 reward；
  - value loss、explained variance、policy loss、entropy、approx KL、clip 比例、`kl_teacher`；
  - 學習率、λ、log std、收集和更新的秒數、GPU 峰值。
- 在輸出目錄建立 `STOP`：trainer 在每次更新之後查看，存好 `checkpoint.pt` 後以 0 結束。移除 `STOP`，用相同的設定、主機和 `--init` 加上 `--resume` 續跑。
- `checkpoint.pt` 保存：
  - actor、critic、兩個 optimizer、更新用的 RNG；
  - 世界、感覺、學生和 teacher 的狀態；
  - 抽樣用的 RNG。
- 結束時用 `export_policy` 匯出到 `final/`。card 的 `provenance.stages` 是 Stage 1 的 stages 加上 `{"name": "ppo", ...}`，`provenance.train_seed` 是 `--seed`。`--seed` 必須等於 Stage 1 的訓練 seed：Stage 1 的 card 有 `train_seed` 時要相同，沒有時路徑裡的 `seed_<n>` 不能和它矛盾，否則報錯，因為評估的回合編號由它決定（#188 review）。
- actor 或 critic 的梯度不是有限值時，那一步不更新，monitor 以 exit 3 結束。

## 取代 Stage 1 的條件
跑完之後，用 `scripts/v2/evaluate_v2.py --seeds pilot_held_out` 分別評估 Stage 1 和 Stage 2 的產物，再照協定的 `stage2_replaces_stage1` 判定：Stage 2 − Stage 1 依訓練 seed 配對，非劣性 margin 0.05。trainer 本身不做這個判定。
