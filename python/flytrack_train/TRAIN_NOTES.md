# 訓練與評估實作契約

本文描述訓練介面；以下訓練指令供後續流程使用。2026-10-15 的 DN 修正只執行單元測試、lint、設定產生與無權重更新的 CPU 診斷，沒有執行實際訓練或 validate-pretrain；證據見 [DN 修正報告](../../runs/dn_fix/REPORT.md)。

## 責任與介面

`config.py` 管設定、隨機種子、artifact 暖啟動與環境步數帳本；`batches.py` 管感測批次和序列；`pretrain.py`、`imitation.py`、`ppo.py`、`es.py` 各自管理演算法；`evaluation.py` 負責實驗矩陣和種子層級統計。純數學回合指標在 `flytrack_core.metrics`，ROS 評估器不依賴訓練套件。

actor 只接受 `vision[2,721]`、`odor[2]`、`valid[3]`。`valid` 順序為左相機、右相機、成對嗅覺。actor state 是完整 `Tensor[B,S]`，含凍結視覺前端、ORN 適應和中央網路記憶。每次回合開始以 `policy.initial_state` 重建完整 state，不能全數清零：ORN 初始 Hill 閾值和 resting voltage 可為非零。

教師和 critic 從 `info` 讀取真值。critic 的 12 維輸入為 `(agent_x, agent_y, sin(yaw), cos(yaw), target_x, target_y, target_vx, target_vy, previous_v, previous_omega, wind_x, wind_y)`。不把這些欄位合併進 actor observation。

## 共用 YAML 與指令

### 有效 actor 參數預算（#77，2026-10-28）

正式 ±5% 依 BC optimizer 的非零學習率 actor 參數比較，以有效 VO 為基準，包含 BC 新增的 B3 讀出列。`parameters.count_effective_parameters(policy, train)` 直接呼叫 optimizer 共用的 `parameter_groups`：`declared` 計 requires_grad，`zero_lr` 計其中 lr=0，`effective` 計其中 lr>0；共同凍結前端、pretrain decoder 和 PPO critic 不計入。`imitation.parameter_groups` 保留相容匯入。

| 變體 | declared | zero_lr | effective | 相對有效 VO |
| --- | ---: | ---: | ---: | ---: |
| V | 16,242 | 1,697 | 14,545 | −0.0069% |
| O | 16,243 | 1,697 | 14,546 | 0% |
| VO | 16,243 | 1,697 | 14,546 | 0% |
| VO-late | 15,790 | 1,697 | 14,093 | −3.1143% |
| GRU | 14,904 | 0 | 14,904 | +2.4612% |
| RS | 16,242 | 1,697 | 14,545 | −0.0069% |

`prepare_experiments.py` 從同一份 `bc.yaml` train 段計數與寫出 BC stage；lr 設定為 learning_rate=0.001、readout_learning_rate=0.01、core_bias_learning_rate=0，readout_bias_learning_rate 未指定（繼承 readout）。`parameter_budget.json` 保留 `actor_trainable_parameters` 為 declared 並標示定義，新增 `parameter_counts`（含分組 lr、各組 numel、zero-lr 名稱／數量）、有效 VO 基準與 `gru_resolution`。

只有 GRU 寫 `budget_target=14546`，走既有 hidden／輸入 bottleneck 解析器，結果 hidden=16、gru_input_size=18；舊 hidden=7、無 bottleneck 的 15,516 有效參數相對 VO 為 +6.6685%，不合格。連接體 `budget_target=null`，架構不由預算解析，產生器以有效計數統一檢查；這讓 YAML 重建不會把宣告容量誤比有效 target。stage model 仍不預加 B3，BC 建 optimizer 前才加列，計數則包含這些列。

`save_actor` 在 card 的 `metadata.actor_parameter_budget` 保存目前 actor 按該 stage train 分組的 declared／zero_lr／effective 與 lr 明細，所以 pretrain 使用自己的 train 段。評估 `require_parameter_match` 重用已載入且已還原 B3 列的 actor，依保存的 `metadata.config.train` 重算並逐欄核對計數、分組 lr／numel／參數名稱與 numel、zero_lr_parameters 及頂層 `trainable_parameters`。lr 用 `math.isclose(rel_tol=1e-9, abs_tol=0)`，零值須精確一致；缺少、無效或不一致的欄位均附 artifact 路徑報錯，±5% 只比較重算的 effective，不退回 declared。`load_policy_artifact` 保留頂層 declared 的架構檢查及舊 card 的一般載入行為；正式預算評估須重產舊 card。CPU 靜態計數與 YAML 重建證據見 `runs/r77/summary.md`，review 修正與真實 artifact 往返證據見 `<runs>/r77_fix/`。

```yaml
variant: VO
seed: 0
sensor_contract_path: configs/sensor_contract.json
graph_path: data/subgraphs/olfactory_v2_nokc/graph.npz
frontend_path: artifacts/frontend/frontend.npz
model: {}
env:
  episode_seconds: 60
  scenario: normal
  target_motion: curved
  plume_model: puff
train: {}
output: runs/VO/seed_0
```

路徑以執行指令時的工作目錄為基準，建議在儲存庫根目錄執行。感測契約亦可改用 inline `sensor_contract`。`init_artifact` 載入同 variant 完整 actor 權重並驗證契約、子圖、前端來源；不載入 critic 或 optimizer。recovery 的身分包含上游 actor（`init_artifact`、`pretrained_artifacts`）的 `model_card.json` 雜湊：上游階段重跑後，下游的 checkpoint 和已完成的 actor 都會驗證失敗，必須刪除下游輸出後重跑，不會沿用由舊上游訓練出來的結果。`scenario_configs` 可指定情境 → YAML 路徑，依當前環境情境讀取，檔案的 `name` 必須吻合，其餘欄位合併至 `env.scenario_config`；inline 情境欄位優先於檔案。`configs/scenarios/*.yaml` 是情境參數的唯一來源：occlusion、noise、conflict 情境若沒有對應的 `scenario_configs`（也沒有 inline `env.scenario_config`），`make_env` 直接報錯，不再默默改用 `ScenarioSpec` 的程式預設值（#2）；程式預設值已改成與 YAML 相同，只供單元測試與 ROS 未指定檔案時使用。`configs/train/` 下的訓練範本都已加上 `scenario_configs`。

**BC 的情境混合（#4 方向 3）。** 只有 BC 的資料收集可以設 `env.scenario_mix`，PPO、ES、pretrain 和評估遇到會直接報錯。原因是 PPO checkpoint 會讀寫 env 的 `_episode_id`，包裝層無法讓它在各個子世界之間保持一致。

- **格式**：每一項是 `{scenario, weight, scenario_config?}`。inline 的 `scenario_config` 會像 `env.scenario_config` 一樣，覆蓋在該情境的 YAML 之上。設了 `scenario_mix` 就不能再設 `env.scenario` 或 `env.scenario_config`。
- **抽籤**：每回合要用哪個情境，只由 episode seed 和固定的 RNG stream（`SCENARIO_MIX_STREAM`）決定，跟 worker 數和 reset 順序都無關，所以 recovery 重收時會得到相同的世界。reset 的 info 會多一個 `mix_scenario`。
- **共用選項**：其他 env 選項，例如 `occlusion_odor_steering`，會套用到每一個子世界。
- **工具**：`scripts/diagnostics/prepare_odor_task_bc.py` 產生方向 3 的 A／B／C 三組設定；`scripts/diagnostics/odor_reliance_eval.py` 在相同 seed 上，配對比較完整模型和 odor lesion。

**正式模型規格（#4、#55，2026-10-24 使用者團隊決定）。** 4-R7 在兩個 seed 都複製成立的組合，納入正式規格：

- **圖**：`configs/train/` 的訓練範本和 `prepare_experiments.py` 預設都改用 no-KC 圖 `data/subgraphs/olfactory_v2_nokc/graph.npz`。`smoke.yaml` 仍用 main 圖，只供測試使用。
- **P2，左右共用的 ORN 適應**：`odor_shared_adaptation` 的預設改為 `true`，`variants.yaml` 也明確寫出，所以六個變體共用同一個編碼器（GRU 也一樣）。#63 之前的舊 card 沒有這個鍵，會以各側獨立適應載入。
- **B3，DNb05 轉向列**：
  - **啟用方式**：在 BC 的 `train.calibrate_extra_turn: {types: [DNb05], eps: 0.1}` 設定。
  - **加入時機**：BC 在建立 optimizer 之前，先把這一列以權重 0 加進 actor。在第 0 輪收完 teacher 資料、第一次 fit 之前，用訓練用的回合算出 center 和 scale。BC 自己的 validation 回合不參與計算。
  - **輸出**：計算結果寫進 `extra_turn_calibration.json`，其中包含 closed-form 的 held-out 相關係數（`dn_only`、`dn_plus_extra`、`extra_only`），只做記錄，不當閘門。結果同時存進 checkpoint，resume 時直接沿用，不重新計算。
  - **前置條件**：起點必須是結構化 readout 的 connectome actor，而且還沒有這一列。
  - **適用範圍**：`prepare_experiments.py` 只對 `variants.yaml` 中 `extra_turn_rows.variants` 列出的變體（O、VO、VO-late）保留這個鍵，參數預算也把這一列算進去（每個讀出分支 +1）。以下變體不套用：
    - V：沒有氣味時 DNb05 不活動（teacher 回合中 L+R < 1e-5）。
    - RS：重接後右側 DNb05 沒有輸入，L/R 比例固定為 1。
    - GRU：沒有 DN readout。
  - **單獨校準**：`scripts/diagnostics/calibrate_extra_turn.py` 和 BC 共用 `flytrack_train/extra_turn.py`，可在既有資料集上單獨校準。
- **B3 中心項的修正**：特徵改為 S/(S+ε)·(L−R−c·S)/max(S,ε)·scale，也就是中心項和比例共用分母。舊公式 (L−R)/max(S,ε) − c 在 0 < S < ε 時，會把 c·scale 依閘門比例漏進轉向（4-R7 的開頭暫態）；S ≥ ε 時兩種公式相同。新公式由 `readout_extra_turn_center_ramp: true` 控制（預設值）。舊 card 沒有這個鍵，以舊公式載入，4-R6／4-R7 的 actor 因此保持原本的行為。暖啟動時可以用這個鍵切換，但起點已經有訓練過的列時會報錯。BC 的 `calibrate_extra_turn` 新增列時一律用新公式，起點即使是舊 card 也一樣，並把這個值記在 `extra_turn_calibration.json`。

```bash
python -m flytrack_train pretrain --config configs/train/pretrain_vision.yaml --variant VO --seed 0 --output runs/pretrain/VO/seed_0/vision
python -m flytrack_train bc --config configs/train/bc.yaml --variant VO --seed 0 --output runs/bc/VO/seed_0
# 以下兩個 RL 階段不在正式鏈（#35 Decision），只用在研究
python -m flytrack_train ppo --config configs/train/ppo.yaml --variant VO --seed 0 --output runs/VO/seed_0
python -m flytrack_train es --config configs/train/es.yaml --variant VO --seed 0 --output runs/es/VO/seed_0
python -m flytrack_train evaluate --config configs/train/evaluate.yaml --output runs/evaluation
python -m flytrack_train export --config configs/train/bc.yaml --artifact runs/bc/VO/seed_0/actor --output artifacts/VO/seed_0
```

設定檔命名由根目錄配置提供；CLI 支援 `--device auto|cpu|cuda|cuda:0`、`--seed`、`--variant` 與 `--output`。`--variant` 不会重寫 `init_artifact` 路徑，換模型時必須同步暖啟動來源。

### BC 結束時的轉向偏差置中（#73）

`train.center_turn_bias` 預設為 `false`；正式 `configs/train/bc.yaml` 也是
`false`，因為 #73 的 GPU 閉迴路對照未通過；`prepare_experiments.py` 的六個變體
均沿用此 train 段。功能保留給診斷和事後工具使用。

> **封存結果（#73，seed 0，100 回合）**：occlusion_long 的 intact 成功率從
> 0.26 降到 0.16；G3 的 D 從 13.6° 降到 3.9°（符號檢定 p = 1.0）。
> occlusion_long 和 G3 均未通過；採用條件要求兩個 seed 都通過，因此不採用置中。

設為 `true` 時，用一個加性純量消除錄下狀態上的平均轉向偏差，避免整段軌跡
被固定的 ω 偏移影響。
只在最後一輪 fit 之後、`save_actor` 之前執行一次，不影響 DAgger 收集；最後一次
recovery checkpoint 保留未置中的參數。續跑即使跳過全部 round，也會重新編碼
訓練回合並得到相同結果。關閉時 actor 和 rounds metrics 保持原本數值及結構。

資料排除與 BC 相同 `validation_every` 的 `held_out_episodes`，不讀 held-out 的
觀測或 label。全程 `no_grad`、eval，以 `SequenceModel.encode` 及 validation
相同的連續 hidden state 重播錄下觀測；結束後恢復各模組原本的 train/eval 狀態。
以每步等權的 float64 二分法求 Δ，使 `mean(ω_max*tanh(z+Δ))` 等於
teacher ω 均值，容許誤差 `1e-6 rad/s`、最多 100 次。teacher 均值不嚴格位於
`(-ω_max, ω_max)`、不收斂或參數精度不足都報錯。

structured connectome 修改 `readout.turn_bias`（未啟用此參數會報錯）；linear
connectome 修改 `readout.linear.bias[1]`；GRU 修改 `readout.bias[1]`。
三者均直接平移 tanh 前的 ω latent，不修改其他權重或 forward 輸出介面。
`metadata.turn_bias_centering` 與 `training_metrics.json` 最後一輪的
`turn_bias_centering` 記錄相同內容：`parameter`、`delta`、`before`、`after`、
`train_mean_error_before`、`train_mean_error_after`、`teacher_mean_omega`、`steps`、
`episodes`、`held_out_excluded`、`tolerance`、`iterations`。metrics 保留 rounds
list，不新增假 round；after 誤差以同一批 latent 和實際寫入的參數差計算。

事後工具共用相同函式，不更新來源目錄；設定預設取 card，可用 `--config` 覆寫，
但 `train.validation_every` 必須與 card 記錄的設定相同，否則 held-out 切分會改變而報錯：

```bash
PYTHONPATH=python python scripts/center_turn_bias.py runs/bc/VO/seed_0 \
  --output runs/centered/VO/seed_0/actor --device cpu < /dev/null
```

輸出是可由 `load_policy_artifact` 載入的新 actor；目錄存在即報錯，除非加
`--overwrite`。card 額外記錄 `applied_post_hoc: true`，以及 `source_artifact`
中的來源 actor 路徑、所有檔案 SHA-256 和來源 `training_metrics.json` SHA-256。
不新增訓練環境步數，保留原 BC 的預算帳本。

## 單模態預訓練

`train.modality` 為 `vision` 或 `odor`，V/O 可由 variant 推定。所有變體可各自預訓練，另一感覺的數值及 valid mask 都被關閉；視覺前端依然固定。視覺標籤為方位 sin/cos、可見性和相對角速度；嗅覺標籤為方位 sin/cos 和是否在羽流中。方位損失只在訊號存在時算，存在性用 BCE。解碼器接該模態群與匯流群；GRU 的 activity 為診斷 hidden-channel 映射，不能解讀成生物神經元。

```yaml
train:
  modality: vision
  total_steps: 120000
  epochs: 20                  # 上限；提早停止決定實際 epoch 數
  batch_episodes: 16
  validation_every: 10
  early_stopping_patience: 3
  sequence_length: 32
  learning_rate: 0.001
  exploration_probability: 0.2
  odor_presence_threshold: 0.01
  angular_velocity_coef: 0.1
  activity_coef: 0.0001
  max_grad_norm: 0.5
```

collection 精確消耗 `total_steps`，不因提早碰撞而縮減預算。

**平行收集。** 環境放在 `--workers` 個子行程（`parallel.EnvPool`），trainer 仍決定每次 reset 與動作。第 `i` 個回合的環境種子與私有亂數只由 `SeedSequence([seed, stream, i])` 決定，回合依編號提交，跨過預算的那個回合被截斷，所以資料與 worker 數、排程時序無關，和單一序列收集器的語意相同。預訓練資料不依賴 actor，設定 `dataset_cache` 時同一種子／模態／世界的回合流只收集一次，所有變體讀取自己預算需要的前綴（檔案鎖保護同時執行）。

**凍結感覺快取。** flyvis 前端與 ORN 適應沒有可訓練參數，`policy.sense()` 輸出的感覺碼只取決於觀測歷史，因此每個收集到的步只算一次，所有 epoch 與 PPO 更新重用；可訓練部分走 `policy.forward_code()`。`policy(obs, state)` 仍是兩者的組合，部署行為不變。同一個前向圖（一個 TBPTT 窗口、一次 no-grad rollout）內，`adjacency_cache()` 只組一次連接矩陣。

**批次 TBPTT。** `batch_episodes`（預設 16）條回合平行前進，每條在自己的 lane 依時間順序走 `sequence_length` 窗口並延續自己的遞迴狀態；一次 optimizer step 平均各 lane 的單窗口損失（每 lane 的損失定義與單回合時完全相同）。梯度在窗口邊界 detach，回合開頭的初始狀態每窗口重建，梯度照常流到靜息 bias。

**提早停止。** 每第 `validation_every`（預設 10）個回合保留為驗證集，不參與擬合；每個 epoch 後計算驗證損失，連續 `early_stopping_patience`（預設 3）個 epoch 沒有相對改善 `early_stopping_min_delta` 就停止，最後還原驗證損失最佳的參數與 optimizer 狀態。`epochs` 是上限。回合數少於 `validation_every` 時不保留驗證集，跑滿 `epochs`。產出 `pretraining_checkpoint.pt` 保留輔助頭供研究診斷，`actor/` 完全不含輔助头。

正式公平流程可讓 V/O 各用 240000 步，VO/VO-late/GRU/RS 各先視覺 120000 步，再以該次 `actor/` 作 `init_artifact` 做嗅覺 120000 步。模型各自暖啟動，可避免從生物模型替基準模型挑選較好的初始權重。

另有選配 `pretrained_artifacts: {vision: PATH, odor: PATH}`，按模態移植連接體 type / type-pair 參數，交集平均，保留新任務 motor readout。限連接體 family、要求相同圖與前端來源；不能與 `init_artifact` 同時使用，也不能套到 GRU。

## Pretrain decoder 輸入尺度

- `train.decoder_input_scale`：`none`（預設，原始線性 decoder）或 `population_rms`；以更新前 intact policy 的第一批完整已編碼 episodes 的全部有效時間步計算兩個族群純量 `sqrt(mean(rate²))`（排除 padding／held-out，RMS ≥ 1e-3）：該模態的 visual／olfactory 去除 convergence，以及 convergence；輸入只除以各族群 RMS，不減平均。統計只算一次，research／recovery checkpoint 的 `decoder` 保存 `input_rms`、`input_convergence`、`input_initialized`，resume 原樣沿用；讀取使用 `PopulationRMSDecoder`（`validate-pretrain` 依 checkpoint 是否含 `input_rms` 自動選用）。此模式 decoder weight／bias 初始化為 0（仍照常消耗 `nn.Linear` 的隨機抽樣）：未減平均的單位 RMS 輸入配上隨機 head 起始 loss 達 9–15，零初始化起始約 1 + ln 2。

## DN 活動正則與階段診斷

pretrain、BC 與 PPO 共用 `losses.activity_penalty(policy, activity)`：連接體策略的 `activity_coef` 只乘上非運動 DN 神經元的平方活性平均，分母是非 DN 數量；保留時間／批次軸供原本的 lane／step mask 彙總。全為運動 DN 時回傳可微的零。GRU 保留全部診斷 hidden channels 的平方平均。DN 仍可能透過對其他神經元的回饋受到間接梯度；這不是證明活動正則造成先前沉默的消融結果。

所有新連接體模型使用 `motor_rate_beta=10.0` 的 DN softplus（其他神經元維持 ReLU），完整定義見 [MODEL_NOTES.md](../flytrack_core/model/MODEL_NOTES.md)。舊 artifact 缺少 β 時保留 ReLU；若要暖啟動並切換為新規則，須明確設定 `model: {motor_rate_beta: 10.0}`，新生成的六變體設定已包含此鍵。單純讀取 artifact 不會修改它。

`scripts/dn_health.py --artifact PATH` 是 CPU 唯讀診斷：使用 model card 記錄的世界設定（或 `--config`）與 held-out 種子 `[1.5e9,1.7e9)`，以教師駕駛世界，重播原始與只交換左右眼影像的配對序列。預設 24 回合，各 burn-in 16 步、量 200 步（4×64 步時同一個 BC 模型曾量得 −157% 與 +30% 兩種結果），從量測段前 16 步計算教師動作的尺度正規化 MSE 梯度；回合提早結束會如實記錄實際步數。沒有 optimizer、參數更新或預訓練 decoder。輸出每個 DN 的正活性比例、有效活性比例與 rate 分布、各支路電位分位數、換眼角速度差、`core.*`／`readout.*` 及其群組梯度範數、角速度誤差與固定零轉向基準。

softplus 的正值靜息項不能當作運動控制證據。除原始 `rate>0` 比例，診斷另用 `rate>1e-3` 計算有效活性，要求至少 4 個 DN 在 ≥10% 步數有效、中央梯度範數 >1e-6、視覺換眼角速度差中位數 ≥0.05 rad/s，所有數值有限。BC 階段另要求策略 ω 與教師 ω 的相關 ≥0.5、R²>0（勝過最佳常數）且 ω 標準差 ≥ 教師的 25%；PPO／ES 要求相關 ≥0.3 與同樣的幅度條件。原本的「MAE 比固定零輸出低 20%」會讓完全固定轉向的模型通過（教師平均 ω 不為零），只看相關則會讓方向正確但幅度只有 1/80 的模型通過（其 R² 約 +0.02）。O 變體依設計完全遮蔽視覺，所以不套視覺換眼門檻；GRU 沒有生物 DN 對應，略過 DN gate。診斷不能取代正式 held-out 評估或 validate-pretrain。

`--output JSON` 可儲存報告，`--require-healthy` 使失敗回傳 exit code 2，並在退出前寫出 metrics（健康時為 0）。`--motor-rate-beta 10` 只供舊權重的記憶體反事實探針，輸出會標記，不能當成訓練結果。`--sparse` 可選相同方程的 CSR 算子。`scripts/run_experiments.py` 在每個連接體階段結束（含已完成階段的恢復檢查）重新呼叫此診斷。除下一節的終端 BC 分類外，未通過寫入 `dn_health_failed` 並停止後續鏈。預設任何一條鏈失敗（階段錯誤或 gate 未過）都會讓其他鏈在下一個階段交界停下；`--keep-going` 讓其他鏈繼續，最後一起回報失敗並以非 0 結束，但終端 BC 的工程失敗仍強制停止整個矩陣。

## BC 模態 gate 與 DN 失敗分類（#25）

依 #4 Decision 6390024036 第 8 點與 Decision 6442266901，每條 chain 的 BC 在 dn_health 後執行 `scripts/diagnostics/modality_sensitivity.py`，包含 V、GRU、RS，也包含已完成階段的續跑。使用 CPU、單執行緒 float32、預設 200 步及診斷種子 1,610,000,000；不繼承 `STEPS`、`SEED`、`BUILD` 覆寫，不使用正式評估種子。比較 BC actor 與 execution plan 中同一 `(variant, seed)` 的 `pretrain_odor` 輸出 actor，使用 BC 的 yaml。原始輸出為 `<bc output>/modality/sensitivity.json`，判定為 `modality/metrics.json`，log 為 `modality.log`。

G4 要求 BC 的 `odor_swap_lr.mean` ≥ **1e-3 rad/s**，且 ≥ **起點的 3 倍**，兩個邊界都包含等號。O、VO、VO-late 強制通過；失敗 row 記為 `modality_gate_failed` 並走一般 chain 失敗路徑（相容 `--keep-going`）。V、GRU、RS 僅記錄，診斷程式出錯也記錄 `error`，不阻擋。沒有 `pretrain_odor` 時 `start`、相對門檻與相對 check 為 null，只比較絕對門檻。metrics 記錄兩個 actor 的 swap mean、`odor_to_vision_ratio`、no_odor／no_vision mean 及所有門檻與 checks；**odor／vision 比值只記錄，沒有比例門檻**。`training_metrics.json` 是每輪一筆的 list；取各輪 `epochs` 最後一筆的 `best_epoch`，形成 metrics 的 `best_epoch` list，缺少時為 null，完全不參與判定。

只有 BC 為 chain 最後階段（`--algorithm none`）才依 dn_health 的 `checks` 分類：`omega_correlation`、`omega_r2`、`omega_amplitude`、`eye_swap_response` 為轉向品質；`finite`、`not_all_silent`、`active_motor_count`、`central_gradient` 與其他未知失敗鍵皆為工程問題。僅品質失敗時 row 保持 `complete`，另記 `dn_health_quality_failures`，繼續模態 gate。有工程失敗時記 `dn_health_failed` 與 `dn_health_engineering_failures`，整個矩陣在下一個階段交界停止，**`--keep-going` 也不能略過**；已在跑的階段可完成。缺少／無效 metrics、診斷例外或非預期 exit code（非 0／2）也算工程問題。pretrain 與 `--algorithm ppo/es` 的 BC、最終階段維持原有嚴格 gate，任何 dn_health 失敗都 raise。GRU 繼續略過 DN gate，但會執行模態診斷。

## 環境退化檢查

`scripts/run_experiments.py` 在任何階段開始前，對 BC 與 PPO／ES 的世界各跑一次 `flytrack_train.audit.audit_environment`（40 回合）：教師成功率須 ≥90%，八種固定動作與教師平均動作的成功率都須 ≤10%，否則拒絕訓練並寫出 `environment_audit.json`；`--skip-environment-audit` 只供診斷。正式評估也先對評估世界做同樣檢查並把結果存入報告。舊世界（永遠逆時針繞圈、果蠅從正後方出發、成功只看距離）中固定左轉成功率 100%，與教師相同，這是 BC 只學到平均動作的主因。

## BC 與 DAgger

```yaml
init_artifact: runs/pretrain/VO/seed_0/odor/actor
train:
  teacher_steps: 60000
  dagger_steps_per_round: 30000
  dagger_rounds: 3
  dagger_beta: 0.5
  epochs: 30
  sequence_length: 32
  learning_rate: 0.001
  core_bias_learning_rate: 0.0
  readout_learning_rate: 0.01
  early_stopping_min_delta: 0.005
  max_grad_norm: 0.5
  activity_coef: 0.0001
```

`imitation.parameter_groups` 依名稱分組：`core.bias`（神經元型別偏置）在 BC 固定，`readout.*` 以 1e-2 學習，其餘用 `learning_rate`。離線 BC 顯示偏置的梯度比連結強度大 30～1000 倍，優化器先調偏置去配合平均動作，同時把已傳到 DN 的方位資訊破壞（DN 左右差 R² 0.67→−0.12）；固定偏置則保留。GRU 沒有連接體偏置，只套用讀出學習率。`early_stopping_min_delta` 為相對改善門檻，0 時驗證損失每 epoch 只降 0.1% 也永不停止。

PPO 和 ES 也讀同一組設定鍵：在 PPO／ES 的設定裡寫 `core_bias_learning_rate: 0`，bias 就會在該階段凍結。pretrain 和 BC 維持凍結 bias，選配 PPO／ES 保留原有可訓練設定。舊 #4／#36 以解凍 bias 維持宣告預算的說明屬歷史；#77 改以 BC effective 比較並重新解析 GRU，不解凍 bias。PPO 的 critic 另成一組；ES 只有一個步長，所以只接受 0（把該組排除在搜尋向量之外），其他群組學習率會報錯。`readout_bias_learning_rate` 只套用在名稱以 `readout.` 開頭、以 `_bias` 結尾的參數（結構化 readout 的 `readout.turn_bias`、`readout.forward_bias`），優先於 `readout_learning_rate`；沒設定時分組和原本完全相同。#35 的 C11 用它把這兩個 bias 凍結在 BC 的值；它們只有 2 個參數，effective 會據實排除。pretrain 也使用 `parameter_groups`（#4、#7）：可以設 `core_bias_learning_rate`、`readout_learning_rate`，沒設定時和原本一樣只有單一學習率。auxiliary decoder 另成一組，所以 pretrain 的 optimizer 至少有 2 個參數組。**#7 分組實作之前開始、還沒跑完的 pretrain checkpoint 無法 resume**（`optimizer.load_state_dict` 的參數組數對不上），要從頭重跑該階段；已完成的 pretrain actor 不受影響，可以照常當 `init_artifact`。#77 只共用分組函式，沒有改變既有分組順序或數量。

先以教師收集精確 60000 步，再依學生訪問的觀測逐步向教師取得標籤。DAgger 收集時學生在 GPU 上對所有平行世界批次推論，每個世界有自己的遞迴狀態；擬合使用與預訓練相同的批次 TBPTT 與提早停止（驗證回合跨所有輪次）。DAgger 逐輪聚合資料，mix probability 為 `beta^(round)`，最後一輪完全由學生行動，標籤仍來自教師。每輪精確 30000 步；即使收集預算在回合中結束，最後的有效前綴仍納入資料。動作 MSE 先除以物理速度範圍，避免兩種單位尺度主導損失。

## Reward 與回報尺度

環境預設使用有界存活獎勵（#19）：`env.distance_penalty_cap_m: 3.0` 把距離
誤差封頂為 3 m，`env.reward_weights.survival: auto` 在初始化時計算每步非碰撞
懲罰上界 `P_max`，並在每個未碰撞步加入 `b=P_max`。碰撞步不加存活獎勵，仍扣
原本的 `collision` 並立即 terminated。推導包含距離、lost、動作變化和負向
progress；watchdog 直接歸零的動作變化也算在內，詳見
[SIM_NOTES.md](../flytrack_sim/SIM_NOTES.md#bounded-survival-reward)。環境屬性
`reward_penalty_bounds`、`max_step_penalty`、`survival_bonus` 可供診斷讀取。

預設速度／權重下，每個存活步 reward ≥0，碰撞步的 reward 上界為負，因此
gamma=0.99 的 PPO 與未折扣的 ES 都不會因「省掉後續負回報」而偏好立即碰撞。
自訂 collision 權重仍需大於單步正向 progress 上界，才有嚴格的最差存活比較。
固定長度下，固定 bonus 不改變套用距離上限後的存活策略排序。成功率、碰撞率與
評估協定不變，`episode_metrics.episode_return` 和 `info['episode']['r']` 累積新 reward。

預設 `control_dt=0.05`，靜止世界 `P_max=b≈1.575`，移動世界約 1.595。60 秒
有 1200 步，完整存活的未折扣回報保守界分別約 `[0,1950]`、`[0,1998]`；若計入
提早碰撞，整回合下界分別約 −11.575、−11.595。gamma=0.99 下，完整存活回報
上界約 162.499、166.499。這些界限是原始 reward；PPO 預設會正規化訓練 reward
（見下節）。critic loss 預設為 `0.5 * value_error².mean()`；只有設定正數的
`train.value_clip` 才使用 `0.5 * max(value_error², clipped_value_error²).mean()`。
應同時記錄回報與
`training_metrics.json` 的 `critic_loss`（value loss），CPU smoke 只能檢查執行及
有限數值，不能代替 500k 步 T3 的學習曲線和碰撞率驗收。

`survival` 也可設有限非負數字，設 0 只關閉 bonus；小於 `P_max` 時不保證存活步
非負。`distance_penalty_cap_m` 可設有限非負數字、0（距離項歸零）或 null
（不封頂，auto 會從場地與軌跡邊界推導較大的上界）。以下兩個設定合用才會
逐位元重現舊 reward：

```yaml
env:
  distance_penalty_cap_m: null
  reward_weights: {survival: 0}
```

新 reward 是核准的預設值，現有訓練 YAML 不必同步增加權重。舊 PPO checkpoint
不可直接 resume；pretrain／BC artifact 可沿用並重新開始 PPO。

## Recurrent PPO

PPO actor 共用 imitation 的參數組：`train.core_bias_learning_rate` 和
`train.readout_learning_rate` 未設（或 null）時沿用 `learning_rate`（PPO 預設
`3e-4`）；設成 0 時該組不執行 optimizer 更新，包含 Adam moments 與 weight decay。
critic 獨立成組，始終使用 base learning rate。群組學習率必須有限且非負。
兩項皆未設且沒有暖身時，保持原本參數順序及更新結果。
`train.readout_bias_learning_rate` 可覆寫 readout bias 的學習率，必須為有限非負數（不接受 bool），設為 0 時凍結 `readout.turn_bias` 和 `readout.forward_bias`。

`train.critic_warmup_updates` 預設 0，必須為非負整數。設 N 時，第 1～N 次
rollout update 的 actor 各組 lr 為 0、參數不變，critic 照常更新；
第 N+1 次開始恢復設定的群組 lr（永久凍結的組仍為 0）。每次 update 都由設定
及 checkpoint 恢復的 iteration 重算學習率，不使用 optimizer 儲存的 lr 作恢復值。
暖身仍收 rollout、算 GAE 和完整 loss／gradient 診斷；`training_metrics.json`
及 checkpoint progress 記錄 `critic_warmup`、`actor_learning_rates`。
暖身期間非永久凍結組仍累積 Adam moments，與候選診斷 hook 相同；設定 lr=0 的
永久凍結組則不累積。舊 checkpoint 仍受既有 source fingerprint
限制，這項變更後應重新開始 PPO；本版本產生的 checkpoint 可在暖身期間續跑。

`train.bc_anchor_coef` 預設 0（未設或 null 亦為 0），必須為有限非負實數，
不接受 bool。只有 PPO 使用這個鍵；imitation 和 pretrain 不使用，ES 只接受
0 或 null，非 0 報錯 `ES supports only 0 for bc_anchor_coef`。
設為正數時必須提供 `init_artifact`，在原本含 activity 項的 PPO loss 上加
`bc_anchor_coef * Σ_p ((p − p_BC)²).sum()`，共用同一次 backward 和梯度裁剪。
錨定集合只含 actor 中 `requires_grad` 為真的參數，排除設定學習率為 0 的
`parameter_groups`，不含 critic；critic warmup 暫時把 lr 設為 0 不會排除參數。
θ_BC 是 `prepare_policy` 載入 `init_artifact` 並完成 warm-start migration 後、
載入 recovery checkpoint 前的 `detach().clone()` 固定副本，包含 B3 新增的
零權重 turn row。續跑每次從 `init_artifact` 重建 θ_BC，依錨定集合順序取
`.cpu().contiguous()` 的 float32 位元組串接，checkpoint 保存
`bc_anchor_sha256`（SHA-256），不符即報錯。coef 為 0 時不複製參考值、不計算或
加入錨定 loss，也不保存 SHA 欄位，預設的運算與 metrics 欄位保持原樣。
正數時 `training_metrics.json` 每次 update 額外記錄 `bc_anchor_loss`
（所有 minibatch 的加權錨定 loss 平均）和 `bc_anchor_distance`
（最後一次 optimizer step 後的 ‖θ − θ_BC‖₂，以 float64 累加）；
`actor_grad_norm` 此時包含錨定梯度。出處：#35 Decision 6372748277，
選項 A 的 C13 = C12 加上全部可訓練 actor 參數的 BC L2 錨定。

```yaml
init_artifact: runs/bc/VO/seed_0/actor
train:
  total_steps: 2000000
  num_envs: 8
  rollout_steps: 256
  sequence_length: 32
  sequence_batch_size: 16
  update_epochs: 4
  learning_rate: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  clip_coef: 0.2
  value_clip: null
  value_coef: 0.5
  entropy_coef: 0.01
  activity_coef: 0.0001
  max_grad_norm: 0.5
  critic_hidden_size: 128
  target_kl: 0.03
  reward_normalization: true
  reward_clip: 10.0
  separate_grad_clip: true
```

多個 scalar 環境由 trainer 擁有（在 `EnvPool` 子行程中同時 step），逐一取得終局 info 後手動 reset，無 Gymnasium autoreset 覆寫終局觀測的風險。`total_steps` 必須是 `num_envs` 的倍數；最後 rollout 可縮短。每個 TBPTT minibatch 保留連續時間窗口，不以任意單步打散遞迴；窗口初始 state 是該 rollout 的 detached behavior state，窗口內按 episode-start reset。

策略先把 actor 物理均值 `[v,omega]` 轉為 `atanh([2v/v_max-1,omega/omega_max])`，再從 latent Gaussian 抽樣，經 tanh 與物理 affine transform 還原。這保留正向速度 sigmoid readout 的含義，log probability 包含 tanh 與物理縮放的 Jacobian。存 latent sample，避免接近界限時對實際動作做不穩定反算。

真終止 `terminated` 不 bootstrap；時間限制 `truncated` 從 reset 前的真終局 privileged state bootstrap。兩者都截斷 GAE recursion，避免下一回合回報滲漏。critic 和 optimizer 只存在 `training_checkpoint.pt`，部署 `actor/` 只有 actor。

`train.reward_normalization` 預設 `true`，採 SB3 VecNormalize 的方式：各環境先更新
折扣累積回報 `R = gamma * R + r`，以平行合併的 running mean/variance 統計所有
環境的 R（初始 mean=0、var=1、count=1e-4）。GAE 與 critic 收到的 reward 為
`clip(r / sqrt(var + 1e-8), -reward_clip, reward_clip)`；`train.reward_clip` 預設 10，
必須有限且大於 0。只除以標準差、不減平均，正負號保持不變，原始非負的存活步
仍然非負。每次先將終局 R 納入統計，再於 terminated 或 truncated 將該環境 R
歸零；rollout 交界不歸零。單環境剛開始時的少量 reward 可能被裁切，穩定後的
一般存活步不會觸及預設上限。running mean、variance、count 與各環境 R 隨 recovery
checkpoint 保存並在 resume 恢復。回合累積回報、`mean_last_20_episode_return` 仍是
原始 reward。關閉此設定時完全使用原始 reward，`reward_clip` 不生效。

`train.separate_grad_clip` 預設 `true`，actor 全部參數（含 `log_std`）與 critic
各自以 `max_grad_norm` 裁剪，避免 critic 的大梯度壓縮 actor 更新。設 `false` 則
沿用合併裁剪。ES 的 rank 正規化不受影響。

`train.value_clip` 預設 `null`，critic 使用一般平方誤差，不裁剪估值變化，讓嚴重
低估的 critic 仍能往 target 更新。設成有限正數時，將估值裁剪至
`old_value ± value_clip`，再取原始與裁剪後平方誤差的較大值；`clip_coef` 僅控制
actor 的 ratio clip，與此設定無關。0、負數、NaN、無限大及非數字均不接受。
例如 old_value=0、target=20 或 160、預測=0.3 時，`value_clip=0.2` 會使 critic
梯度為 0；預設不裁剪時，梯度仍會提高估值。要逐位元重現舊 PPO，須同時設定
`value_clip: 0.2`（等於舊 `clip_coef`）、`reward_normalization: false` 和
`separate_grad_clip: false`。

每個 rollout 更新在 `training_metrics.json` 追加 `value_mean`（更新前 rollout
估值平均）、`return_std`（最後一步使用的正規化分母；停用時為 1）、
`actor_grad_norm`／`critic_grad_norm`（各 minibatch 裁剪前範數的平均）與
`actor_grad_norm_clipped`／`critic_grad_norm_clipped`（裁剪後範數的平均）。估值及
critic loss 使用訓練 reward 的尺度，比較實際折扣回報時必須用相同尺度。

預設課程依累積步數：靜止高斯羽流 → 直線高斯羽流 → 曲線 puff → 視覺遮蔽 → 線索衝突。`train.curriculum` 可覆寫，每項有遞增 `until`（最終 ≥1）與環境參數；只在下一回合套用，不在回合中改变物理環境。

## ES 備援

```yaml
init_artifact: runs/bc/VO/seed_0/actor
train:
  total_steps: 2000000
  population_size: 16
  evaluation_steps: 1250
  sigma: 0.02
  learning_rate: 0.01
  weight_decay: 0.001
```

OpenAI-style mirrored ES 每個 direction 做 ± 擾動，以相同環境 seed 降低回報差的變異。依平均 rank（相同回報同 rank）估計梯度，用 Adam 更新可訓練參數向量。凍結前端不擾動。每次擾動精確執行 `evaluation_steps`，碰撞後另開回合並重建記憶。`total_steps` 必須是 `2*population_size*evaluation_steps` 的倍數。若採 ES，所有六模型均改用 ES 與相同預算；不能把 ES VO 與 PPO 基準直接作正式比較。

## 正式評估

```yaml
evaluation:
  variants: [V, O, VO, VO-late, GRU, RS]
  scenarios: [normal, occlusion, noise, conflict]
  seeds: [0, 1, 2, 3, 4]
  episodes_per_seed: 100
  artifact_template: runs/bc/{variant}/seed_{seed}/actor
  environment_seed: 731
  bootstrap_seed: 1301
  bootstrap_draws: 10000
  require_full_matrix: true
  require_equal_budget: true
  require_shared_sources: true
  require_parameter_match: true
  parameter_tolerance: 0.05
```

亦可使用 `evaluation.artifacts` 的 variant → seed → directory mapping。每組 seed/episode 使用同環境種子，跨模型可成對比較。正式評估驗证相同 graph / frozen frontend hash、相對有效 VO 的 ±5% actor effective 參數（#77，排除 optimizer lr=0 組），並以完整 `stage_budget_ledger` 依演算法合計的環境步數比較公平性；帳本繼承初始 artifact，包含預訓練、BC/DAgger 和 PPO 或 ES。

輸出 `episodes`、`seed_means`、`summary`、`comparisons` 的 JSON/CSV 及 `evaluation_manifest.json`。統計先在每個訓練種子內平均回合，再對五個種子平均作 95% bootstrap；不把 500 個回合當成 500 個獨立模型。預先比較為 VO−V（遮蔽）、VO−GRU、VO−RS（各情境）。重新捕獲只統計成功完成的事件，同時報告尚未完成事件數；空值輸出 JSON `null` 與空白 CSV，避免 NaN JSON。

同一 `evaluate` 指令可彙總既有 Gazebo `episodes.jsonl` 或 CSV：設定 `evaluation.recorded_episode_files: [PATH, ...]`（支援 `*` 和 recursive `**` glob）、`episodes_per_seed: 20`，並提供一致的 artifact mapping/template 作預算驗證。只跑主模型比較時可指定較少 variants 並設 `require_full_matrix: false`。記錄以固定 `training_seed` 分組，逐回合的環境 seed 另放 `environment_seed`，不可把每個回合的環境 seed 當作獨立訓練模型。預設仍驗證 artifact；若只有資料檔，必須顯式關閉 `verify_recorded_artifacts` 及 `require_equal_budget`，manifest 會標記未驗證訓練預算。

已驗證 artifact 時，預設 `verify_recorded_model_cards: true` 還會要求每筆記錄的 `model_card_sha256` 符合矩陣中的實際 model card，避免同 variant/seed 的不同 checkpoint 混算。只有舊記錄缺少此欄位而需要診斷整理時才顯式關閉，不能聲稱已驗證 checkpoint 一致性。

invalid 回合仍以 success=false 納入成功率；距離、方位、重新捕獲、平滑度與碰撞摘要只使用有效完整回合，完整原始部分資料保留在 episodes 報表，另提供 valid_episode / sample_coverage 的種子摘要。模型 walltime 不依賴真值覆蓋率，仍可用於診斷。

2D 評估把每格（變體／種子／情境）的回合分散到 `batch_episodes` 個平行世界，actor 批次推論；每格前 `latency_episodes`（預設 1）個回合仍以部署方式每步單筆呼叫完整 `policy(obs, state)` 並量 latency，其餘回合的 latency 欄位為空，種子平均只用量過的回合。2D latency 是 sensor tensor 組装至 actor CPU 動作輸出的 wall time，CUDA 會同步，排除 rendering。ROS 主 latency 是保留原 sensor timestamp 到收到 cmd 的 simulation-clock age，因 Twist 沒有 header，無法精確關聯是哪一筆輸入；這是診斷資料，不能單獨拿來驗收 wall-time end-to-end p95 <50ms。

策略另發布 `/fly/inference_latency_ms`（std_msgs/Float64），ROS evaluator 在 JSON/CSV 的 `inference_wall_p50_ms` / `inference_wall_p95_ms` 保留真正 perf-counter 模型推論處理時間，含 tensor 準備與 CPU materialization，排除感測生成、DDS 與等待控制 timer。兩組 latency 指標不可混用。

## ROS 評估與錄影資料

`flytrack_eval` 節點只讀取世界真值、情境、感測時間戳和 `/cmd_vel`，只發布 `/eval/metrics`。每回合依 `ScenarioEvent.active=true/false` 開始與結束。碰撞用設定中的場地邊界、機器人/目標半徑及圓形或矩形障礙幾何判斷；不是 Gazebo contact sensor 的直接測量。缺失真值樣本的覆蓋率低於 95% 或回合未完整結束時，記錄 invalid，success=false，不以少量樣本宣告成功。

```bash
ros2 launch flytrack_eval evaluation.launch.py artifact:=/absolute/artifacts/VO/seed_0 output_dir:=runs/gazebo_eval record_bag:=true
ARTIFACT=/absolute/artifacts/VO/seed_0 SENSOR_CONTRACT=/absolute/configs/sensor_contract.json launch_test ros2_ws/src/flytrack_eval/launch_testing/test_truth_isolation.py
```

record node 以 `subprocess` 啟動 `ros2 bag record --use-sim-time`，錄 `/clock`、`/fly/*` 感測及神經活性、`/cmd_vel`、`/world/*` 真值與情境、`/eval/metrics`。關閉時送 SIGINT 並等待 rosbag flush index。truth isolation launch test 啟動真实 actor artifact，檢查 policy subscription 必須只有 `/fly/*` 加 ROS framework topics，禁止 `/world/*`、`/sim/*`、`/eval/*`。每回合記憶重置由 scenario manager 呼叫 `/connectome_policy_node/reset`，不是 actor 訂閱真值情境。

evaluator 提供 `artifact` 時只讀 `model_card.json`，不載入 torch 模型；卡中的 variant 和訓練 seed 優先於手動 `variant` / `training_seed` 參數，並於啟動記錄這項決定。JSONL/CSV 每回合記錄 model card SHA-256。未提供 artifact 時才使用手動標籤，但該類記錄無法通過預設 checkpoint 一致性驗證。

## GRU 訓練入口尺度與暖啟動（#85，2026-11-02）

連接體的氣味 ×100 增益會讓 GRU 的輸入瓶頸飽和。正式矩陣的 GRU override
啟用 `gru_odor_gain_compensation: true`，將氣味 seed features 乘 `1/odor_input_gain`；
其他五變體維持 False。這個常數只抵消連接體所需增益，回到原始 ORN 投影尺度，
不看資料、不新增可訓練參數或另選常數；逐族群 RMS 會移除絕對濃度、引入非線性、
零輸入需 eps 並強制拉平尺度，故不採用（詳見 MODEL_NOTES 的 #85 小節）。

程式預設 False，舊 card 缺欄位仍走原公式，定向測試確認逐位元相容。GRU 的
pretrain_vision → pretrain_odor → BC 必須全鏈一致；`prepare_policy` 若發現 stage
明確指定的補償值與 init_artifact 不同會拒絕，非 bool 也拒絕，不能用暖啟動切換語意。
prepare 實測有效參數仍為 GRU 14,904、hidden 16、瓶頸 18，其他變體有效參數不變。
五個正式 GRU actor（seed 0–4）各在下述 BC puff teacher 序列重播 2×600 步，
凍結程式（35432c3）與新程式的 float32 action 陣列逐位元相同、各自 SHA-256 相同，
新程式載入後的補償欄位均為 False。

訓練前用正式 GRU BC stage 設定（puff 羽流，與評估相同）、torch seed 0–4、normal
teacher 環境 seed 1610000011／1610000012 各 600 步，
要求每個 seed 的瓶頸前 |x|>3 比例 ≤0.05，視覺／氣味時間 std 比 ≥0.02。
本次 True 五個 seed 都為 0／0.056262–0.116145，通過門檻；False 為
0.596481–0.822176／0.000563–0.001161，與 PM 約 0.60–0.80／0.0005–0.0011 一致。
`pretrain_vision` 使用 gaussian 羽流，未補償飽和比例較低（0.285509–0.360509），
所以正式診斷用 BC 的環境；兩份設定的 model 段相同。本次沒有啟動訓練。
這是**事後、未登記**的追加，既有登記結果不改。

## 驗證狀態

`python/tests/test_training.py`（動作 Jacobian、terminal/truncation GAE、非零初始記憶重置、actor input 隔離、seed bootstrap、重新捕獲 censoring、ES tied rank、中斷續訓）已通過；小預算預訓練/BC/PPO 已跑通，ES 尚未實跑。ROS package、launch、錄製與真值隔離測試需要已安裝並建置的 ROS 2 環境及真实 artifact。

checkpoint 目前供保留研究狀態，CLI `init_artifact` 是 actor 權重暖啟動；沒有宣稱恢復完全相同的 optimizer、環境中途狀態或 RNG 軌跡。GPU sparse reduction 的數值可有差異；固定種子不是逐 bit 相同結果的保證。評估需要逐位元重現時（回歸測試、和參考檔比對），在評估 config 設 `evaluation.deterministic: true`，或對 `scripts/diagnostics/policy_episode_eval.py` 設 `DETERMINISTIC=1`（#3）：開啟 `torch.use_deterministic_algorithms`、固定 cuBLAS workspace，CUDA 上的稀疏乘法改用 gather 加分段加總（`flytrack_core.model.sparse.spmm`），因為 cuSPARSE 不受 deterministic 旗標控制，而且不會報錯。這只讓結果固定，不會更準：deterministic GPU、預設 GPU 和 CPU 三者之間仍有 float32 捨入層級的差異，參考檔要用同一個模式產生。訓練沒有提供這個選項（成本 2 倍以上）。
