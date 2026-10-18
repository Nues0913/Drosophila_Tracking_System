# 訓練與評估實作契約

本文描述訓練介面；以下訓練指令供後續流程使用。2026-10-15 的 DN 修正只執行單元測試、lint、設定產生與無權重更新的 CPU 診斷，沒有執行實際訓練或 validate-pretrain；證據見 [DN 修正報告](../../runs/dn_fix/REPORT.md)。

## 責任與介面

`config.py` 管設定、隨機種子、artifact 暖啟動與環境步數帳本；`batches.py` 管感測批次和序列；`pretrain.py`、`imitation.py`、`ppo.py`、`es.py` 各自管理演算法；`evaluation.py` 負責實驗矩陣和種子層級統計。純數學回合指標在 `flytrack_core.metrics`，ROS 評估器不依賴訓練套件。

actor 只接受 `vision[2,721]`、`odor[2]`、`valid[3]`。`valid` 順序為左相機、右相機、成對嗅覺。actor state 是完整 `Tensor[B,S]`，含凍結視覺前端、ORN 適應和中央網路記憶。每次回合開始以 `policy.initial_state` 重建完整 state，不能全數清零：ORN 初始 Hill 閾值和 resting voltage 可為非零。

教師和 critic 從 `info` 讀取真值。critic 的 12 維輸入為 `(agent_x, agent_y, sin(yaw), cos(yaw), target_x, target_y, target_vx, target_vy, previous_v, previous_omega, wind_x, wind_y)`。不把這些欄位合併進 actor observation。

## 共用 YAML 與指令

```yaml
variant: VO
seed: 0
sensor_contract_path: configs/sensor_contract.json
graph_path: data/subgraphs/main/graph.npz
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

路徑以執行指令時的工作目錄為基準，建議在儲存庫根目錄執行。感測契約亦可改用 inline `sensor_contract`。`init_artifact` 載入同 variant 完整 actor 權重並驗證契約、子圖、前端來源；不載入 critic 或 optimizer。`scenario_configs` 可指定情境 → YAML 路徑，依當前環境情境讀取，檔案的 `name` 必須吻合，其餘欄位合併至 `env.scenario_config`；inline 情境欄位優先於檔案。`configs/scenarios/*.yaml` 是情境參數的唯一來源：occlusion、noise、conflict 情境若沒有對應的 `scenario_configs`（也沒有 inline `env.scenario_config`），`make_env` 直接報錯，不再默默改用 `ScenarioSpec` 的程式預設值（#2）；程式預設值已改成與 YAML 相同，只供單元測試與 ROS 未指定檔案時使用。`configs/train/` 下的訓練範本都已加上 `scenario_configs`。

```bash
python -m flytrack_train pretrain --config configs/train/pretrain_vision.yaml --variant VO --seed 0 --output runs/pretrain/VO/seed_0/vision
python -m flytrack_train bc --config configs/train/bc.yaml --variant VO --seed 0 --output runs/bc/VO/seed_0
python -m flytrack_train ppo --config configs/train/ppo.yaml --variant VO --seed 0 --output runs/VO/seed_0
python -m flytrack_train es --config configs/train/es.yaml --variant VO --seed 0 --output runs/es/VO/seed_0
python -m flytrack_train evaluate --config configs/train/evaluate.yaml --output runs/evaluation
python -m flytrack_train export --config configs/train/ppo.yaml --artifact runs/VO/seed_0/actor --output artifacts/VO/seed_0
```

設定檔命名由根目錄配置提供；CLI 支援 `--device auto|cpu|cuda|cuda:0`、`--seed`、`--variant` 與 `--output`。`--variant` 不会重寫 `init_artifact` 路徑，換模型時必須同步暖啟動來源。

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

## DN 活動正則與階段診斷

pretrain、BC 與 PPO 共用 `losses.activity_penalty(policy, activity)`：連接體策略的 `activity_coef` 只乘上非運動 DN 神經元的平方活性平均，分母是非 DN 數量；保留時間／批次軸供原本的 lane／step mask 彙總。全為運動 DN 時回傳可微的零。GRU 保留全部診斷 hidden channels 的平方平均。DN 仍可能透過對其他神經元的回饋受到間接梯度；這不是證明活動正則造成先前沉默的消融結果。

所有新連接體模型使用 `motor_rate_beta=10.0` 的 DN softplus（其他神經元維持 ReLU），完整定義見 [MODEL_NOTES.md](../flytrack_core/model/MODEL_NOTES.md)。舊 artifact 缺少 β 時保留 ReLU；若要暖啟動並切換為新規則，須明確設定 `model: {motor_rate_beta: 10.0}`，新生成的六變體設定已包含此鍵。單純讀取 artifact 不會修改它。

`scripts/dn_health.py --artifact PATH` 是 CPU 唯讀診斷：使用 model card 記錄的世界設定（或 `--config`）與 held-out 種子 `[1.5e9,1.7e9)`，以教師駕駛世界，重播原始與只交換左右眼影像的配對序列。預設 24 回合，各 burn-in 16 步、量 200 步（4×64 步時同一個 BC 模型曾量得 −157% 與 +30% 兩種結果），從量測段前 16 步計算教師動作的尺度正規化 MSE 梯度；回合提早結束會如實記錄實際步數。沒有 optimizer、參數更新或預訓練 decoder。輸出每個 DN 的正活性比例、有效活性比例與 rate 分布、各支路電位分位數、換眼角速度差、`core.*`／`readout.*` 及其群組梯度範數、角速度誤差與固定零轉向基準。

softplus 的正值靜息項不能當作運動控制證據。除原始 `rate>0` 比例，診斷另用 `rate>1e-3` 計算有效活性，要求至少 4 個 DN 在 ≥10% 步數有效、中央梯度範數 >1e-6、視覺換眼角速度差中位數 ≥0.05 rad/s，所有數值有限。BC 階段另要求策略 ω 與教師 ω 的相關 ≥0.5、R²>0（勝過最佳常數）且 ω 標準差 ≥ 教師的 25%；PPO／ES 要求相關 ≥0.3 與同樣的幅度條件。原本的「MAE 比固定零輸出低 20%」會讓完全固定轉向的模型通過（教師平均 ω 不為零），只看相關則會讓方向正確但幅度只有 1/80 的模型通過（其 R² 約 +0.02）。O 變體依設計完全遮蔽視覺，所以不套視覺換眼門檻；GRU 沒有生物 DN 對應，略過 DN gate。診斷不能取代正式 held-out 評估或 validate-pretrain。

`--output JSON` 可儲存報告，`--require-healthy` 使失敗回傳 exit code 2。`--motor-rate-beta 10` 只供舊權重的記憶體反事實探針，輸出會標記，不能當成訓練結果。`--sparse` 可選相同方程的 CSR 算子。`scripts/run_experiments.py` 在每個連接體階段結束（含已完成階段的恢復檢查）重新呼叫此診斷；未通過寫入 `dn_health_failed` 並停止後續鏈；不得為了繼續訓練而默默放寬門檻。

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

pretrain 也使用 `parameter_groups`（#4、#7）：可以設 `core_bias_learning_rate`、`readout_learning_rate`，沒設定時和原本一樣只有單一學習率。auxiliary decoder 另成一組，所以 pretrain 的 optimizer 至少有 2 個參數組。**這個版本之前開始、還沒跑完的 pretrain checkpoint 無法 resume**（`optimizer.load_state_dict` 的參數組數對不上），要從頭重跑該階段；已完成的 pretrain actor 不受影響，可以照常當 `init_artifact`。

先以教師收集精確 60000 步，再依學生訪問的觀測逐步向教師取得標籤。DAgger 收集時學生在 GPU 上對所有平行世界批次推論，每個世界有自己的遞迴狀態；擬合使用與預訓練相同的批次 TBPTT 與提早停止（驗證回合跨所有輪次）。DAgger 逐輪聚合資料，mix probability 為 `beta^(round)`，最後一輪完全由學生行動，標籤仍來自教師。每輪精確 30000 步；即使收集預算在回合中結束，最後的有效前綴仍納入資料。動作 MSE 先除以物理速度範圍，避免兩種單位尺度主導損失。

## Recurrent PPO

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
  value_coef: 0.5
  entropy_coef: 0.01
  activity_coef: 0.0001
  max_grad_norm: 0.5
  critic_hidden_size: 128
  target_kl: 0.03
```

多個 scalar 環境由 trainer 擁有（在 `EnvPool` 子行程中同時 step），逐一取得終局 info 後手動 reset，無 Gymnasium autoreset 覆寫終局觀測的風險。`total_steps` 必須是 `num_envs` 的倍數；最後 rollout 可縮短。每個 TBPTT minibatch 保留連續時間窗口，不以任意單步打散遞迴；窗口初始 state 是該 rollout 的 detached behavior state，窗口內按 episode-start reset。

策略先把 actor 物理均值 `[v,omega]` 轉為 `atanh([2v/v_max-1,omega/omega_max])`，再從 latent Gaussian 抽樣，經 tanh 與物理 affine transform 還原。這保留正向速度 sigmoid readout 的含義，log probability 包含 tanh 與物理縮放的 Jacobian。存 latent sample，避免接近界限時對實際動作做不穩定反算。

真終止 `terminated` 不 bootstrap；時間限制 `truncated` 從 reset 前的真終局 privileged state bootstrap。兩者都截斷 GAE recursion，避免下一回合回報滲漏。critic 和 optimizer 只存在 `training_checkpoint.pt`，部署 `actor/` 只有 actor。

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
  artifact_template: runs/{variant}/seed_{seed}/actor
  environment_seed: 731
  bootstrap_seed: 1301
  bootstrap_draws: 10000
  require_full_matrix: true
  require_equal_budget: true
  require_shared_sources: true
  require_parameter_match: true
  parameter_tolerance: 0.05
```

亦可使用 `evaluation.artifacts` 的 variant → seed → directory mapping。每組 seed/episode 使用同環境種子，跨模型可成對比較。正式評估驗证相同 graph / frozen frontend hash、±5% actor 參數，並以完整 `stage_budget_ledger` 依演算法合計的環境步數比較公平性；帳本繼承初始 artifact，包含預訓練、BC/DAgger 和 PPO 或 ES。

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

## 驗證狀態

`python/tests/test_training.py`（動作 Jacobian、terminal/truncation GAE、非零初始記憶重置、actor input 隔離、seed bootstrap、重新捕獲 censoring、ES tied rank、中斷續訓）已通過；小預算預訓練/BC/PPO 已跑通，ES 尚未實跑。ROS package、launch、錄製與真值隔離測試需要已安裝並建置的 ROS 2 環境及真实 artifact。

checkpoint 目前供保留研究狀態，CLI `init_artifact` 是 actor 權重暖啟動；沒有宣稱恢復完全相同的 optimizer、環境中途狀態或 RNG 軌跡。GPU sparse reduction 的數值可有差異；固定種子不是逐 bit 相同結果的保證。評估需要逐位元重現時（回歸測試、和參考檔比對），在評估 config 設 `evaluation.deterministic: true`，或對 `scripts/diagnostics/policy_episode_eval.py` 設 `DETERMINISTIC=1`（#3）：開啟 `torch.use_deterministic_algorithms`、固定 cuBLAS workspace，CUDA 上的稀疏乘法改用 gather 加分段加總（`flytrack_core.model.sparse.spmm`），因為 cuSPARSE 不受 deterministic 旗標控制，而且不會報錯。這只讓結果固定，不會更準：deterministic GPU、預設 GPU 和 CPU 三者之間仍有 float32 捨入層級的差異，參考檔要用同一個模式產生。訓練沒有提供這個選項（成本 2 倍以上）。
