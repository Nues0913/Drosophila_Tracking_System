# 訓練與評估實作契約

本輪只新增程式碼及文件，未執行測試、訓練、推論、ROS build 或依賴安裝。以下指令供後续流程使用，不能視為已通過驗證。

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

路徑以執行指令時的工作目錄為基準，建議在儲存庫根目錄執行。感測契約亦可改用 inline `sensor_contract`。`init_artifact` 載入同 variant 完整 actor 權重並驗證契約、子圖、前端來源；不載入 critic 或 optimizer。`scenario_configs` 可指定情境 → YAML 路徑，依當前環境情境讀取，檔案的 `name` 必須吻合，其餘欄位合併至 `env.scenario_config`；inline 情境欄位優先於檔案。

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
  epochs: 20
  sequence_length: 32
  learning_rate: 0.001
  exploration_probability: 0.2
  odor_presence_threshold: 0.01
  angular_velocity_coef: 0.1
  activity_coef: 0.0001
  max_grad_norm: 0.5
```

collection 精確消耗 `total_steps`，不因提早碰撞而縮減預算。監督式 TBPTT 隨機排列回合，保持每回合窗口順序；完整記憶在窗口間持續，但梯度在邊界 detach，避免每個窗口重算整段前綴的二次複雜度。窗口之間更新權重，保留上一窗口產生的 detached state，這是串流 TBPTT 的近似。產出 `pretraining_checkpoint.pt` 保留輔助頭供研究診斷，`actor/` 完全不含輔助头。

正式公平流程可讓 V/O 各用 240000 步，VO/VO-late/GRU/RS 各先視覺 120000 步，再以該次 `actor/` 作 `init_artifact` 做嗅覺 120000 步。模型各自暖啟動，可避免從生物模型替基準模型挑選較好的初始權重。

另有選配 `pretrained_artifacts: {vision: PATH, odor: PATH}`，按模態移植連接體 type / type-pair 參數，交集平均，保留新任務 motor readout。限連接體 family、要求相同圖與前端來源；不能與 `init_artifact` 同時使用，也不能套到 GRU。

## BC 與 DAgger

```yaml
init_artifact: runs/pretrain/VO/seed_0/odor/actor
train:
  teacher_steps: 60000
  dagger_steps_per_round: 30000
  dagger_rounds: 3
  dagger_beta: 0.5
  epochs: 10
  sequence_length: 32
  learning_rate: 0.001
  max_grad_norm: 0.5
  activity_coef: 0.0001
```

先以教師收集精確 60000 步，再依學生訪問的觀測逐步向教師取得標籤。DAgger 逐輪聚合資料，mix probability 為 `beta^(round)`，最後一輪完全由學生行動，標籤仍來自教師。每輪精確 30000 步；即使收集預算在回合中結束，最後的有效前綴仍納入資料。動作 MSE 先除以物理速度範圍，避免兩種單位尺度主導損失。

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

多個 scalar 環境由 trainer 擁有，逐一取得終局 info 後手動 reset，無 Gymnasium autoreset 覆寫終局觀測的風險。`total_steps` 必須是 `num_envs` 的倍數；最後 rollout 可縮短。每個 TBPTT minibatch 保留連續時間窗口，不以任意單步打散遞迴；窗口初始 state 是該 rollout 的 detached behavior state，窗口內按 episode-start reset。

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

2D latency 是 sensor tensor 組装至 actor CPU 動作輸出的 wall time，CUDA 會同步，排除 rendering。ROS 主 latency 是保留原 sensor timestamp 到收到 cmd 的 simulation-clock age，因 Twist 沒有 header，無法精確關聯是哪一筆輸入；這是診斷資料，不能單獨拿來驗收 wall-time end-to-end p95 <50ms。

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

checkpoint 目前供保留研究狀態，CLI `init_artifact` 是 actor 權重暖啟動；沒有宣稱恢復完全相同的 optimizer、環境中途狀態或 RNG 軌跡。GPU sparse reduction 的數值可有差異；固定種子不是逐 bit 相同結果的保證。
