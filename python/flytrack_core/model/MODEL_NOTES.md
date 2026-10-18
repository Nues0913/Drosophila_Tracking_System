# 模型、固定前端及部署契約

下面描述實作與驗證流程。目前已執行的測試、前端轉換與小規模訓練見 [docs/status.md](../../../docs/status.md)；尚無正式實驗結果。

## 責任與 API

- `graph.py` 驗證中央子圖，以及建立保度數的隨機拓樸控制。
- `frontend.py` 載入真實 flyvis 匯出參數，執行雙眼固定稀疏速率網路與固定空間池化；部署不依賴 flyvis。
- `policy.py` 管理感測編碼、中央神經動態、六種策略及可訓練參數計數。
- `artifacts.py` 保存完整重建設定、來源雜湊、安全權重載入及單模態暖啟動。
- ROS `flytrack_policy/node.py` 負責訊息與生命週期，`runtime.py` 管理感測快取、隱狀態、看門狗與加速度限制。

公開 API 從 `flytrack_core.model` 匯入：

```python
policy = build_policy(variant, graph_path, contract, frontend_path, **model_config)
state = policy.initial_state(batch_size, device)
action_mean, state, activity = policy(observation, state)
save_policy_artifact(
    directory, policy, policy.model_config, contract, graph_path, frontend_path, metadata={}
)
policy, contract, card = load_policy_artifact(directory, device="cpu")
```

觀測只接受 `vision[B,2,N_eye]`、`odor[B,2]`、`valid[B,3]`。有效性依序是左相機、右相機、成對气味讀數。任何額外欄位都被拒絕，包含目標或機器人真值。action_mean 為 `[v,ω]`，單位 m/s 與 rad/s；`log_std[2]` 是 PPO 用的可訓練潛在高斯標準差。`activity[B,N_central]` 為中央群活性；GRU 的同形狀輸出是重複映射的隱通道診斷值，不對應生物神經元。

state 是單一 `Tensor[B,policy.state_size]`，包含雙眼固定前端電位、ORN 適應 K，以及中央電位或 GRU 隱狀態。狀態跨控制步持續，只在回合重置時以 `initial_state` 重建，不能整段歸零，因為 ORN 的初始 K 及前端 bias 不是零。`history_steps=10` 用於資料緩衝與訓練 burn-in；不能每次控制把舊的十步再灌入同一個持續狀態。

感測編碼器以 no_grad 執行，固定 flyvis／ORN 狀態不參與 BPTT；中央／GRU 狀態保留梯度。這防止將含中央梯度的完整 state 拼接張量傳回感測切片時，無意建立整個固定視葉的反向圖。

## flyvis 方案 A 的真實來源

轉換依据閱讀時的官方 API 與原始碼，實際匯出記錄已安裝 flyvis 的版本、checkpoint SHA-256 和網路設定：

- [官方實作與論文來源](https://github.com/TuragaLab/flyvis)
- [Network 官方 API](https://turagalab.github.io/flyvis/reference/network/)
- [官方速率動態與參數乘積](https://raw.githubusercontent.com/TuragaLab/flyvis/main/flyvis/network/dynamics.py)
- [官方共享參數索引與 Euler 積分](https://raw.githubusercontent.com/TuragaLab/flyvis/main/flyvis/network/network.py)
- [官方 photoreceptor 刺激映射](https://raw.githubusercontent.com/TuragaLab/flyvis/main/flyvis/network/stimulus.py)
- [官方 model directory 與 checkpoint 選取](https://raw.githubusercontent.com/TuragaLab/flyvis/main/flyvis/network/network_view.py)

`scripts/export_flyvis.py` 在 Python 3.12 訓練環境使用官方 Network 的設定初始化模型，然後以 `torch.load(..., weights_only=True)` 載入實際官方 checkpoint。它不呼叫可能使用不安全反序列化的上游 checkpoint recovery。bias、time constant、sign、synapse count、synapse strength 依上游 semantic values 及共享索引展開為每個細胞／邊的固定參數。

轉換只支援上游 passive rate／ReLU 動態，其他模型直接拒絕。真正執行匯出時，會以固定灰階、中央閃光、移動正弦刺激，比較載入的官方模型與展開方程；未通過就不輸出可接受的產物。匯出資料包括 parity 結果及其步長。執行端只接受附有合法來源、checkpoint 雜湊、通過 parity 記錄及完整六角映射的產物；缺少真實權重就拒絕所有六變體，沒有 random、identity 或 untrained 前端備援。來源記錄是重現契約，不是對檔案作者的密碼學驗證。

photoreceptor 索引按 `sensing.hex_eye.axial_coordinates` 的 `(q,r)` 順序轉換，即 r 外圈、q 內圈，不能假定上游 node 順序就是感測順序。中央 trainable τ 使用 `2·dt + softplus(raw_tau)`；固定 flyvis τ 依上游規則下限為 dt，以保存來源模型的數值行為。控制週期 50 ms，神經步長 10 ms，各控制步執行五個 Euler 子步。

早期前端來自 flyvis 使用的視葉連接體，中央 graph 來自 MaleCNS。這是已揭露的混合資料來源，不把前端稱為 MaleCNS 重建。官方 checkpoint 轉換已通過 parity（`docs/frontend_execution.md`）；跨 flyvis 版本相容性與 GPU/CPU 數值差異未驗證。

## 感測投射與缺失

中央 graph 使用 `input_mask` 的位元 1、2 指定視覺及嗅覺種子；`group_mask` 的視覺／嗅覺位元則標記完整路徑，兩者可以重疊。不得將所有下游神經當成感覺輸入。沒有 input_mask 時，必須在設定指定種子型別。

固定 flyvis 輸出預設選 T4a–d、T5a–d，各型別切成五個水平空間池化區。視覺種子按其型別與左右側，均勻分配到同側池化區。這個 LC 投射是保留方向資訊的工程假設，沒有宣稱實測 receptive field。`visual_input_mapping` 可用 body ID 字串或 cell type 為 key，值為 `{"types": ["T4a", ...], "bins": [0, ...]}` 覆寫。欲研究更精確的投射，仍需神經元 RF 資料及獨立驗證。

未知側別的感覺種子預設拒絕；只有顯式 `allow_unknown_side_mean=True` 才採雙眼／雙觸角平均，並由 model_config 保留此決定。沒有用未知側別推測左右。

ORN 先以舊 K 計算 Hill response，再以 `K += min(control_dt/τ_adapt,1)·(c-K)` 更新並限制 K≥1e-4。預設 K=0.3、Hill exponent=2、適應時間 2 秒。不同 ORN 型別按穩定 type 順序對應 `odor_K` 各通道；這是可重現的模擬濃度映射，不是特定氣味的實測受體親和力。缺失氣味時 response=0 且 K 凍結；缺失眼睛時固定前端狀態凍結、輸出池化特徵為零。valid mask 同時轉成對應 seed 的可訓練缺失電流，GRU 則直接收到 valid 特徵。沒有把缺失當成濃度真的等於零。

## 六變體與公平性

| 變體 | 實際差異 |
| --- | --- |
| V | 相同中央 graph，禁止氣味輸入及其 valid 通道 |
| O | 相同中央 graph，禁止雙眼輸入及其 valid 通道 |
| VO | 雙感覺輸入與完整中央連接體 |
| VO-late | 兩套互不交換活性的中央狀態，各使用該模態的 induced subgraph，DN 狀態在讀出才拼接；型別參數共享 |
| GRU | 相同固定前端、ORN 轉換及 LC／ORN 種子投射；以投射後種子電流加 valid flags 驅動無拓樸約束 GRUCell 及小讀出 |
| RS | 中央邊經同源號誌分組的 double-edge swap，保留所有節點入／出度、密度及各目標收到的正負邊數 |

V/O 是**感測消融**，保留整張中央圖和同樣參數數，不是重新萃取成更小的單模態圖。VO-late 共享 α／τ／bias 以控制預算，但各支路的 state 和 recurrent currents 完全隔離；這個比較檢验早期神經活性融合與晚期讀出的差異，不能宣稱比較了兩個獨立學習參數集。

所有變體共用同一份固定種子投射。GRU 的輸入維度為 `len(seed_indices)+3`，依穩定 node 順序取得 LC／ORN 種子的外部感覺電流，再加三個 valid flags；它不直接取得投射前的完整視覺池化或 ORN 通道。因此兩種架構接收相同的感測映射資訊，不能由基準額外讀取未提供中央圖的通道。GRU 自動參數預算計算以這個投射後維度為準。

中央 graph 的邊值始終為固定 `s_pre × N_edge × softplus(α_type-pair)`；只有有邊的位置被儲存。τ 和 bias 同型別共享，符號不能隨訓練翻轉。RS 的原始 type-pair sharing labels 留在**邊的參數槽**，不能按隨機後的型別重算，否則參數預算會變。RS 的 labels 因此不再代表隨機後的實際 type-pair，model_card 明示這項限制。無法產生至少一個有效且改变拓樸的 swap 時拒絕 RS，不把原圖冒充隨機控制。

參數計數只含 `requires_grad=True`，包括 readout、缺失電流及 PPO log_std，排除共同凍結前端。V/O/VO/RS 的中央參數數相同；VO-late 的讀出可能稍大，而兩支路均停用的 type-pair 不配置可訓練 α，避免把不會参与運算的參數算進預算。缺失電流也只配置感覺種子型別。`budget_target` 用於 GRU 自動選最接近的整數 hidden size，`assert_parameter_budget` 用於所有比較。±5% 不可達時必須拒絕／調整有功能的架構，沒有補未使用的假參數。極小測試圖可能無法達到 ±5%，不可當成公平實驗結果。

motor readout 預設在有左右成對 turn DN 及 forward DN 時使用結構讀出；資料不足則明確記錄 linear DN fallback。完全沒有 DN 直接拒絕，不改讀普通中央神經來假造運動端點。

## 運動 DN rate 與沉默修正（A＋C）

新建立的連接體策略 V、O、VO、VO-late、RS 共用同一規則：對 `turn | forward` 運動 DN，`r(v) = log(1 + exp(βv)) / β`，設定 `motor_rate_beta=10.0`；其他中央神經元仍用 `relu(v)`。實作用數值穩定的 `F.softplus(v, beta=β)`。這是避免負電位阻斷動作梯度的**工程假設，並非生物學測量或擬合得到的放電曲線**。沒有加正 bias 限制或額外動作輔助頭。

`MaskedRateCore.rates` 同時用於遞迴輸出、回傳的 activity 與 MotorReadout。VO-late 的兩個獨立支路也如此；停用節點／支路的 rates 明確乘上 active mask，避免 `softplus(0)` 漏出靜息電流。非 DN 的 rate、Euler 方程與積分步長不變，但有 DN 回饋連結時，其他神經元的數值軌跡可以隨 DN 輸出改變。`sense()`／`forward_code()`、`adjacency_cache()`、dense `addmm`／`lerp` 與部署組合 API 保留。

靜息 `r(0)=ln(2)/10≈0.0693147`；相同的左右 rates 在結構轉向讀出中相消，前進讀出仍可能有小幅基線偏移。ReLU 與此 softplus 都是 1-Lipschitz，後者導數 `sigmoid(βv)` 介於 0 和 1；因此原本 `ρ(|W|)<1` 的初始化界仍適用。在適當正權重的無窮範數下可取 `|W|q≤cq`（`ρ(|W|)<c<1`），Euler 差分的界為 `max_i(1-r_i+r_i c)<1`，其中 `r_i=dt/τ_i∈(0,1/2)`。active／edge mask 只會減少此上界。softplus 靜息項會移動固定點，不改變差分收縮界；訓練後 α 可變大，這不是全訓練期間的穩定保證。極負電位下浮點數仍可能下溢，不能宣稱永不沉默。

motor mask 是不存入 state_dict 的固定 buffer，β 是設定值，不新增可訓練參數、不改圖、Dale 號誌、突觸數或 edge mask；GRU 的動態不變。既有 model card 若沒有 `motor_rate_beta`，載入時明確補 `None` 以保留舊 ReLU 語意，不能把舊模型默默當成已修模型。新產物記錄 β；暖啟動舊 artifact 時須在訓練 `model` 明確給 `motor_rate_beta: 10.0` 才會在記憶體切換 rate，之後存出的新 actor 記錄此設定。

活動正則排除運動 DN，定義及限制見 [TRAIN_NOTES.md](../../flytrack_train/TRAIN_NOTES.md)。修正報告見 [docs/dn/fix_A_C_2026-10-15.md](../../../docs/dn/fix_A_C_2026-10-15.md)，後續的尺度與環境修正見下一節及 [docs/fixes_2026-10-17.md](../../../docs/fixes_2026-10-17.md)。

## 初始化尺度、DN 輸入增益與讀出偏置（2026-10-17）

**全網譜半徑上限。** 邊值 `s_pre × N_edge × g_edge × softplus(α)` 中，萃取門檻讓 `N_edge ≥ 60`（中位數 92）。固定 `alpha_init=0.01` 時真實圖 `ρ(|W|)≈32`，ReLU 動態每控制步放大約 2×10⁴ 倍，第 9 步溢位。初始化因此取 `α = min(alpha_init, init_spectral_radius / ρ(|N·g|))`，`init_spectral_radius=0.9`，真實圖得 α≈2.56×10⁻⁴。`ρ` 以每輪依目前估計值重新縮放的位移冪迭代求得：舊版固定 `+I` 位移相對於數千的突觸數可忽略，加上高增益列後不收斂（測試圖上把 α 多壓了約 2 倍）；新版在真實圖上與稠密特徵值解一致到 1e-15，約 0.2 秒。

**DN 輸入增益 `motor_input_gain`（預設 100）。** 一刀切的上限把接進 6 個運動 DN 的 147 條邊也壓到 2.6×10⁻⁴，嗅覺預訓練後方位資訊雖已傳到 DN 左右差（R² 0.66），幅度只有約 3×10⁻⁴，BC 的轉向梯度因此極小（[診斷](../../../docs/dn/diagnosis_2026-10-16.md)）。現在每條終點為運動 DN 的邊乘上固定增益 `g_edge=100`（其他邊為 1），依邊的實際終點計算，RS 重連後同樣適用。它是不存入 state_dict 的 buffer，不新增參數，譜半徑上限連同增益一起計算；真實圖上 DN 不在主導迴路上，`ρ` 仍是 3517.85，其他邊的 α 不受影響。增益反事實測試顯示只放大這段時資訊保留、數值穩定（k=100），連上游一層一起放大則破壞資訊、全網放大 1.5 倍即發散。這是**工程上的尺度修正，不是生物測量**。本機限時驗證（視覺 pretrain 60k＋短版 BC 18k）中，DNa01 左右被壓到電位約 −1～−1.7（沉默），DNa02 的左右差平均 −0.8、時間變動約 0.08，短版 BC 未過轉向門檻；增益值需在完整預算下確認（見 [docs/fixes_2026-10-17.md](../../../docs/fixes_2026-10-17.md)）。

**讀出 ω 偏置 `readout_turn_bias`（預設開）。** 連接體左右兩側接進 DN 的連結不對稱，目標在正前方時 DN 左右差也不為 0（k100 起點約 −0.146）。偏置用來抵消這個結構偏差，讓轉向權重只負責依方位調整；在隨機方向的環境中 teacher 平均 ω≈0.011，偏置不代表行為上的轉向偏好。

舊 model card 沒有這兩個鍵時，載入補 `motor_input_gain=1`、`readout_turn_bias=False`，保留原本推論語意。

**嗅覺通路選項（#4，都是可選設定）。** 已訓練的 VO 策略動作完全不受氣味影響：非運動神經元的 ReLU 把氣味訊號截斷，關掉後梯度也是 0；訓練前氣味對 ω 的影響也只有視覺的約 1/2500。兩個可選設定：
- `olfactory_rate_beta`（預設 `None` = ReLU）：嗅覺 population 中非運動 DN 的節點改用 `softplus(v, beta)`，讓負電位時仍有梯度。和運動 DN 的 softplus 一樣是工程假設，不是生物測量；分支遮罩照常把關閉分支的 rate 設為 0。
- `odor_input_gain`（預設 1）：只放大 ORN 外部輸入（`odor_projection`），不改循環權重和譜半徑。GRU 共用同一個 encoder，所以也會受影響。

量測工具是 `scripts/diagnostics/modality_sensitivity.py`（teacher 駕駛 200 步，最大 |Δω|，單位 rad/s）。目前的證據：

| 階段（VO seed 0） | 增益 | 去掉氣味 | 去掉視覺 | 氣味／視覺 |
|---|---|---|---|---|
| 初始化 | 1 | 6.7e-6 | 5.6e-4 | 0.012 |
| 初始化 | 100 | 6.9e-4 | 5.6e-4 | 1.24 |
| vision pretrain 2400 步（38 次更新） | 1 | 3.2e-9 | 1.6e-3 | 2e-6 |
| vision pretrain 2400 步 | 100 | 6.8e-6 | 1.6e-3 | 4e-3 |
| 從初始化直接短版 BC（18k） | 1 | 4.5e-5 | 0.027 | 0.0017 |
| 從初始化直接短版 BC（18k） | 100 | 3.6e-3 | 0.027 | 0.134 |
| 從初始化直接短版 BC，加 softplus（β=10） | 100 | 9.9e-4 | 0.026 | 0.038 |

- **增益 100 不能當預設**：odor pretrain 的線性 decoder 直接讀嗅覺 rate，增益 100 時 rate 放大約 100 倍（初始化最大 136），loss 發散（4060：0.69 → 208）。要用大於 1 的增益，pretrain decoder 的輸入必須先做尺度處理。
- **vision pretrain 本身就會壓掉氣味**：把 pretrain 後的參數逐類還原成初始值，只有還原 bias 才能救回氣味；alpha、tau、readout 都沒有效果。其中是「只屬於視覺」的 680 個 type 的 bias 改變了網路的工作點：還原它們，氣味的影響回到 5.6e-6，但視覺增加的敏感度也一起消失。只凍結嗅覺 type 救不回來（7e-8）。全部凍結（`core_bias_learning_rate: 0`，pretrain 現在也支援）可以完整保住氣味，但 vision pretrain 第 1 個 epoch 的 loss 從 0.518 變差到 0.619。
- 這些都只是 2400～18k 步的小實驗。完整流程（pretrain → BC）下比例能不能守住，要等 4060 上的縮小鏈。

新 model card 會記錄這兩個值；舊 card 沒有 `odor_input_gain` 時補 1.0，沒有 `olfactory_rate_beta` 時是 `None`，語意不變。`scripts/diagnostics/modality_sensitivity.py` 量測 teacher 駕駛下去掉氣味、氣味 ×10、左右交換、去掉視覺時的最大 |Δω|，以及氣味／視覺比例。

## 執行時 lesion 與匯流電流分析

`policy.set_lesion("none" | "vision" | "odor")` 不增加、移除或重訓參數。VO/RS 等中央圖策略除了停止該模態的感測和缺失電流，也將該模態獨有中央群的電位與所有輸入／輸出邊靜默；兩路共享群、匯流群及 DN 保留。VO-late 則將對應的整個獨立支路靜默，另一支路照常。GRU 只停止相應輸入，沒有生物神經細胞 lesion。移除 lesion 後使用現有持續狀態，恢復過程本身會有暫態；獨立比較回合應重建 initial_state，不能混用兩種評估口徑。

`policy.convergence_current_contributions(activity)` 回傳三個 Tensor `[B,N_convergence]`：`visual_exclusive`、`odor_exclusive`、`shared`。每項是該來源群到每個匯流細胞的 **Σ|邊電流|**，不是抵消後的 |Σ電流|，也不是因果貢獻；群別依 presynaptic 節點的 visual／olfactory 路徑位元定義。來源號誌、邊數、突觸數和活性幅度都可能影響這個比值，不能只憑大小推論某模態必要性。因果相關問題要搭配 lesion 行為實驗。

`convergence_node_ids` 提供輸出欄位對應的 body ID。VO-late 使用最近一次 forward 保存的兩支路活動與各自遮罩分別計算，再加總電流質量；必須傳入最近該次 forward 回傳的 activity，因為已合併的平均活動不能還原兩套獨立電流。GRU 拒絕此生物電流分析。`describe()` 明示 variant、lesion、活動語意及參數數；所有分析與 lesion 執行仍待後續在真實模型產物上驗證。

## 產物與 ROS

各模型產物包含 `model_card.json`、`weights.pt`、`graph.npz`、`frontend.npz`、`sensor_contract.json`。card 包含解決後的完整 model_config、各檔案 SHA-256、感測契約及小眼方向表雜湊、訓練參數數、狀態形狀及來源。weights 只序列化 Tensor state_dict，載入使用 weights_only=True；來源或幾何不一致拒絕啟動。card 最後原子替換，未完整寫入的產物不會通過雜湊檢查。

ROS executable 和 node name 為 `connectome_policy_node`；參數為 `artifact`、`sensor_contract_json`、`device`、`use_sim_time=true`。只訂閱三個 `/fly/*` 感測 topic，不訂閱 `/world/*`。固定 absolute reset 服務為 `/connectome_policy_node/reset`（std_srvs/Trigger），提供情境管理器在每回合重置時清空隱狀態、歷史、所有感測時間戳並立即發布零速度；僅接受嚴格晚於重置模擬時間的感測包，避免 DDS 中上一回合排隊但仍未逾時的資料重新填滿快取。重置後三路感測都重新有效且未逾時，才恢復第一次動作。

看門狗以保留的感測生成時間戳和模擬時鐘計算 200 ms 逾時，拒絕亂序包；單一感測缺失使用 mask 繼續，全部缺失立即零速度。時鐘倒退清空整個 actor 與快取，未來時間戳也不能拿來恢復動作。速度與加速度限制依共同 SensorContract，activity 以 10 Hz 發布。ROS 處理灰階及 RGB/BGR 的影像 stride，不依賴 cv_bridge。

本輪只做靜態閱讀。後續仍需實際執行：官方權重轉換 parity、核心／CSR 梯度測試、六變體產物與參數預算測試、ROS 真值隔離及 reset/watchdog 跨節點測試、GPU 推論延遲量測。沒有聲稱 p95<50 ms、模型已訓練，或 Gazebo 已閉環成功。

`/fly/inference_latency_ms`（std_msgs/Float64、best effort depth 5）是額外診斷 topic，以 perf_counter 測量 tensor 準備到動作與 activity 已實體化到 CPU 的處理 walltime；GPU 的 `.cpu()` 會同步完成運算。這不含相機 callback 的複眼取樣時間，也不是感測生成到 cmd_vel 的端到端延遲；感測時間戳對 cmd_vel 的模擬時間差必須另行記錄並標明量測基礎。
