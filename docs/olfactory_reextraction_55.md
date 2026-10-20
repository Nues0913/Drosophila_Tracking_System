# #55 嗅覺通路候選圖與驗證

2026-10-20；DRI @Nues0913，依使用者直接指派重新萃取。必要非作者 reviewer：@MiLong9487（模型／生物對應／4060）、@cweiting731（共享感測／部署／4070）。[Issue #55](https://github.com/Nues0913/Drosophila_Tracking_System/issues/55) 的 Approved 授權本次候選，不代表 #4 或正式矩陣已驗收。

## 交付與選擇

新增 `configs/extraction/olfactory_v2.yaml` 與 `data/subgraphs/olfactory_v2/`，保留 `main` 圖及訓練預設。原始來源仍為 MaleCNS v1.0 三份官方表，SHA、型別、左右與 NT 規則見候選 manifest；沒有複製 ORN、猜測側別、補人工邊或改 synapse count。新圖 6,244 個神經元、35,627 條邊，graph SHA-256：

`f28f28778497c01f8d33ae80196a9844f9d56f5ee72ba1944b0acf219ad1b531`

| 指標 | 原圖 theta60／視嗅各3跳 | 候選 theta50／視3嗅4跳 |
| --- | ---: | ---: |
| 神經元／邊 | 2,569／14,165 | 6,244／35,627 |
| 被注入的 ORN | 142 | 143 |
| 嗅覺群（下游可達標記） | 613（23.9%） | 3,463（55.5%） |
| 視覺群 | 1,960 | 3,264 |
| 同時屬於視／嗅群 | 4 | 483 |
| **兩模態各三跳內匯流** | **4** | **59** |
| ORN 第一跳目標 | 4 個 PN | 同4個PN＋4個LN |
| LC9／LC11 | 1／1 | 21／20 |
| VO-late 嗅覺分支可達的運動 DN | 0／6 | 6／6 |

重點是補足中間通路，並非增加同一輸入的重複份數。DM1 從73變74；VA2仍69。已知側別的148個候選ORN保留143（96.6%），原表VA2另9個側別未知仍不注入。新圖DM1左35右39、VA2左29右40；不強行左右配平。DM1／VA2仍是兩個工程輸入通道，沒有聲稱涵蓋所有嗅覺受體；模擬 Hill K 的生物合理性仍待 #27。

掃描固定視覺3跳、motor上游5跳，theta40/50/55/60 × 嗅覺3/4/5跳，共12個候選，結果見 [scan JSON](../artifacts/data_execution/olfactory_v2_scan.json)。theta60 即使嗅覺5跳也不能讓每個ORN在隔離分支到達每個DN；theta55嗅4跳同樣失敗。theta55嗅5跳有5,989節點／33,939邊且通路通過，但LC9／LC11只5／8、匯流42。選theta50嗅4跳，比此候選僅多255節點／1,688邊，視覺型別涵蓋較好、匯流59且嗅覺半徑較短。theta40嗅4跳增至9,766／65,437，目前沒有必要先付出這個成本。這是有限掃描的工程選擇，並非最佳生物模型的證明。

## 修正的實際缺口與語意

舊圖在**完整 VO 圖**中，142個ORN均可到6個DN；先前#4核查這一點仍成立。但VO-late會把每條分支限制在該模態 `group_mask` 加 motor：舊圖嗅覺只標3跳，下游雖經視覺路徑留在完整圖，隔離時卻被關掉，使所有嗅覺→DN路徑中斷。不能再以完整圖的可達性替VO-late背書。

萃取器新增 `odor_downstream_hops`，未指定時沿用原 `downstream_hops`。增加 `require_branch_motor_paths: [olfactory, visual]`，在最終固定點剪枝後，按真正分支活性範圍，逐個DN反向查**每個**保留輸入；只從所有輸入一起BFS到任一DN不足以過關。候選兩個分支各自全部通過：嗅143、視583個種子均可到全部6個DN。嗅覺分支各DN的最短／最長範圍合計4–7跳；「嗅4跳」是保留群的聯集半徑，不是每個種子至每個DN最多4條邊。

`max_neurons: null` 明確移除初始3000上限；未填仍保留舊預設3000。圖仍受原始標註、解剖範圍、真實權重門檻、有界視／嗅／motor半徑及固定點剪枝約束。`convergence_hops: 3` 不變，483個重疊群不能當作483個三跳匯流。lesion的exclusive群也隨圖改變（嗅609→2,980，視1,956→2,781），新舊lesion結果不可直接合併統計。

## 實際驗證與限制

- 萃取原始程式commit `8c183b578f6ceeac8402d36f832f6b5dea543678`，manifest另存三個extractor檔案SHA。相同設定連跑兩次，10個檔案全部逐位元相同。
- 獨立 verifier 22項通過：raw／artifact雜湊、原表精確誘導邊集、型別／左右／號誌一致性、有界路徑、三跳匯流、兩分支逐輸入逐DN與manifest一致。原圖19項仍通過。候選0個NT符號補值；原始consensus沒有數值信心欄，不能把0補值說成生物號誌完全確定。
- 一般 `env -u PYTHONPATH .venv/bin/pytest python/tests -q`：**174 passed**。新增回歸覆蓋「完整圖可達但隔離後斷路」、單一種子斷路被聯集掩蓋、獨立嗅覺半徑、保留舊上限／顯式取消，以及校準設定隔離與正式預算保留。Ruff／diff check通過。
- 用真實 frozen frontend、共同G2模型設定、CPU單執行緒、seed55測VO／O／VO-late；同一張空影像，初次有效氣味讀值 `[0,0]`、`[.02,0]`、`[0,.02]`、`[.02,.02]`，之後固定感測碼40個control ticks。這不是環境回放或 trained artifact 評估。

| 40tick最大轉向差（rad/s） | 原圖左／右／雙側氣味 | 候選左／右／雙側氣味 |
| --- | --- | --- |
| VO | 1.57e-6／3.53e-6／5.26e-6 | 5.69e-5／7.87e-6／5.16e-5 |
| O | 5.59e-9／1.81e-6／3.53e-6 | 6.33e-5／1.83e-6／7.02e-5 |
| VO-late | 0／0／0 | 6.82e-5／1.02e-5／8.27e-5 |

新圖初始化轉向更能受氣味影響，仍不是強或正確的控制；前進差至多約3e-8，VO-late仍為0，且左右差不對稱。16tick左氣味對零氣味的轉向contrast對conductance參數梯度最大絕對值：新VO約8.43e-5、新VO-late約1.04e-4，有限；舊VO-late零反應但float32梯度相減仍有約1.6e-12殘差，不能將這種捨入殘差當作通路可訓練證據。完整JSON包含所有數值，沒有把無效果項隱藏。

**NOT RUN：** 新圖pretrain／BC／PPO、訓練後模態使用gate、兩台GPU成本、ROS/Gazebo與G3。連通、初始化反應、非零梯度均不能證明 #4 已解決。尤其BC資料與teacher、bias動態、氣味通道語意仍可能使新圖學不到氣味，需MiLong依#4／#27獨立覆核。

## 證據位置

可直接比對[獨立驗證](../artifacts/data_execution/olfactory_v2_verification.json)、[數值／投射探針](../artifacts/data_execution/olfactory_v2_numeric.json)、[六變體參數](../artifacts/data_execution/olfactory_v2_parameter_budget.json)及[兩次萃取雜湊](../artifacts/data_execution/olfactory_v2_reproducibility.json)。

[CPU原始證據包](https://github.com/Nues0913/Drosophila_Tracking_System/releases/download/explore-4/issue55_olfactory_v2_cpu_evidence.tar.gz)包含log、31項逐檔manifest及實際source／圖／輔助表格；封存來源commit `1a3421e`，extractor仍為manifest記錄的 `8c183b5`。壓縮包SHA-256：`9f8f8492bb6027bdab0cfc683294357a0b2b5f713799dca731db15d136839127`。raw大型原始表不重複打包，依graph manifest SHA取得。

## 參數公平與2–3天目標

六變體實際建構，沿用共同G2設定並重新解GRU預算：

| 變體 | 可訓練actor參數 |
| --- | ---: |
| V／O／VO／RS | 16,663 |
| VO-late | 16,179（−2.90%） |
| GRU | 16,460（−1.22%） |

全部在VO的±5%內，沒有加空參數湊預算。所有實驗若採新圖，須用這一組新預算比較；不能與舊圖6,795參數的結果混成同一公平矩陣。

6,244節點會啟用既有 `dense_threshold: 5000` 的 sparse core。若強制dense，單一float32鄰接矩陣約149MiB（舊25MiB）；這不是訓練peak VRAM，不能用來推算是否可同跑兩條鏈。CPU單批探針僅檢查可執行；過去GPU sparse曾較慢，不能把本次CPU速度當作4060／4070加速承諾。

**2–3天尚未實測保證。** 正式預算不變：6變體×5seed；每鏈pretrain合計240k（V/O各單模態240k，其他各120k）、BC **60k＋3×30k=150k**、PPO2M。全矩陣PPO是60M步；4060原分配seed0/2/4共36M，4070 seed1/3共24M。光PPO若在72h完成，兩台的持續總吞吐至少139／93步每秒，尚未包含任何pretrain、BC、驗收、I/O或重試，故僅是必要下限。不能以這個下限宣布72h可行。

校準後按各variant實測的單／雙工作吞吐與peak VRAM排程，加入資料收集、實際epoch數及最大epoch上界（pretrain40、BC30）、每輪DAgger、驗證／儲存、不同情境rollout成本。計算兩主機各自wall time，取較晚完成者，再列至少25%排程／效能波動餘裕；同時呈現歷史early-stop情境與跑滿epoch上界。短8192步PPO會壓縮curriculum、且從初始化開始，不代表成熟actor的穩態；不足以約束的部分標NOT VERIFIED。若估計上界超72h或OOM，回報瓶頸再由PM決策，不擅自減seed、步數、batch、頻率或改reward。

## 重現命令

從PR固定head的獨立worktree執行，安裝既有extract/train/dev依賴，準備相同raw與frontend；不要覆寫同事正在使用的worktree。

```bash
export PYTHONPATH=python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python scripts/diagnostics/scan_olfactory_graphs.py --output runs/issue55/scan.json
.venv/bin/python -m connectome_extract extract --config configs/extraction/olfactory_v2.yaml
.venv/bin/python scripts/verify_subgraph.py --graph-dir data/subgraphs/olfactory_v2 --output runs/issue55/verify.json
.venv/bin/python scripts/diagnostics/audit_olfactory_candidate.py --output runs/issue55/numeric.json
.venv/bin/python scripts/prepare_experiments.py --graph data/subgraphs/olfactory_v2/graph.npz --output runs/issue55/resolved --seeds 0 --jobs 2
```

最後命令**只解析參數，勿執行產生的run_all.sh**：該工具的正式stage輸出仍使用歷史runs位置。接著用獨立校準產生器隔離輸出，兩套seed供並行測試；沒有啟動任何訓練。下列是4060的seed0／2；4070請分別把兩個 `--seed` 改為1／3，其他設定相同。主機／seed必須寫入交付，兩台不重複認領同一組。

```bash
.venv/bin/python scripts/diagnostics/prepare_graph_calibration.py --parameter-budget runs/issue55/resolved/parameter_budget.json --output runs/issue55/probe0 --seed 0
.venv/bin/python scripts/diagnostics/prepare_graph_calibration.py --parameter-budget runs/issue55/resolved/parameter_budget.json --output runs/issue55/probe1 --seed 2
```

每套產生22份獨立stage設定：pretrain 12k／2epoch、BC 12k teacher＋1×6k DAgger／2epoch、PPO8192；正式batch／sequence_length不動。全部移除init_artifact，拒絕覆蓋既有校準目錄，沒有使用舊graph的checkpoint。`plan.json` 同時記錄探針與正式模板，短預算只屬校準，不會改正式YAML。

## 請兩位teammate測試與Review

本輪授權**每台總GPU占用wall time至多45分鐘，任一單stage至多8分鐘**；含單／雙工作與所有重試，非每個variant各45分鐘。遇OOM／非有限值／契約錯誤停下保留log，不自行調設定重跑。請先在#55回覆固定head與主機，避免干擾既有任務。沒有授權正式矩陣、Gazebo或bag replay。

- @MiLong9487：完整diff與原始圖語意審查；4060測六變體建構及新圖與legacy checkpoint隔離，先VO、O、VO-late的左右氣味／梯度與短pretrain，再PPO成本。覆核#27的ORN／PN／LN／DN標籤、side、NT與型別映射；依#4判定後續learning gate，不將初始化探針當驗收。
- @cweiting731：獨立核對raw／artifact SHA與視嗅分支可達，核對共享frontend與感測／動作契約未改；4070優先VO、VO-late、GRU的PPO單工作與兩工作成本、VRAM。預算有餘再補其他variants與BC；不執行ROS節點或改部署契約。
- 兩台都保留完整命令、git SHA、config、seed、GPU／driver／PyTorch、進程數、thread/worker、開始結束時間、每階段progress、train metrics、peak VRAM（包含同機桌面基線）、errors與雜湊。未測variant/stage列NOT RUN；先回報，不為湊表偷偷延長。

單stage精確入口範例（替換stage與variant時只選產生的設定；pretrain_vision／pretrain_odor的command均為pretrain）：

```bash
/usr/bin/time -v -o runs/issue55/VO-ppo.time \
  timeout --signal=INT --kill-after=30s 8m \
  .venv/bin/python -m flytrack_train ppo \
  --config runs/issue55/probe0/configs/VO/ppo.yaml --device cuda --workers 4 \
  > runs/issue55/VO-ppo.log 2>&1
```

兩工作測試用probe0／probe1同variant的獨立設定同時執行以上入口（各自time/log檔），不得同一output重入；記錄實際重疊區間的合計environment steps／second，不把兩個不重疊的峰值相加。已跑過的probe不覆蓋、不跨設定resume；需要重測用新的明確命名目錄並計入45分鐘。短stage即使完成也只供成本／數值診斷，不通過或取代原訓練gate。證據上傳既有`explore-4` release，兩人各用不重複檔名附SHA，回貼#55／本PR。

## 相容、採用與回退

本PR不改policy方程、前端、感測契約、訓練預設、正式預算或G3完整端到端50ms門檻；主動選新graph才會使用候選。舊圖／模型可留在原固定commit工作樹執行。新graph hash、節點序／type-pair與參數預算不同，**舊pretrain／BC／PPO權重不能遷移或resume到新圖**；需要新一輪訓練驗證。`scripts/verify_subgraph.py`新增CLI也使既有source fingerprint改變，連舊圖resume也應留在舊commit，沒有繞過recovery。匯出artifact內嵌圖的載入流程不變，部署相容仍須review。

兩位必要Review、GPU成本與模態學習結果未齊前，不正式取代main圖、不啟動#28矩陣、不關閉#4。若候選不合用，保留其證據，訓練預設仍指向舊圖；不需回滾ROS／前端工程。
