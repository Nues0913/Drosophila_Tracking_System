# FlyTrack 三 Agent 分工與 GitHub 協作規範

> 適用專案：[Nues0913/Drosophila_Tracking_System](https://github.com/Nues0913/Drosophila_Tracking_System)
> 制定：2026-10-18。合併到 `main` 後生效，三個 Agent 在每次開始工作前都要先讀。
> 本文件是團隊政策，**不是** GitHub 已經自動強制的權限或保護規則；目前沒有技術手段強制的部分由 PM 人工把關（見第 8 節）。

## 0. 給所有 Agent 的簡短規則

1. 開始工作前，先讀本文件、最新的 [docs/status.md](docs/status.md)，以及指派給自己或可以自行認領的 Issue。
2. **一項工作對應一位負責人（DRI）、一個 Issue、一條分支、一個 PR。** 聊天訊息、Agent 對話和 commit message 都不能取代 Issue。
3. **自己主責領域內、低風險、低成本、不改規格的小 Bug**：可以自行開 Issue 或認領、修正並送 PR，不必等 PM 指派（第 4.1 節）。**跨領域、高成本、研究設計或驗收標準的變更**：要先取得 PM 在 Issue 留下的 `Decision: Approved`（第 4.2 節）。
4. 不搶別人已經負責的工作，也不在未協調的情況下修改別人主責的模組。
5. 所有修改都走 PR。每個 PR 至少要有一名**非作者**依第 5.1 節完成審查，**PM 自己的 PR 也一樣**。最後由 PM 決定是否合併。
6. 重度測試和正式訓練由 @milong9487 與 @cweiting731 分攤；@Nues0913 不在低效能主機上重跑。
7. 只回報**實際跑過**的結果，並附上可查證的 log、設定、版本和產物；沒跑的標成 `NOT RUN` 並說明原因。

## 1. 成員與決策權

| 成員 | 角色 | 核心職責 | 不承擔 |
| --- | --- | --- | --- |
| **@Nues0913** | 主 Agent／PM／架構負責人／最終合併決策者 | 需求與里程碑、研究方向、架構與跨模組介面、任務分派與優先級、審查程式碼與 Bug、風險管理、整合驗收、全域文件、合併與發版 | 正式訓練、長時間 GPU 工作、完整實驗矩陣、長時間 Gazebo 閉環 |
| **@milong9487**（GitHub 也顯示為 `MiLong9487`） | Teammate 1：模型與訓練負責人 | 果蠅神經模型、感官整合、模型診斷、訓練演算法與效能、實驗設計與實作、統計分析、訓練與模型相關的重度測試 | 自行變更研究假設、正式實驗預算或驗收門檻、跨模組介面 |
| **@cweiting731** | Teammate 2：模擬與部署負責人 | 2D 世界與感測、Gazebo／ROS 2、控制與節點整合、環境一致性、整合與閉環測試、容器與部署，並**分攤正式訓練與重度驗證** | 自行變更神經模型設計、正式實驗預算或驗收門檻、跨模組介面 |

- **PM 不是所有工作的執行者，也不審查自己的 PR。** PM 審查 teammate 的 PR 時，可以只根據程式碼、差異、測試結果和 teammate 提交的證據下判斷，不能因為自己的主機效能較低，就要求 teammate 重跑整個訓練矩陣。
- **領域主責和算力執行是兩件事。** 模型與訓練規格由 @milong9487 維護，但需要 GPU 的實際工作可以分派給兩名 teammate。@cweiting731 執行訓練任務，不代表取得修改模型研究設定的權限。

## 2. 模組責任邊界

| 範圍 | 主責 | 協作與審核規則 |
| --- | --- | --- |
| 系統架構、需求範圍、跨模組 API、模型比較規則、正式驗收門檻 | @Nues0913 | teammate 可以提案，PM 批准後才改規格 |
| `python/connectome_extract/`、`python/flytrack_core/model/`、`configs/model/`、模型與子圖診斷 | @milong9487 | 跨感測契約、模型變體或影響實驗公平性的變更須 PM 核准 |
| `python/flytrack_train/`、`configs/train/`、`scripts/run_experiments.py`、`scripts/prepare_experiments.py` | @milong9487 | 訓練設定、效能、checkpoint、recovery；不得自行放寬正式預算與 gate |
| `scripts/dn_health.py`、`scripts/diagnostics/`、正式 2D 結果統計 | @milong9487 | 需要改世界或感測實作時，先開跨模組 Issue 邀請 @cweiting731 |
| `python/flytrack_core/world/`、`python/flytrack_core/sensing/`、`python/flytrack_sim/`、`configs/scenarios/` | @cweiting731 | 影響 teacher、reward、資料分布或模型輸入時，須 @milong9487 複核、PM 核准 |
| `ros2_ws/`（interfaces、policy、world、eval、bringup）、`docker/`、`compose*.yaml` | @cweiting731 | 模型匯出格式與感測契約的變更須和 @milong9487 協調、PM 核准 |
| Gazebo 閉環、ROS 真值隔離、sensor-to-command 延遲、部署與故障恢復測試 | @cweiting731 | @milong9487 支援模型 artifact 與推論問題 |
| `python/tests/` 等測試 | 依被測模組決定 | 跨模組的測試至少通知另一位主責人 |
| `configs/sensor_contract.json`、`configs/evaluation_protocol.yaml`、`README.md`、`docs/status.md`、`後續計畫.md`、本文件 | @Nues0913 | teammate 可以送 PR，但不得自行重寫或直接推上 `main` |
| 各模組的筆記、實驗與驗證報告 | 該模組主責人 | 報告須附來源與設定，區分實測和未實測；跨模組結論由 PM 審核 |

**「主責」不是檔案鎖。** 任何 Agent 都可以閱讀、分析別人的模組，並為發現的問題開 Issue；但不得在未協調的情況下直接修改別人主責的模組（第 4.2 節）。

## 3. 算力與重度測試分配

| 成員 | 執行內容 |
| --- | --- |
| @Nues0913（低算力審查端：GTX 1050 3 GB，跑不了 PPO） | 靜態閱讀、差異審查、設定與架構一致性檢查；輕量的 lint、單元測試，以及本機 Docker 的短時間 ROS 節點檢查（**能跑才跑，不是合併前提**）。依 teammate 提供的真實執行紀錄決定 approve、request changes 或 merge，不把沒跑過的測試寫成通過。 |
| @milong9487（訓練與模型驗證主執行者） | 建立共同的實驗設定、參數預算、模型健康 gate 和訓練評估流程；跑模型敏感度、DN／ORN 診斷，以及 pretrain／BC／PPO／ES 的重度測試；對兩名 teammate 產出的權重、checkpoint、紀錄做一致性驗證；提供標準訓練指令和預期輸出給 @cweiting731。 |
| @cweiting731（ROS／模擬主執行者，兼 GPU 訓練） | 跑 ROS／Gazebo 映像、啟動、閉環、感測、真值隔離、latency、回合重置、物理一致性等重度測試；分攤正式訓練矩陣中**明確指派**的變體／種子，並以相同格式交回權重、log 和報告。訓練時發現模型缺陷，先在原 Issue 留證據並通知 @milong9487，**不自行改模型核心來讓它跑過**。需要部署用的模型 artifact 時，由 @milong9487 提供已驗證的版本和來源。 |

### GPU 任務派工

1. PM 先批准**共同的程式碼版本、設定、實驗方案與驗收條件**，再開 GPU 執行 Issue。
2. Issue 內逐一列出 `variant`、`seed`、階段、執行主機、預估時長與負責人。**同一組 `(variant, seed, protocol version)` 只能有一位執行負責人**，不重複算，也不漏算。
3. 初始提議：@milong9487 跑種子 0、2、4，@cweiting731 跑種子 1、3，六個變體都算在內（以種子分工的理由見 [docs/gpu_hosts.md](docs/gpu_hosts.md#分工跑正式訓練)）。這只是提議，要等 GPU 能力和可用時段確認後，以 Issue 中核准的清單為準，也可以依實際速度改派。
4. 先用同一個變體、單一種子校準兩台主機的速度與可重現性，再決定分配比例。不假設兩台電腦一樣快，也不假設 @cweiting731 一定有足夠的 VRAM。
5. 每組都要保存：Git commit SHA、設定與來源雜湊、套件版本與 GPU 資訊、完整命令、seed、訓練時間、metric、模型檔位置、退出狀態與失敗原因。`runs/` 和權重不進 Git，放在 PM 指定的共享位置並附校驗碼。
6. 跨主機環境造成結果不可比時，先停下該組並開 Issue。**禁止私自調整 batch、總步數、情境強度、成功門檻或模型規格來湊結果。** worker 數之類的純執行參數也要記錄，並確認不影響公平性。
7. 某組失敗時，只能依已核准的 runner 策略重試或隔離（`run_experiments.py --keep-going` 只讓其他組繼續跑，不會把失敗算成通過）。gate 失敗不能當成通過；所有失敗和未完成的組都要列出來。

**驗收分工**：模型功能、訓練可重現性和 2D 實驗的主要證據由 @milong9487 提供；ROS／Gazebo、部署和閉環的主要證據由 @cweiting731 提供。PM 做整體的最終驗收。

## 4. Issue：問題、任務與決策的唯一來源

所有**非瑣碎**的功能、Bug、重度測試和跨成員工作，都要先有 Issue 再開 PR。有 Issue 不代表要等 PM 派工：符合第 4.1 節條件的工作可以直接開始；其他工作依第 4.2 節處理。

### 4.1 同領域小 Bug：自行開立、認領、修正

必須**同時**符合以下四點：

1. 位於自己的主責模組（第 2 節），修改範圍是局部的，能用既有規格判斷對錯。例如空值處理、明顯的程式錯誤、補局部回歸測試。
2. **不改**跨模組公開 API、感測或動作契約、資料格式、模型架構、正式訓練設定或預算、評估指標或成功門檻、研究假設、共用的排程策略。
3. 只需要日常就能負擔的驗證：不需要長時間 GPU 訓練、正式實驗矩陣或長時間 Gazebo 閉環，也不佔用別人的 GPU 或已排定的工作。
4. 不涉及重大安全風險、正式結果的可信度、實驗公平性或已知的 P0／阻斷性問題，也不影響別人進行中的 Issue 或 PR。

步驟：

1. 先搜尋有沒有相同的 Issue 或進行中的 PR。
2. 沒有就自己開 Issue，有但沒人處理、又屬於自己主責範圍時可以認領。寫明可重現的現象、目標、範圍和測試方式，類別標 `self-claim:small-bug` 並說明理由。
3. 把 GitHub Assignee 設為自己（唯一的 DRI），並留言說明修正範圍。
4. 直接建分支、修改、跑自己能力範圍內的驗證，然後送 PR；證據還不齊時先開 Draft PR。PR 連結 Issue，依第 5.1 節指定 Reviewer。
5. PM 審查後決定是否合併；DRI 補上測試證據並更新 Issue，符合完成條件後才關閉。

這項授權**只免除開工前的 PM 指派或批准**，不免除 Issue 紀錄、必要的 Review、測試證據和 PM 的合併決定。PM 也不能對這類小 Bug 額外要求事前批准。

### 4.2 必須先經 PM 批准的工作

符合以下任一項，Issue 標 `needs:pm-approval`，提出者要附上影響分析、建議方案、風險、預估算力與時間，以及受影響的領域。**討論完成前不改相關程式，也不啟動重度工作。**

- **跨領域**：要修改另一位 teammate 主責的模組、跨模組契約，或需要兩個領域一起改。即使是小 Bug，也要由 PM 指定負責人和協作方式。
- **高成本**：長時間 GPU 訓練、正式或大量的回歸實驗、長時間 Gazebo 閉環，或會佔用共享算力、延後已排定工作的測試。
- **研究與規格變更**：模型架構、感測或世界設定、teacher 或 reward、研究比較與公平性、seed 數或總步數、gate 或驗收門檻、正式評估協定、跨模組 API，以及會影響既有實驗結論的修改。
- **高風險或阻斷性**：造成控制失效、資料遺失、實驗結果可能無效、主要流程無法繼續，或需要緊急停機的問題。可以立即停止受影響的實驗、保存證據並回報，但修復方向由 PM 決定。

PM 在 Issue 留下 `Decision: Approved`、`Decision: Rejected` 或 `Decision: Needs changes`，寫明 DRI、允許範圍、必要的驗證和資源安排。只有 Approved 之後才能在核准範圍內動工；超出範圍要再請示。**只有 PM 能留 `Decision:`**，未確認的建議不能當作批准。

### 4.3 指派、狀態與紀錄

- **Assignee 只設一位 DRI**；協作者用 @mention、sub-issue 或 checklist 表示。可以獨立交付的工作拆成不同 Issue。
- 自行認領只限自己的領域：@milong9487 認領模型與訓練的小 Bug，@cweiting731 認領模擬、ROS 與部署的小 Bug。PM 可以自行建立和指派任何任務。
- 整體優先級、跨主責改派、需求凍結、重大實驗與驗收門檻都由 PM 最終決定。處理自行認領的小 Bug 不能干擾既有的優先任務；有衝突時先停下並通知 PM。
- 已經有人負責或正在修的 Issue 不能搶走；要變更 DRI 或範圍須先協商，跨主責的改派由 PM 決定。
- 任務狀態：`待確認` → `可自行處理` 或 `等待 PM 決策` → `進行中` → `PR Review` → `已合併／待驗證` → `完成`。目前在 Issue 內文或留言追蹤。
- Label 命名：`area:model`、`area:train`、`area:simulation`、`area:ros`、`area:infra`、`type:bug`、`type:feature`、`type:experiment`、`self-claim:small-bug`、`needs:pm-approval`、`blocked`、`priority:P0`／`P1`／`P2`。這些 label 由 PM 在本文件合併後建立，建立前先寫在 Issue 內文裡。
- 新證據和阻礙都回到同一個 Issue 留言。

### Issue 內文範本

```md
## 背景／現象
（現況、證據、問題為什麼重要）

## 目標與範圍
（要做與不做的事）

## 任務類別與啟動授權
類別：`self-claim:small-bug` 或 `needs:pm-approval`（擇一並說明理由）
啟動狀態：`可自行處理` / `等待 PM Decision:` / `PM 已批准`

## 負責與協作
DRI：@Nues0913、@milong9487 或 @cweiting731（擇一）
協作：@其他成員（如需要）
決策與驗收：@Nues0913

## 實作或驗證方案
（模組、預期改動、測試方法；適用時寫 GPU 主機與 seed）

## 完成條件
- [ ] 預期行為有證據
- [ ] 相關測試完成，或註明未執行及原因
- [ ] 文件與相依 Issue 已更新
- [ ] PR 已由第 5.1 節的必要 Reviewer 審查，PM 已決定合併

## 風險／需要 PM 決定
（沒有就寫「無」；跨模組介面、實驗規格、資源和未解的問題要具體列出）
```

## 5. PR：程式碼變更與審查的唯一入口

### 5.1 必要 Reviewer 與合併權

| PR 作者／類型 | 必要 Reviewer | 合併決定 |
| --- | --- | --- |
| @Nues0913：模型或訓練 | @milong9487 | @Nues0913 |
| @Nues0913：模擬、ROS 或部署 | @cweiting731 | @Nues0913 |
| @Nues0913：同時涉及兩個領域，或全域文件與規範 | @milong9487 **和** @cweiting731 | @Nues0913 |
| @milong9487 或 @cweiting731：自己的領域 | @Nues0913 | @Nues0913 |
| 任何跨領域 PR | 所有受影響領域中非作者的主責人，再由 PM 確認 | @Nues0913 |

- **禁止自我 Review**：作者不能用自己的測試或自我核准取代審查。
- **Review 不等於 Merge**：必要 Reviewer 檢查程式碼、介面、測試證據和風險，給出 `Approve` 或 `Request changes`。所有必要 Review 和修正都完成後，**只有 @Nues0913 決定是否合併**。還有必要 Review 沒完成、或有未解決的重大異議時，不得合併。
- PM 審查 teammate 的 PR 時，看架構、需求、安全性、Bug、維護性和實驗公平性；作者是 PM 時，由對應的 teammate 審查程式碼和技術證據。有爭議時，PM 在 Issue 或 PR 記錄最終決定。

### 5.2 提交與合併流程

1. 依第 4.1／4.2 節確認可以開工。從最新的 `main` 為每個 Issue 建一條分支，例如 `fix/4-odor-pathway`、`feat/ros-closed-loop`。**任何人都不直接推送 `main`。**
2. PR 由實際執行者的帳號建立，標題描述成果，內文用繁體中文並連結 Issue：只是部分處理時寫 `Refs #N`；完全解決時才寫 `Closes #N`（合併到 `main` 時 Issue 會自動關閉）。
3. 每個 PR 只放最小可審查的範圍。修 Bug 時不順便大幅清理無關文件、重構跨模組程式或修改研究參數。
4. 只加入該加的檔案（先看 `git status`，不要用 `git add -A`）。`runs/`、權重、`.venv`、本機路徑不進 Git。
5. PR 內文要寫：改了什麼、為什麼、影響的檔案、測試指令、實際結果、失敗與未測的項目、重度工作的證據、風險和回滾方式。
6. 未通過必要驗證就不合併。環境跑不了的項目標 `NOT RUN`，交給有算力的 teammate。如果 PM 決定帶著風險合併，要公開記錄理由、範圍和後續追蹤的 Issue。
7. 預設用 **squash merge**，避免分支中誤加的檔案或本機路徑進入 `main` 的歷史；合併後刪除自己建立的分支。疊在其他 PR 上的 PR，在 base 合併後改回 `main`。
8. 合併後由 DRI 在 Issue 附上 PR 與結果，並更新相關模組文件。Issue 依完成條件關閉，不是有 PR 就自動結案。

### PR 內文範本

```md
## Related Issue
Refs #<issue-number>

## Owner
DRI: @<github-user>
Required reviewers: @<依第 5.1 節填入所有非作者的 Reviewer>
Cross-module owners affected: @<github-user 或 none>
PM merge decision: @Nues0913

## Changes
- ...

## Evidence
- Commit SHA / config / seed / hardware: ...
- Commands and results: ...
- Artifacts / logs / checksums: ...
- NOT RUN（若有）及原因: ...

## Risks / PM decisions needed
- ...

## Checklist
- [ ] 沒有未經批准的介面或實驗規格變更
- [ ] 測試證據完整且如實呈現
- [ ] 所有必要的非作者 Reviewer 已完成審查，沒有未處理的重大異議
- [ ] PM 已確認是否合併（PM 自己的 PR 也適用）
- [ ] 文件與相關 Issue 已更新
```

## 6. 衝突處理

| 情境 | 處理方式 |
| --- | --- |
| @milong9487 發現 `flytrack_sim/env.py` 的 Bug | 在 Issue 記錄重現步驟和建議，交給 @cweiting731 實作；如果直接影響訓練，先暫停受影響的實驗，由 PM 決定優先級 |
| @cweiting731 發現 `flytrack_core/model/policy.py` 異常 | 開 Issue 並附 log，交給 @milong9487 診斷；不自行調整神經參數 |
| 兩人都想改 `docs/status.md` 或 README | 各自在自己的 Issue 或 PR 更新證據；全域狀態文件由 PM 統一維護，或由 PM 指派一人送文件 PR |
| 模型和 Gazebo 的結果不一致 | 開跨模組 Issue 並指定一位 DRI；兩人分別提供模型輸出和 ROS 訊號的對照，由 PM 裁決是否修改契約 |
| 想減少 seed 數、PPO 步數或放寬 gate | 開 `needs:pm-approval` Issue 並附量化依據；**批准前沿用原設計** |
| PR 審查意見衝突 | 以資料、需求和驗收標準討論；必要 Reviewer 提出的重大問題要先解決，PM 在 PR 記錄最終決定。禁止互相覆蓋修改，或以作者身分自我核准 |

## 7. 目前 Issue 的指派（本文件合併後由 PM 在 GitHub 執行）

| Issue | DRI | 協作方式 |
| --- | --- | --- |
| [#4 氣味無法影響策略](https://github.com/Nues0913/Drosophila_Tracking_System/issues/4) | @milong9487 | 模型診斷、ORN/DN、bias 和訓練策略本來就是其工作；涉及世界或感測時請 @cweiting731 協作 |
| [#9 雜訊強度與 BC 擴增](https://github.com/Nues0913/Drosophila_Tracking_System/issues/9) | @milong9487 | 實驗方案、訓練與評估由其主責；雜訊情境的實作由 @cweiting731 支援；是否調整正式情境由 PM 核准 |
| [#10 正式訓練時間方案](https://github.com/Nues0913/Drosophila_Tracking_System/issues/10) | @Nues0913（決策 Issue） | @milong9487 提交效能與方案比較；@cweiting731 回報 GPU 主機的容量與可用時段；PM 批准排程與公平性變更 |
| ROS／Gazebo 正式閉環與測試（新 Issue） | @cweiting731 | 2D／Gazebo 一致性、回合 reset、真值隔離、端到端延遲與長時間測試 |
| 訓練矩陣的各 `(variant, seed)`（拆成執行 Issue） | 各自負責被指派的批次 | @milong9487 最後彙整模型結果，PM 最終驗收 |

派工前先看 [docs/status.md](docs/status.md) 的未完成事項。歷史 PR 已合併，不代表相關的**實驗問題**已經解決。

## 8. 導入順序與 GitHub 現況

GitHub 現況（2026-10-18 查詢）：@MiLong9487 和 @cweiting731 都有 write 權限；repo 只有 GitHub 預設的 label；`main` 沒有自動強制的 Review 規則，第 5.1 節由 PM 人工把關。

1. 本文件經兩名 teammate Review 後合併，三個 Agent 開始新任務前先讀。
2. 確認兩名 teammate 可用的 GPU、VRAM、ROS/Gazebo 環境和可投入時間；必要時調整**算力分配**，不隨意改**模組主責**。
3. PM 建立第 4.3 節的 label；之後視需要加上 Issue／PR 模板、`CODEOWNERS` 和 `main` 的保護規則。個人帳號的 repo 可能受方案限制而無法啟用，那時仍以人工確認第 5.1 節的非作者審查。
4. 依第 7 節整理現有 Issue 的 DRI 與待決事項，並為 ROS 閉環和 GPU 矩陣開新的 Issue。
5. 先跑單一種子的 benchmark，確認兩名 teammate 都能保存一致的訓練產物，再分配全部的 GPU 批次。
6. 之後一律依照：Issue →（同領域小 Bug 自行認領；其他工作經 PM 批准並指派）→ 分支 → PR → teammate 依需要驗證 → 依作者與領域完成非作者 Review → PM 決定合併 → 更新 Issue。
