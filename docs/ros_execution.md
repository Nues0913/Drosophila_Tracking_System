# ROS／Gazebo 實際執行紀錄

日期：2026-10-15。使用者已授權執行完整計畫；以下只記錄已觀察的狀態，持續更新，未完成項目不當作通過。

## 環境與秘密隔離

- 本機 Ubuntu 26.04.1，x86_64，Docker Engine 29.9.0，Compose v5.6.0；daemon 可用。
- 主機記憶體約 15 GiB，磁碟可用約 267 GiB。桌面與既有程式維持運行；ROS 首次驗證使用 CPU torch、軟體渲染，不占用其 GPU。
- 這是本機執行環境：沒有 `/etc/codex/network-policy.json`，HTTP(S)／ALL proxy 與 `CODEX_PROXY_CERT` 均未設定。沒有變更 DNS、代理、Docker daemon、驅動或系統權限。
- Docker 指令使用空白臨時 `DOCKER_CONFIG=/tmp/flytrack-docker-anonymous` 及 `--env-file /dev/null`，僅匿名存取公開映像，不讀 `~/.docker/config.json`、專案 `.env` 或 neuPrint token；原設定不變。

## 已執行

1. 預設 Compose 與 CPU override 的 `config --quiet` 均回傳 0。
2. 公開 ROS 基底已開始拉取：`osrf/ros:lyrical-desktop-full-resolute@sha256:956cc9eeb9f2dcae30c659ea1c2ad985dc8fc6f25a4a70d8f485106311b0c8c6`。
3. apt 已找到 Lyrical 的 `ros-gz`、`foxglove-bridge`、`launch-testing` 套件，並開始安裝；Python 3.14 的 torch 2.9.1 CPU wheel 已成功下載。

建置命令（此覆寫僅作用於該次命令）：

```bash
DOCKER_CONFIG=/tmp/flytrack-docker-anonymous \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
  docker compose --env-file /dev/null --profile sim build sim
```

首次建置 log：`/tmp/flytrack-ros-build.log`。目前仍須完成 `colcon build`、映像啟動與實際 world／感測／重置檢查。

## 驗證邊界

`scripts/validate_ros_world.py` 是 world／bridge 診斷工具，暫時提供 reset 回覆並發布低速診斷命令，以檢查雙眼影像、氣味、姿態、實際位移和兩個回合。它沒有學習策略；輸出不算追蹤效能、不算真值隔離測試。必須與正式策略在不同 DDS domain 執行。

真正策略 artifact 備妥後，仍須另外執行感測到控制、看門狗、模型 reset、真值隔離、rosbag 完整性及 Gazebo 正式矩陣。既有 sensor age 與 inference wall time 不是完整 sensor generation 到 command 的 wall-time 因果量測。

官方配對依據：[Gazebo ROS 安裝](https://gazebosim.org/docs/jetty/ros_installation/)、[ros_gz 版本表](https://github.com/gazebosim/ros_gz/tree/ros2)、[OSRF Docker 定義](https://github.com/osrf/docker_images)。Lyrical 使用 Jetty 的 ROS vendor 套件；未新增 Gazebo 自家 apt 來源。

## ROS topic 與重置（設計）

原始感測 `/sim/camera/{left,right}/image_raw`、`/sim/odor` 只送干擾節點；策略只讀 `/fly/camera/{left,right}/image_raw`、`/fly/odor`。`/world/agent/pose`、`/world/target/pose`、`/world/scenario` 留在 world 與 eval 邊界。

`/cmd_vel` 由策略發布；回合管理器在 inactive／reset 階段發布零速度，評估器不能發布控制。`/fly/neural_activity` 與額外 `/fly/inference_latency_ms` 為診斷，`/eval/metrics` 為每回合結果。

回合管理器先關閉感測轉送並呼叫 `/connectome_policy_node/reset`，持續零速停穩，再重置 agent/target pose，第二次清除策略緩衝，所有服務成功後才開新回合。氣味與軌跡按照同一事件種子與模擬起點重建。部署第一次控制前等待三路新感測；真值缺失會讓評估回合標記 invalid，不從少量好樣本宣告成功。

`scripts/run_gazebo_matrix.py` 提供後續批次編排，錄製彙總核對每回合 model_card SHA-256。這兩段程式碼尚未實際執行。


# 2026-10-19 本機首輪閉環驗證

## #22：單回合、5回合reset與延遲

這是部署鏈路的工程驗證，使用舊 VO smoke actor，不是策略品質或正式比較結果。
**單回合與5回合流程已實跑，5回合 invalid=0/5；完整因果延遲 p95 尚未達50ms，實體速度也沒有直接Twist量測，Issue維持開啟等待PM判讀。**

## 版本與環境

- 起點 main `83f57d1`，用獨立worktree與固定原始碼副本建置。實跑包含 #38／#39 的局部修正：PR #41（recorder的 `--topics` 與Compose的SIGINT）、PR #40（ControlTrace固定陣列轉Python int）；均已由PM審查合併。三個相關runtime檔的內容與合併後main `2009798`一致，雜湊對照附在交付包。
- Windows／WSL2 Ubuntu24.04，i7-13700HX；Docker Desktop4.43.2／Engine28.3.2，24 logical CPUs、約15.5GiB記憶體。容器Ubuntu26.04、ROS Lyrical、Gazebo10.5.0、Python3.14.4、torch2.9.1+cpu。
- CPU推論、軟體渲染；沒有使用GPU訓練、沒有改控制頻率／感測契約／模型。
- Ubuntu的Docker WSL整合未啟用；已啟動既有Desktop，改由Windows CLI＋固定tracked-only Windows Temp副本完成build／Compose／bind。沒有讀專案.env或既有Docker登入，使用獨立空auth config。無需為本次驗證另外啟用WSL整合。
- actor：#21已驗證的舊smoke，model card SHA-256=`7ccff60ad0620b4ec5275eba48f1d8a8aece59d6b73a8594d93fdd69360f5fdc`。PM在#22明確允許此用途；不把它當作#36正式G2產物。

## 實際執行與結果

1. 現有Dockerfile成功建置，colcon build完成。
2. `colcon test`實際exit5：三個Python套件未收集tests，不能報通過；另開#42追蹤。直接執行既有latency pytest為3passed；#39新回歸加入後ROS映像內4passed；真值隔離launch test為2passed且節點乾淨退出。
3. 以獨立reset診斷服務、沒有學習策略的world probe，完成2回合／4次reset，errors為空；clock遞增、agent有實際位移。雙眼約30Hz、odor約50Hz、pose50Hz，來源計時覆蓋100%。這不是策略回合。
4. 改為真實policy＋evaluator＋recorder，先1回合、再5回合。每回合上限60s，舊smoke在碰撞時正常提前結束；這不使回合invalid，也不代表策略品質通過。
5. 單回合有效；5回合全有效、真值採樣coverage均1.0。policy實際訂閱清單只有clock與fly路徑，沒有world/sim/eval真值訂閱。

| 回合（multi） | sim執行秒 | valid | truth coverage | 完整因果p95（episode metric，ms） |
| --- | ---: | --- | ---: | ---: |
| 0 | 25.089 | True | 1.000 | 58.157 |
| 1 | 22.749 | True | 1.000 | 59.287 |
| 2 | 22.597 | True | 1.000 | 53.028 |
| 3 | 28.098 | True | 1.000 | 51.259 |
| 4 | 18.748 | True | 1.000 | 52.580 |

全部回合使用固定normal、curved／puff與原感測契約；single seed1650000100，multi從1650000200起。所有策略結果均保留，沒有挑掉碰撞回合。

## Reset與資料隔離

- world診斷直接計數4次reset（兩回合各兩次）。實際5個policy回合均有inactive/resetting/active/completed事件，啟動路徑在SetEntityPose和兩次policy reset都回成功後才進active；這是狀態機與事件的佐證，**未另外錄製service RPC事件，不把它說成獨立抓到10筆RPC**。
- bag中每段completed到下一active之間，三路fly感測的新source stamps均為0；新回合推論選入前一回合sample的次數為0。
- 每回合第一個三路全有效command出現在active後414、52、47、51、48ms，均小於1s；最後一個input實際到達到該command發布的wall time約18.4～25.1ms。
- 重置後第一筆agent位置距要求位置最大約7.61µm。第一個有效command前，由pose差分推得的最大平面速度約0.000369m/s（0.369mm/s），其餘四回合為0。
- **尚未直接量Gazebo Twist／速度component；不能宣告實體速度被精確清零。** 上述微小位移／速度只能作為停止與重置觀測，不能代替速度清零的獨立證據或2D/Gazebo一致性校準。

## 延遲與頻率

完整因果延遲依既有SensorTiming／ControlTrace／CausalLatency契約：同一host的CLOCK_MONOTONIC，取該command實際使用的最舊有效sample，由source至完成cmd_vel publish。**不是模擬時鐘age或單獨inference時間。**

| 指標（multi5） | p50 ms | p95 ms | 範圍 |
| --- | ---: | ---: | --- |
| contributing sensor age | 22.000 | 37.450 | 模擬時鐘，control開始時；不含本次推論耗時 |
| 模型處理 wall | 17.024 | 24.763 | inference_latency topic，2352筆 |
| 完整sensor→command wall | 37.544 | 55.623 | 整個bag的2352筆有效inference trace，含回合交界尾端 |
| 完整sensor→command wall（active內） | 37.512 | 54.129 | 依command sim stamp落在active事件窗口，2335筆 |

因果clock/domain/ordering覆蓋率100%，invalid traces=0。active窗口與evaluator的收件時刻關窗不同，邊界延後到達的trace數會略不同；逐回合原始metrics及全bag／active重算都交付，未只挑較小值。兩種完整p95都超過50ms，**未達目標，先回報而未調頻率／契約**。各段分位數不可直接相加當作端到端分位數。

source stamp的正向相鄰間距中位數：雙眼30.303Hz、odor50Hz、agent/target pose50Hz、control traces20Hz。cmd_vel同時有policy與reset coordinator發布，bag中有相同clock stamp的零命令；不把合併訊息數宣稱為單一policy控制頻率。

## Rosbag與關閉

- single bag：metadata＋db3，ros2 bag info exit0，44.9MiB、107201messages、13個指定topic皆有資料。
- multi5 bag：metadata＋db3、226234messages，eval metrics5筆；兩個bag SQLite quick_check都為ok，逐檔SHA在交付包。
- `compose stop --timeout 10 policy eval`（SIGINT）耗時0.89s，低於預設10s；未用SIGKILL，索引已寫出。大規模bag仍需另量，不以此保證所有正式錄製關閉時間。
- 所有本次容器已清理，Docker Desktop本身保留；沒有停止其他project。

## 失敗與未跑項目

- 初始Windows／WSL mount失敗已以隔離Windows副本排除，不代表Gazebo失敗。
- 原始recorder exit2、只修CLI後未flush metadata、evaluator numpy.uint64崩潰，皆保留原始log；各自由#38/#41、#39/#40修正，不把這些失敗嘗試算通過。
- colcon test集合未通過（#42），與手動執行的4+2 tests分開報告。
- NOT RUN：直接實體速度讀值／精確清零驗證、#23分布／碰撞一致性、正式Gazebo矩陣、正式訓練或追蹤品質驗收。
- 尚未達標：sensor→command p95<50ms。請PM決定是否需要額外速度觀測、後續效能診斷與停止允許差；本次沒有私改物理／門檻來湊結果。

## 重現與證據

固定main2009798包含兩個局部修正；使用現有CPU Compose override與同一smoke artifact，指定獨立project／ROS_DOMAIN／seed，eval改用evaluation.launch.py record_bag=true。只跑sim/policy/eval，不跑viz或完整matrix。專用env與override、所有命令、bag離線分析程式、raw metrics／traces、build／test／runtime log在交付包。

```sh
# 從指定worktree執行；已有WSL整合可用本機docker，否則使用Windows native副本
# 所有.env內容均為本次明確建立的診斷參數，不讀專案秘密檔
 docker compose --project-name flytrack-issue22 --env-file issue22.env \
  -f compose.yaml -f compose.cpu.yaml -f compose.issue22.yaml --profile sim up -d sim policy eval
# 等待指定5回合episode資料後，先關policy/eval，再關sim
 docker compose --project-name flytrack-issue22 --env-file issue22.env \
  -f compose.yaml -f compose.cpu.yaml -f compose.issue22.yaml --profile sim stop --timeout 10 policy eval
```

完整artifact與同名SHA-256將以 `issue22_cweiting731_2009798.tar.gz` 交回 `explore-4`；此為部署驗證產物，不是正式結果。Issue #22維持開啟等待PM覆核；延遲數據同時回覆#15的要求。

## colcon 的 Python 測試收集（#42）

ROS Python 套件以 `extras_require={"test": ["pytest"]}` 宣告測試工具，讓 colcon 選用 pytest。
Lyrical 映像中的新版 setuptools 已不接受 `tests_require`；只加這個舊參數仍會落回 unittest，
造成既有 pytest 函式未被收集，且 Python 3.14 在沒有 unittest 時回傳 5。

各套件的 `setup.cfg` 固定本地 testpaths，避免 pytest 向上找到專案根目錄的設定後誤跑核心測試。
`flytrack_eval` 收集 `test/` 的 4 項延遲測試；`flytrack_policy`、`flytrack_world` 目前沒有套件內測試，
colcon 的 pytest adapter 接受空集合，但這不代表兩套件有測試覆蓋。
需要 artifact 的真值隔離檢查仍以 `launch_test` 獨立執行，不混進無 artifact 的預設測試集合。

在實際 Lyrical／Python 3.14 映像內重新 build 後，普通 `colcon test --return-code-on-test-failure`
回傳 0，`colcon test-result --verbose` 為 **4 tests、0 errors、0 failures、0 skipped**。
另只在臨時容器副本將預期延遲 5 ms 改為 6 ms，確認相同指令回傳 1，結果列出 3 個失敗；
此負向驗證沒有修改提交的測試，證明失敗會傳回，而非忽略所有非零狀態。
