# formal-v1 F3 curved 的 2D 展示素材

這套工具只呈現依固定規則選出的示例，不提供新的整體效能主張，也不替代 F3
統計。`demo_2d.py` 的 select、verify、plot 只讀取既有資料，CPU 繪圖使用
matplotlib Agg，不開視窗。replay 子命令呼叫 `episode_trace.py`，由後者載入
actor 並沿用既有路徑重播。

## 正式設定與讀出範圍

實際變體設定檔是 `configs/model/variants.yaml`（沒有 `configs/variants.yaml`）。
`configs/train/bc.yaml` 指定 structured readout，基本轉向配對是 DNa01、DNa02，
前進 pool 是 DNp09，見 `configs/extraction/olfactory_v2_nokc.yaml`。
`extra_turn_rows.variants: [O, VO, VO-late]` 才加入 DNb05：VO-late 有此讀出列，V 沒有。
設定檔說明 V 的 DNb05 缺少 odor 輸入而保持沉默，因此刻意排除。

`--record-dn` 記錄讀出使用的所有 type，以及 DNp09、DNa01、DNa02、DNb05
四種必要 type 的聯集。正式 V 的 DNb05 雖不參與讀出，仍存在於讀出收到的完整
rate tensor `[B, N]` 中，因此照樣從這份 tensor 記錄左右與全部 node 的平均值，
並標示 `in_readout: false`。圖例加上 `(not in readout)`，使用較細的灰線；
圖中只是 DNb05 的 rate，不影響動作，也不代表它對動作有貢獻。VO-late 的 DNb05
有讀出列，標示為 true；其他三種必要 type 在兩個 actor 都參與讀出。

## 欄位與選取規則

`selection/manifest.json` 的 `cells` 按 variant、scenario 排列，每格包含：

- `variant/scenario/seed`、唯一的 `model_card_sha256` 和四類 `counts`。
- `success` 與 `failure`：選取的回合物件，缺少候選時是 JSON `null`，對應
  `success_reason/failure_reason` 是 `"none"`，不補選。空格的 card SHA 也是 null。
- 選中物件包含 `episode`、`environment_seed`、`role`、`failure_class`、原 F3
  的 `success/collision_rate/within_distance_fraction/mean_distance_m` 與 card SHA。
  成功物件的 `failure_class` 是 null；失敗是 collision、distance 或 bearing。
- `selection_rule` 記錄規則原文；`input.path` 相對於 manifest 目錄，`input.sha256`
  是 F3 輸入檔的 SHA-256。CSV 包含所有格子的兩種角色，未選中回合欄位留空。

成功優先；其餘依序判斷 `collision_rate > 0`、`within_distance_fraction < 0.8`、
其他，得到碰撞、距離、方位失敗。距離邊界 0.8 屬方位。失敗分類重用 A1
`failure_category`；成功且碰撞同時成立的矛盾資料，依本展示規格歸成功
（A1 原函式是碰撞優先）。每格成功取最小回合編號；失敗取最多的失敗類別中
最小回合編號，次數平手時依 **碰撞 > 距離 > 方位**。不依重播結果重新選取。

`episode_trace.py --record-dn` 的每個 decision record 多一個 `dn`：

- `branches` 保留讀出輸入順序，單支路 `name="central"`；雙支路依序是
  `"visual"`、`"odor"`。不是 policy 回傳的支路平均 activity。
- `branches[].types[TYPE].left/right/all` 是讀出收到的 rate，type 名稱取 graph
  `type_names`。範圍是 `difference/forward_pool/extra_left/extra_right` 實際涵蓋的
  所有 type，聯集圖上四種必要 type。左右使用該 type 中 side=-1/+1 的節點平均，
  all 是該 type 所有節點平均（含 side=0），不是左右平均再取平均。
- `branches[].types[TYPE].in_readout` 是布林值：該 type 任一 node 在讀出的
  `difference`、`forward_pool`、`extra_left` 或 `extra_right` buffer 中有非零值
  才為 true。判斷不依賴變體名稱或學得的權重；false 的 rate 仍取自同一份輸入。
  必要 type 在 graph 中不存在或缺少任一側時才報錯，錯誤會列出 graph 的所有
  type；其他讀出 type 若缺少某側，該側記 null。
- `forward_latent` 是 sigmoid 前值（含 forward bias）；`omega_latent` 是 tanh
  前值（含 turn bias 與正規化 extra turn 項）。因此 raw action 可由
  `v_max*sigmoid(forward_latent)`、`omega_max*tanh(omega_latent)` 還原。
- 暫存每步清除，回合結束或例外時移除 hook。GRU、linear readout、非 none
  lesion，以及 graph 缺少任一必要 DN type 或該 type 的左右 node 都直接報錯。
  必要 type 沒有用於讀出可以正常記錄。每個 DN 子圖標題註明支路名稱。

`episodes.json` 的 `world` 記錄 unwrapped env 的 `walls=[xmin,ymin,xmax,ymax]`、
`obstacles=[{x,y,radius}, ...]`、`agent_radius_m`、`target_radius_m`，單位公尺。
碰撞類型由同一列原有的 `collision_types` 取得，位置是最後的 `post_agent_pose`。
`analysis_notes.json` 新增 `record_dn: true`、含 `in_readout` 定義的
`dn_definition`，以及 `required_dn_types_not_in_readout` 清單（V 是 `["DNb05"]`，
VO-late 是 `[]`）。關閉旗標不建立
hook、不新增欄位，也不改既有動作計算、序列化與 gzip 寫入方式。

`time_s/agent_pose/target_pose/teacher_action/target_distance_m/target_bearing_rad/
target_visible/occluded` 對齊決策前的 info；`applied_action/collision/post_*` 來自
執行後。圖上的 policy v、ω 使用 `applied_action`；原始 `action` 仍保留在 trace。
方位誤差保留正負、轉成度，繪圖固定在 [−180, 180]，每 90° 一個刻度；相鄰值
差超過 180° 時插入 NaN 斷線，不改 trace 資料。DN 子圖以自己的標題標示
`DN rates — visual branch`、`DN rates — odor branch` 或 `DN rates — central`。
右側子圖共用時間軸，僅最下方顯示時間刻度與 `decision time (s)`。
距離虛線固定為 2.0 m。橘色軌跡與所有時間子圖底色
依決策時 `occluded` 標示，區間從 `time_s` 到 `post_time_s`。

verify 對環境 seed、success、是否碰撞、model card SHA 逐一嚴格比對，缺回合、
重複回合或任何不一致都失敗。notes 既有格式沒有獨立 card SHA 欄位，實際使用
`provenance.inputs_sha256` 中檔名為 `model_card.json` 的唯一雜湊，不對 notes
內的 card 重新序列化。若未來 notes 有 `model_card_sha256` 則優先使用。
`selection/verify.json` 記錄每個回合比對結果，以及距離比例與平均距離的
**重播減 F3** 差值，這兩個差值不作通過門檻。plot 也會先做同樣驗證。

`figures/index.json` 中每張 PNG/GIF 都有回合資料及來源 trace SHA-256，總覽包含
全部回合與來源 trace 清單；路徑相對於 figures 目錄。總覽的空格只畫淺灰框，
中央以預設字型顯示 `none (no episode matches the rule)`，保留變體、情境與角色標題。

關於逐位元比對：既有 provenance 會雜湊 `episode_trace.py` 本身，所以修改此
腳本後 notes 的程式 SHA 必然改變；不同執行時間的 gzip header 也原本就可能
不同。工具保留真實 provenance 與既有 gzip 行為，不偽造 main 的雜湊。
固定 JSONL 位元與動作一致性由合成測試驗證；真實 actor 跨版本比較由沙盒外執行，
比對解壓後的 JSONL、`episodes.json` 與 notes 中 provenance 以外的欄位，
gzip header 時間戳與 provenance 的檔案 SHA 不在比較範圍。

## 完整流程

從 repository root 以 Bash 執行；先設定路徑。F3 是 curved 正式資料；actor 路徑必須
對應這份 F3 使用的 formal-v1 seed 0 artifact，不能換成重訓或 posthoc actor。

```bash
export PYTHONPATH=python
export MPLCONFIGDIR=/tmp/flytrack97-matplotlib
DEMO_PY=.venv/bin/python
DEMO_OUT=runs/diagnostics/demo_2d
DEMO_F3=runs/evaluation/formal/episodes.json
DEMO_V=runs/formal-v1/V/seed_0/actor
DEMO_VO=runs/formal-v1/VO-late/seed_0/actor

"$DEMO_PY" scripts/diagnostics/demo_2d.py select \
  --f3 "$DEMO_F3" --variants V VO-late --seed 0 \
  --scenarios normal occlusion conflict -o "$DEMO_OUT" < /dev/null

"$DEMO_PY" scripts/diagnostics/demo_2d.py replay \
  --manifest "$DEMO_OUT/selection/manifest.json" \
  --artifact "V=$DEMO_V" --artifact "VO-late=$DEMO_VO" \
  --config configs/train/evaluate.yaml -o "$DEMO_OUT" --jobs 1 < /dev/null

# 只收集現有的重播目錄，排除日誌與續跑時保留的不完整目錄。
mapfile -t DEMO_REPLAYS < <(find "$DEMO_OUT/replay" -mindepth 1 -maxdepth 1 \
  -type d ! -name '*.partial_*' < /dev/null)

"$DEMO_PY" scripts/diagnostics/demo_2d.py verify \
  --manifest "$DEMO_OUT/selection/manifest.json" \
  --replay "${DEMO_REPLAYS[@]}" < /dev/null

"$DEMO_PY" scripts/diagnostics/demo_2d.py plot \
  --manifest "$DEMO_OUT/selection/manifest.json" \
  --replay "${DEMO_REPLAYS[@]}" -o "$DEMO_OUT" --dpi 150 --animate < /dev/null
```

請將 `DEMO_F3`、`DEMO_V`、`DEMO_VO` 換成實際資料位置。replay 對每個
（變體，情境）只呼叫一次 `episode_trace.py --record-dn --lesion none --device cpu`，
將選中編號排序、去重後以逗號分隔傳給 `--episodes`，空格略過。manifest 內每個
變體都必須提供 `--artifact VARIANT=目錄`，否則在啟動任何重播前報錯。
`--jobs` 是同時執行的格子數，預設 1，必須大於零；子程序使用目前的 Python，
stdin 接 DEVNULL，失敗會傳回錯誤。stdout 與 stderr 都寫入
`replay/{variant}_{scenario}.log`，結果放在同名目錄。

重跑同一條 replay 命令即可續跑：已有 `episodes.json` 且回合集合等於選中集合的
格子直接略過；不完整或無法讀取的結果目錄先改名成
`{variant}_{scenario}.partial_<UTC時間戳>` 保留，再重新重播，日誌重新寫入。
所有選中的回合仍必須經過 verify；V 的 DNb05 會記錄為未參與讀出的 rate。

PNG 命名 `figures/{variant}_{scenario}_{role}_ep{episode}.png`，總覽是
`figures/overview_trajectories.png`。`--animate` 只對 occlusion 回合輸出
`animations/*.gif`，使用 PillowWriter、不需要 ffmpeg。每 0.1 s 約取一張，
保留首尾且最多 300 張，以 10 fps 播放；超過 30 s 的長回合會加速，游標及
時間標籤仍標示實際模擬時間。未帶 `--animate` 時只生成 PNG。
