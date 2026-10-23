# C12 雙 seed driver：逐項核對與驗收

2026-10-23 續接中斷的工作。核對現有實作後，未發現需要補寫的 driver 邏輯；
這次新增 8 項 C11 共用介面的相容性測試，並重新完成以下驗收。
本文件的 `<runs>` 表示工作樹中的實驗輸出目錄。

## 計劃逐項對照

| 計劃要求 | 實作與核對結果 |
| --- | --- |
| 從 C11 s0 產生兩份 C12 YAML | `prepare_c12.py` 讀取 `<runs>/c11/ppo_c11.yaml`，輸出 `ppo_c12_s0.yaml`、`ppo_c12_s1.yaml`；內部沿用 C11 的 actor/config 名稱以重用既有模組。 |
| 差異白名單，多或少都拒絕 | s0 必須恰有 `output`、`train.curriculum`；s1 另外必須有 `seed: 1`。測試涵蓋額外鍵、刪除鍵及漏掉必要差異。 |
| curriculum 精確內容 | 僅保留 C11 第 3–6 段，逐鍵核對；`until` 寫死為 `[0.385, 0.615, 0.846, 1.0]`。500k 步的邊界是 192500、307500、423000、500000。 |
| 其他訓練設定不變 | 白名單以外不得不同，另外檢查 `total_steps=500000`、`readout_bias_learning_rate=0`、`critic_warmup_updates=10`。smoke 才套用 C11 的縮小設定。 |
| 相同 BC actor、preflight | 兩個 seed 的 `init_artifact` 都必須指向 C11 的 BC；完整 actor 檔案 SHA 逐一比對 C10 manifest。啟動前檢查其他訓練程序，包含 C12 wrapper。 |
| main 的完整 `python/` 與固定 commit | 讀取 `<runs>/c12/main_python.sha256`、`main_python.commit`，核對完整來源檔清單及 `84830e7d01efeb224aedeb3a9db0f08b981a8c8e`。正式模式不符即拒絕；smoke 只警告。這次未提供參考檔，因此正式比對為 NOT RUN。 |
| 共用 protocol100 區塊 | s0、s1、BC 正式評估固定共用 1600600000–1600600099；快照固定共用 1600700000–1600700019。評估 YAML 從 C11 複製，只更換 seed；smoke 另縮短 episode。 |
| BC 只評估一次 | GPU 類階段只有一個 `bc_eval`；CPU 的 `bc_snap_eval`、`bc_snap_dn` 各一次。兩個 seed 的快照報告共用同一份 BC 第 0 點結果及 DN 檔案。 |
| 階段順序、快照時機及 CPU 資源 | 依序 s0 訓練、s1 訓練，再跑兩個 seed 與 BC 的評估。兩份訓練收據都完成後，才啟動背景快照鏈；與評估並行，nice 10，每個 seed 最多兩顆 CPU，workers=0。訓練期間只跑保留 checkpoint 的 sidecar。 |
| 重用 C11 的每 50k 步 hard link | 直接使用 C11 `Sidecar`、`build_all`；保留實際步數、邊界與漏失紀錄。smoke 改成每 12 步，這次兩個 seed 均保留 12、24、36 步；末端快照與 checkpoint 的 inode 相同。 |
| 續跑、帳本、STOP、監控 | 重用 C11 的帳本鎖、程序群組清理、CPU 記憶體限制與記錄工具；完成收據同時檢查輸出 SHA 與語意，訓練 checkpoint 可 resume；支援 STOP 與正式模式的 `MONITOR_SH`。 |
| GPU 總預算 8100 秒與 timeout | 五個 GPU 類階段的所有嘗試均記帳，含失敗及中斷；CPU 不計入。timeout 保留清理時間，預算不足標記剩餘階段及受影響 acceptance 為 NOT RUN。測試以替身程序驗證，沒有啟動 GPU。 |
| smoke 隔離六個正式區塊 | 固定使用 1600000000 與 1600001000，一個 episode；交疊任何 C11、C11-s1、C12 正式區塊都拒絕。執行時 CUDA 不可見，兩個 seed 的訓練及全部評估均確認為 CPU。 |
| 四條件及整體判定 | 直接重用 `accept100.compare`，兩個 seed 分別與共用 BC 配對；固定輸出「C12 一致通過／不一致／都不通過」，規則寫入 manifest 與 acceptance。 |
| 早期快照事先解讀 | 依 50k、100k 邊界取兩個 seed 共四筆；全部滿足 abs(ω) ≤ 0.3 且追蹤 ≥ 0.3、或任一滿足 abs(ω) ≥ 0.6 且追蹤 ≤ 0.1，分別套用指定標籤，否則不下結論；缺筆不下結論，且不影響 acceptance。測試涵蓋含等號邊界。 |
| 只讀來源的中文摘要 | `summarize_c12.py` 僅讀既有資料，輸出 `summary.json`、`summary.md`；包含四項條件及數值、固定判定標籤、各 seed 六項快照指標、新 curriculum 範圍，以及 C11 s0/s1 的同實際步數資料與各自 curriculum 階段。只做精確步數配對，不補插值或附加解讀。 |
| 保護已交付結果及來源檔 | C11 兩個目錄只讀，先記錄完整檔案清單與 SHA，再於 smoke 清理後比對；246 檔完全相同。`python/` 63 個來源檔亦前後相同。未修改 C10、共用頂層腳本或診斷程式。 |

## `c11/jobs.py` 的修改與相容性

| 介面 | 新增參數 | 預設行為 | C12 用途 |
| --- | --- | --- | --- |
| `evaluate(out, name, *, seed_guard=None)` | `seed_guard` | 仍呼叫原本的 C11 seed guard，從 manifest 判定 s0/s1 profile。 | 傳入 C12 的 guard，限制自己的區塊並排除全部六個正式區塊的 smoke 交疊。 |
| `snap_eval(out, *, items=None, seed_guard=None)` | `items`、`seed_guard` | `items=None` 仍取原本的 `snapshot_actors(out)`；guard 同樣按原 manifest 選 C11 profile。 | 分開評估一次 BC 第 0 點及各 seed 的訓練快照，再合併共用 BC 列。 |

原來的 C11 呼叫都沒有傳入這些參數，因此 seed、actor 順序、評估計算及輸出格式沿用原流程，
沒有新增亂數抽樣或改動訓練。僅在記憶體中撤回這些參數及對應分支後，
`jobs.py` 的 SHA-256 完全吻合已交付 C11-s1 manifest 的來源 SHA。
新增的 8 項測試檢查 s0/s1 的預設 guard（包括與環境變數相反的 manifest profile）、
C12 的明確 guard，以及預設／明確 actor 列表選擇；測試在 guard 處中止或使用空列表，
沒有執行正式 episode。

## 驗收結果

- 指定 pytest：`361 passed in 5.22s`。C12 140 項；C11 117 項加共用測試 104 項，原有 221 項全部通過。
- ruff：`All checks passed!`；format：`16 files already formatted`。
- 清掉舊 smoke 後，第一次執行 exit 0，總耗時 46.82 秒，17 階段全部完成。
- 第二次執行 exit 0，總耗時 4.13 秒；17 階段均通過收據及語意驗證後略過，帳本無新增嘗試。
- 建立 STOP 後執行 exit 0，總耗時 0.01 秒，輸出 `STOP present; exiting cleanly`。
- 續跑及 STOP 後，97 個輸出、收據與帳本檔案的 SHA 均未改變。
- 已刪除 `<runs>/c12_smoke/`，並確認沒有殘留 C12 子程序；輸出文字未含本機絕對路徑。

以下為第一次 smoke 的帳本 wall time，單位為秒；背景階段有重疊，不能直接相加當作總耗時。
GPU 類階段這次也都在 CPU 執行，實際 GPU 使用為 0 秒。

| 階段 | s0 | s1 | 共用 BC／整體 |
| --- | ---: | ---: | ---: |
| train | 9.396701 | 8.344572 | — |
| eval | 2.707098 | 2.807389 | 2.908567 |
| dn | 4.312147 | 3.711167 | — |
| snap_build | 3.208063 | 3.208757 | — |
| snap_eval | 3.810425 | 3.809826 | 2.307160 |
| snap_dn | 10.627121 | 11.233785 | 4.012287 |
| acceptance | — | — | 0.101342 |
| summary | — | — | 0.101286 |

檔案集合以相對檔名及各檔 SHA 的排序 JSON 計算彙總 SHA；前後值相同：

| 目錄 | 檔案數 | 前後一致的彙總 SHA-256 |
| --- | ---: | --- |
| `<runs>/c11/` | 123 | `f91f61cc426251e41f2ff5f9bc03081cfcb7683c43377ac6e85ad666e2c29ea7` |
| `<runs>/c11_s1/` | 123 | `46dde71fd32ce86812d37a99eb79ebf43efa99607550c79ea9b8da2c3c6efe56` |

NOT RUN：正式 500k 訓練、正式 protocol100／快照評估、GPU 與正式監控、
實際來源對固定 main 的參考檔比對。參考檔尚未提供；smoke 有依規則警告，
正式模式拒絕缺少參考檔的行為已由測試驗證。

實作沒有偏離計劃。執行程序的偏離：開頭曾用一次唯讀 `git status`，
且初始唯讀命令未全數加上 stdin 重導；未改變 git 狀態，後續命令均關閉 stdin。
