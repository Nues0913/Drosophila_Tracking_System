# v2 狀態

2026-11-23 M4 執行紀錄（UTC+8）。A3 第十五版已提前凍結，WP9 正式訓練進行中。各主機的後續回報見 [四台主機派工](https://github.com/Nues0913/Drosophila_Tracking_System/issues/143#issuecomment-7105260602)。

- 計劃書：`docs/v2/大實驗計劃書.md`（#130）
- 總追蹤：#127
- [M4 後的專案健檢與收件風險](health_2026-11-24.md)
- 截止：2026-11-26 10:00（UTC+8）
- 這份文件每個里程碑更新一次（計劃書 5.8）；日常的進度看 #127 的 checklist。

## 里程碑

| 里程碑 | 時間（UTC+8） | 狀態 |
| --- | --- | --- |
| M0 計劃確認、介面凍結、派工、v1 基準 | 11-19 12:00 | 完成：D1–D7 同意；計劃書 #130、介面 #145、協定草案 #146 合併；WP Issue #131–#144；三台的 v1 bench（#139、#147） |
| M1 各模組合併 | 11-20 22:00 | 完成（#127） |
| M2 端到端跑通、bench ≥ v1 | 11-21 22:00 | 完成：三台都跑完雙階段的試跑和評估（#141）；v2 PPO 每秒步數高於同主機的 v1（2060 1,732 對 229，#192 R11；4070 約 2,280 對 273，#141 7055435416） |
| M3 試跑結束、選定設定 | 11-24 00:00（11-23 延後半天，#138） | 完成，使用者選定 A3 第十五版，六變體試跑已收件（#141、#211） |
| M4 協定凍結、正式派工 | 11-24 04:00（原訂） | 11-23 22:09 合併 #211，#143 已發布開跑令 |
| M4b 正式訓練完成 | 11-25 10:00 | 進行中，30 條訓練鏈按 seed、主機與固定版本收件 |
| M5 正式評估與統計 | 11-25 16:00 | 未開始 |
| M6 Gazebo v2、報告初稿 | 11-25 22:00 | [報告草稿 #216](https://github.com/Nues0913/Drosophila_Tracking_System/pull/216) 已開，4070 已安排 G1 與延遲量測（#140） |
| M7 交付 | 11-26 10:00 | 未開始 |

## 協定

- 評估協定：`configs/v2/evaluation_protocol.yaml`（schema 4）。A3 第十五版的正式訓練設定由協定指向 `configs/v2/train/a3_v15.yaml`。
- M4 已凍結。試跑 seed 為 100、101，正式 seed 為 1–5。矩陣先收兩階段的 held-out；#144 依每個變體五個 seed 的配對結果選階段，再跑正式回合。[正式開跑令](https://github.com/Nues0913/Drosophila_Tracking_System/issues/143#issuecomment-7103593055)。
- #138 的五項探索標準保留為試跑紀錄。第十五版 seed 100 的 wind 遮蔽為 98/200，配對 V2 為 36/200；seed 101 為 23/200，且 wind normal curved 為 0.92。正式評估仍依協定的 G-A 至 G-E。

## 正式派工與收件

| 主機 | 正式 seed | 鏈數 | 核准執行版與任務 |
| --- | --- | ---: | --- |
| GPUtw RTX 3090 | 1、2 | 12 | `3982fbc`，依序跑兩個完整 seed；#215 的合併版是 `7137d66` |
| 4060 | 3 | 6 | `f8e8c21`，11-23 23:27 啟動 seed 3；已釋出 seed 2 並確認該機未曾開跑。[4060 回報](https://github.com/Nues0913/Drosophila_Tracking_System/issues/143#issuecomment-7106983613) |
| 2060 | 4 | 6 | `f8e8c21`，11-23 啟動 |
| 4070 | 5 | 6 | `f8e8c21`，先做 G1、延遲量測，再跑矩陣 |

每個 seed 的六變體在同一台主機完成並保留同機 v1 bench。主機上傳證據、權重和 SHA-256 清單後才算交件。G-A 至 G-E 等 #144 正式判定。[四台主機派工](https://github.com/Nues0913/Drosophila_Tracking_System/issues/143#issuecomment-7105260602)、[4070 的 G1 與延遲安排](https://github.com/Nues0913/Drosophila_Tracking_System/issues/140#issuecomment-7105752727)、[正式收件區](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/research-v2-formal)。
