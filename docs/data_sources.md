# 資料來源與實作口徑

- MaleCNS v1.0：[官方下載頁](https://male-cns.janelia.org/download/)，批次表格位於官方 Google Storage 的 `flyem-male-cns/v1.0/connectome-data/flat-connectome/`。資料使用 CC-BY 授權；正式報告應列出 FlyEM／Janelia、Cambridge、MRC LMB、Google Research 與相關論文。
- 表格欄位對照參考資料發布工具 [flyem-snapshot](https://github.com/janelia-flyem/flyem-snapshot)。程式只接受明列的欄位別名，未知 schema 報錯，不猜測欄位語意。
- 主模型採計劃書方案 A：早期視覺使用 flyvis 預訓練參數，中央連接體來自 MaleCNS。因此報告應描述為「flyvis 早期視覺＋MaleCNS 中央子圖」，不能把整個模型都稱為 MaleCNS 重建。
- 氣味的 DM1／VA2、LC 與 DN 型別已以實際 inventory 與路徑掃描確認存在並可達 DN（`artifacts/data_execution/candidate_scan.json`）；DNp09 作為前進讀出仍是候選，尚待文獻核對。
- 未取得感受野座標時，視覺前端到 LC 種子的映射使用可配置空間池化；這是工程假設，不能宣稱是真實神經元感受野。
- NT 若沒有數值信心欄位，程式使用發布者的 consensus／celltype 預測，並在 manifest 記錄數值信心可用數量；這不等於本專案已驗證所有預測信心 ≥0.5，實際資料中沒有數值信心欄位（manifest 的 `nt_numeric_confidence_available_neurons=0`），報告須說明這點。

## 本次資料下載狀態

2026-10-15 已下載三份官方表格至 `data/raw/`，共 **1,109,008,094 bytes（約 1.11 GB）**。

| 資料 | 檔名 | 大小（bytes） | 本地 SHA-256 |
| --- | --- | ---: | --- |
| 註解 | `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14,483,314 | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` |
| 神經傳導物質 | `body-neurotransmitters-male-cns-v1.0.feather` | 43,282,834 | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` |
| 連接權重 | `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1,051,241,946 | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` |

下載器已檢查預期長度與 Arrow IPC 頭尾框架，並保存本地 SHA-256。下載完成後另向官方 URL 發送 HEAD 請求，確認 Content-Length 與 ETag 相符。本地 SHA-256 並非官方發布的校驗值；這些雜湊也寫在 `data/subgraphs/main/manifest.json` 的 `source_sha256`，萃取時會核對。

- `data/raw/download_manifest.json`：下載狀態、來源 URL、實際位元組數、ETag 與本地 SHA-256（不進 git，重新下載時產生）。
- `data/raw/download_verification.json`：下載後核對結果與官方 Google Storage 物件 generation。

實際 Feather schema 已核對（`artifacts/data_execution/raw_schemas.json`），型別盤點與子圖萃取已完成，見 [status.md](status.md)。
