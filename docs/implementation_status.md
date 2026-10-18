# 實作範圍與尚待驗證的限制

2026-10-15 狀態，之後的修改沒有逐項更新到本表（例如 2026-10-17 的整合修正、#5–#7 的情境來源、確定性評估與嗅覺通路選項）。**目前狀態以 [status.md](status.md) 為準**；本表只保留模組位置與當時的工程近似說明。

| 計劃項目 | 程式碼位置 | 狀態 |
| --- | --- | --- |
| 官方資料取得／續傳／來源雜湊 | `connectome_extract/download.py` | 已執行：三份表格下載並核對；續傳行為未測 |
| 型別盤點、雙向路徑萃取與確定性產物 | `connectome_extract/`、`scripts/verify_subgraph.py` | 已執行：子圖 19 項獨立檢查通過，兩次重跑位元相同 |
| 凍結 flyvis 原生前端 | `model/frontend.py`、`scripts/export_flyvis.py` | 已執行：官方 checkpoint 轉換，parity 最大誤差 1.91e-6 |
| 六變體、Dale 號誌、共享參數與預算檢查 | `model/`、`scripts/prepare_experiments.py` | 已測：真實圖預算在 ±5% 內；初始化譜半徑修正 |
| 六角取樣、ORN、干擾 | `sensing/` | 單元測試通過；2D／Gazebo 輸入分布未量測 |
| 氣味 puff／高斯羽流與軌跡 | `world/` | 單元測試通過；參數需校準 |
| 2D 模擬與教師 | `flytrack_sim/` | 可執行；CPU 約 25 步／秒（PPO 含更新）；教師校準情境缺乏變異 |
| 預訓練、BC／DAgger、PPO、ES | `flytrack_train/` | 縮小版全鏈跑通；正式訓練未開始，無正式 checkpoint |
| 實驗矩陣與種子 bootstrap | `flytrack_train/evaluation.py` | 已寫並有測試；無實驗結果 |
| lesion／匯流來源電流分析 | `scripts/lesion_analysis.py`、`model/` | 已寫；屬事後分析，非因果證明 |
| ROS 訊息、world／policy／eval 節點 | `ros2_ws/src/` | ROS 映像建置成功；未 colcon build 或跨節點測試 |
| Gazebo 差速機器人、bridge 與回合重置 | `flytrack_bringup/`、`flytrack_world/` | 未驗證物理、服務或相機 |
| Docker／Compose、Colab 操作 | `docker/`、`compose*.yaml`、`notebooks/` | 兩個映像本機建置成功；Colab notebook 未實際執行 |

## 必須揭露的工程近似

1. 方案 A 的早期視覺來自 flyvis 的資料與學習參數，只有中央子圖使用 MaleCNS。報告應區分這兩個来源。
2. 未有真實 RF 坐標時，LC 输入映射採型別／側別的空間池化；ORN Hill 閾值也是模擬設定，不是指定化學氣味的實測受體曲線。
3. V/O 為同圖的輸入消融。VO-late 共享型別參數但隔離活性；RS 保留原邊參數槽的共享標籤。這些精確比較口徑寫入 model_card。
4. 2D 使用 2.5D 光線投射，Gymnasium 向量環境為 CPU SyncVectorEnv，沒有聲稱已實作 GPU 張量化世界或達到訓練吞吐量目標。
5. 2D 有大小／亮度的領域隨機化。Gazebo 使用設定中的固定幾何和材質，並非已證明感測分布一致；仍需量測與校準。
6. ROS 碰撞目前用場地、半徑與障礙幾何判斷，未接 Gazebo contact sensor 作獨立真值核對。reset 的 pose 服务、零速度與隱狀態清理仍需跨節點驗證。
7. ROS sensor timestamp 使用模擬時鐘，推論處理耗時使用 wall time。完整 sensor-to-command 因果延遲尚未完成量測契約，不能宣稱 p95 <50 ms。
8. 保存研究 checkpoint 不等於完整可續訓：目前跨階段可用 actor 暖啟動，未恢復環境、rollout、RNG 与 optimizer 的完整訓練狀態。

## ROS topic 與重置

原始感測 `/sim/camera/{left,right}/image_raw`、`/sim/odor` 只送干擾節點；策略只讀 `/fly/camera/{left,right}/image_raw`、`/fly/odor`。`/world/agent/pose`、`/world/target/pose`、`/world/scenario` 留在 world 與 eval 邊界。

`/cmd_vel` 由策略發布；回合管理器在 inactive／reset 階段發布零速度，評估器不能發布控制。`/fly/neural_activity` 與額外 `/fly/inference_latency_ms` 為診斷，`/eval/metrics` 為每回合結果。

回合管理器先關閉感測轉送並呼叫 `/connectome_policy_node/reset`，持續零速停穩，再重置 agent/target pose，第二次清除策略緩衝，所有服務成功後才開新回合。氣味與軌跡按照同一事件種子與模擬起點重建。部署第一次控制前等待三路新感測；真值缺失會讓評估回合標記 invalid，不從少量好樣本宣告成功。

`scripts/run_gazebo_matrix.py` 提供後續批次編排，錄製彙總核對每回合 model_card SHA-256。這兩段程式碼尚未實際執行。

## 後續入口

詳細操作見 [README.md](../README.md)，執行順序、關卡和待確認項目見 [後續計畫.md](../後續計畫.md)。各子模組有 `MODEL_NOTES.md`、`SIM_NOTES.md`、`TRAIN_NOTES.md` 補充介面和比較口徑。
