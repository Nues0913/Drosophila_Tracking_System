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
