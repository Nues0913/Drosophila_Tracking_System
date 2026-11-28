# FlyTrack

用雄果蠅中樞神經系統的連接體當控制器，做目標追蹤。

[![CI](https://github.com/Nues0913/Drosophila_Tracking_System/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Nues0913/Drosophila_Tracking_System/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-formal--v2-2ea44f.svg)](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/formal-v2)

<p align="center">
  <a href="https://youtu.be/Dj_BiW-q4VQ">
    <img src="https://img.youtube.com/vi/Dj_BiW-q4VQ/maxresdefault.jpg" width="720" alt="FlyTrack 專案影片（YouTube）">
  </a>
  <br>
  <b><a href="https://youtu.be/Dj_BiW-q4VQ">▶ 在 YouTube 觀看專案影片</a></b>（4 分鐘）
</p>

<p align="center">
  <a href="docs/v2/report.md">報告</a> ·
  <a href="https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/formal-v2">Release</a> ·
  <a href="https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/research-v2-formal">預訓練權重</a> ·
  <a href="README.md">English</a>
</p>

FlyTrack 是一個課程研究專案。控制器是一個循環發放率網路，接線取自 MaleCNS v1.0 連接體中 8,792 顆神經元的子圖。它用複眼看、用左右觸角聞氣味，也感覺得到風向。輸出是前進速度和轉向速度，從左右兩側各五種下行神經元讀出。我們在 GPU 批次化的 2D 世界中訓練它，先用 DAgger，再用循環 PPO，訓練好的策略也拿到 ROS 2／Gazebo 裡跑。

formal-v2 要測的是：目標被遮住、看不見的時候，氣味和風能不能幫策略繼續跟住目標。策略確實會對這兩種線索反應，但遮蔽時的成功率只比純視覺策略高 2.2 個百分點，沒有達到事先登記的 +10 個百分點。

## 結果

六個變體各用五個 seed 訓練：

- VO2：視覺、氣味和風
- V2：只有視覺
- O2：只有氣味和風
- RW2：保留度數的重接線圖
- ER2：隨機圖
- GRU2：參數量相當的 GRU

所有閘門都在正式訓練開始前寫進[評估協定](configs/v2/evaluation_protocol.yaml)。

| 閘門 | 比較 | 結果 | |
| --- | --- | --- | --- |
| G-A | VO2 在曲線目標、正常情境的成功率（≥ 80%） | puff 世界 100.0%，風世界 98.8% | 通過 |
| G-B | 風世界遮蔽，VO2 − V2（≥ +10 個百分點） | +2.2（19.9% 對 17.7%），Holm p = 0.555 | 未通過 |
| G-C | VO2 不劣於 RW2、ER2、GRU2（界值 −5 個百分點） | ER2 +16.0，RW2 +9.7（p = 0.072），GRU2 −14.2 | 未通過 |
| G-D | VO2 − v1 連接體策略（VO-late） | +51.8（100.0% 對 48.2%），Holm p = 0.001 | 通過 |
| G-E | 同一張 GPU 上，和 v1 訓練器比訓練吞吐量 | 6.06–11.13 倍 | 通過 |

<p align="center">
  <img src="docs/assets/vo_vs_v_occlusion.gif" width="760" alt="同一個遮蔽回合，左邊 VO2，右邊 V2">
  <br>
  <sub>風世界的一個回合，目標在 8 秒到 38 秒被遮住。左邊是 VO2，右邊是 V2。</sub>
</p>

遮蔽期間把左右氣味對調，VO2 在這段時間的追蹤比例從 57.0% 掉到 14.2%；把風向轉 90°，掉到 33.2%。兩者經 Holm 校正後都顯著，可見策略確實用到了這些輸入。遮蔽期間 VO2 跟住目標的時間也比 V2 多（57.0% 對 40.6%）。不過它撞上目標的次數也比較多，有碰撞的回合 VO2 是 41.7%、V2 是 15.0%，所以整體成功率幾乎沒變。

我們在協定之外另做了一次事後檢查：同一批遮蔽回合重跑，碰到目標不再結束回合（撞牆仍會）。這時 VO2 的成功率是 29.9%，V2 是 18.9%，差 +11.0 個百分點（95% t 區間 −0.1 至 +22.1，未校正的單尾 p = 0.025），差距大多來自漫遊目標（+19.4 個百分點）。紀錄和腳本放在 [research-v2-collision-check](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/research-v2-collision-check)。

我們認為部分原因出在解剖結構。嗅覺投射神經元同時接收左右觸角的輸入，在我們的模型裡，左右氣味差約有 79% 在這第一個突觸就抵消了。剩下的氣味轉向大多經由一條兩個突觸的路徑到 DNb05。

<p align="center">
  <img src="docs/assets/gates.png" width="430" alt="五個 seed 的配對比較與 95% t 區間">
  <img src="docs/assets/occlusion_attribution.png" width="460" alt="VO2 各遮蔽時段的追蹤比例">
</p>

Gazebo 方面，我們在 puff 世界做了相容性檢查：用 seed 100 的 VO2 和 V2 策略跑 48 個配對回合，48 回合的結果都和 2D 相同。感測到指令的延遲（p95）VO2 是 42.4 ms，V2 是 42.1 ms。

完整分析見 [v2 報告](docs/v2/report.md)。

## 方法

<p align="center">
  <img src="docs/assets/architecture.png" width="480" alt="系統架構">
</p>

子圖是兩個搜尋的交集：一個從視覺、嗅覺和風感的感覺神經元往下游找，另一個從下行神經元往上游找。種子神經元有 803 顆 LC、148 顆 ORN 和 115 顆 JO。權重是帶正負號的突觸數，依扇入正規化；增益、時間常數和偏置在同一個細胞型別內共用。視覺輸入來自凍結的 [flyvis](https://github.com/TuragaLab/flyvis) 模型，每眼 144 個特徵。

訓練分兩個階段：

1. DAgger，300 萬步。用成對的氣味和風反事實輸入，逼網路用上這些線索。
2. 循環 PPO，1,000 萬步，加上特權 critic、kickstarting 和遮蔽輔助損失。

策略從頭到尾都拿不到目標位置。

細節在 v2 的文件裡：

- [subgraph.md](docs/v2/subgraph.md)：子圖萃取
- [model.md](docs/v2/model.md)：網路模型和讀出
- [interfaces.md](docs/v2/interfaces.md)：感測輸入
- [world.md](docs/v2/world.md)：模擬世界
- [train.md](docs/v2/train.md)：訓練
- [metrics.md](docs/v2/metrics.md)：評估

程式裡 puff 世界叫 `v1compat`，風世界叫 `wind`。

## 安裝

在 Linux x86_64（原生和 WSL2）、Python 3.12、CUDA 12.6 上測試過。子圖和轉換好的 flyvis 前端都已放在 repo 裡。

```bash
git clone https://github.com/Nues0913/Drosophila_Tracking_System.git flytrack && cd flytrack
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --require-hashes -r requirements/ml-cu126-py312.lock  # 只用 CPU：ml-cpu-py312.lock
uv pip install --python .venv/bin/python --no-deps -e .
source .venv/bin/activate
python -m pytest -m "not slow and not gpu and not ros and not legacy and not bench" -q
```

## 使用方式

### 評估預訓練策略

60 個正式 checkpoint 的權重都在 [research-v2-formal](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/research-v2-formal)，每台訓練機器一個壓縮檔：`gputw3090`（seed 1–2）、`4060`（seed 3）、`2060`（seed 4）、`4070`（seed 5）。

```bash
gh release download research-v2-formal -p 'v2_formal_2060_weights.tar.gz' -p 'v2_formal_2060_SHA256SUMS'
sha256sum -c v2_formal_2060_SHA256SUMS
mkdir -p runs/formal_weights && tar xzf v2_formal_2060_weights.tar.gz -C runs/formal_weights
python scripts/v2/evaluate_v2.py \
  --artifacts runs/formal_weights/VO2/seed_4/stage2/final \
  --seeds formal_evaluation --worlds wind --motions curved --episodes 2 \
  --no-sensitivity --device cpu --output runs/eval_demo
```

用 CPU 約一分鐘。結果寫在 `episodes.jsonl` 和 `secondary.jsonl`，格式見 [metrics.md](docs/v2/metrics.md)。

### 訓練

`scripts/v2/run_formal.py` 會對指定的 seed 訓練並評估全部六個變體。可以先跑玩具規模的版本確認環境，RTX 2060 約 13 分鐘：

```bash
python scripts/v2/run_formal.py --host 2060 --seeds 100 --sha "$(git rev-parse HEAD)" --smoke --out runs/smoke
```

要重現 formal-v2，就拿掉 `--smoke`、改用 seed 1–5。runner 用的是正式配方 [a3_v15.yaml](configs/v2/train/a3_v15.yaml)。在 RTX 2060 上，一個 seed 約花 17 小時。`--host` 會從 [hosts.yaml](configs/v2/train/hosts.yaml) 選出適合該 GPU 的批次設定。

### 在 ROS 2／Gazebo 執行

```bash
docker compose --profile sim build
export ARTIFACT=/workspace/runs/formal_weights/VO2/seed_4/stage2/final
export SENSOR_CONTRACT=$ARTIFACT/sensor_contract.json
docker compose --profile sim up sim policy eval viz
```

policy 節點以 20 Hz 發布 `/cmd_vel`。v2 模型要像上面一樣搭配它自己的 `sensor_contract.json`。Foxglove 可以連到 `ws://localhost:8765`。機器沒有 NVIDIA 容器支援的話，加上 `-f compose.yaml -f compose.cpu.yaml`。要批次跑多個回合，可以用 `scripts/run_gazebo_matrix.py`。

## 目錄結構

```text
python/flytrack_v2/         v2 模型、GPU 世界、DAgger 和 PPO、評估
python/flytrack_core/       契約、連接體核心、flyvis 前端
python/connectome_extract/  MaleCNS 下載和子圖萃取
python/flytrack_sim/        v1 環境（G-D 的對照）
python/flytrack_train/      v1 訓練和評估
scripts/v2/                 正式 runner、評估、統計
configs/v2/                 感測契約、評估協定、訓練設定
data/subgraphs/             萃取出的子圖（CC BY 4.0）
artifacts/                  轉換好的 flyvis 前端、輸入正規化
ros2_ws/                    ROS 2 套件
docs/                       文件和報告
```

## 限制

- 訓練配方是只看 VO2 選的，這對 G-B 和 G-C 裡的 VO2 有利。五個 seed 的統計檢定力也有限。
- 同型別、同一側的視覺神經元共用一個感受野，因為 MaleCNS 沒有這些型別逐顆細胞的視網膜位置。
- 用 DNp09 和 DNb05 讀前進速度是我們的建模選擇，不是已確立的生理結果。
- Gazebo 只測了 puff 世界。

## 歷史

第一個正式版本是 [formal-v1](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/formal-v1)（2026-10-28），用 4,502 顆神經元的圖和 CPU 世界（[報告](docs/formal_v1_report.md)）。目前的版本是 [formal-v2](https://github.com/Nues0913/Drosophila_Tracking_System/releases/tag/formal-v2)（2026-11-24）。

## 團隊

[@Nues0913](https://github.com/Nues0913)、[@MiLong9487](https://github.com/MiLong9487)、[@cweiting731](https://github.com/cweiting731)

## 引用

```bibtex
@software{flytrack2026,
  title   = {FlyTrack: Connectome-Constrained Multisensory Target Tracking},
  author  = {{FlyTrack contributors}},
  year    = {2026},
  version = {formal-v2},
  url     = {https://github.com/Nues0913/Drosophila_Tracking_System}
}
```

也請引用本專案用到的資料和模型：

- S. Berg et al. Sexual dimorphism in the complete *Drosophila* male central nervous system connectome. *Cell* 189(18):5504–5526.e15, 2026. [doi:10.1016/j.cell.2026.08.015](https://doi.org/10.1016/j.cell.2026.08.015)
- J. K. Lappalainen et al. Connectome-constrained networks predict neural activity across the fly visual system. *Nature*, 2024. [doi:10.1038/s41586-024-07939-3](https://doi.org/10.1038/s41586-024-07939-3)

## 授權

我們的程式碼採 [MIT 授權](LICENSE)。repo 裡有兩部分沿用原本的授權：

- `data/subgraphs/` 的子圖衍生自 MaleCNS v1.0，仍屬 CC BY 4.0。
- `artifacts/` 的前端轉換自 flyvis（MIT）。

詳見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 致謝

MaleCNS 連接體由 FlyEM（HHMI Janelia）、劍橋大學動物學系、MRC 分子生物學實驗室和 Google Research 共同完成。感謝 flyvis 的作者公開他們的預訓練模型。
