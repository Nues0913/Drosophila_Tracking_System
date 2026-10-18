# 在其他主機跑訓練

訓練只需要 ML 環境（Python 3.12 + PyTorch）。ROS／Gazebo 不用在訓練主機上。

## 選哪種主機

| 主機 | 適合 | 注意 |
| --- | --- | --- |
| 原生 Linux＋NVIDIA GPU | 最推薦 | 直接照 README「組員快速上手」 |
| Windows＋WSL2 | 只有 Windows 電腦的組員 | WSL2 支援 CUDA；驅動裝在 **Windows** 端，WSL 內不要裝 Linux 版 NVIDIA 驅動 |
| Google Colab | 沒有合適 GPU 的組員 | session 會中斷；每階段要存到 Drive |

GPU 建議 8 GB 以上。3 GB 的 GTX 1050 已確認不夠（PPO 更新時 OOM），4–6 GB 是否足夠還沒有量過，請實測後回報。

## Windows：WSL2

1. Windows 安裝最新 NVIDIA 顯示卡驅動（Game Ready 或 Studio 皆可，內含 WSL CUDA 支援）。
2. PowerShell（系統管理員）：`wsl --install -d Ubuntu-24.04`，完成後重開機並建立使用者。
3. 進入 Ubuntu，確認 `nvidia-smi` 看得到 GPU。
4. **把 repo clone 到 WSL 的 Linux 檔案系統**（例如 `~/flytrack`），不要放在 `/mnt/c/...`，否則 I/O 會非常慢。
5. 接著照 README「組員快速上手」第 1 步以後操作。

參考：[NVIDIA CUDA on WSL 使用指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。

## Colab

使用 [notebooks/colab.ipynb](../notebooks/colab.ipynb)：它會 clone repo、安裝鎖定依賴、跑測試，並在每個訓練階段後把 `runs/` 同步到 Google Drive。子圖與前端已在 repo 中，不必在 Colab 重新下載 1.2 GB 原始資料。Colab 中斷後可從上一個已完成階段的 actor 暖啟動；PPO 中途中斷可用 `--resume` 從同階段的 recovery checkpoint 續跑。

## 分工跑正式訓練

`prepare_experiments.py` 會產生 `runs/experiment_configs/run_all.sh`，它呼叫 `scripts/run_experiments.py`：同一個「變體／種子」依序跑 pretrain → BC → PPO，不同組之間預設同時跑 2 組（`--jobs`）。分工時建議**以種子分**，每人跑全部六變體的同一個種子，讓每台機器的差異在變體間平均分攤；同一種子的預訓練資料也只需收集一次：

```bash
python scripts/prepare_experiments.py --algorithm ppo --device cuda --seeds 2
bash runs/experiment_configs/run_all.sh --seeds 2
```

也可以只挑一個變體：

```bash
bash runs/experiment_configs/run_all.sh --variants VO --seeds 0 --jobs 1
```

中斷後直接重跑同一行即可：已完成的階段會驗證後略過，進行中的階段從 checkpoint 續跑。

規則：

- 誰跑哪些「變體／種子」，以 PM 開的 GPU 執行 Issue 為準；同一組只有一位執行負責人（見 [AGENT_TEAM_WORKFLOW.md](../AGENT_TEAM_WORKFLOW.md#gpu-任務派工)）。
- **不要修改** `configs/`、`python/` 後再跑正式實驗；要改就先提 PR，合併後所有人一起重跑，否則六變體不可比較。
- 開跑前記下 `git rev-parse HEAD`；訓練產物的 `model_card.json` 也會記錄程式指紋。
- 先跑 `pytest python/tests`，確認與其他人結果一致（2026-10-18 的 `main` 為 116 passed）。
- 跑完把 `runs/<variant>/seed_<n>/`（含 `actor/` 與 log）打包交回；`runs/` 不進 git。
- 第一次跑請記錄每階段耗時與 `nvidia-smi` 顯存峰值，回報到 [status.md](status.md)。
