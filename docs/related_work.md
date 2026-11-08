# 相關研究，以及 formal-v1 未解問題的文獻對照

最後更新：2026-11-08。這份文件給報告的「相關研究」和「討論」使用。候選解法都還沒驗證，研究方向的入口是 [#35](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35)（[研究筆記 6718280403](https://github.com/Nues0913/Drosophila_Tracking_System/issues/35#issuecomment-6718280403)）。formal-v1 已登記的結論不受這份文件影響。

## 本專案的定位

| 研究 | 做了什麼 | 和本專案的差別 |
| --- | --- | --- |
| [Lappalainen et al., *Nature* 2024](https://doi.org/10.1038/s41586-024-07939-3)（flyvis） | 連接體約束的視覺系統模型，預測神經活動 | 本專案的早期視覺前端直接用它（方案 A） |
| [Shiu et al., *Nature* 2024](https://doi.org/10.1038/s41586-024-07763-9) | FlyWire 全腦的 LIF 模型，預測感覺到運動的處理 | 沒有身體和閉環行為 |
| [Cowley et al., *Nature* 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11136655) | knockout training：模型單元對應到 LC 型別，求偶行為由多種 LC 的群體編碼驅動 | 我們的 LC 輸入是依型別、側別做空間池化，沒有逐神經元對應 |
| [FlyGM, Jin et al., arXiv 2026](https://arxiv.org/abs/2602.17997) | 全腦連接體圖當控制器，用 RL 控制身體；和打亂連接、隨機圖、MLP 比較 | 我們的 RS 對照組退化（#85），不能做出同樣的結構比較 |
| [NeuroMechFly v2, Wang-Chen et al., *Nat Methods* 2024](https://pubmed.ncbi.nlm.nih.gov/39533006/) | 果蠅身體模擬，含視覺和嗅覺導航任務 | 我們用 2.5D 世界和 Gazebo，沒有腿部的生物力學 |
| [Vaxenburg et al., *Nature* 2025](https://doi.org/10.1038/s41586-025-09029-4) | MuJoCo 全身模型，用 RL 訓練走路和飛行 | 同上 |
| [Singh et al., *Nat Mach Intell* 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10601839/) | 用 RL 訓練 RNN 追蹤 turbulent plume | 我們的模型是連接體約束的，訓練用 BC，不是 RL |

## 未解問題與文獻

### 1. 前進速度是常數（#90、#95）
- **本專案**：
  - 速度只由 DNp09 讀出，DNp09 的輸入邊很少；
  - S1 把讀出的輸入放大後，速度損失的梯度壓過轉向損失，seed 0 失去轉向。
- **生物**：
  - [Braun et al., *Nature* 2024](https://doi.org/10.1038/s41586-024-07523-9)：command-like DN 會招募一整群 DN 一起控制行為。
  - Bidaye et al., *Neuron* 2020：BPN 驅動快速直走；[2026 的預印本](https://www.biorxiv.org/content/10.64898/2026.01.04.697356.full.pdf)另外找到能調整速度的腦內神經元。
  - 含意：只用一個 DN 型別讀速度，比生物上的控制窄。
- **機器學習**：S1 的失敗屬於多任務學習的梯度失衡。
  - GradNorm（Chen et al., ICML 2018）處理梯度大小；
  - [PCGrad](https://arxiv.org/abs/2001.06782)（Yu et al., NeurIPS 2020）處理梯度方向。

### 2. 沒有用氣味的方向（#4）
- **本專案**：
  - 遮蔽時拿掉氣味，成功率不變；
  - BC 時給氣味轉向標籤也沒有用（4-R8、R1）。
- **機器學習**：
  - [OGM-GE, Peng et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.pdf)：較強的模態會主導最佳化。
  - [Sensor Dropout, Liu et al., CoRL 2017](https://arxiv.org/abs/1705.10422)：訓練時隨機關掉整個感測。
  - [ADVISOR, Weihs et al., NeurIPS 2021](https://arxiv.org/abs/2007.12173)：teacher 有特權資訊時會出現 imitation gap。我們的 teacher 在遮蔽時知道目標的真實位置。
  - [causal confusion, de Haan et al., NeurIPS 2019](https://arxiv.org/abs/1905.11979)。
- **生物**：
  - 果蠅在 plume 裡主要靠風向：[Álvarez-Salvado et al., *eLife* 2018](https://elifesciences.org/articles/37815)、[Demir et al., *eLife* 2020](https://elifesciences.org/articles/57524)、[Matheson et al., *Nat Commun* 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9360402/)。
  - 左右觸角的比較可以很快引發轉向，但需要左右有足夠的濃度差：Gaudry et al., *Nature* 2013；Duistermars et al., *Curr Biol* 2009。
  - **我們的模型沒有風的輸入**（第 19 點揭露）。

### 3. 遮蔽時直線撞牆
- **本專案**：
  - 正式設定的神經時間常數初值是 50 ms。從 25 個連接體 actor 的權重讀出，每個 actor 的中位數都還是 50 ms，最長的是 0.06–0.52 s；BC 的序列只有 1.6 秒；
  - 協定的遮蔽是 30 秒（第 20 點揭露）。
- **生物**：[Neuser et al., *Nature* 2008](https://doi.org/10.1038/nature07003)：物體消失後，果蠅能記住它的方向幾秒鐘，靠的是 ellipsoid body 的 ring neurons。

### 4. PPO 沒有改善 BC（#35）
- [PIRLNav, Ramrakhya et al., CVPR 2023](https://arxiv.org/abs/2301.07302)：從 BC 初始化 actor 時，要先只訓練 critic，再逐步提高 actor 的學習率，否則一開始就會崩。
- [Rudner et al., NeurIPS 2021](https://arxiv.org/abs/2212.13936)：用 BC 得到的參考策略當 KL 錨點，RL 訓練可能出現病態的動態。

### 5. 視覺追蹤迴路
- Collie et al., bioRxiv 2025：LC10a 經 AOTU019、AOTU025 連到轉向 DN，喚起狀態會提高這條追蹤迴路的增益。
- 和第 15 點揭露一致：LC10a 有追蹤和轉向的行為證據。
