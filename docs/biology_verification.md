# #27 生物對應查證報告

**結論：DM1→Or42b、VA2→Or92a、DNp09→P9 的名稱對應已核實；DNa01／DNa02 的胞體側與左減右轉向讀出方向相容，DNb05 沿用既有側別查證。DNp09 的前進作用有因果證據，但「前進專用」與文獻不一致，因為單側 P9 活化也會同側轉向；LC10a 最適合支持追蹤輸入，LC9／LC11 支持物體偵測，LPLC2 主要是 looming 偵測，LC10d 的迴避功能本次未取得論文全文。號誌統計已核對，但固定 NT 正負號仍是模型假設，不能等同逐突觸生理驗證。這些發現影響功能解釋與訓練報告口徑，沒有建立必須改圖的充分證據。** [T] [B] [Y] [R] [S] [K] [LD]

查詢日：2026-10-25（Asia/Taipei；UTC 日期 2026-10-24）。正式圖 `data/subgraphs/olfactory_v2_nokc`，資料集 `male-cns:v1.0`；4,502 個節點、31,682 條邊、3,251,420 個保留突觸權重。所有本地資料數字由 [bio_verification.json](../artifacts/data_execution/bio_verification.json) 取值或以其中計數相除；文獻年份、識別碼、方法門檻與既有報告的定性結論另標來源。

**判讀標記**：已核實＝親查本地／作者表格，或讀過論文正文／圖說；推論＝由已核實資料外推；未核實＝原文、附表或實驗仍未取得。若只取得作者摘要，明記「摘要已讀；正文未核實」，不把摘要、作者註記、NT 預測或形態對照升級為直接生理證據。資料版本與查詢方法見文末。

## 1. ORN_DM1、ORN_VA2

**MaleCNS 原始標籤〔已核實；強〕**。以下為整個原始型別的欄位值及筆數，並非只列剪枝後節點；null 保留為缺值。逐細胞 body ID 與保留與否見 JSON `key_types[].neurons`。[A]

| type | instance | flywireType | hemibrainType | mancType | synonyms |
| --- | --- | --- | --- | --- | --- |
| ORN_DM1 | ORN_DM1_R×39；ORN_DM1_L×35 | ORN_DM1×74 | ORN_DM1×74 | null×74 | null×74 |
| ORN_VA2 | ORN_VA2_R×41；ORN_VA2_L×33；ORN_VA2_unknown×9 | ORN_VA2×83 | ORN_VA2×83 | null×83 | null×83 |

| type | somaSide | rootSide | statusLabel | 原始總數 | 正式圖 L／R／未知 |
| --- | --- | --- | --- | --- | --- |
| ORN_DM1 | null×74 | R×39；L×35 | Prelim Roughly traced×71；RT Hard to trace×2；Reviewed×1 | 74 | 35／39／0 |
| ORN_VA2 | null×83 | R×41；L×33；unknown×9 | Prelim Roughly traced×65；RT Hard to trace×18 | 83 | 29／40／0 |

| 項目 | MaleCNS 標籤或查詢結果 | 文獻結論／關鍵原文 | 證據類型 | 強度 | 來源 | 狀態 |
| --- | --- | --- | --- | --- | --- | --- |
| DM1 受體 | type／flywireType／hemibrainType=ORN_DM1；receptorType=null | Table 3 列「DM1 / Ab1A / Or42b / Orco」 | 原始分類表＋原論文表格 | 強（類型）；非本標本分子染色 | [A] [T] | 已核實 |
| VA2 受體 | type／flywireType／hemibrainType=ORN_VA2；receptorType=null | 同表列「VA2 / Ab1B / Or92a / Orco」 | 同上 | 強（類型） | [A] [T] | 已核實 |
| DM1 代表配體 | 模擬沒有化學配體欄位 | Gaudry Results／Fig. 2 使用 dilute ethyl acetate 活化 DM1；原文「odors that preferentially activate either DM6 ORNs or DM1 ORNs」 | 原文與圖說；感覺刺激 | 強（可響應）；不等於專一配體 | [G] | 已核實 |
| VA2 代表配體 | 同上 | Or92a Results：「The best ligand for Or92a was diacetyl (2,3‐butanedione).」 | GC 分離後感覺鈣成像 | 強；受實測配體集合限制 | [O] | 已核實 |
| 兩側投射與同側偏好 | rootSide 是進入 CNS 的周邊來源側，未拿它當單側投射證明 | Gaudry Fig. 2／3 顯示雙側 ORN→PN 輸入但同側較強；原文「each ORN spike typically has a larger effect on ipsilateral PNs」 | 雙側記錄、感覺刺激、突觸電流 | 強（DM1／DM6）；VA2 的同幅度未核實 | [G] | 已核實文獻；套用 VA2 為推論 |
| 模擬目標氣味 | 左右感測器取同一抽象濃度場，再經共用 Hill 轉換輸入 ORN | 文獻中的不同受體／配體不能驗證此共用抽象通道 | 程式查核＋生物比較 | 強（程式）；化學對應無證據 | [SIM] [ENC] | 程式已核實；化學身分未核實 |

**VA2 左右限制〔推論〕**：原始 83 個中有 9 個 rootSide=unknown（10.843%），不由 instance、ID 或排序猜左右；已知側原始 L=33、R=41，正式圖 L=29、R=40。除了未知側排除，還有已知側細胞因路徑／剪枝未保留，兩種原因不可混為同一件事。未知側並非已證實隨機遺失，不能宣稱剩餘左右樣本無偏，或把輸出不對稱全歸因於神經迴路。原始 statusLabel 的難追蹤狀態也須保留。

**模擬相容性〔已核實程式；推論生物解釋〕**：[env.py][SIM] 的 `_source_positions` 將移動目標置於氣味源；[plume.py][PLUME] 只產生隨風、擴散、衰減的純量濃度場，`configs/scenarios` 設定雜訊、遮蔽、干擾等條件，未定義 ethyl acetate、diacetyl 或混合物比例。[encoder][ENC] 把同一左右濃度送到 ORN 的 Hill／適應轉換。作為「帶有可偵測氣味的目標」是工程上合理的抽象；尚不能稱為已驗證的特定食物、雌蠅費洛蒙或受體親和力模型。文獻的同側釋放優勢也沒有被這次查詢驗證為模型中相同大小的生理效應。[G] [O]

## 2. LC10a、LC10d、LC9、LC11、LPLC2

**MaleCNS 原始標籤〔已核實；強〕**。以下為整個原始型別的欄位值及筆數，並非只列剪枝後節點；null 保留為缺值。逐細胞 body ID 與保留與否見 JSON `key_types[].neurons`。[A]

| type | instance | flywireType | hemibrainType | mancType | synonyms |
| --- | --- | --- | --- | --- | --- |
| LC10a | LC10a_R×140；LC10a_L×135 | LC10a×275 | null×275 | null×275 | null×275 |
| LC10d | LC10d_L×108；LC10d_R×106 | LC10d×214 | null×214 | null×214 | null×214 |
| LC9 | LC9_R×115；LC9_L×104 | LC9×219 | LC9×219 | null×219 | null×219 |
| LC11 | LC11_R×75；LC11_L×68 | LC11×143 | LC11×143 | null×143 | null×143 |
| LPLC2 | LPLC2_L×94；LPLC2_R×91 | LPLC2×185 | LPLC2×182；null×3 | null×185 | null×185 |

| type | somaSide | rootSide | statusLabel | 原始總數 | 正式圖 L／R／未知 |
| --- | --- | --- | --- | --- | --- |
| LC10a | R×140；L×135 | null×275 | Roughly traced×140；Prelim Roughly traced×135 | 275 | 105／112／0 |
| LC10d | L×108；R×106 | null×214 | Prelim Roughly traced×108；Roughly traced×106 | 214 | 81／89／0 |
| LC9 | R×115；L×104 | null×219 | Roughly traced×115；Prelim Roughly traced×104 | 219 | 5／16／0 |
| LC11 | R×75；L×68 | null×143 | Roughly traced×75；Prelim Roughly traced×68 | 143 | 8／12／0 |
| LPLC2 | L×94；R×91 | null×185 | Prelim Roughly traced×94；Roughly traced×91 | 185 | 65／90／0 |

| 項目 | MaleCNS 標籤或查詢結果 | 文獻結論／關鍵原文 | 證據類型 | 強度 | 來源 | 狀態 |
| --- | --- | --- | --- | --- | --- | --- |
| LC10a | 217／275 保留 | 追蹤與 courtship gain；2021 Results：「unilateral optogenetic silencing of LC10a neurons strongly attenuated turning」；2018 摘要明寫 LC10 | 鈣成像、單側沉默；2018 僅摘要 | 強（2021 生理／因果）；模擬映射未核實 | [H] [R18] [S] | 已核實 2021 原文；2018 亞型細節未核實 |
| LC10d | 170／214 保留 | 作者研究頁：「avoidance of a discrete object」；2026 預印本正文遇存取限制 | 作者自述；未取得原始行為圖 | 中（研究方向）；正文證據未評 | [LD] [LD-P] | 追蹤正向驅動未核實；迴避為待核實文獻候選 |
| LC9 | 21／219 保留 | Schretter Results：「LC9 was preferentially tuned to smaller moving objects」；亦報 slow dark looming response | 原論文鈣成像 | 強（調諧）；追蹤必要性較弱 | [S] | 已核實；適合作物體特徵輸入為推論 |
| LC11 | 20／143 保留 | 小型暗物體移動；原文：「the loss of retinotopic structure by LC11 within the glomerulus」 | 原論文成像、形態、藥理 | 強（特徵偵測）；不等於追蹤專用 | [L11] | 已核實；任務適用程度為推論 |
| LPLC2 | 155／185 保留 | Fig. 2 圖說：「LPLC2 population selectively encodes looming stimuli.」；Results 支持徑向運動拮抗 | 原論文刺激、成像、圖說 | 強（looming） | [K] | 已核實；作一般追蹤方向訊號為候選 |

**代表性限制〔已核實數字；推論影響〕**：LC9 只保留 9.589%，L／R=5／16；LC11 保留 13.986%，L／R=8／12。這是受最小連接權重與 seed→DN 路徑條件選出的細胞子集，不能當作完整母體或視野均勻取樣。均值池化可處理數量不同，不能恢復被移除細胞的視野與連接。

**追蹤輸入的程度〔推論〕**：LC10a 的行為證據最直接；LC9／LC11 提供小物體特徵，並不保證促進接近；LPLC2 可提供接近／碰撞訊號，與泛稱「目標追蹤細胞」不相符；LC10d 需保留迴避候選及原文未核實的限制。每個視覺細胞的受體亞型／配體並非本任務已建立的映射；NT 值見第 6 節，不能把光感受器的 histamine 傳遞套用到 LC。[H] [S] [L11] [K] [LD]

**模型映射〔已核實程式〕**：`SensoryEncoder` 使用 frozen flyvis 視覺輸出與空間池化，再映射到 LC seeds；本次未驗證它能重現每型的大小、對比、looming、求偶狀態調諧或逐細胞感受野。以上文獻功能並未因型別名稱相符而自動轉移到人工前端。[ENC]

## 3. DNa01、DNa02：證據、側別與 ω

**MaleCNS 原始標籤〔已核實；強〕**。以下為整個原始型別的欄位值及筆數，並非只列剪枝後節點；null 保留為缺值。逐細胞 body ID 與保留與否見 JSON `key_types[].neurons`。[A]

| type | instance | flywireType | hemibrainType | mancType | synonyms |
| --- | --- | --- | --- | --- | --- |
| DNa01 | DNa01(VES006)_L×1；DNa01(VES006)_R×1 | DNa01×2 | VES006×2 | DNa01×2 | null×2 |
| DNa02 | DNa02_R×1；DNa02_L×1 | DNa02×2 | DNa02×2 | DNa02×2 | null×2 |

| type | somaSide | rootSide | statusLabel | 原始總數 | 正式圖 L／R／未知 |
| --- | --- | --- | --- | --- | --- |
| DNa01 | L×1；R×1 | null×2 | Roughly traced×2 | 2 | 1／1／0 |
| DNa02 | R×1；L×1 | null×2 | Prelim Roughly traced×2 | 2 | 1／1／0 |

| 項目 | MaleCNS 標籤或查詢結果 | 文獻結論／關鍵原文 | 證據類型 | 強度 | 來源 | 狀態 |
| --- | --- | --- | --- | --- | --- | --- |
| DNa01 轉向 | flywireType=DNa01；hemibrainType=VES006；somaSide 明確 | Rayshubskiy：low-gain、較慢轉向相關；Yang Fig. 2 同側相關 | 單細胞電生理／鈣成像相關 | 強（相關），非單獨充分性 | [R] [Y] | 已核實 |
| DNa01 因果 | 型別相符不建立必要性 | Rayshubskiy Fig. 8：沉默造成小幅步態與轉向相關缺陷；作者「All these effects are weak」 | 雙側抑制、腿部運動；有 off-target 限制 | 中／弱；未證唯一必要 | [R] | 已核實；單側刺激充分性本次未核實 |
| DNa02 轉向 | somaSide=L／R 的雙側單細胞 | 較高增益、短暫、較早預測轉向 | 單細胞電生理、雙側差值 | 強（相關） | [R] | 已核實 |
| DNa02 因果 | 同側相對胞體定義 | Rayshubskiy Fig. 8 單側 ReaChR：「a small average steering bias in the ipsilateral direction」；Yang Fig. 4 活化／抑制改變同側步幅 | 單側刺激；活化與沉默的腿部效應 | 強（可影響）；雙側抑制不足證必要 | [R] [Y] | 已核實 |
| 文獻版本 | 不使用名稱相同就混合版本 | Rayshubskiy 2020 為預印本；本次證據以 2025 eLife 正文為準。Namiki 2018 提供命名與形態，不能獨立建立轉向因果 | 版本與原文查核 | 強 | [R20] [R] [N] | 2025／2018 已核實；2020 歷史差異未逐版核實 |
| 左右參照 | 模型採 somaSide，L→−1、R→+1；兩型 rootSide=null | Yang Results 原文：「the cell with a soma in the right hemisphere」；Fig. 7 亦明定 soma 參照 | 原文定義＋程式查核 | 強 | [A] [Y] [TAB] | 已核實 |
| 軸突跨中線 | 註解沒有骨架；rootSide=null 不代表跨／不跨 | Yang 明寫 DNa02 下降主幹不跨中線；Rayshubskiy Fig. 1 說兩型 VNC 侷限一側；Namiki 依胞體位置命名 | 原文形態；非本標本骨架測量 | DNa02 主幹強；DNa01 形態中 | [Y] [R] [N] | 文獻已核實；MaleCNS 逐分支未核實 |

| 型別 | 正式圖左 body ID | 正式圖右 body ID | 模型側別來源 |
| --- | --- | --- | --- |
| DNa01 | 10442 | 10760 | somaSide |
| DNa02 | 523769 | 10360 | somaSide |

**符號對照〔已核實程式；文獻到模型為推論〕**：[env.py][SIM] 使用 `yaw + omega * dt`，沿平面 x／y 座標，正 ω 為左轉。`MotorReadout` 的每型轉向特徵是 `mean(a_L) − mean(a_R)`，再乘可訓練 `turn_weight`、合計後經 tanh。左胞體較活躍→左轉→正 ω，故正的特徵係數在符號上與文獻一致；右側相反。`side=-1` 是胞體側編碼，不是 ω 的符號；不應因它為負而翻轉行為。權重初始化為正，但訓練後沒有正值約束，這份報告不保證學得係數或每個動態狀態都維持生物方向。[READ] [Y]

**解剖範圍〔推論／未核實〕**：DNa01 的單側 VNC arbor 與同側轉向不等於「腦和 VNC 任何分支都不跨中線」；本次未取得該型 Namiki 原始 XLSX 的中線／connective 側別欄，亦未對 MaleCNS skeleton／neck 截面或分側 ROI 做幾何核對。DNa02 的文獻主幹結論可用，仍不冒充本標本逐分支驗證。

## 4. DNp09：P9 對應與前進候選

**MaleCNS 原始標籤〔已核實；強〕**。以下為整個原始型別的欄位值及筆數，並非只列剪枝後節點；null 保留為缺值。逐細胞 body ID 與保留與否見 JSON `key_types[].neurons`。[A]

| type | instance | flywireType | hemibrainType | mancType | synonyms |
| --- | --- | --- | --- | --- | --- |
| DNp09 | DNp09_L×1；DNp09_R×1 | DNp09×2 | DNp09×2 | DNp09×2 | null×2 |

| type | somaSide | rootSide | statusLabel | 原始總數 | 正式圖 L／R／未知 |
| --- | --- | --- | --- | --- | --- |
| DNp09 | L×1；R×1 | null×2 | Roughly traced×2 | 2 | 1／1／0 |

| 項目 | MaleCNS 標籤或查詢結果 | 文獻結論／關鍵原文 | 證據類型 | 強度 | 來源 | 狀態 |
| --- | --- | --- | --- | --- | --- | --- |
| DNp09=P9 | MaleCNS 的 flywireType／hemibrainType／mancType 均為 DNp09；synonyms=null | Bidaye Results 明寫「DNp09 (henceforth referred to as P9)」 | 原文命名＋跨資料集表格 | 強（命名對應） | [B] [A] | 已核實；不是從 p09 字串猜測 |
| FlyWire／MANC 對照 | 同名類型，未由 null synonyms 推出不存在別名 | Stürner 作者補充檔 4／5／7 可查 DNp09；LM 參照為 R38F04，MANC systematic_type=DNxl017；未直接寫 P9 | 固定版本作者表格 | 強（類型註解）；非新骨架比對 | [NC4] [NC5] [NC7] | 已核實；P9 等同由 Bidaye 正文建立 |
| 前進＋轉向 | 正式圖保留雙側；讀出把兩側均值池化 | Bidaye Results／Fig. 2：「P9 drives forward walking with ipsilateral turning」（Highlights 作「P9 drives forward walking with ipsilateral turns whereas BPN drives straight walking.」） | 單側／雙側光刺激、行為 | 強（充分性）；非純前進 | [B] | 已核實；前進專用說法不一致 |
| 追蹤任務 | 本專案只直接將 DNp09 池化到 v | Bidaye 沉默結果支持雄蠅追逐雌蠅所需 P9 | 沉默行為；模型比較 | 強（原實驗）；模型的 v 標定未驗證 | [B] [READ] | 已核實原結果；模型用途仍是候選 |

**影響〔推論〕**：DNp09 的生物名稱與前進能力已可核實，`docs/data_sources.md` 原先「尚待文獻核對」的名稱疑問在本報告得到回答；但是它也帶有方向成分。結構式讀出將其左右活動均值送到 v，失去直接表達 P9 左右差的能力；圖內的循環作用仍可能間接影響其他轉向 DN。本次沒有做權重、行為或消融量測，不能據此認定模型必定轉錯，也不能把 v 的幅度當作 P9 實測速度編碼。此為功能規格的解釋限制，未在本次改動既有文件、圖或設定。[B] [READ]

## 5. DNb05：沿用已完成的側別查證

**MaleCNS 原始標籤〔已核實；強〕**。以下為整個原始型別的欄位值及筆數，並非只列剪枝後節點；null 保留為缺值。逐細胞 body ID 與保留與否見 JSON `key_types[].neurons`。[A]

| type | instance | flywireType | hemibrainType | mancType | synonyms |
| --- | --- | --- | --- | --- | --- |
| DNb05 | DNb05_R×1；DNb05_L×1 | DNb05×2 | DNb05×2 | DNb05×2 | Matsuo 2016: AMMC-Di7×2 |

| type | somaSide | rootSide | statusLabel | 原始總數 | 正式圖 L／R／未知 |
| --- | --- | --- | --- | --- | --- |
| DNb05 | R×1；L×1 | null×2 | Prelim Roughly traced×2 | 2 | 1／1／0 |

本節整理計劃書提供的 `2026-10-24-dnb05-side/dnb05_side.md` 及 #27 留言 6384888701 的既有結論；沒有重跑其公開 ROI、MANC／FlyWire 身分、骨架或校準查詢。只以本次相同 raw 表核對當前正式圖的身分／左右。既有報告讀取版本為 Yang 2024 的 PMC 作者接受稿，及固定版本作者表格；其限制原樣保留。

| 項目 | MaleCNS 標籤或查詢結果 | 文獻結論／關鍵原文 | 證據類型 | 強度 | 來源 | 狀態 |
| --- | --- | --- | --- | --- | --- | --- |
| 身分 | DNb05_L／R；synonyms=Matsuo 2016: AMMC-Di7；原始狀態 Prelim Roughly traced | 先前報告的跨資料集類型與當前表一致 | 本地表＋既有查證 | 強（表格） | [A] [NC4] [NC5] [NC7] | 已核實 |
| 同側轉向 | 左右依 somaSide | 既有報告引用 Yang 的胞體側定義；左側活動較高對應左轉 | 鈣成像相關＋符號換算 | 強（方向）；不等於刺激因果 | [Y] | 文獻已核實；模型換算為推論 |
| 軸突走向 | rootSide=null，未當成同側證據 | 既有報告支持腦內與主下降通路同側、VNC 主要同側，但記錄少量對側 T1 presynapses | 既有形態原文與公開 ROI | 中至強；無逐點 EM 校核 | [M16] [EL] [ER] | 沿用既有已核實資料；完全不跨中線未核實 |
| B3 額外讀出 | DNb05 在圖內；不屬 extraction 的 turn／forward mask | 讀出以閘控、置中、尺度化的左右差乘可訓練 extra weight | 既有報告結論＋當前程式 | 強（程式）；不建立增益正確性 | [READ] | 程式已核實；校準幅度未重測 |

**沒做的項目〔NOT RUN／未核實〕**：保留既有報告所列的 MaleCNS 對側 T1 訊號 skeleton／EM 校核、MANC 分側 ROI 與 neck 截面、Namiki 補充表原列、FlyWire／hemibrain 分側 ROI；亦未重跑 B3 校準、增益或雄蠅因果行為實驗。原報告能支持方向一致，不能證成所有軸突都不跨側。Matsuo 的 Di 命名指腦內 ipsilateral arborization，不自動保證 VNC 無對側細支。[M16] [Y]

## 6. 神經傳導物質、型別分歧與號誌

| 項目 | MaleCNS 標籤或查詢結果 | 文獻結論／關鍵原文 | 證據類型 | 強度 | 來源 | 狀態 |
| --- | --- | --- | --- | --- | --- | --- |
| 正式圖 NT／號誌 | ACh=3265；GABA=767；Glu=470 | NT 是發布者 consensus 標籤；固定極性並非逐突觸生理量測 | 表格與程式 | 強（計數／規則） | [JSON] [EXTRACT] | 已核實 |
| consensus_nt 來源 | raw 同時有 predicted_nt、celltype_predicted_nt、ground_truth、各自 confidence | MaleCNS Methods：「a copy of celltypePredictedNt」；有實驗 ground truth 時覆蓋，並處理未充分驗證的單胺預測 | 原論文 Methods；本地 schema | 強（方法；讀取 2025 v2） | [MC] [JSON] | 已核實；2026 發表版細節未逐項核實 |
| 數值信心與門檻 | normalized nt_confidence 可用=0；nt_confidence_min=0.5 | 官方以突觸分類混淆矩陣形成 aggregate confidence；consensus 另含實驗覆蓋，並沒有一個對等的 consensus confidence | 程式、原表、Methods | 強 | [TAB] [EXTRACT] [MC] | 已核實；本地門檻未再次篩除共識標籤 |
| 型別多數決 | 本圖 nt_imputed=0；consensus 型別內分歧型別=0 | 官方是池化 presynapse 預測；本程式的第二層處理只補未定 sign，不覆蓋有效 sign | 方法與程式 | 強（演算法）；非準確率 | [MC] [EXTRACT] | 已核實 |
| unknown_nt_policy: drop | 88 個／候選 4903 個 | 含 unclear 及固定符號字典不支援的已知 neuromodulator；不全是 NT 不知道 | 原表、門檻圖重建 | 強（計數／拓撲） | [JSON] [EXTRACT] | 已核實 |
| glutamate 極性 | 模型 glu／glutamate→−1；所有同來源邊共用該 sign | Liu & Wilson Results：「GluClα knockdown virtually abolished the response to iontophoresed glutamate」；AL 有 GluClα 抑制 | 電生理、RNAi、藥理 | 強（AL）；全腦通用負號未建立 | [LW] [EXTRACT] | 已核實局部生理；全圖外推為推論 |

### 6.1 分配規則與信心的實際語義

模型保留的正號節點 3265、負號節點 1237。`NT_SIGNS` 把 acetylcholine／ach 設正號、gaba 與 glutamate／glu 設負號；其餘 NT 沒有固定號誌。`assign_signs` 先對有效數值信心低於門檻者清掉 sign，然後在**所有原始註解同 type** 的既有正／負 sign 中取唯一最多者，只補缺 sign；平手不補。GABA 和 Glu 在這一步屬同一負號，不是彼此競爭的 NT 標籤，原有相反 sign 也不會被多數覆蓋。[EXTRACT]

**不是原表沒有 confidence**：Feather 明確有 `predicted_nt_confidence` 及 `celltype_predicted_nt_confidence`。本設定固定讀 `consensus_nt`，`load_neurotransmitters` 在此模式沒有選用數值信心欄，因而全部 normalized `nt_confidence=NaN`。所以本地 `nt_confidence_min` 條件不會觸發；它不代表每個 consensus 都被本專案驗證達門檻，也不表示原始個別預測沒有信心。官方 Methods 的個別預測以至少 50 presynapses、型別至少 100 pooled presynapses，並採各自 0.5 confidence 門檻；低於門檻標 unclear。型別共識來自池化突觸投票及實驗覆蓋，而非簡單的每細胞一票。本次只讀了該論文 2025 v2 的 Methods，正式發表版的逐項異動未核實。[MC] [TAB]

**版本差異〔已核實資料；解釋有限〕**：2025 Methods 對 octopamine／serotonin 預測有保守處理敘述；本地 v1.0 表仍有這些 consensus 值（本次移除清單亦可見）。不能從較早 Methods 的文字把本地值覆寫成 unclear，亦不能假設後來共識欄每個值都附同等數值信心。這些已知但不具固定極性的 transmitter 在本模型仍會被 drop。[MC] [JSON]

### 6.2 個別預測相對型別眾數的分歧

母體＝原始 annotations 中具非空 type 且可對上 NT 表的細胞，使用 **individual `predicted_nt`**，不是已經型別化的 consensus。排除 unclear／null 後，分歧率＝`(resolved − 最大 NT 類別筆數) / resolved`；全未定者為 null，平手保留多個眾數而不指定勝者。全域有 123 個分歧型別；515／148807 個已解析細胞不屬任一指定單一眾數最大筆數（0.346%）。此率只描述已發布、已過個別預測過濾的標籤，排除的 unclear 比例另列，不能當作未過濾模型錯誤率。

以下是**全域**分歧率最高的型別，按分歧率降序、resolved 降序、type 名升序取前 20。圖內數量為零也保留。權重占比＝該 type 在正式圖作為 presynaptic 節點的 `synapse_count` 總和／全圖突觸總和，單向歸屬，不重複算入兩端。平手的分歧率仍可描述異質性，不能宣稱已有唯一多數。來源：[JSON] `majority_disagreement.individual_top20_global`。

| type | 不同於眾數筆數／resolved | 未定數 | 分歧率 | 唯一眾數 | 正式圖節點 | 出邊突觸權重 | 全圖權重占比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pleural remotor/abductor MN | 2／3 | 11 | 66.667% | 否 | 0 | 0 | 0.000% |
| GNG643 | 8／16 | 2 | 50.000% | 否 | 0 | 0 | 0.000% |
| ENS4 | 2／4 | 1 | 50.000% | 否 | 0 | 0 | 0.000% |
| ME_LO_unclear | 2／4 | 0 | 50.000% | 是 | 0 | 0 | 0.000% |
| SLP304 | 2／4 | 0 | 50.000% | 否 | 0 | 0 | 0.000% |
| lLN2T_b | 2／4 | 0 | 50.000% | 否 | 4 | 15177 | 0.467% |
| AVLP548_b | 1／2 | 1 | 50.000% | 否 | 0 | 0 | 0.000% |
| CB2471 | 1／2 | 3 | 50.000% | 否 | 0 | 0 | 0.000% |
| CB3762 | 1／2 | 2 | 50.000% | 否 | 0 | 0 | 0.000% |
| CB4246 | 1／2 | 0 | 50.000% | 否 | 0 | 0 | 0.000% |
| CL155 | 1／2 | 0 | 50.000% | 否 | 0 | 0 | 0.000% |
| GNG101 | 1／2 | 0 | 50.000% | 否 | 0 | 0 | 0.000% |
| GNG490 | 1／2 | 0 | 50.000% | 否 | 0 | 0 | 0.000% |
| GNG502 | 1／2 | 0 | 50.000% | 否 | 2 | 2723 | 0.084% |
| GNG591 | 1／2 | 0 | 50.000% | 否 | 0 | 0 | 0.000% |
| GNG655 | 1／2 | 0 | 50.000% | 否 | 0 | 0 | 0.000% |
| IN03B008 | 1／2 | 0 | 50.000% | 否 | 0 | 0 | 0.000% |
| INXXX183 | 1／2 | 0 | 50.000% | 否 | 0 | 0 | 0.000% |
| MNx05 | 1／2 | 0 | 50.000% | 否 | 0 | 0 | 0.000% |
| SLP463 | 1／2 | 2 | 50.000% | 否 | 0 | 0 | 0.000% |

正式圖內有個別分歧的型別只有 7 個，以下全部列出；不足前 20 的其餘型別分歧率為零，完整排名見 JSON `individual_top20_present_in_graph`。

| type | 個別 predicted_nt（原始型別） | consensus_nt（原始型別） | 分歧率 | 圖內節點 | 圖內出邊突觸 | 全圖權重占比 |
| --- | --- | --- | --- | --- | --- | --- |
| lLN2T_b | acetylcholine×2；gaba×2 | acetylcholine×4 | 50.000% | 4 | 15177 | 0.467% |
| GNG502 | gaba×1；octopamine×1 | gaba×2 | 50.000% | 2 | 2723 | 0.084% |
| VES105 | gaba×1；octopamine×1 | gaba×2 | 50.000% | 1 | 902 | 0.028% |
| VES106 | octopamine×1；gaba×1 | gaba×2 | 50.000% | 1 | 1197 | 0.037% |
| vLN24 | acetylcholine×2；unclear×1；glutamate×1 | acetylcholine×4 | 33.333% | 3 | 4995 | 0.154% |
| BM | acetylcholine×57；serotonin×7；unclear×5 | acetylcholine×69 | 10.938% | 1 | 283 | 0.009% |
| ORN_DL1 | acetylcholine×78；unclear×4；gaba×1 | acetylcholine×83 | 1.266% | 1 | 196 | 0.006% |

**lLN2T_b〔已核實資料；推論判讀〕**：個別 ACh／GABA 預測平手，consensus 與 raw `ground_truth` 為 ACh。作者 NT 整理表將此型對上 Shang 2007 的 cholinergic LN 免疫證據；這個舊實驗類型到新 type 的映射本次未獨立重建，因此只列為支持 ACh 的作者對照，不能把平手逕判成正式圖錯號。其餘分歧也不能只憑少數票推翻共識。[GT] [SH]

### 6.3 被移除的未定號誌細胞與路徑

以原始 scope、排除 KC／APL／DPM、最小邊權重 50 與原始 hop budgets 重建候選交集，重現候選 4903 個、未定 sign 88 個（1.795%）。只從記憶體候選圖移除這些節點，不執行萃取器寫檔。[JSON]

| type | 移除節點數 | consensus NT | 移除前 incident 突觸權重 |
| --- | --- | --- | --- |
| ALBN1 | 2 | unclear×2 | 8425 |
| ALIN1 | 2 | unclear×2 | 1469 |
| AVLP476 | 2 | dopamine×2 | 3159 |
| AVLP532 | 2 | unclear×2 | 13467 |
| AVLP538 | 2 | unclear×2 | 28198 |
| AVLP610 | 2 | dopamine×2 | 5208 |
| CB2463 | 4 | unclear×4 | 878 |
| CB3417 | 3 | unclear×3 | 1894 |
| CSD | 2 | serotonin×2 | 6592 |
| DNg86 | 2 | unclear×2 | 2017 |
| DNp29 | 1 | unclear×1 | 406 |
| GNG190 | 1 | unclear×1 | 585 |
| LAL082 | 1 | unclear×1 | 762 |
| LAL123 | 2 | unclear×2 | 8607 |
| LAL128 | 1 | dopamine×1 | 567 |
| LHAV4e1_a | 1 | unclear×1 | 126 |
| LHPV6q1 | 1 | unclear×1 | 1330 |
| OA-ASM2 | 1 | unclear×1 | 233 |
| OA-ASM3 | 1 | unclear×1 | 129 |
| OA-VUMa1 | 2 | octopamine×2 | 4180 |
| OA-VUMa2 | 2 | octopamine×2 | 4973 |
| OA-VUMa4 | 2 | octopamine×2 | 2701 |
| OA-VUMa6 | 2 | octopamine×2 | 4421 |
| OA-VUMa8 | 1 | octopamine×1 | 4396 |
| PLP211 | 1 | unclear×1 | 667 |
| PPL101 | 1 | dopamine×1 | 1550 |
| PPL103 | 2 | dopamine×2 | 3787 |
| PPL104 | 1 | dopamine×1 | 253 |
| PPL105 | 2 | dopamine×2 | 1342 |
| PPL107 | 2 | dopamine×2 | 1345 |
| PPL201 | 1 | dopamine×1 | 1066 |
| PPL202 | 2 | dopamine×2 | 757 |
| PPM1201 | 1 | dopamine×1 | 580 |
| PPM1203 | 2 | dopamine×2 | 2818 |
| PPM1205 | 2 | dopamine×2 | 3506 |
| PVLP106 | 2 | unclear×2 | 54667 |
| SIP106m | 1 | dopamine×1 | 888 |
| lLN10 | 2 | unclear×2 | 1675 |
| lLN2F_a | 4 | unclear×4 | 66052 |
| lLN2T_d | 4 | unclear×4 | 14287 |
| v2LN30 | 2 | unclear×2 | 63432 |
| v2LN31 | 2 | unclear×2 | 3507 |
| v2LN3A | 3 | unclear×3 | 3259 |
| v2LN40_2 | 4 | unclear×4 | 1166 |
| v2LN41 | 1 | unclear×1 | 399 |
| vLN27 | 2 | unclear×2 | 1629 |
| vLN29 | 2 | unclear×2 | 771 |

**拓撲影響〔已核實〕**：移除前候選圖突觸權重 3,688,240，接到被移除節點的邊聯集權重 327,602（8.882%）。上表 incident 欄按各型分別計算，跨兩個移除型別的邊會在兩列出現，不能逐列加總取代此聯集值。所有 726 個候選內 seeds 仍可達各指定 DN：遺失 seed×DN 配對 0、最短路徑增加配對 0。正式圖也保留這些 seeds。

| DN body ID | 移除前可達 seeds | 僅移除未定 sign 後 | 正式圖可達 seeds |
| --- | --- | --- | --- |
| 10360 | 726 | 726 | 726 |
| 10442 | 726 | 726 | 726 |
| 10760 | 726 | 726 | 726 |
| 10783 | 726 | 726 | 726 |
| 11177 | 726 | 726 | 726 |
| 523769 | 726 | 726 | 726 |

**限制〔推論〕**：這只能證明無號誌有向圖的可達性與最短跳數未受損，不能證明全部平行路徑、總電流、抑制、神經調質或行為不受影響。對於已知單胺，drop 的原因是模型沒有單一固定極性，而不是發布者沒有辨識 NT。後續固定點剪枝的其他被移除節點亦不能全算入這個未定 sign 計數。

### 6.4 關鍵型別及主要中繼的 NT 對照

**中繼的操作定義**：正式圖位於 seed→DN 保留路徑、未被指定為外部 input 或主要 turn／forward readout 的節點，排除本報告已列關鍵型別後，依其出邊突觸總權重取前 20 型。它可能包含其他 ORN／LC，並非只限 PN／LN，也不是路徑流量或因果重要性排名。下表先列全部關鍵型別，再列此中繼排名；預測／consensus／ground_truth 的完整原始分布在 JSON `key_nt_comparison`。

文獻欄中「作者對照」指 [GT] 的原作者維護實驗整理（本次親讀表格；不是獨立重做實驗），不是自行憑記憶補值。「原實驗未核實」表示未讀到該新 type 的逐列實驗證據；既有 `ground_truth` 非空也不等於本次親讀原實驗。文獻未知的列保持未知，不從相同腦區或名稱補 NT。

| type | 圖內數 | 圖內 consensus | 個別預測（原型別） | raw ground_truth | 出邊權重／占全圖 | 文獻已知值 | 來源 | 一致性與查證狀態 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ORN_DM1 | 74 | ACh×74 | acetylcholine×74 | acetylcholine×74 | 13266／0.408% | ACh；Carlsson 2010／Yasuyama 1999 作者對照 | [GT] [SH] | 一致（作者對照）；原型別實驗未核實 |
| ORN_VA2 | 69 | ACh×69 | acetylcholine×80；unclear×3 | acetylcholine×83 | 12519／0.385% | ACh；同上 | [GT] [SH] | 一致（作者對照）；原型別實驗未核實 |
| LC10a | 217 | ACh×217 | acetylcholine×275 | acetylcholine×275 | 65046／2.001% | ACh；Davis 2020 TAPIN 作者對照 | [GT] [D20] | 一致（作者對照）；逐亞型原始附表未核實 |
| LC10d | 170 | ACh×170 | acetylcholine×214 | null×214 | 35038／1.078% | ACh；Davis 2020 TAPIN 作者對照 | [GT] [D20] | 一致（作者對照）；逐亞型原始附表未核實 |
| LC9 | 21 | ACh×21 | acetylcholine×219 | null×219 | 1389／0.043% | 作者整理僅有非 GABA 證據，未建立正向 ACh 實驗值 | [GT] [JSON] | 未核實，無法判定一致 |
| LC11 | 20 | ACh×20 | acetylcholine×143 | null×143 | 1241／0.038% | 作者整理僅有非 GABA 證據，未建立正向 ACh 實驗值 | [GT] [JSON] | 未核實，無法判定一致 |
| LPLC2 | 155 | ACh×155 | acetylcholine×185 | acetylcholine×185 | 15716／0.483% | ACh；Davis 2020 TAPIN 作者對照 | [GT] [D20] | 一致（作者對照）；原始附表未核實 |
| DNa01 | 2 | ACh×2 | acetylcholine×2 | acetylcholine×2 | 0／0.000% | ACh；Cheong 作者對照；原文 Fig. 4／補圖有 FISH | [GT] [C] | 一致（作者對照）；本次未逐圖量化 |
| DNa02 | 2 | ACh×2 | acetylcholine×2 | acetylcholine×2 | 1874／0.058% | ACh；Cheong 作者對照 | [GT] [C] | 一致（作者對照）；逐型原始 FISH 未核實 |
| DNp09 | 2 | ACh×2 | acetylcholine×2 | acetylcholine×2 | 1543／0.047% | ACh；Cheong 作者對照 | [GT] [C] | 一致（作者對照）；逐型原始 FISH 未核實 |
| DNb05 | 2 | ACh×2 | acetylcholine×2 | acetylcholine×2 | 1242／0.038% | ACh；Cheong 作者對照 | [GT] [C] | 一致（作者對照）；逐型原始 FISH 未核實 |
| LC17 | 352 | ACh×352 | acetylcholine×353 | null×353 | 131091／4.032% | 未知；未取得此型的實驗 NT 原文 | [GT] [JSON] | 未核實，無法判定一致 |
| lLN1_bc | 30 | ACh×30 | acetylcholine×30 | acetylcholine×30 | 118823／3.654% | ACh；Shang 2007 作者對照 | [GT] [SH] | 一致（作者對照）；新舊型別映射為推論 |
| PVLP011 | 2 | GABA×2 | gaba×2 | null×2 | 40069／1.232% | 未知；未取得此型的實驗 NT 原文 | [GT] [JSON] | 未核實，無法判定一致 |
| lLN2F_b | 4 | GABA×4 | gaba×4 | null×4 | 35552／1.093% | 未知；未取得此型的實驗 NT 原文 | [GT] [JSON] | 未核實，無法判定一致 |
| ORN_DP1m | 27 | ACh×27 | acetylcholine×31 | acetylcholine×31 | 28723／0.883% | 未知；未取得此型的實驗 NT 原文 | [GT] [JSON] | 未核實，無法判定一致 |
| lLN2T_c | 4 | ACh×4 | acetylcholine×4 | acetylcholine×4 | 26722／0.822% | ACh；Shang 2007 作者對照 | [GT] [SH] | 一致（作者對照）；映射為推論 |
| lLN2T_a | 5 | ACh×5 | acetylcholine×5 | acetylcholine×5 | 22824／0.702% | ACh；Shang 2007 作者對照 | [GT] [SH] | 一致（作者對照）；映射為推論 |
| lLN2X12 | 11 | ACh×11 | acetylcholine×7；unclear×4 | null×11 | 21735／0.668% | 未知；未取得此型的實驗 NT 原文 | [GT] [JSON] | 未核實，無法判定一致 |
| il3LN6 | 2 | GABA×2 | gaba×2 | gaba×2 | 20542／0.632% | GABA；Tanaka 2012 作者對照 | [GT] [TN] | 一致（作者對照）；原始逐型實驗未核實 |
| lLN2P_c | 9 | GABA×9 | gaba×9 | null×9 | 18871／0.580% | 未知；未取得此型的實驗 NT 原文 | [GT] [JSON] | 未核實，無法判定一致 |
| DP1m_adPN | 2 | ACh×2 | acetylcholine×2 | acetylcholine×2 | 17098／0.526% | ACh；同上 | [GT] [TN] | 一致（作者對照）；原始逐型實驗未核實 |
| AVLP001 | 2 | GABA×2 | gaba×2 | null×2 | 16978／0.522% | 未知；未取得此型的實驗 NT 原文 | [GT] [JSON] | 未核實，無法判定一致 |
| lLN2X11 | 4 | ACh×4 | acetylcholine×3；unclear×1 | null×4 | 15291／0.470% | 未知；未取得此型的實驗 NT 原文 | [GT] [JSON] | 未核實，無法判定一致 |
| AOTU042 | 4 | GABA×4 | gaba×4 | null×4 | 15180／0.467% | 未知；未取得此型的實驗 NT 原文 | [GT] [JSON] | 未核實，無法判定一致 |
| lLN2T_b | 4 | ACh×4 | acetylcholine×2；gaba×2 | acetylcholine×4 | 15177／0.467% | ACh；Shang 2007 作者對照 | [GT] [SH] | consensus 一致；個別預測平手 |
| DM1_lPN | 2 | ACh×2 | acetylcholine×2 | acetylcholine×2 | 14296／0.440% | ACh；Tanaka 2012 免疫／tract 作者對照 | [GT] [TN] | 一致（作者對照）；原始逐型實驗未核實 |
| AL-MBDL1 | 2 | ACh×2 | acetylcholine×2 | null×2 | 14226／0.438% | 未知；未取得此型的實驗 NT 原文 | [GT] [JSON] | 未核實，無法判定一致 |
| PVLP061 | 2 | ACh×2 | acetylcholine×2 | null×2 | 13622／0.419% | 未知；未取得此型的實驗 NT 原文 | [GT] [JSON] | 未核實，無法判定一致 |
| lLN2X05 | 4 | ACh×4 | acetylcholine×4 | acetylcholine×4 | 13268／0.408% | 未知；未取得此型的實驗 NT 原文 | [GT] [JSON] | 未核實，無法判定一致 |
| AVLP080 | 2 | GABA×2 | gaba×2 | null×2 | 13010／0.400% | 未知；未取得此型的實驗 NT 原文 | [GT] [JSON] | 未核實，無法判定一致 |

**GluClα 與模型外推〔已核實／推論〕**：Liu & Wilson 的 AL 實驗證明 glutamate 可經 GluClα 產生抑制，與本模型把 Glu 設負號在方向上相容；原文也保留其他受體／情境的可能性。這不是全圖每個 Glu target 的受體表，本次沒有查每條邊的 postsynaptic GluClα、離子反轉電位、共釋放或受體分布。因此「Glu 一律負號」與「每個 presynaptic 細胞所有邊同號」應寫成模型假設，不能反向宣稱所有 glutamate 都在果蠅中抑制，或引用脊椎動物的通則直接判成本模型錯誤。[LW] [D20] [EXTRACT]

## 7. 匯流神經元

正式圖的 convergence 位元有 59 個細胞，型別清單如下；是兩種模態都可在設定的 convergence hop 範圍到達的節點，不是 visual／olfactory 群的任意重疊，更不是生理上已證實的多感覺整合。查詢直接讀 `group_mask & GROUP_CONVERGENCE`，計數與 manifest 相符。[JSON] [EXTRACT]

| 項目 | MaleCNS 標籤或查詢結果 | 文獻結論／關鍵原文 | 證據類型 | 強度 | 來源 | 狀態 |
| --- | --- | --- | --- | --- | --- | --- |
| 匯流數量 | 59 個；以下依 type 分組 | 沒有文獻將這份人工剪枝子圖定義為相同集合 | 正式圖位元與 manifest | 強（圖定義） | [JSON] [EXTRACT] | 已核實圖；生理匯流未核實 |
| DNp31 已知部分 | 2 個 | Cheong Fig. 12：「power MN-associated DNs」；連接翼動力網路，視覺顯著刺激功能屬推測 | 原論文 EM 迴路與圖說 | 強（連接）；行為功能中／弱 | [C] | 已核實文獻連接；功能為推論 |
| 其他型別的命名 | 保留 MaleCNS 原 type，不以相似字串套功能 | 如 pIP1 不等於 pIP10；未逐型找到可直接對應的功能原文 | 命名與來源限制 | 未評 | [JSON] [N] | 未核實者明列 |

| type | 數量 | L／R／未知 | 可支持的一句話功能／狀態 |
| --- | --- | --- | --- |
| AVLP015 | 1 | 0／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| AVLP597 | 1 | 0／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| AVLP749m | 2 | 0／2／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| CB0204 | 2 | 1／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| CB0316 | 1 | 1／0／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| CB0356 | 1 | 1／0／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| CB0492 | 1 | 1／0／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| DNae007 | 2 | 1／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| DNbe007 | 2 | 1／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| DNde002 | 2 | 1／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| DNg35 | 2 | 1／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| DNg56 | 2 | 1／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| DNge041 | 2 | 1／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| DNge054 | 2 | 1／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| DNp31 | 2 | 1／1／0 | 翼動力 premotor 網路相關 DN；行為功能為推論。[C] |
| DNp56 | 2 | 1／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| DNpe002 | 2 | 1／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| DNpe025 | 1 | 1／0／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| GNG300 | 1 | 0／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| GNG494 | 1 | 0／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| GNG559 | 2 | 1／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| LAL011 | 1 | 1／0／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| LAL183 | 1 | 1／0／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| LHAD1g1 | 2 | 1／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| MZ_lv2PN | 1 | 0／1／0 | 嗅覺 PN 類型為命名與分類推論；本次未讀到直接生理功能原文。 |
| PS304 | 1 | 1／0／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| PVLP022 | 2 | 1／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| PVLP076 | 2 | 1／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| PVLP211m_c | 1 | 1／0／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| PVLP217m | 1 | 0／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| SAD013 | 2 | 1／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| SIP022 | 2 | 1／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| SLP057 | 2 | 1／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| SMP177 | 2 | 1／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| VES013 | 1 | 0／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| VES022 | 1 | 1／0／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| VES079 | 1 | 0／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |
| pIP1 | 2 | 1／1／0 | 此新型別的直接生理功能未核實；不從圖上匯流身分推定功能。 |

## 報告必須揭露的限制

1. 這是固定單一雄蠅 MaleCNS v1.0 的剪枝圖；文獻可能使用雌蠅、其他標本與 driver。類型對應支持引用，不建立性別、個體、增益或動態完全一致。`statusLabel` 的 Roughly／Prelim／Hard to trace 是原始重建註記，不改寫成分子功能的已驗證狀態。[A] [Y] [MC]
2. 受體對應由文獻確認；本地 `receptorType` 對上述關鍵型別皆缺值。模擬沒有具名化學配體，DM1／VA2 共用抽象濃度，不是受體化學模型。[JSON] [T] [SIM] [ENC]
3. ORN 的 rootSide 是來源天線側；雙側投射與同側釋放優勢不能簡化成單側 ORN 投射。VA2 未知側排除及 LC9／LC11 少量、偏側保留使左右與代表性比較受限。[G] [JSON]
4. LC 前端為 flyvis＋工程池化，未重現或測量每型文獻調諧；LC10d 原研究全文未取得，迴避結論保留為候選。LPLC2 的 looming 功能不證明泛化追蹤能力。[ENC] [LD-P] [K]
5. DNa01 的主要強證據是相關性；DNa02 有刺激因果，抑制未顯著破壞整體轉向不能推論其不參與。Namiki 原表中線／下降側別欄與本標本骨架未核實，DNb05 沿用既有報告的缺口。[R] [N] [Y]
6. NT 原表存在數值信心，本專案 consensus 讀取模式沒有消費它；原始預測、型別共識、ground_truth、模型固定 sign 是不同層。作者 NT 整理提供的文獻值不冒充本次逐型親讀原實驗。個別預測不一致率不是錯誤率。[MC] [GT] [TAB]
7. 只驗證未定 sign 移除的有向可達性及最短跳數；沒有活動、電流、行為、受體、共釋放或訓練後符號驗證。匯流為圖論定義，NT 正負號為模型假設。[JSON] [EXTRACT] [READ]
8. 部分來源在瀏覽時遭 reCAPTCHA、403 或檔案格式／大小限制；使用可讀的 PMC／eLife 原文作替代，仍無法取得者保留未核實。沒有重新下載大型 synapse／skeleton 資料，沒有啟動訓練或使用 GPU。

## 需要改圖或改規格的發現

**改圖：無已核實且足以要求改圖的發現。** 以下僅揭露影響，留待 DRI／PM 決策，未提出替換細胞、重萃取、修改參數或其他修改方案，也未建立 Issue。

| 發現 | 影響 | 狀態 |
| --- | --- | --- |
| DNp09=P9 且兼具同側轉向 | 「純前進神經元」的功能解釋與原文不一致；兩側池化到 v 是工程選擇，生物名稱已核實但專用性仍不成立。 | 需要規格／訓練報告口徑決策；未改規格 |
| LC seeds 並非同一類追蹤驅動 | LPLC2 是 looming 特徵；LC10d 迴避為待核實候選；不能把整組輸入稱作已驗證的接近／追蹤通道。 | 文獻功能差異與未核實範圍；無足夠證據指圖錯誤 |
| consensus 讀取未用本地數值信心門檻 | 不能將設定值當成本專案實測保證；原始表有 confidence，文件若稱整份原始資料沒有信心欄會誤導。 | 查證已核實；未修改既有 data_sources／設定 |
| 未定 sign drop 同時排除已知單胺 | 保留圖少了部分調質連接；路徑仍可達不代表功能不受影響。 | 查證已核實；未要求改圖 |
| 個別預測分歧及固定 Glu 負號 | 共識與原實驗需分層解讀；未查每個 postsynaptic 受體，不能聲稱全圖極性已經生理核實。 | 需揭露限制；未判定任何特定邊必須翻號 |

## 資料版本、查詢方法與重現

本地資料版本 `male-cns:v1.0`。annotations 共 211,577 列；NT 共 1,835,518 列。以原始 `type` 精確比對關鍵型別，再以 body ID join 正式圖，列出每個要求欄位的原值分布；不以 ID 推斷側別。權重檔用 Arrow record batches 分批讀取，只保留 scope 內且達門檻的邊。重新計算 raw 與圖檔 SHA-256，並與 manifest 比對；計數、號誌、候選交集與匯流檢查全部通過。[JSON]

| 原始表格（相對 data/raw） | SHA-256 |
| --- | --- |
| body-annotations-male-cns-v1.0-minconf-0.5.feather | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` |
| body-neurotransmitters-male-cns-v1.0.feather | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` |
| connectome-weights-male-cns-v1.0-minconf-0.5.feather | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` |

正式圖 manifest 記錄的產圖程式 commit：`09d704391eba14f0c5972961d2782bb5aed5108a`。計劃書指定的工作基準為 main `897d088`；這兩者角色不同，不把 main 的 commit 冒充原始產圖 commit。本次查詢腳本尚未 commit，以 SHA-256 `600a5c2319577743f69aaa2d7577c3f8c80729080e622d5be54c5aeeb59f16f6` 固定其內容；被讀取的既有程式 SHA-256 另存 JSON `provenance.query_source_sha256`。依任務要求未變更 Git 狀態。

重現命令（由 `<repo>` 執行，`<venv>` 為已有環境；只在呼叫端限制記憶體，不寫入本機路徑）：

```bash
prlimit --as=6442450944 env PYTHONPATH=python <venv>/bin/python scripts/diagnostics/bio_verification.py < /dev/null
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python <venv>/bin/python -m pytest -q python/tests < /dev/null
<venv>/bin/ruff check python scripts < /dev/null
<venv>/bin/ruff format --check python scripts < /dev/null
```

查詢使用 `prlimit --as`，上限 6 GiB；未使用 GPU。測試執行結果見下方驗證記錄。文獻只經讀取工具取得，未計算所有論文 PDF 的雜湊，故不能宣稱 PDF 位元組可重現。Stürner 公開表固定於 commit `94637d7d82920234ef0bfdbf383c000e74a8b45a`；本次查表不代表即時 FlyWire materialization 或 MANC neuPrint 查詢。NT 作者整理表讀取的 blob SHA-1 為 `a99c2e07f24d0ec06b703bdb38f4c37cc2bf29d4`。[NC4] [NC5] [NC7] [GT]

### 驗證記錄

診斷查詢在 `prlimit --as=6442450944` 下 exit 0，JSON 內各項核對全部為 true；新增的小型假表格測試含在全套測試內。

```text
333 passed, 1 skipped, 15 warnings in 316.24s (0:05:16)
All checks passed!
96 files already formatted
```

全套測試以 CPU 執行並隱藏 CUDA 裝置；唯一 skip 為 CUDA 專用測試。Ruff 檢查與格式檢查皆 exit 0。最終文件、腳本與 JSON 的本機絕對路徑掃描無命中。起初額外把整套 pytest 也套上位址空間上限，導致既有前端配置記憶體失敗與無法建立執行緒；依計劃的測試命令不加此額外限制，並固定 CPU 執行緒後重跑，上列結果通過。

執行偏差：首次尚未讀到計劃限制時，唯讀探索曾包含一次 `git status --short`，且該批命令未加 stdin 重導向；之後不再呼叫 Git，Git 狀態未變更。未觸碰根目錄的 `it`、`log`，未修改既有模型程式、設定或子圖。

## 參考文獻與讀取範圍

| 代碼 | 來源 | 本次實際讀取範圍／限制 |
| --- | --- | --- |
| A | [MaleCNS v1.0 原始 annotations／NT／weights；實際本地表與上列 SHA-256。官方來源：](https://male-cns.janelia.org/download/) | 親查表格；原始資料唯讀。 |
| JSON | [本次可重現查詢輸出](../artifacts/data_execution/bio_verification.json) | 本地計數、逐細胞標籤、權重、路徑與檢查。 |
| T | [Task et al. (2022; 網頁列有 2023 修訂), Chemoreceptor co-expression, eLife 11:e72599](https://elifesciences.org/articles/72599) | 已讀 Table 3 的 DM1／VA2 列與正文。DOI 10.7554/eLife.72599。 |
| G | [Gaudry et al. (2013), Asymmetric neurotransmitter release, Nature 493:424–428](https://pmc.ncbi.nlm.nih.gov/articles/PMC3590906/) | 已讀 Results／Fig. 2–4；DOI 10.1038/nature11747。 |
| O | [Olfactory Receptor Responses to Pure Odorants in Drosophila melanogaster (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11891828/) | 已讀 Or92a Results／Fig. 3；只列本次使用的配體結論。 |
| H | [Hindmarsh Sten et al. (2021), Sexual arousal gates visual processing during Drosophila courtship](https://pmc.ncbi.nlm.nih.gov/articles/PMC8973426/) | 已讀工具傳回的原論文 Results：LC10a gain／單側沉默段；直接重開頁面有 reCAPTCHA。DOI 10.1038/s41586-021-03714-w。 |
| R18 | [Ribeiro et al. (2018), Visual Projection Neurons Mediating Directed Courtship](https://pubmed.ncbi.nlm.nih.gov/30033367/) | 只讀作者摘要；正文與 LC10a 亞型細節未核實。DOI 10.1016/j.cell.2018.06.020。 |
| S | [Schretter et al. (2024), Social state alters vision using three circuit mechanisms](https://www.nature.com/articles/s41586-024-08255-6) | 已讀 Results／Fig. 1–3，LC9、LC10a／LC11 感覺與行為結論；未把雌蠅實驗當成本模型驗證。 |
| LD | [Ribeiro Lab 作者研究頁：Neural circuits and computations](https://www.med.lmu.de/imp/en/research-groups/ribeiro-lab-neurogenetics-and-behavior/neural-circuits-and-computations-processing-of-visual-information/) | 僅作者研究自述；不替代同行評審實驗原文。 |
| LD-P | [Ribeiro et al. (2026), A multi-input optic glomerulus mediates opposing behavioral responses](https://doi.org/10.64898/2026.01.26.701771) | 正文未核實：bioRxiv 存取回覆 403／不可用；不引用二手摘要當已讀原文。 |
| L11 | [Keleş & Frye (2017), Object-detecting neurons in Drosophila](https://pmc.ncbi.nlm.nih.gov/articles/PMC5340600/) | 已取得 Results／Fig. 1 的形態與物體調諧原文。DOI 10.1016/j.cub.2017.01.012。 |
| K | [Klapoetke et al. (2017), Ultra-selective looming detection from radial motion opponency](https://pmc.ncbi.nlm.nih.gov/articles/PMC7457385/) | 已取得 Results／PubMed Fig. 2 圖說，非僅二手綜述；PMC 重開有 reCAPTCHA。DOI 10.1038/nature24626。 |
| R | [Rayshubskiy et al. (2025), Neural circuit mechanisms for steering control, eLife 13:RP102230](https://elifesciences.org/articles/102230) | 已讀 Results、Fig. 1／8、Discussion；DOI 10.7554/eLife.102230。 |
| R20 | [Rayshubskiy et al. (2020), 同題預印本 v2](https://doi.org/10.1101/2020.04.04.024703v2) | 2025 原文參考書目明列；本次未逐版讀取 2020 原文，不作獨立複現證據。 |
| Y | [Yang et al. (2024), Fine-grained descending control of steering in walking Drosophila, Cell 187:6290–6308.e27](https://pmc.ncbi.nlm.nih.gov/articles/PMC12778575/) | 2024 發表版的 PMC 作者接受稿，非 2023 預印本；已讀 Results、Fig. 2／4／7、Methods。DOI 10.1016/j.cell.2024.08.033。 |
| N | [Namiki et al. (2018), The functional organization of descending sensory-motor pathways, eLife 7:e34272](https://elifesciences.org/articles/34272) | 已讀命名正文與補圖圖說；原始 XLSX 的指定型別跨側欄未核實。 |
| B | [Bidaye et al. (2020), Two brain pathways initiate distinct forward walking programs, Neuron 108:469–485.e8](https://pmc.ncbi.nlm.nih.gov/articles/PMC9435592/) | 已讀 DNp09=P9、Fig. 2 與 P9 沉默結果。DOI 10.1016/j.neuron.2020.07.032。 |
| MC | [Berg et al., Sexual dimorphism in the complete connectome of the Drosophila male central nervous system](https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/) | 實讀 2025-10-30 v2 Methods：Neurotransmitter prediction／Assignment。發表版為 Cell (2026), DOI 10.1016/j.cell.2026.08.015，未逐項核對新版方法。 |
| LW | [Liu & Wilson (2013), Glutamate is an inhibitory neurotransmitter in the Drosophila olfactory system](https://pmc.ncbi.nlm.nih.gov/articles/PMC3690841/) | 已讀 GluClα knockdown Results 與 Discussion。DOI 10.1073/pnas.1220560110。 |
| GT | [原作者維護的 drosophila_neurotransmitters／gt_data.csv](https://github.com/flyconnectome/drosophila_neurotransmitters/blob/a9417412c8a70fcc9f80a65ca5bc6064eba07be3/gt_data.csv) | URL 固定 commit，已核對與上述 blob SHA-1 相符。親讀指定型別列與 README；型別映射、confidence 是作者整理，不冒充本次讀過全部原實驗。 |
| D20 | [Davis et al. (2020), A genetic, genomic, and computational resource, eLife 9:e50901](https://elifesciences.org/articles/50901) | 已讀 TAPIN／NT 方法正文；關鍵 LC 原始逐亞型附表未核實。 |
| SH | [Shang et al. (2007), Excitatory local circuits and their implications, Cell 128:601–612](https://pmc.ncbi.nlm.nih.gov/articles/PMC2866183/) | 作者摘要／原文搜尋可讀 LN 群體結果；本次未核實新 MaleCNS 型別的原始逐列映射。DOI 10.1016/j.cell.2006.12.034。 |
| TN | [Tanaka et al. (2012), Organization of antennal lobe-associated neurons](https://doi.org/10.1002/cne.23142) | 作者 NT 整理所引；未讀逐型原始免疫表，不標直接實驗已核實。 |
| C | [Cheong et al., eLife reviewed preprint v1 (2024), Transforming descending input into behavior](https://elifesciences.org/reviewed-preprints/96084v1) | 已讀 FISH 方法、Fig. 4 補圖說與 DNp31 Fig. 12；NT 逐型值主要引用作者 GT 表，未重做圖像定量。DOI 10.7554/eLife.96084.1。 |
| NC4 | [Stürner et al. DN identification 補充檔 4](https://github.com/flyconnectome/2023neckconnective/blob/94637d7d82920234ef0bfdbf383c000e74a8b45a/Supplemental_files/Supplemental_file4_DN_identification.tsv) | 親查 DNp09，DNb05 引既有報告。 |
| NC5 | [同文 FAFB DN 補充檔 5](https://github.com/flyconnectome/2023neckconnective/blob/94637d7d82920234ef0bfdbf383c000e74a8b45a/Supplemental_files/Supplemental_file5_FAFB_DNs.tsv) | 親查同名 DNp09；不是即時 FlyWire API。 |
| NC7 | [同文 MANC DN 補充檔 7](https://github.com/flyconnectome/2023neckconnective/blob/94637d7d82920234ef0bfdbf383c000e74a8b45a/Supplemental_files/Supplemental_file7_MANC_DNs.tsv) | 親查同名 DNp09；不是即時 MANC API。 |
| M16 | [Matsuo et al. (2016), J. Comp. Neurol. 的 AMMC 下行神經元研究](https://doi.org/10.1002/cne.23955) | 僅沿用既有 DNb05 報告讀取的原文轉載與命名結論，本次未重查。 |
| EL | [MaleCNS Explorer DNb05_L](https://github.com/reiserlab/celltype-explorer-drosophila-male-cns/blob/789cc6c105798ce2fd70ba85dab394f90899616b/types/DNb05_L.html) | 既有報告固定版本 ROI 證據，本次未重取。 |
| ER | [MaleCNS Explorer DNb05_R](https://github.com/reiserlab/celltype-explorer-drosophila-male-cns/blob/789cc6c105798ce2fd70ba85dab394f90899616b/types/DNb05_R.html) | 同上。 |
| SIM | [模擬 env.py](../python/flytrack_sim/env.py) | 親讀氣味來源與 yaw 更新。 |
| PLUME | [濃度場 plume.py](../python/flytrack_core/world/plume.py) | 親讀純量濃度模型。 |
| ENC | [SensoryEncoder／模型 policy.py](../python/flytrack_core/model/policy.py) | 親讀 ORN Hill 與視覺池化。 |
| READ | [MotorReadout／模型 policy.py](../python/flytrack_core/model/policy.py) | 親讀左右差、前進池化與 DNb05 extra readout。 |
| TAB | [connectome_extract/tables.py](../python/connectome_extract/tables.py) | 親讀 somaSide／rootSide 優先序與 NT 欄位規則。 |
| EXTRACT | [connectome_extract/graph.py](../python/connectome_extract/graph.py) | 親讀 NT_SIGNS、assign_signs、候選交集與匯流定義。 |

[A]: https://male-cns.janelia.org/download/
[JSON]: ../artifacts/data_execution/bio_verification.json
[T]: https://elifesciences.org/articles/72599
[G]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3590906/
[O]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11891828/
[H]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8973426/
[R18]: https://pubmed.ncbi.nlm.nih.gov/30033367/
[S]: https://www.nature.com/articles/s41586-024-08255-6
[LD]: https://www.med.lmu.de/imp/en/research-groups/ribeiro-lab-neurogenetics-and-behavior/neural-circuits-and-computations-processing-of-visual-information/
[LD-P]: https://doi.org/10.64898/2026.01.26.701771
[L11]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5340600/
[K]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7457385/
[R]: https://elifesciences.org/articles/102230
[R20]: https://doi.org/10.1101/2020.04.04.024703v2
[Y]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12778575/
[N]: https://elifesciences.org/articles/34272
[B]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9435592/
[MC]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/
[LW]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3690841/
[GT]: https://github.com/flyconnectome/drosophila_neurotransmitters/blob/a9417412c8a70fcc9f80a65ca5bc6064eba07be3/gt_data.csv
[D20]: https://elifesciences.org/articles/50901
[SH]: https://pmc.ncbi.nlm.nih.gov/articles/PMC2866183/
[TN]: https://doi.org/10.1002/cne.23142
[C]: https://elifesciences.org/reviewed-preprints/96084v1
[NC4]: https://github.com/flyconnectome/2023neckconnective/blob/94637d7d82920234ef0bfdbf383c000e74a8b45a/Supplemental_files/Supplemental_file4_DN_identification.tsv
[NC5]: https://github.com/flyconnectome/2023neckconnective/blob/94637d7d82920234ef0bfdbf383c000e74a8b45a/Supplemental_files/Supplemental_file5_FAFB_DNs.tsv
[NC7]: https://github.com/flyconnectome/2023neckconnective/blob/94637d7d82920234ef0bfdbf383c000e74a8b45a/Supplemental_files/Supplemental_file7_MANC_DNs.tsv
[M16]: https://doi.org/10.1002/cne.23955
[EL]: https://github.com/reiserlab/celltype-explorer-drosophila-male-cns/blob/789cc6c105798ce2fd70ba85dab394f90899616b/types/DNb05_L.html
[ER]: https://github.com/reiserlab/celltype-explorer-drosophila-male-cns/blob/789cc6c105798ce2fd70ba85dab394f90899616b/types/DNb05_R.html
[SIM]: ../python/flytrack_sim/env.py
[PLUME]: ../python/flytrack_core/world/plume.py
[ENC]: ../python/flytrack_core/model/policy.py
[READ]: ../python/flytrack_core/model/policy.py
[TAB]: ../python/connectome_extract/tables.py
[EXTRACT]: ../python/connectome_extract/graph.py
