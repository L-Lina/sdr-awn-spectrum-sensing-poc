# Satellite-like 應用場景與 Latency Requirement 定義

## 一、研究目的

本文件的目的**不是**把目前系統實測的 processing latency 硬套進任何衛星通訊標準，而是回答以下八個問題，作為專案 close 前定義 satellite-like application scenario 的正式依據：

1. 衛星通訊常見的鏈路時間尺度是多少？
2. LEO／MEO／GEO 的 propagation delay 與 end-to-end latency 各是多少？
3. 這些 latency 指標代表什麼，不能代表什麼？
4. 我們目前的 Clean AMC、optimized FGSM pipeline、optimized PGD pipeline 各自落在哪些 processing budget？
5. 哪一種 satellite-like application scenario 最適合目前專案？
6. 我們的 processing module 應該放在 receiver side digital baseband／gateway／ground station／onboard processor 哪一種位置最合理？
7. Attack 的 threat model 應該如何定義，才不會誤稱成「即時 OTA attacker 一定來得及」？
8. 未來若要做 satellite-like simulation，最少需要模擬哪些物理因素？

本輪**只做研究、資料核實與正式研究文件**，不修改 formal pipeline、不新增 simulator、不新增 attack、不跑大型實驗、不碰 RadioML2018 訓練、不做 DVB-S2/S2X modem 實作。

## 二、為什麼需要 Application Latency Validation

`docs/research/PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md` 已經以嚴謹的 mean／median／p95 方法量測了目前系統的 CPU processing latency（clean pipeline、FGSM／PGD 攻擊生成、Top-K 防禦、端到端 Scenario A-E），但那份文件本身**不涉及任何特定應用場景**，只回答「系統本身有多快」。若要進一步論證「這個速度對某個實際應用是否有意義」，就必須先有一個具體、有標準依據的 latency requirement 可以對照——否則任何「我們的 pipeline 夠快」的陳述都缺乏比較基準，也容易被誤解為對某個未經定義的即時場景做出保證。本文件的角色正是提供這個對照基準，並且明確劃出「我們量到的東西」與「應用場景要求的東西」之間的界線。

## 三、衛星通訊時間尺度（Satellite Communication Time-Scale）

在檢視正式標準數字之前，先建立量級對照：地面蜂巢式網路（4G/5G 地面基地台）的往返傳播延遲通常在次毫秒到數毫秒等級（數十公里內），而衛星鏈路依軌道高度不同，傳播延遲從數毫秒（LEO）到數百毫秒（GEO）不等，比地面網路高出 1-2 個數量級。這個量級差異是本文件成立的前提：**衛星應用本身工作在數十到數百毫秒的時間尺度，因此毫秒等級（millisecond-level）的 baseband processing，相較於 Wi-Fi PHY/MAC 這種微秒到次毫秒等級的即時路徑，有更寬鬆、也更值得研究的可行空間**——但這僅代表時間尺度上的相對關係，不代表任何具體的 computation deadline，實際可分配的 processing budget 仍取決於 receiver architecture 與應用本身（詳見第四節）。

## 四、3GPP NTN Latency（一手來源核實）

### 4.1　來源

本節數字直接取自 ETSI 官方發布的 3GPP TS 22.261 PDF 全文（非二手轉述），核對了兩個版本：

- **ETSI TS 122 261 V19.12.0（2025-10）**，對應 3GPP TS 22.261 version 19.12.0 Release 19，"5G; Service requirements for the 5G system"。
  來源：https://www.etsi.org/deliver/etsi_ts/122200_122299/122261/19.12.00_60/ts_122261v191200p.pdf
- **ETSI TS 122 261 V18.13.0（2024-05）**，Release 18，同一份規格較早的版本，用於核對版本間數值差異。
  來源：https://www.etsi.org/deliver/etsi_ts/122200_122299/122261/18.13.00_60/ts_122261v181300p.pdf

兩者皆為 ETSI 官方 deliver 目錄下的正式發布 PDF，非部落格或二手教學文。

### 4.2　Table 7.4.1-1：UE 到衛星的 propagation delay（clause 7.4.1）

**現行版本（V19.12.0 Release 19，與 V18.17.0 Release 18 數字相同，見 4.4 節版本差異說明）：**

| Orbit | UE→satellite propagation min (ms) | UE→satellite propagation max (ms) | UE→ground max propagation delay (ms) |
|---|---|---|---|
| LEO | 1 | 13 | 26 |
| MEO | 24 | 99 | 198 |
| GEO | 120 | 136 | 272 |

規格原文附註：LEO 的延遲範圍是以仰角 90° 對應 300 km 高度、仰角 10° 對應 1 500 km 高度計算；MEO 是以仰角 90° 對應 7 000 km、仰角 10° 對應 25 000 km 計算；GEO 是以仰角 90° 到 10° 對應 35 786 km 計算。「UE to ground max propagation delay」是透過衛星鏈路的 UE 到地面站延遲，不含 inter-satellite link。

### 4.3　Table 7.4.2-1：Supported end-to-end latency（clause 7.4.2）

| Orbit | Supported end-to-end latency | 組成 |
|---|---|---|
| GEO | 285 ms | 衛星單向延遲 + 假設 5 ms 網路延遲 |
| MEO | 95 ms | 衛星單向延遲 + 假設 5 ms 網路延遲 |
| LEO | 35 ms | 衛星單向延遲 + 假設 5 ms 網路延遲 |

此表在 Release 18（V18.13.0 起）與 Release 19（V19.12.0）之間數字**完全一致**，未受 4.4 節的版本修正影響。

### 4.4　版本差異（明確標記，不得混用）

比對 V18.13.0（2024-05，修正前）與 V18.17.0／V19.12.0（修正後）的 Table 7.4.1-1，兩者數字不同：

| Orbit | 修正前（V18.13.0，2024-05） UE→satellite min/max (ms) | 修正前 UE→ground max (ms) | 修正後（V18.17.0／V19.12.0） UE→satellite min/max (ms) | 修正後 UE→ground max (ms) |
|---|---|---|---|---|
| LEO | 3 / 15 | 30 | 1 / 13 | 26 |
| MEO | 27 / 43 | 90 | 24 / 99 | 198 |
| GEO | 120 / 140 | 280 | 120 / 136 | 272 |

版本歷史記錄明確列出這是一次正式修正：change history 中記載「2024-12 SA#106 SP-241760 0819/0820 2 A Correction on the propagation delay via satellite」，分別套用到 Release 18（18.16.0）與 Release 19（19.9.0）。修正後版本額外補上了仰角/軌道高度假設的計算依據（見 4.2 節附註），且 MEO 的 UE-to-ground 上限從 90 ms 大幅提高到 198 ms——這是一次實質內容修正，不是編輯排版微調。**本文件後續所有對照與計算，一律使用修正後（V19.12.0／V18.17.0）的數字，不與修正前的數字混用**；Table 7.4.2-1 的 end-to-end latency 需求（285／95／35 ms）本身未受此次修正影響，兩個版本一致。

## 五、Latency Interpretation：不能把 Propagation Delay 當 Computation Deadline

3GPP TS 22.261 的「supported end-to-end latency」是**系統層級**的服務需求數字，代表一個完整 5G 系統（含無線協定、排程、網路核心）在該軌道類型下應該能夠支援的端到端延遲上限，其中已經包含了假設的 5 ms 網路延遲，但**不包含、也未指定**任何特定應用（例如 AMC、頻譜感測、對抗式攻擊生成）在接收端可以使用的 computation budget。必須明確區分以下五個不同層次的延遲，不得互相替代：

1. **Propagation delay**：訊號在自由空間中傳播所需的實體時間，由光速與距離決定，是第四節表格的數字。
2. **Network latency**：協定層（排程、重傳、佇列）造成的延遲，TS 22.261 假設為固定 5 ms。
3. **Processing latency**：接收端數位訊號處理（本文件之前各章節量測的對象：sensing、AMC 推論、攻擊生成、Top-K）所花費的計算時間，這是唯一與 CPU/GPU 硬體效能直接相關的部分。
4. **Application latency budget**：特定應用（例如互動式語音、串流、監控告警）願意容忍的總延遲，由應用需求方定義，通常包含以上所有層次再加上使用者體感需求。
5. **Scheduling／framing delay**：無線協定的 frame/slot 結構本身造成的排隊與對齊延遲，與底層 numerology 及排程演算法有關。

因此，**文件中不得寫「GEO 有 285 ms，所以 attack 可以算 285 ms」**這類推論——285 ms 是系統層級的 propagation+network 延遲需求，不是留給接收端任何特定演算法的計算時間。正確的問題應該是：「衛星應用本身工作在數十到數百 ms 的時間尺度，因此 millisecond-level baseband processing 相較 Wi-Fi PHY/MAC 更有可行研究空間；實際可分配的 processing budget 仍取決於 receiver architecture 與 application」——第七節即依此原則進行 processing-budget 對照，性質是「量級參照」，不是「deadline 證明」。

## 六、DVB-S2／DVB-S2X 應用與 Modulation Family（一手來源核實）

### 6.1　來源

- **ETSI EN 302 307-1 V1.4.1（2014-11）**："Digital Video Broadcasting (DVB); Second generation framing structure, channel coding and modulation systems for Broadcasting, Interactive Services, News Gathering and other broadband satellite applications; Part 1: DVB-S2"。
  來源：https://www.etsi.org/deliver/etsi_en/302300_302399/30230701/01.04.01_60/en_30230701v010401p.pdf
- **ETSI EN 302 307-2 V1.4.1（2024-08）**：同系列 Part 2，DVB-S2X 延伸規格。
  來源：https://www.etsi.org/deliver/etsi_en/302300_302399/30230702/01.04.01_60/en_30230702v010401p.pdf

### 6.2　DVB-S2 應用領域與調變

依 EN 302 307-1 clause 1（Scope）與內文，DVB-S2 定義了 4 種星座（QPSK、8PSK、16APSK、32APSK），頻譜效率範圍 2–5 bit/s/Hz，code rate 1/4 至 9/10（LDPC+BCH FEC），並針對以下四類應用最佳化：

| 應用類別 | 說明 |
|---|---|
| **Broadcast Services (BS)** | 數位多頻道電視／HDTV 廣播，用於 FSS／BSS 頻段的一次與二次分發，Direct-To-Home（DTH）服務 |
| **Interactive Services (IS)** | 互動式資料服務，含網際網路存取，正向鏈路取代 DVB-S 用於互動系統 |
| **Digital TV Contribution / Satellite News Gathering (DTVC/DSNG)** | 點對點或點對多點的新聞採集與電視貢獻傳輸，非供一般大眾直接接收 |
| **Data content distribution/trunking and other professional applications (PS)** | 專業資料傳輸、trunking，主要為點對點或點對多點 |

QPSK／8PSK 因其準恆包絡特性，適合在衛星功率放大器（HPA）飽和點附近以單載波方式運作；16APSK／32APSK 需要更高功率餘裕，並可透過預失真技術在接近 HPA 飽和點運作（此為 6.3 節「非線性放大器效應」在第九節被列為 OPTIONAL 而非 MUST 的直接依據）。

### 6.3　DVB-S2X 延伸與應用

EN 302 307-2（S2X）於 2014 年核准為 DVB-S2 的非回溯相容延伸，除了沿用 S2 的核心應用領域外，新增了 **VL-SNR（Very Low Signal-to-Noise Ratio）** 應用領域，可在 carrier-to-noise 比低至 -10 dB 的條件下運作，明確服務於：

- 機載服務（商務噴射機）
- 海事應用
- 民航網際網路存取
- 更高頻段或熱帶地區的 VSAT 終端
- 記者與其他專業人員使用的可攜式小型終端

同時 S2X 也提供高容量、高效率的傳輸模式，服務於極高 C/N 比的專業鏈路。技術手段包括更細緻的 MODCOD 步階、更陡峭的 roll-off 濾波、多波束（beam hopping）支援與 transponder bonding。

### 6.4　與本專案的關係

本專案目前使用的 RadioML2016.10a 資料集（詳見 6.5 節）採用的是 11 種通用數位/類比調變（8PSK、AM-DSB、AM-SSB、BPSK、CPFSK、GFSK、PAM4、QAM16、QAM64、QPSK、WBFM），其中 **QPSK、8PSK 與 DVB-S2/S2X 的核心 MODCOD 直接重疊**，可作為「衛星廣播/貢獻鏈路」情境下 AMC 任務的合理調變子集；16APSK／32APSK 目前不在 RadioML2016.10a 的調變清單中，若未來要更貼近 DVB-S2/S2X 高階 MODCOD 場景，需要額外資料或訓練，本輪不處理。**本文件僅回答「可以採哪些 modulation family 讓場景具有標準依據」，不實作完整 DVB stack，也不修改現有的調變清單或訓練資料。**

### 6.5　DeepSig RadioML 官方資料頁核實

來源：https://www.deepsig.ai/datasets（DeepSig 官方資料集頁面）。RadioML2016.10a 由 DeepSig 於第 6 屆 GNU Radio Conference 發布，含 11 種調變（8 種數位、3 種類比），使用 GNU Radio 產生，是較早期 2016.04C 資料集的「更乾淨、更正規化」版本，原始產生程式位於 https://github.com/radioML/dataset（DeepSig 官方頁面註明此程式已不再積極維護）。DeepSig 官方頁面本身將這類資料集定位為「歷史性」（historical）資料集，並建議研究者針對目前的研究改用真實資料——這個定位本身是本文件第九節與第十四節「限制」討論的重要依據：現有資料集的通道模擬（AWGN、LO offset 等泛用 RF 損傷）並非針對衛星通道設計，不能直接宣稱等同衛星通道實測。

## 七、我們目前的實測 Processing Latency 對照

以下數字直接讀自既有正式結果，**本輪未重跑任何 benchmark**：

- 來源檔案：`results/end_to_end_latency_20260818T062625Z/end_to_end_latency_summary.csv`、`processing_budget_table.csv`
- 對應章節：`docs/research/PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md` 第十六節

| Scenario | median (ms) | p95 (ms) |
|---|---|---|
| Clean AMC（Scenario A，baseline 執行緒） | 2.843 | 7.528 |
| Optimized FGSM End-to-End（Scenario C，optimized） | 3.813 | 4.192 |
| Optimized deterministic PGD End-to-End（Scenario D_det，optimized，`random_start=False`） | 7.725 | 8.664 |

（原始 CSV 精確值：Clean AMC median=2.8428735、p95=7.527741…；若與四捨五入後的顯示值有差異，以 CSV 原始值為準。）

## 八、Processing Budget Comparison（對照 3GPP NTN 時間尺度）

| Budget | Clean AMC median fits | Clean AMC p95 fits | FGSM optimized median fits | FGSM optimized p95 fits | PGD optimized median fits | PGD optimized p95 fits |
|---|---|---|---|---|---|---|
| 5 ms | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ |
| 10 ms | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 20 ms | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 35 ms | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 50 ms | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 100 ms | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 250 ms | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**這只是 processing budget comparison，不是任何特定應用場景的 deadline 證明**：三個 scenario 的 median 與 p95 都落在遠低於 LEO 的 35 ms、MEO 的 95 ms、GEO 的 285 ms supported end-to-end latency（第四節）的量級；PGD 在最嚴格的 5 ms 參照下median／p95 皆不符合，但在 10 ms 以上的參照下三者皆符合。這個結果只能解讀為「相較 3GPP NTN 端到端時間尺度的 processing-budget reference」，**不得稱為「LEO computation deadline」**，因為（如第五節所述）35 ms 是系統層級的 propagation+network 延遲需求，不是留給我們這個特定 AMC/attack pipeline 的計算配額；實際上能分配給接收端 baseband 處理的時間，會因 receiver architecture、其他並行任務、以及應用本身的容忍度而遠小於這個系統層級數字。

## 九、Deployment Position Analysis

比較四種可能的部署位置，各自對 raw IQ 可得性、Spectrum Sensing／AMC 可行性、運算資源、latency 壓力、threat model、即時 RF 反應需求進行分析：

| 位置 | Raw IQ 可得性 | Sensing/AMC 可行性 | CPU/GPU/NPU 可用性 | Latency 壓力 | Threat model 一致性 | 即時 RF 反應需求 |
|---|---|---|---|---|---|---|
| **A. UE／衛星終端接收端** | 高（基頻鏈路末端即為 IQ） | 高，UE 本身即需訊號同步/分類 | 受限（成本/功耗/散熱），行動 SoC 漸有 NPU 但仍有限 | 中至高（依服務類型） | 高，與 A0（接收端數位白箱基準）自然一致 | 否（分類本身非逐符元即時路徑） |
| **B. Ground station receiver** | 高，SDR 寬頻擷取常見 | 高，監測/識別為既有應用 | 高，地面基礎設施運算資源充裕 | 低至中 | 高，與 A0 一致 | 否，多為近即時監測 |
| **C. Gateway baseband processing** | 部分，feeder link 多為多使用者聚合鏈路，非單一 IQ | 中，需處理寬頻多工訊號，複雜度較高 | 高，地面基礎設施 | 中至高，若在關鍵路徑上會直接消耗系統 E2E latency 預算 | 中，gateway 為受保護核心基礎設施，實際可及性假設較強（偏向 A1/A2） | 系統整體需符合嚴格 E2E 預算，但 AMC 本身非必要 |
| **D. Onboard satellite processor** | 中，再生式（regenerative）payload 原則上可及但受限 | 中，新一代再生式 payload 漸增但仍屬少數 | 低，太空級運算受限於功耗/散熱/抗輻射要求 | 高，是系統中對延遲最敏感的位置 | 低，x86 桌面級 CPU benchmark 對太空級硬體能力沒有直接代表性，容易誤導成「已驗證機載即時防禦」 | 是，再生式處理通常需符合嚴格逐 frame 預算 |

**推薦的 reference scenario：receiver-side／ground-side digital IQ processing**（即位置 A 與 B 的組合：地面接收端／地面站的頻譜監測與 AMC），理由：

1. Raw IQ 可得性高，與目前 pipeline（真實 IQ → sensing → AMC → 攻擊 → 防禦）直接吻合，不需改變現有架構。
2. 運算資源不受限，不需要對「x86 桌面 CPU 代表衛星硬體」做任何過度延伸的假設。
3. Threat model 與目前的 A0（接收端數位白箱基準，見第十節）自然一致，不需要额外假設實體可及性極高的攻擊者。
4. 完全可離線模擬（現有 RadioML + 真實 AWN + 真實 torchattacks 已經是這個位置的合理離線代理），不需要額外硬體。

位置 C（gateway）與 D（onboard）皆存在較高的實作複雜度與 threat-model 不一致風險（尤其 D 容易被誤解為「已驗證機載即時防禦能力」），本輪不作為主要 reference scenario。

## 十、Threat-Model Consistency（A0／A1／A2）

必須明確區分三種攻擊者假設，避免混淆：

| 假設 | 定義 |
|---|---|
| **A0** | Receiver-side digital／white-box IQ attack benchmark——攻擊者對接收端已解調的數位 IQ 張量有白箱存取權限，直接在數位域生成擾動並評估效果。**這是本專案目前 FGSM/PGD/CW 等攻擊實驗的實際假設**。 |
| **A1** | Independent RF transmitter——攻擊者是一個獨立的射頻發射端，透過實際無線通道注入訊號或干擾，目標接收端只能觀察到疊加後的合成波形。 |
| **A2** | Information-limited／query-based attacker——攻擊者對目標模型只有有限資訊（例如僅能查詢輸出、不知道模型參數），需要透過查詢或側通道推斷攻擊方向。 |

**本輪 application scenario 的定義不得把 A0 的 CPU attack generation latency，直接解釋為 A1 OTA attacker 必須在訊號飛行途中即時計算 perturbation 的時間。** 這是兩個性質完全不同的問題：

- A0 latency（本文件第七節與 `PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md` 全篇量測的對象）是一個**數位訊號處理 benchmark**：給定一個已經存在於記憶體中的 IQ 張量，計算出對抗擾動並疊加，全程沒有涉及真實無線通道、沒有時序同步問題、沒有頻率飄移。
- 若未來要評估 A1（真正的 over-the-air 即時攻擊者），除了攻擊演算法本身的計算時間外，還需要額外考慮：
  - **波形知識（waveform knowledge）**：攻擊者是否事先知道目標訊號的調變、符元速率、成型濾波器
  - **預測視窗（prediction horizon）**：攻擊者必須在目標訊號抵達接收端「之前」完成擾動波形的產生與發射，這個時間窗口由實際幾何與硬體鏈路延遲決定，不是攻擊演算法的計算時間
  - **同步（synchronization）**：攻擊者的發射時序需要與目標訊號對齊
  - **Tx/Rx delay**：攻擊者自身發射鏈路與目標接收鏈路各自的硬體延遲
  - **RF chain delay**：類比前端（濾波器、放大器、ADC/DAC）造成的額外延遲
  - **通道狀態（channel state）**：攻擊訊號與目標訊號經過不同（或部分重疊）的通道，疊加效果不等於數位域直接相加
  - **Doppler／CFO**：衛星鏈路的高相對速度造成的頻率飄移，會直接影響攻擊波形是否能在目標接收端正確疊加產生預期效果
  - **timing uncertainty**：實際系統中時序估計本身有誤差，不是理想化的已知值

因此，本文件與既有效能文件中的所有 speedup／processing-budget 陳述，**都只能代表 A0 的數位處理效能**，不構成對 A1 OTA 即時攻擊可行性的任何驗證或保證。

## 十一、Satellite-like Channel 最小物理因素（MUST／SHOULD／OPTIONAL）

本節**只整理**未來若要建置 satellite-like channel simulator，最少應包含哪些物理因素，**不在本輪實作**。分類依據：是否為衛星通道的核心差異來源、是否直接影響 AMC 或 Spectrum Sensing、是否與 LEO/MEO/GEO 軌道選擇直接相關。

| 因素 | 分類 | 是否影響 AMC | 是否影響 Spectrum Sensing | 與軌道的關係 | 是否需要真實軌道幾何 |
|---|---|---|---|---|---|
| AWGN | **MUST** | 直接影響（既有 SNR 掃描的基礎） | 直接影響（門檻校準基礎） | 所有軌道皆適用，非軌道專屬 | 否 |
| Attenuation／path-loss abstraction（純量等效 SNR 調整，非完整幾何傳播模型） | **MUST** | 間接影響（透過有效 SNR） | 間接影響（透過有效 SNR） | 直接相關——GEO（35 786 km）與 LEO（300–1 500 km）的鏈路預算差異是軌道選擇的核心後果 | 否，純量抽象即可 |
| Propagation delay metadata（僅標記情境使用的軌道對應延遲數值，供場景敘事使用，不對 128-sample 短擷取窗做逐樣本延遲模擬） | **MUST**（於情境文件層級）| 不影響（延遲不改變波形本身） | 不影響 | 直接相關，數值取自第四節 Table 7.4.1-1 | 否 |
| CFO（載波頻率偏移） | **SHOULD** | 顯著影響（PSK/QAM/APSK 星座旋轉） | 較小影響（能量偵測對相位不敏感） | 與軌道相對速度有關，LEO/MEO 較大 | 否，可用經驗範圍近似 |
| Doppler shift（動態、隨時間變化） | **SHOULD** | 顯著影響，且比靜態 CFO 更複雜（隨時間變化） | 較小影響 | 高度相關，是 LEO/MEO 相對於 GEO 的關鍵差異（TS 22.261 clause 7.4.1 已指出 LEO/MEO 因高速移動造成 moving cell pattern 與較大 Doppler 變化） | 部分需要（至少需要相對速度量級假設） |
| Timing offset（符元/取樣時序偏移） | **SHOULD** | 中等影響（影響 segmentation／輸入對齊） | 較小影響 | 通用 RF 鏈路效應，非軌道專屬 | 否 |
| Sample-rate offset（Tx/Rx 時脈漂移） | **OPTIONAL** | 較小影響（128-sample 短擷取窗下累積誤差有限） | 極小影響 | 通用效應，長時間擷取才顯著 | 否 |
| Amplitude scaling（接收增益變化） | **OPTIONAL** | 較小影響（現有 `apply_awn_preprocess` 正規化策略已部分吸收） | 較小影響 | 通用效應 | 否 |
| Non-linear amplifier（HPA 非線性，接近飽和點運作） | **OPTIONAL** | 對 APSK 類調變有影響，但 RadioML2016.10a 本身不含 DVB-S2/S2X 波形 | 不直接影響 | 與 DVB-S2/S2X 的 16/32APSK 預失真設計直接相關（見 6.2 節），但與目前調變清單關聯低 | 否 |

**避免過度複雜**：MUST 項目（AWGN、path-loss 抽象、propagation delay metadata）已經與現有 RadioML 產生流程（`embed_sample_in_noise` 等）的抽象層級相容，不需要引入完整軌道幾何或即時衛星星曆計算；SHOULD 項目（CFO、Doppler、timing offset）是讓場景更貼近真實 LEO/MEO 特性的下一步，但可以用經驗參數範圍近似，不必是精確物理模型；OPTIONAL 項目除非未來場景明確需要（例如真正實作 DVB-S2/S2X 波形），否則不建議投入。

## 十二、Candidate Scenarios 與評分

比較四個候選 application scenario：

| Scenario | Raw IQ availability | Latency compatibility | Implementation complexity | Relevance to current pipeline | Need for hardware | Threat-model consistency | Ability to simulate offline |
|---|---|---|---|---|---|---|---|
| 1. LEO ground-terminal receiver | 高 | 高（35 ms 預算使 ms 級 processing 的比較最有意義） | 中（需要 Doppler/CFO 才算完整） | 高（與現有 pipeline 直接吻合） | 低 | 高（自然對應 A0） | 高 |
| 2. GEO ground-station／gateway monitoring | 高 | 中（285 ms 預算寬鬆，ms 級處理的比較意義較低） | 中 | 中 | 低 | 高 | 高 |
| 3. Generic satellite spectrum monitoring（不綁定特定軌道） | 高 | 中（可彈性描述為次百毫秒等級偵測回應） | 低（不需承諾精確軌道參數） | 高（與既有頻譜感測威脅模型文件框架一致） | 低 | 高 | 高 |
| 4. Onboard signal monitoring | 低（實務上難以取得） | 中至高（理論上最關鍵，但無法用桌面 CPU 代表太空硬體） | 高 | 低（目前無機載硬體建模） | 高 | 低（容易誤稱為已驗證機載即時防禦） | 低 |

## 十三、推薦 Scenario

**主要 scenario：LEO ground-terminal／ground-side receiver spectrum monitoring and AMC**——地面端（終端式或監測式）接收 LEO 下行訊號，執行頻譜感測與自動調變分類。理由：LEO 的 35 ms supported end-to-end latency（第四節）是三種軌道中對延遲最敏感的一類，使得本專案已經量測到的個位數毫秒等級 processing latency（第七節）成為一個有意義的量級對照；同時完全符合第九節建議的 receiver-side／ground-side 部署位置，與 A0 威脅模型天然一致，且可完全用現有工具（真實 IQ、真實 AWN、真實攻擊實作）離線模擬，不需要任何新硬體或新 simulator。

**Backup scenario：GEO ground-station／gateway spectrum monitoring**——技術架構與主要 scenario 相同，僅置換軌道參照點為 GEO（285 ms supported end-to-end latency），作為「延遲寬鬆」情境的對照案例。優點是可以在不改變任何 pipeline 或工具鏈的前提下，僅透過替換第四節引用的軌道數字即可切換，複雜度與威脅模型一致性與主要 scenario 相同。

兩個 scenario 皆**不需要**位置 C（gateway）或 D（onboard）在第九節分析中指出的額外複雜度與威脅模型風險。

## 十四、Simulator Scope Boundary

依第十一節分類，若未來要建置 satellite-like channel simulator（**本輪不實作**），範圍應該是：

- **必須（MUST）涵蓋**：AWGN（沿用現有機制）、以純量方式抽象化的 attenuation／path-loss（區分 LEO/MEO/GEO 三種鏈路預算量級即可，不需完整鏈路預算計算）、propagation delay 作為情境 metadata（用於文件與圖表標註，不是逐樣本 DSP 效果）。
- **應該（SHOULD）在下一階段加入**：CFO 與 Doppler shift（可用依軌道類型設定的經驗範圍參數化，不需要即時軌道幾何運算）、timing offset。
- **可選（OPTIONAL）、暫不列入近期計畫**：sample-rate offset、amplitude scaling（現有正規化流程已部分吸收）、非線性放大器效應（僅在明確要對接 DVB-S2/S2X 波形時才有必要）。

任何 simulator 實作都應該保持「離線、可重現、使用真實建置模組」的既有慣例（與 `experiments/` 目錄下所有正式腳本一致），並在正式導入前經過與本文件相同等級的一手來源核實，不得引入未經查證的物理參數。

## 十五、限制（Limitations）

1. 第四節的 3GPP TS 22.261 數字是**服務需求層級**的規格數字，反映的是 3GPP 標準制定過程中對系統效能的目標設定，不是任何已部署衛星系統的實測數據，也不是任何特定接收機架構的效能保證。
2. 第四節已確認 propagation delay 表格在 2024 年 12 月有一次修正（見 4.4 節），本文件僅核對了 Release 18／19 的修正前後版本，未逐一核對 Release 15-17 的舊版數字是否也有類似或不同的历史差異；若未來需要更完整的版本沿革，需要另外查證。
3. 第六節的 DVB-S2/S2X 分析僅涵蓋應用領域與調變家族層級的對照，不涉及實際 MODCOD 選擇、FEC 參數、或 frame 結構的細節比對，也未查證是否有比 EN 302 307-1 V1.4.1（2014-11）更新的正式版本。
4. 第七節的處理延遲數字取自 24 筆小樣本（4 種調變 × 3 個訊噪比 × 2 個樣本），為既有 `docs/research/PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md` 第十六節的既有結果，其本身的統計限制（樣本數小、單一機器單次執行）已在該文件第 16.8 節說明，本文件不重複驗證，直接沿用其結論。
5. 第九至十三節的 deployment position／scenario 比較是**質性分析**，並非量化評分模型，評分欄位（高/中/低）反映的是研究判斷，不是可重現的計算結果。
6. 第十一節的 MUST/SHOULD/OPTIONAL 通道因素分類是為了未來 simulator 設計的**規劃參考**，本身不構成任何已驗證的通道模型；LEO/MEO 的 Doppler 量級（隨相對速度變化，最大可達數十 kHz 等級，依載波頻率而定）本文件未查證精確數值，需要在實作前另行核實。
7. 第十節的 A0/A1/A2 威脅模型定義是本專案既有威脅模型文件的延伸整理，本文件未重新查證這些定義是否與其他學術文獻的攻擊者分類完全一致。
8. 本文件不構成、也不宣稱「已完成 satellite validation」——所有結論皆為研究層級的場景定義與文獻對照，未經任何實際衛星鏈路或射頻硬體驗證。

## 十六、Primary-Source References

1. ETSI TS 122 261 V19.12.0（2025-10）, 3GPP TS 22.261 version 19.12.0 Release 19, "5G; Service requirements for the 5G system", clause 7.4（Table 7.4.1-1、Table 7.4.2-1）。
   https://www.etsi.org/deliver/etsi_ts/122200_122299/122261/19.12.00_60/ts_122261v191200p.pdf
2. ETSI TS 122 261 V18.13.0（2024-05）, Release 18（版本差異核對用）。
   https://www.etsi.org/deliver/etsi_ts/122200_122299/122261/18.13.00_60/ts_122261v181300p.pdf
3. ETSI TS 122 261 V18.17.0（2025-04）, Release 18（修正後版本核對用）。
   https://www.etsi.org/deliver/etsi_ts/122200_122299/122261/18.17.00_60/ts_122261v181700p.pdf
4. ETSI EN 302 307-1 V1.4.1（2014-11）, "Digital Video Broadcasting (DVB); Second generation framing structure, channel coding and modulation systems for Broadcasting, Interactive Services, News Gathering and other broadband satellite applications; Part 1: DVB-S2", clause 1（Scope）。
   https://www.etsi.org/deliver/etsi_en/302300_302399/30230701/01.04.01_60/en_30230701v010401p.pdf
5. ETSI EN 302 307-2 V1.4.1（2024-08）, 同系列 Part 2: DVB-S2 Extensions (DVB-S2X), Introduction／clause 1（Scope）。
   https://www.etsi.org/deliver/etsi_en/302300_302399/30230702/01.04.01_60/en_30230702v010401p.pdf
6. DeepSig 官方資料集頁面（RadioML2016.10a）。
   https://www.deepsig.ai/datasets
7. 本專案既有效能量測結果（未重跑，直接引用）：`docs/research/PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md` 第十六節；`results/end_to_end_latency_20260818T062625Z/end_to_end_latency_summary.csv`、`processing_budget_table.csv`。
