# RadioML 2018.01A／Satellite-Relevant Dataset Feasibility

## 一、研究目的

Step 1（`SATELLITE_APPLICATION_AND_LATENCY_REQUIREMENTS_ZH_TW.md`）已定義了 satellite-like application scenario 與 latency requirement，其中第六節指出 RadioML2016.10a 的 QPSK／8PSK 與 DVB-S2/S2X 核心 MODCOD 重疊，但 16APSK／32APSK 不在現有調變清單中。本文件的目的是決定：**最後 satellite-like simulator 應使用什麼 modulation／IQ dataset，以及現有 AWN 是否能直接沿用或必須重新訓練**。本文件範圍僅涵蓋研究、資料核實與正式文件撰寫，不包含 simulator 實作、模型訓練、大型 dataset 下載、DVB-S2/S2X modem 開發，亦不修改 formal pipeline。

## 二、Dataset Inventory（Primary-Source Audit）

### 2.1　來源

1. **DeepSig 官方資料集頁面**：https://www.deepsig.ai/datasets
2. **RadioML2018.01A 原始論文**：T. O'Shea, T. Roy, T. C. Clancy, "Over the Air Deep Learning Based Radio Signal Classification," *IEEE Journal of Selected Topics in Signal Processing*, vol. 12, no. 1, pp. 168-179, Feb. 2018（arXiv 版本：arXiv:1712.04578，2017-12-13）。https://arxiv.org/pdf/1712.04578
3. **external/AWN 與 external/adversarial-rf 兩個已釘選 submodule 自身的程式碼與 README**（`data_loader/data_loader.py`、`README.md`、`config/2018.01a.yml`）——這些是本專案實際依賴、已釘選版本的一手程式碼，比任何二手部落格更直接可信，本節視為與前兩項並列的一手來源。

### 2.2　RadioML2018.01A 官方屬性

| 欄位 | 數值 | 來源 |
|---|---|---|
| Dataset official name | RADIOML 2018.01A | DeepSig 官方頁面 |
| 官方檔名 | `GOLD_XYZ_OSC.0001_1024.hdf5` | `external/AWN/data_loader/data_loader.py:27`、`external/adversarial-rf/data_loader/data_loader.py` 同一行（兩個已釘選 submodule 一致） |
| File format | HDF5，內含 `X`（IQ 訊號）、`Y`（one-hot 標籤）、`Z`（每筆樣本的 SNR）三個 dataset | DeepSig 官方頁面（HDF5／complex floating point）＋ `data_loader.py:54-57`（實際讀取 `f['X']`／`f['Y']`／`f['Z']`） |
| Tensor shape（單筆樣本） | 1024 個 IQ 樣本點；讀入後 permute 為 `[N, 2, 1024]` | 論文正文「the number of samples (ℓ) is = 1024」；`data_loader.py:61` 註解 `Signals = Signals.permute(0, 2, 1)  # X:(2555904, 2, 1024)` |
| Total sample count | **2 555 904** 筆（精確值，非約略值） | `data_loader.py:61` 註解直接寫出 `(2555904, 2, 1024)`；與 DeepSig 頁面「約 2 百萬筆」的概略描述量級一致 |
| Modulation class count | 24 | 論文「Difficult dataset...all 24 modulations」；`data_loader.py:16-21` 的 `classes` dict 恰為 24 個 key |
| SNR range | -20 dB 到 +30 dB（**Es/N0**，非單純 SNR，論文明確使用 Es/N0 這個記號） | 論文正文「with low SNR examples (from -20 dB to +30 dB Es/N0)」 |
| SNR step / 每組合筆數 | 論文正文與兩個已釘選 submodule 的程式碼皆未明確寫出逐步驟的 SNR step 或每個 (modulation, SNR) 組合的確切筆數；只能從總筆數反推：2 555 904 ÷ 24 類 ÷ 26 個 SNR 等級（-20 到 +30 dB、間隔 2 dB 剛好是 26 級）＝每組合 4 096 筆，此為由總數回推的**間接推論**，不是任一來源逐字寫出的數字，本文件明確標記其推論性質，不當作逐字引用的一手事實 | 回推自 `data_loader.py` 的總筆數與論文的 SNR 範圍 |
| Datatype | complex floating point（讀入 PyTorch 後轉為 `float32`） | DeepSig 官方頁面；`data_loader.py:60` |
| License | Creative Commons Attribution – NonCommercial – ShareAlike 4.0（CC BY-NC-SA 4.0） | DeepSig 官方頁面 |
| Synthetic channel impairments | 使用模擬通道模型產生（見論文 Section III 與 Figure 2：對每筆樣本獨立抽樣通道參數，包含多路徑衰落延遲擴散 τ 等），DeepSig 官方頁面明確描述為「synthetic simulated channel effects」，**不是** over-the-air 實測 | 論文 Section III；DeepSig 官方頁面 |

### 2.3　來源間的差異與需要明確標記之處

1. **「synthetic」vs「over-the-air」的用詞混淆**：本文件研究過程中，部分二手來源將 RadioML2018.01A 描述為「synthetically generated and over-the-air captured」，但兩個一手來源（DeepSig 官方頁面明確寫「synthetic simulated channel effects」；論文正文也明確區分「Difficult」24-class 資料集是用模擬通道產生，OTA 量測是論文中**另一個獨立、規模小得多的實驗**，不是公開釋出的 24-class HDF5 檔案的一部分）皆指向**公開釋出的 RadioML2018.01A 本身是純模擬通道產生，不含真實 over-the-air 量測**。本文件採用兩個一手來源一致的說法，並在此明確記錄二手來源的用詞混淆，不予採信。
2. **論文正文一處措辭與其自身正式類別清單不一致**：論文在描述資料集動機時寫道「a number of high order modulations (QAM256 and **APSK256**)」，但論文自身給出的正式 24 類清單（以及 `data_loader.py` 的 `classes` dict）中最高階 APSK 僅到 **128APSK**，並不存在 256APSK／APSK256 這個類別。本文件判斷這是論文正文一句概括性描述帶入了業界（DVB-S2X 標準本身確實定義了 256APSK，見第五節）常見的高階調變參照，但**不代表 RadioML2018.01A 資料集本身包含 APSK256 類別**——這點依兩個一手來源（論文正式清單、`data_loader.py` 程式碼）互相印證後明確排除，不採用論文正文那一句概括描述。
3. **DeepSig 官方頁面的重要限定聲明**：DeepSig 官方頁面明確將包含 RadioML2018.01A 在內的這批資料集定位為「early academic research work」，標註「known errata」，並聲明這些資料集「NOT currently used within DeepSig products」，同時建議研究者針對目前的研究改用真實資料或自行以 MATLAB／GNU Radio 產生。本文件在第九節與第十二節限制中呼應此聲明，不誇大 RadioML2018.01A 的資料品質或代表性。

## 三、完整 Modulation Inventory

依 `data_loader.py:16-21`（`external/AWN` 與 `external/adversarial-rf` 兩個已釘選 submodule 一致）逐字核對的官方 24 類清單（原始 dict 中 `b'00K'` 為原始程式碼本身的拼寫，應為 OOK 之誤，本文件如實記錄，不代為修正）：

| Label index | 名稱（原始程式碼拼寫） | 分類（見第五節） |
|---|---|---|
| 0 | OOK（程式碼原文拼作 `00K`） | C |
| 1 | 4ASK | B |
| 2 | 8ASK | B |
| 3 | BPSK | A（family-only，見 5.3 節） |
| 4 | QPSK | A |
| 5 | 8PSK | A |
| 6 | 16PSK | B |
| 7 | 32PSK | B |
| 8 | 16APSK | A |
| 9 | 32APSK | A |
| 10 | 64APSK | A |
| 11 | 128APSK | A |
| 12 | 16QAM | B |
| 13 | 32QAM | B |
| 14 | 64QAM | B |
| 15 | 128QAM | B |
| 16 | 256QAM | B |
| 17 | AM-SSB-WC | C |
| 18 | AM-SSB-SC | C |
| 19 | AM-DSB-WC | C |
| 20 | AM-DSB-SC | C |
| 21 | FM | C |
| 22 | GMSK | C |
| 23 | OQPSK | C |

**特別核對第二節任務指定的 8 個調變是否存在**：BPSK ✓、QPSK ✓、8PSK ✓、16APSK ✓、32APSK ✓、64APSK ✓、128APSK ✓ 皆存在；**256APSK 不存在**（如第 2.3 節所述，最高階 APSK 僅到 128APSK）。以上全部依 `data_loader.py` 程式碼逐字核對，未依記憶填寫。

## 四、與 DVB-S2/S2X 正式 Mapping

依 Step 1 文件第六節已核實的 DVB-S2（EN 302 307-1 V1.4.1）／DVB-S2X（EN 302 307-2 V1.4.1）內容，加上本文件針對 DVB-S2X 擴充 MODCOD 的補充查證：

| RadioML2018 class | DVB-S2 support | DVB-S2X support | Exact constellation match? | Family-only match? | Notes |
|---|---|---|---|---|---|
| BPSK | 否（DVB-S2 核心 4 星座不含 BPSK） | 是（VL-SNR 模式使用 π/2-BPSK） | **否** | 是 | RadioML 的 BPSK 是傳統 BPSK，DVB-S2X 的 VL-SNR 模式使用的是 π/2-BPSK（相位旋轉方案不同），僅家族層級相關，非精確對應 |
| QPSK | 是（核心 4 星座之一） | 是（延續 S2 星座） | 是 | — | 星座點數與相位配置一致 |
| 8PSK | 是（核心 4 星座之一） | 是（延續 S2 星座） | 是 | — | 同上 |
| 16APSK | 是（核心 4 星座之一） | 是（延續並擴充） | **家族層級相符，非逐位元確認** | 是 | RadioML 的「16APSK」是否與 DVB-S2 規格的 ring ratio／星座映射逐位元一致，本文件未逐一核對星座生成參數，僅確認調變家族名稱相符 |
| 32APSK | 是（核心 4 星座之一） | 是（延續並擴充） | 家族層級相符，非逐位元確認 | 是 | 同上 |
| 64APSK | 否（不在 DVB-S2 核心 4 星座） | **是**（S2X 擴充定義 3 種不同環配置的 64APSK） | 家族層級相符，非逐位元確認 | 是 | DVB-S2X 定義了 16+16+16+16、8+16+20+20、4+12+20+28 三種不同的 64APSK 環配置，RadioML 僅有單一「64APSK」標籤，無法確認對應哪一種環配置 |
| 128APSK | 否 | **是**（S2X 擴充，6 環 128 點） | 家族層級相符，非逐位元確認 | 是 | 同上，RadioML 標籤未指定環配置 |
| 16PSK | 否 | 否（S2X 高階調變走 APSK 路線，未見標準內定義 16PSK） | 否 | 否 | 與 DVB-S2/S2X 無直接對應 |
| 32PSK | 否 | 否 | 否 | 否 | 同上 |
| 4ASK／8ASK | 否 | 否 | 否 | 否 | ASK 家族不在 DVB-S2/S2X 定義的調變範圍內 |
| 16QAM／32QAM／64QAM／128QAM／256QAM | 否 | 否 | 否 | 否 | DVB-S2/S2X 因衛星非線性通道（HPA 接近飽和運作）考量，採用準恆包絡的 PSK／APSK 家族，未定義 QAM 星座；QAM 常見於其他衛星或地面標準，但不屬於 DVB-S2/S2X 規格範圍 |
| OOK／GMSK／OQPSK／AM-SSB-*／AM-DSB-*／FM | 否 | 否 | 否 | 否 | 與 DVB-S2/S2X 無直接對應 |

**必須明確區分的重點**：即使 RadioML2018.01A 的類別名稱與 DVB-S2/S2X 規格書的調變家族名稱相同（例如「16APSK」），**這僅代表 modulation-family overlap（調變家族名稱重疊），不代表 standard-compliant DVB waveform（符合標準規範的 DVB 波形）**——RadioML 的星座生成參數（環半徑比、符元映射、pilot 結構、FEC）與 DVB-S2/S2X 規格書定義的精確參數是否一致，本文件未逐一核對，且以 RadioML 論文本身的資料集生成方法（Section III，通用單載波調變模擬工具）來看，**沒有證據顯示其刻意複製了 DVB-S2/S2X 規格書的精確星座參數**。

## 五、DVB-S2X 高階 APSK 補充查證

DVB-S2X 除 DVB-S2 原有的 QPSK/8PSK/16APSK/32APSK 之外，確認擴充定義了 64APSK（3 種環配置：16+16+16+16、8+16+20+20、4+12+20+28）、128APSK（6 環、128 點）、以及 256APSK（2 種星座配置，DVB-S2X 規格書定義，但**不在 RadioML2018.01A 的 24 類清單中**，見第三節）。依證據，64APSK 與 128APSK 皆屬於 DVB-S2X 正式定義的擴充調變，應歸類為第四節的類別 A（見第四節表格）。

## 六、現有 AWN 相容性檢查（程式碼層級核實）

### 6.1　現有 AWN 架構與設定

直接讀取 `external/AWN/models/model.py`、`src/adapters/awn_adapter.py`、`external/AWN/config/2016.10a.yml` 與 `external/AWN/config/2018.01a.yml`：

| 項目 | 現有（2016.10a，本專案實際使用） | 2018.01a 官方建議設定 |
|---|---|---|
| Input shape | `[N, 2, 128]`（由 `src/sensing/normalize.py:to_awn_input` 以 `seg_len=128` 呼叫方傳入決定，非模型架構本身寫死） | `[N, 2, 1024]` |
| Class count（`num_classes`） | **11**（`src/adapters/awn_adapter.py:39` 硬編碼 `_AWN_2016_10A_CFG`） | **24** |
| Wavelet decomposition 深度（`num_levels`／`num_level`） | **1**（同上硬編碼） | **4** |
| Label mapping | `src/sensing/radioml_source.py:32` 的 `RML2016_10A_CLASSES`，與 `external/adversarial-rf/data_loader/data_loader.py` 的 2016.10a class dict 逐字核對一致（11 個 key，特定順序） | `data_loader.py` 的 2018.01a class dict，24 個 key，完全不同的順序與內容 |
| Checkpoint 檔案 | `external/adversarial-rf/2016.10a_AWN.pkl`（sha256=`8af0458f2570c465b5bb0ebad00817944f8171d888cd7cff1324ecb258820695`） | `external/adversarial-rf/2018.01a_AWN.pkl`（見 6.3 節，**已存在於已釘選 submodule 中**） |
| Dataset loader | `src/sensing/radioml_source.py:load_radioml_dict`，讀取 `{(mod,snr): ndarray[1000,2,128]}` 格式的 pickle 檔（RML2016.10a 專用格式） | 需要讀取 HDF5 的 `X`/`Y`/`Z` 三個 dataset，格式完全不同 |

### 6.2　模型架構的關鍵細節：`AdaptiveAvgPool1d` 的影響

逐行閱讀 `external/AWN/models/model.py` 確認：`AWN.forward()` 的最終分類層之前使用 `nn.AdaptiveAvgPool1d(1)`（第 102 行），這使得最終分類器 `self.fc`（第 104-108 行）的輸入維度**只取決於 `out_channels = in_channels * (num_levels + 1)`，不直接取決於輸入序列長度 T**。這代表：

- 純粹從張量形狀運算的角度，**同一個 `num_levels=1` 的模型架構，理論上可以接受任意長度的輸入 T（不會因為 T=1024 而在 forward pass 中發生形狀錯誤）**，因為 `AdaptiveAvgPool1d` 會把任意長度的時間維度壓縮成 1。
- 但這**不代表**現有以 `num_levels=1、T=128` 訓練出的 checkpoint，可以直接對 `T=1024` 的輸入產生有意義的分類結果——checkpoint 的權重是針對 128-sample 視窗內的訊號統計特性（wavelet lifting 分解、卷積核感受野）學習出來的，從未見過 1024-sample 長度下的訊號統計分布，即使 forward pass 不報錯，預測品質沒有依據可以信賴，需要實測驗證，此項驗證不在本文件範圍內。
- 更關鍵的限制是 `num_classes`：`self.fc` 最後一層 `nn.Linear(latent_dim, num_classes)` 的**輸出維度**由 `num_classes` 直接決定，現有 checkpoint 的這一層權重形狀是 `[11, latent_dim]`，若要輸出 24 類（或任何非 11 類的子集合）**必然需要更換並重新訓練這一層**，無法直接沿用。
- 若進一步採用官方建議的 `num_levels=4` 設定（而非勉強沿用 `num_levels=1`），`self.levels`（`nn.ModuleList`）本身的模組數量就從 1 個變成 4 個，這是模型「層數」層級的結構差異，現有 checkpoint 的 state_dict 完全無法載入這個不同結構的模型。

### 6.3　重大發現：`external/adversarial-rf` 已經存在一個 2018.01a 的訓練好 checkpoint

在已釘選的 `external/adversarial-rf` submodule（commit `ced705e`）中，發現 `external/adversarial-rf/2018.01a_AWN.pkl`（sha256=`76ef99283b391b721ec102dc90d66e0372fb171b8844efbdacc4eda838a8fef9`，檔案大小 1 546 545 bytes，約為 2016.10a checkpoint 的 3 倍，與更深的 `num_levels=4` 架構量級相符）。本文件以唯讀方式載入此 checkpoint 之 state_dict（未修改任何檔案）逐一核對：

- `self.levels` 對應的模組編號為 `['0','1','2','3']`，恰為 4 層，與 `config/2018.01a.yml` 的 `num_level: 4` 完全吻合。
- 最終分類層 `fc.2.weight` 形狀為 `torch.Size([24, 320])`，恰為 24 類輸出，與 `num_classes: 24` 完全吻合。
- `external/adversarial-rf/README(1).md` 與 `external/AWN/README.md`（兩者內容一致）明確記載：「We conducted experiments on three datasets, namely RML2016.10a, RML2016.10b, and RML2018.01a」，並提供 2018.01a 的調變清單與樣本數對照表（19 digital + 5 analog = 24 類，2.5 million (2×1024)，與第三節逐字核對一致）——這代表 AWN 論文作者本身確實在 2018.01a 上訓練過模型，`2018.01a_AWN.pkl` 極可能是這個官方訓練流程的產物，而非隨意留下的檔案。
- **核實過程中未找到任何逐 SNR 或整體測試集準確率的具體數字**（兩份 README 皆只描述訓練/評估流程，未附上結果表格），因此本文件**不宣稱**這個 checkpoint 已驗證可用於實際分類任務，只確認其**架構與官方設定檔完全吻合、來源可信**。是否真正可用，需要在未來實際執行 `--mode eval` 驗證（此項驗證未於本文件執行，見第九節限制）。

### 6.4　回答第四節六個問題

1. **現有 AWN 是否能直接吃 RadioML2018 2×1024？** 部分可以，部分不行：張量形狀本身因 `AdaptiveAvgPool1d` 不會報錯，但現有 checkpoint 的 `num_classes=11` 與 2018.01a 需要的 24 類不符，這一層必然無法沿用；若進一步採用官方建議的 `num_levels=4`，整個 wavelet 分解層數也不同，屬於不同架構。
2. **若不能，原因是 architecture、checkpoint、loader 或全部？** 全部：(a) checkpoint 的 `num_classes`／`num_levels` 與 2018.01a 官方設定不同；(b) 本專案 `src/sensing/radioml_source.py` 的 loader 是為 2016.10a 的 pickle／dict 格式與 11 類 label mapping 特製，2018.01a 是完全不同的 HDF5 格式；(c) `src/adapters/awn_adapter.py` 目前把 `num_classes=11`／`num_levels=1` 寫死在 `_AWN_2016_10A_CFG`，沒有任何切換到 2018.01a 設定的介面。
3. **是否可以將 1024 window 切成 128？** 工程上可以，第七節分析四種切分策略。
4. **切成 128 是否仍能保留 dataset label 意義？** 部分保留：調變類型在整個 1024-sample burst 內不會改變，因此切出的任何 128-sample 子視窗的「調變類型」標籤仍然有效；但原始論文刻意選擇 1024 這個長度是為了建立「short-time observation，無法等待更多資料」的困難分類任務（見論文正文），切得更短會改變任務難度本身的定義，且子視窗與全視窗之間的通道實現（該筆樣本抽樣到的特定衰落/相位參數）統計特性可能不同。
5. **是否可以只取 RadioML2018 的 satellite-relevant modulation subset？** 可以，工程上只需篩選對應的 (mod,snr) 組合，不需要用到全部 24 類；但無論取多少類，loader 與 label mapping 仍需改寫以讀取 2018.01a 的 HDF5 格式，這部分工作量與是否取子集無關。
6. **是否必須重新訓練 AWN？** 若要用**新的 class subset**（不同於現有 11 類，也不同於完整 24 類），現有兩個 checkpoint（2016.10a 的 11 類、2018.01a 的 24 類）都無法直接沿用最終分類層，至少該層必須重新訓練；若直接使用完整 24 類且採用 6.3 節發現的既有 `2018.01a_AWN.pkl`，則有機會**完全不需要重新訓練**，但其實際準確率尚未驗證，無法據此下定論。

## 七、Sample-Length Feasibility（1024 vs 128，四種方式分析，不實作）

| 方式 | 是否需重訓 | Information loss | Sensing alignment 相容性 | Attack 相容性 | Latency 影響 | Top-K 相容性 |
|---|---|---|---|---|---|---|
| **1. AWN input 改為 1024** | 需要（除非採用 6.3 節既有 checkpoint 且其驗證通過），且需要放棄現有 11 類 checkpoint | 無（使用完整原始長度） | 需要，但 `energy_detect`／`select_aligned_segments` 本身不假設固定串流長度，主要是呼叫端 `seg_len` 參數改變，架構性改動小 | FGSM/PGD/CW 三種攻擊本身對輸入張量形狀無特定假設，理論上可泛化到 T=1024，但計算量隨張量變大而增加，Step 1 全部的延遲量測結果需要針對 T=1024 重新量測，不能沿用 | 需要重新量測（本文件未進行此項量測），預期因張量變大而全面增加 | `TopKAdapter`／`fft_topk_denoise` 依 FFT bin 數運作，K 值的相對意義隨 T 改變，需要重新校準 K，不能沿用現有 K=20 |
| **2. Sliding 128 windows** | 不需要（沿用現有 11 類或未來 24 類 checkpoint，視窗本身仍是 128 長） | 中，取決於視窗重疊率與聚合方式；理論上可涵蓋全部 1024 樣本的資訊，但每個決策仍只看局部 128 樣本 | 良好，直接沿用現有 128-length 分段邏輯，只是重複套用多次 | 需要新設計：對多視窗個別攻擊、或對多視窗聯合攻擊，是不同的威脅模型，現有 `AttackAdapter` 未處理多視窗聚合場景 | 隨視窗數量倍增（例如 1024/128=8 個視窗，8 倍單視窗成本），但每視窗延遲量級不變 | 每個視窗各自可套用現有 K=20 語意，聚合最終決策需要新邏輯 |
| **3. Center／max-energy 128 crop** | 不需要 | 較高，捨棄約 7/8 的原始上下文；但因為調變類型在整個 burst 內不變，對「調變分類」這個任務而言，捨棄的主要是通道/相位實現的細節，不是類別資訊本身 | **最佳**，直接沿用現有 `select_aligned_segments` 的 max-energy 選窗邏輯，本質上與現有 pipeline 完全相同，不需要新程式 | **最佳**，與現有 pipeline 完全相同的攻擊面，不需要新設計 | 幾乎無額外影響，與現有單視窗延遲相同 | 完全沿用現有 K=20 語意，無需改變 |
| **4. 聚合多個 128-window 預測** | 不需要 | 中，介於方式 2 與 3 之間，取決於聚合的視窗數 | 中，需要新的多視窗擷取＋聚合邏輯，但仍建立在現有 128-length 分段之上 | 需要新設計：攻擊者若要翻轉聚合後的最終決策，需要同時、協同地擾動多個視窗，這本身是一個值得未來研究、但超出本文件範圍的新威脅模型問題 | 隨聚合視窗數增加，但小於方式 2（若聚合視窗數少於全部 sliding windows） | 每視窗沿用 K=20，聚合邏輯需要新設計 |

**方式 3（center／max-energy crop）是唯一完全不需要新程式碼、不需要重新訓練、不改變現有攻擊/防禦/延遲特性的方式**——本質上就是把現有 pipeline 原封不動地套用在一段更長的來源串流上，只是把 1024-sample 的 RadioML2018 樣本視為「來源串流的一部分」而非「必須整段處理的輸入」。這使它成為第八、九節建議中，若未來要以最低風險方式引入 RadioML2018 素材時的**優先技術路徑**（但本文件未實作此路徑，且第九節的建議策略事實上根本不需要用到 RadioML2018 素材，見下）。

## 八、Strategy A／B／C 比較

| | **Strategy A**：沿用 RadioML2016.10a，僅用 BPSK/QPSK/8PSK 建立 satellite-like scenario | **Strategy B**：使用 RadioML2018.01A satellite-relevant subset，重新訓練 AWN | **Strategy C**：自行生成 standard-inspired satellite IQ（QPSK/8PSK/16APSK/32APSK），訓練/微調 AWN |
|---|---|---|---|
| Implementation effort | 極低（篩選既有調變，零新程式碼） | 中至高（需要新 HDF5 loader、新 label mapping、新 class subset 篩選邏輯；即使沿用 6.3 節既有 checkpoint，仍需要驗證流程） | 高（需要自行實作符合規範的 APSK 星座生成器，具體環比、映射參數，屬於 DVB-S2/S2X 波形細節，逼近「做 DVB-S2/S2X modem」的範圍，本文件與後續工作皆應審慎評估） |
| Training requirement | 無 | 視情況：若 6.3 節既有 `2018.01a_AWN.pkl` 驗證可用，可能不需要；若不可用或需要客製 class subset，需要至少微調最終分類層，甚至完整重訓 | 一定需要（自製資料，沒有任何既有 checkpoint 可沿用） |
| Satellite relevance | 中（QPSK/8PSK 與 DVB-S2 核心精確相符，但完全不含 APSK，且資料集本身無衛星通道模擬） | 高（涵蓋 DVB-S2/S2X 完整核心＋擴充調變家族，1024-sample 視窗更接近真實截取情境） | 潛在最高，但風險也最高——若生成參數與規格書不符，「relevance」的宣稱本身就站不住腳 |
| Compatibility with current pipeline | 完全相容，零改動 | 低至中，需要新 loader／新 checkpoint 切換邏輯／可能需要新 latency 量測（若改用 1024 長度） | 中，若刻意生成 128-length 短波形可維持相容性，但生成流程本身是全新模組 |
| Dataset size | 已有（現有 RML2016.10a_dict.pkl） | 完整檔案約 2.5M 筆／2×1024 float，屬於**大型下載**，超出本文件範圍，未執行下載 | 自行控制，可以很小 |
| Time to finish project | 最快 | 較慢，且下載本身超出本文件範圍，故本文件無法驗證 | 最慢，且與本文件「不做 DVB-S2/S2X modem」的範圍限制衝突風險最高 |
| Scientific defensibility | 中，但誠實——明確標示為「調變家族重疊」而非「涵蓋 DVB-S2/S2X 高階 MODCOD」，不誇大 | 高，但需要以第四節「family overlap ≠ standard-compliant waveform」的但書明確限定宣稱範圍 | 風險最高——最容易掉入「APSK 標籤等同 DVB-S2X 合規波形」的陷阱，正是本文件被要求避免的問題 |
| Risk | 低 | 中至高（checkpoint 未驗證、大型下載超出本文件範圍、engineering surface 大） | 高 |

## 九、Final Dataset Recommendation

### 9.1　For Project Close

- **Dataset**：RadioML2016.10a（沿用現有 `RML2016.10a_dict.pkl`，零新下載）。
- **Modulation set**：BPSK／QPSK／8PSK（見第十節 Option 1）。
- **Sample length**：128（沿用現有全部 pipeline，零改動）。
- **AWN strategy**：直接沿用現有 checkpoint（`2016.10a_AWN.pkl`，`num_classes=11`／`num_levels=1`），僅在 satellite-like 情境展示時篩選 BPSK/QPSK/8PSK 三類的結果呈現，不改動模型本身。
- **Retraining**：**否**。
- **Expected engineering cost**：極低——本策略不需要新增或修改任何正式程式碼，僅需在展示/報告層級篩選既有結果。
- **Expected experiment cost**：極低——可直接重用 Step 1（`PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md`、`SATELLITE_APPLICATION_AND_LATENCY_REQUIREMENTS_ZH_TW.md`）已經完成的量測結果，不需要新實驗。
- **Limitations**：不含 APSK，因此不能宣稱涵蓋 DVB-S2/S2X 的高階 MODCOD；資料集本身沒有衛星通道模擬（見 Step 1 第十一節 MUST/SHOULD/OPTIONAL 通道因素，本策略仍需搭配那些因素的後續補強才算完整）。

### 9.2　If Future Extension（超出本文件範圍，僅記錄路徑）

依風險與可行性排序：

1. **RadioML2018.01A APSK 子集合＋驗證既有 `2018.01a_AWN.pkl`**：優先路徑。若未來要推進，第一步應該是（在允許下載大型檔案的前提下）下載完整 HDF5 檔案、撰寫新 loader、對 6.3 節已發現的既有 checkpoint 執行 `--mode eval` 驗證其真實準確率，而不是預設需要重新訓練。若驗證結果可用，後續只需要處理第七節的 sample-length 策略選擇（建議優先採用方式 3：center／max-energy crop，工程風險最低）。
2. **Standard-compliant DVB waveform（真正符合 DVB-S2/S2X 規格書參數的波形產生器）**：只有在「family-only match」被認為不足以支撐研究結論時才需要考慮，且應該被視為與「做 DVB-S2/S2X modem」相近的獨立、大範圍工作項目，需要專門的時間與範疇規劃，不應該在專案收尾階段倉促進行。
3. **真實 SDR IQ 擷取**：最貼近實際衛星訊號特性，但需要硬體、頻率協調、可能的法規限制，是最長期的延伸方向。

## 十、最小 Satellite Modulation Set

| Option | 內容 | DVB relevance | 現有 AWN 支援 | 是否需 retraining | 是否能在 project close 前完成 |
|---|---|---|---|---|---|
| **Option 1** | BPSK／QPSK／8PSK | QPSK／8PSK 精確符合 DVB-S2 核心；BPSK 為 DVB-S2X VL-SNR 家族層級相關（非精確符合） | 完整（現有 11 類皆含） | 否 | **是，已可直接完成** |
| Option 2 | QPSK／8PSK／16APSK／32APSK | 精確符合 DVB-S2 全部 4 個核心星座 | 部分（QPSK/8PSK 有，16/32APSK 完全沒有，需要 RadioML2018 或自行生成） | 是（除非 6.3 節既有 checkpoint 驗證可用） | 不確定，取決於未來驗證結果 |
| Option 3 | QPSK／8PSK（僅 2 類，最保守子集） | 精確符合但覆蓋面更窄 | 完整 | 否 | 是，已可直接完成 |

**最終選擇：Option 1（BPSK／QPSK／8PSK）**，理由見第九節。

## 十一、Future Extension Set

若未來（專案收尾後）決定推進 Option 2，建議的延伸調變集合為 **QPSK／8PSK／16APSK／32APSK**（DVB-S2 核心 4 星座完整覆蓋），是否進一步納入 64APSK／128APSK（DVB-S2X 擴充）取決於屆時的研究範疇決定，非本文件需要拍板的問題。

## 十二、是否真的需要 APSK

研究問題設定為：「Spectrum Sensing + AMC + adversarial attack 在 satellite-like channel 與 latency budget 下是否可行」。針對這個具體問題，APSK 的必要性分析如下：

- **MUST：無**。本專案全部延遲相關的量測結果（`PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md`、`SATELLITE_APPLICATION_AND_LATENCY_REQUIREMENTS_ZH_TW.md`）顯示，pipeline 的處理延遲由**張量形狀與計算圖結構**（AWN 層數、攻擊迭代次數、Top-K 的 FFT bin 數）決定，與輸入訊號實際承載的調變類別**無關**——同樣形狀的 QPSK 樣本與 16APSK 樣本，通過同一個網路的計算時間理論上相同。因此，就「pipeline 在 satellite-like latency budget 下是否可行」這個問題本身而言，納入 APSK **不會提供額外的證據力**，BPSK／QPSK／8PSK 已經足以完整回答這個問題。
- **SHOULD：視研究敘事範疇而定**。如果專案的敘事想進一步宣稱「涵蓋 DVB-S2/S2X 實際會用到的高階 MODCOD」，那麼加入 APSK 會讓這個更廣義的代表性宣稱更站得住腳——但這是一個**敘事／範疇決定**，不是回答核心延遲可行性問題的技術必要條件。依 Step 1 文件第一節明確定義的研究範疇（processing latency 與 application scenario 的對照，不是 DVB-S2/S2X modulation classification benchmark），本專案的敘事範疇並未做出這種更廣義的宣稱。
- **OPTIONAL：作為未來延伸的加分項**，若專案收尾後有餘裕，可依第九節 9.2 節路徑推進，但不應該為了「因為 DVB-S2X 有 APSK 就自動認定一定要做」而在收尾階段倉促加入。

**結論：BPSK／QPSK／8PSK 已足以建立衛星相關的 proof-of-concept，回答本專案實際設定的研究問題；APSK 不是必要條件。**

## 十三、限制（Limitations）

1. 第二節「每組合 4 096 筆」的數字是由總樣本數與 SNR 範圍回推的間接推論，不是任何一手來源逐字寫出的數字，需要在未來若真的使用此資料集時，以實際讀取的 HDF5 檔案核對確認。
2. 第四節的 DVB-S2/S2X mapping 僅比對調變家族名稱，未逐一核對 RadioML2018.01A 的星座生成參數（環半徑比、符元映射方式）是否與 DVB-S2/S2X 規格書的精確定義一致，也未查證 RadioML 論文的資料集生成工具是否參考過 DVB-S2/S2X 規格書的具體數值。
3. 第六節對 `2018.01a_AWN.pkl` checkpoint 的核實僅限於架構層級（張量形狀、層數與官方設定檔的吻合度），**未執行任何推論或評估**，因此不確認、也不否定其實際分類準確率；是否可用需要未來實際執行 `--mode eval` 驗證，本文件明確不執行此項驗證。
4. 第七節的四種 sample-length 策略分析是質性工程判斷，不是量化實驗結果，本文件未實作任何一種策略，數字（如 latency 倍增估計）是基於現有架構理解的合理推論，不是實測值。
5. 第八節的策略比較欄位（高/中/低）反映研究判斷，不是可重現的量化評分模型。
6. 第十二節的 APSK 必要性分析基於「pipeline 延遲與調變內容無關」這項在既有量測中反覆觀察到的性質（見 `PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md` 第十五節，AWN 推論延遲與訊號內容無關、與執行緒設定有關的發現），但本文件未針對 RadioML2018.01A 或任何 APSK 波形做直接的延遲實測，此推論建立在架構理解之上，不是逐一驗證過的實測結論。
7. 本文件不構成、也不宣稱已完成 RadioML2018.01A 的下載、訓練或驗證——所有結論皆為研究層級的資料集可行性評估與文獻/程式碼核實，`external/AWN` 與 `external/adversarial-rf` 皆未被修改。

## 十四、Primary References

1. DeepSig 官方資料集頁面。https://www.deepsig.ai/datasets
2. T. O'Shea, T. Roy, T. C. Clancy, "Over the Air Deep Learning Based Radio Signal Classification," *IEEE Journal of Selected Topics in Signal Processing*, vol. 12, no. 1, pp. 168-179, Feb. 2018（arXiv:1712.04578）。https://arxiv.org/pdf/1712.04578
3. `external/AWN/data_loader/data_loader.py`、`external/adversarial-rf/data_loader/data_loader.py`（已釘選 submodule，commit 分別見專案 `.gitmodules`／既有 manifest 記錄）——2018.01a 官方類別清單、檔名、tensor shape 之程式碼層級一手來源。
4. `external/AWN/models/model.py`、`src/adapters/awn_adapter.py`、`external/AWN/config/2016.10a.yml`、`external/AWN/config/2018.01a.yml`——AWN 架構與現有設定之程式碼層級一手來源。
5. `external/adversarial-rf/README(1).md`、`external/AWN/README.md`——AWN 論文作者官方訓練/評估流程說明。
6. ETSI EN 302 307-1 V1.4.1（2014-11）、ETSI EN 302 307-2 V1.4.1（2024-08）——沿用 Step 1 文件（`SATELLITE_APPLICATION_AND_LATENCY_REQUIREMENTS_ZH_TW.md` 第十六節）已核實之 DVB-S2/S2X 規格來源，本文件額外查證 64APSK／128APSK／256APSK 於 S2X 擴充 MODCOD 中的定義。
7. 本專案既有效能與延遲研究文件（未重跑，直接引用）：`docs/research/PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md`、`docs/research/SATELLITE_APPLICATION_AND_LATENCY_REQUIREMENTS_ZH_TW.md`。
