# 頻譜感測對抗攻擊：下一步工作與 SDR 硬體預算

> 給老師的進度報告
>
> - **專案**：sdr-awn-spectrum-sensing-poc
> - **分支**：nzzz_proposal
> - **日期**：2026-09-01

目前 A0（數位張量注入攻擊）與感測管線已完成。本文分三部分：(1) 還可以做什麼、(2) Stage 3 線纜雙 SDR 硬體預算、(3) 省錢階梯。

---

## 1. 還可以做什麼

依「攻擊 / 對齊 / 防禦 / 工程 / 嚴謹性」五面向盤點。★ 為建議這次報告主打的三條。

狀態圖例：✅ 已完成 · 🟡 進行中／部分 · ⬜ 待辦

### A. 攻擊面 — 目前只完成 A0，最有故事性

| 狀態 | 項目 | 值得做的理由 |
|---|---|---|
| ⬜ | **A1 受控 RF 注入（實體）** | 主威脅；接上 SDR 才是「真的能打」，而非只在 tensor 上打 |
| 🟡 | **G1–G5 系統性掃描** | 協定已就緒，只差跑 Monte Carlo → 產出 PSR–成功率曲線，報告立刻有圖 |
| ⬜ | **A2 query-based 黑箱** | 補上「攻擊者不知模型內部」的現實假設，威脅模型才完整 |

### B. 感測對齊問題 — 你們最獨特的發現（1.68pp 那條線） ★

| 狀態 | 項目 | 值得做的理由 |
|---|---|---|
| ⬜ | **對齊誤差 ablation 曲線** | 刻意錯位 N 個 sample → 掉多少 pp；一張好圖，且直接連到 G4 邊界攻擊 |
| ⬜ | **修正對齊後重跑** | 驗證對齊做對後 1.68pp gap 是否收斂，反證 gap 來源 |

> round-9 已定位 gap 來源是 segmentation grid 沒對齊 burst（偵測 region 起點早真實 burst 約 53–61 samples，naive window 僅 52–63% 重疊 burst），但尚未做「錯位量 → 準確率」的系統掃描。

### C. 防禦面 — 目前結論偏負面，需要救 ★

| 狀態 | 項目 | 值得做的理由 |
|---|---|---|
| ⬜ | **自適應 / per-class K** | Top-K 全域固定 K 無顯著效益（Phase 4）→ 改成自適應或攻擊感知 |
| ⬜ | **adaptive attack 繞過 Top-K** | 用 BPDA/STE 白箱繞過；審稿人一定會問，把負面結論轉成洞察 |

### D. 工程落地 — Stage 2 待辦已寫好

| 狀態 | 項目 | 值得做的理由 |
|---|---|---|
| 🟡 | **`.cfile` loader 強化** | 目前僅 `np.fromfile` + 空檔檢查；缺 endianness / NaN / 振幅 / sample_rate / clipping 驗證，也未接進 pipeline |
| ⬜ | **metadata sidecar + chunking** | sample_rate / center_freq / byte_order / gain 契約 + rolling buffer（跨 chunk 保留 detector 狀態） |

### E. 實驗嚴謹性 — 最低成本、CP 值最高

| 狀態 | 項目 | 值得做的理由 |
|---|---|---|
| 🟡 | **結果 CSV 納管 / checksum** | formal run 的 CSV 不在本機（在另一台）→ 報告數字目前無法重現 |
| ⬜ | **信賴區間 / 顯著性檢定** | 現在很多是單次數字 |

### 建議這次報告主打這 3 條（有圖、故事完整、工作量可控）

1. **B 的對齊 ablation 曲線** — 你們獨有的發現，且能直接連到攻擊。
2. **A 的 G1–G5 PSR–成功率曲線** — 協定已就緒，只差跑。
3. **C 的 adaptive attack 繞過 Top-K** — 把負面防禦結論轉成有洞察的結論。

---

## 2. Stage 3 線纜雙 SDR — 硬體預算

「線纜傳導（cabled）」最貴的兩樣 —— GPSDO 時脈同步與專業功率計 —— 在最小可驗證版本裡都可先省掉：用軟體對齊取代硬體同步、用 RX SDR 自量功率取代功率計。

### 2.0 選型第一準則：擾動能不能活下來（見第 4 節）

選型的決定性軸**不是價格，是 ADC/DAC 位元深度（adversarial 擾動存活）與相位雜訊（RML 高階 QAM 保真）**。8-bit 裝置的量化會直接淹掉脆弱擾動，在承載攻擊的路徑上要排除。

| SDR | ADC/DAC | TX? | 獨立 LO | 對本實驗的判定 |
|---|---|---|---|---|
| RTL-SDR | **8-bit** | ✗ 只收 | — | ❌ 會淹掉擾動；只能做粗略 occupancy，不能收攻擊給 AMC |
| HackRF One | **8-bit** | ✓ 半雙工 | ✓ | ❌ 8-bit 對本題是假省錢，量化直接毀掉擾動 |
| **ADALM-PLUTO** | 12-bit (AD9363) | ✓ 全雙工 | ✓ | ✅ 平價 SDR 的實務地板（主推） |
| LimeSDR / BladeRF 2.0 | 12-bit (AD936x) | ✓ | ✓ | ➖ 與 Pluto 同晶片家族，相位雜訊略好、較貴 |
| USRP B210 | 12-bit (AD9361) | ✓ | 雙 TX 共 LO | ➖ 相位雜訊優於 Pluto，但雙 TX 共 LO → 仍需第二台才有獨立 CFO |
| USRP N210 / X310 | **14–16-bit** | ✓ | ✓ | ⭐ 唯一能把量化移出混淆因子的等級，但每台 ~US$1,700+ |

**兩個關鍵事實：**
1. **12-bit 幾乎是所有平價 TX SDR 的天花板** —— Pluto / Lime / BladeRF / B210 全是 AD936x 家族，位元深度相同；想更高只能跳到 N210/X310（14-bit），價格跳 3–5 倍。
2. **8-bit（HackRF、RTL-SDR）在承載擾動的路徑上直接排除。**

**推薦結論：**
- **主推 PlutoSDR（12-bit）** —— 不是因為它好，而是往上要多花 5 倍，且 **12-bit 本身就是一個真實的部署級接收機**。代價：必須把「量化底噪」當強制校準項 —— 發射前量測擾動在滿刻度下是幾個 LSB，確保 ≥ 數個 LSB 並於報告揭露，否則「攻擊失效」會分不清是通道還是自己的 DAC。
- **收攻擊的那台 RX 最該花錢**（AMC 在 RX 端重建攻擊）：若要乾淨分離「通道 vs 轉換器量化」，RX 升 **USRP N210（14-bit）**、TX 端維持 Pluto（非對稱配置省錢）。
- **RML 高階 QAM（如 64-QAM）保真不足時** → 優先升 **B210**（相位雜訊較好，仍 12-bit）。
- **這一個決定會改變選型**：要驗「真實部署接收機」（防禦方也用便宜 SDR）→ 12-bit 就對，Pluto 即可；要「乾淨分離通道效應、排除自家 ADC 嫌疑」→ RX 升 14-bit N210。

### 2.1 三種預算級距

| 版本 | 能證明什麼 | 預估價 |
|---|---|---|
| **Tier 0 概念驗證** | 1 台全雙工 loopback：`.cfile` 經真實 RF 收發往返、驗證前處理不壞 | **~US$250 / NT$8k** |
| **Tier 1 最小可驗證（推薦）** | 2 個獨立發射源（合法 + 攻擊者，**各自獨立 CFO**）→ RX，真正打 G1–G5 | **~US$850 / NT$27k** |
| **Tier 2 正規實驗室** | 加時脈同步、程控衰減、功率計 → 數字可重現、可寫論文 | **~US$3,500 / NT$110k** |

### Tier 1 明細（最小可驗證）

| 品項 | 型號範例 | 數量 | 小計 (USD) |
|---|---|---:|---:|
| 發射 SDR（合法源 + 攻擊源） | ADALM-PLUTO | 2 | 460 |
| 接收 SDR | ADALM-PLUTO | 1 | 230 |
| 2-way 合波器 | Mini-Circuits ZFSC-2-x | 1 | 50 |
| 固定 + 手動步進衰減器 | SMA 10/20/30 dB + 0–70 dB | 4 | 86 |
| SMA 電纜 + 50Ω 終端 | — | 8 | 65 |
| **合計** | | | **≈ 850** |

**為何用 3 台獨立 SDR 而非 1 台 B210 雙通道**：B210 兩個 TX 共用同一 LO → 兩路 CFO 相同，就模擬不出「攻擊者與合法源失同步」，而這正是 `inject_additive_waveform` 裡 `normalized_cfo` 要驗的核心。獨立的 3 台 PlutoSDR 各有自己的振盪器 → 有真實的獨立 CFO。

**可省項目說明**：
- **省掉 GPSDO（省 ~US$1,000）**：cabled 傳導、短時間量測，用軟體 correlation / preamble 對齊時間即可。
- **省掉功率計（省 ~US$400）**：用 RX SDR 自己量收到的功率反推 achieved PSR（`achieved_psr_db` 本來就這樣算）。
- **不能省的**：位元深度。Tier 1 用 12-bit Pluto，必須把量化底噪當強制校準項（見 2.0）。若要移除量化混淆，RX 升 14-bit N210（+~US$1,700）。

### 2.2 針對本 pipeline 的推薦：TX 省、RX 花

完整目標是「RML replay → adversarial 注入 → 過硬體 → RX → sensing → segmentation → AMC → **adaptive-K Top-K 防禦**」。整條 pipeline **全跑在 RX 收到的 IQ 上** → RX 品質決定下游一切；TX 端擾動可事前數位驗證，Pluto 就夠。**所以錢花在 RX，不是 TX。**

**新增的選型理由：adaptive-K 是頻域防禦。** 它挑 top-K 個 FFT bin；而 **Pluto 的 DC offset / LO 洩漏會在 DC bin 造出強假峰 → 被 Top-K 選中 → 污染防禦評估**（防禦到底在去攻擊，還是去你自家 spur？）。USRP（尤其 N210）DC 校準好、相位雜訊低、spur 少 → FFT 底乾淨、adaptive-K 結果才可信。這比 AMC 或攻擊單獨考量時，更強烈要求一台乾淨的 RX。

| 配置 | 組成 | 花費 | 判定 |
|---|---|---|---|
| **預算地板** | 3× PlutoSDR | ~US$850 | ⚠️ 可跑，但必須軟體 notch DC + 校準量化 + 特徵化 spur，否則 Top-K 被 DC 假峰污染；confound 最多 |
| **推薦（平衡）** | 2× Pluto（TX：合法+攻擊）+ 1× USRP B210（RX） | ~US$2,000 | ✅ 獨立 CFO；B210 RX 相位雜訊/DC/spur 遠優於 Pluto、有 UHD 時戳；仍 12-bit → 量化當校準項揭露 |
| **乾淨科學** | 2× Pluto（TX）+ 1× USRP N210（RX, 14-bit） | ~US$2,200 | ⭐ 同時移除量化 confound + 最乾淨 FFT 底 → adaptive-K 最站得住；審稿人質疑防禦評估時的最佳解 |

**推薦：2× PlutoSDR（TX）+ 1× USRP（RX）**，RX 有預算直接上 N210（14-bit），沒有就 B210。理由：
- **TX 用 Pluto** —— 擾動可事前數位驗證，只是一跳，不必花錢。
- **RX 用 USRP** —— (a) 弱擾動在此重建給 AMC（bit 深度）、(b) **adaptive-K 的 FFT top-K 在此跑，最怕 DC 假峰/spur（Pluto 最弱處）**、(c) achieved PSR 在此量。
- 預算卡死 → 3× Pluto 也能做，但報告需明講已控制三件事：**DC notch、量化底噪校準、spur 特徵化**，否則 adaptive-K 的 recover 會被質疑是去了硬體假峰而非攻擊。

---

## 3. 省錢階梯 — 這是最省的嗎？

Tier 1 還不是地板。真正最省有四階，但**每往下省一階，就放棄一個「真實性」**。照老師在意什麼來選。

| 階 | 做法 | 花費 | 放棄了什麼 |
|---|---|---|---|
| **0. 先別買** | 停在 Stage 1（數位模擬，已完成）+ Stage 2（`.cfile` replay，純軟體） | **US$0** | 沒有真實硬體損傷（CFO/gain/jitter），但科學價值已拿八成 |
| **1. 借不用買** | 用實驗室 / 同學現有 SDR（**最該先做**） | **~US$150 / NT$5k** | 幾乎不放棄（只買被動件） |
| **2. 2 台 PlutoSDR** | Pluto#1 = 合法TX＋RX 同片；Pluto#2 = 攻擊TX | **~US$610 / NT$20k** | 合法路 CFO≈0，但**攻擊者對 RX 仍有獨立 CFO** |
| **3. 3 台 PlutoSDR**（= Tier 1） | 合法 / 攻擊 / RX 三片獨立 | **~US$850 / NT$27k** | 幾乎不放棄，合法路也有真實 CFO |

再往下只能犧牲真實性：
- **1 台 B210 自環**（US$1,500，反而更貴）：雙 TX 共用 LO → 兩路 CFO 相同，對這題意義不大。

> **不要用 8-bit 裝置省錢（撤回先前建議）**：先前提過的「RTL-SDR 當 RX ~US$30」「HackRF 壓到 ~US$300」在本題是**錯的** —— 8-bit 量化會直接淹掉 adversarial 擾動，讓「攻擊失效」變成你自己器材造成的假象（見 2.0 與第 4 節）。承載擾動的收發路徑，位元深度地板是 12-bit（Pluto）。RTL-SDR 只適合做粗略 occupancy 感測，不能拿來收攻擊給 AMC。

---

## 4. TX/RX 可行性：RML 與 adversarial 訊號能否用 PlutoSDR 完整收發

這藏了兩個要分開回答的子問題：**「打得出來嗎」→ 可以**、**「完整送到嗎」→ 不會 bit-exact，而且這正是 Stage 3 要驗的重點**。

### 4.1 打得出來嗎？ → 可以

RML2016 的一個 sample 就是 `[2,128]` 的**基頻複數 IQ**（128 個複數點），adversarial 擾動也是同樣的基頻 IQ。PlutoSDR 的 TX 就是把任意複數 IQ 灌進 DAC，所以：

- RML burst：可直接當 TX 波形送出。
- 乾淨訊號 + adversarial 擾動疊加（對應 `inject_additive_waveform` 的輸出）：也是一段 IQ，一樣送得出去。

### 4.2 完整送到嗎？ → 不會逐點一致，且要分兩種失真

過真實 TX→線纜/空中→RX，收到的 IQ **一定不等於**發出去的。差異分兩類，意義完全不同：

**(A) 你「想要」的損傷 —— 這正是用硬體不用純模擬的理由**
- CFO（Pluto 各自振盪器不同步）、gain error、timing jitter、相位雜訊。
- 這些就是威脅模型要驗的東西（`normalized_cfo` / `gain_error_db` 模擬的就是它）。收到的不一樣，是特意要的。

**(B) 你「不想要」的破壞 —— 可能直接殺掉 adversarial 擾動**
- **12-bit 量化**：Pluto DAC/ADC 只有 12 bit，擾動若小於約 1 LSB → 消失。adversarial 擾動常很小很脆弱，這是最大風險。
- **LO 洩漏 / DC offset**：中心頻率有載波洩漏，burst 在 DC 附近會被污染 → 一定要偏頻發射。
- **PA 非線性、AGC、重採樣**：都會改變波形與絕對振幅。

> **所以「adversarial 攻擊在真實 RF 下還有沒有效」不是已知前提，而是 Stage 3 要回答的問題本身。** 數位模擬 100% 成功，過真實鏈路可能掉很多 —— 這正是有科學價值的地方。

### 4.3 要「確實送到」必須先處理的坑

| 項目 | 為什麼 |
|---|---|
| **重採樣到 Pluto 速率** | RML 是 200 kHz 取樣；Pluto 最低約 520 kSPS → 必須 resample |
| **偏頻發射（別放在 DC）** | 避開 LO 洩漏污染 burst |
| **檢查擾動 ≥ 12-bit LSB** | 否則量化直接吃掉攻擊 |
| **振幅校準到 RadioML 慣例** | `radioml-native` 前處理**不做 rescale** → 硬體 AGC/gain 改了絕對振幅就會讓 AMC 輸入分布飄掉 |
| **128 樣本太短 → 加 preamble / 迴圈** | RX 端要能偵測、對齊這段 burst（回到對齊問題 B） |
| **用 RX 量到的功率反推 achieved PSR** | 對照 `achieved_psr_db`，確認真的打到目標 PSR |

---

## 5. 相關工作、硬體選擇與定位

### 5.1 三層相關工作 — 我們的 gap 在哪

| 層 | 有沒有人做 | 代表工作 |
|---|---|---|
| **(a) 用真實 SDR 打調變訊號做 OTA AMC** | ✅ 成熟、走爛的路 | DeepSig 2018 OTA 版（USRP 發射再錄回）；CNN-LSTM OTA AMC using SDR（arXiv 2511.21040, 2025） |
| **(b) 對 AMC 的 OTA adversarial 攻擊** | 🟡 熱門，但多半只「建模通道」（Rayleigh + path loss），非真發射 | Kim/Sagduyu/Davaslioglu/Erpek/Ulukus 系列（arXiv 2002.02400、2005.05321、2007.16204） |
| **(c) 硬體在環：adversarial 擾動撐過真實 DAC 量化 + literal RML replay** | 🔴 幾乎沒人做 ← **我們的貢獻位置** | RadioShock、Robust Adversarial Attacks（arXiv 2102.00918）較接近，但少；未見 byte-for-byte replay RML2016.10a 過硬體 |

> **定位一句話**：(a) 成熟、(b) 多半只模擬通道、(c) 真的驗證擾動撐過實體量化幾乎沒人做 → 前面談的 12-bit 量化、resample、對齊，正是別人跳過、我們正面處理的 gap。

**References**
- Kim et al., *Over-the-Air Adversarial Attacks on Deep Learning Based Modulation Classifier over Wireless Channels*, CISS 2020 — arXiv:2002.02400
- Kim et al., *Channel-Aware Adversarial Attacks Against Deep Learning-Based Wireless Signal Classifiers* — arXiv:2005.05321
- Kim et al., *Adversarial Attacks with Multiple Antennas Against Deep Learning-Based Modulation Classifiers* — arXiv:2007.16204
- *Robust Adversarial Attacks Against DNN-Based Wireless Communication Systems* — arXiv:2102.00918
- *CNN-LSTM Hybrid Architecture for Over-the-Air AMC Using SDR* — arXiv:2511.21040
- RadioML 2016.10a（DeepSig，O'Shea & West, GRCon 2016）— Zenodo: zenodo.org/records/18397070

> 註：以上為 2026-09 查得之文獻，正式引用前請再核對標題/年份/出處。

### 5.2 為什麼學界多用 USRP —— 不是 bit，是生態

常見誤解是「因為 ADC/DAC bit 和取樣率限制才用 USRP」。**最強反證：DeepSig 與多數人用的入門款 USRP B210 也才 12-bit，跟 Pluto 一樣。** 若純為 bit 就不會選 B210。真正主因依重要性排：

| 原因 | 說明 | Pluto |
|---|---|---|
| **1. 同步生態系** | 10 MHz / PPS、OctoClock、GPSDO、MIMO、多通道相干 | ❌ 原生沒有（OTA/可重現的命脈） |
| **2. UHD 驅動 + 時戳成熟度** | sample-accurate timestamp、timed commands | ⚠️ libiio 較陽春 |
| **3. 校準 + 振盪器品質** | 出廠校準、配 GPSDO 頻率準、相位雜訊低 | ⚠️ TCXO ±25 ppm |
| **4. 可重現性 / 引用慣例** | 審稿人信任、路徑依賴 | — |
| **5. 取樣率範圍 + 瞬時頻寬** | X310 到 100 MHz、可經 DDC 下探 kSa/s | ⚠️ ~520 kHz–56 MHz |
| **6. ADC/DAC 位元深度** | N210/X310 為 14–16 bit（B210 仍 12-bit） | ❌ 12-bit（ENOB ~10–11） |

**位元深度排在最後**：只有跳到 N210/X310 高階才有差，入門 B210 沒贏 Pluto 這點。大部分人選 USRP 是為了 1–4 項（同步、時戳、校準、可重現），不是 bit。

**但對本專案，bit 深度剛好是少數真重要的地方**：一般 AMC 訊號夠強、遠高於底噪，不在乎那 2 bit；而我們的核心是**弱 adversarial 擾動要撐過量化**，這正是 14-bit 有意義的少數場景。因此：
- 只做 cabled、短時間、軟體對齊 → **Pluto 就能滿足**大家用 USRP 的那些理由（同步可省）。
- 想「乾淨分離通道 vs 量化」→ 才把 **RX 升 14-bit N210**，且只當 robustness 附錄，其餘維持 Pluto。

---

## 一句話結論

> 最省 = **先問實驗室有沒有現成 SDR**：有的話只花被動件 ~NT$5k；完全沒有、又要保留「攻擊者獨立 CFO」這個真實條件，硬體地板是 **2 台 PlutoSDR ≈ US$610 / NT$20k**。工作面主打 **B 對齊 ablation、A 的 G1–G5 曲線、C 的 adaptive attack 繞過 Top-K**。

> 註：價格為 2026 年概估，實際依供應商浮動（1 USD ≈ 32 NTD）。
