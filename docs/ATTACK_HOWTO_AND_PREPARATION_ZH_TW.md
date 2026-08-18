# 攻擊手法與準備清單：Threat 怎麼攻擊，需要準備哪些東西

## 本文件目的

回答兩個具體問題：

1. **攻擊者實際上怎麼攻擊這個系統**——不只是「打 AWN 讓它分錯」，而是攻擊整條
   `能量偵測 → 事件建立 → 視窗選擇 → Top-K → AMC` 管線上每一個可被下手的點。
2. **要做這些攻擊（或驗證它們），需要準備哪些東西**——從純數位模擬、`.cfile`
   重播、到雙 SDR 纜線注入與屏蔽 OTA，逐階段的軟硬體、資料、校準與統計前置。

依據程式：`src/sensing/rf_attack_sim.py`、`src/adapters/attack_adapter.py`、
`src/sensing/energy_detection.py`、`src/sensing/segmentation.py`。威脅模型全文見
`docs/THREAT_MODEL_AND_COMPONENT_STATUS_ZH_TW.md`，模擬協定見
`docs/ATTACK_SIMULATION_PROTOCOL_ZH_TW.md`。本文是這兩者的「操作面」濃縮。

---

## 一、先搞清楚攻擊者能做什麼、不能做什麼

這是整份 report 最重要的前提，寫論文或報告時第一句就要講清楚，否則所有攻擊
數字都會被高估。

### 1.1 接收訊號模型

接收器在單一預先通道化通道看到的是**疊加**訊號：

```text
y[n] = h_s * s[n-τ_s]  +  h_a * a[n-τ_a] · e^{j(2π·Δf·n + φ)}  +  w[n]
        └─ 合法訊號 ─┘     └──────── 攻擊者只能動這一項 ────────┘   └ 雜訊 ┘
```

- 攻擊者**只能加入自己的波形 `a[n]`**，透過控制它的頻率、功率、時間、duty cycle。
- 攻擊波形會經過**攻擊者自己的通道 `h_a`**、獨立到達時間 `τ_a`、載波頻偏 `Δf`、
  未知相位 `φ`、增益誤差——這些都不是攻擊者能精確控制的。
- 攻擊者**不能**：直接覆寫接收端 tensor、使用負能量、和合法發射器做完美相位相消。

### 1.2 三層攻擊者（決定你在測哪一種）

| 等級 | 能力 | 用途 | 現況 |
|---|---|---|---|
| **A0 數位注入** | 直接改離線 IQ tensor、知道權重、拿得到精確梯度、不受通道限制 | 演算法上界、比較 FGSM/PGD/CW | **已實作**（`attack_adapter.py`） |
| **A1 受控 RF** | 附近用商用 SDR 加性發射，知道系統設計或有代理模型，能估自己的通道，但不能改主機、不假設精確同步 | 主要且較真實的威脅模型 | 威脅模型已定義，正式流程未實作 |
| **A2 查詢式黑箱** | 不知權重，只靠告警/API/操作員反應等有限回饋，受查詢與發射時間限制 | transfer / universal / 黑箱最佳化 | 未實作 |

> **關鍵誠實聲明**：目前 `AttackAdapter` 的 FGSM/PGD/CW 是 **A0 receiver-side tensor
> attack**，即使用 real AWN backend 也一樣。它證明的是「數位輸入空間存在弱點」，
> **不能**直接宣稱 over-the-air 可行。要變成 A1，可優化變數必須限制成攻擊者的
> 發射波形，並在 loss 裡包含通道、CFO、timing、gain 與整條 hard pipeline。

---

## 二、攻擊者怎麼攻擊：七種手法

攻擊面不只 AWN。下表先總覽，後面逐一展開機制、針對的程式弱點與成功條件。

| # | 攻擊 | 打哪一層 | 想造成的後果 | 現況 |
|---|---|---|---|---|
| G1 | Noise-floor poisoning | 能量偵測門檻 | 弱合法事件被漏掉（可用性/完整性） | 注入器已備，流程未跑 |
| G2 | Phantom occupancy | 事件建立 | 假事件洪泛，耗盡推論/儲存/人力 | 注入器已備，流程未跑 |
| G3 | Event bridging | region 合併 | 兩事件被錯誤合併成一個 | 可由現有 detector 模擬 |
| G4 | Boundary extension | 事件邊界/視窗對齊 | 起訖偏移、128 視窗錯位 | 可由離線 IQ 模擬 |
| G5 | Max-energy window hijacking | 視窗選擇器 | selector 選到攻擊波形而非合法訊號 | 分段已實作，針對攻擊未做 |
| — | AMC evasion (FGSM/PGD/CW) | AWN 分類器 | 調變標籤不可信 | **已實作（A0）** |
| — | Defense-aware adaptive | 完整管線 | 繞過 Top-K / 拒判 DoS | 未實作 |

以下 G1–G5 的 sweep 範圍、成功判準都出自 `ATTACK_SIMULATION_PROTOCOL_ZH_TW.md`
第 5 節；每一種都**必須從 detection 重跑完整管線**，不能用 oracle region 直接切窗。

### G1 — Noise-floor poisoning（污染雜訊底）

**怎麼打**：長時間發射高 duty-cycle 的複數噪聲／OFDM-like／多 tone 波形，蓋住
目前這個通道。

**打中的程式弱點**：`energy_detection.py` 的門檻是**整段 capture 的中位數功率**
乘上倍率：

```text
noise_floor = median(smoothed_power)
threshold   = threshold_factor × noise_floor
```

中位數對單一 burst 穩健（這是設計優點），但也意味著攻擊者只要**污染夠高比例
的 capture**（>50%）就能把中位數、進而把門檻整個抬高，讓原本剛好過門檻的弱合法
訊號沉到門檻以下、偵測不到。

**sweep**：`duty cycle ∈ {0.4, 0.5, 0.6, 0.8, 1.0}`、`PSR ∈ {-18…-3} dB`。

**成功條件**：`clean_detected == True` 且攻擊後 `event_IoU < 0.1`。
（不能把本來就偵測不到的低 SNR 樣本算成功。）

**必須比較的 baseline**：同 PSR 的連續高斯 jammer、同能量低 duty burst jammer、
單 tone。若精心設計的波形沒贏過這些，就只能叫 **jamming**，不能叫 adversarial。

### G2 — Phantom occupancy（假佔用洪泛）

**怎麼打**：在沒有合法活動的時間，反覆發射**剛好超過門檻**的短 burst。

**後果**：每個假 burst 都觸發一次事件建立 → 一次 AWN 推論 → 一次 IQ 保存 →
一個分析佇列項目。這是**資源耗盡型 DoS**，不需要動到任何合法訊號。

**sweep**：`attack_len / energy_window ∈ {0.5,1,2,4}`、
`gap / merge_gap ∈ {0.5,1,2,4}`、`PSR vs 背景 ∈ {3,6,9,12,18} dB`。

**成功條件不是「有一個 false positive」**，而是**資源放大率**：

```text
amplification = 防禦端 compute 或 storage 成本 / 攻擊者 airtime 或 energy
```

要量每分鐘假事件數、每分鐘額外 AMC 次數、每分鐘保存 bytes、分析佇列長度。

### G3 — Event bridging（事件橋接）

**怎麼打**：先有兩個乾淨時可分離的合法 burst，攻擊者**只在兩者中間的安靜間隔**
補入 tone／noise，不需降低任何合法能量。

**打中的程式弱點**：`merge_close_regions()` 會把間隔 ≤ `merge_gap` 的區域合併。
攻擊者用能量把 gap「填平」，讓兩個獨立事件被判成一個。

**sweep**：`clean gap ∈ {merge_gap+1, 2·merge_gap, 4·merge_gap}`、
`bridge on-time / gap ∈ {0.25…1.0}`、`bridge PSR ∈ {-18…-3} dB`。

**成功條件**：`clean_region_count == 2` 且 `attacked_region_count == 1` 且合併後
區域橫跨兩個合法 burst。另報合併後 max-energy 選到哪個視窗、兩個合法 burst 的
retained-sample ratio、AMC 決策是否改變。

### G4 — Boundary extension（邊界延伸）

**怎麼打**：在合法 burst **前加 prefix、後加 suffix**（加性 RF，不是改寫真值邊界）。

**打中的程式弱點**：這正是本專案 round 9 診斷過的**對齊敏感性**（見
`parameter_validation.md` 第 18 節）——能量偵測的平滑本來就會把區域邊緣外擴
53–61 samples；攻擊者故意加前後綴會進一步把偵測到的起訖點推開，使 `naive` 切法
的 128-sample 視窗更嚴重地錯位、只重疊到部分真實 burst → 餵給 AWN 的是
out-of-distribution 輸入。

**sweep**：`prefix/suffix ∈ {16,32,64,128,256} samples`、`PSR ∈ {-18…-3} dB`。

**成功條件**：start/end boundary error、event IoU、naive 視窗中合法樣本比例下降、
AMC clean→attacked 決策改變。**必須同時對 `naive` 與 `max-energy` 測**，且
`naive` 要保持 byte-compatible 實作，不能為了好看改它。

### G5 — Max-energy window hijacking（最大能量視窗劫持）

**怎麼打**：在**同一個偵測區域內**放一個短、局部能量很高的 pulse。目標不是消滅
合法訊號，而是讓：

```text
mean|W_攻擊|² > mean|W_合法|²
```

**打中的程式弱點**：`select_aligned_segments()` 的 `max-energy` 策略每區域只選
**平均功率最高**的那個視窗。攻擊者用一個強 pulse 就能讓 selector 選到自己主導的
視窗，而不是合法 burst。這證明了 max-energy 對齊雖然改善乾淨資料的準確率，卻
**新增了一個可被攻擊的 selector surface**——是設計取捨的直接代價。

**sweep**：`pulse len ∈ {8,16,32,64,128}`、`pulse PSR ∈ {-18…0} dB`、
`pulse offset ∈ 區域內所有可行位置`。

**成功條件**：乾淨時選到的視窗與合法 burst 重疊、攻擊後選到的起點不同、且被選
視窗中攻擊者樣本佔比 ≥ 0.5。

### AMC evasion — FGSM / PGD / CW（唯一已跑出正式結果的）

**現況**：這是 A0，`attack_adapter.py` 已實作、正式跑過（Phase 3，N=3960）。
整體攻擊成功率 **82.78%**，強度 CW (0.928) > PGD (0.886) > FGSM (0.749)。CW 擾動
量最小（mean Linf ≈ 0.0005）卻最致命，符合最佳化式攻擊特性。

**局限**：這是在**已切好、已對齊**的 128-sample tensor 上直接加 δ，繞過了 sensing
與 Top-K。它是「防禦無感（oblivious / defense-unaware）」攻擊，不是完整管線白箱。

### Defense-aware adaptive（尚未實作，最強但最難）

要主張對完整防禦的安全性，攻擊者要對整條路徑最佳化：

```text
能量偵測 → region merge/filter → 視窗選擇 → Top-K/Adaptive-K → AWN → 信心/棄權/事件彙整
```

中間有不可微分的硬選擇（門檻、合併、Top-K），需要 **BPDA／STE 或 soft surrogate**
算梯度，但 forward pass 必須用真正的 hard pipeline；有隨機性時用 **EOT** 對
CFO/phase/timing/gain/channel 取期望，最後用**沒參與最佳化的 held-out
transformation** 評估成功率。

---

## 三、需要準備哪些東西

分四個階段，逐階段列出軟體、資料、硬體、校準與統計的前置。**建議嚴格照順序做**，
不要跳過數位模擬直接上硬體。

### Stage 0 — 通用前置（做任何攻擊實驗前）

**軟體/環境**
- Python 環境含 **torch**（真實 AWN/attack backend 需要；dummy fallback 只能 dry run）。
- 本 repo 的 `src/`，特別是 `rf_attack_sim.py` 的三個注入器：
  `generate_unit_power_complex_noise()`、`generate_unit_power_tone()`、
  `inject_additive_waveform()`。
- pinned submodules：`external/AWN`、`external/adversarial-rf`（模型與 CW/PGD/FGSM、
  `fft_topk_denoise`）。

**資料**
- RML2016.10a 資料集（`RML2016.10a_dict.pkl`，~640MB）作為合法 burst 來源。
- 用 `radioml_source.py` 把 burst 嵌入較長噪聲串流，取得**已知真值位置**
  （`true_start`/`true_end`）——這是所有 sensing 成功判準的基礎。

**每筆結果必須記錄的欄位**（`ATTACK_SIMULATION_PROTOCOL` 第 10 節，節錄關鍵）
```text
attack_name, attack_seed, signal_seed, channel_seed
true_burst_start/end, attack_start/end, attack_duty_cycle
attack_target_psr_db, attack_achieved_psr_db      ← 兩個都要，不能只存發射端 amplitude
cfo, phase, timing_error, gain_error_db, channel_id
energy_window, threshold_factor, min_region_len, merge_gap, alignment_policy
clean/attacked/defended_label + confidence
Pd, Pfa, event_iou, boundary_error
abstain_status, backend_status                    ← real backend 失敗必須記 failure，不可混入 dummy
attack_success, success_definition_version
```

**功率預算觀念（PSR 是主指標，不是發射端 amplitude）**
```text
PSR_dB = 10·log10( P_攻擊 / P_合法 )
建議第一輪 sweep：PSR ∈ {-30,-24,-18,-15,-12,-9,-6,-3,0} dB
```
`P_合法` 在合法訊號的真值 active interval 上算，`P_攻擊` 在攻擊實際發射 interval 上算。

### Stage 1 — 純數位加性模擬（現在就能做，零硬體）

**目標**：證明攻擊目標、功率計算、完整 sensing pipeline 行為正確。**不能宣稱 OTA 可行。**

**要準備/實作**
- 用 `inject_additive_waveform()` 把攻擊波形以指定 PSR 疊到 `clean_iq`，並套上
  CFO、phase、gain error（函式已支援）。
- 把 `attacked_iq` **從 detection 重跑完整管線**（關鍵：不可用 oracle region 直接切窗）。
- G1–G5 各自的成功判準與 sweep（見第二節）。
- **必要 baseline**：同功率 noise jammer、tone jammer、random burst——沒贏過就不是
  adversarial。
- **隨機化範圍**（held-out 用）：
  ```text
  phase        ~ U(0, 2π)
  normalized CFO ~ U(-0.01, 0.01) cycles/sample
  timing error ~ DiscreteU(-64, +64) samples
  gain error   ~ U(-3, +3) dB
  channel delay~ DiscreteU(0, 32) samples
  channel      = AWGN + optional 短 Rician/Rayleigh taps
  ```

**統計前置（讓結果「可靠」而非只對一個 seed 成功）**
- **Calibration set / Evaluation set 分離**：進 evaluation 後不得再依結果調 budget。
- 每個 `(attack, modulation, SNR, PSR)` condition：合法樣本 seeds ≥ 30、
  攻擊波形 seeds ≥ 10、channel/CFO/phase draws ≥ 10。
- 預先註冊可靠度定義：
  ```text
  Reliable@PSR：held-out ASR ≥ 0.90 且 95% Wilson 下界 ≥ 0.80
  ```
  並報告達標的**最小 PSR**；若任何 PSR 都沒達標，誠實寫 not reliable。
- 報**完整曲線**（ASR vs PSR、Pd/Pfa vs PSR、IoU vs PSR、ASR vs CFO/timing/gain、
  資源放大率），不能只報最佳點。

### Stage 2 — `.cfile` 重播（細節版）

**目標**：把合法＋攻擊混合波形存成 complex64 `.cfile`，從正式 capture loader 重播，
驗證整條 detection→decision 在「檔案輸入」路徑上行為正確。這一步把攻擊從
「in-memory numpy 陣列」升級成「和真實擷取相同的 on-disk 格式」，是接真實 SDR 前
唯一能在桌機上做完的整合測試。**仍不能宣稱獨立 RF 發射可行**（波形仍是自己合成的）。

#### 2.1 目前 loader 的實際狀態（別高估它）

`iq_source.py:load_iq_from_file` 目前**只有兩行實質邏輯**：

```python
iq = np.fromfile(path, dtype=np.complex64)   # 假設 host-native endianness
if iq.size == 0: raise ValueError(...)        # 只擋空檔
```

`validate_iq` 也只檢查「是複數」「是一維」，dtype 不符還會**靜默 cast**。也就是說：
**endianness、NaN/Inf、振幅範圍、sample rate、中心頻率、clipping 全部沒驗證**，且
整個檔案一次讀進記憶體、沒有 chunking。這些都要在 Stage 2 補上。

#### 2.2 要準備/補的程式缺口（依做的順序）

1. **接線**：把 `load_iq_from_file` 接進 `ExperimentConfig` 與 `src/utils/pipeline.py`，
   讓正式入口 `experiments/run_full_experiment.py` 能吃 `--iq-source cfile --iq-file
   path.cfile`（目前只吃 synthetic / radioml）。
2. **capture metadata contract**：`.cfile` 只有裸 IQ，沒有 header。要在旁邊放一個
   sidecar（例如 `path.cfile.json`）記錄，並在載入時強制檢查：
   ```text
   sample_rate_hz, center_freq_hz, dtype(="complex64"),
   byte_order(="little"/"big"), hardware, gain_db, capture_utc, n_samples_expected
   ```
   沒有 sample rate，pipeline 裡的 `energy_window`、`min_region_len`、128-sample 視窗
   就只是「樣本數」，無法換算成真實時間/頻寬——這是接硬體前必須先定義的語意。
3. **輸入邊界驗證**（補進 `validate_iq` 或新的 `validate_capture`）：
   - **endianness**：`np.fromfile` 用 host-native；若 sidecar 標的位元序和本機不同，
     要 `.byteswap()`。GNU Radio `gr_complex` 是 little-endian float32×2。
   - **NaN/Inf**：`np.isfinite` 全檢查，非有限值直接 fail（`rf_attack_sim.py` 已對
     注入端這樣做，載入端要一致）。
   - **clipping / 振幅**：統計 `|iq|` 的 max 與接近 full-scale 的樣本比例，clip 太多
     代表 RX 增益設太高，要記 warning 甚至 fail。
   - **長度**：對照 sidecar 的 `n_samples_expected`，抓 dropped samples。
4. **chunking / rolling buffer**：長檔不能一次讀爆記憶體。改成分段讀＋**跨 chunk 保留
   detector 狀態**（否則事件會被 chunk 邊界切斷）。這是邁向 streaming 的前置。
5. **fail-closed backend**：real AWN 載入失敗**必須停**，不可 fallback 成 dummy random
   logits；每筆結果記 `backend_status`，dummy 結果**絕不可**混進 real-backend 表。

#### 2.3 怎麼產生 Stage 2 的測試 `.cfile`

```python
# 1) Stage 1 已驗證的混合波形（合法 burst + 以 PSR 注入的攻擊）
attacked_iq, _, meta = inject_additive_waveform(clean_iq, atk, ...)
# 2) 存成 GNU Radio 相容的 complex64 .cfile（interleaved I,Q float32, little-endian）
attacked_iq.astype(np.complex64).tofile("case_g1_psr-9.cfile")
# 3) 同時寫 sidecar，記 true_burst_start/end 與所有 2.2 的 metadata
```

#### 2.4 驗收判準（round-trip 一致性）

- **同陣列一致性**：同一份 IQ 走 synthetic path 與走 `.cfile` path，偵測區域、選到的
  視窗、clean/attacked/defended label 與 confidence 應**逐 segment 完全一致**（byte
  一致最好）。不一致代表載入/前處理引入了差異，要先修好再往下。
- **格式健全性**：故意餵 wrong-endian、含 NaN、被 clip、長度不符的檔，確認每種都被
  對應的驗證擋下並給出可讀錯誤，而不是靜默 cast 或跑出垃圾結果。
- G1–G5 的成功判準在 `.cfile` path 上重現 Stage 1 的結論。

---

### Stage 3 — 雙 SDR 纜線注入（細節版）

**目標**：第一個能支持「**獨立攻擊發射器物理可行**」的階段。攻擊波形不再是自己疊
上去的數字，而是由**第二台 SDR 真的發射**、經纜線疊加後被 RX 收下——這才開始承受
真實的 CFO、trigger jitter、增益漂移與前端響應。**用纜線而非天線**，避免非法輻射，
也讓通道乾淨可控。

#### 3.1 拓樸與硬體清單

```text
合法 SDR TX ──[ 固定衰減器 ]──┐
                              [ 2:1 合路器 ]──[ 選配可變衰減器 ]── SDR RX
攻擊 SDR TX ──[ 固定衰減器 ]──┘
                              └─（合路器第 3 埠可接功率計/頻譜儀做校準）
```

| 項目 | 數量 | 用途/規格要點 |
|---|---|---|
| SDR（TX 能力） | 2 | 合法＋攻擊各一。USRP（UHD，時序最穩）、PlutoSDR（便宜）、HackRF（半雙工，注意不能同時收發） |
| SDR（RX） | 1 | 可與其中一台共用機型；建議 RX 用 USRP 求穩定 |
| 固定衰減器 | 每條 TX 線 1 個 | **保護 RX 前端**，把 TX 功率壓到 RX 安全輸入範圍（通常 20–40 dB） |
| RF 合路器 / power combiner | 1 | 頻段要涵蓋實驗中心頻率；記其插入損耗 |
| 可變衰減器（選配） | 1 | 放在 RX 前，方便掃 SNR/SJR 而不用改 TX 增益 |
| 功率計或頻譜分析儀 | 1 | **校準 achieved PSR/SJR 的關鍵**，不能只信軟體設定值 |
| SMA 纜線、terminator | 數條 | 未用埠要接 50Ω terminator |
| 同步參考（選配但強烈建議） | — | 共用 10 MHz + PPS（USRP 可），讓 CFO/timing 可控可量 |

#### 3.2 需要自建的軟體（repo 目前沒有）

- **UHD/GNU Radio TX flowgraph**：把 Stage 1/2 的合法與攻擊波形檔透過 File Source →
  USRP Sink 發射。攻擊端要能設定中心頻率、增益、發射起訖時間、duty cycle。
- **UHD/GNU Radio RX flowgraph**：USRP Source → File Sink（`gr_complex`）錄成 `.cfile`，
  直接餵給 Stage 2 已驗證的 loader → 完整 pipeline。
- **時序控制**：兩台 TX 的相對發射時間要能設定（用共用 PPS 觸發，或接受隨機
  `τ_a` 並記錄）。目前 repo 完全沒有硬體控制程式，這些都要新寫。

#### 3.3 校準流程（不可省，順序很重要）

1. **各別量測**：只開合法 TX，用功率計量 RX 端合法訊號功率 `P_s`；只開攻擊 TX，量
   `P_a`。兩者相除得**實際 PSR**，和軟體 target 對照、記下轉換關係。
2. **量這幾台實體 SDR 的真實不理想性**，用量測分布**取代 Stage 1 的任意隨機範圍**：
   - **CFO（Δf）**：兩台 SDR 本振不同源時的頻偏；不共用參考時可能遠大於 Stage 1 假設
     的 ±0.01 cycles/sample。
   - **trigger jitter / timing offset**：每次發射的起始時間抖動。
   - **gain variation / 溫漂**：增益設定與實際輸出的偏差。
   - **前端響應**：帶內平坦度、濾波器形狀。
3. **尺度對齊（最容易被忽略、最會掉準）**：RX 收到的 IQ 振幅取決於 RX 增益設定、ADC
   full-scale、UHD 縮放慣例——**這條尺度和 AWN 訓練用的 RadioML 尺度幾乎肯定不同**。
   因為 `radioml-native` 前處理**完全不重新縮放**，尺度對不上會直接讓 AMC 掉準。做法：
   量一段已知合法訊號，找出把 RX 尺度映回 RadioML 尺度的校準係數，或改用
   `legacy-unit-power`（每段正規化）並驗證兩者差異。
4. **安全**：確認經衰減後 RX 輸入功率在安全範圍內**才**接上，避免 TX 直接燒 RX 前端。

#### 3.4 掃描維度與驗收

- **掃描**：SNR、SJR、**achieved** PSR（用功率計確認，非 target）、CFO、timing offset、
  發射功率、duty cycle。
- **held-out**：calibration 找到的攻擊參數，要用**新的合法樣本、新的發射 seed、不同
  時間/溫度下的 channel realization** 評估，不能拿調參用的那批報成績。
- **驗收**：Stage 1 數位模擬預測的主要效果（例如 G1 的 noise-floor shift、G5 的視窗
  劫持、AMC 決策改變）要在纜線實測中**同方向重現**；同時報和同功率 noise/tone
  jammer baseline 的比較。若纜線上做不出來，就要回頭檢查是 CFO/timing/尺度哪個
  假設在 Stage 1 太樂觀。

> 通過 Stage 3 才第一次有資格說「這個攻擊在有獨立發射器、真實硬體不理想性下仍成立」。
> 但**纜線 ≠ OTA**：多徑、天線、位置、遮蔽物的變異要留到 Stage 4 屏蔽環境才算。

### Stage 4 — 屏蔽 OTA（最後才做，且受法規限制）

**環境/合規**
- 屏蔽箱或**合法授權**環境。**公開頻段不得直接發射攻擊波形**，這是硬性法規紅線。
- 天線、位置、遮蔽物、通道變異的掃描設計。

**目標**：至少一部分主要效果要在 OTA 重現，才能把攻擊稱為「實際且可靠」。

---

## 四、投稿/報告前的最小判準（避免過度宣稱）

出自協定第 11 節，逐條都是「不做到就不能講的話」：

1. 攻擊從 raw IQ / RF 注入點進入，**重跑 detection→decision 完整管線**。
2. 報告 achieved PSR、duty cycle、CFO、timing、gain/channel 不確定性。
3. 和同功率 noise/tone/random-burst baseline 比較。
4. 用 held-out seeds 與 held-out channel transformation。
5. 報完整 success curve 與信賴區間。
6. 主要效果在**纜線雙 SDR** 重現。
7. 至少一部分結果通過**屏蔽 OTA**。
8. **不把 A0 tensor attack 說成 over-the-air attack。**
9. **不把 perfect cancellation 當成一般攻擊能力。**
10. 同時呈現成功與失敗區域，明確標示 claim boundary。

---

## 五、一頁總結（口頭報告用）

> 攻擊者是**附近持商用 SDR 的加性 RF 發射者（A1）**：他能控制自己波形的頻率、
> 功率、時間、duty cycle，知道系統設計，但**不能改接收端 tensor、不能完美同步相消**。
> 除了用 FGSM/PGD/CW 讓 AWN 分錯（目前只做到 A0 tensor 層級，82.78% 成功），他更
> 便宜的打法是攻擊 sensing：**污染雜訊底**藏掉弱訊號、**假佔用洪泛**耗盡資源、
> **橋接**讓兩事件誤併、**邊界延伸**讓 128 視窗錯位、**劫持 max-energy selector**
> 選到自己的波形。
>
> 要驗證這些，準備順序是：**數位加性模擬（現在就能做，需 torch＋RadioML＋PSR 統計
> 設計）→ `.cfile` 重播（需先補 loader 接線與 metadata 驗證）→ 雙 SDR 纜線注入
> （需 SDR＋衰減器＋合路器＋功率計校準）→ 屏蔽 OTA（需合規環境）**。每一階段都要
> 記 achieved PSR 而非發射端 amplitude，都要跟 jammer baseline 比，都要用 held-out
> 評估——否則只能叫 jamming，不能叫 adversarial attack。

---

## 六、程式與文件依據

- 攻擊注入器：`src/sensing/rf_attack_sim.py`
- A0 分類器攻擊：`src/adapters/attack_adapter.py`
- 能量偵測（G1/G3 弱點）：`src/sensing/energy_detection.py`
- 分段/視窗選擇（G4/G5 弱點）：`src/sensing/segmentation.py`
- 完整威脅模型：`docs/THREAT_MODEL_AND_COMPONENT_STATUS_ZH_TW.md`
- 攻擊模擬與可靠性協定：`docs/ATTACK_SIMULATION_PROTOCOL_ZH_TW.md`
- 部署缺口盤點：`docs/DEPLOYMENT_READINESS.md`
- 對齊敏感性根因（G4 依據）：`docs/parameter_validation.md` 第 18 節
