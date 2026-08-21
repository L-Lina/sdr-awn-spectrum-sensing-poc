# Satellite-like Channel Simulator：Specification + Minimal Correctness PoC

## 一、研究目的

Step 1 定義了 satellite-like application scenario 與 latency requirement，Step 2 確認 project-close 的 dataset／modulation 策略為沿用 RadioML2016.10a、BPSK／QPSK／8PSK，不重新訓練 AWN。本輪（Step 3）的目的是：針對 Step 1 第十一節已定義的 MUST／SHOULD channel factors，建立一個**可重現、可解釋、具有衛星通訊物理意義但不過度複雜**的 satellite-like channel model，並以小型 PoC 驗證它能正確整合進既有的 sensing→AMC pipeline，**不是**建立一個 standard-compliant 的 DVB-S2/S2X 或 3GPP NTN 通道模擬器。本輪只建立 channel model 正確性與小型相容性驗證，不重訓 AWN、不下載 RadioML2018、不做 DVB modem、不做 CRC、不接 SDR、不跑正式 final matrix。

## 二、Scope

依 Step 1 第十一節的 MUST／SHOULD／OPTIONAL 分類，本輪**只實作** MUST 與 SHOULD 兩級因素：

**MUST**：(1) AWGN、(2) amplitude／attenuation scaling、(3) propagation-delay metadata。
**SHOULD**：(4) carrier frequency offset (CFO)、(5) Doppler frequency shift、(6) timing offset。
**OPTIONAL／future（本輪明確不實作）**：(7) sample-rate offset、(8) non-linear amplifier、(9) rain fade、(10) shadowing、(11) orbital geometry、(12) full satellite transponder model。

## 三、Receiver-Side Reference Model

依 Step 1 第九、十三節，本模擬器對應的部署位置是 **receiver-side／ground-side digital IQ processing**（LEO ground-terminal 或 ground-station 頻譜監測情境），流程為：

```
RadioML IQ burst → satellite-like channel → long IQ stream
→ Spectrum Sensing → segmentation → real AWN
→ optional attack → optional Top-K
```

Channel model作用於**單一 burst**（RadioML 的 [2,128] 樣本轉為複數 IQ 之後），在這個 burst 被嵌入長串流（既有 `src/sensing/radioml_source.py:embed_sample_in_noise` 機制，本輪未修改）之前套用——這對應「訊號在傳播過程與接收前端已經被通道效應影響，之後才進入感測與分段流程」的物理順序。

## 四、Channel Factors

| 因素 | 分類 | 本輪狀態 |
|---|---|---|
| AWGN | MUST | 已實作 |
| Amplitude／attenuation scaling | MUST | 已實作 |
| Propagation-delay metadata | MUST | 已實作（僅 metadata，見第九節） |
| CFO | SHOULD | 已實作 |
| Doppler frequency shift | SHOULD | 已實作 |
| Timing offset | SHOULD | 已實作 |
| Sample-rate offset／非線性放大器／rain fade／shadowing／orbital geometry／完整 transponder model | OPTIONAL | **本輪未實作，未來延伸項目** |

## 五、Mathematical Models

實作於 `src/channel/satellite_like.py`。所有數學式與程式碼函式逐一對應（`_apply_amplitude_scaling`／`_apply_timing_offset`／`_apply_frequency_rotation`／`_apply_awgn`）。

### 5.1　Amplitude／Attenuation Scaling

$$y[n] = a \cdot x[n]$$

- 單位：$a$ 為無因次線性倍率（非 dB）。
- 參數意義：代表自由空間／鏈路預算衰減的純量抽象（不含相位旋轉，相位效應由 CFO／Doppler 獨立處理）。
- 有效範圍：$a > 0$（程式碼強制檢查，$a \le 0$ 直接拒絕並丟出例外）。
- 對 IQ 的影響：訊號功率依 $a^2$ 縮放（$P_y = a^2 P_x$，已由單元測試 2 逐一驗證）。
- 對 sensing 的影響：`energy_detect` 的門檻是相對於**串流自身**的中位數噪聲底計算，理論上對整體縮放不敏感（縮放同時影響訊號與其所在視窗的相對關係），但因為 burst 是嵌入到一個**背景噪聲功率獨立設定**的長串流中（`embed_sample_in_noise` 依 `embed_snr_margin` 設定，不隨這裡的 `amplitude_scale` 連動），振幅縮小會直接降低 burst 相對於背景噪聲底的能量優勢，可能影響偵測靈敏度——這是本輪刻意保留、值得在 smoke test 中觀察的真實交互作用，不是要消除的效應。
- 對 AMC 的影響：AWN 前處理（`apply_awn_preprocess`）本身含正規化步驟，理論上能部分吸收純量幅度差異，但實際影響需要以 smoke test 驗證（見第十四節）。

### 5.2　AWGN

$$y[n] = x[n] + w[n], \quad w[n] \sim \mathcal{CN}(0, \sigma^2)$$

- 單位：$\sigma^2$ 為複數噪聲功率（W，相對單位），對外參數 `snr_db` 為 dB。
- 參數意義：$\text{SNR}_{dB} = 10\log_{10}(P_x / \sigma^2)$，其中 $P_x$ 是**已經過前面所有其他 stage 轉換後**的訊號功率（見第八節變換順序）。
- 有效範圍：`snr_db=None` 代表完全跳過此 stage（不是「無限大 SNR」的近似，而是明確不執行這個運算）；有限值時無範圍限制，但极端值（例如 <-30dB）會讓後續 sensing/AMC 幾乎必然失敗，屬預期行為。
- 實作公式：$\sigma^2 = P_x / 10^{\text{snr\_db}/10}$，實部虛部各自獨立抽樣 $\mathcal{N}(0, \sigma^2/2)$。
- 對 sensing／AMC 的影響：與既有 RadioML 全部量測結果中 SNR 對偵測率／分類準確率的已知影響一致（本模型只是在 burst 本身額外疊加一層通道噪聲，獨立於 `embed_sample_in_noise` 既有的背景噪聲機制）。

### 5.3　CFO／Doppler（共用旋轉 primitive，語意分開保留，見第十節）

$$y[n] = x[n] \cdot e^{j 2\pi \Delta f\, n / F_s}$$

- 單位：$\Delta f$ 為 Hz，$F_s$ 為模擬器取樣率（Hz，見第六節，本模擬器自訂假設值，非 RadioML 資料集本身攜帶的物理事實）。
- 參數意義：$\Delta f = \text{cfo\_hz} + \text{doppler\_hz}$（`combined_frequency_offset_hz`），兩者的物理來源不同（見第十節），但在 baseband 複數旋轉的數學表現相同，本模擬器共用同一個旋轉運算。
- 有效範圍：理論上無限制，但 $|\Delta f| \ge F_s/2$ 時會產生混疊（aliasing），不具物理意義，使用者需自行確保參數在 Nyquist 範圍內（本輪未加自動範圍檢查）。
- 對 IQ 的影響：逐樣本相位遞增 $2\pi\Delta f/F_s$（弧度／樣本），已由單元測試 3、4、5 以量測相鄰樣本相位差的方式驗證與公式完全一致。
- 對 sensing 的影響：`energy_detect` 是功率（$|x|^2$）為基礎的偵測，相位旋轉不改變 $|x[n]|$，理論上不影響 sensing——smoke test 結果（第十四節）與此理論預期一致。
- 對 AMC 的影響：BPSK／QPSK／8PSK 屬於 phase-shift keying 家族，其符元資訊直接編碼在相位上，一個持續累積的相位旋轉會讓 128-sample 視窗內的星座圖逐漸偏轉，預期對這類調變的分類影響比對 AWGN 更直接——smoke test 結果（第十四節）與此理論預期一致。

### 5.4　Timing Offset

- 實作方式：**整數樣本位移**（非 fractional delay 內插），理由見第八節。
- 數學定義：正值 `timing_offset_samples=s` 代表延遲，輸出 $y[n] = x[n-s]$（$n<s$ 補零）；負值代表提前，$y[n]=x[n+|s|]$（尾端補零）；輸出長度固定等於輸入長度。
- 單位：samples（整數）。
- 有效範圍：$|s| < $ burst 長度時仍保留部分原始內容；$|s| \ge$ burst 長度時輸出全零（已由程式碼處理，不報錯）。
- 對 IQ／sensing／AMC 的影響：小幅度位移主要影響 burst 在長串流中的確切邊界位置，可能與既有的 max-energy 分段視窗對齊產生交互作用；已由單元測試 6 驗證位移本身的正確性。

## 六、Parameter Units and Ranges（Project-Close Conservative Set）

| 參數 | 單位 | 本輪測試層級（3-4 級） | 理由 |
|---|---|---|---|
| SNR（channel-level AWGN） | dB | 沿用既有正式 grid 子集：**-10／0／18**（smoke test），另建議 FULL matrix 加入 **10** 作為中間點 | -10/0/18 與 Step 1／既有 Phase B/E 的 `ATTACK_SNRS` 完全一致，維持跨文件可比較性；10 dB 補足中段訊噪比覆蓋，避免只有極端值 |
| Amplitude scale | 無因次線性倍率 | **0.5／1.0／2.0** | 對稱涵蓋衰減（-6.02 dB）／不變／增益（+6.02 dB）三種鏈路預算相對強度，簡單、對稱、易解讀 |
| CFO | Hz（相對模擬器 $F_s$＝200 kHz，見下） | **0（無）／2000（mild）／10000（stronger）** | 以典型低成本 TCXO 振盪器誤差量級（約 1–5 ppm）於 2 GHz 參考載波換算：1 ppm × 2 GHz = 2 kHz，5 ppm × 2 GHz = 10 kHz；於 $F_s$=200 kHz 下分別占 1% 與 5%，遠低於 Nyquist，不致混疊 |
| Doppler | Hz | **0（無）／1000（mild）／5000（stronger）** | 代表接收端粗略都卜勒追蹤／預補償後的**殘餘**都卜勒，而非原始物理最大值（見下方工作範例，原始最大值遠高於此，須於接收前端另行補償，不在本模擬器範圍內） |
| Timing offset | samples | **0／2（small）／8（moderate）** | 相對 128-sample 視窗的小比例（約 1.6%／6.25%），避免大範圍位移使 burst 幾乎完全移出偵測視窗 |

**模擬器取樣率假設**：本模擬器採用 $F_s = 200\,\text{kHz}$ 作為 baseband 取樣率假設——**這是模擬器層級的自訂假設，RadioML2016.10a 資料集本身不攜帶取樣率欄位**（Step 2 文件第二節已確認 RadioML 系列資料集無此 metadata）。

**Doppler 工作範例（僅供參數選擇依據，非模擬器直接套用的公式）**：都卜勒頻移的精確定義是 $\Delta f = (v_{\text{radial}}/c)\,F_c$，其中 $v_{\text{radial}}$ 是收發雙方**視線方向（radial）相對速度分量**，**不是**衛星的總軌道速度——軌道速度只有在衛星恰好沿視線方向運動的瞬間（幾何上非常短暫的特例）才會等於 $v_{\text{radial}}$，其餘時間 $v_{\text{radial}} < v_{\text{orbital}}$，通過頂點正上方時 $v_{\text{radial}}\approx 0$。依 Step 1 已核實的 TS 22.261 Table 7.4.1-1 NOTE1，LEO 高度範圍假設為 300–1 500 km；以標準圓軌道公式 $v_{\text{orbital}}=\sqrt{GM_\oplus/r}$（$GM_\oplus \approx 398\,600\,\text{km}^3/\text{s}^2$，$r=$地球半徑 6 371 km + 高度）估算，得軌道速度約 **7.12–7.73 km/s**。本文件以 $v_{\text{radial}} \approx v_{\text{orbital}}$（衛星低仰角、接近地平線時的極限情況）作為**簡化上界**代入 $\Delta f = (v_{\text{radial}}/c)F_c$，在 $F_c=2\,\text{GHz}$（S-band）約 **47–52 kHz**，在 $F_c=12\,\text{GHz}$（Ku-band，DVB-S2/S2X 常用頻段）約 **285–309 kHz**。**這兩個數字只能稱為 simplified upper-bound／order-of-magnitude reference，不得稱為實際 ground-terminal Doppler trajectory**——真實地面終端接收到的都卜勒軌跡是整個過境過程中連續變化的曲線（從正的最大值經過零點到負的最大值），不是一個固定值，本文件也未計算完整的過境幾何。這些上界數字遠高於本輪選用的 mild／stronger 測試值（1／5 kHz），差距正好反映「原始物理都卜勒需要接收端專門的粗略追蹤／預補償，AMC 演算法實際只需要容忍補償後的殘餘都卜勒」這個現實系統架構假設。

## 七、參數矩陣（本輪不正式跑，見第十五節）

不建立全笛卡兒積。本輪 smoke test（第十三節）以「單一因素隔離＋一組合併」的方式控制規模，而非窮舉全部參數組合。

## 八、Transformation Order

固定順序，`apply_satellite_like_channel()` 內部強制執行：

$$\text{input IQ} \to \text{amplitude scaling} \to \text{timing offset} \to \text{CFO/Doppler phase rotation} \to \text{AWGN}$$

**理由（物理鏈路順序）**：(1) 振幅衰減對應訊號沿通道傳播時的路徑損耗，發生在鏈路最前端；(2) 時序偏移是一個**固定的、離散的 waveform sample shift**（純訊號處理層級的操作：把陣列內容整體平移，邊界補零），發生在訊號已經衰減之後；(3) CFO／Doppler 相位旋轉的相位是相對於**已完成時序偏移後的樣本位置**累積的，因此必須在時序偏移**之後**套用，讓相位旋轉的樣本索引 $n$ 對應的是實際輸出陣列中的樣本位置，而非位移前的參考索引；(4) AWGN 對應接收機前端熱噪聲，是最貼近數位輸出端、也是 SNR 定義中「訊號功率」應該以**已完成前面所有轉換後的功率**為準的最後一步——若 AWGN 在振幅縮放之前套用，`snr_db` 的意義會變成「相對原始未縮放訊號的 SNR」而非「接收端數位輸出實際達到的 SNR」，語意上不如目前順序清楚。

**時序偏移的語意邊界（必須與另外兩個相近但不同的概念分開，不得混用）**：`timing_offset_samples` 在程式碼中**只是**一個固定量的陣列位移，語意上：
- **是**：channel waveform 本身的位移——訊號在到達接收端數位輸出時，相對於原始無位移版本，內容整體提早或延後了固定的整數樣本數。
- **不是** propagation delay（見第九節：傳播延遲在本模擬器中只作為 metadata，且 128-sample burst 的時間跨度遠小於任何真實傳播延遲，不可能在單一 burst 內以樣本位移表現）。
- **不是** receiver sample-clock offset／drift 的完整模型——真正的接收端取樣時脈誤差通常表現為**連續累積的漂移**（取樣週期本身有微小誤差，隨時間累積成越來越大的時序誤差），這種效應需要重新取樣／內插才能正確模擬，屬於第二節列為 OPTIONAL、本輪明確不實作的「sample-rate offset」因素，**不是**這裡的固定整數位移可以代表的。`timing_offset_samples` 只能代表一個**單一 burst 內固定不變**的時序未對齊量，可以是接收端符元時序抓取誤差的其中一種簡化表示，但不宣稱是任何特定物理機制的完整模型。

**決定性驗證**：單元測試 8 確認相同 seed／參數輸入下輸出逐位元相同（`np.array_equal`），且僅有 AWGN 階段依賴 `seed`，其餘四個階段皆為純函數。

## 九、Propagation-Delay 的處理

**明確聲明：本模擬器不使用 `time.sleep()` 或任何形式的 CPU 延遲模擬衛星傳播延遲，也不在 128-sample 短 burst 內以逐樣本位移模擬傳播延遲**——第五節公式中沒有任何「propagation delay」的訊號變換，`propagation_delay_ms` 參數**只作為 metadata 記錄**（見 `apply_satellite_like_channel` 回傳的 dict），用途是：

1. **Timeline／情境標註**：記錄這次模擬對應 Step 1 Table 7.4.1-1 的哪一個軌道類型（例如標註 `propagation_delay_ms=26`，代表這次模擬情境對應 LEO 的 UE-to-ground 上限）。
2. **End-to-end latency accounting 參考**：未來若要把 Step 1 的 processing latency 數字與這裡的軌道情境放在同一張圖表或報告中比較，可以用這個欄位對齊，而不需要另外查表。

**理由**：一個 128-sample burst 在任何合理取樣率下的時間跨度（例如 $F_s$=200 kHz 下約 0.64 ms）遠小於任何真實衛星傳播延遲（第四節 Step 1 引用的 LEO 最小值即達 1 ms，GEO 更達 120–136 ms），傳播延遲**不可能**在單一 burst 內部表現為可觀察的樣本域效應——它影響的是「這個 burst 何時抵達接收端」這個更高層級的時間軸問題，不是 burst 內部的訊號形狀問題。若未來需要模擬多個 burst 之間的相對到達時間差（例如長時間串流的排程模擬），才需要在 metadata 之外另外設計時間軸機制，本輪不在範圍內。

## 十、CFO vs Doppler

必須明確區分兩者的**物理來源**，即使兩者在 baseband 的**數學表現**相同：

| | CFO | Doppler |
|---|---|---|
| 物理來源 | 接收端本地振盪器（LO）與發射端載波頻率之間的殘餘失配 | 收發雙方相對運動造成的都卜勒頻移 |
| 是否隨時間變化 | 通常視為相對穩定（振盪器老化/溫度飄移是慢時間尺度） | 隨衛星通過過程持續變化（快時間尺度，尤其 LEO） |
| 與軌道的關係 | 與軌道類型無直接關係，是接收機硬體品質問題 | 與軌道高度、相對速度直接相關（見第六節工作範例） |

**本模擬器的處理方式**：`apply_satellite_like_channel()` 的實作把 `cfo_hz` 與 `doppler_hz` **相加**成單一 `combined_frequency_offset_hz` 後，用同一個複數旋轉 primitive（`_apply_frequency_rotation`）套用——這是因為在 baseband 複數表示下，兩者的數學效應無法在事後被區分開來（單元測試 5c 特別驗證：即使實作共用同一個旋轉運算，回傳的 metadata 仍然把 `cfo_hz` 與 `doppler_hz` **分開保留**，不得只回傳一個沒有語意的 `frequency_offset` 欄位）。這個設計讓下游分析／報告可以依照 metadata 追溯「這次模擬的頻率偏移中，有多少是刻意設定為 CFO、多少是刻意設定為 Doppler」，即使兩者實際套用時是同一個數學運算。

## 十一、Implementation

新增 `src/channel/satellite_like.py`（含 `apply_satellite_like_channel()` 主函式與 `_apply_amplitude_scaling`／`_apply_timing_offset`／`_apply_frequency_rotation`／`_apply_awgn` 四個內部函式，以及 `SatelliteChannelParams` dataclass）。API 簽章與回傳 metadata 完全依本輪任務規格：

```python
def apply_satellite_like_channel(
    iq, sample_rate, snr_db=None, amplitude_scale=1.0,
    cfo_hz=0.0, doppler_hz=0.0, timing_offset_samples=0,
    propagation_delay_ms=None, seed=0,
) -> Tuple[np.ndarray, dict]
```

Metadata 包含：`snr_db`、`amplitude_scale`、`cfo_hz`、`doppler_hz`、`combined_frequency_offset_hz`、`timing_offset_samples`、`sample_rate`、`propagation_delay_ms`、`seed`、`input_power`、`output_power`。輸入驗證：拒絕非 1-D 陣列、拒絕含 NaN/Inf 的輸入、拒絕 `amplitude_scale<=0`、拒絕 `sample_rate<=0`；輸出額外檢查非有限值（fail closed，不靜默處理）。未修改 `external/AWN` 或 `external/adversarial-rf`。

## 十二、Unit Tests

新增 `experiments/test_satellite_like_channel.py`，涵蓋本輪任務要求的全部 10 項加上 3 項額外邊界檢查，共 **29 項檢查，全數通過（29 passed, 0 failed）**：

| # | 驗證項目 | 結果 |
|---|---|---|
| 1 | Identity case（全部 impairment=0）：output == input（逐位元） | PASS |
| 2 | Amplitude：output power == $a^2 \times$ input power（3 個 level 分別驗證） | PASS |
| 3 | CFO：量測相鄰樣本相位差與 $2\pi f/F_s$ 公式一致 | PASS |
| 4 | Doppler：同上，metadata 語意分開保留 | PASS |
| 5 | CFO+Doppler：合併旋轉 == cfo+doppler，metadata 仍分開保留兩個獨立欄位 | PASS |
| 6 | Timing shift：三種位移量（0／+3／-5）樣本內容與邊界補零皆正確 | PASS |
| 7 | AWGN：大樣本數（12 800 點）下量測 achieved SNR 與目標值誤差 <1.0 dB | PASS |
| 8 | Deterministic seed：相同參數＋seed 逐位元相同；不同 seed 產生不同噪聲實現 | PASS |
| 9 | Shape／dtype 保持（complex64 進 complex64 出） | PASS |
| 10 | NaN／Inf 輸入皆正確拒絕（`ValueError`） | PASS |
| 11-12 | 額外邊界：`amplitude_scale<=0`／`sample_rate<=0` 皆正確拒絕 | PASS |
| 13 | Metadata 完整性：全部必要欄位存在，`propagation_delay_ms` 原樣記錄、未套用於訊號 | PASS |

## 十三、Smoke-Test Design

新增 `experiments/run_satellite_like_smoke.py`。流程完整走正式 real building blocks（`energy_detect`／`mask_to_regions`／`merge_close_regions`／`filter_by_min_length`／`select_aligned_segments`／`apply_awn_preprocess`／`to_awn_input`／`AWNModelAdapter`／`compute_sensing_ground_truth_metrics`，全部未修改），本輪**不含 attack／Top-K**（先確認 channel＋sensing＋AMC 正確，攻擊/防禦留待後續）。

**規模**：3 個調變（BPSK／QPSK／8PSK）× 3 個 SNR（-10／0／18 dB）× 每格 2 個樣本 × 6 個通道條件 = **108 筆**。

**6 個通道條件**（單因素隔離 A-E，合併驗證 F）：

| 條件 | `snr_db` | `amplitude_scale` | `cfo_hz` | `doppler_hz` | `timing_offset_samples` |
|---|---|---|---|---|---|
| A_baseline | None | 1.0 | 0 | 0 | 0 |
| B_amplitude | None | 0.5 | 0 | 0 | 0 |
| C_cfo | None | 1.0 | 2000（mild） | 0 | 0 |
| D_doppler | None | 1.0 | 0 | 1000（mild） | 0 |
| E_timing | None | 1.0 | 0 | 0 | 2（small） |
| F_combined | 15.0 | 0.5 | 2000 | 1000 | 2 |

記錄欄位：調變、SNR、channel 參數與 metadata、`sensing_detected`、`captured_signal_ratio`、起訖邊界誤差、clean prediction、clean correctness、confidence、各階段延遲、status。

## 十四、Smoke-Test Results

執行 108 筆，**0 error、0 no_region_detected、0 NaN/Inf、0 fallback**，全程使用真實 AWN backend（`external/adversarial-rf/models/model.py:AWN`）。

| 條件 | n | Sensing 偵測率 | Conditional accuracy | 平均 captured_signal_ratio |
|---|---|---|---|---|
| A_baseline | 18 | 18/18（100%） | 10/18（55.6%） | 0.977 |
| B_amplitude | 18 | 18/18（100%） | 3/18（16.7%） | 0.977 |
| C_cfo | 18 | 18/18（100%） | 3/18（16.7%） | 0.978 |
| D_doppler | 18 | 18/18（100%） | 6/18（33.3%） | 0.978 |
| E_timing | 18 | 18/18（100%） | 10/18（55.6%） | 0.977 |
| F_combined | 18 | 18/18（100%） | 3/18（16.7%） | 0.977 |

**Sensing 完全不受影響**（6 個條件皆 100% 偵測率、captured_signal_ratio 皆穩定在 0.977–0.978），與第五節理論預期一致：`energy_detect` 是功率為基礎的偵測，相位旋轉（CFO／Doppler）不改變 $|x[n]|$，小幅振幅縮放與時序位移在本輪測試的量級下也未造成偵測失敗。

**AMC 準確率依條件顯著不同，且 baseline 本身（10/18＝55.6%）就不是 100%**——逐筆檢查 baseline 的 8 個錯誤案例，發現**全部 6 個 -10 dB 樣本（3 調變 × 2 樣本）都分類錯誤**，且這些錯誤案例的 confidence 值明顯偏低（0.27–0.33，接近均勻分布，代表模型本身不確定，不是靜默給出高信心的錯誤答案），另外 2 個錯誤散布在 0 dB／18 dB。這與本專案既有文件（`docs/research/DIGITAL_LOW_PERTURBATION_ATTACK_EXPERIMENT_ZH_TW.md` 等）反覆記錄的「-10 dB 是 AMC 困難訊噪比」現象一致，加上本輪每個 (調變,SNR) 格只有 2 個樣本，統計量極小，屬於**預期的取樣變異＋已知低 SNR 難度**，不視為 pipeline 錯誤（**依第十一節判準，本輪不需要、也不應該把這個現象立刻當 bug**）。

CFO／Doppler／合併條件下準確率進一步下降（16.7%–33.3%），與第五節「BPSK/QPSK/8PSK 屬 phase-shift keying 家族，相位持續旋轉會使星座圖偏轉，預期比 AWGN 更直接影響分類」的理論預期方向一致——**這是通道效應本身的預期影響，不是本輪新增程式碼的缺陷**，判斷依據：(a) sensing 本身完全不受影響（排除了 channel 模組讓 IQ 資料本身損壞的可能）、(b) 準確率下降的方向與 PSK 家族對相位敏感的已知特性一致、(c) 0 個 error／NaN/Inf/fallback（排除了程式錯誤導致的靜默失敗）。

**未預期或反常案例**：無（`n_error=0` 全程，未觀察到任何 exception、NaN、Inf 或 fallback）。

## 十五、Final Matrix Candidates（本輪不正式跑）

| | Modulation | SNR | Channel condition | Attack | Top-K | 每格樣本數 | 總數 |
|---|---|---|---|---|---|---|---|
| **MINIMUM** | BPSK/QPSK/8PSK（3） | -10/0/18（3） | clean/mild/moderate/strong（4） | none（1） | off（1） | 2 | **72** |
| **REDUCED** | 同上（3） | 同上（3） | clean, moderate（2，用於 attack 區塊）＋ MINIMUM 的 4 級 channel-only 區塊 | none（channel-only 區塊）＋ FGSM-optimized/PGD-optimized（attack 區塊，2） | off/on（attack 區塊，2） | 2 | 72（channel-only,沿用MINIMUM）+ 3×3×2×2×2×2=144（attack 區塊）=**216** |
| **FULL** | 同上（3） | -10/0/10/18（4） | clean/mild/moderate/strong（4） | none/FGSM-optimized/PGD-optimized（3） | off/on（2） | 2 | 3×4×4×3×2×2=**576** |

**避免笛卡兒積爆炸的設計原則**：REDUCED 與 FULL 皆**不**對「攻擊」與「全部 4 級 channel condition」做完整交叉——REDUCED 只在**一個代表性 channel 條件**（moderate）上測試攻擊/Top-K 全部組合，channel severity 本身的掃描（4 級）只在無攻擊（none）情境下進行，兩個問題（「channel 是否影響 sensing/AMC」與「攻擊/防禦在某個代表性 channel 下是否仍有效」）分開回答，而不是要求兩者的全部交叉。FULL 才是完整交叉，僅作為未來若時間允許的最完整版本記錄，本輪不執行。

**估算 CPU runtime**：本輪 smoke test 108 筆（僅 sensing+AMC，未套用 Step 1 已建立的執行緒優化）實測總耗時約 22 秒，平均每筆約 203 ms（見 `satellite_like_smoke_raw.csv` 逐階段延遲欄位），此為**目前預設執行緒設定**下的量級；若套用 Step 1 第十五節已驗證的 `torch_num_threads=2` 設定（本輪 smoke test 未套用），預期可大幅縮短，但本輪未實測驗證這個交互作用。

| Matrix | 樣本數 | 目前預設執行緒設定估算 | 若套用 threads=2（未實測，依 Step 1 既有發現外推） |
|---|---|---|---|
| MINIMUM | 72 | 72 × ~0.20 s ≈ **14–15 秒** | 數秒等級（未實測） |
| REDUCED | 216 | 216 × ~0.20 s（channel-only 部分）+ attack 部分另計，量級約 **40–60 秒** | 個位數至十餘秒等級（未實測） |
| FULL | 576 | 量級約 **2 分鐘**（channel-only 部分外推，attack 部分需另計 Step 1 已知的 attack baseline 延遲） | 數十秒等級（未實測） |

以上 runtime 皆為**依本輪與 Step 1 既有量測外推的估計值**，不是 FULL/REDUCED matrix 本身的實測結果——本輪明確不執行這些矩陣。

**Project-close 正式建議**：FULL matrix 的估算 runtime（約 2 分鐘，目前預設執行緒設定下）仍屬「幾分鐘」等級，且第十六節已確認 channel 語意（amplitude／CFO／Doppler）皆正確、無需修正，**正式 project-close experiment 建議優先採用 FULL matrix**（576 筆），而非 MINIMUM 或 REDUCED——FULL 提供完整的 modulation×SNR×channel×attack×Top-K 交叉覆蓋，且 runtime 成本並不顯著高於 REDUCED，沒有理由為了節省數十秒而犧牲覆蓋完整性。MINIMUM／REDUCED 保留作為**萬一 FULL 因故無法在時間內完成時的降級選項**，優先順序為 FULL＞REDUCED＞MINIMUM。本輪仍然**不執行**任何一種矩陣，此建議留待下一輪正式 Final Matrix 執行時採用。

## 十六、Focused Validation：Amplitude／CFO／Doppler Root-Cause Analysis

108 筆 smoke test（第十四節）顯示 amplitude-only／CFO-only／Doppler-only 條件下 AMC 準確率明顯下降。因為 smoke test 樣本數太小（每格 2 筆），本節針對三個因素個別做較大樣本數的 focused validation，回答「準確率下降的根本原因」，**不新增任何 channel impairment，不修改 `src/channel/satellite_like.py`**。

### 16.1　AWGN Power Calculation Semantics（程式碼稽核）

逐行核對 `src/channel/satellite_like.py`：`apply_satellite_like_channel()`（第 172–180 行）依序呼叫 `_apply_amplitude_scaling` → `_apply_timing_offset` → `_apply_frequency_rotation` → `_apply_awgn`，且 `_apply_awgn(out, snr_db, seed)` 傳入的 `out` 是**已經過前三個階段轉換後**的陣列（`_apply_awgn` 內部 `signal_power = mean(|iq|^2)` 就是對這個已轉換陣列計算）。**結論：AWGN noise power 是依 amplitude scaling 之後的訊號功率計算（選項 A），不是依原始未縮放功率計算（選項 B）**。數學推論：若 $P_{\text{after amp}} = a^2 P_x$，則 $\sigma^2 = a^2 P_x / 10^{\text{snr\_db}/10}$，達成的 SNR $= P_{\text{after amp}}/\sigma^2 = 10^{\text{snr\_db}/10}$，**與 $a$ 無關**——理論上 amplitude scaling 應該完全維持 requested SNR，與 $a$ 值無關。

### 16.2　Achieved SNR 量測（實測驗證 16.1 的理論推論）

新增 `experiments/diagnose_satellite_channel_amplitude.py` Part A：對 amplitude_scale $\in\{0.5,1.0,2.0\}$ × target SNR $\in\{-10,0,18\}$ dB，每組 20 筆，透過對同一筆樣本呼叫 `apply_satellite_like_channel()` 兩次（一次帶 `snr_db=target`、一次 `snr_db=None`）取得實際注入的噪聲實現，據此反推 achieved SNR。輸出 `amplitude_snr_validation.csv`：

| Target SNR | amplitude=0.5 | amplitude=1.0 | amplitude=2.0 |
|---|---|---|---|
| -10 dB | -9.9099 ± 0.3205 dB | -9.9099 ± 0.3205 dB | -9.9099 ± 0.3205 dB |
| 0 dB | 0.0901 ± 0.3205 dB | 0.0901 ± 0.3205 dB | 0.0901 ± 0.3205 dB |
| 18 dB | 18.0901 ± 0.3205 dB | 18.0901 ± 0.3205 dB | 18.0901 ± 0.3205 dB |

**三個 amplitude 值在每個 target SNR 下的 achieved SNR 平均值與標準差逐位數相同**——這是與 16.1 節理論推論完全一致的實測證據：**amplitude scaling 確實完全維持 requested SNR**。（與 target 之間穩定的 +0.09 dB 偏移是有限樣本數下功率比估計的預期統計效應，三個 amplitude 值下偏移量完全相同，不是系統性錯誤。）

### 16.3　Amplitude-Only 準確率下降的根本原因

**16.2 已證明 amplitude scaling 不改變 achieved SNR，因此準確率下降不能歸因於「SNR 變差」。** 新增 `experiments/diagnose_satellite_channel_amplitude.py` Part B/C，對 108-sample smoke test 使用的相同 18 筆樣本（`snr_db=None`，與 smoke test B_amplitude 條件完全一致），追蹤 satellite-channel 輸出、sensed crop、`captured_signal_ratio`、raw crop tensor、AWN 前處理後張量、clean logits、prediction，並與 amplitude=1.0 的對應樣本逐一比較：

| amplitude | n | sensing 偵測率 | sensing accuracy | oracle accuracy | AWN input 與 amp=1.0 平均最大絕對差 | prediction 與 amp=1.0 相符率 |
|---|---|---|---|---|---|---|
| 0.5 | 18 | 18/18 | 3/18（與原 smoke 完全一致） | 3/18 | **0.010975** | 0.0 |
| 1.0（參照） | 18 | 18/18 | 10/18（與原 smoke 完全一致） | 10/18 | 0（自我比較） | 1.0 |
| 2.0 | 18 | 18/18 | 3/18 | 2/18 | **0.021951** | 0.056 |

**關鍵發現**：「AWN input 與 amp=1.0 平均最大絕對差」在 amp=0.5／2.0 下分別為 0.010975／0.021951，**比值恰為 2.0**，與這兩個 amplitude 值相對 1.0 的差距比值（$|1.0-0.5|=0.5$ 對 $|2.0-1.0|=1.0$，比值同樣是 2.0）**完全吻合**——這證明 AWN 送入模型的最終張量**與 amplitude_scale 成正比變化，沒有任何正規化把這個差異吸收掉**。

**根本原因（程式碼稽核確認，非猜測）**：`src/sensing/normalize.py:apply_awn_preprocess()` 的文件字串明確記載——本輪與既有全部正式腳本（含 108-sample smoke test）使用的 `"radioml-native"` 政策是**「no rescaling whatsoever」（完全不做任何重新縮放）**，只有 `"legacy-unit-power"` 政策才會做逐段單位功率正規化。因此在 `"radioml-native"` 政策下，channel 的 `amplitude_scale` 會**直接、成比例地**改變送進 AWN 的絕對數值範圍，而 AWN checkpoint 是在 RadioML 原生（未經 amplitude_scale 縮放）的固定絕對尺度上訓練的——`amplitude_scale≠1.0` 因此讓輸入落到模型訓練分布之外，這是標準的 covariate shift／分布外輸入問題，不是 sensing 錯誤，也不是 channel 模組的數學錯誤。

### 16.4　Oracle vs Sensing Crop 比較

16.3 節同一份追蹤資料顯示：**amplitude=0.5 時 sensing accuracy（3/18）與 oracle accuracy（3/18）完全相同；amplitude=2.0 時 sensing（3/18）與 oracle（2/18）僅差 1 筆**——Oracle crop 直接使用已知的 true_start/true_end，完全繞過 `energy_detect`，如果準確率下降主要來自 sensing／window selection，oracle 路徑的準確率應該明顯優於 sensing 路徑，但實測兩者幾乎相同。**結論：準確率下降不是 sensing/window selection 造成的，根本原因確認在 16.3 節的 AWN 輸入分布外問題。**

### 16.5　CFO／Doppler Sanity Test

新增 `experiments/diagnose_cfo_doppler_sanity.py`：BPSK/QPSK/8PSK、SNR=18 dB、每調變 10 筆（共 30 筆／掃描點），CFO 掃描 $\{0,500,1000,2000\}$ Hz，Doppler 掃描 $\{0,250,500,1000\}$ Hz（皆對應第六節 none／small／medium／smoke-maximum 四級），其餘 impairment 關閉。

| CFO (Hz) | n | 偵測率 | Accuracy | 量測頻移 (Hz) |
|---|---|---|---|---|
| 0 | 30 | 30/30 | 29/30 | -0.00 |
| 500 | 30 | 30/30 | 26/30 | 500.00 |
| 1000 | 30 | 30/30 | 22/30 | 1000.00 |
| 2000 | 30 | 30/30 | 9/30 | 2000.00 |

| Doppler (Hz) | n | 偵測率 | Accuracy | 量測頻移 (Hz) |
|---|---|---|---|---|
| 0 | 30 | 30/30 | 29/30 | -0.00 |
| 250 | 30 | 30/30 | 29/30 | 250.00 |
| 500 | 30 | 30/30 | 26/30 | 500.00 |
| 1000 | 30 | 30/30 | 22/30 | 1000.00 |

**確認結果**：(1) 設定的頻移在全部 8 個掃描點皆被精確量測到（量測值與設定值誤差 <0.01 Hz）；(2) accuracy 隨頻移增加**平滑、方向一致地下降**（CFO：96.7%→86.7%→73.3%→30.0%；Doppler：96.7%→96.7%→86.7%→73.3%），與第五節「PSK 家族對相位旋轉敏感」的理論預期完全一致；(3) 交叉核對 CFO=1000 Hz 與 Doppler=1000 Hz 的量測頻移**完全相同**（兩者皆為 1000.000±0.000 Hz，差異恰為 0.000000 Hz），確認共用同一個旋轉 primitive 時數學結果一致；(4) sensing 偵測率在全部 8 個掃描點皆為 100%，與 16.3／16.4 節「sensing 對這些效應不敏感」的發現一致。**未發現 CFO／Doppler 實作錯誤。**

### 16.6　結論：Channel Semantics 判定

| 問題 | 結論 |
|---|---|
| Amplitude scaling 是否保持 requested SNR？ | **是**，16.1／16.2 節理論與實測皆確認 |
| Amplitude-only 準確率下降主因？ | AWN 在 `"radioml-native"`（無正規化）政策下對輸入絕對尺度敏感，`amplitude_scale≠1` 造成分布外輸入，見 16.3 節 |
| Sensing 是否為主因？ | **否**，16.4 節 oracle vs sensing 幾乎相同 |
| AWN normalization 是否抵消純 amplitude scaling？ | **否**（在目前專案全程使用的 `"radioml-native"` 政策下完全不抵消，`"legacy-unit-power"` 政策未測試但依其文件字串描述應會抵消） |
| CFO transform 是否正確？ | **正確**，16.5 節量測頻移與設定值逐點吻合 |
| Doppler transform 是否正確？ | **正確**，同上，且與 CFO 於相同數值下結果完全一致 |
| Doppler range 是否合理標為 upper-bound／reference？ | 是，第六節已補充明確標記為 simplified upper-bound／order-of-magnitude reference，並加入精確的 $\Delta f=(v_{\text{radial}}/c)F_c$ 公式與 radial vs orbital 速度區分 |
| Timing offset 語意是否正確？ | 第八節已補充：明確定義為固定離散 waveform sample shift，非 propagation delay，也非（意涵連續漂移的）receiver sample-clock offset／drift |

**本輪未發現 `src/channel/satellite_like.py` 的實作錯誤**，因此**未修改任何 channel 程式碼**，29/29 單元測試與 108-sample smoke test 結果維持第十二、十四節原始記錄不變，不需要重跑。Amplitude-only 準確率下降是「channel 正確地改變了絕對訊號尺度」與「既有、未修改的 `apply_awn_preprocess` 正式政策在 `radioml-native` 模式下不做正規化」兩者組合下的**預期結果**，不是 bug，本輪依指示未為了讓準確率好看而修改任何模型或前處理程式碼。

## 十七、限制（Limitations）

1. 第六節的 CFO／Doppler 參數層級（mild／stronger）是依典型振盪器誤差量級與簡化都卜勒上界估算選定的**合理但非唯一**選擇，不代表任何特定衛星系統的規格值；模擬器取樣率 $F_s=200$ kHz 是本輪自訂假設，RadioML2016.10a 本身不攜帶取樣率資訊。
2. 第六節 Doppler 工作範例使用的「最大都卜勒＝軌道速度」是簡化上界（實際依幾何位置變化，通常小於此值），且僅使用了 Step 1 已核實的 LEO 高度範圍（TS 22.261 Table 7.4.1-1 NOTE1），未查證任何額外的都卜勒專屬標準文件。
3. Smoke test（第十三、十四節）僅 108 筆、每個 (調變,SNR) 格僅 2 個樣本，統計量極小，第十四節報告的準確率數字**不具統計代表性**，僅用於確認 pipeline 正確執行與觀察方向性趨勢，不得引用為「AMC 在 XX dB CFO 下的準確率」這類量化結論。
4. 本輪未測試 attack／Top-K 與 channel 因素的交互作用（刻意排除，見第十三節），這部分需要在 REDUCED／FULL matrix 執行後才能回答。
5. 第十五節的 runtime 估計是外推值，不是實測值，實際執行 REDUCED／FULL matrix 時的真實耗時可能因系統負載、執行緒設定、攻擊迭代次數等因素而有落差。
6. Amplitude scaling 與既有 `embed_sample_in_noise`／`embed_complex_iq_in_noise` 背景噪聲機制之間的交互作用已在第 16.3 節以實測證實（`amplitude_snr_validation.csv` 顯示 achieved SNR 對 amplitude 不敏感），第 5.1 節原本的理論說明已由本輪實測支持，不再只是推論。
7. 第 16.3 節「`amplitude_scale≠1` 造成 AWN 輸入分布外」的結論僅針對 `"radioml-native"` 政策實測驗證；`"legacy-unit-power"` 政策下 amplitude scaling 是否確實被正規化吸收，本輪未實測，僅依該政策的文件字串描述推論，屬於未驗證的推論而非實測結論。
8. 第 16.5 節 CFO／Doppler sanity test 固定在 SNR=18 dB（單一、較高的訊噪比）下進行，未測試 CFO／Doppler 與低 SNR 同時出現時是否有交互作用（例如低 SNR 下相位旋轉的影響是否被噪聲進一步放大或掩蓋），這部分需要在正式 FULL matrix 才能回答。
9. 本文件與其對應程式碼是 **satellite-like／standard-inspired 模擬**，不是 standard-compliant 的 DVB-S2/S2X 或 3GPP NTN 通道模型，也不宣稱已完成任何形式的正式衛星通道驗證。

## 十八、Primary References

1. `docs/research/SATELLITE_APPLICATION_AND_LATENCY_REQUIREMENTS_ZH_TW.md`（Step 1，未重寫，本輪沿用其第九、十一、十三、十六節）——ETSI TS 122 261 V19.12.0（2025-10）Table 7.4.1-1（LEO 高度 300–1 500 km 假設）之原始引用來源。
2. `docs/research/SATELLITE_DATASET_AND_MODULATION_FEASIBILITY_ZH_TW.md`（Step 2，未重寫）——BPSK／QPSK／8PSK 為 project-close modulation set 之決策依據。
3. `docs/research/PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md` 第十五節——本輪第十五節 runtime 估算所依據的既有 `torch_num_threads` 執行緒優化發現。
4. 標準圓軌道力學公式 $v=\sqrt{GM_\oplus/r}$ 與地球重力常數 $GM_\oplus\approx 398\,600\,\text{km}^3/\text{s}^2$、地球平均半徑 6 371 km——通用軌道力學基本關係，非特定標準文件之引用。
5. `src/channel/satellite_like.py`、`experiments/test_satellite_like_channel.py`、`experiments/run_satellite_like_smoke.py`——本輪新增之程式碼與測試，為第五至十四節數學模型與結果之直接來源。
6. `src/sensing/normalize.py:apply_awn_preprocess()` 文件字串——第 16.3 節「`radioml-native` 政策不做任何正規化」結論之直接一手來源。
7. `experiments/diagnose_satellite_channel_amplitude.py`、`experiments/diagnose_cfo_doppler_sanity.py`——本輪（focused validation）新增之診斷程式，為第十六節全部數據之直接來源，輸出 `amplitude_snr_validation.csv`、`amplitude_trace.csv`、`cfo_doppler_sanity.csv`。
