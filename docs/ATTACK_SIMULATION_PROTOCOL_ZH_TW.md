# 頻譜感知攻擊模擬與可靠性驗證協定

## 1. 目的

本協定說明如何把威脅模型中的攻擊轉成可重現、可量化，而且接近實際 SDR 發射條件的模擬。核心原則是：

> 攻擊者只能從 RF 通道加入自己的複數基頻波形，不能任意覆寫接收端 tensor，也不假設能完美相消合法訊號。

這裡的「可靠」不是指把攻擊功率無限制調大直到成功，而是指攻擊在預先宣告的功率、頻寬、時間、CFO、通道與同步不確定性下，對未參與調參的 seeds／channels 仍有穩定成功率。

目前可使用的 NumPy 注入器位於：

```text
src/sensing/rf_attack_sim.py
```

它提供：

- `generate_unit_power_complex_noise()`：可重現的單位功率複數噪聲。
- `generate_unit_power_tone()`：單位功率窄頻 tone。
- `inject_additive_waveform()`：按照接收端 PSR 注入攻擊，並加入 CFO、相位及接收增益誤差。

## 2. 接收訊號模型

每次模擬都應以接收器看到的訊號為準：

\[
y[n] = h_s * s[n-\tau_s] + h_a * a[n-\tau_a]e^{j(2\pi\Delta f n+\phi)} + w[n].
\]

其中：

- \(s[n]\)：合法發射器波形。
- \(a[n]\)：攻擊 SDR 的基頻波形。
- \(h_s,h_a\)：合法與攻擊通道。
- \(\tau_s,\tau_a\)：獨立時間偏移。
- \(\Delta f\)：攻擊 SDR 與接收 SDR 之間的 CFO。
- \(\phi\)：未知初始相位。
- \(w[n]\)：接收噪聲。

攻擊只能透過 \(a[n]\) 及其發射參數影響結果。除 A0 演算法上界外，不允許直接指定接收端的任意 \(\delta\)。

## 3. 功率預算

### 3.1 主要指標：PSR

以接收端 perturbation-to-signal ratio 定義攻擊強度：

\[
\mathrm{PSR}_{dB}=10\log_{10}\frac{P_a}{P_s}.
\]

`P_s` 必須在合法訊號的 ground-truth active interval 上計算，`P_a` 則在攻擊實際發射 interval 上計算。每次輸出都要保存 target PSR 與 achieved PSR；不能只保存發射端 amplitude。

建議第一輪 sweep：

```text
PSR = {-30, -24, -18, -15, -12, -9, -6, -3, 0} dB
```

這是搜尋範圍，不代表每個值都具物理合理性。最後論文應報告成功率曲線，以及達到指定可靠度所需的最小 PSR，而不是只挑成功點。

### 3.2 其他必要限制

每個 attack record 還應保存：

- 攻擊佔用頻寬／接收通道頻寬。
- 發射長度、duty cycle 與總能量。
- PAPR。
- CFO 與 timing offset。
- phase 與 gain error。
- 頻譜外洩或 ACLR（進入 SDR 實驗後）。
- 若可解調，另報 EVM 與 BER。

## 4. 最小注入範例

```python
import numpy as np

from src.sensing.rf_attack_sim import (
    generate_unit_power_complex_noise,
    inject_additive_waveform,
)

# clean_iq 是包含合法 burst 的一維 complex64 capture。
# 例如合法 burst 位於 [1800:2312]。
attack_waveform = generate_unit_power_complex_noise(
    length=len(clean_iq),
    seed=attack_seed,
)

attacked_iq, attack_component, attack_meta = inject_additive_waveform(
    clean_iq,
    attack_waveform,
    attack_start=0,
    reference_slice=(1800, 2312),
    target_psr_db=-9.0,
    normalized_cfo=cfo_cycles_per_sample,
    phase_rad=phase_rad,
    gain_error_db=gain_error_db,
)
```

後續必須把 `attacked_iq` 從 detection 開始重新跑完整管線，不能先使用 clean signal 的 oracle region 再直接切 128-sample 視窗，否則無法測到 sensing attack。

## 5. 五種主要攻擊的模擬方法

### 5.1 G1：Noise-floor poisoning

#### 攻擊波形

使用高 duty-cycle 的複數噪聲、OFDM-like waveform 或多 tone waveform，覆蓋目前預先通道化的接收通道。不可使用負能量或直接降低合法 burst amplitude。

#### 目前 detector 的特定弱點

目前 detector 使用整段 capture 的 median smoothed power：

```text
noise_floor = median(smoothed_power)
threshold = threshold_factor * noise_floor
```

因此，攻擊者必須污染足夠比例的 capture 才能穩定改變 median。第一輪應 sweep：

```text
duty cycle = {0.40, 0.50, 0.60, 0.80, 1.00}
PSR        = {-18, -15, -12, -9, -6, -3} dB
```

#### 成功條件

同一個 clean capture 必須先成功偵測合法事件；攻擊後若合法 ground-truth interval 的 event overlap 低於門檻，才算成功：

```text
clean_detected == True
and attacked_event_IoU < 0.1
```

另報 noise-floor shift、threshold shift 與 detector recovery time。不能把原本就偵測不到的低 SNR 樣本算成攻擊成功。

#### 必要 baseline

- 同 PSR 的連續高斯噪聲 jammer。
- 相同總能量但低 duty-cycle 的 burst jammer。
- 不做針對性調整的單 tone。

若專門設計的 poisoning waveform 沒有優於這些 baseline，不應聲稱是 adversarial attack；較誠實的名稱是 jamming／noise-floor poisoning vulnerability。

### 5.2 G2：Phantom occupancy

#### 攻擊波形

在沒有合法活動的區間發送多個短 burst。每個 burst 的 on-time 應 sweep detector window 與 `min_region_len` 的相對長度：

```text
attack length / energy_window = {0.5, 1, 2, 4}
inter-burst gap / merge_gap    = {0.5, 1, 2, 4}
PSR relative to background    = {3, 6, 9, 12, 18} dB
```

#### 成功條件

- 每分鐘假事件數。
- 每分鐘額外 AMC 次數。
- 每分鐘保存 IQ bytes。
- analyst queue 的最大與平均長度。

可靠性不能只定義為「至少有一個 false positive」；應定義為在固定 airtime／energy budget 下造成的資源放大率，例如：

\[
\text{amplification}=\frac{\text{defender compute or storage cost}}{\text{attacker airtime or energy}}.
\]

### 5.3 G3：Event bridging

#### 攻擊波形

先建立兩個 clean 時可分離的合法 burst，再只在兩者之間的 quiet gap 發送 tone、noise 或 protocol-shaped burst。攻擊者不需要降低任何能量。

第一輪 sweep：

```text
clean gap = {merge_gap + 1, 2*merge_gap, 4*merge_gap}
bridge on-time / clean gap = {0.25, 0.50, 0.75, 1.00}
bridge PSR = {-18, -15, -12, -9, -6, -3} dB
```

#### 成功條件

```text
clean_region_count == 2
and attacked_region_count == 1
and attacked_region spans both legitimate bursts
```

同時報告合併後選到哪一個 128-sample window、兩個合法 burst 的 retained-sample ratio，以及 AMC 決策是否改變。

### 5.4 G4：Boundary extension

#### 攻擊波形

在合法 burst 前加入 prefix、後加入 suffix，或兩者都加入。prefix／suffix 必須由加性 RF 波形形成，不能直接改寫 ground-truth boundary。

第一輪 sweep：

```text
prefix/suffix length = {16, 32, 64, 128, 256} samples
PSR                  = {-18, -15, -12, -9, -6, -3} dB
```

#### 成功條件

- start boundary error。
- end boundary error。
- event IoU。
- naive segment 中合法樣本比例下降。
- AMC clean-to-attacked decision change。

這個攻擊應分別對 `naive` 與 `max-energy` 測試。`naive` 必須保持既有 byte-compatible 實作，不可以為了讓結果好看而改掉它。

### 5.5 G5：Max-energy window hijacking

#### 攻擊波形

在同一 detected region 內放入短、高局部能量的 attacker pulse。攻擊目標不是讓合法訊號消失，而是使：

\[
\operatorname{mean}|W_A|^2 > \operatorname{mean}|W_L|^2,
\]

讓 max-energy selector 選到 attacker-dominated window。

第一輪 sweep：

```text
pulse length = {8, 16, 32, 64, 128} samples
pulse PSR    = {-18, -15, -12, -9, -6, -3, 0} dB
pulse offset = event 內所有可行位置
```

#### 成功條件

```text
clean selected window overlaps legitimate burst
and attacked selected start != clean selected start
and attacker fraction in selected window >= 0.5
```

另報選取視窗位移、合法訊號 retained ratio、分類結果及 confidence。這項實驗可以直接證明 max-energy alignment 雖改善乾淨資料對齊，也新增可被攻擊的 selector surface。

## 6. AMC 與完整防禦攻擊

現有 `AttackAdapter` 的 FGSM／PGD／CW 是 receiver-side A0 tensor attack；即使模型為 real backend，也不能自動稱為 A1 RF attack。

要建立 A1 攻擊，需把可優化變數限制為 attacker transmitter waveform，並在 loss 中包含：

1. 加性 RF superposition。
2. 隨機 timing offset。
3. CFO、phase 與 gain uncertainty。
4. 可選的 multipath／filter／ADC clipping surrogate。
5. Energy detector、event formation、window selector、Top-K／router、AWN 與 abstention。

若 hard threshold、region merge 或 Top-K 不可微分，可用 BPDA／STE 或 soft surrogate 計算梯度；但 forward pass 必須使用真正的 hard pipeline。若有隨機性，使用 EOT：

\[
\min_a \; \mathbb{E}_{\theta\sim\Theta}[L(F(T_\theta(s,a)),y)]
\]

其中 \(\Theta\) 包含 CFO、phase、timing、gain 與 channel realization。最後成功率必須用未參與 optimization 的 held-out transformations 評估。

## 7. 如何讓結果「可靠」而不是只對一個 seed 成功

### 7.1 分離 calibration 與 evaluation

- Calibration set：尋找波形、PSR 與 attack timing。
- Evaluation set：新的合法樣本、noise seeds、attack seeds、CFO、phase、gain error 與 channel taps。
- 一旦進入 evaluation，不得再依結果調整 attack budget。

### 7.2 Monte Carlo 不確定性

每個 `(attack, modulation, SNR, PSR)` condition 至少使用：

```text
合法樣本 seeds       >= 30
attack waveform seeds >= 10
channel/CFO/phase draws >= 10
```

若計算成本太高，先執行 power analysis，再明確報告較小樣本數的限制。不要把同一 clean sample 的多個 transformation 當成完全獨立樣本。

### 7.3 可靠度定義

建議預先註冊：

```text
Reliable@PSR:
  held-out attack success rate >= 0.90
  and 95% Wilson lower confidence bound >= 0.80
```

同時報告達到 `Reliable@PSR` 的最小 PSR。若任何 PSR 都未達門檻，應誠實報告 attack is not reliable under the tested constraints。

### 7.4 報告完整曲線

至少畫出：

- ASR vs PSR。
- `P_d`／`P_fa` vs PSR。
- Event IoU／boundary error vs PSR。
- ASR vs CFO、timing error、gain error。
- ASR vs channel realization。
- 攻擊 airtime／energy 與 defender cost amplification。

不能只報最佳成功點。

## 8. 建議隨機化範圍

數位模擬的初始範圍可以是：

```text
phase               ~ Uniform(0, 2*pi)
normalized CFO      ~ Uniform(-0.01, 0.01) cycles/sample
timing error        ~ DiscreteUniform(-64, +64) samples
gain error          ~ Uniform(-3, +3) dB
channel delay       ~ DiscreteUniform(0, 32) samples
channel model       = AWGN + optional short Rician/Rayleigh taps
```

這些不是論文最終固定值。進入 cabled SDR 階段後，應先量測實際 USRP／HackRF／PlutoSDR 的 CFO、trigger jitter、gain variation 與 front-end response，再用量測分布替代任意範圍。

## 9. 從數位模擬到實體 SDR

### Stage 1：Digital additive simulation

證明 attack objective、power accounting 與完整 sensing pipeline 行為正確。不能聲稱 OTA feasible。

### Stage 2：`.cfile` replay

把合法與攻擊波形混合後保存為 complex64 `.cfile`，再從正式 capture loader 重播。驗證 metadata、chunking、detector 與 event record。仍不能聲稱獨立 RF 發射可行。

### Stage 3：Cabled dual-SDR

```text
合法 SDR TX -- attenuator --\
                             combiner -> SDR RX
攻擊 SDR TX -- attenuator --/
```

這是第一個能支持獨立攻擊發射器之 physical feasibility 的階段。必須用功率計或接收端校準確認 achieved PSR／SJR，並避免 TX 直接損壞 RX。

### Stage 4：Shielded OTA

在屏蔽箱或合法授權環境測試天線、位置、遮蔽物與 channel variation。公開頻段實驗必須遵守法規，不應將攻擊波形直接在未屏蔽公共環境發射。

## 10. 每筆結果的必要欄位

```text
capture_id, event_id, clean_sample_id
attack_name, attack_seed, signal_seed, channel_seed
sample_rate, center_frequency, channel_bandwidth
true_burst_start, true_burst_end
attack_start, attack_end, attack_duty_cycle
attack_target_psr_db, attack_achieved_psr_db
cfo, phase, timing_error, gain_error_db, channel_id
energy_window, threshold_factor, min_region_len, merge_gap
alignment_policy, segment_hop
clean_regions, attacked_regions
clean_selected_window, attacked_selected_window
clean_label, attacked_label, defended_label
clean_confidence, attacked_confidence, defended_confidence
abstain_status, backend_status
Pd, Pfa, event_iou, boundary_error
attack_success, success_definition_version
```

Real AWN 或 attack backend 載入失敗時必須記錄 failure；不得把 dummy fallback 的結果混入 real-backend robustness table。

## 11. 最小投稿判準

在把攻擊稱為「實際且可靠」前，至少應滿足：

1. 攻擊從 raw IQ／RF injection point 進入，重新執行 detection 到 decision 的完整管線。
2. 報告 achieved PSR、duty cycle、CFO、timing 與 gain/channel uncertainty。
3. 與同功率 noise jammer、tone jammer 及 random burst baseline 比較。
4. 使用 held-out seeds 與 held-out channel transformations。
5. 報告完整 success curve 與 confidence interval。
6. 在 cabled dual-SDR 重現主要效果。
7. 至少一部分結果通過 shielded OTA。
8. 沒有把 A0 tensor attack 描述為 over-the-air attack。
9. 沒有用 perfect cancellation 當作一般攻擊能力。
10. 結果同時呈現成功與失敗區域，並明確標示可支持的 claim boundary。
