# 能量觸發 AMC 系統：威脅模型、元件角色與完成狀態

## 報告目的

本文件用於向指導老師說明三件事：

1. 本研究保護的系統是什麼，以及攻擊者模型如何設定。
2. Spectrum sensing、AMC、攻擊與 Top-K 防禦各自負責什麼。
3. 目前程式已完成哪些部分，還有哪些部分尚未實作或驗證。

目前系統最精確的描述是：

> 預先通道化複數基頻串流上的能量觸發 AMC（Energy-triggered AMC on a pre-channelized complex-baseband stream）。

這句話同時限定了目前的能力：系統處理的是已由上游選好的單一通道，在時間軸上偵測能量活動，再執行自動調變分類。它目前不是寬頻頻譜感測器，也不是完整通訊接收機或自主頻譜控制系統。

---

## 一、研究問題與實際應用

### 1.1 實際應用情境

可以把系統放在工業園區、研究設施或私人無線網路中，作為一個被動式 RF 監測節點。SDR 持續監聽指定通道，系統在發現訊號後建立事件，保存 IQ 證據，並以 AMC 提供可能的調變類型，讓分析人員決定是否需要進一步調查。

例如，監測結果可能是：

```text
事件 ID：127
通道：上游已選定的監測通道
開始時間：20:13:07.125
持續時間：20 ms
能量偵測：有活動
AMC：QPSK
模型信心：0.87
原始 IQ：已保存
```

這個結果只能說「該事件的波形看起來像 QPSK」。它不能證明是哪一台設備、訊號是否合法，也不能知道封包內容。若要回答這些問題，還需要同步、解調、解碼、CRC 與協定解析，目前都不在已實作範圍內。

### 1.2 Spectrum sensing 的角色

Spectrum sensing 位於 SDR 與 AMC 之間，負責回答：

- 通道目前有沒有活動？
- 活動何時開始與結束？
- 哪一段 IQ 應交給 AMC？
- 是否出現異常噪聲底、大量短事件或事件邊界變化？

它是 AMC 前面的觸發器、事件建立器與資料分流器。沒有 spectrum sensing，系統只能固定切割連續 IQ，將大量純噪聲送入模型，既浪費推論資源，也無法建立可信的事件時間軸。

```text
天線／RF 前端
  -> SDR ADC 與數位下變頻
  -> 上游選頻或預先通道化
  -> complex-baseband IQ
  -> 能量偵測
  -> occupied-region 建立、合併與過濾
  -> 128-sample 視窗選擇
  -> AWN 前處理與 AMC
  -> Top-K 防禦與再次分類
  -> CSV、圖表與離線評估結果
```

Spectrum sensing 和其他元件的分工如下：

| 層級 | 回答的問題 | 本專案狀態 |
|---|---|---|
| Spectrum sensing | 何時有訊號活動？事件邊界在哪裡？ | 已有單通道時間能量偵測 |
| AMC | 事件看起來是哪種調變？ | 已有 AWN adapter |
| Demodulation | 訊號包含哪些 symbols／bits？ | 未實作 |
| Protocol analysis | 這是哪種協定、哪台設備、CRC 是否正確？ | 未實作 |
| Control plane | 是否切頻、封鎖或重新配置資源？ | 未實作，近期也不建議自動化 |

---

## 二、Threat model（威脅模型）

### 2.1 保護目標

本研究不處理 RF 訊號的機密性，主要保護監測結果的完整性與可用性：

1. 活動偵測完整性：有合法活動時不能被低成本隱藏。
2. 事件時間軸完整性：事件起訖、分裂及合併結果必須可追溯。
3. AMC 結果完整性：分類與信心應合理反映接收波形。
4. 監測服務可用性：攻擊者不能以大量假事件耗盡推論、儲存及分析資源。
5. 證據完整性：原始 IQ、設定、模型版本與 backend 狀態應被保存。
6. 失效可見性：模型載入失敗、dummy fallback、NaN／Inf、削波或 dropped samples 不能偽裝成正常結果。

### 2.2 接收訊號模型

在單一預先通道化的通道中，接收器看到：

\[
y[n] = (h_s * s)[n-\tau_s] + (h_a * a)[n-\tau_a] + w[n]
\]

其中：

- \(s[n]\) 是合法發射器訊號。
- \(a[n]\) 是攻擊者透過 SDR 發射的波形。
- \(h_s\) 與 \(h_a\) 是兩條不同的 RF 通道。
- \(\tau_s\) 與 \(\tau_a\) 是不同的到達時間。
- \(w[n]\) 是接收器雜訊與背景干擾。

真實攻擊者不能任意覆寫分類器的 tensor。攻擊波形還會受到通道衰減、多徑、頻率偏移、相位偏移、時鐘誤差、發射延遲、前端濾波與 ADC 量化影響。因此，離線的 `clean_iq + delta` 實驗只能作為演算法安全上界，不能單獨證明 over-the-air 攻擊可行。

### 2.3 信任邊界

基準模型假設防禦方控制：

- SDR 主機、作業系統與本地儲存。
- 能量偵測、分段、前處理、AWN 與防禦程式碼。
- AWN checkpoint 與設定檔。
- 實驗中的合路器、衰減器及屏蔽設備。

不可信輸入包括：

- 天線收到的所有 RF 能量。
- 外部 IQ 擷取檔及可能不完整的 metadata。
- 模型的高信心輸出，因為高信心不代表輸入安全或來源合法。

主機入侵、checkpoint 置換與供應鏈攻擊是另一個安全問題。本研究的主要 RF 實驗不把它們和波形攻擊混在一起。

### 2.4 三層攻擊者

#### A0：數位注入攻擊者

A0 可以直接修改離線 IQ tensor，知道模型權重、前處理與所有參數，也能取得精確梯度。A0 不受 RF 通道與同步限制。

用途是驗證演算法、比較 FGSM／PGD／CW，以及找出最壞情況。A0 攻擊成功只能證明數位輸入空間存在弱點，不能單獨支持實體 RF 攻擊主張。

#### A1：受控 RF 攻擊者

A1 是主要且較實際的威脅模型。攻擊者在監測接收器附近使用獨立商用 SDR，可以：

- 監聽合法 RF 活動。
- 控制中心頻率、頻寬、功率、波形、持續時間及 duty cycle。
- 知道系統設計，或取得相同／代理模型。
- 估計自己的發射器到接收器通道。

但 A1 不能修改接收主機、模型、事件資料庫或擷取 metadata，也不假設它能和合法發射器保持精確的時間、頻率及相位同步。

#### A2：查詢式黑箱 RF 攻擊者

A2 不知道完整模型權重或防禦參數，只能透過有限的外部回饋推測結果，例如告警、測試 API 或操作人員反應。它還受到查詢次數、總發射時間與能量限制。

A2 適合評估 transfer attack、universal perturbation，以及 Square／SPSA／NES 類黑箱方法，但目前正式管線尚未完成這些評估。

### 2.5 實際攻擊目標

攻擊者不一定需要直接攻擊 AWN。對 spectrum sensing 與分段流程下手，往往更便宜也更接近真實 RF 條件。

| 攻擊目標 | 作法 | 可能後果 | 目前狀態 |
|---|---|---|---|
| 噪聲底污染 | 長時間發射低功率帶限噪聲，抬高偵測門檻 | 弱合法事件被漏掉 | 威脅模型已定義，正式攻擊流程未實作 |
| 假佔用洪泛 | 反覆發射剛好超過門檻的短 burst | 推論、儲存及分析佇列耗盡 | 威脅模型已定義，營運負載評估未實作 |
| 事件橋接 | 在兩個事件的安靜間隔補入能量 | 兩個事件被錯誤合併 | 可由現有 detector 模擬，尚未形成完整攻擊模組 |
| 邊界延伸 | 在合法 burst 前後加入前綴或尾綴 | 事件起訖與 128-sample 視窗偏移 | 可由離線 IQ 模擬，尚未完整評估 |
| 最大能量視窗劫持 | 插入短而較強的 pulse | `max-energy` 選到攻擊波形 | 分段策略已實作，針對性攻擊未完成 |
| AMC evasion | 讓 AWN 誤分類或降低信心 | 調變標籤不可信 | FGSM／PGD／CW classifier-only 攻擊已實作 |
| 防禦路由操控 | 操控 K、信心或防禦啟動條件 | 繞過防禦或造成拒判 DoS | Adaptive-K／路由器未接入正式管線 |

### 2.6 對 Top-K 的正確攻擊假設

若攻擊只針對未防禦的 AWN 產生 adversarial IQ，之後才套用 Top-K，這只能稱為：

- oblivious attack；或
- defense-unaware attack。

它不是針對完整防禦管線的白箱攻擊。較強的安全結論需要攻擊者知道並最佳化以下完整路徑：

```text
能量偵測
  -> region merge/filter
  -> 視窗選擇
  -> Top-K／Adaptive-K
  -> AWN
  -> 信心、棄權與事件彙整
```

如果中間包含不可微分的選擇，評估還需要 BPDA、可微代理、EOT、gradient-free 方法及多次重啟。目前尚未完成這種 defense-aware adaptive evaluation。

### 2.7 明確排除範圍

目前不主張已處理：

- 接收主機或模型供應鏈入侵。
- 公共頻段上的未授權干擾實驗。
- 多站定位、測向或發射器指紋識別。
- 解調、解碼、CRC 或協定解析。
- 自動封鎖、切頻或頻譜重新配置。
- 沒有 channelizer 時的真正寬頻、多子通道佔用推論。
- 一般附近攻擊者能進行完美相位同步相消。

---

## 三、目前正式資料流程

正式實驗入口是：

```text
experiments/run_full_experiment.py
  -> src/utils/pipeline.py
  -> src/sensing/*
  -> src/adapters/*
  -> src/utils/csv_writer.py / plotting.py
```

目前可證實的離線流程是：

```text
合成 IQ 或 RadioML 樣本
  -> 嵌入較長的噪聲串流
  -> 時間能量偵測
  -> occupied-region 擷取、合併與長度過濾
  -> naive 或 max-energy 視窗選擇
  -> AWN 輸入前處理 [N, 2, 128]
  -> clean AMC
  -> classifier-only adversarial attack
  -> fixed-K FFT Top-K
  -> defended AMC
  -> 逐 segment CSV、感測指標與圖表
```

另外還有較舊的 `scripts/sdr_sensing_to_awn_poc.py`。它可以讀取 `.cfile`，但 AWN 推論仍是隨機 logits placeholder，也沒有正式 attack／Top-K adapters。不能把這支 standalone script 的類別輸出當成真實 AWN 結果。

---

## 四、各元件的工作與完成狀態

### 4.1 狀態定義

- 已完成：程式存在，已接入正式離線 pipeline，可執行對應功能。
- 部分完成：核心程式存在，但只支援離線資料、尚未接線，或仍有安全／部署缺口。
- 未完成：正式 pipeline 沒有對應實作。

「已完成」只代表目前離線 PoC 範圍內完成，不代表已具備現場部署能力。

### 4.2 IQ 資料來源

| 元件 | 工作 | 狀態 | 程式位置與限制 |
|---|---|---|---|
| Synthetic IQ | 產生噪聲加單一 burst，供 smoke test 使用 | 已完成 | `src/sensing/iq_source.py:generate_synthetic_iq`；是簡化測試訊號，不是真實完整調變器 |
| RadioML loader | 載入 RML2016.10a 的 `[2,128]` 樣本 | 已完成 | `src/sensing/radioml_source.py`；屬於離線資料集，不是 live SDR |
| RadioML embedding | 把一個或多個 RadioML 樣本嵌入較長噪聲串流 | 已完成 | 用來建立具有已知真值位置的 sensing 實驗 |
| `.cfile` loader | 讀取 GNU Radio `complex64` IQ 檔 | 部分完成 | `src/sensing/iq_source.py:load_iq_from_file` 已存在，但未接入正式 pipeline |
| Live UHD／USRP | 直接從 SDR 硬體接收 | 未完成 | 沒有 UHD source 或硬體控制程式 |
| GNU Radio／ZMQ streaming | 持續接收 GNU Radio 串流 | 未完成 | 沒有 flowgraph、ZMQ socket 或 rolling buffer |

### 4.3 IQ 驗證

| 工作 | 狀態 | 說明 |
|---|---|---|
| 檢查一維 complex IQ | 已完成 | `validate_iq` 會拒絕非複數或非一維輸入 |
| 檢查 `complex64` | 已完成 | 目前要求固定 dtype |
| NaN／Inf | 未完成 | 尚未在此邊界完整檢查 |
| endianness | 未完成 | `.cfile` 目前使用本機 native byte order |
| sample rate 與中心頻率 metadata | 未完成 | 正式 config 沒有完整 capture metadata contract |
| ADC scaling、增益與 clipping | 未完成 | 尚未用真實 SDR 擷取驗證 |
| 長檔 chunking | 未完成 | 目前 `.cfile` loader 會把整個檔案讀入記憶體 |

### 4.4 時間能量偵測

程式位置：`src/sensing/energy_detection.py`

主要步驟是：

\[
p[n] = |y[n]|^2
\]

\[
\bar p[n] = \frac{1}{W}\sum_{i \in \mathcal{W}_n}p[i]
\]

\[
\hat N = \operatorname{median}(\bar p), \qquad \gamma = \alpha\hat N
\]

\[
m[n] = \mathbb{1}[\bar p[n] > \gamma]
\]

| 子元件 | 工作 | 狀態 |
|---|---|---|
| `energy_detect` | 計算滑動平均功率與門檻 mask | 已完成 |
| `mask_to_regions` | 將 mask 轉為連續時間區域 | 已完成 |
| `merge_close_regions` | 合併間隔過短的區域 | 已完成 |
| `filter_by_min_length` | 移除太短的雜訊區域 | 已完成 |
| 自適應／抗污染噪聲估計 | 限制攻擊者抬高噪聲底 | 未完成 |
| hysteresis detector | 以不同開啟／關閉門檻降低跳動 | 未完成 |
| streaming detector state | 跨 chunk 保存 detector 狀態 | 未完成 |

這裡完成的是單一預先通道化串流上的時間能量偵測。它沒有判斷寬頻輸入中哪些 frequency bins 被占用。

### 4.5 Channelizer 與真正寬頻 sensing

| 工作 | 狀態 | 說明 |
|---|---|---|
| 單一通道時間活動偵測 | 已完成 | 目前系統的 sensing 能力 |
| FFT/STFT occupancy map | 未完成 | 尚未產生 time-frequency 佔用圖 |
| Polyphase filter-bank channelizer | 未完成 | 尚未把寬頻 IQ 分成多個子通道 |
| 每個子通道獨立 noise-floor tracking | 未完成 | 沒有 \(\hat N_k[t]\) 狀態 |
| 跨通道事件關聯 | 未完成 | 沒有 multi-channel event model |

因此，向老師報告時應說「單通道 temporal energy sensing」，不要說已完成 wideband spectrum sensing。

### 4.6 事件分段與視窗選擇

程式位置：`src/sensing/segmentation.py`

| 策略 | 工作 | 狀態 | 限制 |
|---|---|---|---|
| `naive` | 從區域起點依序切固定長度視窗 | 已完成 | 容易受事件邊界偏移影響 |
| `max-energy` | 選擇區域中平均能量最高的視窗 | 已完成 | 每區域只選一個視窗，可能被強短 pulse 劫持 |
| 多重疊視窗 | 對同一事件取多個視窗 | 未完成正式事件流程 | 尚無跨視窗投票與一致性判斷 |
| event object／生命週期 | 追蹤事件開始、更新與結束 | 未完成 | 現在主要是離線 index regions |

### 4.7 AWN 輸入前處理

程式位置：`src/sensing/normalize.py`

| 子元件 | 工作 | 狀態 |
|---|---|---|
| `legacy-unit-power` | 每個 segment 正規化為單位平均功率 | 已完成 |
| `radioml-native` | 保留 RadioML 原始尺度 | 已完成 |
| `to_awn_input` | 將 complex IQ 轉為 float32 `[N,2,T]` | 已完成 |
| 真實 SDR scaling calibration | 對齊 ADC／gain 與模型訓練尺度 | 未完成 |
| 非 128 samples 模型契約 | 支援其他視窗長度 | 未驗證；目前應維持 `[N,2,128]` |

### 4.8 AWN AMC

程式位置：`src/adapters/awn_adapter.py`

| 子元件 | 工作 | 狀態 |
|---|---|---|
| Dummy AWN | 產生 deterministic random logits，供 dry run 使用 | 已完成，但不能當分類結果 |
| Real AWN adapter | 載入 `external/adversarial-rf/models/model.py` 與 checkpoint | 已完成於離線資料路徑 |
| AWN forward | 接收 `[N,2,128]` 並輸出 11 類 logits | 已完成於離線資料路徑 |
| Fail-closed backend | 真實模型失敗時停止，不回退 random logits | 未完成；目前仍可能 fallback dummy |
| 信心校準 | 驗證模型 confidence 是否可信 | 未完成 |
| Unknown／abstention | 低信心或異常輸入時拒判 | 未完成 |
| 真實擷取分類有效性 | 用有標註的 SDR capture 驗證準確率 | 未完成 |

使用 real-backend 旗標不等於實際使用成功。每次結果都必須檢查輸出的 backend 與 status 欄位。

### 4.9 對抗攻擊

程式位置：`src/adapters/attack_adapter.py`

| 攻擊能力 | 狀態 | 說明 |
|---|---|---|
| Dummy attack | 已完成 | deterministic sign-noise，只供 pipeline dry run |
| FGSM | 已完成於 classifier-only 路徑 | 攻擊 AWN，不包含 sensing 與 Top-K |
| PGD | 已完成於 classifier-only 路徑 | 同上 |
| CW | 已完成於 classifier-only 路徑 | 同上 |
| Noise-floor／event attack module | 未完成 | 尚未把 sensing 攻擊做成正式 adapter |
| 完整 defense-aware adaptive attack | 未完成 | 沒有對 detector、segmenter、Top-K 與 AWN 一起最佳化 |
| 黑箱與 transfer attack | 未完成正式評估 | A2 仍是後續工作 |
| OTA transfer validation | 未完成 | 尚未完成纜線或屏蔽 OTA 證據 |

### 4.10 Top-K 頻譜前處理

程式位置：

- `src/adapters/defense_adapter.py`
- `src/adapters/topk_adapter.py`
- `external/adversarial-rf/util/defense.py`

流程是對已選出的 128-sample AMC 視窗執行 FFT，保留 magnitude 最大的 K 個 bins，再轉回時域或交給後續分類。

```text
128-sample IQ
  -> FFT
  -> 保留 Top-K bins
  -> 其餘 bins 清零
  -> IFFT
  -> AWN
```

| 子元件 | 工作 | 狀態 |
|---|---|---|
| NumPy fixed-K Top-K | dry-run 頻域稀疏化 | 已完成 |
| Real fixed-K Top-K adapter | 包裝 external defense 實作 | 已完成於離線路徑 |
| Adaptive-K 函式 | 根據訊號特徵調整 K | 外部程式有函式，但未接入正式 pipeline |
| Defense router | 決定是否啟動防禦及選哪個 K | 未完成 |
| 防禦觸發告警 | 把防禦前後不一致視為安全事件 | 未完成 |
| Adaptive attack evaluation | 針對 Top-K 完整最佳化攻擊 | 未完成 |

Top-K 是 AMC 前的頻域前處理，不是 channelizer，也不是 wideband spectrum sensing。

### 4.11 感測指標

程式位置：`src/sensing/ground_truth_metrics.py`

已實作的離線指標包括：

- Probability of Detection（Pd）。
- sample-level false-positive／false-negative rate。
- 邊界誤差。
- 單一或多 burst 的 region matching。
- batch aggregate sensing fields。

這些指標需要事先知道合法 burst 的真實位置，因此適合合成或 RadioML embedding 實驗。未標註的真實 capture 不會自動產生 ground truth。

尚未完成的現場指標包括：

- 每小時假事件數。
- noise-floor recovery time。
- dropped samples 與 backpressure。
- 每階段 latency。
- peak／steady-state memory。
- 事件佇列長度與分析等待時間。

### 4.12 結果輸出

| 元件 | 工作 | 狀態 |
|---|---|---|
| CSV writer | 寫入逐 segment 實驗結果 | 已完成 |
| Sensing plot | 畫出 IQ／energy／region 圖 | 已完成 |
| Batch aggregation | 彙整多組實驗 | 已完成於既有離線實驗腳本 |
| 事件資料庫 | 保存長期 RF 事件與原始證據 | 未完成 |
| Append-only evidence store | 防止證據被悄悄修改或刪除 | 未完成 |
| 即時 dashboard／告警 | 現場顯示與通知 | 未完成 |

---

## 五、完成度總表

### 5.1 已完成：離線 PoC 可執行的核心

1. 合成 IQ 產生。
2. RadioML 樣本載入與噪聲 embedding。
3. 單通道滑動視窗能量偵測。
4. occupied-region 擷取、合併與最短長度過濾。
5. `naive` 與 `max-energy` 視窗選擇。
6. AWN 輸入前處理及 `[N,2,128]` 格式轉換。
7. Dummy 與 real AWN adapter。
8. Classifier-only FGSM、PGD、CW adapter。
9. Dummy 與 real fixed-K Top-K adapter。
10. 離線 sensing ground-truth metrics。
11. CSV、圖表與批次實驗工具。

### 5.2 部分完成：程式存在，但仍未形成可部署鏈

1. `.cfile` loader 已存在，但沒有接入正式 real-AWN pipeline。
2. 真實 AWN／attack／Top-K backend 已有 adapter，但預設 dry run 可使用 dummy，失敗也可能 fallback。
3. `max-energy` 已完成，但尚未有多視窗事件彙整與攻擊偵測。
4. 外部程式有 Adaptive-K 函式，但正式 adapter 只接 fixed-K。
5. 離線真值指標已完成，但真實 capture 缺少標註與 metadata contract。

### 5.3 未完成：距離真實系統的主要缺口

1. Live USRP／UHD／GNU Radio／ZMQ 串流。
2. 長串流 chunking、rolling buffer 與跨 chunk detector state。
3. 正式 `.cfile` 到 real AWN／attack／Top-K 的接線與驗證。
4. Sample rate、中心頻率、gain、endianness、scaling 與 dropped-sample metadata。
5. Channelizer、STFT occupancy map 與真正 wideband spectrum sensing。
6. 抗 noise-floor poisoning 的偵測器。
7. 多視窗事件彙整、信心校準、unknown 與 abstention。
8. Adaptive-K／defense router 正式整合。
9. 對完整 detector／segmenter／defense／classifier 的 adaptive attacks。
10. 纜線注入與屏蔽 OTA 驗證。
11. Demodulation、decoding、CRC 與 protocol parsing。
12. 事件資料庫、證據完整性與現場告警介面。
13. 每階段 latency、memory 及 real-time feasibility 評估。
14. 自主頻譜配置或控制平面；近期設計建議維持被動 advisory 模式。

---

## 六、目前能說與不能說的研究結論

### 6.1 可以說

- 已建立離線的 energy-triggered AMC 實驗管線。
- 系統能在預先通道化的一維 complex IQ 串流上進行時間能量偵測。
- 系統能從偵測區域選出固定長度視窗，轉換成 AWN 所需的 `[N,2,128]`。
- 已有 real AWN、classifier-only attacks 與 fixed-K Top-K 的 integration boundary。
- 已能在具有已知 burst 位置的離線資料上計算 sensing metrics。
- Threat model 已把 detector、noise-floor estimator、segmenter 與 Top-K 納入攻擊面，而不只研究完美切割的 AMC tensor。

### 6.2 不能說

- 不能說已完成寬頻頻譜感測，因為沒有 channelizer。
- 不能說已能直接接 USRP 或 GNU Radio live stream。
- 不能說真實 `.cfile` 已通過 real AWN 端到端驗證。
- 不能把 dummy random logits 當成模型分類結果。
- 不能把只攻擊 AWN 的 FGSM／PGD／CW 稱為完整防禦白箱攻擊。
- 不能只用 normalized epsilon 宣稱實體 RF 隱蔽性；還需要 PSR、SJR、JNR、CFO、延遲與通道條件。
- 不能說系統能辨識裝置、解碼資料或驗證合法性。
- 不能說系統具備自主封鎖、切頻或資源配置能力。

---

## 七、建議的下一步順序

### 第一階段：先接通真實檔案路徑

1. 把 `load_iq_from_file` 接入 `ExperimentConfig` 與正式 pipeline。
2. 加入 dtype、endianness、有限值、振幅、sample rate、中心頻率及 gain 驗證。
3. 用合成 `.cfile` 做 round-trip test，確認它和相同陣列的 synthetic path 產生一致結果。
4. 用一份已知格式的短 SDR capture 跑過 real AWN 與 fixed-K Top-K。
5. 明確記錄 backend，若 real AWN 載入失敗則 fail closed。

### 第二階段：建立可信的事件層輸出

1. 加入多重疊視窗。
2. 進行跨視窗投票、時間平滑與不一致檢查。
3. 加入 confidence threshold、unknown 與 abstention。
4. 保存防禦前後視窗、guard samples、原始 regions 與共同 event ID。
5. 加入假事件 rate limit，但保留摘要，避免 rate limit 變成證據刪除工具。

### 第三階段：安全評估

1. 實作 noise-floor poisoning、phantom occupancy、event bridging 與 window hijacking。
2. 把 Top-K／Adaptive-K 與事件路由接入完整管線。
3. 進行 defense-aware adaptive attack，包括 BPDA／代理梯度、EOT、gradient-free 與多次重啟。
4. 報告 Pd、Pfa、事件 IoU、強健準確率、棄權率、PSR／SJR／JNR 與營運負載。

### 第四階段：受控實體驗證

1. 先用兩台 SDR、衰減器與合路器進行不輻射的纜線實驗。
2. 掃描 SNR、SJR、CFO、時間偏移、功率及 duty cycle。
3. 再進入法規允許的屏蔽 OTA 環境。
4. 最後才設計 GNU Radio／ZMQ streaming 與 wideband channelizer。

---

## 八、建議的口頭報告順序

### 8.1 一分鐘版本

> 這個研究不是完整的寬頻接收機，而是一個預先通道化單一 IQ 串流上的能量觸發 AMC。Spectrum sensing 先判斷何時有活動並建立事件，再選出 128 個 samples 交給 AWN 分類。Threat model 的主要攻擊者是附近持有商用 SDR 的加性 RF 發射者；他知道系統設計，但不能直接修改接收端 tensor，也沒有和合法發射器精確同步。除了讓 AWN 分錯類，攻擊者也可以污染噪聲底、製造假事件、橋接事件或劫持視窗。目前離線能量偵測、分段、AWN adapter、classifier-only attacks 與 fixed-K Top-K 已完成；live SDR、channelizer、正式 `.cfile` 接線、事件級拒判、完整 adaptive attack 與 OTA 驗證尚未完成。

### 8.2 五分鐘版本

1. 先說明應用：被動監測指定無線通道，保存事件與 IQ 證據。
2. 說明 spectrum sensing 和 AMC 的分工：前者找事件，後者猜調變。
3. 說明 A0、A1、A2，並把 A1 商用 SDR 攻擊者設為主要模型。
4. 說明攻擊面不只 AWN，也包含噪聲底、事件邊界和視窗選擇。
5. 展示正式 pipeline 與各元件狀態表。
6. 誠實區分離線完成、部分整合與未完成部署能力。
7. 最後提出 `.cfile` 接線、事件級可信決策、adaptive evaluation、纜線與屏蔽 OTA 的實作順序。

---

## 九、程式依據

本文件的完成狀態依下列目前程式路徑整理：

- 正式入口：`experiments/run_full_experiment.py`
- 流程協調：`src/utils/pipeline.py`
- 設定與 CLI：`src/utils/config.py`
- IQ source：`src/sensing/iq_source.py`
- RadioML source：`src/sensing/radioml_source.py`
- 能量偵測：`src/sensing/energy_detection.py`
- 分段：`src/sensing/segmentation.py`
- AWN 前處理：`src/sensing/normalize.py`
- 感測指標：`src/sensing/ground_truth_metrics.py`
- AWN adapter：`src/adapters/awn_adapter.py`
- Attack adapter：`src/adapters/attack_adapter.py`
- Top-K adapter：`src/adapters/topk_adapter.py`
- 較舊 standalone PoC：`scripts/sdr_sensing_to_awn_poc.py`
- 部署盤點：`docs/DEPLOYMENT_READINESS.md`
- 完整威脅模型：`docs/PRACTICAL_SPECTRUM_SENSING_THREAT_MODEL_ZH_TW.md`

## 十、總結

目前成果的核心是離線、單通道、能量觸發的 AMC 實驗管線。Spectrum sensing 負責把連續 IQ 轉成可分析事件；AWN 負責提供調變類型；Top-K 是 AMC 前的頻域前處理；attack adapter 用來測試分類器強健性。

研究下一步不應先擴大宣稱，而是先補齊正式 `.cfile` 接線、真實擷取 metadata、fail-closed backend 與事件級可信決策，再進行完整 adaptive attack、纜線注入和屏蔽 OTA。等 channelizer 實作並驗證後，才適合把系統稱為真正的 wideband spectrum sensing system。
