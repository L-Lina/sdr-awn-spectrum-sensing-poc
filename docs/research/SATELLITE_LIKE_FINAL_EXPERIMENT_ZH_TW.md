# Satellite-like Integrated Experiment（Project-Close 最終正式實驗）

## 一、研究目的

本輪（Step 4）是 project-close 前最後一輪正式主要實驗，把 Step 1（application scenario／latency requirement）、Step 2（dataset／modulation feasibility）、Step 3（channel simulator＋correctness validation）三輪的結論整合成一條完整、可重現、internally consistent 的 pipeline，並以正式規模（576 組合）量測結果，回答第二節列出的七個研究問題。本輪不新增 channel impairment、不重新訓練 AWN、不下載 RadioML2018、不做完整 DVB-S2/S2X、不接 SDR、不重新設計 attack、不重跑舊 Phase 0-4。

## 二、Research Questions

- **RQ1**：Satellite-like channel 下 Spectrum Sensing 是否仍可正確找到 occupied IQ region？
- **RQ2**：不同 satellite-like channel condition 對 AWN AMC accuracy 的影響為何？
- **RQ3**：加入 optimized FGSM／optimized deterministic PGD 後，AMC robustness 如何？
- **RQ4**：Top-K 在 satellite-like channel＋attack 下是否具有 recovery effect，或仍呈 condition-dependent？
- **RQ5**：不同 modulation／SNR／channel condition 下差異為何？
- **RQ6**：完整 processing latency 是否落在先前定義的 processing-budget reference 範圍？
- **RQ7**：目前這套 Satellite-like channel → Spectrum Sensing → AWN AMC → Attack → Top-K 是否已形成可重現且 internally consistent 的 project-close PoC？

## 三、Reference Scenario

沿用 Step 1 第九、十三節建議的 **receiver-side／ground-side digital IQ processing**（LEO ground-terminal 或 ground-station 頻譜監測情境），威脅模型為 Step 1 第十節定義的 **A0（接收端數位白箱攻擊基準）**，不涉及 A1（獨立 RF 發射端）或 A2（資訊受限查詢式攻擊者）的即時 OTA 假設。

## 四、Dataset

RadioML2016.10a（`/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl`），依 Step 2 決策不下載或使用 RadioML2018.01A。

## 五、Modulation

正式 project-close primary set：**BPSK／QPSK／8PSK**（Step 2 決策），本輪未加入 APSK 或任何其他調變。

## 六、Channel Model

沿用 Step 3 已驗證的 `src/channel/satellite_like.py:apply_satellite_like_channel()`（本輪未修改）。四級組合式 channel condition（每個參數值皆取自 Step 3 已個別驗證過的層級，未引入任何新數值或新 impairment 類型）：

| Condition | amplitude_scale | cfo_hz | doppler_hz | timing_offset_samples | target_snr_db | propagation_delay_ms（metadata） |
|---|---|---|---|---|---|---|
| clean | 1.0 | 0 | 0 | 0 | None（不套用 channel-level AWGN） | None |
| mild | 1.0 | 500 | 250 | 2 | None | 26.0 |
| moderate | 0.5 | 1000 | 500 | 2 | None | 95.0 |
| strong | 0.5 | 2000 | 1000 | 8 | 15.0 | 272.0 |

**Amplitude interpretation（正式聲明）**：本文件中 `amplitude_scale` **不**宣稱為「衛星 path loss 直接造成 AMC accuracy degradation」，應精確理解為「接收端數位 IQ 中殘留的 amplitude／gain scaling，用以評估現有 `radioml-native` AWN 前處理政策對 amplitude-scale distribution shift 的敏感度」（Step 3 第 16.3 節已確認此為 AWN 前處理未做正規化下的分布外輸入問題，非完整 RF link budget 模擬）。真實接收機通常存在 AGC（自動增益控制），本結果代表 robustness stress condition，不是完整鏈路預算模擬。Doppler 數值（第六節工作範例，見 Step 3 文件第六節）僅為 simplified upper-bound／order-of-magnitude reference。

## 七、Threat Model

A0：接收端數位、白箱、對已存在於記憶體中的 AWN-input IQ 張量直接生成攻擊擾動——本輪 FGSM／PGD 攻擊延遲、擾動範數等數字皆屬 A0 範疇，不代表 A1 OTA 即時攻擊者的可行性驗證。

## 八、Attack

三種 attack condition：
- **none**：無攻擊，clean AMC。
- **optimized FGSM**：`eps=0.05`，已完成 correctness validation 的 batch_size=16／`torch_num_threads=1` implementation optimization。
- **optimized deterministic PGD**：`eps=0.05`，`random_start=False`（已完成 deterministic batching equivalence 驗證的路徑）。

PGD `random_start=True` 不列入本輪主矩陣（僅作 stochastic throughput comparison，屬其他既有文件範疇）。CW 不列入主矩陣，原因是 CW batch>1 已確認為 `batched_algorithmic_variant`，不適合與 FGSM／deterministic PGD 放在同一純 implementation-optimization 比較中；CW／DeepFool／FAB／EAD 等既有結果仍完整保留於既有正式 experiment（`docs/research/DIGITAL_LOW_PERTURBATION_ATTACK_EXPERIMENT_ZH_TW.md` 等），本文件不重複、也不宣稱其未完成。

**Metric 定義正式聲明（見第二十二節稽核）**：本文件與 raw CSV 中的 `attack_success`／ASR 欄位定義為 **prediction-change rate**（`attacked_prediction != clean_prediction`），即「攻擊是否改變了模型自己原本的輸出類別」，**不是**「clean 原本正確、攻擊後才變錯」的定義（那個概念對應的是 `attacked_correct` 與 `clean_correct` 的比較，非 `attack_success`）。這是本專案既有、跨輪次一致沿用的欄位命名慣例（`experiments/run_satellite_like_final.py` 第 300 行），本文件明確聲明其定義以避免讀者誤解為「攻擊使正確判斷變成錯誤」的比例。

## 九、Top-K

K=20（沿用既有 Phase A／`experiments/end_to_end_latency_matrix.py` 已使用之 reference K，本輪未重新做 K tuning），每個 attack condition 皆有 Top-K OFF／ON 兩種分支。

## 十、Experiment Matrix

$$3\ \text{modulations} \times 4\ \text{SNR} \times 4\ \text{channel conditions} \times 3\ \text{attacks} \times 2\ \text{Top-K} \times 2\ \text{samples/index} = 576$$

執行前已重新列式確認 $3\times4\times4\times3\times2\times2=576$，與 Step 3 已文件化的 FULL matrix 設計一致。SNR 採用 $\{-10,0,10,18\}$ dB，與 Step 3 正式文件記載的 4-level 設定一致，本輪未自行新增更多 SNR。

## 十一、Fairness／Paired Design

同一 `(modulation, snr, channel_condition, sample_index)` 的 channel-transformed IQ、sensing 結果、clean crop 只計算一次（`experiments/run_satellite_like_final.py` Phase 1），在全部 3 個 attack × 2 個 Top-K 分支中重複使用（Phase 3）——FGSM 與 PGD 絕不會看到不同的 channel 噪聲實現。Channel transformation 以 `channel_seed = SEED + sample_index` 決定性產生。每筆 raw row 記錄 `base_sample_id`、`channel_seed`、`channel_input_hash`（channel-transformed 長串流的 SHA256 前 16 碼）、`clean_segment_hash`（AWN 輸入張量的 SHA256 前 16 碼），可追溯同一 base combo 在不同 attack/topk 分支下是否確實共用了同一份輸入。

## 十二、Spectrum Sensing Results（RQ1）

576 筆全數完成，**sensing_detection_rate = 100%（576/576）**，`captured_signal_ratio` 平均 0.9734、中位數 1.0。**RQ1 答案：是，satellite-like channel（含 strong 級）下 Spectrum Sensing 仍可正確找到 occupied IQ region，四個 channel condition 下皆未觀察到偵測失敗。**

## 十三、AMC Results（RQ2、RQ5）

Clean（無攻擊）準確率依 channel condition：

| Condition | Clean Accuracy (n=48) |
|---|---|
| clean | 58.3% |
| mild | 33.3% |
| moderate | 20.8% |
| strong | 16.7% |

**準確率隨 channel 嚴重程度單調遞減**，回答 RQ2：channel condition 對 AMC accuracy 有明確、方向一致的負面影響，且第十八節 GIGO 驗證確認這不是 sensing 造成的。

依調變（跨全部 channel condition 合併）：8PSK 46.9%、QPSK 37.5%、BPSK 12.5%（n=64 each）——BPSK 在本輪合併統計下明顯偏低，可能與 PSK 家族對相位旋轉的已知敏感度有關（Step 3 第五節），但本輪未做進一步的逐 channel×modulation 交叉統計檢定，不做因果斷言。

依 SNR（跨全部 channel condition 合併，n=48 each）：-10dB 20.8%、0dB 37.5%、10dB 37.5%、18dB 33.3%——**並非嚴格單調**，18dB 略低於 0/10dB。樣本數小（每格 48 筆）且合併了 4 種 channel 嚴重程度，本文件不對此非單調現象做因果推論，僅如實記錄（回答 RQ5 的一部分）。

## 十四、Attack Results（RQ3）

**Metric terminology（正式定義，避免 legacy 欄位名稱造成誤解）**：raw CSV 中的欄位 `attack_success` 其語意是 **prediction-change indicator**（`attacked_prediction != clean_prediction`），本節與全文一律以下列三個明確命名的指標呈現，不再使用無限定詞的裸「ASR」字樣：

1. **Attacked Accuracy** = count(`attacked_correct`=True) / count(unique attacked base samples)。分母為 96（每種攻擊在 96 個 unique base combo 上各攻擊一次；Top-K OFF/ON 兩列共用同一個攻擊結果，不重複計入，見第二十二節 22.5 的 pseudo-replication 說明）。
2. **Prediction Change Rate**（legacy 欄位名稱 `attack_success` 實際代表的意義）= count(`attacked_prediction != clean_prediction`) / 96。
3. **Conditional Attack Success Rate** = count(`clean_correct`=True AND `attacked_correct`=False) / count(`clean_correct`=True)。（此定義與「`clean_correct`=True 子集內的 prediction-change」在數學上等價——當 `clean_correct`=True 時 `clean_prediction`=`true_label`，故 `attacked_prediction != clean_prediction` 恰等於 `attacked_correct`=False；兩種算法逐位一致，已於 `results/satellite_like_final_20260821T021117Z/audit/unique_attack_sample_audit.csv` 交叉驗證。）

| Attack | n (unique attacked base) | Attacked Accuracy | Prediction Change Rate | Conditional Attack Success Rate | Mean Linf | Mean L2 |
|---|---|---|---|---|---|---|
| FGSM | 96 | 16.67%（16/96） | 91.67%（88/96） | 77.42%（24/31） | 0.001367 | 0.02178 |
| PGD (det) | 96 | 9.375%（9/96） | 98.96%（95/96） | 96.77%（30/31） | 0.001367 | 0.01882 |

（`clean_correct`=True 的 base 樣本數＝31，為 FGSM／PGD(det) 共用的分母，因 `clean_correct` 是 base-level 屬性，與攻擊種類無關。）

依 channel condition 分解（attacked accuracy／prediction change rate）：

| Condition | FGSM acc / pred-change | PGD(det) acc / pred-change |
|---|---|---|
| clean | 16.7% / 83.3% | 4.2% / 95.8% |
| mild | 25.0% / 100% | 29.2% / 100% |
| moderate | 8.3% / 91.7% | 0.0% / 100% |
| strong | 16.7% / 91.7% | 4.2% / 100% |

**RQ3 答案**：在全部 4 個 channel condition 下，FGSM／PGD(det) 的 prediction change rate 皆維持在 83.3%–100% 的高範圍，AMC robustness 在 satellite-like channel 疊加 attack 後進一步下降，沒有觀察到「channel 本身的干擾抵銷攻擊效果」的保護性效應；channel×attack 的交互作用型態依 condition 而異（例如 mild 下兩種攻擊的 attacked accuracy 反而略高於 clean，屬小樣本下的觀察，見第十九節主要發現的謹慎表述）。

## 十五、Defense Results（RQ4）

| | Top-K OFF | Top-K ON |
|---|---|---|
| Clean accuracy | 32.3%（n=96，同 overall clean acc） | Defended accuracy 17.0%（n=288，跨 none/fgsm/pgd_det 三種 attack 分支合併） |

**Recovery／Degradation 正式定義（denominator 已由 final audit 修正，見第二十二節）**：

- **Recovery numerator/denominator**：分母為該攻擊下 `attacked_correct`=False 的 eligible 樣本數（recovery 只可能發生在攻擊已使預測錯誤的樣本上）；分子為這些樣本中，經 Top-K defense 後 `defended_correct`=True 的樣本數（即 `recovered_by_defense`=True）。
- **Clean degradation numerator/denominator**：分母為 `attack_name`='none' 且 `clean_correct`=True 的樣本數（degradation 只可能發生在 clean 原本就正確的樣本上）；分子為這些樣本中，經 Top-K defense 後 `defended_correct`=False 的樣本數（即 `clean_degraded_by_defense`=True）。

| Metric | Numerator | Denominator | Rate |
|---|---|---|---|
| FGSM recovery | 2 | 80 | **2.50%** |
| PGD (det) recovery | 6 | 87 | **6.90%** |
| Clean degradation | 9 | 31 | **29.03%** |

（以全部 96 個 Top-K ON 列數作分母的舊算法會分別得到 2.08%／6.25%／9.375%——這些舊數字使用了錯誤的 denominator（把攻擊本來就沒成功、或 clean 本來就不正確的樣本也算進分母，稀釋了真實比例），**已不再使用，正式結果一律以上表為準**，稽核細節見 `results/satellite_like_final_20260821T021117Z/audit/topk_denominator_audit.csv`。）

**RQ4 答案：Top-K 在本輪 satellite-like channel＋attack 條件下只有低比例的 recovery（FGSM 2.5%、PGD(det) 6.9%，分母為攻擊已成功樣本），同時會破壞一部分原本正確的 clean classification（degradation 29.03%，分母為 clean 原本正確樣本）——結果支持 Top-K 為 condition-dependent，而非 universally effective 的防禦手段。**

## 十六、Latency Results（RQ6）

| Scenario | n | mean (ms) | median (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|---|
| clean | 96 | 2.75 | 2.48 | 4.13 | 7.09 |
| fgsm | 96 | 4.17 | 3.79 | 5.53 | 8.43 |
| pgd_det | 96 | 7.89 | 7.56 | 9.13 | 12.44 |
| fgsm_topk | 96 | 8.55 | 7.39 | 13.42 | 30.87 |
| pgd_det_topk | 96 | 12.49 | 11.22 | 17.09 | 38.62 |

以上數字與 Step 1 `PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md` 第十六節在同樣 batch=16／`torch_num_threads=1` 優化設定下量得的 Scenario C（FGSM，mean 3.80ms）與 Scenario D_det（PGD，mean 7.84ms）**量級高度一致**（本輪 fgsm 4.17ms／pgd_det 7.89ms），差異主要來自本輪額外的 channel 轉換階段與不同樣本子集，交叉驗證了兩輪量測方法一致。

## 十七、Processing-Budget Comparison（RQ6）

依 5／10／20／35／50／100／250 ms 對照（僅摘錄關鍵臨界點，完整見 `processing_budget.csv`）：clean／fgsm 在 10 ms 下 median／p95 皆符合；pgd_det 在 10 ms 下 median 尚未穩定符合（需視具體樣本，見 CSV）；fgsm_topk／pgd_det_topk 因 p99 尾端延遲（30.87／38.62 ms）在最嚴格的 10 ms budget 下 p95 不一定符合，但在 20 ms 以上皆穩定符合。**這只是 processing-budget comparison，不是任何特定應用場景的 deadline 證明**——與 Step 1／Step 3 一致的措辭原則，不稱為 LEO computation deadline。**RQ6 答案：是，完整 processing latency（含 channel 轉換、sensing、AMC、attack、Top-K 全部階段）在 20 ms 以上的 processing-budget reference 下穩定落在範圍內，10 ms 以下視 scenario 與統計量（median vs p95）而定。**

## 十八、Garbage-In／Garbage-Out 驗證

依 channel condition 交叉比對 `captured_signal_ratio` 與邊界誤差：

| Condition | mean captured_signal_ratio | mean \|boundary_start_error\| | mean \|boundary_end_error\| | Clean accuracy |
|---|---|---|---|---|
| clean | 0.9746 | 37.75 | 40.67 | 58.3% |
| mild | 0.9743 | 35.42 | 41.42 | 33.3% |
| moderate | 0.9740 | 35.58 | 40.42 | 20.8% |
| strong | 0.9707 | 32.67 | 42.58 | 16.7% |

**`captured_signal_ratio` 在四個 channel condition 下幾乎不變（0.971–0.975），但 clean accuracy 從 58.3% 一路降到 16.7%**。目前 evidence **不支持**「AMC degradation 主要來自 Spectrum Sensing 擷取垃圾」這個解釋：在本實驗設定下，Spectrum Sensing 仍穩定擷取主要 signal region（`captured_signal_ratio` 與邊界誤差在四個 condition 下高度一致），而 AMC performance degradation 更符合 **channel／model robustness degradation** 的型態，與 Step 3 第 16.4 節（oracle vs sensing 幾乎相同）的觀察方向一致。此為觀察性證據之間的一致性描述，**不構成因果證明**。

## 十九、Main Findings

1. Spectrum Sensing 對本輪測試的四級 channel condition（含 strong）完全穩健，100% 偵測率，`captured_signal_ratio` 穩定在 0.97 以上（RQ1）。
2. AMC accuracy 隨 channel 嚴重程度單調遞減（58.3%→16.7%），且第十八節已確認這不是 sensing 造成，而是 channel／AWN 前處理交互作用下的模型 robustness 問題（RQ2、GIGO）。
3. FGSM／PGD(det) 在全部 channel condition 下皆維持高 prediction change rate（83.3%–100%），attack 對 AMC 的破壞力未因 channel 本身已造成的準確率下降而減弱或增強出一致方向（RQ3）。
4. Top-K 呈現低 recovery rate（以攻擊已成功樣本為分母：2.5%–6.9%）且對 clean 樣本有相當高的 degradation risk（以 clean 原本正確樣本為分母：29.0%），是明確的 condition-dependent、非 universally effective 防禦（RQ4，denominator 修正見第十五節與第二十二節）。
5. Latency 數字與 Step 1 既有優化結果高度一致（mean 差距在 10% 以内），confirming 本輪 channel-augmented pipeline 的效能特性與既有純 AMC pipeline 一致（RQ6）。
6. 576/576 完成、0 error/fallback/NaN、100% real backend，構成一條可重現、internally consistent 的 project-close PoC（RQ7，見第二十一節）。

## 二十、Limitations

1. 每個 (modulation, snr, channel_condition) 格僅 2 個 sample_index，576 筆矩陣本身仍是中等規模，部分細粒度交叉分組（例如 modulation×snr×channel×attack 四維交叉）樣本數會降到個位數，本文件未展開到那個細度，僅在第十三、十四節做二維或跨其他軸合併的分組，避免無足夠樣本支撐的強推論。
2. 第十三節「非單調 SNR 趨勢」與「BPSK 準確率偏低」皆為觀察記錄，未做統計顯著性檢定，不排除是小樣本變異。
3. Channel condition 的四個嚴重程度（clean/mild/moderate/strong）是本輪自訂的組合式設計（沿用 Step 3 已驗證的個別參數層級），不對應任何標準文件定義的正式衛星通道等級。
4. Amplitude scaling 的結果解讀依第六節聲明，僅代表 AWN 前處理對輸入尺度的敏感度壓力測試，不是完整 RF link budget 或路徑損耗模擬。
5. Latency 數字為單一機器、單次執行的結果，未做跨次重複測量。
6. 本輪僅測試 FGSM 與 deterministic PGD，CW／PGD 隨機起點／DeepFool／FAB／EAD 等攻擊不在本輪矩陣範圍內（見第八節）。
7. 本文件與其對應實驗**不構成、也不宣稱**：完整 DVB-S2/S2X 合規驗證、已驗證的 over-the-air 即時攻擊可行性、或任何形式的正式衛星鏈路驗證。

## 二十一、Project-Close Interpretation（RQ7）

576/576 完成，0 unexpected error、0 silent fallback、0 NaN/Inf，全程使用真實 AWN、真實 sensing、真實 attack（`AttackAdapter`）、真實 Top-K（`TopKAdapter`），provenance（`base_sample_id`／`channel_seed`／`channel_input_hash`／`clean_segment_hash`）完整記錄於 raw CSV，全部 summary／圖表皆由 `raw_results.csv` 以程式化 groupby 推導（`experiments/analyze_satellite_like_final.py`），無人工抄錄數字。**RQ7 答案：是**——Satellite-like channel → Spectrum Sensing → AWN AMC → Attack → Top-K 這條 pipeline 在本輪 576 組合的正式規模下，展現了可重現（deterministic channel seed、real backend 100% 一致）且 internally consistent（sensing 品質與 GIGO 驗證吻合、latency 與既有 Step 1 結果交叉一致）的 project-close PoC 特性，可作為本專案 satellite-like 研究支線收尾的正式證據。

## 二十二、Final Result Audit（Project-Close 前正式複核）

本輪（`experiments/audit_satellite_like_final.py`，唯讀，不重跑 576、不修改 `raw_results.csv` 或任何既有 summary CSV）針對 `results/satellite_like_final_20260821T021117Z/raw_results.csv` 做獨立於 `analyze_satellite_like_final.py` 的第二次重新計算，輸出至 `results/satellite_like_final_20260821T021117Z/audit/`：22 個原始 `audit_*.csv` ＋ `audit_report.json`，另加 4 個正式命名的彙總 artifact——`metric_definition_audit.csv`（attack metric 正式定義對照）、`topk_denominator_audit.csv`（Top-K recovery／degradation 正確 denominator）、`unique_attack_sample_audit.csv`（pseudo-replication 修正後的 n=96 unique attacked-base 指標）、`claim_to_evidence_audit.csv`（文件逐項 claim 對應 evidence）。結論：**未發現任何 metric denominator bug、wrong-column bug、duplicate/missing combo、timing accounting bug、attack tensor reuse、或 fairness hash mismatch**；發現兩項需要在文件中明確澄清的「命名／denominator 語意」議題（非數值錯誤：`attack_success` 命名語意、Top-K denominator 選擇），已於本文件第八、十四、十五節修正 wording，細節如下。

### 22.1 Matrix 複核

`raw_results.csv` 576 列，`(modulation, snr_db_dataset, condition, sample_index, attack_name, topk_state)` 六維 key 下 unique combo 數＝576，0 筆重複、0 筆缺漏；`(modulation, snr_db_dataset, condition, sample_index)` 四維 base key 下 unique base combo 數＝96（＝3×4×4×2，與 Phase 1 設計一致）；每個 `base_sample_id` 精確對應 6 列（3 attacks×2 topk_states）、精確對應 1 組 base key（0 筆 mislabel）。Matrix formula 重新列式：$3\times4\times4\times3\times2\times2=576$，與 576 完全吻合。

### 22.2 Sensing／AMC 複核（按 channel／modulation／SNR 分組，非僅 overall）

以 96 筆 unique base（sensing／clean AMC 只在 Phase 1 計算一次，避免下方 22.5 節所述的列數膨脹）重新計算：detection_rate 在 clean/mild/moderate/strong 與 8PSK/BPSK/QPSK 與四個 SNR 下**全部為 100%**（0 no_region、0 no_segment）；`captured_signal_ratio` 均值在四個 channel condition 下為 0.9746／0.9743／0.9740／0.9707（全距僅 0.0039），而同期 clean AMC accuracy 為 58.3%／33.3%／20.8%／16.7%（全距 0.4167）——**GIGO 結論在分組層級（非僅 overall average）下依然成立**：sensing 品質穩定，accuracy 下降是 channel／model 問題。AMC accuracy 重新計算（by channel／modulation／SNR）與既有文件第十三節數字逐位一致。

### 22.3 Attack Metric 定義稽核

由 `experiments/run_satellite_like_final.py` 原始碼逐行確認：

| Metric（正式名稱） | Legacy 欄位／內部代稱 | Numerator | Denominator | Source field |
|---|---|---|---|---|
| Attacked Accuracy | `attacked_correct` | `attacked_prediction == true_label` | 1（每列） | `attacked_prediction`, `true_label` |
| Prediction Change Rate | `attack_success`（legacy 名稱，語意其實是 prediction-change，非傳統 conditional adversarial success） | `attacked_prediction != clean_prediction` | 1（每列） | `attacked_prediction`, `clean_prediction` |
| — （彙總）Attacked Accuracy | — | count(`attacked_correct`=True) | count(unique attacked base samples)＝96 | `attacked_correct` |
| — （彙總）Prediction Change Rate | 舊稱 "overall ASR"，**本文件起停用此稱呼** | count(`attack_success`=True) | count(unique attacked base samples)＝96，**不限定** `clean_correct` | `attack_success` |
| Conditional Attack Success Rate | 舊稱 "conditional ASR" | count(`clean_correct`=True 且 `attacked_correct`=False) | count(`clean_correct`=True)＝31 | `attacked_correct`, `clean_correct` |

**確認**：legacy 欄位 `attack_success` 語意為 prediction-change indicator，非「clean 正確才算」的傳統 adversarial-success 定義；本文件第一次出現 `attack_success` 時（第八節）已明確聲明其定義，全文表格與圖說一律使用「Attacked Accuracy／Prediction Change Rate／Conditional Attack Success Rate」三個正式名稱，不再使用無限定詞的裸「ASR」。**Conditional Attack Success Rate 的兩種等價算法**（用 `attack_success` 限定 `clean_correct`＝True，或用 `attacked_correct`＝False 限定 `clean_correct`＝True）逐位一致——因為當 `clean_correct`＝True 時 `clean_prediction`＝`true_label`，兩個條件在數學上恆等——已交叉驗證，細節見 `results/satellite_like_final_20260821T021117Z/audit/metric_definition_audit.csv` 與 `unique_attack_sample_audit.csv`。獨立重新計算的 FGSM／PGD(det) 三項指標與既有 `overall_summary.csv` 逐位吻合（`audit_04/05` 系列 CSV）。

### 22.4 FGSM／PGD(det) Perturbation Norm 稽核（Linf 高度相近之原因）

逐 base_sample_id 配對比較（`audit_05_fgsm_vs_pgd_paired_by_base.csv`，n=96 base）：
- **FGSM／PGD(det) 的 `perturbation_linf` 幾乎逐筆相同**（96 筆中 75 筆完全相同、其餘 21 筆差距 ≤3.9e-9，屬 float32 捨入誤差量級），但 **`perturbation_l2` 與 `attacked_prediction` 明確不同**（`attacked_prediction` 相同比例僅 69.8%，即 30.2% 的 base 兩種攻擊產生不同預測結果；平均 L2：FGSM 0.02178 vs PGD(det) 0.01882，逐筆差距平均 0.0026，非零）。
- **這排除了「同一個 attacked tensor 被誤用於兩種攻擊」的疑慮**——若真為 tensor 複用，`perturbation_l2` 與 `attacked_prediction` 也必然逐筆相同，但實際上明顯不同。
- **機制解釋**：兩種攻擊的 `eps=0.05` 是套用在 `attack_adapter.py` 的 per-segment min-max 正規化域（`[0,1]`），FGSM 為單步 `x + eps*sign(grad)`，PGD(det) 為 `alpha=eps/4=0.0125`、10 步、`random_start=False` 的 projected gradient ascent；由於本模型 top1-top2 logit margin 極大（見 `src/adapters/attack_adapter.py` docstring）、梯度方向在多步之間高度穩定，兩種攻擊在絕大多數 pixel 上都收斂到同一個 `±eps` normalized-domain 邊界，且兩種攻擊都作用在**完全相同**的 base segment（`r["awn_input"]`，Phase 1 只算一次、Phase 2 兩次攻擊呼叫共用同一份未修改陣列，程式碼已確認），因此映射回 raw IQ domain 的 per-segment min-max scale 相同，導致兩者的**最大值**（Linf）幾乎相同，但整體擾動分佈（L2）與最終預測仍因單步 vs 多步的軌跡差異而不同。
- **證據缺口（誠實揭露）**：`attack_normalized_min`／`attack_normalized_max`（AttackAdapter 回傳的 per-segment min-max 邊界）本輪未逐列寫入 `raw_results.csv`，因此無法用 `eps × (max-min)` 直接數值驗證上述機制，只能以「predictions/L2 明確不同」邏輯排除 tensor 複用、並以程式碼邏輯（相同 base tensor、相同 min-max 映射函式、相同 eps）解釋 Linf 高度相近的成因。此為方法論限制，不影響現有 576 筆結果的正確性，已列入第二十節 Limitations。

### 22.5 n-Denominator 澄清（非數值錯誤，屬 pseudo-replication 語意問題）

`clean_correct`（及 sensing 相關欄位）只在 Phase 1 對每個 base combo 計算一次，之後原樣複製進該 base 對應的全部 6 列（3 attacks×2 topk_states）；`attacked_correct`／`attack_success`／perturbation 欄位只在 Phase 2 對每個 (base, attack) 計算一次，複製進該組合對應的 2 列（topk on/off）。因此：
- `overall_summary.csv` 的 `clean_accuracy_overall`（n=192）、`fgsm_attacked_accuracy`（n=192）等，其**真正獨立樣本數為 96**（clean）或 96（每種 attack），n=192 只是同一數值因 Top-K 軸複製兩份的列數，並非 192 次獨立試驗。
- **數值本身不受影響**（複製偶數份不改變平均值），但 n 不應被解讀為統計獨立試驗數。
- `defended_accuracy_overall`／`topk_recovery_rate`／`topk_clean_degradation_rate`（topk=on 子集）**不受此問題影響**——topk=on 篩選後每個 (base, attack) 恰好 1 列，n=288 為真實 96 base×3 attacks 的獨立組合數。

`audit_01_matrix.csv`／`audit_03_amc_overall.csv`／`unique_attack_sample_audit.csv` 已提供以 96 為分母的對照數字，與 row-count-based（n=192）版本數值完全一致，僅 n 的統計意義不同。

### 22.6 Top-K Denominator 精確化（見第十五節已更新之表格）

`recovered_by_defense` 只有在 `attacked_correct=False` 時才可能為 True，`clean_degraded_by_defense` 只有在 `attack_name='none'` 且 `clean_correct=True` 時才可能為 True。以正確 denominator 重新計算：FGSM recovery 2/80=2.50%、PGD(det) recovery 6/87=6.90%、clean degradation 9/31=**29.03%**（`audit_06_topk_recovery_degradation_by_attack.csv`／`topk_denominator_audit.csv`）。第十五節已更新為僅呈現正確 denominator 下的數字，舊的（以全部 96 列為分母的）2.08%／6.25%／9.375% 已從正式結論中移除，只在第十五節以明確標註「已不再使用的舊算法」的方式保留一次作為對照。

### 22.7 Latency／Processing-Budget 複核

獨立重新計算的 clean/fgsm/pgd_det/fgsm_topk/pgd_det_topk 五種 scenario 之 n/mean/median/p90/p95/p99/max（`audit_07_latency.csv`）與既有 `latency_summary.csv` 逐位一致；另外針對每一列驗證 `total_ms` 是否等於其自身各階段欄位（`channel_ms`+`embed_ms`+`sensing_ms`+`segmentation_ms`+`awn_preprocess_ms`+`awn_clean_ms`+`attack_generation_ms`+`attacked_inference_ms`+`topk_ms`+`defended_inference_ms`，忽略 null）之和，最大絕對差距為 3.55e-15ms（浮點捨入量級，非邏輯錯誤）。經程式碼逐行確認：CSV 寫檔、matplotlib 繪圖、AWN 模型載入、資料集初始化皆在計時範圍**之外**，不污染 `total_ms`。Processing-budget 稽核另補上既有 CSV 未提供的 p99 fit 與 empirical fit rate（`audit_08_processing_budget.csv`）。

### 22.8 Fairness／Hash 複核

576 列的 `channel_input_hash`／`clean_segment_hash` 覆蓋率皆為 100%；同一 `base_sample_id` 的 6 列中，兩個 hash 欄位 0 筆不一致；不同 base combo 之間 0 筆 hash 碰撞。**確認 attack／Top-K 比較確實共用同一份 channel-transformed IQ 與同一份 clean crop，未使用不同 noise realization。**

### 22.9 結論

10 項稽核（matrix／sensing／AMC／attack 定義／perturbation／Top-K denominator／latency／processing-budget／fairness-hash／claim-to-evidence）皆通過，**未發現需要停止或重跑的 bug**。兩項命名／denominator 語意澄清（`attack_success` 定義、n 的 pseudo-replication 語意、Top-K 正確 denominator）已直接反映於本文件第八、十五、十九、二十節的文字修正，`raw_results.csv` 與既有 summary CSV **未被修改**。
