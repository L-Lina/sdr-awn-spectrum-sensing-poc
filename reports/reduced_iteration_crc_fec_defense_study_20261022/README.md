# reduced_iteration_crc_fec_defense_study_20261022

## 1. 內容

本資料夾為下列研究之正式整理：四種以受限擾動或降低擾動為設計目標的對抗攻擊（FGSM、PGD、CW、EAD）之 reduced-iteration 評估、synthetic TX/RX 資料域之 CRC-8 與 FEC payload recoverability、Fixed Top-K（K10、K20、K30、K40）與 Adaptive-K 之 post-attack defense evaluation，以及 clean-only defense control。研究報告為 [`RESEARCH_REPORT_ZH_TW.md`](RESEARCH_REPORT_ZH_TW.md)。

## 2. 產生方式

本資料夾全部內容（`data/`、`figure_sources/`、`figures/`、`RESEARCH_REPORT_ZH_TW.md`、`README.md`）由下列指令自原始 results 重新產生，數值未人工輸入：

```
~/adversarial-rf/.venv/bin/python scripts/build_reduced_iteration_crc_fec_defense_study.py
```

該 script 僅讀取下列五個 run 目錄；缺少必要來源檔或任一 run 之 validation 非 PASS 時即中止。報告中之定性敘述以資料條件檢查保護，來源資料改變時 build 失敗而非產生過時文字。

## 3. 來源

| 代號 | Run 目錄 | 內容 |
|---|---|---|
| R1 | `results/reduced_iteration_attacks_vm_20261022_185732/` | RadioML 2016.10a AMC-only reduced-iteration（88 樣本，27 profiles） |
| R2 | `results/synth_crc_fec_integration_20261022_190837/` | Synthetic TX/RX CRC/FEC chain 整合驗證 |
| R3 | `results/reduced_iteration_crc_fec_vm_20261022_190944/` | Synthetic TX/RX reduced-iteration × CRC × FEC（SNR 18 dB，25 bursts/cell，26 unique profiles） |
| R4 | `results/reduced_iteration_crc_fec_defense_vm_20261022_192248/` | 12 attack profiles × 6 defenses × CRC × FEC（28,800 rows） |
| R5 | `results/clean_crc_fec_defense_vm_20261022_104250/` | Clean-only defense control（2,400 rows，無攻擊） |

各來源檔之路徑、位元組數與 SHA-256 列於 `data/source_manifest.csv`。`results/` 不納入版本控制；完整 run 目錄另以 `research_results_20261022_default_backend.zip` 封存。先前之 `research_results_20261022.zip` 為已被取代之歷史封存，不作為本研究之來源。

## 4. `data/`

所有 CSV 為 UTF-8（無 BOM）。每一列之 `source_file` 欄記錄其來源檔；分母為 0 之比例指標留空，原因記錄於 `na_reasons`。所有 aggregate 均排除 payload-degenerate frame（QPSK 與 PAM4 於 FEC on）。

| 檔案 | 內容 | 報告章節 |
|---|---|---|
| `amc_reduced_iteration_summary.csv` | R1 四種攻擊 27 個 profile 之 conditional ASR、準則門檻、latency median／mean／p95、latency 降低、Linf／L2 | 3 |
| `crc_fec_integration_validation_summary.csv` | R2 parity、clean chain、attack chain、fail-closed 結果 | 4.1 |
| `crc_fec_reduced_iteration_summary.csv` | R3 各 profile × FEC 之 conditional ASR、oracle／predicted CRC 與 frame correct（含計數與分母）、BER、CRC false accept、latency、usable 判定 | 4 |
| `crc_fec_min_usable_summary.csv` | R3 各攻擊 × FEC × 準則之 minimum usable setting；`role` 欄區分 primary 準則與 system-behavior analysis | 4.4 |
| `attack_defense_summary.csv` | R4 12 profiles × FEC × 6 defenses 之 ASR before／after、recovery、oracle／predicted CRC 與 frame correct、latency，並併入 R5 之 clean degradation 與 defense-induced misclassification 對接結果 | 5 |
| `clean_defense_summary.csv` | R5 各防禦 × FEC 之 clean accuracy、clean degradation、oracle／predicted CRC 與 frame correct、BER、defense latency | 6 |
| `defense_tradeoff_summary.csv` | 代表性 profiles 之 recovery、clean degradation、payload recoverability 與 latency 彙整 | 8 |
| `adaptive_k_selection_summary.csv` | R4（attacked）與 R5（clean）之 Adaptive-K route、selected K 分布、blind SNR estimate | 7 |
| `modulation_summary.csv` | 逐調變結果；`table` 欄區分 clean_defense（R5）、attack_defense（R4）、crc_fec_reduced_iteration（R3） | 4.5、6 |
| `latency_summary.csv` | Attack-generation latency（R1、R3）、defense latency（R4、R5）、composite latency（R4，derived） | 8 |
| `experiment_configuration.csv` | 平台、軟體版本、threads、commits、checkpoint／dataset SHA-256、樣本設計、準則、防禦參數、seed 規則 | 2 |
| `source_manifest.csv` | 所有來源檔之路徑、位元組數與 SHA-256 | 全部 |

## 5. `figures/` 與 `figure_sources/`

| 圖 | 檔案 | 資料 |
|---|---|---|
| Figure 1 | `fig1_reduced_iteration_asr_latency.png`：PGD／CW／EAD baseline 與 reduced setting 之 conditional ASR 與 attack-generation latency | `fig1_reduced_iteration_asr_latency.csv` |
| Figure 2 | `fig2_defense_recovery_clean_degradation.png`：各防禦之 attack recovery 與 clean degradation | `fig2_defense_recovery_clean_degradation.csv` |
| Figure 3 | `fig3_oracle_frame_recoverability.png`：防禦 × FEC 之 oracle frame correct | `fig3_oracle_frame_recoverability.csv` |
| Figure 4 | `fig4_adaptive_k_selection_snr.png`：Adaptive-K selected K 與 blind SNR estimate 分布 | `fig4_adaptive_k_selection_snr.csv` |
| Figure 5 | `fig5_defense_tradeoff.png`：recovery、clean degradation 與 defense latency 之取捨 | `fig5_defense_tradeoff.csv` |

`figure_sources/` 之數值由 `data/` 產生（Figure 4 之直方圖計數由 R4、R5 之 `raw_samples.csv` 計算），圖表由 `figure_sources/` 繪製。

## 6. 不包含之內容

模型 checkpoint 與 RadioML dataset 未納入，其 SHA-256 記錄於 `data/experiment_configuration.csv`。逐列 raw 結果保留於原始 run 目錄。
