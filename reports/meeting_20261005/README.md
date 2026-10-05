# meeting_20261005 資料包說明

## 1. 研究進度報告

正式研究進度報告為 [`MEETING_REPORT_20261005_ZH_TW.md`](MEETING_REPORT_20261005_ZH_TW.md)。報告內所有數值皆可由 `data/` 內之 CSV 追溯至 repository `results/` 中的原始實驗輸出。

## 2. `data/` 各 CSV 用途

所有 CSV 為 UTF-8（無 BOM），由原始 artifact 直接轉出，未手動輸入數值。共同欄位：`source_result_dir`、`source_file`、`attack`、`profile`、`defense`、`metric_name`、`raw_count`、`denominator`、`value`、`unit`、`notes`。比例類指標之 `raw_count` 與 `denominator` 為計算該比例之原始計數。

| 檔案 | 用途 | 對應報告章節 |
|---|---|---|
| `attack_acceleration_summary.csv` | 各攻擊 profile 之 mkldnn ON／OFF 中位延遲、配對差、配對結果、語意分類計數、處置，以及 DeepFool 兩個 SEMANTIC_DIFFERENT 樣本之索引欄位 | 第 3 節 |
| `attack_integration_validation_summary.csv` | PGD／BIM 與 extended 13 attacks 之整合驗證：option OFF parity、option ON 語意分類計數、control 攻擊結果 | 第 4 節 |
| `defense_recovery_cpu_summary.csv` | CPU 上 13 列防禦（含 exploratory 與 leakage-free AT）之 recovery、95% CI、clean degradation、retention、net gain、defended accuracy | 第 5 節 |
| `adversarial_training_summary.csv` | Leakage-free AT 訓練紀錄（資料切分、訓練攻擊、best epoch、validation 指標、模型雜湊）、transfer 評估結果，以及 exploratory AT 之訓練／評估重疊紀錄 | 第 5、6 節 |
| `hailo_defense_summary.csv` | CPU 與 Hailo-8 配對評估之 recovery、clean degradation、retention、net gain 與 baseline | 第 7 節 |
| `pipeline_validation_summary.csv` | Main pipeline 與相關實驗之驗證結果；Adaptive-K 與 Hailo-8 pipeline 驗證為 interactive validation，無 persistent artifact | 第 8 節 |
| `SOURCE_MANIFEST.csv` | 每一來源檔之 repository 相對路徑、所在位置、用途、primary／supporting 分類、SHA-256、使用該來源之 CSV，以及其於 `review_evidence/` 之副本位置 | 全部 |

## 3. `figures/`、`figure_sources/` 與 `EVIDENCE_INDEX.md`

| 項目 | 用途 |
|---|---|
| `figures/` | 報告插圖（PNG）：圖 1 攻擊生成延遲 mkldnn ON/OFF 比較（第 3 節）、圖 2 CPU 防禦 recovery 與 clean degradation 取捨（第 5 節）、圖 3 CPU 與 Hailo-8 net gain 比較（第 7 節） |
| `figure_sources/` | 每張圖所使用之數值；每列保留來源 `source_result_dir` 與 `source_csv`。圖 1 之降低比例由原始 ON／OFF 中位延遲計算，計算公式與結果保存於 `reduction_pct`、`reduction_formula` 欄位；圖 3 另保存 CPU 值與 `data/defense_recovery_cpu_summary.csv` 之一致性檢查欄位 |
| `EVIDENCE_INDEX.md` | 依報告章節列出各項 claim／metric 之 source artifact、evidence type、是否有 persistent artifact 與備註，供查核使用 |

圖表均由 `data/` 內 CSV 產生：先由 `data/` CSV 產生 `figure_sources/` CSV，再由 `figure_sources/` CSV 繪圖，數值未經人工重新輸入。

## 4. `review_evidence/` 用途

`review_evidence/` 為查核用之精簡副本，內容為原始 run 目錄中之 summary、validation、manifest 與小型 CSV，未修改內容。資料夾依研究主題分組：

| 資料夾 | 內容 |
|---|---|
| `01_attack_accel/` | Phase 1D-3（PGD／BIM）、15-attack full benchmark、targeted rerun 之 summary 與 per-profile 結果；DeepFool pre-validation 逐樣本分類 |
| `02_attack_integration/` | PGD／BIM 與 extended 13 attacks 之整合驗證 |
| `03_defense_recovery_cpu/` | CPU 防禦比較之 summary、clean degradation、AT 重疊分析、manifest |
| `04_adversarial_training/` | Leakage-free AT 訓練 manifest、模型設定紀錄、training log，以及 AT-only transfer 比較結果 |
| `05_hailo_paired/` | CPU 與 Hailo-8 配對評估之 validation、manifest 與 defense summary |

為避免超過 Windows 路徑長度限制，per-profile 檔案之子目錄已攤平（省略 `profiles/` 一層，`mkldnn/` 併入檔名前綴 `mkldnn_`）；原始路徑記錄於 `SOURCE_MANIFEST.csv` 之 `source_path`。

## 5. 未包含於本資料包之內容

原始大型 results 不在本資料包內，包括已儲存之對抗輸入陣列（`.npy`）、clean input 與 split 索引（`.npz`）、模型 checkpoint（`.pkl`、`.pt`）、完整 adversarial store、逐列 eval parts 與大型 raw log。模型 checkpoint 僅以 SHA-256 記錄於 `SOURCE_MANIFEST.csv` 與 `data/adversarial_training_summary.csv`。

## 6. 與正式 results provenance 之關係

本資料包為研究進度報告之可追溯附件，不取代 repository 中正式 `results/` 目錄之 provenance。查核任何數值時，應以 `SOURCE_MANIFEST.csv` 所列原始檔案及其 SHA-256 為準。
