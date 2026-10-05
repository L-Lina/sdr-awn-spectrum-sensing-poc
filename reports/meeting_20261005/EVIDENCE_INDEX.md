# Evidence Index：meeting_20261005

本文件供查核 `MEETING_REPORT_20261005_ZH_TW.md` 各項研究結論之資料來源。路徑均相對於本資料包根目錄；原始 run 目錄名稱與 repository `results/` 相同，原始路徑與 SHA-256 列於 `data/SOURCE_MANIFEST.csv`。

類型說明：measured＝直接量測之統計彙總；validation＝正確性或 gate 判定；statistics＝配對或推導統計；provenance＝執行紀錄（命令、設定、雜湊）。

## 1. Attack execution acceleration（報告第 3 節）

| Claim／metric | Source artifact | Evidence type | Persistent artifact | 備註 |
|---|---|---|---|---|
| PGD 42.53 → 36.33 ms、BIM 43.84 → 37.63 ms；配對差 −6.20／−6.19 ms；8/8；p = 0.0078 | `review_evidence/01_attack_accel/accel_phase1d3_mkldnn_b1_20260927_135420/validation.json`（`decisions`） | measured／statistics | 有 | Phase 1D-3；`paired_comparisons.csv`、`summary.csv` 為支持檔 |
| Extended attacks 與 FAB eps0.03、FGSM、DeepFool 之 ON／OFF 中位、配對差、語意分類 | `review_evidence/01_attack_accel/attack_b1_accel_full_20261004_100032/b1_acceleration_summary.csv` | measured／validation | 有 | 15-attack full benchmark；`attack_conclusions.csv`、`manifest.json` 為支持檔 |
| FAB eps0.005 321.00 → 304.17 ms、FAB eps0.01 316.45 → 300.91 ms、AutoAttack 64.90 → 57.94 ms | `review_evidence/01_attack_accel/attack_b1_accel_full_20261004_174620/b1_acceleration_summary.csv` | measured／validation | 有 | Targeted rerun；per-profile `profile_result.json`、`mkldnn_summary.csv`（THERMAL_CLEAN 紀錄）為支持檔；AutoAttack fixed-clock 設定見同目錄 `manifest.json` |
| DeepFool 2 個 SEMANTIC_DIFFERENT（`sample_pos` 25、37；`sample_index` 1）、logit margin、最大絕對差 | `review_evidence/01_attack_accel/attack_b1_accel_full_20261004_174620/deepfool__package_default/mkldnn_per_sample_validation.csv` | validation | 有 | pre-validation 逐樣本分類 |
| 彙整表 | `data/attack_acceleration_summary.csv` | measured（衍生彙整） | 有 | 由上列 artifact 產生 |
| 圖 1 | `figures/fig1_attack_acceleration.png`；資料 `figure_sources/fig1_attack_acceleration.csv` | measured（衍生） | 有 | 降低比例由 ON／OFF 中位延遲計算，計算欄位保存於 figure source |

## 2. Extended attack integration（報告第 4 節）

| Claim／metric | Source artifact | Evidence type | Persistent artifact | 備註 |
|---|---|---|---|---|
| Extended 13 attacks：overall PASS、失敗 0、88 樣本、各 profile option ON 分類計數；FGSM／PGD／BIM／DeepFool control 各 88 bit-identical | `review_evidence/02_attack_integration/accel_b1_extended_integration_20261004_223455/validation.json` | validation | 有 | AutoAttack 以 `attack_params["seed"]` 設為每列 formal seed |
| PGD／BIM `b1_pgd_bim_mkldnn_off` 整合驗證 PASS | `review_evidence/02_attack_integration/accel_phase1_final_integration_20260927_153520/validation.json` | validation | 有 | `scope_checks.json` 為支持檔 |
| 彙整表 | `data/attack_integration_validation_summary.csv` | validation（衍生彙整） | 有 | |

## 3. CPU defense comparison（報告第 5 節）

| Claim／metric | Source artifact | Evidence type | Persistent artifact | 備註 |
|---|---|---|---|---|
| 12 列防禦（含固定 K、Adaptive-K、五種濾波器、exploratory AT）之 recovery、CI、clean degradation、retention、net gain、defended accuracy | `review_evidence/03_defense_recovery_cpu/defense_recovery_full_20261004_215947/defense_recovery_summary.csv` | measured | 有 | 分母：clean_correct 31,152、attack success 20,960 |
| Clean degradation（2,200 個唯一 clean 樣本） | 同目錄 `clean_degradation.csv` | measured | 有 | |
| Exploratory AT 之訓練／評估重疊 928／2,200 | 同目錄 `manifest.json`、`at_train_overlap.csv` | provenance／statistics | 有 | 依重疊分組之結果受 SNR 範圍影響，不作為 leakage 效應估計 |
| 已儲存對抗輸入之來源驗證 | `review_evidence/03_defense_recovery_cpu/adaptive_k_effectiveness_20260927_164906/validation.json` | validation | 有 | |
| 彙整表 | `data/defense_recovery_cpu_summary.csv` | measured（衍生彙整） | 有 | |
| 圖 2 | `figures/fig2_cpu_defense_tradeoff.png`；資料 `figure_sources/fig2_cpu_defense_tradeoff.csv` | measured（衍生） | 有 | Exploratory AT 以空心標記區分 |

## 4. Leakage-free adversarial training（報告第 6 節）

| Claim／metric | Source artifact | Evidence type | Persistent artifact | 備註 |
|---|---|---|---|---|
| n_train 92,565、n_val 16,335、排除 sample_index 0–9、訓練攻擊、模型與 warm start SHA-256 | `review_evidence/04_adversarial_training/adversarial_training_20261004_230802/manifest.json` | provenance | 有 | 模型 checkpoint 本身未納入資料包 |
| Best epoch 19、split 紀錄、formal_eval_samples_excluded、functional_check_only | 同目錄 `best_model.config.json` | provenance | 有 | |
| Validation clean 89.50%、robust 68.91%、weighted 0.7921；27 epochs | 同目錄 `training_log.csv` | measured | 有 | |
| Recovery 56.67%（CI 55.99–57.33%）、clean degradation 18.57%、retention 68.62%、net gain +35.90 pp、defended accuracy 46.65% | `review_evidence/04_adversarial_training/defense_recovery_full_20261005_152728/defense_recovery_summary.csv` | measured | 有 | Transfer 評估；分母同第 3 項 |
| AT 模型 clean accuracy 50.73% | 同目錄 `clean_degradation.csv` | measured | 有 | |
| Provenance leakage_free、重疊 0／2,200 | 同目錄 `manifest.json`、`at_train_overlap.csv` | provenance | 有 | |
| 彙整表 | `data/adversarial_training_summary.csv` | 衍生彙整 | 有 | |

## 5. Hailo paired defense（報告第 7 節）

| Claim／metric | Source artifact | Evidence type | Persistent artifact | 備註 |
|---|---|---|---|---|
| Hailo-8 clean accuracy 54.64%、Conditional Attack Success Rate 51.66% | `review_evidence/05_hailo_paired/adaptive_k_hailo_paired_20260928/defense_summary_paired.csv`（scope = overall、defense = none） | measured | 有 | Surrogate transfer |
| 固定 K 與 Adaptive-K 之 CPU／Hailo-8 recovery、clean degradation、net gain | 同檔（scope = overall） | measured | 有 | |
| 配對驗證 gate 全部 PASS | 同目錄 `validation.json` | validation | 有 | `manifest.json` 含 HEF 與來源資訊 |
| 彙整表 | `data/hailo_defense_summary.csv` | measured（衍生彙整） | 有 | CPU 值與 `data/defense_recovery_cpu_summary.csv` 一致 |
| 圖 3 | `figures/fig3_cpu_hailo_defense_comparison.png`；資料 `figure_sources/fig3_cpu_hailo_defense_comparison.csv` | measured（衍生） | 有 | figure source 含 CPU 值一致性檢查欄位 |

## 6. Pipeline integration validation（報告第 8 節）

| Claim／metric | Source artifact | Evidence type | Persistent artifact | 備註 |
|---|---|---|---|---|
| Adaptive-K pipeline：55 PASS／0 FAIL | 無（`experiments/validate_adaptive_k_pipeline_wiring.py` 於 Raspberry Pi 互動執行） | validation | **無**（interactive validation，僅使用暫存目錄） | 數值紀錄於 repository `docs/research/RESEARCH_CLOSEOUT_STATUS_ZH_TW.md` 第十節；未產生 validation.json |
| Hailo-8 pipeline：34 PASS／0 FAIL | 無（`experiments/validate_hailo_main_pipeline_wiring.py` 於 Raspberry Pi 互動執行） | validation | **無**（interactive validation，僅使用暫存目錄） | 同上 |
| Attack B=1 mkldnn 選項整合驗證 | 見第 2 項之兩個 `validation.json` | validation | 有 | |
| 彙整表 | `data/pipeline_validation_summary.csv` | 衍生彙整 | 有 | interactive 項目之 `source_result_dir` 標示為 none |
