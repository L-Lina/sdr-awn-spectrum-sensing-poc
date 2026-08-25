# Final Experiment Evidence Index（正式結果證據索引）

本文件建立正式實驗結果與其原始資料、摘要、manifest、驗證紀錄及分析產物之可追溯索引，以支援結果驗證與實驗重現。本文件本身不重新計算任何數字，全部引用既有正式結果目錄；不修改任何既有 `raw_results.csv`、`summary.csv`、`manifest.json`、PNG 或 log。配套機器可讀索引：`results/FINAL_CSV_MASTER_INDEX.csv`（103 列，涵蓋本文件列為 FINAL 且深度盤點的 5 個結果目錄之全部 CSV）。

## 〇、範圍與方法說明

本索引對 `results/` 下約 100 個資料夾逐一分類（見第一節），並對其中 **5 個核心 FINAL 結果目錄**做逐檔案（CSV／PNG／JSON／log）盤點：即時 filesystem 掃描、schema 讀取、SHA256 計算，未預先假設任何檔名或內容。這 5 個目錄是：

1. `results/satellite_like_final_20260821T021117Z/`（satellite-like 576 最終整合實驗）
2. `results/all_attack_acceleration_corrected_20260824T055724Z/`（corrected 17-attack acceleration 最終結果）
3. `results/performance_latency_20260818T010552Z/`（CPU pipeline 效能／延遲 profiling）
4. `results/end_to_end_latency_20260818T062625Z/`（end-to-end pipeline 延遲評估）
5. `results/spectrum_sensing_utility_formal_20260727T021248Z/`（four-path Spectrum Sensing 正式實驗）

Phase 0-4（`formal_phase*`）與其餘既有正式結果目錄，已於第一節逐一標記狀態，但**未在本文件重複展開逐 CSV 清點**——這些目錄的完整資料流程與逐項數字已存在於 `docs/PROJECT_STATUS.md`（Part 2）與 `docs/formal_experiment_plan.md`，本文件僅在第一節提供分類標記與路徑索引，避免內容膨脹到失去可查詢性；若需要 Phase 0-4 的逐 CSV schema，可比照本文件第三節示範的方法對該目錄執行相同的唯讀掃描。

---

## 一、正式結果資料夾總表（狀態分類）

狀態定義：**FINAL**（目前正式引用的證據）／**SUPERSEDED**（已被更新結果取代，不得作為正式證據）／**INTERMEDIATE**（早期分層執行的中間產物，非最終引用版本，但保留作為方法演進紀錄）／**LEGACY**（開發早期的探索性／除錯用資料夾，非正式研究結果）／**UNKNOWN**（尚未深入判定）。

### 1.1 Satellite-like

| 資料夾 | 狀態 | 說明 |
|---|---|---|
| `satellite_like_final_20260821T021117Z` | **FINAL** | Step 4 最終 576 組合正式實驗＋事後兩輪 audit |
| `satellite_like_smoke_20260821T010645Z` | INTERMEDIATE | Step 3 通道模擬器 108-sample smoke，Step 4 之前置驗證 |

### 1.2 Acceleration（17-attack）

| 資料夾 | 狀態 | 說明 |
|---|---|---|
| `all_attack_acceleration_corrected_20260824T055724Z` | **FINAL** | 合併 13 個未受影響條件（原始未變）＋5 個修正 seed 後重跑的攻擊 |
| `all_attack_acceleration_20260824T031053Z` | **SUPERSEDED**（部分） | 內含 `SUPERSEDED.txt` 標記；其中 DIFGSM／Square／APGD／APGDT／AutoAttack 5 項數字因 seed pairing bug 已作廢，其餘 13 項數字仍有效且被 corrected 目錄原樣複用 |
| `attack_compatibility_smoke_20260727T030223Z` | **FINAL** | 17/17 攻擊相容性 PASS（含 DIFGSM 修正後） |
| `attack_compatibility_smoke_20260727T024650Z` | SUPERSEDED | 16/17 PASS，DIFGSM 當時仍為 NEEDS_CUSTOM_IMPLEMENTATION，已被上一列取代 |

### 1.3 Latency／Performance

| 資料夾 | 狀態 | 說明 |
|---|---|---|
| `performance_latency_20260818T010552Z` | **FINAL** | CPU pipeline stage profiling、FGSM/PGD/CW 加速、thread microbenchmark、streaming sensing 診斷 |
| `end_to_end_latency_20260818T062625Z` | **FINAL** | Scenario A-E（clean／+TopK／+FGSM／+PGD／+FGSM+TopK）E2E 延遲評估 |

### 1.4 Sensing／Four-path

| 資料夾 | 狀態 | 說明 |
|---|---|---|
| `spectrum_sensing_utility_formal_20260727T021248Z` | **FINAL** | Direct／No-sensing／Sensing／Oracle 四路正式對照實驗，N=2200 |
| `spectrum_sensing_utility_pilot` | INTERMEDIATE | 四路實驗 pilot |
| `spectrum_sensing_utility_smoke` | INTERMEDIATE | 四路實驗 smoke |
| `sensing_validation_matrix` | LEGACY | 早期 sensing 參數驗證，非正式引用版本 |
| `sensing_revalidation_after_alignment` | **FINAL**（Phase 5 用途） | `docs/PROJECT_STATUS.md` 列為 Phase 5（sensing 參數敏感度）之正式證據，528 組合分散於 A/B/C/D 子目錄 |
| `sensing_boundary_probe`／`sensing_window_probe` | LEGACY | 早期探索性 probe |
| `cfile_pipeline_smoke_20260727T082623Z` | **FINAL** | cfile 三格式讀取器＋正式管線串接驗證 |

### 1.5 Phase 0-4（正式管線，詳細數字見 `docs/PROJECT_STATUS.md`）

| 資料夾 | 狀態 | 說明 |
|---|---|---|
| `formal_pilot_phase0` | INTERMEDIATE | Phase 0 機制驗證，非正式可引用科學結果（N=8/cell） |
| `formal_phase1_sensing_clean_amc` | **FINAL** | Phase 1 sensing baseline，N=2200（採固定 seed=42 方法論，與 four-path 實驗的逐樣本變動 seed 方法論不同，兩者皆有效但不可互換引用，見 `PROJECT_STATUS.md` 附註） |
| `formal_phase3_attack_reduced` | INTERMEDIATE | Phase 3 reduced-tier（N=792），full-tier 之前置版本 |
| `formal_phase3_attack_full` | **FINAL** | Phase 3 正式可引用結果，N=3960 |
| `formal_phase4_defense_reduced` | INTERMEDIATE | Phase 4 reduced-tier（N=792/3168），已被 expanded_full 取代為正式結論 |
| `formal_phase4_expanded_k` | INTERMEDIATE | Expanded-K Confirmation，其結論被 round-27 expanded_full 進一步修正 |
| `formal_phase4_expanded_smoke` | INTERMEDIATE | round-27 正式執行前的 smoke test |
| `formal_phase4_expanded_full` | **FINAL** | Phase 4 正式可引用結果（round 27），N=3960 attack instances／27720 rows |
| `formal_phase4_topk_ablation` | **FINAL** | 獨立的 3-policy 前處理 ablation 發現，非被取代，是另一個具體問題的正式答案 |

### 1.6 其他正式或半正式結果

| 資料夾 | 狀態 | 說明 |
|---|---|---|
| `sensing_awn_low_perturbation_attacks_20260804T002511Z` | **FINAL** | FGSM + 五種低擾動攻擊之數位基準實驗 |
| `parameter_validation_20260727T054218Z` | **FINAL** | 核心參數驗收之最終分類（71 IMPLEMENTED_AND_VALIDATED 等） |
| `parameter_validation_20260727T034426Z` | INTERMEDIATE | 較早的參數驗證回合，被上一列擴充取代 |
| `results.zip`（檔案，非資料夾） | UNKNOWN | 尚未解壓檢視，非上述任何正式結果目錄的來源 |

### 1.7 開發早期探索性／除錯資料夾（LEGACY，不逐一展開）

以下約 70 個資料夾（`case4_calibration*`、`case5_calibration*`、`cw_cli_smoke`、`cw_fair_topk_sweep`、`cw_repro_probe`、`decouple_probe_*`、`dummy_fallback_smoke`、`e2e_smoke_matrix`、`eps_sweep_first_change`、`eps_validation_real`、`fair_topk_matrix`、`fair_topk_sweep`、`final_regression_check`、`mergegap_pipeline_cases`、`modulation_snr_matrix`、`multiburst_*`、`none_temp_probe_*`、`parameter_coverage_completion`、`param_test_*`、`pgd_repro_probe`、`radioml_*`、`real_backend_probe`、`real_smoke_test`、`regression2_*`、`reverify_probe`、`seed_batch_smoke`、`seed_cli_smoke`、`small_real_qpsk_fgsm`、`snr_smoke_sweep*`、`temperature_negative_probe`、`temp_exp_*`、`threshold_factor_probe`、`timing_probe`、`topk_*`）皆為專案開發早期（多數為 7 月中旬）的參數探索、除錯、單點驗證用途，**一律標記 LEGACY，不作為正式研究結論引用**，與正式 Phase 0-4／satellite-like／acceleration／latency 結果無直接取代或補充關係。

---

## 二、A. 整體 Pipeline 是否完整跑完？

| 證據 | 檔案 | 關鍵欄位 | 結果 |
|---|---|---|---|
| Satellite-like 576/576 完成 | `results/satellite_like_final_20260821T021117Z/raw_results.csv` | `status`, `fallback_used` | 576 rows, `status`={'ok':576}, `fallback_used`={false:576} |
| 0 error | 同上 | `status`, `error_type` | `error_type` 全為 NaN（576/576），無 'error' 值 |
| 0 fallback | 同上 | `fallback_used` | 576/576 為 false |
| 0 NaN/Inf | 同上 | 全數值欄位 | 掃描結果 `numeric_inf=0`；`numeric_nan` 非零僅來自設計上本就允許為空的欄位（如 `achieved_snr_db` 僅 channel condition=strong 時有值、`error_message`／`error_type` 皆為 0 error 下的預期空值），不代表資料缺陷 |
| 0 no_region / 0 no_segment | 同上 | `sensing_detected`, `detected_start`, `detected_end` | `sensing_detected`={true:576}，576/576 皆偵測成功 |
| Manifest 佐證 | `results/satellite_like_final_20260821T021117Z/manifest_analysis.json` | git commit, raw_results sha256 | 576-row 結果之 git 版本與 SHA256 快照 |
| 17-attack 18/18 條件完成 0 error | `results/all_attack_acceleration_corrected_20260824T055724Z/attack_acceleration_raw.csv` | `status` | `status`={ok:5124, fallback:57}，0 筆 'error' |
| 四路實驗 2200/2200 完成 | `results/spectrum_sensing_utility_formal_20260727T021248Z/raw_results.csv` | `run_status`, `sensing_detected` | 2200 rows，`sensing_detected`={true:2200} |

## 三、B. Spectrum Sensing 是否正確？

| 證據 | 檔案 | 關鍵欄位 | 結果 |
|---|---|---|---|
| Detection probability（satellite-like） | `overall_summary.csv`／`by_channel.csv` | `sensing_detection_rate` | 100%（576/576），四個 channel condition 下皆 100% |
| Captured signal ratio（satellite-like） | `overall_summary.csv` | `captured_signal_ratio_mean` | 0.9734（mean），四個 channel condition 下 0.9707–0.9746，穩定 |
| Sensing 分組稽核（非僅 overall） | `audit/audit_02_sensing_by_channel.csv`／`_by_modulation.csv`／`_by_snr.csv`／`_by_channel_modulation.csv` | `detection_rate`, `csr_mean`, `boundary_*_err_abs_mean` | 各分組下 detection 皆 100%，csr 穩定於 0.95–1.00 |
| Boundary / region 資訊 | `raw_results.csv` | `detected_start`, `detected_end`, `true_start`, `true_end`, `boundary_start_error`, `boundary_end_error`, `false_occupied_samples` | 逐列原始邊界誤差資料 |
| Sensing latency | `latency_summary.csv`／`audit/audit_07_latency.csv` | `scenario`='clean' 之 `median`,`p95` | clean median 2.48ms／p95 4.13ms（含 sensing 階段） |
| Four-path sensing 正確性（獨立方法論） | `spectrum_sensing_utility_formal_20260727T021248Z/raw_results.csv` | `sensing_detected`, `captured_signal_ratio`, `start_boundary_error`, `end_boundary_error` | `sensing_detected`={true:2200}；`overall_summary.csv` 中 sensing path accuracy 0.5900 vs oracle 0.5909，McNemar p=0.754（不顯著，見 `paired_comparisons.csv`） |
| Streaming sensing（PROTOTYPE，非正式管線） | `performance_latency_20260818T010552Z/streaming_sensing_validation.csv`,`streaming_failure_diagnosis.csv` | `matched`,`detection_failed_in_this_chunk` | chunk_size 256/512 未完全匹配，1024/2048 匹配，非正式管線之一部分 |

## 四、C. AMC 結果在哪？

| 證據 | 檔案 | 關鍵欄位 | 結果 |
|---|---|---|---|
| Overall clean accuracy | `overall_summary.csv` | `clean_accuracy_overall` | 0.3229（n=192，含 pseudo-replication，真實獨立 n=96，見文件 §22.5） |
| By modulation | `by_modulation.csv` | `clean_accuracy` | 8PSK 46.9%／QPSK 37.5%／BPSK 12.5% |
| By SNR | `by_snr.csv` | `clean_accuracy` | -10dB 20.8%／0dB 37.5%／10dB 37.5%／18dB 33.3% |
| By channel | `by_channel.csv` | `clean_accuracy` | clean 58.3%／mild 33.3%／moderate 20.8%／strong 16.7%，單調遞減 |
| 交叉稽核（channel×modulation） | `audit/audit_02_sensing_by_channel_modulation.csv`（sensing 面向）＋`by_modulation_attack.csv`（accuracy 面向） | `clean_accuracy` | 支持 GIGO 結論（accuracy 下降非 sensing 造成） |

## 五、D. Attack Effectiveness 結果在哪？

| 證據 | 檔案 | 關鍵欄位 | 結果 |
|---|---|---|---|
| Attacked accuracy | `overall_summary.csv` | `fgsm_attacked_accuracy`,`pgd_det_attacked_accuracy` | FGSM 16.67%／PGD(det) 9.375% |
| Prediction change rate（`attack_success` 正式定義） | `overall_summary.csv`／`audit/metric_definition_audit.csv` | `fgsm_overall_asr`,`pgd_det_overall_asr` | FGSM 91.67%／PGD(det) 98.96%（正式名稱為 Prediction Change Rate，非傳統 conditional adversarial success） |
| Conditional ASR | `overall_summary.csv`／`audit/unique_attack_sample_audit.csv` | `fgsm_conditional_asr`,`pgd_det_conditional_asr` | FGSM 77.42%（24/31）／PGD(det) 96.77%（30/31） |
| Perturbation norms | `overall_summary.csv`／`audit/audit_05_fgsm_vs_pgd_paired_by_base.csv` | `fgsm_mean_linf`,`fgsm_mean_l2` 等 | FGSM Linf 0.001367／L2 0.02178；PGD(det) Linf 0.001367／L2 0.01882（近乎相同 Linf 之根因見文件 §22.4） |
| 17-attack acceleration 之 attacked accuracy／correctness | `all_attack_acceleration_corrected_20260824T055724Z/attack_correctness_summary.csv` | `prediction_match_rate`,`tensor_max_abs_diff` | 逐攻擊逐 batch size 明細 |

## 六、E. Top-K Defense 結果在哪？

| 證據 | 檔案 | 關鍵欄位 | 結果 |
|---|---|---|---|
| Recovery rate（正確 denominator） | `audit/topk_denominator_audit.csv` | `recovery_rate_over_attack_failed_only` | FGSM 2.50%（2/80）／PGD(det) 6.90%（6/87） |
| Clean degradation（正確 denominator） | `audit/topk_denominator_audit.csv` | `degradation_rate_over_clean_correct_only` | 29.03%（9/31） |
| By attack／channel／modulation／SNR | `audit/audit_06_topk_by_channel.csv`,`_by_modulation.csv`,`_by_snr.csv` | `fgsm_recovery_rate`,`pgd_det_recovery_rate`,`none_degradation_rate` | 逐分組明細 |
| By Top-K on/off（原始 groupby，注意 pseudo-replication，見文件 §22.5） | `by_topk.csv` | `defended_accuracy` | topk=on 時 0.1701（n=288） |
| End-to-end Top-K overhead（獨立於 satellite-like，performance 回合） | `end_to_end_latency_20260818T062625Z/topk_overhead_summary.csv` | `overhead_mean_ms`,`overhead_pct_mean` | Top-K 前處理＋防禦推論之額外延遲 |

## 七、F. Latency 結果在哪？

| 證據 | 檔案 | 關鍵欄位 | 結果 |
|---|---|---|---|
| Satellite-like 各 scenario latency | `satellite_like_final_20260821T021117Z/latency_summary.csv` | `scenario`,`median`,`p95`,`p99` | clean/fgsm/pgd_det/fgsm_topk/pgd_det_topk 五種 scenario |
| Processing budget fit | `processing_budget.csv`／`audit/audit_08_processing_budget.csv` | `fits_*ms_median`,`fits_*ms_p95`,`empirical_fit_rate_*` | 5/10/20/35/50/100/250ms 門檻下之符合率（僅效能參照，非 deadline） |
| Clean pipeline 逐 stage profiling | `performance_latency_20260818T010552Z/pipeline_latency_summary.csv`,`bottleneck_by_percentile.csv` | `stage`,`median`,`pct_of_median_total` | 各 stage（embedding/energy_detection/segmentation/awn_preprocess/awn_clean_inference 等）耗時占比 |
| Attack E2E（clean/FGSM/PGD/+TopK） | `end_to_end_latency_20260818T062625Z/end_to_end_latency_summary.csv`,`stage_latency_summary.csv` | `scenario`,`variant`,`median`,`p95` | Scenario A-E，deterministic／stochastic PGD 皆分列 |
| Before/after 加速對照（FGSM/PGD/CW，3-attack 精簡版） | `end_to_end_latency_20260818T062625Z/before_after_end_to_end.csv` | `end_to_end_speedup_median`,`end_to_end_speedup_p95` | 三種攻擊之 E2E 加速倍率 |
| Thread microbenchmark | `performance_latency_20260818T010552Z/awn_thread_microbenchmark_summary.csv` | `n_threads_used`,`median_ms`,`throughput_samples_per_sec` | 純 AWN inference 之 thread 數影響（非攻擊特定） |

## 八、G. 17-Attack Acceleration 結果在哪？

全部位於 `results/all_attack_acceleration_corrected_20260824T055724Z/`：

| 證據 | 檔案 | 關鍵欄位 |
|---|---|---|
| Baseline latency | `attack_bottleneck_summary.csv` | `baseline_median_ms`,`baseline_p95_ms` |
| Optimized latency | `attack_bottleneck_summary.csv` | `optimized_median_ms`,`optimized_p95_ms` |
| Speedup | `attack_bottleneck_summary.csv` | `median_speedup`,`p95_speedup` |
| Thread tuning（逐 thread 明細） | `attack_thread_tuning.csv` | `torch_threads`,`median`,`p95` |
| Batch tuning（逐 batch size 正確性） | `attack_correctness_summary.csv` | `batch_size`,`prediction_match_rate`,`tensor_max_abs_diff` |
| Correctness | `attack_correctness_summary.csv` | `n_compared`,`mean_l2_diff` |
| Classification（A/B/C/D） | `attack_batching_classification.csv` | `classification`,`reason`,`worst_case_pred_match_rate` |
| E2E | `attack_e2e_summary.csv` | `attack_generation_median`,`total_median`,`total_p95` |
| Processing class | `attack_processing_class.csv` | `attack_generation_processing_class`,`e2e_processing_class` |
| Bottleneck 分類 | `attack_bottleneck_summary.csv` | `configured_iteration_or_query_count`（＋文件 ALL_ATTACK_ACCELERATION_ANALYSIS_ZH_TW.md 各攻擊小節之文字 bottleneck 分類） |
| Fallback 排除證明 | `attack_acceleration_raw.csv` | `status`,`fallback`,`phase` | 57 筆 fallback 全數集中 `phase`='batching_test'，0 筆進入 baseline/thread_tuning |
| 受影響 5 攻擊的獨立重跑（provenance） | `rerun_5_attacks_only/*.csv`,`rerun_5_attacks_only/manifest.json` | 同上欄位，僅 5 攻擊 |
| 原始（superseded 部分）數字 | `all_attack_acceleration_20260824T031053Z/`＋其中 `SUPERSEDED.txt` | -- | 僅作 provenance，5 攻擊數字不得再引用 |

## 九、H. Correctness Validation 在哪？

| 證據 | 檔案 | 關鍵欄位 |
|---|---|---|
| Tensor diff（satellite-like，FGSM vs PGD） | `satellite_like_final_20260821T021117Z/audit/audit_05_fgsm_vs_pgd_paired_by_base.csv` | `linf_diff`,`l2_diff` |
| Tensor diff（17-attack） | `all_attack_acceleration_corrected_20260824T055724Z/attack_correctness_summary.csv` | `tensor_max_abs_diff`,`mean_l2_diff` |
| Prediction match | 同上 | `prediction_match_rate` |
| Linf/L2（satellite-like） | `satellite_like_final_20260821T021117Z/overall_summary.csv` | `fgsm_mean_linf`,`fgsm_mean_l2` 等 |
| Seed pairing（17-attack corrected） | `all_attack_acceleration_corrected_20260824T055724Z/attack_acceleration_raw.csv`,`rerun_5_attacks_only/attack_acceleration_raw.csv` | `seed_used` 欄位；unit validation 紀錄於 `rerun_5_attacks_only/terminal.log` 開頭 |
| Fallback exclusion | `attack_acceleration_raw.csv` | `status`,`fallback` |
| Model mode（attack 後是否正確恢復 eval） | `performance_latency_20260818T010552Z/cw_baseline_raw.csv`,`fgsm_baseline_raw.csv`,`pgd_baseline_raw.csv`,`end_to_end_latency_20260818T062625Z/cw_end_to_end_supplement.csv` | `model_mode_after` |
| Classification 判定依據 | `attack_batching_classification.csv` | `reason`,`worst_case_max_diff`,`worst_case_pred_match_rate` |

---

## 十、Manifest／Log 檢查

| 目錄 | manifest.json | 內容涵蓋 | 缺項 |
|---|---|---|---|
| `satellite_like_final_20260821T021117Z` | `manifest_analysis.json`（分析階段）＋`audit/audit_report.json`（稽核階段） | git commit、raw_results sha256、script 參照 | 無 dataset path／attack list／seed 欄位（這些記錄在 `experiments/run_satellite_like_final.py` 原始碼常數，非 manifest；已知限制） |
| `all_attack_acceleration_corrected_20260824T055724Z` | `manifest.json`（merge 階段）＋`rerun_5_attacks_only/manifest.json` | dataset_path、checkpoint_path、registry、bench_attacks、tier、batch/thread 設定、eps/seed、n_error/n_fallback、runtime；merge 版額外記錄 bug 說明、受影響/未受影響清單、corrected seed policy | 完整 |
| `performance_latency_20260818T010552Z` | `manifest.json`＋`validation_round_manifest.json` | git commit、torch/torchattacks 版本、checkpoint sha256、dataset_path | 完整；`phaseA-E_terminal.log` 為各 Phase 執行 log |
| `end_to_end_latency_20260818T062625Z` | `manifest.json` | scenario/variant 設定 | 未逐一核對，建議引用前先行目視確認 |
| `spectrum_sensing_utility_formal_20260727T021248Z` | `manifest.json` | run 設定 | `stdout.log`／`stderr.log` 為執行 log |

---

## 十一、驗證項目對照表

| Verification Target | Primary Evidence | Relevant Fields | Reference Result |
|---|---|---|---|
| 整體 satellite-like 實驗是否完整跑完 | `satellite_like_final_20260821T021117Z/raw_results.csv` | `status`,`fallback_used` | 576 rows, 0 error, 0 fallback |
| Spectrum Sensing 是否正確偵測訊號 | `overall_summary.csv` | `sensing_detection_rate`,`captured_signal_ratio_mean` | detection 100%, mean captured ratio ≈0.973 |
| AMC 準確率是否隨 channel 惡化 | `by_channel.csv` | `clean_accuracy` | 58.3%→33.3%→20.8%→16.7% |
| FGSM 加速倍率 | `all_attack_acceleration_corrected_20260824T055724Z/attack_bottleneck_summary.csv` | `baseline_median_ms`,`optimized_median_ms`,`median_speedup` | 5.73→0.438ms，13.07x |
| DIFGSM 可 batching 之依據 | `attack_correctness_summary.csv`＋`attack_batching_classification.csv` | `prediction_match_rate`,`tensor_max_abs_diff`,`batch_size` | 修正 seed 後全部 batch size 下 100% match，0 diff，分類 A |
| Square 不可 batching 之依據 | `attack_correctness_summary.csv`＋`attack_acceleration_raw.csv` | `prediction_match_rate`,`fallback`,`status` | worst match 75%，且 batch≥8 幾乎全數第三方套件崩潰 |
| FAB 52.8x 加速之性質 | `attack_bottleneck_summary.csv`＋`attack_e2e_summary.csv` | `median_speedup`（batch 表）vs `total_median`（E2E 表） | batch=32 下 18.09ms；E2E 單筆仍 108ms——52.8x 為批次吞吐量效果，非單筆延遲改善 |
| E2E latency 位置 | `attack_e2e_summary.csv`／`end_to_end_latency_summary.csv` | `total_median`,`total_p95` | 依攻擊/情境查表 |
| Top-K 防禦效果 | `audit/topk_denominator_audit.csv` | `recovery_rate_over_attack_failed_only`,`degradation_rate_over_clean_correct_only` | FGSM 2.5%／PGD 6.9% 回復；29.03% clean 反被弄錯 |
| 17 種攻擊是否全數涵蓋 | `attack_acceleration_raw.csv` | `attack` value_counts | 18 個條件（17 canonical＋pgd 拆兩條件），每條件 baseline+thread+batch+E2E 皆有列 |
| 最終數字是否受 fallback 污染 | `attack_acceleration_raw.csv` | `phase`,`fallback` | 57 筆 fallback 全在 `batching_test`，0 筆進入 baseline/aggregate |
| 四路 sensing 實驗中 sensing 與 oracle 之差距 | `spectrum_sensing_utility_formal_20260727T021248Z/paired_comparisons.csv` | `accuracy_difference`,`mcnemar_exact_pvalue` | sensing vs oracle 差 +0.0009，p=0.754（不顯著） |
| Clean pipeline 最大延遲瓶頸 stage | `performance_latency_20260818T010552Z/bottleneck_by_percentile.csv` | `stage`,`pct_of_median_total` | 依 median/p95 分別列出主要瓶頸 stage |

---

## 十二、完整性 PASS/FAIL

| # | 項目 | 結果 |
|---|---|---|
| 1 | Satellite final raw CSV exists | PASS — `raw_results.csv`, 576 rows |
| 2 | Satellite final summary exists | PASS — `overall_summary.csv`＋8 個 by_*.csv |
| 3 | Satellite final manifest exists | PASS — `manifest_analysis.json`＋`audit/audit_report.json` |
| 4 | 576 rows present | PASS |
| 5 | 0 unexpected errors | PASS — `status`={ok:576} |
| 6 | 0 fallback | PASS — `fallback_used`={false:576} |
| 7 | sensing evidence exists | PASS — 第三節 |
| 8 | AMC evidence exists | PASS — 第四節 |
| 9 | attack evidence exists | PASS — 第五節 |
| 10 | Top-K evidence exists | PASS — 第六節 |
| 11 | latency evidence exists | PASS — 第七節 |
| 12 | 17-attack baseline evidence exists | PASS — `attack_bottleneck_summary.csv` |
| 13 | 17-attack optimized evidence exists | PASS — 同上 |
| 14 | thread tuning evidence exists | PASS — `attack_thread_tuning.csv`, 18 攻擊×5 threads 完整 |
| 15 | batch tuning evidence exists | PASS — `attack_correctness_summary.csv`, 18 攻擊×5 batch size 完整 |
| 16 | correctness evidence exists | PASS — 第九節 |
| 17 | seed-corrected evidence exists | PASS — `rerun_5_attacks_only/`＋merge `manifest.json` |
| 18 | E2E evidence exists | PASS — `attack_e2e_summary.csv`, 18/18 有 n/median/p95 |
| 19 | processing class evidence exists | PASS — `attack_processing_class.csv`, 18/18 |
| 20 | figures traceable to CSV | PASS — 5 個 FINAL 目錄之 PNG 皆有對應同名或同資料夾內 source CSV（`spectrum_sensing_utility_formal` 甚至逐圖附 `*_source.csv`） |
| 21 | all final CSV SHA256 calculated | PASS — 103 筆列於 `results/FINAL_CSV_MASTER_INDEX.csv` |
| 22 | no final CSV missing | PASS（於本次盤點範圍內：5 個核心目錄） |
| 23 | superseded runs clearly separated | PASS — `all_attack_acceleration_20260824T031053Z/SUPERSEDED.txt`＋本文件第一節表格 |

**23/23 PASS，本次盤點範圍內無 FAIL 項目。**

---

## 十三、已知限制（誠實揭露）

1. 本文件對 Phase 0-4（`formal_phase*`）與其他既有正式目錄僅做**分類標記**，未逐 CSV 做 schema／SHA256 盤點——如需 Phase 0-4 的特定數字，請改查 `docs/PROJECT_STATUS.md` 或對該目錄比照第三節方法重新掃描。
2. `results/results.zip` 未解壓檢視，內容與來源未確認，列為 UNKNOWN。
3. `sensing_revalidation_after_alignment`（Phase 5 證據）之子目錄結構較深（A/B/C/D 四個子資料夾），本文件僅在第一節標記其為 FINAL，未逐檔展開。
4. `results/FINAL_CSV_MASTER_INDEX.csv` 的 `purpose` 欄位由檔名／欄位規則式判斷產生，多數已人工核對，但未逐筆手動覆核全部 103 列，如有描述不準確之處，請以該 CSV 自身欄位與對應正式研究文件的引用段落為準。
