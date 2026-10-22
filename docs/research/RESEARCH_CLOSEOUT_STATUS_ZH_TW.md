# 研究收尾狀態與正式結果彙整

## 一、範圍與證據來源

本文件彙整目前已完成的正式結果。所有數字皆直接取自下列結果目錄；未持久化 artifact 的驗證另行標示，不以推測值填補。

| 項目 | 結果目錄／來源 |
|---|---|
| 17-attack B=1 backend toggle 探索性量測（full） | `results/attack_b1_accel_full_20261004_100032/`（`b1_acceleration_summary.csv`、`attack_conclusions.csv`） |
| Targeted rerun（FAB eps0.005／eps0.01、AutoAttack、DeepFool） | `results/attack_b1_accel_full_20261004_174620/` |
| PGD／BIM backend-toggle opt-in 選項驗證 | `results/accel_phase1_final_integration_20260927_153520/validation.json` |
| Extended 13 attacks backend-toggle opt-in 選項驗證 | `results/accel_b1_extended_integration_20261004_223455/validation.json` |
| Adaptive-K CPU effectiveness | `results/adaptive_k_effectiveness_20260927_164906/`（`validation.json` overall = PASS） |
| Adaptive-K Hailo paired | `results/adaptive_k_hailo_paired_20260928/`（`validation.json` overall = PASS） |
| Formal latency | `formal_latency_20260928d`（彙整表 `reports/meeting_20260928/tables/formal_latency_primary.csv`） |
| Defense recovery full comparison（含 exploratory AT） | `results/defense_recovery_full_20261004_215947/` |
| Leakage-free adversarial training | `results/adversarial_training_20261004_230802/` |
| Leakage-free AT-only transfer comparison | `results/defense_recovery_full_20261005_152728/` |
| Adaptive-K／Hailo main-pipeline wiring validation | 互動執行（terminal 輸出），無持久化 artifact，見第十節 |

## 二、17-attack B=1 backend toggle 探索性 latency 量測（歷史紀錄）

**定位**：本節為 Raspberry Pi 5（ARM64／AArch64）上 PyTorch backend toggle（僅於攻擊呼叫期間關閉 oneDNN／mkldnn）之探索性 latency 量測，數值保留原始量測結果，不構成攻擊加速結論，亦不推論關閉 mkldnn 對 ARM 平台具一般性效果或可跨平台泛化。「處置」欄為當時將 profile 列入 opt-in 選項之分類（歷史紀錄），正式 pipeline 未使用此選項。攻擊計算量之正式分析以降低迭代次數與搜尋預算（reduced-iteration）為主線，見 `reports/reduced_iteration_crc_fec_defense_study_20261022/`。

條件：Raspberry Pi 5、torch 2.14.0+cpu、intra 4／interop 1、正式 88-sample 子集、正式 profile 參數（`configs/hailo_full_matrix.json`）、THERMAL_CLEAN gate、每 profile 8 組 ON/OFF 配對（Phase 1D-3 方法）。延遲為 B=1 完整 attack block（prep + construct + atk + post）之 pooled median；speedup 為配對 median（baseline／candidate）。

| Attack | 當時之處置（歷史紀錄） | mkldnn ON → OFF median (ms) | 配對 speedup | OFF 較快配對 | 語意結果（timed 呼叫） |
|---|---|---|---|---|---|
| PGD | 已整合（`b1_pgd_bim_mkldnn_off`） | 42.53 → 36.33（Phase 1D-3） | — | 8/8 | 88/88 bit-identical |
| BIM | 已整合（`b1_pgd_bim_mkldnn_off`） | 43.84 → 37.63（Phase 1D-3） | — | 8/8 | 87 bit-identical + 1 semantic same |
| MI-FGSM | 整合（extended） | 42.14 → 36.42 | 1.156 | 8/8 | 全部 bit-identical |
| DI-FGSM | 整合（extended） | 44.42 → 38.61 | 1.178 | 8/8 | 全部 bit-identical |
| VMI-FGSM | 整合（extended） | 245.63 → 212.41 | 1.157 | 8/8 | 全部 bit-identical |
| VNI-FGSM | 整合（extended） | 247.95 → 211.65 | 1.161 | 8/8 | 4152 bit + 72 semantic same |
| RFGSM | 整合（extended） | 42.70 → 36.05 | 1.185 | 8/8 | 全部 bit-identical |
| TPGD | 整合（extended） | 46.37 → 39.35 | 1.171 | 8/8 | 4176 bit + 48 semantic same |
| CW | 整合（extended） | 43.97 → 39.10 | 1.124 | 8/8 | 2400 bit + 1824 semantic same |
| FAB eps0.005 | 整合（extended；targeted rerun） | 321.00 → 304.17 | 1.055 | 8/8 | 3720 bit + 504 semantic same |
| FAB eps0.01 | 整合（extended；targeted rerun） | 316.45 → 300.91 | 1.053 | 8/8 | 3408 bit + 816 semantic same |
| FAB eps0.03 | 整合（extended） | 330.63 → 312.47 | 1.059 | 8/8 | 2400 bit + 1824 semantic same |
| Square | 整合（extended） | 43.80 → 33.97 | 1.277 | 8/8 | 全部 bit-identical |
| APGD | 整合（extended） | 58.59 → 51.56 | 1.149 | 8/8 | 全部 bit-identical |
| APGD-T | 整合（extended） | 66.48 → 57.58 | 1.156 | 8/8 | 全部 bit-identical |
| AutoAttack | 整合（extended；targeted rerun） | 64.90 → 57.94（p95 15932.7 → 12301.1） | 1.132 | 8/8 | 1408/1408 bit-identical |
| EAD | 整合（extended） | 1422.14 → 1223.23 | 1.171 | 8/8 | 776 bit + 632 semantic same |
| FGSM | 不整合（NO_SAFE_SPEEDUP_FOUND） | eps0.03：5.32 → 4.73 | 1.127 | 8/8 | 全部 bit-identical；配對 median 減少約 0.5–0.6 ms，低於 1 ms 實用門檻 |
| DeepFool | 不整合 | 不適用（pre-validation 即停止，full 與 targeted rerun 結果相同） | — | — | pre-validation：1 bit + 85 semantic same + 2 SEMANTIC_DIFFERENT |

上表列入 opt-in 選項之 profile：0 SEMANTIC_DIFFERENT、0 INVALID、0 exception、0 mkldnn state mismatch、baseline 可重現、benchmark ON path 與 production `AttackAdapter.apply()` 在 88/88 樣本 bit-identical；sign test p = 0.0078（8/8）；targeted rerun 的 FAB eps0.005／eps0.01 與 AutoAttack 全部 16 planned cells 均為 THERMAL_CLEAN accepted。

**`anchor_formal_sha_match = 0`（VMI-FGSM、VNI-FGSM、RFGSM、TPGD）**：這些攻擊使用 torch 全域 RNG，正式 runner 未於每次呼叫重設 RNG，正式 adversarial_sha256 取決於整個 run 的 RNG 推進，不可重現。`reports/meeting_20260921/meeting_report.md` §4.3 已記錄同一組 8 個 profile（PGD×3、VMI-FGSM、VNI-FGSM、RFGSM、TPGD、AutoAttack）在 CPU／NPU 兩次正式執行之間即非 bit-identical。該欄因此不適用於這些 profile；適用的檢查（同 seed 下 benchmark ON path == production apply()）為 88/88。

**AutoAttack**：正式 profile 未指定 seed；`torchattacks.AutoAttack` 於此情況以 `time.time()` 產生子攻擊 seed（targeted rerun 啟動時對已安裝 torchattacks 原始碼的檢查通過）。Full run 中 production apply() 與 benchmark path 僅 87/88 一致（anchor fidelity 失敗）。Targeted rerun 以 `--autoattack-fixed-clock`（僅 benchmark 範圍，worker log 記錄已啟用）將該 seed 固定為每列 formal seed 後，anchor fidelity 為 88/88，formal SHA 相符 6/88（與 full run 相同）。

**DeepFool 排除**：兩筆 SEMANTIC_DIFFERENT 在 full run 與 targeted rerun 中相同，索引取自 targeted rerun `profiles/deepfool__package_default/mkldnn/per_sample_validation.csv` 與 `worker_context.json`：

| sample_pos（88-sample 子集位置） | modulation | snr_db | sample_index（formal sample_index＝RadioML 該 (modulation, SNR) 陣列之列索引） | formal_seed | formal 網格全域索引（依 `configs/hailo_full_matrix.json` 之 modulations × snrs × sample_indices 順序計算，非 artifact 欄位） | true_label／clean_pred | ON → OFF 攻擊後預測 | attack success ON → OFF |
|---|---|---|---|---|---|---|---|---|
| 25 | BPSK | −20 | 1 | 306358848 | 601 | 4／7 | 7 → 10 | False → False |
| 37 | CPFSK | 6 | 1 | 1459427369 | 931 | 5／5 | 5 → 9 | False → True |

兩筆 ON 輸出的 top-1／top-2 logit margin 分別為 5.96×10⁻⁸ 與 4.77×10⁻⁷（其餘 86 筆 ON margin 中位數為 8.59×10⁻³），即位於決策邊界附近的數值容差範圍內；在僅切換 mkldnn 設定的配對比較中，ON／OFF 輸出最大絕對差為 1.86×10⁻⁹，攻擊後預測不同。此差異可重現（DeepFool 無隨機性，ON 輸出可重現，full run 與 targeted rerun 結果相同），但攻擊後預測與 attack success 改變，不符合當時之分類條件。

**Opt-in 選項狀態**：`src/adapters/attack_adapter.py` 之 `_B1_MKLDNN_OFF_EXTENDED_ATTACKS` = mifgsm、difgsm、vmifgsm、vnifgsm、rfgsm、tpgd、cw、fab、square、apgd、apgdt、autoattack、ead；以 `AttackAdapter(b1_mkldnn_off_extended=True)` 明確 opt-in，預設關閉，僅作用於 B=1 的 `atk(x, y)` 呼叫。PGD／BIM 維持獨立的 `b1_pgd_bim_mkldnn_off`，不受 extended 旗標影響。兩個選項皆預設關閉，正式 pipeline 未使用；延遲數字僅為上述 Pi 5 執行條件下之探索性量測，不作為攻擊加速方法。

## 三、Backend-toggle opt-in 選項驗證

### 3.1 PGD／BIM（`results/accel_phase1_final_integration_20260927_153520/`）

overall = PASS。Section A（option OFF 對 pre-change 參考）：FGSM／PGD／BIM 各 88/88 bit-identical。Section B（option ON）：PGD 88/88 bit-identical、BIM 87 bit-identical + 1 BITWISE_DIFFERENT_SEMANTIC_SAME（modulation = PAM4、snr_db = 6、sample_index = 1，取自 `validation.json` 之 `semantic_same_samples`，與 Phase 1D-3 一致），0 SEMANTIC_DIFFERENT、0 INVALID；FGSM 在 option ON 下維持 bit-identical（未套用）。

### 3.2 Extended 13 attacks（`results/accel_b1_extended_integration_20261004_223455/`）

overall = PASS、n_failures = 0；88 樣本；intra 4／interop 1；torch 2.14.0+cpu；base ref = HEAD；`attack_adapter.py` SHA-256 = `c09a45d7689017eb4607ffeac1e3e7bba2479cb165733cfa534cef2881c4f6bf`（與本 worktree 檔案位元組一致；該值為 CRLF 工作副本之雜湊，提交後的 LF 正規化內容雜湊不同，內容相同）。

驗證內容：option OFF 與 pre-change `AttackAdapter`（git HEAD）bit parity、meta key 與 RNG state 一致；option ON 對 option OFF 之語意分類；集合外攻擊不受影響；兩個 opt-in 旗標的交互；B=1 scope、B=2 不套用、先前狀態保留、例外後狀態還原。AutoAttack 以 production 路徑 `attack_params["seed"]`（與 `--attack-internal-seed` 相同）設為每列 formal seed，取得可重現比較；此 seed 僅用於驗證，正式 profile 未指定 seed。

Option ON 分類計數（88 樣本／profile）：

| Profile | BIT_IDENTICAL | SEMANTIC_SAME | SEMANTIC_DIFFERENT | INVALID |
|---|---|---|---|---|
| mifgsm、difgsm、vmifgsm、rfgsm、square、apgd、apgdt、autoattack | 各 88 | 0 | 0 | 0 |
| vnifgsm | 85 | 3 | 0 | 0 |
| tpgd | 86 | 2 | 0 | 0 |
| cw | 12 | 76 | 0 | 0 |
| fab eps0.005 | 67 | 21 | 0 | 0 |
| fab eps0.01 | 54 | 34 | 0 | 0 |
| fab eps0.03 | 12 | 76 | 0 | 0 |
| ead | 9 | 79 | 0 | 0 |
| 集合外（fgsm×4、pgd×3、bim、deepfool；option ON 不套用） | 各 88 | 0 | 0 | 0 |

## 四、Adaptive-K CPU effectiveness（`results/adaptive_k_effectiveness_20260927_164906/`，scope = overall，24 profiles × 2,200 = 52,800 列）

共同分母：clean accuracy 59.00%、attacked accuracy 30.73%、Conditional Attack Success Rate 67.28%（attack success 20,960 列）。

| 防禦 | recovery (%) | clean degradation (%) | retention (%) | net gain (pp) | defended accuracy (%) |
|---|---|---|---|---|---|
| 無防禦 | 0.00 | 0.00 | 32.72 | 0.00 | 30.73 |
| 固定 K=10 | 27.00 | 63.33 | 30.91 | −1.80 | 25.78 |
| 固定 K=20 | 26.34 | 32.82 | 43.32 | 10.60 | 34.31 |
| 固定 K=30 | 26.96 | 23.65 | 46.17 | 13.45 | 36.16 |
| 固定 K=40 | 27.27 | 12.71 | 48.72 | 16.01 | 37.34 |
| Adaptive-K v2 | 27.08 | 5.24 | 49.57 | 16.85 | 36.96 |

## 五、Adaptive-K Hailo paired（`results/adaptive_k_hailo_paired_20260928/`，scope = overall，52,800 列）

validation overall = PASS（全部 gates 為 true，含 CPU 重算與 Phase B 一致、Adaptive-K defended IQ 雜湊與 Phase B 一致、無攻擊重新生成）。HEF：`awn_2016_10a_c5.hef`，SHA-256 `6e54266f7e5931e432d4e5e67cd8e4a6e14863c16a94ff02a1946d21abc73dbb`。對抗輸入為 PyTorch surrogate 生成後於 Hailo-8 評估（transfer），Hailo-8 評估之 clean accuracy 54.64%、attacked accuracy 32.47%、Conditional Attack Success Rate 51.66%。

| 防禦 | CPU recovery (%) | CPU clean degr. (%) | CPU net gain (pp) | Hailo recovery (%)（95% CI） | Hailo clean degr. (%) | Hailo retention (%) | Hailo net gain (pp) | Hailo defended acc. (%) |
|---|---|---|---|---|---|---|---|---|
| 無防禦 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 48.34 | 0.00 | 32.47 |
| 固定 K=10 | 27.00 | 63.33 | −1.80 | 22.24（21.58–22.91） | 65.97 | 32.65 | −15.69 | 23.65 |
| 固定 K=20 | 26.34 | 32.82 | 10.60 | 19.24（18.61–19.88） | 33.36 | 46.31 | −2.03 | 31.77 |
| 固定 K=30 | 26.96 | 23.65 | 13.45 | 20.24（19.60–20.89） | 24.63 | 49.73 | 1.39 | 33.55 |
| 固定 K=40 | 27.27 | 12.71 | 16.01 | 21.17（20.52–21.83） | 13.23 | 53.29 | 4.95 | 34.91 |
| Adaptive-K v2 | 27.08 | 5.24 | 16.85 | 14.11（13.56–14.68） | 4.91 | 51.50 | 3.16 | 34.33 |

## 六、Formal latency（`formal_latency_20260928d`，overall PASS，10/10 cells THERMAL_CLEAN）

| 指標 | backend | 防禦 | n | median (ms) | p95 (ms) |
|---|---|---|---|---|---|
| Sensing evaluation path | host | — | 2640 | 3.081 | 3.225 |
| 固定 K（topk40） | host | topk40 | 2640 | 0.289 | 0.303 |
| Adaptive-K production | host | adaptive_k_v2 | 2640 | 0.588 | 1.118 |
| CPU AWN adapter | cpu | — | 2640 | 2.415 | 2.575 |
| Hailo adapter round trip | hailo | — | 2640 | 0.656 | 0.688 |
| Clean pipeline（evaluation path） | cpu | none | 528 | 5.600 | 5.792 |
| Clean pipeline（evaluation path） | cpu | adaptive_k_v2 | 528 | 6.064 | 6.585 |
| Clean pipeline（evaluation path） | hailo | none | 528 | 3.791 | 3.876 |
| Clean pipeline（evaluation path） | hailo | adaptive_k_v2 | 528 | 4.385 | 4.919 |

配對差（Hailo − CPU）：分類器 −1.759 ms（440 對）；clean pipeline none −1.836 ms、adaptive_k_v2 −1.651 ms（各 88 對）。上述為處理延遲量測，非任何應用情境的 deadline。

## 七、Defense recovery full comparison（`results/defense_recovery_full_20261004_215947/`）

Runner：`experiments/benchmark_defense_recovery_comparison_pi5.py --full --adv-store-dir results/adaptive_k_effectiveness_20260927_164906`。所有防禦在同一批 stored adversarial IQ（第四節之 52,800 列）上評估，分母為標準 AWN 的 clean_correct（31,152）／attack_success（20,960）；指標定義同第四節。status = complete、攻擊列 633,600（12 defenses × 52,800）、clean 列 26,400、invalid defenses 0、unavailable 0。固定 Top-K 與 Adaptive-K 的 recovered 計數（5659／5520／5651／5715／5677）與第四節完全一致。

| 防禦 | recovery (%)（95% CI） | clean degradation (%) | retention (%) | net gain (pp) | defended acc. (%) | clean defended acc. (%) |
|---|---|---|---|---|---|---|
| 無防禦 | 0.00 | 0.00 | 32.72 | 0.00 | 30.73 | 59.00 |
| 固定 K=10 | 27.00（26.40–27.60） | 63.33 | 30.91 | −1.80 | 25.78 | 27.00 |
| 固定 K=20 | 26.34（25.74–26.94） | 32.82 | 43.32 | 10.60 | 34.31 | 45.55 |
| 固定 K=30 | 26.96（26.36–27.57） | 23.65 | 46.17 | 13.45 | 36.16 | 50.05 |
| 固定 K=40 | 27.27（26.67–27.87） | 12.71 | 48.72 | 16.01 | 37.34 | 55.05 |
| Adaptive-K v2 | 27.08（26.49–27.69） | 5.24 | 49.57 | 16.85 | 36.96 | 57.32 |
| Signal filter：Kalman | 11.15（10.73–11.58） | 80.59 | 15.78 | −16.93 | 15.64 | 15.91 |
| Signal filter：Wiener | 18.78（18.26–19.32） | 39.91 | 35.92 | 3.20 | 30.89 | 42.18 |
| Signal filter：Savitzky-Golay | 16.47（15.97–16.98） | 34.67 | 36.05 | 3.33 | 31.30 | 45.82 |
| Signal filter：Gaussian | 16.15（15.66–16.66） | 33.13 | 36.10 | 3.39 | 31.20 | 45.95 |
| Signal filter：FIR | 20.94（20.39–21.49） | 70.26 | 25.28 | −7.43 | 23.54 | 24.55 |
| AT transfer（exploratory） | 53.53（52.85–54.20） | 15.95 | 66.79 | 34.07 | 45.70 | 52.18 |

五種 signal filter 使用 `external/adversarial-rf/util/defense_registry.py` 之預設參數，未針對本管線調整，亦未指定其中任何一種為唯一正式 Signal Filter。

**Exploratory AT 標示**：`external/adversarial-rf/awn_fpga/2016.10a_AWN_at.pkl`（SHA-256 `b0fa1b84…6ebfca`）由 `adv_train.py` 以 SNR ≥ 0 全部資料隨機切分訓練／驗證（seed 42、val_ratio 0.15），其訓練攻擊為 eps 0.1 之 FGSM／PGD／EAD-L1／EAD-EN，非本專案正式 profile。重現該切分後，2,200 個評估樣本中有 **928** 個位於其訓練集（`at_train_overlap.csv`：in_at_train = True 之 22,272 攻擊列 recovery 61.38%，False 之 30,528 列 recovery 43.47%；後者包含全部 SNR < 0 樣本，兩組差異同時受 SNR 分佈影響，不應解讀為 leakage 效應大小）。此 checkpoint 僅為 exploratory baseline，不作為 leakage-free 正式基準。

## 八、Leakage-free adversarial training（`results/adversarial_training_20261004_230802/`）

`experiments/train_adversarial_awn.py`（SHA-256 `721bcfaf…ac652`）與設定 `configs/adversarial_training_awn.json`（SHA-256 `ade04cb4…0a747`）；兩者雜湊與本 worktree 檔案一致：

| 欄位 | 值 |
|---|---|
| status | complete（Raspberry Pi 5 CPU，torch 2.14.0+cpu；期間 resume 1 次） |
| 架構／warm start | 同 AWN 架構；warm start = 正式 checkpoint（SHA-256 `8af0458f…820695`，與正式 manifest 一致） |
| 資料 | RML2016.10a，SNR ≥ 0 dB（0–18 dB），排除 formal sample_indices 0–9（全部調變 × SNR）；108,900 列 → train 92,565／val 16,335（seed 42，split digest `ca82afe4…66d698`） |
| 訓練攻擊 | 正式 profile fgsm/eps0.03、pgd/eps0.03（round-robin，true label，batch 256）；alpha_adversarial 0.5；WBFM／AM-DSB／AM-SSB 於對抗分支維持 clean |
| epochs | 27（early stop，patience 8；上限 30）；每 epoch 約 676–769 s |
| best epoch | 19：val clean accuracy 89.50%、val robust accuracy（FGSM eps0.03）68.91%、val_weighted 0.7921 |
| best_model.pkl SHA-256 | `e3cdc7dd108a905c8cb98b90732200211bd626ee41ecd73348aa97f275bffc15` |
| best_model.config.json | split_method = train_adversarial_awn、formal_eval_samples_excluded = true、functional_check_only = false |

## 九、Leakage-free AT-only transfer comparison（`results/defense_recovery_full_20261005_152728/`）

`--defenses adversarial_training --adv-store-dir results/adaptive_k_effectiveness_20260927_164906 --at-checkpoint results/adversarial_training_20261004_230802/best_model.pkl`；status = complete、攻擊列 105,600（none + AT）、provenance = **leakage_free**、eval_samples_in_at_train = **0**／2,200、未執行 white-box（0 列）。

| 防禦 | recovery (%)（95% CI） | clean degradation (%) | retention (%) | net gain (pp) | defended acc. (%) | clean defended acc. (%) |
|---|---|---|---|---|---|---|
| 無防禦 | 0.00 | 0.00 | 32.72 | 0.00 | 30.73 | 59.00 |
| AT transfer（leakage-free） | 56.67（55.99–57.33） | 18.57 | 68.62 | 35.90 | 46.65 | 50.73 |

分母與第七節相同（同一批 stored adversarial IQ、標準 AWN 的 clean_correct 31,152／attack_success 20,960）。欄位與定義（`defense_recovery_summary.csv`；`eval_parts/` 逐列重新計數結果相同）：

| 欄位 | 值 | 定義 |
|---|---|---|
| `recovery_pct` | 56.665 | `n_recovered` 11,877 ÷ `n_attack_success` 20,960（攻擊列） |
| `clean_degradation_pct`（summary） | 18.567 | `n_clean_degraded` 5,784 ÷ `n_clean_correct` 31,152（攻擊列，每個 clean 樣本隨 24 個 profile 重複計入） |
| `clean_degradation_pct`（`clean_degradation.csv`） | 18.567 | `n_clean_degraded` 241 ÷ `n_clean_correct` 1,298（2,200 個唯一 clean 樣本）；兩種分母比例相同 |
| `retention_pct`／`retention_nodef_pct` | 68.618／32.717 | `n_retained` 21,376 ÷ 31,152；`n_retained_nodef` 10,192 ÷ 31,152 |
| `retention_gain_pp` | 35.901 | `retention_pct` − `retention_nodef_pct` |
| `n_clean_improved`（`clean_degradation.csv`） | 59 | clean 錯誤樣本經 AT 模型分類後正確之數量 |

## 十、Main pipeline 整合狀態

| 項目 | 程式 | 驗證 | 結果 |
|---|---|---|---|
| Adaptive-K（`--use-real-topk --topk-mode adaptive`） | `src/adapters/topk_adapter.py`、`src/utils/config.py`、`src/utils/pipeline.py` | `experiments/validate_adaptive_k_pipeline_wiring.py` | Pi 互動執行完成：55 PASS、0 FAIL；僅使用暫存目錄，無持久化 artifact |
| Hailo-8 evaluation backend（`--inference-backend hailo --hef-path <hef>`；attack surrogate 固定為 CPU AWN） | `src/utils/config.py`、`src/utils/pipeline.py` | `experiments/validate_hailo_main_pipeline_wiring.py` | Pi 互動執行完成：34 PASS、0 FAIL；僅使用暫存目錄，無持久化 artifact |
| Extended B=1 mkldnn-off（探索性 backend toggle，預設關閉；main pipeline 未使用） | `src/adapters/attack_adapter.py` | `experiments/validate_attack_mkldnn_b1_extended_integration.py` | 輸出一致性驗證 overall PASS（第 3.2 節）；不作為攻擊加速方法 |

## 十一、結論邊界

1. 第七、九節所有防禦皆以「針對標準 AWN 生成、且不知道防禦存在」的對抗輸入評估（non-adaptive）。AT 列為 transfer 設定：對抗輸入由標準 AWN 生成，再由 AT 模型分類；未執行針對 AT 模型本身的 white-box 攻擊，因此不構成對自適應攻擊者的穩健性結論。
2. Leakage-free AT 的訓練攻擊（fgsm/eps0.03、pgd/eps0.03）屬於 24 個評估 profile 之列；其餘 profile 對 AT 而言為訓練未見攻擊。本文件未拆解見過／未見攻擊之差異。
3. Leakage-free AT 僅以 SNR ≥ 0 dB 訓練，評估涵蓋 −20 至 18 dB；AT 模型 clean accuracy（50.73%）低於標準 AWN（59.00%），clean degradation 18.57%。
4. Exploratory 與 leakage-free AT 的訓練設定不同（攻擊組合、eps、資料切分），兩者之差不能作為資料洩漏效應的估計。
5. Hailo-8 結果為 surrogate transfer；未評估直接針對量化模型的攻擊。
6. Backend toggle（mkldnn ON／OFF）之延遲差異為探索性 latency observation，僅適用於 Raspberry Pi 5、torch 2.14.0+cpu、intra 4／interop 1、B=1，不足以建立 ARM 平台之一般性 acceleration claim，亦不可跨平台泛化；AutoAttack 延遲分佈具重尾（p95 遠高於 median），配對比值以 median 表示。
7. Adaptive-K 與 Hailo main-pipeline wiring 驗證為互動執行結果，數字來自執行輸出，未保存可重新檢查的結果檔。
8. 全部結果屬 A0 數位域（攻擊直接作用於記憶體中的 AWN 輸入 IQ），不涉及空中傳輸或實體射頻通道，不能據以推論 OTA／RF 攻擊或防禦的可行性。
9. Adaptive-K 與 AT 呈取捨關係：在同一分母下，leakage-free AT transfer 的 recovery（56.67%）與 net gain（35.90 pp）較高，但 clean degradation（18.57%）較高且需要重新訓練；Adaptive-K v2 的 recovery（27.08%）較低，clean degradation（5.24%）較低且不需訓練。兩者評估設定皆為 non-adaptive，不據此排定一般性的優劣。