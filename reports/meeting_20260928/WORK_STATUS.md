# 工作狀態：meeting_20260928 報告

更新日期：2026-09-28（尚未 git add、commit 或 push，等待人工 review）

## 1. 進度

| 階段 | 狀態 |
|---|---|
| Evidence 稽核 | 完成，見 `analysis/evidence_audit.md` |
| 報告撰寫 | 完成：`meeting_report.md`、`executive_summary.md` |
| 彙總表 | 完成：`tables/*.csv`，數值由 evidence 檔案轉錄，每列附來源 |
| 圖表 | 完成：`figures/fig1`–`fig3`，由 `analysis/make_figures.ps1`（Windows PowerShell 5.1，.NET System.Drawing）直接讀取 evidence 產生；正式繪圖數值來源為 `tables/figure_sources/` |
| Review evidence package | 已建立（`review_evidence/`，不含 raw／per-sample 檔案）；外部查核附件，不屬 repository 正式依賴 |

## 2. 採用的正式 evidence（唯讀，未修改）

皆位於 `meeting_20260928_evidence/results/`：

- 攻擊加速：`accel_phase1a_formal_20260926_152737`、`accel_phase1b_20260926_155342`、`accel_phase1c_bim_diag_20260926_162116`、`accel_phase1c_bim_semantic_audit_20260926_162812`、`accel_phase1c_v2_formal_20260926_164547`、`accel_phase1d0_thermal_20260927_092156`、`accel_phase1d0b_pair_20260927_111224`、`accel_phase1d1_profile_20260927_120159`、`accel_phase1d2_backend_20260927_{123643,124116,124938}`、`accel_phase1d3_mkldnn_b1_20260927_135420`、`accel_phase1_final_integration_20260927_153520`
- Adaptive-K：`adaptive_k_parity_20260927_162249`、`adaptive_k_effectiveness_20260927_164906`、`adaptive_k_hailo_paired_20260928`
- 延遲：`formal_latency_20260928d`（PRIMARY_FORMAL）

## 3. 未採用（僅 provenance）

`accel_phase1c_preflight_20260926_161012`（FAIL）、`accel_phase1d2_backend_20260927_{125606,130303,130330,130336}`（未完成）、`adaptive_k_parity_20260927_160808`（FAIL）、`formal_latency_20260928`（dry-run 中止）、`formal_latency_20260928b`（3/10）、`formal_latency_20260928c`（9/10）。各 preflight／dry-run 目錄僅供參考。

## 4. Review 前注意事項

- `meeting_20260928_evidence/`、根目錄的兩個 `.tar.gz` 與其他 untracked／modified 檔案皆非本報告產生；commit 時請勿一併加入。
- `results/` 已被 `.gitignore` 排除。
- 報告第 8 節列出目前無法判定之處，建議 review 時優先核對。
- 第 10 節 traceability 與第 11 節後續工作只列入在 0921 報告或 repository 規劃文件中有直接來源的項目；找不到原始條目的項目未列為待辦（檢索紀錄見 `analysis/evidence_audit.md` §5）。
- Adaptive-K／Top-K 的 main pipeline 整合狀態依本機 repository 程式碼判定；Pi 端 `src/utils/pipeline.py` 為 modified 且未含於 evidence。
