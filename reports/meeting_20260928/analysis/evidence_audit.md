# Evidence 稽核紀錄（meeting_20260928）

稽核對象：`meeting_20260928_evidence/`（`experiments/` 16 支腳本、`results/` 各目錄）、`reports/meeting_20260921/`、repository 規劃文件。判定依據為各目錄的 `validation.json`（`overall`、gates）、`manifest.json`（`status`、`configuration_role`）與彙總檔；大型 per-sample／raw CSV 未全文讀取，僅透過 summary 與 validation 交叉核對。

## 1. 前次 checkpoint baseline

`reports/meeting_20260921/`：CPU（`results/cpu_full_matrix_20260921_seedaligned`）與 NPU（`results/hailo_full_matrix_20260920_seedaligned`）full matrix；2200 樣本、24 個攻擊 profile、固定 K Top-K ∈ {10, 20, 30, 40}；Phase A 資料驗證 113 項 PASS。已報告：clean 準確度、conditional ASR、surrogate transfer 落差、固定 K 防禦、CPU／NPU 延遲比較。明列限制：未評估自適應 K；延遲量測未記錄硬體狀態。

## 2. 逐目錄判定

| 目錄 | 關鍵欄位 | 判定 |
|---|---|---|
| `accel_phase1_preflight_20260926_152214` | 2 條件；FGSM／BIM formal_equivalence PASS | preflight |
| `accel_phase1a_formal_20260926_152737` | manifest `status = complete`；8 workers；raw rows 12,672 | 採用（熱狀態：溫度升至 84.8 °C，結束時 `throttled=0xe0000`） |
| `accel_phase1b_20260926_155342` | `overall = PASS`；所有變體 264/264 bit-identical | 採用 |
| `accel_phase1c_preflight_20260926_161012` | `overall = FAIL (worker intra1_interop1 aborted, rc=1)`；manifest `status = aborted` | 不採用；provenance |
| `accel_phase1c_bim_diag_20260926_162116` | `overall = PASS (diagnostic fidelity checks)` | 採用（診斷） |
| `accel_phase1c_bim_semantic_audit_20260926_162812` | `overall = COMPLETE`；SEMANTIC_DIFFERENT 0 | 採用（稽核） |
| `accel_phase1c_v2_preflight_20260926_164022` | 12 cells、0 fail、PASS（thermal review 旗標） | preflight |
| `accel_phase1c_v2_formal_20260926_164547` | 72 cells、0 fail；`PASS (workers flagged for thermal/throttling review)` | 採用（絕對值以熱控制結果為準） |
| `accel_phase1d0_thermal_20260927_092156` | 12/12 ran、0 fail、1 THERMAL_REVIEW | 採用；BIM B=1 thread 比較 `claim_allowed = false` |
| `accel_phase1d0b_pair_20260927_111224` | 16/16 ACCEPTED、全部 THERMAL_CLEAN、`overall = PASS` | 採用 |
| `accel_phase1d1_profile_20260927_120159` | 6/6 blocks THERMAL_CLEAN、`overall = PASS` | 採用 |
| `accel_phase1d2_backend_20260927_123643`、`_124116`、`_124938` | `overall = PASS`、6/6 THERMAL_CLEAN | 採用 |
| `accel_phase1d2_backend_20260927_125606`、`_130303`、`_130330`、`_130336` | manifest `status = running`；無 validation.json | 不採用（未完成） |
| `accel_phase1d3_mkldnn_b1_20260927_135420` | 32/32 ACCEPTED、0 thermal review、`overall = PASS` | 採用 |
| `accel_phase1_final_integration_statediag_20260927_153354` | mode `diagnose_state`；FGSM 3 筆 PASS | 診斷，不作為結果 |
| `accel_phase1_final_integration_20260927_153520` | `overall = PASS`；scope checks all pass | 採用 |
| `adaptive_k_parity_20260927_160808` | `overall = FAIL`（gate 6、7 false；batch parity 1110/2514 fail） | 不採用；provenance |
| `adaptive_k_parity_20260927_162249` | `overall = PASS`（gate 6、7 定義已修訂） | 採用（註明 gate 修訂） |
| `adaptive_k_effectiveness_20260927_164906` | `overall = PASS`；row counts 符合 | 採用 |
| `adaptive_k_hailo_paired_20260928` | `overall = PASS`（summarize）；evaluate 25/25 parts；dry_run PASS | 採用 |
| `formal_latency_20260928` | preflight PASS；dry-run `overall = FAIL`，`abort = FileNotFoundError`，0/2 | 不採用 |
| `formal_latency_20260928b` | dry-run PASS；run `overall = FAIL`，3/10 accepted | 不採用 |
| `formal_latency_20260928c` | run `overall = FAIL`，9/10 accepted（`00_r0_clean` START_GATE_REJECTED） | 不採用 |
| `formal_latency_20260928d` | `configuration_role = PRIMARY_FORMAL`；`overall = PASS`；10/10 accepted；全部 gates true | 採用（PRIMARY_FORMAL） |

## 3. Hailo 配對涵蓋之攻擊設定

`eval_parts/00`–`23`：fgsm eps0.005／0.01／0.03／0.05、bim eps0.03_default、pgd eps0.005／0.01／0.03、mifgsm、difgsm、vmifgsm、vnifgsm、rfgsm、tpgd（皆 eps0.03_default）、cw c1_steps20_lr0.01、deepfool package_default、fab eps0.005／0.01／0.03、square eps0.03_default、apgd eps0.03_default、apgdt eps0.03_default、autoattack eps0.03_standard、ead package_default。共 24 個，與 0921 正式 matrix 相同。

## 4. 無法判定或互相矛盾之處

見 `meeting_report.md` §8。

## 5. Repository 規劃文件檢索

- 檢索 `signal filter`、`adversarial training`、`對抗訓練` 等關鍵字：repository 與 evidence 中未見對應實作或實驗設定。
- 檢索 `U1`–`U7`、`recovery-parity`、`K=1-excluded`、`native-Hailo`、`primary transfer metric`：未找到對應文字。
- Target attack decode、targeted-modulation、waveform calibration 腳本僅出現在 Pi 端 manifest 的 `git_status_summary`（untracked），本機 repository 與本次 bundle 均無對應結果目錄。
- 以上檢索未找到原始規劃條目的項目，未列入 `meeting_report.md` 的 traceability 與後續工作。
- Main pipeline 整合檢查：本機 `src/adapters/topk_adapter.py` 註明 `adaptive_k_v2_defense`「not wired yet」；`src/utils/pipeline.py` 只呼叫固定 K 之 `TopKAdapter`，未指定 `--use-real-topk` 時使用 `dummy_topk_defense`，且無 Hailo backend 與 mkldnn 選項；`experiments/run_hailo_full_matrix.py` 與 `configs/hailo_full_matrix.json` 無 Adaptive-K 設定。`docs/PROJECT_STATUS.md` Part 2 §4 將 `adaptive_k_v2_defense` 列為「Not implemented」。
