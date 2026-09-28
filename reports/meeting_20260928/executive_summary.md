# 執行摘要：Pi 5 攻擊生成加速、Adaptive-K 防禦與 CPU／Hailo-8 正式延遲

**報告日期**：2026-09-28　｜　**涵蓋期間**：2026-09-21 checkpoint 之後至 2026-09-28　｜　**平台**：Raspberry Pi 5 ＋ Hailo-8　｜　**資料集**：RadioML 2016.10a

所有數值取自 `meeting_20260928_evidence/results/` 內各實驗的 validation 與彙總檔。完整分析見 [`meeting_report.md`](meeting_report.md)，稽核紀錄見 [`analysis/evidence_audit.md`](analysis/evidence_audit.md)。

**威脅模型**：與 0921 報告相同，為 A0 digital classifier-input attack。Hailo-8 端的攻擊為 surrogate-model transfer attack（對抗樣本由 PyTorch AWN 生成），不是針對量化模型的白箱攻擊。

## 主要結果

| # | 結果 | 數值 | 來源 |
|---|---|---|---|
| 1 | **B=1 攻擊生成加速**（PGD／BIM，88 樣本，熱控制） | 只在攻擊呼叫期間關閉 mkldnn：PGD 42.53 → 36.33 ms，BIM 43.84 → 37.63 ms（中位）；配對差約 −6.2 ms，8/8 對，sign test p = 0.0078；SEMANTIC_DIFFERENT = 0 | `accel_phase1d3_mkldnn_b1_20260927_135420` |
| 2 | **整合驗證** | opt-in 選項 `b1_pgd_bim_mkldnn_off`，預設關閉；選項關閉時輸出與修改前 88/88 相同；範圍檢查全數通過；完整 `apply()` 路徑中位 PGD 46.76 → 39.52 ms、BIM 48.04 → 40.80 ms；正式 CPU full matrix 尚未使用此選項重跑 | `accel_phase1_final_integration_20260927_153520` |
| 3 | **Batching** | PGD／BIM 之 throughput 加速比最高 5.16×（BIM B=32 intra4；PGD 同條件為 5.03×），但 PGD／BIM 的批次完成時間皆高於 40 ms，不降低單樣本決策延遲 | `accel_phase1c_v2_formal_20260926_164547` |
| 4 | **Adaptive-K parity** | 真實資料（2200 clean、264 attacked）之決策差異為 0；此 PASS 是在 batch vs N=1 gate 由 bitwise 條件改為決策＋容差條件後取得（第一次 run 為 FAIL） | `adaptive_k_parity_20260927_{160808,162249}` |
| 5 | **Adaptive-K 效果（CPU，24 profile 合併）** | Clean degradation 5.24%（K=40：12.71%）；retention 49.57%；淨增益 +16.85 pp（K=40：+16.01 pp） | `adaptive_k_effectiveness_20260927_164906` |
| 6 | **Adaptive-K 效果（Hailo-8 配對）** | Clean degradation 4.91%（為各防禦最低）；recovery 14.11%（K=40：21.17%）；淨增益 +3.16 pp（K=40：+4.95 pp） | `adaptive_k_hailo_paired_20260928` |
| 7 | **正式延遲（量測值）** | Hailo adapter 0.656 ms vs CPU adapter 2.42 ms（中位；配對差 −1.76 ms，440/440）；clean pipeline（evaluation path）CPU 5.60 ms、Hailo 3.79 ms；加上 Adaptive-K 後為 6.06、4.38 ms | `formal_latency_20260928d`（PRIMARY_FORMAL，10/10 accepted） |
| 8 | **正式延遲（推導值與輔助值）** | Deployment estimate（推導，非量測）：Hailo 3.63 ms、CPU 5.42 ms；HailoRT HW latency（輔助、非配對）0.17 ms | 同上 |

## 圖表

| | | |
|---|---|---|
| ![圖 1](figures/fig1_attack_acceleration.png) | ![圖 2](figures/fig2_adaptive_k_tradeoff.png) | ![圖 3](figures/fig3_formal_latency.png) |
| **圖 1**　MKLDNN ON／OFF 之 B=1 攻擊延遲 | **圖 2**　固定 K 與 Adaptive-K 之 trade-off（CPU／Hailo-8 分列） | **圖 3**　直接量測延遲（PRIMARY_FORMAL） |

## 結論

1. 在限定條件（PGD／BIM、B=1、88 樣本、熱控制）下，攻擊生成可降至 40 ms 以下，且未觀察到攻擊語意改變；尚未擴及其他攻擊與正式 full matrix。
2. Adaptive-K 在 CPU 上以最低的 clean 代價得到與 K=40 相近的淨增益；在 Hailo-8 上 clean 代價仍最低，但淨增益低於 K=40。防禦的相對優劣依部署 backend 而異。以上為演算法與實驗驗證；本機 repository 的 main pipeline（`src/utils/pipeline.py`、`TopKAdapter`）尚未接入 Adaptive-K。
3. 正式延遲量測區分量測值、配對統計、推導估計與輔助硬體量測；Hailo pipeline 中 host 端 sensing 仍佔主要時間。

## 主要限制

- 攻擊加速只涵蓋 3 種攻擊與 88 樣本；語意相同只比較 attacked prediction 與 attack_success。
- Adaptive-K parity 的 PASS 依賴修訂後的 gate；batched 與 N=1 輸出並非 bitwise 相同。
- 延遲 benchmark 只涵蓋 5 個 block，不含攻擊生成；Hailo 端溫度無法讀取。
- 詳見 [`meeting_report.md` §9](meeting_report.md#9-限制與有效性威脅)；研究工作 traceability 與後續工作見 §10、§11。
