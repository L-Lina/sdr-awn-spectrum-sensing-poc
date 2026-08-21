# Research Documents Index

This directory holds the project's formal research findings, written in
Traditional Chinese for a professional research audience. Each document is
self-contained and cites its own evidence (result directories, CSVs,
figures); none of them are meeting notes or working logs. Suggested reading
order:

1. **[CURRENT_SYSTEM_AND_COMPONENT_STATUS_ZH_TW.md](CURRENT_SYSTEM_AND_COMPONENT_STATUS_ZH_TW.md)**
   -- 系統與元件現況：研究問題、威脅模型（A0/A1/A2、G1-G8 攻擊目標）、各元件
   （Spectrum Sensing／AWN／Attack／Top-K／cfile）完成狀態與驗證範圍。從這份
   文件開始，了解整個系統目前能回答哪些問題、不能回答哪些問題。

2. **[DIGITAL_LOW_PERTURBATION_ATTACK_EXPERIMENT_ZH_TW.md](DIGITAL_LOW_PERTURBATION_ATTACK_EXPERIMENT_ZH_TW.md)**
   -- A0 數位白箱低擾動攻擊基準實驗：FGSM／PGD／CW／DeepFool／EAD／FAB 六種
   攻擊在真實 AWN checkpoint 上的準確率下降、conditional ASR、擾動範數與延遲。

3. **[PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md](PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md)**
   -- 效能與延遲分析：clean AMC／end-to-end pipeline 延遲、FGSM／PGD／CW 加速
   與各自的 batching 安全分類、thread tuning、streaming sensing 雛型延遲。

4. **[SATELLITE_APPLICATION_AND_LATENCY_REQUIREMENTS_ZH_TW.md](SATELLITE_APPLICATION_AND_LATENCY_REQUIREMENTS_ZH_TW.md)**
   -- 衛星應用情境與延遲需求（Step 1）：3GPP TS 22.261 NTN 延遲參考、DVB-S2/S2X
   調變族群、部署情境與威脅模型定義、MUST/SHOULD/OPTIONAL channel factor 分類。

5. **[SATELLITE_DATASET_AND_MODULATION_FEASIBILITY_ZH_TW.md](SATELLITE_DATASET_AND_MODULATION_FEASIBILITY_ZH_TW.md)**
   -- 衛星資料集與調變可行性（Step 2）：RadioML2018.01A 稽核與 project-close
   策略決定（Strategy A：RadioML2016.10a，BPSK/QPSK/8PSK，不重新訓練）。

6. **[SATELLITE_LIKE_CHANNEL_SIMULATOR_DESIGN_ZH_TW.md](SATELLITE_LIKE_CHANNEL_SIMULATOR_DESIGN_ZH_TW.md)**
   -- 衛星式通道模擬器設計與驗證（Step 3）：AWGN／amplitude scaling／CFO／
   Doppler／timing offset 實作、單元測試、108-sample smoke test，以及 amplitude／
   CFO／Doppler 對 AMC accuracy 影響的 root-cause 專項驗證。

7. **[SATELLITE_LIKE_FINAL_EXPERIMENT_ZH_TW.md](SATELLITE_LIKE_FINAL_EXPERIMENT_ZH_TW.md)**
   -- 最終衛星式整合實驗（Step 4）：576 組合正式矩陣，Spectrum Sensing／AMC／
   FGSM／PGD(det)／Top-K 在四級 channel condition 下的完整結果，含事後 final
   result audit（metric 定義、denominator 修正、pseudo-replication 澄清）。

## Not a research document

- `docs/PROJECT_STATUS.md` -- English-language engineering/handoff status
  summary (capability inventory, regression evidence, reproducibility). Not
  a research finding document; read it for "what state is the repo in",
  not "what did we learn."
- `docs/PROJECT_CLOSE_CHECKLIST.md` -- project-close checklist (PASS /
  LIMITATION / NOT_IMPLEMENTED per item with evidence paths).
- `docs/ATTACK_NAME_MAPPING.md`, `docs/ATTACK_COMPATIBILITY_WORKLIST.md`,
  `docs/DEPLOYMENT_READINESS.md`, `docs/parameter_validation.md`,
  `docs/formal_experiment_plan.md` -- engineering audit trails (attack
  registry mapping, SDR deployment readiness, parameter validation, the
  original Phase 0-4 experiment plan). Useful as evidence sources cited
  from the research documents above, not primary reading.

## Conventions used across these documents

- Traditional Chinese, formal research register, no meeting-note tone, no
  AI/tool attribution.
- Every number traces to a specific result directory/CSV/figure -- never
  hand-copied or invented.
- CW/DeepFool/EAD are never collectively referred to as "最小擾動攻擊"; each
  attack is named individually where the distinction matters.
- `attack_success`-derived metrics are always labeled precisely (Attacked
  Accuracy / Prediction Change Rate / Conditional Attack Success Rate,
  never a bare, unqualified "ASR") -- see
  `SATELLITE_LIKE_FINAL_EXPERIMENT_ZH_TW.md` section 14/22 for the full
  definitions.
- RadioML2016.10a is never referred to as a real satellite dataset; the
  satellite-like channel model is never referred to as standards-compliant
  DVB-S2/S2X; `amplitude_scale` is never equated with a full RF link
  budget; processing-latency figures are never referred to as a satellite
  deadline.
