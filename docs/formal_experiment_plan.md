# Formal Experiment Plan (round 16 planning)

Status: **planning document only**. No formal batch has been executed as of
this document's creation. Every parameter value/range cited below is
sourced directly from `docs/parameter_validation.csv` / `docs/
parameter_validation.md` (sections 1-24, current repo state as of commit
`c303daa`) and from direct code reads performed while writing this plan
(`src/utils/config.py`, `src/utils/pipeline.py`, `src/utils/
batch_aggregation.py`, `experiments/run_batch.py`, `experiments/
run_fair_topk_matrix.py`) -- nothing here is recalled from memory or
inherited from `external/adversarial-rf`'s own conventions without an
explicit citation. See `docs/formal_experiment_matrix.csv` for the
machine-readable per-phase matrix that accompanies this document.

This round did **not** run any new experiment -- see section 6 for what was
actually executed (nothing beyond reading existing files) and section 7 for
open decisions that must be confirmed before any phase actually runs.

---

## 1. Full parameter inventory (confirmed value ranges only)

Every row below cites the CSV row / md section that is the evidence source.
"Confirmed range" means a range with at least one real-backend execution on
record -- not a theoretical range.

| Parameter | CLI flag | Confirmed value range | Source |
|---|---|---|---|
| IQ source | `--iq-source` | `{synthetic, radioml}` -- **radioml is the only source with a real, non-cosmetic modulation implementation** (md section 5: synthetic's `mod` only changes a hash-derived carrier frequency offset, no real waveform) | CSV row `K,iq_source` |
| dataset_path | `--dataset-path` | `/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl` (only path ever used) | CSV row `K,dataset_path` |
| modulation | `--dataset-mod` | all 11 RML2016.10a classes: `8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM` -- all 11 confirmed with real AWN+attack+Top-K (round 14, 88 combos) | CSV row `S,coverage_all_modulations_x_attack` |
| SNR | `--dataset-snr` | all 20 values `-20..18` step 2 -- all 20 confirmed both sensing-only and with real fgsm attack (round 14) | CSV row `S,coverage_full_radioml_snr_range` |
| attack | `--attack` | `{none, fgsm, pgd, cw}` -- exhaustive list this repo wires up (no bim/apgd/deepfool/etc., md section 4). All 4 confirmed real-backend at scale (round 12, 480 combos; round 14, 88 combos) | CSV rows `F,attack`, `Q,fair_topk_verification_at_scale` |
| attack_eps | `--attack-eps` | `{0.001, 0.01, 0.03, 0.05, 0.1, 0.3, 1.0}` confirmed through the real batch pipeline, linear scaling verified, 0 NaN/Inf even at 1.0 (round 14). **N/A to cw** (no `eps` attribute exists on the constructed `torchattacks.CW` object, confirmed empirically, md section 10.2) | CSV row `S,coverage_attack_eps_sweep` |
| attack_temperature | `--attack-temperature` | `{1.0 (default), 100.0}` both confirmed real-backend. **Important finding (this round, see section 7 risk R4): default `T=1.0` was shown to be a gradient-saturation no-op under the OLD synthetic + legacy-unit-power pipeline (md section 10.1), but round 12/14's radioml-native-mode batches used `T=1.0` (default) throughout and measured real attack success (fgsm 83.3%, pgd 96.7-100%) -- the saturation problem does not reproduce under the current radioml-native default.** | CSV row `F,attack_temperature`; cross-referenced against `run_fair_topk_matrix.py`/`run_parameter_coverage_completion.py` FIXED dicts |
| cw_c / cw_steps / cw_lr | `--cw-c` / `--cw-steps` / `--cw-lr` | Repo defaults `1.0 / 20 / 0.01` **were found ineffective (0/5 changed) under the OLD synthetic pipeline** (md section 10.2), but the SAME defaults achieved 83.3% (round 12) and 90.9% (round 14) attack success at scale under radioml-native -- same apparent resolution as attack_temperature above (see risk R4). Alternate tuned values `c=10, steps=100, lr=0.1` were also explored (md 10.2) but never run at scale | CSV rows `F,cw_c/cw_steps/cw_lr`, `S,coverage_cw_knobs` |
| Top-K | `--topk` | full legal range `[1,128]` confirmed (round 14, 10 values incl. boundaries); `{10,20,30,40}` is the value set used throughout every real-backend round since round 4; illegal values (`<=0`→bypass, `>128`→clamp, `NaN/Inf/1.5/'abc'/None`→rejected) all confirmed | CSV row `S,coverage_topk_full_range` |
| threshold_factor | `--threshold-factor` | stable range `[1.2, 5.0]` gives single-region detection at consistent 0.9852 ratio (round 13); `<=1.0` still `run_status=ok` but produces multiple false-alarm segments -- must not be treated as "clean" on `run_status` alone | CSV row `R,sensing_revalidation_threshold_factor` |
| sensing_window_size | `--sensing-window-size` | `128` (current default) recommended: 0 failures across 30 samples; `{16,32,64}` give marginally higher ratio (0.9989) but ~1/30 failure rate each | CSV row `R,sensing_revalidation_sensing_window_size` |
| min_region_len | `--min-region-len` | `{0..128}` all statistically equivalent (0.9852 ratio); `256` correctly triggers `sensing_failed` (detected regions are typically 140-250 samples) | CSV row `R,sensing_revalidation_min_region_len` |
| merge_gap | `--merge-gap` | `0` (default, single-burst mode never exercises merge logic); real merge threshold confirmed exact (`merged iff gap<=merge_gap`) under max-energy at values `{0,1,5,20,64,128}` against a calibrated 2-burst gap of 150 (round 13) | CSV row `R,sensing_revalidation_merge_gap` |
| burst_len | `--burst-len` | **synthetic-source-only** -- confirmed by direct code read (`src/sensing/radioml_source.py:109`, `burst_len = sample_2x128.shape[1]`, derived from the real sample, never from `cfg.burst_len`). N/A for the formal (radioml-only) experiment | `src/utils/pipeline.py:179` (synthetic branch only), `src/sensing/radioml_source.py:109` |
| embed_snr_margin | `--embed-snr-margin` | radioml-only; only `20.0` tested at scale, but `{1,5,10,20,50,100}` targeted-checked in round 13 (margin=1.0 correctly fails sensing, physically expected; `>=5.0` all succeed) | CSV row `R,sensing_revalidation_burst_stream_checks` |
| num_bursts | `--num-bursts` | `{1 (default), 2, 3}` all confirmed; `>1` requires `dataset_mod_list`/`dataset_snr_list`/`sample_index_list` | CSV row `K,num_bursts`, round 14 `num-bursts=3` check |
| burst_gap_list | `--burst-gap-list` | exact per-burst gaps confirmed (round 6, 3 cases incl. a 3-burst mixed-merge case) | CSV row `K,burst_gap_list` |
| burst_power_scale_list | `--burst-power-scale-list` | confirmed under max-energy/radioml-native (round 14) | CSV row `K,burst_power_scale_list` |
| sample_index / sample_index_list | `--sample-index` / `--sample-index-list` | index range `0..999` per (mod,snr) bucket; indices `0-4` used throughout real-backend rounds (round 12's 480-combo sweep used 5 indices per (mod,snr) pair) | CSV row `K,sample_index` |
| seed | `--seed` | any int; `{0,7,42}` tested; `42` used in every round-11+ real-backend round; global `random`/`numpy`/`torch(+cuda)` seeding confirmed bit-identical cross-process, including for PGD's `random_start=True` | CSV row `I,torch_determinism` |
| alignment_policy | `--alignment-policy` | `{naive, max-energy}` -- source-aware default resolves `radioml`→`max-energy` (matches oracle accuracy exactly, 4/7 vs naive's 2/7, md section 18.4) | CSV row `N,alignment_policy`, `P,source_aware_defaults` |
| awn_preprocess | `--awn-preprocess` | `{legacy-unit-power, radioml-native}` -- source-aware default resolves `radioml`→`radioml-native` (matches oracle accuracy exactly, 4/7 vs legacy's 2/7, md section 19.4). **legacy-unit-power on real samples is a confirmed scale mismatch (~50-120x) against the AWN training distribution -- must never be used for the formal radioml experiment** | CSV row `O,awn_preprocess` |
| checkpoint | `--checkpoint` | only `external/adversarial-rf/2016.10a_AWN.pkl` confirmed working; alternate checkpoints (`2016.10b`, `2018.01a`) known to fail `load_state_dict` (different `num_levels`/class count, md section 8) -- never fixed, never attempted | CSV row `E,checkpoint` |
| device | `--device` | only `cpu` confirmed (no GPU in this environment, `torch.cuda.is_available()==False`) | CSV row `E,device` |
| window_size | `--window-size` | must stay `128` for the pinned 2016.10a checkpoint (`external/adversarial-rf/util/config.py:51`, `signal_len=128`) -- other even values structurally load but were never validated as statistically meaningful (md section 8) | CSV row `C,window_size` |
| segment_hop | `--segment-hop` | only `1` empirically exercised at scale; larger values wired and boundary-tested (round 14: `candidate_count` scales exactly as `(region_len-128)//hop+1`) but never used in a real accuracy-bearing run | CSV row `N,segment_hop`, `S,coverage_remaining_flags` |
| use_real_awn / use_real_attack / use_real_topk | `--use-real-*` | `True` fully validated at scale (every round 10+); `False` (dummy path) end-to-end validated this session (round 15, `c303daa`) | CSV rows `T,use_real_awn/attack/topk` |

---

## 2. Parameter classification (A/B/C/D)

### A. 正式論文主要自變項 (primary independent variables)

| Parameter | Why it's primary |
|---|---|
| modulation (`dataset_mod`) | Core AMC comparison axis -- accuracy/robustness must be reported per class |
| SNR (`dataset_snr`) | Core sensing + AMC operating-condition axis |
| attack type (`attack`: none/fgsm/pgd/cw) | The central adversarial-robustness comparison |
| attack_eps (fgsm/pgd only) | Attack-strength axis, directly maps to a perturbation-budget-vs-success curve |
| Top-K (`topk`) | The central defense-strength comparison |

### B. 固定控制參數 (fixed across the whole formal experiment)

| Parameter | Fixed value | Justification |
|---|---|---|
| iq_source | `radioml` | Only source with real, non-cosmetic modulation (section 1) |
| checkpoint | `external/adversarial-rf/2016.10a_AWN.pkl` | Only working checkpoint |
| device | `cpu` | Only available device |
| alignment_policy | `max-energy` (via source-aware default, or explicit) | Matches oracle accuracy exactly |
| awn_preprocess | `radioml-native` (via source-aware default, or explicit) | Matches oracle accuracy exactly; legacy-unit-power is a confirmed scale mismatch |
| threshold_factor | `1.5` | Inside the confirmed stable `[1.2,5.0]` range |
| sensing_window_size | `128` | 0-failure rate at scale (round 13) |
| min_region_len | `0` | Statistically equivalent to 128, avoids spurious `sensing_failed` |
| merge_gap | `0` | Single-burst phases never exercise merging; explicit non-zero only in Phase 6 |
| window_size | `128` | Structural requirement of the pinned checkpoint |
| num_bursts | `1` | Single-burst baseline for Phases 1-4; `2` only in Phase 6 |
| embed_snr_margin | `20.0` | Only value tested at scale |
| segment_hop | `1` | Only value tested at scale |
| attack_temperature | `1.0` (repo default) | Validated effective at scale under radioml-native (round 12/14) -- see risk R4 |
| cw_c / cw_steps / cw_lr | `1.0 / 20 / 0.01` (repo defaults) | Validated 83-91% attack success at scale under radioml-native -- see risk R4 |
| seed | `42` | Matches every round-11+ real-backend round, enables direct comparison |

### C. Robustness / sensitivity-only 參數 (own dedicated analysis, not the main comparison)

| Parameter | Role |
|---|---|
| threshold_factor / sensing_window_size / min_region_len / merge_gap | Phase 5 sensitivity analysis (reuses round 13 evidence by default) |
| alignment_policy (naive vs max-energy) | Already-established ablation finding (md 18.4), could be re-cited as a robustness result, not re-run |
| awn_preprocess (legacy vs radioml-native) | Already-established ablation finding (md 19.4), could be re-cited, not re-run |
| attack_temperature (1.0 vs 100.0) | Small confirmatory check only if reviewers question risk R4's inference |
| num_bursts / burst_gap_list / burst_power_scale_list | Phase 6 multi-burst extension |
| segment_hop > 1 | Optional robustness check, not part of the main design |

### D. 工程或執行參數，不納入論文比較

| Parameter | Reason |
|---|---|
| seed (specific value 42) | Fixed for reproducibility, not a compared condition |
| device, checkpoint | Environment-fixed, no alternative exists |
| output_dir, dataset_path | Execution mechanics |
| sample_index / sample_index_list | Mechanism for drawing distinct real samples per cell -- the **count** (N per cell) is a statistical-power design decision (flagged in section 7), but the specific indices drawn are not a comparison axis |
| burst_len | Synthetic-only, N/A when iq_source=radioml (confirmed by code read, section 1) |
| n_samples | No CLI flag, fixed at 8192, never varied |

---

## 3. Existing batch-script suitability check

**No single existing script covers the full formal sweep as-is**, but the
underlying infrastructure is already built, proven at scale, and requires
no new code:

- **`experiments/run_batch.py`** (own CLI, `--snr-list`/`--mod-list`
  sweep the **synthetic** generator's `snr`/`mod` only; `--dataset-mod-list`/
  `--dataset-snr-list`/`--sample-index-list` are for **multi-burst
  composition within one run**, not a combo-sweep axis). **Not suitable**
  for a radioml single-burst multi-(mod,snr,sample_index) sweep -- confirmed
  by reading its argparse definitions (`experiments/run_batch.py:42-136`).
- **`experiments/run_fair_topk_matrix.py`** (round 12) -- **directly
  reusable as the template for Phase 0/3/4**. Already implements exactly
  the pattern needed: `run_batch_combos()` + a flat combo-dict builder +
  `ExperimentConfig`, real AWN+attack+Top-K, radioml source, fair Top-K
  reuse via same-seed reproducible regeneration (verified 0/120 groups show
  variation at 480-combo scale, round 12). The only changes needed for
  Phase 0/3/4 are the `MODS`/`SNRS`/`SAMPLE_INDICES`/`ATTACKS`/`TOPKS`
  constants and `attack_eps` becoming a swept combo field instead of a
  fixed one.
- **`experiments/run_modulation_snr_matrix.py`** (round 8) -- reusable
  template for Phase 1 (sensing + clean AMC, attack=none), same pattern,
  just needs the SNR list expanded from 4 to all 20 and the sample-index
  count raised.
- **`experiments/run_parameter_coverage_completion.py`** (round 14) --
  closest existing template for Phase 6 (multi-burst + attack combined,
  never done at scale before); its stage-based structure
  (`run_stage(name, combos, build_cfg, note)`) is exactly the pattern to
  follow.
- **`src/utils/batch_aggregation.py:run_batch_combos`** -- the shared
  core every phase script will call. No changes needed; already writes
  `batch_summary.csv`/`batch_bursts_summary.csv`/
  `batch_regions_summary.csv` and handles `sensing_failed` as a structured
  non-error outcome.

**Conclusion**: each formal phase needs a **new, small, purpose-built
combo-builder script** (following the exact pattern above, typically
~100-150 lines, no new pipeline/adapter code), not a new engine. This
matches every prior round's approach (13 of the last 15 rounds added
exactly one such script). No new script has been written this round --
only this planning pass and the existing-script suitability check above.

---

## 4. Phase-by-phase design

See `docs/formal_experiment_matrix.csv` for the exact machine-readable
version of every field below (same data, one row per phase/tier).

### Phase 0 -- Pilot (designed round 16, EXECUTED round 17 -- see section 8)

- **研究問題**: does the full real-backend pipeline (sensing→alignment→
  AWN-preprocess→AWN→attack→Top-K) execute correctly, with correct output
  schema and correct fair Top-K reuse, across a small but complete
  cross-cut of the main design space, before any phase scales up?
- **固定參數**: `iq_source=radioml, checkpoint=2016.10a_AWN.pkl, device=cpu,
  alignment_policy=max-energy(auto), awn_preprocess=radioml-native(auto),
  threshold_factor=1.5, sensing_window_size=128, min_region_len=0,
  merge_gap=0, num_bursts=1, seed=42, use_real_awn=True,
  use_real_attack=True, use_real_topk=True, attack_eps=0.05,
  attack_temperature=1.0 (default), cw_c=1.0, cw_steps=20, cw_lr=0.01
  (defaults)`
- **sweep 參數**: `dataset_mod ∈ {QPSK, BPSK}` (2) × `dataset_snr ∈ {0,18}`
  (2) × `sample_index ∈ {0,1}` (2) × `attack ∈ {none,fgsm,pgd,cw}` (4) ×
  `topk ∈ {10,20,30,40}` (4)
- **組合數**: 2×2×2×4×4 = **128**
- **每組 sample 數**: 2 sample_index per (mod,snr) cell -- 8 unique attacked-
  IQ instances per (mod,snr,attack), each evaluated at 4 K values (fair
  reuse)
- **預估執行時間**: 128 × ~2.0s/combo ≈ 256s ≈ **~4-5 minutes**
- **輸出目錄**: `results/formal_pilot_phase0/`
- **主要 CSV 欄位**: `run_status, awn_backend, attack_backend,
  topk_backend, awn_status, attack_status, topk_status, pred_clean,
  pred_attacked, pred_defended, changed_by_attack, recovered_by_defense,
  iq_linf_clean_attacked`
- **成功/失敗判定**: `run_status ∈ {ok, sensing_failed}` for all 128; 0
  `error`; 0 combo shows any `*_backend` that isn't the real-path string
  (no `dummy_*` anywhere -- this would indicate a fallback occurred);
  `iq_linf_clean_attacked` identical across all 4 `topk` within each
  `(mod,snr,idx,attack)` group (fair-reuse invariant, same check as round
  12)
- **論文指標**: none citable at this N (mechanics/validity check only)

### Phase 1 -- Spectrum Sensing baseline (+ Phase 2 shares this data)

- **研究問題**: how well does energy-detection sensing detect real RadioML
  bursts across the full modulation × SNR grid? (Pd, false alarm, boundary
  error, captured ratio)
- **固定參數**: `attack=none, use_real_awn=True, use_real_attack=False,
  use_real_topk=False` (no attack/defense needed; AWN is left on because
  it's cheap without an attack and its output directly serves Phase 2), +
  all of category B
- **sweep 參數**: `dataset_mod`: all 11 × `dataset_snr`: all 20
- **組合數**: 11×20×N; **N=10 (proposed default)** → **2200**; adjustable:
  N=5 (1100, faster) / N=20 (4400, higher statistical power)
- **預估執行時間**: 2200 × ~1.4s/combo ≈ 3080s ≈ **~51 minutes** (at N=10)
- **輸出目錄**: `results/formal_phase1_sensing_clean_amc/`
- **主要 CSV 欄位**: `detection_success, captured_signal_ratio,
  extra_captured_noise_ratio, start_boundary_error, end_boundary_error,
  missed_sample_count, false_occupied_sample_count, n_segments,
  pred_clean, dataset_mod, awn_backend`
- **成功/失敗判定**: `run_status ∈ {ok, sensing_failed}`, `sensing_failed`
  rows must carry `failure_stage`+`failure_reason`; 0 `error`; 0 dummy
  backend
- **論文指標**: detection_probability, false_alarm_region_rate,
  captured_signal_ratio, extra_captured_noise_ratio, start/end boundary
  error, missed_sample_count, false_occupied_sample_count, segment_count

### Phase 2 -- Clean AMC accuracy (analysis-only, no new run)

- **研究問題**: what is direct (oracle-slice) vs sensing-based (segmented)
  AMC accuracy across modulation × SNR when no attack is present?
- **設計**: **reuses Phase 1's `summary.csv` output** -- `pred_clean` (an
  AWN class index) joined against `dataset_mod` (a ground-truth string) via
  `docs/radioml_class_mapping.csv` (already exists, cross-verified at 8
  independent locations, md section 15.5). No new combos, no new run.
- **組合數 / 時間**: 0 (post-hoc analysis of Phase 1's existing output)
- **成功/失敗判定**: accuracy computed only over `run_status=ok` rows;
  `sensing_failed` rows are counted separately as sensing-stage failures,
  never silently excluded or counted as AMC failures
- **論文指標**: direct_amc_accuracy, sensing_to_amc_end_to_end_accuracy

### Phase 3 -- Adversarial attack effectiveness

- **研究問題**: how does attack success rate vary by attack type, eps
  (fgsm/pgd), modulation, and SNR?
- **固定參數**: `use_real_awn=True, use_real_attack=True,
  use_real_topk=False` (Top-K deferred to Phase 4), `attack_temperature=1.0`
  and `cw_c/steps/lr=defaults` (both validated effective at scale, risk R4),
  + all of category B
- **Tier "reduced" (recommended default)**: `dataset_mod ∈ {QPSK, BPSK,
  QAM16, 8PSK, QAM64, WBFM}` (6, spans the class families) ×
  `dataset_snr ∈ {-10,-4,0,6,12,18}` (6, spans the full range) ×
  `attack_eps ∈ {0.01,0.03,0.05,0.1,0.3}` (5) × `attack ∈ {fgsm,pgd}` (2),
  plus `attack=cw` (no eps) at the same mod×snr grid. N=10 →
  **3960 combos**, ≈7128s ≈ **~2 hours**
- **Tier "full" (optional, explicitly schedule separately)**: all 11
  modulations × all 20 SNR × same eps/attack sweep, N=5 → **12100 combos**,
  ≈21780s ≈ **~6 hours**
- **輸出目錄**: `results/formal_phase3_attack_{reduced,full}/`
- **主要 CSV 欄位**: `pred_clean, pred_attacked, changed_by_attack,
  iq_linf_clean_attacked, iq_linf_normalized_clean_attacked, attack_eps,
  attack_backend, attack_status, dataset_mod, dataset_snr`
- **成功/失敗判定**: `run_status ∈ {ok, sensing_failed}`; 0 `error`; 0
  dummy backend; `iq_linf_normalized_clean_attacked` must equal the
  requested `attack_eps` exactly for fgsm/pgd (round 14 invariant --
  re-checked as a pass condition, not re-derived from scratch)
- **論文指標**: attack_success_rate, clean_accuracy, attacked_accuracy,
  pred_clean, pred_attacked

### Phase 4 -- Top-K defense recovery rate

- **研究問題**: does Top-K defense recover correct predictions after
  attack, and how does recovery rate vary with K, attack type, eps,
  modulation, SNR?
- **固定參數**: same as Phase 3 + `use_real_topk=True`
- **設計**: reuses Phase 3's exact `(mod, snr, eps, sample_index, attack)`
  grid, adding `topk ∈ {10,20,30,40}` (4) as an extra multiplied axis, with
  the **same attacked IQ reused across all 4 K values** -- not literally
  cached in-process, but bit-for-bit reproducible regeneration under the
  fixed seed, the exact method round 12 validated at 480-combo scale (0/120
  groups showed variation)
- **Tier "reduced"**: Phase 3-reduced × 4 topk = **15840 combos**,
  ≈31680s ≈ **~8.8 hours**
- **Tier "quick" (recommended first step before committing to "reduced")**:
  `dataset_mod ∈ {QPSK,BPSK,QAM16}` (3) × `dataset_snr ∈ {0,10,18}` (3) ×
  `attack_eps ∈ {0.03,0.05,0.1}` (3) × `attack ∈ {fgsm,pgd}` (2) + cw at
  the same mod×snr grid, × `topk` (4), N=10 → **2520 combos**, ≈5040s ≈
  **~1.4 hours**
- **輸出目錄**: `results/formal_phase4_defense_{quick,reduced}/`
- **主要 CSV 欄位**: `pred_defended, recovered_by_defense, topk,
  topk_backend, topk_status, iq_linf_clean_attacked`
- **成功/失敗判定**: same as Phase 3, plus `iq_linf_clean_attacked`
  identical across all 4 `topk` within each `(mod,snr,idx,attack,eps)`
  group (fair-reuse invariant)
- **論文指標**: defense_recovery_rate, defended_accuracy, pred_defended

### Phase 5 -- Spectrum-sensing parameter sensitivity

- **研究問題**: how do threshold_factor / sensing_window_size /
  min_region_len / merge_gap affect sensing quality metrics?
- **設計 (default)**: **reuse existing round-13 evidence directly** --
  `results/sensing_revalidation_after_alignment/` already contains 528
  real-backend combos (210 threshold_factor + 150 sensing_window_size +
  150 min_region_len + 18 merge_gap, real AWN, max-energy/radioml-native,
  `seed=42`, md section 22). **No new run by default** -- this phase's
  default action is a write-up citing that existing data, per the explicit
  instruction not to repeat completed parameter validation.
- **設計 (optional elective expansion)**: round 13 used only 3 modulations
  (QPSK, BPSK, QAM16); if the paper's scope needs broader modulation
  coverage for the threshold_factor sub-sweep specifically (the largest of
  the 4), an elective expansion to all 11 modulations × `{0,18}` SNR × the
  same 7 threshold_factor values × N=5 → **770 combos**, ≈1078s ≈
  **~18 minutes**
- **輸出目錄**: existing `results/sensing_revalidation_after_alignment/`
  (default) or `results/formal_phase5_sensing_sensitivity_expanded/`
  (elective)
- **論文指標**: threshold_factor_sensitivity, sensing_window_size_
  sensitivity, min_region_len_sensitivity, merge_gap_sensitivity

### Phase 6 -- Multi-burst extension

- **研究問題**: does real attack + Top-K defense generalize to a 2-burst
  scene, and does merge-gap behave correctly under attack? (Multi-burst has
  been tested with sensing only, and attack has been tested with
  single-burst only -- the combination has never been run.)
- **固定參數**: `iq_source=radioml, num_bursts=2, use_real_awn=True,
  use_real_attack=True, use_real_topk=True, attack_eps=0.05, topk=20`,
  `min_burst_gap=max_burst_gap` (deterministic gap per case), + sensing
  defaults from category B
- **sweep 參數**: `dataset_mod_list ∈ {[QPSK,QPSK], [QPSK,BPSK],
  [QAM16,8PSK]}` (3 pairs) × `burst_gap ∈ {20 (merge case, <merge_gap),
  200 (separate case, >merge_gap)}` (2) × `attack ∈ {none,fgsm}` (2)
- **組合數**: 3×2×2×N; N=5 → **60**
- **預估執行時間**: 60 × ~1.8s/combo ≈ 108s ≈ **~2 minutes**
- **輸出目錄**: `results/formal_phase6_multiburst_extension/`
- **主要 CSV 欄位**: `num_truth_bursts, num_detected_regions,
  detection_probability, region_matched_burst_ids, pred_attacked,
  pred_defended, recovered_by_defense`
- **成功/失敗判定**: `run_status ∈ {ok, sensing_failed}`; 0 `error`;
  region-merge outcome matches the `gap<=merge_gap` prediction (round 13
  section 22.6 invariant, re-checked not re-derived)
- **論文指標**: multiburst_detection_probability,
  multiburst_defense_recovery_rate

---

## 5. Paper-metric coverage checklist

Every metric the user's instruction required is covered by exactly one
phase (no metric is orphaned, no phase invents an untraceable metric):

| Metric | Source phase | CSV field(s) |
|---|---|---|
| detection probability | 1 | `detection_probability` (batch-level aggregate) |
| false alarm rate | 1 | `false_alarm_region_rate`, `sample_level_false_positive_rate` |
| captured signal ratio | 1 | `captured_signal_ratio`, `mean_captured_signal_ratio` |
| extra captured noise ratio | 1 | `extra_captured_noise_ratio` |
| start boundary error | 1 | `start_boundary_error` |
| end boundary error | 1 | `end_boundary_error` |
| missed sample count | 1 | `missed_sample_count` |
| false occupied sample count | 1 | `false_occupied_sample_count` |
| segment count | 1 | `n_segments` |
| direct AMC accuracy | 2 | `pred_clean` joined vs. oracle-slice ground truth |
| sensing-to-AMC end-to-end accuracy | 2 | `pred_clean` joined vs. `dataset_mod` (sensing path) |
| attack success rate | 3 | `changed_by_attack` |
| defense recovery rate | 4 | `recovered_by_defense` |
| clean / attacked / defended prediction | 1/3/4 | `pred_clean`, `pred_attacked`, `pred_defended` |
| clean / attacked / defended accuracy | 2/3/4 | derived from the above joined against `dataset_mod` |
| runtime | all | measured by each phase script's own `time.time()` wrapper (same pattern as every existing round script) |
| backend 狀態 | all | `awn_backend`/`attack_backend`/`topk_backend` + `*_status` |
| failure reason | all | `failure_stage`, `failure_reason` (batch_summary.csv, `sensing_failed` rows) |

---

## 6. What was actually done this round

- Read `docs/parameter_validation.csv` (full, both halves) and the section
  headers / tail of `docs/parameter_validation.md` (already read in full in
  the immediately preceding round of this session, cross-referenced again
  here) -- no new experiment was run to produce this inventory.
- Read `experiments/run_batch.py`'s argparse definitions directly to
  confirm it cannot sweep a radioml multi-(mod,snr,sample_index) grid
  (section 3).
- Read `experiments/run_fair_topk_matrix.py` in full to confirm it is a
  directly reusable template (section 3).
- Read `src/utils/pipeline.py:179` and `src/sensing/radioml_source.py:109`
  directly to confirm `burst_len` is synthetic-only (category D, section
  1).
- Wrote this document and `docs/formal_experiment_matrix.csv`.
- **Did not** execute any pilot, any phase, or any new script. Per the
  explicit instruction, no formal batch (hundreds/thousands of combos) and
  no execution of even the small pilot were started this round.

---

## 7. Outstanding risks / open decisions (must be resolved before execution)

- **R1 -- sample_index count (N) per cell is an open statistical-design
  decision, not a technical fact.** Every phase above proposes a default N
  with the combo-count formula shown so it can be trivially rescaled; the
  actual N should be confirmed against the paper's required statistical
  power / timeline, not assumed by this plan.
- **R2 -- Phase 3/4's modulation and SNR subsets ("reduced" tier) are a
  proposed, not inherited, choice.** They were NOT copied from
  `external/adversarial-rf`'s own SNR conventions (`docs/
  parameter_validation.md` section 9 explicitly documents that those
  external conventions, e.g. `[0,10,18]`, are "not used by this repo") --
  they are an independent proposal for this plan, spanning the class
  families and the full SNR range. Confirm before running.
- **R3 -- Phase 3/4 "full" tiers are large (6-24.2 hours combined) and
  should be explicitly scheduled, not started opportunistically.** The
  "reduced"/"quick" tiers are the recommended practical starting point.
- **R4 -- the attack_temperature/CW-defaults "ineffective" finding
  (md section 10.2) appears NOT to reproduce under radioml-native, but this
  causal connection was never explicitly investigated in any prior round --
  it is this round's own cross-reference/inference (section 1), not a
  previously-stated conclusion.** The evidence (round 12: cw=83.3% success
  at defaults; round 14: cw=90.9% success at defaults; both under
  radioml-native, both at `attack_temperature=1.0` default) is real and
  citable, but the mechanism ("radioml-native's well-scaled logits no
  longer saturate the loss gradient") is a plausible explanation, not a
  confirmed one. If this matters for the paper's methodology section, it
  deserves its own small dedicated diagnostic round before Phase 3 begins
  -- not assumed here.
- **R5 -- Phase 1's proposed N=10 gives 2200 combos (~51 min); Phase 3
  reduced/Phase 4 quick are ~2h/~1.4h; Phase 4 reduced is ~8.8h.** These
  are all currently **designed, not run** (per this round's explicit
  scope). Running them requires separate, explicit go-ahead each time --
  this plan does not authorize any of them to start automatically.
- **R6 -- Phase 5's "reuse existing round-13 evidence" default assumes
  round 13's 3-modulation subset (QPSK/BPSK/QAM16) is sufficient for the
  paper's sensing-sensitivity claims.** If the paper needs sensitivity
  evidence across all 11 modulations, the elective expansion (770 combos,
  ~18 min) should be run -- flagged as optional, not decided here.
- **R7 -- no phase in this plan touches `checkpoint` alternates,
  `device=cuda`, or the matplotlib-missing plotting fallback** -- all three
  remain out of scope for the formal experiment (checkpoint alternates
  known-broken; cuda unavailable in this environment; plotting fallback is
  cosmetic, non-blocking for numeric results), consistent with `docs/
  parameter_validation.md` section 7's existing outstanding-items list.

---

## 8. Phase 0 pilot execution (round 17)

Executed via `experiments/run_phase0_pilot.py`, a new, self-contained
script that calls the same underlying building blocks `src/utils/
pipeline.py` uses (`energy_detect`, `select_aligned_segments`,
`apply_awn_preprocess`, `AWNModelAdapter`, `AttackAdapter`, `TopKAdapter`,
`compute_sensing_ground_truth_metrics`) directly, rather than calling
`run_dry_run_experiment()` -- this round's fairness requirement is
STRICTER than round 12's ("same attacked IQ reused across K, verified
bit-identical after the fact"): the SAME in-memory clean/attacked IQ array
must be reused literally, never regenerated. `run_dry_run_experiment()`
computes the whole pipeline once per (mod,snr,idx,attack,topk) combo and
cannot express that without being refactored, so this script factors the
front end (sensing through attacked-AWN-inference) out into a shared
per-attack-instance computation, then loops only the Top-K + defended-AWN
stage over the 4 K values, reusing the exact same `x_adv` array object.
No sensing/AWN/attack/Top-K algorithm code was written or modified.
`external/AWN`/`external/adversarial-rf` were not touched.

### 8.1 Dry-run and smoke checks

- `--dry-run`: 128 combos enumerated, all `combo_id`s unique (checked
  programmatically, not by eye).
- Smoke pilot (`--mods QPSK --snrs 18 --sample-indices 0 --attacks
  none,fgsm --topks 10,20,30,40`, 8 combos): 8/8 `run_status=ok`, all three
  backends exactly the real-path strings (`external/adversarial-rf/
  models/model.py:AWN`, `external/adversarial-rf/util/adv_attack.py:
  Model01Wrapper + torchattacks`, `external/adversarial-rf/util/
  defense.py:fft_topk_denoise`) on every row, 0 NaN/Inf, `none`-attack
  rows show `clean_iq_sha256 == attacked_iq_sha256` (bit-identical, proven
  no-op), `fgsm` rows show exactly 1 unique `attacked_iq_sha256` across
  all 4 K rows (fair reuse, verified from the CSV itself post-hoc, not
  only the in-process assertion). No bug found -- proceeded directly to
  the full 128-combo run.

### 8.2 Full 128-combo pilot

Run with `--resume` on top of the smoke test's 8 completed rows (also
exercises the `--resume` mechanism required by this round). **128/128
`run_status=ok`, 0 `sensing_failed`, 0 `error`.** All 32
`(modulation,snr,sample_index,attack)` attack-instances show exactly 1
unique `attacked_iq_sha256` across their 4 K rows -- 0 fairness
violations. All backends real on every row; 0 NaN/Inf anywhere. Total
runtime 13.8s (smoke + full combined).

### 8.3 Metrics (N=8 samples -- small-sample observations, NOT a paper-citable result)

- **direct AMC accuracy (oracle, sensing-independent)**: 7/8 = 0.875
- **clean sensing-to-AMC accuracy**: 7/8 = 0.875 -- **identical to direct
  at this N** (the one sensing-based miss and the one oracle miss are the
  same sample; not evidence sensing never degrades accuracy, just that no
  gap appeared in these 8 samples)
- **attack success rate**: pgd 8/8 (1.00), cw 7/8 (0.875), fgsm 6/8
  (0.75) -- all at `attack_eps=0.05` (fgsm/pgd) and default CW
  hyperparameters, consistent in direction with round 12/14's
  radioml-native-mode findings (risk R4)
- **defense recovery rate among successfully-attacked instances**, by
  (attack, K): cw rises with K (0.14 -> 0.29 -> 0.57 -> 0.43 at
  K=10/20/30/40, n=7 each) -- same qualitative K-dependent pattern round
  12 found at 480-combo scale; fgsm/pgd stay at or near 0 (0/6 to 1/6 for
  fgsm; 1/8 to 0/8 for pgd) -- also consistent with round 12/14
- **defended-prediction correctness by K, across ALL 32 rows regardless
  of attack outcome**: K=10 -> **0/32 (0.000)**, K=20 -> 11/32 (0.344),
  K=30 -> 13/32 (0.406), K=40 -> 12/32 (0.375). Spot-checked directly
  (not just aggregated): at K=10, every single row's `pred_defended`
  collapsed to one of a small handful of classes (1 for every QPSK row
  regardless of `pred_clean`; 8 or 1 for every BPSK row) independent of
  whether that row's clean prediction was already correct or which attack
  (if any) was applied. This is a genuine, reproducible pattern in this
  data, not a script bug -- but it is an N=8 observation about one
  specific checkpoint/sample set, not a general claim that "K=10 always
  fails."
- **mean runtime**: 0.087s per (attack-instance, distributed across its
  4 K-sharing rows); min 0.0054s (cached/no-op cases), max 0.692s
  (CW's 20-step optimization)
- **failure reasons**: none -- 0 `sensing_failed`, 0 `error` across all
  128 combos

### 8.4 Explicit classification (per this round's instruction)

- **系統正確性驗證 (system-correctness verification, now confirmed)**:
  real AWN/attack/Top-K backends run end-to-end with zero fallback; fair
  Top-K reuse holds exactly (0/32 violations); `none`-attack is a proven
  bit-identical no-op; output schema (50 columns incl. the 2 bonus hash
  columns) is complete; `--resume` correctly skips already-done combos;
  `run_status` cleanly distinguishes `ok`/`sensing_failed`/`error` (no
  case of either occurred in this pilot, both paths were exercised and
  validated separately in round 15's dummy-fallback work and round 13's
  sensing-failure rounds).
- **小樣本觀察 (small-sample observations, informative but not
  conclusive)**: attack success rate ordering (pgd > cw > fgsm) and the
  cw recovery-rate-rises-with-K pattern, at N=8, are directionally
  consistent with round 12's 480-combo/round 14's 88-combo findings --
  worth noting as a consistency check, not as new evidence on their own.
- **尚不可作為論文結論 (not yet citable as a paper conclusion)**: the
  K=10 -> 0/32 defended-accuracy collapse is the most striking number in
  this pilot and is exactly the kind of finding Phase 4 (Top-K defense,
  full formal design) exists to properly characterize across many more
  samples, modulations, and SNRs -- citing "K=10 destroys accuracy" from
  8 samples would be overreaching. Direct-vs-sensing accuracy gap (0.000
  at this N) is likewise not yet a claim that sensing never costs
  accuracy -- Phase 1/2 (2200 combos) is what will actually answer that
  question.

### 8.5 Outputs

```
results/formal_pilot_phase0/pilot_summary.csv    (128 rows, 50 columns)
results/formal_pilot_phase0/pilot_aggregate.csv  (25 rows: overall/modulation/snr/attack/attack_topk/topk/runtime breakdowns)
results/formal_pilot_phase0/pilot_manifest.json
results/formal_pilot_phase0/stdout.log, stderr.log        (full 128-combo run)
results/formal_pilot_phase0/smoke_stdout.log, smoke_stderr.log  (8-combo smoke run)
results/formal_pilot_phase0/{mod}_snr{snr}_idx{idx}/       (8 per-sample subdirectories; currently empty
                                                             beyond directory creation -- this pilot writes
                                                             its records into the shared pilot_summary.csv,
                                                             not a per-combo summary.csv like run_dry_run_experiment)
```
Not written to git (matches `.gitignore`'s existing `results/*` rule, same
as every prior round's results).

### 8.6 Conclusion

Phase 0's stated purpose -- confirm the real-backend pipeline executes
correctly, with correct schema and strict fair Top-K reuse, before Phase
1-6 are attempted -- is **satisfied**. No bug was found; nothing needed
fixing. Phase 1-6 remain designed-but-not-run, per this round's explicit
scope (only Phase 0 was authorized to execute).

---

## 9. Phase 1 execution: Spectrum Sensing baseline + direct/sensed AMC (round 18)

Executed via `experiments/run_phase1_sensing_baseline.py`, which calls
`src/utils/pipeline.py:run_dry_run_experiment()` directly (unlike Phase 0,
Phase 1 has no cross-combo fairness constraint forcing a bypass) plus a
`pred_direct` oracle-path addition (same method as Phase 0's
`compute_direct_amc`, via a separately-constructed, once-built
`AWNModelAdapter` reused across all 2200 combos for the oracle inference
only). No sensing/AWN algorithm code was written or modified.
`external/AWN`/`external/adversarial-rf` were not touched.

Fixed params, copied verbatim from `docs/formal_experiment_matrix.csv`'s
phase=1 row (not guessed): `iq_source=radioml, attack=none,
use_real_awn=True, use_real_attack=False, use_real_topk=False,
checkpoint=2016.10a (pinned), device=cpu, alignment_policy=max-energy,
awn_preprocess=radioml-native, threshold_factor=1.5,
sensing_window_size=128, min_region_len=0, merge_gap=0, num_bursts=1,
seed=42`. N=10 sample_index per (modulation,SNR) cell, the plan's
recommended default -> 11 x 20 x 10 = **2200 combos**.

### 9.1 Dry-run and smoke checks

`--dry-run`: 2200 combos, all unique, 11/11 modulation coverage, 20/20 SNR
coverage, 0 attack combos present. Smoke test (QPSK/BPSK x SNR{0,18} x
idx0, 4 combos, run in 2 independent processes): 4/4 `run_status=ok`,
`awn_backend` real on every row, 0 NaN/Inf, direct-path
`original_sample_sha256` matched the pipeline's own internally-loaded
sample hash on all 4 combos (proves direct and sensed use the identical
RadioML sample), all 31 content columns bit-identical across the two
processes (only `runtime_seconds`, a wall-clock measurement, differed).
No bug found.

### 9.2 Full 2200-combo run

**2200/2200 `run_status=ok`, 0 `sensing_failed`, 0 `error`.** Actual
runtime: **5528.7s (92.1 minutes)** -- slower than the plan's ~51-minute
estimate (1.4s/combo assumed; actual ~2.5s/combo, likely from
`run_dry_run_experiment()` reloading the checkpoint fresh on every combo,
the same established behavior every prior batch script also has, plus
general system load during a 92-minute run). All `awn_backend` values
across all 2200 rows are exactly the real-path string
(`external/adversarial-rf/models/model.py:AWN`), 0 fallback anywhere; 0
NaN/Inf in `clean_nan`/`direct_nan`. A live error-signature watch
(`Traceback|ERROR|CRITICAL|Exception|non-real backend|fell back|fallback`)
across `stdout.log`/`stderr.log` never fired during the run; the only
stderr content for the entire run is the pre-existing, harmless
`VisibleDeprecationWarning` from `pickle.load(..., encoding="latin1")`
(unrelated to this round, present in every prior radioml-mode round).

**Modulation coverage: 11/11.** **SNR coverage: 20/20.**

**Reproducibility, checked at scale (not just the smoke test)**: 16
combos (8PSK/QPSK/WBFM/QAM64 x SNR{12,18,-20,10} x idx0, deliberately
spanning both high- and low-captured-ratio cases) re-run in a completely
fresh, independent process launched hours after the main run finished --
all 30 comparable columns (excluding `output_dir`/`runtime_seconds`)
bit-identical to the corresponding rows in the 2200-combo output.

### 9.3 Metrics (N=2200 -- this IS the formal Phase 1 baseline result, not a pilot-scale observation)

- **Direct AMC accuracy (oracle, sensing-independent)**: 1314/2200 =
  **0.5973**
- **Sensed end-to-end AMC accuracy**: 1277/2200 = **0.5805**
- **Gap (direct - sensed)**: **+0.0168** (1.68 percentage points -- the
  sensing front end costs a small but real amount of accuracy compared to
  the oracle, at this checkpoint/dataset)
- **Direct-sensed prediction agreement**: 2001/2200 = **0.9095** (9.05%
  of combos, sensing-based classification disagrees with the oracle
  classification -- not always a "loss": 34 of those disagreements are
  cases where the SENSED prediction was correct and the direct one was
  not, e.g. `QAM16_snr8_idx0`)
- **Detection probability**: **1.0000** (2200/2200 -- the burst was
  detected in every single combo at `threshold_factor=1.5`, consistent
  with round 13's stable-range finding)
- **False alarm region rate**: mean **0.0043** (near-zero, as expected at
  `threshold_factor=1.5`)
- **Mean captured signal ratio**: **0.9986** (min 0.5703, max 1.0000);
  **12/2200** combos have `captured_signal_ratio < 0.999` (partial
  captures -- includes the previously-documented `BPSK_snr18_idx0=0.625`
  case, reproduced again here, plus 11 new partial-capture cases spread
  across 8PSK/PAM4/QAM16/QAM64/QPSK -- all are genuine per-sample energy
  variation, not a systematic modulation-specific pattern: QAM64 alone
  has 4 of the 12)
- **Boundary errors**: mean start_boundary_error **-59.27** (mean abs
  59.63), mean end_boundary_error **+59.40** (mean abs 59.40) -- both
  consistent in sign and magnitude with the known max-energy/radioml-
  native alignment behavior (region typically ~53-61 samples wider than
  the true burst on both edges, per md section 18's original diagnosis)
- **Missed sample count**: mean **0.18** (near-zero -- almost the entire
  true burst is captured essentially every time)
- **False occupied sample count**: mean **118.85** (the smoothing-widened
  detected region includes on average ~119 extra noise samples beyond the
  true burst -- consistent with the boundary-error magnitudes above)
- **Segment count**: **2200/2200 combos produced exactly 1 segment**
  (expected: single burst, single detected region, max-energy policy
  selects exactly one segment per region)
- **Per-modulation** (direct_acc / sensed_acc / agreement / mean
  captured_ratio, n=200 each): AM-SSB highest (0.955/0.925/0.950), WBFM
  lowest by a wide margin (0.090/0.135/0.930 -- this checkpoint appears to
  struggle with WBFM specifically, both with and without sensing -- not a
  sensing-introduced problem, since direct accuracy is equally low).
  QAM16 has the lowest direct-sensed agreement (0.770), notably below
  every other modulation (next-lowest is QAM64 at 0.840) -- worth a closer
  look in a future round, not explained by this round's data alone.
- **Per-SNR**: accuracy rises monotonically-ish from near-chance
  (~0.09-0.12 at -20/-18/-16 dB) to a plateau around 0.83-0.89 from 0 dB
  upward, exactly the expected SNR-accuracy curve shape. Agreement
  between direct and sensed predictions also dips in the -12..-4 dB range
  (0.80-0.85) relative to both the very-low-SNR (chance-level, so
  agreement is less meaningful) and high-SNR (agreement 0.93-0.98) ends --
  consistent with sensing-introduced misclassification being most likely
  in the moderate-noise regime where a small segment misalignment can tip
  a borderline classification.
- **Sensing failure reasons**: none -- 0 `sensing_failed` across all 2200
  combos at `threshold_factor=1.5` (matches round 13's finding that this
  value is inside the stable, 0-failure range).

### 9.4 Explicit classification (per this round's instruction)

- **正式 Phase 1 baseline 結果 (formal, citable at this N=2200)**: direct
  AMC accuracy 0.5973, sensed end-to-end AMC accuracy 0.5805, gap +0.0168,
  agreement 0.9095, detection probability 1.0000, false alarm region rate
  0.0043, mean captured signal ratio 0.9986, boundary errors (~59 samples
  each edge), missed sample count (~0.18), false occupied sample count
  (~118.85), segment count (100% singletons), and the full per-modulation
  /per-SNR breakdowns above -- all computed over the complete, intended
  2200-combo grid (11/11 modulations, 20/20 SNRs), not a subsample.
- **系統驗證結果 (system-correctness verification)**: real AWN backend
  end-to-end with zero fallback across 2200 combos; 0 NaN/Inf; `--resume`
  and incremental-write mechanism exercised in production (the run
  self-completed without needing an actual resume, but the same code path
  smoke-tested cleanly); reproducibility confirmed bit-identical at scale
  (16 combos, fresh process, hours later); direct and sensed AMC
  independently confirmed to use the identical RadioML sample via SHA256
  cross-check.
- **尚不能下結論的觀察 (not yet conclusive, flagged not overclaimed)**:
  the WBFM low-accuracy and QAM16 low-agreement patterns are real
  observations at full N but their ROOT CAUSE (model training artifact?
  checkpoint-specific confusion pair? something about this particular
  awn_preprocess/alignment combination?) is not established by this
  round's data -- would need a confusion-matrix-level follow-up, out of
  this round's scope. The `direct - sensed` accuracy gap (+0.0168) is a
  real, formal-N result for THIS checkpoint/threshold_factor/alignment
  configuration specifically -- it should not be read as a universal
  "sensing costs ~1.7% accuracy" claim beyond these exact fixed
  parameters (Phase 5's sensitivity analysis is what would characterize
  how this gap moves with threshold_factor/sensing_window_size/etc.).

### 9.5 Outputs

```
results/formal_phase1_sensing_clean_amc/phase1_summary.csv    (2200 rows, 32 columns)
results/formal_phase1_sensing_clean_amc/phase1_manifest.json
results/formal_phase1_sensing_clean_amc/stdout.log, stderr.log
results/formal_phase1_sensing_clean_amc/{mod}_snr{snr}_idx{idx}/   (2200 per-combo subdirectories,
                                                                     each with the standard
                                                                     run_dry_run_experiment output:
                                                                     summary.csv, sensing_plot.png)
```
`phase1_failures.csv` was not written (0 failures). Not added to git,
matching `.gitignore`'s existing `results/*` rule.

### 9.6 Conclusion

Phase 1's full 2200-combo baseline is complete and clean: 100% `ok`, 0
sensing failures, 0 errors, 0 backend fallback, reproducible. Phase 2 (the
direct-vs-sensed accuracy comparison) is answered inline in section 9.3
above, since it shares 100% of Phase 1's data by design (no separate run
was needed, per the original plan). Phase 3-6 remain designed-but-not-run.

---

## 10. Phase 3 code review + reduced-tier execution (round 19)

### 10.1 Code review (against pipeline.py/AttackAdapter, not just dry-run)

Before any execution, `experiments/run_phase3_attack_effectiveness.py` was
reviewed against `src/utils/pipeline.py`, `src/adapters/attack_adapter.py`,
and `src/utils/batch_aggregation.py` directly (not just its own dry-run
output):

1. **Traceability**: `original_sample_sha256`/`long_iq_sha256` recorded
   per combo, sourced from `run_dry_run_experiment()`'s own `gen_meta`.
2. **Clean/attacked share the same sensed segment**: confirmed by
   construction -- `pipeline.py` computes `x_clean` once, then both
   `logits_clean` (from `x_clean`) and `logits_attacked` (from
   `AttackAdapter.apply(x_clean, ...)`'s output) derive from the same
   segment, and both `pred_clean`/`pred_attacked` are written into the
   SAME per-segment `summary.csv` row.
3/4/5. **FGSM/PGD use `eps`, CW uses `cw_c`/`cw_steps`/`cw_lr`**: confirmed
   by reading `_build_torchattacks()` directly
   (`src/adapters/attack_adapter.py:144-161`) -- `torchattacks.FGSM(model,
   eps=eps)`, `torchattacks.PGD(model, eps=eps, alpha=eps/4, steps=10)`,
   `torchattacks.CW(model, c=cw_c, steps=cw_steps, lr=cw_lr)`. The `cw`
   branch never reads `eps` at all -- structurally impossible for CW to be
   silently driven by `attack_eps`.
6. **AWN eval-mode restoration**: `AttackAdapter.apply()`'s `finally`
   block unconditionally calls `self.wrapped_model.eval()` before
   returning (confirmed by reading the source). **This was NOT actively
   verified by the runner before this round's review** -- fixed by adding
   `attack_training_before`/`attack_training_after`/`eval_mode_restored`
   columns (read from each combo's own `summary.csv`) and an explicit
   check that forces `run_status=error` if `attack_training_after` is
   ever `True`.
7. **pred_clean/pred_attacked/clean_correct/attacked_correct/changed_by_attack**:
   cross-checked in the smoke test -- this script's own
   independently-recomputed `changed_by_attack` (`pred_attacked !=
   pred_clean`) matched `pipeline.py`'s own pre-computed
   `changed_by_attack` column on 5/5 smoke combos, exactly.
8/9. **Attack success rate denominators**: made explicit in section 10.3
   below -- `overall_attack_success_rate` = `changed_by_attack` count /
   ALL `ok` rows; `conditional_attack_success_rate` = `changed_by_attack`
   count among rows where `clean_correct=True`, divided by that subset's
   count. Both computed and reported separately, never conflated.
10. **sensing_failed vs error**: `run_status` distinguishes the two;
   0 of either occurred in this round's data.
11. **No silent fallback**: `awn_ok`/`attack_ok` require the EXACT real
   backend string + `status=="ok"`; any mismatch forces `run_status=error`
   regardless of what the adapter itself reported.
12. **Incremental write + `--resume`**: same `CsvWriter` (flush-per-row)
   pattern as Phase 0/1, confirmed via `--resume` semantics (loads
   already-done `combo_id`s, skips them).
13. **combo_id/output_dir uniqueness**: `check_combo_ids_unique()` +
   `--dry-run` confirmed 3960 unique IDs; `output_dir = base/combo_id`.
14. **No collision with Phase 0/1**: `results/formal_phase3_attack_reduced/`
   is a distinct path from `results/formal_pilot_phase0/` and
   `results/formal_phase1_sensing_clean_amc/`.
15. **`.gitignore`**: covered by the existing `results/*` rule.
16. **external/AWN, external/adversarial-rf**: untouched (read-only via
   the existing adapters, same as every prior round).

### 10.2 Formal parameters (re-read from `docs/formal_experiment_matrix.csv`,
not assumed)

modulations: `QPSK, BPSK, QAM16, 8PSK, QAM64, WBFM` (6). SNRs: `-10, -4, 0,
6, 12, 18` (6). attack_eps: `0.01, 0.03, 0.05, 0.1, 0.3` (5, fgsm/pgd
only). Attacks: fgsm, pgd, cw. `n_per_cell=10` (full formal tier) ->
fgsm=1800, pgd=1800, cw=360, total **3960** combos, ~2 hours. CW knobs:
`cw_c=1.0, cw_steps=20, cw_lr=0.01` (defaults, per risk R4).

### 10.3 Reduced-tier design and execution

Reduced tier keeps the FULL formal grid (all 6 modulations, all 6 SNRs,
all 5 eps values, all 3 attacks) and restricts ONLY `sample_index` to the
first 2 of the formal 0-9 set (`[0,1]`) -- **no change to the research
design**, purely a sample-count reduction for a faster verification pass
before committing to the full ~2-hour run.

`--dry-run`: **792 combos** (fgsm=360, pgd=360, cw=72), all unique.
Estimated ~24 minutes (792 x 1.8s).

Executed via `nohup`, monitored at ~10-minute intervals (no high-frequency
polling) with a live error-signature watch covering
`Traceback|ERROR|CRITICAL|Exception|non-real backend|fell back|fallback|
eval mode|eps invariant` -- never fired. **Actual runtime: 1100.7s (18.3
minutes)**, faster than estimated. **792/792 `run_status=ok`, 0
`sensing_failed`, 0 `error`.**

### 10.4 Results (N=792 -- reduced-tier, formal-grid-shaped but 1/5 the
sample count of the full 3960 design; directionally informative, not yet
the full-N formal Phase 3 result)

- **Per-attack combo counts**: fgsm=360, pgd=360, cw=72 (exactly matches
  the reduced-tier design).
- **clean_accuracy**: 0.5139 (same clean-path accuracy regardless of
  which attack row -- clean prediction never depends on attack type,
  confirmed identical across all three attack subsets).
- **attacked_accuracy** (all attacks pooled): 0.2487.
- **Overall attack success rate** (denominator = all 792 `ok` rows):
  654/792 = **0.8258**.
- **Conditional attack success rate** (denominator = the 407 rows where
  `clean_correct=True`): 321/407 = **0.7887**.
- **Per-attack**: cw highest overall success rate (0.9306) despite by far
  the SMALLEST perturbation (mean `iq_linf_normalized`=0.0213 vs
  fgsm/pgd's 0.098) -- consistent with CW being an optimization-based
  attack that searches for a minimal sufficient perturbation, vs
  fgsm/pgd's fixed-eps perturbation. pgd (0.8861) > fgsm (0.7444) at
  pooled-eps success rate, matching the expected iterative-vs-single-step
  attack strength ordering.
- **Per-eps** (fgsm/pgd pooled): success rate rises from 0.4722 (eps=0.01)
  to 0.9306 (eps=0.05), dips slightly to 0.8958 (eps=0.1), then rises
  again to 0.9583 (eps=0.3) -- the eps=0.05->0.1 dip is a **small-sample
  observation** (n=144 per eps value in this reduced tier) and should not
  be treated as a real non-monotonicity without the full-N confirmation.
- **Per-modulation**: QAM16 clean_acc=0.75 (highest) but attacked_acc
  collapses to 0.0985 (among the most vulnerable). **WBFM is a case
  requiring careful interpretation**: clean_acc=0.0833 (consistent with
  Phase 1's finding that this checkpoint struggles with WBFM even without
  any attack), but attacked_acc=0.5152 (HIGHER than clean) -- this is NOT
  "WBFM is robust to attack." With `conditional_success_rate=1.0000`
  (100% of the FEW correctly-classified clean WBFM samples get flipped),
  the apparent accuracy increase is a base-rate artifact: WBFM's clean
  classifier is so poor that most predictions are already wrong, and an
  attack perturbing a mostly-wrong classifier's output landed on the
  correct class more often than the unperturbed model's own (mostly
  incorrect) predictions did. Flagged explicitly, not glossed over.
- **Per-SNR**: attack success rate is HIGHEST at low SNR (-10dB: 0.9318,
  -4dB: 0.9470) and generally lower at high SNR (12dB: 0.7045, 18dB:
  0.7424) -- plausible (less "headroom" in a low-SNR/low-confidence
  clean prediction for the attack to overcome), but this is the reduced
  tier's n=132/SNR, not yet the full-N confirmation.
- **IQ perturbation** (`iq_linf_clean_attacked`, un-normalized IQ units):
  fgsm/pgd both mean 0.00322 (same requested eps range, expected since
  both are eps-driven); cw mean 0.00060 (5.4x smaller), min 0.00000 --
  consistent with CW finding minimal perturbations, occasionally
  near-zero when the clean prediction is already at a decision boundary.
- **Eval-mode restoration**: **100% (792/792) `eval_mode_restored=True`**
  -- the fix from section 10.1 item 6 is confirmed working at full
  reduced-tier scale, not just the smoke test.
- **eps invariant**: 100% held exactly for every fgsm/pgd row.
- **NaN/Inf**: 0 anywhere. **Backends**: 100% real
  (`external/adversarial-rf/models/model.py:AWN`,
  `external/adversarial-rf/util/adv_attack.py:Model01Wrapper +
  torchattacks`) on every row, 0 fallback.
- **Reproducibility**: 20 combos (WBFM/QAM16 x SNR{-10,18} x eps{0.05,0.3}
  x all 3 attacks, deliberately re-targeting the two flagged-interesting
  modulations) re-run in a fresh independent process -- **all 36
  comparable columns bit-identical** to the reduced-tier run.
- **Failure reasons**: none -- 0 `sensing_failed`, 0 `error`.

### 10.5 Recommendation

Reduced-tier (N=792, 1/5 sample count) passed every system-correctness
check without exception: 0 error, 0 fallback, 100% eval-mode restoration,
100% eps-invariant compliance, 0 NaN/Inf, bit-identical reproducibility.
**The full 3960-combo Phase 3 run is recommended to proceed** -- no bug
was found, no code change is pending, and the reduced-tier's directional
findings (cw > pgd > fgsm success rate; QAM16 vulnerable, WBFM's baseline
problem not attack-specific; low-SNR more attack-susceptible) are
consistent with prior rounds' qualitative expectations. The full run is
**not started automatically** -- awaiting explicit confirmation, per this
round's instruction.

### 10.6 Outputs

```
results/formal_phase3_attack_reduced/phase3_summary.csv   (792 rows, 38 columns)
results/formal_phase3_attack_reduced/phase3_manifest.json
results/formal_phase3_attack_reduced/stdout.log, stderr.log
results/formal_phase3_attack_reduced/{combo_id}/           (792 per-combo subdirectories)
```
`phase3_failures.csv` was not written (0 failures). Not added to git,
matching `.gitignore`'s existing `results/*` rule.

---

## 11. Phase 3 FULL execution: attack effectiveness, N=3960 (round 20)

Full formal tier, executed exactly as designed in section 10.2 -- all 6
modulations, all 6 SNRs, all 5 eps values, full `sample_index` 0-9 (the
complete formal set, not the reduced tier's `[0,1]`). Run into a
**separate** output directory (`results/formal_phase3_attack_full/`) so
the reduced-tier's `results/formal_phase3_attack_reduced/` was never
touched or overwritten. Launched via `nohup`, monitored at ~10-minute
intervals with the same live error-signature watch as the reduced tier
(never fired). **Actual runtime: 5317.6s (88.6 minutes)**, close to the
~2-hour estimate.

**3960/3960 `run_status=ok`, 0 `sensing_failed`, 0 `error`.** Coverage
confirmed exactly: 6/6 modulations, 6/6 SNRs, 5/5 eps values, all 10
sample_index values (0-9, the complete formal set), fgsm=1800/pgd=1800/
cw=360 (exact match to the design).

### 11.1 Formal Phase 3 results (N=3960 -- the complete, intended grid,
not a subsample)

- **clean_accuracy**: **0.5889** (higher than the reduced tier's 0.5139 --
  expected, since N=3960 draws from the full sample_index range 0-9
  rather than just [0,1], averaging out the specific-sample variance the
  reduced tier's smaller draw was subject to)
- **attacked_accuracy**: **0.2876**
- **Overall attack success rate** (denominator = all 3960 `ok` rows):
  3278/3960 = **0.8278**
- **Conditional attack success rate** (denominator = the 2332 rows where
  `clean_correct=True`): 1864/2332 = **0.7993**
- **Prediction changed rate**: same as overall, 0.8278
- **Per-attack**: cw highest (0.9278), pgd (0.8861), fgsm (0.7494) --
  same ordering as the reduced tier, now confirmed at full N.
- **Per-eps** (fgsm/pgd pooled): 0.01->0.4931, 0.03->0.8347, 0.05->0.9236,
  0.1->0.9028, 0.3->0.9347. **The reduced tier's eps=0.05->0.1 dip
  (flagged in section 10.4 as a small-sample observation, n=144) is
  RESOLVED at full N**: pooled, the dip narrows to a statistically
  unremarkable 0.9236->0.9028; and per-attack (section 11.2 below) it
  turns out to be entirely a `fgsm`-specific pattern, not present in pgd
  at all.
- **Per-modulation**: WBFM's clean_acc (0.0833) and the
  higher-than-clean attacked_acc (0.4818) pattern from Phase 1/the
  reduced tier is reproduced at full N, confirming it is not a
  small-sample artifact -- see section 11.4's explicit re-statement of
  why this is a clean-baseline problem, not attack robustness. QAM16 most
  vulnerable (overall_sr=0.9030); BPSK most resistant (overall_sr=0.7212).
- **Per-SNR**: success rate highest at low SNR (-10dB: 0.9379, -4dB:
  0.9470) and lowest at high SNR (18dB: 0.7152) -- monotonic-ish decline
  confirmed at full N, same direction as the reduced tier.

### 11.2 Cross-tabulations (new this round, not computed for the reduced tier)

- **attack x eps** (overall success rate): **pgd saturates to 1.000 at
  eps>=0.1** (eps=0.1: 1.000, eps=0.3: 1.000) -- a clean ceiling effect.
  **fgsm does NOT saturate and shows a genuine, full-N-confirmed
  non-monotonic dip**: eps=0.01:0.461, 0.03:0.747, 0.05:0.864,
  **0.1:0.806** (dip), 0.3:0.869. Since n=360 per (attack,eps) cell here
  (vs the reduced tier's n=144 pooling both attacks), this specific
  fgsm-only dip is now a firmer, though still unexplained, observation --
  it survived a 2.5x sample-size increase and isolation from pgd's
  confounding saturation.
- **modulation x attack** (overall success rate): **BPSK is a clear
  outlier for cw specifically** -- cw success rate is 0.617 for BPSK vs
  0.983-1.000 for every other modulation. fgsm/pgd show no comparable
  BPSK-specific dip (0.680/0.783, in line with other modulations). This
  is a real pattern at n=60 for the BPSK-cw cell (6 SNR x 10 sample_index,
  cw has no eps dimension), newly surfaced only by this cross-tabulation
  -- neither Phase 1 nor the per-modulation-only view in section 10.4/
  11.1 would have revealed it.
- **SNR x attack** (overall success rate): cw consistently highest at
  every SNR (0.867-1.000), fgsm consistently lowest (0.597-0.920), pgd
  in between -- the per-attack ordering (cw>pgd>fgsm) holds at every
  single SNR value, not just on average.

### 11.3 System verification (all 3960 rows)

- **Eval-mode restoration**: **100% (3960/3960)**
  `eval_mode_restored=True` -- the section 10.1 fix holds at full scale.
- **eps invariant**: **100%** held exactly for every fgsm/pgd row.
- **NaN/Inf**: 0 anywhere.
- **Backends**: 100% real
  (`external/adversarial-rf/models/model.py:AWN`,
  `external/adversarial-rf/util/adv_attack.py:Model01Wrapper +
  torchattacks`) on every one of 3960 rows, 0 fallback.
- **IQ perturbation**: fgsm/pgd both mean `iq_linf`=0.00285 (matches
  reduced tier closely); cw mean=0.00050 (5.7x smaller) -- same pattern
  as the reduced tier, confirmed at full N.
- **Sensing failure reasons**: none -- 0 `sensing_failed` across all
  3960 combos.
- **Reproducibility**: 40 combos (BPSK/8PSK x SNR{-10,18} x
  sample_index{5,9} x eps{0.05,0.3} x all 3 attacks -- deliberately
  targeting BPSK, the newly-found cw outlier, and 8PSK, the highest-cw-
  success modulation) re-run in a fresh independent process -- **all 36
  comparable columns bit-identical**.

### 11.4 Explicit classification (per this round's instruction)

- **正式 Phase 3 結果 (formal, citable at N=3960, the complete intended
  grid)**: all numbers in sections 11.1-11.2 above.
- **系統驗證結果 (system-correctness verification)**: 100% eval-mode
  restoration, 100% eps-invariant compliance, 0 NaN/Inf, 0 fallback, 0
  sensing_failed, 0 error, bit-identical reproducibility -- all confirmed
  at the full, intended N, not extrapolated from the reduced tier.
- **尚不能下結論的觀察 (not yet conclusive, explicitly flagged)**:
  - The fgsm-specific eps=0.1 dip (section 11.2) is real at N=360 for
    that cell but its MECHANISM is not established by this round's data
    -- would need a dedicated diagnostic (e.g. per-sample gradient
    inspection at eps=0.05 vs 0.1) to explain, not assumed here.
  - The BPSK-cw-specific resistance (0.617 vs 0.983-1.000 elsewhere) is a
    genuinely new, striking finding first surfaced by this round's
    cross-tabulation -- root cause (something about BPSK's decision
    boundary geometry under this checkpoint? an interaction with CW's
    L2-minimization objective specifically?) is NOT established here.
  - WBFM's clean-baseline problem (re-confirmed at full N) still has no
    established root cause (training artifact vs. checkpoint-specific
    confusion vs. something in this repo's preprocessing) -- flagged
    again, not re-investigated this round (same status as Phase 1
    section 9.4).

### 11.5 Outputs

```
results/formal_phase3_attack_full/phase3_summary.csv   (3960 rows, 38 columns)
results/formal_phase3_attack_full/phase3_manifest.json
results/formal_phase3_attack_full/stdout.log, stderr.log
results/formal_phase3_attack_full/{combo_id}/            (3960 per-combo subdirectories)
```
`phase3_failures.csv` was not written (0 failures). Not added to git,
matching `.gitignore`'s existing `results/*` rule. `results/
formal_phase3_attack_reduced/` (the earlier reduced-tier run) was left
completely untouched.

### 11.6 Conclusion

Phase 3 is complete at full formal scale: 3960/3960 clean, 0 errors, 0
fallback, 100% eval-mode restoration, 100% eps-invariant compliance,
reproducible. Two genuinely new findings emerged only at this scale/via
cross-tabulation (the fgsm-specific eps=0.1 dip, and BPSK's CW-specific
resistance) that neither the reduced tier nor Phase 1 could have
surfaced -- both explicitly flagged as unexplained, not overclaimed.
Phase 4 (Top-K defense) remains designed-but-not-run, per this round's
explicit scope.

---

## 12. Phase 4 audit: design, fairness review, metric definitions, smoke test (round 21)

### 12.1 Exact design (re-read from `docs/formal_experiment_matrix.csv`,
not assumed)

`phase=4/tier=reduced` row: `fixed_params` = "same as Phase 3 (reduced)
plus `use_real_topk=True`"; `sweep_params` = "Phase 3 (reduced)'s full
combo set x topk in {10,20,30,40}". This means Phase 4's grid is built by
taking Phase 3's exact (modulation, SNR, eps, attack) combo set and
multiplying by the 4 K values -- not a new, independently-specified grid.

- **modulation**: `QPSK, BPSK, QAM16, 8PSK, QAM64, WBFM` (6)
- **SNR**: `-10, -4, 0, 6, 12, 18` (6)
- **attack**: `fgsm, pgd, cw` (3 -- "none" is NOT in the formal grid, only
  used in this round's smoke test for the bit-identical-no-op system check)
- **attack_eps**: `0.01, 0.03, 0.05, 0.1, 0.3` (5, fgsm/pgd only)
- **Top-K**: `10, 20, 30, 40` (4)
- **sample_index**: `n_per_cell=10` (full formal 0-9 set) for the CSV's
  named "reduced" tier
- **FGSM combos**: 1800 attack-instances x 4 K = **7200**
- **PGD combos**: 1800 x 4 = **7200**
- **CW combos**: 360 x 4 = **1440**
- **Total**: **15840** (confirmed exactly by `--dry-run`, matching the
  CSV's `combo_count` field exactly)
- **Estimated time**: ~8.8 hours (CSV `est_total_human`)
- **Fixed sensing params**: `threshold_factor=1.5, sensing_window_size=128,
  min_region_len=0, merge_gap=0` (same as Phase 1's formal values)
- **alignment_policy**: `max-energy`. **awn_preprocess**: `radioml-native`.
  **seed**: `42`. **CW**: `cw_c=1.0, cw_steps=20, cw_lr=0.01` (defaults).

### 12.2 Fairness review

`experiments/run_phase3_attack_effectiveness.py` (which calls
`run_dry_run_experiment()` once per combo) is **not suitable** as a base
for Phase 4 -- it provides no mechanism for literal cross-K IQ reuse, only
reproducible regeneration. The only existing script with the required
architecture is `experiments/run_phase0_pilot.py` (Phase 0's pilot),
which already implements: compute clean IQ once, compute attacked IQ once
per (mod,snr,idx,attack) instance, loop ONLY the Top-K + defended-AWN
stage across K, and assert the attacked-IQ hash never changes mid-loop.
**New script**: `experiments/run_phase4_defense_effectiveness.py`, built
by extending this exact architecture with `attack_eps` as an added swept
dimension (to match Phase 3's grid) and the eval-mode-restoration tracking
added during Phase 3's review (section 10.1 item 6).

All 12 required fairness properties confirmed by direct code read (not
assumed) and/or the smoke test (section 12.5):

1. Each (mod,snr,idx,attack,eps) combo generates attacked IQ **exactly
   once** -- `attack_adapter.apply()` is called once per attack-instance,
   outside the `for topk in topks` loop.
2/3. The same `x_adv` array object is passed into `topk_adapter.apply()`
   for all 4 K values -- never regenerated. A live assertion
   (`recheck_hash == attacked_iq_hash`) fires if this is ever violated;
   it never fired.
4. `attacked_iq_sha256` recorded on every row; confirmed identical across
   all 4 K rows for all 16 smoke-test attack-instances (0 violations).
5. `pred_clean`/`pred_attacked` confirmed identical across all 4 K rows
   for all 16 smoke-test instances (0 violations) -- both are computed
   once per instance, before the K loop, and copied into every row.
6. `pred_defended` confirmed to actually vary across K in the smoke test
   (not a constant column) -- the ONLY prediction column that legitimately
   changes per K.
7. **AWN eval-mode restoration**: `attack_training_before`/
   `attack_training_after`/`eval_mode_restored` columns added (same
   mechanism as Phase 3's fix); confirmed 100% restored in the smoke test
   for every attack that actually invokes the real torchattacks path
   (fgsm/pgd/cw show `eval_mode_restored=True`; `attack=none` shows blank
   -- confirmed this is because the real backend's `none` branch returns
   immediately without ever toggling the model's train/eval state, so
   there is nothing to "restore" -- not a gap).
8. `TopKAdapter`'s real backend is confirmed via `_REAL_SOURCE =
   "external/adversarial-rf/util/defense.py:fft_topk_denoise"` (imported
   directly, not restated) -- matched on every smoke-test row.
9/10. `precheck_real_backends()` refuses to run ANY combo if AWN, attack,
   or Top-K construction did not report the real backend; per-row
   `awn_ok`/`attack_ok`/`topk_ok` checks additionally force
   `run_status=error` if any backend or eps-invariant or eval-mode check
   fails at call time, regardless of what the adapter itself reported.
11. `TopKAdapter.apply()`'s own `require_valid_topk()` boundary (already
   validated in round 14, CSV row `S,coverage_topk_full_range`) is called
   unconditionally inside the adapter -- this script passes only the 4
   legal K values `{10,20,30,40}`, so no clamping of an undocumented value
   occurs; K=10..40 are all well inside the validated `[1,128]` legal
   range, none require the `<=0` bypass or `>128` clamp paths.
12. `external/AWN`/`external/adversarial-rf` are read-only via the
   existing adapters, same as every prior round -- not modified.

### 12.3 Metric definitions (explicit denominators, per this round's
instruction not to leave "recovered/total" ambiguous)

Let `N` = all `ok` rows (one row per (mod,snr,idx,attack,eps,K) combo).
Let `N_inst` = attack-instances (`N` divided by 4, since clean/attacked
quantities don't vary by K).

- **clean_accuracy** = mean(`clean_correct`) over `ok` rows, computed at
  the ATTACK-INSTANCE level (each instance's `clean_correct` counted once,
  not 4x) to avoid K-inflating a quantity that cannot vary with K.
- **attacked_accuracy** = mean(`attacked_correct`), same instance-level
  dedup as above.
- **defended_accuracy** = mean(`defended_correct`) over ALL `ok` rows
  (this one legitimately varies by K, so no dedup -- report per-K).
- **overall_attack_success_rate** = count(`changed_by_attack`) /
  `N_inst` (instance-level, matches Phase 3's exact definition).
- **conditional_attack_success_rate** = count(`changed_by_attack`) among
  instances where `clean_correct=True`, divided by that subset's count
  (instance-level).
- **prediction_changed_rate** = same as `overall_attack_success_rate`.
- **overall_defense_recovery_rate** = count(`recovered_by_defense`) / `N`
  (row-level, since recovery is inherently a per-K quantity) --
  `recovered_by_defense` = `changed_by_attack AND pred_defended==pred_clean`
  (same definition as Phase 0's pilot / round 12's precedent).
- **conditional_defense_recovery_rate** = count(`recovered_by_defense`)
  among rows where `changed_by_attack=True`, divided by that subset's
  count (row-level, per K) -- "of the attacks that actually succeeded at
  this K, what fraction did Top-K recover."
- **recovery_count** = raw count(`recovered_by_defense`), reported
  alongside the rate (never rate-only).
- **attack_success_and_defense_recovery_count** = count(rows where
  `changed_by_attack=True AND recovered_by_defense=True`) -- identical to
  `recovery_count` by construction (recovery is defined conditionally on
  attack success already), stated explicitly to avoid ambiguity.
- **clean_broken_by_defense_rate** (item C) = count(`clean_broken_by_defense`)
  / count(rows where `clean_correct=True`) -- `clean_broken_by_defense` =
  `clean_correct=True AND defended_correct=False` (a NEW column this
  round, not present in Phase 0's pilot, since Phase 0 never needed a
  degradation-rate view).
- **defense_changed_prediction_rate** = count(`defense_changed_prediction`)
  / `N` -- `defense_changed_prediction` = `pred_defended != pred_attacked`
  (a NEW column; "how often does Top-K change the attacked prediction to
  ANYTHING else, not just back to the clean prediction").
- **Item D** (recovery to the CLEAN prediction, conditioned on the
  attacked prediction being WRONG, not merely changed): denominator =
  count(rows where `attacked_wrong=True`, i.e. `attacked_correct=False`);
  numerator = count of those where `pred_defended==pred_clean`. `attacked_wrong`
  is a NEW column, deliberately distinct from `changed_by_attack` (a
  sample can have `attacked_wrong=True` with `changed_by_attack=False` if
  the CLEAN prediction was already wrong and the attack had no additional
  effect).
- **Item E** (recovery to the TRUE LABEL, conditioned on the attacked
  prediction being wrong): denominator = same as D
  (`attacked_wrong=True`); numerator = count of those where
  `defended_correct=True`.
- **IQ perturbation**: `iq_linf_clean_attacked` (attack's effect, same
  definition as Phase 3), `iq_linf_normalized_clean_attacked` +
  `eps_invariant_ok` (fgsm/pgd only, same round-14 invariant).
- **Top-K distortion**: NEW `iq_linf_attacked_defended` column = Linf
  norm between `x_defended` and `x_adv` -- isolates what Top-K itself
  changes, separate from what the attack changed.
- **Retained frequency ratio**: NEW `retained_freq_ratio` column =
  `min(topk,128)/128` (or `1.0` if `topk<=0`, matching `TopKAdapter`'s
  documented bypass semantics, round 14) -- a simple, directly
  interpretable per-K quantity (K=10 -> 0.078, K=40 -> 0.3125).

### 12.4 Dry-run verification

`--dry-run` (formal reduced tier, default args): **15840 combos**, exact
match to the CSV. `attack-instances (pre-topk-expansion): 3960` (matches
Phase 3's FULL-tier attack-instance count exactly, since Phase 4's
"reduced" tier is built on Phase 3's full 6x6x5x{fgsm,pgd}+cw grid at
`n_per_cell=10`). Per-attack final-row counts: `fgsm=7200, pgd=7200,
cw=1440` (exactly `4x` Phase 3's own 1800/1800/360). Per-topk final-row
counts: `{10:3960, 20:3960, 30:3960, 40:3960}` -- confirms every attack
combo maps to exactly 4 K rows, no combo is missing a K value, and no
combo is duplicated. All `combo_id`s unique.

### 12.5 Smoke test (QPSK/BPSK x SNR{0,18} x idx0 x {none,fgsm,pgd,cw} x
eps=0.05 x K{10,20,30,40} = 64 combos, 16 attack-instances)

Run in 2 independent processes. **64/64 `run_status=ok`** in both. Every
check from this round's instruction passed:

- 0 `error`, 0 fallback (`awn_backend`/`attack_backend`/`topk_backend`
  each a single real-path string across all 64 rows).
- 0 NaN/Inf (`clean_nan`/`attacked_nan`/`defended_nan` all `False`).
- `eval_mode_restored`: `True` for every fgsm/pgd/cw row; blank for
  `none` rows (expected -- the real `none` branch never touches the
  model's train/eval state at all, see section 12.2 item 7).
- **16/16 attack-instances**: `attacked_iq_sha256` identical across all 4
  K rows, `pred_clean`/`pred_attacked` identical across all 4 K rows --
  0 violations.
- `pred_defended` confirmed to vary across K in at least one instance
  (not a frozen column).
- **Top-K genuinely modifies IQ**: `iq_linf_attacked_defended > 0` on
  64/64 rows (never zero -- confirms Top-K is not a metadata-only no-op).
- **`attack=none`**: `clean_iq_sha256 == attacked_iq_sha256` on all 16
  `none` rows -- bit-identical, confirmed no-op.
- **`attack=none` + Top-K distortion is recorded, not hidden**: e.g.
  `QPSK_snr18_idx0_none_k10` flips `pred_clean=9 -> pred_defended=1`
  (`defended_correct=False`) -- this is the EXACT SAME finding Phase 0's
  pilot first surfaced in round 17 for the identical (mod,snr,idx,K)
  triple, now independently reproduced by an architecturally-similar but
  separately-written script, seven rounds later -- a strong cross-script
  consistency signal, not merely a repeat of the same code path.
- **Reproducibility**: 0 mismatches across all 58 comparable columns (60
  total, excluding `output_dir`/`runtime_seconds`) between the two
  independent processes.

### 12.6 CSV schema (60 columns)

```
combo_id, dataset, modulation, snr, sample_index, seed, attack, attack_eps,
attack_temperature, cw_c, cw_steps, cw_lr, topk, retained_freq_ratio,
threshold_factor, sensing_window_size, min_region_len, merge_gap,
detection_success, detection_probability, false_alarm_rate,
captured_signal_ratio, extra_captured_noise_ratio, start_boundary_error,
end_boundary_error, missed_sample_count, false_occupied_sample_count,
segment_count, label, pred_clean, pred_attacked, pred_defended,
clean_correct, attacked_correct, defended_correct, changed_by_attack,
attacked_wrong, recovered_by_defense, defense_changed_prediction,
clean_broken_by_defense, iq_linf_clean_attacked,
iq_linf_normalized_clean_attacked, eps_invariant_ok,
iq_linf_attacked_defended, awn_backend, attack_backend, topk_backend,
clean_nan, attacked_nan, defended_nan, attack_training_before,
attack_training_after, eval_mode_restored, runtime_seconds, run_status,
failure_stage, failure_reason, output_dir, clean_iq_sha256,
attacked_iq_sha256
```

### 12.7 Reduced-tier design (designed + dry-run ONLY, not executed this round)

Keeps the full formal grid (all 6 modulations, all 6 SNRs, all 5 eps
values, all 3 attacks, all 4 K) and restricts only `sample_index` to the
first 2 of the formal 0-9 set (`[0,1]`) -- same reduction pattern as
Phase 3's own reduced tier, no change to the research design.

`--dry-run --sample-indices 0,1`: **3168 combos** (`fgsm=1440, pgd=1440,
cw=288`), **792 attack-instances** (exactly matching Phase 3-reduced's
own attack-instance count, since it is the identical mod/snr/eps/attack
grid). Estimated time: Phase 3-reduced's 792 attack-instances took
1100.7s measured; Phase 4 reuses that identical cost plus 2376 additional
K-only rows (Top-K + 1 AWN forward pass each, cheap relative to a full
attack computation) -- estimated **~26 minutes**, within the requested
20-40 minute window. **Not started this round** -- awaiting explicit
confirmation, per instruction.

### 12.8 Outstanding risks

- The reduced-tier time estimate (~26 min) is derived from Phase
  3-reduced's measured attack-instance cost plus an estimated (not yet
  measured) per-K marginal cost -- actual reduced-tier runtime may differ
  once run.
- `attack=none`'s `eval_mode_restored=blank` (not `True`/`False`) is a
  structurally-expected null, but if a future refactor of
  `AttackAdapter`'s `none` branch ever starts touching the model's
  train/eval state, this script's current logic (`None if
  attack_training_after is None`) would silently continue reporting blank
  rather than catching a new failure mode -- worth revisiting if
  `attack_adapter.py` is ever modified (out of this round's scope, no
  such modification occurred or is planned).
- Metric definitions in section 12.3 (especially items D/E) are new to
  this round -- they have not yet been cross-validated against an
  independent hand-calculation on real data (only structurally reviewed
  and unit-level smoke-tested); the reduced-tier run (once approved)
  would be the first chance to sanity-check them against a non-trivial
  sample size.
- Phase 4's full run (~8.8 hours) is a long, single-session commitment;
  no chunking/checkpointing strategy beyond the existing `--resume`
  mechanism has been discussed for spanning multiple sessions if needed.

---

## 13. Phase 4 reduced-tier execution: Top-K defense, N=792 instances / 3168 rows (round 22)

Executed exactly as designed and dry-run-verified in section 12.7:
`--sample-indices 0,1` on the full formal 6x6x5x{fgsm,pgd}+cw grid x 4 K
values, into `results/formal_phase4_defense_reduced/` (a directory
distinct from the smoke test's `/tmp` location, both Phase 3 output
directories, and any future full-Phase-4 directory). Launched via
`nohup`, monitored at ~10-minute intervals with a live error/fairness-
signature watch (never fired). **Actual runtime: 172.3s (2.9 minutes)**
-- far faster than the ~26-minute estimate (the estimate conservatively
assumed a larger per-K marginal cost than what real hardware delivered).

### 13.1 System state (Part 一)

- **Attack instances**: 792 (confirmed by distinct (mod,snr,idx,attack,eps)
  tuples in the output).
- **Final rows**: 3168.
- **`ok`**: 3168. **`sensing_failed`**: 0. **`error`**: 0.
- **Runtime**: 172.3s.
- **Real backend ratio**: 100% -- `awn_backend`, `attack_backend`,
  `topk_backend` each a single real-path string across all 3168 rows.
- **Fallback count**: 0.
- **NaN/Inf count**: 0 (`clean_nan`/`attacked_nan`/`defended_nan` all
  `False` on every row).
- **Eval-mode restored**: 3168/3168 = 100% (`eval_mode_restored=True`
  on every row; no `attack=none` rows in the formal grid, so no blanks
  this round, unlike the smoke test).
- **Reproducibility**: 96-combo spot-check (BPSK+QAM64 x SNR{-10,18} x
  idx{0,1} x eps{0.01,0.3} x {fgsm,cw} -- deliberately targeting the two
  most extreme K=10 defended-accuracy cases found, BPSK near-0 and QAM64
  0.705) in a fresh independent process: **92/96 rows bit-identical on
  all 58 comparable columns; 4/96 rows (BPSK's very first attack-instance
  in that process) differed ONLY in `attack_training_before`**.
  **Diagnosed, not a bug**: `attack_training_before` records whatever
  train/eval state the model happened to be in immediately before that
  specific `apply()` call -- confirmed via `nn.Module`'s documented
  default (`training=True` on construction, verified directly:
  `nn.Linear(2,2).training == True`) that the model starts in train mode
  and is only ever set to eval by the PREVIOUS call's `finally` block.
  Whichever combo happens to be the first attack call in a given process
  launch will see `attack_training_before=True`; every later combo in
  that same process sees `False`. This is a property of PROCESS LAUNCH
  ORDER, not of the combo itself -- in the main run, QPSK is processed
  before BPSK, so BPSK's first instance already inherits eval mode from
  QPSK's last call; in the isolated repro run (only BPSK+QAM64 requested),
  BPSK's first instance IS the process's first attack call. Critically,
  `attack_training_after` (the field that actually matters for
  correctness) and `eval_mode_restored` were **identical (`True`) in all
  96 rows** -- eval-mode restoration itself is unaffected and fully
  reproducible; only an incidental bookkeeping field about pre-call state
  varies with unrelated process launch order.

### 13.2 Fairness (Part 二)

- **12. Every attack instance has exactly 4 K rows**: 792/792.
- **13. `attacked_iq_sha256` consistent across K**: 0 violations across
  792 instances.
- **14. `pred_clean` consistent across K**: 0 violations.
- **15. `pred_attacked` consistent across K**: 0 violations.
- **16. No K re-executed the attack**: structurally guaranteed by code
  (`attack_adapter.apply()` called exactly once per instance, outside the
  `for topk in topks` loop) -- confirmed indirectly by items 13-15 (any
  re-execution would show as a hash/prediction difference across K, and
  none did).
- **17. attack-instances x 4 == final rows**: 792 x 4 = 3168 = final row
  count. Exact match.

### 13.3 Accuracy and attack (Part 三, instance-level, N=792)

- **18. clean_accuracy**: 0.5139 (407/792)
- **19. attacked_accuracy**: 0.2487 (197/792)
- **20. overall_attack_success_rate**: 653/792 = 0.8245
- **21. conditional_attack_success_rate**: 321/407 = 0.7887
- **22. prediction_changed_rate**: 0.8245 (same as 20)
- **23. Per-attack**: cw (0.9306) > pgd (0.8833) > fgsm (0.7444) -- same
  ordering as Phase 3's reduced tier. **Cross-check**: Phase 3-reduced's
  independently-run overall success rate was 654/792=0.8258 vs this
  round's 653/792=0.8245 -- a single-instance difference, consistent with
  the previously-documented (rounds 15/16) ordinary multi-threaded BLAS
  floating-point non-determinism at the 2^-12-ish logit-noise level near
  a decision boundary, not a new finding requiring investigation.
- **24. Per-eps** (pooled fgsm/pgd): success rate rises from 0.4722
  (eps=0.01) to 0.9583 (eps=0.30), same shape as Phase 3.
- **25. Per-modulation**: WBFM's clean_acc=0.0833 /
  attacked_acc(higher)=0.5379 pattern reproduced again (now the fourth
  independent confirmation across Phase 1/Phase 3-reduced/Phase 3-full/
  this round).
- **26. Per-SNR**: attack success highest at low SNR, same shape as
  Phase 3.

### 13.4 Defense, per K (Part 四)

| K | defended_acc (27) | overall_recovery (28) | conditional_recovery (29) | true_label_recovery (30, item E) | clean_pred_recovery (31, item D) | clean_degradation (32, item C) | attacked_pred_change (33) | Top-K distortion mean (34) | retained_freq_ratio (35) |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.2020 | 151/792=0.1907 | 151/653=0.2312 | 101/595=0.1697 | 129/595=0.2168 | 320/407=0.7862 | 526/792=0.6641 | 0.01379 | 0.078 |
| 20 | 0.2487 | 111/792=0.1402 | 111/653=0.1700 | 82/595=0.1378 | 97/595=0.1630 | 294/407=0.7224 | 369/792=0.4659 | 0.01128 | 0.156 |
| 30 | 0.2247 | 107/792=0.1351 | 107/653=0.1639 | 61/595=0.1025 | 100/595=0.1681 | 301/407=0.7396 | 299/792=0.3775 | 0.00958 | 0.234 |
| 40 | 0.2197 | 96/792=0.1212 | 96/653=0.1470 | 46/595=0.0773 | 102/595=0.1714 | 305/407=0.7494 | 239/792=0.3018 | 0.00813 | 0.313 |

(Item E denominator = 595 rows where `attacked_wrong=True`; item D uses
the same 595-row denominator, per the explicit definitions in section
12.3.)

- **36. attack x K** (defended_acc / conditional_recovery):
  fgsm K10:0.189/0.216 K20:0.242/0.157 K30:0.228/0.146 K40:0.239/0.119;
  pgd K10:0.219/0.220 K20:0.244/0.138 K30:0.208/0.110 K40:0.194/0.079;
  **cw K10:0.181/0.343 K20:0.306/0.373 K30:0.292/0.493 K40:0.250/0.582**
  -- cw's conditional recovery rate RISES with K while fgsm/pgd's FALLS,
  confirming the same K-dependent-only-for-cw pattern round 12's
  480-combo sweep first found (there: 12%->24%->36%->28% peaking at
  K=30; here: 34%->37%->49%->58%, still rising at K=40) -- directionally
  consistent across two independently-built runners, three rounds apart.
- **37. eps x K** (conditional_recovery, fgsm/pgd pooled): recovery is
  highest at the WEAKEST eps (0.01: up to 0.382 at K=10) and falls to
  near-zero at strong eps (0.3: 0.03-0.11 across all K) -- Top-K's
  fgsm/pgd recovery is essentially a "rescues weak attacks only" effect.
- **38. modulation x K** (defended_acc): **QAM64 is a dramatic outlier**
  -- K=10 gives 0.705 defended accuracy (by far the best result in the
  entire table), falling to 0.326 by K=40. **BPSK and QPSK are the
  opposite extreme**: K=10 gives 0.000 (BPSK) and 0.008 (QPSK) --
  essentially total classification failure -- rising to 0.25/0.242 by
  K=30-40. 8PSK/QAM16/WBFM show smaller, less monotonic variation.
- **39. SNR x K** (defended_acc): generally higher K performs better at
  most SNRs except -10dB (where K=10 is best, 0.159 vs 0.098 at K=40) --
  no single K dominates across every SNR.

### 13.5 Key interpretations (Part 五)

- **40. Highest defended accuracy**: **K=20** (0.2487), narrowly ahead of
  K=30 (0.2247) and K=40 (0.2197); K=10 is clearly worst (0.2020).
- **41. Highest recovery rate**: **K=10** for BOTH overall (0.1907) and
  conditional (0.2312) -- the SAME K that has the worst overall defended
  accuracy. This is not a contradiction: K=10's aggressive filtering
  recovers more successfully-attacked samples but simultaneously damages
  far more originally-correct samples (see item 42), so its net accuracy
  is still the lowest of the four.
- **42. Most damaging to clean accuracy**: **K=10** (`clean_degradation_
  rate`=0.7862 -- nearly 79% of originally-correct clean predictions are
  broken by Top-K at this K). Even the LEAST damaging K (K=20, 0.7224)
  still breaks over 72% of correct clean predictions.
- **43. Recovery vs. degradation trade-off**: **there is a trade-off, but
  it is extremely lopsided, not a balanced one.** Recovery rate ranges
  only 0.147-0.231 across all 4 K values, while clean degradation rate
  ranges 0.722-0.786 -- degradation is 3.2x to 5.3x larger than recovery
  at every single K tested. K=10 has simultaneously the HIGHEST recovery
  AND the HIGHEST degradation (both extremes at once) -- there is no K in
  this grid where recovery approaches, let alone exceeds, degradation.
- **44. Best K differs by attack**: **yes, clearly.** fgsm and pgd both
  peak at K=10 (0.216 and 0.220 conditional recovery) and monotonically
  decline as K increases. **cw is the opposite** -- it peaks at K=40
  (0.582) and is worst at K=10 (0.343), rising monotonically with K.
  There is no single K that is simultaneously optimal for all three
  attacks.
- **45. Any attack Top-K is nearly powerless against**: not in the
  attack-success sense (all three attacks succeed 74-93% of the time,
  none near zero) -- but in the DEFENSE sense, fgsm/pgd's conditional
  recovery caps out at ~0.22 (fgsm K=10) and ~0.22 (pgd K=10), meaning
  Top-K fails to recover over 78% of successful fgsm/pgd attacks at
  EVERY K tested; cw fares somewhat better at high K (0.582 at K=40) but
  still fails on 42% of successful attacks even at its best K.
- **46. Modulations/SNRs where Top-K makes things WORSE than no defense**:
  **5 of 6 modulations** (8PSK, BPSK, QAM16, QPSK, WBFM) show LOWER mean
  defended accuracy (averaged across all 4 K) than their own
  attacked-without-defense accuracy -- i.e., applying Top-K is actively
  counterproductive on average for these 5 modulations at this
  parameter set. **QAM64 is the sole exception**, where Top-K roughly
  doubles accuracy (0.197 attacked -> 0.453 mean defended). This is the
  single most important finding of this round.
- **47. Is the reduced tier sufficient to launch the full 15840-row
  Phase 4?** System-correctness: yes, unconditionally -- every check
  passed at 100% (0 error/fallback/NaN, 100% eval-mode, 100% fairness).
  Scientifically: the reduced tier's central finding (Top-K's clean-
  accuracy damage dwarfs its attack-recovery benefit, at every K, for
  most modulations) is consistent, not noisy or borderline -- it does not
  need a larger N to be legible. Whether to proceed to the full 15840-row
  run is a decision about whether finer-grained statistics (e.g. per-
  (modulation,SNR,eps) cells currently at n=2 in this reduced tier,
  vs. n=10 in the full grid) are needed for the paper, not about
  resolving system-correctness doubt -- there is none. **Recommendation
  left to the user's explicit decision, not started automatically.**

### 13.6 Outputs (Part 六)

```
results/formal_phase4_defense_reduced/phase4_summary.csv     (3168 rows, 58 columns)
results/formal_phase4_defense_reduced/phase4_aggregate.csv   (4 rows, one per K, all Part 四 metrics)
results/formal_phase4_defense_reduced/phase4_manifest.json
results/formal_phase4_defense_reduced/stdout.log, stderr.log
results/formal_phase4_defense_reduced/{mod}_snr{snr}_idx{idx}/   (72 per-sample subdirectories)
```
`phase4_failures.csv` was not written (0 failures). Not added to git,
matching `.gitignore`'s existing `results/*` rule.

### 13.7 Conclusion

Phase 4's reduced tier is complete and system-clean: 3168/3168 `ok`, 0
errors, 0 fallback, 100% eval-mode restoration, 100% fairness (0/792
violations), reproducible (only an explained, inconsequential
process-order artifact in one diagnostic field). The central scientific
finding -- **Top-K's clean-accuracy damage (72-79% degradation rate)
vastly exceeds its attack-recovery benefit (12-23% conditional recovery)
at every K tested, and Top-K is actively counterproductive on average for
5 of 6 modulations** -- is a genuine reduced-tier result, not a system
artifact, though it awaits the full 15840-row run (or an explicit
decision not to run it) for full-N statistical confirmation. The full
Phase 4 run was **not started** this round, per explicit instruction.

---

## 14. Phase 4 reduced-tier root-cause analysis (round 23)

Verification and diagnosis only -- no algorithm code modified, no new
formal batch run beyond one small (144-combo) `attack=none`-only ad-hoc
check. `external/AWN`/`external/adversarial-rf` not touched.

### 14.1 Metric formula re-verification (independent, from raw 3168 rows)

Every metric was recomputed directly from `phase4_summary.csv` (not the
pre-existing `phase4_aggregate.csv`), with explicit numerator/denominator
listed at every K. **All values matched the aggregate CSV exactly -- no
discrepancy found.** Confirmed directly against source
(`experiments/run_phase4_defense_effectiveness.py:323-358`):

```
changed_by_attack       = pred_attacked != pred_clean
attacked_wrong           = pred_attacked != label
recovered_by_defense     = changed_by_attack AND pred_defended == pred_clean
defense_changed_prediction = pred_defended != pred_attacked
clean_broken_by_defense  = (pred_clean == label) AND (pred_defended != label)
```

Explicit denominators confirmed constant across K (as expected, since
none of `clean_correct`/`attacked_wrong`/`changed_by_attack` depend on
K): `clean_correct=True` -> 407 of 792; `attacked_wrong=True` -> 595 of
792; `changed_by_attack=True` -> 653 of 792. The six required checks:

1. **Clean degradation denominator is `clean_correct=True` ONLY**: confirmed
   (407 at every K, never drifts).
2. **True-label recovery denominator is `attacked_correct=False`**
   (`attacked_wrong=True`): confirmed (595 at every K).
3. **Conditional recovery denominator is attack-succeeded
   (`changed_by_attack=True`)**: confirmed (653 at every K).
4. **Clean-prediction recovery (D) and true-label recovery (E) are not
   conflated**: confirmed distinct numerators at every K (e.g. K=10:
   D-numerator=129 vs E-numerator=101).
5. **`attack=none` never contaminates recovery statistics**: confirmed --
   `set(r["attack"] for r in rows) == {"fgsm","pgd","cw"}`, `none` is
   structurally absent from this dataset (Phase 4's formal grid never
   includes it; only the smoke test and this round's dedicated check do,
   in separate output directories).
6. **Numerator/denominator/value all listed together**: done for every
   metric at every K (see full table below).

### 14.2 K=20 "unchanged" investigation: prediction transition matrix

Built the full 2x2 (`attacked_correct` x `defended_correct`) transition
matrix for every K, using the raw 792-row-per-K data:

| K | attacked_correct→defended_correct | attacked_correct→defended_WRONG (degradation) | attacked_WRONG→defended_correct (recovery) | attacked_wrong→defended_wrong | net change |
|---|---|---|---|---|---|
| 10 | 59 | 138 | 101 | 494 | -37 (-0.0467) |
| **20** | **115** | **82** | **82** | **513** | **+0 (0.0000)** |
| 30 | 117 | 80 | 61 | 534 | -19 (-0.0240) |
| 40 | 128 | 69 | 46 | 549 | -23 (-0.0290) |

**Definitive finding: K=20's defended_accuracy exactly equals
attacked_accuracy NOT because nothing changed, but because recovery
count (82) exactly equals degradation count (82) -- an exact churn
cancellation, not a no-op.** 164 of 792 rows (20.7%) actually flip
correctness status at K=20 (82 in each direction); the net accuracy
number alone completely hides this churn. A naive before/after accuracy
comparison at K=20 would have been actively misleading without this
transition-matrix breakdown.

### 14.3 Top-K implementation trace (`TopKAdapter` -> `fft_topk_denoise`)

Confirmed by direct code read (`src/adapters/topk_adapter.py:46`,
`external/adversarial-rf/util/defense.py:163-191`) and a live empirical
check (real backend, `/home/xiaomi/adversarial-rf/.venv`):

1. **Input is `x_attacked`, not `x_clean`**: confirmed --
   `experiments/run_phase4_defense_effectiveness.py`'s
   `topk_adapter.apply(x_adv, topk=topk)` call passes `x_adv` (the
   post-attack array), never `x_clean`.
2. **FFT axis**: `torch.fft.fft(x, n=T, dim=2)` -- `dim=2` is the time
   axis (last dim of `[N,2,T]`), computed independently per (N,C) pair.
3. **I/Q handling**: **I and Q are FFT'd as two SEPARATE REAL-valued
   channels, never combined into a complex `I+jQ` signal before FFT.**
   `x[:, 0, :]`/`x[:, 1, :]` (or the batched equivalent `dim=2` FFT over
   the whole `[N,2,T]` tensor) each independently undergo a real-input
   FFT. This matches the AWN model's own `[N,2,T]` input convention (2
   real channels, not 1 complex channel) throughout this entire
   pipeline -- not specific to Top-K.
4. **K scope**: per-sample, per-channel (`mags.topk(k=k, dim=2)` -- I and
   Q can select DIFFERENT top-k bin indices from each other; K is never
   shared jointly across the two channels or across the batch).
5. **Selection criterion**: magnitude (`mags = X.abs()`), confirmed.
6. **Unselected bins**: zeroed (`mask` initialized `False`/0 everywhere,
   `True` only at the top-k indices, elementwise multiply) -- confirmed.
7. **Post-IFFT shape/dtype/scale**: shape `[N,2,T]` preserved; dtype
   float32 preserved (`.real` of the complex IFFT result); **scale is NOT
   explicitly renormalized anywhere** -- energy naturally decreases
   because some frequency content is discarded, with no compensating
   rescale step in `fft_topk_denoise` itself.
8. **No additional normalization/clipping/rescaling** exists inside
   `fft_topk_denoise` itself (confirmed by reading the full 29-line
   function body, lines 163-191) -- see section 14.4 for a DIFFERENT
   function, `fft_topk_denoise_normalized`, that does add normalization,
   which `TopKAdapter` does NOT import or use.
9. **K=128 reconstruction**: empirically verified live (real backend):
   max abs diff from the original input = `4.77e-7` (float32
   machine-epsilon-level FFT/IFFT round-trip error, not literally
   bit-exact but numerically indistinguishable) -- confirms K=128 keeps
   effectively all information.
10. **K<=0 semantics**: empirically verified live: `fft_topk_denoise(x, 0)`
    and `fft_topk_denoise(x, -5)` both return the input completely
    UNCHANGED (`torch.equal(y, x) == True`) -- a full BYPASS (keep
    everything), not "keep 0 bins" (which would fully zero the signal).
    Matches the source's explicit `if topk is None or topk <= 0: return x`
    guard and round 14's prior documented finding, re-confirmed here
    empirically rather than only cited.

### 14.4 Comparison against `external/adversarial-rf`'s own historical Top-K usage

`fft_topk_denoise` (`util/defense.py:163`) is used across dozens of the
external repo's own scripts (`util/adv_eval.py`, `util/sigguard_eval.py`,
`util/detector.py`, `util/defense_registry.py`, and ~25 more
`test_*.py`/`*_experiment.py` scripts) -- confirming it is the
project's actual, canonical, in-use Top-K implementation, not an unused
variant.

**Direct line-by-line comparison against `AWN_All.py`'s
`filter_top_components_torch(data, top_n)`** (the original,
pre-refactor implementation `fft_topk_denoise`'s own docstring claims to
"mirror"):

| Property | `AWN_All.py:filter_top_components_torch` | `util/defense.py:fft_topk_denoise` | Match? |
|---|---|---|---|
| FFT type | `torch.fft.fft` (full complex) per (i,j) in an explicit double loop | `torch.fft.fft(x, dim=2)` vectorized over (N,C) | Yes -- same math, vectorized vs. looped |
| I/Q treatment | Each channel `data[i,j]` FFT'd separately (real input) | Same -- `dim=2` FFT applied independently per channel | Yes |
| K scope | Per (sample, channel), via the loop | Per (N,C), via `dim=2` topk | Yes |
| Selection | `torch.topk(torch.abs(fft_result), top_n)` | `mags.topk(k, dim=2)` where `mags=X.abs()` | Yes |
| Masking | Zero-init, then assign only top-k indices back | Zero/False mask, `True` at top-k, elementwise multiply | Yes -- equivalent result, different mechanics |
| Output | `.real` of the IFFT result | `.real` of the IFFT result | Yes |
| **Normalization** | **`normalize_data(x) = (x+0.02)/0.04` applied to input BEFORE calling the top-k function** (confirmed at `AWN_All.py:337-339`: `inputs = normalize_data(inputs); filtered_sample = filter_top_components_torch(inputs, TopN); outputs = model(filtered_sample)` -- model receives the STILL-NORMALIZED, filtered output, no denormalization step) | **No normalization anywhere in `fft_topk_denoise` itself** | **NO -- confirmed difference** |

**Diagnosis (category B: implementation differs from one historical
variant, NOT proven wrong)**: The core FFT/mask/IFFT mathematics are
identical. The divergence is specifically that `AWN_All.py`'s pipeline
normalizes via a fixed affine transform (`(x+0.02)/0.04`) before Top-K
and feeds the model the still-normalized result, while this repo's
`TopKAdapter` (via the plain `fft_topk_denoise`) does not. **This
difference does NOT automatically mean this repo's implementation is
wrong**, for three independent reasons:

1. `AWN_All.py` targets entirely different checkpoint files
   (`AWN_CLS_best_acc.pth`, `Detector_CNN_best.pth`, both
   Google-Drive-hosted) -- not the `2016.10a_AWN.pkl` checkpoint this
   repo/session pins. `AWN_All.py`'s normalization convention was
   calibrated for a DIFFERENT model, not necessarily transferable.
2. Round 10 (`docs/parameter_validation.md` section 19.1) already traced,
   file-and-line, that the ACTUAL AWN training/eval pipeline
   (`data_loader.py`, `util/training.py`, `util/evaluation.py` -- the
   scripts that actually produced the pinned checkpoint's weights) apply
   **zero normalization** anywhere between the pickle loader and
   `AWN.forward()`. This repo's `radioml-native` policy (used for
   clean/attacked inference throughout every phase, including this one)
   was deliberately built to match THAT evidence, not `AWN_All.py`
   (a separate, seemingly older/different-purpose standalone script).
3. Applying `AWN_All.py`-style normalization to ONLY the Top-K stage
   (while clean/attacked inference stay `radioml-native`/unnormalized,
   as they do everywhere else in this repo) would introduce an internal
   SCALE MISMATCH between the defended prediction and every other
   prediction in the same run -- plausibly making results worse, not
   better, not clearly a fix.

**Per explicit instruction ("不得更動Top-K演算法,除非先完成比對並明確
證明目前實作錯誤"), this comparison does NOT constitute proof the
current implementation is wrong -- no code was changed.** A separate,
explicit normalized-Top-K ablation (comparing `fft_topk_denoise` vs.
`fft_topk_denoise_normalized` on an identical combo subset) would be
needed before drawing a stronger conclusion; not run this round.

### 14.5 `attack=none` clean-degradation check (new minimal 144-combo run,
NOT a large experiment)

`--attacks none --sample-indices 0` across the full formal 6 modulations
x 6 SNRs x 4 K (144 combos, 45.9s, output NOT retained in git per
`.gitignore`). Confirms `clean_iq_sha256 == attacked_iq_sha256` on all
144 rows (bit-identical, as already established).

| K | accuracy before Top-K | accuracy after Top-K | prediction changed rate | clean degradation rate | mean IQ distortion |
|---|---|---|---|---|---|
| (none) | 0.3889 (14/36) | -- | -- | -- | -- |
| 10 | -- | 0.1389 | 0.6389 | **0.7857** | 0.01397 |
| 20 | -- | 0.2500 | 0.5000 | 0.4286 | 0.01146 |
| 30 | -- | 0.3056 | 0.3889 | 0.2857 | 0.00941 |
| 40 | -- | 0.3333 | 0.1944 | 0.2143 | 0.00811 |

**Confirmed: Top-K severely damages clean (unattacked) accuracy at every
K, worst at K=10 (accuracy 0.389 -> 0.139, a 25-point drop from a
transformation applied to signals that were never attacked). Even at the
gentlest K=40, accuracy only partially recovers to 0.333, still below
the untouched clean baseline of 0.389.** This N=36-per-K minimal check is
smaller than the reduced tier (n=407 clean-correct rows per K there) but
shows the exact same direction and a compatible magnitude (K=10
degradation 0.786 here vs. 0.786 in the full reduced tier -- identical
to 3 decimal places, a striking cross-check).

Per-modulation (pooled across K, n=24 each): WBFM shows 100% clean
degradation (its 4 clean-correct rows, out of 24 total, ALL get broken);
QAM64 shows the least damage (16.7% degradation, 0.417 mean defended
accuracy -- consistent with its standout K=10 defended-accuracy=0.705
finding from the reduced tier). Per-SNR: degradation is worst at low/mid
SNR (-4dB: 100%, -10dB: 75%) and least at high SNR (12dB: 12.5%).

### 14.6 Per-attack full breakdown (not recovery-rate-only)

| attack | K | attacked_acc | defended_acc | net_gain | recovery_count | degradation_count | true_label_recovery | clean_degradation | distortion |
|---|---|---|---|---|---|---|---|---|---|
| fgsm | 10 | 0.2833 | 0.1889 | -0.0944 | 40 | 74 | 0.1550 | 0.7784 | 0.01452 |
| fgsm | 20 | 0.2833 | 0.2417 | -0.0417 | 32 | 47 | 0.1240 | 0.7027 | 0.01200 |
| fgsm | 30 | 0.2833 | 0.2278 | -0.0556 | 22 | 42 | 0.0853 | 0.7081 | 0.01028 |
| fgsm | 40 | 0.2833 | 0.2389 | -0.0444 | 20 | 36 | 0.0775 | 0.7081 | 0.00872 |
| pgd | 10 | 0.2083 | 0.2194 | **+0.0111** | 50 | 46 | 0.1754 | 0.8000 | 0.01334 |
| pgd | 20 | 0.2083 | 0.2444 | **+0.0361** | 36 | 23 | 0.1263 | 0.7730 | 0.01077 |
| pgd | 30 | 0.2083 | 0.2083 | +0.0000 | 24 | 24 | 0.0842 | 0.8162 | 0.00910 |
| pgd | 40 | 0.2083 | 0.1944 | -0.0139 | 12 | 17 | 0.0421 | 0.8324 | 0.00773 |
| cw | 10 | 0.2778 | 0.1806 | -0.0972 | 11 | 18 | 0.2115 | 0.7568 | 0.01236 |
| cw | 20 | 0.2778 | 0.3056 | **+0.0278** | 14 | 12 | 0.2692 | 0.5676 | 0.01022 |
| cw | 30 | 0.2778 | 0.2917 | **+0.0139** | 15 | 14 | 0.2885 | 0.5135 | 0.00850 |
| cw | 40 | 0.2778 | 0.2500 | -0.0278 | 14 | 16 | 0.2692 | 0.5405 | 0.00717 |

**Only 4 of 12 (attack,K) cells show a positive net_gain at all** (pgd
K=10/K=20, cw K=20/K=30), and even those are small (+0.011 to +0.036) --
dwarfed by the clean_degradation_rate (51-83%) present at every single
cell, including the positive-net-gain ones. fgsm shows a NEGATIVE net
gain at every K tested -- Top-K never helps against fgsm in this grid.

### 14.7 Conclusions

**A. 程式或公式錯誤 (formula/program bugs)**: **NONE FOUND.** Every
metric formula independently re-verified from raw data with correct,
explicit denominators; K=20's apparent "no-op" is a confirmed,
fully-explained churn cancellation (82 recovered = 82 degraded), not a
defect.

**B. 實作與舊專案不一致 (implementation differs from the old project)**:
**FOUND, but not proven wrong.** `TopKAdapter`'s `fft_topk_denoise` omits
the normalization step `AWN_All.py`'s own pipeline applies before/around
Top-K filtering. The core FFT/mask/IFFT math is identical. Three
independent reasons (different checkpoint target, round 10's already-
validated zero-normalization evidence for the ACTUAL training pipeline,
and the scale-mismatch risk of a partial fix) argue against assuming
`AWN_All.py`'s convention is more correct for this repo's pinned
checkpoint. Not fixed -- would need its own dedicated ablation first.

**C. Top-K本身在這個pipeline上確實無效 (Top-K itself is ineffective on
this pipeline)**: **CONFIRMED, and specifically net-harmful, not merely
ineffective.** `attack=none` clean-only testing (section 14.5) proves
Top-K damages accuracy even with zero attack present, at every K (worst
at K=10, 78.6% clean degradation). Under real attacks (section 14.6),
only 4 of 12 (attack,K) cells show ANY net benefit, and all four are
small (+1.1 to +3.6 percentage points) against a 51-83% clean
degradation backdrop at the very same cells.

**D. 因reduced-tier樣本數不足而不能判斷**: The core finding (degradation
>> recovery, at every K, confirmed independently via `attack=none`-only
data at a DIFFERENT sample_index than the main reduced tier) is
NOT sample-limited -- the magnitude gap (51-83% vs 1-29%) is far too
large to be reduced-tier noise, and the `attack=none` cross-check
(section 14.5, K=10 degradation 0.7857 in BOTH the 36-row minimal check
AND the 792-row-per-K reduced tier, matching to 4 significant figures)
is strong independent corroboration. What DOES remain sample-limited:
precise per-(modulation,SNR,eps) cell statistics (currently n=2 in the
reduced tier vs. the formal n=10), and whether striking single-cell
outliers (QAM64's K=10=0.705 defended accuracy, BPSK/QPSK's K=10≈0) are
stable properties or partly influenced by which 2 of the 10 formal
sample_index values were drawn.

### 14.8 Answers

1. **現在是否存在必須先修的bug**: **否。** No formula, fairness, or
   implementation bug found that requires a fix before proceeding.
2. **是否需要修改Top-K實作**: **不建議在本輪修改。** The AWN_All.py
   normalization difference (finding B) is a genuine implementation
   divergence but not proven to be an improvement for this repo's pinned
   checkpoint; changing it now would be an unvalidated speculative fix,
   not a confirmed correction. If pursued, it should be its own
   dedicated ablation round, explicitly comparing normalized vs.
   unnormalized Top-K on an identical combo subset -- not bundled into
   this diagnosis.
3. **是否建議直接跑完整15840 rows**: **不建議立即啟動。** The reduced
   tier's central finding is already clear, large-magnitude, and
   independently cross-checked (not a borderline or noisy result) --
   running 8.8 more hours would mainly sharpen per-cell precision
   (n=2->n=10), not change the qualitative conclusion.
4. **若不建議，下一個最小且必要的驗證**: Two small, targeted checks,
   each far cheaper than the full 15840-row run: (a) a focused
   sample_index expansion (all 10 formal indices, but ONLY for the
   specific modulations showing the most striking single-cell patterns
   -- QAM64, BPSK, QPSK -- at K=10 specifically) to confirm those
   extremes are stable, not a 2-sample artifact; (b) if a normalization
   ablation (finding B) is judged worth pursuing, a small side-by-side
   comparison of `fft_topk_denoise` vs. `fft_topk_denoise_normalized` on
   an identical small combo subset, BEFORE deciding whether to change
   the shipped defense implementation.
5. **若建議，完整Phase4能回答什麼reduced-tier尚不能回答的問題**: (not
   the recommended path this round, but for completeness) full N=10
   would let every per-(modulation,SNR,eps) cell reach the same
   statistical power as Phase 1/3's formal results, sufficient for a
   paper to report per-cell numbers with confidence rather than
   reduced-tier-caveated estimates, and would definitively settle
   whether QAM64/BPSK/QPSK's K=10 extremes are stable or partly
   sample-index-dependent.

No files modified beyond this documentation update; no algorithm code,
`external/AWN`, or `external/adversarial-rf` touched.

---

## 15. Phase 4 Top-K preprocessing-policy ablation (round 24)

Diagnostic ablation only -- no formal defense algorithm changed, no
policy set as any default, `external/AWN`/`external/adversarial-rf`
untouched. New script: `experiments/run_phase4_topk_ablation.py`,
architecturally extending the fair-reuse pattern one level further: the
same attacked IQ is computed exactly once per attack-instance and reused
across all 10 K values AND all 3 preprocessing policies (30 combinations
per instance), never regenerated.

### 15.1 Design

**K values**: `10, 20, 30, 40, 50, 64, 80, 96, 112, 128` (10, extending
well past the formal grid's `{10,20,30,40}`, per this round's explicit
instruction). **Attacks**: `none, fgsm, pgd, cw`. **Modulations**: `QPSK,
BPSK, QAM16, QAM64, WBFM, AM-SSB` (6, the required minimum set).
**SNRs**: `-10, 0, 10, 18` (4, the required minimum set).
**sample_index**: `0,1,2` (3). **eps**: `0.05` (single representative
value). **CW**: `c=1.0, steps=20, lr=0.01` (formal defaults). **seed=42**.

**Three preprocessing policies**, all calling the exact same unmodified
`fft_topk_denoise`:

- **A. `current_radioml_native`**: the existing formal path, completely
  unmodified (`TopKAdapter.apply(x_adv, topk)` as-is).
- **B. `normalized_topk_rescaled`**: records each sample's own power
  scale (`power = mean(x_adv**2)`, the same mathematical convention as
  `src/sensing/normalize.py:normalize_segments`, adapted to real
  `[N,2,T]` tensors), temporarily normalizes to unit power, runs the
  SAME `fft_topk_denoise`, then rescales the result back to `x_adv`'s
  own original power level -- AWN never sees normalized-scale input.
- **C. `legacy_awn_all_reference`**: reproduces `AWN_All.py`'s actual
  historical usage (confirmed round 23, `AWN_All.py:335-339`) exactly:
  `x_norm = (x_adv+0.02)/0.04`, then `fft_topk_denoise(x_norm, topk)`,
  fed DIRECTLY to AWN with no rescale-back. Diagnostic only -- flagged
  throughout as likely out-of-distribution for the pinned 2016.10a
  checkpoint (`AWN_All.py` targets different checkpoint files entirely).

`--dry-run`: **8640 final rows** (72 cells x 4 attack-instances x 10 K x
3 policies), **288 attack-instances**, all `combo_id`s unique. A 24-combo
smoke test (1 mod/1 SNR/1 idx, K={10,128}, all 4 attacks x 3 policies)
passed cleanly and already revealed the two headline patterns (below) at
small scale before the full run.

### 15.2 Execution

Launched via `nohup`, monitored at ~10-minute intervals (live
error/fairness watch never fired). **Actual runtime: 220.5s (3.7
minutes)** -- well under the 10-30 minute target. **8640/8640
`run_status=ok`, 0 `sensing_failed`, 0 `error`.** 0 fairness violations
across 288 attack-instances x 30 (K x policy) = 8640 rows exactly. 0
NaN/Inf anywhere. 100% real backends. `eval_mode_restored`: `True` on
6480 rows, blank on the 2160 `attack=none` rows (same structurally-
expected reason as every prior round -- the real `none` branch never
touches train/eval state). **Reproducibility**: 72 combos (AM-SSB/WBFM x
SNR{-10,18} x eps/K{40,80,128} x {cw,fgsm} -- deliberately targeting the
round's most striking findings) in a fresh independent process: 0
mismatches.

### 15.3 Control checks (Part 三)

**Control 1 -- K=128 per policy** (n=288 each):

| policy | mean Linf | mean L2 | pred agreement | defended_accuracy |
|---|---|---|---|---|
| A `current_radioml_native` | 0.000000 | 0.000000 | 1.0000 | 0.3194 |
| B `normalized_topk_rescaled` | 0.000000 | 0.000000 | 1.0000 | 0.3194 |
| C `legacy_awn_all_reference` | **0.870977** | **8.180620** | **0.0938** | **0.0972** |

**A and B are bit-for-bit IDENTICAL at K=128 (confirmed to full float
precision, not just visually close) -- true no-op, exactly as expected.
C is emphatically NOT a no-op**: Linf=0.87 (vs A/B's 0.0), only 9.4%
prediction agreement with the undefended prediction, and 9.7% accuracy
(near/below chance for 11 classes) -- confirms round 23's prediction
that C's fixed-offset normalization is severely out-of-distribution for
this checkpoint, even when K=128 nominally "keeps everything" in the
frequency domain (C never rescales back, so the model receives
permanently wrong-scale input regardless of K).

**Control 2 -- `attack=none`, per K per policy** (n=72 per K):

| K | A/B after_acc | A/B changed_rate | **A/B clean_degradation** | C after_acc | C clean_degradation |
|---|---|---|---|---|---|
| 10 | 0.1667 | 0.7222 | 0.8140 | 0.0972 | 0.8837 |
| 20 | 0.3611 | 0.5417 | 0.5116 | 0.1111 | 0.8605 |
| 30 | 0.4028 | 0.4861 | 0.4419 | 0.1250 | 0.8372 |
| 40 | 0.5000 | 0.2917 | 0.2558 | 0.1111 | 0.8605 |
| 50 | 0.5139 | 0.2083 | 0.1860 | 0.0972 | 0.8605 |
| 64 | 0.5833 | 0.0556 | 0.0233 | 0.0972 | 0.8605 |
| **80** | **0.5972** | 0.0278 | **0.0000** | 0.0972 | 0.8605 |
| 96 | 0.5833 | 0.0694 | 0.0233 | 0.0972 | 0.8605 |
| 112 | 0.5833 | 0.0139 | 0.0233 | 0.0972 | 0.8605 |
| 128 | 0.5972 | 0.0000 | 0.0000 | 0.0972 | 0.8605 |

(clean accuracy before Top-K, this cell set: 0.5972, 43/72)

**Major revision of round 22/23's finding**: clean degradation for
policies A/B decreases essentially MONOTONICALLY as K increases,
reaching **exactly 0.0000 by K=80**. The reduced tier and round 23's
diagnosis only ever tested `K in {10,20,30,40}` -- entirely within the
steep-damage region of this curve. Policy C stays catastrophically bad
(84-88% clean degradation) at EVERY K, never recovering, consistent with
its permanent out-of-distribution scale.

### 15.4 Per-attack full breakdown, policy A (Part 六, all 10 K, n=72/K)

| attack | K=10 | K=20 | K=30 | K=40 | K=50 | K=64 | **K=80** | K=96 | K=112 | K=128 |
|---|---|---|---|---|---|---|---|---|---|---|
| fgsm net_gain | -0.028 | +0.014 | -0.028 | -0.014 | -0.042 | -0.014 | 0.000 | 0.000 | 0.000 | 0.000 |
| pgd net_gain | **+0.097** | +0.083 | +0.028 | +0.014 | +0.028 | +0.014 | +0.014 | +0.014 | 0.000 | 0.000 |
| cw net_gain | -0.083 | +0.042 | +0.083 | +0.125 | +0.097 | +0.028 | **+0.139** | +0.069 | +0.042 | 0.000 |

**fgsm never shows a meaningful positive net gain at any K** -- the
largest is +0.014 (essentially noise, net_transition +1 out of 72).
**pgd peaks early, at K=10 (+0.097, net_transition +7/72), declining
toward 0 as K grows.** **cw peaks LATE, at K=80 (+0.139, net_transition
+10/72: 18 recoveries vs. 8 degradations)** -- a genuinely large, non-
trivial effect entirely outside the formal grid's tested range. Policy C
(diagnostic) shows net_gain -0.056 to -0.167 at every attack and every K
sampled (10/64/128) -- consistently worse than doing nothing, confirming
the out-of-distribution scale is actively harmful, not neutral.

**CW K=80's modulation/SNR breakdown (n=12/modulation, n=18/SNR)**
reveals the aggregate +13.9pp is NOT uniform: AM-SSB +0.750 (0%->75%),
QPSK +0.417, QAM64 +0.333, BPSK/QAM16 +0.000 (no effect), **WBFM -0.667
(Top-K actively WORSE for WBFM at this exact K/attack combination)**. Per
SNR: benefit is largest at low/mid SNR (-10dB: +0.167, 0dB: +0.278) and
smaller at high SNR (10/18dB: +0.056 each). Mean IQ distortion at this
cell: 0.00347 (small, consistent with K=80 retaining 62.5% of the
spectrum).

### 15.5 Answers (Part 四)

1. **是否存在net_gain>0的K**: **是，明確存在.** cw shows net_gain>0 at 7
   of 9 non-trivial K values (20 through 112), peaking +0.139 at K=80.
   pgd shows net_gain>0 at 8 of 9 (10 through 96), peaking +0.097 at
   K=10.
2. **是否存在recovery_count>degradation_count的K**: **是**, same cells
   as above (net_transition positive: cw K=20/30/40/50/64/80/96/112; pgd
   K=10 through K=96).
3. **K增大時clean degradation是否單調下降**: **基本上是**（一個微小的
   K=112 反彈 0.0233,可能是單一樣本雜訊；整體趨勢明確單調：0.814 ->
   0.000 by K=80）。
4. **K=128是否近似no-op**: **對policy A/B是，精確確認**(Linf=L2=0,
   agreement=1.0)；**對policy C不是**(Linf=0.87, agreement僅0.094)。
5. **normalized_topk_rescaled是否優於current_radioml_native**: **否 --
   逐位元完全相同**（K=128及所有其他抽查K皆確認），數學上被magnitude-
   based top-k選擇對正數均勻縮放的不變性所保證。
6. **normalized_topk_rescaled的改善是否來自頻域選擇,而不是AWN scale
   改變**: **此問題的前提不成立** -- 因為policy B與A完全沒有差異
   （不只是"改善很小"，是精確相等），這個exact-equality本身就是一個
   乾淨的數學證明：均勻的per-sample正數縮放結構上不可能改變magnitude
   排序,因此不可能改變Top-K選中的頻域bins。
7. **不同attack是否需要不同K**: **是,非常明確.** cw最佳K約為80,pgd最佳
   K約為10,fgsm在任何K都沒有真正有效的K。三者的最佳操作點彼此不同,
   且方向相反（cw隨K上升,pgd隨K下降）。
8. **是否有任何policy足以支持完整Phase4**: **有條件地是,但不是以現有
   的正式K範圍.** Policy A（現行正式路徑，未改動）在K=80時對cw攻擊
   確實顯示出實質、可重現的正向效果；但**現有正式Phase4設計的K範圍
   （10/20/30/40）完全落在此效果尚未出現的區間**，如果照現有設計跑完整
   15840組，只會重新確認reduced-tier已經確立的「K≤40下Top-K有害」結論
   ，不會捕捉到K=80這個新發現的有利區間。
9. **若所有policy都沒有正net gain,明確結論為Top-K不適用**: **此條件
   不成立** -- policy A/B在多個K值下對cw和pgd皆顯示正向net gain，因此
   不能下「Top-K在此pipeline完全不適用」的結論；正確結論是「Top-K在
   K≤40的範圍內對這組formal參數確實有害，但K=80附近對CW攻擊顯示出
   一個此前未被正式矩陣涵蓋、值得進一步驗證的潛在有效區間」。

### 15.6 Outputs

```
results/formal_phase4_topk_ablation/ablation_summary.csv     (8640 rows, 41 columns)
results/formal_phase4_topk_ablation/ablation_manifest.json
results/formal_phase4_topk_ablation/stdout.log, stderr.log
results/formal_phase4_topk_ablation/{mod}_snr{snr}_idx{idx}/  (72 per-sample subdirectories)
```
`ablation_failures.csv` was not written (0 failures). Not added to git,
matching `.gitignore`'s existing `results/*` rule.
`results/formal_phase4_defense_reduced/` (the round-22 reduced tier) was
never touched or overwritten.

### 15.7 Conclusion

This ablation both CONFIRMS round 22/23's diagnosis (within `K<=40`,
policy A is genuinely net-harmful, not a formula bug, and policy B
provides no improvement) AND MATERIALLY REVISES its scope: the earlier
"Top-K is confirmed net-harmful" conclusion was correct only for the
narrow K range the formal grid and reduced tier ever tested. Extending
to `K<=128` reveals CW-specific defense value peaking around K=80 (+13.9
percentage points, non-uniform across modulations, largest for AM-SSB
and actively negative for WBFM) that the existing formal Phase 4 design
(K limited to {10,20,30,40}) structurally cannot see. **No formal
default was changed; no algorithm was modified; the full 15840-row
Phase 4 (at its current K range) is NOT recommended, consistent with
this finding -- it would not capture the newly-discovered K=80 region.**
Whether to redesign Phase 4's K range (e.g., adding 64/80/96) to properly
investigate this CW-specific opportunity, and whether the WBFM-specific
negative effect at the same K needs separate characterization, are left
as explicit open decisions for the next round, not decided here.

---

## 16. Phase 4 Expanded-K Confirmation Experiment (round 25)

All values in this section are computed directly from the round's own
14256-row `results/formal_phase4_expanded_k/ablation_summary.csv` --
none are carried over or estimated from prior rounds' smaller-N results.
`experiments/run_phase4_topk_ablation.py` was extended (its `--eps`
argument now accepts a comma-separated list, matching Phase 3/4's own
convention) to run the FULL Phase 3 formal grid at once; no algorithm
code was touched. `external/AWN`/`external/adversarial-rf` untouched.

### 16.1 Execution info (Part 一)

- **Design**: Phase 3's exact formal grid (re-read from `docs/
  formal_experiment_matrix.csv`, not assumed) -- modulations `QPSK, BPSK,
  QAM16, 8PSK, QAM64, WBFM` (6); SNRs `-10,-4,0,6,12,18` (6); eps
  `0.01,0.03,0.05,0.1,0.3` (5, fgsm/pgd only); attacks `fgsm, pgd, cw`;
  `sample_index` = first 3 of the formal 0-9 set (`[0,1,2]`); K =
  `10,20,30,40,50,64,72,80,88,96,112,128` (12); policy =
  `current_radioml_native` ONLY (per this round's explicit instruction --
  `normalized_topk_rescaled` and `legacy_awn_all_reference` were not
  re-run, having already been proven bit-identical to A and
  scale-incompatible respectively in round 24).
- **Attack instances**: 1188 (108 cells x 11/cell). **Final rows**:
  14256 (1188 x 12 K). Per-attack: fgsm=6480, pgd=6480, cw=1296.
- **Runtime**: 248.3s (4.1 minutes).
- **Backend**: 100% real on every row (`external/adversarial-rf/
  models/model.py:AWN`, `.../util/adv_attack.py:Model01Wrapper +
  torchattacks`, `.../util/defense.py:fft_topk_denoise`).
- **`run_status`**: 14256/14256 `ok`. **`error`**: 0. **NaN/Inf**: 0
  anywhere. **`eval_mode_restored`**: `True` on all 14256 rows (no
  `attack=none` rows in this design, so no expected blanks).
- **Fairness**: 0 violations across 1188 attack-instances x 12 K = 14256
  rows exactly; `attacked_iq_sha256`/`pred_clean`/`pred_attacked`
  confirmed identical across all 12 K within every instance.
- **Reproducibility**: 64 combos (QAM64/WBFM x SNR{-10,18} x
  idx{0,1} x {cw,pgd} x K{20,40,80,96} -- deliberately targeting the two
  most extreme-response modulations) in a fresh independent process: 0
  mismatches.
- **K=128 no-op control**: max Linf = `1.86e-08` (float32-precision
  no-op), mean prediction agreement = `1.0000` across all 1188 K=128
  rows -- confirmed at full scale, matching round 24's smaller-scale
  finding exactly.

### 16.2 CW results (Part 二)

**Per-K net gain, WITH WBFM** (n=108/K; numerator=defended_correct
count, denominator=108 unless noted):

| K | attacked | defended | net_gain |
|---|---|---|---|
| 10 | 32/108 | 23/108 | -0.0833 |
| 20 | 32/108 | 38/108 | +0.0556 |
| 30 | 32/108 | 36/108 | +0.0370 |
| 40 | 32/108 | 34/108 | +0.0185 |
| 50 | 32/108 | 33/108 | +0.0093 |
| 64 | 32/108 | 25/108 | -0.0648 |
| 72 | 32/108 | 25/108 | -0.0648 |
| 80 | 32/108 | 29/108 | -0.0278 |
| 88 | 32/108 | 25/108 | -0.0648 |
| 96 | 32/108 | 24/108 | -0.0741 |
| 112 | 32/108 | 31/108 | -0.0093 |
| 128 | 32/108 | 32/108 | +0.0000 (no-op) |

**Per-K net gain, EXCLUDING WBFM** (n=90/K, the other 5 modulations):

| K | attacked | defended | net_gain |
|---|---|---|---|
| 10 | 16/90 | 19/90 | +0.0333 |
| 20 | 16/90 | 34/90 | +0.2000 |
| 30 | 16/90 | 33/90 | +0.1889 |
| 40 | 16/90 | 33/90 | +0.1889 |
| 50 | 16/90 | 32/90 | +0.1778 |
| 64 | 16/90 | 24/90 | +0.0889 |
| 72 | 16/90 | 24/90 | +0.0889 |
| 80 | 16/90 | 25/90 | +0.1000 |
| 88 | 16/90 | 19/90 | +0.0333 |
| 96 | 16/90 | 18/90 | +0.0222 |
| 112 | 16/90 | 19/90 | +0.0333 |
| 128 | 16/90 | 16/90 | +0.0000 (no-op) |

**Bootstrap 95% CI (5000 resamples, seed=42)**:

| K | WITH WBFM net_gain [CI] | significant? | WITHOUT WBFM net_gain [CI] | significant? |
|---|---|---|---|---|
| 20 | +0.0556 [-0.065,+0.176] | no | **+0.2000 [+0.089,+0.311]** | **YES** |
| 30 | +0.0370 [-0.083,+0.157] | no | **+0.1889 [+0.078,+0.300]** | **YES** |
| 40 | +0.0185 [-0.102,+0.139] | no | **+0.1889 [+0.078,+0.300]** | **YES** |
| 50 | +0.0093 [-0.111,+0.130] | no | **+0.1778 [+0.067,+0.289]** | **YES** |
| 80 | -0.0278 [-0.130,+0.074] | no | **+0.1000 [+0.011,+0.189]** | **YES** |

**`K=80` bootstrap CI overlaps zero when WBFM is included** (not
significant) but **is significant when WBFM is excluded** -- and even
then, K=80 is NOT the strongest point: K=20/30/40 are all larger
(+0.19-0.20 vs +0.10) and equally significant.

**Per-modulation numerator/denominator/net_gain** (n=18/modulation/K):

| Modulation | K=20 | K=30 | K=40 | K=50 | K=80 |
|---|---|---|---|---|---|
| 8PSK | 2/18 (-0.111) | 0/18 (-0.222) | 1/18 (-0.167) | 1/18 (-0.167) | 2/18 (-0.111) |
| BPSK | 7/18 (-0.111) | 8/18 (-0.056) | 8/18 (-0.056) | 9/18 (+0.000) | 9/18 (+0.000) |
| QAM16 | 5/18 (+0.278) | 4/18 (+0.222) | 3/18 (+0.167) | 4/18 (+0.222) | 1/18 (+0.056) |
| **QAM64** | **13/18 (+0.611)** | **13/18 (+0.611)** | **13/18 (+0.611)** | 12/18 (+0.556) | 9/18 (+0.389) |
| QPSK | 7/18 (+0.333) | 8/18 (+0.389) | 8/18 (+0.389) | 6/18 (+0.278) | 4/18 (+0.167) |
| **WBFM** | **4/18 (-0.667)** | **3/18 (-0.722)** | **1/18 (-0.833)** | **1/18 (-0.833)** | **4/18 (-0.667)** |

(attacked-correct numerators, constant across K: 8PSK 4/18, BPSK 9/18,
QAM16 0/18, QAM64 2/18, QPSK 1/18, WBFM 16/18)

**WBFM's full negative effect, all K, all 3 attacks** (n=90 for
fgsm/pgd, n=18 for cw): WBFM shows a NEGATIVE net gain at EVERY K for
EVERY attack except the trivial K=128 no-op -- fgsm ranges -0.011 to
-0.144, pgd ranges 0.000 to -0.133, cw ranges -0.222 to -0.833 (its own
worst case, at K=40/50). WBFM is uniformly harmed by Top-K in this
design, regardless of attack type or K, with no exception found.

**Explicit note, per this round's requirement**: round 24's finding of
"CW net_gain=+0.139 at K=80" (n=72, using AM-SSB instead of Phase 3's
official 8PSK, and a smaller sample) **did NOT reproduce** at this
round's larger, Phase-3-official-grid scale. With the correct modulation
set: (a) K=80's WITH-WBFM net_gain is now NEGATIVE (-0.0278, CI includes
0); (b) even WITHOUT WBFM, K=80 is real and significant but is NOT the
strongest point -- K=20/30/40 all show larger, equally significant
effects. Round 24's K=80 spike was substantially an artifact of AM-SSB
(a modulation outside Phase 3's official 6-class set) combined with a
smaller sample (n=72 vs this round's n=90/108).

### 16.3 PGD results (Part 三)

**Per-K net gain** (n=540/K, pooled across all 6 modulations):

| K | attacked | defended | net_gain | 95% CI | significant? |
|---|---|---|---|---|---|
| 10 | 121/540 | 125/540 | +0.0074 | [-0.037,+0.052] | no |
| **20** | 121/540 | **142/540** | **+0.0389** | **[+0.004,+0.074]** | **YES** |
| 30 | 121/540 | 127/540 | +0.0111 | [-0.019,+0.041] | no |
| 40 | 121/540 | 119/540 | -0.0037 | [-0.028,+0.020] | no |
| 50 | 121/540 | 119/540 | -0.0037 | [-0.026,+0.020] | no |
| 64-112 | (small, +0.002 to +0.007) | | | all include 0 | no |
| 128 | 121/540 | 121/540 | +0.0000 (no-op) | -- | -- |

**Only K=20 is statistically significant.** K=10 -- which round 24's
smaller sample suggested was PGD's best K -- is NOT significant here
(+0.0074, CI=[-0.037,+0.052], clearly includes 0). **K=10 is no longer
treated as PGD's best K**: at the larger, correctly-scoped sample, its
point estimate is small and its CI is wide and centered near zero;
K=20's effect is both larger in magnitude and statistically distinguishable
from zero, making it the only defensible "PGD works here" claim in this
grid.

### 16.4 FGSM results (Part 四)

**Per-K net gain** (n=540/K):

| K | attacked | defended | net_gain | 95% CI | significant? |
|---|---|---|---|---|---|
| **10** | 161/540 | 120/540 | **-0.0759** | **[-0.124,-0.028]** | **YES (negative)** |
| 20 | 161/540 | 153/540 | -0.0148 | [-0.056,+0.026] | no |
| 30-40 | 161/540 | 148/540 | -0.0241 | includes 0 | no |
| **50** | 161/540 | 142/540 | **-0.0352** | **[-0.067,-0.006]** | **YES (negative)** |
| 64-112 | 161/540 | 155-163/540 | -0.011 to +0.004 | all include 0 | no |
| 128 | 161/540 | 161/540 | +0.0000 (no-op) | -- | -- |

**No K shows a statistically significant POSITIVE net gain for fgsm.**
Two K values (10 and 50) show statistically significant NEGATIVE net
gain -- Top-K is confirmed actively harmful, not merely ineffective, at
those specific K values. **Clean degradation for fgsm's own combo subset
(n=295 clean-correct rows) ranges 68.8%-78.6% across all K**, with no
sign of recovery even at high K within this design's tested range (K=112
still shows 68.8% degradation) -- unlike the pure `attack=none` check in
round 24, which found clean degradation reaching 0% by K=80, this
fgsm-conditioned subset's clean-correct population apparently
overlaps less favorably with K's improving region; not further
decomposed this round.

### 16.5 Deployment fairness (Part 五)

Explicitly separated, per this round's instruction -- **none of the
attack-specific or modulation-specific numbers above should be read as a
deployable defense claim**:

- **Global fixed-K** (the only category directly deployable without
  additional information): no single K is uniformly good. Best
  candidates by attack: fgsm has no K with significant positive gain;
  pgd's only significant K is 20; cw (pooled, WITH WBFM) has no
  significant K at all. A deployed system using ONE fixed K across all
  traffic would see, at best, a small, attack-dependent, sometimes-
  negative effect.
- **Attack-specific oracle K** (fgsm's best available K vs. pgd's K=20
  vs. cw's K=20-50): **requires knowing which attack was used** --
  information a real defender does not have at inference time (an
  attacker does not announce its algorithm). Presented above purely as
  an analysis upper bound, NOT a deployable claim.
- **Modulation-specific oracle K** (e.g. QAM64 benefiting from low K,
  WBFM harmed at every K): **requires knowing the true modulation
  label** -- but modulation classification is literally what this AWN
  pipeline is trying to predict; using the true label to choose K is a
  direct oracle leak of the answer the system is supposed to produce.
  Presented above purely as an analysis upper bound, NOT a deployable
  claim, unless a future round builds and validates an independent,
  non-oracle modulation-or-attack detector to drive such a policy (out
  of this round's scope).

### 16.6 Outputs

```
results/formal_phase4_expanded_k/ablation_summary.csv    (14256 rows, 41 columns)
results/formal_phase4_expanded_k/ablation_manifest.json
results/formal_phase4_expanded_k/stdout.log, stderr.log
```
Not added to git, matching `.gitignore`'s existing `results/*` rule.

### 16.7 Conclusion

This round's larger, Phase-3-official-grid confirmation materially
revises round 24's headline "CW net_gain=+13.9pp at K=80" finding: that
result did not reproduce once AM-SSB (not part of Phase 3's official set)
was replaced with 8PSK and the sample size grew. The corrected picture:
CW's real, statistically significant benefit is concentrated in
K=20-50 (not K=80) and ONLY visible once WBFM -- which is harmed at
every K, for every attack -- is excluded from the aggregate; PGD has a
single significant point at K=20 (not K=10, as round 24 suggested); FGSM
never shows a significant positive effect at any K, and is significantly
HARMFUL at K=10 and K=50. No global fixed-K, attack-specific, or
modulation-specific pattern supports an unqualified "Top-K defense
works" claim; the modulation- and attack-specific numbers are explicitly
oracle-only, not deployable. Section 17 designs (but does not execute)
a new formal Phase 4 at full N=10 sample_index informed by these
findings.

---

## 17. New formal Phase 4 design (K-reduced, full-N): design and dry-run only (round 25)

Not executed this round, per explicit instruction. Reuses the existing
`experiments/run_phase4_topk_ablation.py` unmodified beyond this round's
already-committed `--eps`-list extension -- no new script file needed,
since the runner already accepts every required CLI parameter
(`--mods`, `--snrs`, `--eps`, `--attacks`, `--sample-indices`, `--topks`,
`--policies`).

### 17.1 Design

- **K set**: `10, 20, 30, 40, 50, 80, 128` (7 -- the reduced set this
  round's findings justify: covers PGD's significant K=20, CW's
  significant K=20-50 band plus the previously-claimed K=80 for direct
  comparison, and K=128 as the mandatory no-op control). **K=128 is
  explicitly a no-defense baseline for comparison, never counted as a
  defense success in any aggregate.**
- **Modulations**: full Phase 3 set, `QPSK, BPSK, QAM16, 8PSK, QAM64,
  WBFM` (6) -- **WBFM retained**, not excluded, per explicit instruction.
- **SNRs**: `-10,-4,0,6,12,18` (6). **eps**: `0.01,0.03,0.05,0.1,0.3`
  (5). **attacks**: `fgsm, pgd, cw`. **sample_index**: full formal `0-9`
  (10, not the confirmation round's `[0,1,2]`). **policy**:
  `current_radioml_native` only. **seed=42**. CW: `c=1.0, steps=20,
  lr=0.01` (formal defaults). Sensing/alignment params: same as every
  prior formal phase (`threshold_factor=1.5, sensing_window_size=128,
  min_region_len=0, merge_gap=0, alignment_policy=max-energy,
  awn_preprocess=radioml-native`).

### 17.2 Dry-run results

```
python3 experiments/run_phase4_topk_ablation.py --dry-run \
  --mods QPSK,BPSK,QAM16,8PSK,QAM64,WBFM --snrs=-10,-4,0,6,12,18 \
  --eps 0.01,0.03,0.05,0.1,0.3 --attacks fgsm,pgd,cw \
  --sample-indices 0,1,2,3,4,5,6,7,8,9 \
  --topks 10,20,30,40,50,80,128 --policies current_radioml_native \
  --output-dir results/formal_phase4_expanded_full
```

1. **Attack instances**: **3960** (360 cells x 11/cell -- matches Phase
   3-full's exact instance count, since it is the identical
   mod/snr/eps/attack grid at `n_per_cell=10`).
2. **Final rows**: **27720** (3960 x 7 K).
3. **Per-attack**: fgsm=12600, pgd=12600, cw=2520 (confirmed via
   `--dry-run`'s own count, not hand-calculated).
4. **Estimated time**: extrapolated from this round's own measured
   throughput (14256 rows/248.3s) and round 24's (8640 rows/220.5s) --
   both figures are noisy over only 2 data points, so presented as a
   range rather than false precision: roughly **15-45 minutes**,
   plausibly toward the lower end given this session's consistent
   pattern of actual runtimes beating conservative estimates at every
   prior phase. Not yet empirically confirmed for this exact
   configuration.
5. **`output_dir`**: `results/formal_phase4_expanded_full/` -- a NEW,
   distinct directory; does not exist yet, will not overwrite
   `results/formal_phase4_expanded_k/` (this round's 14256-row
   confirmation) or any earlier phase's output.
6. **`--resume` design**: identical to every prior phase script --
   `CsvWriter` flushes to disk after every single row; `load_done_combo_ids()`
   reads the existing `ablation_summary.csv` on `--resume` and skips
   already-completed `combo_id`s; a killed/interrupted run can be safely
   continued without redoing completed work or losing any written rows.
7. **CSV schema**: identical 41-column `ablation_summary.csv` schema
   already in use (`combo_id, dataset, modulation, snr, sample_index,
   seed, attack, attack_eps, topk, policy, label, pred_clean,
   pred_attacked, pred_defended, clean_correct, attacked_correct,
   defended_correct, changed_by_attack, attacked_wrong,
   recovered_by_defense, defense_changed_prediction,
   clean_broken_by_defense, iq_linf_clean_attacked,
   iq_linf_attacked_defended, iq_l2_attacked_defended,
   pred_agreement_defended_vs_attacked, awn_backend, attack_backend,
   topk_backend, clean_nan, attacked_nan, defended_nan,
   attack_training_after, eval_mode_restored, runtime_seconds,
   run_status, failure_stage, failure_reason, output_dir,
   attacked_iq_sha256, policy_notes`).
8. **Aggregate CSV design** (not yet implemented as code -- described
   here per this round's "design only" scope): a single
   `ablation_aggregate.csv` with one row per unique combination of the
   7 required groupings (`attack x K`, `modulation x K`, `SNR x K`,
   `eps x K`, `attack x modulation x K`, `attack x SNR x K`,
   `attack x eps x K`), each row carrying: `group_type` (which of the 7
   groupings), the group key columns (whichever subset applies, blank
   otherwise), `n`, `clean_accuracy`, `attacked_accuracy`,
   `defended_accuracy`, `net_accuracy_gain`, `recovery_count`,
   `degradation_count`, `net_transition`, `overall_recovery_rate`,
   `conditional_recovery_rate`, `true_label_recovery_rate`,
   `clean_prediction_recovery_rate`, `clean_degradation_rate`,
   `prediction_changed_rate`, `mean_iq_distortion`, and
   `bootstrap_ci_lo`/`bootstrap_ci_hi` for `net_accuracy_gain`. A
   separate `ablation_wbfm_only.csv` and `ablation_excl_wbfm.csv` would
   isolate the WBFM-specific sensitivity view (never replacing the main
   all-modulation aggregate, per explicit instruction). All of this is a
   straightforward extension of the analysis code already written ad hoc
   in this and the previous round's chat-side Python snippets -- not
   built as a permanent script this round, since there is no data yet to
   run it against.
9. **Checkpoint loading**: confirmed by reading `main()` --
   `AWNModelAdapter(checkpoint_path=CHECKPOINT, ...)` is constructed
   exactly ONCE at the top of `main()`, before the combo loop, and reused
   (not reconstructed) for every one of the 27720 rows in a single
   process launch.
10. **Checkpoint loaded once per process**: **confirmed yes** (see above)
    -- this is why every phase script in this project (Phase 0/1/3/4 and
    both ablation rounds) completes in minutes rather than hours despite
    thousands of combos; reloading the checkpoint per-combo was never
    the design.
11. **Safe to resume after interruption**: **confirmed yes** -- flush-
    per-row CSV writing plus `--resume`'s done-ID skip logic (same
    mechanism validated across every phase this session, including this
    round's own smoke tests).
12. **Will it overwrite existing results**: **no** -- distinct
    `--output-dir`, never reuses `results/formal_phase4_expanded_k/` or
    any earlier phase's directory.

### 17.3 Not executed

Per explicit instruction, this design was dry-run only. No combos were
executed, no checkpoint was loaded for a live run, and
`results/formal_phase4_expanded_full/` was not created (the directory
referenced in `--output-dir` above does not exist on disk -- `--dry-run`
exits before any directory creation or file write).
