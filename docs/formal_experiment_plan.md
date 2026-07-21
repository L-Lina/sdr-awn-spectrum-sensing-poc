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
