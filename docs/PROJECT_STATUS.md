# Project Status

Single-source progress summary for `sdr-awn-spectrum-sensing-poc`. This is
the **project-close handoff document** -- Part 1 below is the current,
authoritative status; Part 2 (further down) is the preserved detailed
history of Phase 0-4 as written at round 28 (unmodified except where noted).
Language note: this document and `README.md` are the repo's English-language
engineering/handoff documents, a pre-existing convention; the formal
research findings live in Traditional Chinese under `docs/research/*.md`
(see `docs/research/README.md` for the reading order) -- this split predates
this round and is intentional, not an inconsistency to "fix" by translating
this file.

Everything in Part 1 is sourced to actual files/results directories on disk,
re-checked this round via direct file reads, `pandas`, and background
research passes over the current repo state -- not carried over from any
prior chat summary uncritically.

---

# Part 1 -- Project-Close Handoff Summary

## 1. Project Objective

Connect an SDR-style spectrum-sensing front end to the AWN (Adaptive
Wavelet Network) Automatic Modulation Classification model, then formally
characterize: (a) how much accuracy the sensing front end costs vs. an
oracle slice, (b) how effective adversarial attacks (17 registered, not
just FGSM/PGD/CW) are against the pipeline, (c) whether a fixed-K FFT Top-K
defense is deployable, (d) whether the pipeline accepts real captured IQ
files (`.cfile`) and exposes its own parameters correctly, and (e) whether
the same pipeline, wrapped in a satellite-like channel model (AWGN,
amplitude scaling, CFO, Doppler, timing offset), still functions and
produces internally consistent, reproducible results -- as a project-close
proof-of-concept, not a deployed or standards-validated system.

## 2. Final System Architecture

```
IQ source (synthetic stream | RadioML2016.10a sample embedded in noise | .cfile read)
  -> [optional] satellite-like channel (src/channel/satellite_like.py:
       amplitude scaling -> timing offset -> CFO/Doppler rotation -> AWGN)
  -> energy detection (src/sensing/energy_detection.py: sliding-window
       power -> median noise-floor -> threshold -> binary mask)
  -> occupied-region merge / min-length filter
  -> alignment-aware segmentation (naive | max-energy window selection)
  -> AWN preprocessing (radioml-native | legacy-unit-power) -> [N,2,128]
  -> real AWN inference               (src/adapters/awn_adapter.py)
  -> real adversarial attack (optional, 17 registered)  (src/adapters/attack_adapter.py)
  -> real Top-K FFT defense (optional)                  (src/adapters/topk_adapter.py)
  -> defended AWN inference (same adapter, reused)
  -> per-row CSV + manifest + charts
```

Every adapter has a numpy-only dummy fallback for when torch/the pinned
submodules aren't importable; every formal/citable result in this document
was produced through the **real** backend end-to-end, verified per-row via
`*_backend`/`status`/`fallback_used` columns in the output CSV, never
inferred or assumed.

## 3. Supported Input Sources

| Source | Status | Evidence |
|---|---|---|
| Synthetic (numpy-generated noise+burst) | **COMPLETE** | `src/sensing/iq_source.py`; CLI default `iq_source=synthetic`. |
| RadioML2016.10a (offline `.pkl`, real recorded samples embedded in synthetic noise) | **COMPLETE** | `src/sensing/radioml_source.py`; used for every formal Phase 0-4 result and the satellite-like Step 3/4 work. Not a live capture -- loads a fixed `[2,128]` sample and synthetically embeds it in a longer noise stream. |
| `.cfile` (complex64 / interleaved float32 / interleaved int16) | **COMPLETE_WITH_LIMITATION** | See section 9. Wired into the real pipeline and smoke-tested; not yet validated against an actual SDR/GNU-Radio capture file. |
| Live SDR / USRP / GNU Radio streaming | **NOT_IMPLEMENTED** | No `gnuradio`/`uhd`/`zmq` import anywhere in `src/`, `experiments/`, or `scripts/` (confirmed by grep this round); `docs/DEPLOYMENT_READINESS.md` documents this gap in detail. |

## 4. Spectrum Sensing

**Status: COMPLETE (single-channel, time-domain, offline/batch)**, **streaming variant: PROTOTYPE**.

Sliding-window power smoothing -> median noise-floor estimate ->
`threshold_factor`-scaled threshold -> binary occupied mask ->
region merge (`merge_gap`) -> min-length filter (`min_region_len`) ->
`naive`/`max-energy` window selection. Exact formulas and code paths are
documented in `docs/research/CURRENT_SYSTEM_AND_COMPONENT_STATUS_ZH_TW.md`
section 5.1 (kept as the canonical description; not duplicated here).

**Four-path validation** (`experiments/run_spectrum_sensing_utility.py`,
`results/spectrum_sensing_utility_formal_20260727T021248Z/`, N=2200, real
AWN): `direct` (no-noise upper bound) 0.5973, `no_sensing` (fixed-position
crop, ground-truth-blind) 0.0977, `sensing` (real energy-detect pipeline)
0.5900, `oracle` (true-position crop) 0.5909. Paired significance:
`no_sensing` vs `sensing` differs hugely (McNemar p≈8.9e-239, as expected);
`sensing` vs `oracle` does **not** differ significantly (diff +0.0009,
p=0.754) -- sensing costs essentially nothing vs. an oracle crop in this
run. This sensing accuracy (0.5900) differs from the earlier Phase 1 number
(0.5805, section "Phase 1" below) because Phase 1 used a single fixed
`seed=42` for all 2200 combos (fixing burst position across the whole
grid), while this later, independent experiment varies burst position
per-instance -- a documented methodology difference, not a bug in either
run.

**Streaming/stateful variant** (`src/sensing/streaming_detector.py`):
**PROTOTYPE**. Maintains only a small cross-chunk tail buffer, not full
detector state -- no persistent event ID/timestamp/refractory tracking, no
running whole-stream noise-floor estimate (recomputed per chunk).
Empirically fails at small chunk sizes: `chunk_size=256` matched 0/3
offline-detected regions, `chunk_size=512` matched 1/3;
`chunk_size=1024`/`2048` matched 3/3 with small (0-4 sample) boundary
error (`results/performance_latency_20260818T010552Z/streaming_sensing_validation.csv`).
Root cause diagnosed (`experiments/diagnose_streaming_failure.py`): the
median-based noise-floor estimator breaks when a burst's footprint becomes
a large fraction of the chunk+carry buffer at small chunk sizes. **Not
fixed, no live I/O exists.**

## 5. AWN (Automatic Modulation Classification)

**Status: COMPLETE.** Real checkpoint (`external/adversarial-rf/2016.10a_AWN.pkl`,
pinned submodule, byte-identical model code to `external/AWN`), 11-class
RadioML2016.10a label mapping (`QAM16,QAM64,8PSK,WBFM,BPSK,CPFSK,AM-DSB,
GFSK,PAM4,QPSK,AM-SSB`, `src/sensing/radioml_source.py:RML2016_10A_CLASSES`,
cross-checked against `docs/radioml_class_mapping.csv`), `[N,2,128]` input
conversion, eval-mode restoration verified after every attack call (clean
logits reproducible to ≤1e-4 max diff before/after). Model-confidence
calibration and a low-confidence abstention mechanism are **NOT_IMPLEMENTED**.

## 6. Attack (17-attack registry)

**Status: COMPLETE for A0 digital white-box execution; batching-safety
verified for only 3 of 17.**

All 17 requested attacks (`fgsm, pgd, cw, bim, mifgsm, difgsm, vmifgsm,
vnifgsm, rfgsm, tpgd, deepfool, fab, square, apgd, apgdt, autoattack, ead`)
are wired into `src/adapters/attack_adapter.py` and pass
`experiments/run_attack_compatibility_smoke.py`'s full acceptance criteria
(0 fallback, 0 NaN/Inf, eval-mode restored, clean-logits reproducibility,
correct output shape, genuinely nonzero perturbation) --
**17/17 PASS**, `results/attack_compatibility_smoke_20260727T030223Z/`.
`difgsm` required a custom `src/adapters/iq_difgsm.py:IQDIFGSM`
reimplementation (`torchattacks.DIFGSM`'s own `input_diversity()` assumes a
2D-image last-dimension and crashes on this repo's `[N,2,T,1]` layout) --
covered by its own 6-test unit suite (`experiments/test_iq_difgsm.py`), but
**not proven formally equivalent** to the original image-domain DIFGSM in
every respect. `docs/ATTACK_NAME_MAPPING.md` and
`docs/ATTACK_COMPATIBILITY_WORKLIST.md` were updated this round with the
17/17 PASS status (both had a stale `difgsm=NEEDS_CUSTOM_IMPLEMENTATION`
line left over from before the `IQDIFGSM` fix landed).

**Batching-safety classification** (only established for 3 attacks,
`src/adapters/attack_adapter.py:AttackAdapter.apply` docstring,
`docs/research/PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md` sections 9/15):
- **fgsm**: `implementation_optimization` -- bit-identical output at N=1 vs N>1.
- **pgd**: `implementation_optimization` **only when `attack_params={"random_start": False}`** (bit-identical, 0.0 max diff, 60-sample paired test). Left at torchattacks' own default `random_start=True`, PGD is stochastic per-call regardless of batch size -- this is documented as PGD's own randomness, not a batching bug, and must not be reported as `implementation_optimization`.
- **cw**: `batched_algorithmic_variant`, explicitly **not** `implementation_optimization` -- torchattacks' CW uses a whole-batch-summed early-stop cost, so batch_size changes the optimization trajectory (60-sample test: 95.0% prediction match, max diff 0.00138 vs a batch_size=1 baseline).
- **The other 14 attacks (bim, mifgsm, difgsm, vmifgsm, vnifgsm, rfgsm, tpgd, deepfool, fab, square, apgd, apgdt, autoattack, ead) have NOT been individually verified for batching safety** and must not be assumed `implementation_optimization` just because `AttackAdapter.apply()` accepts N>1 for them without raising.

## 7. Top-K Defense

**Status: COMPLETE (implementation/function); COMPLETE_WITH_LIMITATION
(effectiveness -- condition-dependent, not a universally effective
defense, by evaluated result, not by missing code).** Fixed-K FFT
denoising (`src/adapters/topk_adapter.py` -> `fft_topk_denoise`), K
verified functional across `{10,20,30,40,50,80,128}` (Phase 4) and K=20
(satellite-like Step 4, matching Phase A's reference K). Two independent
formal effectiveness evaluations both conclude the same thing from
different angles:
- **Phase 4** (RadioML2016.10a, no channel impairment, `results/formal_phase4_expanded_full/`, N=3960 attack instances/27720 rows): global fixed-K shows **no statistically significant net accuracy benefit at any tested K**, and is significantly net-harmful at K=10/40/50 (bootstrap 95% CI excludes 0). Attack-specific/modulation-specific positive effects exist (CW K=20-50 excl.-WBFM, QAM64 K=10-50) but are oracle-conditioned, not deployable without a non-oracle detector.
- **Satellite-like Step 4** (`results/satellite_like_final_20260821T021117Z/`, N=576, satellite-like channel + FGSM/PGD-det): recovery rate 2.50% (FGSM, 2/80) / 6.90% (PGD-det, 6/87) of attack-failed samples, and clean-accuracy degradation 29.03% (9/31) of originally-correct clean samples (correct denominators, see section 12/13 below).

**Neither the fixed-K knee-based variants (`adaptive_k_defense`,
`adaptive_k_v2_defense`, present in `external/adversarial-rf/util/defense.py`)
nor a routing/selection mechanism between K values or defense branches are
wired into `TopKAdapter` -- NOT_IMPLEMENTED.**

## 8. Parameterization

Two separate, non-overlapping validation rounds exist:

1. **Core CLI/config parameters** (`docs/parameter_validation.md`,
   `docs/parameter_validation.csv`, 94 rows): sensing params
   (`threshold_factor`, `window_size`, `min_region_len`, `merge_gap`),
   `snr`, `attack`/`attack_eps`, `topk`, dataset filters. Status-count
   breakdown of the `implementation_status` column: `passed`=45,
   `implemented`=12, `not_applicable`=12, `implemented_no_cli`=4,
   `not_implemented`=5, `current_session_tested`=3, `not_tested`=3,
   `not_wired`=2, `partial`=1, `not_started`=1 (remainder are
   comma-parsing artifacts in the CSV, not additional distinct statuses).
   `--mod` is `partially_implemented`: it only perturbs a hash-derived
   carrier-frequency-offset cosmetically, no real symbol-mapping/
   constellation change.
2. **A later, separate round** (`experiments/validate_pipeline_parameters.py`,
   `results/parameter_validation_20260727T054218Z/`) added and verified
   `--dataset` (fixed to RML2016.10a), `--mod-filter`/`--snr-filter`,
   `--samples-per-cell`, `--stream-length`, `--burst-insert-position
   {random,center,explicit}`, `--batch-size` (bit-identical predictions
   across batch sizes, verified), `--experiment-name`, `--overwrite`
   (refuses to clobber an existing `summary.csv` by default), and
   strict `[1,128]` boundary rejection for `--topk`. Final classification:
   71 `IMPLEMENTED_AND_VALIDATED`, 1 `NOT_APPLICABLE_FIXED_BY_BACKEND`
   (`apgdt.loss` -- the real `torchattacks.APGDT` has no such kwarg), 1
   `NOT_IMPLEMENTED` (`progress_logging`), 1 `DEFERRED_WITH_REASON`
   (`resume` -- real multi-combo resume lives at the batch-script layer,
   each `experiments/run_phase*.py`'s own `--resume`), 0
   `INVALID_OR_BROKEN`.

`torch_num_threads` is **not** part of either parameter-validation round
above -- it is characterized separately as a performance-tuning knob (see
section 10).

**Dataset-path portability (project-close cleanup, this round)**: the
formal CLI entry point (`run_full_experiment.py` / `src/utils/config.py`)
already required an explicit `--dataset-path` with no hardcoded fallback.
The ~31 standalone `experiments/*.py` batch scripts (Phase 0-4, performance,
attack-compatibility, satellite-like, etc.), however, each hardcoded
`DATASET_PATH = "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl"` as
a module-level constant, tying the repo to one VM. Fixed via a new shared
resolver, `src/utils/dataset_path.py:resolve_dataset_path()` /
`require_dataset_path_exists()`: priority is an explicit value (a script's
own `--dataset-path`, where wired) > the `SDR_AWN_DATASET_PATH` environment
variable > the historical hardcoded path, kept only as the last-resort
default (so the existing VM keeps working with no changes required).
Invalid paths fail fast (`FileNotFoundError`, no silent fallback to a
different dataset). All ~31 scripts now resolve `DATASET_PATH` through this
helper (env-var-overridable uniformly); the four satellite-like scripts
(`run_satellite_like_final.py`, `run_satellite_like_smoke.py`,
`diagnose_cfo_doppler_sanity.py`, `diagnose_satellite_channel_amplitude.py`)
additionally accept a real `--dataset-path` CLI flag and print the resolved
path at startup. The remaining ~27 historical Phase 0-4/performance/
attack-compatibility scripts were **not** individually given a bespoke CLI
flag this round (would mean touching each script's own, already-tested
argument parsing for zero behavioral gain on already-completed, frozen
results) -- they inherit the env-var override only. Two manifest-writing
scripts (`finalize_low_perturbation_results.py`,
`finalize_performance_results.py`) still write the historical hardcoded
path into a provenance dict recording what path an already-completed run
used -- left unchanged, since it documents historical fact, not a live
resolution point. No dataset format, sample selection, modulation, SNR,
checkpoint, or preprocessing semantics were touched -- path resolution
only. Validated via a path-level regression (not a full experiment): valid
explicit `--dataset-path` loads correctly; no flag falls back to the
unchanged legacy default (108-sample satellite-like smoke re-run,
0 error/no_region/NaN/fallback, matching the historical result); an invalid
`--dataset-path` fails fast before any backend loads, with no silent
fallback; the resolved path is printed at startup as run-time provenance.

## 9. `.cfile` Input

**Status: COMPLETE_WITH_LIMITATION.** `src/io/iq_file_source.py` supports
three on-disk formats (`complex64`, `interleaved_float32`,
`interleaved_int16`, explicit endianness, single-channel only), wired into
the formal pipeline via `--iq-source cfile` (`src/utils/config.py`,
`src/utils/pipeline.py:run_dry_run_experiment`) and into a standalone
`experiments/run_cfile_pipeline.py` entry point, both reaching the real
AWN/attack/Top-K backends. Verified: `experiments/test_iq_file_source.py`
(16 unit-test items) and `experiments/run_cfile_pipeline_smoke.py`
(`results/cfile_pipeline_smoke_20260727T082623Z/`) -- all three formats
produce identical region/segment counts and identical clean predictions on
the same underlying data; attack (fgsm/cw/difgsm) and Top-K (K=10/20/128)
smoke passed; `n_error_total=0`, `n_fallback_total=0`, `n_nan_inf_total=0`.

**Limitation, unchanged from `docs/DEPLOYMENT_READINESS.md`'s core finding**:
this was verified only against internally generated equivalence/regression
fixture files, **never against a real SDR/GNU-Radio-captured `.cfile`** --
whether real-capture amplitude/gain/quantization/noise characteristics
match what this pipeline was validated against is unverified. No
sample-rate-aware timing logic exists downstream; the whole file is loaded
into RAM (no streaming/chunked read for long captures).

## 10. Performance / Acceleration

Evidence: `docs/research/PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md`,
`results/performance_latency_20260818T010552Z/`,
`results/end_to_end_latency_20260818T062625Z/`.

- **Clean AMC stage latency** (n=2200): mean 11.460ms, median 2.819ms,
  p95 55.094ms, p99 199.322ms -- heavy tail traced to thread contention,
  not signal content.
- **End-to-end scenario latency** (n=24, baseline threads=default/batch=1):
  clean 4.074ms mean; +Top-K 9.284ms; +FGSM 12.128ms; +PGD(det) 57.690ms;
  +PGD(stoch) 42.448ms (non-reproducible run-to-run, explicitly caveated);
  +FGSM+Top-K 20.772ms.
- **FGSM acceleration** (batch_size=16, torch_num_threads=1): attack-only
  13.83ms -> 0.86ms (16.10x); end-to-end mean 12.128ms -> 3.803ms (3.19x).
  Confirmed `implementation_optimization` (100% prediction match).
- **PGD acceleration, `random_start=False`** (deterministic): attack-only
  51.314ms -> 5.382ms (9.53x); end-to-end mean 57.690ms -> 7.844ms
  (7.35x). Bit-identical batch_size=1 vs 16 output (60-sample test).
- **PGD acceleration, `random_start=True`**: end-to-end mean 42.448ms ->
  7.944ms (5.34x) -- **throughput-only claim, NOT a per-sample equivalence
  claim** (stochastic).
- **CW acceleration**: attack-only 99.99ms -> 9.32ms (10.73x) --
  `batched_algorithmic_variant`, not comparable to FGSM/PGD's
  `implementation_optimization` class (see section 6).
- **Thread tuning**: `torch_num_threads=2` measured optimal on the test
  machine (mean 0.551ms/call), default (16 threads) measured worst (mean
  10.062ms, p95 39.444ms, 12.2% outlier rate) -- explicitly caveated as
  machine-specific, not a generalizable claim.

## 11. Satellite-like Scenario (Step 1-4)

Four-step, project-close research track answering whether the same formal
pipeline still functions and produces internally consistent results under
a satellite-like channel model. All four steps and the subsequent audits
are documented in Traditional Chinese under `docs/research/` (see
`docs/research/README.md` for the reading order); summarized here for the
handoff record only:

- **Step 1**: `SATELLITE_APPLICATION_AND_LATENCY_REQUIREMENTS_ZH_TW.md` --
  3GPP TS 22.261 NTN latency reference, DVB-S2/S2X modulation family
  survey, A0/A1/A2 threat-model mapping, MUST/SHOULD/OPTIONAL channel
  factor classification. Research/requirements survey, no code.
- **Step 2**: `SATELLITE_DATASET_AND_MODULATION_FEASIBILITY_ZH_TW.md` --
  decided project-close **Strategy A**: RadioML2016.10a, BPSK/QPSK/8PSK
  subset, no retraining, no RadioML2018.01A download this round.
- **Step 3**: `SATELLITE_LIKE_CHANNEL_SIMULATOR_DESIGN_ZH_TW.md` +
  `src/channel/satellite_like.py` -- implements the MUST factors (AWGN,
  amplitude scaling, propagation-delay metadata) and SHOULD factors (CFO,
  Doppler, timing offset) only, no OPTIONAL factors, no full DVB modem, no
  SDR. 29/29 unit tests (`experiments/test_satellite_like_channel.py`),
  108-sample smoke test (`results/satellite_like_smoke_20260821T010645Z/`,
  0 error/fallback/NaN). A focused root-cause validation round then proved
  amplitude/CFO/Doppler accuracy sensitivity is a genuine AWN-preprocessing
  covariate-shift effect (the existing, unmodified `radioml-native` policy
  does no rescaling), **not** a channel-simulator bug -- confirmed via
  achieved-SNR measurement, oracle-vs-sensing crop comparison, and
  measured-frequency-shift verification.
- **Step 4**: see section 12/13 below.
- **Final result audit + reporting-consistency cleanup** (two subsequent
  rounds, read-only, never modified `raw_results.csv`): re-derived every
  core number independently from `raw_results.csv`
  (`experiments/audit_satellite_like_final.py`,
  `results/satellite_like_final_20260821T021117Z/audit/`), found 0
  numerical bugs (0 duplicate/missing combos, 0 fairness-hash mismatch, 0
  attack-tensor reuse), but found and corrected two **metric-naming/
  denominator wording** issues in the research document (not the data):
  the legacy `attack_success` field is a prediction-change indicator, not
  a traditional conditional-adversarial-success rate; and the original
  Top-K recovery/degradation percentages used an inflated denominator
  (all Top-K-ON rows) instead of the eligible-sample-only denominator. See
  section 13 for the corrected numbers.

## 12. Final 576-Combination Experiment (Step 4)

`experiments/run_satellite_like_final.py` /
`experiments/analyze_satellite_like_final.py`,
`results/satellite_like_final_20260821T021117Z/`. Matrix:
$3\text{ mod}\times4\text{ SNR}\times4\text{ channel conditions}\times3\text{ attacks}\times2\text{ topk}\times2\text{ idx}=576$,
verified as 576 unique combinations (0 duplicate, 0 missing) both at
generation time and again independently in the audit round. 3
modulations (BPSK/QPSK/8PSK), 4 SNR (-10/0/10/18 dB), 4 channel conditions
(clean/mild/moderate/strong, full parameters in
`SATELLITE_LIKE_FINAL_EXPERIMENT_ZH_TW.md` section 6), 3 attacks
(none/optimized-FGSM/optimized-deterministic-PGD), Top-K on/off (K=20).
Fairness: the same channel-transformed IQ and clean crop are computed once
per base combo and reused, hash-verified, across every attack/Top-K
branch (96 unique base combos, `channel_input_hash`/`clean_segment_hash`
100% coverage, 0 inconsistency, 0 cross-base collision). **0 error / 0
fallback / 0 NaN-Inf / 0 no-region / 0 no-segment across all 576 rows.**

## 13. Final Verified Metrics (Step 4, audit-corrected)

All numbers below traced to `results/satellite_like_final_20260821T021117Z/`'s
`raw_results.csv` and its audit-corrected summary CSVs (`overall_summary.csv`,
`by_channel.csv`, `audit/unique_attack_sample_audit.csv`,
`audit/topk_denominator_audit.csv`) -- not hand-derived.

- **Sensing**: detection rate 100% (96/96 unique base combos);
  `captured_signal_ratio` mean 0.9734 (0.9746/0.9743/0.9740/0.9707 by
  channel severity, near-flat).
- **Clean AMC accuracy**: overall 32.3% (n=96 unique base); by channel
  58.3%/33.3%/20.8%/16.7% (clean/mild/moderate/strong, monotonic
  decrease); by modulation 8PSK 46.9% / QPSK 37.5% / BPSK 12.5%.
- **GIGO check**: `captured_signal_ratio` stays within a 0.0039 range
  across channel severities while clean accuracy swings 0.4167 -- evidence
  is consistent with (not proof of) the accuracy drop being a
  channel/model robustness effect rather than a sensing failure.
- **FGSM** (n=96 unique attacked base): Attacked Accuracy 16.67% (16/96);
  Prediction Change Rate 91.67% (88/96); Conditional Attack Success Rate
  77.42% (24/31, i.e. of the 31 base samples clean-correct to begin with).
- **PGD(det)** (n=96): Attacked Accuracy 9.375% (9/96); Prediction Change
  Rate 98.96% (95/96); Conditional Attack Success Rate 96.77% (30/31).
- **Top-K, correct denominators**: FGSM recovery 2.50% (2/80
  attack-failed samples); PGD(det) recovery 6.90% (6/87); clean-accuracy
  degradation 29.03% (9/31 originally-correct clean samples) -- Top-K is
  **condition-dependent, not a universally effective defense** in this
  setting.
- **Latency** (n=96 per scenario): clean median 2.48ms/p95 4.13ms/p99
  7.09ms; FGSM median 3.79/p95 5.53/p99 8.43; PGD(det) median 7.56/p95
  9.13/p99 12.44; FGSM+Top-K median 7.39/p95 13.42/p99 30.87; PGD(det)+Top-K
  median 11.22/p95 17.09/p99 38.62. Cross-validates against section 10's
  independently measured FGSM/PGD figures (same optimization, different
  sample set, same order of magnitude).
- **amplitude_scale interpretation**: a receiver-side digital IQ
  amplitude/gain-scaling robustness stress condition -- **not** a full RF
  link-budget or path-loss simulation (no AGC, no antenna gain, no
  free-space-loss model exists in `src/channel/satellite_like.py`).

## 14. Reproducibility

- Deterministic seeding throughout (`channel_seed = SEED + sample_index`,
  fixed `SEED=0`), timestamped result directories
  (`results/<name>_<UTCtimestamp>/`), per-row provenance hashes
  (`channel_input_hash`, `clean_segment_hash`, `base_sample_id`).
- Every formal round asserts the real backend (`backend_name`/`status`)
  immediately after adapter construction and raises rather than silently
  falling back.
- `raw_results.csv` for the satellite-like Step 4 run was confirmed
  byte-for-byte unmodified across two subsequent audit rounds via SHA256
  comparison against its own `manifest_analysis.json`.
- Earlier Phase 0-4 rounds established `--resume`/incremental-CSV safety
  (byte-identical no-op on a completed run) and cross-run reproducibility
  spot-checks (16-combo independent-process bit-identical check, Phase 1).

## 15. Known Limitations

- Spectrum sensing is single-channel, time-domain only -- no wideband
  channelizer, no frequency-domain occupancy analysis.
- AWN output has no confidence calibration or abstention mechanism.
- Only 3 of 17 attacks have a verified batching-safety classification;
  the other 14 must not be assumed safe to batch for timing/equivalence
  claims.
- Top-K is a functionally complete, but not a universally effective,
  defense -- effectiveness is condition-dependent (K, attack, modulation,
  and now also channel severity) in every formal evaluation run so far.
- `.cfile` support has never been exercised on a real SDR-captured file.
- Streaming/stateful sensing is a prototype with a known, undiagnosed-fix
  cross-chunk state gap; not production-ready.
- No live SDR/USRP/GNU-Radio path, no real OTA experiment, no complete
  DVB-S2/S2X modem, no standards-compliant APSK extension exist in this
  repo.
- The satellite-like channel model is a project-close-scope simplification
  (MUST+SHOULD factors only); Doppler figures documented as
  simplified/order-of-magnitude references, not actual ground-terminal
  trajectories.
- WBFM's persistently low clean accuracy and QAM16's low direct/sensed
  agreement (Phase 1/3/4 era findings) remain unexplained, not
  investigated further this round.
- Phase 3's true 11-modulation x 20-SNR full sweep and Phase 5's optional
  11-modulation sensing-sensitivity expansion remain designed-not-run (see
  Part 2 below); Phase 6 (multi-burst extension) remains not executed with
  real backends.

## 16. Future Extension

- Live SDR/USRP/GNU-Radio streaming ingestion; real OTA validation in a
  shielded/cabled test setup (`docs/research/CURRENT_SYSTEM_AND_COMPONENT_STATUS_ZH_TW.md`
  section 10, routes A/B).
- Completing the cross-chunk streaming-sensing detector state (event
  ID/timestamp/refractory logic, running whole-stream noise floor).
- A non-oracle attack-identity or modulation detector, to make the
  oracle-conditioned Phase 4 Top-K findings (CW K=20-50, QAM64 K=10-50)
  actionable.
- RadioML2018.01A / broader modulation family (including APSK) migration,
  contingent on retraining or acquiring a compatible AWN checkpoint (an
  unverified `2018.01a_AWN.pkl` exists in `external/adversarial-rf/` as a
  future-extension starting point only, per Step 2's finding).
- Full DVB-S2/S2X-compliant modem/frame-level integration, if a
  standards-compliance claim is ever required.
- Verifying batching safety for the 14 not-yet-classified attacks.

## 17. Project-Close Status

This project's offline scope -- as defined in section 1 -- is complete and
has completed formal experimental verification. This is **not** a claim
that every conceivable future capability is complete: live RF/SDR
ingestion, real OTA validation, a full DVB-S2/S2X stack,
standards-compliant APSK support, and a production-grade streaming
detector are explicit future-extension items, outside this project-close
round's scope (section 16), not gaps in what was promised for this round.

**COMPLETE** (offline, real-backend, formally verified): offline formal
pipeline (synthetic + RadioML2016.10a); spectrum sensing (single-channel,
time-domain); AWN inference; 17-attack formal compatibility (A0 digital);
Top-K implementation/function; core parameterization (incl. dataset-path
portability, this round); `.cfile` formal input wiring; performance/latency
characterization; satellite-like channel simulator; the 576-combination
final integrated experiment; the final result audit (two independent
rounds, numerical + metric-definition correctness).

**COMPLETE_WITH_LIMITATION**: `.cfile` real-backend integration (internal
equivalence fixtures only, no real SDR/GNU-Radio capture file exercised
yet -- see section 9); Top-K effectiveness (condition-dependent across K,
attack, modulation, and channel severity in every formal evaluation so
far -- not a missing feature, an evaluated result -- see section 7).

**PROTOTYPE** (functions, explicitly not production-ready): stateful
streaming spectrum sensing (section 4).

**NOT_IMPLEMENTED / outside project-close scope**: live SDR/USRP/GNU-Radio
stream ingestion; real OTA satellite experiment; full DVB-S2/S2X modem
stack; standards-compliant APSK extension; a production-grade cross-chunk
stateful streaming detector; AWN confidence calibration/abstention;
adaptive/routed Top-K selection.

This document, `docs/research/README.md`, and
`docs/PROJECT_CLOSE_CHECKLIST.md` together are the intended entry points
for anyone picking this project up after close.

---

# Part 2 -- Detailed Phase 0-4 History (round 28, preserved)

Everything below this line is preserved from this document's original
(round 28) version, describing Phase 0-4 in detail. Not re-verified line
by line this round beyond the cross-checks cited in Part 1 above (e.g. the
sensing accuracy discrepancy noted in section 4); treat Part 1 as current
and this part as historical detail/evidence backing it.

## 0. This round's changes (round 28)

**What was actually done this round**: created this document
(`docs/PROJECT_STATUS.md`) by reading the current repo state directly --
`git log`, the full text of `docs/formal_experiment_plan.md` (all 19
sections), `docs/formal_experiment_matrix.csv` (all 11 rows), `docs/
parameter_validation.md` section 6, the adapter source files under
`src/adapters/`, and the actual contents of every `results/formal_*`
directory on disk (via `ls`/`wc -l`/`pandas`, not assumed from row counts in
the docs). No experiment was run, no existing result file was modified, no
code in `src/`, `experiments/`, `external/AWN`, or `external/adversarial-rf`
was changed.

**Verified this round, directly against the repo (not from memory)**:
- `git log`/`git status`/`git diff --stat` confirmed `HEAD=0cccc78`,
  working tree clean before this file was added, `main` in sync with
  `origin/main`.
- Every `results/formal_*` directory referenced in section 2 below was
  confirmed to actually exist on disk with the row/column counts stated
  (cross-checked with `pandas.read_csv` row counts and `.columns`, not just
  file listings).
- `results/formal_phase3_attack_reduced/phase3_summary.csv` (792 data rows,
  6 modulations) and `results/formal_phase3_attack_full/phase3_summary.csv`
  (3960 data rows, same 6 modulations) were both opened and their
  `modulation` column values compared directly.

**Discovered this round** (a genuine finding, not previously written down
in either `formal_experiment_plan.md` or `formal_experiment_matrix.csv`):
`formal_experiment_matrix.csv`'s `phase=3,tier=full` row (a distinct,
never-run, proposed 11-modulation x 20-SNR sweep, `status=
designed_not_run_optional`) names the same `output_dir` value
(`results/formal_phase3_attack_full/`) as the directory that actually holds
the completed, 6-modulation, N=3960 "full sample_index" run described in
`formal_experiment_plan.md` section 11. These are two different sweeps
sharing one directory name by what appears to be a documentation
coincidence/oversight in the matrix, not an error in the completed run
itself (`phase3_summary.csv`'s own modulation-coverage matches the
6-modulation grid the plan document describes, not an 11-modulation one).
Recorded as an open item in section 5/6 below; **not corrected in either
source CSV/doc this round**, since resolving it was not requested and doing
so without explicit confirmation of which row's `status`/`output_dir`
should change risked overwriting one of the two documents incorrectly.

**Additional discovery from this round's re-check pass** (requested by you
to specifically re-confirm Phase 0-6 status before commit): `results/
sensing_revalidation_after_alignment/` (Phase 5's evidence directory)
contains an `E4_single_vs_multi/` subdirectory with a small single-vs-
2-burst sensing check -- re-opened and read directly this round, confirmed
via its own `attack_notes` column that it used the **placeholder/dummy**
attack backend, not real AWN/attack/Top-K, and is not part of the formal
528-combo Phase 5 count. It is a precursor sensing-only check, not a
substitute for the formal Phase 6 design (which needs real backends across
60 combos and remains entirely unrun). Also re-verified this round: Phase 0
(128 rows/50 cols), Phase 1 (2200 rows/32 cols), Phase 4 reduced-tier (3168
rows/60 cols), and Phase 4 Expanded-K (14256 rows/41 cols, 1188 attack
instances) row/column counts, all read directly via `pandas`, all matching
what `formal_experiment_plan.md` states.

**Not done this round** (explicitly, to avoid any reader inferring
otherwise from this document's other sections): no Phase 0-6 experiment was
executed or re-executed; no aggregate CSV was regenerated; no push was
made; the file remains untracked (`git add` not run) pending your
confirmation.

---

## 1. Project goal and current full pipeline

**Goal**: connect an SDR-style spectrum-sensing front end to the AWN
(Automatic Modulation Classification) model, then formally characterize (a)
how much accuracy the sensing front end costs vs. an oracle slice, (b) how
effective FGSM/PGD/CW adversarial attacks are against the pipeline, and (c)
whether a fixed-K FFT Top-K defense is a viable, deployable countermeasure --
all using the real AWN model and real attack/defense implementations from the
two pinned submodules, not placeholders.

**Current pipeline** (as actually implemented in `src/`, distinct from the
older placeholder-stage description still in `README.md`):

```
complex IQ (synthetic OR real RadioML RML2016.10a sample, iq_source=radioml)
  -> energy detection              (src/sensing/energy_detection.py)
  -> occupied region extraction / merge-gap
  -> alignment-aware segmentation  (src/sensing/segmentation.py; max-energy or naive)
  -> AWN preprocessing             (radioml-native or legacy-unit-power)
  -> AWN input [N, 2, 128]         (to_awn_input)
  -> real AWN inference            (src/adapters/awn_adapter.py -> external/adversarial-rf/models/model.py:AWN,
                                     byte-identical to external/AWN/models/model.py at the pinned commits)
  -> real adversarial attack       (src/adapters/attack_adapter.py -> external/adversarial-rf/util/adv_attack.py:
                                     Model01Wrapper + torchattacks FGSM/PGD/CW)
  -> real Top-K FFT defense        (src/adapters/topk_adapter.py -> external/adversarial-rf/util/defense.py:fft_topk_denoise)
  -> defended AWN inference        (same AWNModelAdapter, reused)
```

Every adapter (`AWNModelAdapter`, `AttackAdapter`, `TopKAdapter`) has a
numpy-only dummy fallback for when torch/the submodules aren't importable;
all formal experiment rounds (Phase 0 onward) run through
`/home/xiaomi/adversarial-rf/.venv/bin/python`, which has torch installed, so
all formal results use the **real** backends end-to-end (verified per-row via
`*_backend` string columns in every phase's output CSV, never inferred).

`src/utils/pipeline.py:run_dry_run_experiment()` is the single-shot,
CLI-facing entry point (used by Phase 1). Phases 0 and 3/4 (which need the
same attacked IQ reused across multiple K values or multiple sample indices
without repeating the sensing+attack step) call the same underlying building
blocks directly instead, since `run_dry_run_experiment()` cannot express
literal in-memory reuse.

---

## 2. Completed phases -- status and actual results

### Phase 0 -- Pilot (real-backend mechanics check)
**已完成.** `experiments/run_phase0_pilot.py`, 128/128 combos `ok`, 0 error, 0
fairness violations. Output: `results/formal_pilot_phase0/` (128 rows, 50
cols). Purpose was mechanics/schema validation, not a citable scientific
result (N=8 samples per cell). See plan section 8.

### Phase 1 -- Spectrum Sensing baseline (+ Phase 2, direct-vs-sensed AMC)
**已完成.** `experiments/run_phase1_sensing_baseline.py`, 2200/2200 combos
`ok` (11 modulations x 20 SNRs x 10 sample_index), 0 error, 0
`sensing_failed`. Output: `results/formal_phase1_sensing_clean_amc/` (2200
rows, 32 cols). Runtime 92.1 min.

**Formal, citable results (N=2200):**
- Direct (oracle) AMC accuracy: **0.5973**
- Sensed end-to-end AMC accuracy: **0.5805**
- Gap (direct - sensed): **+0.0168**
- Direct/sensed prediction agreement: **0.9095**
- Detection probability: **1.0000**; false alarm region rate: **0.0043**;
  mean captured signal ratio: **0.9986**
- Phase 2 (direct-vs-sensed comparison) answered inline from the same data,
  no separate run was needed -- see plan section 9.3.

Reproducibility: 16-combo independent-process spot-check, bit-identical.

**Note (added post-round-28, see Part 1 section 4)**: a later, independent
"four-path" experiment (`spectrum_sensing_utility_formal`) measured a
different sensed accuracy (0.5900) under a per-instance-seeded burst
position, vs. this Phase 1 run's single fixed `seed=42`. Both numbers are
valid for their own methodology; they should not be quoted interchangeably.

### Phase 3 -- Adversarial attack effectiveness
**已完成** (both tiers that exist in `formal_experiment_matrix.csv`'s
`phase=3` rows were run):
- Reduced tier (`sample_index` 0-1, N=792): `results/formal_phase3_attack_reduced/`
- Full-N tier (`sample_index` 0-9, N=3960, same 6-modulation/6-SNR/5-eps
  grid as the reduced tier, just full sample count): `results/formal_phase3_attack_full/`,
  3960/3960 `ok`, 0 error, runtime 88.6 min. See plan section 11.

**Formal, citable results (N=3960):**
- Clean accuracy: **0.5889**; attacked accuracy: **0.2876**
- Overall attack success rate: **0.8278**; conditional (on clean-correct): **0.7993**
- Per-attack success: cw **0.9278** > pgd **0.8861** > fgsm **0.7494**
- New cross-tabulation findings (not yet explained, flagged as open):
  fgsm-specific non-monotonic eps=0.1 dip; BPSK's unusually low CW success
  rate (0.617 vs 0.983-1.000 elsewhere)

**Known documentation inconsistency (not resolved by this document):**
`formal_experiment_matrix.csv` also has a *separate* `phase=3,tier=full` row
describing a larger, **never-run** 11-modulation x 20-SNR sweep, whose
`status` field reads `designed_not_run_optional` and whose `output_dir`
field happens to name the same directory
(`results/formal_phase3_attack_full/`) that the actually-completed N=3960
6-modulation run above was written into. These are two different things:
the completed run is the 6-modulation "full-N" run described in plan
section 11; the true 11-modulation expansion has not been run and has no
separate output directory. Flagged here rather than silently resolved.

### Phase 4 -- Top-K defense effectiveness
**Layered history, all documented in plan sections 12-19; the current,
final formal result is the round-27 full-N run.**

1. **Reduced-tier execution** (`results/formal_phase4_defense_reduced/`,
   N=792/3168 rows, K in {10,20,30,40}) -- 已完成. Finding: clean-accuracy
   degradation (72-79%) vastly exceeds attack-recovery benefit (12-23%);
   net-harmful for 5/6 modulations.
2. **Root-cause analysis** (plan section 14) -- 已完成 (analysis only, no
   code change). No formula/fairness bug found; confirmed genuine
   churn-cancellation at K=20; found (but did not fix, and did not prove
   wrong) a normalization difference vs. the historical `AWN_All.py` usage.
3. **3-policy preprocessing ablation, K up to 128** (`results/
   formal_phase4_topk_ablation/`) -- 已完成. Policy A (current, unmodified)
   proven bit-exact identical to policy B (normalize/rescale); policy C
   (legacy `AWN_All.py` replication) confirmed out-of-distribution for this
   checkpoint. Surfaced an initial (later revised) CW K=80 finding that used
   a non-official modulation set (AM-SSB).
4. **Expanded-K Confirmation Experiment** (`results/formal_phase4_expanded_k/`,
   N=1188/14256 rows, Phase 3's official 6-modulation grid) -- 已完成. This
   **revised** the K=80 finding: CW's real, statistically significant
   benefit is K=20-50, visible only when WBFM is excluded; PGD's only
   significant K is 20; FGSM shows no significant positive K anywhere.
5. **New formal Phase 4 design** (K={10,20,30,40,50,80,128}, full N=10,
   WBFM retained) -- designed and dry-run in round 25 (plan section 17), then
   smoke-tested (`results/formal_phase4_expanded_smoke/`, round 26, plan
   section 18), **then formally executed this round (round 27)**.

**Formal, citable Phase 4 result (`results/formal_phase4_expanded_full/`,
N=3960 attack instances / 27720 rows, plan section 19):**
- 0 error / sensing_failed / fallback / NaN / Inf; 100% real backends; 100%
  eval-mode restoration; 100% cross-K fairness (hash-verified); K=128
  no-defense control verified (`pred_defended==pred_attacked` 3960/3960, max
  IQ Linf 1.86e-8); `--resume` re-run confirmed a byte-identical no-op.
- **Global fixed-K (the only directly deployable view, WBFM retained): no K
  shows a statistically significant net accuracy benefit; K=10, K=40, K=50
  are significantly net-harmful** (bootstrap 95% CI excludes 0).
- CW's positive effect (K=20-50, excl-WBFM) and a newly full-N-confirmed
  large QAM64-specific positive effect (K=10-50) both reconfirmed, but both
  are **oracle-conditioned** (on true attack identity / true modulation
  label respectively) and explicitly **not deployable claims**.
- WBFM harmed at every single K tested (7/7), all CI-significant negative.
- Aggregate CSVs (`experiments/analyze_phase4_expanded_full.py`, committed)
  written to `results/formal_phase4_expanded_full/aggregates/` (not in git,
  matches `.gitignore`).

### Not part of the original Phase 0-4 set but present in the matrix
- **Phase 5** (sensing parameter sensitivity): 已完成 via reuse of an
  earlier round's evidence (`results/sensing_revalidation_after_alignment/`,
  subdirectories `A_threshold_factor`(210) + `B_sensing_window_size`(150) +
  `C_min_region_len`(150) + `D_merge_gap`(18) = 528 combos, row counts
  re-verified directly from each subdirectory's CSV this round, pre-dates
  the Phase 0-4 numbering). An **optional 11-modulation elective expansion**
  is designed but not run (`designed_not_run_optional`). The same directory
  also contains `E1_burst_len`/`E2_n_samples`/`E4_single_vs_multi`/
  `E5_embed_snr_margin` subdirectories from the same round -- these are
  **not** part of the formal 528-combo count in `formal_experiment_matrix.csv`
  and are not written up as Phase 5 results in `formal_experiment_plan.md`;
  present on disk but not yet formally integrated.
- **Phase 6** (multi-burst extension, matrix row: `num_bursts=2`, real
  AWN+attack+Top-K, 60 combos): 尚未執行 -- `results/
  formal_phase6_multiburst_extension/` does not exist. A much smaller,
  **non-formal** precursor exists in the Phase-5 directory above
  (`E4_single_vs_multi/`, 1 single-burst + 2 multi-burst rows) but it
  explicitly used the **placeholder/dummy** attack backend (`attack_notes`:
  "--use-real-attack not passed; using placeholder"), not real AWN/attack/
  Top-K, and is not a substitute for the formal Phase 6 design.
- **Phase 4 "quick" tier**: 尚未執行, no results directory exists
  (`results/formal_phase4_defense_quick/` is absent on disk).

---

## 3. Currently verified parameters and functionality

(Consolidated from `docs/parameter_validation.md` section 6 and the formal
phases' own system-verification sections; see those documents for full
per-parameter detail.)

- Real-backend end-to-end execution (AWN + attack + Top-K, no dummy
  fallback) at scale: verified across every formal phase (0/1/3/4), always
  100% real per the `*_backend` string columns.
- `attack=none` bit-identical bypass; `attack=fgsm`/`pgd` end-to-end
  (including exact `eps` enforcement, re-verified at N=3960); `attack=cw`
  execution path and effectiveness at repo defaults (`c=1.0,steps=20,
  lr=0.01`) IS established as of Phase 3 (0.9278 success rate at full N=3960,
  synthetic-IQ). This differs sharply from `parameter_validation.md` section
  10.2's earlier "0/5 predictions changed" finding at the same defaults --
  that earlier test used hand-picked synthetic segments before RadioML/
  `radioml-native` mode existed in this repo, not the later formal pipeline.
  `formal_experiment_plan.md` section 7 (risk R4) explicitly flags the causal
  link between this and `radioml-native` mode as a **plausible, not
  confirmed, inference** -- not re-investigated since, so it is repeated
  here with the same caveat, not stated as settled fact.
- `--topk` reaches the real `fft_topk_denoise` function; behavior
  characterized across K in {10,20,30,40,50,80,128}, up to N=3960
  attack-instances (Phase 4 round 27).
- Fair Top-K reuse (same attacked IQ across all K values for one attack
  instance): verified via SHA256 hash-chain equality at every phase from
  Phase 0 through the round-27 formal run.
- `--resume` / incremental CSV write: verified as safe (no duplication, no
  data loss, byte-identical no-op on a completed run) at multiple scales,
  most recently the round-27 full run.
- RadioML (RML2016.10a) real-sample IQ source, ground-truth sensing
  metrics, alignment-aware segmentation (`max-energy` policy),
  `radioml-native` AWN preprocessing: all exercised at N=2200+ (Phase 1) and
  N=3960+ (Phase 3/4).
- CLI/config boundary validation (`threshold_factor`, `window_size`,
  `min_region_len`, `burst_len`, `snr_db`, `attack_eps`, `attack_temperature`)
  for legal/boundary/negative/zero/NaN/Inf/non-numeric inputs: implemented
  in `src/utils/config.py`, documented in `docs/parameter_validation.csv`.

### 3.1 Update (post-round-27 work): full attack registry + core-parameter acceptance

Three subsequent rounds (not yet reflected in the phase history above,
which predates them) extended `src/adapters/attack_adapter.py` from the
original fgsm/pgd/cw set to **17 attacks** (bim, mifgsm, difgsm, vmifgsm,
vnifgsm, rfgsm, tpgd, deepfool, fab, square, apgd, apgdt, autoattack, ead,
plus the original three), each smoke-tested against the real AWN
checkpoint (`experiments/run_attack_compatibility_smoke.py`,
`docs/ATTACK_NAME_MAPPING.md`, `docs/ATTACK_COMPATIBILITY_WORKLIST.md`).
`difgsm` required a custom, IQ-native reimplementation
(`src/adapters/iq_difgsm.py:IQDIFGSM`) since `torchattacks.DIFGSM`'s own
input-diversity transform assumes a 2D image and crashes on this repo's
`[N,2,T,1]` tensor layout -- installed torchattacks itself was never
modified. `experiments/test_iq_difgsm.py` covers it with 6 dedicated unit
tests (shape, I/Q-channel consistency, gradient, determinism, Linf/NaN/Inf
constraints, no-diversity equivalence).

A further round (`experiments/validate_pipeline_parameters.py`) added and
verified, end-to-end through the real CLI/pipeline: `--dataset` (fixed to
`RML2016.10a`), `--mod-filter`/`--snr-filter` (whitelist guards on the
single selected modulation/SNR, not a batch-iteration driver),
`--samples-per-cell` (a per-cell `sample_index` bound, deliberately a
separate concept from the pre-existing `n_samples`/stream-length field),
`--stream-length` (CLI-facing alias for that pre-existing field),
`--burst-insert-position {random,center,explicit}` (via the new, additive
`src/sensing/radioml_source.py:embed_sample_in_noise_at_position` --
`embed_sample_in_noise` itself untouched), `--batch-size` (real
`AWNModelAdapter.infer()` chunking, verified bit-identical predictions
across batch sizes), `--experiment-name` (sanitized, written to
`summary.csv`), and `--overwrite` (refuses to clobber an existing
`summary.csv` by default). Also fixed this round: `--topk` is now
strictly rejected outside `[1,128]` at the formal CLI/config boundary
(`require_valid_topk_strict`), and `--min-region-len<=0` is rejected at
the CLI boundary specifically (`require_valid_min_region_len_strict`,
called only from `args_to_config`) -- **neither change touches
`TopKAdapter`/`fft_topk_denoise`'s own existing bypass/clamp semantics,
nor the shared `validate_experiment_config` Phase 1's own formal script
depends on (`min_region_len=0`)**, preserving Phase 1's exact
reproducibility. `resume` was deliberately left `DEFERRED_WITH_REASON` --
`mod_filter`/`snr_filter`/`samples_per_cell` are guards/bounds, not batch
iterators, so a single CLI invocation still never produces more than one
row; real multi-combo resume continues to live at the batch-script layer
(each `experiments/run_phase*.py`'s own `--resume`). Final parameter
classification: 71 `IMPLEMENTED_AND_VALIDATED`, 1
`NOT_APPLICABLE_FIXED_BY_BACKEND` (`apgdt.loss` -- the real installed
`torchattacks.APGDT` has no such constructor parameter at all, always
DLR-targeted internally), 1 `NOT_IMPLEMENTED` (`progress_logging` -- no
verbosity flag exists), 1 `DEFERRED_WITH_REASON` (`resume`), 0
`INVALID_OR_BROKEN` (`results/parameter_validation_20260727T054218Z/`).

---

## 4. Not yet done / not yet verified

**已設計但未執行 (designed, not run):**
- Phase 3's true 11-modulation x 20-SNR full sweep (distinct from the
  completed 6-modulation full-N run -- see section 2's documentation-
  inconsistency note)
- Phase 4 "quick" tier
- Phase 5's optional 11-modulation elective sensing-sensitivity expansion
- Phase 6 (multi-burst extension: does real attack + Top-K defense
  generalize to a 2-burst scene; merge-gap behavior under attack) --
  a non-formal, dummy-backend precursor exists (`E4_single_vs_multi/`
  under the Phase 5 directory, section 2), not a substitute
- A non-oracle attack-identity or modulation detector that could make the
  oracle-conditioned Phase 4 findings (CW at K=20-50, QAM64 at K=10-50)
  actually deployable -- explicitly out of scope for every round so far

**尚未實作 (not implemented):**
- `adaptive_k_defense` / `adaptive_k_v2_defense` (the per-sample knee-based
  Top-K variants present in `external/adversarial-rf/util/defense.py` but
  never wired into `TopKAdapter`, per its own module docstring)
- `--checkpoint` alternates (`2016.10b_AWN.pkl`, `2018.01a_AWN.pkl`) --
  known-likely-broken, never tried
- `device=cuda` path -- no GPU available in this environment
- Real modulation waveform synthesis, segmentation overlap/hop-size,
  sample-rate concept, GNU Radio ZMQ streaming / USRP hardware path (the
  original `README.md` PoC-stage scope, still not built)
- A root-cause investigation into WBFM's persistently low clean accuracy
  (0.083-0.093 depending on phase) or QAM16's low direct/sensed agreement --
  flagged as open questions since Phase 1, never investigated

**尚未驗證 (not yet verified, flagged in-doc as open):**
- The fgsm-specific eps=0.1 success-rate dip (Phase 3, section 11.2) --
  real at N=360 but mechanism unexplained
- BPSK's CW-specific attack resistance (0.617 vs 0.983-1.000) -- newly
  surfaced, root cause unknown
- Whether the `AWN_All.py`-style normalization step (omitted from this
  repo's `TopKAdapter`) would change Top-K's effectiveness for the pinned
  checkpoint -- found to differ (Phase 4 root-cause round) but never
  ablated in a dedicated, decision-driving comparison

---

## 5. Known limitations and research risks

(From `formal_experiment_plan.md` section 7, R1-R7, still open as of this
document unless stated otherwise below.)

- **Sample count (N) per cell was a design choice, not a derived
  requirement** -- every phase's N should be checked against the paper's
  actual required statistical power before results are treated as final.
- **Phase 3/4's 6-modulation "reduced/full" subset is this project's own
  proposal**, not inherited from `external/adversarial-rf`'s conventions.
- **The `attack_temperature`/CW-defaults "ineffective at legacy
  preprocessing" vs. "effective under radioml-native" causal link (plan
  R4) is a plausible inference, not a confirmed mechanism** -- would need
  its own diagnostic round if the paper's methodology section needs to
  state it as fact.
- **No phase has touched `checkpoint` alternates, `cuda`, or the
  matplotlib-missing plotting fallback** -- out of scope throughout.
- **Every oracle-conditioned Phase 4 finding (attack-specific or
  modulation-specific K) is a real statistical effect but NOT a deployable
  defense claim** -- this is a standing framing requirement for any paper
  or meeting use of these numbers, not just a caveat.
- **WBFM and QAM16's model-specific weaknesses (low clean accuracy, low
  sensed/direct agreement respectively) have no established root cause** --
  could be training-checkpoint-specific, could be a preprocessing
  interaction; not yet investigated.
- **`formal_experiment_matrix.csv` contains at least one stale/ambiguous
  status field** (the `phase=3,tier=full` row -- see section 2) that should
  be cleaned up before the matrix is treated as a fully authoritative
  index on its own; `formal_experiment_plan.md`'s prose sections are more
  reliable for what was actually run.

---

## 6. Next steps, in priority order

1. **Resolve the Phase 3 matrix documentation inconsistency** (section 2) --
   either correct `formal_experiment_matrix.csv`'s `phase=3,tier=full` row
   or rename/clarify the two different "full" concepts, so the matrix and
   the plan document agree.
2. **Decide whether the Phase 4 oracle-conditioned findings (CW K=20-50,
   QAM64 K=10-50) are worth pursuing further** -- e.g. designing and
   validating a non-oracle attack/modulation detector -- or whether the
   round-27 global-fixed-K negative result is the final word for this
   checkpoint/defense combination.
3. **Root-cause the WBFM low-clean-accuracy and QAM16 low-agreement
   findings** if the paper's scope requires explaining them rather than
   just reporting them.
4. **Decide on the two still-optional expansions** (Phase 3's true
   11-modulation sweep, Phase 5's 11-modulation sensing-sensitivity
   expansion) based on whether the paper's scope needs broader-than-6/
   broader-than-3 modulation coverage.
5. **Phase 6 (multi-burst extension)**, if multi-burst scenes are in the
   paper's scope -- currently fully undesigned-in-execution (matrix row
   exists, no dry-run has been run).
6. **The `AWN_All.py` normalization ablation** (a dedicated,
   decision-driving comparison, not the diagnostic-only ablation already
   done) -- only worth doing if there's appetite to actually change the
   shipped `TopKAdapter`/`fft_topk_denoise` usage, which no round so far
   has recommended.

(Superseded in priority by Part 1's section 16 "Future Extension" list,
which reflects everything completed after round 28 including the
satellite-like track; kept here for historical continuity.)

---

## 7. Latest important commits (as of round 28; see `git log` for current HEAD)

| Commit | Content |
|---|---|
| `0cccc78` (HEAD at round 28) | Formal Phase 4 K-reduced full-N execution (round 27): 27720-row run, verification, `experiments/analyze_phase4_expanded_full.py`, plan section 19 |
| `12e0870` | Smoke test of the new formal Phase 4 design (round 26) |
| `8b164dd` | Confirmed Phase 4 Expanded-K Confirmation Experiment; designed the K-reduced full-N Phase 4 (round 25) |
| `eddca0f` | Phase 4 3-policy Top-K preprocessing ablation, K up to 128 (round 24) |
| `1b6ece2` | Phase 4 reduced-tier root-cause analysis (round 23) |
| `714e51e` | Phase 4 reduced-tier execution, N=792/3168 (round 22) |
| `6d54159` | Phase 3 FULL execution, N=3960 (round 20) |
| `1e96b85` | Phase 1 Sensing Baseline, full 2200 combos (round 18) |
| `60bb22a` | Phase 0 pilot execution (round 17) |

All commits: author/committer `Liu Lina <ji3g4lina@gmail.com>` only, no AI
attribution. Post-round-28 commits (`12afb69` through `a30fee8` at the time
of this update) continued the same author/committer convention -- see
`git log` directly rather than duplicating the full list here.

---

## 8. What can be cited now vs. what cannot yet be concluded

### Can be used in a meeting / paper draft now

- Sensing front end costs a small, real accuracy gap vs. an oracle slice:
  **+1.68 percentage points** (direct 0.5973 vs. sensed 0.5805), at
  `threshold_factor=1.5`/`max-energy`/`radioml-native`, N=2200 (Phase 1
  methodology; see Part 1 section 4 for the later four-path re-measurement
  under different seeding).
- Adversarial attacks are highly effective against this pipeline without
  any defense: **82.78% overall success rate** at N=3960, ordering
  cw > pgd > fgsm (Phase 3, no channel impairment).
- **Fixed-K Top-K FFT defense, applied globally (the only form directly
  deployable without an oracle), does not provide a statistically
  significant net accuracy benefit at any tested K, and is significantly
  harmful at K=10/40/50**, at full formal scale (N=3960 attack instances,
  27720 rows, WBFM included). Reconfirmed under the satellite-like channel
  (Part 1 section 13): low recovery (2.5%/6.9%), non-trivial clean
  degradation (29.03%).
- The pipeline (real AWN + real 17-attack registry + real Top-K, no dummy
  fallback) has been verified end-to-end, reproducible, and error-free at
  every formal scale tested, up to 27720 rows in a single Phase-4 run and
  576 rows in the satellite-like Step 4 run.

### Cannot yet be concluded

- Whether an attack-specific or modulation-specific (i.e. oracle-informed)
  Top-K policy could be deployable -- the statistical effects exist (CW
  K=20-50 excl-WBFM, QAM64 K=10-50) but no non-oracle detector has been
  built or validated to make them actionable.
- Root causes for WBFM's low clean accuracy, QAM16's low direct/sensed
  agreement, the fgsm eps=0.1 dip, and BPSK's CW-specific resistance --
  all real, reproduced findings, no established mechanism for any of them.
- Whether the `AWN_All.py`-style normalization difference in Top-K
  preprocessing would change the deployability conclusion -- found to
  differ, never proven better or worse in a decision-driving ablation.
- Any claim beyond the 6-modulation reduced/full-N grid this project chose
  -- the true 11-modulation full sweep has not been run (see section 2's
  documentation-inconsistency note).
- Anything about real OTA/live-SDR behavior -- every attack/defense/channel
  result in this entire document, including the satellite-like track, is
  digital/offline (A0 threat model or offline channel simulation).
