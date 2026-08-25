# Project-Close Checklist

Status legend: **PASS** (verified, evidence exists), **LIMITATION**
(functions but with a documented gap or unverified boundary), **NOT_IMPLEMENTED**
(no code path exists). Every row cites an evidence path -- no row is marked
PASS on the basis of this checklist's own say-so.

## A. Functional

| Item | Status | Evidence |
|---|---|---|
| Synthetic + RadioML2016.10a IQ sources, real pipeline end-to-end | PASS | `docs/PROJECT_STATUS.md` Part 1 §3; `results/formal_phase1_sensing_clean_amc/` |
| `.cfile` (complex64/interleaved float32/interleaved int16) wired to real backends | LIMITATION | `results/cfile_pipeline_smoke_20260727T082623Z/`; internal-fixture-verified only, no real SDR capture tested |
| Spectrum sensing (energy detect -> region -> segmentation) | PASS | `docs/research/CURRENT_SYSTEM_AND_COMPONENT_STATUS_ZH_TW.md` §5.1; `results/spectrum_sensing_utility_formal_20260727T021248Z/` |
| Streaming/stateful sensing | LIMITATION (prototype) | `src/sensing/streaming_detector.py`; fails at chunk_size 256/512, `results/performance_latency_20260818T010552Z/streaming_sensing_validation.csv` |
| AWN real-checkpoint inference | PASS | `src/adapters/awn_adapter.py`; used in every formal round |
| 17-attack registry (formal compatibility) | PASS | `results/attack_compatibility_smoke_20260727T030223Z/` (17/17 PASS); `docs/ATTACK_NAME_MAPPING.md`/`docs/ATTACK_COMPATIBILITY_WORKLIST.md` stale-status wording corrected in this revision |
| Dataset-path portability (no unoverridable single-VM absolute path in formal executable code) | PASS | `src/utils/dataset_path.py`; all ~31 `experiments/*.py` batch scripts + `docs/PROJECT_STATUS.md` §8; path-level regression (valid/default/invalid-path, fail-fast) verified in this revision |
| Top-K FFT defense (function) | PASS | `src/adapters/topk_adapter.py`; independent load/shape test |
| Live SDR / USRP / GNU Radio ingestion | NOT_IMPLEMENTED | `docs/DEPLOYMENT_READINESS.md`; 0 `gnuradio`/`uhd`/`zmq` imports repo-wide |
| Adaptive/routed Top-K selection | NOT_IMPLEMENTED | `external/adversarial-rf/util/defense.py`'s `adaptive_k*` never wired into `TopKAdapter` |

## B. Correctness

| Item | Status | Evidence |
|---|---|---|
| 0 error/fallback/NaN-Inf, Phase 0-4 (up to 27720 rows) | PASS | `results/formal_phase4_expanded_full/` |
| 0 error/fallback/NaN-Inf, satellite-like 576-combo final run | PASS | `results/satellite_like_final_20260821T021117Z/raw_results.csv` (all `status=ok`, all `fallback_used=False`) |
| Satellite-like final-result audit (matrix/sensing/AMC/attack/Top-K/latency/fairness-hash re-derivation) | PASS, 0 bugs found | `results/satellite_like_final_20260821T021117Z/audit/` (22 CSVs + `audit_report.json`) |
| Metric-definition/denominator reporting-consistency cleanup | PASS | `docs/research/SATELLITE_LIKE_FINAL_EXPERIMENT_ZH_TW.md` §14/15/22; `audit/metric_definition_audit.csv`, `audit/topk_denominator_audit.csv` |
| Satellite-like channel model correctness (AWGN semantics, amplitude/CFO/Doppler root cause) | PASS | `docs/research/SATELLITE_LIKE_CHANNEL_SIMULATOR_DESIGN_ZH_TW.md` §16; 29/29 unit tests |
| Attack batching-safety classification | LIMITATION | Only fgsm/pgd(det)/cw verified (`implementation_optimization` / `batched_algorithmic_variant`); other 14 attacks not individually verified |
| CW batching | PASS (classified correctly as non-equivalent) | `docs/research/PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md` §15.2 |

## C. Reproducibility

| Item | Status | Evidence |
|---|---|---|
| Deterministic seeding (channel, attack, sensing) | PASS | `docs/PROJECT_STATUS.md` Part 1 §14 |
| `raw_results.csv` byte-identical across two independent audit rounds | PASS | SHA256 cross-check against `manifest_analysis.json`, this validation pass |
| `--resume` / incremental CSV safety | PASS | Phase 4 round-27 byte-identical no-op re-run |
| Independent-process bit-identical spot-check | PASS | Phase 1, 16-combo spot-check |
| Timestamped result directories + manifest/provenance hashes | PASS | every `results/<name>_<timestamp>/` directory, current and historical |

## D. Performance

| Item | Status | Evidence |
|---|---|---|
| Clean AMC / end-to-end latency characterized | PASS | `docs/research/PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md` §5, §16 |
| FGSM/PGD(det) batching acceleration verified equivalent | PASS | 16.10x / 9.53x attack-only, bit-identical output |
| PGD(random_start=True) acceleration | LIMITATION (throughput-only) | not a per-sample equivalence claim, documented as such |
| CW acceleration | LIMITATION (batched_algorithmic_variant) | not directly comparable to FGSM/PGD speedup class |
| Thread-count tuning | LIMITATION (machine-specific) | `torch_num_threads=2` optimal on test machine only, not a portable claim |

## E. Satellite-like Validation

| Item | Status | Evidence |
|---|---|---|
| Channel model (MUST: AWGN/amplitude/propagation-delay metadata; SHOULD: CFO/Doppler/timing-offset) | PASS | `src/channel/satellite_like.py`; 29/29 unit tests |
| 576-combination final experiment | PASS | `results/satellite_like_final_20260821T021117Z/` |
| GIGO validation (sensing quality vs. AMC accuracy disambiguation) | PASS (observational, not causal) | `docs/research/SATELLITE_LIKE_FINAL_EXPERIMENT_ZH_TW.md` §18/22.2 |
| Real OTA / live satellite validation | NOT_IMPLEMENTED | explicitly out of scope for this project-close phase; A0 digital threat model only |
| Standards-compliant DVB-S2/S2X modem | NOT_IMPLEMENTED | no frame/FEC/modem layer exists in this repo |
| RadioML2018.01A / APSK migration | NOT_IMPLEMENTED (future extension) | unverified `2018.01a_AWN.pkl` checkpoint noted as a starting point only |

## F. Documentation

| Item | Status | Evidence |
|---|---|---|
| `docs/research/*.md` Traditional Chinese, formal register, evidence-traced | PASS | terminology/AI-attribution/simplified-char sweep for this revision, 0 hits |
| Stale cross-references fixed (difgsm status, cfile wiring status) | PASS | `docs/ATTACK_NAME_MAPPING.md`, `docs/ATTACK_COMPATIBILITY_WORKLIST.md`, `docs/DEPLOYMENT_READINESS.md` update notes added in this revision |
| `docs/research/README.md` reading-order index | PASS | created in this revision |
| `docs/PROJECT_STATUS.md` project-close handoff rewrite | PASS | this revision |
| PROJECT_STATUS.md language convention (English, not Traditional Chinese) | LIMITATION (deliberate, disclosed) | pre-existing convention (matches `README.md`); not translated in this revision -- see PROJECT_STATUS.md's own header note |

## G. Git Hygiene

| Item | Status | Evidence |
|---|---|---|
| `results/` untracked (only `.gitkeep` tracked) | PASS | `git ls-files \| grep '^results/'` |
| `external/AWN`, `external/adversarial-rf` pinned, no diff | PASS | `git diff --submodule=log`, `git submodule status` |
| `origin/nzzz_proposal` untouched | PASS | never checked out, never written to |
| AI attribution = 0 across all new/modified files | PASS | full sweep for this revision |
| `git diff --check` clean | PASS | this revision |

## H. Known Limitations (see `docs/PROJECT_STATUS.md` Part 1 §15 for the full list)

Streaming sensing (prototype only); `.cfile` real-capture validation gap;
14/17 attacks' batching safety unverified; Top-K condition-dependent, not
universally effective; no live SDR/OTA path; WBFM/QAM16 root causes
unexplained; satellite-like channel model is MUST+SHOULD scope only.
