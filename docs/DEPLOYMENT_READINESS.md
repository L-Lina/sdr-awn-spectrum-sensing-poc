# SDR / Device Deployment Readiness Audit

Read-only code audit against the current repo state (no experiment run, no
result modified, no core-pipeline code changed, `external/AWN`/
`external/adversarial-rf` not touched). Every claim below is sourced to a
specific file/line read during this audit, not to any prior conversation
summary.

> **This repo cannot currently be claimed "ready to run directly on an SDR
> or a board."** No real captured IQ signal (file or live) has ever been
> run through the real AWN/attack/Top-K pipeline, end-to-end, even once.
> Everything below documents exactly how far the two existing pipelines get
> and exactly where that claim would first become false if made today.

Three-tier status scale used throughout, evaluated specifically against
**the SDR/board-deployment goal** (a real captured signal reaching real
AWN/attack/Top-K and producing a usable output) -- not against "does Phase
0-4 work," which is a separate, already-answered question
(`docs/PROJECT_STATUS.md`):

- **已完成 (done)** -- wired into the CLI entry point
  (`experiments/run_full_experiment.py`) and executed successfully with
  the real backend. True today only for `iq_source in {synthetic,
  radioml}`.
- **可重用但未接線 (reusable, not wired)** -- the code exists, is generic
  (no hardcoded source assumption), and has been exercised in some
  context (often at scale) -- but is not reachable from
  `run_full_experiment.py` for a real capture without writing new glue
  code, and has never actually been run on real-capture-derived data.
- **尚未實作 (not implemented)** -- no code exists for this anywhere in
  the repo.

---

## 1. Two disconnected pipelines exist -- read this before anything else

This repo contains **two separate implementations** of the sensing-to-AWN
pipeline that must not be conflated:

1. **`scripts/sdr_sensing_to_awn_poc.py`** -- the original standalone PoC.
   Accepts `--input <path.cfile>` (a real capture) or `--demo` (synthetic).
   Has its own local copies of `energy_detect`/`extract_occupied_regions`/
   `segment_regions`/`normalize_segments`/`to_awn_input`. Its
   `run_awn_inference()` is **explicitly still the numpy-only placeholder**
   (docstring: "Placeholder for the AWN forward pass" -- returns
   `rng.normal(...)` random logits, `scripts/sdr_sensing_to_awn_poc.py:204-226`).
   It never imports `src/adapters/`; there is no attack, no Top-K, no real
   model anywhere in this script. **This is the only script that accepts a
   real capture file, and its "AMC result" is random noise, not a
   prediction.**
2. **`src/utils/pipeline.py:run_dry_run_experiment` + `src/adapters/*`,
   invoked via `experiments/run_full_experiment.py`** -- the real-backend
   pipeline used for every formal Phase 0-4 result (real AWN, real
   FGSM/PGD/CW, real Top-K, all CPU-verified at scale, see
   `docs/PROJECT_STATUS.md`). `ExperimentConfig.iq_source` only accepts
   `{"synthetic", "radioml"}` (`src/utils/config.py:243-244`, `485`).
   **This is the pipeline with real models, and it has never once
   accepted a real capture file.**

No code in this repo currently connects "a real captured IQ file" to "the
real AWN model." **Do not read `scripts/sdr_sensing_to_awn_poc.py`
producing a class label as evidence the system works on real signals --
that label is `np.random.default_rng(...).normal(...)`, unrelated to the
input.**

---

## 2. Minimum integration goal (explicit, so scope doesn't drift)

The minimum integration target this audit measures against is:

> **Wire the existing `.cfile` loader (`src/sensing/iq_source.py:
> load_iq_from_file`) into the FORMAL pipeline
> (`experiments/run_full_experiment.py` ->
> `src/utils/pipeline.py:run_dry_run_experiment` -> `src/adapters/*`),
> so a real captured file reaches real AWN/attack/Top-K.**

This explicitly is **not**: extending `scripts/sdr_sensing_to_awn_poc.py`
(the placeholder script) to somehow also gain a real AWN call -- that
script's architecture duplicates logic the formal pipeline already has
correctly, and doing so would create a third, still-separate
implementation rather than closing the actual gap. Section 7 proposes a
concretely scoped first step toward this goal (not implemented this
round).

---

## 3. IQ input sources

| Source | Status | Evidence |
|---|---|---|
| Synthetic (numpy-generated noise+burst) | **已完成** | `src/sensing/iq_source.py:generate_synthetic_iq`; wired into `run_full_experiment.py` -> `run_dry_run_experiment` as `iq_source="synthetic"` (the CLI default); exercised at N=2200+ across Phase 0/1. |
| RadioML (RML2016.10a, offline `.pkl`, real recorded samples embedded in synthetic noise) | **已完成** | `src/sensing/radioml_source.py`; wired into the same entry point as `iq_source="radioml"`; used for every formal Phase 0-4 result (up to N=3960 attack instances / 27720 rows, round 27). **Not a live capture** -- loads a fixed `[2,128]` sample from a local dataset file and synthetically embeds it into a longer noise stream. |
| Raw complex64 `.cfile` (a real GNU Radio / SDR capture file) | **可重用但未接線** | `load_iq_from_file` (`src/sensing/iq_source.py:66-77`) correctly parses `np.fromfile(path, dtype=np.complex64)` and has `validate_iq` to check it -- both are generic, source-agnostic, and would very likely work unmodified once wired in. But `grep`-confirmed: their only actual caller in the whole repo is the disconnected placeholder script (section 1); `run_full_experiment.py`/`ExperimentConfig` have no `"cfile"` option at all. **Never executed against real AWN/attack/Top-K, not even once, in this repo's history.** |
| stdin | **尚未實作** | No `stdin`/`sys.stdin` anywhere in `src/`, `experiments/`, or `scripts/` (grep-confirmed). |
| ZMQ (GNU Radio `ZMQ PUB Sink` streaming) | **尚未實作** | No `zmq`/`ZMQ` import anywhere; only a prose mention in `README.md` as a future idea. |
| GNU Radio (live flowgraph integration) | **尚未實作** | No `gnuradio` Python bindings imported anywhere; only the `.cfile`-via-File-Sink convention is documented, never scripted. |
| USRP / UHD (direct hardware) | **尚未實作** | No `uhd`/USRP driver code anywhere. |

---

## 4. Stage-by-stage status toward the goal in section 2

| Stage | Function | Status | Notes |
|---|---|---|---|
| Read `.cfile` | `src/sensing/iq_source.py:load_iq_from_file` | **可重用但未接線** | Works correctly in isolation; not called by the formal pipeline. |
| Validate IQ | `src/sensing/iq_source.py:validate_iq` | **可重用但未接線** | Generic, no source assumption -- but its only actual callers today feed it synthetic/radioml-derived arrays, never a `.cfile`-derived one. |
| Energy detection | `src/sensing/energy_detection.py:energy_detect` | **可重用但未接線** | Parametric window/threshold, no hardcoded length; exercised at N=2200+ but always on synthetic/radioml input. |
| Region extraction / merge | `mask_to_regions`, `merge_close_regions`, `filter_by_min_length` | **可重用但未接線** | Same file, same caveat. |
| Segmentation / alignment | `src/sensing/segmentation.py:select_aligned_segments` | **可重用但未接線** | `seg_len` is a parameter, not hardcoded; `max-energy` policy makes zero use of ground truth (works identically whether or not a true burst position is known) -- structurally the most "already real-capture-ready" stage in the pipeline, still never actually exercised on real-capture data. |
| AWN preprocessing | `src/sensing/normalize.py:apply_awn_preprocess` | **可重用但未接線** | Pure array transform, same caveat. |
| AWN inference | `src/adapters/awn_adapter.py:AWNModelAdapter.infer` | **可重用但未接線** | Real model, CPU, exercised at N=27720+ rows with 100% real-backend rate -- but every one of those rows came from synthetic or RadioML-embedded IQ, never a real capture. |
| Adversarial attack (FGSM/PGD/CW) | `src/adapters/attack_adapter.py:AttackAdapter.apply` | **可重用但未接線 (and optional for deployment)** | Real backend, CPU, exercised at scale, same source caveat. Not required for a production inference deployment -- only relevant for on-device robustness testing. |
| Top-K defense | `src/adapters/topk_adapter.py:TopKAdapter.apply` | **可重用但未接線** | Real backend, CPU, exercised at scale (K up to 128), same source caveat. |
| Ground-truth / Pd/Pfa metrics | `src/sensing/ground_truth_metrics.py` | **尚未實作 for a real capture, structurally** | Docstring (`ground_truth_metrics.py:1-7`): "only available when the burst position is actually known ahead of time... Never used by the synthetic-IQ path today." A real, unlabeled capture cannot produce these metrics -- not a code gap, a structural limitation (no ground truth exists to compare against). |
| CLI/config wiring for a real capture | `src/utils/config.py` + `src/utils/pipeline.py` + `run_full_experiment.py` | **尚未實作** | The actual blocking gap -- see sections 2, 6, 7. |
| Output (CSV / summary) | `src/utils/csv_writer.py:write_summary_csv` | **可重用但未接線** | Source-agnostic row writer; already used identically for both existing sources, would need zero changes for a third. |

---

## 5. What `.cfile` integration would need to confirm before it can be trusted

Wiring the loader in (section 7) is a few lines. **Trusting the result on
a real capture is a separate, larger question** this audit did not
attempt to answer, because no real capture file exists in this repo to
test against. At minimum, the following must be explicitly confirmed
(none are confirmed today -- `grep`-verified 0 mentions of sample rate,
endianness, or channel count anywhere in `src/`/`scripts/`/
`experiments/run_full_experiment.py`):

| Format question | Current state | Why it matters |
|---|---|---|
| **complex64 vs. interleaved float32/int16** | `load_iq_from_file` assumes `np.complex64` (interleaved `float32` I/Q pairs) via `np.fromfile(..., dtype=np.complex64)`. `validate_iq` raises `TypeError` if the array isn't already complex -- it does **not** know how to interpret a raw interleaved `float32`/`int16` file that hasn't already been read as complex64, and would simply fail if the byte layout doesn't match GNU Radio's `gr_complex` convention. | A capture written by SDR software using a different sample format (e.g. `int16` I/Q, common on some USRP/OpenSDR pipelines) would either raise an error (best case) or silently misinterpret bytes (if someone forced a dtype cast upstream) -- never validated either way. |
| **Endianness** | `np.fromfile` uses the machine's native byte order by default; never explicitly specified or checked anywhere in the code. | GNU Radio's File Sink on x86 writes little-endian, matching this development machine -- but nothing in the code asserts this, and a capture from a different-endianness source (or a byte-swapped transfer) would silently produce garbage, not an error. |
| **Sample rate** | No `sample_rate`/`samp_rate` concept exists anywhere in the codebase (`grep`-confirmed 0 occurrences) -- also independently flagged as "尚未實作" in `docs/parameter_validation.md` section 6. | Every timing-derived quantity in this pipeline (burst length, energy-detection window, segment length) is expressed in raw sample counts, not seconds/Hz. Without a sample-rate concept, there is no way to know what real-world burst duration a 128-sample AWN window actually corresponds to, or to compare across captures taken at different sample rates. |
| **Channel count** | Only ever a single 1-D IQ stream (`validate_iq` requires `ndim == 1`, `iq_source.py:91-92`). | Multi-channel/multi-antenna captures (common on some SDR front ends) are not handled at all -- would need explicit de-interleaving/channel-selection logic that doesn't exist. |
| **Stream length** | No minimum/maximum length enforced beyond "at least one energy-detection window" (`energy_detect` raises if `n < window`, per `scripts/sdr_sensing_to_awn_poc.py:93-94`, mirrored in `src/sensing/energy_detection.py`). No chunking/windowing for very long captures -- the whole file is loaded into memory as one array. | A multi-GB continuous capture would be read entirely into RAM at once (`np.fromfile` has no streaming mode used here) -- untested, and likely the first practical failure point for a long live capture even before the model-inference stage. |
| **Scaling / absolute amplitude** | Two AWN-preprocessing policies exist (`legacy-unit-power` normalizes to unit power per segment; `radioml-native` applies no rescaling at all, per `src/sensing/normalize.py:17-51`) -- but both were validated only against synthetic-generator amplitudes or RadioML's own dataset amplitude scale (`docs/parameter_validation.md` sections 18-19). **Neither has ever been validated against a real SDR receiver's actual output amplitude/gain-setting-dependent scale**, which depends on RF front-end gain, ADC full-scale range, and GNU Radio's own scaling conventions -- all unknown quantities relative to what the pinned AWN checkpoint was trained on. |

None of these is a large fix individually, but **all of them are
currently unverified assumptions, not confirmed facts** -- treating
`.cfile` support as "basically done" once the loader is wired in would be
premature until at least the format and scaling questions are checked
against one real, known-good capture.

---

## 6. Latency measurement, broken out by stage

**Today, only one number exists**: coarse, wall-clock **per-combo**
timing via `time.time()` in `experiments/run_phase*.py` scripts (e.g.
`run_phase0_pilot.py:236,283`), which bundles multiple stages into one
interval. Round 27 produced 27720 rows / 809.3s = **~34 rows/sec
aggregate throughput** -- useful as a very rough order-of-magnitude
figure, **not** a per-stage or single-inference latency measurement.

What a real latency budget (needed before any real-time/on-device
feasibility claim) would require, stage by stage -- **none of the
following is currently isolated or measured anywhere in the repo**
(grep-confirmed no per-stage timing wraps exist):

| Stage | What it would measure | Current state |
|---|---|---|
| **Sensing** (`energy_detect` + region extraction/merge) | Time to go from a raw IQ buffer to a set of occupied regions | Not isolated -- bundled into the per-combo timer. |
| **Segmentation** (`select_aligned_segments` + alignment) | Time to go from occupied regions to fixed-length AWN-input windows, including the `max-energy` policy's candidate-window search (`O(region_len / hop)` power computations per region) | Not isolated. The `max-energy` search cost specifically has never been separately measured and could matter for a long real capture with wide occupied regions. |
| **AWN clean inference** | Single forward pass, model already loaded, one batch | Not isolated -- also confounded with checkpoint *loading* time in `run_dry_run_experiment`'s per-call `AWNModelAdapter` construction (section 4/section 8 below). |
| **Attack generation** (FGSM/PGD/CW) | Time to produce an adversarial perturbation (CW in particular runs an internal 20-step optimization loop, `cw_steps` default) | Not isolated; only relevant if the deployment includes on-device robustness testing, not for a plain inference deployment. |
| **Top-K defense** | FFT -> top-k mask -> IFFT, per segment | Not isolated. |
| **Defended inference** | A second AWN forward pass on the defended segment | Not isolated, same confound as clean inference. |
| **Overall wall-clock** (capture-in to result-out) | End-to-end latency a real deployment would actually experience, including any I/O | The only thing measured today, and only at the coarse "whole combo" granularity described above -- not validated as representative of a true single-shot, capture-to-result latency path. |

**Memory**: not measured at any granularity (0 occurrences of `psutil`,
`tracemalloc`, `resource.getrusage`, or `memory_profiler` anywhere in
`src/`, `experiments/`, or `scripts/`, grep-confirmed). Peak/steady-state
RAM for the AWN model + torch + (if RadioML-path code is reused)
the ~640MB dataset pickle is **unknown**, not just unmeasured-and-assumed-fine.

---

## 7. RadioML / fixed-128 / offline-file assumptions, by module

| Assumption | Where | Severity |
|---|---|---|
| `iq_source` restricted to `{"synthetic", "radioml"}` | `src/utils/config.py:243-244`, `485` | **Blocking** for any real capture. |
| RadioML sample fixed at `[2, 128]` | `src/sensing/radioml_source.py:74-76` (`load_radioml_sample` raises `ValueError` if shape mismatches) | Only affects the RadioML source path itself; segmentation/preprocessing take `seg_len` as a parameter, not a hardcoded 128. |
| RadioML dataset is an offline `.pkl` (~640MB), reloaded fresh every call, no caching | `src/sensing/radioml_source.py:38-50` | Irrelevant to a real-capture path (wouldn't be used there); relevant only if RadioML-embedding stays in use for on-device *validation* runs. |
| AWN model architecture (external, pinned checkpoint) implicitly expects `T=128` | `external/adversarial-rf/models/model.py` (wavelet lifting comment `# 10 9 128`; `Linear` layer dims derived from that decomposition) | **Not fixable by this repo's code.** No explicit `assert T==128` exists anywhere (grep-confirmed no `==128`/`!=128`/`shape[2]` checks), but no formal round has ever run a non-128 `window_size`, and the checkpoint's `Linear` layers were trained against the feature size that only holds for `T=128`. Untested whether a mismatch fails loudly or silently. |
| `requirements.txt` lists only `numpy` | `requirements.txt` | The real backend (torch, torchattacks, checkpoint loading) has never been captured as an installable dependency set in this repo -- every formal round used a pre-existing, externally-managed venv (`/home/xiaomi/adversarial-rf/.venv`). A fresh checkout on a deployment target would get **only the dummy fallback path**. |

---

## 8. CPU-only feasibility

**已完成 for the two existing sources.** Every formal round (Phase 0
through round 27) ran with `--device cpu` (the default and only value
ever exercised, `src/utils/config.py:460`); real AWN + real FGSM/PGD/CW +
real Top-K confirmed 100% real-backend, 0 fallback, at scale up to 27720
rows in one run. `cuda` is a recognized string (passed straight to
torch's `.to(device)` in `awn_adapter.py:99/126`, `attack_adapter.py:198/305`)
but **可重用但未接線 in practice** -- no GPU is available in this
environment and it has never been exercised.

**Per-call adapter construction is a known, unaddressed cost**:
`AWNModelAdapter`/`AttackAdapter`/`TopKAdapter` are constructed once per
`run_dry_run_experiment()` call. When a batch script calls
`run_dry_run_experiment()` once per combo (e.g. Phase 1), the checkpoint
is reloaded from disk every time -- `docs/formal_experiment_plan.md`
section 9 itself attributes Phase 1's slower-than-estimated runtime to
likely checkpoint-reload overhead per combo. Phase 0/3/4's own runner
scripts avoid this by constructing adapters once and reusing them --
**that pattern, not `run_dry_run_experiment()`'s default, is what a
streaming/live deployment would need.**

---

## 9. Next-stage minimum implementation order, with a verification condition per step

**Nothing below was implemented this round.** Each step lists what would
need to be true before moving to the next one, so a future round can
actually check rather than assume:

1. **Wire `.cfile` into `run_full_experiment.py`** (section 7's concrete
   proposal). *Verify*: a synthetic `.cfile` written via
   `np.ndarray.tofile()` round-trips through `--iq-source cfile
   --use-real-awn --use-real-topk --use-real-attack` with `run_status ==
   "ok"`, real (non-dummy) `*_backend` strings, and output shape/dtype
   identical to the synthetic-source path's own output for an equivalent
   array.
2. **Obtain one real SDR/GNU-Radio-captured `.cfile`** (even a short,
   controlled test transmission) and confirm section 5's format table
   against it directly -- dtype, endianness, an estimated/known sample
   rate, single-channel, a manageable file size. *Verify*: the file loads
   without a `validate_iq` error, and its numeric amplitude range is
   sanity-checked (not silently 1e10x or 1e-10x the synthetic/RadioML
   training-scale range) before ever calling AWN on it.
3. **Run the wired pipeline (step 1) against the real file (step 2) with
   real AWN, `awn_preprocess=radioml-native` (no rescale) AND
   `legacy-unit-power` (rescaled), both.** *Verify*: neither run produces
   NaN/Inf in `awn_input_*` fields; predictions are recorded but
   deliberately NOT assumed correct (no ground truth exists for a real
   capture, per section 4) -- this step verifies the pipeline *runs*, not
   that it classifies correctly.
4. **Add isolated per-stage timing** (section 6's seven rows) around a
   fixed, repeatable single-inference benchmark (not a full experiment
   batch). *Verify*: the seven stage timings individually sum to
   approximately the independently-measured overall wall-clock for the
   same run (a sanity check that nothing is double-counted or missed).
5. **Add memory measurement** (peak RSS via `resource.getrusage` or
   `tracemalloc`, at minimum) around the same benchmark. *Verify*: a
   repeatable peak-memory number exists and is compared against whatever
   the actual target board's available RAM is (unknown as of this
   audit -- would need to be supplied).
6. **Only after 1-5 pass**: decide on persistent-adapter reuse (section 8)
   and a real streaming/chunking design (section 1's ZMQ/rolling-buffer
   gap) -- both are architecture decisions that should be informed by
   step 4/5's actual numbers, not guessed at beforehand.
7. **GNU Radio flowgraph / USRP / OpenSDR hookup itself** -- deliberately
   last; doing it before steps 1-6 would validate hardware connectivity
   without validating that the software behind it produces a trustworthy
   result.

---

## 10. A candidate, scoped, ~30-minute gap -- proposed, NOT implemented this round

**Candidate** (unchanged from step 1 above, restated concretely): wire
`iq_source="cfile"` into `ExperimentConfig` + `run_dry_run_experiment` +
`run_full_experiment.py`'s CLI, purely additive.

- `src/utils/config.py`: add `"cfile"` to `validate_experiment_config`'s
  allowed values (`line 243-244`) and to `--iq-source` CLI `choices`
  (`line 485`); add a new `--input-file` CLI arg (str, default `None`);
  require it when `iq_source == "cfile"` (same pattern as `dataset_path`
  for `iq_source == "radioml"`, `config.py:256`).
- `src/utils/pipeline.py`: add a third `elif cfg.iq_source == "cfile":`
  branch (alongside the existing `radioml`/`else: synthetic` branches,
  `pipeline.py:156-184`) calling `load_iq_from_file(cfg.input_file)`,
  setting `gen_meta = {"source_type": "cfile", "input_file": cfg.input_file}`,
  and leaving `radioml_meta = None` (ground-truth metrics correctly stay
  unavailable, matching the synthetic path's existing behavior -- no new
  code needed there).
- Everything downstream (`validate_iq`, `energy_detect`, alignment,
  preprocessing, real AWN/attack/Top-K, CSV writing) already works
  unmodified for any `complex64` array, per section 4 -- genuinely a
  three-line dispatch addition plus two config-surface edits.

**Why this is safe to scope at ~30 minutes and non-disruptive**: strictly
additive `elif` branch; `iq_source` still defaults to `"synthetic"`,
every existing formal experiment script explicitly passes `"synthetic"`
or `"radioml"`, so this cannot change any existing result. Needs: the
three edits, a syntax check, and one smoke test (synthetic `.cfile`
written via `np.ndarray.tofile()`, read back through the new path,
checking `run_status=="ok"` and a real `awn_backend` string) -- no
`results/formal_*` directory touched.

**This closes only step 1 of section 9's seven-step order** -- it does
**not**, by itself, make any "ready for a real board" claim true; steps
2-7 remain entirely open. **Not implemented this round.**

---

## 11. Summary table

| Category | Item |
|---|---|
| **已完成** | Synthetic-source pipeline, end-to-end, real backends; RadioML-source pipeline, end-to-end, real backends (both via `run_full_experiment.py`); CPU-only execution for both sources |
| **可重用但未接線** | `.cfile` reading (`load_iq_from_file`, `validate_iq`); every sensing/segmentation/preprocessing/AWN/attack/Top-K/CSV-output function (all source-agnostic, all real/CPU-tested, none ever run on real-capture-derived data); `device="cuda"` code path; coarse per-combo wall-clock timing infrastructure |
| **尚未實作** | `.cfile`/live-capture CLI wiring into `run_full_experiment.py` (the blocking gap, section 7/10); stdin, ZMQ, GNU Radio live flowgraph, USRP/UHD; rolling-buffer/chunked streaming; persistent-adapter-reuse as a first-class API; format verification against a real captured file (endianness/sample-rate/channel-count/scaling, section 5); 7-stage isolated latency breakdown (section 6); any memory measurement; `requirements.txt` coverage of the real-backend dependency set; ground-truth-based validation against an unlabeled live signal (structural limitation, not a code gap) |
