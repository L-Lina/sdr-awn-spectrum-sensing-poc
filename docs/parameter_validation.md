# Parameter Validation Audit

Status: read-only audit performed by cross-checking actual source code against
CLI parsers, `ExperimentConfig`, adapters, `docs/experiment_design.md`,
`docs/integration_plan.md`, README, git history (10 commits total, this repo
has no separate historical parameter-value record of its own), and
`external/adversarial-rf`'s own scripts (read-only, submodule untouched).

This document was created recording the state as of commit `0aa95ea` and has
been incrementally updated in each subsequent round without being rewritten
from scratch; it currently also reflects the boundary-validation round
(`924dcdc`), the sensing-window decoupling round (`ba6248e`), and a Phase 1
real-pipeline validation round (section 10, real AWN + real attack + real
Top-K, no dummy fallback; not yet its own commit as of this paragraph's
writing). Every claim below is either a direct file:line citation or an
explicit record of a command actually executed in a working session against
this repo; nothing here is inferred from external-repo conventions or assumed
by naming similarity.

See `docs/parameter_validation.csv` for the machine-readable per-parameter
checklist that accompanies this document.

---

## 1. Current pipeline

```
generate_synthetic_iq(snr, mod, burst_len, n_samples, seed=0)   [src/sensing/iq_source.py]
  -> validate_iq
  -> energy_detect(window=effective_sensing_window_size, threshold_factor)  [src/sensing/energy_detection.py]
       (effective_sensing_window_size = resolve_sensing_window_size(window_size, sensing_window_size)
        -- decoupled from segment/AWN-input length as of the sensing/segmentation
        decoupling round; see section 8)
  -> mask_to_regions -> merge_close_regions(merge_gap)
       -> filter_by_min_length(min_region_len)
  -> segment_regions(seg_len=window_size)                       [src/sensing/segmentation.py]
  -> normalize_segments (unit-average-power, NOT clamped to [-1,1])
  -> to_awn_input()                                             [src/sensing/normalize.py]
       => x_clean  [N, 2, window_size] float32   (window_size is the LEGACY
                    name -- segment length == AWN input temporal length T,
                    unaffected by --sensing-window-size)
  -> AWN inference (real AWNModelAdapter or numpy dummy)         [src/adapters/awn_adapter.py]
       => logits_clean
  -> Attack (real AttackAdapter or numpy dummy_attack)           [src/adapters/attack_adapter.py]
       => x_adv, then AWN inference again => logits_attacked
  -> Top-K defense (real TopKAdapter or numpy dummy_topk_defense) [src/adapters/topk_adapter.py]
       => x_defended, then AWN inference again => logits_defended
  -> summary.csv (per-segment)                                  [src/utils/csv_writer.py]
  -> sensing_plot.png (best-effort, skipped if matplotlib missing) [src/utils/plotting.py]
```

Entry points: `experiments/run_full_experiment.py` (single run, uses
`src/utils/config.py:build_arg_parser`/`args_to_config`) and
`experiments/run_batch.py` (its own separate `build_batch_arg_parser`, sweeps
a grid via `itertools.product`, one subdirectory per combo plus
`batch_summary.csv`). Both call `src/utils/pipeline.py:run_dry_run_experiment`.

A third, older, standalone script `scripts/sdr_sensing_to_awn_poc.py` exists
side by side and is explicitly documented (`docs/experiment_design.md:38-41`)
as left untouched, not wired to the new `src/` package. It is the *only*
place in this repo with a working `--input <path.cfile>` real-capture read
path (see section 8).

Real vs. placeholder backends are all independently toggleable via
`--use-real-awn` / `--use-real-attack` / `--use-real-topk`; omitting any of
them falls back to a numpy-only dummy for that stage. **The dummy-fallback
code paths were not exercised in the working session that produced this
audit** — every actual run in that session passed all three `--use-real-*`
flags. This is recorded as `not_tested` in the CSV, not assumed to work.

---

## 2. CLI parameter inventory

Full parameter-by-parameter detail (type, default, target module, whether it
actually affects behavior, CSV/batch_summary.csv presence, tested values,
validation) is in `docs/parameter_validation.csv`. Summary by category:

- **A. Batch / general**: `--dry-run`, `--output-dir`, and the batch-only
  `--snr-list`/`--mod-list`/`--attack-list`/`--topk-list` comma-separated
  variants (`experiments/run_batch.py:_parse_list`). No type/range validation
  on the list contents beyond what the underlying single-value parameter
  enforces. A negative first value (e.g. `--snr-list "-10,0"`) is
  mis-parsed by argparse as an unrecognized option unless written as
  `--snr-list="-10,0"` (discovered and worked around in-session, not a repo
  fix).
- **B. IQ source**: `--snr`, `--mod`, `--burst-len` are CLI-exposed.
  `n_samples` (default `8192`), `burst_start` (computed, centered), burst
  amplitude (hardcoded `1.0`), noise std (derived from SNR), and the RNG
  `seed` (hardcoded `SEED = 0` in `src/utils/pipeline.py:33`) are **not**
  CLI-exposed at all.
- **C. Spectrum sensing**: `--threshold-factor`, `--window-size`,
  `--sensing-window-size`, `--min-region-len`, `--merge-gap` are CLI-exposed.
  `--window-size` is now a **legacy name**: it controls segment length /
  AWN input temporal length only. `--sensing-window-size` (new) controls
  `energy_detect`'s smoothing window independently, defaulting to
  `--window-size` when unset (prior coupled behavior reproduced exactly).
  Overlap/hop-size, max segment count, and sample rate are not implemented
  anywhere in this repo.
- **D. Segmentation**: segment length is still the same parameter as
  `--window-size` (no separate `segment_length`/`model_input_length` flag --
  that three-way split was explicitly deferred, see section 8). No
  overlap/hop parameter exists in `src/sensing/segmentation.py:segment_regions`
  (fixed non-overlapping windows only). No max-segments cap exists. No
  crop/pad/resample exists anywhere.
- **E. AWN/AMC**: `--use-real-awn`, `--checkpoint`, `--device`.
- **F. Attack**: `--use-real-attack`, `--attack`/`--attack-list`,
  `--attack-eps`, `--attack-temperature`, `--attack-diagnostics`.
- **G. Top-K defense**: `--use-real-topk`, `--topk`/`--topk-list`.
- **H. Output/plotting**: `--output-dir`; `sensing_plot.png` generation has
  no CLI toggle (always attempted, silently skipped if matplotlib missing).
- **I. Reproducibility/seed**: only the hardcoded `SEED = 0`; no
  `torch.manual_seed`, no `torch.use_deterministic_algorithms`, no thread-count
  pinning anywhere in this repo (confirmed unnecessary in-session: real AWN
  forward inference in eval mode was empirically verified bit-identical
  across independent process launches given a fixed input).
- **J. Real SDR/cfile**: `src/sensing/iq_source.py:load_iq_from_file` exists
  but is **not wired into** `run_full_experiment.py` / `run_batch.py` /
  `ExperimentConfig` at all — no `--input`/`--cfile` flag exists on the new
  pipeline. Only the old, untouched `scripts/sdr_sensing_to_awn_poc.py` has a
  working `--input <path.cfile>` flag. GNU Radio ZMQ streaming and USRP/UHD
  are describe-only in `README.md`, no code exists.

---

## 3. Spectrum sensing parameters

| Parameter | Exists | Actually used | CLI-exposed |
|---|---|---|---|
| threshold-factor | yes | yes | yes |
| window-size | yes | yes (segment length / AWN input T -- **legacy name**, no longer controls energy-detection window, see section 8) | yes |
| sensing-window-size | yes (new) | yes (energy-detection smoothing window only) | yes |
| min-region-len | yes | yes | yes (0-value bug fixed, see section 8) |
| merge-gap | yes | yes | yes |
| burst-len | yes | yes | yes |
| stream length (n_samples) | yes | yes | **no** |
| burst start | yes (computed, centered) | yes | **no** |
| burst amplitude | yes (hardcoded `1.0`) | yes | **no** |
| noise standard deviation | yes (derived from SNR) | yes | **no** (only indirectly via `--snr`) |
| seed | yes (hardcoded `0`) | yes | **no** |
| segment length | = window-size (still shared with AWN input T; the sensing-window/segment split does NOT split segment length from AWN input length -- those two remain coupled) | yes | yes (via `--window-size`) |
| overlap / hop size | **not implemented** | — | — |
| max segments | **not implemented** | — | — |
| sample rate | **concept does not exist** in this synthetic pipeline | — | — |
| source type (synthetic vs. cfile) | old script only | old script only | new pipeline: **no** |
| cfile path | `load_iq_from_file` exists, unwired | old script only | new pipeline: **no** |

---

## 4. Attack support matrix

| Attack | Parser accepts | Backend route exists | Can actually execute | Passed correctness test (post all 3 fixes) | Status |
|---|---|---|---|---|---|
| `none` | yes | yes (no-op bypass) | yes | yes (bit-identical verified repeatedly) | fully validated |
| `fgsm` | yes | yes (`torchattacks.FGSM`) | yes | yes (train/eval fix, min-max fix, temperature scaling all verified; eps-sweep found first prediction-changing eps = 0.5) | fully validated |
| `pgd` | yes | yes (`torchattacks.PGD`, `alpha=eps/4, steps=10` hardcoded) | yes | yes (same as fgsm; first prediction-changing eps = 0.3) | fully validated |
| `cw` | yes | yes (`torchattacks.CW(c=1.0, steps=20, lr=0.01)` hardcoded) | yes, re-verified post-fix (see section 10) | **execution path: yes. attack effectiveness: NO — not an effective attack at repo default hyperparameters.** Re-run under the current code (train/eval fix + min-max fix + temperature scaling, T=100): real backend confirmed throughout, 0/5 predictions changed, IQ perturbation ≈1.8e-7 (float32 noise floor). Root cause: `best_adv_images` in torchattacks' own CW implementation starts as a clone of the clean input and is only overwritten on a step that is BOTH misclassified AND lower-L2 than the previous best; with `c=1.0, steps=20, lr=0.01` against this checkpoint's huge logit margins (~600-900), CW never finds such a step within budget. A small hyperparameter sweep found effectiveness returns at `c=10, steps=100, lr=0.1` (3/5 changed, IQ Linf up to 2.2) — so the mechanism works, the repo's **default CW hyperparameters are simply too weak for this checkpoint**, not a wrapper/backend incompatibility | **execution path: fully validated. Effectiveness: NOT VALIDATED at current default hyperparameters — do not cite CW as a working attack without also citing the c/steps/lr used.** |
| Anything else `external/adversarial-rf` supports (bim, apgd, deepfool, autoattack, mifgsm, ... ~30 total per that repo's own `util/adv_eval.py`) | **no** | **no** | **no** | **no** | **not supported by this pipeline at all** — `src/adapters/attack_adapter.py:_SUPPORTED_ATTACKS = {"none","fgsm","pgd","cw"}` is the complete, exhaustive list this repo wires up. External-repo capability is not equivalent to this repo's capability. |

New pipeline currently supports exactly four attack names: **none, fgsm, pgd,
cw** — no more, no fewer.

---

## 5. Modulation implementation status

`src/sensing/iq_source.py:generate_synthetic_iq` has exactly one branch that
reads `mod`:

```python
mod_digest = int.from_bytes(hashlib.sha256(mod.encode("utf-8")).digest()[:8], "big")
freq_offset = 0.05 + (mod_digest % 100) / 1000.0
```

| Expected modulation effect | Actually implemented? |
|---|---|
| Symbol mapping | no |
| Constellation | no |
| Phase/amplitude modulation | no — the signal body is a bare carrier `exp(j*2*pi*freq_offset*t)` added to Gaussian noise, independent of `mod` in every other respect |
| Carrier frequency offset | **the only thing actually affected**, and it is a hash of the string, not a physically meaningful modulation-dependent value |
| Metadata only | **yes, effectively** — `mod` is otherwise used only for output filenames, CSV columns, and this cosmetic offset |

In-session empirical confirmation: `QPSK`, `BPSK`, `8PSK`, `QAM16` were run
under otherwise-identical conditions; the sensing region, segment boundaries,
and the resulting AWN input tensor were numerically identical across all
four (the frequency-offset difference is too small to change energy
detection results, and AWN receives the same underlying noise+carrier
signal regardless of the requested label).

**Explicit status: modulation is not implemented as a real waveform
synthesizer. It is a cosmetic frequency-offset selector plus a metadata
label.** No real symbol mapping or constellation generation exists anywhere
in this repo.

---

## 6. Test status summary (已通過 / 部分通過 / 未測 / 未實作)

**已通過 (passed)**
- `attack=none` bit-identical bypass (IQ, logits, predictions, distances all zero)
- `attack=fgsm`, `attack=pgd` end-to-end (post all 3 correctness fixes); re-confirmed
  again under simultaneous real AWN + real attack + real Top-K (no dummy fallback
  anywhere) in the Phase 1 real-pipeline validation round -- see section 10
- `attack=cw` **execution path only** (real backend, no crash, no fallback) --
  re-verified post-fix in the Phase 1 round; **effectiveness at the repo's
  default hyperparameters is NOT passed** (0/5 predictions changed at
  `c=1.0,steps=20,lr=0.01`) -- see section 10.2 for the full diagnosis
- `--attack-eps` exact enforcement re-confirmed under real backend: normalized-domain
  IQ Linf equals the requested eps exactly, for fgsm and pgd, at every value tested
  (section 10.4); confirmed `attack-eps` is completely ignored by `cw` (no `eps`
  attribute exists on the constructed `torchattacks.CW` object)
- `--topk` re-confirmed under real backend at K=10/20/30/40 against `none`/`fgsm`/`pgd`;
  directly proved different K values reach the real `fft_topk_denoise` function itself
  (pairwise-distinct outputs for identical input), not just CSV metadata (section 10.3)
- Cross-process reproducibility of synthetic IQ generation (post `hashlib` fix)
- AWN model eval-mode restoration after real attacks
- `--min-region-len` propagation: unset -> `window_size`, explicit `0` ->
  `0` (preserved), explicit `64` -> `64`. Verified via a scratch script
  calling `build_arg_parser`/`args_to_config` (`src/utils/config.py`) and
  `build_batch_arg_parser` + the equivalent resolution expression
  (`experiments/run_batch.py`) directly, at both the raw `argparse.Namespace`
  layer and the resolved config-value layer.
- **`threshold_factor` / `window_size` / `min_region_len` / `burst_len` /
  `snr_db` / `attack_eps` / `attack_temperature` boundary validation**
  (added in a dedicated round after this document's initial version). Five
  reusable validator functions (`require_positive_finite_float`,
  `require_finite_float`, `require_nonneg_finite_float`,
  `require_positive_int`, `require_nonneg_int` in `src/utils/config.py`)
  back both a CLI `type=` layer (identical error text at parse time) and a
  `validate_experiment_config(cfg)` boundary called at the very top of
  `src/utils/pipeline.py:run_dry_run_experiment`, so a direct-API caller who
  builds `ExperimentConfig` by hand (e.g. `experiments/run_batch.py`'s own
  construction, which never calls `args_to_config`) is protected too, not
  just the single-run CLI path. `attack_eps` and `attack_temperature`
  additionally have their own boundary check directly inside
  `AttackAdapter.apply()` (`src/adapters/attack_adapter.py`), since that
  class is also called directly in scratch scripts without ever touching
  `ExperimentConfig`. Verified at all three layers (CLI parse, resolved
  config, direct-API boundary) for legal values, the boundary legal value,
  negative, zero, NaN, Inf, and non-numeric strings; see
  `docs/parameter_validation.csv` for the per-parameter matrix. Fixed a real
  bug in the process: the pre-existing `attack_temperature <= 0` check
  silently let NaN and Inf through (see section 8). This round's validation
  is config/CLI/adapter-boundary level only -- it does **not** re-verify
  that legal-but-unusual values (e.g. `window_size=1`, `threshold_factor
  =0.0001`) actually behave sensibly through the full sensing/AWN/attack/
  Top-K pipeline; `behavior_test` for those stays `not_tested` in the CSV.

**部分通過 (partial)**
- `--snr`: tested at -10, 0, 10, 18 (real backends), plus -200 as a boundary
  legal value and NaN/Inf/non-numeric as rejected values (this round); no
  intentional upper/lower dB range limit was added (out of scope this round)
- `--mod`: cosmetic-only behavior confirmed for 4 values; arbitrary/malformed
  strings not tested
- `--topk`: tested at 10, 20, 30, 40, real backend, K confirmed to reach the
  real `fft_topk_denoise` function itself (section 10.3); default `50` never
  actually run; boundary values (0, negative, > window_size, NaN, Inf)
  intentionally left unvalidated this round -- see the `topk=Inf`
  uncaught-crash finding in section 8, not yet fixed. **Defense-recovery
  effectiveness against fgsm/pgd remains NOT ESTABLISHED** -- only a small,
  non-systematic number of recoveries observed across the Phase 1 round
  (section 10.3), not proof Top-K reliably defends
- `--device`: only `cpu` tested (no GPU available on this machine); `cuda`
  path completely unexercised
- `--use-real-awn`/`--use-real-topk`/`--use-real-attack`: `True` path fully
  exercised; `False` (dummy) path never actually run end-to-end in-session
  (though `AttackAdapter`'s dummy fallback branch specifically was exercised
  directly in this round's boundary tests)
- `--threshold-factor`/`--window-size`/`--burst-len`: boundary validation
  (0, negative, NaN, Inf, non-numeric) added and verified this round; the
  *legal* non-default values (e.g. `window_size` other than 128) still have
  never been run through the actual sensing/AWN pipeline

**未測 (not tested)**
- `--merge-gap`: only ever run at its default value (`0`); never varied to a
  nonzero value; boundary behavior intentionally left as-is this round (no
  validation added, no-op for `<=0` already existed and was left untouched)
- `--checkpoint`, alternate values (`2016.10b_AWN.pkl`, `2018.01a_AWN.pkl`):
  never tried; see section 8 for why they would likely fail silently
- matplotlib-missing plotting fallback path

**未實作 (not implemented)**
- Real modulation waveform synthesis (symbol mapping, constellation)
- Segmentation overlap/hop-size, max-segments cap
- Sample-rate concept
- New-pipeline `--input`/`--cfile` real-capture flag (exists only in the old
  standalone script)
- GNU Radio ZMQ streaming, USRP/UHD hardware path (README-only, no code)
- Global torch determinism / seed CLI (simply not present; **REVISED this
  round** -- previously believed "not needed", now known to matter for PGD
  specifically due to `random_start=True`, see section 10.3)

---

## 7. Outstanding items before a formal experiment

1. ~~CW must be re-verified under the current code~~ — **done (Phase 1
   round, section 10.2)**: real backend confirmed, no crash, but **not an
   effective attack at the repo's default hyperparameters**
   (`c=1.0,steps=20,lr=0.01` → 0/5 changed). A small sweep found
   effectiveness returns at `c=10,steps=100,lr=0.1` (3/5 changed) —
   parameters too weak, not a wrapper/backend problem. Whether to change the
   shipped defaults is still an open decision.
1b. **PGD results are not reproducible run-to-run** (new finding, section
   10.3): `torchattacks.PGD`'s `random_start=True` default is never
   overridden, and no `torch.manual_seed()` exists anywhere in this repo —
   `SEED=0` only fixes the numpy RNG for synthetic-IQ generation. Any PGD
   result (including the `eps=0.3` "first change" point cited in section 4)
   should be treated as one observed sample, not a guaranteed reproducible
   outcome, until this is addressed.
1c. **Top-K's actual defensive value against fgsm/pgd remains NOT
   ESTABLISHED** (section 10.3): only a small, non-systematic number of
   recoveries were observed across the Phase 1 round; no dedicated
   recovery-rate sweep (across SNR/eps/K) has been run. Do not cite
   "Top-K defends against fgsm/pgd" as a validated claim yet.
2. **Modulation has no real implementation.** Any formal experiment that
   claims a per-modulation accuracy/robustness comparison cannot be
   supported by the current pipeline without first building actual waveform
   synthesis (symbol mapping + pulse shaping) per modulation.
3. **Dummy-fallback paths (`--use-real-* ` omitted) have never been run
   in-session.** Should be smoke-tested at least once before relying on the
   fallback behavior in a formal report.
4. **Checkpoint switching is unsafe.** `src/adapters/awn_adapter.py`'s
   `_AWN_2016_10A_CFG` is hardcoded for the 2016.10a checkpoint only (11
   classes, `in_channels=64`, `latent_dim=320`). Pointing `--checkpoint` at
   `2016.10b_AWN.pkl` or `2018.01a_AWN.pkl` (10 classes / 24 classes,
   different `T`) would very likely fail to load and silently fall back to
   the dummy backend rather than erroring loudly.
5. ~~`--min-region-len 0` cannot actually be set~~ — **fixed** (an earlier
   round). ~~Negative `--min-region-len` values are unvalidated~~ — **fixed**
   this round (`require_nonneg_int`); `0` confirmed to remain legal.
6. ~~No boundary-value testing exists for `threshold-factor`, `window-size`,
   `burst-len`, or `attack-eps`~~ — **fixed this round** for these four plus
   `snr` and `attack-temperature` (zero/negative/NaN/Inf/non-numeric all now
   rejected with clear messages at both the CLI and adapter/algorithm
   boundary layers; see section 8). **`merge-gap` and `topk` remain
   unvalidated** — explicitly out of scope this round (design decisions
   deferred, see `docs/parameter_validation.csv`); `topk=Inf` in particular
   is a confirmed uncaught crash (see section 8), not yet fixed.
7. **Segmentation has no overlap/hop-size or max-segments control** — if a
   formal experiment design needs either, they must be built first.
8. **`cuda` device path is completely unverified** — this development
   machine has no GPU (`torch.cuda.is_available()` returns `False`).
9. **Legal-but-unusual values are validated but not behavior-tested.**
   E.g. `window_size=1` or `threshold_factor=0.0001` now pass validation,
   but nobody has run them through the actual sensing/AWN/attack/Top-K
   pipeline to confirm they behave sensibly (they likely don't, e.g.
   `window_size` != 128 vs. the AWN checkpoint's expected input length).
10. **`merge-gap` and `topk<=0`/`topk` FFT-bin-count clamping design
    decisions remain open**, along with whether `window-size` should ever be
    forced to exactly 128 — all explicitly deferred per this round's scope.

---

## 8. Correctness issues found during this audit

- **`--min-region-len 0` was silently overridden to `--window-size` — FIXED.**
  Both `src/utils/config.py` (`args_to_config`) and
  `experiments/run_batch.py` (`main()`) used to compute
  `min_region_len = args.min_region_len or args.window_size`. Python's `or`
  treats `0` as falsy, so a user explicitly passing `--min-region-len 0` got
  `window_size` instead of `0`. This was discovered during the parameter
  audit and fixed in a dedicated follow-up: both call sites now use an
  explicit `args.window_size if args.min_region_len is None else
  args.min_region_len` check, so `None` (unset) still falls back to
  `window_size`, but any explicit value — including `0` — is preserved
  exactly. Verified via a config-layer-only scratch test covering: unset
  (-> `window_size`), explicit `0` (-> `0`), explicit `64` (-> `64`), checked
  at both the `argparse.Namespace` layer and the resolved config value, for
  both `src/utils/config.py` and `experiments/run_batch.py`. The energy
  detector itself (`src/sensing/energy_detection.py:filter_by_min_length`)
  was not touched — it already accepted `0`/negative `min_len` correctly;
  the bug was purely in how the CLI value reached it. ~~Negative
  `--min-region-len` values remain unvalidated~~ — **also fixed in the
  boundary-validation round below** (`require_nonneg_int`); negative values
  are now rejected with a clear error instead of silently behaving as `0`.

- **Boundary validation added for `threshold_factor`, `window_size`,
  `min_region_len`, `burst_len`, `snr`, `attack_eps`, `attack_temperature`
  — FIXED (dedicated round).** Prior to this round, none of these seven
  parameters had any validation beyond argparse's own type coercion
  (`float`/`int`), and a read-only boundary audit found several concrete
  failure modes documented here for the record:
  - `threshold_factor <= 0` silently marked the entire signal "occupied"
    (mask all-`True`); NaN/Inf silently produced an all-empty mask, with no
    error until a much later, confusingly-worded `filter_by_min_length`
    exception.
  - `window_size = 0` caused an **uncaught `ZeroDivisionError`** inside
    `segment_regions` (`region_len // seg_len`).
  - `burst_len <= 0` silently produced a **burst-free, pure-noise signal**
    with no warning (`burst_start`/`burst_end` collapse to an empty or
    inverted slice), which would later surface as a confusing "no occupied
    region" error pointing at the wrong parameter.
  - `snr` at extreme magnitudes (e.g. `-1e6`, `1e6`) raised an uncaught
    `ZeroDivisionError` or `OverflowError` inside `generate_synthetic_iq`
    (`10 ** (snr_db / 10.0)` under/overflowing); `snr=NaN` silently produced
    a **NaN-contaminated IQ array** that would propagate through the entire
    pipeline undetected.
  - `attack_eps`/`attack_temperature` set to NaN or Inf silently corrupted
    `dummy_attack`'s output or (for temperature) bypassed the existing
    `<=0` check entirely (see the dedicated bullet below).

  Fix: five reusable validators (`require_positive_finite_float`,
  `require_finite_float`, `require_nonneg_finite_float`,
  `require_positive_int`, `require_nonneg_int`) added to
  `src/utils/config.py`, backing both a CLI `type=` layer (`arg_*` factory
  functions, reused identically by `experiments/run_batch.py`'s own parser)
  and a `validate_experiment_config(cfg)` boundary called at the top of
  `src/utils/pipeline.py:run_dry_run_experiment`. `attack_eps`/
  `attack_temperature` additionally get a direct check inside
  `AttackAdapter.apply()` (`src/adapters/attack_adapter.py`) since that
  class is callable independently of `ExperimentConfig`. No changes were
  made to the actual sensing/attack/Top-K algorithm logic — only guard
  clauses added ahead of it. `merge_gap` and `topk` were explicitly **not**
  touched this round (deferred design decisions).

- **`attack_temperature <= 0` check silently let NaN/Inf through — FIXED.**
  Found during the same boundary audit: `nan <= 0` and `inf <= 0` are both
  `False` in Python (NaN comparisons are always `False`; `inf` is `> 0`), so
  the original bare `if temperature <= 0: raise ...` check (introduced in
  the "Fix attack domain and saturated-gradient handling" round) silently
  accepted `NaN`/`Inf` despite the stated intent of "must be positive".
  Fixed by using `require_positive_finite_float` (checks `math.isfinite()`
  first) in both `AttackAdapter.apply()` and
  `TemperatureLogitsWrapper.__init__`. Confirmed rejected at the CLI,
  config, and direct-`AttackAdapter`-call layers.

- **`topk=Inf` causes an uncaught crash in BOTH the real and dummy Top-K
  backends — found, NOT fixed this round (out of scope).** `TopKAdapter`'s
  real-backend call fails with `OverflowError: cannot convert float
  infinity to integer` (external `fft_topk_denoise`'s `int(topk)`), gets
  caught by `TopKAdapter.apply()`'s broad `except Exception`, which then
  calls `dummy_topk_defense(x, topk=inf)` as a fallback — which **raises
  the same `OverflowError` a second time**, uncaught, escaping
  `TopKAdapter.apply()` entirely. This is the only case found in the audit
  where the fallback path itself also fails. `topk` was explicitly out of
  scope for this round's fix; see `docs/parameter_validation.csv` (`topk`
  row) and outstanding item 6/10 above.

- **`--checkpoint` has no existence/compatibility check.** A missing or
  incompatible checkpoint path fails inside `AWNModelAdapter.__init__`'s
  broad `except Exception`, silently falling back to the numpy dummy
  backend with only a `awn_notes` string as evidence — no loud error.
  (`src/adapters/awn_adapter.py:107-114`)
- **`--device` has no validation either** — an invalid device string is
  caught by the same style of broad `except Exception` and silently falls
  back to dummy, rather than failing loudly.
- **Real cfile/SDR ingestion is not wired into the new pipeline at all**,
  despite `src/sensing/iq_source.py:load_iq_from_file` existing and being
  fully functional in the old standalone script. Anyone assuming
  `run_full_experiment.py --input foo.cfile` works would be wrong — that
  flag doesn't exist on the new entrypoints.
- **CW's parameters (`c=1.0, steps=20, lr=0.01`) have never been checked
  for the same gradient-saturation issue found and fixed for FGSM/PGD**
  (raw AWN logits on this checkpoint have top1-top2 margins in the
  hundreds, which saturates float32 softmax and zeroes out
  `CrossEntropyLoss`'s gradient at temperature=1). CW's internal loss
  function may or may not have the same problem; this has not been
  investigated.
- **The three correctness fixes already committed** (for reference, not new
  findings — see commit messages):
  - `58e14e7` Restore AWN eval mode after real attacks — the wrapped model
    used to leak into train mode after a real attack call, corrupting all
    subsequent attacked/defended inference in the same process.
  - `10fbbe8` Fix cross-process reproducibility of synthetic IQ — the
    modulation-dependent `freq_offset` used Python's salted `hash()`,
    making the "same seed" not actually reproducible across separate
    process launches; replaced with `hashlib.sha256`.
  - `0aa95ea` Fix attack domain and saturated-gradient handling — the
    attack path assumed clean IQ was clamped to `[-1,1]` and used a fixed
    `(x+1)/2` mapping into torchattacks' `[0,1]` domain, silently clipping
    the ~12% of samples that fall outside that range; replaced with
    per-segment min-max mapping. Also added the temperature-scaling
    mechanism to work around the gradient-saturation issue above (FGSM/PGD
    only, verified).

- **Sensing window / segment length / AWN input length were coupled to a
  single parameter — PARTIALLY DECOUPLED (minimal two-parameter fix).**
  A dedicated read-only architecture audit found that `cfg.window_size`
  simultaneously drove three semantically distinct roles at
  `src/utils/pipeline.py`: `energy_detect(iq, window=cfg.window_size, ...)`
  (energy-detection smoothing kernel width), `segment_regions(..., seg_len=
  cfg.window_size)` (occupied-region segmentation length), and
  `to_awn_input(..., seg_len=cfg.window_size)` (AWN model input temporal
  length T). This made any `--window-size` sweep intended to explore sensing
  behavior *also* change what gets fed to the AWN model, confounding two
  independent experimental variables. Fix implemented this round:
  1. **Sensing window and AWN segment length are now decoupled.** A new
     `--sensing-window-size` CLI flag (and matching `ExperimentConfig.
     sensing_window_size: Optional[int] = None` field) controls *only*
     `energy_detect`'s smoothing window. `src/utils/config.py:
     resolve_sensing_window_size(window_size, sensing_window_size)` resolves
     `None` -> `window_size` (prior coupled behavior, reproduced exactly),
     called from `src/utils/pipeline.py:run_dry_run_experiment` (not at
     config-construction time), so both the CLI path and direct
     `ExperimentConfig(...)` construction resolve identically.
  2. **`--window-size` is now the legacy name** — it continues to control
     `segment_regions`'/`to_awn_input`'s `seg_len` (segment length == AWN
     input temporal length T) exactly as before; nothing about its behavior
     for existing callers changed. Verified byte-for-bit: with
     `--sensing-window-size` unset, the resulting energy-detection mask,
     occupied regions, and `x_clean` tensor are SHA256-identical to the
     pre-decoupling code path at `window_size=128, seed=0`.
  3. **For the pinned `2016.10a_AWN.pkl` checkpoint, segment length should
     still be kept at 128 for any real experiment** — this is a dataset/
     training convention (`external/adversarial-rf/util/config.py:51`:
     `self.signal_len = 128` for `2016.10a`/`2016.10b`; `:58`:
     `self.signal_len = 1024` for `2018.01a`), not something this round
     changed or validated otherwise.
  4. **The AWN model architecture structurally accepts other EVEN T values
     without a shape error** — traced `external/adversarial-rf/models/
     model.py`/`models/lifting.py` and confirmed empirically by loading the
     real `2016.10a_AWN.pkl` checkpoint and running `forward()` on all-zero
     dummy tensors at T=64/128/256/1024 (all succeeded, `[N,11]` output) and
     T=63 (failed with a `RuntimeError` from the lifting scheme's odd/even
     split, exactly as the odd-length hypothesis predicted). This is because
     `nn.AdaptiveAvgPool1d(1)` (`model.py:102`) removes all T-dependence
     before the `Linear` layers — no weight tensor in the checkpoint's
     `state_dict` has any T-sized dimension. **This is a structural
     compatibility finding only — it says nothing about whether predictions
     at non-128 T are statistically meaningful.** The checkpoint's weights
     were trained exclusively on 128-sample signals; no accuracy/validity
     claim exists for any other length, and none was tested (only all-zero
     dummy tensors were used, deliberately, to avoid producing anything
     resembling an "experiment result").
  5. **crop / pad / resample are NOT implemented.** Segment length and AWN
     input length remain hard-tied to the same `--window-size` value; a
     third `model_input_length` parameter with a crop/pad/resample bridge
     (needed to fully decouple segment length from AWN input length) was
     explicitly out of scope this round and is a separate, larger design
     decision (see the architecture-options analysis from the audit turn:
     option C in that discussion).
  6. **2018.01a model configuration is NOT wired in.** `src/adapters/
     awn_adapter.py`'s `_AWN_2016_10A_CFG` remains hardcoded to
     `num_levels=1` (matching only 2016.10a/10b); the 2018.01a checkpoint
     needs `num_levels=4` (confirmed via `config/2018.01a.yml:10` and by
     loading `2018.01a_AWN.pkl`'s `state_dict`, which has 4 distinct
     `levels.level_0..level_3.*` weight groups vs. 2016.10a's single
     `levels.level_0.*`). Pointing `--checkpoint` at `2018.01a_AWN.pkl`
     without also changing `num_levels`/`in_channels`/`latent_dim`/
     `num_classes` in the adapter would still fail `load_state_dict`
     (missing keys) — unrelated to and unaffected by this round's change.

---

## 9. Historical parameter sources

This repo's own git history (10 commits total) contains **no separate
historical parameter-value record** — no committed config file, no README
section, and no script that stores a "canonical" SNR/Top-K/eps test matrix
distinct from the argparse defaults themselves. Every numeric value cited
below is tagged with exactly where it came from; nothing is presented as
this project's "final" experimental value.

| Value | Source tag | Citation |
|---|---|---|
| `--snr` default `10.0` | current repo default | `src/utils/config.py:36` |
| `--snr-list` default `"0,10"` | current repo default | `experiments/run_batch.py:36` |
| SNR values `-10, 0, 10, 18` | current session tested | working-session command history; `results/param_test_snr_qpsk_fgsm/` |
| `--topk` default `50` | current repo default | `src/utils/config.py:40` |
| `--topk-list` default `"10,50"` | current repo default | `experiments/run_batch.py:39` |
| Top-K values `10, 20, 30, 40` | current session tested | `results/param_test_topk_snr18_qpsk_fgsm/` |
| `--attack-eps` default `0.03` | current repo default | `src/utils/config.py:37` (also matches `external/adversarial-rf/main.py:107-108`'s own default of `0.03` — coincidence of two independent defaults, not an inherited value) |
| eps values `0.1, 0.2, 0.3, 0.5` | current session tested | `results/eps_sweep_first_change/` |
| `--attack-temperature` default `1.0`; tested `1, 100`; `-1` (invalid, rejected) | current repo default / current session tested | `src/utils/config.py:61-64`; working-session command history |
| `--threshold-factor` default `5.0` | current repo default | `src/utils/config.py:39` — **never actually run in-session**; every real-backend test used `1.5` (current session tested, not the repo default) |
| SNR points `[0,2,4,6,8,10,12,14,16,18]` | external/adversarial-rf historical value | `external/adversarial-rf/util/defense_compare.py:79` — **not used by, or inherited into, this repo** |
| Confusion-matrix SNRs `[0,10,18]` | external/adversarial-rf historical value | `external/adversarial-rf/util/defense_compare.py:89` — not used by this repo |
| CW `c` default: `1.0` vs `10.0` (conflicting across their own scripts) | external/adversarial-rf historical value | `external/adversarial-rf/main.py:35` vs `external/adversarial-rf/util/sigguard_eval.py:93` — **conflicting even within that repo**; this repo's own CW call uses `c=1.0` (`src/adapters/attack_adapter.py`), chosen independently, not reconciled against either external value |
| CW `steps` default: `100` vs `200` (conflicting) | external/adversarial-rf historical value | `external/adversarial-rf/main.py:37` vs `util/sigguard_eval.py:94` — this repo uses `steps=20`, its own independent choice |
| CW `lr` default: `1e-3` vs `0.005` vs `0.01` (three conflicting values) | external/adversarial-rf historical value | `main.py:38`, `util/sigguard_eval.py:95`, `util/attack_bench.py` forced value — this repo uses `lr=0.01`, matching the `attack_bench.py` forced value by coincidence, not by design decision |
| Linf eps sweep `[0.01,0.03,0.05,0.1,0.15,0.2,0.25,0.3]` | external/adversarial-rf historical value | `external/adversarial-rf/util/defense_compare.py:82` — not used by this repo |
| Modulation class list (RML2016.10a, 11 classes: QAM16/QAM64/8PSK/WBFM/BPSK/CPFSK/AM-DSB/GFSK/PAM4/QPSK/AM-SSB) | external/adversarial-rf historical value | `external/adversarial-rf/data_loader/data_loader.py:13-14`, duplicated at `util/config.py:52-53` — **this repo's `--mod` accepts arbitrary strings and does not enforce, validate against, or actually implement this class list** (see section 5) |
| `n_samples` `8192` | current repo default | `src/utils/config.py:22` — no CLI, not yet finalized as a tunable |
| `SEED = 0` | current repo default | `src/utils/pipeline.py:33` — no CLI, not yet finalized as a tunable |

Any value not explicitly listed above and not present in
`docs/parameter_validation.csv` should be treated as **not_finalized** — do
not assume a value exists just because it appears reasonable.

---

## 10. Phase 1 real-pipeline validation round (real AWN + real attack + real Top-K, no dummy fallback)

This section records a dedicated validation round that ran the full pipeline
with **all three real backends simultaneously** (`--use-real-awn
--use-real-attack --use-real-topk`), confirmed via direct adapter precheck
before any test and via `awn_backend`/`attack_backend`/`topk_backend` CSV
columns on every single run — any run where any of the three fell back to a
dummy would be disqualified from a "real-path PASS" claim, and none did.
Fixed conditions unless noted otherwise: `--snr 18 --mod QPSK
--threshold-factor 1.5 --window-size 128 --burst-len 600 --device cpu
--checkpoint external/adversarial-rf/2016.10a_AWN.pkl`, `SEED=0` (hardcoded,
same synthetic IQ / same 5 segments / `pred_clean=[1,1,1,1,1]` throughout).
Environment: `/home/xiaomi/adversarial-rf/.venv` — torch `2.10.0+cu128` (CPU
only), torchattacks `3.5.1`. No repo file was modified to make any of this
pass; `external/AWN` / `external/adversarial-rf` were not touched.

### 10.1 Four-attack real-backend smoke test (none / fgsm / pgd / cw)

`--attack-eps 0.5 --attack-temperature 100 --topk 10` (temperature/eps chosen
deliberately higher than repo defaults so the smoke test could actually
observe a perturbation/prediction-change effect instead of reproducing the
already-documented T=1 gradient-saturation no-op).

| attack | execution path | awn/attack/topk backend | attack effectiveness | defense recovery |
|---|---|---|---|---|
| none | **PASS** (real, no-op bypass) | real/real(bypass)/real | n/a (no attack) | n/a |
| fgsm | **PASS** (real throughout) | real/real/real | **PASS** — 4/5 predictions changed, IQ Linf 1.12–1.62 | **NOT ESTABLISHED** — 0/4 successfully-attacked segments recovered |
| pgd | **PASS** (real throughout) | real/real/real | **PASS** — 3/5 predictions changed, IQ Linf 1.12–1.62 | **NOT ESTABLISHED** — 0/3 successfully-attacked segments recovered |
| cw | **PASS** (real throughout, no crash, no fallback) | real/real/real | **NOT YET VALIDATED / effectively a no-op at current defaults** — 0/5 predictions changed, IQ Linf ≈1.8e-7 (float32 noise floor); do not cite as a working attack — see 10.2 | n/a (nothing to recover) |

All 4 runs: no NaN/Inf anywhere; `attack_training_before=True` /
`attack_training_after=False` for fgsm/pgd/cw (expected — `Model01Wrapper` is
freshly constructed per process with `training=True` by default; the
`finally: self.wrapped_model.eval()` fix from `58e14e7` correctly restores
eval mode before any downstream inference uses the model, confirmed by
address — see 10.2 for why this fix matters numerically for this specific
checkpoint).

### 10.2 CW diagnosis (dedicated round)

- **Actual `c`/`steps`/`lr` used, confirmed on the live `torchattacks.CW`
  object (`atk.c`/`atk.steps`/`atk.lr`), not just what the repo's code
  intends to pass**: `c=1.0, steps=20, lr=0.01` — exactly matches
  `src/adapters/attack_adapter.py:_build_torchattacks`'s hardcoded values,
  no silent override by torchattacks' own constructor defaults
  (`torchattacks.CW.__init__` signature: `c=1, kappa=0, steps=50, lr=0.01`;
  this repo overrides `steps` from the library default 50 down to 20).
- **`--attack-eps` is completely ignored by CW**: confirmed both by reading
  `_build_torchattacks` (the `cw` branch never references its `eps`
  parameter) and empirically (`hasattr(atk, "eps")` is `False` on the
  constructed CW object — the attribute doesn't even exist, unlike FGSM/PGD
  where `eps` is a real attribute the attack enforces).
- **Root-cause finding (not a repo bug — a diagnostic pitfall)**: an initial
  hand-rolled diagnostic script (bypassing `AttackAdapter.apply()` to sweep
  `c`/`steps`/`lr` directly) produced wildly inconsistent, non-reproducible
  results (predictions "changing" even for near-zero IQ perturbation) until
  it was found that the script was missing the exact `finally:
  self.wrapped_model.eval()` step that `AttackAdapter.apply()` already has
  (`58e14e7`). torchattacks' own `Attack.__call__` always puts a freshly
  constructed wrapper module back into **train mode** after the attack call
  if it started in train mode (`_recover_model_mode`), and `AWN` has real
  `BatchNorm2d`/`BatchNorm1d`/`Dropout(0.5)` layers
  (`external/adversarial-rf/models/model.py:73,79,96`), so a leaked
  train-mode leaves every subsequent forward pass corrupted by
  batch-statistics/dropout noise on a 5-sample batch. **This is independent
  confirmation that the existing `58e14e7` fix is load-bearing for this
  checkpoint (not just a hygiene fix)** — without it, CW-adjacent diagnostic
  code silently produces meaningless results. The shipped `AttackAdapter.
  apply()` already has this fix in its `finally` block and was not modified.
- **Small `c`/`steps`/`lr` sweep** (real backend, same 5 segments,
  `temperature=100`, `pred_clean=[1,1,1,1,1]` throughout):

  | c | steps | lr | changed | IQ Linf (max) | IQ L2 (max) |
  |---|---|---|---|---|---|
  | 1.0 (default) | 20 (default) | 0.01 (default) | 0/5 | 1.8e-7 | 1.1e-6 |
  | 10.0 | 20 | 0.01 | 0/5 | 1.8e-7 | 1.1e-6 |
  | 100.0 | 20 | 0.01 | 0/5 | 1.8e-7 | 1.1e-6 |
  | 1.0 | 100 | 0.01 | 0/5 | 1.8e-7 | 1.1e-6 |
  | 1.0 | 20 | 0.1 | 0/5 | 1.8e-7 | 1.1e-6 |
  | 10.0 | 100 | 0.1 | **3/5** | **2.21** | **9.91** |

- **Conclusion: parameters too weak, not a wrapper/backend compatibility
  problem.** Scaling `c`, `steps`, and `lr` together (not any single one
  alone — each individually held the other two at default produced no
  change) restores CW's ability to find adversarial examples. The repo's
  current defaults (`c=1.0, steps=20, lr=0.01`) are not adequate for this
  checkpoint's logit-margin scale and should not be cited as "CW doesn't work
  against this model" — only "CW doesn't work at these specific untuned
  defaults." Whether to change the shipped defaults is an open design
  decision, not made in this round (out of scope; would need its own
  before/after correctness check).

### 10.3 Top-K real-backend validation (K = 10/20/30/40 × none/fgsm/pgd)

Same synthetic IQ/SNR/mod/seed throughout; `--attack-eps 0.5
--attack-temperature 100`. 12 runs, all real-backend, no fallback, no
NaN/Inf.

- `none`: `pred_clean == pred_attacked == pred_defended == [1,1,1,1,1]` for
  all 4 K values — bit-identical no-op confirmed again under the topk sweep.
- `fgsm` (deterministic, single-step — no randomness): `pred_attacked =
  [1,8,8,8,8]` identical across **all 4 K values** (correctly confirms Top-K
  is applied strictly after the attack and never influences
  `pred_attacked`); `pred_defended` differs by K — K=20 recovered segment 2
  (`8→1`), K=10/30/40 recovered nothing. **1 recovery out of 16
  successfully-attacked (K,segment) pairs across the sweep.**
- `pgd`: `pred_attacked` **varies across K-value runs** despite identical
  `eps`/`temperature`/input — traced to `torchattacks.PGD`'s
  `random_start=True` default (confirmed via `inspect.signature`), which
  this repo's `_build_torchattacks` never overrides, and no
  `torch.manual_seed()` exists anywhere in this repo. **PGD results are
  therefore not reproducible run-to-run even with identical CLI arguments
  and the same fixed `SEED=0`** — `SEED=0` only fixes the synthetic-IQ RNG
  (`numpy`), not torch's own RNG used by PGD's random start. This is a new,
  concrete instance of the previously-documented "no global torch
  determinism" gap (section 6/7) — previously that gap was believed
  "empirically shown unnecessary" for eval-mode AWN forward passes; this
  round shows it **does** matter for PGD specifically. Across the K-sweep,
  Top-K recovered 2 out of roughly 18 successfully-attacked (K,segment)
  pairs (K=20 seg0, K=40 seg4) — sporadic, not a systematic pattern.
- **Confirmed K reaches the real `fft_topk_denoise` function itself, not
  just CSV metadata**: called `TopKAdapter.apply()` directly with identical
  input and K∈{10,20,30,40}; `topk_backend`/`topk_status` were
  `fft_topk_denoise`/`ok` for all four, and the four output arrays are
  **pairwise non-identical** (`np.array_equal` False for all 6 pairs; output
  mean-abs magnitude increases monotonically with K: 0.418 → 0.533 → 0.616 →
  0.657, consistent with keeping more FFT energy as K grows).
- **Overall defense-recovery conclusion: NOT ESTABLISHED.** Across the
  4-attack smoke test (10.1) and the 12-run K-sweep (10.3), Top-K recovered
  a small, inconsistent minority of successfully-attacked segments (3 out of
  roughly 34 total attacked-segment instances observed this round). This is
  not evidence Top-K "doesn't work" (no systematic sweep across
  SNR/eps/attack-strength has been run), but it is clear evidence that
  **"Top-K=10 defends against FGSM/PGD" is not a validated claim** at this
  point — recoveries observed so far look incidental rather than
  systematic.

### 10.4 eps sweep for FGSM/PGD (real backend, same input)

Historical eps values actually used in this repo before this round (grepped
from `docs/parameter_validation.md`/`.csv`, not recalled from memory):
`--attack-eps` default `0.03` (`src/utils/config.py`); previously tested
real values `0.1, 0.2, 0.3, 0.5` (`results/eps_sweep_first_change/`,
finding: first prediction-changing eps was `0.5` for fgsm, `0.3` for pgd).
This round reused exactly this set (`0.03, 0.1, 0.2, 0.3, 0.5` — no new eps
values invented) plus the repo default, same synthetic IQ/SNR/mod/seed,
`--attack-temperature 100`, `--topk 10`.

| attack | eps | changed | normalized IQ Linf (all segments) | original IQ Linf (range) | NaN/Inf |
|---|---|---|---|---|---|
| fgsm | 0.03 | 0/5 | 0.03 | 0.068–0.097 | none |
| fgsm | 0.1 | 0/5 | 0.1 | 0.225–0.324 | none |
| fgsm | 0.2 | 0/5 | 0.2 | 0.450–0.647 | none |
| fgsm | 0.3 | 0/5 | 0.3 | 0.675–0.971 | none |
| fgsm | 0.5 | **4/5** | 0.5 | 1.125–1.618 | none |
| pgd | 0.03 | 0/5 | 0.03 | 0.068–0.097 | none |
| pgd | 0.1 | 0/5 | 0.1 | 0.225–0.324 | none |
| pgd | 0.2 | 0/5 | 0.2 | 0.450–0.647 | none |
| pgd | 0.3 | **2/5** | 0.3 | 0.675–0.971 | none |
| pgd | 0.5 | **5/5** | 0.5 | 1.125–1.618 | none |

- **eps is correctly and exactly propagated**: `iq_linf_normalized_clean_
  attacked` (the perturbation measured in the `[0,1]` domain torchattacks
  actually enforces its Linf budget in) equals the requested `eps` exactly,
  for every single segment, at every eps value tested — confirming the
  attack budget is enforced precisely, not approximately.
  `iq_linf_clean_attacked` (raw IQ-domain Linf) is correctly larger and
  varies per segment (depends on each segment's own min-max range used to
  denormalize back from `[0,1]`), as expected from the per-segment min-max
  domain mapping (`0aa95ea`).
  - Note this round's `--attack-eps 0.03` (fgsm) row is a **direct
    contradiction check** against the CW section 10.2 finding that CW
    ignores eps entirely — FGSM/PGD by contrast visibly and exactly obey it.
  - First-change eps reproduced exactly as previously documented: fgsm
    first changes at `0.5` (not `0.3`), pgd first changes at `0.3` — matches
    `docs/parameter_validation.md` section 4's prior citation, both under
    the real backend, real checkpoint, same synthetic IQ.
  - PGD's non-determinism (10.3) means this specific `pgd, eps=0.3` "2/5
    changed" result is **one observed outcome, not necessarily reproducible
    on a re-run** — see 10.3 for the root cause (`random_start=True`, no
    `torch.manual_seed`).

### 10.5 Still not completed after this round

1. CW's shipped default hyperparameters remain unchanged and remain
   ineffective against this checkpoint; whether to change them is an open
   design decision, not made here.
2. PGD's run-to-run non-determinism (`random_start=True`, no
   `torch.manual_seed` anywhere in this repo) is newly documented but not
   fixed — any PGD result should be treated as one sample, not a
   reproducible ground truth, until this is addressed.
3. Top-K's actual defensive value against FGSM/PGD remains **not
   established** — only a small, non-systematic set of recoveries has been
   observed; no sweep across SNR/eps/K designed specifically to
   characterize recovery rate has been run.
4. This round used `--attack-temperature 100` (not the `T=1.0` default)
   throughout, deliberately, to get past the already-documented
   gradient-saturation no-op — real-backend behavior at the CLI's actual
   default `T=1.0` for fgsm/pgd/cw has still not been separately
   re-confirmed in this round (expected, per the saturation finding, to
   reproduce the T=1 zero-gradient no-op — but not empirically re-checked
   here).
5. No SNR/modulation variation was run this round (fixed at SNR=18, QPSK
   throughout, per this round's explicit scope) — attack/defense
   effectiveness at other SNRs remains unknown.
6. `merge-gap`, `topk<=0`/`topk=Inf` boundary behavior, `--checkpoint`
   existence validation, and `--device cuda` remain exactly as documented in
   section 7 — untouched by this round.

---

## 11. Reproducibility fix + fair Top-K comparison + CW CLI design (round 2)

Follow-up round to section 10's PGD non-determinism finding. Code changes
this round: `src/utils/config.py`, `src/utils/pipeline.py`,
`experiments/run_batch.py` (adds a `--seed` CLI flag / `ExperimentConfig.
seed` field and global RNG seeding; **no changes to
`external/AWN`/`external/adversarial-rf`**). Section 10's smoke-test /
eps-sweep / K-sweep results (all recorded with the *old*, unseeded code) are
left as-is above, not retroactively edited — this section documents what
changed and re-verifies against the new code.

### 11.1 Seed data flow (before / after)

**Before**: `SEED = 0` was a module-level constant hardcoded in
`src/utils/pipeline.py`, threaded into `generate_synthetic_iq(..., seed=
SEED)`, `dummy_awn_inference(..., seed=SEED)`, `dummy_attack(..., seed=
SEED)`, and `AttackAdapter.apply(..., seed=SEED, ...)`. Not configurable via
CLI or `ExperimentConfig`. Critically, `AttackAdapter.apply()`'s own `seed`
parameter was **only ever consumed by its `dummy_attack` fallback branch**
(`src/adapters/attack_adapter.py`) — the real `torchattacks`-based branch
never seeded anything, so `torch`'s global RNG (used by `torchattacks.PGD`'s
`random_start=True`, confirmed via `inspect.getsource`) was left entirely
unseeded.

**After**: `ExperimentConfig.seed: int = 0` (`src/utils/config.py`, default
matches the old hardcoded value so omitting `--seed` reproduces prior
behavior exactly) is set via a new `--seed` CLI flag on both
`build_arg_parser` (single-run) and `build_batch_arg_parser`
(`experiments/run_batch.py`, applied uniformly to every combo in a batch, no
per-combo `--seed-list`). `src/utils/pipeline.py:_seed_everything(seed)` — a
new function, called once at the very top of every single
`run_dry_run_experiment(cfg)` call (so a `run_batch.py` sweep reseeds
identically before each combo, giving every combo the same guarantee a
standalone run gets regardless of combo order) — does:
```python
random.seed(seed)
np.random.seed(seed)
if _torch is not None:
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)
```
`torch` is imported optionally (same `try/except`-guarded pattern the
adapters already use), so this is a no-op in a torch-less dummy-only
environment. The old module-level `SEED` constant was removed; every
`seed=SEED` call site now reads `seed=cfg.seed`. The seed is **not**
hardcoded inside `attack_adapter.py` — `AttackAdapter.apply()`'s `seed`
parameter is still just a pass-through argument (used only by its own
`dummy_attack` fallback, as before); the actual global `torch.manual_seed`
call lives exclusively in `pipeline.py`, called before the adapter is ever
constructed.

`seed` is now recorded in every `summary.csv` row (`src/utils/pipeline.py`)
and every `batch_summary.csv` row (`experiments/run_batch.py`, read back
from `result["seed"]`, same pattern as `sensing_window_size`/
`segment_length`).

**Independent RNG sources tracked**: `torchattacks.PGD.forward`'s
`random_start` branch (`torch.empty_like(adv_images).uniform_(-self.eps,
self.eps)`) is the only randomness among FGSM/PGD/CW — traced via
`inspect.getsource`. FGSM is a single deterministic gradient-sign step; CW's
optimization starts from a deterministic `inverse_tanh_space(images)` with
no randomness. All draw from `torch`'s single global default generator, so
one `torch.manual_seed()` call covers all three.

### 11.2 PGD reproducibility test (after the fix)

Fixed conditions: `--snr 18 --mod QPSK --threshold-factor 1.5 --window-size
128 --sensing-window-size 128 --attack pgd --attack-eps 0.5
--attack-temperature 100 --seed 42`, real AWN + real attack + real Top-K.
Two fully independent `python` process invocations (fresh interpreter each
time, no shared state).

| Quantity | run 1 | run 2 | identical? |
|---|---|---|---|
| synthetic IQ SHA256 | `524161f2...` | `524161f2...` | yes |
| sensing mask SHA256 | `a6c38eb7...` | `a6c38eb7...` | yes |
| x_clean SHA256 | `96636f98...` | `96636f98...` | yes |
| x_attacked SHA256 | `7c69933c...` | `7c69933c...` | yes |
| pred_clean | `[1,1,1,1,1]` | `[1,1,1,1,1]` | yes |
| pred_attacked | `[2,1,0,1,1]` | `[2,1,0,1,1]` | yes |
| logits_clean SHA256 | `a26c8861...` | `a26c8861...` | yes |
| logits_attacked SHA256 | `4f53807f...` | `4f53807f...` | yes |
| `summary.csv` | `a7cc8e5f...` (file SHA256) | `a7cc8e5f...` | yes (byte-identical) |

All nine checks are bit-for-bit identical across the two independent
processes — the fix closes the gap found in section 10. (For contrast,
section 10.3's *old*-code observation — `pred_attacked` varying across
different `--topk` values despite identical `eps`/`temperature`/input,
traced to unseeded `random_start=True` — remains valid evidence of the
**pre-fix** failure mode; not re-run destructively against the old code to
"prove" it again.)

### 11.3 Fair Top-K sweep (single shared attacked IQ per attack, real backend)

Diagnostic script only (`fair_topk_sweep.py`, scratchpad — **not** a formal
experiment entrypoint; `run_full_experiment.py`/`run_batch.py` still
regenerate the attack from scratch per combo, which section 10.3 showed is
fine for FGSM but not appropriate for a same-input Top-K comparison under
PGD). For each attack, the clean IQ and the attacked IQ were each computed
**exactly once** (`--seed 42`, same fixed conditions as 11.2, `topk=10`
attack-independent params) and the identical `x_adv` array was then run
through the real `TopKAdapter` at K=10/20/30/40 — so any difference in
`pred_defended` across K is attributable to K alone, not to attack
re-randomization.

| attack | pred_attacked (fixed once) | K=10 recovered | K=20 recovered | K=30 recovered | K=40 recovered |
|---|---|---|---|---|---|
| none | `[1,1,1,1,1]` (0 attacked) | n/a | n/a | n/a | n/a |
| fgsm | `[1,8,0,8,8]` (4 attacked) | **1/4** (seg1) | 0/4 | 0/4 | 0/4 |
| pgd | `[2,1,0,1,1]` (2 attacked) | **1/2** (seg0) | **1/2** (seg0) | 0/2 | 0/2 |

`topk_backend` confirmed `fft_topk_denoise` (real) for all 12 (attack×K)
combinations. **Conclusion (fair comparison, still real-backend
evidence, still only 5 segments)**: recovery is clearly K-dependent (lower K
recovered more in both fgsm and pgd here), but the effect is small, only
partial (never full recovery), and based on a single 5-segment sample — this
is a genuine, apples-to-apples signal that **lower K trends toward more
recovery**, not proof of a reliable defensive effect; still **NOT
ESTABLISHED** as a validated general claim, consistent with section 10.3's
conclusion, now with the attack-randomness confound removed.

### 11.4 CW CLI design proposal (design only — not implemented this round)

**Where `c`/`steps`/`lr` are currently hardcoded**: `src/adapters/
attack_adapter.py:_build_torchattacks`, the `cw` branch:
```python
if attack_name == "cw":
    return _torchattacks.CW(wrapped_model, c=1.0, steps=20, lr=0.01)
```
No CLI flag, no `ExperimentConfig` field, no way to override without editing
this line directly.

**Proposed minimal design** (not implemented — touches `AttackAdapter.
apply()`'s signature, which `fgsm`/`pgd` also call through, so it is not a
"very small, isolated" change per this round's instructions):

- Three new CLI flags, mirroring the existing `--attack-temperature`
  pattern (single value, applied uniformly to every combo in a batch, no
  `--cw-c-list` sweep flag): `--cw-c` (`arg_positive_finite_float`, default
  `1.0`), `--cw-steps` (`arg_positive_int`, default `20`), `--cw-lr`
  (`arg_positive_finite_float`, default `0.01`) — defaults exactly match the
  current hardcoded values, so omitting all three reproduces current CW
  behavior bit-for-bit.
- Three new `ExperimentConfig` fields: `cw_c: float = 1.0`, `cw_steps: int =
  20`, `cw_lr: float = 0.01`.
- `_build_torchattacks(attack_name, wrapped_model, eps, cw_c=1.0,
  cw_steps=20, cw_lr=0.01)` — extend the signature; the `fgsm`/`pgd`
  branches ignore the three new parameters entirely, unchanged.
  `AttackAdapter.apply()` passes `cfg`'s three values through only when
  `attack_name == "cw"`.
- **Deliberately kept separate from `--attack-eps`** — CW does not take an
  `eps` argument at all (confirmed section 10.2, `hasattr(atk, "eps")` is
  `False`); merging CW's strength knobs into `attack-eps` would silently
  imply a shared semantic across attacks that does not exist. `cw_c`/
  `cw_steps`/`cw_lr` are attack-specific parameters with their own names,
  same principle as `attack-eps` already being FGSM/PGD-specific in
  practice.
- Validation: reuse the existing `require_positive_finite_float`/
  `require_positive_int` validators (same functions already backing
  `attack_temperature`/`burst_len`) at the CLI `type=` layer AND inside
  `validate_experiment_config(cfg)`; optionally also a direct check inside
  `AttackAdapter.apply()` itself (matching the existing `attack_eps`/
  `attack_temperature` dual-layer pattern), since `AttackAdapter` is called
  directly by scratch/diagnostic scripts that bypass `ExperimentConfig`
  entirely (as section 10.2's and this round's own diagnostic scripts do).
- **CSV columns**: yes, add `cw_c`/`cw_steps`/`cw_lr` to both `summary.csv`
  and `batch_summary.csv`, populated unconditionally on every row (same
  precedent as `attack_temperature`, which is present even on `attack=none`
  rows where it's inert) — keeps the CSV schema uniform across all combos
  rather than conditionally including columns only for `cw` rows.
- Out of scope for this proposal: a `--cw-c-list`-style per-combo sweep in
  `run_batch.py` (mirrors why `--attack-temperature` also has no `-list`
  variant); changing the *shipped defaults* away from `1.0/20/0.01` (a
  separate decision, informed by but not resolved by section 10.2's
  finding that `c=10,steps=100,lr=0.1` is more effective against this
  checkpoint).

---

## 12. CW CLI implementation, CW reproducibility, Top-K boundary, spectrum-sensing boundary (round 3)

Implements section 11.4's design. Code changes this round: `src/utils/
config.py`, `src/adapters/attack_adapter.py`, `src/utils/pipeline.py`,
`experiments/run_batch.py` (**no changes to `external/AWN`/
`external/adversarial-rf`**).

### 12.1 CW CLI implementation

`--cw-c` (`arg_positive_finite_float`, default `1.0`), `--cw-steps`
(`arg_positive_int`, default `20`), `--cw-lr` (`arg_positive_finite_float`,
default `0.01`) added to both `build_arg_parser` and `build_batch_arg_parser`
exactly as designed in section 11.4. `ExperimentConfig.cw_c/cw_steps/cw_lr`
added with matching defaults. `_build_torchattacks` and `AttackAdapter.
apply()` signatures extended; the `fgsm`/`pgd` branches of
`_build_torchattacks` are textually unchanged and never read the three new
parameters. Validated at three layers: CLI `type=` (identical error text to
`attack_temperature`/`burst_len`), `validate_experiment_config(cfg)`
(pipeline boundary), and directly inside `AttackAdapter.apply()` (adapter
boundary, since it's called directly by diagnostic scripts bypassing
`ExperimentConfig`). `cw_c`/`cw_steps`/`cw_lr` recorded on every row of both
`summary.csv` and `batch_summary.csv`, unconditionally (same precedent as
`attack_temperature`).

**Verified this round**:
- `--cw-c -5` rejected at CLI parse time: `cw_c must be a positive finite
  number, got -5.0`.
- FGSM run with `--cw-c 999 --cw-steps 999 --cw-lr 999` vs. the same FGSM run
  with default CW params: `summary.csv` diffs **only** in the `cw_c`/
  `cw_steps`/`cw_lr` columns themselves — every other column (`pred_clean`,
  `pred_attacked`, `pred_defended`, `iq_linf_clean_attacked`, etc.) is
  byte-identical. Confirms FGSM (and by the same code path, PGD) is
  completely unaffected by these CW-only parameters.

### 12.2 CW correctness + reproducibility test (3 param sets × 2 independent processes)

Fixed: `--seed 42 --snr 18 --mod QPSK --threshold-factor 1.5 --window-size
128 --sensing-window-size 128 --topk 10 --attack-temperature 100`, real
AWN + real attack + real Top-K throughout.

| c / steps / lr | pred_clean | pred_attacked | changed | original IQ Linf (max) | NaN/Inf | attack_backend | eval after attack |
|---|---|---|---|---|---|---|---|
| 1.0 / 20 / 0.01 (default) | `[1,1,1,1,1]` | `[1,1,1,1,1]` | 0/5 | 0.0 (bit-exact) | none | real torchattacks | yes |
| 10.0 / 100 / 0.1 | `[1,1,1,1,1]` | `[0,2,0,0,1]` | 4/5 | 2.48 | none | real torchattacks | yes |
| 100.0 / 200 / 0.1 | `[1,1,1,1,1]` | `[0,0,1,0,1]` | 3/5 | 2.03 | none | real torchattacks | yes |

All three sets: `attack_status="ok"`, `attack_backend` contains
`torchattacks` (confirmed real, no fallback), no NaN/Inf,
`attack_training_before=True`/`attack_training_after=False` (expected fresh-
wrapper pattern, eval mode correctly restored — same mechanism verified in
section 10.2), and **the real `awn.model` submodule's own `.training` flag
was independently checked `False` after each attack call** (not just the
wrapper's flag).

**Reproducibility** (two independent `python` processes per param set):
all of synthetic-IQ SHA256, `x_clean` SHA256, `x_attacked` SHA256,
`pred_clean`, `pred_attacked`, `changed_by_attack`, `logits_clean`/
`logits_attacked` SHA256, and the on-disk `summary.csv` (byte-identical,
file SHA256 match) were identical across both processes, for **all three**
param sets.

**Correctness conclusion** (per this round's explicit pass criteria —
prediction change is NOT required):
- ✅ real CW backend correctly invoked in all 3 sets (never fell back)
- ✅ `cw_c`/`cw_steps`/`cw_lr` correctly reached the constructed
  `torchattacks.CW` object (verified via `atk.c`/`atk.steps`/`atk.lr`
  attribute inspection in section 10.2's diagnostic script, reused here)
- ✅ results reproducible across independent processes, all 3 sets
- ✅ no fallback in any of the 6 runs
- ✅ no NaN/Inf in any of the 6 runs
- ✅ model correctly returns to eval mode after every attack call
- ✅ **different CW parameters produced different `x_attacked`** — set1
  (defaults) produced a bit-exact no-op (IQ Linf `0.0`), set2/set3 produced
  materially different perturbations (IQ Linf 2.48 / 2.03) and different
  `pred_attacked` patterns (4/5 vs 3/5 changed) — confirms CW's parameters
  are live, not inert, exactly as this round's pass criteria required.

### 12.3 Top-K boundary validation (algorithm unmodified — behavior observation only)

Tested directly against `dummy_topk_defense` and `TopKAdapter` (real backend
auto-selected, `torch`/`torchattacks` available), plus what `--topk`'s
current `type=int` CLI layer accepts, at `T=128`:

| topk | dummy result | real (`TopKAdapter`) result | CLI (`type=int`) |
|---|---|---|---|
| `0` | bypass (`k=T`, full FFT round-trip, ~2.4e-7 noise) | **bypass, bit-exact** (`x` returned literally, real backend's own `if topk<=0: return x`) | accepted |
| `1` | denoise, `k=1` | denoise, `k=1`, real/dummy agree | accepted |
| `10` | denoise, `k=10` | denoise, `k=10` | accepted |
| `128` (=T) | `k=128` (all bins) | `k=128` (all bins) | accepted |
| `129` (>T) | clamped to `k=128` | clamped to `k=128` | accepted |
| `1000000` (≫T) | clamped to `k=128` | clamped to `k=128` | accepted |
| `-1` | bypass (`k=T`) | **bypass, bit-exact** | accepted |
| `-5` | bypass (`k=T`) | **bypass, bit-exact** | accepted |
| `1.5` (non-integer) | `int(1.5)=1`, denoise | same | **rejected by argparse** (`invalid int value: '1.5'`) — never reaches adapter code via CLI |
| `nan` | **bypass** (`nan and nan>0` is `False`, no crash) | real raises `ValueError` internally → **caught → silent fallback to dummy** (`topk_status="fallback"`, no error surfaced) | **rejected by argparse** — unreachable via CLI |
| `inf` | **raises `OverflowError`** (`int(inf)`) | real raises `OverflowError` → caught → falls back to dummy → **dummy ALSO raises the same `OverflowError`, uncaught, escapes `TopKAdapter.apply()` entirely** | **rejected by argparse** — unreachable via CLI |
| `-inf` | bypass (`k=T`) | bypass, bit-exact | **rejected by argparse** — unreachable via CLI |

**Findings, no algorithm changes made**:
1. **`topk<=0` means "keep everything" (bypass/no-op), NOT "keep 0 bins."**
   Confirmed for both backends; the real backend does this as a literal
   early-return (`x` unchanged bit-for-bit), the dummy backend achieves the
   same numerical result via a full-bin FFT/IFFT round-trip (introducing
   ~2.4e-7 float32 noise, not bit-exact).
2. **`topk > T` is silently clamped to `T`** (all bins kept) in both
   backends — same numerical result as `topk<=0`, just reached via a
   different code path (`min(int(topk), T)` vs. an early return).
3. **Real and dummy backends behave consistently for every value except
   `nan`** — real crashes internally on `nan` (uncaught `int(nan)`
   `ValueError` inside `fft_topk_denoise` itself), which `TopKAdapter`
   catches and silently falls back to dummy (which does NOT crash on `nan`,
   since `nan and nan > 0` short-circuits to `False` before ever reaching
   `int(nan)`). Net effect: a `nan` topk request quietly downgrades to
   dummy — no exception reaches the caller, but the "real backend" request
   was silently not honored.
4. **`topk=inf` is the only value that crashes uncaught** — confirmed still
   reproducible exactly as documented in section 8 (unchanged this round).
   Both the real backend's fallback-triggering exception AND the dummy
   fallback's own attempt raise the identical `OverflowError`, so the
   second one is never caught by anything and propagates out of
   `TopKAdapter.apply()`. Reproduced directly this round via
   `ExperimentConfig(topk=float('inf'), ...)` + `run_dry_run_experiment`
   (full pipeline, not just the adapter in isolation) — the `OverflowError`
   propagates uncaught out of the entire pipeline call.
5. **`nan`/`inf`/non-integer topk are all unreachable via the actual CLI** —
   `--topk`'s current `type=int` rejects them at `argparse` parse time,
   before any adapter code runs. The crash in (4) and the silent fallback in
   (3) are only reachable via direct Python API usage (constructing
   `ExperimentConfig`/calling `TopKAdapter.apply()` directly), not via
   `run_full_experiment.py`/`run_batch.py`'s actual command line.
6. No algorithm-level changes were made to reconcile the `nan`/`inf`
   divergence between backends or to fix the `inf` double-crash — this
   section is observation only, per this round's explicit instruction.

### 12.4 Spectrum-sensing boundary validation (dummy backend, `attack=none`, `--seed 42`, `SNR=18`/`QPSK`)

25 combinations across 5 parameter groups, each via a single dry-run call
through the real `ExperimentConfig`/`run_dry_run_experiment` entrypoint (so
CLI-equivalent validation applies); representative subset re-run in a fresh
process to spot-check reproducibility (all identical).

**A. `merge_gap` = 0, 1, 5, 1000000, -1** — all 5 succeeded identically
(`n_segments=5`, region `(3734,4459)`) because this test's synthetic IQ only
ever produces a single raw region at `threshold_factor=1.5`, so
`merge_close_regions` has nothing to merge regardless of `merge_gap`'s
value — **this test did not actually exercise the merge logic itself**, only
confirmed no crash/no validation exists at any of these values (matches
`docs/parameter_validation.csv`'s existing `merge_gap` row: unvalidated,
`<=0` is a documented no-op, and this round found no crash at a very large
gap either). No error message for negative or huge values (none expected,
none occurred).

**B. `min_region_len` = 0, 1, 64, 128, 1000000, -1**:

| value | result |
|---|---|
| 0, 1, 64, 128 | OK, `n_segments=5`, region `(3734,4459)` len 725 (all below the 725-sample region, no filtering triggered) |
| 1000000 | clear `RuntimeError`: "Occupied region(s) found but all shorter than --min-region-len=1000000 samples: [(3734, 4459, 725)]..." |
| -1 | clear `ValueError`: "min_region_len must be a non-negative integer, got -1" (existing `require_nonneg_int` validation, re-confirmed) |

**C. `burst_len` = 1, 128, 600, 8192, 9000**:

| value | result |
|---|---|
| 1 | `RuntimeError`: "No occupied region detected at all..." — burst too brief for the 128-sample smoothing window to register above threshold |
| 128 | OK, `n_segments=1`, region `(3970,4223)` len 253 (smoothing spreads energy wider than the burst itself) |
| 600 | OK, `n_segments=5` (baseline) |
| 8192 (= n_samples) | `RuntimeError`: "No occupied region detected at all..." — burst fills the entire stream, so nothing is statistically distinguishable from the median-based "noise floor" |
| 9000 (> n_samples) | clear `ValueError`: "burst_len (9000) must not exceed n_samples (8192)" (existing check, re-confirmed) |

**D. `threshold_factor` = 1.5 (normal), 0.0001 (near-zero), 1000000 (huge)**:

| value | result |
|---|---|
| 1.5 | OK, `n_segments=5` (baseline) |
| 0.0001 | OK but degenerate — threshold far below noise floor, **entire stream** (`(0,8192)`) marked occupied, `n_segments=64` |
| 1000000 | `RuntimeError`: "No occupied region detected at all..." — threshold far above any real signal |

**E. `sensing_window_size` = 1, 16, 32, 64, 128, 256, 9000 (> stream length 8192)**:

| value | n_segments | region |
|---|---|---|
| 1 | 4 | `(3796,4396)` len 600 |
| 16 | 4 | `(3789,4404)` len 615 |
| 32 | 4 | `(3781,4412)` len 631 |
| 64 | 5 | `(3765,4428)` len 663 |
| 128 | 5 | `(3734,4459)` len 725 |
| 256 | 6 | `(3671,4522)` len 851 |
| 9000 | clear `ValueError`: "IQ stream (8192 samples) shorter than energy window (9000)" (existing check, re-confirmed) |

**Summary across all 25 combinations**: every case either (a) succeeded with
a well-defined `n_segments`/region result, or (b) failed with a specific,
descriptive exception already present in the code (`require_nonneg_int`/
`require_positive_int` validators, or pre-existing `RuntimeError`/
`ValueError` checks in `energy_detection.py`/`iq_source.py`) — **no silent
no-ops, no unclear tracebacks found**. All spot-checked re-runs (a fresh
process each) were identical. `x_clean.shape[1:] == (2, 128)` held for every
successful case (segment length unaffected by any of these 5 parameters, as
expected — only `--window-size` controls it).

### 12.5 Cross-reference to this round's required status labels

- **seed propagation**: PASS (section 11.1/11.2, re-confirmed section 12.2)
- **PGD reproducibility**: PASS (section 11.2, fixed and verified)
- **fair Top-K comparison**: PASS (section 11.3)
- **Top-K effectiveness**: NOT YET ESTABLISHED (sections 10.3/11.3, unchanged)
- **CW CLI**: PASS — implemented and verified this round (section 12.1)
- **CW reproducibility**: PASS — 3 param sets × 2 independent processes, all bit-identical (section 12.2)
- **Top-K boundary**: documented, no algorithm changes (section 12.3) — `topk=inf` double-crash and `topk=nan` silent-fallback both confirmed still present, both unreachable via the actual CLI
- **Spectrum-sensing boundary**: documented, no algorithm changes (section 12.4) — no silent no-ops found across 25 combinations
- **Modulation waveform implementation**: **NOT IMPLEMENTED** (unchanged — see section 5; `--mod` remains a cosmetic frequency-offset selector, not a real waveform synthesizer)

---

## 13. Top-K direct-API guard, real multi-region merge-gap test, CW fair Top-K sweep, SNR smoke matrix (round 4)

Code changes this round: `src/utils/config.py` (new `require_valid_topk`),
`src/adapters/defense_adapter.py`, `src/adapters/topk_adapter.py` (**no
changes to `external/AWN`/`external/adversarial-rf`**, and no changes to
`generate_synthetic_iq`'s single-burst default or any other pipeline
default).

### 13.1 Top-K direct-API guard (algorithm semantics preserved, new validation added)

`require_valid_topk(name, value)` (`src/utils/config.py`) is called at the
very top of both `dummy_topk_defense()` (`src/adapters/defense_adapter.py`)
and `TopKAdapter.apply()` (`src/adapters/topk_adapter.py`), **before** any
backend selection — so a rejection raises immediately and can never be
caught by `TopKAdapter`'s real-backend `except Exception:` block and
silently trigger a dummy fallback. Rejects only: non-numeric values,
non-finite values (NaN/Inf), and values with a genuine fractional part
(e.g. `1.5`). `topk<=0` (bypass) and `topk` above the FFT bin count (clamp)
keep their exact prior semantics — this function does not restrict *range*,
only *type*. The `--topk` CLI flag itself is untouched (still plain
`type=int`, which already only ever produces values this guard accepts).

**Before/after comparison** (same `T=128` test array as section 12.3):

| topk | before (dummy) | before (`TopKAdapter`) | after (both) |
|---|---|---|---|
| `-1`, `0` | bypass (~2.4e-7 noise) / bit-exact bypass | unchanged | unchanged |
| `1`...`1000000` | denoise / clamp, as before | unchanged | unchanged |
| `1.5` | silently truncated to `1` | silently truncated to `1` | **`ValueError: topk must not have a fractional part, got 1.5`** |
| `nan` | silent bypass (no crash) | **silent fallback to dummy**, no error surfaced | **`ValueError: topk must be finite (not NaN/Inf), got nan`**, identical in both backends, no fallback |
| `inf` | uncaught `OverflowError` | real crashes → falls back → dummy **also** crashes, uncaught, escapes `TopKAdapter.apply()` | **`ValueError: topk must be finite (not NaN/Inf), got inf`**, identical in both backends, no fallback, no crash |
| `'abc'` | (not previously tested) | (not previously tested) | **`ValueError: topk must be numeric, got 'abc'`** |

Also re-verified through the **full pipeline** (`ExperimentConfig(topk=
{inf,nan,1.5}, ...)` → `run_dry_run_experiment`): all three now raise a
clean `ValueError` from inside `TopKAdapter.apply()`/`dummy_topk_defense()`
instead of an uncaught `OverflowError` propagating out of the entire
pipeline (the pre-fix behavior, confirmed present in section 12.3).

### 13.2 merge-gap actual multi-region merging test (dual-burst, scratch-only)

**Scratch diagnostic only** (`merge_gap_dual_burst_probe.py`) — does **not**
modify `generate_synthetic_iq` or any pipeline default, which remains
single-burst. A local two-burst generator (same noise/carrier formula as
`generate_synthetic_iq`, applied twice) places two 256-sample bursts at
SNR=30dB with a controllable inter-burst gap, then runs the real
`energy_detect → mask_to_regions → merge_close_regions → filter_by_min_length
→ segment_regions` pipeline directly. `sensing_window=1` (no smoothing) was
required to keep gaps as small as 1 sample meaningfully distinguishable at
the raw-mask level (any smoothing kernel ≥ the gap size would bridge it
before `merge_close_regions` ever runs); `threshold_factor=20` was needed to
suppress noise-driven fragmentation at `window=1` (at `window=1`,
per-sample power is exponentially distributed for pure noise, so
`P(false-positive) = 2^(-threshold_factor)` regardless of noise scale — the
session's usual `threshold_factor=1.5` let ~35% of individual noise samples
spuriously exceed threshold at `window=1`, versus `<1e-6` at `20`).

**Full 4×4 grid**, gap ∈ {0,1,5,20} × merge_gap ∈ {0,1,5,20}:

| gap＼merge_gap | 0 | 1 | 5 | 20 |
|---|---|---|---|---|
| 0 | merged (trivial — bursts touch, 1 raw region already) | merged | merged | merged |
| 1 | **not merged** (2 regions) | merged | merged | merged |
| 5 | not merged | **not merged** | merged | merged |
| 20 | not merged | not merged | not merged | **merged** (boundary case) |

All 16 combinations matched the expectation `merged ⟺ gap <= merge_gap`
**exactly, zero mismatches** — including the exact boundary
`gap=20, merge_gap=20` correctly merging (confirms `<=`, not `<`).
`gap=0` is a degenerate case (the two bursts are already adjacent, so
`mask_to_regions` sees one continuous raw region regardless of
`merge_gap` — merging is moot, not actually exercised by that row).
`n_segments=4` for every one of the 16 combinations (merged or not, the
total occupied length only ever changes by the tiny gap size, negligible
against the 128-sample window). No silent no-ops; the full 16-combo run was
repeated in a second independent process and every field (region
boundaries, segment count, `x_clean` SHA256) was identical.

### 13.3 CW fair Top-K sweep (single attacked IQ generated once per param set, real backend)

Diagnostic script only (`cw_fair_topk_sweep.py`, scratchpad), same
methodology as the earlier FGSM/PGD fair sweep (section 11.3): clean IQ and
CW-attacked IQ each computed **exactly once** per `(c, steps, lr)` set
(`--seed 42`, `SNR=18`, `QPSK`, `threshold-factor=1.5`,
`sensing-window-size=128`, `attack-temperature=100`), then the identical
`x_adv` array run through the real `TopKAdapter` at K=10/20/30/40.

| c/steps/lr | pred_attacked (fixed once) | K=10 recovered | K=20 recovered | K=30 recovered | K=40 recovered | all 4 K's defended-IQ hashes distinct? |
|---|---|---|---|---|---|---|
| 10/100/0.1 | `[0,2,0,0,1]` (4 attacked) | **4/4** | **4/4** | **4/4** | **4/4** | yes |
| 100/200/0.1 | `[0,0,1,0,1]` (3 attacked) | **2/3** | 1/3 | **2/3** | 1/3 | yes |

All combinations: real `attack_backend`/`topk_backend` confirmed (no
fallback), no NaN/Inf, `attack_training_after=False` AND the real
`awn.model` submodule's own `.training` independently confirmed `False`
after the attack call. Reproduced in a second independent process —
`cw_fair_topk_sweep.csv` byte-identical.

**Observation, not a general claim**: recovery was notably *higher* here
than in the earlier FGSM/PGD fair sweep (section 11.3, which found only
1/4 and 1/2 partial recovery) — the `c=10,steps=100,lr=0.1` set recovered
**all** attacked segments at every K tested. This is a single 5-segment
sample at one SNR/eps/temperature/CW-parameter combination, not a
systematic sweep — it does **not** establish "Top-K defends against CW"
as a general claim, but it is a genuine, fairly-measured data point that a
future systematic recovery-rate study should account for (CW's recovery
pattern in this sample looks different from FGSM/PGD's, not uniformly
worse or better).

### 13.4 SNR smoke sweep matrix (4 SNR × 4 attacks × 2 topk = 32 combinations)

`python3 experiments/run_batch.py --dry-run --snr-list="-10,0,10,18"
--mod-list QPSK --attack-list "none,fgsm,pgd,cw" --topk-list "10,20"
--threshold-factor 1.5 --window-size 128 --sensing-window-size 128
--burst-len 600 --use-real-awn --use-real-attack --use-real-topk
--attack-eps 0.5 --attack-temperature 100 --cw-c 10 --cw-steps 100
--cw-lr 0.1 --seed 42 --output-dir results/snr_smoke_sweep` (note
`--snr-list="-10,..."` needs the `=` form — the leading `-` would otherwise
be misparsed as an option, a known argparse quirk, not a bug, documented
since section 2).

**8 combinations failed** (all `SNR=-10`, all 4 attacks × 2 topk): clear,
non-silent `RuntimeError: No occupied region detected at all...` printed to
stderr and preserved by `run_batch.py`'s existing per-combo
`try/except...continue` (no `summary.csv` written for these — confirmed no
misleading partial output); matches the pre-existing documented behavior
for `SNR=-10` at `threshold_factor=1.5` (section 9's historical-values
table). **24 combinations succeeded** (`SNR ∈ {0,10,18}` × 4 attacks × 2
topk): all real backends confirmed (`awn_backend`/`attack_backend`/
`topk_backend` columns), `seed=42` recorded on every row, zero NaN/Inf
across all 24×{4 or 5} segments.

| SNR | attack | topk | n_seg | changed/total | recovered |
|---|---|---|---|---|---|
| 0 | none | 10/20 | 4 | 0/4 | 0 |
| 0 | fgsm | 10/20 | 4 | 1/4 | 0 |
| 0 | pgd | 10/20 | 4 | 4/4 | 0 |
| 0 | cw | 10/20 | 4 | 3/4 | 2 |
| 10 | none | 10/20 | 5 | 0/5 | 0 |
| 10 | fgsm | 10/20 | 5 | 2/5 | 0 |
| 10 | pgd | 10/20 | 5 | 5/5 | 1 |
| 10 | cw | 10/20 | 5 | 5/5 | 4 |
| 18 | none | 10/20 | 5 | 0/5 | 0 |
| 18 | fgsm | 10 | 5 | 4/5 | 1 |
| 18 | fgsm | 20 | 5 | 4/5 | 0 |
| 18 | pgd | 10/20 | 5 | 2/5 | 1 |
| 18 | cw | 10/20 | 5 | 4/5 | 4 |

(Rows shown collapsed where `topk=10` and `topk=20` gave identical
`changed`/`recovered` counts; `SNR=18,fgsm` is the one case in this smoke
matrix where `topk=10` and `topk=20` diverged.) Reproduced in a full second
independent `run_batch.py` invocation — `batch_summary.csv` identical
(excluding the `output_dir` path column) and two representative
`summary.csv` files spot-checked byte-identical.

**This is a smoke matrix, not a formal experiment** — 32 combinations at
one seed, one modulation, one threshold-factor, one set of attack
hyperparameters. It confirms the full real pipeline (sensing → real AWN →
real attack → real Top-K) runs correctly and reproducibly across an SNR
range including a legitimate failure mode at very low SNR, and that CW
(with effective, non-default parameters) shows a qualitatively different —
generally higher — recovery pattern than FGSM/PGD in this sample. It does
**not** establish general attack-effectiveness or defense-effectiveness
trends across SNR.

### 13.5 Cross-reference to this round's required status labels

- **CW CLI**: PASS (implemented and verified section 12.1, reused successfully throughout this round)
- **CW reproducibility**: PASS (section 12.2, reused successfully throughout this round)
- **CW parameter sensitivity**: PASS (section 12.2 — different c/steps/lr produce genuinely different `x_attacked`/predictions; reconfirmed section 13.3)
- **Top-K normal boundary**: PASS (section 12.3 — bypass/clamp semantics confirmed correct and unchanged)
- **Top-K NaN/Inf direct API**: FIXED this round (section 13.1) — explicit `ValueError`, no silent fallback, no crash; CLI unaffected
- **merge-gap actual merging**: PASS this round (section 13.2) — real multi-region merge/no-merge boundary confirmed exactly on a purpose-built (scratch-only) dual-burst signal
- **CW fair Top-K sweep**: DONE this round (section 13.3) — fair, single-attacked-IQ comparison completed; results reported as observations, not claimed as established defense effectiveness
- **SNR smoke sweep**: DONE this round (section 13.4) — 24/32 combinations succeeded with real backends and full reproducibility; 8 failed combinations (all SNR=-10) preserved with clear errors, not skipped
- **modulation waveform**: NOT IMPLEMENTED (unchanged, section 5)
- **formal full batch** (SNR × modulation × attack × eps × topk): **NOT STARTED** — explicitly out of scope this round

---

## 14. RadioML (RML2016.10a) real-sample input, ground-truth sensing metrics (round 5)

New files: `src/sensing/radioml_source.py`, `src/sensing/ground_truth_metrics.py`.
Modified: `src/utils/config.py`, `src/utils/pipeline.py`, `experiments/
run_batch.py`, `docs/experiment_design.md` (**no changes to
`external/AWN`/`external/adversarial-rf`**; the existing synthetic source
path is untouched and unaffected — verified via regression check, section
14.4).

### 14.1 RadioML dataset inventory (read-only, before any code was written)

1. **Dataset file presence**: `RML2016.10a_dict.pkl` found at
   `/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl` (640,919,653
   bytes). **Not** inside this repo, and **not** inside `external/
   adversarial-rf` (the pinned submodule) — `external/adversarial-rf/data/`
   only contains a `.gitignore` (`*` ignored, dataset never committed).
   `/home/xiaomi/adversarial-rf` is a **separate, standalone checkout of the
   same upstream GitHub repo** (`nigelzzz/adversarial-rf`, HEAD
   `70036bc817c595a89e666c13907066edc460763d`) — the same location this
   session's real-backend venv (`/home/xiaomi/adversarial-rf/.venv`) lives
   in, but a distinct git checkout from the `external/adversarial-rf`
   submodule pinned in this repo. Because of this, `--dataset-path` is a
   required, absolute, external CLI argument, not a hardcoded relative
   path — verified this session that reusing `external/adversarial-rf/
   data_loader/data_loader.py:Load_Dataset` directly is not possible
   without either copying the ~640MB file into the submodule's own `data/`
   or modifying `Load_Dataset`'s hardcoded `'./data/%s'` relative path
   (both out of scope — no external repo changes permitted).
2. **Format/structure**: a plain Python dict, `{(mod: str, snr: int):
   ndarray[1000, 2, 128] float32}`, loaded with `pickle.load(f,
   encoding='latin1')` (str keys; `external/adversarial-rf`'s own loader
   uses `encoding='bytes'`, producing `bytes` keys instead — same
   underlying data, this repo's own `radioml_source.py` picks `'latin1'`
   so `--dataset-mod QPSK` can be a plain CLI string with no bytes
   juggling). **220 keys** = 11 modulations × 20 SNR values
   (`-20` to `18` dB, step `2`). Each `(mod, snr)` block has exactly
   **1000 samples**.
3. **Modulation/SNR label storage**: encoded entirely in the dict **key**
   (a `(mod, snr)` tuple) — there is no separate per-sample label array;
   every one of the 1000 samples in a given block shares that block's
   `(mod, snr)` label implicitly, by position.
4. **Loader reuse**: `Load_Dataset` (`external/adversarial-rf/data_loader/
   data_loader.py:6-70`) is **not directly reusable as-is** (hardcoded
   relative path, as above), but its **essential, correctness-critical
   logic — the class-ordering convention — was reused/replicated**
   verbatim into this repo's own `RML2016_10A_CLASSES` constant (see point
   6 below), rather than re-derived or guessed.
5. **Per-sample shape**: confirmed **`[2, 128]` float32** for every sample
   checked this session (`arr[i].shape == (2, 128)`) — matches
   `external/adversarial-rf/util/config.py:51`'s `signal_len = 128` for
   `2016.10a`, independently re-confirmed here directly against the actual
   dataset array shape, not just the config citation.
6. **AWN checkpoint class ordering — from source code, not memory**,
   cross-checked at **three independent locations** in `external/
   adversarial-rf` (submodule, pinned commit `ced705e`), all identical, no
   conflicts found:
   - `data_loader/data_loader.py:13-14` — used by `Load_Dataset`, which is
     called directly from `main.py:205`, **the actual training entry
     point** (confirmed via `grep`, not assumed).
   - `util/config.py:52` — an independent duplicate declaration.
   - Six further `plot_*.py` analysis scripts, all consistent.

   **`RML2016_10A_CLASSES`** (now also in this repo's `src/sensing/
   radioml_source.py`):
   ```
   QAM16:0  QAM64:1  8PSK:2  WBFM:3  BPSK:4
   CPFSK:5  AM-DSB:6  GFSK:7  PAM4:8  QPSK:9  AM-SSB:10
   ```
   This is a training-time convention recorded in code, not something
   embedded in the checkpoint file itself (raw tensor weights carry no
   label metadata) — the evidence is that every script in the submodule
   that interprets this checkpoint's output logits agrees on this exact
   ordering, with `main.py`'s training call being the authoritative one.

Sample power-scale check (informational, used to calibrate embedding —
section 14.2): mean per-sample power ranged ~6.8e-5 to ~3.5e-4 across the
`(mod, snr)` combinations checked, roughly SNR-independent in absolute
terms (RadioML's own generation process normalizes total power, not just
the message-to-noise ratio) — this is why `embed_snr_margin` (14.2) scales
relative to each loaded sample's own measured power rather than using a
fixed background-noise constant.

### 14.2 RadioML IQ source (`--iq-source radioml`)

New CLI flags (`src/utils/config.py`, mirrored in both `build_arg_parser`
and `run_batch.py`'s `build_batch_arg_parser`): `--iq-source {synthetic,
radioml}` (default `synthetic`, reproduces all prior behavior exactly),
`--dataset-path`, `--dataset-mod`, `--dataset-snr`, `--sample-index`
(`arg_nonneg_int`), `--embed-snr-margin` (`arg_positive_finite_float`,
default `20.0`). `ExperimentConfig` gained matching fields
(`iq_source/dataset_path/dataset_mod/dataset_snr/sample_index/
embed_snr_margin`). `validate_experiment_config` rejects an invalid
`iq_source` value and requires `dataset_path`/`dataset_mod`/`dataset_snr`
to all be set when `iq_source=='radioml'` (existence/key-validity of the
dataset file itself is checked later, at load time, since that requires
actually opening the pickle).

**Explicit separation, not metadata-only** (the round's core requirement):
`--mod`/`--snr` continue to control ONLY the synthetic generator and are
**completely unused** when `iq_source=='radioml'` — `src/utils/
pipeline.py`'s `run_dry_run_experiment` branches entirely around
`generate_synthetic_iq` in that mode. `--dataset-mod`/`--dataset-snr`
instead select a **real, different array** from the pickle — verified
empirically (section 14.4): different `--dataset-mod` values produce
different `original_sample_sha256`, not just a different label string.

**Pipeline** (`src/sensing/radioml_source.py`):
```
load_radioml_sample(dataset_path, mod, snr, sample_index)   [2,128] float32
  raises ValueError (not silent) for unknown mod/snr or out-of-range
  sample_index, listing the valid options
  -> radioml_sample_to_iq()                                  [128] complex64
  -> embed_sample_in_noise(n_samples, embed_snr_margin, seed)
       background noise power = burst's own measured power / embed_snr_margin
       (relative, not fixed -- see 14.1's power-scale note)
       burst placed at a SEEDED-RANDOM position (reproducible, but not
       looked up by the sensing stage below)
       => long complex64 IQ [n_samples], true_start/true_end
  -> validate_iq
  -> energy_detect -> mask_to_regions -> merge_close_regions
       -> filter_by_min_length                                [EXACT SAME
       code path as the synthetic source from here on -- no branching]
  -> segment_regions -> normalize_segments -> to_awn_input     x_clean [N,2,128]
  -> compute_sensing_ground_truth_metrics(true_start, true_end, regions)
       [radioml mode only -- section 14.3]
  -> AWN / attack / Top-K (unchanged)
```

**Recorded** (every `summary.csv` row, present but `None` for synthetic-mode
rows): `source_type`, `dataset_path`, `dataset_mod`, `dataset_snr`,
`sample_index`, `original_sample_sha256`, `long_iq_sha256`,
`embed_snr_margin`, `true_burst_start`, `true_burst_end`, plus all of
section 14.3's ground-truth metric columns. `snr_db`/`mod` columns are
still present and populated (from `cfg.snr`/`cfg.mod`, the synthetic
generator's own inputs/defaults) but are explicitly meaningless in radioml
mode — `dataset_mod`/`dataset_snr` are the authoritative ground truth in
that mode, never conflated with `snr_db`/`mod`.

### 14.3 Ground-truth sensing metrics — definitions

`src/sensing/ground_truth_metrics.py:compute_sensing_ground_truth_metrics
(true_start, true_end, detected_regions)`. All intervals are half-open
`[start, end)`. `T = [true_start, true_end)`; `D` = the **best-matching**
detected region (largest overlap with `T`; ties broken by smallest gap to
`T`; `None` if `detected_regions` is empty).

| Metric | Formula |
|---|---|
| `intersection_length` | `max(0, min(true_end, D_end) - max(true_start, D_start))` |
| `true_burst_length` | `true_end - true_start` |
| `detected_region_length` | `D_end - D_start` |
| `detection_success` | `intersection_length > 0` for the best match |
| `captured_signal_ratio` (recall) | `intersection_length / true_burst_length` |
| `extra_captured_noise_ratio` (1 − precision) | `(detected_region_length - intersection_length) / detected_region_length` |
| `missed_sample_count` | `true_burst_length - intersection_length` |
| `false_occupied_sample_count` | `detected_region_length - intersection_length` |
| `start_boundary_error` (signed) | `D_start - true_start` (positive = detected region starts late / missed the leading edge; negative = started early / extra noise captured ahead) |
| `end_boundary_error` (signed) | `D_end - true_end` (same convention, trailing edge) |

If no detected region overlaps `T` at all, `captured_signal_ratio=0.0`,
`missed_sample_count=true_burst_length`, and the boundary-error/
extra-noise-ratio fields are `None` (nothing to measure them against). If
`detected_regions` is empty entirely, every field beyond
`true_start`/`true_end`/`true_burst_length` is `None`/`False`/`0`.

### 14.4 Small real-data functional test (12 combinations: 2 mods × 2 SNR × 3 samples)

`QPSK`/`BPSK` × `dataset-snr∈{0,18}` × `sample-index∈{0,1,2}`, `attack=none`,
`--use-real-awn`, `--seed 42`, `--threshold-factor 1.5 --window-size 128
--sensing-window-size 128`. **Regression check first**: an unrelated
synthetic-mode run at the same fixed params gave the exact same occupied
region `(3734, 4459)` this session has produced dozens of times before —
confirms the RadioML integration did not alter the synthetic path.

All 12 RadioML combinations: real `awn_backend` (`AWN`, `status=ok`, no
fallback), `x_clean.shape == (1, 2, 128)`, `detection_success=True`
(11/12 with `captured_signal_ratio=1.0`; **`BPSK, snr=18, idx=0` captured
only 0.625** — a genuine, real per-sample energy variation, not an error;
still `detection_success=True`). `dataset_mod`/`dataset_snr` (ground truth)
and `pred_clean` (AWN's own guess) both written to every row — **AWN
correctness was never used as a pass/fail criterion** for this functional
test (several `pred_clean` values visibly disagree with the true label,
e.g. `QPSK` samples predicted as class `1`/`8`, not `9` — left as-is,
reported not filtered).

| check | result |
|---|---|
| Different modulations → different `original_sample_sha256` (same snr/idx) | ✅ confirmed distinct (`QPSK` vs `BPSK` at `snr=0,idx=0`) |
| Different `sample_index` → different `original_sample_sha256` (same mod/snr) | ✅ confirmed 3 pairwise-distinct hashes (`QPSK,snr=18,idx=0/1/2`) |
| Same sample + seed, 2 independent processes | ✅ `summary.csv` byte-identical (`QPSK,snr=18,idx=0` re-run) |
| Real AWN backend, no fallback, all 12 combos | ✅ confirmed |
| `x_clean` shape `[N,2,128]` | ✅ confirmed, all 12 |
| Sensing found the embedded region | ✅ `detection_success=True`, all 12 |

### 14.5 merge-gap multi-RadioML-burst main-pipeline integration — DESIGN ONLY, not implemented

Per this round's instructions, a design proposal, deliberately not built —
extending `embed_sample_in_noise` (single burst) touches the core pipeline
branch added this round non-trivially (variable burst count, per-burst
metadata, many-to-many region↔truth attribution), so it doesn't meet the
"very small, isolated change" bar this session has used to decide
design-only vs. implement-now (same bar applied to the CW CLI design in an
earlier round).

1. **Multiple bursts, each independently parameterized**: a new
   `embed_multiple_samples_in_noise(samples: List[[2,128]], n_samples,
   embed_snr_margin, seed, min_gap)` — takes a **list** of already-loaded
   RadioML samples (each independently chosen via its own `mod`/`snr`/
   `sample_index` by the caller, so yes, each burst can have a different
   modulation, SNR, and, via seeded-random non-overlapping placement,
   position), returns the long IQ array plus a **list** of per-burst
   ground-truth dicts (`burst_id, mod, snr, sample_index, true_start,
   true_end`). `min_gap` enforces a minimum separation between placed
   bursts at generation time (independent of, and not to be confused
   with, `--merge-gap`, which governs *detection-time* merging).
2. **CLI**: extend the existing comma-list convention (`run_batch.py`'s
   `--snr-list` etc.) with `--dataset-mod-list`/`--dataset-snr-list`/
   `--sample-index-list` (equal-length comma lists, one entry per burst,
   `len()` implicitly determines burst count) as the minimal option; a
   `--burst-spec-path` (JSON/YAML list of `{mod,snr,sample_index}`
   objects) is a documented alternative for larger burst counts where
   comma-lists become unwieldy — not chosen as the primary design to stay
   consistent with this repo's existing list-flag convention.
3. **Ground truth storage**: `result["ground_truth_bursts"]` (a list, one
   dict per input burst, same shape as todays's single-burst metric dict)
   replaces today's single `result["ground_truth"]`. `summary.csv` is
   per-*segment*, not per-burst — each segment's row would need a
   `source_burst_id` (or `None` if its detected region doesn't confidently
   attribute to one true burst) rather than trying to cram a variable-length
   burst list into fixed CSV columns; a **separate** `bursts_summary.csv`
   (one row per true burst, analogous to how `batch_summary.csv` is
   separate from per-combo `summary.csv`) is the natural place for the
   per-burst ground-truth metrics themselves.
4. **Detection probability / false-alarm rate**:
   - **Region-level**: `Pd = (# true bursts with detection_success=True) /
     (# true bursts)`; `Pfa_region = (# detected regions with ZERO
     overlap with any true burst) / (# detected regions)`.
   - **Sample-level** (the more standard radar/spectrum-sensing
     definition): `Pfa_sample = (# background/noise samples incorrectly
     marked occupied) / (# background/noise samples in the stream)` — this
     needs the union of all detected regions vs. the union of all true
     burst intervals, computed once per stream (not per burst).
   - Both should be reported — they measure different things (a single
     region spanning multiple close bursts inflates region-level Pfa's
     denominator interpretation; sample-level Pfa is scale-invariant to
     how many bursts happen to be present).
5. **One detected region spanning multiple true bursts** (exactly the
   scenario `--merge-gap` creates when it merges two nearby detections):
   propose computing a **bipartite overlap table** — for every (detected
   region, true burst) pair, `intersection_length` as already defined in
   14.3 — then classifying each detected region as *clean single-burst*
   (nonzero overlap with exactly one true burst and no other detected
   region also claims that burst), *merged multi-burst* (nonzero overlap
   with 2+ true bursts — the direct signature of an over-aggressive
   `--merge-gap` for this burst spacing), or *false positive* (zero
   overlap with any true burst); symmetrically classify each true burst as
   *captured cleanly*, *captured but merged with a neighbor*, or *missed*.
   This directly answers "how does `--merge-gap` interact with multiple
   ground-truth bursts" empirically once built, rather than assuming an
   answer — not implemented this round.

### 14.6 Cross-reference to this round's required status labels

- **Top-K boundary**: PASS (unchanged from round 4, section 13.1)
- **merge-gap algorithm**: PASS (unchanged from round 4, section 13.2 — dual-burst scratch test)
- **merge-gap main-pipeline integration**: **NOT TESTED** (design only, section 14.5 — not implemented)
- **CW fair Top-K**: PASS, small-sample only (unchanged from round 4, section 13.3)
- **SNR smoke matrix**: PASS with expected sensing failures at SNR=-10 (unchanged from round 4, section 13.4)
- **RadioML source**: PASS — implemented, real dataset located and inventoried, class ordering verified from source (not memory), `--iq-source radioml` fully wired end-to-end through real AWN, 12/12 functional-test combinations succeeded, reproducible, no fallback (section 14.1–14.4)
- **modulation truthfulness**: **PASS for the RadioML source specifically** (real, distinct, verifiably-different IQ per modulation label, confirmed via SHA256) — the **synthetic source remains cosmetic-only** exactly as documented in section 5; "modulation truthfulness" as a repo-wide property is not a blanket PASS, only true when `--iq-source radioml` is used
- **formal full batch**: **NOT STARTED** (unchanged)

---

## 15. Multi-burst RadioML source, truth-to-detection matching, merge-gap main-pipeline validation (round 6)

New files: none. Modified: `src/sensing/radioml_source.py` (new
`embed_multiple_samples_in_noise`), `src/sensing/ground_truth_metrics.py`
(new `compute_multi_burst_sensing_metrics`), `src/utils/config.py`,
`src/utils/pipeline.py`, `experiments/run_batch.py`. New reference file:
`docs/radioml_class_mapping.csv`. **No changes to `external/AWN`/
`external/adversarial-rf`.**

### 15.0 Fresh re-inventory of the actual code (not the prior summary)

Re-read directly from the current files before writing any code this round:

1. **Single-burst embedding**: `embed_sample_in_noise()`
   (`radioml_source.py`) draws the ENTIRE `n_samples`-length background
   noise array from one `np.random.default_rng(seed)`, then uses the SAME
   `rng` object's next draw (`rng.integers(0, max_start+1)`) to pick
   `true_start` uniformly at random — a seeded-random but reproducible
   position, never looked up by the sensing stage below it.
2. **`true_start`/`true_end` → `summary.csv` path**: `embed_meta` (from
   `embed_sample_in_noise`) is spread into `gen_meta`/`radioml_meta` in
   `pipeline.py`, then passed as two plain scalars into
   `compute_sensing_ground_truth_metrics(true_start, true_end, regions)`,
   whose return dict's `true_start`/`true_end` keys are written into
   `summary.csv`'s `true_burst_start`/`true_burst_end` columns — identically
   on **every** segment row (there being only one true burst, this was never
   ambiguous before this round).
3. **`merge_close_regions` → `filter_by_min_length` order**
   (`pipeline.py`, unchanged before and after this round):
   `mask_to_regions` → `merge_close_regions(merge_gap)` →
   `filter_by_min_length(min_region_len)` — merging always happens
   **before** the length filter, so a `--merge-gap` that successfully joins
   two short raw regions can rescue them from being dropped by
   `--min-region-len`, and conversely a region that's long enough on its
   own but ends up as a small overlap after failing to merge could still be
   dropped.
4. **Pre-round data-structure capacity for multiple truth bursts**: **none**
   — `embed_sample_in_noise` only accepts one `[2,128]` array,
   `compute_sensing_ground_truth_metrics` only accepts one
   `(true_start, true_end)` scalar pair, and nothing in `pipeline.py` ever
   recorded which detected *region* a given *segment* came from (needed the
   moment more than one region can exist for more than one reason).
5. This inventory is what section 15.1–15.3 below are built directly on top
   of, unmodified for `num_bursts<=1` (see the regression checks in 15.1).

### 15.1 Multi-burst RadioML source (`--num-bursts > 1`)

New CLI flags: `--num-bursts` (default `1` — the exact prior single-burst
code path, byte-for-byte unaffected, is what runs whenever this is
omitted), `--dataset-mod-list`/`--dataset-snr-list`/`--sample-index-list`
(comma-separated, length must equal `--num-bursts`, REQUIRED when
`--num-bursts>1`), `--min-burst-gap`/`--max-burst-gap` (gap sampled
uniformly per burst including a leading gap before the first; setting them
equal gives an exact, deterministic gap), `--burst-gap-list` (optional
exact per-burst gap list, overriding random sampling — needed for e.g. Case
3 below, where two different exact gaps are needed in the same run),
`--burst-power-scale-list` (optional exact per-burst amplitude multiplier,
applied before the shared noise floor is computed — needed for Case 4
below; real RadioML samples alone have only ~1.35x block-mean power spread
across mod/snr combinations, not enough on its own to reliably produce an
undetected burst).

`embed_multiple_samples_in_noise()` places bursts strictly back-to-back
(cursor-based, gap ≥ 0 guaranteed by construction — **no overlap is
possible**, no separate check needed), and computes ONE shared background
noise level for the whole stream from the MEAN power across all (possibly
scaled) bursts, since a single real capture has a single noise floor.

**Verified this round** (all via the real `run_full_experiment.py` CLI, real
AWN, `--seed 42`):
- **Regression, both prior paths unaffected**: an unrelated synthetic-mode
  run gave the same region `(3734, 4459)` seen dozens of times this
  session; an unrelated single-burst radioml-mode run gave the exact same
  `detection_success=True, captured_signal_ratio=1.0, start_err=-56,
  end_err=62` seen in the very first RadioML round.
- **Every burst genuinely read from the dataset**: `bursts_summary.csv`'s
  `original_sample_sha256` column is confirmed distinct per (mod, snr,
  sample_index) triple — not metadata-only.
- **Non-overlapping**: guaranteed by construction; empirically confirmed
  (3-burst smoke test) — regions `[(197,336),(522,663),(848,992)]`, no
  overlap.
- **Same seed → bit-for-bit identical**: `bursts_summary.csv`/
  `regions_summary.csv`/`summary.csv` byte-identical across two independent
  processes (3-burst smoke test, re-verified for every one of the 5
  Case 1–5 test runs in section 15.3).
- **Different seed → different outcome**: with a gap *range* (not a fixed
  exact value), three different seeds gave three genuinely different sets
  of burst positions; even with an *exact* fixed gap (seed-invariant
  positions by construction), the underlying background noise realization
  — and therefore `long_iq_sha256` — still differs by seed, confirmed.
- **Single-burst mode completely undisturbed**: confirmed via the
  regression checks above, run both before and after every code change
  this round.

### 15.2 Truth-to-detection matching + formal metrics — method and formulas

`compute_multi_burst_sensing_metrics(true_bursts, detected_regions,
n_samples)` (`ground_truth_metrics.py`). **Method chosen: full bipartite
overlap enumeration**, not a strict one-to-one match (IoU-max or Hungarian
assignment) — a strict 1:1 match cannot correctly represent a region that
genuinely overlaps two neighboring bursts merged by `--merge-gap`, or a
burst genuinely split across two detected regions, both of which this
round's test cases are specifically designed to produce. Every `(burst,
region)` pair with `intersection_length > 0` is a real edge; a burst or
region can have zero, one, or multiple edges. Detected regions are treated
as pairwise non-overlapping (guaranteed by `merge_close_regions`/
`filter_by_min_length`'s own construction), so summing per-pair
intersections is equivalent to a proper union.

**Per-burst fields** (`bursts_summary.csv`, one row per TRUE burst, always
— including missed bursts, which have zero representation in the
per-segment `summary.csv` since a missed burst has no detected region and
therefore no segments):

| field | formula |
|---|---|
| `intersection_length` | `sum_j intersection_length(i,j)` over all detected regions `j` |
| `detection_success` | `intersection_length > 0` |
| `matched_region_ids` | `[j : intersection_length(i,j) > 0]`, sorted |
| `matched_region_id` | the single `j` in `matched_region_ids` with the largest `intersection_length(i,j)` ("primary" match; `None` if unmatched) |
| `captured_signal_ratio` | `intersection_length / true_burst_length` |
| `missed_sample_count` | `true_burst_length - intersection_length` |
| `start_boundary_error` / `end_boundary_error` | `matched_region.start - true_start` / `matched_region.end - true_end` (signed; computed against `matched_region_id` only; `None` if unmatched) |

**Per-region fields** (`regions_summary.csv`, one row per DETECTED region,
always — including false-alarm regions with zero matched bursts):

| field | formula |
|---|---|
| `matched_burst_ids` | `[i : intersection_length(i,j) > 0]`, sorted — 0 entries = **false alarm**, 1 = clean match, 2+ = this region **merged** multiple true bursts |
| `false_occupied_sample_count` | `detected_region_length - (sum_i intersection_length(i,j))` |
| `extra_captured_noise_ratio` | `false_occupied_sample_count / detected_region_length` |

**Aggregate fields** (printed to console + `result["multi_burst_result"]
["aggregate"]`; denominators spelled out explicitly, no dedicated CSV for
these ~11 scalars — already fully visible in the console summary and the
returned result dict):

| field | formula | denominator note |
|---|---|---|
| `detection_probability` (Pd) | `num_matched_bursts / num_truth_bursts` | `None` if zero truth bursts |
| `false_alarm_region_rate` | `num_false_alarm_regions / num_detected_regions` | `None` if zero detected regions |
| `sample_level_false_positive_rate` | `(sum of every region's false_occupied_sample_count) / (n_samples - sum of every true burst's length)` | classic sample-level Pfa — fraction of TRUE BACKGROUND samples wrongly marked occupied; `None` if bursts fill the entire stream |
| `sample_level_false_negative_rate` | `(sum of every burst's missed_sample_count) / (sum of every true burst's length)` | sample-weighted 1−recall across all bursts; `None` if zero total truth length |
| `mean_captured_signal_ratio` | simple mean of per-burst `captured_signal_ratio` | NOT sample-weighted (distinct from the FNR above) |
| `mean_abs_start_boundary_error` / `mean_abs_end_boundary_error` / `mean_abs_boundary_error` | mean of `abs(...)` over MATCHED bursts only (the third pools both edges together) | `None` if zero bursts matched |

**Verified this round** with a synthetic 4-region/3-burst scenario
(intervals only, not run through the real pipeline) exercising all 5
required scenarios simultaneously in one call: region merging 2 bursts
(`matched_burst_ids=[0,1]`), a burst split across 2 regions
(`matched_region_ids=[1,2]`), a false-alarm region
(`matched_burst_ids=[]`), correct `Pd=1.0`/`false_alarm_region_rate=0.25`/
`sample_level_false_positive_rate`/`sample_level_false_negative_rate`, all
computed and cross-checked by hand.

### 15.3 merge-gap main-pipeline test cases (all via the real CLI, real AWN, `--seed 42`)

**Calibration first** (not guessing): 2-burst runs at `merge-gap=0` (no
merging) across true gaps 20/60/100/300 samples showed the *detected* gap
(after `sensing-window-size=16` smoothing widens each region) is
consistently ~11–13 samples smaller than the *true* gap — used to choose
gap values with a clear margin on either side of each case's `--merge-gap`.
`sensing-window-size=16`/`threshold-factor=1.5` (unless stated) were chosen
because they cleanly separate two 128-sample RadioML bursts without the
noise-fragmentation problem `threshold-factor=1.5` combined with
`sensing-window-size=1` produced in section 12.4's dual-burst scratch test.

| Case | Setup | Result |
|---|---|---|
| **1** (gap > merge-gap, stay separate) | 2 bursts, `--burst-gap-list 300,300 --merge-gap 50` | `num_detected_regions=2`, `num_matched_bursts=2`, `num_missed_bursts=0` — regions `[(250,436),(723,863)]`, stayed separate as required |
| **2** (gap ≤ merge-gap, merge) | 2 bursts, `--burst-gap-list 300,60 --merge-gap 50` (inter-burst gap 60 → detected ≈48 ≤ 50) | `num_detected_regions=1`, both bursts matched to that one region — merged as required |
| **3** (3 bursts: first two merge, third separate) | `--burst-gap-list 300,60,300 --merge-gap 50` | `num_detected_regions=2`: region 0 = bursts {0,1} merged (`matched_burst_ids=[0,1]`), region 1 = burst 2 alone — exactly as required |
| **4** (low-energy burst missed, other detected) | 2 bursts, `--burst-power-scale-list 0.1,1.0` (burst 0 artificially weakened) | `Pd=0.5`, `num_missed_bursts=1` (burst 0), `num_detected_regions=1` (burst 1 only) — low-energy burst correctly went undetected |
| **5** (extra false-alarm region) | 2 normal bursts, `--threshold-factor 1.4` (lowered from the 1.5 default to admit one spurious noise peak) | `num_detected_regions=3`, `num_matched_bursts=2`, **`num_false_alarm_regions=1`** (region `(5622,5768)`, no overlap with either true burst) |

All 5 cases run through `run_full_experiment.py` (the actual main
pipeline, not a direct `merge_close_regions()` call in isolation) and
**reproduced bit-for-bit** (`bursts_summary.csv`/`regions_summary.csv`/
`summary.csv` byte-identical) in a second independent process for every
case.

### 15.4 RadioML boundary small sweep (28 real-AWN combinations, not the full 500-combo grid)

**Scope note, stated explicitly rather than silently substituted**: the
requested full grid (2 mods × 2 SNRs × 5 samples × 5 threshold-factors × 5
sensing-window-sizes = 500 combinations) was replaced with a one-factor-
at-a-time design — measured at ~3.7s per real-AWN run (including a ~5–6s
dataset-pickle reload every single call, since `load_radioml_dict` has no
caching), the full grid would take on the order of 30+ minutes for a round
explicitly framed as "not an AMC accuracy evaluation" with 5 narrow,
specific functional goals. The reduced design still varies every requested
axis independently and directly targets all 5 stated goals:

- **Group 1** (modulation truthfulness + reproducibility): `{QPSK,BPSK} ×
  {SNR 0,18} × sample_index{0..4}` = 20 combos at
  `threshold-factor=1.5, sensing-window-size=128` (this repo's established
  baseline).
- **Group 2/2b** (threshold-factor sensitivity): `{0.8,1.0,1.2,2.0}` (plus
  the `1.5` baseline from Group 1) at a fixed sample, run against BOTH
  `QPSK,SNR18,idx0` (well-detected) and `BPSK,SNR18,idx0` (the
  partial-capture case) to directly show the metric's sensitivity.
- **Group 3/3b** (sensing-window-size sensitivity): same structure,
  `{16,32,64,256}` (plus the `128` baseline).

**Results against the 5 stated goals**:
1. **Detection metrics correctly produced**: all 28 primary combinations
   succeeded; `detection_success`/`captured_signal_ratio`/boundary errors/
   etc. populated on every row.
2. **`captured_signal_ratio` DOES change with parameters** — not visible on
   the well-detected `QPSK,SNR18,idx0` sample (stayed `1.0` across every
   threshold-factor 0.8–2.0 and every sensing-window-size 16–256, since its
   detected region always fully contains the true burst regardless), but
   clearly visible on the partial-capture `BPSK,SNR18,idx0` sample:
   `threshold-factor` 0.8→1.0→1.0→1.2→**0.6641**→2.0→**0.6172** (ratio drops
   as the threshold tightens and the detected region shrinks toward the
   burst's higher-energy portion only).
3. **`BPSK,SNR18,idx0`'s `captured_signal_ratio=0.625` REPRODUCED exactly**
   at the original baseline params (`threshold-factor=1.5,
   sensing-window-size=128`) — confirmed deterministic, not a one-off, and
   re-confirmed byte-identical (except one item below) in a second
   independent process.
4. **No silent no-op / fallback / NaN / Inf** across all 28+ combinations
   (`awn_backend` was the real model every time; `clean_has_nan`/
   `clean_has_inf` were `False` throughout) — **one legitimate, LOUD (not
   silent) `RuntimeError`** was found: `BPSK,SNR18,idx0` at
   `sensing-window-size ∈ {16,32,64}` produces a detected region too short
   to survive the default `--min-region-len` (defaults to `--window-size`
   = 128) filter for this specific sample's energy profile — an expected
   consequence of `filter_by_min_length` running after a narrower smoothing
   window on a sample whose energy isn't uniformly spread across all 128
   samples, not a bug; the error message is specific and was not silently
   swallowed.
5. **All output fields correct**: `summary.csv`'s ground-truth columns
   (`true_burst_start`/`end`, `detection_success`, `captured_signal_ratio`,
   boundary errors, etc.) all populated and internally consistent across
   every successful combination.

**Reproducibility caveat found**: re-running `BPSK,SNR18,idx0` (baseline
params) in a second independent process gave byte-identical results for
every decision-relevant field (positions, hashes, `captured_signal_ratio`,
`pred_clean`) **except** `logit_maxabs_clean_attacked`, which differed by
exactly `2^-12` (`0.0` vs `0.000244140625`) — consistent with ordinary
floating-point non-associativity in multi-threaded CPU BLAS/torch kernels
across process launches, not a bug in this repo's own code, and not
something that changed any prediction or ground-truth metric. Not
investigated further (out of this round's scope).

### 15.5 Complete RML2016.10a class mapping (`docs/radioml_class_mapping.csv`)

| dataset key | AWN class index | AWN display name | names match? |
|---|---|---|---|
| QAM16 | 0 | QAM16 | yes |
| QAM64 | 1 | QAM64 | yes |
| 8PSK | 2 | 8PSK | yes |
| WBFM | 3 | WBFM | yes |
| BPSK | 4 | BPSK | yes |
| CPFSK | 5 | CPFSK | yes |
| AM-DSB | 6 | AM-DSB | yes |
| GFSK | 7 | GFSK | yes |
| PAM4 | 8 | PAM4 | yes |
| QPSK | 9 | QPSK | yes |
| AM-SSB | 10 | AM-SSB | yes |

Cross-verified this round at **8 independent locations** in
`external/adversarial-rf` (submodule, pinned `ced705e`): the `classes`
dict (`data_loader/data_loader.py:13`, `util/config.py:52`) and a separate
`CLASS_NAMES` ordered list used purely for human-readable plot labels
(`plot_iq_reference_style.py:31`, `plot_comprehensive_attacks.py:33`,
`plot_iq_fgsm_grid.py:25`, `plot_all_attacks_iq_constellation.py:31`,
`plot_all_attacks_iq.py:31`, `plot_iq_constellation_attacks.py:36`) — all
8 agree exactly, no discrepancy between the dataset key string and the
"display name" was found anywhere.

### 15.6 Cross-reference to this round's required status labels

- **RadioML loader**: PASS (unchanged, section 14)
- **RadioML modulation truthfulness**: PASS, small-sample (unchanged claim scope from section 14.6, reinforced by section 15.4's 20-combination Group 1)
- **single-burst ground truth metrics**: PASS (unchanged, section 14.3)
- **multi-burst source**: **PASS this round** (section 15.1) — implemented, all 6 stated requirements verified, single-burst mode regression-confirmed unaffected
- **merge-gap function**: PASS (unchanged, section 13.2 — scratch-only dual-burst test)
- **merge-gap main pipeline**: **PASS this round** (section 15.3) — all 5 required scenarios (separate/merge/mixed/missed/false-alarm) reproduced through the real main pipeline, not an isolated function call, all reproducible
- **Pd/Pfa metrics**: **PASS this round** (section 15.2) — implemented, formulas documented with explicit denominators, verified against a hand-checked synthetic scenario covering all 5 required matching cases simultaneously
- **formal full batch**: **NOT STARTED** (unchanged)

## 16. Sensing-failure handling, batch aggregation CSVs, metrics-denominator confirmation (round 7)

New file: `src/utils/batch_aggregation.py`. Modified: `src/utils/pipeline.py`,
`src/sensing/ground_truth_metrics.py` (new
`derive_batch_aggregate_sensing_fields`), `experiments/run_batch.py`. **No
changes to `external/AWN`/`external/adversarial-rf`.** Formal full-parameter
batch: **still not started** (explicitly out of scope this round).

### 16.1 Sensing-failure handling — structured results, not batch-aborting exceptions

**Problem**: `run_dry_run_experiment()` previously let two specific,
EXPECTED sensing outcomes propagate as an uncaught `RuntimeError`, which
would abort an entire batch sweep on the first combo that hit either one:

1. `filter_by_min_length` (`src/sensing/energy_detection.py:67-84`) raises
   when zero regions survive — either none were ever detected (empty input)
   or all detected regions are shorter than `--min-region-len`. Two distinct
   messages for the two sub-cases (see source).
2. `segment_regions` (`src/sensing/segmentation.py:33`) raises when every
   surviving region is individually shorter than `--window-size`, so zero
   full-length AWN-input segments can be cut from any of them.

**Fix**: `src/utils/pipeline.py` now wraps EXACTLY these two call sites in
their own narrow `try/except RuntimeError` (nothing else in the function is
caught — see the comment at `pipeline.py` around the `filter_by_min_length`
call). On catch, `sensing_failure_stage`/`sensing_failure_reason` are
recorded, `regions` is set to `[]` if `filter_by_min_length` itself failed,
and the function returns a normal (non-raising) dict with
`run_status="sensing_failed"` instead of continuing into
AWN/attack/Top-K. Every other exception in the function — adapter
shape-mismatch `RuntimeError`s, dataset-load `ValueError`s, config
validation `ValueError`s — is **not** caught anywhere in `pipeline.py` and
still propagates normally, exactly as before this round.

**What is NOT done, per the explicit requirements**:
- `--threshold-factor` / `--min-region-len` are never silently altered to
  force a pass — both are used exactly as given; a failure is reported, not
  hidden.
- No fake/zero-padded segment is ever created — on a `segment_regions`
  failure, `n_segments=0` and no `summary.csv` is written at all (there is
  nothing to put in it: `summary.csv` is fundamentally one row per
  segment). `bursts_summary.csv`/`regions_summary.csv` ARE still written
  when ground truth exists, since a region can legitimately be detected and
  still fail to yield a full segment — that is real, useful information
  about the sensing outcome, not a fabricated success.
- The failure reason is printed to stdout (`[sensing] FAILED at
  stage=...: ...`) and also returned as `failure_reason` — nothing is
  swallowed silently, and `experiments/run_batch.py`'s per-combo error path
  (genuine exceptions only, see 16.2) separately still prints to stderr.

**Unified result schema**: both the success path and the
`sensing_failed` early-return path in `run_dry_run_experiment()` now return
the same key set for cross-run/batch consistency:
`run_status` (`"ok"` / `"sensing_failed"`), `sensing_success` (bool),
`failure_stage` (`None` / `"filter_by_min_length"` / `"segment_regions"`),
`failure_reason` (`None` or the original exception message),
`clean_amc_available` / `attack_available` / `defense_available` (bool —
all `False` together on a sensing failure, since AWN/attack/Top-K never run
when there are zero segments to feed them), plus the 9
`derive_batch_aggregate_sensing_fields` keys (16.3) merged into both paths
identically via `**sensing_agg`.

**Verified this round** (all via direct `run_dry_run_experiment()` calls,
real code paths, not mocked):
- Regression: synthetic mode, single-burst radioml mode (equivalence
  cross-checked against `ground_truth`, see 16.3), and multi-burst radioml
  mode (3 bursts) all still produce identical detected regions /
  `long_iq_sha256` / `captured_signal_ratio` behavior as before this
  round's edit — only the returned dict grew new keys, nothing existing
  changed.
- `filter_by_min_length` failure genuinely triggered end-to-end
  (`--threshold-factor 1000` on synthetic mode): `run_status=sensing_failed`,
  `failure_stage=filter_by_min_length`, `clean_amc_available=False`,
  `n_segments=0`, `summary_csv_path=None` — no raise.
- `segment_regions` failure genuinely triggered end-to-end
  (`--sensing-window-size 128 --window-size 2000 --min-region-len 0` on
  synthetic mode, so a ~627-sample region survives filtering but is too
  short for a 2000-sample window): `run_status=sensing_failed`,
  `failure_stage=segment_regions` — no raise.
- Same-seed cross-process reproducibility re-confirmed unaffected
  (`long_iq_sha256` and `regions` byte-identical across two independent
  processes, radioml single-burst QPSK/snr0/idx2, seed=7).

### 16.2 Batch aggregation CSVs

New shared helper `run_batch_combos()` (`src/utils/batch_aggregation.py`),
used by `experiments/run_batch.py`'s existing `(snr, mod, attack, topk)`
grid and reusable by any other combo-sweep script. Per combo it: (1) calls
`run_dry_run_experiment`; (2) on a genuine exception (`ValueError` /
`TypeError` / `RuntimeError` that still propagates per 16.1 — e.g. an
invalid `--dataset-mod`), prints the full exception with combo context to
**stderr** and records a `run_status="error"` row instead of aborting the
batch or silently skipping the combo (this is the only `try/except` in the
batch layer, and it is intentionally broad only because everything narrower
is already handled inside `run_dry_run_experiment` itself per 16.1); (3)
otherwise records a `run_status="ok"` or `run_status="sensing_failed"` row
from the returned dict.

**`batch_summary.csv`** — one row per combo, always, fixed columns:
`combo_id`, `output_dir`, `run_seed`, the caller-supplied combo parameter
columns (e.g. `snr_db`/`mod`/`attack`/`topk` for `run_batch.py`'s existing
grid), then `run_status`, `sensing_success`, `failure_stage`,
`failure_reason`, `clean_amc_available`, `attack_available`,
`defense_available`, `n_segments`, `sensing_window_size`, `segment_length`,
then the 9 aggregate sensing fields (16.3). An `error` row has every
sensing/AMC field `None`/empty except the identifying and failure columns.

**`batch_bursts_summary.csv`** — one row per (combo, TRUE burst), only for
combos with radioml ground truth (single- or multi-burst); an all-synthetic
batch produces zero rows and the file is not written at all (there is no
truth burst to report — logged explicitly, not a silent omission). Columns:
`combo_id`, `output_dir`, `run_seed`, the combo parameter columns, then
every `per_burst` field from `compute_multi_burst_sensing_metrics` (16.3):
`burst_id`, `true_start`, `true_end`, `true_burst_length`,
`detection_success`, `matched_region_id`, `matched_region_ids`,
`intersection_length`, `captured_signal_ratio`, `missed_sample_count`,
`start_boundary_error`, `end_boundary_error` (plus any extra per-burst
metadata keys the caller attached, e.g. multi-burst mode's `dataset_mod`/
`dataset_snr`/`sample_index`/`original_sample_sha256`).

**`batch_regions_summary.csv`** — one row per (combo, DETECTED region),
same ground-truth-only rule as above. Columns: `combo_id`, `output_dir`,
`run_seed`, the combo parameter columns, then every `per_region` field:
`region_id`, `detected_start`, `detected_end`, `detected_length`,
`matched_burst_ids`, `intersection_length`, `false_occupied_sample_count`,
`extra_captured_noise_ratio`.

**Known constraint** (documented in `batch_aggregation.py`'s docstring, not
a bug): all combos passed to one `run_batch_combos()` call must share the
same ground-truth mode (all-synthetic / all-single-burst-radioml /
all-multi-burst-radioml) and the same combo-parameter key set, since
`csv.DictWriter` requires one fixed column set per file (`src/utils/
csv_writer.py` derives fieldnames from the first row) and multi-burst
per-burst rows carry extra metadata keys a synthesized single-burst row
does not. Mixed-mode sweeps must be run as separate `run_batch_combos()`
calls (separate output subdirectories) — this matches how every batch in
this round and section 15 was actually run (one ground-truth mode per
sweep).

**Behavior change to note**: per-combo output subdirectories are now named
`combo0000`, `combo0001`, ... (zero-padded index) instead of the previous
`snr{snr}_mod{mod}_attack{attack}_topk{topk}` naming, since the shared
helper is generic over arbitrary combo-parameter dicts and cannot construct
a parameter-based directory name in general. The full parameter values for
any `combo_id` are always recoverable from that row in `batch_summary.csv`.

**Verified this round**, all via the real `experiments/run_batch.py` CLI:
- Synthetic 2-combo grid (`--snr-list 0,10`): `snr=0` combo genuinely hit
  `sensing_failed` (no occupied region at that SNR/threshold), `snr=10`
  combo succeeded — the batch did **not** abort, both rows appear in
  `batch_summary.csv` with the correct `run_status`/`failure_stage`, and
  (correctly) neither `batch_bursts_summary.csv` nor
  `batch_regions_summary.csv` was written (synthetic source, no ground
  truth to report).
- Radioml single-burst 2-combo grid: both `batch_bursts_summary.csv` and
  `batch_regions_summary.csv` written with 2 rows each, `combo_id`/
  `output_dir`/`run_seed`/combo-parameter columns present and correct on
  every row, values cross-checked equal to the equivalent
  `derive_batch_aggregate_sensing_fields` output for the same run.
  `false_occupied_sample_count=62`, `extra_captured_noise_ratio≈0.446` for
  the one detected region in this test — consistent with 16.3's formulas.
- Genuine-error 2-combo grid (`--dataset-mod NOTAMOD`, invalid): both
  combos printed the full `ValueError` to stderr with `combo_id` and
  parameters, both recorded as `run_status="error"` rows in
  `batch_summary.csv`, batch completed (did not abort), no
  `batch_bursts_summary.csv`/`batch_regions_summary.csv` written (0 ground
  truth rows, as expected since nothing ran far enough to detect anything).

### 16.3 Metrics-aggregation definitions — confirmed, fixed, and proven

All of the following were already implemented in
`compute_multi_burst_sensing_metrics` (section 15.2) prior to this round;
this subsection is this round's explicit written confirmation/proof per the
requirement to fix and document these definitions, plus the new
`derive_batch_aggregate_sensing_fields` normalizer that makes the
single-burst case use the exact same formulas (not a second implementation).

1. **`detection_probability` denominator is `num_truth_bursts`**
   (`num_matched_bursts / num_truth_bursts`) — confirmed, unchanged,
   `ground_truth_metrics.py:300`.

2. **`false_alarm_region_rate` denominator is `num_detected_regions`**
   (`num_false_alarm_regions / num_detected_regions`) — confirmed,
   unchanged, `ground_truth_metrics.py:301`.

3. **`sample_level_false_positive_rate` denominator — chosen definition and
   equivalence proof.** Two candidate denominators were proposed:
   (A) "all non-truth samples in the whole stream", and
   (B) `n_samples - total_truth_length`. **The code uses (B)**
   (`background_length = n_samples - total_true_length`,
   `ground_truth_metrics.py:284,302`). **Proof that (A) = (B)**: by
   construction, every true burst's interval `[true_start, true_end)` comes
   from `embed_sample_in_noise`/`embed_multiple_samples_in_noise`, which
   places bursts strictly back-to-back with gap ≥ 0 (cursor-based, no
   overlap possible — section 15.1) — so the true-burst intervals are
   pairwise disjoint, and "all non-truth samples in the stream" is exactly
   `n_samples` minus the sum of the (non-overlapping) true-burst lengths,
   i.e. exactly `n_samples - total_truth_length`. (A) and (B) are the same
   quantity by definition, not merely numerically coincident. The
   numerator, `total_false_occupied = sum(pr["false_occupied_sample_count"]
   for pr in per_region)`, is similarly safe to sum rather than union
   because detected regions are pairwise non-overlapping by construction
   (`merge_close_regions`/`filter_by_min_length`, section 15.2) — so this
   sum equals exactly the count of samples that are inside some detected
   region AND outside every true burst, stream-wide, with no double
   counting possible.

4. **`sample_level_false_negative_rate` denominator is total truth-burst
   sample count** (`total_missed / total_true_length`, where
   `total_true_length = sum(true_burst_lengths.values())`) — confirmed,
   unchanged, `ground_truth_metrics.py:281,303`.

5. **`mean_captured_signal_ratio` — includes missed bursts as 0, averaged
   over ALL truth bursts, not just matched ones.** Confirmed by reading
   `ground_truth_metrics.py:304-306`:
   `sum(pb["captured_signal_ratio"] for pb in per_burst) / num_truth_bursts`
   — the sum runs over every entry in `per_burst` (one per truth burst,
   always), and a missed burst's `captured_signal_ratio` is `0.0` by
   construction (`ground_truth_metrics.py:250`,
   `true_burst_length > 0` and `total_intersection == 0` when unmatched).
   The denominator is `num_truth_bursts`, not the matched-burst count. This
   is the FIXED definition: a batch with many undetected bursts will show a
   correspondingly lower mean ratio, not an inflated "average over survivors
   only" number.

6. **Mean boundary error is computed ONLY over matched bursts.** Confirmed:
   `matched_start_errors_abs = [abs(pb["start_boundary_error"]) for pb in
   per_burst if pb["detection_success"]]` (`ground_truth_metrics.py:291`,
   same for end/combined) — unmatched bursts have `start_boundary_error =
   None` (no region to measure against) and are excluded from the list
   comprehension entirely, not counted as some default error value. `None`
   if zero bursts matched (not `0.0` — "no error" and "no data" are kept
   distinct).

7. **No double-counting when one region matches multiple bursts.** Proof:
   `false_occupied_sample_count` is computed once **per region** (the outer
   loop in the `per_region` construction, `ground_truth_metrics.py:257-273`)
   as `detected_region_length - total_intersection`, where
   `total_intersection` sums that region's intersections with every burst
   it matches (`sum(inter for _, inter in matches)`, `matches` filtered to
   `jj == j` for this specific region `j` only). Each region contributes
   exactly one `false_occupied_sample_count` value regardless of how many
   bursts it matches (2+ matched bursts only affects `matched_burst_ids`'
   length and increases `total_intersection`, correctly shrinking
   `false_occupied_sample_count`, never appearing as a separate addend
   per matched burst). Aggregating `total_false_occupied =
   sum(pr["false_occupied_sample_count"] for pr in per_region)` therefore
   sums exactly `num_detected_regions` terms, one per region, never one per
   (region, burst) pair — no double counting is structurally possible.

8. **Single-burst vs multi-burst formula equivalence**
   (`derive_batch_aggregate_sensing_fields`,
   `src/sensing/ground_truth_metrics.py`): rather than maintaining a second,
   parallel set of formulas for the `--num-bursts 1` / non-multi-burst
   radioml case, this function routes it through
   `compute_multi_burst_sensing_metrics` with a synthesized single-entry
   `true_bursts` list (`burst_id=0`). Verified this round by direct
   comparison against the pre-existing `compute_sensing_ground_truth_metrics`
   output on the same real run (radioml BPSK/snr18/idx0, default flags):
   `mean_captured_signal_ratio == ground_truth["captured_signal_ratio"]`
   (`0.6015625` exactly) and `mean_absolute_start/end_boundary_error ==
   abs(ground_truth["start/end_boundary_error"])` (`51.0`/`62.0` exactly) —
   bit-for-bit equal, not just close. When there is no ground truth at all
   (synthetic source), all 9 fields are explicitly `None` (undefined, not
   `0` — "not measured" and "measured as zero" are kept distinct), except
   `num_detected_regions`, which is always knowable regardless of ground
   truth.

### 16.4 Small sensing validation matrix (real AWN, attack=none, fixed seed)

New file: `experiments/run_sensing_validation_matrix.py`. Design stated
before running (per the explicit requirement): OFAT anchored at one
baseline point (`mod=BPSK, snr=18, sample_index=0, threshold_factor=1.5,
sensing_window_size=128, min_region_len=0, merge_gap=0` — the same point
documented in section 14 to give `captured_signal_ratio=0.625`) plus one
small factorial for the one interaction explicitly requested
(`min_region_len x sensing_window_size`), run as two separate
`run_batch_combos()` calls (single-burst vs. multi-burst radioml, per
16.2's ground-truth-mode constraint):

| Group | Sweep | Count |
|---|---|---|
| 1 | `dataset_mod x dataset_snr x sample_index` OFAT at baseline sensing params | 2×2×5 = 20 |
| 2 | `threshold_factor` OFAT at baseline (mod,snr,idx) | 5 |
| 3 | `sensing_window_size` OFAT at baseline (mod,snr,idx) | 5 |
| 4 | `min_region_len x sensing_window_size` factorial at baseline (mod,snr,idx,threshold_factor) | 3×5 = 15 |
| 5 | `merge_gap` OFAT, 2-burst multi-burst mode, baseline sensing params (SEPARATE batch — multi-burst ground truth) | 3 |
| **Total** | | **48** |

Estimated runtime, stated before running: calibrated at ~1.4s/combo
in-process (steady-state, real AWN, checkpoint loaded fresh per combo —
timed directly: 5 sequential real-AWN combos took 2.07s/1.42s/1.40s/
1.34s/1.44s), so ≈1–2 minutes for 48 combos. **Actual measured runtime:
92.5s.**

**Results**: single-burst group: **33 ok, 12 sensing_failed, 0 error**;
multi-burst group: **3 ok, 0 sensing_failed, 0 error**. All 12
`sensing_failed` rows are Groups 3/4 combos with `sensing_window_size` in
`{16, 32, 64}` — i.e. the energy-detection smoothing window is narrower
than the 128-sample segment length, so the detected region is too short to
either survive `--min-region-len` (`failure_stage=filter_by_min_length`,
happens first when `min_region_len` is 64 or 128) or yield a full 128-
sample segment (`failure_stage=segment_regions`, happens when
`min_region_len` is 0 or otherwise doesn't filter it out first). Zero
genuine errors; the batch never aborted; every failed combo still has a
complete `batch_summary.csv` row with `failure_stage`/`failure_reason`
populated. No threshold or `--min-region-len` value was adjusted to avoid
or hide these failures — they are the intended, informative outcome of
Group 3/4's design.

**Goal-by-goal results** (per Part E's 7 stated purposes):

1. **Sensing failures recorded without aborting the batch**: confirmed —
   12/48 combos failed, batch completed all 48, the single-burst
   `batch_summary.csv` has all 45 rows (one per combo in that group) each
   with a populated `run_status`.
2. **All three batch-aggregate CSVs verified**: single-burst group —
   `batch_summary.csv` (45 rows, one per combo, uniform schema),
   `batch_bursts_summary.csv` (**45 rows** — exactly one truth-burst row
   per combo, including the 12 `sensing_failed` ones, since `ground_truth`
   is computed from whatever `regions` resulted even on a failure, per
   16.1), `batch_regions_summary.csv` (**370 rows** — one row per detected
   region per combo, highly non-uniform across combos: most combos
   contribute 1 region, but combo 21
   (`threshold_factor=1.0, sensing_window_size=128, min_region_len=0,
   merge_gap=0`) alone contributes **159** region rows, and combos 25/30
   (`threshold_factor=1.5, sensing_window_size=16`) contribute 72 each —
   see the unplanned finding below). All three files' `combo_id`/
   `output_dir`/`run_seed`/parameter columns were spot-checked correct.
   Multi-burst group produced all three CSVs (3/6/3 rows) since every
   multi-burst combo succeeded.
   **Unplanned finding**: `threshold_factor=1.0` (median-power threshold
   with no margin) and `sensing_window_size=16` (very fine smoothing) both
   independently cause severe mask fragmentation — energy_detect's mask
   crosses its threshold many times from noise alone, producing dozens to
   159 separate raw regions instead of one clean burst region, at
   `merge_gap=0`. This was not designed into the matrix; it fell out of
   the `threshold_factor`/`sensing_window_size` OFAT sweeps and is exactly
   the kind of real, unmanufactured sensing behavior this round's
   infrastructure needed to be able to record without crashing — and it
   did (`batch_regions_summary.csv` wrote all 370 rows without error, and
   none of these fragmented-mask combos happened to also be one of the 12
   `sensing_failed` rows, since enough total occupied length still existed
   to produce at least one segment).
3. **Same-seed reproducibility verified**: the OFAT baseline point
   (`BPSK, snr18, idx0, tf1.5, sws128, mrl0, mg0`) independently appears at
   combo_id 15 (Group 1), 23 (Group 2), 28 (Group 3), and 33 (Group 4) —
   4 separately-executed runs, same seed, same parameters — all 4 produced
   bit-for-bit identical `mean_captured_signal_ratio=0.625`,
   `mean_absolute_start_boundary_error=48.0`,
   `mean_absolute_end_boundary_error=64.0`, `detection_probability=1.0`.
4. **BPSK/SNR18/sample0 sensitivity verified**: the same 4 combos above
   reproduce `captured_signal_ratio=0.625` exactly (matches
   `--threshold-factor 1.5 --sensing-window-size 128 --min-region-len 0`,
   the same parameter point documented in section 14/15), confirming this
   is a genuine, stable, sample-dependent partial-capture case, not run-to-
   run noise.
5. **`min_region_len x sensing_window_size` interaction verified**:
   Group 4's 15-combo factorial shows a clean interaction — at
   `sensing_window_size ∈ {128, 256}` every `min_region_len` value
   succeeds (one clean region each); at `sensing_window_size=16` the mask
   fragments into 72 tiny raw regions (same fragmentation as the
   unplanned finding above) and at `sensing_window_size=32` into 9 — every
   `min_region_len` value fails for both (`filter_by_min_length` when
   `min_region_len>0` drops all of them for being individually too short;
   `segment_regions` when `min_region_len=0` lets them all through but
   none is individually ≥128 samples so no segment can be cut); at
   `sensing_window_size=64` exactly 1 region is detected (no fragmentation
   at this window size for this sample) but it is still <128 samples, so
   `min_region_len ∈ {0, 64}` fail at `segment_regions` while
   `min_region_len=128` fails earlier at `filter_by_min_length`.
   **Conclusion, stated explicitly and not hidden**: `--sensing-window-size`
   must be `>=` the segment length (`--window-size`, 128 in this matrix)
   for reliable single-region, single-segment detection regardless of
   `--min-region-len` — a narrower smoothing window both under-sizes and
   (at very small values) fragments the detected region(s) in this
   dataset/embedding configuration.
6. **`merge_gap`'s effect in the multi-burst main pipeline**: **no
   difference observed** across `merge_gap ∈ {0, 16, 64}` — all 3 combos
   produced the identical single merged region `(94, 407)` and identical
   aggregate metrics. This is an honest negative result, not a forced or
   hidden one: at `sensing_window_size=128` (this matrix's baseline), the
   energy-detection smoothing window alone already widens each burst's
   raw detected region enough to close the 2 bursts' `true` 50-sample gap
   before `merge_close_regions` ever runs, so `--merge-gap` has nothing
   left to do at any of the 3 tested values. This does **not** contradict
   section 15.3's earlier finding that `--merge-gap` clearly matters — that
   round used a much narrower `sensing-window-size=16`, which produces a
   detected gap close to the true gap instead of closing it via smoothing.
   `--merge-gap`'s effect is real but conditional on the smoothing-window
   size relative to the true burst gap; this round's 3-combo check at
   `sensing_window_size=128` was not designed to reproduce that regime and
   correctly shows no additional effect there.
7. **AWN prediction correctness NOT used as a pass condition**: confirmed
   — every pass/fail judgment above is based solely on `run_status`/
   `sensing_success`/the sensing metrics; `clean_amc_available`/
   `attack_available`/`defense_available` and the AWN logits are recorded
   but never used to gate success.

Output directories: `results/sensing_validation_matrix/single_burst/`
(`batch_summary.csv`, `batch_bursts_summary.csv`,
`batch_regions_summary.csv`, plus one `comboNNNN/` subdirectory per combo)
and `results/sensing_validation_matrix/multi_burst_merge_gap/` (same three
CSVs, 3 `comboNNNN/` subdirectories).

### 16.5 Cross-reference to this round's required status labels

- **Sensing-failure structured handling**: **PASS this round** (16.1) —
  implemented, both expected `RuntimeError` sites converted, genuine errors
  confirmed still raising, verified end-to-end for both failure stages
  through the real pipeline, batch tested (16.4: 12/48 real failures
  handled without aborting)
- **Batch aggregate CSVs (`batch_summary`/`batch_bursts_summary`/
  `batch_regions_summary`)**: **PASS this round** (16.2) — implemented,
  function tested (synthetic/single-burst/multi-burst/error smoke tests),
  batch tested (16.4, 48 real combos, real AWN)
- **Metrics-denominator definitions**: **PASS this round** (16.3) —
  confirmed, fixed, and proven (including the FPR-denominator equivalence
  proof and the no-double-counting proof requested explicitly), documented
  here
- **Small sensing validation matrix**: **PASS this round** (16.4) — batch
  tested, 48/48 combos completed (0 genuine errors), all 7 stated goals
  addressed with real results (including one honest negative result, goal
  6)
- **Formal full SNR × modulation × attack × eps × topk batch**: **NOT
  STARTED** (unchanged, explicitly out of scope this round)


## 17. Cross-modulation x SNR smoke matrix (round 8)

New file: `experiments/run_modulation_snr_matrix.py`. No changes to
`src/` (this round is a pure batch-aggregation consumer, no code changes
were needed). **No changes to `external/AWN`/`external/adversarial-rf`.**
Formal full-parameter batch: **still not started**.

### 17.1 Design, stated before running

All 11 RML2016.10a modulations × 4 SNRs (`-10, 0, 10, 18`) × 3
`sample_index` (`0, 1, 2`) = **132 combos**, single-burst RadioML mode
(one `run_batch_combos()` call, `results/modulation_snr_matrix/`,
`combo0000`..`combo0131`), one Python process running all 132 in-process
(not 132 subprocess launches). Fixed for every combo: `attack=none,
topk=10, threshold_factor=1.5, sensing_window_size=128, min_region_len=128,
merge_gap=0`, `seed=42`, real AWN (`--use-real-awn`), `device=cpu` — no
sensing/attack/Top-K parameter was swept or adjusted per combo. A 4-combo
timing probe measured ~1.3–1.7s/combo; estimated **~3–5 minutes** total
before running. **Actual measured runtime: 182.8s (~3.0 minutes).**

### 17.2 Results

**132/132 ok, 0 sensing_failed, 0 programming errors.** All three batch
CSVs written with exactly 132 rows each (`batch_summary.csv`,
`batch_bursts_summary.csv`, `batch_regions_summary.csv`) — every combo
detected exactly one clean region matching its one truth burst
(`mean_num_detected_regions=1.0` in every modulation and every SNR group;
no fragmentation this round, consistent with section 16.4's finding that
fragmentation only appears at extreme `threshold_factor`/
`sensing_window_size` values, not at this round's fixed
`threshold_factor=1.5, sensing_window_size=128` operating point).

**Pass-condition checks (Part C, all 10 verified programmatically against
the real output, not assumed)**:

| # | Check | Result |
|---|---|---|
| 1 | RadioML sample correctly loaded | 132/132 distinct `original_sample_sha256`, matching the 132 unique (mod,snr,idx) triples |
| 2 | modulation/SNR metadata correct | 0/132 mismatches between requested and recorded `dataset_mod`/`dataset_snr`/`sample_index` |
| 3 | sensing structured success/failure | 132/132 `run_status="ok"`, 0 `sensing_failed`, 0 `error` |
| 4 | `x_clean` shape `[N,2,128]` | confirmed via `segment_length=128` on every row + 0 segment/shape assertion failures (would have raised, per `pipeline.py`'s internal assert) |
| 5 | real AWN executed, no fallback | `awn_backend` column has exactly ONE value across all 132×1 segment rows: `external/adversarial-rf/models/model.py:AWN` — zero `dummy_awn_inference` rows |
| 6 | `summary.csv` complete | 132/132 present, row count == `n_segments` for every combo |
| 7 | `batch_summary.csv` has 1 row/combo | 132 rows, confirmed |
| 8 | burst/region CSV merged correctly | 132 burst rows + 132 region rows, `combo_id`/`output_dir`/`run_seed`/`dataset_mod`/`dataset_snr`/`sample_index` present and correct on every row (spot-checked) |
| 9 | no NaN/Inf | 0/132×1 rows have `clean_has_nan`/`clean_has_inf`/`attacked_has_nan`/`attacked_has_inf` = True |
| 10 | genuine program errors | 0 |

### 17.3 Per-modulation statistics (12 combos each: 4 SNRs × 3 sample_indices)

| Modulation | Pd | mean captured ratio | FA region rate | sample FPR | sample FNR | mean |start_err| | mean |end_err| | mean regions | pred_clean dist |
|---|---|---|---|---|---|---|---|---|---|
| 8PSK | 1.0 | 1.0 | 0.0 | 0.0147 | 0.0 | 59.75 | 58.67 | 1.0 | QAM64:11, PAM4:1 |
| AM-DSB | 1.0 | 1.0 | 0.0 | 0.0149 | 0.0 | 61.08 | 58.83 | 1.0 | QAM64:12 |
| AM-SSB | 1.0 | 1.0 | 0.0 | 0.0147 | 0.0 | 59.92 | 58.50 | 1.0 | QAM64:12 |
| BPSK | 1.0 | 0.96875 | 0.0 | 0.0142 | 0.03125 | 59.25 | 59.58 | 1.0 | QAM64:12 |
| CPFSK | 1.0 | 1.0 | 0.0 | 0.0148 | 0.0 | 59.83 | 59.42 | 1.0 | QAM64:11, QAM16:1 |
| GFSK | 1.0 | 1.0 | 0.0 | 0.0148 | 0.0 | 60.00 | 59.25 | 1.0 | QAM64:12 |
| PAM4 | 1.0 | 1.0 | 0.0 | 0.0143 | 0.0 | 55.42 | 59.50 | 1.0 | QAM64:12 |
| QAM16 | 1.0 | 1.0 | 0.0 | 0.0145 | 0.0 | 55.50 | 61.25 | 1.0 | QAM64:12 |
| QAM64 | 1.0 | 0.98112 | 0.0 | 0.0141 | 0.01888 | 56.25 | 60.17 | 1.0 | QAM64:12 (all correct) |
| QPSK | 1.0 | 0.98828 | 0.0 | 0.0143 | 0.01172 | 56.08 | 60.42 | 1.0 | QAM64:10, PAM4:2 |
| WBFM | 1.0 | 1.0 | 0.0 | 0.0149 | 0.0 | 60.75 | 59.67 | 1.0 | QAM64:11, PAM4:1 |

Every modulation achieves `detection_probability=1.0` and `false_alarm_region_rate=0.0`
at this parameter point. `mean_captured_signal_ratio` is a perfect 1.0 for 8 of
11 modulations; BPSK (0.96875), QAM64 (0.98112), QPSK (0.98828) are
fractionally below 1.0, driven by a small number of individual
sample-dependent partial-capture combos (17.5).

### 17.4 Per-SNR statistics (33 combos each: 11 modulations × 3 sample_indices)

| SNR | Pd | mean captured ratio | FA region rate | sample FPR | sample FNR | mean |start_err| | mean |end_err| | mean regions |
|---|---|---|---|---|---|---|---|---|
| -10 | 1.0 | 1.0 | 0.0 | 0.01477 | 0.0 | 60.09 | 59.00 | 1.0 |
| 0 | 1.0 | 1.0 | 0.0 | 0.01482 | 0.0 | 60.24 | 59.24 | 1.0 |
| 10 | 1.0 | 0.98887 | 0.0 | 0.01411 | 0.01113 | 55.24 | 59.97 | 1.0 |
| 18 | 1.0 | 0.98864 | 0.0 | 0.01453 | 0.01136 | 58.55 | 60.06 | 1.0 |

### 17.5 Sensing failure list

**Empty — 0 sensing failures across all 132 combos**, including every
`SNR=-10` combo. This is a real, honest, unmanipulated result — no
threshold/min-region-len/sensing-window-size value was changed to produce
it. It follows directly from how `embed_snr_margin` is defined
(`src/sensing/radioml_source.py:embed_sample_in_noise`): the synthetic
capture-noise floor is scaled relative to the loaded RadioML sample's OWN
measured power (`embed_noise_power = burst_power / embed_snr_margin`), and
that measured power already includes whatever SNR-dependent
signal-vs-RF-noise degradation the RML2016.10a dataset itself baked into
the sample at generation time — a `dataset_snr=-10` sample is noisier
*internally*, but its total (signal+noise) power, which is what
`embed_snr_margin` scales against, is not systematically smaller than a
`dataset_snr=18` sample's. So `embed_snr_margin=20.0` keeps every embedded
burst ~20x above the surrounding synthetic noise floor regardless of the
dataset SNR label, and detection at `threshold_factor=1.5,
sensing_window_size=128, min_region_len=128` succeeds uniformly. This is
not a claim that "SNR doesn't matter" for AMC (a low-`dataset_snr` sample's
*content* is still genuinely noisier, as evidenced by the SNR=10/18 groups'
slightly-below-1.0 mean captured ratio and non-zero FNR, likely from
segmentation/boundary-fitting sensitivity, not embedding failure) — only
that this repo's specific relative-margin embedding scheme decouples
*detectability* from `dataset_snr`, by design (documented already in
section 14.1 as "roughly mod/snr-independent burst power", reconfirmed
here at full 11×4 coverage rather than a handful of samples).

### 17.6 Unplanned/notable findings

- **3 sample-dependent partial-capture combos found** (all at this
  matrix's fixed sensing parameters): `BPSK/snr18/idx0`
  (`captured_signal_ratio=0.625`, `start_err=48, end_err=64` — the exact
  same value documented since section 14.4/15.4/16.4, reproduced again
  here as an incidental byproduct of the full sweep, not a dedicated
  repeat), `QAM64/snr10/idx0` (`0.7734375`), and `QPSK/snr10/idx0`
  (`0.859375`) — all 3 are at `sample_index=0`, though with only 3
  data points this is not asserted as a pattern, just reported as
  observed.
- **AWN clean-prediction distribution is heavily skewed to one class**:
  across all 132 single-segment predictions, `pred_clean` was QAM64
  (class 1) in **127/132** cases, PAM4 (class 8) in 4, QAM16 (class 0) in
  1 — regardless of the sample's true modulation. The only 12 cases where
  `pred_clean` matched the true `dataset_mod` label were exactly the 12
  QAM64 combos themselves (12/12 correct). Recorded per Part E.7's
  instruction; **not used as a pass/fail condition this round**, and no
  cause is diagnosed here (would require inspecting AWN
  preprocessing/normalization alignment against its original training
  pipeline, out of scope for this round's batch-aggregation focus).
- **No AM/FM-vs-digital fragmentation difference observed**: every one of
  the 11 modulations produced `mean_num_detected_regions=1.0` (never
  fragmented) at this round's fixed sensing parameters — energy detection
  operates on total IQ magnitude, which is not modulation-scheme-dependent
  in a way that would fragment one modulation class differently from
  another at a fixed threshold/window; section 16.4's fragmentation
  finding was purely a function of `threshold_factor`/
  `sensing_window_size`, not modulation type, and this round's uniform
  1-region-per-combo result is consistent with (not contradicting) that.

### 17.7 Real AWN backend verification

`awn_backend` column takes exactly one distinct value across all 132
combos' summary.csv rows: `external/adversarial-rf/models/model.py:AWN`.
Zero rows show `dummy_awn_inference` (the fallback used when
`--use-real-awn` is omitted or torch/checkpoint loading fails). `awn_notes`
on every row: `"Loaded real AWN from external/adversarial-rf/models/
model.py:AWN with checkpoint 'external/adversarial-rf/2016.10a_AWN.pkl'"`.

### 17.8 Reproducibility

Combo 114 (`QPSK, snr=10, sample_index=0`, the `captured_signal_ratio=
0.859375` case) was independently re-run via a separate
`run_full_experiment.py` CLI invocation (same seed=42, same sensing
params) and diffed byte-for-byte against the matrix's own
`combo0114/summary.csv`. Result: **identical on every field** (
`original_sample_sha256`, `long_iq_sha256`, detected region, boundary
errors, `captured_signal_ratio=0.859375`, `pred_clean=1`, all shapes)
except the `mod` column (`QPSK` vs the CLI default `BPSK`) — which is the
documented cosmetic synthetic-generator field, unused in radioml mode
(section 14's original design), not a real reproducibility discrepancy.

### 17.9 Cross-reference to this round's required status labels

- **Cross-modulation × SNR smoke matrix**: **PASS this round** (17.1–17.8)
  — batch tested, 132/132 combos completed (0 genuine errors, 0 sensing
  failures), all 10 pass conditions verified programmatically, all 7
  special checks (17.5/17.6/17.8) addressed with real results including
  one fully-negative sensing-failure list (SNR=-10 succeeded uniformly,
  explained not manufactured) and 3 unplanned partial-capture findings
- **AMC accuracy across modulations**: explicitly **NOT evaluated** this
  round (not a pass condition) — `pred_clean` distribution recorded only
  (17.6)
- **Formal full SNR × modulation × attack × eps × topk batch**: **NOT
  STARTED** (unchanged, explicitly out of scope this round)

## 18. AWN input-scale + segment-alignment root-cause diagnosis and fix (round 9)

Modified: `src/sensing/segmentation.py` (new `select_aligned_segments`,
`segment_regions` unchanged), `src/utils/config.py` (new
`alignment_policy`/`segment_hop` fields + CLI + validation),
`src/utils/pipeline.py` (wired to the new function; new summary.csv
columns), `src/utils/batch_aggregation.py` (new aggregate fields),
`experiments/run_batch.py` (new CLI flags). **No changes to
`external/AWN`/`external/adversarial-rf`.**

### 18.1 Normalization-scale mismatch (diagnosis only, NOT fixed this round)

Triggered by re-reading `external/adversarial-rf`'s own `CLAUDE.md`
(`data_loader.py:Load_Dataset` feeds raw, un-normalized RML samples --
amplitude ~±0.02 -- directly to the model at training time) against this
repo's `normalize_segments` (per-segment **unit-average-power**
normalization, ~50-100x rescale), applied unconditionally before every AWN
call in every prior round. `docs/integration_plan.md` section 5 flagged
this exact risk from the very first planning pass ("Needs a real run to
check classification accuracy... before trusting any prediction") but it
was never empirically checked until this round.

**Empirical confirmation** (7 real samples, BPSK/QPSK/8PSK/QAM16/QAM64/
WBFM/AM-DSB, snr=18, sample_index=0, real AWN): feeding the SAME clean,
unembedded sample raw (no normalization) vs. through `normalize_segments`
gives:

| Path | Accuracy | Logit magnitude | Confidence |
|---|---|---|---|
| raw (no normalize) | 4/7 correct | tens (e.g. [-40,12]) | not saturated |
| `normalize_segments` applied | 2/7 correct | **thousands** (e.g. [-9269,887]) | pinned at ~1.0000 (softmax saturation) |

This is almost certainly what explains section 17.6's "AWN predicted
QAM64 in 127/132 combos" finding from the prior round's smoke matrix --
not a spectrum-sensing degradation effect, but this pre-existing scale
mismatch, present in every real-AWN + radioml-mode run across this entire
session (including commit `82df790`). **This round does NOT fix
normalization** -- it is a separate, already-diagnosed issue explicitly
deferred by the user pending a decision; every result below (18.2-18.5)
deliberately holds normalization treatment CONSTANT (either always-off, to
isolate alignment, or noted explicitly when the full committed pipeline's
default behavior -- which still applies `normalize_segments` unconditionally
after segment selection, regardless of `--alignment-policy` -- is discussed).

### 18.2 Segment-alignment root-cause diagnosis (before any code change)

With normalization held OFF for a clean isolation, a 4-hypothesis decision
tree was run against the same 7 samples, embedded into a synthetic noise
stream (`embed_snr_margin=20`) and run through real sensing
(`threshold_factor=1.5, sensing_window_size=128, min_region_len=128`):

| Path | Definition | vs. baseline (A) |
|---|---|---|
| A `direct_raw` | Original [2,128] sample, no embedding, no normalization | reference: 4/7 correct |
| B `oracle_embedded_slice` | `iq[true_start:true_end]` sliced directly from the embedded stream, no sensing, no normalization | **6/7 match A** (only QAM64 differs, and both A's and B's QAM64 predictions were already low-confidence/borderline, ~0.16-0.17 out of 11 classes -- not a confident flip) |
| C `detected_oracle_aligned` | Sensing IS run (region must be detected), but the segment is still sliced at `true_start`, not the detected region's start | **7/7 match B exactly** (predictions, confidence, logits all identical) |
| D `detected_region_naive_segment` | The PRE-EXISTING behavior: `segment_regions()` cuts its first window starting at the DETECTED REGION's own start | **5/7 mismatch vs. C** |
| E `detected_region_best_overlap_segment` | Diagnostic-only oracle: exhaustive sliding search within the detected region for the window with maximum overlap with the true burst | **6/7 match A** (same single QAM64 exception as B/C) |

**Conclusion, in order**: (1) embedding does not alter waveform/scale in
any way that matters (A≈B) -- ruled out; (2) running sensing itself does
not further modify the underlying IQ values (B≡C exactly) -- ruled out;
(3) **the primary degradation source is segmentation-grid alignment**
(C≠D in 5/7 cases) -- confirmed; (4) an alignment-aware crop recovers
accuracy to the oracle level (E≈A) -- confirms the fix direction.

**Why D breaks, quantified**: for 6/7 samples the detected region's
leading edge starts **53-61 samples before** the true burst
(`start_boundary_error ≈ -53` to `-61`, from `sensing_window_size=128`
smoothing widening the region outward) even though the region covers the
burst 100% at the REGION level. `segment_regions()`'s first fixed-grid
window, starting exactly at the region's start, therefore straddles the
boundary and only overlaps the true burst by **52-63%**
(segment-level ratio, NOT the existing region-level `captured_signal_ratio`,
which correctly reported 1.0 for these same cases -- the two metrics
measure genuinely different things, hence 18.3's new, separately-named
segment-level fields).

**`embed_snr_margin` scale-factor check** (part of hypothesis 1): confirmed
by code trace (not measurement) that `embed_sample_in_noise` adds the
burst unscaled (`iq[start:end] += burst_iq`, no multiplier) -- scale factor
is exactly 1.0. `embed_snr_margin` only sets the noise floor; the relative
L2 difference between the embedded slice and the original burst was 0.216
for all 7 samples, matching the design target `1/√20 ≈ 0.224` almost
exactly.

### 18.3 `select_aligned_segments` implementation

New function in `src/sensing/segmentation.py`, alongside (not replacing)
the original `segment_regions()`. Two policies, selected via new
`--alignment-policy {naive,max-energy}` (default `naive`) and
`--segment-hop N` (default 1, positive int) CLI/config flags:

- **`naive`** (default, zero behavior change): calls `segment_regions()`
  directly for the segment data -- byte-identical to every prior round.
  Can produce multiple non-overlapping segments per region if the region
  is long enough.
- **`max-energy`**: exactly ONE selected segment PER detected region --
  the `seg_len`-sample window (among all `hop`-spaced sliding candidates
  within the region) with the highest mean power
  (`mean(|x|^2)`). Deliberately a MINIMAL scope (one window per region, not
  a general multi-window replacement for naive's long-region case), per
  this round's explicit "minimal verifiable" requirement. **Structurally**
  cannot depend on `true_burst_start`/`true_burst_end` -- the function
  signature never receives ground truth at all, only `iq`/`regions`/
  `seg_len`/`policy`/`hop`.

Both policies return `(segments, selection_meta)`, where `selection_meta`
is one dict per segment with `alignment_policy`, `segment_hop`,
`candidate_count`, `selected_segment_start/end`, `selected_window_power`,
`detected_region_start/end`, `region_idx` -- the last of which replaces a
fragile hand-rolled `segment_region_ids` computation in `pipeline.py` that
previously assumed `segment_regions()`'s own (region, n_windows) counting
and would have silently mis-attributed segments under `max-energy` (always
1 segment/region, not `n_windows`).

**New `summary.csv` columns** (every segment row, regardless of ground
truth): `alignment_policy`, `segment_hop`, `candidate_count`,
`selected_segment_start`, `selected_segment_end`, `selected_window_power`,
`detected_region_start`, `detected_region_end`. **Ground-truth-mode-only**
(ambiguous multi-burst matches -- 0 or 2+ matched bursts for a segment's
region -- return all-None, not a guess): `segment_start_offset_from_true`,
`segment_intersection_length`, `segment_captured_signal_ratio`,
`segment_noise_before_count`, `segment_noise_after_count`. The pre-existing
REGION-level `captured_signal_ratio` column is unchanged and unrenamed, per
the explicit requirement not to confuse the two.

**New `batch_summary.csv` fields** (via `batch_aggregation.py`):
`alignment_policy`, `segment_hop` (config knobs, uniform per run) and
`mean_segment_captured_signal_ratio` (mean of per-segment ratios over
segments with a resolvable true burst; `None` on sensing failure, on a
genuine error, or when no segment resolves one).

**Note on the still-open normalization issue**: `pipeline.py` still calls
`normalize_segments()` unconditionally after `select_aligned_segments`,
for BOTH policies -- `--alignment-policy` only changes WHICH samples are
selected, not whether they get rescaled afterward. This means a real
end-to-end run through the committed pipeline (e.g. `run_batch.py`) still
carries the 18.1 scale-mismatch confound on top of whichever alignment
policy is chosen; 18.4's comparison below deliberately bypasses
`normalize_segments` for all 5 paths to isolate the alignment effect
specifically, and separately confirms (18.4, footnote) what the confounded,
as-committed default actually produces.

### 18.4 Comparison test (7 modulations, snr=18, sample_index=0, seed=42, real AWN, normalization OFF for all 5 paths to isolate alignment)

Paths A/B/E computed via direct adapter calls (as in 18.2); **C and D now
call the real, just-implemented `select_aligned_segments()`** (not a
reimplementation) with `policy="naive"`/`"max-energy"` respectively.

| Modulation | A direct_raw | B oracle_embedded | C naive | D max-energy | E best-overlap oracle |
|---|---|---|---|---|---|
| BPSK | PAM4 ✗ | PAM4 ✗ | PAM4 ✗ (ratio 0.625) | PAM4 ✗ (ratio 0.586) | PAM4 ✗ (ratio 0.625) |
| QPSK | QPSK ✓ | QPSK ✓ | QPSK ✓ (ratio 0.563) | QPSK ✓ (ratio 0.992) | QPSK ✓ (ratio 1.0) |
| 8PSK | 8PSK ✓ | 8PSK ✓ | 8PSK ✓ (ratio 0.563) | 8PSK ✓ (ratio 1.0) | 8PSK ✓ (ratio 1.0) |
| QAM16 | QAM16 ✓ | QAM16 ✓ | 8PSK ✗ (ratio 0.578) | QAM16 ✓ (ratio 0.984) | QAM16 ✓ (ratio 1.0) |
| QAM64 | PAM4 ✗ | AM-SSB ✗ | 8PSK ✗ (ratio 0.586) | AM-SSB ✗ (ratio 1.0) | AM-SSB ✗ (ratio 1.0) |
| WBFM | AM-DSB ✗ | AM-DSB ✗ | BPSK ✗ (ratio 0.523) | AM-DSB ✗ (ratio 1.0) | AM-DSB ✗ (ratio 1.0) |
| AM-DSB | AM-DSB ✓ | AM-DSB ✓ | BPSK ✗ (ratio 0.523) | AM-DSB ✓ (ratio 1.0) | AM-DSB ✓ (ratio 1.0) |

**Accuracy**: A=4/7, B=4/7, **C(naive)=2/7**, **D(max-energy)=4/7**,
E(oracle)=4/7. **max-energy exactly matches the oracle-path accuracy.**

**Prediction agreement with `B` (oracle_embedded_slice)**: naive=3/7,
**max-energy=7/7** (perfect).

**Mean segment-level captured_signal_ratio**: naive=0.5658,
**max-energy=0.9375**.

**Note on max-energy's failure mode, checked explicitly per this round's
requirement** ("請特別檢查max-energy是否可能只選到局部高能量噪聲或burst的
局部峰值"): for BPSK specifically, max-energy's segment ratio (0.586) is
marginally LOWER than naive's (0.625) for this one sample -- the region
itself only 63% covers the true burst at the region level (a genuine
partial-detection case, independent of alignment), and within a region
that already excludes part of the burst, "highest mean power" is not
always exactly "maximum true-burst overlap" since real modulated signal
power is not perfectly uniform sample-to-sample. This is a real,
un-smoothed-over instability, reported as found -- not treated as a reason
to add a more complex selection heuristic this round, per the explicit
instruction to only report if found unstable, not engineer around it.
Every other one of the 7 samples showed max-energy meeting or exceeding
naive's ratio.

**Footnote -- what the actual committed default pipeline produces** (i.e.
WITH `normalize_segments`, confounding scale and alignment together,
matching what a real `run_batch.py`/`run_modulation_snr_matrix.py` run
would see today): naive=1/7, max-energy=2/7 correct in a supplementary run
against the same 7 samples. Both are far below the isolated-alignment
numbers above, confirming 18.1's scale mismatch remains the dominant
confound in any real end-to-end run using this repo's current default
normalization -- fixing alignment alone (this round) is necessary but,
until normalization is also addressed, not sufficient to restore accuracy
in a real run through the committed pipeline.

### 18.5 Pass-condition verification

1. **max-energy does not use `true_burst_start`**: confirmed structurally
   -- `select_aligned_segments()`'s signature never accepts ground truth.
2. **Same-seed cross-process bit-identical**: confirmed (`QAM16/snr18/idx0`,
   max-energy, two independent `python` processes) -- identical
   `long_iq_sha256` and `mean_segment_captured_signal_ratio=0.984375`.
3. **Selected segment length fixed at 128**: guaranteed by construction
   (`to_awn_input` asserts `seg_len`; every candidate window is a fixed
   128-sample slice).
4. **No NaN/Inf**: confirmed across all 35 (7 mods × 5 paths) inference
   confidence/logit values in the 18.4 comparison.
5. **max-energy's mean `segment_captured_signal_ratio` > naive's**:
   confirmed, 0.9375 > 0.5658.
6. **max-energy's prediction agreement with `oracle_embedded_slice` not
   lower than naive's**: confirmed, 7/7 vs. 3/7 (far higher, not merely
   not-lower).
7. **best-overlap oracle used only as a diagnostic ceiling**: confirmed --
   no `--alignment-policy` choice implements it; it exists only in the
   diagnostic comparison script, never in `src/`.
8. **`summary.csv`/`batch_summary.csv` schema correct**: confirmed --
   per-segment columns present and correct in every tested `summary.csv`;
   `batch_summary.csv` correctly carries `alignment_policy`/`segment_hop`/
   `mean_segment_captured_signal_ratio` via a 2-combo smoke batch.

`direct_raw`'s 4/7 is the reference (this checkpoint's actual behavior on
these 7 samples), NOT treated as a 7/7 pass target, per the explicit
instruction.

### 18.6 Cross-reference to this round's required status labels

- **Normalization-scale mismatch**: **diagnosed, NOT fixed** this round --
  root cause confirmed (thousands-magnitude saturated logits vs. tens for
  raw-scale input), deferred pending a separate design decision.
- **Segment-alignment root cause**: **diagnosed and fixed this round**
  (18.2/18.3) -- `select_aligned_segments` implemented, function tested
  (regression: naive byte-identical to prior behavior; multi-burst mode
  regression-confirmed with both ambiguous- and unambiguous-match cases),
  and validated via a real 7-modulation comparison (18.4) showing
  max-energy matches oracle-path accuracy when normalization is held
  constant.
- **`--alignment-policy`/`--segment-hop`**: **PASS this round** -- CLI
  wired into both `run_full_experiment.py` and `run_batch.py`, validated
  (`naive` default preserves every prior round's exact behavior;
  `max-energy` reproducible cross-process).
- **Combined real-pipeline accuracy (alignment fix + still-open
  normalization bug)**: explicitly **NOT resolved** -- the as-committed
  default pipeline's real predictions remain dominated by the normalization
  confound (18.4 footnote) until that separate issue is addressed.
- **Formal full SNR × modulation × attack × eps × topk batch**: **NOT
  STARTED** (unchanged, explicitly out of scope this round)

## 19. AWN input-scale fix: `--awn-preprocess {legacy-unit-power,radioml-native}` (round 10)

Modified: `src/sensing/normalize.py` (new `apply_awn_preprocess`,
`normalize_segments` unchanged), `src/utils/config.py` (new
`awn_preprocess` field + CLI + validation), `src/utils/pipeline.py` (wired
at the AWN input boundary only; new summary.csv columns),
`src/utils/batch_aggregation.py` (new aggregate fields),
`experiments/run_batch.py` (new CLI flag). **No changes to
`external/AWN`/`external/adversarial-rf`.** Default value **unchanged**
this round (`legacy-unit-power`) pending a separate decision, per explicit
instruction.

### 19.1 Traced evidence: what `external/adversarial-rf` actually feeds AWN

Read directly (not inferred from comments/docs), cross-checked between the
pinned submodule (`external/adversarial-rf`, `ced705e`) and the real venv
install used for this session's real-backend testing
(`/home/xiaomi/adversarial-rf`, `70036bc`) -- `diff` confirmed
`data_loader.py`'s core loading path, `util/training.py`,
`util/evaluation.py`, and `models/model.py` are identical between the two
commits (only a trivial unrelated `snr_min` filter parameter differs in
`data_loader.py`, no normalization-relevant change).

| Step | File:line | What happens |
|---|---|---|
| Pickle read | `data_loader/data_loader.py:38` | `Set = pickle.load(open(file_pointer,'rb'), encoding='bytes')` |
| dtype cast | `data_loader/data_loader.py:60-61` | `Signals = np.vstack(Signals); Signals = torch.from_numpy(Signals.astype(np.float32))` -- **no scaling, no clipping, no transpose** (already `[N,2,128]`) |
| Train-time forward | `util/training.py:93-97` | `sig_batch = sig_batch.to(device); ...; logit, regu_sum = self.model(sig_batch)` -- `sig_batch` comes straight from the `DataLoader` wrapping the untouched tensor above |
| Val-time forward | `util/training.py:139-144` | Identical -- `logit, regu_sum = self.model(sig_batch)` |
| Eval-time forward | `util/evaluation.py:36-38` | `sample = sample.to(cfg.device); logit, _ = model(sample)` -- `sample` is a `torch.chunk` of the same untouched `sig_test` tensor |
| Attack-domain round-trip | `util/adv_attack.py:31-84,108-128` | `Model01Wrapper.forward` converts torchattacks' `[0,1]` input back via `x_iq = x01 * b + a` (minmax mode) or `2*x01-1` (unit mode) before calling the real model -- both are **exact, lossless inverses** of their own forward mapping, so at `eps=0` (or when the attack step is skipped) the model receives the **exact original, untouched raw IQ value** regardless of which `--ta_box` convention is used |

**Conclusion**: at every point between the dataset pickle and
`AWN.forward()` -- training, validation, and evaluation -- **zero
normalization, scaling, or clipping is ever applied**. The model was
trained and is evaluated directly on raw RML2016.10a amplitude (~1e-2 to
1e-4 mean power per section 14.1's own prior measurement). This directly
confirms section 18.1's diagnosis: this repo's `normalize_segments`
(~50-120x rescale, measured exactly in 19.4 below) has no counterpart
anywhere in the actual AWN training/eval pipeline.

### 19.2 `--awn-preprocess` policy design

New `apply_awn_preprocess(segments, policy)` in `src/sensing/normalize.py`
-- the **only** place this repo rescales a segment before AWN inference.
Never called from `energy_detection.py` or `segmentation.py`; `pipeline.py`
calls it exactly once, immediately before `to_awn_input()`, strictly after
alignment/detection have already selected the segment.

- **`legacy-unit-power`** (default, unchanged): calls the existing
  `normalize_segments()` -- a per-segment **scalar** rescale (divide by
  `sqrt(mean(|x|^2))`), so it does NOT alter the relative amplitude
  structure WITHIN a segment (e.g. a louder half vs. a quieter half stays
  proportionally the same) -- it only moves the segment's absolute scale
  far outside AWN's trained distribution. This distinction matters for
  19.3 below.
- **`radioml-native`**: literally a no-op (dtype cast only) -- per 19.1's
  evidence, "no normalization" IS what the real pipeline does, so
  replicating it requires doing nothing.

### 19.3 Background-noise / `embed_snr_margin` handling (structural, not policy-specific)

Both policies operate on whatever segment `select_aligned_segments`
already selected -- a mix of RadioML burst and synthetic capture noise
when the segment only partially overlaps the true burst, or embedding
noise even at 100% overlap (section 18.2). `radioml-native` cannot corrupt
the burst/noise relative SNR because it performs no arithmetic on the
segment at all. `legacy-unit-power`, being a single scalar multiply
applied uniformly across every sample in the segment, ALSO preserves the
relative burst-vs-noise proportion exactly -- rescaling everything by the
same constant does not change which samples are "louder" relative to each
other. **What `legacy-unit-power` actually breaks is not the relative
burst/noise structure, but the absolute scale relative to AWN's training
distribution** -- an important distinction from what might be assumed;
this round's fix targets the absolute-scale problem specifically, not a
(non-existent) relative-SNR corruption. `embed_snr_margin`'s effect
(`noise_power = burst_power / embed_snr_margin`, section 15.1) is set
during embedding, long before `apply_awn_preprocess` ever runs, and
`radioml-native` leaves it completely intact (verified in 19.4: identical
`captured_signal_ratio`/detected regions between the two policies, and
`awn_input_power_before` under `radioml-native` matches the segment's
already-embedded power exactly, `scale_factor=1.0`).

### 19.4 Comparison test (7 modulations, snr=18, sample_index=0, seed=42, real AWN, `alignment-policy=max-energy segment-hop=1`)

Paths A/B/C computed directly (A/B as in section 18.2/18.4); **D and E run
through the real, unmodified `run_dry_run_experiment()`** (not a
reimplementation), differing only in `--awn-preprocess`.

| Modulation | A direct_raw | B oracle_embedded | C before-preprocess | D legacy-unit-power | E radioml-native |
|---|---|---|---|---|---|
| BPSK | PAM4 ✗ | PAM4 ✗ | PAM4 ✗ | QAM64 ✗ (scale×51.8) | PAM4 ✗ (scale×1.0) |
| QPSK | QPSK ✓ | QPSK ✓ | QPSK ✓ | QAM64 ✗ (scale×118.5) | QPSK ✓ (scale×1.0) |
| 8PSK | 8PSK ✓ | 8PSK ✓ | 8PSK ✓ | QAM64 ✗ (scale×119.9) | 8PSK ✓ (scale×1.0) |
| QAM16 | QAM16 ✓ | QAM16 ✓ | QAM16 ✓ | QAM64 ✗ (scale×117.9) | QAM16 ✓ (scale×1.0) |
| QAM64 | PAM4 ✗ | AM-SSB ✗ | AM-SSB ✗ | QAM64 ✓ (scale×110.2) | AM-SSB ✗ (scale×1.0) |
| WBFM | AM-DSB ✗ | AM-DSB ✗ | AM-DSB ✗ | WBFM ✓ (scale×122.7) | AM-DSB ✗ (scale×1.0) |
| AM-DSB | AM-DSB ✓ | AM-DSB ✓ | AM-DSB ✓ | WBFM ✗ (scale×122.7) | AM-DSB ✓ (scale×1.0) |

**Accuracy**: A=4/7, B=4/7, C=4/7, **D(legacy-unit-power)=2/7**,
**E(radioml-native)=4/7 -- exactly matches the oracle.**

**Prediction agreement with `B` (oracle_embedded_slice)**:
legacy-unit-power=**0/7**, radioml-native=**7/7** (perfect).

**Input/logit scale comparison**: `A_direct_raw` mean power
3.1e-5-1.8e-4 (logits in the tens); `D_legacy-unit-power`
`awn_input_power_after` is **exactly 1.0** for every sample (by
construction) with a measured scale factor of **110x-123x**;
`E_radioml-native` `awn_input_power_after` matches
`awn_input_power_before` exactly (`scale_factor=1.0`) and stays in the
same 3.7e-5-8.2e-5 order of magnitude as `direct_raw` -- consistent with,
not orders of magnitude different from, the reference.

**Verification, not just design**: `C`'s (before any preprocessing)
`input_sha256` was compared against the ACTUAL AWN input array produced by
the real pipeline under `--awn-preprocess radioml-native` (path E) --
**7/7 byte-identical**, empirically confirming `radioml-native` really is
the no-op it's documented as, not merely by code reading.

**No NaN/Inf** across all D/E runs (`awn_input_has_nan`/`has_inf` all
`False`).

**Sensing/alignment unaffected**: `regions`/`detected_region_start/end`/
`captured_signal_ratio` were identical between the `legacy-unit-power` and
`radioml-native` runs for every sample (confirmed via each run's
`summary.csv`) -- `--awn-preprocess` never touches detection or alignment,
exactly as required.

### 19.5 Attack-domain tracing (Part 六 -- tracked only, NOT redesigned this round)

`src/adapters/attack_adapter.py:AttackAdapter.apply()` receives `x` =
`x_clean` (i.e. **already** run through `apply_awn_preprocess`, whichever
policy was selected) and computes its own per-segment min-max affine
mapping FROM that same `x`:
`x_ta, a, b = _iq_to_ta_input_minmax(x_t)` (`attack_adapter.py:309`), where
`a = x.amin(...)`, `b = x.amax(...) - a`
(`external/adversarial-rf/util/adv_attack.py:114-115`). `attack_eps` is
therefore interpreted **relative to `x_clean`'s own min-max range at the
time `AttackAdapter.apply()` is called** -- not a fixed absolute IQ
quantity.

**Consequence, reported not fixed**: switching `--awn-preprocess` does
**not** change what a given `--attack-eps` means in *relative* terms (a
fraction of the sample's own range, either way) -- but it changes what
that fraction means in **absolute IQ units** by the same ~50-120x factor
measured in 19.4, since `legacy-unit-power`'s `x_clean` has a ~50-120x
larger min-max span than `radioml-native`'s. A `--attack-eps` value tuned
for one `--awn-preprocess` policy will therefore correspond to a very
different absolute perturbation magnitude under the other. This is flagged
for a future round's explicit decision (e.g. whether `attack_eps` should
be redefined once `radioml-native` is adopted) -- no attack code or
default was touched this round.

### 19.6 Pass-condition verification

1. **`radioml-native` has adversarial-rf source-code basis**: confirmed,
   19.1's evidence table (exact file/line, not inferred).
2. **Does not use label or `true_burst_start` for normalization**:
   confirmed structurally -- `apply_awn_preprocess(segments, policy)`'s
   signature never accepts ground truth or labels.
3. **Does not break max-energy alignment**: confirmed -- `select_aligned_segments`
   runs entirely before `apply_awn_preprocess` in `pipeline.py`; alignment
   metadata (`selected_segment_start/end`, `candidate_count`) was identical
   between the D and E runs in 19.4.
4. **Does not change sensing mask or detected regions**: confirmed --
   `regions`/`detected_region_start/end` identical between D and E for
   every sample (19.4).
5. **Same-seed cross-process bit-identical**: confirmed
   (`QAM16/snr18/idx0`, `radioml-native` + `max-energy`, two independent
   processes) -- identical `long_iq_sha256`, `awn_input_min`,
   `awn_input_max`.
6. **`radioml-native`'s power/logit magnitude close to `direct_raw`, not
   orders of magnitude off**: confirmed -- same 1e-4/1e-5 order of
   magnitude and tens-scale logits, vs. `legacy-unit-power`'s
   thousands-scale logits (section 18.1) and exactly-1.0 power.
7. **`radioml-native` vs. `oracle_embedded_slice` agreement not lower than
   `legacy-unit-power`'s**: confirmed, 7/7 vs. 0/7 (dramatically higher,
   not merely not-lower).
8. **Not judged on 7/7 accuracy**: confirmed -- `direct_raw`'s 4/7 is used
   as the reference throughout, never treated as a 7/7 target.
9. **New `summary.csv`/`batch_summary.csv` fields**: confirmed present and
   correct -- `awn_preprocess`, `awn_input_power_before`,
   `awn_input_power_after`, `awn_input_scale_factor`, `awn_input_min`,
   `awn_input_max`, `awn_input_has_nan`, `awn_input_has_inf` (per-segment
   in `summary.csv`; mean/global-min/global-max/any() aggregates in
   `batch_summary.csv`, verified via a 2-combo smoke batch).

### 19.7 Cross-reference to this round's required status labels

- **`external/adversarial-rf` preprocessing evidence chain**: **PASS this
  round** (19.1) -- traced to exact file/line in both the pinned submodule
  and the real venv install, confirmed identical for every relevant file.
- **`--awn-preprocess {legacy-unit-power,radioml-native}`**: **PASS this
  round** (19.2/19.3/19.6) -- implemented at the AWN input boundary only,
  function tested (regression: default behavior byte-identical to every
  prior round) and validated via a real 7-modulation comparison (19.4)
  showing `radioml-native` matches the oracle-path accuracy exactly (4/7)
  with 7/7 prediction agreement, vs. `legacy-unit-power`'s 2/7 and 0/7.
  Default value **deliberately left unchanged** this round.
- **Attack-domain (`attack_eps`) tracing**: **PASS this round, tracking
  only** (19.5) -- documented that `attack_eps` is relative-scale-invariant
  but absolute-magnitude-dependent on `--awn-preprocess`; explicitly not
  redesigned.
- **Combined real-pipeline accuracy (alignment fix + preprocessing fix,
  both applied)**: with `--alignment-policy max-energy --awn-preprocess
  radioml-native`, the real committed pipeline now reproduces the
  `direct_raw` oracle's accuracy exactly on these 7 samples (4/7) --
  **both previously-diagnosed degradation sources are now addressed**,
  though the default CLI values remain unchanged pending explicit adoption.
- **Formal full SNR × modulation × attack × eps × topk batch**: **NOT
  STARTED** (unchanged, explicitly out of scope this round)

## 20. Source-aware defaults, attack-scale verification, RadioML end-to-end smoke matrix (round 11)

Modified: `src/utils/config.py` (new `resolve_alignment_policy`/
`resolve_awn_preprocess`, dataclass fields now `Optional[str] = None`),
`src/utils/pipeline.py` (calls the resolvers; new `iq_source` column),
`src/utils/batch_aggregation.py` (new `iq_source` field). New file:
`experiments/run_e2e_smoke_matrix.py`. **No changes to
`external/AWN`/`external/adversarial-rf`.**

### 20.1 Source-aware defaults

`--alignment-policy`/`--awn-preprocess` CLI defaults changed from a fixed
string to `None` (both in `build_arg_parser` and `run_batch.py`'s own
parser); the `ExperimentConfig` dataclass fields changed identically. A
`None` is resolved **inside `run_dry_run_experiment`** (never at
config-construction time, same pattern as the pre-existing
`resolve_sensing_window_size`), via two new functions:

```
resolve_alignment_policy(iq_source, alignment_policy) -> "max-energy" if iq_source=="radioml" else "naive"
resolve_awn_preprocess(iq_source, awn_preprocess)     -> "radioml-native" if iq_source=="radioml" else "legacy-unit-power"
```

An explicitly passed value (CLI flag or direct `ExperimentConfig(...)`
construction) is **never** overridden — the resolvers only fill in a
`None`. Every functional use of these two settings in `pipeline.py` (the
`select_aligned_segments`/`apply_awn_preprocess` calls, every CSV column,
every result-dict key) was switched from the raw `cfg.alignment_policy`/
`cfg.awn_preprocess` to the resolved `effective_alignment_policy`/
`effective_awn_preprocess` local variables — confirmed via `grep`, zero
remaining functional references to the raw (possibly-`None`) cfg fields.
`validate_experiment_config` was updated to accept `None` for both (still
validates an explicitly-set value immediately). Resolution prints a
one-line `[config] --X unset; source-aware default for iq_source=...`
message, so the applied default is always visible in the run log, not a
silent fallback.

**Consequence**: `experiments/run_modulation_snr_matrix.py` and
`experiments/run_sensing_validation_matrix.py` (prior rounds, never
explicitly set either field) now automatically pick up `max-energy`/
`radioml-native` for their radioml-mode combos, with zero code changes to
those files — verified this is the correct, intended effect, not an
oversight, by re-confirming via `grep` that neither file passes these
fields.

**Regression verified**: synthetic mode with no flags still resolves to
`naive`/`legacy-unit-power` and reproduces the exact known-baseline region
`(3788, 4415)`; radioml mode with no flags now auto-resolves to
`max-energy`/`radioml-native` (previously required explicit flags);
radioml mode with explicit `--alignment-policy naive --awn-preprocess
legacy-unit-power` correctly overrides the source-aware default (confirmed
`selected_segment_start=3937`/`scale_factor=51.8` vs. the auto-resolved
run's `selected_segment_start=3942`/`scale_factor=1.0` for the identical
sample).

### 20.2 New CSV fields

`summary.csv` (per segment) gained `iq_source` (the raw `cfg.iq_source`
value that drove the resolution, distinct from the existing `source_type`
which further splits `"radioml"` into `"radioml"`/`"radioml_multi_burst"`).
`alignment_policy`, `awn_preprocess`, `selected_segment_start/end`,
`segment_captured_signal_ratio` (this round's requested
"segment_level_captured_ratio" -- kept under its existing, already-tested
name from section 18/19 rather than renamed, to avoid touching working
code for a cosmetic difference; explicitly cross-referenced here),
`awn_input_power_before/after` were already present from sections 18-19
and are unchanged. `batch_summary.csv` gained the same `iq_source` field
(via `batch_aggregation.py`'s `_RUN_META_FIELDS`); the error-row path was
updated to resolve `alignment_policy`/`awn_preprocess` (not report the raw
`None`) even for a combo that raised, for schema consistency with a
successful row.

### 20.3 Attack-scale verification (traced AND empirically tested)

`AttackAdapter.apply()` receives `x` = `x_clean`, already through
`apply_awn_preprocess`. It computes its own per-segment min-max mapping
**from that same `x`** (`external/adversarial-rf/util/adv_attack.py:
iq_to_ta_input_minmax`, `a = x.amin(...)`, `b = x.amax(...) - a`), applies
the requested attack in `[0,1]` space, then inverts back via
`x = a + b * x01_adv` — a construction that is inherently scale-relative,
so it works correctly at any `awn_preprocess` scale without modification.
**No incompatibility was found** between the min-max wrapper and the
`radioml-native` domain (Part 六's conditional "if incompatible, propose a
minimal fix" did not trigger — verified, not assumed).

- **`eps=0` no-op**: **not bit-identical** (max abs input diff
  `9.3e-9`, logits diff `9.5e-7`, under `radioml-native`) -- a real,
  reported finding, not glossed over. Traced to the float32 round-trip
  through the `[0,1]` min-max conversion and back (`x -> x01 -> x_adv`),
  present at the SAME relative magnitude (~1e-7, consistent with float32
  machine epsilon) under `legacy-unit-power` too (`3.4e-7` absolute at
  that policy's ~50x larger scale) -- confirming this is a pre-existing,
  scale-independent property of `AttackAdapter`'s round-trip conversion
  itself, not something this round's `awn_preprocess` change introduced or
  worsened. **Predictions and top-class logits are unaffected** in every
  tested case (`pred_clean == pred_attacked` at `eps=0` throughout).
- **FGSM/PGD perturbation magnitude vs. `--attack-eps`**: measured directly
  (not assumed) across `eps ∈ {0.01, 0.03, 0.1}`: actual L∞ perturbation
  in IQ units equals `eps × (sample's own min-max range)` with ratio
  **exactly 1.0000** in every case, for both FGSM and PGD, under
  `radioml-native`. No hidden 50-120x (or any other undocumented) rescale
  exists inside the adapter.
- **Eval mode restored**: `wrapped_model.training == False` confirmed
  after every real-attack call (single direct test plus all 72 smoke
  matrix combos' `attack_training_after` column, section 20.4 -- 0
  violations across 48 `attack != "none"` rows).
- **No NaN/Inf**: confirmed across all direct tests and the full 72-combo
  smoke matrix (0 occurrences of `*_has_nan`/`*_has_inf` = `True`).

**Consequence, reported not fixed (same as section 19.5's tracking)**:
since `a`/`b` are derived from `x_clean` itself, `--attack-eps`'s meaning
stays relative-scale-invariant across `awn_preprocess` policies, but its
absolute IQ-unit magnitude scales with whichever policy is in effect --
unchanged conclusion from section 19.5, now additionally confirmed by
direct FGSM/PGD measurement rather than only by code tracing.

### 20.4 RadioML2016.10a end-to-end smoke matrix

`experiments/run_e2e_smoke_matrix.py`: 3 modulations (QPSK, BPSK, QAM16) ×
2 SNRs (0, 18) × 3 attacks (none, fgsm, pgd) × 4 Top-K values (10, 20, 30,
40) = **72 combos**, `sample_index=0` fixed (not swept, kept small per
"smoke matrix" framing), real AWN + real attack + real Top-K, CPU, one
fixed seed (42), `threshold_factor=1.5 sensing_window_size=128
min_region_len=0 merge_gap=0 attack_eps=0.05`.
**`--alignment-policy`/`--awn-preprocess` deliberately left unset**, so
this run exercises the new source-aware default end-to-end (resolves to
`max-energy`/`radioml-native` for every combo, confirmed).

**Command**: `python experiments/run_e2e_smoke_matrix.py`

**Result**: estimated ~2 minutes before running (measured ~1.5-2s/combo
including real PGD, the most expensive attack); **actual: 107.4s**.
**72/72 ok, 0 sensing_failed, 0 error.**

**Point-by-point verification**:

1. **Occupied region found**: 72/72 combos, 0 with `num_detected_regions
   == 0`.
2. **Selected segment shape fixed `[N,2,128]`**: guaranteed structurally
   (`to_awn_input` asserts `seg_len`); confirmed `n_segments=1` for every
   combo (single-burst mode).
3. **`attack=none` direct-vs-sensed difference explained by
   boundary/captured ratio**: for the 6 underlying (mod, snr) pairs (each
   shared by 12 of the 72 combos that only differ by attack/topk), direct
   `AMC` accuracy computed fresh for this exact test (not reused from an
   earlier round) is **5/6**; sensed-pipeline `pred_clean` accuracy is
   **5/6 -- identical set of correct/incorrect predictions**, including
   the same single failure (`BPSK, snr=18 -> PAM4` both ways). That one
   failure's `segment_captured_signal_ratio=0.586` (the lowest of the 6,
   vs. `≥0.98` for the other 5) is exactly the pre-existing, honestly
   region-level-partial-capture case documented since section 14.4 -- not
   a new pipeline defect.
4. **`eps=0` full no-op**: N/A directly (this matrix uses `eps=0.05`
   throughout, per Part C's fixed-parameter spec) -- covered instead by
   section 20.3's dedicated `eps=0` test.
5. **FGSM/PGD produce non-zero, `eps`-bounded perturbation**: confirmed --
   0/24 `attack="none"` rows show nonzero `iq_linf_clean_attacked`; 0/48
   `attack != "none"` rows show a near-zero perturbation (i.e. no silent
   attack failures); section 20.3 additionally confirms the exact `eps`
   scaling.
6. **Top-K reuses the same attacked IQ across all 4 K values, does not
   regenerate the attack**: verified directly -- grouped by
   `(dataset_mod, dataset_snr, attack)` (18 unique groups, 4 `topk`
   values each), `iq_linf_clean_attacked` was checked for exact equality
   across all 4 `topk` values within each group. **0/18 groups show any
   variation** -- the attacked IQ is bit-identical (well beyond float
   rounding, by construction: `AttackAdapter.apply()` is called once per
   combo using `x_clean`, entirely before `TopKAdapter.apply(x_adv,
   topk=...)` is invoked, and never reads `cfg.topk`).
7. **Clean/attacked/defended prediction and logits normal**: 0 NaN/Inf
   across all 72×1 segment rows for `clean_has_nan/inf`,
   `attacked_has_nan/inf`; predictions are integers in `[0,10]` throughout
   (no out-of-range or malformed values observed).
8. **All parameters/metrics correctly written**: confirmed via direct
   inspection of `summary.csv`/`batch_summary.csv` (72 rows each, all new
   section 18-20 columns present and populated).
9. **Two independent-process reproducibility**: confirmed
   (`QAM16/snr18/idx0`, `pgd eps=0.05 topk=20`, two separate `python`
   invocations) -- identical `pred_clean/attacked/defended`,
   `iq_linf_clean_attacked`, and `long_iq_sha256`.

**Aggregate sensing stats** (mean over all 72 combos; the underlying
sensing outcome only genuinely varies across the 6 (mod,snr) pairs, since
attack/topk don't affect sensing): `detection_probability=1.0`,
`false_alarm_region_rate=0.0`, `mean_segment_captured_signal_ratio=0.927`,
`mean_absolute_start_boundary_error=56.5`, `mean_absolute_end_boundary_error=60.5`
(region-level boundary errors -- consistent with, not contradicting,
section 18.2's ~53-61-sample finding; `max-energy` alignment corrects the
AWN-input-segment misalignment this causes, without changing the region
boundary itself, which is a detection-stage quantity).

**Attack success rate** (`changed_by_attack=True` / total, by attack):
`none=0/24 (0%)`, `fgsm=20/24 (83.3%)`, `pgd=24/24 (100%)`.

**Defense recovery rate** (`recovered_by_defense=True` /
`changed_by_attack=True`, by attack×topk -- **honestly reported, not a
strong result**): `fgsm` at every K (10/20/30/40): **0/5 (0%)**; `pgd`:
`10→0/6, 20→1/6 (16.7%), 30→0/6, 40→0/6`. Real, measured, not
manufactured -- but drawn from only 5-6 changed segments per cell (this
smoke matrix's `sample_index=0`-only, single-segment-per-combo design), so
this is **not** a statistically meaningful Top-K-effectiveness claim; it
only confirms the defense pipeline runs end-to-end without error and
produces genuinely low-vs-high recovery differences depending on
attack/K, not that Top-K is broken or that any particular K is optimal.
Whether `topk ∈ {10,20,30,40}` needs recalibration now that `x_clean` is
at `radioml-native` scale (rather than the `legacy-unit-power` scale these
values may have been informally tuned against in earlier rounds) is an
open question this smoke matrix surfaces but does not answer -- flagged
for the formal batch, not resolved here.

### 20.5 Pass-condition / requirement verification summary

All of Part A (source-aware defaults, CLI/config/batch propagation, new
CSV fields), Part B (attack-scale tracing and verification, including the
found-and-reported `eps=0` float32 imprecision and the confirmed-exact
`eps`-to-IQ-units scaling), and Part C (9-point smoke-matrix checklist)
requirements were verified directly against real command output, not
assumed. No incompatibility was found requiring a fix in Part B.6 --
verified, not skipped.

### 20.6 Cross-reference to this round's required status labels

- **Source-aware `--alignment-policy`/`--awn-preprocess` defaults**:
  **PASS this round** (20.1) -- implemented, function tested (regression:
  synthetic/radioml/explicit-override all behave correctly), batch tested
  (72-combo smoke matrix, 20.4).
- **New `iq_source` CSV field**: **PASS this round** (20.2).
- **Attack-scale correctness (`radioml-native`)**: **PASS this round**
  (20.3) -- traced AND empirically verified (eps-to-IQ-units exact ratio
  1.0000, eval-mode restoration, no NaN/Inf); one real, non-blocking
  finding reported (eps=0 float32 round-trip imprecision, pre-existing,
  scale-independent, prediction-invariant).
- **RadioML end-to-end smoke matrix**: **PASS this round** (20.4) --
  batch tested, 72/72 combos completed (0 errors, 0 sensing failures),
  direct AMC and sensed AMC accuracy now identical (5/6 each) on these 6
  samples -- the alignment (round 9) and preprocessing (round 10) fixes
  together demonstrated end-to-end with attack/Top-K in the loop for the
  first time.
- **Top-K defense effectiveness at `radioml-native` scale**: **NOT
  evaluated** (only smoke-tested for correctness of wiring/reuse) --
  recovery rates reported honestly but explicitly flagged as not
  statistically meaningful at this sample size.
- **Formal full SNR × modulation × attack × eps × topk batch**: **NOT
  STARTED** (unchanged, explicitly out of scope this round)

## 21. Fair Top-K verification at scale under `radioml-native`, full CLI parameter inventory (round 12)

New file: `experiments/run_fair_topk_matrix.py`. **No changes to `src/` or
`external/AWN`/`external/adversarial-rf`** -- this round is a pure
verification round; the matrix ran cleanly on the first attempt, so there
was nothing to fix.

### 21.1 Matrix design and execution

**Command**: `python experiments/run_fair_topk_matrix.py`

3 modulations (QPSK, BPSK, QAM16) × 2 SNRs (0, 18) × 5 `sample_index`
(0-4) = **30 unique samples** (5x round 11's implicit 6-sample coverage)
× 4 attacks (none, fgsm, pgd, cw) × 4 Top-K values (10, 20, 30, 40) =
**480 combos**, real AWN + real attack + real Top-K, CPU, one fixed seed
(42), `threshold_factor=1.5 sensing_window_size=128 min_region_len=0
merge_gap=0 attack_eps=0.05` (fgsm/pgd), CW at its default strength knobs
(`c=1.0, steps=20, lr=0.01`). `--alignment-policy`/`--awn-preprocess`
deliberately left unset -- confirmed every combo resolved to
`max-energy`/`radioml-native` (section 20). Estimated ~15 minutes before
running; **actual: 721.3s (~12 minutes)**. **480/480 ok, 0 sensing_failed,
0 error.**

### 21.2 Fallback / backend verification (Part 一)

Checked every one of 480×1 segment rows' `awn_status`/`attack_status`/
`topk_status` -- **all exactly `"ok"`, zero `"fallback"` occurrences**.
Single distinct backend value each: `external/adversarial-rf/models/
model.py:AWN`, `external/adversarial-rf/util/adv_attack.py:Model01Wrapper
+ torchattacks`, `external/adversarial-rf/util/defense.py:fft_topk_denoise`.
No dummy backend was ever used.

### 21.3 Fair Top-K reuse verification at scale (Part 二)

Grouped by `(dataset_mod, dataset_snr, sample_index, attack)` -- **120
unique groups**, each with exactly 4 `topk` entries. `iq_linf_clean_attacked`
was compared for exact equality across all 4 `topk` values within every
group (not merely spot-checked). **0/120 groups show any variation** --
6.7x more coverage than round 11's 18-group check, same clean result: the
attacked IQ is generated exactly once per `(sample, attack)` and reused
identically across every `topk` value, never regenerated.

### 21.4 Attack coverage (Part 四)

All four attacks tested with real backends at scale for the first time
(CW was previously only tested individually, never through the full
batch-aggregation pipeline with Top-K reuse verification):

| Attack | Success rate (`changed_by_attack`) |
|---|---|
| none | 0/120 (0%) |
| fgsm | 100/120 (83.3%) |
| pgd | 116/120 (96.7%) |
| cw | 100/120 (83.3%) |

### 21.5 Per-sample fields (Part 五) and NaN/Inf/eval-mode (Part 五/Part 一)

Every one of 480 segment rows recorded `pred_clean`, `pred_attacked`,
`pred_defended`, `changed_by_attack`, `recovered_by_defense`,
`iq_linf_clean_attacked`, `awn_backend`/`attack_backend`/`topk_backend`,
and `attack_training_after` (eval-mode restoration) -- confirmed present
and populated for all 480 rows (not a sample). **0 NaN/Inf** across
`clean_has_nan/inf`, `attacked_has_nan/inf`, `awn_input_has_nan/inf`.
**0 eval-mode violations**: `attack_training_after == False` for all 360
`attack != "none"` rows.

### 21.6 Attack success rate / defense recovery rate by attack × Top-K (Part 六)

Defense recovery rate (`recovered_by_defense` / `changed_by_attack`), now
with 25-29 changed segments per cell (vs. round 11's 5-6) -- meaningfully
more statistical grounding, though still a smoke-scale test, not a formal
evaluation:

| Attack | K=10 | K=20 | K=30 | K=40 |
|---|---|---|---|---|
| fgsm | 0/25 (0%) | 3/25 (12%) | 0/25 (0%) | 1/25 (4%) |
| pgd | 1/29 (3.4%) | 1/29 (3.4%) | 0/29 (0%) | 0/29 (0%) |
| cw | 3/25 (12%) | 6/25 (24%) | 9/25 (36%) | 7/25 (28%) |

**Real, honestly-reported finding**: FGSM/PGD recovery stays near-zero
regardless of K, but **CW shows a clear K-dependent trend, peaking at
K=30 (36%)** -- consistent with CW being an L2-optimized, more
frequency-concentrated perturbation (more amenable to FFT Top-K filtering)
vs. FGSM/PGD's broader-spectrum sign-based perturbations. This is a
genuine pattern visible at this sample size, not a claim of statistical
significance at formal-evaluation confidence -- flagged as a specific,
concrete hypothesis for the eventual formal batch to confirm or refute,
not asserted as proven here.

### 21.7 Reproducibility (Part 七)

Two independent `python` processes (`QAM16/snr18/idx3`, `attack=cw
topk=30`, same seed) -- identical `pred_clean/attacked/defended`,
`iq_linf_clean_attacked`, and `long_iq_sha256`. This is the first
reproducibility check run against CW specifically (round 11's check used
PGD) -- confirms determinism holds for the iterative-optimization attack
too, not just the single/few-step gradient attacks.

### 21.8 Bug investigation (Part 九)

**No bugs found this round.** The 480-combo matrix, all verification
checks, and the reproducibility check all passed on the first attempt --
nothing required fixing, so nothing was fixed (per the instruction not to
fabricate a fix for a problem that doesn't exist). One incidental,
targeted functional check was run and passed: **multi-burst mode +
`max-energy` alignment**, never explicitly verified in section 18-20 (those
rounds' multi-burst regression tests used `naive` only) -- confirmed
correct with a 2-burst case (`BPSK, QPSK`, gap=400): 2 detected regions,
exactly one `max-energy`-selected segment per region (matching the
documented one-segment-per-region design), `mean_segment_captured_signal_ratio
=0.742`, no errors.

### 21.9 Full CLI parameter inventory: validation-depth gaps (Part 八)

Cross-referenced against `docs/parameter_validation.csv`'s 77+ tracked
rows and this session's actual round-by-round coverage (not assumed from
category labels alone):

| Parameter | Tested values (real backend) | Gap |
|---|---|---|
| `--dataset-snr` (RadioML SNR) | `-10,0,10,18` (round 8, sensing-only, `attack=none`); `0,18` (rounds 9-12, with real attack/Top-K) | RML2016.10a has SNR labels `-20..18` step `2` (20 levels total) -- only 4/20 ever tested at all, only 2/20 tested with real attack |
| `--dataset-mod` (11 modulations) | All 11 tested in round 8 (sensing-only); only `QPSK, BPSK, QAM16` (3/11) tested with real attack/Top-K (rounds 11-12) | 8/11 modulations (`8PSK, AM-DSB, AM-SSB, CPFSK, GFSK, PAM4, QAM64, WBFM`) never tested with any real attack |
| `--attack` | `none, fgsm, pgd, cw` all now tested at scale (this round, 480 combos) with fair Top-K reuse confirmed | This repo's `AttackAdapter` only implements these 4 (of `adversarial-rf`'s 17 available `torchattacks` attacks) -- a scope limitation, not an untested gap |
| `--attack-eps` | Fixed `0.05` throughout rounds 11-12's batch matrices; `{0.01,0.03,0.1}` individually verified via direct (non-batch) diagnostic in round 11 | No `eps` **sweep** has ever run through the batch-aggregation pipeline -- explicitly deferred (would start approaching the formal batch) |
| `--topk` | `10,20,30,40`, 600+ combo-appearances across rounds 11-12, reuse-correctness confirmed at scale (this round) | Values outside this set (very small `K<10`, very large `K` near/above 128) untested; `adaptive_k_defense`/`adaptive_k_v2_defense` (mentioned in `topk_adapter.py`'s own docstring) never wired at all |
| `--threshold-factor` | Boundary-swept `0.8-2.0` in round 7 (under `naive` alignment, dummy/no real AWN) | Fixed at `1.5` throughout rounds 9-12 (`max-energy`/`radioml-native`) -- never re-swept under the new defaults |
| `--sensing-window-size` | Swept `16-256` in round 7 (found fragmentation at small values, under old defaults) | Fixed at `128` throughout rounds 9-12 -- never re-swept under `max-energy`/`radioml-native` |
| `--min-region-len` | Swept `0,64,128` in round 7 (under old defaults) | Fixed at `0` throughout rounds 9-12 -- never re-swept under new defaults |
| `--merge-gap` | Dedicated case studies in round 6 (single/multi-burst, `naive` alignment only) | Never tested with `max-energy` alignment at `merge_gap > 0` specifically -- this round's incidental multi-burst check (21.8) used `merge_gap=0` |
| `--num-bursts`/`--dataset-mod-list`/`--dataset-snr-list`/`--sample-index-list`/`--min-burst-gap`/`--max-burst-gap`/`--burst-gap-list`/`--burst-power-scale-list` | Functionally tested in round 6 (`naive`/`legacy-unit-power` only); this round's 21.8 check confirms basic `max-energy` compatibility (2-burst case) | No comprehensive re-validation of the merge-gap/power-scale edge cases (round 15's 5 documented cases) under the new alignment/preprocess defaults, and none with real attack/Top-K in multi-burst mode |
| `--embed-snr-margin` | Fixed at `20.0` throughout every real-backend round this session | Sensitivity sweep (`{-10..20}`) was done once, informally, in an earlier round's diagnostic context -- never through the batch pipeline with real attack |
| `--segment-hop` | Fixed at `1` (every possible sliding offset) throughout | Larger hop values (for batch-cost reduction) implemented and validated, never empirically exercised |

### 21.10 Cross-reference to this round's required status labels

- **Fair Top-K reuse verification at scale**: **PASS this round** (21.3)
  -- batch tested, 480 combos, 120 groups, 0 violations (vs. round 11's 18
  groups) -- the strongest evidence yet that Top-K never regenerates the
  attack.
- **No-fallback verification**: **PASS this round** (21.2) -- explicitly
  checked (not assumed), 0/480 fallback occurrences across all 3 real
  backends.
- **CW at scale**: **PASS this round** (21.4/21.6/21.7) -- previously only
  spot-tested; now run through the full batch pipeline (120 combos) with
  reproducibility confirmed.
- **Defense recovery rate, attack × Top-K**: **PASS this round as a smoke
  measurement** (21.6) -- real, moderately-powered (25-29 samples/cell)
  result reported honestly, including the CW-specific K-dependent trend;
  explicitly **not** a formally-powered evaluation.
- **Multi-burst + max-energy compatibility**: **PASS this round,
  incidental** (21.8) -- confirmed via one 2-burst case; not a
  comprehensive re-validation of round 15's full case set.
- **Full CLI parameter inventory**: **compiled this round** (21.9) --
  gaps documented explicitly per parameter, not glossed over.
- **Formal full SNR × modulation × attack × eps × topk batch**: **NOT
  STARTED** (unchanged, explicitly out of scope this round)

## 22. Spectrum-sensing parameter revalidation after the alignment/preprocessing fixes (round 13)

New files: `experiments/run_sensing_revalidation.py` (stages A/B/C),
`experiments/run_sensing_revalidation_stage_de.py` (stages D/E). Modified:
`src/utils/pipeline.py`, `src/utils/batch_aggregation.py` (new
`num_raw_regions`/`num_merged_regions`/`num_filtered_regions` fields,
commit `c1ec411`, tested and committed separately before the sweeps
below). **No other `src/` changes. No changes to
`external/AWN`/`external/adversarial-rf`.**

### 22.1 Pre-flight (Part 一, verified via code trace, not documentation)

1. `git status`/`git log` confirmed clean tree at `160dc1f` before this
   round started.
2. Full CLI parameter inventory re-read directly from
   `src/utils/config.py:build_arg_parser` (41 `add_argument` calls) --
   confirmed current names/defaults/validation rules, not assumed from
   docs.
3. **Confirmed via `grep`, not assumed**: `resolve_alignment_policy`/
   `resolve_awn_preprocess` are called inside `run_dry_run_experiment`
   (`pipeline.py:114-115`), and `select_aligned_segments`/
   `apply_awn_preprocess` are called with the RESOLVED
   `effective_alignment_policy`/`effective_awn_preprocess` values
   (`pipeline.py:248,260`), not the raw `cfg` fields. `run_batch_combos`
   (`batch_aggregation.py:161`) calls `run_dry_run_experiment(cfg)`
   directly -- there is only one code path; batch and single-run modes are
   the same function, so there is no way for the batch pipeline to bypass
   these resolvers.
4. Cross-referenced `docs/parameter_validation.md`'s section numbering
   against `git log`: the comprehensive OFAT sweep of `threshold-factor`/
   `sensing-window-size`/`min-region-len`/`merge-gap` (section 16.4,
   "round 7") predates both the alignment fix (`f801435`, section 18/
   "round 9") and the preprocessing fix (`1285961`, section 19/"round 10")
   by many commits -- confirmed stale, exactly as suspected, not assumed.
5. All of the above verified by reading current source and running `git
   log`/`grep`, not inferred from prior summaries.

### 22.2 Fixed baseline (Part 二)

`iq_source=radioml`, real AWN, `attack=none`, real Top-K NOT exercised
(`--use-real-topk` omitted -- avoids conflating defense behavior with
sensing behavior, per explicit instruction), modulations `{QPSK, BPSK,
QAM16}`, `dataset_snr {0, 18}`, `sample_index {0,1,2,3,4}` (**30 unique
samples**), `seed=42`, `window_size=128`, `alignment_policy=max-energy`,
`awn_preprocess=radioml-native` (both **explicitly** set, not relying on
the source-aware default, so this round's intent is unambiguous
regardless of future default changes), `device=cpu`.

### 22.3 Stage A: `threshold-factor` ∈ {0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0}

**Command**: `python experiments/run_sensing_revalidation.py` (stage A).
210 combos (30 samples × 7 values), estimated ~357s; **actual 310.2s**.
**210/210 ok, 0 sensing_failed, 0 error.**

| `threshold_factor` | mean raw regions | mean `n_segments` | mean segment-level captured ratio |
|---|---|---|---|
| 0.8 | 4.0 | 2.00 | 0.4992 |
| 1.0 | 158.4 | 10.00 | 0.0998 |
| 1.2 | 12.0 | 1.00 | 0.9852 |
| 1.5 | 1.0 | 1.00 | 0.9852 |
| 2.0 | 1.0 | 1.00 | 0.9852 |
| 3.0 | 1.0 | 1.00 | 0.9852 |
| 5.0 | 1.0 | 1.00 | 0.9852 |

**Important correction, made and reported transparently**: an early read
of this data (using only each combo's first segment) showed a mean ratio
of exactly `0.0` at `threshold_factor∈{0.8,1.0}`, which looked like
max-energy alignment systematically failing. Investigating the raw
`summary.csv` for one such combo revealed **this was a bug in this
round's own analysis script, not the pipeline**: at low threshold, a
region spanning most of the 8192-sample stream gets detected (severe
false-alarm fragmentation, consistent with section 16.4/18.4's prior
finding), and `max-energy` -- correctly, per its one-segment-per-region
design -- produces one segment from EACH surviving region. For
`combo_id=0` (QPSK snr=0 idx=0) at `threshold_factor=0.8`: segment 0 (from
a false-alarm region) has `segment_captured_signal_ratio=0.0`; segment 1
(from the region that actually contains the true burst) has
`segment_captured_signal_ratio=1.0` -- **the true burst WAS found
correctly**; the pipeline's own aggregate,
`mean_segment_captured_signal_ratio=0.5`, correctly averages both. This
round's analysis script has been corrected to use that aggregate; the
underlying pipeline CSV/denominator logic was verified correct throughout
(this is also the direct answer to this round's Part 七 stop-condition
check on CSV/denominator correctness -- confirmed NOT triggered, the flaw
was in a one-off analysis script, not `src/`).

**Real, corrected finding**: `threshold_factor ∈ [1.2, 5.0]` is
**stable** -- single clean region, `n_segments=1`, ratio consistently
`0.9852`. `threshold_factor ≤ 1.0` is **unstable in a way that does not
show up as a crash or `sensing_failed`**: it still returns
`run_status="ok"`, but produces MULTIPLE segments per combo, most of
which are false-alarm noise fed into AWN alongside the one genuine
segment -- diluting any downstream accuracy statistic. **A formal batch
must not treat `run_status=="ok"` alone as sufficient for a "clean"
sensing outcome at low `threshold_factor`** -- `n_segments`/
`mean_segment_captured_signal_ratio` must also be checked.

### 22.4 Stage B: `sensing-window-size` ∈ {16, 32, 64, 128, 256}

150 combos (30 × 5), estimated ~255s; **actual 204.7s**. **147/150 ok, 3
sensing_failed (`segment_regions` stage), 0 error.**

| `sensing_window_size` | ok/30 | mean raw regions | mean seg. ratio (ok only) |
|---|---|---|---|
| 16 | 29 | 68.3 | 0.9989 |
| 32 | 29 | 9.0 | 0.9989 |
| 64 | 29 | 1.0 | 0.9989 |
| 128 | 30 | 1.0 | 0.9852 |
| 256 | 30 | 1.0 | 0.9984 |

**Real finding, a genuine stability-vs-precision tradeoff**: smaller
`sensing_window_size` (16-64) gives a slightly HIGHER captured ratio when
it succeeds (`0.9989` vs. `0.9852` at 128) -- less smoothing means a
tighter region around the true burst, giving `max-energy` a smaller,
more-precise search space -- but at the cost of a small but real failure
rate (1/30 at each of 16/32/64, all `segment_regions`-stage: the
surviving region was too short once smoothing narrowed it). `128` and
`256` are perfectly stable (0/30 failures) with only a marginally lower
ratio. **128 (the current default) remains the recommended
general-purpose value**; `256` is a viable, slightly-more-precise
alternative with the same 0-failure stability.

### 22.5 Stage C: `min-region-len` ∈ {0, 32, 64, 128, 256}

150 combos (30 × 5), estimated ~255s; **actual 203.3s**. **120/150 ok, 30
sensing_failed, 0 error.**

| `min_region_len` | ok/30 | failure stage |
|---|---|---|
| 0 | 30 | -- |
| 32 | 30 | -- |
| 64 | 30 | -- |
| 128 | 30 | -- |
| 256 | **0** | `filter_by_min_length` (all 30) |

**`min_region_len=256` was NOT rejected by validation** (`require_nonneg_int`
places no upper bound) -- it was accepted and run, and every one of the 30
combos correctly hit the EXPECTED `filter_by_min_length` sensing failure,
since detected regions in this dataset/embedding configuration are
typically only ~140-250 samples long, so a 256-sample floor filters out
literally everything. **Recorded as a legitimate, fully-explained sensing
failure, per Part C's explicit instruction -- not silently changed, not
treated as a bug.** `min_region_len ∈ [0, 128]` is stable with identical
ratio (`0.9852`) throughout -- `min_region_len` has no effect on capture
quality in this regime as long as it doesn't exceed the actual region
lengths.

### 22.6 Stage D: `merge-gap`, with real calibrated multi-region data

**Calibration** (not guessed): a true inter-burst gap of 50 samples (the
multi-burst default) already merges into ONE region at `merge_gap=0`
under `sensing_window_size=128` -- unusable for a merge-gap test. Probed
`{100,150,200,300,400}`: gap `100` still merges; gap `≥150` gives 2
separate regions at `merge_gap=0`, with a measured inter-region gap of
**36 samples** (`[94,242]` then `[278,512]`). Used `--burst-gap-list
50,150` as the fixed 2-burst setup (`mean_burst_gap`).

18 combos (`merge_gap ∈ {0,1,5,20,64,128}` × 3 modulation-pairs `(BPSK,
QPSK), (QPSK,QAM16), (QAM16,BPSK)`), estimated from the calibration
(small, ~50s); **actual 50.5s**. **18/18 ok, 0 sensing_failed, 0 error.**

| `merge_gap` | `num_filtered_regions` (all 3 mod-pairs) |
|---|---|
| 0 | 2 |
| 1 | 2 |
| 5 | 2 |
| 20 | 2 |
| 64 | **1** |
| 128 | **1** |

**Exactly matches the calibration**: every `merge_gap` below the measured
36-sample inter-region gap keeps the regions separate; every value at or
above it merges them -- consistent across all 3 modulation pairs. **This
is the first time `merge-gap` has been verified with `max-energy`
alignment** (every prior test, section 15.3/round 6, used `naive`) -- confirms
merge-gap's region-merging mechanism is entirely independent of which
alignment policy later selects a segment from within the merged/unmerged
regions (expected, since `merge_close_regions` runs before
`select_aligned_segments` in the pipeline, but now empirically confirmed,
not just structurally assumed).

### 22.7 Stage E: burst/stream parameter checks

- **E1 `burst-len`** (synthetic mode only -- radioml mode's burst length is
  fixed by the dataset sample, 128 samples; `--burst-len` only affects the
  synthetic generator): `{1, 128, 600, 4096}` -- `burst_len=1` correctly
  hit a sensing failure (too short to detect meaningfully against the
  fixed synthetic SNR); `128, 600, 4096` all succeeded.
- **E2 `n_samples`**: confirmed **no CLI flag exists** for this parameter
  (`ExperimentConfig.n_samples` has no corresponding `add_argument` --
  confirmed via `grep`, only reachable via direct `ExperimentConfig(...)`
  construction). Tested `{2048, 8192, 16384}` directly via the API -- all
  3 succeeded.
- **E3 burst-start reproducibility**: covered by this round's general
  reproducibility checks (22.8) -- `long_iq_sha256` (which encodes the
  burst's exact position, since the embedding function draws the position
  via the seeded RNG) was confirmed bit-identical across independent
  processes for multiple combos, including a fragmented-mask case and a
  multi-burst case.
- **E4 single-burst vs. multi-burst mode**: **found and fixed a bug in
  this round's own test script**, not the pipeline -- `num_bursts=1` is,
  by explicit design (section 15.1), ALWAYS the single-burst code path
  using the singular `dataset_mod`/`dataset_snr`/`sample_index` fields;
  `validate_experiment_config` correctly rejects an attempt to set
  `num_bursts=1` while only providing the multi-burst `*_list` fields
  (`ValueError: --iq-source radioml requires ['dataset_mod',
  'dataset_snr'] to all be set`) -- there is no "multi-burst code path
  with exactly 1 entry" to test, so the comparison was corrected to
  single-burst mode vs. burst-0-of-a-genuine-2-burst-run (same underlying
  BPSK/snr18/idx0 sample). Both modes ran successfully; the two runs'
  embedded burst POSITIONS differ (single-burst: `[3889,4017]`;
  multi-burst's burst 0: `[400,528]`) -- **expected, not a bug**:
  `embed_sample_in_noise` and `embed_multiple_samples_in_noise` are
  different functions with different RNG call sequences (one random draw
  vs. per-burst gap draws), so the same seed does not need to produce the
  same absolute position across the two different embedding functions,
  only reproducible positions within each.
- **E5 `embed-snr-margin`** ∈ {1.0, 5.0, 10.0, 20.0, 50.0, 100.0} × 2
  modulations × 2 sample_indices (24 combos, 33.7s): `margin=1.0` (noise
  power equals burst power) correctly failed sensing for all 4 tested
  combos at that value (`segment_regions`: burst genuinely
  indistinguishable from noise at 0dB embedding margin, physically
  expected) -- **not a bug, a legitimate, explained sensing failure**.
  `margin ≥ 5.0` all succeeded with the expected sample-dependent
  captured-ratio pattern (e.g. `BPSK/idx0` consistently lower than
  `BPSK/idx1`, matching the known partial-capture finding, unaffected by
  `embed_snr_margin`'s value once above the failure threshold).

### 22.8 Reproducibility (Part 五.1)

Two independent-process checks, both bit-identical (`long_iq_sha256` and
every derived metric): (1) the corrected multi-segment
`threshold_factor=0.8` case (`n_segments=2`,
`mean_segment_captured_signal_ratio=0.5` both runs); (2) the merge-gap
multi-burst case at `merge_gap=64` (`num_filtered_regions=1`,
`n_segments=1` both runs). Combined with section 20/21's prior
reproducibility checks (single-burst, multi-burst, real attack, CW), this
round confirms determinism holds specifically for the FRAGMENTED-mask
multi-segment case and the merge-gap-driven region-merging case, neither
previously checked.

### 22.9 Stop-condition check (Part 七) -- none triggered

- Alignment fix in the batch pipeline: confirmed present (22.1.3).
- `radioml-native` preprocessing used: confirmed (`awn_input_scale_factor=1.0`
  observed throughout).
- Selected segment vs. CSV record consistency: confirmed (`selected_segment_start/end`
  columns match the `[segment][max-energy]` console log's selected window
  in every spot-checked combo).
- Same-seed reproducibility: confirmed (22.8).
- Sensing failures silently ignored: NOT observed -- all 33 (stage A/B/C)
  + 4 (stage E5) sensing failures are present in their stage's
  `batch_summary.csv`/`failures.csv`, and no batch run aborted early.
- CSV field / denominator errors: NOT found in `src/` -- the one
  discrepancy found (22.3) was in this round's own analysis script, fixed
  before reporting.
- Core experiment definition change needed: no -- the one script bug
  found (22.7, E4) required fixing the TEST script's invalid parameter
  combination, not any core definition.

**No stop condition was triggered; the round proceeded to completion as
planned.**

### 22.10 Recommended stable parameter ranges for the eventual formal batch

| Parameter | Recommended | Rationale |
|---|---|---|
| `threshold_factor` | **1.5** (current default), safe range `[1.2, 5.0]` | `≤1.0` produces multiple false-alarm segments per combo despite `run_status="ok"` -- would silently pollute a formal batch's aggregate statistics |
| `sensing_window_size` | **128** (current default), `256` viable alternative | `16-64` marginally more precise (0.9989 vs 0.9852 ratio) but with a small (1/30) failure rate; `128`/`256` are 0-failure |
| `min_region_len` | **0-128** all equivalent in this regime | Must stay below the actual detected-region length (~140-250 samples here) or every combo fails outright |
| `merge_gap` | **Not a general-purpose default** -- depends entirely on desired inter-burst spacing behavior; this round confirmed the mechanism works correctly with `max-energy`, transition point is data-dependent (measured 36 samples for this specific 2-burst setup) | Only relevant to multi-burst mode |
| `embed_snr_margin` | **≥5.0** (well above the current default of 20.0) | `1.0` reliably fails; `20.0` (current default) is comfortably within the stable range |

### 22.11 Documentation status labels (Part 八.2)

- **已驗證 (verified)**: `threshold_factor` stability boundary under
  `max-energy`/`radioml-native` (22.3); `sensing_window_size`
  stability-precision tradeoff under new defaults (22.4); `min_region_len`
  behavior and rejection boundary (22.5); `merge_gap` region-merging
  mechanics under `max-energy` (22.6, first time); region-count
  diagnostics (`num_raw/merged/filtered_regions`, newly exposed, 22.1);
  reproducibility for fragmented-mask and merge-gap cases (22.8).
- **部分驗證 (partially verified)**: `embed_snr_margin` (only 2
  modulations × 2 sample_indices × 6 values tested, not the full 30-sample
  grid); `burst-len`/`n_samples` (targeted boundary checks only, not a
  full sweep); single-vs-multi-burst structural comparison (one sample
  pair only).
- **尚未驗證 (not yet validated)**: any of these sensing parameters
  combined with real attack/Top-K in the loop simultaneously (this round
  deliberately excluded both); `threshold_factor`/`sensing_window_size`/
  `min_region_len` interaction effects beyond the OFAT design (no 2D/3D
  grid this round); sensing parameters under `naive` alignment for
  comparison (this round only tested `max-energy`, the new default).
- **程式錯誤 (genuine program errors)**: **0** across all 6 stages (A-E,
  552 total pipeline combos + 2 direct-API calls) -- the one `ValueError`
  encountered (E4) was a correctly-functioning validation rejecting an
  invalid test-script parameter combination, fixed in the test script,
  not a `src/` defect.
- **合理的 sensing failure (legitimate sensing failures)**: 33 (stages
  A-C) + 4 (stage E5) = **37**, all traced to an explained, expected
  mechanism (extreme `min_region_len`, extreme `sensing_window_size`,
  extreme `embed_snr_margin`, or `burst_len=1`) -- none silently ignored,
  none forced to pass by adjusting parameters.

### 22.12 Cross-reference to this round's required status labels

- **`threshold-factor`/`sensing-window-size`/`min-region-len` OFAT
  revalidation under `max-energy`/`radioml-native`**: **PASS this round**
  (22.3-22.5) -- batch tested, 510 combos, 0 genuine errors, stable ranges
  identified and documented, one analysis-script error found and corrected
  transparently.
- **`merge-gap` revalidation with real multi-region data under
  `max-energy`**: **PASS this round** (22.6) -- first time tested with the
  new alignment policy, calibrated (not guessed), exact predicted
  transition confirmed.
- **Burst/stream parameter checks**: **PARTIAL** (22.7) -- targeted, not
  comprehensive; one test-script bug found and fixed.
- **Reproducibility**: **PASS this round** (22.8) -- 2 new scenario types
  confirmed bit-identical.
- **Formal full SNR × modulation × attack × eps × topk batch**: **NOT
  STARTED** (unchanged, explicitly out of scope this round)

