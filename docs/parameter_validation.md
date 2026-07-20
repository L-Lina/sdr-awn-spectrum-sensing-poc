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
