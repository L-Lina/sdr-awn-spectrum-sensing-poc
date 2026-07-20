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
