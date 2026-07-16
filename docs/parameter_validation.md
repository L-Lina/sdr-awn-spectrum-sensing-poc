# Parameter Validation Audit

Status: read-only audit performed by cross-checking actual source code against
CLI parsers, `ExperimentConfig`, adapters, `docs/experiment_design.md`,
`docs/integration_plan.md`, README, git history (10 commits total, this repo
has no separate historical parameter-value record of its own), and
`external/adversarial-rf`'s own scripts (read-only, submodule untouched).

This document records the state as of commit `0aa95ea`
(`Fix attack domain and saturated-gradient handling`); `git status` is clean
at that commit. Every claim below is either a direct file:line citation or an
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
  -> energy_detect(window, threshold_factor)                    [src/sensing/energy_detection.py]
  -> mask_to_regions -> merge_close_regions(merge_gap)
       -> filter_by_min_length(min_region_len)
  -> segment_regions(seg_len=window_size)                       [src/sensing/segmentation.py]
  -> normalize_segments (unit-average-power, NOT clamped to [-1,1])
  -> to_awn_input()                                             [src/sensing/normalize.py]
       => x_clean  [N, 2, window_size] float32
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
  `--min-region-len`, `--merge-gap` are CLI-exposed. Overlap/hop-size, max
  segment count, and sample rate are not implemented anywhere in this repo.
- **D. Segmentation**: segment length is the same parameter as
  `--window-size` (no separate flag). No overlap/hop parameter exists in
  `src/sensing/segmentation.py:segment_regions` (fixed non-overlapping
  windows only). No max-segments cap exists.
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
| window-size | yes | yes | yes |
| min-region-len | yes | yes | yes (0-value bug fixed, see section 8) |
| merge-gap | yes | yes | yes |
| burst-len | yes | yes | yes |
| stream length (n_samples) | yes | yes | **no** |
| burst start | yes (computed, centered) | yes | **no** |
| burst amplitude | yes (hardcoded `1.0`) | yes | **no** |
| noise standard deviation | yes (derived from SNR) | yes | **no** (only indirectly via `--snr`) |
| seed | yes (hardcoded `0`) | yes | **no** |
| segment length | = window-size, shared param | yes | yes (via `--window-size`) |
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
| `cw` | yes | yes (`torchattacks.CW(c=1.0, steps=20, lr=0.01)` hardcoded) | yes (ran once without error) | **no** — only tested once, before the train/eval-mode fix, before the min-max correctness fix, and before temperature scaling existed; that old result is invalid under the current code and has not been re-run | **needs re-verification** |
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
- `attack=fgsm`, `attack=pgd` end-to-end (post all 3 correctness fixes)
- `--attack-temperature` positivity validation (`<=0` raises a clear error)
- Cross-process reproducibility of synthetic IQ generation (post `hashlib` fix)
- AWN model eval-mode restoration after real attacks
- `--min-region-len` propagation: unset -> `window_size`, explicit `0` ->
  `0` (preserved), explicit `64` -> `64`. Verified via a scratch script
  calling `build_arg_parser`/`args_to_config` (`src/utils/config.py`) and
  `build_batch_arg_parser` + the equivalent resolution expression
  (`experiments/run_batch.py`) directly, at both the raw `argparse.Namespace`
  layer and the resolved config-value layer. Config-layer only; not
  re-verified through the full sensing/AWN/attack/Top-K pipeline in this
  round (out of scope per this fix's instructions).

**部分通過 (partial)**
- `--snr`: tested at -10, 0, 10, 18 (real backends); no upper-bound or
  fractional-value testing
- `--mod`: cosmetic-only behavior confirmed for 4 values; arbitrary/malformed
  strings not tested
- `--topk`: tested at 10, 20, 30, 40; default `50` never actually run;
  boundary values (0, negative, > window_size) not tested
- `--attack-eps`: tested at 0.03 (dummy era), 0.1/0.2/0.3/0.5 (real,
  post-fix); `eps<=0` has no validation and was only tried once via a
  scratch script, never via CLI
- `--device`: only `cpu` tested (no GPU available on this machine); `cuda`
  path completely unexercised
- `--use-real-awn`/`--use-real-topk`/`--use-real-attack`: `True` path fully
  exercised; `False` (dummy) path never actually run in-session

**未測 (not tested)**
- `--burst-len`, `--window-size`, `--merge-gap`: only ever run at their
  respective default values (600, 128, 0); never varied
- `--threshold-factor`: every real-backend test in-session used `1.5`; the
  CLI default `5.0` was never actually run
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
- Global torch determinism / seed CLI (not needed per in-session empirical
  verification, but also simply not present)

---

## 7. Outstanding items before a formal experiment

1. **CW must be re-verified** under the current code (train/eval fix +
   min-max fix + temperature scaling). The one existing CW result predates
   all three fixes and should not be cited as evidence CW works correctly.
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
5. ~~`--min-region-len 0` cannot actually be set~~ — **fixed**, see section 8.
   Negative `--min-region-len` values are still unvalidated (see section 8);
   that is a separate, not-yet-addressed item.
6. **No boundary-value testing** exists for `threshold-factor`, `window-size`,
   `burst-len`, `merge-gap`, `topk`, or `attack-eps` (zero, negative, or
   extreme values). Only "normal" values have been exercised.
7. **Segmentation has no overlap/hop-size or max-segments control** — if a
   formal experiment design needs either, they must be built first.
8. **`cuda` device path is completely unverified** — this development
   machine has no GPU (`torch.cuda.is_available()` returns `False`).

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
  was not touched — it already accepted `0`/negative `min_len` correctly
  (see the negative-value note below); the bug was purely in how the CLI
  value reached it. **Negative `--min-region-len` values remain
  unvalidated** — `filter_by_min_length`'s `(e - s) >= min_len` comparison
  makes any negative value behave identically to `0` (a no-op filter, no
  crash), but neither argparse, `ExperimentConfig`, nor the energy detector
  block or warn about a negative value; this was intentionally left
  unaddressed in this fix and is not marked as passed anywhere in this
  document.
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
