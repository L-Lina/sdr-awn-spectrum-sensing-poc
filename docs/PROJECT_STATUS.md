# Project Status

Single-source progress summary for `sdr-awn-spectrum-sensing-poc`. Generated
from the current state of the repository: `git log`, `docs/
formal_experiment_plan.md`, `docs/formal_experiment_matrix.csv`, the actual
files under `results/` (gitignored, inspected directly on disk), and the
current adapter/experiment source code. Does not rely on any prior chat
summary. Status labels used throughout:

- **已完成 (done)** -- executed, verified, results exist on disk and/or are
  documented in `formal_experiment_plan.md`.
- **已設計但未執行 (designed, not run)** -- a runner script and/or a row in
  `formal_experiment_matrix.csv` exists with a concrete combo count and
  dry-run, but no execution has happened.
- **尚未實作 (not implemented)** -- no code path exists yet.

Last updated: this round (round 28), which created this document; repo
`HEAD` is still `0cccc78` (round 27, formal Phase 4 K-reduced full-N
execution) -- no code, experiment, or result changed this round.

---

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

---

## 7. Latest important commits

| Commit | Content |
|---|---|
| `0cccc78` (HEAD) | Formal Phase 4 K-reduced full-N execution (round 27): 27720-row run, verification, `experiments/analyze_phase4_expanded_full.py`, plan section 19 |
| `12e0870` | Smoke test of the new formal Phase 4 design (round 26) |
| `8b164dd` | Confirmed Phase 4 Expanded-K Confirmation Experiment; designed the K-reduced full-N Phase 4 (round 25) |
| `eddca0f` | Phase 4 3-policy Top-K preprocessing ablation, K up to 128 (round 24) |
| `1b6ece2` | Phase 4 reduced-tier root-cause analysis (round 23) |
| `714e51e` | Phase 4 reduced-tier execution, N=792/3168 (round 22) |
| `6d54159` | Phase 3 FULL execution, N=3960 (round 20) |
| `1e96b85` | Phase 1 Sensing Baseline, full 2200 combos (round 18) |
| `60bb22a` | Phase 0 pilot execution (round 17) |

All commits: author/committer `Liu Lina <ji3g4lina@gmail.com>` only, no AI
attribution. `main` is currently in sync with `origin/main` at `0cccc78`.

---

## 8. What can be cited now vs. what cannot yet be concluded

### Can be used in a meeting / paper draft now

- Sensing front end costs a small, real accuracy gap vs. an oracle slice:
  **+1.68 percentage points** (direct 0.5973 vs. sensed 0.5805), at
  `threshold_factor=1.5`/`max-energy`/`radioml-native`, N=2200.
- Adversarial attacks are highly effective against this pipeline without
  any defense: **82.78% overall success rate** at N=3960, ordering
  cw > pgd > fgsm.
- **Fixed-K Top-K FFT defense, applied globally (the only form directly
  deployable without an oracle), does not provide a statistically
  significant net accuracy benefit at any tested K, and is significantly
  harmful at K=10/40/50**, at full formal scale (N=3960 attack instances,
  27720 rows, WBFM included). This is the current formal, citable
  conclusion on Top-K defense deployability for this checkpoint.
- The pipeline (real AWN + real FGSM/PGD/CW + real Top-K, no dummy
  fallback) has been verified end-to-end, reproducible, and error-free at
  every formal scale tested (up to 27720 rows in a single run).

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
