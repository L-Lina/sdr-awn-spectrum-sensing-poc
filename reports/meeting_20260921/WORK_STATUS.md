# WORK STATUS — meeting_20260921 report package

Updated: 2026-09-21 (continuation after an interrupted session; nothing committed or pushed).

## 1. Progress summary

| Phase | Status |
|---|---|
| Phase A — Formal Data Validation | **Completed. FORMAL DATA VALIDATION: PASS** (113 checks) |
| Phase B — Formal analysis (B1–B9) | **Completed.** `analysis/phase_b_analysis.py` re-run end to end; 15 tables in `tables/`, headline numbers in `analysis/phase_b_results.json` |
| Phase C — Report package | **Completed.** 8 figures, `meeting_report.md`, `executive_summary.md` |

## 2. Formal result sources (read-only, unmodified)

- NPU: `results/hailo_full_matrix_20260920_seedaligned`
- CPU: `results/cpu_full_matrix_20260921_seedaligned`

SHA-256 of every CSV in both directories was recorded before and after the continuation run: no change. The full matrix was not re-run.

## 3. Continuation notes

- The previous version of this file described Phase B and Phase C as not started. Files on disk showed Phase B tables and figures had in fact been produced before the interruption.
- `analysis/make_figures.py` had a syntax error (unescaped double quotes inside a string in the Figure 5 footnote). It was fixed, and all eight figures were regenerated from the current tables. Figure 8 panel (a) title was shortened; the Figure 6 footnote range was corrected to 10–12%.
- `analysis/phase_b_analysis.py` needed no change.

## 3a. Attack-scope correction (2026-10-22)

Pooled aggregates were restricted to the main attack scope FGSM / PGD / CW / EAD (9 profiles). Earlier versions pooled all 24 profiles (17 attacks).

- `tables/attack_by_snr.csv`, `attack_by_modulation.csv`, `defense_by_snr.csv`, `defense_by_modulation.csv`: the pooled row is now `MAIN4/FGSM_PGD_CW_EAD` (was `ALL/ALL_24`). Per-profile rows are unchanged.
- `tables/defense_pooled_by_topk.csv`: `all_profiles = FGSM_PGD_CW_EAD`.
- `analysis/phase_b_results.json`: `pooled_asr_main4_pct`, `pooled_defense` and `recovered_share_qam64_pam4_amdsb_pct` use the main scope; `amdsb_identical_inputs` is marked as not recomputed.
- `analysis/phase_b_analysis.py` pools only `MAIN_ATTACKS`; `analysis/make_figures.py` plots only the main-scope profiles (Figures 2–8).
- `results/` is not on this machine, so the pooled rows were recomputed from the per-profile rows in `tables/` by `scripts/correct_historical_attack_scope.py`; the script checks that each old pooled row equals the sum of its per-profile counts before replacing it.
- Not recomputed (row-level source not in the repository): pooled attacked / defended inference latency and attack-generation latency median / p95 / p99 in `tables/latency_summary_cpu_npu.csv` (24-profile scope, marked in the report), and the AM-DSB row-level prediction analysis.

## 4. Files

| Path | Content |
|---|---|
| `meeting_report.md` | Full report (Traditional Chinese) |
| `executive_summary.md` | One-page summary |
| `analysis/phase_a_validation.{py,md,json}` | Data validation |
| `analysis/phase_b_analysis.py`, `phase_b_results.json` | Derived tables and headline numbers |
| `analysis/make_figures.py` | Figures 1–8 |
| `figures/` | fig1–fig8 (PNG) |
| `tables/` | Derived CSV tables |

## 5. Reproduce

```bash
python reports/meeting_20260921/analysis/phase_a_validation.py
python reports/meeting_20260921/analysis/phase_b_analysis.py
python reports/meeting_20260921/analysis/make_figures.py
```

Requires `pandas`, `numpy`, `scipy`, `matplotlib` (not present in the project `.venv`; use a separate environment).

## 6. Before committing

- `results/` is git-ignored; `reports/` is untracked. `data/RML2016.10a_dict.pkl` is large and untracked — do not `git add` it.
- Other untracked and modified files in the repository (`src/`, `experiments/`, `configs/`, `deployment/`, logs) pre-date this work.
