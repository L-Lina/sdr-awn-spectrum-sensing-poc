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
