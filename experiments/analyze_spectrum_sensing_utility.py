"""
Aggregate analysis for the four-path (direct / no_sensing / sensing /
oracle) Spectrum Sensing Utility Experiment
(experiments/run_spectrum_sensing_utility.py). Read-only: consumes
raw_results.csv + logits.npz from a completed formal run and produces the
required aggregate CSVs, paired-comparison statistics (with exact McNemar
tests), a failure-case catalogue, and 8 plots (PNG + CSV source each).

Does not modify src/, external/AWN, or external/adversarial-rf.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

PATHS = ["direct", "no_sensing", "sensing", "oracle"]
PAIRS = [("direct", "oracle"), ("no_sensing", "sensing"), ("sensing", "oracle")]


def path_stats(df: pd.DataFrame, path: str) -> dict:
    correct_col = f"{path}_correct"
    conf_col = f"{path}_confidence"
    rt_col = f"{path}_runtime_ms"
    n = len(df)
    correct = int(df[correct_col].sum())
    return {
        "path": path,
        "N": n,
        "correct_count": correct,
        "accuracy": correct / n if n else float("nan"),
        "mean_confidence": float(df[conf_col].mean()) if n else float("nan"),
        "mean_runtime_ms": float(df[rt_col].mean()) if n else float("nan"),
        "median_runtime_ms": float(df[rt_col].median()) if n else float("nan"),
        "p95_runtime_ms": float(df[rt_col].quantile(0.95)) if n else float("nan"),
    }


def build_overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([path_stats(df, p) for p in PATHS])


def build_grouped(df: pd.DataFrame, group_cols) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        for p in PATHS:
            row = dict(zip(group_cols, key))
            row.update(path_stats(g, p))
            rows.append(row)
    return pd.DataFrame(rows)


def mcnemar_exact(b: int, c: int) -> tuple:
    """Exact two-sided McNemar test on discordant pairs (b, c). Returns
    (chi2_statistic_no_continuity_correction, exact_p_value)."""
    n = b + c
    if n == 0:
        return float("nan"), float("nan")
    stat = (b - c) ** 2 / n
    p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    return stat, p


def build_paired_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for first, second in PAIRS:
        fc, sc = df[f"{first}_correct"], df[f"{second}_correct"]
        fp, sp = df[f"{first}_prediction"], df[f"{second}_prediction"]
        both_correct = int((fc & sc).sum())
        first_correct_second_wrong = int((fc & ~sc).sum())
        first_wrong_second_correct = int((~fc & sc).sum())
        both_wrong = int((~fc & ~sc).sum())
        stat, pval = mcnemar_exact(first_correct_second_wrong, first_wrong_second_correct)
        rows.append({
            "comparison": f"{first}_vs_{second}",
            "n": len(df),
            "both_correct": both_correct,
            "first_correct_second_wrong": first_correct_second_wrong,
            "first_wrong_second_correct": first_wrong_second_correct,
            "both_wrong": both_wrong,
            "first_accuracy": float(fc.mean()),
            "second_accuracy": float(sc.mean()),
            "accuracy_difference": float(sc.mean() - fc.mean()),
            "prediction_agreement": float((fp == sp).mean()),
            "mcnemar_statistic": stat,
            "mcnemar_exact_pvalue": pval,
        })
    return pd.DataFrame(rows)


def build_failure_cases(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["experiment_id", "modulation", "snr_db", "sample_index", "label_index",
                "direct_prediction", "direct_correct", "no_sensing_prediction", "no_sensing_correct",
                "sensing_prediction", "sensing_correct", "oracle_prediction", "oracle_correct",
                "sensing_detected", "captured_signal_ratio", "start_boundary_error", "end_boundary_error"]
    cases = []

    def add(mask: pd.Series, case_type: str) -> None:
        sub = df.loc[mask, key_cols].copy()
        sub.insert(0, "case_type", case_type)
        cases.append(sub)

    add(~df.no_sensing_correct & df.sensing_correct, "no_sensing_wrong_sensing_correct")
    add(df.no_sensing_correct & ~df.sensing_correct, "no_sensing_correct_sensing_wrong")
    add(~df.sensing_correct & df.oracle_correct, "sensing_wrong_oracle_correct")
    add(df.direct_correct & ~df.oracle_correct, "direct_correct_oracle_wrong")
    add(~df.sensing_detected, "sensing_failure")

    csr = df.captured_signal_ratio
    add(csr.notna() & ((csr < 0.5) | (csr > 1.0)), "captured_ratio_anomaly")

    be_thresh = 200  # samples; energy_detect's smoothing window is 128, so
    # a boundary error several times the window size is a clear outlier,
    # not a routine few-sample edge effect (docs/formal_experiment_plan.md
    # section 9.3's own ~59-sample typical boundary error is the baseline
    # this threshold is deliberately well above).
    sbe, ebe = df.start_boundary_error, df.end_boundary_error
    add(sbe.notna() & (sbe.abs() > be_thresh), "start_boundary_error_anomaly")
    add(ebe.notna() & (ebe.abs() > be_thresh), "end_boundary_error_anomaly")

    if not cases:
        return pd.DataFrame(columns=["case_type"] + key_cols)
    return pd.concat(cases, ignore_index=True)


def save_plot_with_source(fig, source_df: pd.DataFrame, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    source_df.to_csv(out_dir / f"{name}_source.csv", index=False)


def make_plots(df: pd.DataFrame, overall: pd.DataFrame, by_snr: pd.DataFrame,
                by_mod: pd.DataFrame, out_dir: Path) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    colors = {"direct": "tab:blue", "no_sensing": "tab:orange", "sensing": "tab:green", "oracle": "tab:red"}

    # 1. Overall accuracy, 4-path comparison.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(overall["path"], overall["accuracy"], color=[colors[p] for p in overall["path"]])
    ax.set_ylabel("Accuracy")
    ax.set_title("Overall accuracy by path (N=2200 each)")
    ax.set_ylim(0, 1)
    for i, v in enumerate(overall["accuracy"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    save_plot_with_source(fig, overall, plots_dir, "01_overall_accuracy")

    # 2. Accuracy by SNR, 4-path curves.
    fig, ax = plt.subplots(figsize=(8, 5))
    for p in PATHS:
        sub = by_snr[by_snr.path == p].sort_values("snr_db")
        ax.plot(sub.snr_db, sub.accuracy, marker="o", label=p, color=colors[p])
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Accuracy"); ax.legend(); ax.set_title("Accuracy by SNR")
    save_plot_with_source(fig, by_snr, plots_dir, "02_accuracy_by_snr")

    # 3. Accuracy delta by SNR: sensing-no_sensing, oracle-sensing, direct-oracle.
    pivot = by_snr.pivot(index="snr_db", columns="path", values="accuracy").reset_index()
    pivot["sensing_minus_no_sensing"] = pivot["sensing"] - pivot["no_sensing"]
    pivot["oracle_minus_sensing"] = pivot["oracle"] - pivot["sensing"]
    pivot["direct_minus_oracle"] = pivot["direct"] - pivot["oracle"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, lbl in [("sensing_minus_no_sensing", "sensing - no_sensing"),
                      ("oracle_minus_sensing", "oracle - sensing"),
                      ("direct_minus_oracle", "direct - oracle")]:
        ax.plot(pivot.snr_db, pivot[col], marker="o", label=lbl)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Accuracy delta"); ax.legend(); ax.set_title("Accuracy delta by SNR")
    save_plot_with_source(fig, pivot, plots_dir, "03_accuracy_delta_by_snr")

    # 4. Accuracy by modulation, 4-path grouped bars.
    mods = sorted(df.modulation.unique())
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.2
    x = np.arange(len(mods))
    for i, p in enumerate(PATHS):
        sub = by_mod[by_mod.path == p].set_index("modulation").reindex(mods)
        ax.bar(x + (i - 1.5) * width, sub.accuracy, width=width, label=p, color=colors[p])
    ax.set_xticks(x); ax.set_xticklabels(mods, rotation=45, ha="right")
    ax.set_ylabel("Accuracy"); ax.legend(); ax.set_title("Accuracy by modulation")
    save_plot_with_source(fig, by_mod, plots_dir, "04_accuracy_by_modulation")

    # 5. Sensing gain over no_sensing, by modulation.
    piv_mod = by_mod.pivot(index="modulation", columns="path", values="accuracy").reindex(mods)
    gain = (piv_mod["sensing"] - piv_mod["no_sensing"]).reset_index()
    gain.columns = ["modulation", "sensing_gain_over_no_sensing"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(gain.modulation, gain.sensing_gain_over_no_sensing, color="tab:green")
    ax.axhline(0, color="gray", linewidth=0.8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Accuracy(sensing) - Accuracy(no_sensing)")
    ax.set_title("Sensing gain over no_sensing baseline, by modulation")
    save_plot_with_source(fig, gain, plots_dir, "05_sensing_gain_over_no_sensing_by_modulation")

    # 6. Sensing-to-oracle gap, by modulation.
    gap = (piv_mod["oracle"] - piv_mod["sensing"]).reset_index()
    gap.columns = ["modulation", "oracle_minus_sensing_gap"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(gap.modulation, gap.oracle_minus_sensing_gap, color="tab:red")
    ax.axhline(0, color="gray", linewidth=0.8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Accuracy(oracle) - Accuracy(sensing)")
    ax.set_title("Sensing-to-oracle accuracy gap, by modulation")
    save_plot_with_source(fig, gap, plots_dir, "06_sensing_to_oracle_gap_by_modulation")

    # 7. Boundary error distribution.
    be_df = df[["start_boundary_error", "end_boundary_error"]].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(be_df.start_boundary_error, bins=40, alpha=0.6, label="start_boundary_error")
    ax.hist(be_df.end_boundary_error, bins=40, alpha=0.6, label="end_boundary_error")
    ax.set_xlabel("Boundary error (samples)"); ax.set_ylabel("Count"); ax.legend()
    ax.set_title("Sensing boundary error distribution (N=2200)")
    save_plot_with_source(fig, be_df, plots_dir, "07_boundary_error_distribution")

    # 8. Runtime, 4-path comparison.
    rt_cols = [f"{p}_runtime_ms" for p in PATHS]
    rt_long = df[rt_cols].melt(var_name="path_runtime_col", value_name="runtime_ms")
    rt_long["path"] = rt_long.path_runtime_col.str.replace("_runtime_ms", "", regex=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([df[f"{p}_runtime_ms"] for p in PATHS], tick_labels=PATHS, showfliers=False)
    ax.set_ylabel("Runtime (ms)"); ax.set_title("Per-path runtime comparison (outliers hidden)")
    save_plot_with_source(fig, rt_long, plots_dir, "08_runtime_comparison")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=str, required=True)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    df = pd.read_csv(in_dir / "raw_results.csv")
    print(f"[analyze] loaded {len(df)} rows from {in_dir / 'raw_results.csv'}")

    overall = build_overall_summary(df)
    overall.to_csv(in_dir / "overall_summary.csv", index=False)
    print("[analyze] overall_summary.csv written")

    by_snr = build_grouped(df, ["snr_db"])
    by_snr.to_csv(in_dir / "by_snr.csv", index=False)
    print(f"[analyze] by_snr.csv written ({len(by_snr)} rows)")

    by_mod = build_grouped(df, ["modulation"])
    by_mod.to_csv(in_dir / "by_modulation.csv", index=False)
    print(f"[analyze] by_modulation.csv written ({len(by_mod)} rows)")

    by_mod_snr = build_grouped(df, ["modulation", "snr_db"])
    by_mod_snr.to_csv(in_dir / "by_modulation_snr.csv", index=False)
    print(f"[analyze] by_modulation_snr.csv written ({len(by_mod_snr)} rows)")

    paired = build_paired_comparisons(df)
    paired.to_csv(in_dir / "paired_comparisons.csv", index=False)
    print("[analyze] paired_comparisons.csv written")
    print(paired.to_string(index=False))

    failures = build_failure_cases(df)
    failures.to_csv(in_dir / "failure_cases.csv", index=False)
    print(f"[analyze] failure_cases.csv written ({len(failures)} rows)")

    make_plots(df, overall, by_snr, by_mod, in_dir)
    print(f"[analyze] 8 plots (PNG + source CSV) written to {in_dir / 'plots'}")

    print("[analyze] DONE")


if __name__ == "__main__":
    main()
