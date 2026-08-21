"""
Step 4 analysis: reads results/<final_dir>/raw_results.csv (576 rows,
produced by experiments/run_satellite_like_final.py) and derives every
summary CSV and figure cited in docs/research/SATELLITE_LIKE_FINAL_EXPERIMENT_ZH_TW.md.
Does not recompute or re-derive any number by hand -- every value in every
summary/figure traces back to a groupby over raw_results.csv's own columns.

Writes: overall_summary.csv, by_channel.csv, by_modulation.csv, by_snr.csv,
by_attack.csv, by_channel_attack.csv, by_modulation_attack.csv,
by_snr_attack.csv, by_topk.csv, latency_summary.csv, processing_budget.csv,
failure_cases.csv, manifest.json, and 12 charts under charts/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

BUDGETS_MS = [5, 10, 20, 35, 50, 100, 250]


def to_bool(s):
    # pandas.read_csv auto-converts a column that is ENTIRELY "True"/"False"
    # strings to native numpy bool dtype (bug found this round: comparing a
    # native bool against string literals via `in` is always False, so the
    # old string-only version of this function silently mapped every real
    # True/False to None). Columns with missing values (e.g. defended_correct
    # when topk is off) stay object/string dtype with NaN mixed in, so both
    # native bool and string forms must be handled here.
    if isinstance(s, (bool, np.bool_)):
        return bool(s)
    if isinstance(s, float) and np.isnan(s):
        return None
    if s in ("True", "TRUE", "true"):
        return True
    if s in ("False", "FALSE", "false"):
        return False
    return None


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    bool_cols = ["clean_correct", "sensing_detected", "attacked_correct", "attack_success",
                 "defended_correct", "recovered_by_defense", "clean_degraded_by_defense", "fallback_used"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].map(to_bool)
    # "topk" is NaN when off, 20 when on -- pandas groupby() silently DROPS
    # NaN group keys by default (found this round: by_topk.csv had only 1
    # row instead of 2), so replace with an explicit, groupby-safe label
    # instead of leaving the raw NaN/20 value as the group key.
    df["topk_state"] = df["topk"].apply(lambda v: "on" if pd.notna(v) else "off")
    return df


def pct_stats(series: pd.Series) -> dict:
    s = series.dropna()
    n = len(s)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None, "p99": None}
    return {"n": n, "mean": float(s.mean()), "median": float(s.median()),
            "p95": float(np.percentile(s, 95)), "p99": float(np.percentile(s, 99))}


def rate(series: pd.Series) -> dict:
    s = series.dropna()
    n = len(s)
    if n == 0:
        return {"n": 0, "rate": None}
    return {"n": n, "rate": float(s.mean())}


def write_csv(path: Path, rows, fieldnames=None):
    if not rows:
        Path(path).write_text("")
        return
    if fieldnames is None:
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def overall_summary(df: pd.DataFrame) -> list:
    rows = []
    n_total = len(df)
    sens = rate(df["sensing_detected"])
    csr = pct_stats(df["captured_signal_ratio"])
    clean = df[df["attack_name"] == "none"]
    clean_acc = rate(clean["clean_correct"])
    rows.append({"metric": "n_total_rows", "value": n_total})
    rows.append({"metric": "sensing_detection_rate", "value": sens["rate"], "n": sens["n"]})
    rows.append({"metric": "captured_signal_ratio_mean", "value": csr["mean"], "n": csr["n"]})
    rows.append({"metric": "captured_signal_ratio_median", "value": csr["median"]})
    rows.append({"metric": "clean_accuracy_overall", "value": clean_acc["rate"], "n": clean_acc["n"]})
    for attack in ["fgsm", "pgd_det"]:
        sub = df[df["attack_name"] == attack]
        att_acc = rate(sub["attacked_correct"])
        asr = rate(sub["attack_success"])
        cond = sub[sub["clean_correct"] == True]  # noqa: E712
        cond_asr = rate(cond["attack_success"])
        linf = pct_stats(sub["perturbation_linf"])
        l2 = pct_stats(sub["perturbation_l2"])
        rows.append({"metric": f"{attack}_attacked_accuracy", "value": att_acc["rate"], "n": att_acc["n"]})
        rows.append({"metric": f"{attack}_overall_asr", "value": asr["rate"], "n": asr["n"]})
        rows.append({"metric": f"{attack}_conditional_asr", "value": cond_asr["rate"], "n": cond_asr["n"]})
        rows.append({"metric": f"{attack}_mean_linf", "value": linf["mean"]})
        rows.append({"metric": f"{attack}_mean_l2", "value": l2["mean"]})
    topk_on = df[df["topk"].notna()]
    def_acc = rate(topk_on["defended_correct"])
    recov = rate(topk_on["recovered_by_defense"])
    degrade = rate(topk_on["clean_degraded_by_defense"])
    rows.append({"metric": "defended_accuracy_overall", "value": def_acc["rate"], "n": def_acc["n"]})
    rows.append({"metric": "topk_recovery_rate", "value": recov["rate"], "n": recov["n"]})
    rows.append({"metric": "topk_clean_degradation_rate", "value": degrade["rate"], "n": degrade["n"]})
    return rows


def groupby_summary(df: pd.DataFrame, group_cols) -> list:
    rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n"] = len(sub)
        sens = rate(sub["sensing_detected"])
        row["sensing_detection_rate"] = sens["rate"]
        csr = pct_stats(sub["captured_signal_ratio"])
        row["mean_captured_signal_ratio"] = csr["mean"]
        row["median_captured_signal_ratio"] = csr["median"]
        clean_sub = sub[sub["attack_name"] == "none"]
        cacc = rate(clean_sub["clean_correct"])
        row["clean_accuracy"] = cacc["rate"]
        row["clean_accuracy_n"] = cacc["n"]
        for attack in ["fgsm", "pgd_det"]:
            a = sub[sub["attack_name"] == attack]
            if len(a) == 0:
                continue
            aacc = rate(a["attacked_correct"])
            asr = rate(a["attack_success"])
            row[f"{attack}_attacked_accuracy"] = aacc["rate"]
            row[f"{attack}_asr"] = asr["rate"]
        topk_sub = sub[sub["topk"].notna()]
        if len(topk_sub) > 0:
            dacc = rate(topk_sub["defended_correct"])
            row["defended_accuracy"] = dacc["rate"]
        rows.append(row)
    return rows


def latency_summary(df: pd.DataFrame) -> list:
    rows = []
    scenarios = [
        ("clean", df[(df["attack_name"] == "none") & (df["topk"].isna())]),
        ("fgsm", df[(df["attack_name"] == "fgsm") & (df["topk"].isna())]),
        ("pgd_det", df[(df["attack_name"] == "pgd_det") & (df["topk"].isna())]),
        ("fgsm_topk", df[(df["attack_name"] == "fgsm") & (df["topk"].notna())]),
        ("pgd_det_topk", df[(df["attack_name"] == "pgd_det") & (df["topk"].notna())]),
    ]
    for name, sub in scenarios:
        stats = pct_stats(sub["total_ms"])
        rows.append({"scenario": name, **stats})
    return rows


def processing_budget(df: pd.DataFrame) -> list:
    rows = []
    scenarios = [
        ("clean", df[(df["attack_name"] == "none") & (df["topk"].isna())]),
        ("fgsm", df[(df["attack_name"] == "fgsm") & (df["topk"].isna())]),
        ("pgd_det", df[(df["attack_name"] == "pgd_det") & (df["topk"].isna())]),
        ("fgsm_topk", df[(df["attack_name"] == "fgsm") & (df["topk"].notna())]),
        ("pgd_det_topk", df[(df["attack_name"] == "pgd_det") & (df["topk"].notna())]),
    ]
    for name, sub in scenarios:
        vals = sub["total_ms"].dropna()
        if len(vals) == 0:
            continue
        median_v, p95_v = float(vals.median()), float(np.percentile(vals, 95))
        row = {"scenario": name, "median_total_ms": median_v, "p95_total_ms": p95_v}
        for b in BUDGETS_MS:
            row[f"fits_{b}ms_median"] = median_v <= b
            row[f"fits_{b}ms_p95"] = p95_v <= b
        rows.append(row)
    return rows


def failure_cases(df: pd.DataFrame) -> list:
    fail = df[df["status"] != "ok"]
    return fail.to_dict("records")


def make_charts(df: pd.DataFrame, charts_dir: Path) -> None:
    charts_dir.mkdir(exist_ok=True)
    cond_order = ["clean", "mild", "moderate", "strong"]

    # 1. sensing performance by channel
    g = df.groupby("condition")["sensing_detected"].mean().reindex(cond_order)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(g.index, g.values, color="#4C72B0")
    ax.set_ylabel("sensing detection rate"); ax.set_title("Sensing Performance by Channel Condition")
    ax.set_ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(charts_dir / "01_sensing_by_channel.png", dpi=120); plt.close()

    # 2. AMC accuracy by channel (clean only)
    clean = df[df["attack_name"] == "none"]
    g = clean.groupby("condition")["clean_correct"].mean().reindex(cond_order)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(g.index, g.values, color="#55A868")
    ax.set_ylabel("clean accuracy"); ax.set_title("AMC Accuracy by Channel Condition")
    ax.set_ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(charts_dir / "02_amc_accuracy_by_channel.png", dpi=120); plt.close()

    # 3. AMC accuracy by modulation
    g = clean.groupby("modulation")["clean_correct"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(g.index, g.values, color="#C44E52")
    ax.set_ylabel("clean accuracy"); ax.set_title("AMC Accuracy by Modulation")
    ax.set_ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(charts_dir / "03_amc_accuracy_by_modulation.png", dpi=120); plt.close()

    # 4. AMC accuracy by SNR
    g = clean.groupby("snr_db_dataset")["clean_correct"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(g.index.astype(str), g.values, marker="o", color="#8172B2")
    ax.set_ylabel("clean accuracy"); ax.set_xlabel("SNR (dB)"); ax.set_title("AMC Accuracy by SNR")
    ax.set_ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(charts_dir / "04_amc_accuracy_by_snr.png", dpi=120); plt.close()

    # 5. attacked accuracy / ASR by attack
    fig, ax = plt.subplots(figsize=(7, 4))
    attacks = ["fgsm", "pgd_det"]
    acc_vals = [df[df["attack_name"] == a]["attacked_correct"].mean() for a in attacks]
    asr_vals = [df[df["attack_name"] == a]["attack_success"].mean() for a in attacks]
    x = np.arange(len(attacks)); w = 0.35
    ax.bar(x - w / 2, acc_vals, w, label="attacked accuracy", color="#4C72B0")
    ax.bar(x + w / 2, asr_vals, w, label="ASR (overall)", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(attacks); ax.legend()
    ax.set_title("Attacked Accuracy / ASR by Attack")
    plt.tight_layout(); plt.savefig(charts_dir / "05_attack_accuracy_asr.png", dpi=120); plt.close()

    # 6. channel x attack heatmap (ASR)
    pivot = df[df["attack_name"].isin(["fgsm", "pgd_det"])].pivot_table(
        index="condition", columns="attack_name", values="attack_success", aggfunc="mean"
    ).reindex(cond_order)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center")
    ax.set_title("Channel x Attack ASR Heatmap")
    plt.colorbar(im, ax=ax, label="ASR")
    plt.tight_layout(); plt.savefig(charts_dir / "06_channel_attack_heatmap.png", dpi=120); plt.close()

    # 7. Top-K recovery
    topk_on = df[df["topk"].notna() & df["attack_name"].isin(["fgsm", "pgd_det"])]
    g = topk_on.groupby("attack_name")["recovered_by_defense"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(g.index, g.values, color="#64B5CD")
    ax.set_ylabel("recovery rate"); ax.set_title("Top-K Recovery Rate by Attack")
    plt.tight_layout(); plt.savefig(charts_dir / "07_topk_recovery.png", dpi=120); plt.close()

    # 8. latency by scenario
    scen_labels = ["clean", "fgsm", "pgd_det", "fgsm_topk", "pgd_det_topk"]
    scen_dfs = [
        df[(df["attack_name"] == "none") & (df["topk"].isna())],
        df[(df["attack_name"] == "fgsm") & (df["topk"].isna())],
        df[(df["attack_name"] == "pgd_det") & (df["topk"].isna())],
        df[(df["attack_name"] == "fgsm") & (df["topk"].notna())],
        df[(df["attack_name"] == "pgd_det") & (df["topk"].notna())],
    ]
    means = [d["total_ms"].mean() for d in scen_dfs]
    medians = [d["total_ms"].median() for d in scen_dfs]
    p95s = [float(np.percentile(d["total_ms"].dropna(), 95)) for d in scen_dfs]
    x = np.arange(len(scen_labels)); w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, means, w, label="mean", color="#4C72B0")
    ax.bar(x, medians, w, label="median", color="#55A868")
    ax.bar(x + w, p95s, w, label="p95", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(scen_labels, rotation=15)
    ax.set_ylabel("total_ms"); ax.set_title("Latency by Scenario"); ax.legend()
    plt.tight_layout(); plt.savefig(charts_dir / "08_latency_by_scenario.png", dpi=120); plt.close()

    # 9. processing budget
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, sub in zip(scen_labels, scen_dfs):
        vals = sub["total_ms"].dropna()
        if len(vals) == 0:
            continue
        median_v = float(vals.median())
        fits = [1 if median_v <= b else 0 for b in BUDGETS_MS]
        ax.plot([str(b) for b in BUDGETS_MS], fits, marker="o", label=f"{name} (median)")
    ax.set_xlabel("budget (ms)"); ax.set_ylabel("fits"); ax.legend(fontsize=8)
    ax.set_title("Processing Budget Comparison")
    plt.tight_layout(); plt.savefig(charts_dir / "09_processing_budget.png", dpi=120); plt.close()

    # 10. captured signal ratio distribution by channel
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [df[df["condition"] == c]["captured_signal_ratio"].dropna().values for c in cond_order]
    ax.boxplot(data, labels=cond_order)
    ax.set_ylabel("captured_signal_ratio"); ax.set_title("Captured Signal Ratio by Channel Condition")
    plt.tight_layout(); plt.savefig(charts_dir / "10_captured_signal_ratio.png", dpi=120); plt.close()

    # 11. perturbation norm by attack x channel
    fig, ax = plt.subplots(figsize=(8, 5))
    for attack, color in [("fgsm", "#4C72B0"), ("pgd_det", "#C44E52")]:
        sub = df[df["attack_name"] == attack]
        g = sub.groupby("condition")["perturbation_linf"].mean().reindex(cond_order)
        ax.plot(cond_order, g.values, marker="o", label=attack, color=color)
    ax.set_ylabel("mean Linf"); ax.set_title("Perturbation Norm (Linf) by Attack x Channel"); ax.legend()
    plt.tight_layout(); plt.savefig(charts_dir / "11_perturbation_norm.png", dpi=120); plt.close()

    # 12. overall system pipeline summary
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["sensing\ndetect", "clean\naccuracy", "fgsm\nattacked acc", "pgd\nattacked acc", "topk\nrecovery"]
    vals = [
        df["sensing_detected"].mean(),
        clean["clean_correct"].mean(),
        df[df["attack_name"] == "fgsm"]["attacked_correct"].mean(),
        df[df["attack_name"] == "pgd_det"]["attacked_correct"].mean(),
        topk_on["recovered_by_defense"].mean(),
    ]
    ax.bar(labels, vals, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#64B5CD"])
    ax.set_ylabel("rate"); ax.set_title("Overall System Pipeline Summary")
    ax.set_ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(charts_dir / "12_pipeline_summary.png", dpi=120); plt.close()


def write_manifest(out_dir: Path, raw_path: Path) -> None:
    def git(cmd):
        return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True).strip()

    manifest = {
        "round": "satellite_like_final_experiment",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_base_commit": git(["git", "rev-parse", "HEAD"]),
        "git_working_tree_dirty_status": git(["git", "status", "--porcelain"]) or "(clean)",
        "raw_results_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "script": "experiments/run_satellite_like_final.py",
        "analysis_script": "experiments/analyze_satellite_like_final.py",
    }
    with open(out_dir / "manifest_analysis.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def main() -> None:
    out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    raw_path = out_dir / "raw_results.csv"
    df = load_raw(raw_path)
    print(f"[analyze] loaded {len(df)} rows from {raw_path}", flush=True)

    write_csv(out_dir / "overall_summary.csv", overall_summary(df))
    write_csv(out_dir / "by_channel.csv", groupby_summary(df, ["condition"]))
    write_csv(out_dir / "by_modulation.csv", groupby_summary(df, ["modulation"]))
    write_csv(out_dir / "by_snr.csv", groupby_summary(df, ["snr_db_dataset"]))
    write_csv(out_dir / "by_attack.csv", groupby_summary(df, ["attack_name"]))
    write_csv(out_dir / "by_channel_attack.csv", groupby_summary(df, ["condition", "attack_name"]))
    write_csv(out_dir / "by_modulation_attack.csv", groupby_summary(df, ["modulation", "attack_name"]))
    write_csv(out_dir / "by_snr_attack.csv", groupby_summary(df, ["snr_db_dataset", "attack_name"]))
    write_csv(out_dir / "by_topk.csv", groupby_summary(df, ["topk_state"]))
    write_csv(out_dir / "latency_summary.csv", latency_summary(df))
    write_csv(out_dir / "processing_budget.csv", processing_budget(df))
    write_csv(out_dir / "failure_cases.csv", failure_cases(df))
    print("[analyze] wrote all summary CSVs", flush=True)

    make_charts(df, out_dir / "charts")
    print("[analyze] wrote 12 charts", flush=True)

    write_manifest(out_dir, raw_path)
    print("[analyze] wrote manifest_analysis.json", flush=True)
    print("[analyze] DONE", flush=True)


if __name__ == "__main__":
    main()
