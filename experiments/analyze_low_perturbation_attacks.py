"""
Post-processing / analysis for experiments/run_low_perturbation_attacks.py's
raw output (fgsm_raw_results.csv + low_perturbation_raw_results.csv).

Produces the summary CSVs and charts required for the FGSM + low-perturbation
attack meeting deliverable: by_attack / by_attack_eps / by_attack_snr /
by_attack_modulation / by_attack_modulation_snr / latency_summary /
perturbation_summary / failure_cases, plus 12 PNG charts (each with its own
source CSV).

Conditional attack success rate = among samples where clean_prediction was
CORRECT, the fraction where the attack flipped the prediction to something
else -- NOT overall ASR (which would count originally-misclassified samples
as "attack successes" even though the attack changed nothing about the
correct/incorrect status).

Reads only; writes only into the given --output-dir. Does not touch
external/AWN, external/adversarial-rf, or any other results/ directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 9})


def conditional_asr(df: pd.DataFrame) -> float:
    correct = df[df["clean_correct"] == True]  # noqa: E712
    if len(correct) == 0:
        return float("nan")
    return float((correct["attack_success"] == True).mean())  # noqa: E712


FORMAL_EFFECTIVENESS_N = {"fgsm": 1320, "pgd": 495, "cw": 165, "deepfool": 165, "ead": 165, "fab": 495}
RECALIBRATED_ATTACKS = {"cw", "deepfool"}
RECALIBRATION_STAGES = ("awn_clean_ms", "attack_generation_ms", "awn_attacked_ms", "total_ms")
FORMAL_TIMING_NOTE = "formal run; timing not affected by CPU contention."
RECALIB_TIMING_NOTE_TMPL = (
    "STANDALONE RECALIBRATION (n={n}) -- the original formal run for this attack (effectiveness_n={eff_n}) "
    "overlapped in wall-clock time with a concurrently-running FAB process, causing severe CPU contention "
    "between two multi-threaded torch processes (15-75x slowdown observed). Effectiveness metrics "
    "(accuracy, ASR, perturbation norms) from the effectiveness_n={eff_n} formal run are UNAFFECTED by CPU "
    "speed and remain in use. Only this latency figure was replaced with a standalone, uncontended "
    "re-measurement recorded in cw_deepfool_latency_calibration_raw.csv. The original contended per-row "
    "timing is preserved unmodified in {attack}_raw_results.csv for audit."
)


def apply_latency_recalibration(out: Path) -> None:
    """Patches attack_summary.csv / by_attack.csv / by_attack_eps.csv / latency_summary.csv
    with effectiveness_n / latency_n / latency_source / timing_note columns, and replaces
    the cw/deepfool timing-only figures (awn_clean_ms, attack_generation_ms, awn_attacked_ms,
    total_ms) with a standalone recalibration when cw_deepfool_latency_calibration_raw.csv is
    present in the results directory. Never modifies cw_raw_results.csv, deepfool_raw_results.csv,
    or any effectiveness column (accuracy, ASR, perturbation norms). No-op if the calibration
    file is absent (e.g. a run with no CPU contention has nothing to recalibrate)."""
    calib_path = out / "cw_deepfool_latency_calibration_raw.csv"
    calib = pd.read_csv(calib_path) if calib_path.exists() else None

    fgsm_path = out / "fgsm_summary.csv"
    if fgsm_path.exists():
        df = pd.read_csv(fgsm_path)
        df.insert(df.columns.get_loc("n") + 1, "effectiveness_n", df["n"])
        df["latency_n"] = df["effectiveness_n"]
        df["latency_source"] = "formal_run"
        df["timing_note"] = FORMAL_TIMING_NOTE
        df.to_csv(fgsm_path, index=False)
        print("[analyze] fgsm_summary.csv: effectiveness_n/latency_n/latency_source/timing_note applied")

    for fn in ["attack_summary.csv", "by_attack.csv", "by_attack_eps.csv"]:
        p = out / fn
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df.insert(df.columns.get_loc("n") + 1, "effectiveness_n", df["n"])
        lat_n, lat_src, note = [], [], []
        for a in df["attack_name"]:
            if calib is not None and a in RECALIBRATED_ATTACKS:
                sub = calib[calib["attack_name"] == a]
                lat_n.append(len(sub))
                lat_src.append("standalone_recalibration")
                note.append(RECALIB_TIMING_NOTE_TMPL.format(n=len(sub), eff_n=FORMAL_EFFECTIVENESS_N[a], attack=a))
            else:
                lat_n.append(FORMAL_EFFECTIVENESS_N.get(a))
                lat_src.append("formal_run")
                note.append(FORMAL_TIMING_NOTE)
        df["latency_n"] = lat_n
        df["latency_source"] = lat_src
        df["timing_note"] = note
        if calib is not None:
            for a in RECALIBRATED_ATTACKS:
                sub = calib[calib["attack_name"] == a]
                mask = df["attack_name"] == a
                df.loc[mask, "mean_total_ms"] = sub["total_ms"].mean()
                df.loc[mask, "median_total_ms"] = sub["total_ms"].median()
                df.loc[mask, "p95_total_ms"] = sub["total_ms"].quantile(0.95)
                df.loc[mask, "max_total_ms"] = sub["total_ms"].max()
                df.loc[mask, "mean_attack_generation_ms"] = sub["attack_generation_ms"].mean()
                df.loc[mask, "median_attack_generation_ms"] = sub["attack_generation_ms"].median()
                df.loc[mask, "p95_attack_generation_ms"] = sub["attack_generation_ms"].quantile(0.95)
                df.loc[mask, "samples_per_sec"] = 1000.0 / sub["total_ms"].mean()
        df.to_csv(p, index=False)
        print(f"[analyze] {fn}: effectiveness_n/latency_n/latency_source/timing_note applied")

    lat_path = out / "latency_summary.csv"
    if lat_path.exists():
        df = pd.read_csv(lat_path)
        df.insert(1, "effectiveness_n", df["attack_name"].map(FORMAL_EFFECTIVENESS_N))
        lat_n, lat_src, note = [], [], []
        for _, row in df.iterrows():
            a, stage = row["attack_name"], row["stage"]
            recalibrate = calib is not None and a in RECALIBRATED_ATTACKS and stage in RECALIBRATION_STAGES
            if recalibrate:
                sub = calib[calib["attack_name"] == a]
                lat_n.append(len(sub))
                lat_src.append("standalone_recalibration")
                note.append(RECALIB_TIMING_NOTE_TMPL.format(n=len(sub), eff_n=FORMAL_EFFECTIVENESS_N[a], attack=a))
            else:
                lat_n.append(FORMAL_EFFECTIVENESS_N.get(a))
                lat_src.append("formal_run")
                note.append(FORMAL_TIMING_NOTE)
        df["latency_n"] = lat_n
        df["latency_source"] = lat_src
        df["timing_note"] = note
        if calib is not None:
            for a in RECALIBRATED_ATTACKS:
                sub = calib[calib["attack_name"] == a]
                for stage in RECALIBRATION_STAGES:
                    mask = (df["attack_name"] == a) & (df["stage"] == stage)
                    df.loc[mask, "mean_ms"] = sub[stage].mean()
                    df.loc[mask, "median_ms"] = sub[stage].median()
                    df.loc[mask, "p95_ms"] = sub[stage].quantile(0.95)
                    df.loc[mask, "max_ms"] = sub[stage].max()
                    df.loc[mask, "samples_per_sec"] = 1000.0 / sub[stage].mean()
        df.to_csv(lat_path, index=False)
        print("[analyze] latency_summary.csv: effectiveness_n/latency_n/latency_source/timing_note applied")

    charts = out / "charts"
    c9 = charts / "09_attack_latency_comparison.csv"
    if c9.exists():
        summary = pd.read_csv(out / "attack_summary.csv")
        df = pd.read_csv(c9)
        df = df.merge(
            summary[["attack_name", "mean_total_ms", "median_total_ms", "max_total_ms", "latency_source", "latency_n"]],
            on="attack_name", how="left",
        )
        df["mean"] = df["mean_total_ms"]
        df["median"] = df["median_total_ms"]
        df["max"] = df["max_total_ms"]
        df = df[["attack_name", "mean", "median", "max", "latency_source", "latency_n"]]
        df.to_csv(c9, index=False)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(df["attack_name"], df["mean"])
        ax.set_ylabel("mean total_ms/instance (recalibrated where noted)")
        ax.set_title("Attack latency comparison")
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(charts / "09_attack_latency_comparison.png")
        plt.close(fig)
        print("[analyze] charts/09_attack_latency_comparison.{csv,png}: recalibrated")

    if calib is not None:
        stage_cols = ["sensing_ms", "awn_clean_ms", "attack_generation_ms", "awn_attacked_ms"]
        non_recalibrated = pd.concat(
            [pd.read_csv(out / f"{a}_raw_results.csv") for a in FORMAL_EFFECTIVENESS_N if a not in RECALIBRATED_ATTACKS],
            ignore_index=True,
        )
        non_recalibrated = non_recalibrated[non_recalibrated["status"] == "ok"][stage_cols]
        recalibrated = calib[stage_cols]
        stage_means = pd.concat([non_recalibrated, recalibrated], ignore_index=True).mean()
        c10 = charts / "10_stage_latency_breakdown.csv"
        with open(c10, "w") as f:
            f.write("# mean ms across all 6 attacks; cw/deepfool contribution uses standalone_recalibration, others use formal_run\n")
            stage_means.to_csv(f)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(stage_means.index, stage_means.values)
        ax.set_ylabel("mean ms (across all attacks, recalibrated where noted)")
        ax.set_title("Spectrum Sensing vs AWN vs Attack latency")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(charts / "10_stage_latency_breakdown.png")
        plt.close(fig)
        print("[analyze] charts/10_stage_latency_breakdown.{csv,png}: recalibrated")


def summarize_group(df: pd.DataFrame) -> dict:
    ok = df[df["status"] == "ok"]
    return {
        "n": len(df),
        "n_ok": len(ok),
        "n_error": int((df["status"] == "error").sum()),
        "n_nan_inf": int((df["status"] == "nan_inf").sum()),
        "n_fallback": int(df["fallback_used"].fillna(False).astype(bool).sum()),
        "clean_accuracy": float(ok["clean_correct"].mean()) if len(ok) else float("nan"),
        "attacked_accuracy": float(ok["attacked_correct"].mean()) if len(ok) else float("nan"),
        "overall_attack_success_rate": float(ok["attack_success"].mean()) if len(ok) else float("nan"),
        "conditional_attack_success_rate": conditional_asr(ok) if len(ok) else float("nan"),
        "prediction_change_rate": float(ok["attack_success"].mean()) if len(ok) else float("nan"),
        "mean_confidence_drop": float((ok["clean_confidence"] - ok["attacked_confidence"]).mean()) if len(ok) else float("nan"),
        "mean_linf": float(ok["perturbation_linf"].mean()) if len(ok) else float("nan"),
        "median_linf": float(ok["perturbation_linf"].median()) if len(ok) else float("nan"),
        "p95_linf": float(ok["perturbation_linf"].quantile(0.95)) if len(ok) else float("nan"),
        "mean_l2": float(ok["perturbation_l2"].mean()) if len(ok) else float("nan"),
        "median_l2": float(ok["perturbation_l2"].median()) if len(ok) else float("nan"),
        "p95_l2": float(ok["perturbation_l2"].quantile(0.95)) if len(ok) else float("nan"),
        "mean_l1": float(ok["perturbation_l1"].mean()) if ok["perturbation_l1"].notna().any() else None,
        "mean_total_ms": float(ok["total_ms"].mean()) if len(ok) else float("nan"),
        "median_total_ms": float(ok["total_ms"].median()) if len(ok) else float("nan"),
        "p95_total_ms": float(ok["total_ms"].quantile(0.95)) if len(ok) else float("nan"),
        "max_total_ms": float(ok["total_ms"].max()) if len(ok) else float("nan"),
        "mean_attack_generation_ms": float(ok["attack_generation_ms"].mean()) if len(ok) else float("nan"),
        "median_attack_generation_ms": float(ok["attack_generation_ms"].median()) if len(ok) else float("nan"),
        "p95_attack_generation_ms": float(ok["attack_generation_ms"].quantile(0.95)) if len(ok) else float("nan"),
        "samples_per_sec": float(1000.0 / ok["total_ms"].mean()) if len(ok) and ok["total_ms"].mean() > 0 else float("nan"),
        "sensing_detection_rate": float(ok["sensing_detected"].mean()) if len(ok) else float("nan"),
        "mean_captured_signal_ratio": float(ok["captured_signal_ratio"].mean()) if len(ok) else float("nan"),
        "model_mode_after_all_eval": bool((ok["model_mode_after"] == "eval").all()) if len(ok) else None,
        "clean_logits_reproducible_all": bool(ok["clean_prediction_reproducible"].all()) if len(ok) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)

    fgsm = pd.read_csv(out / "fgsm_raw_results.csv")
    lowp = pd.read_csv(out / "low_perturbation_raw_results.csv")
    all_df = pd.concat([fgsm, lowp], ignore_index=True)
    for c in ["clean_correct", "attacked_correct", "attack_success", "sensing_detected",
              "fallback_used", "clean_prediction_reproducible"]:
        if c in all_df.columns:
            all_df[c] = all_df[c].astype("boolean")

    print(f"[analyze] fgsm rows={len(fgsm)} low_perturbation rows={len(lowp)} total={len(all_df)}")

    # --- consistency check: same (mod,snr,sample_index) must share clean_segment_input_hash
    # across all low-perturbation attacks (item 4-5-6 of the request) ---
    hash_check = lowp.groupby(["modulation", "snr", "sample_index"])["clean_segment_input_hash"].nunique()
    inconsistent = hash_check[hash_check > 1]
    print(f"[analyze] clean-segment-hash consistency across low-perturbation attacks: "
          f"{len(inconsistent)} inconsistent (mod,snr,sample_index) groups out of {len(hash_check)}")

    # --- failure cases ---
    failures = all_df[(all_df["status"] != "ok")]
    failures.to_csv(out / "failure_cases.csv", index=False)
    print(f"[analyze] failure_cases.csv: {len(failures)} rows")

    # --- fgsm_summary.csv (by eps) ---
    fgsm_rows = []
    for eps, g in fgsm.groupby("eps"):
        row = {"eps": eps}
        row.update(summarize_group(g))
        fgsm_rows.append(row)
    pd.DataFrame(fgsm_rows).to_csv(out / "fgsm_summary.csv", index=False)

    # --- attack_summary.csv / by_attack.csv (one row per attack, across all eps/snr/mod) ---
    by_attack_rows = []
    for attack, g in all_df.groupby("attack_name"):
        row = {"attack_name": attack}
        row.update(summarize_group(g))
        by_attack_rows.append(row)
    by_attack_df = pd.DataFrame(by_attack_rows)
    by_attack_df.to_csv(out / "attack_summary.csv", index=False)
    by_attack_df.to_csv(out / "by_attack.csv", index=False)

    # --- by_attack_eps.csv ---
    rows = []
    for (attack, eps), g in all_df.groupby(["attack_name", "eps"], dropna=False):
        row = {"attack_name": attack, "eps": eps}
        row.update(summarize_group(g))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "by_attack_eps.csv", index=False)

    # --- by_attack_snr.csv ---
    rows = []
    for (attack, snr), g in all_df.groupby(["attack_name", "snr"]):
        row = {"attack_name": attack, "snr": snr}
        row.update(summarize_group(g))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "by_attack_snr.csv", index=False)

    # --- by_attack_modulation.csv ---
    rows = []
    for (attack, mod), g in all_df.groupby(["attack_name", "modulation"]):
        row = {"attack_name": attack, "modulation": mod}
        row.update(summarize_group(g))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "by_attack_modulation.csv", index=False)

    # --- by_attack_modulation_snr.csv ---
    rows = []
    for (attack, mod, snr), g in all_df.groupby(["attack_name", "modulation", "snr"]):
        row = {"attack_name": attack, "modulation": mod, "snr": snr}
        row.update(summarize_group(g))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "by_attack_modulation_snr.csv", index=False)

    # --- latency_summary.csv (per attack, plus sensing/clean-inference stage breakdown) ---
    stage_cols = ["sensing_ms", "region_postprocess_ms", "segmentation_ms", "awn_clean_ms",
                  "attack_generation_ms", "awn_attacked_ms", "total_ms"]
    rows = []
    for attack, g in all_df.groupby("attack_name"):
        ok = g[g["status"] == "ok"]
        for col in stage_cols:
            vals = ok[col].dropna()
            rows.append({
                "attack_name": attack, "stage": col, "n": len(vals),
                "mean_ms": float(vals.mean()) if len(vals) else None,
                "median_ms": float(vals.median()) if len(vals) else None,
                "p95_ms": float(vals.quantile(0.95)) if len(vals) else None,
                "max_ms": float(vals.max()) if len(vals) else None,
                "samples_per_sec": float(1000.0 / vals.mean()) if len(vals) and vals.mean() > 0 else None,
            })
    latency_df = pd.DataFrame(rows)
    latency_df.to_csv(out / "latency_summary.csv", index=False)

    # --- perturbation_summary.csv ---
    rows = []
    for attack, g in all_df.groupby("attack_name"):
        ok = g[g["status"] == "ok"]
        rows.append({
            "attack_name": attack,
            "mean_linf": float(ok["perturbation_linf"].mean()) if len(ok) else None,
            "median_linf": float(ok["perturbation_linf"].median()) if len(ok) else None,
            "p95_linf": float(ok["perturbation_linf"].quantile(0.95)) if len(ok) else None,
            "mean_l2": float(ok["perturbation_l2"].mean()) if len(ok) else None,
            "median_l2": float(ok["perturbation_l2"].median()) if len(ok) else None,
            "p95_l2": float(ok["perturbation_l2"].quantile(0.95)) if len(ok) else None,
            "mean_l1": float(ok["perturbation_l1"].mean()) if ok["perturbation_l1"].notna().any() else None,
        })
    pd.DataFrame(rows).to_csv(out / "perturbation_summary.csv", index=False)

    # ============== CHARTS ==============
    charts_dir = out / "charts"
    charts_dir.mkdir(exist_ok=True)
    ok_all = all_df[all_df["status"] == "ok"].copy()
    fgsm_ok = fgsm[fgsm["status"] == "ok"].copy()

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(charts_dir / f"{name}.png")
        plt.close(fig)

    # 1. Clean vs FGSM accuracy by eps
    g = fgsm_ok.groupby("eps").apply(lambda d: pd.Series({
        "clean_accuracy": d["clean_correct"].mean(), "attacked_accuracy": d["attacked_correct"].mean()
    }), include_groups=False).reset_index()
    g.to_csv(charts_dir / "01_clean_vs_fgsm_accuracy_by_eps.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(g["eps"], g["clean_accuracy"], marker="o", label="clean")
    ax.plot(g["eps"], g["attacked_accuracy"], marker="o", label="FGSM attacked")
    ax.set_xlabel("eps"); ax.set_ylabel("accuracy"); ax.set_title("Clean vs FGSM accuracy by eps"); ax.legend()
    save(fig, "01_clean_vs_fgsm_accuracy_by_eps")

    # 2. FGSM attack success rate by eps (overall + conditional)
    g2 = fgsm_ok.groupby("eps").apply(lambda d: pd.Series({
        "overall_asr": d["attack_success"].mean(), "conditional_asr": conditional_asr(d)
    }), include_groups=False).reset_index()
    g2.to_csv(charts_dir / "02_fgsm_asr_by_eps.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(g2["eps"], g2["overall_asr"], marker="o", label="overall ASR")
    ax.plot(g2["eps"], g2["conditional_asr"], marker="o", label="conditional ASR")
    ax.set_xlabel("eps"); ax.set_ylabel("attack success rate"); ax.set_title("FGSM ASR by eps"); ax.legend()
    save(fig, "02_fgsm_asr_by_eps")

    # 3. FGSM accuracy by SNR (eps=0.05)
    g3 = fgsm_ok[fgsm_ok["eps"] == 0.05].groupby("snr").apply(lambda d: pd.Series({
        "clean_accuracy": d["clean_correct"].mean(), "attacked_accuracy": d["attacked_correct"].mean()
    }), include_groups=False).reset_index()
    g3.to_csv(charts_dir / "03_fgsm_accuracy_by_snr.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(g3["snr"], g3["clean_accuracy"], marker="o", label="clean")
    ax.plot(g3["snr"], g3["attacked_accuracy"], marker="o", label="FGSM (eps=0.05)")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("accuracy"); ax.set_title("FGSM accuracy by SNR"); ax.legend()
    save(fig, "03_fgsm_accuracy_by_snr")

    # 4. attacked accuracy per low-perturbation attack
    lowp_ok = ok_all[ok_all["attack_name"] != "fgsm"]
    g4 = lowp_ok.groupby("attack_name")["attacked_correct"].mean().reset_index()
    g4.to_csv(charts_dir / "04_attacked_accuracy_by_attack.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(g4["attack_name"], g4["attacked_correct"])
    ax.set_ylabel("attacked accuracy"); ax.set_title("Attacked accuracy per low-perturbation attack")
    save(fig, "04_attacked_accuracy_by_attack")

    # 5. conditional ASR per attack
    g5 = ok_all.groupby("attack_name").apply(conditional_asr, include_groups=False).reset_index(name="conditional_asr")
    g5.to_csv(charts_dir / "05_conditional_asr_by_attack.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(g5["attack_name"], g5["conditional_asr"])
    ax.set_ylabel("conditional ASR"); ax.set_title("Conditional attack success rate per attack")
    save(fig, "05_conditional_asr_by_attack")

    # 6. mean Linf per attack
    g6 = ok_all.groupby("attack_name")["perturbation_linf"].mean().reset_index()
    g6.to_csv(charts_dir / "06_mean_linf_by_attack.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(g6["attack_name"], g6["perturbation_linf"])
    ax.set_ylabel("mean Linf"); ax.set_title("Mean Linf perturbation per attack"); ax.set_yscale("log")
    save(fig, "06_mean_linf_by_attack")

    # 7. mean L2 per attack
    g7 = ok_all.groupby("attack_name")["perturbation_l2"].mean().reset_index()
    g7.to_csv(charts_dir / "07_mean_l2_by_attack.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(g7["attack_name"], g7["perturbation_l2"])
    ax.set_ylabel("mean L2"); ax.set_title("Mean L2 perturbation per attack"); ax.set_yscale("log")
    save(fig, "07_mean_l2_by_attack")

    # 8. attack success vs perturbation norm (scatter, per-attack mean linf vs conditional ASR)
    g8 = g6.merge(g5, on="attack_name")
    g8.to_csv(charts_dir / "08_asr_vs_perturbation_norm.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for _, r in g8.iterrows():
        ax.scatter(r["perturbation_linf"], r["conditional_asr"])
        ax.annotate(r["attack_name"], (r["perturbation_linf"], r["conditional_asr"]))
    ax.set_xlabel("mean Linf"); ax.set_ylabel("conditional ASR"); ax.set_xscale("log")
    ax.set_title("Attack success vs perturbation norm")
    save(fig, "08_asr_vs_perturbation_norm")

    # 9. attack latency comparison (mean total_ms per attack)
    g9 = ok_all.groupby("attack_name")["total_ms"].agg(["mean", "median", "max"]).reset_index()
    g9.to_csv(charts_dir / "09_attack_latency_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(g9["attack_name"], g9["mean"])
    ax.set_ylabel("mean total_ms/instance"); ax.set_title("Attack latency comparison"); ax.set_yscale("log")
    save(fig, "09_attack_latency_comparison")

    # 10. sensing vs AWN vs attack latency (stacked, mean across all attacks)
    stage_means = ok_all[["sensing_ms", "awn_clean_ms", "attack_generation_ms", "awn_attacked_ms"]].mean()
    stage_means.to_csv(charts_dir / "10_stage_latency_breakdown.csv")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(stage_means.index, stage_means.values)
    ax.set_ylabel("mean ms"); ax.set_title("Spectrum Sensing vs AWN vs Attack latency")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, "10_stage_latency_breakdown")

    # 11. attack success by modulation (fgsm eps=0.05 as representative)
    g11 = fgsm_ok[fgsm_ok["eps"] == 0.05].groupby("modulation").apply(conditional_asr, include_groups=False).reset_index(name="conditional_asr")
    g11.to_csv(charts_dir / "11_asr_by_modulation.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(g11["modulation"], g11["conditional_asr"])
    ax.set_ylabel("conditional ASR (FGSM eps=0.05)"); ax.set_title("Attack success by modulation")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    save(fig, "11_asr_by_modulation")

    # 12. attack success by SNR (fgsm eps=0.05)
    g12 = fgsm_ok[fgsm_ok["eps"] == 0.05].groupby("snr").apply(conditional_asr, include_groups=False).reset_index(name="conditional_asr")
    g12.to_csv(charts_dir / "12_asr_by_snr.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(g12["snr"], g12["conditional_asr"], marker="o")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("conditional ASR (FGSM eps=0.05)"); ax.set_title("Attack success by SNR")
    save(fig, "12_asr_by_snr")

    print(f"[analyze] DONE: {len(list(charts_dir.glob('*.png')))} charts written to {charts_dir}")

    apply_latency_recalibration(out)
    write_latency_outliers(out)


def write_latency_outliers(out: Path, attack: str = "ead", top_n: int = 10) -> None:
    """Records the top_n highest total_ms rows for `attack` without deleting or
    excluding any row from the underlying raw CSV. Rows exceeding mean+3*std are
    tagged unexplained_runtime_outlier; the exact cause is not inferred from the
    available instrumentation. Overwrites latency_outliers.csv only, never the
    raw per-attack CSV."""
    p = out / f"{attack}_raw_results.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d["row_index"] = d.index
    mean, std = d["total_ms"].mean(), d["total_ms"].std()
    threshold = mean + 3 * std
    top_rows = d.nlargest(top_n, "total_ms")["row_index"].tolist()

    records = []
    for idx in top_rows:
        row = d.loc[idx]
        is_outlier = row["total_ms"] > threshold
        prev_row = d.loc[idx - 1] if idx > 0 else None
        next_row = d.loc[idx + 1] if idx < len(d) - 1 else None
        records.append({
            "row_index": idx, "modulation": row["modulation"], "snr": row["snr"],
            "sample_index": row["sample_index"], "attack_params_json": row["attack_params_json"],
            "sensing_ms": row["sensing_ms"], "region_postprocess_ms": row["region_postprocess_ms"],
            "segmentation_ms": row["segmentation_ms"], "awn_clean_ms": row["awn_clean_ms"],
            "attack_generation_ms": row["attack_generation_ms"], "awn_attacked_ms": row["awn_attacked_ms"],
            "total_ms": row["total_ms"],
            "is_first_row_of_process": bool(idx == 0),
            "is_strict_3sigma_outlier": bool(is_outlier),
            "prev_row_total_ms": prev_row["total_ms"] if prev_row is not None else None,
            "next_row_total_ms": next_row["total_ms"] if next_row is not None else None,
            "classification_tag": "unexplained_runtime_outlier" if is_outlier else "elevated_not_strict_outlier",
            "classification": (
                "unexplained runtime outlier -- exact cause not determined from available instrumentation"
                if is_outlier else "elevated but within top-N, not a strict statistical outlier"
            ),
        })
    pd.DataFrame(records).to_csv(out / "latency_outliers.csv", index=False)
    print(f"[analyze] latency_outliers.csv: {attack} total_ms mean={mean:.1f}ms median={d['total_ms'].median():.1f}ms "
          f"p95={d['total_ms'].quantile(0.95):.1f}ms max={d['total_ms'].max():.1f}ms, "
          f"{sum(r['is_strict_3sigma_outlier'] for r in records)} strict outlier(s)")


if __name__ == "__main__":
    main()
