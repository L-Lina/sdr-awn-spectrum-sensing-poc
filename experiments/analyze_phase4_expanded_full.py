#!/usr/bin/env python3
"""Aggregate analysis for the formal (K-reduced {10,20,30,40,50,80,128}, full-N) Phase 4
defense-effectiveness run (results/formal_phase4_expanded_full/ablation_summary.csv).

Reads the raw per-row ablation summary and produces long-format aggregate CSVs across
three deployment-fairness entry points (all modulations / excl-WBFM sensitivity /
WBFM-only) and eight groupings (K alone -- the only directly-deployable "global
fixed-K" view -- plus attack x K, modulation x K, SNR x K, eps x K, attack x
modulation x K, attack x SNR x K, attack x eps x K). Every rate metric is emitted with
its explicit numerator and denominator column, never as a bare average. Net accuracy
gain (mean(defended_correct) - mean(attacked_correct)) gets a numpy-only bootstrap 95%
CI (5000 resamples, seed=42, resampling instance/row indices within the group) so
"stable" findings can be told apart from noise.

Does not modify fft_topk_denoise, TopKAdapter, or any other formal defense code --
read-only analysis over an already-completed run's CSV.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 42

GROUPINGS = [
    ("K_only", ["topk"]),
    ("attack_x_K", ["attack", "topk"]),
    ("modulation_x_K", ["modulation", "topk"]),
    ("snr_x_K", ["snr", "topk"]),
    ("eps_x_K", ["attack_eps", "topk"]),
    ("attack_x_modulation_x_K", ["attack", "modulation", "topk"]),
    ("attack_x_snr_x_K", ["attack", "snr", "topk"]),
    ("attack_x_eps_x_K", ["attack", "attack_eps", "topk"]),
]

KEY_COLS = ["attack", "modulation", "snr", "attack_eps", "topk"]


def bootstrap_net_gain_ci(defended_correct, attacked_correct, n_resamples=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    dc = np.asarray(defended_correct, dtype=float)
    ac = np.asarray(attacked_correct, dtype=float)
    n = len(dc)
    if n == 0:
        return np.nan, np.nan, np.nan
    point = float(dc.mean() - ac.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = dc[idx].mean(axis=1) - ac[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


def group_stats(g: pd.DataFrame) -> dict:
    n = len(g)
    clean_correct = g["clean_correct"].astype(bool)
    attacked_correct = g["attacked_correct"].astype(bool)
    defended_correct = g["defended_correct"].astype(bool)
    changed_by_attack = g["changed_by_attack"].astype(bool)
    attacked_wrong = g["attacked_wrong"].astype(bool)
    recovered = g["recovered_by_defense"].astype(bool)
    pred_changed_by_defense = g["defense_changed_prediction"].astype(bool)
    clean_broken = g["clean_broken_by_defense"].astype(bool)

    clean_acc_num = int(clean_correct.sum())
    attacked_acc_num = int(attacked_correct.sum())
    defended_acc_num = int(defended_correct.sum())
    changed_n = int(changed_by_attack.sum())
    wrong_n = int(attacked_wrong.sum())

    recovery_count = int(recovered.sum())
    degradation_count = int(clean_broken.sum())

    true_label_recovery_num = int((attacked_wrong & (g["pred_defended"] == g["label"])).sum())
    clean_pred_recovery_num = int((attacked_wrong & (g["pred_defended"] == g["pred_clean"])).sum())

    point, lo, hi = bootstrap_net_gain_ci(defended_correct, attacked_correct)

    def rate(num, den):
        return (num / den) if den else float("nan")

    return dict(
        n=n,
        clean_accuracy=rate(clean_acc_num, n), clean_accuracy_num=clean_acc_num, clean_accuracy_den=n,
        attacked_accuracy=rate(attacked_acc_num, n), attacked_accuracy_num=attacked_acc_num, attacked_accuracy_den=n,
        defended_accuracy=rate(defended_acc_num, n), defended_accuracy_num=defended_acc_num, defended_accuracy_den=n,
        net_accuracy_gain=point,
        net_accuracy_gain_ci_lo=lo, net_accuracy_gain_ci_hi=hi,
        net_accuracy_gain_significant=bool(not np.isnan(lo) and (lo > 0 or hi < 0)),
        recovery_count=recovery_count,
        degradation_count=degradation_count,
        net_transition=recovery_count - degradation_count,
        overall_recovery_rate=rate(recovery_count, n), overall_recovery_num=recovery_count, overall_recovery_den=n,
        conditional_recovery_rate=rate(recovery_count, changed_n),
        conditional_recovery_num=recovery_count, conditional_recovery_den=changed_n,
        true_label_recovery_rate=rate(true_label_recovery_num, wrong_n),
        true_label_recovery_num=true_label_recovery_num, true_label_recovery_den=wrong_n,
        clean_pred_recovery_rate=rate(clean_pred_recovery_num, wrong_n),
        clean_pred_recovery_num=clean_pred_recovery_num, clean_pred_recovery_den=wrong_n,
        clean_degradation_rate=rate(degradation_count, clean_acc_num),
        clean_degradation_num=degradation_count, clean_degradation_den=clean_acc_num,
        prediction_changed_rate=rate(int(pred_changed_by_defense.sum()), n),
        prediction_changed_num=int(pred_changed_by_defense.sum()), prediction_changed_den=n,
        iq_linf_attacked_defended_mean=float(g["iq_linf_attacked_defended"].mean()) if n else float("nan"),
        iq_l2_attacked_defended_mean=float(g["iq_l2_attacked_defended"].mean()) if n else float("nan"),
        is_k128_baseline=bool((g["topk"] == 128).all()) if n else False,
    )


def build_scope_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grouping_name, group_cols in GROUPINGS:
        for key_vals, g in df.groupby(group_cols, dropna=False):
            if not isinstance(key_vals, tuple):
                key_vals = (key_vals,)
            row = {k: None for k in KEY_COLS}
            for col, val in zip(group_cols, key_vals):
                row[col] = val
            row["grouping"] = grouping_name
            row.update(group_stats(g))
            rows.append(row)
    cols = ["grouping"] + KEY_COLS + [c for c in rows[0].keys() if c not in ("grouping", *KEY_COLS)]
    return pd.DataFrame(rows)[cols]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/formal_phase4_expanded_full/ablation_summary.csv")
    ap.add_argument("--output-dir", default="results/formal_phase4_expanded_full/aggregates")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scopes = {
        "all_modulations": df,
        "excl_wbfm_sensitivity": df[df["modulation"] != "WBFM"],
        "wbfm_only": df[df["modulation"] == "WBFM"],
    }

    for scope_name, scope_df in scopes.items():
        table = build_scope_table(scope_df)
        out_path = out_dir / f"{scope_name}.csv"
        table.to_csv(out_path, index=False)
        print(f"[analyze] {scope_name}: {len(table)} aggregate rows -> {out_path}")

    print(f"[analyze] done. NOTE: attack-specific and modulation-specific K groupings "
          f"(attack_x_K, modulation_x_K, attack_x_modulation_x_K, ...) are oracle-analysis "
          f"only -- they condition on ground-truth attack identity or modulation label, which "
          f"a real deployed defender does not have. Only K_only (global fixed-K, all_modulations "
          f"scope) is a directly deployable view. K=128 rows are the no-defense baseline "
          f"(is_k128_baseline=True) and must never be counted as a defense success.")


if __name__ == "__main__":
    main()
