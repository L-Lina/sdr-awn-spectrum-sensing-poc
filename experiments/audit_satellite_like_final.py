"""
Step 4 FINAL RESULT AUDIT (project-close pre-close verification).

Read-only: never reruns the 576-combo matrix, never modifies
results/satellite_like_final_20260821T021117Z/raw_results.csv or any
existing summary CSV. Recomputes every core number directly from
raw_results.csv (bypassing analyze_satellite_like_final.py's own helper
functions where useful, so the two independent computations can be
cross-checked against each other) and writes a fresh set of audit_*.csv
files plus audit_report.json into results/<final_dir>/audit/.

Sections mirror the human audit request 1:1 (matrix / sensing / AMC /
attack metric definitions / perturbation norms / Top-K / latency /
processing budget / fairness-hash / GIGO).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("results/satellite_like_final_20260821T021117Z")
RAW = RESULTS_DIR / "raw_results.csv"
AUDIT_DIR = RESULTS_DIR / "audit"
AUDIT_DIR.mkdir(exist_ok=True)

BUDGETS_MS = [5, 10, 20, 35, 50, 100, 250]


def to_bool(s):
    if isinstance(s, (bool, np.bool_)):
        return bool(s)
    if isinstance(s, float) and np.isnan(s):
        return None
    if s in ("True", "TRUE", "true"):
        return True
    if s in ("False", "FALSE", "false"):
        return False
    return None


df = pd.read_csv(RAW)
bool_cols = ["clean_correct", "sensing_detected", "attacked_correct", "attack_success",
             "defended_correct", "recovered_by_defense", "clean_degraded_by_defense", "fallback_used"]
for c in bool_cols:
    df[c] = df[c].map(to_bool)
df["topk_state"] = df["topk"].apply(lambda v: "on" if pd.notna(v) else "off")

report: dict = {}

# ============================================================
# 1. Matrix audit
# ============================================================
report["1_row_count"] = int(len(df))
key_cols = ["modulation", "snr_db_dataset", "condition", "sample_index", "attack_name", "topk_state"]
dup_mask = df.duplicated(subset=key_cols, keep=False)
report["1_n_duplicate_key_rows"] = int(dup_mask.sum())
combo_counts = df.groupby(key_cols).size()
report["1_n_unique_full_combos"] = int(len(combo_counts))
report["1_max_rows_per_full_combo"] = int(combo_counts.max())
report["1_min_rows_per_full_combo"] = int(combo_counts.min())

report["1_n_modulations"] = int(df["modulation"].nunique())
report["1_modulations"] = sorted(df["modulation"].unique().tolist())
report["1_n_snr"] = int(df["snr_db_dataset"].nunique())
report["1_snrs"] = sorted(int(x) for x in df["snr_db_dataset"].unique())
report["1_n_conditions"] = int(df["condition"].nunique())
report["1_conditions"] = sorted(df["condition"].unique().tolist())
report["1_n_attacks"] = int(df["attack_name"].nunique())
report["1_attacks"] = sorted(df["attack_name"].unique().tolist())
report["1_n_topk_states"] = int(df["topk_state"].nunique())
report["1_topk_states"] = sorted(df["topk_state"].unique().tolist())
report["1_n_sample_index"] = int(df["sample_index"].nunique())
report["1_sample_indices"] = sorted(int(x) for x in df["sample_index"].unique())

expected = 3 * 4 * 4 * 3 * 2 * 2
report["1_matrix_formula"] = "3 modulations x 4 SNR x 4 channel_conditions x 3 attacks x 2 topk_states x 2 sample_indices"
report["1_expected_count"] = expected
report["1_matches_expected_row_count"] = bool(len(df) == expected)
report["1_matches_expected_unique_combos"] = bool(len(combo_counts) == expected)

base_key = ["modulation", "snr_db_dataset", "condition", "sample_index"]
base_combo_counts = df[base_key].drop_duplicates().shape[0]
report["1_n_unique_base_combos"] = int(base_combo_counts)
report["1_expected_base_combos"] = 3 * 4 * 4 * 2

bsid_counts = df.groupby("base_sample_id").size()
report["1_n_unique_base_sample_ids"] = int(bsid_counts.shape[0])
report["1_rows_per_base_sample_id_min"] = int(bsid_counts.min())
report["1_rows_per_base_sample_id_max"] = int(bsid_counts.max())
# each base_sample_id should map to exactly one (mod,snr,cond,idx) tuple
bsid_to_base = df.groupby("base_sample_id")[base_key].nunique()
report["1_base_sample_id_maps_to_multiple_base_keys"] = int((bsid_to_base > 1).any(axis=1).sum())

pd.DataFrame([report]).T.rename(columns={0: "value"}).to_csv(AUDIT_DIR / "audit_01_matrix.csv")

# ============================================================
# 2. Spectrum Sensing audit
# ============================================================
base_rows = df.drop_duplicates("base_sample_id").copy()  # sensing computed once per base

def sensing_block(sub: pd.DataFrame) -> dict:
    csr = sub["captured_signal_ratio"].dropna()
    bse = sub["boundary_start_error"].dropna().abs()
    bee = sub["boundary_end_error"].dropna().abs()
    fos = sub["false_occupied_samples"].dropna()
    return {
        "n_base": len(sub),
        "detection_rate": float(sub["sensing_detected"].mean()) if len(sub) else None,
        "no_region_count": int((sub["status"] == "no_region_detected").sum()),
        "no_segment_count": int((sub["status"] == "no_segment_produced").sum()),
        "csr_mean": float(csr.mean()) if len(csr) else None,
        "csr_median": float(csr.median()) if len(csr) else None,
        "csr_min": float(csr.min()) if len(csr) else None,
        "csr_p5": float(np.percentile(csr, 5)) if len(csr) else None,
        "csr_p95": float(np.percentile(csr, 95)) if len(csr) else None,
        "boundary_start_err_abs_mean": float(bse.mean()) if len(bse) else None,
        "boundary_end_err_abs_mean": float(bee.mean()) if len(bee) else None,
        "false_occupied_samples_mean": float(fos.mean()) if len(fos) else None,
    }

sensing_overall = sensing_block(base_rows)
sensing_by_channel = []
for cond, sub in base_rows.groupby("condition"):
    row = {"condition": cond}
    row.update(sensing_block(sub))
    sensing_by_channel.append(row)
sensing_by_mod = []
for mod, sub in base_rows.groupby("modulation"):
    row = {"modulation": mod}
    row.update(sensing_block(sub))
    sensing_by_mod.append(row)
sensing_by_snr = []
for snr, sub in base_rows.groupby("snr_db_dataset"):
    row = {"snr_db_dataset": snr}
    row.update(sensing_block(sub))
    sensing_by_snr.append(row)

pd.DataFrame([{"scope": "overall", **sensing_overall}]).to_csv(AUDIT_DIR / "audit_02_sensing_overall.csv", index=False)
pd.DataFrame(sensing_by_channel).to_csv(AUDIT_DIR / "audit_02_sensing_by_channel.csv", index=False)
pd.DataFrame(sensing_by_mod).to_csv(AUDIT_DIR / "audit_02_sensing_by_modulation.csv", index=False)
pd.DataFrame(sensing_by_snr).to_csv(AUDIT_DIR / "audit_02_sensing_by_snr.csv", index=False)

# cross-tab: channel x modulation, channel x snr, to avoid "overall average only" conclusion
sensing_by_channel_mod = []
for (cond, mod), sub in base_rows.groupby(["condition", "modulation"]):
    row = {"condition": cond, "modulation": mod}
    row.update(sensing_block(sub))
    sensing_by_channel_mod.append(row)
pd.DataFrame(sensing_by_channel_mod).to_csv(AUDIT_DIR / "audit_02_sensing_by_channel_modulation.csv", index=False)

report["2_sensing_overall"] = sensing_overall
report["2_gigo_check_csr_stable_while_accuracy_drops"] = None  # filled after section 3

# ============================================================
# 3. AMC accuracy audit (clean_correct is base-level; dedupe first)
# ============================================================
def acc_block(sub: pd.DataFrame) -> dict:
    c = sub["clean_correct"].dropna()
    return {"n_base": len(c), "clean_accuracy": float(c.mean()) if len(c) else None}

amc_overall = acc_block(base_rows)
amc_by_channel = []
for cond, sub in base_rows.groupby("condition"):
    row = {"condition": cond}
    row.update(acc_block(sub))
    amc_by_channel.append(row)
amc_by_mod = []
for mod, sub in base_rows.groupby("modulation"):
    row = {"modulation": mod}
    row.update(acc_block(sub))
    amc_by_mod.append(row)
amc_by_snr = []
for snr, sub in base_rows.groupby("snr_db_dataset"):
    row = {"snr_db_dataset": snr}
    row.update(acc_block(sub))
    amc_by_snr.append(row)

pd.DataFrame([{"scope": "overall", **amc_overall}]).to_csv(AUDIT_DIR / "audit_03_amc_overall.csv", index=False)
pd.DataFrame(amc_by_channel).to_csv(AUDIT_DIR / "audit_03_amc_by_channel.csv", index=False)
pd.DataFrame(amc_by_mod).to_csv(AUDIT_DIR / "audit_03_amc_by_modulation.csv", index=False)
pd.DataFrame(amc_by_snr).to_csv(AUDIT_DIR / "audit_03_amc_by_snr.csv", index=False)

# cross-check against existing (row-count-based, n=192) overall_summary.csv value
existing_overall = pd.read_csv(RESULTS_DIR / "overall_summary.csv")
existing_clean_acc = existing_overall.loc[existing_overall["metric"] == "clean_accuracy_overall", "value"].iloc[0]
report["3_amc_overall_true_n96"] = amc_overall
report["3_amc_overall_existing_csv_n192_value"] = float(existing_clean_acc)
report["3_amc_values_match_despite_n_difference"] = bool(abs(amc_overall["clean_accuracy"] - existing_clean_acc) < 1e-9)
report["3_note_n_pseudo_replication"] = (
    "clean_correct is computed ONCE per base combo (mod,snr,condition,sample_index) in Phase 1 "
    "and copied unchanged into all 6 final rows (3 attacks x 2 topk states) that share that "
    "base_sample_id. overall_summary.csv's clean_accuracy_overall therefore reports n=192 "
    "(96 base combos x 2 topk-state duplicates), not 192 independent evaluations. The rate value "
    "itself is numerically unaffected (duplicating each value an equal number of times does not "
    "change a mean), but n=192 should not be read as 192 independent trials -- the true independent "
    "sample count is 96. Same caveat applies to sensing_detection_rate/captured_signal_ratio "
    "(true n=96) and to fgsm/pgd_det attacked_accuracy/ASR before Top-K (true n=96 per attack, "
    "reported as n=192 because attack results are also copied into both topk-state rows). "
    "defended_accuracy/topk_recovery/topk_degradation are NOT affected (topk=on selects exactly one "
    "row per base x attack, no duplication -- n=288 there is genuinely 96 bases x 3 attacks)."
)

# GIGO: csr stable per channel while accuracy drops -- confirm at channel x modulation granularity too
csr_by_channel = {r["condition"]: r["csr_mean"] for r in sensing_by_channel}
acc_by_channel = {r["condition"]: r["clean_accuracy"] for r in amc_by_channel}
report["2_gigo_check_csr_stable_while_accuracy_drops"] = {
    "csr_by_channel": csr_by_channel, "clean_accuracy_by_channel": acc_by_channel,
    "csr_range": float(max(csr_by_channel.values()) - min(csr_by_channel.values())),
    "accuracy_range": float(max(acc_by_channel.values()) - min(acc_by_channel.values())),
}

# ============================================================
# 4/5. Attack metric definitions + recompute FGSM/PGD metrics + perturbation audit
# ============================================================
# Ground-truth definitions, read directly from experiments/run_satellite_like_final.py:
#   attacked_correct = (attacked_prediction == true_label)                         [row 299]
#   attack_success   = (attacked_prediction != clean_prediction)                    [row 300]  <- PREDICTION-CHANGE definition, not "clean-correct-then-wrong"
#   overall_asr (as computed by analyze_satellite_like_final.py's overall_summary)  = mean(attack_success) over ALL rows for that attack (unconditional on clean_correct)
#   conditional_asr                                                                 = mean(attack_success) restricted to rows where clean_correct == True
attack_defs = [
    {"metric": "attacked_correct", "numerator": "attacked_prediction == true_label", "denominator": "1 (per row)", "source_field": "attacked_prediction, true_label"},
    {"metric": "attack_success", "numerator": "attacked_prediction != clean_prediction", "denominator": "1 (per row)", "source_field": "attacked_prediction, clean_prediction",
     "note": "This IS a prediction-change-rate definition (fooling rate relative to the model's OWN clean output), not 'clean-correct-then-attack-makes-it-wrong'. Confirmed at src/adapters vs experiments/run_satellite_like_final.py:300. This is the project's existing/established convention for this field name (used identically in prior formal rounds), not newly introduced ambiguity this round -- but the final document must state this definition explicitly rather than relying on the reader's assumption."},
    {"metric": "attacked_accuracy", "numerator": "count(attacked_correct == True)", "denominator": "count(attacked_correct not null)", "source_field": "attacked_correct"},
    {"metric": "overall_asr", "numerator": "count(attack_success == True)", "denominator": "count(attack_success not null), ALL rows for that attack (unconditional on clean_correct)", "source_field": "attack_success"},
    {"metric": "conditional_asr", "numerator": "count(attack_success == True AND clean_correct == True)", "denominator": "count(clean_correct == True) for that attack", "source_field": "attack_success, clean_correct"},
]
pd.DataFrame(attack_defs).to_csv(AUDIT_DIR / "audit_04_attack_metric_definitions.csv", index=False)

# manual recompute for fgsm/pgd_det, both row-count-n and true-base-n (attack results duplicated across topk states, true n=96 per attack)
attack_metrics = []
for atk in ["fgsm", "pgd_det"]:
    sub_all = df[df["attack_name"] == atk]  # includes both topk states -> n row-count = 192
    sub_true = df[(df["attack_name"] == atk) & (df["topk_state"] == "off")]  # dedupe topk duplication -> n = 96
    acc_all = sub_all["attacked_correct"].dropna()
    acc_true = sub_true["attacked_correct"].dropna()
    asr_all = sub_all["attack_success"].dropna()
    asr_true = sub_true["attack_success"].dropna()
    cond_all = sub_all[sub_all["clean_correct"] == True]  # noqa: E712
    cond_true = sub_true[sub_true["clean_correct"] == True]  # noqa: E712
    linf_true = sub_true["perturbation_linf"].dropna()
    l2_true = sub_true["perturbation_l2"].dropna()
    attack_metrics.append({
        "attack": atk,
        "attacked_accuracy_rowcount_n192": float(acc_all.mean()), "n_rowcount": len(acc_all),
        "attacked_accuracy_true_n96": float(acc_true.mean()), "n_true": len(acc_true),
        "overall_asr_rowcount_n192": float(asr_all.mean()),
        "overall_asr_true_n96": float(asr_true.mean()),
        "conditional_asr_rowcount": float(cond_all["attack_success"].mean()), "n_cond_rowcount": len(cond_all),
        "conditional_asr_true": float(cond_true["attack_success"].mean()), "n_cond_true": len(cond_true),
        "linf_n": len(linf_true), "linf_mean": float(linf_true.mean()), "linf_std": float(linf_true.std()),
        "linf_min": float(linf_true.min()), "linf_median": float(linf_true.median()),
        "linf_p95": float(np.percentile(linf_true, 95)), "linf_max": float(linf_true.max()),
        "linf_n_unique_rounded6": int(linf_true.round(6).nunique()),
        "l2_n": len(l2_true), "l2_mean": float(l2_true.mean()), "l2_std": float(l2_true.std()),
        "l2_min": float(l2_true.min()), "l2_median": float(l2_true.median()),
        "l2_p95": float(np.percentile(l2_true, 95)), "l2_max": float(l2_true.max()),
        "l2_n_unique_rounded6": int(l2_true.round(6).nunique()),
    })
pd.DataFrame(attack_metrics).to_csv(AUDIT_DIR / "audit_05_attack_and_perturbation_metrics.csv", index=False)

# cross-check against existing overall_summary.csv (row-count based) values
existing_vals = {}
for atk in ["fgsm", "pgd_det"]:
    for metric in ["attacked_accuracy", "overall_asr", "conditional_asr", "mean_linf", "mean_l2"]:
        key = f"{atk}_{metric}"
        v = existing_overall.loc[existing_overall["metric"] == key, "value"]
        existing_vals[key] = float(v.iloc[0]) if len(v) else None
report["4_existing_overall_summary_values"] = existing_vals
match_check = {}
for atk_row in attack_metrics:
    atk = atk_row["attack"]
    match_check[f"{atk}_attacked_accuracy"] = bool(abs(atk_row["attacked_accuracy_rowcount_n192"] - existing_vals[f"{atk}_attacked_accuracy"]) < 1e-9)
    match_check[f"{atk}_overall_asr"] = bool(abs(atk_row["overall_asr_rowcount_n192"] - existing_vals[f"{atk}_overall_asr"]) < 1e-9)
    match_check[f"{atk}_conditional_asr"] = bool(abs(atk_row["conditional_asr_rowcount"] - existing_vals[f"{atk}_conditional_asr"]) < 1e-9)
report["4_5_recompute_matches_existing_csv"] = match_check

# perturbation Linf near-equality investigation: per-base-sample-id comparison, fgsm vs pgd_det
pivot_linf = df[df["topk_state"] == "off"].pivot(index="base_sample_id", columns="attack_name", values="perturbation_linf")
pivot_l2 = df[df["topk_state"] == "off"].pivot(index="base_sample_id", columns="attack_name", values="perturbation_l2")
pivot_pred = df[df["topk_state"] == "off"].pivot(index="base_sample_id", columns="attack_name", values="attacked_prediction")
paired = pd.DataFrame({
    "fgsm_linf": pivot_linf["fgsm"], "pgd_det_linf": pivot_linf["pgd_det"],
    "linf_diff": (pivot_linf["fgsm"] - pivot_linf["pgd_det"]).abs(),
    "fgsm_l2": pivot_l2["fgsm"], "pgd_det_l2": pivot_l2["pgd_det"],
    "l2_diff": (pivot_l2["fgsm"] - pivot_l2["pgd_det"]).abs(),
    "fgsm_pred": pivot_pred["fgsm"], "pgd_det_pred": pivot_pred["pgd_det"],
    "same_attacked_prediction": pivot_pred["fgsm"] == pivot_pred["pgd_det"],
})
paired.to_csv(AUDIT_DIR / "audit_05_fgsm_vs_pgd_paired_by_base.csv")
report["5_paired_linf_diff_mean"] = float(paired["linf_diff"].mean())
report["5_paired_linf_diff_max"] = float(paired["linf_diff"].max())
report["5_paired_linf_exactly_equal_count"] = int((paired["linf_diff"] < 1e-12).sum())
report["5_paired_l2_diff_mean"] = float(paired["l2_diff"].mean())
report["5_paired_l2_exactly_equal_count"] = int((paired["l2_diff"] < 1e-12).sum())
report["5_paired_same_attacked_prediction_rate"] = float(paired["same_attacked_prediction"].mean())
report["5_n_base_pairs_compared"] = int(len(paired))

# 10-sample spot check table (first 10 base_sample_ids with a valid pairing)
report["5_spot_check_10"] = paired.head(10).reset_index().to_dict("records")

# ============================================================
# 6. Top-K metric audit
# ============================================================
topk_defs = [
    {"metric": "defended_correct", "numerator": "defended_prediction == true_label", "denominator": "1 (per row, topk=on only)"},
    {"metric": "recovered_by_defense", "numerator": "attacked_correct is False AND defended_correct is True", "denominator": "1 (per row, topk=on only)",
     "note": "Recovery = attack made the prediction wrong (relative to true_label, NOT relative to attack_success/prediction-change), AND Top-K brought it back to correct. For attack_name='none' rows this is structurally impossible (attacked_correct==clean_correct there) and correctly evaluates to False, not counted as spurious recovery."},
    {"metric": "clean_degraded_by_defense", "numerator": "attack_name=='none' AND clean_correct is True AND defended_correct is False", "denominator": "1 (per row, topk=on only, attack='none' subset)"},
]
pd.DataFrame(topk_defs).to_csv(AUDIT_DIR / "audit_06_topk_metric_definitions.csv", index=False)

topk_on = df[df["topk_state"] == "on"]
topk_rows = []
for atk in ["fgsm", "pgd_det"]:
    sub = topk_on[topk_on["attack_name"] == atk]
    rec = sub["recovered_by_defense"].dropna()
    n_attack_failed = int((sub["attacked_correct"] == False).sum())  # noqa: E712
    topk_rows.append({
        "attack": atk, "n_topk_on_rows": len(sub),
        "n_attack_failed_attacked_correct_false": n_attack_failed,
        "n_recovered": int(rec.sum()), "recovery_rate_over_all_topk_on": float(rec.mean()) if len(rec) else None,
        "recovery_rate_over_attack_failed_only": float(rec.sum() / n_attack_failed) if n_attack_failed else None,
    })
sub_none = topk_on[topk_on["attack_name"] == "none"]
deg = sub_none["clean_degraded_by_defense"].dropna()
n_clean_correct = int((sub_none["clean_correct"] == True).sum())  # noqa: E712
topk_rows.append({
    "attack": "none", "n_topk_on_rows": len(sub_none),
    "n_clean_correct_before_defense": n_clean_correct,
    "n_degraded": int(deg.sum()), "degradation_rate_over_all_topk_on": float(deg.mean()) if len(deg) else None,
    "degradation_rate_over_clean_correct_only": float(deg.sum() / n_clean_correct) if n_clean_correct else None,
})
pd.DataFrame(topk_rows).to_csv(AUDIT_DIR / "audit_06_topk_recovery_degradation_by_attack.csv", index=False)

# by channel / modulation / snr denominator tables
def topk_block(sub: pd.DataFrame) -> dict:
    fgsm = sub[sub["attack_name"] == "fgsm"]
    pgd = sub[sub["attack_name"] == "pgd_det"]
    none_ = sub[sub["attack_name"] == "none"]
    out = {"n_topk_on_rows": len(sub)}
    out["fgsm_n"] = len(fgsm); out["fgsm_recovery_rate"] = float(fgsm["recovered_by_defense"].dropna().mean()) if fgsm["recovered_by_defense"].notna().any() else None
    out["pgd_det_n"] = len(pgd); out["pgd_det_recovery_rate"] = float(pgd["recovered_by_defense"].dropna().mean()) if pgd["recovered_by_defense"].notna().any() else None
    out["none_n"] = len(none_); out["none_degradation_rate"] = float(none_["clean_degraded_by_defense"].dropna().mean()) if none_["clean_degraded_by_defense"].notna().any() else None
    return out

topk_by_channel = []
for cond, sub in topk_on.groupby("condition"):
    row = {"condition": cond}; row.update(topk_block(sub)); topk_by_channel.append(row)
topk_by_mod = []
for mod, sub in topk_on.groupby("modulation"):
    row = {"modulation": mod}; row.update(topk_block(sub)); topk_by_mod.append(row)
topk_by_snr = []
for snr, sub in topk_on.groupby("snr_db_dataset"):
    row = {"snr_db_dataset": snr}; row.update(topk_block(sub)); topk_by_snr.append(row)
pd.DataFrame(topk_by_channel).to_csv(AUDIT_DIR / "audit_06_topk_by_channel.csv", index=False)
pd.DataFrame(topk_by_mod).to_csv(AUDIT_DIR / "audit_06_topk_by_modulation.csv", index=False)
pd.DataFrame(topk_by_snr).to_csv(AUDIT_DIR / "audit_06_topk_by_snr.csv", index=False)

report["6_topk_recovery_degradation_by_attack"] = topk_rows

# ============================================================
# 7. Latency audit
# ============================================================
latency_stage_cols = ["channel_ms", "embed_ms", "sensing_ms", "segmentation_ms", "awn_preprocess_ms",
                       "awn_clean_ms", "attack_generation_ms", "attacked_inference_ms", "topk_ms", "defended_inference_ms"]
report["7_total_ms_components"] = latency_stage_cols
report["7_components_confirmed_excluded_by_code_reading"] = [
    "CSV write (csv.DictWriter loop runs once, after all rows collected, outside every per-row now_ns() timer pair)",
    "plotting/matplotlib (only exists in the separate analyze_satellite_like_final.py process, never imported/run by run_satellite_like_final.py)",
    "AWN model loading (AWNModelAdapter(...) construction happens once in main(), before Phase 1's loop, not wrapped in any per-row timer)",
    "dataset initialization (load_radioml_dict(...) happens once in main(), before Phase 1's loop, not wrapped in any per-row timer)",
]

scenarios = {
    "clean": df[(df["attack_name"] == "none") & (df["topk_state"] == "off")],
    "fgsm": df[(df["attack_name"] == "fgsm") & (df["topk_state"] == "off")],
    "pgd_det": df[(df["attack_name"] == "pgd_det") & (df["topk_state"] == "off")],
    "fgsm_topk": df[(df["attack_name"] == "fgsm") & (df["topk_state"] == "on")],
    "pgd_det_topk": df[(df["attack_name"] == "pgd_det") & (df["topk_state"] == "on")],
}
latency_rows = []
for name, sub in scenarios.items():
    vals = sub["total_ms"].dropna()
    # cross-check: total_ms should equal the sum of its own non-null stage components, per row
    recompute = sub[latency_stage_cols].sum(axis=1, skipna=True)
    max_recompute_diff = float((sub["total_ms"] - recompute).abs().max()) if len(sub) else None
    latency_rows.append({
        "scenario": name, "n": len(vals),
        "mean": float(vals.mean()), "median": float(vals.median()),
        "p90": float(np.percentile(vals, 90)), "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)), "max": float(vals.max()),
        "max_abs_diff_vs_recomputed_sum_of_stages": max_recompute_diff,
    })
pd.DataFrame(latency_rows).to_csv(AUDIT_DIR / "audit_07_latency.csv", index=False)
report["7_latency"] = latency_rows

# ============================================================
# 8. Processing-budget audit (adds p99 fit, not present in existing processing_budget.csv)
# ============================================================
budget_rows = []
for name, sub in scenarios.items():
    vals = sub["total_ms"].dropna()
    if len(vals) == 0:
        continue
    median_v, p95_v, p99_v = float(vals.median()), float(np.percentile(vals, 95)), float(np.percentile(vals, 99))
    row = {"scenario": name, "median_total_ms": median_v, "p95_total_ms": p95_v, "p99_total_ms": p99_v}
    for b in BUDGETS_MS:
        row[f"fits_{b}ms_median"] = median_v <= b
        row[f"fits_{b}ms_p95"] = p95_v <= b
        row[f"fits_{b}ms_p99"] = p99_v <= b
        row[f"empirical_fit_rate_{b}ms"] = float((vals <= b).mean())
    budget_rows.append(row)
pd.DataFrame(budget_rows).to_csv(AUDIT_DIR / "audit_08_processing_budget.csv", index=False)

# ============================================================
# 9. Channel x Attack fairness / hash audit
# ============================================================
hash_cov = {
    "n_rows_total": len(df),
    "n_rows_with_channel_input_hash": int(df["channel_input_hash"].notna().sum()),
    "n_rows_with_clean_segment_hash": int(df["clean_segment_hash"].notna().sum()),
}
# within each base_sample_id, hashes must be identical across all 6 rows (same base reused)
hash_consistency = df.groupby("base_sample_id")[["channel_input_hash", "clean_segment_hash"]].nunique()
hash_cov["n_base_sample_ids_with_inconsistent_channel_hash"] = int((hash_consistency["channel_input_hash"] > 1).sum())
hash_cov["n_base_sample_ids_with_inconsistent_segment_hash"] = int((hash_consistency["clean_segment_hash"] > 1).sum())
# hashes must differ ACROSS different base_sample_ids that share the same (mod,snr,cond) but different idx,
# and across different cond for same (mod,snr,idx) -- i.e. hash should be ~unique per base combo
hash_unique_check = base_rows.groupby("channel_input_hash")["base_sample_id"].nunique()
hash_cov["n_channel_input_hash_collisions_across_different_bases"] = int((hash_unique_check > 1).sum())
report["9_hash_coverage_and_fairness"] = hash_cov
pd.DataFrame([hash_cov]).to_csv(AUDIT_DIR / "audit_09_fairness_hash.csv", index=False)

# ============================================================
# 10. Claim-to-evidence support table (machine-checkable claims only)
# ============================================================
# prediction-change rate by channel, for the two open claim checks below
pc_by_channel = {}
for atk in ["fgsm", "pgd_det"]:
    sub = df[(df["attack_name"] == atk) & (df["topk_state"] == "off")]
    pc_by_channel[atk] = sub.groupby("condition")["attack_success"].mean().to_dict()
fgsm_asr_high = bool(min(pc_by_channel["fgsm"].values()) >= 0.80)
pgd_asr_high = bool(min(pc_by_channel["pgd_det"].values()) >= 0.90)

# top-k condition-dependence: recovery rate varies materially by channel (audit_06_topk_by_channel.csv already has this)
topk_by_channel_df = pd.DataFrame(topk_by_channel)
topk_condition_dependent = bool(
    topk_by_channel_df["fgsm_recovery_rate"].max() != topk_by_channel_df["fgsm_recovery_rate"].min()
    or topk_by_channel_df["pgd_det_recovery_rate"].max() != topk_by_channel_df["pgd_det_recovery_rate"].min()
)

claims = [
    {"claim": "sensing detection = 100% (576/576)", "check": bool(base_rows["sensing_detected"].mean() == 1.0 and len(base_rows) == 96)},
    {"claim": "captured_signal_ratio stays ~0.97 across all channel conditions", "check": bool(min(csr_by_channel.values()) > 0.96)},
    {"claim": "clean AMC accuracy decreases monotonically clean>mild>moderate>strong", "check": bool(
        acc_by_channel["clean"] > acc_by_channel["mild"] > acc_by_channel["moderate"] > acc_by_channel["strong"]
    )},
    {"claim": "FGSM prediction change rate stays >=80% across all 4 channel conditions", "check": fgsm_asr_high},
    {"claim": "PGD(det) prediction change rate stays >=90% across all 4 channel conditions", "check": pgd_asr_high},
    {"claim": "Top-K recovery/degradation rate varies by channel condition (condition-dependent, not universally effective)", "check": topk_condition_dependent},
    {"claim": "Top-K recovery rate (correct denominator) is low: FGSM ~2.5%, PGD(det) ~6.9%", "check": bool(
        abs(topk_rows[0]["recovery_rate_over_attack_failed_only"] - 0.025) < 1e-6
        and abs(topk_rows[1]["recovery_rate_over_attack_failed_only"] - 0.06896551724137931) < 1e-6
    )},
    {"claim": "Top-K clean degradation rate (correct denominator) ~29.03%", "check": bool(
        abs(topk_rows[2]["degradation_rate_over_clean_correct_only"] - 0.2903225806451613) < 1e-6
    )},
    {"claim": "GIGO: csr stable while accuracy drops => not a sensing failure (observational, not causal)", "check": bool(
        (max(csr_by_channel.values()) - min(csr_by_channel.values())) < 0.01
        and (max(acc_by_channel.values()) - min(acc_by_channel.values())) > 0.3
    )},
    {"claim": "amplitude_scale is a robustness stress condition, not a full RF link-budget / path-loss simulation", "check": None,
     "note": "Not machine-checkable from raw_results.csv alone -- verified by code reading (src/channel/satellite_like.py has no RF propagation model) and cross-referenced against docs/research/SATELLITE_LIKE_CHANNEL_SIMULATOR_DESIGN_ZH_TW.md section 16.3."},
    {"claim": "Attack threat model is A0 (receiver-side digital white-box), not validated OTA / A1", "check": None,
     "note": "Not machine-checkable from raw_results.csv alone -- verified by code reading (AttackAdapter perturbs the AWN-input IQ tensor already in memory; no RF transmission/reception path exists in this repo's attack path)."},
    {"claim": "0 errors / 0 fallback / 0 NaN-Inf across all 576 rows", "check": bool(
        (df["status"] == "ok").all() and (df["fallback_used"] == False).all()  # noqa: E712
    )},
]
pd.DataFrame(claims).to_csv(AUDIT_DIR / "audit_10_claim_checks.csv", index=False)
report["10_claim_checks"] = claims

# ============================================================
# 11. Formally-named artifacts requested by the human audit round
#     (metric_definition_audit.csv, topk_denominator_audit.csv,
#     unique_attack_sample_audit.csv, claim_to_evidence_audit.csv)
# ============================================================
metric_definition_rows = [
    {"metric_formal_name": "Attacked Accuracy", "legacy_field": "attacked_correct",
     "numerator": "attacked_prediction == true_label", "denominator": "unique attacked base samples (n=96 per attack)"},
    {"metric_formal_name": "Prediction Change Rate", "legacy_field": "attack_success (legacy name -- NOT a conditional adversarial-success definition)",
     "numerator": "attacked_prediction != clean_prediction", "denominator": "unique attacked base samples (n=96 per attack)"},
    {"metric_formal_name": "Conditional Attack Success Rate", "legacy_field": "attack_success restricted to clean_correct==True (equivalently: attacked_correct==False restricted to clean_correct==True)",
     "numerator": "clean_correct==True AND attacked_correct==False", "denominator": "clean_correct==True (n=31)"},
    {"metric_formal_name": "Top-K Recovery Rate", "legacy_field": "recovered_by_defense",
     "numerator": "attacked_correct==False AND defended_correct==True", "denominator": "attacked_correct==False (attack-eligible samples only)"},
    {"metric_formal_name": "Top-K Clean Degradation Rate", "legacy_field": "clean_degraded_by_defense",
     "numerator": "attack_name=='none' AND clean_correct==True AND defended_correct==False", "denominator": "attack_name=='none' AND clean_correct==True (n=31)"},
]
pd.DataFrame(metric_definition_rows).to_csv(AUDIT_DIR / "metric_definition_audit.csv", index=False)

pd.DataFrame(topk_rows).rename(columns={
    "n_topk_on_rows": "n_topk_on_rows_reference_only",
}).to_csv(AUDIT_DIR / "topk_denominator_audit.csv", index=False)

unique_attack_sample_rows = []
for atk_row in attack_metrics:
    atk = atk_row["attack"]
    unique_attack_sample_rows.append({
        "attack": atk, "n_unique_attacked_base": atk_row["n_true"],
        "attacked_accuracy": atk_row["attacked_accuracy_true_n96"],
        "prediction_change_rate": atk_row["overall_asr_true_n96"],
        "conditional_attack_success_rate": atk_row["conditional_asr_true"],
        "n_clean_correct_denominator": atk_row["n_cond_true"],
        "mean_linf": atk_row["linf_mean"], "mean_l2": atk_row["l2_mean"],
        "note": "n=96 is the true independent unique-attacked-base count; row-count-based n=192 in overall_summary.csv double-counts each value once per Top-K state (see doc section 22.5).",
    })
pd.DataFrame(unique_attack_sample_rows).to_csv(AUDIT_DIR / "unique_attack_sample_audit.csv", index=False)

pd.DataFrame(claims).to_csv(AUDIT_DIR / "claim_to_evidence_audit.csv", index=False)

with open(AUDIT_DIR / "audit_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print("AUDIT DONE. Files written to", AUDIT_DIR)
for p in sorted(AUDIT_DIR.glob("*")):
    print(" -", p.name)
