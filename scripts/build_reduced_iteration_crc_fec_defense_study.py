"""
Rebuild reports/reduced_iteration_crc_fec_defense_study_20261022/ from the five formal result directories.

Generates, in order:
    data/            research CSVs (every row carries its source_file)
    figure_sources/  values plotted in each figure (derived from data/, Figure 4 from raw_samples.csv)
    figures/         PNG figures (drawn from figure_sources/)
    RESEARCH_REPORT_ZH_TW.md, README.md   (every number rendered from the computed data)

Source result directories are fixed below. The script reads them only; it fails if a required source file is
missing or if any run's validation.json is not PASS. Qualitative statements in the report are guarded by
assertions on the data, so a change in the sources makes the build fail instead of producing stale text.

Run from the repository root (matplotlib and numpy required):
    ~/adversarial-rf/.venv/bin/python scripts/build_reduced_iteration_crc_fec_defense_study.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports" / "reduced_iteration_crc_fec_defense_study_20261022"
R1 = "results/reduced_iteration_attacks_vm_20261022_084556"
R2 = "results/synth_crc_fec_integration_20261022_090735"
R3 = "results/reduced_iteration_crc_fec_vm_20261022_093829"
R4 = "results/reduced_iteration_crc_fec_defense_vm_20261022_103042"
R5 = "results/clean_crc_fec_defense_vm_20261022_104250"
RUNS = (R1, R2, R3, R4, R5)
REQUIRED = {
    R1: ("validation.json", "manifest.json", "profile_summary.csv"),
    R2: ("validation.json", "manifest.json", "v2_clean_chain_summary.csv", "v3_attack_chain_summary.csv"),
    R3: ("validation.json", "manifest.json", "profile_summary.csv", "min_usable.csv", "modulation_summary.csv"),
    R4: ("validation.json", "manifest.json", "profile_summary.csv", "defense_summary.csv", "modulation_summary.csv",
         "adaptive_k_selection_summary.csv", "latency_summary.csv", "raw_samples.csv"),
    R5: ("validation.json", "manifest.json", "defense_clean_summary.csv", "modulation_summary.csv",
         "adaptive_k_selection_summary.csv", "latency_summary.csv", "attack_run_decomposition.csv", "raw_samples.csv"),
}
FOUR = {"fgsm", "pgd", "cw", "ead"}
DEF = ["none", "topk10", "topk20", "topk30", "topk40", "adaptive_k"]
DEF_LABEL = {"none": "none", "topk10": "K10", "topk20": "K20", "topk30": "K30", "topk40": "K40",
             "adaptive_k": "Adaptive-K"}
BANNED = ["顯示", "確認", "命中", "掌握", "綜合測試結果", "證明", "證實", "最佳", "全面優於", "最小擾動", "\u2014",
          "meeting", "evidence", "Written for", "指導教授", "實驗室成員", "本週進度", "下一階段", "下一步", "工程週報", "顯著"]
USED = set()


class BuildError(RuntimeError):
    pass


def require(cond, msg):
    if not cond:
        raise BuildError(msg)


def rd(rel):
    USED.add(rel)
    with (REPO / rel).open(newline="") as f:
        return list(csv.DictReader(f))


def rj(rel):
    USED.add(rel)
    return json.loads((REPO / rel).read_text())


def wr(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def fec_label(v):
    return "on" if str(v) == "True" else "off"


def preflight():
    for run, files in REQUIRED.items():
        for fn in files:
            require((REPO / run / fn).is_file(), f"required source file missing: {run}/{fn}")
        v = json.loads((REPO / run / "validation.json").read_text())
        require(v.get("overall") == "PASS", f"{run}/validation.json overall is {v.get('overall')!r}, not PASS")


# ======================================================================================== data/
def build_data():
    D = {}
    amc = [r for r in rd(f"{R1}/profile_summary.csv") if r["attack"] in FOUR]
    m1 = rj(f"{R1}/manifest.json")
    require(len(amc) == 27, f"expected 27 AMC-only profiles, found {len(amc)}")
    rows = []
    for r in amc:
        red = (r["latency_reduction_vs_ead_9x100_pct"] if r["attack"] == "ead"
               else r["latency_reduction_vs_group_baseline_pct"])
        base = {"pgd": "pgd:eps0.03_steps10", "cw": "cw:c1_steps20_lr0.01",
                "ead": "ead_phase1:bss9_maxit100"}.get(r["attack"], "")
        rows.append({"attack": r["attack"], "group": r["group"], "profile": r["profile"], "eps": r["eps"],
                     "alpha": r["alpha"], "steps": r["steps"], "binary_search_steps": r["binary_search_steps"],
                     "max_iterations": r["max_iterations"], "is_baseline": r["is_baseline"],
                     "n_total": r["n_total"], "n_clean_correct": r["n_clean_correct"],
                     "n_attack_success": r["n_attack_success"], "conditional_asr_pct": r["conditional_asr_pct"],
                     "formal_baseline_conditional_asr_pct": r["baseline_conditional_asr_pct"],
                     "usable_threshold_pct": r["usable_threshold_pct"], "usable_90pct": r["usable_90pct"],
                     "latency_n_calls": r["latency_n_calls"], "latency_median_ms": r["latency_median_ms"],
                     "latency_mean_ms": r["latency_mean_ms"], "latency_p95_ms": r["latency_p95_ms"],
                     "latency_baseline_profile": base, "latency_reduction_pct": red,
                     "linf_median": r["linf_median"], "l2_median": r["l2_median"],
                     "linf_median_successful": r["linf_median_successful"],
                     "l2_median_successful": r["l2_median_successful"],
                     "fallback_count": r["fallback_count"], "invalid_count": r["invalid_count"],
                     "source_file": f"{R1}/profile_summary.csv",
                     "baseline_value_source": f"{R1}/manifest.json:matrix.formal_baseline_asr_pct"})
    wr(OUT / "data" / "amc_reduced_iteration_summary.csv", rows)
    D["AMC"] = {(r["group"], r["profile"]): r for r in rows}
    D["amc_rows"] = rows

    c3 = rd(f"{R3}/profile_summary.csv")
    keep3 = ["group", "attack", "profile", "fec_enabled", "is_baseline", "eps", "alpha", "steps",
             "binary_search_steps", "max_iterations", "reused_from", "n_total", "n_clean_correct", "n_attack_success",
             "n_attack_success_oracle_crc_pass", "n_attack_success_oracle_frame_correct",
             "n_attack_success_predicted_crc_pass", "n_attack_success_predicted_frame_correct", "conditional_asr_pct",
             "oracle_crc_given_attack_success_pct", "oracle_frame_correct_given_attack_success_pct",
             "predicted_crc_given_attack_success_pct", "predicted_frame_correct_given_attack_success_pct",
             "joint_amc_oracle_crc_pct", "clean_oracle_crc_pass_pct", "oracle_ber_mean", "predicted_ber_mean",
             "oracle_crc_false_accept_n", "predicted_crc_false_accept_n", "latency_median_ms", "latency_mean_ms",
             "latency_p95_ms", "latency_reduction_vs_baseline_pct", "baseline_profile", "amc_usable",
             "joint_oracle_usable", "n_degenerate_included", "na_reasons"]
    rows = []
    for r in c3:
        row = {k: r[k] for k in keep3 if k != "fec_enabled"}
        row["fec"] = fec_label(r["fec_enabled"])
        row["source_file"] = f"{R3}/profile_summary.csv"
        rows.append(row)
    wr(OUT / "data" / "crc_fec_reduced_iteration_summary.csv", rows)
    D["C3"] = {(r["group"], r["profile"], r["fec"]): r for r in rows}
    D["c3_rows"] = rows

    rows = []
    for m in rd(f"{R3}/min_usable.csv"):
        s = (D["C3"].get((m["group"], m["minimum_usable_profile"], fec_label(m["fec_enabled"])))
             if m["minimum_usable_profile"] else None)
        rows.append({"attack": m["attack"], "group": m["group"], "fec": fec_label(m["fec_enabled"]),
                     "criterion": m["criterion"],
                     "role": "system-behavior analysis" if m["criterion"] == "joint_predicted_usable" else "primary",
                     "iteration_param": m["iteration_param"], "baseline_profile": m["baseline_profile"],
                     "baseline_conditional_asr_pct": m["baseline_conditional_asr_pct"],
                     "baseline_crc_metric": m["crc_metric"], "baseline_crc_metric_pct": m["baseline_crc_metric_pct"],
                     "minimum_usable_profile": m["minimum_usable_profile"],
                     "minimum_usable_value": m["minimum_usable_value"],
                     "conditional_asr_pct": m["conditional_asr_pct"], "asr_retention_pct": m["asr_retention_pct"],
                     "oracle_crc_given_attack_success_pct": s["oracle_crc_given_attack_success_pct"] if s else "",
                     "crc_metric_pct": m["crc_metric_pct"], "crc_retention_pct": m["crc_retention_pct"],
                     "latency_median_ms": m["latency_median_ms"], "latency_reduction_pct": m["latency_reduction_pct"],
                     "n_degenerate_in_selection": m["n_degenerate_in_selection"], "note": m["note"],
                     "source_file": f"{R3}/min_usable.csv"})
    wr(OUT / "data" / "crc_fec_min_usable_summary.csv", rows)
    D["MIN"] = {(r["group"], r["fec"], r["criterion"]): r for r in rows}

    rows = [{"defense": r["defense"], "defense_label": DEF_LABEL[r["defense"]], "fec": fec_label(r["effective_fec"]),
             **{k: r[k] for k in ("n_total", "n_clean_correct_none", "n_clean_correct_after_defense",
                                  "clean_accuracy_none_pct", "clean_accuracy_after_defense_pct",
                                  "clean_degradation_pp", "n_defense_induced_misclassification",
                                  "n_defense_induced_correction", "oracle_crc_pass_pct", "oracle_frame_correct_pct",
                                  "oracle_ber_mean", "predicted_crc_pass_pct", "predicted_frame_correct_pct",
                                  "oracle_crc_false_accept_n", "predicted_crc_false_accept_n",
                                  "defense_latency_median_ms", "defense_latency_mean_ms", "defense_latency_p95_ms",
                                  "adaptive_selected_k_mean", "n_degenerate_included")},
             "source_file": f"{R5}/defense_clean_summary.csv"} for r in rd(f"{R5}/defense_clean_summary.csv")]
    rows.sort(key=lambda r: (r["fec"], DEF.index(r["defense"])))
    wr(OUT / "data" / "clean_defense_summary.csv", rows)
    D["CLEAN"] = {(r["defense"], r["fec"]): r for r in rows}

    p4 = rd(f"{R4}/profile_summary.csv")
    d4 = {(r["attack"], r["profile"], r["effective_fec"], r["defense"]): r for r in rd(f"{R4}/defense_summary.csv")}
    dec = {(r["attack"], r["profile"], r["effective_fec"], r["defense"]): r
           for r in rd(f"{R5}/attack_run_decomposition.csv")}
    rows = []
    for r in p4:
        k = (r["attack"], r["profile"], r["effective_fec"], r["defense"])
        c = D["CLEAN"][(r["defense"], fec_label(r["effective_fec"]))]
        dd = dec.get(k, {})
        rows.append({"attack": r["attack"], "profile": r["profile"], "fec": fec_label(r["effective_fec"]),
                     "defense": r["defense"], "defense_label": DEF_LABEL[r["defense"]],
                     "is_baseline_profile": r["is_baseline_profile"], "steps": r["steps"],
                     "binary_search_steps": r["binary_search_steps"], "max_iterations": r["max_iterations"],
                     "eps": r["eps"],
                     **{x: r[x] for x in ("n_total", "n_clean_correct", "n_attack_success_before_defense",
                                          "n_attack_success_after_defense", "n_defense_recovery",
                                          "conditional_asr_before_defense_pct", "conditional_asr_after_defense_pct",
                                          "recovery_rate_pct", "n_success_before_oracle_crc_pass",
                                          "oracle_crc_given_attack_success_before_pct",
                                          "oracle_frame_correct_given_attack_success_before_pct",
                                          "oracle_crc_pass_pct", "oracle_frame_correct_pct", "predicted_crc_pass_pct",
                                          "predicted_frame_correct_pct", "predicted_frame_correct_given_recovery_pct",
                                          "oracle_ber_mean", "predicted_ber_mean", "oracle_crc_false_accept_n",
                                          "predicted_crc_false_accept_n", "attack_linf_median", "attack_l2_median",
                                          "attack_latency_median_ms", "defense_latency_median_ms",
                                          "attack_plus_defense_latency_median_ms_derived", "adaptive_selected_k_mean",
                                          "n_degenerate_included", "na_reasons")},
                     "asr_reduction_pp_vs_none": d4[k]["asr_reduction_pp_vs_none"],
                     "clean_degradation_pp_same_defense_fec": c["clean_degradation_pp"],
                     "clean_defense_misclassification_pct_of_clean_correct":
                         dd.get("clean_defense_misclassification_pct_of_clean_correct", ""),
                     "n_success_after_where_clean_defense_also_wrong":
                         dd.get("n_success_after_where_clean_defense_also_wrong", ""),
                     "share_of_success_after_explained_by_clean_defense_pct":
                         dd.get("share_of_success_after_explained_by_clean_defense_pct", ""),
                     "source_file": f"{R4}/profile_summary.csv;{R4}/defense_summary.csv;"
                                    f"{R5}/defense_clean_summary.csv;{R5}/attack_run_decomposition.csv"})
    rows.sort(key=lambda r: (r["attack"], r["profile"], r["fec"], DEF.index(r["defense"])))
    require(len(rows) == 12 * 2 * 6, f"expected 144 attack x defense rows, found {len(rows)}")
    require({r["defense"] for r in rows} == set(DEF), "attack x defense matrix lacks a defense branch")
    wr(OUT / "data" / "attack_defense_summary.csv", rows)
    D["AD"] = {(r["attack"], r["profile"], r["fec"], r["defense"]): r for r in rows}
    D["ad_rows"] = rows

    rep = [("fgsm", "eps0.03"), ("pgd", "eps0.03_steps10"), ("pgd", "eps0.03_steps7"), ("pgd", "eps0.03_steps5"),
           ("cw", "c1_steps20_lr0.01"), ("cw", "c1_steps15_lr0.01"), ("cw", "c1_steps10_lr0.01"),
           ("ead", "bss9_maxit100"), ("ead", "bss5_maxit10")]
    rows = []
    for atk, prof in rep:
        for fec in ("off", "on"):
            for dn in DEF:
                a = D["AD"][(atk, prof, fec, dn)]
                c = D["CLEAN"][(dn, fec)]
                rows.append({"attack": atk, "profile": prof, "fec": fec, "defense": dn, "defense_label": DEF_LABEL[dn],
                             "n_clean_correct": a["n_clean_correct"],
                             "conditional_asr_before_defense_pct": a["conditional_asr_before_defense_pct"],
                             "conditional_asr_after_defense_pct": a["conditional_asr_after_defense_pct"],
                             "recovery_rate_pct": a["recovery_rate_pct"],
                             "clean_accuracy_after_defense_pct": c["clean_accuracy_after_defense_pct"],
                             "clean_degradation_pp": c["clean_degradation_pp"],
                             "share_of_success_after_explained_by_clean_defense_pct":
                                 a["share_of_success_after_explained_by_clean_defense_pct"],
                             "oracle_crc_given_attack_success_before_pct":
                                 a["oracle_crc_given_attack_success_before_pct"],
                             "oracle_frame_correct_given_attack_success_before_pct":
                                 a["oracle_frame_correct_given_attack_success_before_pct"],
                             "clean_oracle_frame_correct_after_defense_pct": c["oracle_frame_correct_pct"],
                             "attack_latency_median_ms": a["attack_latency_median_ms"],
                             "defense_latency_median_ms": a["defense_latency_median_ms"],
                             "attack_plus_defense_latency_median_ms_derived":
                                 a["attack_plus_defense_latency_median_ms_derived"],
                             "source_file": "data/attack_defense_summary.csv;data/clean_defense_summary.csv"})
    wr(OUT / "data" / "defense_tradeoff_summary.csv", rows)
    D["rep_profiles"] = rep

    rows = []
    for cond, rel in (("attacked", f"{R4}/adaptive_k_selection_summary.csv"),
                      ("clean", f"{R5}/adaptive_k_selection_summary.csv")):
        for r in rd(rel):
            rows.append({"input_condition": cond, "grouping": r["grouping"], "attack": r.get("attack", ""),
                         "profile": r.get("profile", ""), "true_modulation": r.get("true_modulation", ""),
                         "fec": fec_label(r["effective_fec"]),
                         **{k: r[k] for k in ("n", "n_wideband_quantize", "n_narrowband_topk", "selected_k_min",
                                              "selected_k_median", "selected_k_mean", "selected_k_max", "n_k_capped",
                                              "selected_k_histogram", "snr_est_db_median", "snr_est_db_min",
                                              "snr_est_db_max", "snr_source")},
                         "injected_snr_db": 18, "source_file": rel})
    wr(OUT / "data" / "adaptive_k_selection_summary.csv", rows)
    D["AK"] = rows

    rows = []
    for r in rd(f"{R5}/modulation_summary.csv"):
        rows.append({"table": "clean_defense", "attack": "", "profile": "", "defense": r["defense"],
                     "fec": fec_label(r["effective_fec"]), "modulation": r["true_modulation"], "n_total": r["n_total"],
                     "clean_accuracy_none_pct": r["clean_accuracy_none_pct"],
                     "clean_accuracy_after_defense_pct": r["clean_accuracy_after_defense_pct"],
                     "clean_degradation_pp": r["clean_degradation_pp"], "oracle_crc_pass_pct": r["oracle_crc_pass_pct"],
                     "oracle_frame_correct_pct": r["oracle_frame_correct_pct"],
                     "predicted_frame_correct_pct": r["predicted_frame_correct_pct"],
                     "adaptive_selected_k_mean": r["adaptive_selected_k_mean"],
                     "source_file": f"{R5}/modulation_summary.csv"})
    for r in rd(f"{R4}/modulation_summary.csv"):
        rows.append({"table": "attack_defense", "attack": r["attack"], "profile": r["profile"], "defense": r["defense"],
                     "fec": fec_label(r["effective_fec"]), "modulation": r["true_modulation"], "n_total": r["n_total"],
                     "n_clean_correct": r["n_clean_correct"],
                     "conditional_asr_before_defense_pct": r["conditional_asr_before_defense_pct"],
                     "conditional_asr_after_defense_pct": r["conditional_asr_after_defense_pct"],
                     "recovery_rate_pct": r["recovery_rate_pct"],
                     "oracle_crc_given_attack_success_before_pct": r["oracle_crc_given_attack_success_before_pct"],
                     "oracle_frame_correct_pct": r["oracle_frame_correct_pct"],
                     "predicted_frame_correct_pct": r["predicted_frame_correct_pct"],
                     "adaptive_selected_k_mean": r["adaptive_selected_k_mean"], "na_reasons": r["na_reasons"],
                     "source_file": f"{R4}/modulation_summary.csv"})
    for r in rd(f"{R3}/modulation_summary.csv"):
        rows.append({"table": "crc_fec_reduced_iteration", "attack": r["attack"], "profile": r["profile"],
                     "defense": "none", "fec": fec_label(r["fec_enabled"]), "modulation": r["modulation"],
                     "n_total": r["n_total"], "payload_degenerate": r["payload_degenerate"],
                     "n_clean_correct": r["n_clean_correct"],
                     "conditional_asr_before_defense_pct": r["conditional_asr_pct"],
                     "oracle_crc_given_attack_success_before_pct": r["oracle_crc_given_attack_success_pct"],
                     "predicted_crc_given_attack_success_pct": r["predicted_crc_given_attack_success_pct"],
                     "predicted_crc_false_accept_n": r["predicted_crc_false_accept_n"], "na_reasons": r["na_reasons"],
                     "source_file": f"{R3}/modulation_summary.csv"})
    wr(OUT / "data" / "modulation_summary.csv", rows)
    D["MOD"] = rows

    rows = []
    for r in amc:
        rows.append({"scope": "amc_only_attack_generation", "attack": r["attack"], "profile": r["profile"], "fec": "",
                     "defense": "", "n": r["latency_n_calls"], "median_ms": r["latency_median_ms"],
                     "mean_ms": r["latency_mean_ms"], "p95_ms": r["latency_p95_ms"],
                     "metric": "attack_generation_latency", "source_file": f"{R1}/profile_summary.csv"})
    for r in c3:
        rows.append({"scope": "crc_fec_attack_generation", "attack": r["attack"], "profile": r["profile"],
                     "fec": fec_label(r["fec_enabled"]), "defense": "", "n": r["n_total"],
                     "median_ms": r["latency_median_ms"], "mean_ms": r["latency_mean_ms"], "p95_ms": r["latency_p95_ms"],
                     "metric": "attack_generation_latency", "source_file": f"{R3}/profile_summary.csv"})
    for r in rd(f"{R4}/latency_summary.csv"):
        if r["scope"] == "pooled_all_attack_inputs":
            rows.append({"scope": "defense_on_attacked_input", "defense": r["defense"], "n": r["n"],
                         "median_ms": r["defense_latency_median_ms"], "mean_ms": r["defense_latency_mean_ms"],
                         "p95_ms": r["defense_latency_p95_ms"], "metric": "defense_latency",
                         "source_file": f"{R4}/latency_summary.csv"})
        elif r["defense"] != "none":
            rows.append({"scope": "attack_plus_defense_derived", "attack": r["attack"], "profile": r["profile"],
                         "fec": fec_label(r["effective_fec"]), "defense": r["defense"], "n": r["n"],
                         "median_ms": r["attack_plus_defense_latency_median_ms_derived"],
                         "p95_ms": r["attack_plus_defense_latency_p95_ms_derived"],
                         "metric": "attack_plus_defense_latency (composite, derived)",
                         "source_file": f"{R4}/latency_summary.csv"})
    for r in rd(f"{R5}/latency_summary.csv"):
        if r["grouping"] == "defense":
            rows.append({"scope": "defense_on_clean_input", "defense": r["defense"], "n": r["n"],
                         "median_ms": r["defense_latency_median_ms"], "mean_ms": r["defense_latency_mean_ms"],
                         "p95_ms": r["defense_latency_p95_ms"], "metric": "defense_latency",
                         "source_file": f"{R5}/latency_summary.csv"})
    wr(OUT / "data" / "latency_summary.csv", rows)
    D["LAT"] = rows

    e = m1["environment"]
    m2, m3, m4, m5 = (rj(f"{R}/manifest.json") for R in (R2, R3, R4, R5))
    D["m1"], D["m3"], D["m4"] = m1, m3, m4
    cfg = [
        ("platform", "all", f"{e['dmi_product_name']} ({e['hypervisor_vendor']})", R1),
        ("cpu_model", "all", e["cpu_model"], R1),
        ("python", "all", e["python"].split()[0], R1),
        ("torch", "all", e["torch"], R1),
        ("torchattacks", "all", e["torchattacks"], R1),
        ("numpy", "all", e["numpy"], R1),
        ("torch_intra_op_threads", "R1/R3/R4/R5",
         "/".join(str(m["environment"]["torch_num_threads"]) for m in (m1, m3, m4, m5)), R1),
        ("torch_inter_op_threads", "R1/R3/R4/R5",
         "/".join(str(m["environment"]["torch_num_interop_threads"]) for m in (m1, m3, m4, m5)), R1),
        ("repo_head", "all", m1["git"]["head"], R1),
        ("adversarial_rf_commit", "all", m1["git"]["adversarial_rf_head"], R1),
        ("checkpoint_sha256", "all", m1["inputs"]["checkpoint_sha256"], R1),
        ("dataset_sha256", "AMC-only (R1)", m1["inputs"]["dataset_sha256"], R1),
        ("attack_adapter_options", "R1/R3/R4", m1["mkldnn_options"]["adapter"], R1),
        ("amc_samples", "R1", "88 = 11 modulations x SNR {-20,-6,6,18} dB x sample_index {0,1}; B=1", R1),
        ("amc_latency_protocol", "R1", "3 untimed warm-up calls per profile pass; 4 replications (352 calls per "
                                       "profile); profile order reversed on odd replications", R1),
        ("amc_usable_criterion", "R1", "conditional ASR >= 0.9 x formal 2,200-sample CPU baseline", R1),
        ("synthetic_bursts", "R3/R4/R5", "8 digital modulations x SNR 18 dB x FEC requested {off,on} x 25 bursts", R3),
        ("synthetic_burst_params", "R3/R4/R5", json.dumps(m3["burst_params"]), R3),
        ("soft_decision_viterbi", "R3/R4/R5", m3["soft_demod"], R3),
        ("crc_usable_criterion", "R3", m3["usable_rule"], R3),
        ("defense_fixed", "R4/R5", m4["defenses"]["topk10"].replace("10", "K"), R4),
        ("defense_adaptive", "R4/R5", m4["defenses"]["adaptive_k"], R4),
        ("adaptive_k_default_parameters", "R4/R5", json.dumps(m4["adaptive_k_default_parameters"]), R4),
        ("adaptive_k_snr_source", "R4/R5", m4["adaptive_k_snr_source"], R4),
        ("defense_evaluation", "R4", m4["evaluation"], R4),
        ("seed_rule", "R3/R4/R5", m4["seed_rule"], R4),
        ("integration_reference", "R2", json.dumps(m2["reference"]), R2),
    ]
    wr(OUT / "data" / "experiment_configuration.csv",
       [{"item": a, "scope": b, "value": c, "source_file": f"{d}/manifest.json"} for a, b, c, d in cfg])

    iv = rj(f"{R2}/validation.json")
    pv = iv["v1_parity"]
    rows = [
        {"check": "overall", "raw_count": "", "denominator": "", "value": iv["overall"]},
        {"check": "v1_tx_identical_bursts", "raw_count": pv["tx_identical"], "denominator": pv["n_bursts"],
         "value": pv["tx_identical"] == pv["n_bursts"]},
        {"check": "v1_encoded_bits_ok_bursts", "raw_count": pv["encoded_bits_ok"], "denominator": pv["n_bursts"],
         "value": pv["encoded_bits_ok"] == pv["n_bursts"]},
        {"check": "v1_decode_crc_identical_comparisons", "raw_count": pv["decode_identical"],
         "denominator": pv["decode_comparisons"], "value": pv["decode_identical"] == pv["decode_comparisons"]},
        {"check": "v1_family_mismatch_legacy_exception_confirmed",
         "raw_count": pv["family_mismatch_confirmed_legacy_exception"], "denominator": "", "value": ""},
        {"check": "v1_integrated_vs_legacy_window_divergence_fsk",
         "raw_count": pv["default_mode_divergence_from_legacy"].get("fsk/perturbed", 0), "denominator": "",
         "value": "all divergences are FSK bursts with a modified window"},
        {"check": "v2_noiseless_oracle_crc_100pct", "raw_count": "", "denominator": "",
         "value": iv["checks"]["v2_noiseless_oracle_crc_100pct"]},
        {"check": "v3_attack_chain_complete", "raw_count": "", "denominator": "",
         "value": iv["checks"]["v3_attack_chain_complete"]},
        {"check": "v4_fail_closed_tests_passed",
         "raw_count": sum(t["pass"] for t in iv["v4_fail_closed"]["tests"].values()),
         "denominator": len(iv["v4_fail_closed"]["tests"]), "value": iv["checks"]["v4_fail_closed"]},
    ]
    for s_ in iv["v2_clean_chain"]:
        if s_["snr_db"] == 18:
            rows.append({"check": f"v2_snr18_clean_oracle_crc_pass_pct[{s_['modulation']},"
                                  f"fec={'on' if s_['fec_enabled'] else 'off'}]",
                         "raw_count": "", "denominator": s_["n"], "value": s_["clean_oracle_crc_pass_pct"]})
    for r in rows:
        r["source_file"] = f"{R2}/validation.json"
    wr(OUT / "data" / "crc_fec_integration_validation_summary.csv", rows)
    D["IV"] = iv
    D["INT"] = {r["check"]: r for r in rows}
    return D


# ======================================================================================== figures
def short(p):
    if p.startswith("bss"):
        b, m = p[3:].split("_maxit")
        return f"bss {b} x max_it {m}"
    return "steps " + p.split("steps")[1].split("_")[0]


BASEP = [("fgsm", "eps0.03", "FGSM eps 0.03"), ("pgd", "eps0.03_steps10", "PGD steps 10"),
         ("cw", "c1_steps20_lr0.01", "CW steps 20"), ("ead", "bss9_maxit100", "EAD 9 x 100"),
         ("ead", "bss5_maxit10", "EAD 5 x 10")]


def build_figures(D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
    c_b, c_r = "#4C72B0", "#DD8452"
    AMC, AD, CLEAN = D["AMC"], D["AD"], D["CLEAN"]
    fs, fg = OUT / "figure_sources", OUT / "figures"

    # Figure 1
    f1 = []
    for label, bg, bp, rg, rp in (("PGD", "pgd", "eps0.03_steps10", "pgd", "eps0.03_steps7"),
                                  ("CW", "cw", "c1_steps20_lr0.01", "cw", "c1_steps15_lr0.01"),
                                  ("EAD", "ead_phase1", "bss9_maxit100", "ead_phase2", "bss5_maxit10")):
        for setting, g, p in (("baseline", bg, bp), ("reduced", rg, rp)):
            a = AMC[(g, p)]
            f1.append({"attack": label, "setting": setting, "group": g, "profile": p,
                       "conditional_asr_pct": float(a["conditional_asr_pct"]), "n_attack_success": a["n_attack_success"],
                       "n_clean_correct": a["n_clean_correct"], "usable_threshold_pct": float(a["usable_threshold_pct"]),
                       "latency_median_ms": float(a["latency_median_ms"]),
                       "latency_reduction_pct": float(a["latency_reduction_pct"]) if setting == "reduced" else 0.0,
                       "source_csv": "data/amc_reduced_iteration_summary.csv"})
    wr(fs / "fig1_reduced_iteration_asr_latency.csv", f1)
    fig = plt.figure(figsize=(12, 3.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.6, 1, 1, 1])
    ax = fig.add_subplot(gs[0])
    atks = ["PGD", "CW", "EAD"]
    for i, (st, col) in enumerate((("baseline", c_b), ("reduced", c_r))):
        v = [next(r for r in f1 if r["attack"] == a and r["setting"] == st)["conditional_asr_pct"] for a in atks]
        b = ax.bar(np.arange(3) + (i - .5) * .36, v, .36, color=col, label=st)
        for bb, y in zip(b, v):
            ax.text(bb.get_x() + bb.get_width() / 2, y + 1, f"{y:.1f}", ha="center", fontsize=8)
    for j, a in enumerate(atks):
        t = next(r for r in f1 if r["attack"] == a)["usable_threshold_pct"]
        ax.hlines(t, j - .42, j + .42, colors="k", linestyles="--", lw=1)
    ax.plot([], [], "k--", lw=1, label="90% criterion")
    ax.set_xticks(range(3)); ax.set_xticklabels(atks); ax.set_ylim(0, 115); ax.set_ylabel("Conditional ASR (%)")
    ax.set_title("(a) AMC conditional ASR", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    for k, a in enumerate(atks):
        ax2 = fig.add_subplot(gs[k + 1])
        rr = [r for r in f1 if r["attack"] == a]
        v = [r["latency_median_ms"] for r in rr]
        b = ax2.bar([0, 1], v, .6, color=[c_b, c_r])
        for bb, y in zip(b, v):
            ax2.text(bb.get_x() + bb.get_width() / 2, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels([short(r["profile"]).replace(" x ", " x\n") for r in rr], fontsize=7.5)
        ax2.set_ylim(0, max(v) * 1.2)
        ax2.set_title(f"({'bcd'[k]}) {a} latency (-{rr[1]['latency_reduction_pct']:.1f}%)", fontsize=10)
        if k == 0:
            ax2.set_ylabel("Attack-generation latency, median (ms)")
    fig.suptitle("Figure 1. RadioML AMC-only: baseline vs reduced-iteration setting "
                 "(x86 VM, B=1, 352 calls per profile)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(fg / "fig1_reduced_iteration_asr_latency.png", dpi=180); plt.close(fig)

    # Figure 2
    f2 = []
    for dn in DEF[1:]:
        for fec in ("off", "on"):
            c = CLEAN[(dn, fec)]
            f2.append({"panel": "clean_degradation", "defense": dn, "defense_label": DEF_LABEL[dn], "fec": fec,
                       "series": "clean", "value": float(c["clean_degradation_pp"]), "unit": "pp",
                       "raw_count": c["n_defense_induced_misclassification"], "denominator": c["n_total"],
                       "source_csv": "data/clean_defense_summary.csv"})
    for atk, prof, lab in BASEP:
        for dn in DEF[1:]:
            a = AD[(atk, prof, "off", dn)]
            f2.append({"panel": "recovery_fec_off", "defense": dn, "defense_label": DEF_LABEL[dn], "fec": "off",
                       "series": lab, "attack": atk, "profile": prof, "value": float(a["recovery_rate_pct"]),
                       "unit": "%", "raw_count": a["n_defense_recovery"],
                       "denominator": a["n_attack_success_before_defense"],
                       "source_csv": "data/attack_defense_summary.csv"})
    wr(fs / "fig2_defense_recovery_clean_degradation.csv", f2)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    d5 = DEF[1:]
    x = np.arange(len(d5))
    for i, lab in enumerate([b[2] for b in BASEP]):
        v = [next(r["value"] for r in f2 if r["panel"] == "recovery_fec_off" and r["series"] == lab
                  and r["defense"] == d) for d in d5]
        a1.bar(x + (i - 2) * .16, v, .16, label=lab, color=plt.cm.tab10(i))
    a1.set_xticks(x); a1.set_xticklabels([DEF_LABEL[d] for d in d5]); a1.set_ylim(0, 80)
    a1.set_ylabel("Recovery rate (% of attack successes)"); a1.legend(fontsize=7.5, ncol=2, frameon=False)
    a1.set_title(f"(a) Attack recovery, FEC off (n_clean_correct = {AD[('pgd', 'eps0.03_steps10', 'off', 'none')]['n_clean_correct']})",
                 fontsize=10)
    for i, fec in enumerate(("off", "on")):
        v = [next(r["value"] for r in f2 if r["panel"] == "clean_degradation" and r["fec"] == fec
                  and r["defense"] == d) for d in d5]
        b = a2.bar(x + (i - .5) * .36, v, .36, color=[c_b, c_r][i], label=f"FEC {fec}")
        for bb, y in zip(b, v):
            a2.text(bb.get_x() + bb.get_width() / 2, y + .8, f"{y:.1f}", ha="center", fontsize=8)
    a2.set_xticks(x); a2.set_xticklabels([DEF_LABEL[d] for d in d5]); a2.set_ylim(0, 66)
    a2.set_ylabel("Clean accuracy degradation (pp)"); a2.legend(fontsize=8, frameon=False)
    a2.set_title("(b) Defense-induced clean AMC degradation (no attack)", fontsize=10)
    fig.suptitle("Figure 2. Synthetic TX/RX, SNR 18 dB: attack recovery and clean degradation per defense",
                 fontsize=10.5)
    fig.tight_layout(); fig.savefig(fg / "fig2_defense_recovery_clean_degradation.png", dpi=180); plt.close(fig)

    # Figure 3
    f3 = []
    for fec in ("off", "on"):
        for dn in DEF:
            c = CLEAN[(dn, fec)]
            f3.append({"fec": fec, "defense": dn, "defense_label": DEF_LABEL[dn], "series": "clean burst",
                       "metric": "oracle_frame_correct_pct", "value": float(c["oracle_frame_correct_pct"]),
                       "denominator": c["n_total"], "source_csv": "data/clean_defense_summary.csv"})
            for atk, prof, lab in BASEP:
                a = AD[(atk, prof, fec, dn)]
                f3.append({"fec": fec, "defense": dn, "defense_label": DEF_LABEL[dn], "series": lab, "attack": atk,
                           "profile": prof, "metric": "oracle_frame_correct_given_attack_success_before_pct",
                           "value": float(a["oracle_frame_correct_given_attack_success_before_pct"]),
                           "denominator": a["n_attack_success_before_defense"],
                           "source_csv": "data/attack_defense_summary.csv"})
    wr(fs / "fig3_oracle_frame_recoverability.csv", f3)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    series = ["clean burst"] + [b[2] for b in BASEP]
    for ax, fec in zip(axes, ("off", "on")):
        x = np.arange(len(DEF))
        for i, s in enumerate(series):
            v = [next(r["value"] for r in f3 if r["fec"] == fec and r["series"] == s and r["defense"] == d)
                 for d in DEF]
            ax.bar(x + (i - 2.5) * .14, v, .14, label=s, color="#4D4D4D" if i == 0 else plt.cm.tab10(i - 1))
        ax.set_xticks(x); ax.set_xticklabels([DEF_LABEL[d] for d in DEF]); ax.set_ylim(0, 105)
        ax.set_title(f"({'ab'[fec == 'on']}) FEC {fec}", fontsize=10)
    axes[0].set_ylabel("Oracle frame correct (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("Figure 3. Oracle-receiver payload recoverability: clean bursts and attack-success bursts "
                 "(CRC pass and bit-exact frame)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(fg / "fig3_oracle_frame_recoverability.png", dpi=180)
    plt.close(fig)

    # Figure 4 (from raw rows)
    f4, raw_sets = [], {}
    for cond, rel in (("clean", f"{R5}/raw_samples.csv"), ("attacked", f"{R4}/raw_samples.csv")):
        USED.add(rel)
        with (REPO / rel).open(newline="") as f:
            for r in csv.DictReader(f):
                if r["defense"] != "adaptive_k" or r["payload_degenerate"] == "True":
                    continue
                raw_sets.setdefault((cond, fec_label(r["effective_fec"])), []).append(
                    (int(r["adaptive_selected_k"]) if r["adaptive_selected_k"] else None,
                     float(r["adaptive_snr_db"]) if r["adaptive_snr_db"] else None, r["adaptive_route"]))
    sbins = np.arange(0, 20.5, 1.0)
    for (cond, fec), vals in sorted(raw_sets.items()):
        ks = Counter(v[0] for v in vals if v[0] is not None)
        for k in range(10, 41):
            f4.append({"panel": "selected_k", "input_condition": cond, "fec": fec, "bin": k, "count": ks.get(k, 0),
                       "n": len(vals), "n_wideband": sum(v[2] == "wideband_quantize" for v in vals),
                       "source_csv": f"{R5 if cond == 'clean' else R4}/raw_samples.csv"})
        h, _ = np.histogram([v[1] for v in vals if v[1] is not None], bins=sbins)
        for lo, cnt in zip(sbins[:-1], h):
            f4.append({"panel": "snr_est_db", "input_condition": cond, "fec": fec, "bin": f"{lo:.0f}-{lo + 1:.0f}",
                       "count": int(cnt), "n": len(vals),
                       "source_csv": f"{R5 if cond == 'clean' else R4}/raw_samples.csv"})
    wr(fs / "fig4_adaptive_k_selection_snr.csv", f4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4))
    for j, fec in enumerate(("off", "on")):
        ax = axes[0, j]
        for cond, col, off in (("clean", c_b, -.2), ("attacked", c_r, .2)):
            rr = [r for r in f4 if r["panel"] == "selected_k" and r["input_condition"] == cond and r["fec"] == fec]
            n = rr[0]["n"]
            ax.bar([r["bin"] + off for r in rr], [100 * r["count"] / n for r in rr], .4, color=col,
                   label=f"{cond} (n={n})")
        ax.set_xlabel("Selected K"); ax.set_ylabel("Share of samples (%)"); ax.legend(fontsize=8, frameon=False)
        ax.set_title(f"({'ab'[j]}) Adaptive-K selected K, FEC {fec}", fontsize=10)
        ax = axes[1, j]
        for cond, col, off in (("clean", c_b, .3), ("attacked", c_r, .7)):
            rr = [r for r in f4 if r["panel"] == "snr_est_db" and r["input_condition"] == cond and r["fec"] == fec]
            n = rr[0]["n"]
            ax.bar([int(r["bin"].split("-")[0]) + off for r in rr], [100 * r["count"] / n for r in rr], .4,
                   color=col, label=cond)
        ax.axvline(18, color="k", ls="--", lw=1)
        ax.text(17.8, ax.get_ylim()[1] * .9, "injected SNR 18 dB", ha="right", fontsize=8)
        ax.set_xlim(0, 20); ax.set_xlabel("Blind SNR estimate (dB, 1 dB bins)"); ax.set_ylabel("Share of samples (%)")
        ax.legend(fontsize=8, frameon=False, loc="center right")
        ax.set_title(f"({'cd'[j]}) Adaptive-K blind SNR estimate, FEC {fec}", fontsize=10)
    fig.suptitle("Figure 4. Adaptive-K decisions (production adaptive_k_v2_snr_defense; "
                 "all samples routed narrowband)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(fg / "fig4_adaptive_k_selection_snr.png", dpi=180); plt.close(fig)
    D["raw_ak"] = raw_sets

    # Figure 5
    f5 = []
    for atk, prof, lab in BASEP:
        for dn in DEF[1:]:
            a = AD[(atk, prof, "off", dn)]
            f5.append({"panel": "tradeoff_fec_off", "series": lab, "attack": atk, "profile": prof, "defense": dn,
                       "defense_label": DEF_LABEL[dn],
                       "clean_degradation_pp": float(CLEAN[(dn, "off")]["clean_degradation_pp"]),
                       "recovery_rate_pct": float(a["recovery_rate_pct"]),
                       "conditional_asr_after_defense_pct": float(a["conditional_asr_after_defense_pct"]),
                       "source_csv": "data/attack_defense_summary.csv;data/clean_defense_summary.csv"})
    for r in D["LAT"]:
        if r["scope"] in ("defense_on_attacked_input", "defense_on_clean_input"):
            f5.append({"panel": "defense_latency", "series": r["scope"], "defense": r["defense"],
                       "defense_label": DEF_LABEL[r["defense"]], "defense_latency_median_ms": float(r["median_ms"]),
                       "defense_latency_p95_ms": float(r["p95_ms"]), "n": r["n"],
                       "source_csv": "data/latency_summary.csv"})
    for atk, prof, lab in BASEP:
        f5.append({"panel": "attack_latency_reference", "series": lab, "attack": atk, "profile": prof,
                   "attack_latency_median_ms": float(AD[(atk, prof, "off", "none")]["attack_latency_median_ms"]),
                   "source_csv": "data/attack_defense_summary.csv"})
    wr(fs / "fig5_defense_tradeoff.csv", f5)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    marks = {"topk10": "X", "topk20": "o", "topk30": "s", "topk40": "^", "adaptive_k": "D"}
    for i, lab in enumerate([b[2] for b in BASEP]):
        for r in [r for r in f5 if r["panel"] == "tradeoff_fec_off" and r["series"] == lab]:
            a1.scatter(r["clean_degradation_pp"], r["recovery_rate_pct"], marker=marks[r["defense"]],
                       color=plt.cm.tab10(i), s=38, zorder=3)
    for dn, mk in marks.items():
        a1.scatter([], [], marker=mk, color="k", label=DEF_LABEL[dn])
    for i, lab in enumerate([b[2] for b in BASEP]):
        a1.scatter([], [], marker="o", color=plt.cm.tab10(i), label=lab)
    a1.set_xlabel("Clean accuracy degradation (pp), FEC off"); a1.set_ylabel("Recovery rate (%), FEC off")
    a1.set_title("(a) Recovery vs defense-induced clean degradation", fontsize=10)
    a1.legend(fontsize=7.5, ncol=1, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    a1.set_xlim(-2, 62); a1.set_ylim(0, 75)
    x = np.arange(5)
    for i, (sc, col, lab) in enumerate((("defense_on_clean_input", c_b, "clean input"),
                                        ("defense_on_attacked_input", c_r, "attacked input"))):
        v = [next(r["defense_latency_median_ms"] for r in f5 if r["panel"] == "defense_latency"
                  and r["series"] == sc and r["defense"] == d) for d in DEF[1:]]
        b = a2.bar(x + (i - .5) * .36, v, .36, color=col, label=lab)
        for bb, y in zip(b, v):
            a2.text(bb.get_x() + bb.get_width() / 2, y + .01, f"{y:.2f}", ha="center", fontsize=7.5)
    a2.set_xticks(x); a2.set_xticklabels([DEF_LABEL[d] for d in DEF[1:]])
    a2.set_ylabel("Defense latency, median (ms)"); a2.legend(fontsize=8, frameon=False)
    a2.set_title("(b) Defense latency (x86 VM)", fontsize=10)
    fig.suptitle("Figure 5. Defense trade-off: recovery, clean degradation and defense latency "
                 "(attack-generation latency: Section 8)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(fg / "fig5_defense_tradeoff.png", dpi=180); plt.close(fig)


# ======================================================================================== report
def f(x):
    return float(x)


def num(v, d=1):
    """Fixed-decimal number; 100 is written without decimals."""
    v = f(v)
    return "100" if abs(v - 100.0) < 1e-12 else f"{v:.{d}f}"


def pc(v, d=1):
    return num(v, d) + "%"


def ms(v, d=2):
    return f"{f(v):.{d}f}"


def rng(vals, d=1, unit="%"):
    lo, hi = min(map(f, vals)), max(map(f, vals))
    return f"{num(lo, d)}{unit}–{num(hi, d)}{unit}"


def kfmt(v):
    v = f(v)
    return str(int(v)) if v == int(v) else f"{v:.1f}"


def build_report(D):
    AMC, C3, MIN, CLEAN, AD, AK, MOD, LAT, INT = (D[k] for k in ("AMC", "C3", "MIN", "CLEAN", "AD", "AK", "MOD",
                                                               "LAT", "INT"))
    m1, m4 = D["m1"], D["m4"]
    e = m1["environment"]
    A = lambda g, p: AMC[(g, p)]  # noqa: E731
    ad = lambda a, p, fe, d: AD[(a, p, fe, d)]  # noqa: E731
    cl = lambda d, fe: CLEAN[(d, fe)]  # noqa: E731
    out = []
    w = out.append

    # ---- AMC values
    pg = {s: A("pgd", f"eps0.03_steps{s}") for s in (1, 2, 3, 5, 7, 10)}
    cw = {s: A("cw", f"c1_steps{s}_lr0.01") for s in (1, 2, 3, 5, 10, 15, 20)}
    e1 = {m: A("ead_phase1", f"bss9_maxit{m}") for m in (10, 20, 30, 50, 75, 100)}
    e2 = {b: A("ead_phase2", f"bss{b}_maxit10") for b in (1, 3, 5, 9)}
    fg = {x: A("fgsm", f"eps{x}") for x in ("0.005", "0.01", "0.03", "0.05")}
    n_cc_amc = pg[10]["n_clean_correct"]
    thr = {k: f(v["usable_threshold_pct"]) for k, v in (("pgd", pg[10]), ("cw", cw[20]), ("ead", e1[100]))}
    fbase = {k: f(v["formal_baseline_conditional_asr_pct"]) for k, v in (("pgd", pg[10]), ("cw", cw[20]),
                                                                          ("ead", e1[100]))}
    require(pg[7]["usable_90pct"] == "True" and pg[5]["usable_90pct"] == "False", "PGD usable boundary changed")
    require(cw[15]["usable_90pct"] == "True" and cw[10]["usable_90pct"] == "False", "CW usable boundary changed")
    require(e2[5]["usable_90pct"] == "True" and e2[3]["usable_90pct"] == "False", "EAD usable boundary changed")
    require(all(v["usable_90pct"] == "True" for v in e1.values()), "EAD Phase 1 usable set changed")

    # ---- synthetic domain counts
    n_off, cc_off = cl("none", "off")["n_total"], cl("none", "off")["n_clean_correct_none"]
    n_on, cc_on = cl("none", "on")["n_total"], cl("none", "on")["n_clean_correct_none"]
    sl = lambda xs: "／".join(xs)  # noqa: E731

    w("# Reduced-Iteration 對抗攻擊、Payload Recoverability 與頻域防禦之評估\n")
    w("**攻擊**：FGSM、PGD、CW、EAD　｜　**分類器**：AWN　｜　**防禦**：Fixed Top-K（K = 10、20、30、40）、Adaptive-K　｜　"
      "**資料域**：RadioML 2016.10a（AMC-only）、synthetic TX/RX（CRC-8 與卷積碼 FEC）\n")
    w("> 本報告之數值均取自 `data/` 內 CSV；各 CSV 之每一列記錄其來源檔（`source_file`），來源檔之路徑、大小與 SHA-256 "
      "列於 `data/source_manifest.csv`。五個來源 run 之 validation 均為 PASS。本資料夾由 "
      "`scripts/build_reduced_iteration_crc_fec_defense_study.py` 自原始 results 重新產生。\n")

    w("## 1. 研究目的\n")
    w("本研究針對四種以受限擾動或降低擾動為設計目標的對抗攻擊（FGSM、PGD、CW、EAD），探討下列問題：\n")
    for i, q in enumerate([
        "降低 iterations／steps 後，攻擊對 AMC 之 conditional attack success rate（conditional ASR）是否維持於定義之效力準則內。",
        "reduced-iteration setting 可降低之 attack-generation latency。",
        "攻擊成功誤導 AMC 後，以真實調變解調之 oracle receiver 能否正確恢復 payload。",
        "FEC-enabled decoding 對 payload recoverability 之影響。",
        "Fixed Top-K（K10、K20、K30、K40）與 Adaptive-K 對 attack recovery 之影響。",
        "各防禦於無攻擊之 clean waveform 上造成之 AMC accuracy degradation。",
        "各防禦對 CRC pass 與 frame correctness 之影響。",
        "攻擊與防禦組合下 latency、攻擊效力、recovery 與 payload recoverability 之取捨。",
        "Adaptive-K 實際選擇之 K 與其 blind SNR estimate 之行為。"], start=1):
        w(f"{i}. **RQ{i}**：{q}")
    w("\nRQ1–RQ2 於 RadioML 2016.10a 之 AMC-only 資料域評估；RQ3–RQ9 需已知 payload bits，於 synthetic TX/RX 資料域評估。"
      "RadioML 2016.10a 不含 payload bits，無法用於 CRC 評估。\n")

    w("## 2. 實驗設計\n")
    w("### 2.1 攻擊設定\n")
    w(f"所有攻擊由 production `AttackAdapter`（torchattacks {e['torchattacks']}、per-sample min-max mapping、temperature 1.0、"
      "untargeted，以模型 clean prediction 為 label）產生，B = 1。每次攻擊呼叫前以該樣本之 formal seed 執行 `torch.manual_seed`。\n")
    w("| 攻擊 | 固定參數 | 掃描參數 | Baseline |\n|---|---|---|---|")
    w("| FGSM | N/A | eps ∈ {0.005, 0.01, 0.03, 0.05}；單步，無 iteration 可降低 | N/A |")
    w("| PGD | eps 0.03、alpha 0.0075（不隨 steps 調整）、random_start = True | steps ∈ {1, 2, 3, 5, 7, 10} | steps 10 |")
    w("| CW | c 1、kappa 0、lr 0.01 | steps ∈ {1, 2, 3, 5, 10, 15, 20} | steps 20 |")
    w("| EAD（EADL1） | lr 0.01、initial_const 1e-3、beta 1e-3、kappa 0、abort_early = True | Phase 1：binary_search_steps（bss）9，"
      "max_iterations ∈ {10, 20, 30, 50, 75, 100}；Phase 2：max_iterations 10，bss ∈ {1, 3, 5, 9} | bss 9 × max_iterations 100 |\n")
    w("防禦評估使用其中 12 個 profile：FGSM 四個 eps、PGD steps {5, 7, 10}、CW steps {10, 15, 20}、EAD bss 5 × max_iterations 10 "
      "與 bss 9 × max_iterations 100。\n")
    w("### 2.2 指標定義\n")
    w("- attack_success = clean_correct AND attacked_pred ≠ true_label；conditional ASR = n_attack_success／n_clean_correct。")
    w("- 防禦評估中：attack_success_before_defense 同上；attack_success_after_defense = clean_correct AND defended_pred ≠ "
      "true_label；defense_recovery = attack_success_before_defense AND defended_pred = true_label；recovery rate = "
      "n_recovery／n_attack_success_before_defense。")
    w("- Clean degradation = clean accuracy（無防禦）減 clean accuracy（防禦後），以 percentage points（pp）表示；防禦後誤判之 "
      "clean burst 稱為 defense-induced misclassification，不計為 attack success。")
    w("- Oracle receiver：以真實調變解調，評估 adversarial waveform 是否保留 payload recoverability。Predicted-AMC receiver："
      "以 AMC 預測調變解調，評估依賴 AMC 決策之接收流程。")
    w("- Oracle／predicted CRC pass：CRC-8 檢查通過。Frame correct：CRC pass 且 recovered frame 與傳送 frame bit-exact。"
      "CRC false accept：CRC pass 但 recovered frame 含 bit error，不視為 payload recovery。")
    w("- 分母為 0 之比例指標記為 NA，原因記錄於 `na_reasons` 欄。\n")
    w("### 2.3 Minimum usable setting 之準則\n")
    w(f"Minimum usable setting 指在目前掃描範圍與既定 efficacy criterion 下的最低可用設定：掃描範圍內 conditional ASR ≥ 90% × "
      f"baseline 之最少 iteration 設定。AMC-only 之 baseline 為正式 2,200 樣本 CPU conditional ASR（PGD {num(fbase['pgd'], 2)}%、"
      f"CW {num(fbase['cw'], 2)}%、EAD {num(fbase['ead'], 2)}%），門檻為 {num(thr['pgd'], 2)}%、{num(thr['cw'], 2)}%、"
      f"{num(thr['ead'], 1)}%。Synthetic TX/RX 之 baseline 為同資料域、同 FEC 設定之 baseline profile；joint oracle criterion "
      "另要求 oracle CRC｜attack success ≥ 90% × baseline。此設定不代表全域最小 iterations 或全域最小 perturbation。"
      f"2,200-sample baseline 作為既有正式效力基準，本輪 {pg[10]['n_total']}-sample sweep 用於在固定可重現子集上比較 "
      f"iteration 設定與 latency；兩者用途不同，因此不以 {pg[10]['n_total']}-sample 結果取代既有正式 baseline。\n")
    w("### 2.4 資料域\n")
    w("| 項目 | AMC-only（RadioML 2016.10a） | Synthetic TX/RX |\n|---|---|---|")
    w(f"| 樣本 | 88：11 種調變 × SNR {{−20, −6, 6, 18}} dB × sample_index {{0, 1}}；clean_correct {n_cc_amc} | "
      "8 種數位調變 × SNR 18 dB × FEC 請求 {off, on} × 25 bursts = 400 bursts |")
    w("| Payload | 無 | 隨機 payload → CRC-8 →（FEC）rate-1/2、K = 7 卷積碼 + block interleaver；16 symbols × 8 sps；"
      "soft-decision Viterbi |")
    w("| 有效 FEC | N/A | BPSK、CPFSK、GFSK 無法啟用 FEC（effective_fec = False） |")
    w("| Payload-degenerate | N/A | QPSK、PAM4 於 FEC on 時 payload 為 0 bits，保留於 raw 資料，排除於所有 aggregate 與 "
      "minimum selection |")
    w(f"| 有效樣本 | N/A | FEC off {n_off} bursts（clean_correct {cc_off}）；FEC on {n_on} bursts（8PSK、QAM16、QAM64；"
      f"clean_correct {cc_on}） |\n")
    p = m4["adaptive_k_default_parameters"]
    w("### 2.5 防禦設定\n")
    w("| 防禦 | 實作 |\n|---|---|\n| none | 輸入不變 |")
    w("| K10、K20、K30、K40 | `TopKAdapter(mode=\"fixed\")` → `fft_topk_denoise`：full complex FFT，I 與 Q 各保留幅度最大之 K 個 bin |")
    w("| Adaptive-K | `TopKAdapter(mode=\"adaptive\")` → production `adaptive_k_v2_snr_defense`（default parameters："
      + "、".join(f"{k} {v}" for k, v in p.items()) + "） |\n")
    w("防禦評估為 post-attack evaluation：每個 burst 與 profile 之攻擊僅產生一次 x_adv，同一份 x_adv（SHA-256 一致）送入六個防禦"
      "分支；攻擊非 defense-aware。Clean-only control 以同一批 clean burst（SHA-256 一致）送入六個防禦分支，不產生攻擊。"
      "Adaptive-K 之 per-sample 決策以 production helper functions 之 trace 取得，trace 輸出與 production 輸出於全部樣本 "
      "bit-identical。\n")
    w("### 2.6 可重現性設定\n")
    w("| 項目 | 設定 |\n|---|---|")
    w(f"| 平台 | {e['dmi_product_name']}，{e['cpu_model'].replace('11th Gen Intel(R) Core(TM) ', 'Intel Core ').split(' @')[0]}；"
      "latency 均為 x86 VM latency |")
    w(f"| 軟體 | Python {e['python'].split()[0]}、torch {e['torch']}、torchattacks {e['torchattacks']}、numpy {e['numpy']} |")
    w(f"| Threads | torch intra-op {e['torch_num_threads']}、inter-op {e['torch_num_interop_threads']} |")
    w(f"| Commits | repository `{m1['git']['head'][:8]}`；`external/adversarial-rf` `{m1['git']['adversarial_rf_head'][:8]}` |")
    ck = m1["inputs"]["checkpoint_sha256"]
    w(f"| Checkpoint | `2016.10a_AWN.pkl`，SHA-256 `{ck[:8]}…{ck[-4:]}` |")
    w(f"| AMC-only latency | 每 profile 4 replications × 88 = {pg[10]['latency_n_calls']} 次呼叫，profile 順序於奇數 replication 反轉 |")
    w("| Synthetic latency | 每 burst × profile 1 次計時呼叫；defense latency 為 `TopKAdapter.apply()` 呼叫時間 |\n")
    w("完整設定見 `data/experiment_configuration.csv`。\n")

    # ---- Section 3
    w("## 3. Reduced-Iteration Attack Evaluation（AMC-only）\n")
    w("### 3.1 Iteration sweep\n")
    w(f"**Table 1.** AMC-only iteration sweep（n_clean_correct = {n_cc_amc}；latency 為 median，降低比例相對 baseline，"
      "EAD 兩個 phase 均相對 bss 9 × max_iterations 100）。\n")
    w("| 攻擊 | 設定 | conditional ASR | 滿足準則 | Latency（ms） | Latency 降低 |\n|---|---|---|---|---|---|")
    w(f"| FGSM | eps 0.005／0.01／0.03／0.05 | {sl(pc(v['conditional_asr_pct']) for v in fg.values())} | N/A | "
      f"{sl(ms(v['latency_median_ms']) for v in fg.values())} | N/A |")
    lo = [pg[s] for s in (1, 2, 3, 5)]
    w(f"| PGD | steps 1／2／3／5 | {sl(pc(v['conditional_asr_pct']) for v in lo)} | 否 | "
      f"{sl(ms(v['latency_median_ms']) for v in lo)} | {sl(pc(v['latency_reduction_pct']) for v in lo)} |")
    w(f"| PGD | **steps 7** | **{pc(pg[7]['conditional_asr_pct'])}** | 是 | **{ms(pg[7]['latency_median_ms'])}** | "
      f"**{pc(pg[7]['latency_reduction_pct'])}** |")
    w(f"| PGD | steps 10（baseline） | {pc(pg[10]['conditional_asr_pct'])} | 是 | {ms(pg[10]['latency_median_ms'])} | N/A |")
    lo = [cw[s] for s in (1, 2, 3, 5, 10)]
    w(f"| CW | steps 1／2／3／5／10 | {sl(pc(v['conditional_asr_pct']) for v in lo)} | 否 | "
      f"{sl(ms(v['latency_median_ms']) for v in lo)} | {sl(pc(v['latency_reduction_pct']) for v in lo)} |")
    w(f"| CW | **steps 15** | **{pc(cw[15]['conditional_asr_pct'])}** | 是 | **{ms(cw[15]['latency_median_ms'])}** | "
      f"**{pc(cw[15]['latency_reduction_pct'])}** |")
    w(f"| CW | steps 20（baseline） | {pc(cw[20]['conditional_asr_pct'])} | 是 | {ms(cw[20]['latency_median_ms'])} | N/A |")
    lo = [e1[m] for m in (10, 20, 30, 50, 75)]
    require(len({v["conditional_asr_pct"] for v in lo}) == 1, "EAD Phase 1 ASR no longer uniform")
    w(f"| EAD | bss 9 × max_it 10／20／30／50／75 | {pc(lo[0]['conditional_asr_pct'])} | 是 | "
      f"{sl(ms(v['latency_median_ms']) for v in lo)} | {sl(pc(v['latency_reduction_pct']) for v in lo)} |")
    w(f"| EAD | bss 9 × max_it 100（baseline） | {pc(e1[100]['conditional_asr_pct'])} | 是 | "
      f"{ms(e1[100]['latency_median_ms'])} | N/A |")
    lo = [e2[1], e2[3]]
    w(f"| EAD | max_it 10 × bss 1／3 | {sl(pc(v['conditional_asr_pct']) for v in lo)} | 否 | "
      f"{sl(ms(v['latency_median_ms']) for v in lo)} | {sl(pc(v['latency_reduction_pct']) for v in lo)} |")
    w(f"| EAD | **max_it 10 × bss 5** | **{pc(e2[5]['conditional_asr_pct'])}** | 是 | **{ms(e2[5]['latency_median_ms'])}** | "
      f"**{pc(e2[5]['latency_reduction_pct'])}** |\n")
    w("### 3.2 結果\n")
    w("在 AMC-only 資料域與 90% 正式 baseline 準則下，於目前掃描範圍與既定 efficacy criterion 下的最低可用設定如下：\n")
    w(f"- PGD 由 10 steps 降至 7 steps，conditional ASR 由 {pc(pg[10]['conditional_asr_pct'])} 降至 "
      f"{pc(pg[7]['conditional_asr_pct'])}，attack-generation latency median 由 {ms(pg[10]['latency_median_ms'])} ms 降至 "
      f"{ms(pg[7]['latency_median_ms'])} ms（降低 {pc(pg[7]['latency_reduction_pct'])}）；5 steps 之 "
      f"{pc(pg[5]['conditional_asr_pct'])} 低於門檻 {num(thr['pgd'], 2)}%。")
    w(f"- CW 由 20 steps 降至 15 steps，conditional ASR 由 {pc(cw[20]['conditional_asr_pct'])} 降至 "
      f"{pc(cw[15]['conditional_asr_pct'])}，latency median 由 {ms(cw[20]['latency_median_ms'])} ms 降至 "
      f"{ms(cw[15]['latency_median_ms'])} ms（降低 {pc(cw[15]['latency_reduction_pct'])}）。")
    w(f"- EAD 由 bss 9 × max_iterations 100 降至 bss 5 × max_iterations 10，conditional ASR 由 "
      f"{pc(e1[100]['conditional_asr_pct'])} 維持於 {pc(e2[5]['conditional_asr_pct'])}，latency median 由 "
      f"{ms(e1[100]['latency_median_ms'])} ms 降至 {ms(e2[5]['latency_median_ms'])} ms（降低 "
      f"{pc(e2[5]['latency_reduction_pct'])}）。bss 3 之 conditional ASR 為 {pc(e2[3]['conditional_asr_pct'])}。")
    fl = [f(v["latency_median_ms"]) for v in fg.values()]
    w(f"- FGSM 為單步攻擊，無 iteration 可降低；latency median 介於 {min(fl):.2f}–{max(fl):.2f} ms。\n")
    w("![Figure 1](figures/fig1_reduced_iteration_asr_latency.png)\n")
    w("**Figure 1.** AMC-only：PGD、CW、EAD 之 baseline 與 reduced setting。(a) conditional ASR，虛線為 90% 準則；(b)–(d) "
      "attack-generation latency median。資料：`figure_sources/fig1_reduced_iteration_asr_latency.csv`。\n")

    # ---- Section 4
    w("## 4. CRC/FEC Payload Recoverability\n")
    w("### 4.1 Synthetic TX/RX chain 之整合驗證\n")
    w("Synthetic TX/RX chain 直接包裝 `external/adversarial-rf/util/synth_txrx.py` 之既有編解碼與調變實作，並沿用 AMC-only "
      "資料域之 `AWNModelAdapter` 與 `AttackAdapter`。\n")
    w("**Table 2.** 整合驗證（`data/crc_fec_integration_validation_summary.csv`）。\n")
    t = lambda k: INT[k]  # noqa: E731
    require(INT["overall"]["value"] == "PASS", "integration validation not PASS")
    require(all(INT[k]["value"] in (True, "True") for k in ("v1_tx_identical_bursts", "v1_encoded_bits_ok_bursts",
                                                           "v1_decode_crc_identical_comparisons",
                                                           "v2_noiseless_oracle_crc_100pct", "v3_attack_chain_complete",
                                                           "v4_fail_closed_tests_passed")), "integration check failed")
    snr18 = [r for k, r in INT.items() if k.startswith("v2_snr18")]
    require(all(f(r["value"]) == 100.0 for r in snr18), "SNR 18 clean oracle CRC not 100%")
    w("| 項目 | 結果 |\n|---|---|")
    w(f"| TX parity（8 調變 × SNR {{0, 10, 18, 100}} × FEC off/on × 10 seeds） | {t('v1_tx_identical_bursts')['raw_count']}/"
      f"{t('v1_tx_identical_bursts')['denominator']} bursts bit-identical |")
    w(f"| Encoded bits 與傳送 symbols 一致；FEC frame 經 Viterbi decoding 還原 | {t('v1_encoded_bits_ok_bursts')['raw_count']}/"
      f"{t('v1_encoded_bits_ok_bursts')['denominator']} |")
    w(f"| Decode／CRC parity（11 種 demod 調變 × clean／perturbed window × soft／hard Viterbi） | "
      f"{int(t('v1_decode_crc_identical_comparisons')['raw_count']):,}/{int(t('v1_decode_crc_identical_comparisons')['denominator']):,} |")
    w("| 無攻擊 clean chain（SNR 100 dB 與 18 dB）之 oracle CRC pass | 全部調變與 FEC 設定均為 100% |")
    w(f"| Fail-closed 測試 | {t('v4_fail_closed_tests_passed')['raw_count']}/{t('v4_fail_closed_tests_passed')['denominator']} |\n")
    w("既有 legacy CRC scripts 於 CPFSK 與 GFSK 之 attacked 情境中解調 clean window；整合後之 chain 解調被修改之 window，兩者"
      f"差異之 {t('v1_integrated_vs_legacy_window_divergence_fsk')['raw_count']} 筆均屬此情形。Legacy scripts 之 FSK attacked "
      "CRC 數值不納入本研究。\n")

    w("### 4.2 攻擊成功後之 oracle payload recoverability\n")
    w("**Table 3.** Synthetic TX/RX baseline profiles（`data/crc_fec_reduced_iteration_summary.csv`）。\n")
    w("| 攻擊 | FEC | conditional ASR | Oracle CRC｜attack success | Predicted CRC｜attack success | Latency median（ms） |\n"
      "|---|---|---|---|---|---|")
    for lab, g, pr in (("PGD steps 10", "pgd", "eps0.03_steps10"), ("CW steps 20", "cw", "c1_steps20_lr0.01"),
                       ("EAD bss 9 × max_it 100", "ead_phase1", "bss9_maxit100")):
        for fe in ("off", "on"):
            r = C3[(g, pr, fe)]
            w(f"| {lab} | {fe} | {pc(r['conditional_asr_pct'])}（{r['n_attack_success']}/{r['n_clean_correct']}） | "
              f"{pc(r['oracle_crc_given_attack_success_pct'])}（{r['n_attack_success_oracle_crc_pass']}/{r['n_attack_success']}） | "
              f"{pc(r['predicted_crc_given_attack_success_pct'])}（{r['n_attack_success_predicted_crc_pass']}/"
              f"{r['n_attack_success']}） | {ms(r['latency_median_ms'])} |")
    c3r = D["c3_rows"]
    orc = {fe: [r["oracle_crc_given_attack_success_pct"] for r in c3r if r["fec"] == fe
                and r["oracle_crc_given_attack_success_pct"] not in ("", "None")] for fe in ("off", "on")}
    for r in c3r:
        if r["oracle_crc_given_attack_success_pct"] not in ("", "None"):
            require(r["oracle_crc_given_attack_success_pct"] == r["oracle_frame_correct_given_attack_success_pct"],
                    "oracle CRC and oracle frame correct differ")
    require(sum(int(r["oracle_crc_false_accept_n"]) for r in c3r) == 0, "oracle CRC false accept present")
    w(f"\n於全部 reduced-iteration profiles（FGSM、PGD、CW、EAD），attack success 條件下之 oracle CRC pass 與 oracle frame "
      f"correct 相同：FEC off 為 {rng(orc['off'])}，FEC on 為 {rng(orc['on'])}；oracle receiver 無 CRC false accept。在 attack "
      "success 條件下，多數樣本仍可於 oracle receiver 下正確恢復 payload。\n")
    w("### 4.3 Predicted-AMC receiver 之系統行為\n")
    prc = [r["predicted_crc_given_attack_success_pct"] for r in c3r if r["predicted_crc_given_attack_success_pct"]
           not in ("", "None")]
    n_pp = sum(int(r["n_attack_success_predicted_crc_pass"]) for r in c3r)
    require(sum(int(r["n_attack_success_predicted_frame_correct"]) for r in c3r) == 0,
            "predicted receiver recovered a correct frame after attack success")
    w(f"於全部 reduced-iteration profiles，predicted-AMC receiver 在 attack success 條件下之 CRC pass 為 {rng(prc)}，frame "
      f"correct 為 0%。Attack success 條件下之 {n_pp} 次 CRC pass 均為 CRC-8 false accept（recovered frame 含 bit error），"
      "不構成 payload recovery。攻擊誤導 AMC 後，依賴 AMC 決策之接收流程在本資料域中未取得正確 payload。\n")
    w("### 4.4 Synthetic 資料域之 minimum usable setting\n")
    w("**Table 4.** Joint oracle criterion 下，於目前掃描範圍與既定 efficacy criterion 下的最低可用設定"
      "（`data/crc_fec_min_usable_summary.csv`）。\n")
    w("| 攻擊 | FEC | Minimum usable setting | conditional ASR | Oracle CRC｜attack success | Latency median（ms） | Latency 降低 |\n"
      "|---|---|---|---|---|---|---|")
    w("| FGSM | off／on | 單步，無 iteration sweep | N/A | N/A | N/A | N/A |")
    for lab, g in (("PGD", "pgd"), ("CW", "cw"), ("EAD", "ead_phase2")):
        for fe in ("off", "on"):
            m = MIN[(g, fe, "joint_oracle_usable")]
            prof = m["minimum_usable_profile"]
            require(prof, f"no minimum usable setting for {g} FEC {fe}")
            name = short(prof).replace(" x ", " × ")
            if prof == m["baseline_profile"].split(":")[1]:
                name += "（掃描範圍內無法降低）"
            w(f"| {lab} | {fe} | {name} | {pc(m['conditional_asr_pct'])} | {pc(m['oracle_crc_given_attack_success_pct'])} | "
              f"{ms(m['latency_median_ms'])} | {num(m['latency_reduction_pct'], 1) if f(m['latency_reduction_pct']) else '0'}% |")
    w("\n### 4.5 FEC 之影響\n")
    q = {r["attack"] + "|" + r["profile"] + "|" + r["fec"]: r for r in MOD
         if r["table"] == "crc_fec_reduced_iteration" and r["modulation"] == "QAM64"}
    q64 = lambda a, pr, fe: q[f"{a}|{pr}|{fe}"]["oracle_crc_given_attack_success_before_pct"]  # noqa: E731
    require(q64("cw", "c1_steps20_lr0.01", "off") == q64("ead", "bss9_maxit100", "off"), "QAM64 CW/EAD values differ")
    require(all(f(q64(a, pr, "on")) == 100 for a, pr in (("fgsm", "eps0.03"), ("pgd", "eps0.03_steps10"),
                                                         ("cw", "c1_steps20_lr0.01"), ("ead", "bss9_maxit100"))),
            "QAM64 FEC-on oracle CRC not 100%")
    w(f"FEC on 之有效樣本僅含 8PSK、QAM16、QAM64，FEC off 另含 BPSK、QPSK、PAM4、CPFSK、GFSK，兩者 conditional ASR 不可直接比較。"
      f"就 payload recoverability 而言，FEC-enabled decoding 使 attack success 條件下之 oracle CRC pass 由 FEC off 之 "
      f"{rng(orc['off'])} 提高至 {rng(orc['on'])}。於逐調變結果中，QAM64 於 FEC off 時受影響最大：FGSM eps 0.03 為 "
      f"{pc(q64('fgsm', 'eps0.03', 'off'))}，PGD steps 10 為 {pc(q64('pgd', 'eps0.03_steps10', 'off'))}，CW steps 20 與 EAD "
      f"bss 9 × max_it 100 皆為 {pc(q64('cw', 'c1_steps20_lr0.01', 'off'))}；FEC on 時均為 100%（`data/modulation_summary.csv`，"
      "table = crc_fec_reduced_iteration）。\n")

    # ---- Section 5
    tab = [("FGSM eps 0.03", "fgsm", "eps0.03"), ("PGD steps 10", "pgd", "eps0.03_steps10"),
           ("PGD steps 7", "pgd", "eps0.03_steps7"), ("CW steps 20", "cw", "c1_steps20_lr0.01"),
           ("CW steps 15", "cw", "c1_steps15_lr0.01"), ("EAD 9 × 100", "ead", "bss9_maxit100"),
           ("EAD 5 × 10", "ead", "bss5_maxit10")]
    w("## 5. Defense Evaluation\n")
    w("### 5.1 攻擊後防禦之 ASR 與 recovery\n")
    for tn, fe, cc in (("5", "off", cc_off), ("6", "on", cc_on)):
        if fe == "off":
            w(f"**Table 5.** FEC off（n_clean_correct = {cc}）。各格為「防禦後 conditional ASR／recovery rate／oracle frame "
              "correct｜attack success」；none 欄為防禦前 conditional ASR 與 oracle frame correct｜attack success。\n")
        else:
            w(f"**Table 6.** FEC on（n_clean_correct = {cc}），格式同 Table 5。\n")
        w("| 攻擊／profile | none | K10 | K20 | K30 | K40 | Adaptive-K |\n|---|---|---|---|---|---|---|")
        for lab, a, pr in tab:
            n0 = ad(a, pr, fe, "none")
            cells = [f"{pc(n0['conditional_asr_before_defense_pct'])}／N/A／"
                     f"{pc(n0['oracle_frame_correct_given_attack_success_before_pct'])}"]
            for dn in DEF[1:]:
                r = ad(a, pr, fe, dn)
                cells.append(f"{num(r['conditional_asr_after_defense_pct'])}／{num(r['recovery_rate_pct'])}／"
                             f"{num(r['oracle_frame_correct_given_attack_success_before_pct'])}")
            w(f"| {lab} | " + " | ".join(cells) + " |")
        w("")
    w("全部 12 個 profile 之結果見 `data/attack_defense_summary.csv`。\n")
    w("### 5.2 結果\n")
    lp = [(a, pr) for _, a, pr in tab if a in ("fgsm", "pgd")]
    d4 = ("topk20", "topk30", "topk40", "adaptive_k")
    rec = lambda fe: [ad(a, pr, fe, d)["recovery_rate_pct"] for a, pr in lp for d in d4]  # noqa: E731
    w(f"- **L∞ 梯度攻擊（FGSM、PGD）**：於 Table 5–6 所列之 FGSM 與 PGD profiles，K20、K30、K40 與 Adaptive-K 之 recovery rate "
      f"於 FEC off 為 {rng(rec('off'))}，於 FEC on 為 {rng(rec('on'))}；防禦後 conditional ASR 與防禦前之差異多在數個 "
      "percentage points 之內。")
    for pr in ("c1_steps20_lr0.01", "c1_steps15_lr0.01"):
        vals = {d: f(ad("cw", pr, "off", d)["recovery_rate_pct"]) for d in DEF[1:]}
        require(max(vals, key=vals.get) == "topk20", f"K20 no longer the highest CW recovery ({pr})")
    cw_on = {(pr, d): ad("cw", pr, "on", d)["recovery_rate_pct"] for pr in ("c1_steps10_lr0.01", "c1_steps15_lr0.01",
                                                                            "c1_steps20_lr0.01") for d in DEF[1:]}
    require(all(cw_on[("c1_steps10_lr0.01", d)] == cw_on[("c1_steps15_lr0.01", d)] == cw_on[("c1_steps20_lr0.01", d)]
                for d in DEF[1:]), "CW FEC-on results no longer identical across steps")
    w(f"- **CW**：FEC off 時 K20 之 recovery rate 最高（steps 20：{pc(ad('cw', 'c1_steps20_lr0.01', 'off', 'topk20')['recovery_rate_pct'])}；"
      f"steps 15：{pc(ad('cw', 'c1_steps15_lr0.01', 'off', 'topk20')['recovery_rate_pct'])}），Adaptive-K 為 "
      f"{pc(ad('cw', 'c1_steps20_lr0.01', 'off', 'adaptive_k')['recovery_rate_pct'])} 與 "
      f"{pc(ad('cw', 'c1_steps15_lr0.01', 'off', 'adaptive_k')['recovery_rate_pct'])}。FEC on 時 CW steps 10、15、20 之結果相同，"
      f"K20 為 {pc(cw_on[('c1_steps20_lr0.01', 'topk20')])}，Adaptive-K 為 {pc(cw_on[('c1_steps20_lr0.01', 'adaptive_k')])}。")
    eb = lambda pr, d: ad("ead", pr, "off", d)["recovery_rate_pct"]  # noqa: E731
    require(f(eb("bss5_maxit10", "topk40")) < f(eb("bss9_maxit100", "topk40")) and
            f(eb("bss5_maxit10", "adaptive_k")) < f(eb("bss9_maxit100", "adaptive_k")), "reduced EAD recovery ordering")
    w(f"- **EAD**：EAD 9 × 100 於 FEC off 時 K40 之 recovery rate 為 {pc(eb('bss9_maxit100', 'topk40'))}，Adaptive-K 為 "
      f"{pc(eb('bss9_maxit100', 'adaptive_k'))}，K20 為 {pc(eb('bss9_maxit100', 'topk20'))}。Reduced EAD（bss 5 × max_it 10）之 "
      f"recovery rate 低於 baseline EAD：K40 為 {pc(eb('bss5_maxit10', 'topk40'))}，Adaptive-K 為 {pc(eb('bss5_maxit10', 'adaptive_k'))}。")
    adr = D["ad_rows"]
    ofc = lambda fe, ds: [r["oracle_frame_correct_given_attack_success_before_pct"] for r in adr  # noqa: E731
                          if r["fec"] == fe and r["defense"] in ds]
    for _, a, pr in tab:
        k20 = f(ad(a, pr, "off", "topk20")["oracle_frame_correct_given_attack_success_before_pct"])
        require(all(k20 < f(ad(a, pr, "off", d)["oracle_frame_correct_given_attack_success_before_pct"])
                    for d in ("topk30", "topk40", "adaptive_k")), f"K20 oracle ordering changed for {a} {pr}")
    w(f"- **Payload recoverability**：於全部 12 個 profile，K20、K30、K40 與 Adaptive-K 於 FEC on 時，attack success 條件下之 "
      f"oracle frame correct 為 {rng(ofc('on', d4))}，FEC off 時為 {rng(ofc('off', d4))}；於 Table 5 所列 profiles，K20 之值低於 "
      f"K30、K40 與 Adaptive-K。K10 於 FEC off 時為 {rng(ofc('off', ('topk10',)))}，FEC on 時為 {rng(ofc('on', ('topk10',)))}。\n")
    w("### 5.3 K10 之解讀\n")
    k10a = lambda fe: [r["conditional_asr_after_defense_pct"] for r in adr if r["fec"] == fe and r["defense"] == "topk10"]  # noqa: E731
    k10x = lambda fe: [r["share_of_success_after_explained_by_clean_defense_pct"] for r in adr  # noqa: E731
                       if r["fec"] == fe and r["defense"] == "topk10"]
    nine = [(a, pr) for a, pr in D["rep_profiles"]]
    xs = lambda d: [ad(a, pr, "off", d)["share_of_success_after_explained_by_clean_defense_pct"] for a, pr in nine]  # noqa: E731
    f005 = ad("fgsm", "eps0.005", "off", "topk10")
    w(f"K10 之防禦後 conditional ASR 與攻擊強度無關，於 FEC off 為 {rng(k10a('off'))}，於 FEC on 為 {rng(k10a('on'))}，包含 FGSM "
      f"eps 0.005（防禦前 {pc(f005['conditional_asr_before_defense_pct'])}，防禦後 {pc(f005['conditional_asr_after_defense_pct'])}）。"
      "將攻擊 run 與 clean-only control 依 burst 對接後（`data/attack_defense_summary.csv`，"
      "`share_of_success_after_explained_by_clean_defense_pct`），K10 防禦後被計為 attack_success_after_defense 之 burst 中，"
      f"同一 burst 之 clean waveform 經 K10 後亦被誤判之比例，於 12 個 profile 中 FEC off 為 {rng(k10x('off'))}，FEC on 為 "
      f"{rng(k10x('on'))}。K10 之防禦後誤判主要來自 defense-induced misclassification（第 6 節），因此不可僅以 attacked ASR "
      f"解讀 K10 之防禦效力。同一指標於表列 {len(nine)} 個 profile（FGSM eps 0.03、PGD steps 5／7／10、CW steps 10／15／20、"
      f"EAD 兩個設定）中，FEC off 時 K20 為 {rng(xs('topk20'))}，Adaptive-K 為 {rng(xs('adaptive_k'))}。\n")
    w("![Figure 2](figures/fig2_defense_recovery_clean_degradation.png)\n")
    w("**Figure 2.** (a) FEC off 時各防禦之 attack recovery rate（baseline 與 reduced EAD profiles）；(b) 無攻擊時各防禦之 clean "
      "accuracy degradation。資料：`figure_sources/fig2_defense_recovery_clean_degradation.csv`。\n")

    # ---- Section 6
    w("## 6. Clean-Signal Defense Impact\n")
    w("**Table 7.** Clean-only control（無攻擊；`data/clean_defense_summary.csv`）。\n")
    w("| 防禦 | FEC | Clean accuracy | Clean degradation | Defense-induced misclassification | Oracle CRC pass | "
      "Oracle frame correct | Oracle BER | Predicted frame correct |\n|---|---|---|---|---|---|---|---|---|")
    for fe in ("off", "on"):
        for dn in DEF:
            c = cl(dn, fe)
            ber = f(c["oracle_ber_mean"])
            w(f"| {DEF_LABEL[dn]} | {fe} | {pc(c['clean_accuracy_after_defense_pct'])} | "
              f"{'N/A' if dn == 'none' else num(c['clean_degradation_pp']) + ' pp'} | "
              f"{'N/A' if dn == 'none' else c['n_defense_induced_misclassification']} | {pc(c['oracle_crc_pass_pct'])} | "
              f"{pc(c['oracle_frame_correct_pct'])} | {'0' if ber == 0 else f'{ber:.4f}'} | {pc(c['predicted_frame_correct_pct'])} |")
    w(f"\nFEC off 之分母為 {n_off} bursts（無防禦時 clean_correct {cc_off}），FEC on 為 {n_on} bursts（{cc_on}）。\n")
    dg = lambda d, fe: num(cl(d, fe)["clean_degradation_pp"]) + " pp"  # noqa: E731
    w(f"- 在固定 SNR 18 dB 之 synthetic TX/RX 條件下，K10 於 FEC 關閉時使 clean accuracy 由 "
      f"{pc(cl('none', 'off')['clean_accuracy_after_defense_pct'])} 降至 {pc(cl('topk10', 'off')['clean_accuracy_after_defense_pct'])}"
      f"（clean degradation {dg('topk10', 'off')}），於 FEC 開啟時降低 {dg('topk10', 'on')}。")
    w(f"- FEC off 時之 clean degradation：K20 為 {dg('topk20', 'off')}，K30 為 {dg('topk30', 'off')}，K40 為 {dg('topk40', 'off')}，"
      f"Adaptive-K 為 {dg('adaptive_k', 'off')}。FEC on 時：K20 為 {dg('topk20', 'on')}，K30 為 {dg('topk30', 'on')}，K40 為 "
      f"{dg('topk40', 'on')}，Adaptive-K 為 {dg('adaptive_k', 'on')}。")
    cof = lambda fe, ds: [cl(d, fe)["oracle_frame_correct_pct"] for d in ds]  # noqa: E731
    require(all(f(v) == 100 for v in cof("on", d4)), "FEC-on clean oracle frame correct not 100% for K20+")
    w(f"- FEC off 時，K20、K30、K40 與 Adaptive-K 之 clean oracle frame correct 為 {rng(cof('off', d4))}，K10 為 "
      f"{pc(cl('topk10', 'off')['oracle_frame_correct_pct'])}。在本實驗之 FEC-on 有效樣本中，K20、K30、K40 與 Adaptive-K 之 "
      "oracle frame correct 均為 100%，表示 FEC-enabled decoding 在此資料域與實驗條件下可降低上述防禦對 payload "
      f"recoverability 的影響；K10 於 FEC on 時之 oracle frame correct 為 {pc(cl('topk10', 'on')['oracle_frame_correct_pct'])}，"
      "仍低於無防禦條件。")
    cm = {(r["defense"], r["fec"], r["modulation"]): r for r in MOD if r["table"] == "clean_defense"}
    ca = lambda d, mo: f"{f(cm[(d, 'off', mo)]['clean_accuracy_after_defense_pct']):.0f}%"  # noqa: E731
    require(all(ca("topk10", mo) == "0%" for mo in ("8PSK", "QPSK", "QAM16")), "K10 per-modulation pattern changed")
    w(f"- 逐調變結果中（`data/modulation_summary.csv`，table = clean_defense），K10 於 FEC off 時使 8PSK、QPSK、QAM16 之 clean "
      f"accuracy 降至 0%，BPSK 降至 {ca('topk10', 'BPSK')}，CPFSK 降至 {ca('topk10', 'CPFSK')}；PAM4 與 QAM64 分別維持 "
      f"{ca('topk10', 'PAM4')} 與 {ca('topk10', 'QAM64')}。K20 時 8PSK 由 {ca('none', '8PSK')} 降至 {ca('topk20', '8PSK')}，"
      f"QAM16 由 {ca('none', 'QAM16')} 降至 {ca('topk20', 'QAM16')}。\n")
    w("![Figure 3](figures/fig3_oracle_frame_recoverability.png)\n")
    w("**Figure 3.** Oracle receiver 之 frame correct：clean burst（全部 burst）與各攻擊之 attack-success burst，FEC off 與 FEC on。"
      "資料：`figure_sources/fig3_oracle_frame_recoverability.csv`。\n")

    # ---- Section 7
    w("## 7. Adaptive-K Behavior\n")
    w("**Table 8.** Adaptive-K 之 K 選擇與 blind SNR estimate（`data/adaptive_k_selection_summary.csv`；injected SNR 18 dB）。\n")
    w("| 輸入 | FEC | n | Wideband route | Selected K（min／median／mean／max） | K ≥ 21 | K = 35 | K 受 SNR cap 限制 | "
      "SNR estimate median（範圍） |\n|---|---|---|---|---|---|---|---|---|")
    akg = {(r["input_condition"], r["fec"]): r for r in AK if r["grouping"] == "effective_fec"}
    share21 = {}
    for cond in ("clean", "attacked"):
        for fe in ("off", "on"):
            r = akg[(cond, fe)]
            require(int(r["n_wideband_quantize"]) == 0, "Adaptive-K wideband route used")
            h = {int(k): v for k, v in json.loads(r["selected_k_histogram"]).items()}
            s21 = 100 * sum(v for k, v in h.items() if k >= 21) / sum(h.values())
            share21[(cond, fe)] = s21
            w(f"| {cond} | {fe} | {int(r['n']):,} | 0 | {r['selected_k_min']}／{kfmt(r['selected_k_median'])}／"
              f"{f(r['selected_k_mean']):.1f}／{r['selected_k_max']} | {pc(s21)} | {h.get(35, 0)} | {int(r['n_k_capped']):,} | "
              f"{f(r['snr_est_db_median']):.2f} dB（{f(r['snr_est_db_min']):.2f}–{f(r['snr_est_db_max']):.2f}） |")
    snr_med = [f(r["snr_est_db_median"]) for r in akg.values()]
    require(min(share21.values()) >= 90.0, "selected K no longer mainly within 21-35")
    w(f"\n- Adaptive-K 之實作為 production `adaptive_k_v2_snr_defense`。SNR 由 defense 輸入波形之 FFT 盲估計（前 "
      f"{p['pilot_k']} 個 PSD bins 與其餘 bins 之能量比），未使用注入之 SNR metadata；此估計值不等同於通道 SNR。")
    w(f"- 全部樣本均走 narrowband route，wideband route 為 0。Blind SNR estimate 之 median 介於 {min(snr_med):.2f}–"
      f"{max(snr_med):.2f} dB（FEC off 約 7 dB、FEC on 約 5 dB），低於注入之 18 dB，因此 SNR-dependent K cap 多插值於 "
      f"{p['k_max_high_snr']} 至 {p['k_max_low_snr']} 之間；selected K 大多位於 21–35（{pc(min(share21.values()))} 以上）。")
    akm = {r["true_modulation"]: r for r in AK if r["input_condition"] == "clean"
           and r["grouping"] == "true_modulation+effective_fec" and r["fec"] == "off"}
    qam = {kfmt(akm[m]["selected_k_median"]) for m in ("8PSK", "QAM16", "QAM64")}
    require(len(qam) == 1, "8PSK/QAM16/QAM64 median K differ")
    w(f"- Clean 輸入之 selected K median 於 FEC off 為 {kfmt(akg[('clean', 'off')]['selected_k_median'])}，FEC on 為 "
      f"{kfmt(akg[('clean', 'on')]['selected_k_median'])}；clean 與 attacked 輸入之 selected K 與 SNR estimate 分布相近，K 主要隨"
      f"調變而異：clean FEC off 時 selected K median 為 GFSK {kfmt(akm['GFSK']['selected_k_median'])}、BPSK "
      f"{kfmt(akm['BPSK']['selected_k_median'])}、PAM4 {kfmt(akm['PAM4']['selected_k_median'])}、CPFSK "
      f"{kfmt(akm['CPFSK']['selected_k_median'])}、QPSK {kfmt(akm['QPSK']['selected_k_median'])}，8PSK、QAM16、QAM64 為 "
      f"{qam.pop()}。")
    deg = {d: f(cl(d, fe)["clean_degradation_pp"]) for d in ("topk30", "topk40", "adaptive_k") for fe in ("off",)}
    require(all(min(f(cl("topk30", fe)["clean_degradation_pp"]), f(cl("topk40", fe)["clean_degradation_pp"]))
                <= f(cl("adaptive_k", fe)["clean_degradation_pp"]) <=
                max(f(cl("topk30", fe)["clean_degradation_pp"]), f(cl("topk40", fe)["clean_degradation_pp"]))
                for fe in ("off", "on")), "Adaptive-K clean degradation not between K30 and K40")
    between, equal, higher, lower = [], [], [], []
    names = {("cw", "c1_steps10_lr0.01"): "CW steps 10", ("cw", "c1_steps15_lr0.01"): "CW steps 15",
             ("cw", "c1_steps20_lr0.01"): "CW steps 20", ("ead", "bss9_maxit100"): "EAD 9 × 100",
             ("pgd", "eps0.03_steps10"): "PGD steps 10", ("fgsm", "eps0.03"): "FGSM eps 0.03",
             ("pgd", "eps0.03_steps5"): "PGD steps 5", ("pgd", "eps0.03_steps7"): "PGD steps 7",
             ("ead", "bss5_maxit10"): "EAD 5 × 10"}
    for (a, pr), nm in names.items():
        k3, k4, ak_ = (f(ad(a, pr, "off", d)["recovery_rate_pct"]) for d in ("topk30", "topk40", "adaptive_k"))
        if ak_ in (k3, k4):
            equal.append(f"{nm}（與 {'K30' if ak_ == k3 else 'K40'} 相同）")
        elif min(k3, k4) < ak_ < max(k3, k4):
            between.append(nm)
        elif ak_ > max(k3, k4):
            higher.append(nm)
        else:
            lower.append(nm)
    parts = []
    if between:
        parts.append(f"對 {'、'.join(between)} 介於 K30 與 K40 之間")
    if equal:
        parts.append(f"對 {'、'.join(equal)}")
    if higher:
        parts.append(f"對 {'、'.join(higher)} 高於 K30 與 K40")
    if lower:
        parts.append(f"對 {'、'.join(lower)} 低於 K30 與 K40")
    w(f"- Adaptive-K 之 clean degradation 介於 K30 與 K40 之間（Table 7）。FEC off 時，Adaptive-K 之 recovery rate "
      f"{'，'.join(parts)}（Table 5、`data/attack_defense_summary.csv`）。\n")
    w("![Figure 4](figures/fig4_adaptive_k_selection_snr.png)\n")
    w("**Figure 4.** Adaptive-K 之 selected K 分布 (a)(b) 與 blind SNR estimate 分布 (c)(d)，clean 與 attacked 輸入，FEC off 與 "
      "FEC on；虛線為 injected SNR 18 dB。資料：`figure_sources/fig4_adaptive_k_selection_snr.csv`。\n")

    # ---- Section 8
    w("## 8. Latency and Trade-off Analysis\n")
    w("### 8.1 Defense latency\n")
    lat = {(r["scope"], r["defense"]): r for r in LAT if r["scope"].startswith("defense_on")}
    na = lat[("defense_on_attacked_input", "topk10")]["n"]
    nc = lat[("defense_on_clean_input", "topk10")]["n"]
    w("**Table 9.** Defense latency（x86 VM；`data/latency_summary.csv`）。\n")
    w(f"| 防禦 | Attacked 輸入 median／mean／p95（ms，n = {int(na):,}） | Clean 輸入 median／mean／p95（ms，n = {int(nc):,}） |\n"
      "|---|---|---|")
    for dn in DEF[1:]:
        a_, c_ = lat[("defense_on_attacked_input", dn)], lat[("defense_on_clean_input", dn)]
        w(f"| {DEF_LABEL[dn]} | {ms(a_['median_ms'], 3)}／{ms(a_['mean_ms'], 3)}／{ms(a_['p95_ms'], 3)} | "
          f"{ms(c_['median_ms'], 3)}／{ms(c_['mean_ms'], 3)}／{ms(c_['p95_ms'], 3)} |")
    w("\n### 8.2 Attack 與 defense 之 composite latency\n")
    w("Composite latency 為同一 burst 之 attack-generation latency 與 defense latency 相加之衍生值（derived），非單次量測。"
      "Synthetic 資料域之 attack-generation latency 為每 burst 一次計時呼叫，與第 3 節 AMC-only 之 "
      f"{pg[10]['latency_n_calls']} 次呼叫 protocol 不同，數值不可直接互換。\n")
    w("| 組合（FEC off） | Attack latency median（ms） | Defense latency median（ms） | Composite median（ms） |\n|---|---|---|---|")
    for lab, a, pr, dn in (("PGD steps 10 + none", "pgd", "eps0.03_steps10", "none"),
                           ("PGD steps 7 + Adaptive-K", "pgd", "eps0.03_steps7", "adaptive_k"),
                           ("CW steps 20 + K20", "cw", "c1_steps20_lr0.01", "topk20"),
                           ("EAD 9 × 100 + none", "ead", "bss9_maxit100", "none"),
                           ("EAD 9 × 100 + K40", "ead", "bss9_maxit100", "topk40"),
                           ("EAD 5 × 10 + K40", "ead", "bss5_maxit10", "topk40")):
        r = ad(a, pr, "off", dn)
        w(f"| {lab} | {ms(r['attack_latency_median_ms'])} | "
          f"{'N/A' if dn == 'none' else ms(r['defense_latency_median_ms'], 3)} | "
          f"{ms(r['attack_plus_defense_latency_median_ms_derived'])} |")
    akl = f(lat[("defense_on_attacked_input", "adaptive_k")]["median_ms"])
    k40l = f(lat[("defense_on_attacked_input", "topk40")]["median_ms"])
    p7 = f(ad("pgd", "eps0.03_steps7", "off", "none")["attack_latency_median_ms"])
    e9 = f(ad("ead", "bss9_maxit100", "off", "none")["attack_latency_median_ms"])
    w(f"\nDefense latency 相對 attack-generation latency 之比例小：Adaptive-K（{akl:.3f} ms）約為 PGD steps 7（{p7:.2f} ms）之 "
      f"{100 * akl / p7:.1f}%，約為 EAD 9 × 100（{e9:.2f} ms）之 {100 * akl / e9:.2f}%。\n")
    w("### 8.3 取捨\n")
    w("![Figure 5](figures/fig5_defense_tradeoff.png)\n")
    w("**Figure 5.** (a) FEC off 時各攻擊之 recovery rate 與該防禦之 clean degradation；(b) defense latency median。資料："
      "`figure_sources/fig5_defense_tradeoff.csv`；彙整表 `data/defense_tradeoff_summary.csv`。\n")
    k10r = [ad(a, pr, "off", "topk10")["recovery_rate_pct"] for a, pr, _ in BASEP]
    for a, pr in (("cw", "c1_steps20_lr0.01"), ("ead", "bss5_maxit10")):
        k20 = f(ad(a, pr, "off", "topk20")["recovery_rate_pct"])
        require(all(k20 > f(ad(a, pr, "off", d)["recovery_rate_pct"]) for d in ("topk30", "topk40", "adaptive_k")),
                f"K20 recovery ordering changed for {a} {pr}")
    w(f"- K10 位於 clean degradation {dg('topk10', 'off')} 之位置，各攻擊之 recovery rate 介於 {rng(k10r)}，其防禦後誤判主要為 "
      "defense-induced misclassification。")
    w(f"- K20 之 clean degradation 為 {dg('topk20', 'off')}（FEC off），對 CW 與 EAD 5 × 10 之 recovery rate 高於 K30、K40 與 "
      "Adaptive-K，但 attack success 條件下之 oracle frame correct 低於 K30、K40 與 Adaptive-K。")
    w(f"- K40 與 Adaptive-K 之 clean degradation 為 {dg('topk40', 'off')} 與 {dg('adaptive_k', 'off')}（FEC off），對 EAD 9 × 100 "
      f"之 recovery rate 為 {pc(eb('bss9_maxit100', 'topk40'))} 與 {pc(eb('bss9_maxit100', 'adaptive_k'))}，對 PGD steps 10 為 "
      f"{pc(ad('pgd', 'eps0.03_steps10', 'off', 'topk40')['recovery_rate_pct'])} 與 "
      f"{pc(ad('pgd', 'eps0.03_steps10', 'off', 'adaptive_k')['recovery_rate_pct'])}。Adaptive-K 之 defense latency 約為 fixed K "
      f"之 {akl / k40l:.1f} 倍（{akl:.3f} ms 與 {k40l:.3f} ms），兩者均遠小於 attack-generation latency。\n")

    # ---- Section 9
    eadr = {d: f(eb("bss9_maxit100", d)) for d in DEF[1:]}
    best = max(eadr, key=eadr.get)
    pgdr = [ad(a, pr, fe, d)["recovery_rate_pct"] for a, pr in (("pgd", "eps0.03_steps5"), ("pgd", "eps0.03_steps7"),
                                                                ("pgd", "eps0.03_steps10"))
            for fe in ("off", "on") for d in ("topk30", "topk40", "adaptive_k")]
    fixl = [f(lat[("defense_on_attacked_input", d)]["median_ms"]) for d in ("topk10", "topk20", "topk30", "topk40")]
    w("## 9. 主要研究發現\n")
    w(f"1. 在 AMC-only 資料域與 90% 正式 baseline 準則下，於目前掃描範圍與既定 efficacy criterion 下的最低可用設定，PGD、CW、EAD "
      f"分別為 steps 7（ASR {pc(pg[7]['conditional_asr_pct'])}）、steps 15（{pc(cw[15]['conditional_asr_pct'])}）與 bss 5 × "
      f"max_iterations 10（{pc(e2[5]['conditional_asr_pct'])}），attack-generation latency median 分別降低 "
      f"{pc(pg[7]['latency_reduction_pct'])}、{pc(cw[15]['latency_reduction_pct'])} 與 {pc(e2[5]['latency_reduction_pct'])}。"
      "FGSM 為單步攻擊，無 iteration 可降低。")
    w(f"2. 在 synthetic TX/RX 資料域，攻擊成功誤導 AMC 後，oracle receiver 之 frame correct 於 FEC off 為 {rng(orc['off'])}，"
      f"於 FEC on 為 {rng(orc['on'])}；在 attack success 條件下，多數樣本仍可於 oracle receiver 下正確恢復 payload。")
    w(f"3. Predicted-AMC receiver 在 attack success 條件下之 frame correct 為 0%；其 {rng(prc)} 之 CRC pass 均為 CRC-8 false "
      "accept，不構成 payload recovery。")
    w(f"4. Fixed Top-K 與 Adaptive-K 之 recovery 隨攻擊類型差異大：對 EAD 9 × 100 之 recovery rate 最高達 "
      f"{pc(eadr[best])}（{DEF_LABEL[best]}，FEC off），對 PGD 之 K30、K40 與 Adaptive-K 於 FEC off 與 FEC on 為 {rng(pgdr)}。"
      "Reduced EAD 之 recovery rate 低於 baseline EAD。")
    w(f"5. K10 本身造成 {dg('topk10', 'off')}（FEC off）與 {dg('topk10', 'on')}（FEC on）之 clean degradation，其防禦後誤判於 "
      f"FEC off 有 {rng(k10x('off'))}、FEC on 有 {rng(k10x('on'))} 可由 defense-induced misclassification 解釋。")
    w(f"6. K40 與 Adaptive-K 之 clean degradation 於 FEC off 為 {dg('topk40', 'off')} 與 {dg('adaptive_k', 'off')}，於 FEC on 為 "
      f"{dg('topk40', 'on')} 與 {dg('adaptive_k', 'on')}；FEC on 時 K20、K30、K40 與 Adaptive-K 之 clean oracle frame correct "
      "均為 100%。")
    w(f"7. Adaptive-K 於全部樣本走 narrowband route，blind SNR estimate median 約 {min(snr_med):.0f}–{max(snr_med):.0f} dB"
      "（injected 18 dB），selected K 大多位於 21–35；其 clean degradation 介於 K30 與 K40 之間。")
    w(f"8. Defense latency（fixed K 約 {min(fixl):.2f}–{max(fixl):.2f} ms，Adaptive-K 約 {akl:.2f} ms）相對 attack-generation "
      "latency 小，攻擊端之 iteration 設定為 composite latency 之主要決定因素。\n")

    # ---- Section 10
    w("## 10. 實驗限制\n")
    w(f"1. **Latency 平台**：所有 latency 為 x86 VM latency（{e['dmi_product_name']}，Intel Core i7-11700，torch intra-op "
      f"{e['torch_num_threads']}、inter-op {e['torch_num_interop_threads']}，B = 1），不代表 native desktop、嵌入式平台或加速器之"
      "延遲。")
    w(f"2. **樣本數**：AMC-only 之 conditional ASR 分母為 {n_cc_amc}；synthetic 資料域 FEC on 之分母為 {cc_on}，recovery rate "
      "之分母為 attack success 數。接近準則門檻之設定具抽樣敏感性。")
    w("3. **掃描範圍與準則**：Minimum usable setting 僅於所列離散設定與 90% baseline-efficacy criterion 下成立。PGD 之 alpha "
      "固定為 eps／4；EAD Phase 2 僅於 max_iterations 10 下掃描 bss。")
    w(f"4. **兩資料域之 baseline 與 latency protocol 不同**：AMC-only 以正式 2,200 樣本 conditional ASR 為 baseline、每 profile "
      f"{pg[10]['latency_n_calls']} 次呼叫；synthetic 資料域以同資料域 baseline profile 為基準、每 burst 一次呼叫。兩資料域之 ASR "
      "與 latency 不可直接比較。")
    g0 = cm[("none", "off", "GFSK")]["clean_accuracy_after_defense_pct"]
    w(f"5. **Synthetic TX/RX 條件**：固定 SNR 18 dB、cfo_std 0、無 multipath；AWN 未於此類 burst 上訓練，合成 GFSK 之 clean "
      f"AMC accuracy 為 {f(g0):.0f}%。FEC on 之結論僅涵蓋 8PSK、QAM16、QAM64；QPSK 與 PAM4 於 FEC on 時為 payload-degenerate，"
      "已排除。")
    w("6. **防禦評估範圍**：防禦評估為 post-attack evaluation，攻擊非 defense-aware；防禦參數使用 production default，未針對"
      "本資料域調整。")
    w("7. **Adaptive-K 之 SNR**：blind SNR estimate 為頻譜能量比之估計值，與通道 SNR 定義不同，本研究不以其評估 SNR 估計準確度。")
    w("8. **威脅層級**：所有結果為 digital-domain 評估，擾動直接作用於記憶體中之 IQ；不包含 over-the-air 傳輸或硬體接收鏈。")
    (OUT / "RESEARCH_REPORT_ZH_TW.md").write_text("\n".join(out) + "\n", encoding="utf-8")


README = """# reduced_iteration_crc_fec_defense_study_20261022

## 1. 內容

本資料夾為下列研究之正式整理：四種以受限擾動或降低擾動為設計目標的對抗攻擊（FGSM、PGD、CW、EAD）之 reduced-iteration 評估、synthetic TX/RX 資料域之 CRC-8 與 FEC payload recoverability、Fixed Top-K（K10、K20、K30、K40）與 Adaptive-K 之 post-attack defense evaluation，以及 clean-only defense control。研究報告為 [`RESEARCH_REPORT_ZH_TW.md`](RESEARCH_REPORT_ZH_TW.md)。

## 2. 產生方式

本資料夾全部內容（`data/`、`figure_sources/`、`figures/`、`RESEARCH_REPORT_ZH_TW.md`、`README.md`）由下列指令自原始 results 重新產生，數值未人工輸入：

```
~/adversarial-rf/.venv/bin/python scripts/build_reduced_iteration_crc_fec_defense_study.py
```

該 script 僅讀取下列五個 run 目錄；缺少必要來源檔或任一 run 之 validation 非 PASS 時即中止。報告中之定性敘述以資料條件檢查保護，來源資料改變時 build 失敗而非產生過時文字。

## 3. 來源

| 代號 | Run 目錄 | 內容 |
|---|---|---|
| R1 | `results/reduced_iteration_attacks_vm_20261022_084556/` | RadioML 2016.10a AMC-only reduced-iteration（88 樣本，27 profiles） |
| R2 | `results/synth_crc_fec_integration_20261022_090735/` | Synthetic TX/RX CRC/FEC chain 整合驗證 |
| R3 | `results/reduced_iteration_crc_fec_vm_20261022_093829/` | Synthetic TX/RX reduced-iteration × CRC × FEC（SNR 18 dB，25 bursts/cell，26 unique profiles） |
| R4 | `results/reduced_iteration_crc_fec_defense_vm_20261022_103042/` | 12 attack profiles × 6 defenses × CRC × FEC（28,800 rows） |
| R5 | `results/clean_crc_fec_defense_vm_20261022_104250/` | Clean-only defense control（2,400 rows，無攻擊） |

各來源檔之路徑、位元組數與 SHA-256 列於 `data/source_manifest.csv`。`results/` 不納入版本控制；完整 run 目錄另以 `research_results_20261022.zip` 封存。

## 4. `data/`

所有 CSV 為 UTF-8（無 BOM）。每一列之 `source_file` 欄記錄其來源檔；分母為 0 之比例指標留空，原因記錄於 `na_reasons`。所有 aggregate 均排除 payload-degenerate frame（QPSK 與 PAM4 於 FEC on）。

| 檔案 | 內容 | 報告章節 |
|---|---|---|
| `amc_reduced_iteration_summary.csv` | R1 四種攻擊 27 個 profile 之 conditional ASR、準則門檻、latency median／mean／p95、latency 降低、Linf／L2 | 3 |
| `crc_fec_integration_validation_summary.csv` | R2 parity、clean chain、attack chain、fail-closed 結果 | 4.1 |
| `crc_fec_reduced_iteration_summary.csv` | R3 各 profile × FEC 之 conditional ASR、oracle／predicted CRC 與 frame correct（含計數與分母）、BER、CRC false accept、latency、usable 判定 | 4 |
| `crc_fec_min_usable_summary.csv` | R3 各攻擊 × FEC × 準則之 minimum usable setting；`role` 欄區分 primary 準則與 system-behavior analysis | 4.4 |
| `attack_defense_summary.csv` | R4 12 profiles × FEC × 6 defenses 之 ASR before／after、recovery、oracle／predicted CRC 與 frame correct、latency，並併入 R5 之 clean degradation 與 defense-induced misclassification 對接結果 | 5 |
| `clean_defense_summary.csv` | R5 各防禦 × FEC 之 clean accuracy、clean degradation、oracle／predicted CRC 與 frame correct、BER、defense latency | 6 |
| `defense_tradeoff_summary.csv` | 代表性 profiles 之 recovery、clean degradation、payload recoverability 與 latency 彙整 | 8 |
| `adaptive_k_selection_summary.csv` | R4（attacked）與 R5（clean）之 Adaptive-K route、selected K 分布、blind SNR estimate | 7 |
| `modulation_summary.csv` | 逐調變結果；`table` 欄區分 clean_defense（R5）、attack_defense（R4）、crc_fec_reduced_iteration（R3） | 4.5、6 |
| `latency_summary.csv` | Attack-generation latency（R1、R3）、defense latency（R4、R5）、composite latency（R4，derived） | 8 |
| `experiment_configuration.csv` | 平台、軟體版本、threads、commits、checkpoint／dataset SHA-256、樣本設計、準則、防禦參數、seed 規則 | 2 |
| `source_manifest.csv` | 所有來源檔之路徑、位元組數與 SHA-256 | 全部 |

## 5. `figures/` 與 `figure_sources/`

| 圖 | 檔案 | 資料 |
|---|---|---|
| Figure 1 | `fig1_reduced_iteration_asr_latency.png`：PGD／CW／EAD baseline 與 reduced setting 之 conditional ASR 與 attack-generation latency | `fig1_reduced_iteration_asr_latency.csv` |
| Figure 2 | `fig2_defense_recovery_clean_degradation.png`：各防禦之 attack recovery 與 clean degradation | `fig2_defense_recovery_clean_degradation.csv` |
| Figure 3 | `fig3_oracle_frame_recoverability.png`：防禦 × FEC 之 oracle frame correct | `fig3_oracle_frame_recoverability.csv` |
| Figure 4 | `fig4_adaptive_k_selection_snr.png`：Adaptive-K selected K 與 blind SNR estimate 分布 | `fig4_adaptive_k_selection_snr.csv` |
| Figure 5 | `fig5_defense_tradeoff.png`：recovery、clean degradation 與 defense latency 之取捨 | `fig5_defense_tradeoff.csv` |

`figure_sources/` 之數值由 `data/` 產生（Figure 4 之直方圖計數由 R4、R5 之 `raw_samples.csv` 計算），圖表由 `figure_sources/` 繪製。

## 6. 不包含之內容

模型 checkpoint 與 RadioML dataset 未納入，其 SHA-256 記錄於 `data/experiment_configuration.csv`。逐列 raw 結果保留於原始 run 目錄。
"""


def write_manifest():
    for run in RUNS:
        USED.add(f"{run}/validation.json")
    USED.update({f"{R2}/v2_clean_chain_summary.csv", f"{R2}/v3_attack_chain_summary.csv"})
    man = []
    for rel in sorted(USED):
        pth = REPO / rel
        man.append({"source_path": rel, "run_directory": str(Path(rel).parent), "file": pth.name,
                    "bytes": pth.stat().st_size, "sha256": hashlib.sha256(pth.read_bytes()).hexdigest()})
    wr(OUT / "data" / "source_manifest.csv", man)
    return len(man)


def self_check():
    text = "\n".join(p.read_text(encoding="utf-8") for p in sorted(OUT.rglob("*")) if p.suffix in (".md", ".csv"))
    hits = {w: text.count(w) for w in BANNED if w in text}
    require(not hits, f"banned wording in generated output: {hits}")


def main():
    preflight()
    snap = {p: (p.stat().st_size, p.stat().st_mtime_ns) for run in RUNS for p in (REPO / run).rglob("*") if p.is_file()}
    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("data", "figure_sources", "figures"):
        (OUT / sub).mkdir(parents=True)
    D = build_data()
    build_figures(D)
    build_report(D)
    (OUT / "README.md").write_text(README, encoding="utf-8")
    n = write_manifest()
    self_check()
    after = {p: (p.stat().st_size, p.stat().st_mtime_ns) for run in RUNS for p in (REPO / run).rglob("*") if p.is_file()}
    require(after == snap, "source result directories changed during the build")
    print(f"built {OUT.relative_to(REPO)} from {len(RUNS)} result directories ({n} source files)")


if __name__ == "__main__":
    try:
        main()
    except BuildError as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
