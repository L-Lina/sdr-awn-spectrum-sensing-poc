"""
Generates the 7 required charts for the End-to-End Latency Matrix round,
reading the CSVs already produced by experiments/end_to_end_latency_matrix.py
in the given --output-dir. Does not run any new benchmark or touch any
other results directory.

Charts:
  01_clean_pipeline_stage_breakdown.png -- Scenario A stage latency, mean/median/p95
  02_fgsm_before_after.png              -- FGSM end-to-end baseline vs optimized
  03_pgd_before_after.png               -- PGD (random_start=False) end-to-end baseline vs optimized
  04_scenario_comparison.png            -- Scenario A-E end-to-end totals (baseline config)
  05_mean_median_p95_comparison.png     -- mean/median/p95 before vs after, FGSM/PGD
  06_stage_percentage_stacked.png       -- stacked stage breakdown per scenario/variant
  07_processing_budget_comparison.png   -- median total_ms vs processing budgets

Run directly:
    python experiments/generate_end_to_end_charts.py --output-dir results/<existing_end_to_end_dir>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STAGE_COLORS = {
    "embedding_ms": "#4C72B0", "energy_detection_ms": "#55A868", "region_postprocess_ms": "#C44E52",
    "segmentation_ms": "#8172B2", "awn_preprocess_ms": "#CCB974", "awn_clean_inference_ms": "#64B5CD",
    "attack_generation_ms": "#DD8452", "awn_attacked_inference_ms": "#937860",
    "topk_ms": "#DA8BC3", "defended_inference_ms": "#8C8C8C",
}
STAGE_ORDER = list(STAGE_COLORS.keys())


def main() -> None:
    out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    def load(name):
        with open(out_dir / name) as f:
            return list(csv.DictReader(f))

    stage_summary = load("stage_latency_summary.csv")
    scen_summary = load("end_to_end_latency_summary.csv")
    before_after = load("before_after_end_to_end.csv")
    budget = load("processing_budget_table.csv")

    # 1. clean pipeline stage breakdown (Scenario A)
    fig, ax = plt.subplots(figsize=(8, 5))
    rows_a = [r for r in stage_summary if r["scenario"] == "A" and r["variant"] == "n/a"]
    stages = [r["stage"] for r in rows_a]
    means = [float(r["mean"]) for r in rows_a]
    medians = [float(r["median"]) for r in rows_a]
    p95s = [float(r["p95"]) for r in rows_a]
    x = np.arange(len(stages))
    w = 0.25
    ax.bar(x - w, means, w, label="mean", color="#4C72B0")
    ax.bar(x, medians, w, label="median", color="#55A868")
    ax.bar(x + w, p95s, w, label="p95", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels([s.replace("_ms", "") for s in stages], rotation=30, ha="right")
    ax.set_ylabel("ms"); ax.set_title("Scenario A (Clean AMC) stage latency: mean vs median vs p95 (n=24)")
    ax.legend()
    plt.tight_layout(); plt.savefig(charts_dir / "01_clean_pipeline_stage_breakdown.png", dpi=120); plt.close()

    # 2. FGSM before/after
    fig, ax = plt.subplots(figsize=(6, 5))
    fgsm_row = [r for r in before_after if r["attack"] == "FGSM"][0]
    labels = ["mean", "median", "p95"]
    base_vals = [float(fgsm_row["baseline_end_to_end_mean_ms"]), float(fgsm_row["baseline_end_to_end_median_ms"]),
                 float(fgsm_row["baseline_end_to_end_p95_ms"])]
    opt_vals = [float(fgsm_row["optimized_end_to_end_mean_ms"]), float(fgsm_row["optimized_end_to_end_median_ms"]),
                float(fgsm_row["optimized_end_to_end_p95_ms"])]
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w / 2, base_vals, w, label="baseline (threads=16, batch=1)", color="#C44E52")
    ax.bar(x + w / 2, opt_vals, w, label="optimized (threads=1, batch=16)", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("end-to-end total ms"); ax.set_title("FGSM: Before vs After (end-to-end, Scenario C, n=24)")
    ax.legend()
    plt.tight_layout(); plt.savefig(charts_dir / "02_fgsm_before_after.png", dpi=120); plt.close()

    # 3. PGD (deterministic) before/after
    fig, ax = plt.subplots(figsize=(6, 5))
    pgd_row = [r for r in before_after if "random_start=False" in r["attack"]][0]
    base_vals = [float(pgd_row["baseline_end_to_end_mean_ms"]), float(pgd_row["baseline_end_to_end_median_ms"]),
                 float(pgd_row["baseline_end_to_end_p95_ms"])]
    opt_vals = [float(pgd_row["optimized_end_to_end_mean_ms"]), float(pgd_row["optimized_end_to_end_median_ms"]),
                float(pgd_row["optimized_end_to_end_p95_ms"])]
    ax.bar(x - w / 2, base_vals, w, label="baseline (threads=16, batch=1)", color="#C44E52")
    ax.bar(x + w / 2, opt_vals, w, label="optimized (threads=1, batch=16)", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("end-to-end total ms")
    ax.set_title("PGD (random_start=False): Before vs After (end-to-end, Scenario D_det, n=24)")
    ax.legend()
    plt.tight_layout(); plt.savefig(charts_dir / "03_pgd_before_after.png", dpi=120); plt.close()

    # 4. end-to-end scenario comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    scenario_labels = [("A", "n/a", "A: Clean"), ("B", "n/a", "B: Clean+TopK"),
                        ("C", "baseline", "C: FGSM"), ("D_det", "baseline", "D: PGD(det)"),
                        ("E", "baseline", "E: FGSM+TopK")]
    means, medians, p95s, xt = [], [], [], []
    for sc, var, lab in scenario_labels:
        r = [x for x in scen_summary if x["scenario"] == sc and x["variant"] == var][0]
        means.append(float(r["mean"])); medians.append(float(r["median"])); p95s.append(float(r["p95"]))
        xt.append(lab)
    x = np.arange(len(xt)); w = 0.25
    ax.bar(x - w, means, w, label="mean", color="#4C72B0")
    ax.bar(x, medians, w, label="median", color="#55A868")
    ax.bar(x + w, p95s, w, label="p95", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(xt, rotation=15, ha="right")
    ax.set_ylabel("end-to-end total ms"); ax.set_title("End-to-End Scenario Comparison (baseline config, n=24 each)")
    ax.legend()
    plt.tight_layout(); plt.savefig(charts_dir / "04_scenario_comparison.png", dpi=120); plt.close()

    # 5. mean vs median vs p95 comparison across baseline vs optimized
    fig, ax = plt.subplots(figsize=(9, 5))
    cells = [("C", "baseline", "FGSM\nbaseline"), ("C", "optimized", "FGSM\noptimized"),
             ("D_det", "baseline", "PGD\nbaseline"), ("D_det", "optimized", "PGD\noptimized")]
    means, medians, p95s, xt = [], [], [], []
    for sc, var, lab in cells:
        r = [x for x in scen_summary if x["scenario"] == sc and x["variant"] == var][0]
        means.append(float(r["mean"])); medians.append(float(r["median"])); p95s.append(float(r["p95"]))
        xt.append(lab)
    x = np.arange(len(xt)); w = 0.25
    ax.bar(x - w, means, w, label="mean", color="#4C72B0")
    ax.bar(x, medians, w, label="median", color="#55A868")
    ax.bar(x + w, p95s, w, label="p95", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(xt)
    ax.set_ylabel("end-to-end total ms"); ax.set_title("Mean vs Median vs p95: Before/After (n=24)")
    ax.legend()
    plt.tight_layout(); plt.savefig(charts_dir / "05_mean_median_p95_comparison.png", dpi=120); plt.close()

    # 6. stage percentage stacked bar
    fig, ax = plt.subplots(figsize=(10, 6))
    cells = [("A", "n/a"), ("B", "n/a"), ("C", "baseline"), ("C", "optimized"),
             ("D_det", "baseline"), ("D_det", "optimized"), ("E", "baseline"), ("E", "optimized")]
    xt = [f"{sc}\n{var}" for sc, var in cells]
    bottoms = np.zeros(len(cells))
    x = np.arange(len(cells))
    for stage in STAGE_ORDER:
        vals = []
        for sc, var in cells:
            matching = [r for r in stage_summary if r["scenario"] == sc and r["variant"] == var and r["stage"] == stage]
            vals.append(float(matching[0]["mean"]) if matching else 0.0)
        ax.bar(x, vals, bottom=bottoms, label=stage.replace("_ms", ""), color=STAGE_COLORS[stage])
        bottoms += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels(xt, rotation=20, ha="right")
    ax.set_ylabel("mean stage latency ms (stacked)"); ax.set_title("Stage Percentage Breakdown by Scenario (mean, n=24)")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(charts_dir / "06_stage_percentage_stacked.png", dpi=120); plt.close()

    # 7. processing budget comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    budgets = [5, 10, 20, 35, 50, 100, 250, 1000]
    cells = [("A", "n/a"), ("C", "baseline"), ("C", "optimized"), ("D_det", "baseline"), ("D_det", "optimized")]
    for sc, var in cells:
        r = [x for x in budget if x["scenario"] == sc and x["variant"] == var][0]
        fits = [1 if r[f"fits_{b}ms_median"] == "True" else 0 for b in budgets]
        ax.plot([str(b) for b in budgets], fits, marker="o", label=f"{sc}/{var} (median)")
    ax.set_xlabel("processing budget (ms)"); ax.set_ylabel("fits (1=yes, 0=no)")
    ax.set_title("Processing Budget Comparison (median total_ms fits budget)")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(charts_dir / "07_processing_budget_comparison.png", dpi=120); plt.close()

    print(f"[charts] wrote 7 charts to {charts_dir}")


if __name__ == "__main__":
    main()
