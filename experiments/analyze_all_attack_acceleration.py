"""
Analysis + figures for experiments/benchmark_all_attack_acceleration.py's
output. Reads --output-dir's CSVs (does not recompute or re-derive any
number by hand) and writes: attack_processing_class.csv (A/B/C/D/E per
attack, attack-generation and E2E separately) and 8 charts under
charts/. Every chart is reproducible from the CSVs alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BUDGETS_MS = [10, 20, 50, 100]
CLASS_LABELS = ["A (<10ms)", "B (10-20ms)", "C (20-50ms)", "D (50-100ms)", "E (>100ms)"]


def processing_class(p95_ms) -> str:
    if p95_ms is None or (isinstance(p95_ms, float) and np.isnan(p95_ms)):
        return None
    if p95_ms < 10:
        return "A (<10ms)"
    if p95_ms < 20:
        return "B (10-20ms)"
    if p95_ms < 50:
        return "C (20-50ms)"
    if p95_ms < 100:
        return "D (50-100ms)"
    return "E (>100ms)"


def main() -> None:
    out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    summary = pd.read_csv(out_dir / "attack_bottleneck_summary.csv")
    e2e = pd.read_csv(out_dir / "attack_e2e_summary.csv")
    thread_tuning = pd.read_csv(out_dir / "attack_thread_tuning.csv")
    classification = pd.read_csv(out_dir / "attack_batching_classification.csv")

    order = summary["attack"].tolist()

    # ---- processing class table ----
    rows = []
    for _, r in summary.iterrows():
        atk = r["attack"]
        e2e_row = e2e[e2e["attack"] == atk]
        e2e_p95 = float(e2e_row["total_p95"].iloc[0]) if len(e2e_row) else None
        rows.append({
            "attack": atk,
            "optimized_attack_generation_p95_ms": r["optimized_p95_ms"],
            "attack_generation_processing_class": processing_class(r["optimized_p95_ms"]),
            "e2e_total_p95_ms": e2e_p95,
            "e2e_processing_class": processing_class(e2e_p95),
        })
    pd.DataFrame(rows).to_csv(out_dir / "attack_processing_class.csv", index=False)

    # ---- 1. baseline latency bar chart ----
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(order, summary.set_index("attack").loc[order, "baseline_median_ms"], color="#4C72B0")
    ax.set_ylabel("baseline median attack_generation_ms"); ax.set_yscale("log")
    ax.set_title("17-Attack Baseline Latency (batch=1, default threads)")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(charts_dir / "01_baseline_latency.png", dpi=120); plt.close()

    # ---- 2. optimized latency bar chart ----
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(order, summary.set_index("attack").loc[order, "optimized_median_ms"], color="#55A868")
    ax.set_ylabel("optimized median attack_generation_ms (best safe config)"); ax.set_yscale("log")
    ax.set_title("17-Attack Optimized Latency (best thread/batch config)")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(charts_dir / "02_optimized_latency.png", dpi=120); plt.close()

    # ---- 3. speedup chart ----
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(order)); w = 0.35
    med_sp = summary.set_index("attack").loc[order, "median_speedup"]
    p95_sp = summary.set_index("attack").loc[order, "p95_speedup"]
    ax.bar(x - w / 2, med_sp, w, label="median speedup", color="#4C72B0")
    ax.bar(x + w / 2, p95_sp, w, label="p95 speedup", color="#C44E52")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("speedup (baseline / optimized)"); ax.set_title("17-Attack Speedup (latency ratio)")
    ax.legend(); plt.tight_layout()
    plt.savefig(charts_dir / "03_speedup.png", dpi=120); plt.close()

    # ---- 4. p95 comparison ----
    fig, ax = plt.subplots(figsize=(12, 5))
    base_p95 = summary.set_index("attack").loc[order, "baseline_p95_ms"]
    opt_p95 = summary.set_index("attack").loc[order, "optimized_p95_ms"]
    ax.bar(x - w / 2, base_p95, w, label="baseline p95", color="#4C72B0")
    ax.bar(x + w / 2, opt_p95, w, label="optimized p95", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha="right"); ax.set_yscale("log")
    ax.set_ylabel("p95 attack_generation_ms"); ax.set_title("Baseline vs Optimized p95 Latency")
    ax.legend(); plt.tight_layout()
    plt.savefig(charts_dir / "04_p95_comparison.png", dpi=120); plt.close()

    # ---- 5. batching classification figure ----
    fig, ax = plt.subplots(figsize=(10, 6))
    cls_order = ["A_implementation_optimization", "B_stochastic_batch_compatible",
                 "C_batched_algorithmic_variant", "D_batching_unsafe"]
    colors = {"A_implementation_optimization": "#55A868", "B_stochastic_batch_compatible": "#64B5CD",
              "C_batched_algorithmic_variant": "#DD8452", "D_batching_unsafe": "#C44E52"}
    counts = classification["classification"].value_counts().reindex(cls_order, fill_value=0)
    ax.barh(cls_order, counts.values, color=[colors[c] for c in cls_order])
    for i, v in enumerate(counts.values):
        ax.text(v + 0.05, i, str(int(v)), va="center")
    ax.set_xlabel("attack count"); ax.set_title("Batching Classification Distribution (17 attacks)")
    plt.tight_layout(); plt.savefig(charts_dir / "05_batching_classification.png", dpi=120); plt.close()

    # ---- 6. thread tuning heatmap ----
    pivot = thread_tuning.pivot_table(index="attack", columns="torch_threads", values="median")
    pivot = pivot.reindex(order)
    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(np.log10(pivot.values.astype(float) + 1e-6), cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("torch_num_threads"); ax.set_title("Thread-Tuning Median Latency (log10 ms, darker=faster)")
    plt.colorbar(im, ax=ax, label="log10(median_ms)")
    plt.tight_layout(); plt.savefig(charts_dir / "06_thread_tuning_heatmap.png", dpi=120); plt.close()

    # ---- 7. attack generation vs E2E latency ----
    fig, ax = plt.subplots(figsize=(12, 5))
    e2e_indexed = e2e.set_index("attack").reindex(order)
    ax.bar(x - w / 2, summary.set_index("attack").loc[order, "optimized_p95_ms"], w,
           label="attack_generation p95 (optimized)", color="#4C72B0")
    ax.bar(x + w / 2, e2e_indexed["total_p95"], w, label="E2E total p95", color="#8172B2")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha="right"); ax.set_yscale("log")
    ax.set_ylabel("p95 (ms)"); ax.set_title("Attack Generation vs End-to-End Latency")
    ax.legend(); plt.tight_layout()
    plt.savefig(charts_dir / "07_attackgen_vs_e2e.png", dpi=120); plt.close()

    # ---- 8. processing-class grouping ----
    pc = pd.read_csv(out_dir / "attack_processing_class.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in zip(axes, ["attack_generation_processing_class", "e2e_processing_class"],
                               ["Attack-Generation Processing Class", "End-to-End Processing Class"]):
        counts = pc[col].value_counts().reindex(CLASS_LABELS, fill_value=0)
        ax.bar(CLASS_LABELS, counts.values, color="#4C72B0")
        ax.set_title(title); ax.set_ylabel("attack count")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.savefig(charts_dir / "08_processing_class.png", dpi=120); plt.close()

    print(f"[analyze] wrote attack_processing_class.csv + 8 charts to {charts_dir}")


if __name__ == "__main__":
    main()
