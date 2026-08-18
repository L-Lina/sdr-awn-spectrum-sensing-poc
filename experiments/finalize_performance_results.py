"""
Phase H/I: aggregates the raw outputs already produced by
benchmark_pipeline_latency.py (Phase A/B), profile_attacks.py (Phase C),
acceleration_pilot.py (Phase D), acceleration_before_after.py (Phase E),
and test_streaming_sensing.py (Phase G) into the required summary files,
charts, a processing-budget comparison, and a manifest with full
environment provenance. Reads existing CSVs/logs only -- does not re-run
any benchmark or touch external/AWN, external/adversarial-rf, or any other
results/ directory.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 9})

BUDGETS_MS = [5, 10, 20, 35, 50, 100, 250, 1000]


def parse_cprofile_selftime(path: Path, total_seconds_line_prefix="function calls") -> dict:
    """Extracts the reported total run time and the backward-pass self-time
    from a cProfile text dump written by experiments/profile_attacks.py."""
    text = path.read_text()
    total_s = None
    backward_s = None
    for line in text.splitlines():
        if "function calls" in line and " in " in line and "seconds" in line:
            try:
                total_s = float(line.split(" in ")[1].split(" seconds")[0])
            except Exception:  # noqa: BLE001
                pass
        if "run_backward" in line:
            parts = line.split()
            try:
                backward_s = float(parts[1])
            except Exception:  # noqa: BLE001
                pass
    return {"total_s": total_s, "backward_self_s": backward_s}


def main() -> None:
    out = Path(sys.argv[sys.argv.index("--output-dir") + 1])

    # ---------------- attack_profile_summary.csv ----------------
    rows = []
    for attack in ["fgsm", "pgd", "cw"]:
        info = parse_cprofile_selftime(out / f"{attack}_cprofile_top40_selftime.txt")
        backward_pct = (info["backward_self_s"] / info["total_s"] * 100.0) if info["total_s"] else None
        rows.append({
            "attack": attack,
            "n_profiled_calls": 30,
            "n_warmup_calls_excluded": 10,
            "cprofile_total_s": info["total_s"],
            "backward_pass_self_time_s": info["backward_self_s"],
            "backward_pass_pct_of_total": backward_pct,
            "profiler_source": f"{attack}_cprofile_top40_selftime.txt / {attack}_torch_profiler_top30.txt",
        })
    pd.DataFrame(rows).to_csv(out / "attack_profile_summary.csv", index=False)
    print("[finalize] attack_profile_summary.csv written")

    # ---------------- charts ----------------
    charts_dir = out / "charts"
    charts_dir.mkdir(exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(charts_dir / f"{name}.png")
        plt.close(fig)

    # 1. CPU stage latency breakdown (Phase A)
    summary = pd.read_csv(out / "pipeline_latency_summary.csv")
    clean_stages = summary[summary["stage"].isin(
        ["embedding_ms", "energy_detection_ms", "region_postprocess_ms", "segmentation_ms",
         "awn_preprocess_ms", "awn_clean_inference_ms"])]
    clean_stages.to_csv(charts_dir / "01_cpu_stage_latency_breakdown.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(clean_stages["stage"], clean_stages["mean"])
    ax.set_ylabel("mean ms"); ax.set_title("CPU stage latency breakdown (clean path, n=2200)")
    ax.set_yscale("log")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, "01_cpu_stage_latency_breakdown")

    # 2. End-to-end pipeline latency (clean_total_ms and topk_total_ms distributions)
    raw = pd.read_csv(out / "pipeline_latency_raw.csv")
    raw_ok = raw[raw["status"] == "ok"]
    g2 = raw_ok[["clean_total_ms", "topk_total_ms"]]
    g2.describe().to_csv(charts_dir / "02_end_to_end_pipeline_latency.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([raw_ok["clean_total_ms"].dropna(), raw_ok["topk_total_ms"].dropna()],
               tick_labels=["clean_total_ms", "topk_total_ms"], showfliers=False)
    ax.set_ylabel("ms"); ax.set_title("End-to-end pipeline latency (n=2200, outliers hidden)")
    save(fig, "02_end_to_end_pipeline_latency")

    # 3. Attack generation baseline (Phase B)
    baseline_rows = []
    for attack in ["fgsm", "pgd", "cw"]:
        d = pd.read_csv(out / f"{attack}_baseline_raw.csv")
        baseline_rows.append({"attack": attack, "mean_ms": d["attack_generation_ms"].mean(),
                               "median_ms": d["attack_generation_ms"].median(),
                               "p95_ms": d["attack_generation_ms"].quantile(0.95)})
    g3 = pd.DataFrame(baseline_rows)
    g3.to_csv(charts_dir / "03_attack_generation_baseline.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(g3["attack"], g3["median_ms"])
    ax.set_ylabel("median attack_generation_ms (n=330)"); ax.set_title("Attack generation baseline")
    save(fig, "03_attack_generation_baseline")

    # 4. Attack baseline vs optimized (Phase E)
    comp = pd.read_csv(out / "attack_acceleration_comparison.csv")
    comp.to_csv(charts_dir / "04_attack_baseline_vs_optimized.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(comp))
    w = 0.35
    ax.bar(x - w/2, comp["baseline_mean_ms"], w, label="baseline (batch=1)")
    ax.bar(x + w/2, comp["optimized_mean_ms"], w, label="optimized (batch=16, 1 thread)")
    ax.set_xticks(x); ax.set_xticklabels(comp["attack"])
    ax.set_ylabel("mean ms/sample (n=330)"); ax.set_title("Attack latency: baseline vs optimized")
    ax.set_yscale("log"); ax.legend()
    save(fig, "04_attack_baseline_vs_optimized")

    # 5. Attack speedup
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(comp["attack"], comp["speedup_mean"])
    ax.set_ylabel("speedup (mean latency baseline/optimized)"); ax.set_title("Attack acceleration speedup")
    save(fig, "05_attack_speedup")

    # 6. Stage percentage of clean_total (pie-like bar)
    stage_pct = clean_stages[["stage", "pct_of_clean_total"]]
    stage_pct.to_csv(charts_dir / "06_stage_percentage.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(stage_pct["stage"], stage_pct["pct_of_clean_total"])
    ax.set_ylabel("% of clean_total_ms (mean)"); ax.set_title("Stage share of clean-path total latency")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, "06_stage_percentage")

    # 7. Streaming vs offline sensing comparison
    stream = pd.read_csv(out / "streaming_sensing_validation.csv")
    match_rate = stream.groupby("chunk_size")["matched"].mean().reset_index()
    match_rate.to_csv(charts_dir / "07_streaming_vs_offline.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(match_rate["chunk_size"].astype(str), match_rate["matched"] * 100)
    ax.set_ylabel("%% regions matched within tolerance")
    ax.set_xlabel("chunk_size (samples)")
    ax.set_title("Streaming vs offline sensing region agreement")
    save(fig, "07_streaming_vs_offline")

    print(f"[finalize] {len(list(charts_dir.glob('*.png')))} charts written to {charts_dir}")

    # ---------------- processing budget comparison ----------------
    budget_rows = []

    def pct_within(vals, budget):
        vals = np.asarray(vals)
        return float((vals <= budget).mean() * 100.0) if len(vals) else None

    clean_vals = raw_ok["clean_total_ms"].dropna().values
    topk_vals = raw_ok["topk_total_ms"].dropna().values
    for budget in BUDGETS_MS:
        row = {"budget_ms": budget,
               "clean_pipeline_pct_within_budget": pct_within(clean_vals, budget),
               "clean_topk_pipeline_pct_within_budget": pct_within(topk_vals, budget)}
        for attack in ["fgsm", "pgd", "cw"]:
            d = pd.read_csv(out / f"{attack}_baseline_raw.csv")
            row[f"{attack}_baseline_pct_within_budget"] = pct_within(d["attack_total_ms"].dropna().values, budget)
        opt_row = comp.set_index("attack")
        for attack in ["fgsm", "pgd", "cw"]:
            # optimized per-sample mean as a point estimate (no raw per-sample optimized CSV was kept beyond the aggregate)
            row[f"{attack}_optimized_mean_within_budget"] = bool(opt_row.loc[attack, "optimized_mean_ms"] <= budget)
        budget_rows.append(row)
    pd.DataFrame(budget_rows).to_csv(out / "processing_budget_comparison.csv", index=False)
    print("[finalize] processing_budget_comparison.csv written")

    # ---------------- manifest ----------------
    import torch
    import torchattacks

    def git(cmd):
        return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True).strip()

    def cpu_model():
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:  # noqa: BLE001
            return None
        return None

    def mem_total():
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    return line.strip()
        except Exception:  # noqa: BLE001
            return None
        return None

    checkpoint_path = REPO_ROOT / "external/adversarial-rf/2016.10a_AWN.pkl"
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git(["git", "rev-parse", "HEAD"]),
        "git_dirty_status": git(["git", "status", "--porcelain"]),
        "cpu_model": cpu_model(),
        "cpu_core_thread_count_nproc": int(subprocess.check_output(["nproc"], text=True).strip()),
        "lscpu_topology": subprocess.run(["lscpu"], capture_output=True, text=True).stdout,
        "ram": mem_total(),
        "os": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torchattacks_version": torchattacks.__version__,
        "torch_default_num_threads_at_start": torch.get_num_threads(),
        "model_checkpoint_path": str(checkpoint_path),
        "model_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "dataset_path": "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl",
        "modulations": ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"],
        "phase_a_snrs": list(range(-20, 20, 2)),
        "phase_b_e_snrs": [-10, 0, 18],
        "seed": 0,
        "phase_a_warmup_count": 50,
        "phase_a_timed_sample_count": 2200,
        "phase_b_warmup_count": 20,
        "phase_b_timed_sample_count_per_attack": 330,
        "phase_c_warmup_count": 10,
        "phase_c_profiled_call_count": 30,
        "phase_d_pilot_sample_count": 60,
        "phase_e_warmup_count": 20,
        "phase_e_timed_sample_count_per_attack": 330,
        "timing_source": "time.perf_counter_ns() (Phase A/B stage timers), time.perf_counter() (Phase D/E wall-clock wrappers)",
        "memory_measurement_method": "not separately measured this round (no peak-RSS instrumentation added to benchmark_pipeline_latency.py / acceleration_before_after.py)",
        "known_environment_characteristic": (
            "lscpu reports 4 sockets x 4 cores x 1 thread = 16 logical CPUs on this machine, consistent with a "
            "virtualized/containerized topology rather than a native 8-core/16-thread desktop part; this is "
            "reported as-is from lscpu output, not independently verified against physical hardware. "
            "AWN clean inference and attack-generation latency both show a heavy-tailed distribution (mean >> "
            "median, occasional very large max values) even after warm-up, observed consistently across Phase A "
            "and Phase B raw CSVs -- the exact cause (OS scheduling, cross-socket thread synchronization, or "
            "another source) was not further diagnosed this round and is not asserted as confirmed."
        ),
        "optimization_selected_for_phase_e": "batching (batch_size=16) + torch.set_num_threads(1). torch.set_num_threads is a pure implementation_optimization for all attacks (environment setting only). Batching's classification is per-attack, confirmed via a 60-sample deterministic paired test (experiments/batch_equivalence_audit.py) and a CW forward()-source audit (docs/research/PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md section 15.1/15.2): fgsm=implementation_optimization (bit-identical per sample across batch sizes); pgd=implementation_optimization when random_start=False (bit-identical, max diff 0.0), but stochastic_comparison under torchattacks' own default random_start=True (each call draws independent random_start noise, so batch_size changes RNG-consumption order, not the algorithm itself); cw=batched_algorithmic_variant, NOT implementation_optimization (torchattacks.CW's early-stop check compares the whole-batch-summed cost, so batching changes the actual optimization trajectory -- 95.0% prediction match, tensor max diff 0.00138 over 60 paired samples).",
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print("[finalize] manifest.json written")


if __name__ == "__main__":
    main()
