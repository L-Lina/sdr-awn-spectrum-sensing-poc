"""
Performance correctness / stability validation, Part 3 (second half):
controlled AWN inference microbenchmark across CPU thread-count settings.

Builds 100 FIXED AWN inputs once (outside all timing loops), deliberately
including both "known slow" (AM-SSB, SNR in {10,12,14,16,18}, the range
identified by experiments/analyze_awn_latency_outliers.py as dominating the
original 2200-row benchmark's top-50 tail) and "known fast" (all other
modulation/SNR combinations from the original run) samples, so the same
fixed input identities repeat across every thread condition and every pass
-- this lets us test whether latency spikes are tied to SPECIFIC input
content/identity (would recur for the same sample every time) or are
time/scheduling-based (would appear at roughly random positions
regardless of which sample is being processed).

For each of 6 thread conditions (current/1/2/4/8/16), calls
AWNModelAdapter.infer() with >=50 warm-up calls (discarded) followed by
>=500 timed calls (5 full passes over the 100 fixed inputs), excluding
data loading, tensor construction, and CSV writing from the timed region.
Records per-call latency tagged with input identity, then aggregates
mean/median/p90/p95/p99/max/CV/outlier_count (>5x that condition's median).

Does not rerun the 2200-sample clean_sensing benchmark. Does not modify
external/AWN or external/adversarial-rf.
"""

from __future__ import annotations

import csv
import gc
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE  # noqa: E402
from src.sensing.energy_detection import energy_detect, filter_by_min_length, mask_to_regions, merge_close_regions  # noqa: E402
from src.sensing.normalize import apply_awn_preprocess, to_awn_input  # noqa: E402
from src.sensing.radioml_source import embed_sample_in_noise, load_radioml_dict  # noqa: E402
from src.sensing.segmentation import select_aligned_segments  # noqa: E402

DATASET_PATH = "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl"
CHECKPOINT_PATH = "external/adversarial-rf/2016.10a_AWN.pkl"
MODULATIONS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
KNOWN_SLOW_SNRS = [10, 12, 14, 16, 18]  # from analyze_awn_latency_outliers.py finding
N_WARMUP = 60
N_PASSES = 5  # 5 * 100 = 500 timed calls
THREAD_CONDITIONS = ["current", 1, 2, 4, 8, 16]


def build_input(radioml_dict: dict, mod: str, snr: int, idx: int) -> np.ndarray:
    sample = radioml_dict[(mod, snr)][idx].astype(np.float32)
    iq, _ = embed_sample_in_noise(sample, 8192, 20.0, seed=idx)
    mask = energy_detect(iq, window=128, threshold_factor=5.0)
    regions = filter_by_min_length(merge_close_regions(mask_to_regions(mask), merge_gap=0), min_len=128)
    segments, _ = select_aligned_segments(iq, regions, seg_len=128, policy="max-energy", hop=1)
    x = apply_awn_preprocess(segments[:1], policy="radioml-native")
    return to_awn_input(x, seg_len=128)


def build_fixed_inputs(radioml_dict: dict) -> List[dict]:
    """100 fixed inputs: 5 samples (idx 0-4) x AM-SSB x 5 'known slow' SNRs (25 items,
    tagged group='am_ssb_slow_snr'), plus 5 samples x remaining 10 modulations x SNR=0
    (50 items, tagged group='other_mod_snr0'), plus 5 samples x AM-SSB x 5 'known fast'
    negative SNRs (25 items, tagged group='am_ssb_fast_snr') = 100 total. radioml_dict
    is loaded ONCE by the caller (load_radioml_dict re-reads the ~640MB pickle from
    disk on every call and is not cached -- looping load_radioml_sample() 100x here
    would be 100x that disk I/O for no reason)."""
    out = []
    for snr in KNOWN_SLOW_SNRS:
        for idx in range(5):
            out.append({"modulation": "AM-SSB", "snr": snr, "sample_index": idx,
                        "group": "am_ssb_slow_snr", "x": build_input(radioml_dict, "AM-SSB", snr, idx)})
    other_mods = [m for m in MODULATIONS if m != "AM-SSB"]
    for mod in other_mods:
        for idx in range(5):
            out.append({"modulation": mod, "snr": 0, "sample_index": idx,
                        "group": "other_mod_snr0", "x": build_input(radioml_dict, mod, 0, idx)})
    for snr in [-20, -16, -12, -8, -4]:
        for idx in range(5):
            out.append({"modulation": "AM-SSB", "snr": snr, "sample_index": idx,
                        "group": "am_ssb_fast_snr", "x": build_input(radioml_dict, "AM-SSB", snr, idx)})
    assert len(out) == 100, len(out)
    return out


def percentile_stats(latencies_ms: np.ndarray) -> dict:
    median = float(np.median(latencies_ms))
    outlier_count = int(np.sum(latencies_ms > 5 * median)) if median > 0 else 0
    mean = float(latencies_ms.mean())
    std = float(latencies_ms.std())
    return {
        "n": len(latencies_ms), "mean_ms": mean, "median_ms": median, "std_ms": std,
        "p90_ms": float(np.percentile(latencies_ms, 90)), "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)), "max_ms": float(latencies_ms.max()),
        "cv": std / mean if mean > 0 else None,
        "outlier_count_gt_5x_median": outlier_count,
        "outlier_rate": outlier_count / len(latencies_ms),
    }


def main() -> None:
    out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    default_threads = torch.get_num_threads()

    print("[thread_bench] loading real AWN backend ...", flush=True)
    awn = AWNModelAdapter(checkpoint_path=CHECKPOINT_PATH, device="cpu")
    assert awn.backend_name == _REAL_MODEL_SOURCE and awn.status == "ok"

    print("[thread_bench] loading RadioML dataset (once) ...", flush=True)
    radioml_dict = load_radioml_dict(DATASET_PATH)

    print("[thread_bench] building 100 fixed inputs ...", flush=True)
    fixed_inputs = build_fixed_inputs(radioml_dict)

    condition_summary_rows = []
    per_call_rows = []

    for cond in THREAD_CONDITIONS:
        if cond == "current":
            torch.set_num_threads(default_threads)
            n_threads_used = default_threads
        else:
            torch.set_num_threads(int(cond))
            n_threads_used = int(cond)

        gc.collect()
        print(f"[thread_bench] condition={cond} (threads={n_threads_used}): warm-up {N_WARMUP} calls ...", flush=True)
        for i in range(N_WARMUP):
            item = fixed_inputs[i % 100]
            awn.infer(item["x"], seed=0)

        print(f"[thread_bench] condition={cond}: {N_PASSES} timed passes over 100 fixed inputs "
              f"({N_PASSES * 100} timed calls) ...", flush=True)
        latencies = []
        for pass_i in range(N_PASSES):
            for pos, item in enumerate(fixed_inputs):
                t0 = time.perf_counter()
                awn.infer(item["x"], seed=0)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(elapsed_ms)
                per_call_rows.append({
                    "condition": str(cond), "n_threads_used": n_threads_used,
                    "pass_index": pass_i, "position_in_pass": pos,
                    "modulation": item["modulation"], "snr": item["snr"],
                    "sample_index": item["sample_index"], "group": item["group"],
                    "latency_ms": elapsed_ms,
                })

        arr = np.asarray(latencies)
        stats = percentile_stats(arr)
        stats["condition"] = str(cond)
        stats["n_threads_used"] = n_threads_used
        stats["throughput_samples_per_sec"] = 1000.0 / stats["mean_ms"] if stats["mean_ms"] > 0 else None
        condition_summary_rows.append(stats)
        print(f"[thread_bench]   -> mean={stats['mean_ms']:.3f} median={stats['median_ms']:.3f} "
              f"p95={stats['p95_ms']:.3f} p99={stats['p99_ms']:.3f} max={stats['max_ms']:.3f} "
              f"cv={stats['cv']:.3f} outliers(>5x median)={stats['outlier_count_gt_5x_median']}", flush=True)

    torch.set_num_threads(default_threads)

    summary_fields = ["condition", "n_threads_used", "n", "mean_ms", "median_ms", "std_ms", "cv",
                       "p90_ms", "p95_ms", "p99_ms", "max_ms", "outlier_count_gt_5x_median", "outlier_rate",
                       "throughput_samples_per_sec"]
    with open(out_dir / "awn_thread_microbenchmark_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for r in condition_summary_rows:
            w.writerow(r)

    per_call_fields = ["condition", "n_threads_used", "pass_index", "position_in_pass",
                        "modulation", "snr", "sample_index", "group", "latency_ms"]
    with open(out_dir / "awn_thread_microbenchmark_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_call_fields)
        w.writeheader()
        for r in per_call_rows:
            w.writerow(r)

    # group-level breakdown per condition (content-dependence check)
    group_rows = []
    for cond in THREAD_CONDITIONS:
        cond_rows = [r for r in per_call_rows if r["condition"] == str(cond)]
        for group in ["am_ssb_slow_snr", "other_mod_snr0", "am_ssb_fast_snr"]:
            vals = np.asarray([r["latency_ms"] for r in cond_rows if r["group"] == group])
            if len(vals) == 0:
                continue
            gstats = percentile_stats(vals)
            gstats["condition"] = str(cond)
            gstats["group"] = group
            group_rows.append(gstats)
    group_fields = ["condition", "group", "n", "mean_ms", "median_ms", "std_ms", "cv",
                     "p90_ms", "p95_ms", "p99_ms", "max_ms", "outlier_count_gt_5x_median", "outlier_rate"]
    with open(out_dir / "awn_thread_microbenchmark_by_group.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=group_fields)
        w.writeheader()
        for r in group_rows:
            w.writerow({k: r[k] for k in group_fields})

    print("[thread_bench] DONE", flush=True)


if __name__ == "__main__":
    main()
