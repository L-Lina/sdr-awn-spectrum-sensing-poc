"""
Performance correctness / stability validation, Part 3 (first half): AWN
latency tail spike investigation, extraction step.

Reads the EXISTING results/performance_latency_20260818T010552Z/
pipeline_latency_raw.csv (2200 rows, already produced by
experiments/benchmark_pipeline_latency.py's clean_sensing mode) -- does NOT
rerun the 2200-sample benchmark. Row order in that CSV is the original
execution order (benchmark_pipeline_latency.py writes one row per sample
immediately after timing it, in a single sequential loop with no
reordering/shuffling), so the CSV row index is used directly as
execution_order below.

Extracts the top-50 awn_clean_inference_ms rows and, for each, records
enough context to check candidate causes without asserting any of them:
modulation, SNR, execution order, previous/next row's awn_clean_inference_ms
(to check if spikes cluster consecutively or are isolated), that row's other
stage latencies (to check whether the spike is AWN-specific or the whole
row is slow, e.g. from OS scheduling contention), and the row's position
modulo common batch/warmup-adjacent periods (32, 50, 100) to check for
periodicity.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    raw_path = out_dir / "pipeline_latency_raw.csv"

    with open(raw_path, newline="") as f:
        rows = list(csv.DictReader(f))

    for i, r in enumerate(rows):
        r["_execution_order"] = i
        r["_awn_ms"] = float(r["awn_clean_inference_ms"])

    ranked = sorted(rows, key=lambda r: r["_awn_ms"], reverse=True)
    top50 = ranked[:50]

    all_awn = sorted(r["_awn_ms"] for r in rows)
    median_awn = all_awn[len(all_awn) // 2]

    out_rows = []
    for r in top50:
        idx = r["_execution_order"]
        prev_r = rows[idx - 1] if idx > 0 else None
        next_r = rows[idx + 1] if idx < len(rows) - 1 else None
        out_rows.append({
            "execution_order": idx,
            "modulation": r["modulation"],
            "snr": r["snr"],
            "sample_index": r["sample_index"],
            "awn_clean_inference_ms": r["_awn_ms"],
            "ratio_to_median": r["_awn_ms"] / median_awn if median_awn > 0 else None,
            "prev_awn_clean_inference_ms": float(prev_r["awn_clean_inference_ms"]) if prev_r else None,
            "next_awn_clean_inference_ms": float(next_r["awn_clean_inference_ms"]) if next_r else None,
            "embedding_ms": r["embedding_ms"],
            "energy_detection_ms": r["energy_detection_ms"],
            "region_postprocess_ms": r["region_postprocess_ms"],
            "segmentation_ms": r["segmentation_ms"],
            "awn_preprocess_ms": r["awn_preprocess_ms"],
            "clean_total_ms": r["clean_total_ms"],
            "row_mod_32": idx % 32,
            "row_mod_50": idx % 50,
            "row_mod_100": idx % 100,
            "is_first_50_rows": idx < 50,
        })

    fieldnames = list(out_rows[0].keys())
    with open(out_dir / "awn_latency_outliers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    # quick diagnostic summary printed to stdout (captured into report, not asserted as causal)
    from collections import Counter
    mod_counts = Counter(r["modulation"] for r in out_rows)
    snr_counts = Counter(r["snr"] for r in out_rows)
    print(f"[outliers] wrote {len(out_rows)} rows to awn_latency_outliers.csv", flush=True)
    print(f"[outliers] median awn_clean_inference_ms over all {len(rows)} rows = {median_awn:.4f} ms", flush=True)
    print(f"[outliers] top-50 min/max awn_clean_inference_ms = {top50[-1]['_awn_ms']:.4f} / {top50[0]['_awn_ms']:.4f} ms", flush=True)
    print(f"[outliers] modulation distribution among top-50: {dict(mod_counts)}", flush=True)
    print(f"[outliers] snr distribution among top-50: {dict(snr_counts)}", flush=True)
    print(f"[outliers] execution_order of top-50 (sorted): {sorted(r['execution_order'] for r in out_rows)}", flush=True)


if __name__ == "__main__":
    main()
