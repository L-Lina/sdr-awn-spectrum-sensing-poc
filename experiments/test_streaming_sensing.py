"""
Offline vs streaming sensing comparison -- answers whether processing a long
IQ stream in fixed-size chunks with a rolling buffer (src/sensing/streaming_
detector.py) produces the same occupied-region and selected-window decisions
as one-shot offline sensing on the whole stream (src/sensing/energy_detection.py
+ src/sensing/segmentation.py, unmodified, called directly).

Does not modify or call into src/utils/pipeline.py -- this is a standalone
prototype comparison, not a change to the formal pipeline.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.sensing.energy_detection import energy_detect, filter_by_min_length, mask_to_regions, merge_close_regions  # noqa: E402
from src.sensing.radioml_source import embed_multiple_samples_in_noise, load_radioml_sample  # noqa: E402
from src.sensing.segmentation import select_aligned_segments  # noqa: E402
from src.sensing.streaming_detector import StreamingDetectorConfig, run_streaming  # noqa: E402

DATASET_PATH = "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl"
THRESHOLD_FACTOR = 5.0
WINDOW = 128
MERGE_GAP = 0
MIN_REGION_LEN = 128
CHUNK_SIZES = [256, 512, 1024, 2048]
BOUNDARY_TOLERANCE_SAMPLES = 4  # explicit, documented tolerance for start/end index agreement


def build_test_stream() -> Tuple[np.ndarray, list]:
    mods = ["QPSK", "BPSK", "QAM16"]
    samples = [load_radioml_sample(DATASET_PATH, m, 0, 0) for m in mods]
    iq, burst_meta = embed_multiple_samples_in_noise(
        samples, n_samples=8192, embed_snr_margin=20.0, seed=7,
        min_burst_gap=600, max_burst_gap=900,
    )
    return iq, burst_meta


def offline_sensing(iq: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray, list]:
    mask = energy_detect(iq, window=WINDOW, threshold_factor=THRESHOLD_FACTOR)
    raw_regions = mask_to_regions(mask)
    merged = merge_close_regions(raw_regions, merge_gap=MERGE_GAP)
    try:
        kept = filter_by_min_length(merged, min_len=MIN_REGION_LEN)
    except RuntimeError:
        kept = []
    segments = np.empty((0, 128), dtype=np.complex64)
    align_meta = []
    if kept:
        segments, align_meta = select_aligned_segments(iq, kept, seg_len=128, policy="max-energy", hop=1)
    return kept, segments, align_meta


def captured_signal_ratio(region: Tuple[int, int], true_start: int, true_end: int) -> float:
    s, e = region
    intersection = max(0, min(e, true_end) - max(s, true_start))
    return intersection / (true_end - true_start) if true_end > true_start else None


def match_regions(offline_regions: List[Tuple[int, int]], streaming_regions: List[Tuple[int, int]],
                   tol: int) -> List[dict]:
    """Greedy nearest-start matching within tolerance; unmatched entries on
    either side are reported explicitly, not silently dropped."""
    matched = []
    used_streaming = set()
    for i, (os_, oe) in enumerate(offline_regions):
        best_j, best_dist = None, None
        for j, (ss, se) in enumerate(streaming_regions):
            if j in used_streaming:
                continue
            dist = abs(ss - os_) + abs(se - oe)
            if best_dist is None or dist < best_dist:
                best_dist, best_j = dist, j
        if best_j is not None and abs(streaming_regions[best_j][0] - os_) <= tol and abs(streaming_regions[best_j][1] - oe) <= tol:
            used_streaming.add(best_j)
            ss, se = streaming_regions[best_j]
            matched.append({
                "offline_start": os_, "offline_end": oe, "streaming_start": ss, "streaming_end": se,
                "start_diff": ss - os_, "end_diff": se - oe, "matched": True,
            })
        else:
            matched.append({
                "offline_start": os_, "offline_end": oe, "streaming_start": None, "streaming_end": None,
                "start_diff": None, "end_diff": None, "matched": False,
            })
    for j, (ss, se) in enumerate(streaming_regions):
        if j not in used_streaming:
            matched.append({
                "offline_start": None, "offline_end": None, "streaming_start": ss, "streaming_end": se,
                "start_diff": None, "end_diff": None, "matched": False,
            })
    return matched


def main() -> None:
    out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1]) if "--output-dir" in sys.argv else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    iq, burst_meta = build_test_stream()
    print(f"[streaming_test] stream length={len(iq)}, {len(burst_meta)} true bursts: "
          f"{[(m['true_start'], m['true_end']) for m in burst_meta]}", flush=True)

    offline_regions, offline_segments, offline_meta = offline_sensing(iq)
    print(f"[streaming_test] offline: {len(offline_regions)} regions, {offline_segments.shape[0]} segments", flush=True)

    rows = []
    for chunk_size in CHUNK_SIZES:
        carry = WINDOW  # carry forward one sensing-window's worth of samples for boundary continuity
        config = StreamingDetectorConfig(
            chunk_size=chunk_size, window=WINDOW, threshold_factor=THRESHOLD_FACTOR,
            merge_gap=MERGE_GAP, min_region_len=MIN_REGION_LEN, carry_samples=carry,
        )
        events = run_streaming(iq, config)
        streaming_regions = [(e.start, e.end) for e in events]
        print(f"[streaming_test] chunk_size={chunk_size}: {len(streaming_regions)} streaming regions: {streaming_regions}", flush=True)

        matched = match_regions(offline_regions, streaming_regions, tol=BOUNDARY_TOLERANCE_SAMPLES)
        for m in matched:
            row = dict(m)
            row["chunk_size"] = chunk_size
            row["carry_samples"] = carry
            if row["offline_start"] is not None:
                # find which true burst this offline region overlaps most, for captured_signal_ratio
                best_burst = max(burst_meta, key=lambda b: max(0, min(row["offline_end"], b["true_end"]) - max(row["offline_start"], b["true_start"])))
                row["offline_captured_signal_ratio"] = captured_signal_ratio(
                    (row["offline_start"], row["offline_end"]), best_burst["true_start"], best_burst["true_end"])
            else:
                row["offline_captured_signal_ratio"] = None
            if row["streaming_start"] is not None:
                best_burst = max(burst_meta, key=lambda b: max(0, min(row["streaming_end"], b["true_end"]) - max(row["streaming_start"], b["true_start"])))
                row["streaming_captured_signal_ratio"] = captured_signal_ratio(
                    (row["streaming_start"], row["streaming_end"]), best_burst["true_start"], best_burst["true_end"])
            else:
                row["streaming_captured_signal_ratio"] = None
            rows.append(row)

    fieldnames = ["chunk_size", "carry_samples", "offline_start", "offline_end", "streaming_start", "streaming_end",
                  "start_diff", "end_diff", "matched", "offline_captured_signal_ratio", "streaming_captured_signal_ratio"]
    with open(out_dir / "streaming_sensing_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n_matched = sum(1 for r in rows if r["matched"])
    n_total = len(rows)
    print(f"[streaming_test] DONE: {n_matched}/{n_total} region comparisons matched within tolerance={BOUNDARY_TOLERANCE_SAMPLES} samples", flush=True)
    print(f"[streaming_test] offline_region_count={len(offline_regions)}", flush=True)
    for cs in CHUNK_SIZES:
        cnt = sum(1 for r in rows if r["chunk_size"] == cs and r["streaming_start"] is not None)
        print(f"[streaming_test] chunk_size={cs}: streaming_region_count={cnt}", flush=True)


if __name__ == "__main__":
    main()
