"""
Performance correctness / stability validation, Part 5: diagnoses (does not
majorly refactor) WHY StreamingDetector (src/sensing/streaming_detector.py,
unmodified) fails to reproduce offline sensing at chunk_size=256 and
partially disagrees at chunk_size=512, using the SAME test stream as
experiments/test_streaming_sensing.py (same seed=7, same 3 RadioML bursts,
same StreamingDetectorConfig).

For each chunk near a true burst, records: combined buffer (carry + chunk)
global range and length, per-chunk median-based noise_floor, threshold,
occupied-sample count/fraction in the smoothed-power array, and (raw burst
length + window) as a fraction of the combined buffer -- since the moving-
average smoothing kernel (width = window) spreads each burst sample's
influence by roughly +/-window/2 in the convolved `smoothed` array, the
EFFECTIVE elevated-sample footprint is closer to (burst_length + window)
than to the raw burst_length alone. This is the direct, evidence-based
mechanism check for the median-based noise-floor estimator's implicit
assumption that occupied samples are a minority of the window it estimates
over.

Writes streaming_failure_diagnosis.csv. Does not implement any fix -- only
diagnoses and (in its trailing docstring / printed summary) proposes the
next minimal fix design, per this round's explicit instruction not to
majorly refactor streaming_detector.py in this validation pass.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.sensing.radioml_source import embed_multiple_samples_in_noise, load_radioml_sample  # noqa: E402
from src.sensing.streaming_detector import StreamingDetector, StreamingDetectorConfig  # noqa: E402

DATASET_PATH = "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl"
WINDOW = 128
THRESHOLD_FACTOR = 5.0


def main() -> None:
    out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)

    mods = ["QPSK", "BPSK", "QAM16"]
    samples = [load_radioml_sample(DATASET_PATH, m, 0, 0) for m in mods]
    iq, burst_meta = embed_multiple_samples_in_noise(
        samples, n_samples=8192, embed_snr_margin=20.0, seed=7,
        min_burst_gap=600, max_burst_gap=900,
    )
    bursts = [(m["true_start"], m["true_end"]) for m in burst_meta]
    print(f"[diagnose] true bursts: {bursts}", flush=True)

    rows = []
    for chunk_size in [256, 512, 1024, 2048]:
        carry = WINDOW
        config = StreamingDetectorConfig(chunk_size=chunk_size, window=WINDOW, threshold_factor=THRESHOLD_FACTOR,
                                          merge_gap=0, min_region_len=128, carry_samples=carry)
        det = StreamingDetector(config)
        n = len(iq)
        for start in range(0, n, chunk_size):
            chunk = iq[start:start + chunk_size]
            combined = np.concatenate([det._carry, chunk]).astype(np.complex64)
            combined_global_start = det._carry_global_offset
            combined_global_end = combined_global_start + len(combined)

            overlaps_burst = any(combined_global_start < be and combined_global_end > bs for bs, be in bursts)
            if overlaps_burst:
                power = np.abs(combined) ** 2
                w = min(det.config.window, len(combined))
                kernel = np.ones(w) / w
                smoothed = np.convolve(power, kernel, mode="same")
                noise_floor = float(np.median(smoothed))
                threshold = noise_floor * det.config.threshold_factor
                occupied = int(np.sum(smoothed > threshold))
                which_burst = next((b for b in bursts if combined_global_start < b[1] and combined_global_end > b[0]), None)
                burst_len = which_burst[1] - which_burst[0] if which_burst else None
                effective_footprint = (burst_len + WINDOW) if burst_len else None
                rows.append({
                    "chunk_size": chunk_size, "carry_samples": carry,
                    "combined_global_start": combined_global_start, "combined_global_end": combined_global_end,
                    "combined_len": len(combined),
                    "burst_true_start": which_burst[0] if which_burst else None,
                    "burst_true_end": which_burst[1] if which_burst else None,
                    "burst_len": burst_len,
                    "effective_footprint_burstlen_plus_window": effective_footprint,
                    "effective_footprint_fraction_of_combined": effective_footprint / len(combined) if effective_footprint else None,
                    "noise_floor": noise_floor, "threshold": threshold,
                    "occupied_samples": occupied, "occupied_fraction": occupied / len(combined),
                    "detection_failed_in_this_chunk": occupied == 0,
                })
            det.process_chunk(chunk)
        det.finalize()
        print(f"[diagnose] chunk_size={chunk_size}: final events={[(e.start, e.end) for e in det.events]}", flush=True)

    fieldnames = ["chunk_size", "carry_samples", "combined_global_start", "combined_global_end", "combined_len",
                  "burst_true_start", "burst_true_end", "burst_len",
                  "effective_footprint_burstlen_plus_window", "effective_footprint_fraction_of_combined",
                  "noise_floor", "threshold", "occupied_samples", "occupied_fraction", "detection_failed_in_this_chunk"]
    with open(out_dir / "streaming_failure_diagnosis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[diagnose] wrote {len(rows)} burst-overlapping chunk rows to streaming_failure_diagnosis.csv", flush=True)
    print("[diagnose] DONE", flush=True)


if __name__ == "__main__":
    main()
