"""
Step 3 pipeline-compatibility smoke test: RadioML burst -> satellite-like
channel (src/channel/satellite_like.py) -> embed in long IQ stream ->
Spectrum Sensing -> segmentation -> real AWN.

Small by design (3 modulations x 3 SNRs x 2 samples x 6 channel conditions
= 108 runs): confirms the channel model integrates correctly into the real
sensing/AMC pipeline, NOT a claim about classification accuracy under
impairment. Does not include attack or Top-K this round (see
docs/research/SATELLITE_LIKE_CHANNEL_SIMULATOR_DESIGN_ZH_TW.md section 13
for why: isolate channel+sensing+AMC correctness first).

Reuses real, unmodified building blocks throughout: energy_detect,
mask_to_regions, merge_close_regions, filter_by_min_length,
select_aligned_segments, apply_awn_preprocess, to_awn_input,
AWNModelAdapter, compute_sensing_ground_truth_metrics. Does not modify
external/AWN, external/adversarial-rf, or any file in src/sensing/.

The only new, local logic here is embed_complex_iq_in_noise() -- a trivial
adaptation of src/sensing/radioml_source.py:embed_sample_in_noise() that
accepts an ALREADY-CHANNEL-TRANSFORMED complex burst (that function itself
takes a raw [2,128] float array and converts it internally, so it cannot
be reused unmodified once the burst has already been passed through
apply_satellite_like_channel -- this mirrors that function's exact noise/
placement statistics, not a new design).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE  # noqa: E402
from src.channel.satellite_like import apply_satellite_like_channel  # noqa: E402
from src.sensing.energy_detection import energy_detect, filter_by_min_length, mask_to_regions, merge_close_regions  # noqa: E402
from src.sensing.ground_truth_metrics import compute_sensing_ground_truth_metrics  # noqa: E402
from src.sensing.normalize import apply_awn_preprocess, to_awn_input  # noqa: E402
from src.sensing.radioml_source import RML2016_10A_CLASSES, load_radioml_dict, radioml_sample_to_iq  # noqa: E402
from src.sensing.segmentation import select_aligned_segments  # noqa: E402
from src.utils.dataset_path import require_dataset_path_exists, resolve_dataset_path  # noqa: E402

DATASET_PATH = resolve_dataset_path()  # priority: env $SDR_AWN_DATASET_PATH > legacy default
CHECKPOINT_PATH = "external/adversarial-rf/2016.10a_AWN.pkl"

MODS = ["BPSK", "QPSK", "8PSK"]
SNRS = [-10, 0, 18]
IDXS = [0, 1]

N_SAMPLES = 8192
EMBED_SNR_MARGIN = 20.0
SIM_SAMPLE_RATE = 200_000.0  # Hz, simulator-level assumption -- see design doc section 6
THRESHOLD_FACTOR = 5.0
SENSING_WINDOW_SIZE = 128
MIN_REGION_LEN = 128
MERGE_GAP = 0
ALIGNMENT_POLICY = "max-energy"
AWN_PREPROCESS = "radioml-native"
SEED = 0

NS_PER_MS = 1_000_000.0

# Six conditions, per design doc section 6 tier definitions.
CONDITIONS: Dict[str, dict] = {
    "A_baseline":    dict(snr_db=None, amplitude_scale=1.0, cfo_hz=0.0,    doppler_hz=0.0,   timing_offset_samples=0),
    "B_amplitude":   dict(snr_db=None, amplitude_scale=0.5, cfo_hz=0.0,    doppler_hz=0.0,   timing_offset_samples=0),
    "C_cfo":         dict(snr_db=None, amplitude_scale=1.0, cfo_hz=2000.0, doppler_hz=0.0,   timing_offset_samples=0),
    "D_doppler":     dict(snr_db=None, amplitude_scale=1.0, cfo_hz=0.0,    doppler_hz=1000.0, timing_offset_samples=0),
    "E_timing":      dict(snr_db=None, amplitude_scale=1.0, cfo_hz=0.0,    doppler_hz=0.0,   timing_offset_samples=2),
    "F_combined":    dict(snr_db=15.0, amplitude_scale=0.5, cfo_hz=2000.0, doppler_hz=1000.0, timing_offset_samples=2),
}


def now_ns() -> int:
    return time.perf_counter_ns()


def embed_complex_iq_in_noise(burst_iq: np.ndarray, n_samples: int, embed_snr_margin: float,
                               seed: int) -> Tuple[np.ndarray, dict]:
    """Same noise/placement statistics as src/sensing/radioml_source.py:
    embed_sample_in_noise(), adapted to accept an ALREADY-COMPLEX,
    already-channel-transformed burst (that function converts a raw
    [2,128] float array internally, so cannot be called unmodified once
    the burst has already been passed through apply_satellite_like_channel).
    burst_power here is measured on the CHANNEL-TRANSFORMED burst (post
    amplitude scaling etc.), matching the original function's own
    burst-relative noise scaling intent."""
    burst_len = len(burst_iq)
    if burst_len >= n_samples:
        raise ValueError(f"burst length ({burst_len}) must be < n_samples ({n_samples})")

    burst_power = float(np.mean(np.abs(burst_iq) ** 2))
    noise_power = burst_power / embed_snr_margin
    noise_std = float(np.sqrt(noise_power / 2.0))

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, n_samples) + 1j * rng.normal(0, noise_std, n_samples)
    iq = noise.astype(np.complex64)

    max_start = n_samples - burst_len
    true_start = int(rng.integers(0, max_start + 1))
    true_end = true_start + burst_len
    iq[true_start:true_end] += burst_iq.astype(np.complex64)

    meta = {"true_start": true_start, "true_end": true_end, "burst_len": burst_len,
            "burst_power": burst_power, "embed_noise_power": noise_power, "n_samples": n_samples}
    return iq, meta


def run_one(awn: AWNModelAdapter, radioml_dict: dict, mod: str, snr: int, idx: int,
            condition_name: str, channel_params: dict) -> dict:
    row: Dict[str, object] = {
        "modulation": mod, "snr": snr, "sample_index": idx, "condition": condition_name,
        "status": "ok", "error_type": None, "error_message": None, "fallback_used": False,
    }
    row.update({f"channel_{k}": v for k, v in channel_params.items()})
    try:
        sample_2x128 = radioml_dict[(mod, snr)][idx].astype(np.float32)
        clean_burst_iq = radioml_sample_to_iq(sample_2x128)

        t0 = now_ns()
        impaired_burst, channel_meta = apply_satellite_like_channel(
            clean_burst_iq, sample_rate=SIM_SAMPLE_RATE, seed=SEED + idx, **channel_params,
        )
        row["channel_apply_ms"] = (now_ns() - t0) / NS_PER_MS
        for k, v in channel_meta.items():
            row[f"channel_meta_{k}"] = v

        t0 = now_ns()
        iq, embed_meta = embed_complex_iq_in_noise(impaired_burst, N_SAMPLES, EMBED_SNR_MARGIN, seed=SEED + idx)
        row["embed_ms"] = (now_ns() - t0) / NS_PER_MS
        true_start, true_end = embed_meta["true_start"], embed_meta["true_end"]

        t0 = now_ns()
        mask = energy_detect(iq, window=SENSING_WINDOW_SIZE, threshold_factor=THRESHOLD_FACTOR)
        raw_regions = mask_to_regions(mask)
        merged_regions = merge_close_regions(raw_regions, merge_gap=MERGE_GAP)
        try:
            kept_regions = filter_by_min_length(merged_regions, min_len=MIN_REGION_LEN)
        except RuntimeError:
            kept_regions = []
        row["sensing_ms"] = (now_ns() - t0) / NS_PER_MS

        gt = compute_sensing_ground_truth_metrics(true_start, true_end, kept_regions)
        row["sensing_detected"] = gt["detection_success"]
        row["captured_signal_ratio"] = gt["captured_signal_ratio"]
        row["start_boundary_error"] = gt["start_boundary_error"]
        row["end_boundary_error"] = gt["end_boundary_error"]

        if not kept_regions:
            row["status"] = "no_region_detected"
            row["clean_prediction"] = None
            row["clean_correct"] = None
            row["confidence"] = None
            return row

        t0 = now_ns()
        segments, _ = select_aligned_segments(iq, kept_regions, seg_len=128, policy=ALIGNMENT_POLICY, hop=1)
        row["segmentation_ms"] = (now_ns() - t0) / NS_PER_MS

        if segments.shape[0] == 0:
            row["status"] = "no_segment_produced"
            row["clean_prediction"] = None
            row["clean_correct"] = None
            row["confidence"] = None
            return row

        t0 = now_ns()
        x = apply_awn_preprocess(segments[:1], policy=AWN_PREPROCESS)
        x = to_awn_input(x, seg_len=128)
        row["awn_preprocess_ms"] = (now_ns() - t0) / NS_PER_MS

        t0 = now_ns()
        logits, meta_awn = awn.infer(x, seed=SEED)
        row["awn_inference_ms"] = (now_ns() - t0) / NS_PER_MS
        if meta_awn["awn_backend"] != _REAL_MODEL_SOURCE:
            row["fallback_used"] = True

        probs = np.exp(logits[0] - np.max(logits[0]))
        probs = probs / probs.sum()
        pred = int(np.argmax(logits[0]))
        row["clean_prediction"] = pred
        row["clean_correct"] = (pred == RML2016_10A_CLASSES[mod])
        row["confidence"] = float(probs[pred])

        if not np.isfinite(logits).all():
            row["status"] = "nan_inf"

    except Exception as exc:  # noqa: BLE001 -- fail closed, record, never fabricate
        row["status"] = "error"
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc)
    return row


def main() -> None:
    out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    dataset_path = sys.argv[sys.argv.index("--dataset-path") + 1] if "--dataset-path" in sys.argv else DATASET_PATH
    require_dataset_path_exists(dataset_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] loading AWN checkpoint {CHECKPOINT_PATH} ...", flush=True)
    awn = AWNModelAdapter(checkpoint_path=CHECKPOINT_PATH, device="cpu")
    if awn.backend_name != _REAL_MODEL_SOURCE or awn.status != "ok":
        raise RuntimeError(f"Real-AWN precheck FAILED: backend={awn.backend_name} status={awn.status}")
    print(f"[smoke] real AWN backend confirmed: {_REAL_MODEL_SOURCE}", flush=True)

    radioml_dict = load_radioml_dict(dataset_path)
    print(f"[smoke] dataset loaded from {dataset_path}, {len(radioml_dict)} (mod,snr) cells", flush=True)

    rows: List[dict] = []
    n_error = n_fallback = n_nan = n_no_region = 0
    total = len(MODS) * len(SNRS) * len(IDXS) * len(CONDITIONS)
    print(f"[smoke] running {total} combos (3 mods x 3 SNRs x 2 idx x {len(CONDITIONS)} conditions)", flush=True)

    i = 0
    for cond_name, cond_params in CONDITIONS.items():
        for mod in MODS:
            for snr in SNRS:
                for idx in IDXS:
                    i += 1
                    row = run_one(awn, radioml_dict, mod, snr, idx, cond_name, cond_params)
                    rows.append(row)
                    if row["status"] == "error":
                        n_error += 1
                    elif row["status"] == "no_region_detected":
                        n_no_region += 1
                    elif row["status"] == "nan_inf":
                        n_nan += 1
                    if row["fallback_used"]:
                        n_fallback += 1

    print(f"[smoke] DONE: {len(rows)} rows, error={n_error} no_region={n_no_region} "
          f"nan={n_nan} fallback={n_fallback}", flush=True)

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)
    with open(out_dir / "satellite_like_smoke_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # per-condition summary
    summary_rows = []
    for cond_name in CONDITIONS:
        cell = [r for r in rows if r["condition"] == cond_name]
        n = len(cell)
        n_detected = sum(1 for r in cell if r.get("sensing_detected"))
        n_correct = sum(1 for r in cell if r.get("clean_correct") is True)
        n_evaluable = sum(1 for r in cell if r.get("clean_correct") is not None)
        ratios = [r["captured_signal_ratio"] for r in cell if r.get("captured_signal_ratio") is not None]
        summary_rows.append({
            "condition": cond_name, "n": n,
            "n_sensing_detected": n_detected, "sensing_detection_rate": n_detected / n if n else None,
            "n_evaluable_predictions": n_evaluable,
            "n_correct": n_correct,
            "conditional_accuracy": n_correct / n_evaluable if n_evaluable else None,
            "mean_captured_signal_ratio": float(np.mean(ratios)) if ratios else None,
            "n_error": sum(1 for r in cell if r["status"] == "error"),
            "n_no_region": sum(1 for r in cell if r["status"] == "no_region_detected"),
            "n_nan_inf": sum(1 for r in cell if r["status"] == "nan_inf"),
            "n_fallback": sum(1 for r in cell if r["fallback_used"]),
        })
    with open(out_dir / "satellite_like_smoke_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
        print(f"[smoke] {r['condition']}: n={r['n']} detected={r['n_sensing_detected']}/{r['n']} "
              f"correct={r['n_correct']}/{r['n_evaluable_predictions']} "
              f"mean_captured_ratio={r['mean_captured_signal_ratio']}", flush=True)

    print("[smoke] wrote satellite_like_smoke_raw.csv and satellite_like_smoke_summary.csv", flush=True)


if __name__ == "__main__":
    main()
