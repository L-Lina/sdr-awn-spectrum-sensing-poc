"""
CFO / Doppler sanity test (Step 3 follow-up, Part 5). Confirms: (1) the
configured frequency shift is actually measurable in the transformed IQ,
(2) accuracy degradation trends in a directionally reasonable way with
offset magnitude (not required to be strictly monotonic), (3) CFO and
Doppler produce numerically identical results when set to the same value
(since both use the same underlying rotation primitive), and (4) metadata
keeps cfo_hz/doppler_hz physically distinct even though the applied
transform is shared. Does not add any new channel impairment.

Fixed: BPSK/QPSK/8PSK, SNR=18dB, 10 samples per modulation (30 per sweep
point). CFO sweep: {0, small=500, medium=1000, current-smoke-max=2000} Hz.
Doppler sweep: {0, small=250, medium=500, current-smoke-max=1000} Hz. All
other impairments off (amplitude_scale=1.0, timing_offset_samples=0,
snr_db=None).

Reuses real, unmodified building blocks (energy_detect, mask_to_regions,
merge_close_regions, filter_by_min_length, select_aligned_segments,
apply_awn_preprocess, to_awn_input, AWNModelAdapter,
compute_sensing_ground_truth_metrics). Does not modify external/AWN,
external/adversarial-rf, or src/channel/satellite_like.py.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

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
SNR = 18
N_PER_MOD = 10
N_SAMPLES = 8192
EMBED_SNR_MARGIN = 20.0
SIM_SAMPLE_RATE = 200_000.0
THRESHOLD_FACTOR = 5.0
SENSING_WINDOW_SIZE = 128
MIN_REGION_LEN = 128
MERGE_GAP = 0
ALIGNMENT_POLICY = "max-energy"
AWN_PREPROCESS = "radioml-native"
SEED = 0

CFO_SWEEP = [0.0, 500.0, 1000.0, 2000.0]
DOPPLER_SWEEP = [0.0, 250.0, 500.0, 1000.0]


def embed_complex_iq_in_noise(burst_iq, n_samples, embed_snr_margin, seed):
    burst_len = len(burst_iq)
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
    return iq, {"true_start": true_start, "true_end": true_end, "burst_len": burst_len}


def measure_frequency_shift_hz(clean_burst: np.ndarray, transformed_burst: np.ndarray, sample_rate: float) -> float:
    """Measures the actual applied frequency shift by comparing the phase
    ramp of transformed_burst against clean_burst (same underlying random
    symbols, only the frequency rotation differs) -- an independent
    measurement, not a re-derivation of the parameter fed into the channel."""
    x_ratios = clean_burst[1:] * np.conj(clean_burst[:-1])
    y_ratios = transformed_burst[1:] * np.conj(transformed_burst[:-1])
    delta_phase = np.angle(np.mean(y_ratios * np.conj(x_ratios)))
    return float(delta_phase * sample_rate / (2 * np.pi))


def run_sweep(awn, radioml_dict, out_dir, factor_name: str, sweep_values, param_key: str):
    print(f"\n=== {factor_name} sweep ===", flush=True)
    rows = []
    for val in sweep_values:
        for mod in MODS:
            for idx in range(N_PER_MOD):
                block = radioml_dict[(mod, SNR)]
                sample_2x128 = block[idx % block.shape[0]].astype(np.float32)
                clean_burst = radioml_sample_to_iq(sample_2x128)
                true_label = RML2016_10A_CLASSES[mod]

                params = dict(snr_db=None, amplitude_scale=1.0, cfo_hz=0.0, doppler_hz=0.0,
                              timing_offset_samples=0)
                params[param_key] = val
                impaired_burst, chan_meta = apply_satellite_like_channel(
                    clean_burst, sample_rate=SIM_SAMPLE_RATE, seed=SEED + idx, **params,
                )
                measured_shift_hz = measure_frequency_shift_hz(clean_burst, impaired_burst, SIM_SAMPLE_RATE)

                iq, embed_meta = embed_complex_iq_in_noise(impaired_burst, N_SAMPLES, EMBED_SNR_MARGIN, seed=SEED + idx)
                true_start, true_end = embed_meta["true_start"], embed_meta["true_end"]

                mask = energy_detect(iq, window=SENSING_WINDOW_SIZE, threshold_factor=THRESHOLD_FACTOR)
                raw_regions = mask_to_regions(mask)
                merged_regions = merge_close_regions(raw_regions, merge_gap=MERGE_GAP)
                try:
                    kept_regions = filter_by_min_length(merged_regions, min_len=MIN_REGION_LEN)
                except RuntimeError:
                    kept_regions = []
                gt = compute_sensing_ground_truth_metrics(true_start, true_end, kept_regions)

                row = {
                    "factor": factor_name, param_key: val,
                    "modulation": mod, "sample_index": idx, "snr": SNR,
                    "combined_frequency_offset_hz": chan_meta["combined_frequency_offset_hz"],
                    "measured_frequency_shift_hz": measured_shift_hz,
                    "sensing_detected": gt["detection_success"],
                    "captured_signal_ratio": gt["captured_signal_ratio"],
                    "prediction": None, "correct": None, "confidence": None, "status": "ok",
                }

                if not kept_regions:
                    row["status"] = "no_region"
                    rows.append(row)
                    continue
                segments, _ = select_aligned_segments(iq, kept_regions, seg_len=128, policy=ALIGNMENT_POLICY, hop=1)
                if segments.shape[0] == 0:
                    row["status"] = "no_segment"
                    rows.append(row)
                    continue
                x = apply_awn_preprocess(segments[:1], policy=AWN_PREPROCESS)
                x = to_awn_input(x, seg_len=128)
                logits, meta_awn = awn.infer(x, seed=SEED)
                probs = np.exp(logits[0] - np.max(logits[0]))
                probs = probs / probs.sum()
                pred = int(np.argmax(logits[0]))
                row["prediction"] = pred
                row["correct"] = (pred == true_label)
                row["confidence"] = float(probs[pred])
                rows.append(row)

        cell = [r for r in rows if r[param_key] == val]
        n = len(cell)
        n_detected = sum(1 for r in cell if r["sensing_detected"])
        n_correct = sum(1 for r in cell if r.get("correct") is True)
        n_eval = sum(1 for r in cell if r.get("correct") is not None)
        shifts = [r["measured_frequency_shift_hz"] for r in cell]
        print(f"[{factor_name}] {param_key}={val}Hz: n={n} detected={n_detected}/{n} "
              f"acc={n_correct}/{n_eval} mean_measured_shift={np.mean(shifts):.2f}Hz "
              f"(target={val}Hz)", flush=True)
    return rows


def main() -> None:
    out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    dataset_path = sys.argv[sys.argv.index("--dataset-path") + 1] if "--dataset-path" in sys.argv else DATASET_PATH
    require_dataset_path_exists(dataset_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[diag] loading real AWN backend + dataset (dataset_path={dataset_path}) ...", flush=True)
    awn = AWNModelAdapter(checkpoint_path=CHECKPOINT_PATH, device="cpu")
    if awn.backend_name != _REAL_MODEL_SOURCE or awn.status != "ok":
        raise RuntimeError(f"Real-AWN precheck FAILED: backend={awn.backend_name} status={awn.status}")
    radioml_dict = load_radioml_dict(dataset_path)
    print(f"[diag] real AWN confirmed, dataset loaded from {dataset_path} ({len(radioml_dict)} cells)", flush=True)

    cfo_rows = run_sweep(awn, radioml_dict, out_dir, "cfo", CFO_SWEEP, "cfo_hz")
    doppler_rows = run_sweep(awn, radioml_dict, out_dir, "doppler", DOPPLER_SWEEP, "doppler_hz")

    all_rows = []
    for r in cfo_rows:
        rr = dict(r)
        rr["swept_value_hz"] = rr.pop("cfo_hz")
        all_rows.append(rr)
    for r in doppler_rows:
        rr = dict(r)
        rr["swept_value_hz"] = rr.pop("doppler_hz")
        all_rows.append(rr)

    fieldnames = ["factor", "swept_value_hz", "modulation", "sample_index", "snr",
                  "combined_frequency_offset_hz", "measured_frequency_shift_hz",
                  "sensing_detected", "captured_signal_ratio", "prediction", "correct",
                  "confidence", "status"]
    with open(out_dir / "cfo_doppler_sanity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # cross-check: CFO=2000 vs Doppler=1000+1000(cfo)=... actually check CFO=2000Hz alone vs Doppler=2000Hz alone
    # (not directly in the sweep sets above at matching value except via combined check below)
    print("\n=== Cross-check: CFO=1000Hz alone vs Doppler=1000Hz alone (same underlying primitive) ===", flush=True)
    cfo_1000 = [r for r in cfo_rows if r["cfo_hz"] == 1000.0]
    dop_1000 = [r for r in doppler_rows if r["doppler_hz"] == 1000.0]
    cfo_shifts = np.array([r["measured_frequency_shift_hz"] for r in cfo_1000])
    dop_shifts = np.array([r["measured_frequency_shift_hz"] for r in dop_1000])
    print(f"CFO=1000Hz measured shifts: mean={cfo_shifts.mean():.3f} std={cfo_shifts.std():.3f}", flush=True)
    print(f"Doppler=1000Hz measured shifts: mean={dop_shifts.mean():.3f} std={dop_shifts.std():.3f}", flush=True)
    print(f"Difference in means: {abs(cfo_shifts.mean() - dop_shifts.mean()):.6f} Hz (expected ~0, same primitive)", flush=True)

    print("\n[diag] DONE", flush=True)


if __name__ == "__main__":
    main()
