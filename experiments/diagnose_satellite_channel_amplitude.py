"""
Focused validation of amplitude/AWGN channel semantics (Step 3 follow-up),
answering: does amplitude_scale preserve requested SNR, and what actually
causes the amplitude-only accuracy drop observed in the 108-sample smoke
test (55.6% baseline -> 16.7% amplitude-only)? Does not add any new channel
impairment, does not modify src/channel/satellite_like.py.

Three parts, all against real backends (real AWN, real sensing):

Part A (achieved SNR grid): for amplitude_scale in {0.5, 1.0, 2.0} x
snr_db target in {-10, 0, 18}, >=20 samples each, calls
apply_satellite_like_channel() TWICE per sample with identical parameters
except snr_db (once at the target value, once at snr_db=None) to isolate
the exact injected noise realization -- achieved SNR is computed from the
REAL noise actually added, not re-derived analytically. Writes
amplitude_snr_validation.csv.

Part B (crop/tensor/logits trace): for the SAME 18-sample set used in the
original smoke test (3 mods x 3 SNRs x 2 idx), amplitude_scale in
{0.5, 1.0, 2.0}, snr_db=None (matching the smoke test's B_amplitude
condition exactly) -- traces satellite-channel output, sensed crop,
captured_signal_ratio, detected window, raw crop tensor, AWN-preprocessed
tensor, clean logits, prediction, at every stage, and computes max-abs-diff
between amplitude=1.0 (reference) and amplitude={0.5,2.0}.

Part C (oracle vs sensing crop): for the same samples/amplitudes, also
extracts a segment using the KNOWN true_start/true_end (bypassing
energy_detect entirely) and compares its accuracy against the
sensing-detected crop's accuracy, to separate "is degradation caused by
sensing/window selection" from "is it downstream of a correctly-selected
window".

Reuses real, unmodified building blocks throughout (energy_detect,
mask_to_regions, merge_close_regions, filter_by_min_length,
select_aligned_segments, apply_awn_preprocess, to_awn_input,
AWNModelAdapter, compute_sensing_ground_truth_metrics). Does not modify
external/AWN or external/adversarial-rf.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

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
N_SAMPLES = 8192
EMBED_SNR_MARGIN = 20.0
SIM_SAMPLE_RATE = 200_000.0
THRESHOLD_FACTOR = 5.0
SENSING_WINDOW_SIZE = 128
MIN_REGION_LEN = 128
MERGE_GAP = 0
ALIGNMENT_POLICY = "max-energy"
AWN_PREPROCESS = "radioml-native"  # matches the original 108-sample smoke test exactly
SEED = 0


def embed_complex_iq_in_noise(burst_iq: np.ndarray, n_samples: int, embed_snr_margin: float, seed: int):
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


def part_a_achieved_snr(radioml_dict: dict, out_dir: Path) -> None:
    print("\n=== Part A: achieved SNR grid ===", flush=True)
    rows = []
    combos = [(m, s, i) for m in MODS for s in SNRS for i in range(20)]  # up to 20 idx per (mod,snr)
    sample_i = 0
    for amp in [0.5, 1.0, 2.0]:
        for target_snr in [-10, 0, 18]:
            n_ok = 0
            for k in range(20):
                mod = MODS[k % len(MODS)]
                snr_cell = SNRS[k % len(SNRS)]
                idx = k // len(SNRS)
                block = radioml_dict[(mod, snr_cell)]
                if idx >= block.shape[0]:
                    idx = idx % block.shape[0]
                sample_2x128 = block[idx].astype(np.float32)
                clean_burst = radioml_sample_to_iq(sample_2x128)

                seed = 10_000 + k
                out_with_awgn, meta_with = apply_satellite_like_channel(
                    clean_burst, sample_rate=SIM_SAMPLE_RATE, snr_db=target_snr, amplitude_scale=amp,
                    cfo_hz=0.0, doppler_hz=0.0, timing_offset_samples=0, seed=seed,
                )
                out_no_awgn, meta_no = apply_satellite_like_channel(
                    clean_burst, sample_rate=SIM_SAMPLE_RATE, snr_db=None, amplitude_scale=amp,
                    cfo_hz=0.0, doppler_hz=0.0, timing_offset_samples=0, seed=seed,
                )
                power_before_amp = meta_with["input_power"]
                power_after_amp = float(np.mean(np.abs(out_no_awgn) ** 2))
                realized_noise = out_with_awgn - out_no_awgn
                injected_noise_power = float(np.mean(np.abs(realized_noise) ** 2))
                achieved_snr_db = 10.0 * np.log10(power_after_amp / injected_noise_power) if injected_noise_power > 0 else float("inf")
                output_power = meta_with["output_power"]

                rows.append({
                    "amplitude_scale": amp, "target_snr_db": target_snr,
                    "modulation": mod, "sample_snr_cell": snr_cell, "sample_index": idx, "seed": seed,
                    "input_power_before_amplitude": power_before_amp,
                    "power_after_amplitude": power_after_amp,
                    "injected_noise_power": injected_noise_power,
                    "achieved_snr_db": achieved_snr_db,
                    "output_power": output_power,
                })
                n_ok += 1
            achieved_list = [r["achieved_snr_db"] for r in rows if r["amplitude_scale"] == amp and r["target_snr_db"] == target_snr]
            print(f"[A] amplitude={amp} target_snr={target_snr}dB: n={n_ok} "
                  f"mean_achieved={np.mean(achieved_list):.4f}dB std={np.std(achieved_list):.4f}dB", flush=True)

    fieldnames = list(rows[0].keys())
    with open(out_dir / "amplitude_snr_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[A] wrote amplitude_snr_validation.csv, {len(rows)} rows", flush=True)


def part_b_c_trace(awn: AWNModelAdapter, radioml_dict: dict, out_dir: Path) -> None:
    print("\n=== Part B/C: crop/tensor/logits trace + oracle vs sensing ===", flush=True)
    idxs = [0, 1]
    rows = []
    for amp in [0.5, 1.0, 2.0]:
        for mod in MODS:
            for snr in SNRS:
                for idx in idxs:
                    sample_2x128 = radioml_dict[(mod, snr)][idx].astype(np.float32)
                    clean_burst = radioml_sample_to_iq(sample_2x128)
                    true_label = RML2016_10A_CLASSES[mod]

                    impaired_burst, chan_meta = apply_satellite_like_channel(
                        clean_burst, sample_rate=SIM_SAMPLE_RATE, snr_db=None, amplitude_scale=amp,
                        cfo_hz=0.0, doppler_hz=0.0, timing_offset_samples=0, seed=SEED + idx,
                    )
                    iq, embed_meta = embed_complex_iq_in_noise(impaired_burst, N_SAMPLES, EMBED_SNR_MARGIN, seed=SEED + idx)
                    true_start, true_end = embed_meta["true_start"], embed_meta["true_end"]

                    # --- sensing crop ---
                    mask = energy_detect(iq, window=SENSING_WINDOW_SIZE, threshold_factor=THRESHOLD_FACTOR)
                    raw_regions = mask_to_regions(mask)
                    merged_regions = merge_close_regions(raw_regions, merge_gap=MERGE_GAP)
                    try:
                        kept_regions = filter_by_min_length(merged_regions, min_len=MIN_REGION_LEN)
                    except RuntimeError:
                        kept_regions = []
                    gt = compute_sensing_ground_truth_metrics(true_start, true_end, kept_regions)

                    row = {
                        "amplitude_scale": amp, "modulation": mod, "snr": snr, "sample_index": idx,
                        "true_start": true_start, "true_end": true_end,
                        "sensing_detected": gt["detection_success"],
                        "captured_signal_ratio": gt["captured_signal_ratio"],
                        "start_boundary_error": gt["start_boundary_error"],
                        "end_boundary_error": gt["end_boundary_error"],
                    }

                    for path_name, use_oracle in [("sensing", False), ("oracle", True)]:
                        if use_oracle:
                            regions_for_path = [(true_start, true_end)]
                        else:
                            regions_for_path = kept_regions
                        if not regions_for_path:
                            row[f"{path_name}_status"] = "no_region"
                            row[f"{path_name}_prediction"] = None
                            row[f"{path_name}_correct"] = None
                            continue
                        segments, _ = select_aligned_segments(iq, regions_for_path, seg_len=128, policy=ALIGNMENT_POLICY, hop=1)
                        if segments.shape[0] == 0:
                            row[f"{path_name}_status"] = "no_segment"
                            row[f"{path_name}_prediction"] = None
                            row[f"{path_name}_correct"] = None
                            continue
                        raw_crop = segments[0].copy()
                        x = apply_awn_preprocess(segments[:1], policy=AWN_PREPROCESS)
                        awn_input = x.copy()
                        x = to_awn_input(x, seg_len=128)
                        logits, meta_awn = awn.infer(x, seed=SEED)
                        pred = int(np.argmax(logits[0]))

                        row[f"{path_name}_status"] = "ok"
                        row[f"{path_name}_raw_crop_max_abs"] = float(np.max(np.abs(raw_crop)))
                        row[f"{path_name}_awn_input_max_abs"] = float(np.max(np.abs(awn_input)))
                        row[f"{path_name}_logits"] = ";".join(f"{v:.6f}" for v in logits[0].tolist())
                        row[f"{path_name}_prediction"] = pred
                        row[f"{path_name}_correct"] = (pred == true_label)
                        row[f"_{path_name}_raw_crop"] = raw_crop  # kept in-memory only, for diff calc below
                        row[f"_{path_name}_awn_input"] = awn_input
                        row[f"_{path_name}_logits_arr"] = logits[0].copy()

                    rows.append(row)

    # compute diffs relative to amplitude=1.0 reference, matched by (modulation, snr, sample_index)
    ref = {(r["modulation"], r["snr"], r["sample_index"]): r for r in rows if r["amplitude_scale"] == 1.0}
    for r in rows:
        key = (r["modulation"], r["snr"], r["sample_index"])
        r0 = ref.get(key)
        for path_name in ["sensing", "oracle"]:
            r[f"{path_name}_crop_max_abs_diff_vs_amp1"] = None
            r[f"{path_name}_awn_input_max_abs_diff_vs_amp1"] = None
            r[f"{path_name}_logits_max_abs_diff_vs_amp1"] = None
            r[f"{path_name}_prediction_match_vs_amp1"] = None
            if r0 is None or r.get(f"{path_name}_status") != "ok" or r0.get(f"{path_name}_status") != "ok":
                continue
            crop_diff = float(np.max(np.abs(r[f"_{path_name}_raw_crop"] - r0[f"_{path_name}_raw_crop"])))
            awn_in_diff = float(np.max(np.abs(r[f"_{path_name}_awn_input"] - r0[f"_{path_name}_awn_input"])))
            logits_diff = float(np.max(np.abs(r[f"_{path_name}_logits_arr"] - r0[f"_{path_name}_logits_arr"])))
            r[f"{path_name}_crop_max_abs_diff_vs_amp1"] = crop_diff
            r[f"{path_name}_awn_input_max_abs_diff_vs_amp1"] = awn_in_diff
            r[f"{path_name}_logits_max_abs_diff_vs_amp1"] = logits_diff
            r[f"{path_name}_prediction_match_vs_amp1"] = (r[f"{path_name}_prediction"] == r0[f"{path_name}_prediction"])

    out_fields = [k for k in rows[0].keys() if not k.startswith("_")]
    with open(out_dir / "amplitude_trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in out_fields})

    # summary
    for amp in [0.5, 1.0, 2.0]:
        cell = [r for r in rows if r["amplitude_scale"] == amp]
        n = len(cell)
        n_detected = sum(1 for r in cell if r["sensing_detected"])
        n_sensing_correct = sum(1 for r in cell if r.get("sensing_correct") is True)
        n_oracle_correct = sum(1 for r in cell if r.get("oracle_correct") is True)
        n_sensing_eval = sum(1 for r in cell if r.get("sensing_correct") is not None)
        n_oracle_eval = sum(1 for r in cell if r.get("oracle_correct") is not None)
        pred_match_sensing = [r["sensing_prediction_match_vs_amp1"] for r in cell if r["sensing_prediction_match_vs_amp1"] is not None]
        pred_match_oracle = [r["oracle_prediction_match_vs_amp1"] for r in cell if r["oracle_prediction_match_vs_amp1"] is not None]
        awn_diffs = [r["sensing_awn_input_max_abs_diff_vs_amp1"] for r in cell if r["sensing_awn_input_max_abs_diff_vs_amp1"] is not None]
        print(f"[B/C] amplitude={amp}: n={n} detected={n_detected}/{n} "
              f"sensing_acc={n_sensing_correct}/{n_sensing_eval} oracle_acc={n_oracle_correct}/{n_oracle_eval} "
              f"pred_match_vs_amp1(sensing)={np.mean(pred_match_sensing) if pred_match_sensing else None} "
              f"pred_match_vs_amp1(oracle)={np.mean(pred_match_oracle) if pred_match_oracle else None} "
              f"mean_awn_input_max_abs_diff={np.mean(awn_diffs) if awn_diffs else None}", flush=True)

    print(f"[B/C] wrote amplitude_trace.csv, {len(rows)} rows", flush=True)


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

    part_a_achieved_snr(radioml_dict, out_dir)
    part_b_c_trace(awn, radioml_dict, out_dir)

    print("\n[diag] DONE", flush=True)


if __name__ == "__main__":
    main()
