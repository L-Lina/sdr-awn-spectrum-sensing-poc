"""
Step 4: Final Satellite-like Integrated Experiment. Project-close primary
experiment, answering RQ1-RQ7 (docs/research/SATELLITE_LIKE_FINAL_EXPERIMENT_ZH_TW.md
section 2).

Pipeline (formal, real backends throughout, no oracle/placeholder/bypass in
the main result path):

    RadioML2016.10a -> satellite-like channel (src/channel/satellite_like.py)
    -> long-stream embedding -> Spectrum Sensing -> occupied-region detection
    -> segmentation/max-energy selection -> AWN preprocessing
    -> real AWN clean inference -> optional attack -> attacked inference
    -> optional Top-K -> defended inference

Fairness/pairing (Step 4 task section 9): for a given (modulation, snr,
channel_condition, sample_index), the channel-transformed IQ, sensing
result, and clean crop are computed ONCE and reused across all attack x
Top-K branches -- FGSM and PGD never see a different channel noise
realization than "none". This is why the two-phase structure below exists:
Phase 1 builds all 96 unique base combos once; Phase 2 batches attacks
across the base set; Phase 3 assembles the 576 final rows by combining
each base result with each (attack, topk) branch.

Matrix: 3 modulations x 4 SNR x 4 channel conditions x 3 attacks x 2 topk
x 2 samples/index = 576 (verified by direct multiplication before running,
see main()).

Reuses real, unmodified building blocks throughout: energy_detect,
mask_to_regions, merge_close_regions, filter_by_min_length,
select_aligned_segments, apply_awn_preprocess, to_awn_input,
AWNModelAdapter, AttackAdapter, TopKAdapter, compute_sensing_ground_truth_metrics,
apply_satellite_like_channel. Does not modify external/AWN or
external/adversarial-rf. Does not modify src/channel/satellite_like.py
(Step 3 already validated its semantics; this round only uses it).
"""

from __future__ import annotations

import csv
import hashlib
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.adapters.attack_adapter import AttackAdapter, _REAL_ATTACK_SOURCE  # noqa: E402
from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE  # noqa: E402
from src.adapters.topk_adapter import TopKAdapter, _REAL_SOURCE as _REAL_TOPK_SOURCE  # noqa: E402
from src.channel.satellite_like import apply_satellite_like_channel  # noqa: E402
from src.sensing.energy_detection import energy_detect, filter_by_min_length, mask_to_regions, merge_close_regions  # noqa: E402
from src.sensing.ground_truth_metrics import compute_sensing_ground_truth_metrics  # noqa: E402
from src.sensing.normalize import apply_awn_preprocess, to_awn_input  # noqa: E402
from src.sensing.radioml_source import RML2016_10A_CLASSES, load_radioml_dict, radioml_sample_to_iq  # noqa: E402
from src.sensing.segmentation import select_aligned_segments  # noqa: E402
from src.utils.dataset_path import require_dataset_path_exists, resolve_dataset_path  # noqa: E402

DATASET_PATH = resolve_dataset_path()  # priority: env $SDR_AWN_DATASET_PATH > legacy default
CHECKPOINT_PATH = "external/adversarial-rf/2016.10a_AWN.pkl"

# --- Formal project-close axes ---
MODS = ["BPSK", "QPSK", "8PSK"]
SNRS = [-10, 0, 10, 18]
IDXS = [0, 1]
ATTACKS = ["none", "fgsm", "pgd_det"]
TOPK_STATES = [False, True]
TOPK_K = 20  # matches Phase A's K=20 Top-K defense measurement (experiments/end_to_end_latency_matrix.py)

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
OPT_BATCH_SIZE = 16
OPT_THREADS = 1

NS_PER_MS = 1_000_000.0

# --- 4-level combined channel-condition ladder ---
# Every parameter value here was already individually validated in Step 3
# (unit tests: amplitude {0.5,1.0,2.0}; CFO/Doppler sanity: {500,1000,2000}Hz
# / {250,500,1000}Hz; timing offset {2,8} samples) -- no NEW impairment type
# or untested parameter value is introduced this round.
CHANNEL_CONDITIONS: Dict[str, dict] = {
    "clean":    dict(snr_db=None, amplitude_scale=1.0, cfo_hz=0.0,    doppler_hz=0.0,   timing_offset_samples=0, propagation_delay_ms=None),
    "mild":     dict(snr_db=None, amplitude_scale=1.0, cfo_hz=500.0,  doppler_hz=250.0, timing_offset_samples=2, propagation_delay_ms=26.0),
    "moderate": dict(snr_db=None, amplitude_scale=0.5, cfo_hz=1000.0, doppler_hz=500.0, timing_offset_samples=2, propagation_delay_ms=95.0),
    "strong":   dict(snr_db=15.0, amplitude_scale=0.5, cfo_hz=2000.0, doppler_hz=1000.0, timing_offset_samples=8, propagation_delay_ms=272.0),
}

ATTACK_PARAMS = {
    "fgsm": {"eps": 0.05},
    "pgd_det": {"eps": 0.05, "random_start": False},  # deterministic, validated in Step 3 predecessor rounds
}
# "pgd_det" is this script's own label (distinguishing deterministic PGD
# from a hypothetical future stochastic-PGD condition); AttackAdapter only
# recognizes the underlying torchattacks family name "pgd" -- random_start
# is passed via attack_params, not via the attack name itself.
ATTACK_REAL_NAME = {"fgsm": "fgsm", "pgd_det": "pgd"}


def now_ns() -> int:
    return time.perf_counter_ns()


def sha256_arr(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


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


def build_base_combos():
    return [(m, s, c, i) for m in MODS for s in SNRS for c in CHANNEL_CONDITIONS for i in IDXS]


def compute_base(awn: AWNModelAdapter, radioml_dict: dict, mod: str, snr: int, cond_name: str, idx: int) -> dict:
    """Phase 1: channel -> embed -> sense -> segment -> clean infer, for ONE
    base combo. Returns a dict including the raw crop tensor and AWN-input
    tensor so Phase 2/3 can reuse them (fairness: never regenerated)."""
    base_id = f"{mod}_{snr}_{cond_name}_{idx}"
    row: Dict[str, object] = {
        "base_sample_id": base_id, "modulation": mod, "snr_db_dataset": snr,
        "sample_index": idx, "condition": cond_name, "status": "ok",
        "error_type": None, "error_message": None, "fallback_used": False,
    }
    try:
        block = radioml_dict[(mod, snr)]
        sample_2x128 = block[idx].astype(np.float32)
        clean_burst = radioml_sample_to_iq(sample_2x128)
        true_label = RML2016_10A_CLASSES[mod]

        channel_seed = SEED + idx
        cond_params = CHANNEL_CONDITIONS[cond_name]
        t0 = now_ns()
        impaired_burst, chan_meta = apply_satellite_like_channel(
            clean_burst, sample_rate=SIM_SAMPLE_RATE, seed=channel_seed, **cond_params,
        )
        channel_ms = (now_ns() - t0) / NS_PER_MS

        t0 = now_ns()
        iq, embed_meta = embed_complex_iq_in_noise(impaired_burst, N_SAMPLES, EMBED_SNR_MARGIN, seed=channel_seed)
        embed_ms = (now_ns() - t0) / NS_PER_MS
        true_start, true_end = embed_meta["true_start"], embed_meta["true_end"]
        channel_input_hash = sha256_arr(iq)

        t0 = now_ns()
        mask = energy_detect(iq, window=SENSING_WINDOW_SIZE, threshold_factor=THRESHOLD_FACTOR)
        raw_regions = mask_to_regions(mask)
        merged_regions = merge_close_regions(raw_regions, merge_gap=MERGE_GAP)
        try:
            kept_regions = filter_by_min_length(merged_regions, min_len=MIN_REGION_LEN)
        except RuntimeError:
            kept_regions = []
        sensing_ms = (now_ns() - t0) / NS_PER_MS
        gt = compute_sensing_ground_truth_metrics(true_start, true_end, kept_regions)

        row.update({
            "channel_seed": channel_seed, "true_label": true_label,
            "target_snr_db": cond_params["snr_db"], "amplitude_scale": cond_params["amplitude_scale"],
            "cfo_hz": cond_params["cfo_hz"], "doppler_hz": cond_params["doppler_hz"],
            "timing_offset_samples": cond_params["timing_offset_samples"],
            "propagation_delay_ms": cond_params["propagation_delay_ms"],
            "achieved_snr_db": None,  # only meaningful when snr_db is not None; computed below if so
            "sample_rate": SIM_SAMPLE_RATE,
            "channel_input_power": chan_meta["input_power"], "channel_output_power": chan_meta["output_power"],
            "combined_frequency_offset_hz": chan_meta["combined_frequency_offset_hz"],
            "channel_input_hash": channel_input_hash,
            "true_start": true_start, "true_end": true_end,
            "sensing_detected": gt["detection_success"],
            "detected_start": gt["best_detected_start"], "detected_end": gt["best_detected_end"],
            "boundary_start_error": gt["start_boundary_error"], "boundary_end_error": gt["end_boundary_error"],
            "captured_signal_ratio": gt["captured_signal_ratio"],
            "false_occupied_samples": gt.get("false_occupied_sample_count"),
            "channel_ms": channel_ms, "embed_ms": embed_ms, "sensing_ms": sensing_ms,
        })

        if cond_params["snr_db"] is not None:
            # achieved SNR relative to the channel's own AWGN, measured against a
            # snr_db=None re-run with identical other params (Step 3 methodology).
            ref_burst, _ = apply_satellite_like_channel(
                clean_burst, sample_rate=SIM_SAMPLE_RATE, seed=channel_seed,
                **{**cond_params, "snr_db": None},
            )
            noise_realized = impaired_burst - ref_burst
            sig_power = float(np.mean(np.abs(ref_burst) ** 2))
            noise_power = float(np.mean(np.abs(noise_realized) ** 2))
            row["achieved_snr_db"] = 10.0 * np.log10(sig_power / noise_power) if noise_power > 0 else float("inf")

        if not kept_regions:
            row["status"] = "no_region_detected"
            row["segmentation_ms"] = None
            row["awn_clean_ms"] = None
            row["clean_prediction"] = None
            row["clean_correct"] = None
            row["clean_confidence"] = None
            row["awn_input"] = None
            row["clean_segment_hash"] = None
            return row

        t0 = now_ns()
        segments, _ = select_aligned_segments(iq, kept_regions, seg_len=128, policy=ALIGNMENT_POLICY, hop=1)
        segmentation_ms = (now_ns() - t0) / NS_PER_MS
        row["segmentation_ms"] = segmentation_ms

        if segments.shape[0] == 0:
            row["status"] = "no_segment_produced"
            row["awn_clean_ms"] = None
            row["clean_prediction"] = None
            row["clean_correct"] = None
            row["clean_confidence"] = None
            row["awn_input"] = None
            row["clean_segment_hash"] = None
            return row

        t0 = now_ns()
        x = apply_awn_preprocess(segments[:1], policy=AWN_PREPROCESS)
        x = to_awn_input(x, seg_len=128)
        awn_preprocess_ms = (now_ns() - t0) / NS_PER_MS

        t0 = now_ns()
        logits, meta_clean = awn.infer(x, seed=SEED)
        awn_clean_ms = (now_ns() - t0) / NS_PER_MS
        if meta_clean["awn_backend"] != _REAL_MODEL_SOURCE:
            row["fallback_used"] = True

        probs = np.exp(logits[0] - np.max(logits[0]))
        probs = probs / probs.sum()
        pred = int(np.argmax(logits[0]))

        row["awn_preprocess_ms"] = awn_preprocess_ms
        row["awn_clean_ms"] = awn_clean_ms
        row["clean_prediction"] = pred
        row["clean_correct"] = (pred == true_label)
        row["clean_confidence"] = float(probs[pred])
        row["clean_segment_hash"] = sha256_arr(x)
        row["awn_input"] = x  # kept for Phase 2/3, stripped before CSV write
        row["clean_logits"] = logits[0].copy()

        if not np.isfinite(logits).all():
            row["status"] = "nan_inf"

    except Exception as exc:  # noqa: BLE001 -- fail closed
        row["status"] = "error"
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc)
        row["awn_input"] = None
    return row


def run_attack_batched(attack: AttackAdapter, awn: AWNModelAdapter, base_rows: List[dict], attack_name: str,
                        threads_module) -> Dict[str, dict]:
    """Phase 2: batch the given attack across all base rows with a valid
    clean crop, batch_size=OPT_BATCH_SIZE, threads=OPT_THREADS (Step 1's
    validated optimization). Returns {base_sample_id: {attack fields}}."""
    valid = [r for r in base_rows if r.get("awn_input") is not None]
    eps = ATTACK_PARAMS[attack_name]["eps"]
    params = ATTACK_PARAMS[attack_name]
    default_threads = threads_module.get_num_threads()
    threads_module.set_num_threads(OPT_THREADS)
    results: Dict[str, dict] = {}
    for start in range(0, len(valid), OPT_BATCH_SIZE):
        batch_rows = valid[start:start + OPT_BATCH_SIZE]
        batch = np.concatenate([r["awn_input"] for r in batch_rows], axis=0)
        t0 = now_ns()
        x_adv_batch, attack_meta = attack.apply(batch, attack=ATTACK_REAL_NAME[attack_name], eps=eps, seed=SEED, attack_params=params)
        batch_ms = (now_ns() - t0) / NS_PER_MS
        per_sample_attack_ms = batch_ms / batch.shape[0]
        fallback = attack_meta["attack_backend"] != _REAL_ATTACK_SOURCE or attack_meta["attack_status"] != "ok"
        for i, r in enumerate(batch_rows):
            x_single = x_adv_batch[i:i + 1]
            t1 = now_ns()
            logits_att, meta_att = awn.infer(x_single, seed=SEED)
            attacked_inf_ms = (now_ns() - t1) / NS_PER_MS
            pred_att = int(np.argmax(logits_att[0]))
            perturb = x_single.astype(np.float64) - r["awn_input"].astype(np.float64)
            results[r["base_sample_id"]] = {
                "attack_generation_ms": per_sample_attack_ms, "attacked_inference_ms": attacked_inf_ms,
                "attacked_prediction": pred_att, "attacked_correct": (pred_att == r["true_label"]),
                "attack_success": (pred_att != r["clean_prediction"]),
                "perturbation_linf": float(np.max(np.abs(perturb))), "perturbation_l2": float(np.linalg.norm(perturb)),
                "model_mode_after": "train" if attack.wrapped_model.training else "eval",
                "fallback_used": fallback or (meta_att["awn_backend"] != _REAL_MODEL_SOURCE),
                "x_adv": x_single,
            }
    threads_module.set_num_threads(default_threads)
    return results


def run_topk(topk_adapter: TopKAdapter, awn: AWNModelAdapter, x_input: np.ndarray, true_label: int, clean_pred: int) -> dict:
    t0 = now_ns()
    x_defended, topk_meta = topk_adapter.apply(x_input, topk=TOPK_K)
    topk_ms = (now_ns() - t0) / NS_PER_MS
    t0 = now_ns()
    logits_def, meta_def = awn.infer(x_defended, seed=SEED)
    defended_inference_ms = (now_ns() - t0) / NS_PER_MS
    pred_def = int(np.argmax(logits_def[0]))
    return {
        "topk_ms": topk_ms, "defended_inference_ms": defended_inference_ms,
        "defended_prediction": pred_def, "defended_correct": (pred_def == true_label),
        "fallback_used": (topk_meta["topk_backend"] != _REAL_TOPK_SOURCE) or (meta_def["awn_backend"] != _REAL_MODEL_SOURCE),
    }


def run_matrix(awn, attack, topk_adapter, radioml_dict, mods, snrs, conditions, idxs, torch_module) -> List[dict]:
    base_combos = [(m, s, c, i) for m in mods for s in snrs for c in conditions for i in idxs]
    print(f"[final] Phase 1: computing {len(base_combos)} base combos (channel+sensing+clean AWN) ...", flush=True)
    base_rows = []
    for m, s, c, i in base_combos:
        base_rows.append(compute_base(awn, radioml_dict, m, s, c, i))
    n_ok_base = sum(1 for r in base_rows if r.get("awn_input") is not None)
    print(f"[final] Phase 1 done: {len(base_rows)} base rows, {n_ok_base} with a valid clean crop", flush=True)

    print("[final] Phase 2: batched attacks (fgsm, pgd_det) ...", flush=True)
    attack_results: Dict[str, Dict[str, dict]] = {}
    for attack_name in ["fgsm", "pgd_det"]:
        attack_results[attack_name] = run_attack_batched(attack, awn, base_rows, attack_name, torch_module)
        print(f"[final]   {attack_name}: {len(attack_results[attack_name])} attacked", flush=True)

    print("[final] Phase 3: assembling final rows (attack x topk branches) ...", flush=True)
    final_rows = []
    for base in base_rows:
        for attack_name in ATTACKS:
            for topk_on in TOPK_STATES:
                row = {k: v for k, v in base.items() if k not in ("awn_input", "clean_logits")}
                row["attack_name"] = attack_name
                row["topk"] = TOPK_K if topk_on else None

                if base.get("awn_input") is None:
                    row["attacked_prediction"] = None
                    row["attacked_correct"] = None
                    row["attack_success"] = None
                    row["defended_prediction"] = None
                    row["defended_correct"] = None
                    final_rows.append(row)
                    continue

                if attack_name == "none":
                    tensor_for_topk = base["awn_input"]
                    row["attack_generation_ms"] = 0.0
                    row["attacked_inference_ms"] = None
                    row["attacked_prediction"] = base["clean_prediction"]
                    row["attacked_correct"] = base["clean_correct"]
                    row["attack_success"] = False
                    row["perturbation_linf"] = 0.0
                    row["perturbation_l2"] = 0.0
                    row["model_mode_after"] = "eval"
                else:
                    ares = attack_results[attack_name].get(base["base_sample_id"])
                    if ares is None:
                        row["status"] = "attack_missing"
                        final_rows.append(row)
                        continue
                    tensor_for_topk = ares["x_adv"]
                    for k in ["attack_generation_ms", "attacked_inference_ms", "attacked_prediction",
                              "attacked_correct", "attack_success", "perturbation_linf", "perturbation_l2",
                              "model_mode_after"]:
                        row[k] = ares[k]
                    row["fallback_used"] = row.get("fallback_used") or ares["fallback_used"]

                if topk_on:
                    tres = run_topk(topk_adapter, awn, tensor_for_topk, base["true_label"], base["clean_prediction"])
                    row.update(tres)
                    row["fallback_used"] = row.get("fallback_used") or tres["fallback_used"]
                    clean_ok = base.get("clean_correct")
                    row["recovered_by_defense"] = (row["attacked_correct"] is False and row["defended_correct"] is True)
                    row["clean_degraded_by_defense"] = (attack_name == "none" and clean_ok is True and row["defended_correct"] is False)
                else:
                    row["topk_ms"] = None
                    row["defended_inference_ms"] = None
                    row["defended_prediction"] = None
                    row["defended_correct"] = None
                    row["recovered_by_defense"] = None
                    row["clean_degraded_by_defense"] = None

                total_ms = sum(x for x in [
                    row.get("channel_ms"), row.get("embed_ms"), row.get("sensing_ms"),
                    row.get("segmentation_ms"), row.get("awn_preprocess_ms"), row.get("awn_clean_ms"),
                    row.get("attack_generation_ms"), row.get("attacked_inference_ms"),
                    row.get("topk_ms"), row.get("defended_inference_ms"),
                ] if x is not None)
                row["total_ms"] = total_ms
                final_rows.append(row)

    return final_rows


def main() -> None:
    out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    smoke_only = "--smoke" in sys.argv
    dataset_path = sys.argv[sys.argv.index("--dataset-path") + 1] if "--dataset-path" in sys.argv else DATASET_PATH
    require_dataset_path_exists(dataset_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    print(f"[final] loading real backends + dataset (dataset_path={dataset_path}) ...", flush=True)
    awn = AWNModelAdapter(checkpoint_path=CHECKPOINT_PATH, device="cpu")
    if awn.backend_name != _REAL_MODEL_SOURCE or awn.status != "ok":
        raise RuntimeError(f"Real-AWN precheck FAILED: backend={awn.backend_name} status={awn.status}")
    attack = AttackAdapter(awn_model=awn.model, device="cpu")
    if attack.wrapped_model is None or attack.backend_name != _REAL_ATTACK_SOURCE:
        raise RuntimeError(f"Real-attack precheck FAILED: backend={attack.backend_name}")
    topk_adapter = TopKAdapter()
    if not topk_adapter.backend_available or topk_adapter.backend_name != _REAL_TOPK_SOURCE:
        raise RuntimeError(f"Real Top-K precheck FAILED: backend={topk_adapter.backend_name}")
    print(f"[final] real backends confirmed: AWN={_REAL_MODEL_SOURCE}, attack={_REAL_ATTACK_SOURCE}, topk={_REAL_TOPK_SOURCE}", flush=True)

    radioml_dict = load_radioml_dict(dataset_path)
    print(f"[final] dataset loaded from {dataset_path}, {len(radioml_dict)} (mod,snr) cells", flush=True)

    if smoke_only:
        mods, snrs, conditions, idxs = ["QPSK"], [0], list(CHANNEL_CONDITIONS.keys()), [0]
        rows = run_matrix(awn, attack, topk_adapter, radioml_dict, mods, snrs, conditions, idxs, torch)
        print(f"[final] SMOKE: {len(rows)} rows (expect 24)", flush=True)
        fname = "satellite_like_final_smoke_raw.csv"
    else:
        rows = run_matrix(awn, attack, topk_adapter, radioml_dict, MODS, SNRS, list(CHANNEL_CONDITIONS.keys()), IDXS, torch)
        print(f"[final] FULL: {len(rows)} rows (expect 576)", flush=True)
        fname = "raw_results.csv"

    n_error = sum(1 for r in rows if r["status"] == "error")
    n_no_region = sum(1 for r in rows if r["status"] == "no_region_detected")
    n_no_segment = sum(1 for r in rows if r["status"] == "no_segment_produced")
    n_nan = sum(1 for r in rows if r["status"] == "nan_inf")
    n_fallback = sum(1 for r in rows if r.get("fallback_used"))
    print(f"[final] status: error={n_error} no_region={n_no_region} no_segment={n_no_segment} "
          f"nan_inf={n_nan} fallback={n_fallback}", flush=True)

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)
    with open(out_dir / fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[final] wrote {fname}", flush=True)


if __name__ == "__main__":
    main()
