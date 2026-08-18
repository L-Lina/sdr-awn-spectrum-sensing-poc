"""
Formal CPU latency benchmark for the spectrum-sensing -> AWN -> attack ->
Top-K pipeline. Reuses the exact real building blocks every other formal
script in this repo uses (energy_detect, mask_to_regions,
merge_close_regions, filter_by_min_length, select_aligned_segments,
apply_awn_preprocess, to_awn_input, AWNModelAdapter, AttackAdapter,
TopKAdapter) -- this is an external-timer wrapper around the same
functions, not a new or placeholder pipeline. No function internal to
src/sensing/*.py or src/adapters/*.py is modified.

Two independently runnable phases:

  --mode clean_sensing  (Phase A): 11 modulations x 20 SNRs x 10 samples
      per cell = 2200 base samples, timing every clean-path stage plus an
      optional Top-K + defended-inference pass.

  --mode attack_baseline  (Phase B): 11 modulations x 3 SNRs x 10 unique
      base samples = 330 samples per attack, for a caller-specified subset
      of {fgsm, pgd, cw}, timing attack-specific stages.

Stages that are NOT separably measurable in the current implementation
(the noise-floor/threshold computation inside energy_detect, and the
max-energy window search inside select_aligned_segments) are recorded as
NA with an explicit reason, per this round's instruction not to hard-split
or estimate coupled internals.

All timing uses time.perf_counter_ns(). Warm-up samples (excluded from the
recorded/aggregated CSV) run before any timed sample. Dataset loading,
checkpoint loading, module import, CSV writing, and plot generation are
never included inside a timed stage or the total.
"""

from __future__ import annotations

import argparse
import csv
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
from src.sensing.energy_detection import energy_detect, filter_by_min_length, mask_to_regions, merge_close_regions  # noqa: E402
from src.sensing.normalize import apply_awn_preprocess, to_awn_input  # noqa: E402
from src.sensing.radioml_source import RML2016_10A_CLASSES, embed_sample_in_noise  # noqa: E402
from src.sensing.segmentation import select_aligned_segments  # noqa: E402

DATASET_PATH = "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl"
CHECKPOINT_PATH = "external/adversarial-rf/2016.10a_AWN.pkl"
DEVICE = "cpu"

MODULATIONS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
ALL_SNRS = list(range(-20, 20, 2))  # 20 values, matches RML2016_10A_VALID_SNRS
ATTACK_SNRS = [-10, 0, 18]

N_SAMPLES = 8192
EMBED_SNR_MARGIN = 20.0
THRESHOLD_FACTOR = 5.0
SENSING_WINDOW_SIZE = 128
MIN_REGION_LEN = 128
MERGE_GAP = 0
ALIGNMENT_POLICY = "max-energy"
AWN_PREPROCESS = "radioml-native"
SEED = 0

NS_PER_MS = 1_000_000.0


def now_ns() -> int:
    return time.perf_counter_ns()


class Backends:
    def __init__(self, need_attack: bool = False, need_topk: bool = False) -> None:
        print(f"[backends] loading AWN checkpoint {CHECKPOINT_PATH} ...", flush=True)
        self.awn = AWNModelAdapter(checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
        if self.awn.backend_name != _REAL_MODEL_SOURCE or self.awn.status != "ok":
            raise RuntimeError(f"Real-AWN precheck FAILED: backend={self.awn.backend_name} status={self.awn.status}")
        print(f"[precheck] real AWN backend confirmed: {_REAL_MODEL_SOURCE}", flush=True)

        self.attack = None
        if need_attack:
            self.attack = AttackAdapter(awn_model=self.awn.model, device=DEVICE)
            if self.attack.wrapped_model is None or self.attack.backend_name != _REAL_ATTACK_SOURCE:
                raise RuntimeError(f"Real-attack precheck FAILED: backend={self.attack.backend_name}")
            print(f"[precheck] real attack backend confirmed: {_REAL_ATTACK_SOURCE}", flush=True)

        self.topk = None
        if need_topk:
            self.topk = TopKAdapter()
            if not self.topk.backend_available or self.topk.backend_name != _REAL_TOPK_SOURCE:
                raise RuntimeError(f"Real Top-K precheck FAILED: backend={self.topk.backend_name}")
            print(f"[precheck] real Top-K backend confirmed: {_REAL_TOPK_SOURCE}", flush=True)

        print(f"[dataset] loading RadioML dict {DATASET_PATH} (one-time, ~640MB) ...", flush=True)
        t0 = time.perf_counter()
        import pickle
        with open(DATASET_PATH, "rb") as f:
            self.radioml_dict = pickle.load(f, encoding="latin1")
        print(f"[dataset] loaded in {time.perf_counter() - t0:.1f}s, {len(self.radioml_dict)} (mod,snr) cells", flush=True)


class BaseSampleError(RuntimeError):
    pass


def clean_pipeline_once(backends: Backends, mod: str, snr: int, sample_index: int,
                          topk: Optional[int]) -> dict:
    """Times every clean-path stage for one (mod, snr, sample_index). Never
    included in the timed region: dict lookup of the RadioML block itself
    (that is dataset access, not pipeline processing)."""
    row: Dict[str, object] = {
        "modulation": mod, "snr": snr, "sample_index": sample_index,
        "status": "ok", "error_type": None, "error_message": None, "fallback_used": False,
    }
    try:
        block = backends.radioml_dict[(mod, snr)]
        if sample_index >= block.shape[0]:
            raise BaseSampleError(f"sample_index {sample_index} out of range for ({mod},{snr})")
        sample_2x128 = block[sample_index].astype(np.float32)

        t0 = now_ns()
        iq, embed_meta = embed_sample_in_noise(sample_2x128, N_SAMPLES, EMBED_SNR_MARGIN, seed=SEED + sample_index)
        row["embedding_ms"] = (now_ns() - t0) / NS_PER_MS

        t0 = now_ns()
        mask = energy_detect(iq, window=SENSING_WINDOW_SIZE, threshold_factor=THRESHOLD_FACTOR)
        row["energy_detection_ms"] = (now_ns() - t0) / NS_PER_MS
        # noise-floor estimation and threshold comparison happen INSIDE
        # energy_detect() (median of the smoothed power array, then a single
        # vectorized comparison) with no separable call boundary in the
        # current implementation -- splitting it would require modifying
        # src/sensing/energy_detection.py, which this round does not do.
        row["noise_floor_threshold_ms"] = None
        row["noise_floor_threshold_na_reason"] = "fused inside energy_detect(); no separable call boundary without modifying src/sensing/energy_detection.py"

        t0 = now_ns()
        raw_regions = mask_to_regions(mask)
        merged_regions = merge_close_regions(raw_regions, merge_gap=MERGE_GAP)
        try:
            kept_regions = filter_by_min_length(merged_regions, min_len=MIN_REGION_LEN)
        except RuntimeError:
            kept_regions = []
        row["region_postprocess_ms"] = (now_ns() - t0) / NS_PER_MS

        if not kept_regions:
            raise BaseSampleError(f"no occupied region for ({mod},{snr},{sample_index})")

        t0 = now_ns()
        segments, align_meta = select_aligned_segments(iq, kept_regions, seg_len=128, policy=ALIGNMENT_POLICY, hop=1)
        row["segmentation_ms"] = (now_ns() - t0) / NS_PER_MS
        # max-energy window search is the inner loop of select_aligned_segments()
        # itself (same function call as segmentation) -- no separable boundary
        # without modifying src/sensing/segmentation.py.
        row["max_energy_selection_ms"] = None
        row["max_energy_selection_na_reason"] = "fused inside select_aligned_segments() max-energy policy loop; no separable call boundary without modifying src/sensing/segmentation.py"

        if segments.shape[0] == 0:
            raise BaseSampleError(f"no segment produced for ({mod},{snr},{sample_index})")

        t0 = now_ns()
        x_clean = apply_awn_preprocess(segments[:1], policy=AWN_PREPROCESS)
        x_clean = to_awn_input(x_clean, seg_len=128)
        row["awn_preprocess_ms"] = (now_ns() - t0) / NS_PER_MS

        t0 = now_ns()
        logits_clean, meta_clean = backends.awn.infer(x_clean, seed=SEED)
        row["awn_clean_inference_ms"] = (now_ns() - t0) / NS_PER_MS
        row["awn_backend"] = meta_clean["awn_backend"]
        if meta_clean["awn_backend"] != _REAL_MODEL_SOURCE:
            row["fallback_used"] = True
        pred_clean = int(np.argmax(logits_clean[0]))
        row["clean_prediction"] = pred_clean
        row["clean_correct"] = (pred_clean == RML2016_10A_CLASSES[mod])

        row["clean_total_ms"] = (
            row["embedding_ms"] + row["energy_detection_ms"] + row["region_postprocess_ms"]
            + row["segmentation_ms"] + row["awn_preprocess_ms"] + row["awn_clean_inference_ms"]
        )

        row["topk_ms"] = None
        row["defended_inference_ms"] = None
        row["topk_total_ms"] = None
        if topk is not None:
            t0 = now_ns()
            x_defended, topk_meta = backends.topk.apply(x_clean, topk=topk)
            row["topk_ms"] = (now_ns() - t0) / NS_PER_MS
            if topk_meta["topk_backend"] != _REAL_TOPK_SOURCE:
                row["fallback_used"] = True

            t0 = now_ns()
            logits_defended, meta_defended = backends.awn.infer(x_defended, seed=SEED)
            row["defended_inference_ms"] = (now_ns() - t0) / NS_PER_MS
            if meta_defended["awn_backend"] != _REAL_MODEL_SOURCE:
                row["fallback_used"] = True
            pred_defended = int(np.argmax(logits_defended[0]))
            row["defended_prediction"] = pred_defended
            row["topk_total_ms"] = row["clean_total_ms"] + row["topk_ms"] + row["defended_inference_ms"]

        if not np.isfinite(logits_clean).all():
            row["status"] = "nan_inf"

    except Exception as exc:  # noqa: BLE001 -- fail closed: record, never silently continue with fabricated timing
        row["status"] = "error"
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc)
    return row


CLEAN_FIELDS = [
    "modulation", "snr", "sample_index",
    "embedding_ms", "energy_detection_ms", "noise_floor_threshold_ms", "noise_floor_threshold_na_reason",
    "region_postprocess_ms", "segmentation_ms", "max_energy_selection_ms", "max_energy_selection_na_reason",
    "awn_preprocess_ms", "awn_clean_inference_ms", "clean_total_ms",
    "topk_ms", "defended_inference_ms", "topk_total_ms",
    "clean_prediction", "clean_correct", "defended_prediction",
    "status", "error_type", "error_message", "fallback_used", "awn_backend",
]


def percentiles(vals: List[float]) -> dict:
    if not vals:
        return {k: None for k in ["n", "mean", "std", "min", "median", "p90", "p95", "p99", "max", "samples_per_sec"]}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": len(arr), "mean": float(arr.mean()), "std": float(arr.std()),
        "min": float(arr.min()), "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)), "max": float(arr.max()),
        "samples_per_sec": float(1000.0 / arr.mean()) if arr.mean() > 0 else None,
    }


def run_clean_sensing(backends: Backends, out_dir: Path, topk: Optional[int], warmup_n: int) -> None:
    combos = [(m, s, i) for m in MODULATIONS for s in ALL_SNRS for i in range(10)]
    print(f"[clean_sensing] {len(combos)} combos (expect 2200)", flush=True)

    print(f"[clean_sensing] warm-up: {warmup_n} samples (excluded from stats)", flush=True)
    for mod, snr, idx in combos[:warmup_n]:
        clean_pipeline_once(backends, mod, snr, idx, topk)

    rows = []
    n_error = n_fallback = n_nan = 0
    t_start = time.perf_counter()
    for i, (mod, snr, idx) in enumerate(combos):
        row = clean_pipeline_once(backends, mod, snr, idx, topk)
        rows.append(row)
        if row["status"] == "error":
            n_error += 1
        if row["status"] == "nan_inf":
            n_nan += 1
        if row["fallback_used"]:
            n_fallback += 1
        if (i + 1) % 200 == 0 or (i + 1) == len(combos):
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed
            eta = (len(combos) - i - 1) / rate if rate > 0 else float("inf")
            print(f"[clean_sensing] {i+1}/{len(combos)} error={n_error} fallback={n_fallback} nan={n_nan} "
                  f"elapsed={elapsed:.1f}s rate={rate:.1f}/s ETA={eta:.1f}s", flush=True)

    with open(out_dir / "pipeline_latency_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CLEAN_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    stage_cols = ["embedding_ms", "energy_detection_ms", "region_postprocess_ms", "segmentation_ms",
                  "awn_preprocess_ms", "awn_clean_inference_ms", "clean_total_ms"]
    if topk is not None:
        stage_cols += ["topk_ms", "defended_inference_ms", "topk_total_ms"]

    total_mean = np.mean([r["clean_total_ms"] for r in ok_rows]) if ok_rows else 1.0
    summary_rows = []
    for col in stage_cols:
        vals = [r[col] for r in ok_rows if r.get(col) is not None]
        stats = percentiles(vals)
        stats["stage"] = col
        stats["pct_of_clean_total"] = (stats["mean"] / total_mean * 100.0) if stats["mean"] is not None and "topk" not in col else (
            stats["mean"] / total_mean * 100.0 if stats["mean"] is not None else None
        )
        summary_rows.append(stats)

    with open(out_dir / "pipeline_latency_summary.csv", "w", newline="") as f:
        fieldnames = ["stage", "n", "mean", "std", "min", "median", "p90", "p95", "p99", "max",
                      "samples_per_sec", "pct_of_clean_total"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(f"[clean_sensing] DONE: {len(rows)} rows, error={n_error} fallback={n_fallback} nan={n_nan}", flush=True)


def attack_instance_once(backends: Backends, mod: str, snr: int, sample_index: int, attack_name: str,
                           eps: Optional[float], attack_params: dict) -> dict:
    row: Dict[str, object] = {
        "attack_name": attack_name, "modulation": mod, "snr": snr, "sample_index": sample_index,
        "status": "ok", "error_type": None, "error_message": None, "fallback_used": False,
    }
    try:
        block = backends.radioml_dict[(mod, snr)]
        sample_2x128 = block[sample_index].astype(np.float32)
        iq, embed_meta = embed_sample_in_noise(sample_2x128, N_SAMPLES, EMBED_SNR_MARGIN, seed=SEED + sample_index)
        mask = energy_detect(iq, window=SENSING_WINDOW_SIZE, threshold_factor=THRESHOLD_FACTOR)
        raw_regions = mask_to_regions(mask)
        merged_regions = merge_close_regions(raw_regions, merge_gap=MERGE_GAP)
        kept_regions = filter_by_min_length(merged_regions, min_len=MIN_REGION_LEN)
        segments, _ = select_aligned_segments(iq, kept_regions, seg_len=128, policy=ALIGNMENT_POLICY, hop=1)
        x_clean = apply_awn_preprocess(segments[:1], policy=AWN_PREPROCESS)
        x_clean = to_awn_input(x_clean, seg_len=128)
        logits_clean, meta_clean = backends.awn.infer(x_clean, seed=SEED)
        pred_clean = int(np.argmax(logits_clean[0]))
        row["clean_prediction"] = pred_clean
        row["clean_correct"] = (pred_clean == RML2016_10A_CLASSES[mod])

        # attack_object_init_ms: torchattacks object construction is fused
        # inside AttackAdapter.apply() (_build_torchattacks is called every
        # invocation, not separately callable from outside apply()) -- so
        # this cannot be timed as an isolated stage without modifying
        # src/adapters/attack_adapter.py. Recorded as NA with reason; the
        # combined cost is included in attack_generation_ms below.
        row["attack_object_init_ms"] = None
        row["attack_object_init_na_reason"] = "torchattacks object construction is fused inside AttackAdapter.apply(); not separately callable without modifying src/adapters/attack_adapter.py"
        row["attack_input_prepare_ms"] = 0.0  # x_clean already built above by the shared clean-path timing

        eps_for_apply = eps if eps is not None else 0.05
        t0 = now_ns()
        x_adv, attack_meta = backends.attack.apply(x_clean, attack=attack_name, eps=eps_for_apply, seed=SEED, attack_params=attack_params)
        row["attack_generation_ms"] = (now_ns() - t0) / NS_PER_MS
        if attack_meta["attack_backend"] != _REAL_ATTACK_SOURCE or attack_meta["attack_status"] != "ok":
            row["fallback_used"] = True

        t0 = now_ns()
        logits_att, meta_att = backends.awn.infer(x_adv, seed=SEED)
        row["awn_attacked_inference_ms"] = (now_ns() - t0) / NS_PER_MS
        if meta_att["awn_backend"] != _REAL_MODEL_SOURCE:
            row["fallback_used"] = True
        pred_att = int(np.argmax(logits_att[0]))
        row["attacked_prediction"] = pred_att
        row["attack_success"] = (pred_att != pred_clean)

        perturb = x_adv.astype(np.float64) - x_clean.astype(np.float64)
        row["perturbation_linf"] = float(np.max(np.abs(perturb)))
        row["perturbation_l2"] = float(np.linalg.norm(perturb))

        row["attack_total_ms"] = row["attack_generation_ms"] + row["awn_attacked_inference_ms"]

        model_mode_after = "train" if backends.attack.wrapped_model.training else "eval"
        row["model_mode_after"] = model_mode_after

        if not np.isfinite(x_adv).all() or not np.isfinite(logits_att).all():
            row["status"] = "nan_inf"

    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc)
    return row


ATTACK_FIELDS = [
    "attack_name", "modulation", "snr", "sample_index",
    "attack_object_init_ms", "attack_object_init_na_reason", "attack_input_prepare_ms",
    "attack_generation_ms", "awn_attacked_inference_ms", "attack_total_ms",
    "clean_prediction", "clean_correct", "attacked_prediction", "attack_success",
    "perturbation_linf", "perturbation_l2", "model_mode_after",
    "status", "error_type", "error_message", "fallback_used",
]


ATTACK_EPS_DEFAULT = {"fgsm": 0.05, "pgd": 0.05, "cw": None}
ATTACK_PARAMS_DEFAULT = {"fgsm": {"eps": 0.05}, "pgd": {"eps": 0.05}, "cw": {}}


def run_attack_baseline(backends: Backends, out_dir: Path, attacks: List[str], warmup_n: int) -> None:
    combos = [(m, s, i) for m in MODULATIONS for s in ATTACK_SNRS for i in range(10)]
    print(f"[attack_baseline] {len(combos)} combos per attack (expect 330)", flush=True)

    for attack_name in attacks:
        eps = ATTACK_EPS_DEFAULT[attack_name]
        params = ATTACK_PARAMS_DEFAULT[attack_name]
        print(f"[attack_baseline] warm-up {attack_name}: {warmup_n} samples", flush=True)
        for mod, snr, idx in combos[:warmup_n]:
            attack_instance_once(backends, mod, snr, idx, attack_name, eps, params)

        rows = []
        n_error = n_fallback = n_nan = 0
        t_start = time.perf_counter()
        for i, (mod, snr, idx) in enumerate(combos):
            row = attack_instance_once(backends, mod, snr, idx, attack_name, eps, params)
            rows.append(row)
            if row["status"] == "error":
                n_error += 1
            if row["status"] == "nan_inf":
                n_nan += 1
            if row["fallback_used"]:
                n_fallback += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(combos):
                elapsed = time.perf_counter() - t_start
                rate = (i + 1) / elapsed
                eta = (len(combos) - i - 1) / rate if rate > 0 else float("inf")
                print(f"[attack_baseline][{attack_name}] {i+1}/{len(combos)} error={n_error} fallback={n_fallback} "
                      f"nan={n_nan} elapsed={elapsed:.1f}s rate={rate:.2f}/s ETA={eta:.1f}s", flush=True)

        with open(out_dir / f"{attack_name}_baseline_raw.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ATTACK_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[attack_baseline][{attack_name}] DONE: {len(rows)} rows, error={n_error} fallback={n_fallback} nan={n_nan}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["clean_sensing", "attack_baseline"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--topk", type=int, default=None, help="clean_sensing only: also run Top-K + defended inference at this K")
    ap.add_argument("--attacks", type=str, default="fgsm,pgd,cw", help="attack_baseline only: comma-separated subset")
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "clean_sensing":
        backends = Backends(need_attack=False, need_topk=(args.topk is not None))
        run_clean_sensing(backends, out_dir, args.topk, args.warmup)
    else:
        attacks = args.attacks.split(",")
        backends = Backends(need_attack=True, need_topk=False)
        run_attack_baseline(backends, out_dir, attacks, args.warmup)


if __name__ == "__main__":
    main()
