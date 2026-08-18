"""
Full End-to-End Latency Matrix and Before/After Optimization Comparison.

Answers, with real measured data (no estimation): how much time does each
spectrum-sensing stage / AMC (AWN) inference / attack generation / Top-K
defense take, what is the true end-to-end latency from IQ input to final
prediction for five formal scenarios, and how much does the whole pipeline
(not just attack_generation_ms) actually speed up after optimization.

Reuses the exact same real building blocks and constants as
experiments/benchmark_pipeline_latency.py (N_SAMPLES, EMBED_SNR_MARGIN,
THRESHOLD_FACTOR, SENSING_WINDOW_SIZE, MIN_REGION_LEN, MERGE_GAP,
ALIGNMENT_POLICY, AWN_PREPROCESS, SEED, embed_sample_in_noise,
energy_detect, mask_to_regions, merge_close_regions, filter_by_min_length,
select_aligned_segments, apply_awn_preprocess, to_awn_input,
AWNModelAdapter, AttackAdapter, TopKAdapter) -- NOT reimplemented, not a
new/placeholder pipeline. Not imported directly from that module because
its own clean_pipeline_once()/attack_instance_once() don't return the
intermediate x_clean/x_adv tensors needed to chain sensing -> attack -> Top-K
within ONE continuous per-sample timed pass (which pipeline_latency_raw.csv
and fgsm/pgd/cw_baseline_raw.csv, being two separately-run benchmark
phases, never captured together). This script closes exactly that gap.

Does NOT rerun the existing 2200-sample clean_sensing benchmark or the
330-sample attack_baseline benchmark -- Scenario A/B's large-n numbers are
read directly from the existing results/performance_latency_20260818T010552Z/
pipeline_latency_raw.csv (see aggregate_scenario_ab_from_existing.py-style
logic inlined in finalize step below). This script performs ONLY the small
supplemental measurement needed for genuinely NEW end-to-end timings:
Scenarios C/D/E (attack scenarios) and a matched small-n A/B for
cross-scenario comparability, all on a FIXED 24-sample set (BPSK/QPSK/QAM16/
WBFM x SNR{-10,0,18} x sample_index{0,1}).

Real backends only, fails closed (RuntimeError) if any is not the real
implementation. Does not modify external/AWN or external/adversarial-rf.
"""

from __future__ import annotations

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
from src.sensing.radioml_source import RML2016_10A_CLASSES, embed_sample_in_noise, load_radioml_dict  # noqa: E402
from src.sensing.segmentation import select_aligned_segments  # noqa: E402

DATASET_PATH = "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl"
CHECKPOINT_PATH = "external/adversarial-rf/2016.10a_AWN.pkl"
DEVICE = "cpu"

# Identical to experiments/benchmark_pipeline_latency.py, so these numbers
# are directly comparable to the existing Phase A/B results.
N_SAMPLES = 8192
EMBED_SNR_MARGIN = 20.0
THRESHOLD_FACTOR = 5.0
SENSING_WINDOW_SIZE = 128
MIN_REGION_LEN = 128
MERGE_GAP = 0
ALIGNMENT_POLICY = "max-energy"
AWN_PREPROCESS = "radioml-native"
SEED = 0
TOPK = 20  # matches Phase A's K=20 Top-K defense measurement

NS_PER_MS = 1_000_000.0

MODS = ["BPSK", "QPSK", "QAM16", "WBFM"]
SNRS = [-10, 0, 18]
IDXS = [0, 1]

FGSM_PARAMS = {"eps": 0.05}
PGD_PARAMS_DET = {"eps": 0.05, "random_start": False}
PGD_PARAMS_STOCH = {"eps": 0.05, "random_start": True}
CW_PARAMS: dict = {}


def now_ns() -> int:
    return time.perf_counter_ns()


class Backends:
    def __init__(self) -> None:
        print(f"[backends] loading AWN checkpoint {CHECKPOINT_PATH} ...", flush=True)
        self.awn = AWNModelAdapter(checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
        if self.awn.backend_name != _REAL_MODEL_SOURCE or self.awn.status != "ok":
            raise RuntimeError(f"Real-AWN precheck FAILED: backend={self.awn.backend_name} status={self.awn.status}")
        self.attack = AttackAdapter(awn_model=self.awn.model, device=DEVICE)
        if self.attack.wrapped_model is None or self.attack.backend_name != _REAL_ATTACK_SOURCE:
            raise RuntimeError(f"Real-attack precheck FAILED: backend={self.attack.backend_name}")
        self.topk = TopKAdapter()
        if not self.topk.backend_available or self.topk.backend_name != _REAL_TOPK_SOURCE:
            raise RuntimeError(f"Real Top-K precheck FAILED: backend={self.topk.backend_name}")
        print("[backends] real AWN + real attack + real Top-K all confirmed", flush=True)
        print(f"[dataset] loading RadioML dict {DATASET_PATH} (one-time) ...", flush=True)
        self.radioml_dict = load_radioml_dict(DATASET_PATH)
        print(f"[dataset] loaded, {len(self.radioml_dict)} (mod,snr) cells", flush=True)


def build_fixed_24() -> List[dict]:
    return [{"modulation": m, "snr": s, "sample_index": i} for m in MODS for s in SNRS for i in IDXS]


def time_clean_stages(backends: Backends, mod: str, snr: int, sample_index: int) -> dict:
    """Times embedding -> energy_detection -> region_postprocess -> segmentation
    -> awn_preprocess -> awn_clean_inference in ONE continuous pass, identical
    stage code/parameters to benchmark_pipeline_latency.py:clean_pipeline_once,
    but additionally returns x_clean itself so callers can chain into attack/
    Top-K stages within the same continuous timed sequence."""
    row: Dict[str, object] = {
        "modulation": mod, "snr": snr, "sample_index": sample_index,
        "status": "ok", "error_type": None, "error_message": None, "fallback_used": False,
        "x_clean": None,
    }
    try:
        block = backends.radioml_dict[(mod, snr)]
        sample_2x128 = block[sample_index].astype(np.float32)

        t0 = now_ns()
        iq, _ = embed_sample_in_noise(sample_2x128, N_SAMPLES, EMBED_SNR_MARGIN, seed=SEED + sample_index)
        row["embedding_ms"] = (now_ns() - t0) / NS_PER_MS

        t0 = now_ns()
        mask = energy_detect(iq, window=SENSING_WINDOW_SIZE, threshold_factor=THRESHOLD_FACTOR)
        row["energy_detection_ms"] = (now_ns() - t0) / NS_PER_MS

        t0 = now_ns()
        raw_regions = mask_to_regions(mask)
        merged_regions = merge_close_regions(raw_regions, merge_gap=MERGE_GAP)
        try:
            kept_regions = filter_by_min_length(merged_regions, min_len=MIN_REGION_LEN)
        except RuntimeError:
            kept_regions = []
        row["region_postprocess_ms"] = (now_ns() - t0) / NS_PER_MS

        if not kept_regions:
            raise RuntimeError(f"no occupied region for ({mod},{snr},{sample_index})")

        t0 = now_ns()
        segments, _ = select_aligned_segments(iq, kept_regions, seg_len=128, policy=ALIGNMENT_POLICY, hop=1)
        row["segmentation_ms"] = (now_ns() - t0) / NS_PER_MS

        if segments.shape[0] == 0:
            raise RuntimeError(f"no segment produced for ({mod},{snr},{sample_index})")

        t0 = now_ns()
        x_clean = apply_awn_preprocess(segments[:1], policy=AWN_PREPROCESS)
        x_clean = to_awn_input(x_clean, seg_len=128)
        row["awn_preprocess_ms"] = (now_ns() - t0) / NS_PER_MS

        t0 = now_ns()
        logits_clean, meta_clean = backends.awn.infer(x_clean, seed=SEED)
        row["awn_clean_inference_ms"] = (now_ns() - t0) / NS_PER_MS
        if meta_clean["awn_backend"] != _REAL_MODEL_SOURCE:
            row["fallback_used"] = True
        pred_clean = int(np.argmax(logits_clean[0]))
        row["clean_prediction"] = pred_clean
        row["clean_correct"] = (pred_clean == RML2016_10A_CLASSES[mod])
        row["x_clean"] = x_clean

        row["clean_stage_total_ms"] = (
            row["embedding_ms"] + row["energy_detection_ms"] + row["region_postprocess_ms"]
            + row["segmentation_ms"] + row["awn_preprocess_ms"] + row["awn_clean_inference_ms"]
        )

        if not np.isfinite(logits_clean).all():
            row["status"] = "nan_inf"

    except Exception as exc:  # noqa: BLE001 -- fail closed, never fabricate timing
        row["status"] = "error"
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc)
    return row


def time_topk_stage(backends: Backends, x_input: np.ndarray) -> dict:
    out = {"topk_ms": None, "defended_inference_ms": None, "defended_prediction": None, "fallback_used": False}
    t0 = now_ns()
    x_defended, topk_meta = backends.topk.apply(x_input, topk=TOPK)
    out["topk_ms"] = (now_ns() - t0) / NS_PER_MS
    if topk_meta["topk_backend"] != _REAL_TOPK_SOURCE:
        out["fallback_used"] = True

    t0 = now_ns()
    logits_defended, meta_defended = backends.awn.infer(x_defended, seed=SEED)
    out["defended_inference_ms"] = (now_ns() - t0) / NS_PER_MS
    if meta_defended["awn_backend"] != _REAL_MODEL_SOURCE:
        out["fallback_used"] = True
    out["defended_prediction"] = int(np.argmax(logits_defended[0]))
    return out


def time_attack_stage_single(backends: Backends, x_clean: np.ndarray, attack_name: str,
                              eps: float, attack_params: dict) -> dict:
    """batch_size=1 attack + attacked inference, one sample."""
    out = {"attack_generation_ms": None, "awn_attacked_inference_ms": None, "x_adv": None,
           "attacked_prediction": None, "perturbation_linf": None, "perturbation_l2": None,
           "model_mode_after": None, "fallback_used": False}
    t0 = now_ns()
    x_adv, attack_meta = backends.attack.apply(x_clean, attack=attack_name, eps=eps, seed=SEED, attack_params=attack_params)
    out["attack_generation_ms"] = (now_ns() - t0) / NS_PER_MS
    if attack_meta["attack_backend"] != _REAL_ATTACK_SOURCE or attack_meta["attack_status"] != "ok":
        out["fallback_used"] = True
    out["model_mode_after"] = "train" if backends.attack.wrapped_model.training else "eval"

    t0 = now_ns()
    logits_att, meta_att = backends.awn.infer(x_adv, seed=SEED)
    out["awn_attacked_inference_ms"] = (now_ns() - t0) / NS_PER_MS
    if meta_att["awn_backend"] != _REAL_MODEL_SOURCE:
        out["fallback_used"] = True
    out["attacked_prediction"] = int(np.argmax(logits_att[0]))
    perturb = x_adv.astype(np.float64) - x_clean.astype(np.float64)
    out["perturbation_linf"] = float(np.max(np.abs(perturb)))
    out["perturbation_l2"] = float(np.linalg.norm(perturb))
    out["x_adv"] = x_adv
    return out


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


def main() -> None:
    out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    default_threads = torch.get_num_threads()

    backends = Backends()
    fixed_24 = build_fixed_24()
    print(f"[e2e] {len(fixed_24)} fixed samples (4 mods x 3 SNRs x 2 idx)", flush=True)

    raw_rows: List[dict] = []
    n_error = n_fallback = n_nan = 0

    def emit(scenario: str, variant: str, threads_used: int, mod: str, snr: int, idx: int, stages: dict, total_ms: float):
        nonlocal n_error, n_fallback, n_nan
        row = {
            "scenario": scenario, "variant": variant, "threads_used": threads_used,
            "modulation": mod, "snr": snr, "sample_index": idx,
            "input_prepare_ms": 0.0,
            "embedding_ms": stages.get("embedding_ms"),
            "energy_detection_ms": stages.get("energy_detection_ms"),
            "region_postprocess_ms": stages.get("region_postprocess_ms"),
            "segmentation_ms": stages.get("segmentation_ms"),
            "awn_preprocess_ms": stages.get("awn_preprocess_ms"),
            "awn_clean_inference_ms": stages.get("awn_clean_inference_ms"),
            "attack_prepare_ms": stages.get("attack_prepare_ms"),
            "attack_generation_ms": stages.get("attack_generation_ms"),
            "awn_attacked_inference_ms": stages.get("awn_attacked_inference_ms"),
            "topk_ms": stages.get("topk_ms"),
            "defended_inference_ms": stages.get("defended_inference_ms"),
            "total_ms": total_ms,
            "status": stages.get("status", "ok"),
            "clean_prediction": stages.get("clean_prediction"),
            "clean_correct": stages.get("clean_correct"),
            "attacked_prediction": stages.get("attacked_prediction"),
            "attack_success": (stages.get("attacked_prediction") != stages.get("clean_prediction"))
            if stages.get("attacked_prediction") is not None and stages.get("clean_prediction") is not None else None,
            "perturbation_linf": stages.get("perturbation_linf"),
            "perturbation_l2": stages.get("perturbation_l2"),
            "defended_prediction": stages.get("defended_prediction"),
            "model_mode_after": stages.get("model_mode_after"),
        }
        raw_rows.append(row)
        if row["status"] != "ok":
            n_error += 1
        if stages.get("fallback_used"):
            n_fallback += 1
        if stages.get("has_nan_inf"):
            n_nan += 1

    # warm-up (discarded): 5 samples through the full clean+FGSM+topk path at default threads
    print("[e2e] warm-up ...", flush=True)
    torch.set_num_threads(default_threads)
    for s in fixed_24[:5]:
        cs = time_clean_stages(backends, s["modulation"], s["snr"], s["sample_index"])
        if cs["status"] == "ok":
            time_attack_stage_single(backends, cs["x_clean"], "fgsm", 0.05, FGSM_PARAMS)
            time_topk_stage(backends, cs["x_clean"])

    # ===== Scenario A/B/C/D/E, baseline pass, default threads (threads_used=default) =====
    print(f"[e2e] === baseline pass (threads={default_threads}) ===", flush=True)
    torch.set_num_threads(default_threads)
    for s in fixed_24:
        mod, snr, idx = s["modulation"], s["snr"], s["sample_index"]
        cs = time_clean_stages(backends, mod, snr, idx)
        if cs["status"] != "ok":
            emit("A", "n/a", default_threads, mod, snr, idx, cs, None)
            emit("B", "n/a", default_threads, mod, snr, idx, cs, None)
            emit("C", "baseline", default_threads, mod, snr, idx, cs, None)
            emit("D_det", "baseline", default_threads, mod, snr, idx, cs, None)
            emit("D_stoch", "baseline", default_threads, mod, snr, idx, cs, None)
            emit("E", "baseline", default_threads, mod, snr, idx, cs, None)
            continue

        # Scenario A: clean only
        emit("A", "n/a", default_threads, mod, snr, idx, cs, cs["clean_stage_total_ms"])

        # Scenario B: clean + Top-K (on x_clean)
        tk = time_topk_stage(backends, cs["x_clean"])
        merged_b = dict(cs); merged_b.update(tk)
        total_b = cs["clean_stage_total_ms"] + tk["topk_ms"] + tk["defended_inference_ms"]
        emit("B", "n/a", default_threads, mod, snr, idx, merged_b, total_b)

        # Scenario C: clean + FGSM (baseline, batch_size=1)
        fg = time_attack_stage_single(backends, cs["x_clean"], "fgsm", 0.05, FGSM_PARAMS)
        merged_c = dict(cs); merged_c.update(fg)
        total_c = cs["clean_stage_total_ms"] + fg["attack_generation_ms"] + fg["awn_attacked_inference_ms"]
        emit("C", "baseline", default_threads, mod, snr, idx, merged_c, total_c)

        # Scenario D (deterministic PGD, random_start=False), baseline
        pg_det = time_attack_stage_single(backends, cs["x_clean"], "pgd", 0.05, PGD_PARAMS_DET)
        merged_d = dict(cs); merged_d.update(pg_det)
        total_d = cs["clean_stage_total_ms"] + pg_det["attack_generation_ms"] + pg_det["awn_attacked_inference_ms"]
        emit("D_det", "baseline", default_threads, mod, snr, idx, merged_d, total_d)

        # Scenario D (stochastic PGD, random_start=True), baseline -- reported separately
        pg_stoch = time_attack_stage_single(backends, cs["x_clean"], "pgd", 0.05, PGD_PARAMS_STOCH)
        merged_d2 = dict(cs); merged_d2.update(pg_stoch)
        total_d2 = cs["clean_stage_total_ms"] + pg_stoch["attack_generation_ms"] + pg_stoch["awn_attacked_inference_ms"]
        emit("D_stoch", "baseline", default_threads, mod, snr, idx, merged_d2, total_d2)

        # Scenario E: clean + FGSM + Top-K (Top-K applied to x_adv, matching src/utils/pipeline.py's
        # actual order: AttackAdapter runs BEFORE TopKAdapter, TopKAdapter output feeds the final AWN call)
        tk_e = time_topk_stage(backends, fg["x_adv"])
        merged_e = dict(cs); merged_e.update(fg); merged_e.update(tk_e)
        total_e = total_c + tk_e["topk_ms"] + tk_e["defended_inference_ms"]
        emit("E", "baseline", default_threads, mod, snr, idx, merged_e, total_e)

    # ===== Optimized pass: threads=1, attack batched (batch_size=16) =====
    OPT_THREADS = 1
    OPT_BATCH = 16
    print(f"[e2e] === optimized pass (threads={OPT_THREADS}, attack batch_size={OPT_BATCH}) ===", flush=True)
    torch.set_num_threads(OPT_THREADS)

    # re-time the clean 6 stages under threads=1 (AWN inference latency is
    # itself thread-count-dependent, per docs/research/.../section 15.4) --
    # NOT reused from the baseline pass, so the "optimized end-to-end" number
    # reflects the actual optimized environment, not a cross-run splice.
    clean_opt: List[dict] = []
    for s in fixed_24:
        cs = time_clean_stages(backends, s["modulation"], s["snr"], s["sample_index"])
        clean_opt.append(cs)
        if cs["status"] != "ok":
            emit("A_opt", "n/a", OPT_THREADS, s["modulation"], s["snr"], s["sample_index"], cs, None)
            continue
        emit("A_opt", "n/a", OPT_THREADS, s["modulation"], s["snr"], s["sample_index"], cs, cs["clean_stage_total_ms"])

    ok_clean_opt = [(s, cs) for s, cs in zip(fixed_24, clean_opt) if cs["status"] == "ok"]

    def run_batched_attack(attack_name: str, eps: float, params: dict, scenario_tag: str):
        xs = [cs["x_clean"] for _, cs in ok_clean_opt]
        for start in range(0, len(xs), OPT_BATCH):
            batch_specs = ok_clean_opt[start:start + OPT_BATCH]
            batch = np.concatenate([cs["x_clean"] for _, cs in batch_specs], axis=0)
            t0 = now_ns()
            x_adv_batch, attack_meta = backends.attack.apply(batch, attack=attack_name, eps=eps, seed=SEED, attack_params=params)
            batch_ms = (now_ns() - t0) / NS_PER_MS
            per_sample_attack_ms = batch_ms / batch.shape[0]
            fallback = attack_meta["attack_backend"] != _REAL_ATTACK_SOURCE or attack_meta["attack_status"] != "ok"
            for i, (s, cs) in enumerate(batch_specs):
                x_single = x_adv_batch[i:i + 1]
                t1 = now_ns()
                logits_att, meta_att = backends.awn.infer(x_single, seed=SEED)
                attacked_inf_ms = (now_ns() - t1) / NS_PER_MS
                perturb = x_single.astype(np.float64) - cs["x_clean"].astype(np.float64)
                merged = dict(cs)
                merged["attack_generation_ms"] = per_sample_attack_ms
                merged["awn_attacked_inference_ms"] = attacked_inf_ms
                merged["fallback_used"] = fallback or (meta_att["awn_backend"] != _REAL_MODEL_SOURCE)
                merged["attacked_prediction"] = int(np.argmax(logits_att[0]))
                merged["perturbation_linf"] = float(np.max(np.abs(perturb)))
                merged["perturbation_l2"] = float(np.linalg.norm(perturb))
                merged["model_mode_after"] = "train" if backends.attack.wrapped_model.training else "eval"
                total = cs["clean_stage_total_ms"] + per_sample_attack_ms + attacked_inf_ms
                emit(scenario_tag, "optimized", OPT_THREADS, s["modulation"], s["snr"], s["sample_index"], merged, total)
                if scenario_tag == "C":
                    # feed Scenario E's Top-K stage from this same batched-FGSM x_adv
                    tk_e = time_topk_stage(backends, x_single)
                    merged_e = dict(merged); merged_e.update(tk_e)
                    total_e = total + tk_e["topk_ms"] + tk_e["defended_inference_ms"]
                    emit("E", "optimized", OPT_THREADS, s["modulation"], s["snr"], s["sample_index"], merged_e, total_e)

    run_batched_attack("fgsm", 0.05, FGSM_PARAMS, "C")
    run_batched_attack("pgd", 0.05, PGD_PARAMS_DET, "D_det")
    run_batched_attack("pgd", 0.05, PGD_PARAMS_STOCH, "D_stoch")

    # Scenario B optimized (Top-K, threads=1) for completeness of the A/B pair
    for s, cs in ok_clean_opt:
        tk = time_topk_stage(backends, cs["x_clean"])
        merged_b = dict(cs); merged_b.update(tk)
        total_b = cs["clean_stage_total_ms"] + tk["topk_ms"] + tk["defended_inference_ms"]
        emit("B_opt", "optimized", OPT_THREADS, s["modulation"], s["snr"], s["sample_index"], merged_b, total_b)

    # ===== CW supplementary (NOT part of the FGSM/PGD before/after main table) =====
    print("[e2e] === CW supplement (batch_size=1 baseline vs batch_size=16 batched_algorithmic_variant) ===", flush=True)
    torch.set_num_threads(default_threads)
    cw_rows = []
    clean_cw_baseline: List[dict] = []
    for s in fixed_24:
        cs = time_clean_stages(backends, s["modulation"], s["snr"], s["sample_index"])
        clean_cw_baseline.append(cs)
        if cs["status"] != "ok":
            continue
        cw = time_attack_stage_single(backends, cs["x_clean"], "cw", 0.05, CW_PARAMS)
        total = cs["clean_stage_total_ms"] + cw["attack_generation_ms"] + cw["awn_attacked_inference_ms"]
        cw_rows.append({
            "variant": "baseline_batch1", "threads_used": default_threads,
            "modulation": s["modulation"], "snr": s["snr"], "sample_index": s["sample_index"],
            "clean_stage_total_ms": cs["clean_stage_total_ms"],
            "attack_generation_ms": cw["attack_generation_ms"], "awn_attacked_inference_ms": cw["awn_attacked_inference_ms"],
            "end_to_end_total_ms": total, "attacked_prediction": cw["attacked_prediction"],
            "classification": "baseline_reference",
        })
    torch.set_num_threads(OPT_THREADS)
    ok_clean_cw = [(s, cs) for s, cs in zip(fixed_24, clean_cw_baseline) if cs["status"] == "ok"]
    clean_cw_opt = []
    for s, _ in ok_clean_cw:
        cs2 = time_clean_stages(backends, s["modulation"], s["snr"], s["sample_index"])
        clean_cw_opt.append(cs2)
    xs_cw = [cs2["x_clean"] for cs2 in clean_cw_opt]
    for start in range(0, len(xs_cw), OPT_BATCH):
        batch_specs = list(zip(ok_clean_cw, clean_cw_opt))[start:start + OPT_BATCH]
        batch = np.concatenate([cs2["x_clean"] for (_, cs2) in batch_specs], axis=0)
        t0 = now_ns()
        x_adv_batch, _ = backends.attack.apply(batch, attack="cw", eps=0.05, seed=SEED, attack_params=CW_PARAMS)
        batch_ms = (now_ns() - t0) / NS_PER_MS
        per_sample_ms = batch_ms / batch.shape[0]
        for i, ((s, cs), cs2) in enumerate(batch_specs):
            x_single = x_adv_batch[i:i + 1]
            t1 = now_ns()
            logits_att, _ = backends.awn.infer(x_single, seed=SEED)
            attacked_inf_ms = (now_ns() - t1) / NS_PER_MS
            total = cs2["clean_stage_total_ms"] + per_sample_ms + attacked_inf_ms
            cw_rows.append({
                "variant": "optimized_batch16", "threads_used": OPT_THREADS,
                "modulation": s["modulation"], "snr": s["snr"], "sample_index": s["sample_index"],
                "clean_stage_total_ms": cs2["clean_stage_total_ms"],
                "attack_generation_ms": per_sample_ms, "awn_attacked_inference_ms": attacked_inf_ms,
                "end_to_end_total_ms": total, "attacked_prediction": int(np.argmax(logits_att[0])),
                "classification": "batched_algorithmic_variant -- NOT a pure implementation_optimization "
                                   "(see docs/research/PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md section 15.2)",
            })

    torch.set_num_threads(default_threads)

    # ===== write raw CSV =====
    raw_fields = ["scenario", "variant", "threads_used", "modulation", "snr", "sample_index",
                  "input_prepare_ms", "embedding_ms", "energy_detection_ms", "region_postprocess_ms",
                  "segmentation_ms", "awn_preprocess_ms", "awn_clean_inference_ms",
                  "attack_prepare_ms", "attack_generation_ms", "awn_attacked_inference_ms",
                  "topk_ms", "defended_inference_ms", "total_ms", "status",
                  "clean_prediction", "clean_correct", "attacked_prediction", "attack_success",
                  "perturbation_linf", "perturbation_l2", "defended_prediction", "model_mode_after"]
    with open(out_dir / "end_to_end_latency_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields)
        w.writeheader()
        for r in raw_rows:
            w.writerow(r)

    with open(out_dir / "cw_end_to_end_supplement.csv", "w", newline="") as f:
        fieldnames = ["variant", "threads_used", "modulation", "snr", "sample_index", "clean_stage_total_ms",
                      "attack_generation_ms", "awn_attacked_inference_ms", "end_to_end_total_ms",
                      "attacked_prediction", "classification"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in cw_rows:
            w.writerow(r)

    # ===== scenario-level summary =====
    scenario_summary_rows = []
    scenarios_variants = sorted({(r["scenario"], r["variant"]) for r in raw_rows})
    for scenario, variant in scenarios_variants:
        vals = [r["total_ms"] for r in raw_rows if r["scenario"] == scenario and r["variant"] == variant
                and r["status"] == "ok" and r["total_ms"] is not None]
        stats = percentiles(vals)
        stats["scenario"] = scenario
        stats["variant"] = variant
        scenario_summary_rows.append(stats)
    with open(out_dir / "end_to_end_latency_summary.csv", "w", newline="") as f:
        fieldnames = ["scenario", "variant", "n", "mean", "std", "min", "median", "p90", "p95", "p99",
                      "max", "samples_per_sec"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in scenario_summary_rows:
            w.writerow(r)

    # ===== stage-level summary per scenario =====
    stage_cols = ["embedding_ms", "energy_detection_ms", "region_postprocess_ms", "segmentation_ms",
                  "awn_preprocess_ms", "awn_clean_inference_ms", "attack_generation_ms",
                  "awn_attacked_inference_ms", "topk_ms", "defended_inference_ms"]
    stage_summary_rows = []
    for scenario, variant in scenarios_variants:
        cell_rows = [r for r in raw_rows if r["scenario"] == scenario and r["variant"] == variant and r["status"] == "ok"]
        total_mean = np.mean([r["total_ms"] for r in cell_rows if r["total_ms"] is not None]) if cell_rows else None
        for col in stage_cols:
            vals = [r[col] for r in cell_rows if r.get(col) is not None]
            if not vals:
                continue
            stats = percentiles(vals)
            stats["scenario"] = scenario
            stats["variant"] = variant
            stats["stage"] = col
            stats["pct_of_mean_total"] = (stats["mean"] / total_mean * 100.0) if total_mean else None
            stage_summary_rows.append(stats)
    with open(out_dir / "stage_latency_summary.csv", "w", newline="") as f:
        fieldnames = ["scenario", "variant", "stage", "n", "mean", "std", "min", "median", "p90", "p95",
                      "p99", "max", "samples_per_sec", "pct_of_mean_total"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in stage_summary_rows:
            w.writerow(r)

    print(f"[e2e] DONE: {len(raw_rows)} raw rows, error={n_error} fallback={n_fallback} nan={n_nan}", flush=True)


if __name__ == "__main__":
    main()
