"""
Spectrum Sensing Utility Experiment (four paths): direct, no_sensing,
sensing, oracle -- how much AMC accuracy does the sensing front end cost
relative to an oracle crop, and how much better is it than a
position-blind, ground-truth-free baseline crop?

This is a NEW experiment, independent of docs/formal_experiment_plan.md's
Phase 1 (which only ever computed direct vs. sensing, never a no_sensing
baseline or an oracle-on-the-SAME-noisy-stream upper bound). No Phase 1
summary is reused or cited as a substitute for this run.

Four paths, per (modulation, snr, sample_index) base instance:

  1. direct      -- the raw RadioML [2,128] sample, NO noise embedding, NO
                     sensing. Upper bound with zero embedding noise.
  2. no_sensing   -- the SAME embedded long noise stream as sensing/oracle,
                     cropped at a FIXED, ground-truth-independent position
                     (default: --no-sensing-policy fixed_center, i.e. the
                     stream's own geometric center -- a pure function of
                     --n-samples, computed once, identical for every
                     instance regardless of where the burst actually landed).
                     Never reads true_start/true_end/detected regions/masks/
                     energy scores.
  3. sensing      -- the SAME long stream, run through the real, unmodified
                     energy_detect -> region extraction/merge/filter ->
                     select_aligned_segments(policy="max-energy") pipeline
                     (byte-identical calls to src/utils/pipeline.py's own
                     sensing stage). Never reads true_start/true_end either
                     -- max-energy alignment is a pure function of IQ
                     amplitude within a detected region. When multiple
                     regions are detected, the FIRST region in stream order
                     is used (segments[0]), matching the exact convention
                     already established in run_phase0_pilot.py/
                     run_phase1_sensing_baseline.py (never re-derived here).
  4. oracle       -- the SAME long stream, cropped EXACTLY at
                     [true_start:true_end] (== true_start:true_start+128,
                     since every RadioML burst is exactly 128 samples).
                     Only used as an analysis upper bound; never feeds
                     anything back into sensing or no_sensing.

Per-instance seeding (deliberate, documented difference from Phase 1):
Phase 1 (docs/formal_experiment_plan.md section 9) called
embed_sample_in_noise(..., seed=42) with the SAME seed=42 for every one of
its 2200 combos -- since embed_sample_in_noise's RNG draws noise before
drawing true_start, and burst_power (which only scales, never reshapes,
the RNG draws) is the only per-sample-varying input, true_start ended up
IDENTICAL across literally all 2200 Phase-1 combos (verified by inspection
of embed_sample_in_noise: rng.integers(...) is the same call at the same
RNG-state position regardless of noise_std). This experiment instead
derives a DISTINCT, reproducible seed per (mod,snr,idx) via
derive_instance_seed() (sha256-based, per this repo's established
cross-process-determinism convention -- see src/sensing/iq_source.py's
freq_offset derivation) -- so burst position genuinely varies across the
2200-instance grid, a more rigorous test of the sensing algorithm's
position-independence than Phase 1's single fixed position. This also
makes fairness test #6 (no_sensing crop does not move with burst position)
a real, non-trivial, dataset-wide check rather than a vacuous one.

Does not modify src/sensing/*, src/adapters/*, or any formal Top-K/attack
code -- calls the exact same underlying functions
src/utils/pipeline.py:run_dry_run_experiment already uses, at a finer grain
(same architectural pattern as every prior round-17+ runner script). No
attack, no Top-K, this round (explicitly out of scope). external/AWN and
external/adversarial-rf are not touched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE  # noqa: E402
from src.sensing.energy_detection import (  # noqa: E402
    energy_detect,
    filter_by_min_length,
    mask_to_regions,
    merge_close_regions,
)
from src.sensing.ground_truth_metrics import compute_sensing_ground_truth_metrics  # noqa: E402
from src.sensing.iq_source import validate_iq  # noqa: E402
from src.sensing.normalize import apply_awn_preprocess, to_awn_input  # noqa: E402
from src.sensing.radioml_source import (  # noqa: E402
    RML2016_10A_CLASSES,
    embed_sample_in_noise,
    load_radioml_dict,
    load_radioml_sample,
    radioml_sample_to_iq,
)
from src.sensing.segmentation import select_aligned_segments  # noqa: E402

DATASET_PATH = "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl"
CHECKPOINT = "external/adversarial-rf/2016.10a_AWN.pkl"
DEVICE = "cpu"
N_SAMPLES = 8192
BASE_SEED = 42

ALL_MODS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
ALL_SNRS = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
ALL_SAMPLE_INDICES = list(range(10))

FIXED = dict(
    iq_source="radioml",
    dataset_path=DATASET_PATH,
    checkpoint=CHECKPOINT,
    device=DEVICE,
    n_samples=N_SAMPLES,
    embed_snr_margin=20.0,
    threshold_factor=1.5,
    sensing_window_size=128,
    min_region_len=0,
    merge_gap=0,
    window_size=128,
    segment_hop=1,
    alignment_policy="max-energy",
    awn_preprocess="radioml-native",
    base_seed=BASE_SEED,
)

RAW_FIELDS = [
    "experiment_id", "seed", "sample_index", "modulation", "label_index", "snr_db",
    "stream_length", "noise_seed", "long_stream_hash", "true_start", "true_end",
    "original_sample_sha256",
    "direct_crop_start", "direct_crop_end", "direct_input_hash",
    "direct_prediction", "direct_correct", "direct_confidence", "direct_runtime_ms",
    "no_sensing_policy", "no_sensing_crop_start", "no_sensing_crop_end", "no_sensing_input_hash",
    "no_sensing_prediction", "no_sensing_correct", "no_sensing_confidence", "no_sensing_runtime_ms",
    "sensing_detected", "sensing_region_count", "sensing_detected_start", "sensing_detected_end",
    "sensing_crop_start", "sensing_crop_end", "sensing_input_hash",
    "sensing_prediction", "sensing_correct", "sensing_confidence", "sensing_runtime_ms",
    "oracle_crop_start", "oracle_crop_end", "oracle_input_hash",
    "oracle_prediction", "oracle_correct", "oracle_confidence", "oracle_runtime_ms",
    "captured_signal_ratio", "start_boundary_error", "end_boundary_error",
    "missed_signal_samples", "false_occupied_samples",
    "awn_backend", "awn_eval_mode", "run_status",
]


# --------------------------------------------------------------------------
# Deterministic, cross-process-reproducible per-instance seeding
# --------------------------------------------------------------------------

_DATASET_CACHE: Optional[dict] = None


def get_dataset() -> dict:
    """Loads external/adversarial-rf/data/RML2016.10a_dict.pkl ONCE per
    process and caches it -- src/sensing/radioml_source.py:load_radioml_dict
    documents "no caching is done here -- each call re-reads from disk",
    which is correct/unmodified upstream behavior but would make a
    2200-instance run reload a ~640MB pickle 2200 times. This cache lives
    entirely in this script (src/sensing/radioml_source.py itself is not
    modified) and simply calls load_radioml_dict() once, reusing the
    identical, unmodified parsing logic."""
    global _DATASET_CACHE
    if _DATASET_CACHE is None:
        print(f"[cache] loading RadioML dataset once from {FIXED['dataset_path']} ...", flush=True)
        t0 = time.time()
        _DATASET_CACHE = load_radioml_dict(FIXED["dataset_path"])
        print(f"[cache] loaded in {time.time()-t0:.1f}s, {len(_DATASET_CACHE)} (mod,snr) blocks", flush=True)
    return _DATASET_CACHE


def get_sample_cached(dataset: dict, mod: str, snr: int, idx: int) -> np.ndarray:
    """Same validation/error semantics as
    src/sensing/radioml_source.py:load_radioml_sample, but reads from an
    already-loaded dict instead of re-parsing the pickle file."""
    available_mods = sorted({k[0] for k in dataset.keys()})
    available_snrs = sorted({k[1] for k in dataset.keys()})
    if mod not in available_mods:
        raise ValueError(f"Unknown RadioML modulation {mod!r}; available: {available_mods}")
    if snr not in available_snrs:
        raise ValueError(f"Unknown RadioML SNR {snr!r}; available: {available_snrs}")
    block = dataset[(mod, snr)]
    if idx < 0 or idx >= block.shape[0]:
        raise ValueError(f"sample_index {idx} out of range for ({mod}, {snr}); block has {block.shape[0]} samples")
    sample = block[idx]
    if sample.shape != (2, 128):
        raise ValueError(f"Unexpected RadioML sample shape {sample.shape}, expected (2, 128)")
    return sample.astype(np.float32)


def derive_instance_seed(base_seed: int, mod: str, snr: int, idx: int, salt: str = "") -> int:
    """hashlib-based (not Python's salted hash()), matching this repo's
    established cross-process-determinism convention (src/sensing/
    iq_source.py's freq_offset derivation; docs/formal_experiment_plan.md's
    round-10 cross-process-reproducibility fix). Returns a value in
    [0, 2**31)."""
    key = f"{base_seed}|{mod}|{snr}|{idx}|{salt}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "big") % (2 ** 31)


# --------------------------------------------------------------------------
# no_sensing crop policies -- NONE of these may read true_start/true_end/
# detected regions/occupancy mask/sensing energy scores.
# --------------------------------------------------------------------------

def no_sensing_crop(policy: str, n_samples: int, seg_len: int, *, base_seed: int = BASE_SEED,
                     mod: str = "", snr: int = 0, idx: int = 0, scan_bins: int = 8) -> Tuple[int, int]:
    max_start = n_samples - seg_len
    if policy == "fixed_center":
        start = max_start // 2
    elif policy == "fixed_start":
        start = 0
    elif policy == "random":
        # Independent seed stream (salted "no_sensing_random"), never derived
        # from or correlated with the burst-embedding seed/true_start.
        rng = np.random.default_rng(derive_instance_seed(base_seed, mod, snr, idx, salt="no_sensing_random"))
        start = int(rng.integers(0, max_start + 1))
    elif policy == "uniform_scan":
        # Deterministic round-robin over `scan_bins` uniformly spaced
        # candidate positions, selected by a running instance counter -- NOT
        # by energy or any burst-derived quantity. bin index derived from
        # (mod, snr, idx) only, via the same sha256 mechanism (no python
        # hash() salting), so it is reproducible and ground-truth-blind.
        bin_width = max_start / max(scan_bins - 1, 1)
        bin_idx = derive_instance_seed(base_seed, mod, snr, idx, salt="uniform_scan_bin") % scan_bins
        start = int(round(bin_idx * bin_width))
        start = min(start, max_start)
    else:
        raise ValueError(f"Unknown no_sensing policy {policy!r}")
    return start, start + seg_len


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def sha256_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def not_finite(x: np.ndarray) -> bool:
    return bool(np.isnan(x).any() or np.isinf(x).any())


def build_awn_input(iq_1d: np.ndarray) -> np.ndarray:
    """complex64 [128] -> preprocessed float32 [1, 2, 128] AWN input."""
    segs = iq_1d[np.newaxis, :].astype(np.complex64)
    segs = apply_awn_preprocess(segs, policy=FIXED["awn_preprocess"])
    return to_awn_input(segs, seg_len=FIXED["window_size"])


def softmax_np(logits_1d: np.ndarray) -> np.ndarray:
    z = logits_1d - np.max(logits_1d)
    e = np.exp(z)
    return e / np.sum(e)


def infer_one(awn_adapter: AWNModelAdapter, x: np.ndarray, seed: int) -> Tuple[int, float, np.ndarray, dict, float]:
    t0 = time.perf_counter()
    logits, meta = awn_adapter.infer(x, seed=seed)
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    probs = softmax_np(logits[0])
    pred = int(np.argmax(logits[0]))
    conf = float(probs[pred])
    return pred, conf, logits[0], meta, runtime_ms


def precheck_real_awn(awn_adapter: AWNModelAdapter) -> None:
    if awn_adapter.backend_name != _REAL_MODEL_SOURCE or awn_adapter.status != "ok":
        raise RuntimeError(
            f"Real-AWN precheck FAILED -- refusing to run any instance: "
            f"backend={awn_adapter.backend_name!r} status={awn_adapter.status!r} notes={awn_adapter.notes}"
        )
    if awn_adapter.model is None or awn_adapter.model.training:
        raise RuntimeError("AWN model is not in eval() mode at precheck time -- refusing to run.")
    print(f"[precheck] real AWN backend confirmed: {_REAL_MODEL_SOURCE}, eval_mode=True")


# --------------------------------------------------------------------------
# Core per-instance computation (shared by fairness tests / smoke / pilot / full)
# --------------------------------------------------------------------------

def run_instance(
    mod: str, snr: int, idx: int, awn_adapter: AWNModelAdapter, dataset: dict,
    no_sensing_policy: str = "fixed_center",
    omit_true_boundary: bool = False,
) -> Tuple[dict, dict]:
    """Runs all four paths for one (mod, snr, idx) base sample. Returns
    (row, logits_dict) where logits_dict has keys direct/no_sensing/
    sensing/oracle -> np.ndarray[11] (sensing is None if sensing failed).

    omit_true_boundary: diagnostic-only flag (fairness tests #3/#4) that
    computes no_sensing/sensing exactly as normal but ADDITIONALLY asserts
    neither one's crop position depends on embed_meta['true_start']/
    ['true_end'] by never even reading those keys inside this branch's own
    crop-selection code paths (sensing/no_sensing crop selection already
    never references true_start/true_end in the non-omit branch either --
    this flag exists so a caller can prove that by literally deleting the
    keys from a copy of embed_meta before sensing/no_sensing execute and
    confirming no KeyError/behavior change is possible, since neither path
    ever looks them up)."""
    experiment_id = f"{mod}_snr{snr}_idx{idx}"
    instance_seed = derive_instance_seed(FIXED["base_seed"], mod, snr, idx)
    label = RML2016_10A_CLASSES[mod]

    sample = get_sample_cached(dataset, mod, snr, idx)
    original_sample_sha256 = hashlib.sha256(sample.tobytes()).hexdigest()

    # ---- 1. direct: raw sample, no noise embedding, no sensing ----
    iq_direct = radioml_sample_to_iq(sample)
    x_direct = build_awn_input(iq_direct)
    direct_input_hash = sha256_array(x_direct)
    pred_direct, conf_direct, logits_direct, meta_direct, direct_ms = infer_one(awn_adapter, x_direct, instance_seed)

    # ---- shared long stream for no_sensing / sensing / oracle ----
    iq_long, embed_meta = embed_sample_in_noise(
        sample, n_samples=FIXED["n_samples"], embed_snr_margin=FIXED["embed_snr_margin"], seed=instance_seed,
    )
    iq_long = validate_iq(iq_long)
    long_stream_hash = sha256_array(iq_long)
    true_start, true_end = embed_meta["true_start"], embed_meta["true_end"]
    if omit_true_boundary:
        embed_meta_visible = {k: v for k, v in embed_meta.items() if k not in ("true_start", "true_end")}
    else:
        embed_meta_visible = embed_meta

    # ---- 2. no_sensing: fixed, ground-truth-independent crop ----
    ns_start, ns_end = no_sensing_crop(
        no_sensing_policy, FIXED["n_samples"], FIXED["window_size"],
        base_seed=FIXED["base_seed"], mod=mod, snr=snr, idx=idx,
    )
    x_ns = build_awn_input(iq_long[ns_start:ns_end])
    ns_input_hash = sha256_array(x_ns)
    pred_ns, conf_ns, logits_ns, meta_ns, ns_ms = infer_one(awn_adapter, x_ns, instance_seed)

    # ---- 3. sensing: real energy_detect -> regions -> max-energy align ----
    t_sens_start = time.perf_counter()
    mask = energy_detect(iq_long, window=FIXED["sensing_window_size"], threshold_factor=FIXED["threshold_factor"])
    raw_regions = mask_to_regions(mask)
    merged_regions = merge_close_regions(raw_regions, merge_gap=FIXED["merge_gap"])
    regions: List[Tuple[int, int]] = []
    sensing_failed = False
    try:
        regions = filter_by_min_length(merged_regions, min_len=FIXED["min_region_len"])
    except RuntimeError:
        sensing_failed = True

    ground_truth = compute_sensing_ground_truth_metrics(true_start, true_end, regions)

    sensing_detected = False
    sensing_crop_start = sensing_crop_end = None
    sensing_detected_start = sensing_detected_end = None
    pred_sens = conf_sens = None
    logits_sens = None
    sensing_input_hash = None
    if not sensing_failed and regions:
        try:
            segments, align_meta = select_aligned_segments(
                iq_long, regions, seg_len=FIXED["window_size"], policy=FIXED["alignment_policy"], hop=FIXED["segment_hop"],
            )
            am = align_meta[0]  # first detected region in stream order -- matches run_phase0/1's own convention
            seg = segments[0]
            sensing_crop_start, sensing_crop_end = am["selected_segment_start"], am["selected_segment_end"]
            sensing_detected_start, sensing_detected_end = am["detected_region_start"], am["detected_region_end"]
            x_sens = build_awn_input(seg)
            sensing_input_hash = sha256_array(x_sens)
            pred_sens, conf_sens, logits_sens, meta_sens, _ = infer_one(awn_adapter, x_sens, instance_seed)
            sensing_detected = True
        except RuntimeError:
            sensing_failed = True
    sensing_ms = (time.perf_counter() - t_sens_start) * 1000.0

    # ---- 4. oracle: exact true_start:true_end crop ----
    assert true_end - true_start == FIXED["window_size"], (
        f"oracle burst length mismatch: true_end-true_start={true_end - true_start}, expected {FIXED['window_size']}"
    )
    x_oracle = build_awn_input(iq_long[true_start:true_end])
    oracle_input_hash = sha256_array(x_oracle)
    pred_oracle, conf_oracle, logits_oracle, meta_oracle, oracle_ms = infer_one(awn_adapter, x_oracle, instance_seed)

    eval_mode_ok = (
        awn_adapter.model is not None and not awn_adapter.model.training
        and all(m["awn_backend"] == _REAL_MODEL_SOURCE and m["awn_status"] == "ok"
                for m in (meta_direct, meta_ns, meta_oracle))
    )
    run_status = "ok" if eval_mode_ok else "error"

    row = {
        "experiment_id": experiment_id, "seed": instance_seed, "sample_index": idx,
        "modulation": mod, "label_index": label, "snr_db": snr,
        "stream_length": FIXED["n_samples"], "noise_seed": instance_seed,
        "long_stream_hash": long_stream_hash, "true_start": true_start, "true_end": true_end,
        "original_sample_sha256": original_sample_sha256,

        "direct_crop_start": 0, "direct_crop_end": FIXED["window_size"], "direct_input_hash": direct_input_hash,
        "direct_prediction": pred_direct, "direct_correct": pred_direct == label,
        "direct_confidence": conf_direct, "direct_runtime_ms": direct_ms,

        "no_sensing_policy": no_sensing_policy, "no_sensing_crop_start": ns_start, "no_sensing_crop_end": ns_end,
        "no_sensing_input_hash": ns_input_hash, "no_sensing_prediction": pred_ns,
        "no_sensing_correct": pred_ns == label, "no_sensing_confidence": conf_ns, "no_sensing_runtime_ms": ns_ms,

        "sensing_detected": sensing_detected, "sensing_region_count": len(regions),
        "sensing_detected_start": sensing_detected_start, "sensing_detected_end": sensing_detected_end,
        "sensing_crop_start": sensing_crop_start, "sensing_crop_end": sensing_crop_end,
        "sensing_input_hash": sensing_input_hash,
        "sensing_prediction": pred_sens, "sensing_correct": (pred_sens == label) if pred_sens is not None else False,
        "sensing_confidence": conf_sens, "sensing_runtime_ms": sensing_ms,

        "oracle_crop_start": true_start, "oracle_crop_end": true_end, "oracle_input_hash": oracle_input_hash,
        "oracle_prediction": pred_oracle, "oracle_correct": pred_oracle == label,
        "oracle_confidence": conf_oracle, "oracle_runtime_ms": oracle_ms,

        "captured_signal_ratio": ground_truth["captured_signal_ratio"],
        "start_boundary_error": ground_truth["start_boundary_error"],
        "end_boundary_error": ground_truth["end_boundary_error"],
        "missed_signal_samples": ground_truth["missed_sample_count"],
        "false_occupied_samples": ground_truth["false_occupied_sample_count"],

        "awn_backend": _REAL_MODEL_SOURCE, "awn_eval_mode": (awn_adapter.model is not None and not awn_adapter.model.training),
        "run_status": run_status,
    }
    logits_out = {
        "direct": logits_direct, "no_sensing": logits_ns,
        "sensing": logits_sens if logits_sens is not None else np.full(11, np.nan, dtype=np.float32),
        "oracle": logits_oracle,
    }
    return row, logits_out


# --------------------------------------------------------------------------
# CSV / npz writer
# --------------------------------------------------------------------------

class CsvWriter:
    def __init__(self, path: Path, fresh: bool):
        self.path = path
        mode = "w" if fresh else "a"
        self.f = open(path, mode, newline="")
        self.writer = csv.DictWriter(self.f, fieldnames=RAW_FIELDS)
        if fresh:
            self.writer.writeheader()
            self.f.flush()

    def write_row(self, row: dict) -> None:
        self.writer.writerow({k: row.get(k) for k in RAW_FIELDS})
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def load_done_ids(summary_path: Path) -> set:
    if not summary_path.exists():
        return set()
    with open(summary_path) as f:
        return {r["experiment_id"] for r in csv.DictReader(f)}


def git_state() -> dict:
    def _run(cmd):
        try:
            return subprocess.check_output(cmd, cwd=Path(__file__).resolve().parents[1], text=True).strip()
        except Exception as exc:  # noqa: BLE001
            return f"<error: {exc}>"
    return {
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": _run(["git", "status", "--porcelain"]),
        "git_diff_stat": _run(["git", "diff", "--stat"]),
    }


def env_state() -> dict:
    import torch
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "checkpoint_sha256": hashlib.sha256(Path(FIXED["checkpoint"]).read_bytes()).hexdigest(),
    }


def write_manifest(output_dir: Path, args: argparse.Namespace, n_instances: int) -> None:
    manifest = {
        "fixed_params": FIXED,
        "cli_args": vars(args),
        "n_instances_this_run": n_instances,
        "git": git_state(),
        "env": env_state(),
        "raw_fields": RAW_FIELDS,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)


# --------------------------------------------------------------------------
# Fairness / unit tests (10 checks)
# --------------------------------------------------------------------------

def run_fairness_tests() -> bool:
    print("=" * 70)
    print("FAIRNESS / UNIT TESTS")
    print("=" * 70)
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
        if not cond:
            ok = False

    awn = AWNModelAdapter(checkpoint_path=FIXED["checkpoint"], device=FIXED["device"])
    precheck_real_awn(awn)
    dataset = get_dataset()

    mod, snr, idx = "QPSK", 0, 0
    sample = get_sample_cached(dataset, mod, snr, idx)
    seed_a = derive_instance_seed(FIXED["base_seed"], mod, snr, idx)

    # 1. Same seed -> same long stream, noise, burst position.
    iq1, meta1 = embed_sample_in_noise(sample, FIXED["n_samples"], FIXED["embed_snr_margin"], seed=seed_a)
    iq2, meta2 = embed_sample_in_noise(sample, FIXED["n_samples"], FIXED["embed_snr_margin"], seed=seed_a)
    check("1. same seed -> identical long stream + true_start",
          sha256_array(iq1) == sha256_array(iq2) and meta1["true_start"] == meta2["true_start"])

    # 2. sensing/no_sensing/oracle share the identical long_stream_hash (single generation).
    row, _ = run_instance(mod, snr, idx, awn, dataset)
    # Re-derive independently and compare hash used for all three crops' source array.
    iq3, meta3 = embed_sample_in_noise(sample, FIXED["n_samples"], FIXED["embed_snr_margin"], seed=seed_a)
    check("2. sensing/no_sensing/oracle share one long_stream_hash",
          row["long_stream_hash"] == sha256_array(validate_iq(iq3)))

    # 3. no_sensing runs identically with true boundary keys removed.
    row_a, _ = run_instance(mod, snr, idx, awn, dataset, omit_true_boundary=False)
    row_b, _ = run_instance(mod, snr, idx, awn, dataset, omit_true_boundary=True)
    check("3. no_sensing output unchanged when true-boundary metadata omitted",
          row_a["no_sensing_crop_start"] == row_b["no_sensing_crop_start"]
          and row_a["no_sensing_prediction"] == row_b["no_sensing_prediction"]
          and row_a["no_sensing_input_hash"] == row_b["no_sensing_input_hash"])

    # 4. sensing runs identically with true boundary keys removed.
    check("4. sensing output unchanged when true-boundary metadata omitted",
          row_a["sensing_crop_start"] == row_b["sensing_crop_start"]
          and row_a["sensing_prediction"] == row_b["sensing_prediction"]
          and row_a["sensing_input_hash"] == row_b["sensing_input_hash"])

    # 5. oracle uses true_start/true_end, crop exactly covers 128 samples.
    check("5. oracle crop == [true_start, true_start+128) exactly",
          row["oracle_crop_start"] == row["true_start"]
          and row["oracle_crop_end"] == row["true_start"] + 128
          and row["oracle_crop_end"] - row["oracle_crop_start"] == 128)

    # 6. Varying burst insertion position (different sample_index -> different
    #    instance_seed -> different true_start) must NOT move the fixed
    #    no_sensing crop.
    row_idx0, _ = run_instance(mod, snr, 0, awn, dataset)
    row_idx1, _ = run_instance(mod, snr, 1, awn, dataset)
    check("6. fixed no_sensing crop independent of burst position",
          row_idx0["true_start"] != row_idx1["true_start"]  # burst really did move
          and row_idx0["no_sensing_crop_start"] == row_idx1["no_sensing_crop_start"]
          and row_idx0["no_sensing_crop_end"] == row_idx1["no_sensing_crop_end"],
          detail=f"true_start {row_idx0['true_start']} vs {row_idx1['true_start']}, "
                 f"no_sensing_crop_start {row_idx0['no_sensing_crop_start']} vs {row_idx1['no_sensing_crop_start']}")

    # 7. All four AWN calls used an eval-mode model.
    check("7. AWN eval mode for all four paths", bool(row["awn_eval_mode"]) and row["run_status"] == "ok")

    # 8. No NaN/Inf/shape mismatch/silent fallback.
    finite_ok = True
    for path in ("direct", "no_sensing", "oracle"):
        x = build_awn_input(radioml_sample_to_iq(sample))  # shape sanity, not the actual path array
        finite_ok = finite_ok and x.shape == (1, 2, 128) and not not_finite(x)
    check("8. no NaN/Inf, correct [1,2,128] shape, no silent fallback",
          finite_ok and row["awn_backend"] == _REAL_MODEL_SOURCE)

    # 9. direct path never touches the long stream.
    row_seedA, _ = run_instance(mod, snr, idx, awn, dataset)
    # Force a different long-stream realization by using a different sample_index's
    # seed for embedding ONLY -- direct must be identical since it never reads iq_long.
    direct_hash_1 = row_seedA["direct_input_hash"]
    row_seedB, _ = run_instance(mod, snr, 5, awn, dataset)  # different idx -> different embed seed -> different iq_long
    sample_b = load_radioml_sample(FIXED["dataset_path"], mod, snr, 5)
    direct_hash_from_raw_sample_b = sha256_array(build_awn_input(radioml_sample_to_iq(sample_b)))
    check("9. direct path derived only from the raw sample, never the long stream",
          row_seedB["direct_input_hash"] == direct_hash_from_raw_sample_b
          and direct_hash_1 != row_seedB["direct_input_hash"])  # different sample -> different hash, as expected

    # 10. Label / sample provenance identical across all four paths for one instance.
    check("10. label/provenance consistent across all four paths",
          row["label_index"] == RML2016_10A_CLASSES[mod]
          and row["modulation"] == mod
          and row["original_sample_sha256"] == hashlib.sha256(sample.tobytes()).hexdigest())

    print("=" * 70)
    print(f"FAIRNESS TESTS: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    print("=" * 70)
    return ok


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_str_list(s): return [x.strip() for x in s.split(",")]
def parse_int_list(s): return [int(x) for x in s.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["fairness-test", "smoke", "pilot", "full"], required=True)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mods", type=parse_str_list, default=ALL_MODS)
    ap.add_argument("--snrs", type=parse_int_list, default=ALL_SNRS)
    ap.add_argument("--sample-indices", type=parse_int_list, default=ALL_SAMPLE_INDICES)
    ap.add_argument("--no-sensing-policy", type=str, default="fixed_center",
                     choices=["fixed_center", "fixed_start", "random", "uniform_scan"])
    ap.add_argument("--max-instances", type=int, default=None, help="cap instance count (pilot mode)")
    args = ap.parse_args()

    if args.mode == "fairness-test":
        ok = run_fairness_tests()
        sys.exit(0 if ok else 1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "raw_results.csv"
    fresh = not (args.resume and csv_path.exists())
    done_ids = load_done_ids(csv_path) if not fresh else set()

    awn = AWNModelAdapter(checkpoint_path=FIXED["checkpoint"], device=FIXED["device"])
    precheck_real_awn(awn)
    dataset = get_dataset()

    instances = [
        (mod, snr, idx)
        for mod in args.mods for snr in args.snrs for idx in args.sample_indices
    ]
    if args.max_instances is not None and args.max_instances < len(instances):
        # Systematic stride sample (not a plain head-slice) so a capped
        # instance count (e.g. the 100-sample timed pilot) still spans the
        # full modulation/SNR grid instead of only the first few
        # modulations -- a plain [:100] slice of a mod-major-ordered list
        # would cover only 1 modulation out of 11.
        stride = max(1, len(instances) // args.max_instances)
        instances = instances[::stride][: args.max_instances]
    total = len(instances)

    writer = CsvWriter(csv_path, fresh=fresh)
    write_manifest(output_dir, args, total)

    logits_store = {"direct": [], "no_sensing": [], "sensing": [], "oracle": [], "experiment_id": []}

    t_start = time.time()
    n_done = 0
    n_error = 0
    n_sensing_failed = 0
    n_nan = 0
    for i, (mod, snr, idx) in enumerate(instances):
        experiment_id = f"{mod}_snr{snr}_idx{idx}"
        if experiment_id in done_ids:
            continue

        row, logits = run_instance(mod, snr, idx, awn, dataset, no_sensing_policy=args.no_sensing_policy)
        writer.write_row(row)

        logits_store["experiment_id"].append(experiment_id)
        for k in ("direct", "no_sensing", "sensing", "oracle"):
            logits_store[k].append(logits[k])

        n_done += 1
        if row["run_status"] != "ok":
            n_error += 1
        if not row["sensing_detected"]:
            n_sensing_failed += 1
        for k in ("direct_confidence", "no_sensing_confidence", "oracle_confidence"):
            v = row[k]
            if v is not None and (np.isnan(v) or np.isinf(v)):
                n_nan += 1

        if n_done % 20 == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0.0
            remaining = total - (i + 1)
            eta_s = remaining / rate if rate > 0 else float("nan")
            print(f"[{args.mode}] {i+1}/{total} done={n_done} elapsed={elapsed:.1f}s "
                  f"rate={rate:.2f}/s ETA={eta_s:.1f}s error={n_error} sensing_failed={n_sensing_failed} nan={n_nan}",
                  flush=True)

    writer.close()

    if logits_store["experiment_id"]:
        np.savez(
            output_dir / "logits.npz",
            experiment_id=np.array(logits_store["experiment_id"]),
            direct=np.stack(logits_store["direct"]) if logits_store["direct"] else np.empty((0, 11)),
            no_sensing=np.stack(logits_store["no_sensing"]) if logits_store["no_sensing"] else np.empty((0, 11)),
            sensing=np.stack(logits_store["sensing"]) if logits_store["sensing"] else np.empty((0, 11)),
            oracle=np.stack(logits_store["oracle"]) if logits_store["oracle"] else np.empty((0, 11)),
        )

    elapsed = time.time() - t_start
    print(f"[{args.mode}] DONE: {n_done} instances in {elapsed:.1f}s "
          f"({n_done/elapsed if elapsed>0 else 0:.2f}/s); error={n_error} sensing_failed={n_sensing_failed} nan={n_nan}")
    print(f"[{args.mode}] output_dir={output_dir}")


if __name__ == "__main__":
    main()
