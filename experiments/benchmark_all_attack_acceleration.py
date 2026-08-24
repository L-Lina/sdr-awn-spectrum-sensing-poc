"""
17-Attack Acceleration Feasibility + Optimization Benchmark (project-close
supplementary round). Answers, per attack, real backend, CPU, RadioML2016.10a:
baseline latency, thread-tuning, batching feasibility/classification,
correctness of any optimization, and end-to-end implications.

Does not modify external/AWN, external/adversarial-rf, or any existing
results/ directory. Does not change any attack's eps/steps/objective/
stopping-threshold/algorithm -- only thread count and batch size are varied
as optimization candidates. AutoAttack's baseline uses version="rand"
(established in a prior compatibility round for CPU tractability, not
introduced this round for speedup, and disclosed as an algorithmic
constraint, not part of the speedup measurement).

Phases (see main()):
  0. registry preflight (confirm 17 attacks from the real adapter registry)
  1. 3-sample pilot per attack -> ETA -> fast-tier (60 samples) vs
     slow-tier (--slow-tier-samples, default 20) subset assignment
  2. baseline benchmark (batch_size=1, default torch threads)
  3. thread tuning (fixed 8-sample subset, all attacks, threads in
     --torch-threads)
  4. batching sweep + correctness validation at the best thread count
     (--batch-sizes, capped to subset size, deduplicated)
  5. small end-to-end validation (sensing -> clean AWN -> attack ->
     attacked inference) at each attack's best safe configuration

Writes: attack_acceleration_raw.csv, attack_acceleration_summary.csv,
attack_correctness_summary.csv, attack_batching_classification.csv,
attack_thread_tuning.csv, attack_bottleneck_summary.csv,
attack_e2e_summary.csv, manifest.json into --output-dir.
(terminal.log is produced by the caller redirecting/tee-ing this script's
stdout, matching experiments/run_cfile_pipeline_smoke.py's convention.)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.adapters.attack_adapter import AttackAdapter, _ATTACK_CLASS_MAP, _REAL_ATTACK_SOURCE  # noqa: E402
from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE  # noqa: E402
from src.sensing.energy_detection import energy_detect, filter_by_min_length, mask_to_regions, merge_close_regions  # noqa: E402
from src.sensing.normalize import apply_awn_preprocess, to_awn_input  # noqa: E402
from src.sensing.radioml_source import RML2016_10A_CLASSES, embed_sample_in_noise, load_radioml_dict, load_radioml_sample  # noqa: E402
from src.sensing.segmentation import select_aligned_segments  # noqa: E402
from src.utils.dataset_path import resolve_dataset_path  # noqa: E402

DATASET_PATH = resolve_dataset_path()  # priority: env $SDR_AWN_DATASET_PATH > legacy default
CHECKPOINT_PATH = "external/adversarial-rf/2016.10a_AWN.pkl"
NS_PER_MS = 1_000_000.0

# --- 17-attack registry (verified against src.adapters.attack_adapter._ATTACK_CLASS_MAP
# at runtime in registry_preflight(); PGD is benchmarked as two named conditions,
# pgd_det/pgd_stoch, per the task's explicit instruction to never claim PGD's
# random_start=True variant is bit-identical -- the "PGD" row in the final
# 17-row summary table uses pgd_det as primary, matching this repo's existing
# convention (batching implementation_optimization is only claimed for
# random_start=False elsewhere in this codebase too).
CANONICAL_17 = ["fgsm", "bim", "pgd", "mifgsm", "difgsm", "vmifgsm", "vnifgsm", "rfgsm",
                "tpgd", "cw", "deepfool", "fab", "square", "apgd", "apgdt", "autoattack", "ead"]
BENCH_ATTACKS = ["fgsm", "bim", "pgd_det", "pgd_stoch", "mifgsm", "difgsm", "vmifgsm", "vnifgsm",
                  "rfgsm", "tpgd", "cw", "deepfool", "fab", "square", "apgd", "apgdt", "autoattack", "ead"]
ATTACK_REAL_NAME = {"pgd_det": "pgd", "pgd_stoch": "pgd"}

EPS = 0.05
SEED = 0
CW_C, CW_STEPS, CW_LR = 1.0, 20.0, 0.01  # this repo's existing AttackAdapter.apply() defaults, unchanged


def attack_params_for(bench_name: str) -> dict:
    """Fixed, documented parameter set per attack -- identical for baseline
    and every optimization candidate (thread/batch), never tuned for speed.
    eps=0.05 (this repo's established convention, see docs/research/
    DIGITAL_LOW_PERTURBATION_ATTACK_EXPERIMENT_ZH_TW.md and the satellite-like
    Step 4 round) for every eps-accepting attack; every other parameter
    (steps, restarts, n_queries, etc.) is left at torchattacks' own installed
    default -- never reduced for this benchmark."""
    if bench_name == "pgd_det":
        return {"eps": EPS, "random_start": False}
    if bench_name == "pgd_stoch":
        return {"eps": EPS, "random_start": True}
    if bench_name == "difgsm":
        return {"eps": EPS, "seed": SEED}
    if bench_name in ("fab", "square", "apgd", "apgdt"):
        return {"eps": EPS, "seed": SEED}
    if bench_name == "autoattack":
        # version="rand": established in a prior compatibility round
        # (results/attack_compatibility_smoke_20260727T030223Z/) for CPU
        # tractability -- NOT introduced this round for speedup, and never
        # compared against a "standard"-ensemble baseline (not run: cost
        # prohibitive on CPU at benchmark scale). Disclosed as an
        # algorithmic_tradeoff/pre-existing scoping choice, not part of the
        # measured speedup.
        return {"eps": EPS, "version": "rand", "seed": SEED}
    if bench_name in ("cw", "deepfool", "ead"):
        return {}
    return {"eps": EPS}  # fgsm, bim, mifgsm, vmifgsm, vnifgsm, rfgsm, tpgd


def real_name(bench_name: str) -> str:
    return ATTACK_REAL_NAME.get(bench_name, bench_name)


def configured_steps(bench_name: str) -> Optional[int]:
    """Configured iteration/query count from the real, installed torchattacks
    class's OWN default constructor signature (introspected, not assumed) --
    NOT a runtime-instrumented forward/backward call count (that was not
    reliably measurable this round without invasive hooking into every one
    of 17 different optimizer/attack implementations; noted as a limitation,
    not silently omitted)."""
    cls = _ATTACK_CLASS_MAP.get(real_name(bench_name))
    if cls is None:
        return None
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return None
    for key in ("steps", "n_queries", "max_iterations"):
        if key in sig.parameters and sig.parameters[key].default is not inspect.Parameter.empty:
            return int(sig.parameters[key].default)
    return 1  # fgsm: single-step by construction


def now_ns() -> int:
    return time.perf_counter_ns()


def sha256_arr(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def pct_stats(values: List[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": len(arr), "mean": float(arr.mean()), "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)), "max": float(arr.max()),
    }


def build_clean_input(mod: str, snr: int, idx: int) -> np.ndarray:
    """Formal preprocessing path (matches experiments/profile_attacks.py):
    RadioML sample -> embed in noise -> real energy_detect -> segmentation
    -> radioml-native AWN preprocessing -> [1,2,128] AWN input tensor."""
    sample = load_radioml_sample(DATASET_PATH, mod, snr, idx)
    iq, _ = embed_sample_in_noise(sample, 8192, 20.0, seed=idx)
    mask = energy_detect(iq, window=128, threshold_factor=5.0)
    regions = filter_by_min_length(merge_close_regions(mask_to_regions(mask), merge_gap=0), min_len=128)
    segments, _ = select_aligned_segments(iq, regions, seg_len=128, policy="max-energy", hop=1)
    x = apply_awn_preprocess(segments[:1], policy="radioml-native")
    return to_awn_input(x, seg_len=128)


class Fixture:
    """One (mod, snr, idx) spec + its true_label + cached AWN-input tensor."""
    __slots__ = ("mod", "snr", "idx", "true_label", "x")

    def __init__(self, mod: str, snr: int, idx: int):
        self.mod, self.snr, self.idx = mod, snr, idx
        self.true_label = RML2016_10A_CLASSES[mod]
        self.x = build_clean_input(mod, snr, idx)


def call_attack(attack: AttackAdapter, awn: AWNModelAdapter, x_batch: np.ndarray, bench_name: str, seed_offset: int = 0):
    params = dict(attack_params_for(bench_name))
    if "seed" in params:
        params["seed"] = SEED + seed_offset
    t0 = now_ns()
    if real_name(bench_name) == "cw":
        x_adv, meta = attack.apply(x_batch, attack="cw", eps=EPS, seed=SEED + seed_offset,
                                    cw_c=CW_C, cw_steps=int(CW_STEPS), cw_lr=CW_LR)
    else:
        x_adv, meta = attack.apply(x_batch, attack=real_name(bench_name), eps=EPS,
                                    seed=SEED + seed_offset, attack_params=params)
    elapsed_ms = (now_ns() - t0) / NS_PER_MS
    return x_adv, meta, elapsed_ms


def registry_preflight(log) -> List[str]:
    from src.adapters.attack_adapter import _ATTACK_ACCEPTED_PARAMS
    actual = sorted(_ATTACK_ACCEPTED_PARAMS.keys())
    log(f"registry preflight: {len(actual)} attacks in _ATTACK_ACCEPTED_PARAMS: {actual}")
    expected = sorted(CANONICAL_17)
    if actual != expected:
        log(f"registry MISMATCH vs canonical 17: expected={expected} actual={actual}")
    else:
        log("registry matches canonical 17 exactly.")
    return actual


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--attacks", default=",".join(BENCH_ATTACKS))
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    ap.add_argument("--torch-threads", default="1,2,4,8,16")
    ap.add_argument("--samples-per-cell", type=int, default=5)
    ap.add_argument("--slow-tier-samples", type=int, default=20)
    ap.add_argument("--thread-tuning-samples", type=int, default=8)
    ap.add_argument("--pilot-samples", type=int, default=3)
    ap.add_argument("--eta-slow-threshold-min", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_start = now_ns()

    def log(msg: str) -> None:
        print(f"[bench] {msg}", flush=True)

    import torch
    default_threads = torch.get_num_threads()

    bench_attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    batch_sizes = sorted({int(b) for b in args.batch_sizes.split(",")})
    thread_settings = sorted({int(t) for t in args.torch_threads.split(",")})

    log(f"loading real backends (dataset_path={DATASET_PATH}) ...")
    awn = AWNModelAdapter(checkpoint_path=CHECKPOINT_PATH, device="cpu")
    if awn.backend_name != _REAL_MODEL_SOURCE or awn.status != "ok":
        raise RuntimeError(f"Real-AWN precheck FAILED: backend={awn.backend_name} status={awn.status}")
    attack_adapter = AttackAdapter(awn_model=awn.model, device="cpu")
    if attack_adapter.wrapped_model is None or attack_adapter.backend_name != _REAL_ATTACK_SOURCE:
        raise RuntimeError(f"Real-attack precheck FAILED: backend={attack_adapter.backend_name}")
    log(f"real backends confirmed: AWN={_REAL_MODEL_SOURCE}, attack={_REAL_ATTACK_SOURCE}, default_torch_threads={default_threads}")

    actual_registry = registry_preflight(log)

    # ---- fixture pools ----
    MODS = ["BPSK", "QPSK", "QAM16", "WBFM"]
    SNRS = [-10, 0, 18]
    fast_specs = [(m, s, i) for m in MODS for s in SNRS for i in range(args.samples_per_cell)]
    log(f"building fast-tier fixture pool: {len(fast_specs)} samples ({MODS} x {SNRS} x {args.samples_per_cell}/cell) ...")
    fast_fixtures = [Fixture(m, s, i) for (m, s, i) in fast_specs]
    log(f"fast-tier fixture pool ready: {len(fast_fixtures)} samples")

    slow_n = min(args.slow_tier_samples, len(fast_fixtures))
    step = max(1, len(fast_fixtures) // slow_n)
    slow_fixtures = fast_fixtures[::step][:slow_n]
    log(f"slow-tier fixed subset: {len(slow_fixtures)} samples (evenly spaced from fast-tier pool)")

    tt_n = min(args.thread_tuning_samples, len(fast_fixtures))
    tt_fixtures = fast_fixtures[:tt_n]
    log(f"thread-tuning fixed subset: {len(tt_fixtures)} samples (shared across all attacks)")

    pilot_fixtures = fast_fixtures[:args.pilot_samples]

    raw_rows: List[dict] = []
    correctness_rows: List[dict] = []
    classification_rows: List[dict] = []
    thread_rows: List[dict] = []
    bottleneck_rows: List[dict] = []
    e2e_rows: List[dict] = []
    tier_by_attack: Dict[str, str] = {}
    n_error = 0
    n_unexpected_fallback = 0

    def safe_call(bench_name, x, seed_offset, phase, extra=None):
        nonlocal n_error, n_unexpected_fallback
        try:
            x_adv, meta, ms = call_attack(attack_adapter, awn, x, bench_name, seed_offset)
            fell_back = meta["attack_backend"] != _REAL_ATTACK_SOURCE or meta["attack_status"] != "ok"
            if fell_back:
                n_unexpected_fallback += 1
            row = {
                "attack": bench_name, "phase": phase, "n_in_batch": x.shape[0],
                "elapsed_ms": ms, "backend": meta["attack_backend"], "status": meta["attack_status"],
                "has_nan": bool(np.isnan(x_adv).any()), "has_inf": bool(np.isinf(x_adv).any()),
                "fallback": fell_back, "error": None,
            }
            if extra:
                row.update(extra)
            raw_rows.append(row)
            return x_adv, meta, ms
        except Exception as exc:  # noqa: BLE001 -- record and continue, never crash the whole matrix
            n_error += 1
            row = {"attack": bench_name, "phase": phase, "n_in_batch": x.shape[0], "elapsed_ms": None,
                   "backend": None, "status": "error", "has_nan": None, "has_inf": None,
                   "fallback": None, "error": f"{type(exc).__name__}: {exc}"}
            if extra:
                row.update(extra)
            raw_rows.append(row)
            log(f"  !! {bench_name} [{phase}] ERROR: {type(exc).__name__}: {exc}")
            return None, None, None

    # ---- Phase 1: pilot + tier assignment ----
    # ETA projects the FULL common-subset-based benchmark for that attack (baseline +
    # thread-tuning + batching ref + batching sweep), not just the baseline pass alone --
    # "完整 common subset 預估 > 30 分鐘/attack" in the task is the whole per-attack
    # benchmark, so the projection must account for every phase that reuses the fast-tier
    # sample count, or a genuinely slow attack could still balloon the batching phase even
    # though its lone baseline pass looked cheap.
    n_extra_batch_configs = max(0, len([b for b in batch_sizes if b <= len(fast_fixtures)]) - 1)
    fast_tier_equiv_calls = (
        len(fast_fixtures)  # baseline
        + len(tt_fixtures) * len(thread_settings)  # thread tuning
        + len(fast_fixtures)  # batching ref (batch=1)
        + len(fast_fixtures) * n_extra_batch_configs  # batching sweep, remaining batch sizes
    )
    log(f"=== Phase 1: {len(pilot_fixtures)}-sample pilot per attack -> ETA (full per-attack pipeline, "
        f"~{fast_tier_equiv_calls} sample-equivalent calls at fast-tier) -> tier assignment ===")
    for bench_name in bench_attacks:
        times = []
        for i, fx in enumerate(pilot_fixtures):
            _, _, ms = safe_call(bench_name, fx.x, seed_offset=i, phase="pilot")
            if ms is not None:
                times.append(ms)
        if not times:
            tier_by_attack[bench_name] = "fast"  # all pilot calls errored; still attempt fast tier, will error again and be visible
            log(f"  {bench_name}: pilot all failed, defaulting to fast tier (errors will be visible in raw CSV)")
            continue
        per_sample_ms = statistics.median(times)
        projected_min = (per_sample_ms * fast_tier_equiv_calls) / 1000.0 / 60.0
        tier = "slow" if projected_min > args.eta_slow_threshold_min else "fast"
        tier_by_attack[bench_name] = tier
        n_used = len(slow_fixtures) if tier == "slow" else len(fast_fixtures)
        log(f"  {bench_name}: pilot median={per_sample_ms:.2f}ms -> projected {projected_min:.2f} min for the full "
            f"fast-tier benchmark -> tier={tier} (n={n_used})")

    # ---- Phase 2: baseline benchmark (batch_size=1, default threads) ----
    log(f"=== Phase 2: baseline benchmark (batch_size=1, torch_threads={default_threads} [current default]) ===")
    torch.set_num_threads(default_threads)
    baseline_stats: Dict[str, dict] = {}
    for bench_name in bench_attacks:
        subset = slow_fixtures if tier_by_attack[bench_name] == "slow" else fast_fixtures
        times = []
        for i, fx in enumerate(subset):
            _, _, ms = safe_call(bench_name, fx.x, seed_offset=i, phase="baseline",
                                  extra={"tier": tier_by_attack[bench_name], "torch_threads": default_threads, "batch_size": 1})
            if ms is not None:
                times.append(ms)
        stats = pct_stats(times)
        baseline_stats[bench_name] = stats
        log(f"  {bench_name}: baseline n={stats['n']} median={stats['median']} p95={stats['p95']} max={stats['max']}")

    # ---- Phase 3: thread tuning ----
    log(f"=== Phase 3: thread tuning (n={len(tt_fixtures)} samples/attack, threads={thread_settings}) ===")
    thread_best: Dict[str, dict] = {}
    for bench_name in bench_attacks:
        per_thread = {}
        for th in thread_settings:
            torch.set_num_threads(th)
            times = []
            for i, fx in enumerate(tt_fixtures):
                _, _, ms = safe_call(bench_name, fx.x, seed_offset=1000 + i, phase="thread_tuning",
                                      extra={"tier": tier_by_attack[bench_name], "torch_threads": th, "batch_size": 1})
                if ms is not None:
                    times.append(ms)
            stats = pct_stats(times)
            per_thread[th] = stats
            thread_rows.append({"attack": bench_name, "torch_threads": th, **stats})
        valid = {th: s for th, s in per_thread.items() if s["median"] is not None}
        if valid:
            best_th = min(valid, key=lambda th: valid[th]["median"])
        else:
            best_th = default_threads
        thread_best[bench_name] = {"best_threads": best_th, "stats": per_thread.get(best_th, {})}
        log(f"  {bench_name}: best_threads={best_th} median={per_thread.get(best_th, {}).get('median')}")
    torch.set_num_threads(default_threads)

    # ---- Phase 4: batching sweep + correctness ----
    log("=== Phase 4: batching sweep + correctness validation (at best thread count) ===")
    batching_best: Dict[str, dict] = {}
    for bench_name in bench_attacks:
        subset = slow_fixtures if tier_by_attack[bench_name] == "slow" else fast_fixtures
        best_th = thread_best[bench_name]["best_threads"]
        torch.set_num_threads(best_th)
        n = len(subset)
        cand_batches = sorted({b for b in batch_sizes if b <= n} | {1})

        # reference: batch_size=1, one call per sample (also the correctness baseline)
        ref_preds, ref_x = [], []
        ref_times = []
        for i, fx in enumerate(subset):
            x_adv, meta, ms = safe_call(bench_name, fx.x, seed_offset=2000 + i, phase="batching_ref",
                                         extra={"tier": tier_by_attack[bench_name], "torch_threads": best_th, "batch_size": 1})
            # a fallback (dummy_attack) result must never contribute to latency/correctness
            # aggregation -- it is not a measurement of the real attack, treat it like an error.
            if x_adv is not None and meta is not None and meta["attack_backend"] == _REAL_ATTACK_SOURCE:
                pred = int(np.argmax(awn.infer(x_adv, seed=SEED)[0][0]))
                ref_preds.append(pred)
                ref_x.append(x_adv)
                ref_times.append(ms)
            else:
                ref_preds.append(None)
                ref_x.append(None)
        ref_stats = pct_stats([t for t in ref_times if t is not None])

        per_batch_result = {1: {"stats": ref_stats, "max_diff": 0.0, "pred_match_rate": 1.0,
                                 "linf_diff": 0.0, "l2_diff": 0.0}}
        for b in [c for c in cand_batches if c != 1]:
            batch_times = []
            batch_preds: List[Optional[int]] = []
            batch_x: List[Optional[np.ndarray]] = []
            for start in range(0, n, b):
                chunk = subset[start:start + b]
                x_in = np.concatenate([fx.x for fx in chunk], axis=0)
                x_adv, meta, ms = safe_call(bench_name, x_in, seed_offset=3000 + start, phase="batching_test",
                                             extra={"tier": tier_by_attack[bench_name], "torch_threads": best_th, "batch_size": b})
                # same fallback-exclusion rule as the batching_ref loop above: a dummy_attack
                # fallback result (e.g. torchattacks.Square/APGDT/AutoAttack occasionally raising
                # "Expected input batch_size (N) to match target batch_size (1)" at N>1 -- an
                # intermittent, input-dependent third-party-library limitation, not an adapter bug,
                # see docs/research/ALL_ATTACK_ACCELERATION_ANALYSIS_ZH_TW.md) must never be
                # counted as a valid latency or correctness measurement for that batch size.
                if x_adv is not None and meta is not None and meta["attack_backend"] == _REAL_ATTACK_SOURCE:
                    per_sample_ms = ms / x_in.shape[0]
                    batch_times.extend([per_sample_ms] * x_in.shape[0])
                    logits = awn.infer(x_adv, seed=SEED)[0]
                    for j in range(x_adv.shape[0]):
                        batch_preds.append(int(np.argmax(logits[j])))
                        batch_x.append(x_adv[j:j + 1])
                else:
                    for _ in range(x_in.shape[0]):
                        batch_preds.append(None)
                        batch_x.append(None)
            # correctness vs ref
            diffs, l2s, matches = [], [], []
            for i in range(n):
                if ref_x[i] is None or batch_x[i] is None:
                    continue
                d = np.abs(ref_x[i].astype(np.float64) - batch_x[i].astype(np.float64))
                diffs.append(float(d.max()))
                l2s.append(float(np.linalg.norm(ref_x[i].astype(np.float64) - batch_x[i].astype(np.float64))))
                matches.append(1 if (ref_preds[i] is not None and ref_preds[i] == batch_preds[i]) else 0)
            match_rate = float(np.mean(matches)) if matches else None
            max_diff = float(np.max(diffs)) if diffs else None
            mean_l2 = float(np.mean(l2s)) if l2s else None
            per_batch_result[b] = {"stats": pct_stats(batch_times), "max_diff": max_diff,
                                    "pred_match_rate": match_rate, "linf_diff": max_diff, "l2_diff": mean_l2}
            correctness_rows.append({
                "attack": bench_name, "batch_size": b, "n_compared": len(matches),
                "tensor_max_abs_diff": max_diff, "prediction_match_rate": match_rate,
                "mean_l2_diff": mean_l2,
            })

        # pick fastest batch size among those classified safe-ish (max_diff small or known stochastic)
        valid_batches = {b: r for b, r in per_batch_result.items() if r["stats"]["median"] is not None}
        best_b = min(valid_batches, key=lambda b: valid_batches[b]["stats"]["median"]) if valid_batches else 1
        batching_best[bench_name] = {"best_batch": best_b, "per_batch": per_batch_result}

        # classification (A/B/C/D)
        worst_diff = max((r["max_diff"] for b, r in per_batch_result.items() if b != 1 and r["max_diff"] is not None), default=None)
        worst_match = min((r["pred_match_rate"] for b, r in per_batch_result.items() if b != 1 and r["pred_match_rate"] is not None), default=None)
        if worst_diff is None:
            cls = "D_batching_unsafe"
            reason = "no successful batch>1 run"
        elif bench_name == "pgd_stoch":
            cls = "B_stochastic_batch_compatible"
            reason = f"random_start=True: inherent per-call RNG draw differs by batch layout; pred_match_rate={worst_match}"
        elif worst_diff < 1e-4 and worst_match == 1.0:
            cls = "A_implementation_optimization"
            reason = f"bit-identical within fp tolerance (max_diff={worst_diff:.2e}), 100% prediction match"
        elif worst_match is not None and worst_match >= 0.90:
            cls = "C_batched_algorithmic_variant"
            reason = f"batching changes optimization trajectory (max_diff={worst_diff:.4g}, pred_match_rate={worst_match:.3f})"
        else:
            cls = "D_batching_unsafe"
            reason = f"batching destabilizes predictions (pred_match_rate={worst_match})"
        classification_rows.append({
            "attack": bench_name, "classification": cls, "reason": reason,
            "best_batch_size": best_b, "best_threads": best_th,
            "worst_case_max_diff": worst_diff, "worst_case_pred_match_rate": worst_match,
        })
        log(f"  {bench_name}: classification={cls} best_batch={best_b} best_threads={best_th}")

    torch.set_num_threads(default_threads)

    # ---- Phase 5: end-to-end validation at best safe config ----
    log("=== Phase 5: end-to-end validation (sensing -> clean AWN -> attack -> attacked inference) ===")
    e2e_specs = fast_specs[:8]  # small, per task section 14 -- not a formal-scale matrix
    # load_radioml_sample() calls load_radioml_dict() internally, which re-reads and
    # re-unpickles the whole ~640MB dataset file from disk on every call (documented,
    # intentional, no caching -- see load_radioml_dict's own docstring). Fine when called
    # once per Fixture (Phases 1-4), but calling it inside this phase's per-iteration timing
    # loop would add ~1.1-1.2s of pure disk I/O to every single "total_ms" measurement,
    # swamping the actual sensing/AWN/attack latency this phase exists to measure -- so the
    # raw samples needed for e2e_specs are pre-fetched ONCE here, outside any timed region.
    e2e_raw_samples = {}
    _radioml_dict_for_e2e = load_radioml_dict(DATASET_PATH)
    for (m, s, idx) in e2e_specs:
        e2e_raw_samples[(m, s, idx)] = _radioml_dict_for_e2e[(m, s)][idx].astype(np.float32)

    def run_e2e_once(bench_name, spec, seed_offset):
        m, s, idx = spec
        t0 = now_ns()
        sample = e2e_raw_samples[(m, s, idx)]
        iq, _ = embed_sample_in_noise(sample, 8192, 20.0, seed=idx)
        mask = energy_detect(iq, window=128, threshold_factor=5.0)
        regions = filter_by_min_length(merge_close_regions(mask_to_regions(mask), merge_gap=0), min_len=128)
        segments, _ = select_aligned_segments(iq, regions, seg_len=128, policy="max-energy", hop=1)
        x = apply_awn_preprocess(segments[:1], policy="radioml-native")
        x = to_awn_input(x, seg_len=128)
        sensing_ms = (now_ns() - t0) / NS_PER_MS
        t1 = now_ns()
        awn.infer(x, seed=SEED)
        clean_ms = (now_ns() - t1) / NS_PER_MS
        x_adv, meta, attack_ms = call_attack(attack_adapter, awn, x, bench_name, seed_offset=seed_offset)
        t2 = now_ns()
        awn.infer(x_adv, seed=SEED)
        attacked_ms = (now_ns() - t2) / NS_PER_MS
        return attack_ms, sensing_ms + clean_ms + attack_ms + attacked_ms

    for bench_name in bench_attacks:
        best_th = thread_best[bench_name]["best_threads"]
        cls = next(r["classification"] for r in classification_rows if r["attack"] == bench_name)
        best_b = batching_best[bench_name]["best_batch"] if cls.startswith(("A_", "B_")) else 1
        torch.set_num_threads(best_th)
        run_e2e_once(bench_name, e2e_specs[0], seed_offset=4999)  # warm-up, discarded (avoids first-call cold-path distortion)
        e2e_times = []
        total_times = []
        for i, spec in enumerate(e2e_specs):
            attack_ms, total_ms = run_e2e_once(bench_name, spec, seed_offset=5000 + i)
            e2e_times.append(attack_ms)
            total_times.append(total_ms)
        e2e_rows.append({
            "attack": bench_name, "config_batch_size": best_b, "config_threads": best_th,
            "config_note": "batch=1 single-sample E2E call (safe config batching applies to offline throughput, not per-event E2E)",
            **{f"attack_generation_{k}": v for k, v in pct_stats(e2e_times).items()},
            **{f"total_{k}": v for k, v in pct_stats(total_times).items()},
        })
        log(f"  {bench_name}: E2E attack_gen_p95={pct_stats(e2e_times)['p95']} total_p95={pct_stats(total_times)['p95']}")
    torch.set_num_threads(default_threads)

    # ---- bottleneck + summary ----
    log("=== building summary tables ===")
    for bench_name in bench_attacks:
        b_stats = baseline_stats[bench_name]
        cls_row = next(r for r in classification_rows if r["attack"] == bench_name)
        best_th = thread_best[bench_name]["best_threads"]
        opt_stats = batching_best[bench_name]["per_batch"][batching_best[bench_name]["best_batch"]]["stats"] \
            if cls_row["classification"].startswith(("A_", "B_")) else thread_best[bench_name]["stats"]
        median_speedup = (b_stats["median"] / opt_stats["median"]) if (b_stats["median"] and opt_stats.get("median")) else None
        p95_speedup = (b_stats["p95"] / opt_stats["p95"]) if (b_stats["p95"] and opt_stats.get("p95")) else None
        steps = configured_steps(bench_name)
        bottleneck_rows.append({
            "attack": bench_name, "tier": tier_by_attack[bench_name],
            "configured_iteration_or_query_count": steps,
            "n_samples_used": len(slow_fixtures) if tier_by_attack[bench_name] == "slow" else len(fast_fixtures),
            "baseline_median_ms": b_stats["median"], "baseline_p95_ms": b_stats["p95"],
            "optimized_median_ms": opt_stats.get("median"), "optimized_p95_ms": opt_stats.get("p95"),
            "median_speedup": median_speedup, "p95_speedup": p95_speedup,
            "best_threads": best_th, "best_batch_size": batching_best[bench_name]["best_batch"],
            "batching_classification": cls_row["classification"],
        })

    def write_csv(path: Path, rows: List[dict]):
        if not rows:
            path.write_text("")
            return
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv(out_dir / "attack_acceleration_raw.csv", raw_rows)
    write_csv(out_dir / "attack_correctness_summary.csv", correctness_rows)
    write_csv(out_dir / "attack_batching_classification.csv", classification_rows)
    write_csv(out_dir / "attack_thread_tuning.csv", thread_rows)
    write_csv(out_dir / "attack_bottleneck_summary.csv", bottleneck_rows)
    write_csv(out_dir / "attack_e2e_summary.csv", e2e_rows)

    summary_rows = []
    for bench_name in bench_attacks:
        b = next(r for r in bottleneck_rows if r["attack"] == bench_name)
        summary_rows.append(b)
    write_csv(out_dir / "attack_acceleration_summary.csv", summary_rows)

    total_runtime_min = (now_ns() - run_start) / NS_PER_MS / 1000.0 / 60.0
    manifest = {
        "round": "all_attack_acceleration_benchmark",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "dataset_path": DATASET_PATH,
        "checkpoint_path": CHECKPOINT_PATH,
        "registry_actual_17": actual_registry,
        "bench_attacks": bench_attacks,
        "tier_by_attack": tier_by_attack,
        "fast_tier_n": len(fast_fixtures), "slow_tier_n": len(slow_fixtures),
        "thread_tuning_n": len(tt_fixtures), "pilot_n": len(pilot_fixtures),
        "batch_sizes_requested": batch_sizes, "thread_settings": thread_settings,
        "default_torch_threads": default_threads, "eps": EPS, "seed": SEED,
        "cw_c": CW_C, "cw_steps": CW_STEPS, "cw_lr": CW_LR,
        "n_error": n_error, "n_unexpected_fallback": n_unexpected_fallback,
        "n_raw_rows": len(raw_rows), "total_runtime_min": total_runtime_min,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    log(f"DONE: {len(bench_attacks)} attacks, n_error={n_error}, n_unexpected_fallback={n_unexpected_fallback}, "
        f"total_runtime={total_runtime_min:.1f} min")
    log(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
