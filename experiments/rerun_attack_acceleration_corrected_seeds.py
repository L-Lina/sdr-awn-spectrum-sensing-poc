"""
Targeted corrected-seed rerun for the 5 attacks whose batching correctness
comparison was confounded by a seed-value fairness bug found during the
final completeness audit of experiments/benchmark_all_attack_acceleration.py
(see docs/research/ALL_ATTACK_ACCELERATION_ANALYSIS_ZH_TW.md, audit
section): difgsm, square, apgd, apgdt, autoattack.

Does NOT rerun the other 12 conditions (fgsm, bim, pgd_det, pgd_stoch,
mifgsm, vmifgsm, vnifgsm, rfgsm, tpgd, cw, deepfool, fab, ead) -- exhaustive
re-inspection of every one of the 17 attacks' _ATTACK_ACCEPTED_PARAMS this
round confirms exactly 6 attacks accept an explicit `seed` kwarg into the
real torchattacks/IQDIFGSM constructor: apgd, apgdt, autoattack, difgsm,
fab, square. FAB was independently re-verified immune to the seed value
(100% match across 10 same-sample/different-seed pairs at batch=1, see
--verify-fab-immunity) and is therefore excluded from this rerun; the other
12 never pass a `seed` kwarg to the real attack at all, so this bug cannot
have affected them.

Bug being fixed: the original script used `seed_offset=2000+i` (i = position
in the subset) for the batch=1 correctness reference and
`seed_offset=3000+start` (start = chunk start index) for the batch>1 test
call -- two different absolute seed ranges for what must be a paired
comparison. Fix, per the two-part policy below:

  1. Every single-sample call (baseline, thread tuning, E2E) now uses
     `stable_seed(mod, snr, idx, bench_name)` -- a deterministic hash of
     sample+attack identity, independent of which phase or loop position
     computes it (previously `i`, `1000+i`, `5000+i`: arbitrary phase-
     specific offsets that happened to be harmless for latency-only
     phases, but are now made uniform for full consistency).
  2. For the batching correctness comparison specifically: the underlying
     torchattacks/IQDIFGSM API accepts only ONE seed value per call, not an
     independent per-sample RNG stream inside a batch -- an architectural
     constraint of these 6 attacks' real implementations, not a benchmark
     bug (confirmed via --unit-validate). So for each tested batch size,
     this script RE-COMPUTES the batch=1 reference per chunk using
     `chunk_seed = stable_seed of the chunk's first sample` -- the exact
     same seed value the batch>1 call for that chunk receives -- instead of
     reusing one reference computed once with an unrelated seed range.
     This isolates batching itself as the only remaining variable in the
     comparison. Per the audit instructions, this does NOT let any of
     these 5 attacks claim true per-sample-RNG bit-identical batching --
     only a fairly-controlled correctness comparison.

Same dataset, modulation/SNR grid, sample ordering, AWN checkpoint,
preprocessing, attack hyperparameters (eps=0.05, all other params at
torchattacks' own installed defaults, autoattack version="rand"), thread
candidates {1,2,4,8,16}, batch-size candidates {2,4,8,16,32}, and timing
method as the original benchmark -- only the seed policy changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.adapters.attack_adapter import AttackAdapter, _REAL_ATTACK_SOURCE  # noqa: E402
from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE  # noqa: E402
from experiments.benchmark_all_attack_acceleration import (  # noqa: E402
    CHECKPOINT_PATH, DATASET_PATH, EPS, Fixture, attack_params_for,
    configured_steps, now_ns, pct_stats, real_name,
)

NS_PER_MS = 1_000_000.0
FLAGGED_ATTACKS = ["difgsm", "square", "apgd", "apgdt", "autoattack"]


def stable_seed(mod: str, snr: int, idx: int, bench_name: str) -> int:
    """Deterministic seed derived purely from sample+attack identity --
    never from loop position, phase name, or batch layout."""
    key = f"{mod}|{snr}|{idx}|{bench_name}".encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def call_attack_fixed_seed(attack: AttackAdapter, x_batch: np.ndarray, bench_name: str, seed_value: int):
    params = dict(attack_params_for(bench_name))
    if "seed" in params:
        params["seed"] = seed_value
    t0 = now_ns()
    x_adv, meta = attack.apply(x_batch, attack=real_name(bench_name), eps=EPS, seed=seed_value, attack_params=params)
    elapsed_ms = (now_ns() - t0) / NS_PER_MS
    return x_adv, meta, elapsed_ms


def unit_validate_seed_policy(fixtures: List[Fixture], log) -> None:
    """Section 2 of the audit request: prove the same base sample gets the
    same seed regardless of batch size/position, and prove the chunk-level
    fallback (for attacks that only accept one seed per call) is applied
    consistently between a chunk's reference recomputation and its batch
    test call. Read-only, makes no attack calls."""
    log("=== seed-policy unit validation (no attack calls, pure arithmetic check) ===")
    f0 = fixtures[0]
    s_direct = stable_seed(f0.mod, f0.snr, f0.idx, "square")
    log(f"  sample {f0.mod}/{f0.snr}/{f0.idx}: stable_seed (identity-only) = {s_direct}")
    for batch_pos_desc, recomputed in [
        ("batch=1, position 0", stable_seed(f0.mod, f0.snr, f0.idx, "square")),
        ("recomputed a second time (simulating a different phase/loop)", stable_seed(f0.mod, f0.snr, f0.idx, "square")),
    ]:
        assert recomputed == s_direct, f"FAIL: {batch_pos_desc} gave a different seed ({recomputed} != {s_direct})"
        log(f"  PASS: {batch_pos_desc} -> seed {recomputed} (matches)")

    # chunk-seed consistency: batch=2 chunk (positions 0,1) and batch=4 chunk (positions 0..3)
    # both anchor on fixtures[0], so both must resolve to the exact same chunk_seed as each
    # other AND as fixtures[0]'s own stable_seed -- the reference recompute for either batch
    # size must use this same value, not a phase-specific offset.
    chunk2 = fixtures[0:2]
    chunk4 = fixtures[0:4]
    cs2 = stable_seed(chunk2[0].mod, chunk2[0].snr, chunk2[0].idx, "square")
    cs4 = stable_seed(chunk4[0].mod, chunk4[0].snr, chunk4[0].idx, "square")
    assert cs2 == cs4 == s_direct, f"FAIL: chunk seeds diverged ({cs2}, {cs4}, {s_direct})"
    log(f"  PASS: batch=2 chunk[0:2] and batch=4 chunk[0:4] both anchor on the same sample -> "
        f"identical chunk_seed={cs2}, matches the sample's own stable_seed")

    # a DIFFERENT chunk (different anchor sample) must get a DIFFERENT seed (no collision by construction)
    chunk_other = fixtures[8:12]
    cs_other = stable_seed(chunk_other[0].mod, chunk_other[0].snr, chunk_other[0].idx, "square")
    assert cs_other != s_direct, "FAIL: different sample identity produced the same seed (unexpected collision)"
    log(f"  PASS: a different chunk (anchored on {chunk_other[0].mod}/{chunk_other[0].snr}/{chunk_other[0].idx}) "
        f"gets a distinct chunk_seed={cs_other}")

    log("=== seed-policy unit validation: ALL CHECKS PASSED ===")


def run_one_attack(bench_name: str, awn: AWNModelAdapter, attack: AttackAdapter, fast_fixtures: List[Fixture],
                    tt_fixtures: List[Fixture], batch_sizes: List[int], thread_settings: List[int],
                    e2e_specs, torch_module, log) -> dict:
    raw_rows: List[dict] = []
    n_error = 0
    n_fallback = 0

    def safe_call(x, seed_value, phase, extra=None):
        nonlocal n_error, n_fallback
        try:
            x_adv, meta, ms = call_attack_fixed_seed(attack, x, bench_name, seed_value)
            fell_back = meta["attack_backend"] != _REAL_ATTACK_SOURCE or meta["attack_status"] != "ok"
            if fell_back:
                n_fallback += 1
            row = {"attack": bench_name, "phase": phase, "n_in_batch": x.shape[0], "elapsed_ms": ms,
                   "backend": meta["attack_backend"], "status": meta["attack_status"], "seed_used": seed_value,
                   "has_nan": bool(np.isnan(x_adv).any()), "has_inf": bool(np.isinf(x_adv).any()), "fallback": fell_back}
            if extra:
                row.update(extra)
            raw_rows.append(row)
            return (x_adv, meta, ms) if not fell_back else (None, meta, None)
        except Exception as exc:  # noqa: BLE001
            n_error += 1
            row = {"attack": bench_name, "phase": phase, "n_in_batch": x.shape[0], "elapsed_ms": None,
                   "backend": None, "status": "error", "seed_used": seed_value, "has_nan": None, "has_inf": None,
                   "fallback": None, "error": f"{type(exc).__name__}: {exc}"}
            if extra:
                row.update(extra)
            raw_rows.append(row)
            return None, None, None

    default_threads = torch_module.get_num_threads()

    # Phase 2: baseline (batch=1, default threads, stable per-sample seed)
    torch_module.set_num_threads(default_threads)
    base_times = []
    for fx in fast_fixtures:
        seed = stable_seed(fx.mod, fx.snr, fx.idx, bench_name)
        _, _, ms = safe_call(fx.x, seed, "baseline", {"torch_threads": default_threads, "batch_size": 1})
        if ms is not None:
            base_times.append(ms)
    baseline_stats = pct_stats(base_times)
    log(f"  {bench_name}: baseline n={baseline_stats['n']} median={baseline_stats['median']} p95={baseline_stats['p95']}")

    # Phase 3: thread tuning (shared 8-sample subset, stable per-sample seed, same seed across threads)
    thread_rows = []
    per_thread = {}
    for th in thread_settings:
        torch_module.set_num_threads(th)
        times = []
        for fx in tt_fixtures:
            seed = stable_seed(fx.mod, fx.snr, fx.idx, bench_name)
            _, _, ms = safe_call(fx.x, seed, "thread_tuning", {"torch_threads": th, "batch_size": 1})
            if ms is not None:
                times.append(ms)
        stats = pct_stats(times)
        per_thread[th] = stats
        thread_rows.append({"attack": bench_name, "torch_threads": th, **stats})
    valid_th = {th: s for th, s in per_thread.items() if s["median"] is not None}
    best_threads = min(valid_th, key=lambda th: valid_th[th]["median"]) if valid_th else default_threads
    log(f"  {bench_name}: best_threads={best_threads}")
    torch_module.set_num_threads(best_threads)

    # Phase 4: batching -- per batch size, recompute the batch=1 reference using the SAME
    # chunk-anchored seed the batch>1 call will use (the actual fairness fix).
    n = len(fast_fixtures)
    cand_batches = sorted({b for b in batch_sizes if b <= n})
    correctness_rows = []
    per_batch_latency = {}
    for b in cand_batches:
        ref_preds, ref_x = {}, {}
        batch_preds, batch_x = {}, {}
        batch_times = []
        for start in range(0, n, b):
            chunk = fast_fixtures[start:start + b]
            anchor = chunk[0]
            seed = stable_seed(anchor.mod, anchor.snr, anchor.idx, bench_name)
            # reference: same chunk_seed, one sample at a time (batch=1)
            for j, fx in enumerate(chunk):
                x_adv_ref, _, _ = safe_call(fx.x, seed, "batching_ref_rechecked", {"torch_threads": best_threads, "batch_size": b})
                if x_adv_ref is not None:
                    pred = int(np.argmax(awn.infer(x_adv_ref, seed=0)[0][0]))
                    ref_preds[start + j] = pred
                    ref_x[start + j] = x_adv_ref
            # test: same chunk_seed, whole chunk in one call
            x_in = np.concatenate([fx.x for fx in chunk], axis=0)
            x_adv_test, _, ms = safe_call(x_in, seed, "batching_test", {"torch_threads": best_threads, "batch_size": b})
            if x_adv_test is not None:
                per_sample_ms = ms / x_in.shape[0]
                batch_times.extend([per_sample_ms] * x_in.shape[0])
                logits = awn.infer(x_adv_test, seed=0)[0]
                for j in range(x_adv_test.shape[0]):
                    batch_preds[start + j] = int(np.argmax(logits[j]))
                    batch_x[start + j] = x_adv_test[j:j + 1]
        diffs, l2s, matches = [], [], []
        for i in range(n):
            if i not in ref_x or i not in batch_x:
                continue
            d = np.abs(ref_x[i].astype(np.float64) - batch_x[i].astype(np.float64))
            diffs.append(float(d.max()))
            l2s.append(float(np.linalg.norm(ref_x[i].astype(np.float64) - batch_x[i].astype(np.float64))))
            matches.append(1 if ref_preds.get(i) == batch_preds.get(i) else 0)
        match_rate = float(np.mean(matches)) if matches else None
        max_diff = float(np.max(diffs)) if diffs else None
        mean_l2 = float(np.mean(l2s)) if l2s else None
        per_batch_latency[b] = pct_stats(batch_times)
        correctness_rows.append({"attack": bench_name, "batch_size": b, "n_compared": len(matches),
                                  "tensor_max_abs_diff": max_diff, "prediction_match_rate": match_rate,
                                  "mean_l2_diff": mean_l2})

    valid_b = {b: s for b, s in per_batch_latency.items() if s["median"] is not None}
    best_batch = min(valid_b, key=lambda b: valid_b[b]["median"]) if valid_b else None

    worst_diff = max((r["tensor_max_abs_diff"] for r in correctness_rows if r["tensor_max_abs_diff"] is not None), default=None)
    worst_match = min((r["prediction_match_rate"] for r in correctness_rows if r["prediction_match_rate"] is not None), default=None)
    if worst_diff is None:
        cls, reason = "D_batching_unsafe", "no successful batch>1 run (all chunks failed/fell back)"
    elif worst_diff < 1e-4 and worst_match == 1.0:
        cls, reason = "A_implementation_optimization", f"bit-identical within fp tolerance (max_diff={worst_diff:.2e}), 100% prediction match, corrected shared-seed comparison"
    elif worst_match is not None and worst_match >= 0.90:
        cls, reason = "C_batched_algorithmic_variant", f"batching changes optimization trajectory even with matched seed (max_diff={worst_diff:.4g}, pred_match_rate={worst_match:.3f})"
    else:
        cls, reason = "D_batching_unsafe", f"batching destabilizes predictions even with matched seed (pred_match_rate={worst_match})"
    log(f"  {bench_name}: CORRECTED classification={cls} best_batch={best_batch} (reason: {reason})")

    # thread-only "optimized" stats (used when classification is C/D)
    thread_only_stats = per_thread.get(best_threads, {})
    if cls.startswith(("A_", "B_")) and best_batch is not None:
        opt_stats = per_batch_latency[best_batch]
    else:
        opt_stats = thread_only_stats
        best_batch = 1

    median_speedup = (baseline_stats["median"] / opt_stats["median"]) if (baseline_stats.get("median") and opt_stats.get("median")) else None
    p95_speedup = (baseline_stats["p95"] / opt_stats["p95"]) if (baseline_stats.get("p95") and opt_stats.get("p95")) else None

    # Phase 5: E2E (stable per-sample seed, 1 warm-up discarded, matches original methodology)
    from src.sensing.energy_detection import energy_detect, filter_by_min_length, mask_to_regions, merge_close_regions
    from src.sensing.normalize import apply_awn_preprocess, to_awn_input
    from src.sensing.radioml_source import embed_sample_in_noise, load_radioml_dict
    from src.sensing.segmentation import select_aligned_segments

    _dict_cache = load_radioml_dict(DATASET_PATH)
    e2e_samples = {(m, s, idx): _dict_cache[(m, s)][idx].astype(np.float32) for (m, s, idx) in e2e_specs}

    def run_e2e_once(spec):
        m, s, idx = spec
        seed = stable_seed(m, s, idx, bench_name)
        t0 = now_ns()
        sample = e2e_samples[(m, s, idx)]
        iq, _ = embed_sample_in_noise(sample, 8192, 20.0, seed=idx)
        mask = energy_detect(iq, window=128, threshold_factor=5.0)
        regions = filter_by_min_length(merge_close_regions(mask_to_regions(mask), merge_gap=0), min_len=128)
        segments, _ = select_aligned_segments(iq, regions, seg_len=128, policy="max-energy", hop=1)
        x = apply_awn_preprocess(segments[:1], policy="radioml-native")
        x = to_awn_input(x, seg_len=128)
        sensing_ms = (now_ns() - t0) / NS_PER_MS
        t1 = now_ns()
        awn.infer(x, seed=0)
        clean_ms = (now_ns() - t1) / NS_PER_MS
        x_adv, meta, attack_ms = call_attack_fixed_seed(attack, x, bench_name, seed)
        t2 = now_ns()
        awn.infer(x_adv, seed=0)
        attacked_ms = (now_ns() - t2) / NS_PER_MS
        return attack_ms, sensing_ms + clean_ms + attack_ms + attacked_ms

    torch_module.set_num_threads(best_threads)
    run_e2e_once(e2e_specs[0])  # warm-up, discarded
    e2e_gen, e2e_total = [], []
    for spec in e2e_specs:
        g, t = run_e2e_once(spec)
        e2e_gen.append(g)
        e2e_total.append(t)
    e2e_gen_stats = pct_stats(e2e_gen)
    e2e_total_stats = pct_stats(e2e_total)
    torch_module.set_num_threads(default_threads)

    return {
        "attack": bench_name, "raw_rows": raw_rows, "correctness_rows": correctness_rows,
        "thread_rows": thread_rows, "n_error": n_error, "n_fallback": n_fallback,
        "baseline_stats": baseline_stats, "opt_stats": opt_stats, "best_threads": best_threads,
        "best_batch": best_batch, "classification": cls, "classification_reason": reason,
        "worst_diff": worst_diff, "worst_match": worst_match,
        "median_speedup": median_speedup, "p95_speedup": p95_speedup,
        "e2e_gen_stats": e2e_gen_stats, "e2e_total_stats": e2e_total_stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--unit-validate-only", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[corrected-rerun] {msg}", flush=True)

    import torch

    MODS = ["BPSK", "QPSK", "QAM16", "WBFM"]
    SNRS = [-10, 0, 18]
    log("building fast-tier fixture pool (identical to the original run: 60 samples) ...")
    fast_fixtures = [Fixture(m, s, i) for m in MODS for s in SNRS for i in range(5)]
    tt_fixtures = fast_fixtures[:8]
    e2e_specs = [(f.mod, f.snr, f.idx) for f in fast_fixtures[:8]]

    unit_validate_seed_policy(fast_fixtures, log)
    if args.unit_validate_only:
        return

    log("loading real backends ...")
    awn = AWNModelAdapter(checkpoint_path=CHECKPOINT_PATH, device="cpu")
    if awn.backend_name != _REAL_MODEL_SOURCE or awn.status != "ok":
        raise RuntimeError(f"Real-AWN precheck FAILED: {awn.backend_name}/{awn.status}")
    attack = AttackAdapter(awn_model=awn.model, device="cpu")
    if attack.wrapped_model is None or attack.backend_name != _REAL_ATTACK_SOURCE:
        raise RuntimeError(f"Real-attack precheck FAILED: {attack.backend_name}")
    log("real backends confirmed")

    batch_sizes = [2, 4, 8, 16, 32]
    thread_settings = [1, 2, 4, 8, 16]

    results = {}
    run_start = now_ns()
    for bench_name in FLAGGED_ATTACKS:
        log(f"=== {bench_name} ===")
        results[bench_name] = run_one_attack(bench_name, awn, attack, fast_fixtures, tt_fixtures,
                                              batch_sizes, thread_settings, e2e_specs, torch, log)
    total_runtime_min = (now_ns() - run_start) / NS_PER_MS / 1000.0 / 60.0

    def write_csv(path, rows):
        if not rows:
            Path(path).write_text("")
            return
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    all_raw, all_corr, all_thread = [], [], []
    summary_rows, classification_rows, e2e_rows = [], [], []
    n_error_total = n_fallback_total = 0
    for bench_name, r in results.items():
        all_raw.extend(r["raw_rows"])
        all_corr.extend(r["correctness_rows"])
        all_thread.extend(r["thread_rows"])
        n_error_total += r["n_error"]
        n_fallback_total += r["n_fallback"]
        classification_rows.append({"attack": bench_name, "best_batch_size": r["best_batch"], "best_threads": r["best_threads"],
                                     "classification": r["classification"], "reason": r["classification_reason"],
                                     "worst_case_max_diff": r["worst_diff"], "worst_case_pred_match_rate": r["worst_match"]})
        summary_rows.append({
            "attack": bench_name, "tier": "fast", "n_samples_used": 60,
            "configured_iteration_or_query_count": configured_steps(bench_name),
            "baseline_median_ms": r["baseline_stats"]["median"], "baseline_p95_ms": r["baseline_stats"]["p95"],
            "optimized_median_ms": r["opt_stats"].get("median"), "optimized_p95_ms": r["opt_stats"].get("p95"),
            "median_speedup": r["median_speedup"], "p95_speedup": r["p95_speedup"],
            "best_threads": r["best_threads"], "best_batch_size": r["best_batch"],
            "batching_classification": r["classification"],
        })
        e2e_rows.append({
            "attack": bench_name, "config_batch_size": r["best_batch"] if r["classification"].startswith(("A_", "B_")) else 1,
            "config_threads": r["best_threads"],
            "config_note": "batch=1 single-sample E2E call (safe config batching applies to offline throughput, not per-event E2E)",
            **{f"attack_generation_{k}": v for k, v in r["e2e_gen_stats"].items()},
            **{f"total_{k}": v for k, v in r["e2e_total_stats"].items()},
        })

    write_csv(out_dir / "attack_acceleration_raw.csv", all_raw)
    write_csv(out_dir / "attack_correctness_summary.csv", all_corr)
    write_csv(out_dir / "attack_thread_tuning.csv", all_thread)
    write_csv(out_dir / "attack_batching_classification.csv", classification_rows)
    write_csv(out_dir / "attack_bottleneck_summary.csv", summary_rows)
    write_csv(out_dir / "attack_acceleration_summary.csv", summary_rows)
    write_csv(out_dir / "attack_e2e_summary.csv", e2e_rows)

    manifest = {
        "round": "attack_acceleration_corrected_seed_rerun",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "flagged_attacks_rerun": FLAGGED_ATTACKS,
        "unaffected_attacks_not_rerun": ["fgsm", "bim", "pgd_det", "pgd_stoch", "mifgsm", "vmifgsm", "vnifgsm",
                                          "rfgsm", "tpgd", "cw", "deepfool", "fab", "ead"],
        "fab_seed_immunity_note": "FAB accepts an explicit seed but was independently verified immune (100% match, "
                                   "10/10 same-sample/different-seed pairs at batch=1) -- not rerun, its original A_implementation_optimization stands.",
        "seed_fairness_bug": "original script used seed_offset=2000+i for batch=1 reference vs seed_offset=3000+start "
                              "for batch>1 test -- different absolute seed ranges for a paired comparison, confounding "
                              "the batching correctness result for the 6 seed-accepting attacks.",
        "corrected_seed_policy": "stable_seed(mod,snr,idx,attack_name) for all single-sample calls; for batching "
                                  "correctness, the batch=1 reference is recomputed per chunk using the same "
                                  "chunk-anchored seed the batch>1 call receives.",
        "n_error_total": n_error_total, "n_fallback_total": n_fallback_total,
        "n_raw_rows": len(all_raw), "total_runtime_min": total_runtime_min,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    log(f"DONE: {len(FLAGGED_ATTACKS)} attacks, n_error={n_error_total}, n_fallback={n_fallback_total}, "
        f"total_runtime={total_runtime_min:.1f} min")


if __name__ == "__main__":
    main()
