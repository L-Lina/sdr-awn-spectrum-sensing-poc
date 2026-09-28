"""
Phase 1D-0b: thermally-controlled PAIRED replication of B=1 thread behaviour (Raspberry Pi 5, CPU only).

Why this exists
---------------
Phase 1D-0 (results/accel_phase1d0_thermal_20260927_092156) found, with temperature and run
order controlled, B=1 single-call medians of
    PGD  intra1 44.92 ms (clean)   intra4 41.55 ms (clean)
    BIM  intra1 45.30 ms (clean)   intra4 42.97 ms (THERMAL_REVIEW: pre-cell reading 56.2 C)
i.e. the opposite thread ranking from the thermally confounded Phase 1C run. Each figure
comes from ONE cell. This benchmark replicates only the B=1 comparison, several times, in
adjacent intra1/intra4 pairs, so that:
  * the PGD intra4 result can be checked for replication, and
  * the BIM B=1 comparison can be decided from THERMAL_CLEAN cells only.
It optimizes nothing and attributes nothing (no backend / NEON / dispatcher / autograd claim).

Scope
-----
attacks PGD eps=0.03, BIM eps=0.03; B=1 only; intra=1/interop=1 vs intra=4/interop=1;
the validated 88-sample subset; Phase 1B optimized path; ControlledStartPGD; Phase 1C v2
correctness rule. Each cell = the exact Phase 1D-0 cell body (warmup >= 2 untimed batches,
then `repeats` passes x 88 samples x {sequential, batched} -- identical computations at B=1).

The measurement worker is NOT re-implemented: the persistent worker of
experiments/bench_attack_thermal_control_pi5.py (Phase 1D-0, validated on the Pi) is
launched unchanged (`--worker` mode) and driven over the same JSON-line pipe protocol.
Thermal helpers (snapshot, in-cell sampler, THERMAL_CLEAN/REVIEW classifier incl. its 1.0 C
start-temperature sensor allowance) are imported from the same file. Nothing is modified.

Order (deterministic, no RNG)
-----------------------------
One replication = two blocks of four cells:
    even replication:  PGD i1 i4 i4 i1 | BIM i4 i1 i1 i4
    odd  replication:  BIM i4 i1 i1 i4 | PGD i1 i4 i4 i1   (exact mirror of the even one)
With an even number of replications every (attack, thread) has the same mean run
position. Pairs are the adjacent cells (1st,2nd) and (3rd,4th) of each block, so every pair
has one intra1 and one intra4 cell run back-to-back, alternately in either order.

Thermal policy (explicit; nothing is silently accepted)
-------------------------------------------------------
release   temp <= --max-start-temp-c (55) on --release-consecutive-reads (2) consecutive
          polls (every --cooldown-poll-seconds, 5 s) AND get_throttled bits 0-3 clear.
gate      right before the cell a fresh snapshot must satisfy the THERMAL_CLEAN start rule
          of Phase 1D-0 (temp <= 55 + 1.0 C sensor allowance, bits 0-3 clear). If not, the
          cell is NOT started; the parent cools again (a "start rejection", recorded).
          More than --max-start-rejections for one attempt -> the planned cell is SKIPPED.
retry     a cell that ran but is THERMAL_REVIEW (current-state bits before/during/after,
          newly set history bits, governor change, ...) is kept in the raw data with
          accepted=False and re-run after a fresh cooldown, up to --max-cell-attempts in
          total; if no attempt is clean the planned cell is REVIEW_UNRESOLVED.
timeout   a cooldown that exceeds --max-cooldown-seconds skips the planned cell
          (--on-cooldown-timeout abort to stop instead); --max-consecutive-skips aborts.
Only accepted (THERMAL_CLEAN + correctness PASS) attempts enter the statistics. A thread
winner is named for an attack ONLY if every planned cell of that attack has an accepted
attempt. vcgencmd must be available (the clean rule depends on it) or the run refuses to
start. get_throttled history bits already set at start are recorded; because they are
sticky, a reboot before the run makes the "newly set history bit" check fully informative.
The governor, clocks, voltage, fan and OS settings are only read, never changed.

Correctness: SEMANTIC_DIFFERENT fails the cell (no retry); INVALID aborts the run (the
worker exits). PGD and BIM at B=1 are expected to be bit-identical to their anchors.

Run on the Pi (from the repo root):
    .venv/bin/python experiments/bench_attack_thermal_pair_pi5.py --dry-run
    .venv/bin/python experiments/bench_attack_thermal_pair_pi5.py
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import bench_attack_accel_pi5 as p1a  # noqa: E402 -- Phase 1A helpers (no torch import at module level)
import bench_attack_overhead_pi5 as p1b  # noqa: E402 -- hashed only
import bench_attack_batch_pi5_v2 as p1c  # noqa: E402 -- Phase 1C v2 constants / stat helpers
import bench_attack_thermal_control_pi5 as p1d0  # noqa: E402 -- Phase 1D-0 worker + thermal helpers

REPO_ROOT = p1a.REPO_ROOT
ATTACKS = ["pgd", "bim"]
THREADS = (1, 4)
BATCH_SIZE = 1
INTEROP_THREADS = 1
CLASSES = p1c.CLASSES
BUDGET_MS = p1c.BUDGET_MS

# (attack, intra_threads) in run order; the odd replication is the exact mirror.
LAYOUT_EVEN = [("pgd", 1), ("pgd", 4), ("pgd", 4), ("pgd", 1), ("bim", 4), ("bim", 1), ("bim", 1), ("bim", 4)]
LAYOUT_ODD = list(reversed(LAYOUT_EVEN))

# Phase 1D-0 B=1 cell results, used only as the replication reference in validation.json.
PHASE1D0_DIR = "results/accel_phase1d0_thermal_20260927_092156"
PHASE1D0_B1_MEDIAN_MS = {("pgd", 1): 44.92, ("pgd", 4): 41.55, ("bim", 1): 45.30, ("bim", 4): 42.97}
PHASE1D0_B1_THERMAL = {("pgd", 1): "THERMAL_CLEAN", ("pgd", 4): "THERMAL_CLEAN", ("bim", 1): "THERMAL_CLEAN",
                       ("bim", 4): "THERMAL_REVIEW"}

# Planning numbers from Phase 1D-0 (timed B=1 cells 22.3-24.3 s, worker setup ~22 s,
# cooldown waits 35-105 s). Estimate only.
EST_CELL_S = 24.5
EST_WORKER_SETUP_S = 25.0

TRACE_FIELDS = ["phase", "planned_cell_id", "attempt_id", "order_index", "t_s", "temp_c",
                "cpu0_scaling_cur_freq_khz", "throttled_raw", "throttled_low", "arm_clock_hz"]
EXTRA_RAW = ["planned_cell_id", "attempt", "accepted", "pair_id", "replication"]


# --------------------------------------------------------------------------- plan
def build_plan(replications: int) -> List[dict]:
    plan = []
    for rep in range(replications):
        layout = LAYOUT_EVEN if rep % 2 == 0 else LAYOUT_ODD
        for j, (attack, t) in enumerate(layout):
            block_pos = j % 4
            plan.append({"replication": rep, "attack": attack, "intra_threads": t, "interop_threads": INTEROP_THREADS,
                         "batch_size": BATCH_SIZE, "condition": f"intra{t}_interop{INTEROP_THREADS}",
                         "pair_id": f"r{rep}_{attack}_p{block_pos // 2}", "pair_role": "first" if block_pos % 2 == 0 else "second"})
    for i, c in enumerate(plan):
        c["planned_index"] = i
        c["planned_cell_id"] = f"{i:02d}_{c['attack']}_B1_{c['condition']}"
    return plan


def plan_balance(plan: List[dict]) -> dict:
    out = {}
    for a in ATTACKS:
        for t in THREADS:
            pos = [c["planned_index"] for c in plan if c["attack"] == a and c["intra_threads"] == t]
            out[f"{a}/intra{t}"] = {"positions": pos, "mean_position": float(np.mean(pos)) if pos else None}
    return out


def estimate(plan: List[dict], args) -> dict:
    n = len(plan)
    active = n * EST_CELL_S + len(THREADS) * EST_WORKER_SETUP_S
    cool = n * args.expected_cooldown_seconds
    worst_single = n * (args.max_cooldown_seconds + EST_CELL_S)
    worst_retry = n * args.max_cell_attempts * (args.max_start_rejections + 1) * (args.max_cooldown_seconds + EST_CELL_S)
    return {
        "n_planned_cells": n,
        "active_compute_minutes": active / 60,
        "cooldown_expected_minutes": cool / 60,
        "cooldown_expected_basis": f"{n} waits x --expected-cooldown-seconds {args.expected_cooldown_seconds} "
                                   "(Phase 1D-0 observed 35-105 s per wait; planning assumption)",
        "wall_expected_minutes": (active + cool) / 60,
        "wall_worst_case_no_retries_minutes": worst_single / 60,
        "wall_theoretical_max_all_retries_minutes": worst_retry / 60,
        "note": "a hopeless cooling setup is cut short by --max-consecutive-skips",
    }


# --------------------------------------------------------------------------- thermal gate
def wait_until_cool(args, trace: List[dict], planned_cell_id: str) -> dict:
    """Idle-poll (read-only) until `release_consecutive_reads` consecutive temps <= threshold and bits 0-3 clear."""
    t0 = time.monotonic()
    first = p1a.cpu_temp_c()
    streak, n, last_log = 0, 0, t0
    while True:
        temp = p1a.cpu_temp_c()
        n += 1
        trace.append({"phase": "cooldown", "planned_cell_id": planned_cell_id, "attempt_id": "", "order_index": "",
                      "t_s": round(time.monotonic() - t0, 3), "temp_c": temp,
                      "cpu0_scaling_cur_freq_khz": p1d0.read_cur_freq_khz(), "throttled_raw": "",
                      "throttled_low": "", "arm_clock_hz": ""})
        if temp is None:
            return {"outcome": "TEMPERATURE_UNAVAILABLE", "waited_s": time.monotonic() - t0, "n_polls": n,
                    "temp_first_c": first, "temp_last_c": None, "throttled_at_release": None}
        streak = streak + 1 if temp <= args.max_start_temp_c else 0
        if streak >= args.release_consecutive_reads:
            th = p1d0.read_throttled()
            trace[-1].update(throttled_raw=th["raw"], throttled_low=th["low"])
            if th["low"] == 0:
                return {"outcome": "REACHED", "waited_s": time.monotonic() - t0, "n_polls": n,
                        "temp_first_c": first, "temp_last_c": temp, "throttled_at_release": th}
            streak = 0
        waited = time.monotonic() - t0
        if waited >= args.max_cooldown_seconds:
            return {"outcome": "TIMEOUT", "waited_s": waited, "n_polls": n, "temp_first_c": first,
                    "temp_last_c": temp, "throttled_at_release": p1d0.read_throttled()}
        if time.monotonic() - last_log >= 60:
            p1a.log(f"[cooldown] {waited:5.0f} s  temp={temp:.2f} C (target <= {args.max_start_temp_c} C)")
            last_log = time.monotonic()
        time.sleep(args.cooldown_poll_seconds)


def start_gate(before: dict, max_temp_c: float) -> List[str]:
    """Same start rule as the Phase 1D-0 THERMAL_CLEAN classifier; failing it means 'do not start'."""
    reasons = []
    t = before["temp_c"]
    if t is None or t > max_temp_c + p1d0.START_TEMP_JITTER_C:
        reasons.append(f"pre-cell temperature {t} C > {max_temp_c} + {p1d0.START_TEMP_JITTER_C} C allowance")
    low = before["throttled"]["low"]
    if low is None:
        reasons.append(f"get_throttled unavailable ({before['throttled']['raw']!r})")
    elif low:
        reasons.append(f"current-state throttle bits set: {before['throttled']['value_hex']} "
                       f"{before['throttled']['low_names']}")
    return reasons


# --------------------------------------------------------------------------- statistics
def _stats(vals: List[float]) -> dict:
    st = p1a.stats(vals)
    return {k: st[k] for k in ("n", "median", "mean", "p95", "p99", "std", "min", "max")}


def sign_test_two_sided(k: int, n: int) -> Optional[float]:
    """Exact two-sided binomial sign test, H0 p=0.5 (ties must be excluded by the caller)."""
    if n == 0:
        return None
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def aggregate(out_dir: Path, plan: List[dict], attempts: List[dict], labels: List[str]) -> tuple:
    by_attempt = {a["attempt_id"]: a for a in attempts}
    rows, samples = [], []
    for lab in labels:
        with (out_dir / f"raw_{lab}.csv").open(newline="") as f:
            rows.extend(csv.DictReader(f))
        with (out_dir / f"per_sample_{lab}.csv").open(newline="") as f:
            samples.extend(csv.DictReader(f))
    for r in rows + samples:
        a = by_attempt[r["cell_id"]]
        r.update(planned_cell_id=a["planned_cell_id"], attempt=a["attempt"], accepted=a["accepted"],
                 pair_id=a["pair_id"], replication=a["replication"])
    rows.sort(key=lambda r: (int(r["order_index"]), int(r["repeat"]), int(r["batch_index"]), int(r["mode_order"])))
    with (out_dir / "raw_measurements.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXTRA_RAW + p1d0.RAW_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with (out_dir / "per_sample_validation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXTRA_RAW + p1d0.SAMPLE_FIELDS)
        w.writeheader()
        w.writerows(samples)

    # ---- per attempt
    summary = []
    for a in attempts:
        cr = [r for r in rows if r["cell_id"] == a["attempt_id"]]
        seq = [r for r in cr if r["mode"] == "sequential"]
        bat = [r for r in cr if r["mode"] == "batched"]
        cs = [s for s in samples if s["cell_id"] == a["attempt_id"]]
        cnt = {m: {cl: sum(1 for s in cs if s["mode"] == m and s["classification"] == cl) for cl in CLASSES}
               for m in ("sequential", "batched")}
        n_sd = cnt["sequential"]["SEMANTIC_DIFFERENT"] + cnt["batched"]["SEMANTIC_DIFFERENT"]
        n_inv = cnt["sequential"]["INVALID"] + cnt["batched"]["INVALID"]
        status = "PASS" if cs and n_sd == 0 and n_inv == 0 else "FAIL"
        a["validation_status"] = status
        calls = [float(v) for r in seq for v in r["seq_call_ms_list"].split(";") if v]
        a["_calls"] = calls
        a["_batched"] = [float(r["total_ms"]) for r in bat]
        a["_kernel"] = [float(r["kernel_ms"]) for r in seq]
        th = a["thermal"]
        nonbit = sorted({f"{s['modulation']}/{s['snr_db']}/{s['sample_index']}" for s in cs
                         if s["classification"] != "BIT_IDENTICAL"})
        rec = {
            "attempt_id": a["attempt_id"], "planned_cell_id": a["planned_cell_id"], "order_index": a["order_index"],
            "replication": a["replication"], "pair_id": a["pair_id"], "attempt": a["attempt"],
            "attack": a["attack"], "condition": a["condition"], "intra_threads": a["intra_threads"],
            "interop_threads": INTEROP_THREADS, "batch_size": BATCH_SIZE,
            "accepted": a["accepted"], "validation_status": status, "thermal_status": th["status"],
            "thermal_reasons": " | ".join(th["reasons"]), "start_rejections_before_attempt": a["start_rejections"],
            "cooldown_waited_s": a["cooldown"]["waited_s"],
            "temp_before_c": th["before"]["temp_c"], "temp_after_c": th["after"]["temp_c"],
            "temp_max_in_cell_c": th["temp_max_in_cell_c"],
            "throttled_before": th["before"]["throttled"]["value_hex"],
            "throttled_after": th["after"]["throttled"]["value_hex"],
            "throttled_low_before": th["before"]["throttled"]["low"], "throttled_low_after": th["after"]["throttled"]["low"],
            "throttled_high_before": th["before"]["throttled"]["high"],
            "throttled_high_after": th["after"]["throttled"]["high"],
            "throttled_low_or_in_cell": th["throttled_low_or_in_cell"], "n_in_cell_vc_samples": th["n_in_cell_vc_samples"],
            "arm_clock_hz_before": th["before"]["arm_clock_hz"], "arm_clock_hz_after": th["after"]["arm_clock_hz"],
            "arm_clock_hz_min_in_cell": th["arm_clock_hz_min_in_cell"],
            "arm_clock_hz_max_in_cell": th["arm_clock_hz_max_in_cell"],
            "governor_unchanged": th["before"]["governor"] == th["after"]["governor"],
            "n_bit_identical_seq": cnt["sequential"]["BIT_IDENTICAL"], "n_bit_identical_batched": cnt["batched"]["BIT_IDENTICAL"],
            "n_bitwise_different_semantic_same": cnt["sequential"]["BITWISE_DIFFERENT_SEMANTIC_SAME"]
                                                 + cnt["batched"]["BITWISE_DIFFERENT_SEMANTIC_SAME"],
            "n_semantic_different": n_sd, "n_invalid": n_inv, "non_bit_identical_samples": ";".join(nonbit),
            **p1c._stat_cols("b1_single_call_ms", calls),
            **p1c._stat_cols("b1_batched_mode_ms", a["_batched"]),
            **p1c._stat_cols("b1_kernel_ms", a["_kernel"]),
            "flag_C_b1_single_call_median_le_40ms": (float(np.median(calls)) <= BUDGET_MS) if calls else "",
        }
        summary.append(rec)
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # ---- accepted attempt per planned cell
    accepted = {}
    for a in attempts:
        if a["accepted"] and a["validation_status"] == "PASS":
            accepted[a["planned_cell_id"]] = a
    planned_status = {c["planned_cell_id"]: c["final_status"] for c in plan}

    # ---- per attack x thread (accepted attempts only)
    cond_rows, cond = [], {}
    for attack in ATTACKS:
        for t in THREADS:
            acc = [accepted[c["planned_cell_id"]] for c in plan
                   if c["attack"] == attack and c["intra_threads"] == t and c["planned_cell_id"] in accepted]
            calls = [v for a in acc for v in a["_calls"]]
            ref = PHASE1D0_B1_MEDIAN_MS[(attack, t)]
            st = _stats(calls)
            cell_meds = [float(np.median(a["_calls"])) for a in acc]
            entry = {
                "attack": attack, "condition": f"intra{t}_interop{INTEROP_THREADS}", "intra_threads": t,
                "n_planned_cells": sum(1 for c in plan if c["attack"] == attack and c["intra_threads"] == t),
                "n_accepted_cells": len(acc), **{f"b1_single_call_ms_{k}": v for k, v in st.items()},
                "b1_batched_mode_ms_median": float(np.median([v for a in acc for v in a["_batched"]])) if acc else None,
                "b1_kernel_ms_median": float(np.median([v for a in acc for v in a["_kernel"]])) if acc else None,
                "per_cell_medians_ms": ";".join(f"{m:.3f}" for m in cell_meds),
                "per_cell_median_min_ms": min(cell_meds) if cell_meds else None,
                "per_cell_median_max_ms": max(cell_meds) if cell_meds else None,
                "phase1d0_reference_median_ms": ref, "phase1d0_reference_thermal": PHASE1D0_B1_THERMAL[(attack, t)],
                "delta_vs_phase1d0_ms": (st["median"] - ref) if acc else None,
                "delta_vs_phase1d0_pct": (100.0 * (st["median"] - ref) / ref) if acc else None,
                "phase1d0_reference_within_per_cell_range": (min(cell_meds) <= ref <= max(cell_meds)) if cell_meds else None,
                "gap_to_40ms_ms": (st["median"] - BUDGET_MS) if acc else None,
                "gap_to_40ms_pct_reduction_needed": (100.0 * (st["median"] - BUDGET_MS) / st["median"]) if acc else None,
            }
            cond[(attack, t)] = entry
            cond_rows.append(entry)
    with (out_dir / "condition_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cond_rows[0].keys()))
        w.writeheader()
        w.writerows(cond_rows)

    # ---- pairs (adjacent intra1/intra4 cells)
    pair_rows = []
    for pid in dict.fromkeys(c["pair_id"] for c in plan):
        members = [c for c in plan if c["pair_id"] == pid]
        c1 = next(c for c in members if c["intra_threads"] == 1)
        c4 = next(c for c in members if c["intra_threads"] == 4)
        a1, a4 = accepted.get(c1["planned_cell_id"]), accepted.get(c4["planned_cell_id"])
        row = {"pair_id": pid, "attack": c1["attack"], "replication": c1["replication"],
               "first_in_pair": "intra1" if c1["pair_role"] == "first" else "intra4",
               "intra1_planned_cell": c1["planned_cell_id"], "intra4_planned_cell": c4["planned_cell_id"],
               "pair_valid": bool(a1 and a4)}
        if a1 and a4:
            m1, m4 = float(np.median(a1["_calls"])), float(np.median(a4["_calls"]))
            row.update(intra1_median_ms=m1, intra4_median_ms=m4, delta_ms_intra4_minus_intra1=m4 - m1,
                       delta_pct=100.0 * (m4 - m1) / m1, ratio_intra4_over_intra1=m4 / m1)
        else:
            row.update(not_valid_reason=f"intra1 {planned_status[c1['planned_cell_id']]}, "
                                        f"intra4 {planned_status[c4['planned_cell_id']]}")
        pair_rows.append(row)
    pf = ["pair_id", "attack", "replication", "first_in_pair", "intra1_planned_cell", "intra4_planned_cell",
          "pair_valid", "intra1_median_ms", "intra4_median_ms", "delta_ms_intra4_minus_intra1", "delta_pct",
          "ratio_intra4_over_intra1", "not_valid_reason"]
    with (out_dir / "paired_comparisons.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pf, restval="")
        w.writeheader()
        w.writerows(pair_rows)

    # ---- per-attack comparison + claim gate
    comparisons = {}
    for attack in ATTACKS:
        planned = [c for c in plan if c["attack"] == attack]
        not_ok = [f"{c['planned_cell_id']}: {c['final_status']}" for c in planned if c["planned_cell_id"] not in accepted]
        claim = not not_ok
        prs = [p for p in pair_rows if p["attack"] == attack and p["pair_valid"]]
        deltas = [p["delta_ms_intra4_minus_intra1"] for p in prs]
        ratios = [p["ratio_intra4_over_intra1"] for p in prs]
        nz = [d for d in deltas if d != 0]
        k_i4_faster = sum(1 for d in nz if d < 0)
        e1, e4 = cond[(attack, 1)], cond[(attack, 4)]
        comp = {
            "claim_allowed": claim, "not_claimable_reasons": not_ok,
            "pooled_median_ms": {"intra1": e1["b1_single_call_ms_median"], "intra4": e4["b1_single_call_ms_median"]},
            "pooled_delta_ms_intra4_minus_intra1": (e4["b1_single_call_ms_median"] - e1["b1_single_call_ms_median"])
            if e1["n_accepted_cells"] and e4["n_accepted_cells"] else None,
            "n_valid_pairs": len(prs),
            "paired_delta_ms": _stats(deltas) if deltas else None,
            "paired_delta_pct_median": float(np.median([p["delta_pct"] for p in prs])) if prs else None,
            "paired_ratio_median": float(np.median(ratios)) if ratios else None,
            "paired_ratio_min_max": [min(ratios), max(ratios)] if ratios else None,
            "n_pairs_intra4_faster": k_i4_faster, "n_pairs_intra1_faster": len(nz) - k_i4_faster,
            "n_pairs_tied": len(deltas) - len(nz),
            "sign_test_two_sided_p": sign_test_two_sided(k_i4_faster, len(nz)),
        }
        if claim and comp["pooled_delta_ms_intra4_minus_intra1"] is not None and deltas:
            pooled_sign = np.sign(comp["pooled_delta_ms_intra4_minus_intra1"])
            paired_sign = np.sign(float(np.median(deltas)))
            comp["directions_agree"] = bool(pooled_sign == paired_sign and pooled_sign != 0)
            if comp["directions_agree"]:
                comp["faster_condition"] = "intra4" if pooled_sign < 0 else "intra1"
        comparisons[attack] = comp

    counts_all = {cl: sum(1 for s in samples if s["classification"] == cl) for cl in CLASSES}
    n_fail = sum(1 for a in attempts if a["validation_status"] != "PASS")
    finals = [c["final_status"] for c in plan]
    overall = "PASS" if n_fail == 0 and all(f == "ACCEPTED" for f in finals) else (
        "FAIL" if n_fail else "INCOMPLETE")
    validation = {
        "rule": "Phase 1C v2 (bitwise AND semantic); SEMANTIC_DIFFERENT fails the attempt; INVALID aborts the run",
        "per_attempt": {a["attempt_id"]: {"status": a["validation_status"], "thermal_status": a["thermal"]["status"],
                                          "accepted": a["accepted"]} for a in attempts},
        "planned_cell_final_status": {c["planned_cell_id"]: c["final_status"] for c in plan},
        "classification_totals_all_attempts": counts_all,
        "n_planned_cells": len(plan), "n_attempts": len(attempts),
        "n_attempts_thermal_review": sum(1 for a in attempts if a["thermal"]["status"] != "THERMAL_CLEAN"),
        "n_start_rejections": sum(c["start_rejections_total"] for c in plan),
        "n_attempts_fail": n_fail,
        "per_condition": {f"{k[0]}/intra{k[1]}": v for k, v in cond.items()},
        "thread_comparisons_b1": comparisons,
        "overall": overall,
        "notes": [
            "Statistics use only accepted attempts (THERMAL_CLEAN and correctness PASS); rejected attempts stay in "
            "raw_measurements.csv / summary.csv with accepted=False.",
            "B=1 single-call = one sequential-mode call (Phase 1B optimized path, clean prediction reused). "
            "b1_batched_mode is the same computation issued through the batched code path.",
            "A faster condition is named only if every planned cell of that attack has an accepted attempt and the "
            "pooled-median and median-paired-delta directions agree.",
            "Phase 1D-0 references are single cells; BIM intra4 there was THERMAL_REVIEW.",
            "No backend, NEON, CMSIS-NN-like, dispatcher or autograd cause is inferred from these timings.",
        ],
    }
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2, default=str))
    return summary, validation, cond_rows, len(rows)


# --------------------------------------------------------------------------- parent
def parent_main(args) -> None:
    plan = build_plan(args.replications)
    est = estimate(plan, args)
    bal = plan_balance(plan)
    p1a.log("[parent] planned order:")
    for c in plan:
        p1a.log(f"   {c['planned_index']:2d}  {c['attack']:4s} {c['condition']}  pair={c['pair_id']}")
    p1a.log(f"[parent] mean positions: { {k: v['mean_position'] for k, v in bal.items()} }")
    p1a.log(f"[parent] estimate: active {est['active_compute_minutes']:.1f} min, cooldown ~"
            f"{est['cooldown_expected_minutes']:.0f} min (assumption), wall ~{est['wall_expected_minutes']:.0f} min; "
            f"worst case without retries {est['wall_worst_case_no_retries_minutes']:.0f} min")
    now = p1d0.snapshot()
    p1a.log(f"[parent] now: temp={now['temp_c']} C throttled={now['throttled']['value_hex']} "
            f"low={now['throttled']['low_names']} history={now['throttled']['high_names']} "
            f"arm_clock={now['arm_clock_hz']} governor={set(now['governor'].values())}")
    if args.replications % 2:
        p1a.log("[warn] odd --replications: mean run positions are not exactly balanced")
    if args.dry_run:
        print(json.dumps({"plan": plan, "balance": bal, "estimate": est, "system_now": now}, indent=2, default=str))
        return
    if now["throttled"]["low"] is None:
        p1a.abort("vcgencmd get_throttled unavailable; the THERMAL_CLEAN rule cannot be evaluated")
    if now["throttled"]["high"]:
        p1a.log(f"[warn] sticky history bits already set at start ({now['throttled']['high_names']}); a newly "
                "occurring event of the same kind cannot be seen in the history bits -- current-state bits are "
                "still checked before/during/after every cell. A reboot before the run avoids this.")

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (REPO_ROOT / (args.out_dir or f"results/accel_phase1d0b_pair_{ts}")).resolve()
    for p in p1a.PROTECTED_RESULT_DIRS + ("results/accel_phase1c_v2_formal_20260926_164547",
                                          "results/accel_phase1a_formal_20260926_152737",
                                          "results/accel_phase1b_20260926_155342", PHASE1D0_DIR):
        prot = (REPO_ROOT / p).resolve()
        if out_dir == prot or prot in out_dir.parents or out_dir in prot.parents:
            p1a.abort(f"refusing to write inside/over results directory {p}")
    out_dir.mkdir(parents=True, exist_ok=False)

    ctx = p1a.load_formal_context(args)
    cfg, manifest, formal_dir = ctx["cfg"], ctx["manifest"], ctx["formal_dir"]
    subset = p1a.build_subset(cfg)
    _, attack_rows = p1a.load_formal_rows(formal_dir, set(subset))
    attack_settings = p1a.resolve_attack_settings(cfg, attack_rows, args.attack_temperature, args.diagnostics)
    attack_settings = {a: attack_settings[a] for a in ATTACKS}
    sensing = p1a.resolve_sensing(cfg)

    p1a.log("[parent] hashing dataset / checkpoint / config ...")
    hashes = {
        "dataset_sha256": p1a.sha256_file(REPO_ROOT / cfg["dataset_path"]),
        "checkpoint_sha256": p1a.sha256_file(REPO_ROOT / cfg["checkpoint_path"]),
        "config_sha256": p1a.sha256_file(ctx["cfg_path"]),
        "formal_manifest_sha256": p1a.sha256_file(formal_dir / "manifest.json"),
        "attack_adapter_py_sha256": p1a.sha256_file(REPO_ROOT / "src/adapters/attack_adapter.py"),
        "adv_attack_py_sha256": p1a.sha256_file(REPO_ROOT / "external/adversarial-rf/util/adv_attack.py"),
        "phase1a_helpers_sha256": p1a.sha256_file(Path(p1a.__file__).resolve()),
        "phase1b_helpers_sha256": p1a.sha256_file(Path(p1b.__file__).resolve()),
        "phase1c_v2_script_sha256": p1a.sha256_file(Path(p1c.__file__).resolve()),
        "phase1d0_worker_script_sha256": p1a.sha256_file(Path(p1d0.__file__).resolve()),
        "this_script_sha256": p1a.sha256_file(Path(__file__).resolve()),
    }
    if hashes["dataset_sha256"] != manifest.get("dataset_sha256"):
        p1a.abort("dataset sha256 differs from formal CPU manifest")
    if hashes["checkpoint_sha256"] != manifest.get("checkpoint_sha256"):
        p1a.abort("checkpoint sha256 differs from formal CPU manifest")
    env_meta = p1a.environment_metadata()

    worker_ctx = {"subset": [list(k) for k in subset], "formal_dir": p1a.rel(formal_dir),
                  "dataset_path": cfg["dataset_path"], "checkpoint_path": cfg["checkpoint_path"],
                  "sensing": sensing, "attack_settings": attack_settings, "attacks": ATTACKS}
    ctx_file = out_dir / "worker_context.json"
    ctx_file.write_text(json.dumps(worker_ctx, indent=2, default=str))

    attempts: List[dict] = []
    trace: List[dict] = []
    manifest_out = {
        "phase": "accel_phase1d0b_thermal_pair", "status": "running",
        "question": "Under controlled thermal state, is PGD/BIM B=1 single-call latency lower with intra=4 than with "
                    "intra=1 (interop=1), and does the Phase 1D-0 PGD intra4 result (41.55 ms) replicate?",
        "command": [sys.executable] + sys.argv, "out_dir": p1a.rel(out_dir), "formal_cpu_dir": p1a.rel(formal_dir),
        "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "hashes": hashes, "environment": env_meta, "system_at_start": now, "attacks": attack_settings,
        "threads": list(THREADS), "interop_threads": INTEROP_THREADS, "batch_size": BATCH_SIZE,
        "replications": args.replications, "repeats": args.repeats, "warmup_batches_per_cell": args.warmup_batches,
        "measurement_worker": "experiments/bench_attack_thermal_control_pi5.py --worker (Phase 1D-0, unchanged)",
        "order": {"layout_even": LAYOUT_EVEN, "layout_odd": LAYOUT_ODD, "balance": bal,
                  "pairs": "adjacent (1st,2nd) and (3rd,4th) cells of each 4-cell block"},
        "thermal_policy": {
            "max_start_temp_c": args.max_start_temp_c, "start_temp_sensor_allowance_c": p1d0.START_TEMP_JITTER_C,
            "release_consecutive_reads": args.release_consecutive_reads,
            "cooldown_poll_seconds": args.cooldown_poll_seconds, "max_cooldown_seconds": args.max_cooldown_seconds,
            "max_start_rejections": args.max_start_rejections, "max_cell_attempts": args.max_cell_attempts,
            "on_cooldown_timeout": args.on_cooldown_timeout, "max_consecutive_skips": args.max_consecutive_skips,
            "in_cell_temp_poll_seconds": args.in_cell_temp_poll_seconds,
            "in_cell_vc_poll_seconds": args.in_cell_vc_poll_seconds,
            "thermal_clean_rule": "Phase 1D-0 classify_thermal: current-state bits 0-3 zero before/during(sampled)/"
                                  "after, no history bit newly set, governor unchanged, start temp <= threshold + "
                                  "allowance, cooldown REACHED",
            "system_settings_changed": "none (read-only)",
        },
        "estimate": est, "plan": plan, "attempts": attempts,
        "phase1d0_reference": {"dir": PHASE1D0_DIR,
                               "b1_median_ms": {f"{k[0]}/intra{k[1]}": v for k, v in PHASE1D0_B1_MEDIAN_MS.items()},
                               "thermal": {f"{k[0]}/intra{k[1]}": v for k, v in PHASE1D0_B1_THERMAL.items()}},
    }
    man_path = out_dir / "manifest.json"
    p1d0._write_json(man_path, manifest_out)

    workers: Dict[str, subprocess.Popen] = {}

    def write_trace():
        with (out_dir / "thermal_trace.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRACE_FIELDS, restval="")
            w.writeheader()
            w.writerows(trace)

    def fail(msg: str):
        for proc in workers.values():
            if proc.poll() is None:
                proc.kill()
        inv = [json.loads(p.read_text()) for p in out_dir.glob("invalid_*.json")]
        p1d0._write_json(out_dir / "validation.json", {"overall": f"FAIL ({msg})", "invalid": inv or None,
                                                       "note": "aborted run; see worker_*.stderr.log"})
        manifest_out.update(status="aborted", abort_reason=msg, finished_utc=_dt.datetime.now(_dt.timezone.utc).isoformat())
        p1d0._write_json(man_path, manifest_out)
        write_trace()
        p1a.abort(msg)

    def recv(label: str) -> dict:
        line = workers[label].stdout.readline()
        if not line:
            fail(f"worker {label} exited (rc={workers[label].wait()}) -- INVALID output, failed fail-closed check or crash")
        return json.loads(line)

    worker_script = Path(p1d0.__file__).resolve()
    for t in THREADS:
        label = f"intra{t}_interop{INTEROP_THREADS}"
        cmd = [sys.executable, str(worker_script), "--worker", "--out-dir", str(out_dir), "--worker-context",
               str(ctx_file), "--worker-threads", str(t), "--repeats", str(args.repeats),
               "--warmup-batches", str(args.warmup_batches)]
        p1a.log(f"[parent] starting persistent worker {label}")
        errf = (out_dir / f"worker_{label}.stderr.log").open("w")
        workers[label] = subprocess.Popen(cmd, cwd=REPO_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=errf, text=True, bufsize=1, env=os.environ.copy())
        t0 = time.monotonic()
        msg = recv(label)
        if msg.get("event") != "ready":
            fail(f"unexpected message from {label}: {msg}")
        manifest_out.setdefault("workers", {})[label] = {**msg, "setup_seconds": time.monotonic() - t0}
        p1d0._write_json(man_path, manifest_out)

    exec_index = 0
    consecutive_skips = 0
    for c in plan:
        c["start_rejections_total"] = 0
        c["final_status"] = "NOT_RUN"
        pid = c["planned_cell_id"]
        for attempt in range(1, args.max_cell_attempts + 1):
            # ---- cool, then gate; a failed gate means "do not start", cool again
            rejections, cd, before = 0, None, None
            while True:
                p1a.log(f"[parent] {pid} attempt {attempt}: cooling to <= {args.max_start_temp_c} C "
                        f"(now {p1a.cpu_temp_c()} C)")
                cd = wait_until_cool(args, trace, pid)
                if cd["outcome"] != "REACHED":
                    break
                before = p1d0.snapshot()
                reasons = start_gate(before, args.max_start_temp_c)
                if not reasons:
                    break
                rejections += 1
                c["start_rejections_total"] += 1
                c.setdefault("start_rejection_log", []).append({"attempt": attempt, "reasons": reasons,
                                                                "temp_c": before["temp_c"], "utc": before["utc"]})
                p1a.log(f"[parent] {pid}: start rejected ({reasons}); cooling again")
                if rejections > args.max_start_rejections:
                    cd = {**cd, "outcome": "START_GATE_EXHAUSTED"}
                    break
            if cd["outcome"] != "REACHED":
                if args.on_cooldown_timeout == "abort" or cd["outcome"] == "TEMPERATURE_UNAVAILABLE":
                    c["final_status"] = f"ABORTED_{cd['outcome']}"
                    fail(f"{pid}: {cd['outcome']} (last temp {cd.get('temp_last_c')} C)")
                c["final_status"] = f"SKIPPED_{cd['outcome']}"
                break
            # ---- run the attempt
            attempt_id = f"{pid}_a{attempt}"
            label = c["condition"]
            sampler = p1d0.CellSampler(attempt_id, exec_index, args.in_cell_temp_poll_seconds,
                                       args.in_cell_vc_poll_seconds).start()
            workers[label].stdin.write(json.dumps({"cmd": "run", "cell": {
                "cell_id": attempt_id, "order_index": exec_index, "attack": c["attack"], "batch_size": BATCH_SIZE}}) + "\n")
            workers[label].stdin.flush()
            msg = recv(label)
            samples = sampler.stop()
            after = p1d0.snapshot()
            if msg.get("event") != "cell_done" or msg.get("cell_id") != attempt_id:
                fail(f"unexpected reply for {attempt_id}: {msg}")
            for s in samples:
                trace.append({"phase": "cell", "planned_cell_id": pid, "attempt_id": s["cell_id"],
                              "order_index": s["order_index"], **{k: s[k] for k in TRACE_FIELDS[4:]}})
            status, treasons = p1d0.classify_thermal(before, after, samples, cd, args.max_start_temp_c)
            temps = [s["temp_c"] for s in samples if s["temp_c"] is not None] + \
                    [v for v in (before["temp_c"], after["temp_c"]) if v is not None]
            clocks = [int(s["arm_clock_hz"]) for s in samples if s["arm_clock_hz"] not in ("", None)]
            lows = [int(s["throttled_low"]) for s in samples if s["throttled_low"] not in ("", None)]
            low_or = 0
            for v in lows + [x for x in (before["throttled"]["low"], after["throttled"]["low"]) if x is not None]:
                low_or |= v
            cc = msg["class_counts"]
            sem_bad = any(cc[m]["SEMANTIC_DIFFERENT"] for m in cc)
            a = {"attempt_id": attempt_id, "planned_cell_id": pid, "attempt": attempt, "order_index": exec_index,
                 "replication": c["replication"], "pair_id": c["pair_id"], "attack": c["attack"],
                 "condition": label, "intra_threads": c["intra_threads"], "start_rejections": rejections,
                 "cooldown": cd, "class_counts": cc, "warmup_s": msg["warmup_s"], "timed_s": msg["timed_s"],
                 "thermal": {"status": status, "reasons": treasons, "before": before, "after": after,
                             "temp_max_in_cell_c": max(temps) if temps else None,
                             "n_in_cell_samples": len(samples), "n_in_cell_vc_samples": len(lows),
                             "throttled_low_or_in_cell": low_or,
                             "arm_clock_hz_min_in_cell": min(clocks) if clocks else None,
                             "arm_clock_hz_max_in_cell": max(clocks) if clocks else None},
                 "accepted": status == "THERMAL_CLEAN" and not sem_bad}
            attempts.append(a)
            exec_index += 1
            p1a.log(f"[parent] {attempt_id} done; temp {before['temp_c']} -> {after['temp_c']} C "
                    f"(max {a['thermal']['temp_max_in_cell_c']}); throttled {before['throttled']['value_hex']} -> "
                    f"{after['throttled']['value_hex']}; {status}{' ' + str(treasons) if treasons else ''}"
                    f"{'; SEMANTIC_DIFFERENT' if sem_bad else ''}")
            p1d0._write_json(man_path, manifest_out)
            write_trace()
            if sem_bad:
                c["final_status"] = "FAIL_SEMANTIC_DIFFERENT"
                break
            if a["accepted"]:
                c["final_status"] = "ACCEPTED"
                break
            c["final_status"] = "REVIEW_UNRESOLVED"  # overwritten if a later attempt is clean
        if c["final_status"].startswith("SKIPPED"):
            consecutive_skips += 1
            p1a.log(f"[warn] {pid} {c['final_status']}")
            if consecutive_skips >= args.max_consecutive_skips:
                fail(f"{consecutive_skips} consecutive planned cells skipped; the thermal threshold is not practical "
                     "on this cooling setup")
        else:
            consecutive_skips = 0
        p1d0._write_json(man_path, manifest_out)

    for label, proc in workers.items():
        proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
        proc.stdin.flush()
        msg = recv(label)
        if msg.get("event") != "bye":
            fail(f"unexpected shutdown message from {label}: {msg}")
        proc.wait(timeout=120)
        manifest_out["workers"][label]["returncode"] = proc.returncode
        wj = out_dir / f"worker_{label}.json"
        if wj.exists():
            manifest_out["workers"][label]["worker_meta"] = json.loads(wj.read_text())

    summary, validation, cond_rows, n = aggregate(out_dir, plan, attempts, list(workers))
    for a in attempts:  # drop in-memory timing lists before persisting the manifest
        for k in ("_calls", "_batched", "_kernel"):
            a.pop(k, None)
    manifest_out.update(status="complete", raw_rows=n, validation_overall=validation["overall"],
                        finished_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(), environment_end=p1d0.snapshot())
    p1d0._write_json(man_path, manifest_out)
    write_trace()
    p1a.log(f"[parent] done: {out_dir}  validation={validation['overall']}")
    for e in cond_rows:
        p1a.log(f"  {e['attack']:4s} {e['condition']}  accepted {e['n_accepted_cells']}/{e['n_planned_cells']}  "
                f"B1 median={e['b1_single_call_ms_median']}  p95={e['b1_single_call_ms_p95']}  "
                f"cell medians=[{e['per_cell_medians_ms']}]  (1D-0 ref {e['phase1d0_reference_median_ms']})")
    for attack, comp in validation["thread_comparisons_b1"].items():
        p1a.log(f"  {attack}: claim_allowed={comp['claim_allowed']} valid_pairs={comp['n_valid_pairs']} "
                f"i4_faster={comp['n_pairs_intra4_faster']} i1_faster={comp['n_pairs_intra1_faster']} "
                f"sign_p={comp['sign_test_two_sided_p']} paired_ratio_median={comp['paired_ratio_median']} "
                f"faster={comp.get('faster_condition', '-')}")
    if validation["overall"] != "PASS":
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=p1a.DEFAULT_CONFIG)
    ap.add_argument("--formal-dir", default=p1a.DEFAULT_FORMAL_CPU_DIR)
    ap.add_argument("--out-dir", default=None, help="default results/accel_phase1d0b_pair_<timestamp> (must not exist)")
    ap.add_argument("--replications", type=int, default=4, help="8 cells each; even values keep positions balanced")
    ap.add_argument("--repeats", type=int, default=3, help="passes over the 88-sample subset per cell")
    ap.add_argument("--warmup-batches", type=int, default=2, help="unrecorded warmup batches per cell (>= 2)")
    ap.add_argument("--max-start-temp-c", type=float, default=55.0)
    ap.add_argument("--release-consecutive-reads", type=int, default=2)
    ap.add_argument("--cooldown-poll-seconds", type=float, default=5.0)
    ap.add_argument("--max-cooldown-seconds", type=float, default=900.0)
    ap.add_argument("--max-start-rejections", type=int, default=5, help="per attempt, before the planned cell is skipped")
    ap.add_argument("--max-cell-attempts", type=int, default=3, help="total runs of one planned cell if THERMAL_REVIEW")
    ap.add_argument("--on-cooldown-timeout", choices=["skip", "abort"], default="skip")
    ap.add_argument("--max-consecutive-skips", type=int, default=2)
    ap.add_argument("--expected-cooldown-seconds", type=float, default=75.0, help="planning assumption only")
    ap.add_argument("--in-cell-temp-poll-seconds", type=float, default=1.0)
    ap.add_argument("--in-cell-vc-poll-seconds", type=float, default=5.0, help="0 disables in-cell vcgencmd polling")
    ap.add_argument("--attack-temperature", type=float, default=None)
    ap.add_argument("--diagnostics", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--dry-run", action="store_true", help="print plan, estimate and current thermal state; no torch")
    args = ap.parse_args()
    if args.replications < 1 or args.repeats < 1 or args.warmup_batches < 2:
        p1a.abort("--replications>=1, --repeats>=1 and --warmup-batches>=2 required")
    if args.release_consecutive_reads < 1 or args.max_cell_attempts < 1 or args.max_start_rejections < 0:
        p1a.abort("--release-consecutive-reads>=1, --max-cell-attempts>=1, --max-start-rejections>=0 required")
    if args.cooldown_poll_seconds <= 0 or args.in_cell_temp_poll_seconds <= 0 or args.max_cooldown_seconds <= 0:
        p1a.abort("poll intervals and --max-cooldown-seconds must be > 0")
    parent_main(args)


if __name__ == "__main__":
    main()
