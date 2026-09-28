"""
Phase 1D-0: temperature-controlled, order-counterbalanced sanity benchmark (Raspberry Pi 5, CPU only).

Why this exists
---------------
Phase 1C v2 formal (results/accel_phase1c_v2_formal_20260926_164547) ran its thread
conditions in the fixed order intra1 -> intra2 -> intra4 with a 30 s cooldown:
    intra1  58.4 -> 80.4 C   get_throttled 0x0     -> 0x80000
    intra2  69.4 -> 82.6 C   get_throttled 0x80000 -> 0xe0008
    intra4  70.5 -> 84.8 C   get_throttled 0xe0000 -> 0xe0006
so thread count was confounded with elapsed time / thermal state. This benchmark
re-measures a small grid with thermal state and time order controlled. It does NOT
optimize anything.

Scope (defaults)
----------------
attacks      PGD eps=0.03, BIM eps=0.03 (formal profiles)
conditions   intra=1/interop=1 and intra=4/interop=1
batch sizes  1, 8, 32
subset       the validated 88-sample subset (11 mods x SNR {-20,-6,6,18} x idx {0,1})
path         the validated Phase 1B optimized path (clean prediction reused, untimed)
PGD start    ControlledStartPGD (imported unchanged from bench_attack_batch_pi5_v2.py)
correctness  the Phase 1C v2 rule: BIT_IDENTICAL / BITWISE_DIFFERENT_SEMANTIC_SAME are
             valid; SEMANTIC_DIFFERENT fails the cell; INVALID aborts the run
repeats 3, warmup batches 2 (per cell, untimed)

A "cell" = one (attack, thread condition, B) measurement: warmup + `repeats` passes over
the 88 samples, each batch timed in both sequential (B independent B=1 calls) and
batched mode, mode order alternating per batch -- exactly the Phase 1C v2 cell body.

Thermal control
---------------
* One persistent worker process per thread condition (torch intra/inter-op threads are
  process-level settings). Both workers load the model, rebuild inputs, re-run the
  Phase 1C v2 batch-semantics audit and build/validate the B=1 anchors ONCE, then block
  on a pipe (no CPU use) until the parent sends a cell. So no anchor/setup work heats
  the SoC between a cooldown and a cell.
* Before EVERY cell the parent polls the SoC temperature (sysfs, every
  --cooldown-poll-seconds, default 5 s) until temp <= --max-start-temp-c (default 55 C)
  AND the current-state throttle bits (get_throttled bits 0-3) are clear. If that is not
  reached within --max-cooldown-seconds the cell is SKIPPED (or the run aborts with
  --on-cooldown-timeout abort); --max-consecutive-skips aborts a hopeless run.
* Nothing is changed on the system: governor, clocks, voltage, fan and OS settings are
  only READ.
* Immediately before and after each cell: temperature, `vcgencmd get_throttled`,
  `vcgencmd measure_clock arm`, scaling governors. During the cell a parent-side
  sampler reads the sysfs temperature + cpu0 scaling_cur_freq every
  --in-cell-temp-poll-seconds (default 1 s; a sysfs read, microseconds) and
  get_throttled + measure_clock arm every --in-cell-vc-poll-seconds (default 10 s;
  0 disables; one short vcgencmd subprocess). The perturbation is recorded, not hidden.

get_throttled handling
----------------------
bits 0-3  (current state):  0 under-voltage now, 1 ARM freq capped now, 2 throttled now,
                            3 soft temperature limit active now
bits 16-19 (sticky history): same four events "has occurred since boot"
The history bits stay set after the first event, so a cell is NOT flagged merely because
they are set. A cell is THERMAL_REVIEW if: any current-state bit is non-zero before,
during (sampled) or immediately after it; a history bit becomes newly set across it;
vcgencmd is unavailable; or the governor changed. Otherwise THERMAL_CLEAN.

Order control
-------------
The six (attack, B) pairs are interleaved across attacks and batch sizes; the two thread
conditions of a pair run back-to-back in ABBA order across pairs, which balances the mean
run position of intra1 and intra4 exactly (12 cells: both 5.5). --order-variant reverse
reverses the whole list (for a replication run). No RNG; the order is stored in the
manifest and in every raw row (order_index).

Thread comparisons
------------------
validation.json lists, per (attack, B), intra1 vs intra4 ratios. A faster condition is
named ONLY when both cells are PASS and THERMAL_CLEAN; otherwise claim_allowed = false.

Run on the Pi (from the repo root):
    .venv/bin/python experiments/bench_attack_thermal_control_pi5.py --dry-run
    .venv/bin/python experiments/bench_attack_thermal_control_pi5.py
Reuses bench_attack_accel_pi5.py (1A), bench_attack_overhead_pi5.py (1B) and
bench_attack_batch_pi5_v2.py (1C v2) by import; none of them is modified.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import bench_attack_accel_pi5 as p1a  # noqa: E402 -- Phase 1A helpers (no torch import at module level)
import bench_attack_overhead_pi5 as p1b  # noqa: E402 -- Phase 1B validated attack path
import bench_attack_batch_pi5_v2 as p1c  # noqa: E402 -- Phase 1C v2 rule, ControlledStartPGD, constants

REPO_ROOT = p1a.REPO_ROOT
ATTACKS = ["pgd", "bim"]
DEFAULT_THREADS = "1,4"
DEFAULT_BATCH_SIZES = "1,8,32"
INTEROP_THREADS = 1
CLASSES = p1c.CLASSES
NORM_EPS_TOL, IQ_EPS_REL_TOL, IQ_EPS_ABS_TOL = p1c.NORM_EPS_TOL, p1c.IQ_EPS_REL_TOL, p1c.IQ_EPS_ABS_TOL
BUDGET_MS = p1c.BUDGET_MS

# Explicit counterbalanced (attack, B) pair order for the default scope: attacks alternate,
# each attack sees small/large/mid B at different times.
CANONICAL_PAIR_ORDER = [("pgd", 1), ("bim", 8), ("pgd", 32), ("bim", 1), ("pgd", 8), ("bim", 32)]

# Phase 1C v2 formal intra1 linear fits (completion ms ~ fixed + marginal*B) and B=1 medians.
# Used ONLY for the pre-run runtime estimate, never reported as results.
EST_B1_MS = {"pgd": 46.4, "bim": 48.6}
EST_FIT = {"pgd": (39.6, 13.1), "bim": (41.5, 13.4)}
EST_WORKER_SETUP_S = 75.0

THROTTLE_LOW_BITS = {0: "under_voltage_now", 1: "arm_freq_capped_now", 2: "throttled_now",
                     3: "soft_temp_limit_now"}
THROTTLE_HIGH_BITS = {16: "under_voltage_occurred", 17: "arm_freq_capping_occurred", 18: "throttling_occurred",
                      19: "soft_temp_limit_occurred"}

# The pre-cell snapshot is read a moment after the cooldown release reading; allow for sensor
# jitter before flagging "started too hot" (the release itself is strictly <= threshold).
START_TEMP_JITTER_C = 1.0

RAW_FIELDS = ["cell_id", "order_index"] + p1c.RAW_FIELDS
SAMPLE_FIELDS = ["cell_id", "order_index"] + p1c.SAMPLE_FIELDS
TRACE_FIELDS = ["cell_id", "order_index", "t_s", "temp_c", "cpu0_scaling_cur_freq_khz",
                "throttled_raw", "throttled_low", "arm_clock_hz"]


# --------------------------------------------------------------------------- system readings (read-only)
def _decode_bits(v: Optional[int], table: Dict[int, str]) -> List[str]:
    return [] if v is None else [name for bit, name in table.items() if v & (1 << bit)]


def read_throttled() -> dict:
    raw = p1a.run_cmd(["vcgencmd", "get_throttled"])
    v = p1c._throttled_value(raw)
    return {"raw": raw, "value_hex": None if v is None else hex(v),
            "low": None if v is None else v & 0xF, "high": None if v is None else (v >> 16) & 0xF,
            "low_names": _decode_bits(v, THROTTLE_LOW_BITS), "high_names": _decode_bits(v, THROTTLE_HIGH_BITS)}


def read_arm_clock_hz() -> Optional[int]:
    raw = p1a.run_cmd(["vcgencmd", "measure_clock", "arm"])  # 'frequency(0)=2400000000'
    if not raw or "=" not in raw:
        return None
    try:
        return int(raw.split("=", 1)[1].strip())
    except ValueError:
        return None


def read_cur_freq_khz() -> Optional[int]:
    raw = p1a.read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


def snapshot() -> dict:
    return {"utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "temp_c": p1a.cpu_temp_c(),
            "throttled": read_throttled(), "arm_clock_hz": read_arm_clock_hz(), "governor": p1c._governors()}


class CellSampler:
    """Parent-side sampler; the parent is otherwise blocked on the worker pipe during a cell."""

    def __init__(self, cell_id: str, order_index: int, temp_poll_s: float, vc_poll_s: float):
        self.cell_id, self.order_index = cell_id, order_index
        self.temp_poll_s, self.vc_poll_s = temp_poll_s, vc_poll_s
        self.samples: List[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        t0 = time.monotonic()
        next_vc = t0 + self.vc_poll_s if self.vc_poll_s > 0 else float("inf")
        while not self._stop.is_set():
            now = time.monotonic()
            row = {"cell_id": self.cell_id, "order_index": self.order_index, "t_s": round(now - t0, 3),
                   "temp_c": p1a.cpu_temp_c(), "cpu0_scaling_cur_freq_khz": read_cur_freq_khz(),
                   "throttled_raw": "", "throttled_low": "", "arm_clock_hz": ""}
            if now >= next_vc:
                th = read_throttled()
                row.update(throttled_raw=th["raw"], throttled_low=th["low"], arm_clock_hz=read_arm_clock_hz())
                next_vc = now + self.vc_poll_s
            self.samples.append(row)
            self._stop.wait(self.temp_poll_s)

    def start(self) -> "CellSampler":
        self._thread.start()
        return self

    def stop(self) -> List[dict]:
        self._stop.set()
        self._thread.join(timeout=30)
        return self.samples


def wait_until_cool(max_temp_c: float, max_s: float, poll_s: float) -> dict:
    """Idle-poll until temp <= max_temp_c and current-state throttle bits are clear. Read-only."""
    t0 = time.monotonic()
    first = p1a.cpu_temp_c()
    n = 0
    last_log = t0
    while True:
        temp = p1a.cpu_temp_c()
        n += 1
        if temp is None:
            return {"outcome": "TEMPERATURE_UNAVAILABLE", "waited_s": time.monotonic() - t0, "n_polls": n,
                    "temp_first_c": first, "temp_last_c": None, "throttled_at_release": None}
        if temp <= max_temp_c:
            th = read_throttled()
            if th["low"] in (0, None):  # None: vcgencmd unavailable -> cell flagged later
                return {"outcome": "REACHED", "waited_s": time.monotonic() - t0, "n_polls": n,
                        "temp_first_c": first, "temp_last_c": temp, "throttled_at_release": th}
        waited = time.monotonic() - t0
        if waited >= max_s:
            return {"outcome": "TIMEOUT", "waited_s": waited, "n_polls": n, "temp_first_c": first,
                    "temp_last_c": temp, "throttled_at_release": read_throttled()}
        if time.monotonic() - last_log >= 60:
            p1a.log(f"[cooldown] {waited:5.0f} s  temp={temp:.1f} C (target <= {max_temp_c} C)")
            last_log = time.monotonic()
        time.sleep(poll_s)


def classify_thermal(before: dict, after: dict, samples: List[dict], cooldown: dict,
                     max_temp_c: float) -> tuple:
    reasons = []
    for when, snap in (("before", before), ("after", after)):
        th = snap["throttled"]
        if th["low"] is None:
            reasons.append(f"get_throttled {when} unavailable/unparseable ({th['raw']!r})")
        elif th["low"]:
            reasons.append(f"current-state throttle bits {when} cell: {th['value_hex']} {th['low_names']}")
    lows = [int(s["throttled_low"]) for s in samples if s["throttled_low"] not in ("", None)]
    if any(lows):
        reasons.append(f"current-state throttle bits non-zero during cell in {sum(1 for v in lows if v)}/"
                       f"{len(lows)} in-cell samples")
    hb, ha = before["throttled"]["high"], after["throttled"]["high"]
    if hb is not None and ha is not None and ha & ~hb:
        new = ha & ~hb
        reasons.append("history bits newly set during cell: "
                       + str([THROTTLE_HIGH_BITS[16 + b] for b in range(4) if new & (1 << b)]))
    if before["governor"] != after["governor"]:
        reasons.append("scaling governor changed during the cell")
    if before["temp_c"] is None or before["temp_c"] > max_temp_c + START_TEMP_JITTER_C:
        reasons.append(f"start temperature {before['temp_c']} C > --max-start-temp-c {max_temp_c} "
                       f"(+{START_TEMP_JITTER_C} C sensor jitter allowance)")
    if cooldown.get("outcome") != "REACHED":
        reasons.append(f"cooldown outcome {cooldown.get('outcome')}")
    return ("THERMAL_REVIEW" if reasons else "THERMAL_CLEAN"), reasons


# --------------------------------------------------------------------------- order
def build_cell_order(attacks: List[str], batch_sizes: List[int], threads: List[int], variant: str) -> List[dict]:
    if sorted(attacks) == sorted(ATTACKS) and sorted(batch_sizes) == [1, 8, 32]:
        pairs = list(CANONICAL_PAIR_ORDER)
    else:  # generic: round-robin over attacks, batch-size index rotated per attack
        pairs = []
        for j in range(len(batch_sizes)):
            for i, a in enumerate(attacks):
                pairs.append((a, batch_sizes[(j + i) % len(batch_sizes)]))
    t_a, t_b = threads
    cells = []
    for k, (attack, B) in enumerate(pairs):
        first_a = (k % 4) in (0, 3)  # ABBA ABBA ...
        for t in ((t_a, t_b) if first_a else (t_b, t_a)):
            cells.append({"attack": attack, "batch_size": B, "intra_threads": t, "interop_threads": INTEROP_THREADS,
                          "condition": f"intra{t}_interop{INTEROP_THREADS}"})
    if variant == "reverse":
        cells.reverse()
    for i, c in enumerate(cells):
        c["order_index"] = i
        c["cell_id"] = f"{i:02d}_{c['attack']}_B{c['batch_size']}_{c['condition']}"
    return cells


def order_balance(cells: List[dict]) -> dict:
    out = {}
    for t in sorted({c["intra_threads"] for c in cells}):
        pos = [c["order_index"] for c in cells if c["intra_threads"] == t]
        out[f"intra{t}"] = {"positions": pos, "mean_position": float(np.mean(pos))}
    return out


# --------------------------------------------------------------------------- estimate
def estimate(cells: List[dict], repeats: int, warmup_batches: int, n: int = 88) -> dict:
    """Active compute from Phase 1C v2 intra1 medians/fits (intra4 treated the same). Estimate only."""
    total = 0.0
    per_cell = {}
    for c in cells:
        a, B = c["attack"], c["batch_size"]
        batches = [min(B, n - i) for i in range(0, n, B)]
        comp = [EST_B1_MS[a] if b == 1 else EST_FIT[a][0] + EST_FIT[a][1] * b for b in batches]
        warm = sum(batches[:warmup_batches]) * EST_B1_MS[a] + sum(comp[:warmup_batches])
        timed = repeats * (n * EST_B1_MS[a] + sum(comp))
        s = (warm + timed) / 1e3 * 1.05  # +5% validation/hashing/CSV
        per_cell[c["cell_id"]] = s
        total += s
    return {"active_cell_seconds": total, "worker_setup_seconds": EST_WORKER_SETUP_S * len({c["condition"] for c in cells}),
            "per_cell_seconds": per_cell}


# --------------------------------------------------------------------------- worker
def worker_main(args) -> None:
    # stdout carries the JSON-line protocol only; everything printed (python or C) goes to stderr/log
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(msg: dict) -> None:
        proto.write(json.dumps(msg, default=str) + "\n")
        proto.flush()

    import torch  # thread settings applied before any parallel work

    threads = args.worker_threads
    torch.set_num_interop_threads(INTEROP_THREADS)
    torch.set_num_threads(threads)
    intra, interop = torch.get_num_threads(), torch.get_num_interop_threads()
    if (intra, interop) != (threads, INTEROP_THREADS):
        p1a.abort(f"thread settings not applied: intra={intra} interop={interop}")

    import torch.nn as nn
    import src.adapters.attack_adapter as am
    from src.adapters.attack_adapter import AttackAdapter, _REAL_ATTACK_SOURCE
    from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE

    ctx = json.loads(Path(args.worker_context).read_text())
    out_dir = Path(args.out_dir)
    label = f"intra{threads}_interop{INTEROP_THREADS}"
    attacks: List[str] = ctx["attacks"]
    settings = ctx["attack_settings"]
    p1a.log(f"[{label}] pid={os.getpid()} attacks={attacks}")

    # ---------------------------------------------------------------- setup: copied from Phase 1C v2 worker_main
    awn = AWNModelAdapter(checkpoint_path=str(REPO_ROOT / ctx["checkpoint_path"]), device="cpu")
    if awn.model is None or awn.backend_name != _REAL_MODEL_SOURCE or awn.status != "ok":
        p1a.abort(f"AWN backend is not real: backend={awn.backend_name} status={awn.status}")
    adapter = AttackAdapter(awn_model=awn.model, device="cpu")
    if adapter.wrapped_model is None or adapter.backend_name != _REAL_ATTACK_SOURCE or adapter.status != "ok":
        p1a.abort(f"attack backend is not real: backend={adapter.backend_name} status={adapter.status}")
    if am._torchattacks is None or awn.model.training:
        p1a.abort("torchattacks unavailable or model in train mode")
    wm = adapter.wrapped_model
    ControlledStartPGD, pgd_src_sha = p1c.make_controlled_pgd(am._torchattacks, torch, nn)

    subset = [tuple(k) for k in ctx["subset"]]
    formal_dir = REPO_ROOT / ctx["formal_dir"]
    base_rows, formal_attack = p1a.load_formal_rows(formal_dir, set(subset))
    inputs = p1a.rebuild_clean_inputs(subset, base_rows, ctx["sensing"], REPO_ROOT / ctx["dataset_path"])
    p1a.log(f"[{label}] rebuilt {len(inputs)} x_clean inputs; all sha256 == formal clean_input_sha256")

    def infer_pred(x) -> int:
        logits, meta = awn.infer(x)
        if meta["awn_backend"] != _REAL_MODEL_SOURCE or meta["awn_status"] != "ok" or not np.isfinite(logits).all():
            p1a.abort(f"AWN inference invalid: {meta}")
        return int(np.argmax(logits[0]))

    for key, inp in inputs.items():  # reused clean prediction (Phase 1B rule), untimed
        inp["clean_pred"] = infer_pred(inp["x"])
        if inp["clean_pred"] != inp["formal_clean_pred"]:
            p1a.abort(f"{key}: clean_pred {inp['clean_pred']} != formal {inp['formal_clean_pred']}")
        if p1b.in_path_clean_pred(am, adapter, torch, inp["x"]) != inp["clean_pred"]:
            p1a.abort(f"{key}: apply()-internal y_pred != pipeline clean_pred; reuse would change the label")

    req_grad_ref = [p.requires_grad for p in wm.parameters()]

    def check_state(where: str) -> None:
        if awn.model.training or wm.training:
            p1a.abort(f"model entered train mode ({where})")
        if (torch.get_num_threads(), torch.get_num_interop_threads()) != (intra, interop):
            p1a.abort(f"torch thread settings drifted ({where})")
        if [p.requires_grad for p in wm.parameters()] != req_grad_ref:
            p1a.abort(f"requires_grad flags not restored exactly ({where})")

    audit: Dict[str, object] = {"pgd_forward_source_sha256": pgd_src_sha, "pgd_forward_matches_copied_logic": True}
    adv_attack_path = REPO_ROOT / "external" / "adversarial-rf" / "util" / "adv_attack.py"
    audit["adv_attack_py_sha256"] = p1a.sha256_file(adv_attack_path)
    audit["adv_attack_py_matches_pinned_audited_source"] = audit["adv_attack_py_sha256"] == p1c.PINNED_ADV_ATTACK_SHA256
    X_all = np.concatenate([inputs[k]["x"] for k in subset], axis=0)
    xa_b, a_b, b_b = am._iq_to_ta_input_minmax(torch.from_numpy(X_all))
    per = [am._iq_to_ta_input_minmax(torch.from_numpy(inputs[k]["x"])) for k in subset]
    audit["minmax_a_b_shape"] = [list(a_b.shape), list(b_b.shape)]
    audit["minmax_per_sample_shape_ok"] = tuple(a_b.shape) == (len(subset), 1, 1) and tuple(b_b.shape) == (len(subset), 1, 1)
    audit["minmax_batch_equals_per_sample_bitwise"] = bool(
        all(torch.equal(a_b[i:i + 1], per[i][1]) and torch.equal(b_b[i:i + 1], per[i][2])
            and torch.equal(xa_b[i:i + 1], per[i][0]) for i in range(len(subset))))
    back_b = am._ta_output_to_iq_minmax(xa_b, a_b, b_b)
    audit["denorm_batch_equals_per_sample_bitwise"] = bool(
        all(torch.equal(back_b[i:i + 1], am._ta_output_to_iq_minmax(per[i][0], per[i][1], per[i][2]))
            for i in range(len(subset))))
    audit["distinct_per_sample_b_values"] = int(torch.unique(b_b).numel())
    try:
        wm.set_minmax(a_b, b_b)
        with torch.no_grad():
            lb = wm(xa_b)
        wm.clear_minmax()
        l1 = []
        for i in range(len(subset)):
            wm.set_minmax(per[i][1], per[i][2])
            with torch.no_grad():
                l1.append(wm(per[i][0]))
            wm.clear_minmax()
        l1 = torch.cat(l1)
        audit["wrapper_batch_argmax_equals_per_sample"] = bool(torch.equal(lb.argmax(1), l1.argmax(1)))
        audit["wrapper_batch_logits_max_abs_diff"] = float((lb - l1).abs().max())
        audit["wrapper_batch_logits_bitwise_equal"] = bool(torch.equal(lb, l1))
    finally:
        wm.eval()
        wm.clear_minmax()
    check_state("audit")
    if not (audit["minmax_per_sample_shape_ok"] and audit["minmax_batch_equals_per_sample_bitwise"]
            and audit["denorm_batch_equals_per_sample_bitwise"] and audit["wrapper_batch_argmax_equals_per_sample"]):
        (out_dir / f"audit_{label}.json").write_text(json.dumps(audit, indent=2, default=str))
        p1a.abort(f"batch-semantics audit failed: {audit}")
    if not audit["adv_attack_py_matches_pinned_audited_source"]:
        p1a.log("[warn] adv_attack.py differs from the pinned file the audit was read from; "
                "runtime audit checks above still passed")

    pgd_hp = None
    noise = {}
    if "pgd" in attacks:
        s = settings["pgd"]
        stock = am._build_torchattacks("pgd", am.TemperatureLogitsWrapper(wm, s["temperature"]), s["eps"],
                                       attack_params=dict(s["attack_params"]))
        pgd_hp = {"eps": stock.eps, "alpha": stock.alpha, "steps": stock.steps, "random_start": stock.random_start}
        audit["pgd_hyperparameters_from_stock_builder"] = pgd_hp
        for key in subset:
            g = torch.Generator().manual_seed(inputs[key]["seed"])
            noise[key] = torch.empty((1, 2, inputs[key]["x"].shape[2], 1), dtype=torch.float32).uniform_(
                -pgd_hp["eps"], pgd_hp["eps"], generator=g)
        audit["pgd_noise_procedure"] = ("per sample: torch.empty([1,2,128,1]).uniform_(-eps, eps, "
                                        "generator=torch.Generator().manual_seed(formal_seed)); batch = torch.cat in sample order")

    def build_attack(attack, attack_model, keys):
        s = settings[attack]
        if attack != "pgd":
            return am._build_torchattacks(attack, attack_model, s["eps"], attack_params=dict(s["attack_params"]))
        atk = ControlledStartPGD(attack_model, eps=pgd_hp["eps"], alpha=pgd_hp["alpha"], steps=pgd_hp["steps"],
                                 random_start=pgd_hp["random_start"])
        atk.set_init_noise(noise[keys[0]] if len(keys) == 1 else torch.cat([noise[k] for k in keys], dim=0))
        return atk

    def attack_block(attack, keys):
        """Phase 1B optimized path for len(keys) samples at once (identical to Phase 1C v2)."""
        s = settings[attack]
        t0 = time.perf_counter()
        x = inputs[keys[0]]["x"] if len(keys) == 1 else np.concatenate([inputs[k]["x"] for k in keys], axis=0)
        training_before = wm.training
        orig_requires_grad = [p.requires_grad for p in wm.parameters()]
        orig_param_devices = [p.device for p in wm.parameters()]
        try:
            x_ta, a, b = am._iq_to_ta_input_minmax(torch.from_numpy(x).to(adapter.device))
            wm.set_minmax(a, b)
            y = torch.tensor([inputs[k]["clean_pred"] for k in keys], dtype=torch.long)
            atk = build_attack(attack, am.TemperatureLogitsWrapper(wm, s["temperature"]), keys)
            t1 = time.perf_counter()
            x_ta_adv = atk(x_ta, y)
            t2 = time.perf_counter()
            lin = (x_ta_adv - x_ta).abs().amax(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.float32)
            x_adv = am._ta_output_to_iq_minmax(x_ta_adv, a, b).detach().cpu().numpy().astype(np.float32)
            b_np = b.detach().cpu().numpy().reshape(-1)
        finally:
            if training_before:
                print("[attack_adapter] warning: wrapped model was already in train mode before this call")
            wm.eval()
            wm.clear_minmax()
            for p, req_grad, dev in zip(wm.parameters(), orig_requires_grad, orig_param_devices):
                p.requires_grad_(req_grad)
                if p.device != dev:
                    p.data = p.data.to(dev)
        if x_adv.shape != x.shape or x_adv.dtype != x.dtype:
            raise RuntimeError(f"output shape/dtype {x_adv.shape} {x_adv.dtype} != input {x.shape} {x.dtype}")
        t3 = time.perf_counter()
        return x_adv, lin, b_np, {"total_ms": (t3 - t0) * 1e3, "prep_ms": (t1 - t0) * 1e3,
                                  "kernel_ms": (t2 - t1) * 1e3, "post_ms": (t3 - t2) * 1e3}

    anchors: Dict[str, Dict[tuple, dict]] = {}
    fidelity = {}
    for attack in attacks:
        anchors[attack] = {}
        n_eq = 0
        for key in subset:
            inp = inputs[key]
            torch.manual_seed(inp["seed"])
            ref, _, _, y_used, _, _ = p1b.attack_path(am, adapter, torch, inp["x"], attack, settings[attack],
                                                      inp["clean_pred"], False, False)
            check_state(f"anchor {attack} {key}")
            mine, _, _, _ = attack_block(attack, [key])
            check_state(f"anchor-own {attack} {key}")
            if y_used != inp["clean_pred"]:
                p1a.abort(f"anchor label mismatch {attack} {key}")
            if not np.array_equal(ref, mine):
                p1a.abort(f"{attack} {key}: benchmark B=1 path is not bit-identical to the validated Phase 1B path "
                          f"(max abs diff {float(np.max(np.abs(ref.astype(np.float64) - mine))):.3e})"
                          + ("; ControlledStartPGD does not reproduce stock PGD's random start" if attack == "pgd" else ""))
            n_eq += 1
            sha = p1a.sha256_array(ref)
            if attack in p1a.DETERMINISTIC_ATTACKS:
                fr = formal_attack[key + (attack, settings[attack]["profile"])]
                if sha != fr.get("adversarial_sha256"):
                    p1a.abort(f"{attack} {key}: anchor sha256 != formal adversarial_sha256")
            pred = infer_pred(ref)
            d_ref = ref - inp["x"]
            anchors[attack][key] = {"x_adv": ref, "sha": sha, "pred": pred,
                                    "success": bool(inp["clean_pred"] == inp["label"] and pred != inp["label"]),
                                    "linf": float(np.max(np.abs(d_ref))), "l2": float(np.linalg.norm(d_ref.reshape(-1)))}
        fidelity[attack] = f"{n_eq}/{len(subset)} bit-identical to Phase 1B path" + (
            " (stock PGD under torch.manual_seed(formal_seed) vs ControlledStartPGD)" if attack == "pgd" else
            "; anchor sha256 == formal adversarial_sha256 for all")
    audit["anchor_fidelity"] = fidelity
    p1a.log(f"[{label}] audit passed; anchors built and validated: {fidelity}")

    # ---------------------------------------------------------------- per-cell measurement
    raw_f = (out_dir / f"raw_{label}.csv").open("w", newline="")
    smp_f = (out_dir / f"per_sample_{label}.csv").open("w", newline="")
    raw_w = csv.DictWriter(raw_f, fieldnames=RAW_FIELDS)
    smp_w = csv.DictWriter(smp_f, fieldnames=SAMPLE_FIELDS)
    raw_w.writeheader()
    smp_w.writeheader()
    semantic_different: List[dict] = []
    invalid: List[dict] = []
    class_counts = {c: 0 for c in CLASSES}

    def validate_outputs(cell, attack, B, bi, rep, mode, keys, x_adv, lin, b_np, cell_counts) -> dict:
        """Phase 1C v2 rule, unchanged: classify every sample against its independent B=1 anchor."""
        agg = {c: 0 for c in CLASSES}
        agg.update({"maxdiff": 0.0, "max_linf_rel": 0.0, "max_l2_rel": 0.0, "n_pred": 0, "n_succ": 0,
                    "n_fin": 0, "n_eps": 0})
        eps = float(settings[attack]["eps"])
        shape_ok = x_adv.shape == (len(keys), 2, x_adv.shape[2]) and x_adv.dtype == np.float32
        for i, key in enumerate(keys):
            inp, anc = inputs[key], anchors[attack][key]
            xi = np.ascontiguousarray(x_adv[i:i + 1])
            fin = bool(np.isfinite(xi).all())
            bit = bool(np.array_equal(xi, anc["x_adv"]))
            md = float(np.max(np.abs(xi.astype(np.float64) - anc["x_adv"].astype(np.float64)))) if fin else float("inf")
            pred = anc["pred"] if bit else (infer_pred(xi) if fin else -1)
            succ = bool(inp["clean_pred"] == inp["label"] and pred != inp["label"])
            diff = xi - inp["x"]
            linf, l2 = float(np.max(np.abs(diff))), float(np.linalg.norm(diff.reshape(-1)))
            bound = eps * float(b_np[i])
            norm_ok = bool(lin[i] <= eps + NORM_EPS_TOL)
            iq_ok = bool(linf <= bound * (1 + IQ_EPS_REL_TOL) + IQ_EPS_ABS_TOL)
            pred_eq, succ_eq = pred == anc["pred"], succ == anc["success"]
            linf_rel, l2_rel = p1c._rel(abs(linf - anc["linf"]), anc["linf"]), p1c._rel(abs(l2 - anc["l2"]), anc["l2"])
            if not (fin and shape_ok and norm_ok and iq_ok):
                cls = "INVALID"
            elif bit:
                cls = "BIT_IDENTICAL"
            elif pred_eq and succ_eq:
                cls = "BITWISE_DIFFERENT_SEMANTIC_SAME"
            else:
                cls = "SEMANTIC_DIFFERENT"
            agg[cls] += 1
            class_counts[cls] += 1
            cell_counts[mode][cls] += 1
            agg["maxdiff"] = max(agg["maxdiff"], md)
            agg["max_linf_rel"] = max(agg["max_linf_rel"], linf_rel)
            agg["max_l2_rel"] = max(agg["max_l2_rel"], l2_rel)
            agg["n_pred"] += pred_eq
            agg["n_succ"] += succ_eq
            agg["n_fin"] += fin
            agg["n_eps"] += norm_ok and iq_ok
            row = {
                "cell_id": cell["cell_id"], "order_index": cell["order_index"],
                "condition": label, "intra_threads": intra, "attack": attack, "batch_size": B, "batch_index": bi,
                "position_in_batch": i, "batch_actual_size": len(keys), "repeat": rep, "mode": mode,
                "modulation": key[0], "snr_db": key[1], "sample_index": key[2], "formal_seed": inp["seed"],
                "label": inp["label"], "clean_pred": inp["clean_pred"],
                "attacked_pred": pred, "anchor_attacked_pred": anc["pred"], "attacked_pred_equal": pred_eq,
                "attack_success": succ, "anchor_attack_success": anc["success"], "attack_success_equal": succ_eq,
                "iq_linf": linf, "anchor_iq_linf": anc["linf"], "linf_abs_diff": abs(linf - anc["linf"]),
                "linf_rel_diff": linf_rel, "iq_l2": l2, "anchor_iq_l2": anc["l2"], "l2_abs_diff": abs(l2 - anc["l2"]),
                "l2_rel_diff": l2_rel, "iq_linf_normalized": float(lin[i]), "norm_eps_bound_ok": norm_ok,
                "per_sample_b": float(b_np[i]), "iq_linf_bound_eps_x_b": bound, "iq_eps_bound_ok": iq_ok,
                "x_adv_sha256": p1a.sha256_array(xi), "anchor_sha256": anc["sha"], "bit_identical_to_anchor": bit,
                "max_abs_diff_vs_anchor": md, "n_diff_elements": int(np.sum(xi != anc["x_adv"])),
                "finite": fin, "shape": "x".join(map(str, xi.shape)), "dtype": str(xi.dtype),
                "shape_dtype_ok": shape_ok, "classification": cls,
            }
            smp_w.writerow({k_: row[k_] for k_ in SAMPLE_FIELDS})
            if cls == "SEMANTIC_DIFFERENT":
                semantic_different.append({k_: row[k_] for k_ in (
                    "cell_id", "condition", "attack", "batch_size", "batch_index", "position_in_batch", "repeat",
                    "mode", "modulation", "snr_db", "sample_index", "label", "clean_pred", "anchor_attacked_pred",
                    "attacked_pred", "anchor_attack_success", "attack_success", "max_abs_diff_vs_anchor",
                    "linf_rel_diff", "l2_rel_diff")})
            elif cls == "INVALID":
                invalid.append({k_: row[k_] for k_ in (
                    "cell_id", "condition", "attack", "batch_size", "batch_index", "repeat", "mode", "modulation",
                    "snr_db", "sample_index", "finite", "shape_dtype_ok", "norm_eps_bound_ok", "iq_eps_bound_ok")})
        agg["shape_ok"] = shape_ok
        return agg

    def run_cell(cell: dict) -> dict:
        """Phase 1C v2 cell body for one (attack, B): warmup (untimed) + repeats x all batches x both modes."""
        attack, B = cell["attack"], int(cell["batch_size"])
        cell_counts = {"sequential": {c: 0 for c in CLASSES}, "batched": {c: 0 for c in CLASSES}}
        batches = [subset[i:i + B] for i in range(0, len(subset), B)]
        t_w0 = time.perf_counter()
        for keys in batches[: min(args.warmup_batches, len(batches))]:
            for k in keys:
                attack_block(attack, [k])
            attack_block(attack, keys)
        check_state(f"warmup {cell['cell_id']}")
        t_w1 = time.perf_counter()
        for rep in range(args.repeats):
            for bi, keys in enumerate(batches):
                temp = p1a.cpu_temp_c()
                order = ("sequential", "batched") if (rep + bi) % 2 == 0 else ("batched", "sequential")
                for pos, mode in enumerate(order):
                    if mode == "sequential":
                        outs, lins, bs, calls = [], [], [], []
                        t0 = time.perf_counter()
                        for k in keys:
                            xo, li, bb, tm = attack_block(attack, [k])
                            outs.append(xo)
                            lins.append(li)
                            bs.append(bb)
                            calls.append(tm)
                        total = (time.perf_counter() - t0) * 1e3
                        x_adv = np.concatenate(outs, axis=0)
                        lin, b_np = np.concatenate(lins), np.concatenate(bs)
                        tm = {"total_ms": total, "prep_ms": sum(c["prep_ms"] for c in calls),
                              "kernel_ms": sum(c["kernel_ms"] for c in calls),
                              "post_ms": sum(c["post_ms"] for c in calls)}
                        call_list = ";".join(f"{c['total_ms']:.6f}" for c in calls)
                    else:
                        x_adv, lin, b_np, tm = attack_block(attack, keys)
                        call_list = ""
                    check_state(f"{cell['cell_id']} batch={bi} {mode}")
                    agg = validate_outputs(cell, attack, B, bi, rep, mode, keys, x_adv, lin, b_np, cell_counts)
                    raw_w.writerow({
                        "cell_id": cell["cell_id"], "order_index": cell["order_index"],
                        "condition": label, "intra_threads": intra, "interop_threads": interop,
                        "attack": attack, "attack_profile": settings[attack]["profile"], "eps": settings[attack]["eps"],
                        "batch_size": B, "batch_index": bi, "batch_actual_size": len(keys),
                        "is_partial_batch": len(keys) != B, "repeat": rep, "mode": mode, "mode_order": pos,
                        "total_ms": tm["total_ms"], "per_sample_ms": tm["total_ms"] / len(keys),
                        "prep_ms": tm["prep_ms"], "kernel_ms": tm["kernel_ms"], "post_ms": tm["post_ms"],
                        "seq_call_ms_list": call_list,
                        "n_bit_identical": agg["BIT_IDENTICAL"],
                        "n_bitwise_different_semantic_same": agg["BITWISE_DIFFERENT_SEMANTIC_SAME"],
                        "n_semantic_different": agg["SEMANTIC_DIFFERENT"], "n_invalid": agg["INVALID"],
                        "max_abs_diff_vs_anchor": agg["maxdiff"], "max_linf_rel_diff": agg["max_linf_rel"],
                        "max_l2_rel_diff": agg["max_l2_rel"],
                        "n_pred_match_anchor": agg["n_pred"], "n_success_match_anchor": agg["n_succ"],
                        "n_finite": agg["n_fin"], "n_eps_bound_ok": agg["n_eps"], "shape_dtype_ok": agg["shape_ok"],
                        "sample_keys": "|".join(f"{k[0]}:{k[1]}:{k[2]}" for k in keys), "cpu_temp_c": temp,
                    })
                    if invalid:  # INVALID -> abort immediately, after persisting what we have
                        raw_f.flush()
                        smp_f.flush()
                        (out_dir / f"invalid_{label}.json").write_text(json.dumps(invalid, indent=2, default=str))
                        (out_dir / f"semantic_different_{label}.json").write_text(
                            json.dumps(semantic_different, indent=2, default=str))
                        p1a.abort(f"[{label}] INVALID output: {invalid[0]} (see invalid_{label}.json)")
                raw_f.flush()
                smp_f.flush()
        t_end = time.perf_counter()
        return {"warmup_s": t_w1 - t_w0, "timed_s": t_end - t_w1, "class_counts": cell_counts,
                "n_batches_per_repeat": len(batches)}

    send({"event": "ready", "label": label, "pid": os.getpid(), "intra_threads": intra, "interop_threads": interop,
          "anchor_fidelity": fidelity})
    while True:
        line = sys.stdin.readline()  # blocks idle (no CPU) between cells
        if not line:
            break
        msg = json.loads(line)
        if msg.get("cmd") == "run":
            cell = msg["cell"]
            p1a.log(f"[{label}] cell {cell['cell_id']} start")
            res = run_cell(cell)
            p1a.log(f"[{label}] cell {cell['cell_id']} done warmup={res['warmup_s']:.1f}s timed={res['timed_s']:.1f}s "
                    f"classes={res['class_counts']}")
            send({"event": "cell_done", "cell_id": cell["cell_id"], **res})
        elif msg.get("cmd") == "quit":
            break
    raw_f.close()
    smp_f.close()
    (out_dir / f"semantic_different_{label}.json").write_text(json.dumps(semantic_different, indent=2, default=str))
    meta = {
        "label": label, "pid": os.getpid(), "intra_threads": intra, "interop_threads": interop,
        "env_thread_vars": {k: os.environ.get(k) for k in p1a.ENV_THREAD_VARS},
        "torch_version": torch.__version__, "torch_parallel_info": torch.__config__.parallel_info(),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "torchattacks_version": getattr(am._torchattacks, "__version__", None),
        "attack_adapter_source_sha256": p1a.sha256_file(Path(am.__file__)),
        "batch_semantics_audit": audit, "classification_counts": class_counts,
        "n_semantic_different": len(semantic_different), "n_invalid": len(invalid),
    }
    (out_dir / f"worker_{label}.json").write_text(json.dumps(meta, indent=2, default=str))
    send({"event": "bye", "label": label})


# --------------------------------------------------------------------------- parent: aggregate
def _f(r, k):
    return float(r[k])


def aggregate(out_dir: Path, cells: List[dict], labels: List[str], attack_settings: dict) -> tuple:
    rows, samples = [], []
    for lab in labels:
        with (out_dir / f"raw_{lab}.csv").open(newline="") as f:
            rows.extend(csv.DictReader(f))
        with (out_dir / f"per_sample_{lab}.csv").open(newline="") as f:
            samples.extend(csv.DictReader(f))
    rows.sort(key=lambda r: (int(r["order_index"]), int(r["repeat"]), int(r["batch_index"]), int(r["mode_order"])))
    with (out_dir / "raw_measurements.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with (out_dir / "per_sample_validation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS)
        w.writeheader()
        w.writerows(samples)

    summary, per_cell_val, sem_diff = [], {}, []
    for c in cells:
        th = c.get("thermal", {})
        base = {
            "cell_id": c["cell_id"], "order_index": c["order_index"], "condition": c["condition"],
            "intra_threads": c["intra_threads"], "interop_threads": c["interop_threads"], "attack": c["attack"],
            "attack_profile": attack_settings[c["attack"]]["profile"], "batch_size": c["batch_size"],
            "cell_run_status": c.get("run_status"),
            "thermal_status": th.get("status", ""), "thermal_reasons": " | ".join(th.get("reasons", [])),
            "cooldown_outcome": c.get("cooldown", {}).get("outcome"),
            "cooldown_waited_s": c.get("cooldown", {}).get("waited_s"),
            "temp_before_c": th.get("before", {}).get("temp_c"), "temp_after_c": th.get("after", {}).get("temp_c"),
            "temp_max_in_cell_c": th.get("temp_max_in_cell_c"),
            "throttled_before": th.get("before", {}).get("throttled", {}).get("value_hex"),
            "throttled_after": th.get("after", {}).get("throttled", {}).get("value_hex"),
            "throttled_low_before": th.get("before", {}).get("throttled", {}).get("low"),
            "throttled_low_after": th.get("after", {}).get("throttled", {}).get("low"),
            "throttled_high_before": th.get("before", {}).get("throttled", {}).get("high"),
            "throttled_high_after": th.get("after", {}).get("throttled", {}).get("high"),
            "throttled_low_max_in_cell": th.get("throttled_low_or_in_cell"),
            "arm_clock_hz_before": th.get("before", {}).get("arm_clock_hz"),
            "arm_clock_hz_after": th.get("after", {}).get("arm_clock_hz"),
            "arm_clock_hz_min_in_cell": th.get("arm_clock_hz_min_in_cell"),
            "arm_clock_hz_max_in_cell": th.get("arm_clock_hz_max_in_cell"),
            "cell_warmup_s": c.get("result", {}).get("warmup_s"), "cell_timed_s": c.get("result", {}).get("timed_s"),
        }
        cr = [r for r in rows if r["cell_id"] == c["cell_id"]]
        if c.get("run_status") != "RAN" or not cr:
            per_cell_val[c["cell_id"]] = {"status": c.get("run_status") or "NOT_RUN"}
            summary.append({**base, "validation_status": c.get("run_status") or "NOT_RUN"})
            continue
        seq = [r for r in cr if r["mode"] == "sequential"]
        bat = [r for r in cr if r["mode"] == "batched"]
        cs = [s for s in samples if s["cell_id"] == c["cell_id"]]
        cs_b = [s for s in cs if s["mode"] == "batched"]
        cs_s = [s for s in cs if s["mode"] == "sequential"]

        def counts(ss):
            return {cl: sum(1 for s in ss if s["classification"] == cl) for cl in CLASSES}

        cb, cq = counts(cs_b), counts(cs_s)
        nb = len(cs_b)
        status = "PASS" if (cb["SEMANTIC_DIFFERENT"] + cq["SEMANTIC_DIFFERENT"] == 0
                            and cb["INVALID"] + cq["INVALID"] == 0 and nb > 0) else "FAIL"
        show = status == "PASS"

        def fmax(ss, k):
            vals = [float(s[k]) for s in ss]
            return max(vals) if vals else float("nan")

        nonbit = sorted({f"{s['modulation']}/{s['snr_db']}/{s['sample_index']}" for s in cs
                         if s["classification"] != "BIT_IDENTICAL"})
        per_cell_val[c["cell_id"]] = {
            "status": status, "thermal_status": base["thermal_status"],
            "batched": {"n": nb, **{cl: {"n": v, "pct": 100.0 * v / nb if nb else float("nan")} for cl, v in cb.items()}},
            "sequential": {"n": len(cs_s), **{cl: {"n": v} for cl, v in cq.items()}},
            "non_bit_identical_samples": nonbit,
            "max_abs_x_adv_diff": fmax(cs, "max_abs_diff_vs_anchor"),
        }
        sem_diff += [{k_: s[k_] for k_ in ("cell_id", "condition", "attack", "batch_size", "batch_index",
                                            "position_in_batch", "repeat", "mode", "modulation", "snr_db",
                                            "sample_index", "label", "clean_pred", "anchor_attacked_pred",
                                            "attacked_pred", "anchor_attack_success", "attack_success")}
                     for s in cs if s["classification"] == "SEMANTIC_DIFFERENT"]

        B = int(c["batch_size"])
        full_b = [r for r in bat if not p1a.parse_bool(r["is_partial_batch"])]
        n_samples = sum(int(r["batch_actual_size"]) for r in bat)
        calls = [float(v) for r in seq for v in r["seq_call_ms_list"].split(";") if v]
        sum_b = sum(_f(r, "total_ms") for r in bat)
        sum_s = sum(_f(r, "total_ms") for r in seq)
        paired: Dict[tuple, list] = {}
        for r in seq:
            paired.setdefault((r["repeat"], r["batch_index"]), [None, None])[0] = _f(r, "total_ms")
        for r in bat:
            paired.setdefault((r["repeat"], r["batch_index"]), [None, None])[1] = _f(r, "total_ms")
        ratios = [s_ / b_ for s_, b_ in paired.values() if s_ and b_]
        bt = p1a.stats([_f(r, "total_ms") for r in full_b])
        am_ = p1a.stats([_f(r, "per_sample_ms") for r in bat])
        b1 = p1a.stats(calls)
        rec = {
            **base,
            "validation_status": status,
            "n_samples_per_repeat": n_samples // max(1, len({r["repeat"] for r in bat})),
            "n_batches": len(bat), "n_full_batches": len(full_b),
            "partial_batch_size": next((int(r["batch_actual_size"]) for r in bat
                                        if p1a.parse_bool(r["is_partial_batch"])), ""),
            "n_batched_outputs": nb, "n_bit_identical": cb["BIT_IDENTICAL"],
            "n_bitwise_different_semantic_same": cb["BITWISE_DIFFERENT_SEMANTIC_SAME"],
            "n_semantic_different": cb["SEMANTIC_DIFFERENT"], "n_invalid": cb["INVALID"],
            "sequential_n_not_bit_identical": len(cs_s) - cq["BIT_IDENTICAL"],
            "sequential_n_semantic_different": cq["SEMANTIC_DIFFERENT"],
            "non_bit_identical_samples": ";".join(nonbit),
            "max_abs_x_adv_diff": per_cell_val[c["cell_id"]]["max_abs_x_adv_diff"],
            **p1c._stat_cols("b1_single_call_ms", calls),
            **p1c._stat_cols("batch_completion_ms_fullbatches", [_f(r, "total_ms") for r in full_b]),
            **p1c._stat_cols("amortized_batch_ms_per_sample", [_f(r, "per_sample_ms") for r in bat]),
            **p1c._stat_cols("sequential_ms_per_sample", [_f(r, "per_sample_ms") for r in seq]),
            **p1c._stat_cols("batch_kernel_ms_fullbatches", [_f(r, "kernel_ms") for r in full_b]),
            **p1c._stat_cols("batch_kernel_ms_per_sample", [_f(r, "kernel_ms") / int(r["batch_actual_size"]) for r in bat]),
            **p1c._stat_cols("sequential_kernel_ms_per_sample",
                             [_f(r, "kernel_ms") / int(r["batch_actual_size"]) for r in seq]),
            "throughput_batch_samples_per_sec": (n_samples / sum_b * 1e3) if show and sum_b > 0 else "",
            "throughput_sequential_samples_per_sec": (n_samples / sum_s * 1e3) if show and sum_s > 0 else "",
            "speedup_batch_vs_sequential_total": (sum_s / sum_b) if show and sum_b > 0 else "",
            "speedup_batch_vs_sequential_paired_median": float(np.median(ratios)) if show and ratios else "",
            "speedup_batch_vs_sequential_paired_p5": float(np.percentile(ratios, 5)) if show and ratios else "",
            "flag_A_batch_completion_median_le_40ms": (bt["median"] <= BUDGET_MS) if bt["n"] else "",
            "flag_B_amortized_median_per_sample_le_40ms": am_["median"] <= BUDGET_MS,
            "flag_C_b1_single_call_median_le_40ms": b1["median"] <= BUDGET_MS,
            "latency_note": "amortized per-sample time (flag B) is NOT a single-sample decision latency; flag A is "
                            "batch completion (plus batch-fill wait in a live system); flag C is the B=1 decision latency",
        }
        summary.append(rec)

    fields: List[str] = []
    for r in summary:
        for k in r:
            if k not in fields:
                fields.append(k)
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(summary)

    # ---- intra-thread comparisons: a faster condition is named only for PASS + THERMAL_CLEAN pairs
    comparisons = {}
    by_key = {(r["attack"], int(r["batch_size"]), int(r["intra_threads"])): r for r in summary}
    thread_vals = sorted({int(c["intra_threads"]) for c in cells})
    if len(thread_vals) == 2:
        ta, tb = thread_vals
        for attack, B in sorted({(c["attack"], int(c["batch_size"])) for c in cells}):
            ra, rb = by_key.get((attack, B, ta)), by_key.get((attack, B, tb))
            why = []
            for t, r in ((ta, ra), (tb, rb)):
                if r is None or r.get("validation_status") != "PASS":
                    why.append(f"intra{t} cell not PASS ({None if r is None else r.get('validation_status')})")
                elif r.get("thermal_status") != "THERMAL_CLEAN":
                    why.append(f"intra{t} cell {r.get('thermal_status')}: {r.get('thermal_reasons')}")
            ok = not why
            entry = {"claim_allowed": ok, "not_claimable_reasons": why}
            if ra and rb and ra.get("validation_status") == "PASS" and rb.get("validation_status") == "PASS":
                for m, lower_is_better in (("b1_single_call_ms_median", True),
                                           ("batch_completion_ms_fullbatches_median", True),
                                           ("amortized_batch_ms_per_sample_median", True),
                                           ("throughput_batch_samples_per_sec", False)):
                    va, vb = float(ra[m]), float(rb[m])
                    entry[m] = {f"intra{ta}": va, f"intra{tb}": vb, f"ratio_intra{tb}_over_intra{ta}": vb / va if va else None}
                    if ok:
                        better = ta if ((va <= vb) == lower_is_better) else tb
                        entry[m]["better_condition"] = f"intra{better}"
                        entry[m]["difference_pct"] = 100.0 * abs(vb - va) / min(va, vb) if min(va, vb) else None
            comparisons[f"{attack}/B{B}"] = entry

    counts_all = {cl: sum(v["batched"][cl]["n"] for v in per_cell_val.values() if "batched" in v) for cl in CLASSES}
    ran = [v for v in per_cell_val.values() if "batched" in v]
    n_fail = sum(1 for v in ran if v["status"] != "PASS")
    n_skip = sum(1 for v in per_cell_val.values() if "batched" not in v)
    n_review = sum(1 for r in summary if r.get("thermal_status") == "THERMAL_REVIEW")
    overall = "PASS" if ran and n_fail == 0 and n_skip == 0 else ("FAIL" if n_fail else "INCOMPLETE")
    validation = {
        "rule": "Phase 1C v2 (bitwise AND semantic); SEMANTIC_DIFFERENT fails the cell; INVALID aborts the run",
        "per_cell": per_cell_val, "batched_classification_totals": counts_all,
        "semantic_different_samples": sem_diff,
        "n_cells_planned": len(cells), "n_cells_ran": len(ran), "n_cells_fail": n_fail, "n_cells_skipped": n_skip,
        "n_cells_thermal_review": n_review,
        "thread_comparisons": comparisons,
        "overall": overall + (f" ({n_review} cell(s) THERMAL_REVIEW)" if n_review else ""),
        "notes": [
            "Known BIM batch-shape numerical difference (PAM4/6 dB/sample 1, B>1) is expected as "
            "BITWISE_DIFFERENT_SEMANTIC_SAME and is listed per cell in non_bit_identical_samples.",
            "THERMAL_REVIEW is raised by current-state get_throttled bits (0-3) before/during/after a cell, history "
            "bits (16-19) newly set across a cell, missing vcgencmd, governor change, or a failed cooldown; history "
            "bits that were already set do not by themselves flag a cell.",
            "A faster thread condition is named only when both cells are PASS and THERMAL_CLEAN.",
            "Amortized ms/sample is not a decision latency. Nothing here demonstrates NEON-specific causes.",
        ],
    }
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2, default=str))
    return summary, validation, len(rows)


# --------------------------------------------------------------------------- parent
def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def parent_main(args) -> None:
    threads = [int(t) for t in args.threads.split(",")]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    attacks = [a.strip().lower() for a in args.attacks.split(",")]
    if len(threads) != 2 or len(set(threads)) != 2 or any(t not in (1, 2, 4) for t in threads):
        p1a.abort("--threads must name exactly two distinct conditions from 1,2,4 (default 1,4)")
    if any(b < 1 for b in batch_sizes) or len(set(batch_sizes)) != len(batch_sizes):
        p1a.abort("--batch-sizes must be distinct positive integers")
    if not attacks or any(a not in ATTACKS for a in attacks) or len(set(attacks)) != len(attacks):
        p1a.abort(f"--attacks must be a subset of {ATTACKS}")

    cells = build_cell_order(attacks, batch_sizes, threads, args.order_variant)
    est = estimate(cells, args.repeats, args.warmup_batches)
    n_waits = len(cells)
    exp_cool = n_waits * args.expected_cooldown_seconds
    worst_cool = n_waits * args.max_cooldown_seconds
    active_min = (est["active_cell_seconds"] + est["worker_setup_seconds"]) / 60
    plan = {
        "cells": [{k: c[k] for k in ("order_index", "cell_id", "attack", "batch_size", "condition")} for c in cells],
        "order_design": ("(attack,B) pairs interleaved across attacks and B; the two thread conditions of each pair "
                         "back-to-back in ABBA order across pairs; --order-variant=" + args.order_variant),
        "order_balance": order_balance(cells),
        "estimate_minutes": {
            "active_compute_cells": est["active_cell_seconds"] / 60,
            "worker_setup": est["worker_setup_seconds"] / 60,
            "active_total": active_min,
            "cooldown_expected_assumption": exp_cool / 60,
            "cooldown_expected_assumption_basis": f"{n_waits} waits x --expected-cooldown-seconds "
                                                  f"{args.expected_cooldown_seconds} (planning assumption, not measured)",
            "cooldown_worst_case": worst_cool / 60,
            "wall_expected": active_min + exp_cool / 60,
            "wall_worst_case": active_min + worst_cool / 60,
        },
    }
    p1a.log("[parent] cell order:")
    for c in cells:
        p1a.log(f"   {c['order_index']:2d}  {c['attack']:4s} B={c['batch_size']:<3d} {c['condition']}")
    p1a.log(f"[parent] order balance: { {k: v['mean_position'] for k, v in plan['order_balance'].items()} }")
    e = plan["estimate_minutes"]
    p1a.log(f"[parent] estimate: active {e['active_total']:.1f} min (cells {e['active_compute_cells']:.1f} + setup "
            f"{e['worker_setup']:.1f}); cooldown expected ~{e['cooldown_expected_assumption']:.0f} min (assumption), "
            f"worst case {e['cooldown_worst_case']:.0f} min; wall expected ~{e['wall_expected']:.0f} min, worst case "
            f"{e['wall_worst_case']:.0f} min")
    if e["wall_worst_case"] > 60:
        p1a.log("[parent] NOTE: worst-case wall time exceeds 60 min; run under nohup (see module docstring)")
    now = snapshot()
    p1a.log(f"[parent] now: temp={now['temp_c']} C throttled={now['throttled']['value_hex']} "
            f"low={now['throttled']['low_names']} high={now['throttled']['high_names']} "
            f"arm_clock={now['arm_clock_hz']} governor={set(now['governor'].values())}")
    if args.dry_run:
        print(json.dumps({"plan": plan, "system_now": now}, indent=2, default=str))
        return

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (REPO_ROOT / (args.out_dir or f"results/accel_phase1d0_thermal_{ts}")).resolve()
    for p in p1a.PROTECTED_RESULT_DIRS + ("results/accel_phase1c_v2_formal_20260926_164547",
                                          "results/accel_phase1a_formal_20260926_152737",
                                          "results/accel_phase1b_20260926_155342"):
        prot = (REPO_ROOT / p).resolve()
        if out_dir == prot or prot in out_dir.parents or out_dir in prot.parents:
            p1a.abort(f"refusing to write inside/over results directory {p}")
    out_dir.mkdir(parents=True, exist_ok=False)

    ctx = p1a.load_formal_context(args)
    cfg, manifest, formal_dir = ctx["cfg"], ctx["manifest"], ctx["formal_dir"]
    subset = p1a.build_subset(cfg)
    base_rows, attack_rows = p1a.load_formal_rows(formal_dir, set(subset))
    attack_settings = p1a.resolve_attack_settings(cfg, attack_rows, args.attack_temperature, args.diagnostics)
    attack_settings = {a: attack_settings[a] for a in attacks}
    sensing = p1a.resolve_sensing(cfg)

    p1a.log("[parent] hashing dataset / checkpoint / config ...")
    hashes = {
        "dataset_sha256": p1a.sha256_file(REPO_ROOT / cfg["dataset_path"]),
        "checkpoint_sha256": p1a.sha256_file(REPO_ROOT / cfg["checkpoint_path"]),
        "config_sha256": p1a.sha256_file(ctx["cfg_path"]),
        "formal_manifest_sha256": p1a.sha256_file(formal_dir / "manifest.json"),
        "attack_adapter_py_sha256": p1a.sha256_file(REPO_ROOT / "src/adapters/attack_adapter.py"),
        "adv_attack_py_sha256": p1a.sha256_file(REPO_ROOT / "external/adversarial-rf/util/adv_attack.py"),
        "adv_attack_py_pinned_audited_sha256": p1c.PINNED_ADV_ATTACK_SHA256,
        "phase1a_helpers_sha256": p1a.sha256_file(Path(p1a.__file__).resolve()),
        "phase1b_helpers_sha256": p1a.sha256_file(Path(p1b.__file__).resolve()),
        "phase1c_v2_script_sha256": p1a.sha256_file(Path(p1c.__file__).resolve()),
        "this_script_sha256": p1a.sha256_file(Path(__file__).resolve()),
    }
    if hashes["dataset_sha256"] != manifest.get("dataset_sha256"):
        p1a.abort("dataset sha256 differs from formal CPU manifest")
    if hashes["checkpoint_sha256"] != manifest.get("checkpoint_sha256"):
        p1a.abort("checkpoint sha256 differs from formal CPU manifest")
    env_meta = p1a.environment_metadata()

    worker_ctx = {"subset": [list(k) for k in subset], "formal_dir": p1a.rel(formal_dir),
                  "dataset_path": cfg["dataset_path"], "checkpoint_path": cfg["checkpoint_path"],
                  "sensing": sensing, "attack_settings": attack_settings, "attacks": attacks}
    ctx_file = out_dir / "worker_context.json"
    ctx_file.write_text(json.dumps(worker_ctx, indent=2, default=str))

    manifest_out = {
        "phase": "accel_phase1d0_thermal_control", "status": "running",
        "question": "With thermal state and run order controlled, how do intra=1 and intra=4 compare for PGD/BIM "
                    "at B=1/8/32 (single-sample latency, batch completion, amortized, throughput)?",
        "command": [sys.executable] + sys.argv, "out_dir": p1a.rel(out_dir), "formal_cpu_dir": p1a.rel(formal_dir),
        "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "hashes": hashes, "environment": env_meta, "attacks": attack_settings,
        "threads": threads, "interop_threads": INTEROP_THREADS, "batch_sizes": batch_sizes,
        "repeats": args.repeats, "warmup_batches_per_cell": args.warmup_batches,
        "thermal_control": {
            "max_start_temp_c": args.max_start_temp_c, "max_cooldown_seconds": args.max_cooldown_seconds,
            "cooldown_poll_seconds": args.cooldown_poll_seconds, "on_cooldown_timeout": args.on_cooldown_timeout,
            "max_consecutive_skips": args.max_consecutive_skips,
            "release_condition": "sysfs temperature <= max_start_temp_c AND get_throttled bits 0-3 == 0",
            "in_cell_temp_poll_seconds": args.in_cell_temp_poll_seconds,
            "in_cell_vc_poll_seconds": args.in_cell_vc_poll_seconds,
            "system_settings_changed": "none (governor, clocks, voltage, fan, OS settings are only read)",
            "throttle_bits": {"current_state_bits_0_3": THROTTLE_LOW_BITS, "sticky_history_bits_16_19": THROTTLE_HIGH_BITS},
            "thermal_review_rule": "current-state bits non-zero before/during(sampled)/after the cell, history bits "
                                   "newly set across the cell, vcgencmd unavailable, governor changed, or cooldown "
                                   "not reached; pre-existing history bits alone do not flag a cell",
        },
        "plan": plan,
        "correctness_rule": "Phase 1C v2 (see bench_attack_batch_pi5_v2.py / its formal manifest.correctness_rule_v2)",
        "path": "Phase 1B optimized path: clean prediction reused (outside timing), parameters not frozen, no print, "
                "no attack-object reuse; sequential mode = B independent B=1 calls of the same code",
        "pgd_random_start": "ControlledStartPGD imported from bench_attack_batch_pi5_v2.py; per-sample noise from "
                            "torch.Generator().manual_seed(formal_seed_i); B=1 fidelity vs stock PGD re-proven per worker",
        "timing": {"clock": "time.perf_counter", "definitions": "identical to Phase 1C v2 manifest.timing",
                   "cell_includes": "warmup (untimed, unrecorded) + repeats x all batches x {sequential, batched}"},
        "workers": {}, "cells": cells,
        "subset": {"modulations": cfg["modulations"], "snrs": p1a.SUBSET_SNRS,
                   "sample_indices": p1a.SUBSET_SAMPLE_INDICES, "n_source_samples": len(subset)},
    }
    man_path = out_dir / "manifest.json"
    _write_json(man_path, manifest_out)
    trace_rows: List[dict] = []

    workers: Dict[str, subprocess.Popen] = {}

    def shutdown_workers():
        for proc in workers.values():
            if proc.poll() is None:
                proc.kill()

    def fail(msg: str, cond: Optional[str] = None):
        shutdown_workers()
        inv = [json.loads(p.read_text()) for p in out_dir.glob("invalid_*.json")]
        _write_json(out_dir / "validation.json", {
            "overall": f"FAIL ({msg})", "invalid": inv or None,
            "note": "no throughput/speedup is reported for an aborted run; see worker_*.stderr.log"})
        manifest_out.update(status="aborted", abort_reason=msg, failed_condition=cond,
                            finished_utc=_dt.datetime.now(_dt.timezone.utc).isoformat())
        _write_json(man_path, manifest_out)
        _write_trace(out_dir, trace_rows)
        p1a.abort(msg)

    def recv(label: str) -> dict:
        line = workers[label].stdout.readline()
        if not line:
            rc = workers[label].wait()
            fail(f"worker {label} exited (rc={rc}) -- INVALID output, failed fail-closed check or crash", label)
        return json.loads(line)

    for t in threads:  # workers set up one after another; setup heat is removed by the first cooldown
        label = f"intra{t}_interop{INTEROP_THREADS}"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--out-dir", str(out_dir),
               "--worker-context", str(ctx_file), "--worker-threads", str(t),
               "--repeats", str(args.repeats), "--warmup-batches", str(args.warmup_batches)]
        p1a.log(f"[parent] starting persistent worker {label}")
        errf = (out_dir / f"worker_{label}.stderr.log").open("w")
        workers[label] = subprocess.Popen(cmd, cwd=REPO_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=errf, text=True, bufsize=1, env=os.environ.copy())
        t0 = time.monotonic()
        msg = recv(label)
        if msg.get("event") != "ready":
            fail(f"unexpected message from {label}: {msg}", label)
        manifest_out["workers"][label] = {**msg, "setup_seconds": time.monotonic() - t0}
        _write_json(man_path, manifest_out)

    consecutive_skips = 0
    for c in cells:
        label = c["condition"]
        p1a.log(f"[parent] cell {c['order_index'] + 1}/{len(cells)} {c['cell_id']}: cooling to <= "
                f"{args.max_start_temp_c} C (now {p1a.cpu_temp_c()} C)")
        cd = wait_until_cool(args.max_start_temp_c, args.max_cooldown_seconds, args.cooldown_poll_seconds)
        c["cooldown"] = cd
        if cd["outcome"] != "REACHED":
            if args.on_cooldown_timeout == "abort" or cd["outcome"] == "TEMPERATURE_UNAVAILABLE":
                c["run_status"] = "ABORTED_COOLDOWN"
                fail(f"cooldown {cd['outcome']} before {c['cell_id']} (last temp {cd['temp_last_c']} C after "
                     f"{cd['waited_s']:.0f} s)", label)
            c["run_status"] = "SKIPPED_COOLDOWN_TIMEOUT"
            consecutive_skips += 1
            p1a.log(f"[warn] SKIPPED {c['cell_id']}: temp {cd['temp_last_c']} C after {cd['waited_s']:.0f} s")
            _write_json(man_path, manifest_out)
            if consecutive_skips >= args.max_consecutive_skips:
                fail(f"{consecutive_skips} consecutive cells could not reach {args.max_start_temp_c} C within "
                     f"{args.max_cooldown_seconds} s; the threshold is not practical on this cooling setup")
            continue
        consecutive_skips = 0
        before = snapshot()
        sampler = CellSampler(c["cell_id"], c["order_index"], args.in_cell_temp_poll_seconds,
                              args.in_cell_vc_poll_seconds).start()
        workers[label].stdin.write(json.dumps({"cmd": "run", "cell": {
            k: c[k] for k in ("cell_id", "order_index", "attack", "batch_size")}}) + "\n")
        workers[label].stdin.flush()
        msg = recv(label)
        samples = sampler.stop()
        after = snapshot()
        trace_rows.extend(samples)
        if msg.get("event") != "cell_done" or msg.get("cell_id") != c["cell_id"]:
            fail(f"unexpected reply for {c['cell_id']}: {msg}", label)
        status, reasons = classify_thermal(before, after, samples, cd, args.max_start_temp_c)
        temps = [s["temp_c"] for s in samples if s["temp_c"] is not None] + \
                [v for v in (before["temp_c"], after["temp_c"]) if v is not None]
        clocks = [int(s["arm_clock_hz"]) for s in samples if s["arm_clock_hz"] not in ("", None)]
        lows = [int(s["throttled_low"]) for s in samples if s["throttled_low"] not in ("", None)]
        low_or = 0
        for v in lows + [x for x in (before["throttled"]["low"], after["throttled"]["low"]) if x is not None]:
            low_or |= v
        c["run_status"] = "RAN"
        c["result"] = {k: msg[k] for k in ("warmup_s", "timed_s", "class_counts", "n_batches_per_repeat")}
        c["thermal"] = {"status": status, "reasons": reasons, "before": before, "after": after,
                        "temp_max_in_cell_c": max(temps) if temps else None,
                        "n_in_cell_samples": len(samples), "n_in_cell_vc_samples": len(lows),
                        "throttled_low_or_in_cell": low_or,
                        "arm_clock_hz_min_in_cell": min(clocks) if clocks else None,
                        "arm_clock_hz_max_in_cell": max(clocks) if clocks else None}
        _write_json(man_path, manifest_out)
        _write_trace(out_dir, trace_rows)
        p1a.log(f"[parent] {c['cell_id']} done in {msg['warmup_s'] + msg['timed_s']:.1f} s; "
                f"temp {before['temp_c']} -> {after['temp_c']} C (max {c['thermal']['temp_max_in_cell_c']}); "
                f"throttled {before['throttled']['value_hex']} -> {after['throttled']['value_hex']}; {status}"
                + (f" {reasons}" if reasons else ""))

    for label, proc in workers.items():
        proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
        proc.stdin.flush()
        msg = recv(label)
        if msg.get("event") != "bye":
            fail(f"unexpected shutdown message from {label}: {msg}", label)
        proc.wait(timeout=120)
        manifest_out["workers"][label]["returncode"] = proc.returncode
        wj = out_dir / f"worker_{label}.json"
        if wj.exists():
            manifest_out["workers"][label]["worker_meta"] = json.loads(wj.read_text())

    summary, validation, n = aggregate(out_dir, cells, list(workers), attack_settings)
    manifest_out.update(status="complete", raw_rows=n, validation_overall=validation["overall"],
                        finished_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(), environment_end=snapshot())
    _write_json(man_path, manifest_out)
    _write_trace(out_dir, trace_rows)
    p1a.log(f"[parent] done: {out_dir}  validation={validation['overall']}")
    for r in summary:
        if r.get("validation_status") in ("PASS", "FAIL"):
            p1a.log(f"  {r['order_index']:2d} {r['condition']} {r['attack']:4s} B={r['batch_size']:<2}  "
                    f"B1 med={r['b1_single_call_ms_median']:7.2f}  completion med="
                    f"{r['batch_completion_ms_fullbatches_median']:8.2f}  amort med="
                    f"{r['amortized_batch_ms_per_sample_median']:7.2f}  "
                    f"bit={r['n_bit_identical']}/{r['n_batched_outputs']} sem_same={r['n_bitwise_different_semantic_same']} "
                    f"{r['validation_status']} {r['thermal_status']}")
        else:
            p1a.log(f"  {r['order_index']:2d} {r['condition']} {r['attack']:4s} B={r['batch_size']:<2}  "
                    f"{r['validation_status']}")
    if not validation["overall"].startswith("PASS"):
        sys.exit(2)


def _write_trace(out_dir: Path, trace_rows: List[dict]) -> None:
    with (out_dir / "thermal_trace.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACE_FIELDS)
        w.writeheader()
        w.writerows(trace_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=p1a.DEFAULT_CONFIG)
    ap.add_argument("--formal-dir", default=p1a.DEFAULT_FORMAL_CPU_DIR)
    ap.add_argument("--out-dir", default=None, help="default results/accel_phase1d0_thermal_<timestamp> (must not exist)")
    ap.add_argument("--threads", default=DEFAULT_THREADS, help="exactly two intra-op conditions (default 1,4)")
    ap.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES)
    ap.add_argument("--attacks", default=",".join(ATTACKS), help="subset of pgd,bim")
    ap.add_argument("--repeats", type=int, default=3, help="passes over the 88-sample subset per cell")
    ap.add_argument("--warmup-batches", type=int, default=2, help="unrecorded warmup batches per cell, both modes")
    ap.add_argument("--order-variant", choices=["forward", "reverse"], default="forward")
    ap.add_argument("--max-start-temp-c", type=float, default=55.0)
    ap.add_argument("--max-cooldown-seconds", type=float, default=900.0)
    ap.add_argument("--cooldown-poll-seconds", type=float, default=5.0)
    ap.add_argument("--on-cooldown-timeout", choices=["skip", "abort"], default="skip")
    ap.add_argument("--max-consecutive-skips", type=int, default=2)
    ap.add_argument("--expected-cooldown-seconds", type=float, default=180.0,
                    help="planning assumption for the wall-time estimate only")
    ap.add_argument("--in-cell-temp-poll-seconds", type=float, default=1.0)
    ap.add_argument("--in-cell-vc-poll-seconds", type=float, default=10.0, help="0 disables in-cell vcgencmd polling")
    ap.add_argument("--attack-temperature", type=float, default=None)
    ap.add_argument("--diagnostics", choices=["auto", "on", "off"], default="auto",
                    help="only affects the untimed Phase 1B anchor call")
    ap.add_argument("--dry-run", action="store_true", help="print order, estimate and current thermal state; no torch")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-context", help=argparse.SUPPRESS)
    ap.add_argument("--worker-threads", type=int, default=1, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.repeats < 1 or (args.warmup_batches < 2 and not args.worker):
        p1a.abort("--repeats>=1 and --warmup-batches>=2 required")
    if args.cooldown_poll_seconds <= 0 or args.in_cell_temp_poll_seconds <= 0 or args.max_cooldown_seconds <= 0:
        p1a.abort("poll intervals and --max-cooldown-seconds must be > 0")
    if args.worker:
        worker_main(args)
    else:
        parent_main(args)


if __name__ == "__main__":
    main()
