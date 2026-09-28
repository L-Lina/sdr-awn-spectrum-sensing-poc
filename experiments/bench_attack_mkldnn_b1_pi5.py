"""
Phase 1D-3 (attack-acceleration Experiment #1): mkldnn OFF only around the B=1 attack call.
Raspberry Pi 5, CPU only. EXPERIMENT ONLY -- the formal pipeline is not touched.

Basis (Phase 1D-2, results/accel_phase1d2_backend_20260927_124938, all blocks THERMAL_CLEAN)
------------------------------------------------------------------------------------------
At B=1 oneDNN executes only the four forward GEMMs of AWN (matmul gemm:acl, each with a jit:uni
weight reorder per call); convolutions run on the ATen slow path and backward GEMMs outside oneDNN.
Forward-only microbenchmark: mkldnn ON 2.568 ms vs OFF 1.962 ms median (OFF faster in 20/20 reps,
no overlap), max |logit diff| 1.34e-5, not bitwise equal, argmax equal. At B=32 OFF was 4.24x SLOWER,
so this experiment is strictly B=1. Stable Phase 1D-0b B=1 intra4 references: PGD 43.04 ms,
BIM 44.24 ms.

Question
--------
Does running ONLY `atk(x, y)` with mkldnn disabled reduce true B=1 attack latency for PGD/BIM
eps=0.03 while preserving attack semantics?

Conditions
----------
PGD and BIM eps=0.03 (formal profiles), B=1, intra 4 / interop 1 (one persistent worker process,
set before any torch work), the validated 88-sample subset, the Phase 1B optimized path (clean
prediction reused, untimed), attack temperature 1.0 (asserted), ControlledStartPGD with the same
per-sample start tensors for ON and OFF, a fresh attack object per call, nothing else changed.

Scope of the setting
--------------------
Inside the timed call:  prev = torch._C._get_mkldnn_enabled(); set(setting); verify;
atk(x, y); finally set(prev). ON cells go through exactly the same set/restore statements with
setting=True, so both conditions carry identical switching overhead. The global setting is checked
to be back to its original value (True) after every call; clean inference, attacked-prediction
inference and everything outside atk() always run with the default (ON).

Correctness (mkldnn ON is the anchor)
-------------------------------------
Setup (untimed, fail closed): ON path == validated Phase 1B path bit-for-bit for all 88 samples
(stock PGD under torch.manual_seed(formal_seed) vs ControlledStartPGD; BIM anchor sha256 == formal
adversarial_sha256). Pre-validation: every OFF output (88 per attack) is classified against its ON
anchor:
  BIT_IDENTICAL | BITWISE_DIFFERENT_SEMANTIC_SAME | SEMANTIC_DIFFERENT | INVALID
Semantic same = same attacked prediction AND same attack_success AND finite AND shape/dtype OK AND
normalized (<= eps + 1e-6) and IQ (<= eps*b_i*(1+1e-5)+1e-7) eps bounds (Phase 1C v2 tolerances).
Every timed OFF output is classified again; every timed ON output must be bit-identical to its ON
anchor (otherwise the run aborts as non-deterministic). OFF repeatability (bitwise equal to the
first OFF output of that sample) is recorded.
STOP: any SEMANTIC_DIFFERENT -> that attack is FAIL, its remaining cells are skipped, no performance
claim and no recommendation for it. INVALID -> abort.

Order / pairing (deterministic, no RNG)
---------------------------------------
One replication = two 4-cell blocks; a pair = adjacent (1st,2nd) or (3rd,4th) cells of a block:
    even replication:  PGD ON OFF OFF ON | BIM OFF ON ON OFF
    odd  replication:  exact mirror      (BIM OFF ON ON OFF | PGD ON OFF OFF ON)
With an even number of replications every (attack, setting) has the same mean run position, each
setting runs first in half of the pairs, and each attack runs first in half of the replications.
Default 4 replications -> 8 ON/OFF pairs per attack.

Thermal control (Phase 1D-0b protocol, nothing on the system is changed)
------------------------------------------------------------------------
release: 2 consecutive polls <= 55 C and get_throttled bits 0-3 clear; start gate: fresh snapshot
<= 55 + 1.0 C and bits 0-3 clear (else cool again, max --max-start-rejections); in-cell sampling of
temperature (1 s) and get_throttled + measure_clock arm (5 s) from the parent process; the 1D-0
THERMAL_CLEAN classifier; a THERMAL_REVIEW cell is kept (accepted=False) and re-run, up to
--max-cell-attempts; only accepted THERMAL_CLEAN cells enter statistics.

Timing: normal wall-clock latency (time.perf_counter); no torch.profiler. B=1 latency = the Phase
1B attack_block total (prep + construct + scoped atk + post), as in Phases 1B-1D-0b; call_ms = atk
only.

Decision (per attack; mkldnn OFF is recommended for B=1 attack generation only if ALL hold)
  1. zero SEMANTIC_DIFFERENT (pre-validation and timed)
  2. every planned cell has an accepted THERMAL_CLEAN attempt
  3. OFF faster in every valid pair
  4. practically useful: median paired reduction >= --min-useful-reduction-ms (default 1.0 ms)
Also reported: whether the OFF (and ON) pooled median is <= 40 ms. Nothing is applied to clean
inference, B>1, batch experiments or the formal pipeline.

Run on the Pi (from the repo root):
    .venv/bin/python experiments/bench_attack_mkldnn_b1_pi5.py --dry-run
    .venv/bin/python experiments/bench_attack_mkldnn_b1_pi5.py
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import bench_attack_accel_pi5 as p1a  # noqa: E402 -- Phase 1A helpers (no torch import at module level)
import bench_attack_overhead_pi5 as p1b  # noqa: E402 -- Phase 1B validated attack path (anchors)
import bench_attack_batch_pi5_v2 as p1c  # noqa: E402 -- ControlledStartPGD, V2 tolerances, stats helpers
import bench_attack_thermal_control_pi5 as p1d0  # noqa: E402 -- thermal snapshot / sampler / classifier
import bench_attack_thermal_pair_pi5 as p1d0b  # noqa: E402 -- cooldown release, start gate, sign test

REPO_ROOT = p1a.REPO_ROOT
ATTACKS = ["pgd", "bim"]
SETTINGS = ["ON", "OFF"]
INTRA_THREADS = 4
INTEROP_THREADS = 1
TEMPERATURE = 1.0
CLASSES = p1c.CLASSES
NORM_EPS_TOL, IQ_EPS_REL_TOL, IQ_EPS_ABS_TOL = p1c.NORM_EPS_TOL, p1c.IQ_EPS_REL_TOL, p1c.IQ_EPS_ABS_TOL
BUDGET_MS = p1c.BUDGET_MS
PHASE1D0B_REF_MS = {"pgd": 43.0392985, "bim": 44.2387895}
PHASE1D2_DIR = "results/accel_phase1d2_backend_20260927_124938"

LAYOUT_EVEN = [("pgd", "ON"), ("pgd", "OFF"), ("pgd", "OFF"), ("pgd", "ON"),
               ("bim", "OFF"), ("bim", "ON"), ("bim", "ON"), ("bim", "OFF")]
LAYOUT_ODD = list(reversed(LAYOUT_EVEN))

EST_CALL_S = 0.044
EST_SETUP_S = 60.0

RAW_FIELDS = ["attempt_id", "planned_cell_id", "order_index", "attack", "setting", "rep", "sample_pos",
              "modulation", "snr_db", "sample_index", "total_ms", "prep_ms", "construct_ms", "call_ms", "post_ms",
              "classification", "bit_identical_to_on_anchor", "off_repeatable", "mkldnn_restored", "cpu_temp_c"]
SAMPLE_FIELDS = ["phase", "attempt_id", "attack", "setting", "rep", "sample_pos", "modulation", "snr_db",
                 "sample_index", "formal_seed", "label", "clean_pred", "attacked_pred", "on_attacked_pred",
                 "attacked_pred_equal", "attack_success", "on_attack_success", "attack_success_equal",
                 "iq_linf", "on_iq_linf", "linf_abs_diff", "linf_rel_diff", "iq_l2", "on_iq_l2", "l2_abs_diff",
                 "l2_rel_diff", "iq_linf_normalized", "norm_eps_bound_ok", "per_sample_b", "iq_linf_bound_eps_x_b",
                 "iq_eps_bound_ok", "finite", "shape", "dtype", "shape_dtype_ok", "x_adv_sha256", "on_sha256",
                 "bit_identical_to_on_anchor", "max_abs_diff_vs_on", "n_diff_elements", "off_repeatable",
                 "classification"]


# --------------------------------------------------------------------------- plan (no torch)
def build_plan(replications: int) -> List[dict]:
    plan = []
    for rep in range(replications):
        layout = LAYOUT_EVEN if rep % 2 == 0 else LAYOUT_ODD
        for j, (attack, setting) in enumerate(layout):
            bp = j % 4
            plan.append({"replication": rep, "attack": attack, "setting": setting,
                         "pair_id": f"r{rep}_{attack}_p{bp // 2}", "pair_role": "first" if bp % 2 == 0 else "second"})
    for i, c in enumerate(plan):
        c["planned_index"] = i
        c["planned_cell_id"] = f"{i:02d}_{c['attack']}_B1_mkldnn{c['setting']}"
    return plan


def plan_balance(plan: List[dict]) -> dict:
    out = {}
    for a in ATTACKS:
        for s in SETTINGS:
            pos = [c["planned_index"] for c in plan if c["attack"] == a and c["setting"] == s]
            out[f"{a}/{s}"] = {"positions": pos, "mean_position": float(np.mean(pos)) if pos else None}
    firsts = {s: sum(1 for c in plan if c["pair_role"] == "first" and c["setting"] == s) for s in SETTINGS}
    first_attack = {a: sum(1 for r in {c["replication"] for c in plan}
                           if next(c for c in plan if c["replication"] == r)["attack"] == a) for a in ATTACKS}
    return {"mean_positions": out, "pairs_where_setting_runs_first": firsts,
            "replications_where_attack_runs_first": first_attack}


def estimate(plan: List[dict], args) -> dict:
    per_cell = (args.warmup_samples + args.repeats * 88) * EST_CALL_S * 1.1
    preval = 88 * 2 * 3 * EST_CALL_S
    active = (EST_SETUP_S + preval + len(plan) * per_cell) / 60
    cool = len(plan) * args.expected_cooldown_seconds / 60
    return {"n_planned_cells": len(plan), "cell_seconds": per_cell, "active_minutes": active,
            "cooldown_expected_minutes": cool,
            "cooldown_basis": f"{len(plan)} waits x {args.expected_cooldown_seconds} s (planning assumption; Phase "
                              "1D-0b waits were 70-175 s after 24 s cells, these cells are ~13 s)",
            "wall_expected_minutes": active + cool,
            "wall_worst_case_no_retries_minutes": active + len(plan) * args.max_cooldown_seconds / 60}


# --------------------------------------------------------------------------- worker (torch)
def worker_main(args) -> None:
    proto = os.fdopen(os.dup(1), "w", buffering=1)  # JSON-line protocol; all prints go to stderr
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(msg: dict) -> None:
        proto.write(json.dumps(msg, default=str) + "\n")
        proto.flush()

    import torch
    torch.set_num_interop_threads(INTEROP_THREADS)
    torch.set_num_threads(INTRA_THREADS)
    intra, interop = torch.get_num_threads(), torch.get_num_interop_threads()
    if (intra, interop) != (INTRA_THREADS, INTEROP_THREADS):
        p1a.abort(f"thread settings not applied: intra={intra} interop={interop}")
    import torch.nn as nn
    import src.adapters.attack_adapter as am
    from src.adapters.attack_adapter import AttackAdapter, _REAL_ATTACK_SOURCE
    from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE

    get_mk = getattr(torch._C, "_get_mkldnn_enabled", None)
    set_mk = getattr(torch._C, "_set_mkldnn_enabled", None)
    if get_mk is None or set_mk is None:
        p1a.abort("torch._C._get/_set_mkldnn_enabled unavailable; cannot scope the setting explicitly")
    default_mk = bool(get_mk())
    if default_mk is not True:
        p1a.abort(f"mkldnn is not enabled by default in this process ({default_mk}); ON anchor would be wrong")

    ctx = json.loads(Path(args.worker_context).read_text())
    out_dir = Path(args.out_dir)
    settings = ctx["attack_settings"]
    subset = [tuple(k) for k in ctx["subset"]]

    awn = AWNModelAdapter(checkpoint_path=str(REPO_ROOT / ctx["checkpoint_path"]), device="cpu")
    if awn.model is None or awn.backend_name != _REAL_MODEL_SOURCE or awn.status != "ok":
        p1a.abort(f"AWN backend is not real: {awn.backend_name} {awn.status}")
    adapter = AttackAdapter(awn_model=awn.model, device="cpu")
    if adapter.wrapped_model is None or adapter.backend_name != _REAL_ATTACK_SOURCE or adapter.status != "ok":
        p1a.abort("attack backend is not real")
    if am._torchattacks is None or awn.model.training:
        p1a.abort("torchattacks unavailable or model in train mode")
    wm = adapter.wrapped_model
    ControlledStartPGD, _ = p1c.make_controlled_pgd(am._torchattacks, torch, nn)
    base_rows, formal_attack = p1a.load_formal_rows(REPO_ROOT / ctx["formal_dir"], set(subset))
    inputs = p1a.rebuild_clean_inputs(subset, base_rows, ctx["sensing"], REPO_ROOT / ctx["dataset_path"])

    def infer_pred(x) -> int:  # always with the default (mkldnn ON) setting
        if bool(get_mk()) is not default_mk:
            p1a.abort("mkldnn setting leaked outside the attack call")
        logits, meta = awn.infer(x)
        if meta["awn_backend"] != _REAL_MODEL_SOURCE or meta["awn_status"] != "ok" or not np.isfinite(logits).all():
            p1a.abort(f"AWN inference invalid: {meta}")
        return int(np.argmax(logits[0]))

    for key, inp in inputs.items():
        inp["clean_pred"] = infer_pred(inp["x"])
        if inp["clean_pred"] != inp["formal_clean_pred"]:
            p1a.abort(f"{key}: clean_pred != formal")
        if p1b.in_path_clean_pred(am, adapter, torch, inp["x"]) != inp["clean_pred"]:
            p1a.abort(f"{key}: apply()-internal y_pred != pipeline clean_pred")

    req_grad_ref = [p.requires_grad for p in wm.parameters()]

    def check_state(where: str) -> None:
        if awn.model.training or wm.training:
            p1a.abort(f"model entered train mode ({where})")
        if (torch.get_num_threads(), torch.get_num_interop_threads()) != (intra, interop):
            p1a.abort(f"torch thread settings drifted ({where})")
        if [p.requires_grad for p in wm.parameters()] != req_grad_ref:
            p1a.abort(f"requires_grad flags not restored ({where})")
        if bool(get_mk()) is not default_mk:
            p1a.abort(f"mkldnn global setting not restored ({where})")

    s_pgd = settings["pgd"]
    stock = am._build_torchattacks("pgd", am.TemperatureLogitsWrapper(wm, s_pgd["temperature"]), s_pgd["eps"],
                                   attack_params=dict(s_pgd["attack_params"]))
    pgd_hp = {"eps": stock.eps, "alpha": stock.alpha, "steps": stock.steps, "random_start": stock.random_start}
    noise = {}
    for key in subset:  # one tensor per sample, reused unchanged by ON and OFF
        g = torch.Generator().manual_seed(inputs[key]["seed"])
        noise[key] = torch.empty((1, 2, inputs[key]["x"].shape[2], 1), dtype=torch.float32).uniform_(
            -pgd_hp["eps"], pgd_hp["eps"], generator=g)

    def build_attack(attack, attack_model, key):
        s = settings[attack]
        if attack != "pgd":
            return am._build_torchattacks(attack, attack_model, s["eps"], attack_params=dict(s["attack_params"]))
        atk = ControlledStartPGD(attack_model, eps=pgd_hp["eps"], alpha=pgd_hp["alpha"], steps=pgd_hp["steps"],
                                 random_start=pgd_hp["random_start"])
        atk.set_init_noise(noise[key])
        return atk

    def attack_block(attack, key, mkldnn_on: bool):
        """Phase 1B optimized path at B=1; the mkldnn setting is scoped to atk(x, y) only."""
        s = settings[attack]
        pc = time.perf_counter
        t0 = pc()
        x = inputs[key]["x"]
        training_before = wm.training
        orig_requires_grad = [p.requires_grad for p in wm.parameters()]
        orig_param_devices = [p.device for p in wm.parameters()]
        try:
            x_ta, a, b = am._iq_to_ta_input_minmax(torch.from_numpy(x).to(adapter.device))
            wm.set_minmax(a, b)
            y = torch.tensor([inputs[key]["clean_pred"]], dtype=torch.long)
            t1 = pc()
            atk = build_attack(attack, am.TemperatureLogitsWrapper(wm, s["temperature"]), key)
            t2 = pc()
            prev = bool(get_mk())
            set_mk(mkldnn_on)
            try:
                if bool(get_mk()) is not mkldnn_on:
                    raise RuntimeError("mkldnn setting did not take effect")
                x_ta_adv = atk(x_ta, y)
            finally:
                set_mk(prev)
            t3 = pc()
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
        t4 = pc()
        return x_adv, lin, b_np, {"total_ms": (t4 - t0) * 1e3, "prep_ms": (t1 - t0) * 1e3,
                                  "construct_ms": (t2 - t1) * 1e3, "call_ms": (t3 - t2) * 1e3,
                                  "post_ms": (t4 - t3) * 1e3}, bool(get_mk()) is default_mk

    # ---------------------------------------------------------------- ON anchors (fail closed)
    anchors: Dict[str, Dict[tuple, dict]] = {}
    fidelity = {}
    for attack in ATTACKS:
        anchors[attack] = {}
        for key in subset:
            inp = inputs[key]
            torch.manual_seed(inp["seed"])
            ref, _, _, y_used, _, _ = p1b.attack_path(am, adapter, torch, inp["x"], attack, settings[attack],
                                                      inp["clean_pred"], False, False)
            check_state(f"anchor {attack} {key}")
            on, lin, b_np, _, _ = attack_block(attack, key, True)
            check_state(f"anchor-on {attack} {key}")
            if y_used != inp["clean_pred"] or not np.array_equal(ref, on):
                p1a.abort(f"{attack} {key}: mkldnn-ON benchmark path is not bit-identical to the Phase 1B path")
            sha = p1a.sha256_array(on)
            if attack in p1a.DETERMINISTIC_ATTACKS:
                if sha != formal_attack[key + (attack, settings[attack]["profile"])].get("adversarial_sha256"):
                    p1a.abort(f"{attack} {key}: ON anchor sha256 != formal adversarial_sha256")
            pred = infer_pred(on)
            d = on - inp["x"]
            anchors[attack][key] = {"x_adv": on, "sha": sha, "pred": pred,
                                    "success": bool(inp["clean_pred"] == inp["label"] and pred != inp["label"]),
                                    "linf": float(np.max(np.abs(d))), "l2": float(np.linalg.norm(d.reshape(-1)))}
        fidelity[attack] = f"{len(subset)}/{len(subset)} mkldnn-ON path bit-identical to Phase 1B path" + (
            " (stock PGD under torch.manual_seed(formal_seed) vs ControlledStartPGD)" if attack == "pgd"
            else "; ON anchor sha256 == formal adversarial_sha256 for all")

    smp_f = (out_dir / "per_sample_worker.csv").open("w", newline="")
    raw_f = (out_dir / "raw_worker.csv").open("w", newline="")
    smp_w = csv.DictWriter(smp_f, fieldnames=SAMPLE_FIELDS)
    raw_w = csv.DictWriter(raw_f, fieldnames=RAW_FIELDS)
    smp_w.writeheader()
    raw_w.writeheader()
    off_first_sha: Dict[str, Dict[tuple, str]] = {a: {} for a in ATTACKS}
    failed: Dict[str, List[dict]] = {a: [] for a in ATTACKS}

    def classify(phase, attempt_id, attack, setting, rep, pos, key, x_adv, lin, b_np) -> tuple:
        inp, anc = inputs[key], anchors[attack][key]
        eps = float(settings[attack]["eps"])
        xi = np.ascontiguousarray(x_adv[0:1])
        shape_ok = xi.shape == anc["x_adv"].shape and xi.dtype == np.float32
        fin = bool(np.isfinite(xi).all())
        bit = bool(np.array_equal(xi, anc["x_adv"]))
        md = float(np.max(np.abs(xi.astype(np.float64) - anc["x_adv"].astype(np.float64)))) if fin else float("inf")
        pred = anc["pred"] if bit else (infer_pred(xi) if fin else -1)
        succ = bool(inp["clean_pred"] == inp["label"] and pred != inp["label"])
        dlt = xi - inp["x"]
        linf, l2 = float(np.max(np.abs(dlt))), float(np.linalg.norm(dlt.reshape(-1)))
        bound = eps * float(b_np[0])
        norm_ok = bool(lin[0] <= eps + NORM_EPS_TOL)
        iq_ok = bool(linf <= bound * (1 + IQ_EPS_REL_TOL) + IQ_EPS_ABS_TOL)
        pred_eq, succ_eq = pred == anc["pred"], succ == anc["success"]
        if not (fin and shape_ok and norm_ok and iq_ok):
            cls = "INVALID"
        elif bit:
            cls = "BIT_IDENTICAL"
        elif pred_eq and succ_eq:
            cls = "BITWISE_DIFFERENT_SEMANTIC_SAME"
        else:
            cls = "SEMANTIC_DIFFERENT"
        sha = p1a.sha256_array(xi)
        rep_ok = ""
        if setting == "OFF":
            first = off_first_sha[attack].setdefault(key, sha)
            rep_ok = sha == first
        smp_w.writerow({
            "phase": phase, "attempt_id": attempt_id, "attack": attack, "setting": setting, "rep": rep,
            "sample_pos": pos, "modulation": key[0], "snr_db": key[1], "sample_index": key[2],
            "formal_seed": inp["seed"], "label": inp["label"], "clean_pred": inp["clean_pred"],
            "attacked_pred": pred, "on_attacked_pred": anc["pred"], "attacked_pred_equal": pred_eq,
            "attack_success": succ, "on_attack_success": anc["success"], "attack_success_equal": succ_eq,
            "iq_linf": linf, "on_iq_linf": anc["linf"], "linf_abs_diff": abs(linf - anc["linf"]),
            "linf_rel_diff": p1c._rel(abs(linf - anc["linf"]), anc["linf"]), "iq_l2": l2, "on_iq_l2": anc["l2"],
            "l2_abs_diff": abs(l2 - anc["l2"]), "l2_rel_diff": p1c._rel(abs(l2 - anc["l2"]), anc["l2"]),
            "iq_linf_normalized": float(lin[0]), "norm_eps_bound_ok": norm_ok, "per_sample_b": float(b_np[0]),
            "iq_linf_bound_eps_x_b": bound, "iq_eps_bound_ok": iq_ok, "finite": fin,
            "shape": "x".join(map(str, xi.shape)), "dtype": str(xi.dtype), "shape_dtype_ok": shape_ok,
            "x_adv_sha256": sha, "on_sha256": anc["sha"], "bit_identical_to_on_anchor": bit,
            "max_abs_diff_vs_on": md, "n_diff_elements": int(np.sum(xi != anc["x_adv"])) if shape_ok else -1,
            "off_repeatable": rep_ok, "classification": cls})
        if cls == "INVALID":
            smp_f.flush()
            raw_f.flush()
            (out_dir / "invalid.json").write_text(json.dumps(
                {"attempt_id": attempt_id, "attack": attack, "setting": setting, "sample": list(key), "finite": fin,
                 "shape_dtype_ok": shape_ok, "norm_eps_bound_ok": norm_ok, "iq_eps_bound_ok": iq_ok}, indent=2))
            p1a.abort(f"INVALID output {attack} {setting} {key}")
        if cls == "SEMANTIC_DIFFERENT":
            failed[attack].append({"phase": phase, "attempt_id": attempt_id, "sample": list(key),
                                   "on_pred": anc["pred"], "off_pred": pred, "on_success": anc["success"],
                                   "off_success": succ, "max_abs_diff": md})
        return cls, bit, rep_ok

    # ---------------------------------------------------------------- OFF pre-validation (untimed)
    preval = {}
    for attack in ATTACKS:
        counts = {c: 0 for c in CLASSES}
        for pos, key in enumerate(subset):
            x_adv, lin, b_np, _, restored = attack_block(attack, key, False)
            check_state(f"preval {attack} {key}")
            cls, _, _ = classify("prevalidation", "", attack, "OFF", "", pos, key, x_adv, lin, b_np)
            counts[cls] += 1
        preval[attack] = {"counts": counts, "status": "FAIL" if counts["SEMANTIC_DIFFERENT"] else "PASS"}
    smp_f.flush()
    send({"event": "ready", "pid": os.getpid(), "intra_threads": intra, "interop_threads": interop,
          "anchor_fidelity": fidelity, "prevalidation": preval,
          "failed_samples": {a: failed[a] for a in ATTACKS}, "mkldnn_default": default_mk,
          "torch_version": torch.__version__, "pgd_hyperparameters": pgd_hp})

    # ---------------------------------------------------------------- cells
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        msg = json.loads(line)
        if msg.get("cmd") == "quit":
            break
        cell = msg["cell"]
        attack, setting = cell["attack"], cell["setting"]
        on = setting == "ON"
        n_fail_before = len(failed[attack])
        counts = {c: 0 for c in CLASSES}
        t_w0 = time.perf_counter()
        for key in subset[: args.warmup_samples]:
            attack_block(attack, key, on)
        check_state(f"warmup {cell['attempt_id']}")
        t_w1 = time.perf_counter()
        for rep in range(args.repeats):
            for pos, key in enumerate(subset):
                temp = p1a.cpu_temp_c()
                x_adv, lin, b_np, tm, restored = attack_block(attack, key, on)
                check_state(f"{cell['attempt_id']} {key}")
                if on:
                    bit = bool(np.array_equal(x_adv, anchors[attack][key]["x_adv"]))
                    if not bit:
                        p1a.abort(f"mkldnn-ON output not bit-identical to its ON anchor ({attack} {key}); "
                                  "the ON path is not deterministic -- comparison invalid")
                    cls, rep_ok = "BIT_IDENTICAL", ""
                else:
                    cls, bit, rep_ok = classify("timed", cell["attempt_id"], attack, setting, rep, pos, key,
                                                x_adv, lin, b_np)
                counts[cls] += 1
                raw_w.writerow({"attempt_id": cell["attempt_id"], "planned_cell_id": cell["planned_cell_id"],
                                "order_index": cell["order_index"], "attack": attack, "setting": setting, "rep": rep,
                                "sample_pos": pos, "modulation": key[0], "snr_db": key[1], "sample_index": key[2],
                                **tm, "classification": cls, "bit_identical_to_on_anchor": bit,
                                "off_repeatable": rep_ok, "mkldnn_restored": restored, "cpu_temp_c": temp})
            raw_f.flush()
            smp_f.flush()
        send({"event": "cell_done", "attempt_id": cell["attempt_id"], "class_counts": counts,
              "new_semantic_different": failed[attack][n_fail_before:], "warmup_s": t_w1 - t_w0,
              "timed_s": time.perf_counter() - t_w1})
    raw_f.close()
    smp_f.close()
    send({"event": "bye"})


# --------------------------------------------------------------------------- parent: aggregation
def _st(vals: List[float]) -> dict:
    st = p1a.stats(vals)
    return {k: st[k] for k in ("n", "mean", "median", "p95", "p99", "std", "min", "max")}


def aggregate(out_dir: Path, plan: List[dict], attempts: List[dict], ready: dict, args) -> tuple:
    with (out_dir / "raw_worker.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    acc_map = {a["attempt_id"]: a for a in attempts}
    for r in rows:
        a = acc_map.get(r["attempt_id"], {})
        r["accepted"] = a.get("accepted", False)
        r["pair_id"] = a.get("pair_id", "")
    rows.sort(key=lambda r: (int(r["order_index"]), int(r["rep"]), int(r["sample_pos"])))
    with (out_dir / "raw_measurements.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS + ["accepted", "pair_id"])
        w.writeheader()
        w.writerows(rows)
    with (out_dir / "per_sample_worker.csv").open(newline="") as f:
        samples = list(csv.DictReader(f))
    with (out_dir / "per_sample_validation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS)
        w.writeheader()
        w.writerows(samples)

    summary = []
    for a in attempts:
        cr = [r for r in rows if r["attempt_id"] == a["attempt_id"]]
        tot = [float(r["total_ms"]) for r in cr]
        a["_tot"] = tot
        th = a["thermal"]
        summary.append({
            "attempt_id": a["attempt_id"], "planned_cell_id": a["planned_cell_id"], "order_index": a["order_index"],
            "replication": a["replication"], "pair_id": a["pair_id"], "attempt": a["attempt"], "attack": a["attack"],
            "setting": a["setting"], "accepted": a["accepted"], "thermal_status": th["status"],
            "thermal_reasons": " | ".join(th["reasons"]), "start_rejections": a["start_rejections"],
            "cooldown_waited_s": a["cooldown"]["waited_s"], "temp_before_c": th["before"]["temp_c"],
            "temp_after_c": th["after"]["temp_c"], "temp_max_in_cell_c": th["temp_max_in_cell_c"],
            "throttled_before": th["before"]["throttled"]["value_hex"],
            "throttled_after": th["after"]["throttled"]["value_hex"],
            "throttled_low_or_in_cell": th["throttled_low_or_in_cell"], "n_in_cell_vc_samples": th["n_in_cell_vc_samples"],
            "throttled_high_before": th["before"]["throttled"]["high"], "throttled_high_after": th["after"]["throttled"]["high"],
            "arm_clock_hz_min_in_cell": th["arm_clock_hz_min_in_cell"], "arm_clock_hz_max_in_cell": th["arm_clock_hz_max_in_cell"],
            "governor_unchanged": th["before"]["governor"] == th["after"]["governor"],
            **{f"n_{c.lower()}": a["class_counts"][c] for c in CLASSES},
            **p1c._stat_cols("b1_latency_ms", tot),
            **p1c._stat_cols("call_ms", [float(r["call_ms"]) for r in cr]),
            "warmup_s": a["warmup_s"], "timed_s": a["timed_s"]})
    write_csv(out_dir / "summary.csv", summary)

    accepted = {a["planned_cell_id"]: a for a in attempts if a["accepted"]}
    cond = {}
    for attack in ATTACKS:
        for s in SETTINGS:
            acc = [accepted[c["planned_cell_id"]] for c in plan if c["attack"] == attack and c["setting"] == s
                   and c["planned_cell_id"] in accepted]
            vals = [v for a in acc for v in a["_tot"]]
            st = _st(vals)
            cond[(attack, s)] = {
                "attack": attack, "setting": s, "n_planned_cells": sum(1 for c in plan if c["attack"] == attack
                                                                       and c["setting"] == s),
                "n_accepted_cells": len(acc), **{f"b1_latency_ms_{k}": v for k, v in st.items()},
                "per_cell_medians_ms": [float(np.median(a["_tot"])) for a in acc],
                "phase1d0b_reference_ms": PHASE1D0B_REF_MS[attack],
                "delta_vs_phase1d0b_ms": (st["median"] - PHASE1D0B_REF_MS[attack]) if acc else None,
                "median_le_40ms": (st["median"] <= BUDGET_MS) if acc else None}

    pair_rows = []
    for pid in dict.fromkeys(c["pair_id"] for c in plan):
        cs = [c for c in plan if c["pair_id"] == pid]
        c_on = next(c for c in cs if c["setting"] == "ON")
        c_off = next(c for c in cs if c["setting"] == "OFF")
        a_on, a_off = accepted.get(c_on["planned_cell_id"]), accepted.get(c_off["planned_cell_id"])
        row = {"pair_id": pid, "attack": c_on["attack"], "replication": c_on["replication"],
               "first_in_pair": "ON" if c_on["pair_role"] == "first" else "OFF", "pair_valid": bool(a_on and a_off)}
        if a_on and a_off:
            m_on, m_off = float(np.median(a_on["_tot"])), float(np.median(a_off["_tot"]))
            row.update(on_median_ms=m_on, off_median_ms=m_off, delta_ms_off_minus_on=m_off - m_on,
                       delta_pct=100.0 * (m_off - m_on) / m_on, ratio_off_over_on=m_off / m_on)
        else:
            row["not_valid_reason"] = f"ON {c_on['final_status']}, OFF {c_off['final_status']}"
        pair_rows.append(row)
    pf = ["pair_id", "attack", "replication", "first_in_pair", "pair_valid", "on_median_ms", "off_median_ms",
          "delta_ms_off_minus_on", "delta_pct", "ratio_off_over_on", "not_valid_reason"]
    with (out_dir / "paired_comparisons.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pf, restval="")
        w.writeheader()
        w.writerows(pair_rows)

    decisions = {}
    for attack in ATTACKS:
        pre = ready["prevalidation"][attack]["counts"]
        timed_sd = sum(a["class_counts"]["SEMANTIC_DIFFERENT"] for a in attempts if a["attack"] == attack)
        n_sd = pre["SEMANTIC_DIFFERENT"] + timed_sd
        planned = [c for c in plan if c["attack"] == attack]
        not_acc = [f"{c['planned_cell_id']}: {c['final_status']}" for c in planned if c["planned_cell_id"] not in accepted]
        prs = [p for p in pair_rows if p["attack"] == attack and p["pair_valid"]]
        deltas = [p["delta_ms_off_minus_on"] for p in prs]
        nz = [d for d in deltas if d != 0]
        k_off = sum(1 for d in nz if d < 0)
        med_delta = float(np.median(deltas)) if deltas else None
        off_bits = sum(a["class_counts"]["BIT_IDENTICAL"] for a in attempts if a["attack"] == attack and a["setting"] == "OFF")
        off_same = sum(a["class_counts"]["BITWISE_DIFFERENT_SEMANTIC_SAME"] for a in attempts
                       if a["attack"] == attack and a["setting"] == "OFF")
        c1 = n_sd == 0
        c2 = not not_acc
        c3 = bool(prs) and k_off == len(prs)
        c4 = med_delta is not None and med_delta <= -args.min_useful_reduction_ms
        on_c, off_c = cond[(attack, "ON")], cond[(attack, "OFF")]
        d = {
            "status": "FAIL_SEMANTIC_DIFFERENT" if not c1 else "PASS",
            "criteria": {"1_zero_semantic_differences": c1, "2_all_planned_cells_accepted_thermal_clean": c2,
                         "3_off_faster_in_every_valid_pair": c3,
                         f"4_median_paired_reduction_ge_{args.min_useful_reduction_ms}_ms": c4},
            "not_accepted_cells": not_acc,
            "semantic_different_total": n_sd, "prevalidation_counts": pre,
            "timed_off_bit_identical": off_bits, "timed_off_bitwise_different_semantic_same": off_same,
            "on_median_ms": on_c["b1_latency_ms_median"], "off_median_ms": off_c["b1_latency_ms_median"],
            "pooled_ratio_off_over_on": (off_c["b1_latency_ms_median"] / on_c["b1_latency_ms_median"])
            if on_c["n_accepted_cells"] and off_c["n_accepted_cells"] else None,
            "n_valid_pairs": len(prs), "n_pairs_off_faster": k_off, "n_pairs_on_faster": len(nz) - k_off,
            "paired_delta_ms": _st(deltas) if deltas else None, "paired_median_delta_ms": med_delta,
            "paired_delta_pct_median": float(np.median([p["delta_pct"] for p in prs])) if prs else None,
            "paired_ratio_median": float(np.median([p["ratio_off_over_on"] for p in prs])) if prs else None,
            "sign_test_two_sided_p": p1d0b.sign_test_two_sided(k_off, len(nz)),
            "on_median_le_40ms": on_c["median_le_40ms"], "off_median_le_40ms": off_c["median_le_40ms"],
        }
        if not c1:
            d["recommendation"] = "DO_NOT_ADOPT (semantic difference; performance claims withheld)"
            for k in ("on_median_ms", "off_median_ms", "pooled_ratio_off_over_on", "paired_median_delta_ms",
                      "paired_ratio_median", "paired_delta_pct_median"):
                d[k] = None
        elif c1 and c2 and c3 and c4:
            d["recommendation"] = "RECOMMEND mkldnn OFF around the B=1 attack call only (not clean inference, not B>1, " \
                                  "not the formal pipeline without explicit approval)"
        else:
            d["recommendation"] = "DO_NOT_RECOMMEND (criteria not all met: " + ", ".join(
                k for k, v in d["criteria"].items() if not v) + ")"
        decisions[attack] = d

    n_review = sum(1 for a in attempts if a["thermal"]["status"] != "THERMAL_CLEAN")
    validation = {
        "anchor_fidelity": ready["anchor_fidelity"], "prevalidation": ready["prevalidation"],
        "semantic_different_samples": {a: ready["failed_samples"][a] + [x for at in attempts if at["attack"] == a
                                                                        for x in at.get("new_semantic_different", [])]
                                       for a in ATTACKS},
        "planned_cell_final_status": {c["planned_cell_id"]: c["final_status"] for c in plan},
        "n_planned_cells": len(plan), "n_attempts": len(attempts), "n_attempts_thermal_review": n_review,
        "n_start_rejections": sum(c.get("start_rejections_total", 0) for c in plan),
        "per_condition": {f"{k[0]}/{k[1]}": v for k, v in cond.items()},
        "decisions": decisions,
        "overall": "FAIL" if any(d["status"] != "PASS" for d in decisions.values()) else (
            "PASS" if all(c["final_status"] == "ACCEPTED" for c in plan) else "INCOMPLETE"),
        "notes": ["mkldnn setting scoped to atk(x, y) only via torch._C._set_mkldnn_enabled with restore; ON cells use "
                  "the same statements with True.",
                  "ON is the anchor; timed ON outputs must be bit-identical to it (else abort).",
                  "Only accepted (THERMAL_CLEAN) cells enter statistics; pairs = adjacent ON/OFF cells.",
                  "No setting is applied to clean inference, B>1, batch experiments or the formal pipeline."],
    }
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2, default=str))
    for a in attempts:
        a.pop("_tot", None)
    return validation, cond


def write_csv(path: Path, rows: List[dict]) -> None:
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["empty"], restval="")
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- parent
def parent_main(args) -> None:
    plan = build_plan(args.replications)
    est = estimate(plan, args)
    bal = plan_balance(plan)
    for c in plan:
        p1a.log(f"   {c['planned_index']:2d}  {c['attack']:4s} mkldnn {c['setting']:3s} pair={c['pair_id']}")
    p1a.log(f"[parent] balance: {json.dumps(bal)}")
    p1a.log(f"[parent] estimate: active ~{est['active_minutes']:.1f} min + cooldown ~{est['cooldown_expected_minutes']:.0f} "
            f"min = ~{est['wall_expected_minutes']:.0f} min (worst case w/o retries "
            f"{est['wall_worst_case_no_retries_minutes']:.0f} min)")
    now = p1d0.snapshot()
    p1a.log(f"[parent] now: temp={now['temp_c']} throttled={now['throttled']['value_hex']} clock={now['arm_clock_hz']}")
    if args.replications % 2:
        p1a.log("[warn] odd --replications: positions are not exactly balanced")
    if args.dry_run:
        print(json.dumps({"plan": plan, "balance": bal, "estimate": est, "system_now": now}, indent=2, default=str))
        return
    if now["throttled"]["low"] is None:
        p1a.abort("vcgencmd get_throttled unavailable; THERMAL_CLEAN cannot be evaluated")

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (REPO_ROOT / (args.out_dir or f"results/accel_phase1d3_mkldnn_b1_{ts}")).resolve()
    for p in p1a.PROTECTED_RESULT_DIRS + (
            "results/accel_phase1c_v2_formal_20260926_164547", "results/accel_phase1a_formal_20260926_152737",
            "results/accel_phase1b_20260926_155342", "results/accel_phase1d0_thermal_20260927_092156",
            "results/accel_phase1d0b_pair_20260927_111224", "results/accel_phase1d1_profile_20260927_120159",
            PHASE1D2_DIR):
        prot = (REPO_ROOT / p).resolve()
        if out_dir == prot or prot in out_dir.parents or out_dir in prot.parents:
            p1a.abort(f"refusing to write inside/over results directory {p}")
    out_dir.mkdir(parents=True, exist_ok=False)

    ctx = p1a.load_formal_context(args)
    cfg, fman, formal_dir = ctx["cfg"], ctx["manifest"], ctx["formal_dir"]
    subset = p1a.build_subset(cfg)
    _, attack_rows = p1a.load_formal_rows(formal_dir, set(subset))
    settings = p1a.resolve_attack_settings(cfg, attack_rows, args.attack_temperature, args.diagnostics)
    settings = {a: settings[a] for a in ATTACKS}
    for a in ATTACKS:
        if float(settings[a]["temperature"]) != TEMPERATURE:
            p1a.abort(f"{a}: attack temperature {settings[a]['temperature']} != {TEMPERATURE}")
    hashes = {"dataset_sha256": p1a.sha256_file(REPO_ROOT / cfg["dataset_path"]),
              "checkpoint_sha256": p1a.sha256_file(REPO_ROOT / cfg["checkpoint_path"]),
              "config_sha256": p1a.sha256_file(ctx["cfg_path"]),
              "attack_adapter_py_sha256": p1a.sha256_file(REPO_ROOT / "src/adapters/attack_adapter.py"),
              "phase1b_helpers_sha256": p1a.sha256_file(Path(p1b.__file__).resolve()),
              "phase1c_v2_script_sha256": p1a.sha256_file(Path(p1c.__file__).resolve()),
              "phase1d0b_script_sha256": p1a.sha256_file(Path(p1d0b.__file__).resolve()),
              "this_script_sha256": p1a.sha256_file(Path(__file__).resolve())}
    if hashes["dataset_sha256"] != fman.get("dataset_sha256") or hashes["checkpoint_sha256"] != fman.get("checkpoint_sha256"):
        p1a.abort("dataset/checkpoint sha256 differs from formal CPU manifest")
    worker_ctx = {"subset": [list(k) for k in subset], "formal_dir": p1a.rel(formal_dir),
                  "dataset_path": cfg["dataset_path"], "checkpoint_path": cfg["checkpoint_path"],
                  "sensing": p1a.resolve_sensing(cfg), "attack_settings": settings}
    ctx_file = out_dir / "worker_context.json"
    ctx_file.write_text(json.dumps(worker_ctx, indent=2, default=str))

    attempts: List[dict] = []
    trace: List[dict] = []
    manifest = {
        "phase": "accel_phase1d3_mkldnn_b1", "status": "running", "command": [sys.executable] + sys.argv,
        "question": "Does disabling mkldnn only around atk(x, y) reduce true B=1 PGD/BIM latency while preserving "
                    "attack semantics?",
        "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "hashes": hashes,
        "environment": p1a.environment_metadata(), "system_at_start": now, "attacks": settings,
        "threads": {"intra": INTRA_THREADS, "interop": INTEROP_THREADS}, "batch_size": 1,
        "scope": "torch._C._set_mkldnn_enabled(setting) immediately before atk(x, y), restored in finally; ON uses "
                 "the same statements; clean/attacked-prediction inference always default (ON)",
        "replications": args.replications, "repeats": args.repeats, "warmup_samples": args.warmup_samples,
        "order": {"layout_even": LAYOUT_EVEN, "layout_odd": LAYOUT_ODD, "balance": bal},
        "decision_rule": {"min_useful_reduction_ms": args.min_useful_reduction_ms,
                          "criteria": ["zero SEMANTIC_DIFFERENT", "all planned cells accepted THERMAL_CLEAN",
                                       "OFF faster in every valid pair", "median paired reduction >= threshold"]},
        "references": {"phase1d0b_b1_median_ms": PHASE1D0B_REF_MS, "phase1d2_dir": PHASE1D2_DIR},
        "thermal_policy": {"max_start_temp_c": args.max_start_temp_c,
                           "start_gate_allowance_c": p1d0.START_TEMP_JITTER_C,
                           "release_consecutive_reads": args.release_consecutive_reads,
                           "cooldown_poll_seconds": args.cooldown_poll_seconds,
                           "max_cooldown_seconds": args.max_cooldown_seconds,
                           "max_start_rejections": args.max_start_rejections,
                           "max_cell_attempts": args.max_cell_attempts,
                           "in_cell_temp_poll_seconds": 1.0, "in_cell_vc_poll_seconds": args.in_cell_vc_poll_seconds,
                           "system_settings_changed": "none (read-only)"},
        "estimate": est, "plan": plan, "attempts": attempts,
    }
    man_path = out_dir / "manifest.json"
    p1d0._write_json(man_path, manifest)

    def save():
        p1d0._write_json(man_path, manifest)
        with (out_dir / "thermal_trace.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=p1d0b.TRACE_FIELDS, restval="")
            w.writeheader()
            w.writerows(trace)

    errf = (out_dir / "worker.stderr.log").open("w")
    wcmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--out-dir", str(out_dir), "--worker-context",
            str(ctx_file), "--repeats", str(args.repeats), "--warmup-samples", str(args.warmup_samples)]
    worker = subprocess.Popen(wcmd, cwd=REPO_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf,
                              text=True, bufsize=1, env=os.environ.copy())

    def fail(msg: str):
        if worker.poll() is None:
            worker.kill()
        inv = out_dir / "invalid.json"
        p1d0._write_json(out_dir / "validation.json", {"overall": f"FAIL ({msg})",
                                                       "invalid": json.loads(inv.read_text()) if inv.exists() else None})
        manifest.update(status="aborted", abort_reason=msg, finished_utc=_dt.datetime.now(_dt.timezone.utc).isoformat())
        save()
        p1a.abort(msg)

    def recv() -> dict:
        line = worker.stdout.readline()
        if not line:
            fail(f"worker exited (rc={worker.wait()}) -- INVALID output, non-deterministic ON path, failed fail-closed "
                 "check or crash; see worker.stderr.log")
        return json.loads(line)

    t0 = time.monotonic()
    ready = recv()
    if ready.get("event") != "ready":
        fail(f"unexpected worker message {ready}")
    manifest["worker"] = {**ready, "setup_seconds": time.monotonic() - t0}
    attack_failed = {a: ready["prevalidation"][a]["status"] == "FAIL" for a in ATTACKS}
    for a in ATTACKS:
        p1a.log(f"[parent] pre-validation {a}: {ready['prevalidation'][a]}")
    save()

    exec_index, consecutive_skips = 0, 0
    for c in plan:
        c["start_rejections_total"] = 0
        c["final_status"] = "NOT_RUN"
        pid = c["planned_cell_id"]
        if attack_failed[c["attack"]]:
            c["final_status"] = "SKIPPED_ATTACK_FAILED_SEMANTIC"
            continue
        for attempt in range(1, args.max_cell_attempts + 1):
            rejections, cd, before = 0, None, None
            while True:
                p1a.log(f"[parent] {pid} attempt {attempt}: cooling (now {p1a.cpu_temp_c()} C)")
                cd = p1d0b.wait_until_cool(args, trace, pid)
                if cd["outcome"] != "REACHED":
                    break
                before = p1d0.snapshot()
                reasons = p1d0b.start_gate(before, args.max_start_temp_c)
                if not reasons:
                    break
                rejections += 1
                c["start_rejections_total"] += 1
                c.setdefault("start_rejection_log", []).append({"attempt": attempt, "reasons": reasons})
                if rejections > args.max_start_rejections:
                    cd = {**cd, "outcome": "START_GATE_EXHAUSTED"}
                    break
            if cd["outcome"] != "REACHED":
                if args.on_cooldown_timeout == "abort" or cd["outcome"] == "TEMPERATURE_UNAVAILABLE":
                    fail(f"{pid}: {cd['outcome']}")
                c["final_status"] = f"SKIPPED_{cd['outcome']}"
                break
            attempt_id = f"{pid}_a{attempt}"
            sampler = p1d0.CellSampler(attempt_id, exec_index, 1.0, args.in_cell_vc_poll_seconds).start()
            worker.stdin.write(json.dumps({"cmd": "run", "cell": {
                "attempt_id": attempt_id, "planned_cell_id": pid, "order_index": exec_index,
                "attack": c["attack"], "setting": c["setting"]}}) + "\n")
            worker.stdin.flush()
            msg = recv()
            samples = sampler.stop()
            after = p1d0.snapshot()
            if msg.get("event") != "cell_done" or msg.get("attempt_id") != attempt_id:
                fail(f"unexpected reply for {attempt_id}: {msg}")
            for s_ in samples:
                trace.append({"phase": "cell", "planned_cell_id": pid, "attempt_id": attempt_id,
                              "order_index": exec_index, **{k: s_[k] for k in p1d0b.TRACE_FIELDS[4:]}})
            status, treasons = p1d0.classify_thermal(before, after, samples, cd, args.max_start_temp_c)
            temps = [s_["temp_c"] for s_ in samples if s_["temp_c"] is not None] + \
                    [v for v in (before["temp_c"], after["temp_c"]) if v is not None]
            clocks = [int(s_["arm_clock_hz"]) for s_ in samples if s_["arm_clock_hz"] not in ("", None)]
            lows = [int(s_["throttled_low"]) for s_ in samples if s_["throttled_low"] not in ("", None)]
            low_or = 0
            for v in lows + [x for x in (before["throttled"]["low"], after["throttled"]["low"]) if x is not None]:
                low_or |= v
            sem_bad = bool(msg["new_semantic_different"])
            a = {"attempt_id": attempt_id, "planned_cell_id": pid, "attempt": attempt, "order_index": exec_index,
                 "replication": c["replication"], "pair_id": c["pair_id"], "attack": c["attack"],
                 "setting": c["setting"], "start_rejections": rejections, "cooldown": cd,
                 "class_counts": msg["class_counts"], "new_semantic_different": msg["new_semantic_different"],
                 "warmup_s": msg["warmup_s"], "timed_s": msg["timed_s"],
                 "thermal": {"status": status, "reasons": treasons, "before": before, "after": after,
                             "temp_max_in_cell_c": max(temps) if temps else None, "n_in_cell_vc_samples": len(lows),
                             "throttled_low_or_in_cell": low_or,
                             "arm_clock_hz_min_in_cell": min(clocks) if clocks else None,
                             "arm_clock_hz_max_in_cell": max(clocks) if clocks else None},
                 "accepted": status == "THERMAL_CLEAN" and not sem_bad}
            attempts.append(a)
            exec_index += 1
            p1a.log(f"[parent] {attempt_id}: {status}{' ' + str(treasons) if treasons else ''}; temp "
                    f"{before['temp_c']} -> {after['temp_c']}; classes {msg['class_counts']}")
            save()
            if sem_bad:
                attack_failed[c["attack"]] = True
                c["final_status"] = "FAIL_SEMANTIC_DIFFERENT"
                p1a.log(f"[STOP] {c['attack']}: SEMANTIC_DIFFERENT output(s); remaining {c['attack']} cells skipped")
                break
            if a["accepted"]:
                c["final_status"] = "ACCEPTED"
                break
            c["final_status"] = "REVIEW_UNRESOLVED"
        if c["final_status"].startswith("SKIPPED_") and "ATTACK_FAILED" not in c["final_status"]:
            consecutive_skips += 1
            if consecutive_skips >= args.max_consecutive_skips:
                fail(f"{consecutive_skips} consecutive planned cells skipped; thermal threshold not practical")
        else:
            consecutive_skips = 0
        save()

    worker.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
    worker.stdin.flush()
    if recv().get("event") != "bye":
        fail("worker did not shut down cleanly")
    worker.wait(timeout=120)
    validation, cond = aggregate(out_dir, plan, attempts, ready, args)
    manifest.update(status="complete", validation_overall=validation["overall"],
                    finished_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(), environment_end=p1d0.snapshot())
    save()
    p1a.log(f"[done] {out_dir}  validation={validation['overall']}")
    for attack, d in validation["decisions"].items():
        p1a.log(f"  {attack}: {d['status']} ON med={d['on_median_ms']} OFF med={d['off_median_ms']} "
                f"pairs OFF-faster {d['n_pairs_off_faster']}/{d['n_valid_pairs']} median delta={d['paired_median_delta_ms']} "
                f"sign p={d['sign_test_two_sided_p']} OFF<=40ms={d['off_median_le_40ms']} -> {d['recommendation']}")
    if validation["overall"] == "FAIL":
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=p1a.DEFAULT_CONFIG)
    ap.add_argument("--formal-dir", default=p1a.DEFAULT_FORMAL_CPU_DIR)
    ap.add_argument("--out-dir", default=None, help="default results/accel_phase1d3_mkldnn_b1_<timestamp>")
    ap.add_argument("--replications", type=int, default=4, help="8 cells each; 4 -> 8 ON/OFF pairs per attack")
    ap.add_argument("--repeats", type=int, default=3, help="passes over the 88 samples per cell")
    ap.add_argument("--warmup-samples", type=int, default=2, help="untimed warmup calls per cell (>= 2)")
    ap.add_argument("--min-useful-reduction-ms", type=float, default=1.0)
    ap.add_argument("--max-start-temp-c", type=float, default=55.0)
    ap.add_argument("--release-consecutive-reads", type=int, default=2)
    ap.add_argument("--cooldown-poll-seconds", type=float, default=5.0)
    ap.add_argument("--max-cooldown-seconds", type=float, default=900.0)
    ap.add_argument("--max-start-rejections", type=int, default=5)
    ap.add_argument("--max-cell-attempts", type=int, default=3)
    ap.add_argument("--on-cooldown-timeout", choices=["skip", "abort"], default="skip")
    ap.add_argument("--max-consecutive-skips", type=int, default=2)
    ap.add_argument("--expected-cooldown-seconds", type=float, default=75.0, help="planning assumption only")
    ap.add_argument("--in-cell-vc-poll-seconds", type=float, default=5.0)
    ap.add_argument("--attack-temperature", type=float, default=None)
    ap.add_argument("--diagnostics", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-context", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.replications < 1 or args.repeats < 1 or args.warmup_samples < 2 or args.max_cell_attempts < 1:
        p1a.abort("--replications>=1, --repeats>=1, --warmup-samples>=2, --max-cell-attempts>=1 required")
    if args.cooldown_poll_seconds <= 0 or args.max_cooldown_seconds <= 0 or args.release_consecutive_reads < 1:
        p1a.abort("--cooldown-poll-seconds>0, --max-cooldown-seconds>0, --release-consecutive-reads>=1 required")
    if args.worker:
        worker_main(args)
    else:
        parent_main(args)


if __name__ == "__main__":
    main()
