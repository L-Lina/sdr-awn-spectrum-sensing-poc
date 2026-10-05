"""
B=1 attack-acceleration investigation for every formal attack profile (Raspberry Pi 5, CPU only).
EXPERIMENT ONLY -- src/adapters/attack_adapter.py and the formal pipeline are not modified.

Generalizes Phase 1D-3 (experiments/bench_attack_mkldnn_b1_pi5.py, PGD/BIM) to the remaining
formal attacks of configs/hailo_full_matrix.json. One script, one methodology:

Profiles
--------
Read directly from the formal config (attacks[].profiles[]); per profile the attack call uses the
same arguments as experiments/run_cpu_full_matrix.py: eps = profile["eps"], attack_params =
profile["params"], cw_c/cw_steps/cw_lr = profile values (defaults 1.0/20/0.01), temperature 1.0,
diagnostics off. The config must equal the formal CPU manifest's config block, and every subset
row of the formal attack_results.csv must carry the same eps / attack_params, or the profile is
not run. PGD and BIM (already decided in Phase 1D-3) are excluded unless --include-controls.

Measured path (per call, identical to Phase 1D-3's attack_block)
-----------------------------------------------------------------
untimed torch.manual_seed(formal per-row seed) ->
[timed] minmax mapping + set_minmax + attack construction + scoped atk(x, y) + output mapping
+ state restoration. Clean prediction is reused (validated equal to apply()'s internal y_pred).
For the mkldnn phase the setting is applied with torch._C._set_mkldnn_enabled immediately around
atk(x, y) and restored in finally; the baseline (ON) goes through the same statements. The
mkldnn state before / during / after every call is recorded.

Anchors (fail closed per profile)
---------------------------------
For every subset sample, under torch.manual_seed(seed), the production AttackAdapter.apply()
output (real backend, status ok) must be bit-identical to the benchmark ON path. The ON anchor is
compared with the formal adversarial_sha256 (required for p1a.DETERMINISTIC_ATTACKS, recorded for
the rest). A profile failing this is NEEDS_INVESTIGATION and gets no timing.

Semantic classification (each output vs the ON anchor of the same sample)
--------------------------------------------------------------------------
BIT_IDENTICAL | BITWISE_DIFFERENT_SEMANTIC_SAME (same attacked prediction AND same attack
success) | SEMANTIC_DIFFERENT | INVALID (non-finite, wrong shape/dtype, or -- for attacks with a
fixed Linf budget -- an eps-bound violation that the ON anchor of that sample does not have;
Phase 1C v2 tolerances). Attacks without a fixed eps budget (cw, deepfool, ead) are compared on
prediction / success / finiteness / shape only; their perturbation size is recorded, not gated.
Baseline outputs that are not bit-identical to their anchor are counted (baseline_nonrepeatable).

Phases per profile
------------------
1. mkldnn:  baseline ON vs candidate OFF, one persistent worker, intra 4 / interop 1 (Phase 1D-3).
2. threads: only for profiles whose mkldnn result is NO_MKLDNN_BENEFIT (always in --smoke):
            baseline intra 4 vs candidate intra 1 (interop 1, mkldnn default ON), one persistent
            worker per thread count -- the two settings Phase 1D-0 / 1D-0b already used.
Order: replication r is [base, cand, cand, base] for even r and [cand, base, base, cand] for odd
r; pairs are cells (1,2) and (3,4). Default 4 replications -> 8 pairs per phase, as Phase 1D-3.
Each cell: --warmup-samples untimed calls, then `repeats` passes over the subset. `repeats` is
--repeats (3, as Phase 1D-3), reduced per profile to keep one cell <= --max-cell-seconds
(estimated from the formal Pi latency table); the effective value is recorded. Attack parameters,
steps, budgets and seeds are never changed.

Thermal control (--full; Phase 1D-0b protocol, read-only)
--------------------------------------------------------
cool until --release-consecutive-reads polls <= --max-start-temp-c and get_throttled bits 0-3
clear; start gate on a fresh snapshot; in-cell sampling; Phase 1D-0 THERMAL_CLEAN classifier;
THERMAL_REVIEW cells are kept (accepted=False) and re-run up to --max-cell-attempts. Only
accepted cells enter statistics. --smoke does not gate (results are not formal).

Classification per profile and phase
------------------------------------
NEEDS_INVESTIGATION  fidelity failure, any INVALID / SEMANTIC_DIFFERENT / exception, baseline not
                     repeatable, state not restored (worker aborts), worker error, or a planned
                     cell without an accepted THERMAL_CLEAN attempt.
SPEEDUP_CANDIDATE    otherwise, candidate faster in every valid pair AND median paired reduction
                     >= max(--min-useful-reduction-ms, --min-useful-reduction-pct % of baseline).
NO_<PHASE>_BENEFIT   otherwise.

Resume
------
Every profile writes profiles/<attack>__<profile>/profile_result.json when complete.
--resume <run dir> skips completed profiles and re-runs the others (an incomplete profile
directory is renamed to *.incomplete_<timestamp>, never deleted). The resumed run must use the same
mode, subset, replications, repeats and config hash.

Run on the Pi from the repo root:
    .venv/bin/python experiments/benchmark_attack_b1_acceleration_full_pi5.py --smoke
    .venv/bin/python experiments/benchmark_attack_b1_acceleration_full_pi5.py --full
    .venv/bin/python experiments/benchmark_attack_b1_acceleration_full_pi5.py --full --resume results/<dir>
    (add --dry-run to print the plan and time estimate only)
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
from typing import Dict, List, Optional, Tuple

import numpy as np

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import bench_attack_accel_pi5 as p1a  # noqa: E402 -- formal context, subset, clean-input rebuild, utils
import bench_attack_batch_pi5_v2 as p1c  # noqa: E402 -- V2 tolerances, correctness classes
import bench_attack_overhead_pi5 as p1b  # noqa: E402 -- apply()-internal clean prediction check
import bench_attack_thermal_control_pi5 as p1d0  # noqa: E402 -- snapshot / sampler / THERMAL_CLEAN classifier
import bench_attack_thermal_pair_pi5 as p1d0b  # noqa: E402 -- cooldown release, start gate, sign test

REPO_ROOT = p1a.REPO_ROOT
CONTROL_ATTACKS = ("pgd", "bim")
INTEROP_THREADS = 1
MKLDNN_THREADS = 4
THREAD_BASE, THREAD_CAND = 4, 1
TEMPERATURE = 1.0
CLASSES = p1c.CLASSES
NORM_EPS_TOL, IQ_EPS_REL_TOL, IQ_EPS_ABS_TOL = p1c.NORM_EPS_TOL, p1c.IQ_EPS_REL_TOL, p1c.IQ_EPS_ABS_TOL

# Attacks whose output is projected onto a fixed Linf eps ball (bound gated relative to the anchor).
EPS_LINF_BOUNDED = {"fgsm", "bim", "pgd", "mifgsm", "difgsm", "vmifgsm", "vnifgsm", "rfgsm", "tpgd",
                    "fab", "square", "apgd", "apgdt", "autoattack"}
FAMILY = {
    "fgsm": "one_step_gradient",
    "bim": "iterative_gradient", "pgd": "iterative_gradient", "mifgsm": "iterative_gradient",
    "difgsm": "iterative_gradient", "vmifgsm": "iterative_gradient", "vnifgsm": "iterative_gradient",
    "rfgsm": "iterative_gradient", "tpgd": "iterative_gradient", "apgd": "iterative_gradient",
    "apgdt": "iterative_gradient",
    "cw": "optimization_boundary_search", "deepfool": "optimization_boundary_search",
    "fab": "optimization_boundary_search", "ead": "optimization_boundary_search",
    "square": "score_query_based", "autoattack": "composite",
}

SPEEDUP, NEEDS = "SPEEDUP_CANDIDATE", "NEEDS_INVESTIGATION"
NO_BENEFIT = {"mkldnn": "NO_MKLDNN_BENEFIT", "threads": "NO_THREAD_BENEFIT"}
PHASE_LABELS = {"mkldnn": ("mkldnn_ON", "mkldnn_OFF"), "threads": (f"intra{THREAD_BASE}", f"intra{THREAD_CAND}")}

SMOKE_SAMPLE_POSITIONS = (0, 43, 87)  # fixed positions in the 88-sample subset (spread over mods/SNRs)
DEFAULT_EST_CALL_MS = 500.0

RAW_FIELDS = ["attempt_id", "planned_cell_id", "order_index", "label", "rep", "sample_pos", "modulation",
              "snr_db", "sample_index", "total_ms", "prep_ms", "construct_ms", "call_ms", "post_ms",
              "mkldnn_before", "mkldnn_during", "mkldnn_after", "exception", "classification",
              "bit_identical_to_anchor", "cpu_temp_c"]
SAMPLE_FIELDS = ["phase", "attempt_id", "label", "rep", "sample_pos", "modulation", "snr_db", "sample_index",
                 "formal_seed", "true_label", "clean_pred", "attacked_pred", "anchor_attacked_pred",
                 "attacked_pred_equal", "attack_success", "anchor_attack_success", "attack_success_equal",
                 "iq_linf", "anchor_iq_linf", "iq_l2", "anchor_iq_l2", "iq_linf_normalized", "eps_bound_gated",
                 "norm_eps_bound_ok", "anchor_norm_eps_bound_ok", "iq_eps_bound_ok", "anchor_iq_eps_bound_ok",
                 "new_bound_violation", "finite", "shape_dtype_ok", "x_adv_sha256", "anchor_sha256",
                 "bit_identical_to_anchor", "max_abs_diff_vs_anchor", "classification", "exception",
                 "attacked_top2_margin", "anchor_top2_margin"]


# =========================================================================== shared (no torch)
def slug(attack: str, profile_id: str) -> str:
    return f"{attack}__{profile_id}".replace("/", "_")


def p05(vals: List[float]) -> float:
    return float(np.percentile(np.asarray(vals, dtype=np.float64), 5)) if vals else float("nan")


def lat_stats(vals: List[float]) -> dict:
    if not vals:
        return {"n": 0, **{k: float("nan") for k in ("median", "p05", "p95", "mean", "std", "min", "max")}}
    st = p1a.stats(vals)
    return {"n": st["n"], "median": st["median"], "p05": p05(vals), "p95": st["p95"], "mean": st["mean"],
            "std": st["std"], "min": st["min"], "max": st["max"]}


def write_csv(path: Path, rows: List[dict], fields: Optional[List[str]] = None) -> None:
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["empty"], restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def formal_profiles(cfg: dict, include_controls: bool) -> List[dict]:
    """Every formal profile, with the attack-call arguments run_cpu_full_matrix.py derives from it."""
    out = []
    for entry in cfg["attacks"]:
        attack = str(entry["name"]).lower()
        if attack in CONTROL_ATTACKS and not include_controls:
            continue
        for prof in entry["profiles"]:
            out.append({
                "attack": attack, "profile_id": prof["id"], "slug": slug(attack, prof["id"]),
                "eps": float(prof.get("eps", 0.03)), "attack_params": dict(prof.get("params", {})),
                "cw_c": float(prof.get("cw_c", 1.0)), "cw_steps": int(prof.get("cw_steps", 20)),
                "cw_lr": float(prof.get("cw_lr", 0.01)), "temperature": TEMPERATURE, "diagnostics": False,
                "eps_bound_gated": attack in EPS_LINF_BOUNDED, "family": FAMILY.get(attack, "unknown"),
                "formal_profile_raw": prof,
            })
    return out


def load_formal_rows(formal_dir: Path, subset_keys: set, wanted: set) -> Tuple[Dict[tuple, dict], Dict[tuple, dict]]:
    base, attack = {}, {}
    for r in read_csv(formal_dir / "base_results.csv"):
        k = (r["modulation"], int(float(r["snr_db"])), int(float(r["sample_index"])))
        if k in subset_keys:
            base[k] = r
    for r in read_csv(formal_dir / "attack_results.csv"):
        k = (r["modulation"], int(float(r["snr_db"])), int(float(r["sample_index"])))
        if k in subset_keys and (r["attack"], r["attack_profile"]) in wanted:
            attack[k + (r["attack"], r["attack_profile"])] = r
    if subset_keys - set(base):
        p1a.abort(f"formal base_results.csv lacks {len(subset_keys - set(base))} subset rows")
    return base, attack


def check_profile_against_formal(prof: dict, subset: List[tuple], attack_rows: Dict[tuple, dict]) -> List[str]:
    problems = []
    for key in subset:
        r = attack_rows.get(key + (prof["attack"], prof["profile_id"]))
        if r is None:
            problems.append(f"{key}: no formal attack row")
            continue
        if not math.isclose(float(r["eps_interface_value"]), prof["eps"], rel_tol=0, abs_tol=1e-12):
            problems.append(f"{key}: formal eps {r['eps_interface_value']} != config {prof['eps']}")
        if json.loads(r["attack_params_json"]) != prof["attack_params"]:
            problems.append(f"{key}: formal attack_params {r['attack_params_json']} != config {prof['attack_params']}")
        if r.get("attack_status", "ok") != "ok":
            problems.append(f"{key}: formal attack_status {r.get('attack_status')}")
    return problems


def est_call_ms(attack: str, profile_id: str) -> float:
    rows = read_csv(REPO_ROOT / p1a.FORMAL_LATENCY_TABLE)
    for r in rows:
        if r.get("backend") == "CPU" and r.get("profile") == f"{attack}/{profile_id}":
            return float(r["mean_ms"])
    return DEFAULT_EST_CALL_MS


def effective_repeats(args, n_samples: int, call_ms: float) -> int:
    if args.smoke:
        return 1
    cap = int(args.max_cell_seconds // (n_samples * call_ms / 1000.0)) if call_ms > 0 else args.repeats
    return max(1, min(args.repeats, cap))


def build_plan(replications: int, base: str, cand: str) -> List[dict]:
    plan = []
    for rep in range(replications):
        layout = [base, cand, cand, base] if rep % 2 == 0 else [cand, base, base, cand]
        for j, label in enumerate(layout):
            plan.append({"replication": rep, "label": label, "pair_id": f"r{rep}_p{j // 2}",
                         "pair_role": "first" if j % 2 == 0 else "second"})
    for i, c in enumerate(plan):
        c["planned_index"] = i
        c["planned_cell_id"] = f"{i:02d}_{c['label']}"
    return plan


# =========================================================================== worker (torch)
def worker_main(args) -> None:
    proto = os.fdopen(os.dup(1), "w", buffering=1)  # JSON-line protocol; all prints go to stderr
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(msg: dict) -> None:
        proto.write(json.dumps(msg, default=str) + "\n")
        proto.flush()

    import torch
    torch.set_num_interop_threads(INTEROP_THREADS)
    torch.set_num_threads(args.worker_threads)
    intra, interop = torch.get_num_threads(), torch.get_num_interop_threads()
    if (intra, interop) != (args.worker_threads, INTEROP_THREADS):
        p1a.abort(f"thread settings not applied: intra={intra} interop={interop}")
    import src.adapters.attack_adapter as am
    from src.adapters.attack_adapter import AttackAdapter, _REAL_ATTACK_SOURCE
    from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE

    get_mk = getattr(torch._C, "_get_mkldnn_enabled", None)
    set_mk = getattr(torch._C, "_set_mkldnn_enabled", None)
    if get_mk is None or set_mk is None:
        p1a.abort("torch._C._get/_set_mkldnn_enabled unavailable; cannot scope the setting explicitly")
    default_mk = bool(get_mk())
    if default_mk is not True:
        p1a.abort(f"mkldnn is not enabled by default in this process ({default_mk})")

    ctx = json.loads(Path(args.worker_context).read_text())
    out_dir = Path(args.out_dir)
    prof = ctx["profile"]
    attack, eps = prof["attack"], float(prof["eps"])
    subset = [tuple(k) for k in ctx["subset"]]
    tag = args.tag

    awn = AWNModelAdapter(checkpoint_path=str(REPO_ROOT / ctx["checkpoint_path"]), device="cpu")
    if awn.model is None or awn.backend_name != _REAL_MODEL_SOURCE or awn.status != "ok":
        p1a.abort(f"AWN backend is not real: {awn.backend_name} {awn.status}")
    adapter = AttackAdapter(awn_model=awn.model, device="cpu")
    if adapter.wrapped_model is None or adapter.backend_name != _REAL_ATTACK_SOURCE or adapter.status != "ok":
        p1a.abort("attack backend is not real")
    if am._torchattacks is None or awn.model.training:
        p1a.abort("torchattacks unavailable or model in train mode")
    wm = adapter.wrapped_model

    # clean inputs rebuilt once by the parent; re-verified here against the formal hashes
    npz = np.load(REPO_ROOT / ctx["clean_inputs_npz"])
    inputs: Dict[tuple, dict] = {}
    for i, key in enumerate(subset):
        meta = ctx["clean_inputs_meta"][i]
        x = np.ascontiguousarray(npz[f"x{i}"])
        if p1a.sha256_array(x) != meta["formal_clean_input_sha256"]:
            p1a.abort(f"{key}: clean input sha256 != formal clean_input_sha256")
        inputs[key] = {"x": x, "seed": int(meta["seed"]), "label": int(meta["label"]),
                       "formal_clean_pred": int(meta["formal_clean_pred"])}

    def infer_pred_margin(x) -> tuple:  # always with the default (mkldnn ON) setting
        """(argmax class, top-1 minus top-2 logit) -- the margin shows how close an output sits to a
        decision boundary (diagnostic only; never part of the classification rule)."""
        if bool(get_mk()) is not default_mk:
            p1a.abort("mkldnn setting leaked outside the attack call")
        logits, meta = awn.infer(x)
        if meta["awn_backend"] != _REAL_MODEL_SOURCE or meta["awn_status"] != "ok" or not np.isfinite(logits).all():
            p1a.abort(f"AWN inference invalid: {meta}")
        top2 = np.sort(logits[0])[-2:]
        return int(np.argmax(logits[0])), float(top2[1] - top2[0])

    def infer_pred(x) -> int:
        return infer_pred_margin(x)[0]

    # AutoAttack fixed-clock control (opt-in, --autoattack-fixed-clock): torchattacks.AutoAttack without an
    # explicit seed derives its sub-attack seeds from time.time(), so two constructions of the same formal
    # profile are not reproducible. The control replaces that clock, inside the autoattack module only, by
    # the formal per-row seed of the sample being attacked -- for the production apply() anchor and for the
    # benchmark path alike. Attack parameters are unchanged; only the otherwise wall-clock seed is fixed.
    clock = {"t": None}
    if ctx.get("autoattack_fixed_clock") and attack == "autoattack":
        import inspect
        aa_mod = sys.modules[am._torchattacks.AutoAttack.__module__]
        aa_src = inspect.getsource(am._torchattacks.AutoAttack)
        if "time.time()" not in aa_src or getattr(aa_mod, "time", None) is None or not hasattr(aa_mod.time, "time"):
            p1a.abort("torchattacks.AutoAttack does not seed from the module-level time.time(); "
                      "--autoattack-fixed-clock is not applicable to this torchattacks version")

        class _FixedClock:
            @staticmethod
            def time() -> float:
                if clock["t"] is None:
                    raise RuntimeError("fixed AutoAttack clock used before a sample seed was set")
                return float(clock["t"])
        aa_mod.time = _FixedClock
        print("[bench] AutoAttack fixed-clock control active (seed = formal per-row seed)")

    for key, inp in inputs.items():
        inp["clean_pred"] = infer_pred(inp["x"])
        if inp["clean_pred"] != inp["formal_clean_pred"]:
            p1a.abort(f"{key}: clean_pred != formal")
        if p1b.in_path_clean_pred(am, adapter, torch, inp["x"]) != inp["clean_pred"]:
            p1a.abort(f"{key}: apply()-internal y_pred != pipeline clean_pred")

    req_grad_ref = [p.requires_grad for p in wm.parameters()]

    def check_state(where: str) -> None:  # fail closed: any leak aborts the worker
        if awn.model.training or wm.training:
            p1a.abort(f"model entered train mode ({where})")
        if (torch.get_num_threads(), torch.get_num_interop_threads()) != (intra, interop):
            p1a.abort(f"torch thread settings drifted ({where})")
        if [p.requires_grad for p in wm.parameters()] != req_grad_ref:
            p1a.abort(f"requires_grad flags not restored ({where})")
        if bool(get_mk()) is not default_mk:
            p1a.abort(f"mkldnn global setting not restored ({where})")

    def attack_block(key, mkldnn_on: bool):
        """Phase 1D-3 attack_block, generalized to any formal profile."""
        pc = time.perf_counter
        t0 = pc()
        x = inputs[key]["x"]
        training_before = wm.training
        orig_requires_grad = [p.requires_grad for p in wm.parameters()]
        orig_param_devices = [p.device for p in wm.parameters()]
        mk = {"mkldnn_before": None, "mkldnn_during": None, "mkldnn_after": None}
        try:
            x_ta, a, b = am._iq_to_ta_input_minmax(torch.from_numpy(x).to(adapter.device))
            wm.set_minmax(a, b)
            y = torch.tensor([inputs[key]["clean_pred"]], dtype=torch.long)
            t1 = pc()
            atk = am._build_torchattacks(attack, am.TemperatureLogitsWrapper(wm, prof["temperature"]), eps,
                                         cw_c=prof["cw_c"], cw_steps=prof["cw_steps"], cw_lr=prof["cw_lr"],
                                         attack_params=dict(prof["attack_params"]))
            t2 = pc()
            prev = bool(get_mk())
            mk["mkldnn_before"] = prev
            set_mk(mkldnn_on)
            try:
                mk["mkldnn_during"] = bool(get_mk())
                if mk["mkldnn_during"] is not mkldnn_on:
                    raise RuntimeError("mkldnn setting did not take effect")
                x_ta_adv = atk(x_ta, y)
            finally:
                set_mk(prev)
                mk["mkldnn_after"] = bool(get_mk())
            t3 = pc()
            lin = (x_ta_adv - x_ta).abs().amax(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.float32)
            x_adv = am._ta_output_to_iq_minmax(x_ta_adv, a, b).detach().cpu().numpy().astype(np.float32)
            b_np = b.detach().cpu().numpy().reshape(-1)
        finally:
            wm.eval()
            wm.clear_minmax()
            for p, req_grad, dev in zip(wm.parameters(), orig_requires_grad, orig_param_devices):
                p.requires_grad_(req_grad)
                if p.device != dev:
                    p.data = p.data.to(dev)
            if training_before:
                print("[bench] warning: wrapped model was already in train mode before this call")
        t4 = pc()
        return x_adv, lin, b_np, {"total_ms": (t4 - t0) * 1e3, "prep_ms": (t1 - t0) * 1e3,
                                  "construct_ms": (t2 - t1) * 1e3, "call_ms": (t3 - t2) * 1e3,
                                  "post_ms": (t4 - t3) * 1e3}, mk

    def seeded_block(key, mkldnn_on: bool):
        torch.manual_seed(inputs[key]["seed"])  # untimed RNG reset before every call
        clock["t"] = inputs[key]["seed"]
        return attack_block(key, mkldnn_on)

    def bound_flags(x_adv, lin, b_np, key) -> tuple:
        linf = float(np.max(np.abs(x_adv - inputs[key]["x"])))
        norm_ok = bool(lin[0] <= eps + NORM_EPS_TOL)
        iq_ok = bool(linf <= eps * float(b_np[0]) * (1 + IQ_EPS_REL_TOL) + IQ_EPS_ABS_TOL)
        return norm_ok, iq_ok

    # ---------------------------------------------------------------- anchors
    anchors_path = REPO_ROOT / ctx["anchors_npz"]
    anchors_meta_path = REPO_ROOT / ctx["anchors_meta_json"]
    fidelity = {"n": len(subset), "n_apply_bit_identical": 0, "n_formal_sha_match": 0, "failures": [],
                "formal_sha_required": attack in p1a.DETERMINISTIC_ATTACKS}
    anchors: Dict[tuple, dict] = {}
    if args.phase == "mkldnn":
        arrays = {}
        for i, key in enumerate(subset):
            inp = inputs[key]
            try:
                torch.manual_seed(inp["seed"])
                clock["t"] = inp["seed"]
                ref, meta = adapter.apply(inp["x"], attack=attack, eps=eps, temperature=prof["temperature"],
                                          seed=inp["seed"], diagnostics=False, cw_c=prof["cw_c"],
                                          cw_steps=prof["cw_steps"], cw_lr=prof["cw_lr"],
                                          attack_params=dict(prof["attack_params"]))
                check_state(f"anchor-apply {key}")
                if meta["attack_status"] != "ok" or meta["attack_backend"] != _REAL_ATTACK_SOURCE:
                    raise RuntimeError(f"production apply() did not run the real attack: {meta['attack_notes']}")
                on, lin, b_np, _, _ = seeded_block(key, True)
                check_state(f"anchor-on {key}")
            except SystemExit:
                raise
            except Exception as exc:  # noqa: BLE001 - recorded, profile becomes NEEDS_INVESTIGATION
                fidelity["failures"].append({"sample": list(key), "reason": f"{type(exc).__name__}: {exc}"})
                continue
            if not np.array_equal(ref, on):
                fidelity["failures"].append({"sample": list(key), "reason": "benchmark ON path != production apply()"})
                continue
            fidelity["n_apply_bit_identical"] += 1
            sha = p1a.sha256_array(on)
            formal_sha = ctx["formal_sha"].get("|".join(map(str, key)))
            if sha == formal_sha:
                fidelity["n_formal_sha_match"] += 1
            elif fidelity["formal_sha_required"]:
                fidelity["failures"].append({"sample": list(key), "reason": "ON anchor sha256 != formal adversarial_sha256"})
            pred, margin = infer_pred_margin(on)
            d = on - inp["x"]
            norm_ok, iq_ok = bound_flags(on, lin, b_np, key)
            anchors[key] = {"sha": sha, "pred": pred, "margin": margin,
                            "success": bool(inp["clean_pred"] == inp["label"] and pred != inp["label"]),
                            "linf": float(np.max(np.abs(d))), "l2": float(np.linalg.norm(d.reshape(-1))),
                            "norm_ok": norm_ok, "iq_ok": iq_ok}
            arrays[f"a{i}"] = on
            anchors[key]["x_adv"] = on
        np.savez(anchors_path, **arrays)
        anchors_meta_path.write_text(json.dumps(
            {"|".join(map(str, k)): {kk: vv for kk, vv in v.items() if kk != "x_adv"} for k, v in anchors.items()},
            indent=2))
    else:
        loaded = np.load(anchors_path)
        meta_all = json.loads(anchors_meta_path.read_text())
        for i, key in enumerate(subset):
            k = "|".join(map(str, key))
            if k in meta_all and f"a{i}" in loaded:
                anchors[key] = {**meta_all[k], "x_adv": np.ascontiguousarray(loaded[f"a{i}"])}
            else:
                fidelity["failures"].append({"sample": list(key), "reason": "anchor missing from mkldnn phase"})
        fidelity["n_apply_bit_identical"] = None
        fidelity["n_formal_sha_match"] = None
    fidelity_ok = not fidelity["failures"] and len(anchors) == len(subset)

    smp_f = (out_dir / f"per_sample_{tag}.csv").open("w", newline="")
    raw_f = (out_dir / f"raw_{tag}.csv").open("w", newline="")
    smp_w = csv.DictWriter(smp_f, fieldnames=SAMPLE_FIELDS, restval="")
    raw_w = csv.DictWriter(raw_f, fieldnames=RAW_FIELDS, restval="")
    smp_w.writeheader()
    raw_w.writeheader()

    def classify(phase, attempt_id, label, rep, pos, key, x_adv, lin, b_np, exc_text) -> str:
        inp, anc = inputs[key], anchors[key]
        row = {"phase": phase, "attempt_id": attempt_id, "label": label, "rep": rep, "sample_pos": pos,
               "modulation": key[0], "snr_db": key[1], "sample_index": key[2], "formal_seed": inp["seed"],
               "true_label": inp["label"], "clean_pred": inp["clean_pred"], "anchor_attacked_pred": anc["pred"],
               "anchor_attack_success": anc["success"], "anchor_iq_linf": anc["linf"], "anchor_iq_l2": anc["l2"],
               "eps_bound_gated": prof["eps_bound_gated"], "anchor_norm_eps_bound_ok": anc["norm_ok"],
               "anchor_iq_eps_bound_ok": anc["iq_ok"], "anchor_sha256": anc["sha"], "exception": exc_text}
        if exc_text:
            row["classification"] = "EXCEPTION"
            smp_w.writerow(row)
            return "EXCEPTION"
        xi = np.ascontiguousarray(x_adv[0:1])
        shape_ok = xi.shape == anc["x_adv"].shape and xi.dtype == np.float32
        fin = bool(np.isfinite(xi).all())
        bit = bool(shape_ok and np.array_equal(xi, anc["x_adv"]))
        md = float(np.max(np.abs(xi.astype(np.float64) - anc["x_adv"].astype(np.float64)))) if (fin and shape_ok) \
            else float("inf")
        if bit:
            pred, margin = anc["pred"], anc.get("margin", "")
        else:
            pred, margin = infer_pred_margin(xi) if (fin and shape_ok) else (-1, "")
        succ = bool(inp["clean_pred"] == inp["label"] and pred != inp["label"])
        dlt = xi - inp["x"] if shape_ok else np.full_like(inp["x"], np.inf)
        norm_ok, iq_ok = bound_flags(xi, lin, b_np, key) if (fin and shape_ok) else (False, False)
        new_violation = bool(prof["eps_bound_gated"] and ((anc["norm_ok"] and not norm_ok) or (anc["iq_ok"] and not iq_ok)))
        if not (fin and shape_ok) or new_violation:
            cls = "INVALID"
        elif bit:
            cls = "BIT_IDENTICAL"
        elif pred == anc["pred"] and succ == anc["success"]:
            cls = "BITWISE_DIFFERENT_SEMANTIC_SAME"
        else:
            cls = "SEMANTIC_DIFFERENT"
        row.update({"attacked_pred": pred, "attacked_pred_equal": pred == anc["pred"], "attack_success": succ,
                    "attack_success_equal": succ == anc["success"],
                    "iq_linf": float(np.max(np.abs(dlt))), "iq_l2": float(np.linalg.norm(dlt.reshape(-1))),
                    "iq_linf_normalized": float(lin[0]) if lin is not None else "", "norm_eps_bound_ok": norm_ok,
                    "iq_eps_bound_ok": iq_ok, "new_bound_violation": new_violation, "finite": fin,
                    "shape_dtype_ok": shape_ok, "x_adv_sha256": p1a.sha256_array(xi),
                    "bit_identical_to_anchor": bit, "max_abs_diff_vs_anchor": md, "classification": cls,
                    "attacked_top2_margin": margin, "anchor_top2_margin": anc.get("margin", "")})
        smp_w.writerow(row)
        return cls

    def run_call(phase, attempt_id, label, rep, pos, key, mkldnn_on):
        """One seeded B=1 call + state check + classification; exceptions are recorded, not raised."""
        exc_text, x_adv, lin, b_np = "", None, None, None
        tm = {"total_ms": "", "prep_ms": "", "construct_ms": "", "call_ms": "", "post_ms": ""}
        mk = {"mkldnn_before": "", "mkldnn_during": "", "mkldnn_after": ""}
        try:
            x_adv, lin, b_np, tm, mk = seeded_block(key, mkldnn_on)
        except Exception as exc:  # noqa: BLE001 - counted; profile becomes NEEDS_INVESTIGATION
            exc_text = f"{type(exc).__name__}: {exc}"
        check_state(f"{phase} {attempt_id} {key}")
        cls = classify(phase, attempt_id, label, rep, pos, key, x_adv, lin, b_np, exc_text)
        return cls, tm, mk, exc_text

    # ---------------------------------------------------------------- pre-validation (untimed)
    preval_counts = {c: 0 for c in CLASSES + ("EXCEPTION",)}
    if fidelity_ok:
        pre_on = args.phase == "threads"  # mkldnn phase pre-validates the candidate (OFF)
        for pos, key in enumerate(subset):
            cls, _, _, _ = run_call("prevalidation", "", tag if args.phase == "threads" else PHASE_LABELS["mkldnn"][1],
                                    "", pos, key, pre_on)
            preval_counts[cls] += 1
    smp_f.flush()
    send({"event": "ready", "pid": os.getpid(), "tag": tag, "intra_threads": intra, "interop_threads": interop,
          "fidelity": fidelity, "fidelity_ok": fidelity_ok, "prevalidation": preval_counts,
          "mkldnn_default": default_mk, "torch_version": torch.__version__})

    # ---------------------------------------------------------------- cells
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        msg = json.loads(line)
        if msg.get("cmd") == "quit":
            break
        cell = msg["cell"]
        on = bool(cell["mkldnn_on"])
        counts = {c: 0 for c in CLASSES + ("EXCEPTION",)}
        n_mk_bad = 0
        t_w0 = time.perf_counter()
        for key in subset[: args.warmup_samples]:
            try:
                seeded_block(key, on)
            except Exception:  # noqa: BLE001 - warmup only; the timed call records exceptions
                pass
            check_state(f"warmup {cell['attempt_id']}")
        t_w1 = time.perf_counter()
        for rep in range(cell["repeats"]):
            for pos, key in enumerate(subset):
                temp = p1a.cpu_temp_c()
                cls, tm, mk, exc_text = run_call("timed", cell["attempt_id"], cell["label"], rep, pos, key, on)
                counts[cls] += 1
                if not exc_text and (mk["mkldnn_during"] is not on or mk["mkldnn_after"] is not default_mk):
                    n_mk_bad += 1
                raw_w.writerow({"attempt_id": cell["attempt_id"], "planned_cell_id": cell["planned_cell_id"],
                                "order_index": cell["order_index"], "label": cell["label"], "rep": rep,
                                "sample_pos": pos, "modulation": key[0], "snr_db": key[1], "sample_index": key[2],
                                **tm, **mk, "exception": exc_text, "classification": cls,
                                "bit_identical_to_anchor": cls == "BIT_IDENTICAL", "cpu_temp_c": temp})
            raw_f.flush()
            smp_f.flush()
        send({"event": "cell_done", "attempt_id": cell["attempt_id"], "class_counts": counts,
              "n_mkldnn_state_mismatch": n_mk_bad, "warmup_s": t_w1 - t_w0, "timed_s": time.perf_counter() - t_w1})
    raw_f.close()
    smp_f.close()
    send({"event": "bye"})


# =========================================================================== parent: one phase
class WorkerError(RuntimeError):
    pass


def start_worker(args, ctx_file: Path, phase_dir: Path, phase: str, tag: str, threads: int) -> subprocess.Popen:
    errf = (phase_dir / f"worker_{tag}.stderr.log").open("w")
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--worker-context", str(ctx_file),
           "--out-dir", str(phase_dir), "--worker-threads", str(threads), "--phase", phase, "--tag", tag,
           "--warmup-samples", str(args.warmup_samples)]
    return subprocess.Popen(cmd, cwd=REPO_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf,
                            text=True, bufsize=1, env=os.environ.copy())


def recv(worker: subprocess.Popen, tag: str) -> dict:
    line = worker.stdout.readline()
    if not line:
        raise WorkerError(f"worker {tag} exited (rc={worker.wait()}) -- failed fail-closed check (state not "
                          f"restored / backend not real / input mismatch) or crash; see worker_{tag}.stderr.log")
    return json.loads(line)


def stop_workers(workers: Dict[str, subprocess.Popen]) -> None:
    for tag, w in workers.items():
        if w.poll() is None:
            try:
                w.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                w.stdin.flush()
                recv(w, tag)
                w.wait(timeout=120)
            except Exception:  # noqa: BLE001
                w.kill()


def run_phase(args, prof: dict, phase: str, prof_dir: Path, ctx_file: Path, repeats: int) -> dict:
    base, cand = PHASE_LABELS[phase]
    phase_dir = prof_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args.replications, base, cand)
    if phase == "mkldnn":
        route = {base: ("w4", True), cand: ("w4", False)}
        worker_threads = {"w4": MKLDNN_THREADS}
    else:
        route = {base: (base, True), cand: (cand, True)}
        worker_threads = {base: THREAD_BASE, cand: THREAD_CAND}
    result = {"phase": phase, "baseline": base, "candidate": cand, "repeats": repeats, "plan": plan, "attempts": [],
              "ready": {}, "error": None}
    trace: List[dict] = []
    workers: Dict[str, subprocess.Popen] = {}
    try:
        for tag, t in worker_threads.items():  # mkldnn phase: anchors are produced by its single worker
            workers[tag] = start_worker(args, ctx_file, phase_dir, phase, tag, t)
            ready = recv(workers[tag], tag)
            if ready.get("event") != "ready":
                raise WorkerError(f"unexpected worker message {ready}")
            result["ready"][tag] = ready
            p1a.log(f"[{prof['slug']}/{phase}] worker {tag} ready: fidelity_ok={ready['fidelity_ok']} "
                    f"preval={ready['prevalidation']}")
        blocked = [f"{tag}: fidelity failed" for tag, r in result["ready"].items() if not r["fidelity_ok"]] + \
                  [f"{tag}: pre-validation {r['prevalidation']}" for tag, r in result["ready"].items()
                   if r["prevalidation"]["SEMANTIC_DIFFERENT"] or r["prevalidation"]["INVALID"]
                   or r["prevalidation"]["EXCEPTION"]]
        if blocked:
            result["blocked"] = blocked
            for c in plan:
                c["final_status"] = "SKIPPED_PREVALIDATION_FAILED"
        else:
            run_cells(args, prof, phase, plan, route, workers, result, trace, repeats)
    except WorkerError as exc:
        result["error"] = str(exc)
        p1a.log(f"[{prof['slug']}/{phase}] ERROR {exc}")
    finally:
        stop_workers(workers)
        write_csv(phase_dir / "thermal_trace.csv", trace, p1d0b.TRACE_FIELDS)
    return aggregate_phase(args, prof, phase, phase_dir, result)


def run_cells(args, prof, phase, plan, route, workers, result, trace, repeats) -> None:
    exec_index, consecutive_skips = 0, 0
    for c in plan:
        c["final_status"] = "NOT_RUN"
        c["start_rejections_total"] = 0
    for c in plan:
        pid = c["planned_cell_id"]
        tag, mkldnn_on = route[c["label"]]
        for attempt in range(1, args.max_cell_attempts + 1):
            if args.smoke:
                cd = {"outcome": "REACHED", "waited_s": 0.0, "note": "smoke: no thermal gate"}
                before = p1d0.snapshot()
            else:
                rejections = 0
                while True:
                    cd = p1d0b.wait_until_cool(args, trace, pid)
                    if cd["outcome"] != "REACHED":
                        break
                    before = p1d0.snapshot()
                    reasons = p1d0b.start_gate(before, args.max_start_temp_c)
                    if not reasons:
                        break
                    rejections += 1
                    c["start_rejections_total"] += 1
                    if rejections > args.max_start_rejections:
                        cd = {**cd, "outcome": "START_GATE_EXHAUSTED"}
                        break
                if cd["outcome"] != "REACHED":
                    c["final_status"] = f"SKIPPED_{cd['outcome']}"
                    break
            attempt_id = f"{pid}_a{attempt}"
            sampler = p1d0.CellSampler(attempt_id, exec_index, 1.0, args.in_cell_vc_poll_seconds).start()
            w = workers[tag]
            w.stdin.write(json.dumps({"cmd": "run", "cell": {
                "attempt_id": attempt_id, "planned_cell_id": pid, "order_index": exec_index, "label": c["label"],
                "mkldnn_on": mkldnn_on, "repeats": repeats}}) + "\n")
            w.stdin.flush()
            msg = recv(w, tag)
            samples = sampler.stop()
            after = p1d0.snapshot()
            if msg.get("event") != "cell_done" or msg.get("attempt_id") != attempt_id:
                raise WorkerError(f"unexpected reply for {attempt_id}: {msg}")
            for s_ in samples:
                trace.append({"phase": "cell", "planned_cell_id": pid, "attempt_id": attempt_id,
                              "order_index": exec_index, **{k: s_[k] for k in p1d0b.TRACE_FIELDS[4:]}})
            if args.smoke:
                status, treasons = "SMOKE_NOT_GATED", []
            else:
                status, treasons = p1d0.classify_thermal(before, after, samples, cd, args.max_start_temp_c)
            temps = [s_["temp_c"] for s_ in samples if s_["temp_c"] is not None]
            cc = msg["class_counts"]
            bad = cc["SEMANTIC_DIFFERENT"] + cc["INVALID"] + cc["EXCEPTION"] + msg["n_mkldnn_state_mismatch"]
            a = {"attempt_id": attempt_id, "planned_cell_id": pid, "attempt": attempt, "order_index": exec_index,
                 "replication": c["replication"], "pair_id": c["pair_id"], "label": c["label"],
                 "cooldown": cd, "class_counts": cc, "n_mkldnn_state_mismatch": msg["n_mkldnn_state_mismatch"],
                 "warmup_s": msg["warmup_s"], "timed_s": msg["timed_s"],
                 "thermal": {"status": status, "reasons": treasons, "temp_before_c": before["temp_c"],
                             "temp_after_c": after["temp_c"], "temp_max_in_cell_c": max(temps) if temps else None,
                             "throttled_before": before["throttled"]["value_hex"],
                             "throttled_after": after["throttled"]["value_hex"]},
                 "accepted": status in ("THERMAL_CLEAN", "SMOKE_NOT_GATED") and not bad}
            result["attempts"].append(a)
            exec_index += 1
            p1a.log(f"[{prof['slug']}/{phase}] {attempt_id}: {status}; classes {cc}")
            if bad:
                c["final_status"] = "FAIL_SEMANTIC_OR_RUNTIME"
                result["stopped"] = f"{attempt_id}: {cc}, mkldnn state mismatches {msg['n_mkldnn_state_mismatch']}"
                for rest in plan:
                    if rest["final_status"] == "NOT_RUN" and rest is not c:
                        rest["final_status"] = "SKIPPED_AFTER_FAILURE"
                return
            if a["accepted"]:
                c["final_status"] = "ACCEPTED"
                break
            c["final_status"] = "REVIEW_UNRESOLVED"
        if c["final_status"].startswith("SKIPPED_"):
            consecutive_skips += 1
            if consecutive_skips >= args.max_consecutive_skips:
                result["stopped"] = f"{consecutive_skips} consecutive planned cells skipped (thermal)"
                for rest in plan:
                    if rest["final_status"] == "NOT_RUN":
                        rest["final_status"] = "SKIPPED_THERMAL_NOT_PRACTICAL"
                return
        else:
            consecutive_skips = 0


def aggregate_phase(args, prof: dict, phase: str, phase_dir: Path, result: dict) -> dict:
    base, cand = result["baseline"], result["candidate"]
    plan, attempts = result["plan"], result["attempts"]
    acc = {a["attempt_id"]: a for a in attempts}
    rows = []
    for f in sorted(phase_dir.glob("raw_*.csv")):
        rows += read_csv(f)
    for r in rows:
        a = acc.get(r["attempt_id"], {})
        r["accepted"] = a.get("accepted", False)
        r["pair_id"] = a.get("pair_id", "")
    rows.sort(key=lambda r: (int(r["order_index"]), int(r["rep"]), int(r["sample_pos"])))
    write_csv(phase_dir / "raw_measurements.csv", rows, RAW_FIELDS + ["accepted", "pair_id"])
    samples = []
    for f in sorted(phase_dir.glob("per_sample_*.csv")):
        samples += read_csv(f)
    write_csv(phase_dir / "per_sample_validation.csv", samples, SAMPLE_FIELDS)

    tot = {a["attempt_id"]: [float(r["total_ms"]) for r in rows if r["attempt_id"] == a["attempt_id"]
                             and r["total_ms"] not in ("", None)] for a in attempts}
    write_csv(phase_dir / "summary.csv", [{
        "attempt_id": a["attempt_id"], "planned_cell_id": a["planned_cell_id"], "pair_id": a["pair_id"],
        "label": a["label"], "accepted": a["accepted"], "thermal_status": a["thermal"]["status"],
        "thermal_reasons": " | ".join(a["thermal"]["reasons"]), "temp_before_c": a["thermal"]["temp_before_c"],
        "temp_after_c": a["thermal"]["temp_after_c"], "temp_max_in_cell_c": a["thermal"]["temp_max_in_cell_c"],
        "throttled_before": a["thermal"]["throttled_before"], "throttled_after": a["thermal"]["throttled_after"],
        "cooldown_waited_s": a["cooldown"].get("waited_s"),
        **{f"n_{k.lower()}": v for k, v in a["class_counts"].items()},
        "n_mkldnn_state_mismatch": a["n_mkldnn_state_mismatch"],
        **{f"b1_latency_ms_{k}": v for k, v in lat_stats(tot[a["attempt_id"]]).items()},
        "warmup_s": a["warmup_s"], "timed_s": a["timed_s"]} for a in attempts])

    accepted = {a["planned_cell_id"]: a for a in attempts if a["accepted"]}
    pooled = {lab: lat_stats([v for c in plan if c["label"] == lab and c["planned_cell_id"] in accepted
                              for v in tot[accepted[c["planned_cell_id"]]["attempt_id"]]]) for lab in (base, cand)}
    pairs = []
    for pid in dict.fromkeys(c["pair_id"] for c in plan):
        cs = [c for c in plan if c["pair_id"] == pid]
        cb = next(c for c in cs if c["label"] == base)
        cc_ = next(c for c in cs if c["label"] == cand)
        ab, ac = accepted.get(cb["planned_cell_id"]), accepted.get(cc_["planned_cell_id"])
        row = {"pair_id": pid, "first_in_pair": base if cb["pair_role"] == "first" else cand,
               "pair_valid": bool(ab and ac)}
        if ab and ac:
            mb, mc = float(np.median(tot[ab["attempt_id"]])), float(np.median(tot[ac["attempt_id"]]))
            row.update(baseline_median_ms=mb, candidate_median_ms=mc, delta_ms_candidate_minus_baseline=mc - mb,
                       delta_pct=100.0 * (mc - mb) / mb, speedup_baseline_over_candidate=mb / mc)
        else:
            row["not_valid_reason"] = f"{base} {cb.get('final_status')}, {cand} {cc_.get('final_status')}"
        pairs.append(row)
    write_csv(phase_dir / "paired_comparisons.csv", pairs,
              ["pair_id", "first_in_pair", "pair_valid", "baseline_median_ms", "candidate_median_ms",
               "delta_ms_candidate_minus_baseline", "delta_pct", "speedup_baseline_over_candidate",
               "not_valid_reason"])

    # ---- decision
    counts = {k: 0 for k in CLASSES + ("EXCEPTION",)}
    base_nonrep = 0
    for a in attempts:
        for k, v in a["class_counts"].items():
            counts[k] += v
    for r in rows:
        if r["label"] == base and r["classification"] != "BIT_IDENTICAL":
            base_nonrep += 1
    pre = {tag: r["prevalidation"] for tag, r in result["ready"].items()}
    pre_bad = sum(p["SEMANTIC_DIFFERENT"] + p["INVALID"] + p["EXCEPTION"] for p in pre.values())
    fid_ok = bool(result["ready"]) and all(r["fidelity_ok"] for r in result["ready"].values())
    mk_bad = sum(a["n_mkldnn_state_mismatch"] for a in attempts)
    not_acc = [f"{c['planned_cell_id']}: {c.get('final_status')}" for c in plan if c["planned_cell_id"] not in accepted]
    valid = [p for p in pairs if p["pair_valid"]]
    deltas = [p["delta_ms_candidate_minus_baseline"] for p in valid]
    nz = [d for d in deltas if d != 0]
    k_faster = sum(1 for d in nz if d < 0)
    med_delta = float(np.median(deltas)) if deltas else None
    threshold = max(args.min_useful_reduction_ms, args.min_useful_reduction_pct / 100.0 *
                    (pooled[base]["median"] if pooled[base]["n"] else 0.0))
    reasons = []
    if result["error"]:
        reasons.append(f"worker error: {result['error']}")
    if not fid_ok:
        reasons.append("anchor fidelity failed")
    if pre_bad:
        reasons.append(f"pre-validation failures {pre}")
    if counts["SEMANTIC_DIFFERENT"] or counts["INVALID"] or counts["EXCEPTION"]:
        reasons.append(f"timed failures SD={counts['SEMANTIC_DIFFERENT']} INVALID={counts['INVALID']} "
                       f"EXC={counts['EXCEPTION']}")
    if mk_bad:
        reasons.append(f"mkldnn state mismatches {mk_bad}")
    if base_nonrep:
        reasons.append(f"baseline not repeatable in {base_nonrep} timed calls")
    if not_acc:
        reasons.append(f"{len(not_acc)} planned cells without an accepted attempt")
    if not valid:
        reasons.append("no valid pairs")
    consistent = bool(valid) and k_faster == len(valid)
    useful = med_delta is not None and med_delta <= -threshold
    if reasons:
        cls = NEEDS
    elif consistent and useful:
        cls = SPEEDUP
    else:
        cls = NO_BENEFIT[phase]
    out = {
        "phase": phase, "baseline": base, "candidate": cand, "classification": cls, "reasons": reasons,
        "smoke": bool(args.smoke), "repeats": result["repeats"], "replications": args.replications,
        "baseline_latency_ms": pooled[base], "candidate_latency_ms": pooled[cand],
        "pooled_speedup_baseline_over_candidate": (pooled[base]["median"] / pooled[cand]["median"])
        if pooled[base]["n"] and pooled[cand]["n"] else None,
        "n_valid_pairs": len(valid), "n_pairs_candidate_faster": k_faster, "n_pairs_baseline_faster": len(nz) - k_faster,
        "paired_median_delta_ms": med_delta,
        "paired_median_delta_pct": float(np.median([p["delta_pct"] for p in valid])) if valid else None,
        "paired_median_speedup": float(np.median([p["speedup_baseline_over_candidate"] for p in valid])) if valid else None,
        "sign_test_two_sided_p": p1d0b.sign_test_two_sided(k_faster, len(nz)),
        "useful_reduction_threshold_ms": threshold,
        "criteria": {"consistent_direction": consistent, "useful_reduction": useful},
        "semantic_counts_timed": counts, "baseline_nonrepeatable_calls": base_nonrep,
        "prevalidation": pre, "fidelity": {t: r["fidelity"] for t, r in result["ready"].items()},
        "mkldnn_state_mismatches": mk_bad,
        "n_planned_cells": len(plan), "n_attempts": len(attempts),
        "n_attempts_not_clean": sum(1 for a in attempts if not a["accepted"]),
        "not_accepted_cells": not_acc, "stopped": result.get("stopped"), "blocked": result.get("blocked"),
        "error": result["error"], "plan": plan, "attempts": attempts,
        "worker_ready": {t: {k: v for k, v in r.items() if k not in ("fidelity",)} for t, r in result["ready"].items()},
    }
    p1d0._write_json(phase_dir / "phase_result.json", out)
    return out


# =========================================================================== parent: run
def final_label(mk: dict, th: Optional[dict]) -> str:
    if mk["classification"] == NEEDS:
        return NEEDS
    if mk["classification"] == SPEEDUP:
        return SPEEDUP
    if th is None:
        return mk["classification"]
    if th["classification"] == NEEDS:
        return NEEDS
    return SPEEDUP if th["classification"] == SPEEDUP else "NO_MKLDNN_OR_THREAD_BENEFIT"


def summary_row(prof: dict, res: dict) -> dict:
    row = {"attack": prof["attack"], "profile_id": prof["profile_id"], "family": prof["family"],
           "eps": prof["eps"], "attack_params": json.dumps(prof["attack_params"], sort_keys=True),
           "eps_bound_gated": prof["eps_bound_gated"], "status": res.get("status"),
           "effective_repeats": res.get("effective_repeats"), "final_classification": res.get("final_classification"),
           "recommended_candidate": res.get("recommended_candidate", "")}
    for phase in ("mkldnn", "threads"):
        ph = res.get(phase)
        if not ph:
            row[f"{phase}_classification"] = res.get(f"{phase}_skipped", "")
            continue
        b, c = ph["baseline_latency_ms"], ph["candidate_latency_ms"]
        sc = ph["semantic_counts_timed"]
        row.update({
            f"{phase}_classification": ph["classification"],
            f"{phase}_baseline": ph["baseline"], f"{phase}_candidate": ph["candidate"],
            f"{phase}_baseline_median_ms": b["median"], f"{phase}_baseline_p05_ms": b["p05"],
            f"{phase}_baseline_p95_ms": b["p95"], f"{phase}_candidate_median_ms": c["median"],
            f"{phase}_candidate_p05_ms": c["p05"], f"{phase}_candidate_p95_ms": c["p95"],
            f"{phase}_paired_median_delta_ms": ph["paired_median_delta_ms"],
            f"{phase}_paired_median_speedup": ph["paired_median_speedup"],
            f"{phase}_pairs_candidate_faster": f"{ph['n_pairs_candidate_faster']}/{ph['n_valid_pairs']}",
            f"{phase}_sign_test_p": ph["sign_test_two_sided_p"],
            f"{phase}_n_bit_identical": sc["BIT_IDENTICAL"],
            f"{phase}_n_semantic_same": sc["BITWISE_DIFFERENT_SEMANTIC_SAME"],
            f"{phase}_n_semantic_different": sc["SEMANTIC_DIFFERENT"], f"{phase}_n_invalid": sc["INVALID"],
            f"{phase}_n_exception": sc["EXCEPTION"], f"{phase}_mkldnn_state_mismatches": ph["mkldnn_state_mismatches"],
            f"{phase}_baseline_nonrepeatable": ph["baseline_nonrepeatable_calls"],
            f"{phase}_cells_not_accepted": len(ph["not_accepted_cells"]),
            f"{phase}_attempts_not_clean": ph["n_attempts_not_clean"],
            f"{phase}_reasons": " | ".join(ph["reasons"])})
    mk = res.get("mkldnn") or {}
    fid = next(iter((mk.get("fidelity") or {}).values()), {}) if mk else {}
    row["anchor_apply_bit_identical"] = fid.get("n_apply_bit_identical")
    row["anchor_formal_sha_match"] = fid.get("n_formal_sha_match")
    row["anchor_success_rate"] = res.get("anchor_success_rate")
    return row


def write_run_summary(out_dir: Path, profiles: List[dict], smoke: bool) -> dict:
    rows, by_attack = [], {}
    for prof in profiles:
        p = out_dir / "profiles" / prof["slug"] / "profile_result.json"
        res = json.loads(p.read_text()) if p.exists() else {"status": "NOT_RUN"}
        rows.append(summary_row(prof, res))
        by_attack.setdefault(prof["attack"], []).append(rows[-1])
    write_csv(out_dir / "b1_acceleration_summary.csv", rows)
    conclusions = []
    for attack, rs in by_attack.items():
        labels = [r["final_classification"] for r in rs]
        if any(r["status"] != "complete" for r in rs):
            concl = "INCOMPLETE"
        elif any(l == NEEDS for l in labels):
            concl = NEEDS
        elif all(l == SPEEDUP for l in labels):
            cands = sorted({r["recommended_candidate"] for r in rs})
            concl = f"{SPEEDUP} ({'+'.join(cands)})" if len(cands) == 1 else f"{SPEEDUP} (per-profile candidates differ: {cands})"
        elif any(l == SPEEDUP for l in labels):
            concl = "MIXED_ACROSS_PROFILES"
        else:
            concl = "NO_SAFE_SPEEDUP_FOUND"
        conclusions.append({"attack": attack, "family": FAMILY.get(attack, "unknown"), "n_profiles": len(rs),
                            "profiles": " ".join(r["profile_id"] for r in rs),
                            "profile_classifications": " ".join(str(l) for l in labels),
                            "attack_conclusion": "SMOKE_ONLY (not formal): " + concl if smoke else concl})
    write_csv(out_dir / "attack_conclusions.csv", conclusions)
    summary = {"smoke": smoke, "profiles": rows, "attacks": conclusions,
               "integration_candidates": {r["attack"] + "/" + r["profile_id"]: r["recommended_candidate"]
                                          for r in rows if r["final_classification"] == SPEEDUP and not smoke}}
    p1d0._write_json(out_dir / "b1_acceleration_summary.json", summary)
    return summary


def run_profile(args, prof: dict, out_dir: Path, ctx_base: dict, repeats: int) -> dict:
    prof_dir = out_dir / "profiles" / prof["slug"]
    prof_dir.mkdir(parents=True, exist_ok=False)
    ctx = {**ctx_base, "profile": {k: v for k, v in prof.items() if k != "formal_profile_raw"},
           "formal_sha": ctx_base["formal_sha_by_profile"][prof["slug"]],
           "anchors_npz": p1a.rel(prof_dir / "anchors.npz"),
           "anchors_meta_json": p1a.rel(prof_dir / "anchors_meta.json")}
    ctx.pop("formal_sha_by_profile")
    ctx_file = prof_dir / "worker_context.json"
    ctx_file.write_text(json.dumps(ctx, indent=2, default=str))
    res = {"attack": prof["attack"], "profile_id": prof["profile_id"], "effective_repeats": repeats,
           "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}
    mk = run_phase(args, prof, "mkldnn", prof_dir, ctx_file, repeats)
    res["mkldnn"] = {k: v for k, v in mk.items() if k not in ("plan", "attempts")}
    th = None
    if args.smoke or (mk["classification"] == NO_BENEFIT["mkldnn"] and not args.skip_thread_phase):
        th = run_phase(args, prof, "threads", prof_dir, ctx_file, repeats)
        res["threads"] = {k: v for k, v in th.items() if k not in ("plan", "attempts")}
    else:
        res["threads_skipped"] = ("not needed (mkldnn result " + mk["classification"] + ")")
    res["final_classification"] = final_label(mk, th)
    res["recommended_candidate"] = ("mkldnn_OFF" if mk["classification"] == SPEEDUP else
                                    f"intra{THREAD_CAND}" if th and th["classification"] == SPEEDUP else "")
    am = prof_dir / "anchors_meta.json"
    if am.exists():
        meta = json.loads(am.read_text())
        res["anchor_success_rate"] = (sum(1 for v in meta.values() if v["success"]) / len(meta)) if meta else None
    res["status"] = "complete"
    res["finished_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    p1d0._write_json(prof_dir / "profile_result.json", res)
    return res


def parent_main(args) -> None:
    mode = "smoke" if args.smoke else "full"
    ctx = p1a.load_formal_context(args)
    cfg, fman, formal_dir = ctx["cfg"], ctx["manifest"], ctx["formal_dir"]
    full_subset = p1a.build_subset(cfg)
    subset = [full_subset[i] for i in SMOKE_SAMPLE_POSITIONS] if args.smoke else full_subset
    profiles = formal_profiles(cfg, args.include_controls)
    if args.attacks:
        want = {a.strip().lower() for a in args.attacks.split(",") if a.strip()}
        profiles = [p for p in profiles if p["attack"] in want]
    if args.profiles:  # targeted rerun: exact formal profiles, e.g. fab/eps0.005,fab/eps0.01
        want_p = {s.strip() for s in args.profiles.split(",") if s.strip()}
        unknown = want_p - {f"{p['attack']}/{p['profile_id']}" for p in profiles}
        if unknown:
            p1a.abort(f"--profiles not in the formal config (after --attacks/--include-controls): {sorted(unknown)}")
        profiles = [p for p in profiles if f"{p['attack']}/{p['profile_id']}" in want_p]
    if not profiles:
        p1a.abort("no profiles selected")
    plan_rows = []
    total_s = 0.0
    for prof in profiles:
        ms = est_call_ms(prof["attack"], prof["profile_id"])
        reps = effective_repeats(args, len(subset), ms)
        prof["_est_call_ms"], prof["_repeats"] = ms, reps
        cells = 4 * args.replications
        calls_mk = 2 * len(subset) + len(subset) + cells * (args.warmup_samples + reps * len(subset))
        calls_th = 2 * len(subset) + cells * (args.warmup_samples + reps * len(subset))
        s = (calls_mk + calls_th) * ms / 1000.0 * 1.1
        cool = 0 if args.smoke else 2 * cells * args.expected_cooldown_seconds
        total_s += s + cool + 3 * 45.0
        plan_rows.append({"profile": prof["slug"], "family": prof["family"], "est_call_ms": ms,
                          "effective_repeats": reps, "est_minutes_if_both_phases": (s + cool + 135) / 60})
    for r in plan_rows:
        p1a.log(f"   {r['profile']:32s} {r['family']:28s} est {r['est_call_ms']:8.1f} ms/call repeats={r['effective_repeats']}"
                f"  <= {r['est_minutes_if_both_phases']:.0f} min")
    p1a.log(f"[parent] {mode}: {len(profiles)} profiles / {len({p['attack'] for p in profiles})} attacks, "
            f"{len(subset)} samples, upper-bound estimate (thread phase for every profile) ~{total_s / 3600:.1f} h")
    now = p1d0.snapshot()
    if args.dry_run:
        print(json.dumps({"mode": mode, "profiles": plan_rows, "upper_bound_hours": total_s / 3600,
                          "system_now": now}, indent=2, default=str))
        return
    if not args.smoke and now["throttled"]["low"] is None:
        p1a.abort("vcgencmd get_throttled unavailable; THERMAL_CLEAN cannot be evaluated")

    hashes = {"dataset_sha256": p1a.sha256_file(REPO_ROOT / cfg["dataset_path"]),
              "checkpoint_sha256": p1a.sha256_file(REPO_ROOT / cfg["checkpoint_path"]),
              "config_sha256": p1a.sha256_file(ctx["cfg_path"]),
              "attack_adapter_py_sha256": p1a.sha256_file(REPO_ROOT / "src/adapters/attack_adapter.py"),
              "this_script_sha256": p1a.sha256_file(Path(__file__).resolve())}
    if hashes["dataset_sha256"] != fman.get("dataset_sha256") or hashes["checkpoint_sha256"] != fman.get("checkpoint_sha256"):
        p1a.abort("dataset/checkpoint sha256 differs from formal CPU manifest")
    run_key = {"mode": mode, "subset": [list(k) for k in subset], "replications": args.replications,
               "repeats": args.repeats, "max_cell_seconds": args.max_cell_seconds,
               "warmup_samples": args.warmup_samples, "config_sha256": hashes["config_sha256"],
               "include_controls": args.include_controls, "profiles": [p["slug"] for p in profiles],
               "autoattack_fixed_clock": args.autoattack_fixed_clock}

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume:
        out_dir = (REPO_ROOT / args.resume).resolve()
        man_path = out_dir / "manifest.json"
        if not man_path.exists():
            p1a.abort(f"--resume: {man_path} not found")
        manifest = json.loads(man_path.read_text())
        if manifest.get("run_key") != run_key:
            p1a.abort("--resume: run parameters differ from the original run (mode/subset/replications/repeats/config)")
        manifest.setdefault("resumes", []).append({"utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                                                   "command": [sys.executable] + sys.argv})
    else:
        out_dir = (REPO_ROOT / (args.out_dir or f"results/attack_b1_accel_{mode}_{ts}")).resolve()
        protected = list(p1a.PROTECTED_RESULT_DIRS) + [str(p.relative_to(REPO_ROOT)) for p in
                                                       (REPO_ROOT / "results").glob("accel_phase1*")]
        for pth in protected:
            prot = (REPO_ROOT / pth).resolve()
            if out_dir == prot or prot in out_dir.parents or out_dir in prot.parents:
                p1a.abort(f"refusing to write inside/over results directory {pth}")
        out_dir.mkdir(parents=True, exist_ok=False)
        manifest = {"benchmark": "attack_b1_acceleration_full", "mode": mode, "run_key": run_key,
                    "command": [sys.executable] + sys.argv,
                    "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "hashes": hashes,
                    "environment": p1a.environment_metadata(), "system_at_start": now,
                    "formal_dir": p1a.rel(formal_dir), "config": p1a.rel(ctx["cfg_path"]),
                    "threads": {"mkldnn_phase": {"intra": MKLDNN_THREADS, "interop": INTEROP_THREADS},
                                "thread_phase": {"baseline_intra": THREAD_BASE, "candidate_intra": THREAD_CAND,
                                                 "interop": INTEROP_THREADS}},
                    "decision_rule": {"min_useful_reduction_ms": args.min_useful_reduction_ms,
                                      "min_useful_reduction_pct": args.min_useful_reduction_pct},
                    "thermal_policy": None if args.smoke else {
                        "max_start_temp_c": args.max_start_temp_c, "release_consecutive_reads": args.release_consecutive_reads,
                        "cooldown_poll_seconds": args.cooldown_poll_seconds,
                        "max_cooldown_seconds": args.max_cooldown_seconds,
                        "max_start_rejections": args.max_start_rejections, "max_cell_attempts": args.max_cell_attempts,
                        "max_consecutive_skips": args.max_consecutive_skips,
                        "in_cell_vc_poll_seconds": args.in_cell_vc_poll_seconds,
                        "system_settings_changed": "none (read-only)"},
                    "profiles": [{k: v for k, v in p.items() if not k.startswith("_")} for p in profiles],
                    "effective_repeats": {p["slug"]: p["_repeats"] for p in profiles},
                    "estimate": plan_rows}
    p1d0._write_json(out_dir / "manifest.json", manifest)

    # formal rows + clean inputs (rebuilt once, verified against formal hashes)
    wanted = {(p["attack"], p["profile_id"]) for p in profiles}
    base_rows, attack_rows = load_formal_rows(formal_dir, set(subset), wanted)
    inputs_npz = out_dir / "clean_inputs.npz"
    if not inputs_npz.exists():
        inputs = p1a.rebuild_clean_inputs(subset, base_rows, p1a.resolve_sensing(cfg), REPO_ROOT / cfg["dataset_path"])
        np.savez(inputs_npz, **{f"x{i}": inputs[k]["x"] for i, k in enumerate(subset)})
    meta = [{"key": list(k), "seed": int(float(base_rows[k]["seed"])),
             "label": int(float(base_rows[k]["label"])) if "label" in base_rows[k] else None,
             "formal_clean_pred": int(float(base_rows[k]["clean_pred"])),
             "formal_clean_input_sha256": base_rows[k]["clean_input_sha256"]} for k in subset]
    if any(m["label"] is None for m in meta):
        from src.sensing.radioml_source import RML2016_10A_CLASSES
        for m in meta:
            m["label"] = RML2016_10A_CLASSES[m["key"][0]]
    ctx_base = {"subset": [list(k) for k in subset], "checkpoint_path": cfg["checkpoint_path"],
                "autoattack_fixed_clock": bool(args.autoattack_fixed_clock),
                "clean_inputs_npz": p1a.rel(inputs_npz), "clean_inputs_meta": meta,
                "formal_sha_by_profile": {p["slug"]: {"|".join(map(str, k)): attack_rows.get(
                    k + (p["attack"], p["profile_id"]), {}).get("adversarial_sha256") for k in subset}
                    for p in profiles}}

    for prof in profiles:
        prof_dir = out_dir / "profiles" / prof["slug"]
        done = prof_dir / "profile_result.json"
        if done.exists() and json.loads(done.read_text()).get("status") == "complete":
            p1a.log(f"[skip] {prof['slug']}: already complete")
            continue
        if prof_dir.exists():
            moved = prof_dir.with_name(prof_dir.name + f".incomplete_{ts}")
            prof_dir.rename(moved)
            p1a.log(f"[resume] moved incomplete {prof_dir.name} -> {moved.name}")
        problems = check_profile_against_formal(prof, subset, attack_rows)
        if problems:
            prof_dir.mkdir(parents=True)
            p1d0._write_json(done, {"attack": prof["attack"], "profile_id": prof["profile_id"], "status": "complete",
                                    "final_classification": NEEDS,
                                    "mkldnn_skipped": "formal parameter cross-check failed",
                                    "formal_cross_check_problems": problems[:20]})
            p1a.log(f"[{prof['slug']}] formal cross-check failed: {problems[:3]}")
            continue
        p1a.log(f"[{prof['slug']}] start ({prof['family']}, repeats={prof['_repeats']})")
        try:
            res = run_profile(args, prof, out_dir, ctx_base, prof["_repeats"])
            p1a.log(f"[{prof['slug']}] done: mkldnn={res['mkldnn']['classification']} "
                    f"threads={res.get('threads', {}).get('classification', res.get('threads_skipped'))} "
                    f"-> {res['final_classification']}")
        except Exception as exc:  # noqa: BLE001 - unattended: record and continue with the next profile
            p1a.log(f"[{prof['slug']}] ERROR {type(exc).__name__}: {exc}")
            (prof_dir / "profile_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        write_run_summary(out_dir, profiles, args.smoke)

    summary = write_run_summary(out_dir, profiles, args.smoke)
    manifest.update(status="complete", finished_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
                    environment_end=p1d0.snapshot())
    p1d0._write_json(out_dir / "manifest.json", manifest)
    p1a.log(f"[done] {out_dir}")
    for c in summary["attacks"]:
        p1a.log(f"  {c['attack']:10s} {c['attack_conclusion']}  [{c['profile_classifications']}]")
    if args.smoke:
        bad = [r for r in summary["profiles"] if r["status"] != "complete"
               or r.get("mkldnn_n_invalid") or r.get("mkldnn_n_exception") or r.get("mkldnn_mkldnn_state_mismatches")
               or r.get("anchor_apply_bit_identical") != len(subset)]
        p1a.log(f"[smoke] {'PASS' if not bad else 'FAIL'}: {len(summary['profiles']) - len(bad)}/"
                f"{len(summary['profiles'])} profiles ran end-to-end with real backends, anchors == production "
                f"apply(), no INVALID/exception/state leak")
        for r in bad:
            p1a.log(f"  [smoke FAIL] {r['attack']}/{r['profile_id']}: status={r['status']} "
                    f"reasons={r.get('mkldnn_reasons')}")
        if bad:
            sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="3 fixed samples, 1 replication, no thermal gate")
    mode.add_argument("--full", action="store_true", help="88-sample formal subset + thermal gate")
    ap.add_argument("--config", default=p1a.DEFAULT_CONFIG)
    ap.add_argument("--formal-dir", default=p1a.DEFAULT_FORMAL_CPU_DIR)
    ap.add_argument("--out-dir", default=None, help="default results/attack_b1_accel_<mode>_<timestamp>")
    ap.add_argument("--resume", default=None, help="existing run directory; completed profiles are skipped")
    ap.add_argument("--attacks", default=None, help="comma-separated subset of attacks (default: all non-control)")
    ap.add_argument("--profiles", default=None,
                    help="targeted rerun: comma-separated attack/profile_id keys from the formal config")
    ap.add_argument("--autoattack-fixed-clock", action="store_true",
                    help="autoattack only: replace AutoAttack's wall-clock seed by the formal per-row seed "
                         "(production apply() anchor and benchmark path alike); aborts if not applicable")
    ap.add_argument("--include-controls", action="store_true", help="also run the pgd/bim formal profiles")
    ap.add_argument("--skip-thread-phase", action="store_true")
    ap.add_argument("--replications", type=int, default=None, help="default 4 (--full) / 1 (--smoke)")
    ap.add_argument("--repeats", type=int, default=3, help="passes over the subset per cell (Phase 1D-3: 3)")
    ap.add_argument("--max-cell-seconds", type=float, default=180.0,
                    help="per profile, repeats is reduced (min 1) so one cell stays under this estimate")
    ap.add_argument("--warmup-samples", type=int, default=2)
    ap.add_argument("--min-useful-reduction-ms", type=float, default=1.0)
    ap.add_argument("--min-useful-reduction-pct", type=float, default=1.0)
    ap.add_argument("--max-start-temp-c", type=float, default=55.0)
    ap.add_argument("--release-consecutive-reads", type=int, default=2)
    ap.add_argument("--cooldown-poll-seconds", type=float, default=5.0)
    ap.add_argument("--max-cooldown-seconds", type=float, default=900.0)
    ap.add_argument("--max-start-rejections", type=int, default=5)
    ap.add_argument("--max-cell-attempts", type=int, default=3)
    ap.add_argument("--max-consecutive-skips", type=int, default=2)
    ap.add_argument("--expected-cooldown-seconds", type=float, default=75.0, help="planning assumption only")
    ap.add_argument("--in-cell-vc-poll-seconds", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-context", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--worker-threads", type=int, default=MKLDNN_THREADS, help=argparse.SUPPRESS)
    ap.add_argument("--phase", choices=["mkldnn", "threads"], default="mkldnn", help=argparse.SUPPRESS)
    ap.add_argument("--tag", default="w4", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker_main(args)
        return
    if not (args.smoke or args.full):
        ap.error("one of --smoke / --full is required")
    if args.replications is None:
        args.replications = 1 if args.smoke else 4
    if args.replications < 1 or args.repeats < 1 or args.warmup_samples < 2 or args.max_cell_attempts < 1:
        p1a.abort("--replications>=1, --repeats>=1, --warmup-samples>=2, --max-cell-attempts>=1 required")
    parent_main(args)


if __name__ == "__main__":
    main()
