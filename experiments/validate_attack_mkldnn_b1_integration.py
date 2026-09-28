"""
Regression validation of the integrated Phase 1 attack acceleration:
AttackAdapter(b1_pgd_bim_mkldnn_off=True) disables mkldnn ONLY around atk(x, y) for PGD/BIM at B=1.
Raspberry Pi 5, CPU only, intra 4 / interop 1. This is regression validation, not a performance study.

Evidence behind the option (Phase 1D-3, results/accel_phase1d3_mkldnn_b1_20260927_135420): PGD B=1
median 42.53 -> 36.33 ms, BIM 43.84 -> 37.63 ms, OFF faster in 8/8 thermally clean pairs each
(sign test p = 0.0078), PGD OFF bit-identical (88/88 + 2112 timed), BIM 87/88 bit-identical + the
known PAM4/6 dB/idx 1 BITWISE_DIFFERENT_SEMANTIC_SAME case, 0 SEMANTIC_DIFFERENT, 0 INVALID.

Sections (all through the real, integrated AttackAdapter.apply())
-----------------------------------------------------------------
A  Option OFF (default adapter): apply() must be bit-identical to the pre-change reference for FGSM,
   PGD and BIM on all 88 samples. Reference = experiments/bench_attack_overhead_pi5.py:attack_path
   (an independent statement-level replica of the UNCHANGED apply(), in-path clean prediction) under
   the same torch.manual_seed(formal_seed); FGSM/BIM must also equal the formal adversarial_sha256.
   The returned meta keys must be exactly the legacy key set and the global RNG state after the call
   must equal the reference's.
B  Option ON: PGD/BIM B=1 outputs classified against the option-OFF outputs of section A
   (BIT_IDENTICAL / BITWISE_DIFFERENT_SEMANTIC_SAME / SEMANTIC_DIFFERENT / INVALID, Phase 1C v2
   tolerances); meta flag must say applied; RNG state after the call must equal option OFF's.
   FGSM B=1 with the option ON must stay bit-identical (option not applied). Agreement with the
   Phase 1D-3 correctness pattern is reported. Any SEMANTIC_DIFFERENT or INVALID in PGD/BIM -> FAIL.
C  Scope guards, proven with runtime probes (a forward pre-hook on the AWN module records the mkldnn
   state of every model forward; a temporary wrapper around attack_adapter._build_torchattacks
   records the state at the moment atk() is entered):
     PGD B=1 / BIM B=1 (option ON): atk runs with mkldnn False; the in-path clean-prediction forward
       runs with True; the global state is True again after apply(); meta applied = True
     FGSM B=1 (option ON), PGD B=2, BIM B=2 (option ON), PGD B=1 (option OFF): atk runs with True
     clean inference (awn.infer) before/after: every forward with True
     previous-state restore: with the global state set to False beforehand, apply() leaves it False
     synthetic exception inside atk(): state restored (apply falls back as designed) and the
       mkldnn_disabled_scope helper restores after a raised exception
     model mode / requires_grad flags unchanged after every case
   All monkeypatches are in-memory, scoped with try/finally, and removed before section E.
E  Latency sanity (short, thermally gated, one block per attack): 88 samples x --latency-repeats,
   option OFF vs ON adapter alternating per sample (ABBA parity), full apply() wall time. Reports
   medians, paired per-sample deltas and the share of samples where ON was faster. Direction is
   CONFIRMED if the median paired delta < 0 and >= 90 % of paired samples are faster; the block's
   thermal status is reported and a non-clean block makes the direction check inconclusive.

Nothing on the system is changed. Nothing is written outside the new result directory.

Run on the Pi (from the repo root):
    .venv/bin/python experiments/validate_attack_mkldnn_b1_integration.py
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as _dt
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import bench_attack_accel_pi5 as p1a  # noqa: E402
import bench_attack_overhead_pi5 as p1b  # noqa: E402 -- pre-change apply() replica (reference)
import bench_attack_batch_pi5_v2 as p1c  # noqa: E402 -- V2 tolerances / helpers
import bench_attack_thermal_control_pi5 as p1d0  # noqa: E402 -- thermal snapshot / sampler / classifier
import bench_attack_thermal_pair_pi5 as p1d0b  # noqa: E402 -- cooldown release + start gate

REPO_ROOT = p1a.REPO_ROOT
ATTACKS_ALL = ["fgsm", "pgd", "bim"]
OPT_ATTACKS = ["pgd", "bim"]
INTRA_THREADS, INTEROP_THREADS = 4, 1
TEMPERATURE = 1.0
CLASSES = p1c.CLASSES
LEGACY_META_KEYS = {
    "attack_backend", "attack_status", "attack_notes", "attack_training_before", "attack_training_after",
    "attack_input_min", "attack_input_max", "attack_normalized_min", "attack_normalized_max",
    "attack_output_has_nan", "attack_output_has_inf", "attack_temperature", "attack_iq_linf_normalized",
    "attack_gradient_nonzero_count", "attack_gradient_total_count", "attack_gradient_maxabs",
    "cw_c", "cw_steps", "cw_lr"}
PHASE1D3_DIR = "results/accel_phase1d3_mkldnn_b1_20260927_135420"
PHASE1D3_EXPECTED = {"pgd": {"BIT_IDENTICAL": 88, "BITWISE_DIFFERENT_SEMANTIC_SAME": 0},
                     "bim": {"BIT_IDENTICAL": 87, "BITWISE_DIFFERENT_SEMANTIC_SAME": 1,
                             "semantic_same_samples": [["PAM4", 6, 1]]}}
PHASE1D3_PAIRED_DELTA_MS = {"pgd": -6.2015, "bim": -6.1940}
CORR_FIELDS = ["section", "attack", "variant", "sample_pos", "modulation", "snr_db", "sample_index", "formal_seed",
               "label", "clean_pred", "attacked_pred", "reference_attacked_pred", "attack_success",
               "reference_attack_success", "x_adv_sha256", "reference_sha256", "formal_adversarial_sha256",
               "bit_identical", "max_abs_diff", "n_diff_elements", "iq_linf", "reference_iq_linf", "iq_l2",
               "reference_iq_l2", "iq_linf_normalized", "norm_eps_bound_ok", "iq_eps_bound_ok", "finite",
               "shape_dtype_ok", "meta_keys_ok", "mkldnn_applied_flag", "rng_state_equal", "attack_status",
               "classification"]
LAT_FIELDS = ["attack", "rep", "sample_pos", "modulation", "snr_db", "sample_index", "variant", "order",
              "total_ms", "bit_identical_to_expected", "cpu_temp_c"]


def write_csv(path: Path, fields: List[str], rows: List[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run(args) -> None:
    import torch
    torch.set_num_interop_threads(INTEROP_THREADS)
    torch.set_num_threads(INTRA_THREADS)
    if (torch.get_num_threads(), torch.get_num_interop_threads()) != (INTRA_THREADS, INTEROP_THREADS):
        p1a.abort("thread settings not applied")
    import src.adapters.attack_adapter as am
    from src.adapters.attack_adapter import AttackAdapter, _REAL_ATTACK_SOURCE, mkldnn_disabled_scope
    from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE

    get_mk, set_mk = torch._C._get_mkldnn_enabled, torch._C._set_mkldnn_enabled
    if bool(get_mk()) is not True:
        p1a.abort("mkldnn is not enabled by default in this process")
    now = p1d0.snapshot()
    if now["throttled"]["low"] is None:
        p1a.abort("vcgencmd get_throttled unavailable")

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_dir = (f"results/accel_phase1_final_integration_statediag_{ts}" if args.diagnose_state
                   else f"results/accel_phase1_final_integration_{ts}")
    out_dir = (REPO_ROOT / (args.out_dir or default_dir)).resolve()
    for p in p1a.PROTECTED_RESULT_DIRS + (PHASE1D3_DIR,):
        prot = (REPO_ROOT / p).resolve()
        if out_dir == prot or prot in out_dir.parents or out_dir in prot.parents:
            p1a.abort(f"refusing to write inside/over {p}")
    out_dir.mkdir(parents=True, exist_ok=False)

    ctx = p1a.load_formal_context(args)
    cfg, fman, formal_dir = ctx["cfg"], ctx["manifest"], ctx["formal_dir"]
    subset = p1a.build_subset(cfg)
    base_rows, formal_attack = p1a.load_formal_rows(formal_dir, set(subset))
    settings = p1a.resolve_attack_settings(cfg, formal_attack, args.attack_temperature, args.diagnostics)
    for a in ATTACKS_ALL:
        if float(settings[a]["temperature"]) != TEMPERATURE:
            p1a.abort(f"{a}: attack temperature {settings[a]['temperature']} != {TEMPERATURE}")
    hashes = {"dataset_sha256": p1a.sha256_file(REPO_ROOT / cfg["dataset_path"]),
              "checkpoint_sha256": p1a.sha256_file(REPO_ROOT / cfg["checkpoint_path"]),
              "attack_adapter_py_sha256": p1a.sha256_file(REPO_ROOT / "src/adapters/attack_adapter.py"),
              "phase1b_reference_script_sha256": p1a.sha256_file(Path(p1b.__file__).resolve()),
              "this_script_sha256": p1a.sha256_file(Path(__file__).resolve())}
    if hashes["dataset_sha256"] != fman.get("dataset_sha256") or hashes["checkpoint_sha256"] != fman.get("checkpoint_sha256"):
        p1a.abort("dataset/checkpoint sha256 differs from formal CPU manifest")

    awn = AWNModelAdapter(checkpoint_path=str(REPO_ROOT / cfg["checkpoint_path"]), device="cpu")
    if awn.model is None or awn.backend_name != _REAL_MODEL_SOURCE or awn.status != "ok":
        p1a.abort("AWN backend is not real")
    legacy = AttackAdapter(awn_model=awn.model, device="cpu")                            # option OFF (default)
    opt = AttackAdapter(awn_model=awn.model, device="cpu", b1_pgd_bim_mkldnn_off=True)  # option ON
    for ad in (legacy, opt):
        if ad.wrapped_model is None or ad.backend_name != _REAL_ATTACK_SOURCE or ad.status != "ok":
            p1a.abort("attack backend is not real")
    if legacy.b1_pgd_bim_mkldnn_off is not False:
        p1a.abort("default AttackAdapter does not have the option disabled")
    inputs = p1a.rebuild_clean_inputs(subset, base_rows, p1a.resolve_sensing(cfg), REPO_ROOT / cfg["dataset_path"])

    def infer_pred(x) -> int:
        logits, meta = awn.infer(x)
        if meta["awn_backend"] != _REAL_MODEL_SOURCE or meta["awn_status"] != "ok" or not np.isfinite(logits).all():
            p1a.abort(f"AWN inference invalid: {meta}")
        return int(np.argmax(logits[0]))

    for key, inp in inputs.items():
        inp["clean_pred"] = infer_pred(inp["x"])
        if inp["clean_pred"] != inp["formal_clean_pred"]:
            p1a.abort(f"{key}: clean_pred != formal")

    # ---------------------------------------------------------------- explicit process/model state snapshots
    roots = {"awn": awn.model, "legacy_wrapper": legacy.wrapped_model, "opt_wrapper": opt.wrapped_model}

    def snapshot(tag: str) -> dict:
        nps = np.random.get_state()
        return {
            "tag": tag,
            "module_training": {f"{r}/{n or '<root>'}": m.training for r, root in roots.items()
                                for n, m in root.named_modules()},
            "param_requires_grad": {f"{r}/{n}": p.requires_grad for r, root in roots.items()
                                    for n, p in root.named_parameters()},
            "mkldnn_enabled": bool(get_mk()),
            "intra_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "cpu_rng_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            "numpy_rng_sha256": hashlib.sha256(nps[1].tobytes() + repr((nps[0], nps[2], nps[3], nps[4])).encode()).hexdigest(),
        }

    INVARIANT_FIELDS = ("module_training", "param_requires_grad", "mkldnn_enabled", "intra_threads",
                        "interop_threads", "numpy_rng_sha256")

    def state_diff(a: dict, b: dict, fields=INVARIANT_FIELDS) -> dict:
        out = {}
        for f in fields:
            va, vb = a[f], b[f]
            if isinstance(va, dict):
                ks = sorted(k for k in set(va) | set(vb) if va.get(k) != vb.get(k))
                if ks:
                    out[f] = {k: {"before": va.get(k), "after": vb.get(k)} for k in ks}
            elif va != vb:
                out[f] = {"before": va, "after": vb}
        return out

    state_report = {"fresh_adapters": None, "canonicalization": None, "call_mismatches": [],
                    "section_end_checks": {}}

    def save_state_report():
        (out_dir / "state_diagnostics.json").write_text(json.dumps(state_report, indent=2, default=str))

    # A freshly constructed Model01Wrapper is an nn.Module in its default training=True state until the
    # first apply()/replica call forces eval() in its finally block (pre-existing adapter behavior, not
    # part of the Phase 1 change). Record that, then put both wrappers into the canonical eval state so
    # every call starts from an equivalent state; afterwards every call must leave all invariant fields
    # unchanged.
    fresh = snapshot("fresh_adapters_after_construction")
    state_report["fresh_adapters"] = {
        "modules_in_training_mode": sorted(k for k, v in fresh["module_training"].items() if v),
        "mkldnn_enabled": fresh["mkldnn_enabled"], "intra_threads": fresh["intra_threads"],
        "interop_threads": fresh["interop_threads"]}
    legacy.wrapped_model.eval()
    opt.wrapped_model.eval()
    baseline = snapshot("canonical_baseline")
    state_report["canonicalization"] = {
        "action": "legacy.wrapped_model.eval(); opt.wrapped_model.eval() (validator only; equals what the first "
                  "apply() call does in its finally block)",
        "changed_fields": state_diff(fresh, baseline),
        "modules_in_training_mode_after": sorted(k for k, v in baseline["module_training"].items() if v)}
    save_state_report()
    if state_report["canonicalization"]["modules_in_training_mode_after"]:
        p1a.abort("modules still in training mode after canonicalization; see state_diagnostics.json")

    def check_invariants(where: str) -> None:
        """Strict: every invariant field must equal the canonical baseline."""
        d = state_diff(baseline, snapshot(where))
        state_report["section_end_checks"][where] = d or "unchanged"
        if d:
            save_state_report()
            p1a.abort(f"model/process state changed ({where}): {sorted(d)} -- see state_diagnostics.json")

    def call_mismatch(record: dict) -> None:
        state_report["call_mismatches"].append(record)
        save_state_report()
        p1a.abort(f"state mismatch in {record['where']}: {record['summary']} -- see state_diagnostics.json")

    def call(adapter, attack, x, seed, snaps=None):
        """Integrated apply() under the formal seed; its one status print per call is captured (both variants).
        snaps (optional dict) receives 'before' (taken right after manual_seed) and 'after' snapshots."""
        s = settings[attack]
        torch.manual_seed(seed)
        if snaps is not None:
            snaps["before"] = snapshot("before_apply")
        with contextlib.redirect_stdout(io.StringIO()):
            x_adv, meta = adapter.apply(x, attack=attack, eps=s["eps"], temperature=s["temperature"], seed=seed,
                                        diagnostics=s["diagnostics"], attack_params=dict(s["attack_params"]))
        if snaps is not None:
            snaps["after"] = snapshot("after_apply")
        return x_adv, meta, torch.get_rng_state().clone()

    def b_of(x):
        return am._iq_to_ta_input_minmax(torch.from_numpy(x))[2].detach().cpu().numpy().reshape(-1)

    corr_rows: List[dict] = []
    failures: List[str] = []

    # ---------------------------------------------------------------- A: option OFF == pre-change path
    legacy_out: Dict[str, Dict[tuple, dict]] = {a: {} for a in ATTACKS_ALL}
    sec_a = {}
    a_attacks = args.diagnose_attacks.split(",") if args.diagnose_state else ATTACKS_ALL
    a_subset = subset[: args.diagnose_state] if args.diagnose_state else subset
    for attack in a_attacks:
        n_ok = 0
        for pos, key in enumerate(a_subset):
            inp = inputs[key]
            torch.manual_seed(inp["seed"])
            s_rep_before = snapshot("before_replica")
            with contextlib.redirect_stdout(io.StringIO()):
                ref, _, _, _, _, _ = p1b.attack_path(am, legacy, torch, inp["x"], attack, settings[attack],
                                                     None, False, False)
            s_rep_after = snapshot("after_replica")
            ref_rng = torch.get_rng_state().clone()
            snaps: Dict[str, dict] = {}
            x_adv, meta, rng = call(legacy, attack, inp["x"], inp["seed"], snaps)
            diffs = {"replica_call_changed": state_diff(s_rep_before, s_rep_after),
                     "apply_call_changed": state_diff(snaps["before"], snaps["after"]),
                     "initial_states_differ": state_diff(s_rep_before, snaps["before"],
                                                         INVARIANT_FIELDS + ("cpu_rng_sha256",)),
                     "final_states_differ": state_diff(s_rep_after, snaps["after"],
                                                       INVARIANT_FIELDS + ("cpu_rng_sha256",))}
            if any(diffs.values()):
                call_mismatch({"where": f"A {attack} {list(key)}", "summary": {k: sorted(v) for k, v in diffs.items() if v},
                               "diffs": diffs, "state_before_replica": s_rep_before, "state_after_replica": s_rep_after,
                               "state_before_apply": snaps["before"], "state_after_apply": snaps["after"]})
            if meta["attack_status"] != "ok" or meta["attack_backend"] != _REAL_ATTACK_SOURCE:
                p1a.abort(f"A {attack} {key}: apply() fell back ({meta['attack_status']})")
            bit = bool(np.array_equal(ref, x_adv))
            sha = p1a.sha256_array(x_adv)
            formal_sha = formal_attack[key + (attack, settings[attack]["profile"])].get("adversarial_sha256") \
                if attack in p1a.DETERMINISTIC_ATTACKS else ""
            formal_ok = (sha == formal_sha) if formal_sha else True
            keys_ok = set(meta) == LEGACY_META_KEYS
            rng_ok = bool(torch.equal(ref_rng, rng))
            pred = infer_pred(x_adv)
            d = x_adv - inp["x"]
            legacy_out[attack][key] = {"x_adv": x_adv, "sha": sha, "pred": pred, "rng": rng,
                                       "success": bool(inp["clean_pred"] == inp["label"] and pred != inp["label"]),
                                       "linf": float(np.max(np.abs(d))), "l2": float(np.linalg.norm(d.reshape(-1)))}
            ok = bit and formal_ok and keys_ok and rng_ok
            n_ok += ok
            corr_rows.append({"section": "A_option_off_vs_prechange", "attack": attack, "variant": "option_off",
                              "sample_pos": pos, "modulation": key[0], "snr_db": key[1], "sample_index": key[2],
                              "formal_seed": inp["seed"], "label": inp["label"], "clean_pred": inp["clean_pred"],
                              "attacked_pred": pred, "x_adv_sha256": sha, "reference_sha256": p1a.sha256_array(ref),
                              "formal_adversarial_sha256": formal_sha, "bit_identical": bit,
                              "max_abs_diff": float(np.max(np.abs(ref.astype(np.float64) - x_adv))),
                              "meta_keys_ok": keys_ok, "rng_state_equal": rng_ok, "attack_status": meta["attack_status"],
                              "classification": "BIT_IDENTICAL" if ok else "MISMATCH"})
            if not ok:
                failures.append(f"A {attack} {key}: bit={bit} formal_sha={formal_ok} meta_keys={keys_ok} rng={rng_ok}")
        check_invariants(f"end of section A ({attack})")
        sec_a[attack] = {"n": len(a_subset), "n_identical": n_ok, "status": "PASS" if n_ok == len(a_subset) else "FAIL"}
        p1a.log(f"[A] {attack}: {sec_a[attack]}")
    save_state_report()
    if args.diagnose_state:
        ok = all(v["status"] == "PASS" for v in sec_a.values()) and not failures
        (out_dir / "validation.json").write_text(json.dumps(
            {"mode": "diagnose_state", "section_A": sec_a, "failures": failures,
             "fresh_adapters": state_report["fresh_adapters"], "overall": "PASS" if ok else "FAIL"}, indent=2, default=str))
        write_csv(out_dir / "correctness.csv", CORR_FIELDS, corr_rows)
        p1a.log(f"[diagnose] fresh-adapter modules in training mode: {state_report['fresh_adapters']['modules_in_training_mode']}")
        p1a.log(f"[diagnose] canonicalization changed: {sorted(state_report['canonicalization']['changed_fields'])}")
        p1a.log(f"[diagnose] per-call state mismatches: {len(state_report['call_mismatches'])}; section A: {sec_a}")
        p1a.log(f"[diagnose] wrote {out_dir}")
        sys.exit(0 if ok else 2)

    # ---------------------------------------------------------------- B: option ON vs option OFF
    sec_b = {}
    eps_tol = (p1c.NORM_EPS_TOL, p1c.IQ_EPS_REL_TOL, p1c.IQ_EPS_ABS_TOL)
    opt_out: Dict[str, Dict[tuple, str]] = {a: {} for a in ATTACKS_ALL}
    for attack in ATTACKS_ALL:
        counts = {c: 0 for c in CLASSES}
        semsame = []
        n_flag_ok = n_rng_ok = 0
        for pos, key in enumerate(subset):
            inp, anc = inputs[key], legacy_out[attack][key]
            snaps = {}
            x_adv, meta, rng = call(opt, attack, inp["x"], inp["seed"], snaps)
            d_call = state_diff(snaps["before"], snaps["after"])
            if d_call:
                call_mismatch({"where": f"B {attack} {list(key)}", "summary": sorted(d_call), "diffs": d_call,
                               "state_before_apply": snaps["before"], "state_after_apply": snaps["after"]})
            if meta["attack_status"] != "ok" or meta["attack_backend"] != _REAL_ATTACK_SOURCE:
                p1a.abort(f"B {attack} {key}: apply() fell back ({meta['attack_status']})")
            applied = meta.get("attack_b1_pgd_bim_mkldnn_off_applied")
            flag_ok = applied is (attack in OPT_ATTACKS)
            rng_ok = bool(torch.equal(anc["rng"], rng))
            n_flag_ok += flag_ok
            n_rng_ok += rng_ok
            eps = float(settings[attack]["eps"])
            shape_ok = x_adv.shape == anc["x_adv"].shape and x_adv.dtype == np.float32
            fin = bool(np.isfinite(x_adv).all())
            bit = bool(np.array_equal(x_adv, anc["x_adv"]))
            pred = anc["pred"] if bit else (infer_pred(x_adv) if fin else -1)
            succ = bool(inp["clean_pred"] == inp["label"] and pred != inp["label"])
            d = x_adv - inp["x"]
            linf, l2 = float(np.max(np.abs(d))), float(np.linalg.norm(d.reshape(-1)))
            lin = float(np.asarray(meta["attack_iq_linf_normalized"]).reshape(-1)[0])
            bnd = eps * float(b_of(inp["x"])[0])
            norm_ok = lin <= eps + eps_tol[0]
            iq_ok = linf <= bnd * (1 + eps_tol[1]) + eps_tol[2]
            if not (fin and shape_ok and norm_ok and iq_ok):
                cls = "INVALID"
            elif bit:
                cls = "BIT_IDENTICAL"
            elif pred == anc["pred"] and succ == anc["success"]:
                cls = "BITWISE_DIFFERENT_SEMANTIC_SAME"
                semsame.append(list(key))
            else:
                cls = "SEMANTIC_DIFFERENT"
            counts[cls] += 1
            opt_out[attack][key] = p1a.sha256_array(x_adv)
            corr_rows.append({"section": "B_option_on_vs_option_off", "attack": attack, "variant": "option_on",
                              "sample_pos": pos, "modulation": key[0], "snr_db": key[1], "sample_index": key[2],
                              "formal_seed": inp["seed"], "label": inp["label"], "clean_pred": inp["clean_pred"],
                              "attacked_pred": pred, "reference_attacked_pred": anc["pred"], "attack_success": succ,
                              "reference_attack_success": anc["success"], "x_adv_sha256": opt_out[attack][key],
                              "reference_sha256": anc["sha"], "bit_identical": bit,
                              "max_abs_diff": float(np.max(np.abs(anc["x_adv"].astype(np.float64) - x_adv))) if fin else "inf",
                              "n_diff_elements": int(np.sum(x_adv != anc["x_adv"])) if shape_ok else -1,
                              "iq_linf": linf, "reference_iq_linf": anc["linf"], "iq_l2": l2, "reference_iq_l2": anc["l2"],
                              "iq_linf_normalized": lin, "norm_eps_bound_ok": norm_ok, "iq_eps_bound_ok": iq_ok,
                              "finite": fin, "shape_dtype_ok": shape_ok, "mkldnn_applied_flag": applied,
                              "rng_state_equal": rng_ok, "attack_status": meta["attack_status"], "classification": cls})
            if cls == "INVALID":
                write_csv(out_dir / "correctness.csv", CORR_FIELDS, corr_rows)
                p1a.abort(f"B {attack} {key}: INVALID output with the option ON")
        check_invariants(f"end of section B ({attack})")
        if attack in OPT_ATTACKS:
            exp = PHASE1D3_EXPECTED[attack]
            consistent = (counts["BIT_IDENTICAL"] == exp["BIT_IDENTICAL"]
                          and counts["BITWISE_DIFFERENT_SEMANTIC_SAME"] == exp["BITWISE_DIFFERENT_SEMANTIC_SAME"]
                          and sorted(semsame) == sorted(exp.get("semantic_same_samples", [])))
            status = "PASS" if (counts["SEMANTIC_DIFFERENT"] == 0 and counts["INVALID"] == 0
                                and n_flag_ok == len(subset) and n_rng_ok == len(subset)) else "FAIL"
        else:
            consistent = None
            status = "PASS" if (counts["BIT_IDENTICAL"] == len(subset) and n_flag_ok == len(subset)
                                and n_rng_ok == len(subset)) else "FAIL"
        sec_b[attack] = {"counts": counts, "semantic_same_samples": semsame, "applied_flag_ok": n_flag_ok,
                         "rng_state_equal": n_rng_ok, "consistent_with_phase1d3": consistent, "status": status}
        if status != "PASS":
            failures.append(f"B {attack}: {sec_b[attack]}")
        p1a.log(f"[B] {attack}: {sec_b[attack]}")

    # ---------------------------------------------------------------- C: scope guards with runtime probes
    probe = {"active": False, "forward_states": [], "atk_states": []}

    def pre_hook(_m, _inp):
        if probe["active"]:
            probe["forward_states"].append(bool(get_mk()))

    class _Proxy:
        def __init__(self, atk, raise_exc=False):
            self._atk, self._raise = atk, raise_exc

        def __call__(self, *a, **k):
            probe["atk_states"].append(bool(get_mk()))
            probe["forward_states"].append("ATK_START")
            if self._raise:
                raise RuntimeError("synthetic exception inside atk() (scope-guard test)")
            return self._atk(*a, **k)

    orig_build = am._build_torchattacks
    raise_flag = {"on": False}

    def build_probe(*a, **k):
        return _Proxy(orig_build(*a, **k), raise_flag["on"])

    handle = awn.model.register_forward_pre_hook(pre_hook)
    checks: Dict[str, dict] = {}
    k1, k2 = subset[0], subset[1]
    x1 = inputs[k1]["x"]
    x2 = np.concatenate([inputs[k1]["x"], inputs[k2]["x"]], axis=0)

    def probed(name, adapter, attack, x, expect_atk, expect_applied, expect_after=True, raise_exc=False):
        probe.update(active=True, forward_states=[], atk_states=[])
        raise_flag["on"] = raise_exc
        before_state = bool(get_mk())
        try:
            snaps = {}
            _, meta, _ = call(adapter, attack, x, inputs[k1]["seed"], snaps)
        finally:
            probe["active"] = False
            raise_flag["on"] = False
        fs = probe["forward_states"]
        split = fs.index("ATK_START") if "ATK_START" in fs else len(fs)
        pre_forward, atk_forward = fs[:split], [v for v in fs[split + 1:]]
        after = bool(get_mk())
        applied = meta.get("attack_b1_pgd_bim_mkldnn_off_applied", None)
        rec = {"attack": attack, "batch_size": int(x.shape[0]),
               "adapter": "option_on" if adapter is opt else "option_off",
               "global_state_before": before_state, "atk_entry_states": probe["atk_states"],
               "forward_states_before_atk": pre_forward, "forward_states_during_and_after_atk": atk_forward,
               "global_state_after": after, "meta_applied": applied, "attack_status": meta["attack_status"],
               "state_changed_by_call": state_diff(snaps["before"], snaps["after"])}
        rec["model_state_ok"] = not rec["state_changed_by_call"]
        ok = (probe["atk_states"] == [expect_atk] and after is expect_after and rec["model_state_ok"]
              and all(v is before_state for v in pre_forward))
        if not raise_exc:  # the first forward after atk() is entered belongs to the attack
            ok = ok and len(atk_forward) > 0 and (atk_forward[0] is expect_atk)
        if expect_applied is not None:
            ok = ok and applied is expect_applied
        rec["pass"] = bool(ok)
        checks[name] = rec
        if not ok:
            failures.append(f"C {name}: {rec}")

    am._build_torchattacks = build_probe
    try:
        probe.update(active=True, forward_states=[])
        infer_pred(x1)
        checks["clean_inference_before"] = {"forward_states": list(probe["forward_states"]),
                                            "pass": probe["forward_states"] == [True] * len(probe["forward_states"])
                                            and len(probe["forward_states"]) > 0}
        probe["active"] = False
        probed("pgd_b1_option_on", opt, "pgd", x1, False, True)
        probed("bim_b1_option_on", opt, "bim", x1, False, True)
        probed("fgsm_b1_option_on_not_applied", opt, "fgsm", x1, True, False)
        probed("pgd_b2_option_on_not_applied", opt, "pgd", x2, True, False)
        probed("bim_b2_option_on_not_applied", opt, "bim", x2, True, False)
        probed("pgd_b1_option_off_default", legacy, "pgd", x1, True, None)
        probed("bim_b1_option_off_default", legacy, "bim", x1, True, None)
        probe.update(active=True, forward_states=[])
        infer_pred(x1)
        checks["clean_inference_after"] = {"forward_states": list(probe["forward_states"]),
                                           "pass": probe["forward_states"] == [True] * len(probe["forward_states"])
                                           and len(probe["forward_states"]) > 0}
        probe["active"] = False
        # previous-state restore: global False beforehand must stay False afterwards
        set_mk(False)
        try:
            probed("pgd_b1_option_on_previous_state_false", opt, "pgd", x1, False, True, expect_after=False)
        finally:
            set_mk(True)
        # synthetic exception inside atk(): apply() falls back by design; the state must be restored
        probed("pgd_b1_option_on_synthetic_exception", opt, "pgd", x1, False, None, raise_exc=True)
        checks["pgd_b1_option_on_synthetic_exception"]["expected_fallback"] = \
            checks["pgd_b1_option_on_synthetic_exception"]["attack_status"] == "fallback"
        if not checks["pgd_b1_option_on_synthetic_exception"]["expected_fallback"]:
            checks["pgd_b1_option_on_synthetic_exception"]["pass"] = False
            failures.append("C synthetic exception: apply() did not report fallback")
    finally:
        am._build_torchattacks = orig_build
        handle.remove()
        set_mk(True)
    # direct helper test
    state_inside, raised = None, False
    try:
        with mkldnn_disabled_scope(torch):
            state_inside = bool(get_mk())
            raise ValueError("synthetic")
    except ValueError:
        raised = True
    checks["helper_restores_after_exception"] = {"state_inside": state_inside, "exception_propagated": raised,
                                                 "state_after": bool(get_mk()),
                                                 "pass": state_inside is False and raised and bool(get_mk()) is True}
    if not checks["helper_restores_after_exception"]["pass"]:
        failures.append("C helper_restores_after_exception")
    if am._build_torchattacks is not orig_build:
        p1a.abort("monkeypatch not removed")
    check_invariants("end of section C")
    scope_ok = all(v.get("pass") for v in checks.values())
    (out_dir / "scope_checks.json").write_text(json.dumps({"all_pass": scope_ok, "checks": checks}, indent=2, default=str))
    write_csv(out_dir / "correctness.csv", CORR_FIELDS, corr_rows)
    p1a.log(f"[C] scope checks all_pass={scope_ok}")

    # ---------------------------------------------------------------- E: latency sanity (thermally gated)
    trace: List[dict] = []
    lat_rows: List[dict] = []
    blocks = []
    lat_summary = {}
    for attack in OPT_ATTACKS:
        rec = {"block": f"latency_{attack}", "start_rejections": 0}
        while True:
            cd = p1d0b.wait_until_cool(args, trace, rec["block"])
            if cd["outcome"] != "REACHED":
                break
            before = p1d0.snapshot()
            if not p1d0b.start_gate(before, args.max_start_temp_c):
                break
            rec["start_rejections"] += 1
            if rec["start_rejections"] > args.max_start_rejections:
                cd = {**cd, "outcome": "START_GATE_EXHAUSTED"}
                break
        rec["cooldown"] = cd
        if cd["outcome"] != "REACHED":
            rec.update(thermal_status="NOT_RUN", reasons=[cd["outcome"]])
            blocks.append(rec)
            lat_summary[attack] = {"direction": "NOT_RUN", "reason": cd["outcome"]}
            continue
        sampler = p1d0.CellSampler(rec["block"], len(blocks), 2.0, 0.0).start()
        try:
            for key in subset[:4]:
                for ad in (legacy, opt):
                    call(ad, attack, inputs[key]["x"], inputs[key]["seed"])
            for rep in range(args.latency_repeats):
                for pos, key in enumerate(subset):
                    order = (legacy, opt) if (rep + pos) % 2 == 0 else (opt, legacy)
                    for oi, ad in enumerate(order):
                        variant = "option_off" if ad is legacy else "option_on"
                        temp = p1a.cpu_temp_c()
                        t0 = time.perf_counter()
                        x_adv, _, _ = call(ad, attack, inputs[key]["x"], inputs[key]["seed"])
                        dt = (time.perf_counter() - t0) * 1e3
                        exp_sha = legacy_out[attack][key]["sha"] if ad is legacy else opt_out[attack][key]
                        lat_rows.append({"attack": attack, "rep": rep, "sample_pos": pos, "modulation": key[0],
                                         "snr_db": key[1], "sample_index": key[2], "variant": variant, "order": oi,
                                         "total_ms": dt, "bit_identical_to_expected": p1a.sha256_array(x_adv) == exp_sha,
                                         "cpu_temp_c": temp})
        finally:
            samples = sampler.stop()
        after = p1d0.snapshot()
        for s_ in samples:
            trace.append({"phase": "block", "planned_cell_id": rec["block"], "attempt_id": rec["block"],
                          "order_index": s_["order_index"], **{k: s_[k] for k in p1d0b.TRACE_FIELDS[4:]}})
        status, reasons = p1d0.classify_thermal(before, after, samples, cd, args.max_start_temp_c)
        temps = [s_["temp_c"] for s_ in samples if s_["temp_c"] is not None]
        rec.update(thermal_status=status, reasons=reasons, temp_before_c=before["temp_c"], temp_after_c=after["temp_c"],
                   temp_max_c=max(temps) if temps else None, throttled_before=before["throttled"]["value_hex"],
                   throttled_after=after["throttled"]["value_hex"], arm_clock_hz_after=after["arm_clock_hz"],
                   governor_unchanged=before["governor"] == after["governor"])
        blocks.append(rec)
        rows_a = [r for r in lat_rows if r["attack"] == attack]
        off = {(r["rep"], r["sample_pos"]): r["total_ms"] for r in rows_a if r["variant"] == "option_off"}
        on = {(r["rep"], r["sample_pos"]): r["total_ms"] for r in rows_a if r["variant"] == "option_on"}
        deltas = [on[k] - off[k] for k in off if k in on]
        faster = float(np.mean([d < 0 for d in deltas])) if deltas else float("nan")
        med = float(np.median(deltas)) if deltas else float("nan")
        outputs_ok = all(r["bit_identical_to_expected"] for r in rows_a)
        confirmed = med < 0 and faster >= 0.9
        lat_summary[attack] = {
            "option_off_median_ms": float(np.median(list(off.values()))),
            "option_on_median_ms": float(np.median(list(on.values()))),
            "option_off_p95_ms": float(np.percentile(list(off.values()), 95)),
            "option_on_p95_ms": float(np.percentile(list(on.values()), 95)),
            "n_pairs": len(deltas), "median_paired_delta_ms": med, "share_pairs_option_on_faster": faster,
            "phase1d3_median_paired_delta_ms": PHASE1D3_PAIRED_DELTA_MS[attack],
            "outputs_reproduced_bitwise": outputs_ok, "thermal_status": status,
            "direction": ("CONFIRMED" if confirmed else "NOT_CONFIRMED") if status == "THERMAL_CLEAN"
            else "INCONCLUSIVE_THERMAL_REVIEW",
            "note": "full apply() wall time incl. its in-path clean forward and status print; absolute values are not "
                    "comparable to the Phase 1D-3 benchmark path"}
        if not outputs_ok:
            failures.append(f"E {attack}: outputs not reproduced bit-for-bit during latency sanity")
        p1a.log(f"[E] {attack}: {lat_summary[attack]}")
    write_csv(out_dir / "latency_sanity.csv", LAT_FIELDS, lat_rows)
    write_csv(out_dir / "thermal_trace.csv", p1d0b.TRACE_FIELDS, trace)
    with (out_dir / "thermal_blocks.csv").open("w", newline="") as f:
        fields = ["block", "thermal_status", "reasons", "start_rejections", "temp_before_c", "temp_after_c",
                  "temp_max_c", "throttled_before", "throttled_after", "arm_clock_hz_after", "governor_unchanged"]
        w = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
        w.writeheader()
        for b in blocks:
            w.writerow({**b, "reasons": " | ".join(b.get("reasons") or [])})

    correctness_pass = all(v["status"] == "PASS" for v in sec_a.values()) and \
        all(v["status"] == "PASS" for v in sec_b.values())
    overall = "PASS" if correctness_pass and scope_ok and not failures else "FAIL"
    validation = {
        "overall": overall, "section_A_option_off_vs_prechange": sec_a, "section_B_option_on": sec_b,
        "section_C_scope_checks_all_pass": scope_ok, "section_E_latency_sanity": lat_summary,
        "latency_direction_confirmed_all": all(v.get("direction") == "CONFIRMED" for v in lat_summary.values()),
        "failures": failures,
        "rules": {"fail_on": "any SEMANTIC_DIFFERENT or INVALID for PGD/BIM B=1 with the option ON; any option-OFF "
                             "deviation from the pre-change path; any scope-guard failure",
                  "latency": "sanity only; direction reported, not a new performance study"},
    }
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2, default=str))
    manifest = {"phase": "accel_phase1_final_integration", "command": [sys.executable] + sys.argv,
                "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "hashes": hashes,
                "environment": p1a.environment_metadata(), "system_at_start": now, "attacks": settings,
                "threads": {"intra": INTRA_THREADS, "interop": INTEROP_THREADS},
                "option": "AttackAdapter(b1_pgd_bim_mkldnn_off=True): mkldnn disabled only around atk(x, y) for "
                          "pgd/bim with N == 1 (src/adapters/attack_adapter.py)",
                "phase1d3_reference_dir": PHASE1D3_DIR, "latency_repeats": args.latency_repeats,
                "thermal_blocks": blocks, "validation_overall": overall}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    p1a.log(f"[done] {out_dir}  overall={overall}  latency_direction={ {a: v.get('direction') for a, v in lat_summary.items()} }")
    if overall != "PASS":
        for f_ in failures[:20]:
            p1a.log(f"  FAIL: {f_}")
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=p1a.DEFAULT_CONFIG)
    ap.add_argument("--formal-dir", default=p1a.DEFAULT_FORMAL_CPU_DIR)
    ap.add_argument("--out-dir", default=None, help="default results/accel_phase1_final_integration_<timestamp>")
    ap.add_argument("--latency-repeats", type=int, default=1)
    ap.add_argument("--diagnose-state", type=int, default=0,
                    help="diagnostic mode: run only section A on the first N samples with full state snapshots, "
                         "write state_diagnostics.json and stop (0 = full validation)")
    ap.add_argument("--diagnose-attacks", default="fgsm", help="attacks for --diagnose-state (subset of fgsm,pgd,bim)")
    ap.add_argument("--max-start-temp-c", type=float, default=55.0)
    ap.add_argument("--release-consecutive-reads", type=int, default=2)
    ap.add_argument("--cooldown-poll-seconds", type=float, default=5.0)
    ap.add_argument("--max-cooldown-seconds", type=float, default=900.0)
    ap.add_argument("--max-start-rejections", type=int, default=5)
    ap.add_argument("--attack-temperature", type=float, default=None)
    ap.add_argument("--diagnostics", choices=["auto", "on", "off"], default="auto")
    args = ap.parse_args()
    if args.latency_repeats < 1 or args.diagnose_state < 0:
        p1a.abort("--latency-repeats >= 1 and --diagnose-state >= 0 required")
    if any(a not in ATTACKS_ALL for a in args.diagnose_attacks.split(",")):
        p1a.abort("--diagnose-attacks must be a subset of fgsm,pgd,bim")
    run(args)


if __name__ == "__main__":
    main()
