"""
Phase 1B attack-path overhead-removal benchmark (Raspberry Pi 5, CPU only).

Question answered (and ONLY this question): without changing the mathematical
attack definition, how much B=1 attack-path latency is removed by eliminating
avoidable runtime overhead?

Variants (all B=1, intra-op=1 / inter-op=1, one fresh worker process):
  apply_untouched          unchanged src.adapters.attack_adapter.AttackAdapter.apply
                           (reference anchor; total latency only)
  baseline                 step-for-step replica of apply()'s real-attack branch
                           (must be bit-identical to apply_untouched on every call)
  reuse_clean_pred         baseline, but the attack label is the clean prediction the
                           pipeline already produced (AWNModelAdapter.infer on the SAME
                           x_clean) instead of apply()'s duplicate clean forward
  freeze_model_params      baseline, but every AWN parameter has requires_grad=False
                           while the attack object is built and run, so autograd only
                           differentiates w.r.t. the input; original flags restored
  reuse_clean_pred+freeze_model_params
  baseline_no_print        logging ablation ONLY: baseline minus the per-call status
                           print; never combined with the algorithmic variants

Not tested here (deliberately): attack-object reuse, batching, any change to
steps/eps/alpha/random_start/normalization/temperature, any source edit.

Correctness (fail closed by default): torch.manual_seed(formal per-row seed) is
reset immediately before EVERY variant call, so PGD's random_start draw is
identical across variants. Every variant's x_adv must be bit-identical to the
baseline's x_adv for the same (sample, attack, repeat) -- including PGD -- and
baseline must be bit-identical to apply_untouched. Any difference aborts the run
(or, with --on-mismatch record, marks the variant REJECTED and withholds its
speedup).

Reuses the validated Phase 1A helpers from experiments/bench_attack_accel_pi5.py
by import (formal config/row loading, sensing reconstruction with the
clean_input_sha256 fail-closed check, statistics, environment metadata). That
script is not modified.

Run on the Pi (from the repo root):
    .venv/bin/python experiments/bench_attack_overhead_pi5.py
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
from typing import Dict, List, Optional

import numpy as np

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import bench_attack_accel_pi5 as p1a  # noqa: E402 -- validated Phase 1A helpers (module import has no torch side effects)

REPO_ROOT = p1a.REPO_ROOT
INTRA_THREADS = 1
INTEROP_THREADS = 1
CONDITION = "intra1_interop1"

V_APPLY = "apply_untouched"
V_BASE = "baseline"
V_REUSE = "reuse_clean_pred"
V_FREEZE = "freeze_model_params"
V_BOTH = "reuse_clean_pred+freeze_model_params"
V_NOPRINT = "baseline_no_print"

# variant -> (reuse_clean_pred, freeze_model_params, print_status_line)
VARIANT_FLAGS = {
    V_BASE: (False, False, True),
    V_REUSE: (True, False, True),
    V_FREEZE: (False, True, True),
    V_BOTH: (True, True, True),
    V_NOPRINT: (False, False, False),
}
VARIANT_KIND = {
    V_APPLY: "reference (unchanged AttackAdapter.apply)",
    V_BASE: "baseline (apply-equivalent replica)",
    V_REUSE: "algorithmic-overhead removal",
    V_FREEZE: "algorithmic-overhead removal",
    V_BOTH: "algorithmic-overhead removal",
    V_NOPRINT: "runtime/logging ablation (separate; not combined with algorithmic variants)",
}

PHASES = ["phase_A_prep_ms", "phase_B_clean_pred_ms", "phase_freeze_ms", "phase_C_construct_ms",
          "phase_D_attack_ms", "phase_diag_ms", "phase_E_post_ms", "print_ms"]

RAW_FIELDS = [
    "condition", "intra_threads", "interop_threads",
    "attack", "attack_profile", "eps", "modulation", "snr_db", "sample_index", "formal_seed",
    "repeat", "variant", "variant_position", "torch_call_seed",
    "total_ms", *PHASES,
    "y_pred_used", "y_pred_source", "clean_pred_pipeline",
    "attacked_pred", "label", "clean_correct", "attacked_correct", "attack_success",
    "iq_linf", "iq_l2", "iq_linf_normalized", "x_adv_sha256", "x_adv_shape", "x_adv_dtype", "x_adv_finite",
    "bit_identical_to_baseline", "max_abs_diff_vs_baseline", "attacked_pred_eq_baseline",
    "attack_success_eq_baseline", "iq_linf_eq_baseline", "iq_l2_eq_baseline", "sha256_eq_baseline",
    "requires_grad_restored", "n_params_requires_grad_true_before", "cpu_temp_c",
]

SYSTEM_INTERPRETATION = (
    "In the formal pipeline the clean AWN inference (clean_*_inference_ms) already runs BEFORE attack "
    "generation, and its prediction is what reuse_clean_pred feeds to the attack. AttackAdapter.apply() "
    "currently repeats that clean forward internally (phase B). reuse_clean_pred therefore removes a "
    "DUPLICATE forward from the attack path; it does not remove the pipeline's own clean inference. An "
    "end-to-end interpretation is: sensing + clean inference (unchanged, counted once) + attack path "
    "(variant total_ms, which for reuse variants contains only the label-tensor construction in phase B) "
    "+ attacked inference. Never add phase B's baseline cost back on top of the pipeline's clean inference."
)


# --------------------------------------------------------------------------- attack path variants
def attack_path(am, adapter, torch, x, attack, s, reuse_pred: Optional[int], freeze: bool, do_print: bool):
    """Replica of AttackAdapter.apply()'s real-attack branch (same helpers, same
    order, same state capture/restoration) with three switches:
      reuse_pred  -> skip the in-path clean forward; label = this pipeline prediction
      freeze      -> requires_grad_(False) on every wrapped-model parameter for the
                     attack construction + execution; apply()'s own finally-block
                     restoration (per-parameter original flags) undoes it
      do_print    -> emit apply()'s per-call status line (timed separately as print_ms)
    Raises instead of falling back to dummy_attack (fail closed)."""
    cw_c, cw_steps, cw_lr = 1.0, 20, 0.01  # apply() defaults; unused by fgsm/pgd/bim
    eps, temperature, diagnostics = s["eps"], s["temperature"], s["diagnostics"]
    attack_params = dict(s["attack_params"])
    wm = adapter.wrapped_model
    ph = {k: 0.0 for k in PHASES}
    t_start = time.perf_counter()
    # ---- A: validation, state capture, numpy->torch, min/max normalization, set_minmax
    am.require_positive_finite_float("attack_temperature", temperature)
    am.require_nonneg_finite_float("attack_eps", eps)
    am.require_positive_finite_float("cw_c", cw_c)
    am.require_positive_int("cw_steps", cw_steps)
    am.require_positive_finite_float("cw_lr", cw_lr)
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"AttackAdapter expects input [N, 2, T], got {x.shape}")
    attack_name = am._validate_attack_name(attack)
    input_shape, input_dtype = x.shape, x.dtype
    _ = float(np.min(x)), float(np.max(x))
    training_before = wm.training
    orig_requires_grad = [p.requires_grad for p in wm.parameters()]
    orig_param_devices = [p.device for p in wm.parameters()]
    try:
        x_t = torch.from_numpy(x).to(adapter.device)
        x_ta, a, b = am._iq_to_ta_input_minmax(x_t)
        _ = float(x_ta.min().item()), float(x_ta.max().item())
        wm.set_minmax(a, b)
        ph["phase_A_prep_ms"] = p1a.elapsed_ms(t_start)
        # ---- B: attack label
        t = time.perf_counter()
        if reuse_pred is None:
            with torch.no_grad():
                y_pred = wm(x_ta).argmax(dim=1)
            y_src = "in_path_clean_forward"
        else:
            y_pred = torch.tensor([reuse_pred], dtype=torch.long, device=x_ta.device)
            y_src = "reused_pipeline_clean_pred"
        ph["phase_B_clean_pred_ms"] = p1a.elapsed_ms(t)
        # ---- freeze (variant only)
        if freeze:
            t = time.perf_counter()
            for p in wm.parameters():
                p.requires_grad_(False)
            ph["phase_freeze_ms"] = p1a.elapsed_ms(t)
        # ---- C: TemperatureLogitsWrapper + attack-object construction (unchanged; no reuse)
        t = time.perf_counter()
        attack_model = am.TemperatureLogitsWrapper(wm, temperature)
        atk = am._build_torchattacks(attack_name, attack_model, eps, cw_c=cw_c, cw_steps=cw_steps, cw_lr=cw_lr,
                                     attack_params=attack_params)
        ph["phase_C_construct_ms"] = p1a.elapsed_ms(t)
        # ---- D: the attack itself
        t = time.perf_counter()
        x_ta_adv = atk(x_ta, y_pred)
        ph["phase_D_attack_ms"] = p1a.elapsed_ms(t)
        # ---- E (part 1): normalized-domain L_inf + output conversion
        t = time.perf_counter()
        iq_linf_normalized = (x_ta_adv - x_ta).abs().amax(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.float32)
        x_adv = am._ta_output_to_iq_minmax(x_ta_adv, a, b).detach().cpu().numpy().astype(np.float32)
        e1 = p1a.elapsed_ms(t)
        if diagnostics:
            t = time.perf_counter()
            wm.eval()
            x_ta_grad = x_ta.clone().detach().requires_grad_(True)
            diag_out = attack_model(x_ta_grad)
            diag_loss = torch.nn.CrossEntropyLoss()(diag_out, y_pred)
            diag_grad = torch.autograd.grad(diag_loss, x_ta_grad, retain_graph=False, create_graph=False)[0]
            _ = (diag_grad != 0).sum(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.int64)
            _ = np.full(x.shape[0], diag_grad[0].numel(), dtype=np.int64)
            _ = diag_grad.abs().amax(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.float32)
            del diag_grad, diag_out, diag_loss, x_ta_grad
            ph["phase_diag_ms"] = p1a.elapsed_ms(t)
    finally:
        # ---- E (part 2): identical to apply()'s finally block (also undoes the freeze)
        t = time.perf_counter()
        if training_before:
            print("[attack_adapter] warning: wrapped model was already in train mode before this call")
        wm.eval()
        wm.clear_minmax()
        for p, req_grad, dev in zip(wm.parameters(), orig_requires_grad, orig_param_devices):
            p.requires_grad_(req_grad)
            if p.device != dev:
                p.data = p.data.to(dev)
        e2 = p1a.elapsed_ms(t)
    # ---- E (part 3): shape/dtype checks, meta NaN/Inf flags; print timed separately
    t = time.perf_counter()
    _ = wm.training
    if x_adv.shape != input_shape:
        raise RuntimeError(f"output shape {x_adv.shape} != input shape {input_shape}")
    if x_adv.dtype != input_dtype:
        raise RuntimeError(f"output dtype {x_adv.dtype} != input dtype {input_dtype}")
    e3 = p1a.elapsed_ms(t)
    if do_print:
        t = time.perf_counter()
        print(f"[attack_adapter] backend={adapter.backend_name} status=ok input={input_shape} output={x_adv.shape}")
        ph["print_ms"] = p1a.elapsed_ms(t)
    t = time.perf_counter()
    _ = bool(np.isnan(x_adv).any()), bool(np.isinf(x_adv).any())
    ph["phase_E_post_ms"] = e1 + e2 + e3 + p1a.elapsed_ms(t)
    ph["total_ms"] = p1a.elapsed_ms(t_start)
    return x_adv, ph, float(iq_linf_normalized[0]), int(y_pred[0].item()), y_src, orig_requires_grad


def in_path_clean_pred(am, adapter, torch, x) -> int:
    """The label apply() would compute internally (minmax -> set_minmax -> wrapped
    forward -> argmax), with the same state restoration. Precheck only, untimed."""
    wm = adapter.wrapped_model
    try:
        x_ta, a, b = am._iq_to_ta_input_minmax(torch.from_numpy(x).to(adapter.device))
        wm.set_minmax(a, b)
        with torch.no_grad():
            return int(wm(x_ta).argmax(dim=1)[0].item())
    finally:
        wm.eval()
        wm.clear_minmax()


# --------------------------------------------------------------------------- worker
def worker_main(args) -> None:
    import torch  # thread settings applied before any parallel work

    torch.set_num_interop_threads(INTEROP_THREADS)
    torch.set_num_threads(INTRA_THREADS)
    intra, interop = torch.get_num_threads(), torch.get_num_interop_threads()
    if (intra, interop) != (INTRA_THREADS, INTEROP_THREADS):
        p1a.abort(f"thread settings not applied: intra={intra} interop={interop}")

    import src.adapters.attack_adapter as am
    from src.adapters.attack_adapter import AttackAdapter, _REAL_ATTACK_SOURCE
    from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE

    ctx = json.loads(Path(args.worker_context).read_text())
    out_dir = Path(args.out_dir)
    variants: List[str] = ctx["variants"]
    p1a.log(f"[worker] intra={intra} interop={interop} pid={os.getpid()} variants={variants}")

    awn = AWNModelAdapter(checkpoint_path=str(REPO_ROOT / ctx["checkpoint_path"]), device="cpu")
    if awn.model is None or awn.backend_name != _REAL_MODEL_SOURCE or awn.status != "ok":
        p1a.abort(f"AWN backend is not real: backend={awn.backend_name} status={awn.status} notes={awn.notes}")
    adapter = AttackAdapter(awn_model=awn.model, device="cpu")  # one instance, as in the formal CPU runner
    if adapter.wrapped_model is None or adapter.backend_name != _REAL_ATTACK_SOURCE or adapter.status != "ok":
        p1a.abort(f"attack backend is not real: backend={adapter.backend_name} status={adapter.status}")
    if awn.model.training:
        p1a.abort("AWN model is in train mode after load")

    subset = [tuple(k) for k in ctx["subset"]]
    formal_dir = REPO_ROOT / ctx["formal_dir"]
    base_rows, formal_attack = p1a.load_formal_rows(formal_dir, set(subset))
    inputs = p1a.rebuild_clean_inputs(subset, base_rows, ctx["sensing"], REPO_ROOT / ctx["dataset_path"])
    p1a.log(f"[worker] rebuilt {len(inputs)} x_clean inputs; all sha256 == formal clean_input_sha256")

    def infer_pred(x) -> int:
        logits, meta = awn.infer(x)
        if meta["awn_backend"] != _REAL_MODEL_SOURCE or meta["awn_status"] != "ok":
            p1a.abort(f"AWN inference fell back: {meta}")
        if not np.isfinite(logits).all():
            p1a.abort("non-finite logits")
        return int(np.argmax(logits[0]))

    # Pipeline clean prediction (the value reuse_clean_pred consumes): real AWN, same x_clean.
    # Fail closed unless it equals BOTH the formal clean_pred and the label apply() computes internally.
    for key, inp in inputs.items():
        inp["clean_pred"] = infer_pred(inp["x"])
        if inp["clean_pred"] != inp["formal_clean_pred"]:
            p1a.abort(f"{key}: pipeline clean_pred {inp['clean_pred']} != formal clean_pred {inp['formal_clean_pred']}")
        inner = in_path_clean_pred(am, adapter, torch, inp["x"])
        if inner != inp["clean_pred"]:
            p1a.abort(f"{key}: apply()-internal y_pred {inner} != pipeline clean_pred {inp['clean_pred']}; "
                      "reuse_clean_pred would change the attack label for this sample")
    p1a.log("[worker] pipeline clean_pred == formal clean_pred == apply()-internal y_pred for all inputs")

    params = list(adapter.wrapped_model.parameters())
    req_grad_ref = [p.requires_grad for p in params]
    n_req_true = sum(req_grad_ref)
    if n_req_true == 0:
        p1a.log("[warn] no parameter requires grad before the attack; freeze_model_params is a no-op here")

    def check_state(where: str) -> bool:
        if awn.model.training or adapter.wrapped_model.training:
            p1a.abort(f"model entered train mode ({where})")
        if (torch.get_num_threads(), torch.get_num_interop_threads()) != (intra, interop):
            p1a.abort(f"torch thread settings drifted ({where})")
        return [p.requires_grad for p in adapter.wrapped_model.parameters()] == req_grad_ref

    settings = ctx["attack_settings"]

    def run_variant(variant: str, key, attack):
        s, inp = settings[attack], inputs[key]
        torch.manual_seed(inp["seed"])  # reset immediately before EVERY variant call (untimed)
        if variant == V_APPLY:
            t0 = time.perf_counter()
            x_adv, meta = adapter.apply(inp["x"], attack=attack, eps=s["eps"], temperature=s["temperature"],
                                        seed=inp["seed"], diagnostics=s["diagnostics"],
                                        attack_params=dict(s["attack_params"]))
            ms = p1a.elapsed_ms(t0)
            if meta["attack_backend"] != _REAL_ATTACK_SOURCE or meta["attack_status"] != "ok":
                p1a.abort(f"attack fell back at runtime: {meta['attack_backend']} {meta['attack_notes']}")
            lin = meta["attack_iq_linf_normalized"]
            ph = {"total_ms": ms, **{k: None for k in PHASES}}
            return x_adv, ph, float(lin[0]), None, "apply_internal", None
        reuse, freeze, do_print = VARIANT_FLAGS[variant]
        return attack_path(am, adapter, torch, inp["x"], attack, s,
                           inp["clean_pred"] if reuse else None, freeze, do_print)

    # ---- warmup (not recorded)
    for attack in ctx["attacks"]:
        for key in subset[: args.warmup]:
            for v in variants:
                run_variant(v, key, attack)
    if not check_state("after warmup"):
        p1a.abort("requires_grad flags not restored after warmup")
    p1a.log(f"[worker] warmup done ({args.warmup} samples x {len(ctx['attacks'])} attacks x {len(variants)} variants)")

    failures: List[dict] = []
    raw_path = out_dir / "raw_measurements.csv"
    n_rows = 0
    group_idx = 0
    with raw_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        for rep in range(args.repeats):
            for key in subset:
                temp = p1a.cpu_temp_c()
                inp = inputs[key]
                for attack in ctx["attacks"]:
                    rot = group_idx % len(variants)  # rotate call order to spread position effects
                    order = variants[rot:] + variants[:rot]
                    group_idx += 1
                    res = {}
                    for v in order:
                        x_adv, ph, lin, y_used, y_src, snapshot = run_variant(v, key, attack)
                        now = [p.requires_grad for p in adapter.wrapped_model.parameters()]
                        restored = check_state(f"{attack} {key} {v}") and (snapshot is None or now == snapshot)
                        if not restored:
                            # never recoverable: later calls would capture a corrupted "original" state
                            _write_failure(out_dir, failures + [{"attack": attack, "variant": v, "sample": list(key),
                                                                 "repeat": rep,
                                                                 "problems": ["requires_grad not restored exactly"]}])
                            p1a.abort(f"requires_grad flags not restored exactly after {v} {attack} {key}")
                        res[v] = {"x_adv": x_adv, "ph": ph, "lin": lin, "y_used": y_used, "y_src": y_src,
                                  "restored": restored}
                    base = res[V_BASE]
                    base["pred"] = infer_pred(base["x_adv"])
                    bdiff = base["x_adv"] - inp["x"]
                    base["linf"] = float(np.max(np.abs(bdiff)))
                    base["l2"] = float(np.linalg.norm(bdiff.reshape(-1)))
                    base["sha"] = p1a.sha256_array(base["x_adv"])
                    clean_ok = inp["clean_pred"] == inp["label"]
                    for pos, v in enumerate(order):
                        r = res[v]
                        xa = r["x_adv"]
                        finite = bool(np.isfinite(xa).all())
                        shape_ok = xa.shape == inp["x"].shape and xa.dtype == np.float32
                        pred = base["pred"] if v == V_BASE else infer_pred(xa)
                        diff = xa - inp["x"]
                        linf = float(np.max(np.abs(diff)))
                        l2 = float(np.linalg.norm(diff.reshape(-1)))
                        sha = p1a.sha256_array(xa)
                        bit_eq = bool(xa.shape == base["x_adv"].shape and np.array_equal(xa, base["x_adv"]))
                        maxdiff = float(np.max(np.abs(xa.astype(np.float64) - base["x_adv"].astype(np.float64)))) \
                            if xa.shape == base["x_adv"].shape else float("inf")
                        att_ok = pred == inp["label"]
                        success = bool(clean_ok and not att_ok)
                        base_success = bool(clean_ok and base["pred"] != inp["label"])
                        row = {
                            "condition": CONDITION, "intra_threads": intra, "interop_threads": interop,
                            "attack": attack, "attack_profile": settings[attack]["profile"], "eps": settings[attack]["eps"],
                            "modulation": key[0], "snr_db": key[1], "sample_index": key[2], "formal_seed": inp["seed"],
                            "repeat": rep, "variant": v, "variant_position": pos, "torch_call_seed": inp["seed"],
                            "y_pred_used": r["y_used"], "y_pred_source": r["y_src"], "clean_pred_pipeline": inp["clean_pred"],
                            "attacked_pred": pred, "label": inp["label"], "clean_correct": clean_ok,
                            "attacked_correct": att_ok, "attack_success": success,
                            "iq_linf": linf, "iq_l2": l2, "iq_linf_normalized": r["lin"], "x_adv_sha256": sha,
                            "x_adv_shape": "x".join(map(str, xa.shape)), "x_adv_dtype": str(xa.dtype), "x_adv_finite": finite,
                            "bit_identical_to_baseline": bit_eq, "max_abs_diff_vs_baseline": maxdiff,
                            "attacked_pred_eq_baseline": pred == base["pred"], "attack_success_eq_baseline": success == base_success,
                            "iq_linf_eq_baseline": linf == base["linf"], "iq_l2_eq_baseline": l2 == base["l2"],
                            "sha256_eq_baseline": sha == base["sha"],
                            "requires_grad_restored": r["restored"], "n_params_requires_grad_true_before": n_req_true,
                            "cpu_temp_c": temp,
                        }
                        row.update(r["ph"])
                        w.writerow({k: row.get(k) for k in RAW_FIELDS})
                        n_rows += 1
                        problems = []
                        if not finite:
                            problems.append("non-finite x_adv")
                        if not shape_ok:
                            problems.append(f"shape/dtype {xa.shape} {xa.dtype}")
                        if not r["restored"]:
                            problems.append("requires_grad not restored exactly")
                        if not bit_eq:
                            problems.append(f"x_adv differs from baseline (max abs diff {maxdiff:.3e})")
                        if r["y_used"] is not None and r["y_used"] != inp["clean_pred"]:
                            problems.append(f"attack label {r['y_used']} != pipeline clean_pred {inp['clean_pred']}")
                        if problems:
                            failures.append({"attack": attack, "variant": v, "sample": list(key), "repeat": rep,
                                             "problems": problems})
                    f.flush()
                    if failures and args.on_mismatch == "abort":
                        _write_failure(out_dir, failures)
                        p1a.abort(f"semantic difference detected: {failures[0]} (see worker_failures.json)")
            p1a.log(f"[worker] repeat {rep + 1}/{args.repeats} done, temp={p1a.cpu_temp_c()}")
    if failures:
        _write_failure(out_dir, failures)

    worker_meta = {
        "pid": os.getpid(), "intra_threads": intra, "interop_threads": interop,
        "env_thread_vars": {k: os.environ.get(k) for k in p1a.ENV_THREAD_VARS},
        "torch_version": torch.__version__, "torch_parallel_info": torch.__config__.parallel_info(),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "awn_backend": awn.backend_name, "attack_backend": adapter.backend_name,
        "attack_adapter_source_sha256": p1a.sha256_file(Path(am.__file__)),
        "torchattacks_version": getattr(am._torchattacks, "__version__", None),
        "n_wrapped_params": len(params), "n_params_requires_grad_true_before": n_req_true,
        "n_inputs": len(inputs), "raw_rows": n_rows, "n_failures": len(failures),
        "formal_attack_rows_loaded": len(formal_attack),
    }
    (out_dir / "worker.json").write_text(json.dumps(worker_meta, indent=2, default=str))
    p1a.log(f"[worker] wrote {n_rows} rows; failures={len(failures)}")


def _write_failure(out_dir: Path, failures: List[dict]) -> None:
    (out_dir / "worker_failures.json").write_text(json.dumps(failures[:2000], indent=2, default=str))


# --------------------------------------------------------------------------- parent: aggregation / validation
def _b(v) -> bool:
    return p1a.parse_bool(v)


def aggregate_and_validate(out_dir: Path, rows: List[dict], attack_settings: dict, variants: List[str],
                           subset: List[tuple], formal_dir: Path) -> tuple:
    _, formal_attack = p1a.load_formal_rows(formal_dir, set(subset))
    validation = {"per_attack_variant": {}, "baseline_vs_apply_untouched": {}, "baseline_vs_formal": {}, "notes": []}
    status: Dict[tuple, str] = {}

    for attack in attack_settings:
        pid = attack_settings[attack]["profile"]
        ar = [r for r in rows if r["attack"] == attack]
        # anchor 1: baseline replica == unchanged apply()
        if V_APPLY in variants:
            ap = [r for r in ar if r["variant"] == V_APPLY]
            validation["baseline_vs_apply_untouched"][attack] = {
                "bit_identical": f"{sum(_b(r['bit_identical_to_baseline']) for r in ap)}/{len(ap)}"}
        # anchor 2: baseline == formal CPU rows (FGSM/BIM required; PGD informational)
        br = [r for r in ar if r["variant"] == V_BASE]
        m_sha = m_pred = m_succ = 0
        for r in br:
            fr = formal_attack[(r["modulation"], int(r["snr_db"]), int(r["sample_index"]), attack, pid)]
            m_sha += int(r["x_adv_sha256"] == fr.get("adversarial_sha256"))
            m_pred += int(int(r["attacked_pred"]) == int(float(fr["attacked_pred"])))
            m_succ += int(_b(r["attack_success"]) == p1a.parse_bool(fr["attack_success"]))
        formal_ok = (m_sha == m_pred == m_succ == len(br)) if attack in p1a.DETERMINISTIC_ATTACKS else True
        validation["baseline_vs_formal"][attack] = {
            "x_adv_sha256_match": f"{m_sha}/{len(br)}", "attacked_pred_match": f"{m_pred}/{len(br)}",
            "attack_success_match": f"{m_succ}/{len(br)}",
            "required": attack in p1a.DETERMINISTIC_ATTACKS, "pass": formal_ok,
            "note": None if attack in p1a.DETERMINISTIC_ATTACKS else
            "pgd random_start=True: formal RNG call order not reproduced; informational only",
        }
        apply_ok = all(_b(r["bit_identical_to_baseline"]) for r in ar if r["variant"] == V_APPLY)
        for v in variants:
            vr = [r for r in ar if r["variant"] == v]
            n = len(vr)

            def cnt(col):
                return sum(_b(r[col]) for r in vr)

            checks = {
                "bit_identical_to_baseline": cnt("bit_identical_to_baseline"),
                "sha256_eq_baseline": cnt("sha256_eq_baseline"),
                "attacked_pred_eq_baseline": cnt("attacked_pred_eq_baseline"),
                "attack_success_eq_baseline": cnt("attack_success_eq_baseline"),
                "iq_linf_eq_baseline": cnt("iq_linf_eq_baseline"),
                "iq_l2_eq_baseline": cnt("iq_l2_eq_baseline"),
                "x_adv_finite": cnt("x_adv_finite"),
                "requires_grad_restored": cnt("requires_grad_restored"),
                "shape_1x2x128": sum(r["x_adv_shape"] == "1x2x128" for r in vr),
                "dtype_float32": sum(r["x_adv_dtype"] == "float32" for r in vr),
            }
            label_ok = all(r["y_pred_used"] in ("", None) or int(r["y_pred_used"]) == int(r["clean_pred_pipeline"])
                           for r in vr)
            ok = n > 0 and all(c == n for c in checks.values()) and label_ok and apply_ok and formal_ok
            st = "PASS" if ok else "FAIL"
            status[(attack, v)] = st
            validation["per_attack_variant"][f"{attack}/{v}"] = {
                "n": n, "status": st, "kind": VARIANT_KIND[v],
                **{k: f"{c}/{n}" for k, c in checks.items()},
                "attack_label_equals_pipeline_clean_pred": label_ok,
                "max_abs_diff_vs_baseline": max((float(r["max_abs_diff_vs_baseline"]) for r in vr), default=None),
                "depends_on_anchor_baseline_eq_apply": apply_ok, "depends_on_anchor_baseline_eq_formal": formal_ok,
            }

    # ---- summary
    summary = []
    for attack in attack_settings:
        ar = [r for r in rows if r["attack"] == attack]
        base_total = {(r["modulation"], r["snr_db"], r["sample_index"], r["repeat"]): float(r["total_ms"])
                      for r in ar if r["variant"] == V_BASE}
        base_stats = {m: p1a.stats([float(r[m]) for r in ar if r["variant"] == V_BASE]) for m in ["total_ms", *PHASES]}
        for v in variants:
            vr = [r for r in ar if r["variant"] == v]
            accepted = status[(attack, v)] == "PASS"
            for m in ["total_ms", *PHASES]:
                vals = [float(r[m]) for r in vr if r[m] not in ("", None)]
                if not vals:
                    continue
                st = p1a.stats(vals)
                bs = base_stats[m]
                rec = {"attack": attack, "attack_profile": attack_settings[attack]["profile"], "variant": v,
                       "variant_kind": VARIANT_KIND[v], "measure": m, "batch_size": 1,
                       "intra_threads": INTRA_THREADS, "interop_threads": INTEROP_THREADS,
                       "n": st["n"], "mean_ms": st["mean"], "median_ms": st["median"], "p95_ms": st["p95"],
                       "p99_ms": st["p99"], "min_ms": st["min"], "max_ms": st["max"], "std_ms": st["std"],
                       "validation_status": status[(attack, v)]}
                # speedups / savings are withheld unless the variant passed validation
                show = accepted and bs["n"] == st["n"]
                rec["baseline_median_ms"] = bs["median"]
                both_pos = bs["median"] > 0 and st["median"] > 0 and bs["mean"] > 0 and st["mean"] > 0
                rec["speedup_median_vs_baseline"] = bs["median"] / st["median"] if show and both_pos else ""
                rec["speedup_mean_vs_baseline"] = bs["mean"] / st["mean"] if show and both_pos else ""
                rec["ms_saved_median_vs_baseline"] = bs["median"] - st["median"] if show else ""
                rec["ms_saved_mean_vs_baseline"] = bs["mean"] - st["mean"] if show else ""
                if m == "total_ms" and show:
                    paired = [base_total[(r["modulation"], r["snr_db"], r["sample_index"], r["repeat"])] - float(r["total_ms"])
                              for r in vr]
                    rec["paired_ms_saved_median"] = float(np.median(paired))
                    rec["paired_ms_saved_p95"] = float(np.percentile(paired, 95))
                    rec["paired_ms_saved_p5"] = float(np.percentile(paired, 5))
                    rec["attack_path_total_median_ms"] = st["median"]
                    rec["attack_path_total_median_le_40ms"] = st["median"] <= 40.0
                    rec["attack_path_total_median_le_20ms"] = st["median"] <= 20.0
                else:
                    for k in ("paired_ms_saved_median", "paired_ms_saved_p95", "paired_ms_saved_p5",
                              "attack_path_total_median_ms", "attack_path_total_median_le_40ms",
                              "attack_path_total_median_le_20ms"):
                        rec[k] = ""
                summary.append(rec)
    fields = list(summary[0].keys())
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)

    validation["overall"] = "PASS" if all(s == "PASS" for s in status.values()) else "FAIL"
    validation["system_interpretation"] = SYSTEM_INTERPRETATION
    validation["notes"] += [
        "Every variant call is preceded by torch.manual_seed(formal per-row seed); PGD random_start draws are "
        "therefore identical across variants and bit-identity to baseline is REQUIRED for all three attacks.",
        "Speedup / ms-saved fields in summary.csv are blank for any attack x variant whose status is FAIL.",
        "The 20-40 ms target is judged only on measure=total_ms (the full B=1 attack path), never on a phase.",
        "These variants remove Python/autograd overhead only; they are not NEON/Arm-specific acceleration.",
    ]
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2, default=str))
    return summary, validation


# --------------------------------------------------------------------------- parent
def parent_main(args) -> None:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (REPO_ROOT / (args.out_dir or f"results/accel_phase1b_{ts}")).resolve()
    for p in p1a.PROTECTED_RESULT_DIRS:
        prot = (REPO_ROOT / p).resolve()
        if out_dir == prot or prot in out_dir.parents or out_dir in prot.parents:
            p1a.abort(f"refusing to write inside/over formal results directory {p}")
    out_dir.mkdir(parents=True, exist_ok=False)

    ctx = p1a.load_formal_context(args)
    cfg, manifest, formal_dir = ctx["cfg"], ctx["manifest"], ctx["formal_dir"]
    subset = p1a.build_subset(cfg)
    base_rows, attack_rows = p1a.load_formal_rows(formal_dir, set(subset))
    attack_settings = p1a.resolve_attack_settings(cfg, attack_rows, args.attack_temperature, args.diagnostics)
    sensing = p1a.resolve_sensing(cfg)

    p1a.log("[parent] hashing dataset / checkpoint / config ...")
    hashes = {
        "dataset_sha256": p1a.sha256_file(REPO_ROOT / cfg["dataset_path"]),
        "checkpoint_sha256": p1a.sha256_file(REPO_ROOT / cfg["checkpoint_path"]),
        "config_sha256": p1a.sha256_file(ctx["cfg_path"]),
        "formal_manifest_sha256": p1a.sha256_file(formal_dir / "manifest.json"),
        "formal_base_results_sha256": p1a.sha256_file(formal_dir / "base_results.csv"),
        "formal_attack_results_sha256": p1a.sha256_file(formal_dir / "attack_results.csv"),
        "attack_adapter_py_sha256": p1a.sha256_file(REPO_ROOT / "src/adapters/attack_adapter.py"),
        "phase1a_helpers_sha256": p1a.sha256_file(Path(p1a.__file__).resolve()),
        "this_script_sha256": p1a.sha256_file(Path(__file__).resolve()),
    }
    if hashes["dataset_sha256"] != manifest.get("dataset_sha256"):
        p1a.abort("dataset sha256 differs from formal CPU manifest")
    if hashes["checkpoint_sha256"] != manifest.get("checkpoint_sha256"):
        p1a.abort("checkpoint sha256 differs from formal CPU manifest")

    env_meta = p1a.environment_metadata()
    if any(env_meta["env_thread_vars"].values()):
        p1a.log(f"[warn] thread env vars set in this shell: {env_meta['env_thread_vars']} (inherited unchanged, recorded)")

    variants = [V_APPLY, V_BASE, V_REUSE, V_FREEZE, V_BOTH] + ([] if args.no_print_ablation else [V_NOPRINT])
    worker_ctx = {
        "subset": [list(k) for k in subset], "formal_dir": p1a.rel(formal_dir),
        "dataset_path": cfg["dataset_path"], "checkpoint_path": cfg["checkpoint_path"],
        "sensing": sensing, "attack_settings": attack_settings, "attacks": list(p1a.TARGET_PROFILES),
        "variants": variants,
    }
    ctx_file = out_dir / "worker_context.json"
    ctx_file.write_text(json.dumps(worker_ctx, indent=2, default=str))

    manifest_out = {
        "phase": "accel_phase1b_overhead_removal", "status": "running",
        "question": "B=1 attack-path latency removed by eliminating avoidable runtime overhead "
                    "(no change to the attack's mathematical definition)",
        "command": [sys.executable] + sys.argv, "out_dir": p1a.rel(out_dir),
        "formal_cpu_dir": p1a.rel(formal_dir), "config_path": args.config,
        "dataset_path": cfg["dataset_path"], "checkpoint_path": cfg["checkpoint_path"],
        "hashes": hashes, "environment": env_meta,
        "threads": {"intra_op": INTRA_THREADS, "inter_op": INTEROP_THREADS,
                    "note": "best B=1 setting from Phase 1A; set in a fresh worker process before any parallel work"},
        "batch_size": 1, "repeats": args.repeats, "warmup_samples_per_attack": args.warmup,
        "subset": {"modulations": cfg["modulations"], "snrs": p1a.SUBSET_SNRS,
                   "sample_indices": p1a.SUBSET_SAMPLE_INDICES, "n_source_samples": len(subset),
                   "samples": [{"modulation": k[0], "snr_db": k[1], "sample_index": k[2],
                                "formal_seed": int(float(base_rows[k]["seed"])),
                                "clean_input_sha256": base_rows[k]["clean_input_sha256"]} for k in subset]},
        "attacks": attack_settings, "sensing_parameters": sensing,
        "variants": {v: {"kind": VARIANT_KIND[v],
                         "reuse_clean_pred": VARIANT_FLAGS.get(v, (None,))[0],
                         "freeze_model_params": VARIANT_FLAGS.get(v, (None, None))[1],
                         "print_status_line": VARIANT_FLAGS.get(v, (None, None, True))[2]} for v in variants},
        "not_tested": ["attack-object reuse", "batching", "any change to steps/eps/alpha/random_start",
                       "normalization changes", "source edits to AttackAdapter"],
        "timing": {
            "clock": "time.perf_counter",
            "total_ms": "whole B=1 attack path for the variant (apply_untouched: same boundary as formal attack_ms)",
            "phase_A_prep_ms": "validation + state capture + numpy->torch + min/max normalization + set_minmax",
            "phase_B_clean_pred_ms": "in-path duplicate clean forward (baseline/freeze/no_print) OR label-tensor "
                                     "construction from the reused pipeline clean_pred (reuse variants)",
            "phase_freeze_ms": "requires_grad_(False) loop (freeze variants only, else 0)",
            "phase_C_construct_ms": "TemperatureLogitsWrapper + _build_torchattacks (unchanged, not reused)",
            "phase_D_attack_ms": "atk(x_ta, y_pred)",
            "phase_diag_ms": "apply(diagnostics=True) extra grad pass, only if the formal run used it",
            "phase_E_post_ms": "normalized L_inf + output conversion + finite-block state restoration "
                               "(incl. undoing the freeze) + shape/dtype checks + NaN/Inf flags; excludes print",
            "print_ms": "per-call status print (0 for baseline_no_print); stdout goes to worker.stdout.log",
            "excluded": "torch.manual_seed reset, attacked AWN inference, hashing, comparisons, CSV writes",
        },
        "rng": "torch.manual_seed(formal per-row seed) immediately before every variant call",
        "system_interpretation": SYSTEM_INTERPRETATION,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2, default=str))

    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--out-dir", str(out_dir),
           "--worker-context", str(ctx_file), "--repeats", str(args.repeats), "--warmup", str(args.warmup),
           "--on-mismatch", args.on_mismatch]
    manifest_out["temp_before_c"] = p1a.cpu_temp_c()
    p1a.log(f"[parent] starting fresh worker process (intra=1, interop=1), temp={manifest_out['temp_before_c']}")
    with (out_dir / "worker.stdout.log").open("w") as lf:
        rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=lf, env=os.environ.copy()).returncode
    manifest_out["temp_after_c"] = p1a.cpu_temp_c()
    manifest_out["worker_returncode"] = rc

    raw_file = out_dir / "raw_measurements.csv"
    rows = list(csv.DictReader(raw_file.open(newline=""))) if raw_file.exists() else []
    if rc != 0:
        fail_file = out_dir / "worker_failures.json"
        (out_dir / "validation.json").write_text(json.dumps({
            "overall": "FAIL (worker aborted)", "worker_returncode": rc, "raw_rows_written": len(rows),
            "failures": json.loads(fail_file.read_text()) if fail_file.exists() else None,
            "note": "no speedup is reported for an aborted run; see worker.stdout.log and the console stderr",
        }, indent=2, default=str))
        manifest_out["status"] = "aborted"
        (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2, default=str))
        p1a.abort(f"worker exited with {rc}")

    summary, validation = aggregate_and_validate(out_dir, rows, attack_settings, variants, subset, formal_dir)
    validation["worker"] = json.loads((out_dir / "worker.json").read_text())
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2, default=str))
    manifest_out.update(status="complete", raw_rows=len(rows), validation_overall=validation["overall"],
                        finished_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
                        environment_end={"cpu_temp_c": p1a.cpu_temp_c(),
                                         "vcgencmd_throttled": p1a.run_cmd(["vcgencmd", "get_throttled"])})
    (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2, default=str))
    p1a.log(f"[parent] done: {out_dir}  validation={validation['overall']}")
    for rec in summary:
        if rec["measure"] == "total_ms":
            p1a.log(f"  {rec['attack']:5s} {rec['variant']:40s} median={rec['median_ms']:8.3f} ms  "
                    f"speedup={rec['speedup_median_vs_baseline'] or '-'}  status={rec['validation_status']}")
    if validation["overall"] != "PASS":
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=p1a.DEFAULT_CONFIG)
    ap.add_argument("--formal-dir", default=p1a.DEFAULT_FORMAL_CPU_DIR)
    ap.add_argument("--out-dir", default=None, help="default results/accel_phase1b_<timestamp> (must not exist)")
    ap.add_argument("--repeats", type=int, default=3, help="passes over the 88-sample subset")
    ap.add_argument("--warmup", type=int, default=5, help="unrecorded warmup samples per attack")
    ap.add_argument("--attack-temperature", type=float, default=None,
                    help="only if the formal config/rows do not record it; must equal the formal value")
    ap.add_argument("--diagnostics", choices=["auto", "on", "off"], default="auto",
                    help="apply(diagnostics=...); auto = match formal rows (same rule as Phase 1A)")
    ap.add_argument("--no-print-ablation", action="store_true", help="skip the baseline_no_print logging ablation")
    ap.add_argument("--on-mismatch", choices=["abort", "record"], default="abort",
                    help="abort (default, fail closed) or record: keep measuring, mark variant FAIL, withhold speedups")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-context", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        p1a.abort("--repeats>=1 and --warmup>=0 required")
    if args.worker:
        worker_main(args)
    else:
        parent_main(args)


if __name__ == "__main__":
    main()
