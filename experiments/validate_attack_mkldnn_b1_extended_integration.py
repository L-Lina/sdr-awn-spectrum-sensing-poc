"""
Integration validation of AttackAdapter(b1_mkldnn_off_extended=True): mkldnn disabled ONLY around atk(x, y)
at B=1 for the attacks in src/adapters/attack_adapter.py:_B1_MKLDNN_OFF_EXTENDED_ATTACKS.
Raspberry Pi 5, CPU, intra 4 / interop 1 (the benchmark condition). Regression validation, not a
performance study (latency evidence: results/attack_b1_accel_full_20261004_100032).

Every call goes through the real AttackAdapter.apply() with the formal profile arguments of
configs/hailo_full_matrix.json (same derivation as experiments/run_cpu_full_matrix.py), preceded by
torch.manual_seed(formal per-row seed), on the first --samples of the validated 88-sample subset.

A  Option OFF parity: the default adapter must be bit-identical to the pre-change AttackAdapter loaded
   from --base-ref (git show <ref>:src/adapters/attack_adapter.py) for every formal profile, with the
   same meta key set and the same torch RNG state after the call.
   autoattack: torchattacks.AutoAttack without an explicit seed derives its sub-attack seeds from the
   wall clock, so every autoattack call here passes a deterministic seed through the production path
   attack_params["seed"] (the same route as --attack-internal-seed), set to the formal per-row seed.
   This is a validation control only; the formal profile parameters are otherwise unchanged.
B  Option ON semantics (extended attacks): output classified against option OFF --
   BIT_IDENTICAL | BITWISE_DIFFERENT_SEMANTIC_SAME (same attacked prediction and attack success) |
   SEMANTIC_DIFFERENT | INVALID (non-finite, wrong shape/dtype, or a new eps-bound violation for
   attacks with a fixed Linf budget). Meta flag attack_b1_mkldnn_off_extended_applied must be True,
   the legacy flag key absent, the RNG state equal to option OFF's.
   Attacks outside the set (fgsm, pgd, bim, deepfool) with the option ON: bit-identical to OFF,
   flag False. Interplay on the first --interplay-samples: with both options ON, pgd/bim equal the
   legacy-only adapter (legacy flag True, extended False) and extended attacks equal the extended-only
   adapter (extended True, legacy False).
C  Scope and state (runtime probes): mkldnn state recorded when atk() is entered and on every AWN
   forward. Per extended attack at B=1: atk with False, clean-prediction forward with True, global
   state True afterwards. B=2 and option OFF: atk with True. Previous state False is preserved.
   An exception inside atk() leaves the state restored. Model eval mode and requires_grad unchanged.

PASS requires zero failures in A, B and C. Output: <out>/validation.json, per_sample.csv.

Run on the Pi (from the repo root):
    .venv/bin/python experiments/validate_attack_mkldnn_b1_extended_integration.py
    quick: --samples 8
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as _dt
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import bench_attack_accel_pi5 as p1a  # noqa: E402
import bench_attack_batch_pi5_v2 as p1c  # noqa: E402 -- V2 tolerances
import benchmark_attack_b1_acceleration_full_pi5 as fb  # noqa: E402 -- formal profile derivation

REPO_ROOT = p1a.REPO_ROOT
INTRA, INTEROP = 4, 1
NOT_EXTENDED = ["fgsm", "pgd", "bim", "deepfool"]
# Attacks whose formal profile leaves the internal seed to the wall clock: validated with attack_params["seed"].
DETERMINISTIC_SEED_ATTACKS = {"autoattack"}
LEGACY_KEY, EXT_KEY = "attack_b1_pgd_bim_mkldnn_off_applied", "attack_b1_mkldnn_off_extended_applied"
FIELDS = ["section", "attack", "profile", "variant", "sample_pos", "modulation", "snr_db", "sample_index",
          "label", "clean_pred", "pred", "reference_pred", "success", "reference_success", "bit_identical",
          "max_abs_diff", "linf_normalized", "reference_linf_normalized", "finite", "shape_dtype_ok",
          "meta_keys_ok", "flag_ok", "rng_equal", "classification", "note"]


def validation_params(prof: dict, seed: int) -> dict:
    params = dict(prof["attack_params"])
    if prof["attack"] in DETERMINISTIC_SEED_ATTACKS and params.get("seed") is None:
        params["seed"] = int(seed)
    return params


def load_base_adapter_module(ref: str):
    src = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{ref}:src/adapters/attack_adapter.py"],
                         capture_output=True, text=True, check=True).stdout
    tmp = Path(tempfile.mkdtemp(prefix="attack_adapter_base_")) / "attack_adapter_base.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("attack_adapter_base", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=p1a.DEFAULT_CONFIG)
    ap.add_argument("--formal-dir", default=p1a.DEFAULT_FORMAL_CPU_DIR)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--base-ref", default="HEAD", help="git ref holding the pre-change attack_adapter.py")
    ap.add_argument("--samples", type=int, default=88, help="first N of the 88-sample subset")
    ap.add_argument("--interplay-samples", type=int, default=4)
    args = ap.parse_args()

    import torch
    torch.set_num_interop_threads(INTEROP)
    torch.set_num_threads(INTRA)
    import src.adapters.attack_adapter as am
    from src.adapters.attack_adapter import AttackAdapter, _REAL_ATTACK_SOURCE, mkldnn_disabled_scope
    from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE
    get_mk, set_mk = torch._C._get_mkldnn_enabled, torch._C._set_mkldnn_enabled
    if bool(get_mk()) is not True:
        p1a.abort("mkldnn is not enabled by default in this process")
    extended = sorted(am._B1_MKLDNN_OFF_EXTENDED_ATTACKS)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (REPO_ROOT / (args.out_dir or f"results/accel_b1_extended_integration_{ts}")).resolve()
    out.mkdir(parents=True, exist_ok=False)

    ctx = p1a.load_formal_context(args)
    cfg = ctx["cfg"]
    subset = p1a.build_subset(cfg)[: args.samples]
    base_rows, _ = p1a.load_formal_rows(ctx["formal_dir"], set(subset))
    inputs = p1a.rebuild_clean_inputs(subset, base_rows, p1a.resolve_sensing(cfg), REPO_ROOT / cfg["dataset_path"])
    profiles = fb.formal_profiles(cfg, include_controls=True)

    awn = AWNModelAdapter(checkpoint_path=str(REPO_ROOT / cfg["checkpoint_path"]), device="cpu")
    if awn.model is None or awn.backend_name != _REAL_MODEL_SOURCE or awn.status != "ok":
        p1a.abort("AWN backend is not real")
    base_mod = load_base_adapter_module(args.base_ref)
    if hasattr(base_mod, "_B1_MKLDNN_OFF_EXTENDED_ATTACKS"):
        p1a.abort(f"--base-ref {args.base_ref} already contains the extended option; pass the pre-change ref")
    ad = {"base": base_mod.AttackAdapter(awn_model=awn.model, device="cpu"),
          "off": AttackAdapter(awn_model=awn.model, device="cpu"),
          "ext": AttackAdapter(awn_model=awn.model, device="cpu", b1_mkldnn_off_extended=True),
          "legacy": AttackAdapter(awn_model=awn.model, device="cpu", b1_pgd_bim_mkldnn_off=True),
          "both": AttackAdapter(awn_model=awn.model, device="cpu", b1_pgd_bim_mkldnn_off=True,
                                b1_mkldnn_off_extended=True)}
    for name, a in ad.items():
        if a.wrapped_model is None or a.status != "ok":
            p1a.abort(f"adapter {name}: attack backend is not real")
    wm = ad["off"].wrapped_model
    req_grad_ref = [p.requires_grad for p in awn.model.parameters()]

    rows: List[dict] = []
    failures: List[str] = []
    counts: Dict[str, Dict[str, int]] = {}

    def fail(msg: str) -> None:
        failures.append(msg)
        p1a.log(f"[FAIL] {msg}")

    def call(name: str, prof: dict, x: np.ndarray, seed: int):
        torch.manual_seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            x_adv, meta = ad[name].apply(x, attack=prof["attack"], eps=prof["eps"], temperature=prof["temperature"],
                                         seed=seed, diagnostics=False, cw_c=prof["cw_c"], cw_steps=prof["cw_steps"],
                                         cw_lr=prof["cw_lr"], attack_params=validation_params(prof, seed))
        rng = torch.get_rng_state().clone()
        if awn.model.training or ad[name].wrapped_model.training:
            fail(f"{name} {prof['slug']}: model left in train mode")
        if [p.requires_grad for p in awn.model.parameters()] != req_grad_ref:
            fail(f"{name} {prof['slug']}: requires_grad flags changed")
        if bool(get_mk()) is not True:
            fail(f"{name} {prof['slug']}: global mkldnn state not restored")
            set_mk(True)
        if meta.get("attack_status") != "ok" or meta.get("attack_backend") != _REAL_ATTACK_SOURCE:
            fail(f"{name} {prof['slug']}: attack fell back ({meta.get('attack_notes')})")
        return x_adv, meta, rng

    def pred_of(x_adv: np.ndarray) -> int:
        with contextlib.redirect_stdout(io.StringIO()):
            logits, meta = awn.infer(x_adv)
        return int(np.argmax(logits[0]))

    def lin(meta) -> float:
        v = meta.get("attack_iq_linf_normalized")
        return float(v[0]) if v is not None else float("nan")

    base_keys = None
    # ---------------------------------------------------------------- A + B
    for prof in profiles:
        attack, slug = prof["attack"], prof["slug"]
        in_ext = attack in extended
        if not in_ext and attack not in NOT_EXTENDED:
            continue
        c = counts.setdefault(slug, {k: 0 for k in p1c.CLASSES})
        for pos, key in enumerate(subset):
            inp = inputs[key]
            x, seed, label = inp["x"], inp["seed"], inp["label"]
            clean_pred = inp["formal_clean_pred"]
            x_base, m_base, r_base = call("base", prof, x, seed)
            x_off, m_off, r_off = call("off", prof, x, seed)
            if base_keys is None:
                base_keys = set(m_base)
            a_bit = bool(np.array_equal(x_base, x_off))
            a_keys = set(m_off) == set(m_base) and EXT_KEY not in m_off and LEGACY_KEY not in m_off
            a_rng = bool(torch.equal(r_base, r_off))
            if not (a_bit and a_keys and a_rng):
                fail(f"A {slug} {key}: option OFF differs from base (bit={a_bit} keys={a_keys} rng={a_rng})")
            p_off = pred_of(x_off)
            s_off = clean_pred == label and p_off != label
            rows.append({"section": "A", "attack": attack, "profile": slug, "variant": "off_vs_base",
                         "sample_pos": pos, "modulation": key[0], "snr_db": key[1], "sample_index": key[2],
                         "label": label, "clean_pred": clean_pred, "pred": p_off, "bit_identical": a_bit,
                         "meta_keys_ok": a_keys, "rng_equal": a_rng,
                         "classification": "BIT_IDENTICAL" if a_bit else "DIFFERENT"})
            x_on, m_on, r_on = call("ext", prof, x, seed)
            fin = bool(np.isfinite(x_on).all())
            shape_ok = x_on.shape == x_off.shape and x_on.dtype == np.float32
            bit = bool(shape_ok and np.array_equal(x_on, x_off))
            p_on = p_off if bit else (pred_of(x_on) if (fin and shape_ok) else -1)
            s_on = clean_pred == label and p_on != label
            eps_gated = attack in fb.EPS_LINF_BOUNDED
            l_on, l_off = lin(m_on), lin(m_off)
            new_violation = eps_gated and l_off <= prof["eps"] + p1c.NORM_EPS_TOL < l_on
            flag_ok = (m_on.get(EXT_KEY) is in_ext) and LEGACY_KEY not in m_on
            rng_ok = bool(torch.equal(r_on, r_off))
            if not (fin and shape_ok) or new_violation:
                cls = "INVALID"
            elif bit:
                cls = "BIT_IDENTICAL"
            elif p_on == p_off and s_on == s_off:
                cls = "BITWISE_DIFFERENT_SEMANTIC_SAME"
            else:
                cls = "SEMANTIC_DIFFERENT"
            c[cls] += 1
            if not flag_ok or not rng_ok:
                fail(f"B {slug} {key}: flag_ok={flag_ok} rng_equal={rng_ok}")
            if in_ext and cls in ("SEMANTIC_DIFFERENT", "INVALID"):
                fail(f"B {slug} {key}: option ON {cls}")
            if not in_ext and not bit:
                fail(f"B {slug} {key}: attack outside the set changed with the option ON")
            rows.append({"section": "B", "attack": attack, "profile": slug,
                         "variant": "ext_on_vs_off" if in_ext else "ext_on_not_in_set", "sample_pos": pos,
                         "modulation": key[0], "snr_db": key[1], "sample_index": key[2], "label": label,
                         "clean_pred": clean_pred, "pred": p_on, "reference_pred": p_off, "success": s_on,
                         "reference_success": s_off, "bit_identical": bit,
                         "max_abs_diff": float(np.max(np.abs(x_on.astype(np.float64) - x_off))) if shape_ok else "",
                         "linf_normalized": l_on, "reference_linf_normalized": l_off, "finite": fin,
                         "shape_dtype_ok": shape_ok, "flag_ok": flag_ok, "rng_equal": rng_ok, "classification": cls})
            if pos < args.interplay_samples and (in_ext or attack in ("pgd", "bim")):
                x_both, m_both, _ = call("both", prof, x, seed)
                ref = "ext" if in_ext else "legacy"
                x_ref, _, _ = call(ref, prof, x, seed)
                ok = (np.array_equal(x_both, x_ref) and m_both.get(LEGACY_KEY) is (not in_ext)
                      and m_both.get(EXT_KEY) is in_ext)
                if not ok:
                    fail(f"B interplay {slug} {key}: both-options adapter != {ref}-only adapter or flags wrong")
                rows.append({"section": "B", "attack": attack, "profile": slug, "variant": f"both_vs_{ref}",
                             "sample_pos": pos, "bit_identical": bool(np.array_equal(x_both, x_ref)), "flag_ok": ok,
                             "classification": "OK" if ok else "FAIL"})
        p1a.log(f"[A/B] {slug:32s} {c}")

    # ---------------------------------------------------------------- C: scope / state probes
    probe = {"in_atk": False, "atk_states": [], "fwd_in": [], "fwd_out": [], "raise": False}
    real_build = am._build_torchattacks

    class _Probe:
        def __init__(self, atk):
            self.atk = atk

        def __call__(self, *a, **kw):
            probe["atk_states"].append(bool(get_mk()))
            probe["in_atk"] = True
            try:
                if probe["raise"]:
                    raise RuntimeError("synthetic exception inside atk()")
                return self.atk(*a, **kw)
            finally:
                probe["in_atk"] = False

    def hook(_module, _inputs):
        (probe["fwd_in"] if probe["in_atk"] else probe["fwd_out"]).append(bool(get_mk()))

    def reset():
        probe.update({"atk_states": [], "fwd_in": [], "fwd_out": [], "raise": False})

    handle = awn.model.register_forward_pre_hook(hook)
    am._build_torchattacks = lambda *a, **kw: _Probe(real_build(*a, **kw))
    key0 = subset[0]
    x0, seed0 = inputs[key0]["x"], inputs[key0]["seed"]
    by_attack = {p["attack"]: p for p in profiles}
    try:
        for attack in extended:
            prof = by_attack[attack]
            reset()
            _, meta, _ = call("ext", prof, x0, seed0)
            ok = (probe["atk_states"] == [False] and probe["fwd_in"] and not any(probe["fwd_in"])
                  and probe["fwd_out"] and all(probe["fwd_out"]) and bool(get_mk()) and meta.get(EXT_KEY) is True)
            if not ok:
                fail(f"C {attack} B=1 ON: atk={probe['atk_states']} fwd_in_any_true={any(probe['fwd_in'])} "
                     f"fwd_out_all_true={all(probe['fwd_out']) if probe['fwd_out'] else None}")
            rows.append({"section": "C", "attack": attack, "variant": "b1_on_scope", "classification": "OK" if ok else "FAIL"})
            reset()
            call("off", prof, x0, seed0)
            if probe["atk_states"] != [True]:
                fail(f"C {attack} option OFF: atk ran with mkldnn {probe['atk_states']}")
        prof = by_attack["cw"]
        x2 = np.concatenate([inputs[k]["x"] for k in subset[:2]], axis=0) if len(subset) >= 2 else np.repeat(x0, 2, 0)
        reset()
        _, meta, _ = call("ext", prof, x2, seed0)
        if probe["atk_states"] != [True] or meta.get(EXT_KEY) is not False:
            fail(f"C cw B=2: atk={probe['atk_states']} flag={meta.get(EXT_KEY)} (must not apply)")
        reset()
        set_mk(False)
        torch.manual_seed(seed0)
        with contextlib.redirect_stdout(io.StringIO()):
            ad["ext"].apply(x0, attack="cw", eps=prof["eps"], cw_c=prof["cw_c"], cw_steps=prof["cw_steps"],
                            cw_lr=prof["cw_lr"], attack_params=dict(prof["attack_params"]))
        prev_kept = bool(get_mk()) is False
        set_mk(True)
        if not prev_kept:
            fail("C previous mkldnn state False was not preserved by apply()")
        reset()
        probe["raise"] = True
        torch.manual_seed(seed0)
        with contextlib.redirect_stdout(io.StringIO()):
            _, meta = ad["ext"].apply(x0, attack="cw", eps=prof["eps"], cw_c=prof["cw_c"], cw_steps=prof["cw_steps"],
                                      cw_lr=prof["cw_lr"], attack_params=dict(prof["attack_params"]))
        probe["raise"] = False
        if bool(get_mk()) is not True or meta.get("attack_status") != "fallback":
            fail(f"C exception inside atk(): mkldnn={bool(get_mk())} status={meta.get('attack_status')}")
            set_mk(True)
        try:
            with mkldnn_disabled_scope(torch):
                raise ValueError("probe")
        except ValueError:
            pass
        if bool(get_mk()) is not True:
            fail("C mkldnn_disabled_scope did not restore after an exception")
            set_mk(True)
        if awn.model.training or wm.training or [p.requires_grad for p in awn.model.parameters()] != req_grad_ref:
            fail("C model mode / requires_grad changed during probes")
    finally:
        am._build_torchattacks = real_build
        handle.remove()

    with (out / "per_sample.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    result = {"overall": "PASS" if not failures else "FAIL", "n_failures": len(failures), "failures": failures[:200],
              "extended_attacks": extended, "not_extended_checked": NOT_EXTENDED,
              "deterministic_seed_attacks": sorted(DETERMINISTIC_SEED_ATTACKS),
              "deterministic_seed_rule": "attack_params[\"seed\"] = formal per-row seed (production path)",
              "option_on_class_counts": counts, "samples": len(subset), "base_ref": args.base_ref,
              "threads": {"intra": INTRA, "interop": INTEROP}, "torch_version": torch.__version__,
              "attack_adapter_sha256": p1a.sha256_file(REPO_ROOT / "src/adapters/attack_adapter.py"),
              "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}
    (out / "validation.json").write_text(json.dumps(result, indent=2, default=str))
    p1a.log(f"[done] {out} overall={result['overall']} failures={len(failures)}")
    for slug, cc in counts.items():
        p1a.log(f"  {slug:32s} {cc}")
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
