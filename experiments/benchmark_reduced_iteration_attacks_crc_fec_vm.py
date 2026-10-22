"""
Reduced-iteration attacks x AMC x CRC x FEC x latency (synthetic TX/RX domain, x86 VM, CPU, B=1).
EXPERIMENT ONLY -- production adapters and external/adversarial-rf are not modified.

Domain: synthetic bursts with known payload bits (src/adapters/synth_txrx_adapter.py -> external/
adversarial-rf/util/synth_txrx.py). RadioML2016 has no payload bits; its AMC-only counterpart is
experiments/benchmark_reduced_iteration_attacks_vm.py, whose attack profiles (FGSM eps; PGD steps; CW steps;
EAD max_iterations then binary_search_steps) are imported unchanged.

Per burst (src/utils/synth_crc_pipeline.py:run_burst_chain):
  TX bits -> CRC-8 -> [rate-1/2 K=7 conv code + block interleave] -> modulation -> IQ
  -> AttackAdapter (production) -> AWN -> demod -> [soft Viterbi] -> CRC-8
Two receivers demodulate the attacked window:
  oracle     : true modulation -- is the adversarial waveform itself still decodable?
  predicted  : the attacked AMC prediction -- does a receiver that relies on AMC still pass CRC?

Metrics per (group, attack, profile, fec_enabled), payload-degenerate frames excluded:
  n_total, n_clean_correct, n_attack_success (formal: clean_correct AND NOT attacked_correct)
  conditional_asr_pct                     = n_attack_success / n_clean_correct
  oracle_crc_given_attack_success_pct     = n_attack_success_oracle_crc_pass / n_attack_success
  predicted_crc_given_attack_success_pct  = n_attack_success_predicted_crc_pass / n_attack_success
  joint_amc_oracle_crc_pct                = n_attack_success_oracle_crc_pass / n_clean_correct
  joint_amc_predicted_crc_pct             = n_attack_success_predicted_crc_pass / n_clean_correct
  A zero denominator gives null plus a reason in na_reasons, never 0.
  BER (frame bits incl. CRC, bounded to the frame) of the oracle / predicted receivers over demodulated frames.
  *_frame_correct_given_attack_success_pct: CRC pass AND recovered frame bit-exact. CRC-8 can accept a wrong
  frame (observed mainly when the predicted receiver demodulates QAM64 as QAM16), so a CRC pass of the
  predicted receiver is not always correct data; both are reported, minimum usable uses CRC pass as specified.
  Latency: AttackAdapter.apply() wall time (primary, as in the AMC study); full chain wall time (secondary).

Degenerate frames: synth_txrx's FEC capacity rule leaves 0 payload bits for QPSK and PAM4 with FEC (coded
frame all zeros, constant data symbols). They are run and kept in raw_bursts.csv (payload_degenerate=True)
but excluded from every summary and from minimum-usable selection.

Duplicate profiles (EAD Phase 2 bss9 at Phase 1's max_iterations) are run once; the Phase 2 summary row is
built from the same raw rows and records reused_from.

Minimum usable, per attack x FEC (baseline = formal default profile of that attack in this domain:
PGD steps 10, CW steps 20, EAD 9 x 100 for both EAD phases):
  amc_usable              conditional ASR >= 90 % of baseline
  joint_oracle_usable     amc_usable AND oracle_crc_given_attack_success >= 90 % of baseline
  joint_predicted_usable  amc_usable AND predicted_crc_given_attack_success >= 90 % of baseline
FGSM has no iterations and is reported, not minimized.

Run from the repo root:
    ~/adversarial-rf/.venv/bin/python experiments/benchmark_reduced_iteration_attacks_crc_fec_vm.py \
        --checkpoint ~/adversarial-rf/2016.10a_AWN.pkl [--smoke]
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import io
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ADV_RF_ROOT = REPO_ROOT / "external" / "adversarial-rf"
for _p in (REPO_ROOT, ADV_RF_ROOT, REPO_ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import benchmark_reduced_iteration_attacks_vm as amc_study  # noqa: E402 -- profiles, env metadata, helpers
import run_cpu_full_matrix as formal  # noqa: E402 -- derive_seed
from src.adapters import synth_txrx_adapter as stx  # noqa: E402
from src.utils import synth_crc_pipeline as scp  # noqa: E402

BASE_SEED = 42
EXPECTED_CHECKPOINT_SHA256 = amc_study.EXPECTED_CHECKPOINT_SHA256
EXPECTED_ADV_RF_COMMIT = amc_study.EXPECTED_ADV_RF_COMMIT
EXPECTED_AWN_COMMIT = "0b3d2b2d3d5d4761e73eb91e223844b4b8585b53"
RAW_DROP = {"tx_channel_bits"}  # long derived string; frame bits and recovered bits are kept
BASELINE = {"pgd": ("pgd", "eps0.03_steps10"), "cw": ("cw", "c1_steps20_lr0.01"),
            "ead_phase1": ("ead_phase1", "bss9_maxit100"), "ead_phase2": ("ead_phase1", "bss9_maxit100")}
ITER_PARAM = {"pgd": "steps", "cw": "steps", "ead_phase1": "max_iterations", "ead_phase2": "binary_search_steps"}
ALLOWED_DEMOD_STATUS = {stx.DEMOD_OK, stx.DEMOD_NOT_DIGITAL, stx.DEMOD_FAMILY_MISMATCH}


def log(msg: str, fh) -> None:
    line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, file=sys.stderr, flush=True)
    fh.write(line + "\n")
    fh.flush()


def attack_kwargs(prof: dict) -> dict:
    """amc_study profile -> AttackAdapter.apply kwargs (identical arguments to the AMC study)."""
    return {"cw_c": prof["cw_c"], "cw_steps": prof["cw_steps"], "cw_lr": prof["cw_lr"],
            "attack_params": dict(prof["attack_params"])}


def ratio(num: int, den: int, name: str, reasons: list):
    if den == 0:
        reasons.append(f"{name}: denominator 0")
        return None
    return 100.0 * num / den


def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def summarize_rows(rr: list) -> dict:
    reasons = []
    cc = [r for r in rr if r["clean_correct"]]
    succ = [r for r in rr if r["attack_success"]]
    s_or = sum(r["attacked_oracle_crc_pass"] for r in succ)
    s_pr = sum(r["attacked_amc_crc_pass"] for r in succ)
    f_or = sum(r["attacked_oracle_frame_correct"] for r in succ)
    f_pr = sum(r["attacked_amc_frame_correct"] for r in succ)
    lat = amc_study.dist([r["attack_latency_ms"] for r in rr])
    chain = amc_study.dist([r["chain_latency_ms"] for r in rr])
    out = {
        "n_total": len(rr), "n_clean_correct": len(cc), "n_attack_success": len(succ),
        "n_attack_success_oracle_crc_pass": s_or, "n_attack_success_predicted_crc_pass": s_pr,
        "conditional_asr_pct": ratio(len(succ), len(cc), "conditional_asr", reasons),
        "oracle_crc_given_attack_success_pct": ratio(s_or, len(succ), "oracle_crc_given_attack_success", reasons),
        "predicted_crc_given_attack_success_pct": ratio(s_pr, len(succ), "predicted_crc_given_attack_success",
                                                        reasons),
        "n_attack_success_oracle_frame_correct": f_or, "n_attack_success_predicted_frame_correct": f_pr,
        "oracle_frame_correct_given_attack_success_pct": ratio(f_or, len(succ), "oracle_frame_correct", reasons),
        "predicted_frame_correct_given_attack_success_pct": ratio(f_pr, len(succ), "predicted_frame_correct",
                                                                  reasons),
        "joint_amc_oracle_crc_pct": ratio(s_or, len(cc), "joint_amc_oracle_crc", reasons),
        "joint_amc_predicted_crc_pct": ratio(s_pr, len(cc), "joint_amc_predicted_crc", reasons),
        "clean_oracle_crc_pass_pct": ratio(sum(r["clean_oracle_crc_pass"] for r in rr), len(rr), "clean_oracle", reasons),
        "clean_predicted_crc_pass_pct": ratio(sum(r["clean_amc_crc_pass"] for r in rr), len(rr), "clean_pred", reasons),
        "attacked_oracle_crc_pass_pct": ratio(sum(r["attacked_oracle_crc_pass"] for r in rr), len(rr), "att_or", reasons),
        "attacked_predicted_crc_pass_pct": ratio(sum(r["attacked_amc_crc_pass"] for r in rr), len(rr), "att_pr",
                                                 reasons),
        "oracle_ber_mean": mean_or_none([r["attacked_oracle_ber"] for r in rr]),
        "predicted_ber_mean": mean_or_none([r["attacked_amc_ber"] for r in rr]),
        "oracle_ber_mean_given_attack_success": mean_or_none([r["attacked_oracle_ber"] for r in succ]),
        "predicted_ber_mean_given_attack_success": mean_or_none([r["attacked_amc_ber"] for r in succ]),
        "n_predicted_not_demodulated": sum(r["attacked_amc_demod_status"] != stx.DEMOD_OK for r in rr),
        "n_predicted_not_digital": sum(r["attacked_amc_demod_status"] == stx.DEMOD_NOT_DIGITAL for r in rr),
        "n_predicted_family_mismatch": sum(r["attacked_amc_demod_status"] == stx.DEMOD_FAMILY_MISMATCH for r in rr),
        "crc_false_accept_n": sum(r[f"{sc}_crc_false_accept"] for r in rr for sc in scp.DEMOD_SCENARIOS),
        "oracle_crc_false_accept_n": sum(r["attacked_oracle_crc_false_accept"] for r in rr),
        "predicted_crc_false_accept_n": sum(r["attacked_amc_crc_false_accept"] for r in rr),
        "latency_median_ms": lat["median"], "latency_mean_ms": lat["mean"], "latency_p95_ms": lat["p95"],
        "chain_latency_median_ms": chain["median"], "chain_latency_p95_ms": chain["p95"],
        "linf_median": amc_study.dist([r["linf"] for r in rr])["median"],
        "l2_median": amc_study.dist([r["l2"] for r in rr])["median"],
        "n_degenerate_included": sum(bool(r["payload_degenerate"]) for r in rr),
    }
    if not succ:
        reasons.append("no attack successes: BER/CRC given success undefined")
    out["na_reasons"] = "; ".join(reasons)
    return out


def build_summaries(rows: list, profile_meta: dict) -> list:
    """rows: non-degenerate raw rows. profile_meta: (group, profile) -> dict incl. source key for reuse."""
    out = []
    for (group, profile), meta in profile_meta.items():
        src_group, src_profile = meta["source"]
        for fec in (False, True):
            rr = [r for r in rows if r["group"] == src_group and r["profile"] == src_profile
                  and r["fec_enabled"] == fec]
            if not rr:
                continue
            out.append({"group": group, "attack": meta["attack"], "profile": profile, "fec_enabled": fec,
                        "soft_demod": True, "is_baseline": meta["is_baseline"], "eps": meta["eps"],
                        "alpha": meta["alpha"], "steps": meta["steps"],
                        "binary_search_steps": meta["binary_search_steps"], "max_iterations": meta["max_iterations"],
                        "reused_from": None if meta["source"] == (group, profile) else f"{src_group}:{src_profile}",
                        **summarize_rows(rr)})
    # retention vs the attack's formal default baseline (same domain, same FEC setting)
    idx = {(s["group"], s["profile"], s["fec_enabled"]): s for s in out}
    for s in out:
        b = idx.get(BASELINE[s["group"]] + (s["fec_enabled"],)) if s["group"] in BASELINE else None
        s["baseline_profile"] = f"{BASELINE[s['group']][0]}:{BASELINE[s['group']][1]}" if b else None
        for m in ("conditional_asr_pct", "oracle_crc_given_attack_success_pct",
                  "predicted_crc_given_attack_success_pct"):
            bv = b[m] if b else None
            s[f"{m}_baseline"] = bv
            s[f"{m}_retention_pct"] = (100.0 * s[m] / bv) if (bv and s[m] is not None) else None
        s["latency_reduction_vs_baseline_pct"] = (100 * (1 - s["latency_median_ms"] / b["latency_median_ms"])
                                                  if b and b["latency_median_ms"] else None)

        def ok(m):
            v, bv = s[m], s[f"{m}_baseline"]
            return None if (v is None or bv is None) else bool(v >= 0.9 * bv)
        s["amc_usable"] = ok("conditional_asr_pct")
        oc, pc = ok("oracle_crc_given_attack_success_pct"), ok("predicted_crc_given_attack_success_pct")
        s["joint_oracle_usable"] = None if s["amc_usable"] is None or oc is None else (s["amc_usable"] and oc)
        s["joint_predicted_usable"] = None if s["amc_usable"] is None or pc is None else (s["amc_usable"] and pc)
    return out


def min_usable(summary: list) -> list:
    out = []
    crit_metric = {"amc_usable": None, "joint_oracle_usable": "oracle_crc_given_attack_success_pct",
                   "joint_predicted_usable": "predicted_crc_given_attack_success_pct"}
    for group, k in ITER_PARAM.items():
        for fec in (False, True):
            ss = sorted([s for s in summary if s["group"] == group and s["fec_enabled"] == fec], key=lambda s: s[k])
            if not ss:
                continue
            for crit, metric in crit_metric.items():
                hit = [s for s in ss if s[crit]]
                h = hit[0] if hit else None
                row = {"attack": ss[0]["attack"], "group": group, "fec_enabled": fec, "criterion": crit,
                       "iteration_param": k, "baseline_profile": ss[0]["baseline_profile"],
                       "baseline_conditional_asr_pct": ss[0]["conditional_asr_pct_baseline"],
                       "baseline_crc_metric_pct": ss[0][f"{metric}_baseline"] if metric else None,
                       "crc_metric": metric,
                       "n_degenerate_in_selection": sum(s["n_degenerate_included"] for s in ss),
                       "minimum_usable_profile": h["profile"] if h else None,
                       "minimum_usable_value": h[k] if h else None,
                       "conditional_asr_pct": h["conditional_asr_pct"] if h else None,
                       "asr_retention_pct": h["conditional_asr_pct_retention_pct"] if h else None,
                       "crc_metric_pct": h[metric] if (h and metric) else None,
                       "crc_retention_pct": h[f"{metric}_retention_pct"] if (h and metric) else None,
                       "latency_median_ms": h["latency_median_ms"] if h else None,
                       "latency_reduction_pct": h["latency_reduction_vs_baseline_pct"] if h else None,
                       "note": None if h else "no profile in the sweep meets the criterion "
                                              "(or baseline metric undefined)"}
                out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--snrs", default="18")
    ap.add_argument("--mods", default=",".join(stx.digital_modulations()))
    ap.add_argument("--bursts-per-cell", type=int, default=25)
    ap.add_argument("--fec", default="0,1", help="FEC requested settings, comma separated 0/1")
    ap.add_argument("--ead-phase2-max-iterations", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--interop-threads", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="baseline + lowest setting per attack, 1 burst/cell")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    import torch
    import torchattacks as ta
    torch.set_num_interop_threads(args.interop_threads)
    torch.set_num_threads(args.torch_threads)
    from src.adapters.attack_adapter import AttackAdapter
    from src.adapters.awn_adapter import AWNModelAdapter

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (REPO_ROOT / (args.out_dir or f"results/reduced_iteration_crc_fec_vm_{'smoke_' if args.smoke else ''}{ts}"))
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fh = (out / "run.log").open("w")
    t_start = time.monotonic()
    checks, state = {}, {"completed": False}
    rows, profile_meta, run_keys = [], {}, []
    n_bursts_total = 0
    try:
        stx.require_backend()
        state["git"] = amc_study.git_metadata()
        state["environment"] = amc_study.environment_metadata(torch, ta)
        sub = state["git"]["submodule_status"] or ""
        checks["adversarial_rf_at_pin"] = state["git"]["adversarial_rf_head"] == EXPECTED_ADV_RF_COMMIT
        checks["awn_submodule_at_pin"] = EXPECTED_AWN_COMMIT in sub
        ckpt = Path(args.checkpoint).expanduser().resolve()
        state["checkpoint_sha256"] = amc_study.sha256_file(ckpt)
        checks["checkpoint_hash_ok"] = state["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
        if not all(checks.values()):
            raise RuntimeError(f"preflight failed: {checks}")
        with contextlib.redirect_stdout(io.StringIO()):
            awn = AWNModelAdapter(str(ckpt))
            adapter = AttackAdapter(awn_model=awn.model, b1_pgd_bim_mkldnn_off=True, b1_mkldnn_off_extended=True)
        checks["real_backends_loaded"] = awn.status == "ok" and adapter.status == "ok"
        if not checks["real_backends_loaded"]:
            raise RuntimeError(f"backend not loaded: awn={awn.notes} attack={adapter.notes}")

        snrs = [int(s) for s in args.snrs.split(",")]
        mods = args.mods.split(",")
        fecs = [bool(int(f)) for f in args.fec.split(",")]
        n_bursts = 1 if args.smoke else args.bursts_per_cell
        bursts = []
        for mod in mods:
            for snr in snrs:
                for fec in fecs:
                    for i in range(n_bursts):
                        seed = formal.derive_seed(BASE_SEED, mod, snr, i, f"synth_txrx|fec{int(fec)}")
                        bursts.append((mod, snr, fec, i, seed, stx.generate_frame(mod, snr, seed, fec)))
        n_bursts_total = len(bursts)
        log(f"{len(bursts)} bursts ({mods} x SNR {snrs} x FEC {fecs} x {n_bursts}); out={out}", fh)

        def run_group(group: str, profiles: list) -> None:
            if args.smoke:
                profiles = amc_study.smoke_select(profiles)
            for prof in profiles:
                meta = {k: prof[k] for k in ("attack", "eps", "alpha", "steps", "binary_search_steps",
                                             "max_iterations", "is_baseline")}
                dup = next(((g, p) for (g, p) in run_keys if p == prof["profile"] and
                            profile_meta[(g, p)]["attack"] == prof["attack"]), None)
                if dup is not None:  # identical attack configuration already run: reuse, keep mapping
                    profile_meta[(group, prof["profile"])] = {**meta, "source": dup}
                    log(f"[{group}] {prof['attack']}/{prof['profile']}: duplicate of {dup[0]} -- reused", fh)
                    continue
                profile_meta[(group, prof["profile"])] = {**meta, "source": (group, prof["profile"])}
                run_keys.append((group, prof["profile"]))
                t0 = time.monotonic()
                kw = attack_kwargs(prof)
                for w in range(args.warmup):  # untimed
                    _, _, _, _, seed, b = bursts[w % len(bursts)]
                    scp.run_burst_chain(b, awn=awn, attack_adapter=adapter, attack=prof["attack"], eps=prof["eps"],
                                        torch_seed=seed, attack_seed=0, attack_kwargs=kw)
                for mod, snr, fec, i, seed, b in bursts:
                    a_seed = formal.derive_seed(BASE_SEED, mod, snr, i,
                                                f"{prof['attack']}|{prof['profile']}|fec{int(fec)}")
                    tc = time.perf_counter_ns()
                    r, _ = scp.run_burst_chain(b, awn=awn, attack_adapter=adapter, attack=prof["attack"],
                                               eps=prof["eps"], torch_seed=seed, attack_seed=a_seed,
                                               attack_kwargs=kw)
                    r["chain_latency_ms"] = (time.perf_counter_ns() - tc) / 1e6
                    r.update(group=group, profile=prof["profile"], is_baseline=prof["is_baseline"],
                             steps=prof["steps"], binary_search_steps=prof["binary_search_steps"],
                             max_iterations=prof["max_iterations"], burst_index=i, burst_seed=seed,
                             payload_degenerate=r["n_data_bits"] == 0)
                    rows.append({k: v for k, v in r.items() if k not in RAW_DROP})
                log(f"[{group}] {prof['attack']}/{prof['profile']}: {len(bursts)} bursts "
                    f"({time.monotonic() - t0:.0f} s)", fh)

        run_group("fgsm", amc_study.fgsm_profiles())
        run_group("pgd", amc_study.pgd_profiles())
        run_group("cw", amc_study.cw_profiles())
        run_group("ead_phase1", amc_study.ead_phase1_profiles())
        run_group("ead_phase2", amc_study.ead_phase2_profiles(args.ead_phase2_max_iterations))
        state["completed"] = True
    except Exception as exc:  # noqa: BLE001 -- recorded; validation FAILs
        state["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        log(f"ERROR: {state['error']}", fh)

    amc_study.write_csv(out / "raw_bursts.csv", rows)
    valid = [r for r in rows if not r["payload_degenerate"]]
    summary = build_summaries(valid, profile_meta) if valid else []
    mins = min_usable(summary) if summary else []
    amc_study.write_csv(out / "profile_summary.csv", summary)
    amc_study.write_csv(out / "min_usable.csv", mins)
    mod_rows = []
    for (g, p), meta in profile_meta.items():
        if meta["source"] != (g, p):
            continue
        for mod in sorted({r["true_modulation"] for r in rows}):
            for fec in (False, True):
                rr = [r for r in rows if r["group"] == g and r["profile"] == p and r["true_modulation"] == mod
                      and r["fec_enabled"] == fec]
                if rr:
                    mod_rows.append({"group": g, "attack": meta["attack"], "profile": p, "modulation": mod,
                                     "fec_enabled": fec, "payload_degenerate": rr[0]["payload_degenerate"],
                                     **summarize_rows(rr)})
    amc_study.write_csv(out / "modulation_summary.csv", mod_rows)

    # ---------------------------------------------------------------- validation
    checks["completed"] = state["completed"]
    checks["attack_fallback_zero"] = bool(rows) and all(
        r["attack_status"] == "ok" and r["attack_backend"] != "dummy_attack" for r in rows)
    checks["demod_status_allowed_only"] = all(r[f"{sc}_demod_status"] in ALLOWED_DEMOD_STATUS
                                              for r in rows for sc in scp.DEMOD_SCENARIOS)
    checks["crc_pass_is_bool"] = all(isinstance(r[f"{sc}_crc_pass"], (bool, np.bool_))
                                     for r in rows for sc in scp.DEMOD_SCENARIOS)
    checks["no_nan_inf"] = all(math.isfinite(r["linf"]) and math.isfinite(r["l2"]) and
                               math.isfinite(r["attack_latency_ms"]) for r in rows)
    expected_profiles = len(run_keys)
    per_key = {}
    for r in rows:
        per_key[(r["group"], r["profile"])] = per_key.get((r["group"], r["profile"]), 0) + 1
    checks["expected_matrix_complete"] = (
        bool(run_keys) and all(per_key.get(k) == n_bursts_total for k in run_keys)
        and len(rows) == expected_profiles * n_bursts_total
        and (args.smoke or expected_profiles == 26))
    checks["degenerate_excluded_from_summary_and_min"] = (
        all(s["n_degenerate_included"] == 0 for s in summary)
        and all(m["n_degenerate_in_selection"] == 0 for m in mins))
    checks["all_four_attacks_present"] = {r["attack"] for r in rows} >= {"fgsm", "pgd", "cw", "ead"}
    gated = ["completed", "attack_fallback_zero", "demod_status_allowed_only", "crc_pass_is_bool", "no_nan_inf",
             "expected_matrix_complete", "degenerate_excluded_from_summary_and_min", "all_four_attacks_present",
             "adversarial_rf_at_pin", "awn_submodule_at_pin", "checkpoint_hash_ok", "real_backends_loaded"]
    failed = [k for k in gated if not checks.get(k)]
    validation = {
        "overall": "PASS" if not failed else "FAIL", "failed_checks": failed, "checks": checks,
        "dataset_sha256": "not applicable: synthetic TX/RX domain uses no RadioML dataset",
        "n_rows": len(rows), "n_valid_rows": len(valid), "n_degenerate_rows": len(rows) - len(valid),
        "n_unique_profiles_run": expected_profiles, "n_profiles_in_summary": len(profile_meta),
        "reused_profiles": {f"{g}:{p}": f"{m['source'][0]}:{m['source'][1]}"
                            for (g, p), m in profile_meta.items() if m["source"] != (g, p)},
        "informational": {"crc_false_accept_total": sum(r[f"{sc}_crc_false_accept"] for r in rows
                                                        for sc in scp.DEMOD_SCENARIOS),
                          "note": "CRC-8 false accepts (~1/256 for corrupted frames) are reported, not gated"},
        "error": state.get("error")}
    (out / "validation.json").write_text(json.dumps(validation, indent=2, default=str))
    manifest = {
        "experiment": "reduced_iteration_crc_fec_vm", "mode": "smoke" if args.smoke else "full",
        "command": [sys.executable] + sys.argv, "args": vars(args), "elapsed_s": time.monotonic() - t_start,
        "git": state.get("git"), "environment": state.get("environment"),
        "checkpoint_sha256": state.get("checkpoint_sha256"),
        "domain": "synthetic TX/RX (payload bits known); NOT RadioML2016",
        "burst_params": stx.LEGACY_CRC_PIPELINE_BURST, "synth_txrx": stx.SYNTH_TXRX_SOURCE, "soft_demod": True,
        "seed_rule": "burst: derive_seed(42, mod, snr, i, 'synth_txrx|fec<0|1>'); torch.manual_seed(burst seed) "
                     "before every attack; attack seed derive_seed(..., '<attack>|<profile>|fec<0|1>')",
        "baselines": BASELINE, "usable_rule": "metric >= 0.9 x same-domain baseline (same FEC setting)",
        "degenerate_frames": "QPSK/PAM4 with FEC: 0 payload bits, constant data symbols; kept in raw, "
                             "excluded from summaries and minimum selection",
        "fsk_window_note": "attacked FSK windows are demodulated (legacy scripts demodulated the clean FSK "
                           "window; see synth_txrx_adapter.demodulate)",
        "latency": {"primary": "AttackAdapter.apply() wall time (perf_counter_ns), one timed call per burst "
                               "after --warmup untimed calls per profile",
                    "secondary": "run_burst_chain() wall time (AMC x2, attack, demod x4, FEC decode, CRC)",
                    "platform": "VMware VM on i7-11700, torch intra 1 / interop 1; not a native-desktop latency"},
        "mkldnn_note": "AttackAdapter B=1 mkldnn-off opt-ins as in the AMC study; validated on Raspberry Pi only",
        "validation_overall": validation["overall"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log(f"validation {validation['overall']} failed={failed} ({time.monotonic() - t_start:.0f} s)", fh)
    fh.close()
    sys.exit(0 if validation["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
