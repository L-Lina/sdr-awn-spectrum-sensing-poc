"""
Clean-only defense control (synthetic TX/RX domain, x86 VM, CPU, B=1). No adversarial attack is generated.
EXPERIMENT ONLY -- production defense / synth_txrx code is not modified.

Purpose: separate defense-induced clean AMC / payload degradation from attack recovery measured in
experiments/benchmark_reduced_iteration_crc_fec_defense_vm.py.

Domain and seeds: identical to that run (SNR 18 dB, 8 digital modulations x FEC requested {off, on} x 25 bursts,
burst seed derive_seed(42, mod, snr, i, 'synth_txrx|fec<0|1>')), so every clean burst is the same waveform the
attack run attacked.

Per burst: the clean window (SHA-256 recorded, re-checked per branch) goes to
  none | TopKAdapter(fixed) K=10,20,30,40 | TopKAdapter(adaptive) = adaptive_k_v2_snr_defense (defaults)
and each output gets AMC (defended_pred), oracle demod (true modulation) and predicted-AMC demod (defended_pred),
FEC decode when coded, CRC-8 check, bit / frame correctness. Defense latency = TopKAdapter.apply() wall time.
Adaptive-K decisions come from the same untimed production-helper trace as the attack run (output verified
bit-identical to TopKAdapter for every sample). SNR source: blind estimate from the defense input waveform.

Terminology: a defended clean burst whose prediction is wrong is a defense-induced misclassification
(clean_correct_after_defense = False); it is never counted as attack success.

attack_run_decomposition.csv joins this control with the attack run on (attack, profile, modulation, FEC request,
burst index): for every clean-correct burst counted as attack_success_after_defense in the attack run, it records
whether the same defense alone already misclassifies the clean burst.

Run from the repo root:
    ~/adversarial-rf/.venv/bin/python experiments/benchmark_clean_crc_fec_defense_vm.py \
        --checkpoint ~/adversarial-rf/2016.10a_AWN.pkl [--smoke]
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as _dt
import inspect
import json
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ADV_RF_ROOT = REPO_ROOT / "external" / "adversarial-rf"
for _p in (REPO_ROOT, ADV_RF_ROOT, REPO_ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import benchmark_reduced_iteration_attacks_vm as amc_study  # noqa: E402 -- env metadata, helpers
import benchmark_reduced_iteration_crc_fec_defense_vm as dstudy  # noqa: E402 -- receive, adaptive_trace, constants
import run_cpu_full_matrix as formal  # noqa: E402 -- derive_seed, sha256_array
from src.adapters import synth_txrx_adapter as stx  # noqa: E402
from src.utils import synth_crc_pipeline as scp  # noqa: E402

BASE_SEED = dstudy.BASE_SEED
DEFENSES = dstudy.DEFENSES
ATTACK_DEFENSE_RUN = "results/reduced_iteration_crc_fec_defense_vm_20261022_103042"


def log(msg: str, fh) -> None:
    line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, file=sys.stderr, flush=True)
    fh.write(line + "\n")
    fh.flush()


def pct(num, den, name, reasons):
    if den == 0:
        reasons.append(f"{name}: denominator 0")
        return None
    return 100.0 * num / den


def summarize(rr: list) -> dict:
    reasons = []
    n = len(rr)
    cc0 = sum(r["clean_correct_none"] for r in rr)
    cc1 = sum(r["clean_correct_after_defense"] for r in rr)
    dl = [r["defense_latency_ms"] for r in rr if r["defense_latency_ms"] is not None]
    d = amc_study.dist(dl)
    a0 = pct(cc0, n, "clean_accuracy_none", reasons)
    a1 = pct(cc1, n, "clean_accuracy_after_defense", reasons)
    ks = [r["adaptive_selected_k"] for r in rr if r["adaptive_selected_k"] is not None]
    return {
        "n_total": n, "n_valid": n, "n_clean_correct_none": cc0, "n_clean_correct_after_defense": cc1,
        "clean_accuracy_none_pct": a0, "clean_accuracy_after_defense_pct": a1,
        "clean_degradation_pp": None if (a0 is None or a1 is None) else a0 - a1,
        "n_defense_induced_misclassification": sum(r["clean_correct_none"] and not r["clean_correct_after_defense"]
                                                    for r in rr),
        "n_defense_induced_correction": sum((not r["clean_correct_none"]) and r["clean_correct_after_defense"]
                                            for r in rr),
        "oracle_crc_pass_pct": pct(sum(r["oracle_crc_pass"] for r in rr), n, "oracle_crc", reasons),
        "oracle_frame_correct_pct": pct(sum(r["oracle_frame_correct"] for r in rr), n, "oracle_fc", reasons),
        "oracle_ber_mean": dstudy.mean_or_none([r["oracle_ber"] for r in rr]),
        "predicted_crc_pass_pct": pct(sum(r["predicted_crc_pass"] for r in rr), n, "pred_crc", reasons),
        "predicted_frame_correct_pct": pct(sum(r["predicted_frame_correct"] for r in rr), n, "pred_fc", reasons),
        "predicted_ber_mean": dstudy.mean_or_none([r["predicted_ber"] for r in rr]),
        "oracle_crc_false_accept_n": sum(r["oracle_crc_false_accept"] for r in rr),
        "predicted_crc_false_accept_n": sum(r["predicted_crc_false_accept"] for r in rr),
        "defense_latency_median_ms": d["median"] if dl else None,
        "defense_latency_mean_ms": d["mean"] if dl else None,
        "defense_latency_p95_ms": d["p95"] if dl else None,
        "adaptive_selected_k_mean": float(np.mean(ks)) if ks else None,
        "n_degenerate_included": sum(bool(r["payload_degenerate"]) for r in rr),
        "na_reasons": "; ".join(reasons),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--snr", type=int, default=18)
    ap.add_argument("--bursts-per-cell", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--interop-threads", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="2 bursts/cell")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    import torch
    import torchattacks as ta
    torch.set_num_interop_threads(args.interop_threads)
    torch.set_num_threads(args.torch_threads)
    from src.adapters import topk_adapter as tka
    from src.adapters.awn_adapter import AWNModelAdapter
    import util.defense as D

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (REPO_ROOT / (args.out_dir or f"results/clean_crc_fec_defense_vm_{'smoke_' if args.smoke else ''}{ts}")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    fh = (out / "run.log").open("w")
    t_start = time.monotonic()
    checks, state, counters, rows = {}, {"completed": False}, Counter(), []
    n_bursts_total = 0
    defaults = {}
    try:
        stx.require_backend()
        state["git"] = amc_study.git_metadata()
        state["environment"] = amc_study.environment_metadata(torch, ta)
        checks["adversarial_rf_at_pin"] = state["git"]["adversarial_rf_head"] == dstudy.EXPECTED_ADV_RF_COMMIT
        checks["awn_submodule_at_pin"] = dstudy.EXPECTED_AWN_COMMIT in (state["git"]["submodule_status"] or "")
        ckpt = Path(args.checkpoint).expanduser().resolve()
        state["checkpoint_sha256"] = amc_study.sha256_file(ckpt)
        checks["checkpoint_hash_ok"] = state["checkpoint_sha256"] == dstudy.EXPECTED_CHECKPOINT_SHA256
        if not all(checks.values()):
            raise RuntimeError(f"preflight failed: {checks}")
        with contextlib.redirect_stdout(dstudy._Null()):
            awn = AWNModelAdapter(str(ckpt))
            topk_fixed = tka.TopKAdapter(mode="fixed")
            topk_adapt = tka.TopKAdapter(mode="adaptive")
        checks["real_backends_loaded"] = (awn.status == "ok" and topk_fixed.backend_available
                                          and topk_fixed.backend_name == tka._REAL_SOURCE
                                          and topk_adapt.backend_name == tka._ADAPTIVE_SOURCE)
        checks["adaptive_is_production_function"] = (tka._real_adaptive_k_v2_snr_defense is D.adaptive_k_v2_snr_defense
                                                     and tka._real_fft_topk_denoise is D.fft_topk_denoise)
        if not (checks["real_backends_loaded"] and checks["adaptive_is_production_function"]):
            raise RuntimeError(f"defense backend check failed: {checks}")
        defaults = {k: v.default for k, v in inspect.signature(D.adaptive_k_v2_snr_defense).parameters.items()
                    if k != "x"}

        def defend(name, x):
            if name == "none":
                return x, None, {"defense_backend": "none", "selected_k": None}
            if name.startswith("topk"):
                k = int(name[4:])
                with contextlib.redirect_stdout(dstudy._Null()):
                    t0 = time.perf_counter_ns()
                    y, meta = topk_fixed.apply(x, topk=k)
                    dt = (time.perf_counter_ns() - t0) / 1e6
                if meta.get("topk_status") != "ok" or meta.get("topk_backend") != tka._REAL_SOURCE:
                    counters["defense_fallback"] += 1
                    raise RuntimeError(f"fixed Top-K fallback: {meta}")
                if not np.array_equal(D.fft_topk_denoise(torch.from_numpy(x), topk=k).numpy().astype(np.float32), y):
                    counters["fixed_k_mismatch"] += 1
                    raise RuntimeError(f"TopKAdapter K={k} output != production fft_topk_denoise(K={k})")
                with torch.no_grad():
                    _, idx = torch.fft.fft(torch.from_numpy(x), dim=2).abs().topk(k=min(k, x.shape[2]), dim=2)
                if int(idx.shape[2]) != k:
                    counters["fixed_k_mismatch"] += 1
                    raise RuntimeError("effective K != requested K")
                return y, dt, {"defense_backend": meta["topk_backend"], "selected_k": int(idx.shape[2])}
            with contextlib.redirect_stdout(dstudy._Null()):
                t0 = time.perf_counter_ns()
                y, meta = topk_adapt.apply(x)
                dt = (time.perf_counter_ns() - t0) / 1e6
            if meta.get("topk_status") != "ok" or meta.get("topk_backend") != tka._ADAPTIVE_SOURCE:
                counters["defense_fallback"] += 1
                raise RuntimeError(f"Adaptive-K not on production backend: {meta}")
            yt, info = dstudy.adaptive_trace(D, defaults, x)
            if not np.array_equal(yt, y):
                counters["adaptive_trace_mismatch"] += 1
                raise RuntimeError("Adaptive-K trace does not reproduce the production output")
            i0 = info[0]
            return y, dt, {"defense_backend": meta["topk_backend"], "selected_k": i0["selected_k"],
                           "adaptive_route": i0["route"], "adaptive_snr_value": i0["snr_value"],
                           "adaptive_k_max": i0["k_max"], "adaptive_knee_uncapped": i0["knee_uncapped"]}

        n_b = 2 if args.smoke else args.bursts_per_cell
        bursts = []
        for mod in stx.digital_modulations():
            for fec in (False, True):
                for i in range(n_b):
                    seed = formal.derive_seed(BASE_SEED, mod, args.snr, i, f"synth_txrx|fec{int(fec)}")
                    bursts.append((mod, fec, i, seed, stx.generate_frame(mod, args.snr, seed, fec)))
        n_bursts_total = len(bursts)
        log(f"{len(bursts)} clean bursts x {len(DEFENSES)} defenses (no attack); out={out}", fh)
        for w in range(args.warmup):  # untimed defense warm-up
            x = np.ascontiguousarray(bursts[w % len(bursts)][4]["iq_tensor"], dtype=np.float32)
            for dname in DEFENSES[1:]:
                defend(dname, x)

        for mod, fec, i, seed, b in bursts:
            x = np.ascontiguousarray(b["iq_tensor"], dtype=np.float32)
            x.setflags(write=False)
            x_sha = formal.sha256_array(x)
            label = scp.RML2016_10A_CLASSES[mod]
            clean_pred_none = None
            none_recv = None
            for dname in DEFENSES:
                in_sha = formal.sha256_array(x)
                if in_sha != x_sha:
                    counters["clean_hash_mismatch"] += 1
                    raise RuntimeError("clean waveform changed between defense branches")
                y, dlat, dinfo = defend(dname, x)
                if not np.isfinite(y).all() or y.shape != x.shape:
                    raise RuntimeError(f"invalid defended window ({dname})")
                pred, recv = dstudy.receive(b, awn, y)
                if dname == "none":
                    clean_pred_none = pred
                tx = stx.tx_bit_trace(b)
                rows.append({
                    "defense": dname, "true_modulation": mod, "true_label": label, "snr_db": args.snr,
                    "burst_index": i, "burst_seed": seed, "fec_requested": fec, "effective_fec": bool(b["fec"]),
                    "payload_degenerate": len(b["data_bits"]) == 0, "n_data_bits": int(len(b["data_bits"])),
                    "n_frame_bits": int(len(tx["frame_bits"])), "soft_demod": True,
                    "clean_pred_none": clean_pred_none, "clean_pred_none_mod": scp.IDX_TO_MOD[clean_pred_none],
                    "clean_correct_none": clean_pred_none == label,
                    "defended_pred": pred, "defended_pred_mod": scp.IDX_TO_MOD[pred],
                    "clean_correct_after_defense": pred == label,
                    "defense_induced_misclassification": (clean_pred_none == label) and pred != label,
                    **recv,
                    "clean_input_sha256": x_sha, "defense_input_sha256": in_sha,
                    "defense_backend": dinfo["defense_backend"], "defense_latency_ms": dlat,
                    "selected_k": dinfo["selected_k"], "adaptive_route": dinfo.get("adaptive_route"),
                    "adaptive_selected_k": dinfo["selected_k"] if dname == "adaptive_k" else None,
                    "adaptive_snr_value": dinfo.get("adaptive_snr_value"),
                    "adaptive_snr_db": (10 * math.log10(dinfo["adaptive_snr_value"])
                                        if dinfo.get("adaptive_snr_value") else None),
                    "adaptive_snr_source": dstudy.SNR_SOURCE if dname == "adaptive_k" else None,
                    "adaptive_k_max": dinfo.get("adaptive_k_max"),
                    "adaptive_knee_uncapped": dinfo.get("adaptive_knee_uncapped"),
                })
        log(f"{len(rows)} rows", fh)
        state["completed"] = True
    except Exception as exc:  # noqa: BLE001 -- recorded; validation FAILs
        state["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        log(f"ERROR: {state['error']}", fh)

    amc_study.write_csv(out / "raw_samples.csv", rows)
    valid = [r for r in rows if not r["payload_degenerate"]]

    def group(keys, src):
        g = {}
        for r in src:
            g.setdefault(tuple(r[k] for k in keys), []).append(r)
        return g

    summ = [{**dict(zip(k_, k)), **summarize(rr)}
            for k_ in (("defense", "effective_fec"),) for k, rr in group(k_, valid).items()]
    amc_study.write_csv(out / "defense_clean_summary.csv", summ)
    mkeys = ("defense", "effective_fec", "true_modulation")
    amc_study.write_csv(out / "modulation_summary.csv",
                        [{**dict(zip(mkeys, k)), **summarize(rr)} for k, rr in group(mkeys, valid).items()])

    ak = [r for r in valid if r["defense"] == "adaptive_k"]
    ak_rows = []
    for keys in (("effective_fec",), ("true_modulation", "effective_fec")):
        for k, rr in group(keys, ak).items():
            ks = [r["adaptive_selected_k"] for r in rr if r["adaptive_selected_k"] is not None]
            snrs = [r["adaptive_snr_db"] for r in rr if r["adaptive_snr_db"] is not None]
            hist = Counter(ks)
            ak_rows.append({"grouping": "+".join(keys), **dict(zip(keys, k)), "n": len(rr),
                            "n_wideband_quantize": sum(r["adaptive_route"] == "wideband_quantize" for r in rr),
                            "n_narrowband_topk": len(ks), "selected_k_min": min(ks) if ks else None,
                            "selected_k_median": float(np.median(ks)) if ks else None,
                            "selected_k_mean": float(np.mean(ks)) if ks else None,
                            "selected_k_max": max(ks) if ks else None,
                            "n_k_capped": sum(r["adaptive_knee_uncapped"] is not None and
                                              r["adaptive_knee_uncapped"] > r["adaptive_k_max"] for r in rr),
                            "selected_k_histogram": json.dumps({str(x): hist[x] for x in sorted(hist)}),
                            "snr_est_db_median": float(np.median(snrs)) if snrs else None,
                            "snr_est_db_min": min(snrs) if snrs else None,
                            "snr_est_db_max": max(snrs) if snrs else None, "snr_source": dstudy.SNR_SOURCE})
    amc_study.write_csv(out / "adaptive_k_selection_summary.csv", ak_rows)

    lat = []
    for keys in (("defense",), ("defense", "effective_fec")):
        for k, rr in group(keys, valid).items():
            if k[0] == "none":
                continue
            d = amc_study.dist([r["defense_latency_ms"] for r in rr])
            lat.append({"grouping": "+".join(keys), **dict(zip(keys, k)), "n": d["n"],
                        "defense_latency_median_ms": d["median"], "defense_latency_mean_ms": d["mean"],
                        "defense_latency_p95_ms": d["p95"], "defense_latency_min_ms": d["min"],
                        "defense_latency_max_ms": d["max"]})
    amc_study.write_csv(out / "latency_summary.csv", lat)

    # ---------------------------------------------------------------- cross-run decomposition (attack run)
    decomp, cross = [], {"clean_pred_compared": 0, "clean_pred_identical": 0}
    ap_path = REPO_ROOT / ATTACK_DEFENSE_RUN / "raw_samples.csv"
    if ap_path.exists() and not args.smoke:
        clean = {(r["true_modulation"], r["fec_requested"], r["burst_index"], r["defense"]): r for r in rows}
        groups = {}
        with ap_path.open(newline="") as f:
            for r in csv.DictReader(f):
                if r["payload_degenerate"] == "True":
                    continue
                key = (r["true_modulation"], r["fec_requested"] == "True", int(r["burst_index"]), r["defense"])
                c = clean.get(key)
                if c is None:
                    continue
                if r["defense"] == "none":
                    cross["clean_pred_compared"] += 1
                    cross["clean_pred_identical"] += int(int(r["clean_pred"]) == c["clean_pred_none"])
                g = groups.setdefault((r["attack"], r["profile"], r["effective_fec"] == "True", r["defense"]),
                                      Counter())
                if r["clean_correct"] != "True":
                    continue
                g["n_clean_correct"] += 1
                if r["attack_success_after_defense"] == "True":
                    g["n_success_after"] += 1
                    g["n_success_after_and_clean_defense_wrong"] += int(not c["clean_correct_after_defense"])
                if r["attack_success_before_defense"] == "True":
                    g["n_success_before"] += 1
                g["n_clean_defense_wrong"] += int(not c["clean_correct_after_defense"])
        for (atk, prof, fec, dname), g in sorted(groups.items()):
            n = g["n_clean_correct"]
            decomp.append({"attack": atk, "profile": prof, "effective_fec": fec, "defense": dname,
                           "n_clean_correct": n,
                           "asr_before_defense_pct": 100 * g["n_success_before"] / n if n else None,
                           "asr_after_defense_pct": 100 * g["n_success_after"] / n if n else None,
                           "clean_defense_misclassification_pct_of_clean_correct":
                               100 * g["n_clean_defense_wrong"] / n if n else None,
                           "n_success_after": g["n_success_after"],
                           "n_success_after_where_clean_defense_also_wrong": g["n_success_after_and_clean_defense_wrong"],
                           "share_of_success_after_explained_by_clean_defense_pct":
                               100 * g["n_success_after_and_clean_defense_wrong"] / g["n_success_after"]
                               if g["n_success_after"] else None,
                           "note": "explained = the same defense on the clean burst already misclassifies it"})
    amc_study.write_csv(out / "attack_run_decomposition.csv", decomp)

    # ---------------------------------------------------------------- validation
    checks["completed"] = state["completed"]
    checks["defense_fallback_zero"] = counters["defense_fallback"] == 0 and all(
        r["defense_backend"] in ("none", tka._REAL_SOURCE, tka._ADAPTIVE_SOURCE) for r in rows)
    checks["adaptive_k_production_and_trace_identical"] = counters["adaptive_trace_mismatch"] == 0 and all(
        r["defense_backend"] == tka._ADAPTIVE_SOURCE for r in rows if r["defense"] == "adaptive_k")
    checks["fixed_k_matches_selected_k"] = counters["fixed_k_mismatch"] == 0 and all(
        r["selected_k"] == int(r["defense"][4:]) for r in rows if r["defense"].startswith("topk"))
    by_burst = group(("true_modulation", "fec_requested", "burst_index"), rows)
    checks["same_clean_waveform_across_defenses"] = counters["clean_hash_mismatch"] == 0 and all(
        len({r["clean_input_sha256"] for r in g}) == 1 and all(r["defense_input_sha256"] == r["clean_input_sha256"]
                                                              for r in g)
        and sorted(r["defense"] for r in g) == sorted(DEFENSES) for g in by_burst.values())
    checks["demod_status_allowed_only"] = all(r[f"{s}_demod_status"] in dstudy.ALLOWED_DEMOD_STATUS
                                              for r in rows for s in ("oracle", "predicted"))
    checks["crc_pass_is_bool"] = all(isinstance(r[f"{s}_crc_pass"], (bool, np.bool_))
                                     for r in rows for s in ("oracle", "predicted"))
    checks["no_nan_inf"] = all(r["defense_latency_ms"] is None or math.isfinite(r["defense_latency_ms"]) for r in rows)
    checks["payload_degenerate_excluded_from_aggregates"] = all(s["n_degenerate_included"] == 0 for s in summ)
    expected = n_bursts_total * len(DEFENSES)
    checks["matrix_complete"] = bool(rows) and len(rows) == expected and len(by_burst) == n_bursts_total
    gated = ["completed", "defense_fallback_zero", "adaptive_k_production_and_trace_identical",
             "fixed_k_matches_selected_k", "same_clean_waveform_across_defenses", "demod_status_allowed_only",
             "crc_pass_is_bool", "no_nan_inf", "payload_degenerate_excluded_from_aggregates", "matrix_complete",
             "adversarial_rf_at_pin", "awn_submodule_at_pin", "checkpoint_hash_ok", "real_backends_loaded",
             "adaptive_is_production_function"]
    failed = [k for k in gated if not checks.get(k)]
    validation = {"overall": "PASS" if not failed else "FAIL", "failed_checks": failed, "checks": checks,
                  "counters": dict(counters), "n_rows": len(rows), "n_valid_rows": len(valid),
                  "n_degenerate_rows": len(rows) - len(valid), "expected_rows": expected, "n_bursts": n_bursts_total,
                  "informational": {"attack_run_clean_pred_reproduction": cross, "attack_run": ATTACK_DEFENSE_RUN},
                  "error": state.get("error")}
    (out / "validation.json").write_text(json.dumps(validation, indent=2, default=str))
    manifest = {
        "experiment": "clean_crc_fec_defense_vm", "mode": "smoke" if args.smoke else "full",
        "command": [sys.executable] + sys.argv, "args": vars(args), "elapsed_s": time.monotonic() - t_start,
        "script_sha256": amc_study.sha256_file(Path(__file__).resolve()),
        "git": state.get("git"), "environment": state.get("environment"),
        "checkpoint_sha256": state.get("checkpoint_sha256"),
        "domain": "synthetic TX/RX (payload bits known); NOT RadioML2016; no adversarial attack generated",
        "burst_params": stx.LEGACY_CRC_PIPELINE_BURST, "soft_demod": True,
        "defenses": {"none": "clean window unchanged",
                     **{f"topk{k}": f"TopKAdapter(mode='fixed').apply(x, topk={k}) -> {tka._REAL_SOURCE}"
                        for k in dstudy.FIXED_KS},
                     "adaptive_k": f"TopKAdapter(mode='adaptive').apply(x) -> {tka._ADAPTIVE_SOURCE} (defaults)"},
        "adaptive_k_default_parameters": defaults, "adaptive_k_snr_source": dstudy.SNR_SOURCE,
        "seed_rule": "burst: derive_seed(42, mod, snr, i, 'synth_txrx|fec<0|1>'), same bursts as the attack run",
        "terminology": "defense-induced misclassification = clean-correct burst misclassified after defense; "
                       "not attack success",
        "degenerate_frames": "QPSK/PAM4 with FEC: 0 payload bits; kept in raw, excluded from all summaries",
        "latency": {"defense": "TopKAdapter.apply() wall time (perf_counter_ns); stdout to null sink",
                    "platform": "x86 VM latency: VMware VM, i7-11700, torch intra 1 / interop 1"},
        "decomposition_source": ATTACK_DEFENSE_RUN, "validation_overall": validation["overall"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log(f"validation {validation['overall']} failed={failed} ({time.monotonic() - t_start:.0f} s)", fh)
    fh.close()
    sys.exit(0 if validation["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
