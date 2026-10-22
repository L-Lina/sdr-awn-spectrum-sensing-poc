"""
Reduced-iteration attacks x post-attack defense x AMC x CRC x FEC (synthetic TX/RX domain, x86 VM, CPU, B=1).
EXPERIMENT ONLY -- production attack / defense / synth_txrx code is not modified.

Domain and seeds: identical to experiments/benchmark_reduced_iteration_attacks_crc_fec_vm.py (SNR 18 dB, 8 digital
modulations x FEC requested {off, on} x 25 bursts; burst seed derive_seed(42, mod, snr, i, 'synth_txrx|fec<0|1>');
torch.manual_seed(burst seed) before the attack; attack seed derive_seed(..., '<attack>|<profile>|fec<0|1>')).
Attack profile ids are those of that run, so x_adv is regenerated under the same seeds.

Per burst x profile (post-attack defense evaluation; the attack is NOT defense-aware):
  1. clean window -> AWN (clean_pred), clean oracle / predicted demod + CRC
  2. production AttackAdapter generates x_adv ONCE (attack-generation latency timed)
  3. the same x_adv (SHA-256 recorded and re-checked per branch) goes to each defense:
       none | TopKAdapter(fixed) K=10,20,30,40 | TopKAdapter(adaptive) = adaptive_k_v2_snr_defense (defaults)
     (defense latency timed around TopKAdapter.apply only)
  4. per defense output: AWN (defended_pred), oracle demod (true modulation) and predicted-AMC demod (defended_pred),
     FEC decode when the burst is coded, CRC-8 check, bit / frame correctness.

Definitions (true_label = AWN class of the transmitted modulation):
  attack_success_before_defense = clean_correct AND attacked_pred_before_defense != true_label
  attack_success_after_defense  = clean_correct AND defended_pred != true_label
  defense_recovery              = attack_success_before_defense AND defended_pred == true_label
  conditional ASR before / after = n_success_before / n_clean_correct, n_success_after / n_clean_correct
  recovery_rate                 = n_recovery / n_success_before
For 'none', defended_pred = attacked_pred_before_defense.

Adaptive-K trace: adaptive_k_v2_snr_defense does not return its per-sample decisions. An untimed trace re-runs the
production helper functions (_shared_fft_and_route, _estimate_snr_from_fft, _knee_from_fft,
_apply_per_sample_topk, with the production function's own default parameters read via inspect.signature) on the
same input and records route (wideband_quantize / narrowband_topk), SNR estimate, k_max, uncapped knee and selected
K. The trace output must be bit-identical to the TopKAdapter output for every sample, or the run fails.
SNR source: blind estimate from the defense input waveform itself (top-pilot_k PSD bins / remaining bins of its own
FFT); no experiment metadata or injected SNR is used. Wideband-routed samples get no SNR estimate and no K.

Payload-degenerate frames (QPSK, PAM4 with FEC: 0 payload bits) are kept in raw_samples.csv and excluded from every
summary. BPSK / CPFSK / GFSK never code (effective_fec=False) as in the legacy chain.

Run from the repo root:
    ~/adversarial-rf/.venv/bin/python experiments/benchmark_reduced_iteration_crc_fec_defense_vm.py \
        --checkpoint ~/adversarial-rf/2016.10a_AWN.pkl [--smoke]
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as _dt
import inspect
import io
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

import benchmark_reduced_iteration_attacks_vm as amc_study  # noqa: E402 -- profiles, env metadata, helpers
import run_cpu_full_matrix as formal  # noqa: E402 -- derive_seed, sha256_array
from src.adapters import synth_txrx_adapter as stx  # noqa: E402
from src.utils import synth_crc_pipeline as scp  # noqa: E402

BASE_SEED = 42
EXPECTED_CHECKPOINT_SHA256 = amc_study.EXPECTED_CHECKPOINT_SHA256
EXPECTED_ADV_RF_COMMIT = amc_study.EXPECTED_ADV_RF_COMMIT
EXPECTED_AWN_COMMIT = "0b3d2b2d3d5d4761e73eb91e223844b4b8585b53"
PRIOR_CRC_RUN = "results/reduced_iteration_crc_fec_vm_20261022_190944"
FIXED_KS = (10, 20, 30, 40)
DEFENSES = ("none",) + tuple(f"topk{k}" for k in FIXED_KS) + ("adaptive_k",)
ALLOWED_DEMOD_STATUS = {stx.DEMOD_OK, stx.DEMOD_NOT_DIGITAL, stx.DEMOD_FAMILY_MISMATCH}
SNR_SOURCE = ("blind waveform estimate: production _estimate_snr_from_fft on the defense input's own FFT "
              "(sum of top-pilot_k PSD bins / sum of remaining bins, linear); no experiment metadata or injected SNR")


class _Null:
    def write(self, _s):
        return 0

    def flush(self):
        pass


def log(msg: str, fh) -> None:
    line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, file=sys.stderr, flush=True)
    fh.write(line + "\n")
    fh.flush()


def profiles() -> list:
    out = list(amc_study.fgsm_profiles())
    out += [p for p in amc_study.pgd_profiles() if p["steps"] in (5, 7, 10)]
    out += [p for p in amc_study.cw_profiles() if p["steps"] in (10, 15, 20)]
    out += [amc_study.ead_profile("ead", 5, 10, False), amc_study.ead_profile("ead", 9, 100, True)]
    return out


def attack_kwargs(prof: dict) -> dict:
    return {"cw_c": prof["cw_c"], "cw_steps": prof["cw_steps"], "cw_lr": prof["cw_lr"],
            "attack_params": dict(prof["attack_params"])}


# ---------------------------------------------------------------------------- defense helpers
def adaptive_trace(D, defaults: dict, x: np.ndarray):
    """Re-run the production adaptive_k_v2_snr_defense body with its production helpers to expose decisions."""
    import torch
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    with torch.no_grad():
        result, X, nb = D._shared_fft_and_route(xt, defaults["flatness_threshold"], defaults["quant_levels"])
        info = [{"route": "wideband_quantize", "snr_value": None, "k_max": None, "knee_uncapped": None,
                 "selected_k": None} for _ in range(x.shape[0])]
        if nb.any():
            X_n = X[nb]
            snr = D._estimate_snr_from_fft(X_n, defaults["pilot_k"])
            t = ((snr - defaults["snr_low"]) / (defaults["snr_high"] - defaults["snr_low"] + 1e-12)).clamp(0.0, 1.0)
            kmax = (defaults["k_max_low_snr"] + t * (defaults["k_max_high_snr"] - defaults["k_max_low_snr"])).round().int()
            knee = D._knee_from_fft(X_n, defaults["ratio_thresh"])
            k = torch.min(knee, kmax)
            result[nb] = D._apply_per_sample_topk(X_n, k)
            for j, idx in enumerate(torch.nonzero(nb).flatten().tolist()):
                info[idx] = {"route": "narrowband_topk", "snr_value": float(snr[j]), "k_max": int(kmax[j]),
                             "knee_uncapped": int(knee[j]), "selected_k": int(k[j])}
    return result.numpy().astype(np.float32), info


def receive(burst, awn, x_win, quiet=True):
    """AMC on a window + oracle / predicted demod + CRC; returns (pred, dict)."""
    pred, _ = scp._infer(awn, x_win, "defended", quiet)
    mod = burst["mod_type"]
    frame = burst["frame_bits"].astype(np.uint8)
    out = {}
    for name, dm in (("oracle", mod), ("predicted", scp.IDX_TO_MOD[pred])):
        d = stx.demodulate(burst, dm, x_win, True)
        rec = np.asarray(d["recovered_bits"], dtype=np.uint8)
        be = stx.bit_errors(frame, rec)
        if rec.size == 0:
            fbe, ber = None, None
        else:
            n = min(rec.size, frame.size)
            fbe = int(np.count_nonzero(frame[:n] != rec[:n]) + max(0, frame.size - rec.size))
            ber = fbe / frame.size
        out.update({f"{name}_demod_mod": dm, f"{name}_demod_status": d["demod_status"],
                    f"{name}_crc_pass": d["crc_pass"], f"{name}_recovered_bit_count": int(rec.size),
                    f"{name}_bit_error_count": fbe, f"{name}_ber": ber,
                    f"{name}_frame_correct": bool(d["crc_pass"] and be == 0),
                    f"{name}_crc_false_accept": bool(d["crc_pass"] and be not in (0, None))})
    return pred, out


# ---------------------------------------------------------------------------- summaries
def pct(num, den, name, reasons):
    if den == 0:
        reasons.append(f"{name}: denominator 0")
        return None
    return 100.0 * num / den


def mean_or_none(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def summarize(rr: list) -> dict:
    reasons = []
    cc = [r for r in rr if r["clean_correct"]]
    sb = [r for r in rr if r["attack_success_before_defense"]]
    sa = [r for r in rr if r["attack_success_after_defense"]]
    rec = [r for r in rr if r["defense_recovery"]]
    dl = [r["defense_latency_ms"] for r in rr if r["defense_latency_ms"] is not None]
    al = amc_study.dist([r["attack_latency_ms"] for r in rr])
    comp = [r["attack_plus_defense_latency_ms_derived"] for r in rr]
    dld = amc_study.dist(dl)
    return {
        "n_total": len(rr), "n_clean_correct": len(cc),
        "n_attack_success_before_defense": len(sb), "n_attack_success_after_defense": len(sa),
        "n_defense_recovery": len(rec),
        "conditional_asr_before_defense_pct": pct(len(sb), len(cc), "asr_before", reasons),
        "conditional_asr_after_defense_pct": pct(len(sa), len(cc), "asr_after", reasons),
        "recovery_rate_pct": pct(len(rec), len(sb), "recovery_rate", reasons),
        "oracle_crc_pass_pct": pct(sum(r["oracle_crc_pass"] for r in rr), len(rr), "oracle_crc", reasons),
        "oracle_frame_correct_pct": pct(sum(r["oracle_frame_correct"] for r in rr), len(rr), "oracle_fc", reasons),
        "predicted_crc_pass_pct": pct(sum(r["predicted_crc_pass"] for r in rr), len(rr), "pred_crc", reasons),
        "predicted_frame_correct_pct": pct(sum(r["predicted_frame_correct"] for r in rr), len(rr), "pred_fc", reasons),
        "n_success_before_oracle_crc_pass": sum(r["oracle_crc_pass"] for r in sb),
        "oracle_crc_given_attack_success_before_pct": pct(sum(r["oracle_crc_pass"] for r in sb), len(sb),
                                                          "oracle_crc|success_before", reasons),
        "oracle_frame_correct_given_attack_success_before_pct": pct(sum(r["oracle_frame_correct"] for r in sb),
                                                                    len(sb), "oracle_fc|success_before", reasons),
        "oracle_crc_given_attack_success_after_pct": pct(sum(r["oracle_crc_pass"] for r in sa), len(sa),
                                                         "oracle_crc|success_after", reasons),
        "oracle_frame_correct_given_attack_success_after_pct": pct(sum(r["oracle_frame_correct"] for r in sa),
                                                                   len(sa), "oracle_fc|success_after", reasons),
        "predicted_crc_given_attack_success_after_pct": pct(sum(r["predicted_crc_pass"] for r in sa), len(sa),
                                                            "pred_crc|success_after", reasons),
        "predicted_frame_correct_given_attack_success_after_pct": pct(sum(r["predicted_frame_correct"] for r in sa),
                                                                      len(sa), "pred_fc|success_after", reasons),
        "predicted_frame_correct_given_recovery_pct": pct(sum(r["predicted_frame_correct"] for r in rec), len(rec),
                                                          "pred_fc|recovery", reasons),
        "oracle_ber_mean": mean_or_none([r["oracle_ber"] for r in rr]),
        "predicted_ber_mean": mean_or_none([r["predicted_ber"] for r in rr]),
        "oracle_crc_false_accept_n": sum(r["oracle_crc_false_accept"] for r in rr),
        "predicted_crc_false_accept_n": sum(r["predicted_crc_false_accept"] for r in rr),
        "attack_linf_median": amc_study.dist([r["attack_linf"] for r in rr])["median"],
        "attack_l2_median": amc_study.dist([r["attack_l2"] for r in rr])["median"],
        "attack_latency_median_ms": al["median"], "attack_latency_mean_ms": al["mean"], "attack_latency_p95_ms": al["p95"],
        "defense_latency_median_ms": dld["median"] if dl else None, "defense_latency_mean_ms": dld["mean"] if dl else None,
        "defense_latency_p95_ms": dld["p95"] if dl else None,
        "attack_plus_defense_latency_median_ms_derived": float(np.median(comp)),
        "adaptive_selected_k_mean": mean_or_none([r["adaptive_selected_k"] for r in rr]),
        "adaptive_n_wideband": sum(r["adaptive_route"] == "wideband_quantize" for r in rr),
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
    ap.add_argument("--smoke", action="store_true", help="1 burst/cell; FGSM eps0.03, PGD steps5, CW steps10, EAD 5x10")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--prior-crc-run", default=PRIOR_CRC_RUN,
                    help="reduced-iteration CRC/FEC run used for the informational attack-reproduction check")
    ap.add_argument("--save-x-adv", action="store_true", help="write x_adv arrays (x_adv.npz) after the run")
    ap.add_argument("--legacy-backend-toggle", action="store_true",
                    help="comparison only: reproduce the superseded run that passed both mkldnn-off opt-ins")
    args = ap.parse_args()

    import torch
    import torchattacks as ta
    torch.set_num_interop_threads(args.interop_threads)
    torch.set_num_threads(args.torch_threads)
    from src.adapters import topk_adapter as tka
    from src.adapters.attack_adapter import AttackAdapter
    from src.adapters.awn_adapter import AWNModelAdapter
    import util.defense as D  # same module object TopKAdapter imported from (sys.path set by topk_adapter)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (REPO_ROOT / (args.out_dir or
                        f"results/{amc_study.run_dir_name('reduced_iteration_crc_fec_defense_vm', args)}")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    fh = (out / "run.log").open("w")
    t_start = time.monotonic()
    checks, state = {}, {"completed": False}
    rows = []
    counters = Counter()
    profs = profiles()
    if args.smoke:
        keep = {"eps0.03", "eps0.03_steps5", "c1_steps10_lr0.01", "bss5_maxit10"}
        profs = [p for p in profs if p["profile"] in keep]
    n_bursts_total = 0
    adaptive_defaults = {}
    try:
        stx.require_backend()
        state["git"] = amc_study.git_metadata()
        state["environment"] = amc_study.environment_metadata(torch, ta)
        checks["adversarial_rf_at_pin"] = state["git"]["adversarial_rf_head"] == EXPECTED_ADV_RF_COMMIT
        checks["awn_submodule_at_pin"] = EXPECTED_AWN_COMMIT in (state["git"]["submodule_status"] or "")
        ckpt = Path(args.checkpoint).expanduser().resolve()
        state["checkpoint_sha256"] = amc_study.sha256_file(ckpt)
        checks["checkpoint_hash_ok"] = state["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
        if not all(checks.values()):
            raise RuntimeError(f"preflight failed: {checks}")

        with contextlib.redirect_stdout(_Null()):
            awn = AWNModelAdapter(str(ckpt))
            adapter = AttackAdapter(awn_model=awn.model, **amc_study.backend_toggle_kwargs(args.legacy_backend_toggle))
            topk_fixed = tka.TopKAdapter(mode="fixed")
            topk_adapt = tka.TopKAdapter(mode="adaptive")
        checks["real_backends_loaded"] = (awn.status == "ok" and adapter.status == "ok" and topk_fixed.backend_available
                                          and topk_fixed.backend_name == tka._REAL_SOURCE
                                          and topk_adapt.backend_name == tka._ADAPTIVE_SOURCE)
        checks["backend_toggle_matches_configuration"] = amc_study.backend_toggle_ok(args.legacy_backend_toggle, adapter)
        state["backend_toggle"] = amc_study.backend_toggle_manifest(args.legacy_backend_toggle, adapter)
        checks["adaptive_is_production_function"] = (tka._real_adaptive_k_v2_snr_defense is D.adaptive_k_v2_snr_defense
                                                     and tka._real_fft_topk_denoise is D.fft_topk_denoise)
        if not (checks["real_backends_loaded"] and checks["adaptive_is_production_function"]):
            raise RuntimeError(f"defense/attack backend check failed: {checks}")
        adaptive_defaults = {k: v.default for k, v in inspect.signature(D.adaptive_k_v2_snr_defense).parameters.items()
                             if k != "x"}

        mods = stx.digital_modulations()
        n_b = 1 if args.smoke else args.bursts_per_cell
        bursts = []
        for mod in mods:
            for fec in (False, True):
                for i in range(n_b):
                    seed = formal.derive_seed(BASE_SEED, mod, args.snr, i, f"synth_txrx|fec{int(fec)}")
                    bursts.append((mod, fec, i, seed, stx.generate_frame(mod, args.snr, seed, fec)))
        n_bursts_total = len(bursts)
        log(f"{len(bursts)} bursts x {len(profs)} profiles x {len(DEFENSES)} defenses; out={out}", fh)

        def defend(name: str, x_adv: np.ndarray):
            """Returns (x_def, latency_ms or None, info dict)."""
            if name == "none":
                return x_adv, None, {"defense_backend": "none", "selected_k": None}
            if name.startswith("topk"):
                k = int(name[4:])
                with contextlib.redirect_stdout(_Null()):
                    t0 = time.perf_counter_ns()
                    y, meta = topk_fixed.apply(x_adv, topk=k)
                    dt = (time.perf_counter_ns() - t0) / 1e6
                if meta.get("topk_status") != "ok" or meta.get("topk_backend") != tka._REAL_SOURCE:
                    counters["defense_fallback"] += 1
                    raise RuntimeError(f"fixed Top-K fallback: {meta}")
                direct = D.fft_topk_denoise(torch.from_numpy(x_adv), topk=k).numpy().astype(np.float32)
                if not np.array_equal(direct, y):
                    counters["fixed_k_mismatch"] += 1
                    raise RuntimeError(f"TopKAdapter K={k} output != production fft_topk_denoise(K={k})")
                with torch.no_grad():  # effective K = number of retained FFT bins per channel
                    X = torch.fft.fft(torch.from_numpy(x_adv), dim=2)
                    _, idx = X.abs().topk(k=min(k, X.shape[2]), dim=2)
                eff = int(idx.shape[2])
                if eff != k:
                    counters["fixed_k_mismatch"] += 1
                    raise RuntimeError(f"effective K {eff} != requested {k}")
                return y, dt, {"defense_backend": meta["topk_backend"], "selected_k": eff}
            with contextlib.redirect_stdout(_Null()):
                t0 = time.perf_counter_ns()
                y, meta = topk_adapt.apply(x_adv)
                dt = (time.perf_counter_ns() - t0) / 1e6
            if meta.get("topk_status") != "ok" or meta.get("topk_backend") != tka._ADAPTIVE_SOURCE:
                counters["defense_fallback"] += 1
                raise RuntimeError(f"Adaptive-K not on production backend: {meta}")
            yt, info = adaptive_trace(D, adaptive_defaults, x_adv)
            if not np.array_equal(yt, y):
                counters["adaptive_trace_mismatch"] += 1
                raise RuntimeError("Adaptive-K trace does not reproduce the production output")
            i0 = info[0]
            return y, dt, {"defense_backend": meta["topk_backend"], "selected_k": i0["selected_k"],
                           "adaptive_route": i0["route"], "adaptive_snr_value": i0["snr_value"],
                           "adaptive_k_max": i0["k_max"], "adaptive_knee_uncapped": i0["knee_uncapped"]}

        x_store = {}
        for prof in profs:
            t0p = time.monotonic()
            kw = attack_kwargs(prof)
            for w in range(args.warmup):  # untimed
                _, _, _, seed, b = bursts[w % len(bursts)]
                scp.run_burst_chain(b, awn=awn, attack_adapter=adapter, attack=prof["attack"], eps=prof["eps"],
                                    torch_seed=seed, attack_seed=0, attack_kwargs=kw)
            for mod, fec, i, seed, b in bursts:
                a_seed = formal.derive_seed(BASE_SEED, mod, args.snr, i,
                                            f"{prof['attack']}|{prof['profile']}|fec{int(fec)}")
                base_row, x_adv = scp.run_burst_chain(b, awn=awn, attack_adapter=adapter, attack=prof["attack"],
                                                      eps=prof["eps"], torch_seed=seed, attack_seed=a_seed,
                                                      attack_kwargs=kw)
                x_adv = np.ascontiguousarray(x_adv, dtype=np.float32)
                x_adv.setflags(write=False)
                adv_sha = formal.sha256_array(x_adv)
                if args.save_x_adv:
                    x_store[f"{prof['attack']}|{prof['profile']}|{mod}|{args.snr}|fec{int(fec)}|{i}"] = np.array(x_adv)
                label = base_row["true_label"]
                for dname in DEFENSES:
                    in_sha = formal.sha256_array(x_adv)
                    if in_sha != adv_sha:
                        counters["xadv_hash_mismatch"] += 1
                        raise RuntimeError("x_adv changed between defense branches")
                    x_def, dlat, dinfo = defend(dname, x_adv)
                    if not np.isfinite(x_def).all() or x_def.shape != x_adv.shape:
                        raise RuntimeError(f"invalid defended window ({dname})")
                    if dname == "none":
                        dpred = base_row["attacked_pred"]
                        _, recv = receive(b, awn, x_def)
                        if _ != dpred:
                            raise RuntimeError("none-branch prediction differs from attacked prediction")
                    else:
                        dpred, recv = receive(b, awn, x_def)
                    cc = bool(base_row["clean_correct"])
                    sb = bool(cc and base_row["attacked_pred"] != label)
                    sa = bool(cc and dpred != label)
                    row = {
                        "attack": prof["attack"], "profile": prof["profile"], "is_baseline_profile": prof["is_baseline"],
                        "eps": prof["eps"], "alpha": prof["alpha"], "steps": prof["steps"],
                        "binary_search_steps": prof["binary_search_steps"], "max_iterations": prof["max_iterations"],
                        "defense": dname, "true_modulation": mod, "true_label": label, "snr_db": args.snr,
                        "burst_index": i, "burst_seed": seed, "attack_seed": a_seed,
                        "fec_requested": fec, "effective_fec": base_row["fec_enabled"],
                        "payload_degenerate": base_row["n_data_bits"] == 0, "n_data_bits": base_row["n_data_bits"],
                        "n_frame_bits": base_row["n_frame_bits"], "soft_demod": True,
                        "clean_pred": base_row["clean_pred"], "clean_pred_mod": base_row["clean_pred_mod"],
                        "clean_correct": cc,
                        "attacked_pred_before_defense": base_row["attacked_pred"],
                        "attacked_pred_before_defense_mod": base_row["attacked_pred_mod"],
                        "defended_pred": dpred, "defended_pred_mod": scp.IDX_TO_MOD[dpred],
                        "attack_success_before_defense": sb, "attack_success_after_defense": sa,
                        "defense_recovery": bool(sb and dpred == label),
                        "clean_oracle_crc_pass": base_row["clean_oracle_crc_pass"],
                        "clean_predicted_crc_pass": base_row["clean_amc_crc_pass"],
                        **recv,
                        "attack_linf": base_row["linf"], "attack_l2": base_row["l2"],
                        "x_adv_sha256": adv_sha, "defense_input_sha256": in_sha,
                        "attack_status": base_row["attack_status"], "attack_backend": base_row["attack_backend"],
                        "attack_latency_ms": base_row["attack_latency_ms"],
                        "defense_backend": dinfo["defense_backend"], "defense_latency_ms": dlat,
                        "attack_plus_defense_latency_ms_derived": base_row["attack_latency_ms"] + (dlat or 0.0),
                        "selected_k": dinfo["selected_k"],
                        "adaptive_route": dinfo.get("adaptive_route"),
                        "adaptive_selected_k": dinfo["selected_k"] if dname == "adaptive_k" else None,
                        "adaptive_snr_value": dinfo.get("adaptive_snr_value"),
                        "adaptive_snr_db": (10 * math.log10(dinfo["adaptive_snr_value"])
                                            if dinfo.get("adaptive_snr_value") else None),
                        "adaptive_snr_source": SNR_SOURCE if dname == "adaptive_k" else None,
                        "adaptive_k_max": dinfo.get("adaptive_k_max"),
                        "adaptive_knee_uncapped": dinfo.get("adaptive_knee_uncapped"),
                    }
                    rows.append(row)
            log(f"{prof['attack']}/{prof['profile']}: {len(bursts)} bursts x {len(DEFENSES)} defenses "
                f"({time.monotonic() - t0p:.0f} s)", fh)
        if args.save_x_adv:
            amc_study.save_x_adv(out / "x_adv.npz", x_store)
        state["completed"] = True
    except Exception as exc:  # noqa: BLE001 -- recorded; validation FAILs
        state["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        log(f"ERROR: {state['error']}", fh)

    # ------------------------------------------------------------------ outputs
    amc_study.write_csv(out / "raw_samples.csv", rows)
    valid = [r for r in rows if not r["payload_degenerate"]]

    def group(keys, src):
        g = {}
        for r in src:
            g.setdefault(tuple(r[k] for k in keys), []).append(r)
        return g

    pkeys = ("attack", "profile", "effective_fec", "defense")
    prof_rows = []
    for k, rr in group(pkeys, valid).items():
        prof_rows.append({**dict(zip(pkeys, k)), "is_baseline_profile": rr[0]["is_baseline_profile"],
                          "steps": rr[0]["steps"], "binary_search_steps": rr[0]["binary_search_steps"],
                          "max_iterations": rr[0]["max_iterations"], "eps": rr[0]["eps"], **summarize(rr)})
    amc_study.write_csv(out / "profile_summary.csv", prof_rows)

    # defense effect vs 'none' of the same attack/profile/FEC
    idx = {(r["attack"], r["profile"], r["effective_fec"], r["defense"]): r for r in prof_rows}
    def_rows = []
    for r in prof_rows:
        n = idx[(r["attack"], r["profile"], r["effective_fec"], "none")]
        dif = lambda a, b: None if (a is None or b is None) else a - b  # noqa: E731
        def_rows.append({**{k: r[k] for k in ("attack", "profile", "effective_fec", "defense", "n_total", "n_clean_correct",
                                           "n_attack_success_before_defense", "n_attack_success_after_defense",
                                           "n_defense_recovery", "conditional_asr_before_defense_pct",
                                           "conditional_asr_after_defense_pct", "recovery_rate_pct",
                                           "oracle_crc_given_attack_success_before_pct",
                                           "oracle_frame_correct_given_attack_success_before_pct",
                                           "oracle_crc_pass_pct", "predicted_crc_pass_pct",
                                           "predicted_frame_correct_pct", "predicted_frame_correct_given_recovery_pct",
                                           "defense_latency_median_ms", "attack_latency_median_ms",
                                           "attack_plus_defense_latency_median_ms_derived",
                                           "adaptive_selected_k_mean", "na_reasons")},
                         "asr_reduction_pp_vs_none": dif(n["conditional_asr_after_defense_pct"],
                                                         r["conditional_asr_after_defense_pct"]),
                         "oracle_crc_given_success_before_change_pp_vs_none":
                             dif(r["oracle_crc_given_attack_success_before_pct"],
                                 n["oracle_crc_given_attack_success_before_pct"]),
                         "predicted_frame_correct_change_pp_vs_none": dif(r["predicted_frame_correct_pct"],
                                                                          n["predicted_frame_correct_pct"])})
    amc_study.write_csv(out / "defense_summary.csv", def_rows)

    mkeys = ("attack", "profile", "effective_fec", "true_modulation", "defense")
    amc_study.write_csv(out / "modulation_summary.csv",
                        [{**dict(zip(mkeys, k)), **summarize(rr)} for k, rr in group(mkeys, valid).items()])

    ak = [r for r in valid if r["defense"] == "adaptive_k"]
    ak_rows = []
    for keys in (("attack", "profile", "effective_fec"), ("attack", "effective_fec"), ("effective_fec",),
                 ("true_modulation", "effective_fec")):
        for k, rr in group(keys, ak).items():
            ks = [r["adaptive_selected_k"] for r in rr if r["adaptive_selected_k"] is not None]
            snrs = [r["adaptive_snr_db"] for r in rr if r["adaptive_snr_db"] is not None]
            hist = Counter(ks)
            ak_rows.append({"grouping": "+".join(keys), **{kk: vv for kk, vv in zip(keys, k)}, "n": len(rr),
                            "n_wideband_quantize": sum(r["adaptive_route"] == "wideband_quantize" for r in rr),
                            "n_narrowband_topk": len(ks),
                            "selected_k_min": min(ks) if ks else None, "selected_k_median": float(np.median(ks)) if ks else None,
                            "selected_k_mean": float(np.mean(ks)) if ks else None, "selected_k_max": max(ks) if ks else None,
                            "n_k_capped": sum(r["adaptive_knee_uncapped"] is not None and
                                              r["adaptive_knee_uncapped"] > r["adaptive_k_max"] for r in rr),
                            "selected_k_histogram": json.dumps({str(x): hist[x] for x in sorted(hist)}),
                            "snr_est_db_median": float(np.median(snrs)) if snrs else None,
                            "snr_est_db_min": min(snrs) if snrs else None, "snr_est_db_max": max(snrs) if snrs else None,
                            "snr_source": SNR_SOURCE})
    amc_study.write_csv(out / "adaptive_k_selection_summary.csv", ak_rows)

    lat_rows = []
    for k, rr in group(("attack", "profile", "effective_fec", "defense"), valid).items():
        a = amc_study.dist([r["attack_latency_ms"] for r in rr])
        dl = [r["defense_latency_ms"] for r in rr if r["defense_latency_ms"] is not None]
        d = amc_study.dist(dl)
        c = amc_study.dist([r["attack_plus_defense_latency_ms_derived"] for r in rr])
        lat_rows.append({"scope": "per_profile", "attack": k[0], "profile": k[1], "effective_fec": k[2], "defense": k[3],
                         "n": len(rr), "attack_latency_median_ms": a["median"], "attack_latency_mean_ms": a["mean"],
                         "attack_latency_p95_ms": a["p95"],
                         "defense_latency_median_ms": d["median"] if dl else None,
                         "defense_latency_mean_ms": d["mean"] if dl else None,
                         "defense_latency_p95_ms": d["p95"] if dl else None,
                         "attack_plus_defense_latency_median_ms_derived": c["median"],
                         "attack_plus_defense_latency_p95_ms_derived": c["p95"]})
    for dname in DEFENSES[1:]:
        dl = [r["defense_latency_ms"] for r in valid if r["defense"] == dname]
        d = amc_study.dist(dl)
        lat_rows.append({"scope": "pooled_all_attack_inputs", "defense": dname, "n": len(dl),
                         "defense_latency_median_ms": d["median"], "defense_latency_mean_ms": d["mean"],
                         "defense_latency_p95_ms": d["p95"]})
    amc_study.write_csv(out / "latency_summary.csv", lat_rows)

    # ------------------------------------------------------------------ validation
    checks["completed"] = state["completed"]
    checks["attack_fallback_zero"] = bool(rows) and all(r["attack_status"] == "ok" for r in rows)
    checks["defense_fallback_zero"] = counters["defense_fallback"] == 0 and all(
        r["defense_backend"] in ("none", tka._REAL_SOURCE, tka._ADAPTIVE_SOURCE) for r in rows)
    checks["adaptive_k_production_and_trace_identical"] = (counters["adaptive_trace_mismatch"] == 0 and all(
        r["defense_backend"] == tka._ADAPTIVE_SOURCE for r in rows if r["defense"] == "adaptive_k"))
    checks["fixed_k_matches_selected_k"] = counters["fixed_k_mismatch"] == 0 and all(
        r["selected_k"] == int(r["defense"][4:]) for r in rows if r["defense"].startswith("topk"))
    by_burst = {}
    for r in rows:
        by_burst.setdefault((r["attack"], r["profile"], r["true_modulation"], r["fec_requested"], r["burst_index"]),
                            []).append(r)
    checks["same_x_adv_across_defenses"] = counters["xadv_hash_mismatch"] == 0 and all(
        len({r["x_adv_sha256"] for r in g}) == 1 and all(r["defense_input_sha256"] == r["x_adv_sha256"] for r in g)
        and sorted(r["defense"] for r in g) == sorted(DEFENSES) for g in by_burst.values())
    checks["demod_status_allowed_only"] = all(r[f"{s}_demod_status"] in ALLOWED_DEMOD_STATUS
                                              for r in rows for s in ("oracle", "predicted"))
    checks["crc_pass_is_bool"] = all(isinstance(r[f"{s}_crc_pass"], (bool, np.bool_))
                                     for r in rows for s in ("oracle", "predicted"))
    checks["no_nan_inf"] = all(math.isfinite(r["attack_linf"]) and math.isfinite(r["attack_latency_ms"]) and
                               (r["defense_latency_ms"] is None or math.isfinite(r["defense_latency_ms"])) for r in rows)
    checks["payload_degenerate_excluded_from_aggregates"] = all(
        s["n_degenerate_included"] == 0 for s in prof_rows) and all(not r["payload_degenerate"] for r in ak)
    expected = len(profs) * n_bursts_total * len(DEFENSES)
    checks["matrix_complete"] = bool(rows) and len(rows) == expected and len(by_burst) == len(profs) * n_bursts_total
    checks["all_four_attacks_present"] = {r["attack"] for r in rows} >= {"fgsm", "pgd", "cw", "ead"}
    # informational: x_adv regenerated under the prior run's seeds -> same attacked prediction / linf
    prior = {}
    pp = REPO_ROOT / args.prior_crc_run / "raw_bursts.csv"
    if pp.exists():
        with pp.open(newline="") as f:
            for r in csv.DictReader(f):
                prior[(r["attack"], r["profile"], r["true_modulation"], r["fec_requested"] == "True",
                       int(r["burst_index"]))] = r
    cmp_n = cmp_ok = 0
    for key, g in by_burst.items():
        p = prior.get(key)
        if p is None:
            continue
        r = g[0]
        cmp_n += 1
        cmp_ok += int(int(p["attacked_pred"]) == r["attacked_pred_before_defense"]
                      and abs(float(p["linf"]) - r["attack_linf"]) <= 1e-12)
    gated = ["completed", "attack_fallback_zero", "defense_fallback_zero", "adaptive_k_production_and_trace_identical",
             "fixed_k_matches_selected_k", "same_x_adv_across_defenses", "demod_status_allowed_only", "crc_pass_is_bool",
             "no_nan_inf", "payload_degenerate_excluded_from_aggregates", "matrix_complete", "all_four_attacks_present",
             "adversarial_rf_at_pin", "awn_submodule_at_pin", "checkpoint_hash_ok", "real_backends_loaded",
             "adaptive_is_production_function", "backend_toggle_matches_configuration"]
    failed = [k for k in gated if not checks.get(k)]
    validation = {"overall": "PASS" if not failed else "FAIL", "failed_checks": failed, "checks": checks,
                  "counters": dict(counters), "n_rows": len(rows), "n_valid_rows": len(valid),
                  "n_degenerate_rows": len(rows) - len(valid), "expected_rows": expected,
                  "n_burst_profile_units": len(by_burst), "n_profiles": len(profs), "n_bursts": n_bursts_total,
                  "informational": {"prior_run_attack_reproduction": {"compared": cmp_n, "identical_pred_and_linf": cmp_ok,
                                                                      "prior_run": args.prior_crc_run}},
                  "error": state.get("error")}
    (out / "validation.json").write_text(json.dumps(validation, indent=2, default=str))
    manifest = {
        "experiment": "reduced_iteration_crc_fec_defense_vm", "mode": "smoke" if args.smoke else "full",
        "command": [sys.executable] + sys.argv, "args": vars(args), "elapsed_s": time.monotonic() - t_start,
        "script_sha256": amc_study.sha256_file(Path(__file__).resolve()),
        "git": state.get("git"), "environment": state.get("environment"),
        "checkpoint_sha256": state.get("checkpoint_sha256"),
        "domain": "synthetic TX/RX (payload bits known); NOT RadioML2016",
        "burst_params": stx.LEGACY_CRC_PIPELINE_BURST, "synth_txrx": stx.SYNTH_TXRX_SOURCE, "soft_demod": True,
        "profiles": [{k: p[k] for k in ("attack", "profile", "eps", "alpha", "steps", "binary_search_steps",
                                        "max_iterations", "is_baseline")} | {"attack_params": p["attack_params"]}
                     for p in profs],
        "defenses": {
            "none": "x_adv passed unchanged",
            **{f"topk{k}": f"TopKAdapter(mode='fixed').apply(x_adv, topk={k}) -> {tka._REAL_SOURCE}; full complex FFT, "
               f"top-{k} bins kept per I/Q channel" for k in FIXED_KS},
            "adaptive_k": f"TopKAdapter(mode='adaptive').apply(x_adv) -> {tka._ADAPTIVE_SOURCE} with default parameters",
        },
        "adaptive_k_default_parameters": adaptive_defaults,
        "adaptive_k_snr_source": SNR_SOURCE,
        "adaptive_k_trace": "untimed re-run of the production helper functions; output verified bit-identical to "
                            "TopKAdapter output for every sample",
        "evaluation": "post-attack defense evaluation: one x_adv per burst/profile shared by all defense branches; "
                      "the attack is not defense-aware",
        "seed_rule": "burst: derive_seed(42, mod, snr, i, 'synth_txrx|fec<0|1>'); torch.manual_seed(burst seed) "
                     "before every attack; attack seed derive_seed(..., '<attack>|<profile>|fec<0|1>')",
        "degenerate_frames": "QPSK/PAM4 with FEC: 0 payload bits; kept in raw, excluded from all summaries",
        "effective_fec": "BPSK/CPFSK/GFSK cannot code (legacy capacity rule): effective_fec=False when FEC requested",
        "latency": {"attack_generation": "AttackAdapter.apply() wall time, one timed call per burst/profile "
                                         "(metadata; formal attack latency from earlier runs)",
                    "defense": "TopKAdapter.apply() wall time per branch (perf_counter_ns); stdout to null sink",
                    "attack_plus_defense_latency_ms_derived": "composite = attack latency + defense latency of the "
                                                              "same burst; derived, not a measured call",
                    "platform": "x86 VM latency: VMware VM, i7-11700, torch intra 1 / interop 1"},
        "backend_toggle": state.get("backend_toggle") or amc_study.backend_toggle_manifest(args.legacy_backend_toggle),
        "validation_overall": validation["overall"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log(f"validation {validation['overall']} failed={failed} ({time.monotonic() - t_start:.0f} s)", fh)
    fh.close()
    sys.exit(0 if validation["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
