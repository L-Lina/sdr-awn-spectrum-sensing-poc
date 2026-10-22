"""
Integration validation: synthetic TX/RX + CRC-8 + FEC chain (src/adapters/synth_txrx_adapter.py,
src/utils/synth_crc_pipeline.py) against the legacy reference external/adversarial-rf.

V1 parity (fixed seeds, no AMC): for every digital modulation x SNR x FEC on/off x seed
   a. TX     : the literal legacy call (crc_defense_pipeline_fec.py:run_scenarios ->
               synth_txrx.generate_burst(..., target_rms=DEFAULT_TARGET_RMS, cfo_std=0.0, n_guard=16)) vs
               generate_frame(): every returned array / scalar identical (IQ window, guarded IQ, bits, pilots,
               symbols, cfo, phase, fec flags).
   b. encode : tx_bit_trace() channel bits == synth_txrx.symbols_to_bits() of the transmitted data symbols
               (constellation); FEC frames also round-trip through block_deinterleave + hard viterbi_decode.
   c. decode : legacy crc_defense_pipeline_fec.demod_burst() (+ its AMC non-digital rule) vs demodulate() for
               every one of the 11 AWN classes as demod modulation, clean window and a perturbed window, soft
               and hard Viterbi: crc_pass and recovered bits identical. FAMILY_MISMATCH outcomes must be ones
               where legacy demodulate_burst() actually raises (legacy recorded crc False through its except).
               Same-family demodulation must never raise.
   d. crc    : every wrapper crc_pass equals synth_txrx.crc8_check(recovered bits).
V2 clean chain (no attack, real AWN): encode -> transmit -> AMC -> decode -> CRC; noiseless (SNR 100 dB)
   oracle CRC must pass with 0 bit errors for every modulation / FEC setting; SNR 18 reported.
V3 attack chain: FGSM / PGD / CW / EAD (formal default + lowest reduced setting) through the full chain via
   the production AttackAdapter; plus PGD with the production Top-K defense inserted.
V4 fail-closed: dummy attack backend, attack runtime fallback (EAD max_iterations < 10), dummy AWN, Top-K
   fallback, missing synth_txrx backend, a same-family demod exception, a non-boolean crc_pass -- each must
   raise; none may be recorded as a CRC result.

Run from the repo root:
    ~/adversarial-rf/.venv/bin/python experiments/validate_synth_crc_fec_integration.py \
        --checkpoint ~/adversarial-rf/2016.10a_AWN.pkl
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
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ADV_RF_ROOT = REPO_ROOT / "external" / "adversarial-rf"
for _p in (REPO_ROOT, ADV_RF_ROOT, REPO_ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_cpu_full_matrix as formal  # noqa: E402 -- derive_seed
from src.adapters import synth_txrx_adapter as stx  # noqa: E402
from src.utils import synth_crc_pipeline as scp  # noqa: E402

EXPECTED_CHECKPOINT_SHA256 = "8af0458f2570c465b5bb0ebad00817944f8171d888cd7cff1324ecb258820695"
EXPECTED_ADV_RF_COMMIT = "ced705ed861bde352841b0b8bd038a6449e0fa4d"
BASE_SEED = 42
PARITY_SNRS = [0, 10, 18, 100]
ALL_CLASSES = sorted(scp.IDX_TO_MOD.values())

_LOG = None


def log(msg: str) -> None:
    line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, file=sys.stderr, flush=True)
    if _LOG:
        _LOG.write(line + "\n")
        _LOG.flush()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def write_csv(path: Path, rows: list) -> None:
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["empty"], restval="")
        w.writeheader()
        w.writerows(rows)


def same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
    return a == b


def burst_seed(mod, snr, i, fec) -> int:
    return formal.derive_seed(BASE_SEED, mod, int(snr), int(i), f"synth_txrx|fec{int(bool(fec))}")


# =========================================================================== V1 parity
def v1_parity(args, st, ref) -> dict:
    rows, problems = [], []
    n_family_mismatch_legacy_raised = 0
    divergence = {}  # default-mode vs legacy-compat differences, by (family, window)
    for mod in stx.digital_modulations():
        for snr in PARITY_SNRS:
            for fec in (False, True):
                for i in range(args.parity_seeds):
                    seed = burst_seed(mod, snr, i, fec)
                    ref_b = st.generate_burst(mod, n_symbols=16, n_pilots=2, sps=8, beta=0.35, snr_db=snr,
                                              target_rms=ref.DEFAULT_TARGET_RMS, cfo_std=0.0,
                                              rng=np.random.default_rng(seed), n_guard=16, fec=fec)
                    wr_b = stx.generate_frame(mod, snr, seed, fec)
                    tag = f"{mod}/snr{snr}/fec{int(fec)}/seed{seed}"
                    # a. TX
                    tx_ok = all(same(ref_b[k], wr_b[k]) for k in ref_b)
                    if not tx_ok:
                        problems.append(f"{tag}: TX burst differs on " +
                                        ",".join(k for k in ref_b if not same(ref_b[k], wr_b[k])))
                    # b. encoded bits
                    tr = stx.tx_bit_trace(wr_b)
                    if mod in st.CONSTELLATION_MODS:
                        pilot = set(wr_b["pilot_positions"])
                        data_syms = np.array([s for j, s in enumerate(wr_b["symbols_tx"]) if j not in pilot])
                        enc_ok = same(st.symbols_to_bits(data_syms, mod).astype(np.uint8), tr["channel_bits"])
                    else:
                        enc_ok = same(tr["channel_bits"], wr_b["frame_bits"])
                    rt_ok = True
                    if wr_b["fec"]:
                        dec = st.viterbi_decode(st.block_deinterleave(tr["channel_bits"], wr_b["fec_coded_len"]),
                                                soft=False)
                        rt_ok = same(dec.astype(np.uint8), wr_b["frame_bits"].astype(np.uint8))
                    if not (enc_ok and rt_ok):
                        problems.append(f"{tag}: encoded-bit trace mismatch (map={enc_ok}, fec roundtrip={rt_ok})")
                    # c/d. decode parity
                    noise_rng = np.random.default_rng(seed + 1)
                    x_pert = (wr_b["iq_tensor"] + 0.5 * ref.DEFAULT_TARGET_RMS *
                              noise_rng.standard_normal(wr_b["iq_tensor"].shape)).astype(np.float32)
                    n_cmp = n_ok = 0
                    for window_name, x_win in (("clean", None), ("perturbed", x_pert)):
                        for soft in (True, False):
                            for dm in ALL_CLASSES:
                                n_cmp += 1
                                if dm in st.ALL_DIGITAL_MODS:
                                    ref_r = ref.demod_burst(
                                        ref_b, dm, fec=ref_b["fec"], soft_demod=soft,
                                        override_iq_complex=None if x_win is None else
                                        ref.tensor_to_complex(x_win[0]))
                                else:  # legacy AMC rule: non-digital prediction -> crc False, no demod
                                    ref_r = {"crc_pass": False, "recovered_bits": np.array([], dtype=np.uint8)}
                                try:
                                    # parity is gated in legacy-compat mode; default mode is checked below
                                    wr_r = stx.demodulate(wr_b, dm, x_win, soft, legacy_fsk_ignores_window=True)
                                    wr_d = stx.demodulate(wr_b, dm, x_win, soft)
                                except Exception as exc:  # noqa: BLE001 -- recorded as a parity failure
                                    problems.append(f"{tag} {window_name} soft={soft} demod={dm}: wrapper raised "
                                                    f"{type(exc).__name__}: {exc}")
                                    continue
                                if not (wr_d["crc_pass"] == wr_r["crc_pass"] and
                                        same(wr_d["recovered_bits"], wr_r["recovered_bits"])):
                                    key = f"{stx.modulation_family(mod)}/{window_name}"
                                    divergence[key] = divergence.get(key, 0) + 1
                                    if key != "fsk/perturbed":
                                        problems.append(f"{tag} {window_name} demod={dm}: default mode differs "
                                                        "from legacy outside the documented FSK-window case")
                                if wr_d["demod_status"] == stx.DEMOD_OK and (
                                        bool(st.crc8_check(wr_d["recovered_bits"])) != wr_d["crc_pass"]):
                                    problems.append(f"{tag} demod={dm}: default crc_pass != crc8_check(recovered)")
                                ok = (bool(ref_r["crc_pass"]) == wr_r["crc_pass"] and
                                      same(np.asarray(ref_r["recovered_bits"], dtype=np.uint8), wr_r["recovered_bits"]))
                                if wr_r["demod_status"] == stx.DEMOD_FAMILY_MISMATCH:
                                    try:
                                        st.demodulate_burst(
                                            wr_b["iq_complex"], dm, n_pilots=wr_b["n_pilots"],
                                            pilot_symbols=wr_b.get("pilot_symbols"),
                                            pilot_bits=wr_b.get("pilot_bits"), sps=8, beta=0.35,
                                            pilot_positions=wr_b.get("pilot_positions"),
                                            iq_full=wr_b.get("iq_full"), iq_win_start=wr_b.get("iq_win_start"),
                                            n_guard=wr_b.get("n_guard"), fec=wr_b["fec"], soft_demod=soft,
                                            fec_coded_len=wr_b.get("fec_coded_len"))
                                        ok = False
                                        problems.append(f"{tag} demod={dm}: FAMILY_MISMATCH but legacy "
                                                        "demodulate_burst did not raise")
                                    except Exception:  # noqa: BLE001 -- expected: legacy exception path
                                        n_family_mismatch_legacy_raised += 1
                                if wr_r["demod_status"] == stx.DEMOD_OK:
                                    if bool(st.crc8_check(wr_r["recovered_bits"])) != wr_r["crc_pass"]:
                                        ok = False
                                        problems.append(f"{tag} demod={dm}: crc_pass != crc8_check(recovered)")
                                if not ok:
                                    problems.append(f"{tag} {window_name} soft={soft} demod={dm}: decode parity "
                                                    f"ref crc={ref_r['crc_pass']} wrapper crc={wr_r['crc_pass']} "
                                                    f"status={wr_r['demod_status']}")
                                n_ok += int(ok)
                    rows.append({"modulation": mod, "snr_db": snr, "fec_requested": fec, "fec_enabled": wr_b["fec"],
                                 "seed": seed, "n_data_bits": len(wr_b["data_bits"]), "tx_identical": tx_ok,
                                 "encoded_bits_ok": enc_ok and rt_ok, "decode_comparisons": n_cmp,
                                 "decode_identical": n_ok,
                                 "clean_oracle_crc_pass": stx.demodulate(wr_b, mod)["crc_pass"]})
    total_cmp = sum(r["decode_comparisons"] for r in rows)
    total_ok = sum(r["decode_identical"] for r in rows)
    return {"rows": rows, "problems": problems, "n_bursts": len(rows),
            "tx_identical": sum(r["tx_identical"] for r in rows),
            "encoded_bits_ok": sum(r["encoded_bits_ok"] for r in rows),
            "decode_comparisons": total_cmp, "decode_identical": total_ok,
            "family_mismatch_confirmed_legacy_exception": n_family_mismatch_legacy_raised,
            "default_mode_divergence_from_legacy": divergence,
            "divergence_note": "legacy demod_burst() ignores the override window when iq_full is None (all FSK "
                               "bursts); default mode demodulates the given window. Parity is gated in "
                               "legacy-compat mode (legacy_fsk_ignores_window=True).",
            "pass": not problems and total_cmp == total_ok}


# =========================================================================== V2 / V3
def chain_rows(awn, attack_adapter, snrs, n_bursts, profiles, topk_adapter=None, topk=None) -> list:
    rows = []
    for prof in profiles:
        for mod in stx.digital_modulations():
            for snr in snrs:
                for fec in (False, True):
                    for i in range(n_bursts):
                        seed = burst_seed(mod, snr, i, fec)
                        b = stx.generate_frame(mod, snr, seed, fec)
                        a_seed = formal.derive_seed(BASE_SEED, mod, int(snr), i,
                                                    f"{prof['attack']}|{prof['profile']}|fec{int(fec)}")
                        r, _ = scp.run_burst_chain(b, awn=awn, attack_adapter=attack_adapter, attack=prof["attack"],
                                                   eps=prof["eps"], torch_seed=seed, attack_seed=a_seed,
                                                   attack_kwargs=prof.get("kwargs"), topk_adapter=topk_adapter,
                                                   topk=topk)
                        r.update(profile=prof["profile"], burst_index=i)
                        rows.append(r)
    return rows


def summarize(rows: list, keys=("attack", "profile", "defense", "fec_enabled")) -> list:
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for k, rr in groups.items():
        cc = [r for r in rr if r["clean_correct"]]
        lat = [r["attack_latency_ms"] for r in rr if r["attack_latency_ms"] is not None]
        out.append({**dict(zip(keys, k)), "n": len(rr), "n_clean_correct": len(cc),
                    "conditional_asr_pct": 100 * sum(r["attack_success"] for r in rr) / len(cc) if cc else None,
                    **{f"{sc}_crc_pass_pct": 100 * np.mean([r[f"{sc}_crc_pass"] for r in rr])
                       for sc in scp.DEMOD_SCENARIOS},
                    "attack_success_and_crc_pass_oracle_n": sum(r["attack_success_and_crc_pass_oracle"] for r in rr),
                    "attack_success_and_crc_pass_amc_n": sum(r["attack_success_and_crc_pass_amc"] for r in rr),
                    "crc_false_accept_n": sum(r[f"{sc}_crc_false_accept"] for r in rr for sc in scp.DEMOD_SCENARIOS),
                    "latency_median_ms": float(np.median(lat)) if lat else None})
    return out


# =========================================================================== V4
def v4_fail_closed(awn, ckpt, st) -> dict:
    import torch  # noqa: F401
    from src.adapters.attack_adapter import AttackAdapter
    from src.adapters.awn_adapter import AWNModelAdapter
    from src.adapters.topk_adapter import TopKAdapter

    burst = stx.generate_frame("QPSK", 18, burst_seed("QPSK", 18, 0, False), False)
    q = contextlib.redirect_stdout(io.StringIO())
    results = {}

    def expect_raise(name, fn, exc_types=(Exception,)):
        try:
            fn()
            results[name] = {"raised": False, "pass": False}
        except exc_types as exc:
            results[name] = {"raised": True, "pass": True, "exception": f"{type(exc).__name__}: {str(exc)[:200]}"}

    with q:
        real_attack = AttackAdapter(awn_model=awn.model)
        dummy_attack = AttackAdapter(awn_model=None)
        dummy_awn = AWNModelAdapter(str(REPO_ROOT / "nonexistent_checkpoint.pkl"))
    expect_raise("dummy_attack_backend_rejected", lambda: scp.run_burst_chain(
        burst, awn=awn, attack_adapter=dummy_attack, attack="fgsm", eps=0.03, torch_seed=0, attack_seed=0),
        (scp.ChainError,))
    expect_raise("attack_runtime_fallback_rejected (EAD max_iterations=5)", lambda: scp.run_burst_chain(
        burst, awn=awn, attack_adapter=real_attack, attack="ead", eps=0.03, torch_seed=0, attack_seed=0,
        attack_kwargs={"attack_params": {"binary_search_steps": 1, "max_iterations": 5}}), (scp.ChainError,))
    expect_raise("dummy_awn_rejected", lambda: scp.run_burst_chain(
        burst, awn=dummy_awn, attack_adapter=real_attack, attack="none", eps=0.0, torch_seed=0, attack_seed=0),
        (scp.ChainError,))
    with q:
        topk_fb = TopKAdapter()
    topk_fb.backend_available = False
    expect_raise("topk_fallback_rejected", lambda: scp.run_burst_chain(
        burst, awn=awn, attack_adapter=real_attack, attack="none", eps=0.0, torch_seed=0, attack_seed=0,
        topk_adapter=topk_fb, topk=20), (scp.ChainError,))

    saved = stx._st
    stx._st = None
    try:
        expect_raise("missing_synth_txrx_backend_rejected", lambda: stx.generate_frame("QPSK", 18, 0, False),
                     (RuntimeError,))
    finally:
        stx._st = saved

    broken = dict(burst)
    broken["pilot_symbols"] = None
    expect_raise("same_family_demod_exception_propagates", lambda: stx.demodulate(broken, "QPSK"))

    orig = st.demodulate_burst
    st.demodulate_burst = lambda *a, **k: {**orig(*a, **k), "crc_pass": None}
    try:
        expect_raise("non_boolean_crc_pass_rejected", lambda: stx.demodulate(burst, "QPSK"), (TypeError,))
    finally:
        st.demodulate_burst = orig
    return {"tests": results, "pass": all(v["pass"] for v in results.values())}


# =========================================================================== main
def main() -> None:
    global _LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--parity-seeds", type=int, default=10)
    ap.add_argument("--clean-bursts", type=int, default=25)
    ap.add_argument("--attack-bursts", type=int, default=2)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    import torch
    import torchattacks
    torch.set_num_interop_threads(1)
    torch.set_num_threads(args.torch_threads)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (REPO_ROOT / (args.out_dir or f"results/synth_crc_fec_integration_{ts}")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    _LOG = (out / "run.log").open("w")
    t_start = time.monotonic()
    checks, report = {}, {}
    try:
        st = stx.require_backend()
        import crc_defense_pipeline_fec as ref  # legacy reference (external/adversarial-rf)
        from src.adapters.attack_adapter import AttackAdapter
        from src.adapters.awn_adapter import AWNModelAdapter
        from src.adapters.topk_adapter import TopKAdapter

        adv_head = (ADV_RF_ROOT / ".git").exists() and __import__("subprocess").run(
            ["git", "-C", str(ADV_RF_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        checks["adversarial_rf_at_pin"] = adv_head == EXPECTED_ADV_RF_COMMIT
        ckpt = Path(args.checkpoint).expanduser().resolve()
        checks["checkpoint_hash_ok"] = sha256_file(ckpt) == EXPECTED_CHECKPOINT_SHA256

        log("V1 parity ...")
        v1 = v1_parity(args, st, ref)
        write_csv(out / "v1_parity.csv", v1["rows"])
        report["v1_parity"] = {k: v for k, v in v1.items() if k != "rows"}
        report["v1_parity"]["problems"] = v1["problems"][:50]
        checks["v1_parity"] = v1["pass"]
        log(f"V1: bursts {v1['n_bursts']}, TX identical {v1['tx_identical']}, encoded ok {v1['encoded_bits_ok']}, "
            f"decode identical {v1['decode_identical']}/{v1['decode_comparisons']}, problems {len(v1['problems'])}")

        with contextlib.redirect_stdout(io.StringIO()):
            awn = AWNModelAdapter(str(ckpt))
            attack_adapter = AttackAdapter(awn_model=awn.model, b1_pgd_bim_mkldnn_off=True,
                                           b1_mkldnn_off_extended=True)
            topk_adapter = TopKAdapter()
        checks["real_backends_loaded"] = awn.status == "ok" and attack_adapter.status == "ok"

        log("V2 clean chain ...")
        none_prof = [{"attack": "none", "profile": "none", "eps": 0.0}]
        v2_rows = chain_rows(awn, attack_adapter, [18, 100], args.clean_bursts, none_prof)
        write_csv(out / "v2_clean_chain.csv", v2_rows)
        per = {}
        for r in v2_rows:
            k = (r["true_modulation"], r["snr_db"], r["fec_enabled"])
            per.setdefault(k, []).append(r)
        v2_summary = [{"modulation": k[0], "snr_db": k[1], "fec_enabled": k[2], "n": len(v),
                       "clean_amc_acc_pct": 100 * np.mean([r["clean_correct"] for r in v]),
                       "clean_oracle_crc_pass_pct": 100 * np.mean([r["clean_oracle_crc_pass"] for r in v]),
                       "clean_amc_crc_pass_pct": 100 * np.mean([r["clean_amc_crc_pass"] for r in v]),
                       "oracle_bit_errors_total": sum(r["clean_oracle_bit_errors"] or 0 for r in v)}
                      for k, v in sorted(per.items())]
        write_csv(out / "v2_clean_chain_summary.csv", v2_summary)
        noiseless = [s for s in v2_summary if s["snr_db"] == 100]
        checks["v2_noiseless_oracle_crc_100pct"] = all(
            s["clean_oracle_crc_pass_pct"] == 100.0 and s["oracle_bit_errors_total"] == 0 for s in noiseless)
        report["v2_clean_chain"] = v2_summary
        log(f"V2: {len(v2_rows)} bursts; noiseless oracle CRC 100% = {checks['v2_noiseless_oracle_crc_100pct']}")

        log("V3 attack chain ...")
        profiles = [
            {"attack": "fgsm", "profile": "eps0.03", "eps": 0.03, "kwargs": {}},
            {"attack": "pgd", "profile": "eps0.03_steps10", "eps": 0.03,
             "kwargs": {"attack_params": {"alpha": 0.0075, "steps": 10, "random_start": True}}},
            {"attack": "pgd", "profile": "eps0.03_steps1", "eps": 0.03,
             "kwargs": {"attack_params": {"alpha": 0.0075, "steps": 1, "random_start": True}}},
            {"attack": "cw", "profile": "c1_steps20_lr0.01", "eps": 0.03,
             "kwargs": {"cw_c": 1.0, "cw_steps": 20, "cw_lr": 0.01, "attack_params": {"kappa": 0.0}}},
            {"attack": "cw", "profile": "c1_steps1_lr0.01", "eps": 0.03,
             "kwargs": {"cw_c": 1.0, "cw_steps": 1, "cw_lr": 0.01, "attack_params": {"kappa": 0.0}}},
            {"attack": "ead", "profile": "bss9_maxit100", "eps": 0.03, "kwargs": {}},
            {"attack": "ead", "profile": "bss5_maxit10", "eps": 0.03,
             "kwargs": {"attack_params": {"binary_search_steps": 5, "max_iterations": 10, "lr": 0.01,
                                          "initial_const": 1e-3, "beta": 1e-3, "kappa": 0, "abort_early": True}}},
        ]
        v3_rows = chain_rows(awn, attack_adapter, [18], args.attack_bursts, profiles)
        v3_rows += chain_rows(awn, attack_adapter, [18], args.attack_bursts, profiles[1:2], topk_adapter, 20)
        write_csv(out / "v3_attack_chain.csv", v3_rows)
        v3_summary = summarize(v3_rows)
        write_csv(out / "v3_attack_chain_summary.csv", v3_summary)
        expected = (len(profiles) + 1) * len(stx.digital_modulations()) * 2 * args.attack_bursts
        checks["v3_attack_chain_complete"] = (len(v3_rows) == expected and all(
            r["attack_status"] == "ok" for r in v3_rows) and {r["attack"] for r in v3_rows} >= {"fgsm", "pgd", "cw",
                                                                                                 "ead"})
        report["v3_attack_chain"] = v3_summary
        log(f"V3: {len(v3_rows)}/{expected} attacked bursts through the full chain")

        log("V4 fail-closed ...")
        v4 = v4_fail_closed(awn, ckpt, st)
        report["v4_fail_closed"] = v4
        checks["v4_fail_closed"] = v4["pass"]
        checks["completed"] = True
    except Exception as exc:  # noqa: BLE001 -- recorded; validation FAILs
        report["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        log(f"ERROR: {report['error']}")
        checks["completed"] = False
    failed = [k for k, v in checks.items() if not v]
    validation = {"overall": "PASS" if not failed and checks.get("completed") else "FAIL",
                  "failed_checks": failed, "checks": checks, **report}
    (out / "validation.json").write_text(json.dumps(validation, indent=2, default=str))
    manifest = {"experiment": "synth_crc_fec_integration_validation", "command": [sys.executable] + sys.argv,
                "args": vars(args), "elapsed_s": time.monotonic() - t_start,
                "torch": torch.__version__, "torchattacks": torchattacks.__version__, "numpy": np.__version__,
                "torch_threads": torch.get_num_threads(), "python": sys.version,
                "reference": {"synth_txrx": stx.SYNTH_TXRX_SOURCE,
                              "legacy_pipeline": "external/adversarial-rf/crc_defense_pipeline_fec.py",
                              "adversarial_rf_commit": EXPECTED_ADV_RF_COMMIT},
                "burst_params": stx.LEGACY_CRC_PIPELINE_BURST, "base_seed": BASE_SEED,
                "seed_rule": "derive_seed(42, mod, snr, burst_index, 'synth_txrx|fec<0|1>') -> "
                             "np.random.default_rng(seed) per burst; torch.manual_seed(seed) before the attack"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log(f"validation {validation['overall']} failed={failed} -> {out} ({time.monotonic() - t_start:.0f} s)")
    _LOG.close()
    sys.exit(0 if validation["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
