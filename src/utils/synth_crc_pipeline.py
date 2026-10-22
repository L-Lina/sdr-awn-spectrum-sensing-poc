"""
Synthetic-burst CRC/FEC experiment domain: one burst through the full chain

    TX bits -> CRC-8 -> [FEC encode + interleave] -> modulation / IQ      (synth_txrx_adapter.generate_frame)
    -> adversarial attack on the 128-sample AMC window                    (AttackAdapter.apply, production)
    -> [optional defense: FFT Top-K]                                      (TopKAdapter.apply, production)
    -> AMC                                                                (AWNModelAdapter.infer, production)
    -> demodulation (oracle = true modulation; AMC-assisted = predicted)  (synth_txrx_adapter.demodulate)
    -> [FEC decode] -> CRC check

Insertion points are those of the legacy pipelines (external/adversarial-rf/crc_defense_pipeline_fec.py:
run_scenarios): the attack and the defense act on burst['iq_tensor'] (the window AWN classifies); the
receiver demodulates that same window spliced back into the guarded burst.

Shared conventions with the RadioML AMC domain (experiments/run_cpu_full_matrix.py):
  - AMC input is the raw IQ window, as "radioml-native" (src/sensing/normalize.py) is a dtype-only no-op;
  - AttackAdapter is called exactly as in the formal matrix (temperature 1.0, diagnostics off, untargeted
    against the model's own clean prediction, per-row attack seed), preceded by torch.manual_seed(torch_seed);
  - attack_success = clean_correct AND NOT attacked_correct (formal definition);
  - every adapter result is checked fail-closed: any fallback / non-real backend / non-finite output raises.

This module never swallows exceptions.
"""

from __future__ import annotations

import contextlib
import io
import time
from typing import Dict, Optional, Tuple

import numpy as np

from src.adapters import synth_txrx_adapter as stx
from src.sensing.radioml_source import RML2016_10A_CLASSES

IDX_TO_MOD = {v: k for k, v in RML2016_10A_CLASSES.items()}
DEMOD_SCENARIOS = ("clean_oracle", "clean_amc", "attacked_oracle", "attacked_amc")


class ChainError(RuntimeError):
    """A component of the chain fell back, used a non-real backend, or produced invalid output."""


def _quiet(enabled: bool):
    return contextlib.redirect_stdout(io.StringIO()) if enabled else contextlib.nullcontext()


def _infer(awn, x: np.ndarray, what: str, quiet: bool) -> Tuple[int, np.ndarray]:
    with _quiet(quiet):
        logits, meta = awn.infer(x)
    if meta.get("awn_status") != "ok" or "dummy" in str(meta.get("awn_backend", "")).lower():
        raise ChainError(f"AWN {what} inference fallback/error: {meta}")
    return int(np.argmax(logits[0])), logits


def bits_str(bits: np.ndarray) -> str:
    return "".join(str(int(b)) for b in np.asarray(bits, dtype=np.uint8))


def run_burst_chain(
    burst: dict,
    *,
    awn,
    attack_adapter,
    attack: str,
    eps: float,
    torch_seed: int,
    attack_seed: int,
    attack_kwargs: Optional[Dict[str, object]] = None,
    topk_adapter=None,
    topk: Optional[int] = None,
    soft_demod: bool = True,
    quiet: bool = True,
) -> Tuple[dict, np.ndarray]:
    """Run one synthetic burst through the chain. Returns (row, x_after_attack_and_defense).

    attack_kwargs: forwarded to AttackAdapter.apply (cw_c / cw_steps / cw_lr / attack_params).
    attack 'none' skips the attack (attacked_* then describe the clean / defended window).
    Row latency field attack_latency_ns times only the AttackAdapter.apply() call."""
    import torch
    from src.adapters.attack_adapter import _REAL_ATTACK_SOURCE

    mod = burst["mod_type"]
    if mod not in RML2016_10A_CLASSES:
        raise ChainError(f"burst modulation {mod!r} is not an AWN class")
    label = RML2016_10A_CLASSES[mod]
    x_clean = np.ascontiguousarray(burst["iq_tensor"], dtype=np.float32)
    if x_clean.shape != (1, 2, 128):
        raise ChainError(f"unexpected burst window shape {x_clean.shape}")

    clean_pred, _ = _infer(awn, x_clean, "clean", quiet)

    attack_name = (attack or "none").lower()
    latency_ns = None
    attack_meta: Dict[str, object] = {}
    if attack_name == "none":
        x_att = x_clean
    else:
        if attack_adapter.status != "ok" or attack_adapter.backend_name != _REAL_ATTACK_SOURCE:
            raise ChainError(f"AttackAdapter is not on the real backend: {attack_adapter.status} / "
                             f"{attack_adapter.backend_name} / {attack_adapter.notes}")
        torch.manual_seed(int(torch_seed))
        with _quiet(quiet):
            t0 = time.perf_counter_ns()
            x_att, attack_meta = attack_adapter.apply(x_clean, attack=attack_name, eps=eps, temperature=1.0,
                                                      seed=int(attack_seed), diagnostics=False,
                                                      **dict(attack_kwargs or {}))
            latency_ns = time.perf_counter_ns() - t0
        if attack_meta.get("attack_status") != "ok" or attack_meta.get("attack_backend") != _REAL_ATTACK_SOURCE:
            raise ChainError(f"attack fallback/error ({attack_name}): status={attack_meta.get('attack_status')} "
                             f"backend={attack_meta.get('attack_backend')} notes={attack_meta.get('attack_notes')}")
    if x_att.shape != x_clean.shape or x_att.dtype != x_clean.dtype or not np.isfinite(x_att).all():
        raise ChainError(f"invalid attacked window shape={x_att.shape} dtype={x_att.dtype} "
                         f"finite={bool(np.isfinite(x_att).all())}")

    defense_meta: Dict[str, object] = {}
    x_rx = x_att
    if topk_adapter is not None and topk is not None:
        with _quiet(quiet):
            x_rx, defense_meta = topk_adapter.apply(x_att, topk=topk)
        if defense_meta.get("topk_status") != "ok" or "dummy" in str(defense_meta.get("topk_backend", "")).lower():
            raise ChainError(f"Top-K defense fallback/error: {defense_meta}")
        if x_rx.shape != x_clean.shape or not np.isfinite(x_rx).all():
            raise ChainError("invalid defended window")

    attacked_pred, _ = _infer(awn, x_rx, "attacked", quiet)
    clean_correct = clean_pred == label
    attacked_correct = attacked_pred == label

    tx = stx.tx_bit_trace(burst)
    frame_bits = tx["frame_bits"]
    if not np.array_equal(frame_bits, burst["frame_bits"]):
        raise ChainError("tx_bit_trace frame bits disagree with the burst's frame_bits")

    demod = {
        "clean_oracle": stx.demodulate(burst, mod, None, soft_demod),
        "clean_amc": stx.demodulate(burst, IDX_TO_MOD[clean_pred], None, soft_demod),
        "attacked_oracle": stx.demodulate(burst, mod, x_rx, soft_demod),
        "attacked_amc": stx.demodulate(burst, IDX_TO_MOD[attacked_pred], x_rx, soft_demod),
    }

    delta = (x_att.astype(np.float64) - x_clean.astype(np.float64))
    row = {
        "domain": "synthetic_txrx",
        "true_modulation": mod, "true_label": label, "snr_db": burst["snr_db"],
        "fec_requested": bool(burst.get("fec_requested", burst["fec"])), "fec_enabled": bool(burst["fec"]),
        "soft_demod": bool(soft_demod), "n_data_bits": int(len(burst["data_bits"])),
        "n_frame_bits": int(len(frame_bits)), "fec_coded_len": int(burst.get("fec_coded_len") or 0),
        "attack": attack_name, "eps": eps, "torch_seed": int(torch_seed), "attack_seed": int(attack_seed),
        "clean_pred": clean_pred, "clean_pred_mod": IDX_TO_MOD[clean_pred],
        "attacked_pred": attacked_pred, "attacked_pred_mod": IDX_TO_MOD[attacked_pred],
        "clean_correct": bool(clean_correct), "attacked_correct": bool(attacked_correct),
        "attack_success": bool(clean_correct and not attacked_correct),
        "pred_changed": attacked_pred != clean_pred,
        "attack_latency_ns": latency_ns, "attack_latency_ms": None if latency_ns is None else latency_ns / 1e6,
        "attack_status": attack_meta.get("attack_status", "none"),
        "attack_backend": attack_meta.get("attack_backend", "none"),
        "defense": f"topk{topk}" if (topk_adapter is not None and topk is not None) else "none",
        "defense_backend": defense_meta.get("topk_backend", "none"),
        "linf": float(np.max(np.abs(delta))), "l2": float(np.linalg.norm(delta.reshape(-1))),
        "tx_frame_bits": bits_str(frame_bits),
        "tx_channel_bits": bits_str(tx["channel_bits"]),
    }
    for sc in DEMOD_SCENARIOS:
        d = demod[sc]
        row[f"{sc}_demod_mod"] = d["demod_mod"]
        row[f"{sc}_demod_status"] = d["demod_status"]
        row[f"{sc}_crc_pass"] = d["crc_pass"]
        row[f"{sc}_recovered_bits"] = bits_str(d["recovered_bits"])
        row[f"{sc}_bit_errors"] = stx.bit_errors(frame_bits, d["recovered_bits"])
        rec = np.asarray(d["recovered_bits"], dtype=np.uint8)
        row[f"{sc}_recovered_bit_count"] = int(rec.size)
        # BER over the transmitted frame (data + CRC): errors in the overlapping bits + missing frame bits,
        # divided by the frame length (extra bits recovered under a higher-order demod are not frame bits).
        # None when the receiver did not demodulate.
        if rec.size == 0 or len(frame_bits) == 0:
            row[f"{sc}_frame_bit_errors"], row[f"{sc}_ber"] = None, None
        else:
            n = min(rec.size, len(frame_bits))
            fbe = int(np.count_nonzero(frame_bits[:n] != rec[:n]) + max(0, len(frame_bits) - rec.size))
            row[f"{sc}_frame_bit_errors"], row[f"{sc}_ber"] = fbe, fbe / len(frame_bits)
        # receiver accepted (CRC pass) AND the recovered frame is exactly the transmitted frame
        row[f"{sc}_frame_correct"] = bool(d["crc_pass"] and row[f"{sc}_bit_errors"] == 0)
        # CRC accepted a frame whose bits differ from what was sent (CRC-8 undetected error)
        row[f"{sc}_crc_false_accept"] = bool(d["crc_pass"] and row[f"{sc}_bit_errors"] not in (0, None))
    row["recovered_bits"] = row["attacked_amc_recovered_bits"]
    row["crc_pass"] = row["attacked_amc_crc_pass"]
    row["attack_success_and_crc_pass_oracle"] = bool(row["attack_success"] and row["attacked_oracle_crc_pass"])
    row["attack_success_and_crc_pass_amc"] = bool(row["attack_success"] and row["attacked_amc_crc_pass"])
    return row, x_rx
