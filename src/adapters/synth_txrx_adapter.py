"""
Synthetic TX/RX + CRC-8 + FEC adapter.

Thin wrapper around the existing implementation in
external/adversarial-rf/util/synth_txrx.py (the reference behavior). This
module contains NO CRC / convolutional-code / Viterbi / interleaver / modem
algorithm of its own -- every bit-level and IQ-level operation is a call into
synth_txrx:

    TX bits -> crc8 -> [conv_encode (rate 1/2, K=7) -> block_interleave]
            -> bits_to_symbols / FSK -> RRC pulse shaping -> channel(SNR)
            = synth_txrx.generate_burst(...)
    RX      -> matched filter -> pilot phase/amplitude recovery -> demap
            -> [LLR / hard bits -> block_deinterleave -> viterbi_decode]
            -> crc8_check
            = synth_txrx.demodulate_burst(...)

Domain boundary: this is the synthetic-burst experiment domain. RadioML2016
samples carry no payload bits, so CRC/FEC can only be evaluated on bursts
produced here; the RadioML AMC path (src/utils/pipeline.py,
experiments/run_cpu_full_matrix.py) is unchanged and separate.

Default burst parameters (LEGACY_CRC_PIPELINE_BURST) are exactly those of the
legacy CRC pipelines' generate_burst() call (external/adversarial-rf/
crc_defense_pipeline_fec.py:run_scenarios, crc_defense_pipeline.py,
crc_defense_fec_multi_attack.py): 16 symbols x 8 sps = 128 samples, 2 pilots,
RRC beta 0.35, target RMS 0.0082 (DEFAULT_TARGET_RMS), cfo_std 0.0, 16 guard
symbols; demodulation uses sps 8 / beta 0.35 / soft LLR Viterbi as in their
demod_burst().

Intentional divergence from legacy (see demodulate(legacy_fsk_ignores_window=...)): an attacked / defended
FSK window IS demodulated here; legacy scripts demodulated the clean FSK window instead.

Fail-closed differences from the legacy scripts (behavioral, not algorithmic):
  - there is no dummy fallback: if synth_txrx cannot be imported every entry
    point raises (require_backend()).
  - legacy demod_burst() wraps demodulate_burst() in `except Exception:
    return {'crc_pass': False}`. Here the two receiver outcomes that legacy
    code reached through that exception path are classified explicitly and
    BEFORE calling demodulate_burst: the demod modulation is not a digital
    modulation synth_txrx can demodulate (analog AMC prediction ->
    NOT_DIGITAL_MODULATION) or is of the other family than the transmitted
    burst (FSK <-> constellation: the burst carries no pilot bits / pilot
    symbols for that family -> FAMILY_MISMATCH). Both yield crc_pass=False
    with that status recorded. Any other exception propagates.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_ADVERSARIAL_RF_ROOT = Path(__file__).resolve().parents[2] / "external" / "adversarial-rf"
SYNTH_TXRX_SOURCE = "external/adversarial-rf/util/synth_txrx.py"

# crc_defense_pipeline_fec.py:run_scenarios -> generate_burst(mod, n_symbols=16, n_pilots=2, sps=8,
# beta=0.35, snr_db=snr, target_rms=target_rms, cfo_std=0.0, rng=rng, n_guard=16, fec=fec_flag),
# target_rms = DEFAULT_TARGET_RMS = 0.0082.
LEGACY_CRC_PIPELINE_BURST: Dict[str, object] = {
    "n_symbols": 16, "n_pilots": 2, "sps": 8, "beta": 0.35, "target_rms": 0.0082, "cfo_std": 0.0, "n_guard": 16,
}
# crc_defense_pipeline_fec.py:demod_burst -> demodulate_burst(..., sps=8, beta=0.35, ..., soft_demod=True)
DEMOD_SPS = 8
DEMOD_BETA = 0.35

DEMOD_OK = "OK"
DEMOD_NOT_DIGITAL = "NOT_DIGITAL_MODULATION"
DEMOD_FAMILY_MISMATCH = "FAMILY_MISMATCH"

_st = None
_import_error: Exception | None = None

try:
    _path_str = str(_ADVERSARIAL_RF_ROOT)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)
    import util.synth_txrx as _st  # type: ignore
except Exception as exc:  # noqa: BLE001 - recorded; require_backend() raises on first use
    _import_error = exc


def require_backend():
    """Return the synth_txrx module or raise -- never a dummy substitute."""
    if _st is None:
        raise RuntimeError(
            f"{SYNTH_TXRX_SOURCE} could not be imported ({type(_import_error).__name__}: {_import_error}); "
            "the synthetic TX/RX CRC/FEC chain has no fallback. Initialize the external/adversarial-rf submodule."
        )
    return _st


def digital_modulations() -> list:
    return sorted(require_backend().ALL_DIGITAL_MODS)


def modulation_family(mod: str) -> Optional[str]:
    st = require_backend()
    if mod in st.CONSTELLATION_MODS:
        return "constellation"
    if mod in st.FSK_MODS:
        return "fsk"
    return None


def fec_possible(mod: str, n_symbols: int = 16, n_pilots: int = 2) -> bool:
    """Legacy feasibility rule (crc_defense_pipeline_fec.py:run): FSK never codes; constellation mods only
    when synth_txrx.fec_payload_capacity() leaves room for the 8 CRC bits."""
    st = require_backend()
    if mod in st.FSK_MODS:
        return False
    return bool(st.fec_payload_capacity(n_symbols - n_pilots, st.get_bits_per_symbol(mod))[1])


def generate_frame(mod: str, snr_db: float, seed: int, fec: bool, **overrides) -> dict:
    """TX: random payload bits -> CRC-8 -> [FEC encode + interleave] -> modulation -> IQ burst.

    One burst per call from np.random.default_rng(seed). Returns synth_txrx.generate_burst()'s dict unchanged
    plus 'fec_requested' and 'burst_params'. burst['fec'] is what was actually applied (synth_txrx disables FEC
    for 1-bps modulations; FSK never codes)."""
    st = require_backend()
    params = dict(LEGACY_CRC_PIPELINE_BURST)
    params.update(overrides)
    rng = np.random.default_rng(seed)
    burst = st.generate_burst(mod, snr_db=snr_db, rng=rng, fec=fec, **params)
    burst["fec_requested"] = bool(fec)
    burst["burst_params"] = params
    return burst


def tx_bit_trace(burst: dict) -> dict:
    """Re-derive the transmitted bit stream of a burst with synth_txrx's own encoder functions.

    frame_bits = data_bits + crc8(data_bits); coded_bits = conv_encode(frame_bits);
    interleaved_bits = block_interleave(coded_bits); channel_bits = the bits mapped onto data symbols.
    Used to expose / validate the encoded bits; does not affect TX."""
    st = require_backend()
    frame = np.concatenate([burst["data_bits"], st.crc8(burst["data_bits"])]).astype(np.uint8)
    out = {"frame_bits": frame, "coded_bits": None, "interleaved_bits": None}
    if burst["fec"]:
        bps = st.get_bits_per_symbol(burst["mod_type"])
        n_data_symbols = len(burst["symbols_tx"]) - burst["n_pilots"]
        cap = n_data_symbols * bps
        coded = st.conv_encode(frame)
        inter = st.block_interleave(coded)
        out.update(coded_bits=coded, interleaved_bits=inter, channel_bits=inter[:cap].astype(np.uint8))
    else:
        out["channel_bits"] = frame
    return out


def window_to_complex(x_window: np.ndarray) -> np.ndarray:
    """[1, 2, T] (or [2, T]) float IQ window -> complex 1-D array (legacy tensor_to_complex)."""
    x = np.asarray(x_window)
    if x.ndim == 3:
        if x.shape[0] != 1:
            raise ValueError(f"expected a single window [1, 2, T], got {x.shape}")
        x = x[0]
    if x.ndim != 2 or x.shape[0] != 2:
        raise ValueError(f"expected IQ window [2, T], got {x.shape}")
    return x[0, :] + 1j * x[1, :]


def demodulate(burst: dict, demod_mod: str, x_window: Optional[np.ndarray] = None,
               soft_demod: bool = True, legacy_fsk_ignores_window: bool = False) -> dict:
    """RX: demodulate `burst` assuming modulation `demod_mod` (true modulation = oracle receiver; AMC
    prediction = AMC-assisted receiver), optionally with the 128-sample AMC window replaced by `x_window`
    (attacked / defended IQ), then FEC decode (if the burst is coded) and CRC-8 check.

    Window replacement mirrors legacy demod_burst(override_iq_complex=...): the window is spliced into a copy
    of iq_full at iq_win_start so the matched filter keeps its guard context (constellation bursts); FSK
    bursts have no iq_full and are demodulated on the window directly. FEC decode is applied iff the burst was
    coded (legacy fec=actual_fec), using the burst's fec_coded_len.

    legacy_fsk_ignores_window: reproduce a legacy quirk, for parity validation ONLY. Every legacy CRC script
    applies the override only `if override_iq_complex is not None and iq_full is not None`; FSK bursts have
    no iq_full, so legacy silently demodulated the CLEAN FSK window in its attacked / defended scenarios.
    Default False: the given window is always the one demodulated."""
    st = require_backend()
    family_tx = modulation_family(burst["mod_type"])
    family_rx = modulation_family(demod_mod) if demod_mod in st.ALL_DIGITAL_MODS else None
    base = {"demod_mod": demod_mod, "fec_enabled": bool(burst["fec"]), "soft_demod": bool(soft_demod)}
    if family_rx is None:
        return {**base, "demod_status": DEMOD_NOT_DIGITAL, "crc_pass": False,
                "recovered_bits": np.array([], dtype=np.uint8), "data_bits": np.array([], dtype=np.uint8)}
    if family_rx != family_tx:
        return {**base, "demod_status": DEMOD_FAMILY_MISMATCH, "crc_pass": False,
                "recovered_bits": np.array([], dtype=np.uint8), "data_bits": np.array([], dtype=np.uint8)}

    iq_full = burst.get("iq_full")
    iq_win_start = burst.get("iq_win_start")
    n_guard = burst.get("n_guard")
    if x_window is not None and iq_full is None and legacy_fsk_ignores_window:
        iq_complex = burst["iq_complex"]
    elif x_window is not None:
        iq_complex = window_to_complex(x_window)
        if len(iq_complex) != len(burst["iq_complex"]):
            raise ValueError(f"window length {len(iq_complex)} != burst window {len(burst['iq_complex'])}")
        if iq_full is not None:
            iq_full = iq_full.copy()
            iq_full[iq_win_start:iq_win_start + len(iq_complex)] = iq_complex
    else:
        iq_complex = burst["iq_complex"]

    res = st.demodulate_burst(
        iq_complex, demod_mod, n_pilots=burst["n_pilots"], pilot_symbols=burst.get("pilot_symbols"),
        pilot_bits=burst.get("pilot_bits"), sps=DEMOD_SPS, beta=DEMOD_BETA,
        pilot_positions=burst.get("pilot_positions"), iq_full=iq_full, iq_win_start=iq_win_start,
        n_guard=n_guard, fec=bool(burst["fec"]), soft_demod=soft_demod, fec_coded_len=burst.get("fec_coded_len"),
    )
    crc = res["crc_pass"]
    if not isinstance(crc, (bool, np.bool_)):
        raise TypeError(f"synth_txrx crc_pass must be boolean, got {type(crc).__name__}: {crc!r}")
    return {**base, "demod_status": DEMOD_OK, "crc_pass": bool(crc),
            "recovered_bits": np.asarray(res["recovered_bits"], dtype=np.uint8),
            "data_bits": np.asarray(res["data_bits"], dtype=np.uint8)}


def bit_errors(tx_bits: np.ndarray, rx_bits: np.ndarray) -> Optional[int]:
    """Hamming distance over the transmitted frame; a length mismatch counts the missing bits as errors.
    None when nothing was recovered (non-demodulated outcome)."""
    tx = np.asarray(tx_bits, dtype=np.uint8)
    rx = np.asarray(rx_bits, dtype=np.uint8)
    if rx.size == 0:
        return None
    n = min(tx.size, rx.size)
    return int(np.count_nonzero(tx[:n] != rx[:n]) + abs(tx.size - rx.size))
