"""
IQ stream sources: synthetic generator (for dry-run experiments) and a raw
complex64 .cfile reader (for a future real GNU Radio capture hookup).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def generate_synthetic_iq(
    n_samples: int,
    burst_len: int,
    snr_db: float,
    mod: str,
    seed: Optional[int] = 0,
) -> tuple[np.ndarray, dict]:
    """
    Generate noise + a single occupied burst (complex64), simulating a GNU
    Radio IQ capture at a given SNR. `mod` is carried through as metadata and
    used only to vary the burst's cosmetic frequency offset -- this is NOT a
    real modulation waveform synthesizer (that's out of scope while AWN
    inference is still a placeholder; see docs/integration_plan.md).
    """
    if burst_len > n_samples:
        raise ValueError(f"burst_len ({burst_len}) must not exceed n_samples ({n_samples})")

    burst_start = (n_samples - burst_len) // 2
    burst_end = burst_start + burst_len

    burst_amp = 1.0
    noise_std = float(np.sqrt(burst_amp ** 2 / (2.0 * 10 ** (snr_db / 10.0))))

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, n_samples) + 1j * rng.normal(0, noise_std, n_samples)
    iq = noise.astype(np.complex64)

    freq_offset = 0.05 + (abs(hash(mod)) % 100) / 1000.0
    t = np.arange(burst_len)
    carrier = np.exp(1j * 2 * np.pi * freq_offset * t)
    iq[burst_start:burst_end] += (burst_amp * carrier).astype(np.complex64)

    meta = {
        "n_samples": n_samples,
        "burst_start": burst_start,
        "burst_end": burst_end,
        "burst_len": burst_len,
        "snr_db": snr_db,
        "mod": mod,
        "noise_std": noise_std,
        "freq_offset": freq_offset,
    }
    print(f"[gen] synthetic IQ: {n_samples} samples, burst at [{burst_start}:{burst_end}], "
          f"snr={snr_db} dB, mod={mod}")
    return iq, meta


def load_iq_from_file(path: str) -> np.ndarray:
    """
    Load IQ samples from a raw complex64 .cfile.

    Intended hookup point for a GNU Radio flowgraph:
        UHD: USRP Source -> File Sink (output type = complex64) -> captured_iq.cfile
    """
    iq = np.fromfile(path, dtype=np.complex64)
    if iq.size == 0:
        raise ValueError(f"No samples read from '{path}' - file empty, wrong path, or wrong dtype")
    print(f"[load] read {iq.size} IQ samples from {path}")
    return iq


def validate_iq(iq: np.ndarray) -> np.ndarray:
    """Ensure IQ stream is complex and 1-D; raise rather than silently casting."""
    if not np.iscomplexobj(iq):
        raise TypeError(
            f"Expected a complex IQ stream, got dtype={iq.dtype}. "
            "If reading a .cfile, confirm the GNU Radio capture used complex64 (gr_complex), "
            "not real-valued or interleaved int16 samples."
        )
    if iq.dtype != np.complex64:
        print(f"[warn] input dtype is {iq.dtype}, casting to complex64")
        iq = iq.astype(np.complex64)
    if iq.ndim != 1:
        raise ValueError(f"Expected 1-D IQ stream, got shape={iq.shape}")
    return iq
