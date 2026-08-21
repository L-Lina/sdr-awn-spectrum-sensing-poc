"""
Satellite-like channel model for a single RadioML IQ burst.

Scope, per docs/research/SATELLITE_LIKE_CHANNEL_SIMULATOR_DESIGN_ZH_TW.md
section 2: implements exactly the project-close MUST and SHOULD channel
factors identified in Step 1 (docs/research/SATELLITE_APPLICATION_AND_
LATENCY_REQUIREMENTS_ZH_TW.md section 11) -- AWGN, amplitude/attenuation
scaling, carrier frequency offset (CFO), Doppler frequency shift, timing
offset -- plus propagation-delay metadata (recorded, never simulated as an
actual CPU delay or sample-domain effect within a short burst; see the
design doc section 9 for why). OPTIONAL factors (sample-rate offset,
non-linear amplifier, rain fade, shadowing, orbital geometry, full
transponder model) are explicitly OUT of scope this round and are not
implemented here.

This is a "satellite-like" / "standard-inspired" simulation, not a
standard-compliant DVB-S2/S2X or 3GPP NTN channel model -- see the design
doc for the exact mathematical model, parameter units, and primary-source
justification behind every constant used here. Operates on a single burst
(1-D complex IQ array), independent of and never called by
src/sensing/radioml_source.py:embed_sample_in_noise (that function's own
background-noise mechanism for the long synthetic stream is unmodified and
unrelated to this module's channel-level AWGN).

Does not modify external/AWN or external/adversarial-rf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class SatelliteChannelParams:
    sample_rate: float
    snr_db: Optional[float] = None  # None => no AWGN stage applied
    amplitude_scale: float = 1.0
    cfo_hz: float = 0.0
    doppler_hz: float = 0.0
    timing_offset_samples: int = 0
    propagation_delay_ms: Optional[float] = None  # metadata only, see module docstring
    seed: int = 0


def _apply_amplitude_scaling(iq: np.ndarray, amplitude_scale: float) -> np.ndarray:
    """y[n] = a * x[n], a real positive scalar. Represents free-space /
    link-budget attenuation as a pure magnitude scaling (no phase rotation
    -- phase effects are handled separately by CFO/Doppler)."""
    if amplitude_scale <= 0:
        raise ValueError(f"amplitude_scale must be > 0, got {amplitude_scale}")
    return iq * amplitude_scale


def _apply_timing_offset(iq: np.ndarray, timing_offset_samples: int) -> np.ndarray:
    """Integer-sample shift (not fractional-delay interpolation -- see
    design doc section 8 for why integer shift was chosen for this round).
    Positive timing_offset_samples delays the signal (zero-pads the leading
    edge, truncates the trailing edge); negative advances it (zero-pads
    the trailing edge, truncates the leading edge). Output length is
    always equal to input length."""
    n = len(iq)
    if timing_offset_samples == 0:
        return iq.copy()
    out = np.zeros(n, dtype=iq.dtype)
    if timing_offset_samples > 0:
        s = timing_offset_samples
        if s >= n:
            return out  # fully shifted out
        out[s:] = iq[:n - s]
    else:
        s = -timing_offset_samples
        if s >= n:
            return out
        out[:n - s] = iq[s:]
    return out


def _apply_frequency_rotation(iq: np.ndarray, combined_freq_hz: float, sample_rate: float) -> np.ndarray:
    """y[n] = x[n] * exp(j*2*pi*combined_freq_hz*n/sample_rate). Applied
    AFTER timing offset (see design doc section 8) so the sample index n
    used for phase accumulation is the same index the (already timing-
    shifted) samples occupy in the output array -- i.e. phase accumulates
    against the receiver's own sample clock, not some pre-shift reference."""
    n = np.arange(len(iq))
    rotation = np.exp(1j * 2.0 * np.pi * combined_freq_hz * n / sample_rate)
    return iq * rotation.astype(iq.dtype if np.iscomplexobj(iq) else np.complex128)


def _apply_awgn(iq: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """y[n] = x[n] + w[n], w[n] complex circularly-symmetric Gaussian,
    E[|w[n]|^2] = signal_power / 10^(snr_db/10). signal_power is measured
    on the ALREADY-TRANSFORMED iq (post amplitude/timing/frequency stages),
    so snr_db is the achieved SNR at the receiver's digital output, not
    relative to the original untransformed burst."""
    signal_power = float(np.mean(np.abs(iq) ** 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise_std = float(np.sqrt(noise_power / 2.0))  # split across real/imag
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, len(iq)) + 1j * rng.normal(0, noise_std, len(iq))
    return iq + noise.astype(iq.dtype)


def apply_satellite_like_channel(
    iq: np.ndarray,
    sample_rate: float,
    snr_db: Optional[float] = None,
    amplitude_scale: float = 1.0,
    cfo_hz: float = 0.0,
    doppler_hz: float = 0.0,
    timing_offset_samples: int = 0,
    propagation_delay_ms: Optional[float] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, dict]:
    """Applies the project-close MUST/SHOULD satellite-like channel
    factors to a single burst, in a fixed order (see design doc section 8
    for the physical rationale):

        input IQ -> amplitude scaling -> timing offset
                 -> CFO/Doppler phase rotation -> AWGN

    iq: 1-D complex64/complex128 array (this repo's IQ convention, matching
        src/sensing/radioml_source.py:radioml_sample_to_iq's output shape).
    sample_rate: Hz, the SIMULATOR's assumed baseband sample rate (a
        declared simulator-level assumption -- RadioML2016.10a itself does
        not carry a sample-rate field; see design doc section 6).
    snr_db: None means "no AWGN stage" (skip it entirely, not the same as
        an infinite/very-high SNR draw); otherwise the achieved SNR in dB
        of the OUTPUT relative to the (already amplitude/timing/frequency
        -transformed) signal power.
    amplitude_scale: positive real linear multiplier (not dB).
    cfo_hz, doppler_hz: Hz, recorded and reported SEPARATELY in metadata
        (see design doc section 10 for why they must never be collapsed
        into one undifferentiated "frequency_offset" field), but applied
        as a single combined phase rotation (cfo_hz + doppler_hz) since
        both manifest identically as a complex-exponential rotation in
        baseband and this implementation has no way to distinguish their
        physical origin once applied.
    timing_offset_samples: integer, see _apply_timing_offset.
    propagation_delay_ms: METADATA ONLY -- recorded in the returned dict
        for scenario/timeline bookkeeping (e.g. tagging which orbit's
        propagation-delay reference this run corresponds to, per Step 1
        Table 7.4.1-1); never used to delay execution (no time.sleep) and
        never applied as a sample-domain shift (a 128-sample burst at any
        reasonable sample rate is far shorter than any real propagation
        delay, so it cannot manifest as an in-burst effect -- see design
        doc section 9).
    seed: drives the AWGN draw only (all other stages are deterministic
        given their parameters).

    Returns (transformed_iq, metadata). metadata always includes: snr_db,
    amplitude_scale, cfo_hz, doppler_hz, combined_frequency_offset_hz,
    timing_offset_samples, sample_rate, propagation_delay_ms, seed,
    input_power, output_power.

    Raises ValueError if the input contains NaN/Inf (fail closed -- never
    silently processes already-corrupted input) or if the OUTPUT would
    contain NaN/Inf (defensive check; no stage here is expected to produce
    non-finite values for finite, valid parameters).
    """
    if iq.ndim != 1:
        raise ValueError(f"apply_satellite_like_channel expects a 1-D IQ array, got shape {iq.shape}")
    if not np.isfinite(iq).all():
        raise ValueError("input iq contains NaN/Inf -- refusing to process")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")

    input_power = float(np.mean(np.abs(iq) ** 2))

    out = _apply_amplitude_scaling(iq, amplitude_scale)
    out = _apply_timing_offset(out, timing_offset_samples)

    combined_freq_hz = cfo_hz + doppler_hz
    if combined_freq_hz != 0.0:
        out = _apply_frequency_rotation(out, combined_freq_hz, sample_rate)

    if snr_db is not None:
        out = _apply_awgn(out, snr_db, seed)

    if not np.isfinite(out).all():
        raise ValueError("apply_satellite_like_channel produced non-finite output -- check input parameters")

    output_power = float(np.mean(np.abs(out) ** 2))

    metadata = {
        "snr_db": snr_db,
        "amplitude_scale": amplitude_scale,
        "cfo_hz": cfo_hz,
        "doppler_hz": doppler_hz,
        "combined_frequency_offset_hz": combined_freq_hz,
        "timing_offset_samples": timing_offset_samples,
        "sample_rate": sample_rate,
        "propagation_delay_ms": propagation_delay_ms,
        "seed": seed,
        "input_power": input_power,
        "output_power": output_power,
    }
    return out, metadata
