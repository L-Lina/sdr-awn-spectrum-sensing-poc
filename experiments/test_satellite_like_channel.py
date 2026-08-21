"""
Unit tests for src/channel/satellite_like.py -- the satellite-like channel
model added this round (Step 3). Verifies the mathematical model actually
implemented matches the design doc (docs/research/SATELLITE_LIKE_CHANNEL_
SIMULATOR_DESIGN_ZH_TW.md section 5), not just that the function runs.

All tests operate on synthetic complex tones/noise generated in-process;
does not require the RadioML dataset or the real AWN backend.

Run directly:
    python experiments/test_satellite_like_channel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.channel.satellite_like import apply_satellite_like_channel  # noqa: E402

PASS = []
FAIL = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    (PASS if cond else FAIL).append(name)
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))


def make_tone(n: int = 128, freq_hz: float = 0.0, sample_rate: float = 200_000.0,
              amp: float = 1.0, seed: int = 0) -> np.ndarray:
    """A short complex tone (or, at freq_hz=0, a pseudo-random-phase-free
    constant-magnitude complex signal derived from a fixed random symbol
    sequence) used as a controlled test input."""
    rng = np.random.default_rng(seed)
    # random unit-magnitude symbols (like a PSK burst) at baseband, then
    # optionally modulated onto a tone -- gives a signal with known power
    # (amp^2) and, when freq_hz != 0, a known, measurable phase ramp.
    phases = rng.uniform(0, 2 * np.pi, n)
    symbols = amp * np.exp(1j * phases)
    n_idx = np.arange(n)
    tone = np.exp(1j * 2 * np.pi * freq_hz * n_idx / sample_rate)
    return (symbols * tone).astype(np.complex64)


def measure_phase_increment(iq: np.ndarray) -> float:
    """Average per-sample phase increment (radians), via the angle of the
    mean product of consecutive-sample ratios -- robust to the underlying
    random symbol phases as long as they are i.i.d. and the SAME sequence
    is compared between two runs with only a frequency difference."""
    ratios = iq[1:] * np.conj(iq[:-1])
    return float(np.angle(np.mean(ratios)))


def main() -> None:
    n = 128
    fs = 200_000.0

    # 1. identity case: all impairments off/neutral -> output == input
    x = make_tone(n, freq_hz=0.0, sample_rate=fs, seed=1)
    y, meta = apply_satellite_like_channel(
        x, sample_rate=fs, snr_db=None, amplitude_scale=1.0,
        cfo_hz=0.0, doppler_hz=0.0, timing_offset_samples=0, seed=0,
    )
    check("1. identity case: output == input", np.array_equal(x, y), f"max diff={np.max(np.abs(x-y))}")
    check("1b. identity case: metadata combined_frequency_offset_hz == 0", meta["combined_frequency_offset_hz"] == 0.0)

    # 2. amplitude scaling: output power scales as a^2 * input power
    for a in [0.5, 1.0, 2.0]:
        x = make_tone(n, freq_hz=0.0, sample_rate=fs, seed=2)
        y, meta = apply_satellite_like_channel(
            x, sample_rate=fs, snr_db=None, amplitude_scale=a,
            cfo_hz=0.0, doppler_hz=0.0, timing_offset_samples=0, seed=0,
        )
        expected_power = meta["input_power"] * a * a
        check(f"2. amplitude_scale={a}: output power == a^2 * input power",
              abs(meta["output_power"] - expected_power) / expected_power < 1e-9,
              f"got {meta['output_power']}, expected {expected_power}")

    # 3. CFO: measured phase increment matches 2*pi*cfo_hz/fs
    cfo = 2000.0
    x = make_tone(n, freq_hz=0.0, sample_rate=fs, seed=3)
    y, meta = apply_satellite_like_channel(
        x, sample_rate=fs, snr_db=None, amplitude_scale=1.0,
        cfo_hz=cfo, doppler_hz=0.0, timing_offset_samples=0, seed=0,
    )
    x_ratios = x[1:] * np.conj(x[:-1])
    y_ratios = y[1:] * np.conj(y[:-1])
    delta_phase = np.angle(np.mean(y_ratios * np.conj(x_ratios)))
    expected_phase = 2 * np.pi * cfo / fs
    check("3. CFO: measured phase increment matches 2*pi*cfo_hz/fs",
          abs(delta_phase - expected_phase) < 1e-6,
          f"got {delta_phase}, expected {expected_phase}")
    check("3b. CFO metadata: cfo_hz recorded separately from doppler_hz",
          meta["cfo_hz"] == cfo and meta["doppler_hz"] == 0.0)

    # 4. Doppler: same mechanism as CFO, but recorded under doppler_hz
    doppler = 1000.0
    x = make_tone(n, freq_hz=0.0, sample_rate=fs, seed=4)
    y, meta = apply_satellite_like_channel(
        x, sample_rate=fs, snr_db=None, amplitude_scale=1.0,
        cfo_hz=0.0, doppler_hz=doppler, timing_offset_samples=0, seed=0,
    )
    x_ratios = x[1:] * np.conj(x[:-1])
    y_ratios = y[1:] * np.conj(y[:-1])
    delta_phase = np.angle(np.mean(y_ratios * np.conj(x_ratios)))
    expected_phase = 2 * np.pi * doppler / fs
    check("4. Doppler: measured phase increment matches 2*pi*doppler_hz/fs",
          abs(delta_phase - expected_phase) < 1e-6,
          f"got {delta_phase}, expected {expected_phase}")
    check("4b. Doppler metadata: doppler_hz recorded separately from cfo_hz",
          meta["doppler_hz"] == doppler and meta["cfo_hz"] == 0.0)

    # 5. CFO + Doppler: combined rotation == cfo + doppler
    x = make_tone(n, freq_hz=0.0, sample_rate=fs, seed=5)
    y, meta = apply_satellite_like_channel(
        x, sample_rate=fs, snr_db=None, amplitude_scale=1.0,
        cfo_hz=cfo, doppler_hz=doppler, timing_offset_samples=0, seed=0,
    )
    check("5. combined_frequency_offset_hz == cfo_hz + doppler_hz",
          meta["combined_frequency_offset_hz"] == cfo + doppler)
    x_ratios = x[1:] * np.conj(x[:-1])
    y_ratios = y[1:] * np.conj(y[:-1])
    delta_phase = np.angle(np.mean(y_ratios * np.conj(x_ratios)))
    expected_phase = 2 * np.pi * (cfo + doppler) / fs
    check("5b. CFO+Doppler: measured phase increment matches combined frequency",
          abs(delta_phase - expected_phase) < 1e-6,
          f"got {delta_phase}, expected {expected_phase}")
    # metadata must still keep them distinguishable
    check("5c. CFO+Doppler: metadata keeps cfo_hz and doppler_hz distinct (not collapsed)",
          meta["cfo_hz"] == cfo and meta["doppler_hz"] == doppler)

    # 6. timing shift: sample shift correctness
    x = np.arange(n, dtype=np.complex64)  # distinctive ramp, easy to check shift
    for shift in [0, 3, -5]:
        y, meta = apply_satellite_like_channel(
            x, sample_rate=fs, snr_db=None, amplitude_scale=1.0,
            cfo_hz=0.0, doppler_hz=0.0, timing_offset_samples=shift, seed=0,
        )
        check(f"6. timing_offset_samples={shift}: output length preserved", len(y) == n)
        if shift == 0:
            ok = np.array_equal(y, x)
        elif shift > 0:
            ok = np.array_equal(y[shift:], x[:n - shift]) and np.all(y[:shift] == 0)
        else:
            s = -shift
            ok = np.array_equal(y[:n - s], x[s:]) and np.all(y[n - s:] == 0)
        check(f"6b. timing_offset_samples={shift}: shift content correct", ok)

    # 7. AWGN: achieved SNR within tolerance
    x = make_tone(n * 100, freq_hz=0.0, sample_rate=fs, seed=7)  # larger n for stable SNR estimate
    target_snr_db = 10.0
    y, meta = apply_satellite_like_channel(
        x, sample_rate=fs, snr_db=target_snr_db, amplitude_scale=1.0,
        cfo_hz=0.0, doppler_hz=0.0, timing_offset_samples=0, seed=42,
    )
    noise = y - x
    achieved_signal_power = float(np.mean(np.abs(x) ** 2))
    achieved_noise_power = float(np.mean(np.abs(noise) ** 2))
    achieved_snr_db = 10.0 * np.log10(achieved_signal_power / achieved_noise_power)
    check("7. AWGN: achieved SNR within 1.0 dB of target (large-n estimate)",
          abs(achieved_snr_db - target_snr_db) < 1.0,
          f"target={target_snr_db}, achieved={achieved_snr_db:.3f}")

    # 8. deterministic seed: same seed + params -> identical output
    x = make_tone(n, freq_hz=0.0, sample_rate=fs, seed=8)
    y1, _ = apply_satellite_like_channel(x, sample_rate=fs, snr_db=5.0, amplitude_scale=1.0,
                                          cfo_hz=100.0, doppler_hz=50.0, timing_offset_samples=2, seed=99)
    y2, _ = apply_satellite_like_channel(x, sample_rate=fs, snr_db=5.0, amplitude_scale=1.0,
                                          cfo_hz=100.0, doppler_hz=50.0, timing_offset_samples=2, seed=99)
    check("8. deterministic seed: identical params + seed -> bit-identical output", np.array_equal(y1, y2))
    y3, _ = apply_satellite_like_channel(x, sample_rate=fs, snr_db=5.0, amplitude_scale=1.0,
                                          cfo_hz=100.0, doppler_hz=50.0, timing_offset_samples=2, seed=100)
    check("8b. different seed -> different output (AWGN draw differs)", not np.array_equal(y1, y3))

    # 9. shape/dtype preserved
    x64 = make_tone(n, freq_hz=0.0, sample_rate=fs, seed=9).astype(np.complex64)
    y64, _ = apply_satellite_like_channel(x64, sample_rate=fs, snr_db=10.0, amplitude_scale=1.5,
                                           cfo_hz=10.0, doppler_hz=5.0, timing_offset_samples=1, seed=0)
    check("9. shape preserved", y64.shape == x64.shape)
    check("9b. dtype preserved (complex64 in -> complex64 out)", y64.dtype == np.complex64)

    # 10. NaN/Inf rejection
    x_bad = make_tone(n, freq_hz=0.0, sample_rate=fs, seed=10)
    x_bad[5] = np.nan + 1j * 0
    try:
        apply_satellite_like_channel(x_bad, sample_rate=fs, snr_db=10.0, amplitude_scale=1.0,
                                      cfo_hz=0.0, doppler_hz=0.0, timing_offset_samples=0, seed=0)
        check("10. NaN input rejected", False, "did not raise")
    except ValueError:
        check("10. NaN input rejected", True)

    x_bad2 = make_tone(n, freq_hz=0.0, sample_rate=fs, seed=11)
    x_bad2[5] = np.inf + 1j * 0
    try:
        apply_satellite_like_channel(x_bad2, sample_rate=fs, snr_db=10.0, amplitude_scale=1.0,
                                      cfo_hz=0.0, doppler_hz=0.0, timing_offset_samples=0, seed=0)
        check("10b. Inf input rejected", False, "did not raise")
    except ValueError:
        check("10b. Inf input rejected", True)

    # additional: invalid amplitude_scale / sample_rate rejected
    try:
        apply_satellite_like_channel(x, sample_rate=fs, snr_db=None, amplitude_scale=0.0,
                                      cfo_hz=0.0, doppler_hz=0.0, timing_offset_samples=0, seed=0)
        check("11. amplitude_scale=0 rejected", False, "did not raise")
    except ValueError:
        check("11. amplitude_scale=0 rejected", True)

    try:
        apply_satellite_like_channel(x, sample_rate=0.0, snr_db=None, amplitude_scale=1.0,
                                      cfo_hz=0.0, doppler_hz=0.0, timing_offset_samples=0, seed=0)
        check("12. sample_rate=0 rejected", False, "did not raise")
    except ValueError:
        check("12. sample_rate=0 rejected", True)

    # metadata completeness
    x = make_tone(n, freq_hz=0.0, sample_rate=fs, seed=13)
    _, meta = apply_satellite_like_channel(x, sample_rate=fs, snr_db=10.0, amplitude_scale=1.2,
                                            cfo_hz=100.0, doppler_hz=50.0, timing_offset_samples=1,
                                            propagation_delay_ms=26.0, seed=0)
    required_keys = {"snr_db", "amplitude_scale", "cfo_hz", "doppler_hz", "combined_frequency_offset_hz",
                      "timing_offset_samples", "sample_rate", "propagation_delay_ms", "seed",
                      "input_power", "output_power"}
    check("13. metadata contains all required keys", required_keys.issubset(set(meta.keys())),
          f"missing: {required_keys - set(meta.keys())}")
    check("13b. propagation_delay_ms recorded as metadata, unchanged (not applied to signal)",
          meta["propagation_delay_ms"] == 26.0)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
