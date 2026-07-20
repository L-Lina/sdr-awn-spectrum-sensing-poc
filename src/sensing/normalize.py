"""Per-segment normalization and conversion to AWN's expected input layout."""

from __future__ import annotations

import numpy as np

AWN_PREPROCESS_POLICIES = ("legacy-unit-power", "radioml-native")


def normalize_segments(segments: np.ndarray) -> np.ndarray:
    """Per-segment unit-average-power normalization."""
    power = np.mean(np.abs(segments) ** 2, axis=1, keepdims=True)
    power = np.maximum(power, 1e-12)
    return (segments / np.sqrt(power)).astype(np.complex64)


def apply_awn_preprocess(segments: np.ndarray, policy: str = "legacy-unit-power") -> np.ndarray:
    """
    The AWN-input-boundary preprocessing dispatcher (docs/parameter_validation.md
    section 19) -- the ONLY place a caller should choose how a segment's
    amplitude gets treated right before AWN inference. Never called from
    src/sensing/segmentation.py or src/sensing/energy_detection.py -- alignment
    and detection operate on the untouched IQ stream regardless of this policy.

    "legacy-unit-power" (default -- unchanged behavior, this round does NOT
    change the default): calls normalize_segments() -- per-segment unit-
    average-power normalization, a SCALAR (not per-sample-differential)
    rescale, so it preserves the RELATIVE amplitude structure within a
    segment (e.g. burst-vs-noise-floor ratio) exactly, but moves the
    ABSOLUTE amplitude far outside the ~1e-2-scale distribution the real
    AWN checkpoint was actually trained on (docs/parameter_validation.md
    section 18.1, section 19.1's traced evidence from
    external/adversarial-rf/data_loader.py, util/training.py,
    util/evaluation.py -- none of which normalize at all).

    "radioml-native" (docs/parameter_validation.md section 19.1): performs
    NO rescaling whatsoever -- a segment's amplitude is left exactly as
    constructed upstream (the original RadioML sample's own amplitude for
    direct/oracle-style inputs, or burst+synthetic-noise at whatever
    embed_snr_margin set for an embedded/sensed segment), matching the
    traced evidence that external/adversarial-rf never applies any
    normalization between its pickle loader and AWN.forward(). Since this
    is a no-op, it structurally cannot alter the burst/noise relative SNR
    or interfere with embed_snr_margin's effect -- there is nothing here
    that could break either.
    """
    if policy not in AWN_PREPROCESS_POLICIES:
        raise ValueError(f"awn_preprocess must be one of {AWN_PREPROCESS_POLICIES}, got {policy!r}")
    if policy == "legacy-unit-power":
        return normalize_segments(segments)
    return segments.astype(np.complex64)  # "radioml-native": no-op, dtype-only


def to_awn_input(segments: np.ndarray, seg_len: int) -> np.ndarray:
    """Convert complex64 [N, seg_len] segments to AWN's expected float32 [N, 2, seg_len] array."""
    if segments.dtype != np.complex64:
        raise TypeError(f"Expected complex64 segments, got {segments.dtype}")
    if segments.ndim != 2 or segments.shape[1] != seg_len:
        raise ValueError(f"Expected shape [N, {seg_len}], got {segments.shape}")

    x = np.stack([segments.real, segments.imag], axis=1).astype(np.float32)  # [N, 2, seg_len]
    print(f"[awn_input] converted to shape={x.shape}, dtype={x.dtype}")
    return x
