"""Per-segment normalization and conversion to AWN's expected input layout."""

from __future__ import annotations

import numpy as np


def normalize_segments(segments: np.ndarray) -> np.ndarray:
    """Per-segment unit-average-power normalization."""
    power = np.mean(np.abs(segments) ** 2, axis=1, keepdims=True)
    power = np.maximum(power, 1e-12)
    return (segments / np.sqrt(power)).astype(np.complex64)


def to_awn_input(segments: np.ndarray, seg_len: int) -> np.ndarray:
    """Convert complex64 [N, seg_len] segments to AWN's expected float32 [N, 2, seg_len] array."""
    if segments.dtype != np.complex64:
        raise TypeError(f"Expected complex64 segments, got {segments.dtype}")
    if segments.ndim != 2 or segments.shape[1] != seg_len:
        raise ValueError(f"Expected shape [N, {seg_len}], got {segments.shape}")

    x = np.stack([segments.real, segments.imag], axis=1).astype(np.float32)  # [N, 2, seg_len]
    print(f"[awn_input] converted to shape={x.shape}, dtype={x.dtype}")
    return x
