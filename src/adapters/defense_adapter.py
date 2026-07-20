"""
Simplified Top-K FFT denoising defense (numpy-only stand-in).

This mirrors the interface and core behavior of the real defender in
external/adversarial-rf/util/defense.py:fft_topk_denoise (keep the K
largest-magnitude FFT bins per I/Q channel, zero the rest, inverse FFT back
to time domain). It is intentionally a simplified, dependency-free version
so the dry-run pipeline can exercise the same [N, 2, T] -> [N, 2, T]
contract end to end.

TODO(phase 4+): swap this for the real torch-based fft_topk_denoise (and its
adaptive_k_defense / adaptive_k_v2_defense variants) once torch is wired in,
per docs/integration_plan.md section 2.
"""

from __future__ import annotations

import numpy as np

from src.utils.config import require_valid_topk


def dummy_topk_defense(x: np.ndarray, topk: int) -> np.ndarray:
    """Keep only the top-k FFT magnitude bins per sample/channel, reconstruct via inverse FFT."""
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"Expected input of shape [N, 2, T], got {x.shape}")

    topk = require_valid_topk("topk", topk)
    n, c, t = x.shape
    k = min(topk, t) if topk > 0 else t

    X = np.fft.fft(x, n=t, axis=2)
    mags = np.abs(X)
    idx = np.argsort(-mags, axis=2)[:, :, :k]
    mask = np.zeros_like(mags, dtype=bool)
    np.put_along_axis(mask, idx, True, axis=2)
    X_filt = np.where(mask, X, 0)
    y = np.fft.ifft(X_filt, n=t, axis=2).real.astype(np.float32)

    print(f"[PLACEHOLDER] dummy_topk_defense: topk={k}/{t} -> output {y.shape}")
    return y
