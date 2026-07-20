"""
TopKAdapter: wraps the real Top-K / adaptive-K FFT defender from
external/adversarial-rf/util/defense.py when it's importable, and falls
back to the numpy-only dummy_topk_defense placeholder otherwise.

Real functions this adapter targets (see external/adversarial-rf/util/defense.py,
and reports/adaptive_k_report_CN.md for the adaptive-K architecture writeup):
  - fft_topk_denoise(x: torch.Tensor[N,2,T], topk: int) -> torch.Tensor[N,2,T]
    Fixed-K FFT Top-K denoise; this is the one wired in here.
  - adaptive_k_defense(x, ratio_thresh=0.05) -> per-sample knee-based K (not wired yet)
  - adaptive_k_v2_defense(x, ratio_thresh, k_max, flatness_threshold, quant_levels)
    (not wired yet)

`util/defense.py` does `import torch` at module scope, so importing it
requires torch to be installed. Per this phase's constraints, torch is NOT
installed and packages are not to be installed -- so the import below is
expected to fail and fall back to the dummy implementation. The real-import
path is still fully wired so this adapter starts using the real function
automatically the moment torch becomes available, with no code changes.

This module never modifies external/adversarial-rf; it only reads from it
(adds its path to sys.path for the duration of the import attempt).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.adapters.defense_adapter import dummy_topk_defense
from src.utils.config import require_valid_topk

_ADVERSARIAL_RF_ROOT = Path(__file__).resolve().parents[2] / "external" / "adversarial-rf"
_REAL_SOURCE = "external/adversarial-rf/util/defense.py:fft_topk_denoise"

_real_fft_topk_denoise = None
_import_error: Exception | None = None

try:
    _path_str = str(_ADVERSARIAL_RF_ROOT)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)
    from util.defense import fft_topk_denoise as _real_fft_topk_denoise  # type: ignore
except Exception as exc:  # noqa: BLE001 - torch missing, or any other import-time failure
    _import_error = exc


class TopKAdapter:
    """Uniform Top-K defense interface, real backend if available, dummy fallback otherwise."""

    def __init__(self) -> None:
        self.backend_available = _real_fft_topk_denoise is not None
        if self.backend_available:
            self.backend_name = _REAL_SOURCE
            self.notes = f"Loaded real fft_topk_denoise from {_REAL_SOURCE}"
        else:
            self.backend_name = "dummy_topk_defense"
            self.notes = (
                f"Real Top-K import failed ({type(_import_error).__name__}: {_import_error}); "
                f"fell back to src/adapters/defense_adapter.py:dummy_topk_defense. "
                f"{_REAL_SOURCE} requires torch, which is not installed in this phase "
                "-- see docs/integration_plan.md."
            )

    def apply(self, x: np.ndarray, topk: int) -> Tuple[np.ndarray, Dict[str, str]]:
        """x: [N, 2, T] float32. Returns (y, meta) with y of the same shape."""
        # Validated BEFORE any backend selection/try-except below, so an
        # invalid topk (NaN/Inf/non-numeric/fractional) raises immediately
        # here and never gets a chance to be caught by the real-backend
        # try/except and silently fall back to dummy_topk_defense. topk<=0
        # (bypass) and topk > T (clamp) semantics are unaffected.
        topk = require_valid_topk("topk", topk)
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"TopKAdapter expects input [N, 2, T], got {x.shape}")
        input_shape = x.shape

        if self.backend_available:
            try:
                import torch

                x_t = torch.from_numpy(x)
                y_t = _real_fft_topk_denoise(x_t, topk=topk)
                y = y_t.detach().cpu().numpy().astype(np.float32)
                backend, status, notes = self.backend_name, "ok", self.notes
            except Exception as exc:  # noqa: BLE001 - real backend failed at call time
                y = dummy_topk_defense(x, topk=topk)
                backend = "dummy_topk_defense"
                status = "fallback"
                notes = f"Real Top-K call failed at runtime ({type(exc).__name__}: {exc}); used numpy fallback."
        else:
            y = dummy_topk_defense(x, topk=topk)
            backend, status, notes = self.backend_name, "fallback", self.notes

        if y.shape != input_shape:
            raise RuntimeError(f"TopKAdapter output shape {y.shape} != input shape {input_shape}")

        print(f"[topk_adapter] backend={backend} status={status} input={input_shape} output={y.shape}")
        return y, {"topk_backend": backend, "topk_status": status, "topk_notes": notes}
