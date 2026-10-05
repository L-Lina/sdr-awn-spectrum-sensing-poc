"""
TopKAdapter: wraps the real Top-K / adaptive-K FFT defender from
external/adversarial-rf/util/defense.py when it's importable, and falls
back to the numpy-only dummy_topk_defense placeholder otherwise.

Real functions this adapter targets (see external/adversarial-rf/util/defense.py,
and reports/adaptive_k_report_CN.md for the adaptive-K architecture writeup):
  - fft_topk_denoise(x: torch.Tensor[N,2,T], topk: int) -> torch.Tensor[N,2,T]
    Fixed-K FFT Top-K denoise; this is the one wired in here.
  - adaptive_k_v2_snr_defense(x: torch.Tensor[N,2,T], ...) -> torch.Tensor[N,2,T]
    Adaptive-K v2 with SNR-adaptive K cap (wideband routing + per-sample
    knee K capped by an SNR estimate taken from the same FFT). Wired in as
    mode="adaptive", called with its own default parameters. Opt-in only;
    unlike the fixed-K path it has NO dummy fallback -- an import or call
    failure raises instead of silently degrading to fixed-K or dummy.
  - adaptive_k_defense(x, ratio_thresh=0.05) -> per-sample knee-based K (not wired)
  - adaptive_k_v2_defense(x, ratio_thresh, k_max, flatness_threshold, quant_levels)
    (not wired)

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
from typing import Dict, Optional, Tuple

import numpy as np

from src.adapters.defense_adapter import dummy_topk_defense
from src.utils.config import TOPK_MODES, require_valid_topk

_ADVERSARIAL_RF_ROOT = Path(__file__).resolve().parents[2] / "external" / "adversarial-rf"
_REAL_SOURCE = "external/adversarial-rf/util/defense.py:fft_topk_denoise"
_ADAPTIVE_SOURCE = "external/adversarial-rf/util/defense.py:adaptive_k_v2_snr_defense"

_real_fft_topk_denoise = None
_import_error: Exception | None = None
_real_adaptive_k_v2_snr_defense = None
_adaptive_import_error: Exception | None = None

try:
    _path_str = str(_ADVERSARIAL_RF_ROOT)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)
    from util.defense import fft_topk_denoise as _real_fft_topk_denoise  # type: ignore
except Exception as exc:  # noqa: BLE001 - torch missing, or any other import-time failure
    _import_error = exc

# Separate import so a missing adaptive symbol can never affect the fixed-K import above.
try:
    from util.defense import adaptive_k_v2_snr_defense as _real_adaptive_k_v2_snr_defense  # type: ignore
except Exception as exc:  # noqa: BLE001 - torch missing, or any other import-time failure
    _adaptive_import_error = exc


class TopKAdapter:
    """Uniform Top-K defense interface.

    mode="fixed" (default): fixed-K fft_topk_denoise, real backend if available,
    dummy fallback otherwise (unchanged legacy behavior).
    mode="adaptive": production adaptive_k_v2_snr_defense; raises at construction
    if it cannot be imported and at apply() if the call fails -- never falls back.
    """

    def __init__(self, mode: str = "fixed") -> None:
        if mode not in TOPK_MODES:
            raise ValueError(f"TopKAdapter mode must be one of {TOPK_MODES}, got {mode!r}")
        self.mode = mode
        if mode == "adaptive":
            if _real_adaptive_k_v2_snr_defense is None:
                raise RuntimeError(
                    f"TopKAdapter(mode='adaptive') requires {_ADAPTIVE_SOURCE}, which failed to import "
                    f"({type(_adaptive_import_error).__name__}: {_adaptive_import_error}). "
                    "Adaptive-K has no dummy/fixed-K fallback; check that torch is installed and that "
                    "the external/adversarial-rf submodule is checked out."
                )
            self.backend_available = True
            self.backend_name = _ADAPTIVE_SOURCE
            self.notes = f"Loaded real adaptive_k_v2_snr_defense from {_ADAPTIVE_SOURCE} (default parameters)"
            return
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

    def apply(self, x: np.ndarray, topk: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, str]]:
        """x: [N, 2, T] float32. Returns (y, meta) with y of the same shape.

        topk is required in fixed mode and must be None in adaptive mode (K is
        chosen per sample by adaptive_k_v2_snr_defense)."""
        if self.mode == "adaptive":
            return self._apply_adaptive(x, topk)
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

    def _apply_adaptive(self, x: np.ndarray, topk: Optional[int]) -> Tuple[np.ndarray, Dict[str, str]]:
        if topk is not None:
            raise ValueError(f"TopKAdapter(mode='adaptive') selects K per sample; topk must be None, got {topk!r}")
        if not isinstance(x, np.ndarray) or x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"TopKAdapter expects input [N, 2, T], got {getattr(x, 'shape', type(x))}")
        input_shape = x.shape

        import torch

        # Explicit numpy -> torch boundary: float32, C-contiguous, CPU. No try/except here --
        # any failure propagates to the caller instead of degrading to another defense.
        x_t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        with torch.no_grad():
            y_t = _real_adaptive_k_v2_snr_defense(x_t)
        if not isinstance(y_t, torch.Tensor):
            raise RuntimeError(f"adaptive_k_v2_snr_defense returned {type(y_t).__name__}, expected torch.Tensor")
        y = y_t.detach().cpu().numpy().astype(np.float32)

        if y.shape != input_shape:
            raise RuntimeError(f"TopKAdapter output shape {y.shape} != input shape {input_shape}")
        if not np.isfinite(y).all():
            raise RuntimeError("adaptive_k_v2_snr_defense produced non-finite values")

        print(f"[topk_adapter] backend={self.backend_name} status=ok mode=adaptive input={input_shape} output={y.shape}")
        return y, {
            "topk_backend": self.backend_name,
            "topk_status": "ok",
            "topk_notes": self.notes,
            "topk_mode": "adaptive",
        }
