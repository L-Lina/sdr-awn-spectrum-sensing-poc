"""
Placeholder + real AWN inference adapter.

dummy_awn_inference(...) is the numpy-only placeholder (kept as the default
and as the fallback target for AWNModelAdapter below).

AWNModelAdapter wraps the real AWN model class from
external/adversarial-rf/models/model.py (byte-identical to
external/AWN/models/model.py at the currently pinned submodule commits --
diff-verified) plus checkpoint loading, per docs/integration_plan.md section 2.

`models/model.py` only imports torch/torch.nn and its own `models.lifting`
sibling, so importing `models.model.AWN` directly (bypassing
`util/utils.py:create_AWN_model`, which also pulls in unrelated model
classes like VTCNN2/ResNet1D/MCLDNN) keeps this adapter's import surface
minimal. Either way, torch is required and is NOT installed in this phase --
so the import below is expected to fail and fall back to the dummy
implementation, exactly like src/adapters/topk_adapter.py. This module never
modifies external/AWN or external/adversarial-rf; it only reads from the
latter (adds its path to sys.path for the duration of the import attempt).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_ADVERSARIAL_RF_ROOT = Path(__file__).resolve().parents[2] / "external" / "adversarial-rf"
_REAL_MODEL_SOURCE = "external/adversarial-rf/models/model.py:AWN"

# Mirrors external/adversarial-rf/config/2016.10a.yml, diff-verified identical
# to external/AWN/config/2016.10a.yml. Built directly here instead of going
# through adversarial-rf's util/config.py:Config (which creates training/
# and inference/ directories as a side effect of construction).
_AWN_2016_10A_CFG = dict(
    num_classes=11,
    num_levels=1,
    in_channels=64,
    kernel_size=3,
    latent_dim=320,
    regu_details=0.01,
    regu_approx=0.01,
)

_real_AWN_cls = None
_import_error: Exception | None = None

try:
    _path_str = str(_ADVERSARIAL_RF_ROOT)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)
    from models.model import AWN as _real_AWN_cls  # type: ignore
except Exception as exc:  # noqa: BLE001 - torch missing, or any other import-time failure
    _import_error = exc


def dummy_awn_inference(x: np.ndarray, n_classes: int = 11, seed: Optional[int] = 0) -> np.ndarray:
    """Validate the [N, 2, T] input shape and return random logits (numpy, no torch)."""
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"AWN expects input [N, 2, T], got {x.shape}")

    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(x.shape[0], n_classes)).astype(np.float32)
    print(f"[PLACEHOLDER] dummy_awn_inference: input={x.shape} -> logits={logits.shape}")
    return logits


class AWNModelAdapter:
    """
    Loads the real AWN model + checkpoint if torch is importable and the
    checkpoint file loads successfully; falls back to dummy_awn_inference
    otherwise (missing torch, missing/corrupt checkpoint, shape mismatch,
    etc.). Construct once and reuse across calls to avoid reloading the
    checkpoint repeatedly.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.backend_name = "dummy_awn_inference"
        self.status = "fallback"
        self.notes = ""

        if _real_AWN_cls is None:
            self.notes = (
                f"Real AWN import failed ({type(_import_error).__name__}: {_import_error}); "
                f"falling back to dummy_awn_inference. {_REAL_MODEL_SOURCE} requires torch, "
                "which is not installed in this phase -- see docs/integration_plan.md."
            )
            return

        try:
            import torch

            model = _real_AWN_cls(**_AWN_2016_10A_CFG).to(device)
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            self.model = model
            self.backend_name = _REAL_MODEL_SOURCE
            self.status = "ok"
            self.notes = f"Loaded real AWN from {_REAL_MODEL_SOURCE} with checkpoint '{checkpoint_path}'"
        except Exception as exc:  # noqa: BLE001 - torch present but checkpoint missing/bad/etc.
            self.model = None
            self.backend_name = "dummy_awn_inference"
            self.status = "fallback"
            self.notes = (
                f"Real AWN load failed ({type(exc).__name__}: {exc}) using checkpoint "
                f"'{checkpoint_path}'; falling back to dummy_awn_inference."
            )

    def infer(self, x: np.ndarray, n_classes: int = 11, seed: Optional[int] = 0) -> Tuple[np.ndarray, Dict[str, str]]:
        """x: [N, 2, T] float32. Returns (logits: [N, n_classes], meta)."""
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"AWN expects input [N, 2, T], got {x.shape}")

        if self.model is not None:
            try:
                import torch

                with torch.no_grad():
                    x_t = torch.from_numpy(x).to(self.device)
                    logit, _regu_sum = self.model(x_t)  # real forward returns (logit, regu_sum) tuple
                    logits = logit.detach().cpu().numpy().astype(np.float32)
                backend, status, notes = self.backend_name, "ok", self.notes
            except Exception as exc:  # noqa: BLE001 - real backend failed at call time
                logits = dummy_awn_inference(x, n_classes=n_classes, seed=seed)
                backend, status = "dummy_awn_inference", "fallback"
                notes = f"Real AWN forward failed at runtime ({type(exc).__name__}: {exc}); used numpy fallback."
        else:
            logits = dummy_awn_inference(x, n_classes=n_classes, seed=seed)
            backend, status, notes = self.backend_name, self.status, self.notes

        print(f"[awn_adapter] backend={backend} status={status} input={x.shape} logits={logits.shape}")
        return logits, {"awn_backend": backend, "awn_status": status, "awn_notes": notes}
