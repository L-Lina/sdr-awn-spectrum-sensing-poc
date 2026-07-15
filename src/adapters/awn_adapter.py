"""
Placeholder AWN inference adapter.

TODO(phase 2+): swap for the real model via external/adversarial-rf's
create_AWN_model(cfg) + torch.load checkpoint (e.g. external/adversarial-rf/2016.10a_AWN.pkl),
per docs/integration_plan.md section 2. Remember the real model's forward()
returns a (logit, regu_sum) tuple, not bare logits.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def dummy_awn_inference(x: np.ndarray, n_classes: int = 11, seed: Optional[int] = 0) -> np.ndarray:
    """Validate the [N, 2, T] input shape and return random logits (numpy, no torch)."""
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"AWN expects input [N, 2, T], got {x.shape}")

    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(x.shape[0], n_classes)).astype(np.float32)
    print(f"[PLACEHOLDER] dummy_awn_inference: input={x.shape} -> logits={logits.shape}")
    return logits
