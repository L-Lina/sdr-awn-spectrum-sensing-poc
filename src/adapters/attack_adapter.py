"""
Placeholder adversarial attack adapter.

TODO(phase 3+): swap for a real attack via external/adversarial-rf's
util/adv_attack.py (Model01Wrapper + torchattacks, or the internal
cw_l2_attack), per docs/integration_plan.md section 2. That requires the
real (differentiable, torch) AWN model from awn_adapter.py to be wired in
first -- gradient-based attacks cannot run against the numpy placeholder.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_NO_OP_ATTACKS = {"", "none"}


def dummy_attack(x: np.ndarray, attack: str, epsilon: float = 0.02, seed: Optional[int] = 0) -> np.ndarray:
    """
    Apply a deterministic sign-noise perturbation as a stand-in for a real
    gradient-based attack. `attack` is accepted as a free-form label (e.g.
    "fgsm", "pgd") and only affects logging in this placeholder -- it does
    not yet select a real attack algorithm.
    """
    attack_name = (attack or "none").lower()
    if attack_name in _NO_OP_ATTACKS:
        print("[PLACEHOLDER] dummy_attack: attack='none' -> no-op")
        return x

    rng = np.random.default_rng(seed)
    perturbation = (epsilon * rng.choice([-1.0, 1.0], size=x.shape)).astype(np.float32)
    x_adv = (x + perturbation).astype(np.float32)
    print(f"[PLACEHOLDER] dummy_attack: attack='{attack_name}' eps={epsilon} -> perturbed {x_adv.shape}")
    return x_adv
