"""
Unit tests for src/adapters/iq_difgsm.py:IQDIFGSM -- the IQ-native DIFGSM
reimplementation. Standalone script (no pytest installed in this
environment), same PASS/FAIL-printing convention as
experiments/run_spectrum_sensing_utility.py --mode fairness-test. Exits
nonzero if any test fails.

Uses a tiny synthetic linear model (no real AWN checkpoint needed) so these
tests run in well under a second and exercise IQDIFGSM's own mechanics in
isolation from the rest of the pipeline -- the separate, real-checkpoint
verification is experiments/run_attack_compatibility_smoke.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adapters.iq_difgsm import IQDIFGSM  # noqa: E402


class TinyModel(nn.Module):
    """Deterministic-given-weights linear classifier over a flattened
    [2,128] (or [2,128,1]) input -> 11 classes (matching AWN's class
    count), fixed random weights (seeded) so gradients are non-degenerate
    but reproducible across test runs."""

    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(11, 2 * 128, generator=g) * 0.1, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(-1)
        flat = x.reshape(x.shape[0], -1)
        return flat @ self.weight.T


def check(name: str, cond: bool, detail: str = "") -> bool:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    return cond


def test_a_shape() -> bool:
    model = TinyModel()
    atk = IQDIFGSM(model, eps=0.05, alpha=0.01, steps=1, diversity_prob=1.0, seed=1)
    x = torch.rand(2, 2, 128)
    out_transform = atk.input_diversity(x)
    ok1 = check("A1. input_diversity([2,2,128]) -> [2,2,128]", tuple(out_transform.shape) == (2, 2, 128),
                detail=str(tuple(out_transform.shape)))

    x4 = torch.rand(2, 2, 128, 1)
    labels = torch.tensor([0, 1])
    out_full = atk(x4, labels)
    ok2 = check("A2. full forward([2,2,128,1]) -> [2,2,128,1]", tuple(out_full.shape) == (2, 2, 128, 1),
                detail=str(tuple(out_full.shape)))
    return ok1 and ok2


def test_b_channel_consistency() -> bool:
    """I and Q are transformed identically -- if both channels start
    identical, they must remain identical after input_diversity (would
    only hold if the SAME resize length + padding offset was applied to
    both, not independently randomized per channel)."""
    model = TinyModel()
    atk = IQDIFGSM(model, eps=0.05, alpha=0.01, steps=1, diversity_prob=1.0, resize_rate=0.5, seed=7)
    base = torch.rand(4, 1, 128)
    x = base.repeat(1, 2, 1)  # channel 0 == channel 1, exactly
    assert torch.equal(x[:, 0, :], x[:, 1, :])

    out = atk.input_diversity(x)
    identical_after = torch.equal(out[:, 0, :], out[:, 1, :])
    return check("B. I/Q channels transformed identically (same resize+pad, not per-channel random)",
                  identical_after)


def test_c_gradient() -> bool:
    model = TinyModel()
    atk = IQDIFGSM(model, eps=0.05, alpha=0.01, steps=1, diversity_prob=1.0, seed=3)
    x = torch.rand(2, 2, 128, 1)
    labels = torch.tensor([0, 1])

    x_leaf = x.clone().detach().requires_grad_(True)
    transformed = atk.input_diversity(x_leaf.squeeze(-1)).unsqueeze(-1)
    logits = model(transformed)
    loss = nn.CrossEntropyLoss()(logits, labels)
    grad = torch.autograd.grad(loss, x_leaf, retain_graph=False, create_graph=False)[0]

    ok1 = check("C1. gradient is not None", grad is not None)
    ok2 = check("C2. gradient is finite (no NaN/Inf)", bool(torch.isfinite(grad).all()))
    ok3 = check("C3. gradient has correct shape", tuple(grad.shape) == tuple(x.shape))
    return ok1 and ok2 and ok3


def test_d_deterministic() -> bool:
    model = TinyModel()
    x = torch.rand(3, 2, 128, 1)
    labels = torch.tensor([0, 1, 2])

    atk1 = IQDIFGSM(model, eps=0.05, alpha=0.01, steps=5, diversity_prob=0.9, seed=42)
    out1 = atk1(x, labels)
    atk2 = IQDIFGSM(model, eps=0.05, alpha=0.01, steps=5, diversity_prob=0.9, seed=42)
    out2 = atk2(x, labels)
    same_seed_same_result = torch.equal(out1, out2)
    ok1 = check("D1. same seed -> bit-identical output", same_seed_same_result)

    atk3 = IQDIFGSM(model, eps=0.05, alpha=0.01, steps=5, diversity_prob=0.9, seed=123)
    out3 = atk3(x, labels)
    different_seed_different_result = not torch.equal(out1, out3)
    ok2 = check("D2. different seed (diversity enabled) -> different output", different_seed_different_result)
    return ok1 and ok2


def test_e_constraints() -> bool:
    model = TinyModel()
    atk = IQDIFGSM(model, eps=0.05, alpha=0.02, steps=8, diversity_prob=0.7, seed=9)
    x = torch.rand(5, 2, 128, 1)
    labels = torch.tensor([0, 1, 2, 3, 4])
    out = atk(x, labels)

    ok1 = check("E1. no NaN in output", not bool(torch.isnan(out).any()))
    ok2 = check("E2. no Inf in output", not bool(torch.isinf(out).any()))
    linf = (out - x).abs().max().item()
    ok3 = check("E3. Linf(adv - clean) <= eps + 1e-6", linf <= atk.eps + 1e-6, detail=f"linf={linf:.6f} eps={atk.eps}")
    ok4 = check("E4. output shape == input shape", tuple(out.shape) == tuple(x.shape))
    ok5 = check("E5. output dtype == input dtype", out.dtype == x.dtype)
    ok6 = check("E6. output device == input device", out.device == x.device)
    ok7 = check("E7. output is in [0,1] valid range", bool((out >= 0).all() and (out <= 1).all()))
    return all([ok1, ok2, ok3, ok4, ok5, ok6, ok7])


def test_f_no_diversity_equivalence() -> bool:
    """diversity_prob=0 must still run the full iterative momentum attack
    (not degrade into a no-op or crash)."""
    model = TinyModel()
    atk = IQDIFGSM(model, eps=0.05, alpha=0.01, steps=5, decay=1.0, diversity_prob=0.0, seed=5)
    x = torch.rand(2, 2, 128, 1)
    labels = torch.tensor([0, 1])
    out = atk(x, labels)

    ok1 = check("F1. runs without error at diversity_prob=0", True)
    perturbation = (out - x).abs().max().item()
    ok2 = check("F2. produces a nonzero perturbation (iterative momentum attack still executes)",
                perturbation > 0, detail=f"linf={perturbation:.6f}")
    ok3 = check("F3. output finite and within eps at diversity_prob=0",
                bool(torch.isfinite(out).all()) and perturbation <= atk.eps + 1e-6)
    # Sanity: with diversity_prob=0, input_diversity() must always return
    # the input unchanged (coin < 0.0 is never true).
    unchanged = torch.equal(atk.input_diversity(x.squeeze(-1)), x.squeeze(-1))
    ok4 = check("F4. input_diversity() is a true no-op when diversity_prob=0", unchanged)
    return ok1 and ok2 and ok3 and ok4


def main() -> None:
    print("=" * 70)
    print("IQDIFGSM UNIT TESTS")
    print("=" * 70)
    results = {
        "A. shape": test_a_shape(),
        "B. channel-consistency": test_b_channel_consistency(),
        "C. gradient": test_c_gradient(),
        "D. deterministic": test_d_deterministic(),
        "E. constraint": test_e_constraints(),
        "F. no-diversity equivalence": test_f_no_diversity_equivalence(),
    }
    print("=" * 70)
    all_pass = all(results.values())
    for name, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
    print("=" * 70)
    print("ALL PASS" if all_pass else "FAILURES PRESENT")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
