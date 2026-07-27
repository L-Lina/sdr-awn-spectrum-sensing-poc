"""
IQ-compatible DIFGSM (Diverse-Input MI-FGSM), operating natively on
[N, 2, T] real-valued IQ tensors -- NOT a reshape-into-image trick.

torchattacks.DIFGSM's own input_diversity() assumes a roughly-square 2D
image (it resizes/pads BOTH spatial dims of a [N,C,H,W] tensor) and, when
fed this repo's [N,2,T,1] tensor convention (Model01Wrapper's image-shaped
input contract), unconditionally computes a resize target on the tensor's
LAST dimension -- which is a singleton (W=1) here, not a real spatial
axis -- and crashes (see docs/ATTACK_NAME_MAPPING.md's difgsm section for
the full root-cause trace). That is an architectural mismatch, not a
parameter-tuning problem, and installed torchattacks is never modified to
work around it.

This module reimplements DIFGSM's actual algorithm (iterative FGSM step +
momentum accumulation + a diversity transform applied before each gradient
step + Linf epsilon-ball projection + valid-range clipping) directly
against the [N,2,T] shape, with the diversity transform confined to the
time axis only:
  - a single random target length is drawn per forward() call (not per
    channel), then torch.nn.functional.interpolate(..., mode="linear") is
    applied to the WHOLE [N,2,T] tensor at once -- interpolate's 1D
    resampling treats dim=1 (size 2, I/Q) exactly like any other channel
    dimension and applies the identical position-wise kernel to every
    channel, so I and Q are transformed identically by construction, not
    by extra bookkeeping.
  - the same random left/right padding (torch.nn.functional.pad on the
    last dim) restores the original length T, again applied to the whole
    tensor at once, so both channels get the same offset.
  - batch size and channel count are untouched by either operation (both
    only ever act on dim=-1).

Does not import or modify external/adversarial-rf's torchattacks
installation, external/AWN, or external/adversarial-rf -- pure PyTorch
(torch.nn.functional), reusing only the already-established
Model01Wrapper/TemperatureLogitsWrapper calling convention from
src/adapters/attack_adapter.py (this class's `model` constructor argument
is expected to be exactly that wrapper, called as `model(x)` -> logits,
identical to every torchattacks-based attack already wired in).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class IQDIFGSM:
    """
    IQ-native reimplementation of DIFGSM (Xie et al., "Improving
    Transferability of Adversarial Examples with Input Diversity"),
    operating on [N, 2, T] (or [N, 2, T, 1], squeezed internally) tensors
    in the [0, 1] domain (the same per-segment min-max-mapped domain every
    other attack in this repo's AttackAdapter operates in).

    Not an nn.Module -- deliberately a plain object exposing only
    __call__(images, labels) -> adv_images, matching exactly the subset of
    torchattacks' Attack interface AttackAdapter.apply() actually uses, so
    it drops into _ATTACK_CLASS_MAP / _build_torchattacks's existing
    generic dispatch with zero special-casing at the call site.

    Every random draw (diversity-transform target length, padding offset,
    the diversity_prob coin flip, and the optional random-start noise)
    goes through a single seeded torch.Generator private to this instance
    -- never the global torch RNG -- so constructing two instances with the
    same seed and calling them on the same input reproduces the identical
    adversarial output, without disturbing or being disturbed by any other
    randomness elsewhere in the pipeline (attack seeding, sensing, etc).
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 8 / 255,
        alpha: float = 2 / 255,
        steps: int = 10,
        decay: float = 1.0,
        resize_rate: float = 0.9,
        diversity_prob: float = 0.5,
        random_start: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if not (0.0 < resize_rate <= 1.0):
            raise ValueError(
                f"IQDIFGSM resize_rate must be in (0, 1] (shrink-then-pad convention "
                f"only -- see module docstring), got {resize_rate}"
            )
        if not (0.0 <= diversity_prob <= 1.0):
            raise ValueError(f"IQDIFGSM diversity_prob must be in [0, 1], got {diversity_prob}")
        self.model = model
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.decay = float(decay)
        self.resize_rate = float(resize_rate)
        self.diversity_prob = float(diversity_prob)
        self.random_start = bool(random_start)
        self.seed = seed
        self.generator = torch.Generator(device="cpu")
        if seed is not None:
            self.generator.manual_seed(int(seed))
        # torchattacks-interface compatibility (AttackAdapter/docs never
        # invoke targeted mode for this attack, but exposing the same
        # attribute name avoids surprising anyone introspecting atk.*).
        self.supported_mode = ["default"]

    def input_diversity(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, 2, T]. Returns a tensor of the SAME shape [N, 2, T].

        With probability (1 - diversity_prob), returns x unchanged. With
        probability diversity_prob: draws one random target length
        rnd_t in [int(T*resize_rate), T], linearly resamples the whole
        [N,2,T] tensor to length rnd_t (identical transform for both
        channels -- interpolate never treats dim=1 specially), then pads
        back to T with a random left/right split (again identical for
        both channels, since F.pad on dim=-1 applies to the whole tensor).
        """
        n, c, t = x.shape
        low = max(1, int(t * self.resize_rate))
        high = t
        if low >= high:
            rnd_t = t
        else:
            rnd_t = int(torch.randint(low, high, (1,), generator=self.generator).item())

        if rnd_t == t:
            resized = x
        else:
            resized = F.interpolate(x, size=rnd_t, mode="linear", align_corners=False)

        pad_total = t - rnd_t
        if pad_total > 0:
            pad_left = int(torch.randint(0, pad_total + 1, (1,), generator=self.generator).item())
            pad_right = pad_total - pad_left
            padded = F.pad(resized, (pad_left, pad_right), mode="constant", value=0.0)
        else:
            padded = resized

        assert padded.shape == (n, c, t), f"IQDIFGSM.input_diversity shape drift: {padded.shape} != {(n, c, t)}"

        coin = torch.rand(1, generator=self.generator).item()
        return padded if coin < self.diversity_prob else x

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.forward(images, labels)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        had_trailing_dim = images.dim() == 4
        x0 = images.clone().detach()
        if had_trailing_dim:
            x0 = x0.squeeze(-1)
        if x0.dim() != 3 or x0.shape[1] != 2:
            raise ValueError(f"IQDIFGSM expects [N,2,T] or [N,2,T,1], got {tuple(images.shape)}")
        labels = labels.clone().detach()

        momentum = torch.zeros_like(x0)
        if self.random_start:
            noise = torch.empty_like(x0).uniform_(-self.eps, self.eps, generator=self.generator)
            adv = torch.clamp(x0 + noise, min=0.0, max=1.0).detach()
        else:
            adv = x0.clone().detach()

        loss_fn = nn.CrossEntropyLoss()
        for _ in range(self.steps):
            adv.requires_grad_(True)
            diversified = self.input_diversity(adv)
            model_input = diversified.unsqueeze(-1) if had_trailing_dim else diversified
            logits = self.model(model_input)
            loss = loss_fn(logits, labels)
            grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]

            grad_norm = grad.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
            grad = grad / grad_norm
            grad = grad + momentum * self.decay
            momentum = grad

            adv = adv.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv - x0, min=-self.eps, max=self.eps)
            adv = torch.clamp(x0 + delta, min=0.0, max=1.0).detach()

        return adv.unsqueeze(-1) if had_trailing_dim else adv
