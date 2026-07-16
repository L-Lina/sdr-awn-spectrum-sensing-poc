"""
Placeholder + real adversarial attack adapter.

dummy_attack(...) is the numpy-only placeholder (kept as the default and as
the fallback target for AttackAdapter below).

AttackAdapter wraps the real attack path: Model01Wrapper from
external/adversarial-rf/util/adv_attack.py plus the third-party torchattacks
library, mirroring the construction pattern in
external/adversarial-rf/util/multi_attack_eval.py, e.g.:
    torchattacks.FGSM(wrapped_model, eps=eps)
    torchattacks.PGD(wrapped_model, eps=eps, alpha=eps/4, steps=steps)
    torchattacks.CW(wrapped_model, c=c, steps=steps, lr=lr)
per docs/integration_plan.md section 2.

A real gradient-based attack additionally needs a *real* (differentiable)
AWN model -- if AWNModelAdapter fell back to the numpy dummy (e.g. because
torch isn't installed), there is nothing to backprop through, so this
adapter also falls back in that case regardless of whether torchattacks
itself is importable.

Neither torch nor the third-party torchattacks package is installed in this
phase (packages are not to be installed) -- so real-attack construction is
expected to fail and fall back to dummy_attack, exactly like
topk_adapter.py / awn_adapter.py. This module never modifies external/AWN or
external/adversarial-rf; it only reads from the latter (adds its path to
sys.path for the duration of the import attempt).

Supported attack names (first version, per docs/integration_plan.md): none,
fgsm, pgd, cw.

x_clean is only unit-average-power normalized (src/sensing/normalize.py), not
clamped to [-1,1] -- roughly 12% of samples fall outside that range in
practice. A fixed (x+1)/2 mapping into torchattacks' assumed [0,1] domain
would therefore silently clip those samples via torchattacks' own
torch.clamp(..., 0, 1) before any attack-specific perturbation is even
applied, contaminating clean-vs-attacked diffs with clipping artifacts. This
adapter uses the existing per-segment min-max mapping already provided by
external/adversarial-rf/util/adv_attack.py (iq_to_ta_input_minmax /
ta_output_to_iq_minmax / Model01Wrapper.set_minmax) instead of the fixed-range
iq_to_ta_input / ta_output_to_iq, so every sample round-trips losslessly
through the [0,1] domain regardless of its original magnitude.

This checkpoint's raw logits are large enough (top1-top2 margins in the
hundreds) that float32 softmax saturates exactly, making CrossEntropyLoss's
gradient exactly zero regardless of eps. TemperatureLogitsWrapper below
divides logits by a positive temperature T *only* inside the attack's own
loss computation (attack_logits = logits / T) to de-saturate that gradient;
it is never used for clean/attacked/defended AWN inference anywhere else.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_ADVERSARIAL_RF_ROOT = Path(__file__).resolve().parents[2] / "external" / "adversarial-rf"
_REAL_ATTACK_SOURCE = "external/adversarial-rf/util/adv_attack.py:Model01Wrapper + torchattacks"

_NO_OP_ATTACKS = {"", "none"}
_SUPPORTED_ATTACKS = {"none", "fgsm", "pgd", "cw"}

_Model01Wrapper = None
_iq_to_ta_input_minmax = None
_ta_output_to_iq_minmax = None
_torchattacks = None
_nn = None
_import_error: Exception | None = None

try:
    _path_str = str(_ADVERSARIAL_RF_ROOT)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)
    from util.adv_attack import Model01Wrapper as _Model01Wrapper  # type: ignore
    from util.adv_attack import iq_to_ta_input_minmax as _iq_to_ta_input_minmax  # type: ignore
    from util.adv_attack import ta_output_to_iq_minmax as _ta_output_to_iq_minmax  # type: ignore
    import torchattacks as _torchattacks  # type: ignore
    import torch.nn as _nn  # type: ignore
except Exception as exc:  # noqa: BLE001 - torch/torchattacks missing, or any other import-time failure
    _import_error = exc


if _nn is not None:
    class TemperatureLogitsWrapper(_nn.Module):
        """
        Divides the wrapped model's logits by a fixed positive temperature.

        Used ONLY inside AttackAdapter.apply()'s internal attack-loss
        computation (constructing the torchattacks FGSM/PGD object) -- never
        for clean/attacked/defended AWN inference, which always calls
        self.wrapped_model / the real AWN model directly. Does not copy the
        wrapped model, does not detach, does not use no_grad -- the gradient
        graph flows through this wrapper exactly as it would through
        self.wrapped_model directly, just scaled by 1/temperature.
        """

        def __init__(self, wrapped_model, temperature: float) -> None:
            super().__init__()
            if temperature <= 0:
                raise ValueError(f"temperature must be positive, got {temperature}")
            self.wrapped_model = wrapped_model
            self.temperature = temperature

        def forward(self, x):
            return self.wrapped_model(x) / self.temperature
else:
    TemperatureLogitsWrapper = None  # torch unavailable -- apply() falls back to dummy_attack before this is ever used


def _validate_attack_name(attack: str) -> str:
    attack_name = (attack or "none").lower()
    if attack_name not in _SUPPORTED_ATTACKS:
        raise ValueError(f"Unsupported attack '{attack_name}' (supported: {sorted(_SUPPORTED_ATTACKS)})")
    return attack_name


def dummy_attack(x: np.ndarray, attack: str, epsilon: float = 0.02, seed: Optional[int] = 0) -> np.ndarray:
    """
    Apply a deterministic sign-noise perturbation as a stand-in for a real
    gradient-based attack. Restricted to the first-version attack list
    (none, fgsm, pgd, cw) -- the name only affects logging/eps in this
    placeholder, not a real attack algorithm.
    """
    attack_name = _validate_attack_name(attack)
    if attack_name in _NO_OP_ATTACKS:
        print("[PLACEHOLDER] dummy_attack: attack='none' -> no-op")
        return x

    rng = np.random.default_rng(seed)
    perturbation = (epsilon * rng.choice([-1.0, 1.0], size=x.shape)).astype(np.float32)
    x_adv = (x + perturbation).astype(np.float32)
    print(f"[PLACEHOLDER] dummy_attack: attack='{attack_name}' eps={epsilon} -> perturbed {x_adv.shape}")
    return x_adv


def _build_torchattacks(attack_name: str, wrapped_model, eps: float):
    if attack_name == "fgsm":
        return _torchattacks.FGSM(wrapped_model, eps=eps)
    if attack_name == "pgd":
        return _torchattacks.PGD(wrapped_model, eps=eps, alpha=eps / 4, steps=10)
    if attack_name == "cw":
        # Deliberately modest steps -- this phase only builds the interface,
        # not a full-scale attack run (see docs/integration_plan.md).
        return _torchattacks.CW(wrapped_model, c=1.0, steps=20, lr=0.01)
    raise ValueError(f"No real-attack builder for '{attack_name}'")


class AttackAdapter:
    """
    Uniform attack interface: apply(x, attack, eps) -> (x_adv, meta).

    Falls back to dummy_attack when torch/torchattacks aren't both
    importable, when no real (differentiable) AWN model is supplied, or on
    any runtime failure while constructing/running the real attack.
    """

    def __init__(self, awn_model=None, device: str = "cpu") -> None:
        """awn_model: the real torch nn.Module from AWNModelAdapter, or None if it's running dummy."""
        self.device = device
        self.wrapped_model = None
        self.backend_name = "dummy_attack"
        self.status = "fallback"
        self.notes = ""

        if _torchattacks is None or _Model01Wrapper is None:
            self.notes = (
                f"Real attack import failed ({type(_import_error).__name__}: {_import_error}); "
                f"falling back to dummy_attack. {_REAL_ATTACK_SOURCE} requires torch and the "
                "third-party torchattacks package, neither of which is installed in this phase "
                "-- see docs/integration_plan.md."
            )
            return

        if awn_model is None:
            self.notes = (
                "Real AWN model unavailable (AWNModelAdapter fell back to the numpy dummy), so "
                "there is no differentiable model to attack; falling back to dummy_attack."
            )
            return

        try:
            self.wrapped_model = _Model01Wrapper(awn_model).to(device)
            self.backend_name = _REAL_ATTACK_SOURCE
            self.status = "ok"
            self.notes = f"Loaded real attack path via {_REAL_ATTACK_SOURCE}"
        except Exception as exc:  # noqa: BLE001
            self.wrapped_model = None
            self.backend_name = "dummy_attack"
            self.status = "fallback"
            self.notes = f"Model01Wrapper construction failed ({type(exc).__name__}: {exc}); using dummy fallback."

    def apply(
        self,
        x: np.ndarray,
        attack: str,
        eps: float,
        temperature: float = 1.0,
        seed: Optional[int] = 0,
        diagnostics: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, str]]:
        """x: [N, 2, T] float32. Returns (x_adv, meta) with x_adv of the same shape.

        temperature: positive T dividing AWN logits inside the attack's own
        loss computation only (attack_logits = logits / T). T=1.0 reproduces
        prior behavior exactly. Clean/attacked/defended AWN inference
        elsewhere in the pipeline always uses raw, untouched logits.

        diagnostics: if True, run one extra autograd.grad pass (same attack
        model, same clean-prediction label) purely to report gradient
        nonzero-count/maxabs in the returned meta; never affects x_adv.
        """
        if temperature <= 0:
            raise ValueError(f"attack_temperature must be positive, got {temperature}")
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"AttackAdapter expects input [N, 2, T], got {x.shape}")
        attack_name = _validate_attack_name(attack)
        input_shape = x.shape
        attack_input_min = float(np.min(x))
        attack_input_max = float(np.max(x))

        if attack_name in _NO_OP_ATTACKS:
            # Bypass the min-max mapping AND the temperature wrapper
            # entirely so attack='none' returns the exact same array
            # bit-for-bit, regardless of what temperature was requested.
            print(f"[attack_adapter] attack='none' -> no-op (backend={self.backend_name})")
            return x, {
                "attack_backend": self.backend_name,
                "attack_status": "ok",
                "attack_notes": "attack='none' -> no-op",
                "attack_training_before": None,
                "attack_training_after": None,
                "attack_input_min": attack_input_min,
                "attack_input_max": attack_input_max,
                "attack_normalized_min": None,
                "attack_normalized_max": None,
                "attack_output_has_nan": bool(np.isnan(x).any()),
                "attack_output_has_inf": bool(np.isinf(x).any()),
                "attack_temperature": temperature,
                "attack_iq_linf_normalized": None,
                "attack_gradient_nonzero_count": None,
                "attack_gradient_total_count": None,
                "attack_gradient_maxabs": None,
            }

        normalized_min = None
        normalized_max = None
        iq_linf_normalized = None
        gradient_nonzero_count = None
        gradient_total_count = None
        gradient_maxabs = None

        if self.wrapped_model is not None:
            # Model01Wrapper is a fresh nn.Module and defaults to train mode
            # regardless of the real AWN submodule's eval() state, so
            # torchattacks' internal restore-previous-mode logic (which reads
            # this flag) leaves the wrapper -- and the real AWN model it
            # wraps -- in train mode after the attack call. Record the
            # pre-attack state, let torchattacks switch modes freely while it
            # computes the attack, then force eval mode back unconditionally
            # in `finally` so later attacked/defended AWN inference in this
            # process is never corrupted by train-mode dropout/batchnorm
            # behavior.
            training_before = self.wrapped_model.training
            try:
                import torch

                x_t = torch.from_numpy(x).to(self.device)
                # Per-segment min-max mapping (not the fixed (x+1)/2 range
                # assumption) -- see module docstring. a/b are [N,1,1] and
                # broadcast over the channel + time dims of each segment.
                x_ta, a, b = _iq_to_ta_input_minmax(x_t)
                normalized_min = float(x_ta.min().item())
                normalized_max = float(x_ta.max().item())
                # Model01Wrapper.forward() must invert x01 -> IQ using these
                # same a/b before handing off to the real AWN model, or the
                # wrapped model would see the wrong scale entirely.
                self.wrapped_model.set_minmax(a, b)
                with torch.no_grad():
                    y_pred = self.wrapped_model(x_ta).argmax(dim=1)

                # Temperature scaling applies ONLY to the model torchattacks
                # sees for its internal loss -- self.wrapped_model itself
                # (and therefore clean/attacked/defended AWN inference
                # elsewhere in the pipeline) is never touched by this.
                attack_model = TemperatureLogitsWrapper(self.wrapped_model, temperature)
                atk = _build_torchattacks(attack_name, attack_model, eps)
                x_ta_adv = atk(x_ta, y_pred)
                iq_linf_normalized = (
                    (x_ta_adv - x_ta).abs().amax(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.float32)
                )
                x_adv_t = _ta_output_to_iq_minmax(x_ta_adv, a, b)
                x_adv = x_adv_t.detach().cpu().numpy().astype(np.float32)
                backend, status, notes = self.backend_name, "ok", self.notes

                if diagnostics:
                    try:
                        import torch.nn as nn

                        # torchattacks' own mode bookkeeping can leave the
                        # model in train mode right here (same mechanism the
                        # finally block below corrects) -- force eval first
                        # so this diagnostic gradient reflects the same
                        # eval-mode conditions the real attack computed
                        # under, not train-mode noise.
                        self.wrapped_model.eval()
                        x_ta_grad = x_ta.clone().detach().requires_grad_(True)
                        diag_out = attack_model(x_ta_grad)
                        diag_loss = nn.CrossEntropyLoss()(diag_out, y_pred)
                        diag_grad = torch.autograd.grad(
                            diag_loss, x_ta_grad, retain_graph=False, create_graph=False
                        )[0]
                        gradient_nonzero_count = (
                            (diag_grad != 0).sum(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.int64)
                        )
                        gradient_total_count = np.full(x.shape[0], diag_grad[0].numel(), dtype=np.int64)
                        gradient_maxabs = (
                            diag_grad.abs().amax(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.float32)
                        )
                        del diag_grad, diag_out, diag_loss, x_ta_grad
                    except Exception as diag_exc:  # noqa: BLE001 - diagnostics must never break the real attack output
                        print(
                            f"[attack_adapter] diagnostics pass failed "
                            f"({type(diag_exc).__name__}: {diag_exc}); gradient stats left as None, "
                            "real attack output unaffected."
                        )
                        gradient_nonzero_count = None
                        gradient_total_count = None
                        gradient_maxabs = None
            except Exception as exc:  # noqa: BLE001 - real backend failed at call time
                x_adv = dummy_attack(x, attack=attack_name, epsilon=eps, seed=seed)
                backend = "dummy_attack"
                status = "fallback"
                notes = f"Real attack call failed at runtime ({type(exc).__name__}: {exc}); used numpy fallback."
            finally:
                if training_before:
                    print("[attack_adapter] warning: wrapped model was already in train mode before this call")
                self.wrapped_model.eval()
                self.wrapped_model.clear_minmax()
            training_after = self.wrapped_model.training
        else:
            x_adv = dummy_attack(x, attack=attack_name, epsilon=eps, seed=seed)
            backend, status, notes = self.backend_name, self.status, self.notes
            training_before = None
            training_after = None

        if x_adv.shape != input_shape:
            raise RuntimeError(f"AttackAdapter output shape {x_adv.shape} != input shape {input_shape}")

        print(f"[attack_adapter] backend={backend} status={status} input={input_shape} output={x_adv.shape}")
        return x_adv, {
            "attack_backend": backend,
            "attack_status": status,
            "attack_notes": notes,
            "attack_training_before": training_before,
            "attack_training_after": training_after,
            "attack_input_min": attack_input_min,
            "attack_input_max": attack_input_max,
            "attack_normalized_min": normalized_min,
            "attack_normalized_max": normalized_max,
            "attack_output_has_nan": bool(np.isnan(x_adv).any()),
            "attack_output_has_inf": bool(np.isinf(x_adv).any()),
            "attack_temperature": temperature,
            "attack_iq_linf_normalized": iq_linf_normalized,
            "attack_gradient_nonzero_count": gradient_nonzero_count,
            "attack_gradient_total_count": gradient_total_count,
            "attack_gradient_maxabs": gradient_maxabs,
        }
