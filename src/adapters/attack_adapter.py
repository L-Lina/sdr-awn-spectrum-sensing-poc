"""
Placeholder + real adversarial attack adapter.

dummy_attack(...) is the numpy-only placeholder (kept as the default and as
the fallback target for AttackAdapter below).

AttackAdapter wraps the real attack path: Model01Wrapper from
external/adversarial-rf/util/adv_attack.py plus the third-party torchattacks
library, mirroring the construction pattern in
external/adversarial-rf/util/multi_attack_eval.py.

A real gradient-based attack additionally needs a *real* (differentiable)
AWN model -- if AWNModelAdapter fell back to the numpy dummy (e.g. because
torch isn't installed), there is nothing to backprop through, so this
adapter also falls back in that case regardless of whether torchattacks
itself is importable.

This module never modifies external/AWN or external/adversarial-rf; it only
reads from the latter (adds its path to sys.path for the duration of the
import attempt).

Supported attack names: none, fgsm, pgd, cw (original, backward-compatible
set -- every prior formal round, docs/formal_experiment_plan.md phases 0-4,
depends on these three's exact existing default parameter values, which are
UNCHANGED below), plus (this round) bim, mifgsm, difgsm, vmifgsm, vnifgsm,
rfgsm, tpgd, deepfool, fab, square, apgd, apgdt, autoattack, ead (=
torchattacks.EADL1, see _ATTACK_CLASS_MAP). Every constructor kwarg name and
default below was read directly from the INSTALLED torchattacks==3.5.1
package via inspect.signature(), not assumed from memory or from
external/adversarial-rf's own (sometimes older-version-targeting) call
sites -- see docs/ATTACK_NAME_MAPPING.md for the full per-attack
parameter/targeted-mode/input-constraint table and the introspection
transcript it was built from.

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
through the [0,1] domain regardless of its original magnitude. This mapping
is shared by ALL attacks below (not just fgsm/pgd/cw) -- every torchattacks
class ultimately calls the wrapped model's forward() on whatever tensor it
is given, and the mapping/wrapper boundary is identical for all of them.

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

from src.adapters.iq_difgsm import IQDIFGSM
from src.utils.config import (
    require_nonneg_finite_float,
    require_positive_finite_float,
    require_positive_int,
)

_ADVERSARIAL_RF_ROOT = Path(__file__).resolve().parents[2] / "external" / "adversarial-rf"
_REAL_ATTACK_SOURCE = "external/adversarial-rf/util/adv_attack.py:Model01Wrapper + torchattacks"

_NO_OP_ATTACKS = {"", "none"}

# AWN's real class count (external/adversarial-rf/models/model.py:AWN,
# num_classes=11 in src/adapters/awn_adapter.py:_AWN_2016_10A_CFG) -- FAB/
# APGDT/AutoAttack default to torchattacks' own n_classes=10 (a CIFAR-10
# convention) if not told otherwise, which would be silently wrong for this
# 11-class problem. Used as the default (not hardcoded-forced) below --
# still overridable via an explicit n_classes in attack_params, e.g. for a
# deliberate mismatch test.
_AWN_N_CLASSES = 11

# Every constructor kwarg name below, per attack, verified this round via
# inspect.signature(torchattacks.<Class>.__init__) against the INSTALLED
# torchattacks==3.5.1 -- see docs/ATTACK_NAME_MAPPING.md for the raw
# introspection transcript. "eps" is included only for attacks whose real
# signature has it (deepfool/cw/ead do not).
_ATTACK_ACCEPTED_PARAMS: Dict[str, set] = {
    "fgsm": {"eps"},
    "bim": {"eps", "alpha", "steps"},
    "pgd": {"eps", "alpha", "steps", "random_start"},
    "mifgsm": {"eps", "alpha", "steps", "decay"},
    # difgsm uses the IQ-native src/adapters/iq_difgsm.py:IQDIFGSM, not
    # torchattacks.DIFGSM (see _ATTACK_CLASS_MAP and docs/ATTACK_NAME_MAPPING.md
    # for why) -- "seed" is IQDIFGSM-only (no torchattacks attack accepts it
    # as a constructor kwarg; reuses the same CLI flag/config field --attack-
    # internal-seed already defined for fab/square/apgd/apgdt/autoattack).
    "difgsm": {"eps", "alpha", "steps", "decay", "resize_rate", "diversity_prob", "random_start", "seed"},
    "vmifgsm": {"eps", "alpha", "steps", "decay", "N", "beta"},
    "vnifgsm": {"eps", "alpha", "steps", "decay", "N", "beta"},
    "rfgsm": {"eps", "alpha", "steps"},
    "tpgd": {"eps", "alpha", "steps"},
    "cw": {"c", "kappa", "steps", "lr"},
    "deepfool": {"steps", "overshoot"},
    "fab": {"norm", "eps", "steps", "n_restarts", "alpha_max", "eta", "beta", "seed", "multi_targeted", "n_classes"},
    "square": {"norm", "eps", "n_queries", "n_restarts", "p_init", "loss", "resc_schedule", "seed"},
    "apgd": {"norm", "eps", "steps", "n_restarts", "seed", "loss", "eot_iter", "rho"},
    "apgdt": {"norm", "eps", "steps", "n_restarts", "seed", "eot_iter", "rho", "n_classes"},
    "autoattack": {"norm", "eps", "version", "n_classes", "seed"},
    "ead": {"kappa", "lr", "binary_search_steps", "max_iterations", "abort_early", "initial_const", "beta"},
}
_SUPPORTED_ATTACKS = {"none"} | set(_ATTACK_ACCEPTED_PARAMS)

# forward(images, labels) -- confirmed this round via inspect.getsource() on
# every class's .forward(): every attack except tpgd requires real labels
# (this adapter always passes the model's OWN clean prediction, y_pred, as
# an untargeted "move away from this" label -- the same convention the
# pre-existing fgsm/pgd/cw branches already used). tpgd's forward signature
# is (images, labels=None) and never references labels internally (pure
# KL-divergence between clean and perturbed output distributions) -- passing
# y_pred to it is harmless (accepted positionally, then ignored), so no
# special-casing is needed at the call site.
_ATTACK_NEEDS_LABELS = {name: (name != "tpgd") for name in _SUPPORTED_ATTACKS if name != "none"}

# Empirically verified this round via atk.supported_mode on a constructed
# instance of each class (torchattacks==3.5.1) -- not assumed from
# documentation. 'targeted' present means .set_mode_targeted_by_label()
# etc. work; this adapter only ever uses 'default' (untargeted) mode this
# round (matches every prior formal experiment's convention) -- this table
# is for docs/ATTACK_NAME_MAPPING.md's record, not exercised by targeted
# calls here.
_ATTACK_TARGETED_SUPPORT: Dict[str, list] = {
    "fgsm": ["default", "targeted"], "bim": ["default", "targeted"], "pgd": ["default", "targeted"],
    "mifgsm": ["default", "targeted"],
    # difgsm: IQDIFGSM (src/adapters/iq_difgsm.py, IQ-native reimplementation,
    # NOT torchattacks.DIFGSM -- see docs/ATTACK_NAME_MAPPING.md) only
    # implements untargeted mode; no set_mode_targeted_* method exists on it.
    "difgsm": ["default"],
    "vmifgsm": ["default", "targeted"],
    "vnifgsm": ["default", "targeted"], "rfgsm": ["default", "targeted"], "tpgd": ["default"],
    "cw": ["default", "targeted"], "deepfool": ["default"], "fab": ["default", "targeted"],
    "square": ["default", "targeted"], "apgd": ["default"], "apgdt": ["default (inherently targeted internally)"],
    "autoattack": ["default"], "ead": ["default", "targeted"],
}

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

_ATTACK_CLASS_MAP: Dict[str, object] = {}
if _torchattacks is not None:
    _ATTACK_CLASS_MAP = {
        "fgsm": _torchattacks.FGSM, "bim": _torchattacks.BIM, "pgd": _torchattacks.PGD,
        "mifgsm": _torchattacks.MIFGSM, "difgsm": IQDIFGSM, "vmifgsm": _torchattacks.VMIFGSM,
        "vnifgsm": _torchattacks.VNIFGSM, "rfgsm": _torchattacks.RFGSM, "tpgd": _torchattacks.TPGD,
        "cw": _torchattacks.CW, "deepfool": _torchattacks.DeepFool, "fab": _torchattacks.FAB,
        "square": _torchattacks.Square, "apgd": _torchattacks.APGD, "apgdt": _torchattacks.APGDT,
        "autoattack": _torchattacks.AutoAttack,
        "ead": _torchattacks.EADL1,  # this repo's "ead" name aliases EADL1 (the old project's adv_train.py
                                      # treats 'eadl1'/'eaden' as two separate names; EADEN is reachable
                                      # here via attack_params={"_ead_variant": "eaden"}, see _build_torchattacks).
    }


if _nn is not None:
    class TemperatureLogitsWrapper(_nn.Module):
        """
        Divides the wrapped model's logits by a fixed positive temperature.

        Used ONLY inside AttackAdapter.apply()'s internal attack-loss
        computation (constructing the torchattacks attack object) -- never
        for clean/attacked/defended AWN inference, which always calls
        self.wrapped_model / the real AWN model directly. Does not copy the
        wrapped model, does not detach, does not use no_grad -- the gradient
        graph flows through this wrapper exactly as it would through
        self.wrapped_model directly, just scaled by 1/temperature.
        """

        def __init__(self, wrapped_model, temperature: float) -> None:
            super().__init__()
            require_positive_finite_float("attack_temperature", temperature)
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
    gradient-based attack. The perturbation itself is identical regardless
    of which supported attack name was requested (the name only affects
    logging/eps in this placeholder, never a real attack-specific algorithm).
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


def _build_torchattacks(
    attack_name: str, wrapped_model, eps: float,
    cw_c: float = 1.0, cw_steps: int = 20, cw_lr: float = 0.01,
    attack_params: Optional[Dict[str, object]] = None,
):
    """
    fgsm/pgd/cw: EXACT pre-existing hardcoded defaults, unchanged, so every
    prior formal round's behavior is bit-for-bit reproduced when
    attack_params is None/empty (pgd: alpha=eps/4, steps=10; cw: c/steps/lr
    from cw_c/cw_steps/cw_lr) -- attack_params may still override individual
    values for these three (e.g. an explicit {"steps": 20} for pgd), but an
    omitted attack_params changes nothing about their behavior.

    All other attacks (bim, mifgsm, difgsm, vmifgsm, vnifgsm, rfgsm, tpgd,
    deepfool, fab, square, apgd, apgdt, autoattack, ead): generic dispatch
    via _ATTACK_ACCEPTED_PARAMS -- only keys that class's real constructor
    accepts are passed; a key absent from attack_params (or explicitly None)
    is simply never passed, letting torchattacks' OWN installed default
    apply (verified via inspect.signature this round, not duplicated/
    hardcoded here, so it can never drift from the installed version).
    fab/apgdt/autoattack default n_classes to _AWN_N_CLASSES (11) unless
    attack_params explicitly overrides it -- torchattacks' own default of 10
    is a CIFAR-10 convention and would be silently wrong for this 11-class
    problem.
    """
    params = dict(attack_params or {})
    ead_variant = params.pop("_ead_variant", "eadl1")

    if attack_name == "fgsm":
        return _torchattacks.FGSM(wrapped_model, eps=params.get("eps", eps))
    if attack_name == "pgd":
        return _torchattacks.PGD(
            wrapped_model, eps=params.get("eps", eps),
            alpha=params.get("alpha", eps / 4), steps=params.get("steps", 10),
            **({"random_start": params["random_start"]} if params.get("random_start") is not None else {}),
        )
    if attack_name == "cw":
        return _torchattacks.CW(
            wrapped_model, c=params.get("c", cw_c), kappa=params.get("kappa", 0),
            steps=params.get("steps", cw_steps), lr=params.get("lr", cw_lr),
        )

    if attack_name not in _ATTACK_CLASS_MAP:
        raise ValueError(f"No real-attack builder for '{attack_name}'")

    accepted = _ATTACK_ACCEPTED_PARAMS[attack_name]
    kwargs = {}
    if "eps" in accepted:
        kwargs["eps"] = params.get("eps", eps)
    for key, value in params.items():
        if key in accepted and key != "eps" and value is not None:
            kwargs[key] = value
    if attack_name in ("fab", "apgdt", "autoattack") and "n_classes" not in kwargs:
        kwargs["n_classes"] = _AWN_N_CLASSES

    cls = _torchattacks.EADEN if (attack_name == "ead" and ead_variant == "eaden") else _ATTACK_CLASS_MAP[attack_name]
    return cls(wrapped_model, **kwargs)


class AttackAdapter:
    """
    Uniform attack interface: apply(x, attack, eps) -> (x_adv, meta).

    Falls back to dummy_attack when torch/torchattacks aren't both
    importable, when no real (differentiable) AWN model is supplied, or on
    any runtime failure while constructing/running the real attack. Never
    reports a fallback as a success -- attack_status/attack_backend in the
    returned meta always reflect what actually ran.
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
        cw_c: float = 1.0,
        cw_steps: int = 20,
        cw_lr: float = 0.01,
        attack_params: Optional[Dict[str, object]] = None,
    ) -> Tuple[np.ndarray, Dict[str, str]]:
        """x: [N, 2, T] float32. Returns (x_adv, meta) with x_adv of the same shape.

        temperature: positive T dividing AWN logits inside the attack's own
        loss computation only (attack_logits = logits / T). T=1.0 reproduces
        prior behavior exactly. Clean/attacked/defended AWN inference
        elsewhere in the pipeline always uses raw, untouched logits.

        diagnostics: if True, run one extra autograd.grad pass (same attack
        model, same clean-prediction label) purely to report gradient
        nonzero-count/maxabs in the returned meta; never affects x_adv.

        cw_c/cw_steps/cw_lr: CW-only strength knobs (unchanged from prior
        rounds). attack_params: generic dict of extra per-attack kwargs (see
        _ATTACK_ACCEPTED_PARAMS / docs/ATTACK_NAME_MAPPING.md) for every
        attack beyond fgsm/pgd/cw's original three-attack surface; also
        usable to override individual fgsm/pgd/cw defaults if explicitly
        needed (never changes their default behavior when omitted).
        """
        require_positive_finite_float("attack_temperature", temperature)
        require_nonneg_finite_float("attack_eps", eps)
        require_positive_finite_float("cw_c", cw_c)
        require_positive_int("cw_steps", cw_steps)
        require_positive_finite_float("cw_lr", cw_lr)
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"AttackAdapter expects input [N, 2, T], got {x.shape}")
        attack_name = _validate_attack_name(attack)
        input_shape = x.shape
        input_dtype = x.dtype
        attack_input_min = float(np.min(x))
        attack_input_max = float(np.max(x))

        if attack_name in _NO_OP_ATTACKS:
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
                "cw_c": cw_c, "cw_steps": cw_steps, "cw_lr": cw_lr,
            }

        normalized_min = None
        normalized_max = None
        iq_linf_normalized = None
        gradient_nonzero_count = None
        gradient_total_count = None
        gradient_maxabs = None

        if self.wrapped_model is not None:
            training_before = self.wrapped_model.training
            orig_requires_grad = [p.requires_grad for p in self.wrapped_model.parameters()]
            orig_param_devices = [p.device for p in self.wrapped_model.parameters()]
            try:
                import torch

                x_t = torch.from_numpy(x).to(self.device)
                x_ta, a, b = _iq_to_ta_input_minmax(x_t)
                normalized_min = float(x_ta.min().item())
                normalized_max = float(x_ta.max().item())
                self.wrapped_model.set_minmax(a, b)
                with torch.no_grad():
                    y_pred = self.wrapped_model(x_ta).argmax(dim=1)

                attack_model = TemperatureLogitsWrapper(self.wrapped_model, temperature)
                atk = _build_torchattacks(
                    attack_name, attack_model, eps,
                    cw_c=cw_c, cw_steps=cw_steps, cw_lr=cw_lr, attack_params=attack_params,
                )
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
                # Full state restoration (item 6 of this round's instruction):
                # model.eval(), original train/eval flag (recorded, not
                # silently kept), requires_grad per-parameter, and device --
                # unconditionally, whether the attack succeeded or the
                # except branch above already ran.
                if training_before:
                    print("[attack_adapter] warning: wrapped model was already in train mode before this call")
                self.wrapped_model.eval()
                self.wrapped_model.clear_minmax()
                for p, req_grad, dev in zip(self.wrapped_model.parameters(), orig_requires_grad, orig_param_devices):
                    p.requires_grad_(req_grad)
                    if p.device != dev:
                        p.data = p.data.to(dev)
            training_after = self.wrapped_model.training
        else:
            x_adv = dummy_attack(x, attack=attack_name, epsilon=eps, seed=seed)
            backend, status, notes = self.backend_name, self.status, self.notes
            training_before = None
            training_after = None

        if x_adv.shape != input_shape:
            raise RuntimeError(f"AttackAdapter output shape {x_adv.shape} != input shape {input_shape}")
        if x_adv.dtype != input_dtype:
            raise RuntimeError(f"AttackAdapter output dtype {x_adv.dtype} != input dtype {input_dtype}")

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
            "cw_c": cw_c, "cw_steps": cw_steps, "cw_lr": cw_lr,
        }
