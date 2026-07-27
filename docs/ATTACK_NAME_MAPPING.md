# Attack Name Mapping

Ground truth for `src/adapters/attack_adapter.py`'s attack registry. Every
constructor signature, label requirement, and targeted-mode entry below was
verified this round by directly introspecting the INSTALLED
`torchattacks==3.5.1` package (`inspect.signature`, `inspect.getsource` on
each class's `.forward()`, and `atk.supported_mode` on a constructed
instance) -- not assumed from memory or from
`external/adversarial-rf`'s own call sites, which sometimes target an older
torchattacks version with different defaults. `external/adversarial-rf` was
still used as the SOURCE for which attacks to port and roughly how they were
called (`util/multi_attack_eval.py`, `util/adv_eval.py`) -- see
`docs/ATTACK_COMPATIBILITY_WORKLIST.md` for that cross-reference.

All attacks share the same input pipeline: `AttackAdapter` maps the `[N,2,T]`
AWN-input tensor into `[0,1]` via `external/adversarial-rf/util/
adv_attack.py`'s per-segment min-max mapping (`iq_to_ta_input_minmax`),
constructs the requested `torchattacks` object around a
`TemperatureLogitsWrapper(Model01Wrapper(awn_model))`, calls
`atk(x_ta, y_pred)` (`y_pred` = the model's own clean prediction --
untargeted "move away from this" convention, unchanged from the original
fgsm/pgd/cw code), then maps the result back via
`ta_output_to_iq_minmax`. No attack has a different input-shape or
dtype constraint than this shared `[N,2,T]` float32 contract.

| CLI name | torchattacks class | Required/accepted params (installed 3.5.1 signature) | Targeted support (`supported_mode`, verified) | Label needed by `.forward()`? |
|---|---|---|---|---|
| `fgsm` | `FGSM` | `eps` | default, targeted | yes (own clean pred) |
| `bim` | `BIM` | `eps, alpha, steps` | default, targeted | yes |
| `pgd` | `PGD` | `eps, alpha, steps, random_start` | default, targeted | yes |
| `mifgsm` | `MIFGSM` | `eps, alpha, steps, decay` | default, targeted | yes |
| `difgsm` | `DIFGSM` | `eps, alpha, steps, decay, resize_rate, diversity_prob, random_start` | default, targeted | yes |
| `vmifgsm` | `VMIFGSM` | `eps, alpha, steps, decay, N, beta` | default, targeted | yes |
| `vnifgsm` | `VNIFGSM` | `eps, alpha, steps, decay, N, beta` | default, targeted | yes |
| `rfgsm` | `RFGSM` | `eps, alpha, steps` (no `random_start` -- randomness is unconditional/built in) | default, targeted | yes |
| `tpgd` | `TPGD` | `eps, alpha, steps` (no `random_start`) | **default only** | **no** -- `forward(images, labels=None)` never reads `labels` (pure KL-divergence between clean/perturbed output); `y_pred` is still passed positionally for calling-convention uniformity, harmlessly ignored |
| `cw` | `CW` | `c, kappa, steps, lr` (no `eps` -- confirmed no `eps` attribute exists on a constructed object, same finding as prior rounds) | default, targeted | yes |
| `deepfool` | `DeepFool` | `steps, overshoot` (no `eps` -- distance is whatever the nearest decision boundary requires, not a budget) | **default only** | yes |
| `fab` | `FAB` | `norm, eps, steps, n_restarts, alpha_max, eta, beta, seed, multi_targeted, n_classes` | default, targeted | yes |
| `square` | `Square` | `norm, eps, n_queries, n_restarts, p_init, loss, resc_schedule, seed` (no `n_classes`) | default, targeted | yes |
| `apgd` | `APGD` | `norm, eps, steps, n_restarts, seed, loss, eot_iter, rho` (no `n_classes`) | **default only** | yes |
| `apgdt` | `APGDT` | `norm, eps, steps, n_restarts, seed, eot_iter, rho, n_classes` (no `loss` -- always DLR-targeted internally) | **default only (inherently targeted internally -- the "T" IS the targeted variant; the toggle doesn't exist because it's always on)** | yes |
| `autoattack` | `AutoAttack` | `norm, eps, version, n_classes, seed` (no `steps` -- the ensemble's own sub-attacks each use their own fixed iteration counts) | **default only** (internally runs both untargeted and targeted sub-attacks, but the outer interface is untargeted) | yes |
| `ead` | `EADL1` (default) or `EADEN` via `attack_params={"_ead_variant": "eaden"}` | `kappa, lr, binary_search_steps, max_iterations, abort_early, initial_const, beta` (no `eps`, no `steps` -- uses `max_iterations` instead) | default, targeted | yes |

## `n_classes` correctness note

`fab`, `apgdt`, and `autoattack` all default to torchattacks' own
`n_classes=10` (a CIFAR-10 convention) if not told otherwise --
**silently wrong** for AWN's real 11-class problem
(`external/adversarial-rf/models/model.py:AWN`,
`num_classes=11` in `src/adapters/awn_adapter.py:_AWN_2016_10A_CFG`).
`AttackAdapter._build_torchattacks` defaults `n_classes` to **11** for
these three unless `attack_params` explicitly overrides it (CLI:
`--attack-n-classes`). Verified empirically this round: `FAB`/`APGDT`
store the derived `n_target_classes = n_classes - 1` (confirmed `== 10`
when `n_classes=11` is passed); `AutoAttack` stores `n_classes` directly
(confirmed `== 11`).

## `beta` name collision

`fab`'s `beta` (step-size growth factor, torchattacks default `0.9`) and
`ead`'s `beta` (L1-regularization weight, torchattacks default `0.001`)
are semantically unrelated parameters that happen to share a name. This
repo exposes both through the SAME `--attack-beta` CLI flag /
`attack_beta` config field -- `_build_torchattacks` only forwards it to
whichever attack's `_ATTACK_ACCEPTED_PARAMS` includes `"beta"`, so no
value ever reaches the wrong attack, but a caller setting `--attack-beta`
should be aware it means a different thing depending on which attack is
selected that run.

## `eps` applicability

Every attack above accepts `eps` in its real constructor **except**
`cw`, `deepfool`, and `ead` -- for those three, `--attack-eps` is parsed
and validated (`require_nonneg_finite_float`) but never forwarded to the
constructed attack object, exactly like the pre-existing `cw` behavior
from prior rounds.

## `difgsm`: NEEDS_CUSTOM_IMPLEMENTATION (found via smoke test, not fixable by parameter tuning)

`torchattacks.DIFGSM.input_diversity()` unconditionally computes
`img_resize = int(x.shape[-1] * resize_rate)` on the tensor's **last**
dimension before ever checking `diversity_prob`. `Model01Wrapper` reshapes
AWN input to `[N,2,T,1]` (4D, matching torchattacks' expected image
layout), so `x.shape[-1] == 1` always. `int(1 * resize_rate) == 0` for any
`resize_rate < 1` (the default is `0.9`); `resize_rate >= 1` instead makes
`torch.randint(low=1, high=1, ...)` raise directly. Either way,
`F.interpolate` ends up called with a target size of 0 and crashes
(`RuntimeError: Input and output sizes should be greater than 0, but got
input (H: 128, W: 1) output (H: 0, W: 0)`). **This is true for every
`resize_rate`/`diversity_prob` combination** -- confirmed by reading
`input_diversity`'s source directly, not inferred from one failing config.
Fixing this needs a custom `input_diversity()` override that only resizes
the `T=128` axis (leaving the trailing singleton dimension alone), or a
different tensor layout specifically for this attack -- not attempted this
round. `experiments/run_attack_compatibility_smoke.py` reports `difgsm` as
`NEEDS_CUSTOM_IMPLEMENTATION`, not `PASS` or a plain `FAIL`, since the
class instantiates fine and the failure is architectural, not a bad
parameter choice or a missing dependency.

## Backward compatibility

`fgsm`, `pgd`, `cw`'s default behavior (no `attack_params` overrides) is
byte-for-byte unchanged from every prior formal round: `pgd` still
defaults to `alpha=eps/4, steps=10`; `cw` still reads `cw_c`/`cw_steps`/
`cw_lr` (not the new generic `attack_kappa`/`attack_lr` fields, unless
explicitly passed via `attack_params`). Verified this round via direct
construction comparison (`_build_torchattacks("pgd", ..., eps=0.05)` ->
`atk.alpha == 0.05/4`, `atk.steps == 10`) before any smoke test was run.
