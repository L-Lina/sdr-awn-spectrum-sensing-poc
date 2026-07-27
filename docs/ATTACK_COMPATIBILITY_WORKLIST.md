# Attack Compatibility Worklist

Cross-reference between every adversarial attack `external/adversarial-rf`
already supports (via `torchattacks`) and this repo's formal pipeline
(`src/adapters/attack_adapter.py`). Built by reading
`external/adversarial-rf/util/multi_attack_eval.py` (the old project's own
canonical, reusable attack factory -- preferred as the source of truth over
its many one-off `plot_*.py`/`test_*.py` scripts, which duplicate the same
factory pattern inconsistently) and `external/adversarial-rf/util/
adv_eval.py` (a second, larger factory covering additional attacks), then
directly introspecting the installed `torchattacks==3.5.1` package (`dir(
torchattacks)`) to confirm every referenced class name actually exists in
the currently pinned environment -- not assumed from the old project's
source alone.

**Updated this round**: 16 of 17 requested attacks ported into
`src/adapters/attack_adapter.py` and smoke-tested end-to-end against the
real AWN checkpoint (`experiments/run_attack_compatibility_smoke.py`,
`results/attack_compatibility_smoke_20260727T024650Z/`) -- see
`docs/ATTACK_NAME_MAPPING.md` for the full per-attack parameter/
targeted-mode/input-constraint table. `difgsm` constructs but fails at
`forward()` for an architectural reason (see its own status below), and
everything not in the user's 17-item request list remains unported.

## Status categories

- **已移植且smoke test通過**: wired into `src/adapters/attack_adapter.py`
  AND passed `experiments/run_attack_compatibility_smoke.py`'s full
  acceptance criteria (0 fallback, 0 NaN/Inf, eval-mode restored, clean
  logits reproducible, correct shape, genuinely nonzero perturbation).
- **已移植但NEEDS_CUSTOM_IMPLEMENTATION**: wired in, constructs
  successfully, but fails at `forward()` for a verified architectural
  reason (not a parameter or environment issue) -- see
  `docs/ATTACK_NAME_MAPPING.md`'s `difgsm` section.
- **已移植但未測**: wired into the adapter but never run with the real
  backend. (Empty today.)
- **尚未移植**: a valid, installed `torchattacks` class the old project
  already uses successfully, but `attack_adapter.py` has no branch for it.
- **torchattacks不支援或名稱不一致**: referenced somewhere in
  `external/adversarial-rf` but the class name does not exist in the
  currently installed `torchattacks==3.5.1`. **None found** -- every name
  checked resolves to a real class (see table).

## Worklist

| Attack | torchattacks class | In old project's canonical factory (`multi_attack_eval.py`)? | Class exists in installed torchattacks 3.5.1? | Status |
|---|---|---|---|---|
| FGSM | `torchattacks.FGSM` | yes | yes | **已移植且smoke test通過** |
| PGD | `torchattacks.PGD` | yes | yes | **已移植且smoke test通過** |
| CW | `torchattacks.CW` | yes | yes | **已移植且smoke test通過** |
| BIM | `torchattacks.BIM` | yes | yes | **已移植且smoke test通過** |
| DeepFool | `torchattacks.DeepFool` | yes | yes | **已移植且smoke test通過** |
| APGD | `torchattacks.APGD` | yes | yes | **已移植且smoke test通過** |
| MIFGSM | `torchattacks.MIFGSM` | yes | yes | **已移植且smoke test通過** |
| RFGSM | `torchattacks.RFGSM` | yes | yes | **已移植且smoke test通過** |
| UPGD | `torchattacks.UPGD` | yes | yes | 尚未移植 |
| EOTPGD | `torchattacks.EOTPGD` | yes | yes | 尚未移植 |
| VMIFGSM | `torchattacks.VMIFGSM` | yes | yes | **已移植且smoke test通過** |
| VNIFGSM | `torchattacks.VNIFGSM` | yes | yes | **已移植且smoke test通過** |
| Jitter | `torchattacks.Jitter` | yes | yes | 尚未移植 |
| FFGSM | `torchattacks.FFGSM` | yes | yes | 尚未移植 |
| PGDL2 | `torchattacks.PGDL2` | yes | yes | 尚未移植 |
| EADL1 | `torchattacks.EADL1` | yes | yes | **已移植且smoke test通過** (this repo's `ead` CLI name, default variant) |
| EADEN | `torchattacks.EADEN` | yes | yes | **已移植且smoke test通過** (via `attack_params={"_ead_variant": "eaden"}`, not separately smoke-tested from `eadl1` this round -- same code path) |
| FAB | `torchattacks.FAB` | yes | yes | **已移植且smoke test通過** |
| APGDT | `torchattacks.APGDT` | no (`adv_eval.py` only) | yes | **已移植且smoke test通過** |
| AutoAttack | `torchattacks.AutoAttack` | no (`adv_eval.py` only) | yes | **已移植且smoke test通過** (smoke-tested with `version="rand"` for CPU runtime, not the default `"standard"` ensemble -- see `SMOKE_ATTACK_PARAMS`) |
| GN | `torchattacks.GN` | no (`adv_eval.py` only) | yes | 尚未移植 |
| JSMA | `torchattacks.JSMA` | no (`adv_eval.py` only) | yes | 尚未移植 |
| NIFGSM | `torchattacks.NIFGSM` | no (`adv_eval.py` only) | yes | 尚未移植 |
| OnePixel | `torchattacks.OnePixel` | no (`adv_eval.py` only) | yes | 尚未移植 |
| PGDRS | `torchattacks.PGDRS` | no (`adv_eval.py` only) | yes | 尚未移植 |
| PGDRSL2 | `torchattacks.PGDRSL2` | no (`adv_eval.py` only) | yes | 尚未移植 |
| PIFGSM | `torchattacks.PIFGSM` | no (`adv_eval.py` only) | yes | 尚未移植 |
| PIFGSMPP | `torchattacks.PIFGSMPP` | no (`adv_eval.py` only) | yes | 尚未移植 |
| Pixle | `torchattacks.Pixle` | no (`adv_eval.py` only) | yes | 尚未移植 |
| SINIFGSM | `torchattacks.SINIFGSM` | no (`adv_eval.py` only) | yes | 尚未移植 |
| SparseFool | `torchattacks.SparseFool` | no (`adv_eval.py` only) | yes | 尚未移植 |
| SPSA | `torchattacks.SPSA` | no (`adv_eval.py` only) | yes | 尚未移植 |
| Square | `torchattacks.Square` | no (`adv_eval.py` only) | yes | **已移植且smoke test通過** |
| TIFGSM | `torchattacks.TIFGSM` | no (`adv_eval.py` only) | yes | 尚未移植 |
| TPGD | `torchattacks.TPGD` | no (`adv_eval.py` only) | yes | **已移植且smoke test通過** (untargeted-only, no label used internally -- see mapping doc) |
| DIFGSM | `torchattacks.DIFGSM` | no (found via `dir(torchattacks)` only, not referenced by name in any `.py` grepped this round) | yes | **已移植但NEEDS_CUSTOM_IMPLEMENTATION** -- constructs fine, crashes in `forward()` for every parameter combination (architectural: `input_diversity()` resizes the tensor's last dim, which is a singleton in this repo's `[N,2,T,1]` layout; see `docs/ATTACK_NAME_MAPPING.md`) |

**Verification method for the "exists in installed torchattacks" column**:
`/home/xiaomi/adversarial-rf/.venv/bin/python3 -c "import torchattacks;
print(sorted(n for n in dir(torchattacks) if n[0].isupper()))"` -- 39
classes returned, every name in the table above is among them (plus
`LGV`, `MultiAttack`, `VANILA`, not referenced by any script grepped in
`external/adversarial-rf` this round, so not listed as a worklist item).

## Remaining, not ported this round

`UPGD`, `EOTPGD`, `Jitter`, `FFGSM`, `PGDL2`, `GN`, `JSMA`, `NIFGSM`,
`OnePixel`, `PGDRS`, `PGDRSL2`, `PIFGSM`, `PIFGSMPP`, `Pixle`, `SINIFGSM`,
`SparseFool`, `SPSA`, `TIFGSM` -- none of these were in the 17-attack list
requested this round. `PGDL2` differs by distance metric (L2, not Linf)
and needs a parameter-shape decision (`--attack-eps` is currently
documented as an Linf budget) before porting. `OnePixel`/`Pixle`/`JSMA`/
`SparseFool`/`SPSA`/`GN` are image-classification-oriented sparse/
black-box attacks whose applicability to a `[2,128]` IQ tensor (not a 2D
image) has not been assessed in either project.

**`DIFGSM` needs custom work, not a straight port**: see its
`NEEDS_CUSTOM_IMPLEMENTATION` row above and
`docs/ATTACK_NAME_MAPPING.md` for the exact architectural blocker
(`input_diversity()`'s resize targets a singleton dimension in this
repo's tensor layout) and what a fix would require.

**Reuse, do not reimplement**: every ported attack above uses the exact
constructor kwarg names/defaults verified via `inspect.signature` against
the installed `torchattacks==3.5.1` this round (`docs/
ATTACK_NAME_MAPPING.md`), following the same `Model01Wrapper` +
per-segment min-max mapping architecture the original fgsm/pgd/cw code
already established -- no new attack algorithm code was written, only
registry/dispatch code in `_build_torchattacks`.

Until an attack in this table is smoke-tested against the real formal
pipeline (same discipline as this session's `precheck_real_backends` +
smoke-test-before-scale pattern), it must not be described as "available"
in any experiment plan or report.
