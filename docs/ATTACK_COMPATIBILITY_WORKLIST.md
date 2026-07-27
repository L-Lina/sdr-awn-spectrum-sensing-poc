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

**No attack beyond fgsm/pgd/cw has been smoke-tested against the formal
pipeline in this round** (this document does not claim otherwise anywhere
below) -- building this worklist did not delay or block the four-path
Spectrum Sensing Utility Experiment, per instruction.

## Status categories

- **已移植且smoke test通過**: wired into `src/adapters/attack_adapter.py`
  AND exercised end-to-end with the real backend (this session's Phase
  0-4 work, `docs/formal_experiment_plan.md`).
- **已移植但未測**: wired into the adapter but never run with the real
  backend. (Empty today -- everything currently wired has also been
  tested.)
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
| BIM | `torchattacks.BIM` | yes | yes | 尚未移植 |
| DeepFool | `torchattacks.DeepFool` | yes | yes | 尚未移植 |
| APGD | `torchattacks.APGD` | yes | yes | 尚未移植 |
| MIFGSM | `torchattacks.MIFGSM` | yes | yes | 尚未移植 |
| RFGSM | `torchattacks.RFGSM` | yes | yes | 尚未移植 |
| UPGD | `torchattacks.UPGD` | yes | yes | 尚未移植 |
| EOTPGD | `torchattacks.EOTPGD` | yes | yes | 尚未移植 |
| VMIFGSM | `torchattacks.VMIFGSM` | yes | yes | 尚未移植 |
| VNIFGSM | `torchattacks.VNIFGSM` | yes | yes | 尚未移植 |
| Jitter | `torchattacks.Jitter` | yes | yes | 尚未移植 |
| FFGSM | `torchattacks.FFGSM` | yes | yes | 尚未移植 |
| PGDL2 | `torchattacks.PGDL2` | yes | yes | 尚未移植 |
| EADL1 | `torchattacks.EADL1` | yes | yes | 尚未移植 |
| EADEN | `torchattacks.EADEN` | yes | yes | 尚未移植 |
| FAB | `torchattacks.FAB` | yes | yes | 尚未移植 |
| APGDT | `torchattacks.APGDT` | no (`adv_eval.py` only) | yes | 尚未移植 |
| AutoAttack | `torchattacks.AutoAttack` | no (`adv_eval.py` only) | yes | 尚未移植 |
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
| Square | `torchattacks.Square` | no (`adv_eval.py` only) | yes | 尚未移植 |
| TIFGSM | `torchattacks.TIFGSM` | no (`adv_eval.py` only) | yes | 尚未移植 |
| TPGD | `torchattacks.TPGD` | no (`adv_eval.py` only) | yes | 尚未移植 |
| DIFGSM | `torchattacks.DIFGSM` | no (found via `dir(torchattacks)` only, not referenced by name in any `.py` grepped this round) | yes | 尚未移植 (unused by old project too, lowest priority) |

**Verification method for the "exists in installed torchattacks" column**:
`/home/xiaomi/adversarial-rf/.venv/bin/python3 -c "import torchattacks;
print(sorted(n for n in dir(torchattacks) if n[0].isupper()))"` -- 39
classes returned, every name in the table above is among them (plus
`LGV`, `MultiAttack`, `VANILA`, not referenced by any script grepped in
`external/adversarial-rf` this round, so not listed as a worklist item).

## Priority recommendation (not started this round)

If/when attack coverage beyond fgsm/pgd/cw is needed:

1. **BIM, MIFGSM, FFGSM, RFGSM** -- single-family relatives of FGSM/PGD
   already in `attack_adapter.py`; same `eps`/`alpha`/`steps` parameter
   shape, lowest-risk to port (mostly copy the existing `fgsm`/`pgd`
   branch pattern in `_build_torchattacks`).
2. **PGDL2, DeepFool, EADL1/EADEN, FAB** -- different distance metric
   (L2, or none) than the existing Linf-budget attacks; needs a
   parameter-shape decision (this repo's `--attack-eps` is currently
   documented as an Linf budget) before porting, not just a copy-paste.
3. **APGD, VMIFGSM, VNIFGSM, UPGD, EOTPGD, Jitter** -- more
   parameters/restarts, higher runtime cost per instance; port only if a
   specific research question needs them.
4. **Everything `adv_eval.py`-only** (APGDT, AutoAttack, GN, JSMA,
   NIFGSM, OnePixel, PGDRS/PGDRSL2, PIFGSM/PIFGSMPP, Pixle, SINIFGSM,
   SparseFool, SPSA, Square, TIFGSM, TPGD) -- lowest priority; several
   (OnePixel, Pixle, JSMA, SparseFool, Square, SPSA) are
   image-classification-oriented sparse/black-box attacks whose
   applicability to a `[2,128]` IQ tensor (not a 2D image) has not been
   assessed at all, in either project.

**Reuse, do not reimplement**: every attack above already has a working
`torchattacks` constructor call in `external/adversarial-rf/util/
multi_attack_eval.py` or `adv_eval.py`. Porting one into
`attack_adapter.py`'s `_build_torchattacks` means copying that
already-validated constructor call (same pattern as the existing
fgsm/pgd/cw branches), not writing new attack code.

Until an attack in this table is smoke-tested against the real formal
pipeline (same discipline as this session's `precheck_real_backends` +
smoke-test-before-scale pattern), it must not be described as "available"
in any experiment plan or report.
