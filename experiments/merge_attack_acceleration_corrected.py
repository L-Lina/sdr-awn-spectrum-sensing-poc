"""
Merges the original 18-condition attack-acceleration benchmark
(results/all_attack_acceleration_20260824T031053Z/) with the corrected-seed
rerun of the 5 attacks confounded by the seed-fairness bug found in the
final completeness audit (difgsm, square, apgd, apgdt, autoattack), into a
new results/all_attack_acceleration_corrected_<timestamp>/ directory. The
12 unaffected conditions (fgsm, bim, pgd_det, pgd_stoch, mifgsm, vmifgsm,
vnifgsm, rfgsm, tpgd, cw, deepfool, fab, ead) are copied as-is from the
original (already independently validated) run -- not rerun. Does not
modify either source directory.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_DIR = Path(sys.argv[sys.argv.index("--original-dir") + 1])
CORRECTED_5_DIR = Path(sys.argv[sys.argv.index("--corrected-5-dir") + 1])
OUT_DIR = Path(sys.argv[sys.argv.index("--output-dir") + 1])

FLAGGED = ["difgsm", "square", "apgd", "apgdt", "autoattack"]
UNAFFECTED = ["fgsm", "bim", "pgd_det", "pgd_stoch", "mifgsm", "vmifgsm", "vnifgsm",
              "rfgsm", "tpgd", "cw", "deepfool", "fab", "ead"]


def merge_csv(filename: str, key_col: str = "attack") -> pd.DataFrame:
    orig = pd.read_csv(ORIGINAL_DIR / filename)
    corr = pd.read_csv(CORRECTED_5_DIR / filename)
    orig_kept = orig[orig[key_col].isin(UNAFFECTED)]
    corr_kept = corr[corr[key_col].isin(FLAGGED)]
    merged = pd.concat([orig_kept, corr_kept], ignore_index=True)
    return merged


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fname in ["attack_acceleration_raw.csv", "attack_correctness_summary.csv",
                  "attack_batching_classification.csv", "attack_thread_tuning.csv",
                  "attack_bottleneck_summary.csv", "attack_acceleration_summary.csv",
                  "attack_e2e_summary.csv"]:
        merged = merge_csv(fname)
        merged.to_csv(OUT_DIR / fname, index=False)
        print(f"[merge] wrote {fname}: {len(merged)} rows "
              f"({(merged['attack'].isin(UNAFFECTED)).sum()} unaffected + "
              f"{(merged['attack'].isin(FLAGGED)).sum()} corrected)")

    # sanity: exactly 18 rows (one per condition) in summary-level tables
    bn = pd.read_csv(OUT_DIR / "attack_bottleneck_summary.csv")
    assert len(bn) == 18, f"expected 18 summary rows, got {len(bn)}"
    assert set(bn["attack"]) == set(UNAFFECTED + FLAGGED), "attack set mismatch after merge"
    print(f"[merge] sanity OK: {len(bn)} attacks in merged summary, set matches expected 18")

    orig_manifest = json.loads((ORIGINAL_DIR / "manifest.json").read_text())
    corr_manifest = json.loads((CORRECTED_5_DIR / "manifest.json").read_text())
    manifest = {
        "round": "all_attack_acceleration_corrected_merge",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_original_result_directory": str(ORIGINAL_DIR),
        "source_original_manifest": orig_manifest,
        "source_corrected_5_result_directory": str(CORRECTED_5_DIR),
        "source_corrected_5_manifest": corr_manifest,
        "seed_fairness_bug_description": (
            "The original benchmark's batching correctness comparison used seed_offset=2000+i "
            "for the batch=1 reference and seed_offset=3000+start for the batch>1 test call -- "
            "two different absolute seed ranges for what must be a paired comparison. This "
            "confounds any attack whose real torchattacks/IQDIFGSM constructor accepts an "
            "explicit `seed` kwarg, since the two sides of the comparison then use different "
            "random search trajectories independent of batching itself."
        ),
        "attacks_confirmed_affected_and_rerun": FLAGGED,
        "attacks_confirmed_unaffected_not_rerun": UNAFFECTED,
        "fab_note": (
            "FAB also accepts an explicit seed but was independently verified immune "
            "(100% match across 10 same-sample/different-seed pairs at batch=1) -- not rerun, "
            "its original A_implementation_optimization classification stands unchanged."
        ),
        "corrected_seed_policy": (
            "stable_seed(mod,snr,idx,attack_name): deterministic, identity-derived, independent "
            "of loop position/phase/batch layout, used for every single-sample call; for the "
            "batching correctness comparison specifically, the batch=1 reference is recomputed "
            "per chunk using the same chunk-anchored seed the batch>1 call for that chunk "
            "receives, isolating batching as the only variable."
        ),
        "rerun_scope": "5 of 18 conditions rerun (difgsm, square, apgd, apgdt, autoattack); "
                        "the other 13 conditions (12 unaffected + FAB) copied unchanged from the "
                        "original, independently-validated run.",
        "headline_result": (
            "DIFGSM's classification flips from D_batching_unsafe (original, seed-confounded) to "
            "A_implementation_optimization (corrected, bit-identical at every batch size) -- the "
            "original 'batch-shared diversity randomness' narrative is retracted as an artifact "
            "of the seed bug, not a genuine property of IQDIFGSM. Square/APGD/APGDT/AutoAttack "
            "remain D_batching_unsafe under the corrected, fair comparison, with revised match "
            "rates."
        ),
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[merge] wrote manifest.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
