"""Merges per-attack raw CSVs (written by separate run_low_perturbation_attacks.py
invocations -- pgd+fab from one process, cw/deepfool/ead re-run from a second
process after a bugfix) into the single low_perturbation_raw_results.csv,
and writes the run's manifest.json with full provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)

    attacks = ["pgd", "cw", "deepfool", "ead", "fab"]
    dfs = []
    for a in attacks:
        p = out / f"{a}_raw_results.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))
        else:
            print(f"[finalize] WARNING: missing {p}")
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    combined.to_csv(out / "low_perturbation_raw_results.csv", index=False)
    print(f"[finalize] low_perturbation_raw_results.csv: {len(combined)} rows from {len(dfs)} attacks")

    fgsm = pd.read_csv(out / "fgsm_raw_results.csv")
    print(f"[finalize] fgsm_raw_results.csv: {len(fgsm)} rows")

    import torch
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_status_porcelain": git_status,
        "env": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": "cpu",
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
        },
        "dataset_path": "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl",
        "checkpoint_path": "external/adversarial-rf/2016.10a_AWN.pkl",
        "checkpoint_sha256": sha256_file("external/adversarial-rf/2016.10a_AWN.pkl"),
        "seed": 0,
        "sensing_config": {
            "n_samples": 8192, "embed_snr_margin": 20.0, "threshold_factor": 5.0,
            "sensing_window_size": 128, "min_region_len": 128, "merge_gap": 0,
            "alignment_policy": "max-energy", "awn_preprocess": "radioml-native",
        },
        "fgsm_config": {
            "modulations": "all 11", "snrs": [-10, 0, 18], "samples_per_cell": 10,
            "eps_list": [0.005, 0.01, 0.03, 0.05], "n_base_samples": 330, "n_instances": 1320,
        },
        "low_perturbation_config": {
            "modulations": "all 11", "snrs": [-10, 0, 18], "samples_per_cell": 5, "n_base_samples": 165,
            "pgd_eps_list": [0.005, 0.01, 0.03],
            "cw_params": {"c": 1.0, "kappa": 0, "steps": 20, "lr": 0.01},
            "deepfool_params": "torchattacks library defaults (steps/overshoot not overridden)",
            "ead_params": "torchattacks library defaults (beta/initial_const/max_iterations/lr/kappa not overridden)",
            "fab_eps_list": [0.005, 0.01, 0.03],
        },
        "topk_enabled": False,
        "n_fgsm_rows": len(fgsm),
        "n_low_perturbation_rows": len(combined),
        "fgsm_n_error": int((fgsm["status"] == "error").sum()),
        "fgsm_n_fallback": int(fgsm["fallback_used"].fillna(False).astype(bool).sum()),
        "fgsm_n_nan_inf": int((fgsm["status"] == "nan_inf").sum()),
        "low_perturbation_n_error": int((combined["status"] == "error").sum()) if len(combined) else None,
        "low_perturbation_n_fallback": int(combined["fallback_used"].fillna(False).astype(bool).sum()) if len(combined) else None,
        "low_perturbation_n_nan_inf": int((combined["status"] == "nan_inf").sum()) if len(combined) else None,
        "known_issue_fixed_during_run": (
            "Initial low_perturbation run passed eps=None to AttackAdapter.apply() for "
            "cw/deepfool/ead (non-eps-based attacks), which unconditionally validates eps as "
            "a required non-negative finite float regardless of whether the specific attack "
            "uses it -- all 165 instances of each failed immediately (fast-fail, not a slow "
            "real computation). Fixed in experiments/run_low_perturbation_attacks.py by "
            "passing a structurally-required placeholder eps to apply() while keeping the "
            "CSV's own eps column correctly None for these non-eps-based attacks. cw/deepfool/"
            "ead were re-run after the fix; pgd/fab results from the original run were valid "
            "and unaffected (both pass an explicit eps)."
        ),
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[finalize] manifest.json written")


if __name__ == "__main__":
    main()
