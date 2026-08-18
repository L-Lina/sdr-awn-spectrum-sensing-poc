"""
Submission-ready provenance manifest for THIS round's performance
correctness / stability validation artifacts (batch_equivalence_audit.py,
analyze_awn_latency_outliers.py, awn_thread_microbenchmark.py,
diagnose_streaming_failure.py, and the bottleneck_by_percentile.csv
recomputation), which ran after Phase A-L's own manifest.json was already
written and frozen.

Writes a NEW file, validation_round_manifest.json, into the SAME existing
results directory -- does NOT edit manifest.json or any other Phase A-L
file in place (results/ directories are gitignored and existing files in
them must never be modified; adding a new file alongside them is not a
modification of those files).

Run directly:
    python experiments/write_validation_round_manifest.py --output-dir results/<existing_dir>
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CHECKPOINT_PATH = REPO_ROOT / "external/adversarial-rf/2016.10a_AWN.pkl"
DATASET_PATH = "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl"
MODULATIONS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]


def git(cmd):
    return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True).strip()


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        return None
    return None


def mem_total():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return line.strip()
    except Exception:  # noqa: BLE001
        return None
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    if not out_dir.exists():
        raise FileNotFoundError(f"{out_dir} does not exist -- this manifest documents an EXISTING results "
                                 "directory's validation-round artifacts, it does not create a new run.")

    import platform
    import torch
    import torchattacks

    manifest = {
        "round": "performance_correctness_stability_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_directory": str(out_dir),
        "note": "This file documents provenance for the validation-round artifacts only "
                "(batch_equivalence_audit.py, analyze_awn_latency_outliers.py, "
                "awn_thread_microbenchmark.py, diagnose_streaming_failure.py, "
                "bottleneck_by_percentile.csv). Phase A-L's own artifacts are documented in "
                "manifest.json in this same directory, written earlier and NOT modified by this file.",

        # --- git provenance ---
        "git_base_commit": git(["git", "rev-parse", "HEAD"]),
        "git_working_tree_dirty_status": git(["git", "status", "--porcelain"]) or "(clean)",

        # --- hardware provenance ---
        "cpu_model": cpu_model(),
        "cpu_logical_core_count": int(subprocess.check_output(["nproc"], text=True).strip()),
        "cpu_physical_core_count": (lambda: (
            subprocess.check_output(
                "lscpu | awk -F: '/^Core\\(s\\) per socket/{c=$2} /^Socket\\(s\\)/{s=$2} "
                "END{print c*s}'", shell=True, text=True,
            ).strip() or None
        ))(),
        "ram": mem_total(),
        "os": platform.platform(),

        # --- software provenance ---
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchattacks_version": torchattacks.__version__,
        "torch_default_num_threads_at_manifest_time": torch.get_num_threads(),

        # --- model/dataset provenance ---
        "awn_checkpoint_path": str(CHECKPOINT_PATH),
        "awn_checkpoint_sha256": sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.exists() else None,
        "dataset_path": DATASET_PATH,
        "dataset_name": "RML2016.10a",
        "modulations": MODULATIONS,

        # --- per-artifact reproduction parameters ---
        "artifacts": {
            "pgd_batch_equivalence_deterministic.csv": {
                "script": "experiments/batch_equivalence_audit.py",
                "sample_generation": "experiments/acceleration_pilot.py:build_pilot_inputs(60) -- "
                                      "60 samples cycling through all 11 modulations at SNR=0",
                "sample_count": 60, "seed": 0,
                "batch_sizes_compared": [1, 16],
                "attack_params": {"eps": 0.05, "random_start": False},
                "torch_num_threads": "left at process default (not explicitly set by this script)",
                "warmup_count": 0,
                "timing_clock": "not a timing artifact -- correctness/equivalence comparison only",
                "profiler_method": "none (direct AttackAdapter.apply() calls, no profiler attached)",
            },
            "pgd_batch_stochastic_comparison.csv": {
                "script": "experiments/batch_equivalence_audit.py",
                "sample_generation": "experiments/acceleration_pilot.py:build_pilot_inputs(60)",
                "sample_count": 60, "seed": 0,
                "batch_sizes_compared": [1, 16],
                "attack_params": {"eps": 0.05, "random_start": True},
                "torch_num_threads": "left at process default",
                "warmup_count": 0,
            },
            "cw_batch_equivalence.csv": {
                "script": "experiments/batch_equivalence_audit.py",
                "sample_generation": "experiments/acceleration_pilot.py:build_pilot_inputs(60)",
                "sample_count": 60, "seed": 0,
                "batch_sizes_compared": [1, 16],
                "attack_params": {"c": 1.0, "kappa": 0, "steps": 20, "lr": 0.01, "eps": 0.05},
                "torch_num_threads": "left at process default",
                "warmup_count": 0,
            },
            "awn_latency_outliers.csv": {
                "script": "experiments/analyze_awn_latency_outliers.py",
                "source_data": "pipeline_latency_raw.csv (Phase A, n=2200, NOT rerun this round)",
                "extraction": "top-50 rows by awn_clean_inference_ms",
            },
            "awn_thread_microbenchmark_summary.csv / _by_group.csv / _raw.csv": {
                "script": "experiments/awn_thread_microbenchmark.py",
                "sample_count": 100,
                "sample_composition": "25 AM-SSB @ SNR in {10,12,14,16,18} (known-slow group) + "
                                       "50 other modulations @ SNR=0 (5 samples each) + "
                                       "25 AM-SSB @ SNR in {-20,-16,-12,-8,-4} (known-fast group)",
                "seed_per_sample": "sample_index (0-4), matching embed_sample_in_noise's own seed param",
                "torch_num_threads_conditions": ["current(default)", 1, 2, 4, 8, 16],
                "warmup_count_per_condition": 60,
                "timed_call_count_per_condition": 500,
                "timing_clock": "time.perf_counter() (monotonic)",
                "profiler_method": "none (direct AWNModelAdapter.infer() wall-clock timing)",
            },
            "bottleneck_by_percentile.csv": {
                "script": "inline one-off computation (see docs/research/PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md "
                          "section 15.5 for the exact aggregation logic)",
                "source_data": "pipeline_latency_raw.csv (Phase A, n=2200, NOT rerun this round)",
            },
            "streaming_failure_diagnosis.csv": {
                "script": "experiments/diagnose_streaming_failure.py",
                "stream_construction": "3 RadioML bursts (QPSK, BPSK, QAM16) @ SNR=0, seed=7, "
                                        "same as experiments/test_streaming_sensing.py:build_test_stream()",
                "chunk_sizes": [256, 512, 1024, 2048],
                "carry_samples": 128,
                "window": 128, "threshold_factor": 5.0,
            },
        },
    }

    out_path = out_dir / "validation_round_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[manifest] wrote {out_path}")


if __name__ == "__main__":
    main()
