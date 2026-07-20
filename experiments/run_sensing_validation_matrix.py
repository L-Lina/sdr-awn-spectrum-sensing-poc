"""
Small, hand-designed sensing validation matrix (round 7 / Part E). Purpose:
exercise the new sensing-failure handling and batch-aggregation CSVs
(src/utils/pipeline.py, src/utils/batch_aggregation.py) across a real,
varied set of sensing parameters -- NOT a full parameter sweep, NOT an AMC
accuracy evaluation. attack=none throughout; real AWN inference
(--use-real-awn) throughout; a single fixed seed throughout.

Design: OFAT (one-factor-at-a-time) anchored at one baseline point, plus one
explicit small factorial for the one interaction Part E asked to see
(min-region-len x sensing-window-size). Baseline chosen as
mod=BPSK, snr=18, sample_index=0, threshold_factor=1.5,
sensing_window_size=128, min_region_len=0, merge_gap=0 -- this is the exact
combination previously found (docs/parameter_validation.md section 14/16)
to reproduce captured_signal_ratio=0.625 for BPSK/snr18/sample0, so using it
as the OFAT anchor also lets every group that revisits this point serve as
a free same-seed reproducibility cross-check (the same params appear 4x
across groups 1-4 below, always through independently re-run combos).

Groups (all single-burst radioml, one run_batch_combos call --
results/sensing_validation_matrix/single_burst/):
  1. (dataset_mod x dataset_snr x sample_index) OFAT: 2x2x5 = 20 combos
     (baseline threshold_factor/sensing_window_size/min_region_len/merge_gap)
  2. threshold_factor OFAT at the baseline (mod,snr,idx): 5 combos
  3. sensing_window_size OFAT at the baseline (mod,snr,idx): 5 combos
  4. min_region_len x sensing_window_size small factorial at the baseline
     (mod,snr,idx,threshold_factor): 3x5 = 15 combos
  Subtotal: 45 combos

Group 5 (multi-burst radioml, SEPARATE run_batch_combos call --
results/sensing_validation_matrix/multi_burst_merge_gap/, since mixing
ground-truth modes in one CSV set is unsupported -- see
batch_aggregation.py's docstring):
  5. merge_gap OFAT with 2 back-to-back bursts (BPSK/snr18/idx0,
     BPSK/snr18/idx1), baseline threshold_factor/sensing_window_size/
     min_region_len: 3 combos

TOTAL: 48 combos. Measured steady-state cost ~1.4s/combo in-process with
real AWN (checkpoint loaded fresh per combo) -- see terminal output of this
script for actual timing. Estimated total runtime: ~1-2 minutes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.batch_aggregation import run_batch_combos  # noqa: E402
from src.utils.config import ExperimentConfig  # noqa: E402

SEED = 42
DATASET_PATH = "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl"
CHECKPOINT = "external/adversarial-rf/2016.10a_AWN.pkl"
BASE_OUTPUT = Path("results/sensing_validation_matrix")

BASELINE = dict(
    dataset_mod="BPSK", dataset_snr=18, sample_index=0,
    threshold_factor=1.5, sensing_window_size=128, min_region_len=0, merge_gap=0,
)

MODS = ["QPSK", "BPSK"]
SNRS = [0, 18]
SAMPLE_INDICES = [0, 1, 2, 3, 4]
THRESHOLD_FACTORS = [0.8, 1.0, 1.2, 1.5, 2.0]
SENSING_WINDOW_SIZES = [16, 32, 64, 128, 256]
MIN_REGION_LENS = [0, 64, 128]
MERGE_GAPS = [0, 16, 64]


def _single_burst_combo(mod, snr, sample_index, threshold_factor, sensing_window_size, min_region_len, merge_gap):
    return {
        "dataset_mod": mod, "dataset_snr": snr, "sample_index": sample_index,
        "threshold_factor": threshold_factor, "sensing_window_size": sensing_window_size,
        "min_region_len": min_region_len, "merge_gap": merge_gap,
    }


def build_single_burst_combos():
    combos = []

    # Group 1: (mod x snr x sample_index) OFAT at baseline sensing params.
    for mod in MODS:
        for snr in SNRS:
            for idx in SAMPLE_INDICES:
                combos.append(_single_burst_combo(
                    mod, snr, idx,
                    BASELINE["threshold_factor"], BASELINE["sensing_window_size"],
                    BASELINE["min_region_len"], BASELINE["merge_gap"],
                ))

    # Group 2: threshold_factor OFAT at the baseline (mod, snr, idx).
    for tf in THRESHOLD_FACTORS:
        combos.append(_single_burst_combo(
            BASELINE["dataset_mod"], BASELINE["dataset_snr"], BASELINE["sample_index"],
            tf, BASELINE["sensing_window_size"], BASELINE["min_region_len"], BASELINE["merge_gap"],
        ))

    # Group 3: sensing_window_size OFAT at the baseline (mod, snr, idx).
    for sws in SENSING_WINDOW_SIZES:
        combos.append(_single_burst_combo(
            BASELINE["dataset_mod"], BASELINE["dataset_snr"], BASELINE["sample_index"],
            BASELINE["threshold_factor"], sws, BASELINE["min_region_len"], BASELINE["merge_gap"],
        ))

    # Group 4: min_region_len x sensing_window_size small factorial.
    for mrl in MIN_REGION_LENS:
        for sws in SENSING_WINDOW_SIZES:
            combos.append(_single_burst_combo(
                BASELINE["dataset_mod"], BASELINE["dataset_snr"], BASELINE["sample_index"],
                BASELINE["threshold_factor"], sws, mrl, BASELINE["merge_gap"],
            ))

    return combos


def build_multi_burst_combos():
    combos = []
    for mg in MERGE_GAPS:
        combos.append({
            "merge_gap": mg,
            "threshold_factor": BASELINE["threshold_factor"],
            "sensing_window_size": BASELINE["sensing_window_size"],
            "min_region_len": BASELINE["min_region_len"],
        })
    return combos


def build_single_burst_cfg(combo: dict, run_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        snr=10.0, mod=combo["dataset_mod"], attack="none", topk=50,
        threshold_factor=combo["threshold_factor"],
        window_size=128,
        sensing_window_size=combo["sensing_window_size"],
        min_region_len=combo["min_region_len"],
        merge_gap=combo["merge_gap"],
        burst_len=600,
        output_dir=str(run_dir),
        dry_run=True,
        use_real_topk=False,
        use_real_awn=True,
        checkpoint=CHECKPOINT,
        device="cpu",
        attack_eps=0.03,
        use_real_attack=False,
        attack_temperature=1.0,
        attack_diagnostics=False,
        seed=SEED,
        cw_c=1.0, cw_steps=20, cw_lr=0.01,
        iq_source="radioml",
        dataset_path=DATASET_PATH,
        dataset_mod=combo["dataset_mod"],
        dataset_snr=combo["dataset_snr"],
        sample_index=combo["sample_index"],
        embed_snr_margin=20.0,
        num_bursts=1,
        dataset_mod_list=None, dataset_snr_list=None, sample_index_list=None,
        min_burst_gap=50, max_burst_gap=50, burst_gap_list=None, burst_power_scale_list=None,
    )


def build_multi_burst_cfg(combo: dict, run_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        snr=10.0, mod="BPSK", attack="none", topk=50,
        threshold_factor=combo["threshold_factor"],
        window_size=128,
        sensing_window_size=combo["sensing_window_size"],
        min_region_len=combo["min_region_len"],
        merge_gap=combo["merge_gap"],
        burst_len=600,
        output_dir=str(run_dir),
        dry_run=True,
        use_real_topk=False,
        use_real_awn=True,
        checkpoint=CHECKPOINT,
        device="cpu",
        attack_eps=0.03,
        use_real_attack=False,
        attack_temperature=1.0,
        attack_diagnostics=False,
        seed=SEED,
        cw_c=1.0, cw_steps=20, cw_lr=0.01,
        iq_source="radioml",
        dataset_path=DATASET_PATH,
        dataset_mod=None, dataset_snr=None, sample_index=None,
        embed_snr_margin=20.0,
        num_bursts=2,
        dataset_mod_list=["BPSK", "BPSK"], dataset_snr_list=[18, 18], sample_index_list=[0, 1],
        min_burst_gap=50, max_burst_gap=50, burst_gap_list=None, burst_power_scale_list=None,
    )


def main() -> None:
    single_burst_combos = build_single_burst_combos()
    multi_burst_combos = build_multi_burst_combos()
    total = len(single_burst_combos) + len(multi_burst_combos)
    print(f"[matrix] single-burst combos: {len(single_burst_combos)}")
    print(f"[matrix] multi-burst combos:  {len(multi_burst_combos)}")
    print(f"[matrix] TOTAL combos: {total} (estimated ~1-2 minutes at ~1.4s/combo)")

    t0 = time.time()
    single_result = run_batch_combos(
        BASE_OUTPUT / "single_burst", single_burst_combos, build_single_burst_cfg,
    )
    multi_result = run_batch_combos(
        BASE_OUTPUT / "multi_burst_merge_gap", multi_burst_combos, build_multi_burst_cfg,
    )
    elapsed = time.time() - t0

    print(f"\n[matrix] done in {elapsed:.1f}s")
    print(f"[matrix] single-burst: ok={single_result['n_ok']} sensing_failed={single_result['n_sensing_failed']} error={single_result['n_error']}")
    print(f"[matrix] multi-burst:  ok={multi_result['n_ok']} sensing_failed={multi_result['n_sensing_failed']} error={multi_result['n_error']}")


if __name__ == "__main__":
    main()
