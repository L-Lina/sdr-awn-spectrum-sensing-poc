"""
Post-alignment-fix spectrum-sensing parameter revalidation (round 13).
Purpose: re-check threshold-factor / sensing-window-size / min-region-len /
merge-gap under the CURRENT defaults (alignment-policy=max-energy,
awn-preprocess=radioml-native, docs/parameter_validation.md section 20) --
every prior sweep of these parameters (docs section 16.4, "round 7") ran
under the OLD naive/legacy-unit-power defaults, before the alignment
(section 18/round 9) and preprocessing (section 19/round 10) fixes existed.
NOT the formal full-parameter batch (explicitly out of scope). attack=none,
real Top-K not exercised (avoids conflating defense behavior with sensing
behavior).

Fixed baseline throughout: iq_source=radioml, real AWN, attack=none, CPU,
seed=42, window_size=128 (AWN input length), modulations {QPSK,BPSK,QAM16},
dataset_snr {0,18}, sample_index {0,1,2,3,4} -- 30 unique samples --
alignment_policy=max-energy, awn_preprocess=radioml-native (explicit, not
relying on the source-aware default, so this script's intent is
unambiguous even if the default ever changes).

Design: separate OFAT stages (A: threshold-factor, B: sensing-window-size,
C: min-region-len), run against the same 30-sample base grid, plus a
separately-calibrated multi-burst stage (D: merge-gap, which needs REAL
multiple detected regions -- a single-burst grid can never exercise
merging) and a small set of targeted burst/stream checks (E). Each stage
prints its own combo count and time estimate before running, per this
round's explicit "not one giant combination" instruction.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.batch_aggregation import run_batch_combos  # noqa: E402
from src.utils.config import ExperimentConfig  # noqa: E402
from src.utils.csv_writer import write_summary_csv  # noqa: E402

SEED = 42
DATASET_PATH = "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl"
CHECKPOINT = "external/adversarial-rf/2016.10a_AWN.pkl"
BASE_OUTPUT = Path("results/sensing_revalidation_after_alignment")

MODS = ["QPSK", "BPSK", "QAM16"]
SNRS = [0, 18]
SAMPLE_INDICES = [0, 1, 2, 3, 4]

BASELINE = dict(threshold_factor=1.5, sensing_window_size=128, min_region_len=0, merge_gap=0)


def _enrich_row(row: dict) -> dict:
    """Reads this combo's own summary.csv (if it exists) for the per-segment
    fields batch_summary.csv doesn't carry (it's a per-combo aggregate, not
    per-segment): selected_segment_start/end, segment-level (NOT region-level)
    captured ratio, pred_clean, awn_backend, x_clean shape. Single-burst mode
    always yields exactly one segment when successful, so row 0 is used --
    reads the SAME column select_aligned_segments actually wrote (Part 5.6:
    this is the alignment-policy-selected segment, never a fixed
    detected-region-start slice)."""
    enriched = {
        "true_burst_start": None, "true_burst_end": None,
        "detected_region_start": None, "detected_region_end": None,
        "start_boundary_error": None, "end_boundary_error": None,
        "detection_success": None,
        "selected_segment_start": None, "selected_segment_end": None,
        "segment_level_captured_ratio": None,
        "x_clean_shape": None, "pred_clean": None, "awn_backend": None,
    }
    summary_path = Path(row["output_dir"]) / "summary.csv"
    if not summary_path.exists():
        return enriched
    with open(summary_path) as f:
        srows = list(csv.DictReader(f))
    if not srows:
        return enriched
    r = srows[0]
    enriched.update({
        "true_burst_start": r.get("true_burst_start"), "true_burst_end": r.get("true_burst_end"),
        "detected_region_start": r.get("best_detected_start"), "detected_region_end": r.get("best_detected_end"),
        "start_boundary_error": r.get("start_boundary_error"), "end_boundary_error": r.get("end_boundary_error"),
        "detection_success": r.get("detection_success"),
        "selected_segment_start": r.get("selected_segment_start"), "selected_segment_end": r.get("selected_segment_end"),
        "segment_level_captured_ratio": r.get("segment_captured_signal_ratio"),
        "x_clean_shape": f"(1, 2, {r.get('segment_length')})",
        "pred_clean": r.get("pred_clean"), "awn_backend": r.get("awn_backend"),
    })
    return enriched


def run_stage(stage_name: str, combos: list, build_cfg, estimate_note: str) -> dict:
    stage_dir = BASE_OUTPUT / stage_name
    print(f"\n{'='*70}\n[stage {stage_name}] {len(combos)} combos. {estimate_note}\n{'='*70}")
    t0 = time.time()
    result = run_batch_combos(stage_dir, combos, build_cfg)
    elapsed = time.time() - t0
    print(f"[stage {stage_name}] done in {elapsed:.1f}s -- ok={result['n_ok']} sensing_failed={result['n_sensing_failed']} error={result['n_error']}")

    with open(stage_dir / "batch_summary.csv") as f:
        batch_rows = list(csv.DictReader(f))
    enriched_rows = [{**r, **_enrich_row(r)} for r in batch_rows]
    write_summary_csv(stage_dir / "sensing_parameter_summary.csv", enriched_rows)

    failures = [r for r in enriched_rows if r["run_status"] != "ok"]
    if failures:
        write_summary_csv(stage_dir / "failures.csv", failures)
    else:
        print(f"[stage {stage_name}] no failures -- failures.csv not written")

    return {"result": result, "enriched_rows": enriched_rows, "elapsed": elapsed}


def build_cfg_common(mod, snr, idx, threshold_factor, sensing_window_size, min_region_len, merge_gap, run_dir):
    return ExperimentConfig(
        snr=10.0, mod=mod, attack="none", topk=50,
        threshold_factor=threshold_factor,
        window_size=128,
        sensing_window_size=sensing_window_size,
        min_region_len=min_region_len,
        merge_gap=merge_gap,
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
        dataset_mod=mod,
        dataset_snr=snr,
        sample_index=idx,
        embed_snr_margin=20.0,
        num_bursts=1,
        dataset_mod_list=None, dataset_snr_list=None, sample_index_list=None,
        min_burst_gap=50, max_burst_gap=50, burst_gap_list=None, burst_power_scale_list=None,
        alignment_policy="max-energy",
        segment_hop=1,
        awn_preprocess="radioml-native",
    )


def stage_A_threshold_factor():
    values = [0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
    combos = [
        {"dataset_mod": mod, "dataset_snr": snr, "sample_index": idx,
         "threshold_factor": tf, "min_region_len": BASELINE["min_region_len"], "merge_gap": BASELINE["merge_gap"]}
        for mod in MODS for snr in SNRS for idx in SAMPLE_INDICES for tf in values
    ]

    def build_cfg(combo, run_dir):
        return build_cfg_common(combo["dataset_mod"], combo["dataset_snr"], combo["sample_index"],
                                 combo["threshold_factor"], BASELINE["sensing_window_size"],
                                 combo["min_region_len"], combo["merge_gap"], run_dir)

    return run_stage("A_threshold_factor", combos, build_cfg,
                      f"threshold_factor in {values}, 30 samples x {len(values)} values, ~1.5-2s/combo, est ~{len(combos)*1.7:.0f}s")


def stage_B_sensing_window_size():
    values = [16, 32, 64, 128, 256]
    combos = [
        {"dataset_mod": mod, "dataset_snr": snr, "sample_index": idx,
         "threshold_factor": BASELINE["threshold_factor"], "min_region_len": BASELINE["min_region_len"],
         "merge_gap": BASELINE["merge_gap"], "sensing_window_size_requested": sws}
        for mod in MODS for snr in SNRS for idx in SAMPLE_INDICES for sws in values
    ]

    def build_cfg(combo, run_dir):
        return build_cfg_common(combo["dataset_mod"], combo["dataset_snr"], combo["sample_index"],
                                 combo["threshold_factor"], combo["sensing_window_size_requested"],
                                 combo["min_region_len"], combo["merge_gap"], run_dir)

    return run_stage("B_sensing_window_size", combos, build_cfg,
                      f"sensing_window_size in {values}, 30 samples x {len(values)} values, est ~{len(combos)*1.7:.0f}s")


def stage_C_min_region_len():
    values = [0, 32, 64, 128, 256]
    combos = [
        {"dataset_mod": mod, "dataset_snr": snr, "sample_index": idx,
         "threshold_factor": BASELINE["threshold_factor"], "merge_gap": BASELINE["merge_gap"],
         "min_region_len": mrl}
        for mod in MODS for snr in SNRS for idx in SAMPLE_INDICES for mrl in values
    ]

    def build_cfg(combo, run_dir):
        return build_cfg_common(combo["dataset_mod"], combo["dataset_snr"], combo["sample_index"],
                                 combo["threshold_factor"], BASELINE["sensing_window_size"],
                                 combo["min_region_len"], combo["merge_gap"], run_dir)

    return run_stage("C_min_region_len", combos, build_cfg,
                      f"min_region_len in {values}, 30 samples x {len(values)} values, est ~{len(combos)*1.7:.0f}s")


if __name__ == "__main__":
    stage_A_threshold_factor()
    stage_B_sensing_window_size()
    stage_C_min_region_len()
