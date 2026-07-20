"""
Shared batch-aggregation core, used by experiments/run_batch.py's existing
(snr, mod, attack, topk) grid and by any other combo-sweep script (e.g. the
sensing validation matrix in experiments/run_sensing_validation_matrix.py)
that needs the same three output CSVs. See docs/parameter_validation.md
section 16 for the full schema and metric-denominator documentation this
implements.

Design:
  - run_dry_run_experiment() (src/utils/pipeline.py) already converts the two
    EXPECTED sensing-failure outcomes (no occupied region survives
    --min-region-len; no full-length segment fits in any surviving region)
    into a normal, structured return value with run_status="sensing_failed"
    -- it does not raise for those. This module does not special-case that
    outcome at all; it is handled uniformly by reading result["run_status"].
  - A GENUINE error (anything else run_dry_run_experiment raises --
    ValueError/TypeError from bad config, RuntimeError from an adapter
    shape-mismatch, dataset load errors, etc.) is caught here, ONCE, at the
    per-combo call site, and turned into a run_status="error" row instead of
    aborting the whole batch -- but the underlying exception is still
    printed to stderr with its combo context, never silently swallowed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.sensing.ground_truth_metrics import compute_multi_burst_sensing_metrics
from src.utils.config import ExperimentConfig
from src.utils.csv_writer import write_summary_csv
from src.utils.pipeline import run_dry_run_experiment

# Fixed key set written into every batch_summary.csv row regardless of
# run_status, so the CSV schema never depends on which combos succeeded.
_SENSING_AGG_FIELDS = [
    "num_truth_bursts",
    "num_detected_regions",
    "detection_probability",
    "false_alarm_region_rate",
    "sample_level_false_positive_rate",
    "sample_level_false_negative_rate",
    "mean_captured_signal_ratio",
    "mean_absolute_start_boundary_error",
    "mean_absolute_end_boundary_error",
]

_RUN_META_FIELDS = [
    "run_status",
    "sensing_success",
    "failure_stage",
    "failure_reason",
    "clean_amc_available",
    "attack_available",
    "defense_available",
    "n_segments",
    "sensing_window_size",
    "segment_length",
    # Segment-alignment fields (docs/parameter_validation.md section 18) --
    # config knobs (uniform per run) plus one aggregate (mean over segments
    # with a resolvable true burst; None on sensing failure or when no
    # segment has one). Per-segment detail lives in each run's summary.csv,
    # not here.
    "alignment_policy",
    "segment_hop",
    "mean_segment_captured_signal_ratio",
    # AWN-input-boundary preprocessing fields (docs/parameter_validation.md
    # section 19) -- awn_preprocess is a config knob (uniform per run); the
    # rest are run-level aggregates over that run's segments (mean for the
    # two power fields and the derived scale factor, global min/max, any()
    # for has_nan/has_inf) -- per-segment detail lives in that run's own
    # summary.csv, not here.
    "awn_preprocess",
    "mean_awn_input_power_before",
    "mean_awn_input_power_after",
    "mean_awn_input_scale_factor",
    "awn_input_min",
    "awn_input_max",
    "awn_input_has_nan",
    "awn_input_has_inf",
]


def _burst_and_region_rows(result: dict, n_samples: int):
    """
    Normalizes single-burst (result["ground_truth"]) and multi-burst
    (result["multi_burst_result"]) truth-vs-detection rows into ONE schema,
    by routing the single-burst case through compute_multi_burst_sensing_metrics
    exactly like derive_batch_aggregate_sensing_fields does (see
    src/sensing/ground_truth_metrics.py) -- not a second, divergent formula.

    Returns ([], []) when neither is available (synthetic source: there is
    no ground truth to report a truth-burst or truth-matched-region row
    for -- batch_bursts_summary.csv/batch_regions_summary.csv simply get
    zero rows from that combo; it still gets its one batch_summary.csv row).
    """
    multi_burst_result = result.get("multi_burst_result")
    ground_truth = result.get("ground_truth")
    regions = result.get("regions") or []

    if multi_burst_result is not None:
        mb = multi_burst_result
    elif ground_truth is not None:
        true_bursts = [{
            "burst_id": 0,
            "true_start": ground_truth["true_start"],
            "true_end": ground_truth["true_end"],
        }]
        mb = compute_multi_burst_sensing_metrics(true_bursts, regions, n_samples)
    else:
        return [], []

    return mb["per_burst"], mb["per_region"]


def run_batch_combos(
    base_dir: Path,
    combos: List[Dict],
    build_cfg: Callable[[Dict, Path], ExperimentConfig],
) -> dict:
    """
    Runs run_dry_run_experiment once per entry in `combos`, writing
    batch_summary.csv (always, one row per combo including error/sensing-
    failed combos), batch_bursts_summary.csv, and batch_regions_summary.csv
    (both omitted entirely if zero combos have ground truth, e.g. an
    all-synthetic batch) under base_dir.

    combos: list of flat scalar-valued dicts, one per combo. Every dict in
    this list MUST have the exact same set of keys (values may differ) --
    those keys become extra columns in every output CSV, letting the caller
    control which swept parameters are recorded. Similarly, every combo
    should use the SAME ground-truth mode (all-synthetic, all-single-burst-
    radioml, or all-multi-burst-radioml) within one call, since multi-burst
    per_burst rows carry extra dataset_mod/dataset_snr/sample_index/... keys
    that a single-burst combo's synthesized per_burst row does not -- mixing
    modes within one call would produce inconsistent CSV columns across
    rows. Run separate batches (separate output subdirectories) for
    different ground-truth modes.
    build_cfg(combo, run_dir) -> ExperimentConfig for that combo.

    Returns a dict with counts (n_ok/n_sensing_failed/n_error) and the
    Path (or None) of each of the three CSVs actually written.
    """
    base_dir = Path(base_dir)
    summary_rows: List[Dict] = []
    burst_rows_all: List[Dict] = []
    region_rows_all: List[Dict] = []
    n_ok = n_sensing_failed = n_error = 0

    for combo_id, combo in enumerate(combos):
        run_dir = base_dir / f"combo{combo_id:04d}"
        cfg = build_cfg(combo, run_dir)

        try:
            result = run_dry_run_experiment(cfg)
        except (ValueError, TypeError, RuntimeError) as exc:
            print(f"[batch][ERROR] combo_id={combo_id} params={combo}: {exc}", file=sys.stderr)
            n_error += 1
            row = {
                "combo_id": combo_id,
                "output_dir": str(run_dir),
                "run_seed": cfg.seed,
                **combo,
                "run_status": "error",
                "sensing_success": None,
                "failure_stage": "exception",
                "failure_reason": str(exc),
                "clean_amc_available": False,
                "attack_available": False,
                "defense_available": False,
                "n_segments": None,
                "sensing_window_size": None,
                "segment_length": None,
                "alignment_policy": cfg.alignment_policy,
                "segment_hop": cfg.segment_hop,
                "mean_segment_captured_signal_ratio": None,
                "awn_preprocess": cfg.awn_preprocess,
                "mean_awn_input_power_before": None,
                "mean_awn_input_power_after": None,
                "mean_awn_input_scale_factor": None,
                "awn_input_min": None,
                "awn_input_max": None,
                "awn_input_has_nan": None,
                "awn_input_has_inf": None,
            }
            for field in _SENSING_AGG_FIELDS:
                row[field] = None
            summary_rows.append(row)
            continue

        if result["run_status"] == "ok":
            n_ok += 1
        else:
            n_sensing_failed += 1

        row = {"combo_id": combo_id, "output_dir": result["output_dir"], "run_seed": result["seed"], **combo}
        for field in _RUN_META_FIELDS:
            row[field] = result.get(field)
        for field in _SENSING_AGG_FIELDS:
            row[field] = result.get(field)
        summary_rows.append(row)

        burst_rows, region_rows = _burst_and_region_rows(result, cfg.n_samples)
        for br in burst_rows:
            burst_rows_all.append({
                "combo_id": combo_id,
                "output_dir": result["output_dir"],
                "run_seed": result["seed"],
                **combo,
                **{k: (str(v) if isinstance(v, list) else v) for k, v in br.items()},
            })
        for rr in region_rows:
            region_rows_all.append({
                "combo_id": combo_id,
                "output_dir": result["output_dir"],
                "run_seed": result["seed"],
                **combo,
                **{k: (str(v) if isinstance(v, list) else v) for k, v in rr.items()},
            })

    batch_summary_csv = base_dir / "batch_summary.csv"
    write_summary_csv(batch_summary_csv, summary_rows)

    batch_bursts_summary_csv: Optional[Path] = None
    if burst_rows_all:
        batch_bursts_summary_csv = base_dir / "batch_bursts_summary.csv"
        write_summary_csv(batch_bursts_summary_csv, burst_rows_all)
    else:
        print("[batch] no combo produced ground-truth burst rows -- batch_bursts_summary.csv not written")

    batch_regions_summary_csv: Optional[Path] = None
    if region_rows_all:
        batch_regions_summary_csv = base_dir / "batch_regions_summary.csv"
        write_summary_csv(batch_regions_summary_csv, region_rows_all)
    else:
        print("[batch] no combo produced ground-truth region rows -- batch_regions_summary.csv not written")

    print(
        f"[batch] {len(combos)} combo(s): ok={n_ok} sensing_failed={n_sensing_failed} error={n_error}"
    )

    return {
        "n_ok": n_ok,
        "n_sensing_failed": n_sensing_failed,
        "n_error": n_error,
        "batch_summary_csv": batch_summary_csv,
        "batch_bursts_summary_csv": batch_bursts_summary_csv,
        "batch_regions_summary_csv": batch_regions_summary_csv,
    }
