"""
Ground-truth spectrum-sensing metrics: compares the sensing stage's detected
occupied region(s) against a known true burst interval (only available when
the burst position is actually known ahead of time, e.g. RadioML-embedding
mode -- src/sensing/radioml_source.py). Never used by the synthetic-IQ path
today (that source's own true position is not currently tracked end-to-end
by the pipeline either), but written source-agnostic in case that changes.

All formulas operate on half-open sample intervals [start, end).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

__all__ = [
    "compute_sensing_ground_truth_metrics",
    "compute_multi_burst_sensing_metrics",
]


def compute_sensing_ground_truth_metrics(
    true_start: int,
    true_end: int,
    detected_regions: List[Tuple[int, int]],
) -> dict:
    """
    Definitions (T = [true_start, true_end), D = best-matching detected
    region -- the one in detected_regions with the largest overlap with T,
    or None if detected_regions is empty or none overlap):

      intersection_length   = max(0, min(true_end, D_end) - max(true_start, D_start))
      true_burst_length     = true_end - true_start
      detected_region_length = D_end - D_start
      detection_success      = intersection_length > 0 for the best match
                                (at least one detected region overlaps T at all)
      captured_signal_ratio  = intersection_length / true_burst_length
                                ("recall" -- how much of the real burst did
                                sensing actually capture)
      extra_captured_noise_ratio = (detected_region_length - intersection_length)
                                    / detected_region_length
                                    ("1 - precision" -- what fraction of what
                                    was captured is actually just noise
                                    padding, not the real burst)
      missed_sample_count       = true_burst_length - intersection_length
                                   (samples of the real burst NOT captured)
      false_occupied_sample_count = detected_region_length - intersection_length
                                     (samples in the detected region that are
                                     NOT part of the real burst)
      start_boundary_error = D_start - true_start (signed: positive means the
                              detected region starts AFTER the true burst
                              started, i.e. missed the leading edge; negative
                              means it starts BEFORE, i.e. captured extra
                              noise ahead of the burst)
      end_boundary_error   = D_end - true_end (signed, same convention for
                              the trailing edge)

    If no detected region overlaps T at all (detection_success=False), the
    "best match" falls back to the region with the SMALLEST gap to T (so
    boundary errors are still meaningful/interpretable as "how far off"),
    or None if detected_regions is empty entirely -- in that case every
    metric that requires a detected region is None.
    """
    true_burst_length = true_end - true_start

    best_region: Optional[Tuple[int, int]] = None
    best_intersection = -1
    best_gap = None
    for (d_start, d_end) in detected_regions:
        intersection = max(0, min(true_end, d_end) - max(true_start, d_start))
        gap = max(0, max(true_start, d_start) - min(true_end, d_end))
        if intersection > best_intersection or (intersection == best_intersection and
                                                  (best_gap is None or gap < best_gap)):
            best_intersection = intersection
            best_gap = gap
            best_region = (d_start, d_end)

    if best_region is None:
        return {
            "detection_success": False,
            "true_start": true_start,
            "true_end": true_end,
            "true_burst_length": true_burst_length,
            "detected_region_count": 0,
            "best_detected_start": None,
            "best_detected_end": None,
            "start_boundary_error": None,
            "end_boundary_error": None,
            "intersection_length": 0,
            "captured_signal_ratio": 0.0,
            "extra_captured_noise_ratio": None,
            "missed_sample_count": true_burst_length,
            "false_occupied_sample_count": None,
        }

    d_start, d_end = best_region
    detected_region_length = d_end - d_start
    intersection_length = best_intersection
    captured_signal_ratio = intersection_length / true_burst_length if true_burst_length > 0 else 0.0
    extra_captured_noise_ratio = (
        (detected_region_length - intersection_length) / detected_region_length
        if detected_region_length > 0 else None
    )
    missed_sample_count = true_burst_length - intersection_length
    false_occupied_sample_count = detected_region_length - intersection_length

    return {
        "detection_success": intersection_length > 0,
        "true_start": true_start,
        "true_end": true_end,
        "true_burst_length": true_burst_length,
        "detected_region_count": len(detected_regions),
        "best_detected_start": d_start,
        "best_detected_end": d_end,
        "start_boundary_error": d_start - true_start,
        "end_boundary_error": d_end - true_end,
        "intersection_length": intersection_length,
        "captured_signal_ratio": captured_signal_ratio,
        "extra_captured_noise_ratio": extra_captured_noise_ratio,
        "missed_sample_count": missed_sample_count,
        "false_occupied_sample_count": false_occupied_sample_count,
    }


def compute_multi_burst_sensing_metrics(
    true_bursts: List[dict],
    detected_regions: List[Tuple[int, int]],
    n_samples: int,
) -> dict:
    """
    Truth-to-detection matching for MULTIPLE true bursts against MULTIPLE
    detected regions.

    Matching rule: full bipartite overlap enumeration, NOT a strict 1:1
    match (Hungarian algorithm or IoU-max 1:1 assignment would force every
    burst/region into at most one partner, which cannot correctly represent
    a region that genuinely overlaps two neighboring bursts merged by
    --merge-gap, or a burst genuinely split across two detected regions --
    both of which docs/parameter_validation.md section 15's merge-gap test
    cases are specifically designed to produce). Every (burst, region) pair
    with intersection_length > 0 is a real edge in the match; a burst or
    region can have zero, one, or multiple edges.

    true_bursts: list of dicts each containing at least "burst_id",
    "true_start", "true_end" (additional keys, e.g. dataset_mod, are passed
    through unchanged into the per-burst output).

    Definitions (B_i = [true_bursts[i].true_start, true_bursts[i].true_end),
    R_j = detected_regions[j], both half-open intervals):

      intersection_length(i,j) = max(0, min(B_i.end, R_j.end) - max(B_i.start, R_j.start))

    Per-burst (one dict per input true burst):
      total_intersection_length = sum_j intersection_length(i,j) over all j
                                   (safe to sum, not union, because detected
                                   regions are pairwise non-overlapping by
                                   construction -- they come from
                                   merge_close_regions/filter_by_min_length,
                                   which only ever produce disjoint intervals)
      detection_success   = total_intersection_length > 0
      matched_region_ids  = [j for j with intersection_length(i,j) > 0], sorted
      matched_region_id   = the SINGLE region in matched_region_ids with the
                             largest intersection_length(i,j) ("primary"
                             match; None if matched_region_ids is empty) --
                             boundary errors below are computed against this
                             one region specifically
      captured_signal_ratio = total_intersection_length / true_burst_length
      missed_sample_count    = true_burst_length - total_intersection_length
      start_boundary_error / end_boundary_error: same signed convention as
                             compute_sensing_ground_truth_metrics, computed
                             against matched_region_id; None if unmatched

    Per-region (one dict per detected region):
      total_intersection_length = sum_i intersection_length(i,j) over all i
      matched_burst_ids   = [i for i with intersection_length(i,j) > 0], sorted
                             (0 entries = false alarm; 1 = clean match;
                             2+ = this region MERGED multiple true bursts)
      false_occupied_sample_count = detected_region_length - total_intersection_length
      extra_captured_noise_ratio  = false_occupied_sample_count / detected_region_length

    Aggregate (denominators spelled out explicitly -- see
    docs/parameter_validation.md section 15.2 for the same table):
      num_truth_bursts          = len(true_bursts)
      num_detected_regions      = len(detected_regions)
      num_matched_bursts        = count of bursts with detection_success=True
      num_missed_bursts         = num_truth_bursts - num_matched_bursts
      num_false_alarm_regions   = count of regions with 0 matched_burst_ids
      detection_probability     = num_matched_bursts / num_truth_bursts
                                   (Pd; undefined/None if num_truth_bursts==0)
      false_alarm_region_rate   = num_false_alarm_regions / num_detected_regions
                                   (region-level Pfa; None if num_detected_regions==0)
      sample_level_false_positive_rate =
          (sum of every region's false_occupied_sample_count)
          / (n_samples - sum of every true burst's true_burst_length)
          -- classic sample-level Pfa: fraction of TRUE BACKGROUND samples
          incorrectly marked occupied. None if the denominator is 0 (bursts
          fill the entire stream).
      sample_level_false_negative_rate =
          (sum of every burst's missed_sample_count)
          / (sum of every true burst's true_burst_length)
          -- fraction of TRUE SIGNAL samples NOT captured (= 1 - recall,
          sample-weighted across all bursts, not averaged per-burst).
          None if there are zero truth bursts or all have zero length.
      mean_captured_signal_ratio = mean of per-burst captured_signal_ratio
                                   (simple mean over bursts, NOT
                                   sample-weighted -- distinct from
                                   sample_level_false_negative_rate above)
      mean_abs_start_boundary_error / mean_abs_end_boundary_error =
          mean of abs(start_boundary_error) / abs(end_boundary_error) over
          MATCHED bursts only (unmatched bursts have no boundary to
          measure); None if zero bursts are matched
      mean_abs_boundary_error = mean of the two above combined (all start
          and end errors from matched bursts pooled together)
    """
    true_burst_lengths = {b["burst_id"]: b["true_end"] - b["true_start"] for b in true_bursts}

    # Build the full intersection matrix once.
    intersections = {}  # (burst_id, region_idx) -> intersection_length
    for b in true_bursts:
        for j, (r_start, r_end) in enumerate(detected_regions):
            inter = max(0, min(b["true_end"], r_end) - max(b["true_start"], r_start))
            if inter > 0:
                intersections[(b["burst_id"], j)] = inter

    per_burst = []
    for b in true_bursts:
        burst_id = b["burst_id"]
        true_burst_length = true_burst_lengths[burst_id]
        matches = [(j, inter) for (bid, j), inter in intersections.items() if bid == burst_id]
        matched_region_ids = sorted(j for j, _ in matches)
        total_intersection = sum(inter for _, inter in matches)
        detection_success = total_intersection > 0

        matched_region_id = None
        start_err = end_err = None
        if matches:
            matched_region_id = max(matches, key=lambda t: t[1])[0]
            r_start, r_end = detected_regions[matched_region_id]
            start_err = r_start - b["true_start"]
            end_err = r_end - b["true_end"]

        entry = dict(b)  # pass through burst_id + any caller-supplied metadata
        entry.update({
            "true_burst_length": true_burst_length,
            "detection_success": detection_success,
            "matched_region_id": matched_region_id,
            "matched_region_ids": matched_region_ids,
            "intersection_length": total_intersection,
            "captured_signal_ratio": (total_intersection / true_burst_length) if true_burst_length > 0 else 0.0,
            "missed_sample_count": true_burst_length - total_intersection,
            "start_boundary_error": start_err,
            "end_boundary_error": end_err,
        })
        per_burst.append(entry)

    per_region = []
    for j, (r_start, r_end) in enumerate(detected_regions):
        detected_region_length = r_end - r_start
        matches = [(bid, inter) for (bid, jj), inter in intersections.items() if jj == j]
        matched_burst_ids = sorted(bid for bid, _ in matches)
        total_intersection = sum(inter for _, inter in matches)
        false_occupied = detected_region_length - total_intersection
        per_region.append({
            "region_id": j,
            "detected_start": r_start,
            "detected_end": r_end,
            "detected_length": detected_region_length,
            "matched_burst_ids": matched_burst_ids,
            "intersection_length": total_intersection,
            "false_occupied_sample_count": false_occupied,
            "extra_captured_noise_ratio": (false_occupied / detected_region_length) if detected_region_length > 0 else None,
        })

    num_truth_bursts = len(true_bursts)
    num_detected_regions = len(detected_regions)
    num_matched_bursts = sum(1 for pb in per_burst if pb["detection_success"])
    num_missed_bursts = num_truth_bursts - num_matched_bursts
    num_false_alarm_regions = sum(1 for pr in per_region if not pr["matched_burst_ids"])

    total_true_length = sum(true_burst_lengths.values())
    total_false_occupied = sum(pr["false_occupied_sample_count"] for pr in per_region)
    total_missed = sum(pb["missed_sample_count"] for pb in per_burst)
    background_length = n_samples - total_true_length

    matched_boundary_errors_abs = []
    for pb in per_burst:
        if pb["detection_success"]:
            matched_boundary_errors_abs.append(abs(pb["start_boundary_error"]))
            matched_boundary_errors_abs.append(abs(pb["end_boundary_error"]))
    matched_start_errors_abs = [abs(pb["start_boundary_error"]) for pb in per_burst if pb["detection_success"]]
    matched_end_errors_abs = [abs(pb["end_boundary_error"]) for pb in per_burst if pb["detection_success"]]

    aggregate = {
        "num_truth_bursts": num_truth_bursts,
        "num_detected_regions": num_detected_regions,
        "num_matched_bursts": num_matched_bursts,
        "num_missed_bursts": num_missed_bursts,
        "num_false_alarm_regions": num_false_alarm_regions,
        "detection_probability": (num_matched_bursts / num_truth_bursts) if num_truth_bursts > 0 else None,
        "false_alarm_region_rate": (num_false_alarm_regions / num_detected_regions) if num_detected_regions > 0 else None,
        "sample_level_false_positive_rate": (total_false_occupied / background_length) if background_length > 0 else None,
        "sample_level_false_negative_rate": (total_missed / total_true_length) if total_true_length > 0 else None,
        "mean_captured_signal_ratio": (
            sum(pb["captured_signal_ratio"] for pb in per_burst) / num_truth_bursts
        ) if num_truth_bursts > 0 else None,
        "mean_abs_start_boundary_error": (
            sum(matched_start_errors_abs) / len(matched_start_errors_abs)
        ) if matched_start_errors_abs else None,
        "mean_abs_end_boundary_error": (
            sum(matched_end_errors_abs) / len(matched_end_errors_abs)
        ) if matched_end_errors_abs else None,
        "mean_abs_boundary_error": (
            sum(matched_boundary_errors_abs) / len(matched_boundary_errors_abs)
        ) if matched_boundary_errors_abs else None,
    }

    return {"per_burst": per_burst, "per_region": per_region, "aggregate": aggregate}
