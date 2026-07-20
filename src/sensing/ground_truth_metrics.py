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
