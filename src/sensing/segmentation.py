"""Windowing of occupied regions into fixed-length segments."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def segment_regions(iq: np.ndarray, regions: List[Tuple[int, int]], seg_len: int) -> np.ndarray:
    """
    Slice each occupied region into non-overlapping seg_len windows, starting
    at each region's own start. Tail samples that don't fill a full window
    are dropped and logged.

    This is the ORIGINAL, unchanged segmentation behavior (kept exactly as-is
    for backward compatibility -- select_aligned_segments()'s "naive" policy
    below calls this function directly for its segment data, so naive-policy
    behavior is guaranteed byte-identical to every prior round's output).
    Does not know about true burst position -- see docs/parameter_validation.md
    section 17.x/18 for why relying on region-start-aligned segmentation
    degrades AMC accuracy even when the detected region fully covers the
    true burst (segment misaligns with the burst within the region).
    """
    segments = []

    for start, end in regions:
        region_len = end - start
        n_windows = region_len // seg_len
        if n_windows < 1:
            print(f"[warn] region [{start}:{end}] ({region_len} samples) < seg_len={seg_len}, skipped")
            continue

        for w in range(n_windows):
            s = start + w * seg_len
            segments.append(iq[s:s + seg_len])

        leftover = region_len - n_windows * seg_len
        if leftover > 0:
            print(f"[segment] region [{start}:{end}]: {n_windows} window(s), {leftover} leftover sample(s) dropped")

    if not segments:
        raise RuntimeError(f"No segments of length {seg_len} could be extracted from detected regions")

    segments = np.stack(segments).astype(np.complex64)
    print(f"[segment] {segments.shape[0]} windows of {seg_len} samples")
    return segments


def select_aligned_segments(
    iq: np.ndarray,
    regions: List[Tuple[int, int]],
    seg_len: int,
    policy: str = "naive",
    hop: int = 1,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Alignment-aware segment selection (docs/parameter_validation.md section
    18) -- addresses the root cause diagnosed in that section: segment_regions()
    always cuts its first window starting exactly at a region's own start,
    which is typically 53-61 samples before the true burst start (energy_detect's
    smoothing widens the region on the leading edge), so even a region with
    100% region-level captured_signal_ratio can yield an AWN input window that
    is only ~52-63% true-burst signal.

    Returns (segments [N, seg_len] complex64, selection_meta: one dict per
    segment, in the same order as `segments`), where each meta dict has:
    alignment_policy, segment_hop, candidate_count, selected_segment_start,
    selected_segment_end, selected_window_power, detected_region_start,
    detected_region_end, region_idx (index into the input `regions` list --
    used by callers, e.g. src/utils/pipeline.py, to reconstruct which
    detected region each segment came from without re-deriving it).

    policy="naive": IDENTICAL segment data to segment_regions(iq, regions,
    seg_len) (called directly, not reimplemented) -- can yield MULTIPLE
    non-overlapping segments per region if the region is long enough. This
    is the default, so any caller that never sets --alignment-policy gets
    byte-for-byte the same behavior as every prior round.

    policy="max-energy": exactly ONE selected segment PER region -- the
    seg_len window (among all `hop`-spaced sliding candidates within that
    region, hop=1 meaning every possible offset) with the highest mean
    power (mean(|x|^2) over the window). This is a DELIBERATE MINIMAL
    scope, not a general replacement for naive's multi-window-per-region
    case -- a region long enough for multiple non-overlapping windows still
    only contributes one segment under max-energy. Selection NEVER reads or
    references true_burst_start/true_burst_end (no ground truth is passed
    into this function at all) -- it is purely a function of `iq` amplitude
    within the detected region, satisfying the "must not depend on
    true_burst_start" requirement structurally, not just by convention.

    Raises the same RuntimeError as segment_regions() when zero segments
    result (e.g. every region shorter than seg_len).
    """
    if not isinstance(hop, int) or hop < 1:
        raise ValueError(f"hop must be a positive integer, got {hop!r}")
    if policy not in ("naive", "max-energy"):
        raise ValueError(f"Unknown alignment policy {policy!r}; choices: naive, max-energy")

    if policy == "naive":
        segments = segment_regions(iq, regions, seg_len)
        meta = []
        for region_idx, (r_start, r_end) in enumerate(regions):
            region_len = r_end - r_start
            if region_len < seg_len:
                continue
            n_windows = region_len // seg_len
            candidate_count = (region_len - seg_len) // hop + 1
            for w in range(n_windows):
                s = r_start + w * seg_len
                seg = iq[s:s + seg_len]
                meta.append({
                    "alignment_policy": "naive",
                    "segment_hop": hop,
                    "candidate_count": candidate_count,
                    "selected_segment_start": s,
                    "selected_segment_end": s + seg_len,
                    "selected_window_power": float(np.mean(np.abs(seg) ** 2)),
                    "detected_region_start": r_start,
                    "detected_region_end": r_end,
                    "region_idx": region_idx,
                })
        assert len(meta) == segments.shape[0], (
            f"naive-policy metadata count ({len(meta)}) != segment_regions() output count "
            f"({segments.shape[0]}) -- selection_meta iteration drifted from segment_regions()'s "
            "own logic; this is a bug in select_aligned_segments, not in segment_regions."
        )
        return segments, meta

    # policy == "max-energy"
    segments = []
    meta = []
    for region_idx, (r_start, r_end) in enumerate(regions):
        region_len = r_end - r_start
        if region_len < seg_len:
            print(f"[warn] region [{r_start}:{r_end}] ({region_len} samples) < seg_len={seg_len}, skipped")
            continue

        candidate_count = (region_len - seg_len) // hop + 1
        best_start, best_power, best_seg = None, -1.0, None
        for cand_start in range(r_start, r_end - seg_len + 1, hop):
            cand = iq[cand_start:cand_start + seg_len]
            power = float(np.mean(np.abs(cand) ** 2))
            if power > best_power:
                best_power = power
                best_start = cand_start
                best_seg = cand

        segments.append(best_seg)
        meta.append({
            "alignment_policy": "max-energy",
            "segment_hop": hop,
            "candidate_count": candidate_count,
            "selected_segment_start": best_start,
            "selected_segment_end": best_start + seg_len,
            "selected_window_power": best_power,
            "detected_region_start": r_start,
            "detected_region_end": r_end,
            "region_idx": region_idx,
        })
        print(f"[segment][max-energy] region [{r_start}:{r_end}]: {candidate_count} candidate(s) "
              f"(hop={hop}), selected [{best_start}:{best_start + seg_len}] power={best_power:.6e}")

    if not segments:
        raise RuntimeError(f"No segments of length {seg_len} could be extracted from detected regions")

    segments = np.stack(segments).astype(np.complex64)
    print(f"[segment] {segments.shape[0]} windows of {seg_len} samples (policy=max-energy)")
    return segments, meta
