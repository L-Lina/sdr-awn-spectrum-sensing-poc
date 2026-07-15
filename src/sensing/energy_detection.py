"""
Sliding-window energy detection: mask -> raw contiguous regions -> merge
close regions -> filter by minimum length.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def energy_detect(iq: np.ndarray, window: int, threshold_factor: float) -> np.ndarray:
    """
    Sliding-window energy detector.
    Returns a boolean mask (same length as iq) marking samples considered 'occupied'.
    Threshold = median windowed power * threshold_factor (median is robust to a single burst).
    """
    n = len(iq)
    if n < window:
        raise ValueError(f"IQ stream ({n} samples) shorter than energy window ({window})")

    power = np.abs(iq) ** 2
    kernel = np.ones(window) / window
    smoothed = np.convolve(power, kernel, mode="same")

    noise_floor = float(np.median(smoothed))
    threshold = noise_floor * threshold_factor
    mask = smoothed > threshold

    print(
        f"[energy] noise_floor={noise_floor:.2e}, threshold={threshold:.2e}, "
        f"occupied_samples={int(mask.sum())}/{n}"
    )
    return mask


def mask_to_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Turn a boolean occupancy mask into contiguous (start, end) index ranges, no filtering."""
    diff = np.diff(mask.astype(np.int8))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0] + 1)

    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]

    return list(zip(starts, ends))


def merge_close_regions(regions: List[Tuple[int, int]], merge_gap: int) -> List[Tuple[int, int]]:
    """Merge consecutive regions whose gap is <= merge_gap samples."""
    if not regions or merge_gap <= 0:
        return list(regions)

    merged = [regions[0]]
    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def filter_by_min_length(regions: List[Tuple[int, int]], min_len: int) -> List[Tuple[int, int]]:
    """Drop regions shorter than min_len, with distinct errors for 'none at all' vs 'none long enough'."""
    if not regions:
        raise RuntimeError(
            "No occupied region detected at all. Lower --threshold-factor, check burst "
            "amplitude/SNR, or verify the capture actually contains a signal."
        )

    filtered = [(s, e) for s, e in regions if (e - s) >= min_len]
    if not filtered:
        too_short = [(s, e, e - s) for s, e in regions]
        raise RuntimeError(
            f"Occupied region(s) found but all shorter than --min-region-len={min_len} samples: "
            f"{too_short}. Lower --min-region-len, increase --merge-gap, or capture a longer burst."
        )

    print(f"[regions] {len(filtered)} occupied region(s) >= {min_len} samples: {filtered}")
    return filtered
