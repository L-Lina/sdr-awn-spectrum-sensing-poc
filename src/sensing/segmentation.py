"""Windowing of occupied regions into fixed-length segments."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def segment_regions(iq: np.ndarray, regions: List[Tuple[int, int]], seg_len: int) -> np.ndarray:
    """
    Slice each occupied region into non-overlapping seg_len windows.
    Tail samples that don't fill a full window are dropped and logged.
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
