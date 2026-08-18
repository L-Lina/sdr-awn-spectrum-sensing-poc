"""
Offline prototype of a chunked/stateful energy detector, built to answer one
question empirically: does processing a long IQ stream in fixed-size chunks
with a rolling buffer produce the same occupied-region/segment decisions as
one-shot offline sensing on the whole stream at once?

This is explicitly a PROTOTYPE for that comparison, not a replacement for
the formal pipeline. src/utils/pipeline.py and src/sensing/energy_detection.py
are not modified or called from here in a way that changes their behavior --
StreamingDetector re-implements the same energy_detect() formula
(power -> moving average -> median noise floor -> threshold -> mask) so it
can be run incrementally over chunks, then reuses mask_to_regions,
merge_close_regions, and filter_by_min_length UNMODIFIED from
src/sensing/energy_detection.py for the region-forming step, and
select_aligned_segments UNMODIFIED from src/sensing/segmentation.py for
window selection.

Known, explicitly unimplemented in this prototype (do not claim otherwise):
  - Cross-chunk event splitting is handled only by carrying a small tail
    buffer forward (see `carry_samples`); an event whose true extent is
    larger than one chunk + carry buffer will not be reconstructed correctly.
  - No overlapping-window event deduplication beyond the simple region-merge
    already provided by merge_close_regions.
  - No persistent event ID, timestamp, or refractory/guard period tracking.
  - Noise-floor estimation is recomputed per chunk (using only that chunk's
    own samples plus the carried tail), not maintained as a running estimate
    across the whole stream -- this is a real difference from the offline
    path's single whole-stream median and is expected to cause boundary
    disagreement for chunks that contain little or no signal.
  - Not a live/real-time detector: there is no actual streaming I/O, no
    hardware source, and no timing guarantees. It processes an in-memory
    numpy array split into slices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from src.sensing.energy_detection import filter_by_min_length, mask_to_regions, merge_close_regions


@dataclass
class StreamingDetectorConfig:
    chunk_size: int
    window: int
    threshold_factor: float
    merge_gap: int
    min_region_len: int
    carry_samples: int = 0  # samples carried from the tail of one chunk into the next, for cross-chunk continuity


@dataclass
class StreamingEvent:
    start: int
    end: int
    chunk_index: int  # index of the chunk in which this event was FIRST observed


class StreamingDetector:
    """Processes a long IQ array in fixed-size chunks, maintaining a small
    carry-forward buffer so a region straddling a chunk boundary by at most
    `carry_samples` samples can still be detected as one contiguous region.
    Each chunk's noise floor is estimated independently from that chunk's
    own (carry-buffer-extended) samples -- this is the main behavioral
    difference from the offline whole-stream median and is the primary
    source of any offline/streaming disagreement measured by
    experiments/test_streaming_sensing.py.
    """

    def __init__(self, config: StreamingDetectorConfig) -> None:
        self.config = config
        self._carry: np.ndarray = np.zeros(0, dtype=np.complex64)
        self._carry_global_offset = 0  # global sample index of self._carry[0]
        self.events: List[StreamingEvent] = []
        self._chunk_index = 0

    def _detect_mask(self, chunk_with_carry: np.ndarray) -> np.ndarray:
        """Same formula as src/sensing/energy_detection.py:energy_detect(),
        re-implemented here (not imported) so it can run on a chunk shorter
        than the formal pipeline's usual multi-thousand-sample streams
        without raising energy_detect()'s length precondition."""
        power = np.abs(chunk_with_carry) ** 2
        window = min(self.config.window, len(chunk_with_carry))
        kernel = np.ones(window) / window
        smoothed = np.convolve(power, kernel, mode="same")
        noise_floor = float(np.median(smoothed))
        threshold = noise_floor * self.config.threshold_factor
        return smoothed > threshold

    def process_chunk(self, chunk: np.ndarray) -> List[StreamingEvent]:
        """Feed one chunk in. Returns events finalized by this call (an
        event within the carry buffer that might still extend into the
        NEXT chunk is not finalized yet -- see process_chunk's handling of
        the trailing region)."""
        combined = np.concatenate([self._carry, chunk]).astype(np.complex64)
        combined_global_start = self._carry_global_offset

        mask = self._detect_mask(combined)
        raw_regions = mask_to_regions(mask)
        merged = merge_close_regions(raw_regions, merge_gap=self.config.merge_gap)
        try:
            kept = filter_by_min_length(merged, min_len=self.config.min_region_len)
        except RuntimeError:
            kept = []

        new_events = []
        trailing_region = None
        for s, e in kept:
            global_s = s + combined_global_start
            global_e = e + combined_global_start
            # a region touching the very end of `combined` might continue
            # into the next chunk -- don't finalize it yet, carry it forward.
            if e >= len(combined) - 1:
                trailing_region = (global_s, global_e)
                continue
            ev = StreamingEvent(start=global_s, end=global_e, chunk_index=self._chunk_index)
            new_events.append(ev)
            self.events.append(ev)

        # carry forward: the trailing region (if any) plus a fixed tail
        # window, so the next chunk's combined buffer can pick up where
        # this one left off.
        carry_len = self.config.carry_samples
        if trailing_region is not None:
            carry_start_local = max(0, trailing_region[0] - combined_global_start)
            self._carry = combined[carry_start_local:]
            self._carry_global_offset = combined_global_start + carry_start_local
        else:
            tail = combined[-carry_len:] if carry_len > 0 else combined[len(combined):]
            self._carry = tail
            self._carry_global_offset = combined_global_start + len(combined) - len(tail)

        self._chunk_index += 1
        return new_events

    def finalize(self) -> List[StreamingEvent]:
        """Call after the last chunk to flush the final carry buffer as
        one last region check (in case a trailing region does not extend
        into any further chunk because the stream simply ended)."""
        if len(self._carry) >= self.config.min_region_len:
            mask = self._detect_mask(self._carry)
            raw_regions = mask_to_regions(mask)
            merged = merge_close_regions(raw_regions, merge_gap=self.config.merge_gap)
            try:
                kept = filter_by_min_length(merged, min_len=self.config.min_region_len)
            except RuntimeError:
                kept = []
            for s, e in kept:
                ev = StreamingEvent(start=s + self._carry_global_offset, end=e + self._carry_global_offset,
                                     chunk_index=self._chunk_index)
                self.events.append(ev)
        return self.events


def run_streaming(iq: np.ndarray, config: StreamingDetectorConfig) -> List[StreamingEvent]:
    """Convenience driver: splits `iq` into config.chunk_size slices, feeds
    them through a fresh StreamingDetector, and returns the finalized event
    list. Does not implement any I/O -- `iq` is assumed already fully in
    memory (this is a rolling-buffer PROCESSING prototype, not a live
    streaming source)."""
    det = StreamingDetector(config)
    n = len(iq)
    for start in range(0, n, config.chunk_size):
        chunk = iq[start:start + config.chunk_size]
        det.process_chunk(chunk)
    det.finalize()
    return det.events
