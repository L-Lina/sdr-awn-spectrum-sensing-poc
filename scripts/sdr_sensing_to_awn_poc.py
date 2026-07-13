"""
SDR/GNU Radio spectrum sensing -> AWN AMC model input, proof of concept.

Pipeline:
    complex IQ stream (synthetic or from a GNU Radio .cfile capture)
      -> energy detection (find occupied regions)
      -> windowing into fixed-length segments
      -> per-segment normalization
      -> convert complex64 [N, window] -> float32 [N, 2, window]  (AWN input shape)
      -> run_awn_inference()  (placeholder - no torch, no real AWN model yet)

No hardware / GNU Radio / torch dependency. Only numpy is required.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# 1. IQ source: synthetic signal generator OR .cfile reader
# --------------------------------------------------------------------------

def generate_synthetic_iq(
    n_samples: int = 8192,
    burst_start: int = 3000,
    burst_len: int = 600,
    noise_std: float = 0.05,
    burst_amp: float = 1.0,
    freq_offset: float = 0.05,
    seed: Optional[int] = 0,
) -> np.ndarray:
    """Generate noise + a single occupied burst (complex64), simulating a GNU Radio IQ capture."""
    rng = np.random.default_rng(seed)

    noise = rng.normal(0, noise_std, n_samples) + 1j * rng.normal(0, noise_std, n_samples)
    iq = noise.astype(np.complex64)

    burst_end = min(burst_start + burst_len, n_samples)
    t = np.arange(burst_end - burst_start)
    carrier = np.exp(1j * 2 * np.pi * freq_offset * t)
    iq[burst_start:burst_end] += (burst_amp * carrier).astype(np.complex64)

    print(f"[gen] synthetic IQ: {n_samples} samples, burst at [{burst_start}:{burst_end}]")
    return iq


def load_iq_from_file(path: str) -> np.ndarray:
    """
    Load IQ samples from a raw complex64 .cfile.

    This is the intended hookup point for a GNU Radio flowgraph:
        UHD: USRP Source -> File Sink (output type = complex64) -> captured_iq.cfile
    """
    iq = np.fromfile(path, dtype=np.complex64)
    if iq.size == 0:
        raise ValueError(f"No samples read from '{path}' - file empty, wrong path, or wrong dtype")
    print(f"[load] read {iq.size} IQ samples from {path}")
    return iq


def validate_iq(iq: np.ndarray) -> np.ndarray:
    """Ensure IQ stream is complex and 1-D; raise rather than silently casting."""
    if not np.iscomplexobj(iq):
        raise TypeError(
            f"Expected a complex IQ stream, got dtype={iq.dtype}. "
            "If reading a .cfile, confirm the GNU Radio capture used complex64 (gr_complex), "
            "not real-valued or interleaved int16 samples."
        )
    if iq.dtype != np.complex64:
        print(f"[warn] input dtype is {iq.dtype}, casting to complex64")
        iq = iq.astype(np.complex64)
    if iq.ndim != 1:
        raise ValueError(f"Expected 1-D IQ stream, got shape={iq.shape}")
    return iq


# --------------------------------------------------------------------------
# 2. Energy detection
# --------------------------------------------------------------------------

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


def extract_occupied_regions(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    """Turn boolean occupancy mask into contiguous (start, end) index ranges, dropping short regions."""
    diff = np.diff(mask.astype(np.int8))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0] + 1)

    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]

    all_regions = list(zip(starts, ends))
    regions = [(s, e) for s, e in all_regions if (e - s) >= min_len]

    if not all_regions:
        raise RuntimeError(
            "No occupied region detected at all. Lower --threshold-factor, check burst "
            "amplitude, or verify the capture actually contains a signal."
        )
    if not regions:
        too_short = [(s, e, e - s) for s, e in all_regions]
        raise RuntimeError(
            f"Occupied region(s) found but all shorter than min_len={min_len} samples: {too_short}. "
            "Lower --window-size or capture a longer burst."
        )

    print(f"[regions] found {len(regions)} occupied region(s) >= {min_len} samples: {regions}")
    return regions


# --------------------------------------------------------------------------
# 3. Windowing into fixed-length segments
# --------------------------------------------------------------------------

def segment_regions(iq: np.ndarray, regions: List[Tuple[int, int]], seg_len: int) -> np.ndarray:
    """
    Slice each occupied region into non-overlapping seg_len windows.
    Regions shorter than seg_len are skipped with a warning (already filtered out by
    extract_occupied_regions when min_len=seg_len, but kept here for safety/reuse).
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


# --------------------------------------------------------------------------
# 4. Normalize + convert to AWN input shape [N, 2, window]
# --------------------------------------------------------------------------

def normalize_segments(segments: np.ndarray) -> np.ndarray:
    """Per-segment unit-average-power normalization."""
    power = np.mean(np.abs(segments) ** 2, axis=1, keepdims=True)
    power = np.maximum(power, 1e-12)
    return (segments / np.sqrt(power)).astype(np.complex64)


def to_awn_input(segments: np.ndarray, seg_len: int) -> np.ndarray:
    """Convert complex64 [N, seg_len] segments to AWN's expected float32 [N, 2, seg_len] array."""
    if segments.dtype != np.complex64:
        raise TypeError(f"Expected complex64 segments, got {segments.dtype}")
    if segments.ndim != 2 or segments.shape[1] != seg_len:
        raise ValueError(f"Expected shape [N, {seg_len}], got {segments.shape}")

    x = np.stack([segments.real, segments.imag], axis=1).astype(np.float32)  # [N, 2, seg_len]
    print(f"[awn_input] converted to shape={x.shape}, dtype={x.dtype}")
    return x


# --------------------------------------------------------------------------
# 5. AWN inference placeholder (no torch, no real model yet)
# --------------------------------------------------------------------------

def run_awn_inference(x: np.ndarray, n_classes: int = 11, seed: Optional[int] = 0) -> np.ndarray:
    """
    Placeholder for the AWN forward pass. Later this should be swapped for something like:

        import torch
        from awn.model import AWN
        model = AWN(...)
        model.load_state_dict(torch.load("awn.pth"))
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(x)).numpy()

    For now this returns random logits with the correct output shape (numpy only, no torch)
    so the rest of the pipeline (defense hooks, reporting) can be developed independently
    of the real model.
    """
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"AWN expects input [N, 2, window], got {x.shape}")

    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(x.shape[0], n_classes)).astype(np.float32)
    print(f"[PLACEHOLDER] run_awn_inference: input={x.shape} -> logits={logits.shape}")
    return logits


# --------------------------------------------------------------------------
# main pipeline
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SDR/GNU Radio spectrum sensing -> AWN [N, 2, window] input, proof of concept"
    )
    parser.add_argument("--demo", action="store_true", help="Generate a synthetic IQ stream instead of reading a file")
    parser.add_argument("--input", type=str, default=None, help="Path to a raw complex64 .cfile (GNU Radio File Sink output)")
    parser.add_argument("--window-size", type=int, default=128, help="Segment length in samples (also used as the energy-detection window); AWN expects 128")
    parser.add_argument("--threshold-factor", type=float, default=5.0, help="Energy detection threshold = median power * this factor")
    parser.add_argument("--output", type=str, default=None, help="Path to save the [N, 2, window] tensor as .npy")
    return parser


def run_pipeline(args: argparse.Namespace) -> np.ndarray:
    if args.input:
        iq = load_iq_from_file(args.input)
    elif args.demo:
        iq = generate_synthetic_iq()
    else:
        raise SystemExit("Must specify --demo (synthetic IQ) or --input <path.cfile>")

    iq = validate_iq(iq)

    mask = energy_detect(iq, window=args.window_size, threshold_factor=args.threshold_factor)
    regions = extract_occupied_regions(mask, min_len=args.window_size)
    segments = segment_regions(iq, regions, seg_len=args.window_size)
    segments = normalize_segments(segments)

    x = to_awn_input(segments, seg_len=args.window_size)
    logits = run_awn_inference(x)

    print("\n--- Pipeline summary ---")
    print(f"IQ stream length:        {len(iq)} samples")
    print(f"Detected occupied regions: {len(regions)} -> {regions}")
    print(f"Number of segments:      {x.shape[0]}")
    print(f"Output tensor shape:     {x.shape} ({x.dtype})")
    print(f"Dummy logits shape:      {logits.shape}")

    if args.output:
        np.save(args.output, x)
        print(f"\nSaved AWN input tensor to {args.output}")

    return x


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        run_pipeline(args)
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
