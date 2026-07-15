"""Shared experiment configuration and argparse definition for the runners in experiments/."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    snr: float
    mod: str
    attack: str
    topk: int
    threshold_factor: float
    window_size: int
    min_region_len: int
    merge_gap: int
    burst_len: int
    output_dir: str
    dry_run: bool
    n_samples: int = 8192
    use_real_topk: bool = False


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Argparse shared by experiments/run_full_experiment.py and run_batch.py (single-run flags)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--snr", type=float, default=10.0, help="Synthetic burst SNR in dB")
    parser.add_argument("--mod", type=str, default="BPSK", help="Modulation label tag (cosmetic only in this phase)")
    parser.add_argument("--attack", type=str, default="none", help="Attack name placeholder, e.g. fgsm, pgd, none")
    parser.add_argument("--topk", type=int, default=50, help="Top-K FFT bins kept by the defense placeholder")
    parser.add_argument("--threshold-factor", type=float, default=5.0, help="Energy threshold = median power * this factor")
    parser.add_argument("--window-size", type=int, default=128, help="Segment length / energy-detection window; AWN expects 128")
    parser.add_argument("--min-region-len", type=int, default=None, help="Minimum occupied region length to keep (default: --window-size)")
    parser.add_argument("--merge-gap", type=int, default=0, help="Merge occupied regions separated by <= this many samples")
    parser.add_argument("--burst-len", type=int, default=600, help="Synthetic burst length in samples")
    parser.add_argument("--output-dir", type=str, default="results/run", help="Directory for summary.csv and sensing_plot.png")
    parser.add_argument("--dry-run", action="store_true", help="Run the placeholder pipeline (required in this phase)")
    parser.add_argument("--use-real-topk", action="store_true",
                        help="Route the Top-K defense through TopKAdapter (real fft_topk_denoise if torch is "
                             "available, else falls back to the numpy dummy with notes in summary.csv)")
    return parser


def args_to_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        snr=args.snr,
        mod=args.mod,
        attack=args.attack,
        topk=args.topk,
        threshold_factor=args.threshold_factor,
        window_size=args.window_size,
        min_region_len=args.min_region_len or args.window_size,
        merge_gap=args.merge_gap,
        burst_len=args.burst_len,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        use_real_topk=args.use_real_topk,
    )
