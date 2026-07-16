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
    use_real_awn: bool = False
    checkpoint: str = "external/adversarial-rf/2016.10a_AWN.pkl"
    device: str = "cpu"
    attack_eps: float = 0.03
    use_real_attack: bool = False
    attack_temperature: float = 1.0
    attack_diagnostics: bool = False


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Argparse shared by experiments/run_full_experiment.py and run_batch.py (single-run flags)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--snr", type=float, default=10.0, help="Synthetic burst SNR in dB")
    parser.add_argument("--mod", type=str, default="BPSK", help="Modulation label tag (cosmetic only in this phase)")
    parser.add_argument("--attack", type=str, default="none", help="Attack name: none, fgsm, pgd, or cw")
    parser.add_argument("--attack-eps", type=float, default=0.03, help="Attack epsilon (Linf budget for fgsm/pgd)")
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
    parser.add_argument("--use-real-awn", action="store_true",
                        help="Route AWN inference through AWNModelAdapter (real AWN model + checkpoint if "
                             "torch is available, else falls back to the numpy dummy with notes in summary.csv)")
    parser.add_argument("--checkpoint", type=str, default="external/adversarial-rf/2016.10a_AWN.pkl",
                        help="Path to the AWN checkpoint (.pkl) used when --use-real-awn is set")
    parser.add_argument("--device", type=str, default="cpu", help="torch device for real AWN inference (cpu or cuda)")
    parser.add_argument("--use-real-attack", action="store_true",
                        help="Route the attack through AttackAdapter (real torchattacks-based attack if torch, "
                             "torchattacks, and a real AWN model are all available, else falls back to the "
                             "numpy dummy with notes in summary.csv)")
    parser.add_argument("--attack-temperature", type=float, default=1.0,
                        help="Positive temperature T dividing AWN logits inside the attack's internal loss "
                             "only (attack_logits = logits / T); clean/attacked/defended inference elsewhere "
                             "always use raw logits. T=1.0 reproduces prior behavior (must be > 0).")
    parser.add_argument("--attack-diagnostics", action="store_true",
                        help="Run an extra diagnostic-only autograd.grad pass per real attack call to report "
                             "gradient nonzero-count/maxabs in summary.csv. Adds runtime cost per segment; "
                             "leave off for large batches.")
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
        use_real_awn=args.use_real_awn,
        checkpoint=args.checkpoint,
        device=args.device,
        attack_eps=args.attack_eps,
        use_real_attack=args.use_real_attack,
        attack_temperature=args.attack_temperature,
        attack_diagnostics=args.attack_diagnostics,
    )
