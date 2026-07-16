"""Shared experiment configuration and argparse definition for the runners in experiments/."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional


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
    # None means "use window_size" (legacy behavior, unchanged for anyone who
    # never passes --sensing-window-size). Resolved in
    # src/utils/pipeline.py:run_dry_run_experiment via
    # resolve_sensing_window_size() below, NOT at config-construction time,
    # so direct-API callers who build ExperimentConfig by hand (not through
    # args_to_config/run_batch.py) still get correct behavior.
    sensing_window_size: Optional[int] = None
    use_real_topk: bool = False
    use_real_awn: bool = False
    checkpoint: str = "external/adversarial-rf/2016.10a_AWN.pkl"
    device: str = "cpu"
    attack_eps: float = 0.03
    use_real_attack: bool = False
    attack_temperature: float = 1.0
    attack_diagnostics: bool = False


# ---------------------------------------------------------------------------
# Reusable boundary validators. Each raises a plain ValueError with a message
# of the form "<name> must be ..., got <value>" -- used both by the argparse
# type= helpers below (CLI-time errors) and by validate_experiment_config()
# (called from src/utils/pipeline.py as the adapter/algorithm-boundary guard
# for direct-API callers who construct ExperimentConfig without going through
# argparse at all, e.g. experiments/run_batch.py's own ExperimentConfig(...)
# construction). merge_gap and topk are intentionally not covered by any of
# this yet -- see docs/parameter_validation.md for why.
# ---------------------------------------------------------------------------

def require_positive_finite_float(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value}")
    return value


def require_finite_float(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def require_nonneg_finite_float(name: str, value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value}")
    return value


def require_positive_int(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def require_nonneg_int(name: str, value: int) -> int:
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value}")
    return value


def validate_experiment_config(cfg: ExperimentConfig) -> None:
    """Boundary validation for direct-API callers of run_dry_run_experiment(cfg)
    that bypass argparse entirely. Covers exactly the parameters with a
    finalized rule as of this round; merge_gap and topk are out of scope."""
    require_positive_finite_float("threshold_factor", cfg.threshold_factor)
    require_positive_int("window_size", cfg.window_size)
    if cfg.sensing_window_size is not None:
        require_positive_int("sensing_window_size", cfg.sensing_window_size)
    require_nonneg_int("min_region_len", cfg.min_region_len)
    require_positive_int("burst_len", cfg.burst_len)
    require_finite_float("snr_db", cfg.snr)
    require_nonneg_finite_float("attack_eps", cfg.attack_eps)
    require_positive_finite_float("attack_temperature", cfg.attack_temperature)


def resolve_sensing_window_size(window_size: int, sensing_window_size: Optional[int]) -> int:
    """--sensing-window-size controls only energy_detect's smoothing window;
    --window-size (legacy name) continues to control segment_regions'/
    to_awn_input's seg_len (segment length == AWN input temporal length,
    UNCHANGED). When --sensing-window-size is unset (None), the effective
    sensing window falls back to window_size -- this is the single point
    where that fallback happens, called from
    src/utils/pipeline.py:run_dry_run_experiment so it applies uniformly
    regardless of whether the caller went through argparse or built
    ExperimentConfig directly."""
    return window_size if sensing_window_size is None else sensing_window_size


# ---------------------------------------------------------------------------
# argparse type= factories built on the same validators above, so a CLI
# parse-time error and a direct-API ValueError use identical wording. Reused
# by both build_arg_parser() below and experiments/run_batch.py's own parser.
# ---------------------------------------------------------------------------

def arg_positive_finite_float(name: str):
    def _parse(raw: str) -> float:
        try:
            value = float(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} must be a positive finite number, got {raw!r}")
        try:
            return require_positive_finite_float(name, value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc))
    return _parse


def arg_finite_float(name: str):
    def _parse(raw: str) -> float:
        try:
            value = float(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} must be finite, got {raw!r}")
        try:
            return require_finite_float(name, value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc))
    return _parse


def arg_nonneg_finite_float(name: str):
    def _parse(raw: str) -> float:
        try:
            value = float(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} must be a finite non-negative number, got {raw!r}")
        try:
            return require_nonneg_finite_float(name, value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc))
    return _parse


def arg_positive_int(name: str):
    def _parse(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} must be a positive integer, got {raw!r}")
        try:
            return require_positive_int(name, value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc))
    return _parse


def arg_nonneg_int(name: str):
    def _parse(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} must be a non-negative integer, got {raw!r}")
        try:
            return require_nonneg_int(name, value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc))
    return _parse


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Argparse shared by experiments/run_full_experiment.py and run_batch.py (single-run flags)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--snr", type=arg_finite_float("snr_db"), default=10.0, help="Synthetic burst SNR in dB")
    parser.add_argument("--mod", type=str, default="BPSK", help="Modulation label tag (cosmetic only in this phase)")
    parser.add_argument("--attack", type=str, default="none", help="Attack name: none, fgsm, pgd, or cw")
    parser.add_argument("--attack-eps", type=arg_nonneg_finite_float("attack_eps"), default=0.03, help="Attack epsilon (Linf budget for fgsm/pgd)")
    parser.add_argument("--topk", type=int, default=50, help="Top-K FFT bins kept by the defense placeholder")
    parser.add_argument("--threshold-factor", type=arg_positive_finite_float("threshold_factor"), default=5.0, help="Energy threshold = median power * this factor")
    parser.add_argument("--window-size", type=arg_positive_int("window_size"), default=128,
                        help="Legacy name -- controls segment length AND AWN input temporal length "
                             "(segment_regions'/to_awn_input's seg_len). Real AWN checkpoint currently "
                             "expects 128 (not enforced here). Does NOT control energy-detection smoothing "
                             "window unless --sensing-window-size is left unset.")
    parser.add_argument("--sensing-window-size", type=arg_positive_int("sensing_window_size"), default=None,
                        help="Energy-detection smoothing window (energy_detect's window= argument), "
                             "independent of segment length / AWN input length. Defaults to --window-size "
                             "when unset, reproducing prior (coupled) behavior exactly.")
    parser.add_argument("--min-region-len", type=arg_nonneg_int("min_region_len"), default=None, help="Minimum occupied region length to keep (default: --window-size); 0 is allowed")
    parser.add_argument("--merge-gap", type=int, default=0, help="Merge occupied regions separated by <= this many samples")
    parser.add_argument("--burst-len", type=arg_positive_int("burst_len"), default=600, help="Synthetic burst length in samples")
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
    parser.add_argument("--attack-temperature", type=arg_positive_finite_float("attack_temperature"), default=1.0,
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
        sensing_window_size=args.sensing_window_size,
        min_region_len=(
            args.window_size if args.min_region_len is None else args.min_region_len
        ),
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
