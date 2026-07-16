"""
Batch experiment runner skeleton: sweeps a parameter grid through the same
dry-run pipeline used by run_full_experiment.py, one output subdirectory per
combination, plus one aggregated batch_summary.csv.

Phase 1 skeleton only -- runs sequentially, no parallelization yet (TODO:
phase 2+). Real (non-dry-run) mode is not implemented yet either.

Example:
    python3 experiments/run_batch.py --dry-run \\
        --snr-list 0,10 --mod-list BPSK,QPSK --attack-list none,fgsm --topk-list 10,50 \\
        --output-dir results/batch_demo
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402

from src.utils.config import ExperimentConfig  # noqa: E402
from src.utils.csv_writer import write_summary_csv  # noqa: E402
from src.utils.pipeline import run_dry_run_experiment  # noqa: E402


def _parse_list(raw: str, cast):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def build_batch_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parameter-grid sweep over the dry-run pipeline")
    parser.add_argument("--snr-list", type=str, default="0,10", help="Comma-separated SNR values in dB")
    parser.add_argument("--mod-list", type=str, default="BPSK,QPSK", help="Comma-separated modulation tags")
    parser.add_argument("--attack-list", type=str, default="none,fgsm", help="Comma-separated attack names")
    parser.add_argument("--topk-list", type=str, default="10,50", help="Comma-separated Top-K values")
    parser.add_argument("--threshold-factor", type=float, default=5.0)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--min-region-len", type=int, default=None)
    parser.add_argument("--merge-gap", type=int, default=0)
    parser.add_argument("--burst-len", type=int, default=600)
    parser.add_argument("--output-dir", type=str, default="results/batch_run", help="Base directory for per-combo subdirs + batch_summary.csv")
    parser.add_argument("--dry-run", action="store_true", help="Run the placeholder pipeline (required in this phase)")
    parser.add_argument("--use-real-topk", action="store_true",
                        help="Route the Top-K defense through TopKAdapter (real fft_topk_denoise if torch is "
                             "available, else falls back to the numpy dummy with notes in summary.csv)")
    parser.add_argument("--use-real-awn", action="store_true",
                        help="Route AWN inference through AWNModelAdapter (real AWN model + checkpoint if "
                             "torch is available, else falls back to the numpy dummy with notes in summary.csv)")
    parser.add_argument("--use-real-attack", action="store_true",
                        help="Route the attack through AttackAdapter (real torchattacks-based attack if torch, "
                             "torchattacks, and a real AWN model are all available, else falls back to the "
                             "numpy dummy with notes in summary.csv)")
    parser.add_argument("--attack-eps", type=float, default=0.03, help="Attack epsilon (Linf budget for fgsm/pgd)")
    parser.add_argument("--checkpoint", type=str, default="external/adversarial-rf/2016.10a_AWN.pkl",
                        help="Path to the AWN checkpoint (.pkl) used when --use-real-awn is set")
    parser.add_argument("--device", type=str, default="cpu", help="torch device for real AWN inference (cpu or cuda)")
    parser.add_argument("--attack-temperature", type=float, default=1.0,
                        help="Positive temperature T dividing AWN logits inside the attack's internal loss "
                             "only; clean/attacked/defended inference always use raw logits. T=1.0 reproduces "
                             "prior behavior (must be > 0). Applied uniformly to every combo in this batch.")
    parser.add_argument("--attack-diagnostics", action="store_true",
                        help="Run an extra diagnostic-only autograd.grad pass per real attack call to report "
                             "gradient nonzero-count/maxabs in summary.csv. Adds runtime cost per segment; "
                             "leave off for large batches.")
    return parser


def main() -> None:
    parser = build_batch_arg_parser()
    args = parser.parse_args()

    if not args.dry_run:
        parser.error(
            "Only --dry-run is supported in this phase (real AWN/attack/defense "
            "wiring is a later phase -- see docs/integration_plan.md)."
        )

    snrs = _parse_list(args.snr_list, float)
    mods = _parse_list(args.mod_list, str)
    attacks = _parse_list(args.attack_list, str)
    topks = _parse_list(args.topk_list, int)
    min_region_len = (
        args.window_size if args.min_region_len is None else args.min_region_len
    )

    base_dir = Path(args.output_dir)
    batch_rows = []

    combos = list(itertools.product(snrs, mods, attacks, topks))
    print(f"[batch] running {len(combos)} combination(s)")

    for snr, mod, attack, topk in combos:
        run_dir = base_dir / f"snr{snr}_mod{mod}_attack{attack}_topk{topk}"
        cfg = ExperimentConfig(
            snr=snr,
            mod=mod,
            attack=attack,
            topk=topk,
            threshold_factor=args.threshold_factor,
            window_size=args.window_size,
            min_region_len=min_region_len,
            merge_gap=args.merge_gap,
            burst_len=args.burst_len,
            output_dir=str(run_dir),
            dry_run=True,
            use_real_topk=args.use_real_topk,
            use_real_awn=args.use_real_awn,
            checkpoint=args.checkpoint,
            device=args.device,
            attack_eps=args.attack_eps,
            use_real_attack=args.use_real_attack,
            attack_temperature=args.attack_temperature,
            attack_diagnostics=args.attack_diagnostics,
        )

        try:
            result = run_dry_run_experiment(cfg)
        except (ValueError, TypeError, RuntimeError) as exc:
            print(f"[batch][ERROR] combo snr={snr} mod={mod} attack={attack} topk={topk}: {exc}", file=sys.stderr)
            continue

        batch_rows.append({
            "snr_db": snr,
            "mod": mod,
            "attack": attack,
            "topk": topk,
            "attack_temperature": args.attack_temperature,
            "n_segments": result["n_segments"],
            "output_dir": result["output_dir"],
        })

    if batch_rows:
        write_summary_csv(base_dir / "batch_summary.csv", batch_rows)
    else:
        print("[batch] no successful runs -- batch_summary.csv not written")


if __name__ == "__main__":
    main()
