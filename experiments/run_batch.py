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

from src.utils.batch_aggregation import run_batch_combos  # noqa: E402
from src.utils.config import (  # noqa: E402
    ExperimentConfig,
    _parse_comma_list,
    arg_nonneg_finite_float,
    arg_nonneg_int,
    arg_positive_finite_float,
    arg_positive_int,
)


def _parse_list(raw: str, cast):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def build_batch_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parameter-grid sweep over the dry-run pipeline")
    parser.add_argument("--snr-list", type=str, default="0,10", help="Comma-separated SNR values in dB")
    parser.add_argument("--mod-list", type=str, default="BPSK,QPSK", help="Comma-separated modulation tags")
    parser.add_argument("--attack-list", type=str, default="none,fgsm", help="Comma-separated attack names")
    parser.add_argument("--topk-list", type=str, default="10,50", help="Comma-separated Top-K values")
    parser.add_argument("--threshold-factor", type=arg_positive_finite_float("threshold_factor"), default=5.0)
    parser.add_argument("--window-size", type=arg_positive_int("window_size"), default=128,
                        help="Legacy name -- controls segment length AND AWN input temporal length. "
                             "Does NOT control energy-detection smoothing window unless "
                             "--sensing-window-size is left unset.")
    parser.add_argument("--sensing-window-size", type=arg_positive_int("sensing_window_size"), default=None,
                        help="Energy-detection smoothing window, independent of segment length / AWN input "
                             "length. Defaults to --window-size when unset (prior behavior unchanged). "
                             "Applied uniformly to every combo in this batch.")
    parser.add_argument("--min-region-len", type=arg_nonneg_int("min_region_len"), default=None)
    parser.add_argument("--merge-gap", type=int, default=0)
    parser.add_argument("--burst-len", type=arg_positive_int("burst_len"), default=600)
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
    parser.add_argument("--attack-eps", type=arg_nonneg_finite_float("attack_eps"), default=0.03, help="Attack epsilon (Linf budget for fgsm/pgd)")
    parser.add_argument("--checkpoint", type=str, default="external/adversarial-rf/2016.10a_AWN.pkl",
                        help="Path to the AWN checkpoint (.pkl) used when --use-real-awn is set")
    parser.add_argument("--device", type=str, default="cpu", help="torch device for real AWN inference (cpu or cuda)")
    parser.add_argument("--attack-temperature", type=arg_positive_finite_float("attack_temperature"), default=1.0,
                        help="Positive temperature T dividing AWN logits inside the attack's internal loss "
                             "only; clean/attacked/defended inference always use raw logits. T=1.0 reproduces "
                             "prior behavior (must be > 0). Applied uniformly to every combo in this batch.")
    parser.add_argument("--attack-diagnostics", action="store_true",
                        help="Run an extra diagnostic-only autograd.grad pass per real attack call to report "
                             "gradient nonzero-count/maxabs in summary.csv. Adds runtime cost per segment; "
                             "leave off for large batches.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Global reproducibility seed: seeds random/numpy/torch(+cuda if available) at the "
                             "start of every combo's run_dry_run_experiment call, and is threaded through to "
                             "synthetic-IQ generation and every dummy/real attack call. Applied uniformly to "
                             "every combo in this batch. Default 0 reproduces prior (hardcoded) behavior.")
    parser.add_argument("--cw-c", type=arg_positive_finite_float("cw_c"), default=1.0,
                        help="CW-ONLY. torchattacks.CW's c (misclassification-loss weight). Ignored entirely "
                             "by fgsm/pgd. NOT the same knob as --attack-eps, which CW does not use at all. "
                             "Applied uniformly to every combo in this batch.")
    parser.add_argument("--cw-steps", type=arg_positive_int("cw_steps"), default=20,
                        help="CW-ONLY. torchattacks.CW's optimization step count. Ignored entirely by fgsm/pgd. "
                             "Applied uniformly to every combo in this batch.")
    parser.add_argument("--cw-lr", type=arg_positive_finite_float("cw_lr"), default=0.01,
                        help="CW-ONLY. torchattacks.CW's Adam learning rate. Ignored entirely by fgsm/pgd. "
                             "NOT the same knob as --attack-eps, which CW does not use at all. Applied "
                             "uniformly to every combo in this batch.")
    parser.add_argument("--iq-source", type=str, choices=["synthetic", "radioml"], default="synthetic",
                        help="'synthetic' (default) or 'radioml' -- see run_full_experiment.py's --iq-source "
                             "help. Applied uniformly to every combo in this batch; --snr-list/--mod-list "
                             "still drive the combo grid but are unused in radioml mode.")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="RADIOML-ONLY, REQUIRED when --iq-source radioml. Applied uniformly to every combo.")
    parser.add_argument("--dataset-mod", type=str, default=None,
                        help="RADIOML-ONLY, REQUIRED when --iq-source radioml. Applied uniformly to every combo "
                             "(NOT swept per --mod-list entry -- this is a single fixed value for the batch).")
    parser.add_argument("--dataset-snr", type=int, default=None,
                        help="RADIOML-ONLY, REQUIRED when --iq-source radioml. Applied uniformly to every combo "
                             "(NOT swept per --snr-list entry -- this is a single fixed value for the batch).")
    parser.add_argument("--sample-index", type=arg_nonneg_int("sample_index"), default=0,
                        help="RADIOML-ONLY. Applied uniformly to every combo in this batch.")
    parser.add_argument("--embed-snr-margin", type=arg_positive_finite_float("embed_snr_margin"), default=20.0,
                        help="RADIOML-ONLY. Applied uniformly to every combo in this batch.")
    parser.add_argument("--num-bursts", type=arg_positive_int("num_bursts"), default=1,
                        help="RADIOML-ONLY. See run_full_experiment.py's --num-bursts help. Applied uniformly "
                             "to every combo in this batch.")
    parser.add_argument("--dataset-mod-list", type=str, default=None,
                        help="MULTI-BURST-ONLY, REQUIRED when --num-bursts > 1. Applied uniformly to every combo.")
    parser.add_argument("--dataset-snr-list", type=str, default=None,
                        help="MULTI-BURST-ONLY, REQUIRED when --num-bursts > 1. Applied uniformly to every combo.")
    parser.add_argument("--sample-index-list", type=str, default=None,
                        help="MULTI-BURST-ONLY, REQUIRED when --num-bursts > 1. Applied uniformly to every combo.")
    parser.add_argument("--min-burst-gap", type=arg_nonneg_int("min_burst_gap"), default=50,
                        help="MULTI-BURST-ONLY. Applied uniformly to every combo in this batch.")
    parser.add_argument("--max-burst-gap", type=arg_nonneg_int("max_burst_gap"), default=50,
                        help="MULTI-BURST-ONLY. Applied uniformly to every combo in this batch.")
    parser.add_argument("--burst-gap-list", type=str, default=None,
                        help="MULTI-BURST-ONLY. See run_full_experiment.py's --burst-gap-list help. Applied "
                             "uniformly to every combo in this batch.")
    parser.add_argument("--burst-power-scale-list", type=str, default=None,
                        help="MULTI-BURST-ONLY. See run_full_experiment.py's --burst-power-scale-list help. "
                             "Applied uniformly to every combo in this batch.")
    parser.add_argument("--alignment-policy", type=str, choices=["naive", "max-energy"], default="naive",
                        help="Segment-alignment policy. See run_full_experiment.py's --alignment-policy help "
                             "and docs/parameter_validation.md section 18. Applied uniformly to every combo.")
    parser.add_argument("--segment-hop", type=arg_positive_int("segment_hop"), default=1,
                        help="Sliding-window step for max-energy's candidate search. Applied uniformly to "
                             "every combo in this batch.")
    parser.add_argument("--awn-preprocess", type=str, choices=["legacy-unit-power", "radioml-native"],
                        default="legacy-unit-power",
                        help="AWN-input-boundary preprocessing. See run_full_experiment.py's --awn-preprocess "
                             "help and docs/parameter_validation.md section 19. Applied uniformly to every combo.")
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

    combo_tuples = list(itertools.product(snrs, mods, attacks, topks))
    print(f"[batch] running {len(combo_tuples)} combination(s)")

    combos = [
        {"snr_db": snr, "mod": mod, "attack": attack, "topk": topk}
        for snr, mod, attack, topk in combo_tuples
    ]

    def build_cfg(combo: dict, run_dir: Path) -> ExperimentConfig:
        return ExperimentConfig(
            snr=combo["snr_db"],
            mod=combo["mod"],
            attack=combo["attack"],
            topk=combo["topk"],
            threshold_factor=args.threshold_factor,
            window_size=args.window_size,
            sensing_window_size=args.sensing_window_size,
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
            seed=args.seed,
            cw_c=args.cw_c,
            cw_steps=args.cw_steps,
            cw_lr=args.cw_lr,
            iq_source=args.iq_source,
            dataset_path=args.dataset_path,
            dataset_mod=args.dataset_mod,
            dataset_snr=args.dataset_snr,
            sample_index=args.sample_index,
            embed_snr_margin=args.embed_snr_margin,
            num_bursts=args.num_bursts,
            dataset_mod_list=_parse_comma_list(args.dataset_mod_list, str, "dataset_mod_list"),
            dataset_snr_list=_parse_comma_list(args.dataset_snr_list, int, "dataset_snr_list"),
            sample_index_list=_parse_comma_list(args.sample_index_list, int, "sample_index_list"),
            min_burst_gap=args.min_burst_gap,
            max_burst_gap=args.max_burst_gap,
            burst_gap_list=_parse_comma_list(args.burst_gap_list, int, "burst_gap_list"),
            burst_power_scale_list=_parse_comma_list(args.burst_power_scale_list, float, "burst_power_scale_list"),
            alignment_policy=args.alignment_policy,
            segment_hop=args.segment_hop,
            awn_preprocess=args.awn_preprocess,
        )

    run_batch_combos(base_dir, combos, build_cfg)


if __name__ == "__main__":
    main()
