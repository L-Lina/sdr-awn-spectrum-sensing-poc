"""
Single-run experiment entrypoint.

Phase 1 skeleton: only --dry-run is implemented (synthetic IQ through
placeholder AWN/attack/Top-K). Real model/attack/defense wiring lands in
later phases -- see docs/integration_plan.md.

Example:
    python3 experiments/run_full_experiment.py --dry-run --snr 0 --mod QPSK \\
        --attack fgsm --topk 10 --threshold-factor 5 --output-dir results/demo_run
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import args_to_config, build_arg_parser  # noqa: E402
from src.utils.pipeline import run_dry_run_experiment  # noqa: E402


def main() -> None:
    parser = build_arg_parser("SDR sensing -> AWN -> attack -> Top-K defense, single-run experiment")
    args = parser.parse_args()

    if not args.dry_run:
        parser.error(
            "Only --dry-run is supported in this phase (real AWN/attack/defense "
            "wiring is a later phase -- see docs/integration_plan.md)."
        )

    cfg = args_to_config(args)

    try:
        run_dry_run_experiment(cfg)
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
