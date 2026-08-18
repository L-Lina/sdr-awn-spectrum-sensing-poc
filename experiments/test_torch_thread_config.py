"""
Unit / round-trip tests for the torch_num_threads config surface added this
round: src/utils/config.py (ExperimentConfig.torch_num_threads field,
validate_experiment_config's positive-int-or-None check, the --torch-threads
CLI flag, args_to_config wiring) and src/utils/pipeline.py (run_dry_run_
experiment's torch.set_num_threads() call site and the torch_num_threads/
torch_actual_num_threads fields recorded into every summary dict).

Explicitly verifies backward compatibility: omitting --torch-threads (or
leaving ExperimentConfig.torch_num_threads at its default) must leave
torch's own/environment thread count completely untouched, and every
Phase 0-4 default behavior must be unaffected.

Run directly:
    python experiments/test_torch_thread_config.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import ExperimentConfig, args_to_config, build_arg_parser, validate_experiment_config  # noqa: E402
from src.utils.pipeline import run_dry_run_experiment  # noqa: E402

PASS = []
FAIL = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    (PASS if cond else FAIL).append(name)
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))


def base_cfg(output_dir: str, **overrides) -> ExperimentConfig:
    # threshold_factor=1.5 / burst_len=600 / min_region_len=0 reliably
    # produces a detected region for the default synthetic IQ source (same
    # fixed params as experiments/run_dummy_fallback_smoke.py's FIXED dict),
    # so the e2e checks below (9-11) exercise the full pipeline through to
    # summary.csv, not just the early-return "no region detected" branch.
    kwargs = dict(
        snr=0.0, mod="BPSK", attack="fgsm", topk=10, threshold_factor=1.5,
        window_size=128, min_region_len=0, merge_gap=0, burst_len=600,
        output_dir=output_dir, dry_run=True,
    )
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def main() -> None:
    import torch

    default_threads_at_start = torch.get_num_threads()

    with tempfile.TemporaryDirectory(prefix="torch_thread_config_test_") as tmp:
        tmp_path = Path(tmp)

        # 1. dataclass default is None
        cfg = base_cfg(str(tmp_path / "d1"))
        check("dataclass default torch_num_threads is None", cfg.torch_num_threads is None)

        # 2. validate_experiment_config accepts None
        try:
            validate_experiment_config(cfg)
            check("validate accepts torch_num_threads=None", True)
        except Exception as exc:  # noqa: BLE001
            check("validate accepts torch_num_threads=None", False, f"raised {exc!r}")

        # 3. validate_experiment_config accepts a positive int
        cfg_pos = base_cfg(str(tmp_path / "d2"), torch_num_threads=3)
        try:
            validate_experiment_config(cfg_pos)
            check("validate accepts torch_num_threads=3", True)
        except Exception as exc:  # noqa: BLE001
            check("validate accepts torch_num_threads=3", False, f"raised {exc!r}")

        # 4. validate_experiment_config rejects 0
        cfg_zero = base_cfg(str(tmp_path / "d3"), torch_num_threads=0)
        try:
            validate_experiment_config(cfg_zero)
            check("validate rejects torch_num_threads=0", False, "did not raise")
        except ValueError:
            check("validate rejects torch_num_threads=0", True)

        # 5. validate_experiment_config rejects negative
        cfg_neg = base_cfg(str(tmp_path / "d4"), torch_num_threads=-5)
        try:
            validate_experiment_config(cfg_neg)
            check("validate rejects torch_num_threads=-5", False, "did not raise")
        except ValueError:
            check("validate rejects torch_num_threads=-5", True)

        # 6. CLI round-trip: omitting --torch-threads yields None
        parser = build_arg_parser("test")
        args = parser.parse_args([
            "--dry-run", "--snr", "0", "--mod", "BPSK", "--attack", "fgsm",
            "--topk", "10", "--threshold-factor", "5", "--output-dir", str(tmp_path / "d5"),
        ])
        cfg_from_cli = args_to_config(args)
        check("CLI omitted --torch-threads -> config field is None", cfg_from_cli.torch_num_threads is None)

        # 7. CLI round-trip: --torch-threads 4 yields 4
        args2 = parser.parse_args([
            "--dry-run", "--snr", "0", "--mod", "BPSK", "--attack", "fgsm",
            "--topk", "10", "--threshold-factor", "5", "--output-dir", str(tmp_path / "d6"),
            "--torch-threads", "4",
        ])
        cfg_from_cli2 = args_to_config(args2)
        check("CLI --torch-threads 4 -> config field is 4", cfg_from_cli2.torch_num_threads == 4)

        # 8. CLI rejects --torch-threads 0 at parse time (arg_positive_int)
        try:
            parser.parse_args([
                "--dry-run", "--snr", "0", "--mod", "BPSK", "--attack", "fgsm",
                "--topk", "10", "--threshold-factor", "5", "--output-dir", str(tmp_path / "d7"),
                "--torch-threads", "0",
            ])
            check("CLI rejects --torch-threads 0", False, "did not raise SystemExit")
        except SystemExit:
            check("CLI rejects --torch-threads 0", True)

        # 9. end-to-end: omitting torch_num_threads leaves torch's actual
        # thread count exactly as it was before the call (backward compat).
        torch.set_num_threads(default_threads_at_start)
        cfg_e2e_default = base_cfg(
            str(tmp_path / "e2e_default"), use_real_awn=False, use_real_attack=False, use_real_topk=False,
        )
        run_dry_run_experiment(cfg_e2e_default)
        check(
            "omitting torch_num_threads leaves torch.get_num_threads() unchanged",
            torch.get_num_threads() == default_threads_at_start,
            f"expected {default_threads_at_start}, got {torch.get_num_threads()}",
        )

        # 10. end-to-end: explicit torch_num_threads=2 actually changes
        # torch.get_num_threads() to 2, and the summary dict records both
        # the requested and actual value.
        cfg_e2e_t2 = base_cfg(
            str(tmp_path / "e2e_t2"), use_real_awn=False, use_real_attack=False, use_real_topk=False,
            torch_num_threads=2,
        )
        result = run_dry_run_experiment(cfg_e2e_t2)
        check("torch_num_threads=2 sets torch.get_num_threads() to 2", torch.get_num_threads() == 2)
        check(
            "summary dict records torch_num_threads=2",
            result.get("torch_num_threads") == 2,
            f"got {result.get('torch_num_threads')!r}",
        )
        check(
            "summary dict records torch_actual_num_threads=2",
            result.get("torch_actual_num_threads") == 2,
            f"got {result.get('torch_actual_num_threads')!r}",
        )

        # restore, so this test doesn't leak a changed thread count to
        # anything run afterward in the same process
        torch.set_num_threads(default_threads_at_start)

        # 11. summary.csv on disk also carries both columns
        import csv
        with open(tmp_path / "e2e_t2" / "summary.csv") as f:
            row = next(csv.DictReader(f))
        check("summary.csv has torch_num_threads column", "torch_num_threads" in row)
        check("summary.csv has torch_actual_num_threads column", "torch_actual_num_threads" in row)
        check(
            "summary.csv torch_actual_num_threads == '2'",
            row.get("torch_actual_num_threads") == "2",
            f"got {row.get('torch_actual_num_threads')!r}",
        )

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
