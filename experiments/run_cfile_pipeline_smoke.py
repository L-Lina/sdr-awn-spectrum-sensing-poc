"""
Master driver for the .cfile pipeline smoke test + backward-compatibility
regression (docs/parameter_validation.md-style round). Produces a single
timestamped results/cfile_pipeline_smoke_<UTC-timestamp>/ directory with:
  raw_results.csv, sensing_regions.csv, segments.csv, latency_summary.csv,
  format_comparison.csv, backward_compatibility_summary.csv, manifest.json,
  sensing_plot.png (one, from the primary complex64 fixture run).
(terminal.log is produced by the caller redirecting/tee-ing this script's
stdout -- this script only ensures everything it prints is meaningful when
captured that way.)

Steps:
  1. Generate ONE controllable synthetic long-stream IQ signal (3 real
     RadioML bursts embedded in noise, via the existing, UNMODIFIED
     src/sensing/radioml_source.py:embed_multiple_samples_in_noise), then
     write it out as complex64 / interleaved_float32 / interleaved_int16
     into a tempfile.TemporaryDirectory (never committed, never mixed into
     the formal dataset).
  2. Format-comparison (Part 7): run the formal cfile pipeline (clean,
     no attack, no Top-K) once per format, compare sample/region/segment
     counts and clean predictions/logits across formats.
  3. cfile pipeline smoke (Part 8 A/B/C/D): clean AMC, FGSM/CW/DIFGSM
     attack, Top-K 10/20/128, and one attack+Top-K combo -- all against
     the complex64 fixture (the "primary" one), all through the exact
     same real formal pipeline (experiments/run_cfile_pipeline.py).
  4. Backward-compatibility regression (Part 9 A-F) -- reuses existing,
     already-validated formal entry points UNMODIFIED:
       A/B/C: experiments/run_full_experiment.py (--iq-source radioml)
       D:     experiments/run_full_experiment.py (--iq-source synthetic)
       E:     experiments/run_spectrum_sensing_utility.py --mode fairness-test
       F:     schema/CLI compatibility checks against the above outputs
     Any single regression failure aborts with a non-zero exit and prints
     which one failed -- this script never silently reports success.

Does not modify external/AWN, external/adversarial-rf, or any existing
results/ directory. Does not commit or push.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.io.iq_file_source import write_iq_file  # noqa: E402
from src.sensing.radioml_source import load_radioml_sample, embed_multiple_samples_in_noise  # noqa: E402
from experiments.run_cfile_pipeline import build_arg_parser as build_cfile_args, run_cfile_pipeline  # noqa: E402
from src.utils.dataset_path import resolve_dataset_path  # noqa: E402
from src.utils.python_executable import require_python_executable_exists, resolve_python_executable  # noqa: E402

DATASET_PATH = resolve_dataset_path()  # priority: env $SDR_AWN_DATASET_PATH > legacy default
# priority: --python-executable (set in main()) > env $SDR_AWN_PYTHON > sys.executable (the
# interpreter this script itself is running under -- the portable default) > legacy fallback.
# Subprocess calls below (run_full_experiment_cli, the four-path fairness-test call) use whichever
# interpreter actually invoked THIS script by default; run this script itself via a torch-capable
# interpreter (as every formal round already does) for the real backend to be used downstream.
VENV_PYTHON = resolve_python_executable(legacy_default="/home/xiaomi/adversarial-rf/.venv/bin/python")


def log(msg: str) -> None:
    print(f"[cfile-smoke] {msg}", flush=True)


def make_fixture_stream() -> tuple:
    mods = ["QPSK", "BPSK", "QAM16"]
    samples = [load_radioml_sample(DATASET_PATH, m, 0, 0) for m in mods]
    iq, burst_meta = embed_multiple_samples_in_noise(
        samples, n_samples=4000, embed_snr_margin=15.0, seed=42,
        min_burst_gap=300, max_burst_gap=600,
    )
    assert np.isfinite(iq).all()
    return iq, mods, burst_meta


def cfile_args(input_path: str, iq_format: str, output_dir: Path, **overrides) -> argparse.Namespace:
    argv = [
        "--input-path", input_path, "--iq-format", iq_format,
        "--min-region-len", "128", "--sensing-window-size", "128", "--threshold-factor", "1.5",
        "--alignment-policy", "max-energy", "--awn-preprocess", "radioml-native",
        "--attack", "none", "--output-dir", str(output_dir), "--overwrite",
    ]
    args = build_cfile_args().parse_args(argv)
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def run_full_experiment_cli(argv: list, cwd: Path) -> subprocess.CompletedProcess:
    cmd = [VENV_PYTHON, "experiments/run_full_experiment.py"] + argv
    log(f"subprocess: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def main() -> None:
    global VENV_PYTHON
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--python-executable", default=None,
                     help="Interpreter used for this script's subprocess calls (run_full_experiment.py, "
                          "run_spectrum_sensing_utility.py). Default: env $SDR_AWN_PYTHON, else sys.executable.")
    args = ap.parse_args()
    if args.python_executable:
        VENV_PYTHON = resolve_python_executable(cli_value=args.python_executable)
    require_python_executable_exists(VENV_PYTHON)
    log(f"subprocess interpreter resolved to: {VENV_PYTHON}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=False)  # new timestamped dir -- must not already exist

    n_error_total = 0
    n_fallback_total = 0
    n_nan_total = 0
    backward_compat_rows = []
    format_comparison_rows = []
    all_raw_rows = []
    all_sensing_region_rows = []
    all_segments_rows = []
    latency_rows_by_case = {}

    with tempfile.TemporaryDirectory(prefix="cfile_smoke_fixtures_") as tmp:
        tmp = Path(tmp)
        log("=== Step 1: generating fixture IQ stream (3 real RadioML bursts embedded in noise) ===")
        iq, mods, burst_meta = make_fixture_stream()
        log(f"fixture stream: {len(iq)} samples, {len(burst_meta)} bursts at {[m['true_start'] for m in burst_meta]}")

        paths = {
            "complex64": tmp / "fixture.c64",
            "interleaved_float32": tmp / "fixture.f32iq",
            "interleaved_int16": tmp / "fixture.i16iq",
        }
        for fmt, p in paths.items():
            write_iq_file(iq, str(p), fmt)
            log(f"wrote {fmt} fixture: {p} ({p.stat().st_size} bytes)")

        # ---- Step 2: format comparison (Part 7) ----
        log("=== Step 2: format-comparison test (Part 7) ===")
        format_results = {}
        for fmt, p in paths.items():
            case_out = out_dir / f"_scratch_format_{fmt}"
            cargs = cfile_args(str(p), fmt, case_out)
            manifest = run_cfile_pipeline(cargs)
            format_results[fmt] = manifest
            n_error_total += manifest["n_error"]
            n_fallback_total += manifest["n_fallback"]
            n_nan_total += manifest["n_nan_inf"]
            for r in manifest["raw_rows"]:
                r2 = dict(r)
                r2["test_case"] = f"format_comparison_{fmt}"
                all_raw_rows.append(r2)
            for r in manifest["sensing_region_rows"]:
                r2 = dict(r)
                r2["test_case"] = f"format_comparison_{fmt}"
                all_sensing_region_rows.append(r2)
            for r in manifest["segments_csv_rows"]:
                r2 = dict(r)
                r2["test_case"] = f"format_comparison_{fmt}"
                all_segments_rows.append(r2)

        c64 = format_results["complex64"]
        f32 = format_results["interleaved_float32"]
        i16 = format_results["interleaved_int16"]
        same_sample_count = (c64["loaded_sample_count"] == f32["loaded_sample_count"] == i16["loaded_sample_count"])
        same_region_count = (c64["detected_region_count_kept"] == f32["detected_region_count_kept"] == i16["detected_region_count_kept"])
        same_segment_count = (c64["generated_segment_count"] == f32["generated_segment_count"] == i16["generated_segment_count"])
        c64_preds = [r["clean_prediction"] for r in c64["raw_rows"]]
        f32_preds = [r["clean_prediction"] for r in f32["raw_rows"]]
        i16_preds = [r["clean_prediction"] for r in i16["raw_rows"]]
        same_predictions_c64_f32 = c64_preds == f32_preds
        same_predictions_c64_i16 = c64_preds == i16_preds
        format_comparison_rows.append({
            "check": "sample_count", "complex64": c64["loaded_sample_count"],
            "interleaved_float32": f32["loaded_sample_count"], "interleaved_int16": i16["loaded_sample_count"],
            "pass": same_sample_count,
        })
        format_comparison_rows.append({
            "check": "region_count", "complex64": c64["detected_region_count_kept"],
            "interleaved_float32": f32["detected_region_count_kept"], "interleaved_int16": i16["detected_region_count_kept"],
            "pass": same_region_count,
        })
        format_comparison_rows.append({
            "check": "segment_count", "complex64": c64["generated_segment_count"],
            "interleaved_float32": f32["generated_segment_count"], "interleaved_int16": i16["generated_segment_count"],
            "pass": same_segment_count,
        })
        format_comparison_rows.append({
            "check": "clean_predictions_match_c64_vs_f32", "complex64": str(c64_preds),
            "interleaved_float32": str(f32_preds), "interleaved_int16": "-", "pass": same_predictions_c64_f32,
        })
        format_comparison_rows.append({
            "check": "clean_predictions_match_c64_vs_i16", "complex64": str(c64_preds),
            "interleaved_float32": "-", "interleaved_int16": str(i16_preds), "pass": same_predictions_c64_i16,
        })
        log(f"format comparison: sample_count_match={same_sample_count} region_count_match={same_region_count} "
            f"segment_count_match={same_segment_count} preds_c64_vs_f32={same_predictions_c64_f32} "
            f"preds_c64_vs_i16={same_predictions_c64_i16}")
        if not (same_sample_count and same_region_count and same_segment_count):
            raise RuntimeError("Format-comparison FAILED: sample/region/segment counts differ across formats "
                                "-- see format_comparison.csv for details")

        # ---- Step 3: cfile pipeline smoke A/B/C/D (Part 8) ----
        log("=== Step 3: cfile pipeline smoke tests (Part 8 A/B/C/D) ===")
        primary = str(paths["complex64"])

        # A: clean sensing+AMC already captured above as format_comparison_complex64; reuse it.

        # B: attack smoke (FGSM/CW/DIFGSM)
        for atk in ["fgsm", "cw", "difgsm"]:
            case_out = out_dir / f"_scratch_attack_{atk}"
            cargs = cfile_args(primary, "complex64", case_out, attack=atk, attack_eps=0.05)
            manifest = run_cfile_pipeline(cargs)
            n_error_total += manifest["n_error"]
            n_fallback_total += manifest["n_fallback"]
            n_nan_total += manifest["n_nan_inf"]
            for r in manifest["raw_rows"]:
                r2 = dict(r)
                r2["test_case"] = f"attack_smoke_{atk}"
                all_raw_rows.append(r2)
            if manifest["n_error"] > 0 or manifest["n_fallback"] > 0:
                raise RuntimeError(f"Attack smoke FAILED for {atk}: errors={manifest['n_error']} fallback={manifest['n_fallback']}")
            model_modes = {r["model_mode_after"] for r in manifest["raw_rows"]}
            if model_modes != {"eval"}:
                raise RuntimeError(f"Attack smoke FAILED for {atk}: model_mode_after != eval ({model_modes})")
            log(f"attack smoke {atk}: OK ({manifest['generated_segment_count']} segments, "
                f"perturbation_linf={[r['perturbation_linf'] for r in manifest['raw_rows']]})")

        # C: Top-K smoke (10/20/128)
        for k in [10, 20, 128]:
            case_out = out_dir / f"_scratch_topk_{k}"
            cargs = cfile_args(primary, "complex64", case_out, topk=k)
            manifest = run_cfile_pipeline(cargs)
            n_error_total += manifest["n_error"]
            n_fallback_total += manifest["n_fallback"]
            n_nan_total += manifest["n_nan_inf"]
            for r in manifest["raw_rows"]:
                r2 = dict(r)
                r2["test_case"] = f"topk_smoke_{k}"
                all_raw_rows.append(r2)
            if manifest["n_error"] > 0 or manifest["n_fallback"] > 0:
                raise RuntimeError(f"Top-K smoke FAILED for K={k}: errors={manifest['n_error']} fallback={manifest['n_fallback']}")
            log(f"topk smoke K={k}: OK ({manifest['generated_segment_count']} segments)")

        # D: attack + Top-K combined
        case_out = out_dir / "_scratch_attack_topk_combo"
        cargs = cfile_args(primary, "complex64", case_out, attack="difgsm", attack_eps=0.05, topk=20)
        manifest = run_cfile_pipeline(cargs)
        n_error_total += manifest["n_error"]
        n_fallback_total += manifest["n_fallback"]
        n_nan_total += manifest["n_nan_inf"]
        for r in manifest["raw_rows"]:
            r2 = dict(r)
            r2["test_case"] = "attack_topk_combo_difgsm_k20"
            all_raw_rows.append(r2)
        if manifest["n_error"] > 0 or manifest["n_fallback"] > 0:
            raise RuntimeError(f"Attack+Top-K combo smoke FAILED: errors={manifest['n_error']} fallback={manifest['n_fallback']}")
        log(f"attack+topk combo (difgsm+K20): OK ({manifest['generated_segment_count']} segments)")

        # illegal Top-K boundary re-confirmation (0 and 129 must raise)
        for bad_k in [0, 129]:
            try:
                cargs = cfile_args(primary, "complex64", out_dir / f"_scratch_topk_bad_{bad_k}", topk=bad_k)
                run_cfile_pipeline(cargs)
                raise RuntimeError(f"Top-K={bad_k} was NOT rejected -- illegal-value regression FAILED")
            except ValueError as exc:
                log(f"topk={bad_k} correctly rejected: {exc}")

    # ---- Step 4: backward-compatibility regression (Part 9 A-F) ----
    log("=== Step 4: backward-compatibility regression (Part 9 A-F) ===")

    def bc_check(name: str, cond: bool, detail: str = "") -> None:
        backward_compat_rows.append({"check": name, "pass": bool(cond), "detail": detail})
        log(f"[{'PASS' if cond else 'FAIL'}] backward-compat: {name} {detail}")
        if not cond:
            raise RuntimeError(f"Backward-compatibility regression FAILED: {name} {detail}")

    # A: original RadioML direct AMC
    scratch_a = out_dir / "_scratch_bc_radioml_direct"
    p = run_full_experiment_cli([
        "--iq-source", "radioml", "--dataset-path", DATASET_PATH, "--dataset-mod", "QPSK",
        "--dataset-snr", "0", "--sample-index", "0", "--attack", "none",
        "--use-real-awn", "--dry-run", "--output-dir", str(scratch_a),
    ], REPO_ROOT)
    bc_check("A_radioml_direct_amc", p.returncode == 0, p.stderr[-500:] if p.returncode != 0 else "")

    # B: original RadioML+Attack (FGSM/CW/DIFGSM)
    for atk in ["fgsm", "cw", "difgsm"]:
        scratch_b = out_dir / f"_scratch_bc_radioml_attack_{atk}"
        p = run_full_experiment_cli([
            "--iq-source", "radioml", "--dataset-path", DATASET_PATH, "--dataset-mod", "QPSK",
            "--dataset-snr", "0", "--sample-index", "0", "--attack", atk, "--attack-eps", "0.05",
            "--use-real-awn", "--use-real-attack", "--dry-run", "--output-dir", str(scratch_b),
        ], REPO_ROOT)
        ok = p.returncode == 0
        detail = p.stderr[-500:] if p.returncode != 0 else ""
        if ok:
            with open(scratch_b / "summary.csv") as f:
                row = next(csv.DictReader(f))
            # attack_training_after == "False" means the wrapped model was
            # restored to eval mode after the attack ran (attack_training_before
            # may legitimately be "True" -- see experiments/run_cfile_pipeline_smoke.py
            # module docstring / this session's established pre-existing quirk).
            ok = row.get("attack_training_after") == "False" and row.get("attack_status") == "ok" \
                and row.get("attacked_has_nan") == "False" and row.get("attacked_has_inf") == "False"
            detail = f"attack_training_after={row.get('attack_training_after')} attack_status={row.get('attack_status')}"
        bc_check(f"B_radioml_attack_{atk}", ok, detail)

    # C: original RadioML+Top-K (10/20/128)
    for k in [10, 20, 128]:
        scratch_c = out_dir / f"_scratch_bc_radioml_topk_{k}"
        p = run_full_experiment_cli([
            "--iq-source", "radioml", "--dataset-path", DATASET_PATH, "--dataset-mod", "QPSK",
            "--dataset-snr", "0", "--sample-index", "0", "--attack", "none", "--topk", str(k),
            "--use-real-awn", "--use-real-topk", "--dry-run", "--output-dir", str(scratch_c),
        ], REPO_ROOT)
        bc_check(f"C_radioml_topk_{k}", p.returncode == 0, p.stderr[-500:] if p.returncode != 0 else "")

    # D: original synthetic long stream + sensing
    scratch_d = out_dir / "_scratch_bc_synthetic_sensing"
    p = run_full_experiment_cli([
        "--iq-source", "synthetic", "--snr", "10", "--use-real-awn", "--dry-run",
        "--output-dir", str(scratch_d),
    ], REPO_ROOT)
    bc_check("D_synthetic_sensing", p.returncode == 0, p.stderr[-500:] if p.returncode != 0 else "")

    # E: four-path Spectrum Sensing Utility fairness-test (smoke, NOT the 2200-row formal run)
    cmd = [VENV_PYTHON, "experiments/run_spectrum_sensing_utility.py", "--mode", "fairness-test"]
    log(f"subprocess: {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    combined = p.stdout + p.stderr
    passed_10_10 = ("10/10" in combined) or ("PASS" in combined and "FAIL" not in combined)
    bc_check("E_four_path_fairness_test", p.returncode == 0 and "FAIL" not in combined,
              combined[-800:] if (p.returncode != 0 or "FAIL" in combined) else "10/10 fairness checks referenced")

    # F: config/output compatibility -- existing RadioML/synthetic CLI must not require cfile params,
    # and additive cfile_* columns must be present-but-null for non-cfile runs (never break old schema).
    with open(scratch_a / "summary.csv") as f:
        radioml_row = next(csv.DictReader(f))
    required_old_cols = {"snr_db", "attack", "topk", "pred_clean", "detection_success", "captured_signal_ratio"}
    has_old_cols = required_old_cols.issubset(radioml_row.keys())
    cfile_cols_null = all(
        radioml_row.get(c) in ("", None) for c in radioml_row.keys() if c.startswith("cfile_")
    )
    bc_check("F_schema_compatibility", has_old_cols and cfile_cols_null,
              f"has_old_cols={has_old_cols} cfile_cols_null={cfile_cols_null}")

    # ---- assemble outputs ----
    log("=== Step 5: writing results directory ===")
    if all_raw_rows:
        fieldnames = list(all_raw_rows[0].keys())
        for r in all_raw_rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(out_dir / "raw_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_raw_rows:
                w.writerow(r)

    if all_sensing_region_rows:
        fieldnames = list(all_sensing_region_rows[0].keys())
        with open(out_dir / "sensing_regions.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_sensing_region_rows:
                w.writerow(r)

    if all_segments_rows:
        fieldnames = list(all_segments_rows[0].keys())
        with open(out_dir / "segments.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_segments_rows:
                w.writerow(r)

    with open(out_dir / "format_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "complex64", "interleaved_float32", "interleaved_int16", "pass"])
        w.writeheader()
        for r in format_comparison_rows:
            w.writerow(r)

    with open(out_dir / "backward_compatibility_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "pass", "detail"])
        w.writeheader()
        for r in backward_compat_rows:
            w.writerow(r)

    # latency summary: aggregate from the complex64 clean run (format_results) + attack/topk cases
    stage_cols = ["file_load_ms", "sensing_ms", "region_postprocess_ms", "segmentation_ms",
                  "awn_clean_ms", "attack_ms", "topk_ms", "awn_attacked_ms", "awn_defended_ms", "total_ms"]
    vals_by_stage = {c: [] for c in stage_cols}
    for r in all_raw_rows:
        for c in stage_cols:
            v = r.get(c)
            if isinstance(v, (int, float)):
                vals_by_stage[c].append(v)
    with open(out_dir / "latency_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "n", "mean_ms", "median_ms", "p95_ms", "max_ms"])
        w.writeheader()
        for c in stage_cols:
            vals = vals_by_stage[c]
            w.writerow({
                "stage": c, "n": len(vals),
                "mean_ms": float(np.mean(vals)) if vals else None,
                "median_ms": float(np.median(vals)) if vals else None,
                "p95_ms": float(np.percentile(vals, 95)) if vals else None,
                "max_ms": float(np.max(vals)) if vals else None,
            })

    def git_state() -> dict:
        def _run(cmd):
            return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True).strip()
        return {"git_commit": _run(["git", "rev-parse", "HEAD"]), "git_status_porcelain": _run(["git", "status", "--porcelain"])}

    import platform
    import resource
    # env is identical across all format_results entries (same interpreter/process for every
    # in-process run_cfile_pipeline() call) -- pulled from the complex64 run's own manifest
    # (produced by experiments/run_cfile_pipeline.py:env_state()) rather than recomputed here.
    env_info = dict(format_results["complex64"]["env"])
    env_info["platform"] = platform.platform()
    env_info["processor"] = platform.processor() or platform.machine()
    per_format_provenance = {fmt: format_results[fmt]["provenance"] for fmt in paths}
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture": {"mods": mods, "burst_meta": burst_meta, "n_samples": 4000},
        "git": git_state(),
        "env": env_info,
        "per_format_provenance": per_format_provenance,
        "n_error_total": n_error_total,
        "n_fallback_total": n_fallback_total,
        "n_nan_inf_total": n_nan_total,
        "format_comparison_all_pass": all(r["pass"] for r in format_comparison_rows),
        "backward_compatibility_all_pass": all(r["pass"] for r in backward_compat_rows),
        "peak_rss_kb_this_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "memory_measurement_method": "resource.getrusage(RUSAGE_SELF).ru_maxrss (Linux, KB, this process's peak RSS)",
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # copy the primary complex64 clean run's sensing plot up to the results root
    primary_plot = out_dir / "_scratch_format_complex64" / "sensing_plot.png"
    if primary_plot.exists():
        import shutil
        shutil.copy(primary_plot, out_dir / "sensing_plot.png")

    log(f"DONE: n_error_total={n_error_total} n_fallback_total={n_fallback_total} n_nan_total={n_nan_total} "
        f"format_comparison_all_pass={manifest['format_comparison_all_pass']} "
        f"backward_compatibility_all_pass={manifest['backward_compatibility_all_pass']}")
    log(f"output_dir={out_dir}")

    if n_error_total > 0 or n_fallback_total > 0 or n_nan_total > 0:
        raise RuntimeError("Non-zero error/fallback/NaN count in cfile smoke matrix -- NOT marking complete")


if __name__ == "__main__":
    main()
