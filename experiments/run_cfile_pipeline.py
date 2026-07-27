"""
Formal .cfile / real-IQ-file pipeline entry point.

.cfile / IQ file -> src/io/iq_file_source.py (real loader) -> real
energy_detect -> region extraction/merge/filter -> real max-energy
alignment -> [N,2,128] -> real AWN clean inference -> optional real attack
-> optional real Top-K -> real AWN attacked/defended inference -> per-segment
CSV + sensing-region CSV + segment CSV + latency/memory instrumentation +
manifest + sensing plot.

Deliberately NOT scripts/sdr_sensing_to_awn_poc.py -- that script's AWN
inference is still the numpy-only placeholder (random logits); this script
uses src/adapters/awn_adapter.py:AWNModelAdapter (the real checkpoint) and
the exact same src/adapters/attack_adapter.py / topk_adapter.py real
backends every formal round this session has used, following the same
"call the low-level building blocks directly, not run_dry_run_experiment"
fair-reuse architecture established by experiments/run_phase0_pilot.py
(needed here too, since one file can produce multiple regions/segments
sharing one real AWN/attack/Top-K adapter set).

No ground truth exists for a real capture -- true_start/true_end/oracle
crop/detection_probability/captured_signal_ratio/boundary_error are never
computed or fabricated here (unlike RadioML mode, which has them via
embed_sample_in_noise's own seeded placement). Accuracy/attack-success
metrics are only computed if the caller supplies --true-label-mod;
otherwise they are left None/unavailable, never guessed.

Does not modify external/AWN, external/adversarial-rf, or
src/sensing/iq_source.py's existing (dead, from the real pipeline's
perspective) load_iq_from_file. Does not touch any existing results/
directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adapters.attack_adapter import AttackAdapter, _REAL_ATTACK_SOURCE  # noqa: E402
from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE  # noqa: E402
from src.adapters.topk_adapter import TopKAdapter, _REAL_SOURCE as _REAL_TOPK_SOURCE  # noqa: E402
from src.io.iq_file_source import load_iq_file, SUPPORTED_IQ_FORMATS, SUPPORTED_ENDIANNESS  # noqa: E402
from src.sensing.energy_detection import energy_detect, filter_by_min_length, mask_to_regions, merge_close_regions  # noqa: E402
from src.sensing.iq_source import validate_iq  # noqa: E402
from src.sensing.normalize import apply_awn_preprocess, to_awn_input  # noqa: E402
from src.sensing.radioml_source import RML2016_10A_CLASSES  # noqa: E402
from src.sensing.segmentation import select_aligned_segments  # noqa: E402
from src.utils.config import build_attack_params, RML2016_10A_MODULATIONS  # noqa: E402
from src.utils.plotting import plot_sensing_result  # noqa: E402

RAW_FIELDS = [
    "segment_id", "region_id", "input_path", "file_sha256", "iq_format", "endianness", "scale",
    "sample_rate", "loaded_sample_count", "crop_start", "crop_end", "input_hash",
    "true_label_mod", "true_label_index",
    "clean_prediction", "clean_confidence", "clean_correct",
    "attack_name", "attacked_prediction", "attacked_confidence", "attacked_correct", "attack_success",
    "perturbation_linf", "perturbation_l2",
    "clean_logits_max_abs_diff", "clean_prediction_reproducible",
    "topk", "defended_prediction", "defended_confidence", "defended_correct",
    "status", "error_type", "error_message", "fallback_used",
    "awn_backend", "attack_backend", "topk_backend", "model_mode_after",
    "file_load_ms", "sensing_ms", "region_postprocess_ms", "segmentation_ms",
    "awn_clean_ms", "attack_ms", "topk_ms", "awn_attacked_ms", "awn_defended_ms", "total_ms",
]


def softmax_np(logits_1d: np.ndarray) -> np.ndarray:
    z = logits_1d - np.max(logits_1d)
    e = np.exp(z)
    return e / np.sum(e)


def sha256_array(x: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def build_awn_input(iq_1d: np.ndarray, awn_preprocess: str) -> np.ndarray:
    segs = iq_1d[np.newaxis, :].astype(np.complex64)
    segs = apply_awn_preprocess(segs, policy=awn_preprocess)
    return to_awn_input(segs, seg_len=128)


def infer_one(awn_adapter: AWNModelAdapter, x: np.ndarray, seed: int):
    t0 = time.perf_counter()
    logits, meta = awn_adapter.infer(x, seed=seed)
    ms = (time.perf_counter() - t0) * 1000.0
    probs = softmax_np(logits[0])
    pred = int(np.argmax(logits[0]))
    conf = float(probs[pred])
    return pred, conf, logits[0], meta, ms


def git_state() -> dict:
    def _run(cmd):
        try:
            return subprocess.check_output(cmd, cwd=Path(__file__).resolve().parents[1], text=True).strip()
        except Exception as exc:  # noqa: BLE001
            return f"<error: {exc}>"
    return {
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": _run(["git", "status", "--porcelain"]),
    }


def env_state(checkpoint_path: str) -> dict:
    import hashlib
    import torch
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "device": "cpu",
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        "memory_measurement_method": "resource.getrusage(RUSAGE_SELF).ru_maxrss (Linux, KB, process peak RSS since start)",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-path", required=True)
    ap.add_argument("--iq-format", choices=SUPPORTED_IQ_FORMATS, default="complex64")
    ap.add_argument("--iq-endianness", choices=SUPPORTED_ENDIANNESS, default="native")
    ap.add_argument("--iq-scale", type=float, default=None)
    ap.add_argument("--iq-sample-rate", type=float, default=None)
    ap.add_argument("--iq-offset-samples", type=int, default=0)
    ap.add_argument("--iq-max-samples", type=int, default=None)
    ap.add_argument("--true-label-mod", type=str, default=None, choices=RML2016_10A_MODULATIONS)

    ap.add_argument("--threshold-factor", type=float, default=1.5)
    ap.add_argument("--sensing-window-size", type=int, default=128)
    ap.add_argument("--min-region-len", type=int, default=128)
    ap.add_argument("--merge-gap", type=int, default=0)
    ap.add_argument("--alignment-policy", choices=["naive", "max-energy"], default="max-energy")
    ap.add_argument("--awn-preprocess", choices=["legacy-unit-power", "radioml-native"], default="radioml-native")
    ap.add_argument("--segment-hop", type=int, default=1)

    ap.add_argument("--attack", type=str, default="none")
    ap.add_argument("--attack-eps", type=float, default=0.05)
    ap.add_argument("--attack-steps", type=int, default=None)
    ap.add_argument("--topk", type=int, default=None, help="Omit/None to disable Top-K entirely.")

    ap.add_argument("--checkpoint", type=str, default="external/adversarial-rf/2016.10a_AWN.pkl")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--experiment-name", type=str, default=None)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def run_cfile_pipeline(args: argparse.Namespace) -> dict:
    """Core cfile pipeline logic, callable directly (not just via CLI) so
    driver scripts (format-comparison / smoke-test matrices) can invoke it
    repeatedly in-process with different argparse.Namespace configs without
    paying subprocess + checkpoint-reload cost per case. Returns the
    manifest dict (which also nests raw_rows/sensing_region_rows/
    segments_csv_rows for callers that want the in-memory data, not just
    the written CSVs)."""
    if args.topk is not None and not (1 <= args.topk <= 128):
        raise ValueError(f"--topk must satisfy 1 <= topk <= 128, got {args.topk}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be > 0, got {args.batch_size}")

    out_dir = Path(args.output_dir)
    existing = out_dir / "raw_results.csv"
    if existing.exists() and not args.overwrite:
        raise FileExistsError(f"{existing} already exists -- refusing to overwrite without --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    t_total_start = time.perf_counter()

    print(f"[cfile-pipeline] loading AWN checkpoint {args.checkpoint} ...", flush=True)
    awn_adapter = AWNModelAdapter(checkpoint_path=args.checkpoint, device=args.device)
    if awn_adapter.backend_name != _REAL_MODEL_SOURCE or awn_adapter.status != "ok":
        raise RuntimeError(f"Real-AWN precheck FAILED: backend={awn_adapter.backend_name} status={awn_adapter.status}")
    print(f"[precheck] real AWN backend confirmed: {_REAL_MODEL_SOURCE}", flush=True)

    attack_adapter = AttackAdapter(awn_model=awn_adapter.model, device=args.device) if args.attack != "none" else None
    if attack_adapter is not None and (attack_adapter.wrapped_model is None or attack_adapter.backend_name != _REAL_ATTACK_SOURCE):
        raise RuntimeError(f"Real-attack precheck FAILED: backend={attack_adapter.backend_name}")
    topk_adapter = TopKAdapter() if args.topk is not None else None
    if topk_adapter is not None and not topk_adapter.backend_available:
        raise RuntimeError("Real Top-K backend unavailable")

    # ---- 1. file load ----
    t0 = time.perf_counter()
    iq, provenance = load_iq_file(
        args.input_path, iq_format=args.iq_format, endianness=args.iq_endianness,
        scale=args.iq_scale, sample_rate=args.iq_sample_rate,
        offset_samples=args.iq_offset_samples, max_samples=args.iq_max_samples, channel_count=1,
    )
    iq = validate_iq(iq)
    file_load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[cfile-pipeline] file_load_ms={file_load_ms:.2f} loaded_samples={len(iq)}", flush=True)

    # ---- 2. sensing (energy detect) ----
    t0 = time.perf_counter()
    mask = energy_detect(iq, window=args.sensing_window_size, threshold_factor=args.threshold_factor)
    raw_regions = mask_to_regions(mask)
    sensing_ms = (time.perf_counter() - t0) * 1000.0

    # ---- 3. region post-processing (merge/filter) ----
    t0 = time.perf_counter()
    merged_regions = merge_close_regions(raw_regions, merge_gap=args.merge_gap)
    sensing_region_rows = []
    kept_regions = []
    for i, (s, e) in enumerate(merged_regions):
        length = e - s
        power = np.mean(np.abs(iq[s:e]) ** 2) if length > 0 else 0.0
        peak = np.max(np.abs(iq[s:e]) ** 2) if length > 0 else 0.0
        selected = length >= args.min_region_len
        sensing_region_rows.append({
            "region_id": i, "start": s, "end": e, "length": length,
            "mean_power": float(power), "peak_power": float(peak),
            "selected": selected, "rejection_reason": None if selected else f"length {length} < min_region_len {args.min_region_len}",
        })
        if selected:
            kept_regions.append((s, e))
    try:
        kept_regions = filter_by_min_length(merged_regions, min_len=args.min_region_len)
    except RuntimeError:
        kept_regions = []
    region_postprocess_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[cfile-pipeline] sensing_ms={sensing_ms:.2f} region_postprocess_ms={region_postprocess_ms:.2f} "
          f"raw_regions={len(raw_regions)} merged_regions={len(merged_regions)} kept_regions={len(kept_regions)}", flush=True)

    # ---- 4. segmentation (alignment) ----
    t0 = time.perf_counter()
    segment_rows_meta = []
    segments = np.empty((0, 128), dtype=np.complex64)
    if kept_regions:
        try:
            segments, align_meta = select_aligned_segments(
                iq, kept_regions, seg_len=128, policy=args.alignment_policy, hop=args.segment_hop,
            )
            for i, am in enumerate(align_meta):
                segment_rows_meta.append(am)
        except RuntimeError as exc:
            print(f"[cfile-pipeline] segmentation failed: {exc}", flush=True)
    segmentation_ms = (time.perf_counter() - t0) * 1000.0
    n_segments = segments.shape[0]
    print(f"[cfile-pipeline] segmentation_ms={segmentation_ms:.2f} generated_segments={n_segments}", flush=True)

    true_label_index = RML2016_10A_CLASSES[args.true_label_mod] if args.true_label_mod else None

    raw_rows: List[dict] = []
    segments_csv_rows: List[dict] = []
    n_error = 0
    n_fallback = 0
    n_nan = 0

    attack_params = {"steps": args.attack_steps} if args.attack_steps is not None else {}

    for seg_id in range(n_segments):
        row = {k: None for k in RAW_FIELDS}
        row.update({
            "segment_id": seg_id, "region_id": segment_rows_meta[seg_id]["region_idx"] if seg_id < len(segment_rows_meta) else None,
            "input_path": provenance["input_path"], "file_sha256": provenance["file_sha256"],
            "iq_format": provenance["iq_format"], "endianness": provenance["endianness"],
            "scale": provenance["scale"], "sample_rate": provenance["sample_rate"],
            "loaded_sample_count": provenance["loaded_sample_count"],
            "crop_start": segment_rows_meta[seg_id]["selected_segment_start"] if seg_id < len(segment_rows_meta) else None,
            "crop_end": segment_rows_meta[seg_id]["selected_segment_end"] if seg_id < len(segment_rows_meta) else None,
            "true_label_mod": args.true_label_mod, "true_label_index": true_label_index,
            "attack_name": args.attack, "topk": args.topk,
            "file_load_ms": file_load_ms, "sensing_ms": sensing_ms,
            "region_postprocess_ms": region_postprocess_ms, "segmentation_ms": segmentation_ms,
        })
        t_seg_start = time.perf_counter()
        try:
            iq_seg = segments[seg_id]
            x_clean = build_awn_input(iq_seg, args.awn_preprocess)
            row["input_hash"] = sha256_array(x_clean)
            segments_csv_rows.append({
                "segment_id": seg_id, "source_region": row["region_id"],
                "crop_start": row["crop_start"], "crop_end": row["crop_end"],
                "shape": str(x_clean.shape), "input_hash": row["input_hash"],
            })

            pred_clean, conf_clean, logits_clean, meta_clean, awn_clean_ms = infer_one(awn_adapter, x_clean, args.seed)
            row["clean_prediction"], row["clean_confidence"] = pred_clean, conf_clean
            row["awn_clean_ms"] = awn_clean_ms
            row["awn_backend"] = meta_clean["awn_backend"]
            if true_label_index is not None:
                row["clean_correct"] = (pred_clean == true_label_index)

            x_current = x_clean
            attack_ms = 0.0
            attacked_pred = attacked_conf = None
            if attack_adapter is not None:
                t0 = time.perf_counter()
                x_adv, attack_meta = attack_adapter.apply(
                    x_clean, attack=args.attack, eps=args.attack_eps, seed=args.seed, attack_params=attack_params,
                )
                attack_ms = (time.perf_counter() - t0) * 1000.0
                row["attack_backend"] = attack_meta["attack_backend"]
                row["model_mode_after"] = "eval" if not awn_adapter.model.training else "train"
                perturb = x_adv.astype(np.float64) - x_clean.astype(np.float64)
                row["perturbation_linf"] = float(np.max(np.abs(perturb)))
                row["perturbation_l2"] = float(np.linalg.norm(perturb))
                if attack_meta["attack_status"] != "ok" or attack_meta["attack_backend"] != _REAL_ATTACK_SOURCE:
                    n_fallback += 1
                attacked_pred, attacked_conf, logits_attacked, meta_attacked, awn_attacked_ms = infer_one(awn_adapter, x_adv, args.seed)
                row["attacked_prediction"], row["attacked_confidence"] = attacked_pred, attacked_conf
                row["awn_attacked_ms"] = awn_attacked_ms
                row["attack_success"] = (attacked_pred != pred_clean)
                if true_label_index is not None:
                    row["attacked_correct"] = (attacked_pred == true_label_index)
                x_current = x_adv

                # Reproducibility check: re-run clean inference on the SAME x_clean
                # after the attack ran, to prove the attack left no side effects on
                # the shared AWN model (BatchNorm running stats, stray grad state,
                # etc.) beyond the training-mode flag already checked above -- the
                # same property experiments/run_attack_compatibility_smoke.py already
                # validates for RadioML inputs (clean_logits_max_abs_diff).
                pred_clean_repro, _conf_repro, logits_clean_repro, _meta_repro, _ms_repro = infer_one(
                    awn_adapter, x_clean, args.seed
                )
                row["clean_logits_max_abs_diff"] = float(np.max(np.abs(logits_clean_repro - logits_clean)))
                row["clean_prediction_reproducible"] = (pred_clean_repro == pred_clean)
            row["attack_ms"] = attack_ms

            topk_ms = 0.0
            if topk_adapter is not None:
                t0 = time.perf_counter()
                x_def, topk_meta = topk_adapter.apply(x_current, topk=args.topk)
                topk_ms = (time.perf_counter() - t0) * 1000.0
                row["topk_backend"] = topk_meta["topk_backend"]
                if topk_meta["topk_status"] != "ok" or topk_meta["topk_backend"] != _REAL_TOPK_SOURCE:
                    n_fallback += 1
                defended_pred, defended_conf, logits_defended, meta_defended, awn_defended_ms = infer_one(awn_adapter, x_def, args.seed)
                row["defended_prediction"], row["defended_confidence"] = defended_pred, defended_conf
                row["awn_defended_ms"] = awn_defended_ms
                if true_label_index is not None:
                    row["defended_correct"] = (defended_pred == true_label_index)
                if not np.isfinite(x_def).all():
                    n_nan += 1
            row["topk_ms"] = topk_ms

            if not np.isfinite(x_clean).all():
                n_nan += 1
            row["status"] = "ok"
            row["fallback_used"] = row.get("awn_backend") != _REAL_MODEL_SOURCE
        except Exception as exc:  # noqa: BLE001 - one segment's failure must not abort the others
            row["status"] = "error"
            row["error_type"] = type(exc).__name__
            row["error_message"] = str(exc)
            n_error += 1
        row["total_ms"] = (time.perf_counter() - t_seg_start) * 1000.0
        raw_rows.append(row)
        print(f"[cfile-pipeline] segment={seg_id+1}/{n_segments} status={row['status']} "
              f"clean_pred={row['clean_prediction']} total_ms={row['total_ms']:.2f} "
              f"error={n_error} fallback={n_fallback} nan={n_nan}", flush=True)

    total_ms = (time.perf_counter() - t_total_start) * 1000.0

    # ---- outputs ----
    with open(out_dir / "raw_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        for r in raw_rows:
            w.writerow(r)

    with open(out_dir / "sensing_regions.csv", "w", newline="") as f:
        fieldnames = list(sensing_region_rows[0].keys()) if sensing_region_rows else ["region_id"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sensing_region_rows:
            w.writerow(r)

    with open(out_dir / "segments.csv", "w", newline="") as f:
        fieldnames = list(segments_csv_rows[0].keys()) if segments_csv_rows else ["segment_id"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in segments_csv_rows:
            w.writerow(r)

    stage_cols = ["file_load_ms", "sensing_ms", "region_postprocess_ms", "segmentation_ms",
                  "awn_clean_ms", "attack_ms", "topk_ms", "awn_attacked_ms", "awn_defended_ms", "total_ms"]
    latency_rows = []
    for col in stage_cols:
        vals = [r[col] for r in raw_rows if isinstance(r.get(col), (int, float))]
        latency_rows.append({
            "stage": col, "n": len(vals),
            "mean_ms": float(np.mean(vals)) if vals else None,
            "median_ms": float(np.median(vals)) if vals else None,
            "p95_ms": float(np.percentile(vals, 95)) if vals else None,
            "max_ms": float(np.max(vals)) if vals else None,
        })
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    throughput = n_segments / (total_ms / 1000.0) if total_ms > 0 else 0.0
    with open(out_dir / "latency_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "n", "mean_ms", "median_ms", "p95_ms", "max_ms"])
        w.writeheader()
        for r in latency_rows:
            w.writerow(r)
        w.writerow({"stage": "OVERALL_total_ms", "n": 1, "mean_ms": total_ms, "median_ms": total_ms,
                    "p95_ms": total_ms, "max_ms": total_ms})

    manifest = {
        "cli_args": vars(args),
        "provenance": provenance,
        "git": git_state(),
        "env": env_state(args.checkpoint),
        "peak_rss_kb": peak_rss_kb,
        "peak_rss_mb": peak_rss_kb / 1024.0,
        "input_file_size_bytes": provenance["file_size_bytes"],
        "loaded_sample_count": provenance["loaded_sample_count"],
        "detected_region_count_raw": len(raw_regions),
        "detected_region_count_merged": len(merged_regions),
        "detected_region_count_kept": len(kept_regions),
        "generated_segment_count": n_segments,
        "average_ms_per_segment": (total_ms / n_segments) if n_segments else None,
        "throughput_segments_per_sec": throughput,
        "n_error": n_error, "n_fallback": n_fallback, "n_nan_inf": n_nan,
        "total_ms": total_ms,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    plot_created = False
    try:
        plot_created = plot_sensing_result(iq, kept_regions, out_dir / "sensing_plot.png")
    except Exception as exc:  # noqa: BLE001
        print(f"[cfile-pipeline] sensing plot failed (non-fatal): {exc}", flush=True)

    print(f"\n[cfile-pipeline] DONE: {n_segments} segments, {n_error} errors, {n_fallback} fallback, "
          f"{n_nan} nan/inf, total_ms={total_ms:.1f}, throughput={throughput:.2f} segments/sec, "
          f"peak_rss_mb={peak_rss_kb/1024.0:.1f}")
    print(f"[cfile-pipeline] output_dir={out_dir}")
    print(f"[cfile-pipeline] sensing_plot: {'saved' if plot_created else 'skipped (matplotlib unavailable or no regions)'}")

    manifest["raw_rows"] = raw_rows
    manifest["sensing_region_rows"] = sensing_region_rows
    manifest["segments_csv_rows"] = segments_csv_rows
    manifest["output_dir"] = str(out_dir)
    return manifest


def main() -> None:
    args = build_arg_parser().parse_args()
    run_cfile_pipeline(args)


if __name__ == "__main__":
    main()
