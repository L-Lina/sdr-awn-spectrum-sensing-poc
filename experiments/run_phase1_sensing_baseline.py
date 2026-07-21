"""
Phase 1 -- Spectrum Sensing baseline (+ Phase 2 direct/sensed AMC comparison,
computed inline since it shares every input) -- docs/formal_experiment_plan.md
section 4 (Phase 1/2) / docs/formal_experiment_matrix.csv (phase=1, phase=2).
Purpose: evaluate the spectrum-sensing front end's end-to-end effect on AMC,
with NO adversarial attack involved at all.

Fixed params (docs/formal_experiment_matrix.csv, phase=1 row -- copied
verbatim, not guessed): iq_source=radioml, attack=none, use_real_awn=True,
use_real_attack=False, use_real_topk=False, checkpoint=2016.10a (pinned),
device=cpu, alignment_policy=max-energy, awn_preprocess=radioml-native,
threshold_factor=1.5, sensing_window_size=128, min_region_len=0,
merge_gap=0, num_bursts=1, seed=42.

Design: reuses src/utils/pipeline.py:run_dry_run_experiment() directly, one
call per (mod,snr,idx) combo -- unlike Phase 0, Phase 1 has no cross-combo
fairness/reuse constraint, so there is no reason to bypass the existing,
already-validated pipeline function the way Phase 0 had to. The ONE thing
run_dry_run_experiment() does not compute is the "direct" (oracle, raw
RadioML sample, no sensing/embedding) AMC prediction -- added here via the
same building blocks Phase 0 used (radioml_sample_to_iq,
apply_awn_preprocess, to_awn_input, AWNModelAdapter.infer), through a
SEPARATE, once-constructed AWNModelAdapter (cheap to reuse across all
combos, since it never depends on sensing state).

run_batch_combos() (src/utils/batch_aggregation.py) was considered and NOT
used: it writes batch_summary.csv exactly once at the very end of the
whole batch, with no per-row flush and no --resume support (confirmed by
reading its source directly) -- too fragile for a ~2200-combo run. This
script instead wraps run_dry_run_experiment() in the same incremental-
write + --resume CsvWriter pattern experiments/run_phase0_pilot.py already
established and validated.

No sensing/AWN algorithm code was written or modified. external/AWN and
external/adversarial-rf are not touched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE  # noqa: E402
from src.sensing.normalize import apply_awn_preprocess, to_awn_input  # noqa: E402
from src.sensing.radioml_source import (  # noqa: E402
    RML2016_10A_CLASSES,
    load_radioml_sample,
    radioml_sample_to_iq,
)
from src.utils.config import ExperimentConfig  # noqa: E402
from src.utils.pipeline import run_dry_run_experiment  # noqa: E402

DATASET_PATH = "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl"
CHECKPOINT = "external/adversarial-rf/2016.10a_AWN.pkl"
DEVICE = "cpu"

ALL_MODS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
ALL_SNRS = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
DEFAULT_N_PER_CELL = 10

FIXED = dict(
    iq_source="radioml",
    attack="none",
    use_real_awn=True,
    use_real_attack=False,
    use_real_topk=False,
    alignment_policy="max-energy",
    awn_preprocess="radioml-native",
    threshold_factor=1.5,
    sensing_window_size=128,
    min_region_len=0,
    merge_gap=0,
    window_size=128,
    num_bursts=1,
    embed_snr_margin=20.0,
    seed=42,
    topk=50,  # unused/unreported this phase (use_real_topk=False keeps the dummy no-attack path cheap); value is inert
    attack_eps=0.03,  # unused (attack=none is a no-op regardless of eps)
    attack_temperature=1.0,
    cw_c=1.0, cw_steps=20, cw_lr=0.01,  # unused (attack=none never reaches CW)
)

SUMMARY_FIELDS = [
    "combo_id", "dataset", "modulation", "snr", "sample_index", "seed",
    "label", "pred_direct", "pred_clean", "direct_correct", "clean_correct",
    "direct_sensed_agreement",
    "detection_success", "detection_probability", "false_alarm_region_rate",
    "captured_signal_ratio", "extra_captured_noise_ratio",
    "start_boundary_error", "end_boundary_error",
    "missed_sample_count", "false_occupied_sample_count", "segment_count",
    "awn_backend", "clean_nan", "direct_nan",
    "runtime_seconds", "run_status", "failure_stage", "failure_reason", "output_dir",
    "original_sample_sha256", "long_iq_sha256",
]


def build_combo_grid(mods, snrs, sample_indices) -> List[dict]:
    combos = []
    for mod in mods:
        for snr in snrs:
            for idx in sample_indices:
                combos.append({
                    "combo_id": f"{mod}_snr{snr}_idx{idx}",
                    "modulation": mod, "snr": snr, "sample_index": idx,
                })
    return combos


def check_combo_ids_unique(combos: List[dict]) -> None:
    ids = [c["combo_id"] for c in combos]
    if len(ids) != len(set(ids)):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise AssertionError(f"Duplicate combo_id(s) found: {dupes}")


def not_finite(x: np.ndarray) -> bool:
    return bool(np.isnan(x).any() or np.isinf(x).any())


def precheck_real_awn(awn_adapter: AWNModelAdapter) -> None:
    if awn_adapter.backend_name != _REAL_MODEL_SOURCE or awn_adapter.status != "ok":
        raise RuntimeError(
            f"Real-AWN precheck FAILED -- refusing to run any combo: "
            f"backend={awn_adapter.backend_name!r} status={awn_adapter.status!r} notes={awn_adapter.notes}"
        )
    print(f"[precheck] real AWN backend confirmed: {_REAL_MODEL_SOURCE}")


def load_done_combo_ids(summary_path: Path) -> set:
    if not summary_path.exists():
        return set()
    with open(summary_path) as f:
        rows = list(csv.DictReader(f))
    return {r["combo_id"] for r in rows}


class CsvWriter:
    def __init__(self, path: Path, fresh: bool):
        self.path = path
        mode = "w" if fresh else "a"
        self.f = open(path, mode, newline="")
        self.writer = csv.DictWriter(self.f, fieldnames=SUMMARY_FIELDS)
        if fresh:
            self.writer.writeheader()
            self.f.flush()

    def write_row(self, row: dict) -> None:
        self.writer.writerow({k: row.get(k) for k in SUMMARY_FIELDS})
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def compute_direct_amc(awn_adapter: AWNModelAdapter, original_sample: np.ndarray, seed: int) -> dict:
    """Oracle path: raw [2,128] RadioML sample -> awn_preprocess -> AWN,
    bypassing sensing/embedding/alignment entirely. Same method as
    experiments/run_phase0_pilot.py:compute_direct_amc."""
    iq_direct = radioml_sample_to_iq(original_sample)
    segments_direct = iq_direct[np.newaxis, :]
    segments_direct = apply_awn_preprocess(segments_direct, policy=FIXED["awn_preprocess"])
    x_direct = to_awn_input(segments_direct, seg_len=128)
    logits_direct, awn_meta_direct = awn_adapter.infer(x_direct, seed=seed)
    pred_direct = int(np.argmax(logits_direct, axis=1)[0])
    return {"pred_direct": pred_direct, "x_direct": x_direct, "awn_meta_direct": awn_meta_direct}


def run_combo(combo: dict, direct_awn_adapter: AWNModelAdapter, output_dir: Path) -> dict:
    mod, snr, idx = combo["modulation"], combo["snr"], combo["sample_index"]
    combo_id = combo["combo_id"]
    seed = FIXED["seed"]
    label = RML2016_10A_CLASSES[mod]
    t0 = time.time()

    original_sample = load_radioml_sample(DATASET_PATH, mod, snr, idx)
    original_sample_sha256 = hashlib.sha256(original_sample.tobytes()).hexdigest()

    direct = compute_direct_amc(direct_awn_adapter, original_sample, seed)
    pred_direct = direct["pred_direct"]
    direct_nan = not_finite(direct["x_direct"])
    direct_awn_ok = (
        direct["awn_meta_direct"]["awn_backend"] == _REAL_MODEL_SOURCE
        and direct["awn_meta_direct"]["awn_status"] == "ok"
    )

    combo_output_dir = output_dir / combo_id
    cfg = ExperimentConfig(
        snr=10.0, mod=mod, attack=FIXED["attack"], topk=FIXED["topk"],
        threshold_factor=FIXED["threshold_factor"], window_size=FIXED["window_size"],
        sensing_window_size=FIXED["sensing_window_size"], min_region_len=FIXED["min_region_len"],
        merge_gap=FIXED["merge_gap"], burst_len=600,
        output_dir=str(combo_output_dir), dry_run=True,
        use_real_topk=FIXED["use_real_topk"], use_real_awn=FIXED["use_real_awn"],
        checkpoint=CHECKPOINT, device=DEVICE,
        attack_eps=FIXED["attack_eps"], use_real_attack=FIXED["use_real_attack"],
        attack_temperature=FIXED["attack_temperature"], attack_diagnostics=False,
        seed=seed, cw_c=FIXED["cw_c"], cw_steps=FIXED["cw_steps"], cw_lr=FIXED["cw_lr"],
        iq_source=FIXED["iq_source"], dataset_path=DATASET_PATH,
        dataset_mod=mod, dataset_snr=snr, sample_index=idx,
        embed_snr_margin=FIXED["embed_snr_margin"], num_bursts=FIXED["num_bursts"],
        dataset_mod_list=None, dataset_snr_list=None, sample_index_list=None,
        min_burst_gap=50, max_burst_gap=50, burst_gap_list=None, burst_power_scale_list=None,
        alignment_policy=FIXED["alignment_policy"], segment_hop=1, awn_preprocess=FIXED["awn_preprocess"],
    )

    row = {
        "combo_id": combo_id, "dataset": "RML2016.10a", "modulation": mod, "snr": snr,
        "sample_index": idx, "seed": seed, "label": label, "pred_direct": pred_direct,
        "direct_correct": pred_direct == label, "direct_nan": direct_nan,
        "output_dir": str(combo_output_dir), "original_sample_sha256": original_sample_sha256,
    }

    try:
        result = run_dry_run_experiment(cfg)
    except (ValueError, TypeError, RuntimeError) as exc:
        row.update({
            "run_status": "error", "failure_stage": "exception", "failure_reason": str(exc),
            "runtime_seconds": time.time() - t0,
        })
        print(f"[ERROR] {combo_id}: {exc}", file=sys.stderr)
        return row

    row["long_iq_sha256"] = result.get("long_iq_sha256")

    if result["run_status"] == "sensing_failed":
        row.update({
            "run_status": "sensing_failed",
            "failure_stage": result["failure_stage"], "failure_reason": result["failure_reason"],
            "detection_probability": result.get("detection_probability"),
            "false_alarm_region_rate": result.get("false_alarm_region_rate"),
            "segment_count": 0,
            "direct_sensed_agreement": None,
            "runtime_seconds": time.time() - t0,
        })
        if not direct_awn_ok:
            row["run_status"] = "error"
            row["failure_reason"] = f"direct-AMC AWN backend not real: {direct['awn_meta_direct']}"
        return row

    gt = result["ground_truth"] or {}
    summary_csv_path = result.get("summary_csv_path")
    pred_clean = None
    clean_nan = None
    if summary_csv_path:
        with open(summary_csv_path) as f:
            seg_rows = list(csv.DictReader(f))
        if seg_rows:
            # Single-burst mode always yields exactly one segment when
            # successful, so row 0 is used (same convention as
            # experiments/run_sensing_revalidation.py:_enrich_row).
            pred_clean = int(seg_rows[0]["pred_clean"])
            clean_nan = seg_rows[0]["clean_has_nan"] == "True" or seg_rows[0]["clean_has_inf"] == "True"

    sensed_awn_ok = result.get("awn_backend") == _REAL_MODEL_SOURCE and result.get("awn_status") == "ok"
    run_status = "ok"
    failure_reason = None
    if not (direct_awn_ok and sensed_awn_ok):
        run_status = "error"
        failure_reason = (
            f"non-real backend: direct_awn_ok={direct_awn_ok} sensed_awn_ok={sensed_awn_ok} "
            f"({result.get('awn_backend')})"
        )
        print(f"[ERROR] {combo_id}: {failure_reason}")

    row.update({
        "run_status": run_status, "failure_stage": None, "failure_reason": failure_reason,
        "pred_clean": pred_clean,
        "clean_correct": (pred_clean == label) if pred_clean is not None else None,
        "direct_sensed_agreement": (pred_direct == pred_clean) if pred_clean is not None else None,
        "detection_success": gt.get("detection_success"),
        "detection_probability": result.get("detection_probability"),
        "false_alarm_region_rate": result.get("false_alarm_region_rate"),
        "captured_signal_ratio": gt.get("captured_signal_ratio"),
        "extra_captured_noise_ratio": gt.get("extra_captured_noise_ratio"),
        "start_boundary_error": gt.get("start_boundary_error"),
        "end_boundary_error": gt.get("end_boundary_error"),
        "missed_sample_count": gt.get("missed_sample_count"),
        "false_occupied_sample_count": gt.get("false_occupied_sample_count"),
        "segment_count": result.get("n_segments"),
        "awn_backend": result.get("awn_backend"),
        "clean_nan": clean_nan,
        "runtime_seconds": time.time() - t0,
    })
    return row


def write_manifest(output_dir: Path, combos: List[dict], args) -> None:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    manifest = {
        "phase": 1, "tier": "default",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repo_commit": commit,
        "fixed_params": FIXED,
        "checkpoint": CHECKPOINT, "device": DEVICE, "dataset_path": DATASET_PATH,
        "mods": args.mods, "snrs": args.snrs, "sample_indices": args.sample_indices,
        "total_combos": len(combos),
        "combo_id_scheme": "{modulation}_snr{snr}_idx{sample_index}",
        "max_combos": args.max_combos,
    }
    with open(output_dir / "phase1_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def write_failures_csv(summary_path: Path, failures_path: Path) -> int:
    if not summary_path.exists():
        return 0
    with open(summary_path) as f:
        rows = list(csv.DictReader(f))
    failures = [r for r in rows if r["run_status"] != "ok"]
    if failures:
        with open(failures_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            w.writeheader()
            w.writerows(failures)
    return len(failures)


def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",")]


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                     help="Enumerate combo IDs and check uniqueness/coverage; execute nothing.")
    ap.add_argument("--output-dir", type=str, default="results/formal_phase1_sensing_clean_amc")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-combos", type=int, default=None)
    ap.add_argument("--mods", type=parse_str_list, default=ALL_MODS)
    ap.add_argument("--snrs", type=parse_int_list, default=ALL_SNRS)
    ap.add_argument("--n-per-cell", type=int, default=DEFAULT_N_PER_CELL,
                     help="Number of sample_index values per (mod,snr) cell (0..n-1). "
                          "Ignored if --sample-indices is passed explicitly.")
    ap.add_argument("--sample-indices", type=parse_int_list, default=None)
    args = ap.parse_args()
    sample_indices = args.sample_indices if args.sample_indices is not None else list(range(args.n_per_cell))
    args.sample_indices = sample_indices

    combos = build_combo_grid(args.mods, args.snrs, sample_indices)
    check_combo_ids_unique(combos)
    print(f"[phase1] {len(combos)} combos: mods={len(args.mods)} snrs={len(args.snrs)} "
          f"sample_indices={sample_indices} (n_per_cell={len(sample_indices)})")
    print(f"[phase1] attack=none, use_real_awn={FIXED['use_real_awn']}, "
          f"use_real_attack={FIXED['use_real_attack']}, use_real_topk={FIXED['use_real_topk']}")

    if args.dry_run:
        mods_covered = sorted({c["modulation"] for c in combos})
        snrs_covered = sorted({c["snr"] for c in combos})
        idx_covered = sorted({c["sample_index"] for c in combos})
        print(f"[phase1] modulation coverage: {len(mods_covered)}/{len(ALL_MODS)} -> {mods_covered}")
        print(f"[phase1] snr coverage: {len(snrs_covered)}/{len(ALL_SNRS)} -> {snrs_covered}")
        print(f"[phase1] sample_index coverage: {idx_covered}")
        print(f"[phase1] attack combos present (must be 0): "
              f"{sum(1 for c in combos if 'attack' in c)}")
        print(f"[phase1] --dry-run: {len(combos)} combos enumerated, all combo_ids unique. Nothing executed.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "phase1_summary.csv"

    done_ids = load_done_combo_ids(summary_path) if args.resume else set()
    if not args.resume and summary_path.exists():
        raise RuntimeError(
            f"{summary_path} already exists and --resume was not passed. "
            "Pass --resume to continue, or remove/move the existing output directory."
        )
    if done_ids:
        print(f"[resume] {len(done_ids)} combo_ids already done, will be skipped")

    write_manifest(output_dir, combos, args)

    print("[phase1] constructing shared AWNModelAdapter for direct/oracle inference...")
    direct_awn_adapter = AWNModelAdapter(checkpoint_path=CHECKPOINT, device=DEVICE)
    precheck_real_awn(direct_awn_adapter)

    writer = CsvWriter(summary_path, fresh=not summary_path.exists())
    attempted = 0
    t0 = time.time()
    try:
        for combo in combos:
            if combo["combo_id"] in done_ids:
                continue
            if args.max_combos is not None and attempted >= args.max_combos:
                print(f"[phase1] --max-combos {args.max_combos} reached, stopping")
                break
            row = run_combo(combo, direct_awn_adapter, output_dir)
            writer.write_row(row)
            attempted += 1
    finally:
        writer.close()

    elapsed = time.time() - t0
    print(f"[phase1] done in {elapsed:.1f}s ({attempted} combos attempted this run)")

    failures_path = output_dir / "phase1_failures.csv"
    n_failures = write_failures_csv(summary_path, failures_path)
    print(f"[phase1] {n_failures} failure row(s) written to {failures_path}" if n_failures else
          f"[phase1] 0 failures -- {failures_path} not written")


if __name__ == "__main__":
    main()
