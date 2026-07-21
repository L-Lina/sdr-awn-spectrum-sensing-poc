"""
Phase 4 -- Top-K defense effectiveness -- docs/formal_experiment_plan.md
section 4 (Phase 4) / docs/formal_experiment_matrix.csv (phase=4,
tier=reduced/quick). Purpose: measure Top-K defense recovery rate across
K x attack x eps x modulation x SNR, with the SAME attacked IQ reused
literally (not regenerated) across all 4 K values -- the same strict
fairness requirement Phase 0's pilot was built to satisfy, extended here
to Phase 3's full (mod, snr, eps, attack) combo dimensions.

Does NOT call src/utils/pipeline.py:run_dry_run_experiment() -- for the
same reason experiments/run_phase0_pilot.py doesn't: that function
computes clean/attack/Top-K in one pass per combo and only guarantees
"fair" Top-K via reproducible regeneration under a fixed seed (round 12's
method, verified equal after the fact). This round's requirement is
stricter: the SAME in-memory attacked IQ array must be reused literally
across all 4 K values, never regenerated. This script is architecturally
identical to experiments/run_phase0_pilot.py (same building-block calls:
energy_detect, select_aligned_segments, apply_awn_preprocess,
AWNModelAdapter, AttackAdapter, TopKAdapter, ground-truth metrics -- all
unmodified, already-validated functions), extended with attack_eps as a
swept dimension (fgsm/pgd only; cw and none have no eps) to match Phase
3's combo grid, and with the eval-mode-restoration tracking added during
Phase 3's code review (attack_training_before/after, eval_mode_restored).

No sensing/AWN/attack/Top-K algorithm code was written or modified.
external/AWN and external/adversarial-rf are not touched.
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
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adapters.attack_adapter import AttackAdapter, _REAL_ATTACK_SOURCE  # noqa: E402
from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE  # noqa: E402
from src.adapters.topk_adapter import TopKAdapter, _REAL_SOURCE as _REAL_TOPK_SOURCE  # noqa: E402
from src.sensing.energy_detection import (  # noqa: E402
    energy_detect,
    filter_by_min_length,
    mask_to_regions,
    merge_close_regions,
)
from src.sensing.ground_truth_metrics import (  # noqa: E402
    compute_sensing_ground_truth_metrics,
    derive_batch_aggregate_sensing_fields,
)
from src.sensing.iq_source import validate_iq  # noqa: E402
from src.sensing.normalize import apply_awn_preprocess, to_awn_input  # noqa: E402
from src.sensing.radioml_source import (  # noqa: E402
    RML2016_10A_CLASSES,
    embed_sample_in_noise,
    load_radioml_sample,
)
from src.sensing.segmentation import select_aligned_segments  # noqa: E402
from src.utils.pipeline import _seed_everything  # noqa: E402

DATASET_PATH = "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl"
CHECKPOINT = "external/adversarial-rf/2016.10a_AWN.pkl"
DEVICE = "cpu"
N_SAMPLES = 8192
WINDOW_SIZE = 128

DEFAULT_MODS = ["QPSK", "BPSK", "QAM16", "8PSK", "QAM64", "WBFM"]
DEFAULT_SNRS = [-10, -4, 0, 6, 12, 18]
DEFAULT_EPS = [0.01, 0.03, 0.05, 0.1, 0.3]
DEFAULT_ATTACKS = ["fgsm", "pgd", "cw"]  # "none" is smoke-test-only, not part of the formal grid
DEFAULT_TOPKS = [10, 20, 30, 40]
DEFAULT_N_PER_CELL = 10

FIXED = dict(
    iq_source="radioml",
    alignment_policy="max-energy",
    awn_preprocess="radioml-native",
    threshold_factor=1.5,
    sensing_window_size=128,
    min_region_len=0,
    merge_gap=0,
    window_size=WINDOW_SIZE,
    segment_hop=1,
    num_bursts=1,
    embed_snr_margin=20.0,
    seed=42,
    attack_temperature=1.0,
    cw_c=1.0, cw_steps=20, cw_lr=0.01,
)

SUMMARY_FIELDS = [
    "combo_id", "dataset", "modulation", "snr", "sample_index", "seed",
    "attack", "attack_eps", "attack_temperature", "cw_c", "cw_steps", "cw_lr",
    "topk", "retained_freq_ratio",
    "threshold_factor", "sensing_window_size", "min_region_len", "merge_gap",
    "detection_success", "detection_probability", "false_alarm_rate",
    "captured_signal_ratio", "extra_captured_noise_ratio",
    "start_boundary_error", "end_boundary_error",
    "missed_sample_count", "false_occupied_sample_count", "segment_count",
    "label", "pred_clean", "pred_attacked", "pred_defended",
    "clean_correct", "attacked_correct", "defended_correct",
    "changed_by_attack", "attacked_wrong", "recovered_by_defense",
    "defense_changed_prediction", "clean_broken_by_defense",
    "iq_linf_clean_attacked", "iq_linf_normalized_clean_attacked", "eps_invariant_ok",
    "iq_linf_attacked_defended",
    "awn_backend", "attack_backend", "topk_backend",
    "clean_nan", "attacked_nan", "defended_nan",
    "attack_training_before", "attack_training_after", "eval_mode_restored",
    "runtime_seconds", "run_status", "failure_stage", "failure_reason", "output_dir",
    "clean_iq_sha256", "attacked_iq_sha256",
]


def build_attack_instances(eps_values: List[float], attacks: List[str]) -> List[tuple]:
    """Returns [(attack, eps_or_None), ...] -- eps only applies to fgsm/pgd
    (matches experiments/run_phase3_attack_effectiveness.py's exact scheme:
    cw and none have no eps concept)."""
    instances = []
    if "none" in attacks:
        instances.append(("none", None))
    for a in ("fgsm", "pgd"):
        if a in attacks:
            for eps in eps_values:
                instances.append((a, eps))
    if "cw" in attacks:
        instances.append(("cw", None))
    return instances


def build_combo_grid(mods, snrs, sample_indices, eps_values, attacks, topks) -> List[dict]:
    instances = build_attack_instances(eps_values, attacks)
    combos = []
    for mod in mods:
        for snr in snrs:
            for idx in sample_indices:
                for attack, eps in instances:
                    eps_suffix = f"_eps{eps}" if eps is not None else ""
                    for topk in topks:
                        combo_id = f"{mod}_snr{snr}_idx{idx}_{attack}{eps_suffix}_k{topk}"
                        combos.append({
                            "combo_id": combo_id, "modulation": mod, "snr": snr,
                            "sample_index": idx, "attack": attack, "attack_eps": eps, "topk": topk,
                        })
    return combos


def check_combo_ids_unique(combos: List[dict]) -> None:
    ids = [c["combo_id"] for c in combos]
    if len(ids) != len(set(ids)):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise AssertionError(f"Duplicate combo_id(s) found: {dupes}")


def sha256_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def not_finite(x: np.ndarray) -> bool:
    return bool(np.isnan(x).any() or np.isinf(x).any())


def precheck_real_backends(awn_adapter: AWNModelAdapter, attack_adapter: AttackAdapter, topk_adapter: TopKAdapter) -> None:
    problems = []
    if awn_adapter.backend_name != _REAL_MODEL_SOURCE or awn_adapter.status != "ok":
        problems.append(f"AWN: backend={awn_adapter.backend_name!r} status={awn_adapter.status!r} notes={awn_adapter.notes}")
    if attack_adapter.backend_name != _REAL_ATTACK_SOURCE or attack_adapter.status != "ok":
        problems.append(f"Attack: backend={attack_adapter.backend_name!r} status={attack_adapter.status!r} notes={attack_adapter.notes}")
    if not topk_adapter.backend_available or topk_adapter.backend_name != _REAL_TOPK_SOURCE:
        problems.append(f"Top-K: backend={topk_adapter.backend_name!r} available={topk_adapter.backend_available} notes={topk_adapter.notes}")
    if problems:
        msg = "Real-backend precheck FAILED -- refusing to run any combo:\n" + "\n".join(f"  - {p}" for p in problems)
        raise RuntimeError(msg)
    print(f"[precheck] real backends confirmed: awn={_REAL_MODEL_SOURCE}, attack={_REAL_ATTACK_SOURCE}, topk={_REAL_TOPK_SOURCE}")


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


def run_sample(
    mod: str, snr: int, idx: int, output_base: Path,
    awn_adapter: AWNModelAdapter, attack_adapter: AttackAdapter, topk_adapter: TopKAdapter,
    instances: List[tuple], topks: List[int], done_ids: set, writer: CsvWriter,
) -> None:
    def _cid(attack, eps, topk):
        eps_suffix = f"_eps{eps}" if eps is not None else ""
        return f"{mod}_snr{snr}_idx{idx}_{attack}{eps_suffix}_k{topk}"

    sample_combo_ids = {_cid(a, e, k) for (a, e) in instances for k in topks}
    if sample_combo_ids <= done_ids:
        print(f"[skip] {mod} snr={snr} idx={idx}: all {len(sample_combo_ids)} combos already done")
        return

    output_dir = output_base / f"{mod}_snr{snr}_idx{idx}"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = FIXED["seed"]
    label = RML2016_10A_CLASSES[mod]

    t_sample_start = time.time()
    original_sample = load_radioml_sample(DATASET_PATH, mod, snr, idx)
    iq, embed_meta = embed_sample_in_noise(
        original_sample, n_samples=N_SAMPLES, embed_snr_margin=FIXED["embed_snr_margin"], seed=seed,
    )
    iq = validate_iq(iq)

    mask = energy_detect(iq, window=FIXED["sensing_window_size"], threshold_factor=FIXED["threshold_factor"])
    raw_regions = mask_to_regions(mask)
    merged_regions = merge_close_regions(raw_regions, merge_gap=FIXED["merge_gap"])

    sensing_failure_stage = None
    sensing_failure_reason = None
    regions = []
    try:
        regions = filter_by_min_length(merged_regions, min_len=FIXED["min_region_len"])
    except RuntimeError as exc:
        sensing_failure_stage = "filter_by_min_length"
        sensing_failure_reason = str(exc)

    ground_truth = None
    x_clean = None
    if sensing_failure_stage is None:
        ground_truth = compute_sensing_ground_truth_metrics(embed_meta["true_start"], embed_meta["true_end"], regions)
        try:
            segments, alignment_meta = select_aligned_segments(
                iq, regions, seg_len=FIXED["window_size"], policy=FIXED["alignment_policy"], hop=FIXED["segment_hop"],
            )
            segments = apply_awn_preprocess(segments, policy=FIXED["awn_preprocess"])
            x_clean = to_awn_input(segments, seg_len=FIXED["window_size"])
        except RuntimeError as exc:
            sensing_failure_stage = "segment_regions"
            sensing_failure_reason = str(exc)

    sensing_agg = derive_batch_aggregate_sensing_fields(ground_truth, None, regions, N_SAMPLES)

    if sensing_failure_stage is not None:
        print(f"[sensing] FAILED {mod} snr={snr} idx={idx}: stage={sensing_failure_stage}: {sensing_failure_reason}")
        runtime = time.time() - t_sample_start
        for attack, eps in instances:
            for topk in topks:
                combo_id = _cid(attack, eps, topk)
                if combo_id in done_ids:
                    continue
                row = _base_row(combo_id, mod, snr, idx, seed, attack, eps, topk, label)
                row.update({
                    "detection_success": False,
                    "detection_probability": sensing_agg["detection_probability"],
                    "false_alarm_rate": sensing_agg["false_alarm_region_rate"],
                    "segment_count": 0,
                    "runtime_seconds": runtime,
                    "run_status": "sensing_failed",
                    "failure_stage": sensing_failure_stage,
                    "failure_reason": sensing_failure_reason,
                    "output_dir": str(output_dir),
                })
                writer.write_row(row)
        return

    logits_clean, awn_meta_clean = awn_adapter.infer(x_clean, seed=seed)
    pred_clean = int(np.argmax(logits_clean, axis=1)[0])
    clean_iq_hash = sha256_array(x_clean)
    clean_nan = not_finite(x_clean)
    n_segments = x_clean.shape[0]

    gt_row_fields = {
        "detection_success": ground_truth["detection_success"],
        "detection_probability": sensing_agg["detection_probability"],
        "false_alarm_rate": sensing_agg["false_alarm_region_rate"],
        "captured_signal_ratio": ground_truth["captured_signal_ratio"],
        "extra_captured_noise_ratio": ground_truth["extra_captured_noise_ratio"],
        "start_boundary_error": ground_truth["start_boundary_error"],
        "end_boundary_error": ground_truth["end_boundary_error"],
        "missed_sample_count": ground_truth["missed_sample_count"],
        "false_occupied_sample_count": ground_truth["false_occupied_sample_count"],
        "segment_count": n_segments,
    }

    for attack, eps in instances:
        inst_ids = {_cid(attack, eps, k) for k in topks}
        if inst_ids <= done_ids:
            print(f"[skip] {mod} snr={snr} idx={idx} attack={attack} eps={eps}: all {len(topks)} K's already done")
            continue

        t_attack_start = time.time()
        _seed_everything(seed)  # matches pipeline.py's per-run seeding discipline (PGD random_start)
        x_adv, attack_meta = attack_adapter.apply(
            x_clean, attack=attack, eps=(eps if eps is not None else 0.03),  # inert for cw/none
            temperature=FIXED["attack_temperature"], seed=seed, diagnostics=False,
            cw_c=FIXED["cw_c"], cw_steps=FIXED["cw_steps"], cw_lr=FIXED["cw_lr"],
        )
        logits_attacked, awn_meta_attacked = awn_adapter.infer(x_adv, seed=seed)
        pred_attacked = int(np.argmax(logits_attacked, axis=1)[0])
        attacked_iq_hash = sha256_array(x_adv)  # computed ONCE per attack instance, reused for all 4 K rows below
        attacked_nan = not_finite(x_adv)
        changed_by_attack = pred_attacked != pred_clean
        attacked_wrong = pred_attacked != label

        iq_linf_ca = float(np.max(np.abs(x_adv - x_clean)))
        # eps invariant check (fgsm/pgd only) -- attack_meta carries the
        # normalized-domain Linf if AttackAdapter computed it (real backend).
        iq_linf_norm = attack_meta.get("attack_iq_linf_normalized")
        iq_linf_norm_val = float(iq_linf_norm[0]) if iq_linf_norm is not None else None
        eps_invariant_ok = (
            (abs(iq_linf_norm_val - eps) < 1e-6) if (eps is not None and iq_linf_norm_val is not None) else None
        )

        attack_training_before = attack_meta.get("attack_training_before")
        attack_training_after = attack_meta.get("attack_training_after")
        eval_mode_restored = (attack_training_after is False) if attack_training_after is not None else None

        seen_hashes_this_attack = set()
        for topk in topks:
            combo_id = _cid(attack, eps, topk)
            if combo_id in done_ids:
                continue

            x_defended, topk_meta = topk_adapter.apply(x_adv, topk=topk)
            recheck_hash = sha256_array(x_adv)
            assert recheck_hash == attacked_iq_hash, (
                f"attacked IQ mutated between K values for {attack} eps={eps}: "
                f"{attacked_iq_hash} -> {recheck_hash}"
            )
            seen_hashes_this_attack.add(recheck_hash)

            logits_defended, awn_meta_defended = awn_adapter.infer(x_defended, seed=seed)
            pred_defended = int(np.argmax(logits_defended, axis=1)[0])
            defended_nan = not_finite(x_defended)
            recovered_by_defense = changed_by_attack and (pred_defended == pred_clean)
            defense_changed_prediction = pred_defended != pred_attacked
            clean_broken_by_defense = (pred_clean == label) and (pred_defended != label)
            iq_linf_ad = float(np.max(np.abs(x_defended - x_adv)))
            retained_freq_ratio = (min(topk, WINDOW_SIZE) / WINDOW_SIZE) if topk > 0 else 1.0

            awn_ok = all(
                m["awn_backend"] == _REAL_MODEL_SOURCE and m["awn_status"] == "ok"
                for m in (awn_meta_clean, awn_meta_attacked, awn_meta_defended)
            )
            attack_ok = attack_meta["attack_backend"] == _REAL_ATTACK_SOURCE and attack_meta["attack_status"] == "ok"
            topk_ok = topk_meta["topk_backend"] == _REAL_TOPK_SOURCE and topk_meta["topk_status"] == "ok"

            run_status, failure_reason = "ok", None
            if not (awn_ok and attack_ok and topk_ok):
                run_status = "error"
                failure_reason = (
                    f"non-real backend: awn_ok={awn_ok}({awn_meta_clean['awn_backend']}/"
                    f"{awn_meta_attacked['awn_backend']}/{awn_meta_defended['awn_backend']}) "
                    f"attack_ok={attack_ok}({attack_meta['attack_backend']}) "
                    f"topk_ok={topk_ok}({topk_meta['topk_backend']})"
                )
            elif eps_invariant_ok is False:
                run_status = "error"
                failure_reason = f"eps invariant violated: requested={eps} actual={iq_linf_norm_val}"
            elif eval_mode_restored is False:
                run_status = "error"
                failure_reason = f"AWN model NOT restored to eval mode after attack (attack_training_after={attack_training_after})"
            if run_status == "error":
                print(f"[ERROR] {combo_id}: {failure_reason}")

            row = _base_row(combo_id, mod, snr, idx, seed, attack, eps, topk, label)
            row.update(gt_row_fields)
            row.update({
                "retained_freq_ratio": retained_freq_ratio,
                "pred_clean": pred_clean, "pred_attacked": pred_attacked, "pred_defended": pred_defended,
                "clean_correct": pred_clean == label, "attacked_correct": pred_attacked == label,
                "defended_correct": pred_defended == label,
                "changed_by_attack": changed_by_attack, "attacked_wrong": attacked_wrong,
                "recovered_by_defense": recovered_by_defense,
                "defense_changed_prediction": defense_changed_prediction,
                "clean_broken_by_defense": clean_broken_by_defense,
                "iq_linf_clean_attacked": iq_linf_ca, "iq_linf_normalized_clean_attacked": iq_linf_norm_val,
                "eps_invariant_ok": eps_invariant_ok, "iq_linf_attacked_defended": iq_linf_ad,
                "awn_backend": awn_meta_defended["awn_backend"], "attack_backend": attack_meta["attack_backend"],
                "topk_backend": topk_meta["topk_backend"],
                "clean_nan": clean_nan, "attacked_nan": attacked_nan, "defended_nan": defended_nan,
                "attack_training_before": attack_training_before, "attack_training_after": attack_training_after,
                "eval_mode_restored": eval_mode_restored,
                "runtime_seconds": time.time() - t_attack_start,
                "run_status": run_status, "failure_stage": None, "failure_reason": failure_reason,
                "output_dir": str(output_dir),
                "clean_iq_sha256": clean_iq_hash, "attacked_iq_sha256": attacked_iq_hash,
            })
            writer.write_row(row)

        if len(seen_hashes_this_attack) != 1:
            raise AssertionError(
                f"FAIR TOP-K REUSE VIOLATION for {mod} snr={snr} idx={idx} attack={attack} eps={eps}: "
                f"attacked IQ hash varied across K values: {seen_hashes_this_attack}"
            )
        print(f"[fair-topk] {mod} snr={snr} idx={idx} attack={attack} eps={eps}: "
              f"1 unique attacked_iq_sha256 across {len(topks)} K values -- OK")


def _base_row(combo_id, mod, snr, idx, seed, attack, eps, topk, label) -> dict:
    return {
        "combo_id": combo_id, "dataset": "RML2016.10a", "modulation": mod, "snr": snr,
        "sample_index": idx, "seed": seed, "attack": attack, "attack_eps": eps,
        "attack_temperature": FIXED["attack_temperature"],
        "cw_c": FIXED["cw_c"], "cw_steps": FIXED["cw_steps"], "cw_lr": FIXED["cw_lr"],
        "topk": topk, "threshold_factor": FIXED["threshold_factor"],
        "sensing_window_size": FIXED["sensing_window_size"], "min_region_len": FIXED["min_region_len"],
        "merge_gap": FIXED["merge_gap"], "label": label,
    }


def write_manifest(output_dir: Path, combos: List[dict], instances: List[tuple], args) -> None:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    manifest = {
        "phase": 4, "tier": args.tier_label,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repo_commit": commit,
        "fixed_params": FIXED,
        "checkpoint": CHECKPOINT, "device": DEVICE, "dataset_path": DATASET_PATH,
        "mods": args.mods, "snrs": args.snrs, "sample_indices": args.sample_indices,
        "eps_values": args.eps, "attacks": args.attacks, "topks": args.topks,
        "attack_instances_per_cell": len(instances),
        "total_combos": len(combos),
        "combo_id_scheme": "{modulation}_snr{snr}_idx{sample_index}_{attack}[_eps{eps}]_k{topk}",
        "max_combos": args.max_combos,
    }
    with open(output_dir / "phase4_manifest.json", "w") as f:
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


def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",")]


def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",")]


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output-dir", type=str, default="results/formal_phase4_defense_reduced")
    ap.add_argument("--tier-label", type=str, default="reduced")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-combos", type=int, default=None)
    ap.add_argument("--mods", type=parse_str_list, default=DEFAULT_MODS)
    ap.add_argument("--snrs", type=parse_int_list, default=DEFAULT_SNRS)
    ap.add_argument("--eps", type=parse_float_list, default=DEFAULT_EPS)
    ap.add_argument("--attacks", type=parse_str_list, default=DEFAULT_ATTACKS)
    ap.add_argument("--topks", type=parse_int_list, default=DEFAULT_TOPKS)
    ap.add_argument("--n-per-cell", type=int, default=DEFAULT_N_PER_CELL)
    ap.add_argument("--sample-indices", type=parse_int_list, default=None)
    args = ap.parse_args()
    sample_indices = args.sample_indices if args.sample_indices is not None else list(range(args.n_per_cell))
    args.sample_indices = sample_indices

    instances = build_attack_instances(args.eps, args.attacks)
    combos = build_combo_grid(args.mods, args.snrs, sample_indices, args.eps, args.attacks, args.topks)
    check_combo_ids_unique(combos)
    n_cells = len(args.mods) * len(args.snrs) * len(sample_indices)
    print(f"[phase4] {len(combos)} combos (final rows): {n_cells} cells x {len(instances)} attack-instances/cell "
          f"x {len(args.topks)} topk = {n_cells * len(instances) * len(args.topks)}")
    print(f"[phase4] attack-instances (pre-topk-expansion): {n_cells * len(instances)}")
    print(f"[phase4] mods={args.mods} snrs={args.snrs} sample_indices={sample_indices} "
          f"eps={args.eps} attacks={args.attacks} topks={args.topks}")

    if args.dry_run:
        from collections import Counter
        attack_counts = Counter(c["attack"] for c in combos)
        print(f"[phase4] per-attack final-row counts: {dict(attack_counts)}")
        topk_counts = Counter(c["topk"] for c in combos)
        print(f"[phase4] per-topk final-row counts (must all be equal): {dict(topk_counts)}")
        print(f"[phase4] --dry-run: {len(combos)} combos enumerated, all combo_ids unique. Nothing executed.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "phase4_summary.csv"

    done_ids = load_done_combo_ids(summary_path) if args.resume else set()
    if not args.resume and summary_path.exists():
        raise RuntimeError(
            f"{summary_path} already exists and --resume was not passed. "
            "Pass --resume to continue, or remove/move the existing output directory."
        )
    if done_ids:
        print(f"[resume] {len(done_ids)} combo_ids already done, will be skipped")

    write_manifest(output_dir, combos, instances, args)

    print("[phase4] constructing real AWN/attack/Top-K adapters (once, reused across the whole run)...")
    awn_adapter = AWNModelAdapter(checkpoint_path=CHECKPOINT, device=DEVICE)
    attack_adapter = AttackAdapter(awn_model=awn_adapter.model, device=DEVICE)
    topk_adapter = TopKAdapter()
    precheck_real_backends(awn_adapter, attack_adapter, topk_adapter)

    writer = CsvWriter(summary_path, fresh=not summary_path.exists())

    samples = [(m, s, i) for m in args.mods for s in args.snrs for i in sample_indices]
    attempted = 0
    t0 = time.time()
    try:
        for mod, snr, idx in samples:
            def _cid(attack, eps, topk):
                eps_suffix = f"_eps{eps}" if eps is not None else ""
                return f"{mod}_snr{snr}_idx{idx}_{attack}{eps_suffix}_k{topk}"
            sample_ids = {_cid(a, e, k) for (a, e) in instances for k in args.topks}
            if sample_ids <= done_ids:
                continue
            if args.max_combos is not None and attempted >= args.max_combos:
                print(f"[phase4] --max-combos {args.max_combos} reached, stopping")
                break
            run_sample(mod, snr, idx, output_dir, awn_adapter, attack_adapter, topk_adapter,
                       instances, args.topks, done_ids, writer)
            attempted += len(sample_ids - done_ids)
            done_ids |= sample_ids
    finally:
        writer.close()

    elapsed = time.time() - t0
    print(f"[phase4] done in {elapsed:.1f}s ({attempted} combos attempted this run)")

    failures_path = output_dir / "phase4_failures.csv"
    n_failures = write_failures_csv(summary_path, failures_path)
    print(f"[phase4] {n_failures} failure row(s) written to {failures_path}" if n_failures else
          f"[phase4] 0 failures -- {failures_path} not written")


if __name__ == "__main__":
    main()
