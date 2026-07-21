"""
Phase 3 -- Adversarial attack effectiveness -- docs/formal_experiment_plan.md
section 4 (Phase 3, "reduced" tier) / docs/formal_experiment_matrix.csv
(phase=3, tier=reduced). Purpose: measure attack success rate (fgsm/pgd/cw)
across eps x modulation x SNR, real AWN + real attack, no Top-K defense
(deferred to Phase 4).

Fixed params (docs/formal_experiment_matrix.csv, phase=3/tier=reduced row
-- copied verbatim, not guessed): iq_source=radioml, use_real_awn=True,
use_real_attack=True, use_real_topk=False, attack_temperature=1.0
(default -- validated effective under radioml-native at scale, risk R4),
cw_c/steps/lr=defaults (1.0/20/0.01 -- validated 83-91% success at scale
under radioml-native, risk R4), same sensing defaults as Phase 1
(alignment_policy=max-energy, awn_preprocess=radioml-native,
threshold_factor=1.5, sensing_window_size=128, min_region_len=0,
merge_gap=0, seed=42).

Design: calls src/utils/pipeline.py:run_dry_run_experiment() directly per
combo -- Phase 3 has no cross-combo fairness/reuse constraint (unlike
Phase 4, which will need the same fair-Top-K-reuse treatment Phase 0
already validated), so there is no reason to bypass the existing,
already-validated pipeline function. Wrapped in the same incremental-
write + --resume CsvWriter pattern experiments/run_phase0_pilot.py and
experiments/run_phase1_sensing_baseline.py already established (3960
combos at ~1.8s/combo is large enough that resumability matters).

No sensing/AWN/attack algorithm code was written or modified. external/AWN
and external/adversarial-rf are not touched.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adapters.attack_adapter import _REAL_ATTACK_SOURCE  # noqa: E402
from src.adapters.awn_adapter import _REAL_MODEL_SOURCE  # noqa: E402
from src.sensing.radioml_source import RML2016_10A_CLASSES  # noqa: E402
from src.utils.config import ExperimentConfig  # noqa: E402
from src.utils.pipeline import run_dry_run_experiment  # noqa: E402

DATASET_PATH = "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl"
CHECKPOINT = "external/adversarial-rf/2016.10a_AWN.pkl"
DEVICE = "cpu"

DEFAULT_MODS = ["QPSK", "BPSK", "QAM16", "8PSK", "QAM64", "WBFM"]
DEFAULT_SNRS = [-10, -4, 0, 6, 12, 18]
DEFAULT_EPS = [0.01, 0.03, 0.05, 0.1, 0.3]
DEFAULT_N_PER_CELL = 10

FIXED = dict(
    iq_source="radioml",
    use_real_awn=True,
    use_real_attack=True,
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
    topk=50,  # unused/unreported this phase (use_real_topk=False; deferred to Phase 4)
    attack_temperature=1.0,
    cw_c=1.0, cw_steps=20, cw_lr=0.01,
)

SUMMARY_FIELDS = [
    "combo_id", "dataset", "modulation", "snr", "sample_index", "seed",
    "attack", "attack_eps", "attack_temperature", "cw_c", "cw_steps", "cw_lr",
    "label", "pred_clean", "pred_attacked", "clean_correct", "attacked_correct",
    "changed_by_attack",
    "iq_linf_clean_attacked", "iq_linf_normalized_clean_attacked", "eps_invariant_ok",
    "detection_success", "captured_signal_ratio", "segment_count",
    "awn_backend", "attack_backend", "clean_nan", "attacked_nan",
    "attack_training_before", "attack_training_after", "eval_mode_restored",
    "runtime_seconds", "run_status", "failure_stage", "failure_reason", "output_dir",
    "original_sample_sha256", "long_iq_sha256",
]


def build_combo_grid(mods, snrs, sample_indices, eps_values) -> List[dict]:
    combos = []
    for mod in mods:
        for snr in snrs:
            for idx in sample_indices:
                for eps in eps_values:
                    for attack in ("fgsm", "pgd"):
                        combos.append({
                            "combo_id": f"{mod}_snr{snr}_idx{idx}_{attack}_eps{eps}",
                            "modulation": mod, "snr": snr, "sample_index": idx,
                            "attack": attack, "attack_eps": eps,
                        })
                combos.append({
                    "combo_id": f"{mod}_snr{snr}_idx{idx}_cw",
                    "modulation": mod, "snr": snr, "sample_index": idx,
                    "attack": "cw", "attack_eps": None,
                })
    return combos


def check_combo_ids_unique(combos: List[dict]) -> None:
    ids = [c["combo_id"] for c in combos]
    if len(ids) != len(set(ids)):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise AssertionError(f"Duplicate combo_id(s) found: {dupes}")


def not_finite(x: np.ndarray) -> bool:
    return bool(np.isnan(x).any() or np.isinf(x).any())


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


def run_combo(combo: dict, output_dir: Path) -> dict:
    mod, snr, idx = combo["modulation"], combo["snr"], combo["sample_index"]
    attack, eps = combo["attack"], combo["attack_eps"]
    combo_id = combo["combo_id"]
    seed = FIXED["seed"]
    label = RML2016_10A_CLASSES[mod]
    t0 = time.time()

    combo_output_dir = output_dir / combo_id
    cfg = ExperimentConfig(
        snr=10.0, mod=mod, attack=attack, topk=FIXED["topk"],
        threshold_factor=FIXED["threshold_factor"], window_size=FIXED["window_size"],
        sensing_window_size=FIXED["sensing_window_size"], min_region_len=FIXED["min_region_len"],
        merge_gap=FIXED["merge_gap"], burst_len=600,
        output_dir=str(combo_output_dir), dry_run=True,
        use_real_topk=FIXED["use_real_topk"], use_real_awn=FIXED["use_real_awn"],
        checkpoint=CHECKPOINT, device=DEVICE,
        attack_eps=eps if eps is not None else 0.03,  # inert for cw (no eps attribute exists on torchattacks.CW)
        use_real_attack=FIXED["use_real_attack"],
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
        "sample_index": idx, "seed": seed, "attack": attack, "attack_eps": eps,
        "attack_temperature": FIXED["attack_temperature"], "cw_c": FIXED["cw_c"],
        "cw_steps": FIXED["cw_steps"], "cw_lr": FIXED["cw_lr"],
        "label": label, "output_dir": str(combo_output_dir),
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
    gen_meta = result.get("gen_meta") or {}
    row["original_sample_sha256"] = gen_meta.get("original_sample_sha256")

    if result["run_status"] == "sensing_failed":
        row.update({
            "run_status": "sensing_failed",
            "failure_stage": result["failure_stage"], "failure_reason": result["failure_reason"],
            "segment_count": 0,
            "runtime_seconds": time.time() - t0,
        })
        return row

    gt = result["ground_truth"] or {}
    summary_csv_path = result.get("summary_csv_path")
    seg0 = None
    if summary_csv_path:
        with open(summary_csv_path) as f:
            seg_rows = list(csv.DictReader(f))
        if seg_rows:
            seg0 = seg_rows[0]  # single-burst mode -> exactly one segment when successful

    pred_clean = int(seg0["pred_clean"]) if seg0 else None
    pred_attacked = int(seg0["pred_attacked"]) if seg0 else None
    iq_linf = float(seg0["iq_linf_clean_attacked"]) if seg0 else None
    iq_linf_norm = (
        float(seg0["iq_linf_normalized_clean_attacked"])
        if seg0 and seg0["iq_linf_normalized_clean_attacked"] not in (None, "", "None") else None
    )
    clean_nan = (seg0["clean_has_nan"] == "True" or seg0["clean_has_inf"] == "True") if seg0 else None
    attacked_nan = (seg0["attacked_has_nan"] == "True" or seg0["attacked_has_inf"] == "True") if seg0 else None
    # AttackAdapter.apply()'s finally block always calls
    # self.wrapped_model.eval() before returning (src/adapters/
    # attack_adapter.py) -- attack_training_after records
    # self.wrapped_model.training READ RIGHT AFTER that call, so it must
    # be False on every real-attack row. Checked explicitly here (not just
    # trusted structurally) per this round's review requirement.
    attack_training_before = seg0["attack_training_before"] if seg0 else None
    attack_training_after = seg0["attack_training_after"] if seg0 else None
    eval_mode_restored = (attack_training_after == "False") if seg0 else None

    # Pass condition from docs/formal_experiment_matrix.csv (phase=3 row):
    # iq_linf_normalized_clean_attacked must equal the requested attack_eps
    # exactly for fgsm/pgd (round 14 invariant, re-checked here, not
    # re-derived). N/A for cw (no eps attribute on torchattacks.CW).
    eps_invariant_ok = None
    if attack in ("fgsm", "pgd") and eps is not None and iq_linf_norm is not None:
        eps_invariant_ok = abs(iq_linf_norm - eps) < 1e-6

    awn_ok = result.get("awn_backend") == _REAL_MODEL_SOURCE and result.get("awn_status") == "ok"
    attack_ok = result.get("attack_backend") == _REAL_ATTACK_SOURCE and result.get("attack_status") == "ok"
    run_status = "ok"
    failure_reason = None
    if not (awn_ok and attack_ok):
        run_status = "error"
        failure_reason = (
            f"non-real backend: awn_ok={awn_ok}({result.get('awn_backend')}) "
            f"attack_ok={attack_ok}({result.get('attack_backend')})"
        )
        print(f"[ERROR] {combo_id}: {failure_reason}")
    elif eps_invariant_ok is False:
        run_status = "error"
        failure_reason = f"eps invariant violated: requested={eps} actual_iq_linf_normalized={iq_linf_norm}"
        print(f"[ERROR] {combo_id}: {failure_reason}")
    elif eval_mode_restored is False:
        run_status = "error"
        failure_reason = f"AWN model NOT restored to eval mode after attack (attack_training_after={attack_training_after})"
        print(f"[ERROR] {combo_id}: {failure_reason}")

    row.update({
        "run_status": run_status, "failure_stage": None, "failure_reason": failure_reason,
        "pred_clean": pred_clean, "pred_attacked": pred_attacked,
        "clean_correct": (pred_clean == label) if pred_clean is not None else None,
        "attacked_correct": (pred_attacked == label) if pred_attacked is not None else None,
        "changed_by_attack": (pred_attacked != pred_clean) if (pred_clean is not None and pred_attacked is not None) else None,
        "iq_linf_clean_attacked": iq_linf, "iq_linf_normalized_clean_attacked": iq_linf_norm,
        "eps_invariant_ok": eps_invariant_ok,
        "detection_success": gt.get("detection_success"),
        "captured_signal_ratio": gt.get("captured_signal_ratio"),
        "segment_count": result.get("n_segments"),
        "awn_backend": result.get("awn_backend"), "attack_backend": result.get("attack_backend"),
        "clean_nan": clean_nan, "attacked_nan": attacked_nan,
        "attack_training_before": attack_training_before, "attack_training_after": attack_training_after,
        "eval_mode_restored": eval_mode_restored,
        "runtime_seconds": time.time() - t0,
    })
    return row


def write_manifest(output_dir: Path, combos: List[dict], args) -> None:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    manifest = {
        "phase": 3, "tier": "reduced",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repo_commit": commit,
        "fixed_params": FIXED,
        "checkpoint": CHECKPOINT, "device": DEVICE, "dataset_path": DATASET_PATH,
        "mods": args.mods, "snrs": args.snrs, "sample_indices": args.sample_indices,
        "eps_values": args.eps, "attacks": ["fgsm", "pgd", "cw"],
        "total_combos": len(combos),
        "combo_id_scheme": "{modulation}_snr{snr}_idx{sample_index}_{attack}_eps{eps} (cw has no eps suffix)",
        "max_combos": args.max_combos,
    }
    with open(output_dir / "phase3_manifest.json", "w") as f:
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
    ap.add_argument("--output-dir", type=str, default="results/formal_phase3_attack_reduced")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-combos", type=int, default=None)
    ap.add_argument("--mods", type=parse_str_list, default=DEFAULT_MODS)
    ap.add_argument("--snrs", type=parse_int_list, default=DEFAULT_SNRS)
    ap.add_argument("--eps", type=parse_float_list, default=DEFAULT_EPS)
    ap.add_argument("--n-per-cell", type=int, default=DEFAULT_N_PER_CELL)
    ap.add_argument("--sample-indices", type=parse_int_list, default=None)
    args = ap.parse_args()
    sample_indices = args.sample_indices if args.sample_indices is not None else list(range(args.n_per_cell))
    args.sample_indices = sample_indices

    combos = build_combo_grid(args.mods, args.snrs, sample_indices, args.eps)
    check_combo_ids_unique(combos)
    n_cells = len(args.mods) * len(args.snrs) * len(sample_indices)
    print(f"[phase3] {len(combos)} combos: mods={len(args.mods)} snrs={len(args.snrs)} "
          f"sample_indices={sample_indices} eps={args.eps} -> {n_cells} cells x "
          f"(2 attacks x {len(args.eps)} eps + 1 cw) = {n_cells * (2*len(args.eps)+1)}")

    if args.dry_run:
        mods_covered = sorted({c["modulation"] for c in combos})
        snrs_covered = sorted({c["snr"] for c in combos})
        attacks_covered = Counter(c["attack"] for c in combos)
        print(f"[phase3] modulation coverage: {mods_covered}")
        print(f"[phase3] snr coverage: {snrs_covered}")
        print(f"[phase3] attack counts: {dict(attacks_covered)}")
        print(f"[phase3] --dry-run: {len(combos)} combos enumerated, all combo_ids unique. Nothing executed.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "phase3_summary.csv"

    done_ids = load_done_combo_ids(summary_path) if args.resume else set()
    if not args.resume and summary_path.exists():
        raise RuntimeError(
            f"{summary_path} already exists and --resume was not passed. "
            "Pass --resume to continue, or remove/move the existing output directory."
        )
    if done_ids:
        print(f"[resume] {len(done_ids)} combo_ids already done, will be skipped")

    write_manifest(output_dir, combos, args)

    writer = CsvWriter(summary_path, fresh=not summary_path.exists())
    attempted = 0
    t0 = time.time()
    try:
        for combo in combos:
            if combo["combo_id"] in done_ids:
                continue
            if args.max_combos is not None and attempted >= args.max_combos:
                print(f"[phase3] --max-combos {args.max_combos} reached, stopping")
                break
            row = run_combo(combo, output_dir)
            writer.write_row(row)
            attempted += 1
    finally:
        writer.close()

    elapsed = time.time() - t0
    print(f"[phase3] done in {elapsed:.1f}s ({attempted} combos attempted this run)")

    failures_path = output_dir / "phase3_failures.csv"
    n_failures = write_failures_csv(summary_path, failures_path)
    print(f"[phase3] {n_failures} failure row(s) written to {failures_path}" if n_failures else
          f"[phase3] 0 failures -- {failures_path} not written")


if __name__ == "__main__":
    main()
