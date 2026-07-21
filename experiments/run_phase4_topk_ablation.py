"""
Phase 4 Top-K preprocessing-policy ablation (round 24). Purpose: DIAGNOSE
why fixed-K Top-K filtering was found net-harmful in the Phase 4
reduced-tier (docs/formal_experiment_plan.md sections 13-14) -- NOT to
change the formal defense implementation. Compares three preprocessing
policies around the SAME unmodified fft_topk_denoise() call:

  A. current_radioml_native -- the existing formal path, completely
     unmodified: TopKAdapter.apply(x_adv, topk) as-is, no normalization.
  B. normalized_topk_rescaled -- records each sample's own power scale,
     temporarily normalizes x_adv to unit average power (same
     mathematical convention as src/sensing/normalize.py:
     normalize_segments, adapted here to real [N,2,T] tensors since that
     function only accepts complex64 [N,T]), runs the SAME
     fft_topk_denoise on the normalized array, then rescales the result
     back to x_adv's OWN original power level before AWN inference --
     AWN never sees normalized-scale input, only correctly-rescaled
     defended IQ.
  C. legacy_awn_all_reference -- reproduces external/adversarial-rf/
     AWN_All.py's actual historical usage line-for-line (confirmed in
     round 23's diagnosis): x_norm = (x_adv + 0.02) / 0.04, then
     fft_topk_denoise(x_norm, topk), fed DIRECTLY to AWN with no
     rescale-back (matching AWN_All.py:337-339 exactly). Diagnostic
     ONLY -- AWN_All.py targets different checkpoint files
     (AWN_CLS_best_acc.pth / Detector_CNN_best.pth, not the pinned
     2016.10a_AWN.pkl this repo uses), and round 10 already traced the
     ACTUAL training pipeline for the pinned checkpoint to apply ZERO
     normalization -- so policy C's input scale is expected to be
     substantially out-of-distribution for this checkpoint. This is
     explicitly flagged in the output, not presented as a fair
     apples-to-apples legacy replication.

None of the three policies modifies fft_topk_denoise() itself or
TopKAdapter -- all three call the exact same underlying function, only
the surrounding normalize/denormalize wrapper differs. No policy is
written back into src/adapters/topk_adapter.py or set as any default.

Architecturally extends experiments/run_phase4_defense_effectiveness.py's
fair-reuse pattern one level further: the SAME attacked IQ (x_adv) is
computed exactly once per attack-instance and reused across BOTH all 10
K values AND all 3 preprocessing policies (30 combinations per instance),
never regenerated. No sensing/AWN/attack/Top-K algorithm code was written
or modified. external/AWN and external/adversarial-rf are not touched
(only read).
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
from typing import List, Optional

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

DEFAULT_MODS = ["QPSK", "BPSK", "QAM16", "QAM64", "WBFM", "AM-SSB"]
DEFAULT_SNRS = [-10, 0, 10, 18]
DEFAULT_ATTACKS = ["none", "fgsm", "pgd", "cw"]
DEFAULT_EPS = [0.05]  # single representative value by default; --eps accepts a comma-separated list
DEFAULT_TOPKS = [10, 20, 30, 40, 50, 64, 80, 96, 112, 128]
DEFAULT_POLICIES = ["current_radioml_native", "normalized_topk_rescaled", "legacy_awn_all_reference"]
DEFAULT_N_PER_CELL = 3

FIXED = dict(
    iq_source="radioml", alignment_policy="max-energy", awn_preprocess="radioml-native",
    threshold_factor=1.5, sensing_window_size=128, min_region_len=0, merge_gap=0,
    window_size=WINDOW_SIZE, segment_hop=1, num_bursts=1, embed_snr_margin=20.0, seed=42,
    attack_temperature=1.0, cw_c=1.0, cw_steps=20, cw_lr=0.01,
    awn_all_norm_offset=0.02, awn_all_norm_scale=0.04,  # AWN_All.py's exact constants
)

SUMMARY_FIELDS = [
    "combo_id", "dataset", "modulation", "snr", "sample_index", "seed",
    "attack", "attack_eps", "topk", "policy",
    "label", "pred_clean", "pred_attacked", "pred_defended",
    "clean_correct", "attacked_correct", "defended_correct",
    "changed_by_attack", "attacked_wrong", "recovered_by_defense", "defense_changed_prediction",
    "clean_broken_by_defense",
    "iq_linf_clean_attacked", "iq_linf_attacked_defended", "iq_l2_attacked_defended",
    "pred_agreement_defended_vs_attacked",
    "awn_backend", "attack_backend", "topk_backend",
    "clean_nan", "attacked_nan", "defended_nan",
    "attack_training_after", "eval_mode_restored",
    "runtime_seconds", "run_status", "failure_stage", "failure_reason", "output_dir",
    "attacked_iq_sha256", "policy_notes",
    # Additional traceability hashes (round 25) -- not required for fairness
    # (attacked_iq_sha256 alone already proves that), but needed to verify
    # cross-process reproducibility explicitly at every pipeline stage
    # (original capture -> sensing/segment selection -> clean AWN input ->
    # attacked -> defended), not just via numeric column equality.
    "original_iq_sha256", "clean_iq_sha256", "defended_iq_sha256",
    "selected_segment_start", "selected_segment_end",
]


def apply_policy(x_adv: np.ndarray, topk: int, topk_adapter: TopKAdapter, policy: str):
    """Returns (x_defended, topk_meta, policy_notes). All three policies
    call the exact same, unmodified fft_topk_denoise (via topk_adapter),
    differing only in the normalize/denormalize wrapper around it."""
    if policy == "current_radioml_native":
        x_def, meta = topk_adapter.apply(x_adv, topk=topk)
        return x_def, meta, "no normalization (formal path, unmodified)"

    if policy == "normalized_topk_rescaled":
        # Same mathematical convention as src/sensing/normalize.py:
        # normalize_segments (unit-average-power), adapted to real [N,2,T]
        # tensors (that function only accepts complex64 [N,T]).
        power = np.mean(x_adv ** 2, axis=(1, 2), keepdims=True)
        power = np.maximum(power, 1e-12)
        scale = np.sqrt(power)
        x_norm = (x_adv / scale).astype(np.float32)
        x_topk_norm, meta = topk_adapter.apply(x_norm, topk=topk)
        x_def = (x_topk_norm * scale).astype(np.float32)  # rescale back to x_adv's OWN original power
        return x_def, meta, f"normalized to unit power (scale={float(scale.flatten()[0]):.6e}), rescaled back before AWN"

    if policy == "legacy_awn_all_reference":
        # Reproduces external/adversarial-rf/AWN_All.py:335-339 exactly:
        # normalize_data(x) = (x + 0.02) / 0.04, then filter_top_components_torch,
        # fed DIRECTLY to the model -- NO rescale-back. Diagnostic only; see
        # module docstring for why this is expected out-of-distribution for
        # the pinned 2016.10a checkpoint.
        offset, sc = FIXED["awn_all_norm_offset"], FIXED["awn_all_norm_scale"]
        x_norm = ((x_adv + offset) / sc).astype(np.float32)
        x_def, meta = topk_adapter.apply(x_norm, topk=topk)  # fed directly, no rescale-back
        return x_def, meta, (
            "AWN_All.py-style (x+0.02)/0.04, NOT rescaled back -- diagnostic only, "
            "likely out-of-distribution for the pinned 2016.10a checkpoint (round 10 evidence: "
            "the real training pipeline applies zero normalization)"
        )

    raise ValueError(f"Unknown policy {policy!r}")


def build_attack_instances(attacks: List[str], eps_values: List[float]) -> List[tuple]:
    """eps_values: list of representative eps values swept for fgsm/pgd
    (matches experiments/run_phase3_attack_effectiveness.py's and
    run_phase4_defense_effectiveness.py's multi-eps convention). cw and
    none have no eps concept, same as those scripts."""
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


def build_combo_grid(mods, snrs, sample_indices, attacks, eps, topks, policies) -> List[dict]:
    instances = build_attack_instances(attacks, eps)
    combos = []
    for mod in mods:
        for snr in snrs:
            for idx in sample_indices:
                for attack, e in instances:
                    eps_suffix = f"_eps{e}" if e is not None else ""
                    for topk in topks:
                        for policy in policies:
                            combo_id = f"{mod}_snr{snr}_idx{idx}_{attack}{eps_suffix}_k{topk}_{policy}"
                            combos.append({
                                "combo_id": combo_id, "modulation": mod, "snr": snr,
                                "sample_index": idx, "attack": attack, "attack_eps": e,
                                "topk": topk, "policy": policy,
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


def precheck_real_backends(awn_adapter, attack_adapter, topk_adapter) -> None:
    problems = []
    if awn_adapter.backend_name != _REAL_MODEL_SOURCE or awn_adapter.status != "ok":
        problems.append(f"AWN: backend={awn_adapter.backend_name!r} status={awn_adapter.status!r}")
    if attack_adapter.backend_name != _REAL_ATTACK_SOURCE or attack_adapter.status != "ok":
        problems.append(f"Attack: backend={attack_adapter.backend_name!r} status={attack_adapter.status!r}")
    if not topk_adapter.backend_available or topk_adapter.backend_name != _REAL_TOPK_SOURCE:
        problems.append(f"Top-K: backend={topk_adapter.backend_name!r} available={topk_adapter.backend_available}")
    if problems:
        raise RuntimeError("Real-backend precheck FAILED:\n" + "\n".join(f"  - {p}" for p in problems))
    print(f"[precheck] real backends confirmed: awn={_REAL_MODEL_SOURCE}, attack={_REAL_ATTACK_SOURCE}, topk={_REAL_TOPK_SOURCE}")


def load_done_combo_ids(summary_path: Path) -> set:
    if not summary_path.exists():
        return set()
    with open(summary_path) as f:
        return {r["combo_id"] for r in csv.DictReader(f)}


class CsvWriter:
    def __init__(self, path: Path, fresh: bool):
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


def run_sample(mod, snr, idx, output_base, awn_adapter, attack_adapter, topk_adapter,
               instances, topks, policies, done_ids, writer) -> None:
    def _cid(attack, eps, topk, policy):
        eps_suffix = f"_eps{eps}" if eps is not None else ""
        return f"{mod}_snr{snr}_idx{idx}_{attack}{eps_suffix}_k{topk}_{policy}"

    sample_ids = {_cid(a, e, k, p) for (a, e) in instances for k in topks for p in policies}
    if sample_ids <= done_ids:
        print(f"[skip] {mod} snr={snr} idx={idx}: all {len(sample_ids)} combos already done")
        return

    output_dir = output_base / f"{mod}_snr{snr}_idx{idx}"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = FIXED["seed"]
    label = RML2016_10A_CLASSES[mod]
    t0 = time.time()

    original_sample = load_radioml_sample(DATASET_PATH, mod, snr, idx)
    iq, embed_meta = embed_sample_in_noise(
        original_sample, n_samples=N_SAMPLES, embed_snr_margin=FIXED["embed_snr_margin"], seed=seed,
    )
    iq = validate_iq(iq)
    original_iq_hash = sha256_array(iq)
    mask = energy_detect(iq, window=FIXED["sensing_window_size"], threshold_factor=FIXED["threshold_factor"])
    raw_regions = mask_to_regions(mask)
    merged_regions = merge_close_regions(raw_regions, merge_gap=FIXED["merge_gap"])

    sensing_failure_stage = sensing_failure_reason = None
    regions = []
    try:
        regions = filter_by_min_length(merged_regions, min_len=FIXED["min_region_len"])
    except RuntimeError as exc:
        sensing_failure_stage, sensing_failure_reason = "filter_by_min_length", str(exc)

    ground_truth = None
    x_clean = None
    seg_start = seg_end = None
    if sensing_failure_stage is None:
        ground_truth = compute_sensing_ground_truth_metrics(embed_meta["true_start"], embed_meta["true_end"], regions)
        try:
            segments, alignment_meta = select_aligned_segments(
                iq, regions, seg_len=FIXED["window_size"], policy=FIXED["alignment_policy"], hop=FIXED["segment_hop"],
            )
            seg_start = alignment_meta[0]["selected_segment_start"]
            seg_end = alignment_meta[0]["selected_segment_end"]
            segments = apply_awn_preprocess(segments, policy=FIXED["awn_preprocess"])
            x_clean = to_awn_input(segments, seg_len=FIXED["window_size"])
        except RuntimeError as exc:
            sensing_failure_stage, sensing_failure_reason = "segment_regions", str(exc)

    if sensing_failure_stage is not None:
        runtime = time.time() - t0
        for attack, eps in instances:
            for topk in topks:
                for policy in policies:
                    combo_id = _cid(attack, eps, topk, policy)
                    if combo_id in done_ids:
                        continue
                    row = _base_row(combo_id, mod, snr, idx, seed, attack, eps, topk, policy, label)
                    row.update({
                        "run_status": "sensing_failed", "failure_stage": sensing_failure_stage,
                        "failure_reason": sensing_failure_reason, "runtime_seconds": runtime,
                        "output_dir": str(output_dir), "original_iq_sha256": original_iq_hash,
                    })
                    writer.write_row(row)
        return

    logits_clean, awn_meta_clean = awn_adapter.infer(x_clean, seed=seed)
    pred_clean = int(np.argmax(logits_clean, axis=1)[0])
    clean_nan = not_finite(x_clean)
    clean_iq_hash = sha256_array(x_clean)

    for attack, eps in instances:
        inst_ids = {_cid(attack, eps, k, p) for k in topks for p in policies}
        if inst_ids <= done_ids:
            continue

        t_att = time.time()
        _seed_everything(seed)
        if attack == "none":
            # dummy_attack's own none-branch is an exact no-op; here we use the
            # SAME real AttackAdapter path as fgsm/pgd/cw for consistency with
            # Phase 0's established finding that real-path 'none' is also an
            # exact bit-identical no-op (round 17).
            x_adv, attack_meta = attack_adapter.apply(
                x_clean, attack="none", eps=0.03, temperature=FIXED["attack_temperature"],
                seed=seed, diagnostics=False, cw_c=FIXED["cw_c"], cw_steps=FIXED["cw_steps"], cw_lr=FIXED["cw_lr"],
            )
        else:
            x_adv, attack_meta = attack_adapter.apply(
                x_clean, attack=attack, eps=(eps if eps is not None else 0.03),
                temperature=FIXED["attack_temperature"], seed=seed, diagnostics=False,
                cw_c=FIXED["cw_c"], cw_steps=FIXED["cw_steps"], cw_lr=FIXED["cw_lr"],
            )
        logits_attacked, awn_meta_attacked = awn_adapter.infer(x_adv, seed=seed)
        pred_attacked = int(np.argmax(logits_attacked, axis=1)[0])
        attacked_iq_hash = sha256_array(x_adv)
        attacked_nan = not_finite(x_adv)
        changed_by_attack = pred_attacked != pred_clean
        attacked_wrong = pred_attacked != label
        iq_linf_ca = float(np.max(np.abs(x_adv - x_clean)))
        attack_training_after = attack_meta.get("attack_training_after")
        eval_mode_restored = (attack_training_after is False) if attack_training_after is not None else None

        seen_hashes = set()
        for topk in topks:
            for policy in policies:
                combo_id = _cid(attack, eps, topk, policy)
                if combo_id in done_ids:
                    continue

                x_defended, topk_meta, policy_notes = apply_policy(x_adv, topk, topk_adapter, policy)
                defended_iq_hash = sha256_array(x_defended)
                recheck_hash = sha256_array(x_adv)
                assert recheck_hash == attacked_iq_hash, (
                    f"attacked IQ mutated for {attack} eps={eps}: {attacked_iq_hash} -> {recheck_hash}"
                )
                seen_hashes.add(recheck_hash)

                logits_defended, awn_meta_defended = awn_adapter.infer(x_defended, seed=seed)
                pred_defended = int(np.argmax(logits_defended, axis=1)[0])
                defended_nan = not_finite(x_defended)
                recovered_by_defense = changed_by_attack and (pred_defended == pred_clean)
                defense_changed_prediction = pred_defended != pred_attacked
                clean_broken_by_defense = (pred_clean == label) and (pred_defended != label)
                iq_linf_ad = float(np.max(np.abs(x_defended - x_adv)))
                iq_l2_ad = float(np.linalg.norm((x_defended - x_adv).reshape(-1)))
                pred_agree = pred_defended == pred_attacked

                awn_ok = all(
                    m["awn_backend"] == _REAL_MODEL_SOURCE and m["awn_status"] == "ok"
                    for m in (awn_meta_clean, awn_meta_attacked, awn_meta_defended)
                )
                attack_ok = attack_meta["attack_backend"] == _REAL_ATTACK_SOURCE and attack_meta["attack_status"] == "ok"
                topk_ok = topk_meta["topk_backend"] == _REAL_TOPK_SOURCE and topk_meta["topk_status"] == "ok"

                run_status, failure_reason = "ok", None
                if not (awn_ok and attack_ok and topk_ok):
                    run_status = "error"
                    failure_reason = f"non-real backend: awn={awn_ok} attack={attack_ok} topk={topk_ok}"
                elif eval_mode_restored is False:
                    run_status = "error"
                    failure_reason = f"eval mode not restored (attack_training_after={attack_training_after})"
                if run_status == "error":
                    print(f"[ERROR] {combo_id}: {failure_reason}")

                row = _base_row(combo_id, mod, snr, idx, seed, attack, eps, topk, policy, label)
                row.update({
                    "pred_clean": pred_clean, "pred_attacked": pred_attacked, "pred_defended": pred_defended,
                    "clean_correct": pred_clean == label, "attacked_correct": pred_attacked == label,
                    "defended_correct": pred_defended == label,
                    "changed_by_attack": changed_by_attack, "attacked_wrong": attacked_wrong,
                    "recovered_by_defense": recovered_by_defense,
                    "defense_changed_prediction": defense_changed_prediction,
                    "clean_broken_by_defense": clean_broken_by_defense,
                    "iq_linf_clean_attacked": iq_linf_ca, "iq_linf_attacked_defended": iq_linf_ad,
                    "iq_l2_attacked_defended": iq_l2_ad, "pred_agreement_defended_vs_attacked": pred_agree,
                    "awn_backend": awn_meta_defended["awn_backend"], "attack_backend": attack_meta["attack_backend"],
                    "topk_backend": topk_meta["topk_backend"],
                    "clean_nan": clean_nan, "attacked_nan": attacked_nan, "defended_nan": defended_nan,
                    "attack_training_after": attack_training_after, "eval_mode_restored": eval_mode_restored,
                    "runtime_seconds": time.time() - t_att,
                    "run_status": run_status, "failure_stage": None, "failure_reason": failure_reason,
                    "output_dir": str(output_dir), "attacked_iq_sha256": attacked_iq_hash,
                    "policy_notes": policy_notes,
                    "original_iq_sha256": original_iq_hash, "clean_iq_sha256": clean_iq_hash,
                    "defended_iq_sha256": defended_iq_hash,
                    "selected_segment_start": seg_start, "selected_segment_end": seg_end,
                })
                writer.write_row(row)

        if len(seen_hashes) != 1:
            raise AssertionError(f"FAIRNESS VIOLATION {mod} snr={snr} idx={idx} attack={attack}: {seen_hashes}")


def _base_row(combo_id, mod, snr, idx, seed, attack, eps, topk, policy, label) -> dict:
    return {
        "combo_id": combo_id, "dataset": "RML2016.10a", "modulation": mod, "snr": snr,
        "sample_index": idx, "seed": seed, "attack": attack, "attack_eps": eps,
        "topk": topk, "policy": policy, "label": label,
    }


def write_manifest(output_dir, combos, instances, args) -> None:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    manifest = {
        "phase": "4_topk_ablation", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repo_commit": commit, "fixed_params": FIXED, "checkpoint": CHECKPOINT,
        "device": DEVICE, "dataset_path": DATASET_PATH,
        "mods": args.mods, "snrs": args.snrs, "sample_indices": args.sample_indices,
        "attacks": args.attacks, "eps": args.eps, "topks": args.topks, "policies": args.policies,
        "attack_instances_per_cell": len(instances), "total_combos": len(combos),
        "combo_id_scheme": "{modulation}_snr{snr}_idx{sample_index}_{attack}[_eps{eps}]_k{topk}_{policy}",
    }
    with open(output_dir / "ablation_manifest.json", "w") as f:
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


def parse_int_list(s): return [int(x) for x in s.split(",")]
def parse_str_list(s): return [x.strip() for x in s.split(",")]
def parse_float_list(s): return [float(x) for x in s.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output-dir", type=str, default="results/formal_phase4_topk_ablation")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-combos", type=int, default=None)
    ap.add_argument("--mods", type=parse_str_list, default=DEFAULT_MODS)
    ap.add_argument("--snrs", type=parse_int_list, default=DEFAULT_SNRS)
    ap.add_argument("--attacks", type=parse_str_list, default=DEFAULT_ATTACKS)
    ap.add_argument("--eps", type=parse_float_list, default=DEFAULT_EPS)
    ap.add_argument("--topks", type=parse_int_list, default=DEFAULT_TOPKS)
    ap.add_argument("--policies", type=parse_str_list, default=DEFAULT_POLICIES)
    ap.add_argument("--n-per-cell", type=int, default=DEFAULT_N_PER_CELL)
    ap.add_argument("--sample-indices", type=parse_int_list, default=None)
    args = ap.parse_args()
    sample_indices = args.sample_indices if args.sample_indices is not None else list(range(args.n_per_cell))
    args.sample_indices = sample_indices

    instances = build_attack_instances(args.attacks, args.eps)
    combos = build_combo_grid(args.mods, args.snrs, sample_indices, args.attacks, args.eps, args.topks, args.policies)
    check_combo_ids_unique(combos)
    n_cells = len(args.mods) * len(args.snrs) * len(sample_indices)
    print(f"[ablation] {len(combos)} final rows: {n_cells} cells x {len(instances)} attack-instances/cell "
          f"x {len(args.topks)} topk x {len(args.policies)} policies")
    print(f"[ablation] attack-instances (pre-expansion): {n_cells * len(instances)}")
    print(f"[ablation] mods={args.mods} snrs={args.snrs} sample_indices={sample_indices} "
          f"attacks={args.attacks} eps={args.eps} topks={args.topks} policies={args.policies}")

    if args.dry_run:
        from collections import Counter
        print(f"[ablation] per-policy row counts: {dict(Counter(c['policy'] for c in combos))}")
        print(f"[ablation] per-topk row counts: {dict(Counter(c['topk'] for c in combos))}")
        print(f"[ablation] per-attack row counts: {dict(Counter(c['attack'] for c in combos))}")
        print(f"[ablation] --dry-run: {len(combos)} combos enumerated, all combo_ids unique. Nothing executed.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "ablation_summary.csv"

    done_ids = load_done_combo_ids(summary_path) if args.resume else set()
    if not args.resume and summary_path.exists():
        raise RuntimeError(f"{summary_path} already exists and --resume was not passed.")
    if done_ids:
        print(f"[resume] {len(done_ids)} combo_ids already done, will be skipped")

    write_manifest(output_dir, combos, instances, args)

    print("[ablation] constructing real AWN/attack/Top-K adapters (once, reused across the whole run)...")
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
            def _cid(attack, eps, topk, policy):
                eps_suffix = f"_eps{eps}" if eps is not None else ""
                return f"{mod}_snr{snr}_idx{idx}_{attack}{eps_suffix}_k{topk}_{policy}"
            sample_ids = {_cid(a, e, k, p) for (a, e) in instances for k in args.topks for p in args.policies}
            if sample_ids <= done_ids:
                continue
            if args.max_combos is not None and attempted >= args.max_combos:
                print(f"[ablation] --max-combos {args.max_combos} reached, stopping")
                break
            run_sample(mod, snr, idx, output_dir, awn_adapter, attack_adapter, topk_adapter,
                       instances, args.topks, args.policies, done_ids, writer)
            attempted += len(sample_ids - done_ids)
            done_ids |= sample_ids
    finally:
        writer.close()

    elapsed = time.time() - t0
    print(f"[ablation] done in {elapsed:.1f}s ({attempted} combos attempted this run)")
    failures_path = output_dir / "ablation_failures.csv"
    n_fail = write_failures_csv(summary_path, failures_path)
    print(f"[ablation] {n_fail} failure row(s) written to {failures_path}" if n_fail else
          f"[ablation] 0 failures -- {failures_path} not written")


if __name__ == "__main__":
    main()
