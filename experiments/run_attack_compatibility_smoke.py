"""
Attack compatibility smoke test: exercises every attack name newly wired
into src/adapters/attack_adapter.py (plus the original fgsm/pgd/cw) against
the real AWN checkpoint, real torchattacks, CPU only. NOT a scale/accuracy
experiment -- proves each attack constructs, runs, produces valid output,
and correctly restores model state, on a small fixed cross-cut.

Per-attack step/query/restart counts below are deliberately reduced
(--steps 5, --n-queries 50, etc.) purely to keep total smoke-test runtime
bounded on CPU -- they are NOT recommended production attack-strength
settings and are recorded verbatim in each row's parameters_json so this is
never mistaken for a real attack-effectiveness result.

Flow per (attack, modulation, snr, sample_index):
  clean input -> AWN clean inference (before)
    -> attack generation (AttackAdapter.apply)
      -> AWN attacked inference
        -> AWN inference on the ORIGINAL clean input again (after)
  clean logits before/after are compared (max abs diff) as a
  state-restoration correctness check -- attacking must not leave the
  model in any state that changes its own clean-input predictions.

Does not modify external/AWN or external/adversarial-rf. Does not touch any
existing results/ directory or rerun the four-path Spectrum Sensing Utility
Experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adapters.attack_adapter import (  # noqa: E402
    AttackAdapter,
    _ATTACK_TARGETED_SUPPORT,
    _REAL_ATTACK_SOURCE,
    _SUPPORTED_ATTACKS,
)
from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE  # noqa: E402
from src.sensing.normalize import apply_awn_preprocess, to_awn_input  # noqa: E402
from src.sensing.radioml_source import RML2016_10A_CLASSES, load_radioml_dict, radioml_sample_to_iq  # noqa: E402
from src.utils.dataset_path import resolve_dataset_path  # noqa: E402

DATASET_PATH = resolve_dataset_path()  # priority: env $SDR_AWN_DATASET_PATH > legacy default
CHECKPOINT = "external/adversarial-rf/2016.10a_AWN.pkl"
DEVICE = "cpu"
AWN_PREPROCESS = "radioml-native"

ATTACKS = ["fgsm", "bim", "pgd", "mifgsm", "difgsm", "vmifgsm", "vnifgsm", "rfgsm", "tpgd",
           "cw", "deepfool", "fab", "square", "apgd", "apgdt", "autoattack", "ead"]

# Deliberately reduced for smoke-test speed on CPU -- see module docstring.
SMOKE_ATTACK_PARAMS: Dict[str, dict] = {
    "fgsm": {}, "bim": {"steps": 5}, "pgd": {"steps": 5},
    "mifgsm": {"steps": 5}, "difgsm": {"steps": 5},
    "vmifgsm": {"steps": 5, "N": 2}, "vnifgsm": {"steps": 5, "N": 2},
    "rfgsm": {"steps": 5}, "tpgd": {"steps": 5},
    "cw": {}, "deepfool": {"steps": 10},
    "fab": {"steps": 5, "n_restarts": 1}, "square": {"n_queries": 50, "n_restarts": 1},
    "apgd": {"steps": 5, "n_restarts": 1}, "apgdt": {"steps": 5, "n_restarts": 1},
    "autoattack": {"version": "rand"}, "ead": {"max_iterations": 10, "binary_search_steps": 2},
}

RAW_FIELDS = [
    "attack", "status", "error_type", "error_message",
    "modulation", "snr", "sample_index", "true_label",
    "clean_prediction_before", "clean_prediction_after", "clean_logits_max_abs_diff",
    "attacked_prediction", "clean_correct", "attacked_correct", "attack_success",
    "input_shape", "output_shape", "dtype", "device", "has_nan", "has_inf",
    "perturbation_linf", "perturbation_l2", "runtime_ms",
    "model_mode_before", "model_mode_during", "model_mode_after",
    "fallback_used", "parameters_json",
]

SUMMARY_FIELDS = [
    "attack", "implemented", "constructed", "tested_samples", "passed_samples", "failed_samples",
    "fallback_count", "nan_count", "mode_restore_pass", "clean_reproducibility_pass",
    "mean_runtime_ms", "mean_linf", "mean_l2", "final_status", "notes",
]


def build_awn_input(sample_2x128: np.ndarray) -> np.ndarray:
    iq = radioml_sample_to_iq(sample_2x128)
    segs = iq[np.newaxis, :].astype(np.complex64)
    segs = apply_awn_preprocess(segs, policy=AWN_PREPROCESS)
    return to_awn_input(segs, seg_len=128)


def softmax_np(logits_1d: np.ndarray) -> np.ndarray:
    z = logits_1d - np.max(logits_1d)
    e = np.exp(z)
    return e / np.sum(e)


def run_one(attack: str, mod: str, snr: int, idx: int, dataset: dict,
            awn_adapter: AWNModelAdapter, eps: float) -> dict:
    row = {k: None for k in RAW_FIELDS}
    row.update({"attack": attack, "modulation": mod, "snr": snr, "sample_index": idx})
    label = RML2016_10A_CLASSES[mod]
    row["true_label"] = label

    params = SMOKE_ATTACK_PARAMS.get(attack, {})
    row["parameters_json"] = json.dumps({"eps": eps, **params})

    t0 = time.perf_counter()
    try:
        block = dataset[(mod, snr)]
        sample = block[idx].astype(np.float32)
        x_clean = build_awn_input(sample)
        row["input_shape"] = str(x_clean.shape)
        row["dtype"] = str(x_clean.dtype)
        row["device"] = DEVICE

        logits_before, meta_before = awn_adapter.infer(x_clean, seed=0)
        pred_before = int(np.argmax(logits_before[0]))
        row["clean_prediction_before"] = pred_before
        row["clean_correct"] = (pred_before == label)

        attack_adapter = AttackAdapter(awn_model=awn_adapter.model, device=DEVICE)
        model_mode_before = attack_adapter.wrapped_model.training if attack_adapter.wrapped_model is not None else None
        row["model_mode_before"] = "train" if model_mode_before else ("eval" if model_mode_before is not None else None)

        x_adv, attack_meta = attack_adapter.apply(
            x_clean, attack=attack, eps=eps, temperature=1.0, seed=0, diagnostics=False,
            attack_params=params,
        )
        # attack_training_before is captured inside apply() right as the real
        # attack starts -- the closest available signal to "model mode while
        # the attack itself was running" without invasive instrumentation of
        # AttackAdapter's internals (see module docstring).
        during_flag = attack_meta.get("attack_training_before")
        row["model_mode_during"] = "train" if during_flag else ("eval" if during_flag is not None else "n/a (dummy fallback)")
        model_mode_after = attack_adapter.wrapped_model.training if attack_adapter.wrapped_model is not None else None
        row["model_mode_after"] = "train" if model_mode_after else ("eval" if model_mode_after is not None else None)

        fallback_used = attack_meta["attack_status"] != "ok" or attack_meta["attack_backend"] != _REAL_ATTACK_SOURCE
        row["fallback_used"] = fallback_used

        row["output_shape"] = str(x_adv.shape)
        row["has_nan"] = bool(np.isnan(x_adv).any())
        row["has_inf"] = bool(np.isinf(x_adv).any())

        diff = x_adv.astype(np.float64) - x_clean.astype(np.float64)
        row["perturbation_linf"] = float(np.max(np.abs(diff)))
        row["perturbation_l2"] = float(np.linalg.norm(diff))

        logits_attacked, _ = awn_adapter.infer(x_adv, seed=0)
        pred_attacked = int(np.argmax(logits_attacked[0]))
        row["attacked_prediction"] = pred_attacked
        row["attacked_correct"] = (pred_attacked == label)
        row["attack_success"] = (pred_attacked != pred_before)

        logits_after, _ = awn_adapter.infer(x_clean, seed=0)
        pred_after = int(np.argmax(logits_after[0]))
        row["clean_prediction_after"] = pred_after
        row["clean_logits_max_abs_diff"] = float(np.max(np.abs(logits_after[0] - logits_before[0])))

        row["status"] = "error" if fallback_used else "ok"
        if fallback_used:
            row["error_type"] = "fallback"
            row["error_message"] = attack_meta.get("attack_notes")
    except Exception as exc:  # noqa: BLE001 - one attack's failure must not abort the others
        row["status"] = "error"
        row["error_type"] = type(exc).__name__
        row["error_message"] = f"{exc}\n{traceback.format_exc(limit=3)}"
        row["fallback_used"] = None
    row["runtime_ms"] = (time.perf_counter() - t0) * 1000.0
    return row


# Attacks known this round to construct successfully but fail at forward()
# for an architectural reason (not an environment/parameter issue) --
# verified by reading the relevant torchattacks source, not guessed from
# the traceback alone. See docs/ATTACK_NAME_MAPPING.md.
_KNOWN_NEEDS_CUSTOM_IMPL = {
    "difgsm": (
        "torchattacks.DIFGSM.input_diversity() unconditionally computes "
        "img_resize=int(x.shape[-1]*resize_rate) on the tensor's LAST dim "
        "before checking diversity_prob. Model01Wrapper reshapes AWN input "
        "to [N,2,T,1] (4D, torchattacks' expected image layout), so "
        "x.shape[-1]==1 always; int(1*resize_rate)==0 for any resize_rate<1, "
        "and resize_rate>=1 makes torch.randint(low=1,high=1,...) raise "
        "'from must be smaller than to' instead. F.interpolate then receives "
        "a 0-sized target and crashes -- true for EVERY resize_rate/"
        "diversity_prob combination, not fixable by tuning those knobs. "
        "Needs a custom input_diversity() override that only resizes the "
        "T=128 axis and leaves the trailing singleton dim untouched, or a "
        "different tensor layout for this attack specifically -- not "
        "attempted this round."
    ),
}


def summarize(attack: str, rows: List[dict]) -> dict:
    supported = attack in _SUPPORTED_ATTACKS
    implemented = supported
    constructed = any(r["status"] in ("ok", "error") and r.get("output_shape") is not None for r in rows)

    if attack in _KNOWN_NEEDS_CUSTOM_IMPL and rows and all(r["status"] != "ok" for r in rows):
        return {
            "attack": attack, "implemented": implemented, "constructed": constructed,
            "tested_samples": len(rows), "passed_samples": 0, "failed_samples": len(rows),
            "fallback_count": sum(1 for r in rows if r.get("fallback_used")),
            "nan_count": sum(1 for r in rows if r.get("has_nan") or r.get("has_inf")),
            "mode_restore_pass": all(r.get("model_mode_after") == "eval" for r in rows if r.get("model_mode_after") is not None),
            "clean_reproducibility_pass": None,
            "mean_runtime_ms": float(np.mean([r["runtime_ms"] for r in rows if r.get("runtime_ms") is not None])) if rows else None,
            "mean_linf": None, "mean_l2": None,
            "final_status": "NEEDS_CUSTOM_IMPLEMENTATION",
            "notes": _KNOWN_NEEDS_CUSTOM_IMPL[attack],
        }
    n = len(rows)
    passed = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] != "ok"]
    fallback_count = sum(1 for r in rows if r.get("fallback_used"))
    nan_count = sum(1 for r in rows if r.get("has_nan") or r.get("has_inf"))
    mode_restore_pass = all(r.get("model_mode_after") == "eval" for r in rows if r.get("model_mode_after") is not None)
    repro_tol = 1e-4
    clean_repro = [r["clean_logits_max_abs_diff"] for r in rows if r.get("clean_logits_max_abs_diff") is not None]
    clean_reproducibility_pass = bool(clean_repro) and all(d <= repro_tol for d in clean_repro)

    runtimes = [r["runtime_ms"] for r in rows if r.get("runtime_ms") is not None]
    linfs = [r["perturbation_linf"] for r in rows if r.get("perturbation_linf") is not None]
    l2s = [r["perturbation_l2"] for r in rows if r.get("perturbation_l2") is not None]

    notes = []
    if not supported:
        final_status = "UNSUPPORTED_BY_INSTALLED_VERSION"
        notes.append("attack name not in AttackAdapter._SUPPORTED_ATTACKS")
    elif n == 0:
        final_status = "FAIL"
        notes.append("no samples tested")
    elif len(passed) < n:
        final_status = "FAIL"
        bad = failed[0]
        notes.append(f"{len(failed)}/{n} samples failed, first: {bad.get('error_type')}: {str(bad.get('error_message'))[:200]}")
    elif fallback_count > 0:
        final_status = "FAIL"
        notes.append(f"{fallback_count} sample(s) silently used the dummy fallback")
    elif nan_count > 0:
        final_status = "FAIL"
        notes.append(f"{nan_count} sample(s) had NaN/Inf in attacked output")
    elif not mode_restore_pass:
        final_status = "FAIL"
        notes.append("model was not left in eval() mode after at least one call")
    elif not clean_reproducibility_pass:
        final_status = "FAIL"
        notes.append(f"clean logits not reproducible after attack (max diff up to {max(clean_repro):.3e} > {repro_tol:.0e})")
    else:
        # zero-perturbation check: a "legitimate failure" (attack ran but
        # produced literally zero perturbation) must be flagged, not
        # silently counted as PASS -- deepfool/cw/ead can legitimately
        # converge to ~0 perturbation for an already-tightly-classified
        # sample, so this is a warning in notes, not an automatic FAIL,
        # UNLESS it happens for every single sample (which would indicate
        # the attack never actually perturbed anything).
        zero_pert = sum(1 for v in linfs if v == 0.0)
        if zero_pert == n:
            final_status = "FAIL"
            notes.append("perturbation was exactly zero for ALL samples -- attack did not perturb the input")
        else:
            final_status = "PASS"
            if zero_pert > 0:
                notes.append(f"{zero_pert}/{n} sample(s) had exactly-zero perturbation (legitimate for some attacks at low budget)")

    return {
        "attack": attack, "implemented": implemented, "constructed": constructed,
        "tested_samples": n, "passed_samples": len(passed), "failed_samples": len(failed),
        "fallback_count": fallback_count, "nan_count": nan_count,
        "mode_restore_pass": mode_restore_pass, "clean_reproducibility_pass": clean_reproducibility_pass,
        "mean_runtime_ms": float(np.mean(runtimes)) if runtimes else None,
        "mean_linf": float(np.mean(linfs)) if linfs else None,
        "mean_l2": float(np.mean(l2s)) if l2s else None,
        "final_status": final_status, "notes": "; ".join(notes) if notes else "",
    }


def build_name_mapping_rows() -> List[dict]:
    from src.adapters.attack_adapter import _ATTACK_ACCEPTED_PARAMS, _ATTACK_CLASS_MAP
    rows = []
    for name in ATTACKS:
        cls = _ATTACK_CLASS_MAP.get(name)
        rows.append({
            "cli_name": name,
            "torchattacks_class": cls.__name__ if cls else None,
            "source": "external/adversarial-rf/util/multi_attack_eval.py or adv_eval.py (see docs/ATTACK_COMPATIBILITY_WORKLIST.md)",
            "accepted_params": ",".join(sorted(_ATTACK_ACCEPTED_PARAMS.get(name, set()))),
            "targeted_support": ",".join(_ATTACK_TARGETED_SUPPORT.get(name, [])),
            "smoke_test_params": json.dumps(SMOKE_ATTACK_PARAMS.get(name, {})),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--mods", type=str, default="QPSK,BPSK")
    ap.add_argument("--snrs", type=str, default="0")
    ap.add_argument("--sample-indices", type=str, default="0,1")
    ap.add_argument("--eps", type=float, default=0.05)
    args = ap.parse_args()

    mods = [m.strip() for m in args.mods.split(",")]
    snrs = [int(s) for s in args.snrs.split(",")]
    idxs = [int(i) for i in args.sample_indices.split(",")]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[precheck] loading AWN checkpoint {CHECKPOINT} ...", flush=True)
    awn_adapter = AWNModelAdapter(checkpoint_path=CHECKPOINT, device=DEVICE)
    if awn_adapter.backend_name != _REAL_MODEL_SOURCE or awn_adapter.status != "ok":
        raise RuntimeError(f"Real-AWN precheck FAILED: backend={awn_adapter.backend_name} status={awn_adapter.status}")
    print(f"[precheck] real AWN backend confirmed: {_REAL_MODEL_SOURCE}", flush=True)

    print(f"[cache] loading RadioML dataset once from {DATASET_PATH} ...", flush=True)
    t0 = time.time()
    dataset = load_radioml_dict(DATASET_PATH)
    print(f"[cache] loaded in {time.time()-t0:.1f}s", flush=True)

    combos = [(a, m, s, i) for a in ATTACKS for m in mods for s in snrs for i in idxs]
    total = len(combos)
    print(f"[smoke] {len(ATTACKS)} attacks x {len(mods)} mods x {len(snrs)} snrs x {len(idxs)} samples = {total} runs")

    raw_path = out_dir / "attack_compatibility_raw.csv"
    raw_f = open(raw_path, "w", newline="")
    raw_writer = csv.DictWriter(raw_f, fieldnames=RAW_FIELDS)
    raw_writer.writeheader()

    all_rows_by_attack: Dict[str, List[dict]] = {a: [] for a in ATTACKS}
    t_start = time.time()
    n_done = 0
    n_error = 0
    n_fallback = 0
    n_nan = 0
    for attack, mod, snr, idx in combos:
        row = run_one(attack, mod, snr, idx, dataset, awn_adapter, args.eps)
        raw_writer.writerow(row)
        raw_f.flush()
        all_rows_by_attack[attack].append(row)

        n_done += 1
        if row["status"] != "ok":
            n_error += 1
        if row.get("fallback_used"):
            n_fallback += 1
        if row.get("has_nan") or row.get("has_inf"):
            n_nan += 1

        elapsed = time.time() - t_start
        rate = n_done / elapsed if elapsed > 0 else 0.0
        eta = (total - n_done) / rate if rate > 0 else float("nan")
        print(f"[smoke] attack={attack:12s} mod={mod:5s} snr={snr:3d} idx={idx} "
              f"status={row['status']:5s} | {n_done}/{total} done elapsed={elapsed:.1f}s "
              f"rate={rate:.2f}/s ETA={eta:.1f}s errors={n_error} fallback={n_fallback} nan={n_nan}",
              flush=True)

    raw_f.close()

    summary_rows = [summarize(a, all_rows_by_attack[a]) for a in ATTACKS]
    with open(out_dir / "attack_compatibility_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    with open(out_dir / "attack_name_mapping.csv", "w", newline="") as f:
        mapping_rows = build_name_mapping_rows()
        w = csv.DictWriter(f, fieldnames=list(mapping_rows[0].keys()))
        w.writeheader()
        for r in mapping_rows:
            w.writerow(r)

    print("\n" + "=" * 100)
    print(f"{'attack':12s} {'final_status':32s} {'passed':7s} {'notes'}")
    print("=" * 100)
    for r in summary_rows:
        print(f"{r['attack']:12s} {r['final_status']:32s} {r['passed_samples']}/{r['tested_samples']:<5d} {r['notes']}")
    print("=" * 100)

    n_pass = sum(1 for r in summary_rows if r["final_status"] == "PASS")
    print(f"\n[smoke] DONE: {n_pass}/{len(ATTACKS)} attacks PASS, {total} total runs in {time.time()-t_start:.1f}s")
    print(f"[smoke] output_dir={out_dir}")


if __name__ == "__main__":
    main()
