"""
Performance correctness / stability validation, Part 1 (PGD batching
equivalence) and Part 2 (CW batching semantics audit).

Builds the same 60-sample pilot set used by experiments/acceleration_pilot.py
(imports build_pilot_inputs directly, not re-implemented), then runs each
attack twice on the IDENTICAL samples: baseline (batch_size=1) vs optimized
(batch_size=16), all other params (eps/alpha/steps/model/preprocessing)
held fixed. Compares per-sample: attacked prediction, conditional success,
Linf, L2, attacked-tensor max-abs-diff, clean logits, attacked-logits
max-abs-diff.

PGD is run twice: once with random_start=False (deterministic equivalence
test -- isolates whether BATCHING ITSELF changes PGD's output) and once
with random_start=True (stochastic comparison -- reproduces the originally
observed condition, kept separate so a stochastic difference is never
mislabeled a batching bug).

CW has no random_start; its single baseline-vs-optimized run is inherently
a deterministic equivalence test. Per the task's audit questions, this
script also records whether CW's batch-level early-stop check
(`if cost.item() > prev_cost: return best_adv_images`, confirmed via
inspect.getsource(torchattacks.CW.forward) to use the whole-batch-summed
scalar cost, not a per-sample value) actually fires during these runs, by
re-implementing the same trigger condition instrumentation is not injected
into torchattacks itself (never modified) -- instead we detect its effect
indirectly via the per-sample diff columns below.

Does not modify external/adversarial-rf or external/AWN. Does not retrain
or download any dataset. Uses only the already-real AWNModelAdapter /
AttackAdapter backends (fails closed if the real backend is not loaded).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.acceleration_pilot import N_PILOT, build_pilot_inputs  # noqa: E402
from src.adapters.attack_adapter import AttackAdapter, _REAL_ATTACK_SOURCE  # noqa: E402
from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE  # noqa: E402

CHECKPOINT_PATH = "external/adversarial-rf/2016.10a_AWN.pkl"


def fmt_vec(v: np.ndarray) -> str:
    return ";".join(f"{x:.8g}" for x in v.tolist())


def run_attack_capture(awn: AWNModelAdapter, attack: AttackAdapter, pilot: List[dict],
                        attack_name: str, eps: float, attack_params: dict, batch_size: int,
                        seed: int = 0) -> Dict[str, list]:
    """Runs the pilot set once at the given batch_size, capturing per-sample
    x_adv tensor, attacked logits/pred, and clean logits/pred (clean
    inference is always done one sample at a time via awn.infer, so it is
    itself batch_size-invariant -- included as a control column)."""
    xs = [p["x"] for p in pilot]
    n = len(xs)

    clean_logits: List[np.ndarray] = [None] * n
    clean_preds: List[int] = [None] * n
    for i, x in enumerate(xs):
        logits, _ = awn.infer(x, seed=0)
        clean_logits[i] = logits[0].copy()
        clean_preds[i] = int(np.argmax(logits[0]))

    x_adv_out: List[np.ndarray] = [None] * n
    attacked_logits: List[np.ndarray] = [None] * n
    attacked_preds: List[int] = [None] * n
    n_error = 0

    for start in range(0, n, batch_size):
        batch = np.concatenate(xs[start:start + batch_size], axis=0)
        try:
            x_adv, meta = attack.apply(batch, attack=attack_name, eps=eps, seed=seed, attack_params=attack_params)
            assert meta["attack_backend"] == _REAL_ATTACK_SOURCE and meta["attack_status"] == "ok", meta
        except Exception as exc:  # noqa: BLE001
            n_error += 1
            print(f"[equiv] ERROR attack={attack_name} batch_size={batch_size} start={start}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            continue
        for i in range(batch.shape[0]):
            idx = start + i
            x_single = x_adv[i:i + 1]
            x_adv_out[idx] = x_single.copy()
            logits_att, meta_att = awn.infer(x_single, seed=0)
            assert meta_att["awn_backend"] == _REAL_MODEL_SOURCE
            attacked_logits[idx] = logits_att[0].copy()
            attacked_preds[idx] = int(np.argmax(logits_att[0]))

    return {
        "clean_logits": clean_logits, "clean_preds": clean_preds,
        "x_adv": x_adv_out, "attacked_logits": attacked_logits, "attacked_preds": attacked_preds,
        "n_error": n_error,
    }


def compare_runs(pilot: List[dict], baseline: Dict[str, list], optimized: Dict[str, list]) -> List[dict]:
    rows = []
    for i, p in enumerate(pilot):
        clean_x = p["x"]
        true_label = p["true_label"]

        b_clean_pred = baseline["clean_preds"][i]
        o_clean_pred = optimized["clean_preds"][i]
        clean_pred_match = b_clean_pred == o_clean_pred
        clean_logits_maxabsdiff = float(np.max(np.abs(
            baseline["clean_logits"][i].astype(np.float64) - optimized["clean_logits"][i].astype(np.float64)
        )))
        clean_correct = b_clean_pred == true_label  # clean inference is batch_size-invariant; use baseline's value

        b_adv = baseline["x_adv"][i]
        o_adv = optimized["x_adv"][i]
        row = {
            "sample_id": i,
            "modulation": p["modulation"],
            "sample_index": p["sample_index"],
            "true_label": true_label,
            "clean_pred": b_clean_pred,
            "clean_pred_match_baseline_vs_optimized": clean_pred_match,
            "clean_logits_maxabsdiff_baseline_vs_optimized": clean_logits_maxabsdiff,
            "clean_correct": clean_correct,
        }

        if b_adv is None or o_adv is None:
            row.update({
                "baseline_pred": baseline["attacked_preds"][i], "optimized_pred": optimized["attacked_preds"][i],
                "pred_match": None, "baseline_success": None, "optimized_success": None,
                "baseline_linf": None, "optimized_linf": None, "baseline_l2": None, "optimized_l2": None,
                "tensor_max_abs_diff": None, "baseline_logits": None, "optimized_logits": None,
                "attacked_logits_max_abs_diff": None,
            })
            rows.append(row)
            continue

        b_pred = baseline["attacked_preds"][i]
        o_pred = optimized["attacked_preds"][i]
        b_perturb = b_adv.astype(np.float64) - clean_x.astype(np.float64)
        o_perturb = o_adv.astype(np.float64) - clean_x.astype(np.float64)
        b_linf = float(np.max(np.abs(b_perturb)))
        o_linf = float(np.max(np.abs(o_perturb)))
        b_l2 = float(np.linalg.norm(b_perturb))
        o_l2 = float(np.linalg.norm(o_perturb))
        tensor_diff = float(np.max(np.abs(b_adv.astype(np.float64) - o_adv.astype(np.float64))))
        b_logits = baseline["attacked_logits"][i]
        o_logits = optimized["attacked_logits"][i]
        logits_diff = float(np.max(np.abs(b_logits.astype(np.float64) - o_logits.astype(np.float64))))

        row.update({
            "baseline_pred": b_pred, "optimized_pred": o_pred, "pred_match": b_pred == o_pred,
            "baseline_success": bool(clean_correct and b_pred != b_clean_pred),
            "optimized_success": bool(clean_correct and o_pred != o_clean_pred),
            "baseline_linf": b_linf, "optimized_linf": o_linf,
            "baseline_l2": b_l2, "optimized_l2": o_l2,
            "tensor_max_abs_diff": tensor_diff,
            "baseline_logits": fmt_vec(b_logits), "optimized_logits": fmt_vec(o_logits),
            "attacked_logits_max_abs_diff": logits_diff,
        })
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize(rows: List[dict], label: str) -> None:
    valid = [r for r in rows if r["pred_match"] is not None]
    n = len(valid)
    if n == 0:
        print(f"[equiv] {label}: no valid rows", flush=True)
        return
    pred_match_rate = sum(1 for r in valid if r["pred_match"]) / n
    max_tensor_diff = max(r["tensor_max_abs_diff"] for r in valid)
    max_logits_diff = max(r["attacked_logits_max_abs_diff"] for r in valid)
    b_succ = [r["baseline_success"] for r in valid if r["clean_correct"]]
    o_succ = [r["optimized_success"] for r in valid if r["clean_correct"]]
    b_asr = float(np.mean(b_succ)) if b_succ else float("nan")
    o_asr = float(np.mean(o_succ)) if o_succ else float("nan")
    print(f"[equiv] {label}: n={n} pred_match_rate={pred_match_rate:.4f} "
          f"max_tensor_diff={max_tensor_diff:.6g} max_logits_diff={max_logits_diff:.6g} "
          f"baseline_asr={b_asr:.4f} optimized_asr={o_asr:.4f}", flush=True)


def main() -> None:
    out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    opt_batch_size = int(sys.argv[sys.argv.index("--optimized-batch-size") + 1]) if "--optimized-batch-size" in sys.argv else 16

    import torch
    default_threads = torch.get_num_threads()

    print("[equiv] loading real backends ...", flush=True)
    awn = AWNModelAdapter(checkpoint_path=CHECKPOINT_PATH, device="cpu")
    assert awn.backend_name == _REAL_MODEL_SOURCE and awn.status == "ok"
    attack = AttackAdapter(awn_model=awn.model, device="cpu")
    assert attack.wrapped_model is not None and attack.backend_name == _REAL_ATTACK_SOURCE
    print("[equiv] real backends confirmed", flush=True)

    print(f"[equiv] building {N_PILOT}-sample pilot set (same generator as acceleration_pilot.py) ...", flush=True)
    pilot = build_pilot_inputs(N_PILOT)

    eps = 0.05

    fieldnames = [
        "sample_id", "modulation", "sample_index", "true_label", "clean_pred",
        "clean_pred_match_baseline_vs_optimized", "clean_logits_maxabsdiff_baseline_vs_optimized", "clean_correct",
        "baseline_pred", "optimized_pred", "pred_match", "baseline_success", "optimized_success",
        "baseline_linf", "optimized_linf", "baseline_l2", "optimized_l2", "tensor_max_abs_diff",
        "baseline_logits", "optimized_logits", "attacked_logits_max_abs_diff",
    ]

    # ---- Part 1a: PGD deterministic equivalence test (random_start=False) ----
    print("[equiv] === PGD Test A/B: random_start=False (deterministic equivalence) ===", flush=True)
    torch.set_num_threads(default_threads)
    pgd_det_params = {"eps": eps, "random_start": False}
    baseline_pgd_det = run_attack_capture(awn, attack, pilot, "pgd", eps, pgd_det_params, batch_size=1)
    optimized_pgd_det = run_attack_capture(awn, attack, pilot, "pgd", eps, pgd_det_params, batch_size=opt_batch_size)
    rows_pgd_det = compare_runs(pilot, baseline_pgd_det, optimized_pgd_det)
    write_csv(out_dir / "pgd_batch_equivalence_deterministic.csv", rows_pgd_det, fieldnames)
    summarize(rows_pgd_det, "PGD random_start=False (deterministic)")

    # ---- Part 1b: PGD stochastic comparison (random_start=True, the repo default) ----
    print("[equiv] === PGD Test A/B: random_start=True (stochastic comparison) ===", flush=True)
    pgd_stoch_params = {"eps": eps, "random_start": True}
    baseline_pgd_stoch = run_attack_capture(awn, attack, pilot, "pgd", eps, pgd_stoch_params, batch_size=1, seed=0)
    optimized_pgd_stoch = run_attack_capture(awn, attack, pilot, "pgd", eps, pgd_stoch_params, batch_size=opt_batch_size, seed=0)
    rows_pgd_stoch = compare_runs(pilot, baseline_pgd_stoch, optimized_pgd_stoch)
    write_csv(out_dir / "pgd_batch_stochastic_comparison.csv", rows_pgd_stoch, fieldnames)
    summarize(rows_pgd_stoch, "PGD random_start=True (stochastic)")

    # ---- Part 2: CW batch equivalence (CW has no random_start; inherently deterministic) ----
    print("[equiv] === CW Test A/B: batch_size=1 vs batch_size=%d (deterministic equivalence) ===" % opt_batch_size, flush=True)
    cw_params: dict = {}
    baseline_cw = run_attack_capture(awn, attack, pilot, "cw", eps, cw_params, batch_size=1)
    optimized_cw = run_attack_capture(awn, attack, pilot, "cw", eps, cw_params, batch_size=opt_batch_size)
    rows_cw = compare_runs(pilot, baseline_cw, optimized_cw)

    cw_fieldnames = [
        "sample_id", "baseline_pred", "optimized_pred", "pred_match",
        "baseline_success", "optimized_success",
        "baseline_linf", "optimized_linf", "baseline_l2", "optimized_l2",
        "baseline_logits", "optimized_logits", "tensor_max_abs_diff",
    ]
    cw_rows_out = [{k: r[k] for k in cw_fieldnames} for r in rows_cw]
    write_csv(out_dir / "cw_batch_equivalence.csv", cw_rows_out, cw_fieldnames)
    summarize(rows_cw, "CW batch_size=1 vs batch_size=%d" % opt_batch_size)

    print("[equiv] DONE", flush=True)


if __name__ == "__main__":
    main()
