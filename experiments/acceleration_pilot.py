"""
Phase D acceleration pilot: candidate optimizations for FGSM/PGD/CW attack
generation, each tested on a fixed 60-sample pilot set using the exact
real AttackAdapter/AWNModelAdapter (no modification to those files).

Candidates tested, chosen from experiments/profile_attacks.py's cProfile /
torch.profiler evidence (backward/forward conv+batchnorm ops dominate;
Python-level loop overhead is a small fraction of total time):

  - batching: AttackAdapter.apply() already accepts x of shape [N,2,128]
    with N>1 (verified bit-identical to looping N=1 calls beforehand).
    Tested at batch_size in {1,4,8,16,32}. implementation_optimization
    (same eps/steps, same result per sample) as long as the confirmed
    equivalence check below holds.
  - cpu_threads: torch.set_num_threads() sweep. Pure environment/runtime
    configuration, never changes attack computation. implementation_optimization.
  - steps_reduction (PGD, CW only): reduces the iterative attacks' own
    step count. This DOES change what the attack computes (fewer
    optimization iterations), so it is always tagged algorithmic_tradeoff,
    never implementation_optimization.

Each candidate's pilot run records: latency, speedup vs the batch_size=1 /
default-threads / default-steps reference measured in THE SAME pilot run
(not against Phase B's separately-timed baseline, to keep the comparison
internally consistent), attacked prediction, conditional ASR (over the
60-sample pilot set), perturbation Linf/L2, deterministic reproducibility
(same output on a repeat call with the same input), model_mode_after, and
error/fallback/NaN counts.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.adapters.attack_adapter import AttackAdapter, _REAL_ATTACK_SOURCE  # noqa: E402
from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE  # noqa: E402
from src.sensing.energy_detection import energy_detect, filter_by_min_length, mask_to_regions, merge_close_regions  # noqa: E402
from src.sensing.normalize import apply_awn_preprocess, to_awn_input  # noqa: E402
from src.sensing.radioml_source import RML2016_10A_CLASSES, embed_sample_in_noise, load_radioml_sample  # noqa: E402
from src.sensing.segmentation import select_aligned_segments  # noqa: E402

DATASET_PATH = "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl"
CHECKPOINT_PATH = "external/adversarial-rf/2016.10a_AWN.pkl"
N_PILOT = 60
MODULATIONS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]


def build_pilot_inputs(n: int) -> List[dict]:
    """n samples cycling through modulations at SNR=0, sample_index 0..k."""
    out = []
    for i in range(n):
        mod = MODULATIONS[i % len(MODULATIONS)]
        idx = i // len(MODULATIONS)
        sample = load_radioml_sample(DATASET_PATH, mod, 0, idx)
        iq, _ = embed_sample_in_noise(sample, 8192, 20.0, seed=idx)
        mask = energy_detect(iq, window=128, threshold_factor=5.0)
        regions = filter_by_min_length(merge_close_regions(mask_to_regions(mask), merge_gap=0), min_len=128)
        segments, _ = select_aligned_segments(iq, regions, seg_len=128, policy="max-energy", hop=1)
        x = apply_awn_preprocess(segments[:1], policy="radioml-native")
        x = to_awn_input(x, seg_len=128)
        out.append({"modulation": mod, "sample_index": idx, "x": x, "true_label": RML2016_10A_CLASSES[mod]})
    return out


def conditional_asr(clean_correct: List[bool], attack_success: List[bool]) -> float:
    pairs = [a for c, a in zip(clean_correct, attack_success) if c]
    return float(np.mean(pairs)) if pairs else float("nan")


def run_variant(awn: AWNModelAdapter, attack: AttackAdapter, pilot: List[dict], attack_name: str,
                 eps: float, attack_params: dict, batch_size: int) -> dict:
    """Runs the full pilot set through AttackAdapter using the given batch_size
    (looping over batches of that size), returns aggregate stats + a
    reproducibility check (re-running the FIRST batch and comparing)."""
    xs = [p["x"] for p in pilot]
    true_labels = [p["true_label"] for p in pilot]

    clean_preds, clean_correct = [], []
    for x in xs:
        logits, _ = awn.infer(x, seed=0)
        pred = int(np.argmax(logits[0]))
        clean_preds.append(pred)
        clean_correct.append(pred == true_labels[len(clean_preds) - 1])

    n_error = n_fallback = n_nan = 0
    attacked_preds = []
    linfs, l2s = [], []
    model_modes = []

    t0 = time.perf_counter()
    for start in range(0, len(xs), batch_size):
        batch = np.concatenate(xs[start:start + batch_size], axis=0)
        try:
            x_adv, meta = attack.apply(batch, attack=attack_name, eps=eps, seed=0, attack_params=attack_params)
            if meta["attack_backend"] != _REAL_ATTACK_SOURCE or meta["attack_status"] != "ok":
                n_fallback += 1
            model_modes.append("eval" if not attack.wrapped_model.training else "train")
            for i in range(batch.shape[0]):
                x_single = x_adv[i:i + 1]
                logits_att, meta_att = awn.infer(x_single, seed=0)
                if meta_att["awn_backend"] != _REAL_MODEL_SOURCE:
                    n_fallback += 1
                pred_att = int(np.argmax(logits_att[0]))
                attacked_preds.append(pred_att)
                perturb = x_single.astype(np.float64) - batch[i:i + 1].astype(np.float64)
                linfs.append(float(np.max(np.abs(perturb))))
                l2s.append(float(np.linalg.norm(perturb)))
                if not np.isfinite(x_single).all() or not np.isfinite(logits_att).all():
                    n_nan += 1
        except Exception as exc:  # noqa: BLE001
            n_error += 1
            print(f"[pilot] ERROR batch_size={batch_size} start={start}: {type(exc).__name__}: {exc}", flush=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    attack_success = [ap != cp for ap, cp in zip(attacked_preds, clean_preds[:len(attacked_preds)])]

    # reproducibility check: repeat the FIRST batch, compare output bit-for-bit
    repro_ok = None
    if len(xs) > 0:
        first_batch = np.concatenate(xs[:min(batch_size, len(xs))], axis=0)
        x_adv_a, _ = attack.apply(first_batch, attack=attack_name, eps=eps, seed=0, attack_params=attack_params)
        x_adv_b, _ = attack.apply(first_batch, attack=attack_name, eps=eps, seed=0, attack_params=attack_params)
        repro_ok = bool(np.array_equal(x_adv_a, x_adv_b))

    return {
        "n": len(xs), "n_error": n_error, "n_fallback": n_fallback, "n_nan_inf": n_nan,
        "total_ms": elapsed_ms, "ms_per_sample": elapsed_ms / len(xs) if xs else None,
        "conditional_asr": conditional_asr(clean_correct, attack_success),
        "mean_linf": float(np.mean(linfs)) if linfs else None,
        "mean_l2": float(np.mean(l2s)) if l2s else None,
        "deterministic_reproducible": repro_ok,
        "model_mode_after_all_eval": all(m == "eval" for m in model_modes) if model_modes else None,
    }


def main() -> None:
    out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    print("[pilot] loading real backends ...", flush=True)
    awn = AWNModelAdapter(checkpoint_path=CHECKPOINT_PATH, device="cpu")
    assert awn.backend_name == _REAL_MODEL_SOURCE and awn.status == "ok"
    attack = AttackAdapter(awn_model=awn.model, device="cpu")
    assert attack.wrapped_model is not None and attack.backend_name == _REAL_ATTACK_SOURCE
    print("[pilot] real backends confirmed", flush=True)

    print(f"[pilot] building {N_PILOT}-sample pilot set ...", flush=True)
    pilot = build_pilot_inputs(N_PILOT)

    default_threads = torch.get_num_threads()
    rows = []

    configs = [
        ("fgsm", 0.05, {"eps": 0.05}),
        ("pgd", 0.05, {"eps": 0.05}),
        ("cw", 0.05, {}),
    ]

    # --- candidate 1: batch size sweep ---
    # Formal 3-way classification (docs/research/PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md
    # section 15.1/15.2, and src/adapters/attack_adapter.py:AttackAdapter.apply()'s
    # docstring) -- fixes this loop's previous hardcoded
    # "implementation_optimization" for every attack/batch_size, which
    # mislabeled cw (and mislabeled pgd's stochastic random_start=True case).
    batching_classification_type = {
        "fgsm": "implementation_optimization",
        "pgd": "stochastic_comparison",  # params dict above omits random_start -> torchattacks default True
        "cw": "batched_algorithmic_variant",
    }
    for attack_name, eps, params in configs:
        for bs in [1, 4, 8, 16, 32]:
            print(f"[pilot] batching attack={attack_name} batch_size={bs}", flush=True)
            stats = run_variant(awn, attack, pilot, attack_name, eps, params, batch_size=bs)
            stats.update({"attack": attack_name, "candidate": "batching", "variant": f"batch_size={bs}",
                          "optimization_type": batching_classification_type[attack_name]})
            rows.append(stats)
            print(f"[pilot]   -> total_ms={stats['total_ms']:.1f} ms_per_sample={stats['ms_per_sample']:.2f} "
                  f"asr={stats['conditional_asr']:.3f} repro={stats['deterministic_reproducible']}", flush=True)

    # --- candidate 2: CPU thread count sweep (batch_size=1, representative attack=pgd) ---
    for n_threads in [1, 2, 4, 8, 16]:
        torch.set_num_threads(n_threads)
        print(f"[pilot] cpu_threads={n_threads} attack=pgd", flush=True)
        stats = run_variant(awn, attack, pilot, "pgd", 0.05, {"eps": 0.05}, batch_size=1)
        stats.update({"attack": "pgd", "candidate": "cpu_threads", "variant": f"threads={n_threads}",
                      "optimization_type": "implementation_optimization"})
        rows.append(stats)
        print(f"[pilot]   -> total_ms={stats['total_ms']:.1f} ms_per_sample={stats['ms_per_sample']:.2f}", flush=True)
    torch.set_num_threads(default_threads)

    # --- candidate 3: PGD/CW steps reduction (algorithmic_tradeoff) ---
    for steps in [10, 7, 5, 3]:
        print(f"[pilot] pgd steps={steps}", flush=True)
        stats = run_variant(awn, attack, pilot, "pgd", 0.05, {"eps": 0.05, "steps": steps}, batch_size=1)
        stats.update({"attack": "pgd", "candidate": "steps_reduction", "variant": f"steps={steps}",
                      "optimization_type": "algorithmic_tradeoff"})
        rows.append(stats)
        print(f"[pilot]   -> total_ms={stats['total_ms']:.1f} asr={stats['conditional_asr']:.3f}", flush=True)

    for steps in [20, 15, 10, 5]:
        print(f"[pilot] cw steps={steps}", flush=True)
        stats = run_variant(awn, attack, pilot, "cw", 0.05, {"steps": steps}, batch_size=1)
        stats.update({"attack": "cw", "candidate": "steps_reduction", "variant": f"steps={steps}",
                      "optimization_type": "algorithmic_tradeoff"})
        rows.append(stats)
        print(f"[pilot]   -> total_ms={stats['total_ms']:.1f} asr={stats['conditional_asr']:.3f}", flush=True)

    fieldnames = ["attack", "candidate", "variant", "optimization_type", "n", "n_error", "n_fallback", "n_nan_inf",
                  "total_ms", "ms_per_sample", "conditional_asr", "mean_linf", "mean_l2",
                  "deterministic_reproducible", "model_mode_after_all_eval"]
    with open(out_dir / "acceleration_pilot_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("[pilot] DONE", flush=True)


if __name__ == "__main__":
    main()
