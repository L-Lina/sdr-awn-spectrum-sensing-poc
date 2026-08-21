"""
Phase E: formal before/after benchmark. Re-runs the EXACT same 330
base samples per attack (same modulation/SNR/sample_index grid, same seed,
same eps/attack parameters, same AWN checkpoint, same CPU environment) used
by experiments/benchmark_pipeline_latency.py's attack_baseline mode, once
at baseline batch_size=1 and once at the optimized configuration chosen
from experiments/acceleration_pilot.py's results. Writes
attack_acceleration_comparison.csv.
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
from src.sensing.radioml_source import RML2016_10A_CLASSES, embed_sample_in_noise  # noqa: E402
from src.sensing.segmentation import select_aligned_segments  # noqa: E402
from src.utils.dataset_path import resolve_dataset_path  # noqa: E402

DATASET_PATH = resolve_dataset_path()  # priority: env $SDR_AWN_DATASET_PATH > legacy default
CHECKPOINT_PATH = "external/adversarial-rf/2016.10a_AWN.pkl"
MODULATIONS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
ATTACK_SNRS = [-10, 0, 18]
WARMUP = 20


def build_inputs(radioml_dict, combos):
    out = []
    for mod, snr, idx in combos:
        sample = radioml_dict[(mod, snr)][idx].astype(np.float32)
        iq, _ = embed_sample_in_noise(sample, 8192, 20.0, seed=idx)
        mask = energy_detect(iq, window=128, threshold_factor=5.0)
        regions = filter_by_min_length(merge_close_regions(mask_to_regions(mask), merge_gap=0), min_len=128)
        segments, _ = select_aligned_segments(iq, regions, seg_len=128, policy="max-energy", hop=1)
        x = apply_awn_preprocess(segments[:1], policy="radioml-native")
        x = to_awn_input(x, seg_len=128)
        out.append({"modulation": mod, "snr": snr, "sample_index": idx, "x": x, "true_label": RML2016_10A_CLASSES[mod]})
    return out


def run_pass(awn, attack, inputs, attack_name, eps, params, batch_size) -> dict:
    xs = [p["x"] for p in inputs]
    true_labels = [p["true_label"] for p in inputs]
    clean_preds, clean_correct = [], []
    for x in xs:
        logits, _ = awn.infer(x, seed=0)
        pred = int(np.argmax(logits[0]))
        clean_preds.append(pred)
        clean_correct.append(pred == true_labels[len(clean_preds) - 1])

    per_sample_ms = []
    attacked_preds = []
    linfs, l2s = [], []
    n_error = n_fallback = n_nan = 0
    for start in range(0, len(xs), batch_size):
        batch = np.concatenate(xs[start:start + batch_size], axis=0)
        try:
            t0 = time.perf_counter()
            x_adv, meta = attack.apply(batch, attack=attack_name, eps=eps, seed=0, attack_params=params)
            elapsed = (time.perf_counter() - t0) * 1000.0
            per_batch_ms = elapsed / batch.shape[0]
            per_sample_ms.extend([per_batch_ms] * batch.shape[0])
            if meta["attack_backend"] != _REAL_ATTACK_SOURCE or meta["attack_status"] != "ok":
                n_fallback += 1
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
            print(f"[before_after] ERROR: {type(exc).__name__}: {exc}", flush=True)

    attack_success = [ap != cp for ap, cp in zip(attacked_preds, clean_preds[:len(attacked_preds)])]
    pairs = [a for c, a in zip(clean_correct, attack_success) if c]
    conditional_asr = float(np.mean(pairs)) if pairs else float("nan")

    arr = np.asarray(per_sample_ms)
    return {
        "n": len(xs), "n_error": n_error, "n_fallback": n_fallback, "n_nan_inf": n_nan,
        "mean_ms": float(arr.mean()), "median_ms": float(np.median(arr)), "p95_ms": float(np.percentile(arr, 95)),
        "samples_per_sec": float(1000.0 / arr.mean()) if arr.mean() > 0 else None,
        "conditional_asr": conditional_asr,
        "mean_linf": float(np.mean(linfs)), "mean_l2": float(np.mean(l2s)),
        "attacked_preds": attacked_preds,
    }


def main() -> None:
    out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    attacks_arg = sys.argv[sys.argv.index("--attacks") + 1] if "--attacks" in sys.argv else "fgsm,pgd,cw"
    opt_batch_size = int(sys.argv[sys.argv.index("--optimized-batch-size") + 1]) if "--optimized-batch-size" in sys.argv else 8
    opt_threads = int(sys.argv[sys.argv.index("--optimized-threads") + 1]) if "--optimized-threads" in sys.argv else None
    baseline_threads = int(sys.argv[sys.argv.index("--baseline-threads") + 1]) if "--baseline-threads" in sys.argv else None

    import torch
    default_threads = torch.get_num_threads()

    print("[before_after] loading real backends + dataset ...", flush=True)
    awn = AWNModelAdapter(checkpoint_path=CHECKPOINT_PATH, device="cpu")
    assert awn.backend_name == _REAL_MODEL_SOURCE and awn.status == "ok"
    attack = AttackAdapter(awn_model=awn.model, device="cpu")
    assert attack.wrapped_model is not None and attack.backend_name == _REAL_ATTACK_SOURCE

    import pickle
    with open(DATASET_PATH, "rb") as f:
        radioml_dict = pickle.load(f, encoding="latin1")
    print(f"[before_after] dataset loaded, {len(radioml_dict)} cells", flush=True)

    combos = [(m, s, i) for m in MODULATIONS for s in ATTACK_SNRS for i in range(10)]
    print(f"[before_after] {len(combos)} combos per attack (expect 330)", flush=True)
    inputs = build_inputs(radioml_dict, combos)

    # pgd's params dict deliberately omits random_start, so torchattacks'
    # own default (True) applies -- this run therefore measures batching
    # AND pgd's inherent random-start stochasticity together, not batching
    # in isolation (see CLASSIFICATION_NOTES["pgd"] below and
    # experiments/batch_equivalence_audit.py for the isolated,
    # random_start=False deterministic equivalence test).
    configs = {"fgsm": (0.05, {"eps": 0.05}), "pgd": (0.05, {"eps": 0.05}), "cw": (0.05, {})}

    # Formal 3-way classification (docs/research/PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md
    # section 15.1/15.2) -- fixes this script's previous hardcoded
    # "implementation_optimization" for every attack, which mislabeled cw.
    # A. confirmed implementation optimization: batching verified to leave
    #    output bit-identical to per-sample execution.
    # B. stochastic comparison: batching interacts with an attack's own
    #    randomness (pgd's default random_start=True); the measured
    #    difference is not evidence of a batching bug.
    # C. algorithmic variant: batching itself demonstrably changes the
    #    optimization trajectory (cw's batch-level early-stop check).
    CLASSIFICATION_TYPE = {
        "fgsm": "implementation_optimization",
        "pgd": "stochastic_comparison",
        "cw": "batched_algorithmic_variant",
    }
    CLASSIFICATION_NOTES = {
        "fgsm": "Confirmed bit-identical per sample across batch sizes (see acceleration_pilot.py's N=4 vs 4xN=1 check).",
        "pgd": "Batching itself confirmed bit-identical to per-sample execution when random_start=False "
               "(experiments/batch_equivalence_audit.py, 60-sample paired test, max diff 0.0). This particular "
               "330-sample run used torchattacks' default random_start=True, so prediction_match_rate below "
               "reflects PGD's own run-to-run randomness, not a batching defect.",
        "cw": "torchattacks.CW's early-stop check compares the whole-batch-summed cost, so a batch of N stops "
              "together rather than each sample stopping at its own optimum -- confirmed to change the "
              "optimization trajectory (experiments/batch_equivalence_audit.py, 60-sample paired test, 95.0% "
              "prediction match, tensor max diff 0.00138). Must not be reported as pure implementation_optimization.",
    }
    rows = []
    for attack_name in attacks_arg.split(","):
        eps, params = configs[attack_name]

        if baseline_threads is not None:
            torch.set_num_threads(baseline_threads)
        print(f"[before_after] warm-up {attack_name} baseline: {WARMUP} (threads={torch.get_num_threads()})", flush=True)
        for p in inputs[:WARMUP]:
            attack.apply(p["x"], attack=attack_name, eps=eps, seed=0, attack_params=params)
        print(f"[before_after] BASELINE {attack_name} (batch_size=1, threads={torch.get_num_threads()}, n=330)", flush=True)
        baseline = run_pass(awn, attack, inputs, attack_name, eps, params, batch_size=1)
        print(f"[before_after]   baseline mean_ms={baseline['mean_ms']:.2f} median_ms={baseline['median_ms']:.2f} "
              f"asr={baseline['conditional_asr']:.3f}", flush=True)

        if opt_threads is not None:
            torch.set_num_threads(opt_threads)
        print(f"[before_after] warm-up {attack_name} optimized: {WARMUP} (threads={torch.get_num_threads()})", flush=True)
        for start in range(0, WARMUP, opt_batch_size):
            batch = np.concatenate([p["x"] for p in inputs[start:start + opt_batch_size]], axis=0)
            attack.apply(batch, attack=attack_name, eps=eps, seed=0, attack_params=params)
        print(f"[before_after] OPTIMIZED {attack_name} (batch_size={opt_batch_size}, threads={torch.get_num_threads()}, n=330)", flush=True)
        optimized = run_pass(awn, attack, inputs, attack_name, eps, params, batch_size=opt_batch_size)
        print(f"[before_after]   optimized mean_ms={optimized['mean_ms']:.2f} median_ms={optimized['median_ms']:.2f} "
              f"asr={optimized['conditional_asr']:.3f}", flush=True)
        torch.set_num_threads(default_threads)

        prediction_match = float(np.mean([a == b for a, b in zip(baseline["attacked_preds"], optimized["attacked_preds"])]))

        rows.append({
            "attack": attack_name,
            "optimization": f"batching_batch_size_{opt_batch_size}" + (f"_threads_{opt_threads}" if opt_threads is not None else ""),
            "optimization_type": CLASSIFICATION_TYPE[attack_name],
            "classification_note": CLASSIFICATION_NOTES[attack_name],
            "n": baseline["n"],
            "baseline_mean_ms": baseline["mean_ms"], "baseline_median_ms": baseline["median_ms"], "baseline_p95_ms": baseline["p95_ms"],
            "optimized_mean_ms": optimized["mean_ms"], "optimized_median_ms": optimized["median_ms"], "optimized_p95_ms": optimized["p95_ms"],
            "speedup_mean": baseline["mean_ms"] / optimized["mean_ms"] if optimized["mean_ms"] else None,
            "speedup_median": baseline["median_ms"] / optimized["median_ms"] if optimized["median_ms"] else None,
            "baseline_samples_per_sec": baseline["samples_per_sec"], "optimized_samples_per_sec": optimized["samples_per_sec"],
            "conditional_asr_before": baseline["conditional_asr"], "conditional_asr_after": optimized["conditional_asr"],
            "linf_before": baseline["mean_linf"], "linf_after": optimized["mean_linf"],
            "l2_before": baseline["mean_l2"], "l2_after": optimized["mean_l2"],
            "prediction_match_rate": prediction_match,
            "n_error_baseline": baseline["n_error"], "n_error_optimized": optimized["n_error"],
            "n_fallback_baseline": baseline["n_fallback"], "n_fallback_optimized": optimized["n_fallback"],
            "n_nan_inf_baseline": baseline["n_nan_inf"], "n_nan_inf_optimized": optimized["n_nan_inf"],
        })

    fieldnames = ["attack", "optimization", "optimization_type", "classification_note", "n",
                  "baseline_mean_ms", "baseline_median_ms", "baseline_p95_ms",
                  "optimized_mean_ms", "optimized_median_ms", "optimized_p95_ms",
                  "speedup_mean", "speedup_median", "baseline_samples_per_sec", "optimized_samples_per_sec",
                  "conditional_asr_before", "conditional_asr_after", "linf_before", "linf_after",
                  "l2_before", "l2_after", "prediction_match_rate",
                  "n_error_baseline", "n_error_optimized", "n_fallback_baseline", "n_fallback_optimized",
                  "n_nan_inf_baseline", "n_nan_inf_optimized"]
    with open(out_dir / "attack_acceleration_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("[before_after] DONE", flush=True)


if __name__ == "__main__":
    main()
