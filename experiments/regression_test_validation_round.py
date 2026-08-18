"""
Minimal regression test for this round's changes: src/utils/config.py
(torch_num_threads), src/utils/pipeline.py (torch.set_num_threads wiring),
src/adapters/attack_adapter.py (docstring-only batching classification
note, no behavior change), experiments/acceleration_before_after.py and
experiments/acceleration_pilot.py (optimization_type classification fix).

Deliberately small -- 4 modulations (BPSK, QPSK, QAM16, WBFM) x 3 SNRs
(-10, 0, 18 dB) x 2 sample indices = 24 fixed samples, reused across
parts A-E below. Does NOT rerun the 2200-sample clean benchmark or any
330-sample attack matrix. All real backends (fails closed if not loaded).

Parts:
  A. Clean pipeline (real AWN + real sensing) on the 24 samples.
  B. FGSM: batch_size=1 vs a small optimized batch, full equivalence.
  C. PGD, random_start=False: batch_size=1 vs optimized batch, deterministic
     equivalence.
  D. PGD, random_start=True (torchattacks' own default): validity checks
     only, no per-sample match required (stochastic).
  E. CW: batch_size=1 baseline vs batch_size>1 variant; confirms both are
     valid AND that the variant is correctly classifiable as
     batched_algorithmic_variant, not pure implementation_optimization.
  F. torch_num_threads config/CLI/manifest round-trip at None/1/2/4.

Run directly:
    python experiments/regression_test_validation_round.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.adapters.attack_adapter import AttackAdapter, _REAL_ATTACK_SOURCE  # noqa: E402
from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE  # noqa: E402
from src.sensing.energy_detection import energy_detect, filter_by_min_length, mask_to_regions, merge_close_regions  # noqa: E402
from src.sensing.normalize import apply_awn_preprocess, to_awn_input  # noqa: E402
from src.sensing.radioml_source import RML2016_10A_CLASSES, embed_sample_in_noise, load_radioml_dict  # noqa: E402
from src.sensing.segmentation import select_aligned_segments  # noqa: E402
from src.utils.config import ExperimentConfig, args_to_config, build_arg_parser, validate_experiment_config  # noqa: E402
from src.utils.pipeline import run_dry_run_experiment  # noqa: E402

DATASET_PATH = "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl"
CHECKPOINT_PATH = "external/adversarial-rf/2016.10a_AWN.pkl"
MODS = ["BPSK", "QPSK", "QAM16", "WBFM"]
SNRS = [-10, 0, 18]
N_PER_CELL = 2

PASS = []
FAIL = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    (PASS if cond else FAIL).append(name)
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))


def build_fixed_24(radioml_dict) -> List[dict]:
    out = []
    for mod in MODS:
        for snr in SNRS:
            for idx in range(N_PER_CELL):
                sample = radioml_dict[(mod, snr)][idx].astype(np.float32)
                iq, _ = embed_sample_in_noise(sample, 8192, 20.0, seed=idx)
                mask = energy_detect(iq, window=128, threshold_factor=5.0)
                regions = filter_by_min_length(merge_close_regions(mask_to_regions(mask), merge_gap=0), min_len=128)
                segments, _ = select_aligned_segments(iq, regions, seg_len=128, policy="max-energy", hop=1)
                x = apply_awn_preprocess(segments[:1], policy="radioml-native")
                x = to_awn_input(x, seg_len=128)
                out.append({"modulation": mod, "snr": snr, "sample_index": idx, "x": x,
                            "true_label": RML2016_10A_CLASSES[mod]})
    return out


def part_a(awn: AWNModelAdapter, samples: List[dict]) -> None:
    print("\n--- Part A: clean pipeline (real AWN + real sensing) ---")
    n_error = n_fallback = n_nan = 0
    for s in samples:
        try:
            logits, meta = awn.infer(s["x"], seed=0)
        except Exception as exc:  # noqa: BLE001
            n_error += 1
            print(f"  ERROR {s['modulation']}/{s['snr']}/{s['sample_index']}: {exc}")
            continue
        if meta["awn_backend"] != _REAL_MODEL_SOURCE or meta["awn_status"] != "ok":
            n_fallback += 1
        if not np.isfinite(logits).all():
            n_nan += 1
    check("Part A: 24/24 samples produced a clean prediction", n_error == 0, f"n_error={n_error}")
    check("Part A: 0 fallback to non-real AWN backend", n_fallback == 0, f"n_fallback={n_fallback}")
    check("Part A: 0 NaN/Inf in clean logits", n_nan == 0, f"n_nan={n_nan}")


def clean_preds(awn: AWNModelAdapter, samples: List[dict]) -> List[int]:
    preds = []
    for s in samples:
        logits, _ = awn.infer(s["x"], seed=0)
        preds.append(int(np.argmax(logits[0])))
    return preds


def run_batched(awn, attack, samples, attack_name, eps, params, batch_size, seed=0):
    xs = [s["x"] for s in samples]
    x_adv_out, attacked_logits_out, attacked_preds_out = [None] * len(xs), [None] * len(xs), [None] * len(xs)
    training_flags = []
    n_error = 0
    for start in range(0, len(xs), batch_size):
        batch = np.concatenate(xs[start:start + batch_size], axis=0)
        try:
            x_adv, meta = attack.apply(batch, attack=attack_name, eps=eps, seed=seed, attack_params=params)
        except Exception as exc:  # noqa: BLE001
            n_error += 1
            print(f"  ERROR batch_size={batch_size} start={start}: {exc}")
            continue
        if meta["attack_backend"] != _REAL_ATTACK_SOURCE or meta["attack_status"] != "ok":
            n_error += 1
        training_flags.append(attack.wrapped_model.training)
        for i in range(batch.shape[0]):
            idx = start + i
            x_single = x_adv[i:i + 1]
            x_adv_out[idx] = x_single.copy()
            logits_att, _ = awn.infer(x_single, seed=0)
            attacked_logits_out[idx] = logits_att[0].copy()
            attacked_preds_out[idx] = int(np.argmax(logits_att[0]))
    return {
        "x_adv": x_adv_out, "attacked_logits": attacked_logits_out, "attacked_preds": attacked_preds_out,
        "n_error": n_error, "model_mode_after_eval": all(not t for t in training_flags) if training_flags else None,
    }


def part_b(awn, attack, samples, c_preds) -> None:
    print("\n--- Part B: FGSM batch_size=1 vs batch_size=8 ---")
    eps = 0.05
    baseline = run_batched(awn, attack, samples, "fgsm", eps, {"eps": eps}, batch_size=1)
    optimized = run_batched(awn, attack, samples, "fgsm", eps, {"eps": eps}, batch_size=8)
    check("Part B: baseline 0 errors", baseline["n_error"] == 0, f"{baseline['n_error']}")
    check("Part B: optimized 0 errors", optimized["n_error"] == 0, f"{optimized['n_error']}")
    check("Part B: model_mode_after=eval (baseline)", baseline["model_mode_after_eval"] is True)
    check("Part B: model_mode_after=eval (optimized)", optimized["model_mode_after_eval"] is True)

    pred_match = all(a == b for a, b in zip(baseline["attacked_preds"], optimized["attacked_preds"]))
    check("Part B: prediction match 100% (24/24)", pred_match)

    max_tensor_diff = max(
        float(np.max(np.abs(b.astype(np.float64) - o.astype(np.float64))))
        for b, o in zip(baseline["x_adv"], optimized["x_adv"])
    )
    check("Part B: attacked tensor max diff == 0.0 (Linf/L2 numerically identical)", max_tensor_diff == 0.0,
          f"max_diff={max_tensor_diff}")

    clean_correct = [cp == s["true_label"] for cp, s in zip(c_preds, samples)]
    b_succ = [ap != cp for ap, cp in zip(baseline["attacked_preds"], c_preds)]
    o_succ = [ap != cp for ap, cp in zip(optimized["attacked_preds"], c_preds)]
    b_asr = [s for s, c in zip(b_succ, clean_correct) if c]
    o_asr = [s for s, c in zip(o_succ, clean_correct) if c]
    b_asr_rate = float(np.mean(b_asr)) if b_asr else float("nan")
    o_asr_rate = float(np.mean(o_asr)) if o_asr else float("nan")
    check("Part B: ASR match between baseline and optimized", b_asr_rate == o_asr_rate,
          f"baseline={b_asr_rate} optimized={o_asr_rate}")


def part_c(awn, attack, samples) -> None:
    print("\n--- Part C: PGD random_start=False, batch_size=1 vs 8 (deterministic equivalence) ---")
    eps = 0.05
    params = {"eps": eps, "random_start": False}
    baseline = run_batched(awn, attack, samples, "pgd", eps, params, batch_size=1)
    optimized = run_batched(awn, attack, samples, "pgd", eps, params, batch_size=8)
    check("Part C: baseline 0 errors", baseline["n_error"] == 0)
    check("Part C: optimized 0 errors", optimized["n_error"] == 0)
    pred_match = all(a == b for a, b in zip(baseline["attacked_preds"], optimized["attacked_preds"]))
    max_tensor_diff = max(
        float(np.max(np.abs(b.astype(np.float64) - o.astype(np.float64))))
        for b, o in zip(baseline["x_adv"], optimized["x_adv"])
    )
    check("Part C: prediction match 100% (deterministic equivalence)", pred_match)
    check("Part C: attacked tensor max diff == 0.0", max_tensor_diff == 0.0, f"max_diff={max_tensor_diff}")


def part_d(awn, attack, samples, c_preds) -> None:
    print("\n--- Part D: PGD random_start=True, batch_size=1 (stochastic, validity only) ---")
    eps = 0.05
    params = {"eps": eps, "random_start": True}
    result = run_batched(awn, attack, samples, "pgd", eps, params, batch_size=1)
    check("Part D: 0 errors", result["n_error"] == 0)
    linfs = [
        float(np.max(np.abs(adv.astype(np.float64) - s["x"].astype(np.float64))))
        for adv, s in zip(result["x_adv"], samples)
    ]
    all_finite = all(np.isfinite(adv).all() for adv in result["x_adv"])
    check("Part D: all attacked tensors finite (no NaN/Inf)", all_finite)
    check("Part D: all Linf perturbations are valid, non-negative, finite",
          all(l >= 0 and np.isfinite(l) for l in linfs))
    clean_correct = [cp == s["true_label"] for cp, s in zip(c_preds, samples)]
    succ = [ap != cp for ap, cp in zip(result["attacked_preds"], c_preds)]
    conditional = [s for s, c in zip(succ, clean_correct) if c]
    asr = float(np.mean(conditional)) if conditional else float("nan")
    check("Part D: ASR is a valid probability in [0, 1]", 0.0 <= asr <= 1.0 or np.isnan(asr), f"asr={asr}")
    print(f"  (informational, not asserted exactly: ASR={asr:.3f}, mean Linf={np.mean(linfs):.6f}; "
          f"this is a stochastic attack, per-sample match is NOT required)")


def part_e(awn, attack, samples) -> None:
    print("\n--- Part E: CW batch_size=1 baseline vs batch_size=8 variant ---")
    eps = 0.05
    baseline = run_batched(awn, attack, samples, "cw", eps, {}, batch_size=1)
    variant = run_batched(awn, attack, samples, "cw", eps, {}, batch_size=8)
    check("Part E: baseline (batch_size=1) 0 errors -- valid", baseline["n_error"] == 0)
    check("Part E: variant (batch_size=8) 0 errors -- valid", variant["n_error"] == 0)
    all_finite_baseline = all(np.isfinite(adv).all() for adv in baseline["x_adv"])
    all_finite_variant = all(np.isfinite(adv).all() for adv in variant["x_adv"])
    check("Part E: baseline attacked tensors all finite", all_finite_baseline)
    check("Part E: variant attacked tensors all finite", all_finite_variant)

    # Metadata classification check: batch_size>1 CW must be labeled
    # batched_algorithmic_variant, matching AttackAdapter.apply()'s docstring
    # and docs/research/PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md section 15.2.
    # There is no runtime flag on the meta dict itself (torchattacks/
    # AttackAdapter don't tag batch-size-dependent semantics), so this is a
    # STATIC classification check against the documented/scripted mapping,
    # not something the summary CSV can get wrong silently: this test reads
    # the SAME classification map now embedded in acceleration_before_after.py
    # and acceleration_pilot.py and asserts cw maps to batched_algorithmic_variant.
    import experiments.acceleration_before_after as aba
    import experiments.acceleration_pilot as apilot

    # aba's CLASSIFICATION_TYPE is a local variable inside main(), not a
    # module-level constant -- re-derive it here by reading the module
    # source's mapping intent directly instead of importing a private name.
    # (Kept intentionally simple: this checks the file text, not behavior,
    # because the mapping is a documentation/labeling concern, not a
    # runtime code path this repo executes automatically.)
    aba_src = Path(aba.__file__).read_text()
    apilot_src = Path(apilot.__file__).read_text()
    check(
        "Part E: acceleration_before_after.py maps cw -> batched_algorithmic_variant",
        '"cw": "batched_algorithmic_variant"' in aba_src,
    )
    check(
        "Part E: acceleration_before_after.py does NOT map cw -> implementation_optimization",
        '"cw": "implementation_optimization"' not in aba_src,
    )
    check(
        "Part E: acceleration_pilot.py maps cw -> batched_algorithmic_variant",
        '"cw": "batched_algorithmic_variant"' in apilot_src,
    )
    check(
        "Part E: AttackAdapter.apply() docstring documents cw as NOT implementation_optimization",
        "NOT an implementation_optimization at N>1" in Path(REPO_ROOT / "src/adapters/attack_adapter.py").read_text(),
    )


def part_f() -> None:
    print("\n--- Part F: torch_num_threads config/CLI/manifest round-trip (None/1/2/4) ---")
    with tempfile.TemporaryDirectory(prefix="regression_f_") as tmp:
        tmp_path = Path(tmp)
        parser = build_arg_parser("test")
        for val in [None, 1, 2, 4]:
            argv = ["--dry-run", "--snr", "0", "--mod", "BPSK", "--attack", "fgsm", "--topk", "10",
                    "--threshold-factor", "1.5", "--burst-len", "600",
                    "--output-dir", str(tmp_path / f"f_{val}")]
            if val is not None:
                argv += ["--torch-threads", str(val)]
            args = parser.parse_args(argv)
            cfg = args_to_config(args)
            check(f"Part F: CLI round-trip torch_num_threads={val}", cfg.torch_num_threads == val)
            try:
                validate_experiment_config(cfg)
                check(f"Part F: validate_experiment_config accepts torch_num_threads={val}", True)
            except Exception as exc:  # noqa: BLE001
                check(f"Part F: validate_experiment_config accepts torch_num_threads={val}", False, str(exc))

        import torch
        default_threads = torch.get_num_threads()
        cfg2 = ExperimentConfig(
            snr=0.0, mod="BPSK", attack="fgsm", topk=10, threshold_factor=1.5,
            window_size=128, min_region_len=0, merge_gap=0, burst_len=600,
            output_dir=str(tmp_path / "f_manifest"), dry_run=True, torch_num_threads=4,
        )
        result = run_dry_run_experiment(cfg2)
        check("Part F: manifest (summary dict) records torch_num_threads=4", result.get("torch_num_threads") == 4)
        check("Part F: manifest (summary dict) records torch_actual_num_threads=4",
              result.get("torch_actual_num_threads") == 4)
        torch.set_num_threads(default_threads)


def main() -> None:
    print("[regression] loading real backends + dataset ...")
    awn = AWNModelAdapter(checkpoint_path=CHECKPOINT_PATH, device="cpu")
    assert awn.backend_name == _REAL_MODEL_SOURCE and awn.status == "ok", "real AWN backend required for this regression test"
    attack = AttackAdapter(awn_model=awn.model, device="cpu")
    assert attack.wrapped_model is not None and attack.backend_name == _REAL_ATTACK_SOURCE, "real attack backend required"
    radioml_dict = load_radioml_dict(DATASET_PATH)

    print("[regression] building fixed 24-sample set (4 mods x 3 SNRs x 2 samples) ...")
    samples = build_fixed_24(radioml_dict)
    assert len(samples) == 24, len(samples)
    c_preds = clean_preds(awn, samples)

    part_a(awn, samples)
    part_b(awn, attack, samples, c_preds)
    part_c(awn, attack, samples)
    part_d(awn, attack, samples, c_preds)
    part_e(awn, attack, samples)
    part_f()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
