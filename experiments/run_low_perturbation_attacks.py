"""
Formal RadioML -> synthetic long stream -> Spectrum Sensing -> AWN clean/attacked
inference experiment: FGSM baseline + 5 representative low-perturbation attacks
(PGD, CW, DeepFool, EAD, FAB).

Data flow (per base sample):
  RadioML [2,128] sample -> embed_sample_in_noise (real, unmodified,
  src/sensing/radioml_source.py) -> energy_detect -> mask_to_regions ->
  merge_close_regions -> filter_by_min_length -> select_aligned_segments
  (max-energy) -> apply_awn_preprocess -> to_awn_input -> real AWN clean
  inference -> [per attack] real AttackAdapter -> real AWN attacked
  inference -> [reproducibility check] real AWN clean inference again.

Reuses the SAME real backend classes as every other formal script this
session (AWNModelAdapter/AttackAdapter real checkpoint, real torchattacks),
loaded ONCE and reused across the whole run (loading the ~640MB RadioML
pickle or the AWN checkpoint per-sample would be prohibitively slow) --
this is the same "fair-reuse" low-level-building-blocks pattern already
used by experiments/run_phase0_pilot.py and
experiments/run_attack_compatibility_smoke.py, not a new architecture.

Top-K is never invoked in this script (attack effects must not be
confounded with the defense).

Does not modify external/AWN, external/adversarial-rf, or any existing
results/ directory. Does not commit or push.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.adapters.attack_adapter import AttackAdapter, _REAL_ATTACK_SOURCE  # noqa: E402
from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE  # noqa: E402
from src.sensing.energy_detection import energy_detect, filter_by_min_length, mask_to_regions, merge_close_regions  # noqa: E402
from src.sensing.normalize import apply_awn_preprocess, to_awn_input  # noqa: E402
from src.sensing.radioml_source import RML2016_10A_CLASSES, embed_sample_in_noise, radioml_sample_to_iq  # noqa: E402
from src.sensing.segmentation import select_aligned_segments  # noqa: E402
from src.utils.dataset_path import resolve_dataset_path  # noqa: E402

DATASET_PATH = resolve_dataset_path()  # priority: env $SDR_AWN_DATASET_PATH > legacy default
CHECKPOINT_PATH = "external/adversarial-rf/2016.10a_AWN.pkl"
DEVICE = "cpu"

MODULATIONS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
VALID_SNRS = list(range(-20, 20, 2))

# formal, already-validated sensing/embedding defaults (src/utils/config.py's
# own build_arg_parser defaults) -- not new choices invented for this round.
N_SAMPLES = 8192
EMBED_SNR_MARGIN = 20.0
THRESHOLD_FACTOR = 5.0
SENSING_WINDOW_SIZE = 128
MIN_REGION_LEN = 128
MERGE_GAP = 0
ALIGNMENT_POLICY = "max-energy"
AWN_PREPROCESS = "radioml-native"
SEED = 0

RAW_FIELDS = [
    "attack_name", "modulation", "snr", "sample_index", "true_label", "true_label_index",
    "eps",
    "clean_prediction", "clean_confidence", "clean_correct",
    "attacked_prediction", "attacked_confidence", "attacked_correct", "attack_success",
    "perturbation_linf", "perturbation_l2", "perturbation_l1",
    "sensing_detected", "detected_start", "detected_end", "crop_start", "crop_end",
    "captured_signal_ratio", "missed_signal_samples", "true_burst_start", "true_burst_end",
    "clean_segment_input_hash",
    "sensing_ms", "region_postprocess_ms", "segmentation_ms",
    "awn_clean_ms", "attack_generation_ms", "awn_attacked_ms", "total_ms",
    "model_mode_before", "model_mode_after",
    "clean_logits_max_abs_diff", "clean_prediction_reproducible",
    "status", "error_type", "error_message", "fallback_used", "awn_backend", "attack_backend",
    "seed",
    "pgd_alpha", "pgd_steps", "pgd_random_start",
    "cw_c", "cw_kappa", "cw_steps", "cw_lr",
    "deepfool_steps", "deepfool_overshoot",
    "ead_beta", "ead_initial_const", "ead_max_iterations", "ead_lr", "ead_kappa",
    "fab_norm", "fab_steps", "fab_n_restarts",
    "attack_params_json",
]


def sha256_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def softmax_np(logits_1d: np.ndarray) -> np.ndarray:
    z = logits_1d - np.max(logits_1d)
    e = np.exp(z)
    return e / np.sum(e)


class Backends:
    def __init__(self) -> None:
        print(f"[backends] loading AWN checkpoint {CHECKPOINT_PATH} ...", flush=True)
        self.awn = AWNModelAdapter(checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
        if self.awn.backend_name != _REAL_MODEL_SOURCE or self.awn.status != "ok":
            raise RuntimeError(f"Real-AWN precheck FAILED: backend={self.awn.backend_name} status={self.awn.status}")
        print(f"[precheck] real AWN backend confirmed: {_REAL_MODEL_SOURCE}", flush=True)

        self.attack = AttackAdapter(awn_model=self.awn.model, device=DEVICE)
        if self.attack.wrapped_model is None or self.attack.backend_name != _REAL_ATTACK_SOURCE:
            raise RuntimeError(f"Real-attack precheck FAILED: backend={self.attack.backend_name}")
        print(f"[precheck] real attack backend confirmed: {_REAL_ATTACK_SOURCE}", flush=True)

        print(f"[dataset] loading RadioML dict {DATASET_PATH} (one-time, ~640MB) ...", flush=True)
        t0 = time.perf_counter()
        import pickle
        with open(DATASET_PATH, "rb") as f:
            self.radioml_dict = pickle.load(f, encoding="latin1")
        print(f"[dataset] loaded in {time.perf_counter() - t0:.1f}s, {len(self.radioml_dict)} (mod,snr) cells", flush=True)


def infer_one(backends: Backends, x: np.ndarray, seed: int = SEED):
    t0 = time.perf_counter()
    logits, meta = backends.awn.infer(x, seed=seed)
    ms = (time.perf_counter() - t0) * 1000.0
    probs = softmax_np(logits[0])
    pred = int(np.argmax(logits[0]))
    conf = float(probs[pred])
    return pred, conf, logits[0], meta, ms


class BaseSampleError(RuntimeError):
    pass


def build_base_sample(backends: Backends, mod: str, snr: int, sample_index: int) -> dict:
    """RadioML sample -> embed -> sensing -> segment -> clean inference.
    Returns everything a per-attack instance needs, including the resolved
    x_clean array and its logits (for reproducibility diffing)."""
    t_total0 = time.perf_counter()
    block = backends.radioml_dict[(mod, snr)]
    if sample_index >= block.shape[0]:
        raise BaseSampleError(f"sample_index {sample_index} out of range for ({mod},{snr})")
    sample_2x128 = block[sample_index].astype(np.float32)
    true_label_index = RML2016_10A_CLASSES[mod]

    iq, embed_meta = embed_sample_in_noise(sample_2x128, N_SAMPLES, EMBED_SNR_MARGIN, seed=SEED + sample_index)
    true_start, true_end = embed_meta["true_start"], embed_meta["true_end"]

    t0 = time.perf_counter()
    mask = energy_detect(iq, window=SENSING_WINDOW_SIZE, threshold_factor=THRESHOLD_FACTOR)
    raw_regions = mask_to_regions(mask)
    sensing_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    merged_regions = merge_close_regions(raw_regions, merge_gap=MERGE_GAP)
    try:
        kept_regions = filter_by_min_length(merged_regions, min_len=MIN_REGION_LEN)
    except RuntimeError:
        kept_regions = []
    region_postprocess_ms = (time.perf_counter() - t0) * 1000.0

    sensing_detected = len(kept_regions) > 0
    detected_start = detected_end = None
    captured_signal_ratio = None
    missed_signal_samples = None
    if kept_regions:
        # pick the region overlapping the true burst most, for ground-truth QC fields
        best = max(kept_regions, key=lambda r: max(0, min(r[1], true_end) - max(r[0], true_start)))
        detected_start, detected_end = int(best[0]), int(best[1])
        intersection = max(0, min(detected_end, true_end) - max(detected_start, true_start))
        captured_signal_ratio = intersection / (true_end - true_start)
        missed_signal_samples = (true_end - true_start) - intersection

    t0 = time.perf_counter()
    if not kept_regions:
        raise BaseSampleError(f"no occupied region detected for ({mod},{snr},{sample_index})")
    segments, align_meta = select_aligned_segments(iq, kept_regions, seg_len=128, policy=ALIGNMENT_POLICY, hop=1)
    segmentation_ms = (time.perf_counter() - t0) * 1000.0
    if segments.shape[0] == 0:
        raise BaseSampleError(f"no segment produced for ({mod},{snr},{sample_index})")
    crop_start = int(align_meta[0]["selected_segment_start"])
    crop_end = int(align_meta[0]["selected_segment_end"])

    x_clean = apply_awn_preprocess(segments[:1], policy=AWN_PREPROCESS)
    x_clean = to_awn_input(x_clean, seg_len=128)
    clean_input_hash = sha256_array(x_clean)

    pred_clean, conf_clean, logits_clean, meta_clean, awn_clean_ms = infer_one(backends, x_clean)

    return dict(
        mod=mod, snr=snr, sample_index=sample_index, true_label_index=true_label_index,
        true_start=true_start, true_end=true_end,
        sensing_detected=sensing_detected, detected_start=detected_start, detected_end=detected_end,
        crop_start=crop_start, crop_end=crop_end,
        captured_signal_ratio=captured_signal_ratio, missed_signal_samples=missed_signal_samples,
        x_clean=x_clean, clean_input_hash=clean_input_hash,
        pred_clean=pred_clean, conf_clean=conf_clean, logits_clean=logits_clean,
        awn_backend=meta_clean["awn_backend"],
        sensing_ms=sensing_ms, region_postprocess_ms=region_postprocess_ms,
        segmentation_ms=segmentation_ms, awn_clean_ms=awn_clean_ms,
        base_total_ms=(time.perf_counter() - t_total0) * 1000.0,
    )


def run_attack_instance(backends: Backends, base: dict, attack_name: str, eps: float,
                          attack_params: dict, extra_row_fields: dict) -> dict:
    row = {k: None for k in RAW_FIELDS}
    row.update({
        "attack_name": attack_name, "modulation": base["mod"], "snr": base["snr"],
        "sample_index": base["sample_index"], "true_label": base["mod"],
        "true_label_index": base["true_label_index"], "eps": eps,
        "clean_prediction": base["pred_clean"], "clean_confidence": base["conf_clean"],
        "clean_correct": base["pred_clean"] == base["true_label_index"],
        "sensing_detected": base["sensing_detected"], "detected_start": base["detected_start"],
        "detected_end": base["detected_end"], "crop_start": base["crop_start"], "crop_end": base["crop_end"],
        "captured_signal_ratio": base["captured_signal_ratio"], "missed_signal_samples": base["missed_signal_samples"],
        "true_burst_start": base["true_start"], "true_burst_end": base["true_end"],
        "clean_segment_input_hash": base["clean_input_hash"],
        "sensing_ms": base["sensing_ms"], "region_postprocess_ms": base["region_postprocess_ms"],
        "segmentation_ms": base["segmentation_ms"], "awn_clean_ms": base["awn_clean_ms"],
        "awn_backend": base["awn_backend"], "seed": SEED,
        "attack_params_json": json.dumps(attack_params, default=str),
    })
    row.update(extra_row_fields)
    t_total0 = time.perf_counter()
    try:
        model_mode_before = "train" if backends.attack.wrapped_model.training else "eval"
        row["model_mode_before"] = model_mode_before

        # AttackAdapter.apply()'s `eps` parameter is validated as a required
        # non-negative finite float UNCONDITIONALLY (require_nonneg_finite_float),
        # even for attacks whose _ATTACK_ACCEPTED_PARAMS never includes "eps"
        # (cw/deepfool/ead) -- those attacks never actually use this value
        # (see src/adapters/attack_adapter.py:_build_torchattacks, which only
        # sets kwargs["eps"] when "eps" in _ATTACK_ACCEPTED_PARAMS[attack_name]).
        # A structurally-required placeholder is passed here; the row's own
        # "eps" column (set above from the caller's `eps` argument, which
        # stays None for non-eps-based attacks) is what actually gets reported.
        eps_for_apply = eps if eps is not None else 0.05
        t0 = time.perf_counter()
        x_adv, attack_meta = backends.attack.apply(
            base["x_clean"], attack=attack_name, eps=eps_for_apply, seed=SEED, attack_params=attack_params,
        )
        attack_generation_ms = (time.perf_counter() - t0) * 1000.0
        row["attack_generation_ms"] = attack_generation_ms
        row["attack_backend"] = attack_meta["attack_backend"]
        model_mode_after = "train" if backends.attack.wrapped_model.training else "eval"
        row["model_mode_after"] = model_mode_after

        perturb = x_adv.astype(np.float64) - base["x_clean"].astype(np.float64)
        row["perturbation_linf"] = float(np.max(np.abs(perturb)))
        row["perturbation_l2"] = float(np.linalg.norm(perturb))
        row["perturbation_l1"] = float(np.sum(np.abs(perturb)))

        fallback = (attack_meta["attack_status"] != "ok") or (attack_meta["attack_backend"] != _REAL_ATTACK_SOURCE)

        pred_att, conf_att, logits_att, meta_att, awn_attacked_ms = infer_one(backends, x_adv)
        row["attacked_prediction"] = pred_att
        row["attacked_confidence"] = conf_att
        row["awn_attacked_ms"] = awn_attacked_ms
        row["attacked_correct"] = pred_att == base["true_label_index"]
        row["attack_success"] = pred_att != base["pred_clean"]
        fallback = fallback or (meta_att["awn_backend"] != _REAL_MODEL_SOURCE)

        pred_repro, _c, logits_repro, _m, _ms = infer_one(backends, base["x_clean"])
        row["clean_logits_max_abs_diff"] = float(np.max(np.abs(logits_repro - base["logits_clean"])))
        row["clean_prediction_reproducible"] = (pred_repro == base["pred_clean"])

        has_nan_inf = (not np.isfinite(x_adv).all()) or (not np.isfinite(logits_att).all())
        row["fallback_used"] = fallback
        row["status"] = "ok"
        if has_nan_inf:
            row["status"] = "nan_inf"
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc)
    row["total_ms"] = base["base_total_ms"] + (time.perf_counter() - t_total0) * 1000.0
    return row


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


class ProgressTracker:
    def __init__(self, total: int, label: str) -> None:
        self.total = total
        self.label = label
        self.done = 0
        self.n_error = 0
        self.n_fallback = 0
        self.n_nan = 0
        self.t0 = time.perf_counter()

    def update(self, row: dict) -> None:
        self.done += 1
        if row["status"] == "error":
            self.n_error += 1
        if row["status"] == "nan_inf":
            self.n_nan += 1
        if row.get("fallback_used"):
            self.n_fallback += 1
        elapsed = time.perf_counter() - self.t0
        rate = self.done / elapsed if elapsed > 0 else 0
        eta = (self.total - self.done) / rate if rate > 0 else float("inf")
        if self.done % 10 == 0 or self.done == self.total:
            print(f"[{self.label}] {self.done}/{self.total} error={self.n_error} fallback={self.n_fallback} "
                  f"nan={self.n_nan} elapsed={elapsed:.1f}s rate={rate:.2f}/s ETA={eta:.1f}s", flush=True)


def direct_amc(backends: Backends, sample_2x128: np.ndarray):
    """RadioML [2,128] -> AWN directly, NO sensing/embedding -- the
    'direct AMC' path (distinct from the synthetic-long-stream+sensing
    path build_base_sample exercises)."""
    iq = radioml_sample_to_iq(sample_2x128)[np.newaxis, :]  # complex64 [1, 128]
    x = apply_awn_preprocess(iq, policy=AWN_PREPROCESS)
    x = to_awn_input(x, seg_len=128)
    return infer_one(backends, x)


def run_smoke_test(backends: Backends) -> bool:
    print("\n=== SMOKE TEST: AMC/AWN/Sensing/Attack full-package integration ===", flush=True)
    ok = True

    # 4. all 11 modulations loadable; 5. all 20 SNRs selectable
    missing = []
    for mod in MODULATIONS:
        for snr in VALID_SNRS:
            if (mod, snr) not in backends.radioml_dict:
                missing.append((mod, snr))
    print(f"[smoke] 11 modulations x 20 SNRs = 220 cells checked; missing={len(missing)}")
    if missing:
        print(f"[smoke] MISSING CELLS: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        ok = False

    # 3. AWN input shape check via one real base sample
    base_probe = build_base_sample(backends, "QPSK", 0, 0)
    shape_ok = base_probe["x_clean"].shape == (1, 2, 128)
    print(f"[smoke] AWN input shape check: {base_probe['x_clean'].shape} == (1,2,128): {shape_ok}")
    ok = ok and shape_ok

    # 9. 17-attack registry still available
    from src.adapters.attack_adapter import _ATTACK_ACCEPTED_PARAMS
    n_attacks = len(_ATTACK_ACCEPTED_PARAMS)
    print(f"[smoke] attack registry size: {n_attacks} (expect 17)")
    ok = ok and (n_attacks == 17)

    smoke_mods = ["BPSK", "QPSK", "QAM16", "WBFM"]
    smoke_snrs = [-10, 0, 18]
    rows = []
    n_error = n_fallback = n_nan = 0
    for mod in smoke_mods:
        for snr in smoke_snrs:
            for sample_index in [0, 1]:
                # 6. RadioML direct AMC
                pred_d, conf_d, logits_d, meta_d, ms_d = direct_amc(backends, backends.radioml_dict[(mod, snr)][sample_index].astype(np.float32))
                if meta_d["awn_backend"] != _REAL_MODEL_SOURCE:
                    n_fallback += 1
                if not np.isfinite(logits_d).all():
                    n_nan += 1

                # 7. synthetic long stream + sensing AMC, 8-11: FGSM attack + attacked inference + mode/reproducibility
                try:
                    base = build_base_sample(backends, mod, snr, sample_index)
                except BaseSampleError as exc:
                    print(f"[smoke] WARN base sample failed for ({mod},{snr},{sample_index}): {exc}")
                    n_error += 1
                    continue
                row = run_attack_instance(backends, base, "fgsm", 0.05, {"eps": 0.05}, {})
                rows.append(row)
                if row["status"] == "error":
                    n_error += 1
                    print(f"[smoke] ERROR: {row['error_type']}: {row['error_message']}")
                if row["status"] == "nan_inf":
                    n_nan += 1
                if row.get("fallback_used"):
                    n_fallback += 1

    print(f"[smoke] {len(rows)} FGSM smoke instances: error={n_error} fallback={n_fallback} nan={n_nan}")
    mode_after_ok = all(r["model_mode_after"] == "eval" for r in rows if r["model_mode_after"] is not None)
    repro_ok = all(r["clean_prediction_reproducible"] for r in rows if r["clean_prediction_reproducible"] is not None)
    print(f"[smoke] model_mode_after all eval: {mode_after_ok}")
    print(f"[smoke] clean_prediction_reproducible all True: {repro_ok}")
    ok = ok and (n_error == 0) and (n_fallback == 0) and (n_nan == 0) and mode_after_ok and repro_ok

    print(f"=== SMOKE TEST {'PASS' if ok else 'FAIL'} ===\n", flush=True)
    return ok, rows


def run_fgsm(backends: Backends, mods: List[str], snrs: List[int], samples_per_cell: int,
             eps_list: List[float], label: str) -> List[dict]:
    combos = [(m, s, i) for m in mods for s in snrs for i in range(samples_per_cell)]
    total = len(combos) * len(eps_list)
    tracker = ProgressTracker(total, label)
    rows = []
    for mod, snr, sample_index in combos:
        try:
            base = build_base_sample(backends, mod, snr, sample_index)
        except BaseSampleError as exc:
            for eps in eps_list:
                row = {k: None for k in RAW_FIELDS}
                row.update({"attack_name": "fgsm", "modulation": mod, "snr": snr, "sample_index": sample_index,
                            "eps": eps, "status": "error", "error_type": "BaseSampleError", "error_message": str(exc)})
                rows.append(row)
                tracker.update(row)
            continue
        for eps in eps_list:
            row = run_attack_instance(backends, base, "fgsm", eps, {"eps": eps}, {})
            rows.append(row)
            tracker.update(row)
    return rows


LOW_PERTURBATION_ATTACKS = ["pgd", "cw", "deepfool", "ead", "fab"]


def build_low_perturbation_instances(mods, snrs, samples_per_cell):
    combos = [(m, s, i) for m in mods for s in snrs for i in range(samples_per_cell)]
    instances = []  # (attack_name, eps, attack_params, extra_row_fields)
    for eps in [0.005, 0.01, 0.03]:
        instances.append(("pgd", eps, {"eps": eps}, {"pgd_alpha": None, "pgd_steps": None, "pgd_random_start": None}))
    instances.append(("cw", None, {}, {"cw_c": 1.0, "cw_kappa": 0, "cw_steps": 20, "cw_lr": 0.01}))
    instances.append(("deepfool", None, {}, {"deepfool_steps": None, "deepfool_overshoot": None}))
    instances.append(("ead", None, {}, {"ead_beta": None, "ead_initial_const": None, "ead_max_iterations": None,
                                          "ead_lr": None, "ead_kappa": None}))
    for eps in [0.005, 0.01, 0.03]:
        instances.append(("fab", eps, {"eps": eps}, {"fab_norm": None, "fab_steps": None, "fab_n_restarts": None}))
    return combos, instances


def run_low_perturbation(backends: Backends, mods: List[str], snrs: List[int], samples_per_cell: int,
                          attacks_filter: Optional[List[str]] = None) -> Dict[str, List[dict]]:
    combos, instances = build_low_perturbation_instances(mods, snrs, samples_per_cell)
    if attacks_filter is not None:
        instances = [inst for inst in instances if inst[0] in attacks_filter]
    results_by_attack: Dict[str, List[dict]] = {a: [] for a in LOW_PERTURBATION_ATTACKS}

    for attack_name, eps, attack_params, extra in instances:
        eps_tag = f"_eps{eps}" if eps is not None else ""
        label = f"{attack_name}{eps_tag}"
        tracker = ProgressTracker(len(combos), label)
        for mod, snr, sample_index in combos:
            try:
                base = build_base_sample(backends, mod, snr, sample_index)
            except BaseSampleError as exc:
                row = {k: None for k in RAW_FIELDS}
                row.update({"attack_name": attack_name, "modulation": mod, "snr": snr, "sample_index": sample_index,
                            "eps": eps, "status": "error", "error_type": "BaseSampleError", "error_message": str(exc)})
                results_by_attack[attack_name].append(row)
                tracker.update(row)
                continue
            row = run_attack_instance(backends, base, attack_name, eps, attack_params, extra)
            # actual params used may include library defaults not in attack_params (None means "library default").
            # cw's attack_params IS explicit here (matches existing validated defaults: c=1.0,kappa=0,steps=20,lr=0.01).
            if attack_name == "cw":
                row["attack_params_json"] = json.dumps({"c": 1.0, "kappa": 0, "steps": 20, "lr": 0.01})
            results_by_attack[attack_name].append(row)
            tracker.update(row)
    return results_by_attack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["smoke", "fgsm_pilot", "fgsm_formal", "low_perturbation"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--attacks", type=str, default=None, help="comma-separated subset of pgd,cw,deepfool,ead,fab (low_perturbation mode only)")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backends = Backends()

    if args.mode == "smoke":
        ok, rows = run_smoke_test(backends)
        write_csv(out_dir / "smoke_raw_results.csv", rows, RAW_FIELDS)
        with open(out_dir / "smoke_status.json", "w") as f:
            json.dump({"pass": ok}, f)
        if not ok:
            sys.exit(1)

    elif args.mode == "fgsm_pilot":
        # 50 timed samples: 5 mods-representative x ~ combos x eps sweep, kept small & fast
        mods = ["8PSK", "AM-DSB", "BPSK", "CPFSK", "QAM16"]
        snrs = [-10, 0, 18]
        rows = run_fgsm(backends, mods, snrs, samples_per_cell=1, eps_list=[0.005, 0.01, 0.03, 0.05],
                         label="fgsm_pilot")
        rows = rows[:50] if len(rows) > 50 else rows
        write_csv(out_dir / "fgsm_pilot_raw_results.csv", rows, RAW_FIELDS)
        n_ok = sum(1 for r in rows if r["status"] == "ok")
        total_ms = sum(r["total_ms"] for r in rows if r["total_ms"])
        mean_ms = total_ms / len(rows) if rows else 0
        print(f"[pilot] {len(rows)} rows, {n_ok} ok, mean_total_ms={mean_ms:.2f}, "
              f"eta_1320_instances_s={mean_ms * 1320 / 1000:.1f}")

    elif args.mode == "fgsm_formal":
        rows = run_fgsm(backends, MODULATIONS, [-10, 0, 18], samples_per_cell=10,
                         eps_list=[0.005, 0.01, 0.03, 0.05], label="fgsm_formal")
        write_csv(out_dir / "fgsm_raw_results.csv", rows, RAW_FIELDS)
        n_error = sum(1 for r in rows if r["status"] == "error")
        n_fallback = sum(1 for r in rows if r.get("fallback_used"))
        n_nan = sum(1 for r in rows if r["status"] == "nan_inf")
        print(f"[fgsm_formal] DONE: {len(rows)} rows, error={n_error} fallback={n_fallback} nan={n_nan}")

    elif args.mode == "low_perturbation":
        attacks_filter = args.attacks.split(",") if args.attacks else None
        results = run_low_perturbation(backends, MODULATIONS, [-10, 0, 18], samples_per_cell=5,
                                        attacks_filter=attacks_filter)
        all_rows = []
        for attack_name, rows in results.items():
            if not rows:
                continue
            write_csv(out_dir / f"{attack_name}_raw_results.csv", rows, RAW_FIELDS)
            all_rows.extend(rows)
        write_csv(out_dir / "low_perturbation_raw_results.csv", all_rows, RAW_FIELDS)
        n_error = sum(1 for r in all_rows if r["status"] == "error")
        n_fallback = sum(1 for r in all_rows if r.get("fallback_used"))
        n_nan = sum(1 for r in all_rows if r["status"] == "nan_inf")
        print(f"[low_perturbation] DONE: {len(all_rows)} rows, error={n_error} fallback={n_fallback} nan={n_nan}")


if __name__ == "__main__":
    main()
