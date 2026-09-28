"""
Adaptive-K Phase A: REFERENCE PARITY VALIDATION ONLY (Raspberry Pi 5 / any CPU).

Compares
  golden semantic reference  references/adaptive_k/adaptive_k_v2.py : adaptive_k_v2   (NumPy, float64 inside)
  production implementation  external/adversarial-rf/util/defense.py : adaptive_k_v2_snr_defense (PyTorch)
Neither implementation is edited. No Adaptive-K benchmark, no full attack matrix.

Terminology: the quantity both implementations call "snr" / "snr_est" is a LINEAR spectral
concentration ratio, sum(top-10 PSD bins) / sum(remaining PSD bins). It is called
"spectral_ratio" everywhere in this script and its outputs. Its thresholds 3 and 10 are linear,
not dB.

Modes
  A  NumPy reference (float32 input, float64 internally, float32 output)  vs  PyTorch float32 (production)
  B  NumPy reference (float64 input/output)                              vs  PyTorch float64
     -> separates precision effects from algorithmic differences
  C  PyTorch float32 production: batched call vs N=1 calls, classified with the SAME decision set and
     declared boundaries as A/B (semantic + numerical parity). Batch shape can change FFT rounding, so raw
     bit equality of flatness / spectral_ratio / output bytes / Top-K indices is recorded as a diagnostic,
     not required.

PyTorch decision instrumentation (project-side, inside this script)
  pt_instrumented() calls the pinned private helpers in exactly the public function's order
  (_shared_fft_and_route -> _estimate_snr_from_fft -> t / k_max_per_sample (statements copied from
  the public function) -> _knee_from_fft -> torch.min -> _apply_per_sample_topk). Flatness is
  recomputed from the helper's own FFT tensor with the identical expression and must reproduce the
  helper's routing exactly; the Top-K kept indices are recomputed with the helper's identical
  grouped topk. For EVERY PyTorch call the instrumented output bytes must equal
  adaptive_k_v2_snr_defense(x) bytes -- otherwise the run ABORTS.

Inputs
  1 synthetic edge cases (deterministic; threshold targets reached by bisection, achieved value
    recorded, "target_reached" flag -- never assumed)
  2 all 2,200 formal clean samples (11 mods x 20 RadioML SNR labels x 10), rebuilt and hash-checked
    against formal clean_input_sha256; same domain as the project's fixed Top-K defense
    (unit-average-power IQ, no AWN_All affine normalization)
  3 the validated 88-sample subset x {FGSM, PGD, BIM} eps=0.03 via the validated Phase 1B attack path
    (experiments/bench_attack_overhead_pi5.py:attack_path, default AttackAdapter) under formal seeds;
    FGSM/BIM must equal formal adversarial_sha256; PGD hash recorded

Per-sample classification (modes A and B)
  INVALID                            non-finite / wrong shape or dtype output
  EXACT                              all decisions equal and output bytes equal
  NUMERICALLY_EQUIVALENT             all decisions equal, max|dy| <= 1e-5*max|x| + abs_floor
  OUTPUT_DIFFERENT                   all decisions equal but max|dy| above tolerance
  DECISION_DIFFERENT_NEAR_THRESHOLD  a decision differs AND every differing decision is explained by a
                                     declared margin (float64 values of the input):
                                       wideband  <- |flatness - 0.4| / 0.4 <= margin_rel
                                       k_cap     <- distance of the interpolated cap to a half-integer
                                                    <= kcap_margin, or |ratio-3|/3 or |ratio-10|/10 <= margin_rel
                                       knee      <- |sorted ratio at the knee boundary - 0.05| / 0.05 <= margin_rel
                                       k         <- explained knee or k_cap difference
                                       topk_mask <- (|X|_(K) - |X|_(K+1)) / peak <= margin_rel (tie/boundary)
                                       quant_level (wideband) <- differing level within quant_margin of a
                                                    half-level (float32 vs float64 rounding of the same value)
  DECISION_DIFFERENT_UNEXPLAINED     otherwise
Decisions compared: wideband; for narrowband: k_cap, knee, K, and the Top-K kept-bin sets modulo
conjugate pairs (keeping bin k or T-k yields the same real IFFT output). Wideband samples get no K.

PASS requires: no INVALID (A, B, C); instrumented == public bit-for-bit (enforced by abort); no UNEXPLAINED
in modes B and A; no OUTPUT_DIFFERENT in modes A/B; mode C no UNEXPLAINED; mode C no OUTPUT_DIFFERENT.

Run on the Pi (from the repo root):
    .venv/bin/python experiments/validate_adaptive_k_v2_parity.py --dry-run
    .venv/bin/python experiments/validate_adaptive_k_v2_parity.py
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as _dt
import hashlib
import importlib.util
import inspect
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import bench_attack_accel_pi5 as p1a  # noqa: E402 -- formal context / input rebuild helpers (no torch at import)
import bench_attack_overhead_pi5 as p1b  # noqa: E402 -- validated Phase 1B attack path

REPO_ROOT = p1a.REPO_ROOT
REFERENCE_PATH = REPO_ROOT / "references" / "adaptive_k" / "adaptive_k_v2.py"
DEFENSE_PATH = REPO_ROOT / "external" / "adversarial-rf" / "util" / "defense.py"
EXPECTED_REFERENCE_SHA256 = "f818dd0229ae3a02271a6bc546593d3d5df2857e4ac63a6d2f26bc8ddf81a891"
EXPECTED_DEFENSE_SHA256 = "291a4ae4b5da17c91c52b8db345259b76134799f717975630464c961dfbfc528"
EXPECTED_ADVRF_COMMIT = "ced705ed861bde352841b0b8bd038a6449e0fa4d"
FORMAL_ARRAYS_DIR = "results/cpu_full_matrix_20260921_seedaligned"
ATTACKS = ["fgsm", "pgd", "bim"]
T_LEN = 128
LINEAR_NOTE = "spectral_ratio thresholds 3 and 10 are linear, not dB"

# Algorithm constants; verified at runtime against BOTH the reference module and the PyTorch defaults.
FLATNESS_THRESHOLD = 0.4
QUANT_LEVELS = 32
RATIO_THRESH = 0.05
PILOT_K = 10
SPECTRAL_RATIO_LOW = 3.0    # linear
SPECTRAL_RATIO_HIGH = 10.0  # linear
K_CAP_LOW_RATIO = 35
K_CAP_HIGH_RATIO = 20

CLASSES = ["EXACT", "NUMERICALLY_EQUIVALENT", "DECISION_DIFFERENT_NEAR_THRESHOLD",
           "DECISION_DIFFERENT_UNEXPLAINED", "OUTPUT_DIFFERENT", "INVALID"]
PARITY_FIELDS = [
    "set", "sample_id", "category", "modulation", "snr_label_db", "sample_index", "attack", "mode",
    "ref_flatness", "pt_flatness", "flatness_abs_diff", "ref_wideband", "pt_wideband",
    "ref_spectral_ratio", "pt_spectral_ratio", "spectral_ratio_rel_diff", "ref_k_cap", "pt_k_cap",
    "ref_knee", "pt_knee", "ref_k", "pt_k", "topk_mask_equivalent",
    "input_max_abs", "tolerance", "max_abs_diff", "mean_abs_diff", "rel_l2_diff", "output_bytes_equal",
    "margin_flatness_rel", "margin_ratio3_rel", "margin_ratio10_rel", "margin_kcap_half", "margin_knee_rel",
    "margin_topk_gap_rel", "margin_quant_half", "differing_decisions", "explained_by", "classification"]
BATCH_FIELDS = ["set", "sample_id", "batch_size", "wideband_equal", "k_cap_equal", "knee_equal", "k_equal",
                "batch_k", "n1_k", "topk_semantic_equivalent", "topk_index_differences",
                "topk_differences_conjugate_pairs",
                "flatness_bits_equal", "spectral_ratio_bits_equal", "topk_indices_equal", "output_bytes_equal",
                "input_max_abs", "tolerance", "max_abs_diff",
                "margin_flatness_rel", "margin_ratio3_rel", "margin_ratio10_rel", "margin_kcap_half",
                "margin_knee_rel", "margin_topk_gap_rel", "margin_quant_half",
                "differing_decisions", "explained_by", "classification"]
SYN_EXTRA = ["target_quantity", "target_value", "achieved_value", "target_reached", "construction"]


# --------------------------------------------------------------------------- small helpers (no torch)
def sha256_file(p: Path) -> str:
    return p1a.sha256_file(p)


def sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_csv(path: Path, fields: List[str], rows: List[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_reference():
    spec = importlib.util.spec_from_file_location("adaptive_k_v2_reference", REFERENCE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module level imports numpy only
    return mod


def formal_array_files() -> dict:
    d = REPO_ROOT / FORMAL_ARRAYS_DIR
    if not d.exists():
        return {"dir": FORMAL_ARRAYS_DIR, "exists": False}
    files = [p for p in d.rglob("*") if p.suffix in (".npy", ".npz")]
    return {"dir": FORMAL_ARRAYS_DIR, "exists": True, "n_npy_npz_files": len(files),
            "examples": [p1a.rel(p) for p in files[:10]],
            "note": "informational only; not used by this parity phase"}


def estimate_minutes(n_clean: int = 2200, n_att: int = 264) -> dict:
    return {"load_dataset_and_rebuild_2200_inputs_min": 1.5, "model_and_attack_generation_min": 0.7,
            "parity_modes_A_B_C_min": (n_clean + n_att + 80) * 3 * 0.004 / 60 + 0.5,
            "total_min_approx": 3.5,
            "note": "planning estimate; the run is expected to take a few minutes, far below 60"}


# --------------------------------------------------------------------------- reference-side (float64) quantities
def ref_details(ref, x: np.ndarray, k: Optional[int]) -> dict:
    """Float64 quantities used ONLY for margins and the reference Top-K kept sets (replicating the
    reference's own topk_filter selection rule). The reference output itself comes from ref.adaptive_k_v2."""
    X = np.fft.fft(x.astype(np.float64), n=x.shape[1], axis=1)
    psd = (np.abs(X) ** 2).mean(axis=0)
    sorted_mag = np.sort(np.sqrt(psd))[::-1]
    ratio = sorted_mag / max(sorted_mag[0], 1e-12)
    below = np.nonzero(ratio < RATIO_THRESH)[0]
    kb = int(below[0]) if below.size else len(ratio)
    cand = [ratio[i] for i in (kb - 1, kb) if 0 <= i < len(ratio)]
    knee_margin = min(abs(r - RATIO_THRESH) / RATIO_THRESH for r in cand) if cand else math.inf
    out = {"knee_margin_rel": knee_margin, "kept": None, "topk_gap_rel": math.inf}
    if k is not None:
        T = X.shape[1]
        kk = min(int(k), T)
        kept, gaps = [], []
        for c in range(X.shape[0]):
            mag = np.abs(X[c])
            idx = np.argsort(-mag, kind="stable")[:kk]  # the reference's own rule
            kept.append(np.sort(idx))
            sm = np.sort(mag)[::-1]
            if kk < T:
                gaps.append((sm[kk - 1] - sm[kk]) / max(sm[0], 1e-300))
        out["kept"] = kept
        out["topk_gap_rel"] = min(gaps) if gaps else math.inf
    return out


def conj_class_counts(idx: np.ndarray, T: int) -> np.ndarray:
    cls = np.minimum(idx, (T - idx) % T)
    return np.bincount(cls, minlength=T // 2 + 1)


def kcap_value(ratio: float) -> float:
    t = (ratio - SPECTRAL_RATIO_LOW) / (SPECTRAL_RATIO_HIGH - SPECTRAL_RATIO_LOW + 1e-12)
    t = min(max(t, 0.0), 1.0)
    return K_CAP_LOW_RATIO + t * (K_CAP_HIGH_RATIO - K_CAP_LOW_RATIO)


# --------------------------------------------------------------------------- synthetic cases (numpy only)
def build_synthetic(ref) -> List[dict]:
    T = T_LEN
    n = np.arange(T)
    rng = np.random.default_rng(20260927)
    noise = rng.standard_normal((2, T))
    cases: List[dict] = []

    def tone(f, amp=1.0):
        return np.stack([amp * np.cos(2 * np.pi * f * n / T), amp * np.sin(2 * np.pi * f * n / T)])

    def add(name, cat, x64, tq="", tv="", av="", reached="", constr=""):
        cases.append({"sample_id": name, "category": cat, "x": np.ascontiguousarray(x64.astype(np.float32)),
                      "target_quantity": tq, "target_value": tv, "achieved_value": av,
                      "target_reached": reached, "construction": constr})

    def sf_of(x32):
        return ref.spectral_flatness(np.fft.fft(x32.astype(np.float64), n=T, axis=1))

    def ratio_of(x32):
        X = np.fft.fft(x32.astype(np.float64), n=T, axis=1)
        return ref.estimate_snr((np.abs(X) ** 2).mean(axis=0))

    def bisect(make, metric, target, lo, hi, iters=200):
        """Log-space bisection on a scalar construction parameter; returns (param, achieved) or (None, None)."""
        flo = metric(make(lo)) - target
        fhi = metric(make(hi)) - target
        if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
            return None, None
        best = (lo, flo)
        for _ in range(iters):
            mid = math.sqrt(lo * hi)
            fm = metric(make(mid)) - target
            if abs(fm) < abs(best[1]):
                best = (mid, fm)
            if fm == 0 or hi / lo - 1 < 1e-15:
                break
            if (fm > 0) == (flo > 0):
                lo, flo = mid, fm
            else:
                hi = mid
        return best[0], metric(make(best[0]))

    # 1-6 structural cases
    add("all_zeros", "zeros", np.zeros((2, T)))
    add("constant_nonzero", "constant", np.stack([np.full(T, 0.5), np.full(T, -0.3)]))
    for s in (1e-6, 1e-12, 1e-20, 1e-30):
        add(f"near_zero_noise_{s:g}", "near_zero", noise * s, constr=f"standard normal x {s:g}")
    add("pure_on_bin_tone_f5", "pure_tone", tone(5))
    add("white_noise", "wideband_noise", noise, constr="standard normal (expected wideband branch)")
    add("two_equal_tones_f5_f17", "equal_tones", tone(5) + tone(17))
    add("high_dynamic_range", "high_dynamic_range", tone(9, 1e3) + 1e-3 * noise)

    # 7 flatness near 0.4: x = a * tone + noise, bisection on a
    base_t = tone(6)
    make_sf = lambda a: (a * base_t + noise).astype(np.float32)  # noqa: E731
    for d in (-1e-3, -1e-5, 0.0, 1e-5, 1e-3):
        tgt = FLATNESS_THRESHOLD * (1 + d)
        a, ach = bisect(make_sf, sf_of, tgt, 1e-3, 1e3)
        if a is None:
            add(f"flatness_target_{tgt:.8f}", "near_flatness", noise, "flatness", tgt, "", False, "bracket failed")
        else:
            add(f"flatness_target_{tgt:.8f}", "near_flatness", a * base_t + noise, "flatness", tgt, ach,
                abs(ach - tgt) <= 1e-6 * tgt + 1e-7, f"a={a:.17g} * tone(6) + noise")

    # 8 + 10 spectral ratio near 3 / 10 and K-cap half-integer boundaries: 5 tones + b * noise
    tones5 = sum(tone(f) for f in (3, 11, 19, 29, 41))
    make_r = lambda b: (tones5 + b * noise).astype(np.float32)  # noqa: E731
    targets = []
    for base in (SPECTRAL_RATIO_LOW, SPECTRAL_RATIO_HIGH):
        for d in (-1e-4, -1e-7, 0.0, 1e-7, 1e-4):
            targets.append(("spectral_ratio", base * (1 + d), f"ratio_{base:g}"))
    for kint in (20, 27, 34):  # interpolated cap = kint + 0.5  <=>  ratio = 3 + 7 * t
        t_half = (K_CAP_LOW_RATIO - (kint + 0.5)) / (K_CAP_LOW_RATIO - K_CAP_HIGH_RATIO)
        r_half = SPECTRAL_RATIO_LOW + t_half * (SPECTRAL_RATIO_HIGH - SPECTRAL_RATIO_LOW)
        for d in (-1e-4, -1e-7, 0.0, 1e-7, 1e-4):
            targets.append(("spectral_ratio_for_kcap_half", r_half * (1 + d), f"kcap_{kint}.5"))
    for tq, tgt, tag in targets:
        b, ach = bisect(make_r, ratio_of, tgt, 1e-4, 1e3)
        cat = "near_ratio_threshold" if tq == "spectral_ratio" else "near_kcap_half"
        name = f"{tag}_target_{tgt:.10f}"
        if b is None:
            add(name, cat, tones5, tq, tgt, "", False, "bracket failed")
        else:
            add(name, cat, tones5 + b * noise, tq, tgt, ach, abs(ach - tgt) <= 1e-6 * tgt,
                f"5 tones + b={b:.17g} * noise; achieved k_cap value {kcap_value(ach):.9f}")

    # 9 knee ratio near 0.05: two tones, amplitude ratio r (on-bin, noiseless => sorted ratio = r)
    for d in (-1e-3, -1e-6, 0.0, 1e-6, 1e-3):
        r = RATIO_THRESH * (1 + d)
        x = tone(7) + tone(23, r)
        x32 = x.astype(np.float32)
        X = np.fft.fft(x32.astype(np.float64), axis=1)
        sm = np.sort(np.sqrt((np.abs(X) ** 2).mean(axis=0)))[::-1]
        ach = float(sm[2] / sm[0])
        add(f"knee_ratio_target_{r:.9f}", "near_knee", x, "knee_ratio", r, ach, abs(ach - r) <= 1e-6 * r,
            f"tone(7) + {r:.9g} * tone(23)")

    # 11 Top-K boundary ties: 5 strong tones + 8 equal (or near-equal) weak tones -> K cap cuts the weak group
    strong = sum(tone(f) for f in (3, 7, 11, 19, 23))
    weak_f = [30 + 4 * j for j in range(8)]
    add("topk_tie_equal_weak_tones", "topk_tie", strong + sum(tone(f, 0.1) for f in weak_f),
        constr="5 tones amp 1 + 8 tones amp 0.1 (equal magnitudes at the K boundary)")
    add("topk_near_tie_weak_tones", "topk_tie",
        strong + sum(tone(f, 0.1 * (1 + 1e-4 * j)) for j, f in enumerate(weak_f)),
        constr="5 tones amp 1 + 8 tones amp 0.1*(1+1e-4 j) (near ties)")

    # 12 odd K splitting a conjugate pair: real-valued I/Q rows with geometric tone amplitudes
    for decay in (0.75, 0.8, 0.85):
        amps = [decay ** j for j in range(16)]
        x = sum(tone(2 + 3 * j, a) for j, a in enumerate(amps))
        y, info = ref.adaptive_k_v2(x.astype(np.float32), return_info=True)
        k = info.get("k")
        add(f"odd_k_conjugate_decay_{decay}", "odd_k_conjugate", x, "k_parity", "odd",
            "" if k is None else k, (k is not None and k % 2 == 1 and k < info["knee"]),
            f"16 tones amp {decay}^j; reference K={k}, knee={info.get('knee')}, k_cap={info.get('k_max')}")
    return cases


# --------------------------------------------------------------------------- main run
def run(args) -> None:
    t_start = time.monotonic()
    import torch
    torch.set_num_interop_threads(1)
    torch.set_num_threads(4)

    ref_sha, def_sha = sha256_file(REFERENCE_PATH), sha256_file(DEFENSE_PATH)
    if ref_sha != EXPECTED_REFERENCE_SHA256:
        p1a.abort(f"reference sha256 {ref_sha} != expected {EXPECTED_REFERENCE_SHA256}")
    if def_sha != EXPECTED_DEFENSE_SHA256:
        p1a.abort(f"defense.py sha256 {def_sha} != expected {EXPECTED_DEFENSE_SHA256}")
    commit = p1a.run_cmd(["git", "-C", str(REPO_ROOT / "external" / "adversarial-rf"), "rev-parse", "HEAD"])

    ref = load_reference()
    advrf = str(REPO_ROOT / "external" / "adversarial-rf")
    if advrf not in sys.path:
        sys.path.insert(0, advrf)
    import util.defense as D  # noqa: E402

    # constants must agree in all three places
    sig = inspect.signature(D.adaptive_k_v2_snr_defense).parameters
    consts = {
        "flatness_threshold": (FLATNESS_THRESHOLD, ref.FLATNESS_THRESHOLD, sig["flatness_threshold"].default),
        "quant_levels": (QUANT_LEVELS, ref.QUANT_LEVELS, sig["quant_levels"].default),
        "ratio_thresh": (RATIO_THRESH, ref.RATIO_THRESH, sig["ratio_thresh"].default),
        "pilot_k": (PILOT_K, ref.PILOT_K, sig["pilot_k"].default),
        "spectral_ratio_low (snr_low, linear)": (SPECTRAL_RATIO_LOW, ref.SNR_LOW, sig["snr_low"].default),
        "spectral_ratio_high (snr_high, linear)": (SPECTRAL_RATIO_HIGH, ref.SNR_HIGH, sig["snr_high"].default),
        "k_cap_low_ratio (k_max_low_snr)": (K_CAP_LOW_RATIO, ref.K_MAX_LOW_SNR, sig["k_max_low_snr"].default),
        "k_cap_high_ratio (k_max_high_snr)": (K_CAP_HIGH_RATIO, ref.K_MAX_HIGH_SNR, sig["k_max_high_snr"].default),
    }
    for k, (a, b, c) in consts.items():
        if not (a == b == c):
            p1a.abort(f"constant mismatch {k}: script={a} reference={b} pytorch_default={c}")

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (REPO_ROOT / (args.out_dir or f"results/adaptive_k_parity_{ts}")).resolve()
    for p in p1a.PROTECTED_RESULT_DIRS:
        prot = (REPO_ROOT / p).resolve()
        if out_dir == prot or prot in out_dir.parents or out_dir in prot.parents:
            p1a.abort(f"refusing to write inside/over {p}")
    out_dir.mkdir(parents=True, exist_ok=False)

    # ---------------------------------------------------------------- PyTorch instrumentation
    counters = {"pytorch_calls": 0, "bit_identity_checks_passed": 0}

    def pt_instrumented(x):
        """Exact operation order of adaptive_k_v2_snr_defense, returning per-sample decisions."""
        N, C, T = x.shape
        result, X, is_nb = D._shared_fft_and_route(x, FLATNESS_THRESHOLD, QUANT_LEVELS)
        power = X.abs() ** 2 + 1e-20  # identical expression to _shared_fft_and_route
        log_power = torch.log(power)
        geo_mean = torch.exp(log_power.mean(dim=2))
        arith_mean = power.mean(dim=2)
        flatness = (geo_mean / (arith_mean + 1e-12)).mean(dim=1)
        if not torch.equal(~(flatness > FLATNESS_THRESHOLD), is_nb):
            p1a.abort("recomputed flatness does not reproduce the helper's routing")
        ratio = torch.full((N,), float("nan"), dtype=x.dtype)
        kcap = torch.full((N,), -1, dtype=torch.int64)
        knee_out = torch.full((N,), -1, dtype=torch.int64)
        k_out = torch.full((N,), -1, dtype=torch.int64)
        kept: Dict[int, np.ndarray] = {}
        if is_nb.any():
            X_n = X[is_nb]
            # ---- statements below copied from adaptive_k_v2_snr_defense
            snr_est = D._estimate_snr_from_fft(X_n, PILOT_K)
            t = (snr_est - SPECTRAL_RATIO_LOW) / (SPECTRAL_RATIO_HIGH - SPECTRAL_RATIO_LOW + 1e-12)
            t = t.clamp(0.0, 1.0)
            k_max_per_sample = (K_CAP_LOW_RATIO + t * (K_CAP_HIGH_RATIO - K_CAP_LOW_RATIO)).round().int()
            knee = D._knee_from_fft(X_n, RATIO_THRESH)
            knee_capped = torch.min(knee, k_max_per_sample)
            result[is_nb] = D._apply_per_sample_topk(X_n, knee_capped)
            # ---- instrumentation only (does not touch result)
            nb_idx = torch.nonzero(is_nb).flatten()
            ratio[nb_idx] = snr_est
            kcap[nb_idx] = k_max_per_sample.to(torch.int64)
            knee_out[nb_idx] = knee.to(torch.int64)
            k_out[nb_idx] = knee_capped.to(torch.int64)
            for kv in sorted(set(knee_capped.cpu().tolist())):  # same grouping + topk as the helper
                kmask = (knee_capped == kv)
                X_sub = X_n[kmask]
                _, idx = X_sub.abs().topk(k=min(int(kv), T), dim=2)
                for j, gi in enumerate(nb_idx[kmask].tolist()):
                    kept[gi] = np.sort(idx[j].cpu().numpy(), axis=1)
        public = D.adaptive_k_v2_snr_defense(x)
        counters["pytorch_calls"] += 1
        a = result.detach().cpu().contiguous().numpy()
        b = public.detach().cpu().contiguous().numpy()
        if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
            (out_dir / "ABORT_instrumentation_mismatch.json").write_text(json.dumps(
                {"shape": list(x.shape), "dtype": str(x.dtype), "max_abs_diff": float(np.max(np.abs(a - b)))}))
            p1a.abort("instrumented PyTorch output is NOT bit-identical to adaptive_k_v2_snr_defense(x)")
        counters["bit_identity_checks_passed"] += 1
        return a, {"flatness": flatness.detach().cpu().numpy(), "wideband": (~is_nb).cpu().numpy(),
                   "ratio": ratio.cpu().numpy(), "kcap": kcap.numpy(), "knee": knee_out.numpy(),
                   "k": k_out.numpy(), "kept": kept}

    # ---------------------------------------------------------------- inputs
    p1a.log("[parity] building synthetic cases ...")
    synthetic = build_synthetic(ref)
    ctx = p1a.load_formal_context(args)
    cfg, fman, formal_dir = ctx["cfg"], ctx["manifest"], ctx["formal_dir"]
    all_keys = [(m, s, i) for m in cfg["modulations"] for s in cfg["snrs"] for i in cfg["sample_indices"]]
    if len(all_keys) != 2200:
        p1a.abort(f"formal grid has {len(all_keys)} samples, expected 2200")
    hashes = {"dataset_sha256": p1a.sha256_file(REPO_ROOT / cfg["dataset_path"]),
              "checkpoint_sha256": p1a.sha256_file(REPO_ROOT / cfg["checkpoint_path"])}
    if hashes["dataset_sha256"] != fman.get("dataset_sha256") or hashes["checkpoint_sha256"] != fman.get("checkpoint_sha256"):
        p1a.abort("dataset/checkpoint sha256 differs from formal CPU manifest")
    p1a.log("[parity] rebuilding the 2,200 formal clean inputs (hash-checked) ...")
    base_rows, formal_attack = p1a.load_formal_rows(formal_dir, set(all_keys))
    inputs = p1a.rebuild_clean_inputs(all_keys, base_rows, p1a.resolve_sensing(cfg), REPO_ROOT / cfg["dataset_path"])
    clean = [{"sample_id": f"{k[0]}|{k[1]}|{k[2]}", "category": "clean", "modulation": k[0], "snr_label_db": k[1],
              "sample_index": k[2], "x": np.ascontiguousarray(inputs[k]["x"][0]),
              "formal_sha": base_rows[k]["clean_input_sha256"], "hash_input": inputs[k]["x"]} for k in all_keys]

    p1a.log("[parity] generating the 88-sample FGSM/PGD/BIM eps=0.03 attacked subset ...")
    import src.adapters.attack_adapter as am
    from src.adapters.attack_adapter import AttackAdapter, _REAL_ATTACK_SOURCE
    from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE
    awn = AWNModelAdapter(checkpoint_path=str(REPO_ROOT / cfg["checkpoint_path"]), device="cpu")
    if awn.model is None or awn.backend_name != _REAL_MODEL_SOURCE or awn.status != "ok":
        p1a.abort("AWN backend is not real")
    adapter = AttackAdapter(awn_model=awn.model, device="cpu")  # default: legacy behavior
    if adapter.wrapped_model is None or adapter.backend_name != _REAL_ATTACK_SOURCE:
        p1a.abort("attack backend is not real")
    settings = p1a.resolve_attack_settings(cfg, formal_attack, args.attack_temperature, args.diagnostics)
    attacked = []
    for key in p1a.build_subset(cfg):
        inp = inputs[key]
        for attack in ATTACKS:
            torch.manual_seed(inp["seed"])
            with contextlib.redirect_stdout(io.StringIO()):
                x_adv, _, _, _, _, _ = p1b.attack_path(am, adapter, torch, inp["x"], attack, settings[attack],
                                                       None, False, False)
            fr = formal_attack[key + (attack, settings[attack]["profile"])].get("adversarial_sha256")
            if attack in p1a.DETERMINISTIC_ATTACKS and p1a.sha256_array(x_adv) != fr:
                p1a.abort(f"{attack} {key}: regenerated adversarial sha256 != formal adversarial_sha256")
            attacked.append({"sample_id": f"{key[0]}|{key[1]}|{key[2]}|{attack}", "category": "attacked",
                             "modulation": key[0], "snr_label_db": key[1], "sample_index": key[2], "attack": attack,
                             "x": np.ascontiguousarray(x_adv[0]), "formal_sha": fr, "hash_input": x_adv})

    hash_rows = []
    for setname, items in (("synthetic", synthetic), ("clean", clean), ("attacked", attacked)):
        for it in items:
            h = p1a.sha256_array(it.get("hash_input", it["x"]))
            fs = it.get("formal_sha", "")
            hash_rows.append({"set": setname, "sample_id": it["sample_id"], "modulation": it.get("modulation", ""),
                              "snr_label_db": it.get("snr_label_db", ""), "sample_index": it.get("sample_index", ""),
                              "attack": it.get("attack", ""), "input_sha256": h, "formal_sha256": fs,
                              "matches_formal": (h == fs) if fs else ""})
    write_csv(out_dir / "input_hashes.csv", ["set", "sample_id", "modulation", "snr_label_db", "sample_index",
                                             "attack", "input_sha256", "formal_sha256", "matches_formal"], hash_rows)

    # ---------------------------------------------------------------- comparison
    def explain_diffs(diffs: List[str], m: dict) -> tuple:
        """Shared by modes A, B and C: every differing decision must be explained by a declared boundary.
        m holds float64 margins of the input: flat, r3, r10, kcap, knee, gap, quant."""
        ok_all, explained = True, []
        for dname in diffs:
            why = []
            if dname == "wideband" and m["flat"] <= args.margin_rel:
                why.append("flatness=0.4")
            if dname in ("k_cap", "k"):
                if m["kcap"] <= args.kcap_margin:
                    why.append("k_cap_half_integer")
                if m["r3"] <= args.margin_rel:
                    why.append("spectral_ratio=3(linear)")
                if m["r10"] <= args.margin_rel:
                    why.append("spectral_ratio=10(linear)")
            if dname in ("knee", "k") and m["knee"] <= args.margin_rel:
                why.append("knee_ratio=0.05")
            if dname == "topk_mask" and m["gap"] <= args.margin_rel:
                why.append("topk_boundary_tie")
            if dname == "quant_level" and m["quant"] <= args.quant_margin:
                why.append("quantization_half_level")
            if not why:
                ok_all = False
            explained.append(f"{dname}:{'+'.join(why) if why else 'UNEXPLAINED'}")
        return ok_all, explained

    def compare(items, mode, pt_out, pt_info) -> List[dict]:
        rows = []
        for i, it in enumerate(items):
            x32 = it["x"]
            x_ref_in = x32 if mode == "A" else x32.astype(np.float64)
            y_ref, info = ref.adaptive_k_v2(x_ref_in, return_info=True)
            y_pt = pt_out[i]
            ref_wb, pt_wb = bool(info["wideband"]), bool(pt_info["wideband"][i])
            ref_k = None if ref_wb else int(info["k"])
            det = ref_details(ref, x32, ref_k)
            inp_max = float(np.max(np.abs(x32.astype(np.float64)))) if x32.size else 0.0
            tol = args.rel_tol * inp_max + args.abs_floor
            valid = (y_ref.shape == x_ref_in.shape and y_pt.shape == x_ref_in.shape and y_ref.dtype == y_pt.dtype
                     and np.isfinite(y_ref).all() and np.isfinite(y_pt).all())
            d = np.abs(y_ref.astype(np.float64) - y_pt.astype(np.float64)) if valid else None
            maxd = float(d.max()) if valid else math.inf
            meand = float(d.mean()) if valid else math.inf
            nrm = float(np.linalg.norm(y_ref.astype(np.float64)))
            rel_l2 = float(np.linalg.norm(y_ref.astype(np.float64) - y_pt.astype(np.float64)) / max(nrm, 1e-300)) \
                if valid else math.inf
            bytes_eq = bool(valid and y_ref.tobytes() == y_pt.tobytes())
            r_ratio = None if ref_wb else float(info["snr_est"])
            p_ratio = None if pt_wb else float(pt_info["ratio"][i])
            m = {"flat": abs(float(info["flatness"]) - FLATNESS_THRESHOLD) / FLATNESS_THRESHOLD,
                 "r3": abs(r_ratio - SPECTRAL_RATIO_LOW) / SPECTRAL_RATIO_LOW if r_ratio is not None else math.inf,
                 "r10": abs(r_ratio - SPECTRAL_RATIO_HIGH) / SPECTRAL_RATIO_HIGH if r_ratio is not None else math.inf,
                 "kcap": (abs((kcap_value(r_ratio) % 1.0) - 0.5) if r_ratio is not None else math.inf),
                 "knee": det["knee_margin_rel"], "gap": det["topk_gap_rel"]}
            diffs, mask_eq = [], ""
            m["quant"] = math.inf
            if ref_wb != pt_wb:
                diffs.append("wideband")
            elif ref_wb:
                # wideband: quantization level indices (reference: float64; PyTorch: input dtype, mimicked in numpy)
                x64 = x32.astype(np.float64)
                rng64 = x64.max() - x64.min() + 1e-12
                u64 = (x64 - x64.min()) / rng64 * (QUANT_LEVELS - 1)
                lev_ref = np.round(u64)
                xp = x32 if mode == "A" else x64
                rngp = xp.max() - xp.min() + xp.dtype.type(1e-12)
                lev_pt = np.round((xp - xp.min()) / rngp * xp.dtype.type(QUANT_LEVELS - 1))
                differ = lev_ref != lev_pt.astype(np.float64)
                if differ.any():
                    diffs.append("quant_level")
                    m["quant"] = float(np.min(np.abs((u64[differ] % 1.0) - 0.5)))
            else:
                if int(info["k_max"]) != int(pt_info["kcap"][i]):
                    diffs.append("k_cap")
                if int(info["knee"]) != int(pt_info["knee"][i]):
                    diffs.append("knee")
                if ref_k != int(pt_info["k"][i]):
                    diffs.append("k")
                else:
                    pk = pt_info["kept"].get(i)
                    mask_eq = pk is not None and all(
                        np.array_equal(conj_class_counts(det["kept"][c], x32.shape[1]),
                                       conj_class_counts(pk[c], x32.shape[1])) for c in range(x32.shape[0]))
                    if not mask_eq:
                        diffs.append("topk_mask")
            explained = []
            if not valid:
                cls = "INVALID"
            elif not diffs:
                cls = "EXACT" if bytes_eq else ("NUMERICALLY_EQUIVALENT" if maxd <= tol else "OUTPUT_DIFFERENT")
            else:
                ok_all, explained = explain_diffs(diffs, m)
                cls = "DECISION_DIFFERENT_NEAR_THRESHOLD" if ok_all else "DECISION_DIFFERENT_UNEXPLAINED"
            rows.append({
                "sample_id": it["sample_id"], "category": it.get("category", ""), "modulation": it.get("modulation", ""),
                "snr_label_db": it.get("snr_label_db", ""), "sample_index": it.get("sample_index", ""),
                "attack": it.get("attack", ""), "mode": mode,
                "ref_flatness": float(info["flatness"]), "pt_flatness": float(pt_info["flatness"][i]),
                "flatness_abs_diff": abs(float(info["flatness"]) - float(pt_info["flatness"][i])),
                "ref_wideband": ref_wb, "pt_wideband": pt_wb,
                "ref_spectral_ratio": "" if r_ratio is None else r_ratio, "pt_spectral_ratio": "" if p_ratio is None else p_ratio,
                "spectral_ratio_rel_diff": "" if (r_ratio is None or p_ratio is None)
                else abs(r_ratio - p_ratio) / max(abs(r_ratio), 1e-300),
                "ref_k_cap": "" if ref_wb else int(info["k_max"]), "pt_k_cap": "" if pt_wb else int(pt_info["kcap"][i]),
                "ref_knee": "" if ref_wb else int(info["knee"]), "pt_knee": "" if pt_wb else int(pt_info["knee"][i]),
                "ref_k": "" if ref_wb else ref_k, "pt_k": "" if pt_wb else int(pt_info["k"][i]),
                "topk_mask_equivalent": mask_eq, "input_max_abs": inp_max, "tolerance": tol, "max_abs_diff": maxd,
                "mean_abs_diff": meand, "rel_l2_diff": rel_l2, "output_bytes_equal": bytes_eq,
                "margin_flatness_rel": m["flat"], "margin_ratio3_rel": m["r3"], "margin_ratio10_rel": m["r10"],
                "margin_kcap_half": m["kcap"], "margin_knee_rel": m["knee"], "margin_topk_gap_rel": m["gap"],
                "margin_quant_half": m["quant"],
                "differing_decisions": ";".join(diffs), "explained_by": " | ".join(explained), "classification": cls})
        return rows

    def run_pt(items, dtype):
        xb = torch.from_numpy(np.stack([it["x"] for it in items]).astype(dtype))
        return pt_instrumented(xb)

    sets = {"synthetic": synthetic, "clean": clean, "attacked": attacked}
    results: Dict[str, Dict[str, List[dict]]] = {}
    batch_cache = {}
    for setname, items in sets.items():
        results[setname] = {}
        for mode, dtype in (("A", np.float32), ("B", np.float64)):
            p1a.log(f"[parity] mode {mode} on {setname} ({len(items)} samples) ...")
            out, info = run_pt(items, dtype)
            if mode == "A":
                batch_cache[setname] = (out, info)
            rows = compare(items, mode, out, info)
            for r in rows:
                r["set"] = setname
            results[setname][mode] = rows

    # ---------------------------------------------------------------- mode C: batched vs N=1 (float32 production)
    # Semantic/numerical parity, same decision set and declared boundaries as modes A/B. Raw floating-point
    # bit equality of intermediates/outputs is recorded as a DIAGNOSTIC only (batch shape may change FFT
    # rounding); it is not a PASS requirement.
    def margins_f64(x32: np.ndarray, k: Optional[int]) -> dict:
        """Float64 margins of the input itself, independent of either implementation's branch outcome."""
        X = np.fft.fft(x32.astype(np.float64), n=x32.shape[1], axis=1)
        sf = ref.spectral_flatness(X)
        r64 = ref.estimate_snr((np.abs(X) ** 2).mean(axis=0))
        det = ref_details(ref, x32, k)
        x64 = x32.astype(np.float64)
        u64 = (x64 - x64.min()) / (x64.max() - x64.min() + 1e-12) * (QUANT_LEVELS - 1)
        return {"flat": abs(sf - FLATNESS_THRESHOLD) / FLATNESS_THRESHOLD,
                "r3": abs(r64 - SPECTRAL_RATIO_LOW) / SPECTRAL_RATIO_LOW,
                "r10": abs(r64 - SPECTRAL_RATIO_HIGH) / SPECTRAL_RATIO_HIGH,
                "kcap": abs((kcap_value(r64) % 1.0) - 0.5), "knee": det["knee_margin_rel"],
                "gap": det["topk_gap_rel"], "quant": float(np.min(np.abs((u64 % 1.0) - 0.5)))}

    def index_diff(kb: np.ndarray, k1: np.ndarray, T: int) -> tuple:
        """Per channel: kept bins only in the batched run / only in the N=1 run, and whether every such bin's
        conjugate partner (T - k) sits on the other side (conjugate-equivalent: identical real IFFT output)."""
        parts, conj_all = [], True
        for c in range(kb.shape[0]):
            b_only = sorted(set(kb[c].tolist()) - set(k1[c].tolist()))
            n_only = sorted(set(k1[c].tolist()) - set(kb[c].tolist()))
            conj = sorted(((T - v) % T) for v in b_only) == n_only
            conj_all = conj_all and conj
            if b_only or n_only:
                parts.append(f"ch{c}: batch_only={b_only} n1_only={n_only} conjugate_pairs={conj}")
        return " ; ".join(parts), conj_all

    batch_rows = []
    for setname, items in sets.items():
        out_b, info_b = batch_cache[setname]
        for i, it in enumerate(items):
            out_1, info_1 = run_pt([it], np.float32)
            x32 = it["x"]
            T = x32.shape[1]
            fb = lambda a: np.asarray(a).tobytes()  # noqa: E731
            yb, y1 = out_b[i], out_1[0]
            wb_b, wb_1 = bool(info_b["wideband"][i]), bool(info_1["wideband"][0])
            kb, k1 = info_b["kept"].get(i), info_1["kept"].get(0)
            valid = (yb.shape == y1.shape == x32.shape and yb.dtype == y1.dtype
                     and np.isfinite(yb).all() and np.isfinite(y1).all())
            maxd = float(np.max(np.abs(yb.astype(np.float64) - y1.astype(np.float64)))) if valid else math.inf
            inp_max = float(np.max(np.abs(x32.astype(np.float64))))
            tol = args.rel_tol * inp_max + args.abs_floor
            bytes_eq = bool(valid and yb.tobytes() == y1.tobytes())
            raw_idx_eq = (kb is None and k1 is None) or (kb is not None and k1 is not None and np.array_equal(kb, k1))
            diffs, sem_mask_eq, idx_detail, conj_pairs = [], "", "", ""
            if wb_b != wb_1:
                diffs.append("wideband")
            elif not wb_b:
                if int(info_b["kcap"][i]) != int(info_1["kcap"][0]):
                    diffs.append("k_cap")
                if int(info_b["knee"][i]) != int(info_1["knee"][0]):
                    diffs.append("knee")
                if int(info_b["k"][i]) != int(info_1["k"][0]):
                    diffs.append("k")
                else:
                    sem_mask_eq = all(np.array_equal(conj_class_counts(kb[c], T), conj_class_counts(k1[c], T))
                                      for c in range(x32.shape[0]))
                    idx_detail, conj_pairs = index_diff(kb, k1, T)
                    if not sem_mask_eq:
                        diffs.append("topk_mask")
            k_for_gap = None if wb_b else int(info_b["k"][i])
            m = margins_f64(x32, k_for_gap)
            explained = []
            if not valid:
                cls = "INVALID"
            elif not diffs:
                cls = "EXACT" if bytes_eq else ("NUMERICALLY_EQUIVALENT" if maxd <= tol else "OUTPUT_DIFFERENT")
            else:
                ok_all, explained = explain_diffs(diffs, m)
                cls = "DECISION_DIFFERENT_NEAR_THRESHOLD" if ok_all else "DECISION_DIFFERENT_UNEXPLAINED"
            batch_rows.append({
                "set": setname, "sample_id": it["sample_id"], "batch_size": len(items),
                "wideband_equal": wb_b == wb_1,
                "k_cap_equal": int(info_b["kcap"][i]) == int(info_1["kcap"][0]),
                "knee_equal": int(info_b["knee"][i]) == int(info_1["knee"][0]),
                "k_equal": int(info_b["k"][i]) == int(info_1["k"][0]),
                "batch_k": "" if wb_b else int(info_b["k"][i]), "n1_k": "" if wb_1 else int(info_1["k"][0]),
                "topk_semantic_equivalent": sem_mask_eq, "topk_index_differences": idx_detail,
                "topk_differences_conjugate_pairs": conj_pairs,
                # diagnostics only (not gates)
                "flatness_bits_equal": fb(info_b["flatness"][i]) == fb(info_1["flatness"][0]),
                "spectral_ratio_bits_equal": fb(info_b["ratio"][i]) == fb(info_1["ratio"][0]),
                "topk_indices_equal": raw_idx_eq, "output_bytes_equal": bytes_eq,
                "input_max_abs": inp_max, "tolerance": tol, "max_abs_diff": maxd,
                "margin_flatness_rel": m["flat"], "margin_ratio3_rel": m["r3"], "margin_ratio10_rel": m["r10"],
                "margin_kcap_half": m["kcap"], "margin_knee_rel": m["knee"], "margin_topk_gap_rel": m["gap"],
                "margin_quant_half": m["quant"],
                "differing_decisions": ";".join(diffs), "explained_by": " | ".join(explained), "classification": cls})

    # ---------------------------------------------------------------- outputs
    syn_meta = {c["sample_id"]: c for c in synthetic}
    syn_rows = []
    for mode in ("A", "B"):
        for r in results["synthetic"][mode]:
            c = syn_meta[r["sample_id"]]
            syn_rows.append({**r, **{k: c[k] for k in SYN_EXTRA}})
    write_csv(out_dir / "synthetic_cases.csv", PARITY_FIELDS + SYN_EXTRA, syn_rows)
    write_csv(out_dir / "clean_parity.csv", PARITY_FIELDS, results["clean"]["A"] + results["clean"]["B"])
    write_csv(out_dir / "attacked_subset_parity.csv", PARITY_FIELDS, results["attacked"]["A"] + results["attacked"]["B"])
    write_csv(out_dir / "pytorch_batch_parity.csv", BATCH_FIELDS, batch_rows)

    def counts(rows):
        return {c: sum(1 for r in rows if r["classification"] == c) for c in CLASSES}

    summary_counts = {s: {m: counts(results[s][m]) for m in ("A", "B")} for s in sets}
    all_ab = [r for s in sets for m in ("A", "B") for r in results[s][m]]
    gates = {
        "1_no_invalid": all(r["classification"] != "INVALID" for r in all_ab + batch_rows),
        "2_instrumented_bit_identical_to_public": counters["bit_identity_checks_passed"] == counters["pytorch_calls"],
        "3_float64_no_unexplained_decision_mismatch": all(
            r["classification"] != "DECISION_DIFFERENT_UNEXPLAINED" for s in sets for r in results[s]["B"]),
        "4_float32_no_unexplained_decision_mismatch": all(
            r["classification"] != "DECISION_DIFFERENT_UNEXPLAINED" for s in sets for r in results[s]["A"]),
        "5_matching_decisions_within_tolerance": all(r["classification"] != "OUTPUT_DIFFERENT" for r in all_ab),
        "6_batch_vs_n1_no_unexplained_decision_mismatch": all(
            r["classification"] != "DECISION_DIFFERENT_UNEXPLAINED" for r in batch_rows),
        "7_batch_vs_n1_matching_decisions_within_tolerance": all(
            r["classification"] != "OUTPUT_DIFFERENT" for r in batch_rows),
    }
    overall = "PASS" if all(gates.values()) else "FAIL"
    mode_c_counts = {s: counts([r for r in batch_rows if r["set"] == s]) for s in sets}
    real_c = [r for r in batch_rows if r["set"] in ("clean", "attacked")]
    real_check = {s: {"n": sum(1 for r in batch_rows if r["set"] == s),
                      "decision_different_unexplained": mode_c_counts[s]["DECISION_DIFFERENT_UNEXPLAINED"],
                      "output_different": mode_c_counts[s]["OUTPUT_DIFFERENT"], "invalid": mode_c_counts[s]["INVALID"],
                      "decision_different_near_threshold": mode_c_counts[s]["DECISION_DIFFERENT_NEAR_THRESHOLD"]}
                  for s in ("clean", "attacked")}

    def dstats(vals):
        v = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float64)
        if v.size == 0:
            return {"n": 0}
        return {"n": int(v.size), "max": float(v.max()), "median": float(np.median(v)),
                "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99))}

    same_decision = [r for r in batch_rows if r["classification"] in ("EXACT", "NUMERICALLY_EQUIVALENT",
                                                                         "OUTPUT_DIFFERENT")]
    mode_c_diagnostics = {
        s: {"n": sum(1 for r in batch_rows if r["set"] == s),
            "flatness_bit_differences": sum(1 for r in batch_rows if r["set"] == s and not r["flatness_bits_equal"]),
            "spectral_ratio_bit_differences": sum(1 for r in batch_rows if r["set"] == s
                                                  and not r["spectral_ratio_bits_equal"]),
            "raw_topk_index_differences": sum(1 for r in batch_rows if r["set"] == s and not r["topk_indices_equal"]),
            "output_byte_differences": sum(1 for r in batch_rows if r["set"] == s and not r["output_bytes_equal"]),
            "output_abs_diff_all_rows": dstats([r["max_abs_diff"] for r in batch_rows if r["set"] == s]),
            "output_abs_diff_rows_with_equal_decisions": dstats(
                [r["max_abs_diff"] for r in same_decision if r["set"] == s])}
        for s in sets}
    mode_c_flagged = [{k: r[k] for k in ("set", "sample_id", "classification", "differing_decisions", "explained_by",
                                         "topk_index_differences", "topk_differences_conjugate_pairs",
                                         "max_abs_diff", "tolerance")}
                      for r in batch_rows if r["classification"] not in ("EXACT", "NUMERICALLY_EQUIVALENT")]
    flagged = [{k: r[k] for k in ("set", "mode", "sample_id", "classification", "differing_decisions", "explained_by",
                                  "max_abs_diff", "tolerance")}
               for r in all_ab if r["classification"] not in ("EXACT", "NUMERICALLY_EQUIVALENT")]
    wb_share = {s: {m: sum(1 for r in results[s][m] if r["ref_wideband"]) / max(len(results[s][m]), 1)
                    for m in ("A", "B")} for s in sets}
    syn_targets = [{k: c[k] for k in ("sample_id", "category", "target_quantity", "target_value", "achieved_value",
                                      "target_reached", "construction")} for c in synthetic if c["target_quantity"]]
    summary = {
        "overall": overall, "gates": gates, "classification_counts": summary_counts,
        "mode_C_classification_counts": mode_c_counts,
        "mode_C_real_data_check": real_check,
        "mode_C_diagnostics_not_gates": mode_c_diagnostics,
        "mode_C_non_exact_non_equivalent_samples": mode_c_flagged,
        "mode_C_note": "batched vs N=1 compared on semantic decisions (wideband, k_cap, knee, K, Top-K kept bins "
                       "modulo conjugate pairs) and numerical output tolerance; raw floating-point bit equality of "
                       "flatness / spectral_ratio / outputs is reported as a diagnostic only",
        "pytorch_calls": counters["pytorch_calls"], "bit_identity_checks_passed": counters["bit_identity_checks_passed"],
        "non_exact_non_equivalent_samples": flagged,
        "reference_wideband_share": wb_share,
        "synthetic_targets": syn_targets,
        "tolerance_rule": f"max_abs_diff <= {args.rel_tol} * max|input| + {args.abs_floor}",
        "margins": {"margin_rel": args.margin_rel, "kcap_margin_half_integer": args.kcap_margin,
                    "quant_margin_half_level": args.quant_margin},
        "note": LINEAR_NOTE,
    }
    (out_dir / "parity_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    manifest = {
        "phase": "adaptive_k_phaseA_reference_parity", "command": [sys.executable] + sys.argv,
        "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "wall_seconds": time.monotonic() - t_start,
        "reference": {"path": p1a.rel(REFERENCE_PATH), "sha256": ref_sha, "expected": EXPECTED_REFERENCE_SHA256,
                      "upstream": "nigelzzz/adversarial-rf@796a452 awn_fpga/adaptive_k_v2.py"},
        "production": {"path": p1a.rel(DEFENSE_PATH), "function": "adaptive_k_v2_snr_defense", "sha256": def_sha,
                       "expected": EXPECTED_DEFENSE_SHA256, "adversarial_rf_commit": commit,
                       "expected_commit": EXPECTED_ADVRF_COMMIT},
        "dataset": {"path": cfg["dataset_path"], "sha256": hashes["dataset_sha256"]},
        "checkpoint": {"path": cfg["checkpoint_path"], "sha256": hashes["checkpoint_sha256"]},
        "formal_cpu_dir": p1a.rel(formal_dir), "formal_adversarial_arrays": formal_array_files(),
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0]},
        "dtype_device": {"mode_A": "reference float32 in / float64 internal / float32 out vs PyTorch float32 CPU",
                         "mode_B": "reference float64 vs PyTorch float64 CPU",
                         "mode_C": "PyTorch float32 CPU, batched vs N=1",
                         "torch_threads": {"intra": torch.get_num_threads(), "interop": torch.get_num_interop_threads()}},
        "thresholds": {k: v[0] for k, v in consts.items()}, "note": LINEAR_NOTE,
        "inputs": {"synthetic": len(synthetic), "clean": len(clean), "attacked": len(attacked),
                   "attacked_subset": "88 validated samples x {fgsm, pgd, bim} eps=0.03, formal seeds, Phase 1B path",
                   "domain": "unit-average-power IQ (same as the project's fixed Top-K), no AWN_All normalization"},
        "attack_settings": {a: settings[a] for a in ATTACKS},
        "classification_rules": __doc__.split("Per-sample classification")[1].split("PASS requires")[0].strip(),
        "overall": overall,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    p1a.log(f"[parity] done in {time.monotonic() - t_start:.0f} s: {out_dir}  overall={overall}")
    for g, v in gates.items():
        p1a.log(f"  {g}: {v}")
    for s in sets:
        p1a.log(f"  {s}: A={summary_counts[s]['A']}  B={summary_counts[s]['B']}")
    if overall != "PASS":
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=p1a.DEFAULT_CONFIG)
    ap.add_argument("--formal-dir", default=p1a.DEFAULT_FORMAL_CPU_DIR)
    ap.add_argument("--out-dir", default=None, help="default results/adaptive_k_parity_<timestamp>")
    ap.add_argument("--rel-tol", type=float, default=1e-5, help="output tolerance relative to max|input|")
    ap.add_argument("--abs-floor", type=float, default=1e-9, help="absolute tolerance floor for near-zero inputs")
    ap.add_argument("--margin-rel", type=float, default=1e-4,
                    help="relative margin around flatness / spectral-ratio / knee thresholds and Top-K gaps")
    ap.add_argument("--kcap-margin", type=float, default=1e-3,
                    help="max distance (in K units) of the interpolated cap from a half-integer")
    ap.add_argument("--quant-margin", type=float, default=1e-3,
                    help="max distance (in quantization-level units) of a differing level from a half-level")
    ap.add_argument("--attack-temperature", type=float, default=None)
    ap.add_argument("--diagnostics", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--dry-run", action="store_true", help="check sources and print the plan; no torch")
    args = ap.parse_args()
    if args.dry_run:
        ref_ok = REFERENCE_PATH.exists() and sha256_file(REFERENCE_PATH) == EXPECTED_REFERENCE_SHA256
        def_ok = DEFENSE_PATH.exists() and sha256_file(DEFENSE_PATH) == EXPECTED_DEFENSE_SHA256
        print(json.dumps({
            "reference": {"path": str(REFERENCE_PATH), "exists": REFERENCE_PATH.exists(),
                          "sha256": sha256_file(REFERENCE_PATH) if REFERENCE_PATH.exists() else None, "ok": ref_ok},
            "defense_py": {"path": str(DEFENSE_PATH), "exists": DEFENSE_PATH.exists(),
                           "sha256": sha256_file(DEFENSE_PATH) if DEFENSE_PATH.exists() else None, "ok": def_ok},
            "adversarial_rf_commit": p1a.run_cmd(["git", "-C", str(REPO_ROOT / "external" / "adversarial-rf"),
                                                  "rev-parse", "HEAD"]),
            "formal_adversarial_arrays": formal_array_files(),
            "inputs": {"synthetic": "~50 cases", "clean": 2200, "attacked": "88 x 3 = 264"},
            "modes": ["A: ref float64-internal vs torch float32", "B: ref float64 vs torch float64",
                      "C: torch float32 batched vs N=1"],
            "estimate_minutes": estimate_minutes(), "note": LINEAR_NOTE}, indent=2))
        return
    run(args)


if __name__ == "__main__":
    main()
