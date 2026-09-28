"""
Adaptive-K Phase B: CPU EFFECTIVENESS BENCHMARK (Raspberry Pi 5). Standalone; the formal pipeline is untouched.

Compares six defenses on the SAME stored IQ (paired):
    none | topk10 | topk20 | topk30 | topk40 | adaptive_k_v2
Adaptive-K v2 is a branch-based adaptive defense (wideband -> 32-level quantization; narrowband ->
adaptive Top-K), not "another K". Production implementations (pinned, never modified):
    external/adversarial-rf/util/defense.py : fft_topk_denoise, adaptive_k_v2_snr_defense
Phase A parity (PASS): results/adaptive_k_parity_20260927_162249.

Terminology: "spectral_ratio" = sum(top-10 PSD bins) / sum(remaining bins), LINEAR. Its thresholds 3 and 10 are
linear, not dB. Only the RadioML label is in dB (snr_label_db).

Population (formal): 2,200 clean samples (11 modulations x 20 RadioML SNR labels x 10 indices, formal order and
per-row seeds from results/cpu_full_matrix_20260921_seedaligned/base_results.csv, rebuilt and hash-checked) and
24 attack profiles (17 attacks) read from the formal config (manifest config == configs/hailo_full_matrix.json).
The formal run stored no adversarial arrays, so adversarial IQ is regenerated ONCE, stored, hashed, and every
defense is evaluated on that stored IQ.

Stages (resumable; pass the same --out-dir to resume)
  probe      regenerate the first --probe-samples samples of every profile and compare with the formal
             adversarial_sha256 (fast pre-flight before the multi-hour generation; writes probe_report.json)
  generate   clean store + adversarial store, chunked per (profile, modulation) = 200 samples per chunk,
             each chunk with .npy + metadata CSV + completion marker (count, file sha256, per-row sha256 digest);
             valid chunks are reused, invalid ones are renamed aside and regenerated
  evaluate   clean evaluation (2,200 x 6 defenses) + per-profile attacked evaluation (2,200 x 6 per profile),
             B=1 calls exactly like the formal defense path; per-part markers
  summarize  per-sample tables, all aggregations, formal reproduction checks, row-count gates, validation.json
  all        generate -> evaluate -> summarize

Attack call (formal-equivalent, per sample): torch.manual_seed(formal_seed); AttackAdapter(default).apply(x,
attack=<name>, eps=<profile eps>, temperature=<resolved>, seed=<formal_seed>, diagnostics=<resolved>,
cw_c/cw_steps/cw_lr (cw only), attack_params=<profile attack_params>). Torch thread settings are left at the
process defaults, as in the formal run. The Phase 1 B=1 mkldnn option is NOT used (default AttackAdapter).

Formal metric definitions (reports/meeting_20260921/analysis/phase_b_analysis.py), with the formal column
clean_topk_* generalised to clean_defended_* for every defense:
  clean_correct = clean_pred == label; attacked_correct; attack_success = clean_correct & ~attacked_correct
  conditional ASR = sum(attack_success) / sum(clean_correct)
  clean_degraded = clean_correct & ~clean_defended_correct;  clean degradation = sum / sum(clean_correct)
  recovered = attack_success & defended_correct;  recovery = sum(recovered) / sum(attack_success)
  retained = clean_correct & defended_correct;  retention = sum(retained) / sum(clean_correct)
  retained_nodef = clean_correct & ~attack_success;  net gain (retention_gain_pp) = retention - retention_nodef
  defended accuracy = sum(defended_correct) / n

Run on the Pi (from the repo root):
    .venv/bin/python experiments/benchmark_adaptive_k_effectiveness_pi5.py --dry-run
    .venv/bin/python experiments/benchmark_adaptive_k_effectiveness_pi5.py --stage probe
    nohup .venv/bin/python experiments/benchmark_adaptive_k_effectiveness_pi5.py --stage all --out-dir <dir> &
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as _dt
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import bench_attack_accel_pi5 as p1a  # noqa: E402 -- formal context / input rebuild helpers (no torch at import)

REPO_ROOT = p1a.REPO_ROOT
DEFENSE_PATH = REPO_ROOT / "external" / "adversarial-rf" / "util" / "defense.py"
REFERENCE_PATH = REPO_ROOT / "references" / "adaptive_k" / "adaptive_k_v2.py"
EXPECTED_DEFENSE_SHA256 = "291a4ae4b5da17c91c52b8db345259b76134799f717975630464c961dfbfc528"
EXPECTED_REFERENCE_SHA256 = "f818dd0229ae3a02271a6bc546593d3d5df2857e4ac63a6d2f26bc8ddf81a891"
EXPECTED_ADVRF_COMMIT = "ced705ed861bde352841b0b8bd038a6449e0fa4d"
PHASE_A_DIR = "results/adaptive_k_parity_20260927_162249"
LINEAR_NOTE = "spectral_ratio thresholds 3 and 10 are linear, not dB"

T_LEN = 128
N_CLEAN = 2200
N_PROFILES = 24
TOPK_VALUES = [10, 20, 30, 40]
DEFENSES = ["none", "topk10", "topk20", "topk30", "topk40", "adaptive_k_v2"]
DEFENSE_LABEL = {"none": "no defense", "topk10": "fixed Top-K K=10", "topk20": "fixed Top-K K=20",
                 "topk30": "fixed Top-K K=30", "topk40": "fixed Top-K K=40",
                 "adaptive_k_v2": "branch-based adaptive defense (wideband quantization + narrowband adaptive Top-K)"}

# Adaptive-K constants (verified against the pinned function's defaults at runtime)
FLATNESS_THRESHOLD, QUANT_LEVELS, RATIO_THRESH, PILOT_K = 0.4, 32, 0.05, 10
SPECTRAL_RATIO_LOW, SPECTRAL_RATIO_HIGH, K_CAP_LOW_RATIO, K_CAP_HIGH_RATIO = 3.0, 10.0, 35, 20

# Reproducibility classes from formal evidence (reports/meeting_20260921 section 4.3: CPU and NPU formal runs
# regenerated the adversarial inputs independently; 16 profiles were bit-identical on all 2,200 rows, 8 were not).
CLASS_HISTORICALLY_NON_BIT_IDENTICAL = {
    "pgd/eps0.005", "pgd/eps0.01", "pgd/eps0.03", "vmifgsm/eps0.03_default", "vnifgsm/eps0.03_default",
    "rfgsm/eps0.03_default", "tpgd/eps0.03_default", "autoattack/eps0.03_standard"}
CLASS_EXPECTED_BIT_IDENTICAL = {
    "fgsm/eps0.005", "fgsm/eps0.01", "fgsm/eps0.03", "fgsm/eps0.05", "bim/eps0.03_default",
    "mifgsm/eps0.03_default", "difgsm/eps0.03_default", "cw/c1_steps20_lr0.01", "deepfool/package_default",
    "fab/eps0.005", "fab/eps0.01", "fab/eps0.03", "square/eps0.03_default", "apgd/eps0.03_default",
    "apgdt/eps0.03_default", "ead/package_default"}
CLASS_EVIDENCE = {
    "EXPECTED_BIT_IDENTICAL": "CPU and NPU formal runs regenerated bit-identical adversarial IQ on all 2,200 rows "
                              "(FGSM/BIM additionally re-verified in Phase 1 and Phase A); must match formal "
                              "adversarial_sha256 -> any mismatch FAILS",
    "HISTORICALLY_NON_BIT_IDENTICAL": "CPU vs NPU formal runs differed (meeting report 4.3); attacks that draw from "
                                      "the global torch RNG; regenerated once, differences recorded, never a FAIL "
                                      "by themselves"}
# Formal CPU per-sample mean generation latency (ms), reports/meeting_20260921/tables/attack_generation_latency.csv,
# used ONLY for the runtime estimate.
FORMAL_GEN_MEAN_MS = {
    "fgsm/eps0.005": 9.61, "fgsm/eps0.01": 9.63, "fgsm/eps0.03": 9.62, "fgsm/eps0.05": 9.52,
    "bim/eps0.03_default": 56.26, "pgd/eps0.005": 54.89, "pgd/eps0.01": 55.20, "pgd/eps0.03": 55.19,
    "mifgsm/eps0.03_default": 55.87, "difgsm/eps0.03_default": 57.52, "vmifgsm/eps0.03_default": 309.62,
    "vnifgsm/eps0.03_default": 311.97, "rfgsm/eps0.03_default": 55.94, "tpgd/eps0.03_default": 58.85,
    "cw/c1_steps20_lr0.01": 65.08, "deepfool/package_default": 112.25, "fab/eps0.005": 416.48,
    "fab/eps0.01": 418.11, "fab/eps0.03": 419.47, "square/eps0.03_default": 2029.01,
    "apgd/eps0.03_default": 75.01, "apgdt/eps0.03_default": 151.13, "autoattack/eps0.03_standard": 1795.89,
    "ead/package_default": 1839.67}

ADV_META_FIELDS = ["global_index", "profile_index", "attack", "attack_profile", "profile", "modulation",
                   "snr_label_db", "sample_index", "row_in_chunk", "label", "formal_seed", "clean_pred",
                   "attacked_pred", "clean_correct", "attacked_correct", "attack_success", "iq_linf", "iq_l2",
                   "attack_ms", "adversarial_sha256", "formal_adversarial_sha256", "sha_matches_formal",
                   "reproducibility_class"]
CLEAN_META_FIELDS = ["global_index", "modulation", "snr_label_db", "sample_index", "label", "formal_seed",
                     "formal_clean_pred", "clean_input_sha256"]
AK_FIELDS = ["input_kind", "global_index", "attack", "attack_profile", "profile", "modulation", "snr_label_db",
             "sample_index", "spectral_flatness", "wideband", "branch", "spectral_ratio", "k_cap", "knee",
             "selected_k", "clean_branch", "clean_selected_k", "attack_changes_branch", "branch_transition",
             "delta_k", "defended_sha256", "route_ms", "select_ms", "topk_transform_ms", "public_total_ms"]
CLEAN_EVAL_FIELDS = ["global_index", "modulation", "snr_label_db", "sample_index", "label", "defense",
                     "clean_pred", "defended_pred", "clean_correct", "clean_defended_correct", "clean_degraded",
                     "clean_improved", "clean_preserved", "formal_clean_topk_pred", "formal_match",
                     "adaptive_branch", "adaptive_selected_k", "defense_ms", "infer_ms"]
ATT_EVAL_FIELDS = ["global_index", "attack", "attack_profile", "profile", "modulation", "snr_label_db",
                   "sample_index", "label", "defense", "clean_pred", "attacked_pred", "defended_pred",
                   "clean_correct", "attacked_correct", "attack_success", "defended_correct", "recovered",
                   "retained", "retained_nodef", "clean_defended_correct", "clean_degraded",
                   "adversarial_sha256", "sha_matches_formal", "formal_defended_pred", "formal_match",
                   "adaptive_branch", "adaptive_selected_k", "adaptive_clean_branch", "branch_transition",
                   "defense_ms", "infer_ms"]


# =========================================================================== small helpers (no torch)
def sha_file(p: Path) -> str:
    return p1a.sha256_file(p)


def sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def b2s(v) -> str:
    return "True" if v else "False"


def s2b(v: str) -> bool:
    return p1a.parse_bool(v)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr, allow_pickle=False)
    os.replace(tmp, path)


def write_csv(path: Path, fields: List[str], rows) -> int:
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
            n += 1
    os.replace(tmp, path)
    return n


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def wilson(k: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def pct(num: float, den: float) -> float:
    return 100.0 * num / den if den else float("nan")


def dist_stats(v) -> dict:
    a = np.asarray(v, dtype=np.float64)
    if a.size == 0:
        return {"count": 0, "mean": "", "median": "", "std": "", "p25": "", "p75": "", "min": "", "max": ""}
    return {"count": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "std": float(a.std()),
            "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)), "min": float(a.min()),
            "max": float(a.max())}


def lat_stats(v) -> dict:
    a = np.asarray([x for x in v if x == x], dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99))}


def profile_class(pkey: str) -> str:
    if pkey in CLASS_EXPECTED_BIT_IDENTICAL:
        return "EXPECTED_BIT_IDENTICAL"
    if pkey in CLASS_HISTORICALLY_NON_BIT_IDENTICAL:
        return "HISTORICALLY_NON_BIT_IDENTICAL"
    return "UNKNOWN"


# =========================================================================== formal artifacts
def load_formal(formal_dir: Path) -> dict:
    """Streams the formal CSVs keeping only the fields Phase B needs (read-only)."""
    att, att_cols, temps, grad_nonempty = {}, [], {}, {}
    with (formal_dir / "attack_results.csv").open(newline="") as f:
        rd = csv.DictReader(f)
        att_cols = rd.fieldnames or []
        tcol = next((c for c in ("attack_temperature", "temperature") if c in att_cols), None)
        gcols = [c for c in att_cols if "gradient" in c.lower()]
        for r in rd:
            k = (r["modulation"], int(float(r["snr_db"])), int(float(r["sample_index"])), r["attack"], r["attack_profile"])
            att[k] = {"sha": r.get("adversarial_sha256", ""), "clean_pred": int(float(r["clean_pred"])),
                      "attacked_pred": int(float(r["attacked_pred"])), "iq_linf": r.get("iq_linf", ""),
                      "iq_l2": r.get("iq_l2", "")}
            pk = (r["attack"], r["attack_profile"])
            if tcol:
                temps.setdefault(pk, set()).add(float(r[tcol]))
            if any(r.get(c) not in (None, "", "None", "nan", "NaN") for c in gcols):
                grad_nonempty[pk] = True
    dfn, dfn_n = {}, 0
    with (formal_dir / "defense_results.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            dfn_n += 1
            k = (r["modulation"], int(float(r["snr_db"])), int(float(r["sample_index"])), r["attack"],
                 r["attack_profile"], int(float(r["topk"])))
            dfn[k] = (int(float(r["defended_pred"])), int(float(r["clean_topk_pred"])))
    return {"att": att, "att_cols": att_cols, "temps": temps, "grad_nonempty": grad_nonempty,
            "dfn": dfn, "dfn_rows": dfn_n}


def resolve_profiles(cfg: dict, formal: dict, cli_temperature: Optional[float], diag_mode: str) -> List[dict]:
    """All attack profiles in formal config order; parameters read from the formal config, temperature cross-checked
    against the formal rows (same resolution rules as bench_attack_accel_pi5.resolve_attack_settings)."""
    out = []
    for entry in cfg.get("attacks", []):
        name = str(entry.get("name", "")).lower()
        for prof in entry.get("profiles", []):
            pid = prof.get("id")
            params = prof.get("attack_params") or {}
            if not isinstance(params, dict):
                p1a.abort(f"{name}/{pid}: attack_params is not a dict")
            eps, eps_src = p1a._first_present([("profile", prof), ("profile.attack_params", params)],
                                              ("eps", "attack_eps", "epsilon"))
            scopes = [("profile", prof), ("attack_entry", entry), ("config", cfg), ("config.attack", cfg.get("attack")),
                      ("config.attack_defaults", cfg.get("attack_defaults"))]
            temp_cfg, temp_src = p1a._first_present(scopes, ("attack_temperature", "temperature"))
            rows_t = formal["temps"].get((name, pid))
            if rows_t is not None and len(rows_t) != 1:
                p1a.abort(f"{name}/{pid}: formal rows carry multiple temperatures {sorted(rows_t)}")
            temp_rows = next(iter(rows_t)) if rows_t else None
            if temp_cfg is not None and temp_rows is not None and not math.isclose(float(temp_cfg), temp_rows):
                p1a.abort(f"{name}/{pid}: config temperature {temp_cfg} != formal row temperature {temp_rows}")
            if cli_temperature is not None:
                known = temp_cfg if temp_cfg is not None else temp_rows
                if known is not None and not math.isclose(float(known), cli_temperature):
                    p1a.abort(f"{name}/{pid}: --attack-temperature {cli_temperature} contradicts formal {known}")
                temperature, t_src = cli_temperature, "cli --attack-temperature"
            elif temp_cfg is not None:
                temperature, t_src = float(temp_cfg), temp_src
            elif temp_rows is not None:
                temperature, t_src = temp_rows, "formal attack_results.csv column"
            else:
                p1a.abort(f"{name}/{pid}: attack temperature not found; pass --attack-temperature")
            diagnostics = {"auto": bool(formal["grad_nonempty"].get((name, pid), False)), "on": True, "off": False}[diag_mode]
            cw = {}
            if name == "cw":
                cw = {"cw_c": float(prof.get("c", params.get("c", 1.0))),
                      "cw_steps": int(prof.get("steps", params.get("steps", 20))),
                      "cw_lr": float(prof.get("lr", params.get("lr", 0.01)))}
            pkey = f"{name}/{pid}"
            out.append({"profile_index": len(out), "attack": name, "attack_profile": pid, "profile": pkey,
                        "eps": float(eps) if eps is not None else 0.0, "eps_applicable": eps is not None,
                        "eps_source": eps_src or "not in profile (attack has no eps parameter; value unused)",
                        "attack_params": dict(params), "cw_kwargs": cw, "temperature": float(temperature),
                        "temperature_source": t_src, "diagnostics": diagnostics, "raw_profile": prof,
                        "attack_entry_name": entry.get("name"), "reproducibility_class": profile_class(pkey)})
    return out


def check_sources() -> dict:
    ref_sha = sha_file(REFERENCE_PATH) if REFERENCE_PATH.exists() else None
    def_sha = sha_file(DEFENSE_PATH) if DEFENSE_PATH.exists() else None
    commit = p1a.run_cmd(["git", "-C", str(REPO_ROOT / "external" / "adversarial-rf"), "rev-parse", "HEAD"])
    pa = REPO_ROOT / PHASE_A_DIR / "parity_summary.json"
    pa_overall = json.loads(pa.read_text()).get("overall") if pa.exists() else None
    return {"reference_sha256": ref_sha, "reference_ok": ref_sha == EXPECTED_REFERENCE_SHA256,
            "defense_py_sha256": def_sha, "defense_py_ok": def_sha == EXPECTED_DEFENSE_SHA256,
            "adversarial_rf_commit": commit, "adversarial_rf_commit_ok": (commit or "").strip() == EXPECTED_ADVRF_COMMIT,
            "phase_a_dir": PHASE_A_DIR, "phase_a_overall": pa_overall, "phase_a_ok": pa_overall == "PASS"}


def storage_estimate() -> dict:
    per = 2 * T_LEN * 4
    clean_b, adv_b = N_CLEAN * per, N_CLEAN * N_PROFILES * per
    return {"per_sample_bytes": per, "clean_iq_bytes": clean_b, "attacked_iq_bytes": adv_b,
            "npy_headers_bytes": (1 + N_PROFILES * 11) * 128,
            "adversarial_metadata_csv_bytes_approx": N_CLEAN * N_PROFILES * 330,
            "evaluation_csvs_bytes_approx": N_CLEAN * N_PROFILES * 6 * 330 + N_CLEAN * 6 * 250 + 55000 * 260,
            "total_bytes_approx": clean_b + adv_b + N_CLEAN * N_PROFILES * 330
                                  + N_CLEAN * N_PROFILES * 6 * 330 + N_CLEAN * 6 * 250 + 55000 * 260}


def runtime_estimate() -> dict:
    gen_s = sum(FORMAL_GEN_MEAN_MS.values()) * N_CLEAN / 1e3
    infer_s = (N_CLEAN + N_CLEAN * N_PROFILES) * 2.5e-3
    eval_s = (N_CLEAN + N_CLEAN * N_PROFILES) * (6 * 2.5e-3 + 4 * 0.3e-3 + 1.5e-3)
    return {"generate_attacks_hours": gen_s / 3600, "generate_attacked_inference_min": infer_s / 60,
            "evaluate_min": eval_s / 60, "summarize_min": 5.0,
            "total_hours_approx": (gen_s + infer_s + eval_s + 300) / 3600,
            "probe_min_approx": sum(FORMAL_GEN_MEAN_MS.values()) * 2 / 1e3 / 60 + 1.5,
            "basis": "formal CPU mean generation latency x 2,200 per profile; planning estimate"}


# =========================================================================== torch-side context
class Ctx:
    pass


def open_context(args, need_model: bool = True) -> Ctx:
    src = check_sources()
    for k in ("reference_ok", "defense_py_ok", "adversarial_rf_commit_ok", "phase_a_ok"):
        if not src[k]:
            p1a.abort(f"source check failed: {k} ({json.dumps(src)})")
    c = Ctx()
    c.src = src
    ctx = p1a.load_formal_context(args)
    c.cfg, c.fman, c.formal_dir = ctx["cfg"], ctx["manifest"], ctx["formal_dir"]
    c.cfg_path = ctx["cfg_path"]
    c.keys = [(m, int(s), int(i)) for m in c.cfg["modulations"] for s in c.cfg["snrs"] for i in c.cfg["sample_indices"]]
    if len(c.keys) != N_CLEAN:
        p1a.abort(f"formal grid has {len(c.keys)} samples, expected {N_CLEAN}")
    c.formal = load_formal(c.formal_dir)
    c.profiles = resolve_profiles(c.cfg, c.formal, args.attack_temperature, args.diagnostics)
    if len(c.profiles) != N_PROFILES:
        p1a.abort(f"formal config lists {len(c.profiles)} profiles, expected {N_PROFILES}")
    if len(c.formal["att"]) != N_CLEAN * N_PROFILES or c.formal["dfn_rows"] != N_CLEAN * N_PROFILES * len(TOPK_VALUES):
        p1a.abort(f"formal row counts wrong: attack={len(c.formal['att'])} defense={c.formal['dfn_rows']}")
    want = {k + (p["attack"], p["attack_profile"]) for k in c.keys for p in c.profiles}
    if want != set(c.formal["att"]):
        p1a.abort("formal attack_results key set != grid x config profiles (missing profile or naming mismatch)")
    unk = [p["profile"] for p in c.profiles if p["reproducibility_class"] == "UNKNOWN"]
    if unk:
        p1a.abort(f"profiles without a reproducibility classification: {unk}")
    c.hashes = {"dataset_sha256": sha_file(REPO_ROOT / c.cfg["dataset_path"]),
                "checkpoint_sha256": sha_file(REPO_ROOT / c.cfg["checkpoint_path"]),
                "config_sha256": sha_file(c.cfg_path)}
    if c.hashes["dataset_sha256"] != c.fman.get("dataset_sha256") or \
            c.hashes["checkpoint_sha256"] != c.fman.get("checkpoint_sha256"):
        p1a.abort("dataset/checkpoint sha256 differs from formal CPU manifest")
    if not need_model:
        return c
    import torch
    c.torch = torch
    advrf = str(REPO_ROOT / "external" / "adversarial-rf")
    if advrf not in sys.path:
        sys.path.insert(0, advrf)
    import util.defense as D  # noqa: E402
    import inspect
    c.D = D
    sig = inspect.signature(D.adaptive_k_v2_snr_defense).parameters
    for name, val in (("flatness_threshold", FLATNESS_THRESHOLD), ("quant_levels", QUANT_LEVELS),
                      ("ratio_thresh", RATIO_THRESH), ("pilot_k", PILOT_K), ("snr_low", SPECTRAL_RATIO_LOW),
                      ("snr_high", SPECTRAL_RATIO_HIGH), ("k_max_low_snr", K_CAP_LOW_RATIO),
                      ("k_max_high_snr", K_CAP_HIGH_RATIO)):
        if sig[name].default != val:
            p1a.abort(f"adaptive_k_v2_snr_defense default {name}={sig[name].default} != expected {val}")
    import src.adapters.attack_adapter as am
    from src.adapters.attack_adapter import AttackAdapter, _REAL_ATTACK_SOURCE
    from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE
    c.am, c.REAL_ATTACK, c.REAL_MODEL = am, _REAL_ATTACK_SOURCE, _REAL_MODEL_SOURCE
    c.awn = AWNModelAdapter(checkpoint_path=str(REPO_ROOT / c.cfg["checkpoint_path"]), device="cpu")
    if c.awn.model is None or c.awn.backend_name != _REAL_MODEL_SOURCE or c.awn.status != "ok":
        p1a.abort("AWN backend is not real")
    c.adapter = AttackAdapter(awn_model=c.awn.model, device="cpu")  # default: Phase 1 B=1 option OFF
    if c.adapter.wrapped_model is None or c.adapter.backend_name != _REAL_ATTACK_SOURCE or c.adapter.status != "ok":
        p1a.abort("attack backend is not real")
    if getattr(c.adapter, "b1_pgd_bim_mkldnn_off", False):
        p1a.abort("the Phase 1 B=1 mkldnn option must be OFF for the formal-equivalent attack call")
    c.threads = {"intra": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
                 "note": "process defaults (not set), as in the formal run"}
    c.ak_checks = {"calls": 0, "bit_identical": 0}
    return c


def infer_pred(c: Ctx, x: np.ndarray) -> Tuple[int, float]:
    t0 = time.perf_counter()
    logits, meta = c.awn.infer(x)
    dt = (time.perf_counter() - t0) * 1e3
    if meta["awn_backend"] != c.REAL_MODEL or meta["awn_status"] != "ok" or not np.isfinite(logits).all():
        p1a.abort(f"AWN inference invalid (non-finite logits or backend fallback): {meta}")
    return int(np.argmax(logits[0])), dt


def check_iq(x: np.ndarray, where: str) -> None:
    if x.shape != (1, 2, T_LEN) or x.dtype != np.float32 or not np.isfinite(x).all():
        p1a.abort(f"invalid IQ at {where}: shape={x.shape} dtype={x.dtype} finite={bool(np.isfinite(x).all())}")


# =========================================================================== defenses
def fixed_topk(c: Ctx, x: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
    """Same call as src/adapters/topk_adapter.py (formal defense path), B=1."""
    t0 = time.perf_counter()
    y = c.D.fft_topk_denoise(c.torch.from_numpy(x), topk=k).detach().cpu().numpy().astype(np.float32)
    return y, (time.perf_counter() - t0) * 1e3


def adaptive_k(c: Ctx, x: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Production output = adaptive_k_v2_snr_defense(x). Decisions come from an instrumented replica of the
    public function's exact operation order (pinned private helpers) whose output must be bit-identical."""
    torch, D = c.torch, c.D
    xt = torch.from_numpy(x)
    t0 = time.perf_counter()
    public = D.adaptive_k_v2_snr_defense(xt)
    public_ms = (time.perf_counter() - t0) * 1e3
    y_pub = public.detach().cpu().contiguous().numpy()
    # ---- instrumented replica (same statements as the public function)
    s0 = time.perf_counter()
    result, X, is_nb = D._shared_fft_and_route(xt, FLATNESS_THRESHOLD, QUANT_LEVELS)
    s1 = time.perf_counter()
    info = {"route_ms": (s1 - s0) * 1e3, "select_ms": "", "topk_transform_ms": "", "public_total_ms": public_ms}
    if is_nb.any():
        X_n = X[is_nb]
        snr_est = D._estimate_snr_from_fft(X_n, PILOT_K)
        t = (snr_est - SPECTRAL_RATIO_LOW) / (SPECTRAL_RATIO_HIGH - SPECTRAL_RATIO_LOW + 1e-12)
        t = t.clamp(0.0, 1.0)
        k_max_per_sample = (K_CAP_LOW_RATIO + t * (K_CAP_HIGH_RATIO - K_CAP_LOW_RATIO)).round().int()
        knee = D._knee_from_fft(X_n, RATIO_THRESH)
        knee_capped = torch.min(knee, k_max_per_sample)
        s2 = time.perf_counter()
        result[is_nb] = D._apply_per_sample_topk(X_n, knee_capped)
        s3 = time.perf_counter()
        info.update(select_ms=(s2 - s1) * 1e3, topk_transform_ms=(s3 - s2) * 1e3, wideband=False,
                    spectral_ratio=float(snr_est[0]), k_cap=int(k_max_per_sample[0]), knee=int(knee[0]),
                    selected_k=int(knee_capped[0]))
    else:
        info.update(wideband=True, spectral_ratio="", k_cap="", knee="", selected_k="")
    power = X.abs() ** 2 + 1e-20  # identical expression to _shared_fft_and_route (metadata only)
    geo = torch.exp(torch.log(power).mean(dim=2))
    flat = (geo / (power.mean(dim=2) + 1e-12)).mean(dim=1)
    if bool(flat[0] > FLATNESS_THRESHOLD) != info["wideband"]:
        p1a.abort("recomputed flatness does not reproduce the helper routing")
    info["spectral_flatness"] = float(flat[0])
    y_ins = result.detach().cpu().contiguous().numpy()
    c.ak_checks["calls"] += 1
    if y_ins.shape != y_pub.shape or y_ins.dtype != y_pub.dtype or y_ins.tobytes() != y_pub.tobytes():
        p1a.abort("Adaptive-K instrumentation output is NOT bit-identical to adaptive_k_v2_snr_defense")
    c.ak_checks["bit_identical"] += 1
    return y_pub.astype(np.float32, copy=False), info


def apply_defense(c: Ctx, name: str, x: np.ndarray, where: str):
    if name == "none":
        return x, 0.0, None
    if name.startswith("topk"):
        y, ms = fixed_topk(c, x, int(name[4:]))
        info = None
    else:
        y, info = adaptive_k(c, x)
        ms = info["public_total_ms"]
    check_iq(y, f"{where} defense={name} output")
    return y, ms, info


# =========================================================================== stores
def clean_paths(out: Path) -> dict:
    d = out / "clean_store"
    return {"dir": d, "npy": d / "clean_store.npy", "meta": d / "clean_metadata.csv", "marker": d / "COMPLETE.json"}


def chunk_paths(out: Path, p: dict, mod_index: int, mod: str) -> dict:
    d = out / "adversarial_store" / f"{p['profile_index']:02d}_{p['attack']}__{p['attack_profile']}"
    stem = f"chunk_{mod_index:02d}_{mod}"
    return {"dir": d, "npy": d / f"{stem}.npy", "meta": d / f"{stem}.csv", "marker": d / f"{stem}.COMPLETE.json"}


def rows_digest(shas: List[str]) -> str:
    return hashlib.sha256("\n".join(shas).encode()).hexdigest()


def validate_store(paths: dict, expect_n: int, sha_field: str, shape_ok=(2, T_LEN)) -> Tuple[bool, str]:
    """A stored chunk is reusable only if marker, file hash, row count and every per-row hash agree."""
    if not paths["marker"].exists():
        return False, "no marker"
    try:
        mk = json.loads(paths["marker"].read_text())
        if not (paths["npy"].exists() and paths["meta"].exists()):
            return False, "missing file"
        if sha_file(paths["npy"]) != mk["npy_sha256"]:
            return False, "npy sha256 mismatch"
        arr = np.load(paths["npy"], allow_pickle=False)
        meta = read_csv(paths["meta"])
        if arr.dtype != np.float32 or arr.shape != (expect_n,) + tuple(shape_ok) or len(meta) != expect_n \
                or mk["count"] != expect_n:
            return False, "count/shape mismatch"
        shas = [sha_arr(arr[j:j + 1]) for j in range(expect_n)]
        if shas != [m[sha_field] for m in meta] or rows_digest(shas) != mk["rows_digest"]:
            return False, "per-row sha mismatch"
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 - any corruption => not reusable
        return False, f"unreadable: {exc!r}"


def set_aside(paths: dict) -> None:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    for k in ("npy", "meta", "marker"):
        if paths[k].exists():
            os.replace(paths[k], paths[k].with_name(paths[k].name + f".invalid_{ts}"))


def ensure_clean_store(c: Ctx, out: Path) -> Tuple[np.ndarray, List[dict]]:
    cp = clean_paths(out)
    cp["dir"].mkdir(parents=True, exist_ok=True)
    ok, why = validate_store(cp, N_CLEAN, "clean_input_sha256")
    if ok:
        return np.load(cp["npy"], allow_pickle=False), read_csv(cp["meta"])
    if cp["marker"].exists() or cp["npy"].exists():
        p1a.log(f"[clean] stored clean data not reusable ({why}); rebuilding")
        set_aside(cp)
    base_rows, _ = p1a.load_formal_rows(c.formal_dir, set(c.keys))
    inputs = p1a.rebuild_clean_inputs(c.keys, base_rows, p1a.resolve_sensing(c.cfg), REPO_ROOT / c.cfg["dataset_path"])
    arr = np.stack([inputs[k]["x"][0] for k in c.keys]).astype(np.float32)
    meta = []
    for gi, k in enumerate(c.keys):
        br = base_rows[k]
        h = p1a.sha256_array(inputs[k]["x"])
        if h != br["clean_input_sha256"]:
            p1a.abort(f"{k}: rebuilt clean input sha256 != formal")
        meta.append({"global_index": gi, "modulation": k[0], "snr_label_db": k[1], "sample_index": k[2],
                     "label": inputs[k]["label"], "formal_seed": inputs[k]["seed"],
                     "formal_clean_pred": inputs[k]["formal_clean_pred"], "clean_input_sha256": h})
    atomic_save_npy(cp["npy"], arr)
    write_csv(cp["meta"], CLEAN_META_FIELDS, meta)
    atomic_write_text(cp["marker"], json.dumps({"count": N_CLEAN, "npy_sha256": sha_file(cp["npy"]),
                                                "rows_digest": rows_digest([m["clean_input_sha256"] for m in meta]),
                                                "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}))
    return arr, read_csv(cp["meta"])


# =========================================================================== attack generation
def attack_one(c: Ctx, p: dict, x: np.ndarray, seed: int) -> Tuple[np.ndarray, float]:
    c.torch.manual_seed(seed)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        x_adv, meta = c.adapter.apply(x, attack=p["attack"], eps=p["eps"], temperature=p["temperature"], seed=seed,
                                      diagnostics=p["diagnostics"], attack_params=dict(p["attack_params"]),
                                      **p["cw_kwargs"])
    ms = (time.perf_counter() - t0) * 1e3
    if meta.get("attack_status") != "ok" or meta.get("attack_backend") != c.REAL_ATTACK:
        p1a.abort(f"{p['profile']}: attack fell back or failed: {meta.get('attack_status')} {meta.get('attack_notes')}")
    return x_adv, ms


def stage_probe(c: Ctx, out: Path, n: int) -> dict:
    clean, cmeta = ensure_clean_store(c, out)
    rep = {"probe_samples_per_profile": n, "profiles": []}
    for p in c.profiles:
        res = []
        for gi in range(n):
            k = c.keys[gi]
            x = clean[gi:gi + 1]
            x_adv, ms = attack_one(c, p, x, int(cmeta[gi]["formal_seed"]))
            check_iq(x_adv, f"probe {p['profile']} {k}")
            fsha = c.formal["att"][k + (p["attack"], p["attack_profile"])]["sha"]
            res.append({"sample": list(k), "sha_matches_formal": p1a.sha256_array(x_adv) == fsha, "attack_ms": ms})
        n_eq = sum(r["sha_matches_formal"] for r in res)
        status = ("PASS" if n_eq == n else "FAIL") if p["reproducibility_class"] == "EXPECTED_BIT_IDENTICAL" \
            else "RECORDED"
        rep["profiles"].append({"profile": p["profile"], "class": p["reproducibility_class"], "n": n,
                                "n_sha_match": n_eq, "status": status, "samples": res})
        p1a.log(f"[probe] {p['profile']:32s} {p['reproducibility_class']:31s} {n_eq}/{n} match formal -> {status}")
    rep["overall"] = "PASS" if all(r["status"] != "FAIL" for r in rep["profiles"]) else "FAIL"
    atomic_write_text(out / "probe_report.json", json.dumps(rep, indent=2, default=str))
    return rep


def stage_generate(c: Ctx, out: Path) -> dict:
    clean, cmeta = ensure_clean_store(c, out)
    mods = list(c.cfg["modulations"])
    status = {"chunks_total": 0, "chunks_reused": 0, "chunks_generated": 0}
    for p in c.profiles:
        for mi, mod in enumerate(mods):
            cp = chunk_paths(out, p, mi, mod)
            cp["dir"].mkdir(parents=True, exist_ok=True)
            idx = [gi for gi, k in enumerate(c.keys) if k[0] == mod]
            status["chunks_total"] += 1
            ok, why = validate_store(cp, len(idx), "adversarial_sha256")
            if ok:
                status["chunks_reused"] += 1
                continue
            if cp["marker"].exists() or cp["npy"].exists():
                p1a.log(f"[generate] {cp['npy'].name} not reusable ({why}); regenerating")
                set_aside(cp)
            t_chunk = time.monotonic()
            arr = np.empty((len(idx), 2, T_LEN), dtype=np.float32)
            meta = []
            for j, gi in enumerate(idx):
                k = c.keys[gi]
                cm = cmeta[gi]
                x = clean[gi:gi + 1]
                seed = int(cm["formal_seed"])
                x_adv, ms = attack_one(c, p, x, seed)
                check_iq(x_adv, f"{p['profile']} {k}")
                sha = p1a.sha256_array(x_adv)
                fr = c.formal["att"][k + (p["attack"], p["attack_profile"])]
                eq = sha == fr["sha"]
                if p["reproducibility_class"] == "EXPECTED_BIT_IDENTICAL" and not eq:
                    atomic_write_text(out / "ABORT_attack_reproduction.json", json.dumps(
                        {"profile": p["profile"], "sample": list(k), "regenerated_sha256": sha,
                         "formal_sha256": fr["sha"], "class": p["reproducibility_class"]}, indent=2))
                    p1a.abort(f"{p['profile']} {k}: regenerated adversarial IQ != formal (expected bit-identical)")
                label = int(cm["label"])
                clean_pred = int(cm["formal_clean_pred"])
                att_pred, _ = infer_pred(c, x_adv)
                d = x_adv.astype(np.float64) - x.astype(np.float64)
                clean_correct = clean_pred == label
                meta.append({"global_index": gi, "profile_index": p["profile_index"], "attack": p["attack"],
                             "attack_profile": p["attack_profile"], "profile": p["profile"], "modulation": k[0],
                             "snr_label_db": k[1], "sample_index": k[2], "row_in_chunk": j, "label": label,
                             "formal_seed": seed, "clean_pred": clean_pred, "attacked_pred": att_pred,
                             "clean_correct": b2s(clean_correct), "attacked_correct": b2s(att_pred == label),
                             "attack_success": b2s(clean_correct and att_pred != label),
                             "iq_linf": float(np.max(np.abs(d))), "iq_l2": float(np.linalg.norm(d.reshape(-1))),
                             "attack_ms": ms, "adversarial_sha256": sha, "formal_adversarial_sha256": fr["sha"],
                             "sha_matches_formal": b2s(eq), "reproducibility_class": p["reproducibility_class"]})
                arr[j] = x_adv[0]
            atomic_save_npy(cp["npy"], arr)
            write_csv(cp["meta"], ADV_META_FIELDS, meta)
            atomic_write_text(cp["marker"], json.dumps({
                "profile": p["profile"], "modulation": mod, "count": len(idx), "npy_sha256": sha_file(cp["npy"]),
                "rows_digest": rows_digest([m["adversarial_sha256"] for m in meta]),
                "n_sha_matches_formal": sum(1 for m in meta if m["sha_matches_formal"] == "True"),
                "seconds": time.monotonic() - t_chunk,
                "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}))
            status["chunks_generated"] += 1
            p1a.log(f"[generate] {p['profile']} {mod}: {len(idx)} samples in {time.monotonic() - t_chunk:.0f} s")
    return status


def load_adversarial(c: Ctx, out: Path, p: dict) -> Tuple[np.ndarray, List[dict]]:
    arrs, metas = [], []
    for mi, mod in enumerate(c.cfg["modulations"]):
        cp = chunk_paths(out, p, mi, mod)
        n = sum(1 for k in c.keys if k[0] == mod)
        ok, why = validate_store(cp, n, "adversarial_sha256")
        if not ok:
            p1a.abort(f"adversarial chunk {cp['npy']} invalid or missing ({why}); run --stage generate first")
        arrs.append(np.load(cp["npy"], allow_pickle=False))
        metas.extend(read_csv(cp["meta"]))
    return np.concatenate(arrs), metas


# =========================================================================== evaluation
def eval_paths(out: Path, name: str) -> dict:
    d = out / "eval_parts"
    return {"dir": d, "rows": d / f"{name}.csv", "ak": d / f"{name}.adaptive.csv", "marker": d / f"{name}.COMPLETE.json"}


def part_valid(ep: dict, n_rows: int, n_ak: int) -> bool:
    if not ep["marker"].exists():
        return False
    try:
        mk = json.loads(ep["marker"].read_text())
        return (mk["rows"] == n_rows and mk["adaptive_rows"] == n_ak and ep["rows"].exists() and ep["ak"].exists()
                and sha_file(ep["rows"]) == mk["rows_sha256"] and sha_file(ep["ak"]) == mk["adaptive_sha256"])
    except Exception:  # noqa: BLE001
        return False


def finish_part(ep: dict, n_rows: int, n_ak: int, extra: dict) -> None:
    atomic_write_text(ep["marker"], json.dumps({"rows": n_rows, "adaptive_rows": n_ak,
                                                "rows_sha256": sha_file(ep["rows"]),
                                                "adaptive_sha256": sha_file(ep["ak"]), **extra,
                                                "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}))


def formal_clean_topk(c: Ctx, k: tuple, K: int) -> int:
    p0 = c.profiles[0]
    return c.formal["dfn"][k + (p0["attack"], p0["attack_profile"], K)][1]


def stage_evaluate(c: Ctx, out: Path) -> dict:
    clean, cmeta = ensure_clean_store(c, out)
    ep = eval_paths(out, "clean")
    ep["dir"].mkdir(parents=True, exist_ok=True)
    status = {"parts_reused": 0, "parts_evaluated": 0}
    if part_valid(ep, N_CLEAN * len(DEFENSES), N_CLEAN):
        status["parts_reused"] += 1
    else:
        rows, ak_rows = [], []
        for gi, k in enumerate(c.keys):
            cm = cmeta[gi]
            x = clean[gi:gi + 1]
            label = int(cm["label"])
            clean_pred, inf0 = infer_pred(c, x)
            if clean_pred != int(cm["formal_clean_pred"]):
                p1a.abort(f"{k}: clean prediction {clean_pred} != formal {cm['formal_clean_pred']}")
            for dname in DEFENSES:
                y, dms, info = apply_defense(c, dname, x, f"clean {k}")
                pred, inf = (clean_pred, inf0) if dname == "none" else infer_pred(c, y)
                f_pred = formal_clean_topk(c, k, int(dname[4:])) if dname.startswith("topk") else ""
                if f_pred != "" and pred != f_pred:
                    p1a.abort(f"{k} {dname}: clean defended prediction {pred} != formal clean_topk_pred {f_pred}")
                cc, dc = clean_pred == label, pred == label
                rows.append({"global_index": gi, "modulation": k[0], "snr_label_db": k[1], "sample_index": k[2],
                             "label": label, "defense": dname, "clean_pred": clean_pred, "defended_pred": pred,
                             "clean_correct": b2s(cc), "clean_defended_correct": b2s(dc),
                             "clean_degraded": b2s(cc and not dc), "clean_improved": b2s((not cc) and dc),
                             "clean_preserved": b2s(cc and dc), "formal_clean_topk_pred": f_pred,
                             "formal_match": "" if f_pred == "" else "True",
                             "adaptive_branch": ("wideband" if info["wideband"] else "narrowband") if info else "",
                             "adaptive_selected_k": info["selected_k"] if info else "", "defense_ms": dms,
                             "infer_ms": inf})
                if info:
                    ak_rows.append({"input_kind": "clean", "global_index": gi, "modulation": k[0],
                                    "snr_label_db": k[1], "sample_index": k[2],
                                    "spectral_flatness": info["spectral_flatness"], "wideband": b2s(info["wideband"]),
                                    "branch": "wideband" if info["wideband"] else "narrowband",
                                    "spectral_ratio": info["spectral_ratio"], "k_cap": info["k_cap"],
                                    "knee": info["knee"], "selected_k": info["selected_k"],
                                    "defended_sha256": sha_arr(y), "route_ms": info["route_ms"],
                                    "select_ms": info["select_ms"], "topk_transform_ms": info["topk_transform_ms"],
                                    "public_total_ms": info["public_total_ms"]})
        n1 = write_csv(ep["rows"], CLEAN_EVAL_FIELDS, rows)
        n2 = write_csv(ep["ak"], AK_FIELDS, ak_rows)
        finish_part(ep, n1, n2, {"part": "clean"})
        status["parts_evaluated"] += 1
        p1a.log(f"[evaluate] clean: {n1} rows")
    clean_rows = read_csv(ep["rows"])
    clean_def = {(int(r["global_index"]), r["defense"]): r for r in clean_rows}
    clean_ak = {int(r["global_index"]): r for r in read_csv(ep["ak"])}

    for p in c.profiles:
        ep = eval_paths(out, f"{p['profile_index']:02d}_{p['attack']}__{p['attack_profile']}")
        if part_valid(ep, N_CLEAN * len(DEFENSES), N_CLEAN):
            status["parts_reused"] += 1
            continue
        t_part = time.monotonic()
        arr, metas = load_adversarial(c, out, p)
        rows, ak_rows, n_formal_cmp = [], [], 0
        for j, m in enumerate(metas):
            gi = int(m["global_index"])
            k = c.keys[gi]
            x = arr[j:j + 1]
            check_iq(x, f"stored {p['profile']} {k}")
            if sha_arr(x) != m["adversarial_sha256"]:
                p1a.abort(f"stored adversarial IQ hash mismatch {p['profile']} {k}")
            label, clean_pred, att_pred = int(m["label"]), int(m["clean_pred"]), int(m["attacked_pred"])
            cc, ac = clean_pred == label, att_pred == label
            asucc = cc and not ac
            sha_eq = m["sha_matches_formal"] == "True"
            cak = clean_ak[gi]
            for dname in DEFENSES:
                y, dms, info = apply_defense(c, dname, x, f"{p['profile']} {k}")
                pred, inf = (att_pred, "") if dname == "none" else infer_pred(c, y)
                dc = pred == label
                f_pred, f_match = "", ""
                if dname.startswith("topk") and sha_eq:
                    f_pred = c.formal["dfn"][k + (p["attack"], p["attack_profile"], int(dname[4:]))][0]
                    f_match = pred == f_pred
                    n_formal_cmp += 1
                    if not f_match:
                        atomic_write_text(out / "ABORT_fixed_k_reproduction.json", json.dumps(
                            {"profile": p["profile"], "sample": list(k), "defense": dname, "defended_pred": pred,
                             "formal_defended_pred": f_pred}, indent=2))
                        p1a.abort(f"{p['profile']} {k} {dname}: fixed-K prediction differs from formal on "
                                  "bit-identical adversarial input (unexplained)")
                cdc = s2b(clean_def[(gi, dname)]["clean_defended_correct"])
                trans, ab, ak = "", "", ""
                if info:
                    ab = "wideband" if info["wideband"] else "narrowband"
                    ak = info["selected_k"]
                    trans = f"{'WB' if cak['branch'] == 'wideband' else 'NB'}->{'WB' if info['wideband'] else 'NB'}"
                rows.append({"global_index": gi, "attack": p["attack"], "attack_profile": p["attack_profile"],
                             "profile": p["profile"], "modulation": k[0], "snr_label_db": k[1], "sample_index": k[2],
                             "label": label, "defense": dname, "clean_pred": clean_pred, "attacked_pred": att_pred,
                             "defended_pred": pred, "clean_correct": b2s(cc), "attacked_correct": b2s(ac),
                             "attack_success": b2s(asucc), "defended_correct": b2s(dc),
                             "recovered": b2s(asucc and dc), "retained": b2s(cc and dc),
                             "retained_nodef": b2s(cc and not asucc), "clean_defended_correct": b2s(cdc),
                             "clean_degraded": b2s(cc and not cdc), "adversarial_sha256": m["adversarial_sha256"],
                             "sha_matches_formal": m["sha_matches_formal"], "formal_defended_pred": f_pred,
                             "formal_match": "" if f_match == "" else b2s(f_match), "adaptive_branch": ab,
                             "adaptive_selected_k": ak, "adaptive_clean_branch": cak["branch"] if info else "",
                             "branch_transition": trans, "defense_ms": dms, "infer_ms": inf})
                if info:
                    both_nb = (not info["wideband"]) and cak["branch"] == "narrowband"
                    ak_rows.append({"input_kind": "attacked", "global_index": gi, "attack": p["attack"],
                                    "attack_profile": p["attack_profile"], "profile": p["profile"],
                                    "modulation": k[0], "snr_label_db": k[1], "sample_index": k[2],
                                    "spectral_flatness": info["spectral_flatness"], "wideband": b2s(info["wideband"]),
                                    "branch": ab, "spectral_ratio": info["spectral_ratio"], "k_cap": info["k_cap"],
                                    "knee": info["knee"], "selected_k": info["selected_k"],
                                    "clean_branch": cak["branch"], "clean_selected_k": cak["selected_k"],
                                    "attack_changes_branch": b2s(cak["branch"] != ab), "branch_transition": trans,
                                    "delta_k": (int(info["selected_k"]) - int(cak["selected_k"])) if both_nb else "",
                                    "defended_sha256": sha_arr(y), "route_ms": info["route_ms"],
                                    "select_ms": info["select_ms"], "topk_transform_ms": info["topk_transform_ms"],
                                    "public_total_ms": info["public_total_ms"]})
        n1 = write_csv(ep["rows"], ATT_EVAL_FIELDS, rows)
        n2 = write_csv(ep["ak"], AK_FIELDS, ak_rows)
        finish_part(ep, n1, n2, {"part": p["profile"], "formal_fixed_k_comparisons": n_formal_cmp})
        status["parts_evaluated"] += 1
        p1a.log(f"[evaluate] {p['profile']}: {n1} rows in {time.monotonic() - t_part:.0f} s "
                f"(formal fixed-K comparisons {n_formal_cmp}, all matched)")
    status["adaptive_instrumentation_checks"] = dict(c.ak_checks)
    return status


# =========================================================================== summaries
def defense_metrics(rows: List[dict]) -> dict:
    n = len(rows)
    g = lambda f: sum(1 for r in rows if r[f] == "True")  # noqa: E731
    ncc, nas, nrec, ndc = g("clean_correct"), g("attack_success"), g("recovered"), g("defended_correct")
    nret, nretnd, ncdc, ncdeg, nac = g("retained"), g("retained_nodef"), g("clean_defended_correct"), \
        g("clean_degraded"), g("attacked_correct")
    lo, hi = wilson(nrec, nas)
    alo, ahi = wilson(nas, ncc)
    return {"n": n, "n_clean_correct": ncc, "n_attacked_correct": nac, "n_attack_success": nas,
            "n_recovered": nrec, "n_defended_correct": ndc, "n_retained": nret, "n_retained_nodef": nretnd,
            "n_clean_defended_correct": ncdc, "n_clean_degraded": ncdeg,
            "clean_acc_pct": pct(ncc, n), "attacked_acc_pct": pct(nac, n), "cond_asr_pct": pct(nas, ncc),
            "cond_asr_ci95_lo_pct": 100 * alo, "cond_asr_ci95_hi_pct": 100 * ahi,
            "recovery_pct": pct(nrec, nas), "recovery_ci95_lo_pct": 100 * lo, "recovery_ci95_hi_pct": 100 * hi,
            "defended_acc_pct": pct(ndc, n), "retention_pct": pct(nret, ncc), "retention_nodef_pct": pct(nretnd, ncc),
            "retention_gain_pp": pct(nret, ncc) - pct(nretnd, ncc), "clean_defended_acc_pct": pct(ncdc, n),
            "clean_degradation_pct": pct(ncdeg, ncc)}


def group_rows(rows: List[dict], keyf):
    out: Dict[tuple, List[dict]] = {}
    for r in rows:
        out.setdefault(keyf(r), []).append(r)
    return out


def stage_summarize(c: Ctx, out: Path, eval_status: Optional[dict]) -> dict:
    clean_ep = eval_paths(out, "clean")
    if not part_valid(clean_ep, N_CLEAN * len(DEFENSES), N_CLEAN):
        p1a.abort("clean evaluation part missing/invalid; run --stage evaluate")
    clean_rows = read_csv(clean_ep["rows"])
    ak_rows = read_csv(clean_ep["ak"])
    att_rows: List[dict] = []
    adv_meta: List[dict] = []
    part_files = []
    keep = ("global_index", "attack", "attack_profile", "profile", "modulation", "snr_label_db", "sample_index",
            "defense", "defended_pred", "clean_correct", "attacked_correct", "attack_success", "defended_correct",
            "recovered", "retained", "retained_nodef", "clean_defended_correct", "clean_degraded",
            "sha_matches_formal", "formal_match", "adaptive_branch", "adaptive_selected_k", "branch_transition",
            "defense_ms", "infer_ms")
    for p in c.profiles:
        ep = eval_paths(out, f"{p['profile_index']:02d}_{p['attack']}__{p['attack_profile']}")
        if not part_valid(ep, N_CLEAN * len(DEFENSES), N_CLEAN):
            p1a.abort(f"evaluation part for {p['profile']} missing/invalid; run --stage evaluate")
        part_files.append(ep["rows"])
        with ep["rows"].open(newline="") as f:  # reduced, string-interned rows (316,800 in total)
            for r in csv.DictReader(f):
                att_rows.append({k: sys.intern(r[k]) for k in keep})
        ak_rows.extend(read_csv(ep["ak"]))
        _, metas = load_adversarial(c, out, p)
        adv_meta.extend(metas)

    # ---------------------------------------------------------------- row-count gates
    counts = {"clean_samples": len({r["global_index"] for r in clean_rows}), "attack_profiles": len(c.profiles),
              "attacked_samples": len(adv_meta), "clean_eval_rows": len(clean_rows), "attacked_eval_rows": len(att_rows),
              "adaptive_decision_rows": len(ak_rows)}
    expected = {"clean_samples": N_CLEAN, "attack_profiles": N_PROFILES, "attacked_samples": N_CLEAN * N_PROFILES,
                "clean_eval_rows": N_CLEAN * len(DEFENSES), "attacked_eval_rows": N_CLEAN * N_PROFILES * len(DEFENSES),
                "adaptive_decision_rows": N_CLEAN + N_CLEAN * N_PROFILES}
    count_gate = counts == expected

    # ---------------------------------------------------------------- per-sample outputs + store index
    write_csv(out / "clean_per_sample.csv", CLEAN_EVAL_FIELDS, clean_rows)
    tmp = out / "attacked_per_sample.csv.tmp"
    with tmp.open("w", newline="") as fo:  # stream-concatenate the validated parts (identical headers)
        for i, pf in enumerate(part_files):
            with pf.open(newline="") as fi:
                header = fi.readline()
                if header.strip().split(",") != ATT_EVAL_FIELDS:
                    p1a.abort(f"unexpected header in {pf}")
                if i == 0:
                    fo.write(header)
                for line in fi:
                    fo.write(line)
    os.replace(tmp, out / "attacked_per_sample.csv")
    write_csv(out / "adaptive_decisions.csv", AK_FIELDS, ak_rows)
    write_csv(out / "adversarial_hashes.csv", ADV_META_FIELDS, adv_meta)
    write_csv(out / "attack_profiles.csv",
              ["profile_index", "attack", "attack_profile", "profile", "reproducibility_class", "eps", "eps_applicable",
               "eps_source", "attack_params", "cw_kwargs", "temperature", "temperature_source", "diagnostics",
               "raw_profile"],
              [{**p, "attack_params": json.dumps(p["attack_params"]), "cw_kwargs": json.dumps(p["cw_kwargs"]),
                "raw_profile": json.dumps(p["raw_profile"])} for p in c.profiles])

    # ---------------------------------------------------------------- attack reproduction (vs formal)
    rep_rows = []
    for p in c.profiles:
        ms = [m for m in adv_meta if m["profile"] == p["profile"]]
        f = c.formal["att"]
        eq = [m for m in ms if m["sha_matches_formal"] == "True"]
        pred_eq = sum(1 for m in ms if int(m["attacked_pred"]) ==
                      f[(m["modulation"], int(m["snr_label_db"]), int(m["sample_index"]), p["attack"],
                         p["attack_profile"])]["attacked_pred"])
        dlinf = [abs(float(m["iq_linf"]) - float(f[(m["modulation"], int(m["snr_label_db"]), int(m["sample_index"]),
                                                    p["attack"], p["attack_profile"])]["iq_linf"])) for m in ms
                 if f[(m["modulation"], int(m["snr_label_db"]), int(m["sample_index"]), p["attack"],
                       p["attack_profile"])]["iq_linf"] not in ("", None)]
        cls = p["reproducibility_class"]
        verdict = ("EXACT_MATCH" if len(eq) == len(ms) else
                   ("UNEXPLAINED_MISMATCH" if cls == "EXPECTED_BIT_IDENTICAL" else "EXPECTED_MISMATCH_RECORDED"))
        rep_rows.append({"profile": p["profile"], "reproducibility_class": cls, "n": len(ms), "n_sha_match": len(eq),
                         "pct_sha_match": pct(len(eq), len(ms)), "n_attacked_pred_match_formal": pred_eq,
                         "max_abs_diff_iq_linf_vs_formal": max(dlinf) if dlinf else "",
                         "median_abs_diff_iq_linf_vs_formal": float(np.median(dlinf)) if dlinf else "",
                         "verdict": verdict})
    write_csv(out / "attack_reproduction.csv", list(rep_rows[0].keys()), rep_rows)
    attack_repro_gate = all(r["verdict"] != "UNEXPLAINED_MISMATCH" for r in rep_rows)

    # ---------------------------------------------------------------- fixed-K reproduction (vs formal)
    rc_rows = []
    by_pd = group_rows(att_rows, lambda r: (r["profile"], r["defense"]))
    for p in c.profiles:
        for K in TOPK_VALUES:
            rr = by_pd.get((p["profile"], f"topk{K}"), [])
            sha_eq = [r for r in rr if r["sha_matches_formal"] == "True"]
            match = sum(1 for r in sha_eq if r["formal_match"] == "True")
            diff = [r for r in rr if r["sha_matches_formal"] != "True"]
            agree_diff = sum(1 for r in diff if int(r["defended_pred"]) == c.formal["dfn"][
                (r["modulation"], int(r["snr_label_db"]), int(r["sample_index"]), p["attack"], p["attack_profile"],
                 K)][0])
            rc_rows.append({"profile": p["profile"], "reproducibility_class": p["reproducibility_class"], "topk": K,
                            "n": len(rr), "n_input_bit_identical": len(sha_eq), "n_exact_match": match,
                            "n_unexplained_mismatch": len(sha_eq) - match,
                            "n_expected_mismatch_input_differs": len(diff),
                            "n_prediction_agreement_where_input_differs": agree_diff,
                            "pct_prediction_agreement_where_input_differs": pct(agree_diff, len(diff))})
    for K in TOPK_VALUES:
        cr = [r for r in clean_rows if r["defense"] == f"topk{K}"]
        rc_rows.append({"profile": "CLEAN", "reproducibility_class": "clean input (hash-checked)", "topk": K,
                        "n": len(cr), "n_input_bit_identical": len(cr),
                        "n_exact_match": sum(1 for r in cr if r["formal_match"] == "True"),
                        "n_unexplained_mismatch": sum(1 for r in cr if r["formal_match"] != "True"),
                        "n_expected_mismatch_input_differs": 0, "n_prediction_agreement_where_input_differs": "",
                        "pct_prediction_agreement_where_input_differs": ""})
    write_csv(out / "reproduction_check.csv", list(rc_rows[0].keys()), rc_rows)
    fixed_k_repro_gate = all(r["n_unexplained_mismatch"] == 0 for r in rc_rows)

    # ---------------------------------------------------------------- clean summary
    cs = []
    for scope, keyf in (("overall", lambda r: "ALL"), ("modulation", lambda r: r["modulation"]),
                        ("snr_label_db", lambda r: r["snr_label_db"])):
        for gk, grp in group_rows(clean_rows, lambda r: (keyf(r), r["defense"])).items():
            n = len(grp)
            cc = sum(1 for r in grp if r["clean_correct"] == "True")
            dc = sum(1 for r in grp if r["clean_defended_correct"] == "True")
            deg = sum(1 for r in grp if r["clean_degraded"] == "True")
            imp = sum(1 for r in grp if r["clean_improved"] == "True")
            pres = sum(1 for r in grp if r["clean_preserved"] == "True")
            cs.append({"scope": scope, "group": gk[0], "defense": gk[1], "defense_label": DEFENSE_LABEL[gk[1]],
                       "n": n, "n_clean_correct": cc, "clean_acc_pct": pct(cc, n), "n_clean_defended_correct": dc,
                       "clean_defended_acc_pct": pct(dc, n), "n_clean_degraded": deg,
                       "clean_degradation_pct": pct(deg, cc), "n_clean_preserved": pres,
                       "clean_preservation_pct": pct(pres, cc), "n_clean_improved": imp,
                       "acc_change_pp": pct(dc, n) - pct(cc, n)})
    write_csv(out / "clean_summary.csv", list(cs[0].keys()), cs)

    # ---------------------------------------------------------------- attack summary (defense-independent)
    none_rows = [r for r in att_rows if r["defense"] == "none"]
    meta_by = {(m["profile"], m["global_index"]): m for m in adv_meta}
    asum = []
    for scope, keyf in (("overall", lambda r: "ALL_24"), ("profile", lambda r: r["profile"]),
                        ("attack", lambda r: r["attack"]), ("modulation", lambda r: r["modulation"]),
                        ("snr_label_db", lambda r: r["snr_label_db"])):
        for gk, grp in group_rows(none_rows, keyf).items():
            mtr = defense_metrics(grp)
            lin = [float(meta_by[(r["profile"], r["global_index"])]["iq_linf"]) for r in grp]
            l2 = [float(meta_by[(r["profile"], r["global_index"])]["iq_l2"]) for r in grp]
            asum.append({"scope": scope, "group": gk, **{k: mtr[k] for k in (
                "n", "n_clean_correct", "n_attacked_correct", "n_attack_success", "clean_acc_pct", "attacked_acc_pct",
                "cond_asr_pct", "cond_asr_ci95_lo_pct", "cond_asr_ci95_hi_pct")},
                "mean_iq_linf": float(np.mean(lin)), "mean_iq_l2": float(np.mean(l2))})
    write_csv(out / "attack_summary.csv", list(asum[0].keys()), asum)

    # ---------------------------------------------------------------- defense summary (all aggregations)
    ds = []
    scopes = [("overall", lambda r: "ALL_24"), ("profile", lambda r: r["profile"]), ("attack", lambda r: r["attack"]),
              ("modulation", lambda r: r["modulation"]), ("snr_label_db", lambda r: r["snr_label_db"])]
    for scope, keyf in scopes:
        for gk, grp in group_rows(att_rows, lambda r: (keyf(r), r["defense"])).items():
            ds.append({"scope": scope, "group": gk[0], "defense": gk[1], "defense_label": DEFENSE_LABEL[gk[1]],
                       **defense_metrics(grp)})
    ak_att = [r for r in att_rows if r["defense"] == "adaptive_k_v2"]
    for scope, keyf in (("adaptive_attacked_branch", lambda r: r["adaptive_branch"]),
                        ("adaptive_selected_k_narrowband", lambda r: r["adaptive_selected_k"]),
                        ("adaptive_branch_transition", lambda r: r["branch_transition"])):
        for gk, grp in group_rows([r for r in ak_att if scope != "adaptive_selected_k_narrowband"
                                   or r["adaptive_branch"] == "narrowband"], keyf).items():
            ds.append({"scope": scope, "group": gk, "defense": "adaptive_k_v2",
                       "defense_label": DEFENSE_LABEL["adaptive_k_v2"], **defense_metrics(grp)})
    write_csv(out / "defense_summary.csv", list(ds[0].keys()), ds)

    # ---------------------------------------------------------------- fixed-K comparison (trade-offs, no winner)
    comp = []
    by = {(d["scope"], d["group"], d["defense"]): d for d in ds}
    for scope in ("overall", "profile", "attack", "snr_label_db", "modulation"):
        for grp in sorted({d["group"] for d in ds if d["scope"] == scope}, key=str):
            a = by.get((scope, grp, "adaptive_k_v2"))
            for dname in DEFENSES:
                d = by.get((scope, grp, dname))
                if d is None or a is None:
                    continue
                comp.append({"scope": scope, "group": grp, "defense": dname, "defense_label": DEFENSE_LABEL[dname],
                             **{m: d[m] for m in ("clean_degradation_pct", "recovery_pct", "retention_pct",
                                                  "retention_gain_pp", "defended_acc_pct", "clean_defended_acc_pct")},
                             **{f"adaptive_minus_this_{m}": a[m] - d[m] for m in (
                                 "clean_degradation_pct", "recovery_pct", "retention_pct", "retention_gain_pp",
                                 "defended_acc_pct")},
                             "note": "trade-off table; no single winner score. adaptive_k_v2 = "
                                     + DEFENSE_LABEL["adaptive_k_v2"]})
    write_csv(out / "comparison_fixed_k.csv", list(comp[0].keys()), comp)

    # ---------------------------------------------------------------- K distribution (narrowband only)
    kd = []
    clean_ak = [r for r in ak_rows if r["input_kind"] == "clean"]
    att_ak = [r for r in ak_rows if r["input_kind"] == "attacked"]
    for kind, rs in (("clean", clean_ak), ("attacked", att_ak)):
        nb = [r for r in rs if r["branch"] == "narrowband"]
        for scope, keyf in (("overall", lambda r: "ALL"), ("snr_label_db", lambda r: r["snr_label_db"]),
                            ("modulation", lambda r: r["modulation"]),
                            ("profile", lambda r: r.get("profile") or "CLEAN")):
            for gk, grp in group_rows(nb, keyf).items():
                kd.append({"kind": "stats", "input_kind": kind, "scope": scope, "group": gk, "k_value": "",
                           **dist_stats([int(r["selected_k"]) for r in grp]),
                           "n_narrowband": len(grp), "note": "narrowband samples only; wideband has no K"})
        for kv, grp in sorted(group_rows(nb, lambda r: int(r["selected_k"])).items()):
            kd.append({"kind": "histogram", "input_kind": kind, "scope": "overall", "group": "ALL", "k_value": kv,
                       "count": len(grp), "n_narrowband": len(nb)})
    write_csv(out / "k_distribution.csv", ["kind", "input_kind", "scope", "group", "k_value", "count", "mean",
                                           "median", "std", "p25", "p75", "min", "max", "n_narrowband", "note"], kd)

    # ---------------------------------------------------------------- wideband share
    wb = []
    for kind, rs in (("clean", clean_ak), ("attacked", att_ak)):
        for scope, keyf in (("overall", lambda r: "ALL"), ("snr_label_db", lambda r: r["snr_label_db"]),
                            ("modulation", lambda r: r["modulation"]),
                            ("profile", lambda r: r.get("profile") or "CLEAN")):
            for gk, grp in group_rows(rs, keyf).items():
                nwb = sum(1 for r in grp if r["branch"] == "wideband")
                wb.append({"input_kind": kind, "scope": scope, "group": gk, "n": len(grp), "n_wideband": nwb,
                           "n_narrowband": len(grp) - nwb, "wideband_share_pct": pct(nwb, len(grp))})
    write_csv(out / "wideband_summary.csv", list(wb[0].keys()), wb)

    # ---------------------------------------------------------------- branch transitions (clean -> attacked)
    bt = []
    for scope, keyf in (("overall", lambda r: "ALL_24"), ("profile", lambda r: r["profile"]),
                        ("snr_label_db", lambda r: r["snr_label_db"]), ("modulation", lambda r: r["modulation"])):
        for gk, grp in group_rows(att_ak, keyf).items():
            n = len(grp)
            cnt = {t: sum(1 for r in grp if r["branch_transition"] == t) for t in ("NB->NB", "NB->WB", "WB->NB", "WB->WB")}
            dk = [int(r["delta_k"]) for r in grp if r["delta_k"] not in ("", None)]
            st = dist_stats(dk)
            bt.append({"scope": scope, "group": gk, "n": n, **{f"n_{t}": v for t, v in cnt.items()},
                       **{f"pct_{t}": pct(v, n) for t, v in cnt.items()},
                       "attack_changes_branch_pct": pct(cnt["NB->WB"] + cnt["WB->NB"], n),
                       **{f"delta_k_nb_nb_{k}": v for k, v in st.items()}})
    write_csv(out / "branch_transition.csv", list(bt[0].keys()), bt)

    # ---------------------------------------------------------------- lightweight latency
    lat = []
    for kind, rs in (("clean", clean_rows), ("attacked", att_rows)):
        for dname in DEFENSES[1:]:
            v = [float(r["defense_ms"]) for r in rs if r["defense"] == dname]
            lat.append({"input_kind": kind, "component": f"{dname}_defense_total", **lat_stats(v)})
        v = [float(r["infer_ms"]) for r in rs if r["defense"] != "none" and r["infer_ms"] not in ("", None)]
        lat.append({"input_kind": kind, "component": "awn_inference_defended", **lat_stats(v)})
    for kind, rs in (("clean", clean_ak), ("attacked", att_ak)):
        for comp_name in ("route_ms", "select_ms", "topk_transform_ms"):
            v = [float(r[comp_name]) for r in rs if r[comp_name] not in ("", None)]
            lat.append({"input_kind": kind, "component": f"adaptive_k_v2_instrumented_{comp_name}", **lat_stats(v)})
    for r in lat:
        r["note"] = ("lightweight per-call perf_counter timing, no thermal control; instrumented split is from the "
                     "replica (route = FFT + flatness routing + wideband quantization; select = spectral_ratio, "
                     "k_cap, knee; transform = narrowband Top-K + IFFT); detailed latency deferred to Phase C")
    write_csv(out / "latency_summary.csv", ["input_kind", "component", "n", "mean_ms", "median_ms", "p95_ms",
                                            "p99_ms", "note"], lat)

    finite_ok = all(math.isfinite(float(r["spectral_flatness"])) for r in ak_rows)
    gates = {
        "sources_hashes_commit_phaseA": all(c.src[k] for k in ("reference_ok", "defense_py_ok",
                                                                "adversarial_rf_commit_ok", "phase_a_ok")),
        "dataset_checkpoint_hash": True,  # enforced (abort) in open_context
        "all_24_profiles_present": len(c.profiles) == N_PROFILES,
        "row_counts": count_gate,
        "iq_logits_defense_outputs_finite": finite_ok,  # non-finite IQ / logits / outputs abort during the run
        "adaptive_instrumentation_bit_identical": (eval_status is None or eval_status.get(
            "adaptive_instrumentation_checks", {}).get("calls") == eval_status.get(
            "adaptive_instrumentation_checks", {}).get("bit_identical")),
        "attack_reproduction_no_unexplained_mismatch": attack_repro_gate,
        "fixed_k_reproduction_no_unexplained_mismatch": fixed_k_repro_gate,
    }
    validation = {"overall": "PASS" if all(gates.values()) else "FAIL", "gates": gates, "row_counts": counts,
                  "expected_row_counts": expected,
                  "schema_note": "defense 'none' is materialised as its own row (defended = attacked) so attacked "
                                 "rows = 52,800 x 6 and clean rows = 2,200 x 6",
                  "attack_reproduction": rep_rows, "note": LINEAR_NOTE}
    atomic_write_text(out / "validation.json", json.dumps(validation, indent=2, default=str))
    return validation


# =========================================================================== manifest / main
def write_manifest(c: Ctx, out: Path, args, stage_info: dict) -> None:
    mp = out / "manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else {"created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}
    import torch
    man.update({
        "phase": "adaptive_k_phaseB_effectiveness", "last_command": [sys.executable] + sys.argv,
        "updated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "sources": c.src, "dataset": {"path": c.cfg["dataset_path"], "sha256": c.hashes["dataset_sha256"]},
        "checkpoint": {"path": c.cfg["checkpoint_path"], "sha256": c.hashes["checkpoint_sha256"]},
        "formal_config": {"path": p1a.rel(c.cfg_path), "sha256": c.hashes["config_sha256"]},
        "formal_cpu_dir": p1a.rel(c.formal_dir), "phase_a_parity_dir": PHASE_A_DIR,
        "environment": p1a.environment_metadata(),
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0]},
        "threads": getattr(c, "threads", None),
        "attack_temperatures": sorted({p["temperature"] for p in c.profiles}),
        "attack_profiles": [{k: p[k] for k in ("profile_index", "profile", "attack", "attack_profile", "eps",
                                               "eps_applicable", "eps_source", "attack_params", "cw_kwargs",
                                               "temperature", "temperature_source", "diagnostics",
                                               "reproducibility_class", "raw_profile")} for p in c.profiles],
        "reproducibility_class_evidence": CLASS_EVIDENCE,
        "defenses": {d: DEFENSE_LABEL[d] for d in DEFENSES},
        "sample_count": N_CLEAN, "expected_row_counts": {
            "clean_samples": N_CLEAN, "attack_profiles": N_PROFILES, "attacked_samples": N_CLEAN * N_PROFILES,
            "clean_eval_rows": N_CLEAN * len(DEFENSES), "attacked_eval_rows": N_CLEAN * N_PROFILES * len(DEFENSES),
            "adaptive_decision_rows": N_CLEAN + N_CLEAN * N_PROFILES},
        "attack_call": "torch.manual_seed(formal_seed); AttackAdapter(default).apply(x, attack, eps, temperature, "
                       "seed=formal_seed, diagnostics, [cw_c/cw_steps/cw_lr for cw], attack_params); default threads; "
                       "Phase 1 B=1 mkldnn option OFF",
        "defense_call": "B=1: fft_topk_denoise(x, K) (same as TopKAdapter); adaptive_k_v2_snr_defense(x) with a "
                        "bit-identity-checked instrumented replica for decisions",
        "storage": storage_estimate(), "note": LINEAR_NOTE,
    })
    man.setdefault("stages", []).append({"stage": args.stage, "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                                         **stage_info})
    atomic_write_text(mp, json.dumps(man, indent=2, default=str))


def dry_run(args) -> None:
    src = check_sources()
    info = {"sources": src, "storage_estimate": storage_estimate(), "runtime_estimate": runtime_estimate()}
    try:
        c = open_context(args, need_model=False)
        info["profiles"] = [{k: p[k] for k in ("profile_index", "profile", "reproducibility_class", "eps",
                                               "eps_source", "attack_params", "cw_kwargs", "temperature",
                                               "temperature_source", "diagnostics", "raw_profile")}
                            for p in c.profiles]
        info["formal_rows"] = {"attack": len(c.formal["att"]), "defense": c.formal["dfn_rows"]}
        info["formal_attack_columns"] = c.formal["att_cols"]
        if args.out_dir:
            out = (REPO_ROOT / args.out_dir).resolve()
            st = {"clean_store_valid": validate_store(clean_paths(out), N_CLEAN, "clean_input_sha256")[0],
                  "chunks_valid": 0, "chunks_total": 0}
            for p in c.profiles:
                for mi, mod in enumerate(c.cfg["modulations"]):
                    st["chunks_total"] += 1
                    n = sum(1 for k in c.keys if k[0] == mod)
                    st["chunks_valid"] += validate_store(chunk_paths(out, p, mi, mod), n, "adversarial_sha256")[0]
            info["resume_state"] = st
    except SystemExit as exc:
        info["context_error"] = str(exc)
    print(json.dumps(info, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=p1a.DEFAULT_CONFIG)
    ap.add_argument("--formal-dir", default=p1a.DEFAULT_FORMAL_CPU_DIR)
    ap.add_argument("--out-dir", default=None, help="result dir; pass an existing one to resume")
    ap.add_argument("--stage", choices=["probe", "generate", "evaluate", "summarize", "all"], default="all")
    ap.add_argument("--probe-samples", type=int, default=2)
    ap.add_argument("--attack-temperature", type=float, default=None)
    ap.add_argument("--diagnostics", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        dry_run(args)
        return
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (REPO_ROOT / (args.out_dir or f"results/adaptive_k_effectiveness_{ts}")).resolve()
    for p in p1a.PROTECTED_RESULT_DIRS + (PHASE_A_DIR,):
        prot = (REPO_ROOT / p).resolve()
        if out == prot or prot in out.parents or out in prot.parents:
            p1a.abort(f"refusing to write inside/over {p}")
    out.mkdir(parents=True, exist_ok=True)
    p1a.log(f"[phaseB] out_dir={out} stage={args.stage}")
    c = open_context(args, need_model=True)
    info: dict = {}
    if args.stage == "probe":
        info["probe"] = stage_probe(c, out, args.probe_samples)
        write_manifest(c, out, args, {"probe_overall": info["probe"]["overall"]})
        if info["probe"]["overall"] != "PASS":
            sys.exit(2)
        return
    ev = None
    if args.stage in ("generate", "all"):
        info["generate"] = stage_generate(c, out)
        write_manifest(c, out, args, {"generate": info["generate"]})
    if args.stage in ("evaluate", "all"):
        ev = stage_evaluate(c, out)
        write_manifest(c, out, args, {"evaluate": ev})
    if args.stage in ("summarize", "all"):
        val = stage_summarize(c, out, ev)
        write_manifest(c, out, args, {"summarize_overall": val["overall"], "gates": val["gates"]})
        p1a.log(f"[phaseB] validation={val['overall']} gates={val['gates']}")
        if val["overall"] != "PASS":
            sys.exit(2)


if __name__ == "__main__":
    main()
