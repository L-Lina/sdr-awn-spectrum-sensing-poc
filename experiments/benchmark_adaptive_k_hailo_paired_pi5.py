"""
Adaptive-K Phase C: PAIRED CPU-vs-HAILO EFFECTIVENESS (Raspberry Pi 5 + Hailo-8). Standalone; nothing else is edited.

Same stored IQ as Phase B (results/adaptive_k_effectiveness_20260927_164906), same six defenses:
    none | topk10 | topk20 | topk30 | topk40 | adaptive_k_v2
adaptive_k_v2 = branch-based adaptive defense (wideband 32-level whole-sample quantization + narrowband adaptive
Top-K), frozen exactly as Phase B. No attack is generated: this script never imports the attack adapter.

Source audit (verified before writing this script)
  * Phase B stored IQ (clean_store.npy, adversarial_store/*/chunk_*.npy) is float32 [N, 2, 128] AFTER
    apply_awn_preprocess + to_awn_input, i.e. exactly the tensor handed to the classifier.
  * Formal Hailo run (experiments/run_hailo_full_matrix.py) feeds that same tensor, B=1, to
    src/adapters/hailo_awn_adapter.py:HailoAWNAdapter.infer(x) with NO further host preprocessing. The adapter adds
    a trailing axis ([1, 2, 128] -> [1, 2, 128, 1], HEF NHWC input_layer1 (2, 128, 1)) and uses FLOAT32 host
    vstreams (quantized=False), so HailoRT applies the HEF-defined UINT8 input quantization and output
    dequantization (conv10 (1, 1, 11)). Phase C calls the same adapter class, unmodified, with the same HEF
    (config hef_path, sha256 checked against the formal Hailo manifest).
  * Identity: all 2,200 Phase B clean inputs are byte-identical to the formal Hailo clean inputs
    (clean_input_sha256), and 35,381 / 52,800 stored adversarial inputs are byte-identical to the formal Hailo
    adversarial inputs (adversarial_sha256). On those rows (clean, attacked-none, fixed Top-K) the formal Hailo
    prediction is an exact historical reference; Adaptive-K never ran on Hailo, so it has none.
  * CPU predictions are REUSED from Phase B (eval_parts/*.csv == clean_per_sample.csv / attacked_per_sample.csv).
  * Defended IQ was not stored by Phase B. It is reconstructed deterministically from the stored input with the
    frozen Phase B functions themselves (benchmark_adaptive_k_effectiveness_pi5.apply_defense ->
    fft_topk_denoise / adaptive_k_v2_snr_defense, pinned defense.py sha256 + adversarial-rf commit). Proof of
    identity: Adaptive-K output sha256 == Phase B defended_sha256 (+ same branch / selected K); fixed Top-K output
    re-inferred on the CPU AWN must reproduce Phase B's CPU prediction (--cpu-verify all; verification only,
    reported CPU numbers are Phase B's).

Metric definitions per backend b (formal / Phase B definitions, each backend on its own predictions):
  clean_correct_b, attacked_correct_b, attack_success_b = clean_correct_b & ~attacked_correct_b,
  defended_correct_b, recovered_b = attack_success_b & defended_correct_b, retained_b = clean_correct_b &
  defended_correct_b, retained_nodef_b = clean_correct_b & ~attack_success_b, clean_defended_correct_b (defense on
  the clean input), clean_degraded_b = clean_correct_b & ~clean_defended_correct_b.
  conditional ASR = sum(attack_success)/sum(clean_correct); recovery = sum(recovered)/sum(attack_success);
  retention = sum(retained)/sum(clean_correct); net gain = retention - retention_nodef;
  clean degradation = sum(clean_degraded)/sum(clean_correct).
  attack transfer = P(attack_success_hailo | attack_success_cpu & clean_correct_hailo);
  recovery transfer = P(recovered_hailo | recovered_cpu & attack_success_hailo).
  For defense 'none' the defended prediction is the attacked prediction.

Terminology: spectral_ratio thresholds 3 and 10 are LINEAR, not dB; snr_label_db is the RadioML label.

Stages (pass the same --out-dir to every stage):
  preflight  Phase B / formal Hailo / source / device checks, no inference beyond opening the HEF -> preflight.json
  dry-run    deterministic subset: 2 clean samples per modulation (22) + 2 attacked samples per profile (48),
             x 6 defenses, CPU re-verification, Hailo inference -> dry_run/ (+ dry_run/validation.json)
  evaluate   full paired evaluation (2,200 x 6 clean + 52,800 x 6 attacked), resumable per part;
             REFUSES to run unless dry_run/validation.json in the same --out-dir is PASS
  summarize  paired per-sample tables, aggregations, deltas, validation.json

Run on the Pi (from the repo root):
    .venv/bin/python experiments/benchmark_adaptive_k_hailo_paired_pi5.py --stage preflight --out-dir <dir>
    .venv/bin/python experiments/benchmark_adaptive_k_hailo_paired_pi5.py --stage dry-run   --out-dir <dir>
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # no __pycache__ writes outside the Phase C output directory

import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import benchmark_adaptive_k_effectiveness_pi5 as pb  # noqa: E402 -- frozen Phase B (defenses, stores, metrics)

p1a = pb.p1a
REPO_ROOT = pb.REPO_ROOT

PHASE_B_DIR = "results/adaptive_k_effectiveness_20260927_164906"
HAILO_FORMAL_DIR = "results/hailo_full_matrix_20260920_seedaligned"
CPU_FORMAL_DIR = "results/cpu_full_matrix_20260921_seedaligned"
CONFIG_PATH = "configs/hailo_full_matrix.json"
HAILO_ADAPTER_PATH = "src/adapters/hailo_awn_adapter.py"
HAILO_RUNNER_PATH = "experiments/run_hailo_full_matrix.py"
PHASE_B_SCRIPT = "experiments/benchmark_adaptive_k_effectiveness_pi5.py"

EXPECTED_GIT_HEAD = "5aac399b93be3c0778eb8fbb4a0329b285c232b6"
EXPECTED_HEF_SHA256 = "6e54266f7e5931e432d4e5e67cd8e4a6e14863c16a94ff02a1946d21abc73dbb"
EXPECTED_HAILO_BACKEND = "Hailo-8:awn_2016_10a_c5.hef"
EXPECTED_HAILO_DEVICE = "0001:01:00.0"
EXPECTED_HAILORT_VERSION = "4.23.0"
# sha256 of the Pi-side files audited for this script (copied from the Pi on 2026-09-28)
AUDITED_SHA256 = {
    HAILO_ADAPTER_PATH: "ba8c0c84c96d31e36e81d4d09929cc351bbb6dff6435e4e2f7caea613c282dfa",
    HAILO_RUNNER_PATH: "72bd32f0161b47bbb0a2c8eb7f1780c295b8ecf244dceb1f9203a2d524dcfac2",
    CONFIG_PATH: "6c1a33e8acffa29aa210ea2de95f0854a451dbaceb0fc1d25dd60d65046716f8",
    PHASE_B_SCRIPT: "7d2989fc337925d96b0e28d521055a413b1289d6bfa12a396ed5c8e640116633",
}
HEF_INPUT_SHAPE, HEF_OUTPUT_SHAPE = (2, 128, 1), (1, 1, 11)

N_CLEAN, N_PROFILES, DEFENSES, TOPK_VALUES = pb.N_CLEAN, pb.N_PROFILES, pb.DEFENSES, pb.TOPK_VALUES
N_CHUNKS = 264
EXPECTED_PHASE_B_COUNTS = {"clean_samples": 2200, "attack_profiles": 24, "attacked_samples": 52800,
                           "clean_eval_rows": 13200, "attacked_eval_rows": 316800, "adaptive_decision_rows": 55000}
PROTECTED_DIRS = (PHASE_B_DIR, HAILO_FORMAL_DIR, CPU_FORMAL_DIR, pb.PHASE_A_DIR)
WATCHED_FILES = (CONFIG_PATH, HAILO_ADAPTER_PATH, HAILO_RUNNER_PATH, PHASE_B_SCRIPT,
                 "experiments/bench_attack_accel_pi5.py", "experiments/benchmark_adaptive_k_hailo_paired_pi5.py",
                 "src/adapters/awn_adapter.py", "src/adapters/topk_adapter.py", "src/adapters/attack_adapter.py",
                 "src/utils/config.py", "src/utils/pipeline.py", "external/adversarial-rf/util/defense.py",
                 "references/adaptive_k/adaptive_k_v2.py", "hailort.log", "pyhailort.log")
ATTACK_MODULES = ("src.adapters.attack_adapter", "util.adv_attack", "src.adapters.iq_difgsm")

# dry-run subset (deterministic): clean = per modulation (snr 18, idx 0) [mostly narrowband] and (snr -10, idx 1)
# [mostly wideband]; attacked = per profile index p: modulation[p % 11] at (snr 18, idx p % 10) and
# (snr 2, idx (p + 5) % 10)
DRY_CLEAN_POINTS = ((18, 0), (-10, 1))
CLEAN_AGREEMENT_PLAUSIBLE_MIN = 0.70  # formal CPU-vs-Hailo clean agreement was 90.7 % (1996 / 2200)
MIN_DISTINCT_HAILO_CLASSES = 3

FLAGS = ["clean_correct", "attacked_correct", "attack_success", "defended_correct", "recovered", "retained",
         "retained_nodef", "clean_defended_correct", "clean_degraded"]
PREDS = ["clean_pred", "attacked_pred", "defended_pred"]
ID_FIELDS = ["global_index", "attack", "attack_profile", "profile", "attack_family", "modulation", "snr_label_db",
             "sample_index", "label", "defense", "adaptive_branch", "adaptive_selected_k", "adaptive_clean_branch",
             "branch_transition", "input_sha256", "defended_sha256"]
ATT_FIELDS = (ID_FIELDS + [f"{f}_cpu" for f in PREDS + FLAGS] + [f"{f}_hailo" for f in PREDS + FLAGS]
              + ["cpu_hailo_agree", "cpu_hailo_agree_attacked", "hailo_formal_identity", "hailo_formal_pred",
                 "hailo_formal_match", "cpu_recomputed_pred", "cpu_recompute_match", "adaptive_defended_sha_match",
                 "defense_ms", "hailo_infer_ms"])
CLEAN_FLAGS = ["clean_correct", "clean_defended_correct", "clean_degraded", "clean_improved"]
CLEAN_FIELDS = (["global_index", "modulation", "snr_label_db", "sample_index", "label", "defense", "adaptive_branch",
                 "adaptive_selected_k", "input_sha256", "defended_sha256"]
                + [f"{f}_cpu" for f in ["clean_pred", "defended_pred"] + CLEAN_FLAGS]
                + [f"{f}_hailo" for f in ["clean_pred", "defended_pred"] + CLEAN_FLAGS]
                + ["cpu_hailo_agree", "hailo_formal_identity", "hailo_formal_pred", "hailo_formal_match",
                   "cpu_recomputed_pred", "cpu_recompute_match", "adaptive_defended_sha_match", "defense_ms",
                   "hailo_infer_ms", "in_dry_run_clean_subset"])

b2s, s2b, sha_arr, sha_file, pct = pb.b2s, pb.s2b, pb.sha_arr, pb.sha_file, pb.pct
read_csv, write_csv, atomic_write_text = pb.read_csv, pb.write_csv, pb.atomic_write_text


def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def log(msg: str) -> None:
    p1a.log(f"[phaseC] {msg}")


# =========================================================================== write guard (no Phase B modification)
def guard_snapshot(out: Path) -> dict:
    """stat of every file under the protected result dirs + sha256 of watched sources + git state."""
    snap: dict = {"protected": {}, "watched": {}}
    for d in PROTECTED_DIRS:
        root = REPO_ROOT / d
        files = {}
        if root.exists():
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    st = p.stat()
                    files[str(p.relative_to(root))] = [st.st_size, st.st_mtime_ns]
        snap["protected"][d] = files if root.exists() else None
    for f in WATCHED_FILES:
        p = REPO_ROOT / f
        snap["watched"][f] = sha_file(p) if p.is_file() else None
    snap["git_status"] = p1a.run_cmd(["git", "status", "--porcelain"])
    diff = p1a.run_cmd(["git", "diff"])
    snap["git_diff_sha256"] = None if diff is None else hashlib.sha256(diff.encode()).hexdigest()
    return snap


def guard_compare(pre: dict, post: dict) -> dict:
    changed = {}
    for d in PROTECTED_DIRS:
        a, b = pre["protected"][d], post["protected"][d]
        if a != b:
            a, b = a or {}, b or {}
            changed[d] = sorted({k for k in set(a) | set(b) if a.get(k) != b.get(k)})[:20]
    watched = sorted(f for f in WATCHED_FILES if pre["watched"][f] != post["watched"][f])
    return {"protected_dirs_unchanged": not changed, "protected_changes": changed,
            "watched_files_unchanged": not watched, "watched_changes": watched,
            "git_status_unchanged": pre["git_status"] == post["git_status"],
            "git_diff_unchanged": pre["git_diff_sha256"] == post["git_diff_sha256"]}


# =========================================================================== context
class Ctx:
    pass


def load_profiles(cfg: dict) -> List[dict]:
    rows = read_csv(REPO_ROOT / PHASE_B_DIR / "attack_profiles.csv")
    prof = [{"profile_index": int(r["profile_index"]), "attack": r["attack"], "attack_profile": r["attack_profile"],
             "profile": r["profile"]} for r in rows]
    from_cfg = [f"{str(a.get('name', '')).lower()}/{p.get('id')}" for a in cfg["attacks"] for p in a["profiles"]]
    if [p["profile"] for p in prof] != from_cfg or [p["profile_index"] for p in prof] != list(range(len(prof))):
        p1a.abort("Phase B attack_profiles.csv does not match the formal config profile order")
    return prof


def load_hailo_formal() -> dict:
    d = REPO_ROOT / HAILO_FORMAL_DIR
    base, att, dfn = {}, {}, {}
    with (d / "base_results.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            base[(r["modulation"], int(r["snr_db"]), int(r["sample_index"]))] = {
                "clean_pred": int(r["clean_pred"]), "sha": r["clean_input_sha256"], "backend": r["hilo_backend"]}
    with (d / "attack_results.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            att[(r["modulation"], int(r["snr_db"]), int(r["sample_index"]), r["attack"], r["attack_profile"])] = {
                "attacked_pred": int(r["attacked_pred"]), "sha": r["adversarial_sha256"]}
    with (d / "defense_results.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            dfn[(r["modulation"], int(r["snr_db"]), int(r["sample_index"]), r["attack"], r["attack_profile"],
                 int(r["topk"]))] = (int(r["defended_pred"]), int(r["clean_topk_pred"]))
    return {"base": base, "att": att, "dfn": dfn, "manifest": json.loads((d / "manifest.json").read_text())}


def open_context(args, out: Path) -> Ctx:
    c = Ctx()
    c.out, c.pb_dir = out, REPO_ROOT / PHASE_B_DIR
    c.cfg = json.loads((REPO_ROOT / CONFIG_PATH).read_text())
    c.keys = [(m, int(s), int(i)) for m in c.cfg["modulations"] for s in c.cfg["snrs"] for i in c.cfg["sample_indices"]]
    if len(c.keys) != N_CLEAN:
        p1a.abort(f"config grid has {len(c.keys)} samples, expected {N_CLEAN}")
    c.gi_of = {k: gi for gi, k in enumerate(c.keys)}
    c.profiles = load_profiles(c.cfg)
    c.cpu_verify = args.cpu_verify == "all"
    c.bad = {}  # gate -> list of failure messages
    c.hailo = None
    c.hailo_preds: set = set()
    c.hclean: Dict[int, Dict[str, int]] = {}
    return c


def expect(c: Ctx, cond: bool, gate: str, msg: str) -> bool:
    lst = c.bad.setdefault(gate, [])
    if not cond:
        lst.append(msg)
    return cond


def open_torch_side(c: Ctx) -> None:
    """Frozen Phase B defense functions (pinned defense.py) + CPU AWN for verification. No attack adapter."""
    src = pb.check_sources()
    c.src = src
    for k in ("reference_ok", "defense_py_ok", "adversarial_rf_commit_ok", "phase_a_ok"):
        if not src[k]:
            p1a.abort(f"source check failed: {k} ({json.dumps(src)})")
    import torch
    import inspect
    c.torch = torch
    advrf = str(REPO_ROOT / "external" / "adversarial-rf")
    if advrf not in sys.path:
        sys.path.insert(0, advrf)
    import util.defense as D  # noqa: E402
    c.D = D
    sig = inspect.signature(D.adaptive_k_v2_snr_defense).parameters
    c.adaptive_defaults = {}
    for name, val in (("flatness_threshold", pb.FLATNESS_THRESHOLD), ("quant_levels", pb.QUANT_LEVELS),
                      ("ratio_thresh", pb.RATIO_THRESH), ("pilot_k", pb.PILOT_K), ("snr_low", pb.SPECTRAL_RATIO_LOW),
                      ("snr_high", pb.SPECTRAL_RATIO_HIGH), ("k_max_low_snr", pb.K_CAP_LOW_RATIO),
                      ("k_max_high_snr", pb.K_CAP_HIGH_RATIO)):
        c.adaptive_defaults[name] = sig[name].default
        if sig[name].default != val:
            p1a.abort(f"adaptive_k_v2_snr_defense default {name}={sig[name].default} != frozen {val}")
    c.ak_checks = {"calls": 0, "bit_identical": 0}  # updated by pb.adaptive_k
    if c.cpu_verify:
        from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE
        c.REAL_MODEL = _REAL_MODEL_SOURCE
        c.awn = AWNModelAdapter(checkpoint_path=str(REPO_ROOT / c.cfg["checkpoint_path"]), device="cpu")
        if c.awn.model is None or c.awn.backend_name != _REAL_MODEL_SOURCE or c.awn.status != "ok":
            p1a.abort("CPU AWN backend is not real")


def open_hailo(c: Ctx) -> None:
    from src.adapters.hailo_awn_adapter import HailoAWNAdapter  # the formal Hailo path, unmodified
    hef = Path(c.cfg["hef_path"]).resolve()
    c.hailo = HailoAWNAdapter(str(hef))
    if c.hailo.status != "ok" or c.hailo.backend_name != EXPECTED_HAILO_BACKEND:
        p1a.abort(f"Hailo adapter not ok: status={c.hailo.status} backend={c.hailo.backend_name}")
    c.hailo_io = hailo_io_info(c.hailo)


def hailo_io_info(h) -> dict:
    def vinfo(v) -> dict:
        d = {"name": getattr(v, "name", None), "shape": list(getattr(v, "shape", []))}
        fmt = getattr(v, "format", None)
        d["hef_format_type"] = str(getattr(fmt, "type", None))
        d["hef_format_order"] = str(getattr(fmt, "order", None))
        q = getattr(v, "quant_info", None)
        for a in ("qp_scale", "qp_zp", "limvals_min", "limvals_max"):
            try:
                d[a] = float(getattr(q, a))
            except Exception:  # noqa: BLE001 - metadata only
                d[a] = None
        return d

    def host_fmt(params) -> dict:
        out = {}
        try:
            for name, prm in dict(params).items():
                ubf = getattr(prm, "user_buffer_format", None)
                out[str(name)] = {"type": str(getattr(ubf, "type", None)), "order": str(getattr(ubf, "order", None))}
        except Exception as exc:  # noqa: BLE001 - metadata only
            out["error"] = repr(exc)
        return out

    hpf = h._hpf
    return {"hef_path": h.hef_path, "input": vinfo(h.input_info), "output": vinfo(h.output_info),
            "host_input_params": host_fmt(h.input_params), "host_output_params": host_fmt(h.output_params),
            "host_layout": "x[..., np.newaxis]: [N, 2, 128] -> [N, 2, 128, 1] (adapter source)",
            "host_format": "FLOAT32, quantized=False (HailoRT applies HEF quantization / dequantization)",
            "hailo_platform_version": str(getattr(hpf, "__version__", None))}


def close_hailo(c: Ctx) -> None:
    if c.hailo is not None:
        c.hailo.close()
        c.hailo = None


# =========================================================================== inference / defenses
def check_x(x: np.ndarray, where: str) -> None:
    if x.shape != (1, 2, pb.T_LEN) or x.dtype != np.float32 or not np.isfinite(x).all():
        p1a.abort(f"invalid IQ at {where}: shape={x.shape} dtype={x.dtype}")


def hailo_pred(c: Ctx, x: np.ndarray, where: str) -> Tuple[int, float]:
    check_x(x, where)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):  # the adapter prints one line per call
        logits, meta = c.hailo.infer(x)
    ms = (time.perf_counter() - t0) * 1e3
    if meta.get("awn_status") != "ok" or meta.get("awn_backend") != EXPECTED_HAILO_BACKEND:
        p1a.abort(f"Hailo inference fallback/error at {where}: {meta}")
    if logits.shape != (1, 11) or not np.isfinite(logits).all():
        p1a.abort(f"Hailo logits invalid at {where}: shape={logits.shape}")
    pred = int(np.argmax(logits[0]))
    c.hailo_preds.add(pred)
    return pred, ms


def cpu_pred(c: Ctx, x: np.ndarray) -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        pred, _ = pb.infer_pred(c, x)
    return pred


def branch_fields(info: Optional[dict]) -> Tuple[str, str]:
    if not info:
        return "", ""
    return ("wideband" if info["wideband"] else "narrowband"), str(info["selected_k"])


def check_adaptive(c: Ctx, where: str, y: np.ndarray, info: dict, ak_row: Optional[dict], cpu_row: dict) -> str:
    """Adaptive-K reconstruction == Phase B (output sha256, branch, K); wideband has no K, narrowband has K."""
    br, k = branch_fields(info)
    if br == "wideband":
        expect(c, k == "", "wideband_rows_have_no_k", f"{where}: wideband selected_k={k!r}")
    else:
        expect(c, k.isdigit() and int(k) >= 1, "narrowband_rows_have_k", f"{where}: narrowband selected_k={k!r}")
    if ak_row is None:
        expect(c, False, "defense_semantics_ok", f"{where}: Phase B adaptive decision row missing")
        return ""
    sha_ok = sha_arr(y) == ak_row["defended_sha256"]
    expect(c, sha_ok, "defense_semantics_ok", f"{where}: adaptive output sha256 != Phase B defended_sha256")
    expect(c, br == ak_row["branch"] and k == ak_row["selected_k"], "adaptive_semantics_ok",
           f"{where}: branch/K {br}/{k} != Phase B {ak_row['branch']}/{ak_row['selected_k']}")
    expect(c, br == cpu_row["adaptive_branch"] and k == cpu_row["adaptive_selected_k"], "adaptive_semantics_ok",
           f"{where}: branch/K {br}/{k} != Phase B per-sample {cpu_row['adaptive_branch']}/"
           f"{cpu_row['adaptive_selected_k']}")
    return b2s(sha_ok)


# =========================================================================== per-sample evaluation
def eval_clean_sample(c: Ctx, gi: int, x: np.ndarray, cm: dict, cpu_rows: Dict[tuple, dict],
                      ak_row: Optional[dict], hf: dict, in_subset: bool) -> List[dict]:
    k = c.keys[gi]
    where = f"clean gi={gi} {k}"
    label = int(cm["label"])
    xsha = sha_arr(x)
    expect(c, (int(cm["global_index"]), cm["modulation"], int(cm["snr_label_db"]), int(cm["sample_index"]))
           == (gi,) + k and xsha == cm["clean_input_sha256"], "sample_identity_ok", f"{where}: clean store identity")
    fb = hf["base"].get(k)
    exact = fb is not None and fb["sha"] == xsha
    expect(c, exact, "sample_identity_ok", f"{where}: formal Hailo clean_input_sha256 differs")
    h0, hms0 = hailo_pred(c, x, where)
    p0 = c.profiles[0]
    rows = []
    for d in DEFENSES:
        cr = cpu_rows.get((gi, d))
        if not expect(c, cr is not None, "cpu_rows_found", f"{where} {d}: Phase B CPU row missing"):
            continue
        expect(c, (cr["modulation"], int(cr["snr_label_db"]), int(cr["sample_index"]), int(cr["label"])) == k + (label,),
               "sample_identity_ok", f"{where} {d}: Phase B CPU row identity")
        y, dms, info = pb.apply_defense(c, d, x, where)
        hp, hms = (h0, hms0) if d == "none" else hailo_pred(c, y, f"{where} {d}")
        br, kk = branch_fields(info)
        ak_match = check_adaptive(c, f"{where} {d}", y, info, ak_row, cr) if info else ""
        cc_pred, cd_pred = int(cr["clean_pred"]), int(cr["defended_pred"])
        rc = ""
        if c.cpu_verify:
            rc = cpu_pred(c, y)
            expect(c, rc == cd_pred, "defense_semantics_ok", f"{where} {d}: CPU re-inference {rc} != Phase B {cd_pred}")
        f_pred = ""
        if exact and d == "none":
            f_pred = fb["clean_pred"]
        elif exact and d.startswith("topk"):
            f_pred = hf["dfn"][k + (p0["attack"], p0["attack_profile"], int(d[4:]))][1]
        f_match = "" if f_pred == "" else b2s(hp == f_pred)
        if f_pred != "":
            expect(c, hp == f_pred, "hailo_matches_formal_where_identical",
                   f"{where} {d}: Hailo {hp} != formal Hailo {f_pred} on identical input")
        ccpu, dcpu, chl, dhl = cc_pred == label, cd_pred == label, h0 == label, hp == label
        expect(c, cr["clean_correct"] == b2s(ccpu) and cr["clean_defended_correct"] == b2s(dcpu), "cpu_rows_found",
               f"{where} {d}: Phase B CPU flags inconsistent with predictions")
        rows.append({"global_index": gi, "modulation": k[0], "snr_label_db": k[1], "sample_index": k[2],
                     "label": label, "defense": d, "adaptive_branch": br, "adaptive_selected_k": kk,
                     "input_sha256": xsha, "defended_sha256": sha_arr(y),
                     "clean_pred_cpu": cc_pred, "defended_pred_cpu": cd_pred, "clean_correct_cpu": b2s(ccpu),
                     "clean_defended_correct_cpu": b2s(dcpu), "clean_degraded_cpu": b2s(ccpu and not dcpu),
                     "clean_improved_cpu": b2s((not ccpu) and dcpu),
                     "clean_pred_hailo": h0, "defended_pred_hailo": hp, "clean_correct_hailo": b2s(chl),
                     "clean_defended_correct_hailo": b2s(dhl), "clean_degraded_hailo": b2s(chl and not dhl),
                     "clean_improved_hailo": b2s((not chl) and dhl), "cpu_hailo_agree": b2s(cd_pred == hp),
                     "hailo_formal_identity": ("exact_input" if exact else "input_differs") if d != "adaptive_k_v2"
                     else "no_formal_condition", "hailo_formal_pred": f_pred, "hailo_formal_match": f_match,
                     "cpu_recomputed_pred": rc, "cpu_recompute_match": "" if rc == "" else b2s(rc == cd_pred),
                     "adaptive_defended_sha_match": ak_match, "defense_ms": dms, "hailo_infer_ms": hms,
                     "in_dry_run_clean_subset": b2s(in_subset)})
    c.hclean[gi] = {r["defense"]: int(r["defended_pred_hailo"]) for r in rows}
    return rows


def eval_attacked_sample(c: Ctx, p: dict, x: np.ndarray, am: dict, cpu_rows: Dict[tuple, dict],
                         ak_row: Optional[dict], hf: dict) -> List[dict]:
    gi = int(am["global_index"])
    k = c.keys[gi]
    where = f"{p['profile']} gi={gi} {k}"
    label = int(am["label"])
    xsha = sha_arr(x)
    expect(c, am["profile"] == p["profile"] and (am["modulation"], int(am["snr_label_db"]), int(am["sample_index"])) == k
           and xsha == am["adversarial_sha256"], "sample_identity_ok", f"{where}: adversarial store identity")
    fa = hf["att"].get(k + (p["attack"], p["attack_profile"]))
    exact = fa is not None and fa["sha"] == xsha
    hc = c.hclean.get(gi)
    if hc is None:
        p1a.abort(f"{where}: Hailo clean results for gi={gi} not available")
    ha, hams = hailo_pred(c, x, where)
    rows = []
    for d in DEFENSES:
        cr = cpu_rows.get((gi, d))
        if not expect(c, cr is not None, "cpu_rows_found", f"{where} {d}: Phase B CPU row missing"):
            continue
        expect(c, cr["profile"] == p["profile"] and cr["adversarial_sha256"] == xsha and int(cr["label"]) == label
               and (cr["modulation"], int(cr["snr_label_db"]), int(cr["sample_index"])) == k,
               "sample_identity_ok", f"{where} {d}: Phase B CPU row identity")
        y, dms, info = pb.apply_defense(c, d, x, where)
        hp, hms = (ha, hams) if d == "none" else hailo_pred(c, y, f"{where} {d}")
        br, kk = branch_fields(info)
        ak_match = check_adaptive(c, f"{where} {d}", y, info, ak_row, cr) if info else ""
        # CPU (reused from Phase B)
        c_clean, c_att, c_def = int(cr["clean_pred"]), int(cr["attacked_pred"]), int(cr["defended_pred"])
        cc, ac, dc = c_clean == label, c_att == label, c_def == label
        asu = cc and not ac
        cdc = s2b(cr["clean_defended_correct"])
        cpu_flags = {"clean_correct": cc, "attacked_correct": ac, "attack_success": asu, "defended_correct": dc,
                     "recovered": asu and dc, "retained": cc and dc, "retained_nodef": cc and not asu,
                     "clean_defended_correct": cdc, "clean_degraded": cc and not cdc}
        expect(c, all(cr[f] == b2s(v) for f, v in cpu_flags.items()), "cpu_rows_found",
               f"{where} {d}: Phase B CPU flags inconsistent with predictions")
        rc = ""
        if c.cpu_verify:
            rc = cpu_pred(c, y)
            expect(c, rc == c_def, "defense_semantics_ok", f"{where} {d}: CPU re-inference {rc} != Phase B {c_def}")
        # Hailo
        h_clean, h_cdef = hc["none"], hc[d]
        hcc, hac, hdc = h_clean == label, ha == label, hp == label
        hasu = hcc and not hac
        hcdc = h_cdef == label
        hl_flags = {"clean_correct": hcc, "attacked_correct": hac, "attack_success": hasu, "defended_correct": hdc,
                    "recovered": hasu and hdc, "retained": hcc and hdc, "retained_nodef": hcc and not hasu,
                    "clean_defended_correct": hcdc, "clean_degraded": hcc and not hcdc}
        # historical formal Hailo reference (only where the input is byte-identical)
        f_pred = ""
        if exact and d == "none":
            f_pred = fa["attacked_pred"]
        elif exact and d.startswith("topk"):
            f_pred = hf["dfn"][k + (p["attack"], p["attack_profile"], int(d[4:]))][0]
        if f_pred != "":
            expect(c, hp == f_pred, "hailo_matches_formal_where_identical",
                   f"{where} {d}: Hailo {hp} != formal Hailo {f_pred} on identical input")
        row = {"global_index": gi, "attack": p["attack"], "attack_profile": p["attack_profile"], "profile": p["profile"],
               "attack_family": p["attack"], "modulation": k[0], "snr_label_db": k[1], "sample_index": k[2],
               "label": label, "defense": d, "adaptive_branch": br, "adaptive_selected_k": kk,
               "adaptive_clean_branch": cr["adaptive_clean_branch"], "branch_transition": cr["branch_transition"],
               "input_sha256": xsha, "defended_sha256": sha_arr(y),
               "clean_pred_cpu": c_clean, "attacked_pred_cpu": c_att, "defended_pred_cpu": c_def,
               "clean_pred_hailo": h_clean, "attacked_pred_hailo": ha, "defended_pred_hailo": hp,
               "cpu_hailo_agree": b2s(c_def == hp), "cpu_hailo_agree_attacked": b2s(c_att == ha),
               "hailo_formal_identity": ("exact_input" if exact else "input_differs") if d != "adaptive_k_v2"
               else "no_formal_condition", "hailo_formal_pred": f_pred,
               "hailo_formal_match": "" if f_pred == "" else b2s(hp == f_pred),
               "cpu_recomputed_pred": rc, "cpu_recompute_match": "" if rc == "" else b2s(rc == c_def),
               "adaptive_defended_sha_match": ak_match, "defense_ms": dms, "hailo_infer_ms": hms}
        row.update({f"{f}_cpu": b2s(v) for f, v in cpu_flags.items()})
        row.update({f"{f}_hailo": b2s(v) for f, v in hl_flags.items()})
        rows.append(row)
    return rows


# =========================================================================== Phase B inputs
def pb_part_name(p: dict) -> str:
    return f"{p['profile_index']:02d}_{p['attack']}__{p['attack_profile']}"


def load_pb_clean(c: Ctx) -> Tuple[np.ndarray, List[dict], Dict[tuple, dict], Dict[int, dict]]:
    cp = pb.clean_paths(c.pb_dir)
    ok, why = pb.validate_store(cp, N_CLEAN, "clean_input_sha256")
    if not ok:
        p1a.abort(f"Phase B clean store invalid: {why}")
    ep = pb.eval_paths(c.pb_dir, "clean")
    if not pb.part_valid(ep, N_CLEAN * len(DEFENSES), N_CLEAN):
        p1a.abort("Phase B clean evaluation part invalid")
    rows = {(int(r["global_index"]), r["defense"]): r for r in read_csv(ep["rows"])}
    ak = {int(r["global_index"]): r for r in read_csv(ep["ak"])}
    return np.load(cp["npy"], allow_pickle=False), read_csv(cp["meta"]), rows, ak


def load_pb_profile(c: Ctx, p: dict, mods: Optional[set] = None):
    """Stored adversarial IQ + Phase B CPU rows + Phase B adaptive decisions for one profile."""
    arrs, metas = [], []
    for mi, mod in enumerate(c.cfg["modulations"]):
        if mods is not None and mod not in mods:
            continue
        cp = pb.chunk_paths(c.pb_dir, p, mi, mod)
        n = sum(1 for k in c.keys if k[0] == mod)
        ok, why = pb.validate_store(cp, n, "adversarial_sha256")
        if not ok:
            p1a.abort(f"Phase B adversarial chunk {cp['npy']} invalid ({why})")
        arrs.append(np.load(cp["npy"], allow_pickle=False))
        metas.extend(read_csv(cp["meta"]))
    ep = pb.eval_paths(c.pb_dir, pb_part_name(p))
    if not pb.part_valid(ep, N_CLEAN * len(DEFENSES), N_CLEAN):
        p1a.abort(f"Phase B evaluation part for {p['profile']} invalid")
    rows = {(int(r["global_index"]), r["defense"]): r for r in read_csv(ep["rows"])}
    ak = {int(r["global_index"]): r for r in read_csv(ep["ak"])}
    return np.concatenate(arrs), metas, rows, ak


def count_rows(path: Path) -> Tuple[List[str], int]:
    with path.open(newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        return header, sum(1 for _ in rd)


def rows_equal_stream(a: Path, parts: List[Path]) -> bool:
    """Row-level equality of a Phase B per-sample CSV with the concatenation of its evaluation parts."""
    with a.open(newline="") as fa:
        ra = csv.reader(fa)
        ha = next(ra)
        for pth in parts:
            with pth.open(newline="") as fp:
                rp = csv.reader(fp)
                if next(rp) != ha:
                    return False
                for row in rp:
                    if next(ra, None) != row:
                        return False
        return next(ra, None) is None


# =========================================================================== preflight
def stage_preflight(c: Ctx) -> dict:
    g: Dict[str, bool] = {}
    info: dict = {"utc": now_utc()}
    pbd = c.pb_dir
    # Phase B validation / counts / provenance
    val = json.loads((pbd / "validation.json").read_text())
    g["phaseB_validation_pass"] = val.get("overall") == "PASS" and all(val.get("gates", {}).values())
    g["phaseB_counts_match"] = val.get("row_counts") == EXPECTED_PHASE_B_COUNTS
    man = json.loads((pbd / "manifest.json").read_text())
    head = (p1a.run_cmd(["git", "rev-parse", "HEAD"]) or "").strip()
    info["git_head"] = head
    info["phaseB_git_head"] = man.get("environment", {}).get("git_head")
    g["git_head_matches_expected"] = head == EXPECTED_GIT_HEAD == info["phaseB_git_head"]
    cfg_sha = sha_file(REPO_ROOT / CONFIG_PATH)
    # clean store + adversarial chunks
    cp = pb.clean_paths(pbd)
    ok, why = pb.validate_store(cp, N_CLEAN, "clean_input_sha256")
    g["clean_store_2200_valid"] = ok
    info["clean_store"] = why
    markers = sorted((pbd / "adversarial_store").glob("*/*.COMPLETE.json"))
    info["adversarial_complete_markers"] = len(markers)
    n_valid, n_samples = 0, 0
    for p in c.profiles:
        for mi, mod in enumerate(c.cfg["modulations"]):
            n = sum(1 for k in c.keys if k[0] == mod)
            ok_c, _ = pb.validate_store(pb.chunk_paths(pbd, p, mi, mod), n, "adversarial_sha256")
            n_valid += ok_c
            n_samples += n if ok_c else 0
    info["adversarial_chunks_valid"], info["attacked_samples_valid"] = n_valid, n_samples
    g["adversarial_264_complete"] = len(markers) == N_CHUNKS and n_valid == N_CHUNKS
    g["attack_profiles_24"] = len(c.profiles) == N_PROFILES
    g["attacked_samples_52800"] = n_samples == N_CLEAN * N_PROFILES
    # Phase B CPU per-sample files
    try:
        hc, nc = count_rows(pbd / "clean_per_sample.csv")
        ha, na = count_rows(pbd / "attacked_per_sample.csv")
        hk, nk = count_rows(pbd / "adaptive_decisions.csv")
        info["phaseB_rows"] = {"clean": nc, "attacked": na, "adaptive": nk}
        parts_ok = all(pb.part_valid(pb.eval_paths(pbd, pb_part_name(p)), N_CLEAN * len(DEFENSES), N_CLEAN)
                       for p in c.profiles) and pb.part_valid(pb.eval_paths(pbd, "clean"), N_CLEAN * len(DEFENSES),
                                                                N_CLEAN)
        g["cpu_per_sample_files_readable"] = (hc == pb.CLEAN_EVAL_FIELDS and ha == pb.ATT_EVAL_FIELDS
                                              and hk == pb.AK_FIELDS and nc == 13200 and na == 316800 and nk == 55000
                                              and parts_ok)
        g["cpu_parts_equal_per_sample_csv"] = (
            rows_equal_stream(pbd / "clean_per_sample.csv", [pb.eval_paths(pbd, "clean")["rows"]])
            and rows_equal_stream(pbd / "attacked_per_sample.csv",
                                  [pb.eval_paths(pbd, pb_part_name(p))["rows"] for p in c.profiles]))
    except Exception as exc:  # noqa: BLE001
        info["phaseB_rows_error"] = repr(exc)
        g["cpu_per_sample_files_readable"] = g["cpu_parts_equal_per_sample_csv"] = False
    # formal Hailo artifacts + identity join
    hf = load_hailo_formal()
    hm = hf["manifest"]
    g["formal_hailo_complete"] = (hm.get("run_status") == "complete" and hm.get("completed_base_rows") == 2200
                                  and hm.get("completed_attack_rows") == 52800
                                  and hm.get("completed_defense_rows") == 211200)
    g["config_matches_phaseB_and_formal_hailo"] = (hm.get("config") == c.cfg
                                                   and cfg_sha == man.get("formal_config", {}).get("sha256"))
    cmeta = read_csv(cp["meta"])
    clean_id = sum(1 for m in cmeta if hf["base"].get(c.keys[int(m["global_index"])], {}).get("sha")
                   == m["clean_input_sha256"])
    info["clean_inputs_identical_to_formal_hailo"] = clean_id
    g["clean_inputs_identical_to_formal_hailo"] = clean_id == N_CLEAN
    ident = {}
    for p in c.profiles:
        n_eq = 0
        for mi, mod in enumerate(c.cfg["modulations"]):
            for m in read_csv(pb.chunk_paths(pbd, p, mi, mod)["meta"]):
                fa = hf["att"].get(c.keys[int(m["global_index"])] + (p["attack"], p["attack_profile"]))
                n_eq += fa is not None and fa["sha"] == m["adversarial_sha256"]
        ident[p["profile"]] = n_eq
    info["attacked_inputs_identical_to_formal_hailo"] = ident
    info["attacked_inputs_identical_total"] = sum(ident.values())
    # sources / HEF / adapter
    hef = Path(c.cfg["hef_path"]).resolve()
    info["hef_path"] = str(hef)
    g["hef_exists"] = hef.is_file()
    info["hef_sha256"] = sha_file(hef) if hef.is_file() else None
    g["hef_sha256_matches_formal"] = info["hef_sha256"] == EXPECTED_HEF_SHA256 == hm.get("hef_sha256")
    g["hef_path_matches_formal"] = str(hm.get("config", {}).get("hef_path")) == c.cfg["hef_path"]
    info["source_sha256"] = {f: (sha_file(REPO_ROOT / f) if (REPO_ROOT / f).is_file() else None) for f in AUDITED_SHA256}
    info["source_sha256_matches_audited"] = {f: info["source_sha256"][f] == h for f, h in AUDITED_SHA256.items()}
    g["hailo_adapter_source_audited"] = info["source_sha256_matches_audited"][HAILO_ADAPTER_PATH]
    formal_backend = {v["backend"] for v in hf["base"].values()}
    g["formal_backend_matches"] = formal_backend == {EXPECTED_HAILO_BACKEND}
    # device
    scan = p1a.run_cmd(["hailortcli", "scan"]) or ""
    ident_fw = p1a.run_cmd(["hailortcli", "fw-control", "identify"]) or ""
    info["hailortcli_scan"], info["hailortcli_identify"] = scan, ident_fw
    g["hailo_device_available"] = EXPECTED_HAILO_DEVICE in scan
    g["hailort_version_4_23_0"] = EXPECTED_HAILORT_VERSION in ident_fw
    # torch side: frozen defenses (+ CPU AWN for verification)
    open_torch_side(c)
    info["sources"], info["adaptive_defaults"] = c.src, c.adaptive_defaults
    g["defense_sources_pinned"] = all(c.src[k] for k in ("reference_ok", "defense_py_ok", "adversarial_rf_commit_ok",
                                                          "phase_a_ok"))
    # Hailo adapter (formal path): open, record IO metadata, close
    try:
        open_hailo(c)
        io_ = c.hailo_io
        info["hailo_io"] = io_
        g["hailo_adapter_ok"] = True
        g["input_shape_dtype_ok"] = (tuple(io_["input"]["shape"]) == HEF_INPUT_SHAPE
                                     and tuple(io_["output"]["shape"]) == HEF_OUTPUT_SHAPE
                                     and np.load(cp["npy"], mmap_mode="r").dtype == np.float32
                                     and np.load(cp["npy"], mmap_mode="r").shape == (N_CLEAN, 2, pb.T_LEN))
    except Exception as exc:  # noqa: BLE001
        info["hailo_error"] = repr(exc)
        g["hailo_adapter_ok"] = g["input_shape_dtype_ok"] = False
    finally:
        close_hailo(c)
    g["no_attack_generation"] = not any(m in sys.modules for m in ATTACK_MODULES)
    info["torchattacks_imported"] = "torchattacks" in sys.modules
    return {"gates": g, "info": info}


# =========================================================================== dry-run
def dry_subset(c: Ctx) -> Tuple[List[int], List[Tuple[dict, int]]]:
    mods = list(c.cfg["modulations"])
    clean = [c.gi_of[(m, s, i)] for m in mods for (s, i) in DRY_CLEAN_POINTS]
    att = []
    for p in c.profiles:
        pi = p["profile_index"]
        m = mods[pi % len(mods)]
        att.append((p, c.gi_of[(m, 18, pi % 10)]))
        att.append((p, c.gi_of[(m, 2, (pi + 5) % 10)]))
    return clean, att


def stage_dry_run(c: Ctx, pre: dict) -> dict:
    failed = [k for k, v in pre["gates"].items() if not v]
    if failed:
        return {"gates": dict(pre["gates"]), "info": {"skipped": f"preflight gates failed: {failed}"}}
    d = c.out / "dry_run"
    d.mkdir(parents=True, exist_ok=True)
    clean_gis, att_pairs = dry_subset(c)
    hf = load_hailo_formal()
    clean, cmeta, cpu_clean, ak_clean = load_pb_clean(c)
    open_hailo(c)
    try:
        need = sorted(set(clean_gis) | {gi for _, gi in att_pairs})
        clean_rows = []
        for gi in need:
            clean_rows += eval_clean_sample(c, gi, clean[gi:gi + 1], cmeta[gi], cpu_clean, ak_clean.get(gi), hf,
                                            gi in clean_gis)
        att_rows = []
        for p in c.profiles:
            gis = [gi for (pp, gi) in att_pairs if pp is p]
            arr, metas, cpu_rows, ak = load_pb_profile(c, p, mods={c.keys[gi][0] for gi in gis})
            pos = {int(m["global_index"]): j for j, m in enumerate(metas)}
            for gi in gis:
                j = pos[gi]
                att_rows += eval_attacked_sample(c, p, arr[j:j + 1], metas[j], cpu_rows, ak.get(gi), hf)
    finally:
        close_hailo(c)
    write_csv(d / "dry_run_clean_per_sample.csv", CLEAN_FIELDS, clean_rows)
    write_csv(d / "dry_run_attacked_per_sample.csv", ATT_FIELDS, att_rows)
    summarize_rows(c, [r for r in clean_rows if r["in_dry_run_clean_subset"] == "True"], att_rows, d)
    sub_clean = [r for r in clean_rows if r["in_dry_run_clean_subset"] == "True" and r["defense"] == "none"]
    agree = pct(sum(1 for r in sub_clean if r["cpu_hailo_agree"] == "True"), len(sub_clean)) / 100.0
    ak_rows = [r for r in clean_rows + att_rows if r["defense"] == "adaptive_k_v2"]
    branches = {r["adaptive_branch"] for r in ak_rows}
    hist = [r for r in clean_rows + att_rows if r["hailo_formal_match"] != ""]
    g = dict(pre["gates"])
    g.update({
        "sample_identity_ok": not c.bad.get("sample_identity_ok"),
        "cpu_rows_found": not c.bad.get("cpu_rows_found"),
        "defense_semantics_ok": not c.bad.get("defense_semantics_ok") and c.cpu_verify,
        "adaptive_semantics_ok": (not c.bad.get("adaptive_semantics_ok")
                                  and c.ak_checks["calls"] == c.ak_checks["bit_identical"] > 0),
        "wideband_rows_have_no_k": not c.bad.get("wideband_rows_have_no_k") and "wideband" in branches,
        "narrowband_rows_have_k": not c.bad.get("narrowband_rows_have_k") and "narrowband" in branches,
        "all_outputs_finite": True,  # non-finite IQ / defense output / logits abort the run (check_x, hailo_pred)
        "hailo_predictions_nonconstant": len(c.hailo_preds) >= MIN_DISTINCT_HAILO_CLASSES,
        "clean_cpu_hailo_agreement_plausible": agree >= CLEAN_AGREEMENT_PLAUSIBLE_MIN,
        "hailo_matches_formal_where_identical": not c.bad.get("hailo_matches_formal_where_identical") and len(hist) > 0,
        "formal_hailo_path_reused": (pre["gates"].get("hailo_adapter_source_audited", False)
                                     and pre["gates"].get("hef_sha256_matches_formal", False)
                                     and pre["gates"].get("hef_path_matches_formal", False)),
        "row_counts_ok": (len([r for r in clean_rows if r["in_dry_run_clean_subset"] == "True"]) == 22 * len(DEFENSES)
                          and len(att_rows) == 48 * len(DEFENSES)),
        "no_attack_generation": not any(m in sys.modules for m in ATTACK_MODULES),
    })
    info = {"n_clean_subset": len(clean_gis), "n_clean_support": len(need), "n_attacked": len(att_pairs),
            "clean_rows": len(clean_rows), "attacked_rows": len(att_rows),
            "clean_cpu_hailo_agreement": agree, "hailo_distinct_classes": sorted(c.hailo_preds),
            "adaptive_branches_covered": sorted(branches), "adaptive_instrumentation_checks": c.ak_checks,
            "historical_formal_hailo_comparisons": len(hist),
            "historical_formal_hailo_matches": sum(1 for r in hist if r["hailo_formal_match"] == "True"),
            "historical_identity_note": "exact identity = byte-identical stored input (clean: all; attacked: rows whose "
                                        "Phase B adversarial_sha256 == formal Hailo adversarial_sha256) for defenses "
                                        "none / topk*; adaptive_k_v2 never ran on Hailo -> no historical reference",
            "failures": {k: v[:20] for k, v in c.bad.items() if v}}
    return {"gates": g, "info": info}


# =========================================================================== evaluate (full; not run yet)
def c_paths(c: Ctx, name: str) -> dict:
    d = c.out / "eval_parts"
    return {"dir": d, "rows": d / f"{name}.csv", "marker": d / f"{name}.COMPLETE.json"}


def c_part_valid(ep: dict, n: int) -> bool:
    try:
        mk = json.loads(ep["marker"].read_text())
        return mk["rows"] == n and sha_file(ep["rows"]) == mk["rows_sha256"]
    except Exception:  # noqa: BLE001
        return False


def c_finish(ep: dict, n: int, extra: dict) -> None:
    atomic_write_text(ep["marker"], json.dumps({"rows": n, "rows_sha256": sha_file(ep["rows"]), **extra,
                                                "created_utc": now_utc()}))


def stage_evaluate(c: Ctx) -> dict:
    dv = c.out / "dry_run" / "validation.json"
    if not dv.exists() or json.loads(dv.read_text()).get("overall") != "PASS":
        p1a.abort(f"{dv} missing or not PASS; run --stage dry-run first")
    open_torch_side(c)
    hf = load_hailo_formal()
    clean, cmeta, cpu_clean, ak_clean = load_pb_clean(c)
    status = {"parts_reused": 0, "parts_evaluated": 0}
    open_hailo(c)
    try:
        ep = c_paths(c, "clean")
        ep["dir"].mkdir(parents=True, exist_ok=True)
        if c_part_valid(ep, N_CLEAN * len(DEFENSES)):
            status["parts_reused"] += 1
        else:
            rows = []
            for gi in range(N_CLEAN):
                rows += eval_clean_sample(c, gi, clean[gi:gi + 1], cmeta[gi], cpu_clean, ak_clean.get(gi), hf, False)
            n = write_csv(ep["rows"], CLEAN_FIELDS, rows)
            c_finish(ep, n, {"part": "clean", "failures": {k: len(v) for k, v in c.bad.items()}})
            status["parts_evaluated"] += 1
            log(f"clean: {n} rows")
        c.hclean = {}
        for r in read_csv(ep["rows"]):
            c.hclean.setdefault(int(r["global_index"]), {})[r["defense"]] = int(r["defended_pred_hailo"])
        for p in c.profiles:
            ep = c_paths(c, pb_part_name(p))
            if c_part_valid(ep, N_CLEAN * len(DEFENSES)):
                status["parts_reused"] += 1
                continue
            t0 = time.monotonic()
            before = {k: len(v) for k, v in c.bad.items()}
            arr, metas, cpu_rows, ak = load_pb_profile(c, p)
            rows = []
            for j, m in enumerate(metas):
                rows += eval_attacked_sample(c, p, arr[j:j + 1], m, cpu_rows, ak.get(int(m["global_index"])), hf)
            n = write_csv(ep["rows"], ATT_FIELDS, rows)
            c_finish(ep, n, {"part": p["profile"],
                             "failures": {k: len(v) - before.get(k, 0) for k, v in c.bad.items()}})
            status["parts_evaluated"] += 1
            log(f"{p['profile']}: {n} rows in {time.monotonic() - t0:.0f} s")
    finally:
        close_hailo(c)
    status["adaptive_instrumentation_checks"] = dict(c.ak_checks)
    status["failures"] = {k: v[:20] for k, v in c.bad.items() if v}
    return status


# =========================================================================== aggregation
def view(r: dict, b: str) -> dict:
    return {f: r[f"{f}_{b}"] for f in FLAGS}


def paired_metrics(grp: List[dict]) -> dict:
    mc = pb.defense_metrics([view(r, "cpu") for r in grp])
    mh = pb.defense_metrics([view(r, "hailo") for r in grp])
    out = {"n": len(grp)}
    out.update({f"cpu_{k}": v for k, v in mc.items() if k != "n"})
    out.update({f"hailo_{k}": v for k, v in mh.items() if k != "n"})
    for k in ("clean_acc_pct", "attacked_acc_pct", "cond_asr_pct", "recovery_pct", "defended_acc_pct",
              "retention_pct", "retention_gain_pp", "clean_defended_acc_pct", "clean_degradation_pct"):
        out[f"delta_hailo_minus_cpu_{k}"] = mh[k] - mc[k]
    n = len(grp)
    t = lambda r, f: r[f] == "True"  # noqa: E731
    out["agree_defended_pred_pct"] = pct(sum(t(r, "cpu_hailo_agree") for r in grp), n)
    out["agree_attacked_pred_pct"] = pct(sum(t(r, "cpu_hailo_agree_attacked") for r in grp), n)
    base_t = [r for r in grp if t(r, "attack_success_cpu") and t(r, "clean_correct_hailo")]
    out["n_attack_transfer_base"] = len(base_t)
    out["attack_transfer_pct"] = pct(sum(t(r, "attack_success_hailo") for r in base_t), len(base_t))
    base_r = [r for r in grp if t(r, "recovered_cpu") and t(r, "attack_success_hailo")]
    out["n_recovery_transfer_base"] = len(base_r)
    out["recovery_transfer_pct"] = pct(sum(t(r, "recovered_hailo") for r in base_r), len(base_r))
    return out


def summarize_rows(c: Ctx, clean_rows: List[dict], att_rows: List[dict], d: Path) -> dict:
    lab = pb.DEFENSE_LABEL
    cs = []
    for scope, keyf in (("overall", lambda r: "ALL"), ("modulation", lambda r: r["modulation"]),
                        ("snr_label_db", lambda r: str(r["snr_label_db"])),
                        ("adaptive_branch", lambda r: r["adaptive_branch"] or "n/a"),
                        ("adaptive_selected_k", lambda r: str(r["adaptive_selected_k"]) or "n/a")):
        for (g, dname), grp in pb.group_rows(clean_rows, lambda r: (keyf(r), r["defense"])).items():
            if scope.startswith("adaptive") and dname != "adaptive_k_v2":
                continue
            if scope == "adaptive_selected_k" and grp[0]["adaptive_branch"] != "narrowband":
                continue
            n = len(grp)
            row = {"scope": scope, "group": g, "defense": dname, "defense_label": lab[dname], "n": n}
            for b in ("cpu", "hailo"):
                cc = sum(1 for r in grp if r[f"clean_correct_{b}"] == "True")
                dc = sum(1 for r in grp if r[f"clean_defended_correct_{b}"] == "True")
                dg = sum(1 for r in grp if r[f"clean_degraded_{b}"] == "True")
                row.update({f"{b}_clean_acc_pct": pct(cc, n), f"{b}_clean_defended_acc_pct": pct(dc, n),
                            f"{b}_clean_degradation_pct": pct(dg, cc), f"{b}_n_clean_correct": cc})
            row["delta_hailo_minus_cpu_clean_acc_pct"] = row["hailo_clean_acc_pct"] - row["cpu_clean_acc_pct"]
            row["delta_hailo_minus_cpu_clean_defended_acc_pct"] = (row["hailo_clean_defended_acc_pct"]
                                                                   - row["cpu_clean_defended_acc_pct"])
            row["delta_hailo_minus_cpu_clean_degradation_pct"] = (row["hailo_clean_degradation_pct"]
                                                                  - row["cpu_clean_degradation_pct"])
            row["agree_defended_pred_pct"] = pct(sum(1 for r in grp if r["cpu_hailo_agree"] == "True"), n)
            row["agree_clean_pred_pct"] = pct(sum(1 for r in grp if int(r["clean_pred_cpu"])
                                                  == int(r["clean_pred_hailo"])), n)
            cs.append(row)
    if cs:
        write_csv(d / "clean_summary_paired.csv", list(cs[0].keys()), cs)
    ds = []
    scopes = [("overall", lambda r: "ALL"), ("profile", lambda r: r["profile"]),
              ("attack_family", lambda r: r["attack_family"]), ("modulation", lambda r: r["modulation"]),
              ("snr_label_db", lambda r: str(r["snr_label_db"]))]
    for scope, keyf in scopes:
        for (g, dname), grp in pb.group_rows(att_rows, lambda r: (keyf(r), r["defense"])).items():
            ds.append({"scope": scope, "group": g, "defense": dname, "defense_label": lab[dname],
                       **paired_metrics(grp)})
    ak = [r for r in att_rows if r["defense"] == "adaptive_k_v2"]
    for scope, rows_s, keyf in (
            ("adaptive_branch", ak, lambda r: r["adaptive_branch"]),
            ("adaptive_selected_k_narrowband", [r for r in ak if r["adaptive_branch"] == "narrowband"],
             lambda r: str(r["adaptive_selected_k"])),
            ("adaptive_branch_transition", ak, lambda r: r["branch_transition"]),
            ("adaptive_branch_x_attack_family", ak, lambda r: f"{r['adaptive_branch']}|{r['attack_family']}"),
            ("adaptive_branch_x_snr_label_db", ak, lambda r: f"{r['adaptive_branch']}|{r['snr_label_db']}")):
        for g, grp in pb.group_rows(rows_s, keyf).items():
            ds.append({"scope": scope, "group": g, "defense": "adaptive_k_v2", "defense_label": lab["adaptive_k_v2"],
                       **paired_metrics(grp)})
    if ds:
        write_csv(d / "defense_summary_paired.csv", list(ds[0].keys()), ds)
    hr = []
    for kind, rows in (("clean", clean_rows), ("attacked", att_rows)):
        for (prof, dname), grp in pb.group_rows(rows, lambda r: (r.get("profile", "CLEAN"), r["defense"])).items():
            ex = [r for r in grp if r["hailo_formal_match"] != ""]
            hr.append({"input_kind": kind, "profile": prof, "defense": dname, "n": len(grp),
                       "n_exact_input_with_formal_hailo": len(ex),
                       "n_hailo_matches_formal": sum(1 for r in ex if r["hailo_formal_match"] == "True"),
                       "n_cpu_recompute_checked": sum(1 for r in grp if r["cpu_recompute_match"] != ""),
                       "n_cpu_recompute_match": sum(1 for r in grp if r["cpu_recompute_match"] == "True"),
                       "n_adaptive_sha_checked": sum(1 for r in grp if r["adaptive_defended_sha_match"] != ""),
                       "n_adaptive_sha_match": sum(1 for r in grp if r["adaptive_defended_sha_match"] == "True")})
    if hr:
        write_csv(d / "reproduction_paired.csv", list(hr[0].keys()), hr)
    lat = []
    for kind, rows in (("clean", clean_rows), ("attacked", att_rows)):
        for dname in DEFENSES:
            v = [float(r["hailo_infer_ms"]) for r in rows if r["defense"] == dname and r["hailo_infer_ms"] != ""]
            lat.append({"input_kind": kind, "defense": dname, "component": "hailo_infer_ms", **pb.lat_stats(v)})
    write_csv(d / "hailo_latency_summary.csv", ["input_kind", "defense", "component", "n", "mean_ms", "median_ms",
                                                "p95_ms", "p99_ms"], lat)
    return {"clean_summary_rows": len(cs), "defense_summary_rows": len(ds)}


def stage_summarize(c: Ctx) -> dict:
    ep = c_paths(c, "clean")
    if not c_part_valid(ep, N_CLEAN * len(DEFENSES)):
        p1a.abort("Phase C clean part missing/invalid; run --stage evaluate")
    clean_rows = read_csv(ep["rows"])
    part_failures = {"clean": json.loads(ep["marker"].read_text()).get("failures", {})}
    keep = ["profile", "attack_family", "modulation", "snr_label_db", "defense", "adaptive_branch",
            "adaptive_selected_k", "branch_transition", "defended_pred_hailo", "cpu_hailo_agree",
            "cpu_hailo_agree_attacked", "hailo_formal_match", "cpu_recompute_match", "adaptive_defended_sha_match",
            "hailo_infer_ms"] + [f"{f}_{b}" for b in ("cpu", "hailo") for f in FLAGS]
    att_rows, part_files = [], []
    for p in c.profiles:
        ep = c_paths(c, pb_part_name(p))
        if not c_part_valid(ep, N_CLEAN * len(DEFENSES)):
            p1a.abort(f"Phase C part for {p['profile']} missing/invalid; run --stage evaluate")
        part_failures[p["profile"]] = json.loads(ep["marker"].read_text()).get("failures", {})
        part_files.append(ep["rows"])
        with ep["rows"].open(newline="") as f:  # reduced, string-interned rows (316,800 in total)
            for r in csv.DictReader(f):
                att_rows.append({k: sys.intern(r[k]) for k in keep})
    write_csv(c.out / "clean_paired_per_sample.csv", CLEAN_FIELDS, clean_rows)
    tmp = c.out / "attacked_paired_per_sample.csv.tmp"
    with tmp.open("w", newline="") as fo:
        for i, pf in enumerate(part_files):
            with pf.open(newline="") as fi:
                header = fi.readline()
                if i == 0:
                    fo.write(header)
                for line in fi:
                    fo.write(line)
    os.replace(tmp, c.out / "attacked_paired_per_sample.csv")
    summarize_rows(c, clean_rows, att_rows, c.out)
    t = lambda rows, f: [r for r in rows if r[f] != ""]  # noqa: E731
    allr = clean_rows + att_rows
    ak = [r for r in allr if r["defense"] == "adaptive_k_v2"]
    gates = {
        "row_counts": len(clean_rows) == N_CLEAN * len(DEFENSES) and len(att_rows) == N_CLEAN * N_PROFILES * len(DEFENSES),
        "all_profiles_present": {r["profile"] for r in att_rows} == {p["profile"] for p in c.profiles},
        "hailo_matches_formal_where_identical": all(r["hailo_formal_match"] == "True"
                                                    for r in t(allr, "hailo_formal_match")),
        "cpu_recompute_matches_phaseB": all(r["cpu_recompute_match"] == "True" for r in t(allr, "cpu_recompute_match")),
        "cpu_recompute_complete": len(t(allr, "cpu_recompute_match")) == len(allr),
        "adaptive_defended_sha_matches_phaseB": all(r["adaptive_defended_sha_match"] == "True" for r in ak),
        "wideband_rows_have_no_k": all(r["adaptive_selected_k"] == "" for r in ak if r["adaptive_branch"] == "wideband"),
        "narrowband_rows_have_k": all(r["adaptive_selected_k"].isdigit() for r in ak
                                      if r["adaptive_branch"] == "narrowband"),
        "hailo_predictions_nonconstant": len({r["defended_pred_hailo"] for r in allr}) >= MIN_DISTINCT_HAILO_CLASSES,
        "no_attack_generation": not any(m in sys.modules for m in ATTACK_MODULES),
        "evaluate_parts_recorded_no_failures": all(n == 0 for f in part_failures.values() for n in f.values()),
    }
    return {"gates": gates, "part_failures": part_failures, "counts": {"clean_rows": len(clean_rows), "attacked_rows": len(att_rows),
                                        "hailo_formal_comparisons": len(t(allr, "hailo_formal_match"))}}


# =========================================================================== manifest / main
def write_manifest(c: Ctx, args, stage_info: dict) -> None:
    mp = c.out / "manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else {"created_utc": now_utc()}
    man.update({
        "phase": "adaptive_k_phaseC_hailo_paired", "last_command": [sys.executable] + sys.argv, "updated_utc": now_utc(),
        "phase_b_dir": PHASE_B_DIR, "formal_hailo_dir": HAILO_FORMAL_DIR, "config": CONFIG_PATH,
        "hef_path": c.cfg["hef_path"], "expected_hef_sha256": EXPECTED_HEF_SHA256,
        "audited_source_sha256": AUDITED_SHA256, "expected_git_head": EXPECTED_GIT_HEAD,
        "defenses": {d: pb.DEFENSE_LABEL[d] for d in DEFENSES},
        "cpu_predictions": "reused from Phase B eval parts (== clean_per_sample.csv / attacked_per_sample.csv); "
                           "CPU re-inference is verification only" if c.cpu_verify else "reused from Phase B",
        "defended_iq": "reconstructed from stored Phase B IQ with the frozen Phase B apply_defense (pinned defense.py)",
        "hailo_path": "src/adapters/hailo_awn_adapter.py:HailoAWNAdapter.infer, B=1, FLOAT32 host vstreams",
        "note": pb.LINEAR_NOTE,
    })
    try:
        man["environment"] = p1a.environment_metadata()
    except Exception as exc:  # noqa: BLE001 - metadata only
        man["environment_error"] = repr(exc)
    man.setdefault("stages", []).append({"stage": args.stage, "utc": now_utc(), **stage_info})
    atomic_write_text(mp, json.dumps(man, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["preflight", "dry-run", "evaluate", "summarize"], required=True)
    ap.add_argument("--out-dir", default=None, help="Phase C result dir (reuse the same one for every stage)")
    ap.add_argument("--cpu-verify", choices=["all", "none"], default="all",
                    help="re-infer CPU AWN on every reconstructed input to prove it equals Phase B (default all)")
    args = ap.parse_args()
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (REPO_ROOT / (args.out_dir or f"results/adaptive_k_hailo_paired_{ts}")).resolve()
    for pdir in PROTECTED_DIRS + p1a.PROTECTED_RESULT_DIRS:
        prot = (REPO_ROOT / pdir).resolve()
        if out == prot or prot in out.parents or out in prot.parents:
            p1a.abort(f"refusing to write inside/over {pdir}")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["HAILORT_LOGGER_PATH"] = str(out)  # HailoRT logs go to the Phase C dir, not the repo root
    os.chdir(out)
    log(f"out_dir={out} stage={args.stage}")
    pre_guard = guard_snapshot(out)
    c = open_context(args, out)
    try:
        if args.stage == "preflight":
            res = stage_preflight(c)
        elif args.stage == "dry-run":
            pre = stage_preflight(c)
            res = stage_dry_run(c, pre)
            res["info"]["preflight_info"] = pre["info"]
        elif args.stage == "evaluate":
            st = stage_evaluate(c)
            res = {"gates": {k: not v for k, v in c.bad.items()}, "info": st}
        else:
            res = stage_summarize(c)
        res["gates"]["completed_without_abort"] = True
    except (SystemExit, Exception) as exc:  # noqa: BLE001 -- fail closed, but still write the validation file
        close_hailo(c)
        res = {"gates": {"completed_without_abort": False},
               "info": {"abort": repr(exc), "failures": {k: v[:20] for k, v in c.bad.items() if v}}}
    guard = guard_compare(pre_guard, guard_snapshot(out))
    res.setdefault("gates", {})
    res["gates"]["no_phaseB_modification"] = guard["protected_dirs_unchanged"]
    res["gates"]["no_writes_outside_phaseC_dir"] = all(guard[k] for k in ("protected_dirs_unchanged",
                                                                          "watched_files_unchanged",
                                                                          "git_status_unchanged",
                                                                          "git_diff_unchanged"))
    res["write_guard"] = guard
    res["write_guard_scope"] = ("stat of every file in " + ", ".join(PROTECTED_DIRS) + "; sha256 of watched sources "
                                "and repo-root HailoRT logs; git status --porcelain and git diff (results/ is "
                                "git-ignored, so writes elsewhere under results/ are not covered)")
    res["overall"] = "PASS" if res["gates"] and all(res["gates"].values()) else "FAIL"
    res["stage"], res["utc"], res["note"] = args.stage, now_utc(), pb.LINEAR_NOTE
    target = {"preflight": out / "preflight.json", "dry-run": out / "dry_run" / "validation.json",
              "evaluate": out / "evaluate_status.json", "summarize": out / "validation.json"}[args.stage]
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, json.dumps(res, indent=2, default=str))
    write_manifest(c, args, {"overall": res["overall"], "gates": res["gates"]})
    log(f"{args.stage}: overall={res['overall']} -> {target}")
    for k, v in res["gates"].items():
        log(f"  {'PASS' if v else 'FAIL'}  {k}")
    if res["overall"] != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
