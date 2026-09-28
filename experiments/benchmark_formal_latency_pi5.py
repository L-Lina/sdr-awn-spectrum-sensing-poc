"""
FORMAL LATENCY BENCHMARK (Raspberry Pi 5 + Hailo-8): thermally controlled, paired, one thread configuration per
process. Standalone; no production code, config, external/ or Phase A/B/C result is modified; no attack is generated.

PRIMARY configuration (the formal result): the formal Pi runtime -- torch thread settings are NOT touched, the process
defaults are used and must be intra-op = 4 / inter-op = 4 (as in every formal CPU / Hailo / Phase B / Phase C run),
batch size 1. SECONDARY thread sensitivity (optional, never part of the formal result): --threads-mode explicit
--intra-threads N --interop-threads M in a SEPARATE process and a SEPARATE --out-dir; every output is labelled
configuration_role = SECONDARY_THREAD_SENSITIVITY.

Primary metrics (see METRICS for class and exact definition):
  A sensing: A_sensing_evaluation_path_wall_ms   measured    wall of the UNMODIFIED formal evaluation function, which
                                                              includes evaluation-only ground-truth metrics + sha256
                                                              -> evaluation-path timing, NOT deployment sensing latency
             A_diag_sensing_<stage> (x5)         diagnostic  the function's own internal stage timers
             A_sensing_core_derived_ms           derived     sum of the 5 stage timers (deployment-relevant estimate)
  B host fixed-K defense latency                 measured
  C host Adaptive-K production latency           measured    (branch-based adaptive defense: wideband 32-level
                                                              quantization / narrowband adaptive Top-K)
  D Adaptive-K instrumented route/select/transform  diagnostic (Phase B replica, bit-identity checked)
  E CPU AWN inference latency (formal 4/4)       measured    (+ bare forward: diagnostic)
  F Hailo adapter round-trip latency             measured    (+ bare HailoRT InferVStreams call: diagnostic)
  G clean pipeline:
      G_clean_pipeline_evaluation_path_measured_ms   measured    one continuous timer (evaluation-path sensing ->
                                                                  defense -> classifier); contains evaluation bookkeeping
      G_clean_pipeline_deployment_estimate_ms        derived     sensing core stage sum + measured defense + measured
                                                                  classifier, same input and pass; NOT measured
  H attack-generation-excluded attacked pipeline (sensing + post-sensing defense/classifier composite; the source
    capture is sensed, then the STORED Phase B adversarial IQ is substituted; attack generation is excluded; NOT an
    attacked end-to-end latency):
      H_attacked_pipeline_attackgen_excluded_evaluation_path_measured_ms   measured_composite
      H_attacked_pipeline_attackgen_excluded_deployment_estimate_ms        derived
  Evaluation overhead (derived, paired): evaluation-path wall - core/deployment estimate, reported separately.
  No evaluation-path timing is deployment latency; no derived sum is a measured end-to-end latency.
  auxiliary: NPU hardware latency from `hailortcli run --measure-latency` (separate process, synthetic input, unpaired)
  derived:   paired differences, incl. "adapter/wrapper overhead estimate" (adapter wall - bare call wall). The bare
             HailoRT call still contains host quantization, transfer, NPU execution and dequantization; it is NOT
             decomposed and the estimate is NOT a PCIe / host-to-Hailo transfer measurement.
Host defense and routing time is host CPU work and is never attributed to the Hailo NPU.

Protocol: cells = input blocks (clean + --profiles), 88 inputs each (11 modulations x SNR {-20,-6,6,18} x idx {0,1});
per cell an untimed warm-up of --warmup-inputs inputs, then --passes timed passes; CPU/Hailo order alternates per input,
defense order rotates per pass; blocks run forward / reverse per replication (even --replications). Thermal method of
Phase 1D-0 (helpers imported unchanged) with governor phases, per attempt (see acquire_thermal_start):
  ondemand -> cool to <= --max-start-temp-c with throttle bits 0-3 clear -> switch all cores to performance -> verify
  (all governors performance, throttle bits clear, temperature start gate, ARM clock at performance state) -> untimed
  warm-up -> fresh start gate -> TIMED REGION (1 s temperature / 10 s throttle trace) -> post snapshot -> ondemand.
  Any failed gate -> ondemand, cool again, retry; THERMAL_REVIEW cells retried. Governor switching is never inside a
  timed metric; every transition is logged (governor_transitions.csv). The CPU governor is the ONLY system setting the
  script changes (original governors restored at exit); frequency limits, clocks and fan are never changed.
GC disabled during timed cells. Correctness on every call (frozen inputs, defended outputs, predictions vs Phase C).
Zero accepted cells -> overall FAIL with reason NO_ACCEPTED_CELLS (no summary on nonexistent results).

Run on the Pi (from the repo root):
  .venv/bin/python experiments/benchmark_formal_latency_pi5.py --stage preflight --out-dir <dir>
  .venv/bin/python experiments/benchmark_formal_latency_pi5.py --stage dry-run   --out-dir <dir>
  .venv/bin/python experiments/benchmark_formal_latency_pi5.py --stage run       --out-dir <dir>
  .venv/bin/python experiments/benchmark_formal_latency_pi5.py --stage summarize --out-dir <dir>
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import datetime as _dt  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import benchmark_adaptive_k_effectiveness_pi5 as pb  # noqa: E402 -- frozen Phase B (defenses, stores)
import bench_attack_thermal_control_pi5 as p1d0  # noqa: E402 -- Phase 1D-0 thermal helpers (no torch at import)

p1a = pb.p1a
REPO_ROOT = pb.REPO_ROOT
PHASE_B_DIR = "results/adaptive_k_effectiveness_20260927_164906"
PHASE_C_DIR = "results/adaptive_k_hailo_paired_20260928"
CONFIG_PATH = "configs/hailo_full_matrix.json"
PROTECTED_DIRS = (PHASE_B_DIR, PHASE_C_DIR, "results/hailo_full_matrix_20260920_seedaligned",
                  "results/cpu_full_matrix_20260921_seedaligned", pb.PHASE_A_DIR)
WATCHED_FILES = (CONFIG_PATH, "src/adapters/hailo_awn_adapter.py", "src/adapters/awn_adapter.py",
                 "src/adapters/topk_adapter.py", "src/adapters/attack_adapter.py", "src/utils/config.py",
                 "src/utils/pipeline.py", "external/adversarial-rf/util/defense.py",
                 "experiments/run_hailo_full_matrix.py", "experiments/benchmark_adaptive_k_effectiveness_pi5.py",
                 "experiments/benchmark_adaptive_k_hailo_paired_pi5.py", "experiments/bench_attack_thermal_control_pi5.py",
                 "experiments/benchmark_formal_latency_pi5.py", "hailort.log", "pyhailort.log")
EXPECTED_HEF_SHA256 = "6e54266f7e5931e432d4e5e67cd8e4a6e14863c16a94ff02a1946d21abc73dbb"
EXPECTED_HAILO_BACKEND = "Hailo-8:awn_2016_10a_c5.hef"
EXPECTED_HAILO_DEVICE = "0001:01:00.0"
FORMAL_THREADS = (4, 4)  # (intra-op, inter-op) = Pi process defaults used by every formal run
DEFENSES = pb.DEFENSES  # none, topk10..40, adaptive_k_v2
FIXED_K = [d for d in DEFENSES if d.startswith("topk")]
SUBSET_SNRS, SUBSET_IDX = (-20, -6, 6, 18), (0, 1)
DRY_SNRS, DRY_IDX = (-20, 18), (0,)
DEFAULT_PROFILES = "fgsm/eps0.03,pgd/eps0.03,cw/c1_steps20_lr0.01,deepfool/package_default"
SENSING_STAGES = ["embedding_ms", "energy_detection_ms", "region_postprocess_ms", "segmentation_alignment_ms",
                  "awn_preprocess_ms"]
ROLE_PRIMARY, ROLE_SECONDARY = "PRIMARY_FORMAL", "SECONDARY_THREAD_SENSITIVITY"

# metric id -> (letter, class, definition). Classes: measured | measured_composite | diagnostic | derived | auxiliary
METRICS: Dict[str, Tuple[str, str, str]] = {
    "A_sensing_evaluation_path_wall_ms": ("A", "measured", "measured wall time of the unmodified formal evaluation "
                                          "function run_hailo_full_matrix.build_sensing_input, INCLUDING evaluation-only "
                                          "ground-truth metrics and sha256 of x_clean; evaluation-path timing, not "
                                          "deployment sensing latency"),
    **{f"A_diag_sensing_{s}": ("A", "diagnostic", f"internal stage timer '{s}' reported by the formal function")
       for s in SENSING_STAGES},
    "A_sensing_core_derived_ms": ("A", "derived", "derived sum of the 5 internal sensing/preprocessing stage timers "
                                  "(embedding, energy detection, region post-process, segment alignment, AWN "
                                  "preprocess); excludes the evaluation-only ground-truth metrics and sha256, which the "
                                  "internal timers do not cover; NOT a directly measured wall time"),
    "B_host_fixedk_defense_ms": ("B", "measured", "host CPU: numpy->torch->fft_topk_denoise(K)->numpy"),
    "C_host_adaptive_production_ms": ("C", "measured", "host CPU: numpy->torch->adaptive_k_v2_snr_defense->numpy "
                                      "(production public function)"),
    "D_diag_ak_route_ms": ("D", "diagnostic", "instrumented replica: FFT + flatness routing + wideband quantization"),
    "D_diag_ak_select_ms": ("D", "diagnostic", "instrumented replica: spectral_ratio (linear), K cap, knee (narrowband)"),
    "D_diag_ak_topk_transform_ms": ("D", "diagnostic", "instrumented replica: narrowband per-sample Top-K + IFFT"),
    "E_cpu_awn_adapter_ms": ("E", "measured", "AWNModelAdapter.infer (formal call path), formal 4/4 threads"),
    "E_diag_cpu_forward_bare_ms": ("E", "diagnostic", "bare torch no_grad forward of the same model"),
    "F_hailo_adapter_roundtrip_ms": ("F", "measured", "HailoAWNAdapter.infer (formal call path) round trip"),
    "F_diag_hailort_infervstreams_ms": ("F", "diagnostic", "bare HailoRT InferVStreams.infer call (still contains host "
                                        "quantization, transfer, NPU execution, dequantization; not decomposed)"),
    "G_clean_pipeline_evaluation_path_measured_ms": (
        "G", "measured", "clean inputs, one continuous measured timer: evaluation-path sensing of the capture (includes "
        "evaluation-only ground-truth metrics + sha256) -> defense on the sensed x -> classifier -> argmax; "
        "evaluation-path timing, not deployment latency"),
    "G_clean_pipeline_deployment_estimate_ms": (
        "G", "derived", "DERIVED, not measured: A_sensing_core_derived_ms + measured host defense (B/C; 0 for none) + "
        "measured classifier adapter call (E or F), all from the same input and pass"),
    "H_attacked_pipeline_attackgen_excluded_evaluation_path_measured_ms": (
        "H", "measured_composite", "attack-generation-excluded attacked pipeline, one continuous measured timer: "
        "evaluation-path sensing of the source capture, then the STORED Phase B adversarial IQ is substituted, defense, "
        "classifier, argmax; attack generation excluded; not an attacked end-to-end latency and not deployment latency"),
    "H_attacked_pipeline_attackgen_excluded_deployment_estimate_ms": (
        "H", "derived", "DERIVED, not measured: A_sensing_core_derived_ms of the source capture + measured host defense "
        "on the stored adversarial IQ (B/C; 0 for none) + measured classifier adapter call (E or F), same input and "
        "pass; attack generation excluded"),
}
DEPLOYMENT_ESTIMATE = {True: "G_clean_pipeline_deployment_estimate_ms",
                       False: "H_attacked_pipeline_attackgen_excluded_deployment_estimate_ms"}
EVAL_PATH_MEASURED = {True: "G_clean_pipeline_evaluation_path_measured_ms",
                      False: "H_attacked_pipeline_attackgen_excluded_evaluation_path_measured_ms"}
AUX_HW = ("auxiliary", "HailoRT-reported NPU hardware latency, separate process, synthetic input, batch 1, unpaired")

RAW_FIELDS = ["attempt_id", "planned_cell_id", "block", "pass", "input_pos", "global_index", "component", "backend",
              "defense", "branch", "selected_k", "ms"]
CELL_FIELDS = ["attempt_id", "planned_cell_id", "order_index", "replication", "block", "attempt", "accepted",
               "correctness", "correctness_detail", "thermal_status", "thermal_reasons", "start_rejections",
               "cooldown_waited_s", "temp_before_c", "temp_after_c", "temp_max_in_cell_c", "throttled_before",
               "throttled_after", "arm_clock_before_hz", "arm_clock_after_hz", "governor_before", "governor_after",
               "hailo_chip_temp_before", "hailo_chip_temp_after", "torch_threads_before", "torch_threads_after",
               "cell_seconds", "n_timed_rows", "configuration_role", "perf_clock_check", "start_gate_history"]
TRACE_FIELDS = ["attempt_id"] + [f for f in p1d0.TRACE_FIELDS if f != "cell_id"]
BAD_KEYS = ("clean_sensing_identity", "source_capture_sensing_identity", "stored_adversarial_identity",
            "defended_sha", "adaptive_branch_k", "cpu_pred", "hailo_pred", "backend_status")


def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def log(msg: str) -> None:
    p1a.log(f"[latency] {msg}")


def ns() -> int:
    return time.perf_counter_ns()


def append_csv(path: Path, fields: List[str], rows: List[dict]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


# =========================================================================== write guard
def guard_snapshot() -> dict:
    snap = {"protected": {}, "watched": {}}
    for d in PROTECTED_DIRS:
        root = REPO_ROOT / d
        snap["protected"][d] = ({str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns]
                                 for p in sorted(root.rglob("*")) if p.is_file()} if root.exists() else None)
    for f in WATCHED_FILES:
        p = REPO_ROOT / f
        snap["watched"][f] = pb.sha_file(p) if p.is_file() else None
    snap["git_status"] = p1a.run_cmd(["git", "status", "--porcelain"])
    diff = p1a.run_cmd(["git", "diff"])
    snap["git_diff_sha256"] = None if diff is None else hashlib.sha256(diff.encode()).hexdigest()
    return snap


def guard_compare(pre: dict, post: dict) -> dict:
    changed = [d for d in PROTECTED_DIRS if pre["protected"][d] != post["protected"][d]]
    watched = [f for f in WATCHED_FILES if pre["watched"][f] != post["watched"][f]]
    return {"protected_dirs_unchanged": not changed, "protected_changed": changed,
            "watched_files_unchanged": not watched, "watched_changed": watched,
            "git_status_unchanged": pre["git_status"] == post["git_status"],
            "git_diff_unchanged": pre["git_diff_sha256"] == post["git_diff_sha256"]}


# =========================================================================== context
class Ctx:
    pass


def select_inputs(cfg: dict, dry: bool) -> List[int]:
    keys = [(m, int(s), int(i)) for m in cfg["modulations"] for s in cfg["snrs"] for i in cfg["sample_indices"]]
    gi_of = {k: j for j, k in enumerate(keys)}
    snrs, idxs = (DRY_SNRS, DRY_IDX) if dry else (SUBSET_SNRS, SUBSET_IDX)
    return [gi_of[(m, s, i)] for m in cfg["modulations"] for s in snrs for i in idxs]


def open_context(args, dry: bool) -> Ctx:
    c = Ctx()
    c.args, c.dry = args, dry
    c.role = ROLE_PRIMARY if args.threads_mode == "formal-default" else ROLE_SECONDARY
    c.cfg = json.loads((REPO_ROOT / CONFIG_PATH).read_text())
    c.keys = [(m, int(s), int(i)) for m in c.cfg["modulations"] for s in c.cfg["snrs"] for i in c.cfg["sample_indices"]]
    c.gis = select_inputs(c.cfg, dry)
    profiles_all = [dict(r, profile_index=int(r["profile_index"]))
                    for r in pb.read_csv(REPO_ROOT / PHASE_B_DIR / "attack_profiles.csv")]
    names = [p.strip() for p in args.profiles.split(",") if p.strip()]
    if dry:
        names = names[:1]
    byname = {p["profile"]: p for p in profiles_all}
    missing = [n for n in names if n not in byname]
    if missing:
        p1a.abort(f"unknown profiles {missing}")
    c.profiles = [byname[n] for n in names]
    c.blocks = ["clean"] + names
    c.pipeline_defenses = [d.strip() for d in args.pipeline_defenses.split(",") if d.strip()]
    if any(d not in DEFENSES for d in c.pipeline_defenses):
        p1a.abort(f"--pipeline-defenses must be a subset of {DEFENSES}")
    return c


def configure_threads(c: Ctx) -> None:
    """Primary: touch nothing, verify the process defaults are the formal 4/4. Secondary: set once, before any
    parallel work (this process then runs only that configuration)."""
    import torch
    c.torch = torch
    if c.role == ROLE_SECONDARY:
        torch.set_num_interop_threads(int(c.args.interop_threads))
        torch.set_num_threads(int(c.args.intra_threads))
    c.threads = (torch.get_num_threads(), torch.get_num_interop_threads())
    c.expected_threads = FORMAL_THREADS if c.role == ROLE_PRIMARY else (int(c.args.intra_threads),
                                                                        int(c.args.interop_threads))
    if c.threads != c.expected_threads:
        p1a.abort(f"torch threads {c.threads} != expected {c.expected_threads} for {c.role}")


def open_backends(c: Ctx) -> None:
    configure_threads(c)
    src = pb.check_sources()
    for k in ("reference_ok", "defense_py_ok", "adversarial_rf_commit_ok", "phase_a_ok"):
        if not src[k]:
            p1a.abort(f"source check failed: {k}")
    c.src = src
    advrf = str(REPO_ROOT / "external" / "adversarial-rf")
    if advrf not in sys.path:
        sys.path.insert(0, advrf)
    import util.defense as D
    c.D = D
    c.ak_checks = {"calls": 0, "bit_identical": 0}
    import run_hailo_full_matrix as rhm  # formal sensing chain (build_sensing_input); nothing in it is modified
    c.rhm = rhm
    from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE
    c.awn = AWNModelAdapter(checkpoint_path=str(REPO_ROOT / c.cfg["checkpoint_path"]), device="cpu")
    if c.awn.model is None or c.awn.backend_name != _REAL_MODEL_SOURCE or c.awn.status != "ok":
        p1a.abort("CPU AWN backend is not real")
    c.REAL_MODEL = _REAL_MODEL_SOURCE
    from src.adapters.hailo_awn_adapter import HailoAWNAdapter
    c.HailoAWNAdapter = HailoAWNAdapter
    c.hef = str(Path(c.cfg["hef_path"]).resolve())
    open_hailo(c)
    log("loading RadioML dataset and Phase C expectations (untimed)")
    c.dataset = rhm.load_radioml_dict(str(REPO_ROOT / c.cfg["dataset_path"]))
    load_inputs(c)


def open_hailo(c: Ctx) -> None:
    c.hailo = c.HailoAWNAdapter(c.hef)
    if c.hailo.status != "ok" or c.hailo.backend_name != EXPECTED_HAILO_BACKEND:
        p1a.abort(f"Hailo adapter not ok: {c.hailo.status}")


def close_hailo(c: Ctx) -> None:
    if getattr(c, "hailo", None) is not None:
        c.hailo.close()
        c.hailo = None


def hailo_chip_temp(c: Ctx) -> str:
    try:
        t = c.hailo._vdevice.get_physical_devices()[0].control.get_chip_temperature()
        return json.dumps({"ts0_c": float(t.ts0_temperature), "ts1_c": float(t.ts1_temperature)})
    except Exception as exc:  # noqa: BLE001 - metadata only
        return json.dumps({"unavailable": repr(exc)[:120]})


def load_inputs(c: Ctx) -> None:
    """Stored Phase B IQ + Phase C expectations. Clean: Phase B clean store. Attacked: Phase B adversarial chunk,
    sha256 == chunk metadata adversarial_sha256 == Phase C input_sha256 (the stored adversarial IQ itself)."""
    gis = set(c.gis)
    c.exp, c.inp_sha = {}, {}
    with (REPO_ROOT / PHASE_C_DIR / "clean_paired_per_sample.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            gi = int(r["global_index"])
            if gi in gis:
                c.exp[("clean", gi, r["defense"])] = {"defended_sha": r["defended_sha256"],
                                                      "cpu": int(r["defended_pred_cpu"]),
                                                      "hailo": int(r["defended_pred_hailo"]),
                                                      "branch": r["adaptive_branch"], "k": r["adaptive_selected_k"]}
                c.inp_sha[("clean", gi)] = r["input_sha256"]
    want = {p["profile"] for p in c.profiles}
    with (REPO_ROOT / PHASE_C_DIR / "attacked_paired_per_sample.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            gi = int(r["global_index"])
            if gi in gis and r["profile"] in want:
                c.exp[(r["profile"], gi, r["defense"])] = {"defended_sha": r["defended_sha256"],
                                                           "cpu": int(r["defended_pred_cpu"]),
                                                           "hailo": int(r["defended_pred_hailo"]),
                                                           "branch": r["adaptive_branch"],
                                                           "k": r["adaptive_selected_k"]}
                c.inp_sha[(r["profile"], gi)] = r["input_sha256"]
    clean = np.load(REPO_ROOT / PHASE_B_DIR / "clean_store" / "clean_store.npy", allow_pickle=False)
    cmeta = pb.read_csv(REPO_ROOT / PHASE_B_DIR / "clean_store" / "clean_metadata.csv")
    c.x = {}
    bad = []
    for gi in c.gis:
        x = np.ascontiguousarray(clean[gi:gi + 1])
        if pb.sha_arr(x) != cmeta[gi]["clean_input_sha256"] or pb.sha_arr(x) != c.inp_sha.get(("clean", gi)):
            bad.append(("clean", gi))
        c.x[("clean", gi)] = x
    for p in c.profiles:
        for mi, mod in enumerate(c.cfg["modulations"]):
            cp = pb.chunk_paths(REPO_ROOT / PHASE_B_DIR, p, mi, mod)
            arr = None
            for j, m in enumerate(pb.read_csv(cp["meta"])):
                gi = int(m["global_index"])
                if gi not in gis:
                    continue
                arr = np.load(cp["npy"], allow_pickle=False) if arr is None else arr
                x = np.ascontiguousarray(arr[j:j + 1])
                if pb.sha_arr(x) != m["adversarial_sha256"] or pb.sha_arr(x) != c.inp_sha.get((p["profile"], gi)):
                    bad.append((p["profile"], gi))
                c.x[(p["profile"], gi)] = x
    if bad or len(c.x) != len(c.blocks) * len(c.gis):
        p1a.abort(f"stored input identity mismatch vs Phase B / Phase C: {bad[:5]} n={len(c.x)}")
    missing = [(b, gi, d) for b in c.blocks for gi in c.gis for d in DEFENSES if (b, gi, d) not in c.exp]
    if missing:
        p1a.abort(f"Phase C expectations missing: {missing[:5]}")


# =========================================================================== measured operations
def run_defense(c: Ctx, d: str, x: np.ndarray) -> np.ndarray:
    if d == "none":
        return x
    xt = c.torch.from_numpy(x)
    if d.startswith("topk"):
        return c.D.fft_topk_denoise(xt, topk=int(d[4:])).detach().cpu().numpy().astype(np.float32)
    return c.D.adaptive_k_v2_snr_defense(xt).detach().cpu().contiguous().numpy().astype(np.float32, copy=False)


def sense(c: Ctx, gi: int):
    mod, snr, idx = c.keys[gi]
    sample = c.dataset[(mod, snr)][idx].astype(np.float32)
    return c.rhm.build_sensing_input(sample, c.cfg, mod, snr, idx)


def classify(c: Ctx, backend: str, y: np.ndarray) -> Tuple[int, str]:
    logits, meta = c.awn.infer(y) if backend == "cpu" else c.hailo.infer(y)
    return int(np.argmax(logits[0])), meta.get("awn_status", "")


# =========================================================================== one cell body
def cell_body(c: Ctx, block: str, attempt_id: str, planned: str, gis: List[int], passes: int, record: bool,
              bad: Dict[str, int]) -> List[dict]:
    rows: List[dict] = []
    torch = c.torch

    def rec(p, pos, gi, comp, backend, d, ms, branch="", k=""):
        if record:
            rows.append({"attempt_id": attempt_id, "planned_cell_id": planned, "block": block, "pass": p,
                         "input_pos": pos, "global_index": gi, "component": comp, "backend": backend, "defense": d,
                         "branch": branch, "selected_k": k, "ms": ms})

    is_clean = block == "clean"
    for p in range(passes):
        rot = p % len(DEFENSES)
        dorder = DEFENSES[rot:] + DEFENSES[:rot]
        for pos, gi in enumerate(gis):
            backends = ("cpu", "hailo") if (pos + p) % 2 == 0 else ("hailo", "cpu")
            # ---- A: formal sensing
            t0 = ns()
            x_s, _meta, timing = sense(c, gi)
            t1 = ns()
            same = pb.sha_arr(x_s) == c.inp_sha[("clean", gi)]
            bad["clean_sensing_identity" if is_clean else "source_capture_sensing_identity"] += not same
            rec(p, pos, gi, "A_sensing_evaluation_path_wall_ms", "host", "", (t1 - t0) / 1e6)
            core_ms = sum(float(timing[s]) for s in SENSING_STAGES)  # derived: internal stage timers only
            rec(p, pos, gi, "A_sensing_core_derived_ms", "host", "", core_ms)
            def_ms: Dict[str, float] = {"none": 0.0}  # measured host defense per defense (none: no call)
            cls_ms: Dict[tuple, float] = {}  # measured classifier adapter call per (backend, defense)
            for s in SENSING_STAGES:
                rec(p, pos, gi, f"A_diag_sensing_{s}", "host", "", float(timing[s]))
            # classifier input: clean -> the freshly sensed x; attacked -> the stored adversarial IQ (identity re-checked)
            if is_clean:
                x = x_s
            else:
                x = c.x[(block, gi)]
                bad["stored_adversarial_identity"] += pb.sha_arr(x) != c.inp_sha[(block, gi)]
            for d in dorder:
                e = c.exp[(block, gi, d)]
                t0 = ns()
                y = run_defense(c, d, x)
                t1 = ns()
                bad["defended_sha"] += pb.sha_arr(y) != e["defended_sha"]
                if d != "none":
                    def_ms[d] = (t1 - t0) / 1e6
                if d.startswith("topk"):
                    rec(p, pos, gi, "B_host_fixedk_defense_ms", "host", d, (t1 - t0) / 1e6)
                elif d == "adaptive_k_v2":
                    rec(p, pos, gi, "C_host_adaptive_production_ms", "host", d, (t1 - t0) / 1e6, e["branch"], e["k"])
                    _y2, info = pb.adaptive_k(c, x)  # diagnostic replica; bit-identity checked inside (aborts on diff)
                    br = "wideband" if info["wideband"] else "narrowband"
                    bad["adaptive_branch_k"] += (br != e["branch"]) or (str(info["selected_k"]) != e["k"])
                    rec(p, pos, gi, "D_diag_ak_route_ms", "host", d, info["route_ms"], br, e["k"])
                    if info["select_ms"] != "":
                        rec(p, pos, gi, "D_diag_ak_select_ms", "host", d, info["select_ms"], br, e["k"])
                        rec(p, pos, gi, "D_diag_ak_topk_transform_ms", "host", d, info["topk_transform_ms"], br, e["k"])
                for b in backends:
                    if b == "cpu":
                        t0 = ns()
                        pred, st = classify(c, "cpu", y)
                        t1 = ns()
                        cls_ms[("cpu", d)] = (t1 - t0) / 1e6
                        rec(p, pos, gi, "E_cpu_awn_adapter_ms", "cpu", d, (t1 - t0) / 1e6, e["branch"], e["k"])
                        t0 = ns()
                        with torch.no_grad():
                            c.awn.model(torch.from_numpy(y))
                        t1 = ns()
                        rec(p, pos, gi, "E_diag_cpu_forward_bare_ms", "cpu", d, (t1 - t0) / 1e6, e["branch"], e["k"])
                    else:
                        t0 = ns()
                        pred, st = classify(c, "hailo", y)
                        t1 = ns()
                        cls_ms[("hailo", d)] = (t1 - t0) / 1e6
                        rec(p, pos, gi, "F_hailo_adapter_roundtrip_ms", "hailo", d, (t1 - t0) / 1e6, e["branch"], e["k"])
                        t0 = ns()
                        c.hailo._pipeline.infer({c.hailo.input_info.name: y[..., np.newaxis]})
                        t1 = ns()
                        rec(p, pos, gi, "F_diag_hailort_infervstreams_ms", "hailo", d, (t1 - t0) / 1e6, e["branch"],
                            e["k"])
                    bad[f"{b}_pred"] += pred != e[b]
                    bad["backend_status"] += st != "ok"
            # ---- G / H deployment estimates: DERIVED (core stage sum + measured defense + measured classifier, same pass)
            for d in c.pipeline_defenses:
                e = c.exp[(block, gi, d)]
                for b in ("cpu", "hailo"):
                    rec(p, pos, gi, DEPLOYMENT_ESTIMATE[is_clean], b, d, core_ms + def_ms[d] + cls_ms[(b, d)],
                        e["branch"], e["k"])
            # ---- G / H evaluation-path measurements: one continuous timer per (defense, backend)
            comp = EVAL_PATH_MEASURED[is_clean]
            for d in c.pipeline_defenses:
                e = c.exp[(block, gi, d)]
                for b in backends:
                    t0 = ns()
                    x_s2, _m, _t = sense(c, gi)
                    xin = x_s2 if is_clean else c.x[(block, gi)]
                    pred, st = classify(c, b, run_defense(c, d, xin))
                    t1 = ns()
                    bad[f"{b}_pred"] += pred != e[b]
                    bad["backend_status"] += st != "ok"
                    rec(p, pos, gi, comp, b, d, (t1 - t0) / 1e6, e["branch"], e["k"])
    return rows


# =========================================================================== plan / thermal-controlled execution
def build_plan(c: Ctx, replications: int) -> List[dict]:
    plan = []
    for r in range(replications):
        order = list(range(len(c.blocks))) if r % 2 == 0 else list(reversed(range(len(c.blocks))))
        for bi in order:
            plan.append({"planned_cell_id": f"{len(plan):02d}_r{r}_{c.blocks[bi].replace('/', '_')}",
                         "replication": r, "block": c.blocks[bi], "order_index": len(plan)})
    return plan



# =========================================================================== CPU governor phases (the ONLY system setting
# this script changes: ondemand for untimed cooldown, performance for the timed region; never frequency limits, clocks,
# fan or anything else)
GOV_GLOB = "/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor"
GOV_FIELDS = ["utc", "monotonic_s", "attempt_id", "reason", "target", "before", "after", "ok", "error", "switch_s"]


def governor_paths() -> List[str]:
    import glob
    return sorted(glob.glob(GOV_GLOB), key=lambda p: int(re.search(r"cpu(\d+)/", p).group(1)))


def read_governors() -> Dict[str, Optional[str]]:
    return {p: p1a.read_text(p) for p in governor_paths()}


def _write_governor(path: str, gov: str, method: str) -> None:
    if method == "direct":
        with open(path, "w") as f:
            f.write(gov)
    elif method == "sudo-n-tee":
        subprocess.run(["sudo", "-n", "/usr/bin/tee", path], input=gov + "\n", text=True, capture_output=True,
                       timeout=15, check=True)
    else:
        raise RuntimeError(f"no governor write method ({method})")


def detect_governor_write_method() -> Tuple[Optional[str], str]:
    """No-op probe: rewrites each core's CURRENT governor value (no setting change) to test writability."""
    cur = read_governors()
    if not cur or any(v is None for v in cur.values()):
        return None, "governors unreadable"
    last = "no method tried"
    for method in ("direct", "sudo-n-tee"):
        try:
            for path, v in cur.items():
                _write_governor(path, v, method)
            if read_governors() == cur:
                return method, "ok (no-op rewrite of the current value)"
            last = f"{method}: read-back differs"
        except Exception as exc:  # noqa: BLE001
            last = f"{method}: {exc!r}"[:300]
    return None, last


class GovernorController:
    """Switches ALL cores between ondemand and performance; every transition is recorded with timestamps."""

    def __init__(self, method: str):
        self.method = method
        self.original = read_governors()
        self.transitions: List[dict] = []

    def set(self, gov: str, reason: str, attempt_id: str = "") -> bool:
        before = read_governors()
        utc, m0, err = now_utc(), time.monotonic(), ""
        try:
            for path in governor_paths():
                if before.get(path) != gov:
                    _write_governor(path, gov, self.method)
        except Exception as exc:  # noqa: BLE001
            err = repr(exc)[:300]
        after = read_governors()
        ok = not err and bool(after) and all(v == gov for v in after.values())
        self.transitions.append({"utc": utc, "monotonic_s": m0, "attempt_id": attempt_id, "reason": reason,
                                 "target": gov, "before": json.dumps(sorted({str(v) for v in before.values()})),
                                 "after": json.dumps(sorted({str(v) for v in after.values()})), "ok": ok,
                                 "error": err, "switch_s": time.monotonic() - m0})
        if not ok:
            log(f"governor switch to {gov} FAILED ({reason}): {err or after}")
        return ok

    def restore_original(self) -> bool:
        ok = True
        for path, gov in self.original.items():
            if gov and read_governors().get(path) != gov:
                try:
                    _write_governor(path, gov, self.method)
                except Exception:  # noqa: BLE001
                    ok = False
        self.transitions.append({"utc": now_utc(), "monotonic_s": time.monotonic(), "attempt_id": "",
                                 "reason": "restore_original_at_exit", "target": json.dumps(self.original),
                                 "before": "", "after": json.dumps(read_governors()), "ok": ok, "error": "",
                                 "switch_s": ""})
        return ok


def wait_perf_clock(timeout_s: float, frac: float) -> dict:
    """After switching to performance: cpu0 scaling_cur_freq and vcgencmd ARM clock must reach frac x cpuinfo_max."""
    raw = p1a.read_text("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    max_khz = int(raw) if raw and raw.strip().isdigit() else 0
    t0 = time.monotonic()
    while True:
        cur, arm = p1d0.read_cur_freq_khz(), p1d0.read_arm_clock_hz()
        ok = (max_khz > 0 and cur is not None and cur >= frac * max_khz and arm is not None
              and arm >= frac * max_khz * 1000)
        waited = time.monotonic() - t0
        if ok or waited >= timeout_s:
            return {"ok": ok, "cur_freq_khz": cur, "arm_clock_hz": arm, "cpuinfo_max_khz": max_khz,
                    "fraction_required": frac, "waited_s": waited}
        time.sleep(0.1)


def start_gate_reasons(a, snap: dict, expect_gov: str) -> List[str]:
    r = []
    if snap["temp_c"] is None or snap["temp_c"] > a.max_start_temp_c + p1d0.START_TEMP_JITTER_C:
        r.append(f"temperature {snap['temp_c']} C > {a.max_start_temp_c} (+{p1d0.START_TEMP_JITTER_C}) C")
    if snap["throttled"]["low"] != 0:
        r.append(f"current throttle bits {snap['throttled']['value_hex']}")
    govs = {str(v) for v in snap["governor"].values()}
    if govs != {expect_gov}:
        r.append(f"governors {sorted(govs)} != {expect_gov}")
    return r


def acquire_thermal_start(c: Ctx, attempt_id: str, warmup_fn=None) -> dict:
    """State machine (untimed):
    COOLDOWN(ondemand) -> [<= max_start_temp_c, throttle bits 0-3 clear] -> SWITCH(performance) -> VERIFY(governors,
    throttle, temperature gate, ARM clock) -> [WARMUP] -> FRESH START GATE -> READY (caller starts the timed region).
    Any failed check -> back to ondemand, cool again (rejection); > max_start_rejections -> not started."""
    a, gov = c.args, c.gov
    rej, history = 0, []
    while True:
        if not gov.set("ondemand", "cooldown", attempt_id):
            return {"ok": False, "outcome": "GOVERNOR_SWITCH_FAILED_ONDEMAND", "rejections": rej, "history": history}
        cool = p1d0.wait_until_cool(a.max_start_temp_c, a.max_cooldown_seconds, a.cooldown_poll_seconds)
        if cool["outcome"] != "REACHED":
            return {"ok": False, "outcome": f"COOLDOWN_{cool['outcome']}", "cool": cool, "rejections": rej,
                    "history": history}
        if not gov.set("performance", "timed_region_start", attempt_id):
            gov.set("ondemand", "performance_switch_failed", attempt_id)
            return {"ok": False, "outcome": "GOVERNOR_SWITCH_FAILED_PERFORMANCE", "cool": cool, "rejections": rej,
                    "history": history}
        clock = wait_perf_clock(a.perf_clock_timeout_seconds, a.perf_clock_fraction)
        snap = p1d0.snapshot()
        reasons = start_gate_reasons(a, snap, "performance")
        if not clock["ok"]:
            reasons.append(f"ARM clock not at performance state {clock}")
        stage = "post_switch"
        if not reasons and warmup_fn is not None:
            warmup_fn()
            snap = p1d0.snapshot()
            reasons = start_gate_reasons(a, snap, "performance")
            stage = "post_warmup"
        history.append({"stage": stage, "utc": snap["utc"], "temp_c": snap["temp_c"],
                        "throttled": snap["throttled"]["value_hex"], "cooldown_waited_s": cool["waited_s"],
                        "clock": clock, "reasons": reasons})
        if not reasons:
            return {"ok": True, "outcome": "READY", "cool": cool, "before": snap, "clock": clock,
                    "rejections": rej, "history": history}
        rej += 1
        log(f"{attempt_id}: start gate rejected at {stage}: {reasons} (rejection {rej})")
        if rej > a.max_start_rejections:
            gov.set("ondemand", "start_gate_rejected", attempt_id)
            return {"ok": False, "outcome": "START_GATE_REJECTED", "cool": cool, "rejections": rej,
                    "history": history}


START_FAIL_FIELDS = ["attempt_id", "planned_cell_id", "outcome", "rejections", "history", "utc"]


def run_plan(c: Ctx, out: Path, plan: List[dict], passes: int, warmup_inputs: int) -> dict:
    a = c.args
    status = {"planned": len(plan), "accepted": 0, "not_started": {}, "unresolved": [], "start_rejections": 0,
              "executed_attempts": 0}
    for cell in plan:
        accepted = False
        for attempt in range(1, a.max_cell_attempts + 1):
            attempt_id = f"{cell['planned_cell_id']}_a{attempt}"

            def warm(block=cell["block"], aid=attempt_id, pid=cell["planned_cell_id"]):
                with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):  # untimed, discarded
                    cell_body(c, block, aid, pid, c.gis[:warmup_inputs], 1, False, {k: 0 for k in BAD_KEYS})

            acq = acquire_thermal_start(c, attempt_id, warm)
            status["start_rejections"] += acq["rejections"]
            if not acq["ok"]:
                status["not_started"][cell["planned_cell_id"]] = acq["outcome"]
                append_csv(out / "cell_start_failures.csv", START_FAIL_FIELDS, [{
                    "attempt_id": attempt_id, "planned_cell_id": cell["planned_cell_id"], "outcome": acq["outcome"],
                    "rejections": acq["rejections"], "history": json.dumps(acq["history"], default=str),
                    "utc": now_utc()}])
                log(f"{cell['planned_cell_id']}: NOT STARTED ({acq['outcome']})")
                break
            before, cool = acq["before"], acq["cool"]
            thr_before = [c.torch.get_num_threads(), c.torch.get_num_interop_threads()]
            bad = {k: 0 for k in BAD_KEYS}
            chip_before = hailo_chip_temp(c)
            sampler = p1d0.CellSampler(attempt_id, cell["order_index"], a.in_cell_temp_poll_seconds,
                                       a.in_cell_vc_poll_seconds).start()
            t_cell = time.monotonic()
            gc.collect()
            gc.disable()
            try:  # ---- timed region (performance governor; no governor switching inside)
                with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):  # adapters print per call
                    rows = cell_body(c, cell["block"], attempt_id, cell["planned_cell_id"], c.gis, passes, True, bad)
            finally:
                gc.enable()
            cell_s = time.monotonic() - t_cell
            samples = sampler.stop()
            after = p1d0.snapshot()
            chip_after = hailo_chip_temp(c)
            thr_after = [c.torch.get_num_threads(), c.torch.get_num_interop_threads()]
            c.gov.set("ondemand", "post_cell", attempt_id)  # after the post-snapshot; cooldown only under ondemand
            status["executed_attempts"] += 1
            th_status, reasons = p1d0.classify_thermal(before, after, samples, cool, a.max_start_temp_c)
            if {str(v) for v in after["governor"].values()} != {"performance"}:
                th_status, reasons = "THERMAL_REVIEW", reasons + ["governor not performance at end of timed region"]
            if tuple(thr_before) != c.expected_threads or thr_before != thr_after:
                th_status = "THERMAL_REVIEW"
                reasons = reasons + [f"torch threads {thr_before}->{thr_after} != {list(c.expected_threads)}"]
            correct = "PASS" if not any(bad.values()) else "FAIL"
            ok = th_status == "THERMAL_CLEAN" and correct == "PASS"
            temps = [s["temp_c"] for s in samples if s["temp_c"] is not None]
            append_csv(out / "raw_calls.csv", RAW_FIELDS, rows)
            append_csv(out / "thermal_trace.csv", TRACE_FIELDS, [dict(s, attempt_id=attempt_id) for s in samples])
            append_csv(out / "cells.csv", CELL_FIELDS, [{
                "attempt_id": attempt_id, "planned_cell_id": cell["planned_cell_id"], "order_index": cell["order_index"],
                "replication": cell["replication"], "block": cell["block"], "attempt": attempt, "accepted": ok,
                "correctness": correct, "correctness_detail": json.dumps(bad), "thermal_status": th_status,
                "thermal_reasons": " | ".join(reasons), "start_rejections": acq["rejections"],
                "cooldown_waited_s": cool["waited_s"], "temp_before_c": before["temp_c"], "temp_after_c": after["temp_c"],
                "temp_max_in_cell_c": max(temps) if temps else "", "throttled_before": before["throttled"]["value_hex"],
                "throttled_after": after["throttled"]["value_hex"], "arm_clock_before_hz": before["arm_clock_hz"],
                "arm_clock_after_hz": after["arm_clock_hz"], "governor_before": json.dumps(before["governor"]),
                "governor_after": json.dumps(after["governor"]), "hailo_chip_temp_before": chip_before,
                "hailo_chip_temp_after": chip_after, "torch_threads_before": json.dumps(thr_before),
                "torch_threads_after": json.dumps(thr_after), "cell_seconds": cell_s, "n_timed_rows": len(rows),
                "configuration_role": c.role, "perf_clock_check": json.dumps(acq["clock"]),
                "start_gate_history": json.dumps(acq["history"], default=str)}])
            log(f"{attempt_id}: {th_status} correctness={correct} timed {cell_s:.1f} s cooldown {cool['waited_s']:.0f} s "
                f"T {before['temp_c']}->{after['temp_c']} C max {max(temps) if temps else 'NA'} {reasons}")
            if correct == "FAIL":
                break  # correctness failure is not thermal: no retry
            if ok:
                accepted = True
                break
        if accepted:
            status["accepted"] += 1
        elif cell["planned_cell_id"] not in status["not_started"]:
            status["unresolved"].append(cell["planned_cell_id"])
    status["adaptive_instrumentation_checks"] = dict(c.ak_checks)
    return status


def global_warmup(c: Ctx, rounds: int) -> None:
    """Untimed: sensing, every defense and both classifiers on a few inputs (lazy init, allocator, HailoRT)."""
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        for _ in range(rounds):
            for gi in c.gis[:5]:
                sense(c, gi)
                for d in DEFENSES:
                    y = run_defense(c, d, c.x[("clean", gi)])
                    c.awn.infer(y)
                    c.hailo.infer(y)


# =========================================================================== auxiliary NPU hardware latency
def hw_latency_probe(c: Ctx, out: Path, frames: int) -> dict:
    """Same governor / thermal state machine as a cell (no warm-up). If the start gate cannot be met the probe is NOT
    run and is recorded thermally invalid; a probe whose post-run state is THERMAL_REVIEW is excluded from formal
    auxiliary reporting (its values are kept only under 'excluded_values')."""
    close_hailo(c)  # the device must be free for hailortcli
    a = c.args
    res = {"class": AUX_HW[0], "definition": AUX_HW[1], "attempts": [], "hw_latency_ms": None,
           "overall_latency_ms": None, "thermally_valid": False, "excluded_from_formal_auxiliary_reporting": True}
    acq = acquire_thermal_start(c, "hw_latency_probe")
    res["start"] = {k: v for k, v in acq.items() if k != "before"}
    if not acq["ok"]:
        res["thermal_status"] = f"NOT_RUN_{acq['outcome']}"
        pb.atomic_write_text(out / "hw_latency_auxiliary.json", json.dumps(res, indent=2, default=str))
        return res
    before = acq["before"]
    for cmd in (["hailortcli", "run", c.hef, "--measure-latency", "--measure-overall-latency", "--batch-size", "1",
                 "--frames-count", str(frames)],
                ["hailortcli", "run", c.hef, "--measure-latency", "--batch-size", "1", "--frames-count", str(frames)]):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            txt = (p.stdout or "") + "\n" + (p.stderr or "")
            att = {"cmd": cmd, "returncode": p.returncode, "output_tail": txt[-3000:]}
            for key, pat in (("hw_latency_ms", r"hw\s*latency[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*ms"),
                             ("overall_latency_ms", r"overall\s*latency[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*ms"),
                             ("fps", r"\bfps[^0-9]*([0-9]+(?:\.[0-9]+)?)")):
                m = re.search(pat, txt, re.I)
                att[key] = float(m.group(1)) if m else None
            res["attempts"].append(att)
            if p.returncode == 0 and att["hw_latency_ms"] is not None:
                break
        except Exception as exc:  # noqa: BLE001 - auxiliary
            res["attempts"].append({"cmd": cmd, "error": repr(exc)})
    after = p1d0.snapshot()
    c.gov.set("ondemand", "post_hw_latency_probe", "hw_latency_probe")
    th_status, reasons = p1d0.classify_thermal(before, after, [], acq["cool"], a.max_start_temp_c)
    res.update({"before": before, "after": after, "thermal_status": th_status, "thermal_reasons": reasons})
    ok = [x for x in res["attempts"] if x.get("hw_latency_ms") is not None]
    values = {"hw_latency_ms": ok[-1]["hw_latency_ms"] if ok else None,
              "overall_latency_ms": ok[-1].get("overall_latency_ms") if ok else None}
    if th_status == "THERMAL_CLEAN":
        res.update(values, thermally_valid=True, excluded_from_formal_auxiliary_reporting=False)
    else:
        res["excluded_values"] = values
    pb.atomic_write_text(out / "hw_latency_auxiliary.json", json.dumps(res, indent=2, default=str))
    return res


# =========================================================================== statistics
def q(a: np.ndarray, p: float) -> float:
    return float(np.percentile(a, p)) if a.size else float("nan")


def dist(v: List[float]) -> dict:
    a = np.asarray(v, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean_ms": float(a.mean()), "std_ms": float(a.std(ddof=0)), "median_ms": q(a, 50),
            "p05_ms": q(a, 5), "p25_ms": q(a, 25), "p75_ms": q(a, 75), "p95_ms": q(a, 95), "p99_ms": q(a, 99),
            "max_ms": float(a.max())}


def sign_test(diffs: np.ndarray) -> float:
    nz = diffs[diffs != 0]
    n, k = int(nz.size), int((nz > 0).sum())
    if n == 0:
        return float("nan")
    k = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int = 0) -> dict:
    d = b - a
    rng = np.random.default_rng(seed)
    boots = np.median(d[rng.integers(0, d.size, size=(n_boot, d.size))], axis=1)
    return {"n_pairs": int(d.size), "median_a_ms": q(a, 50), "median_b_ms": q(b, 50),
            "median_diff_b_minus_a_ms": q(d, 50), "diff_ci95_lo_ms": q(boots, 2.5), "diff_ci95_hi_ms": q(boots, 97.5),
            "mean_diff_ms": float(d.mean()), "median_ratio_b_over_a": float(np.nanmedian(b / np.where(a > 0, a, np.nan))),
            "n_b_slower": int((d > 0).sum()), "n_b_faster": int((d < 0).sum()), "sign_test_p": sign_test(d)}


def stage_summarize(out: Path, n_boot: int, role: str) -> dict:
    """Never raises on missing results: zero executed / zero accepted cells -> FAIL with an explicit reason."""
    if not (out / "cells.csv").exists() or not (out / "raw_calls.csv").exists():
        return {"gates": {"accepted_cells_nonzero": False, "result_files_present": False},
                "failure_reason": "NO_ACCEPTED_CELLS: no timed cell was executed (thermal cooldown / start-gate failure; "
                                  "see cell_start_failures.csv); summary not computed",
                "accepted_cells": 0}
    cells = pb.read_csv(out / "cells.csv")
    acc = {r["attempt_id"] for r in cells if r["accepted"] == "True"}
    if not acc:
        return {"gates": {"accepted_cells_nonzero": False, "result_files_present": True},
                "failure_reason": "NO_ACCEPTED_CELLS: timed cells ran but none was accepted (THERMAL_REVIEW and/or "
                                  "correctness FAIL; see cells.csv); summary not computed",
                "accepted_cells": 0, "attempts": len(cells)}
    data: Dict[tuple, List[float]] = {}
    per_input: Dict[tuple, List[float]] = {}
    per_cell: Dict[tuple, List[float]] = {}
    with (out / "raw_calls.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            if r["attempt_id"] not in acc:
                continue
            ms = float(r["ms"])
            kind = "clean" if r["block"] == "clean" else "attacked"
            for k_kind in ("all", kind):
                data.setdefault((r["component"], r["backend"], r["defense"], k_kind, ""), []).append(ms)
                if r["defense"] == "adaptive_k_v2" and r["branch"]:
                    data.setdefault((r["component"], r["backend"], r["defense"], k_kind, r["branch"]), []).append(ms)
            per_input.setdefault((r["component"], r["backend"], r["defense"], r["block"], r["global_index"]),
                                 []).append(ms)
            per_cell.setdefault((r["planned_cell_id"], r["component"], r["backend"], r["defense"]), []).append(ms)
    comp_rows = []
    for k, v in sorted(data.items()):
        letter, cls, _ = METRICS.get(k[0], ("?", "?", ""))
        comp_rows.append({"metric": letter, "metric_class": cls, "component": k[0], "backend": k[1], "defense": k[2],
                          "input_kind": k[3], "ak_branch": k[4], "configuration_role": role, **dist(v)})
    fields = ["metric", "metric_class", "component", "backend", "defense", "input_kind", "ak_branch",
              "configuration_role", "n", "mean_ms", "std_ms", "median_ms", "p05_ms", "p25_ms", "p75_ms", "p95_ms",
              "p99_ms", "max_ms"]
    pb.write_csv(out / "component_summary.csv", fields, comp_rows)
    # measured timings only (evaluation-path sensing / pipeline rows remain evaluation-path, not deployment latency)
    pb.write_csv(out / "primary_metrics_measured.csv", fields,
                 [r for r in comp_rows if r["metric_class"] in ("measured", "measured_composite")
                  and r["input_kind"] == "all" and r["ak_branch"] == ""])
    # deployment-relevant DERIVED estimates, kept in a separate file so they are never read as measured
    pb.write_csv(out / "deployment_estimates_derived.csv", fields,
                 [r for r in comp_rows if r["component"] in ("A_sensing_core_derived_ms",)
                  + tuple(DEPLOYMENT_ESTIMATE.values()) and r["ak_branch"] == ""])
    pb.write_csv(out / "cell_medians.csv", ["planned_cell_id", "component", "backend", "defense", "median_ms", "n"],
                 [{"planned_cell_id": k[0], "component": k[1], "backend": k[2], "defense": k[3],
                   "median_ms": q(np.asarray(v), 50), "n": len(v)} for k, v in sorted(per_cell.items())])
    med = {k: float(np.median(v)) for k, v in per_input.items()}
    inputs = sorted({(k[3], k[4]) for k in med})
    pairs = []

    def add(label, cls, a_key, b_key, note=""):
        va = [med.get(a_key + i) for i in inputs]
        vb = [med.get(b_key + i) for i in inputs]
        keep = [(x, y) for x, y in zip(va, vb) if x is not None and y is not None]
        if keep:
            pairs.append({"comparison": label, "class": cls, "a": "|".join(a_key), "b": "|".join(b_key),
                          "configuration_role": role, "note": note,
                          **paired(np.asarray([x for x, _ in keep]), np.asarray([y for _, y in keep]), n_boot)})

    for d in DEFENSES:
        add("F_minus_E_hailo_minus_cpu_classifier", "derived_paired", ("E_cpu_awn_adapter_ms", "cpu", d),
            ("F_hailo_adapter_roundtrip_ms", "hailo", d))
        add("F_adapter_wrapper_overhead_estimate", "derived_paired", ("F_diag_hailort_infervstreams_ms", "hailo", d),
            ("F_hailo_adapter_roundtrip_ms", "hailo", d),
            "adapter wall - bare HailoRT call wall; NOT PCIe / host-to-Hailo transfer overhead")
        add("E_adapter_wrapper_overhead_estimate", "derived_paired", ("E_diag_cpu_forward_bare_ms", "cpu", d),
            ("E_cpu_awn_adapter_ms", "cpu", d))
        for comp in list(EVAL_PATH_MEASURED.values()) + list(DEPLOYMENT_ESTIMATE.values()):
            add(f"{comp}__hailo_minus_cpu", "derived_paired", (comp, "cpu", d), (comp, "hailo", d))
        for clean in (True, False):  # measured evaluation overhead, kept separate from the deployment estimate
            for b in ("cpu", "hailo"):
                add(f"{EVAL_PATH_MEASURED[clean][0]}_evaluation_overhead_measured_minus_deployment_estimate_{b}",
                    "derived_paired", (DEPLOYMENT_ESTIMATE[clean], b, d), (EVAL_PATH_MEASURED[clean], b, d),
                    "evaluation-path measured pipeline minus derived deployment estimate (evaluation bookkeeping, "
                    "second sensing call and timer granularity); not a deployment latency")
    add("A_sensing_evaluation_overhead", "derived_paired", ("A_sensing_core_derived_ms", "host", ""),
        ("A_sensing_evaluation_path_wall_ms", "host", ""),
        "measured evaluation-path wall minus derived core stage sum = evaluation-only bookkeeping + untimed glue")
    for comp in list(EVAL_PATH_MEASURED.values()) + list(DEPLOYMENT_ESTIMATE.values()):
        for b in ("cpu", "hailo"):
            add(f"{comp}__adaptive_minus_none_{b}", "derived_paired", (comp, b, "none"), (comp, b, "adaptive_k_v2"))
    for d in FIXED_K:
        add(f"C_minus_B_adaptive_minus_{d}", "derived_paired", ("B_host_fixedk_defense_ms", "host", d),
            ("C_host_adaptive_production_ms", "host", "adaptive_k_v2"))
    if pairs:
        pb.write_csv(out / "paired_statistics.csv", list(pairs[0].keys()), pairs)
    plan_path = out / "plan.json"
    planned = len(json.loads(plan_path.read_text())) if plan_path.exists() else len({r["planned_cell_id"] for r in cells})
    acc_cells = {r["planned_cell_id"] for r in cells if r["accepted"] == "True"}
    corr_fail = [r["attempt_id"] for r in cells if r["correctness"] != "PASS"]
    measured_cells = [float(r["cell_seconds"]) for r in cells]
    cooldowns = [float(r["cooldown_waited_s"]) for r in cells]
    gates = {"accepted_cells_nonzero": True, "result_files_present": True,
             "all_planned_cells_accepted": planned > 0 and len(acc_cells) == planned,
             "correctness_all_attempts_pass": not corr_fail,
             "accepted_cells_thermal_clean": all(r["thermal_status"] == "THERMAL_CLEAN"
                                                 for r in cells if r["accepted"] == "True"),
             "configuration_role_consistent": {r["configuration_role"] for r in cells} == {role},
             "paired_statistics_written": bool(pairs)}
    return {"gates": gates, "planned_cells": planned, "accepted_cells": len(acc_cells), "attempts": len(cells),
            "correctness_failures": corr_fail, "n_pair_rows": len(pairs), "n_bootstrap": n_boot,
            "measured_cell_seconds": {k.replace("_ms", "_s"): v for k, v in dist(measured_cells).items()},
            "measured_cooldown_seconds": {k.replace("_ms", "_s"): v for k, v in dist(cooldowns).items()}}


# =========================================================================== preflight
def stage_preflight(c: Ctx) -> dict:
    g, info = {}, {"utc": now_utc(), "configuration_role": c.role}
    for name, d in (("phaseB", PHASE_B_DIR), ("phaseC", PHASE_C_DIR)):
        g[f"{name}_validation_pass"] = json.loads((REPO_ROOT / d / "validation.json").read_text()).get("overall") == "PASS"
    hef = Path(c.cfg["hef_path"]).resolve()
    g["hef_sha256_ok"] = hef.is_file() and pb.sha_file(hef) == EXPECTED_HEF_SHA256
    man = json.loads((REPO_ROOT / PHASE_B_DIR / "manifest.json").read_text())
    g["dataset_checkpoint_sha_ok"] = (pb.sha_file(REPO_ROOT / c.cfg["dataset_path"]) == man["dataset"]["sha256"]
                                      and pb.sha_file(REPO_ROOT / c.cfg["checkpoint_path"]) == man["checkpoint"]["sha256"])
    scan = p1a.run_cmd(["hailortcli", "scan"]) or ""
    info["hailortcli_scan"] = scan
    info["hailortcli_identify"] = p1a.run_cmd(["hailortcli", "fw-control", "identify"])
    g["hailo_device_available"] = EXPECTED_HAILO_DEVICE in scan
    snap = p1d0.snapshot()
    info["snapshot"] = snap
    g["vcgencmd_available"] = snap["throttled"]["low"] is not None
    g["temperature_readable"] = snap["temp_c"] is not None
    g["throttle_current_bits_clear"] = snap["throttled"]["low"] == 0
    info["throttle_history_bits"] = snap["throttled"]["high_names"]
    info["reboot_recommended_history_bits_set"] = bool(snap["throttled"]["high"])
    govs = read_governors()
    info["governors"] = govs
    g["governors_readable_all_cpus"] = (len(govs) == (os.cpu_count() or 0) > 0
                                        and all(v is not None for v in govs.values()))
    g["governor_ondemand_or_performance"] = bool(govs) and {str(v) for v in govs.values()} <= {"ondemand",
                                                                                               "performance"}
    avail = p1a.read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors") or ""
    info["available_governors"] = avail
    g["governors_ondemand_and_performance_available"] = {"ondemand", "performance"} <= set(avail.split())
    method, detail = detect_governor_write_method()  # no-op rewrite of the current value
    c.gov_method = method
    info["governor_write_method"], info["governor_write_probe"] = method, detail
    g["governor_writable"] = method is not None
    info["cpufreq"] = {k: p1a.read_text(f"/sys/devices/system/cpu/cpu0/cpufreq/{k}")
                       for k in ("scaling_min_freq", "scaling_max_freq", "cpuinfo_max_freq")}
    info["thread_env"] = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                         "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    if c.role == ROLE_PRIMARY:
        g["no_thread_env_overrides"] = not any(info["thread_env"].values())
    info["loadavg"] = p1a.read_text("/proc/loadavg")
    try:
        la = float((info["loadavg"] or "nan").split()[0])
    except ValueError:
        la = float("nan")
    g["system_idle_loadavg_below_0_5"] = la < 0.5
    g["replications_even"] = c.dry or c.args.replications % 2 == 0
    info["plan_cells"] = len(build_plan(c, c.args.replications))
    info["blocks"], info["inputs_per_block"] = c.blocks, len(c.gis)
    info["runtime_estimate"] = "not stated before the dry-run has measured cell and cooldown durations"
    return {"gates": g, "info": info}


def projection_from_dry_run(sm: dict, c: Ctx) -> dict:
    """Projection from the MEASURED dry-run cells (scaled by inputs x passes); labelled as a projection only."""
    cs, cd = sm.get("measured_cell_seconds", {}), sm.get("measured_cooldown_seconds", {})
    if not cs.get("n"):
        return {"note": "no measured dry-run cell"}
    dry_units = len(c.gis) * 1
    full_units = len(select_inputs(c.cfg, False)) * c.args.formal_passes
    per_unit = cs["median_s"] / dry_units
    full_cells = (1 + len([p for p in c.args.profiles.split(",") if p.strip()])) * c.args.formal_replications
    return {"class": "projection_from_dry_run_not_a_measurement", "measured_dry_cell_median_s": cs["median_s"],
            "measured_dry_cooldown_median_s": cd.get("median_s"), "full_run_cells": full_cells,
            "projected_full_cell_s": per_unit * full_units,
            "projected_full_run_minutes": full_cells * (per_unit * full_units + (cd.get("median_s") or 0)) / 60,
            "note": "cell-seconds scale with inputs x passes; hotter full cells may lengthen cooldowns"}


def write_manifest(c: Ctx, args, stage_info: dict, out: Path) -> None:
    mp = out / "manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else {"created_utc": now_utc()}
    man.update({
        "phase": "formal_latency_benchmark", "configuration_role": c.role, "last_command": [sys.executable] + sys.argv,
        "updated_utc": now_utc(),
        "protocol": {
            "batch_size": 1, "threads_mode": args.threads_mode,
            "expected_torch_threads_intra_interop": list(FORMAL_THREADS) if c.role == ROLE_PRIMARY
            else [args.intra_threads, args.interop_threads],
            "observed_torch_threads_intra_interop": list(getattr(c, "threads", ()) or ()),
            "blocks": c.blocks, "inputs_per_block": len(c.gis), "passes": args.passes,
            "warmup_inputs_per_cell": args.warmup_inputs, "global_warmup_rounds": args.global_warmup_rounds,
            "replications": args.replications, "pipeline_defenses_G_H": c.pipeline_defenses,
            "thermal": {"method": "Phase 1D-0 helpers (imported unchanged)", "max_start_temp_c": args.max_start_temp_c,
                        "start_temp_jitter_c": p1d0.START_TEMP_JITTER_C, "max_cell_attempts": args.max_cell_attempts,
                        "in_cell_temp_poll_s": args.in_cell_temp_poll_seconds,
                        "in_cell_vc_poll_s": args.in_cell_vc_poll_seconds,
                        "governor_phases": "ondemand during every untimed cooldown; switched to performance "
                                           "immediately before the (untimed) warm-up and the timed region; verified "
                                           "(all governors performance, throttle bits clear, temperature start gate, "
                                           f"ARM clock >= {args.perf_clock_fraction} x cpuinfo_max within "
                                           f"{args.perf_clock_timeout_seconds} s); fresh start gate after warm-up; "
                                           "switched back to ondemand after the post-cell snapshot; any failed gate "
                                           "-> ondemand, cool again, retry",
                        "governor_switch_timing": "never inside any timed metric; every transition recorded in "
                                                  "governor_transitions.csv",
                        "system_settings": "the benchmark intentionally switches ONLY the CPU scaling governor "
                                           "(ondemand for cooldown, performance for timed execution) and restores the "
                                           "original governors at exit; it does not change frequency limits, clocks, "
                                           "fan settings or any other system setting"},
            "gc": "collected before, disabled during each timed cell", "timer": "time.perf_counter_ns",
            "stdout": "adapter per-call prints redirected to os.devnull during cells",
            "order": "blocks forward / reverse per replication; CPU/Hailo order alternates per input; defense order "
                     "rotates per pass",
            "statistics": "accepted attempts only; per-input medians over passes x accepted cells; paired b-a median "
                          "difference, percentile bootstrap 95% CI (seed 0), two-sided sign test"},
        "metrics": {k: {"letter": v[0], "class": v[1], "definition": v[2]} for k, v in METRICS.items()},
        "auxiliary": {"hw_latency_auxiliary.json": {"class": AUX_HW[0], "definition": AUX_HW[1]}},
        "claim_boundaries": [
            "A_sensing_evaluation_path_wall_ms and the G/H *_evaluation_path_measured_ms timers are MEASURED on the "
            "unmodified formal evaluation function, which includes evaluation-only ground-truth metrics and sha256; "
            "they are evaluation-path timings and must never be presented as deployment latency.",
            "A_sensing_core_derived_ms and the G/H *_deployment_estimate_ms values are DERIVED (sums of separately "
            "measured components from the same input and pass); they are deployment-relevant estimates and must never "
            "be presented as measured end-to-end latency.",
            "Measured evaluation overhead (evaluation-path measured minus derived estimate) is reported separately in "
            "paired_statistics.csv and is not part of any deployment estimate.",
            "Only G_clean_pipeline_evaluation_path_measured_ms is a genuinely continuous clean pipeline measurement; "
            "H (measured_composite) is a continuous timer over sensing + post-sensing defense/classifier with the stored "
            "adversarial IQ substituted -- attack generation excluded, not an attacked end-to-end latency.",
            "'derived' values (stage sums, deployment estimates, paired differences, wrapper overhead estimates) are "
            "not measured wall times.",
            "'diagnostic' timings (instrumented Adaptive-K replica, bare forward / bare HailoRT call) are not the "
            "production path.",
            "Hailo adapter / HailoRT times include host-side quantization, transfer and dequantization; no PCIe or NPU "
            "decomposition is claimed; the NPU hardware latency is auxiliary and unpaired.",
            "Host defense and Adaptive-K routing run on the host CPU and are never attributed to the Hailo NPU."],
        "sources": getattr(c, "src", None), "hef_path": c.cfg["hef_path"], "phase_b_dir": PHASE_B_DIR,
        "phase_c_dir": PHASE_C_DIR, "note": pb.LINEAR_NOTE})
    try:
        man["environment"] = p1a.environment_metadata()
    except Exception as exc:  # noqa: BLE001
        man["environment_error"] = repr(exc)
    man.setdefault("stages", []).append({"stage": args.stage, "utc": now_utc(), **stage_info})
    pb.atomic_write_text(mp, json.dumps(man, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["preflight", "dry-run", "run", "summarize"], required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--threads-mode", choices=["formal-default", "explicit"], default="formal-default",
                    help="formal-default = PRIMARY (process defaults, must be 4/4); explicit = SECONDARY sensitivity")
    ap.add_argument("--intra-threads", type=int, default=None, help="explicit mode only")
    ap.add_argument("--interop-threads", type=int, default=None, help="explicit mode only")
    ap.add_argument("--profiles", default=DEFAULT_PROFILES)
    ap.add_argument("--replications", type=int, default=2)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--warmup-inputs", type=int, default=10)
    ap.add_argument("--global-warmup-rounds", type=int, default=5)
    ap.add_argument("--pipeline-defenses", default="none,adaptive_k_v2", help="defenses timed in G / H")
    ap.add_argument("--max-start-temp-c", type=float, default=55.0)
    ap.add_argument("--max-cooldown-seconds", type=float, default=900.0)
    ap.add_argument("--cooldown-poll-seconds", type=float, default=5.0)
    ap.add_argument("--max-start-rejections", type=int, default=3)
    ap.add_argument("--max-cell-attempts", type=int, default=3)
    ap.add_argument("--in-cell-temp-poll-seconds", type=float, default=1.0)
    ap.add_argument("--in-cell-vc-poll-seconds", type=float, default=10.0)
    ap.add_argument("--perf-clock-fraction", type=float, default=0.97,
                    help="after switching to performance, cpu0 cur freq and ARM clock must reach this x cpuinfo_max")
    ap.add_argument("--perf-clock-timeout-seconds", type=float, default=5.0)
    ap.add_argument("--hw-latency-frames", type=int, default=2000)
    ap.add_argument("--skip-hw-latency", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    if args.threads_mode == "explicit" and (args.intra_threads is None or args.interop_threads is None):
        p1a.abort("--threads-mode explicit requires --intra-threads and --interop-threads")
    if args.threads_mode == "formal-default" and (args.intra_threads is not None or args.interop_threads is not None):
        p1a.abort("--intra-threads / --interop-threads are only allowed with --threads-mode explicit (SECONDARY)")
    args.formal_passes, args.formal_replications = args.passes, args.replications
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (REPO_ROOT / (args.out_dir or f"results/formal_latency_{ts}")).resolve()
    for pdir in PROTECTED_DIRS:
        prot = (REPO_ROOT / pdir).resolve()
        if out == prot or prot in out.parents or out in prot.parents:
            p1a.abort(f"refusing to write inside/over {pdir}")
    dry = args.stage == "dry-run"
    if dry:
        out = out / "dry_run"
        args.passes, args.warmup_inputs, args.global_warmup_rounds, args.replications = 1, 2, 1, 1
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("run", "dry-run") and (out / "cells.csv").exists():
        p1a.abort(f"{out}/cells.csv exists; use a fresh --out-dir (runs are never mixed)")
    os.environ["HAILORT_LOGGER_PATH"] = str(out)
    os.chdir(out)
    pre_guard = guard_snapshot()
    c = open_context(args, dry)
    log(f"out_dir={out} stage={args.stage} role={c.role}")
    res: dict = {"gates": {}, "info": {}}
    try:
        if args.stage == "summarize":
            res = stage_summarize(out, args.bootstrap, c.role)
            res["info"] = {}
        else:
            pf = stage_preflight(c)
            res = {"gates": dict(pf["gates"]), "info": {"preflight": pf["info"]}}
            if args.stage in ("run", "dry-run"):
                failed = [k for k, v in pf["gates"].items() if not v]
                if failed:
                    raise SystemExit(f"[ABORT] preflight gates failed: {failed}")
                c.gov = GovernorController(c.gov_method)
                res["info"]["original_governors"] = c.gov.original
                open_backends(c)
                res["gates"]["torch_threads_match_role"] = c.threads == c.expected_threads
                log("global warm-up (untimed)")
                global_warmup(c, args.global_warmup_rounds)
                plan = build_plan(c, args.replications)
                pb.atomic_write_text(out / "plan.json", json.dumps(plan, indent=2))
                st = run_plan(c, out, plan, args.passes, args.warmup_inputs)
                res["info"]["run"] = st
                # Adaptive-K instrumentation: 0 calls = NOT_EVALUATED (never reported as a bit-identity mismatch)
                chk = st["adaptive_instrumentation_checks"]
                ak_status = ("NOT_EVALUATED" if chk["calls"] == 0 else
                             "BIT_IDENTICAL" if chk["bit_identical"] == chk["calls"] else "MISMATCH")
                res["info"]["adaptive_instrumentation"] = {"status": ak_status, "calls": chk["calls"],
                                                           "bit_identical": chk["bit_identical"]}
                res["gates"]["adaptive_instrumentation_evaluated_and_bit_identical"] = ak_status == "BIT_IDENTICAL"
                if not args.skip_hw_latency:
                    hw = hw_latency_probe(c, out, args.hw_latency_frames)
                    res["info"]["auxiliary_hw_latency"] = {
                        "thermally_valid": hw["thermally_valid"], "thermal_status": hw.get("thermal_status"),
                        "hw_latency_ms": hw["hw_latency_ms"], "overall_latency_ms": hw["overall_latency_ms"],
                        "excluded_from_formal_auxiliary_reporting": hw["excluded_from_formal_auxiliary_reporting"]}
                if st["accepted"] == 0:  # zero-cell failure: explicit reason, no summary on nonexistent results
                    res["gates"]["accepted_cells_nonzero"] = False
                    res["info"]["failure_reason"] = (
                        "NO_ACCEPTED_CELLS: " + ("thermal cooldown / start-gate failure, no timed cell executed "
                                                 f"(not started: {st['not_started']})" if st["executed_attempts"] == 0
                                                 else "timed cells executed but none accepted (see cells.csv)"))
                    log(res["info"]["failure_reason"])
                else:
                    sm = stage_summarize(out, args.bootstrap, c.role)
                    res["gates"].update(sm["gates"])
                    res["info"]["summary"] = {k: v for k, v in sm.items() if k != "gates"}
                    if dry:
                        res["info"]["projection"] = projection_from_dry_run(sm, c)
                res["gates"]["no_attack_generation"] = True  # no AttackAdapter constructed; attacks read from store
        res["gates"]["completed_without_abort"] = True
    except (SystemExit, Exception) as exc:  # noqa: BLE001 -- fail closed but always write validation
        res.setdefault("gates", {})["completed_without_abort"] = False
        res.setdefault("info", {})["abort"] = repr(exc)
    finally:
        close_hailo(c)
        gov = getattr(c, "gov", None)
        if gov is not None:  # restore the governors found at start; log every transition
            res.setdefault("info", {})["governor_restored_at_exit"] = gov.restore_original()
            append_csv(out / "governor_transitions.csv", GOV_FIELDS, gov.transitions)
            res["info"]["governor_transitions"] = len(gov.transitions)
            res["info"]["governor_transitions_failed"] = sum(1 for t in gov.transitions if not t["ok"])
    guard = guard_compare(pre_guard, guard_snapshot())
    res["gates"]["no_protected_modification"] = guard["protected_dirs_unchanged"]
    res["gates"]["no_writes_outside_output_dir"] = all(guard[k] for k in ("protected_dirs_unchanged",
                                                                          "watched_files_unchanged",
                                                                          "git_status_unchanged", "git_diff_unchanged"))
    res["write_guard"] = guard
    res["configuration_role"] = c.role
    res["overall"] = "PASS" if res["gates"] and all(res["gates"].values()) else "FAIL"
    res["stage"], res["utc"] = args.stage, now_utc()
    target = out / ("preflight.json" if args.stage == "preflight" else "validation.json")
    pb.atomic_write_text(target, json.dumps(res, indent=2, default=str))
    write_manifest(c, args, {"overall": res["overall"], "gates": res["gates"]}, out)
    log(f"{args.stage}: overall={res['overall']} role={c.role} -> {target}")
    for k, v in res["gates"].items():
        log(f"  {'PASS' if v else 'FAIL'}  {k}")
    if res["overall"] != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
