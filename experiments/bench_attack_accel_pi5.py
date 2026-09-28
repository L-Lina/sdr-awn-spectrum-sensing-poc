"""
Phase 1A attack-generation acceleration baseline (Raspberry Pi 5, CPU only).

Question answered (and ONLY this question): for batch size B=1 single-sample
attack generation on the Pi 5 Cortex-A76, what is the latency effect of
PyTorch intra-op thread count 1 / 2 / 4 (inter-op threads fixed at 1)?

Scope (deliberately narrow, baseline measurement only -- nothing is optimized):
  - attacks: FGSM eps=0.03, PGD eps=0.03, BIM eps=0.03 (formal profiles
    fgsm/eps0.03, pgd/eps0.03, bim/eps0.03_default, read from the formal config)
  - B = 1 only; no batching, no attack-object reuse, no removal of the duplicate
    clean y_pred forward, no source edits, no hyperparameter/RNG changes.
  - each thread condition runs in its OWN fresh Python process (this script
    re-invokes itself with --worker), so torch thread settings can never leak
    between conditions.

Two timing paths are measured for every (sample, attack, repeat):
  1. "adapter_apply": the UNTOUCHED src.adapters.attack_adapter.AttackAdapter.apply
     call, timed exactly like the formal runner's attack_ms
     (t0 = time.perf_counter(); apply(...); elapsed).
  2. "instrumented": a step-for-step replica of AttackAdapter.apply's real-attack
     branch, built from the SAME module-level helpers (_iq_to_ta_input_minmax,
     TemperatureLogitsWrapper, _build_torchattacks, _ta_output_to_iq_minmax), with
     perf_counter marks between phases A..E. It is a separate call and is NEVER
     reported as formal-equivalent latency. Its fidelity is checked per call by
     comparing its x_adv against the untouched apply() output (same torch seed).

Formal-input fidelity: the formal CPU full-matrix runner is not imported here.
Each source sample is rebuilt with the repo's own sensing chain (RadioML sample ->
embed_sample_in_noise -> energy_detect -> region post-process -> max-energy
segment -> AWN preprocess) using the per-row seed RECORDED in the formal
base_results.csv, and the run ABORTS unless sha256(x_clean) equals the formal
row's clean_input_sha256. So the benchmark provably attacks the exact same AWN
input tensors as the formal run.

Never writes into results/cpu_full_matrix_20260921_seedaligned or
results/hailo_full_matrix_20260920_seedaligned; never modifies external/,
configs/, .har/.hef, or any source file. Real backends only; fails closed.

Run on the Pi (from the repo root):
    .venv/bin/python experiments/bench_attack_accel_pi5.py
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = "configs/hailo_full_matrix.json"
DEFAULT_FORMAL_CPU_DIR = "results/cpu_full_matrix_20260921_seedaligned"
PROTECTED_RESULT_DIRS = (
    "results/cpu_full_matrix_20260921_seedaligned",
    "results/hailo_full_matrix_20260920_seedaligned",
)
FORMAL_LATENCY_TABLE = "reports/meeting_20260921/tables/attack_generation_latency.csv"

# Deterministic stratified subset: 11 modulations (all, from the formal config)
# x 4 representative SNRs (low / mid-low / mid-high / high, 12 dB spacing across
# the formal -20..18 dB grid) x 2 sample indices = 88 source samples.
SUBSET_SNRS = [-20, -6, 6, 18]
SUBSET_SAMPLE_INDICES = [0, 1]

# attack name -> formal profile id (as it appears in attack_results.csv and in
# reports/meeting_20260921/tables/attack_generation_latency.csv)
TARGET_PROFILES: Dict[str, str] = {"fgsm": "eps0.03", "pgd": "eps0.03", "bim": "eps0.03_default"}
EXPECTED_EPS = 0.03
DETERMINISTIC_ATTACKS = {"fgsm", "bim"}  # pgd uses torchattacks' default random_start=True

ENV_THREAD_VARS = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]

# Sensing parameters: looked up in the formal config's "sensing" block under
# these aliases; if absent, the documented repo defaults below are used (the
# same constants as experiments/benchmark_pipeline_latency.py). Either way the
# rebuilt x_clean must hash-match the formal row, or the run aborts.
SENSING_KEYS: Dict[str, Tuple[Tuple[str, ...], object]] = {
    "n_samples": (("n_samples", "stream_len", "num_samples"), 8192),
    "embed_snr_margin": (("embed_snr_margin", "snr_margin"), 20.0),
    "threshold_factor": (("threshold_factor", "energy_threshold_factor"), 5.0),
    "window": (("sensing_window_size", "window", "window_size", "energy_window"), 128),
    "min_region_len": (("min_region_len", "min_len"), 128),
    "merge_gap": (("merge_gap",), 0),
    "alignment_policy": (("alignment_policy", "policy"), "max-energy"),
    "awn_preprocess": (("awn_preprocess", "preprocess_policy"), "radioml-native"),
    "seg_len": (("seg_len", "segment_len", "segment_length"), 128),
    "hop": (("segment_hop", "hop", "alignment_hop"), 1),
}

RAW_FIELDS = [
    "condition", "round", "intra_threads_requested", "interop_threads_requested",
    "intra_threads_actual", "interop_threads_actual",
    "attack", "attack_profile", "eps", "modulation", "snr_db", "sample_index", "formal_seed",
    "repeat", "path", "path_order", "torch_call_seed",
    "total_ms",
    "phase_A_prep_ms", "phase_B_clean_forward_ms", "phase_C_construct_ms", "phase_D_attack_ms",
    "phase_diag_ms", "phase_E_post_ms",
    "clean_pred", "attacked_pred", "label", "clean_correct", "attacked_correct", "attack_success",
    "iq_linf", "iq_l2", "iq_linf_normalized", "x_adv_sha256", "paths_xadv_equal",
    "cpu_temp_c",
]


# --------------------------------------------------------------------------- utils
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def rel(p: Path) -> str:
    """Repo-relative path when possible (worker re-joins it onto REPO_ROOT; absolute paths survive that join)."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(p).resolve())


def run_cmd(cmd: List[str]) -> Optional[str]:
    try:
        return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001 - metadata only
        return None


def read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text().strip()
    except Exception:  # noqa: BLE001 - metadata only
        return None


def cpu_temp_c() -> Optional[float]:
    raw = read_text("/sys/class/thermal/thermal_zone0/temp")
    try:
        return float(raw) / 1000.0 if raw is not None else None
    except ValueError:
        return None


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def abort(msg: str) -> None:
    raise SystemExit(f"[ABORT] {msg}")


def parse_bool(v: str) -> bool:
    if v in ("True", "true", "1"):
        return True
    if v in ("False", "false", "0"):
        return False
    raise ValueError(f"not a strict boolean: {v!r}")


def stats(vals: List[float]) -> Dict[str, float]:
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return {k: float("nan") for k in ("n", "mean", "median", "p95", "p99", "min", "max", "std")}
    # same definitions as the formal latency_summary.csv (np.percentile linear, population std ddof=0)
    return {
        "n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
        "min": float(a.min()), "max": float(a.max()), "std": float(a.std(ddof=0)),
    }


# --------------------------------------------------------------------------- formal config / rows
def load_formal_context(args) -> dict:
    cfg_path = REPO_ROOT / args.config
    formal_dir = REPO_ROOT / args.formal_dir
    for p in (cfg_path, formal_dir / "manifest.json", formal_dir / "base_results.csv", formal_dir / "attack_results.csv"):
        if not p.exists():
            abort(f"required formal artifact missing: {p}")
    cfg = json.loads(cfg_path.read_text())
    manifest = json.loads((formal_dir / "manifest.json").read_text())
    if manifest.get("config") != cfg:
        abort(f"{args.config} differs from the formal CPU manifest's config block; refusing to guess which one ran")
    return {"cfg": cfg, "manifest": manifest, "cfg_path": cfg_path, "formal_dir": formal_dir}


def find_profile(cfg: dict, attack: str, profile_id: str) -> Tuple[dict, dict]:
    for a in cfg.get("attacks", []):
        if str(a.get("name", "")).lower() != attack:
            continue
        for p in a.get("profiles", []):
            if p.get("id") == profile_id:
                return a, p
    abort(f"formal config has no attack '{attack}' profile '{profile_id}'")
    raise AssertionError  # unreachable


def _first_present(dicts: List[Tuple[str, dict]], keys: Tuple[str, ...]):
    for where, d in dicts:
        if not isinstance(d, dict):
            continue
        for k in keys:
            if k in d and d[k] is not None:
                return d[k], f"{where}.{k}"
    return None, None


def resolve_attack_settings(cfg: dict, attack_rows: Dict[tuple, dict], cli_temperature: Optional[float],
                            diagnostics_mode: str) -> Dict[str, dict]:
    """Per attack: eps, attack_params, temperature, diagnostics -- from the formal
    config, cross-checked against the formal attack rows where columns exist."""
    out = {}
    sample_cols = set(next(iter(attack_rows.values())).keys()) if attack_rows else set()
    for attack, pid in TARGET_PROFILES.items():
        entry, prof = find_profile(cfg, attack, pid)
        params = prof.get("attack_params") or {}
        if not isinstance(params, dict):
            abort(f"{attack}/{pid}: attack_params is not a dict: {params!r}")
        scopes = [("profile", prof), ("attack_entry", entry), ("config", cfg), ("config.attack", cfg.get("attack")),
                  ("config.attack_defaults", cfg.get("attack_defaults"))]
        eps, eps_src = _first_present([("profile", prof), ("profile.attack_params", params)], ("eps", "attack_eps", "epsilon"))
        if eps is None:
            abort(f"{attack}/{pid}: cannot find eps in formal profile {prof!r}")
        eps = float(eps)
        if not math.isclose(eps, EXPECTED_EPS, rel_tol=0, abs_tol=1e-12):
            abort(f"{attack}/{pid}: formal eps={eps} != expected {EXPECTED_EPS}")
        if attack == "pgd" and params.get("random_start") is not None:
            log(f"[note] formal pgd profile sets random_start={params['random_start']} explicitly")

        temp_cfg, temp_src = _first_present(scopes, ("attack_temperature", "temperature"))
        rows_for = [r for k, r in attack_rows.items() if k[3] == attack and k[4] == pid]
        temp_rows = None
        for col in ("attack_temperature", "temperature"):
            if col in sample_cols:
                vals = {float(r[col]) for r in rows_for}
                if len(vals) != 1:
                    abort(f"{attack}/{pid}: formal rows carry multiple {col} values {sorted(vals)}")
                temp_rows = vals.pop()
                break
        if temp_cfg is not None and temp_rows is not None and not math.isclose(float(temp_cfg), temp_rows):
            abort(f"{attack}/{pid}: config temperature {temp_cfg} ({temp_src}) != formal row temperature {temp_rows}")
        if cli_temperature is not None:
            temperature, t_src = cli_temperature, "cli --attack-temperature"
            known = temp_cfg if temp_cfg is not None else temp_rows
            if known is not None and not math.isclose(float(known), cli_temperature):
                abort(f"{attack}/{pid}: --attack-temperature {cli_temperature} contradicts formal value {known}")
        elif temp_cfg is not None:
            temperature, t_src = float(temp_cfg), temp_src
        elif temp_rows is not None:
            temperature, t_src = temp_rows, "formal attack_results.csv column"
        else:
            abort(f"{attack}/{pid}: attack temperature not found in formal config or rows; "
                  "pass --attack-temperature explicitly (value used by the formal CPU runner)")

        # diagnostics=True in the formal run would add an extra forward+backward inside apply(),
        # which is part of formal attack_ms. Detect it from non-empty gradient columns.
        grad_cols = [c for c in sample_cols if "gradient" in c.lower()]
        formal_diag = any(r.get(c) not in (None, "", "None", "nan", "NaN") for r in rows_for for c in grad_cols)
        diagnostics = {"auto": formal_diag, "on": True, "off": False}[diagnostics_mode]

        out[attack] = {
            "attack": attack, "profile": pid, "eps": eps, "eps_source": eps_src,
            "attack_params": dict(params), "temperature": float(temperature), "temperature_source": t_src,
            "diagnostics": bool(diagnostics), "diagnostics_mode": diagnostics_mode,
            "formal_gradient_columns": sorted(grad_cols), "formal_diagnostics_detected": bool(formal_diag),
            "formal_profile_raw": prof,
        }
    return out


def load_formal_rows(formal_dir: Path, subset_keys: set) -> Tuple[Dict[tuple, dict], Dict[tuple, dict]]:
    base, attack = {}, {}
    with (formal_dir / "base_results.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            k = (r["modulation"], int(float(r["snr_db"])), int(float(r["sample_index"])))
            if k in subset_keys:
                base[k] = r
    wanted = set(TARGET_PROFILES.items())
    with (formal_dir / "attack_results.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            k = (r["modulation"], int(float(r["snr_db"])), int(float(r["sample_index"])))
            if k in subset_keys and (r["attack"], r["attack_profile"]) in wanted:
                attack[k + (r["attack"], r["attack_profile"])] = r
    missing_b = subset_keys - set(base)
    if missing_b:
        abort(f"formal base_results.csv lacks {len(missing_b)} subset rows, e.g. {sorted(missing_b)[:3]}")
    need = {k + p for k in subset_keys for p in wanted}
    if need - set(attack):
        abort(f"formal attack_results.csv lacks {len(need - set(attack))} subset attack rows")
    return base, attack


def build_subset(cfg: dict) -> List[tuple]:
    mods = list(cfg["modulations"])
    for s in SUBSET_SNRS:
        if s not in cfg["snrs"]:
            abort(f"subset SNR {s} not in formal config snrs")
    for i in SUBSET_SAMPLE_INDICES:
        if i not in cfg["sample_indices"]:
            abort(f"subset sample_index {i} not in formal config sample_indices")
    return [(m, s, i) for m in mods for s in SUBSET_SNRS for i in SUBSET_SAMPLE_INDICES]


def resolve_sensing(cfg: dict) -> Dict[str, dict]:
    block = cfg.get("sensing") or {}
    out = {}
    for name, (aliases, default) in SENSING_KEYS.items():
        val, src = _first_present([("config.sensing", block), ("config", cfg)], aliases)
        out[name] = {"value": val if val is not None else default,
                     "source": src if val is not None else "repo default (benchmark_pipeline_latency.py constant)"}
    return out


# --------------------------------------------------------------------------- worker
def rebuild_clean_inputs(subset, base_rows, sensing, dataset_path: Path) -> Dict[tuple, dict]:
    from src.sensing.energy_detection import energy_detect, filter_by_min_length, mask_to_regions, merge_close_regions
    from src.sensing.normalize import apply_awn_preprocess, to_awn_input
    from src.sensing.radioml_source import RML2016_10A_CLASSES, embed_sample_in_noise, load_radioml_dict
    from src.sensing.segmentation import select_aligned_segments

    sv = {k: v["value"] for k, v in sensing.items()}
    data = load_radioml_dict(str(dataset_path))
    inputs = {}
    for key in subset:
        mod, snr, idx = key
        br = base_rows[key]
        seed = int(float(br["seed"]))
        sample = data[(mod, snr)][idx].astype(np.float32)
        iq, emb = embed_sample_in_noise(sample, int(sv["n_samples"]), float(sv["embed_snr_margin"]), seed=seed)
        if "true_start" in br and emb["true_start"] != int(float(br["true_start"])):
            abort(f"{key}: rebuilt true_start {emb['true_start']} != formal {br['true_start']} (seed/embedding mismatch)")
        mask = energy_detect(iq, window=int(sv["window"]), threshold_factor=float(sv["threshold_factor"]))
        regions = filter_by_min_length(merge_close_regions(mask_to_regions(mask), merge_gap=int(sv["merge_gap"])),
                                       min_len=int(sv["min_region_len"]))
        segs, sel = select_aligned_segments(iq, regions, seg_len=int(sv["seg_len"]),
                                            policy=str(sv["alignment_policy"]), hop=int(sv["hop"]))
        want_start = int(float(br["selected_segment_start"]))
        pick = [j for j, m in enumerate(sel) if int(m["selected_segment_start"]) == want_start]
        if len(pick) != 1:
            abort(f"{key}: formal selected_segment_start={want_start} not uniquely reproduced "
                  f"(rebuilt starts={[m['selected_segment_start'] for m in sel]})")
        j = pick[0]
        x = to_awn_input(apply_awn_preprocess(segs[j:j + 1], policy=str(sv["awn_preprocess"])), seg_len=int(sv["seg_len"]))
        if x.shape != (1, 2, int(sv["seg_len"])) or x.dtype != np.float32:
            abort(f"{key}: unexpected x_clean shape/dtype {x.shape} {x.dtype}")
        h = sha256_array(x)
        if h != br["clean_input_sha256"]:
            abort(f"{key}: rebuilt x_clean sha256 {h} != formal clean_input_sha256 {br['clean_input_sha256']}")
        label = RML2016_10A_CLASSES[mod]
        if "label" in br and int(float(br["label"])) != label:
            abort(f"{key}: label {label} != formal label {br['label']}")
        inputs[key] = {"x": x, "seed": seed, "label": label, "sha256": h,
                       "formal_clean_pred": int(float(br["clean_pred"]))}
    return inputs


def instrumented_apply(am, adapter, torch, x, attack, eps, temperature, seed, diagnostics, attack_params):
    """Step-for-step replica of AttackAdapter.apply()'s real-attack branch
    (src/adapters/attack_adapter.py), same helpers, same order, with timing marks.
    Differences from apply(): raises instead of falling back to dummy_attack
    (fail closed), and returns phase timings. Not formal-equivalent timing."""
    cw_c, cw_steps, cw_lr = 1.0, 20, 0.01  # apply() defaults; unused by fgsm/pgd/bim
    wm = adapter.wrapped_model
    ph = {}
    t_start = time.perf_counter()
    # ---- A: validation, state capture, numpy->torch, min/max normalization, set_minmax
    am.require_positive_finite_float("attack_temperature", temperature)
    am.require_nonneg_finite_float("attack_eps", eps)
    am.require_positive_finite_float("cw_c", cw_c)
    am.require_positive_int("cw_steps", cw_steps)
    am.require_positive_finite_float("cw_lr", cw_lr)
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"AttackAdapter expects input [N, 2, T], got {x.shape}")
    attack_name = am._validate_attack_name(attack)
    input_shape, input_dtype = x.shape, x.dtype
    _ = float(np.min(x)), float(np.max(x))
    training_before = wm.training
    orig_requires_grad = [p.requires_grad for p in wm.parameters()]
    orig_param_devices = [p.device for p in wm.parameters()]
    diag_ms = 0.0
    try:
        x_t = torch.from_numpy(x).to(adapter.device)
        x_ta, a, b = am._iq_to_ta_input_minmax(x_t)
        _ = float(x_ta.min().item()), float(x_ta.max().item())
        wm.set_minmax(a, b)
        ph["phase_A_prep_ms"] = (time.perf_counter() - t_start) * 1000.0
        # ---- B: duplicate clean y_pred forward
        t = time.perf_counter()
        with torch.no_grad():
            y_pred = wm(x_ta).argmax(dim=1)
        ph["phase_B_clean_forward_ms"] = elapsed_ms(t)
        # ---- C: TemperatureLogitsWrapper + attack-object construction
        t = time.perf_counter()
        attack_model = am.TemperatureLogitsWrapper(wm, temperature)
        atk = am._build_torchattacks(attack_name, attack_model, eps, cw_c=cw_c, cw_steps=cw_steps, cw_lr=cw_lr,
                                     attack_params=attack_params)
        ph["phase_C_construct_ms"] = elapsed_ms(t)
        # ---- D: the attack itself
        t = time.perf_counter()
        x_ta_adv = atk(x_ta, y_pred)
        ph["phase_D_attack_ms"] = elapsed_ms(t)
        # ---- E (part 1): diagnostics metric + output conversion
        t_e = time.perf_counter()
        iq_linf_normalized = (x_ta_adv - x_ta).abs().amax(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.float32)
        x_adv_t = am._ta_output_to_iq_minmax(x_ta_adv, a, b)
        x_adv = x_adv_t.detach().cpu().numpy().astype(np.float32)
        e1 = elapsed_ms(t_e)
        if diagnostics:
            t = time.perf_counter()
            wm.eval()
            x_ta_grad = x_ta.clone().detach().requires_grad_(True)
            diag_out = attack_model(x_ta_grad)
            diag_loss = torch.nn.CrossEntropyLoss()(diag_out, y_pred)
            diag_grad = torch.autograd.grad(diag_loss, x_ta_grad, retain_graph=False, create_graph=False)[0]
            _ = (diag_grad != 0).sum(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.int64)
            _ = np.full(x.shape[0], diag_grad[0].numel(), dtype=np.int64)
            _ = diag_grad.abs().amax(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.float32)
            del diag_grad, diag_out, diag_loss, x_ta_grad
            diag_ms = elapsed_ms(t)
    finally:
        # ---- E (part 2): state restoration, identical to apply()'s finally block
        t_r = time.perf_counter()
        if training_before:
            print("[attack_adapter] warning: wrapped model was already in train mode before this call")
        wm.eval()
        wm.clear_minmax()
        for p, req_grad, dev in zip(wm.parameters(), orig_requires_grad, orig_param_devices):
            p.requires_grad_(req_grad)
            if p.device != dev:
                p.data = p.data.to(dev)
        e2 = elapsed_ms(t_r)
    # ---- E (part 3): shape/dtype checks, status print, meta NaN/Inf flags
    t = time.perf_counter()
    _ = wm.training
    if x_adv.shape != input_shape:
        raise RuntimeError(f"output shape {x_adv.shape} != input shape {input_shape}")
    if x_adv.dtype != input_dtype:
        raise RuntimeError(f"output dtype {x_adv.dtype} != input dtype {input_dtype}")
    print(f"[attack_adapter] backend={adapter.backend_name} status=ok input={input_shape} output={x_adv.shape}")
    _ = bool(np.isnan(x_adv).any()), bool(np.isinf(x_adv).any())
    ph["phase_E_post_ms"] = e1 + e2 + elapsed_ms(t)
    ph["phase_diag_ms"] = diag_ms
    ph["total_ms"] = elapsed_ms(t_start)
    return x_adv, ph, iq_linf_normalized


def worker_main(args) -> None:
    import torch  # noqa: E402 -- thread settings must be applied before any parallel work

    requested_intra = args.worker_threads
    if args.worker_formal_default:
        requested_intra, requested_interop = None, None  # PyTorch defaults (formal CPU shell had no overrides)
    else:
        requested_interop = 1
        torch.set_num_interop_threads(1)
        torch.set_num_threads(requested_intra)
    intra, interop = torch.get_num_threads(), torch.get_num_interop_threads()
    if requested_intra is not None and (intra != requested_intra or interop != requested_interop):
        abort(f"thread settings not applied: intra={intra} interop={interop}")

    import src.adapters.attack_adapter as am  # noqa: E402
    from src.adapters.attack_adapter import AttackAdapter, _REAL_ATTACK_SOURCE  # noqa: E402
    from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE  # noqa: E402

    ctx = json.loads(Path(args.worker_context).read_text())
    out_dir = Path(args.out_dir)
    label = args.worker_label
    log(f"[{label}] intra={intra} interop={interop} pid={os.getpid()}")

    awn = AWNModelAdapter(checkpoint_path=str(REPO_ROOT / ctx["checkpoint_path"]), device="cpu")
    if awn.model is None or awn.backend_name != _REAL_MODEL_SOURCE or awn.status != "ok":
        abort(f"AWN backend is not real: backend={awn.backend_name} status={awn.status} notes={awn.notes}")
    # one AttackAdapter instance, exactly like the formal CPU runner
    adapter = AttackAdapter(awn_model=awn.model, device="cpu")
    if adapter.wrapped_model is None or adapter.backend_name != _REAL_ATTACK_SOURCE or adapter.status != "ok":
        abort(f"attack backend is not real: backend={adapter.backend_name} status={adapter.status} notes={adapter.notes}")
    if awn.model.training:
        abort("AWN model is in train mode after load")

    subset = [tuple(k) for k in ctx["subset"]]
    formal_dir = REPO_ROOT / ctx["formal_dir"]
    base_rows, attack_rows = load_formal_rows(formal_dir, set(subset))
    inputs = rebuild_clean_inputs(subset, base_rows, ctx["sensing"], REPO_ROOT / ctx["dataset_path"])
    log(f"[{label}] rebuilt {len(inputs)} x_clean inputs; all sha256 == formal clean_input_sha256")

    def infer_pred(x):
        logits, meta = awn.infer(x)
        if meta["awn_backend"] != _REAL_MODEL_SOURCE or meta["awn_status"] != "ok":
            abort(f"AWN inference fell back: {meta}")
        if not np.isfinite(logits).all():
            abort("non-finite logits")
        return int(np.argmax(logits[0]))

    for key, inp in inputs.items():
        inp["clean_pred"] = infer_pred(inp["x"])

    settings = ctx["attack_settings"]
    attacks = ctx["attacks"]
    req_grad_ref = [p.requires_grad for p in adapter.wrapped_model.parameters()]

    def check_state(where: str) -> None:
        if awn.model.training or adapter.wrapped_model.training:
            abort(f"model entered train mode ({where})")
        if [p.requires_grad for p in adapter.wrapped_model.parameters()] != req_grad_ref:
            abort(f"parameter requires_grad changed ({where})")
        if torch.get_num_threads() != intra or torch.get_num_interop_threads() != interop:
            abort(f"torch thread settings drifted ({where})")

    def run_apply(key, attack):
        s, inp = settings[attack], inputs[key]
        torch.manual_seed(inp["seed"])
        t0 = time.perf_counter()
        x_adv, meta = adapter.apply(inp["x"], attack=attack, eps=s["eps"], temperature=s["temperature"],
                                    seed=inp["seed"], diagnostics=s["diagnostics"],
                                    attack_params=dict(s["attack_params"]))
        ms = elapsed_ms(t0)
        if meta["attack_backend"] != _REAL_ATTACK_SOURCE or meta["attack_status"] != "ok":
            abort(f"attack fell back at runtime: {meta['attack_backend']} {meta['attack_status']} {meta['attack_notes']}")
        if meta["attack_training_after"]:
            abort("attack_training_after is True")
        lin = meta["attack_iq_linf_normalized"]
        return x_adv, {"total_ms": ms}, float(lin[0]) if lin is not None else float("nan")

    def run_instr(key, attack):
        s, inp = settings[attack], inputs[key]
        torch.manual_seed(inp["seed"])
        x_adv, ph, lin = instrumented_apply(am, adapter, torch, inp["x"], attack, s["eps"], s["temperature"],
                                            inp["seed"], s["diagnostics"], dict(s["attack_params"]))
        return x_adv, ph, float(lin[0])

    # ---- warmup (not recorded)
    for attack in attacks:
        for key in subset[: args.warmup]:
            run_apply(key, attack)
            run_instr(key, attack)
    check_state("after warmup")
    log(f"[{label}] warmup done ({args.warmup} samples x {len(attacks)} attacks x 2 paths)")

    raw_path = out_dir / f"raw_{label}.csv"
    n_rows = 0
    with raw_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        for rep in range(args.repeats):
            for si, key in enumerate(subset):
                temp = cpu_temp_c()
                inp = inputs[key]
                for attack in attacks:
                    order = ("adapter_apply", "instrumented") if (rep + si) % 2 == 0 else ("instrumented", "adapter_apply")
                    results = {}
                    for path in order:
                        x_adv, ph, lin = (run_apply if path == "adapter_apply" else run_instr)(key, attack)
                        check_state(f"{attack} {key} {path}")
                        if x_adv.shape != inp["x"].shape or x_adv.dtype != np.float32:
                            abort(f"shape/dtype changed: {x_adv.shape} {x_adv.dtype}")
                        if not np.isfinite(x_adv).all():
                            abort(f"NaN/Inf in x_adv for {attack} {key} {path}")
                        results[path] = (x_adv, ph, lin)
                    equal = bool(np.array_equal(results["adapter_apply"][0], results["instrumented"][0]))
                    for pos, path in enumerate(order):
                        x_adv, ph, lin = results[path]
                        pred = infer_pred(x_adv)
                        diff = x_adv - inp["x"]
                        clean_ok = inp["clean_pred"] == inp["label"]
                        att_ok = pred == inp["label"]
                        row = {
                            "condition": label, "round": args.worker_round,
                            "intra_threads_requested": requested_intra, "interop_threads_requested": requested_interop,
                            "intra_threads_actual": intra, "interop_threads_actual": interop,
                            "attack": attack, "attack_profile": settings[attack]["profile"], "eps": settings[attack]["eps"],
                            "modulation": key[0], "snr_db": key[1], "sample_index": key[2], "formal_seed": inp["seed"],
                            "repeat": rep, "path": path, "path_order": pos, "torch_call_seed": inp["seed"],
                            "clean_pred": inp["clean_pred"], "attacked_pred": pred, "label": inp["label"],
                            "clean_correct": clean_ok, "attacked_correct": att_ok,
                            "attack_success": bool(clean_ok and not att_ok),
                            "iq_linf": float(np.max(np.abs(diff))), "iq_l2": float(np.linalg.norm(diff.reshape(-1))),
                            "iq_linf_normalized": lin, "x_adv_sha256": sha256_array(x_adv),
                            "paths_xadv_equal": equal, "cpu_temp_c": temp,
                        }
                        row.update(ph)
                        w.writerow({k: row.get(k) for k in RAW_FIELDS})
                        n_rows += 1
            log(f"[{label}] repeat {rep + 1}/{args.repeats} done, temp={cpu_temp_c()}")

    worker_meta = {
        "label": label, "round": args.worker_round, "pid": os.getpid(),
        "intra_threads_requested": requested_intra, "interop_threads_requested": requested_interop,
        "intra_threads_actual": intra, "interop_threads_actual": interop,
        "formal_default_control": bool(args.worker_formal_default),
        "env_thread_vars": {k: os.environ.get(k) for k in ENV_THREAD_VARS},
        "torch_version": torch.__version__, "torch_parallel_info": torch.__config__.parallel_info(),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled), "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "awn_backend": awn.backend_name, "attack_backend": adapter.backend_name,
        "attack_adapter_source_sha256": sha256_file(Path(am.__file__)),
        "torchattacks_version": getattr(am._torchattacks, "__version__", None),
        "clean_pred_matches_formal": sum(int(v["clean_pred"] == v["formal_clean_pred"]) for v in inputs.values()),
        "n_inputs": len(inputs), "raw_rows": n_rows, "raw_file": raw_path.name,
    }
    (out_dir / f"worker_{label}.json").write_text(json.dumps(worker_meta, indent=2, default=str))
    log(f"[{label}] wrote {n_rows} rows -> {raw_path.name}")


# --------------------------------------------------------------------------- parent: metadata
def environment_metadata() -> dict:
    import torch

    cpuinfo = read_text("/proc/cpuinfo") or ""
    models = sorted({ln.split(":", 1)[1].strip() for ln in cpuinfo.splitlines()
                     if ln.lower().startswith(("model name", "model\t", "model ")) and ":" in ln})
    status = run_cmd(["git", "status", "--porcelain"]) or ""
    gov = {}
    for cpu in range(os.cpu_count() or 0):
        gov[f"cpu{cpu}"] = read_text(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
    return {
        "datetime_local": _dt.datetime.now().astimezone().isoformat(),
        "datetime_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "hostname": platform.node(),
        "git_head": run_cmd(["git", "rev-parse", "HEAD"]),
        "git_status_summary": {
            "n_entries": len([l for l in status.splitlines() if l.strip()]),
            "n_untracked": len([l for l in status.splitlines() if l.startswith("??")]),
            "n_modified": len([l for l in status.splitlines() if l[:2].strip() and not l.startswith("??")]),
            "entries_head": status.splitlines()[:50],
        },
        "git_submodules": run_cmd(["git", "submodule", "status"]),
        "python_version": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "architecture": platform.machine(),
        "cpu_models": models, "lscpu": run_cmd(["lscpu"]), "os_cpu_count": os.cpu_count(),
        "cpu_max_freq_khz": read_text("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"),
        "scaling_governor": gov,
        "cpu_temp_c": cpu_temp_c(),
        "vcgencmd_temp": run_cmd(["vcgencmd", "measure_temp"]),
        "vcgencmd_throttled": run_cmd(["vcgencmd", "get_throttled"]),
        "env_thread_vars": {k: os.environ.get(k) for k in ENV_THREAD_VARS},
        "torch_version": torch.__version__,
        "torch_config_show": torch.__config__.show(),
        "torch_parallel_info_parent_default": torch.__config__.parallel_info(),
        "torch_default_threads_parent": {"intra": torch.get_num_threads(), "interop": torch.get_num_interop_threads()},
    }


# --------------------------------------------------------------------------- parent: aggregation
def aggregate(out_dir: Path, conditions: List[dict], attack_settings: dict, subset: List[tuple],
              formal_dir: Path) -> Tuple[List[dict], dict, int]:
    rows = []
    for c in conditions:
        with (out_dir / f"raw_{c['label']}.csv").open(newline="") as f:
            rows.extend(csv.DictReader(f))
    merged = out_dir / "raw_measurements.csv"
    with merged.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        w.writerows(rows)

    # formal references
    formal_table = {}
    with (REPO_ROOT / FORMAL_LATENCY_TABLE).open(newline="") as f:
        for r in csv.DictReader(f):
            if r["backend"] == "CPU":
                formal_table[r["profile"]] = float(r["median_ms"])
    _, formal_attack = load_formal_rows(formal_dir, set(subset))
    formal_subset_median = {}
    for attack, pid in TARGET_PROFILES.items():
        vals = [float(r["attack_generation_ms"]) for k, r in formal_attack.items() if k[3] == attack and k[4] == pid]
        formal_subset_median[attack] = float(np.median(vals)) if vals else float("nan")

    groups: Dict[str, str] = {}
    for c in conditions:
        groups.setdefault(c["group"], c["group"])
    measures = ["total_ms", "phase_A_prep_ms", "phase_B_clean_forward_ms", "phase_C_construct_ms",
                "phase_D_attack_ms", "phase_diag_ms", "phase_E_post_ms"]
    summary = []
    t4 = {}
    for g in groups:
        g_rows = [r for r in rows if r["condition"].rsplit("_r", 1)[0] == g]
        for attack in attack_settings:
            for path in ("adapter_apply", "instrumented"):
                pr = [r for r in g_rows if r["attack"] == attack and r["path"] == path]
                for m in (measures if path == "instrumented" else ["total_ms"]):
                    vals = [float(r[m]) for r in pr if r[m] not in ("", None)]
                    st = stats(vals)
                    rec = {"condition": g, "intra_threads": pr[0]["intra_threads_actual"] if pr else "",
                           "interop_threads": pr[0]["interop_threads_actual"] if pr else "",
                           "attack": attack, "attack_profile": attack_settings[attack]["profile"],
                           "path": path, "measure": m, "batch_size": 1}
                    rec.update({f"{k}_ms" if k != "n" else "n": v for k, v in st.items()})
                    rec["samples_per_sec"] = 1000.0 / st["mean"] if st["n"] and st["mean"] > 0 else float("nan")
                    summary.append(rec)
                    if g == "intra4_interop1":
                        t4[(attack, path, m)] = st
    for rec in summary:
        ref = t4.get((rec["attack"], rec["path"], rec["measure"]))
        rec["speedup_median_vs_t4_interop1"] = (ref["median"] / rec["median_ms"]) if ref and rec["median_ms"] else float("nan")
        rec["speedup_mean_vs_t4_interop1"] = (ref["mean"] / rec["mean_ms"]) if ref and rec["mean_ms"] else float("nan")
        comparable = rec["path"] == "adapter_apply" and rec["measure"] == "total_ms"
        prof = f"{rec['attack']}/{rec['attack_profile']}"
        ft = formal_table.get(prof) if comparable else None
        fs = formal_subset_median.get(rec["attack"]) if comparable else None
        rec["formal_table_median_ms"] = ft if ft is not None else ""
        rec["speedup_median_vs_formal_table"] = (ft / rec["median_ms"]) if ft and rec["median_ms"] else ""
        rec["formal_subset_median_ms"] = fs if fs is not None else ""
        rec["speedup_median_vs_formal_subset"] = (fs / rec["median_ms"]) if fs and rec["median_ms"] else ""
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    refs = {"formal_table_cpu_median_ms": {p: formal_table.get(f"{p}/{TARGET_PROFILES[p]}") for p in TARGET_PROFILES},
            "formal_subset_cpu_median_ms": formal_subset_median}
    return rows, refs, len(rows)


def validate(rows: List[dict], attack_settings: dict, subset: List[tuple], formal_dir: Path,
             conditions: List[dict]) -> dict:
    _, formal_attack = load_formal_rows(formal_dir, set(subset))
    out = {"per_condition": {}, "cross_condition": {}, "notes": []}
    tol_abs = 1e-6
    for c in conditions:
        lab = c["label"]
        crow = [r for r in rows if r["condition"] == lab]
        per = {}
        for attack in attack_settings:
            pid = attack_settings[attack]["profile"]
            ar = [r for r in crow if r["attack"] == attack and r["path"] == "adapter_apply"]
            n = len(ar)
            m_pred = m_succ = m_hash = 0
            linf_diffs, l2_diffs, lin_norm = [], [], []
            for r in ar:
                k = (r["modulation"], int(r["snr_db"]), int(r["sample_index"]), attack, pid)
                fr = formal_attack[k]
                m_pred += int(int(r["attacked_pred"]) == int(float(fr["attacked_pred"])))
                m_succ += int(parse_bool(r["attack_success"]) == parse_bool(fr["attack_success"]))
                m_hash += int(r["x_adv_sha256"] == fr.get("adversarial_sha256"))
                linf_diffs.append(abs(float(r["iq_linf"]) - float(fr["iq_linf"])))
                l2_diffs.append(abs(float(r["iq_l2"]) - float(fr["iq_l2"])))
                lin_norm.append(float(r["iq_linf_normalized"]))
            replica_eq = [parse_bool(r["paths_xadv_equal"]) for r in crow if r["attack"] == attack]
            rec = {
                "n_adapter_apply_calls": n,
                "attacked_pred_match_formal": f"{m_pred}/{n}",
                "attack_success_match_formal": f"{m_succ}/{n}",
                "x_adv_sha256_match_formal": f"{m_hash}/{n}",
                "iq_linf_max_abs_diff_vs_formal": max(linf_diffs) if linf_diffs else None,
                "iq_l2_max_abs_diff_vs_formal": max(l2_diffs) if l2_diffs else None,
                "max_linf_normalized_domain": max(lin_norm) if lin_norm else None,
                "linf_normalized_within_eps": all(v <= attack_settings[attack]["eps"] + tol_abs for v in lin_norm),
                "attack_success_rate": (sum(parse_bool(r["attack_success"]) for r in ar) / n) if n else None,
                "formal_subset_attack_success_rate": (
                    sum(parse_bool(formal_attack[(s[0], s[1], s[2], attack, pid)]["attack_success"]) for s in subset) / len(subset)),
                "instrumented_replica_xadv_equal_to_apply": f"{sum(replica_eq)}/{len(replica_eq)}",
            }
            if attack in DETERMINISTIC_ATTACKS:
                rec["formal_equivalence"] = (
                    "PASS" if m_pred == n and m_succ == n and max(linf_diffs + [0]) <= tol_abs and max(l2_diffs + [0]) <= 1e-5
                    else "MISMATCH (reported, not aborted; thread-count float reduction order can differ from formal 4/4 run)")
                rec["x_adv_hash_note"] = ("hash equality additionally requires bit-identical float results; "
                                          "formal CPU run used intra=4/interop=4")
            else:
                rec["formal_equivalence"] = "INFORMATIONAL ONLY (pgd random_start=True; formal RNG call ordering not reproduced)"
                rec["semantic_check"] = ("PASS" if rec["linf_normalized_within_eps"] else "FAIL") + \
                    " (finite, shape-preserving, L_inf <= eps in the [0,1] attack domain)"
            per[attack] = rec
        out["per_condition"][lab] = per
    # across thread conditions: same inputs, same torch seed per call -> compare x_adv hashes
    for attack in attack_settings:
        by_cond: Dict[str, Dict[tuple, str]] = {}
        for r in rows:
            if r["attack"] == attack and r["path"] == "adapter_apply":
                by_cond.setdefault(r["condition"], {})[(r["modulation"], r["snr_db"], r["sample_index"], r["repeat"])] = r["x_adv_sha256"]
        labels = sorted(by_cond)
        pair = {}
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = by_cond[labels[i]], by_cond[labels[j]]
                common = set(a) & set(b)
                pair[f"{labels[i]}__vs__{labels[j]}"] = f"{sum(a[k] == b[k] for k in common)}/{len(common)}"
        out["cross_condition"][attack] = {"x_adv_sha256_identical": pair}
    all_rep = all(parse_bool(r["paths_xadv_equal"]) for r in rows)
    out["instrumented_breakdown_valid"] = all_rep
    if not all_rep:
        out["notes"].append("instrumented replica diverged from untouched apply() on some calls; "
                            "treat the phase breakdown as INVALID for those conditions")
    out["notes"].append("Fail-closed checks enforced during the run (any failure aborts): real AWN backend, real attack "
                        "backend, no runtime fallback, eval mode, requires_grad unchanged, torch thread settings "
                        "unchanged, finite x_adv, shape/dtype unchanged, x_clean sha256 == formal clean_input_sha256.")
    return out


# --------------------------------------------------------------------------- parent main
def parent_main(args) -> None:
    import torch  # noqa: F401 -- metadata only; parent never runs attacks

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (REPO_ROOT / (args.out_dir or f"results/accel_phase1_{ts}")).resolve()
    for p in PROTECTED_RESULT_DIRS:
        prot = (REPO_ROOT / p).resolve()
        if out_dir == prot or prot in out_dir.parents or out_dir in prot.parents:
            abort(f"refusing to write inside/over formal results directory {p}")
    out_dir.mkdir(parents=True, exist_ok=False)

    ctx = load_formal_context(args)
    cfg, manifest, formal_dir = ctx["cfg"], ctx["manifest"], ctx["formal_dir"]
    subset = build_subset(cfg)
    base_rows, attack_rows = load_formal_rows(formal_dir, set(subset))
    attack_settings = resolve_attack_settings(cfg, attack_rows, args.attack_temperature, args.diagnostics)
    sensing = resolve_sensing(cfg)

    dataset_path = REPO_ROOT / cfg["dataset_path"]
    checkpoint_path = REPO_ROOT / cfg["checkpoint_path"]
    log("[parent] hashing dataset / checkpoint / config ...")
    hashes = {
        "dataset_sha256": sha256_file(dataset_path), "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": sha256_file(ctx["cfg_path"]),
        "formal_manifest_sha256": sha256_file(formal_dir / "manifest.json"),
        "formal_base_results_sha256": sha256_file(formal_dir / "base_results.csv"),
        "formal_attack_results_sha256": sha256_file(formal_dir / "attack_results.csv"),
        "formal_latency_table_sha256": sha256_file(REPO_ROOT / FORMAL_LATENCY_TABLE),
        "attack_adapter_py_sha256": sha256_file(REPO_ROOT / "src/adapters/attack_adapter.py"),
        "this_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    if hashes["dataset_sha256"] != manifest.get("dataset_sha256"):
        abort("dataset sha256 differs from formal CPU manifest")
    if hashes["checkpoint_sha256"] != manifest.get("checkpoint_sha256"):
        abort("checkpoint sha256 differs from formal CPU manifest")
    if args.formal_runner:
        hashes["formal_runner_sha256"] = sha256_file(REPO_ROOT / args.formal_runner)

    env_meta = environment_metadata()
    if any(env_meta["env_thread_vars"].values()):
        log(f"[warn] thread env vars are set in this shell: {env_meta['env_thread_vars']} "
            "(formal CPU shell had none set). They are inherited unchanged and recorded.")

    threads = [int(t) for t in args.threads.split(",")]
    if any(t not in (1, 2, 4) for t in threads) or len(set(threads)) != len(threads):
        abort("--threads must be a permutation/subset of 1,2,4")
    conditions = []
    for rnd in range(args.rounds):
        order = threads if rnd % 2 == 0 else list(reversed(threads))
        for t in order:
            conditions.append({"group": f"intra{t}_interop1", "label": f"intra{t}_interop1_r{rnd}", "threads": t,
                               "formal_default": False, "round": rnd})
        if args.formal_default_control:
            conditions.append({"group": "formal_default_threads", "label": f"formal_default_threads_r{rnd}",
                               "threads": 0, "formal_default": True, "round": rnd})

    worker_ctx = {
        "subset": [list(k) for k in subset], "formal_dir": rel(formal_dir),
        "dataset_path": cfg["dataset_path"], "checkpoint_path": cfg["checkpoint_path"],
        "sensing": sensing, "attack_settings": attack_settings, "attacks": list(TARGET_PROFILES),
    }
    ctx_file = out_dir / "worker_context.json"
    ctx_file.write_text(json.dumps(worker_ctx, indent=2, default=str))

    manifest_out = {
        "phase": "accel_phase1a_thread_baseline", "status": "running",
        "question": "B=1 single-sample attack-generation latency vs PyTorch intra-op threads {1,2,4}, interop=1",
        "command": [sys.executable] + sys.argv, "out_dir": rel(out_dir),
        "formal_cpu_dir": rel(formal_dir), "config_path": args.config,
        "dataset_path": cfg["dataset_path"], "checkpoint_path": cfg["checkpoint_path"],
        "hashes": hashes, "environment": env_meta,
        "subset": {"modulations": cfg["modulations"], "snrs": SUBSET_SNRS, "sample_indices": SUBSET_SAMPLE_INDICES,
                   "n_source_samples": len(subset),
                   "samples": [{"modulation": k[0], "snr_db": k[1], "sample_index": k[2],
                                "formal_seed": int(float(base_rows[k]["seed"])),
                                "clean_input_sha256": base_rows[k]["clean_input_sha256"]} for k in subset]},
        "attacks": attack_settings, "sensing_parameters": sensing, "batch_size": 1,
        "repeats": args.repeats, "warmup_samples_per_attack": args.warmup, "rounds": args.rounds,
        "conditions": conditions,
        "timing": {
            "clock": "time.perf_counter",
            "adapter_apply": "t0=perf_counter(); AttackAdapter.apply(...); elapsed -- same boundary as formal attack_ms",
            "instrumented": "separate replica call; phases A..E + diag; total_ms is the replica's own wall time",
            "excluded_from_both": "torch.manual_seed(call_seed), AWN attacked inference, hashing, CSV writing",
        },
        "rng": "torch.manual_seed(formal per-row seed) before EVERY attack call (both paths); "
               "PGD random_start draws therefore repeat across paths/threads but are NOT claimed to match the formal run",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2, default=str))

    for c in conditions:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--out-dir", str(out_dir),
               "--worker-context", str(ctx_file), "--worker-label", c["label"], "--worker-round", str(c["round"]),
               "--worker-threads", str(c["threads"]), "--repeats", str(args.repeats), "--warmup", str(args.warmup)]
        if c["formal_default"]:
            cmd.append("--worker-formal-default")
        c["temp_before_c"] = cpu_temp_c()
        log(f"[parent] starting fresh process for {c['label']} (temp={c['temp_before_c']})")
        with (out_dir / f"worker_{c['label']}.stdout.log").open("w") as lf:
            rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=lf, env=os.environ.copy()).returncode
        c["temp_after_c"] = cpu_temp_c()
        c["returncode"] = rc
        if rc != 0:
            manifest_out.update(status="aborted", failed_condition=c["label"])
            (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2, default=str))
            abort(f"worker {c['label']} exited with {rc}; see worker_{c['label']}.stdout.log and stderr above")
        if args.cooldown > 0:
            time.sleep(args.cooldown)

    rows, refs, n = aggregate(out_dir, conditions, attack_settings, subset, formal_dir)
    val = validate(rows, attack_settings, subset, formal_dir, conditions)
    val["workers"] = {c["label"]: json.loads((out_dir / f"worker_{c['label']}.json").read_text()) for c in conditions}
    (out_dir / "validation.json").write_text(json.dumps(val, indent=2, default=str))
    manifest_out.update(status="complete", raw_rows=n, formal_references=refs, finished_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
                        environment_end={"cpu_temp_c": cpu_temp_c(), "vcgencmd_throttled": run_cmd(["vcgencmd", "get_throttled"])})
    (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2, default=str))
    log(f"[parent] done: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--formal-dir", default=DEFAULT_FORMAL_CPU_DIR)
    ap.add_argument("--out-dir", default=None, help="default results/accel_phase1_<timestamp> (must not exist)")
    ap.add_argument("--threads", default="4,2,1", help="intra-op thread conditions and run order (subset of 1,2,4)")
    ap.add_argument("--repeats", type=int, default=3, help="passes over the 88-sample subset per condition")
    ap.add_argument("--warmup", type=int, default=5, help="unrecorded warmup samples per attack per process")
    ap.add_argument("--rounds", type=int, default=1, help="repeat all conditions; order reverses each round")
    ap.add_argument("--cooldown", type=float, default=0.0, help="seconds to idle between processes")
    ap.add_argument("--attack-temperature", type=float, default=None,
                    help="only if the formal config/rows do not record it; must equal the formal value")
    ap.add_argument("--diagnostics", choices=["auto", "on", "off"], default="auto",
                    help="apply(diagnostics=...); auto = match formal rows (non-empty gradient columns)")
    ap.add_argument("--formal-default-control", action="store_true",
                    help="also run one process with NO torch thread overrides (formal CPU shell = 4/4)")
    ap.add_argument("--formal-runner", default=None, help="optional path of the formal CPU runner, sha256 recorded")
    # internal
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-context", help=argparse.SUPPRESS)
    ap.add_argument("--worker-label", help=argparse.SUPPRESS)
    ap.add_argument("--worker-round", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--worker-threads", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--worker-formal-default", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.rounds < 1:
        abort("--repeats>=1, --warmup>=0, --rounds>=1 required")
    if args.worker:
        worker_main(args)
    else:
        parent_main(args)


if __name__ == "__main__":
    main()
