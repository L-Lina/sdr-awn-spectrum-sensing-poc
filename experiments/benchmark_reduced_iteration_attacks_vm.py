"""
Reduced-iteration attack benchmark (x86 desktop / VM, CPU only, B=1, AMC-only).
EXPERIMENT ONLY -- src/adapters/attack_adapter.py, external/adversarial-rf and the formal pipeline
are not modified.

Question
--------
How far can the iteration / step count of PGD, CW and EAD be reduced before AMC attack efficacy
(formal conditional ASR) drops below a usable threshold, and how much attack-generation latency
does that save on this platform? FGSM has no iterations; it is run at its four formal eps values
as the single-step latency baseline.

Samples (formal convention, rebuilt -- no formal CSV needed)
-----------------------------------------------------------
88 = 11 modulations x SNR {-20, -6, 6, 18} x sample_index {0, 1} (the subset of
experiments/bench_attack_accel_pi5.py). Each clean input is rebuilt by
experiments/run_cpu_full_matrix.py:build_sensing_input (derive_seed(base_seed, mod, snr, idx, "")
-> embed -> energy detection -> alignment -> AWN preprocess), B=1.

Metric (formal definition, experiments/run_cpu_full_matrix.py / run_hailo_full_matrix.py)
-----------------------------------------------------------------------------------------
clean_pred = argmax AWN(x_clean), attacked_pred = argmax AWN(x_adv), label = true RML2016 class.
clean_correct = clean_pred == label; attacked_correct = attacked_pred == label;
attack_success = clean_correct AND NOT attacked_correct;
conditional ASR = sum(attack_success) / sum(clean_correct).
pred_changed (attacked_pred != clean_pred) is recorded for information only.

Attack call
-----------
production AttackAdapter(awn.model).apply() with its default options (b1_pgd_bim_mkldnn_off=False,
b1_mkldnn_off_extended=False: PyTorch default backend path, no backend toggle) and the formal arguments
of run_cpu_full_matrix.py (temperature 1.0, diagnostics off,
seed = derive_seed(..., f"{attack}|{profile_id}")), preceded by an untimed
torch.manual_seed(formal sample seed) on every call. Any attack_status != "ok" or a backend other
than the real torchattacks path aborts the run. A configuration that torchattacks itself rejects is
detected by an untimed probe outside the adapter and recorded as INVALID_CONFIGURATION.

Latency
-------
Primary latency = wall time of the AttackAdapter.apply() call (time.perf_counter_ns; includes the
adapter's min-max mapping, internal clean forward, attack construction, atk(x, y) and state
restoration; excludes seeding, AWN attacked inference and metric computation). Adapter stdout is
redirected to a null sink for every call (timed or not). Per profile: --warmup untimed calls, then
the 88 samples; --replications passes, with the profile order of each attack group reversed on odd
replications (fixed alternating order). No thermal gate / governor / cooldown (not a Pi benchmark).

--save-x-adv writes replication-0 x_adv arrays to x_adv_rep0.npz after the run (not timed).
--legacy-backend-toggle is for comparison only: it reproduces the superseded configuration that passed both
mkldnn-off opt-ins, writes to a *_legacy_toggle_repro_* directory and is never a formal run.

Run from the repo root:
    ~/adversarial-rf/.venv/bin/python experiments/benchmark_reduced_iteration_attacks_vm.py \
        --dataset ~/adversarial-rf/data/RML2016.10a_dict.pkl --checkpoint ~/adversarial-rf/2016.10a_AWN.pkl --smoke
    (same without --smoke for the full run)
"""

from __future__ import annotations

import argparse
import contextlib
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
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ADV_RF_ROOT = REPO_ROOT / "external" / "adversarial-rf"
for _p in (REPO_ROOT, ADV_RF_ROOT, REPO_ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_cpu_full_matrix as formal  # noqa: E402 -- derive_seed, build_sensing_input, sha256_array

FORMAL_CONFIG = "configs/hailo_full_matrix.json"
FORMAL_ATTACK_SUMMARY = "reports/meeting_20260921/tables/attack_summary.csv"
EXPECTED_DATASET_SHA256 = "b29ccc25b00d0718cd3b70ffa9158662ec83f6d9b63ffd845c7bcbe3b3096e8c"
EXPECTED_CHECKPOINT_SHA256 = "8af0458f2570c465b5bb0ebad00817944f8171d888cd7cff1324ecb258820695"
EXPECTED_ADV_RF_COMMIT = "ced705ed861bde352841b0b8bd038a6449e0fa4d"

SUBSET_SNRS = [-20, -6, 6, 18]
SUBSET_SAMPLE_INDICES = [0, 1]
SMOKE_SAMPLE_POSITIONS = (0, 6, 30, 43, 78, 87)  # includes SNR 18 samples (clean-correct likely)

PGD_EPS = 0.03
PGD_ALPHA = PGD_EPS / 4
PGD_STEPS = [1, 2, 3, 5, 7, 10]
CW_STEPS = [1, 2, 3, 5, 10, 15, 20]
EAD_MAX_ITERS = [10, 20, 30, 50, 75, 100]
EAD_BSS = [1, 3, 5, 9]
FGSM_EPS = [0.005, 0.01, 0.03, 0.05]
EAD_FIXED = {"lr": 0.01, "initial_const": 1e-3, "beta": 1e-3, "kappa": 0, "abort_early": True}

# usable threshold = 90% of the formal 2200-sample CPU baseline conditional ASR (given, not re-derived)
USABLE_THRESHOLD_PCT = {"pgd": 72.36, "cw": 82.17, "ead": 90.0}
FORMAL_BASELINE_PROFILE = {"pgd": "pgd/eps0.03", "cw": "cw/c1_steps20_lr0.01", "ead": "ead/package_default"}

RAW_FIELDS = ["group", "attack", "profile", "rep", "order_index", "sample_pos", "modulation", "snr_db",
              "sample_index", "formal_seed", "attack_seed", "true_label", "clean_pred", "clean_correct",
              "attacked_pred", "attacked_correct", "attack_success", "pred_changed", "latency_ns", "latency_ms",
              "linf", "l2", "finite", "attack_status", "attack_backend", "mkldnn_off_applied",
              "mkldnn_state_after", "x_adv_sha256", "same_output_as_rep0", "same_success_as_rep0"]


class _NullSink:
    def write(self, _s):
        return 0

    def flush(self):
        pass


_NULL = _NullSink()
_LOG_FH = None


def log(msg: str) -> None:
    line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, file=sys.stderr, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


class Abort(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: List[str]) -> Optional[str]:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text().strip()
    except Exception:  # noqa: BLE001
        return None


def pct(vals: List[float], q: float) -> float:
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q)) if vals else float("nan")


def dist(vals: List[float]) -> dict:
    if not vals:
        return {k: float("nan") for k in ("n", "median", "mean", "p95", "p05", "std", "min", "max")}
    a = np.asarray(vals, dtype=np.float64)
    return {"n": int(a.size), "median": float(np.median(a)), "mean": float(a.mean()), "p95": pct(vals, 95),
            "p05": pct(vals, 5), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max())}


def write_csv(path: Path, rows: List[dict], fields: Optional[List[str]] = None) -> None:
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["empty"], restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# =========================================================================== profiles
LEGACY_TOGGLE_KWARGS = {"b1_pgd_bim_mkldnn_off": True, "b1_mkldnn_off_extended": True}


def backend_toggle_kwargs(legacy: bool) -> dict:
    """AttackAdapter keyword options: defaults (no backend toggle) unless reproducing the superseded run."""
    return dict(LEGACY_TOGGLE_KWARGS) if legacy else {}


def backend_toggle_manifest(legacy: bool, adapter=None) -> dict:
    flags = {"b1_pgd_bim_mkldnn_off": bool(getattr(adapter, "b1_pgd_bim_mkldnn_off", legacy)),
             "b1_mkldnn_off_extended": bool(getattr(adapter, "b1_mkldnn_off_extended", legacy))}
    if legacy:
        return {"mkldnn_backend_toggle": "enabled (comparison-only reproduction of the superseded configuration)",
                "run_role": "legacy_toggle_reproduction; not a formal run", **flags}
    return {"mkldnn_backend_toggle": "disabled / default path (PyTorch default backend; AttackAdapter defaults)",
            "run_role": "formal", **flags}


def backend_toggle_ok(legacy: bool, adapter) -> bool:
    want = bool(legacy)
    return (bool(adapter.b1_pgd_bim_mkldnn_off) == want and bool(adapter.b1_mkldnn_off_extended) == want)


def save_x_adv(path: Path, arrays: dict) -> None:
    """Write {key: x_adv} as one compressed npz (keys sorted); written after all timed calls."""
    keys = sorted(arrays)
    np.savez_compressed(path, keys=np.array(keys), **{f"a{i}": arrays[k] for i, k in enumerate(keys)})


def run_dir_name(base: str, args) -> str:
    tag = "smoke_" if getattr(args, "smoke", False) else ""
    tag += "legacy_toggle_repro_" if getattr(args, "legacy_backend_toggle", False) else ""
    return f"{base}_{tag}{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"


def make_profile(group: str, attack: str, profile_id: str, *, eps: float, attack_params: dict,
                 cw_c: float = 1.0, cw_steps: int = 20, cw_lr: float = 0.01, alpha=None, steps=None,
                 bss=None, max_iter=None, is_baseline: bool = False) -> dict:
    return {"group": group, "attack": attack, "profile": profile_id, "eps": eps, "attack_params": attack_params,
            "cw_c": cw_c, "cw_steps": cw_steps, "cw_lr": cw_lr, "alpha": alpha, "steps": steps,
            "binary_search_steps": bss, "max_iterations": max_iter, "is_baseline": is_baseline}


def fgsm_profiles() -> List[dict]:
    return [make_profile("fgsm", "fgsm", f"eps{e}", eps=e, attack_params={}, steps=1) for e in FGSM_EPS]


def pgd_profiles() -> List[dict]:
    return [make_profile("pgd", "pgd", f"eps0.03_steps{s}", eps=PGD_EPS,
                         attack_params={"alpha": PGD_ALPHA, "steps": s, "random_start": True},
                         alpha=PGD_ALPHA, steps=s, is_baseline=(s == 10)) for s in PGD_STEPS]


def cw_profiles() -> List[dict]:
    return [make_profile("cw", "cw", f"c1_steps{s}_lr0.01", eps=0.03, attack_params={"kappa": 0.0},
                         cw_c=1.0, cw_steps=s, cw_lr=0.01, steps=s, is_baseline=(s == 20)) for s in CW_STEPS]


def ead_profile(group: str, bss: int, mi: int, is_baseline: bool) -> dict:
    return make_profile(group, "ead", f"bss{bss}_maxit{mi}", eps=0.03,
                        attack_params={"binary_search_steps": bss, "max_iterations": mi, **EAD_FIXED},
                        bss=bss, max_iter=mi, is_baseline=is_baseline)


def ead_phase1_profiles() -> List[dict]:
    return [ead_profile("ead_phase1", 9, mi, mi == 100) for mi in EAD_MAX_ITERS]


def ead_phase2_profiles(mi: int) -> List[dict]:
    return [ead_profile("ead_phase2", b, mi, b == 9) for b in EAD_BSS]


def smoke_select(profiles: List[dict]) -> List[dict]:
    """baseline + lowest-iteration profile (FGSM: smallest and largest eps)."""
    if profiles[0]["attack"] == "fgsm":
        return [profiles[0], profiles[-1]]
    base = [p for p in profiles if p["is_baseline"]]
    return [profiles[0]] + base


# =========================================================================== environment
def environment_metadata(torch, ta) -> dict:
    lscpu = run_cmd(["lscpu"]) or ""
    lsd = {}
    for line in lscpu.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            lsd[k.strip()] = v.strip()
    flags = lsd.get("Flags", "")
    return {
        "hostname": platform.node(), "platform": platform.platform(), "machine": platform.machine(),
        "cpu_model": lsd.get("Model name"), "cpu_count_os": os.cpu_count(),
        "lscpu_cpus": lsd.get("CPU(s)"), "threads_per_core": lsd.get("Thread(s) per core"),
        "hypervisor_vendor": lsd.get("Hypervisor vendor"), "virtualization_type": lsd.get("Virtualization type"),
        "cpuinfo_hypervisor_flag": "hypervisor" in flags.split(),
        "systemd_detect_virt": run_cmd(["systemd-detect-virt"]),
        "dmi_product_name": read_text("/sys/class/dmi/id/product_name"),
        "dmi_sys_vendor": read_text("/sys/class/dmi/id/sys_vendor"),
        "scaling_governor_cpu0_readonly": read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "lscpu_isa_flags_subset": [f for f in ("avx2", "avx512f", "avx512_vnni", "fma") if f in flags.split()],
        "python": sys.version, "python_executable": sys.executable,
        "torch": torch.__version__, "torchattacks": getattr(ta, "__version__", None), "numpy": np.__version__,
        "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
        "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "mkldnn_enabled_default": bool(torch._C._get_mkldnn_enabled()),
        "torch_parallel_info": torch.__config__.parallel_info(),
        "env_OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "env_MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "loadavg_at_start": read_text("/proc/loadavg"),
    }


def git_metadata() -> dict:
    return {
        "head": run_cmd(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        "branch": run_cmd(["git", "-C", str(REPO_ROOT), "branch", "--show-current"]),
        "status_short": run_cmd(["git", "-C", str(REPO_ROOT), "status", "--short"]),
        "submodule_status": run_cmd(["git", "-C", str(REPO_ROOT), "submodule", "status"]),
        "adversarial_rf_head": run_cmd(["git", "-C", str(ADV_RF_ROOT), "rev-parse", "HEAD"]),
        "adversarial_rf_status_short": run_cmd(["git", "-C", str(ADV_RF_ROOT), "status", "--short"]),
    }


# =========================================================================== main experiment
def run(args, out_dir: Path, state: dict) -> None:
    import torch
    import torchattacks as ta

    from src.adapters import attack_adapter as am
    from src.adapters.attack_adapter import AttackAdapter
    from src.adapters.awn_adapter import AWNModelAdapter
    from src.sensing.radioml_source import RML2016_10A_CLASSES, load_radioml_dict

    if args.interop_threads:
        torch.set_num_interop_threads(args.interop_threads)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    state["environment"] = environment_metadata(torch, ta)
    state["git"] = git_metadata()
    log(f"env: {state['environment']['cpu_model']} | torch {torch.__version__} threads {torch.get_num_threads()} "
        f"| torchattacks {ta.__version__} | mkldnn avail={state['environment']['mkldnn_available']} "
        f"enabled={state['environment']['mkldnn_enabled_default']}")

    checks = state["checks"]
    checks["adversarial_rf_at_pin"] = state["git"]["adversarial_rf_head"] == EXPECTED_ADV_RF_COMMIT
    if not checks["adversarial_rf_at_pin"]:
        raise Abort(f"external/adversarial-rf HEAD {state['git']['adversarial_rf_head']} != {EXPECTED_ADV_RF_COMMIT}")
    checks["torchattacks_3_5_1"] = getattr(ta, "__version__", None) == "3.5.1"
    if not checks["torchattacks_3_5_1"]:
        raise Abort(f"torchattacks {getattr(ta, '__version__', None)} != 3.5.1")

    dataset, ckpt = Path(args.dataset).expanduser().resolve(), Path(args.checkpoint).expanduser().resolve()
    log("hashing dataset / checkpoint ...")
    state["inputs"] = {"dataset_path": str(dataset), "dataset_sha256": sha256_file(dataset),
                       "checkpoint_path": str(ckpt), "checkpoint_sha256": sha256_file(ckpt),
                       "expected_dataset_sha256": EXPECTED_DATASET_SHA256,
                       "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256}
    checks["dataset_hash_ok"] = state["inputs"]["dataset_sha256"] == EXPECTED_DATASET_SHA256
    checks["checkpoint_hash_ok"] = state["inputs"]["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    if not (checks["dataset_hash_ok"] and checks["checkpoint_hash_ok"]):
        raise Abort(f"dataset/checkpoint hash mismatch: {state['inputs']}")

    with contextlib.redirect_stdout(_NULL):
        awn = AWNModelAdapter(str(ckpt), device="cpu")
        adapter = AttackAdapter(awn_model=awn.model, device="cpu", **backend_toggle_kwargs(args.legacy_backend_toggle))
    checks["backend_toggle_matches_configuration"] = backend_toggle_ok(args.legacy_backend_toggle, adapter)
    state["backend_toggle"] = backend_toggle_manifest(args.legacy_backend_toggle, adapter)
    checks["real_awn_loaded"] = awn.status == "ok" and awn.model is not None
    checks["real_attack_backend_loaded"] = (adapter.status == "ok" and adapter.backend_name == am._REAL_ATTACK_SOURCE
                                            and am._import_error is None)
    if not (checks["real_awn_loaded"] and checks["real_attack_backend_loaded"]):
        raise Abort(f"real backend not loaded: awn={awn.status}/{awn.notes} attack={adapter.status}/{adapter.notes}")
    log(f"real AWN + attack backend loaded ({adapter.backend_name})")

    cfg = json.loads((REPO_ROOT / FORMAL_CONFIG).read_text())
    base_seed = int(cfg["sensing"]["base_seed"])
    subset = [(m, s, i) for m in cfg["modulations"] for s in SUBSET_SNRS for i in SUBSET_SAMPLE_INDICES]
    if len(subset) != 88:
        raise Abort(f"subset size {len(subset)} != 88")
    positions = list(SMOKE_SAMPLE_POSITIONS) if args.smoke else list(range(len(subset)))

    log("loading dataset and rebuilding formal clean inputs ...")
    data = load_radioml_dict(str(dataset))
    samples = []
    for pos in positions:
        mod, snr, idx = subset[pos]
        sample = data[(mod, snr)][idx].astype(np.float32)
        with contextlib.redirect_stdout(_NULL):
            x_clean, sense_meta, _ = formal.build_sensing_input(sample, cfg, mod, snr, idx)
            logits, meta = awn.infer(x_clean)
        if meta["awn_status"] != "ok":
            raise Abort(f"clean AWN fallback {meta}")
        label = RML2016_10A_CLASSES[mod]
        clean_pred = int(np.argmax(logits[0]))
        samples.append({"pos": pos, "mod": mod, "snr": snr, "idx": idx, "x": x_clean, "label": label,
                        "seed": int(sense_meta["seed"]), "clean_pred": clean_pred,
                        "clean_correct": clean_pred == label, "clean_sha256": sense_meta["clean_input_sha256"]})
    del data
    n_cc = sum(s["clean_correct"] for s in samples)
    log(f"{len(samples)} clean inputs rebuilt; clean_correct {n_cc}/{len(samples)}")
    state["samples"] = [{k: v for k, v in s.items() if k != "x"} for s in samples]

    formal_asr = {}
    for r in csv.DictReader((REPO_ROOT / FORMAL_ATTACK_SUMMARY).open()):
        if r["backend"] == "CPU":
            formal_asr[r["profile"]] = float(r["cond_asr_pct"])
    state["formal_baseline_asr_pct"] = {a: formal_asr[p] for a, p in FORMAL_BASELINE_PROFILE.items()}
    state["formal_baseline_asr_pct"].update({f"fgsm/eps{e}": formal_asr[f"fgsm/eps{e}"] for e in FGSM_EPS})

    raw_path = out_dir / "raw_samples.csv"
    raw_f = raw_path.open("w", newline="")
    raw_w = csv.DictWriter(raw_f, fieldnames=RAW_FIELDS, extrasaction="ignore")
    raw_w.writeheader()
    results: Dict[str, dict] = {}  # profile key -> accumulators
    state["profile_results"] = results

    def probe_valid(prof: dict, s: dict) -> Optional[str]:
        """Untimed probe outside the adapter: does torchattacks accept and run this configuration?"""
        wm = adapter.wrapped_model
        try:
            x_t = torch.from_numpy(s["x"])
            x_ta, a, b = am._iq_to_ta_input_minmax(x_t)
            wm.set_minmax(a, b)
            with torch.no_grad():
                y = wm(x_ta).argmax(dim=1)
            atk = am._build_torchattacks(prof["attack"], am.TemperatureLogitsWrapper(wm, 1.0), prof["eps"],
                                         cw_c=prof["cw_c"], cw_steps=prof["cw_steps"], cw_lr=prof["cw_lr"],
                                         attack_params=dict(prof["attack_params"]))
            torch.manual_seed(s["seed"])
            with contextlib.redirect_stdout(_NULL):
                atk(x_ta, y)
            return None
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"
        finally:
            wm.eval()
            wm.clear_minmax()

    def call(prof: dict, s: dict):
        attack_seed = formal.derive_seed(base_seed, s["mod"], s["snr"], s["idx"], f"{prof['attack']}|{prof['profile']}")
        torch.manual_seed(s["seed"])  # untimed, formal per-sample seed, identical for every profile
        with contextlib.redirect_stdout(_NULL):
            t0 = time.perf_counter_ns()
            x_adv, meta = adapter.apply(s["x"], attack=prof["attack"], eps=prof["eps"], temperature=1.0,
                                        seed=attack_seed, diagnostics=False, cw_c=prof["cw_c"],
                                        cw_steps=prof["cw_steps"], cw_lr=prof["cw_lr"],
                                        attack_params=dict(prof["attack_params"]))
            dt = time.perf_counter_ns() - t0
        mk_after = bool(torch._C._get_mkldnn_enabled())
        if meta.get("attack_status") != "ok" or meta.get("attack_backend") != am._REAL_ATTACK_SOURCE:
            state["fallback_count"] += 1
            raise Abort(f"attack fallback/error for {prof['attack']}/{prof['profile']} sample {s['pos']}: "
                        f"status={meta.get('attack_status')} backend={meta.get('attack_backend')} "
                        f"notes={meta.get('attack_notes')}")
        if mk_after != state["environment"]["mkldnn_enabled_default"]:
            raise Abort("global mkldnn state not restored after apply()")
        return x_adv, meta, dt, attack_seed, mk_after

    def expected_mkldnn_off(attack: str) -> bool:
        return args.legacy_backend_toggle and attack in ("pgd", "cw", "ead")

    def run_group(group: str, profiles: List[dict]) -> None:
        if args.smoke:
            profiles = smoke_select(profiles)
        valid = []
        for prof in profiles:
            key = pkey(prof)
            res = results.setdefault(key, {"prof": prof, "status": "PENDING", "rows": [], "lat_ms": [],
                                           "invalid_reason": None, "rep0_sha": {}, "rep0_succ": {},
                                           "output_mismatch": 0, "success_mismatch": 0, "nonfinite": 0,
                                           "mkldnn_flag_bad": 0})
            reason = probe_valid(prof, samples[0])
            if reason is not None:
                res["status"], res["invalid_reason"] = "INVALID_CONFIGURATION", reason
                log(f"[{group}] {key}: INVALID_CONFIGURATION ({reason}) -- skipped")
            else:
                valid.append(prof)
        reps = 1 if args.smoke else args.replications
        for rep in range(reps):
            order = valid if rep % 2 == 0 else list(reversed(valid))
            for oi, prof in enumerate(order):
                key = pkey(prof)
                res = results[key]
                t_prof = time.monotonic()
                for w in range(args.warmup):
                    call(prof, samples[w % len(samples)])
                for s in samples:
                    x_adv, meta, dt, attack_seed, mk_after = call(prof, s)
                    with contextlib.redirect_stdout(_NULL):
                        logits, ameta = awn.infer(x_adv)
                    if ameta["awn_status"] != "ok":
                        raise Abort(f"attacked AWN fallback {ameta}")
                    finite = bool(np.isfinite(x_adv).all())
                    if not finite:
                        res["nonfinite"] += 1
                    if x_adv.shape != s["x"].shape or x_adv.dtype != s["x"].dtype:
                        raise Abort(f"{key}: bad x_adv shape/dtype {x_adv.shape} {x_adv.dtype}")
                    applied_key = ("attack_b1_pgd_bim_mkldnn_off_applied" if prof["attack"] == "pgd"
                                   else "attack_b1_mkldnn_off_extended_applied")
                    applied = bool(meta.get(applied_key, False)) or bool(
                        meta.get("attack_b1_pgd_bim_mkldnn_off_applied", False))
                    if applied != expected_mkldnn_off(prof["attack"]):
                        res["mkldnn_flag_bad"] += 1
                    att_pred = int(np.argmax(logits[0]))
                    att_correct = att_pred == s["label"]
                    success = bool(s["clean_correct"] and not att_correct)
                    delta = (x_adv - s["x"]).astype(np.float64)
                    sha = formal.sha256_array(x_adv)
                    if rep == 0:
                        res["rep0_sha"][s["pos"]], res["rep0_succ"][s["pos"]] = sha, success
                        if args.save_x_adv:
                            x_store[f"{group}|{prof['attack']}|{prof['profile']}|{s['pos']}"] = np.array(x_adv)
                        same_out, same_succ = True, True
                    else:
                        same_out = sha == res["rep0_sha"][s["pos"]]
                        same_succ = success == res["rep0_succ"][s["pos"]]
                        res["output_mismatch"] += int(not same_out)
                        res["success_mismatch"] += int(not same_succ)
                    row = {"group": group, "attack": prof["attack"], "profile": prof["profile"], "rep": rep,
                           "order_index": oi, "sample_pos": s["pos"], "modulation": s["mod"], "snr_db": s["snr"],
                           "sample_index": s["idx"], "formal_seed": s["seed"], "attack_seed": attack_seed,
                           "true_label": s["label"], "clean_pred": s["clean_pred"],
                           "clean_correct": s["clean_correct"], "attacked_pred": att_pred,
                           "attacked_correct": att_correct, "attack_success": success,
                           "pred_changed": att_pred != s["clean_pred"], "latency_ns": dt, "latency_ms": dt / 1e6,
                           "linf": float(np.max(np.abs(delta))), "l2": float(np.linalg.norm(delta.reshape(-1))),
                           "finite": finite, "attack_status": meta["attack_status"],
                           "attack_backend": meta["attack_backend"], "mkldnn_off_applied": applied,
                           "mkldnn_state_after": mk_after, "x_adv_sha256": sha, "same_output_as_rep0": same_out,
                           "same_success_as_rep0": same_succ}
                    raw_w.writerow(row)
                    res["rows"].append(row)
                    res["lat_ms"].append(dt / 1e6)
                raw_f.flush()
                rr = [r for r in res["rows"] if r["rep"] == rep]
                log(f"[{group}] rep {rep} {key}: {len(rr)} calls, median "
                    f"{np.median([r['latency_ms'] for r in rr]):.2f} ms, success "
                    f"{sum(r['attack_success'] for r in rr)}/{sum(r['clean_correct'] for r in rr)} "
                    f"({time.monotonic() - t_prof:.0f} s)")
        for prof in valid:
            results[pkey(prof)]["status"] = "COMPLETE"

    x_store = {}
    run_group("fgsm", fgsm_profiles())
    run_group("pgd", pgd_profiles())
    run_group("cw", cw_profiles())
    run_group("ead_phase1", ead_phase1_profiles())

    # EAD Phase 2: smallest max_iterations whose Phase 1 conditional ASR >= 90 %
    p1 = []
    for mi in EAD_MAX_ITERS:
        r = results.get(f"ead_phase1:ead/bss9_maxit{mi}")
        if r and r["status"] == "COMPLETE":
            asr = conditional_asr(r["rows"])
            p1.append((mi, asr))
    usable = [mi for mi, asr in p1 if not math.isnan(asr) and asr >= USABLE_THRESHOLD_PCT["ead"]]
    if args.smoke:
        chosen, note = (min(usable) if usable else 100), "smoke run (subset of Phase 1 profiles)"
    elif usable:
        chosen, note = min(usable), "smallest Phase 1 max_iterations with conditional ASR >= 90%"
    else:
        chosen, note = 100, "NO reduced max_iterations reached 90% in Phase 1; outer-loop sweep run at max_iterations=100"
    state["ead_phase2_selection"] = {"phase1_asr_pct": p1, "chosen_max_iterations": chosen, "note": note}
    log(f"EAD Phase 2: max_iterations={chosen} ({note})")
    run_group("ead_phase2", ead_phase2_profiles(chosen))
    raw_f.close()
    if args.save_x_adv:
        save_x_adv(out_dir / "x_adv_rep0.npz", x_store)
    state["completed"] = True


def pkey(prof: dict) -> str:
    """Result key; includes the group so EAD Phase 2's bss9 profile never merges with Phase 1's."""
    return f"{prof['group']}:{prof['attack']}/{prof['profile']}"


def conditional_asr(rows: List[dict], rep: int = 0) -> float:
    rr = [r for r in rows if r["rep"] == rep]
    den = sum(bool(r["clean_correct"]) for r in rr)
    return 100.0 * sum(bool(r["attack_success"]) for r in rr) / den if den else float("nan")


# =========================================================================== outputs
def write_outputs(args, out_dir: Path, state: dict) -> dict:
    results = state.get("profile_results", {})
    n_expected = len(SMOKE_SAMPLE_POSITIONS) if args.smoke else 88
    reps = 1 if args.smoke else args.replications
    formal_base = state.get("formal_baseline_asr_pct", {})

    # per-group latency baselines (in-run)
    group_base = {}
    for key, r in results.items():
        if r["prof"]["is_baseline"] and r["lat_ms"]:
            group_base[r["prof"]["group"]] = (key, float(np.median(r["lat_ms"])), float(np.mean(r["lat_ms"])))
    ead_full_base = results.get("ead_phase1:ead/bss9_maxit100")
    ead_full_med = float(np.median(ead_full_base["lat_ms"])) if ead_full_base and ead_full_base["lat_ms"] else None

    prof_rows, lat_rows = [], []
    for key, r in results.items():
        p = r["prof"]
        rows0 = [x for x in r["rows"] if x["rep"] == 0]
        n_cc = sum(bool(x["clean_correct"]) for x in rows0)
        n_succ = sum(bool(x["attack_success"]) for x in rows0)
        asr = conditional_asr(r["rows"]) if rows0 else float("nan")
        if p["attack"] == "fgsm":
            fb = formal_base.get(f"fgsm/{p['profile']}")
        else:
            fb = formal_base.get(p["attack"])
        thr = USABLE_THRESHOLD_PCT.get(p["attack"])
        lat = dist(r["lat_ms"])
        succ_rows = [x for x in rows0 if x["attack_success"]]
        linf_all, l2_all = [x["linf"] for x in rows0], [x["l2"] for x in rows0]
        gb = group_base.get(p["group"])
        red = (100.0 * (1 - lat["median"] / gb[1])) if gb and lat["n"] else None
        red_full = (100.0 * (1 - lat["median"] / ead_full_med)) if (p["attack"] == "ead" and ead_full_med
                                                                     and lat["n"]) else None
        prof_rows.append({
            "group": p["group"], "attack": p["attack"], "profile": p["profile"], "status": r["status"],
            "invalid_reason": r["invalid_reason"], "is_baseline": p["is_baseline"], "eps": p["eps"],
            "alpha": p["alpha"], "steps": p["steps"], "binary_search_steps": p["binary_search_steps"],
            "max_iterations": p["max_iterations"],
            "cw_c": p["cw_c"] if p["attack"] == "cw" else None, "cw_lr": p["cw_lr"] if p["attack"] == "cw" else None,
            "n_total": len(rows0), "n_clean_correct": n_cc, "n_attack_success": n_succ,
            "conditional_asr_pct": asr, "baseline_conditional_asr_pct": fb,
            "baseline_source": "formal CPU 2200-sample cond_asr_pct (" + FORMAL_ATTACK_SUMMARY + ")",
            "asr_retention_pct": (100.0 * asr / fb) if (fb and not math.isnan(asr)) else None,
            "usable_threshold_pct": thr,
            "usable_90pct": (asr >= thr) if (thr is not None and not math.isnan(asr)) else None,
            "n_pred_changed_rep0": sum(bool(x["pred_changed"]) for x in rows0),
            "latency_median_ms": lat["median"], "latency_mean_ms": lat["mean"], "latency_p95_ms": lat["p95"],
            "latency_n_calls": lat["n"], "latency_reduction_vs_group_baseline_pct": red,
            "latency_reduction_vs_ead_9x100_pct": red_full,
            "linf_median": dist(linf_all)["median"], "linf_mean": dist(linf_all)["mean"],
            "linf_p95": dist(linf_all)["p95"], "l2_median": dist(l2_all)["median"], "l2_mean": dist(l2_all)["mean"],
            "l2_p95": dist(l2_all)["p95"],
            "linf_median_successful": dist([x["linf"] for x in succ_rows])["median"],
            "l2_median_successful": dist([x["l2"] for x in succ_rows])["median"],
            "fallback_count": 0, "invalid_count": 1 if r["status"] == "INVALID_CONFIGURATION" else 0,
            "nonfinite_count": r["nonfinite"], "rep_output_mismatch": r["output_mismatch"],
            "rep_success_mismatch": r["success_mismatch"], "mkldnn_flag_unexpected": r["mkldnn_flag_bad"],
        })
        per_rep = {f"rep{k}_median_ms": float(np.median([x["latency_ms"] for x in r["rows"] if x["rep"] == k]))
                   for k in range(reps) if any(x["rep"] == k for x in r["rows"])}
        lat_rows.append({"group": p["group"], "attack": p["attack"], "profile": p["profile"], "status": r["status"],
                         "n_calls": lat["n"], "median_ms": lat["median"], "mean_ms": lat["mean"],
                         "p95_ms": lat["p95"], "p05_ms": lat["p05"], "std_ms": lat["std"], "min_ms": lat["min"],
                         "max_ms": lat["max"], **per_rep,
                         "group_baseline_profile": gb[0] if gb else None,
                         "group_baseline_median_ms": gb[1] if gb else None,
                         "median_reduction_vs_group_baseline_pct": red,
                         "mean_reduction_vs_group_baseline_pct": (100.0 * (1 - lat["mean"] / gb[2]))
                         if gb and lat["n"] else None,
                         "median_reduction_vs_ead_9x100_pct": red_full})
    write_csv(out_dir / "profile_summary.csv", prof_rows)
    write_csv(out_dir / "latency_summary.csv", lat_rows)

    checks = state["checks"]
    checks["fallback_count_zero"] = state["fallback_count"] == 0
    checks["no_nan_inf"] = all(r["nonfinite"] == 0 for r in results.values())
    complete = [r for r in results.values() if r["status"] == "COMPLETE"]
    checks["sample_matrix_complete"] = bool(results) and all(
        len(r["rows"]) == n_expected * reps and len({x["sample_pos"] for x in r["rows"]}) == n_expected
        for r in complete) and all(r["status"] in ("COMPLETE", "INVALID_CONFIGURATION") for r in results.values())
    checks["mkldnn_opt_in_applied_as_expected"] = all(r["mkldnn_flag_bad"] == 0 for r in results.values())
    checks.setdefault("backend_toggle_matches_configuration", False)
    checks["attack_success_repeatable_across_reps"] = all(r["success_mismatch"] == 0 for r in results.values())
    checks["script_completed"] = bool(state.get("completed"))
    fail_keys = ["real_awn_loaded", "real_attack_backend_loaded", "fallback_count_zero", "dataset_hash_ok",
                 "checkpoint_hash_ok", "sample_matrix_complete", "no_nan_inf", "script_completed",
                 "adversarial_rf_at_pin", "torchattacks_3_5_1", "mkldnn_opt_in_applied_as_expected",
                 "backend_toggle_matches_configuration",
                 "attack_success_repeatable_across_reps"]
    failed = [k for k in fail_keys if not checks.get(k, False)]
    validation = {
        "overall": "PASS" if not failed else "FAIL", "failed_checks": failed, "checks": checks,
        "mode": "smoke" if args.smoke else "full",
        "invalid_configurations": {k: r["invalid_reason"] for k, r in results.items()
                                   if r["status"] == "INVALID_CONFIGURATION"},
        "informational": {
            "bitwise_output_mismatch_across_reps": {k: r["output_mismatch"] for k, r in results.items()},
            "note": "bitwise mismatches across replications are reported, not gated; attack_success mismatches are gated.",
        },
        "abort_reason": state.get("abort_reason"),
    }
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2, default=str))
    return validation


def main() -> None:
    global _LOG_FH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="RML2016.10a_dict.pkl (sha256-checked)")
    ap.add_argument("--checkpoint", required=True, help="2016.10a_AWN.pkl (sha256-checked)")
    ap.add_argument("--smoke", action="store_true", help="3 samples, baseline + lowest setting per attack, 1 rep")
    ap.add_argument("--replications", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=3, help="untimed calls before each profile pass")
    ap.add_argument("--torch-threads", type=int, default=0, help="0 = keep torch default")
    ap.add_argument("--interop-threads", type=int, default=0, help="0 = keep torch default")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--save-x-adv", action="store_true", help="write replication-0 x_adv arrays (x_adv_rep0.npz)")
    ap.add_argument("--legacy-backend-toggle", action="store_true",
                    help="comparison only: reproduce the superseded run that passed both mkldnn-off opt-ins")
    args = ap.parse_args()
    if args.replications < 2 and not args.smoke:
        ap.error("--replications must be >= 2 for the full run")

    name = run_dir_name("reduced_iteration_attacks_vm", args)
    out_dir = (REPO_ROOT / (args.out_dir or f"results/{name}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    _LOG_FH = (out_dir / "run.log").open("w")
    state = {"checks": {}, "fallback_count": 0, "completed": False, "started_at": _dt.datetime.now().isoformat()}
    t0 = time.monotonic()
    log(f"output: {out_dir}")
    rc = 0
    try:
        run(args, out_dir, state)
    except Abort as exc:
        state["abort_reason"] = str(exc)
        log(f"ABORT: {exc}")
        rc = 2
    except Exception as exc:  # noqa: BLE001
        state["abort_reason"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        log(f"ERROR: {state['abort_reason']}")
        rc = 3
    finally:
        validation = write_outputs(args, out_dir, state)
        manifest = {
            "experiment": "reduced_iteration_attacks_vm", "mode": "smoke" if args.smoke else "full",
            "command": [sys.executable] + sys.argv, "args": vars(args),
            "started_at": state["started_at"], "finished_at": _dt.datetime.now().isoformat(),
            "elapsed_s": time.monotonic() - t0,
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "git": state.get("git"), "environment": state.get("environment"), "inputs": state.get("inputs"),
            "formal_config": FORMAL_CONFIG, "formal_config_sha256": sha256_file(REPO_ROOT / FORMAL_CONFIG),
            "samples": {"definition": "11 modulations x SNR {-20,-6,6,18} x sample_index {0,1}, rebuilt with "
                                      "run_cpu_full_matrix.build_sensing_input; B=1",
                        "positions": list(SMOKE_SAMPLE_POSITIONS) if args.smoke else "0..87",
                        "rows": state.get("samples")},
            "metric": {"attack_success": "clean_correct AND NOT attacked_correct (formal)",
                       "conditional_asr": "sum(attack_success)/sum(clean_correct), replication 0",
                       "pred_changed": "informational only, not the formal success definition"},
            "matrix": {"fgsm_eps": FGSM_EPS, "pgd": {"eps": PGD_EPS, "alpha": PGD_ALPHA, "random_start": True,
                                                     "steps": PGD_STEPS},
                       "cw": {"c": 1.0, "kappa": 0.0, "lr": 0.01, "steps": CW_STEPS},
                       "ead_phase1": {"binary_search_steps": 9, "max_iterations": EAD_MAX_ITERS, **EAD_FIXED},
                       "ead_phase2": {"binary_search_steps": EAD_BSS, **EAD_FIXED},
                       "usable_threshold_pct": USABLE_THRESHOLD_PCT,
                       "formal_baseline_asr_pct": state.get("formal_baseline_asr_pct")},
            "ead_phase2_selection": state.get("ead_phase2_selection"),
            "latency_protocol": {
                "timer": "time.perf_counter_ns around AttackAdapter.apply() only",
                "includes": "adapter min-max mapping, internal clean forward, attack construction, atk(x, y), "
                            "output mapping, state restoration (adapter stdout redirected to a null sink)",
                "excludes": "torch.manual_seed, AWN attacked inference, metrics, hashing, CSV writing",
                "warmup_untimed_calls_per_profile_pass": args.warmup,
                "replications": 1 if args.smoke else args.replications,
                "order": "profiles of each attack group in listed order on even replications, reversed on odd",
                "rng": "torch.manual_seed(formal per-sample seed = derive_seed(base, mod, snr, idx, '')) before "
                       "every call; identical across profiles",
                "platform": "x86 desktop/VM; no thermal gate, governor change, cooldown or Hailo",
                "threads": "set by --torch-threads / --interop-threads (see environment.torch_num_threads). Pre-run "
                           "probe on this VMware VM (16 vCPU): torch default 16 intra threads gave pathological B=1 "
                           "tails (PGD steps10 median 29.7 ms, p95 613 ms) vs intra 1 (median 14.7 ms, p95 18.7 ms); "
                           "the formal run uses intra 1 / interop 1.",
            },
            "backend_toggle": state.get("backend_toggle") or backend_toggle_manifest(args.legacy_backend_toggle),
            "attack_adapter_options": ("AttackAdapter(b1_pgd_bim_mkldnn_off=True, b1_mkldnn_off_extended=True)"
                                       if args.legacy_backend_toggle else
                                       "AttackAdapter defaults (b1_pgd_bim_mkldnn_off=False, "
                                       "b1_mkldnn_off_extended=False)"),
            "validation_overall": validation["overall"],
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        log(f"validation: {validation['overall']} failed={validation['failed_checks']} "
            f"({time.monotonic() - t0:.0f} s)")
        if _LOG_FH:
            _LOG_FH.close()
    sys.exit(rc if rc else (0 if validation["overall"] == "PASS" else 1))


if __name__ == "__main__":
    main()
