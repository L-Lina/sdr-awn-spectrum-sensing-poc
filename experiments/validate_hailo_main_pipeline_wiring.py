"""
Code-level validation of the Hailo-8 main-pipeline wiring
(src/utils/config.py: inference_backend / hef_path; src/utils/pipeline.py:
attack surrogate vs evaluation backend split). Runs on the Raspberry Pi with
the Hailo-8 attached; uses the small synthetic fixture, not the full matrix.

Checks
------
 A. CPU default path unchanged: summary.csv byte-identical to --base-ref
    (default HEAD) for the placeholder path and for real AWN + real FGSM +
    real fixed-K (K=20); no backend columns added to CPU rows.
 B. inference_backend=hailo constructs HailoAWNAdapter and routes clean /
    attacked / defended inference through HailoAWNAdapter.infer; the device
    is closed afterwards.
 C. CPU and Hailo classifier outputs are [N, 11].
 D. Missing HEF fails explicitly (no CPU fallback); hailo without hef_path,
    hailo without --use-real-awn and hef_path with cpu are rejected.
 E. Attack generation receives the CPU PyTorch AWN (torch.nn.Module of the
    CPU AWNModelAdapter) as its model.
 F. HailoAWNAdapter.infer is never called while the attack is generated.
 G. Hailo + real fixed-K runs (defense_mode=fixed, defense_topk=K).
 H. CPU + Adaptive-K runs.
 I. Hailo + Adaptive-K runs.
 J. Metadata: Hailo rows carry inference_backend / attack_surrogate_backend /
    hef_path / hef_sha256 / defense_mode / defense_topk; the result dict carries
    them for every mode; no column named hilo_backend.

Run on the Pi from the repo root:
    .venv/bin/python experiments/validate_hailo_main_pipeline_wiring.py
    (HEF defaults to configs/hailo_full_matrix.json:hef_path; override with --hef-path)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import py_compile
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MODIFIED = ["src/adapters/topk_adapter.py", "src/utils/config.py", "src/utils/pipeline.py"]
N_CLASSES = 11

PASS: list = []
FAIL: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))


def base_kwargs(output_dir: str, **overrides) -> dict:
    # Same synthetic fixture as experiments/test_torch_thread_config.py:base_cfg.
    kwargs = dict(
        snr=0.0, mod="BPSK", attack="fgsm", topk=20, threshold_factor=1.5,
        window_size=128, min_region_len=0, merge_gap=0, burst_len=600,
        output_dir=output_dir, dry_run=True,
    )
    kwargs.update(overrides)
    return kwargs


REAL = dict(use_real_awn=True, use_real_attack=True)

_RUNNER = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from src.utils.config import ExperimentConfig
from src.utils.pipeline import run_dry_run_experiment
run_dry_run_experiment(ExperimentConfig(**json.loads(sys.argv[2])))
"""


def run_in_tree(tree: Path, kwargs: dict) -> bytes:
    proc = subprocess.run([sys.executable, "-c", _RUNNER, str(tree), json.dumps(kwargs)],
                          cwd=str(tree), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"pipeline run failed in {tree}:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
    return (Path(kwargs["output_dir"]) / "summary.csv").read_bytes()


def export_base_tree(ref: str, dest: Path) -> None:
    archive = subprocess.run(["git", "-C", str(REPO), "archive", ref, "src"], capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        tf.extractall(dest)
    os.symlink(REPO / "external", dest / "external")  # same checkpoint / submodule for both trees


def read_rows(out_dir: Path) -> list:
    with open(out_dir / "summary.csv", newline="") as f:
        return list(csv.DictReader(f))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef-path", default=None, help="default: configs/hailo_full_matrix.json hef_path")
    ap.add_argument("--base-ref", default="HEAD")
    args = ap.parse_args()
    hef = args.hef_path or json.loads((REPO / "configs" / "hailo_full_matrix.json").read_text())["hef_path"]
    if not Path(hef).is_file():
        print(f"[ABORT] HEF not found: {hef}")
        return 2

    for rel in MODIFIED:
        try:
            py_compile.compile(str(REPO / rel), doraise=True)
            check(f"py_compile {rel}", True)
        except py_compile.PyCompileError as exc:
            check(f"py_compile {rel}", False, str(exc))

    import numpy as np
    import torch

    import src.utils.pipeline as pl
    from src.adapters.awn_adapter import AWNModelAdapter
    from src.adapters.hailo_awn_adapter import HailoAWNAdapter
    from src.utils.config import ExperimentConfig, args_to_config, build_arg_parser, validate_experiment_config

    # ---------------------------------------------------------------- spies (module namespace of pipeline)
    state = {"in_attack": False, "hailo_init": 0, "hailo_infer": 0, "hailo_infer_in_attack": 0,
             "hailo_close": 0, "hailo_logit_shapes": [], "attack_models": [], "cpu_awn": [],
             "cpu_infer_shapes": []}

    class SpyHailo(HailoAWNAdapter):
        def __init__(self, hef_path):
            state["hailo_init"] += 1
            super().__init__(hef_path)

        def infer(self, x, n_classes=11, seed=None):
            state["hailo_infer"] += 1
            if state["in_attack"]:
                state["hailo_infer_in_attack"] += 1
            logits, meta = super().infer(x, n_classes=n_classes, seed=seed)
            state["hailo_logit_shapes"].append(tuple(logits.shape))
            return logits, meta

        def close(self):
            state["hailo_close"] += 1
            super().close()

    class SpyCPU(AWNModelAdapter):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            state["cpu_awn"].append(self)

        def infer(self, x, n_classes=11, seed=0):
            logits, meta = super().infer(x, n_classes=n_classes, seed=seed)
            state["cpu_infer_shapes"].append(tuple(logits.shape))
            return logits, meta

    real_attack_cls = pl.AttackAdapter

    class SpyAttack(real_attack_cls):
        def __init__(self, awn_model=None, device="cpu", **kw):
            state["attack_models"].append(awn_model)
            super().__init__(awn_model=awn_model, device=device, **kw)

        def apply(self, *a, **kw):
            state["in_attack"] = True
            try:
                return super().apply(*a, **kw)
            finally:
                state["in_attack"] = False

    originals = (pl.HailoAWNAdapter, pl.AWNModelAdapter, pl.AttackAdapter)
    pl.HailoAWNAdapter, pl.AWNModelAdapter, pl.AttackAdapter = SpyHailo, SpyCPU, SpyAttack

    def reset():
        for k in state:
            state[k] = False if k == "in_attack" else (0 if isinstance(state[k], int) else [])

    def run(out_dir: Path, **ov) -> dict:
        reset()
        return pl.run_dry_run_experiment(ExperimentConfig(**base_kwargs(str(out_dir), **ov)))

    try:
        with tempfile.TemporaryDirectory(prefix="hailo_wiring_") as tmp_s:
            tmp = Path(tmp_s)

            # ---------------- D: config / CLI rejection and missing HEF (no fallback)
            parser = build_arg_parser("validate_hailo_main_pipeline_wiring")
            check("D CLI default inference_backend == cpu",
                  args_to_config(parser.parse_args(["--dry-run"])).inference_backend == "cpu")
            for label, argv in (
                    ("hailo without --hef-path", ["--dry-run", "--use-real-awn", "--inference-backend", "hailo"]),
                    ("hailo without --use-real-awn", ["--dry-run", "--inference-backend", "hailo", "--hef-path", hef]),
                    ("--hef-path with cpu", ["--dry-run", "--use-real-awn", "--hef-path", hef])):
                try:
                    validate_experiment_config(args_to_config(parser.parse_args(argv)))
                    check(f"D rejected: {label}", False)
                except ValueError:
                    check(f"D rejected: {label}", True)
            missing = str(tmp / "does_not_exist.hef")
            try:
                run(tmp / "missing_hef", **REAL, inference_backend="hailo", hef_path=missing)
                check("D missing HEF raises (no CPU fallback)", False)
            except FileNotFoundError:
                check("D missing HEF raises (no CPU fallback)", True)
            check("D missing HEF: no summary.csv written", not (tmp / "missing_hef" / "summary.csv").exists())

            # ---------------- A: CPU default path vs base ref
            base_tree = tmp / "base"
            export_base_tree(args.base_ref, base_tree)
            for label, ov in (("placeholder_default", {}),
                              ("real_awn_fgsm_fixedK20", dict(**REAL, use_real_topk=True, topk=20))):
                cur = run_in_tree(REPO, base_kwargs(str(tmp / f"cur_{label}"), **ov))
                base = run_in_tree(base_tree, base_kwargs(str(tmp / f"base_{label}"), **ov))
                check(f"A CPU {label}: summary.csv byte-identical to {args.base_ref}", cur == base)
                header = cur.splitlines()[0].decode()
                check(f"A CPU {label}: no backend columns added",
                      "inference_backend" not in header and "hef_sha256" not in header)

            # ---------------- CPU real path in-process: C (cpu), E, J (result dict)
            res = run(tmp / "cpu_fixed", **REAL, use_real_topk=True, topk=20)
            cpu_awn = state["cpu_awn"][0] if state["cpu_awn"] else None
            check("C CPU logits are [N, 11]", res["awn_logits_shape"][1] == N_CLASSES
                  and all(s[1] == N_CLASSES for s in state["cpu_infer_shapes"]), str(state["cpu_infer_shapes"]))
            check("E CPU mode: attack model is the CPU AWN torch module",
                  cpu_awn is not None and state["attack_models"] and state["attack_models"][0] is cpu_awn.model
                  and isinstance(cpu_awn.model, torch.nn.Module))
            check("J CPU result dict: inference_backend=cpu, defense_mode=fixed, defense_topk=20, no HEF",
                  res["inference_backend"] == "cpu" and res["defense_mode"] == "fixed" and res["defense_topk"] == 20
                  and res["hef_path"] is None and res["hef_sha256"] is None, str({k: res.get(k) for k in (
                      "inference_backend", "defense_mode", "defense_topk", "hef_path")}))
            check("B CPU mode never constructs HailoAWNAdapter", state["hailo_init"] == 0)

            # ---------------- H: CPU + Adaptive-K
            res = run(tmp / "cpu_adaptive", **REAL, use_real_topk=True, topk_mode="adaptive")
            rows = read_rows(tmp / "cpu_adaptive")
            check("H CPU + Adaptive-K runs", len(rows) > 0 and all(
                r["topk_mode"] == "adaptive" and r["topk_backend"].endswith("adaptive_k_v2_snr_defense")
                for r in rows))
            check("H/J CPU + Adaptive-K result dict: defense_mode=adaptive, defense_topk=None",
                  res["defense_mode"] == "adaptive" and res["defense_topk"] is None)

            # ---------------- G: Hailo + fixed-K  (B, C, E, F, J)
            hef_sha = sha256_file(hef)
            res = run(tmp / "hailo_fixed", **REAL, use_real_topk=True, topk=20, inference_backend="hailo",
                      hef_path=hef)
            rows = read_rows(tmp / "hailo_fixed")
            cpu_awn = state["cpu_awn"][0] if state["cpu_awn"] else None
            n_seg = res["n_segments"]
            check("B Hailo: HailoAWNAdapter constructed once", state["hailo_init"] == 1, str(state["hailo_init"]))
            check("B Hailo: clean/attacked/defended inference via HailoAWNAdapter.infer",
                  state["hailo_infer"] >= 3, str(state["hailo_infer"]))
            check("B Hailo: device closed after the run", state["hailo_close"] >= 1)
            check("B Hailo: CPU AWN not used for evaluation inference", state["cpu_infer_shapes"] == [],
                  str(state["cpu_infer_shapes"]))
            check("C Hailo logits are [N, 11]", bool(state["hailo_logit_shapes"]) and all(
                s == (s[0], N_CLASSES) for s in state["hailo_logit_shapes"]), str(state["hailo_logit_shapes"]))
            check("E Hailo mode: attack model is the CPU AWN torch module",
                  cpu_awn is not None and len(state["attack_models"]) == 1
                  and state["attack_models"][0] is cpu_awn.model and isinstance(cpu_awn.model, torch.nn.Module))
            check("F HailoAWNAdapter.infer never called during attack generation",
                  state["hailo_infer_in_attack"] == 0, str(state["hailo_infer_in_attack"]))
            check("F real attack ran on the surrogate (attack_status ok, not dummy)",
                  all(r["attack_status"] == "ok" and r["attack_backend"] != "dummy_attack" for r in rows))
            check("G Hailo + fixed-K runs", len(rows) == n_seg and all(
                r["defense_mode"] == "fixed" and r["defense_topk"] == "20"
                and r["topk_backend"].endswith("fft_topk_denoise") for r in rows))
            check("J Hailo rows: backend / surrogate / HEF identity", all(
                r["inference_backend"] == "hailo" and r["awn_backend"].startswith("Hailo-8")
                and r["attack_surrogate_backend"] == cpu_awn.backend_name
                and r["hef_path"] == str(Path(hef).expanduser().resolve()) and r["hef_sha256"] == hef_sha
                for r in rows), str(rows[0] if rows else {}))
            check("J no 'hilo_backend' column", "hilo_backend" not in (rows[0].keys() if rows else []))
            x_ok = all(np.isfinite(float(r["iq_linf_clean_attacked"])) for r in rows if "iq_linf_clean_attacked" in r)
            check("G Hailo + fixed-K: finite attack perturbation recorded", x_ok)

            # ---------------- I: Hailo + Adaptive-K
            res = run(tmp / "hailo_adaptive", **REAL, use_real_topk=True, topk_mode="adaptive",
                      inference_backend="hailo", hef_path=hef)
            rows = read_rows(tmp / "hailo_adaptive")
            check("I Hailo + Adaptive-K runs", len(rows) > 0 and all(
                r["inference_backend"] == "hailo" and r["topk_mode"] == "adaptive" and r["defense_mode"] == "adaptive"
                and r["defense_topk"] == "" and r["topk_backend"].endswith("adaptive_k_v2_snr_defense") for r in rows))
            check("I Hailo + Adaptive-K: Hailo used, not during attack",
                  state["hailo_infer"] >= 3 and state["hailo_infer_in_attack"] == 0 and state["hailo_close"] >= 1)

            # ---------------- placeholder (legacy default) on Hailo: metadata says placeholder
            res = run(tmp / "hailo_placeholder", **REAL, inference_backend="hailo", hef_path=hef)
            rows = read_rows(tmp / "hailo_placeholder")
            check("J Hailo + legacy placeholder defense labelled defense_mode=placeholder",
                  len(rows) > 0 and all(r["defense_mode"] == "placeholder"
                                        and r["topk_backend"] == "dummy_topk_defense" for r in rows))
    finally:
        pl.HailoAWNAdapter, pl.AWNModelAdapter, pl.AttackAdapter = originals

    print(f"\nSUMMARY: {len(PASS)} PASS, {len(FAIL)} FAIL")
    for name in FAIL:
        print(f"  FAIL: {name}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
