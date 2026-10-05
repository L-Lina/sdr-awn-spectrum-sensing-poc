"""
Code-level validation of the Adaptive-K main-pipeline wiring:
src/adapters/topk_adapter.py (TopKAdapter mode="fixed"|"adaptive"),
src/utils/config.py (topk_mode field, --topk-mode flag, validation) and
src/utils/pipeline.py (topk_mode dispatch).

Checks
------
 1. py_compile of the three modified modules.
 2. Import path: TopKAdapter's adaptive function object IS
    external/adversarial-rf/util/defense.py:adaptive_k_v2_snr_defense
    (identity, not a copy).
 3. Fixed-K adapter unchanged: for K in {10,20,30,40} the current
    TopKAdapter().apply() is bitwise equal to the --base-ref (default HEAD)
    TopKAdapter and to fft_topk_denoise directly; meta keys/values equal.
 4. Adaptive adapter: output bitwise equal to a direct
    adaptive_k_v2_snr_defense call; shape [N,2,T], float32, finite; a spy
    proves exactly one production call per apply(); topk!=None is rejected.
 5. Config/CLI: default topk_mode == "fixed"; --topk-mode adaptive without
    --use-real-topk raises; unknown mode rejected by argparse.
 6. Pipeline, fixed paths: summary.csv from the current tree is byte-identical
    to the --base-ref tree for (no --use-real-topk) and (--use-real-topk,
    K=10/20/30/40); explicit topk_mode="fixed" equals the default.
 7. Pipeline, adaptive path: run_dry_run_experiment reaches the production
    function (spy call count > 0), never calls fft_topk_denoise or
    dummy_topk_defense, and records topk_mode/topk_backend in summary.csv.
 8. Missing dependency: with the production symbol removed, constructing
    TopKAdapter(mode="adaptive") raises instead of falling back.

Run on the Pi from the repo root:
    .venv/bin/python experiments/validate_adaptive_k_pipeline_wiring.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import py_compile
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MODIFIED = ["src/adapters/topk_adapter.py", "src/utils/config.py", "src/utils/pipeline.py"]
FIXED_KS = [10, 20, 30, 40]

PASS: list = []
FAIL: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))


def base_kwargs(output_dir: str, **overrides) -> dict:
    # Same synthetic fixture as experiments/test_torch_thread_config.py:base_cfg
    # (reliably yields a detected region, so the Top-K stage is exercised).
    kwargs = dict(
        snr=0.0, mod="BPSK", attack="fgsm", topk=10, threshold_factor=1.5,
        window_size=128, min_region_len=0, merge_gap=0, burst_len=600,
        output_dir=output_dir, dry_run=True,
    )
    kwargs.update(overrides)
    return kwargs


_RUNNER = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from src.utils.config import ExperimentConfig
from src.utils.pipeline import run_dry_run_experiment
run_dry_run_experiment(ExperimentConfig(**json.loads(sys.argv[2])))
"""


def run_pipeline_in(tree: Path, kwargs: dict) -> bytes:
    """Run run_dry_run_experiment from `tree` in a fresh interpreter; return summary.csv bytes."""
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(tree), json.dumps(kwargs)],
        cwd=str(tree), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pipeline run failed in {tree}:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
    return (Path(kwargs["output_dir"]) / "summary.csv").read_bytes()


def export_base_tree(ref: str, dest: Path) -> None:
    """Extract src/ at `ref` into dest and copy the checked-out adversarial-rf util/ next to it."""
    archive = subprocess.run(["git", "-C", str(REPO), "archive", ref, "src"], capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        tf.extractall(dest)
    shutil.copytree(REPO / "external" / "adversarial-rf" / "util", dest / "external" / "adversarial-rf" / "util")


# Runs the base-ref TopKAdapter (fixed mode) in a fresh interpreter and saves its outputs.
_BASE_ADAPTER_CODE = r"""
import sys, json, numpy as np
sys.path.insert(0, sys.argv[1])
from src.adapters.topk_adapter import TopKAdapter
x = np.load(sys.argv[2])
out = {}
for k in json.loads(sys.argv[3]):
    y, meta = TopKAdapter().apply(x, topk=k)
    np.save(sys.argv[4] + f"_k{k}.npy", y)
    out[str(k)] = meta
print("META_JSON=" + json.dumps(out))
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ref", default="HEAD", help="git ref holding the pre-change code (default HEAD)")
    args = ap.parse_args()

    # 1. syntax
    for rel in MODIFIED:
        try:
            py_compile.compile(str(REPO / rel), doraise=True)
            check(f"py_compile {rel}", True)
        except py_compile.PyCompileError as exc:
            check(f"py_compile {rel}", False, str(exc))

    import numpy as np
    import torch

    import src.adapters.topk_adapter as ta
    from src.adapters.topk_adapter import TopKAdapter
    from src.utils.config import ExperimentConfig, args_to_config, build_arg_parser, validate_experiment_config

    sys.path.insert(0, str(REPO / "external" / "adversarial-rf"))
    import util.defense as prod

    # 2. import path / identity
    check("adaptive symbol is the production function object",
          ta._real_adaptive_k_v2_snr_defense is prod.adaptive_k_v2_snr_defense)
    src_file = Path(prod.adaptive_k_v2_snr_defense.__code__.co_filename).resolve()
    check("adaptive function defined in external/adversarial-rf/util/defense.py",
          src_file == (REPO / "external" / "adversarial-rf" / "util" / "defense.py").resolve(), str(src_file))
    check("fixed symbol is the production fft_topk_denoise", ta._real_fft_topk_denoise is prod.fft_topk_denoise)

    rng = np.random.default_rng(1234)
    x = rng.standard_normal((6, 2, 128)).astype(np.float32)
    # one narrowband-like sample (pure tones) so both routing branches run
    t = np.arange(128, dtype=np.float32)
    x[0, 0] = np.cos(2 * np.pi * 5 * t / 128)
    x[0, 1] = np.sin(2 * np.pi * 5 * t / 128)

    with tempfile.TemporaryDirectory(prefix="adaptive_k_wiring_") as tmp_s:
        tmp = Path(tmp_s)
        base_tree = tmp / "base"
        export_base_tree(args.base_ref, base_tree)

        # 3. fixed-K adapter vs base ref and vs direct production call
        np.save(tmp / "x.npy", x)
        proc = subprocess.run(
            [sys.executable, "-c", _BASE_ADAPTER_CODE, str(base_tree), str(tmp / "x.npy"),
             json.dumps(FIXED_KS), str(tmp / "base_y")],
            cwd=str(base_tree), capture_output=True, text=True,
        )
        if proc.returncode != 0:
            check("base-ref TopKAdapter ran", False, proc.stderr[-2000:])
            base_meta = {}
        else:
            line = [ln for ln in proc.stdout.splitlines() if ln.startswith("META_JSON=")][-1]
            base_meta = json.loads(line[len("META_JSON="):])
        adapter_fixed = TopKAdapter()
        check("default mode is 'fixed'", adapter_fixed.mode == "fixed")
        for k in FIXED_KS:
            y, meta = adapter_fixed.apply(x, topk=k)
            y_direct = prod.fft_topk_denoise(torch.from_numpy(x), topk=k).numpy().astype(np.float32)
            check(f"fixed K={k} == direct fft_topk_denoise (bitwise)", np.array_equal(y, y_direct))
            base_path = tmp / f"base_y_k{k}.npy"
            if base_path.exists():
                y_base = np.load(base_path)
                check(f"fixed K={k} == base-ref TopKAdapter (bitwise)",
                      np.array_equal(y, y_base) and y.dtype == y_base.dtype)
                check(f"fixed K={k} meta == base-ref meta", meta == base_meta.get(str(k)),
                      f"{meta} vs {base_meta.get(str(k))}")

        # 4. adaptive adapter
        calls = {"n": 0}
        real_fn = ta._real_adaptive_k_v2_snr_defense

        def spy(x_t, *a, **kw):
            calls["n"] += 1
            check("spy: input is torch.float32 [N,2,T]",
                  isinstance(x_t, torch.Tensor) and x_t.dtype == torch.float32 and x_t.ndim == 3 and x_t.shape[1] == 2)
            check("spy: called with production defaults (no extra args)", not a and not kw, f"{a} {kw}")
            return real_fn(x_t, *a, **kw)

        ta._real_adaptive_k_v2_snr_defense = spy
        try:
            adapter_ad = TopKAdapter(mode="adaptive")
            y_ad, meta_ad = adapter_ad.apply(x)
        finally:
            ta._real_adaptive_k_v2_snr_defense = real_fn
        y_ad_direct = prod.adaptive_k_v2_snr_defense(torch.from_numpy(x)).numpy().astype(np.float32)
        check("adaptive: exactly one production call per apply()", calls["n"] == 1, str(calls["n"]))
        check("adaptive: output == direct production call (bitwise)", np.array_equal(y_ad, y_ad_direct))
        check("adaptive: shape preserved", y_ad.shape == x.shape, str(y_ad.shape))
        check("adaptive: dtype float32", y_ad.dtype == np.float32, str(y_ad.dtype))
        check("adaptive: all finite", bool(np.isfinite(y_ad).all()))
        check("adaptive: output differs from input (defense applied)", not np.array_equal(y_ad, x))
        check("adaptive: meta", meta_ad.get("topk_mode") == "adaptive"
              and meta_ad.get("topk_status") == "ok"
              and meta_ad.get("topk_backend", "").endswith("adaptive_k_v2_snr_defense"), str(meta_ad))
        try:
            TopKAdapter(mode="adaptive").apply(x, topk=20)
            check("adaptive: topk!=None rejected", False)
        except ValueError:
            check("adaptive: topk!=None rejected", True)
        try:
            TopKAdapter(mode="auto")
            check("unknown TopKAdapter mode rejected", False)
        except ValueError:
            check("unknown TopKAdapter mode rejected", True)

        # 5. config / CLI
        parser = build_arg_parser("validate_adaptive_k_pipeline_wiring")
        check("CLI default topk_mode == 'fixed'", args_to_config(parser.parse_args(["--dry-run"])).topk_mode == "fixed")
        check("CLI --use-real-topk alone keeps fixed mode",
              args_to_config(parser.parse_args(["--dry-run", "--use-real-topk"])).topk_mode == "fixed")
        cfg_ad = args_to_config(parser.parse_args(["--dry-run", "--use-real-topk", "--topk-mode", "adaptive"]))
        check("CLI --topk-mode adaptive parsed", cfg_ad.topk_mode == "adaptive" and cfg_ad.use_real_topk)
        try:
            validate_experiment_config(
                args_to_config(parser.parse_args(["--dry-run", "--topk-mode", "adaptive"])))
            check("adaptive without --use-real-topk rejected", False)
        except ValueError:
            check("adaptive without --use-real-topk rejected", True)
        try:
            parser.parse_args(["--dry-run", "--topk-mode", "auto"])
            check("CLI unknown --topk-mode rejected", False)
        except SystemExit:
            check("CLI unknown --topk-mode rejected", True)
        check("ExperimentConfig default topk_mode == 'fixed'",
              ExperimentConfig(**base_kwargs(str(tmp / "unused"))).topk_mode == "fixed")

        # 6. pipeline fixed paths: current tree vs base ref, byte-identical summary.csv
        fixed_cases = [("dummy_topk", dict(use_real_topk=False))] + [
            (f"real_topk_k{k}", dict(use_real_topk=True, topk=k)) for k in FIXED_KS]
        for label, ov in fixed_cases:
            cur = run_pipeline_in(REPO, base_kwargs(str(tmp / f"cur_{label}"), **ov))
            base = run_pipeline_in(base_tree, base_kwargs(str(tmp / f"base_{label}"), **ov))
            check(f"pipeline {label}: summary.csv byte-identical to base-ref", cur == base)
            check(f"pipeline {label}: no topk_mode column", b"topk_mode" not in cur.splitlines()[0])
        explicit = run_pipeline_in(REPO, base_kwargs(str(tmp / "cur_explicit_fixed"), use_real_topk=True,
                                                     topk=20, topk_mode="fixed"))
        default = (tmp / "cur_real_topk_k20" / "summary.csv").read_bytes()
        check("pipeline explicit topk_mode='fixed' == default", explicit == default)

        # 7. pipeline adaptive path (in-process, with spies)
        from src.utils import pipeline as pl

        counters = {"adaptive": 0, "fixed": 0, "dummy": 0}
        real_fixed = ta._real_fft_topk_denoise
        real_dummy = pl.dummy_topk_defense

        def spy_adaptive(x_t, *a, **kw):
            counters["adaptive"] += 1
            return real_fn(x_t, *a, **kw)

        def spy_fixed(*a, **kw):
            counters["fixed"] += 1
            return real_fixed(*a, **kw)

        def spy_dummy(*a, **kw):
            counters["dummy"] += 1
            return real_dummy(*a, **kw)

        ta._real_adaptive_k_v2_snr_defense = spy_adaptive
        ta._real_fft_topk_denoise = spy_fixed
        pl.dummy_topk_defense = spy_dummy
        try:
            out_dir = tmp / "cur_adaptive"
            pl.run_dry_run_experiment(ExperimentConfig(**base_kwargs(
                str(out_dir), use_real_topk=True, topk_mode="adaptive")))
        finally:
            ta._real_adaptive_k_v2_snr_defense = real_fn
            ta._real_fft_topk_denoise = real_fixed
            pl.dummy_topk_defense = real_dummy
        check("pipeline adaptive: production function reached", counters["adaptive"] > 0, str(counters))
        check("pipeline adaptive: fft_topk_denoise not called", counters["fixed"] == 0, str(counters))
        check("pipeline adaptive: dummy_topk_defense not called", counters["dummy"] == 0, str(counters))
        with open(out_dir / "summary.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        check("pipeline adaptive: summary rows written", len(rows) > 0)
        check("pipeline adaptive: rows record topk_mode=adaptive, topk empty",
              all(r.get("topk_mode") == "adaptive" and r.get("topk") == "" for r in rows))
        check("pipeline adaptive: rows record production backend, status ok",
              all(r["topk_backend"].endswith("adaptive_k_v2_snr_defense") and r["topk_status"] == "ok" for r in rows))

        # 8. missing dependency -> explicit error, no fallback
        ta._real_adaptive_k_v2_snr_defense = None
        try:
            TopKAdapter(mode="adaptive")
            check("missing adaptive dependency raises", False)
        except RuntimeError:
            check("missing adaptive dependency raises", True)
        try:
            pl.run_dry_run_experiment(ExperimentConfig(**base_kwargs(
                str(tmp / "cur_adaptive_missing"), use_real_topk=True, topk_mode="adaptive")))
            check("pipeline adaptive with missing dependency raises", False)
        except RuntimeError:
            check("pipeline adaptive with missing dependency raises", True)
        finally:
            ta._real_adaptive_k_v2_snr_defense = real_fn

    print(f"\nSUMMARY: {len(PASS)} PASS, {len(FAIL)} FAIL")
    for name in FAIL:
        print(f"  FAIL: {name}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
