"""
Phase 1C v2 batched / vectorized attack-generation benchmark (Raspberry Pi 5, CPU only).

V2 = the original experiments/bench_attack_batch_pi5.py (unmodified, kept for provenance)
with ONE change of correctness rule, adopted after the BIM batch diagnostics:
  - experiments/diagnose_bim_batch_divergence_pi5.py showed that batch-shape-dependent
    floating-point differences in the AWN forward/backward path (not loss reduction,
    batch partner or position) can flip one near-zero gradient sign during iterative BIM;
  - experiments/audit_bim_batch_semantics_pi5.py (88 samples x B in {1..40}) found
    697/704 bit-identical, 7/704 BITWISE_DIFFERENT_SEMANTIC_SAME (one sample, every B>1,
    max |dx| = 0.0010695941746234894), 0 SEMANTIC_DIFFERENT, 0 INVALID.
So bitwise identity is no longer the SOLE criterion. Every batched/sequential sample is
still compared to its independent B=1 anchor and classified:
  BIT_IDENTICAL                    valid
  BITWISE_DIFFERENT_SEMANTIC_SAME  valid for timing; numerical difference recorded
  SEMANTIC_DIFFERENT               attacked prediction or attack_success changed ->
                                   the attack x B x thread cell FAILS; its speedup /
                                   throughput are withheld (run continues)
  INVALID                          NaN/Inf, bad shape/dtype, eps-bound violation, backend
                                   fallback, model/thread/state corruption -> abort now
Nothing is silently ignored: every bitwise mismatch is recorded per sample and per cell.
B=1 anchors, the formal adversarial_sha256 checks (FGSM/BIM) and the ControlledStartPGD
fidelity check are unchanged and still fail closed.

Question answered (and ONLY this question): can batched attack generation improve
throughput and amortized per-sample latency on the Pi 5 Cortex-A76 WITHOUT changing
attack semantics? Four quantities are kept strictly apart:
  1. batch completion latency     (wall time of one batched call of size B)
  2. amortized latency per sample (batch completion latency / B)
  3. throughput                   (samples / second)
  4. B=1 single-sample latency    (one sequential call)
Amortized latency is NOT a per-decision latency: a sample in a batch waits for the
whole batch (plus, in a live system, for the batch to fill). The 20-40 ms real-time
budget must never be judged on amortized latency alone.

Scope: FGSM / PGD / BIM eps=0.03 (formal profiles); batch sizes 1,2,4,8,16,20,32,40;
intra-op threads 1,2,4 with inter-op 1, each thread condition in a fresh process.
Path = the validated Phase 1B optimized path: clean prediction reused (outside timing),
model parameters NOT frozen, no attack-object reuse, no hyperparameter/normalization
change, no source edits.

Batch-semantics audit (done by source inspection, then re-verified at runtime):
  * external/adversarial-rf util/adv_attack.py @ced705e:
      iq_to_ta_input_minmax: a = x.amin(dim=(1,2), keepdim=True), b = (amax - a).clamp_min(1e-6)
        -> per-sample [N,1,1], never shared across the batch
      Model01Wrapper.set_minmax(a, b) stores the [N,1,1] tensors; forward computes
        x_iq = x01 * b + a (per-sample broadcast); ta_output_to_iq_minmax is the same
        per-sample affine inverse.
  * torchattacks 3.5.1 FGSM/PGD/BIM.forward: per-sample labels; eps/alpha are scalars in
    the [0,1] attack domain (so the IQ-domain bound is eps * b_i per sample); every
    clamp/projection is element-wise; loss = CrossEntropyLoss() with MEAN reduction, so
    each sample's gradient is scaled by 1/B -- mathematically irrelevant after .sign()
    but it can change float rounding (hence the bitwise checks below).
    Attack.__call__ draws no random numbers. PGD's ONLY randomness is
    torch.empty_like(adv_images).uniform_(-eps, eps) on the global generator, which a
    batch consumes in a different order than B independent calls.
  * AWN (models/model.py): in eval mode BatchNorm uses running statistics and Dropout is
    inactive; the batch-wide regu_* terms are returned separately and discarded by
    Model01Wrapper (never part of the attack loss). Logits are per-sample.

PGD random_start control: every sample i gets its own start noise
    noise_i = torch.empty([1,2,128,1]).uniform_(-eps, eps, generator=Generator().manual_seed(formal_seed_i))
i.e. exactly the tensor stock PGD draws for that sample in the Phase 1B B=1 path after
torch.manual_seed(formal_seed_i). ControlledStartPGD (defined below, benchmark-local) is
a line-for-line copy of the installed torchattacks.PGD.forward in which ONLY that draw is
replaced by the pre-generated tensor (stacked per sample for a batch). Same distribution,
same eps, same steps/alpha. Its fidelity is proven at runtime: for every sample,
ControlledStartPGD at B=1 must be bit-identical to the validated Phase 1B path running
stock PGD under torch.manual_seed(formal_seed_i). The installed PGD.forward source is also
compared against the copied logic before anything runs.

Correctness: see the V2 rule at the top. Anchors = the validated Phase 1B B=1 path;
FGSM/BIM anchors must equal the formal adversarial_sha256.

Reuses validated helpers from experiments/bench_attack_accel_pi5.py (Phase 1A) and
experiments/bench_attack_overhead_pi5.py (Phase 1B) by import; neither is modified.

Run on the Pi (from the repo root):
    .venv/bin/python experiments/bench_attack_batch_pi5_v2.py
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import inspect
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import bench_attack_accel_pi5 as p1a  # noqa: E402 -- Phase 1A helpers (no torch import at module level)
import bench_attack_overhead_pi5 as p1b  # noqa: E402 -- Phase 1B validated attack path

REPO_ROOT = p1a.REPO_ROOT
DEFAULT_BATCH_SIZES = "1,2,4,8,16,20,32,40"
DEFAULT_THREADS = "1,2,4"
INTEROP_THREADS = 1
ATTACKS = list(p1a.TARGET_PROFILES)  # fgsm, pgd, bim

# sha256 of external/adversarial-rf/util/adv_attack.py at the pinned submodule commit
# ced705ed861bde352841b0b8bd038a6449e0fa4d (raw GitHub file, LF line endings), whose
# min/max + Model01Wrapper code the batch-safety audit above was read from.
PINNED_ADV_ATTACK_SHA256 = "8dae3a7f0984db1fc3761fa1346120dd6f5f6c11dbe68f13b9d9eab8e1f266e8"

# Phase 1B validated medians (reuse_clean_pred, intra1/interop1) and Phase 1A thread
# ratios -- used ONLY for the pre-run runtime estimate, never reported as results.
EST_B1_MS = {"fgsm": 5.60, "pgd": 44.58, "bim": 45.84}
EST_THREAD_FACTOR = {1: 1.0, 2: 1.05, 4: 1.02}
EST_INFER_MS = 2.5
EST_PROCESS_OVERHEAD_S = 45.0

# Installed torchattacks.PGD.forward, reduced to code lines (docstring, comments and blank
# lines removed, each line stripped) -- copied from torchattacks v3.5.1 attacks/pgd.py.
EXPECTED_STOCK_PGD_FORWARD = """\
def forward(self, images, labels):
images = images.clone().detach().to(self.device)
labels = labels.clone().detach().to(self.device)
if self.targeted:
target_labels = self.get_target_label(images, labels)
loss = nn.CrossEntropyLoss()
adv_images = images.clone().detach()
if self.random_start:
adv_images = adv_images + torch.empty_like(adv_images).uniform_(
-self.eps, self.eps
)
adv_images = torch.clamp(adv_images, min=0, max=1).detach()
for _ in range(self.steps):
adv_images.requires_grad = True
outputs = self.get_logits(adv_images)
if self.targeted:
cost = -loss(outputs, target_labels)
else:
cost = loss(outputs, labels)
grad = torch.autograd.grad(
cost, adv_images, retain_graph=False, create_graph=False
)[0]
adv_images = adv_images.detach() + self.alpha * grad.sign()
delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
adv_images = torch.clamp(images + delta, min=0, max=1).detach()
return adv_images"""

CLASSES = ("BIT_IDENTICAL", "BITWISE_DIFFERENT_SEMANTIC_SAME", "SEMANTIC_DIFFERENT", "INVALID")
NORM_EPS_TOL = 1e-6        # normalized [0,1] domain: L_inf <= eps + tol
IQ_EPS_REL_TOL = 1e-5      # IQ domain: L_inf <= eps * b_i * (1 + rel) + abs
IQ_EPS_ABS_TOL = 1e-7      # covers the documented float32 min/max round-trip error (~9.3e-9)
BUDGET_MS = 40.0

RAW_FIELDS = [
    "condition", "intra_threads", "interop_threads", "attack", "attack_profile", "eps",
    "batch_size", "batch_index", "batch_actual_size", "is_partial_batch", "repeat", "mode", "mode_order",
    "total_ms", "per_sample_ms", "prep_ms", "kernel_ms", "post_ms", "seq_call_ms_list",
    "n_bit_identical", "n_bitwise_different_semantic_same", "n_semantic_different", "n_invalid",
    "max_abs_diff_vs_anchor", "max_linf_rel_diff", "max_l2_rel_diff",
    "n_pred_match_anchor", "n_success_match_anchor", "n_finite", "n_eps_bound_ok", "shape_dtype_ok",
    "sample_keys", "cpu_temp_c",
]
SAMPLE_FIELDS = [
    "condition", "intra_threads", "attack", "batch_size", "batch_index", "position_in_batch", "batch_actual_size",
    "repeat", "mode", "modulation", "snr_db", "sample_index", "formal_seed", "label", "clean_pred",
    "attacked_pred", "anchor_attacked_pred", "attacked_pred_equal",
    "attack_success", "anchor_attack_success", "attack_success_equal",
    "iq_linf", "anchor_iq_linf", "linf_abs_diff", "linf_rel_diff",
    "iq_l2", "anchor_iq_l2", "l2_abs_diff", "l2_rel_diff",
    "iq_linf_normalized", "norm_eps_bound_ok", "per_sample_b", "iq_linf_bound_eps_x_b", "iq_eps_bound_ok",
    "x_adv_sha256", "anchor_sha256", "bit_identical_to_anchor", "max_abs_diff_vs_anchor", "n_diff_elements",
    "finite", "shape", "dtype", "shape_dtype_ok", "classification",
]


def _rel(d: float, ref: float) -> float:
    return d / abs(ref) if ref != 0 else (0.0 if d == 0 else float("inf"))


def _throttled_value(raw: Optional[str]) -> Optional[int]:
    """Parses `vcgencmd get_throttled` ('throttled=0x0'); None if unavailable/unparseable."""
    if not raw or "=" not in raw:
        return None
    try:
        return int(raw.split("=", 1)[1].strip(), 16)
    except ValueError:
        return None


def _governors() -> Dict[str, Optional[str]]:
    return {f"cpu{c}": p1a.read_text(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_governor")
            for c in range(os.cpu_count() or 0)}


# --------------------------------------------------------------------------- PGD controlled start
def _code_lines(src: str) -> str:
    src = re.sub(r'[rR]?"""[\s\S]*?"""', "", src, count=1)  # drop the docstring
    lines = [ln.strip() for ln in src.splitlines()]
    return "\n".join(ln for ln in lines if ln and not ln.startswith("#"))


def make_controlled_pgd(torchattacks, torch, nn):
    stock_src = inspect.getsource(torchattacks.PGD.forward)
    if _code_lines(stock_src) != EXPECTED_STOCK_PGD_FORWARD:
        p1a.abort("installed torchattacks.PGD.forward differs from the torchattacks 3.5.1 logic copied into "
                  "ControlledStartPGD; refusing to benchmark a possibly different PGD.\n--- installed ---\n"
                  + _code_lines(stock_src))

    class ControlledStartPGD(torchattacks.PGD):
        """torchattacks.PGD with the random start supplied per sample instead of drawn
        from the global RNG. Everything else is the installed 3.5.1 forward verbatim."""

        _init_noise = None

        def set_init_noise(self, noise):
            object.__setattr__(self, "_init_noise", noise)

        def forward(self, images, labels):
            images = images.clone().detach().to(self.device)
            labels = labels.clone().detach().to(self.device)
            if self.targeted:
                target_labels = self.get_target_label(images, labels)
            loss = nn.CrossEntropyLoss()
            adv_images = images.clone().detach()
            if self.random_start:
                noise = self._init_noise
                if noise is None or tuple(noise.shape) != tuple(adv_images.shape) or noise.dtype != adv_images.dtype:
                    raise RuntimeError(f"controlled PGD start noise missing/mismatched: "
                                       f"{None if noise is None else tuple(noise.shape)} vs {tuple(adv_images.shape)}")
                # replaces: torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
                adv_images = adv_images + noise.to(adv_images.device)
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            for _ in range(self.steps):
                adv_images.requires_grad = True
                outputs = self.get_logits(adv_images)
                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)
                grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            return adv_images

    return ControlledStartPGD, p1a.sha256_array(np.frombuffer(stock_src.encode(), dtype=np.uint8))


# --------------------------------------------------------------------------- worker
def worker_main(args) -> None:
    import torch  # thread settings applied before any parallel work

    threads = args.worker_threads
    torch.set_num_interop_threads(INTEROP_THREADS)
    torch.set_num_threads(threads)
    intra, interop = torch.get_num_threads(), torch.get_num_interop_threads()
    if (intra, interop) != (threads, INTEROP_THREADS):
        p1a.abort(f"thread settings not applied: intra={intra} interop={interop}")

    import torch.nn as nn
    import src.adapters.attack_adapter as am
    from src.adapters.attack_adapter import AttackAdapter, _REAL_ATTACK_SOURCE
    from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE

    ctx = json.loads(Path(args.worker_context).read_text())
    out_dir = Path(args.out_dir)
    label = f"intra{threads}_interop{INTEROP_THREADS}"
    attacks: List[str] = ctx["attacks"]
    batch_sizes: List[int] = ctx["batch_sizes"]
    settings = ctx["attack_settings"]
    p1a.log(f"[{label}] pid={os.getpid()} attacks={attacks} batch_sizes={batch_sizes}")

    awn = AWNModelAdapter(checkpoint_path=str(REPO_ROOT / ctx["checkpoint_path"]), device="cpu")
    if awn.model is None or awn.backend_name != _REAL_MODEL_SOURCE or awn.status != "ok":
        p1a.abort(f"AWN backend is not real: backend={awn.backend_name} status={awn.status}")
    adapter = AttackAdapter(awn_model=awn.model, device="cpu")
    if adapter.wrapped_model is None or adapter.backend_name != _REAL_ATTACK_SOURCE or adapter.status != "ok":
        p1a.abort(f"attack backend is not real: backend={adapter.backend_name} status={adapter.status}")
    if am._torchattacks is None or awn.model.training:
        p1a.abort("torchattacks unavailable or model in train mode")
    wm = adapter.wrapped_model
    ControlledStartPGD, pgd_src_sha = make_controlled_pgd(am._torchattacks, torch, nn)

    subset = [tuple(k) for k in ctx["subset"]]
    formal_dir = REPO_ROOT / ctx["formal_dir"]
    base_rows, formal_attack = p1a.load_formal_rows(formal_dir, set(subset))
    inputs = p1a.rebuild_clean_inputs(subset, base_rows, ctx["sensing"], REPO_ROOT / ctx["dataset_path"])
    p1a.log(f"[{label}] rebuilt {len(inputs)} x_clean inputs; all sha256 == formal clean_input_sha256")

    def infer_pred(x) -> int:
        logits, meta = awn.infer(x)
        if meta["awn_backend"] != _REAL_MODEL_SOURCE or meta["awn_status"] != "ok" or not np.isfinite(logits).all():
            p1a.abort(f"AWN inference invalid: {meta}")
        return int(np.argmax(logits[0]))

    for key, inp in inputs.items():  # reused clean prediction (Phase 1B rule), untimed
        inp["clean_pred"] = infer_pred(inp["x"])
        if inp["clean_pred"] != inp["formal_clean_pred"]:
            p1a.abort(f"{key}: clean_pred {inp['clean_pred']} != formal {inp['formal_clean_pred']}")
        if p1b.in_path_clean_pred(am, adapter, torch, inp["x"]) != inp["clean_pred"]:
            p1a.abort(f"{key}: apply()-internal y_pred != pipeline clean_pred; reuse would change the label")

    req_grad_ref = [p.requires_grad for p in wm.parameters()]

    def check_state(where: str) -> None:
        if awn.model.training or wm.training:
            p1a.abort(f"model entered train mode ({where})")
        if (torch.get_num_threads(), torch.get_num_interop_threads()) != (intra, interop):
            p1a.abort(f"torch thread settings drifted ({where})")
        if [p.requires_grad for p in wm.parameters()] != req_grad_ref:
            p1a.abort(f"requires_grad flags not restored exactly ({where})")

    # ---------------------------------------------------------------- runtime batch-semantics audit
    audit: Dict[str, object] = {"pgd_forward_source_sha256": pgd_src_sha, "pgd_forward_matches_copied_logic": True}
    adv_attack_path = REPO_ROOT / "external" / "adversarial-rf" / "util" / "adv_attack.py"
    audit["adv_attack_py_sha256"] = p1a.sha256_file(adv_attack_path)
    audit["adv_attack_py_matches_pinned_audited_source"] = audit["adv_attack_py_sha256"] == PINNED_ADV_ATTACK_SHA256
    X_all = np.concatenate([inputs[k]["x"] for k in subset], axis=0)
    xa_b, a_b, b_b = am._iq_to_ta_input_minmax(torch.from_numpy(X_all))
    per = [am._iq_to_ta_input_minmax(torch.from_numpy(inputs[k]["x"])) for k in subset]
    audit["minmax_a_b_shape"] = [list(a_b.shape), list(b_b.shape)]
    audit["minmax_per_sample_shape_ok"] = tuple(a_b.shape) == (len(subset), 1, 1) and tuple(b_b.shape) == (len(subset), 1, 1)
    audit["minmax_batch_equals_per_sample_bitwise"] = bool(
        all(torch.equal(a_b[i:i + 1], per[i][1]) and torch.equal(b_b[i:i + 1], per[i][2])
            and torch.equal(xa_b[i:i + 1], per[i][0]) for i in range(len(subset))))
    back_b = am._ta_output_to_iq_minmax(xa_b, a_b, b_b)
    audit["denorm_batch_equals_per_sample_bitwise"] = bool(
        all(torch.equal(back_b[i:i + 1], am._ta_output_to_iq_minmax(per[i][0], per[i][1], per[i][2]))
            for i in range(len(subset))))
    audit["distinct_per_sample_b_values"] = int(torch.unique(b_b).numel())
    try:  # wrapper state with a batch: logits per sample vs B=1
        wm.set_minmax(a_b, b_b)
        with torch.no_grad():
            lb = wm(xa_b)
        wm.clear_minmax()
        l1 = []
        for i in range(len(subset)):
            wm.set_minmax(per[i][1], per[i][2])
            with torch.no_grad():
                l1.append(wm(per[i][0]))
            wm.clear_minmax()
        l1 = torch.cat(l1)
        audit["wrapper_batch_argmax_equals_per_sample"] = bool(torch.equal(lb.argmax(1), l1.argmax(1)))
        audit["wrapper_batch_logits_max_abs_diff"] = float((lb - l1).abs().max())
        audit["wrapper_batch_logits_bitwise_equal"] = bool(torch.equal(lb, l1))
    finally:
        wm.eval()
        wm.clear_minmax()
    check_state("audit")
    if not (audit["minmax_per_sample_shape_ok"] and audit["minmax_batch_equals_per_sample_bitwise"]
            and audit["denorm_batch_equals_per_sample_bitwise"] and audit["wrapper_batch_argmax_equals_per_sample"]):
        (out_dir / f"audit_{label}.json").write_text(json.dumps(audit, indent=2, default=str))
        p1a.abort(f"batch-semantics audit failed: {audit}")
    if not audit["adv_attack_py_matches_pinned_audited_source"]:
        p1a.log("[warn] adv_attack.py differs from the pinned file the audit was read from; "
                "runtime audit checks above still passed")
    p1a.log(f"[{label}] batch-semantics audit passed: {audit}")

    # ---------------------------------------------------------------- attack construction parameters
    pgd_hp = None
    if "pgd" in attacks:
        s = settings["pgd"]
        stock = am._build_torchattacks("pgd", am.TemperatureLogitsWrapper(wm, s["temperature"]), s["eps"],
                                       attack_params=dict(s["attack_params"]))
        pgd_hp = {"eps": stock.eps, "alpha": stock.alpha, "steps": stock.steps, "random_start": stock.random_start}
        audit["pgd_hyperparameters_from_stock_builder"] = pgd_hp
        noise = {}
        for key in subset:
            g = torch.Generator().manual_seed(inputs[key]["seed"])
            noise[key] = torch.empty((1, 2, inputs[key]["x"].shape[2], 1), dtype=torch.float32).uniform_(
                -pgd_hp["eps"], pgd_hp["eps"], generator=g)
        audit["pgd_noise_procedure"] = ("per sample: torch.empty([1,2,128,1]).uniform_(-eps, eps, "
                                        "generator=torch.Generator().manual_seed(formal_seed)); batch = torch.cat in sample order")

    def build_attack(attack, attack_model, keys):
        s = settings[attack]
        if attack != "pgd":
            return am._build_torchattacks(attack, attack_model, s["eps"], attack_params=dict(s["attack_params"]))
        atk = ControlledStartPGD(attack_model, eps=pgd_hp["eps"], alpha=pgd_hp["alpha"], steps=pgd_hp["steps"],
                                 random_start=pgd_hp["random_start"])
        atk.set_init_noise(noise[keys[0]] if len(keys) == 1 else torch.cat([noise[k] for k in keys], dim=0))
        return atk

    def attack_block(attack, keys):
        """Phase 1B optimized path (reuse clean pred, no freeze, no print) for len(keys) samples at once.
        Timed: stacking, numpy->torch, per-sample min/max, set_minmax, label tensor, wrapper + attack
        construction (prep); atk(...) (kernel); normalized L_inf, denormalization, numpy, state
        restoration, shape/dtype checks (post)."""
        s = settings[attack]
        t0 = time.perf_counter()
        x = inputs[keys[0]]["x"] if len(keys) == 1 else np.concatenate([inputs[k]["x"] for k in keys], axis=0)
        training_before = wm.training
        orig_requires_grad = [p.requires_grad for p in wm.parameters()]
        orig_param_devices = [p.device for p in wm.parameters()]
        try:
            x_ta, a, b = am._iq_to_ta_input_minmax(torch.from_numpy(x).to(adapter.device))
            wm.set_minmax(a, b)
            y = torch.tensor([inputs[k]["clean_pred"] for k in keys], dtype=torch.long)
            atk = build_attack(attack, am.TemperatureLogitsWrapper(wm, s["temperature"]), keys)
            t1 = time.perf_counter()
            x_ta_adv = atk(x_ta, y)
            t2 = time.perf_counter()
            lin = (x_ta_adv - x_ta).abs().amax(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.float32)
            x_adv = am._ta_output_to_iq_minmax(x_ta_adv, a, b).detach().cpu().numpy().astype(np.float32)
            b_np = b.detach().cpu().numpy().reshape(-1)
        finally:
            if training_before:
                print("[attack_adapter] warning: wrapped model was already in train mode before this call")
            wm.eval()
            wm.clear_minmax()
            for p, req_grad, dev in zip(wm.parameters(), orig_requires_grad, orig_param_devices):
                p.requires_grad_(req_grad)
                if p.device != dev:
                    p.data = p.data.to(dev)
        if x_adv.shape != x.shape or x_adv.dtype != x.dtype:
            raise RuntimeError(f"output shape/dtype {x_adv.shape} {x_adv.dtype} != input {x.shape} {x.dtype}")
        t3 = time.perf_counter()
        return x_adv, lin, b_np, {"total_ms": (t3 - t0) * 1e3, "prep_ms": (t1 - t0) * 1e3,
                                  "kernel_ms": (t2 - t1) * 1e3, "post_ms": (t3 - t2) * 1e3}

    # ---------------------------------------------------------------- B=1 anchors (validated Phase 1B path)
    anchors: Dict[str, Dict[tuple, dict]] = {}
    fidelity = {}
    for attack in attacks:
        anchors[attack] = {}
        n_eq = 0
        for key in subset:
            inp = inputs[key]
            torch.manual_seed(inp["seed"])
            ref, _, _, y_used, _, _ = p1b.attack_path(am, adapter, torch, inp["x"], attack, settings[attack],
                                                      inp["clean_pred"], False, False)
            check_state(f"anchor {attack} {key}")
            mine, _, _, _ = attack_block(attack, [key])
            check_state(f"anchor-own {attack} {key}")
            if y_used != inp["clean_pred"]:
                p1a.abort(f"anchor label mismatch {attack} {key}")
            if not np.array_equal(ref, mine):
                p1a.abort(f"{attack} {key}: benchmark B=1 path is not bit-identical to the validated Phase 1B path "
                          f"(max abs diff {float(np.max(np.abs(ref.astype(np.float64) - mine))):.3e})"
                          + ("; ControlledStartPGD does not reproduce stock PGD's random start" if attack == "pgd" else ""))
            n_eq += 1
            sha = p1a.sha256_array(ref)
            if attack in p1a.DETERMINISTIC_ATTACKS:
                fr = formal_attack[key + (attack, settings[attack]["profile"])]
                if sha != fr.get("adversarial_sha256"):
                    p1a.abort(f"{attack} {key}: anchor sha256 != formal adversarial_sha256")
            pred = infer_pred(ref)
            d_ref = ref - inp["x"]
            anchors[attack][key] = {"x_adv": ref, "sha": sha, "pred": pred,
                                    "success": bool(inp["clean_pred"] == inp["label"] and pred != inp["label"]),
                                    "linf": float(np.max(np.abs(d_ref))), "l2": float(np.linalg.norm(d_ref.reshape(-1)))}
        fidelity[attack] = f"{n_eq}/{len(subset)} bit-identical to Phase 1B path" + (
            " (stock PGD under torch.manual_seed(formal_seed) vs ControlledStartPGD)" if attack == "pgd" else
            "; anchor sha256 == formal adversarial_sha256 for all")
    audit["anchor_fidelity"] = fidelity
    p1a.log(f"[{label}] anchors built and validated: {fidelity}")

    # ---------------------------------------------------------------- benchmark
    raw_f = (out_dir / f"raw_{label}.csv").open("w", newline="")
    smp_f = (out_dir / f"per_sample_{label}.csv").open("w", newline="")
    raw_w = csv.DictWriter(raw_f, fieldnames=RAW_FIELDS)
    smp_w = csv.DictWriter(smp_f, fieldnames=SAMPLE_FIELDS)
    raw_w.writeheader()
    smp_w.writeheader()
    semantic_different: List[dict] = []   # recorded, cell FAILS, run continues
    invalid: List[dict] = []              # abort immediately
    class_counts = {c: 0 for c in CLASSES}

    def validate_outputs(attack, B, bi, rep, mode, keys, x_adv, lin, b_np) -> dict:
        """V2 rule: classify every sample against its independent B=1 anchor (bitwise AND semantic)."""
        agg = {c: 0 for c in CLASSES}
        agg.update({"maxdiff": 0.0, "max_linf_rel": 0.0, "max_l2_rel": 0.0, "n_pred": 0, "n_succ": 0,
                    "n_fin": 0, "n_eps": 0})
        eps = float(settings[attack]["eps"])
        shape_ok = x_adv.shape == (len(keys), 2, x_adv.shape[2]) and x_adv.dtype == np.float32
        for i, key in enumerate(keys):
            inp, anc = inputs[key], anchors[attack][key]
            xi = np.ascontiguousarray(x_adv[i:i + 1])
            fin = bool(np.isfinite(xi).all())
            bit = bool(np.array_equal(xi, anc["x_adv"]))
            md = float(np.max(np.abs(xi.astype(np.float64) - anc["x_adv"].astype(np.float64)))) if fin else float("inf")
            # identical bytes -> identical deterministic inference; otherwise infer (never skipped)
            pred = anc["pred"] if bit else (infer_pred(xi) if fin else -1)
            succ = bool(inp["clean_pred"] == inp["label"] and pred != inp["label"])
            diff = xi - inp["x"]
            linf, l2 = float(np.max(np.abs(diff))), float(np.linalg.norm(diff.reshape(-1)))
            bound = eps * float(b_np[i])
            norm_ok = bool(lin[i] <= eps + NORM_EPS_TOL)
            iq_ok = bool(linf <= bound * (1 + IQ_EPS_REL_TOL) + IQ_EPS_ABS_TOL)
            pred_eq, succ_eq = pred == anc["pred"], succ == anc["success"]
            linf_rel, l2_rel = _rel(abs(linf - anc["linf"]), anc["linf"]), _rel(abs(l2 - anc["l2"]), anc["l2"])
            if not (fin and shape_ok and norm_ok and iq_ok):
                cls = "INVALID"
            elif bit:
                cls = "BIT_IDENTICAL"
            elif pred_eq and succ_eq:
                cls = "BITWISE_DIFFERENT_SEMANTIC_SAME"
            else:
                cls = "SEMANTIC_DIFFERENT"
            agg[cls] += 1
            class_counts[cls] += 1
            agg["maxdiff"] = max(agg["maxdiff"], md)
            agg["max_linf_rel"] = max(agg["max_linf_rel"], linf_rel)
            agg["max_l2_rel"] = max(agg["max_l2_rel"], l2_rel)
            agg["n_pred"] += pred_eq
            agg["n_succ"] += succ_eq
            agg["n_fin"] += fin
            agg["n_eps"] += norm_ok and iq_ok
            row = {
                "condition": label, "intra_threads": intra, "attack": attack, "batch_size": B, "batch_index": bi,
                "position_in_batch": i, "batch_actual_size": len(keys), "repeat": rep, "mode": mode,
                "modulation": key[0], "snr_db": key[1], "sample_index": key[2], "formal_seed": inp["seed"],
                "label": inp["label"], "clean_pred": inp["clean_pred"],
                "attacked_pred": pred, "anchor_attacked_pred": anc["pred"], "attacked_pred_equal": pred_eq,
                "attack_success": succ, "anchor_attack_success": anc["success"], "attack_success_equal": succ_eq,
                "iq_linf": linf, "anchor_iq_linf": anc["linf"], "linf_abs_diff": abs(linf - anc["linf"]),
                "linf_rel_diff": linf_rel, "iq_l2": l2, "anchor_iq_l2": anc["l2"], "l2_abs_diff": abs(l2 - anc["l2"]),
                "l2_rel_diff": l2_rel, "iq_linf_normalized": float(lin[i]), "norm_eps_bound_ok": norm_ok,
                "per_sample_b": float(b_np[i]), "iq_linf_bound_eps_x_b": bound, "iq_eps_bound_ok": iq_ok,
                "x_adv_sha256": p1a.sha256_array(xi), "anchor_sha256": anc["sha"], "bit_identical_to_anchor": bit,
                "max_abs_diff_vs_anchor": md, "n_diff_elements": int(np.sum(xi != anc["x_adv"])),
                "finite": fin, "shape": "x".join(map(str, xi.shape)), "dtype": str(xi.dtype),
                "shape_dtype_ok": shape_ok, "classification": cls,
            }
            smp_w.writerow(row)
            if cls == "SEMANTIC_DIFFERENT":
                semantic_different.append({k_: row[k_] for k_ in (
                    "condition", "attack", "batch_size", "batch_index", "position_in_batch", "repeat", "mode",
                    "modulation", "snr_db", "sample_index", "label", "clean_pred", "anchor_attacked_pred",
                    "attacked_pred", "anchor_attack_success", "attack_success", "max_abs_diff_vs_anchor",
                    "linf_rel_diff", "l2_rel_diff")})
            elif cls == "INVALID":
                invalid.append({k_: row[k_] for k_ in (
                    "condition", "attack", "batch_size", "batch_index", "repeat", "mode", "modulation", "snr_db",
                    "sample_index", "finite", "shape_dtype_ok", "norm_eps_bound_ok", "iq_eps_bound_ok")})
        agg["shape_ok"] = shape_ok
        return agg

    for attack in attacks:
        for B in batch_sizes:
            batches = [subset[i:i + B] for i in range(0, len(subset), B)]  # final partial batch allowed
            for keys in batches[: min(args.warmup_batches, len(batches))]:  # warmup, untimed, unrecorded
                for k in keys:
                    attack_block(attack, [k])
                attack_block(attack, keys)
            check_state(f"warmup {attack} B={B}")
            for rep in range(args.repeats):
                for bi, keys in enumerate(batches):
                    temp = p1a.cpu_temp_c()
                    order = ("sequential", "batched") if (rep + bi) % 2 == 0 else ("batched", "sequential")
                    for pos, mode in enumerate(order):
                        if mode == "sequential":
                            outs, lins, bs, calls = [], [], [], []
                            t0 = time.perf_counter()
                            for k in keys:
                                xo, li, bb, tm = attack_block(attack, [k])
                                outs.append(xo)
                                lins.append(li)
                                bs.append(bb)
                                calls.append(tm)
                            total = (time.perf_counter() - t0) * 1e3
                            x_adv = np.concatenate(outs, axis=0)
                            lin, b_np = np.concatenate(lins), np.concatenate(bs)
                            tm = {"total_ms": total, "prep_ms": sum(c["prep_ms"] for c in calls),
                                  "kernel_ms": sum(c["kernel_ms"] for c in calls),
                                  "post_ms": sum(c["post_ms"] for c in calls)}
                            call_list = ";".join(f"{c['total_ms']:.6f}" for c in calls)
                        else:
                            x_adv, lin, b_np, tm = attack_block(attack, keys)
                            call_list = ""
                        check_state(f"{attack} B={B} batch={bi} {mode}")
                        agg = validate_outputs(attack, B, bi, rep, mode, keys, x_adv, lin, b_np)
                        raw_w.writerow({
                            "condition": label, "intra_threads": intra, "interop_threads": interop,
                            "attack": attack, "attack_profile": settings[attack]["profile"], "eps": settings[attack]["eps"],
                            "batch_size": B, "batch_index": bi, "batch_actual_size": len(keys),
                            "is_partial_batch": len(keys) != B, "repeat": rep, "mode": mode, "mode_order": pos,
                            "total_ms": tm["total_ms"], "per_sample_ms": tm["total_ms"] / len(keys),
                            "prep_ms": tm["prep_ms"], "kernel_ms": tm["kernel_ms"], "post_ms": tm["post_ms"],
                            "seq_call_ms_list": call_list,
                            "n_bit_identical": agg["BIT_IDENTICAL"],
                            "n_bitwise_different_semantic_same": agg["BITWISE_DIFFERENT_SEMANTIC_SAME"],
                            "n_semantic_different": agg["SEMANTIC_DIFFERENT"], "n_invalid": agg["INVALID"],
                            "max_abs_diff_vs_anchor": agg["maxdiff"], "max_linf_rel_diff": agg["max_linf_rel"],
                            "max_l2_rel_diff": agg["max_l2_rel"],
                            "n_pred_match_anchor": agg["n_pred"], "n_success_match_anchor": agg["n_succ"],
                            "n_finite": agg["n_fin"], "n_eps_bound_ok": agg["n_eps"], "shape_dtype_ok": agg["shape_ok"],
                            "sample_keys": "|".join(f"{k[0]}:{k[1]}:{k[2]}" for k in keys), "cpu_temp_c": temp,
                        })
                        if invalid:  # INVALID -> abort immediately, after persisting what we have
                            raw_f.flush()
                            smp_f.flush()
                            (out_dir / f"invalid_{label}.json").write_text(json.dumps(invalid, indent=2, default=str))
                            (out_dir / f"semantic_different_{label}.json").write_text(
                                json.dumps(semantic_different, indent=2, default=str))
                            p1a.abort(f"[{label}] INVALID output: {invalid[0]} (see invalid_{label}.json)")
                    raw_f.flush()
                    smp_f.flush()
            p1a.log(f"[{label}] {attack} B={B} done ({len(batches)} batches x {args.repeats} repeats), "
                    f"temp={p1a.cpu_temp_c()}, classes so far={class_counts}")
    raw_f.close()
    smp_f.close()
    (out_dir / f"semantic_different_{label}.json").write_text(json.dumps(semantic_different, indent=2, default=str))

    meta = {
        "label": label, "pid": os.getpid(), "intra_threads": intra, "interop_threads": interop,
        "env_thread_vars": {k: os.environ.get(k) for k in p1a.ENV_THREAD_VARS},
        "torch_version": torch.__version__, "torch_parallel_info": torch.__config__.parallel_info(),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "torchattacks_version": getattr(am._torchattacks, "__version__", None),
        "attack_adapter_source_sha256": p1a.sha256_file(Path(am.__file__)),
        "batch_semantics_audit": audit, "classification_counts": class_counts,
        "n_semantic_different": len(semantic_different), "n_invalid": len(invalid),
    }
    (out_dir / f"worker_{label}.json").write_text(json.dumps(meta, indent=2, default=str))
    p1a.log(f"[{label}] finished; classes={class_counts}")


# --------------------------------------------------------------------------- parent: estimate / aggregate
def estimate_minutes(threads: List[int], attacks: List[str], batch_sizes: List[int], repeats: int,
                     warmup_batches: int, n: int = 88) -> float:
    """Upper-bound estimate: assumes a batched call costs as much as B sequential B=1 calls."""
    total_s = 0.0
    for t in threads:
        total_s += EST_PROCESS_OVERHEAD_S
        for a in attacks:
            ms = EST_B1_MS[a] * EST_THREAD_FACTOR[t]
            warm = sum(min(warmup_batches, -(-n // B)) * B for B in batch_sizes)
            calls = repeats * len(batch_sizes) * n * 2 + warm * 2 + n * 2  # timed + warmup + anchors
            total_s += calls * ms / 1e3 + (repeats * len(batch_sizes) * n * 2 * 0.1 + n) * EST_INFER_MS / 1e3
    return total_s / 60.0


def _f(r, k):
    return float(r[k])


def _stat_cols(prefix: str, vals: List[float]) -> dict:
    st = p1a.stats(vals)
    return {f"{prefix}_{k}": st[k] for k in ("n", "mean", "median", "p95", "p99", "std")}


def aggregate(out_dir: Path, conditions: List[dict], attack_settings: dict, batch_sizes: List[int],
              attacks: List[str]):
    labels = [c["label"] for c in conditions]
    rows = []
    for lab in labels:
        with (out_dir / f"raw_{lab}.csv").open(newline="") as f:
            rows.extend(csv.DictReader(f))
    with (out_dir / "raw_measurements.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        w.writerows(rows)
    samples = []
    for lab in labels:
        with (out_dir / f"per_sample_{lab}.csv").open(newline="") as g:
            samples.extend(csv.DictReader(g))
    with (out_dir / "per_sample_validation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS)
        w.writeheader()
        w.writerows(samples)

    validation = {"rule": "V2 (bitwise AND semantic; see manifest.correctness_rule_v2)",
                  "per_condition_attack_batch": {}, "workers": {}, "semantic_different_samples": [],
                  "workers_flagged_for_review": [c["label"] for c in conditions if c.get("review_flag")],
                  "notes": []}
    summary = []
    for c in conditions:
        lab = c["label"]
        wmeta = json.loads((out_dir / f"worker_{lab}.json").read_text())
        validation["workers"][lab] = {**wmeta, "environment_before_after": {
            k: c.get(k) for k in ("temp_before_c", "temp_after_c", "throttled_before", "throttled_after",
                                  "governor_before", "governor_after", "review_flag", "review_reason")}}
        for attack in attacks:
            for B in batch_sizes:
                cr = [r for r in rows if r["condition"] == lab and r["attack"] == attack and int(r["batch_size"]) == B]
                if not cr:
                    continue
                seq = [r for r in cr if r["mode"] == "sequential"]
                bat = [r for r in cr if r["mode"] == "batched"]
                cs = [s for s in samples if s["condition"] == lab and s["attack"] == attack and int(s["batch_size"]) == B]
                cs_b = [s for s in cs if s["mode"] == "batched"]
                cs_s = [s for s in cs if s["mode"] == "sequential"]

                def counts(ss):
                    return {cl: sum(1 for s in ss if s["classification"] == cl) for cl in CLASSES}

                cb, cq = counts(cs_b), counts(cs_s)
                nb = len(cs_b)
                sem_diff_total = cb["SEMANTIC_DIFFERENT"] + cq["SEMANTIC_DIFFERENT"]
                invalid_total = cb["INVALID"] + cq["INVALID"]
                status = "PASS" if (sem_diff_total == 0 and invalid_total == 0 and nb > 0) else "FAIL"
                show = status == "PASS"

                def fmax(ss, k):
                    vals = [float(s[k]) for s in ss]
                    return max(vals) if vals else float("nan")

                cell_val = {
                    "status": status,
                    "batched": {"n": nb, **{cl: {"n": v, "pct": 100.0 * v / nb if nb else float("nan")}
                                             for cl, v in cb.items()}},
                    "sequential": {"n": len(cs_s), **{cl: {"n": v} for cl, v in cq.items()}},
                    "max_abs_x_adv_diff": fmax(cs, "max_abs_diff_vs_anchor"),
                    "max_linf_rel_diff": fmax(cs, "linf_rel_diff"),
                    "max_l2_rel_diff": fmax(cs, "l2_rel_diff"),
                }
                validation["per_condition_attack_batch"][f"{lab}/{attack}/B{B}"] = cell_val
                validation["semantic_different_samples"] += [
                    {k_: s[k_] for k_ in ("condition", "attack", "batch_size", "batch_index", "position_in_batch",
                                          "repeat", "mode", "modulation", "snr_db", "sample_index", "label",
                                          "clean_pred", "anchor_attacked_pred", "attacked_pred",
                                          "anchor_attack_success", "attack_success", "max_abs_diff_vs_anchor")}
                    for s in cs if s["classification"] == "SEMANTIC_DIFFERENT"]

                full_b = [r for r in bat if not p1a.parse_bool(r["is_partial_batch"])]
                n_samples = sum(int(r["batch_actual_size"]) for r in bat)
                calls = [float(v) for r in seq for v in r["seq_call_ms_list"].split(";") if v]
                sum_b = sum(_f(r, "total_ms") for r in bat)
                sum_s = sum(_f(r, "total_ms") for r in seq)
                paired: Dict[tuple, list] = {}
                for r in seq:
                    paired.setdefault((r["repeat"], r["batch_index"]), [None, None])[0] = _f(r, "total_ms")
                for r in bat:
                    paired.setdefault((r["repeat"], r["batch_index"]), [None, None])[1] = _f(r, "total_ms")
                ratios = [s_ / b_ for s_, b_ in paired.values() if s_ and b_]
                bt = p1a.stats([_f(r, "total_ms") for r in full_b])
                am_ = p1a.stats([_f(r, "per_sample_ms") for r in bat])
                b1 = p1a.stats(calls)

                rec = {
                    "condition": lab, "intra_threads": wmeta["intra_threads"], "interop_threads": wmeta["interop_threads"],
                    "attack": attack, "attack_profile": attack_settings[attack]["profile"], "batch_size": B,
                    "n_samples_per_repeat": n_samples // max(1, len({r["repeat"] for r in bat})),
                    "n_batches": len(bat), "n_full_batches": len(full_b),
                    "partial_batch_size": next((int(r["batch_actual_size"]) for r in bat
                                                if p1a.parse_bool(r["is_partial_batch"])), ""),
                    "validation_status": status,
                    "worker_review_flag": bool(c.get("review_flag")),
                    # ---- correctness (batched outputs; sequential counts alongside)
                    "n_batched_outputs": nb,
                    "n_bit_identical": cb["BIT_IDENTICAL"],
                    "pct_bit_identical": 100.0 * cb["BIT_IDENTICAL"] / nb if nb else float("nan"),
                    "n_bitwise_different_semantic_same": cb["BITWISE_DIFFERENT_SEMANTIC_SAME"],
                    "pct_bitwise_different_semantic_same":
                        100.0 * cb["BITWISE_DIFFERENT_SEMANTIC_SAME"] / nb if nb else float("nan"),
                    "n_semantic_different": cb["SEMANTIC_DIFFERENT"],
                    "pct_semantic_different": 100.0 * cb["SEMANTIC_DIFFERENT"] / nb if nb else float("nan"),
                    "n_invalid": cb["INVALID"],
                    "sequential_n_not_bit_identical": len(cs_s) - cq["BIT_IDENTICAL"],
                    "sequential_n_semantic_different": cq["SEMANTIC_DIFFERENT"],
                    "max_abs_x_adv_diff": cell_val["max_abs_x_adv_diff"],
                    "max_linf_rel_diff": cell_val["max_linf_rel_diff"],
                    "max_l2_rel_diff": cell_val["max_l2_rel_diff"],
                    # ---- timing (always recorded; claims gated by validation_status)
                    **_stat_cols("batch_completion_ms_fullbatches", [_f(r, "total_ms") for r in full_b]),
                    **_stat_cols("amortized_batch_ms_per_sample", [_f(r, "per_sample_ms") for r in bat]),
                    **_stat_cols("sequential_ms_per_sample", [_f(r, "per_sample_ms") for r in seq]),
                    **_stat_cols("b1_single_call_ms", calls),
                    **_stat_cols("sequential_total_ms_fullbatches",
                                 [_f(r, "total_ms") for r in seq if not p1a.parse_bool(r["is_partial_batch"])]),
                    **_stat_cols("batch_kernel_ms_fullbatches", [_f(r, "kernel_ms") for r in full_b]),
                    **_stat_cols("batch_kernel_ms_per_sample",
                                 [_f(r, "kernel_ms") / int(r["batch_actual_size"]) for r in bat]),
                    **_stat_cols("sequential_kernel_ms_per_sample",
                                 [_f(r, "kernel_ms") / int(r["batch_actual_size"]) for r in seq]),
                    "throughput_batch_samples_per_sec": (n_samples / sum_b * 1e3) if show and sum_b > 0 else "",
                    "throughput_sequential_samples_per_sec": (n_samples / sum_s * 1e3) if show and sum_s > 0 else "",
                    "speedup_batch_vs_sequential_total": (sum_s / sum_b) if show and sum_b > 0 else "",
                    "speedup_batch_vs_sequential_paired_median": float(np.median(ratios)) if show and ratios else "",
                    "speedup_batch_vs_sequential_paired_p5": float(np.percentile(ratios, 5)) if show and ratios else "",
                    # ---- three SEPARATE budget flags (never merged)
                    "flag_A_batch_completion_median_le_40ms": (bt["median"] <= BUDGET_MS) if bt["n"] else "",
                    "flag_B_amortized_median_per_sample_le_40ms": am_["median"] <= BUDGET_MS,
                    "flag_C_b1_single_call_median_le_40ms": b1["median"] <= BUDGET_MS,
                    "latency_note": "amortized per-sample time (flag B) is NOT a single-sample decision latency; a "
                                    "batched sample completes only when its whole batch completes (flag A), plus the "
                                    "time to fill the batch; flag C is the B=1 decision latency",
                }
                summary.append(rec)
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    cells = validation["per_condition_attack_batch"]
    totals = {cl: sum(v["batched"][cl]["n"] for v in cells.values()) for cl in CLASSES}
    n_tot = sum(v["batched"]["n"] for v in cells.values())
    validation["batched_classification_totals"] = {cl: {"n": v, "pct": 100.0 * v / n_tot if n_tot else float("nan")}
                                                   for cl, v in totals.items()}
    validation["n_cells"] = len(cells)
    validation["n_cells_fail"] = sum(1 for v in cells.values() if v["status"] != "PASS")
    validation["overall"] = "PASS" if cells and validation["n_cells_fail"] == 0 else "FAIL"
    if validation["workers_flagged_for_review"]:
        validation["overall"] += " (workers flagged for thermal/throttling review)"
    validation["notes"] += [
        "V2 rule: BIT_IDENTICAL and BITWISE_DIFFERENT_SEMANTIC_SAME are valid for timing; every bitwise "
        "difference is recorded per sample; SEMANTIC_DIFFERENT fails its cell (speedup/throughput withheld); "
        "INVALID aborts the worker.",
        "Semantic equality = attacked prediction and attack_success equal to the independent B=1 anchor, with "
        "normalized- and IQ-domain eps bounds, finite values, shape and dtype valid.",
        "PGD random_start is controlled per sample (manifest.pgd_random_start); ControlledStartPGD B=1 == stock PGD "
        "under torch.manual_seed(formal_seed) is re-proven in every worker before timing.",
        "Batch-completion and batch-kernel stats use full batches only; amortized, throughput and speedup use all "
        "batches including the final partial batch (no sample is duplicated).",
        "Flags A (batch completion), B (amortized per sample) and C (B=1 call) are separate; only A and C describe "
        "latency experienced by an individual sample.",
        "Observed speedups are PyTorch batched-execution effects; nothing here demonstrates NEON-specific causes.",
    ]
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2, default=str))
    return summary, validation, len(rows)


# --------------------------------------------------------------------------- parent
def parent_main(args) -> None:
    threads = [int(t) for t in args.threads.split(",")]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    attacks = [a.strip().lower() for a in args.attacks.split(",")]
    if any(t not in (1, 2, 4) for t in threads) or len(set(threads)) != len(threads):
        p1a.abort("--threads must be a subset of 1,2,4")
    if any(b < 1 for b in batch_sizes) or len(set(batch_sizes)) != len(batch_sizes):
        p1a.abort("--batch-sizes must be distinct positive integers")
    if any(a not in ATTACKS for a in attacks):
        p1a.abort(f"--attacks must be a subset of {ATTACKS}")
    est = estimate_minutes(threads, attacks, batch_sizes, args.repeats, args.warmup_batches)
    p1a.log(f"[parent] estimated upper-bound runtime: {est:.1f} min "
            f"(threads={threads} attacks={attacks} batch_sizes={batch_sizes} repeats={args.repeats})")
    if est > args.max_minutes and not args.allow_long:
        p1a.abort(f"estimated {est:.1f} min > --max-minutes {args.max_minutes}; run a staged sweep "
                  "(e.g. --threads 1 / --attacks pgd / --batch-sizes 1,8,16) or pass --allow-long")

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (REPO_ROOT / (args.out_dir or f"results/accel_phase1c_v2_{ts}")).resolve()
    for p in p1a.PROTECTED_RESULT_DIRS:
        prot = (REPO_ROOT / p).resolve()
        if out_dir == prot or prot in out_dir.parents or out_dir in prot.parents:
            p1a.abort(f"refusing to write inside/over formal results directory {p}")
    out_dir.mkdir(parents=True, exist_ok=False)

    ctx = p1a.load_formal_context(args)
    cfg, manifest, formal_dir = ctx["cfg"], ctx["manifest"], ctx["formal_dir"]
    subset = p1a.build_subset(cfg)
    base_rows, attack_rows = p1a.load_formal_rows(formal_dir, set(subset))
    attack_settings = p1a.resolve_attack_settings(cfg, attack_rows, args.attack_temperature, args.diagnostics)
    attack_settings = {a: attack_settings[a] for a in attacks}
    sensing = p1a.resolve_sensing(cfg)

    p1a.log("[parent] hashing dataset / checkpoint / config ...")
    hashes = {
        "dataset_sha256": p1a.sha256_file(REPO_ROOT / cfg["dataset_path"]),
        "checkpoint_sha256": p1a.sha256_file(REPO_ROOT / cfg["checkpoint_path"]),
        "config_sha256": p1a.sha256_file(ctx["cfg_path"]),
        "formal_manifest_sha256": p1a.sha256_file(formal_dir / "manifest.json"),
        "formal_base_results_sha256": p1a.sha256_file(formal_dir / "base_results.csv"),
        "formal_attack_results_sha256": p1a.sha256_file(formal_dir / "attack_results.csv"),
        "attack_adapter_py_sha256": p1a.sha256_file(REPO_ROOT / "src/adapters/attack_adapter.py"),
        "adv_attack_py_sha256": p1a.sha256_file(REPO_ROOT / "external/adversarial-rf/util/adv_attack.py"),
        "adv_attack_py_pinned_audited_sha256": PINNED_ADV_ATTACK_SHA256,
        "phase1a_helpers_sha256": p1a.sha256_file(Path(p1a.__file__).resolve()),
        "phase1b_helpers_sha256": p1a.sha256_file(Path(p1b.__file__).resolve()),
        "this_script_sha256": p1a.sha256_file(Path(__file__).resolve()),
    }
    for name, fname in (("phase1c_v1_script_sha256", "bench_attack_batch_pi5.py"),
                        ("bim_diagnostic_script_sha256", "diagnose_bim_batch_divergence_pi5.py"),
                        ("bim_semantic_audit_script_sha256", "audit_bim_batch_semantics_pi5.py")):
        pth = _EXPERIMENTS_DIR / fname
        hashes[name] = p1a.sha256_file(pth) if pth.exists() else None
    if hashes["dataset_sha256"] != manifest.get("dataset_sha256"):
        p1a.abort("dataset sha256 differs from formal CPU manifest")
    if hashes["checkpoint_sha256"] != manifest.get("checkpoint_sha256"):
        p1a.abort("checkpoint sha256 differs from formal CPU manifest")

    env_meta = p1a.environment_metadata()
    if any(env_meta["env_thread_vars"].values()):
        p1a.log(f"[warn] thread env vars set in this shell: {env_meta['env_thread_vars']} (inherited unchanged, recorded)")

    worker_ctx = {
        "subset": [list(k) for k in subset], "formal_dir": p1a.rel(formal_dir),
        "dataset_path": cfg["dataset_path"], "checkpoint_path": cfg["checkpoint_path"],
        "sensing": sensing, "attack_settings": attack_settings, "attacks": attacks, "batch_sizes": batch_sizes,
    }
    ctx_file = out_dir / "worker_context.json"
    ctx_file.write_text(json.dumps(worker_ctx, indent=2, default=str))
    conditions = [{"label": f"intra{t}_interop{INTEROP_THREADS}", "threads": t} for t in threads]
    manifest_out = {
        "phase": "accel_phase1c_batching_v2", "status": "running",
        "correctness_rule_v2": {
            "BIT_IDENTICAL": "x_adv bytes/sha256 equal to the independent B=1 anchor -> valid",
            "BITWISE_DIFFERENT_SEMANTIC_SAME": "valid output, attacked prediction and attack_success equal to the "
                                               "anchor, eps bounds + finite/shape/dtype valid -> valid for timing, "
                                               "numerical difference recorded",
            "SEMANTIC_DIFFERENT": "attacked prediction or attack_success differs -> cell FAIL, speedup withheld",
            "INVALID": "NaN/Inf, wrong shape/dtype, normalized or IQ eps-bound violation, backend fallback, "
                       "model/thread/state corruption -> worker aborts immediately",
            "tolerances": {"normalized_eps": NORM_EPS_TOL, "iq_eps_rel": IQ_EPS_REL_TOL, "iq_eps_abs": IQ_EPS_ABS_TOL},
            "basis": "BIM semantic audit (88 x B in {1..40}, intra1/interop1): 697/704 bit-identical, 7/704 "
                     "BITWISE_DIFFERENT_SEMANTIC_SAME (PAM4/6 dB/sample 1, every B>1, max |dx| 0.0010695941746234894), "
                     "0 SEMANTIC_DIFFERENT, 0 INVALID; iteration diagnostic attributes it to batch-shape-dependent "
                     "floating point in the AWN forward/backward path",
            "unchanged_fail_closed": ["clean_input_sha256", "FGSM/BIM B=1 anchor == formal adversarial_sha256",
                                      "benchmark B=1 path == Phase 1B path (incl. ControlledStartPGD vs stock PGD)",
                                      "installed PGD.forward == copied logic", "batch min/max audit"],
        },
        "budget_flags": {"A": "batch completion median (full batches) <= 40 ms",
                         "B": "amortized batch median per sample <= 40 ms (NOT a decision latency)",
                         "C": "B=1 single-call median <= 40 ms"},
        "question": "Does batched attack generation improve throughput / amortized per-sample latency on the "
                    "Pi 5 without changing attack semantics?",
        "command": [sys.executable] + sys.argv, "out_dir": p1a.rel(out_dir), "formal_cpu_dir": p1a.rel(formal_dir),
        "config_path": args.config, "dataset_path": cfg["dataset_path"], "checkpoint_path": cfg["checkpoint_path"],
        "hashes": hashes, "environment": env_meta,
        "conditions": conditions, "interop_threads": INTEROP_THREADS, "batch_sizes": batch_sizes, "attacks": attack_settings,
        "repeats": args.repeats, "warmup_batches_per_attack_batchsize": args.warmup_batches,
        "estimated_upper_bound_minutes": est,
        "subset": {"modulations": cfg["modulations"], "snrs": p1a.SUBSET_SNRS,
                   "sample_indices": p1a.SUBSET_SAMPLE_INDICES, "n_source_samples": len(subset),
                   "order": "modulation-major, then SNR, then sample_index (formal config order); batches are "
                            "consecutive chunks of this order; the final batch may be partial; no duplication",
                   "samples": [{"modulation": k[0], "snr_db": k[1], "sample_index": k[2],
                                "formal_seed": int(float(base_rows[k]["seed"])),
                                "clean_input_sha256": base_rows[k]["clean_input_sha256"]} for k in subset]},
        "sensing_parameters": sensing,
        "path": "Phase 1B optimized path: clean prediction reused (outside timing), parameters not frozen, "
                "no print, no attack-object reuse; sequential mode = B independent B=1 calls of the same code",
        "pgd_random_start": "controlled: noise_i = torch.empty([1,2,128,1]).uniform_(-eps, eps, "
                            "generator=torch.Generator().manual_seed(formal_seed_i)); batched start = torch.cat of "
                            "the same per-sample tensors; ControlledStartPGD replaces only the uniform_ draw",
        "timing": {
            "clock": "time.perf_counter",
            "batched.total_ms": "one call: np.concatenate + numpy->torch + per-sample min/max + set_minmax + label "
                                "tensor + wrapper/attack construction + atk(...) + normalized L_inf + "
                                "denormalization + numpy + state restoration + shape/dtype checks",
            "sequential.total_ms": "wall time of B back-to-back B=1 calls of the same code on the same samples",
            "prep_ms / kernel_ms / post_ms": "kernel = atk(...) only; sequential values are sums over its B calls",
            "excluded": "sensing reconstruction, clean inference (reused), PGD noise generation (pre-generated for "
                        "both modes), attacked inference, hashing, validation, CSV writes",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2, default=str))

    for c in conditions:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--out-dir", str(out_dir),
               "--worker-context", str(ctx_file), "--worker-threads", str(c["threads"]),
               "--repeats", str(args.repeats), "--warmup-batches", str(args.warmup_batches)]
        c["intra_threads"], c["interop_threads"] = c["threads"], INTEROP_THREADS
        c["temp_before_c"] = p1a.cpu_temp_c()
        c["throttled_before"] = p1a.run_cmd(["vcgencmd", "get_throttled"])
        c["governor_before"] = _governors()
        c["started_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        p1a.log(f"[parent] starting fresh process for {c['label']} (temp={c['temp_before_c']}, "
                f"throttled={c['throttled_before']})")
        with (out_dir / f"worker_{c['label']}.stdout.log").open("w") as lf:
            rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=lf, env=os.environ.copy()).returncode
        c.update(returncode=rc, temp_after_c=p1a.cpu_temp_c(),
                 throttled_after=p1a.run_cmd(["vcgencmd", "get_throttled"]), governor_after=_governors(),
                 finished_utc=_dt.datetime.now(_dt.timezone.utc).isoformat())
        reasons = []
        for when in ("before", "after"):
            tv = _throttled_value(c[f"throttled_{when}"])
            if tv is None:
                reasons.append(f"get_throttled {when} unavailable/unparseable ({c[f'throttled_{when}']!r})")
            elif tv != 0:
                reasons.append(f"get_throttled {when} = {hex(tv)} (non-zero)")
        if c["governor_before"] != c["governor_after"]:
            reasons.append("scaling governor changed during the worker")
        c["review_flag"], c["review_reason"] = bool(reasons), reasons
        if reasons:
            p1a.log(f"[warn] {c['label']} flagged for review: {reasons}")
        (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2, default=str))
        if rc != 0:
            inv = out_dir / f"invalid_{c['label']}.json"
            sd = out_dir / f"semantic_different_{c['label']}.json"
            (out_dir / "validation.json").write_text(json.dumps({
                "overall": f"FAIL (worker {c['label']} aborted, rc={rc})",
                "invalid": json.loads(inv.read_text()) if inv.exists() else None,
                "semantic_different_so_far": json.loads(sd.read_text()) if sd.exists() else None,
                "note": "no throughput/speedup is reported for an aborted run; INVALID outputs, backend fallback, "
                        "state corruption or a failed fail-closed precheck abort the worker; see the worker stdout "
                        "log and stderr",
            }, indent=2, default=str))
            manifest_out.update(status="aborted", failed_condition=c["label"])
            (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2, default=str))
            p1a.abort(f"worker {c['label']} exited with {rc}")
        if args.cooldown > 0:
            time.sleep(args.cooldown)

    summary, validation, n = aggregate(out_dir, conditions, attack_settings, batch_sizes, attacks)
    manifest_out.update(status="complete", raw_rows=n, validation_overall=validation["overall"],
                        finished_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
                        environment_end={"cpu_temp_c": p1a.cpu_temp_c(),
                                         "vcgencmd_throttled": p1a.run_cmd(["vcgencmd", "get_throttled"])})
    (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2, default=str))
    p1a.log(f"[parent] done: {out_dir}  validation={validation['overall']}")
    for r in summary:
        p1a.log(f"  {r['condition']} {r['attack']:4s} B={r['batch_size']:>2}  "
                f"B1 call med={r['b1_single_call_ms_median']:7.2f}  "
                f"batch completion med={r['batch_completion_ms_fullbatches_median']:8.2f}  "
                f"amortized med={r['amortized_batch_ms_per_sample_median']:7.2f}  "
                f"speedup={r['speedup_batch_vs_sequential_total'] or '-'}  "
                f"bit={r['n_bit_identical']}/{r['n_batched_outputs']} "
                f"sem_same={r['n_bitwise_different_semantic_same']} sem_diff={r['n_semantic_different']}  "
                f"{r['validation_status']}")
    if validation["overall"] != "PASS":
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=p1a.DEFAULT_CONFIG)
    ap.add_argument("--formal-dir", default=p1a.DEFAULT_FORMAL_CPU_DIR)
    ap.add_argument("--out-dir", default=None, help="default results/accel_phase1c_v2_<timestamp> (must not exist)")
    ap.add_argument("--threads", default=DEFAULT_THREADS, help="intra-op thread conditions, run order (subset of 1,2,4)")
    ap.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES)
    ap.add_argument("--attacks", default=",".join(ATTACKS), help="subset of fgsm,pgd,bim")
    ap.add_argument("--repeats", type=int, default=3, help="passes over the 88-sample subset per (attack, B)")
    ap.add_argument("--warmup-batches", type=int, default=2, help="unrecorded warmup batches per (attack, B), both modes")
    ap.add_argument("--max-minutes", type=float, default=60.0, help="refuse to start if the estimate exceeds this")
    ap.add_argument("--allow-long", action="store_true", help="start even if the estimate exceeds --max-minutes")
    ap.add_argument("--cooldown", type=float, default=0.0, help="seconds to idle between thread-condition processes")
    ap.add_argument("--attack-temperature", type=float, default=None)
    ap.add_argument("--diagnostics", choices=["auto", "on", "off"], default="auto",
                    help="only affects the untimed Phase 1B anchor call; the batched path never runs diagnostics")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-context", help=argparse.SUPPRESS)
    ap.add_argument("--worker-threads", type=int, default=1, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.repeats < 1 or args.warmup_batches < 0:
        p1a.abort("--repeats>=1 and --warmup-batches>=0 required")
    if args.worker:
        worker_main(args)
    else:
        parent_main(args)


if __name__ == "__main__":
    main()
