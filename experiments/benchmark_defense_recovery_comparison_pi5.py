"""
Defense recovery comparison on the SAME attacked samples (Raspberry Pi 5, CPU evaluation backend).
Standalone; the formal pipeline and every formal result directory are untouched.

Defenses (one registry, one runner)
-----------------------------------
  none                            no defense
  fixed_topk_k{10,20,30,40}       external/adversarial-rf/util/defense.py:fft_topk_denoise
  adaptive_k_v2_snr               external/adversarial-rf/util/defense.py:adaptive_k_v2_snr_defense
  signal_filter_<name>            external/adversarial-rf/util/defense_registry.py DEFENSE_REGISTRY entries
                                  kalman | wiener | savitzky_golay | gaussian | fir (util/defense_baselines.py),
                                  applied through the registry's own _apply_filter with its default parameters
  adversarial_training_transfer_<provenance>
                                  classifier replaced by an adversarially trained AWN checkpoint
                                  (--at-checkpoint, required; same AWN architecture), evaluated on the stored
                                  adversarial IQ crafted against the standard AWN. <provenance> is
                                  leakage_free only for a complete experiments/train_adversarial_awn.py run
                                  (formal evaluation samples excluded); every other checkpoint, including
                                  external/adversarial-rf/awn_fpga/2016.10a_AWN_at.pkl, is exploratory
  adversarial_training_whitebox_<provenance>
                                  optional (--at-whitebox): attacks regenerated against the AT checkpoint
                                  itself -- different adversarial IQ, its own denominators, separate table
No implementation is re-written here. A defense whose code or checkpoint is unavailable is reported as
UNAVAILABLE with the reason; no numbers are produced for it and nothing falls back to the standard AWN.
EXTRA_SIGNAL_FILTERS below is the registration point for a further signal-filter definition.

Samples and attacks
-------------------
Formal grid (configs/hailo_full_matrix.json == formal CPU manifest config): 11 modulations x 20 SNRs x 10
indices = 2,200 clean samples, 24 attack profiles / 17 attacks read from the config, formal per-row seeds.
Clean and adversarial IQ come from the Phase B store layout of
experiments/benchmark_adaptive_k_effectiveness_pi5.py (functions imported, not copied): generated once with
the formal-equivalent attack call (torch.manual_seed(formal_seed); default AttackAdapter.apply), hash-checked
against the formal adversarial_sha256 for the bit-reproducible profiles. --adv-store-dir reuses an existing
Phase B store read-only, i.e. exactly the attacked samples Phase B evaluated. Every defense is evaluated on
the same stored IQ, so all comparisons are paired.

Metrics (project definitions, experiments/benchmark_adaptive_k_effectiveness_pi5.py:defense_metrics /
reports/meeting_20260921/analysis/phase_b_analysis.py; standard AWN defines clean_correct / attack_success)
  clean_correct  = clean_pred == label           attacked_correct = attacked_pred == label
  attack_success = clean_correct & ~attacked_correct
  defended_correct = defended_pred == label       recovered = attack_success & defended_correct
  recovery rate  = sum(recovered) / sum(attack_success)
  clean_defended_correct = defense applied to the clean input is correct
  clean degradation = sum(clean_correct & ~clean_defended_correct) / sum(clean_correct)
  retention = sum(clean_correct & defended_correct) / sum(clean_correct)
  net gain  = retention - retention_nodef,  retention_nodef = sum(clean_correct & ~attack_success)/sum(clean_correct)
Denominator: identical for every defense (same samples, same standard-AWN attack_success). The repository
defines no separate shared-success denominator, so none is produced. adversarial_training_whitebox uses the
AT model's own clean_correct / attack_success and is written to at_whitebox_summary.csv only.

AT provenance check: when the checkpoint has a sibling <stem>.config.json written by
external/adversarial-rf/adv_train.py, the AT train/validation split (Load_Dataset order, snr_min,
default_rng(seed).permutation, val_ratio 0.15) is reproduced and every evaluated sample is labelled
in_at_train; AT metrics are additionally reported split by that flag (at_train_overlap.csv).

Run on the Pi from the repo root:
    .venv/bin/python experiments/benchmark_defense_recovery_comparison_pi5.py --smoke --at-checkpoint <ckpt>
    nohup .venv/bin/python experiments/benchmark_defense_recovery_comparison_pi5.py --full \
        --adv-store-dir <phase B dir> --at-checkpoint <ckpt> > defense_recovery_full.log 2>&1 &
    resume: same command with --resume <out dir>
    AT only (no Top-K / Adaptive-K / signal-filter computation; "none" is kept as the zero-cost reference):
        --defenses adversarial_training  (names or families: no_defense, fixed_topk, adaptive_k, signal_filter,
        adversarial_training; combinable with --smoke / --full / --resume / --adv-store-dir / --at-whitebox)
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

import numpy as np

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import benchmark_adaptive_k_effectiveness_pi5 as akb  # noqa: E402 -- formal context, stores, attack call, metrics

p1a = akb.p1a
REPO_ROOT = akb.REPO_ROOT
T_LEN = akb.T_LEN
TOPK_VALUES = [10, 20, 30, 40]
SIGNAL_FILTERS = ["kalman", "wiener", "savitzky_golay", "gaussian", "fir"]
# Registration point: name -> callable(torch.Tensor[N,2,T]) -> torch.Tensor[N,2,T] for a further, formally
# defined signal filter. Entries are evaluated as signal_filter_<name> exactly like the registry filters.
EXTRA_SIGNAL_FILTERS: Dict[str, Callable] = {}
SMOKE_KEYS = [("QPSK", 18, 0), ("BPSK", 6, 1), ("QAM16", 18, 2)]
EVAL_BACKENDS = ("cpu",)
DEFENSE_FAMILIES = ("no_defense", "fixed_topk", "adaptive_k", "signal_filter", "adversarial_training")

ROW_FIELDS = ["global_index", "attack", "attack_profile", "profile", "modulation", "snr_label_db", "sample_index",
              "label", "defense", "defense_family", "clean_pred", "attacked_pred", "defended_pred",
              "clean_defended_pred", "clean_correct", "attacked_correct", "attack_success", "defended_correct",
              "recovered", "retained", "retained_nodef", "clean_defended_correct", "clean_degraded", "in_at_train",
              "defense_ms"]
CLEAN_FIELDS = ["global_index", "modulation", "snr_label_db", "sample_index", "label", "defense", "defense_family",
                "clean_pred", "clean_defended_pred", "clean_correct", "clean_defended_correct", "clean_degraded",
                "clean_improved", "in_at_train", "defense_ms"]
METRIC_KEYS = ["n", "n_clean_correct", "n_attacked_correct", "n_attack_success", "n_recovered", "n_defended_correct",
               "n_retained", "n_retained_nodef", "n_clean_defended_correct", "n_clean_degraded", "clean_acc_pct",
               "attacked_acc_pct", "cond_asr_pct", "recovery_pct", "recovery_ci95_lo_pct", "recovery_ci95_hi_pct",
               "defended_acc_pct", "retention_pct", "retention_nodef_pct", "retention_gain_pp",
               "clean_defended_acc_pct", "clean_degradation_pct"]
b2s = akb.b2s


class DefenseOutputError(RuntimeError):
    pass


# =========================================================================== defenses
def build_defenses(c, args) -> tuple:
    """Returns (defenses, unavailable). Each defense: name, family, kind ('transform'|'model'), fn / model."""
    torch, D = c.torch, c.D
    defs: List[dict] = [{"name": "none", "family": "no_defense", "kind": "transform", "fn": None,
                         "source": "identity"}]
    for k in TOPK_VALUES:
        defs.append({"name": f"fixed_topk_k{k}", "family": "fixed_topk", "kind": "transform",
                     "fn": (lambda xt, k=k: D.fft_topk_denoise(xt, topk=k)),
                     "source": "external/adversarial-rf/util/defense.py:fft_topk_denoise", "param": {"topk": k}})
    defs.append({"name": "adaptive_k_v2_snr", "family": "adaptive_k", "kind": "transform",
                 "fn": D.adaptive_k_v2_snr_defense,
                 "source": "external/adversarial-rf/util/defense.py:adaptive_k_v2_snr_defense"})
    unavailable: Dict[str, str] = {}
    try:
        import util.defense_registry as R  # noqa: E402 -- same sys.path entry as util.defense
        for name in SIGNAL_FILTERS:
            fn = R.DEFENSE_REGISTRY.get(name)
            if fn is None:
                unavailable[f"signal_filter_{name}"] = "DEFENSE_REGISTRY entry is None (util.defense_baselines import failed)"
                continue
            defs.append({"name": f"signal_filter_{name}", "family": "signal_filter", "kind": "transform",
                         "fn": (lambda xt, fn=fn, name=name: R._apply_filter(fn, xt, name, SimpleNamespace())),
                         "source": f"external/adversarial-rf/util/defense_registry.py:DEFENSE_REGISTRY['{name}'] "
                                   "via _apply_filter (registry default parameters)"})
    except Exception as exc:  # noqa: BLE001 - e.g. scipy missing
        for name in SIGNAL_FILTERS:
            unavailable[f"signal_filter_{name}"] = f"util.defense_registry import failed: {type(exc).__name__}: {exc}"
    for name, fn in EXTRA_SIGNAL_FILTERS.items():
        defs.append({"name": f"signal_filter_{name}", "family": "signal_filter", "kind": "transform", "fn": fn,
                     "source": "EXTRA_SIGNAL_FILTERS registration"})
    sel = parse_defense_selection(args.defenses, defs, unavailable)
    if sel is not None:
        # "none" is always kept: it needs no computation (stored attacked prediction) and is the reference.
        defs = [d for d in defs if d["name"] == "none" or d["name"] in sel or d["family"] in sel]
        unavailable = {n: why for n, why in unavailable.items() if n in sel or "signal_filter" in sel}
    at_selected = sel is None or "adversarial_training" in sel or any(s.startswith("adversarial_training_") for s in sel)
    at = load_at_model(args) if at_selected else {
        "status": "not_selected", "reason": "adversarial_training not in --defenses"}
    if at["status"] == "ok":
        at["provenance"], at["provenance_reason"] = at_provenance(at, c.cfg)
        defs.append({"name": f"adversarial_training_transfer_{at['provenance']}", "family": "adversarial_training",
                     "kind": "model", "model": at["adapter"],
                     "source": f"{at['path']} [{at['provenance']}: {at['provenance_reason']}]"})
    elif at_selected:
        unavailable["adversarial_training_transfer"] = at["reason"]
        if args.at_whitebox:
            unavailable["adversarial_training_whitebox"] = at["reason"]
    if sel is not None and not any(d["name"] != "none" for d in defs):
        p1a.abort(f"--defenses {args.defenses!r} leaves nothing to evaluate (unavailable: {unavailable})")
    return defs, unavailable, at


def parse_defense_selection(arg: Optional[str], defs: List[dict], unavailable: Dict[str, str]) -> Optional[set]:
    """--defenses accepts defense names and/or family names (DEFENSE_FAMILIES); None selects everything."""
    if not arg:
        return None
    sel = {s.strip() for s in arg.split(",") if s.strip()}
    valid = ({d["name"] for d in defs} | set(unavailable) | set(DEFENSE_FAMILIES)
             | {"adversarial_training_transfer", "adversarial_training_whitebox"})
    unknown = sorted(s for s in sel if s not in valid and not s.startswith("adversarial_training_"))
    if unknown:
        p1a.abort(f"--defenses: unknown {unknown}; valid names/families: {sorted(valid)}")
    return sel


def at_provenance(at: dict, cfg: dict) -> tuple:
    """leakage_free only for a complete experiments/train_adversarial_awn.py run that excluded every formal
    evaluation sample index; everything else (incl. external/adversarial-rf/awn_fpga/2016.10a_AWN_at.pkl, whose
    adv_train.py split covers formal evaluation rows) is exploratory."""
    cj = at.get("config_json") or {}
    if cj.get("split_method") == "train_adversarial_awn":
        excluded = set(cj.get("excluded_sample_indices") or [])
        if not cj.get("formal_eval_samples_excluded") or not set(cfg["sample_indices"]) <= excluded:
            return "exploratory", "train_adversarial_awn run did not exclude all formal sample indices"
        if cj.get("functional_check_only"):
            return "exploratory", "train_adversarial_awn functional check (--max-batches-per-epoch), not a full run"
        return "leakage_free", "train_adversarial_awn: formal evaluation samples excluded from train and validation"
    if str(cj.get("snr_filter", "")).startswith("snr_min="):
        return "exploratory", ("adv_train.py random split over SNR >= snr_min rows; formal evaluation samples can be "
                               "training samples (see in_at_train / at_train_overlap.csv)")
    return "exploratory", "no training-split record next to the checkpoint; overlap with evaluation samples unknown"


def load_at_model(args) -> dict:
    if not args.at_checkpoint:
        return {"status": "unavailable", "reason": "--at-checkpoint not supplied (no adversarially trained model)"}
    path = Path(args.at_checkpoint)
    path = path if path.is_absolute() else (REPO_ROOT / path)
    if not path.is_file():
        return {"status": "unavailable", "reason": f"AT checkpoint not found: {path}"}
    from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE
    ad = AWNModelAdapter(checkpoint_path=str(path), device="cpu")
    if ad.model is None or ad.status != "ok" or ad.backend_name != _REAL_MODEL_SOURCE:
        # AWNModelAdapter falls back to a dummy on load failure; that must never stand in for the AT model.
        return {"status": "unavailable", "reason": f"AT checkpoint did not load as the AWN architecture: {ad.notes}"}
    cfg_json = path.with_name(path.stem + ".config.json")
    return {"status": "ok", "adapter": ad, "path": p1a.rel(path), "sha256": akb.sha_file(path),
            "config_json": json.loads(cfg_json.read_text()) if cfg_json.exists() else None,
            "config_json_path": p1a.rel(cfg_json) if cfg_json.exists() else None}


def at_train_membership(c, at: dict, keys: List[tuple]) -> Optional[Dict[tuple, bool]]:
    """Reproduces external/adversarial-rf/adv_train.py:build_loaders (Load_Dataset order: sorted mods x sorted
    SNRs >= snr_min x 1000 rows; default_rng(seed).permutation; first n_train are training)."""
    cj = at.get("config_json") if at else None
    if cj and cj.get("split_method") == "train_adversarial_awn":
        # experiments/train_adversarial_awn.py: excluded sample indices never enter train or validation.
        excluded = set(cj.get("excluded_sample_indices") or [])
        if not cj.get("formal_eval_samples_excluded") or not {k[2] for k in keys} <= excluded:
            return None
        return {k: False for k in keys}
    if not cj or not str(cj.get("snr_filter", "")).startswith("snr_min="):
        return None
    snr_min = int(str(cj["snr_filter"]).split("=", 1)[1])
    seed = int(cj.get("seed", 42))
    val_ratio = 0.15  # hard-coded in adv_train.py main() -> build_loaders(val_ratio=0.15)
    from src.sensing.radioml_source import load_radioml_dict
    data = load_radioml_dict(str(REPO_ROOT / c.cfg["dataset_path"]))
    mods = sorted({m for m, _ in data})
    snrs = sorted({s for _, s in data if s >= snr_min})
    offset, pos = 0, {}
    for m in mods:
        for s in snrs:
            pos[(m, s)] = offset
            offset += data[(m, s)].shape[0]
    perm = np.random.default_rng(seed).permutation(offset)
    n_train = offset - int(offset * val_ratio)
    train = set(perm[:n_train].tolist())
    return {k: ((k[0], k[1]) in pos and (pos[(k[0], k[1])] + k[2]) in train) for k in keys}


def run_transform(d: dict, c, x: np.ndarray) -> tuple:
    if d["fn"] is None:
        return x, 0.0
    t0 = time.perf_counter()
    y = d["fn"](c.torch.from_numpy(np.ascontiguousarray(x)))
    ms = (time.perf_counter() - t0) * 1e3
    y = y.detach().cpu().numpy().astype(np.float32) if hasattr(y, "detach") else np.asarray(y, dtype=np.float32)
    if y.shape != x.shape or not np.isfinite(y).all():
        raise DefenseOutputError(f"{d['name']}: output shape {y.shape} / finite={bool(np.isfinite(y).all())}")
    return y, ms


def predict(c, d: Optional[dict], x: np.ndarray) -> int:
    if d is None or d["kind"] == "transform":
        with contextlib.redirect_stdout(io.StringIO()):
            return akb.infer_pred(c, x)[0]
    with contextlib.redirect_stdout(io.StringIO()):
        logits, meta = d["model"].infer(x)
    if meta["awn_status"] != "ok" or not np.isfinite(logits).all():
        p1a.abort(f"AT model inference invalid: {meta}")
    return int(np.argmax(logits[0]))


def defended(c, d: dict, x: np.ndarray) -> tuple:
    """(prediction, defense_ms) of defense d on input x (transform -> standard AWN, or model replacement)."""
    if d["kind"] == "model":
        return predict(c, d, x), 0.0
    y, ms = run_transform(d, c, x)
    return predict(c, None, y), ms


# =========================================================================== evaluation
def clean_rows(c, defs, clean: np.ndarray, cmeta: List[dict], idx: List[int], membership, bad: dict) -> List[dict]:
    rows = []
    for gi in idx:
        cm, x = cmeta[gi], clean[gi:gi + 1]
        label, cp = int(cm["label"]), int(cm["formal_clean_pred"])
        for d in defs:
            if d["name"] in bad:
                continue
            try:
                dp, ms = (cp, 0.0) if d["name"] == "none" else defended(c, d, x)
            except DefenseOutputError as exc:
                bad[d["name"]] = str(exc)
                continue
            cc, dc = cp == label, dp == label
            rows.append({"global_index": gi, "modulation": cm["modulation"], "snr_label_db": cm["snr_label_db"],
                         "sample_index": cm["sample_index"], "label": label, "defense": d["name"],
                         "defense_family": d["family"], "clean_pred": cp, "clean_defended_pred": dp,
                         "clean_correct": b2s(cc), "clean_defended_correct": b2s(dc),
                         "clean_degraded": b2s(cc and not dc), "clean_improved": b2s(dc and not cc),
                         "in_at_train": "" if membership is None or d["family"] != "adversarial_training"
                         else b2s(membership[c.keys[gi]]), "defense_ms": ms})
    return rows


def attack_rows(c, defs, p: dict, x_adv_all: np.ndarray, ameta: List[dict], clean: np.ndarray,
                cdef: Dict[tuple, dict], membership, bad: dict) -> List[dict]:
    rows = []
    for j, am in enumerate(ameta):
        gi = int(am["global_index"])
        x_adv = x_adv_all[j:j + 1]
        label, cp, ap = int(am["label"]), int(am["clean_pred"]), int(am["attacked_pred"])
        cc, ac = cp == label, ap == label
        succ = cc and not ac
        for d in defs:
            if d["name"] in bad:
                continue
            try:
                dp, ms = (ap, 0.0) if d["name"] == "none" else defended(c, d, x_adv)
            except DefenseOutputError as exc:
                bad[d["name"]] = str(exc)
                continue
            cr = cdef.get((d["name"], gi))
            if cr is None:
                bad[d["name"]] = f"no clean-input evaluation row for global_index {gi}"
                continue
            dc, cdc = dp == label, cr["clean_defended_correct"] == "True"
            rows.append({"global_index": gi, "attack": p["attack"], "attack_profile": p["attack_profile"],
                         "profile": p["profile"], "modulation": am["modulation"], "snr_label_db": am["snr_label_db"],
                         "sample_index": am["sample_index"], "label": label, "defense": d["name"],
                         "defense_family": d["family"], "clean_pred": cp, "attacked_pred": ap, "defended_pred": dp,
                         "clean_defended_pred": cr["clean_defended_pred"], "clean_correct": b2s(cc),
                         "attacked_correct": b2s(ac), "attack_success": b2s(succ), "defended_correct": b2s(dc),
                         "recovered": b2s(succ and dc), "retained": b2s(cc and dc),
                         "retained_nodef": b2s(cc and not succ), "clean_defended_correct": b2s(cdc),
                         "clean_degraded": b2s(cc and not cdc), "in_at_train": cr["in_at_train"], "defense_ms": ms})
    return rows


def whitebox_rows(c, at_adapter, at_model_def: dict, p: dict, clean: np.ndarray, cmeta: List[dict],
                  idx: List[int], membership) -> List[dict]:
    """Attacks regenerated against the AT model (same profile parameters, same formal seeds)."""
    rows = []
    for gi in idx:
        cm, x = cmeta[gi], clean[gi:gi + 1]
        seed, label = int(cm["formal_seed"]), int(cm["label"])
        c.torch.manual_seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            x_adv, meta = at_adapter.apply(x, attack=p["attack"], eps=p["eps"], temperature=p["temperature"],
                                           seed=seed, diagnostics=p["diagnostics"],
                                           attack_params=dict(p["attack_params"]), **p["cw_kwargs"])
        if meta.get("attack_status") != "ok" or meta.get("attack_backend") != c.REAL_ATTACK:
            p1a.abort(f"whitebox {p['profile']}: attack fell back: {meta.get('attack_notes')}")
        akb.check_iq(x_adv, f"whitebox {p['profile']} {gi}")
        cp, ap = predict(c, at_model_def, x), predict(c, at_model_def, x_adv)
        cc, ac = cp == label, ap == label
        succ = cc and not ac
        rows.append({"global_index": gi, "attack": p["attack"], "attack_profile": p["attack_profile"],
                     "profile": p["profile"], "modulation": cm["modulation"], "snr_label_db": cm["snr_label_db"],
                     "sample_index": cm["sample_index"], "label": label, "defense": at_model_def["name"].replace("_transfer_", "_whitebox_"),
                     "defense_family": "adversarial_training", "clean_pred": cp, "attacked_pred": ap,
                     "defended_pred": ap, "clean_defended_pred": cp, "clean_correct": b2s(cc),
                     "attacked_correct": b2s(ac), "attack_success": b2s(succ), "defended_correct": b2s(ac),
                     "recovered": b2s(False), "retained": b2s(cc and ac), "retained_nodef": b2s(cc and not succ),
                     "clean_defended_correct": b2s(cc), "clean_degraded": b2s(False),
                     "in_at_train": "" if membership is None else b2s(membership[c.keys[gi]]), "defense_ms": ""})
    return rows


# =========================================================================== parts (resume)
def part_paths(out: Path, name: str) -> dict:
    d = out / "eval_parts"
    return {"dir": d, "csv": d / f"{name}.csv", "marker": d / f"{name}.COMPLETE.json"}


def load_part(pp: dict, fields_hash: str, bad: Dict[str, str]) -> Optional[List[dict]]:
    """Reuses a completed part; restores the invalid-defense record it was written with."""
    if not pp["marker"].exists() or not pp["csv"].exists():
        return None
    mk = json.loads(pp["marker"].read_text())
    if mk.get("fields_hash") != fields_hash or akb.sha_file(pp["csv"]) != mk.get("csv_sha256"):
        return None
    rows = akb.read_csv(pp["csv"])
    if len(rows) != mk.get("rows"):
        return None
    bad.update(mk.get("invalid_defenses") or {})
    return rows


def save_part(pp: dict, fields: List[str], rows: List[dict], fields_hash: str, extra: dict) -> None:
    pp["dir"].mkdir(parents=True, exist_ok=True)
    akb.write_csv(pp["csv"], fields, rows)
    akb.atomic_write_text(pp["marker"], json.dumps({"rows": len(rows), "csv_sha256": akb.sha_file(pp["csv"]),
                                                     "fields_hash": fields_hash, **extra,
                                                     "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}))


# =========================================================================== summaries
def metric_row(rows: List[dict]) -> dict:
    m = akb.defense_metrics(rows)
    return {k: m[k] for k in METRIC_KEYS}


def summarize(out: Path, defs, unavailable: Dict[str, str], bad: Dict[str, str], att_rows: List[dict],
              cl_rows: List[dict], wb_rows: List[dict], membership) -> dict:
    # A defense that produced an invalid output anywhere is reported only as INVALID_OUTPUT; its partial rows
    # never enter any table (otherwise its denominator would differ from the other defenses).
    att_rows = [r for r in att_rows if r["defense"] not in bad]
    cl_rows = [r for r in cl_rows if r["defense"] not in bad]
    by_def = akb.group_rows(att_rows, lambda r: (r["defense"],))
    fam = {d["name"]: d["family"] for d in defs}
    summary = []
    for d in defs:
        name = d["name"]
        if name in bad:
            summary.append({"defense": name, "defense_family": d["family"], "status": f"INVALID_OUTPUT: {bad[name]}"})
            continue
        rs = by_def.get((name,), [])
        summary.append({"defense": name, "defense_family": d["family"], "status": "ok" if rs else "NO_ROWS",
                        "source": d["source"], **(metric_row(rs) if rs else {})})
    for name, why in unavailable.items():
        summary.append({"defense": name, "defense_family": "adversarial_training" if "adversarial" in name
                        else "signal_filter", "status": f"UNAVAILABLE: {why}"})
    fields = ["defense", "defense_family", "status", "source"] + METRIC_KEYS
    akb.write_csv(out / "defense_recovery_summary.csv", fields, summary)

    def grouped(keyf, keynames):
        res = []
        for key, rs in sorted(akb.group_rows(att_rows, keyf).items(), key=lambda kv: str(kv[0])):
            res.append({**dict(zip(keynames, key)), "defense_family": fam.get(key[0], ""), **metric_row(rs)})
        return res
    akb.write_csv(out / "defense_recovery_by_attack.csv", ["defense", "attack", "defense_family"] + METRIC_KEYS,
                  grouped(lambda r: (r["defense"], r["attack"]), ["defense", "attack"]))
    akb.write_csv(out / "defense_recovery_by_profile.csv",
                  ["defense", "attack", "attack_profile", "profile", "defense_family"] + METRIC_KEYS,
                  grouped(lambda r: (r["defense"], r["attack"], r["attack_profile"], r["profile"]),
                          ["defense", "attack", "attack_profile", "profile"]))
    cd = []
    for (name,), rs in akb.group_rows(cl_rows, lambda r: (r["defense"],)).items():
        n = len(rs)
        g = lambda f: sum(1 for r in rs if r[f] == "True")  # noqa: E731
        ncc, ncdc, ndeg, nimp = g("clean_correct"), g("clean_defended_correct"), g("clean_degraded"), g("clean_improved")
        cd.append({"defense": name, "defense_family": fam.get(name, ""), "n_clean": n, "n_clean_correct": ncc,
                   "n_clean_defended_correct": ncdc, "n_clean_degraded": ndeg, "n_clean_improved": nimp,
                   "clean_acc_pct": akb.pct(ncc, n), "clean_defended_acc_pct": akb.pct(ncdc, n),
                   "clean_degradation_pct": akb.pct(ndeg, ncc), "clean_improved_pct_of_incorrect": akb.pct(nimp, n - ncc)})
    akb.write_csv(out / "clean_degradation.csv", list(cd[0].keys()) if cd else ["defense"], cd)
    extra = {}
    at_rows = [r for r in att_rows if r["defense_family"] == "adversarial_training"]
    if membership is not None and at_rows:
        ov = []
        for (name, flag), rs in akb.group_rows(at_rows, lambda r: (r["defense"], r["in_at_train"])).items():
            ov.append({"defense": name, "in_at_train": flag, **metric_row(rs)})
        akb.write_csv(out / "at_train_overlap.csv", ["defense", "in_at_train"] + METRIC_KEYS, ov)
        extra["at_eval_samples_in_at_train"] = sum(1 for v in membership.values() if v)
        extra["at_eval_samples_total"] = len(membership)
    if wb_rows:
        wb = [{"scope": "overall", "profile": "", **metric_row(wb_rows)}]
        for (prof,), rs in sorted(akb.group_rows(wb_rows, lambda r: (r["profile"],)).items()):
            wb.append({"scope": "profile", "profile": prof, **metric_row(rs)})
        akb.write_csv(out / "at_whitebox_summary.csv", ["scope", "profile"] + METRIC_KEYS, wb)
    return {"summary": summary, **extra}


# =========================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help=f"{len(SMOKE_KEYS)} fixed samples x all profiles, in memory")
    mode.add_argument("--full", action="store_true", help="2,200 samples x 24 profiles, stored and resumable")
    ap.add_argument("--resume", default=None, help="existing --full output dir; valid parts are reused")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--adv-store-dir", default=None,
                    help="existing Phase B result dir whose clean_store/ and adversarial_store/ are reused read-only")
    ap.add_argument("--at-checkpoint", default=None, help="adversarially trained AWN state_dict (explicit; no default)")
    ap.add_argument("--at-whitebox", action="store_true", help="also regenerate attacks against the AT model")
    ap.add_argument("--defenses", default=None,
                    help="comma-separated defense names and/or families (" + ", ".join(DEFENSE_FAMILIES) + "); "
                         "e.g. adversarial_training for an AT-only run. none (no computation) is always kept")
    ap.add_argument("--profiles", default=None, help="comma-separated subset of attack/profile keys")
    ap.add_argument("--eval-backend", choices=EVAL_BACKENDS, default="cpu")
    ap.add_argument("--config", default=p1a.DEFAULT_CONFIG)
    ap.add_argument("--formal-dir", default=p1a.DEFAULT_FORMAL_CPU_DIR)
    ap.add_argument("--attack-temperature", type=float, default=None)
    ap.add_argument("--diagnostics", choices=["auto", "on", "off"], default="auto")
    args = ap.parse_args()
    mode_name = "smoke" if args.smoke else "full"
    if args.resume and args.smoke:
        p1a.abort("--resume applies to --full only")

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (REPO_ROOT / (args.resume or args.out_dir or f"results/defense_recovery_{mode_name}_{ts}")).resolve()
    for pth in p1a.PROTECTED_RESULT_DIRS + (akb.PHASE_A_DIR,) + ((args.adv_store_dir,) if args.adv_store_dir else ()):
        prot = (REPO_ROOT / pth).resolve()
        if out == prot or prot in out.parents or out in prot.parents:
            p1a.abort(f"refusing to write inside/over {pth}")
    if args.resume and not (out / "manifest.json").exists():
        p1a.abort(f"--resume: {out}/manifest.json not found")
    out.mkdir(parents=True, exist_ok=bool(args.resume))

    c = akb.open_context(args, need_model=True)
    profiles = c.profiles
    if args.profiles:
        want = set(args.profiles.split(","))
        profiles = [p for p in profiles if p["profile"] in want]
        if not profiles:
            p1a.abort(f"--profiles matched no formal profile: {sorted(want)}")
        c.profiles = profiles  # akb.stage_generate iterates c.profiles: generate only what is evaluated
    defs, unavailable, at = build_defenses(c, args)
    at_def = next((d for d in defs if d["family"] == "adversarial_training"), None)
    sources = {"defense_py_sha256": akb.sha_file(akb.DEFENSE_PATH)}
    for f in ("util/defense_baselines.py", "util/defense_registry.py"):
        pth = REPO_ROOT / "external" / "adversarial-rf" / f
        sources[f + "_sha256"] = akb.sha_file(pth) if pth.exists() else None
    run_key = {"mode": mode_name, "defenses": [d["name"] for d in defs], "profiles": [p["profile"] for p in profiles],
               "at_checkpoint_sha256": at.get("sha256"), "at_whitebox": args.at_whitebox,
               "adv_store_dir": args.adv_store_dir, "config_sha256": c.hashes["config_sha256"]}
    man_path = out / "manifest.json"
    if args.resume:
        old = json.loads(man_path.read_text())
        if old.get("run_key") != run_key:
            p1a.abort("--resume: defenses / profiles / AT checkpoint / store differ from the original run")
    manifest = {
        "benchmark": "defense_recovery_comparison", "mode": mode_name, "run_key": run_key,
        "command": [sys.executable] + sys.argv, "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "evaluation_backend": args.eval_backend, "hashes": c.hashes, "sources": {**c.src, **sources},
        "defenses": [{k: v for k, v in d.items() if k not in ("fn", "model")} for d in defs],
        "unavailable": unavailable,
        "adversarial_training": {k: v for k, v in at.items() if k != "adapter"},
        "metric_definitions": "see module docstring; experiments/benchmark_adaptive_k_effectiveness_pi5.py:defense_metrics",
        "denominator": "identical for all defenses: standard-AWN clean_correct / attack_success on the same stored "
                       "adversarial IQ; no shared-success definition exists in the repository; AT white-box uses its "
                       "own denominators and is reported separately",
        "signal_filter_note": "registry default parameters from external/adversarial-rf/util/defense_registry.py:"
                              "_apply_filter; the filters assume raw RadioML IQ scale, which is what the formal "
                              "radioml-native preprocessing feeds AWN",
    }
    akb.atomic_write_text(man_path, json.dumps(manifest, indent=2, default=str))

    # ---------------------------------------------------------------- samples
    if args.smoke:
        keys = [k for k in SMOKE_KEYS if k in c.keys]
        base_rows, _ = p1a.load_formal_rows(c.formal_dir, set(keys))
        inputs = p1a.rebuild_clean_inputs(keys, base_rows, p1a.resolve_sensing(c.cfg), REPO_ROOT / c.cfg["dataset_path"])
        idx = [c.keys.index(k) for k in keys]
        clean = np.zeros((len(c.keys), 2, T_LEN), dtype=np.float32)
        cmeta: List[dict] = [{} for _ in c.keys]
        for gi, k in zip(idx, keys):
            clean[gi] = inputs[k]["x"][0]
            cmeta[gi] = {"global_index": gi, "modulation": k[0], "snr_label_db": k[1], "sample_index": k[2],
                         "label": inputs[k]["label"], "formal_seed": inputs[k]["seed"],
                         "formal_clean_pred": inputs[k]["formal_clean_pred"]}
    else:
        idx = list(range(len(c.keys)))
        if args.adv_store_dir:
            store = (REPO_ROOT / args.adv_store_dir).resolve()
            cp = akb.clean_paths(store)
            ok, why = akb.validate_store(cp, akb.N_CLEAN, "clean_input_sha256")
            if not ok:
                p1a.abort(f"--adv-store-dir clean store invalid ({why})")
            clean, cmeta = np.load(cp["npy"], allow_pickle=False), akb.read_csv(cp["meta"])
        else:
            store = out
            clean, cmeta = akb.ensure_clean_store(c, out)
            gen = akb.stage_generate(c, out)  # resumable per (profile, modulation) chunk
            manifest["generate"] = gen
    keys_eval = [c.keys[gi] for gi in idx]
    membership = at_train_membership(c, at, keys_eval) if at.get("status") == "ok" else None
    if at.get("status") == "ok":
        manifest["adversarial_training"]["train_split_reproduced"] = membership is not None
        manifest["adversarial_training"]["eval_samples_in_at_train"] = (
            sum(1 for v in membership.values() if v) if membership is not None else "unknown (no config json)")
    akb.atomic_write_text(man_path, json.dumps(manifest, indent=2, default=str))

    # ---------------------------------------------------------------- clean evaluation
    bad: Dict[str, str] = {}
    fh = akb.rows_digest(ROW_FIELDS + CLEAN_FIELDS + run_key["defenses"])
    pp = part_paths(out, "clean")
    cl = None if args.smoke else load_part(pp, fh, bad)
    if cl is None:
        cl = clean_rows(c, defs, clean, cmeta, idx, membership, bad)
        if not args.smoke:
            save_part(pp, CLEAN_FIELDS, cl, fh, {"n_samples": len(idx), "invalid_defenses": dict(bad)})
    if args.smoke:
        akb.write_csv(out / "clean_rows.csv", CLEAN_FIELDS, cl)
    cdef = {(r["defense"], int(r["global_index"])): r for r in cl}

    # ---------------------------------------------------------------- attacked evaluation
    att_rows: List[dict] = []
    smoke_checks = []
    for p in profiles:
        pp = part_paths(out, f"{p['profile_index']:02d}_{p['attack']}__{p['attack_profile']}")
        rows = None if args.smoke else load_part(pp, fh, bad)
        if rows is None:
            t0 = time.monotonic()
            if args.smoke:
                arr, ameta = [], []
                for gi in idx:
                    k, cm = c.keys[gi], cmeta[gi]
                    x_adv, ms = akb.attack_one(c, p, clean[gi:gi + 1], int(cm["formal_seed"]))
                    akb.check_iq(x_adv, f"smoke {p['profile']} {k}")
                    sha = p1a.sha256_array(x_adv)
                    eq = sha == c.formal["att"][k + (p["attack"], p["attack_profile"])]["sha"]
                    smoke_checks.append({"profile": p["profile"], "sample": list(k), "sha_matches_formal": eq,
                                         "class": p["reproducibility_class"]})
                    arr.append(x_adv[0])
                    ameta.append({**cm, "clean_pred": cm["formal_clean_pred"],
                                  "attacked_pred": akb.infer_pred(c, x_adv)[0]})
                x_adv_all = np.stack(arr).astype(np.float32)
            else:
                x_adv_all, ameta = akb.load_adversarial(c, store, p)
            rows = attack_rows(c, defs, p, x_adv_all, ameta, clean, cdef, membership, bad)
            if not args.smoke:
                save_part(pp, ROW_FIELDS, rows, fh, {"profile": p["profile"], "invalid_defenses": dict(bad)})
            p1a.log(f"[eval] {p['profile']:32s} {len(rows)} rows in {time.monotonic() - t0:.0f} s")
        att_rows += rows
    if args.smoke:
        akb.write_csv(out / "attack_rows.csv", ROW_FIELDS, att_rows)

    # ---------------------------------------------------------------- optional AT white-box
    wb_rows: List[dict] = []
    if args.at_whitebox and at_def is not None:
        from src.adapters.attack_adapter import AttackAdapter
        at_attack = AttackAdapter(awn_model=at["adapter"].model, device="cpu")
        if at_attack.backend_name != c.REAL_ATTACK or at_attack.status != "ok":
            p1a.abort("AT white-box: attack backend is not real")
        for p in profiles:
            pp = part_paths(out, f"whitebox_{p['profile_index']:02d}_{p['attack']}__{p['attack_profile']}")
            rows = None if args.smoke else load_part(pp, fh, bad)
            if rows is None:
                rows = whitebox_rows(c, at_attack, at_def, p, clean, cmeta, idx, membership)
                if not args.smoke:
                    save_part(pp, ROW_FIELDS, rows, fh, {"profile": p["profile"], "invalid_defenses": dict(bad)})
                p1a.log(f"[whitebox] {p['profile']:32s} {len(rows)} rows")
            wb_rows += rows

    res = summarize(out, defs, unavailable, bad, att_rows, cl, wb_rows, membership)
    manifest.update(status="complete", finished_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
                    invalid_defenses=bad, n_attack_rows=len(att_rows), n_clean_rows=len(cl),
                    n_whitebox_rows=len(wb_rows), **{k: v for k, v in res.items() if k != "summary"})
    if args.smoke:
        fails = [s for s in smoke_checks if s["class"] == "EXPECTED_BIT_IDENTICAL" and not s["sha_matches_formal"]]
        expected = {d["name"] for d in defs}
        seen = {r["defense"] for r in att_rows}
        manifest["smoke"] = {"formal_sha_checks": smoke_checks, "formal_sha_failures": fails,
                             "defenses_without_rows": sorted(expected - seen), "invalid_defenses": bad}
        ok = not fails and not bad and expected == seen
        manifest["smoke"]["overall"] = "PASS" if ok else "FAIL"
    akb.atomic_write_text(man_path, json.dumps(manifest, indent=2, default=str))
    p1a.log(f"[done] {out}")
    for r in res["summary"]:
        p1a.log(f"  {r['defense']:34s} {r['status'][:60]:60s} recovery={r.get('recovery_pct', '')} "
                f"clean_degr={r.get('clean_degradation_pct', '')} net_gain_pp={r.get('retention_gain_pp', '')}")
    if args.smoke:
        p1a.log(f"[smoke] {manifest['smoke']['overall']}")
        if manifest["smoke"]["overall"] != "PASS":
            sys.exit(2)


if __name__ == "__main__":
    main()
