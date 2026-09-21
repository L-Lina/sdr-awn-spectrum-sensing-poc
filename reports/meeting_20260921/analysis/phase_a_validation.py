"""Phase A: formal data validation for the CPU / Hailo-8 full-matrix results.

Read-only with respect to results/, data/, external/ and the HEF file.
Outputs: analysis/phase_a_validation.json and analysis/phase_a_validation.md
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

DIRS = {
    "NPU": ROOT / "results/hailo_full_matrix_20260920_seedaligned",
    "CPU": ROOT / "results/cpu_full_matrix_20260921_seedaligned",
}
EXPECTED = {"base": 2200, "attack": 52800, "defense": 211200}
MODS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
SNRS = list(range(-20, 20, 2))
IDXS = list(range(10))
TOPKS = [10, 20, 30, 40]
N_PROFILES = 24
LATENCY_STAGES_BASE = ["embedding_ms", "energy_detection_ms", "region_postprocess_ms",
                       "segmentation_alignment_ms", "awn_preprocess_ms"]
BOOL_COLS = {
    "base": ["clean_correct"],
    "attack": ["clean_correct", "attacked_correct", "attack_success"],
    "defense": ["clean_correct", "attacked_correct", "attack_success", "clean_topk_correct",
                "clean_degraded", "defended_correct", "recovered"],
}
KEY_BASE = ["modulation", "snr_db", "sample_index"]
KEY_ATK = KEY_BASE + ["attack", "attack_profile"]
KEY_DEF = KEY_ATK + ["topk"]

checks: list[dict] = []
notes: list[str] = []


def record(cid: str, name: str, ok: bool, detail: str = "", mandatory: bool = True):
    checks.append({"id": cid, "name": name, "pass": bool(ok), "detail": detail, "mandatory": mandatory})


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load(tag: str):
    d = DIRS[tag]
    dfs = {}
    for kind in ["base", "attack", "defense"]:
        dfs[kind] = pd.read_csv(d / f"{kind}_results.csv", keep_default_na=True)
    manifest = json.loads((d / "manifest.json").read_text())
    summary = json.loads((d / "summary.json").read_text())
    lat = pd.read_csv(d / "latency_summary.csv")
    return dfs, manifest, summary, lat


def expected_profiles(manifest):
    out = []
    for a in manifest["config"]["attacks"]:
        for p in a["profiles"]:
            out.append((a["name"], p["id"]))
    return out


def strict_bool_ok(series: pd.Series) -> bool:
    return series.dtype == bool


data = {t: load(t) for t in DIRS}

# ---------------------------------------------------------------- A1 / A2 / A3 per backend
for tag, (dfs, man, summ, lat) in data.items():
    b, a, d = dfs["base"], dfs["attack"], dfs["defense"]
    # A1
    for kind, df in dfs.items():
        record(f"A1.{tag}.{kind}_rows", f"{tag} {kind} rows == {EXPECTED[kind]}", len(df) == EXPECTED[kind],
               f"observed {len(df)}")
    record(f"A1.{tag}.manifest_counts", f"{tag} manifest completed_* rows match expected",
           (man["completed_base_rows"], man["completed_attack_rows"], man["completed_defense_rows"])
           == (2200, 52800, 211200) and man["run_status"] == "complete",
           f"manifest completed=({man['completed_base_rows']},{man['completed_attack_rows']},{man['completed_defense_rows']}) "
           f"run_status={man['run_status']}")
    record(f"A1.{tag}.summary_counts", f"{tag} summary.json counts match CSV row counts",
           summ["counts"] == {"base_rows": len(b), "attack_rows": len(a), "defense_rows": len(d)},
           f"summary counts={summ['counts']}")

    # A2 grid completeness
    cfg = man["config"]
    record(f"A2.{tag}.config_grid", f"{tag} manifest config grid matches expected",
           cfg["modulations"] == MODS and cfg["snrs"] == SNRS and cfg["sample_indices"] == IDXS
           and cfg["topk_values"] == TOPKS and man["attack_profile_count"] == N_PROFILES,
           f"n_mod={len(cfg['modulations'])} n_snr={len(cfg['snrs'])} n_idx={len(cfg['sample_indices'])} "
           f"topk={cfg['topk_values']} profiles={man['attack_profile_count']}")
    profiles = expected_profiles(man)
    record(f"A2.{tag}.profile_count", f"{tag} manifest lists {N_PROFILES} unique attack profiles",
           len(profiles) == N_PROFILES and len(set(profiles)) == N_PROFILES, f"{len(profiles)} listed")

    exp_base = set(itertools.product(MODS, SNRS, IDXS))
    obs_base = set(map(tuple, b[KEY_BASE].itertuples(index=False, name=None)))
    record(f"A2.{tag}.base_keys", f"{tag} base key set == 11x20x10 grid",
           obs_base == exp_base and len(exp_base) == 2200,
           f"missing={len(exp_base - obs_base)} unexpected={len(obs_base - exp_base)}")
    exp_atk = set(itertools.product(MODS, SNRS, IDXS, profiles))
    exp_atk = {(m, s, i, p[0], p[1]) for (m, s, i, p) in exp_atk}
    obs_atk = set(map(tuple, a[KEY_ATK].itertuples(index=False, name=None)))
    record(f"A2.{tag}.attack_keys", f"{tag} attack key set == base grid x 24 profiles",
           obs_atk == exp_atk, f"missing={len(exp_atk - obs_atk)} unexpected={len(obs_atk - exp_atk)}")
    exp_def = {k + (kk,) for k in exp_atk for kk in TOPKS}
    obs_def = set(map(tuple, d[KEY_DEF].itertuples(index=False, name=None)))
    record(f"A2.{tag}.defense_keys", f"{tag} defense key set == attack grid x TopK{{10,20,30,40}}",
           obs_def == exp_def, f"missing={len(exp_def - obs_def)} unexpected={len(obs_def - exp_def)}")
    per_cell = b.groupby(["modulation", "snr_db"]).size()
    record(f"A2.{tag}.samples_per_cell", f"{tag} 10 samples per modulation x SNR cell",
           len(per_cell) == 220 and (per_cell == 10).all(), f"cells={len(per_cell)} min={per_cell.min()} max={per_cell.max()}")

    # A3 integrity
    for kind, df in dfs.items():
        dup = int(df.duplicated(KEY_BASE if kind == "base" else KEY_ATK if kind == "attack" else KEY_DEF).sum())
        full_dup = int(df.duplicated().sum())
        record(f"A3.{tag}.{kind}_dup", f"{tag} {kind} duplicate keys / duplicate rows = 0", dup == 0 and full_dup == 0,
               f"dup_keys={dup} dup_rows={full_dup}")
        num = df.select_dtypes(include=[np.number])
        nan_cols = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
        inf_cols = {c: int(np.isinf(num[c]).sum()) for c in num.columns if np.isinf(num[c]).any()}
        if kind == "base":
            allowed = {"missed_signal_samples", "false_occupied_samples"}
            unexpected_nan = {c: n for c, n in nan_cols.items() if c not in allowed}
            allowed_full = all(nan_cols.get(c) == len(df) for c in allowed if c in nan_cols)
            record(f"A3.{tag}.{kind}_nan", f"{tag} {kind} NaN only in structurally-empty columns", not unexpected_nan and allowed_full,
                   f"NaN columns={nan_cols}; unexpected={unexpected_nan}")
            if nan_cols:
                notes.append(f"{tag} base_results: columns {sorted(nan_cols)} are empty in all {len(df)} rows "
                             f"(not used in any analysis).")
        else:
            record(f"A3.{tag}.{kind}_nan", f"{tag} {kind} NaN = 0", not nan_cols, f"NaN columns={nan_cols}")
        record(f"A3.{tag}.{kind}_inf", f"{tag} {kind} Inf = 0", not inf_cols, f"Inf columns={inf_cols}")
        record(f"A3.{tag}.{kind}_bool", f"{tag} {kind} boolean columns strictly True/False",
               all(strict_bool_ok(df[c]) for c in BOOL_COLS[kind]),
               "dtypes=" + str({c: str(df[c].dtype) for c in BOOL_COLS[kind]}))
        status_cols = [c for c in df.columns if c.endswith("status")]
        bad_status = {c: int((df[c] != "ok").sum()) for c in status_cols}
        record(f"A3.{tag}.{kind}_status", f"{tag} {kind} status columns all 'ok'", all(v == 0 for v in bad_status.values()),
               f"non-ok counts={bad_status}")
        # backends
        be_cols = [c for c in df.columns if c.endswith("backend")]
        for c in be_cols:
            vals = sorted(df[c].unique())
            txt = " ".join(vals).lower()
            has_bad = any(w in txt for w in ["dummy", "fallback", "mock", "fake"])
            record(f"A3.{tag}.{kind}.{c}", f"{tag} {kind}.{c}: single expected backend, no dummy/fallback",
                   len(vals) == 1 and not has_bad, f"values={vals}")
        # string sanity for every object column: no dummy/fallback/error/nan text
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        hits = {}
        for c in obj_cols:
            s = df[c].astype(str).str.lower()
            n = int(s.str.contains(r"dummy|fallback|error|traceback|exception|failed|nan|inf", regex=True).sum()) if c not in ("modulation", "attack", "attack_profile", "adversarial_sha256", "clean_input_sha256") else 0
            if n:
                hits[c] = n
        record(f"A3.{tag}.{kind}_strings", f"{tag} {kind} no dummy/fallback/error text in string columns", not hits, f"hits={hits}")
    # latency values
    lat_cols_b = LATENCY_STAGES_BASE + [c for c in b.columns if c.startswith("clean_") and c.endswith("_inference_ms")]
    lat_cols_a = [c for c in a.columns if c.endswith("_ms")]
    lat_cols_d = [c for c in d.columns if c.endswith("_ms")]
    okp = all((b[c] > 0).all() for c in lat_cols_b) and all((a[c] > 0).all() for c in lat_cols_a) and all((d[c] > 0).all() for c in lat_cols_d)
    record(f"A3.{tag}.latency_positive", f"{tag} all latency columns finite and > 0", okp, "")
    # shape / range consistency
    seg_len = (b.selected_segment_end - b.selected_segment_start)
    true_len = (b.true_end - b.true_start)
    record(f"A3.{tag}.shape", f"{tag} selected segment length == 128 and true region length == 128 for all rows",
           bool((seg_len == 128).all() and (true_len == 128).all()),
           f"seg_len unique={sorted(seg_len.unique())} true_len unique={sorted(true_len.unique())}")
    label_map = b.groupby("modulation").label.nunique()
    inv = b.groupby("label").modulation.nunique()
    record(f"A3.{tag}.label_map", f"{tag} modulation<->label one-to-one", bool((label_map == 1).all() and (inv == 1).all() and len(inv) == 11),
           str(b.groupby("modulation").label.first().to_dict()))
    preds_ok = all(df_[c].between(0, 10).all() for df_, c in [(b, "clean_pred"), (a, "attacked_pred"), (d, "defended_pred"), (d, "clean_topk_pred")])
    record(f"A3.{tag}.pred_range", f"{tag} predictions in class range [0,10], no missing", preds_ok, "")
    hex_ok = all(df_[c].astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
                 for df_, c in [(b, "clean_input_sha256"), (a, "adversarial_sha256"), (d, "adversarial_sha256")])
    record(f"A3.{tag}.hash_format", f"{tag} sha256 fields well-formed (64 hex)", hex_ok, "")
    lab_consistent = (a.merge(b[KEY_BASE + ["label"]], on=KEY_BASE, suffixes=("", "_b")).eval("label == label_b").all()
                      and d.merge(b[KEY_BASE + ["label"]], on=KEY_BASE, suffixes=("", "_b")).eval("label == label_b").all())
    record(f"A3.{tag}.label_consistent", f"{tag} labels consistent across base/attack/defense", bool(lab_consistent), "")
    # metric-definition consistency
    r1 = ((b.clean_pred == b.label) == b.clean_correct).all() and ((a.attacked_pred == a.label) == a.attacked_correct).all() \
        and ((d.defended_pred == d.label) == d.defended_correct).all() and ((d.clean_topk_pred == d.label) == d.clean_topk_correct).all()
    record(f"A3.{tag}.correct_flags", f"{tag} *_correct flags agree with pred==label", bool(r1), "")
    r2 = ((a.clean_correct & ~a.attacked_correct) == a.attack_success).all()
    r3 = ((d.clean_correct & ~d.clean_topk_correct) == d.clean_degraded).all()
    r4 = ((d.attack_success & d.defended_correct) == d.recovered).all()
    record(f"A3.{tag}.metric_defs", f"{tag} attack_success = clean_correct & ~attacked_correct; clean_degraded = clean_correct & ~clean_topk_correct; recovered = attack_success & defended_correct",
           bool(r2 and r3 and r4), f"attack_success={bool(r2)} clean_degraded={bool(r3)} recovered={bool(r4)}")
    # cross-table consistency
    m = a.merge(b[KEY_BASE + ["clean_pred", "clean_correct"]], on=KEY_BASE, suffixes=("", "_b"))
    record(f"A3.{tag}.attack_vs_base", f"{tag} attack.clean_pred/clean_correct == base", bool((m.clean_pred == m.clean_pred_b).all() and (m.clean_correct == m.clean_correct_b).all() and len(m) == len(a)), "")
    md = d.merge(a[KEY_ATK + ["clean_correct", "attacked_correct", "attack_success", "adversarial_sha256"]], on=KEY_ATK, suffixes=("", "_a"))
    record(f"A3.{tag}.defense_vs_attack", f"{tag} defense clean/attacked/success flags and adversarial hash == attack table",
           bool(len(md) == len(d) and (md.clean_correct == md.clean_correct_a).all() and (md.attacked_correct == md.attacked_correct_a).all()
                and (md.attack_success == md.attack_success_a).all() and (md.adversarial_sha256 == md.adversarial_sha256_a).all()), "")
    perturb_ok = all(np.isfinite(a[c]).all() and (a[c] >= 0).all() for c in ["iq_linf", "iq_l2", "iq_l1"])
    record(f"A3.{tag}.perturbation", f"{tag} iq_linf/iq_l2/iq_l1 finite and non-negative", perturb_ok, "")
    # latency_summary recomputation
    lat_ok = True
    detail = []
    src = {}
    for c in lat_cols_b:
        src[c] = b[c]
    for c in lat_cols_a:
        src[c] = a[c]
    for c in lat_cols_d:
        src[c] = d[c]
    for _, row in lat.iterrows():
        s = src[row["stage"]]
        rec = dict(n=len(s), mean=s.mean(), median=s.median(), p95=np.percentile(s, 95), p99=np.percentile(s, 99),
                   min=s.min(), max=s.max(), std=s.std(ddof=0))
        for k, v in rec.items():
            if not np.isclose(v, row[k], rtol=1e-9, atol=1e-12):
                lat_ok = False
                detail.append(f"{row['stage']}.{k}: recomputed={v} file={row[k]}")
    record(f"A3.{tag}.latency_recompute", f"{tag} latency_summary.csv recomputed from raw CSV (mean/median/p95/p99/min/max/std, population std ddof=0)",
           lat_ok and len(lat) == len(src), "; ".join(detail[:5]) or f"{len(lat)} stages match")
    # summary.json latency vs csv
    sj_ok = all(np.isclose(summ["latency"][r.stage][k], r[k], rtol=1e-12) for _, r in lat.iterrows() for k in ["mean", "median", "p95", "p99"])
    record(f"A3.{tag}.summary_latency", f"{tag} summary.json latency == latency_summary.csv", bool(sj_ok), "")

# ---------------------------------------------------------------- A4 alignment
bn, bc = data["NPU"][0]["base"], data["CPU"][0]["base"]
al = bc.merge(bn, on=KEY_BASE, suffixes=("_cpu", "_npu"), validate="one_to_one")
record("A4.rows", "CPU/NPU base merge on (modulation,snr,sample_index) is one-to-one, 2200 rows", len(al) == 2200, f"merged={len(al)}")
align_fields = ["label", "seed", "true_start", "true_end", "region_count", "selected_segment_start", "selected_segment_end",
                "detected_region_start", "detected_region_end", "captured_signal_ratio", "start_boundary_error",
                "end_boundary_error", "clean_input_sha256"]
mism = {f: int((al[f + "_cpu"] != al[f + "_npu"]).sum()) for f in align_fields}
record("A4.fields", "CPU/NPU sensing fields identical row-by-row (label, seed, true_start/end, selected segment start/end, detected region, clean_input_sha256, ...)",
       all(v == 0 for v in mism.values()), f"mismatch counts={mism}")

# ---------------------------------------------------------------- A5 clean comparison
cc, cn = al["clean_correct_cpu"], al["clean_correct_npu"]
agree = int((al.clean_pred_cpu == al.clean_pred_npu).sum())
a5 = {
    "n": len(al),
    "prediction_agreement_count": agree,
    "prediction_agreement_pct": 100 * agree / len(al),
    "disagreement_count": len(al) - agree,
    "both_correct": int((cc & cn).sum()),
    "cpu_correct_npu_wrong": int((cc & ~cn).sum()),
    "cpu_wrong_npu_correct": int((~cc & cn).sum()),
    "both_wrong": int((~cc & ~cn).sum()),
    "cpu_clean_accuracy_pct": 100 * cc.mean(),
    "npu_clean_accuracy_pct": 100 * cn.mean(),
    "accuracy_diff_pp_npu_minus_cpu": 100 * (cn.mean() - cc.mean()),
}
# ---------------------------------------------------------------- A6 provenance
manc, mann = data["CPU"][1], data["NPU"][1]
prov = {}
ds_path = ROOT / mann["config"]["dataset_path"]
ck_path = ROOT / mann["config"]["checkpoint_path"]
hef_path = Path(mann["config"]["hef_path"])
prov["dataset_path_exists"] = ds_path.exists()
prov["checkpoint_path_exists"] = ck_path.exists()
prov["hef_path_exists"] = hef_path.exists()
prov["dataset_sha256_recomputed"] = sha256_file(ds_path) if ds_path.exists() else None
prov["checkpoint_sha256_recomputed"] = sha256_file(ck_path) if ck_path.exists() else None
prov["hef_sha256_recomputed"] = sha256_file(hef_path) if hef_path.exists() else None
for tag, man in [("CPU", manc), ("NPU", mann)]:
    record(f"A6.{tag}.dataset_hash", f"{tag} manifest dataset_sha256 == recomputed sha256 of {man['config']['dataset_path']}",
           prov["dataset_sha256_recomputed"] == man["dataset_sha256"], f"manifest={man['dataset_sha256']}")
    record(f"A6.{tag}.checkpoint_hash", f"{tag} manifest checkpoint_sha256 == recomputed sha256 of {man['config']['checkpoint_path']}",
           prov["checkpoint_sha256_recomputed"] == man["checkpoint_sha256"], f"manifest={man['checkpoint_sha256']}")
record("A6.NPU.hef_hash", "NPU manifest hef_sha256 == recomputed sha256 of HEF", prov["hef_sha256_recomputed"] == mann.get("hef_sha256"),
       f"manifest={mann.get('hef_sha256')}")
record("A6.NPU.hef_backend_col", "NPU per-row hilo_backend names the manifest HEF file",
       set(data["NPU"][0]["base"].hilo_backend.unique()) == {f"Hailo-8:{hef_path.name}"}, "")
record("A6.CPU.no_hef_hash", "CPU manifest has no hef_sha256 (CPU backend does not deploy a HEF)", "hef_sha256" not in manc,
       "CPU manifest contains 'hef_path' in config only as inherited config text" if "hef_path" in manc["config"] else "", mandatory=False)
record("A6.CPU.backend", "CPU manifest inference_backend == 'PyTorch CPU AWN'", manc.get("inference_backend") == "PyTorch CPU AWN", str(manc.get("inference_backend")))
record("A6.NPU.threat_model", "NPU manifest threat_model states PyTorch-surrogate generation and quantized Hailo-8 evaluation",
       "differentiable PyTorch AWN surrogate" in mann["threat_model"] and "quantized Hailo-8" in mann["threat_model"], mann["threat_model"])
record("A6.CPU.threat_model", "CPU manifest threat_model states generation and evaluation on original PyTorch AWN",
       "original differentiable PyTorch AWN" in manc["threat_model"], manc["threat_model"])
record("A6.config_equal", "CPU and NPU manifest 'config' blocks identical", manc["config"] == mann["config"], "")
record("A6.hashes_equal", "CPU and NPU manifests share dataset/checkpoint hashes", manc["dataset_sha256"] == mann["dataset_sha256"]
       and manc["checkpoint_sha256"] == mann["checkpoint_sha256"], "")
cfg_disk = json.loads((ROOT / "configs/hailo_full_matrix.json").read_text()) if (ROOT / "configs/hailo_full_matrix.json").exists() else None
record("A6.config_on_disk", "configs/hailo_full_matrix.json (current working tree) == manifest config", cfg_disk == mann["config"],
       "config file is untracked in git; compared for reference", mandatory=False)
record("A6.seed", "manifest sensing.base_seed == 42 and per-row seeds identical across CPU/NPU",
       mann["config"]["sensing"]["base_seed"] == 42 == manc["config"]["sensing"]["base_seed"] and mism["seed"] == 0,
       f"base_seed CPU={manc['config']['sensing']['base_seed']} NPU={mann['config']['sensing']['base_seed']}; "
       f"unique per-row seeds={bc.seed.nunique()}")
if manc["config"]["experiment_name"] == mann["config"]["experiment_name"]:
    notes.append(f"Both manifests carry config.experiment_name='{mann['config']['experiment_name']}' "
                 "(name inherited from the shared config file); backends are distinguished by the results directory names "
                 "and by the manifest fields 'inference_backend' / 'hef_sha256'.")
notes.append("Manifests record base_seed=42 and per-row seeds but no git commit hash / software versions.")

# ---------------------------------------------------------------- logs (informational)
log_scan = {}
pat = re.compile(r"traceback|exception|fallback|dummy|\berror\b|failed|nan\b", re.I)
for tag, d in DIRS.items():
    for name in ["nohup.log", "run.log"]:
        p = d / name
        if not p.exists():
            continue
        n = 0
        hits = 0
        samples = []
        with p.open(errors="replace") as f:
            for line in f:
                n += 1
                if pat.search(line):
                    hits += 1
                    if len(samples) < 3:
                        samples.append(line.strip()[:200])
        log_scan[f"{tag}/{name}"] = {"lines": n, "flagged_lines": hits, "samples": samples}
        record(f"A3.{tag}.{name}", f"{tag} {name}: no traceback/exception/fallback/dummy/error/failed lines", hits == 0,
               f"lines={n} flagged={hits} samples={samples}", mandatory=False)

# ---------------------------------------------------------------- informational: adversarial-example identity CPU vs NPU
ac_, an_ = data["CPU"][0]["attack"], data["NPU"][0]["attack"]
mm = ac_.merge(an_, on=KEY_ATK, suffixes=("_cpu", "_npu"), validate="one_to_one")
adv_same = int((mm.adversarial_sha256_cpu == mm.adversarial_sha256_npu).sum())
info = {
    "attack_rows_merged": len(mm),
    "adversarial_sha256_identical_rows": adv_same,
    "adversarial_sha256_differing_rows": len(mm) - adv_same,
    "iq_linf_max_abs_diff": float((mm.iq_linf_cpu - mm.iq_linf_npu).abs().max()),
    "iq_l2_max_abs_diff": float((mm.iq_l2_cpu - mm.iq_l2_npu).abs().max()),
    "clean_pred_identical_in_attack_table": int((mm.clean_pred_cpu == mm.clean_pred_npu).sum()),
}
if adv_same != len(mm):
    dd = mm[mm.adversarial_sha256_cpu != mm.adversarial_sha256_npu]
    info["differing_by_attack_profile"] = dd.groupby(["attack", "attack_profile"]).size().to_dict().__repr__()

# ---------------------------------------------------------------- result
all_pass = all(c["pass"] for c in checks if c["mandatory"])
result = {
    "verdict": "FORMAL DATA VALIDATION: PASS" if all_pass else "FORMAL DATA VALIDATION: FAIL",
    "n_checks": len(checks),
    "n_mandatory_failed": sum(1 for c in checks if c["mandatory"] and not c["pass"]),
    "n_informational_failed": sum(1 for c in checks if not c["mandatory"] and not c["pass"]),
    "checks": checks,
    "notes": notes,
    "clean_comparison": a5,
    "provenance": prov,
    "log_scan": log_scan,
    "cpu_npu_adversarial_identity": info,
    "sources": {t: str(p.relative_to(ROOT)) for t, p in DIRS.items()},
}
(OUT / "phase_a_validation.json").write_text(json.dumps(result, indent=2, default=str))

lines = [f"# Phase A validation\n", f"**{result['verdict']}**  (checks: {len(checks)}, mandatory failed: {result['n_mandatory_failed']}, informational failed: {result['n_informational_failed']})\n",
         "| ID | Check | Result | Detail |", "|---|---|---|---|"]
for c in checks:
    lines.append(f"| {c['id']} | {c['name']} | {'PASS' if c['pass'] else ('FAIL' if c['mandatory'] else 'NOTE')} | {str(c['detail'])[:160].replace('|', '/')} |")
lines += ["", "## Notes"] + [f"- {n}" for n in notes]
lines += ["", "## Clean CPU vs NPU", "```", json.dumps(a5, indent=2), "```",
          "", "## CPU/NPU adversarial-example identity (informational)", "```", json.dumps(info, indent=2), "```"]
(OUT / "phase_a_validation.md").write_text("\n".join(lines))

print(result["verdict"])
for c in checks:
    if not c["pass"]:
        print("  FAIL" if c["mandatory"] else "  NOTE", c["id"], "|", c["name"], "|", str(c["detail"])[:300])
print(json.dumps(a5, indent=2))
print(json.dumps(info, indent=2))
sys.exit(0 if all_pass else 1)
