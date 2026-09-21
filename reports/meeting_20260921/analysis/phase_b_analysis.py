#!/usr/bin/env python3
"""Phase B formal analysis (B1-B8 tables) for the meeting_20260921 report package.

Reads only the two formal result directories (read-only) and writes derived
tables to ../tables/ plus a headline-number JSON to phase_b_results.json.

Metric definitions (verified against the raw data in Phase A):
  attack_success = clean_correct AND NOT attacked_correct
  conditional ASR = sum(attack_success) / sum(clean_correct)
  clean_degraded  = clean_correct AND NOT clean_topk_correct
  recovered       = attack_success AND defended_correct
  recovery rate   = sum(recovered) / sum(attack_success)

No NaN filling and no silent row drops: undefined ratios (zero denominator)
stay NaN in the output tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parents[1]
NPU_DIR = ROOT / "results" / "hailo_full_matrix_20260920_seedaligned"
CPU_DIR = ROOT / "results" / "cpu_full_matrix_20260921_seedaligned"
OUT = PKG / "tables"
OUT.mkdir(exist_ok=True)

KEY = ["modulation", "snr_db", "sample_index"]
PKEY = ["attack", "attack_profile"]
BOOL_COLS = ["clean_correct", "attacked_correct", "attack_success",
             "clean_topk_correct", "clean_degraded", "defended_correct", "recovered"]
PRE_STAGES = ["embedding_ms", "energy_detection_ms", "region_postprocess_ms",
              "segmentation_alignment_ms", "awn_preprocess_ms"]

RESULTS: dict = {}


# --------------------------------------------------------------------- loading
def load(directory: Path, infer_prefix: str) -> dict[str, pd.DataFrame]:
    """Load base/attack/defense CSVs, unify the inference-latency column names."""
    base = pd.read_csv(directory / "base_results.csv")
    att = pd.read_csv(directory / "attack_results.csv")
    dfn = pd.read_csv(directory / "defense_results.csv")
    base = base.rename(columns={f"clean_{infer_prefix}_inference_ms": "clean_infer_ms"})
    att = att.rename(columns={f"attacked_{infer_prefix}_inference_ms": "attacked_infer_ms"})
    dfn = dfn.rename(columns={f"defended_{infer_prefix}_inference_ms": "defended_infer_ms"})
    for df in (base, att, dfn):
        for c in BOOL_COLS:
            if c in df.columns:
                assert df[c].dtype == bool, (directory.name, c, df[c].dtype)
    assert (len(base), len(att), len(dfn)) == (2200, 52800, 211200)
    return {"base": base, "att": att, "dfn": dfn}


def profile_order() -> list[tuple[str, str]]:
    m = json.load(open(NPU_DIR / "manifest.json"))
    order = [(a["name"], p["id"]) for a in m["config"]["attacks"] for p in a["profiles"]]
    assert len(order) == 24
    return order


ORDER = profile_order()


def add_profile_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["profile"] = df["attack"] + "/" + df["attack_profile"]
    return df


def sort_profiles(df: pd.DataFrame) -> pd.DataFrame:
    rank = {f"{a}/{p}": i for i, (a, p) in enumerate(ORDER)}
    return df.sort_values(by="profile", key=lambda s: s.map(rank), kind="stable")


def wilson(k: pd.Series, n: pd.Series, z: float = 1.96):
    k = k.astype(float)
    n = n.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    lo, hi = c - h, c + h
    lo = lo.where(n > 0)
    hi = hi.where(n > 0)
    return lo, hi


def ratio(num, den):
    num = num.astype(float)
    den = den.astype(float)
    return (num / den).where(den > 0)


# ------------------------------------------------------------- B1 clean baseline
def clean_table(cpu_b: pd.DataFrame, npu_b: pd.DataFrame, by: str | None) -> pd.DataFrame:
    m = cpu_b[KEY + ["label", "clean_pred", "clean_correct"]].merge(
        npu_b[KEY + ["label", "clean_pred", "clean_correct"]],
        on=KEY, suffixes=("_cpu", "_npu"), validate="one_to_one")
    assert (m["label_cpu"] == m["label_npu"]).all()
    m["agree"] = m["clean_pred_cpu"] == m["clean_pred_npu"]
    m["both_correct"] = m["clean_correct_cpu"] & m["clean_correct_npu"]
    m["cpu_only_correct"] = m["clean_correct_cpu"] & ~m["clean_correct_npu"]
    m["npu_only_correct"] = ~m["clean_correct_cpu"] & m["clean_correct_npu"]
    m["both_wrong"] = ~m["clean_correct_cpu"] & ~m["clean_correct_npu"]
    m["_all"] = "ALL"
    g = m.groupby(by if by else "_all")
    t = g.agg(n=("agree", "size"),
              cpu_correct=("clean_correct_cpu", "sum"),
              npu_correct=("clean_correct_npu", "sum"),
              agree=("agree", "sum"),
              both_correct=("both_correct", "sum"),
              cpu_only_correct=("cpu_only_correct", "sum"),
              npu_only_correct=("npu_only_correct", "sum"),
              both_wrong=("both_wrong", "sum")).reset_index()
    t = t.rename(columns={(by if by else "_all"): "group_value"})
    t.insert(0, "group_by", by if by else "overall")
    t["cpu_acc_pct"] = 100 * t.cpu_correct / t.n
    t["npu_acc_pct"] = 100 * t.npu_correct / t.n
    t["acc_diff_pp_npu_minus_cpu"] = t.npu_acc_pct - t.cpu_acc_pct
    t["agreement_pct"] = 100 * t.agree / t.n
    return t


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return float("nan")
    return float(stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def b1(cpu, npu):
    ov = clean_table(cpu["base"], npu["base"], None)
    bm = clean_table(cpu["base"], npu["base"], "modulation")
    bs = clean_table(cpu["base"], npu["base"], "snr_db")
    clean = pd.concat([ov, bm, bs], ignore_index=True)
    clean["group_value"] = clean["group_value"].astype(str)
    clean.to_csv(OUT / "clean_summary.csv", index=False)
    # agreement table: same rows, agreement / 2x2 breakdown columns only
    agr = clean[["group_by", "group_value", "n", "agree", "agreement_pct", "both_correct",
                 "cpu_only_correct", "npu_only_correct", "both_wrong"]].copy()
    r = ov.iloc[0]
    RESULTS["clean"] = {
        "n": int(r.n), "cpu_correct": int(r.cpu_correct), "npu_correct": int(r.npu_correct),
        "cpu_acc_pct": float(r.cpu_acc_pct), "npu_acc_pct": float(r.npu_acc_pct),
        "diff_pp": float(r.acc_diff_pp_npu_minus_cpu),
        "agree": int(r.agree), "agreement_pct": float(r.agreement_pct),
        "both_correct": int(r.both_correct), "cpu_only": int(r.cpu_only_correct),
        "npu_only": int(r.npu_only_correct), "both_wrong": int(r.both_wrong),
        "mcnemar_exact_p": mcnemar_exact(int(r.cpu_only_correct), int(r.npu_only_correct)),
    }
    return clean, agr


# --------------------------------------------------- attack / defense aggregates
def attack_agg(att: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = att.groupby(keys, sort=False)
    t = g.agg(n=("attack_success", "size"),
              n_clean_correct=("clean_correct", "sum"),
              n_attacked_correct=("attacked_correct", "sum"),
              n_attack_success=("attack_success", "sum"),
              mean_linf=("iq_linf", "mean"),
              mean_l2=("iq_l2", "mean")).reset_index()
    t["clean_acc_pct"] = 100 * ratio(t.n_clean_correct, t.n)
    t["attacked_acc_pct"] = 100 * ratio(t.n_attacked_correct, t.n)
    t["cond_asr_pct"] = 100 * ratio(t.n_attack_success, t.n_clean_correct)
    lo, hi = wilson(t.n_attack_success, t.n_clean_correct)
    t["cond_asr_ci95_lo_pct"] = 100 * lo
    t["cond_asr_ci95_hi_pct"] = 100 * hi
    return t


def defense_agg(dfn: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    # retained = correct both clean and after attack + defense;
    # retained_nodef = correct both clean and after attack alone (= clean_correct & ~attack_success)
    d = dfn.assign(retained=dfn.clean_correct & dfn.defended_correct,
                   retained_nodef=dfn.clean_correct & ~dfn.attack_success)
    g = d.groupby(keys, sort=False)
    t = g.agg(n=("recovered", "size"),
              n_clean_correct=("clean_correct", "sum"),
              n_attack_success=("attack_success", "sum"),
              n_recovered=("recovered", "sum"),
              n_defended_correct=("defended_correct", "sum"),
              n_retained=("retained", "sum"),
              n_retained_nodef=("retained_nodef", "sum"),
              n_clean_topk_correct=("clean_topk_correct", "sum"),
              n_clean_degraded=("clean_degraded", "sum")).reset_index()
    t["recovery_pct"] = 100 * ratio(t.n_recovered, t.n_attack_success)
    lo, hi = wilson(t.n_recovered, t.n_attack_success)
    t["recovery_ci95_lo_pct"] = 100 * lo
    t["recovery_ci95_hi_pct"] = 100 * hi
    t["defended_acc_pct"] = 100 * ratio(t.n_defended_correct, t.n)
    t["retention_pct"] = 100 * ratio(t.n_retained, t.n_clean_correct)
    t["retention_nodef_pct"] = 100 * ratio(t.n_retained_nodef, t.n_clean_correct)
    t["retention_gain_pp"] = t.retention_pct - t.retention_nodef_pct
    t["clean_topk_acc_pct"] = 100 * ratio(t.n_clean_topk_correct, t.n)
    t["clean_degradation_pct"] = 100 * ratio(t.n_clean_degraded, t.n_clean_correct)
    return t


def stack_backends(fn, cpu_df, npu_df, keys):
    a = fn(cpu_df, keys)
    a.insert(0, "backend", "CPU")
    b = fn(npu_df, keys)
    b.insert(0, "backend", "NPU")
    return pd.concat([a, b], ignore_index=True)


def with_pooled(df: pd.DataFrame) -> pd.DataFrame:
    """Attach 'profile' label and a pooled marker column (per-profile rows only)."""
    return sort_profiles(add_profile_label(df))


# ------------------------------------------------------------------ B2 attacks
def b2(cpu, npu):
    t = stack_backends(attack_agg, cpu["att"], npu["att"], PKEY)
    t = with_pooled(t)
    t = t.sort_values(["backend", "profile"], key=lambda s: s if s.name == "backend"
                      else s.map({f"{a}/{p}": i for i, (a, p) in enumerate(ORDER)}), kind="stable")
    first = ["backend", "profile", "attack", "attack_profile"]
    t = t[first + [c for c in t.columns if c not in first]]
    t.to_csv(OUT / "attack_summary.csv", index=False)
    return t


# ---------------------------------------------------------- B3 CPU vs NPU transfer
def b3(cpu, npu):
    k = KEY + PKEY
    cols = ["clean_correct", "attacked_pred", "attacked_correct", "attack_success",
            "adversarial_sha256", "iq_linf"]
    m = cpu["att"][k + cols].merge(npu["att"][k + cols], on=k, suffixes=("_c", "_n"),
                                   validate="one_to_one")
    assert len(m) == 52800
    m["adv_identical"] = m.adversarial_sha256_c == m.adversarial_sha256_n
    m["pred_agree"] = m.attacked_pred_c == m.attacked_pred_n
    m["both_clean"] = m.clean_correct_c & m.clean_correct_n
    m["succ_c_b"] = m.attack_success_c & m.both_clean
    m["succ_n_b"] = m.attack_success_n & m.both_clean
    m["ident_both"] = m.adv_identical & m.both_clean
    m["succ_c_ib"] = m.attack_success_c & m.ident_both
    m["succ_n_ib"] = m.attack_success_n & m.ident_both
    m = add_profile_label(m)
    rows = []
    for prof, d in m.groupby("profile", sort=False):
        nb = int(d.both_clean.sum())
        ib = int(d.ident_both.sum())
        b = int((d.succ_c_b & ~d.succ_n_b).sum())
        c = int((~d.succ_c_b & d.succ_n_b).sum())
        rows.append({
            "profile": prof, "attack": d.attack.iloc[0], "attack_profile": d.attack_profile.iloc[0],
            "n": len(d),
            "adv_identical_rows": int(d.adv_identical.sum()),
            "adv_identical_pct": 100 * d.adv_identical.mean(),
            "cpu_cond_asr_pct": 100 * d.attack_success_c.sum() / d.clean_correct_c.sum(),
            "npu_cond_asr_pct": 100 * d.attack_success_n.sum() / d.clean_correct_n.sum(),
            "n_both_clean_correct": nb,
            "cpu_asr_both_clean_pct": 100 * d.succ_c_b.sum() / nb,
            "npu_asr_both_clean_pct": 100 * d.succ_n_b.sum() / nb,
            "cpu_succ_npu_not": b, "npu_succ_cpu_not": c,
            "mcnemar_exact_p": mcnemar_exact(b, c),
            "attacked_pred_agree_pct": 100 * d.pred_agree.mean(),
            "n_identical_and_both_clean": ib,
            "cpu_asr_identical_pct": 100 * d.succ_c_ib.sum() / ib if ib else np.nan,
            "npu_asr_identical_pct": 100 * d.succ_n_ib.sum() / ib if ib else np.nan,
        })
    t = pd.DataFrame(rows)
    t["asr_diff_pp_npu_minus_cpu"] = t.npu_cond_asr_pct - t.cpu_cond_asr_pct
    t["asr_diff_both_clean_pp"] = t.npu_asr_both_clean_pct - t.cpu_asr_both_clean_pct
    t["asr_diff_identical_pp"] = t.npu_asr_identical_pct - t.cpu_asr_identical_pct
    t["backend_attributable"] = t.adv_identical_rows == t.n
    t = sort_profiles(t)
    t.to_csv(OUT / "cpu_npu_transfer_comparison.csv", index=False)
    return t


# ---------------------------------------------------------------- B4 defense
def b4(cpu, npu):
    keys = PKEY + ["topk"]
    t = stack_backends(defense_agg, cpu["dfn"], npu["dfn"], keys)
    t = add_profile_label(t)
    rank = {f"{a}/{p}": i for i, (a, p) in enumerate(ORDER)}
    t = t.sort_values(["backend", "profile", "topk"],
                      key=lambda s: s.map(rank) if s.name == "profile" else s, kind="stable")
    first = ["backend", "profile", "attack", "attack_profile", "topk"]
    t = t[first + [c for c in t.columns if c not in first]]
    t.to_csv(OUT / "defense_summary.csv", index=False)

    # clean-only Top-K effect (independent of attack): dedupe across profiles after
    # asserting every profile carries identical clean-side values.
    clean_rows = []
    for name, d in (("CPU", cpu["dfn"]), ("NPU", npu["dfn"])):
        chk = d.groupby(KEY + ["topk"])[["clean_topk_pred", "clean_topk_correct",
                                         "clean_degraded", "clean_correct"]].nunique()
        assert (chk.max() == 1).all(), (name, chk.max())
        u = d.drop_duplicates(KEY + ["topk"])
        assert len(u) == 2200 * 4
        for kk, g in u.groupby("topk"):
            cc = int(g.clean_correct.sum())
            clean_rows.append({
                "backend": name, "topk": kk, "n": len(g), "n_clean_correct": cc,
                "clean_topk_acc_pct": 100 * g.clean_topk_correct.mean(),
                "clean_acc_pct": 100 * g.clean_correct.mean(),
                "n_clean_degraded": int(g.clean_degraded.sum()),
                "clean_degradation_pct": 100 * g.clean_degraded.sum() / cc,
                "n_clean_improved": int((~g.clean_correct & g.clean_topk_correct).sum()),
            })
    ct = pd.DataFrame(clean_rows)
    ct["acc_change_pp"] = ct.clean_topk_acc_pct - ct.clean_acc_pct
    ct.to_csv(OUT / "clean_topk_summary.csv", index=False)
    return t, ct


def defense_pooled(cpu, npu):
    """Pooled over all 24 profiles, by Top-K (overall)."""
    a = defense_agg(cpu["dfn"].assign(all_profiles="ALL_24"), ["all_profiles", "topk"])
    b = defense_agg(npu["dfn"].assign(all_profiles="ALL_24"), ["all_profiles", "topk"])
    a.insert(0, "backend", "CPU")
    b.insert(0, "backend", "NPU")
    return pd.concat([a, b], ignore_index=True)


# ------------------------------------------------------ B5 / B6 modulation & SNR
def by_dim(cpu, npu, dim: str, fname_prefix: str):
    """Per-profile and pooled-over-24-profiles tables along one dimension."""
    keys = [dim] + PKEY
    a = stack_backends(attack_agg, cpu["att"], npu["att"], keys)
    a = add_profile_label(a)
    pooled_att = stack_backends(attack_agg, cpu["att"].assign(attack="ALL", attack_profile="ALL_24"),
                                npu["att"].assign(attack="ALL", attack_profile="ALL_24"), keys)
    pooled_att = add_profile_label(pooled_att)
    at = pd.concat([a, pooled_att], ignore_index=True)
    rank = {f"{x}/{y}": i for i, (x, y) in enumerate(ORDER)}
    rank["ALL/ALL_24"] = len(ORDER)
    at = at.sort_values(["backend", dim, "profile"],
                        key=lambda s: s.map(rank) if s.name == "profile" else s, kind="stable")
    first = ["backend", dim, "profile", "attack", "attack_profile"]
    at = at[first + [c for c in at.columns if c not in first]]
    at.to_csv(OUT / f"attack_by_{fname_prefix}.csv", index=False)

    dkeys = [dim] + PKEY + ["topk"]
    d = stack_backends(defense_agg, cpu["dfn"], npu["dfn"], dkeys)
    dp = stack_backends(defense_agg, cpu["dfn"].assign(attack="ALL", attack_profile="ALL_24"),
                        npu["dfn"].assign(attack="ALL", attack_profile="ALL_24"), dkeys)
    dd = pd.concat([add_profile_label(d), add_profile_label(dp)], ignore_index=True)
    dd = dd.sort_values(["backend", dim, "profile", "topk"],
                        key=lambda s: s.map(rank) if s.name == "profile" else s, kind="stable")
    first = ["backend", dim, "profile", "attack", "attack_profile", "topk"]
    dd = dd[first + [c for c in dd.columns if c not in first]]
    dd.to_csv(OUT / f"defense_by_{fname_prefix}.csv", index=False)
    return at, dd


# ------------------------------------------------------------------ B7 / B8 latency
def derived_latency_frames(d, name):
    base, att, dfn = d["base"], d["att"], d["dfn"]
    base = base.copy()
    base["sensing_pre_ms"] = base[PRE_STAGES].sum(axis=1)
    base["e2e_clean_ms"] = base["sensing_pre_ms"] + base["clean_infer_ms"]
    pre = base[KEY + ["sensing_pre_ms"]]
    dfn = dfn.merge(pre, on=KEY, how="left", validate="many_to_one")
    assert dfn.sensing_pre_ms.notna().all()
    # defended end-to-end uses the recorded defense_pipeline_ms (Top-K + inference wall time)
    dfn["e2e_defended_ms"] = dfn.sensing_pre_ms + dfn.defense_pipeline_ms
    att = att.merge(pre, on=KEY, how="left", validate="many_to_one")
    assert att.sensing_pre_ms.notna().all()
    att["e2e_attacked_eval_ms"] = att.sensing_pre_ms + att.attacked_infer_ms
    # recorded *_pipeline_ms are separately measured wall times: always slightly larger
    # than the itemised stages (un-itemised glue overhead). Record the residual.
    ra = att.attack_pipeline_ms - (att.attack_generation_ms + att.attacked_infer_ms)
    rd = dfn.defense_pipeline_ms - (dfn.topk_transform_ms + dfn.defended_infer_ms)
    assert (ra > 0).all() and (rd > 0).all() and ra.max() < 1.0 and rd.max() < 1.0
    RESULTS.setdefault("pipeline_residual_ms", {})[name] = {
        "attack_mean": float(ra.mean()), "attack_max": float(ra.max()),
        "defense_mean": float(rd.mean()), "defense_max": float(rd.max())}
    return base, att, dfn


LAT_SERIES = [
    ("base", "embedding_ms", "sensing"),
    ("base", "energy_detection_ms", "sensing"),
    ("base", "region_postprocess_ms", "sensing"),
    ("base", "segmentation_alignment_ms", "sensing"),
    ("base", "awn_preprocess_ms", "sensing"),
    ("base", "sensing_pre_ms", "sensing (5 stages, derived sum)"),
    ("base", "clean_infer_ms", "inference"),
    ("base", "e2e_clean_ms", "end-to-end clean (derived sum)"),
    ("att", "attacked_infer_ms", "inference"),
    ("att", "e2e_attacked_eval_ms", "end-to-end attacked evaluation, attack generation excluded (derived)"),
    ("dfn", "topk_transform_ms", "defense"),
    ("dfn", "defended_infer_ms", "inference"),
    ("dfn", "defense_pipeline_ms", "defense (Top-K + inference)"),
    ("dfn", "e2e_defended_ms", "end-to-end defended, attack generation excluded (sensing stages + recorded defense_pipeline_ms)"),
    ("att", "attack_generation_ms", "attack generation (offline, CPU surrogate)"),
    ("att", "attack_pipeline_ms", "attack generation + attacked inference"),
]


def pstats(x: pd.Series) -> dict:
    a = x.to_numpy(dtype=float)
    assert np.isfinite(a).all() and (a > 0).all()
    return {"n": len(a), "mean": a.mean(), "median": np.median(a),
            "p95": np.percentile(a, 95), "p99": np.percentile(a, 99),
            "min": a.min(), "max": a.max()}


def b7_b8(cpu, npu):
    fc = dict(zip(("base", "att", "dfn"), derived_latency_frames(cpu, "CPU")))
    fn = dict(zip(("base", "att", "dfn"), derived_latency_frames(npu, "NPU")))
    rows = []
    for tab, col, role in LAT_SERIES:
        c = pstats(fc[tab][col])
        n = pstats(fn[tab][col])
        r = {"stage": col, "role": role, "n_cpu": c["n"], "n_npu": n["n"]}
        for s in ("mean", "median", "p95", "p99"):
            r[f"cpu_{s}_ms"] = c[s]
            r[f"npu_{s}_ms"] = n[s]
            r[f"speedup_{s}"] = c[s] / n[s]
        rows.append(r)
    lat = pd.DataFrame(rows)
    lat.to_csv(OUT / "latency_summary_cpu_npu.csv", index=False)

    # cross-check against the latency_summary.csv shipped with each run
    chk = []
    for name, d, tag in (("CPU", CPU_DIR, "cpu"), ("NPU", NPU_DIR, "npu")):
        ls = pd.read_csv(d / "latency_summary.csv").set_index("stage")
        for stg in ls.index:
            src = lat.set_index("stage")
            # unified names differ for inference columns
            key = stg.replace("_cpu_inference", "_infer").replace("_hailo_inference", "_infer")
            for s in ("mean", "median", "p95", "p99"):
                mine = src.loc[key, f"{tag}_{s}_ms"]
                chk.append(abs(mine - ls.loc[stg, s]) / ls.loc[stg, s])
    assert max(chk) < 1e-9, max(chk)
    RESULTS["latency_crosscheck_max_rel_err"] = float(max(chk))

    # attack-generation latency by profile (offline cost)
    g = []
    for name, d in (("CPU", cpu["att"]), ("NPU", npu["att"])):
        x = add_profile_label(d).groupby("profile", sort=False).attack_generation_ms.agg(
            mean="mean", median="median", p95=lambda s: np.percentile(s, 95),
            p99=lambda s: np.percentile(s, 99), max="max").reset_index()
        x.insert(0, "backend", name)
        g.append(x)
    ag = sort_profiles(pd.concat(g, ignore_index=True))
    ag = ag.rename(columns={"mean": "mean_ms", "median": "median_ms", "p95": "p95_ms",
                            "p99": "p99_ms", "max": "max_ms"})
    ag.sort_values(["backend"], kind="stable").to_csv(OUT / "attack_generation_latency.csv", index=False)
    return lat



# ------------------------------------------- supporting evidence for interpretation
def b_extra(cpu, npu, tr):
    """Confusion matrices, Top-K prediction collapse, AM-DSB -> WBFM shift."""
    lab = (cpu["base"].drop_duplicates("label").set_index("label").modulation.sort_index())
    assert (lab.index.to_list() == list(range(11)))
    lab_of = lab.to_dict()
    inv = {v: k for k, v in lab_of.items()}

    # clean confusion (true x predicted), long form
    rows = []
    for name, d in (("CPU", cpu), ("NPU", npu)):
        ct = pd.crosstab(d["base"].label, d["base"].clean_pred).reindex(
            index=range(11), columns=range(11), fill_value=0)
        for t in range(11):
            for q in range(11):
                rows.append({"backend": name, "true_label": t, "true_modulation": lab_of[t],
                             "pred_label": q, "pred_modulation": lab_of[q], "count": int(ct.loc[t, q])})
    conf = pd.DataFrame(rows)
    conf.to_csv(OUT / "clean_confusion_cpu_npu.csv", index=False)
    g = conf[(conf.true_modulation == "GFSK") & (conf.pred_modulation == "WBFM")].set_index("backend")["count"]
    RESULTS["gfsk_to_wbfm_clean"] = {k: int(v) for k, v in g.items()}

    # Top-K clean prediction distribution (attack independent) per backend and K
    rows = []
    for name, d in (("CPU", cpu), ("NPU", npu)):
        u = d["dfn"].drop_duplicates(KEY + ["topk"])
        for kk, gg in u.groupby("topk"):
            vc = gg.clean_topk_pred.value_counts().reindex(range(11), fill_value=0)
            for q in range(11):
                rows.append({"backend": name, "topk": kk, "pred_label": q,
                             "pred_modulation": lab_of[q], "count": int(vc[q]),
                             "share_pct": 100 * vc[q] / len(gg)})
    pdist = pd.DataFrame(rows)
    pdist.to_csv(OUT / "clean_topk_pred_distribution.csv", index=False)
    top4 = (pdist.sort_values(["backend", "topk", "count"], ascending=[True, True, False])
            .groupby(["backend", "topk"]).head(4).groupby(["backend", "topk"]).share_pct.sum())
    RESULTS["topk_top4_pred_share_pct"] = {f"{b}_K{k}": float(v) for (b, k), v in top4.items()}

    # share of recovered samples contributed by QAM64 + PAM4 + AM-DSB (pooled over profiles)
    fav = ["QAM64", "PAM4", "AM-DSB"]
    out = {}
    for name, d in (("CPU", cpu), ("NPU", npu)):
        r = d["dfn"][d["dfn"].recovered]
        for kk, gg in r.groupby("topk"):
            out[f"{name}_K{kk}"] = float(100 * gg.modulation.isin(fav).mean())
    RESULTS["recovered_share_qam64_pam4_amdsb_pct"] = out

    # AM-DSB clean-correct rows, profiles with identical adversarial inputs:
    # fraction of attacked predictions equal to WBFM
    k = KEY + PKEY
    m = cpu["att"][k + ["clean_correct", "attacked_pred", "adversarial_sha256"]].merge(
        npu["att"][k + ["clean_correct", "attacked_pred", "adversarial_sha256"]], on=k,
        suffixes=("_c", "_n"), validate="one_to_one")
    m = add_profile_label(m)
    ident = set(tr[tr.backend_attributable].profile)
    x = m[(m.modulation == "AM-DSB") & m.clean_correct_c & m.clean_correct_n
          & m.profile.isin(ident) & (m.adversarial_sha256_c == m.adversarial_sha256_n)]
    RESULTS["amdsb_identical_inputs"] = {
        "n_rows": int(len(x)),
        "cpu_attacked_pred_wbfm_pct": float(100 * (x.attacked_pred_c == inv["WBFM"]).mean()),
        "npu_attacked_pred_wbfm_pct": float(100 * (x.attacked_pred_n == inv["WBFM"]).mean()),
        "cpu_attacked_pred_amdsb_pct": float(100 * (x.attacked_pred_c == inv["AM-DSB"]).mean()),
        "npu_attacked_pred_amdsb_pct": float(100 * (x.attacked_pred_n == inv["AM-DSB"]).mean()),
    }
    # GFSK-excluded clean accuracy
    for name, d in (("CPU", cpu), ("NPU", npu)):
        b = d["base"]
        RESULTS.setdefault("clean_acc_excl_gfsk_pct", {})[name] = float(
            100 * b[b.modulation != "GFSK"].clean_correct.mean())

# --------------------------------------------------------------------------- main
def main():
    cpu = load(CPU_DIR, "cpu")
    npu = load(NPU_DIR, "hailo")

    clean, agr = b1(cpu, npu)
    agr.to_csv(OUT / "cpu_npu_prediction_agreement.csv", index=False)

    # cross-check with Phase A
    pa = json.load(open(HERE / "phase_a_validation.json"))["clean_comparison"]
    RESULTS["phase_a_clean_comparison_keys"] = sorted(pa.keys())

    att = b2(cpu, npu)
    tr = b3(cpu, npu)
    dfn, ctop = b4(cpu, npu)
    pooled = defense_pooled(cpu, npu)
    pooled.to_csv(OUT / "defense_pooled_by_topk.csv", index=False)
    a_mod, d_mod = by_dim(cpu, npu, "modulation", "modulation")
    a_snr, d_snr = by_dim(cpu, npu, "snr_db", "snr")
    lat = b7_b8(cpu, npu)
    b_extra(cpu, npu, tr)

    # internal consistency: pooled ASR equals mean of per-profile ASR (equal denominators)
    for be in ("CPU", "NPU"):
        s = att[att.backend == be]
        assert s.n_clean_correct.nunique() == 1
        pooled_asr = 100 * s.n_attack_success.sum() / s.n_clean_correct.sum()
        assert abs(pooled_asr - s.cond_asr_pct.mean()) < 1e-9
        RESULTS.setdefault("pooled_asr_all24_pct", {})[be] = float(pooled_asr)

    RESULTS["attack_zero_success_profiles"] = {
        be: att[(att.backend == be) & (att.n_attack_success == 0)].profile.tolist()
        for be in ("CPU", "NPU")}
    RESULTS["defense_undefined_recovery_rows"] = int(dfn.recovery_pct.isna().sum())
    RESULTS["adv_identical_all_profiles"] = tr[tr.backend_attributable].profile.tolist()
    RESULTS["adv_not_identical_profiles"] = tr[~tr.backend_attributable].profile.tolist()
    RESULTS["clean_topk"] = ctop.to_dict(orient="records")
    RESULTS["pooled_defense"] = pooled[["backend", "topk", "recovery_pct", "clean_degradation_pct",
                                        "retention_pct", "retention_nodef_pct",
                                        "retention_gain_pp"]].to_dict(orient="records")
    # identity check: undefended retention == 100 - conditional ASR (pooled, per backend)
    for be in ("CPU", "NPU"):
        r0 = pooled[(pooled.backend == be)].retention_nodef_pct
        assert (abs(r0 - (100 - RESULTS["pooled_asr_all24_pct"][be])) < 1e-9).all()
    json.dump(RESULTS, open(HERE / "phase_b_results.json", "w"), indent=2, default=float)
    print("Phase B tables written to", OUT.relative_to(ROOT))
    for f in sorted(OUT.glob("*.csv")):
        print(f"  {f.name}: {len(pd.read_csv(f))} rows")


if __name__ == "__main__":
    main()
