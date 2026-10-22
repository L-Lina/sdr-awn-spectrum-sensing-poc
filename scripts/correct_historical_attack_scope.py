"""
Restrict the pooled (aggregate) results of the historical reports to the main attack scope.

Main attack scope: FGSM, PGD, CW, EAD. In reports/meeting_20260921, meeting_20260928 and meeting_20261005 every
pooled aggregate (ALL / ALL_24 rows, pooled-over-profiles defense tables, overall Hailo paired rows) was computed
over all 24 attack profiles (17 attacks). This script recomputes those aggregates in place from the per-profile /
per-attack count rows committed in the same report directories, pooling only the main-scope profiles
(FGSM eps 0.005/0.01/0.03/0.05, PGD eps 0.005/0.01/0.03, CW, EAD: 9 profiles). Per-profile and per-attack rows of
other attacks are left unchanged as historical records.

Sources (all inside the repository):
    meeting_20260921/tables/{attack_summary,defense_summary,attack_by_snr,attack_by_modulation,
                             defense_by_snr,defense_by_modulation}.csv          per-profile counts
    meeting_20261005/review_evidence/03_defense_recovery_cpu/defense_recovery_full_20261004_215947/
        defense_recovery_by_attack.csv                                         per-attack counts (CPU defenses)
    meeting_20261005/review_evidence/05_hailo_paired/adaptive_k_hailo_paired_20260928/
        defense_summary_paired.csv (scope = profile)                           per-profile counts (CPU and Hailo-8)

Aggregates without a per-attack source in the repository are not recomputed; their rows are relabelled
ALL_24_HISTORICAL and annotated (leakage-free AT transfer evaluation; exploratory AT train/eval-overlap split).

The script first checks that the per-profile / per-attack counts reproduce the existing all-profile aggregates
exactly, then writes the main-scope aggregates. Run from the repository root:
    ~/adversarial-rf/.venv/bin/python scripts/correct_historical_attack_scope.py
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R21 = REPO / "reports" / "meeting_20260921"
R28 = REPO / "reports" / "meeting_20260928"
R05 = REPO / "reports" / "meeting_20261005"
BYATT = R05 / "review_evidence/03_defense_recovery_cpu/defense_recovery_full_20261004_215947/defense_recovery_by_attack.csv"
PAIRED = R05 / "review_evidence/05_hailo_paired/adaptive_k_hailo_paired_20260928/defense_summary_paired.csv"
MAIN = ("fgsm", "pgd", "cw", "ead")
TAG = "FGSM_PGD_CW_EAD"
POOL_ATTACK, POOL_AP = "MAIN4", TAG
POOL_PROFILE = f"{POOL_ATTACK}/{POOL_AP}"
HIST = "ALL_24_HISTORICAL"
HIST_NOTE = ("historical all-24-profile scope; no per-attack source in the repository; "
             "not a main-scope (FGSM/PGD/CW/EAD) result")
MAIN_SOURCE_NOTE = "main attack scope FGSM/PGD/CW/EAD (9 profiles)"


class CorrectionError(RuntimeError):
    pass


def require(cond, msg):
    if not cond:
        raise CorrectionError(msg)


def rcsv(path: Path):
    raw = path.read_bytes()
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    rd = csv.DictReader(io.StringIO(text))
    return list(rd.fieldnames), list(rd), bom, ('"' + rd.fieldnames[0] + '"') in text.splitlines()[0]


def wcsv(path: Path, fields, rows, bom=False, quote_all=False):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n",
                       quoting=csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + buf.getvalue().encode("utf-8"))


def fnum(x):
    if isinstance(x, int):
        return str(x)
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return repr(float(x))


def pct(k, n):
    return 100.0 * k / n if n else float("nan")


def wilson(k, n, z=1.96):
    if not n:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def isum(rows, k):
    return sum(int(float(r[k])) for r in rows)


# ======================================================================================== 0921
ATT_COUNTS = ("n", "n_clean_correct", "n_attacked_correct", "n_attack_success")
DEF_COUNTS = ("n", "n_clean_correct", "n_attack_success", "n_recovered", "n_defended_correct", "n_retained",
              "n_retained_nodef", "n_clean_topk_correct", "n_clean_degraded")


def attack_pool(rows):
    s = {k: isum(rows, k) for k in ATT_COUNTS}
    lo, hi = wilson(s["n_attack_success"], s["n_clean_correct"])
    return {**{k: s[k] for k in ATT_COUNTS},
            "mean_linf": sum(float(r["mean_linf"]) * int(r["n"]) for r in rows) / s["n"],
            "mean_l2": sum(float(r["mean_l2"]) * int(r["n"]) for r in rows) / s["n"],
            "clean_acc_pct": pct(s["n_clean_correct"], s["n"]),
            "attacked_acc_pct": pct(s["n_attacked_correct"], s["n"]),
            "cond_asr_pct": pct(s["n_attack_success"], s["n_clean_correct"]),
            "cond_asr_ci95_lo_pct": lo, "cond_asr_ci95_hi_pct": hi}


def defense_pool(rows):
    s = {k: isum(rows, k) for k in DEF_COUNTS}
    lo, hi = wilson(s["n_recovered"], s["n_attack_success"])
    ret, ret0 = pct(s["n_retained"], s["n_clean_correct"]), pct(s["n_retained_nodef"], s["n_clean_correct"])
    return {**{k: s[k] for k in DEF_COUNTS},
            "recovery_pct": pct(s["n_recovered"], s["n_attack_success"]),
            "recovery_ci95_lo_pct": lo, "recovery_ci95_hi_pct": hi,
            "defended_acc_pct": pct(s["n_defended_correct"], s["n"]),
            "retention_pct": ret, "retention_nodef_pct": ret0, "retention_gain_pp": ret - ret0,
            "clean_topk_acc_pct": pct(s["n_clean_topk_correct"], s["n"]),
            "clean_degradation_pct": pct(s["n_clean_degraded"], s["n_clean_correct"])}


def close(a, b, tol=1e-9):
    a, b = float(a), float(b)
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol * max(1.0, abs(b))


def fix_0921_by_dim(fname, dim, kind):
    path = R21 / "tables" / fname
    fields, rows, bom, qa = rcsv(path)
    per = [r for r in rows if r["attack"] not in ("ALL", POOL_ATTACK)]
    pooled_idx = [i for i, r in enumerate(rows) if r["attack"] in ("ALL", POOL_ATTACK)]
    require(pooled_idx, f"{fname}: no pooled rows found")
    keys = ("backend", dim) + (("topk",) if kind == "defense" else ())
    fn = attack_pool if kind == "attack" else defense_pool
    for i in pooled_idx:
        old = rows[i]
        k = tuple(old[x] for x in keys)
        same = [r for r in per if tuple(r[x] for x in keys) == k]
        if old["attack"] == "ALL":  # integrity: per-profile rows reproduce the existing all-24 pooled row
            chk = fn(same)
            cnt = ATT_COUNTS if kind == "attack" else DEF_COUNTS
            require(all(int(old[c]) == chk[c] for c in cnt), f"{fname} {k}: per-profile counts != pooled row")
        main = [r for r in same if r["attack"] in MAIN]
        require(len(main) == 9, f"{fname} {k}: expected 9 main-scope profiles, found {len(main)}")
        new = fn(main)
        rows[i] = {**old, "profile": POOL_PROFILE, "attack": POOL_ATTACK, "attack_profile": POOL_AP,
                   **{c: fnum(v) for c, v in new.items()}}
    wcsv(path, fields, rows, bom, qa)
    return len(pooled_idx)


def fix_0921():
    out = {}
    out["attack_by_snr"] = fix_0921_by_dim("attack_by_snr.csv", "snr_db", "attack")
    out["attack_by_modulation"] = fix_0921_by_dim("attack_by_modulation.csv", "modulation", "attack")
    out["defense_by_snr"] = fix_0921_by_dim("defense_by_snr.csv", "snr_db", "defense")
    out["defense_by_modulation"] = fix_0921_by_dim("defense_by_modulation.csv", "modulation", "defense")

    # pooled by Top-K
    _, dsum, _, _ = rcsv(R21 / "tables/defense_summary.csv")
    path = R21 / "tables/defense_pooled_by_topk.csv"
    fields, rows, bom, qa = rcsv(path)
    new_rows = []
    for old in rows:
        same = [r for r in dsum if r["backend"] == old["backend"] and r["topk"] == old["topk"]]
        if old["all_profiles"] == "ALL_24":
            chk = defense_pool(same)
            require(all(int(old[c]) == chk[c] for c in DEF_COUNTS), f"pooled_by_topk {old['backend']} {old['topk']}")
        main = [r for r in same if r["attack"] in MAIN]
        require(len(main) == 9, "defense_summary main-scope profile count")
        new_rows.append({**old, "all_profiles": TAG, **{c: fnum(v) for c, v in defense_pool(main).items()}})
    wcsv(path, fields, new_rows, bom, qa)
    out["defense_pooled_by_topk"] = len(new_rows)

    # phase_b_results.json
    pj = R21 / "analysis/phase_b_results.json"
    res = json.loads(pj.read_text())
    _, asum, _, _ = rcsv(R21 / "tables/attack_summary.csv")
    pooled_asr = {}
    for be in ("CPU", "NPU"):
        m = [r for r in asum if r["backend"] == be and r["attack"] in MAIN]
        require(len(m) == 9, "attack_summary main-scope profile count")
        pooled_asr[be] = pct(isum(m, "n_attack_success"), isum(m, "n_clean_correct"))
    res.pop("pooled_asr_all24_pct", None)
    res["pooled_asr_main4_pct"] = pooled_asr
    res["pooled_defense"] = [{"backend": r["backend"], "topk": int(r["topk"]), **{
        k: float(r[k]) for k in ("recovery_pct", "clean_degradation_pct", "retention_pct", "retention_nodef_pct",
                                 "retention_gain_pp")}} for r in new_rows]
    _, dmod, _, _ = rcsv(R21 / "tables/defense_by_modulation.csv")
    share = {}
    for be in ("CPU", "NPU"):
        for k in ("10", "20", "30", "40"):
            m = [r for r in dmod if r["backend"] == be and r["topk"] == k and r["attack"] in MAIN]
            fav = sum(int(r["n_recovered"]) for r in m if r["modulation"] in ("QAM64", "PAM4", "AM-DSB"))
            share[f"{be}_K{k}"] = pct(fav, sum(int(r["n_recovered"]) for r in m))
    res["recovered_share_qam64_pam4_amdsb_pct"] = share
    res["amdsb_identical_inputs"] = {
        "status": "not recomputed for the main attack scope",
        "reason": "requires per-row adversarial_sha256 and attacked_pred from results/cpu_full_matrix_20260921_"
                  "seedaligned and results/hailo_full_matrix_20260920_seedaligned, which are not in the repository"}
    res["main_attack_scope"] = {"attacks": list(MAIN), "profiles": 9,
                                "note": "pooled aggregates restricted to FGSM/PGD/CW/EAD"}
    pj.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    return out, pooled_asr, share


# ======================================================================================== paired (0928 / 1005)
def paired_main():
    _, rows, _, _ = rcsv(PAIRED)
    prof = [r for r in rows if r["scope"] == "profile"]
    out = {}
    for d in ("none", "topk10", "topk20", "topk30", "topk40", "adaptive_k_v2"):
        same = [r for r in prof if r["defense"] == d]
        require(len(same) == 24, f"paired: {d} has {len(same)} profile rows")
        overall = next(r for r in rows if r["scope"] == "overall" and r["defense"] == d)
        cnt = [c for c in overall if c.startswith(("cpu_n_", "hailo_n_"))]
        require(all(isum(same, c) == int(overall[c]) for c in cnt), f"paired: profile rows != overall ({d})")
        main = [r for r in same if r["group"].split("/")[0] in MAIN]
        require(len(main) == 9, f"paired: {d} main-scope rows {len(main)}")
        res = {}
        for be in ("cpu", "hailo"):
            s = {c: isum(main, f"{be}_{c}") for c in ("n_clean_correct", "n_attacked_correct", "n_attack_success",
                                                      "n_recovered", "n_defended_correct", "n_retained",
                                                      "n_retained_nodef", "n_clean_defended_correct",
                                                      "n_clean_degraded")}
            n = isum(main, "n")
            ret, ret0 = pct(s["n_retained"], s["n_clean_correct"]), pct(s["n_retained_nodef"], s["n_clean_correct"])
            res[be] = {**s, "n": n, "recovery_pct": pct(s["n_recovered"], s["n_attack_success"]),
                       "clean_degradation_pct": pct(s["n_clean_degraded"], s["n_clean_correct"]),
                       "retention_pct": ret, "retention_nodef_pct": ret0, "net_gain_pp": ret - ret0,
                       "defended_acc_pct": pct(s["n_defended_correct"], n),
                       "clean_acc_pct": pct(s["n_clean_correct"], n),
                       "cond_attack_success_rate_pct": pct(s["n_attack_success"], s["n_clean_correct"])}
        out[d] = res
    return out


def fix_0928(P):
    path = R28 / "tables/adaptive_k_cpu_hailo_overall.csv"
    fields, rows, bom, qa = rcsv(path)
    names = {"無防禦": "none", "固定 K=10": "topk10", "固定 K=20": "topk20", "固定 K=30": "topk30",
             "固定 K=40": "topk40", "Adaptive-K v2": "adaptive_k_v2"}
    cols = [("clean degradation (%)", "clean_degradation_pct"), ("recovery (%)", "recovery_pct"),
            ("retention (%)", "retention_pct"), ("retention 淨增益 (pp)", "net_gain_pp")]
    for r in rows:
        d = P[names[r[fields[0]]]]
        for be, lab in (("cpu", "CPU"), ("hailo", "Hailo")):
            for title, key in cols:
                r[f"{lab} {title}"] = f"{d[be][key]:.2f}"
        r["來源檔"] = ("adaptive_k_hailo_paired_20260928/defense_summary_paired.csv（scope=profile，"
                     "FGSM/PGD/CW/EAD 9 個 profile 合併）" if names[r[fields[0]]] == "none" else "同上")
    wcsv(path, fields, rows, bom, qa)

    path = R28 / "tables/figure_sources/fig2_adaptive_k_tradeoff.csv"
    fields, rows, bom, qa = rcsv(path)
    for r in rows:
        be = "cpu" if r["backend"] == "CPU" else "hailo"
        d = P[r["defense"]][be]
        r.update({"clean_degradation_pct": f"{d['clean_degradation_pct']:.2f}", "recovery_pct": f"{d['recovery_pct']:.2f}",
                  "retention_pct": f"{d['retention_pct']:.2f}", "retention_nodef_pct": f"{d['retention_nodef_pct']:.2f}",
                  "retention_gain_pp": f"{d['net_gain_pp']:.2f}",
                  "source": f"adaptive_k_hailo_paired_20260928/defense_summary_paired.csv (scope=profile, {be}_*, "
                            f"{MAIN_SOURCE_NOTE})"})
    wcsv(path, fields, rows, bom, qa)
    return rows


# ======================================================================================== 1005
BY_COUNTS = ("n", "n_clean_correct", "n_attacked_correct", "n_attack_success", "n_recovered", "n_defended_correct",
             "n_retained", "n_retained_nodef", "n_clean_defended_correct", "n_clean_degraded")


def byattack_main():
    _, rows, _, _ = rcsv(BYATT)
    _, summ, _, _ = rcsv(BYATT.parent / "defense_recovery_summary.csv")
    out = {}
    for sm in summ:
        same = [r for r in rows if r["defense"] == sm["defense"]]
        require(all(isum(same, c) == int(sm[c]) for c in BY_COUNTS), f"by_attack != summary ({sm['defense']})")
        main = [r for r in same if r["attack"] in MAIN]
        require(len(main) == 4, f"by_attack main rows for {sm['defense']}")
        s = {c: isum(main, c) for c in BY_COUNTS}
        lo, hi = wilson(s["n_recovered"], s["n_attack_success"])
        ret, ret0 = pct(s["n_retained"], s["n_clean_correct"]), pct(s["n_retained_nodef"], s["n_clean_correct"])
        out[sm["defense"]] = {
            "counts": s,
            "recovery_pct": (pct(s["n_recovered"], s["n_attack_success"]), s["n_recovered"], s["n_attack_success"]),
            "recovery_ci95_lo_pct": (lo, None, None), "recovery_ci95_hi_pct": (hi, None, None),
            "clean_degradation_pct": (pct(s["n_clean_degraded"], s["n_clean_correct"]), s["n_clean_degraded"],
                                      s["n_clean_correct"]),
            "retention_pct": (ret, s["n_retained"], s["n_clean_correct"]),
            "retention_nodef_pct": (ret0, s["n_retained_nodef"], s["n_clean_correct"]),
            "net_gain_pp": (ret - ret0, None, None),
            "attacked_acc_pct": (pct(s["n_attacked_correct"], s["n"]), s["n_attacked_correct"], s["n"]),
            "defended_acc_pct": (pct(s["n_defended_correct"], s["n"]), s["n_defended_correct"], s["n"]),
            "clean_acc_pct": (pct(s["n_clean_correct"], s["n"]), s["n_clean_correct"], s["n"]),
            "cond_attack_success_rate_pct": (pct(s["n_attack_success"], s["n_clean_correct"]), s["n_attack_success"],
                                             s["n_clean_correct"]),
        }
    return out


def relabel_rows(rows, B):
    for r in rows:
        if r["profile"] != "ALL_24":
            continue
        if r["source_result_dir"].endswith("defense_recovery_full_20261004_215947") and r["metric_name"] in \
                B.get(r["defense"], {}) and r["source_file"] == "defense_recovery_summary.csv":
            v, k, n = B[r["defense"]][r["metric_name"]]
            r.update({"profile": TAG, "value": fnum(v), "raw_count": "" if k is None else str(k),
                      "denominator": "" if n is None else str(n),
                      "source_file": "defense_recovery_by_attack.csv",
                      "notes": (r["notes"] + "; " if r["notes"] else "") + MAIN_SOURCE_NOTE})
        else:
            r.update({"profile": HIST, "notes": (r["notes"] + "; " if r["notes"] else "") + HIST_NOTE})


def fix_1005(B, P):
    for fn in ("defense_recovery_cpu_summary.csv", "adversarial_training_summary.csv"):
        path = R05 / "data" / fn
        fields, rows, bom, qa = rcsv(path)
        relabel_rows(rows, B)
        wcsv(path, fields, rows, bom, qa)

    path = R05 / "data/hailo_defense_summary.csv"
    fields, rows, bom, qa = rcsv(path)
    cnt = {"recovery_pct": ("n_recovered", "n_attack_success"), "clean_degradation_pct": ("n_clean_degraded",
                                                                                         "n_clean_correct"),
           "retention_pct": ("n_retained", "n_clean_correct"), "defended_acc_pct": ("n_defended_correct", "n"),
           "clean_acc_pct": ("n_clean_correct", "n"), "cond_attack_success_rate_pct": ("n_attack_success",
                                                                                       "n_clean_correct")}
    for r in rows:
        if r["profile"] != "ALL":
            continue
        be, key = r["metric_name"].split("_", 1)
        d = P[r["defense"]][be]
        r["value"] = fnum(d[key])
        if key in cnt:
            k, n = cnt[key]
            r["raw_count"], r["denominator"] = str(d[k]), str(d[n])
        r["profile"] = TAG
        r["notes"] = (r["notes"] + "; " if r["notes"] else "") + f"scope=profile rows pooled over {MAIN_SOURCE_NOTE}"
    wcsv(path, fields, rows, bom, qa)

    # figure sources
    path = R05 / "figure_sources/fig2_cpu_defense_tradeoff.csv"
    fields, rows, bom, qa = rcsv(path)
    out = []
    for r in rows:
        if r["defense"] == "adversarial_training_transfer_leakage_free":
            continue  # no per-attack source: not plotted in the main-scope figure
        b = B[r["defense"]]
        r.update({"clean_degradation_pct": fnum(b["clean_degradation_pct"][0]),
                  "clean_degradation_raw": str(b["clean_degradation_pct"][1]),
                  "clean_degradation_denominator": str(b["clean_degradation_pct"][2]),
                  "recovery_pct": fnum(b["recovery_pct"][0]), "recovery_raw": str(b["recovery_pct"][1]),
                  "recovery_denominator": str(b["recovery_pct"][2]),
                  "recovery_ci95_lo_pct": fnum(b["recovery_ci95_lo_pct"][0]),
                  "recovery_ci95_hi_pct": fnum(b["recovery_ci95_hi_pct"][0]),
                  "notes": MAIN_SOURCE_NOTE + ("; train/eval overlap (exploratory)" if "exploratory" in r["defense"]
                                               else "")})
        out.append(r)
    wcsv(path, fields, out, bom, qa)
    fig2 = out

    path = R05 / "figure_sources/fig3_cpu_hailo_defense_comparison.csv"
    fields, rows, bom, qa = rcsv(path)
    for r in rows:
        p = P[r["hailo_paired_defense"]]
        r.update({"cpu_net_gain_pp": fnum(p["cpu"]["net_gain_pp"]), "hailo_net_gain_pp": fnum(p["hailo"]["net_gain_pp"]),
                  "cpu_recovery_pct": fnum(p["cpu"]["recovery_pct"]),
                  "hailo_recovery_pct": fnum(p["hailo"]["recovery_pct"]),
                  "cpu_net_gain_pp_in_defense_recovery_cpu_summary": fnum(B[r["cpu_comparison_defense"]]["net_gain_pp"][0])})
        r["cpu_values_match"] = str(close(r["cpu_net_gain_pp"], r["cpu_net_gain_pp_in_defense_recovery_cpu_summary"]))
        require(r["cpu_values_match"] == "True", f"CPU net gain mismatch between sources for {r['label']}")
    wcsv(path, fields, rows, bom, qa)
    return fig2, rows


# ======================================================================================== figures
def figures(fig28, fig2_05, fig3_05):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    for cand in ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",):
        if Path(cand).exists():
            font_manager.fontManager.addfont(cand)
            # Latin glyphs from DejaVu Sans, CJK glyphs from the fallback font
            plt.rcParams["font.family"] = ["DejaVu Sans", font_manager.FontProperties(fname=cand).get_name()]
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

    # 0928 Figure 2
    labels = {"topk10": "固定 K=10", "topk20": "固定 K=20", "topk30": "固定 K=30", "topk40": "固定 K=40",
              "adaptive_k_v2": "Adaptive-K v2"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
    for ax, be, title in zip(axes, ("CPU", "Hailo-8"), ("CPU（PyTorch AWN）", "Hailo-8（surrogate transfer）")):
        rr = [r for r in fig28 if r["backend"] == be]
        x = range(len(rr))
        for j, (key, col, lab) in enumerate((("clean_degradation_pct", "#8C8C8C", "Clean degradation (%)"),
                                             ("recovery_pct", "#4C72B0", "Recovery (%)"),
                                             ("retention_gain_pp", "#55A868", "Retention 淨增益 (pp)"))):
            v = [float(r[key]) for r in rr]
            bars = ax.bar([i + (j - 1) * 0.26 for i in x], v, 0.26, color=col, label=lab)
            for b, y in zip(bars, v):
                ax.text(b.get_x() + b.get_width() / 2, y + (0.8 if y >= 0 else -2.4), f"{y:.1f}", ha="center", fontsize=8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(list(x)); ax.set_xticklabels([labels[r["defense"]] for r in rr])
        ax.set_title(title, fontsize=12); ax.set_ylim(-30, 80)
    axes[0].legend(loc="upper right", fontsize=9, frameon=False)
    fig.suptitle("圖 2　固定 K 與 Adaptive-K v2 之防禦 trade-off（FGSM／PGD／CW／EAD 9 個 profile 合併）", fontsize=13)
    fig.text(0.01, 0.01, "Retention 淨增益 = 含防禦之 retention − 無防禦 retention。來源：adaptive_k_hailo_paired_20260928/"
             "defense_summary_paired.csv（scope = profile，主研究攻擊範圍 FGSM、PGD、CW、EAD）。A0 digital 攻擊。",
             fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(R28 / "figures/fig2_adaptive_k_tradeoff.png", dpi=150)
    plt.close(fig)

    # 1005 Figure 2
    style = {"fixed_topk": ("o", "#2A7AD5"), "adaptive_k": ("D", "#E8653A"), "signal_filter": ("s", "#1FA774"),
             "adversarial_training_exploratory": ("^", "#E85A9B")}
    fig, ax = plt.subplots(figsize=(11, 6.4))
    for r in fig2_05:
        mk, col = style[r["group"]]
        face = "none" if r["group"] == "adversarial_training_exploratory" else col
        ax.scatter(float(r["clean_degradation_pct"]), float(r["recovery_pct"]), marker=mk, s=60, facecolors=face,
                   edgecolors=col, zorder=3)
        off = (0, -15) if r["label"] == "Gaussian" else (0, 8)  # keep neighbouring labels apart
        ax.annotate(r["label"], (float(r["clean_degradation_pct"]), float(r["recovery_pct"])),
                    textcoords="offset points", xytext=off, ha="center", fontsize=9)
    for g, lab in (("fixed_topk", "Fixed Top-K"), ("adaptive_k", "Adaptive-K"), ("signal_filter", "Signal filters（預設參數）"),
                   ("adversarial_training_exploratory", "Adversarial training（exploratory）")):
        mk, col = style[g]
        ax.scatter([], [], marker=mk, facecolors="none" if "exploratory" in g else col, edgecolors=col, label=lab)
    ax.set_xlim(0, 90); ax.set_ylim(0, max(float(r["recovery_pct"]) for r in fig2_05) + 10)
    ax.set_xlabel("Clean degradation（%）"); ax.set_ylabel("Recovery（%）"); ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=9)
    n_succ = fig2_05[0]["recovery_denominator"]
    ax.set_title(f"圖 2、CPU 數位域防禦之 recovery 與 clean degradation 取捨（FGSM／PGD／CW／EAD；attack success "
                 f"{int(n_succ):,} 列）\nexploratory AT 為空心標記（存在 train/eval overlap）；leakage-free AT 無逐攻擊來源，"
                 "不列入本圖", fontsize=11)
    fig.tight_layout()
    fig.savefig(R05 / "figures/fig2_cpu_defense_tradeoff.png", dpi=110)
    plt.close(fig)

    # 1005 Figure 3
    fig, ax = plt.subplots(figsize=(11, 6.2))
    x = list(range(len(fig3_05)))
    for j, (key, col, lab) in enumerate((("cpu_net_gain_pp", "#2A7AD5", "CPU"), ("hailo_net_gain_pp", "#E8653A",
                                                                                   "Hailo-8"))):
        v = [float(r[key]) for r in fig3_05]
        bars = ax.bar([i + (j - 0.5) * 0.36 for i in x], v, 0.36, color=col, label=lab)
        for b, y in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, y + (0.6 if y >= 0 else -1.6), f"{y:+.2f}", ha="center", fontsize=9)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([r["label"] for r in fig3_05])
    ax.set_ylabel("Net gain（pp）＝ retention − 無防禦 retention"); ax.set_xlabel("防禦"); ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("圖 3、CPU 與 Hailo-8 防禦 net gain 比較（FGSM／PGD／CW／EAD 9 個 profile 合併）\n"
                 "同一批對抗輸入（PyTorch AWN 生成）；Hailo-8 為 surrogate transfer 評估；數位域結果", fontsize=11)
    fig.tight_layout()
    fig.savefig(R05 / "figures/fig3_cpu_hailo_defense_comparison.png", dpi=110)
    plt.close(fig)


def main():
    out, pooled_asr, share = fix_0921()
    P = paired_main()
    B = byattack_main()
    fig28 = fix_0928(P)
    fig2_05, fig3_05 = fix_1005(B, P)
    figures(fig28, fig2_05, fig3_05)
    summary = {"0921_rows_rewritten": out, "0921_pooled_asr_main4_pct": pooled_asr,
               "0921_recovered_share": share,
               "paired_main": {d: {be: {k: v for k, v in x.items() if not k.startswith("n_")} for be, x in r.items()}
                               for d, r in P.items()},
               "cpu_defense_main": {d: {k: v[0] for k, v in r.items() if k != "counts"} for d, r in B.items()}}
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
