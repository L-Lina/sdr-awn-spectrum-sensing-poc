#!/usr/bin/env python3
"""
Compare the superseded 20261022 attack runs (AttackAdapter called with both mkldnn-off opt-ins) with the
default-backend reruns (AttackAdapter defaults), per attack / profile.

Inputs (repository-relative run directories, see RUNS below):
  old     superseded run (no x_adv arrays saved; x_adv_sha256 recorded in R1 and R4 only)
  new     default-backend rerun (--save-x-adv)
  legacy  comparison-only reproduction of the superseded configuration (--legacy-backend-toggle --save-x-adv)

Array-level differences (bit-identical count, max / mean absolute difference) are computed between new and
legacy. They stand for new vs old only where legacy reproduces old (repro_fidelity.csv: attacked prediction,
Linf, L2 and, where recorded, x_adv SHA-256 identical). Every other metric is computed between new and old
directly from the recorded CSVs.

Outputs (results/backend_toggle_rerun_comparison_<timestamp>/):
  semantic_comparison.csv   attacked_pred agreement, x_adv bit-identical count, max / mean |dx|, Linf / L2
                            difference, conditional ASR, oracle CRC, oracle frame correct, defense recovery
  latency_comparison.csv    attack-generation latency median / mean / p95, old vs new
  repro_fidelity.csv        legacy vs old
  summary.json
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def load_runs(path: Path) -> dict:
    """RUNS file: lines 'R1_default=results/...', 'R1_legacy=...' (written by the rerun driver)."""
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line and line[:2] in ("R1", "R2", "R3", "R4"):
            k, v = line.split("=", 1)
            out[k] = v.strip()
    return out


OLD = {"R1": "results/reduced_iteration_attacks_vm_20261022_084556",
       "R2": "results/synth_crc_fec_integration_20261022_090735",
       "R3": "results/reduced_iteration_crc_fec_vm_20261022_093829",
       "R4": "results/reduced_iteration_crc_fec_defense_vm_20261022_103042"}
SPEC = {
    "R1": {"raw": "raw_samples.csv", "npz": "x_adv_rep0.npz", "filter": lambda r: r["rep"] == "0",
           "key": lambda r: f"{r['group']}|{r['attack']}|{r['profile']}|{r['sample_pos']}",
           "unit": lambda r: (r["attack"], r["profile"], ""), "pred": "attacked_pred", "linf": "linf", "l2": "l2",
           "success": "attack_success", "clean": "clean_correct", "sha": "x_adv_sha256", "lat": "latency_ms",
           "lat_filter": lambda r: True},
    "R2": {"raw": "v3_attack_chain.csv", "npz": "x_adv_v3.npz", "filter": lambda r: True,
           "key": lambda r: (f"{r['attack']}|{r['profile']}|{r['defense']}|{r['true_modulation']}|{r['snr_db']}|"
                             f"fec{int(r['fec_requested'] == 'True')}|{r['burst_index']}"),
           "unit": lambda r: (r["attack"], r["profile"] + ("" if r["defense"] == "none" else f"+{r['defense']}"), ""),
           "pred": "attacked_pred", "linf": "linf", "l2": "l2", "success": "attack_success", "clean": "clean_correct",
           "sha": None, "lat": "attack_latency_ms", "lat_filter": lambda r: True,
           "crc": "attacked_oracle_crc_pass"},
    "R3": {"raw": "raw_bursts.csv", "npz": "x_adv.npz", "filter": lambda r: True,
           "key": lambda r: (f"{r['attack']}|{r['profile']}|{r['true_modulation']}|{r['snr_db']}|"
                             f"fec{int(r['fec_requested'] == 'True')}|{r['burst_index']}"),
           "unit": lambda r: (r["attack"], r["profile"], f"fec{int(r['fec_requested'] == 'True')}"),
           "pred": "attacked_pred", "linf": "linf", "l2": "l2", "success": "attack_success", "clean": "clean_correct",
           "sha": None, "lat": "attack_latency_ms", "lat_filter": lambda r: True,
           "crc": "attacked_oracle_crc_pass", "frame": "attacked_oracle_frame_correct",
           "valid": lambda r: r["payload_degenerate"] != "True"},
    "R4": {"raw": "raw_samples.csv", "npz": "x_adv.npz", "filter": lambda r: r["defense"] == "none",
           "key": lambda r: (f"{r['attack']}|{r['profile']}|{r['true_modulation']}|{r['snr_db']}|"
                             f"fec{int(r['fec_requested'] == 'True')}|{r['burst_index']}"),
           "unit": lambda r: (r["attack"], r["profile"], f"fec{int(r['fec_requested'] == 'True')}"),
           "pred": "attacked_pred_before_defense", "linf": "attack_linf", "l2": "attack_l2",
           "success": "attack_success_before_defense", "clean": "clean_correct", "sha": "x_adv_sha256",
           "lat": "attack_latency_ms", "lat_filter": lambda r: r["defense"] == "none",
           "valid": lambda r: r["payload_degenerate"] != "True"},
}


def rows(run: str, fn: str) -> list:
    with (REPO / run / fn).open(newline="") as f:
        return list(csv.DictReader(f))


def arrays(run: str, fn: str) -> dict:
    z = np.load(REPO / run / fn)
    return {str(k): z[f"a{i}"] for i, k in enumerate(z["keys"])}


def pct(k, n):
    return 100.0 * k / n if n else None


def dist(v):
    v = np.asarray(v, dtype=float)
    return {"median": float(np.median(v)), "mean": float(v.mean()), "p95": float(np.percentile(v, 95))} if len(v) else {}


def compare(rid: str, old: str, new: str, legacy: str):
    sp = SPEC[rid]
    ro, rn, rl = (rows(r, sp["raw"]) for r in (old, new, legacy))
    xn, xl = arrays(new, sp["npz"]), arrays(legacy, sp["npz"])
    idx = {name: {sp["key"](r): r for r in rr if sp["filter"](r)} for name, rr in (("old", ro), ("new", rn), ("legacy", rl))}
    keys = sorted(idx["new"])
    if set(keys) != set(idx["old"]) or set(keys) != set(idx["legacy"]):
        raise SystemExit(f"{rid}: row keys differ between runs")
    units = defaultdict(list)
    for k in keys:
        units[sp["unit"](idx["new"][k])].append(k)
    sem, fid = [], []
    for (atk, prof, fec), ks in sorted(units.items()):
        o = [idx["old"][k] for k in ks]
        n = [idx["new"][k] for k in ks]
        lg = [idx["legacy"][k] for k in ks]
        diffs = [np.abs(xn[k].astype(np.float64) - xl[k].astype(np.float64)) for k in ks]
        bit = sum(int(np.array_equal(xn[k], xl[k])) for k in ks)
        valid = [i for i, k in enumerate(ks) if sp.get("valid", lambda r: True)(n[i])]

        def rate(rr, field, cond=None):
            sel = [rr[i] for i in valid if cond is None or rr[i][cond] == "True"]
            return pct(sum(r[field] == "True" for r in sel), len(sel))

        row = {"run": rid, "attack": atk, "profile": prof, "fec": fec, "n": len(ks),
               "attacked_pred_agreement_new_vs_old": sum(a[sp["pred"]] == b[sp["pred"]] for a, b in zip(n, o)),
               "x_adv_bit_identical_new_vs_legacy": bit,
               "x_adv_sha_identical_new_vs_old": (sum(a[sp["sha"]] == b[sp["sha"]] for a, b in zip(n, o))
                                                  if sp["sha"] else ""),
               "max_abs_x_adv_diff": float(max(d.max() for d in diffs)),
               "mean_abs_x_adv_diff": float(np.mean([d.mean() for d in diffs])),
               "max_abs_linf_diff_new_vs_old": max(abs(float(a[sp["linf"]]) - float(b[sp["linf"]])) for a, b in zip(n, o)),
               "max_abs_l2_diff_new_vs_old": max(abs(float(a[sp["l2"]]) - float(b[sp["l2"]])) for a, b in zip(n, o)),
               "conditional_asr_old_pct": rate(o, sp["success"], sp["clean"]),
               "conditional_asr_new_pct": rate(n, sp["success"], sp["clean"])}
        if sp.get("crc"):
            row["oracle_crc_pass_old_pct"], row["oracle_crc_pass_new_pct"] = rate(o, sp["crc"]), rate(n, sp["crc"])
        if sp.get("frame"):
            row["oracle_frame_correct_old_pct"] = rate(o, sp["frame"])
            row["oracle_frame_correct_new_pct"] = rate(n, sp["frame"])
        sem.append(row)
        fid.append({"run": rid, "attack": atk, "profile": prof, "fec": fec, "n": len(ks),
                    "attacked_pred_identical": sum(a[sp["pred"]] == b[sp["pred"]] for a, b in zip(lg, o)),
                    "linf_identical": sum(a[sp["linf"]] == b[sp["linf"]] for a, b in zip(lg, o)),
                    "l2_identical": sum(a[sp["l2"]] == b[sp["l2"]] for a, b in zip(lg, o)),
                    "sha_identical": (sum(a[sp["sha"]] == b[sp["sha"]] for a, b in zip(lg, o)) if sp["sha"] else "")})
    lat = []
    for name, rr, run in (("old", ro, old), ("new", rn, new)):
        g = defaultdict(list)
        for r in rr:
            if sp["lat_filter"](r):
                g[(r["attack"], r["profile"])].append(float(r[sp["lat"]]))
        for (atk, prof), v in g.items():
            lat.append({"run": rid, "which": name, "attack": atk, "profile": prof, "n": len(v), **dist(v)})
    return sem, fid, lat, ro, rn


def r4_defense(old: str, new: str) -> list:
    """Per attack/profile/FEC/defense: conditional ASR after defense, recovery, oracle CRC / frame correct."""
    out = []
    for which, run in (("old", old), ("new", new)):
        g = defaultdict(lambda: defaultdict(int))
        for r in rows(run, "raw_samples.csv"):
            if r["payload_degenerate"] == "True":
                continue
            c = g[(r["attack"], r["profile"], r["effective_fec"], r["defense"])]
            c["n"] += 1
            cc = r["clean_correct"] == "True"
            c["cc"] += cc
            c["after"] += r["attack_success_after_defense"] == "True"
            c["before"] += r["attack_success_before_defense"] == "True"
            c["rec"] += r["defense_recovery"] == "True"
            c["crc"] += r["oracle_crc_pass"] == "True"
            c["frame"] += r["oracle_frame_correct"] == "True"
        for (atk, prof, fec, d), c in g.items():
            out.append({"which": which, "attack": atk, "profile": prof, "effective_fec": fec, "defense": d,
                        "n": c["n"], "asr_after_pct": pct(c["after"], c["cc"]), "recovery_pct": pct(c["rec"], c["before"]),
                        "oracle_crc_pass_pct": pct(c["crc"], c["n"]), "oracle_frame_correct_pct": pct(c["frame"], c["n"])})
    return out


def write(path: Path, rr: list) -> None:
    fields = []
    for r in rr:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="", lineterminator="\n")
        w.writeheader()
        w.writerows(rr)


def main() -> None:
    runs = load_runs(Path(sys.argv[1]))
    out = REPO / "results" / f"backend_toggle_rerun_comparison_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    sem, fid, lat = [], [], []
    for rid in ("R1", "R2", "R3", "R4"):
        s, f, l, _, _ = compare(rid, OLD[rid], runs[f"{rid}_default"], runs[f"{rid}_legacy"])
        sem += s
        fid += f
        lat += l
    d4 = r4_defense(OLD["R4"], runs["R4_default"])
    write(out / "semantic_comparison.csv", sem)
    write(out / "latency_comparison.csv", lat)
    write(out / "repro_fidelity.csv", fid)
    write(out / "defense_comparison_r4.csv", d4)
    # defense metrics: differences between old and new per unit
    dd = {}
    for r in d4:
        dd.setdefault((r["attack"], r["profile"], r["effective_fec"], r["defense"]), {})[r["which"]] = r
    def_changed = sum(1 for v in dd.values() if any(v["old"][m] != v["new"][m] for m in
                                                     ("asr_after_pct", "recovery_pct", "oracle_crc_pass_pct",
                                                      "oracle_frame_correct_pct")))
    summary = {
        "runs": {"old": OLD, "new": {k: runs[f"{k}_default"] for k in OLD}, "legacy": {k: runs[f"{k}_legacy"] for k in OLD}},
        "rows_compared": sum(r["n"] for r in sem),
        "x_adv_bit_identical_new_vs_legacy": sum(r["x_adv_bit_identical_new_vs_legacy"] for r in sem),
        "attacked_pred_agreement_new_vs_old": sum(r["attacked_pred_agreement_new_vs_old"] for r in sem),
        "max_abs_x_adv_diff": max(r["max_abs_x_adv_diff"] for r in sem),
        "legacy_reproduces_old": all(r["attacked_pred_identical"] == r["n"] and r["linf_identical"] == r["n"]
                                     and r["l2_identical"] == r["n"] and (r["sha_identical"] in ("", r["n"]))
                                     for r in fid),
        "units_with_asr_change": sum(1 for r in sem if r["conditional_asr_old_pct"] != r["conditional_asr_new_pct"]),
        "r4_defense_units": len(dd), "r4_defense_units_with_metric_change": def_changed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(out.relative_to(REPO))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
