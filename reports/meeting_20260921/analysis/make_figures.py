#!/usr/bin/env python3
"""Figures 1-8 for the meeting_20260921 report package.

Reads only the derived tables in ../tables/ (produced by phase_b_analysis.py from the
formal CSVs) and writes static PNGs to ../figures/.
Palette: CPU = blue, NPU = orange; Top-K ordinal blue ramp; neutral grays for chrome.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PKG = Path(__file__).resolve().parents[1]
T = PKG / "tables"
FIG = PKG / "figures"
FIG.mkdir(exist_ok=True)

INK, INK2, MUTED, GRID = "#1f2328", "#59636e", "#8b949e", "#e6e9ed"
CPU, NPU = "#2a78d6", "#eb6834"
KRAMP = {10: "#86b6ef", 20: "#5598e7", 30: "#256abf", 40: "#104281"}
SENSE, TOPK_C = "#c3cad2", "#7c8792"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f1f6fd", "#86b6ef", "#256abf", "#0d3a73"])
BCOL = {"CPU": CPU, "NPU": NPU}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 8.5,
    "lines.linewidth": 2.0, "lines.markersize": 5.5,
})

asum = pd.read_csv(T / "attack_summary.csv")
tr = pd.read_csv(T / "cpu_npu_transfer_comparison.csv")
dsum = pd.read_csv(T / "defense_summary.csv")
clean = pd.read_csv(T / "clean_summary.csv")
ctop = pd.read_csv(T / "clean_topk_summary.csv")
pooled = pd.read_csv(T / "defense_pooled_by_topk.csv")
a_mod = pd.read_csv(T / "attack_by_modulation.csv")
d_mod = pd.read_csv(T / "defense_by_modulation.csv")
a_snr = pd.read_csv(T / "attack_by_snr.csv")
d_snr = pd.read_csv(T / "defense_by_snr.csv")
lat = pd.read_csv(T / "latency_summary_cpu_npu.csv").set_index("stage")
agl = pd.read_csv(T / "attack_generation_latency.csv")

PROFILES = asum[asum.backend == "CPU"].profile.tolist()
NONIDENT = set(tr[~tr.backend_attributable].profile)
NAMES = {"fgsm": "FGSM", "bim": "BIM", "pgd": "PGD", "mifgsm": "MI-FGSM", "difgsm": "DI-FGSM",
         "vmifgsm": "VMI-FGSM", "vnifgsm": "VNI-FGSM", "rfgsm": "RFGSM", "tpgd": "TPGD",
         "cw": "CW", "deepfool": "DeepFool", "fab": "FAB", "square": "Square", "apgd": "APGD",
         "apgdt": "APGD-T", "autoattack": "AutoAttack", "ead": "EAD"}
MULTI = {"fgsm", "pgd", "fab"}


def plabel(p: str, dagger: bool = True) -> str:
    atk, prof = p.split("/")
    s = NAMES[atk]
    if atk in MULTI:
        s += " ε=" + prof.replace("eps", "")
    if dagger and p in NONIDENT:
        s += " †"
    return s


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def foot(fig, text, y=-0.01):
    fig.text(0.01, y, text, fontsize=7.5, color=INK2, ha="left", va="top", wrap=True)


DAGGER_NOTE = ("† adversarial inputs are not bit-identical between the CPU and NPU runs for this profile; "
               "the CPU–NPU difference cannot be attributed to the evaluation backend alone.")


# ------------------------------------------------------------------- Figure 1
def fig1():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    s = clean[clean.group_by == "snr_db"].copy()
    s["snr"] = s.group_value.astype(int)
    a.plot(s.snr, s.cpu_acc_pct, color=CPU, marker="o", label="CPU (PyTorch)")
    a.plot(s.snr, s.npu_acc_pct, color=NPU, marker="s", label="NPU (Hailo-8, quantized)")
    a.set(xlabel="SNR (dB)", ylabel="Clean accuracy (%)", ylim=(0, 100), xticks=range(-20, 20, 4))
    a.set_title("(a) Clean accuracy by SNR (n = 110 per SNR)")
    a.legend(loc="upper left")
    m = clean[clean.group_by == "modulation"].copy()
    m = m.sort_values("cpu_acc_pct", ascending=True).reset_index(drop=True)
    y = np.arange(len(m))
    b.barh(y + 0.2, m.cpu_acc_pct, 0.38, color=CPU, label="CPU")
    b.barh(y - 0.2, m.npu_acc_pct, 0.38, color=NPU, label="NPU")
    b.set_yticks(y, m.group_value)
    b.set(xlabel="Clean accuracy (%)", xlim=(0, 100))
    b.set_title("(b) Clean accuracy by modulation (n = 200 each)")
    for i, r in m.iterrows():
        d = r.acc_diff_pp_npu_minus_cpu
        b.text(max(r.cpu_acc_pct, r.npu_acc_pct) + 1.5, i, f"{d:+.1f} pp", va="center", fontsize=8,
               color=INK if abs(d) >= 10 else INK2, fontweight="bold" if abs(d) >= 10 else "normal")
    b.legend(loc="lower right")
    b.grid(axis="y", visible=False)
    o = clean[clean.group_by == "overall"].iloc[0]
    foot(fig, f"Overall (n = 2200): CPU {o.cpu_acc_pct:.2f}%, NPU {o.npu_acc_pct:.2f}%, difference "
              f"{o.acc_diff_pp_npu_minus_cpu:+.2f} percentage points (NPU − CPU). Labels in (b): NPU − CPU in pp.")
    save(fig, "fig1_clean_accuracy.png")


# ------------------------------------------------------------------- Figure 2
def fig2():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 8.6), sharey=True,
                               gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.06})
    y = np.arange(len(PROFILES))[::-1]
    c = asum[asum.backend == "CPU"].set_index("profile").loc[PROFILES]
    n = asum[asum.backend == "NPU"].set_index("profile").loc[PROFILES]
    a.barh(y + 0.2, c.cond_asr_pct, 0.38, color=CPU, label="CPU (white-box, PyTorch)")
    a.barh(y - 0.2, n.cond_asr_pct, 0.38, color=NPU, label="NPU (surrogate transfer to Hailo-8)")
    for yy, v in zip(y, c.cond_asr_pct):
        a.text(v + 1, yy + 0.2, f"{v:.1f}", va="center", fontsize=7, color=INK2)
    for yy, v in zip(y, n.cond_asr_pct):
        a.text(v + 1, yy - 0.2, f"{v:.1f}", va="center", fontsize=7, color=INK2)
    a.set_yticks(y, [plabel(p) for p in PROFILES])
    a.set(xlabel="Conditional ASR (%)", xlim=(0, 112))
    a.set_title("(a) Conditional attack success rate by profile")
    a.legend(loc="upper right")
    a.grid(axis="y", visible=False)
    b.scatter(c.mean_linf * 1e3, y + 0.2, color=CPU, s=26, zorder=3)
    b.scatter(n.mean_linf * 1e3, y - 0.2, color=NPU, s=26, zorder=3, marker="s")
    b.set(xlabel="Mean L∞ of the perturbation (×10⁻³, raw I/Q units)")
    b.set_title("(b) Perturbation size (as measured)")
    b.grid(axis="y", visible=True)
    b.tick_params(axis="y", left=False)
    foot(fig, "Conditional ASR = successful attacks / clean-correct samples (2200 samples per profile; "
              "denominator 1298 for CPU, 1202 for NPU).  " + DAGGER_NOTE, y=0.005)
    save(fig, "fig2_attack_asr_by_profile.png")


# ------------------------------------------------------------------- Figure 3
def fig3():
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for ax, atk in zip(axs, ["fgsm", "pgd", "fab"]):
        for be, mk in (("CPU", "o"), ("NPU", "s")):
            d = asum[(asum.backend == be) & (asum.attack == atk)].copy()
            d["eps"] = d.attack_profile.str.replace("eps", "").astype(float)
            d = d.sort_values("eps")
            ax.plot(d.eps, d.cond_asr_pct, color=BCOL[be], marker=mk, label=be)
            ax.fill_between(d.eps, d.cond_asr_ci95_lo_pct, d.cond_asr_ci95_hi_pct, color=BCOL[be], alpha=0.12, lw=0)
        ax.set_title(NAMES[atk] + (" †" if atk == "pgd" else ""))
        ax.set_xlabel("ε (attack-interface value)")
        ax.set_xscale("log")
        ax.set_xticks([0.005, 0.01, 0.03, 0.05], ["0.005", "0.01", "0.03", "0.05"])
        ax.minorticks_off()
    axs[0].set_ylabel("Conditional ASR (%)")
    axs[0].set_ylim(0, 100)
    axs[0].legend(loc="upper left")
    foot(fig, "Shaded band: 95% Wilson interval, treating samples as independent. FAB was evaluated at ε = 0.005, 0.01, 0.03 only. "
              "† PGD adversarial inputs differ between the CPU and NPU runs (not bit-identical).", y=0.0)
    save(fig, "fig3_epsilon_curves.png")


# ------------------------------------------------------------------- heatmap helper
def heat(ax, M, xt, yt, vmin=0, vmax=100, fs=7, fmt="{:.0f}"):
    im = ax.imshow(M, cmap=SEQ, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xt)), xt)
    ax.set_yticks(range(len(yt)), yt)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs,
                    color="white" if (v - vmin) / (vmax - vmin) > 0.55 else INK)
    return im


# ------------------------------------------------------------------- Figure 4
def fig4():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 9), sharey=True, gridspec_kw={"wspace": 0.05})
    Ks = [10, 20, 30, 40]
    for ax, be in zip(axs, ["CPU", "NPU"]):
        d = dsum[dsum.backend == be].pivot(index="profile", columns="topk", values="recovery_pct").loc[PROFILES]
        im = heat(ax, d.to_numpy(), [f"K={k}" for k in Ks], [plabel(p) for p in PROFILES], vmin=0, vmax=70)
        ax.set_title(f"({'a' if be == 'CPU' else 'b'}) {be}: recovery rate (%)")
    cb = fig.colorbar(im, ax=axs, shrink=0.5, pad=0.02, aspect=30)
    cb.set_label("Recovery rate (%)", color=INK2)
    cb.outline.set_visible(False)
    foot(fig, "Recovery rate = recovered / successful attacks. The successful-attack sets differ between CPU and NPU, "
              "so rates are not paired.  " + DAGGER_NOTE, y=0.06)
    save(fig, "fig4_topk_recovery_heatmap.png")


# ------------------------------------------------------------------- Figure 5
def fig5():
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 4.2))
    Ks = [10, 20, 30, 40]
    p = pooled
    for be, mk in (("CPU", "o"), ("NPU", "s")):
        d = p[p.backend == be].sort_values("topk")
        axs[0].plot(d.topk, d.clean_degradation_pct, color=BCOL[be], marker=mk, label=be)
        axs[1].plot(d.topk, d.recovery_pct, color=BCOL[be], marker=mk, label=be)
        axs[1].fill_between(d.topk, d.recovery_ci95_lo_pct, d.recovery_ci95_hi_pct, color=BCOL[be], alpha=0.15, lw=0)
        axs[2].plot(d.topk, d.retention_pct, color=BCOL[be], marker=mk, label=f"{be} with Top-K")
        axs[2].axhline(d.retention_nodef_pct.iloc[0], color=BCOL[be], ls=":", lw=1.6, label=f"{be} no defense")
    axs[0].set(ylabel="Clean degradation (%)", ylim=(0, 100))
    axs[0].set_title("(a) Clean degradation")
    axs[1].set(ylabel="Recovery rate (%)", ylim=(0, 100))
    axs[1].set_title("(b) Recovery, pooled over 24 profiles")
    axs[2].set(ylabel="Retention (%)", ylim=(0, 100))
    axs[2].set_title("(c) Retention after attack + defense")
    for ax in axs:
        ax.set_xlabel("Top-K")
        ax.set_xticks(Ks)
    axs[0].legend(loc="upper right")
    axs[2].legend(loc="lower right", fontsize=7.5)
    foot(fig, "Clean degradation = clean-correct samples misclassified after Top-K / clean-correct samples. "
              "Recovery = recovered / successful attacks. Retention = samples correct both clean and after attack (+ defense) / clean-correct samples; 'no defense' = 100% − conditional ASR. "
              "Pooled over all 24 attack profiles; band in (b): 95% Wilson interval, treating samples as independent.", y=0.0)
    save(fig, "fig5_topk_tradeoff.png")


# ------------------------------------------------------------------- Figure 6
def fig6():
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True)
    P = "ALL/ALL_24"
    A = a_snr[a_snr.profile == P]
    D = d_snr[d_snr.profile == P]
    a = axs[0, 0]
    for be, mk in (("CPU", "o"), ("NPU", "s")):
        d = A[A.backend == be].sort_values("snr_db")
        a.plot(d.snr_db, d.clean_acc_pct, color=BCOL[be], marker=mk, ls="--", lw=1.4, ms=4, label=f"{be} clean")
        a.plot(d.snr_db, d.attacked_acc_pct, color=BCOL[be], marker=mk, label=f"{be} attacked")
    a.set(ylabel="Accuracy (%)", ylim=(0, 100))
    a.set_title("(a) Accuracy by SNR, attacked = mean over 24 profiles")
    a.legend(ncol=2, loc="upper left")
    b = axs[0, 1]
    for be, mk in (("CPU", "o"), ("NPU", "s")):
        d = A[A.backend == be].sort_values("snr_db")
        b.plot(d.snr_db, d.cond_asr_pct, color=BCOL[be], marker=mk, label=be)
        b.fill_between(d.snr_db, d.cond_asr_ci95_lo_pct, d.cond_asr_ci95_hi_pct, color=BCOL[be], alpha=0.12, lw=0)
    b.set(ylabel="Conditional ASR (%)", ylim=(0, 100))
    b.set_title("(b) Conditional ASR by SNR (pooled over 24 profiles)")
    b.legend(loc="lower left")
    for ax, be, lab in ((axs[1, 0], "CPU", "(c)"), (axs[1, 1], "NPU", "(d)")):
        for k in (10, 20, 30, 40):
            d = D[(D.backend == be) & (D.topk == k)].sort_values("snr_db")
            ax.plot(d.snr_db, d.recovery_pct, color=KRAMP[k], marker="o", ms=3.5, lw=1.8, label=f"K={k}")
        ax.set(ylabel="Recovery rate (%)", ylim=(0, 60), xlabel="SNR (dB)", xticks=range(-20, 20, 4))
        ax.set_title(f"{lab} {be}: Top-K recovery by SNR (pooled over 24 profiles)")
        ax.legend(ncol=4, loc="upper left")
    foot(fig, "At low SNR (≤ −14 dB) clean accuracy is near chance (≈10–12% for 11 classes); the conditional metrics there rest on few "
              "clean-correct samples (11–13 per profile out of 110; 264–312 in the 24-profile pooled cell).", y=0.03)
    save(fig, "fig6_snr_analysis.png")


# ------------------------------------------------------------------- Figure 7
def fig7():
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 6.4), sharey=True, gridspec_kw={"wspace": 0.05})
    P = "ALL/ALL_24"
    order = clean[clean.group_by == "modulation"].sort_values("cpu_acc_pct", ascending=False).group_value.tolist()
    cols = ["Clean acc.", "Cond. ASR", "Rec. K=10", "Rec. K=20", "Rec. K=30", "Rec. K=40"]
    for ax, be, lab in zip(axs, ["CPU", "NPU"], ["(a)", "(b)"]):
        A = a_mod[(a_mod.backend == be) & (a_mod.profile == P)].set_index("modulation").loc[order]
        D = d_mod[(d_mod.backend == be) & (d_mod.profile == P)]
        M = np.column_stack([A.clean_acc_pct.to_numpy(), A.cond_asr_pct.to_numpy()] +
                            [D[D.topk == k].set_index("modulation").loc[order].recovery_pct.to_numpy() for k in (10, 20, 30, 40)])
        im = heat(ax, M, cols, order, vmin=0, vmax=100, fs=8.5, fmt="{:.1f}")
        ax.set_title(f"{lab} {be}")
        ax.tick_params(axis="x", labelrotation=30)
    cb = fig.colorbar(im, ax=axs, shrink=0.6, pad=0.02, aspect=30)
    cb.set_label("%", color=INK2)
    cb.outline.set_visible(False)
    foot(fig, "ASR and recovery are pooled over all 24 attack profiles. Clean-correct samples per profile (CPU/NPU): "
              "WBFM 28/19, AM-DSB 130/133, GFSK 137/73; small denominators make WBFM cells unstable.", y=-0.02)
    save(fig, "fig7_modulation_analysis.png")


# ------------------------------------------------------------------- Figure 8
def fig8():
    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.05], height_ratios=[1, 1], hspace=0.42, wspace=0.28)
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[1, 0])
    c = fig.add_subplot(gs[:, 1])

    # (a) inference / Top-K latency
    stages = [("clean_infer_ms", "Clean inference"), ("attacked_infer_ms", "Attacked inference"),
              ("defended_infer_ms", "Defended inference"), ("topk_transform_ms", "Top-K transform")]
    y = np.arange(len(stages))[::-1]
    for be, off in (("cpu", 0.2), ("npu", -0.2)):
        col = CPU if be == "cpu" else NPU
        med = [lat.loc[s, f"{be}_median_ms"] for s, _ in stages]
        p95 = [lat.loc[s, f"{be}_p95_ms"] for s, _ in stages]
        p99 = [lat.loc[s, f"{be}_p99_ms"] for s, _ in stages]
        a.barh(y + off, med, 0.36, color=col, label=be.upper() + " median")
        a.scatter(p95, y + off, marker="|", s=90, color=INK, zorder=4, linewidths=1.6)
        a.scatter(p99, y + off, marker="x", s=24, color=INK, zorder=4, linewidths=1.2)
    for yy, (s, _) in zip(y, stages):
        sp = lat.loc[s, "speedup_median"]
        a.text(4.55, yy, f"{sp:.2f}×", va="center", ha="right", fontsize=9, fontweight="bold")
    a.set_yticks(y, [n for _, n in stages])
    a.set(xlim=(0, 4.6), xticks=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], xlabel="ms")
    a.set_title("(a) Per-stage latency (median)")
    a.grid(axis="y", visible=False)
    h, l = a.get_legend_handles_labels()
    h += [Line2D([], [], marker="|", color=INK, ls="", ms=9), Line2D([], [], marker="x", color=INK, ls="", ms=5)]
    l += ["p95", "p99"]
    a.legend(h, l, loc="lower right", bbox_to_anchor=(0.9, 0.0), ncol=2, fontsize=8)
    a.text(4.55, len(stages) - 0.45, "median CPU / NPU", ha="right", fontsize=8, color=INK2, va="bottom")

    # (b) stacked end-to-end (mean)
    labels, sense, topk, infer, cols_, tot = [], [], [], [], [], []
    for pipe, ekey, ikey in (("Clean", "e2e_clean_ms", "clean_infer_ms"), ("Defended", "e2e_defended_ms", "defended_infer_ms")):
        for be in ("cpu", "npu"):
            s_ = lat.loc["sensing_pre_ms", f"{be}_mean_ms"]
            i_ = lat.loc[ikey, f"{be}_mean_ms"]
            e_ = lat.loc[ekey, f"{be}_mean_ms"]
            labels.append(f"{pipe} · {be.upper()}")
            sense.append(s_)
            infer.append(i_)
            topk.append(e_ - s_ - i_)
            cols_.append(CPU if be == "cpu" else NPU)
            tot.append(e_)
    yy = np.arange(len(labels))[::-1]
    b.barh(yy, sense, 0.55, color=SENSE)
    b.barh(yy, topk, 0.55, left=sense, color=TOPK_C)
    b.barh(yy, infer, 0.55, left=np.array(sense) + np.array(topk), color=cols_)
    for i in range(len(labels)):
        b.text(tot[i] + 0.12, yy[i], f"{tot[i]:.2f} ms", va="center", fontsize=8.5)
    for pipe_i in (0, 2):
        sp = tot[pipe_i] / tot[pipe_i + 1]
        b.text(10.4, (yy[pipe_i] + yy[pipe_i + 1]) / 2, f"mean CPU / NPU\n{sp:.2f}×", ha="right", va="center",
               fontsize=8.5, fontweight="bold")
    b.set_yticks(yy, labels)
    b.set(xlim=(0, 10.5), xlabel="ms (mean)")
    b.set_title("(b) End-to-end online latency, mean")
    b.grid(axis="y", visible=False)
    b.legend(handles=[Patch(color=SENSE, label="Sensing (5 pre-inference stages)"),
                      Patch(color=TOPK_C, label="Top-K (recorded pipeline − inference)"),
                      Patch(color=CPU, label="Inference, CPU"), Patch(color=NPU, label="Inference, NPU")],
             loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)

    # (c) attack generation latency by profile (offline)
    yy = np.arange(len(PROFILES))[::-1]
    for be, off in (("CPU", 0.2), ("NPU", -0.2)):
        d = agl[agl.backend == be].set_index("profile").loc[PROFILES]
        c.hlines(yy + off, d.median_ms, d.p95_ms, color=BCOL[be], lw=1.4, alpha=0.6)
        c.scatter(d.median_ms, yy + off, color=BCOL[be], s=22, zorder=3, marker="o" if be == "CPU" else "s",
                  label=f"{be} run: median")
        c.scatter(d.p95_ms, yy + off, color="white", edgecolor=BCOL[be], s=22, zorder=3, linewidths=1.3,
                  marker="o" if be == "CPU" else "s", label=f"{be} run: p95")
    c.set_xscale("log")
    c.set_yticks(yy, [plabel(p, dagger=False) for p in PROFILES])
    c.set(xlabel="Attack generation time per sample (ms, log scale)")
    c.set_title("(c) Attack generation latency (offline, PyTorch surrogate on host CPU)")
    c.legend(loc="upper right", fontsize=8)
    c.grid(axis="y", visible=True)
    foot(fig, "Sensing = embedding, energy detection, region post-processing, segmentation alignment, AWN pre-processing (all host-side in both runs). "
              "Top-K segment in (b) = recorded defense_pipeline_ms − defended inference (includes the Top-K transform and a small un-itemised timing overhead). "
              "In (c) both runs generate attacks on the host CPU; the difference between the two runs is run-to-run, not an NPU effect.", y=-0.06)
    save(fig, "fig8_latency.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
