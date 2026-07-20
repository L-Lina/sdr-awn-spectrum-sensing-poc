"""
Full-parameter coverage completion (round 14). Purpose: fill the
validation-depth gaps identified in docs/parameter_validation.md section
21.9 -- real (not dummy, not cosmetic-label) backend coverage for every
CLI parameter that was previously only smoke-tested or tested on a small
subset. NOT the formal full-parameter batch (explicitly out of scope).

Stages, run independently, each printing its own combo count/estimate:
  A. All 11 RML2016.10a modulations x attack{none,fgsm,pgd,cw} (previously
     only 3/11 modulations tested with any real attack).
  B. Extended RadioML SNR coverage -- ALL 20 dataset SNR values
     (-20..18 step 2), sensing+AMC (attack=none) for 3 modulations, plus a
     smaller real-attack SNR sweep for 1 modulation (previously only
     4/20 SNR values ever tested, only 2/20 with real attack).
  C. Top-K wide/boundary range (previously only {10,20,30,40} tested).
  D. attack-eps sweep through the real batch pipeline (previously fixed
     at 0.05, or individually verified outside the batch pipeline).
  E. CW c/steps/lr variation (previously always at defaults).
  F. Remaining under-tested flags: attack-temperature, attack-diagnostics,
     segment-hop>1, burst-power-scale-list under current defaults,
     num-bursts=3, checkpoint/device error-path direct-API checks.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.batch_aggregation import run_batch_combos  # noqa: E402
from src.utils.config import ExperimentConfig  # noqa: E402

SEED = 42
DATASET_PATH = "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl"
CHECKPOINT = "external/adversarial-rf/2016.10a_AWN.pkl"
BASE_OUTPUT = Path("results/parameter_coverage_completion")

ALL_MODS = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
ALL_SNRS = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

FIXED = dict(threshold_factor=1.5, sensing_window_size=128, min_region_len=0, merge_gap=0)


def base_cfg_kwargs(run_dir):
    return dict(
        snr=10.0, topk=20,
        threshold_factor=FIXED["threshold_factor"], window_size=128,
        sensing_window_size=FIXED["sensing_window_size"], min_region_len=FIXED["min_region_len"],
        merge_gap=FIXED["merge_gap"], burst_len=600,
        output_dir=str(run_dir), dry_run=True,
        use_real_topk=True, use_real_awn=True, checkpoint=CHECKPOINT, device="cpu",
        attack_eps=0.05, attack_temperature=1.0, attack_diagnostics=False,
        seed=SEED, cw_c=1.0, cw_steps=20, cw_lr=0.01,
        iq_source="radioml", dataset_path=DATASET_PATH,
        embed_snr_margin=20.0, num_bursts=1,
        dataset_mod_list=None, dataset_snr_list=None, sample_index_list=None,
        min_burst_gap=50, max_burst_gap=50, burst_gap_list=None, burst_power_scale_list=None,
        alignment_policy="max-energy", segment_hop=1, awn_preprocess="radioml-native",
    )


def run_stage(stage_name: str, combos: list, build_cfg, estimate_note: str) -> dict:
    stage_dir = BASE_OUTPUT / stage_name
    print(f"\n{'='*70}\n[stage {stage_name}] {len(combos)} combos. {estimate_note}\n{'='*70}")
    t0 = time.time()
    result = run_batch_combos(stage_dir, combos, build_cfg)
    elapsed = time.time() - t0
    print(f"[stage {stage_name}] done in {elapsed:.1f}s -- ok={result['n_ok']} sensing_failed={result['n_sensing_failed']} error={result['n_error']}")
    return {"result": result, "elapsed": elapsed}


# ---------------------------------------------------------------------------
# Stage A: all 11 modulations x attack{none,fgsm,pgd,cw}
# ---------------------------------------------------------------------------

def stage_A_all_modulations_x_attack():
    snrs = [0, 18]
    attacks = ["none", "fgsm", "pgd", "cw"]
    combos = [
        {"dataset_mod": mod, "dataset_snr": snr, "attack": attack}
        for mod in ALL_MODS for snr in snrs for attack in attacks
    ]

    def build_cfg(combo, run_dir):
        use_real_attack = combo["attack"] != "none"
        kwargs = base_cfg_kwargs(run_dir)
        kwargs.update(mod=combo["dataset_mod"], attack=combo["attack"],
                      use_real_attack=use_real_attack,
                      dataset_mod=combo["dataset_mod"], dataset_snr=combo["dataset_snr"], sample_index=0)
        return ExperimentConfig(**kwargs)

    return run_stage("A_all_modulations_x_attack", combos, build_cfg,
                      f"11 modulations x 2 SNR x 4 attacks = {len(combos)}, real AWN+attack+Top-K(K=20), est ~{len(combos)*1.8:.0f}s")


# ---------------------------------------------------------------------------
# Stage B: extended SNR range
# ---------------------------------------------------------------------------

def stage_B1_full_snr_sensing():
    mods = ["QPSK", "BPSK", "QAM16"]
    combos = [{"dataset_mod": mod, "dataset_snr": snr} for mod in mods for snr in ALL_SNRS]

    def build_cfg(combo, run_dir):
        kwargs = base_cfg_kwargs(run_dir)
        kwargs.update(mod=combo["dataset_mod"], attack="none", use_real_attack=False,
                      dataset_mod=combo["dataset_mod"], dataset_snr=combo["dataset_snr"], sample_index=0)
        return ExperimentConfig(**kwargs)

    return run_stage("B1_full_snr_sensing", combos, build_cfg,
                      f"3 modulations x ALL 20 SNR values = {len(combos)}, real AWN, attack=none, est ~{len(combos)*1.5:.0f}s")


def stage_B2_full_snr_with_attack():
    combos = [{"dataset_snr": snr} for snr in ALL_SNRS]

    def build_cfg(combo, run_dir):
        kwargs = base_cfg_kwargs(run_dir)
        kwargs.update(mod="BPSK", attack="fgsm", use_real_attack=True,
                      dataset_mod="BPSK", dataset_snr=combo["dataset_snr"], sample_index=0)
        return ExperimentConfig(**kwargs)

    return run_stage("B2_full_snr_with_attack", combos, build_cfg,
                      f"BPSK x ALL 20 SNR values x fgsm = {len(combos)}, real AWN+attack+Top-K, est ~{len(combos)*1.8:.0f}s")


# ---------------------------------------------------------------------------
# Stage C: Top-K wide/boundary range
# ---------------------------------------------------------------------------

def stage_C_topk_range():
    values = [1, 5, 10, 20, 30, 40, 64, 100, 127, 128]
    combos = [{"topk": k} for k in values]

    def build_cfg(combo, run_dir):
        kwargs = base_cfg_kwargs(run_dir)
        kwargs.update(mod="BPSK", attack="fgsm", use_real_attack=True, topk=combo["topk"],
                      dataset_mod="BPSK", dataset_snr=18, sample_index=0)
        return ExperimentConfig(**kwargs)

    return run_stage("C_topk_range", combos, build_cfg,
                      f"topk in {values}, BPSK/snr18/idx0/fgsm, real AWN+attack+Top-K, est ~{len(combos)*1.8:.0f}s")


# ---------------------------------------------------------------------------
# Stage D: attack-eps sweep
# ---------------------------------------------------------------------------

def stage_D_attack_eps():
    eps_values = [0.001, 0.01, 0.03, 0.05, 0.1, 0.3, 1.0]
    combos = [{"attack": a, "attack_eps": e} for a in ["fgsm", "pgd"] for e in eps_values]

    def build_cfg(combo, run_dir):
        kwargs = base_cfg_kwargs(run_dir)
        kwargs.update(mod="BPSK", attack=combo["attack"], use_real_attack=True, attack_eps=combo["attack_eps"],
                      dataset_mod="BPSK", dataset_snr=18, sample_index=0)
        return ExperimentConfig(**kwargs)

    return run_stage("D_attack_eps", combos, build_cfg,
                      f"attack_eps in {eps_values} x {{fgsm,pgd}} = {len(combos)}, est ~{len(combos)*1.8:.0f}s")


# ---------------------------------------------------------------------------
# Stage E: CW knob variation
# ---------------------------------------------------------------------------

def stage_E_cw_knobs():
    combos = [
        {"cw_c": 1.0, "cw_steps": 20, "cw_lr": 0.01},  # baseline
        {"cw_c": 0.1, "cw_steps": 20, "cw_lr": 0.01},
        {"cw_c": 10.0, "cw_steps": 20, "cw_lr": 0.01},
        {"cw_c": 1.0, "cw_steps": 5, "cw_lr": 0.01},
        {"cw_c": 1.0, "cw_steps": 50, "cw_lr": 0.01},
        {"cw_c": 1.0, "cw_steps": 20, "cw_lr": 0.001},
        {"cw_c": 1.0, "cw_steps": 20, "cw_lr": 0.1},
    ]

    def build_cfg(combo, run_dir):
        kwargs = base_cfg_kwargs(run_dir)
        kwargs.update(mod="BPSK", attack="cw", use_real_attack=True,
                      cw_c=combo["cw_c"], cw_steps=combo["cw_steps"], cw_lr=combo["cw_lr"],
                      dataset_mod="BPSK", dataset_snr=18, sample_index=0)
        return ExperimentConfig(**kwargs)

    return run_stage("E_cw_knobs", combos, build_cfg,
                      f"cw_c/cw_steps/cw_lr OFAT around defaults = {len(combos)}, est ~{len(combos)*2:.0f}s")


if __name__ == "__main__":
    stage_A_all_modulations_x_attack()
    stage_B1_full_snr_sensing()
    stage_B2_full_snr_with_attack()
    stage_C_topk_range()
    stage_D_attack_eps()
    stage_E_cw_knobs()
