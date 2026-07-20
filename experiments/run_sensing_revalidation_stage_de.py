"""
Stage D (merge-gap, with REAL multi-region calibrated data) and Stage E
(burst/stream parameter checks) of the post-alignment-fix sensing
revalidation (round 13) -- see run_sensing_revalidation.py's module
docstring for the shared context (max-energy/radioml-native, attack=none,
real Top-K not exercised, not the formal batch).

Stage D calibration (done interactively, not re-derived here): at
sensing_window_size=128 (this round's baseline), a true inter-burst gap of
50 samples (the multi-burst default) already merges into ONE region at
merge_gap=0 -- not usable for a merge-gap test. A true gap of 150 samples
(via --burst-gap-list 50,150) gives exactly 2 SEPARATE detected regions at
merge_gap=0, with a measured inter-region gap of 36 samples (94-242, then
278-512) -- so merge_gap in {0,1,5,20} should stay separate and {64,128}
should merge, giving a real, meaningful transition inside the tested range
(not guessed).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.batch_aggregation import run_batch_combos  # noqa: E402
from src.utils.config import ExperimentConfig  # noqa: E402
from src.utils.csv_writer import write_summary_csv  # noqa: E402

SEED = 42
DATASET_PATH = "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl"
CHECKPOINT = "external/adversarial-rf/2016.10a_AWN.pkl"
BASE_OUTPUT = Path("results/sensing_revalidation_after_alignment")


def run_stage(stage_name: str, combos: list, build_cfg, estimate_note: str, enrich=None) -> dict:
    stage_dir = BASE_OUTPUT / stage_name
    print(f"\n{'='*70}\n[stage {stage_name}] {len(combos)} combos. {estimate_note}\n{'='*70}")
    t0 = time.time()
    result = run_batch_combos(stage_dir, combos, build_cfg)
    elapsed = time.time() - t0
    print(f"[stage {stage_name}] done in {elapsed:.1f}s -- ok={result['n_ok']} sensing_failed={result['n_sensing_failed']} error={result['n_error']}")

    with open(stage_dir / "batch_summary.csv") as f:
        batch_rows = list(csv.DictReader(f))
    enriched_rows = [{**r, **(enrich(r) if enrich else {})} for r in batch_rows]
    write_summary_csv(stage_dir / "sensing_parameter_summary.csv", enriched_rows)

    failures = [r for r in enriched_rows if r["run_status"] != "ok"]
    if failures:
        write_summary_csv(stage_dir / "failures.csv", failures)
    else:
        print(f"[stage {stage_name}] no failures -- failures.csv not written")

    return {"result": result, "enriched_rows": enriched_rows, "elapsed": elapsed}


# ---------------------------------------------------------------------------
# Stage D: merge-gap, with a calibrated 2-burst setup that gives real,
# separate detected regions at merge_gap=0.
# ---------------------------------------------------------------------------

def _enrich_multi_burst(row: dict) -> dict:
    """bursts_summary.csv/regions_summary.csv-derived fields, since a
    multi-burst combo has >1 truth burst / detected region (batch_summary's
    aggregate ratios apply across all bursts/regions in the run, not a
    single one)."""
    enriched = {"num_truth_bursts_check": None, "num_detected_regions_check": None,
                "region_starts_ends": None, "inter_region_gap_min": None}
    regions_path = Path(row["output_dir"]) / "regions_summary.csv"
    if not regions_path.exists():
        return enriched
    with open(regions_path) as f:
        rrows = list(csv.DictReader(f))
    starts_ends = [(int(r["detected_start"]), int(r["detected_end"])) for r in rrows]
    starts_ends.sort()
    gaps = [starts_ends[i + 1][0] - starts_ends[i][1] for i in range(len(starts_ends) - 1)]
    enriched["num_detected_regions_check"] = len(rrows)
    enriched["region_starts_ends"] = str(starts_ends)
    enriched["inter_region_gap_min"] = min(gaps) if gaps else None
    return enriched


def stage_D_merge_gap():
    values = [0, 1, 5, 20, 64, 128]
    mod_pairs = [("BPSK", "QPSK"), ("QPSK", "QAM16"), ("QAM16", "BPSK")]
    combos = [
        {"mod_a": a, "mod_b": b, "merge_gap": mg, "burst_gap_list": "50,150"}
        for (a, b) in mod_pairs for mg in values
    ]

    def build_cfg(combo, run_dir):
        return ExperimentConfig(
            snr=10.0, mod="BPSK", attack="none", topk=50,
            threshold_factor=1.5, window_size=128, sensing_window_size=128,
            min_region_len=0, merge_gap=combo["merge_gap"], burst_len=600,
            output_dir=str(run_dir), dry_run=True,
            use_real_topk=False, use_real_awn=True,
            checkpoint=CHECKPOINT, device="cpu",
            attack_eps=0.03, use_real_attack=False, attack_temperature=1.0, attack_diagnostics=False,
            seed=SEED, cw_c=1.0, cw_steps=20, cw_lr=0.01,
            iq_source="radioml", dataset_path=DATASET_PATH,
            dataset_mod=None, dataset_snr=None, sample_index=None,
            embed_snr_margin=20.0,
            num_bursts=2,
            dataset_mod_list=[combo["mod_a"], combo["mod_b"]],
            dataset_snr_list=[18, 18], sample_index_list=[0, 1],
            min_burst_gap=50, max_burst_gap=50,
            burst_gap_list=[50, 150], burst_power_scale_list=None,
            alignment_policy="max-energy", segment_hop=1, awn_preprocess="radioml-native",
        )

    return run_stage("D_merge_gap", combos, build_cfg,
                      f"merge_gap in {values} x {len(mod_pairs)} mod-pairs (calibrated true gap=150, "
                      f"measured inter-region gap=36 at merge_gap=0 -- expect separate below 36, merged above)",
                      enrich=_enrich_multi_burst)


# ---------------------------------------------------------------------------
# Stage E: burst/stream parameter checks -- targeted, not a sweep.
# ---------------------------------------------------------------------------

def stage_E_burst_stream_checks():
    """
    E1: burst-len boundary/normal (synthetic mode only -- radioml mode's
        burst length is fixed by the dataset sample itself, 128 samples;
        --burst-len only affects the synthetic generator).
    E2: n_samples -- no CLI flag exists (ExperimentConfig.n_samples has no
        argparse flag; confirmed via grep in Part A). Recorded as NOT
        independently testable via CLI, only via direct ExperimentConfig
        construction -- one direct-API check included.
    E3: burst start reproducibility under a fixed seed (radioml single-
        burst mode) -- two independent processes, same params.
    E4: single-burst vs multi-burst mode, same underlying samples.
    E5: embed-snr-margin sweep (reasonable + boundary values).
    """
    results = {}

    # E1: burst-len boundary + normal (synthetic mode, since it's the only
    # mode --burst-len actually controls)
    print(f"\n{'='*70}\n[stage E1_burst_len] 4 combos (synthetic mode boundary/normal values)\n{'='*70}")
    combos = [{"burst_len": v} for v in [1, 128, 600, 4096]]

    def build_cfg_e1(combo, run_dir):
        return ExperimentConfig(
            snr=10.0, mod="BPSK", attack="none", topk=50,
            threshold_factor=1.5, window_size=128, sensing_window_size=128,
            min_region_len=0, merge_gap=0, burst_len=combo["burst_len"],
            output_dir=str(run_dir), dry_run=True,
            use_real_topk=False, use_real_awn=True,
            checkpoint=CHECKPOINT, device="cpu",
            attack_eps=0.03, use_real_attack=False, attack_temperature=1.0, attack_diagnostics=False,
            seed=SEED, cw_c=1.0, cw_steps=20, cw_lr=0.01,
            iq_source="synthetic",
            embed_snr_margin=20.0, num_bursts=1,
            alignment_policy="naive", segment_hop=1, awn_preprocess="legacy-unit-power",
        )
    results["E1_burst_len"] = run_stage("E1_burst_len", combos, build_cfg_e1, "burst_len in [1,128,600,4096]")

    # E2: n_samples -- direct-API only, no CLI flag
    print(f"\n{'='*70}\n[stage E2_n_samples] 3 combos, direct ExperimentConfig(n_samples=...) since no CLI flag exists\n{'='*70}")
    combos = [{"n_samples": v} for v in [2048, 8192, 16384]]

    def build_cfg_e2(combo, run_dir):
        return ExperimentConfig(
            snr=10.0, mod="BPSK", attack="none", topk=50,
            threshold_factor=1.5, window_size=128, sensing_window_size=128,
            min_region_len=0, merge_gap=0, burst_len=600,
            output_dir=str(run_dir), dry_run=True,
            use_real_topk=False, use_real_awn=True,
            checkpoint=CHECKPOINT, device="cpu",
            attack_eps=0.03, use_real_attack=False, attack_temperature=1.0, attack_diagnostics=False,
            seed=SEED, cw_c=1.0, cw_steps=20, cw_lr=0.01,
            iq_source="synthetic", n_samples=combo["n_samples"],
            embed_snr_margin=20.0, num_bursts=1,
            alignment_policy="naive", segment_hop=1, awn_preprocess="legacy-unit-power",
        )
    results["E2_n_samples"] = run_stage("E2_n_samples", combos, build_cfg_e2, "n_samples in [2048,8192,16384] (direct API only)")

    # E3: burst start reproducibility (radioml single-burst) -- handled
    # separately via a dedicated 2-process check, not a batch stage.

    # E4: single-burst mode vs a genuine 2-burst multi-burst run whose FIRST
    # burst is the exact same sample (BPSK/snr18/idx0). Note: num_bursts=1
    # is BY DESIGN always the single-burst code path (dataset_mod/dataset_snr/
    # sample_index singular fields) -- there is no "multi-burst code path
    # with exactly 1 entry" to compare against (validate_experiment_config
    # requires num_bursts>1 to use the *_list fields at all), so this
    # compares single-burst mode against burst 0 of a real 2-burst run
    # instead, which is the only valid comparison this design supports.
    print(f"\n{'='*70}\n[stage E4_single_vs_multi] 2 combos: single-burst mode vs burst-0-of-a-real-2-burst-run, same sample\n{'='*70}")

    def build_cfg_single(run_dir):
        return ExperimentConfig(
            snr=10.0, mod="BPSK", attack="none", topk=50,
            threshold_factor=1.5, window_size=128, sensing_window_size=128,
            min_region_len=0, merge_gap=0, burst_len=600,
            output_dir=str(run_dir), dry_run=True,
            use_real_topk=False, use_real_awn=True,
            checkpoint=CHECKPOINT, device="cpu",
            attack_eps=0.03, use_real_attack=False, attack_temperature=1.0, attack_diagnostics=False,
            seed=SEED, cw_c=1.0, cw_steps=20, cw_lr=0.01,
            iq_source="radioml", dataset_path=DATASET_PATH,
            dataset_mod="BPSK", dataset_snr=18, sample_index=0,
            embed_snr_margin=20.0, num_bursts=1,
            alignment_policy="max-energy", segment_hop=1, awn_preprocess="radioml-native",
        )

    def build_cfg_multi2(run_dir):
        return ExperimentConfig(
            snr=10.0, mod="BPSK", attack="none", topk=50,
            threshold_factor=1.5, window_size=128, sensing_window_size=128,
            min_region_len=0, merge_gap=0, burst_len=600,
            output_dir=str(run_dir), dry_run=True,
            use_real_topk=False, use_real_awn=True,
            checkpoint=CHECKPOINT, device="cpu",
            attack_eps=0.03, use_real_attack=False, attack_temperature=1.0, attack_diagnostics=False,
            seed=SEED, cw_c=1.0, cw_steps=20, cw_lr=0.01,
            iq_source="radioml", dataset_path=DATASET_PATH,
            dataset_mod=None, dataset_snr=None, sample_index=None,
            embed_snr_margin=20.0, num_bursts=2,
            dataset_mod_list=["BPSK", "QPSK"], dataset_snr_list=[18, 18], sample_index_list=[0, 1],
            min_burst_gap=400, max_burst_gap=400,
            alignment_policy="max-energy", segment_hop=1, awn_preprocess="radioml-native",
        )

    from src.utils.pipeline import run_dry_run_experiment
    e4_dir = BASE_OUTPUT / "E4_single_vs_multi"
    r_single = run_dry_run_experiment(build_cfg_single(e4_dir / "single"))
    r_multi2 = run_dry_run_experiment(build_cfg_multi2(e4_dir / "multi_2burst"))
    print(f"[E4] single-burst: regions={r_single['regions']} n_segments={r_single['n_segments']} "
          f"true_start/end via ground_truth={r_single['ground_truth']['true_start']}/{r_single['ground_truth']['true_end']}")
    print(f"[E4] multi(2 bursts): regions={r_multi2['regions']} n_segments={r_multi2['n_segments']}")
    with open(Path(r_multi2["bursts_summary_csv_path"])) as f:
        multi_bursts = list(csv.DictReader(f))
    burst0 = multi_bursts[0]
    print(f"[E4] multi burst_0 (same BPSK/snr18/idx0 sample): true_start/end={burst0['true_start']}/{burst0['true_end']} "
          f"captured_signal_ratio={burst0['captured_signal_ratio']}")
    results["E4_single_vs_multi"] = {"single": r_single, "multi_2burst": r_multi2, "multi_burst0_row": burst0}

    # E5: embed-snr-margin sweep
    values = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    combos = [{"embed_snr_margin": v, "dataset_mod": mod, "sample_index": idx}
              for v in values for mod in ["BPSK", "QPSK"] for idx in [0, 1]]

    def build_cfg_e5(combo, run_dir):
        return ExperimentConfig(
            snr=10.0, mod=combo["dataset_mod"], attack="none", topk=50,
            threshold_factor=1.5, window_size=128, sensing_window_size=128,
            min_region_len=0, merge_gap=0, burst_len=600,
            output_dir=str(run_dir), dry_run=True,
            use_real_topk=False, use_real_awn=True,
            checkpoint=CHECKPOINT, device="cpu",
            attack_eps=0.03, use_real_attack=False, attack_temperature=1.0, attack_diagnostics=False,
            seed=SEED, cw_c=1.0, cw_steps=20, cw_lr=0.01,
            iq_source="radioml", dataset_path=DATASET_PATH,
            dataset_mod=combo["dataset_mod"], dataset_snr=18, sample_index=combo["sample_index"],
            embed_snr_margin=combo["embed_snr_margin"], num_bursts=1,
            alignment_policy="max-energy", segment_hop=1, awn_preprocess="radioml-native",
        )
    results["E5_embed_snr_margin"] = run_stage(
        "E5_embed_snr_margin", combos, build_cfg_e5,
        f"embed_snr_margin in {values} x 2 mods x 2 sample_index"
    )

    return results


if __name__ == "__main__":
    stage_D_merge_gap()
    stage_E_burst_stream_checks()
