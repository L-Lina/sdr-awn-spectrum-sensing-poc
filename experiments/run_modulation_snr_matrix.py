"""
Cross-modulation x SNR smoke matrix (round 8 / Part B). Purpose: exercise
the real RadioML loader, real AWN backend, and the batch-aggregation
infrastructure (src/utils/batch_aggregation.py) across every RML2016.10a
modulation and a representative SNR spread -- NOT an AMC-accuracy
evaluation, NOT a formal batch. attack=none, topk=10 (fixed, not swept),
real AWN, CPU, a single fixed seed throughout. Sensing parameters
(threshold-factor/sensing-window-size/min-region-len/merge-gap) are held at
one fixed point for every combo -- NOT tuned per combo, NOT adjusted to
force failed combos to pass.

Grid: 11 modulations x 4 SNRs x 3 sample_indices = 132 combos, single-burst
RadioML mode (one run_batch_combos() call -- see batch_aggregation.py's
ground-truth-mode constraint). Estimated ~3-5 minutes (measured ~1.3-1.7s/
combo steady-state, real AWN + real per-combo RadioML dict reload).
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
BASE_OUTPUT = Path("results/modulation_snr_matrix")

MODULATIONS = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
    "PAM4", "QAM16", "QAM64", "QPSK", "WBFM",
]
SNRS = [-10, 0, 10, 18]
SAMPLE_INDICES = [0, 1, 2]

FIXED = dict(
    attack="none", topk=10,
    threshold_factor=1.5, sensing_window_size=128, min_region_len=128, merge_gap=0,
)


def build_combos():
    combos = []
    for mod in MODULATIONS:
        for snr in SNRS:
            for idx in SAMPLE_INDICES:
                combos.append({"dataset_mod": mod, "dataset_snr": snr, "sample_index": idx})
    return combos


def build_cfg(combo: dict, run_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        snr=10.0, mod=combo["dataset_mod"], attack=FIXED["attack"], topk=FIXED["topk"],
        threshold_factor=FIXED["threshold_factor"],
        window_size=128,
        sensing_window_size=FIXED["sensing_window_size"],
        min_region_len=FIXED["min_region_len"],
        merge_gap=FIXED["merge_gap"],
        burst_len=600,
        output_dir=str(run_dir),
        dry_run=True,
        use_real_topk=False,
        use_real_awn=True,
        checkpoint=CHECKPOINT,
        device="cpu",
        attack_eps=0.03,
        use_real_attack=False,
        attack_temperature=1.0,
        attack_diagnostics=False,
        seed=SEED,
        cw_c=1.0, cw_steps=20, cw_lr=0.01,
        iq_source="radioml",
        dataset_path=DATASET_PATH,
        dataset_mod=combo["dataset_mod"],
        dataset_snr=combo["dataset_snr"],
        sample_index=combo["sample_index"],
        embed_snr_margin=20.0,
        num_bursts=1,
        dataset_mod_list=None, dataset_snr_list=None, sample_index_list=None,
        min_burst_gap=50, max_burst_gap=50, burst_gap_list=None, burst_power_scale_list=None,
    )


def main() -> None:
    combos = build_combos()
    print(f"[matrix] modulations={len(MODULATIONS)} snrs={len(SNRS)} sample_indices={len(SAMPLE_INDICES)}")
    print(f"[matrix] TOTAL combos: {len(combos)} (estimated ~3-5 minutes at ~1.3-1.7s/combo)")
    print(f"[matrix] fixed params: {FIXED}, seed={SEED}, real AWN, cpu")

    t0 = time.time()
    result = run_batch_combos(BASE_OUTPUT, combos, build_cfg)
    elapsed = time.time() - t0

    print(f"\n[matrix] done in {elapsed:.1f}s")
    print(f"[matrix] ok={result['n_ok']} sensing_failed={result['n_sensing_failed']} error={result['n_error']}")
    print(f"[matrix] batch_summary_csv={result['batch_summary_csv']}")
    print(f"[matrix] batch_bursts_summary_csv={result['batch_bursts_summary_csv']}")
    print(f"[matrix] batch_regions_summary_csv={result['batch_regions_summary_csv']}")


if __name__ == "__main__":
    main()
