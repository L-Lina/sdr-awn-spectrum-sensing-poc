"""
RadioML2016.10a end-to-end smoke matrix (round 11 / Part C). Purpose:
exercise the full sensing -> alignment -> AWN-preprocess -> AWN -> attack ->
Top-K defense chain with the newly-adopted source-aware defaults
(--alignment-policy/--awn-preprocess left unset so iq_source="radioml"
auto-resolves to "max-energy"/"radioml-native", docs/parameter_validation.md
section 20) -- NOT an accuracy evaluation, NOT the formal full-parameter
batch (explicitly out of scope this round).

Grid: 3 modulations x 2 SNRs x 3 attacks x 4 topk values = 72 combos, real
AWN, real attack (fgsm/pgd; none is a no-op), real Top-K, CPU, one fixed
seed, sample_index=0 throughout (not swept -- kept small per "smoke
matrix", not a formal sweep). Measured ~1.5-2s/combo (real AWN + real
PGD + real Top-K, the most expensive combination); estimated ~2 minutes
total before running.
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
BASE_OUTPUT = Path("results/e2e_smoke_matrix")

MODS = ["QPSK", "BPSK", "QAM16"]
SNRS = [0, 18]
ATTACKS = ["none", "fgsm", "pgd"]
TOPKS = [10, 20, 30, 40]
SAMPLE_INDEX = 0

FIXED = dict(
    threshold_factor=1.5, sensing_window_size=128, min_region_len=0, merge_gap=0,
    attack_eps=0.05,
)


def build_combos():
    combos = []
    for mod in MODS:
        for snr in SNRS:
            for attack in ATTACKS:
                for topk in TOPKS:
                    combos.append({"dataset_mod": mod, "dataset_snr": snr, "attack": attack, "topk": topk})
    return combos


def build_cfg(combo: dict, run_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        snr=10.0, mod=combo["dataset_mod"], attack=combo["attack"], topk=combo["topk"],
        threshold_factor=FIXED["threshold_factor"],
        window_size=128,
        sensing_window_size=FIXED["sensing_window_size"],
        min_region_len=FIXED["min_region_len"],
        merge_gap=FIXED["merge_gap"],
        burst_len=600,
        output_dir=str(run_dir),
        dry_run=True,
        use_real_topk=True,
        use_real_awn=True,
        checkpoint=CHECKPOINT,
        device="cpu",
        attack_eps=FIXED["attack_eps"],
        use_real_attack=True,
        attack_temperature=1.0,
        attack_diagnostics=False,
        seed=SEED,
        cw_c=1.0, cw_steps=20, cw_lr=0.01,
        iq_source="radioml",
        dataset_path=DATASET_PATH,
        dataset_mod=combo["dataset_mod"],
        dataset_snr=combo["dataset_snr"],
        sample_index=SAMPLE_INDEX,
        embed_snr_margin=20.0,
        num_bursts=1,
        dataset_mod_list=None, dataset_snr_list=None, sample_index_list=None,
        min_burst_gap=50, max_burst_gap=50, burst_gap_list=None, burst_power_scale_list=None,
        # alignment_policy/awn_preprocess deliberately left as the dataclass
        # default (None) so this run exercises the NEW source-aware default
        # (section 20) -- iq_source="radioml" here resolves to
        # "max-energy"/"radioml-native" automatically.
    )


def main() -> None:
    combos = build_combos()
    print(f"[matrix] mods={len(MODS)} snrs={len(SNRS)} attacks={len(ATTACKS)} topks={len(TOPKS)}")
    print(f"[matrix] TOTAL combos: {len(combos)} (estimated ~2 minutes at ~1.5-2s/combo)")
    print(f"[matrix] fixed params: {FIXED}, seed={SEED}, real AWN, real attack, real Top-K, cpu")
    print("[matrix] alignment_policy/awn_preprocess left unset -- source-aware default applies")

    t0 = time.time()
    result = run_batch_combos(BASE_OUTPUT, combos, build_cfg)
    elapsed = time.time() - t0

    print(f"\n[matrix] done in {elapsed:.1f}s")
    print(f"[matrix] ok={result['n_ok']} sensing_failed={result['n_sensing_failed']} error={result['n_error']}")
    print(f"[matrix] batch_summary_csv={result['batch_summary_csv']}")


if __name__ == "__main__":
    main()
