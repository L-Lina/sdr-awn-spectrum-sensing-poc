"""
Fair Top-K verification matrix under radioml-native preprocessing (round 12).
Purpose: verify, at a larger sample count than round 11's 6-sample smoke test,
that Top-K's K=10/20/30/40 always share exactly the same attacked IQ (the
attack is computed once per (sample, attack) combo, never regenerated per K),
across all four attack types (none/fgsm/pgd/cw), with real AWN + real attack
+ real Top-K backends only (no fallback tolerated -- verified after the run,
not merely assumed). NOT an accuracy evaluation, NOT the formal full-parameter
batch (explicitly out of scope this round).

Grid: 3 modulations (QPSK, BPSK, QAM16) x 2 SNRs (0, 18) x 5 sample_indices
(0-4) = 30 unique samples, x 4 attacks (none, fgsm, pgd, cw) x 4 Top-K values
(10, 20, 30, 40) = 480 combos. sample_index expansion (5x vs round 11's
implicit 1) is the "expand sample count" lever per this round's explicit
instruction; attack and topk dimensions are fixed at exactly what was asked
for (all 4 attacks, all 4 K values). eps=0.05 fixed for fgsm/pgd (not swept
this round -- an eps sweep would push toward the formal batch, explicitly
out of scope); cw uses its default strength knobs (c=1.0, steps=20, lr=0.01).
Measured ~1.4-2.5s/combo (real AWN + real attack incl. CW's 20-step
optimization + real Top-K, CPU); estimated ~15 minutes total before running.
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
BASE_OUTPUT = Path("results/fair_topk_matrix")

MODS = ["QPSK", "BPSK", "QAM16"]
SNRS = [0, 18]
SAMPLE_INDICES = [0, 1, 2, 3, 4]
ATTACKS = ["none", "fgsm", "pgd", "cw"]
TOPKS = [10, 20, 30, 40]

FIXED = dict(
    threshold_factor=1.5, sensing_window_size=128, min_region_len=0, merge_gap=0,
    attack_eps=0.05, cw_c=1.0, cw_steps=20, cw_lr=0.01,
)


def build_combos():
    combos = []
    for mod in MODS:
        for snr in SNRS:
            for idx in SAMPLE_INDICES:
                for attack in ATTACKS:
                    for topk in TOPKS:
                        combos.append({
                            "dataset_mod": mod, "dataset_snr": snr, "sample_index": idx,
                            "attack": attack, "topk": topk,
                        })
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
        cw_c=FIXED["cw_c"], cw_steps=FIXED["cw_steps"], cw_lr=FIXED["cw_lr"],
        iq_source="radioml",
        dataset_path=DATASET_PATH,
        dataset_mod=combo["dataset_mod"],
        dataset_snr=combo["dataset_snr"],
        sample_index=combo["sample_index"],
        embed_snr_margin=20.0,
        num_bursts=1,
        dataset_mod_list=None, dataset_snr_list=None, sample_index_list=None,
        min_burst_gap=50, max_burst_gap=50, burst_gap_list=None, burst_power_scale_list=None,
        # alignment_policy/awn_preprocess deliberately left unset (None) --
        # iq_source="radioml" resolves to "max-energy"/"radioml-native"
        # (docs/parameter_validation.md section 20).
    )


def main() -> None:
    combos = build_combos()
    n_samples = len(MODS) * len(SNRS) * len(SAMPLE_INDICES)
    print(f"[matrix] unique samples: {n_samples} (mods={len(MODS)} x snrs={len(SNRS)} x sample_indices={len(SAMPLE_INDICES)})")
    print(f"[matrix] attacks={ATTACKS} topks={TOPKS}")
    print(f"[matrix] TOTAL combos: {len(combos)} (estimated ~15 minutes at ~1.4-2.5s/combo)")
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
