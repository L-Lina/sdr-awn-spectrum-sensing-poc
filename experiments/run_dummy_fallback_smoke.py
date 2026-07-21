"""
Dummy-fallback end-to-end smoke test (round 15). Purpose: the numpy-only
dummy backends (dummy_awn_inference / dummy_attack / dummy_topk_defense)
have existed since the earliest rounds of this repo but had NEVER been run
end-to-end through the actual run_dry_run_experiment() entry point in any
prior round -- every real-backend round explicitly passed all three
--use-real-* flags. This gap is documented in docs/parameter_validation.md
section 7 item 3 and CSV rows for use_real_awn/use_real_attack/use_real_topk
("False (dummy) path never actually executed in-session").

This round closes exactly that gap: confirm the dummy path executes to
completion without crashing, is labeled unambiguously as dummy (never
mistakable for a real-backend result), and is reproducible. Does NOT modify
sensing/AWN/attack/Top-K algorithm code -- src/utils/pipeline.py already
routes cfg.use_real_awn/use_real_attack/use_real_topk == False directly to
the dummy_* functions (lines ~370-417); this script only exercises that
existing, already-wired path.

4 required combos (synthetic/radioml x none/fgsm), each run TWICE (same
seed) to confirm reproducibility. synthetic and radioml combos are run as
separate run_batch_combos() calls per the documented ground-truth-mode
constraint (they'd otherwise produce inconsistent CSV columns in one call).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.adapters.attack_adapter import dummy_attack  # noqa: E402
from src.adapters.awn_adapter import dummy_awn_inference  # noqa: E402
from src.adapters.defense_adapter import dummy_topk_defense  # noqa: E402
from src.utils.batch_aggregation import run_batch_combos  # noqa: E402
from src.utils.config import ExperimentConfig  # noqa: E402

SEED = 42
DATASET_PATH = "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl"
BASE_OUTPUT = Path("results/dummy_fallback_smoke")

FIXED = dict(threshold_factor=1.5, sensing_window_size=128, min_region_len=0, merge_gap=0)

EXPECTED_BACKENDS = {
    "awn_backend": "dummy_awn_inference",
    "attack_backend": "dummy_attack",
    "topk_backend": "dummy_topk_defense",
}
EXPECTED_STATUS_FIELDS = ["awn_status", "attack_status", "topk_status"]

# Columns allowed to legitimately differ between two reproducibility runs
# (paths only -- never content).
_PATH_ONLY_COLUMNS = {
    "output_dir", "summary_csv_path", "bursts_summary_csv_path",
    "regions_summary_csv_path", "plot_path",
}


def build_cfg_synthetic(combo: dict, run_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        snr=10.0, mod="QPSK", attack=combo["attack"], topk=20,
        threshold_factor=FIXED["threshold_factor"], window_size=128,
        sensing_window_size=FIXED["sensing_window_size"],
        min_region_len=FIXED["min_region_len"], merge_gap=FIXED["merge_gap"],
        burst_len=600, output_dir=str(run_dir), dry_run=True,
        use_real_topk=False, use_real_awn=False, use_real_attack=False,
        checkpoint="external/adversarial-rf/2016.10a_AWN.pkl", device="cpu",
        attack_eps=0.05, attack_temperature=1.0, attack_diagnostics=False,
        seed=SEED, cw_c=1.0, cw_steps=20, cw_lr=0.01,
        iq_source="synthetic", dataset_path=None, dataset_mod=None, dataset_snr=None,
        embed_snr_margin=20.0, num_bursts=1,
        dataset_mod_list=None, dataset_snr_list=None, sample_index_list=None,
        min_burst_gap=50, max_burst_gap=50, burst_gap_list=None, burst_power_scale_list=None,
    )


def build_cfg_radioml(combo: dict, run_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        snr=10.0, mod="QPSK", attack=combo["attack"], topk=20,
        threshold_factor=FIXED["threshold_factor"], window_size=128,
        sensing_window_size=FIXED["sensing_window_size"],
        min_region_len=FIXED["min_region_len"], merge_gap=FIXED["merge_gap"],
        burst_len=600, output_dir=str(run_dir), dry_run=True,
        use_real_topk=False, use_real_awn=False, use_real_attack=False,
        checkpoint="external/adversarial-rf/2016.10a_AWN.pkl", device="cpu",
        attack_eps=0.05, attack_temperature=1.0, attack_diagnostics=False,
        seed=SEED, cw_c=1.0, cw_steps=20, cw_lr=0.01,
        iq_source="radioml", dataset_path=DATASET_PATH,
        dataset_mod="QPSK", dataset_snr=18, sample_index=0,
        embed_snr_margin=20.0, num_bursts=1,
        dataset_mod_list=None, dataset_snr_list=None, sample_index_list=None,
        min_burst_gap=50, max_burst_gap=50, burst_gap_list=None, burst_power_scale_list=None,
    )


def run_stage(stage_name: str, builder, run_idx: int) -> dict:
    combos = [{"attack": "none"}, {"attack": "fgsm"}]
    stage_dir = BASE_OUTPUT / f"{stage_name}_run{run_idx}"
    print(f"\n[stage {stage_name}_run{run_idx}] {len(combos)} combos, dummy backends only (use_real_* all False)")
    result = run_batch_combos(stage_dir, combos, builder)
    print(f"[stage {stage_name}_run{run_idx}] done -- ok={result['n_ok']} sensing_failed={result['n_sensing_failed']} error={result['n_error']}")
    return result


def load_csv_rows(path: Path) -> list:
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


def load_batch_rows(stage_dir: Path) -> list:
    return load_csv_rows(stage_dir / "batch_summary.csv")


def load_all_combo_summary_rows(stage_dir: Path, n_combos: int) -> list:
    """Backend/status/NaN/Inf fields live in each combo's own summary.csv
    (per-segment), not batch_summary.csv (which aggregates sensing/CSV
    metadata only, no backend columns) -- confirmed by inspecting both
    headers directly before writing this check."""
    rows = []
    for combo_id in range(n_combos):
        combo_dir = stage_dir / f"combo{combo_id:04d}"
        summary_path = combo_dir / "summary.csv"
        if summary_path.exists():
            rows.extend(load_csv_rows(summary_path))
    return rows


def check_batch_run_status(stage_name: str, batch_rows: list) -> list:
    problems = []
    for r in batch_rows:
        combo_desc = f"{stage_name} attack={r.get('attack')}"
        if r["run_status"] not in ("ok", "sensing_failed"):
            problems.append(f"{combo_desc}: unexpected run_status={r['run_status']!r}")
        elif r["run_status"] == "sensing_failed":
            if not r.get("failure_stage") or not r.get("failure_reason"):
                problems.append(f"{combo_desc}: sensing_failed but missing failure_stage/failure_reason")
    return problems


def check_backends_and_nan(stage_name: str, segment_rows: list) -> list:
    problems = []
    if not segment_rows:
        problems.append(f"{stage_name}: no per-segment summary.csv rows found (all combos sensing_failed?)")
        return problems
    for r in segment_rows:
        combo_desc = f"{stage_name} attack={r.get('attack')} segment={r.get('segment_id')}"
        for field, expected in EXPECTED_BACKENDS.items():
            actual = r.get(field)
            if actual != expected:
                problems.append(f"{combo_desc}: {field}={actual!r}, expected {expected!r}")
        for field in EXPECTED_STATUS_FIELDS:
            actual = r.get(field)
            if actual is None or actual != "ok" or "fallback" in str(actual).lower():
                problems.append(f"{combo_desc}: {field}={actual!r} (must be exactly 'ok', never mention fallback)")
        for nan_field in ("clean_has_nan", "attacked_has_nan", "awn_input_has_nan"):
            if nan_field in r and str(r[nan_field]).strip().lower() not in ("false", "0"):
                problems.append(f"{combo_desc}: {nan_field}={r[nan_field]!r} (NaN present)")
        for inf_field in ("clean_has_inf", "attacked_has_inf", "awn_input_has_inf"):
            if inf_field in r and str(r[inf_field]).strip().lower() not in ("false", "0"):
                problems.append(f"{combo_desc}: {inf_field}={r[inf_field]!r} (Inf present)")
    return problems


def check_reproducibility(stage_name: str, rows1: list, rows2: list) -> list:
    problems = []
    if len(rows1) != len(rows2):
        problems.append(f"{stage_name}: row count differs {len(rows1)} vs {len(rows2)}")
        return problems
    all_cols = set(rows1[0].keys()) | set(rows2[0].keys())
    compare_cols = all_cols - _PATH_ONLY_COLUMNS
    for i, (a, b) in enumerate(zip(rows1, rows2)):
        for col in compare_cols:
            va, vb = a.get(col), b.get(col)
            if va != vb:
                problems.append(f"{stage_name} row{i} col={col}: run1={va!r} run2={vb!r}")
    return problems


def check_direct_api_no_nan_inf() -> list:
    """Directly exercise the three dummy functions (not just through the
    pipeline/CSV) and check their raw numpy outputs for NaN/Inf."""
    problems = []
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(5, 2, 128)).astype(np.float32)

    logits = dummy_awn_inference(x, seed=SEED)
    if np.isnan(logits).any() or np.isinf(logits).any():
        problems.append("dummy_awn_inference: NaN/Inf in logits")
    if logits.shape != (5, 11):
        problems.append(f"dummy_awn_inference: unexpected shape {logits.shape}")

    x_adv = dummy_attack(x, attack="fgsm", epsilon=0.05, seed=SEED)
    if np.isnan(x_adv).any() or np.isinf(x_adv).any():
        problems.append("dummy_attack: NaN/Inf in output")

    x_def = dummy_topk_defense(x_adv, topk=20)
    if np.isnan(x_def).any() or np.isinf(x_def).any():
        problems.append("dummy_topk_defense: NaN/Inf in output")

    return problems


def main() -> None:
    print("=" * 70)
    print("[dummy_fallback_smoke] 4 required combos x 2 runs each = 8 pipeline")
    print("runs, plus direct-API NaN/Inf checks. All dummy backends (no torch,")
    print("no real checkpoint). Estimated well under 1 minute total.")
    print("=" * 70)

    all_problems = []

    r_syn_1 = run_stage("synthetic", build_cfg_synthetic, 1)
    r_syn_2 = run_stage("synthetic", build_cfg_synthetic, 2)
    r_rml_1 = run_stage("radioml", build_cfg_radioml, 1)
    r_rml_2 = run_stage("radioml", build_cfg_radioml, 2)

    n_combos = 2  # attack in {none, fgsm}, fixed per run_stage()

    batch_syn_1 = load_batch_rows(BASE_OUTPUT / "synthetic_run1")
    batch_syn_2 = load_batch_rows(BASE_OUTPUT / "synthetic_run2")
    batch_rml_1 = load_batch_rows(BASE_OUTPUT / "radioml_run1")
    batch_rml_2 = load_batch_rows(BASE_OUTPUT / "radioml_run2")

    seg_syn_1 = load_all_combo_summary_rows(BASE_OUTPUT / "synthetic_run1", n_combos)
    seg_syn_2 = load_all_combo_summary_rows(BASE_OUTPUT / "synthetic_run2", n_combos)
    seg_rml_1 = load_all_combo_summary_rows(BASE_OUTPUT / "radioml_run1", n_combos)
    seg_rml_2 = load_all_combo_summary_rows(BASE_OUTPUT / "radioml_run2", n_combos)

    for stage_name, batch_rows in [
        ("synthetic_run1", batch_syn_1), ("synthetic_run2", batch_syn_2),
        ("radioml_run1", batch_rml_1), ("radioml_run2", batch_rml_2),
    ]:
        all_problems += check_batch_run_status(stage_name, batch_rows)

    all_problems += check_backends_and_nan("synthetic_run1", seg_syn_1)
    all_problems += check_backends_and_nan("synthetic_run2", seg_syn_2)
    all_problems += check_backends_and_nan("radioml_run1", seg_rml_1)
    all_problems += check_backends_and_nan("radioml_run2", seg_rml_2)

    all_problems += check_reproducibility("synthetic (batch_summary.csv)", batch_syn_1, batch_syn_2)
    all_problems += check_reproducibility("radioml (batch_summary.csv)", batch_rml_1, batch_rml_2)
    all_problems += check_reproducibility("synthetic (per-segment summary.csv)", seg_syn_1, seg_syn_2)
    all_problems += check_reproducibility("radioml (per-segment summary.csv)", seg_rml_1, seg_rml_2)

    all_problems += check_direct_api_no_nan_inf()

    print("\n" + "=" * 70)
    if all_problems:
        print(f"[dummy_fallback_smoke] FAIL -- {len(all_problems)} problem(s):")
        for p in all_problems:
            print(f"  - {p}")
        sys.exit(1)
    else:
        print("[dummy_fallback_smoke] PASS -- all backend labels, NaN/Inf checks, "
              "and reproducibility checks passed for all 4 combos x 2 runs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
