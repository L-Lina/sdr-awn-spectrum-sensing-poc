"""
Shared dry-run pipeline: synthetic IQ -> energy detection -> [N,2,128] ->
dummy AWN -> dummy attack -> dummy Top-K defense -> summary.csv -> sensing plot.

Used by both experiments/run_full_experiment.py (single run) and
experiments/run_batch.py (parameter grid).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from src.adapters.attack_adapter import AttackAdapter, dummy_attack
from src.adapters.awn_adapter import AWNModelAdapter, dummy_awn_inference
from src.adapters.defense_adapter import dummy_topk_defense
from src.adapters.topk_adapter import TopKAdapter
from src.sensing.energy_detection import (
    energy_detect,
    filter_by_min_length,
    mask_to_regions,
    merge_close_regions,
)
from src.sensing.iq_source import generate_synthetic_iq, validate_iq
from src.sensing.normalize import normalize_segments, to_awn_input
from src.sensing.segmentation import segment_regions
from src.utils.config import ExperimentConfig
from src.utils.csv_writer import write_summary_csv
from src.utils.plotting import plot_sensing_result

SEED = 0


def run_dry_run_experiment(cfg: ExperimentConfig) -> Dict:
    if not cfg.dry_run:
        raise NotImplementedError(
            "Only --dry-run is supported in this phase; real AWN/attack/defense "
            "wiring comes in a later phase (see docs/integration_plan.md)."
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    iq, gen_meta = generate_synthetic_iq(
        n_samples=cfg.n_samples,
        burst_len=cfg.burst_len,
        snr_db=cfg.snr,
        mod=cfg.mod,
        seed=SEED,
    )
    iq = validate_iq(iq)

    mask = energy_detect(iq, window=cfg.window_size, threshold_factor=cfg.threshold_factor)
    raw_regions = mask_to_regions(mask)
    merged_regions = merge_close_regions(raw_regions, merge_gap=cfg.merge_gap)
    regions = filter_by_min_length(merged_regions, min_len=cfg.min_region_len)

    segments = segment_regions(iq, regions, seg_len=cfg.window_size)
    segments = normalize_segments(segments)
    x_clean = to_awn_input(segments, seg_len=cfg.window_size)

    awn_adapter = None
    if cfg.use_real_awn:
        awn_adapter = AWNModelAdapter(checkpoint_path=cfg.checkpoint, device=cfg.device)

        def run_awn(x):
            return awn_adapter.infer(x, seed=SEED)
    else:
        def run_awn(x):
            logits = dummy_awn_inference(x, seed=SEED)
            meta = {
                "awn_backend": "dummy_awn_inference",
                "awn_status": "ok",
                "awn_notes": "--use-real-awn not passed; using placeholder AWN inference by default.",
            }
            return logits, meta

    logits_clean, awn_meta = run_awn(x_clean)

    attack_input_shape = x_clean.shape
    if cfg.use_real_attack:
        real_model = awn_adapter.model if awn_adapter is not None else None
        x_adv, attack_meta = AttackAdapter(awn_model=real_model, device=cfg.device).apply(
            x_clean, attack=cfg.attack, eps=cfg.attack_eps, seed=SEED
        )
    else:
        x_adv = dummy_attack(x_clean, attack=cfg.attack, epsilon=cfg.attack_eps, seed=SEED)
        attack_meta = {
            "attack_backend": "dummy_attack",
            "attack_status": "ok",
            "attack_notes": "--use-real-attack not passed; using placeholder attack by default.",
        }
    if x_adv.shape != attack_input_shape:
        raise RuntimeError(f"Attack output shape {x_adv.shape} != input shape {attack_input_shape}")

    logits_attacked, _ = run_awn(x_adv)

    input_shape = x_adv.shape
    if cfg.use_real_topk:
        x_defended, topk_meta = TopKAdapter().apply(x_adv, topk=cfg.topk)
    else:
        x_defended = dummy_topk_defense(x_adv, topk=cfg.topk)
        topk_meta = {
            "topk_backend": "dummy_topk_defense",
            "topk_status": "ok",
            "topk_notes": "--use-real-topk not passed; using placeholder Top-K defense by default.",
        }
    if x_defended.shape != input_shape:
        raise RuntimeError(f"Top-K defense output shape {x_defended.shape} != input shape {input_shape}")

    logits_defended, _ = run_awn(x_defended)

    pred_clean = np.argmax(logits_clean, axis=1)
    pred_attacked = np.argmax(logits_attacked, axis=1)
    pred_defended = np.argmax(logits_defended, axis=1)

    rows = []
    for i in range(x_clean.shape[0]):
        rows.append({
            "segment_id": i,
            "snr_db": cfg.snr,
            "mod": cfg.mod,
            "attack": cfg.attack,
            "attack_eps": cfg.attack_eps,
            "topk": cfg.topk,
            "threshold_factor": cfg.threshold_factor,
            "window_size": cfg.window_size,
            "pred_clean": int(pred_clean[i]),
            "pred_attacked": int(pred_attacked[i]),
            "pred_defended": int(pred_defended[i]),
            "topk_backend": topk_meta["topk_backend"],
            "topk_status": topk_meta["topk_status"],
            "topk_notes": topk_meta["topk_notes"],
            "awn_backend": awn_meta["awn_backend"],
            "awn_status": awn_meta["awn_status"],
            "awn_notes": awn_meta["awn_notes"],
            "attack_backend": attack_meta["attack_backend"],
            "attack_status": attack_meta["attack_status"],
            "attack_notes": attack_meta["attack_notes"],
        })

    summary_csv_path = output_dir / "summary.csv"
    write_summary_csv(summary_csv_path, rows)

    plot_path = output_dir / "sensing_plot.png"
    plot_created = plot_sensing_result(iq, regions, plot_path)

    result = {
        "n_segments": x_clean.shape[0],
        "regions": regions,
        "output_dir": str(output_dir),
        "summary_csv_path": str(summary_csv_path),
        "plot_path": str(plot_path) if plot_created else None,
        "gen_meta": gen_meta,
        "topk_input_shape": tuple(input_shape),
        "topk_output_shape": tuple(x_defended.shape),
        "awn_input_shape": tuple(x_clean.shape),
        "awn_logits_shape": tuple(logits_clean.shape),
        "attack_input_shape": tuple(attack_input_shape),
        "attack_output_shape": tuple(x_adv.shape),
        **topk_meta,
        **awn_meta,
        **attack_meta,
    }

    print("\n--- Dry-run summary ---")
    print(f"IQ stream length:   {len(iq)} samples")
    print(f"Occupied regions:   {len(regions)} -> {regions}")
    print(f"Number of segments: {result['n_segments']}")
    print(f"AWN input shape:    {x_clean.shape}")
    print(f"AWN backend:        {awn_meta['awn_backend']} (status={awn_meta['awn_status']})")
    print(f"AWN logits shape:   {result['awn_logits_shape']}")
    print(f"Attack backend:     {attack_meta['attack_backend']} (status={attack_meta['attack_status']})")
    print(f"Attack shape check: input={result['attack_input_shape']} -> output={result['attack_output_shape']}")
    print(f"Top-K backend:      {topk_meta['topk_backend']} (status={topk_meta['topk_status']})")
    print(f"Top-K shape check:  input={result['topk_input_shape']} -> output={result['topk_output_shape']}")
    print(f"summary.csv:        {summary_csv_path}")
    print(f"sensing plot:       {result['plot_path'] or '(skipped, matplotlib not installed)'}")

    return result
