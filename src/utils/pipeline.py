"""
Shared dry-run pipeline: synthetic IQ -> energy detection -> [N,2,128] ->
dummy AWN -> dummy attack -> dummy Top-K defense -> summary.csv -> sensing plot.

Used by both experiments/run_full_experiment.py (single run) and
experiments/run_batch.py (parameter grid).
"""

from __future__ import annotations

import random
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
from src.utils.config import (
    ExperimentConfig,
    resolve_sensing_window_size,
    validate_experiment_config,
)
from src.utils.csv_writer import write_summary_csv
from src.utils.plotting import plot_sensing_result

try:
    import torch as _torch  # type: ignore
except Exception:  # noqa: BLE001 - torch not installed in dummy-only environments
    _torch = None


def _seed_everything(seed: int) -> None:
    """
    Seed every RNG this pipeline (or a real attack backend it calls into) can
    draw from, once per run_dry_run_experiment call. random/numpy are seeded
    defensively -- this repo's own dummy_* functions already use local
    np.random.default_rng(seed) instances unaffected by the global numpy
    seed, but a global seed still covers any third-party code that reads
    global RNG state. torch.manual_seed is the one that actually matters for
    reproducibility here: torchattacks.PGD's random_start draws from
    torch.empty_like(...).uniform_(...), which reads torch's global default
    generator -- traced via inspect.getsource(torchattacks.PGD.forward),
    confirmed to be the only independent RNG source among FGSM/PGD/CW (FGSM
    is a single deterministic step; CW's Adam init is a deterministic
    function of the clean image, no randomness).
    """
    random.seed(seed)
    np.random.seed(seed)
    if _torch is not None:
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)


def run_dry_run_experiment(cfg: ExperimentConfig) -> Dict:
    if not cfg.dry_run:
        raise NotImplementedError(
            "Only --dry-run is supported in this phase; real AWN/attack/defense "
            "wiring comes in a later phase (see docs/integration_plan.md)."
        )
    # Adapter/algorithm boundary: catches direct-API callers who construct
    # ExperimentConfig by hand (bypassing argparse's type= validation
    # entirely), e.g. experiments/run_batch.py's own ExperimentConfig(...)
    # construction, or anyone importing this module directly.
    validate_experiment_config(cfg)

    # Seed every RNG source (random/numpy/torch/+cuda) at the start of every
    # single experiment -- not just once per process -- so run_batch.py's
    # multi-combo loop gives each combo the same reproducibility guarantee a
    # standalone run_full_experiment.py call gets, regardless of combo order.
    _seed_everything(cfg.seed)

    # sensing_window_size (energy_detect's smoothing window) and window_size
    # (segment_regions'/to_awn_input's seg_len == AWN input temporal length)
    # are deliberately decoupled here -- window_size is the legacy name and
    # keeps controlling segment length / AWN input length unchanged; only the
    # energy-detection call below uses the resolved sensing window. Resolved
    # here (not at config-construction time) so this applies uniformly
    # whether cfg came from argparse or was built directly.
    effective_sensing_window_size = resolve_sensing_window_size(
        cfg.window_size, cfg.sensing_window_size
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    iq, gen_meta = generate_synthetic_iq(
        n_samples=cfg.n_samples,
        burst_len=cfg.burst_len,
        snr_db=cfg.snr,
        mod=cfg.mod,
        seed=cfg.seed,
    )
    iq = validate_iq(iq)

    mask = energy_detect(iq, window=effective_sensing_window_size, threshold_factor=cfg.threshold_factor)
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
            return awn_adapter.infer(x, seed=cfg.seed)
    else:
        def run_awn(x):
            logits = dummy_awn_inference(x, seed=cfg.seed)
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
            x_clean, attack=cfg.attack, eps=cfg.attack_eps, temperature=cfg.attack_temperature,
            seed=cfg.seed, diagnostics=cfg.attack_diagnostics,
            cw_c=cfg.cw_c, cw_steps=cfg.cw_steps, cw_lr=cfg.cw_lr,
        )
    else:
        x_adv = dummy_attack(x_clean, attack=cfg.attack, epsilon=cfg.attack_eps, seed=cfg.seed)
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

    changed_by_attack = pred_attacked != pred_clean
    recovered_by_defense = changed_by_attack & (pred_defended == pred_clean)

    iq_diff_clean_attacked = x_adv - x_clean
    iq_linf_clean_attacked = np.max(np.abs(iq_diff_clean_attacked), axis=(1, 2))
    iq_l2_clean_attacked = np.linalg.norm(
        iq_diff_clean_attacked.reshape(iq_diff_clean_attacked.shape[0], -1), axis=1
    )
    logit_maxabs_clean_attacked = np.max(np.abs(logits_attacked - logits_clean), axis=1)

    clean_has_nan = np.isnan(x_clean).any(axis=(1, 2))
    clean_has_inf = np.isinf(x_clean).any(axis=(1, 2))
    attacked_has_nan = np.isnan(x_adv).any(axis=(1, 2))
    attacked_has_inf = np.isinf(x_adv).any(axis=(1, 2))

    # Per-segment arrays from AttackAdapter when the real attack path ran
    # (None for attack='none' / dummy fallback / --attack-diagnostics off).
    attack_iq_linf_normalized = attack_meta.get("attack_iq_linf_normalized")
    attack_gradient_nonzero_count = attack_meta.get("attack_gradient_nonzero_count")
    attack_gradient_total_count = attack_meta.get("attack_gradient_total_count")
    attack_gradient_maxabs = attack_meta.get("attack_gradient_maxabs")

    rows = []
    for i in range(x_clean.shape[0]):
        rows.append({
            "segment_id": i,
            "seed": cfg.seed,
            "snr_db": cfg.snr,
            "mod": cfg.mod,
            "attack": cfg.attack,
            "attack_eps": cfg.attack_eps,
            "topk": cfg.topk,
            "threshold_factor": cfg.threshold_factor,
            "window_size": cfg.window_size,
            # window_size above is preserved as-is (legacy name/column, kept
            # for backward-compat with existing analysis scripts). The two
            # columns below make the now-decoupled semantics explicit:
            # sensing_window_size is what energy_detect actually used;
            # segment_length is the segmentation/AWN-input length (always
            # equal to cfg.window_size in this round -- no crop/pad/resample
            # implemented).
            "sensing_window_size": effective_sensing_window_size,
            "segment_length": cfg.window_size,
            "pred_clean": int(pred_clean[i]),
            "pred_attacked": int(pred_attacked[i]),
            "pred_defended": int(pred_defended[i]),
            "changed_by_attack": bool(changed_by_attack[i]),
            "recovered_by_defense": bool(recovered_by_defense[i]),
            "iq_linf_clean_attacked": float(iq_linf_clean_attacked[i]),
            "iq_l2_clean_attacked": float(iq_l2_clean_attacked[i]),
            # Same value as iq_linf_clean_attacked (both are the per-segment
            # Linf norm of x_adv - x_clean); kept as its own column name for
            # compatibility with anything expecting a "maxabs" field.
            "iq_maxabs_clean_attacked": float(iq_linf_clean_attacked[i]),
            "logit_maxabs_clean_attacked": float(logit_maxabs_clean_attacked[i]),
            "attacked_has_nan": bool(attacked_has_nan[i]),
            "attacked_has_inf": bool(attacked_has_inf[i]),
            "clean_has_nan": bool(clean_has_nan[i]),
            "clean_has_inf": bool(clean_has_inf[i]),
            "attack_training_before": attack_meta.get("attack_training_before"),
            "attack_training_after": attack_meta.get("attack_training_after"),
            "attack_temperature": cfg.attack_temperature,
            # CW-only knobs (src/adapters/attack_adapter.py); recorded on
            # every row regardless of cfg.attack, same precedent as
            # attack_temperature above -- inert for none/fgsm/pgd.
            "cw_c": cfg.cw_c,
            "cw_steps": cfg.cw_steps,
            "cw_lr": cfg.cw_lr,
            "iq_linf_normalized_clean_attacked": (
                float(attack_iq_linf_normalized[i]) if attack_iq_linf_normalized is not None else None
            ),
            "attack_gradient_nonzero_count": (
                int(attack_gradient_nonzero_count[i]) if attack_gradient_nonzero_count is not None else None
            ),
            "attack_gradient_total_count": (
                int(attack_gradient_total_count[i]) if attack_gradient_total_count is not None else None
            ),
            "attack_gradient_maxabs": (
                float(attack_gradient_maxabs[i]) if attack_gradient_maxabs is not None else None
            ),
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
        "seed": cfg.seed,
        "sensing_window_size": effective_sensing_window_size,
        "segment_length": cfg.window_size,
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
    print(f"Seed:               {cfg.seed} (random/numpy/torch[+cuda] seeded at start of this run)")
    print(f"IQ stream length:   {len(iq)} samples")
    print(f"Sensing window:     {effective_sensing_window_size} (energy_detect smoothing window)")
    print(f"Segment length:     {cfg.window_size} (segment_regions/to_awn_input seg_len == AWN input length)")
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
