"""
Shared dry-run pipeline: synthetic IQ -> energy detection -> [N,2,128] ->
dummy AWN -> dummy attack -> dummy Top-K defense -> summary.csv -> sensing plot.

Used by both experiments/run_full_experiment.py (single run) and
experiments/run_batch.py (parameter grid).
"""

from __future__ import annotations

import hashlib
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
from src.sensing.ground_truth_metrics import (
    compute_multi_burst_sensing_metrics,
    compute_sensing_ground_truth_metrics,
    derive_batch_aggregate_sensing_fields,
)
from src.sensing.iq_source import generate_synthetic_iq, validate_iq
from src.sensing.normalize import apply_awn_preprocess, to_awn_input
from src.sensing.radioml_source import (
    embed_multiple_samples_in_noise,
    embed_sample_in_noise,
    load_radioml_sample,
)
from src.sensing.segmentation import select_aligned_segments
from src.utils.config import (
    ExperimentConfig,
    resolve_alignment_policy,
    resolve_awn_preprocess,
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

    # Source-aware alignment/preprocessing defaults (docs/parameter_validation.md
    # section 20) -- same None-means-resolve-downstream pattern as
    # effective_sensing_window_size above. An explicitly passed
    # cfg.alignment_policy/cfg.awn_preprocess is never overridden; only a
    # None gets a source-aware value filled in here.
    effective_alignment_policy = resolve_alignment_policy(cfg.iq_source, cfg.alignment_policy)
    effective_awn_preprocess = resolve_awn_preprocess(cfg.iq_source, cfg.awn_preprocess)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    radioml_meta = None
    multi_burst_truths = None  # list[dict], input to compute_multi_burst_sensing_metrics
    if cfg.iq_source == "radioml" and cfg.num_bursts > 1:
        # Multi-burst path -- completely separate from the single-burst
        # branch below (never entered when num_bursts<=1), so single-burst
        # behavior is byte-for-byte unaffected by this branch's existence.
        # dataset_mod_list/dataset_snr_list/sample_index_list presence and
        # length already checked in validate_experiment_config(); per-entry
        # mod/snr key existence and sample_index range are checked here, at
        # load time (each load_radioml_sample call), same as single-burst.
        samples = []
        per_burst_labels = []
        for i in range(cfg.num_bursts):
            mod_i, snr_i, idx_i = cfg.dataset_mod_list[i], cfg.dataset_snr_list[i], cfg.sample_index_list[i]
            sample = load_radioml_sample(cfg.dataset_path, mod_i, snr_i, idx_i)
            samples.append(sample)
            per_burst_labels.append({
                "burst_id": i,
                "dataset_mod": mod_i,
                "dataset_snr": snr_i,
                "sample_index": idx_i,
                "original_sample_sha256": hashlib.sha256(sample.tobytes()).hexdigest(),
            })
        iq, per_burst_embed_meta = embed_multiple_samples_in_noise(
            samples, n_samples=cfg.n_samples, embed_snr_margin=cfg.embed_snr_margin, seed=cfg.seed,
            min_burst_gap=cfg.min_burst_gap, max_burst_gap=cfg.max_burst_gap, gap_list=cfg.burst_gap_list,
            power_scale_list=cfg.burst_power_scale_list,
        )
        multi_burst_truths = [
            {**label, **embed} for label, embed in zip(per_burst_labels, per_burst_embed_meta)
        ]
        gen_meta = {
            "source_type": "radioml_multi_burst",
            "dataset_path": cfg.dataset_path,
            "num_bursts": cfg.num_bursts,
        }
    elif cfg.iq_source == "radioml":
        # dataset_path/dataset_mod/dataset_snr presence already checked in
        # validate_experiment_config(); mod/snr key existence and
        # sample_index range are checked here, at load time, since they
        # require actually opening the dataset file.
        original_sample = load_radioml_sample(cfg.dataset_path, cfg.dataset_mod, cfg.dataset_snr, cfg.sample_index)
        original_sample_sha256 = hashlib.sha256(original_sample.tobytes()).hexdigest()
        iq, embed_meta = embed_sample_in_noise(
            original_sample, n_samples=cfg.n_samples, embed_snr_margin=cfg.embed_snr_margin, seed=cfg.seed,
        )
        gen_meta = {
            "source_type": "radioml",
            "dataset_path": cfg.dataset_path,
            "dataset_mod": cfg.dataset_mod,
            "dataset_snr": cfg.dataset_snr,
            "sample_index": cfg.sample_index,
            "original_sample_sha256": original_sample_sha256,
            **embed_meta,
        }
        radioml_meta = gen_meta
    else:
        iq, gen_meta = generate_synthetic_iq(
            n_samples=cfg.n_samples,
            burst_len=cfg.burst_len,
            snr_db=cfg.snr,
            mod=cfg.mod,
            seed=cfg.seed,
        )
        gen_meta["source_type"] = "synthetic"
    iq = validate_iq(iq)
    long_iq_sha256 = hashlib.sha256(iq.tobytes()).hexdigest()

    mask = energy_detect(iq, window=effective_sensing_window_size, threshold_factor=cfg.threshold_factor)
    raw_regions = mask_to_regions(mask)
    merged_regions = merge_close_regions(raw_regions, merge_gap=cfg.merge_gap)

    # Two EXPECTED sensing-outcome failure modes exist below, and only
    # these two: filter_by_min_length raises RuntimeError when no region
    # survives (either none were ever detected, or all detected regions
    # were shorter than --min-region-len), and segment_regions raises
    # RuntimeError when every surviving region is individually shorter than
    # --window-size (so zero full windows can be cut from any of them).
    # Both are narrowly try/excepted around EXACTLY their own call site --
    # nothing else in this function is caught here, so a genuine bug
    # anywhere else (adapter shape mismatches, config errors, dataset
    # loading errors) still raises normally, uncaught, exactly as before
    # this round. See docs/parameter_validation.md section 16.1 for the
    # full rationale and docs/parameter_validation.csv for the flag this
    # behavior is recorded against.
    sensing_failure_stage = None
    sensing_failure_reason = None
    try:
        regions = filter_by_min_length(merged_regions, min_len=cfg.min_region_len)
    except RuntimeError as exc:
        sensing_failure_stage = "filter_by_min_length"
        sensing_failure_reason = str(exc)
        regions = []

    # Region-count diagnostics (docs/parameter_validation.md section 22) --
    # exposes the three intermediate stage counts (mask_to_regions ->
    # merge_close_regions -> filter_by_min_length) that were previously only
    # local variables, never returned. Purely additive/diagnostic -- does
    # not change which regions are used anywhere downstream.
    num_raw_regions = len(raw_regions)
    num_merged_regions = len(merged_regions)
    num_filtered_regions = len(regions)

    # Ground-truth/multi-burst metrics are computed regardless of whether
    # filter_by_min_length succeeded -- both compute_*_sensing_metrics
    # functions handle an empty `regions` list correctly (every truth burst
    # comes back "missed", not an error), so this is real, honest
    # information about the sensing outcome even on a failure, not a
    # fabricated success.
    ground_truth = None
    if radioml_meta is not None:
        ground_truth = compute_sensing_ground_truth_metrics(
            radioml_meta["true_start"], radioml_meta["true_end"], regions,
        )

    multi_burst_result = None
    if multi_burst_truths is not None:
        multi_burst_result = compute_multi_burst_sensing_metrics(multi_burst_truths, regions, cfg.n_samples)

    x_clean = None
    alignment_meta = None
    segment_region_ids = None
    if sensing_failure_stage is None:
        try:
            # select_aligned_segments (src/sensing/segmentation.py, docs/
            # parameter_validation.md section 18/20) replaces the old direct
            # segment_regions() call -- "naive" produces byte-identical
            # segment data to every pre-round-9 round via segment_regions()
            # internally; "max-energy" instead picks, per region, the single
            # highest-mean-power seg_len window (never using true burst
            # position). effective_alignment_policy is the source-aware-
            # resolved policy (section 20), not the raw possibly-None
            # cfg.alignment_policy. alignment_meta's region_idx field
            # replaces the old hand-rolled segment_region_ids loop, which
            # assumed segment_regions()'s naive-only (region, n_windows)
            # counting and would have silently mis-attributed segments under
            # max-energy (always 1 segment/region, not n_windows).
            segments, alignment_meta = select_aligned_segments(
                iq, regions, seg_len=cfg.window_size, policy=effective_alignment_policy, hop=cfg.segment_hop,
            )
            if multi_burst_truths is not None:
                segment_region_ids = [m["region_idx"] for m in alignment_meta]
            # AWN-input-boundary preprocessing (src/sensing/normalize.py:
            # apply_awn_preprocess, docs/parameter_validation.md section 19)
            # -- the ONLY place segment amplitude is rescaled before AWN;
            # alignment (above) and detection (energy_detect, earlier) never
            # see or depend on this. power_before/after are per-segment
            # mean(|x|^2) captured on either side, purely for diagnostics.
            awn_input_power_before = np.mean(np.abs(segments) ** 2, axis=1)  # [N]
            segments = apply_awn_preprocess(segments, policy=effective_awn_preprocess)
            awn_input_power_after = np.mean(np.abs(segments) ** 2, axis=1)  # [N]
            x_clean = to_awn_input(segments, seg_len=cfg.window_size)
        except RuntimeError as exc:
            # Same expected-failure semantics as before this round (retained
            # under the historical "segment_regions" stage name for
            # continuity with docs/parameter_validation.md section 16's
            # documented failure_stage values and any existing consumer
            # checking that exact string) -- now covers select_aligned_segments
            # too, since it raises the identical RuntimeError in the same
            # zero-valid-window case.
            sensing_failure_stage = "segment_regions"
            sensing_failure_reason = str(exc)
            segment_region_ids = None  # no segments exist to attribute to a region
            alignment_meta = None
            awn_input_power_before = awn_input_power_after = None

    if segment_region_ids is not None and x_clean is not None:
        assert len(segment_region_ids) == x_clean.shape[0], (
            f"segment/region attribution count mismatch: {len(segment_region_ids)} vs {x_clean.shape[0]}"
        )

    # Aggregate sensing fields, normalized across synthetic/single-burst/
    # multi-burst sources -- computed even on a sensing failure (using
    # whatever `regions` resulted, possibly []), since detection outcome
    # is meaningful and known regardless of whether segmentation itself
    # later succeeded.
    sensing_agg = derive_batch_aggregate_sensing_fields(ground_truth, multi_burst_result, regions, cfg.n_samples)

    if sensing_failure_stage is not None:
        # SENSING FAILURE: an expected, structured outcome, NOT a program
        # error -- return normally (do not raise) so a batch loop calling
        # this function can record a row for this combo instead of
        # aborting or silently dropping it. No fake segments are created:
        # summary.csv (fundamentally one row per SEGMENT) is not written
        # at all when there are zero segments. bursts_summary.csv/
        # regions_summary.csv (which describe the SENSING outcome, not
        # per-segment AMC results) ARE written whenever multi_burst_result
        # was computable, since a region can legitimately be detected and
        # then still fail to yield any full-length segment.
        print(f"\n[sensing] FAILED at stage={sensing_failure_stage}: {sensing_failure_reason}")
        bursts_summary_csv_path = None
        regions_summary_csv_path = None
        if multi_burst_result is not None:
            bursts_summary_csv_path = output_dir / "bursts_summary.csv"
            burst_rows = [
                {k: (str(v) if isinstance(v, list) else v) for k, v in pb.items()}
                for pb in multi_burst_result["per_burst"]
            ]
            write_summary_csv(bursts_summary_csv_path, burst_rows)

            regions_summary_csv_path = output_dir / "regions_summary.csv"
            region_rows = [
                {k: (str(v) if isinstance(v, list) else v) for k, v in pr.items()}
                for pr in multi_burst_result["per_region"]
            ]
            if region_rows:
                write_summary_csv(regions_summary_csv_path, region_rows)
            else:
                regions_summary_csv_path = None  # write_summary_csv refuses an empty row list

        return {
            "run_status": "sensing_failed",
            "sensing_success": False,
            "failure_stage": sensing_failure_stage,
            "failure_reason": sensing_failure_reason,
            "clean_amc_available": False,
            "attack_available": False,
            "defense_available": False,
            **sensing_agg,
            "num_raw_regions": num_raw_regions,
            "num_merged_regions": num_merged_regions,
            "num_filtered_regions": num_filtered_regions,
            "iq_source": cfg.iq_source,
            "alignment_policy": effective_alignment_policy,
            "segment_hop": cfg.segment_hop,
            "mean_segment_captured_signal_ratio": None,
            "awn_preprocess": effective_awn_preprocess,
            "mean_awn_input_power_before": None,
            "mean_awn_input_power_after": None,
            "mean_awn_input_scale_factor": None,
            "awn_input_min": None,
            "awn_input_max": None,
            "awn_input_has_nan": None,
            "awn_input_has_inf": None,
            "n_segments": 0,
            "seed": cfg.seed,
            "sensing_window_size": effective_sensing_window_size,
            "segment_length": cfg.window_size,
            "regions": regions,
            "output_dir": str(output_dir),
            "summary_csv_path": None,
            "bursts_summary_csv_path": str(bursts_summary_csv_path) if bursts_summary_csv_path else None,
            "regions_summary_csv_path": str(regions_summary_csv_path) if regions_summary_csv_path else None,
            "plot_path": None,
            "gen_meta": gen_meta,
            "long_iq_sha256": long_iq_sha256,
            "ground_truth": ground_truth,
            "multi_burst_result": multi_burst_result,
        }

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

    region_lookup = None
    burst_truth_lookup = None
    if multi_burst_result is not None:
        region_lookup = {pr["region_id"]: pr for pr in multi_burst_result["per_region"]}
        burst_truth_lookup = {b["burst_id"]: b for b in multi_burst_truths}

    def _segment_ground_truth_fields(seg_start: int, seg_end: int, i: int) -> dict:
        """
        Segment-level (NOT region-level) capture metrics -- distinct from the
        existing region-level `captured_signal_ratio` column above, which can
        read 1.0 even when the specific 128-sample AWN input window only
        partially overlaps the true burst (docs/parameter_validation.md
        section 18). true_start/true_end resolution: single-burst radioml
        mode uses `ground_truth` directly (one truth burst for the whole
        run); multi-burst mode looks up this segment's region's SINGLE
        matched burst (via region_lookup/burst_truth_lookup) -- ambiguous
        (0 or 2+ matched bursts) or synthetic-source (ground_truth and
        multi_burst_result both None) cases return all-None, not a guess.
        """
        true_start = true_end = None
        if ground_truth is not None:
            true_start, true_end = ground_truth["true_start"], ground_truth["true_end"]
        elif region_lookup is not None and segment_region_ids is not None:
            matched = region_lookup[segment_region_ids[i]]["matched_burst_ids"]
            if len(matched) == 1:
                b = burst_truth_lookup[matched[0]]
                true_start, true_end = b["true_start"], b["true_end"]

        if true_start is None:
            return {
                "segment_start_offset_from_true": None,
                "segment_intersection_length": None,
                "segment_captured_signal_ratio": None,
                "segment_noise_before_count": None,
                "segment_noise_after_count": None,
            }

        true_len = true_end - true_start
        inter_start = max(seg_start, true_start)
        inter_end = min(seg_end, true_end)
        intersection_length = max(0, inter_end - inter_start)
        noise_before = max(0, min(seg_end, true_start) - seg_start)
        noise_after = max(0, seg_end - max(seg_start, true_end))
        return {
            "segment_start_offset_from_true": seg_start - true_start,
            "segment_intersection_length": intersection_length,
            "segment_captured_signal_ratio": (intersection_length / true_len) if true_len > 0 else None,
            "segment_noise_before_count": noise_before,
            "segment_noise_after_count": noise_after,
        }

    rows = []
    for i in range(x_clean.shape[0]):
        rows.append({
            "segment_id": i,
            "seed": cfg.seed,
            # iq_source is the raw cfg/CLI value ("synthetic"/"radioml") that
            # drove effective_alignment_policy/effective_awn_preprocess's
            # source-aware resolution (docs/parameter_validation.md section
            # 20) -- distinct from source_type below, which further splits
            # "radioml" into "radioml"/"radioml_multi_burst".
            "iq_source": cfg.iq_source,
            # source_type distinguishes real ground truth (radioml) from the
            # synthetic generator's own inputs below. snr_db/mod are the
            # SYNTHETIC generator's inputs -- unused (but still populated
            # with whatever value/default was passed) when source_type ==
            # 'radioml'; dataset_mod/dataset_snr below are the REAL,
            # authoritative labels in that mode.
            "source_type": gen_meta["source_type"],
            "snr_db": cfg.snr,
            "mod": cfg.mod,
            "dataset_path": radioml_meta["dataset_path"] if radioml_meta else None,
            "dataset_mod": radioml_meta["dataset_mod"] if radioml_meta else None,
            "dataset_snr": radioml_meta["dataset_snr"] if radioml_meta else None,
            "sample_index": radioml_meta["sample_index"] if radioml_meta else None,
            "original_sample_sha256": radioml_meta["original_sample_sha256"] if radioml_meta else None,
            "long_iq_sha256": long_iq_sha256,
            "embed_snr_margin": radioml_meta["embed_snr_margin"] if radioml_meta else None,
            "true_burst_start": ground_truth["true_start"] if ground_truth else None,
            "true_burst_end": ground_truth["true_end"] if ground_truth else None,
            "detected_region_count": ground_truth["detected_region_count"] if ground_truth else None,
            "best_detected_start": ground_truth["best_detected_start"] if ground_truth else None,
            "best_detected_end": ground_truth["best_detected_end"] if ground_truth else None,
            "start_boundary_error": ground_truth["start_boundary_error"] if ground_truth else None,
            "end_boundary_error": ground_truth["end_boundary_error"] if ground_truth else None,
            "intersection_length": ground_truth["intersection_length"] if ground_truth else None,
            "detection_success": ground_truth["detection_success"] if ground_truth else None,
            "captured_signal_ratio": ground_truth["captured_signal_ratio"] if ground_truth else None,
            "extra_captured_noise_ratio": ground_truth["extra_captured_noise_ratio"] if ground_truth else None,
            "missed_sample_count": ground_truth["missed_sample_count"] if ground_truth else None,
            "false_occupied_sample_count": ground_truth["false_occupied_sample_count"] if ground_truth else None,
            # Segment-alignment fields (docs/parameter_validation.md section
            # 18) -- always populated (independent of ground truth), one per
            # actually-selected AWN input segment.
            "alignment_policy": alignment_meta[i]["alignment_policy"],
            "segment_hop": alignment_meta[i]["segment_hop"],
            "candidate_count": alignment_meta[i]["candidate_count"],
            "selected_segment_start": alignment_meta[i]["selected_segment_start"],
            "selected_segment_end": alignment_meta[i]["selected_segment_end"],
            "selected_window_power": alignment_meta[i]["selected_window_power"],
            "detected_region_start": alignment_meta[i]["detected_region_start"],
            "detected_region_end": alignment_meta[i]["detected_region_end"],
            # AWN-input-boundary preprocessing fields (docs/parameter_validation.md
            # section 19) -- "before"/"after" bracket exactly the
            # apply_awn_preprocess() call above; awn_input_min/max/has_nan/
            # has_inf describe the ACTUAL array handed to AWN.infer() (post-
            # preprocessing, real+imag combined, matching to_awn_input's [2,T] layout).
            "awn_preprocess": effective_awn_preprocess,
            "awn_input_power_before": float(awn_input_power_before[i]),
            "awn_input_power_after": float(awn_input_power_after[i]),
            "awn_input_scale_factor": (
                float(np.sqrt(awn_input_power_after[i] / awn_input_power_before[i]))
                if awn_input_power_before[i] > 0 else None
            ),
            "awn_input_min": float(np.min(x_clean[i])),
            "awn_input_max": float(np.max(x_clean[i])),
            "awn_input_has_nan": bool(np.isnan(x_clean[i]).any()),
            "awn_input_has_inf": bool(np.isinf(x_clean[i]).any()),
            # Segment-level (NOT region-level) capture metrics -- see
            # _segment_ground_truth_fields()'s docstring above; distinct from
            # (and must not be confused with) the region-level
            # "captured_signal_ratio" column above.
            **_segment_ground_truth_fields(
                alignment_meta[i]["selected_segment_start"], alignment_meta[i]["selected_segment_end"], i
            ),
            # Multi-burst mode only (num_bursts>1) -- None for single-burst/
            # synthetic rows. source_region_id is this segment's detected
            # region (see bursts_summary.csv/regions_summary.csv, written
            # alongside this file, for the full per-burst/per-region
            # breakdown and the run's aggregate Pd/Pfa metrics).
            "num_bursts": cfg.num_bursts if multi_burst_result is not None else None,
            "source_region_id": segment_region_ids[i] if segment_region_ids is not None else None,
            "region_matched_burst_ids": (
                str(region_lookup[segment_region_ids[i]]["matched_burst_ids"]) if region_lookup else None
            ),
            "region_false_occupied_sample_count": (
                region_lookup[segment_region_ids[i]]["false_occupied_sample_count"] if region_lookup else None
            ),
            "region_extra_captured_noise_ratio": (
                region_lookup[segment_region_ids[i]]["extra_captured_noise_ratio"] if region_lookup else None
            ),
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

    bursts_summary_csv_path = None
    regions_summary_csv_path = None
    if multi_burst_result is not None:
        # One row per TRUE burst (always -- including missed bursts, which
        # have zero representation in the per-segment summary.csv above
        # since a missed burst has no detected region and therefore no
        # segments) and one row per DETECTED region (always -- including
        # false-alarm regions with zero matched bursts). The aggregate
        # Pd/Pfa numbers are in result["multi_burst_aggregate"] (and printed
        # below); not worth a third one-row CSV file for a handful of
        # scalars already visible in the console summary and the returned
        # result dict.
        bursts_summary_csv_path = output_dir / "bursts_summary.csv"
        burst_rows = [
            {k: (str(v) if isinstance(v, list) else v) for k, v in pb.items()}
            for pb in multi_burst_result["per_burst"]
        ]
        write_summary_csv(bursts_summary_csv_path, burst_rows)

        regions_summary_csv_path = output_dir / "regions_summary.csv"
        region_rows = [
            {k: (str(v) if isinstance(v, list) else v) for k, v in pr.items()}
            for pr in multi_burst_result["per_region"]
        ]
        write_summary_csv(regions_summary_csv_path, region_rows)

    plot_path = output_dir / "sensing_plot.png"
    plot_created = plot_sensing_result(iq, regions, plot_path)

    _seg_ratios = [r["segment_captured_signal_ratio"] for r in rows if r["segment_captured_signal_ratio"] is not None]
    mean_segment_captured_signal_ratio = (sum(_seg_ratios) / len(_seg_ratios)) if _seg_ratios else None

    # AWN-input-boundary preprocessing aggregates (docs/parameter_validation.md
    # section 19) -- mean over segments for the two power fields and the
    # derived scale factor; global min/max and any() for has_nan/has_inf,
    # mirroring how a batch-level row summarizes a run's many segments.
    _power_before = [r["awn_input_power_before"] for r in rows]
    _power_after = [r["awn_input_power_after"] for r in rows]
    _scale_factors = [r["awn_input_scale_factor"] for r in rows if r["awn_input_scale_factor"] is not None]

    result = {
        "run_status": "ok",
        "sensing_success": True,
        "failure_stage": None,
        "failure_reason": None,
        # If we've reached this point, AWN/attack/Top-K all ran to
        # completion without raising (a shape-mismatch would have raised
        # already, above) -- "available" reflects the stage completed, real
        # or dummy backend notwithstanding (backend type is separately
        # recorded in awn_backend/attack_backend/topk_backend).
        "clean_amc_available": True,
        "attack_available": True,
        "defense_available": True,
        **sensing_agg,
        "num_raw_regions": num_raw_regions,
        "num_merged_regions": num_merged_regions,
        "num_filtered_regions": num_filtered_regions,
        "iq_source": cfg.iq_source,
        "alignment_policy": effective_alignment_policy,
        "segment_hop": cfg.segment_hop,
        # Mean of per-segment (NOT per-region) captured_signal_ratio, over
        # segments with a resolvable true burst (see
        # _segment_ground_truth_fields's docstring) -- None if no segment
        # has one (synthetic source, or multi-burst segments whose region
        # ambiguously matched 0/2+ bursts).
        "mean_segment_captured_signal_ratio": mean_segment_captured_signal_ratio,
        "awn_preprocess": effective_awn_preprocess,
        "mean_awn_input_power_before": float(np.mean(_power_before)),
        "mean_awn_input_power_after": float(np.mean(_power_after)),
        "mean_awn_input_scale_factor": (sum(_scale_factors) / len(_scale_factors)) if _scale_factors else None,
        "awn_input_min": float(min(r["awn_input_min"] for r in rows)),
        "awn_input_max": float(max(r["awn_input_max"] for r in rows)),
        "awn_input_has_nan": any(r["awn_input_has_nan"] for r in rows),
        "awn_input_has_inf": any(r["awn_input_has_inf"] for r in rows),
        "n_segments": x_clean.shape[0],
        "seed": cfg.seed,
        "sensing_window_size": effective_sensing_window_size,
        "segment_length": cfg.window_size,
        "regions": regions,
        "output_dir": str(output_dir),
        "summary_csv_path": str(summary_csv_path),
        "bursts_summary_csv_path": str(bursts_summary_csv_path) if bursts_summary_csv_path else None,
        "regions_summary_csv_path": str(regions_summary_csv_path) if regions_summary_csv_path else None,
        "plot_path": str(plot_path) if plot_created else None,
        "gen_meta": gen_meta,
        "long_iq_sha256": long_iq_sha256,
        "ground_truth": ground_truth,
        "multi_burst_result": multi_burst_result,
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
    print(f"IQ source:          {gen_meta['source_type']}")
    if radioml_meta is not None:
        print(f"RadioML sample:     dataset_mod={radioml_meta['dataset_mod']} dataset_snr={radioml_meta['dataset_snr']} "
              f"sample_index={radioml_meta['sample_index']}")
        print(f"True burst:         [{radioml_meta['true_start']}:{radioml_meta['true_end']}]")
        if ground_truth is not None:
            print(f"Ground truth:       detection_success={ground_truth['detection_success']} "
                  f"captured_signal_ratio={ground_truth['captured_signal_ratio']:.4f} "
                  f"start_err={ground_truth['start_boundary_error']} end_err={ground_truth['end_boundary_error']}")
    if multi_burst_result is not None:
        agg = multi_burst_result["aggregate"]
        print(f"Multi-burst truth:  {[(b['true_start'], b['true_end']) for b in multi_burst_truths]}")
        print(f"Pd={agg['detection_probability']} false_alarm_region_rate={agg['false_alarm_region_rate']} "
              f"sample_FPR={agg['sample_level_false_positive_rate']} sample_FNR={agg['sample_level_false_negative_rate']}")
        print(f"num_truth_bursts={agg['num_truth_bursts']} num_detected_regions={agg['num_detected_regions']} "
              f"num_matched_bursts={agg['num_matched_bursts']} num_missed_bursts={agg['num_missed_bursts']} "
              f"num_false_alarm_regions={agg['num_false_alarm_regions']}")
        print(f"bursts_summary.csv: {bursts_summary_csv_path}")
        print(f"regions_summary.csv: {regions_summary_csv_path}")
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
